"""
Run M3aSURE latent alignment inference on 3DGen-Bench

Usage:
    uve run inference/inf_3dgenbench.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import gc
import json
import math
import torch
from tqdm import tqdm

from inference.inference_helpers import (
    build_remaining_loader,
    load_checkpoint,
    notify_oom,
    save_checkpoint,
    save_3dgenbench_predictions,
    device_dtype,
    autocast_context,
)

from data import (
    CachedInference3dgenbenchDataset,
    Inference3dgenbenchDataset
)
from models import (
    M3aSURELatentAlignment,
    M3aSURECachedLatentAlignment,
    load_weights, load_meta,
)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def _extract_score(scores_data: dict, models: list[str]) -> tuple[float, float]:
    return (
        float(scores_data[models[0]]["score"]),
        float(scores_data[models[1]]["score"]),
    )


def _extract_mean_score(scores_data: dict, models: list[str]) -> tuple[float, float]:
    return (
        float(scores_data[models[0]]["mean_score"]),
        float(scores_data[models[1]]["mean_score"]),
    )


def main() -> None:
    DEFAULT_DATA_ROOT_3DGENBENCH = "/workspace/m3asure/data/assets/3dgenbench_processed_12_views"
    DEFAULT_SOURCE_IMG_ROOT = os.path.join(DEFAULT_DATA_ROOT_3DGENBENCH, "images_prompts_510")
    DEFAULT_CKPT_NAME = "m3asure_one_train_final2"
    CHECKPOINT_INTERVAL = 100

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT_3DGENBENCH, help="3dgenbench root directory")
    parser.add_argument("--weights", type=str,
                        default=f"/workspace/m3asure/results/probe_checkpoints/{DEFAULT_CKPT_NAME}/latent_alignment_best_acc.pt")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=None, help="Cap number of trials for debugging")
    parser.add_argument("--probe-cache", type=str, default=None,
                        help="Root directory of pre-extracted VGGT features. When set, skips VGGT loading and uses CachedLatentAlignmentProbe.")
    args = parser.parse_args()

    img_prompts_root = DEFAULT_SOURCE_IMG_ROOT
    ckpt_name = DEFAULT_CKPT_NAME
    data_root = args.data_root
    results_base = "/workspace/m3asure/results"
    results_dir = os.path.join(results_base, "3d_gen_bench", ckpt_name)

    if args.run_id:
        results_dir += f"_{args.run_id}"
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = device_dtype(device)
    print(f"device: {device}")
    print(f"autocast dtype: {dtype}")

    battles_data = None
    use_cache = args.probe_cache is not None

    if use_cache:
        # use pre-extracted features
        battles_json_path = os.path.join(data_root, f"{args.split}.json")
        with open(battles_json_path, "r") as f:
            battles_data = json.load(f)
        dataset = CachedInference3dgenbenchDataset(
            battles_json=battles_data,
            cache_root=args.probe_cache,
        )
    else:
        # end-to-end inference on the fly (production)
        battles_json_path = os.path.join(data_root, f"{args.split}.json")
        with open(battles_json_path, "r") as f:
            battles_data = json.load(f)
        dataset = Inference3dgenbenchDataset(
            battles_json=battles_data,
            data_root=data_root,
            source_root=img_prompts_root,
            preprocess=load_and_preprocess_images,
        )

    print(f"Dataset ready: {len(dataset)} evaluations.")

    reward_checkpoint = load_weights(args.weights, device)
    probe_kwargs = load_meta(reward_checkpoint)

    if use_cache:
        metric = M3aSURECachedLatentAlignment(
            dropout=0.0, expected_views=12, **probe_kwargs,
        ).to(device)
        metric.load_reward_state_dict(reward_checkpoint["reward"])
        metric.eval()
        print(
            f"Loaded cached probe from epoch={reward_checkpoint['epoch']} "
            f"valid_loss={reward_checkpoint['valid_loss']:.6f} "
            f"probe_arch={probe_kwargs}  (VGGT skipped)"
        )
    else:
        vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        metric = M3aSURELatentAlignment(vggt_model=vggt_model, dropout=0.0, expected_views=12, **probe_kwargs).to(
            device)
        metric.load_reward_state_dict(reward_checkpoint["reward"])
        metric.eval()
        print(
            f"Loaded reward checkpoint from epoch={reward_checkpoint['epoch']} "
            f"valid_loss={reward_checkpoint['valid_loss']:.6f} "
            f"sigma={float(metric.sigma):.6f} "
            f"probe_arch={probe_kwargs}"
        )

    inf_ckpt_path = os.path.join(results_dir, "latent_alignment_checkpoint.json")
    latent_scores, done = load_checkpoint(inf_ckpt_path)
    loader, remaining_indices = build_remaining_loader(
        dataset,
        done,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"Skipping {len(dataset) - len(remaining_indices)} already-done evaluations, "
        f"{len(remaining_indices)} remaining."
    )
    completed_keys = set(latent_scores.keys())
    evals_since_save = 0
    pbar = tqdm(total=len(remaining_indices))

    for item in loader:
        battle_id = item["battle_id"][0] if isinstance(item["battle_id"], list) else item["battle_id"]

        if battle_id not in latent_scores:
            latent_scores[battle_id] = {}

        try:
            model_name = item["model_name"]
            if use_cache:
                image_vectors = item["image_vectors"].to(device, non_blocking=True)
                if image_vectors.ndim == 3:
                    image_vectors = image_vectors.unsqueeze(0)
                images = image_vectors
            else:
                images = item["processed_images"].to(device, non_blocking=True)

            with torch.no_grad(), autocast_context(device, dtype):
                result = metric(images)
            del images
            score = float(result["score"][0].detach().cpu())
            mean_score = float(result["per_view_scores"][0].mean().detach().cpu())
            scaled_mean_score = float(result["score"][0].detach().cpu())
            if not math.isfinite(score) or not math.isfinite(mean_score) or not math.isfinite(scaled_mean_score):
                raise RuntimeError(f"Non-finite latent alignment score for {battle_id}/{model_name}.")

            latent_scores[battle_id][model_name] = {
                "score": score,
                "mean_score": mean_score,
                "scaled_mean_score": scaled_mean_score,
                "sigma": float(metric.sigma),
                "per_view_scores": result["per_view_scores"][0].detach().cpu().tolist(),
                "per_view_scaled_scores": result["per_view_scaled_scores"][0].detach().cpu().tolist(),
            }

            del result
            done.add((battle_id, model_name))
            pbar.update(1)
            pbar.set_postfix_str(f"{battle_id}/{model_name}: {score:.4f}")

        except torch.cuda.OutOfMemoryError:
            save_checkpoint(latent_scores, inf_ckpt_path)
            model_n = item.get("model_name", "unknown")
            notify_oom("inference/latent_alignment_inference.py", battle_id, model_n, inf_ckpt_path)
            raise
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if battle_id not in completed_keys:
            completed_keys.add(battle_id)
            evals_since_save += 1
            if evals_since_save >= CHECKPOINT_INTERVAL:
                save_checkpoint(latent_scores, inf_ckpt_path)
                print(f"\nCheckpoint saved ({len(completed_keys)} evaluations done)")
                evals_since_save = 0

    pbar.close()
    save_checkpoint(latent_scores, inf_ckpt_path)
    print(f"Done. Computed scores for {len(latent_scores)} evaluations.")

    save_3dgenbench_predictions(
        latent_scores,
        battles_data,
        results_dir=results_dir,
        metric_name="m3asure_latent_alignment",
        extract_fns={
            "": _extract_score,
            "_mean": _extract_mean_score,
        },
    )


if __name__ == "__main__":
    main()
