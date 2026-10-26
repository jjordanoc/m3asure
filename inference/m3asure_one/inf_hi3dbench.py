"""
Run M3aSURE latent alignment inference on Hi3DBench assets

Usage:
    uv run data/hi3dbench/...
    uv run inference/inf_hi3dbench.py --subset 200 --seed 42
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import gc
import math
import torch
from tqdm import tqdm

from inference.inference_helpers import (
    build_remaining_loader,
    load_checkpoint,
    notify_oom,
    save_checkpoint,
    save_hi3dbench_predictions,
    device_dtype,
    autocast_context,
)

from data import (
    build_hi3dbench_asset_dataset, HI3DBENCH_VIEW_INDICES
)
from models import (
    M3aSURELatentAlignment,
    load_weights, load_meta,
)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def main() -> None:
    DEFAULT_DATA_ROOT_3DGENBENCH = "/workspace/m3asure/data/assets/3d_gen_bench_processed_12_views"
    DEFAULT_DATA_ROOT_HI3DBENCH = "/workspace/m3asure/data/assets/hi_3d_bench_processed/ring"
    DEFAULT_SOURCE_IMG_ROOT = os.path.join(DEFAULT_DATA_ROOT_3DGENBENCH, "images_prompts_510")
    CHECKPOINT_INTERVAL = 100

    parser = argparse.ArgumentParser()
    ckpt_name = "m3asure_one_train_linear_e100_ablation2_margin10_v12_mean_pool_reg"
    parser.add_argument("--checkpoint", type=str,
                        default=f"/workspace/m3asure/results/probe_checkpoints/{ckpt_name}/latent_alignment_best_acc.pt")
    parser.add_argument("--subset", type=int, default=200,
                        help="Number of prompts to randomly sample (asset mode only)")  # 200 default
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --subset sampling")
    parser.add_argument("--data-root", type=str,
                        default=DEFAULT_DATA_ROOT_HI3DBENCH, )
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=None, help="Cap number of trials for debugging")
    args = parser.parse_args()

    data_root = args.data_root
    img_prompts_root = DEFAULT_SOURCE_IMG_ROOT
    results_base = f"{PROJECT_ROOT}/results"
    results_dir = os.path.join(results_base, "hi_3d_bench", ckpt_name)

    if args.run_id:
        results_dir += f"_{args.run_id}"
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = device_dtype(device)
    print(f"device: {device}")
    print(f"autocast dtype: {dtype}")
    print(f"dataset: {args.dataset}  mode: {args.mode}")

    dataset = build_hi3dbench_asset_dataset(
        data_root=data_root,
        source_root=img_prompts_root,
        view_indices=HI3DBENCH_VIEW_INDICES,
        subset=args.subset,
        seed=args.seed,
        preprocess=load_and_preprocess_images,
    )

    print(f"Dataset ready: {len(dataset)} evaluations.")

    reward_checkpoint = load_weights(args.checkpoint, device)
    probe_kwargs = load_meta(reward_checkpoint)

    # TODO: add cache

    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    metric = M3aSURELatentAlignment(vggt_model=vggt_model, dropout=0.0, expected_views=12, **probe_kwargs).to(device)
    metric.load_reward_state_dict(reward_checkpoint["reward"])
    metric.eval()
    print(
        f"Loaded reward checkpoint from epoch={reward_checkpoint['epoch']} "
        f"valid_loss={reward_checkpoint['valid_loss']:.6f} "
        f"sigma={float(metric.sigma.detach().cpu()):.6f} "
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
            # Shared scoring path for both battle and asset modes
            model_name = item["model_name"]
            images = item["processed_images"].to(device, non_blocking=True)
            with torch.no_grad(), autocast_context(device, dtype):
                result = metric(images)

            score = float(result["score"][0].detach().cpu())
            mean_score = float(result["per_view_scores"][0].mean().detach().cpu())
            scaled_mean_score = float(result["score"][0].detach().cpu())
            if not math.isfinite(score) or not math.isfinite(mean_score) or not math.isfinite(scaled_mean_score):
                raise RuntimeError(f"Non-finite latent alignment score for {battle_id}/{model_name}.")

            latent_scores[battle_id][model_name] = {
                "score": score,
                "mean_score": mean_score,
                "scaled_mean_score": scaled_mean_score,
                "sigma": float(metric.sigma.detach().cpu()),
                "per_view_scores": result["per_view_scores"][0].detach().cpu().tolist(),
                "per_view_scaled_scores": result["per_view_scaled_scores"][0].detach().cpu().tolist(),
            }

            del images, result
            done.add((battle_id, model_name))
            pbar.update(1)
            pbar.set_postfix_str(f"{battle_id}/{model_name}: {score:.4f}")

        except torch.cuda.OutOfMemoryError:
            save_checkpoint(latent_scores, inf_ckpt_path)
            model_n = "adv_toys4k" if args.dataset == "adv_toys4k" else item.get("model_name", "unknown")
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

    save_hi3dbench_predictions(latent_scores, results_dir)


if __name__ == "__main__":
    main()
