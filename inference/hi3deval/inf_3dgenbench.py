"""
Run Hi3DEval on 3DGen-Bench battles.

Evaluates two metrics:
  - geometry_plausibility: uses 16 normal render frames
  - prompt_alignment: uses 8 RGB render frames (cosine similarity mode)

Each metric requires its own QFormer checkpoint, we run two passes.

Usage:
    uve run data/3dgenbench/preprocess_hi3deval.py --split test
    uve run inf_3dgenbench.py --split test
"""

from pathlib import Path
import sys

M3ASURE_ROOT = Path("/workspace/m3asure")
sys.path.insert(0, str(M3ASURE_ROOT))
HI3DEVAL_ROOT = M3ASURE_ROOT / "Hi3DEval" / "obj_scoring"
# ADD HI3DEVAL TO PATH TO IMPORT ITS MODULES!
sys.path.insert(0, str(HI3DEVAL_ROOT))

import argparse
import gc
import json
import math
import os
from contextlib import nullcontext
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from inference.inference_helpers import save_checkpoint, save_3dgenbench_predictions, collate_single
from helpers import load_battle_checkpoint, make_dim_extractor, extract_aggregate
from models import EVAL_DIMS, load_hi3deval_model, get_hi3deval_transform, score_single_model, METRIC_CONFIGS
from data import Inference3dgenbenchDataset


def main() -> None:
    DEFAULT_DATA_ROOT = "/workspace/m3asure/data/assets/3d_gen_bench_processed"
    DEFAULT_ORIGINAL_ROOT = "/workspace/m3asure/data/assets/3d_gen_bench_original"
    DEFAULT_HI3DEVAL_IMAGES_DIR = "/workspace/m3asure/data/assets/3d_gen_bench_hi3deval"
    DEFAULT_CKPT_ROOT = str(HI3DEVAL_ROOT)
    CHECKPOINT_INTERVAL = 100
    NUM_WORKERS = 0

    parser = argparse.ArgumentParser(description="Run Hi3DEval on 3DGen-Bench or AdvToys4K.")
    parser.add_argument("--split", choices=("test"), default="test")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--original-root", type=str, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--processed-hi3deval-dir", type=str, default=DEFAULT_HI3DEVAL_IMAGES_DIR)
    parser.add_argument("--ckpt-root", type=str, default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--max-trials", type=int, default=None, help="Cap number of trials for debugging")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"device: {device}, dtype: {dtype}")

    battles_json_path = os.path.join(args.data_root, f"{args.split}.json")
    with open(battles_json_path, "r", encoding="utf-8") as f:
        battles_data = json.load(f)
    results_dir = f"/workspace/m3asure/results/3d_gen_bench/hi3deval_{args.split}"
    if args.run_id:
        results_dir += f"_{args.run_id}"

    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(results_dir, "hi3deval_checkpoint.json")
    scores, done = load_battle_checkpoint(checkpoint_path)

    source_roots = (
        Path(args.data_root),
        Path(args.original_root) / "gallery" / "rgba",
        Path(args.original_root) / "gallery" / "images_prompts_510",
        Path(args.data_root) / "images_prompts_510",
    )

    transform = get_hi3deval_transform()

    for dim in EVAL_DIMS:
        metric_cfg = METRIC_CONFIGS[dim]
        print(f"\n{'=' * 60}")
        print(f"Scoring dimension: {dim} (mode={metric_cfg['mode']}, frames={metric_cfg['num_frames']})")
        print(f"{'=' * 60}")

        # skip if done dim
        battles_needing_dim = [
            bid for bid in battles_data
            if battles_data[bid]["task"] == "image2shape"
               and (bid not in scores or dim not in scores[bid].get("_dims_done", []))
        ]
        if not battles_needing_dim:
            print(f"All battles already scored for {dim}, skipping.")
            continue

        dataset = Inference3dgenbenchDataset(
            battles_data=battles_data,
            hi3deval_images_dir=args.processed_hi3deval_dir,
            source_roots=source_roots,
            mode=metric_cfg["mode"],
            num_frames=metric_cfg["num_frames"],
            transform=transform,
        )

        # skip
        remaining_indices = [
            idx for idx, sample in enumerate(dataset.samples)
            if sample["battle_id"] not in done
               or (sample["battle_id"] in scores
                   and dim not in scores[sample["battle_id"]].get("_dims_done", []))
        ]
        if args.max_trials is not None:
            remaining_indices = remaining_indices[:args.max_trials]
        if not remaining_indices:
            print(f"No remaining battles for {dim}, skipping.")
            continue

        model, config = load_hi3deval_model(
            ckpt_root=args.ckpt_root,
            qformer_ckpt=metric_cfg["qformer_ckpt"],
            num_frames=metric_cfg["num_frames"],
            is_prompt_alignment_cosine_mlp=metric_cfg["is_prompt_alignment_cosine_mlp"],
            device=device,
        )

        loader = DataLoader(
            torch.utils.data.Subset(dataset, remaining_indices),
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_single,
            pin_memory=False,
        )
        print(
            f"Dataset: {len(dataset)} trials total. "
            f"{len(remaining_indices)} needing {dim} scoring."
        )

        battles_since_save = 0
        pbar = tqdm(total=len(remaining_indices), desc=dim)

        for item in loader:
            battle_id = item["battle_id"]
            models = item["models"]

            if battle_id not in scores:
                scores[battle_id] = {}

            try:
                autocast_context = (
                    torch.amp.autocast("cuda", dtype=dtype)
                    if device == "cuda"
                    else nullcontext()
                )
                with autocast_context:
                    score_0 = score_single_model(
                        model=model,
                        frames=item["frames_0"],
                        reference=item["reference"],
                        is_text_prompt=False,
                        is_prompt_alignment=metric_cfg["is_prompt_alignment"],
                        is_prompt_alignment_cosine_mlp=metric_cfg["is_prompt_alignment_cosine_mlp"],
                        device=device,
                    )
                    score_1 = score_single_model(
                        model=model,
                        frames=item["frames_1"],
                        reference=item["reference"],
                        is_text_prompt=False,
                        is_prompt_alignment=metric_cfg["is_prompt_alignment"],
                        is_prompt_alignment_cosine_mlp=metric_cfg["is_prompt_alignment_cosine_mlp"],
                        device=device,
                    )

                if not math.isfinite(score_0) or not math.isfinite(score_1):
                    raise RuntimeError(
                        f"Non-finite Hi3DEval score for {battle_id}: {score_0}, {score_1}"
                    )

                for m in models:
                    if m not in scores[battle_id]:
                        scores[battle_id][m] = {
                            "score": 0.0,
                            "per_dim_scores": {},
                            "task": item["task"],
                            "offline_idx": item["offline_idx"],
                        }

                scores[battle_id][models[0]]["per_dim_scores"][dim] = score_0
                scores[battle_id][models[1]]["per_dim_scores"][dim] = score_1

                if "_dims_done" not in scores[battle_id]:
                    scores[battle_id]["_dims_done"] = []
                if dim not in scores[battle_id]["_dims_done"]:
                    scores[battle_id]["_dims_done"].append(dim)

            except Exception:
                save_checkpoint(scores, checkpoint_path)
                raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            battles_since_save += 1
            pbar.update(1)
            pbar.set_postfix_str(
                f"{battle_id}: {models[0]}={score_0:.4f} {models[1]}={score_1:.4f}"
            )

            if battles_since_save >= CHECKPOINT_INTERVAL:
                save_checkpoint(scores, checkpoint_path)
                print(f"\nCheckpoint saved ({dim})")
                battles_since_save = 0

        pbar.close()
        save_checkpoint(scores, checkpoint_path)
        print(f"Done scoring {dim}.")

        del model, config
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("Computing aggregate scores")
    print(f"{'=' * 60}")

    for battle_id in list(scores.keys()):
        battle_scores = scores[battle_id]
        dims_done = battle_scores.get("_dims_done", [])
        if not dims_done:
            continue

        for model_name in [m for m in battle_scores if m != "_dims_done"]:
            model_data = battle_scores[model_name]
            per_dim = model_data.get("per_dim_scores", {})
            available_scores = [per_dim[d] for d in dims_done if d in per_dim]
            if available_scores:
                model_data["score"] = float(np.mean(available_scores))

        # Remove internal tracking key before saving
        if "_dims_done" in battle_scores:
            del battle_scores["_dims_done"]

    save_checkpoint(scores, checkpoint_path)

    save_3dgenbench_predictions(
        scores,
        battles_data,
        results_dir=results_dir,
        metric_name="hi3deval",
        extract_fns={
            "": extract_aggregate,
            **{f"_{dim}": make_dim_extractor(dim) for dim in EVAL_DIMS},
        },
    )

    pred_path = os.path.join(results_dir, "hi3deval_scores.json")
    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    for bid in list(predictions.keys())[:5]:
        prediction = predictions[bid]
        print(
            f"  {bid}: {prediction['models']} -> "
            f"raw={prediction['raw_scores']} "
            f"pred={prediction['hi3deval_prediction']}"
        )


if __name__ == "__main__":
    main()
