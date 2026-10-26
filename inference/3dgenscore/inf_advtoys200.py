"""
Run 3DGenScore on AdvToys200 trials.

Usage:
    uve run data/3d_gen_bench/download_3dgenscore_weights.py
    uve run inf_advtoys200.py --split test
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from contextlib import nullcontext
import argparse
import gc
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from inference.inference_helpers import (
    save_checkpoint,
    collate_single,
)

from models import load_3dgenscore_model, EVAL_DIMS, calc_probs

from data import InferenceAdvtoys200Dataset

from helpers import (
    load_battle_checkpoint,
    format_result,
)


def main() -> None:
    DEFAULT_THREEDGEN_BENCH_ROOT = PROJECT_ROOT / "3DGen-Bench"
    sys.path.insert(0, str(DEFAULT_THREEDGEN_BENCH_ROOT))
    PROCESSOR_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    DEFAULT_CON_IMAGES_DIR = (
        "/workspace/m3asure/data/assets/3d_gen_bench_3dgenscore/con_images"
    )
    DEFAULT_CHECKPOINT_DIR = (
        "/workspace/m3asure/3DGen-Bench/checkpoint/3dgen-score-mvclip-v1"
    )
    DEFAULT_ADVTOYS200_ROOT = "/workspace/m3asure/data/assets/adv_toys4k_processed"
    CHECKPOINT_INTERVAL = 100
    NUM_WORKERS = 0

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_ADVTOYS200_ROOT,
    )
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--weights", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--con-images-dir", type=str, default=DEFAULT_CON_IMAGES_DIR)
    parser.add_argument("--source-root", type=str, default="")
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Cap number of trials for debugging",
    )
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    print(f"device: {device}")

    processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)
    model = load_3dgenscore_model(args.weights, device)

    results_dir = (
        f"/workspace/m3asure/results/adv_toys4k/{args.run_id}"
        if args.run_id
        else "/workspace/m3asure/results/adv_toys4k/default"
    )
    checkpoint_path = os.path.join(results_dir, "3dgenscore_checkpoint.json")
    dataset = InferenceAdvtoys200Dataset(args.data_root)
    scores, done = load_battle_checkpoint(checkpoint_path)

    remaining_indices = [
        idx
        for idx, sample in enumerate(dataset.samples)
        if sample["battle_id"] not in done
    ]
    if args.max_trials is not None:
        remaining_indices = remaining_indices[: args.max_trials]

    loader = DataLoader(
        torch.utils.data.Subset(dataset, remaining_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_single,
        pin_memory=False,
    )
    print(
        f"Dataset ready: {len(dataset)} battles. "
        f"Skipping {len(dataset) - len(remaining_indices)} done, "
        f"{len(remaining_indices)} remaining."
    )

    os.makedirs(results_dir, exist_ok=True)
    battles_since_save = 0
    pbar = tqdm(total=len(remaining_indices))

    for item in loader:
        battle_id = item["battle_id"]
        try:
            autocast_context = (
                torch.amp.autocast("cuda", dtype=dtype)
                if device == "cuda"
                else nullcontext()
            )
            with autocast_context:
                probs = calc_probs(
                    model=model,
                    processor=processor,
                    device=device,
                    reference=item["reference"],
                    reference_type=item["reference_type"],
                    normal_image_0=item["normal_image_0"],
                    normal_image_1=item["normal_image_1"],
                    rgb_image_0=item["rgb_image_0"],
                    rgb_image_1=item["rgb_image_1"],
                )
            scores[battle_id] = format_result(item, probs)
        except Exception:
            save_checkpoint(scores, checkpoint_path)
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        done.add(battle_id)
        battles_since_save += 1
        pbar.update(1)

        w = scores[battle_id]["3dgenscore_without_oddity"]
        o = scores[battle_id]["3dgenscore_with_oddity"]
        pbar.set_postfix_str(
            f"{battle_id}: geo={w.get('geometry_score', 0):.3f}/{o.get('geometry_score', 0):.3f}"
        )

        if battles_since_save >= CHECKPOINT_INTERVAL:
            save_checkpoint(scores, checkpoint_path)
            print(f"\nCheckpoint saved ({len(done)} battles done)")
            battles_since_save = 0

    pbar.close()
    save_checkpoint(scores, checkpoint_path)
    print(f"Done. Computed scores for {len(scores)} battles.")

    pred_path = os.path.join(results_dir, "3dgenscore_predictions.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    summary = {}
    for dim in EVAL_DIMS:
        raw_deltas = [
            r["3dgenscore_without_oddity"][f"{dim}_score"]
            - r["3dgenscore_with_oddity"][f"{dim}_score"]
            for r in scores.values()
        ]

        arr_deltas = np.array(raw_deltas)
        summary[f"{dim}_delta"] = {
            "mean": float(np.mean(arr_deltas)),
            "std": float(np.std(arr_deltas)),
            "median": float(np.median(arr_deltas)),
            "min": float(np.min(arr_deltas)),
            "max": float(np.max(arr_deltas)),
            "fraction_positive": float(np.mean(arr_deltas > 0)),
            "n": len(arr_deltas),
        }

    summary_path = os.path.join(results_dir, "3dgenscore_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("AdvToys4K 3DGenScore Results Summary")
    print("=" * 60)
    for key, stats in summary.items():
        print(f"\n{key}:")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
