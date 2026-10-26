"""
Run M3aSURE projector on 3DGen-Bench battles.

Usage:
   uv run inf_3dgenbench.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import gc
import json
import torch
from tqdm import tqdm
import argparse

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

from data import InferenceAdvtoys200Dataset
from models import M3aSURESemanticProjector
from inference.inference_helpers import (
    load_checkpoint,
    save_checkpoint,
    build_remaining_loader,
    save_predictions,
    notify_oom,
    notify,
)


def main():
    DEFAULT_DATA_ROOT = "/workspace/m3asure/data/assets/adv_toys4k_processed"
    RESULTS_DIR = f"/workspace/m3asure/results/advtoys200/projector"
    CHECKPOINT_INTERVAL = 200
    NUM_WORKERS = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    args = parser.parse_args()

    data_root = args.data_root
    if args.run_id:
        RESULTS_DIR += f"/projector_{args.run_id}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    vggt_model.training = False

    anyup_model = torch.hub.load('anyup', 'anyup_multi_backbone', use_natten=False, source="local")
    anyup_model.eval()
    for p in anyup_model.parameters():
        p.requires_grad_(False)
    print(f"AnyUp loaded  –  {sum(p.numel() for p in anyup_model.parameters()) / 1e6:.1f}M params")

    import projector

    metric = M3aSURESemanticProjector(
        vggt_model=vggt_model,
        anyup_model=anyup_model,
    ).to(device)
    print("M3aSURE metric ready.")

    dataset = InferenceAdvtoys200Dataset(
        root=data_root,
        preprocess_fn=load_and_preprocess_images,
    )
    print(f"Dataset ready: {len(dataset)} evaluations.")

    # inf ckpt
    os.makedirs(RESULTS_DIR, exist_ok=True)
    CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "projector_checkpoint.json")
    projector_scores, done = load_checkpoint(CHECKPOINT_PATH)
    remaining_indices = [
        idx for idx, sample in enumerate(dataset.samples) if sample["battle_id"] not in done
    ]
    if args.max_trials is not None:
        remaining_indices = remaining_indices[:args.max_trials]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, remaining_indices),
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    notify(
        f"Skipping {len(dataset) - len(remaining_indices)} already-done evaluations, "
        f"{len(remaining_indices)} remaining."
    )
    notify(
        f"**`projector` run started**\n"
        f"Run ID: `{args.run_id or 'default'}`\n"
        f"Pending evals: `{len(remaining_indices)}`\n"
        f"Checkpoint: `{CHECKPOINT_PATH}`"
    )

    completed_battles = set(projector_scores.keys())
    battles_since_save = 0

    notify(f"Running Projector on {len(remaining_indices)} pending model evaluations …")
    pbar = tqdm(total=len(remaining_indices))

    for item in loader:
        battle_id = item["battle_id"][0] if isinstance(item["battle_id"], list) else item["battle_id"]

        if battle_id not in projector_scores:
            projector_scores[battle_id] = {}

        try:
            clean_images = item["clean_images"].to(device)
            poisoned_images = item["poisoned_images"].to(device)

            with torch.amp.autocast('cuda', dtype=dtype):
                if clean_images.dim() == 5 and clean_images.size(0) == 1:
                    clean_images = clean_images.squeeze(0)
                    poisoned_images = poisoned_images.squeeze(0)

                res_clean = metric(
                    clean_images,
                    return_details=False,
                    masking="mutual-recall",
                    mask_background=True,
                    pair_mode=None,
                )
                res_poisoned = metric(
                    poisoned_images,
                    return_details=False,
                    masking="mutual-recall",
                    mask_background=True,
                    pair_mode=None,
                )

            projector_scores[battle_id] = {
                "trial": battle_id,
                "dataset": "adv_toys4k",
                "projector_without_oddity": {
                    "long_range_score": float(res_clean["long_range_score"])
                },
                "projector_with_oddity": {
                    "long_range_score": float(res_poisoned["long_range_score"])
                }
            }

            del clean_images, poisoned_images, res_clean, res_poisoned
            done.add(battle_id)
            pbar.update(1)
            pbar.set_postfix_str(
                f"{battle_id}: clean={projector_scores[battle_id]['projector_without_oddity']['long_range_score']:.4f}")
        except torch.cuda.OutOfMemoryError:
            save_checkpoint(projector_scores, CHECKPOINT_PATH)
            model_n = "adv_toys4k" if args.dataset == "adv_toys4k" else item.get("model_name", "unknown")
            notify_oom("projector_inference.py", battle_id, model_n, CHECKPOINT_PATH)
            raise
        finally:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        if battle_id not in completed_battles:
            completed_battles.add(battle_id)
            battles_since_save += 1
            if battles_since_save >= CHECKPOINT_INTERVAL:
                save_checkpoint(projector_scores, CHECKPOINT_PATH)
                print(f"\nCheckpoint saved ({len(completed_battles)} battles done)")
                battles_since_save = 0

    pbar.close()

    save_checkpoint(projector_scores, CHECKPOINT_PATH)
    print(f"Done. Computed scores for {len(projector_scores)} battles.")
    notify(
        f"**`projector` inference loop complete**\n"
        f"Battles scored: `{len(projector_scores)}`\n"
        f"Checkpoint: `{CHECKPOINT_PATH}`"
    )

    summary = {}
    raw_deltas = [
        r["projector_without_oddity"]["long_range_score"] - r["projector_with_oddity"]["long_range_score"]
        for r in projector_scores.values()
    ]

    arr_deltas = np.array(raw_deltas)
    summary["long_range_score_delta"] = {
        "mean": float(np.mean(arr_deltas)),
        "std": float(np.std(arr_deltas)),
        "median": float(np.median(arr_deltas)),
        "min": float(np.min(arr_deltas)),
        "max": float(np.max(arr_deltas)),
        "fraction_positive": float(np.mean(arr_deltas > 0)),
        "n": len(arr_deltas),
    }

    summary_path = os.path.join(RESULTS_DIR, "projector_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("AdvToys4K Projector Results Summary")
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
