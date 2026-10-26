"""
Run MEt3R on AdvToys200 trials.

Usage:
    Install met3r.
    uv run inf_3dgenbench.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import gc
import json
import torch
from tqdm import tqdm

from inference.inference_helpers import (
    load_checkpoint,
    save_checkpoint,
    build_remaining_loader,
    save_predictions,
)
from data import load_images_as_tensor_met3r, InferenceAdvtoys200Dataset

try:
    import met3r
except ImportError as e:
    raise ImportError("Make sure to install MEt3R properly.") from e


def main():
    DEFAULT_DATA_ROOT = "/workspace/m3asure/data/assets/adv_toys4k_processed"
    RESULTS_DIR = "/workspace/m3asure/results/advtoys200/met3r"
    CHECKPOINT_INTERVAL = 200
    NUM_WORKERS = 0
    IMG_SIZE = 256
    MODE = "src-target"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--mode", type=str, default="src-target")
    parser.add_argument("--set", type=str, default="valid")
    parser.add_argument("--max-trials", type=int, default=None, help="Cap number of trials for debugging")
    args = parser.parse_args()
    if args.mode == "sliding":
        MODE = "sliding"
    if args.run_id:
        RESULTS_DIR += f"_{args.run_id}"

    data_root = args.data_root
    print(f"Running inference for {args.run_id} in {MODE} mode")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    metric = met3r.MEt3R(img_size=IMG_SIZE).to(device)
    print("MEt3R metric ready.")

    dataset = InferenceAdvtoys200Dataset(
        root=data_root,
        preprocess_fn=load_images_as_tensor_met3r,
    )
    print(f"Dataset ready: {len(dataset)} evaluations.")

    # ckpt
    os.makedirs(RESULTS_DIR, exist_ok=True)
    CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "met3r_checkpoint.json")

    # load ckpt
    met3r_scores, done = load_checkpoint(CHECKPOINT_PATH)
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
    print(
        f"Skipping {len(dataset) - len(remaining_indices)} already-done evaluations, "
        f"{len(remaining_indices)} remaining."
    )

    completed_battles = set(met3r_scores.keys())
    battles_since_save = 0

    print(f"Running MEt3R on {len(remaining_indices)} pending model evaluations …")
    pbar = tqdm(total=len(remaining_indices))

    for item in loader:
        battle_id = item["battle_id"][0] if isinstance(item["battle_id"], list) else item["battle_id"]

        if battle_id not in met3r_scores:
            met3r_scores[battle_id] = {}

        try:
            clean_images = item["clean_images"].to(device)
            poisoned_images = item["poisoned_images"].to(device)

            if clean_images.dim() == 5 and clean_images.size(0) == 1:
                clean_images = clean_images.squeeze(0)
                poisoned_images = poisoned_images.squeeze(0)

            n_frames = clean_images.shape[0]
            if n_frames < 2:
                raise ValueError("AdvToys4k dataset must have at least 2 frames (source + render).")

            if MODE == "sliding":
                pairs_clean = torch.stack(
                    [torch.stack([clean_images[i], clean_images[i + 1]], dim=0) for i in range(n_frames - 1)],
                    dim=0)
                pairs_poisoned = torch.stack(
                    [torch.stack([poisoned_images[i], poisoned_images[i + 1]], dim=0) for i in range(n_frames - 1)],
                    dim=0)
            elif MODE == "src-target":
                pairs_clean = torch.stack(
                    [torch.stack([clean_images[0], clean_images[i]], dim=0) for i in range(1, n_frames)], dim=0)
                pairs_poisoned = torch.stack(
                    [torch.stack([poisoned_images[0], poisoned_images[i]], dim=0) for i in range(1, n_frames)],
                    dim=0)

            with torch.amp.autocast("cuda", dtype=dtype):
                scores_clean, *_ = metric(pairs_clean)
                scores_poisoned, *_ = metric(pairs_poisoned)

            met3r_scores[battle_id] = {
                "trial": battle_id,
                "dataset": "adv_toys4k",
                "met3r_without_oddity": {
                    "score": scores_clean.mean().item()
                },
                "met3r_with_oddity": {
                    "score": scores_poisoned.mean().item()
                }
            }

            del clean_images, poisoned_images, pairs_clean, pairs_poisoned, scores_clean, scores_poisoned
            done.add(battle_id)
            pbar.update(1)
            pbar.set_postfix_str(
                f"{battle_id}: clean={met3r_scores[battle_id]['met3r_without_oddity']['score']:.4f}")

        except torch.cuda.OutOfMemoryError:
            save_checkpoint(met3r_scores, CHECKPOINT_PATH)
            model_n = "adv_toys4k" if args.dataset == "adv_toys4k" else item.get("model_name", "unknown")
            notify_oom("inference/latent_alignment_inference.py", battle_id, model_n, inf_ckpt_path)
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        if battle_id not in completed_battles:
            completed_battles.add(battle_id)
            battles_since_save += 1
            if battles_since_save >= CHECKPOINT_INTERVAL:
                save_checkpoint(met3r_scores, CHECKPOINT_PATH)
                print(f"\nCheckpoint saved ({len(completed_battles)} battles done)")
                battles_since_save = 0

    pbar.close()

    save_checkpoint(met3r_scores, CHECKPOINT_PATH)
    print(f"Done. Computed scores for {len(met3r_scores)} battles.")

    summary = {}

    raw_deltas = [
        r["met3r_with_oddity"]["score"] - r["met3r_without_oddity"]["score"]
        for r in met3r_scores.values()
    ]

    arr_deltas = np.array(raw_deltas)
    summary["score_delta"] = {
        "mean": float(np.mean(arr_deltas)),
        "std": float(np.std(arr_deltas)),
        "median": float(np.median(arr_deltas)),
        "min": float(np.min(arr_deltas)),
        "max": float(np.max(arr_deltas)),
        "fraction_clean_wins": float(np.mean(arr_deltas > 0)),
        "n": len(arr_deltas),
    }

    summary_path = os.path.join(RESULTS_DIR, "met3r_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("AdvToys4K MEt3R Results Summary")
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
