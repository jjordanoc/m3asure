"""
Run MEt3R on 3DGen-Bench battles.

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
from data import Inference3dgenbenchDataset, load_images_as_tensor_met3r

try:
    import met3r
except ImportError as e:
    raise ImportError("Make sure to install MEt3R properly.") from e


def main():
    DEFAULT_DATA_ROOT = "/workspace/m3asure/data/assets/3d_gen_bench_processed"
    SOURCE_IMG_ROOT = os.path.join(DEFAULT_DATA_ROOT, "images_prompts_510")

    RESULTS_DIR = "/workspace/m3asure/results/3dgenbench/met3r"
    CHECKPOINT_INTERVAL = 200
    NUM_WORKERS = 0
    IMG_SIZE = 256
    MODE = "src-target"  # best for met3r

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
    BATTLES_JSON_PATH = os.path.join(data_root, f"{args.set}.json")

    print(f"Running inference for {args.run_id} in {MODE} mode")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    metric = met3r.MEt3R(img_size=IMG_SIZE).to(device)
    print("MEt3R metric ready.")

    with open(BATTLES_JSON_PATH, "r") as f:
        battles_data = json.load(f)

    dataset = Inference3dgenbenchDataset(
        battles_json=battles_data,
        data_root=data_root,
        source_root=SOURCE_IMG_ROOT,
        preprocess=load_images_as_tensor_met3r,
        include_source=True,
    )
    print(f"Dataset ready: {len(dataset)} evaluations.")

    # ckpt
    os.makedirs(RESULTS_DIR, exist_ok=True)
    CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "met3r_checkpoint.json")
    met3r_scores, done = load_checkpoint(CHECKPOINT_PATH)
    loader, remaining_indices = build_remaining_loader(
        dataset,
        done,
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
            model_name = item["model_name"]
            frames = item["processed_images"]  # (N, C, H, W)
            n_frames = frames.shape[0]

            if MODE == "sliding":
                # sliding-window pairs (original): (N-1, 2, C, H, W)
                pairs = torch.stack(
                    [torch.stack([frames[i], frames[i + 1]], dim=0) for i in range(n_frames - 1)],
                    dim=0,
                ).to(device)
            elif MODE == "src-target":
                # pair source-target: (N-1, 2, C, H, W)
                pairs = torch.stack(
                    [torch.stack([frames[0], frames[i]], dim=0) for i in range(1, n_frames)],
                    dim=0,
                ).to(device)
            else:
                raise ValueError(f"Invalid mode: {MODE}")

            with torch.amp.autocast("cuda", dtype=dtype):
                scores, *_ = metric(pairs)

            mean_score = scores.mean().item()
            per_pair = scores.cpu().tolist()
            met3r_scores[battle_id][model_name] = {
                "score": mean_score,
                "per_pair_scores": per_pair,
            }

            del frames, pairs, scores
            done.add((battle_id, model_name))
            pbar.update(1)
            pbar.set_postfix_str(
                f"{battle_id}/{model_name}: "
                f"{met3r_scores[battle_id][model_name]['score']:.4f}"
            )

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

    def extract_score(scores_data, models):
        """for met3r lower = more consistent, negate"""
        s0 = -scores_data.get(models[0], {}).get("score", 0.0)
        s1 = -scores_data.get(models[1], {}).get("score", 0.0)
        return s0, s1

    save_predictions(
        met3r_scores,
        battles_data,
        results_dir=RESULTS_DIR,
        metric_name="met3r",
        extract_fns={"": extract_score},
    )

    pred_path = os.path.join(RESULTS_DIR, "met3r_scores.json")
    with open(pred_path, "r") as f:
        predictions = json.load(f)
    for bid in list(predictions.keys())[:5]:
        p = predictions[bid]
        print(f"  {bid}: {p['models']} → raw={p['raw_scores']} pred={p['met3r_prediction']}")


if __name__ == "__main__":
    main()
