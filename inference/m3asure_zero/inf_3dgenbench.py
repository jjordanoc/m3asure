"""
Run M3aSURE projector on 3DGen-Bench battles.

Usage:

   uv run inf_3dgenbench.py --extra met3r
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

from data import Inference3dgenbenchDataset
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
    DEFAULT_DATA_ROOT = "/workspace/m3asure/data/assets/3dgenbench_processed_12_views"
    SOURCE_IMG_ROOT = os.path.join(DEFAULT_DATA_ROOT, "images_prompts_510")
    RESULTS_DIR = f"/workspace/m3asure/results/projector"
    CHECKPOINT_INTERVAL = 200
    NUM_WORKERS = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"])
    args = parser.parse_args()

    data_root = args.data_root
    BATTLES_JSON_PATH = os.path.join(data_root, f"{args.split}.json")
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

    metric = M3aSURESemanticProjector(
        vggt_model=vggt_model,
        anyup_model=anyup_model,
    ).to(device)
    print("M3aSURE metric ready.")

    with open(BATTLES_JSON_PATH, 'r') as f:
        battles_data = json.load(f)

    dataset = Inference3dgenbenchDataset(
        battles_json=battles_data,
        data_root=data_root,
        source_root=SOURCE_IMG_ROOT,
        preprocess=load_and_preprocess_images,
        include_source=True,
    )
    print(f"Dataset ready: {len(dataset)} evaluations.")

    # inf ckpt
    os.makedirs(RESULTS_DIR, exist_ok=True)
    CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "projector_checkpoint.json")
    projector_scores, done = load_checkpoint(CHECKPOINT_PATH)
    loader, remaining_indices = build_remaining_loader(
        dataset,
        done,
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
    print(f"Running Projector on {len(remaining_indices)} pending model evaluations …")
    pbar = tqdm(total=len(remaining_indices))

    for item in loader:
        battle_id = item["battle_id"][0] if isinstance(item["battle_id"], list) else item["battle_id"]

        if battle_id not in projector_scores:
            projector_scores[battle_id] = {}

        try:
            model_name = item["model_name"]
            images = item["processed_images"].to(device)  # (N, 3, H, W)

            with torch.amp.autocast('cuda', dtype=dtype):
                result = metric(
                    images,
                    return_details=False,
                    masking="mutual-recall",
                    mask_background=True,
                    pair_mode=None,
                )

            projector_scores[battle_id][model_name] = {
                "score": result["score"],
                "neighbor_score": result.get("neighbor_score", result["score"]),
                "long_range_score": result.get("long_range_score", result["score"]),
                "per_pair_scores": result["per_pair_scores"],
            }

            del images, result
            done.add((battle_id, model_name))
            pbar.update(1)
            pbar.set_postfix_str(
                f"{battle_id}/{model_name}: "
                f"{projector_scores[battle_id][model_name]['score']:.4f}"
            )

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

    def extract_score(scores_data, models):
        s0 = scores_data.get(models[0], {})["score"]
        s1 = scores_data.get(models[1], {})["score"]
        return s0, s1

    def extract_score_long_range(scores_data, models):
        s0 = scores_data.get(models[0], {})["long_range_score"]
        s1 = scores_data.get(models[1], {})["long_range_score"]
        return s0, s1

    def extract_score_neighbor(scores_data, models):
        s0 = scores_data.get(models[0], {})["neighbor_score"]
        s1 = scores_data.get(models[1], {})["neighbor_score"]
        return s0, s1

    save_predictions(
        projector_scores,
        battles_data,
        results_dir=RESULTS_DIR,
        metric_name="projector",
        extract_fns={
            "": extract_score,
            "_long_range": extract_score_long_range,
            "_neighbor": extract_score_neighbor,
        },
    )
    pred_path = os.path.join(RESULTS_DIR, "projector_scores.json")
    with open(pred_path, "r") as f:
        predictions = json.load(f)
    for bid in list(predictions.keys())[:5]:
        p = predictions[bid]
        print(f"  {bid}: {p['models']} → raw={p['raw_scores']} pred={p['projector_prediction']}")


if __name__ == "__main__":
    main()
