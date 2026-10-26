"""
Run 3DGenScore on 3DGen-Bench battles.

Usage:
    uve run data/3d_gen_bench/download_3dgenscore_weights.py
    uve run data/3d_gen_bench/preprocess_3dgenscore.py --split test
    uve run inf_3dgenbench.py --split test
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from contextlib import nullcontext
import argparse
import gc
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from inference.inference_helpers import (
    save_checkpoint,
    collate_single,
)
from inference_helpers import save_3dgenbench_predictions


from models import load_3dgenscore_model, EVAL_DIMS, calc_probs

from data import Inference3dgenbenchDataset

from helpers import (
    load_battle_checkpoint,
    format_result,
    extract_aggregate,
    make_dim_extractor,
    legacy_score_format,
)


def main() -> None:
    DEFAULT_THREEDGEN_BENCH_ROOT = PROJECT_ROOT / "3DGen-Bench"
    sys.path.insert(0, str(DEFAULT_THREEDGEN_BENCH_ROOT))
    PROCESSOR_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    DEFAULT_DATA_ROOT = "/workspace/m3asure/data/assets/3d_gen_bench_processed"
    DEFAULT_ORIGINAL_ROOT = "/workspace/m3asure/data/assets/3d_gen_bench_original"
    DEFAULT_CON_IMAGES_DIR = (
        "/workspace/m3asure/data/assets/3d_gen_bench_3dgenscore/con_images"
    )
    DEFAULT_WEIGHTS_DIR = (
        "/workspace/m3asure/3DGen-Bench/checkpoint/3dgen-score-mvclip-v1"
    )
    CHECKPOINT_INTERVAL = 100
    NUM_WORKERS = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--original-root", type=str, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--con-images-dir", type=str, default=DEFAULT_CON_IMAGES_DIR)
    parser.add_argument("--source-root", type=str, default="")
    parser.add_argument("--text-gallery-path", type=str, default="")
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

    results_dir = f"/workspace/m3asure/results/3dgenscore_{args.split}"
    if args.run_id:
        results_dir += f"_{args.run_id}"

    # load ckpt
    inf_ckpt_path = os.path.join(results_dir, "3dgenscore_checkpoint.json")
    battles_json_path = os.path.join(args.data_root, f"{args.split}.json")
    with open(battles_json_path, "r", encoding="utf-8") as f:
        battles_data = json.load(f)
    dataset = Inference3dgenbenchDataset(
        battles_data=battles_data,
        con_images_dir=args.con_images_dir,
        source_root=args.source_root,
        original_root=args.original_root,
    )
    scores, done = load_battle_checkpoint(inf_ckpt_path)
    scores = legacy_score_format(scores, battles_data)
    remaining_indices = [
        idx
        for idx, sample in enumerate(dataset.samples)
        if sample["battle_id"] not in done
    ]
    if args.max_trials is not None:
        remaining_indices = remaining_indices[: args.max_trials]

    loader = DataLoader(
        torch.utils.data.Subset(dataset, remaining_indices),
        batch_size=1,
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
                    normal_image_0=item["normal_image_0"],
                    normal_image_1=item["normal_image_1"],
                    rgb_image_0=item["rgb_image_0"],
                    rgb_image_1=item["rgb_image_1"],
                )
            scores[battle_id] = format_result(item, probs)
        except Exception:
            save_checkpoint(scores, inf_ckpt_path)
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        done.add(battle_id)
        battles_since_save += 1
        pbar.update(1)
        pbar.set_postfix_str(
            f"{battle_id}: {scores[battle_id][item['models'][0]]['score']:.4f}/"
            f"{scores[battle_id][item['models'][1]]['score']:.4f}"
        )

        if battles_since_save >= CHECKPOINT_INTERVAL:
            save_checkpoint(scores, inf_ckpt_path)
            print(f"\nCheckpoint saved ({len(done)} battles done)")
            battles_since_save = 0

    pbar.close()
    save_checkpoint(scores, inf_ckpt_path)
    print(f"Done. Computed scores for {len(scores)} battles.")
    save_3dgenbench_predictions(
        scores,
        battles_data,
        results_dir=results_dir,
        metric_name="3dgenscore",
        extract_fns={
            "": extract_aggregate,
            **{f"_{dim}": make_dim_extractor(dim) for dim in EVAL_DIMS},
        },
    )

    pred_path = os.path.join(results_dir, "3dgenscore_scores.json")
    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    for bid in list(predictions.keys())[:5]:
        prediction = predictions[bid]
        print(
            f"  {bid}: {prediction['models']} -> "
            f"raw={prediction['raw_scores']} "
            f"pred={prediction['3dgenscore_prediction']}"
        )


if __name__ == "__main__":
    main()
