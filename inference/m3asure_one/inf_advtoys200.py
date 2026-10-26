"""
Run M3aSURE latent alignment inference on AdvToys200

Usage:
    uve run inference/inf_advtoys200.py --subset 200 --seed 42
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import gc
import torch
from tqdm import tqdm

from inference.inference_helpers import (
    build_remaining_loader,
    load_checkpoint,
    notify_oom,
    save_checkpoint,
    save_predictions,
    device_dtype,
    autocast_context,
    save_advtoys200_predictions,
)

from data import (
    InferenceAdvtoys200Dataset
)
from models import (
    M3aSURELatentAlignment,
    load_weights, load_meta,
)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def main() -> None:
    DEFAULT_ROOT_ADVTOYS200 = "/workspace/m3asure/data/assets/adv_toys4k_processed_4_views"
    CHECKPOINT_INTERVAL = 100

    parser = argparse.ArgumentParser()
    ckpt_name = "m3asure_one_train_linear_e100_ablation2_margin10_v12_mean_pool_reg"
    parser.add_argument("--weights", type=str,
                        default=f"/workspace/m3asure/results/probe_checkpoints/{ckpt_name}/latent_alignment_best_acc.pt")
    parser.add_argument("--data-root", type=str,
                        default=DEFAULT_ROOT_ADVTOYS200)
    parser.add_argument("--subset", type=int, default=200,
                        help="Number of prompts to randomly sample (asset mode only)")  # 200 default
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --subset sampling")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=None, help="Cap number of trials for debugging")
    args = parser.parse_args()

    data_root = args.data_root
    results_base = f"{PROJECT_ROOT}/results"
    results_dir = os.path.join(results_base, "adv_toys4k", ckpt_name)
    if args.run_id:
        results_dir += f"_{args.run_id}"
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = device_dtype(device)
    print(f"device: {device}")
    print(f"autocast dtype: {dtype}")
    print(f"dataset: {args.dataset}  mode: {args.mode}")

    dataset = InferenceAdvtoys200Dataset(
        root=data_root,
        preprocess_fn=load_and_preprocess_images,
    )
    print(f"Dataset ready: {len(dataset)} evaluations.")

    reward_checkpoint = load_weights(args.checkpoint, device)
    probe_kwargs = load_meta(reward_checkpoint)

    # TODO: add cache

    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    metric = M3aSURELatentAlignment(vggt_model=vggt_model, dropout=0.0, expected_views=12, **probe_kwargs).to(
        device)
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

    # restore ckpt
    remaining_indices = [
        idx for idx, sample in enumerate(dataset.samples) if sample["battle_id"] not in done
    ]
    if args.max_trials is not None:
        remaining_indices = remaining_indices[:args.max_trials]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, remaining_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(
        f"Skipping {len(dataset) - len(remaining_indices)} already-done evaluations, "
        f"{len(remaining_indices)} remaining."
    )

    # ── Inference loop ────────────────────────────────────────────────────
    completed_keys = set(latent_scores.keys())
    evals_since_save = 0
    pbar = tqdm(total=len(remaining_indices))

    for item in loader:
        battle_id = item["battle_id"][0] if isinstance(item["battle_id"], list) else item["battle_id"]

        if battle_id not in latent_scores:
            latent_scores[battle_id] = {}

        try:
            clean_images = item["clean_images"].to(device, non_blocking=True)
            poisoned_images = item["poisoned_images"].to(device, non_blocking=True)

            # adv_toys4k has 4 total images: 1 source + 3 renders. Override expected_views.
            original_ev = metric.expected_views
            metric.expected_views = 3

            with torch.no_grad(), autocast_context(device, dtype):
                if clean_images.dim() == 5 and clean_images.size(0) == 1:
                    clean_images = clean_images.squeeze(0)
                    poisoned_images = poisoned_images.squeeze(0)
                res_clean = metric(clean_images.unsqueeze(0))
                res_poisoned = metric(poisoned_images.unsqueeze(0))

            latent_scores[battle_id] = {
                "trial": battle_id,
                "dataset": "adv_toys4k",
                "latent_alignment_without_oddity": {
                    "score": float(res_clean["score"][0].detach().cpu()),
                    "per_view_scores": res_clean["per_view_scores"]
                },
                "latent_alignment_with_oddity": {
                    "score": float(res_poisoned["score"][0].detach().cpu()),
                    "per_view_scores": res_poisoned["per_view_scores"]
                }
            }

            metric.expected_views = original_ev
            del clean_images, poisoned_images, res_clean, res_poisoned
            done.add(battle_id)
            pbar.update(1)
            pbar.set_postfix_str(
                f"{battle_id}: clean={latent_scores[battle_id]['latent_alignment_without_oddity']['score']:.4f}")

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

    save_advtoys200_predictions(latent_scores, results_dir)


if __name__ == "__main__":
    main()
