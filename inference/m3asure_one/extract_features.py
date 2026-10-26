"""
Extract VGGT features offline for all (model_name, offline_idx) pairs and cache as .pt files.

Usage:
    uve run inference/extract_features.py --batch-size 4
"""

import sys
import os

from data import sorted_render_paths, prepare_pair_images

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import gc
import json
from contextlib import nullcontext
import torch
from torch import Tensor
from tqdm import tqdm

from data import (
    DEFAULT_VIEW_INDICES,
    collect_unique_samples
)
from vggt.models.vggt import VGGT
from jaxtyping import Float


def _device_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def _extract_image_vectors(
        vggt_model: VGGT,
        pair_images: Float[Tensor, "b 2 3 h w"],
        token_dim: int = 2048,
) -> Float[Tensor, "b 2 2048"]:
    """Extract pooled image patch tokens from VGGT.

    Args:
        vggt_model: Frozen VGGT model.
        pair_images: (B_pairs, 2, 3, H, W), source at frame 0 and render at frame 1.

    Returns:
        (B_pairs, 2, 2048) pooled final patch-token vectors.
    """
    with torch.no_grad():
        aggregated_tokens_list, patch_start_idx = vggt_model.aggregator(pair_images)
        patch_tokens = aggregated_tokens_list[-1][:, :, patch_start_idx:, :]
        image_vectors = patch_tokens.mean(dim=2)

    if image_vectors.shape[-1] != token_dim:
        raise ValueError(
            f"Expected VGGT pooled token dim {token_dim}, got {image_vectors.shape[-1]}."
        )
    return image_vectors


def main() -> None:
    DATA_ROOT = "/workspace/m3asure/data/assets/3dgenbench_processed_12_views"
    SOURCE_IMG_ROOT = os.path.join(DATA_ROOT, "images_prompts_510")
    DEFAULT_CACHE_ROOT = "/workspace/m3asure/data/assets/3d_gen_bench_vggt_pooled_embeddings"
    DEFAULT_TRAIN_JSON = os.path.join(DATA_ROOT, "train.json")
    DEFAULT_VALID_JSON = os.path.join(DATA_ROOT, "valid.json")
    DEFAULT_TEST_JSON = os.path.join(DATA_ROOT, "test.json")

    parser = argparse.ArgumentParser(description="Extract and cache VGGT features for probe training.")
    parser.add_argument("--cache-root", type=str, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--batch-size", type=int, default=11,
                        help="Number of (model, object) sequences to process per VGGT forward pass. "
                             "Each sequence produces 4 view-pairs. Increase to use more GPU memory.")  # 4 gives ~ 5gb vram
    parser.add_argument("--train-json", type=str, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--valid-json", type=str, default=DEFAULT_VALID_JSON)
    parser.add_argument("--test-json", type=str, default=DEFAULT_TEST_JSON)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _device_dtype(device)
    print(f"device: {device}")
    print(f"autocast dtype: {dtype}")

    view_indices = tuple(DEFAULT_VIEW_INDICES)
    json_paths = [args.train_json, args.valid_json, args.test_json]
    samples = collect_unique_samples(json_paths, DATA_ROOT, SOURCE_IMG_ROOT, view_indices)
    print(f"Total unique (model, object) pairs: {len(samples)}")

    remaining = []
    for s in samples:
        cache_path = os.path.join(args.cache_root, s["model_name"], f"{s['offline_idx']}.pt")
        if os.path.exists(cache_path):
            continue
        remaining.append(s)

    skipped = len(samples) - len(remaining)
    if skipped:
        print(f"Skipping {skipped} already-cached samples, {len(remaining)} remaining.")
    if not remaining:
        print("All features are already cached. Nothing to do.")
        return

    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    vggt_model.eval()
    for param in vggt_model.parameters():
        param.requires_grad_(False)
    batch_size = args.batch_size
    pbar = tqdm(total=len(remaining), desc="Extracting features")

    i = 0
    while i < len(remaining):
        batch_samples = remaining[i: i + batch_size]
        actual_batch = len(batch_samples)

        try:
            pair_list = []
            for s in batch_samples:
                pairs = prepare_pair_images(s["source_path"], s["render_paths"])
                pair_list.append(pairs)  # each is (V, 2, 3, H, W)

            # stack all views for efficient inference (B * V, 2, 3, H, W)
            num_views = pair_list[0].shape[0]
            batched_pairs = torch.cat(pair_list, dim=0).to(device, non_blocking=True)

            with _autocast_context(device, dtype):
                image_vectors = _extract_image_vectors(vggt_model, batched_pairs)  # (B * V, 2, D)

            # restore shape (B, V, 2, D)
            image_vectors = image_vectors.reshape(actual_batch, num_views, 2, -1)

            # store
            for j, s in enumerate(batch_samples):
                cache_path = os.path.join(args.cache_root, s["model_name"], f"{s['offline_idx']}.pt")
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(image_vectors[j].cpu(), cache_path)

            del batched_pairs, image_vectors, pair_list
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM at batch starting index {i} with batch_size={batch_size}.")
            # retry same index with smaller batch
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
                print(f"Reducing batch_size to {batch_size} and retrying.")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            else:
                raise
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        pbar.update(actual_batch)
        i += actual_batch

    pbar.close()
    print(f"Done. Cached features for {len(remaining)} (model, object) pairs under {args.cache_root}")


if __name__ == "__main__":
    main()
