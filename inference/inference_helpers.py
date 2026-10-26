"""
Shared checkpoint and I/O utilities for inference scripts.
"""

import os
import json
from typing import Callable
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from contextlib import nullcontext

try:
    from _runpod_hook import send_discord_notification as notify
except ModuleNotFoundError:
    notify = print


def notify_oom(
    script: str, battle_id: str, model_name: str, checkpoint_path: str
) -> None:
    notify(
        f"**OOM** in `{script}`\n"
        f"sample: `{battle_id}` / `{model_name}`\n"
        f"checkpoint: `{checkpoint_path}`\n"
        f"Rerun the script to resume from checkpoint."
    )


def device_dtype(device: torch.device) -> torch.dtype:
    """mps safety"""
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def autocast_context(device: torch.device, dtype: torch.dtype):
    """mps safety"""
    if device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def load_checkpoint(checkpoint_path: str) -> tuple[dict, set]:
    """
    Load existing checkpoint and return (scores_dict, set_of_done_keys).

    Each done key is a (battle_id, model_name) tuple.
    """

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            scores = json.load(f)
        done = {(bid, model) for bid, models in scores.items() for model in models}
        print(f"Resumed from checkpoint: {len(done)} model evaluations already done.")
        return scores, done
    else:
        notify(f"Run started from scratch.")
    return {}, set()


def save_checkpoint(scores: dict, checkpoint_path: str) -> None:
    """Atomically write checkpoint (write to tmp then rename)."""
    notify(f"Saving checkpoint to {checkpoint_path}")
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(scores, f, indent=2, default=make_serializable)
    os.replace(tmp_path, checkpoint_path)


def collate_single(batch):
    """Pass-through collate for batch_size=1 — avoids stacking dicts."""
    return batch[0]


def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Type {type(obj)} is not serializable")


def scores_to_predictions(
    scores: dict, battles_data: dict, extract_score: Callable, model_name: str = "model"
) -> dict:
    """
    Args:
        scores: dict with [bid][score_key][model_name] -> float
        battles_data: original battles data with model order
        extract_score: extract score from score dict

    Returns:
        predictions dict
    """
    predictions = {}

    for bid, scores_data in scores.items():
        if bid not in battles_data:
            continue

        models = battles_data[bid]["models"]
        score_0, score_1 = extract_score(scores_data, models)

        # higher score = more consistent
        if score_0 > score_1:
            pred = {"label_0": 1.0, "label_1": 0.0}
        elif score_1 > score_0:
            pred = {"label_0": 0.0, "label_1": 1.0}
        else:
            pred = {"label_0": 0.5, "label_1": 0.5}

        predictions[bid] = {
            "name": bid,
            "models": models,
            f"{model_name}_prediction": pred,
            "raw_scores": {models[0]: score_0, models[1]: score_1},
        }

    return predictions


def save_3dgenbench_predictions(
    scores: dict,
    battles_data: dict,
    results_dir: str,
    metric_name: str,
    extract_fns: dict[str, Callable],
) -> None:
    """
    Save full raw scores and convert to predictions for each extract function.
    Args:
        scores: The raw scores dict ``{battle_id: {model_name: {...}}}``.
        battles_data: Original battles JSON used for ``scores_to_predictions``.
        results_dir: Directory to write output files into.
        metric_name: Base name of the metric (e.g. ``"m3asure"`` or ``"met3r"``).
        extract_fns: Mapping from suffix to an ``extract_score`` callable.
            For example ``{"": extract_score, "_long_range": extract_lr}``
            produces ``<metric>_scores.json`` and ``<metric>_scores_long_range.json``.
    """
    os.makedirs(results_dir, exist_ok=True)

    full_path = os.path.join(results_dir, f"{metric_name}_full.json")
    with open(full_path, "w") as f:
        json.dump(scores, f, indent=2, default=make_serializable)
    print(f"Saved raw scores to {full_path}")

    summary_lines = [
        f"**`{metric_name}` inference complete**",
        f"Battles: {len(scores)}  |  Dir: `{results_dir}`",
    ]

    for suffix, extract_fn in extract_fns.items():
        pred_name = f"{metric_name}{suffix}"
        predictions = scores_to_predictions(
            scores,
            battles_data,
            extract_score=extract_fn,
            model_name=pred_name,
        )
        pred_path = os.path.join(results_dir, f"{pred_name}_scores.json")
        with open(pred_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved {len(predictions)} predictions to {pred_path}")
        summary_lines.append(f"`{pred_name}`: {len(predictions)} predictions")

    notify("\n".join(summary_lines))


def build_remaining_loader(
    dataset,
    done: set,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, list[int]]:
    """
    Build DataLoader that yields only unfinished dataset samples.
    """
    remaining_indices = [
        i
        for i, sample in enumerate(dataset.samples)
        if (sample["battle_id"], sample["model_name"]) not in done
    ]
    loader = DataLoader(
        Subset(dataset, remaining_indices),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_single,
        pin_memory=pin_memory,
    )
    return loader, remaining_indices


def save_hi3dbench_predictions(latent_scores: dict, results_dir: str) -> None:
    """Reshape per-asset scores to {model: {prompt_idx: scores}} and save."""
    per_model: dict[str, dict[str, dict]] = {}
    for bid, models_data in latent_scores.items():
        for model_name, score_data in models_data.items():
            if model_name not in per_model:
                per_model[model_name] = {}
            prompt_idx = bid.rsplit("_", 1)[-1] if "_" in bid else bid
            per_model[model_name][prompt_idx] = score_data

    full_path = os.path.join(results_dir, "latent_alignment_full.json")
    with open(full_path, "w") as f:
        json.dump(per_model, f, indent=2, default=make_serializable)
    print(f"Saved per-asset scores to {full_path}")

    print("\n" + "=" * 60)
    print("Per-Asset Ranking (by mean_score)")
    print("=" * 60)
    ranking = []
    for model_name, prompts in per_model.items():
        scores = [v["mean_score"] for v in prompts.values() if "mean_score" in v]
        if scores:
            ranking.append((model_name, np.mean(scores), np.std(scores), len(scores)))
    ranking.sort(key=lambda r: r[1], reverse=True)
    for i, (model, avg, std, n) in enumerate(ranking, 1):
        print(f"  {i}. {model:25s}  mean={avg:.4f}  std={std:.4f}  n={n}")


def save_advtoys200_predictions(latent_scores: dict, results_dir: str) -> None:
    """AdvToys200 summary"""
    summary = {}
    raw_deltas = [
        r["latent_alignment_without_oddity"]["score"]
        - r["latent_alignment_with_oddity"]["score"]
        for r in latent_scores.values()
    ]

    arr_deltas = np.array(raw_deltas)
    summary["score_delta"] = {
        "mean": float(np.mean(arr_deltas)),
        "std": float(np.std(arr_deltas)),
        "median": float(np.median(arr_deltas)),
        "min": float(np.min(arr_deltas)),
        "max": float(np.max(arr_deltas)),
        "fraction_positive": float(np.mean(arr_deltas > 0)),
        "n": len(arr_deltas),
    }

    summary_path = os.path.join(results_dir, "latent_alignment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("AdvToys4K Latent Alignment Results Summary")
    print("=" * 60)
    for key, stats in summary.items():
        print(f"\n{key}:")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
