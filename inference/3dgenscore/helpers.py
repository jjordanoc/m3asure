import json
import math
import os
from typing import Any
from models import EVAL_DIMS


def load_battle_checkpoint(checkpoint_path: str) -> tuple[dict[str, Any], set[str]]:
    if not os.path.exists(checkpoint_path):
        return {}, set()

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        scores = json.load(f)

    done = set(scores.keys())
    print(f"Resumed from checkpoint: {len(done)} battles already done.")
    return scores, done


def format_result(
    item: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    # raw = results["raw"]
    probs = results["probs"]

    if len(probs) != len(EVAL_DIMS):
        raise ValueError(
            f"Expected {len(EVAL_DIMS)} score dimensions, got {len(probs)}."
        )

    for dim, pair_probs in zip(EVAL_DIMS, probs, strict=True):
        if len(pair_probs) != 2:
            raise ValueError(
                f"{dim} returned {len(pair_probs)} probabilities; expected 2."
            )
        for prob in pair_probs:
            if not math.isfinite(float(prob)):
                raise RuntimeError(
                    f"Non-finite 3DGenScore probability for {dim}: {prob}"
                )

    if item.get("dataset") == "adv_toys4k":
        without_oddity = {}
        with_oddity = {}
        for idx, dim in enumerate(EVAL_DIMS):
            without_oddity[f"{dim}_score"] = float(probs[idx][0])
            with_oddity[f"{dim}_score"] = float(probs[idx][1])

        return {
            "trial": item["battle_id"],
            "dataset": "adv_toys4k",
            "3dgenscore_without_oddity": without_oddity,
            "3dgenscore_with_oddity": with_oddity,
        }

    aggregate = [
        sum(pair_probs[0] for pair_probs in probs) / len(probs),
        sum(pair_probs[1] for pair_probs in probs) / len(probs),
    ]

    models = item["models"]
    return {
        models[0]: {
            "score": float(aggregate[0]),
            "per_pair_scores": [float(pair_probs[0]) for pair_probs in probs],
            "probabilities": {
                dim: float(pair_probs[0])
                for dim, pair_probs in zip(EVAL_DIMS, probs, strict=True)
            },
            "task": item["task"],
            "offline_idx": item["offline_idx"],
        },
        models[1]: {
            "score": float(aggregate[1]),
            "per_pair_scores": [float(pair_probs[1]) for pair_probs in probs],
            "probabilities": {
                dim: float(pair_probs[1])
                for dim, pair_probs in zip(EVAL_DIMS, probs, strict=True)
            },
            "task": item["task"],
            "offline_idx": item["offline_idx"],
        },
    }


def extract_aggregate(
    scores_data: dict[str, Any], models: list[str]
) -> tuple[float, float]:
    return scores_data[models[0]]["score"], scores_data[models[1]]["score"]


def make_dim_extractor(dim: str):
    def extract_dim(
        scores_data: dict[str, Any], models: list[str]
    ) -> tuple[float, float]:
        return (
            scores_data[models[0]]["probabilities"][dim],
            scores_data[models[1]]["probabilities"][dim],
        )

    return extract_dim


def legacy_score_format(
    scores: dict[str, Any], battles_data: dict[str, Any]
) -> dict[str, Any]:
    upgraded = {}
    for battle_id, score_data in scores.items():
        if battle_id not in battles_data:
            upgraded[battle_id] = score_data
            continue
        if "aggregate" not in score_data or "probabilities" not in score_data:
            upgraded[battle_id] = score_data
            continue

        models = battles_data[battle_id]["models"]
        upgraded[battle_id] = {
            models[0]: {
                "score": float(score_data["aggregate"][0]),
                "per_pair_scores": [
                    float(score_data["probabilities"][dim][0]) for dim in EVAL_DIMS
                ],
                "probabilities": {
                    dim: float(score_data["probabilities"][dim][0]) for dim in EVAL_DIMS
                },
                "task": score_data["task"],
                "offline_idx": score_data["offline_idx"],
            },
            models[1]: {
                "score": float(score_data["aggregate"][1]),
                "per_pair_scores": [
                    float(score_data["probabilities"][dim][1]) for dim in EVAL_DIMS
                ],
                "probabilities": {
                    dim: float(score_data["probabilities"][dim][1]) for dim in EVAL_DIMS
                },
                "task": score_data["task"],
                "offline_idx": score_data["offline_idx"],
            },
        }
    return upgraded
