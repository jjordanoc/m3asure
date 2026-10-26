import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dataclasses import dataclass
from typing import Callable, Iterable, Literal
from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import Dataset
from jaxtyping import Float
from vggt.utils.load_fn import load_and_preprocess_images
import json
import random

DEFAULT_VIEW_INDICES = tuple(range(12))
HI3DBENCH_VIEW_INDICES = (0, 1, 2, 3)


def prepare_pair_images(
        source_path: str,
        render_paths: list[str],
) -> Float[Tensor, "v 2 3 h w"]:
    """
    Build the (V, 2, 3, H, W) pair tensor for a single sample.
    We use either V=4 or V=12 throughout the paper.
    """

    all_paths = [source_path, *render_paths]
    all_images = load_and_preprocess_images(all_paths)  # (1+V, 3, H, W)

    source = all_images[0:1]  # (1, 3, H, W)
    renders = all_images[1:]  # (V, 3, H, W)
    num_views = renders.shape[0]

    # pair each render with the source
    source_expanded = source.expand(num_views, -1, -1, -1)  # (V, 3, H, W)
    pair_images = torch.stack((source_expanded, renders), dim=1)  # (V, 2, 3, H, W)
    return pair_images


def load_json(path: str) -> dict:
    """just a wrapper around json.loads"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with open(path, "r") as f:
        return json.load(f)


def collect_unique_samples(
        json_paths: list[str],
        data_root: str,
        source_root: str,
        view_indices: tuple[int, ...],
) -> list[dict]:
    """Collect all unique (model_name, offline_idx) pairs across JSONs."""
    seen: set[tuple[str, int]] = set()
    samples: list[dict] = []

    for json_path in json_paths:
        if not os.path.exists(json_path):
            print(f"Warning: JSON file not found, skipping: {json_path}")
            continue

        battles = load_json(json_path)
        for battle_data in battles.values():
            if battle_data.get("task") != "image2shape":
                continue
            offline_idx = int(battle_data["offline_idx"])
            for model_name in battle_data["models"]:
                key = (model_name, offline_idx)
                if key in seen:
                    continue
                seen.add(key)

                try:
                    source_path = resolve_source_path(source_root, offline_idx)
                    render_paths = sorted_render_paths(
                        data_root, model_name, offline_idx, view_indices
                    )
                except FileNotFoundError as e:
                    print(f"Warning: skipping {key}: {e}")
                    continue
                except IndexError as e:
                    print(f"Warning: skipping {key}: {e}")
                    continue

                samples.append(
                    {
                        "model_name": model_name,
                        "offline_idx": offline_idx,
                        "source_path": source_path,
                        "render_paths": render_paths,
                    }
                )

    return samples


def resolve_source_path(source_root: str, offline_idx: int) -> str:
    png_path = os.path.join(source_root, f"{offline_idx}.png")
    if os.path.exists(png_path):
        return png_path

    jpg_path = os.path.join(source_root, f"{offline_idx}.jpg")
    if os.path.exists(jpg_path):
        return jpg_path

    raise FileNotFoundError(f"Missing source image for offline_idx={offline_idx}: {png_path}")


def parse_azimuth_from_filename(filename):
    import re
    match = re.search(r'azimuth[-_]?([0-9]+\.?[0-9]*)', filename)
    if match:
        raw_val = float(match.group(1))
        return raw_val
    return None


def sorted_render_paths(
        data_root: str,
        model_name: str,
        offline_idx: int,
        view_indices: Iterable[int],
) -> list[str]:
    gen_dir = os.path.join(data_root, model_name, str(offline_idx))
    if not os.path.isdir(gen_dir):
        raise FileNotFoundError(f"Missing render directory: {gen_dir}")

    render_entries: list[tuple[float, str]] = []
    for filename in os.listdir(gen_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        azimuth = parse_azimuth_from_filename(filename)
        if azimuth is None:
            raise ValueError(f"Could not parse azimuth from render filename: {filename}")
        render_entries.append((float(azimuth), os.path.join(gen_dir, filename)))

    if not render_entries:
        raise FileNotFoundError(f"No render images found in {gen_dir}")

    render_entries.sort(key=lambda item: item[0])
    render_paths = [path for _, path in render_entries]
    selected_paths: list[str] = []
    for view_idx in view_indices:
        if view_idx < 0 or view_idx >= len(render_paths):
            raise IndexError(
                f"View index {view_idx} is out of range for {gen_dir}; "
                f"found {len(render_paths)} render images."
            )
        selected_paths.append(render_paths[view_idx])

    return selected_paths


"""
torch datasets used
"""


class InferenceAdvtoys200Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, preprocess_fn):
        self.root = Path(root)
        self.preprocess = preprocess_fn
        with open(self.root / "manifest.json") as f:
            self.manifest = json.load(f)
        self.samples = []
        for battle_id, item in self.manifest.items():
            cls = item["class"]
            uid = item["object_uid"]
            clean_dir = self.root / cls / uid / "clean"
            poisoned_dir = self.root / cls / uid / "poisoned"

            clean_paths = [str(clean_dir / "rgb" / f"rgb_{a}.png") for a in [0, 90, 180, 270]]
            poisoned_paths = [str(poisoned_dir / "rgb" / f"rgb_{a}.png") for a in [0, 90, 180, 270]]

            self.samples.append({
                "battle_id": battle_id,
                "dataset": "adv_toys4k",
                "clean_paths": clean_paths,
                "poisoned_paths": poisoned_paths,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        return {
            "battle_id": item["battle_id"],
            "dataset": item["dataset"],
            "clean_images": self.preprocess(item["clean_paths"]),
            "poisoned_images": self.preprocess(item["poisoned_paths"]),
        }


@dataclass(frozen=True)
class ModelEvalSample:
    battle_id: str
    model_name: str
    offline_idx: int
    source_path: str
    render_paths: tuple[str, ...]

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)


class Inference3dgenbenchDataset(Dataset):
    """Model-level source/render sequences for inference."""

    def __init__(
            self,
            battles_json: dict,
            data_root: str,
            source_root: str,
            preprocess: Callable[[list[str]], Tensor],
            view_indices: Iterable[int] = DEFAULT_VIEW_INDICES,
    ) -> None:
        self.preprocess = preprocess
        self.samples: list[ModelEvalSample] = []
        view_indices_tuple = tuple(view_indices)

        for battle_id, battle_data in battles_json.items():
            models = battle_data["models"]
            offline_idx = int(battle_data["offline_idx"])
            source_path = resolve_source_path(source_root, offline_idx)
            for model_name in models:
                if battle_data["task"] != "image2shape":
                    continue
                try:
                    render_paths = sorted_render_paths(data_root, model_name, offline_idx, view_indices_tuple)
                except IndexError as e:
                    print(f"Warning: skipping {model_name} {offline_idx}: {e}")
                    continue
                self.samples.append(
                    ModelEvalSample(
                        battle_id=battle_id,
                        model_name=model_name,
                        offline_idx=offline_idx,
                        source_path=source_path,
                        render_paths=tuple(render_paths),
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        paths = [sample.source_path, *sample.render_paths]
        return {
            "battle_id": sample.battle_id,
            "model_name": sample.model_name,
            "original_idx": sample.offline_idx,
            "src_path": sample.source_path,
            "tgt_paths": list(sample.render_paths),
            "processed_images": self.preprocess(paths),
        }


class Training3dgenbenchDataset(Dataset):
    def __init__(
            self,
            battles_json: dict,
            data_root: str,
            source_root: str,
            preprocess: Callable[[list[str]], Tensor],
            criterion: str = "geometry",
            view_indices: Iterable[int] = DEFAULT_VIEW_INDICES,
            tie_policy: Literal["skip"] = "skip",
    ) -> None:
        self.preprocess = preprocess
        self.samples: list[dict] = []
        view_indices_tuple = tuple(view_indices)

        if tie_policy != "skip":
            raise ValueError(f"Unsupported tie_policy={tie_policy}. Only 'skip' is implemented.")

        for battle_id, battle_data in battles_json.items():
            if battle_data["task"] != "image2shape":
                continue
            models = battle_data["models"]

            if len(models) != 2:
                raise ValueError(f"Expected exactly two models for battle {battle_id}, got {len(models)}.")
            label_0 = float(battle_data[criterion]["label_0"])
            label_1 = float(battle_data[criterion]["label_1"])

            if label_0 == label_1:
                continue

            winner_idx = 0 if label_0 > label_1 else 1
            loser_idx = 1 - winner_idx
            offline_idx = int(battle_data["offline_idx"])
            source_path = resolve_source_path(source_root, offline_idx)

            winner_model = models[winner_idx]
            loser_model = models[loser_idx]
            try:
                self.samples.append(
                    {
                        "battle_id": battle_id,
                        "offline_idx": offline_idx,
                        "winner_model": winner_model,
                        "loser_model": loser_model,
                        "winner_paths": [
                            source_path,
                            *sorted_render_paths(data_root, winner_model, offline_idx, view_indices_tuple),
                        ],
                        "loser_paths": [
                            source_path,
                            *sorted_render_paths(data_root, loser_model, offline_idx, view_indices_tuple),
                        ],
                    }
                )
            except IndexError as e:
                print(f"Warning: skipping {winner_model} {loser_model} {offline_idx}: {e}")
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        return {
            "battle_id": sample["battle_id"],
            "offline_idx": sample["offline_idx"],
            "winner_model": sample["winner_model"],
            "loser_model": sample["loser_model"],
            "winner_images": self.preprocess(sample["winner_paths"]),
            "loser_images": self.preprocess(sample["loser_paths"]),
        }


def build_hi3dbench_asset_dataset(
        data_root: str,
        source_root: str,
        view_indices: tuple[int, ...],
        subset: int | None,
        seed: int,
        preprocess,
) -> Inference3dgenbenchDataset:
    """Build a per-asset dataset by scanning model directories on disk.

    No battles JSON needed — discovers all (model, prompt_idx) pairs from the
    filesystem and wraps them in a LatentAlignmentModelDataset.
    """
    # Discover models (sub-directories that aren't images_prompts_*)
    model_dirs = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d)) and not d.startswith("images_prompts")
    )
    if not model_dirs:
        raise FileNotFoundError(f"No model directories found in {data_root}")

    # Discover prompt indices — union across all models
    all_prompt_indices: set[int] = set()
    for model in model_dirs:
        model_path = os.path.join(data_root, model)
        for entry in os.listdir(model_path):
            if os.path.isdir(os.path.join(model_path, entry)):
                try:
                    all_prompt_indices.add(int(entry))
                except ValueError:
                    continue
    prompt_indices = sorted(all_prompt_indices)

    # Subset selection (deterministic given seed)
    if subset is not None and subset < len(prompt_indices):
        rng = random.Random(seed)
        prompt_indices = sorted(rng.sample(prompt_indices, subset))
        print(f"Subset: selected {len(prompt_indices)} prompts (seed={seed})")

    # Build synthetic battles_json so LatentAlignmentModelDataset can consume it
    # Each "battle" has a single model — we just need the iteration structure.
    battles_json: dict = {}
    for prompt_idx in prompt_indices:
        for model in model_dirs:
            bid = f"{model}_{prompt_idx}"
            battles_json[bid] = {
                "models": [model],
                "offline_idx": prompt_idx,
                "task": "image2shape",
            }

    dataset = Inference3dgenbenchDataset(
        battles_json=battles_json,
        data_root=data_root,
        source_root=source_root,
        preprocess=preprocess,
        view_indices=view_indices,
    )
    return dataset


def _cache_path(cache_root: str, model_name: str, offline_idx: int) -> str:
    """Return the expected path for a cached feature tensor."""
    return os.path.join(cache_root, model_name, f"{offline_idx}.pt")


class CachedTraining3dgenbenchDataset(Dataset):
    """
    Winner/loser battle rows that load pre-extracted VGGT feature tensors.

    Each ``.pt`` file stores a tensor of shape ``(num_views, 2, token_dim)``
    produced by the feature extraction step.
    """

    def __init__(
            self,
            battles_json: dict,
            cache_root: str,
            criterion: str = "geometry",
            tie_policy: Literal["skip"] = "skip",
    ) -> None:
        self.cache_root = cache_root
        self.samples: list[dict] = []

        if tie_policy != "skip":
            raise ValueError(f"Unsupported tie_policy={tie_policy}. Only 'skip' is implemented.")

        for battle_id, battle_data in battles_json.items():
            if battle_data["task"] != "image2shape":
                continue
            models = battle_data["models"]

            if len(models) != 2:
                raise ValueError(f"Expected exactly two models for battle {battle_id}, got {len(models)}.")
            label_0 = float(battle_data[criterion]["label_0"])
            label_1 = float(battle_data[criterion]["label_1"])

            if label_0 == label_1:
                continue

            winner_idx = 0 if label_0 > label_1 else 1
            loser_idx = 1 - winner_idx
            offline_idx = int(battle_data["offline_idx"])

            winner_model = models[winner_idx]
            loser_model = models[loser_idx]

            winner_cache = _cache_path(cache_root, winner_model, offline_idx)
            loser_cache = _cache_path(cache_root, loser_model, offline_idx)
            if not os.path.exists(winner_cache):
                raise FileNotFoundError(f"Missing cached features: {winner_cache}")
            if not os.path.exists(loser_cache):
                raise FileNotFoundError(f"Missing cached features: {loser_cache}")

            self.samples.append(
                {
                    "battle_id": battle_id,
                    "offline_idx": offline_idx,
                    "winner_model": winner_model,
                    "loser_model": loser_model,
                    "winner_cache": winner_cache,
                    "loser_cache": loser_cache,
                }
            )

        # Preload all feature tensors into RAM — 434MB fits trivially and
        # eliminates per-iteration torch.load disk I/O.
        unique_paths: set[str] = set()
        for s in self.samples:
            unique_paths.add(s["winner_cache"])
            unique_paths.add(s["loser_cache"])
        self._tensor_cache: dict[str, Tensor] = {
            p: torch.load(p, weights_only=True) for p in unique_paths
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        return {
            "battle_id": sample["battle_id"],
            "offline_idx": sample["offline_idx"],
            "winner_model": sample["winner_model"],
            "loser_model": sample["loser_model"],
            "winner_vectors": self._tensor_cache[sample["winner_cache"]],
            "loser_vectors": self._tensor_cache[sample["loser_cache"]],
        }


class CachedInference3dgenbenchDataset(Dataset):
    """
    Per-model dataset that loads pre-extracted VGGT feature tensors for inference.

    Each ``.pt`` file stores a tensor of shape ``(num_views, 2, token_dim)``
    produced by the feature extraction step. Unlike CachedPreferenceDataset,
    this iterates per (battle_id, model) rather than per preference pair.
    """

    def __init__(
            self,
            battles_json: dict,
            cache_root: str,
    ) -> None:
        self.cache_root = cache_root
        self.samples: list[dict] = []

        for battle_id, battle_data in battles_json.items():
            if battle_data["task"] != "image2shape":
                continue
            models = battle_data["models"]
            offline_idx = int(battle_data["offline_idx"])

            for model_name in models:
                cache_file = _cache_path(cache_root, model_name, offline_idx)
                if not os.path.exists(cache_file):
                    print(f"Warning: missing cache for {model_name}/{offline_idx}, skipping")
                    continue
                self.samples.append({
                    "battle_id": battle_id,
                    "model_name": model_name,
                    "offline_idx": offline_idx,
                    "cache_path": cache_file,
                })

        # Preload into RAM
        unique_paths: set[str] = {s["cache_path"] for s in self.samples}
        self._tensor_cache: dict[str, Tensor] = {
            p: torch.load(p, weights_only=True) for p in unique_paths
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        return {
            "battle_id": sample["battle_id"],
            "model_name": sample["model_name"],
            "offline_idx": sample["offline_idx"],
            "image_vectors": self._tensor_cache[sample["cache_path"]],
        }
