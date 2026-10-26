"""
Data adapters to run inference on 3DGenScore for each dataset.

Adapted from https://github.com/3DTopia/Hi3DEval/
"""

from pathlib import Path
import json
from typing import Any
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Inference3dgenbenchDataset(Dataset):
    def __init__(
            self,
            battles_data: dict[str, dict[str, Any]],
            hi3deval_images_dir: str,
            source_roots: tuple[str | Path, ...],
            mode: str,
            num_frames: int,
            transform: transforms.Compose,
    ) -> None:
        self.battles_data = battles_data
        self.images_dir = Path(hi3deval_images_dir)
        self.source_roots = tuple(Path(r) for r in source_roots)
        self.mode = mode
        self.num_frames = num_frames
        self.transform = transform
        self.samples = self._build_samples()

    def _build_samples(self) -> list[dict[str, Any]]:
        samples = []
        for battle_id, record in self.battles_data.items():
            task = record["task"]
            if task != "image2shape":
                continue
            offline_idx = int(record["offline_idx"])
            models = record["models"]
            if len(models) != 2:
                raise ValueError(f"{battle_id} has {len(models)} models; expected 2.")

            # Check that frames exist for both models
            try:
                frames_0 = self._frame_paths(task, models[0], offline_idx)
                frames_1 = self._frame_paths(task, models[1], offline_idx)
            except FileNotFoundError as e:
                print(f"Skipping {battle_id} because {e}")
                continue

            samples.append({
                "battle_id": battle_id,
                "task": task,
                "offline_idx": offline_idx,
                "models": models,
                "reference_type": "image",
                "reference": self._reference(offline_idx),
                "frames_0": frames_0,
                "frames_1": frames_1,
            })
        return samples

    def _reference(self, offline_idx: int) -> str | Path:
        """Find the source/reference image for a given offline_idx."""
        for source_root in self.source_roots:
            for ext in (".png", ".jpg"):
                path = source_root / f"{offline_idx}{ext}"
                if path.is_file():
                    return path
        searched = "\n".join(
            str(r / f"{offline_idx}.png") for r in self.source_roots
        )
        raise FileNotFoundError(
            f"Missing source image for offline_idx={offline_idx}.\nSearched:\n{searched}"
        )

    def _frame_paths(self, task: str, model: str, offline_idx: int) -> list[Path]:
        """Get paths to individual frame PNGs for a model/idx/mode."""
        paths = []
        for frame_idx in range(self.num_frames):
            path = self.images_dir / task / model / f"{offline_idx}_{self.mode}_{frame_idx:02d}.png"
            if not path.is_file():
                raise FileNotFoundError(f"Missing Hi3DEval frame: {path}")
            paths.append(path)
        return paths

    def _load_frames(self, paths: list[Path]) -> torch.Tensor:
        """Load and transform a list of frame paths into a tensor [T, C, H, W]."""
        frames = []
        for path in paths:
            img = Image.open(path).convert("RGB")
            frames.append(self.transform(img))
        return torch.stack(frames)  # [T, C, H, W]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        reference = Image.open(item["reference"]).convert("RGB")

        return {
            "battle_id": item["battle_id"],
            "task": item["task"],
            "offline_idx": item["offline_idx"],
            "models": item["models"],
            "reference_type": item["reference_type"],
            "reference": item["reference"],
            "frames_0": self._load_frames(item["frames_0"]),
            "frames_1": self._load_frames(item["frames_1"]),
        }


class InferenceAdvtoys200Dataset(Dataset):
    """
    Loads preprocessed 16-view frames for AdvToys200.

    Directory layout:
        root/{class}/{uid}/clean/rgb/rgb_{azimuth}.png
        root/{class}/{uid}/clean/normal/normal_{azimuth}.png
        root/{class}/{uid}/poisoned/rgb/rgb_{azimuth}.png
        root/{class}/{uid}/poisoned/normal/normal_{azimuth}.png
    """

    def __init__(self, root: str, mode: str, num_frames: int, transform: transforms.Compose) -> None:
        self.root = Path(root)
        self.mode = mode  # "rgb" or "normal"
        self.num_frames = num_frames
        self.transform = transform
        with open(self.root / "manifest.json") as f:
            self.manifest = json.load(f)
        self.samples = self._build_samples()

    def _build_samples(self) -> list[dict[str, Any]]:
        samples = []
        for battle_id, item in self.manifest.items():
            cls = item["class"]
            uid = item["object_uid"]
            azimuths = item["azimuths"]
            clean_dir = self.root / cls / uid / "clean"
            poisoned_dir = self.root / cls / uid / "poisoned"
            try:
                frames_clean = self._frame_paths(clean_dir, azimuths)
                frames_poisoned = self._frame_paths(poisoned_dir, azimuths)
            except FileNotFoundError as e:
                print(f"Skipping {battle_id} because {e}")
                continue
            samples.append({
                "dataset": "adv_toys4k",
                "battle_id": battle_id,
                "task": "image2shape",
                "offline_idx": 0,
                "models": ["clean", "poisoned"],
                "reference_type": "image",
                "reference": str(clean_dir / "rgb" / "rgb_0.png"),
                "frames_0": frames_clean,
                "frames_1": frames_poisoned,
            })
        return samples

    def _frame_paths(self, variant_dir: Path, azimuths: list[int]) -> list[Path]:
        paths = []
        subdir = variant_dir / self.mode
        for az in azimuths[:self.num_frames]:
            path = subdir / f"{self.mode}_{az}.png"
            if not path.is_file():
                raise FileNotFoundError(f"Missing frame: {path}")
            paths.append(path)
        return paths

    def _load_frames(self, paths: list[Path]) -> torch.Tensor:
        frames = []
        for path in paths:
            img = Image.open(path).convert("RGB")
            frames.append(self.transform(img))
        return torch.stack(frames)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        return {
            "dataset": "adv_toys4k",
            "battle_id": item["battle_id"],
            "task": item["task"],
            "offline_idx": item["offline_idx"],
            "models": item["models"],
            "reference_type": item["reference_type"],
            "reference": item["reference"],
            "frames_0": self._load_frames(item["frames_0"]),
            "frames_1": self._load_frames(item["frames_1"]),
        }
