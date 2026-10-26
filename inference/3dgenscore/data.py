"""
Data adapters to run inference on 3DGenScore for each dataset

Adapted from https://github.com/3DTopia/3DGen-Bench
"""

import json
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def create_square_image(frames: list[Image.Image], ncols: int = 2) -> Image.Image:
    if len(frames) % ncols != 0:
        raise ValueError(f"Expected frame count divisible by ncols, got {len(frames)}.")

    rows: list[np.ndarray] = []
    for row_start in range(0, len(frames), ncols):
        row = [np.asarray(frame) for frame in frames[row_start : row_start + ncols]]
        rows.append(np.concatenate(row, axis=1))

    con_image = np.concatenate(rows, axis=0)
    return Image.fromarray(con_image).convert("RGB")


class Inference3dgenbenchDataset(Dataset):
    def __init__(
        self,
        battles_data: dict[str, dict[str, Any]],
        con_images_dir: str,
        source_root: str,
        original_root: str,
    ) -> None:
        self.battles_data = battles_data
        self.con_images_dir = Path(con_images_dir)
        self.source_roots = (
            Path(source_root),
            Path(original_root) / "gallery" / "rgba",
            Path(original_root) / "gallery" / "images_prompts_510",
        )

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

            try:
                normal_image_0_path = self._con_image_path(
                    task, models[0], offline_idx, "normal"
                )
                normal_image_1_path = self._con_image_path(
                    task, models[1], offline_idx, "normal"
                )
                rgb_image_0_path = self._con_image_path(
                    task, models[0], offline_idx, "rgb"
                )
                rgb_image_1_path = self._con_image_path(
                    task, models[1], offline_idx, "rgb"
                )
            except FileNotFoundError as e:
                print(f"Skipping {battle_id} because {e}")
                continue

            samples.append(
                {
                    "battle_id": battle_id,
                    "task": task,
                    "offline_idx": offline_idx,
                    "models": models,
                    "reference_type": "image",
                    "reference": self._reference(offline_idx),
                    "normal_image_0": normal_image_0_path,
                    "normal_image_1": normal_image_1_path,
                    "rgb_image_0": rgb_image_0_path,
                    "rgb_image_1": rgb_image_1_path,
                }
            )
        return samples

    def _reference(self, offline_idx: int) -> str | Path:
        searched = []
        for source_root in self.source_roots:
            source_path = source_root / f"{offline_idx}.png"
            searched.append(source_path)
            if source_path.is_file():
                return source_path
            jpg_path = source_path.with_suffix(".jpg")
            searched.append(jpg_path)
            if jpg_path.is_file():
                return jpg_path

        searched_paths = "\n".join(str(path) for path in searched)
        raise FileNotFoundError(
            f"Missing source image for offline_idx={offline_idx}.\n"
            f"Searched:\n{searched_paths}"
        )

    def _con_image_path(
        self, task: str, model: str, offline_idx: int, mode: str
    ) -> Path:
        path = self.con_images_dir / task / model / f"{offline_idx}_{mode}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing 3DGenScore {mode} image: {path}")
        return path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        reference = item["reference"]
        if item["reference_type"] == "image":
            reference = Image.open(reference).convert("RGB")

        return {
            **item,
            "reference": reference,
            "normal_image_0": Image.open(item["normal_image_0"]).convert("RGB"),
            "normal_image_1": Image.open(item["normal_image_1"]).convert("RGB"),
            "rgb_image_0": Image.open(item["rgb_image_0"]).convert("RGB"),
            "rgb_image_1": Image.open(item["rgb_image_1"]).convert("RGB"),
        }


class InferenceAdvtoys200Dataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        with open(self.root / "manifest.json") as f:
            self.manifest = json.load(f)
        self.samples = []
        for battle_id, item in self.manifest.items():
            cls = item["class"]
            uid = item["object_uid"]
            clean_dir = self.root / cls / uid / "clean"
            poisoned_dir = self.root / cls / uid / "poisoned"
            self.samples.append(
                {
                    "dataset": "adv_toys4k",
                    "battle_id": battle_id,
                    "task": "image2shape",
                    "offline_idx": 0,
                    "models": ["clean", "poisoned"],
                    "reference_type": "image",
                    "reference": str(clean_dir / "rgb" / "rgb_0.png"),
                    "clean_dir": str(clean_dir),
                    "poisoned_dir": str(poisoned_dir),
                }
            )

    def __len__(self):
        return len(self.samples)

    def _load_and_stitch(self, dir_path: str, mode: str) -> Image.Image:
        dir_p = Path(dir_path) / mode
        frames = []
        for angle in [0, 90, 180, 270]:
            img_path = dir_p / f"{mode}_{angle}.png"
            with Image.open(img_path) as img:
                frames.append(img.copy().convert("RGB"))
        return create_square_image(frames, 2)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        return {
            **item,
            "reference": Image.open(item["reference"]).convert("RGB"),
            "normal_image_0": self._load_and_stitch(item["clean_dir"], "normal"),
            "normal_image_1": self._load_and_stitch(item["poisoned_dir"], "normal"),
            "rgb_image_0": self._load_and_stitch(item["clean_dir"], "rgb"),
            "rgb_image_1": self._load_and_stitch(item["poisoned_dir"], "rgb"),
        }
