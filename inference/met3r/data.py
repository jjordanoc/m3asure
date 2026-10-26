import json
import os
from collections import defaultdict
from typing import Callable
from typing import List, Optional

import numpy as np
from PIL import Image
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score, roc_auc_score

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def load_images_as_tensor_met3r(images_path: List[str], image_res: int = 256) -> torch.Tensor:
    """
    Load a list of images for MEt3R

    Args:
        images_path: List[str] of image paths
    Returns:
        [2, C, H, W] tensor normalized to [-1, 1]
    """
    tensor_list = []

    met3r_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    for img_path in images_path:
        img_pil = Image.open(img_path).convert('RGB')
        resized_img = img_pil.resize((image_res, image_res), Image.Resampling.LANCZOS)
        img_tensor = met3r_transform(resized_img)
        tensor_list.append(img_tensor)

    return torch.stack(tensor_list, dim=0)


class Inference3dgenbenchDataset():
    """
    Yields [source_img, target_img_az1, target_img_az2, target_img_az3, ...] sequences for each object, containing all azimuths.

    source_img is the source image of the object (image prompt).
    target_img_az1, target_img_az2, target_img_az3, ... are generated images of the object (novel views or rendered images) at different azimuths.
    """

    def __init__(self, battles_json, data_root, source_root, preprocess: Callable,
                 frames_to_sample: Optional[List] = None, include_source: bool = False):
        if frames_to_sample is None:
            self.frames_to_sample = list(range(12))
        else:
            self.frames_to_sample = frames_to_sample
        self.samples = []
        self.battles_json = battles_json
        self.data_root = data_root
        self.source_root = source_root
        self.preprocess = preprocess
        self._prepare_dataset()
        self._build_indices()
        self.include_source = include_source

    def _build_indices(self):
        from collections import defaultdict
        self.model_to_indices = defaultdict(list)
        self.battle_to_indices = defaultdict(list)
        for i, s in enumerate(self.samples):
            self.model_to_indices[s["model_name"]].append(i)
            self.battle_to_indices[s["battle_id"]].append(i)
        self.all_models = sorted(self.model_to_indices.keys())

    def __len__(self):
        return len(self.samples)

    def get_source_image(self, img_idx: int) -> torch.Tensor:
        source_path = os.path.join(self.source_root, f"{img_idx}.png")
        if not os.path.exists(source_path) and not os.path.exists(source_path.replace(".png", ".jpg")):
            return None
        return load_image_as_tensor_prism(source_path)

    def _prepare_dataset(self):
        print(f"Parsing {len(self.battles_json)} battles...")

        skipped_source = 0
        skipped_gen_dir = 0
        skipped_dirs = []

        for battle_id, battle_data in self.battles_json.items():
            # Extract metadata
            models = sorted(battle_data["models"])  # Sort models alphabetically
            obj_idx = battle_data["offline_idx"]  # e.g. 6
            source_path = os.path.join(self.source_root, f"{obj_idx}.png")

            # Check if source exists (optional optimization)
            if not os.path.exists(source_path) and not os.path.exists(source_path.replace(".png", ".jpg")):
                skipped_source += 1
                continue

            for model_name in models:
                gen_dir = os.path.join(self.data_root, model_name, str(obj_idx))
                if not os.path.isdir(gen_dir):
                    skipped_gen_dir += 1
                    continue

                filenames = os.listdir(gen_dir)
                filenames.sort(key=lambda x: parse_azimuth_from_filename(x))

                all_azimuth_paths = []
                az_values = []

                for filename in filenames:
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        print("Skipping file", filename, "due to invalid extension.")
                        continue

                    az = parse_azimuth_from_filename(filename)

                    target_path = os.path.join(gen_dir, filename)
                    all_azimuth_paths.append(target_path)
                    az_values.append(az)

                filtered_azimuth_paths = [all_azimuth_paths[i] for i in self.frames_to_sample]
                filtered_azimuth_values = [az_values[i] for i in self.frames_to_sample]

                # Store everything needed for the metric AND the battle logic
                self.samples.append({
                    "src_path": source_path,
                    "tgt_paths": filtered_azimuth_paths,
                    "azimuths": filtered_azimuth_values,
                    "battle_id": battle_id,
                    "model_name": model_name,
                    "original_idx": obj_idx
                })

        print(f"Prepared {len(self.samples)} lists across all battles.")
        if skipped_source or skipped_gen_dir:
            print("Ingored text-to-3D models")
            print(f"Skipped: {skipped_source} battles (missing source image), "
                  f"{skipped_gen_dir} model dirs (missing gen directory)")

    def __getitem__(self, idx):
        item = self.samples[idx]
        paths = [item["src_path"]] + item["tgt_paths"] if self.include_source else item["tgt_paths"]
        processed_images = self.preprocess(paths)
        return {
            **item,
            "processed_images": processed_images,
        }


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
