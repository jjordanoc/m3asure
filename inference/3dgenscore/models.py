"""
Inference adapters for 3DGenScore.

Adapted from https://github.com/3DTopia/3DGen-Bench
"""

import os
import sys
from typing import Any
import importlib
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoProcessor

EVAL_DIMS = (
    "geometry",
    "alignment",
)  # omitting "geo_detail", "texture", "geo_texture",


def instantiate_with_cfg(cfg: Any, **kwargs: Any) -> Any:
    module_path, attr = cfg._target_.rsplit(".", 1)
    module = importlib.import_module(module_path)
    target = getattr(module, attr)
    return target(cfg, **kwargs)


def load_3dgenscore_model(checkpoint_dir: str, device: str) -> torch.nn.Module:
    cfg_path = os.path.join(checkpoint_dir, "config.yaml")
    ckpt_path = os.path.join(checkpoint_dir, "model.pkl")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing 3DGenScore config: {cfg_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing 3DGenScore checkpoint: {ckpt_path}")

    cfg = OmegaConf.load(cfg_path)
    model = instantiate_with_cfg(cfg.model)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(f"Loaded 3DGenScore checkpoint from {ckpt_path}")
    return model


@torch.no_grad()
def calc_probs(
    model: torch.nn.Module,
    processor: AutoProcessor,
    device: str,
    reference: str | Image.Image,
    normal_image_0: Image.Image,
    normal_image_1: Image.Image,
    rgb_image_0: Image.Image,
    rgb_image_1: Image.Image,
) -> list[list[float]]:
    reference_inputs = processor(
        images=reference,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    reference_outputs = model.get_image_features(**reference_inputs)
    (reference_features,) = reference_outputs
    if not isinstance(reference_features, torch.Tensor):
        vision_outputs = model.clip.vision_model(**reference_inputs)
        reference_features = model.clip.visual_projection(vision_outputs[1])
    normal_inputs = processor(
        images=[normal_image_0, normal_image_1],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    rgb_inputs = processor(
        images=[rgb_image_0, rgb_image_1],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    (normal_features,) = model.get_multi_view_normal_features(**normal_inputs)
    (rgb_features,) = model.get_multi_view_rgb_features(**rgb_inputs)
    normal_0_features, normal_1_features = normal_features.chunk(2, dim=0)
    rgb_0_features, rgb_1_features = rgb_features.chunk(2, dim=0)

    geo_0 = model.geo_logit_scale.exp() * torch.diag(
        torch.einsum("bd,cd->bc", reference_features, normal_0_features)
    )
    geo_1 = model.geo_logit_scale.exp() * torch.diag(
        torch.einsum("bd,cd->bc", reference_features, normal_1_features)
    )
    geo_detail_0 = model.geo_detail_logit_proj(normal_0_features).squeeze(1)
    geo_detail_1 = model.geo_detail_logit_proj(normal_1_features).squeeze(1)
    texture_0 = model.texture_logit_proj(rgb_0_features).squeeze(1)
    texture_1 = model.texture_logit_proj(rgb_1_features).squeeze(1)
    geo_texture_0 = model.geo_texture_logit_scale.exp() * torch.diag(
        torch.einsum("bd,cd->bc", normal_0_features, rgb_0_features)
    )
    geo_texture_1 = model.geo_texture_logit_scale.exp() * torch.diag(
        torch.einsum("bd,cd->bc", normal_1_features, rgb_1_features)
    )
    align_0 = model.align_logit_scale.exp() * torch.diag(
        torch.einsum("bd,cd->bc", reference_features, rgb_0_features)
    )
    align_1 = model.align_logit_scale.exp() * torch.diag(
        torch.einsum("bd,cd->bc", reference_features, rgb_1_features)
    )

    scores_dict_0 = {
        "geometry": geo_0,
        "geo_detail": geo_detail_0,
        "texture": texture_0,
        "geo_texture": geo_texture_0,
        "alignment": align_0,
    }
    scores_dict_1 = {
        "geometry": geo_1,
        "geo_detail": geo_detail_1,
        "texture": texture_1,
        "geo_texture": geo_texture_1,
        "alignment": align_1,
    }

    image_0_scores = torch.cat([scores_dict_0[k] for k in EVAL_DIMS], dim=0)
    image_1_scores = torch.cat([scores_dict_1[k] for k in EVAL_DIMS], dim=0)

    scores = torch.stack([image_0_scores, image_1_scores], dim=-1)
    return {
        "raw": scores.cpu().tolist(),
        "probs": torch.softmax(scores, dim=-1).cpu().tolist(),
    }
