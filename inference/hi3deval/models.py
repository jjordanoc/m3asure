from pathlib import Path

import sys

M3ASURE_ROOT = Path("/workspace/m3asure")
sys.path.insert(0, str(M3ASURE_ROOT))

import os
from typing import Any
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from omegaconf import OmegaConf

# Hi3DEval evaluation dimensions we care about
EVAL_DIMS = ("geometry", "alignment")

# Metric configs: each metric uses different renders, frame counts, and checkpoints
METRIC_CONFIGS = {
    "geometry": {
        "mode": "normal",
        "num_frames": 16,
        "qformer_ckpt": "trained/Geo_plausibility/mp_rank_00_model_states.pt",
        "is_prompt_alignment": False,
        "is_prompt_alignment_cosine_mlp": False,
    },
    "alignment": {
        "mode": "rgb",
        "num_frames": 16,
        "qformer_ckpt": "trained/Prompt_alignment_cos/mp_rank_00_model_states.pt",
        "is_prompt_alignment": True,
        "is_prompt_alignment_cosine_mlp": True,
    },
}

IMAGE_RES = 448


def get_hi3deval_transform() -> transforms.Compose:
    """Build the image transform matching Hi3DEval's test transform."""
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    return transforms.Compose([
        transforms.Resize(
            (IMAGE_RES, IMAGE_RES),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_hi3deval_config(
        ckpt_root: str,
        qformer_ckpt: str,
        num_frames: int,
        is_prompt_alignment_cosine_mlp: bool = False,
) -> Any:
    """Build a config object mimicking Hi3DEval's config.py structure."""

    config = OmegaConf.create({
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "train_stage": 2,
        "train_full": False,
        "use_half_precision": True,
        "use_bf16": True,
        "auto_resume2": False,
        "is_prompt_alignment": is_prompt_alignment_cosine_mlp,
        "is_prompt_alignment_cosine_mlp": is_prompt_alignment_cosine_mlp,
        "model": {
            "model_cls": "InternVideo2_CLIP",
            "vision_encoder": {
                "name": "InternViT",
                "config": os.path.join(ckpt_root, "configs", "encoder"),
                "ckpt": os.path.join(ckpt_root, "ckpts", "mp_rank_00_model_states.pt"),
            },
            "clip_encoder": {
                "name": "ViT-H-14",
                "ckpt": os.path.join(ckpt_root, "ckpts", "CLIP-ViT-H-14", "open_clip_model.safetensors"),
            },
            "qformer": {
                "config": os.path.join(ckpt_root, "configs", "qformer"),
                "ckpt": os.path.join(ckpt_root, qformer_ckpt),
            },
            "tokenizer": "ViT-H-14",
            "temp": 1 / 100.0,
            "temp_min": 1 / 100.0,
            "freeze_clip": True,
        },
        "criterion": {
            "loss_weight": {
                "loss_format": 0.2,
                "loss_prompt": 0.8,
                "loss_score": 1.0,
            },
            "loss_type": "mae",
        },
        "inputs": {
            "image_res": IMAGE_RES,
            "video_input": {
                "num_frames": num_frames,
                "num_frames_test": num_frames,
                "sample_type": "rand",
                "sample_type_test": "middle",
                "random_aug": False,
            },
        },
    })
    return config


def load_hi3deval_model(
        ckpt_root: str,
        qformer_ckpt: str,
        num_frames: int,
        is_prompt_alignment_cosine_mlp: bool,
        device: str,
) -> tuple[Any, Any]:
    """Load an InternVideo2_CLIP model with the specified QFormer checkpoint."""
    config = build_hi3deval_config(
        ckpt_root=ckpt_root,
        qformer_ckpt=qformer_ckpt,
        num_frames=num_frames,
        is_prompt_alignment_cosine_mlp=is_prompt_alignment_cosine_mlp,
    )
    try:
        from models import InternVideo2_CLIP
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Make sure to clone the Hi3DEval repo https://github.com/3DTopia/Hi3DEval/tree/main \nModify the paths in the inf_*.py configs accordingly. ")

    model = InternVideo2_CLIP(config=config, is_pretrain=False)
    model = model.to(torch.device(device))
    model = model.to(torch.bfloat16)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    print(f"Loaded Hi3DEval model with QFormer: {qformer_ckpt}")
    return model, config


@torch.no_grad()
def score_single_model(
        model: Any,
        frames: torch.Tensor,
        reference: str | Path,
        is_text_prompt: bool,
        is_prompt_alignment: bool,
        is_prompt_alignment_cosine_mlp: bool,
        device: str,
) -> float:
    """Score a single model's renders using Hi3DEval.

    Args:
        model: InternVideo2_CLIP model
        frames: [T, C, H, W] tensor of rendered frames (already transformed)
        reference: path to the reference/prompt image, or text string
        is_text_prompt: whether reference is text (True) or image path (False)
        is_prompt_alignment: whether this is prompt alignment scoring
        is_prompt_alignment_cosine_mlp: whether to use cosine MLP for prompt alignment
        device: torch device string

    Returns:
        Scalar quality score
    """
    # frames: [T, C, H, W] -> [1, 1, T, C, H, W] (batch B=1, crops N=1)
    image = frames.unsqueeze(0).unsqueeze(0).to(device)

    # Prepare prompt
    if is_text_prompt:
        prompt = [str(reference)]
        is_text_prompt_tensor = torch.tensor([True])
    else:
        prompt = [str(reference)]
        is_text_prompt_tensor = torch.tensor([False])

    # Tokenize text (empty text for scoring)
    text_input = model.tokenizer(["quality"]).to(device)

    # Create dummy idx
    idx = torch.tensor([0]).to(device)

    output = model.generate(
        image,
        text_input,
        idx=idx,
        prompt=prompt,
        is_text_prompt=is_text_prompt_tensor,
        is_prompt_alignment=is_prompt_alignment,
    )

    return float(output.squeeze().cpu())
