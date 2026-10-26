"""
M3aSURE-Zero implementation.

Adapted from https://github.com/mohammadasim98/met3r
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Literal, Tuple, Optional, Dict, Any, Union
from torch import Tensor
from torch.utils.data import Dataset

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import project_world_points_to_cam
from vggt.models.vggt import VGGT

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
)
from helpers import to_bool_mask, normalize_imagenet
from jaxtyping import Float


def render_features_met3r(
        features_i: Tensor,
        features_j: Tensor,
        world_points_i: Tensor,
        world_points_j: Tensor,
        extrinsics: Tensor,
        intrinsics: Tensor,
        image_h: int,
        image_w: int,
        fg_mask_i: Optional[Tensor] = None,
        fg_mask_j: Optional[Tensor] = None,
        points_per_pixel: int = 10,
        bin_size: int = 0,
        radius: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Render both views' features through a canonical camera.

    MEt3R's rasterization approach: each view's 3-D points carry
    their features, both are rendered from a shared canonical camera via
    PyTorch3D ``PointsRasterizer`` + ``AlphaCompositor``.

    A mask is obtained from mutually visible regions.

    Args:
        features_i:      (C, fH, fW) feature map of view i.
        features_j:      (C, fH, fW) feature map of view j.
        world_points_i:  (H, W, 3)  world-space 3-D points of view i.
        world_points_j:  (H, W, 3)  world-space 3-D points of view j.
        extrinsics:      (3, 4)  extrinsics of the canonical camera in OpenCV convention [R|t].
        intrinsics:      (3, 3)  intrinsics of the canonical camera.
        image_h / image_w: rendering resolution.
        fg_mask_i:       (H, W) bool foreground mask for view i.
        fg_mask_j:       (H, W) bool foreground mask for view j.
        points_per_pixel: rasterizer k (default 10, same as MEt3R).
        bin_size:        rasterizer bin_size (0 = brute-force).
        radius:          point splatting radius in NDC (default 0.01).

    Returns:
        rendered_i:   (C, rH, rW) rendered features from view i's points.
        rendered_j:   (C, rH, rW) rendered features from view j's points.
        overlap_mask: (rH, rW) bool, True where both views rendered content.
    """
    device = features_i.device
    H, W = world_points_i.shape[:2]
    C = features_i.shape[0]

    R = extrinsics[:3, :3].float()  # (3, 3)
    t = extrinsics[:3, 3].float()  # (3,)

    pts_i = world_points_i.reshape(-1, 3).float()  # (HW, 3)
    pts_j = world_points_j.reshape(-1, 3).float()  # (HW, 3)

    pts_i_cam = pts_i @ R.T + t  # (HW, 3)
    pts_j_cam = pts_j @ R.T + t  # (HW, 3)

    feat_i_flat = features_i.permute(1, 2, 0).reshape(-1, C).float()
    feat_j_flat = features_j.permute(1, 2, 0).reshape(-1, C).float()

    if fg_mask_i is not None:
        keep_i = fg_mask_i.reshape(-1).bool()
        pts_i_cam = pts_i_cam[keep_i]
        feat_i_flat = feat_i_flat[keep_i]
    if fg_mask_j is not None:
        keep_j = fg_mask_j.reshape(-1).bool()
        pts_j_cam = pts_j_cam[keep_j]
        feat_j_flat = feat_j_flat[keep_j]

    # attach features to pointcloud
    point_cloud = Pointclouds(
        points=[pts_i_cam, pts_j_cam],
        features=[feat_i_flat, feat_j_flat],
    )

    fx = intrinsics[0, 0].float()
    fy = intrinsics[1, 1].float()

    focal_ndc = torch.tensor(
        [[1.0 + fx / image_w, 1.0 + fy / image_h]],
        device=device, dtype=torch.float32,
    ).expand(2, -1)  # (2, 2)

    R_flip = torch.eye(3, device=device, dtype=torch.float32)
    R_flip[0, 0] = -1.0
    R_flip[1, 1] = -1.0
    R_flip = R_flip.unsqueeze(0).expand(2, -1, -1)  # (2, 3, 3)

    T_zero = torch.zeros(2, 3, device=device, dtype=torch.float32)

    cameras = PerspectiveCameras(
        device=device, R=R_flip, T=T_zero, focal_length=focal_ndc,
    )

    raster_settings = PointsRasterizationSettings(
        image_size=(image_h, image_w),
        radius=radius,
        points_per_pixel=points_per_pixel,
        bin_size=bin_size,
    )
    rasterizer = PointsRasterizer(
        cameras=cameras, raster_settings=raster_settings,
    )
    compositor = AlphaCompositor()

    BACKGROUND = -10000.0

    with torch.autocast("cuda", enabled=False):
        fragments = rasterizer(point_cloud)

    r = raster_settings.radius
    dists2 = fragments.dists.permute(0, 3, 1, 2)
    weights = 1.0 - dists2 / (r * r)

    rendering = compositor(
        fragments.idx.long().permute(0, 3, 1, 2),
        weights,
        point_cloud.features_packed().permute(1, 0),
        background_color=[BACKGROUND] * C,
    )  # (2, C, rH, rW)

    non_overlap = (rendering == BACKGROUND)
    overlap_mask = (
        (1.0 - non_overlap.float()).prod(dim=1).prod(dim=0).bool()
    )
    rendering[non_overlap] = 0.0
    rendered_i = rendering[0]  # (C, rH, rW)
    rendered_j = rendering[1]  # (C, rH, rW)

    return rendered_i, rendered_j, overlap_mask


def cosine_similarity_map(
        feat_a: Tensor,
        feat_b: Tensor,
        eps: float = 1e-8,
) -> Tensor:
    a = F.normalize(feat_a.float(), dim=0, eps=eps)
    b = F.normalize(feat_b.float(), dim=0, eps=eps)
    return (a * b).sum(dim=0)


def aggregate_score(similarity_map: Tensor, mask: Tensor) -> Tensor:
    """mean of similarity_map over pixels where mask is True."""
    m = mask.float()
    total = m.sum()
    if total == 0:
        return torch.tensor(0.0, device=similarity_map.device)
    return (similarity_map * m).sum() / total


class M3aSURESemanticProjector(nn.Module):
    def __init__(
            self,
            vggt_model: VGGT,
            anyup_model: nn.Module,
            patch_size: int = 14,
            q_chunk_size: Optional[int] = None,
    ):
        super().__init__()
        self.vggt = vggt_model
        self.anyup = anyup_model
        self.patch_size = patch_size
        self.q_chunk_size = q_chunk_size

        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        self.dinov2.eval()
        for p in self.dinov2.parameters():
            p.requires_grad_(False)

    def _upsample_features(
            self,
            agg_features: Float[Tensor, "n c hp wp"],
            images: Float[Tensor, "n 3 h w"],
    ) -> Float["n c h w"]:
        """
        Upscale features to image size
        Args:
            agg_features: low-res patch features.
            images: original images in [0, 1].
        Returns:
            High-resolution features.
        """
        N = agg_features.shape[0]

        images_norm = normalize_imagenet(images)
        hr_list = []
        # TODO: parallelize (VRAM issues, set up natten)
        for i in range(N):
            hr_i = self.anyup(
                images_norm[i: i + 1].float(),
                agg_features[i: i + 1].float(),
                output_size=images.shape[-2:],
                q_chunk_size=self.q_chunk_size,
            )
            hr_list.append(hr_i)
        return torch.cat(hr_list, dim=0)

    def _extract_dinov2_features(
            self,
            images: Float[Tensor, "n 3 h w"],
    ) -> Float[Tensor, "n c hp wp"]:
        images_norm = normalize_imagenet(images)

        features = self.dinov2.forward_features(images_norm.float())
        patch_tokens = features["x_norm_patchtokens"]  # (N, N_patches, C)

        N, _, C = patch_tokens.shape
        H_p = images.shape[2] // self.patch_size
        W_p = images.shape[3] // self.patch_size

        spatial = (
            patch_tokens
            .reshape(N, H_p, W_p, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return spatial

    @torch.no_grad()
    def forward(
            self,
            images: Float[Tensor, "b n 3 h w"],
            mask_background: bool = True,
            return_details: bool = False,
            return_features: bool = False,
    ) -> Dict[str, Any]:
        """Compute M3aSURE for a sequence of images.

        Args:
            images: input target views in [0, 1].
            mask_background: if True, automatically detect and mask background.
            return_details: for debugging, keep per-pair similarity maps and masks.
            return_features: for visualizing, keep high-resolution and warped feature maps (can consume a lot of ram).
        Returns:
            dict with keys:
                score             – float, mean cosine similarity over all N views
                per_pair_scores   – List[float].
                pairs             – List[Tuple[int,int]].
                sim_maps          – List[Tensor]  (if return_details).
                masks             – List[Tensor]  (if return_details).
                predictions       – Dict (VGGT outputs, CPU).
                hr_features       – Tensor (if return_details).
        """
        B, N, C, H, W = images.shape
        assert B == 1, "Batch size > 1 not yet supported"

        if mask_background:
            fg_masks = torch.stack([
                to_bool_mask(images[0, s], inverse=True) for s in range(N)
            ]).unsqueeze(0).to(images.device)  # (1, N, H, W)

        predictions = self.vggt(images)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            predictions["pose_enc"].float(), image_size_hw=(H, W)
        )
        images_flat = images.reshape(B * N, C, H, W)

        agg_features = self._extract_dinov2_features(images_flat)

        hr_features = self._upsample_features(agg_features, images_flat)  # (B*N, feat_dim, feat_H, feat_W)
        del agg_features  # save ram

        feat_H, feat_W = hr_features.shape[-2:]
        feat_dim = hr_features.shape[1]
        hr_features = hr_features.reshape(B, N, feat_dim, feat_H, feat_W)

        # pairwise projection
        eval_pairs = sorted(set((0, i) for i in range(1, N)))
        proj_mode = projection_mode or self.projection_mode
        per_pair_scores: List[float] = []
        sim_maps: List[Tensor] = []
        masks_out: List[Tensor] = []
        warped_features_out: List[Tensor] = []

        for i, j in eval_pairs:
            feat_i = hr_features[0, i]  # (C, Hp, Wp)
            feat_j = hr_features[0, j]  # (C, Hp, Wp)
            wp_i = predictions["world_points"][0, i]  # (H, W, 3)

            fg_mask_i = fg_masks[0, i]
            fg_mask_j = fg_masks[0, j]
            wp_j = predictions["world_points"][0, j]  # (H, W, 3)

            rendered_i, rendered_j, overlap_mask = render_features_canonical(
                features_i=feat_i,
                features_j=feat_j,
                world_points_i=wp_i,
                world_points_j=wp_j,
                extrinsics=extrinsics[0, i],
                intrinsics=intrinsics[0, i],
                image_h=H,
                image_w=W,
                fg_mask_i=fg_mask_i,
                fg_mask_j=fg_mask_j,
            )

            sim_map = cosine_similarity_map(rendered_i, rendered_j)
            combined_mask = overlap_mask
            score = aggregate_score(sim_map, combined_mask)

            per_pair_scores.append(score.item())
            if return_details:
                sim_maps.append(sim_map.detach().cpu())
                masks_out.append(combined_mask.detach().cpu())
                if return_features:
                    warped_features_out.append(rendered_j.detach().cpu())

        avg_score = (
            sum(per_pair_scores) / len(per_pair_scores)
            if per_pair_scores
            else 0.0
        )
        result: Dict[str, Any] = {
            "score": avg_score,
            "per_pair_scores": per_pair_scores,
            "pairs": eval_pairs,
        }

        if return_details:
            result["sim_maps"] = sim_maps
            result["masks"] = masks_out
            if return_features:
                result["warped_features"] = warped_features_out
                result["hr_features"] = hr_features.detach()
            result["predictions"] = {
                k: v.detach() if isinstance(v, Tensor) else v
                for k, v in predictions.items()
                if k != "images"
            }
        return result
