"""
M3aSURE-One implementation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from vggt.models.vggt import VGGT


def load_weights(weights_checkpoint_path: str, device: torch.device) -> dict:
    """Load a reward model checkpoint from disk."""
    if not os.path.exists(weights_checkpoint_path):
        raise FileNotFoundError(f"Reward checkpoint does not exist: {weights_checkpoint_path}")
    return torch.load(weights_checkpoint_path, map_location=device)


def load_meta(checkpoint: dict) -> dict:
    reward = checkpoint.get("reward", {})
    kwargs: dict = {}
    if "hidden_dim" in reward:
        kwargs["hidden_dim"] = reward["hidden_dim"]
    if "output_dim" in reward:
        kwargs["output_dim"] = int(reward["output_dim"])
    return kwargs


class LatentAlignmentProbe(nn.Module):
    def __init__(
            self,
            input_dim: int = 4096,
            hidden_dim: int | None = None,
            output_dim: int = 512,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            layers: list[nn.Module] = [
                nn.Linear(input_dim, output_dim)
            ]
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
        else:
            layers: list[nn.Module] = [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, pair_vectors: Tensor) -> Tensor:
        """Project concatenated vectors.

        Args:
            pair_vectors: (..., 4096) concatenated source/render features.

        Returns:
            (..., 512) L2-normalized projected vectors.
        """
        return F.normalize(self.net(pair_vectors.float()), p=2, dim=-1)


class M3aSURELatentAlignment(nn.Module):
    """Frozen-VGGT latent alignment scorer."""

    def __init__(
            self,
            vggt_model: VGGT,
            token_dim: int = 2048,
            sigma_init: float = 5.0,
            expected_views: int = 4,
            hidden_dim: int | None = None,
            output_dim: int = 512,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vggt_model = vggt_model
        self.token_dim = token_dim  # denoted as D
        self.expected_views = expected_views  # denoted as N
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.probe = LatentAlignmentProbe(
            input_dim=2 * token_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.sigma = float(sigma_init)
        self.vggt_model.eval()
        for param in self.vggt_model.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True) -> "M3aSURELatentAlignment":
        super().train(mode)
        self.vggt_model.eval()
        return self

    @jaxtyped(typechecker=beartype)
    def _extract_image_vectors(self, pair_images: Float[Tensor, "b 2 3 h w"]) -> Float[Tensor, "b 2 d"]:
        """
        Extract pooled image patch tokens from VGGT.

        Args:
            pair_images: (B, 2, 3, H, W), source at frame 0 and render at frame 1.

        Returns:
            (B, 2, 2048) pooled final patch-token vectors.
        """
        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.vggt_model.aggregator(pair_images)
            patch_tokens = aggregated_tokens_list[-1][:, :, patch_start_idx:, :]
            image_vectors = patch_tokens.mean(dim=2)  # pool across patch dimension

        if image_vectors.shape[-1] != self.token_dim:
            raise ValueError(
                f"Expected VGGT pooled token dim {self.token_dim}, got {image_vectors.shape[-1]}."
            )
        return image_vectors

    @jaxtyped(typechecker=beartype)
    def score_sequences(self, images: Float[Tensor, "b n 3 h w"]) -> dict[str, Tensor]:
        """Score source plus four generated views.

        Args:
            images: (B, N, 3, H, W) with source at index 0.

        Returns:
            Tensor dictionary containing:
                per_view_scores: (B, N) raw cosine similarities.
                per_view_scaled_scores: (B, N) sigma-scaled similarities.
                training_score: (B,) mean scaled score for Bradley-Terry.
                inference_score: (B,) median raw score for deployment.
        """
        batch_size, seq_len = images.shape[:2]
        expected_seq_len = self.expected_views + 1
        if seq_len != expected_seq_len:
            raise ValueError(f"Expected {expected_seq_len} images including source, got {seq_len}.")

        source = images[:, :1].expand(-1, self.expected_views, -1, -1, -1)  # repeat source N times
        renders = images[:, 1:]
        pair_images = torch.stack((source, renders), dim=2).reshape(
            batch_size * self.expected_views,
            2,
            *images.shape[2:],
        )

        image_vectors = self._extract_image_vectors(pair_images)
        src_vectors = image_vectors[:, 0, :]
        gen_vectors = image_vectors[:, 1, :]

        pred_vectors = torch.cat((src_vectors, gen_vectors), dim=-1)
        target_vectors = torch.cat((src_vectors, src_vectors), dim=-1)
        z_pred = self.probe(pred_vectors)
        z_target = self.probe(target_vectors)

        raw_scores = F.cosine_similarity(z_pred, z_target, dim=-1).reshape(batch_size, self.expected_views)  # (B,N)
        scaled_scores = raw_scores * self.sigma
        return {
            "per_view_scores": raw_scores,
            "per_view_scaled_scores": scaled_scores,
            "training_score": scaled_scores.mean(dim=1),
            "score": raw_scores.mean(dim=1),
        }

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        return self.score_sequences(images)

    def reward_state_dict(self) -> dict[str, object]:
        """Return only trainable reward parameters, excluding frozen VGGT."""
        return {
            "probe": self.probe.state_dict(),
            "token_dim": self.token_dim,
            "expected_views": self.expected_views,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
        }

    def load_reward_state_dict(self, state: dict[str, object]) -> None:
        """Load trainable reward parameters saved by ``reward_state_dict``."""
        token_dim = int(state["token_dim"])
        expected_views = int(state["expected_views"])
        if token_dim != self.token_dim:
            raise ValueError(f"Checkpoint token_dim={token_dim} does not match model token_dim={self.token_dim}.")
        if expected_views != self.expected_views:
            print(
                f"Checkpoint expected_views={expected_views} does not match model expected_views={self.expected_views}."
            )
        self.probe.load_state_dict(state["probe"])


class M3aSURECachedLatentAlignment(nn.Module):
    """Probe-only scorer that operates on pre-extracted VGGT features."""

    def __init__(
            self,
            token_dim: int = 2048,
            sigma_init: float = 5.0,
            expected_views: int = 4,
            hidden_dim: int | None = None,
            output_dim: int = 512,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.expected_views = expected_views
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.probe = LatentAlignmentProbe(
            input_dim=2 * token_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.sigma = float(sigma_init)

    def forward(self, image_vectors: Float[Tensor, "b n 2 d"]) -> dict[str, Tensor]:
        """Score from pre-extracted image vectors.

        Args:
            image_vectors: Pairwise collection of vectors where frame 0 is the source and frame 1 is the render for each view.
        Returns:
            Dictionary of scored sequences
        """
        batch_size, num_views = image_vectors.shape[:2]

        flat = image_vectors.reshape(batch_size * num_views, 2, self.token_dim)
        src_vectors = flat[:, 0, :]
        gen_vectors = flat[:, 1, :]

        pred_vectors = torch.cat((src_vectors, gen_vectors), dim=-1)
        target_vectors = torch.cat((src_vectors, src_vectors), dim=-1)
        z_pred = self.probe(pred_vectors)
        z_target = self.probe(target_vectors)

        raw_scores = F.cosine_similarity(z_pred, z_target, dim=-1).reshape(
            batch_size, num_views
        )
        scaled_scores = raw_scores * self.sigma
        return {
            "per_view_scores": raw_scores,
            "per_view_scaled_scores": scaled_scores,
            "score": raw_scores.mean(dim=1),
            "training_score": scaled_scores.mean(dim=1),
            "inference_score": raw_scores.median(dim=1).values,
        }

    def reward_state_dict(self) -> dict[str, object]:
        """Return trainable reward parameters."""
        return {
            "probe": self.probe.state_dict(),
            "token_dim": self.token_dim,
            "expected_views": self.expected_views,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
        }

    def load_reward_state_dict(self, state: dict[str, object]) -> None:
        """Load trainable reward parameters."""
        token_dim = int(state["token_dim"])
        expected_views = int(state["expected_views"])
        if token_dim != self.token_dim:
            raise ValueError(f"Checkpoint token_dim={token_dim} does not match model token_dim={self.token_dim}.")
        if expected_views != self.expected_views:
            raise ValueError(
                f"Checkpoint expected_views={expected_views} does not match model expected_views={self.expected_views}."
            )
        self.probe.load_state_dict(state["probe"])
