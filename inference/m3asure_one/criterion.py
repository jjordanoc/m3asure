import torch
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float


def bt_loss(winner_score: Tensor, loser_score: Tensor) -> Tensor:
    return F.softplus(loser_score - winner_score).mean()


def margin_loss(winner_score: Tensor, loser_score: Tensor, margin: float = 0.1) -> Tensor:
    return F.margin_ranking_loss(winner_score, loser_score, target=torch.ones_like(winner_score), margin=margin)