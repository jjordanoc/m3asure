"""
Train the learned M3aSURE latent alignment reward.

Expects pre-extracted VGGT features produced by ``extract_features.py`` in the cache directory.

Usage:
    uve run inference/extract_features.py --batch-size 4
    uve run inference/train.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from inference.inference_helpers import notify, autocast_context, device_dtype

from criterion import bt_loss, margin_loss
from data import CachedTraining3dgenbenchDataset, load_json
from models import M3aSURECachedLatentAlignment


@torch.no_grad()
def _compute_preference_metrics(
        probe: M3aSURECachedLatentAlignment,
        loader: DataLoader,
        device: torch.device,
        dtype: torch.dtype,
        description: str,
        pooling_type: str = "mean",
) -> dict[str, float]:
    """Compute Acc and AUC over preference pairs (ties already excluded by the dataset)."""
    probe.eval()
    all_winner_scores: list[float] = []
    all_loser_scores: list[float] = []

    for batch in tqdm(loader, desc=description, leave=False):
        winner_vectors = batch["winner_vectors"].to(device, non_blocking=True)
        loser_vectors = batch["loser_vectors"].to(device, non_blocking=True)

        with autocast_context(device, dtype):
            if pooling_type == "min":
                w_score = probe(winner_vectors)["per_view_scores"].min(dim=1).values
                l_score = probe(loser_vectors)["per_view_scores"].min(dim=1).values
            else:
                w_score = probe(winner_vectors)["score"]
                l_score = probe(loser_vectors)["score"]

        all_winner_scores.extend(w_score.cpu().tolist())
        all_loser_scores.extend(l_score.cpu().tolist())

    winner = np.asarray(all_winner_scores)
    loser = np.asarray(all_loser_scores)
    score_diff = winner - loser

    acc = float(np.mean(score_diff > 0))

    # mirror every pair: (label=1, d) + (label=0, -d) so both classes exist
    labels = np.concatenate([np.ones(len(score_diff), dtype=int),
                             np.zeros(len(score_diff), dtype=int)])
    scores = np.concatenate([score_diff, -score_diff])
    auc = float(roc_auc_score(labels, scores))
    return {"acc": acc, "auc": auc}


def _run_epoch(
        probe: M3aSURECachedLatentAlignment,
        loader: DataLoader,
        device: torch.device,
        dtype: torch.dtype,
        optimizer: torch.optim.Optimizer,
        description: str,
        grad_clip: float = 0.0,
        loss_type: str = "bt",
        pooling_type: str = "mean",
        margin: float = 0.1,
) -> float:
    is_training = optimizer is not None
    probe.train(is_training)
    loss_sum = 0.0
    sample_count = 0

    for batch in tqdm(loader, desc=description, leave=False):
        winner_vectors = batch["winner_vectors"].to(device, non_blocking=True)
        loser_vectors = batch["loser_vectors"].to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training), autocast_context(device, dtype):
            if loss_type == "bt":
                winner_score = probe(winner_vectors)["training_score"]
                loser_score = probe(loser_vectors)["training_score"]
                loss = bt_loss(winner_score, loser_score)
            elif loss_type == "margin":
                if pooling_type == "min":
                    winner_score = probe(winner_vectors)["per_view_scores"].min(dim=1).values
                    loser_score = probe(loser_vectors)["per_view_scores"].min(dim=1).values
                else:
                    winner_score = probe(winner_vectors)["score"]
                    loser_score = probe(loser_vectors)["score"]
                loss = margin_loss(winner_score, loser_score, margin)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

        if is_training:
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=grad_clip)
            optimizer.step()

        batch_size = int(winner_vectors.shape[0])
        loss_sum += float(loss.detach().cpu()) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError(f"No samples were available for {description}.")
    return loss_sum / sample_count


def _save_checkpoint(
        probe: M3aSURECachedLatentAlignment,
        optimizer: torch.optim.Optimizer,
        path: str,
        epoch: int,
        train_loss: float,
        valid_loss: float,
        config: dict,
) -> None:
    tmp_path = path + ".tmp"
    torch.save(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "reward": probe.reward_state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def _save_hyperparams(output_dir: str, config: dict, filename: str = "hyperparams.json") -> None:
    """Save hyperparameters to a JSON file in the output directory."""
    path = os.path.join(output_dir, filename)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(config, f, indent=2)
    os.replace(tmp_path, path)


def main() -> None:
    DATA_ROOT = "/workspace/m3asure/data/assets/3dgenbench_processed_12_views"
    DEFAULT_CACHE_ROOT = "/workspace/m3asure/data/assets/3dgenbench_vggt_pooled_embeddings"
    DEFAULT_VALID_JSON = os.path.join(DATA_ROOT, "valid.json")
    DEFAULT_TRAIN_JSON = os.path.join(DATA_ROOT, "train.json")
    DEFAULT_OUTPUT_DIR = f"/workspace/m3asure/results/probe_checkpoints/m3asure_one_train"

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=str, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--valid-json", type=str, default=DEFAULT_VALID_JSON)
    parser.add_argument("--cache-root", type=str, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--criterion", type=str, default="geometry")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--sigma-init", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout probability in probe MLP (0 = disabled)")
    parser.add_argument("--hidden-dim", type=str, default="none",
                        help="Hidden dimension of the probe MLP. "
                             "Set to 'none' or '0' for a linear probe.")
    parser.add_argument("--output-dim", type=int, default=512,
                        help="Output dimension of the probe MLP")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = disabled)")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["none", "cosine"],
                        help="LR scheduler: none, cosine")
    parser.add_argument("--loss-type", type=str, default="bt", choices=["bt", "margin"],
                        help="Loss type to use: bt (Bradley-Terry) or margin")
    parser.add_argument("--margin", type=float, default=0.1,
                        help="Margin for margin loss")
    parser.add_argument("--pooling", type=str, default="mean")

    args = parser.parse_args()

    output_dir = args.output_dir
    _hd = args.hidden_dim.strip().lower()
    args.hidden_dim = None if _hd in ("none", "0") else int(_hd)
    if args.run_id:
        output_dir += f"_{args.run_id}"
    os.makedirs(output_dir, exist_ok=True)
    config = vars(args)

    _save_hyperparams(output_dir, config, "hyperparams.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = device_dtype(device)
    print(f"device: {device}")
    print(f"autocast dtype: {dtype}")
    print(f"hyperparams: {json.dumps(config, indent=2)}")

    train_data = load_json(args.train_json)
    valid_data = load_json(args.valid_json)

    train_dataset = CachedTraining3dgenbenchDataset(
        battles_json=train_data,
        cache_root=args.cache_root,
        criterion=args.criterion,
    )
    valid_dataset = CachedTraining3dgenbenchDataset(
        battles_json=valid_data,
        cache_root=args.cache_root,
        criterion=args.criterion,
    )
    print(f"train battles after tie filtering: {len(train_dataset)}")
    print(f"valid battles after tie filtering: {len(valid_dataset)}")

    use_workers = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=use_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=use_workers,
    )

    probe = M3aSURECachedLatentAlignment(
        sigma_init=args.sigma_init,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
        expected_views=12,
    ).to(device)

    trainable_params = [param for param in probe.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_valid_loss = float("inf")
    best_valid_acc = 0.0
    best_valid_auc = 0.0
    epochs_without_improvement = 0
    best_loss_path = os.path.join(output_dir, "latent_alignment_best_loss.pt")
    best_acc_path = os.path.join(output_dir, "latent_alignment_best_acc.pt")
    best_auc_path = os.path.join(output_dir, "latent_alignment_best_auc.pt")
    latest_path = os.path.join(output_dir, "latent_alignment_latest.pt")

    train_losses: list[float] = []
    valid_losses: list[float] = []
    train_accs: list[float] = []
    train_aucs: list[float] = []
    valid_accs: list[float] = []
    valid_aucs: list[float] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(
            probe=probe,
            loader=train_loader,
            device=device,
            dtype=dtype,
            optimizer=optimizer,
            description=f"train epoch {epoch}",
            grad_clip=args.grad_clip,
            loss_type=args.loss_type,
            pooling_type=args.pooling,
            margin=args.margin,
        )
        valid_loss = _run_epoch(
            probe=probe,
            loader=valid_loader,
            device=device,
            dtype=dtype,
            optimizer=None,
            description=f"valid epoch {epoch}",
            loss_type=args.loss_type,
            pooling_type=args.pooling,
            margin=args.margin,
        )

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_metrics = _compute_preference_metrics(probe, train_loader, device, dtype, f"train metrics {epoch}",
                                                    pooling_type=args.pooling)
        val_metrics = _compute_preference_metrics(probe, valid_loader, device, dtype, f"val metrics {epoch}",
                                                  pooling_type=args.pooling)
        train_accs.append(train_metrics["acc"])
        train_aucs.append(train_metrics["auc"])
        valid_accs.append(val_metrics["acc"])
        valid_aucs.append(val_metrics["auc"])

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        _save_checkpoint(probe, optimizer, latest_path, epoch, train_loss, valid_loss, config)
        sigma_value = probe.sigma
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"valid_loss={valid_loss:.6f} sigma={sigma_value:.6f} lr={current_lr:.2e} "
            f"train_acc={train_metrics['acc']:.3f} train_auc={train_metrics['auc']:.3f} "
            f"val_acc={val_metrics['acc']:.3f} val_auc={val_metrics['auc']:.3f}"
        )

        improved = False

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            _save_checkpoint(probe, optimizer, best_loss_path, epoch, train_loss, valid_loss, config)
            notify(f"best loss checkpoint: epoch={epoch} valid_loss={valid_loss:.6f}")

        if val_metrics["acc"] > best_valid_acc:
            best_valid_acc = val_metrics["acc"]
            _save_checkpoint(probe, optimizer, best_acc_path, epoch, train_loss, valid_loss, config)
            notify(f"best acc checkpoint: epoch={epoch} val_acc={val_metrics['acc']:.3f}")
            improved = True

        if val_metrics["auc"] > best_valid_auc:
            best_valid_auc = val_metrics["auc"]
            _save_checkpoint(probe, optimizer, best_auc_path, epoch, train_loss, valid_loss, config)
            notify(f"best auc checkpoint: epoch={epoch} val_auc={val_metrics['auc']:.3f}")
            improved = True

        if improved:
            epochs_without_improvement = 0
            best_info = {
                **config,
                "best_epoch_loss": epoch if valid_loss <= best_valid_loss else None,
                "best_valid_loss": best_valid_loss,
                "best_epoch_acc": epoch if val_metrics["acc"] >= best_valid_acc else None,
                "best_valid_acc": best_valid_acc,
                "best_epoch_auc": epoch if val_metrics["auc"] >= best_valid_auc else None,
                "best_valid_auc": best_valid_auc,
                "best_sigma": sigma_value,
            }
            _save_hyperparams(output_dir, best_info, "best_hyperparams.json")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                notify(
                    f"early stopping at epoch={epoch}; "
                    f"best_valid_loss={best_valid_loss:.6f}"
                )
                break

    epochs_range = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # loss curves
    axes[0].plot(epochs_range, train_losses, label="Train", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs_range, valid_losses, label="Valid", color="#ff7f0e", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=13)
    axes[0].set_ylabel("Loss", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].tick_params(labelsize=11)

    # acc curves
    axes[1].plot(epochs_range, train_accs, label="Train", color="#1f77b4", linewidth=2)
    axes[1].plot(epochs_range, valid_accs, label="Valid", color="#ff7f0e", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=13)
    axes[1].set_ylabel("Acc.", fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].tick_params(labelsize=11)

    # auc curves
    axes[2].plot(epochs_range, train_aucs, label="Train", color="#1f77b4", linewidth=2)
    axes[2].plot(epochs_range, valid_aucs, label="Valid", color="#ff7f0e", linewidth=2)
    axes[2].set_xlabel("Epoch", fontsize=13)
    axes[2].set_ylabel("AUC", fontsize=13)
    axes[2].legend(fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].tick_params(labelsize=11)

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(plot_path, format="png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
