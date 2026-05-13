"""
train.py
--------
Training of CNN architectures on ERSP spectrograms from BCI-IV-2b.

Usage:
    python src/train.py --model spectnet
    python src/train.py --model eegnet --subjects 1 2 3
    python src/train.py --model shallowconvnet --all_subjects
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RESULTS_DIR, FIGURES_DIR, METRICS_DIR, SUBJECTS,
    N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS,
    BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, VAL_SPLIT, RANDOM_SEED, DEVICE
)
from dataset import build_loaders
from models import get_model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


def train_model(model_name: str,
                subjects: list = None,
                device_str: str = DEVICE) -> dict:
    """
    Train a CNN model on training data (sessions 1-3)
    and evaluate on the test set (sessions 4-5).

    Returns a dict with training history and final results.
    """
    if subjects is None:
        subjects = SUBJECTS

    device = torch.device(device_str)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"\n══════════════════════════════════════════════")
    print(f"  Training: {model_name.upper()} | Subjects: {subjects}")
    print(f"  Device: {device} | Max epochs: {MAX_EPOCHS}")
    print(f"══════════════════════════════════════════════")

    # ── Data ──
    train_loader, val_loader, test_loader = build_loaders(subjects=subjects)

    # ── Model ──
    model = get_model(
        model_name,
        n_channels=N_CHANNELS,
        n_freq=IMG_FREQ_BINS,
        n_time=IMG_TIME_BINS,
        n_classes=2
    ).to(device)
    n_params = model.count_parameters()
    print(f"\n  Trainable parameters: {n_params:,}")

    # ── Optimizer and loss ──
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=False
    )

    # ── History ──
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   []
    }

    best_val_loss = float("inf")
    best_epoch    = 0
    best_state    = None
    patience_count = 0

    print(f"\n  {'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>9} | {'Val Acc':>8} | {'LR':>8}")
    print(f"  {'-'*65}")

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr_current = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_epoch     = epoch
            best_state     = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        # Print every 10 epochs or on improvement
        if epoch % 10 == 0 or patience_count == 0:
            flag = " *" if patience_count == 0 else ""
            print(f"  {epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.1%} | "
                  f"{val_loss:>9.4f} | {val_acc:>7.1%} | {lr_current:>8.2e}{flag}")

        if patience_count >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(best: epoch {best_epoch}, val_loss={best_val_loss:.4f})")
            break

    t_elapsed = time.time() - t0
    print(f"\n  Training complete in {t_elapsed:.1f} s")

    # ── Restore best model ──
    model.load_state_dict(best_state)

    # ── Final evaluation on test set ──
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n  ── Final evaluation (sessions 4-5) ──")
    print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.1%}")

    # ── Save model ──
    subj_tag = "all" if subjects == SUBJECTS else "-".join(map(str, subjects))
    model_path = RESULTS_DIR / f"{model_name}_{subj_tag}.pth"
    torch.save({
        "model_name": model_name,
        "subjects":   subjects,
        "best_epoch": best_epoch,
        "state_dict": best_state,
        "test_acc":   test_acc,
        "test_loss":  test_loss,
        "n_params":   n_params,
    }, str(model_path))
    print(f"  Model saved: {model_path.name}")

    # ── Learning curves ──
    _plot_learning_curves(history, model_name, subj_tag, best_epoch)

    return {
        "model_name": model_name,
        "subjects":   subjects,
        "best_epoch": best_epoch,
        "test_acc":   test_acc,
        "test_loss":  test_loss,
        "n_params":   n_params,
        "train_time": t_elapsed,
        "history":    history,
    }


def _plot_learning_curves(history: dict, model_name: str,
                           subj_tag: str, best_epoch: int):
    """Generate and save learning curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Training",   color="#2C7BB6")
    ax1.plot(epochs, history["val_loss"],   label="Validation", color="#D7191C")
    ax1.axvline(best_epoch, color="green", linestyle="--", linewidth=0.8,
                label=f"Best epoch ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (cross-entropy)")
    ax1.set_title(f"{model_name.upper()} — Loss curve")
    ax1.legend()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]],
             label="Training",   color="#2C7BB6")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],
             label="Validation", color="#D7191C")
    ax2.axvline(best_epoch, color="green", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{model_name.upper()} — Accuracy curve")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig_path = FIGURES_DIR / f"learning_curves_{model_name}_{subj_tag}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Curves saved: {fig_path.name}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CNN training — BCI-IV-2b ERSP"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=["spectnet", "eegnet", "shallowconvnet"],
                        help="Architecture to train")
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="Subjects to include (e.g. --subjects 1 2 3)")
    parser.add_argument("--all_subjects", action="store_true",
                        help="Use all 9 subjects")
    parser.add_argument("--device", type=str, default=DEVICE,
                        choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    subjects = SUBJECTS if args.all_subjects else args.subjects

    results = train_model(
        model_name=args.model,
        subjects=subjects,
        device_str=args.device
    )

    print(f"\n  ── Summary ──")
    print(f"  Model:      {results['model_name']}")
    print(f"  Parameters: {results['n_params']:,}")
    print(f"  Test Acc:   {results['test_acc']:.1%}")
    print(f"  Time:       {results['train_time']:.1f} s")


if __name__ == "__main__":
    main()
