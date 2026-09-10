"""
train_subject_specific.py
--------------------------
Subject-specific CNN training on BCI-IV-2b ERSP spectrograms.

Paradigm
--------
For each subject (1-9) a separate model is trained using ONLY that
subject's own data:
    - Training:   sessions 1-3  (suffix 'T')
    - Evaluation: sessions 4-5  (suffix 'E')

This is the standard intra-subject protocol in the BCI literature and
contrasts with the subject-pooled approach in train.py.

Output layout
-------------
results/subject_specific/
    figures/
        lc_{model}_S{nn}.png          learning curves per (model, subject)
        confusion_{model}_S{nn}.png   normalised confusion matrix
        summary_accuracy.png          accuracy per subject, all models
        summary_kappa.png             kappa per subject, all models
    metrics/
        per_subject_results.csv       one row per (model, subject)
        summary_by_model.csv          mean ± std across subjects per model
    {model}_S{nn}.pth                 best checkpoint per (model, subject)

Usage
-----
    # All models, all subjects
    python src/train_subject_specific.py

    # One model only
    python src/train_subject_specific.py --model eegnet

    # Subset of subjects
    python src/train_subject_specific.py --subjects 1 2 3

    # GPU (if available)
    python src/train_subject_specific.py --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RESULTS_DIR, SUBJECTS, CLASS_NAMES,
    N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS,
    BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, VAL_SPLIT, RANDOM_SEED, DEVICE, MODELS
)
from dataset import build_loaders, ERSPDataset
from models import get_model

# ── Output directories ────────────────────────────────────────────────────────
SS_DIR      = RESULTS_DIR / "subject_specific"
SS_FIGS     = SS_DIR / "figures"
SS_METRICS  = SS_DIR / "metrics"

for d in [SS_DIR, SS_FIGS, SS_METRICS]:
    d.mkdir(parents=True, exist_ok=True)


# ── Training loop (single subject) ───────────────────────────────────────────

def _train_epoch(model, loader, optimizer, criterion, device):
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
def _eval_epoch(model, loader, criterion, device):
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


@torch.no_grad()
def _predict_all(model, dataset, device):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    preds, labels = [], []
    for X, y in loader:
        preds.append(model(X.to(device)).argmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def train_one_subject(model_name: str, subject: int,
                      device_str: str = DEVICE) -> dict:
    """
    Train one CNN architecture on a single subject and evaluate it.

    Parameters
    ----------
    model_name : one of 'eegnet', 'shallowconvnet', 'spectnet'
    subject    : integer in [1, 9]
    device_str : 'cpu', 'cuda', or 'mps'

    Returns
    -------
    dict with training history and final evaluation metrics
    """
    subj_tag = f"S{subject:02d}"
    device   = torch.device(device_str)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"\n  ── {model_name.upper()} | Subject {subj_tag} ──")

    train_loader, val_loader, _ = build_loaders(
        subjects=[subject],
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        seed=RANDOM_SEED
    )

    model = get_model(
        model_name,
        n_channels=N_CHANNELS,
        n_freq=IMG_FREQ_BINS,
        n_time=IMG_TIME_BINS,
        n_classes=2
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   []
    }
    best_val_loss  = float("inf")
    best_epoch     = 0
    best_state     = None
    patience_count = 0

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = _eval_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_loss < best_val_loss:
            best_val_loss  = vl_loss
            best_epoch     = epoch
            best_state     = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            break

    elapsed = time.time() - t0

    # Restore best checkpoint
    model.load_state_dict(best_state)

    # Evaluate on test set (sessions 4-5)
    test_ds   = ERSPDataset(subjects=[subject], suffix="E")
    y_pred, y_true = _predict_all(model, test_ds, device)

    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa":     cohen_kappa_score(y_true, y_pred),
    }

    print(f"     best epoch={best_epoch}  "
          f"acc={metrics['accuracy']:.1%}  kappa={metrics['kappa']:.3f}  "
          f"({elapsed:.0f}s)")

    # Save checkpoint
    ckpt_path = SS_DIR / f"{model_name}_{subj_tag}.pth"
    torch.save({
        "model_name": model_name,
        "subject":    subject,
        "best_epoch": best_epoch,
        "state_dict": best_state,
        **metrics,
        "n_params":   model.count_parameters(),
    }, str(ckpt_path))

    # Figures
    _plot_learning_curves(history, model_name, subj_tag, best_epoch)
    _plot_confusion(y_true, y_pred, model_name, subj_tag)

    return {
        "model":   model_name,
        "subject": subject,
        **metrics,
        "best_epoch":  best_epoch,
        "train_time":  elapsed,
        "n_params":    model.count_parameters(),
    }


# ── Figure helpers ────────────────────────────────────────────────────────────

def _plot_learning_curves(history: dict, model_name: str,
                          subj_tag: str, best_epoch: int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Training",   color="#2C7BB6")
    ax1.plot(epochs, history["val_loss"],   label="Validation", color="#D7191C")
    ax1.axvline(best_epoch, color="green", linestyle="--", linewidth=0.8,
                label=f"Best epoch ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (cross-entropy)")
    ax1.set_title(f"{model_name.upper()} — {subj_tag} — Loss")
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
    ax2.set_title(f"{model_name.upper()} — {subj_tag} — Accuracy")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    path = SS_FIGS / f"lc_{model_name}_{subj_tag}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion(y_true, y_pred, model_name: str, subj_tag: str):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=list(CLASS_NAMES.values()),
        yticklabels=list(CLASS_NAMES.values()),
        ax=ax, cbar_kws={"label": "Proportion"}
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix — {model_name.upper()} — {subj_tag}\n"
                 f"(row-normalised, subject-specific)")
    plt.tight_layout()
    path = SS_FIGS / f"confusion_{model_name}_{subj_tag}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(df: pd.DataFrame):
    """
    Two summary figures:
    1. Grouped bar chart: accuracy per subject, one bar per model.
    2. Box plot: distribution of accuracy and kappa across subjects per model.
    """
    model_names = df["model"].unique()
    subjects    = sorted(df["subject"].unique())
    palette     = {"eegnet": "#2C7BB6", "shallowconvnet": "#D7191C", "spectnet": "#1A9641"}

    # ── 1. Per-subject accuracy bar chart ────────────────────────────────────
    x     = np.arange(len(subjects))
    n_m   = len(model_names)
    width = 0.8 / n_m

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, mname in enumerate(model_names):
        sub = df[df["model"] == mname].sort_values("subject")
        acc = sub["accuracy"].values * 100
        bars = ax.bar(
            x + (i - n_m / 2 + 0.5) * width, acc, width,
            label=mname.upper(),
            color=palette.get(mname, f"C{i}"),
            alpha=0.85
        )
        ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=7)

    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", label="Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s:02d}" for s in subjects])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim([0, 110])
    ax.set_title("Subject-specific accuracy — BCI-IV-2b\n"
                 "Sessions 4-5 (evaluation)", fontsize=12)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(SS_FIGS / "summary_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Box plot: accuracy and kappa distributions ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, label, fmt in zip(
        axes,
        ["accuracy", "kappa"],
        ["Accuracy (%)", "Cohen's Kappa"],
        [lambda v: v * 100, lambda v: v]
    ):
        data = [fmt(df[df["model"] == m][metric].values)
                for m in model_names]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, mname in zip(bp["boxes"], model_names):
            patch.set_facecolor(palette.get(mname, "steelblue"))
            patch.set_alpha(0.75)
        # overlay individual points
        for j, (vals, mname) in enumerate(zip(data, model_names)):
            jitter = np.random.default_rng(0).uniform(-0.1, 0.1, size=len(vals))
            ax.scatter(np.full(len(vals), j + 1) + jitter, vals,
                       color=palette.get(mname, "steelblue"),
                       zorder=3, s=30, alpha=0.9)
        if metric == "accuracy":
            ax.axhline(50, color="gray", linestyle="--", linewidth=0.8,
                       label="Chance (50%)")
            ax.legend(fontsize=9)
        ax.set_xticks(range(1, len(model_names) + 1))
        ax.set_xticklabels([m.upper() for m in model_names])
        ax.set_ylabel(label)
        ax.set_title(f"{label} across subjects (n=9)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Subject-specific results — BCI-IV-2b", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(str(SS_FIGS / "summary_boxplot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_summary_tables(df: pd.DataFrame):
    """Save per-subject CSV and aggregated summary by model."""
    # Per-subject table
    out = df[["model", "subject", "accuracy", "precision",
              "recall", "f1", "kappa", "best_epoch", "n_params"]].copy()
    for col in ["accuracy", "precision", "recall", "f1"]:
        out[col] = out[col].map("{:.1%}".format)
    out["kappa"]    = out["kappa"].map("{:.3f}".format)
    out["n_params"] = out["n_params"].map("{:,}".format)
    out = out.rename(columns={
        "model": "Model", "subject": "Subject",
        "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1-score",
        "kappa": "Kappa", "best_epoch": "Best Epoch",
        "n_params": "Parameters"
    })
    csv1 = SS_METRICS / "per_subject_results.csv"
    out.to_csv(str(csv1), index=False)
    print(f"\n  Saved: {csv1.name}")

    # Summary by model: mean ± std
    rows = []
    for mname, grp in df.groupby("model"):
        rows.append({
            "Model":      mname.upper(),
            "Accuracy":   f"{grp['accuracy'].mean():.1%} ± {grp['accuracy'].std():.1%}",
            "F1-score":   f"{grp['f1'].mean():.1%} ± {grp['f1'].std():.1%}",
            "Kappa":      f"{grp['kappa'].mean():.3f} ± {grp['kappa'].std():.3f}",
            "Parameters": f"{int(grp['n_params'].iloc[0]):,}",
        })
    df_sum = pd.DataFrame(rows)
    csv2 = SS_METRICS / "summary_by_model.csv"
    df_sum.to_csv(str(csv2), index=False)
    print(f"  Saved: {csv2.name}")

    print("\n  ══ Subject-specific summary (mean ± std across 9 subjects) ══")
    print(df_sum.to_string(index=False))


# ── Entry point ───────────────────────────────────────────────────────────────

def run(model_names: list, subjects: list, device_str: str):
    total = len(model_names) * len(subjects)
    print(f"\n{'═'*54}")
    print(f"  Subject-specific training")
    print(f"  Models:   {[m.upper() for m in model_names]}")
    print(f"  Subjects: {subjects}")
    print(f"  Total runs: {total}")
    print(f"{'═'*54}")

    all_results = []
    t_global = time.time()

    for model_name in model_names:
        print(f"\n{'─'*54}")
        print(f"  Model: {model_name.upper()}")
        print(f"{'─'*54}")
        for subject in subjects:
            result = train_one_subject(model_name, subject, device_str)
            all_results.append(result)

    total_time = time.time() - t_global
    print(f"\n  All runs completed in {total_time/60:.1f} min")

    df = pd.DataFrame(all_results)
    _save_summary_tables(df)
    _plot_summary(df)
    print(f"\n  Figures saved to: {SS_FIGS}")
    print(f"  Metrics saved to: {SS_METRICS}")


def main():
    parser = argparse.ArgumentParser(
        description="Subject-specific CNN training — BCI-IV-2b"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["spectnet", "eegnet", "shallowconvnet"],
        help="Single model to train (default: all three)"
    )
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=None,
        help="Subjects to process (default: 1-9)"
    )
    parser.add_argument(
        "--device", type=str, default=DEVICE,
        choices=["cpu", "cuda", "mps"]
    )
    args = parser.parse_args()

    model_names = [args.model] if args.model else MODELS
    subjects    = args.subjects if args.subjects else SUBJECTS

    run(model_names, subjects, args.device)


if __name__ == "__main__":
    main()
