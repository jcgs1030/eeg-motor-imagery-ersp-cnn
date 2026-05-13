"""
evaluate.py
-----------
Comparative evaluation of trained models on BCI-IV-2b.
Generates full metrics, confusion matrix, and summary table.

Usage:
    python src/evaluate.py                    # evaluate all saved models
    python src/evaluate.py --model spectnet   # evaluate SpectNet only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
    RESULTS_DIR, FIGURES_DIR, METRICS_DIR, SUBJECTS,
    N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS, DEVICE, CLASS_NAMES
)
from dataset import ERSPDataset
from models import get_model


@torch.no_grad()
def predict_all(model, dataset, device) -> tuple:
    """Obtain predictions and ground-truth labels for the entire dataset."""
    model.eval()
    all_preds, all_labels = [], []
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    for X, y in loader:
        X = X.to(device)
        preds = model(X).argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the full set of classification metrics."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa":     cohen_kappa_score(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, subj_tag: str):
    """Generate and save the normalised confusion matrix."""
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
    ax.set_title(f"Confusion matrix — {model_name.upper()}\n"
                 f"(row-normalised, subjects: {subj_tag})")
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"confusion_{model_name}_{subj_tag}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved: {fig_path.name}")


def evaluate_model(model_name: str, subjects: list = None,
                   device_str: str = DEVICE) -> dict:
    """
    Load a saved model and evaluate it on the test set.
    """
    if subjects is None:
        subjects = SUBJECTS

    subj_tag = "all" if subjects == SUBJECTS else "-".join(map(str, subjects))
    device   = torch.device(device_str)

    model_path = RESULTS_DIR / f"{model_name}_{subj_tag}.pth"
    if not model_path.exists():
        print(f"  Model not found: {model_path.name}")
        print(f"  Run first: python src/train.py --model {model_name}")
        return {}

    # Load checkpoint
    ckpt = torch.load(str(model_path), map_location=device)

    # Instantiate model
    model = get_model(
        model_name,
        n_channels=N_CHANNELS,
        n_freq=IMG_FREQ_BINS,
        n_time=IMG_TIME_BINS,
        n_classes=2
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    # Test dataset (sessions 4-5)
    test_ds = ERSPDataset(subjects=subjects, suffix="E")

    y_pred, y_true = predict_all(model, test_ds, device)
    metrics = compute_metrics(y_true, y_pred)

    print(f"\n  ── {model_name.upper()} (subjects: {subj_tag}) ──")
    print(f"  Accuracy:  {metrics['accuracy']:.1%}")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall:    {metrics['recall']:.1%}")
    print(f"  F1-score:  {metrics['f1']:.1%}")
    print(f"  Kappa:     {metrics['kappa']:.3f}")

    plot_confusion_matrix(y_true, y_pred, model_name, subj_tag)

    return {"model": model_name, "subjects": subj_tag, **metrics,
            "n_params": ckpt.get("n_params", 0)}


def compare_all_models(subjects: list = None, device_str: str = DEVICE):
    """
    Evaluate all saved models and generate a comparative table.
    """
    if subjects is None:
        subjects = SUBJECTS

    model_names = ["spectnet", "eegnet", "shallowconvnet"]
    results = []

    print("\n══════════════════════════════════════════════")
    print("  Comparative model evaluation")
    print("══════════════════════════════════════════════")

    for name in model_names:
        r = evaluate_model(name, subjects, device_str)
        if r:
            results.append(r)

    if not results:
        print("\n  No trained models found to compare.")
        return

    # Summary table
    df = pd.DataFrame(results)
    df = df.rename(columns={
        "model": "Model", "accuracy": "Accuracy",
        "precision": "Precision", "recall": "Recall",
        "f1": "F1-score", "kappa": "Kappa", "n_params": "Parameters"
    })

    # Format as percentage
    for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
        if col in df.columns:
            df[col] = df[col].map("{:.1%}".format)
    if "Kappa" in df.columns:
        df["Kappa"] = df["Kappa"].map("{:.3f}".format)
    if "Parameters" in df.columns:
        df["Parameters"] = df["Parameters"].map("{:,}".format)

    print("\n\n  ══ Comparative summary table ══")
    print(df.to_string(index=False))

    # Save CSV
    subj_tag = "all" if subjects == SUBJECTS else "-".join(map(str, subjects))
    csv_path = METRICS_DIR / f"comparative_results_{subj_tag}.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\n  Table saved: {csv_path.name}")

    # Comparative bar chart
    df_raw = pd.DataFrame(results)
    _plot_comparison(df_raw, subj_tag)


def _plot_comparison(df: pd.DataFrame, subj_tag: str):
    """Bar chart with accuracy, F1, and kappa per model."""
    metrics = ["accuracy", "f1", "kappa"]
    labels  = ["Accuracy", "F1-score", "Kappa"]
    colors  = ["#2C7BB6", "#D7191C", "#1A9641"]
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        if metric in df.columns:
            bars = ax.bar(x + i * width, df[metric], width,
                          label=label, color=color, alpha=0.85)
            ax.bar_label(bars, fmt="{:.1%}" if metric != "kappa" else "{:.3f}",
                         padding=2, fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df["model"].str.upper(), fontsize=10)
    ax.set_ylabel("Metric")
    ax.set_ylim([0, 1.1])
    ax.set_title(
        f"CNN model comparison — BCI-IV-2b (subjects: {subj_tag})\n"
        f"Sessions 4-5 (evaluation)",
        fontsize=11
    )
    ax.legend()
    ax.axhline(0.5, color="gray", linewidth=0.6, linestyle="--",
               label="Chance (50%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    fig_path = FIGURES_DIR / f"model_comparison_{subj_tag}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison chart saved: {fig_path.name}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CNN model evaluation — BCI-IV-2b"
    )
    parser.add_argument("--model", type=str, default=None,
                        choices=["spectnet", "eegnet", "shallowconvnet"],
                        help="Model to evaluate (None = all)")
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    if args.model:
        evaluate_model(args.model, args.subjects, args.device)
    else:
        compare_all_models(args.subjects, args.device)


if __name__ == "__main__":
    main()
