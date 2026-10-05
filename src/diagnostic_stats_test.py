"""
diagnostic_stats_test.py
--------------------------
Diagnostic only — does not touch the production pipeline or any saved
checkpoint/metrics file.

Paired statistical comparison (Wilcoxon signed-rank, per-subject) between
the top-performing methods in the project's cross-experiment comparison
table. The headline table in EXPERIMENTS.md ranks methods by mean accuracy
alone; with std of +-12-16% across only 9 subjects, several of those
rankings may not be statistically distinguishable from noise. This script
checks that directly instead of assuming it.

For the pooled CNN models (Experiment 1, a single shared model with no
natural per-subject breakdown), this first evaluates each saved checkpoint
separately against each subject's own evaluation-session trials to produce
a 9-value per-subject accuracy vector comparable to the subject-specific
methods.

Usage:
    python src/diagnostic_stats_test.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon, ttest_rel

sys.path.insert(0, str(Path(__file__).parent))
from config import SUBJECTS, DATA_PROC, N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS, DEVICE
from models import get_model
from dataset import ERSPDataset


def per_subject_acc_pooled(model_name: str) -> dict:
    """Evaluate the pooled ('all subjects') checkpoint separately on each
    subject's own evaluation-session (E) trials."""
    ckpt_path = Path(f"results/{model_name}_all.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = get_model(model_name, n_channels=N_CHANNELS, n_freq=IMG_FREQ_BINS,
                      n_time=IMG_TIME_BINS, n_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    accs = {}
    for subj in SUBJECTS:
        ds = ERSPDataset(subjects=[subj], suffix="E")
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in loader:
                pred = model(X).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        accs[subj] = correct / total
    return accs


def load_per_subject_csv(path: str, model_filter=None, model_col="Model",
                         subj_col="Subject", acc_col="Accuracy") -> dict:
    df = pd.read_csv(path)
    if model_filter is not None:
        df = df[df[model_col].str.lower() == model_filter.lower()]
    df = df.sort_values(subj_col)
    accs = {}
    for _, row in df.iterrows():
        val = row[acc_col]
        val = float(str(val).replace("%", "")) / 100 if isinstance(val, str) else val
        accs[int(row[subj_col])] = val
    return accs


def main():
    print("Computing per-subject accuracy for pooled CNN checkpoints (evaluated per-subject)...")
    methods = {}
    for m in ["eegnet", "shallowconvnet", "spectnet"]:
        methods[f"{m} (pooled)"] = per_subject_acc_pooled(m)
        print(f"  {m} (pooled) done")

    methods["CSP+LDA"] = load_per_subject_csv(
        "results/csp_lda/metrics/per_subject_results.csv",
        model_filter="CSP+LDA", model_col="Classifier")
    methods["FBCSP+LDA"] = load_per_subject_csv(
        "results/fbcsp/metrics/per_subject_results.csv",
        model_filter="FBCSP+LDA", model_col="Classifier")
    methods["Riem-TS+LDA"] = load_per_subject_csv(
        "results/riemannian/metrics/per_subject_results.csv",
        model_filter="Riem-TS+LDA", model_col="Classifier")
    methods["eegnet (subj-specific)"] = load_per_subject_csv(
        "results/subject_specific/metrics/per_subject_results.csv", model_filter="eegnet")
    methods["shallowconvnet (subj-specific)"] = load_per_subject_csv(
        "results/subject_specific/metrics/per_subject_results.csv", model_filter="shallowconvnet")
    methods["spectnet (subj-specific)"] = load_per_subject_csv(
        "results/subject_specific/metrics/per_subject_results.csv", model_filter="spectnet")

    print("\n== Per-subject accuracy (rows=subjects, cols=methods) ==")
    df = pd.DataFrame(methods).sort_index()
    df.index.name = "Subject"
    print((df * 100).round(1).to_string())
    print("\nMean:")
    print((df.mean() * 100).round(1).to_string())

    pairs = [
        ("FBCSP+LDA", "spectnet (pooled)"),
        ("FBCSP+LDA", "Riem-TS+LDA"),
        ("FBCSP+LDA", "spectnet (subj-specific)"),
        ("spectnet (pooled)", "spectnet (subj-specific)"),
        ("spectnet (pooled)", "eegnet (pooled)"),
        ("spectnet (pooled)", "shallowconvnet (pooled)"),
        ("FBCSP+LDA", "CSP+LDA"),
    ]

    print("\n== Paired tests (Wilcoxon signed-rank + paired t-test, n=9 subjects) ==")
    for a, b in pairs:
        va = np.array([methods[a][s] for s in SUBJECTS])
        vb = np.array([methods[b][s] for s in SUBJECTS])
        diff = va - vb
        try:
            w_stat, w_p = wilcoxon(va, vb)
        except ValueError as e:
            w_stat, w_p = float("nan"), float("nan")
        t_stat, t_p = ttest_rel(va, vb)
        sig = "**" if w_p < 0.05 else ("*" if w_p < 0.10 else "")
        print(f"  {a:28s} vs {b:28s} | mean diff={diff.mean()*100:+5.1f}pp | "
              f"Wilcoxon p={w_p:.3f} | paired-t p={t_p:.3f}  {sig}")

    print("\n(** p<0.05, * p<0.10 on Wilcoxon; n=9 is a very small sample for either test)")


if __name__ == "__main__":
    main()
