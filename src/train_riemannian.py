"""
train_riemannian.py
--------------------
Riemannian geometry classifiers for BCI-IV-2b.

Instead of operating on raw EEG signals, these methods work on the
covariance matrices of the EEG epochs as points on the Riemannian
manifold of Symmetric Positive Definite (SPD) matrices.

Two classifiers are implemented:

1. MDM (Minimum Distance to Mean)
   Computes the Riemannian mean of each class's covariance matrices
   and classifies each test trial to the nearest class mean using
   the geodesic distance on the SPD manifold.

   Reference: Barachant, A. et al. (2012). Multiclass Brain-Computer
   Interface Classification by Riemannian Geometry. IEEE TBME, 59(4).

2. Tangent Space + LDA (TS+LDA)
   Projects covariance matrices to the tangent space at the Riemannian
   mean of the training set, obtaining a Euclidean feature vector per
   trial, then applies LDA. More flexible than MDM.

   Reference: Barachant, A. et al. (2013). Classification of covariance
   matrices using a Riemannian-based kernel for BCI applications.
   Neurocomputing, 112.

Why Riemannian for domain adaptation?
--------------------------------------
Riemannian methods are inherently more robust to inter-session
variability than Euclidean approaches because the geodesic distance
on the SPD manifold is invariant to linear transformations of the
signal (congruence invariance). This makes them less sensitive to
amplitude and orientation differences between sessions.

Protocol
--------
Subject-specific: sessions 1-3 train → sessions 4-5 test.
Covariance matrices computed on the MI window (0-4 s).

Output layout
-------------
results/riemannian/
    figures/
        confusion_{clf}_S{nn}.png
        summary_accuracy.png
        summary_boxplot.png
        comparison_all_methods.png  (all experiments)
    metrics/
        per_subject_results.csv
        summary.csv

Usage
-----
    python src/train_riemannian.py
    python src/train_riemannian.py --subjects 1 2 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROC, RESULTS_DIR, SUBJECTS, CLASS_NAMES, RANDOM_SEED
)

mne.set_log_level("WARNING")

# ── Output directories ────────────────────────────────────────────────────────
RIEM_DIR     = RESULTS_DIR / "riemannian"
RIEM_FIGS    = RIEM_DIR / "figures"
RIEM_METRICS = RIEM_DIR / "metrics"

for d in [RIEM_DIR, RIEM_FIGS, RIEM_METRICS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MI_TMIN         = 0.0
MI_TMAX         = 4.0
EVENT_LABEL_MAP = {"left": 0, "right": 1}

# Covariance estimator: "cov" (standard), "lwf" (Ledoit-Wolf, regularised)
COV_ESTIMATOR = "lwf"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_epochs(subject: int, suffix: str):
    tag      = f"S{subject:02d}{suffix}"
    fif_path = DATA_PROC / f"{tag}-epo.fif"
    if not fif_path.exists():
        raise FileNotFoundError(f"{fif_path.name} not found.")
    epochs = mne.read_epochs(str(fif_path), verbose=False)
    epochs = epochs.copy().crop(tmin=MI_TMIN, tmax=MI_TMAX)
    X = epochs.get_data()
    code_to_label = {v: EVENT_LABEL_MAP[k] for k, v in epochs.event_id.items()}
    y = np.array([code_to_label[ev] for ev in epochs.events[:, 2]])
    return X, y


# ── Classifiers ───────────────────────────────────────────────────────────────

def build_classifiers():
    """
    Returns dict of {name: sklearn-compatible Pipeline}.

    Both pipelines share the same covariance estimation step.
    pyriemann's Covariances transformer converts (N, C, T) epochs
    to (N, C, C) SPD covariance matrices.
    """
    cov = Covariances(estimator=COV_ESTIMATOR)

    return {
        "Riem-MDM": Pipeline([
            ("cov", Covariances(estimator=COV_ESTIMATOR)),
            ("mdm", MDM(metric="riemann")),
        ]),
        "Riem-TS+LDA": Pipeline([
            ("cov", Covariances(estimator=COV_ESTIMATOR)),
            ("ts",  TangentSpace(metric="riemann")),
            ("lda", LinearDiscriminantAnalysis(solver="svd")),
        ]),
    }


# ── Per-subject training ──────────────────────────────────────────────────────

def train_subject(subject: int) -> list[dict]:
    subj_tag = f"S{subject:02d}"
    print(f"\n  Subject {subj_tag}", end="", flush=True)

    X_train, y_train = load_epochs(subject, "T")
    X_test,  y_test  = load_epochs(subject, "E")

    results = []
    for clf_name, clf in build_classifiers().items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="macro",
                                         zero_division=0),
            "recall":    recall_score(y_test, y_pred, average="macro",
                                      zero_division=0),
            "f1":        f1_score(y_test, y_pred, average="macro",
                                  zero_division=0),
            "kappa":     cohen_kappa_score(y_test, y_pred),
        }

        tag = clf_name.lower().replace("+", "_").replace("-", "_")
        _plot_confusion(y_test, y_pred, clf_name, subj_tag, tag)

        print(f"  | {clf_name}: acc={metrics['accuracy']:.1%}  "
              f"kappa={metrics['kappa']:.3f}", end="", flush=True)

        results.append({"classifier": clf_name, "subject": subject, **metrics})

    print()
    return results


# ── Figure helpers ────────────────────────────────────────────────────────────

def _plot_confusion(y_true, y_pred, clf_name, subj_tag, file_tag):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=list(CLASS_NAMES.values()),
                yticklabels=list(CLASS_NAMES.values()),
                ax=ax, cbar_kws={"label": "Proportion"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix — {clf_name}\n(row-normalised, {subj_tag})")
    plt.tight_layout()
    fig.savefig(str(RIEM_FIGS / f"confusion_{file_tag}_{subj_tag}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(df: pd.DataFrame):
    classifiers = df["classifier"].unique()
    subjects    = sorted(df["subject"].unique())
    palette     = {"Riem-MDM": "#882255", "Riem-TS+LDA": "#44AA99"}

    x, n_c, width = np.arange(len(subjects)), len(classifiers), 0.8 / len(classifiers)
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, clf in enumerate(classifiers):
        sub  = df[df["classifier"] == clf].sort_values("subject")
        acc  = sub["accuracy"].values * 100
        bars = ax.bar(x + (i - n_c / 2 + 0.5) * width, acc, width,
                      label=clf, color=palette.get(clf, f"C{i}"), alpha=0.85)
        ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=8)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--",
               label="Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s:02d}" for s in subjects])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim([0, 110])
    ax.set_title("Riemannian classifier accuracy — BCI-IV-2b\n"
                 "Sessions 4-5 (evaluation)", fontsize=12)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(RIEM_FIGS / "summary_accuracy.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label, fmt in zip(
        axes,
        ["accuracy", "kappa"],
        ["Accuracy (%)", "Cohen's Kappa"],
        [lambda v: v * 100, lambda v: v]
    ):
        data = [fmt(df[df["classifier"] == c][metric].values) for c in classifiers]
        bp   = ax.boxplot(data, patch_artist=True,
                          medianprops={"color": "black", "linewidth": 2})
        for patch, clf in zip(bp["boxes"], classifiers):
            patch.set_facecolor(palette.get(clf, "steelblue"))
            patch.set_alpha(0.75)
        for j, (vals, clf) in enumerate(zip(data, classifiers)):
            jitter = np.random.default_rng(0).uniform(-0.1, 0.1, size=len(vals))
            ax.scatter(np.full(len(vals), j + 1) + jitter, vals,
                       color=palette.get(clf, "steelblue"), zorder=3, s=35, alpha=0.9)
        if metric == "accuracy":
            ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(1, len(classifiers) + 1))
        ax.set_xticklabels(classifiers)
        ax.set_ylabel(label)
        ax.set_title(f"{label} across subjects (n=9)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.suptitle("Riemannian classifiers — BCI-IV-2b", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(str(RIEM_FIGS / "summary_boxplot.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _plot_all_methods(df_riem: pd.DataFrame):
    """All-experiments comparison bar chart."""

    def _parse(s):
        try:
            mean = float(str(s).split("%")[0]) / 100
            std  = float(str(s).split("±")[1].replace("%", "").strip()) / 100
            return mean, std
        except Exception:
            return None, None

    rows = []

    # Load prior experiment summaries
    for csv_path, name_col, acc_col in [
        (RESULTS_DIR / "subject_specific" / "metrics" / "summary_by_model.csv",
         "Model", "Accuracy"),
        (RESULTS_DIR / "csp_lda"    / "metrics" / "summary.csv", "Classifier", "Accuracy"),
        (RESULTS_DIR / "ea_csp_lda" / "metrics" / "summary.csv", "Classifier", "Accuracy"),
        (RESULTS_DIR / "fbcsp"      / "metrics" / "summary.csv", "Classifier", "Accuracy"),
    ]:
        if not csv_path.exists():
            continue
        for _, r in pd.read_csv(str(csv_path)).iterrows():
            m, s = _parse(r[acc_col])
            rows.append({"method": r[name_col], "mean": m, "std": s})

    # Current experiment
    for clf, grp in df_riem.groupby("classifier"):
        rows.append({
            "method": clf,
            "mean":   grp["accuracy"].mean(),
            "std":    grp["accuracy"].std(),
        })

    df_comp = pd.DataFrame(rows).dropna(subset=["mean"])

    palette = {
        "EEGNET": "#2C7BB6", "SHALLOWCONVNET": "#D7191C", "SPECTNET": "#1A9641",
        "CSP+LDA": "#E66100", "CSP+SVM": "#5D3A9B",
        "EA+CSP+LDA": "#0077BB", "EA+CSP+SVM": "#CC3311",
        "FBCSP+LDA": "#009988", "FBCSP+SVM": "#EE7733",
        "Riem-MDM": "#882255", "Riem-TS+LDA": "#44AA99",
    }

    fig, ax = plt.subplots(figsize=(16, 5))
    x      = np.arange(len(df_comp))
    colors = [palette.get(m, "gray") for m in df_comp["method"]]
    bars   = ax.bar(x, df_comp["mean"] * 100, 0.65,
                    color=colors, alpha=0.85,
                    yerr=df_comp["std"] * 100,
                    capsize=4, error_kw={"linewidth": 1.2})
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=7)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--",
               label="Chance (50%)")

    # Group separators
    for sep in [2.5, 4.5, 6.5, 8.5]:
        if sep < len(df_comp):
            ax.axvline(sep, color="black", linewidth=0.5, linestyle=":")

    ax.set_xticks(x)
    ax.set_xticklabels(df_comp["method"], fontsize=7.5,
                       rotation=12, ha="right")
    ax.set_ylabel("Accuracy (%) — mean ± std across 9 subjects")
    ax.set_ylim([0, 90])
    ax.set_title(
        "All experiments — BCI-IV-2b (subject-specific, sessions 4-5)\n"
        "Exp 2: CNN  |  Exp 3: CSP  |  Exp 4: EA  |  Exp 5a: FBCSP  |  Exp 5b: Riemannian",
        fontsize=10
    )
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(RIEM_FIGS / "comparison_all_methods.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison figure saved: comparison_all_methods.png")


# ── Summary tables ────────────────────────────────────────────────────────────

def _save_tables(df: pd.DataFrame):
    out = df[["classifier", "subject", "accuracy", "precision",
              "recall", "f1", "kappa"]].copy()
    for col in ["accuracy", "precision", "recall", "f1"]:
        out[col] = out[col].map("{:.1%}".format)
    out["kappa"] = out["kappa"].map("{:.3f}".format)
    out = out.rename(columns={
        "classifier": "Classifier", "subject": "Subject",
        "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1-score", "kappa": "Kappa"
    })
    csv1 = RIEM_METRICS / "per_subject_results.csv"
    out.to_csv(str(csv1), index=False)

    rows = []
    for clf, grp in df.groupby("classifier"):
        rows.append({
            "Classifier": clf,
            "Accuracy":   f"{grp['accuracy'].mean():.1%} ± {grp['accuracy'].std():.1%}",
            "F1-score":   f"{grp['f1'].mean():.1%} ± {grp['f1'].std():.1%}",
            "Kappa":      f"{grp['kappa'].mean():.3f} ± {grp['kappa'].std():.3f}",
        })
    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(str(RIEM_METRICS / "summary.csv"), index=False)

    print(f"\n  Saved: {csv1.name}, summary.csv")
    print("\n  ══ Riemannian summary (mean ± std across 9 subjects) ══")
    print(df_sum.to_string(index=False))
    return df_sum


# ── Entry point ───────────────────────────────────────────────────────────────

def run(subjects: list):
    print(f"\n{'═'*60}")
    print(f"  Riemannian classifiers — BCI-IV-2b")
    print(f"  Subjects: {subjects}")
    print(f"  Methods: MDM (geodesic), TS+LDA (tangent space)")
    print(f"  Covariance estimator: {COV_ESTIMATOR}  |  MI window: {MI_TMIN}–{MI_TMAX} s")
    print(f"{'═'*60}\n")

    all_results = []
    for subject in subjects:
        all_results.extend(train_subject(subject))

    df = pd.DataFrame(all_results)
    _save_tables(df)
    _plot_summary(df)
    _plot_all_methods(df)
    print(f"\n  Figures: {RIEM_FIGS}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Riemannian classifiers — BCI-IV-2b"
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    args = parser.parse_args()
    run(args.subjects if args.subjects else SUBJECTS)


if __name__ == "__main__":
    main()
