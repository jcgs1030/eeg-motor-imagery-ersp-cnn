"""
train_ea_csp_lda.py
--------------------
Euclidean Alignment (EA) + CSP + LDA/SVM for BCI-IV-2b.

Euclidean Alignment (He et al., IEEE TNSRE, 2020) is a session-level
normalisation that reduces cross-session variability by whitening the
covariance structure of each recording session independently:

    R  = (1/N) Σᵢ XᵢXᵢᵀ / T      (mean trial covariance of the session)
    X̃ᵢ = R^(-1/2) · Xᵢ           (whitened epoch — aligned to identity)

After alignment, the Riemannian centre of mass of each session is moved
to the identity matrix, making the distribution more comparable across
the offline (train, sessions 1-3) and online (test, sessions 4-5) splits.

EA is unsupervised: applied independently to each split using only that
split's own signal statistics — no test labels are used.

Post-alignment pipeline (same as Experiment 3):
    EA → CSP (2 components) → log-variance features → LDA / SVM

Protocol
--------
Subject-specific: sessions 1-3 train, sessions 4-5 test.
EA computed separately for each split.

Output layout
-------------
results/ea_csp_lda/
    figures/
        confusion_{clf}_S{nn}.png   per-subject confusion matrix
        summary_accuracy.png        per-subject accuracy bar chart
        summary_boxplot.png         distribution across subjects
        comparison_all_methods.png  all experiments side-by-side
    metrics/
        per_subject_results.csv
        summary.csv                 mean ± std per classifier

Usage
-----
    python src/train_ea_csp_lda.py              # all subjects
    python src/train_ea_csp_lda.py --subjects 1 2 3
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
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
EA_DIR     = RESULTS_DIR / "ea_csp_lda"
EA_FIGS    = EA_DIR / "figures"
EA_METRICS = EA_DIR / "metrics"

for d in [EA_DIR, EA_FIGS, EA_METRICS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MI_TMIN         = 0.0
MI_TMAX         = 4.0
N_CSP_COMPS     = 2
EVENT_LABEL_MAP = {"left": 0, "right": 1}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_epochs(subject: int, suffix: str):
    """Load .fif epochs, crop to MI window, return (X, y)."""
    tag      = f"S{subject:02d}{suffix}"
    fif_path = DATA_PROC / f"{tag}-epo.fif"

    if not fif_path.exists():
        raise FileNotFoundError(
            f"{fif_path.name} not found. "
            f"Run first: python src/preprocessing.py --subject {subject}"
        )

    epochs = mne.read_epochs(str(fif_path), verbose=False)
    epochs = epochs.copy().crop(tmin=MI_TMIN, tmax=MI_TMAX)

    X = epochs.get_data()                                               # (N, C, T)
    code_to_label = {v: EVENT_LABEL_MAP[k] for k, v in epochs.event_id.items()}
    y = np.array([code_to_label[ev] for ev in epochs.events[:, 2]])    # (N,)
    return X, y


# ── Euclidean Alignment ───────────────────────────────────────────────────────

def euclidean_alignment(X: np.ndarray) -> np.ndarray:
    """
    Apply Euclidean Alignment to a set of EEG epochs.

    Each epoch Xᵢ ∈ ℝ^(C×T) is whitened by the square-root inverse of
    the session's mean covariance matrix R:

        R     = (1/N) Σᵢ [ XᵢXᵢᵀ / T ]
        X̃ᵢ   = R^(-1/2) · Xᵢ

    This maps the Riemannian centre of mass of the session to the identity,
    removing inter-session scaling and orientation differences without
    using any class labels.

    Parameters
    ----------
    X : (N, C, T)  — raw bandpass-filtered epochs

    Returns
    -------
    X_aligned : (N, C, T)  — whitened epochs
    R         : (C, C)     — mean covariance used for alignment
    """
    N, C, T = X.shape

    # Per-trial covariance (unscaled — T divides to keep units consistent)
    covs = np.array([x @ x.T / T for x in X])          # (N, C, C)
    R    = covs.mean(axis=0)                            # (C, C)

    # R^(-1/2) via eigendecomposition (symmetric positive-definite matrix)
    eigvals, eigvecs = np.linalg.eigh(R)
    # Clamp tiny negative eigenvalues introduced by floating-point errors
    eigvals = np.maximum(eigvals, 1e-10)
    R_inv_sqrt = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T   # (C, C)

    X_aligned = np.array([R_inv_sqrt @ x for x in X])  # (N, C, T)
    return X_aligned, R


# ── CSP features ─────────────────────────────────────────────────────────────

def extract_csp_features(X_train, y_train, X_test):
    csp = CSP(
        n_components=N_CSP_COMPS,
        reg=None,
        log=True,
        norm_trace=False,
        transform_into="average_power"
    )
    feats_train = csp.fit_transform(X_train, y_train)
    feats_test  = csp.transform(X_test)
    return feats_train, feats_test, csp


# ── Classifiers ───────────────────────────────────────────────────────────────

def build_classifiers():
    return {
        "EA+CSP+LDA": LinearDiscriminantAnalysis(solver="svd"),
        "EA+CSP+SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", C=1.0, gamma="scale",
                           random_state=RANDOM_SEED))
        ]),
    }


# ── Per-subject training ──────────────────────────────────────────────────────

def train_subject(subject: int) -> list[dict]:
    subj_tag = f"S{subject:02d}"
    print(f"\n  Subject {subj_tag}", end="", flush=True)

    # 1. Load raw epochs
    X_train_raw, y_train = load_epochs(subject, "T")
    X_test_raw,  y_test  = load_epochs(subject, "E")

    # 2. Euclidean Alignment — independently per split
    X_train_ea, R_train = euclidean_alignment(X_train_raw)
    X_test_ea,  R_test  = euclidean_alignment(X_test_raw)

    # 3. CSP feature extraction on aligned data
    feats_train, feats_test, _ = extract_csp_features(
        X_train_ea, y_train, X_test_ea
    )

    # 4. Classify
    results = []
    for clf_name, clf in build_classifiers().items():
        clf.fit(feats_train, y_train)
        y_pred = clf.predict(feats_test)

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

        tag = clf_name.lower().replace("+", "_")
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
    sns.heatmap(
        cm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=list(CLASS_NAMES.values()),
        yticklabels=list(CLASS_NAMES.values()),
        ax=ax, cbar_kws={"label": "Proportion"}
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix — {clf_name}\n(row-normalised, {subj_tag})")
    plt.tight_layout()
    path = EA_FIGS / f"confusion_{file_tag}_{subj_tag}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(df: pd.DataFrame):
    classifiers = df["classifier"].unique()
    subjects    = sorted(df["subject"].unique())
    palette     = {"EA+CSP+LDA": "#0077BB", "EA+CSP+SVM": "#CC3311"}

    # Per-subject bar chart
    x     = np.arange(len(subjects))
    n_c   = len(classifiers)
    width = 0.8 / n_c

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, clf in enumerate(classifiers):
        sub = df[df["classifier"] == clf].sort_values("subject")
        acc = sub["accuracy"].values * 100
        bars = ax.bar(
            x + (i - n_c / 2 + 0.5) * width, acc, width,
            label=clf, color=palette.get(clf, f"C{i}"), alpha=0.85
        )
        ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=8)

    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--",
               label="Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s:02d}" for s in subjects])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim([0, 110])
    ax.set_title("EA + CSP accuracy — BCI-IV-2b\nSessions 4-5 (evaluation)",
                 fontsize=12)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(EA_FIGS / "summary_accuracy.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # Box plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label, fmt in zip(
        axes,
        ["accuracy", "kappa"],
        ["Accuracy (%)", "Cohen's Kappa"],
        [lambda v: v * 100, lambda v: v]
    ):
        data = [fmt(df[df["classifier"] == c][metric].values)
                for c in classifiers]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, clf in zip(bp["boxes"], classifiers):
            patch.set_facecolor(palette.get(clf, "steelblue"))
            patch.set_alpha(0.75)
        for j, (vals, clf) in enumerate(zip(data, classifiers)):
            jitter = np.random.default_rng(0).uniform(-0.1, 0.1, size=len(vals))
            ax.scatter(np.full(len(vals), j + 1) + jitter, vals,
                       color=palette.get(clf, "steelblue"), zorder=3, s=35, alpha=0.9)
        if metric == "accuracy":
            ax.axhline(50, color="gray", linestyle="--", linewidth=0.8,
                       label="Chance (50%)")
            ax.legend(fontsize=9)
        ax.set_xticks(range(1, len(classifiers) + 1))
        ax.set_xticklabels(classifiers)
        ax.set_ylabel(label)
        ax.set_title(f"{label} across subjects (n=9)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("EA + CSP baselines — BCI-IV-2b", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(str(EA_FIGS / "summary_boxplot.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _plot_all_methods_comparison(df_ea: pd.DataFrame):
    """
    Aggregate bar chart comparing all experiments:
    CNN (subject-specific) + CSP baselines + EA baselines.
    Loads summary CSVs from prior experiments if available.
    """
    def _load_mean_std(csv_path, name_col, acc_col):
        """Parse mean ± std from a summary CSV."""
        if not csv_path.exists():
            return []
        df = pd.read_csv(str(csv_path))
        rows = []
        for _, r in df.iterrows():
            name = r[name_col]
            s    = str(r[acc_col])
            try:
                mean = float(s.split("%")[0]) / 100
                std  = float(s.split("±")[1].replace("%", "").strip()) / 100
            except Exception:
                mean, std = None, None
            rows.append({"method": name, "mean": mean, "std": std})
        return rows

    rows = []

    # Experiment 2: CNN subject-specific
    rows += _load_mean_std(
        RESULTS_DIR / "subject_specific" / "metrics" / "summary_by_model.csv",
        "Model", "Accuracy"
    )

    # Experiment 3: CSP
    csp_csv = RESULTS_DIR / "csp_lda" / "metrics" / "summary.csv"
    if csp_csv.exists():
        for _, r in pd.read_csv(str(csp_csv)).iterrows():
            s = str(r["Accuracy"])
            try:
                mean = float(s.split("%")[0]) / 100
                std  = float(s.split("±")[1].replace("%", "").strip()) / 100
            except Exception:
                mean, std = None, None
            rows.append({"method": r["Classifier"], "mean": mean, "std": std})

    # Experiment 4: EA (current)
    for clf in df_ea["classifier"].unique():
        sub = df_ea[df_ea["classifier"] == clf]["accuracy"]
        rows.append({
            "method": clf,
            "mean":   sub.mean(),
            "std":    sub.std(),
        })

    df_comp = pd.DataFrame(rows).dropna(subset=["mean"])

    palette = {
        "EEGNET":        "#2C7BB6",
        "SHALLOWCONVNET":"#D7191C",
        "SPECTNET":      "#1A9641",
        "CSP+LDA":       "#E66100",
        "CSP+SVM":       "#5D3A9B",
        "EA+CSP+LDA":    "#0077BB",
        "EA+CSP+SVM":    "#CC3311",
    }

    fig, ax = plt.subplots(figsize=(13, 5))
    x      = np.arange(len(df_comp))
    colors = [palette.get(m, "gray") for m in df_comp["method"]]
    bars   = ax.bar(x, df_comp["mean"] * 100, 0.65,
                    color=colors, alpha=0.85,
                    yerr=df_comp["std"] * 100,
                    capsize=4, error_kw={"linewidth": 1.2})
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=8)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--",
               label="Chance (50%)")

    # Vertical separators between experiment groups
    ax.axvline(2.5, color="black", linewidth=0.5, linestyle=":")
    ax.axvline(4.5, color="black", linewidth=0.5, linestyle=":")

    ax.set_xticks(x)
    ax.set_xticklabels(df_comp["method"], fontsize=8, rotation=10, ha="right")
    ax.set_ylabel("Accuracy (%) — mean ± std across 9 subjects")
    ax.set_ylim([0, 90])
    ax.set_title(
        "All experiments — BCI-IV-2b (subject-specific, sessions 4-5)\n"
        "Exp 2: CNN  |  Exp 3: CSP  |  Exp 4: EA+CSP",
        fontsize=11
    )
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(EA_FIGS / "comparison_all_methods.png"), dpi=150,
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
    csv1 = EA_METRICS / "per_subject_results.csv"
    out.to_csv(str(csv1), index=False)
    print(f"\n  Saved: {csv1.name}")

    rows = []
    for clf, grp in df.groupby("classifier"):
        rows.append({
            "Classifier": clf,
            "Accuracy":   f"{grp['accuracy'].mean():.1%} ± {grp['accuracy'].std():.1%}",
            "F1-score":   f"{grp['f1'].mean():.1%} ± {grp['f1'].std():.1%}",
            "Kappa":      f"{grp['kappa'].mean():.3f} ± {grp['kappa'].std():.3f}",
        })
    df_sum = pd.DataFrame(rows)
    csv2 = EA_METRICS / "summary.csv"
    df_sum.to_csv(str(csv2), index=False)
    print(f"  Saved: {csv2.name}")

    print("\n  ══ EA+CSP summary (mean ± std across 9 subjects) ══")
    print(df_sum.to_string(index=False))
    return df_sum


# ── Entry point ───────────────────────────────────────────────────────────────

def run(subjects: list):
    print(f"\n{'═'*58}")
    print(f"  Euclidean Alignment + CSP baselines — BCI-IV-2b")
    print(f"  Subjects: {subjects}")
    print(f"  Pipeline: EA → CSP ({N_CSP_COMPS} components) → LDA / SVM")
    print(f"  MI window: {MI_TMIN}–{MI_TMAX} s")
    print(f"{'═'*58}\n")

    all_results = []
    for subject in subjects:
        results = train_subject(subject)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    df_sum = _save_tables(df)
    _plot_summary(df)
    _plot_all_methods_comparison(df)

    print(f"\n  Figures: {EA_FIGS}")
    print(f"  Metrics: {EA_METRICS}")
    return df_sum


def main():
    parser = argparse.ArgumentParser(
        description="EA + CSP + LDA/SVM baselines — BCI-IV-2b"
    )
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=None,
        help="Subjects to process (default: 1-9)"
    )
    args = parser.parse_args()
    run(args.subjects if args.subjects else SUBJECTS)


if __name__ == "__main__":
    main()
