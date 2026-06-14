"""
train_fbcsp.py
--------------
Filter Bank CSP (FBCSP) baseline for BCI-IV-2b.

FBCSP decomposes the EEG signal into multiple frequency sub-bands and
applies CSP independently to each, then concatenates the log-variance
features before classification. This captures band-specific ERD/ERS
patterns that a single broad-band CSP misses.

Reference
---------
Ang, K.K. et al. (2008). Filter Bank Common Spatial Pattern (FBCSP)
Algorithm using Online Adaptive and Regularized Learning.
Proc. IEEE IJCNN.

Filter bank design
------------------
7 overlapping sub-bands spanning 4–36 Hz, as per the original paper:
    [4-8], [8-12], [12-16], [16-20], [20-24], [24-28], [28-32] Hz
Each band is bandpass-filtered with a 5th-order Butterworth filter.
CSP with 2 components is applied per band.

Feature selection
-----------------
All band features are concatenated (7 bands × 2 components = 14 features)
and classified with LDA and SVM.

Protocol
--------
Subject-specific: sessions 1-3 train → sessions 4-5 test.

Output layout
-------------
results/fbcsp/
    figures/
        confusion_{clf}_S{nn}.png
        summary_accuracy.png
        summary_boxplot.png
        comparison_all_methods.png
    metrics/
        per_subject_results.csv
        summary.csv

Usage
-----
    python src/train_fbcsp.py
    python src/train_fbcsp.py --subjects 1 2 3
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
from scipy.signal import butter, filtfilt
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
    DATA_PROC, RESULTS_DIR, SUBJECTS, CLASS_NAMES, SFREQ, RANDOM_SEED
)

mne.set_log_level("WARNING")

# ── Output directories ────────────────────────────────────────────────────────
FBCSP_DIR     = RESULTS_DIR / "fbcsp"
FBCSP_FIGS    = FBCSP_DIR / "figures"
FBCSP_METRICS = FBCSP_DIR / "metrics"

for d in [FBCSP_DIR, FBCSP_FIGS, FBCSP_METRICS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Filter bank definition ────────────────────────────────────────────────────
# 7 sub-bands of 4 Hz width, 4-32 Hz (covers mu and beta with context)
FILTER_BANK = [
    ( 4,  8),
    ( 8, 12),   # mu band
    (12, 16),
    (16, 20),   # low beta
    (20, 24),
    (24, 28),   # high beta
    (28, 32),
]
N_CSP_COMPS   = 2     # per band
BUTTER_ORDER  = 5
MI_TMIN       = 0.0
MI_TMAX       = 4.0
EVENT_LABEL_MAP = {"left": 0, "right": 1}


# ── Bandpass filter ───────────────────────────────────────────────────────────

def bandpass(X: np.ndarray, low: float, high: float,
             fs: float = SFREQ, order: int = BUTTER_ORDER) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass to (N, C, T) array.
    filtfilt is used to avoid phase distortion.
    """
    nyq  = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, X, axis=-1)


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


# ── FBCSP feature extraction ─────────────────────────────────────────────────

def extract_fbcsp_features(X_train, y_train, X_test):
    """
    For each sub-band: bandpass filter → fit CSP → extract log-variance.
    Concatenate features from all bands.

    Returns
    -------
    feats_train : (N_train, n_bands * N_CSP_COMPS)
    feats_test  : (N_test,  n_bands * N_CSP_COMPS)
    """
    train_parts, test_parts = [], []

    for low, high in FILTER_BANK:
        X_tr_bp = bandpass(X_train, low, high)
        X_te_bp = bandpass(X_test,  low, high)

        csp = CSP(
            n_components=N_CSP_COMPS,
            reg=None,
            log=True,
            norm_trace=False,
            transform_into="average_power"
        )
        train_parts.append(csp.fit_transform(X_tr_bp, y_train))
        test_parts.append(csp.transform(X_te_bp))

    feats_train = np.concatenate(train_parts, axis=1)
    feats_test  = np.concatenate(test_parts,  axis=1)
    return feats_train, feats_test


# ── Classifiers ───────────────────────────────────────────────────────────────

def build_classifiers():
    return {
        "FBCSP+LDA": LinearDiscriminantAnalysis(solver="svd"),
        "FBCSP+SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", C=1.0, gamma="scale",
                           random_state=RANDOM_SEED))
        ]),
    }


# ── Per-subject training ──────────────────────────────────────────────────────

def train_subject(subject: int) -> list[dict]:
    subj_tag = f"S{subject:02d}"
    print(f"\n  Subject {subj_tag}", end="", flush=True)

    X_train, y_train = load_epochs(subject, "T")
    X_test,  y_test  = load_epochs(subject, "E")

    feats_train, feats_test = extract_fbcsp_features(X_train, y_train, X_test)

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
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=list(CLASS_NAMES.values()),
                yticklabels=list(CLASS_NAMES.values()),
                ax=ax, cbar_kws={"label": "Proportion"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix — {clf_name}\n(row-normalised, {subj_tag})")
    plt.tight_layout()
    (FBCSP_FIGS / f"confusion_{file_tag}_{subj_tag}.png").parent.mkdir(exist_ok=True)
    fig.savefig(str(FBCSP_FIGS / f"confusion_{file_tag}_{subj_tag}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(df: pd.DataFrame):
    classifiers = df["classifier"].unique()
    subjects    = sorted(df["subject"].unique())
    palette     = {"FBCSP+LDA": "#009988", "FBCSP+SVM": "#EE7733"}

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
    ax.set_title("FBCSP accuracy — BCI-IV-2b\nSessions 4-5 (evaluation)",
                 fontsize=12)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(FBCSP_FIGS / "summary_accuracy.png"), dpi=150,
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
    plt.suptitle("FBCSP baselines — BCI-IV-2b", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(str(FBCSP_FIGS / "summary_boxplot.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


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
    csv1 = FBCSP_METRICS / "per_subject_results.csv"
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
    df_sum.to_csv(str(FBCSP_METRICS / "summary.csv"), index=False)

    print(f"\n  Saved: {csv1.name}, summary.csv")
    print("\n  ══ FBCSP summary (mean ± std across 9 subjects) ══")
    print(df_sum.to_string(index=False))
    return df_sum


# ── Entry point ───────────────────────────────────────────────────────────────

def run(subjects: list):
    bands_str = ", ".join(f"{l}-{h}" for l, h in FILTER_BANK)
    print(f"\n{'═'*60}")
    print(f"  FBCSP — BCI-IV-2b")
    print(f"  Subjects: {subjects}")
    print(f"  Filter bank ({len(FILTER_BANK)} bands): {bands_str} Hz")
    print(f"  CSP components per band: {N_CSP_COMPS}  "
          f"→ {len(FILTER_BANK) * N_CSP_COMPS} total features")
    print(f"{'═'*60}\n")

    all_results = []
    for subject in subjects:
        all_results.extend(train_subject(subject))

    df = pd.DataFrame(all_results)
    _save_tables(df)
    _plot_summary(df)
    print(f"\n  Figures: {FBCSP_FIGS}")
    return df


def main():
    parser = argparse.ArgumentParser(description="FBCSP baselines — BCI-IV-2b")
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    args = parser.parse_args()
    run(args.subjects if args.subjects else SUBJECTS)


if __name__ == "__main__":
    main()
