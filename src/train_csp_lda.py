"""
train_csp_lda.py
----------------
Classical CSP + LDA (and CSP + SVM) baselines for BCI-IV-2b.

Common Spatial Patterns (CSP) learns spatial filters from the raw
band-pass filtered EEG that maximise the variance ratio between the
two motor imagery classes. Log-variance features from the projected
signal are then classified with LDA (and optionally SVM).

This is the canonical classical baseline for 2-class MI-BCI and
typically outperforms CNN approaches when training data is limited
or when cross-session domain shift is present.

Protocol
--------
- Subject-specific: one model per subject (sessions 1-3 → train,
  sessions 4-5 → test). Same split as CNN experiments.
- Input: band-pass filtered epochs loaded directly from .fif files
  (no ERSP conversion — CSP operates on raw filtered EEG).
- MI window used for feature extraction: 0 to 4 s post-cue.
- CSP components: 2 (maximum useful with 3 EEG channels;
  1 filter per class extreme captures the lateralisation pattern).

Output layout
-------------
results/csp_lda/
    figures/
        confusion_lda_S{nn}.png    per-subject confusion matrix
        confusion_svm_S{nn}.png
        summary_accuracy.png       CSP+LDA vs CSP+SVM vs CNN (SS)
        summary_boxplot.png        distribution across subjects
    metrics/
        per_subject_results.csv    one row per (classifier, subject)
        summary.csv                mean ± std per classifier

Usage
-----
    python src/train_csp_lda.py                  # all subjects
    python src/train_csp_lda.py --subjects 1 2 3
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
CSP_DIR     = RESULTS_DIR / "csp_lda"
CSP_FIGS    = CSP_DIR / "figures"
CSP_METRICS = CSP_DIR / "metrics"

for d in [CSP_DIR, CSP_FIGS, CSP_METRICS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MI_TMIN       = 0.0   # seconds post-cue — start of imagery window
MI_TMAX       = 4.0   # seconds post-cue — end of imagery window
N_CSP_COMPS   = 2     # 1 per class extreme (max useful with 3 channels)
EVENT_LABEL_MAP = {"left": 0, "right": 1}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_subject_data(subject: int, suffix: str):
    """
    Load band-pass filtered epochs from .fif, crop to the MI window,
    and return (X, y) arrays ready for CSP.

    Parameters
    ----------
    subject : int
    suffix  : 'T' (training) or 'E' (evaluation)

    Returns
    -------
    X : (n_trials, n_channels, n_times)
    y : (n_trials,)  — 0=left, 1=right
    """
    tag      = f"S{subject:02d}{suffix}"
    fif_path = DATA_PROC / f"{tag}-epo.fif"

    if not fif_path.exists():
        raise FileNotFoundError(
            f"{fif_path.name} not found. "
            f"Run first: python src/preprocessing.py --subject {subject}"
        )

    epochs = mne.read_epochs(str(fif_path), verbose=False)
    epochs = epochs.copy().crop(tmin=MI_TMIN, tmax=MI_TMAX)

    X = epochs.get_data()                        # (N, C, T)
    code_to_label = {v: EVENT_LABEL_MAP[k] for k, v in epochs.event_id.items()}
    y = np.array([code_to_label[ev] for ev in epochs.events[:, 2]])

    return X, y


# ── CSP feature extraction ────────────────────────────────────────────────────

def extract_csp_features(X_train, y_train, X_test):
    """
    Fit CSP on training data and transform both sets.

    Returns
    -------
    feats_train, feats_test : (n_trials, N_CSP_COMPS)
    csp                     : fitted CSP object
    """
    csp = CSP(
        n_components=N_CSP_COMPS,
        reg=None,
        log=True,          # log-variance features
        norm_trace=False,
        transform_into="average_power"
    )
    feats_train = csp.fit_transform(X_train, y_train)
    feats_test  = csp.transform(X_test)
    return feats_train, feats_test, csp


# ── Classifiers ───────────────────────────────────────────────────────────────

def build_classifiers():
    """Return dict of {name: sklearn estimator} to benchmark."""
    return {
        "CSP+LDA": LinearDiscriminantAnalysis(solver="svd"),
        "CSP+SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", C=1.0, gamma="scale",
                           random_state=RANDOM_SEED, probability=False))
        ]),
    }


# ── Per-subject training ──────────────────────────────────────────────────────

def train_subject(subject: int) -> list[dict]:
    """
    Run all classifiers for one subject.

    Returns a list of result dicts (one per classifier).
    """
    subj_tag = f"S{subject:02d}"
    print(f"\n  Subject {subj_tag}", end="", flush=True)

    X_train, y_train = load_subject_data(subject, "T")
    X_test,  y_test  = load_subject_data(subject, "E")

    feats_train, feats_test, csp = extract_csp_features(X_train, y_train, X_test)

    results = []
    clfs    = build_classifiers()

    for clf_name, clf in clfs.items():
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

        results.append({
            "classifier": clf_name,
            "subject":    subject,
            **metrics,
        })

    print()
    return results


# ── Figure helpers ────────────────────────────────────────────────────────────

def _plot_confusion(y_true, y_pred, clf_name: str,
                    subj_tag: str, file_tag: str):
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
    ax.set_title(f"Confusion matrix — {clf_name}\n"
                 f"(row-normalised, {subj_tag})")
    plt.tight_layout()
    path = CSP_FIGS / f"confusion_{file_tag}_{subj_tag}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(df: pd.DataFrame):
    """
    Two summary figures:
    1. Per-subject accuracy, one bar per classifier.
    2. Box plot of accuracy and kappa across subjects per classifier.
    """
    classifiers = df["classifier"].unique()
    subjects    = sorted(df["subject"].unique())
    palette     = {
        "CSP+LDA": "#E66100",
        "CSP+SVM": "#5D3A9B",
    }

    # ── 1. Per-subject accuracy bar chart ────────────────────────────────────
    x     = np.arange(len(subjects))
    n_c   = len(classifiers)
    width = 0.8 / n_c

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, clf in enumerate(classifiers):
        sub = df[df["classifier"] == clf].sort_values("subject")
        acc = sub["accuracy"].values * 100
        bars = ax.bar(
            x + (i - n_c / 2 + 0.5) * width, acc, width,
            label=clf,
            color=palette.get(clf, f"C{i}"),
            alpha=0.85
        )
        ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=8)

    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--",
               label="Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s:02d}" for s in subjects])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim([0, 110])
    ax.set_title("CSP baseline accuracy — BCI-IV-2b\n"
                 "Sessions 4-5 (evaluation)", fontsize=12)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(CSP_FIGS / "summary_accuracy.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── 2. Box plot: accuracy and kappa ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, label, fmt in zip(
        axes,
        ["accuracy", "kappa"],
        ["Accuracy (%)", "Cohen's Kappa"],
        [lambda v: v * 100, lambda v: v]
    ):
        data = [fmt(df[df["classifier"] == c][metric].values)
                for c in classifiers]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, clf in zip(bp["boxes"], classifiers):
            patch.set_facecolor(palette.get(clf, "steelblue"))
            patch.set_alpha(0.75)
        for j, (vals, clf) in enumerate(zip(data, classifiers)):
            jitter = np.random.default_rng(0).uniform(-0.1, 0.1, size=len(vals))
            ax.scatter(np.full(len(vals), j + 1) + jitter, vals,
                       color=palette.get(clf, "steelblue"),
                       zorder=3, s=35, alpha=0.9)
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

    plt.suptitle("CSP baselines — BCI-IV-2b", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(str(CSP_FIGS / "summary_boxplot.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _plot_comparison_vs_cnn(df_csp: pd.DataFrame):
    """
    Bar chart comparing CSP+LDA vs the three CNN subject-specific
    models, using the mean accuracy across subjects.

    Loads CNN results from results/subject_specific/metrics/summary_by_model.csv
    if available.
    """
    cnn_csv = RESULTS_DIR / "subject_specific" / "metrics" / "summary_by_model.csv"
    if not cnn_csv.exists():
        return

    df_cnn = pd.read_csv(str(cnn_csv))

    # Parse "XX.X% ± YY.Y%" strings
    def parse_mean(s):
        try:
            return float(s.split("%")[0]) / 100
        except Exception:
            return None

    def parse_std(s):
        try:
            return float(s.split("±")[1].replace("%", "").strip()) / 100
        except Exception:
            return None

    rows = []
    for _, r in df_cnn.iterrows():
        rows.append({
            "method": r["Model"],
            "mean":   parse_mean(r["Accuracy"]),
            "std":    parse_std(r["Accuracy"]),
        })

    # Add CSP results
    for clf in df_csp["classifier"].unique():
        sub = df_csp[df_csp["classifier"] == clf]["accuracy"]
        rows.append({
            "method": clf,
            "mean":   sub.mean(),
            "std":    sub.std(),
        })

    df_comp = pd.DataFrame(rows)

    palette = {
        "EEGNET":        "#2C7BB6",
        "SHALLOWCONVNET":"#D7191C",
        "SPECTNET":      "#1A9641",
        "CSP+LDA":       "#E66100",
        "CSP+SVM":       "#5D3A9B",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_comp))
    colors = [palette.get(m, "gray") for m in df_comp["method"]]
    bars = ax.bar(x, df_comp["mean"] * 100, 0.6,
                  color=colors, alpha=0.85,
                  yerr=df_comp["std"] * 100,
                  capsize=4, error_kw={"linewidth": 1.2})
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--",
               label="Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(df_comp["method"], fontsize=9)
    ax.set_ylabel("Accuracy (%) — mean ± std across 9 subjects")
    ax.set_ylim([0, 90])
    ax.set_title("CNN vs CSP baselines — BCI-IV-2b (subject-specific)\n"
                 "Sessions 4-5 (evaluation)", fontsize=11)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(CSP_FIGS / "comparison_cnn_vs_csp.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison figure saved: comparison_cnn_vs_csp.png")


# ── Summary tables ────────────────────────────────────────────────────────────

def _save_tables(df: pd.DataFrame):
    # Per-subject table
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
    csv1 = CSP_METRICS / "per_subject_results.csv"
    out.to_csv(str(csv1), index=False)
    print(f"\n  Saved: {csv1.name}")

    # Summary by classifier
    rows = []
    for clf, grp in df.groupby("classifier"):
        rows.append({
            "Classifier": clf,
            "Accuracy":   f"{grp['accuracy'].mean():.1%} ± {grp['accuracy'].std():.1%}",
            "F1-score":   f"{grp['f1'].mean():.1%} ± {grp['f1'].std():.1%}",
            "Kappa":      f"{grp['kappa'].mean():.3f} ± {grp['kappa'].std():.3f}",
        })
    df_sum = pd.DataFrame(rows)
    csv2 = CSP_METRICS / "summary.csv"
    df_sum.to_csv(str(csv2), index=False)
    print(f"  Saved: {csv2.name}")

    print("\n  ══ CSP baseline summary (mean ± std across 9 subjects) ══")
    print(df_sum.to_string(index=False))


# ── Entry point ───────────────────────────────────────────────────────────────

def run(subjects: list):
    print(f"\n{'═'*54}")
    print(f"  CSP baselines — BCI-IV-2b")
    print(f"  Subjects: {subjects}")
    print(f"  Classifiers: CSP+LDA, CSP+SVM")
    print(f"  MI window: {MI_TMIN}–{MI_TMAX} s | CSP components: {N_CSP_COMPS}")
    print(f"{'═'*54}\n")

    all_results = []
    for subject in subjects:
        results = train_subject(subject)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    _save_tables(df)
    _plot_summary(df)
    _plot_comparison_vs_cnn(df)
    print(f"\n  Figures: {CSP_FIGS}")
    print(f"  Metrics: {CSP_METRICS}")


def main():
    parser = argparse.ArgumentParser(
        description="CSP+LDA and CSP+SVM baselines — BCI-IV-2b"
    )
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=None,
        help="Subjects to process (default: 1-9)"
    )
    args = parser.parse_args()
    run(args.subjects if args.subjects else SUBJECTS)


if __name__ == "__main__":
    main()
