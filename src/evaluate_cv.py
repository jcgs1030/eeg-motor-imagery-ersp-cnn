"""
evaluate_cv.py
--------------
Within-training-set cross-validation for all classifiers (Protocol A).

Uses 5-fold stratified CV entirely within sessions 1-3 (true labels).
This protocol is self-contained: it does not rely on the evaluation
sessions and provides a reliable upper-bound estimate of performance
under ideal train/test conditions.

Classifiers evaluated:
    CNN (EEGNet, ShallowConvNet, SpectNet)  — on ERSP spectrograms
    CSP+LDA, CSP+SVM                        — on raw filtered epochs
    FBCSP+LDA                               — multi-band CSP
    Riem-MDM, Riem-TS+LDA                  — Riemannian covariance

Output
------
results/cv_within_train/
    figures/
        cv_accuracy.png    per-subject accuracy (mean ± std across folds)
        cv_boxplot.png     distribution across subjects per method
    metrics/
        per_subject_cv.csv
        summary_cv.csv

Usage
-----
    python src/evaluate_cv.py
    python src/evaluate_cv.py --subjects 1 2 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
from scipy.signal import butter, filtfilt
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROC, RESULTS_DIR, SUBJECTS, SFREQ, RANDOM_SEED,
    N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, DEVICE
)
from dataset import ERSPDataset
from models import get_model

mne.set_log_level("WARNING")

# ── Output ────────────────────────────────────────────────────────────────────
CV_DIR     = RESULTS_DIR / "cv_within_train"
CV_FIGS    = CV_DIR / "figures"
CV_METRICS = CV_DIR / "metrics"
for d in [CV_DIR, CV_FIGS, CV_METRICS]:
    d.mkdir(parents=True, exist_ok=True)

N_SPLITS       = 5
MI_TMIN        = 0.0
MI_TMAX        = 4.0
FILTER_BANK    = [(4,8),(8,12),(12,16),(16,20),(20,24),(24,28),(28,32)]
EVENT_LABEL_MAP = {"left": 0, "right": 1}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_train_eeg(subject: int):
    ep = mne.read_epochs(str(DATA_PROC / f"S{subject:02d}T-epo.fif"), verbose=False)
    ep = ep.copy().crop(tmin=MI_TMIN, tmax=MI_TMAX)
    X = ep.get_data()
    code2lab = {v: EVENT_LABEL_MAP[k] for k, v in ep.event_id.items()}
    y = np.array([code2lab[e] for e in ep.events[:, 2]])
    return X, y


def load_train_ersp(subject: int):
    data = np.load(str(DATA_PROC / f"S{subject:02d}T-ersp.npz"))
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.long)
    return X, y


def bandpass(X, low, high, fs=SFREQ, order=5):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, X, axis=-1)


# ── Classical CV fold ─────────────────────────────────────────────────────────

def _classical_pipelines():
    return {
        "CSP+LDA": Pipeline([("csp", CSP(n_components=4, log=True)),
                              ("lda", LinearDiscriminantAnalysis())]),
        "CSP+SVM": Pipeline([("csp", CSP(n_components=4, log=True)),
                              ("scaler", StandardScaler()),
                              ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
                                          random_state=RANDOM_SEED))]),
        "Riem-MDM":    Pipeline([("cov", Covariances("lwf")), ("mdm", MDM("riemann"))]),
        "Riem-TS+LDA": Pipeline([("cov", Covariances("lwf")),
                                  ("ts",  TangentSpace("riemann")),
                                  ("lda", LinearDiscriminantAnalysis())]),
    }


def _fbcsp_features(X_tr, y_tr, X_te):
    parts_tr, parts_te = [], []
    for lo, hi in FILTER_BANK:
        Xtr_bp = bandpass(X_tr, lo, hi)
        Xte_bp = bandpass(X_te, lo, hi)
        csp = CSP(n_components=2, log=True)
        parts_tr.append(csp.fit_transform(Xtr_bp, y_tr))
        parts_te.append(csp.transform(Xte_bp))
    return np.concatenate(parts_tr, 1), np.concatenate(parts_te, 1)


def cv_classical(X, y, subject):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    results = {name: [] for name in list(_classical_pipelines().keys()) + ["FBCSP+LDA"]}

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        for name, clf in _classical_pipelines().items():
            clf.fit(X_tr, y_tr)
            results[name].append(clf.score(X_te, y_te))

        # FBCSP
        ft_tr, ft_te = _fbcsp_features(X_tr, y_tr, X_te)
        lda = LinearDiscriminantAnalysis()
        lda.fit(ft_tr, y_tr)
        results["FBCSP+LDA"].append(lda.score(ft_te, y_te))

    return {name: np.array(scores) for name, scores in results.items()}


# ── CNN CV fold ───────────────────────────────────────────────────────────────

def _train_cnn_fold(model_name, X_tr, y_tr, X_val, y_val, device):
    model = get_model(model_name, n_channels=N_CHANNELS,
                      n_freq=IMG_FREQ_BINS, n_time=IMG_TIME_BINS,
                      n_classes=2).to(device)
    opt  = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=10)
    loader_tr  = DataLoader(TensorDataset(X_tr, y_tr),  batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    best_val, best_w, patience_cnt = float("inf"), None, 0
    for _ in range(100):       # max 100 epochs per fold (fast CV)
        model.train()
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = sum(crit(model(xb.to(device)), yb.to(device)).item() * len(yb)
                     for xb, yb in loader_val) / len(y_val)
        sched.step(vl)
        if vl < best_val:
            best_val = vl; best_w = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= PATIENCE:
            break

    model.load_state_dict(best_w)
    model.eval()
    with torch.no_grad():
        logits = model(X_val.to(device))
        acc = (logits.argmax(1) == y_val.to(device)).float().mean().item()
    return acc


def cv_cnn(X, y, subject):
    skf    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    device = torch.device(DEVICE)
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    results = {m: [] for m in ["EEGNet", "ShallowConvNet", "SpectNet"]}
    model_map = {"EEGNet": "eegnet", "ShallowConvNet": "shallowconvnet", "SpectNet": "spectnet"}

    for tr_idx, te_idx in skf.split(X.numpy(), y.numpy()):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        for name, mkey in model_map.items():
            acc = _train_cnn_fold(mkey, X_tr, y_tr, X_te, y_te, device)
            results[name].append(acc)

    return {name: np.array(scores) for name, scores in results.items()}


# ── Per-subject ───────────────────────────────────────────────────────────────

def evaluate_subject(subject: int) -> list[dict]:
    subj_tag = f"S{subject:02d}"
    print(f"\n  Subject {subj_tag}", end="", flush=True)

    X_eeg, y_eeg = load_train_eeg(subject)
    X_ersp, y_ersp = load_train_ersp(subject)

    rows = []

    # Classical methods (on raw EEG)
    cl_scores = cv_classical(X_eeg, y_eeg, subject)
    for name, scores in cl_scores.items():
        print(f"  | {name}: {scores.mean():.1%}", end="", flush=True)
        rows.append({"classifier": name, "subject": subject,
                     "cv_mean": scores.mean(), "cv_std": scores.std()})

    # CNNs (on ERSP)
    cnn_scores = cv_cnn(X_ersp, y_ersp, subject)
    for name, scores in cnn_scores.items():
        print(f"  | {name}: {scores.mean():.1%}", end="", flush=True)
        rows.append({"classifier": name, "subject": subject,
                     "cv_mean": scores.mean(), "cv_std": scores.std()})

    print()
    return rows


# ── Figures & tables ──────────────────────────────────────────────────────────

def _plot_cv(df: pd.DataFrame):
    classifiers = df["classifier"].unique()
    subjects    = sorted(df["subject"].unique())
    palette = {
        "CSP+LDA":"#E66100","CSP+SVM":"#5D3A9B",
        "FBCSP+LDA":"#009988",
        "Riem-MDM":"#882255","Riem-TS+LDA":"#44AA99",
        "EEGNet":"#2C7BB6","ShallowConvNet":"#D7191C","SpectNet":"#1A9641",
    }

    x, n_c, w = np.arange(len(subjects)), len(classifiers), 0.8 / len(classifiers)
    fig, ax = plt.subplots(figsize=(16, 5))
    for i, clf in enumerate(classifiers):
        sub  = df[df["classifier"] == clf].sort_values("subject")
        acc  = sub["cv_mean"].values * 100
        err  = sub["cv_std"].values * 100
        bars = ax.bar(x + (i - n_c / 2 + 0.5) * w, acc, w,
                      label=clf, color=palette.get(clf, f"C{i}"), alpha=0.85,
                      yerr=err, capsize=2, error_kw={"linewidth": 0.8})
        ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=6)

    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels([f"S{s:02d}" for s in subjects])
    ax.set_ylabel("Accuracy (%) — 5-fold CV mean ± std")
    ax.set_ylim([0, 120])
    ax.set_title("Within-train 5-fold CV — BCI-IV-2b (sessions 1-3 only)", fontsize=12)
    ax.legend(fontsize=7, ncol=4)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(CV_FIGS / "cv_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Box plot
    fig, ax = plt.subplots(figsize=(12, 5))
    data = [df[df["classifier"] == c]["cv_mean"].values * 100 for c in classifiers]
    bp   = ax.boxplot(data, patch_artist=True,
                      medianprops={"color": "black", "linewidth": 2})
    for patch, clf in zip(bp["boxes"], classifiers):
        patch.set_facecolor(palette.get(clf, "steelblue")); patch.set_alpha(0.75)
    for j, (vals, clf) in enumerate(zip(data, classifiers)):
        jitter = np.random.default_rng(0).uniform(-0.1, 0.1, size=len(vals))
        ax.scatter(np.full(len(vals), j + 1) + jitter, vals,
                   color=palette.get(clf, "steelblue"), zorder=3, s=35, alpha=0.9)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(1, len(classifiers) + 1))
    ax.set_xticklabels(classifiers, rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("5-fold CV accuracy (%)")
    ax.set_title("Distribution across subjects — within-train CV", fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(CV_FIGS / "cv_boxplot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_tables(df: pd.DataFrame):
    out = df.copy()
    out["Accuracy (mean)"] = out["cv_mean"].map("{:.1%}".format)
    out["Accuracy (std)"]  = out["cv_std"].map("{:.1%}".format)
    out = out.rename(columns={"classifier":"Classifier","subject":"Subject"})
    out[["Classifier","Subject","Accuracy (mean)","Accuracy (std)"]].to_csv(
        str(CV_METRICS / "per_subject_cv.csv"), index=False)

    rows = []
    for clf, grp in df.groupby("classifier"):
        rows.append({
            "Classifier": clf,
            "CV Accuracy": f"{grp['cv_mean'].mean():.1%} ± {grp['cv_mean'].std():.1%}",
        })
    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(str(CV_METRICS / "summary_cv.csv"), index=False)
    print(f"\n  Saved: per_subject_cv.csv, summary_cv.csv")
    print("\n  ══ Within-train 5-fold CV summary (mean ± std across 9 subjects) ══")
    print(df_sum.to_string(index=False))


# ── Entry point ───────────────────────────────────────────────────────────────

def run(subjects: list):
    print(f"\n{'═'*60}")
    print(f"  Within-train 5-fold CV — BCI-IV-2b")
    print(f"  Subjects: {subjects} | Splits: {N_SPLITS}")
    print(f"{'═'*60}\n")

    all_rows = []
    for subject in subjects:
        all_rows.extend(evaluate_subject(subject))

    df = pd.DataFrame(all_rows)
    _save_tables(df)
    _plot_cv(df)
    print(f"\n  Figures: {CV_FIGS}")


def main():
    parser = argparse.ArgumentParser(description="Within-train CV — BCI-IV-2b")
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    args = parser.parse_args()
    run(args.subjects if args.subjects else SUBJECTS)


if __name__ == "__main__":
    main()
