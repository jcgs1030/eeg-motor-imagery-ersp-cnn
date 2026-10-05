"""
experiment_ensemble.py
-------------------------
Exploratory ensemble of FBCSP+LDA (best classical method, multi-band
log-variance on raw filtered epochs) and SpectNet pooled (best CNN method,
full time-frequency ERSP). Does not touch any production result file.

Both classifiers are evaluated on the SAME per-subject test trials, in the
SAME trial order (both read from the identical S{subj}E-epo.fif file, FBCSP
only crops a copy of the array for its own feature window - no reordering
happens), so softmax/predict_proba outputs are directly comparable and can
be averaged per trial.

Usage:
    python src/experiment_ensemble.py                 # simple average, all 9 subjects
    python src/experiment_ensemble.py --weight 0.6     # 0.6*FBCSP + 0.4*SpectNet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import mne
import torch
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, cohen_kappa_score

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROC, SUBJECTS, SFREQ, N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS, DEVICE
)
from ersp import compute_ersp_image
from models import get_model
from train_fbcsp import (
    load_epochs, extract_fbcsp_features, EVENT_LABEL_MAP, MI_TMIN, MI_TMAX
)

mne.set_log_level("ERROR")


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cnn_proba_for_subject(subject: int, model) -> tuple[np.ndarray, np.ndarray]:
    """Full (uncropped) epochs, same order as FBCSP's test set, ERSP-encoded."""
    path = DATA_PROC / f"S{subject:02d}E-epo.fif"
    epochs = mne.read_epochs(str(path), verbose=False)
    X = epochs.get_data()  # (N, C, T) full EPOCH_TMIN..EPOCH_TMAX
    code_to_label = {v: EVENT_LABEL_MAP[k] for k, v in epochs.event_id.items()}
    y = np.array([code_to_label[ev] for ev in epochs.events[:, 2]])

    imgs = np.zeros((len(X), N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS), dtype=np.float32)
    for i in range(len(X)):
        for ch in range(N_CHANNELS):
            imgs[i, ch] = compute_ersp_image(X[i, ch], sfreq=SFREQ)

    with torch.no_grad():
        logits = model(torch.from_numpy(imgs)).numpy()
    return softmax(logits), y


def compute_ensemble_per_subject(weight: float = 0.5, subjects: list = None) -> list[dict]:
    """Core computation, reusable from other scripts (e.g. plotting).
    Returns a list of per-subject dicts with FBCSP / SpectNet / ensemble
    accuracy and kappa."""
    subjects = subjects or SUBJECTS

    ckpt = torch.load("results/spectnet_all.pth", map_location="cpu", weights_only=False)
    model = get_model("spectnet", n_channels=N_CHANNELS, n_freq=IMG_FREQ_BINS,
                      n_time=IMG_TIME_BINS, n_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    rows = []
    for subj in subjects:
        # load_epochs() crops to MI_TMIN..MI_TMAX internally (see train_fbcsp.py)
        X_train, y_train = load_epochs(subj, "T")
        X_test_fbcsp, y_test_fbcsp = load_epochs(subj, "E")

        feats_train, feats_test = extract_fbcsp_features(X_train, y_train, X_test_fbcsp)
        clf = LinearDiscriminantAnalysis(solver="svd")
        clf.fit(feats_train, y_train)
        proba_fbcsp = clf.predict_proba(feats_test)

        proba_cnn, y_test_cnn = cnn_proba_for_subject(subj, model)

        assert np.array_equal(y_test_fbcsp, y_test_cnn), \
            f"S{subj:02d}: trial order mismatch between FBCSP and CNN test sets!"
        y_test = y_test_fbcsp

        proba_ens = weight * proba_fbcsp + (1 - weight) * proba_cnn

        pred_fbcsp = proba_fbcsp.argmax(axis=1)
        pred_cnn   = proba_cnn.argmax(axis=1)
        pred_ens   = proba_ens.argmax(axis=1)

        rows.append({
            "subject": subj,
            "acc_fbcsp": accuracy_score(y_test, pred_fbcsp),
            "acc_cnn":   accuracy_score(y_test, pred_cnn),
            "acc_ens":   accuracy_score(y_test, pred_ens),
            "kappa_fbcsp": cohen_kappa_score(y_test, pred_fbcsp),
            "kappa_cnn":   cohen_kappa_score(y_test, pred_cnn),
            "kappa_ens":   cohen_kappa_score(y_test, pred_ens),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="FBCSP+LDA / SpectNet ensemble")
    parser.add_argument("--weight", type=float, default=0.5,
                        help="Weight on FBCSP in the average (default 0.5 = simple average)")
    args = parser.parse_args()

    rows = compute_ensemble_per_subject(weight=args.weight)
    for row in rows:
        print(f"  S{row['subject']:02d}: FBCSP={row['acc_fbcsp']:.1%}  SpectNet={row['acc_cnn']:.1%}  "
              f"Ensemble={row['acc_ens']:.1%}")

    acc_fbcsp = np.array([r["acc_fbcsp"] for r in rows])
    acc_cnn   = np.array([r["acc_cnn"] for r in rows])
    acc_ens   = np.array([r["acc_ens"] for r in rows])

    print("\n== Summary (mean over 9 subjects) ==")
    print(f"  FBCSP+LDA         : {acc_fbcsp.mean():.1%} ± {acc_fbcsp.std():.1%}")
    print(f"  SpectNet (pooled) : {acc_cnn.mean():.1%} ± {acc_cnn.std():.1%}")
    print(f"  Ensemble (w={args.weight}) : {acc_ens.mean():.1%} ± {acc_ens.std():.1%}")

    w_stat, w_p = wilcoxon(acc_ens, acc_fbcsp)
    print(f"\n  Wilcoxon Ensemble vs. FBCSP+LDA alone: p={w_p:.3f} "
          f"(mean diff={100*(acc_ens.mean()-acc_fbcsp.mean()):+.1f}pp)")
    w_stat2, w_p2 = wilcoxon(acc_ens, acc_cnn)
    print(f"  Wilcoxon Ensemble vs. SpectNet alone : p={w_p2:.3f} "
          f"(mean diff={100*(acc_ens.mean()-acc_cnn.mean()):+.1f}pp)")


if __name__ == "__main__":
    main()
