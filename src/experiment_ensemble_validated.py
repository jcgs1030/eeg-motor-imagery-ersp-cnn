"""
experiment_ensemble_validated.py
------------------------------------
Fixes a methodological issue flagged by the advisor: in experiment_ensemble.py,
the ensemble weight `w` was chosen by sweeping it and picking whichever value
gave the best accuracy on the TEST set (sessions 4-5) - the same set used to
report the final number. That is leakage: the reported 78.8% is biased
upward by having tuned a hyperparameter against it.

This script selects w on a VALIDATION split carved out of the TRAINING
sessions (1-3) instead, and evaluates the final, fixed ensemble on the test
set exactly once.

Per subject, per run:
  1. Split sessions 1-3 (T) into T-train / T-val (stratified, test_size=
     VAL_SPLIT, seed=<this run's seed>) - same split fraction/seed convention
     used for CNN training elsewhere in the project.
  2. Fit FBCSP+LDA on T-train only.
  3. Evaluate FBCSP+LDA and SpectNet pooled (existing frozen checkpoint) on
     T-val -> sweep w, pick the value that maximises MEAN validation
     accuracy across the 9 subjects (one w for the whole population, not
     one per subject - matches how it was reported originally).
  4. Refit FBCSP+LDA on the FULL T split (train+val) - standard practice
     once a hyperparameter has been chosen on a held-out slice: use all
     available training data for the model that will actually be tested.
  5. Evaluate the ensemble at the CHOSEN w on E (test, sessions 4-5) once.

The whole thing (steps 1-5) is repeated for N_RUNS different seeds, per the
advisor's request for statistical robustness. Only the T-train/T-val split
is randomised across runs - E (test, sessions 4-5) is loaded fresh every
run via load_full_and_fbcsp_arrays(subj, "E"), which involves no split at
all, so it is byte-identical across all runs and is never touched by any
run's seed. The final number is reported as mean +/- std ACROSS RUNS (on
top of the existing mean +/- std across subjects within each run), so both
sources of variability are visible separately.

Caveat (documented, not fixed here): SpectNet pooled's own checkpoint
(results/spectnet_all.pth) was trained before this validation split existed,
on the full T data (with its own internal, different 15% val split used
only for early stopping) - so it has technically seen T-val trials during
its own training. Retraining SpectNet with a clean 3-way split per run is
out of scope here (scoped explicitly as a lighter re-run: only the ensemble
weight selection is repeated across seeds, SpectNet itself is kept fixed);
what this script addresses is specifically the weight-selection leakage,
now checked for robustness across 5 independent train/val splits.

Usage:
    python src/experiment_ensemble_validated.py
    python src/experiment_ensemble_validated.py --weights 0.3 0.4 0.5 0.6 0.7 0.8
    python src/experiment_ensemble_validated.py --n_runs 5
    python src/experiment_ensemble_validated.py --seeds 42 43 44 45 46
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
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROC, SUBJECTS, SFREQ, N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS,
    DEVICE, VAL_SPLIT, RANDOM_SEED,
)
from ersp import compute_ersp_image
from models import get_model
from train_fbcsp import extract_fbcsp_features, EVENT_LABEL_MAP, MI_TMIN, MI_TMAX

mne.set_log_level("ERROR")

DEFAULT_WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def load_full_and_fbcsp_arrays(subject: int, suffix: str):
    """Loads one subject/session-group once; returns the FULL (uncropped)
    epoch array (for ERSP/CNN) and the MI-window-cropped array (for FBCSP),
    in the SAME trial order, plus labels."""
    path = DATA_PROC / f"S{subject:02d}{suffix}-epo.fif"
    epochs = mne.read_epochs(str(path), verbose=False)
    code_to_label = {v: EVENT_LABEL_MAP[k] for k, v in epochs.event_id.items()}
    y = np.array([code_to_label[ev] for ev in epochs.events[:, 2]])
    X_full = epochs.get_data()
    X_fbcsp = epochs.copy().crop(tmin=MI_TMIN, tmax=MI_TMAX).get_data()
    return X_full, X_fbcsp, y


def cnn_proba_from_full_epochs(X_full: np.ndarray, model) -> np.ndarray:
    """Computes ERSP images from a (N, C, T) full-epoch array and returns
    the model's softmax probabilities."""
    imgs = np.zeros((len(X_full), N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS), dtype=np.float32)
    for i in range(len(X_full)):
        for ch in range(N_CHANNELS):
            imgs[i, ch] = compute_ersp_image(X_full[i, ch], sfreq=SFREQ)
    with torch.no_grad():
        logits = model(torch.from_numpy(imgs)).numpy()
    return softmax(logits)


def select_weight_on_validation(subjects: list, weights: list, spectnet_model,
                                seed: int) -> tuple[float, dict]:
    """Returns (best_w, per_subject_val_data) where per_subject_val_data
    holds everything needed later to refit FBCSP on full T and to report
    validation accuracy per weight, per subject.

    seed controls ONLY the T-train/T-val split below - it never touches E
    (test), which is loaded separately in evaluate_on_test() with no split
    at all."""
    val_acc_by_w = {w: [] for w in weights}
    per_subject_val_data = {}

    for subj in subjects:
        X_full, X_fbcsp, y = load_full_and_fbcsp_arrays(subj, "T")
        idx = np.arange(len(y))
        idx_train, idx_val = train_test_split(
            idx, test_size=VAL_SPLIT, stratify=y, random_state=seed
        )

        # FBCSP fit on T-train only
        feats_train, feats_val = extract_fbcsp_features(
            X_fbcsp[idx_train], y[idx_train], X_fbcsp[idx_val]
        )
        clf = LinearDiscriminantAnalysis(solver="svd")
        clf.fit(feats_train, y[idx_train])
        proba_fbcsp_val = clf.predict_proba(feats_val)

        proba_cnn_val = cnn_proba_from_full_epochs(X_full[idx_val], spectnet_model)
        y_val = y[idx_val]

        for w in weights:
            proba_ens = w * proba_fbcsp_val + (1 - w) * proba_cnn_val
            acc = accuracy_score(y_val, proba_ens.argmax(axis=1))
            val_acc_by_w[w].append(acc)

        per_subject_val_data[subj] = {
            "X_full": X_full, "X_fbcsp": X_fbcsp, "y": y,  # full T, for refit
        }
        print(f"  S{subj:02d} validation done "
              f"(n_train={len(idx_train)}, n_val={len(idx_val)})")

    mean_val_acc = {w: float(np.mean(accs)) for w, accs in val_acc_by_w.items()}
    best_w = max(mean_val_acc, key=mean_val_acc.get)

    print("\n  == Mean validation accuracy by weight (sessions 1-3 only) ==")
    for w in weights:
        marker = "  <-- chosen" if w == best_w else ""
        print(f"    w={w:.1f}: {mean_val_acc[w]:.1%}{marker}")

    return best_w, per_subject_val_data


def evaluate_on_test(subjects: list, best_w: float, per_subject_val_data: dict,
                     spectnet_model) -> list[dict]:
    """Refits FBCSP on the FULL T split (train+val) per subject, then
    evaluates the ensemble at the fixed, validation-chosen weight on E
    (test) ONCE."""
    rows = []
    for subj in subjects:
        d = per_subject_val_data[subj]
        X_full_T, X_fbcsp_T, y_T = d["X_full"], d["X_fbcsp"], d["y"]

        X_full_E, X_fbcsp_E, y_E = load_full_and_fbcsp_arrays(subj, "E")

        feats_train, feats_test = extract_fbcsp_features(X_fbcsp_T, y_T, X_fbcsp_E)
        clf = LinearDiscriminantAnalysis(solver="svd")
        clf.fit(feats_train, y_T)
        proba_fbcsp_test = clf.predict_proba(feats_test)

        proba_cnn_test = cnn_proba_from_full_epochs(X_full_E, spectnet_model)

        proba_ens = best_w * proba_fbcsp_test + (1 - best_w) * proba_cnn_test

        pred_fbcsp = proba_fbcsp_test.argmax(axis=1)
        pred_cnn   = proba_cnn_test.argmax(axis=1)
        pred_ens   = proba_ens.argmax(axis=1)

        rows.append({
            "subject": subj,
            "acc_fbcsp": accuracy_score(y_E, pred_fbcsp),
            "acc_cnn":   accuracy_score(y_E, pred_cnn),
            "acc_ens":   accuracy_score(y_E, pred_ens),
            "kappa_fbcsp": cohen_kappa_score(y_E, pred_fbcsp),
            "kappa_cnn":   cohen_kappa_score(y_E, pred_cnn),
            "kappa_ens":   cohen_kappa_score(y_E, pred_ens),
        })
    return rows


def run_once(subjects: list, weights: list, spectnet_model, seed: int) -> dict:
    """Runs the full validation-weight-selection + test-evaluation procedure
    once, for one T-train/T-val seed. Returns a dict with the chosen w and
    the per-subject test rows for this run."""
    print(f"\n-- Run seed={seed} --")
    print("  Step 1: selecting w on validation (sessions 1-3 only, test set untouched)")
    best_w, per_subject_val_data = select_weight_on_validation(
        subjects, weights, spectnet_model, seed=seed
    )
    print(f"    Chosen weight: w={best_w:.1f}")

    print("  Step 2: refitting FBCSP on full T, evaluating ensemble on E (test) ONCE")
    rows = evaluate_on_test(subjects, best_w, per_subject_val_data, spectnet_model)

    acc_ens = np.array([r["acc_ens"] for r in rows])
    print(f"    Ensemble test accuracy this run: {acc_ens.mean():.1%} ± {acc_ens.std():.1%} "
          f"(across {len(subjects)} subjects)")

    return {"seed": seed, "best_w": best_w, "rows": rows}


def main():
    parser = argparse.ArgumentParser(
        description="FBCSP+LDA / SpectNet ensemble with weight chosen on validation, not test"
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    parser.add_argument("--weights", type=float, nargs="+", default=DEFAULT_WEIGHTS)
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of T-train/T-val splits to repeat, each with a "
                             "different seed, for statistical robustness. Test set "
                             "(sessions 4-5) is never re-split, regardless of n_runs.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Explicit list of seeds to use instead of "
                             "RANDOM_SEED..RANDOM_SEED+n_runs-1.")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else \
        [RANDOM_SEED + i for i in range(args.n_runs)]

    ckpt = torch.load("results/spectnet_all.pth", map_location="cpu", weights_only=False)
    spectnet_model = get_model("spectnet", n_channels=N_CHANNELS, n_freq=IMG_FREQ_BINS,
                               n_time=IMG_TIME_BINS, n_classes=2)
    spectnet_model.load_state_dict(ckpt["state_dict"])
    spectnet_model.eval()

    runs = [run_once(args.subjects, args.weights, spectnet_model, seed) for seed in seeds]

    print(f"\n{'='*60}\n  Per-run summary ({len(runs)} runs, test set identical across all)\n{'='*60}")
    run_acc_fbcsp, run_acc_cnn, run_acc_ens, run_kappa_ens = [], [], [], []
    for run in runs:
        acc_fbcsp = np.array([r["acc_fbcsp"] for r in run["rows"]])
        acc_cnn   = np.array([r["acc_cnn"]   for r in run["rows"]])
        acc_ens   = np.array([r["acc_ens"]   for r in run["rows"]])
        kappa_ens = np.array([r["kappa_ens"] for r in run["rows"]])
        print(f"  seed={run['seed']:>4}  w={run['best_w']:.1f}  "
              f"FBCSP={acc_fbcsp.mean():.1%}  SpectNet={acc_cnn.mean():.1%}  "
              f"Ensemble={acc_ens.mean():.1%} ± {acc_ens.std():.1%} (across subjects)  "
              f"kappa={kappa_ens.mean():.3f}")
        run_acc_fbcsp.append(acc_fbcsp.mean())
        run_acc_cnn.append(acc_cnn.mean())
        run_acc_ens.append(acc_ens.mean())
        run_kappa_ens.append(kappa_ens.mean())

    run_acc_fbcsp = np.array(run_acc_fbcsp)
    run_acc_cnn   = np.array(run_acc_cnn)
    run_acc_ens   = np.array(run_acc_ens)
    run_kappa_ens = np.array(run_kappa_ens)

    print(f"\n  == Final test-set summary, mean ± std ACROSS {len(runs)} RUNS "
          f"(each run's own number is already a mean across {len(args.subjects)} subjects) ==")
    print(f"  FBCSP+LDA         : {run_acc_fbcsp.mean():.1%} ± {run_acc_fbcsp.std():.1%}")
    print(f"  SpectNet (pooled) : {run_acc_cnn.mean():.1%} ± {run_acc_cnn.std():.1%}")
    print(f"  Ensemble          : {run_acc_ens.mean():.1%} ± {run_acc_ens.std():.1%}  "
          f"kappa={run_kappa_ens.mean():.3f} ± {run_kappa_ens.std():.3f}")

    # Wilcoxon on the LAST run's per-subject rows, kept for continuity with the
    # single-run version of this script - a significance test across only
    # n_runs=5 run-level means would be underpowered to be meaningful.
    last_rows = runs[-1]["rows"]
    acc_fbcsp_last = np.array([r["acc_fbcsp"] for r in last_rows])
    acc_cnn_last   = np.array([r["acc_cnn"]   for r in last_rows])
    acc_ens_last   = np.array([r["acc_ens"]   for r in last_rows])
    w_stat, w_p = wilcoxon(acc_ens_last, acc_fbcsp_last)
    print(f"\n  Wilcoxon Ensemble vs. FBCSP+LDA alone (last run, seed={runs[-1]['seed']}): "
          f"p={w_p:.3f} (mean diff={100*(acc_ens_last.mean()-acc_fbcsp_last.mean()):+.1f}pp)")
    w_stat2, w_p2 = wilcoxon(acc_ens_last, acc_cnn_last)
    print(f"  Wilcoxon Ensemble vs. SpectNet alone  (last run, seed={runs[-1]['seed']}): "
          f"p={w_p2:.3f} (mean diff={100*(acc_ens_last.mean()-acc_cnn_last.mean()):+.1f}pp)")


if __name__ == "__main__":
    main()
