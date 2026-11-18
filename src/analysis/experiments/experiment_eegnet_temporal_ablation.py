"""
experiment_eegnet_temporal_ablation.py
---------------------------------------
Exploratory — does NOT touch the production checkpoints, CSVs or figures
(every run uses `save_outputs=False`). Writes only its own CSV under
`results/metrics/`.

Question under test. "EEGNet Temporal Kernel Scaling at 32 Bins"
(EXPERIMENTS.md) fixed a real defect — at `IMG_TIME_BINS=32` the 128-bin
kernel and pooling lengths left a SINGLE time bin before the classifier — and
subject-specific accuracy then *fell* 2.9pp. That result was confounded: the
fix moved the time axis from 1 bin to 4 AND cut parameters from 6,098 to
1,970, because the kernels themselves shrank. Either could explain the drop.

The confound is separable, because EEGNet's parameter count is dominated by
its two kernel lengths and is almost independent of its pooling:

    conv1    : n_channels * F1 * filter_time_length      =  24 * k1
    sep_conv : (F1*D) * F2 * sep_filter_length           = 256 * k2
    fc       : 16 * final_time_bins * n_classes + 2

Changing only the pooling moves `final_time_bins` from 1 to 4, which adds a
mere 96 parameters to `fc`. So a 2x2 design varies one factor at a time:

                        kernels 64/16        kernels 16/4
    pools 4/8 -> 1 bin  (1)  6,098 params    (3)  1,874 params
    pools 2/4 -> 4 bins (2)  6,194 params    (4)  1,970 params

    (1) = the original 32-bin model (the defect)
    (4) = the shipped fix
    (1) vs (2) = the temporal axis, at ~constant capacity (+96 params)
    (3) vs (4) = the same, at low capacity
    (1) vs (3) = capacity, at a constant single time bin
    (2) vs (4) = the same, at 4 time bins

Protocol: identical to Experiment 2 (subject-specific, sessions 1-3 train /
4-5 test, `RANDOM_SEED`, same splits and early stopping), so condition (1)
reproduces the pre-fix EEGNet row and (4) the current one. Only the
subject-specific setting is run: it is where the 2.9pp drop appeared, and it
is the only one of the two that yields a 9-value vector for a paired test
(the pooled model is a single run and was flat at +0.2pp either way).

Usage:
    python src/analysis/experiments/experiment_eegnet_temporal_ablation.py
    python src/analysis/experiments/experiment_eegnet_temporal_ablation.py --subjects 1 4
"""

import argparse
import time

import pandas as pd
from scipy.stats import wilcoxon

from src.config import DEVICE, METRICS_DIR, SUBJECTS
from src.p3_classification.cnn.train_subject_specific import train_one_subject

# ── The 2x2 grid ─────────────────────────────────────────────────────────────
# (label, filter_time_length, sep_filter_length, pool1, pool2)
# (label, filter_time_length, sep_filter_length, pool1, pool2, stratify)
CONDITIONS = [
    ("1-bin / big kernels (pre-fix)",   64, 16, 4, 8, True),
    ("4-bin / big kernels",             64, 16, 2, 4, True),
    ("1-bin / small kernels",           16,  4, 4, 8, True),
    ("4-bin / small kernels (shipped)", 16,  4, 2, 4, True),
    # Reference point, outside the 2x2: the pre-fix architecture on the
    # UNSTRATIFIED split, which is what the committed pre-fix row was trained
    # with (the stratified split became the default in the external-review
    # pass, between that row being committed and the retraining). Including it
    # separates a third factor that would otherwise contaminate any
    # before/after comparison against the committed numbers.
    ("1-bin / big kernels, unstratified split (committed pre-fix row)",
     64, 16, 4, 8, False),
]


def run_condition(label: str, k1: int, k2: int, p1: int, p2: int,
                  stratify: bool, subjects: list, device_str: str) -> list:
    print(f"\n{'─' * 70}")
    print(f"  {label}   kernels={k1}/{k2}  pools={p1}/{p2}  "
          f"stratify={stratify}")
    print(f"{'─' * 70}")
    model_kwargs = {
        "filter_time_length": k1,
        "sep_filter_length": k2,
        "pool1_length": p1,
        "pool2_length": p2,
    }
    rows = []
    for subject in subjects:
        r = train_one_subject(
            "eegnet", subject, device_str,
            save_outputs=False,          # never clobber the committed results
            model_kwargs=model_kwargs,
            stratify=stratify,
        )
        rows.append({
            "condition": label,
            "kernels": f"{k1}/{k2}",
            "pools": f"{p1}/{p2}",
            "stratify": stratify,
            "subject": subject,
            "accuracy": r["accuracy"],
            "kappa": r["kappa"],
            "best_epoch": r["best_epoch"],
            "n_params": r["n_params"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="EEGNet 2x2 ablation: temporal bins vs. capacity at 32 bins"
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    print("\n══════════════════════════════════════════════════════")
    print("  EEGNet temporal-axis vs. capacity ablation (22x32)")
    print(f"  Subjects: {args.subjects}")
    print("══════════════════════════════════════════════════════")

    t0 = time.time()
    rows = []
    for label, k1, k2, p1, p2, strat in CONDITIONS:
        rows += run_condition(label, k1, k2, p1, p2, strat,
                              args.subjects, args.device)
    df = pd.DataFrame(rows)

    out_csv = METRICS_DIR / "eegnet_temporal_ablation.csv"
    df.to_csv(str(out_csv), index=False)

    # ── Summary ──
    print(f"\n\n  ══ Mean ± std across {len(args.subjects)} subjects ══\n")
    print(f"  {'Condition':34s} {'kernels':>8} {'pools':>6} {'params':>7} "
          f"{'accuracy':>16} {'kappa':>7}")
    order = [c[0] for c in CONDITIONS]
    for label in order:
        g = df[df.condition == label]
        print(f"  {label:34s} {g.kernels.iloc[0]:>8} {g.pools.iloc[0]:>6} "
              f"{int(g.n_params.iloc[0]):>7,} "
              f"{g.accuracy.mean():>8.1%} ± {g.accuracy.std():<5.1%} "
              f"{g.kappa.mean():>7.3f}")

    # ── Paired contrasts (same subjects, same seed) ──
    acc = {label: df[df.condition == label].sort_values("subject").accuracy.values
           for label in order}

    print(f"\n  ══ Paired contrasts (Wilcoxon, n={len(args.subjects)}) ══\n")
    contrasts = [
        (order[0], order[1], "temporal axis, ~constant capacity (+96 params)"),
        (order[2], order[3], "temporal axis, at low capacity"),
        (order[0], order[2], "capacity, at a constant 1 time bin"),
        (order[1], order[3], "capacity, at a constant 4 time bins"),
        (order[0], order[3], "pre-fix vs shipped (both factors, the confound)"),
        (order[4], order[0], "the stratified-split change alone, pre-fix architecture"),
        (order[4], order[3], "committed pre-fix row vs shipped (all three factors)"),
    ]
    for a, b, what in contrasts:
        d = acc[b] - acc[a]
        try:
            _, p = wilcoxon(acc[b], acc[a])
        except ValueError:
            p = float("nan")
        print(f"  {what}")
        print(f"    {a}  ->  {b}")
        print(f"    delta = {d.mean():+.1%}   p = {p:.3f}   "
              f"higher on {int((d > 0).sum())}/{len(d)} subjects\n")

    print(f"  Saved: {out_csv}")
    print(f"  Total time: {(time.time() - t0) / 60:.1f} min\n")


if __name__ == "__main__":
    main()
