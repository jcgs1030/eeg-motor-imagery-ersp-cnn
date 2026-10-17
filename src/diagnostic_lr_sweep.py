"""
diagnostic_lr_sweep.py
------------------------
Diagnostic only — does NOT overwrite results/subject_specific/metrics/*.csv
or any committed result. Calls train_one_subject() directly for a single
architecture across a few learning rates and reports results separately.

Motivation: ShallowConvNet collapsed to near-chance after the ERSP epsilon
fix (see EXPERIMENTS.md "Second Diagnostic Finding") while EEGNet/SpectNet
were only mildly affected. All three architectures share the same global
LEARNING_RATE (1e-3) in config.py - never tuned per architecture. This
tests whether a lower learning rate (gentler updates against
ShallowConvNet's square->log nonlinearity, which amplifies large early
gradients more than ReLU-based architectures) recovers any of the lost
accuracy.

Usage:
    python src/diagnostic_lr_sweep.py                      # default LR grid
    python src/diagnostic_lr_sweep.py --lrs 1e-3 3e-4 1e-4
    python src/diagnostic_lr_sweep.py --subjects 1 4 3 2   # fast subset
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import SUBJECTS, DEVICE
from train_subject_specific import train_one_subject


def main():
    parser = argparse.ArgumentParser(description="Learning-rate sweep for one architecture")
    parser.add_argument("--model", type=str, default="shallowconvnet")
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-4, 1e-4])
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    print(f"LR sweep for {args.model}, subjects={args.subjects}, lrs={args.lrs}\n")

    all_results = {}
    for lr in args.lrs:
        accs, kappas = [], []
        for subj in args.subjects:
            r = train_one_subject(args.model, subj, args.device, lr=lr, save_outputs=False)
            accs.append(r["accuracy"])
            kappas.append(r["kappa"])
            print(f"  lr={lr:.0e} S{subj:02d}: acc={r['accuracy']*100:.1f}%  kappa={r['kappa']:.3f}")
        all_results[lr] = (np.mean(accs), np.std(accs), np.mean(kappas))

    print("\n== Summary ==")
    print(f"{'LR':>8} | {'Mean Acc':>10} | {'Std':>6} | {'Mean Kappa':>10}")
    for lr, (m, s, k) in all_results.items():
        print(f"{lr:8.0e} | {m*100:9.1f}% | {s*100:5.1f}% | {k:10.3f}")


if __name__ == "__main__":
    main()
