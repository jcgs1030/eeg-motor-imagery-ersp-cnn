"""
experiment_augmentation.py
-----------------------------
Exploratory: does simple data augmentation (Gaussian noise + time-shift,
frequency axis untouched - bands stay fixed at 8-30Hz) recover any of
ShallowConvNet's collapsed accuracy after the ERSP epsilon fix? Small
per-subject training sets (~340 trials) are a plausible contributor to the
collapse (see EXPERIMENTS.md "Second Diagnostic Finding"); augmentation is
the standard remedy for that specific problem.

Does not touch any committed result file.

Usage:
    python src/experiment_augmentation.py                    # default: noise+shift
    python src/experiment_augmentation.py --subjects 1 2 3 4  # faster subset
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import SUBJECTS, DEVICE
from train_subject_specific import train_one_subject
from augmentation import Compose, GaussianNoise, TimeShift


def main():
    parser = argparse.ArgumentParser(description="Data augmentation experiment")
    parser.add_argument("--model", type=str, default="shallowconvnet")
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--noise_std", type=float, default=0.03)
    parser.add_argument("--max_shift", type=int, default=6)
    args = parser.parse_args()

    transform = Compose([
        GaussianNoise(std=args.noise_std),
        TimeShift(max_shift=args.max_shift),
    ])

    print(f"Augmentation experiment: {args.model}, noise_std={args.noise_std}, "
          f"max_shift={args.max_shift}, subjects={args.subjects}\n")

    baseline_accs, aug_accs = [], []
    for subj in args.subjects:
        r_base = train_one_subject(args.model, subj, args.device, save_outputs=False)
        r_aug  = train_one_subject(args.model, subj, args.device, train_transform=transform, save_outputs=False)
        baseline_accs.append(r_base["accuracy"])
        aug_accs.append(r_aug["accuracy"])
        print(f"  S{subj:02d}: baseline={r_base['accuracy']:.1%}  "
              f"augmented={r_aug['accuracy']:.1%}  "
              f"(kappa {r_base['kappa']:.3f} -> {r_aug['kappa']:.3f})")

    baseline_accs = np.array(baseline_accs)
    aug_accs = np.array(aug_accs)
    print("\n== Summary ==")
    print(f"  Baseline (no augmentation): {baseline_accs.mean():.1%} ± {baseline_accs.std():.1%}")
    print(f"  Augmented                 : {aug_accs.mean():.1%} ± {aug_accs.std():.1%}")
    print(f"  Mean diff: {(aug_accs.mean()-baseline_accs.mean())*100:+.1f}pp")


if __name__ == "__main__":
    main()
