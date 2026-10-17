"""
diagnostic_db_heatmap.py
--------------------------
Diagnostic only — does not touch the production pipeline or cached files.

Generates the grand-average ERSP heatmap in REAL dB units (not the [0,1]
normalised scale the CNN consumes), with a diverging colour scale centred at
0 dB, using the 'jet' colormap. This is the format the thesis advisor asked
for directly: a time-frequency heatmap with a signed dB colour bar, matching
the classic EEGLAB-style ERSP plot convention.

Also useful as a comparison tool: run with --cmap to try different
colormaps. 'jet' gives noticeably sharper perceptual contrast than smooth
diverging maps like RdBu_r for the modest (2-6 dB) magnitudes typical of
motor-imagery ERD/ERS - see EXPERIMENTS.md "Fourth Diagnostic Finding".

Usage:
    python src/diagnostic_db_heatmap.py                  # jet, +-6dB, pooled 9 subjects
    python src/diagnostic_db_heatmap.py --cmap turbo
    python src/diagnostic_db_heatmap.py --clip 4.0
    python src/diagnostic_db_heatmap.py --subject 4       # single subject instead of pooled
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SUBJECTS, CHANNELS, N_CHANNELS, ERSP_FMIN, ERSP_FMAX,
    ANALYSIS_TMIN, ANALYSIS_TMAX, DATA_PROC, FIGURES_DIR,
    IMG_FREQ_BINS, IMG_TIME_BINS,
)

DB_CLIP = 6.0  # matches the +-6dB clip already baked into the stored [0,1] tensors


def main():
    parser = argparse.ArgumentParser(description="Grand-average ERSP heatmap in real dB units")
    parser.add_argument("--cmap", type=str, default="jet")
    parser.add_argument("--clip", type=float, default=DB_CLIP)
    parser.add_argument("--subject", type=int, default=None,
                        help="Single subject (1-9) instead of the 9-subject pooled average")
    args = parser.parse_args()

    subjects = [args.subject] if args.subject is not None else SUBJECTS

    sum_db = {0: np.zeros((N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS)),
             1: np.zeros((N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS))}
    counts = {0: 0, 1: 0}

    for subj in subjects:
        for suf in ("T", "E"):
            p = DATA_PROC / f"S{subj:02d}{suf}-ersp.npz"
            if not p.exists():
                continue
            d = np.load(p)
            X, y = d["X"], d["y"]
            for cls in (0, 1):
                mask = y == cls
                if mask.sum() == 0:
                    continue
                db = X[mask] * (2 * DB_CLIP) - DB_CLIP  # undo [0,1] normalisation -> real dB
                sum_db[cls] += db.sum(axis=0)
                counts[cls] += int(mask.sum())

    fig, axes = plt.subplots(N_CHANNELS, 2, figsize=(10, 4 * N_CHANNELS))
    title_scope = f"Sujeto S{args.subject:02d}" if args.subject is not None else "Grand-average — 9 sujetos pooled"
    fig.suptitle(
        f"ERSP en dB reales — {title_scope} — colormap '{args.cmap}'",
        fontsize=11, fontweight="bold"
    )
    for ch_i, ch_name in enumerate(CHANNELS[:N_CHANNELS]):
        for cls_i, (cls, name) in enumerate([(0, "Left"), (1, "Right")]):
            avg_db = sum_db[cls][ch_i] / counts[cls]
            ax = axes[ch_i, cls_i]
            im = ax.imshow(
                avg_db, aspect="auto", origin="lower", cmap=args.cmap,
                vmin=-args.clip, vmax=args.clip,
                extent=[ANALYSIS_TMIN, ANALYSIS_TMAX, ERSP_FMIN, ERSP_FMAX]
            )
            ax.axvline(0, color="black", lw=1, ls="--")
            ax.set_title(f"{ch_name} — {name} (n={counts[cls]})", fontsize=9)
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Frecuencia (Hz)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ERSP (dB)")
    plt.tight_layout()

    scope_tag = f"S{args.subject:02d}" if args.subject is not None else "grand_average"
    out = FIGURES_DIR / f"ersp_{scope_tag}_dB_heatmap_{args.cmap}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
