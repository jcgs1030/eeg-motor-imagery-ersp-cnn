"""
diagnostic_classic_erd_curve.py
---------------------------------
Diagnostic only — does not touch the production pipeline or cached files.

Generates the classic Pfurtscheller-style ERD/ERS curve (% power change vs.
baseline, over time, per frequency band) as a complement to the 2D ERSP
heatmaps used elsewhere in this project. The heatmaps' y-axis is frequency
(always positive) and ERD/ERS is encoded in colour; this plot instead puts
% power change directly on the y-axis, matching the literature convention
(e.g. Pfurtscheller & Lopes da Silva, 1999) and going negative for ERD.

Requires the epsilon fix in src/ersp.py (see EXPERIMENTS.md "Second
Diagnostic Finding — ERSP Epsilon Bias") — with the original fixed 1e-12
epsilon, band power values were biased far more negative than any real
ERD/ERS magnitude.

Usage:
    python src/diagnostic_classic_erd_curve.py                 # grand-average, all subjects
    python src/diagnostic_classic_erd_curve.py --subject 4      # single subject
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROC, FIGURES_DIR, SUBJECTS, SFREQ,
    STFT_WIN_LEN, STFT_HOP, EPOCH_TMIN, BASELINE, ANALYSIS_TMIN, ANALYSIS_TMAX,
)

mne.set_log_level("ERROR")

CH_IDX = 0  # C3


def compute_band_curves(sig_trials: np.ndarray, fmin: float, fmax: float):
    """% power change vs. baseline, band-averaged, one curve per trial."""
    curves = []
    t_curve = None
    for trial in sig_trials:
        freqs, times_stft, Zxx = stft(
            trial, fs=SFREQ, window="hann", nperseg=STFT_WIN_LEN,
            noverlap=STFT_WIN_LEN - STFT_HOP, padded=True
        )
        power = np.abs(Zxx) ** 2
        t_abs = times_stft + EPOCH_TMIN
        bl_mask = (t_abs >= BASELINE[0]) & (t_abs <= BASELINE[1])
        an_mask = (t_abs >= ANALYSIS_TMIN) & (t_abs <= ANALYSIS_TMAX)
        fmask = (freqs >= fmin) & (freqs <= fmax)
        eps = power.max() * 1e-12
        baseline_mean = power[fmask][:, bl_mask].mean() + eps
        band_power_t = power[fmask][:, an_mask].mean(axis=0)
        curves.append((band_power_t - baseline_mean) / baseline_mean * 100)
        if t_curve is None:
            t_curve = t_abs[an_mask]
    return np.array(curves), t_curve


def plot_classic_erd_curve(subjects: list, label: str, save_fig: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (fmin, fmax, band_name) in zip(axes, [(8, 13, "Mu (8-13Hz)"), (14, 30, "Beta (14-30Hz)")]):
        for cls_name, color in [("right", "#c0392b"), ("left", "#2980b9")]:
            all_curves, t_curve = [], None
            for subj in subjects:
                for suf in ("T", "E"):
                    p = DATA_PROC / f"S{subj:02d}{suf}-epo.fif"
                    if not p.exists():
                        continue
                    epochs = mne.read_epochs(str(p), verbose=False)
                    if cls_name not in epochs.event_id:
                        continue
                    sig = epochs[cls_name].get_data()[:, CH_IDX, :]
                    curves, t = compute_band_curves(sig, fmin, fmax)
                    all_curves.append(curves)
                    t_curve = t
            all_curves = np.concatenate(all_curves, axis=0)
            mean_curve = all_curves.mean(axis=0)
            sem = all_curves.std(axis=0) / np.sqrt(len(all_curves))
            ax.plot(t_curve, mean_curve, color=color, lw=1.8,
                    label=f"{cls_name.capitalize()} (n={len(all_curves)})")
            ax.fill_between(t_curve, mean_curve - sem, mean_curve + sem, color=color, alpha=0.15)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(f"{band_name} — {label}, canal C3")
        ax.set_xlabel("Tiempo (s, relativo al cue)")
        ax.set_ylabel("% cambio de potencia vs. baseline\n(negativo = ERD, positivo = ERS)")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Curva clásica ERD/ERS (estilo Pfurtscheller) — {label}, C3, Left vs Right", fontweight="bold")
    plt.tight_layout()

    if save_fig:
        tag = "grandaverage" if len(subjects) > 1 else f"S{subjects[0]:02d}"
        out = FIGURES_DIR / f"erd_curve_classic_{tag}_C3.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"    Figure saved to: {out.name}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Classic ERD/ERS curve (Pfurtscheller-style)")
    parser.add_argument("--subject", type=int, default=None, help="Single subject (1-9); default: grand-average of all")
    args = parser.parse_args()

    if args.subject:
        plot_classic_erd_curve([args.subject], f"S{args.subject:02d}")
    else:
        plot_classic_erd_curve(SUBJECTS, "Grand-average, 9 sujetos")


if __name__ == "__main__":
    main()
