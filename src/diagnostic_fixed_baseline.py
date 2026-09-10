"""
diagnostic_fixed_baseline.py
-----------------------------
Diagnostic only — does NOT touch the production pipeline or the cached
*-ersp.npz files.

Root cause under test: compute_ersp_image() (src/ersp.py) estimates the
baseline power by reusing STFT frames from the trial's own STFT. Each STFT
frame is STFT_WIN_LEN=256 samples (1.024s) wide, but BASELINE=(-0.5, 0.0) is
only 0.5s long — shorter than a single STFT window. So every "baseline"
frame already leaks post-cue signal into the reference power:

    frame centered at t=0.000s (local) -> spans [-0.500, +0.012]s (absolute)
    frame centered at t=0.256s (local) -> spans [-0.756, +0.268]s (absolute)

This script recomputes the baseline power independently, via a Welch
periodogram over ONLY the pre-cue samples (no sliding window that can
extend past t=0), interpolated onto the same frequency bins as the trial's
STFT. Everything else (STFT of the full trial, dB formula, ROI crop, resize,
+-6dB clip) is identical to compute_ersp_image(), so the two are directly
comparable.

Usage:
    python src/diagnostic_fixed_baseline.py --suffix both
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from scipy.signal import stft, welch

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROC, FIGURES_DIR, SUBJECTS, CHANNELS, N_CHANNELS, SFREQ,
    EPOCH_TMIN, EPOCH_TMAX,
    STFT_WIN_LEN, STFT_HOP,
    ERSP_FMIN, ERSP_FMAX,
    IMG_FREQ_BINS, IMG_TIME_BINS,
)
from ersp import _resize_2d

mne.set_log_level("WARNING")


def compute_ersp_image_fixed_baseline(epoch_data: np.ndarray,
                                      baseline_data: np.ndarray,
                                      sfreq: float = SFREQ) -> np.ndarray:
    freqs, times_stft, Zxx = stft(
        epoch_data, fs=sfreq, window="hann",
        nperseg=STFT_WIN_LEN, noverlap=STFT_WIN_LEN - STFT_HOP, padded=True
    )
    power = np.abs(Zxx) ** 2

    # Baseline power estimated ONLY from the pre-cue segment, independent of
    # the trial STFT's sliding window (which is longer than the baseline).
    freqs_bl, psd_bl = welch(baseline_data, fs=sfreq,
                             nperseg=len(baseline_data), noverlap=0)
    baseline_mean = np.interp(freqs, freqs_bl, psd_bl)[:, None] + 1e-12

    ersp = 10 * np.log10(power / baseline_mean + 1e-12)

    freq_mask = (freqs >= ERSP_FMIN) & (freqs <= ERSP_FMAX)
    ersp_roi = ersp[freq_mask, :]
    ersp_resized = _resize_2d(ersp_roi, IMG_FREQ_BINS, IMG_TIME_BINS)

    DB_CLIP = 6.0
    ersp_clipped = np.clip(ersp_resized, -DB_CLIP, DB_CLIP)
    ersp_norm = (ersp_clipped + DB_CLIP) / (2.0 * DB_CLIP)
    return ersp_norm.astype(np.float32)


def plot_grand_average_fixed_baseline(subjects=None, suffix="both", save_fig=True):
    subjects = subjects or SUBJECTS
    suffixes = ["T", "E"] if suffix == "both" else [suffix]

    sum_by_class = {0: np.zeros((N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS)),
                    1: np.zeros((N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS))}
    n_trials_pooled = {0: 0, 1: 0}
    n_subjects_used = 0
    bl_end = int(abs(EPOCH_TMIN) * SFREQ)

    for subj in subjects:
        loaded_any = False
        for suf in suffixes:
            epo_path = DATA_PROC / f"S{subj:02d}{suf}-epo.fif"
            if not epo_path.exists():
                continue
            epochs = mne.read_epochs(str(epo_path), verbose=False)

            for cls_name, cls_label in [("left", 0), ("right", 1)]:
                if cls_name not in epochs.event_id:
                    continue
                ep_data = epochs[cls_name].get_data()
                for trial_idx in range(ep_data.shape[0]):
                    for ch_idx in range(ep_data.shape[1]):
                        signal = ep_data[trial_idx, ch_idx, :]
                        baseline = signal[:bl_end]
                        img = compute_ersp_image_fixed_baseline(signal, baseline, sfreq=SFREQ)
                        sum_by_class[cls_label][ch_idx] += img
                    n_trials_pooled[cls_label] += 1
            loaded_any = True
        n_subjects_used += int(loaded_any)

    if n_subjects_used == 0:
        print("  No epoch .fif files found. Run preprocessing.py first.")
        return

    fig, axes = plt.subplots(N_CHANNELS, 2, figsize=(10, 4 * N_CHANNELS))
    fig.suptitle(
        f"Grand-average ERSP (FIXED baseline, diagnostic) — {n_subjects_used} subjects pooled "
        f"({'sessions 1-5' if suffix == 'both' else ('sessions 1-3' if suffix == 'T' else 'sessions 4-5')})\n"
        f"(columns: Left n={n_trials_pooled[0]} | Right n={n_trials_pooled[1]} — rows: C3, Cz, C4)",
        fontsize=11, fontweight="bold"
    )

    for ch_i, ch_name in enumerate(CHANNELS[:N_CHANNELS]):
        for cls_i, (cls_label, cls_name) in enumerate([(0, "Left"), (1, "Right")]):
            ax = axes[ch_i, cls_i]
            grand_avg = sum_by_class[cls_label][ch_i] / n_trials_pooled[cls_label]
            im = ax.imshow(
                grand_avg, aspect="auto", origin="lower",
                cmap="RdYlBu_r", vmin=0, vmax=1,
                extent=[EPOCH_TMIN, EPOCH_TMAX, ERSP_FMIN, ERSP_FMAX]
            )
            ax.axvline(0, color="white", linewidth=1.0, linestyle="--")
            ax.axhspan(8, 13, alpha=0.15, color="cyan")
            ax.axhspan(14, 30, alpha=0.10, color="yellow")
            ax.set_title(f"{ch_name} — {cls_name} (n={n_trials_pooled[cls_label]})", fontsize=9)
            ax.set_ylabel("Frequency (Hz)", fontsize=8)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label="Norm. ERSP").ax.tick_params(labelsize=6)

    plt.tight_layout()

    if save_fig:
        fig_path = FIGURES_DIR / f"ersp_grand_average_stft_fixedbaseline_{suffix}.png"
        fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        print(f"    Figure saved to: {fig_path.name}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Diagnostic: fixed-baseline STFT grand average")
    parser.add_argument("--suffix", type=str, default="both", choices=["T", "E", "both"])
    args = parser.parse_args()
    plot_grand_average_fixed_baseline(suffix=args.suffix)


if __name__ == "__main__":
    main()
