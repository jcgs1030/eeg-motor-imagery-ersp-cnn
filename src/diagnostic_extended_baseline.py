"""
diagnostic_extended_baseline.py
---------------------------------
Diagnostic only — does NOT touch the production pipeline, cached .fif/.npz
files, or trained models.

Two changes vs. the previous diagnostic (diagnostic_fixed_baseline.py, which
did NOT fix the flat appearance):

1. Epochs are re-extracted with tmin=-3.0s instead of -0.5s (session 1-3
   fixation is 3.000s long, verified in verify_timeline.py), so the -0.5..0s
   window we actually analyse sits 2.5s away from the true start of the
   extracted signal. This avoids the STFT zero-padding taper (padded=True)
   biasing power estimates near the edge of the extracted epoch, which is
   exactly where the plotted baseline window fell before.
2. The baseline reference is the mean STFT power over -1.0..0.0s (a clean,
   1-second pre-cue window, now unaffected by edge tapering).
3. The grand-average plot uses a colour scale fit to the actual data range
   (percentile-based), not a fixed [0,1], since the true modulation may be
   a few tenths of a dB and get washed out by a wide fixed scale.

Usage:
    python src/diagnostic_extended_baseline.py --suffix T
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from scipy.signal import stft

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SUBJECTS, CHANNELS, N_CHANNELS, SFREQ, FIGURES_DIR,
    STFT_WIN_LEN, STFT_HOP, ERSP_FMIN, ERSP_FMAX,
    IMG_FREQ_BINS, IMG_TIME_BINS, TRAIN_SESSIONS,
)
from preprocessing import load_raw, apply_filter
from ersp import _resize_2d

mne.set_log_level("WARNING")

EXT_TMIN = -3.0   # full pre-cue fixation window (verified: trial_start->cue = 3.000s)
EXT_TMAX = 4.0
PLOT_TMIN = -0.5  # same analysis window as the production pipeline
PLOT_TMAX = 4.0
BL_REF_TMIN = -1.0   # clean 1s window for the baseline reference, far from the
BL_REF_TMAX = 0.0    # extraction edge at -3.0s


def extract_extended_epochs(subject: int, session: int):
    raw = load_raw(subject, session)
    apply_filter(raw)
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    target_ids = {}
    for k in ("769",):
        if k in event_id:
            target_ids["left"] = event_id[k]
    for k in ("770",):
        if k in event_id:
            target_ids["right"] = event_id[k]
    if not target_ids:
        return None

    epochs = mne.Epochs(
        raw, events, event_id=target_ids,
        tmin=EXT_TMIN, tmax=EXT_TMAX,
        baseline=None, reject={"eeg": 100e-6},
        preload=True, verbose=False
    )
    return epochs


def compute_ersp_extended(signal_full: np.ndarray, sfreq: float = SFREQ) -> np.ndarray:
    """signal_full spans EXT_TMIN..EXT_TMAX (7s). Baseline ref from BL_REF window,
    analysis/plot cropped to PLOT_TMIN..PLOT_TMAX."""
    freqs, times_stft, Zxx = stft(
        signal_full, fs=sfreq, window="hann",
        nperseg=STFT_WIN_LEN, noverlap=STFT_WIN_LEN - STFT_HOP, padded=True
    )
    power = np.abs(Zxx) ** 2
    # times_stft is 0..(EXT_TMAX-EXT_TMIN) seconds from the start of signal_full.
    t_abs = times_stft + EXT_TMIN  # convert to time relative to cue onset

    bl_mask = (t_abs >= BL_REF_TMIN) & (t_abs <= BL_REF_TMAX)
    baseline_mean = power[:, bl_mask].mean(axis=-1, keepdims=True) + 1e-12

    ersp = 10 * np.log10(power / baseline_mean + 1e-12)

    plot_mask = (t_abs >= PLOT_TMIN) & (t_abs <= PLOT_TMAX)
    ersp_plot = ersp[:, plot_mask]

    freq_mask = (freqs >= ERSP_FMIN) & (freqs <= ERSP_FMAX)
    ersp_roi = ersp_plot[freq_mask, :]
    ersp_resized = _resize_2d(ersp_roi, IMG_FREQ_BINS, IMG_TIME_BINS)
    return ersp_resized.astype(np.float32)  # raw dB, NOT clipped/scaled to [0,1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="T", choices=["T"])
    parser.add_argument("--subject", type=str, default="all")
    args = parser.parse_args()

    subjects = SUBJECTS if args.subject == "all" else [int(args.subject)]

    sum_db = {0: np.zeros((N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS)),
             1: np.zeros((N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS))}
    n_trials = {0: 0, 1: 0}

    for subj in subjects:
        for session in TRAIN_SESSIONS:
            epochs = extract_extended_epochs(subj, session)
            if epochs is None:
                continue
            for cls_name, cls_label in [("left", 0), ("right", 1)]:
                if cls_name not in epochs.event_id:
                    continue
                ep_data = epochs[cls_name].get_data()
                for trial_idx in range(ep_data.shape[0]):
                    for ch_idx in range(ep_data.shape[1]):
                        img_db = compute_ersp_extended(ep_data[trial_idx, ch_idx, :])
                        sum_db[cls_label][ch_idx] += img_db
                    n_trials[cls_label] += 1
        print(f"  S{subj:02d} done — pooled so far: left={n_trials[0]}, right={n_trials[1]}")

    fig, axes = plt.subplots(N_CHANNELS, 2, figsize=(10, 4 * N_CHANNELS))

    # Data-driven color scale: symmetric around 0 dB, sized to the actual
    # data range (not a fixed +-6dB clip), so a modest real modulation is
    # not washed out by an oversized fixed scale.
    all_vals = np.concatenate([
        (sum_db[c][ch] / n_trials[c]).ravel()
        for c in (0, 1) for ch in range(N_CHANNELS)
    ])
    vabs = np.percentile(np.abs(all_vals), 98)
    print(f"  Data-driven color scale: +-{vabs:.3f} dB (98th percentile of |grand-average dB|)")

    fig.suptitle(
        f"Grand-average ERSP — EXTENDED baseline (-1.0..0.0s ref, 3s fixation available)\n"
        f"{len(subjects)} subjects pooled, sessions 1-3 | color scale: +-{vabs:.2f} dB (data-driven)",
        fontsize=11, fontweight="bold"
    )

    for ch_i, ch_name in enumerate(CHANNELS[:N_CHANNELS]):
        for cls_i, (cls_label, cls_name) in enumerate([(0, "Left"), (1, "Right")]):
            ax = axes[ch_i, cls_i]
            grand_avg_db = sum_db[cls_label][ch_i] / n_trials[cls_label]
            im = ax.imshow(
                grand_avg_db, aspect="auto", origin="lower",
                cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                extent=[PLOT_TMIN, PLOT_TMAX, ERSP_FMIN, ERSP_FMAX]
            )
            ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
            ax.axhspan(8, 13, alpha=0.12, color="cyan")
            ax.axhspan(14, 30, alpha=0.08, color="yellow")
            ax.set_title(f"{ch_name} — {cls_name} (n={n_trials[cls_label]})", fontsize=9)
            ax.set_ylabel("Frequency (Hz)", fontsize=8)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label="ERSP (dB)").ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "ersp_grand_average_stft_extendedbaseline_T.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    print(f"    Figure saved to: {fig_path.name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
