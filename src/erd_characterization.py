"""
erd_characterization.py
--------------------------
ERD/ERS characterization using MNE's native TFR machinery, replacing the
hand-rolled STFT/baseline/epsilon pipeline in ersp.py:

  1. Morlet wavelet TFR (Epochs.compute_tfr) + apply_baseline(mode="logratio"),
     instead of custom STFT + manual baseline/epsilon handling. MNE's
     "logratio" mode computes log10(power / baseline_power); multiplying by
     10 recovers the classic ERSP definition in dB,
     10*log10(power/baseline), the same formula used in ersp.py, so results
     are directly comparable.
  2. Per-subject average computed FIRST (trials pooled only within a
     subject, via compute_tfr(average=True)), THEN a grand-average across
     the 9 subject-level maps (unweighted mean, n=9) - not a single average
     over all pooled trials, which would let subjects with more trials
     dominate and would not support subject-level statistics.
  3. Cluster-based permutation test (one-sample, against 0 = "no change
     from baseline") across the 9 subject-level maps, to mark which
     time-frequency pixels are statistically significant - not just
     visually apparent.

This is a characterization tool, independent of the classification
pipeline (ersp.py / train*.py / the FBCSP+SpectNet ensemble), which is
paused and left exactly as documented in EXPERIMENTS.md.

Usage:
    python src/erd_characterization.py                       # all channels, both classes
    python src/erd_characterization.py --channels C3 C4
    python src/erd_characterization.py --subjects 1 2 3       # faster subset
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROC, FIGURES_DIR, SUBJECTS, CHANNELS, N_CHANNELS,
    BASELINE, ANALYSIS_TMIN, ANALYSIS_TMAX, ERSP_FMIN, ERSP_FMAX,
)

mne.set_log_level("WARNING")

FREQS = np.arange(ERSP_FMIN, ERSP_FMAX + 1, 1.0)   # 8..30 Hz, 1 Hz steps
N_CYCLES = FREQS / 2.0                              # standard variable-cycle Morlet
DECIM = 4                                           # 250Hz -> 62.5Hz effective, plenty for 8-30Hz
CLASS_NAMES = {"left": "Left", "right": "Right"}
ALPHA = 0.05
N_PERMUTATIONS = 2000

OUT_DIR = FIGURES_DIR / "erd_characterization"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_subject_epochs(subject: int) -> mne.Epochs:
    """All available sessions (1-5) for one subject, pooled - this tool
    characterizes ERD/ERS, it does not hold out a classifier test set."""
    parts = []
    for suf in ("T", "E"):
        p = DATA_PROC / f"S{subject:02d}{suf}-epo.fif"
        if p.exists():
            parts.append(mne.read_epochs(str(p), verbose=False))
    epochs = mne.concatenate_epochs(parts)
    epochs.pick(CHANNELS[:N_CHANNELS])
    return epochs


def subject_average_tfr(epochs: mne.Epochs, cls_name: str):
    """
    Per-subject average ERSP (dB) for one class.

    average=True computes the trial-averaged power directly (not per-trial
    dB values averaged afterwards) - this is the standard ERSP definition:
    ERSP(f,t) = 10*log10( mean_trials(power(f,t)) / mean_trials(baseline) ).

    Returns an AverageTFR, data shape (n_channels, n_freqs, n_times),
    cropped to the analysis window (same ANALYSIS_TMIN..ANALYSIS_TMAX as
    the production ersp.py pipeline).
    """
    ep_cls = epochs[cls_name]
    tfr = ep_cls.compute_tfr(
        method="morlet", freqs=FREQS, n_cycles=N_CYCLES,
        average=True, return_itc=False, use_fft=True, decim=DECIM,
    )
    tfr.apply_baseline(baseline=BASELINE, mode="logratio")
    tfr.data *= 10.0  # logratio -> 10*log10(power/baseline) = classic ERSP dB
    tfr.crop(tmin=ANALYSIS_TMIN, tmax=ANALYSIS_TMAX)
    return tfr


def compute_all_subjects(subjects: list):
    """Returns (results, times, freqs) where
    results[cls_name]["maps"] is an array (n_subjects, n_channels, n_freqs, n_times).
    """
    results = {cls: {"maps": []} for cls in CLASS_NAMES}
    times = freqs = None
    for subj in subjects:
        print(f"  Subject S{subj:02d}...")
        epochs = load_subject_epochs(subj)
        for cls_name in CLASS_NAMES:
            if cls_name not in epochs.event_id:
                continue
            tfr = subject_average_tfr(epochs, cls_name)
            results[cls_name]["maps"].append(tfr.data)
            if times is None:
                times, freqs = tfr.times, tfr.freqs
    for cls_name in CLASS_NAMES:
        results[cls_name]["maps"] = np.stack(results[cls_name]["maps"], axis=0)
    return results, times, freqs


def cluster_test_significance(subject_maps_1ch: np.ndarray):
    """
    subject_maps_1ch: (n_subjects, n_freqs, n_times) for ONE channel/class.

    One-sample cluster-based permutation test against 0 (no ERD/ERS
    relative to baseline). Returns (mask, min_cluster_p):
    mask is a boolean array (n_freqs, n_times), True where the pixel
    belongs to a cluster with p < ALPHA.
    """
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        subject_maps_1ch, n_permutations=N_PERMUTATIONS, tail=0,
        out_type="mask", seed=42, verbose=False,
    )
    mask = np.zeros(subject_maps_1ch.shape[1:], dtype=bool)
    for cl, pv in zip(clusters, cluster_pv):
        if pv < ALPHA:
            mask |= cl
    min_pv = float(cluster_pv.min()) if len(cluster_pv) else 1.0
    return mask, min_pv


def plot_channel_class(grand_avg, sig_mask, times, freqs, ch_name, cls_label,
                       n_subjects, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        grand_avg, aspect="auto", origin="lower", cmap="jet",
        vmin=-6, vmax=6, extent=[times[0], times[-1], freqs[0], freqs[-1]]
    )
    ax.axvline(0, color="white", linewidth=1.0, linestyle="--")
    if sig_mask.any():
        ax.contour(times, freqs, sig_mask.astype(float),
                  levels=[0.5], colors="black", linewidths=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(
        f"{ch_name} — {cls_label} (grand-average, n={n_subjects} subjects)\n"
        f"Contour: cluster-corrected p<{ALPHA} vs. baseline"
    )
    plt.colorbar(im, ax=ax, label="ERSP (dB)")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="ERD/ERS characterization (MNE)")
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    parser.add_argument("--channels", type=str, nargs="+",
                        default=CHANNELS[:N_CHANNELS], choices=CHANNELS[:N_CHANNELS])
    args = parser.parse_args()

    print(f"\n{'='*60}\n  ERD/ERS Characterization — MNE Morlet TFR\n{'='*60}")
    t0 = time.time()
    results, times, freqs = compute_all_subjects(args.subjects)
    print(f"\n  TFR computation done in {time.time()-t0:.0f}s")

    n_subjects = len(args.subjects)
    summary_rows = []
    for cls_name, cls_label in CLASS_NAMES.items():
        maps = results[cls_name]["maps"]  # (n_subjects, n_channels, n_freqs, n_times)
        if maps.size == 0:
            continue
        for ch_name in args.channels:
            ch_i = CHANNELS.index(ch_name)
            subj_maps = maps[:, ch_i, :, :]            # (n_subjects, n_freqs, n_times)
            grand_avg = subj_maps.mean(axis=0)          # (n_freqs, n_times)
            sig_mask, min_pv = cluster_test_significance(subj_maps)
            out_path = OUT_DIR / f"erd_{ch_name}_{cls_name}.png"
            plot_channel_class(grand_avg, sig_mask, times, freqs,
                              ch_name, cls_label, n_subjects, out_path)
            summary_rows.append({
                "channel": ch_name, "class": cls_label,
                "min_cluster_p": min_pv,
                "pct_significant": 100 * sig_mask.mean(),
            })

    print(f"\n  {'Channel':>8} {'Class':>8} {'min cluster p':>15} {'% sig. pixels':>15}")
    for r in summary_rows:
        print(f"  {r['channel']:>8} {r['class']:>8} {r['min_cluster_p']:>15.4f} "
              f"{r['pct_significant']:>14.1f}%")

    print(f"\n  Figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
