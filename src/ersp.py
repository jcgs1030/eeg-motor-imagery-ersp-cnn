"""
ersp.py
-------
Generation of ERSP (Event-Related Spectral Perturbation) spectrograms
from preprocessed epochs of BCI-IV-2b.

Usage:
    python src/ersp.py                        # generate all subjects
    python src/ersp.py --subject 1            # subject 1 only
    python src/ersp.py --subject 1 --plot     # generate and visualise
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
    DATA_PROC, FIGURES_DIR, SUBJECTS, CHANNELS, N_CHANNELS, SFREQ,
    EPOCH_TMIN, EPOCH_TMAX,
    STFT_WIN_LEN, STFT_HOP,
    ERSP_FMIN, ERSP_FMAX,
    IMG_FREQ_BINS, IMG_TIME_BINS, IMG_SIZE,
    TRAIN_SUFFIX, EVAL_SUFFIX
)

mne.set_log_level("WARNING")


# ── Core ERSP ────────────────────────────────────────────────────────────────

def compute_ersp_image(epoch_data: np.ndarray,
                       baseline_data: np.ndarray,
                       sfreq: float = SFREQ) -> np.ndarray:
    """
    Compute the ERSP image for a single trial.

    ERSP(f, t) = 10 * log10( P(f, t) / P_baseline(f) )

    Parameters
    ----------
    epoch_data    : array (n_samples,) — full trial signal (baseline + MI)
    baseline_data : array (n_samples,) — pre-stimulus baseline window only
    sfreq         : sampling frequency

    Returns
    -------
    ersp_img : array (IMG_FREQ_BINS, IMG_TIME_BINS) — normalised ERSP image [0, 1]
    """
    # ── STFT of the full trial ──
    freqs, times_stft, Zxx = stft(
        epoch_data,
        fs=sfreq,
        window="hann",
        nperseg=STFT_WIN_LEN,
        noverlap=STFT_WIN_LEN - STFT_HOP,
        padded=True
    )
    power = np.abs(Zxx) ** 2  # (n_freqs_stft, n_times_stft)

    # ── Baseline mean from STFT time bins within the pre-stimulus window ──
    # baseline_data length / sfreq gives the baseline duration in seconds;
    # times_stft are seconds from signal start so we select bins up to that point.
    bl_duration = len(baseline_data) / sfreq
    bl_mask = times_stft <= bl_duration
    if bl_mask.sum() == 0:
        bl_mask[0] = True  # fallback: use first frame
    baseline_mean = power[:, bl_mask].mean(axis=-1, keepdims=True) + 1e-12

    # ── Divisive ERSP (in dB) ──
    ersp = 10 * np.log10(power / baseline_mean + 1e-12)

    # ── Select frequency range of interest ──
    freq_mask = (freqs >= ERSP_FMIN) & (freqs <= ERSP_FMAX)
    ersp_roi = ersp[freq_mask, :]    # (n_freq_roi, n_times_stft)

    # ── Resize to IMG_SIZE with interpolation ──
    ersp_resized = _resize_2d(ersp_roi, IMG_FREQ_BINS, IMG_TIME_BINS)

    # ── Normalise to [0, 1] ──
    vmin, vmax = ersp_resized.min(), ersp_resized.max()
    if vmax - vmin > 1e-8:
        ersp_norm = (ersp_resized - vmin) / (vmax - vmin)
    else:
        ersp_norm = np.zeros_like(ersp_resized)

    return ersp_norm.astype(np.float32)


def _resize_2d(arr: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Resize a 2D array to (n_rows, n_cols) using bilinear interpolation."""
    from scipy.ndimage import zoom
    zoom_r = n_rows / arr.shape[0]
    zoom_c = n_cols / arr.shape[1]
    return zoom(arr, (zoom_r, zoom_c), order=1)


# ── Per-subject processing ───────────────────────────────────────────────────

def generate_ersp_for_subject(subject: int, suffix: str,
                               save: bool = True) -> dict:
    """
    Generate ERSP spectrograms for all trials of one subject.

    Returns
    -------
    result : dict with keys:
        'X'       : array (n_trials, n_channels, IMG_FREQ_BINS, IMG_TIME_BINS)
        'y'       : array (n_trials,) — labels 0=left, 1=right
        'subject' : subject number
        'suffix'  : 'T' or 'E'
    """
    tag = f"S{subject:02d}{suffix}"
    epo_path = DATA_PROC / f"{tag}-epo.fif"

    if not epo_path.exists():
        raise FileNotFoundError(
            f"{epo_path.name} not found. "
            f"Run first: python src/preprocessing.py --subject {subject}"
        )

    print(f"\n  Generating ERSP for {tag}...")
    epochs = mne.read_epochs(str(epo_path), verbose=False)

    # Time indices for baseline and imagery window
    bl_start = 0
    bl_end   = int(abs(EPOCH_TMIN) * SFREQ)  # up to t=0

    X_list, y_list = [], []

    for cls_name, cls_label in [("left", 0), ("right", 1)]:
        if cls_name not in epochs.event_id:
            continue
        ep_data = epochs[cls_name].get_data()  # (n_trials, n_channels, n_times)

        for trial_idx in range(ep_data.shape[0]):
            trial_channels = []
            valid = True

            for ch_idx in range(ep_data.shape[1]):
                signal = ep_data[trial_idx, ch_idx, :]
                baseline = signal[:bl_end]

                ersp_img = compute_ersp_image(signal, baseline, sfreq=SFREQ)

                if ersp_img.shape != IMG_SIZE:
                    valid = False
                    break
                trial_channels.append(ersp_img)

            if valid and len(trial_channels) == ep_data.shape[1]:
                X_list.append(np.stack(trial_channels, axis=0))  # (C, F, T)
                y_list.append(cls_label)

    if not X_list:
        raise ValueError(f"Could not generate spectrograms for {tag}")

    X = np.array(X_list, dtype=np.float32)  # (N, C, F, T)
    y = np.array(y_list,  dtype=np.int64)   # (N,)

    n_left  = (y == 0).sum()
    n_right = (y == 1).sum()
    print(f"    ERSP generated: {len(X)} trials "
          f"(left: {n_left}, right: {n_right}) — shape: {X.shape}")

    result = {"X": X, "y": y, "subject": subject, "suffix": suffix}

    if save:
        out_path = DATA_PROC / f"{tag}-ersp.npz"
        np.savez_compressed(str(out_path), X=X, y=y,
                            subject=subject, suffix=suffix)
        print(f"    Saved to: {out_path.name}")

    return result


# ── Visualisation ────────────────────────────────────────────────────────────

def plot_ersp_examples(subject: int, suffix: str = TRAIN_SUFFIX,
                       n_examples: int = 3, save_fig: bool = True):
    """
    Visualise example ERSP images for one subject,
    showing n_examples trials per class.
    """
    tag = f"S{subject:02d}{suffix}"
    npz_path = DATA_PROC / f"{tag}-ersp.npz"

    if not npz_path.exists():
        print(f"  {npz_path.name} not found. "
              f"Run first: python src/ersp.py --subject {subject}")
        return

    data = np.load(str(npz_path))
    X, y = data["X"], data["y"]

    idx_left  = np.where(y == 0)[0][:n_examples]
    idx_right = np.where(y == 1)[0][:n_examples]

    n_rows = 2 * N_CHANNELS  # 2 classes × 3 channels
    n_cols = n_examples
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows))
    fig.suptitle(
        f"ERSP spectrograms — Subject {subject:02d} "
        f"({'Training' if suffix == 'T' else 'Evaluation'})\n"
        f"Rows: Left (C3/Cz/C4) | Right (C3/Cz/C4) — "
        f"X axis: time | Y axis: frequency (8–30 Hz)",
        fontsize=11, fontweight="bold"
    )

    cls_data = [(idx_left, "Left",  "#2C7BB6"),
                (idx_right, "Right", "#D7191C")]
    row = 0
    for indices, cls_name, color in cls_data:
        for ch_i, ch_name in enumerate(CHANNELS[:N_CHANNELS]):
            for col, trial_idx in enumerate(indices):
                ax = axes[row, col] if n_cols > 1 else axes[row]
                img = X[trial_idx, ch_i, :, :]  # (F, T)
                im = ax.imshow(
                    img, aspect="auto", origin="lower",
                    cmap="RdYlBu_r",
                    extent=[EPOCH_TMIN, EPOCH_TMAX, ERSP_FMIN, ERSP_FMAX]
                )
                if col == 0:
                    ax.set_ylabel(f"{cls_name}\n{ch_name}\nFrequency (Hz)",
                                  fontsize=8)
                if row == n_rows - 1:
                    ax.set_xlabel("Time (s)", fontsize=8)
                ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
                ax.tick_params(labelsize=7)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
            row += 1

    plt.tight_layout()

    if save_fig:
        fig_path = FIGURES_DIR / f"ersp_examples_{tag}.png"
        fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        print(f"    Figure saved to: {fig_path.name}")
        plt.close(fig)
    else:
        plt.show()


def plot_ersp_average(subject: int, suffix: str = TRAIN_SUFFIX,
                      save_fig: bool = True):
    """
    Visualise the average ERSP per class and channel for one subject.
    """
    tag = f"S{subject:02d}{suffix}"
    npz_path = DATA_PROC / f"{tag}-ersp.npz"

    if not npz_path.exists():
        print(f"  {npz_path.name} not found.")
        return

    data = np.load(str(npz_path))
    X, y = data["X"], data["y"]

    fig, axes = plt.subplots(N_CHANNELS, 2,
                             figsize=(10, 4 * N_CHANNELS))
    fig.suptitle(
        f"Average ERSP per class — Subject {subject:02d}\n"
        f"(columns: Left | Right — rows: C3, Cz, C4)",
        fontsize=11, fontweight="bold"
    )

    for ch_i, ch_name in enumerate(CHANNELS[:N_CHANNELS]):
        for cls_i, (cls_label, cls_name) in enumerate([(0, "Left"), (1, "Right")]):
            ax = axes[ch_i, cls_i]
            mask = y == cls_label
            if mask.sum() == 0:
                ax.set_visible(False)
                continue
            avg_ersp = X[mask, ch_i, :, :].mean(axis=0)
            im = ax.imshow(
                avg_ersp, aspect="auto", origin="lower",
                cmap="RdYlBu_r", vmin=0, vmax=1,
                extent=[EPOCH_TMIN, EPOCH_TMAX, ERSP_FMIN, ERSP_FMAX]
            )
            ax.axvline(0, color="white", linewidth=1.0, linestyle="--")
            ax.axhspan(8, 13, alpha=0.15, color="cyan")    # mu band
            ax.axhspan(14, 30, alpha=0.10, color="yellow") # beta band
            ax.set_title(f"{ch_name} — {cls_name} (n={mask.sum()})", fontsize=9)
            ax.set_ylabel("Frequency (Hz)", fontsize=8)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label="Norm. ERSP").ax.tick_params(labelsize=6)

    plt.tight_layout()

    if save_fig:
        fig_path = FIGURES_DIR / f"ersp_average_{tag}.png"
        fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        print(f"    Average figure saved to: {fig_path.name}")
        plt.close(fig)
    else:
        plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ERSP spectrogram generation — BCI-IV-2b"
    )
    parser.add_argument("--subject", type=str, default="all",
                        help="Subject number (1–9) or 'all'")
    parser.add_argument("--suffix", type=str, default="both",
                        choices=["T", "E", "both"])
    parser.add_argument("--plot", action="store_true",
                        help="Generate example and average figures")
    args = parser.parse_args()

    print("\n══════════════════════════════════════════════")
    print("  BCI-IV-2b — ERSP Spectrogram Module")
    print("══════════════════════════════════════════════")

    subjects = SUBJECTS if args.subject == "all" else [int(args.subject)]
    suffixes = ["T", "E"] if args.suffix == "both" else [args.suffix]

    for subj in subjects:
        for suf in suffixes:
            epo_path = DATA_PROC / f"S{subj:02d}{suf}-epo.fif"
            if not epo_path.exists():
                print(f"\n  S{subj:02d}{suf}: epochs not found — "
                      f"run preprocessing.py first.")
                continue
            try:
                generate_ersp_for_subject(subj, suf)
                if args.plot:
                    plot_ersp_examples(subj, suf)
                    plot_ersp_average(subj, suf)
            except Exception as e:
                print(f"\n  S{subj:02d}{suf}: ERROR — {e}")

    print("\n── ERSP generation complete ──\n")


if __name__ == "__main__":
    main()
