"""
preprocessing.py
----------------
Loading, filtering, artefact removal, and segmentation
of EEG signals from the BCI Competition IV - Dataset 2b.

Usage from terminal:
    python src/preprocessing.py --verify            # only verify files
    python src/preprocessing.py --subject 1         # process subject 1
    python src/preprocessing.py --subject all       # process all subjects
    python src/preprocessing.py --subject 1 --plot  # process and plot
"""

import argparse
import sys
import warnings
from pathlib import Path

import mne
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend (server-compatible)
import matplotlib.pyplot as plt

# Add src/ to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_RAW, DATA_PROC, FIGURES_DIR,
    SUBJECTS, TRAIN_SUFFIX, EVAL_SUFFIX, TRAIN_SESSIONS, TEST_SESSIONS,
    CHANNELS, SFREQ,
    EVENT_LEFT, EVENT_RIGHT, EVENT_LEFT_ONLINE, EVENT_RIGHT_ONLINE,
    EVENT_LABELS, CLASS_NAMES,
    FILT_LOW, FILT_HIGH, FILT_METHOD,
    EPOCH_TMIN, EPOCH_TMAX, BASELINE, REJECT_THRESH,
    ICA_N_COMPS, ICA_METHOD, ICA_SEED
)

SESSION_MAP = {TRAIN_SUFFIX: TRAIN_SESSIONS, EVAL_SUFFIX: TEST_SESSIONS}

# Canonical event IDs used for all sessions (GDF internal IDs differ between sessions)
CANONICAL_EVENT_ID = {"left": 1, "right": 2}

# Suppress MNE logs (keep only warnings and errors)
mne.set_log_level("WARNING")


# ── Utility functions ────────────────────────────────────────────────────────

def get_gdf_path(subject: int, session: int) -> Path:
    """Return the GDF file path for a given subject and session number (1–5)."""
    suffix = TRAIN_SUFFIX if session in TRAIN_SESSIONS else EVAL_SUFFIX
    return DATA_RAW / f"B{subject:02d}0{session}{suffix}.gdf"


def verify_dataset() -> bool:
    """
    Verify that all 45 BCI-IV-2b GDF files are present
    (9 subjects × 5 sessions).
    Returns True if the dataset is complete.
    """
    all_sessions = TRAIN_SESSIONS + TEST_SESSIONS
    print("\n── Verifying BCI-IV-2b dataset files (9 subjects × 5 sessions) ──")
    missing = []
    for subj in SUBJECTS:
        for ses in all_sessions:
            path = get_gdf_path(subj, ses)
            status = "OK" if path.exists() else "MISSING"
            symbol = "✓" if path.exists() else "✗"
            print(f"  {symbol} {path.name}  ({status})")
            if not path.exists():
                missing.append(path.name)

    if missing:
        print(f"\n  WARNING: {len(missing)} file(s) missing.")
        print(f"  Download the dataset from: https://www.bbci.de/competition/iv/download/")
        print(f"  and place the files in: {DATA_RAW}/")
        return False
    else:
        print(f"\n  Dataset complete: 45 GDF files found in {DATA_RAW}/")
        return True


def load_raw(subject: int, session: int) -> mne.io.Raw:
    """
    Load the GDF file for a subject/session and configure the EEG channels.

    Parameters
    ----------
    subject : int
        Subject number (1–9).
    session : int
        Session number (1–5).

    Returns
    -------
    raw : mne.io.Raw
        Raw object with data loaded and channels configured.
    """
    path = get_gdf_path(subject, session)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. "
            f"Verify the file is in {DATA_RAW}/"
        )

    # Load with MNE (includes EEG and EOG channels).
    # BCI-IV-2b GDF headers store hardware filter metadata (0.5–100 Hz) that
    # MNE interprets as highpass > lowpass, triggering a harmless RuntimeWarning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Highpass cutoff frequency.*greater than lowpass.*",
            category=RuntimeWarning,
        )
        raw = mne.io.read_raw_gdf(str(path), preload=True, verbose=False)

    # Select only channels of interest (C3, Cz, C4)
    # BCI-IV-2b names EEG channels as "EEG:C3", "EEG:Cz", "EEG:C4"
    # and EOG channels as "EOG:ch01", "EOG:ch02", "EOG:ch03"
    available = raw.ch_names
    eeg_picks = [ch for ch in available if any(c in ch for c in CHANNELS)]

    if len(eeg_picks) == 0:
        # Fallback: take the first 3 channels if names differ
        eeg_picks = available[:3]
        print(f"  Warning: channels not found by name. "
              f"Using: {eeg_picks}")

    # Keep only the selected EEG channels
    raw.pick_channels(eeg_picks)

    # Rename channels to standard format if needed
    rename_map = {}
    for ch in raw.ch_names:
        for std in CHANNELS:
            if std in ch and ch != std:
                rename_map[ch] = std
    if rename_map:
        raw.rename_channels(rename_map)

    # Set channel type to EEG
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

    return raw


def apply_filter(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Apply FIR band-pass filter (8–30 Hz) to retain
    mu and beta bands associated with motor imagery.
    """
    raw.filter(
        l_freq=FILT_LOW,
        h_freq=FILT_HIGH,
        method=FILT_METHOD,
        fir_window="hamming",
        verbose=False
    )
    return raw


def apply_ica(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Apply ICA to remove ocular and muscular artefacts.
    With 3 channels, ICA capacity is limited but applied
    as a signal hygiene step.
    """
    ica = mne.preprocessing.ICA(
        n_components=ICA_N_COMPS,
        method=ICA_METHOD,
        random_state=ICA_SEED,
        max_iter=200,
        verbose=False
    )
    ica.fit(raw, verbose=False)
    # With 3 channels, no component is excluded automatically
    # (would require separate EOG channels for correlation)
    ica.apply(raw, verbose=False)
    return raw


def extract_epochs(raw: mne.io.Raw) -> mne.Epochs:
    """
    Segment the signal into epochs aligned to motor imagery events
    from BCI-IV-2b.

    Events:
        769 → left hand  (class 0)
        770 → right hand (class 1)
    """
    # Extract events from the stimulus channel
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Filter only target events — training (769/770) or evaluation (781/783)
    left_keys  = [str(EVENT_LEFT),  str(EVENT_LEFT_ONLINE),  "769", "781"]
    right_keys = [str(EVENT_RIGHT), str(EVENT_RIGHT_ONLINE), "770", "783"]

    target_ids = {}
    for k in left_keys:
        if k in event_id:
            target_ids["left"] = event_id[k]
            break
    for k in right_keys:
        if k in event_id:
            target_ids["right"] = event_id[k]
            break

    if not target_ids:
        # Last-resort: scan by keyword
        for k, v in event_id.items():
            if "left" in str(k).lower():
                target_ids["left"] = v
            elif "right" in str(k).lower():
                target_ids["right"] = v

    if not target_ids:
        print(f"  Available events: {event_id}")
        raise ValueError(
            "Motor imagery events (769/770) not found. "
            "Check the event IDs in the GDF file."
        )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=target_ids,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=BASELINE,
        reject={"eeg": REJECT_THRESH},
        preload=True,
        verbose=False
    )

    return epochs


def normalize_event_ids(epochs: mne.Epochs) -> mne.Epochs:
    """Remap per-session GDF event IDs to canonical {left:1, right:2} so
    mne.concatenate_epochs does not fail on ID mismatches across sessions."""
    for label, new_id in CANONICAL_EVENT_ID.items():
        if label in epochs.event_id:
            old_id = epochs.event_id[label]
            if old_id != new_id:
                epochs.events[epochs.events[:, 2] == old_id, 2] = new_id
    epochs.event_id = {k: CANONICAL_EVENT_ID[k]
                       for k in CANONICAL_EVENT_ID if k in epochs.event_id}
    return epochs


def process_subject(subject: int, suffix: str,
                    apply_ica_flag: bool = False,
                    save: bool = True) -> mne.Epochs:
    """
    Full pipeline for one subject and one split (T/E).
    Loads and concatenates all sessions in the split.

    Parameters
    ----------
    subject       : subject number (1–9)
    suffix        : 'T' (sessions 1-3) or 'E' (sessions 4-5)
    apply_ica_flag: if True, apply ICA (slower on CPU)
    save          : if True, save epochs to data/processed/

    Returns
    -------
    epochs : mne.Epochs concatenated across all sessions of the split
    """
    tag = f"S{subject:02d}{suffix}"
    sessions = SESSION_MAP[suffix]
    print(f"\n  Processing {tag} (sessions {sessions})...")

    all_epochs = []
    for session in sessions:
        path = get_gdf_path(subject, session)
        if not path.exists():
            print(f"    Session {session}: {path.name} not found, skipped.")
            continue

        # 1. Load
        raw = load_raw(subject, session)
        print(f"    Session {session} loaded: {len(raw.ch_names)} channels, "
              f"{raw.n_times / raw.info['sfreq']:.1f} s")

        # 2. Filter
        raw = apply_filter(raw)

        # 3. ICA (optional, may be slow)
        if apply_ica_flag:
            raw = apply_ica(raw)

        # 4. Epoching
        try:
            epochs = extract_epochs(raw)
            epochs = normalize_event_ids(epochs)
            all_epochs.append(epochs)
            n_left  = len(epochs["left"])  if "left"  in epochs.event_id else 0
            n_right = len(epochs["right"]) if "right" in epochs.event_id else 0
            print(f"    Session {session} epochs: {len(epochs)} "
                  f"(left: {n_left}, right: {n_right})")
        except Exception as e:
            print(f"    Session {session}: ERROR — {e}")

    if not all_epochs:
        raise RuntimeError(f"No epochs could be extracted for {tag}.")

    epochs = mne.concatenate_epochs(all_epochs)
    n_total = len(epochs)
    n_left  = len(epochs["left"])  if "left"  in epochs.event_id else 0
    n_right = len(epochs["right"]) if "right" in epochs.event_id else 0
    print(f"    {tag} total: {n_total} epochs "
          f"(left: {n_left}, right: {n_right})")

    if save:
        out_path = DATA_PROC / f"{tag}-epo.fif"
        epochs.save(str(out_path), overwrite=True, verbose=False)
        print(f"    Saved to: {out_path.name}")

    return epochs


# ── Visualisation functions ───────────────────────────────────────────────────

def plot_subject_overview(subject: int, suffix: str = TRAIN_SUFFIX,
                          save_fig: bool = True):
    """
    Generate a 4-panel figure for one subject:
      (a) Raw signal (first 10 s)
      (b) Filtered signal (first 10 s)
      (c) Epoch average per class (channel C3)
      (d) Power spectrum per class (channel C3)
    """
    tag = f"S{subject:02d}{suffix}"
    print(f"\n  Generating overview for {tag}...")

    # Use the first session of the requested split for visualisation
    session = SESSION_MAP[suffix][0]
    raw = load_raw(subject, session)
    raw_filt = raw.copy()
    apply_filter(raw_filt)

    # Epochs
    epochs = extract_epochs(raw_filt.copy())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"BCI-IV-2b — Subject {subject:02d} ({'Training' if suffix == 'T' else 'Evaluation'})\n"
        f"Channel C3 | Filter {FILT_LOW}–{FILT_HIGH} Hz | "
        f"Epochs: {len(epochs)} ({EPOCH_TMIN} to {EPOCH_TMAX} s)",
        fontsize=12, fontweight="bold"
    )

    times_raw = raw.times[:int(10 * SFREQ)]   # first 10 s
    ch_idx = 0  # first channel (C3)

    # ── Panel (a): Raw signal ──
    ax = axes[0, 0]
    data_raw = raw.get_data(picks=[ch_idx])[0, :int(10 * SFREQ)] * 1e6  # in µV
    ax.plot(times_raw, data_raw, color="#2C7BB6", linewidth=0.6)
    ax.set_title("(a) Raw EEG signal — C3 (first 10 s)", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_xlim([0, 10])
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel (b): Filtered signal ──
    ax = axes[0, 1]
    data_filt = raw_filt.get_data(picks=[ch_idx])[0, :int(10 * SFREQ)] * 1e6
    ax.plot(times_raw, data_filt, color="#D7191C", linewidth=0.6)
    ax.set_title(f"(b) Filtered signal {FILT_LOW}–{FILT_HIGH} Hz — C3", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_xlim([0, 10])
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel (c): Epoch average per class ──
    ax = axes[1, 0]
    epoch_times = epochs.times
    colors_cls = {"left": "#2C7BB6", "right": "#D7191C"}
    for cls_name, color in colors_cls.items():
        if cls_name in epochs.event_id:
            ep_data = epochs[cls_name].get_data(picks=[ch_idx])[:, 0, :] * 1e6
            mean_ep = ep_data.mean(axis=0)
            std_ep  = ep_data.std(axis=0)
            ax.plot(epoch_times, mean_ep, color=color,
                    label=CLASS_NAMES.get(0 if cls_name == "left" else 1, cls_name),
                    linewidth=1.2)
            ax.fill_between(epoch_times, mean_ep - std_ep, mean_ep + std_ep,
                            color=color, alpha=0.15)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", label="Cue onset")
    ax.axhline(0, color="gray",  linewidth=0.4, linestyle=":")
    ax.set_title("(c) Epoch average per class — C3", fontsize=10)
    ax.set_xlabel("Time relative to onset (s)")
    ax.set_ylabel("Mean amplitude (µV) ± SD")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel (d): Power spectrum per class ──
    ax = axes[1, 1]
    for cls_name, color in colors_cls.items():
        if cls_name in epochs.event_id:
            ep_data = epochs[cls_name].get_data(picks=[ch_idx])[:, 0, :]
            # Welch PSD over the imagery window (0–4 s)
            tmin_idx = int((0.0 - EPOCH_TMIN) * SFREQ)
            tmax_idx = int((4.0 - EPOCH_TMIN) * SFREQ)
            ep_mi = ep_data[:, tmin_idx:tmax_idx]
            from scipy.signal import welch
            freqs_w, psd = welch(ep_mi, fs=SFREQ, nperseg=256,
                                 noverlap=128, axis=-1)
            psd_db = 10 * np.log10(psd.mean(axis=0) + 1e-12)
            mask = (freqs_w >= 4) & (freqs_w <= 40)
            ax.plot(freqs_w[mask], psd_db[mask], color=color,
                    label=CLASS_NAMES.get(0 if cls_name == "left" else 1, cls_name),
                    linewidth=1.2)

    ax.axvspan(8, 13, alpha=0.1, color="green", label="Mu band")
    ax.axvspan(14, 30, alpha=0.1, color="orange", label="Beta band")
    ax.set_title("(d) Power spectrum per class — C3 (0–4 s)", fontsize=10)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_fig:
        fig_path = FIGURES_DIR / f"overview_{tag}.png"
        fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        print(f"    Figure saved to: {fig_path.name}")
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_all_subjects_summary(suffix: str = TRAIN_SUFFIX):
    """
    Generate a summary figure with the number of valid epochs
    per subject and class, to detect problematic subjects.
    """
    print("\n── Epoch summary per subject ──")
    n_left_list, n_right_list = [], []
    subj_labels = []

    for subj in SUBJECTS:
        session = SESSION_MAP[suffix][0]
        path = get_gdf_path(subj, session)
        if not path.exists():
            print(f"  S{subj:02d}: session {session} not found, skipped.")
            continue
        try:
            raw = load_raw(subj, session)
            apply_filter(raw)
            epochs = extract_epochs(raw)
            nl = len(epochs["left"])  if "left"  in epochs.event_id else 0
            nr = len(epochs["right"]) if "right" in epochs.event_id else 0
            n_left_list.append(nl)
            n_right_list.append(nr)
            subj_labels.append(f"S{subj:02d}")
            print(f"  S{subj:02d}: left={nl:3d}, right={nr:3d}, total={nl+nr:3d}")
        except Exception as e:
            print(f"  S{subj:02d}: ERROR — {e}")

    if not subj_labels:
        print("  No data to plot.")
        return

    x = np.arange(len(subj_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_l = ax.bar(x - width/2, n_left_list,  width, label="Left",  color="#2C7BB6", alpha=0.85)
    bars_r = ax.bar(x + width/2, n_right_list, width, label="Right", color="#D7191C", alpha=0.85)
    ax.bar_label(bars_l, padding=2, fontsize=8)
    ax.bar_label(bars_r, padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(subj_labels)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Number of valid epochs")
    ax.set_title(
        f"BCI-IV-2b — Valid epochs per subject and class\n"
        f"({'Training (sessions 1–3)' if suffix == 'T' else 'Evaluation (sessions 4–5)'})",
        fontsize=11
    )
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    fig_path = FIGURES_DIR / f"epochs_summary_{suffix}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    print(f"\n  Summary figure saved to: {fig_path.name}")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="EEG signal preprocessing — BCI-IV-2b"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Only verify that GDF files are present"
    )
    parser.add_argument(
        "--subject", type=str, default=None,
        help="Subject number to process (1–9) or 'all'"
    )
    parser.add_argument(
        "--suffix", type=str, default="T", choices=["T", "E", "both"],
        help="Split to process: T (train), E (eval), both"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate visualisation figures"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Generate epoch summary figure per subject"
    )
    parser.add_argument(
        "--ica", action="store_true",
        help="Apply ICA for artefact removal (slower)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n══════════════════════════════════════════════")
    print("  BCI-IV-2b — Preprocessing Module")
    print("══════════════════════════════════════════════")

    # ── Verification ──
    if args.verify or (args.subject is None and not args.summary):
        ok = verify_dataset()
        if args.verify:
            return

    # ── Global summary ──
    if args.summary:
        suffixes = ["T", "E"] if args.suffix == "both" else [args.suffix]
        for s in suffixes:
            plot_all_subjects_summary(suffix=s)
        return

    # ── Process subject(s) ──
    if args.subject is not None:
        suffixes = ["T", "E"] if args.suffix == "both" else [args.suffix]

        if args.subject == "all":
            subjects = SUBJECTS
        else:
            try:
                subjects = [int(args.subject)]
            except ValueError:
                print(f"  ERROR: --subject must be a number 1–9 or 'all'")
                return

        print(f"\n── Processing subject(s): {subjects} | Split(s): {suffixes} ──")

        for subj in subjects:
            for suf in suffixes:
                try:
                    process_subject(subj, suf, apply_ica_flag=args.ica)
                    if args.plot:
                        plot_subject_overview(subj, suf)
                except Exception as e:
                    print(f"\n  S{subj:02d}{suf}: ERROR — {e}")

    print("\n── Preprocessing complete ──\n")


if __name__ == "__main__":
    main()
