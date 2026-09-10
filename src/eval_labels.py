"""
eval_labels.py
--------------
True class labels for the BCI-IV-2b evaluation sessions (4 and 5).

The evaluation GDF files (B0x04E.gdf, B0x05E.gdf) do NOT contain the
left/right cue markers (769/770). Instead every trial is marked with the
generic events 781 (feedback) and 783 (cue *unknown*). The BCI Competition
IV deliberately withheld the true evaluation labels, distributing them
separately.

This module retrieves those true labels from MOABB's copy of the dataset
(`BNCI2014_004`), which bundles the official evaluation labels, and exposes
them per subject and per session in chronological trial order. MOABB is used
ONLY as the source of truth for labels — the EEG signal pipeline continues to
operate on the local GDF files.

Session mapping (MOABB ↔ BCI-IV-2b protocol):
    MOABB '3test'  →  session 4  (B0x04E.gdf)
    MOABB '4test'  →  session 5  (B0x05E.gdf)

Label convention (matches config.CLASS_NAMES): 0 = Left, 1 = Right.

Usage
-----
    from eval_labels import get_eval_labels
    labels = get_eval_labels(1)        # {4: array([...]), 5: array([...])}

    # Force a refresh of the cache (re-download via MOABB):
    python src/eval_labels.py --rebuild
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import mne

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, SUBJECTS

mne.set_log_level("ERROR")

# MOABB test-session key → BCI-IV-2b session number
MOABB_SESSION_MAP = {"3test": 4, "4test": 5}

# MOABB annotation label → class index (0 = Left, 1 = Right)
LABEL_MAP = {"left_hand": 0, "right_hand": 1}

CACHE_PATH = DATA_PROC / "eval_labels.npz"


def _extract_from_moabb() -> dict:
    """
    Download (once) and extract true evaluation labels via MOABB.

    Returns
    -------
    dict: {subject: {session_number: np.ndarray(labels)}}
          labels are 0=Left, 1=Right in chronological trial order.
    """
    from moabb.datasets import BNCI2014_004

    ds = BNCI2014_004()
    all_labels = {}

    print("  Fetching evaluation labels via MOABB (BNCI2014_004)...")
    for subject in SUBJECTS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = ds.get_data(subjects=[subject])

        subj_labels = {}
        for moabb_sess, sess_num in MOABB_SESSION_MAP.items():
            raw = data[subject][moabb_sess]["0"]
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            inv = {v: k for k, v in event_id.items()}
            # Chronological label sequence
            seq = [LABEL_MAP[inv[code]] for code in events[:, 2]]
            subj_labels[sess_num] = np.array(seq, dtype=np.int64)

        n4 = len(subj_labels[4])
        n5 = len(subj_labels[5])
        print(f"    S{subject:02d}: session 4 = {n4} trials, "
              f"session 5 = {n5} trials")
        all_labels[subject] = subj_labels

    return all_labels


def _save_cache(all_labels: dict):
    """Flatten the nested dict to a single .npz for fast reloading."""
    flat = {}
    for subject, sessions in all_labels.items():
        for sess_num, labels in sessions.items():
            flat[f"S{subject:02d}_sess{sess_num}"] = labels
    np.savez_compressed(str(CACHE_PATH), **flat)
    print(f"  Cached labels to: {CACHE_PATH.name}")


def _load_cache() -> dict:
    """Reload the nested dict from the .npz cache."""
    data = np.load(str(CACHE_PATH))
    all_labels = {}
    for key in data.files:
        # key format: "S01_sess4"
        subj = int(key[1:3])
        sess = int(key.split("sess")[1])
        all_labels.setdefault(subj, {})[sess] = data[key]
    return all_labels


def build_cache(rebuild: bool = False) -> dict:
    """
    Return the full label dict, building the MOABB cache if needed.
    """
    if CACHE_PATH.exists() and not rebuild:
        return _load_cache()
    all_labels = _extract_from_moabb()
    _save_cache(all_labels)
    return all_labels


def get_eval_labels(subject: int, rebuild: bool = False) -> dict:
    """
    Return the true evaluation labels for one subject.

    Parameters
    ----------
    subject : int (1-9)
    rebuild : if True, force MOABB re-download instead of using the cache

    Returns
    -------
    dict: {4: np.ndarray, 5: np.ndarray}  — labels (0=Left, 1=Right)
          in chronological trial order for sessions 4 and 5.
    """
    all_labels = build_cache(rebuild=rebuild)
    if subject not in all_labels:
        raise KeyError(f"No evaluation labels found for subject {subject}.")
    return all_labels[subject]


def main():
    parser = argparse.ArgumentParser(
        description="Build/inspect BCI-IV-2b evaluation labels (MOABB)"
    )
    parser.add_argument("--rebuild", action="store_true",
                        help="Force re-download via MOABB, ignoring the cache")
    args = parser.parse_args()

    print("\n══════════════════════════════════════════════")
    print("  BCI-IV-2b — Evaluation label retrieval")
    print("══════════════════════════════════════════════\n")

    all_labels = build_cache(rebuild=args.rebuild)

    print("\n  ── Summary ──")
    for subject in SUBJECTS:
        s = all_labels[subject]
        for sess in (4, 5):
            y = s[sess]
            print(f"  S{subject:02d} session {sess}: "
                  f"{len(y)} trials (left={int((y == 0).sum())}, "
                  f"right={int((y == 1).sum())})")
    print(f"\n  Cache: {CACHE_PATH}")


if __name__ == "__main__":
    main()
