"""
verify_timeline.py
-------------------
Empirically verifies the BCI-IV-2b trial timeline directly from the raw
event markers in the GDF files, instead of relying only on the dataset
documentation.

Event codes used (see config.py / EXPERIMENTS.md):
    768 = trial start (fixation cross + warning beep)
    769/770 = cue onset, class known (training sessions 1-3)
    783     = cue onset, class unknown in the file (evaluation sessions 4-5)
    781     = continuous feedback onset (evaluation sessions only)

Usage:
    python src/verify_timeline.py                # all 9 subjects, T + E
    python src/verify_timeline.py --subject 1     # single subject
"""

import argparse
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import SUBJECTS, TRAIN_SESSIONS, TEST_SESSIONS, SFREQ, RESULTS_DIR
from preprocessing import load_raw

mne.set_log_level("ERROR")

OUT_PATH = RESULTS_DIR / "trial_timeline_verification.txt"


def verify_file(subject: int, session: int, lines: list) -> None:
    is_eval = session in TEST_SESSIONS
    tag = f"S{subject:02d} session {session} ({'eval' if is_eval else 'train'})"

    try:
        raw = load_raw(subject, session)
    except FileNotFoundError as e:
        lines.append(f"{tag}: SKIPPED — {e}")
        return

    events, event_id = mne.events_from_annotations(raw, verbose=False)

    start_id = event_id.get("768")
    cue_id = event_id.get("783") if is_eval else None
    if not is_eval:
        cue_id = event_id.get("769") or event_id.get("770")
    fb_id = event_id.get("781") if is_eval else None

    if start_id is None or cue_id is None:
        lines.append(f"{tag}: missing 768 or cue event — available: {event_id}")
        return

    starts = events[events[:, 2] == start_id][:, 0]
    # cue events: both 769 and 770 for training
    if is_eval:
        cues = events[events[:, 2] == cue_id][:, 0]
    else:
        left_id, right_id = event_id.get("769"), event_id.get("770")
        cue_mask = np.isin(events[:, 2], [i for i in (left_id, right_id) if i is not None])
        cues = events[cue_mask][:, 0]
        cues.sort()

    n = min(len(starts), len(cues))
    start_to_cue = (cues[:n] - starts[:n]) / SFREQ
    iti = np.diff(starts) / SFREQ

    msg = (
        f"{tag}: n_trials={n} | "
        f"trial_start->cue = {start_to_cue.mean():.3f}s (std={start_to_cue.std():.4f}) | "
        f"inter-trial interval mean={iti.mean():.2f}s (min={iti.min():.2f}, max={iti.max():.2f})"
    )

    if is_eval and fb_id is not None:
        fbs = events[events[:, 2] == fb_id][:, 0]
        n_fb = min(n, len(fbs))
        cue_to_fb = (fbs[:n_fb] - cues[:n_fb]) / SFREQ
        msg += f" | cue->feedback = {cue_to_fb.mean():.3f}s (std={cue_to_fb.std():.4f})"

    lines.append(msg)
    print(msg)


def main():
    parser = argparse.ArgumentParser(
        description="Empirically verify the BCI-IV-2b trial timeline from raw GDF events"
    )
    parser.add_argument("--subject", type=str, default="all",
                        help="Subject number (1-9) or 'all'")
    args = parser.parse_args()

    subjects = SUBJECTS if args.subject == "all" else [int(args.subject)]

    print("\n══════════════════════════════════════════════")
    print("  BCI-IV-2b — Trial timeline verification")
    print("══════════════════════════════════════════════")
    print("Reference (Leeb et al., 2008): trial start (beep+cross) at t=0,")
    print("directional cue at t=3s (1.25s duration), imagery until ~t=7s,")
    print("online feedback bar (sessions 4-5 only) shown shortly after the cue.\n")

    lines = [
        "BCI-IV-2b trial timeline — verified empirically from raw GDF event timestamps.",
        "Reference: Leeb et al. (2008) describes trial start at t=0, cue onset at t=3s.",
        "",
    ]

    for subj in subjects:
        for session in TRAIN_SESSIONS + TEST_SESSIONS:
            verify_file(subj, session, lines)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
