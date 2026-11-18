"""
diagnostic_baseline_frame_support.py
-------------------------------------
Diagnostic only — does NOT touch the production pipeline or the cached
*-ersp.npz files.

Finding under test: how much post-cue signal still reaches the ERSP baseline
reference after the "Baseline Window Contamination" fix.

`compute_ersp_image()` selects baseline frames by their CENTRE:

    bl_mask = (t_abs >= BASELINE[0]) & (t_abs <= BASELINE[1])

but each STFT frame is `STFT_WIN_LEN` samples wide, so a frame centred at
t carries signal from t ± STFT_WIN_LEN/(2*SFREQ). With
BASELINE = (-1.0, 0.0) and a 256-sample window at 250 Hz, the frame centred
at t = 0 spans [-0.512, +0.512] s: half a second of motor imagery inside
what the formula treats as a pre-stimulus reference.

The fix that extended EPOCH_TMIN to -3.0 s removed a different problem — the
STFT's zero-padding taper at the extracted trial's own edge — and diluted
this one by averaging 28 frames instead of ~3. It did not eliminate it. This
script quantifies exactly what remains, so the limitation can be stated with
numbers instead of adjectives.

It also reports the symmetric question at the other end of the epoch:
ANALYSIS_TMAX == EPOCH_TMAX, so the last analysis frames sit on the trailing
taper with no buffer at all.

Usage:
    python src/analysis/diagnostics/diagnostic_baseline_frame_support.py
"""

import numpy as np
from scipy.signal import stft

from src.config import (
    ANALYSIS_TMAX, ANALYSIS_TMIN, BASELINE, EPOCH_TMAX, EPOCH_TMIN,
    SFREQ, STFT_HOP, STFT_WIN_LEN,
)


def frame_grid() -> np.ndarray:
    """Centres (in seconds relative to cue onset) of the STFT frames that
    compute_ersp_image() produces for one epoch. Uses the same call as the
    production code so the grid cannot drift from it."""
    n_samples = int(round((EPOCH_TMAX - EPOCH_TMIN) * SFREQ)) + 1
    _, times_stft, _ = stft(
        np.zeros(n_samples), fs=SFREQ, window="hann",
        nperseg=STFT_WIN_LEN, noverlap=STFT_WIN_LEN - STFT_HOP, padded=True,
    )
    return times_stft + EPOCH_TMIN


def main():
    half = STFT_WIN_LEN / (2 * SFREQ)   # half-support of one frame, seconds
    centres = frame_grid()

    bl = centres[(centres >= BASELINE[0]) & (centres <= BASELINE[1])]
    an = centres[(centres >= ANALYSIS_TMIN) & (centres <= ANALYSIS_TMAX)]

    print("\n══════════════════════════════════════════════")
    print("  Diagnostic — STFT frame support vs. the cue")
    print("══════════════════════════════════════════════\n")
    print(f"  Epoch                 : {EPOCH_TMIN} .. {EPOCH_TMAX} s "
          f"({len(centres)} STFT frames)")
    print(f"  STFT window           : {STFT_WIN_LEN} samples "
          f"= {2 * half:.3f} s  (half-support ±{half:.3f} s)")
    print(f"  Hop                   : {STFT_HOP} samples "
          f"= {STFT_HOP / SFREQ * 1000:.0f} ms")

    print(f"\n  BASELINE {BASELINE} selects {len(bl)} frames, "
          f"centres {bl[0]:+.3f} .. {bl[-1]:+.3f} s")
    straddling = bl[bl + half > 0.0]
    print(f"    frames whose support crosses the cue : {len(straddling)} "
          f"({100 * len(straddling) / len(bl):.0f}% of the reference)")
    print(f"    most post-cue signal in one frame    : "
          f"{(bl + half).max():.3f} s")
    print(f"    first fully pre-cue centre would be  : "
          f"{-half:.3f} s or earlier")

    clean = centres[(centres >= EPOCH_TMIN + half) & (centres <= -half)]
    if len(clean):
        print(f"    a fully pre-cue window of the same width is available at "
              f"({clean[-1] - (BASELINE[1] - BASELINE[0]):+.2f}, {clean[-1]:+.2f}) s "
              f"— {len(clean)} frames are clean")

    print(f"\n  ANALYSIS ({ANALYSIS_TMIN}, {ANALYSIS_TMAX}) selects {len(an)} frames, "
          f"centres {an[0]:+.3f} .. {an[-1]:+.3f} s")
    over = an[an + half > EPOCH_TMAX]
    print(f"    frames extending past EPOCH_TMAX     : {len(over)} "
          f"({100 * len(over) / len(an):.0f}%) -> zero-padded taper")
    print(f"    buffer after ANALYSIS_TMAX           : "
          f"{EPOCH_TMAX - ANALYSIS_TMAX:.3f} s "
          f"(needs {half:.3f} s for a clean trailing frame)")

    print("\n  Note: averaging many frames dilutes the post-cue contribution,")
    print("  it does not remove it. See EXPERIMENTS.md -> 'Diagnostic Finding —")
    print("  Baseline Window Contamination', subsection 'What the fix did not")
    print("  solve'.\n")


if __name__ == "__main__":
    main()
