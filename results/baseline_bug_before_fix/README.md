# Baseline window bug — before / diagnostic / confirmation

Evidence trail for the baseline-window contamination finding documented in
[`EXPERIMENTS.md`](../../EXPERIMENTS.md#diagnostic-finding--baseline-window-contamination-identified-reprocessing-pending).
Kept here (outside `results/figures/`, which is gitignored as regenerable output)
because these three images are the historical record of the finding, referenced
from the main README and the thesis material.

At the time these were generated, the production pipeline still used the
original `BASELINE = (-0.5, 0.0)` window — none of this has been applied to
Experiments 1–5b yet.

1. **`01_BEFORE_grand_average_flat_no_erd_visible.png`** — grand-average ERSP
   (STFT, 9 subjects pooled, sessions 1–5) using the original 0.5 s baseline.
   No visible ERD/ERS dynamics, no left/right asymmetry.
2. **`02_STILL_FLAT_after_welch_reestimate_same_0.5s_window.png`** — same 0.5 s
   window, but the baseline power recomputed independently via Welch instead of
   reusing STFT frames. Still flat — rules out "frame reuse" as the sole cause.
3. **`03_CONFIRMED_erd_recovered_with_3s_extended_baseline.png`** — epochs
   re-extracted with the full 3 s pre-cue fixation as baseline reference
   (sessions 1–3 only). The expected contralateral ERD reappears, both
   per-subject and pooled — confirming the 0.5 s window itself (too short,
   too close to the STFT's zero-padding edge) was the actual cause.
