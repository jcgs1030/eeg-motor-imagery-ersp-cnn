# Experimental Log — EEG Motor Imagery Classification

**Dataset:** BCI Competition IV – Dataset 2b  
**Task:** Binary classification — left-hand vs. right-hand motor imagery  
**Channels:** C3, Cz, C4 (3 electrodes over the motor cortex)  
**Protocol:** Sessions 1–3 (offline, no feedback) → train | Sessions 4–5 (online, with feedback) → test

---

## Critical Bug Fix — Evaluation Label Correction

> **All results from Experiments 1–5b recorded prior to this fix were invalid.**
> The true fix and corrected results are documented here.

### Root cause

The evaluation GDF files (`B*04E.gdf`, `B*05E.gdf`) do **not** contain the true class
labels (left/right). The BCI Competition IV intentionally withheld evaluation labels,
distributing them in separate files. The GDF files contain only generic event types:

- `781` — BCI feedback (continuous, fires in every trial)
- `783` — cue onset ("class unknown" in the competition protocol)

The original code mapped `781→left` and `783→right` (treating event *types* as class
*labels*). Since both events appear in every single trial in a fixed order, this forced
every classifier to output ~50% regardless of the EEG content.

### Evidence

A 5-fold CV using only the training sessions (real labels) immediately confirmed the
signal is learnable: S01=70.5%, S04=90.7%, S08=68.4% (CSP+LDA). The pipeline,
filtering, and feature extraction were all correct — only the evaluation labels were wrong.

### Fix applied (this session)

- **`src/eval_labels.py`** — retrieves the true evaluation labels from MOABB
  (`BNCI2014_004`), which bundles the official competition labels. Labels are cached to
  `data/processed/eval_labels.npz` (one download, then local cache).
- **`src/config.py`** — removed the incorrect `781→left, 783→right` mapping. Added
  `EVENT_CUE_EVAL = 783` (used for epoch alignment only; class is assigned externally).
- **`src/preprocessing.py`** — new `extract_epochs_eval()` function: aligns evaluation
  epochs to event `783` (the actual cue onset), then assigns the MOABB labels in
  chronological order, using `epochs.selection` to account for any artifact-rejected trials.
- All data regenerated: `S0xE-epo.fif`, `S0xE-ersp.npz`, all results CSVs and figures.

### Post-fix validation: S04 CSP+LDA = 92.5% (was 49%) ✓

---

## Preprocessing Pipeline (all experiments)

**Script:** `src/preprocessing.py`

| Step | Parameter | Value |
|---|---|---|
| Channel selection | C3, Cz, C4 | Motor cortex electrodes |
| Bandpass filter | 8–30 Hz, FIR Hamming | Mu and beta bands |
| Epoch window | −0.5 to 4.0 s | Relative to cue onset |
| Baseline correction | None (removed in fix, see below) | ERSP handles it internally |
| Artifact rejection | Peak-to-peak threshold 100 µV | Per-channel per-epoch |

**Fix applied (commit `3a242de`):** MNE's amplitude baseline correction was initially applied
(`baseline=(-0.5, 0.0)`) before ERSP computation. Since the ERSP formula already normalises
by the spectral power of the pre-stimulus window, the amplitude correction constituted a
double-baselining that distorted the power reference. It was removed (`baseline=None`).

---

## Representation: ERSP Spectrograms

**Script:** `src/ersp.py`

ERSP (Event-Related Spectral Perturbation) quantifies the change in spectral power
relative to a pre-stimulus baseline:

```
ERSP(f, t) = 10 · log₁₀[ P(f, t) / P_baseline(f) ]
```

| Parameter | Value |
|---|---|
| Transform | STFT, Hann window |
| Window length | 256 samples (1.024 s at 250 Hz) |
| Overlap | 75% (hop = 64 samples) |
| Frequency range | 8–30 Hz → 22 bins |
| Time bins | 128 (after resize) |
| Output shape | (3 channels, 22 freq, 128 time) per trial |

**Fix applied (commit `3a242de`):** The original code normalised each ERSP image
independently to [0, 1] using per-trial min-max scaling. This destroyed inter-trial
comparability: the lateralisation pattern (ERD in the contralateral hemisphere, ERS
in the ipsilateral) depends on consistent dB magnitudes across trials.  
**Fix:** replaced with a fixed ±6 dB clip followed by linear scaling to [0, 1].
After the fix, 0.5 = no change, < 0.5 = ERD (desynchronisation), > 0.5 = ERS.

---

## Diagnostic Finding — Baseline Window Contamination (fixed and reprocessed)

> **Status: APPLIED.** `EPOCH_TMIN`/`BASELINE` were extended in `src/config.py`,
> all `S0x{T,E}-epo.fif` / `S0x{T,E}-ersp.npz` files were regenerated, and
> Experiments 1–5b below were all re-run. Results below reflect the corrected
> baseline. See "Reprocessing outcome" at the end of this section for what
> actually changed.

### Root cause

`STFT_WIN_LEN = 256` samples (1.024 s at 250 Hz) is **longer** than the baseline
window `BASELINE = (-0.5, 0.0)` (0.5 s). Since `compute_ersp_image()` (`src/ersp.py`)
estimates the baseline reference power by reusing STFT frames from the trial's own
STFT, every "baseline" frame already leaks signal from **after** the cue:

```
frame centered at t=0.000s (local) -> spans absolute [-0.500, +0.012]s
frame centered at t=0.256s (local) -> spans absolute [-0.756, +0.268]s
```

The second frame alone mixes in up to 268 ms of post-cue signal into what should be
a pure pre-stimulus reference. This does not change the shape of the data (still
`(3, 22, 128)` per trial), but it biases the ERSP normalisation and visually flattens
the ERD/ERS contrast in any grand-average plot.

### Evidence (`src/verify_timeline.py`, `src/diagnostic_fixed_baseline.py`, `src/diagnostic_extended_baseline.py`)

- `verify_timeline.py` confirmed empirically (45/45 GDF files, std = 0.0000) that the
  trial start-to-cue interval is exactly **3.000 s**, i.e. sessions 1–3 have 3 full
  seconds of clean pre-cue fixation available — far more than the 0.5 s currently used.
- A first diagnostic (`diagnostic_fixed_baseline.py`) recomputed the baseline via an
  independent Welch periodogram over the *same* 0.5 s window (no STFT frame reuse).
  Result: **no visible change** — ruling out "frame reuse" alone as sufficient
  explanation; the 0.5 s window itself is simply too short and too close to the
  epoch's own STFT zero-padding edge (`padded=True`).
- A second diagnostic (`diagnostic_extended_baseline.py`) re-extracted epochs with
  `tmin=-3.0s` (using the full fixation period) and computed the baseline reference
  from a clean `-1.0..0.0s` window, far from any padding edge. Result, channel C3
  (contralateral to right-hand imagery), mu band (8–13 Hz):

  | Scope | Class | Baseline (dB) | Imagery (dB) | Δ |
  |---|---|---|---|---|
  | Subject S04 alone (n=210/class) | Left (ipsilateral) | -6.28 | -6.31 | **-0.03** (flat, as expected) |
  | Subject S04 alone (n=210/class) | Right (contralateral) | -6.05 | -8.92 | **-2.87** (clear ERD) |
  | 9 subjects pooled (n=1840/class) | Left (ipsilateral) | -9.80 | -9.95 | **-0.15** (near flat) |
  | 9 subjects pooled (n=1840/class) | Right (contralateral) | -9.87 | -10.85 | **-0.98** (ERD, correct direction) |

  This reproduces the textbook contralateral ERD pattern (Pfurtscheller & Lopes da
  Silva, 1999) at both the single-subject and population level. The pooled effect is
  much smaller than the single-subject effect, consistent with the inter-subject
  variability already documented above (S04 ≫ S03 in classification accuracy).

### Conclusion

The original `(-0.5, 0.0)` baseline window is a genuine measurement bug, not just a
visualisation issue: it biases the same ERSP tensors that feed every CNN in
Experiments 1–2 and every classical baseline in Experiments 3–5b. The ERD/ERS
physiological effect is present in the data and recoverable once the baseline window
is long enough to avoid STFT edge contamination.

### Fix applied

`EPOCH_TMIN` extended from -0.5s to **-3.0s** (the full pre-cue fixation period),
`BASELINE` moved to **(-1.0, 0.0)** (a clean 1s window, 2s away from the epoch's
own extraction edge), and new `ANALYSIS_TMIN`/`ANALYSIS_TMAX` (-0.5/4.0, unchanged)
mark the output window that actually gets resized to `IMG_TIME_BINS`. `compute_ersp_image()`
now computes the STFT baseline reference from frames inside `BASELINE`, in absolute
time relative to cue onset, then crops to the analysis window before resizing — the
output tensor shape is unchanged, `(3, 22, 128)` per trial. `compute_ersp_image_wavelet()`
was updated the same way for consistency.

All epochs and ERSP tensors were regenerated (`preprocessing.py --subject all --suffix both`,
then `ersp.py --subject all --suffix both`). Trial counts are essentially unchanged
(6519/6520 total — one S06 evaluation trial now fails the artefact-rejection threshold
because the extended pre-cue window happens to contain an artefact there; negligible).

### Reprocessing outcome

Experiments 3–5b (all classical baselines: CSP, EA+CSP, FBCSP, Riemannian) operate
directly on band-pass filtered epochs, **not** on ERSP tensors — they were re-run for
consistency but, as expected, their results are statistically unchanged from before
this fix (differences are within training/artefact-rejection noise). This confirms
the bug was isolated to the ERSP representation, as the root-cause analysis predicted.

**Experiment 1 (CNN, subject-pooled) changed dramatically** — this is the only
experiment where the fix had a large practical effect, because it is the only one
that trains a single model across all 9 subjects' ERSP tensors and evaluates it on
a session split (T→E) it never saw:

| Model | Subjects | Before fix | After fix |
|---|---|---|---|
| EEGNet | 1 | 49.4% (chance) | **67.5%** |
| EEGNet | all (9) | 48.3% (chance) | **73.2%** |
| ShallowConvNet | 1 | 48.6% (chance) | **69.7%** |
| ShallowConvNet | all (9) | 49.6% (chance) | **70.7%** |
| SpectNet | 1 | 48.9% (chance) | **63.4%** |
| SpectNet | all (9) | 48.9% (chance) | **74.5%** |

Confusion matrices confirm this is genuine discrimination, not class collapse
(e.g. SpectNet all-subjects: 81.8% recall on Left, 67.3% on Right — both well above
chance; kappa 0.24–0.49 across all six runs).

**Working hypothesis for why Experiment 1 specifically was so sensitive:** the
STFT edge-padding artefact biased the baseline reference in a way that plausibly
differed between training (offline, sessions 1–3) and evaluation (online, sessions
4–5) segments — e.g. different trial-to-trial timing near the artefact-affected
region. A subject-specific model (Experiment 2) could still exploit idiosyncratic
per-subject cues despite the bias. A pooled model, forced to rely on whatever
signal is common across subjects, was instead pulled toward the artefact — which
is exactly the kind of feature that would generalise poorly from T to E. Once the
artefact was removed, the model could rely on the genuine (and now confirmed
present) contralateral ERD/ERS pattern shared across subjects, which does
generalise T→E. This is offered as the most likely explanation, not an
independently re-verified causal claim — it was not tested by, e.g., deliberately
reintroducing only the edge artefact in isolation.

Experiment 2 (subject-specific CNN) changed only marginally (within noise —
see its Results table below), consistent with this hypothesis: a subject-specific
model was less dependent on the artefact to begin with, so removing it had less
room to help.

### STFT vs. Wavelet transform (`src/ersp.py: compute_ersp_image_wavelet`, `plot_stft_vs_wavelet`)

As a related methodological check, a Morlet continuous-wavelet-transform version of
the ERSP was implemented for side-by-side comparison (`results/figures/stft_vs_wavelet_*.png`,
`results/figures/ersp_grand_average_wavelet_{T,E,both}.png`). Because wavelet time
resolution adapts with frequency (narrower window at high frequencies) instead of
using STFT's single fixed 1.024 s window, it is inherently less exposed to the same
edge-contamination issue.

**Post-reprocessing observation:** with the corrected baseline, the STFT grand-average
(`ersp_grand_average_stft_both.png`) still shows the population-level contralateral
ERD only as a small, visually subtle effect (confirmed numerically: C3 mu-band
Δ ≈ -0.98 dB Right vs. -0.15 dB Left, pooled across 9 subjects) — the fixed-width
STFT window smooths the ~0.5-1.5 s post-cue transient into the surrounding baseline
period. The **wavelet grand-average makes the same effect visually obvious**: in
`ersp_grand_average_wavelet_both.png`, C3-Right and C4-Left both show a clearly
deeper mu-band patch concentrated at ~0.5-1.5 s post-cue that C3-Left and C4-Right
do not, at the population level (9 subjects, ~3260 trials/class) — not just for a
single good subject. The pattern is consistent in both the T-only
(`ersp_grand_average_wavelet_T.png`) and E-only (`ersp_grand_average_wavelet_E.png`)
splits, i.e. it survives the offline→online protocol change.

This is the clearest evidence in the project that the transform choice materially
affects what the ERD/ERS pattern looks like at the population level, even though the
production pipeline (and the CNN models) still use STFT. It does not change any
classifier result (no model was trained on wavelet-based tensors) — it is a
visualisation/interpretability finding, not a new pipeline output.

**Superseded by later fixes (partially):** the comparison above was run before
the epsilon fix (Second Diagnostic Finding) and the hop/colormap fixes (Third
and Fourth). With those applied, the **STFT** grand-average now also shows a
clear population-level contralateral ERD on its own
(`results/figures/ersp_grand_average_stft_both.png`,
`results/figures/ersp_grand_average_dB_heatmap_jet.png`) — wavelet is no
longer necessary to see the effect. The wavelet images referenced above were
not regenerated with the later fixes and reflect the intermediate
(baseline-fixed-only) pipeline state; they remain valid as a transform
comparison, just not as "proof you need wavelet to see this."

---

## Second Diagnostic Finding — ERSP Epsilon Bias (fixed and reprocessed)

> **Status: APPLIED.** `src/ersp.py`'s `compute_ersp_image()` and
> `compute_ersp_image_wavelet()` were fixed, all ERSP tensors were regenerated
> again, and Experiments 1-2 (the only ones that consume ERSP tensors) were
> re-run a second time. This is a second, independent bug from the baseline-window
> one above — found while investigating why the advisor's expected ERD/ERS
> scale (literature convention ±3 dB) looked nothing like ours.

### Root cause

`compute_ersp_image()` added a fixed `epsilon = 1e-12` to both the baseline
denominator and the final power ratio, "for numerical stability" against
`log(0)`. MNE stores EEG amplitude in volts, so squared STFT power sits in the
**~1e-12 to 1e-27** range (verified empirically: `power.min()=4.37e-27,
power.max()=1.59e-12` for a representative trial) — i.e. the epsilon was the
*same order of magnitude as the signal itself*, not negligible. This silently
inflated the denominator in `10*log10(power/baseline_mean + eps)`, biasing
every ERSP value strongly negative.

### Evidence

Before the fix, decoding the stored `[0,1]` tensors back to dB (`db = X*12 -
6`) showed the mu/beta bands were almost entirely saturated at the -6 dB clip
floor:

| Band | Median (should be ≈ -1 to -3 dB per ERD/ERS literature) | % of pixels clipped at -6 dB floor |
|---|---|---|
| Mu (8-13 Hz) | **-6.00 dB** (at the floor) | 76.7% |
| Beta (14-30 Hz) | **-6.00 dB** (at the floor) | 95.6% |

Recomputing the *unclipped* dB directly from raw signal (bypassing the stored
tensors) confirmed the true values were far more extreme than any reasonable
ERD/ERS magnitude — median -12 to -13 dB, with outliers to -71 dB. Scaling the
epsilon to the trial's own power range (`eps = power.max() * 1e-12`) instead
of a fixed absolute constant immediately narrowed this to a literature-
plausible range:

| Method | Median | % beyond ±6 dB |
|---|---|---|
| Fixed epsilon (bug) | -12.4 dB | n/a (already clipped away) |
| Relative epsilon (fix) | **-2.01 dB** | 32.7% (genuine single-trial outliers, not bug) |

### Fix applied

`eps = power.max() * 1e-12` replaces the fixed `1e-12` in both
`compute_ersp_image()` and `compute_ersp_image_wavelet()` (`src/ersp.py`). The
±6 dB clip and [0,1] scaling in the output format are unchanged — with the
epsilon fixed, ±6 dB now genuinely covers the bulk of the distribution instead
of clipping ~80-96% of it away.

### Reprocessing outcome — classification did NOT improve

All ERSP tensors were regenerated a second time and Experiments 1 and 2
(the only ERSP-consuming experiments) were re-trained. Classical baselines
(Experiments 3-5b) are unaffected, as before.

| Model | Baseline-fix only | + Epsilon fix |
|---|---|---|
| EEGNet, pooled (1 subj) | 67.5% | 61.9% |
| EEGNet, pooled (9 subj) | 73.2% | 70.1% |
| ShallowConvNet, pooled (1 subj) | 69.7% | **51.2%** |
| ShallowConvNet, pooled (9 subj) | 70.7% | **58.4%** |
| SpectNet, pooled (1 subj) | 63.4% | 65.0% |
| SpectNet, pooled (9 subj) | 74.5% | 70.0% |
| EEGNet, subject-specific | 72.0% ± 14.5% | 67.9% ± 11.7% |
| ShallowConvNet, subject-specific | 69.0% ± 14.2% | **55.7% ± 4.8%** |
| SpectNet, subject-specific | 72.7% ± 14.4% | 71.0% ± 13.2% |

Contrary to the hypothesis that the saturated ±6 dB clip was destroying useful
information for the CNN, fixing it **did not improve any model**, and
**ShallowConvNet collapsed to near-chance** across all 9 subjects (kappa
-0.01 to 0.24 per subject; confusion matrix confirms partial, not total, class
collapse — 50.6% Left / 66.3% Right recall for the pooled model, not a
degenerate single-class predictor). EEGNet and SpectNet regressed mildly
(1-4 points) in most configurations.

**Working hypothesis:** the pre-fix representation, while numerically wrong,
was also extremely sparse/low-entropy (most pixels pinned to exactly 0.0) —
this may have acted as an accidental strong regulariser for CNNs training on
very little data (~200-440 trials/subject). The corrected representation
exposes the real, noisier trial-to-trial variability, which a small-sample
CNN may overfit to rather than generalise from. ShallowConvNet's square→log
nonlinearity (designed to mimic CSP log-variance) is plausibly more sensitive
to this than EEGNet/SpectNet's more conventional conv-ReLU stacks, but this
was not independently verified (e.g. by ablating the nonlinearity) and is
offered as the most likely explanation, not a confirmed cause.

**Conclusion for the thesis:** the epsilon fix (like the baseline-window fix)
is methodologically necessary — the pre-fix ERSP values were not
scientifically defensible (a "conservative" ±6 dB clip that silently discarded
80-96% of the real dynamic range is not a design choice, it is broken
normalisation). But **physiological correctness and classification accuracy
are separate questions**: correcting a bug is not guaranteed to, and here did
not, improve downstream classification. Both findings should be presented to
the thesis committee together, not just the encouraging one.

### Classic ERD/ERS curve (Pfurtscheller-style), post both fixes

As a complementary visualisation to the 2D ERSP heatmaps (where the whole
frequency axis is always positive — time-frequency spectrograms do not show
"negative" values on the y-axis; ERD/ERS is encoded in colour, not position),
a classic 1D band-averaged curve was generated: `% power change vs. baseline`
over time, per frequency band, matching the standard Pfurtscheller ERD/ERS
plot convention the advisor expected.

`results/figures/erd_curve_classic_grandaverage_C3.png` — grand-average, 9
subjects, channel C3, Left vs. Right, mu and beta bands:

- **Beta (14-30 Hz):** clean, textbook contralateral ERD — Right dips to
  -10% to -12% and stays there from ~0.5s to ~3.5s post-cue; Left stays
  positive (+5% to +10%) for most of the trial. SEM bands barely overlap.
- **Mu (8-13 Hz):** same directional contrast (Right more negative than Left)
  but noisier — Left rises to implausibly large ERS values (+60%), likely
  reflecting mu's larger natural amplitude/variability rather than a cleaner
  task-specific signal. Beta is the band to lead with when presenting this.

`results/figures/erd_curve_classic_S04_C3.png` shows the same plot for S04
alone, for comparison with the grand-average.

---

## Third Diagnostic Finding — STFT Hop / resize_2d Undersampling (fixed and reprocessed)

> **Status: APPLIED.** `STFT_HOP` changed from 64 to 9 samples in
> `src/config.py`, all ERSP tensors regenerated, Experiments 1-2 re-run.

### Root cause

Raised by the thesis advisor: `_resize_2d()` (`src/ersp.py`) uses bilinear
interpolation (`scipy.ndimage.zoom`) to stretch the raw STFT output to
`(IMG_FREQ_BINS, IMG_TIME_BINS) = (22, 128)`. The frequency axis needed
essentially no stretching (22 real bins -> 22 output bins, ~1.00x), but the
time axis was being stretched **7.11x**: with `STFT_WIN_LEN=256` and the
original `STFT_HOP=64` (256ms), the 4.5s analysis window produced only
**18 real STFT frames**. The other ~110 output time-bins were pure
interpolation between those 18 points, not new information - the true
temporal resolution was ~256ms (one hop), not the ~35ms that 128 bins would
suggest.

### Fix applied

`STFT_HOP` reduced to 9 samples (36ms), chosen so the analysis window yields
125 real frames - close to `IMG_TIME_BINS=128` (zoom factor ~1.02x, negligible
interpolation). `STFT_OVERLAP` is now derived from `STFT_HOP` instead of the
other way around. `STFT_WIN_LEN` (and therefore frequency resolution) is
unchanged.

### Evidence

A direct visual test (before committing to a full reprocessing) compared S04
and S03 at the old vs. new hop (one-off test images, deleted after the
production files below were regenerated with the fix): S04's contralateral
ERD became sharply localised (a well-defined dark patch
in mu band, extending into 17.5-25Hz beta, concentrated at ~0.5-2s post-cue)
versus the old diffuse version; S03 (the "weak responder" control) stayed
flat under both hop settings - confirming the sharper pattern is signal, not
added noise. The production grand-average
(`results/figures/ersp_grand_average_stft_both.png`) shows the same
sharpening at the population level (9 subjects).

### Reprocessing outcome

Experiments 3-5b unaffected (raw-epoch methods, as with the previous two
fixes). Experiments 1-2 (the only ERSP consumers) were re-run:

| Model | + epsilon fix only | + hop fix (current) |
|---|---|---|
| EEGNet, pooled (1 subj) | 61.9% | 65.0% |
| EEGNet, pooled (9 subj) | 70.1% | 70.0% |
| ShallowConvNet, pooled (1 subj) | 51.2% | 50.3% |
| ShallowConvNet, pooled (9 subj) | 58.4% | 60.3% |
| **SpectNet, pooled (1 subj)** | 65.0% | 64.7% |
| **SpectNet, pooled (9 subj)** | 70.0% | **71.7%** |
| EEGNet, subject-specific | 67.9% ± 11.7% | 69.0% ± 11.6% |
| ShallowConvNet, subject-specific | 55.7% ± 4.8% | 54.5% ± 6.9% |
| SpectNet, subject-specific | 71.0% ± 13.2% | 70.8% ± 13.1% |

Unlike the epsilon fix (which cost accuracy broadly), the hop fix is
net-neutral-to-positive: SpectNet pooled improved to 71.7% (kappa 0.434),
EEGNet held steady, and ShallowConvNet's collapse (established in the Second
Diagnostic Finding) persists essentially unchanged - confirming that
collapse is not explained by temporal undersampling either. The hop fix is
the safest of the three ERSP fixes applied in this project: it measurably
improved the physiological interpretability of the heatmaps without
measurably hurting any model.

---

## Fourth Diagnostic Finding — Colormap Choice for ERSP Heatmaps (applied, visualisation only)

> **Status: APPLIED.** Visualisation-only change - does not affect any
> tensor, model, or metric. `src/ersp.py`'s plotting functions switched from
> `cmap="RdYlBu_r"` to `cmap="jet"`.

### Observation

Even after both the baseline-window and epsilon fixes, the grand-average
heatmap plotted with a smooth diverging colormap (`RdYlBu_r`) still made the
contralateral ERD difficult to see at a glance - the actual signal (2-6 dB)
occupies a small fraction of a perceptually-smooth gradient. The thesis
advisor pointed to the classic EEGLAB-style ERSP plot convention (jet/turbo
colormap, sharp hue transitions) as the expected presentation format.

### Fix applied

All `cmap="RdYlBu_r"` occurrences in `src/ersp.py`'s plotting functions
(`plot_ersp_examples`, `plot_ersp_average`, `plot_ersp_grand_average`,
`plot_ersp_grand_average_wavelet`, `plot_stft_vs_wavelet`) changed to
`cmap="jet"`. Compared side-by-side against `turbo` (the perceptually-improved
modern alternative to `jet`): `jet`'s sharper, earlier green-to-cyan-to-blue
transition gives more apparent contrast for the 2-6 dB range typical of
motor-imagery ERD/ERS than `turbo`'s smoother ramp - `jet` was kept.

A standalone diagnostic script, `src/diagnostic_db_heatmap.py`, generates the
grand-average heatmap in **real dB units** (not the [0,1] normalised scale
stored for CNN training) with a signed, diverging colour bar centred at
0 dB - the literal format the advisor asked to see
(`results/figures/ersp_grand_average_dB_heatmap_jet.png`).

### Result

With `jet` and the corrected pipeline (baseline + epsilon + hop fixes), the
contralateral ERD is now visible without any additional explanation needed:
C3-Right and C4-Left show a clearly deeper blue patch (mu band, extending
into 17.5-27.5Hz beta) at ~0.5-2s post-cue that C3-Left/C4-Right do not, at
the population level (9 subjects). This is purely a rendering change -
colormap choice does not alter the underlying ERSP values, only how
perceptible a given magnitude of difference is to the human eye.

---

## Follow-up Experiments — Statistical Validation, Ensembling, Improvement Attempts

After the three ERSP fixes above, four follow-up questions were investigated:
is the cross-experiment ranking statistically real, can ShallowConvNet's
collapse be fixed cheaply, does combining the best classical and CNN methods
help, and does data augmentation help with the small per-subject sample size.
None of this changes the production pipeline or any committed per-experiment
result file (all scripts are exploratory, under `src/diagnostic_*.py` /
`src/experiment_*.py`).

### 1. Paired statistical test (`src/diagnostic_stats_test.py`)

The cross-experiment comparison table ranks methods by mean accuracy alone;
with ±12-16% std across only 9 subjects, several rankings could plausibly be
noise. Pooled CNN checkpoints (Experiment 1, a single shared model) were
evaluated separately on each subject's own test trials to get a 9-value
accuracy vector comparable to the subject-specific methods, then compared
pairwise with a paired Wilcoxon signed-rank test (n=9).

| Comparison | Mean diff | Wilcoxon p | Significant? |
|---|---|---|---|
| FBCSP+LDA vs. SpectNet (pooled) | +4.4pp | 0.027 | Yes |
| FBCSP+LDA vs. Riem-TS+LDA | +3.0pp | 0.078 | Marginal |
| FBCSP+LDA vs. SpectNet (subject-specific) | +5.1pp | 0.004 | Yes |
| SpectNet pooled vs. SpectNet subject-specific | +0.7pp | 0.910 | **No - equivalent** |
| SpectNet pooled vs. EEGNet pooled | +1.7pp | 0.051 | Marginal |
| SpectNet pooled vs. ShallowConvNet pooled | +11.3pp | 0.004 | Yes |
| FBCSP+LDA vs. CSP+LDA | +5.0pp | 0.008 | Yes |

Takeaways: FBCSP's edge over SpectNet is real, not noise. ShallowConvNet's
collapse (Second Diagnostic Finding) is statistically robust across all 9
subjects, not a bad run. Pooled vs. subject-specific training for SpectNet
is statistically indistinguishable - a single shared model is as good as 9
separate ones, which favours the simpler pooled approach.

### 2. ShallowConvNet learning-rate sweep (`src/diagnostic_lr_sweep.py`)

All three CNN architectures share one global `LEARNING_RATE=1e-3` in
`config.py`, never tuned per architecture. Tested 1e-4 / 3e-4 / 1e-3 on
ShallowConvNet (9 subjects each):

`results/figures/shallowconvnet_lr_sweep.png`

| LR | Mean accuracy | Mean kappa |
|---|---|---|
| 1e-4 | 53.3% ± 3.9% | 0.065 |
| 3e-4 | 55.8% ± 5.3% | 0.117 |
| 1e-3 (current) | 54.5% ± 6.5% | 0.090 |

No clear trend (all three overlap well within one std of each other) -
**learning rate is ruled out** as the cause of the collapse. The problem is
more likely architectural (ShallowConvNet's square→log nonlinearity,
designed to mimic CSP log-variance, interacting badly with the corrected,
noisier ERSP distribution) rather than a mistunable hyperparameter.

### 3. Ensemble: FBCSP+LDA + SpectNet pooled (`src/experiment_ensemble.py`)

FBCSP's `predict_proba` (LDA) and SpectNet pooled's softmax output are
averaged per trial (`weight * FBCSP + (1-weight) * SpectNet`). Both are
evaluated on the identical per-subject test trials in the identical order
(both read from the same `S{subj}E-epo.fif`), so no alignment step beyond an
explicit equality assertion is needed.

`results/figures/ensemble_per_subject.png`,
`results/figures/final_comparison_all_methods.png`

| Weight (FBCSP) | Ensemble accuracy | vs. FBCSP alone | vs. SpectNet alone |
|---|---|---|---|
| 0.5 | 78.7% ± 12.8% | p=0.027 | p=0.008 |
| **0.6** | **78.8% ± 12.7%** | **p=0.008** | **p=0.004** |
| 0.7 | 77.8% ± 12.5% | p=0.008 | p=0.008 |
| 0.4 | 77.0% ± 12.5% | p=0.637 (not sig.) | p=0.008 |

**w=0.6 is the new best result in the project: 78.8% ± 12.7%, significantly
better than either method alone.** At this weight the ensemble matches or
beats the better of the two individual methods for all 9 subjects (see the
slope plot) - FBCSP and SpectNet appear to make at least partially
independent errors, which is exactly the condition under which averaging
helps.

### 4. Data augmentation on ShallowConvNet (`src/augmentation.py`, `src/experiment_augmentation.py`)

Simple Gaussian noise (std=0.03, normalised [0,1] scale) and random circular
time-shift (±6 bins) applied only to the training split via a new
`train_transform` parameter on `build_loaders()` / `train_one_subject()`.
**The frequency axis (and therefore the fixed 8-30Hz mu/beta band) is never
touched by either transform**, per project constraint.

`results/figures/augmentation_per_subject.png`

| | Mean accuracy |
|---|---|
| Baseline (no augmentation) | 54.5% ± 6.5% |
| Augmented | 56.3% ± 5.6% |
| Difference | +1.8pp, Wilcoxon p=0.156 (not significant, n=9) |

6 of 9 subjects improved, 2 were unchanged, 1 regressed slightly. Directionally
positive and reduces cross-subject std, but not statistically conclusive at
n=9 - promising enough to revisit with a larger augmentation budget (more
transform variety, or mixup between same-class trials) rather than treating
ShallowConvNet's collapse as solved.

### 5. Fine-tuning SpectNet pooled — per-subject and global continuation (`src/experiment_finetune.py`)

Motivated by the literature review (BCI-IV-2a fine-tuning schedules reporting
+22pp from conservative per-subject fine-tuning of a pooled model — see
references added to the advisor document). Both variants start from the same
converged checkpoint, `results/spectnet_all.pth`.

**Per-subject mode**: freeze `conv1`/`conv2` (the convolutional feature
extractor), fine-tune only `fc1`+`fc2` with a low learning rate (1e-4) on
each subject's own ~340-trial training split, early stopping on that
subject's own validation loss (patience=10). Produces 9 specialised models.

**Global mode**: unfreeze everything, continue training the SAME pooled
checkpoint on the full 9-subject pooled training set again at a lower
learning rate (3e-5), early stopping as before. Produces one model (answers
"fine-tuning sobre el promedio/pooled en vez de por sujeto").

| | Mean accuracy | vs. pooled baseline (71.7%) |
|---|---|---|
| Pooled baseline (no fine-tuning) | 71.7% | — |
| Per-subject fine-tuning | 71.6% | +0.0pp, Wilcoxon p=0.984 |
| Global continuation | 71.5% | -0.1pp, Wilcoxon p=0.488 |

Neither variant helps, and both are a long way from the +22pp some IV-2a
fine-tuning papers report. Per-subject fine-tuning is a wash in aggregate but
has real per-subject spread (S05 +3.1pp, S08 +2.5pp, S09 +1.9pp vs. S02
-2.5pp, S06 -3.1pp, S03 -1.6pp) - it helps subjects that already have
reasonable signal and hurts weak-signal subjects further, consistent with
the "S02/S03 near chance" finding in section 05 of the advisor document: if
the base representation carries little class-relevant information for a
subject, no amount of local adaptation recovers it. The global continuation
run converges to its best validation loss at epoch 1 and plateaus
immediately (early stopping triggers within ~10 epochs) - the pooled
checkpoint was already at its optimum before this experiment started, so
there was nothing left to gain from more training on the same pooled data,
consistent with the "early stopping always binds, not MAX_EPOCHS" finding
discussed with the user.

### A note on code safety found while running these experiments

`train_one_subject()` was writing its checkpoint and figures unconditionally
to the shared `results/subject_specific/` paths regardless of caller - the
LR-sweep and augmentation experiments above were silently overwriting the
official committed ShallowConvNet results on every run. Caught via `git
status` before anything was committed, restored with `git checkout`, and
fixed by adding a `save_outputs=False` option that only the production
`run()` CLI entry point omits.

---

## Experiment 1 — CNN Subject-Pooled

**Scripts:** `src/train.py`, `src/evaluate.py`  
**Results:** `results/figures/`, `results/metrics/`

### Design
All 9 subjects' data concatenated into a single dataset. One model trained
for the full cohort. Two variants: subject 1 only, and all 9 subjects.

### Models

| Architecture | Parameters | Reference |
|---|---|---|
| EEGNet | 6,194 | Lawhern et al., J. Neural Eng., 2018 |
| ShallowConvNet | 38,442 | Schirrmeister et al., Hum. Brain Mapp., 2017 |
| SpectNet | 604,146 | Ruffini et al., arXiv, 2018 |

### Results

**Current (post baseline-window + epsilon + hop fixes — see the three
"Diagnostic Finding" sections above):**

| Model | Subjects | Accuracy | Precision | Recall | F1-score | Kappa |
|---|---|---|---|---|---|---|
| EEGNet | 1 | 65.0% | 65.5% | 65.0% | 64.7% | 0.300 |
| ShallowConvNet | 1 | 50.3% | 50.8% | 50.3% | 41.4% | 0.006 |
| SpectNet | 1 | 64.7% | 64.9% | 64.7% | 64.6% | 0.294 |
| EEGNet | all (9) | 70.0% | 70.2% | 70.0% | 69.9% | 0.401 |
| ShallowConvNet | all (9) | 60.3% | 60.3% | 60.3% | 60.3% | 0.205 |
| **SpectNet** | **all (9)** | **71.7%** | 71.7% | 71.7% | 71.7% | **0.434** |

After the baseline-window + epsilon fixes but before the hop fix (kept for
the record):

| Model | Subjects | Accuracy | Precision | Recall | F1-score | Kappa |
|---|---|---|---|---|---|---|
| EEGNet | 1 | 61.9% | 61.9% | 61.9% | 61.8% | 0.238 |
| ShallowConvNet | 1 | 51.2% | 52.7% | 51.2% | 43.6% | 0.025 |
| SpectNet | 1 | 65.0% | 65.1% | 65.0% | 65.0% | 0.300 |
| EEGNet | all (9) | 70.1% | 70.1% | 70.1% | 70.1% | 0.401 |
| ShallowConvNet | all (9) | 58.4% | 58.6% | 58.4% | 58.2% | 0.169 |
| SpectNet | all (9) | 70.0% | 70.0% | 70.0% | 70.0% | 0.400 |

After the baseline-window fix but before the epsilon fix (kept for the
record — these were trained on ERSP tensors where ±6 dB saturated 77-96% of
the mu/beta bands):

| Model | Subjects | Accuracy | Precision | Recall | F1-score | Kappa |
|---|---|---|---|---|---|---|
| EEGNet | 1 | 65.6% | 74.4% | 65.6% | 62.2% | 0.312 |
| ShallowConvNet | 1 | 70.9% | 71.2% | 70.9% | 70.8% | 0.419 |
| SpectNet | 1 | 62.2% | 70.3% | 62.2% | 58.0% | 0.244 |
| EEGNet | all (9) | 73.2% | 73.8% | 73.2% | 73.0% | 0.463 |
| ShallowConvNet | all (9) | 70.7% | 70.8% | 70.7% | 70.7% | 0.414 |
| SpectNet | all (9) | 74.5% | 75.1% | 74.5% | 74.4% | 0.491 |

Original results, before either fix (kept for the record):

| Model | Subjects | Accuracy | Kappa |
|---|---|---|---|
| EEGNet | 1 | 49.4% | −0.012 |
| ShallowConvNet | 1 | 48.6% | −0.028 |
| SpectNet | 1 | 48.9% | −0.022 |
| EEGNet | all (9) | 48.3% | −0.033 |
| ShallowConvNet | all (9) | 49.6% | −0.008 |
| SpectNet | all (9) | 48.9% | −0.023 |

### Analysis
**Before any fix**, all models performed at chance level (50%) and confusion
matrices showed collapse to predicting one class. Two causes were proposed at
the time: (1) subject-pooled approach invalid for EEG due to inter-subject
variability, (2) domain shift T→E (offline vs. online sessions).

**After the baseline-window fix**, this framing turned out to be wrong or at
least incomplete: pooled CNNs jumped to 63-75%, matching or exceeding
Experiment 2, showing the pooled approach itself was not inherently invalid.

**After the epsilon fix** (a second, independent bug — see "Second Diagnostic
Finding" above), accuracy for EEGNet and SpectNet regressed mildly (2-4
points) and ShallowConvNet collapsed to near-chance (51-58%). The genuinely
physiologically-correct ERSP representation does not classify better than the
numerically-biased one — if anything, slightly worse for this dataset's
sample size. See "Second Diagnostic Finding" for the proposed explanation
(the pre-fix sparsity may have acted as accidental regularisation).

**After the hop fix** (a third, independent issue — see "Third Diagnostic
Finding" above), SpectNet pooled improved to 71.7% (its best result across
all three fixes), EEGNet held steady, and ShallowConvNet changed only
slightly (58.4%→60.3%). Unlike the epsilon fix, this one was net-neutral-to-
positive — reducing interpolation in the time axis measurably helped the
largest model (SpectNet) without hurting the others.

---

## Experiment 2 — CNN Subject-Specific

**Script:** `src/train_subject_specific.py`  
**Results:** `results/subject_specific/figures/`, `results/subject_specific/metrics/`

### Design
One model trained per (architecture, subject) pair using only that subject's
own sessions 1–3. Evaluated on that subject's sessions 4–5.
This is the standard intra-subject protocol in the BCI literature.

### Results (mean ± std across 9 subjects) — **current, post all three fixes**

| Model | Accuracy | F1-score | Kappa |
|---|---|---|---|
| EEGNet | 69.0% ± 11.6% | 68.5% ± 12.0% | 0.379 ± 0.232 |
| ShallowConvNet | **54.5% ± 6.9%** | 52.6% ± 8.6% | **0.090 ± 0.138** |
| SpectNet | 70.8% ± 13.1% | 68.7% ± 17.0% | 0.416 ± 0.261 |

After baseline-window + epsilon fixes but before the hop fix (kept for the
record):

| Model | Accuracy | F1-score | Kappa |
|---|---|---|---|
| EEGNet | 67.9% ± 11.7% | 67.5% ± 12.0% | 0.358 ± 0.235 |
| ShallowConvNet | 55.7% ± 4.8% | 54.5% ± 6.2% | 0.114 ± 0.096 |
| SpectNet | 71.0% ± 13.2% | 68.7% ± 17.2% | 0.419 ± 0.263 |

After the baseline-window fix but before the epsilon fix (kept for the
record):

| Model | Accuracy | F1-score | Kappa |
|---|---|---|---|
| EEGNet | 72.0% ± 14.5% | 69.2% ± 18.4% | 0.440 ± 0.289 |
| ShallowConvNet | 69.0% ± 14.2% | 68.4% ± 14.6% | 0.381 ± 0.284 |
| SpectNet | 72.7% ± 14.4% | 70.4% ± 18.4% | 0.453 ± 0.288 |

### Analysis
EEGNet and SpectNet remain clearly above chance (69–71% mean) after all three
fixes. **ShallowConvNet is the exception: it stays collapsed at 54.5% ± 6.9%**
(kappa 0.090, barely above zero) — uniformly weak across all 9 subjects,
including subjects that were previously strong (S04: 89.4%→62.2%,
S08: 87.8%→61.9%, with the epsilon fix; the hop fix did not recover this).
This rules out STFT temporal undersampling as the cause of the collapse,
narrowing it down further to something about the epsilon fix specifically
interacting badly with ShallowConvNet's square→log nonlinearity. See "Second
Diagnostic Finding — ERSP Epsilon Bias" above for the full comparison and
working hypothesis. High variance across subjects for EEGNet/SpectNet
(±12–13%) still reflects genuine inter-subject variability: S04 remains the
strongest responder, S02/S03 remain weak.

---

## Experiment 3 — Classical Baseline: CSP + LDA / SVM

**Script:** `src/train_csp_lda.py`  
**Results:** `results/csp_lda/figures/`, `results/csp_lda/metrics/`

### Design
Common Spatial Patterns (CSP) learns spatial filters that maximise the
variance ratio between left and right MI classes directly from raw
band-pass filtered EEG (no ERSP conversion). Log-variance features from
the 2 CSP components are classified with LDA and SVM.

CSP is the canonical classical baseline for 2-class MI-BCI and is known
to outperform deep learning methods when training data is limited.

| Parameter | Value |
|---|---|
| MI window | 0–4 s post-cue |
| CSP components | 2 (maximum discriminative with 3 channels) |
| LDA solver | SVD |
| SVM kernel | RBF, C=1.0 |

### Results (mean ± std across 9 subjects)

| Classifier | Accuracy | Kappa |
|---|---|---|
| CSP+LDA | 71.0% ± 12.6% | 69.8% ± 13.3% | 0.419 ± 0.252 |
| CSP+SVM | 72.5% ± 12.0% | 71.4% ± 13.1% | 0.450 ± 0.241 |

### Analysis
CSP, which operates directly on raw filtered epochs and was never exposed to
the ERSP baseline-window bug, reaches 71.0–72.5% mean accuracy — solidly
above chance and in line with the BCI-IV-2b literature (65–85% depending on
method). This is useful as an independent sanity check: it confirms the T→E
domain shift is not so severe as to make this task infeasible, which is
consistent with what Experiment 1 later showed for pooled CNNs once its
ERSP bug was fixed.

---

## Experiment 4 — Euclidean Alignment + CSP + LDA

**Script:** `src/train_ea_csp_lda.py`  
**Results:** `results/ea_csp_lda/figures/`, `results/ea_csp_lda/metrics/`

### Design
Euclidean Alignment (EA) — He et al., IEEE TNSRE, 2020 — is a
session-level normalisation technique that reduces cross-session variability
by whitening each session's covariance structure independently:

```
R = (1/N) Σᵢ XᵢXᵢᵀ/T          (mean covariance of the session)
X̃ᵢ = R^(-1/2) · Xᵢ            (whitened epoch)
```

Applied independently to training (sessions 1–3) and test (sessions 4–5)
before CSP spatial filtering. The test set is aligned using its own mean
covariance (unsupervised — no test labels used).

### Results (mean ± std across 9 subjects)

| Classifier | Accuracy | Kappa |
|---|---|---|
| EA+CSP+LDA | 71.0% ± 12.6% | 69.8% ± 13.3% | 0.419 ± 0.252 |
| EA+CSP+SVM | 72.5% ± 12.0% | 71.4% ± 13.1% | 0.450 ± 0.241 |

### Analysis
EA+CSP produces results essentially identical to plain CSP (Experiment 3) —
71.0–72.5% mean accuracy either way. This means the T→E domain shift in this
dataset, to the extent it exists, is not primarily a **scaling or orientation**
difference in the covariance space (which EA corrects) — session-level
whitening neither helps nor hurts here. This is a useful negative result: it
rules out one specific hypothesis about the nature of the offline→online
shift, without needing EA to "fix" anything for the pooled-CNN result in
Experiment 1 to make sense.

---

## Experiment 5a — Filter Bank CSP (FBCSP)

**Script:** `src/train_fbcsp.py`
**Results:** `results/fbcsp/figures/`, `results/fbcsp/metrics/`

### Design
Instead of applying CSP to the full 8–30 Hz band, FBCSP decomposes the
signal into 7 overlapping sub-bands of 4 Hz width (4–32 Hz) and applies
CSP independently to each. Log-variance features from all bands are
concatenated (7 × 2 = 14 features) before classification.

| Parameter | Value |
|---|---|
| Filter bank | 7 bands: [4-8], [8-12], [12-16], [16-20], [20-24], [24-28], [28-32] Hz |
| Filter type | 5th-order Butterworth, zero-phase (filtfilt) |
| CSP components per band | 2 |
| Total features | 14 |

### Results (mean ± std across 9 subjects)

| Classifier | Accuracy | Kappa |
|---|---|---|
| FBCSP+LDA | 75.9% ± 13.4% | 75.5% ± 13.7% | 0.519 ± 0.269 |
| FBCSP+SVM | 73.6% ± 13.5% | 72.7% ± 14.5% | 0.472 ± 0.271 |

### Analysis
FBCSP+LDA (75.9%) improves meaningfully over broad-band CSP+LDA (71.0%,
Experiment 3) — the multi-band decomposition captures band-specific ERD/ERS
differences that a single 8–30 Hz band misses. It is the best-performing
classical baseline, and was the best method overall until Experiment 1's
pooled SpectNet/EEGNet results (74.5%/73.2%, after the baseline-window fix)
came close to matching it.

---

## Experiment 5b — Riemannian Geometry (MDM / TS+LDA)

**Script:** `src/train_riemannian.py`
**Results:** `results/riemannian/figures/`, `results/riemannian/metrics/`

### Design
Riemannian methods operate on the covariance matrices of EEG epochs as
points on the manifold of Symmetric Positive Definite (SPD) matrices,
using geodesic distances that are invariant to linear signal transformations.

Two classifiers:

- **MDM (Minimum Distance to Mean):** Computes the Riemannian mean per class
  and assigns test trials to the nearest class mean by geodesic distance.
- **TS+LDA (Tangent Space + LDA):** Projects covariance matrices to the
  tangent space at the Riemannian training mean, then applies LDA.

Covariance estimator: Ledoit-Wolf regularisation (lwf) for numerical stability
with 3-channel data.

### Results (mean ± std across 9 subjects)

| Classifier | Accuracy | Kappa |
|---|---|---|
| Riem-MDM | 70.8% ± 13.8% | 69.2% ± 15.4% | 0.416 ± 0.276 |
| Riem-TS+LDA | 72.9% ± 13.4% | 72.0% ± 13.9% | 0.459 ± 0.268 |

### Analysis
Riemannian TS+LDA (72.9%) performs comparably to plain CSP+LDA (71.0%) and
EA+CSP+LDA (71.0%), and slightly below FBCSP+LDA (75.9%). The theoretical
robustness of Riemannian geometry to certain covariance transformations does
not translate into a clear advantage on this specific offline→online shift —
consistent with EA's similarly neutral effect in Experiment 4.

Taken together, Experiments 3–5b (CSP, EA, FBCSP, MDM, TS+LDA) all land in the
70–76% range: the T→E shift in BCI-IV-2b is real (accuracy is well below what
subject-specific same-session cross-validation would give) but not so severe
that standard classical methods fail outright — and, as Experiment 1 shows
after the baseline-window fix, a correctly-computed CNN input representation
can reach the same range with a single pooled model.

---

## Cross-Experiment Comparison — Corrected Results

All results below use **true evaluation labels** from MOABB. ERSP-based
methods (Experiment 1-2) reflect **all three** ERSP fixes — baseline-window,
epsilon, and STFT hop (see the three "Diagnostic Finding" sections above).
Metric: accuracy across sessions 4-5 (evaluation); Experiments 2-5b are
mean ± std across 9 subjects (subject-specific protocol); Experiment 1 is a
single pooled model, no std to report.

| Method | Exp | Accuracy | F1-score | Kappa |
|---|---|---|---|---|
| **Ensemble: FBCSP+LDA + SpectNet (w=0.6)** | **follow-up** | **78.8% ± 12.7%** | — | — |
| FBCSP+LDA | 5a | 75.9% ± 13.4% | 75.5% ± 13.7% | 0.519 |
| FBCSP+SVM | 5a | 73.6% ± 13.5% | 72.7% ± 14.5% | 0.472 |
| Riem-TS+LDA | 5b | 72.9% ± 13.4% | 72.0% ± 13.9% | 0.459 |
| CSP+SVM | 3 | 72.5% ± 12.0% | 71.4% ± 13.1% | 0.450 |
| EA+CSP+SVM | 4 | 72.5% ± 12.0% | 71.4% ± 13.1% | 0.450 |
| **SpectNet (pooled, all 9)** | **1** | **71.7%** | 71.7% | 0.434 |
| SpectNet (subject-specific) | 2 | 70.8% ± 13.1% | 68.7% ± 17.0% | 0.416 |
| EEGNet (pooled, all 9) | 1 | 70.0% | 69.9% | 0.401 |
| CSP+LDA | 3 | 71.0% ± 12.6% | 69.8% ± 13.3% | 0.419 |
| EA+CSP+LDA | 4 | 71.0% ± 12.6% | 69.8% ± 13.3% | 0.419 |
| Riem-MDM | 5b | 70.8% ± 13.8% | 69.2% ± 15.4% | 0.416 |
| EEGNet (subject-specific) | 2 | 69.0% ± 11.6% | 68.5% ± 12.0% | 0.379 |
| ShallowConvNet (pooled, all 9) | 1 | 60.3% | 60.3% | 0.205 |
| ShallowConvNet (subject-specific) | 2 | 54.5% ± 6.9% | 52.6% ± 8.6% | 0.090 |

**Best single method: FBCSP+LDA at 75.9% mean accuracy (Kappa 0.519)**, still
clearly ahead of every individual ERSP+CNN method even after the hop fix
recovered some ground for SpectNet pooled (70.0%→71.7%). **Best result
overall: the FBCSP+SpectNet ensemble at 78.8% ± 12.7%** (see "Follow-up
Experiments" above) — the only method found in this project that
significantly beats FBCSP alone (p=0.008). The epsilon fix (see "Second
Diagnostic Finding") is what cost the CNN methods their individual edge —
before it, pooled SpectNet reached 74.5%, essentially tying FBCSP; the hop
fix (see "Third Diagnostic Finding") partially recovered that gap, and
ensembling with FBCSP recovers the rest and then some. ShallowConvNet, both
pooled and subject-specific, remains the weakest method in the entire
comparison — ruled out learning rate and (so far) data augmentation as fixes,
see "Follow-up Experiments".

---

## Key Findings

1. **Evaluation labels were broken (critical bug):** The GDF evaluation files contain
   no true class labels. The original code mapped generic event types (781, 783) as
   left/right, forcing all classifiers to ~50%. Fixed by retrieving true labels from
   MOABB (`src/eval_labels.py`) and applying them during preprocessing.

2. **ERSP per-trial normalisation matters:** Per-trial min-max normalisation destroys
   inter-trial comparability. Fixed to ±6 dB clip (commit `3a242de`).

3. **ERSP baseline window was too short (second critical bug):** the 0.5s baseline
   window was shorter than the STFT's own 1.024s analysis window, causing every
   baseline reference to leak post-cue signal. This mainly flattened the visible
   ERD/ERS contrast and — much more consequentially — appears to have taught
   subject-pooled CNNs (Experiment 1) to rely on an artefact that did not
   generalise from training to evaluation sessions, collapsing them to chance.
   Fixed by extending the pre-cue window to the full 3s fixation period. See
   "Diagnostic Finding — Baseline Window Contamination" above.

4. **Subject-pooling is viable for EEG, once the baseline bug is fixed:**
   pooled CNN models (Experiment 1) went from chance (48–50%) to genuinely
   above-chance performance with a single shared model instead of 9 separate
   ones — the pooled approach itself was never the problem.

5. **A second, independent bug (`1e-12` epsilon, comparable in scale to
   volts²-range EEG power) was silently clipping 77–96% of the mu/beta bands
   to the same floor value.** Fixing it produced ERSP magnitudes matching the
   ERD/ERS literature (median ≈ -2 dB instead of an artificial -12 to -13 dB),
   confirmed visually with a classic Pfurtscheller-style ERD curve (beta band,
   grand-average: Right dips to -10 to -12%, Left stays positive). But it did
   **not** improve classification — EEGNet/SpectNet regressed mildly and
   ShallowConvNet collapsed to near-chance (55.7% ± 4.8%, kappa 0.114). See
   "Second Diagnostic Finding — ERSP Epsilon Bias".

6. **A third, independent issue: `_resize_2d()` was interpolating the time
   axis 7.11x** (only 18 real STFT frames for 128 output bins, with
   `STFT_HOP=64`). Reducing the hop to 9 samples (125 real frames, ~1.02x
   interpolation) sharpened the visible ERD both per-subject and at the
   population level, and was the only one of the three ERSP fixes that
   **improved** a model's accuracy (SpectNet pooled: 70.0%→71.7%) rather than
   costing one. See "Third Diagnostic Finding — STFT Hop / resize_2d
   Undersampling".

7. **Colormap choice matters as much as the data for perceived contrast.**
   The same corrected ERSP values, plotted with `jet` instead of a smooth
   diverging colormap (`RdYlBu_r`), make the contralateral ERD visible without
   further explanation — a purely cosmetic change with no effect on any
   tensor or metric. See "Fourth Diagnostic Finding — Colormap Choice".

8. **FBCSP remains the top individual method (75.9%)** and, after all three
   ERSP fixes, is still clearly ahead of every individual ERSP+CNN method
   (best CNN: SpectNet pooled at 71.7%). The ERSP bugs all needed fixing for
   scientific correctness, but only the hop fix gave any CNN method a boost —
   the classical multi-band approach remains ahead of any single CNN on this
   dataset.

9. **High inter-subject variability (±12–16%):** Some subjects show near-perfect
   classification (S04 > 90%, up to 95.0% with EEGNet subject-specific before
   the epsilon fix) while others are near chance (S02, S03 ~50–58%). This is
   consistent with the BCI-IV-2b literature and reflects genuine differences in
   the clarity of individual EEG motor imagery responses.

10. **Paired statistics confirm the ranking is real, not noise** (Wilcoxon,
    n=9): FBCSP significantly beats SpectNet pooled (p=0.027) and subject-
    specific SpectNet (p=0.004); ShallowConvNet's collapse is statistically
    robust (p=0.004 vs. SpectNet pooled), ruling out a single bad run;
    pooled vs. subject-specific training is statistically equivalent for
    SpectNet (p=0.91). See "Follow-up Experiments" above.

11. **Ensembling FBCSP+LDA with SpectNet pooled gives the best result in the
    project: 78.8% ± 12.7%** (weighted average of predict_proba/softmax,
    w=0.6 favouring FBCSP), significantly ahead of FBCSP alone (p=0.008) and
    SpectNet alone (p=0.004). The two methods appear to make at least
    partially independent errors — classical multi-band features and a
    full time-frequency CNN are complementary, not redundant, on this task.

12. **Learning rate was ruled out as the cause of ShallowConvNet's collapse**
    (sweep over 1e-4/3e-4/1e-3: 53.3-55.8%, no clear trend, all within one
    std of each other). **Data augmentation (noise + time-shift) gave a
    modest, non-significant improvement** (54.5%→56.3%, p=0.156, n=9) —
    promising direction, not a fix on its own. The architecture's
    square→log nonlinearity remains the leading unconfirmed hypothesis.

---

## References

- Leeb, R. et al. (2008). BCI Competition 2008 — Graz Data Set B. TU Graz.
- Pfurtscheller, G. & Lopes da Silva, F.H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clin. Neurophysiol.*, 110(11).
- Lawhern, V.J. et al. (2018). EEGNet. *J. Neural Eng.*, 15(5), 056013.
- Schirrmeister, R.T. et al. (2017). Deep learning with CNNs for EEG. *Hum. Brain Mapp.*, 38.
- Ruffini, G. et al. (2018). Deep learning using EEG spectrograms. *arXiv*.
- He, H. & Wu, D. (2019). Transfer learning for EEG-BCI: review. *IEEE TNSRE*, 27(1).
- He, H. et al. (2020). Transfer learning for BCI: Euclidean alignment. *IEEE TNSRE*, 68(6).
- Gramfort, A. et al. (2014). MNE software. *NeuroImage*, 86.
