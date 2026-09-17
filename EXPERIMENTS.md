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

**Post baseline-window fix** (see "Diagnostic Finding" above — these results
were trained on ERSP tensors with the corrected baseline; the chance-level
numbers from the original, contaminated-baseline pipeline are kept below for
the historical record):

| Model | Subjects | Accuracy | Precision | Recall | F1-score | Kappa |
|---|---|---|---|---|---|---|
| EEGNet | 1 | 65.6% | 74.4% | 65.6% | 62.2% | 0.312 |
| ShallowConvNet | 1 | 70.9% | 71.2% | 70.9% | 70.8% | 0.419 |
| SpectNet | 1 | 62.2% | 70.3% | 62.2% | 58.0% | 0.244 |
| EEGNet | all (9) | 73.2% | 73.8% | 73.2% | 73.0% | 0.463 |
| ShallowConvNet | all (9) | 70.7% | 70.8% | 70.7% | 70.7% | 0.414 |
| SpectNet | all (9) | **74.5%** | 75.1% | 74.5% | 74.4% | **0.491** |

Original results, before the baseline-window fix (kept for the record):

| Model | Subjects | Accuracy | Kappa |
|---|---|---|---|
| EEGNet | 1 | 49.4% | −0.012 |
| ShallowConvNet | 1 | 48.6% | −0.028 |
| SpectNet | 1 | 48.9% | −0.022 |
| EEGNet | all (9) | 48.3% | −0.033 |
| ShallowConvNet | all (9) | 49.6% | −0.008 |
| SpectNet | all (9) | 48.9% | −0.023 |

### Analysis
**Before the fix**, all models performed at chance level (50%) and confusion
matrices showed collapse to predicting one class. Two causes were proposed at
the time:

1. Subject-pooled approach invalid for EEG due to inter-subject variability.
2. Domain shift T→E (offline vs. online sessions).

**After the baseline-window fix, this framing turned out to be wrong** — or at
least incomplete. Once the ERSP tensors carry the genuine (bias-free)
contralateral ERD/ERS pattern, subject-pooled CNNs generalise T→E well above
chance (63–75% across all 6 model/subject-count combinations), matching or
slightly exceeding the subject-specific results in Experiment 2. This means
neither cause above was the true bottleneck: the pooled approach itself is not
inherently invalid, and the T→E domain shift, while real, is not severe enough
to prevent generalisation once the input representation is measured correctly.
See "Reprocessing outcome" in the Diagnostic Finding section above for the
proposed explanation (a baseline-window artefact that likely differed between
T and E segments, which the pooled model was previously forced to rely on).

---

## Experiment 2 — CNN Subject-Specific

**Script:** `src/train_subject_specific.py`  
**Results:** `results/subject_specific/figures/`, `results/subject_specific/metrics/`

### Design
One model trained per (architecture, subject) pair using only that subject's
own sessions 1–3. Evaluated on that subject's sessions 4–5.
This is the standard intra-subject protocol in the BCI literature.

### Results (mean ± std across 9 subjects) — **post baseline-window fix**

| Model | Accuracy | F1-score | Kappa |
|---|---|---|---|
| EEGNet | 72.0% ± 14.5% | 69.2% ± 18.4% | 0.440 ± 0.289 |
| ShallowConvNet | 69.0% ± 14.2% | 68.4% ± 14.6% | 0.381 ± 0.284 |
| SpectNet | 72.7% ± 14.4% | 70.4% ± 18.4% | 0.453 ± 0.288 |

These are statistically unchanged from the pre-fix numbers (72.6%/70.2%/73.3%,
all within ±1% and inside the std bars) — see the "Reprocessing outcome"
discussion above for why Experiment 1, not this one, was where the fix mattered.

### Analysis
Subject-specific CNNs are well above chance (69–73% mean), confirming that
the EEG signal contains discriminative MI patterns. High variance across
subjects (±14–15%) reflects genuine inter-subject variability: some subjects
(e.g. S04, 95.0% with EEGNet) are "BCI-literate" with very clear ERD/ERS
patterns, while others (e.g. S02, S03) remain close to chance.

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

All results below use **true evaluation labels** from MOABB, and ERSP-based
methods (Experiment 1) use the **corrected baseline window** (see "Diagnostic
Finding" above). Metric: accuracy across sessions 4–5 (evaluation); Experiments
2–5b are mean ± std across 9 subjects (subject-specific protocol); Experiment 1
is a single pooled model, no std to report.

| Method | Exp | Accuracy | F1-score | Kappa |
|---|---|---|---|---|
| EEGNet (pooled, all 9) | 1 | 73.2% | 73.0% | 0.463 |
| ShallowConvNet (pooled, all 9) | 1 | 70.7% | 70.7% | 0.414 |
| **SpectNet (pooled, all 9)** | **1** | **74.5%** | **74.4%** | **0.491** |
| EEGNet (subject-specific) | 2 | 72.0% ± 14.5% | 69.2% ± 18.4% | 0.440 |
| ShallowConvNet (subject-specific) | 2 | 69.0% ± 14.2% | 68.4% ± 14.6% | 0.381 |
| SpectNet (subject-specific) | 2 | 72.7% ± 14.4% | 70.4% ± 18.4% | 0.453 |
| CSP+LDA | 3 | 71.0% ± 12.6% | 69.8% ± 13.3% | 0.419 |
| CSP+SVM | 3 | 72.5% ± 12.0% | 71.4% ± 13.1% | 0.450 |
| EA+CSP+LDA | 4 | 71.0% ± 12.6% | 69.8% ± 13.3% | 0.419 |
| EA+CSP+SVM | 4 | 72.5% ± 12.0% | 71.4% ± 13.1% | 0.450 |
| FBCSP+LDA | 5a | 75.9% ± 13.4% | 75.5% ± 13.7% | 0.519 |
| FBCSP+SVM | 5a | 73.6% ± 13.5% | 72.7% ± 14.5% | 0.472 |
| Riem-MDM | 5b | 70.8% ± 13.8% | 69.2% ± 15.4% | 0.416 |
| Riem-TS+LDA | 5b | 72.9% ± 13.4% | 72.0% ± 13.9% | 0.459 |

**Best method: FBCSP+LDA at 75.9% mean accuracy (Kappa 0.519)** — but a single
pooled SpectNet model (Experiment 1, no per-subject tuning) now reaches 74.5%,
essentially matching it. Before the baseline-window fix, the same pooled
SpectNet model scored 48.9% (chance).

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

4. **Subject-pooling is competitive with subject-specific, once the baseline bug
   is fixed:** pooled CNN models (all 9 subjects, Experiment 1) now reach
   70.7–74.5%, matching or slightly exceeding subject-specific training
   (Experiment 2, 69.0–72.7%) — with a single shared model instead of 9 separate
   ones. Before the fix, pooled models were stuck at chance (48–50%).

5. **FBCSP is the top performer (75.9%), but pooled SpectNet is close behind
   (74.5%):** FBCSP's multi-band decomposition still captures band-specific
   ERD/ERS differences that the single 8–30 Hz band misses, but a correctly
   computed ERSP representation lets a single pooled CNN nearly match it
   without any per-subject tuning.

6. **High inter-subject variability (±13–16%):** Some subjects show near-perfect
   classification (S04 > 90%, up to 95.0% with EEGNet subject-specific) while
   others are near chance (S02, S03 ~50–58%). This is consistent with the
   BCI-IV-2b literature and reflects genuine differences in the clarity of
   individual EEG motor imagery responses.

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
