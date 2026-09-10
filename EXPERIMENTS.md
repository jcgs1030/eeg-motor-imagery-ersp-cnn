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

## Diagnostic Finding — Baseline Window Contamination (identified, reprocessing pending)

> **Status: identified and confirmed, NOT yet applied to the production pipeline.**
> Experiments 1–5b below still use the original `BASELINE = (-0.5, 0.0)` window.
> A full reprocessing (regenerate all ERSP tensors + retrain all experiments) is
> planned as a follow-up once this finding is documented.

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

### Planned fix (not yet applied)

Extend `BASELINE` (and `EPOCH_TMIN`) to make use of the full 3 s pre-cue fixation
period, regenerate all `S0x{T,E}-epo.fif` and `S0x{T,E}-ersp.npz` files, and re-run
Experiments 1–5b. Until that reprocessing happens, all results below should be read
with this caveat in mind — they were trained on ERSP tensors with the contaminated
baseline.

### STFT vs. Wavelet transform (`src/ersp.py: compute_ersp_image_wavelet`, `plot_stft_vs_wavelet`)

As a related methodological check, a Morlet continuous-wavelet-transform version of
the ERSP was implemented for side-by-side comparison (`results/figures/stft_vs_wavelet_*.png`,
`results/figures/ersp_grand_average_wavelet_{T,E,both}.png`). Because wavelet time
resolution adapts with frequency (narrower window at high frequencies) instead of
using STFT's single fixed 1.024 s window, it is inherently less exposed to the same
edge-contamination issue. The wavelet grand-average shows a short, sharper transient
(~0.3–0.5 s post-cue) near the mu/beta border that the STFT version smooths away —
consistent with STFT's coarser temporal resolution, independent of the baseline bug
above.

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

| Model | Subjects | Accuracy | Kappa |
|---|---|---|---|
| EEGNet | 1 | 49.4% | −0.012 |
| ShallowConvNet | 1 | 48.6% | −0.028 |
| SpectNet | 1 | 48.9% | −0.022 |
| EEGNet | all (9) | 48.3% | −0.033 |
| ShallowConvNet | all (9) | 49.6% | −0.008 |
| SpectNet | all (9) | 48.9% | −0.023 |

### Analysis
All models perform at chance level (50%). Confusion matrices show models
collapse to predicting one class. Two causes identified:

1. **Subject-pooled approach is invalid for EEG:** inter-subject variability
   in signal morphology is too large for a single model to generalise.
2. **Domain shift T→E:** sessions 1–3 (offline, no feedback) differ
   structurally from sessions 4–5 (online, with visual feedback). Models
   trained on offline sessions cannot generalise to online conditions.

---

## Experiment 2 — CNN Subject-Specific

**Script:** `src/train_subject_specific.py`  
**Results:** `results/subject_specific/figures/`, `results/subject_specific/metrics/`

### Design
One model trained per (architecture, subject) pair using only that subject's
own sessions 1–3. Evaluated on that subject's sessions 4–5.
This is the standard intra-subject protocol in the BCI literature.

### Results (mean ± std across 9 subjects) — **corrected labels**

| Model | Accuracy | F1-score | Kappa |
|---|---|---|---|
| EEGNet | 72.6% ± 15.6% | 68.5% ± 21.9% | 0.451 ± 0.312 |
| ShallowConvNet | 70.2% ± 14.3% | 70.1% ± 14.3% | 0.404 ± 0.286 |
| SpectNet | 73.3% ± 14.8% | 70.8% ± 18.9% | 0.465 ± 0.297 |

### Analysis
Subject-specific CNNs now well above chance (70–73% mean), confirming that
the EEG signal contains discriminative MI patterns. High variance across
subjects (±14–15%) reflects genuine inter-subject variability: some subjects
(e.g. S04) are "BCI-literate" with very clear ERD/ERS patterns, while
others (e.g. S03) show weaker or inconsistent responses.

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
The classical CSP baseline — which is specifically designed for this type
of spatial filtering and is not susceptible to the ERSP representation
problems — also obtains chance-level accuracy. This is the definitive
confirmation that the performance ceiling is **not caused by the model
architecture or the ERSP representation**, but by the inherent
offline→online domain shift in BCI-IV-2b.

The literature reports 65–85% accuracy on this dataset using methods that
either (a) adapt the classifier using early online-session trials, or
(b) apply domain alignment techniques before classification.

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
EA aligns the covariance structure of each session to the identity, which should
reduce the statistical discrepancy between offline training and online test sessions.
However, the accuracy remains at chance level (~50%). This reveals that the
offline→online shift in BCI-IV-2b is not primarily a **scaling or orientation**
difference in the covariance space (which EA corrects), but a deeper **distributional
shift** in the spectral patterns driven by the feedback modality change.

The identical performance of EA+CSP and plain CSP (Experiment 3) is itself an
informative finding: the domain shift is not of the type that Euclidean alignment
addresses, and more advanced methods (e.g., Riemannian geometry, adaptive BCI
classifiers) would be required to bridge the offline→online gap for this dataset.

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
FBCSP marginally improves over broad-band CSP (49.9% → 50.2%) but the
improvement is not meaningful — both remain at chance level. The multi-band
feature enrichment does not compensate for the offline→online distributional
shift. Notably, S02 achieves 53% with FBCSP+LDA, showing that individual
subjects may have stronger band-specific ERD/ERS patterns, but the effect
is not consistent across the cohort.

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
Despite the theoretical robustness of Riemannian methods to inter-session
variability, performance remains at chance. The congruence invariance of
geodesic distances does not help here because the domain shift is not a
linear transformation of the covariance structure — it reflects a fundamentally
different neural process (feedback-modulated imagery vs. offline imagery).

The collective result across Experiments 3–5b (CSP, EA, FBCSP, MDM, TS+LDA)
constitutes strong evidence that no standard signal-processing method can
bridge the offline→online gap in BCI-IV-2b using sessions 1–3 as training
data without any online adaptation.

---

## Cross-Experiment Comparison — Corrected Results

All results below use **true evaluation labels** from MOABB (fixed pipeline).
Metric: mean ± std accuracy across 9 subjects, subject-specific protocol,
sessions 1–3 train / sessions 4–5 test.

| Method | Exp | Accuracy | F1-score | Kappa |
|---|---|---|---|---|
| EEGNet (pooled, all 9) | 1 | 72.4% | — | — |
| ShallowConvNet (pooled, all 9) | 1 | 70.6% | — | — |
| SpectNet (pooled, all 9) | 1 | 73.5% | — | — |
| EEGNet (subject-specific) | 2 | 72.6% ± 15.6% | 68.5% ± 21.9% | 0.451 |
| ShallowConvNet (subject-specific) | 2 | 70.2% ± 14.3% | 70.1% ± 14.3% | 0.404 |
| SpectNet (subject-specific) | 2 | 73.3% ± 14.8% | 70.8% ± 18.9% | 0.465 |
| CSP+LDA | 3 | 71.0% ± 12.6% | 69.8% ± 13.3% | 0.419 |
| CSP+SVM | 3 | 72.5% ± 12.0% | 71.4% ± 13.1% | 0.450 |
| EA+CSP+LDA | 4 | 71.0% ± 12.6% | 69.8% ± 13.3% | 0.419 |
| EA+CSP+SVM | 4 | 72.5% ± 12.0% | 71.4% ± 13.1% | 0.450 |
| **FBCSP+LDA** | **5a** | **75.9% ± 13.4%** | **75.5% ± 13.7%** | **0.519** |
| FBCSP+SVM | 5a | 73.6% ± 13.5% | 72.7% ± 14.5% | 0.472 |
| Riem-MDM | 5b | 70.8% ± 13.8% | 69.2% ± 15.4% | 0.416 |
| Riem-TS+LDA | 5b | 72.9% ± 13.4% | 72.0% ± 13.9% | 0.459 |

**Best method: FBCSP+LDA at 75.9% mean accuracy (Kappa 0.519).**

---

## Key Findings

1. **Evaluation labels were broken (critical bug):** The GDF evaluation files contain
   no true class labels. The original code mapped generic event types (781, 783) as
   left/right, forcing all classifiers to ~50%. Fixed by retrieving true labels from
   MOABB (`src/eval_labels.py`) and applying them during preprocessing.

2. **ERSP per-trial normalisation matters:** Per-trial min-max normalisation destroys
   inter-trial comparability. Fixed to ±6 dB clip (commit `3a242de`).

3. **Subject-pooling is competitive with subject-specific:** Once labels are correct,
   pooled CNN models (using all 9 subjects) reach 70–73%, comparable to subject-specific
   training, suggesting the ERSP representation captures cross-subject patterns.

4. **FBCSP is the top performer (75.9%):** The multi-band decomposition captures
   band-specific ERD/ERS differences that the single 8–30 Hz band misses.

5. **High inter-subject variability (±13–16%):** Some subjects show near-perfect
   classification (S04 > 90%) while others are near chance (S03 ~55%). This is
   consistent with the BCI-IV-2b literature and reflects genuine differences in
   the clarity of individual EEG motor imagery responses.

---

## References

- Leeb, R. et al. (2008). BCI Competition 2008 — Graz Data Set B. TU Graz.
- Lawhern, V.J. et al. (2018). EEGNet. *J. Neural Eng.*, 15(5), 056013.
- Schirrmeister, R.T. et al. (2017). Deep learning with CNNs for EEG. *Hum. Brain Mapp.*, 38.
- Ruffini, G. et al. (2018). Deep learning using EEG spectrograms. *arXiv*.
- He, H. & Wu, D. (2019). Transfer learning for EEG-BCI: review. *IEEE TNSRE*, 27(1).
- He, H. et al. (2020). Transfer learning for BCI: Euclidean alignment. *IEEE TNSRE*, 68(6).
- Gramfort, A. et al. (2014). MNE software. *NeuroImage*, 86.
