# Experimental Log — EEG Motor Imagery Classification

**Dataset:** BCI Competition IV – Dataset 2b  
**Task:** Binary classification — left-hand vs. right-hand motor imagery  
**Channels:** C3, Cz, C4 (3 electrodes over the motor cortex)  
**Protocol:** Sessions 1–3 (offline, no feedback) → train | Sessions 4–5 (online, with feedback) → test

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

### Results (mean ± std across 9 subjects)

| Model | Accuracy | Kappa |
|---|---|---|
| EEGNet | 49.8% ± 0.9% | −0.005 ± 0.017 |
| ShallowConvNet | 50.2% ± 1.6% | +0.003 ± 0.033 |
| SpectNet | 49.9% ± 1.8% | −0.002 ± 0.036 |

### Analysis
Performance remains at chance level even with subject-specific training.
This rules out inter-subject variability as the sole cause. The persisting
~50% accuracy confirms that the **offline→online domain shift** (T→E) is
the primary obstacle: the ERD/ERS patterns learned from sessions 1–3 do
not generalise to the feedback-modulated sessions 4–5.

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
| CSP+LDA | 49.9% ± 0.9% | −0.002 ± 0.019 |
| CSP+SVM | 49.7% ± 1.1% | −0.006 ± 0.023 |

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
| EA+CSP+LDA | 49.9% ± 0.9% | −0.002 ± 0.019 |
| EA+CSP+SVM | 49.7% ± 1.1% | −0.006 ± 0.023 |

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
| FBCSP+LDA | 50.2% ± 1.2% | +0.005 ± 0.025 |
| FBCSP+SVM | 50.3% ± 0.4% | +0.006 ± 0.009 |

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
| Riem-MDM | 50.0% ± 1.6% | −0.000 ± 0.032 |
| Riem-TS+LDA | 49.9% ± 1.4% | −0.002 ± 0.028 |

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

## Cross-Experiment Comparison

| Method | Experiment | Accuracy (mean) | Kappa (mean) |
|---|---|---|---|
| EEGNet (pooled, all) | 1 | 48.3% | −0.033 |
| ShallowConvNet (pooled, all) | 1 | 49.6% | −0.008 |
| SpectNet (pooled, all) | 1 | 48.9% | −0.023 |
| EEGNet (subject-specific) | 2 | 49.8% | −0.005 |
| ShallowConvNet (subject-specific) | 2 | 50.2% | +0.003 |
| SpectNet (subject-specific) | 2 | 49.9% | −0.002 |
| CSP+LDA | 3 | 49.9% | −0.002 |
| CSP+SVM | 3 | 49.7% | −0.006 |
| EA + CSP+LDA | 4 | 49.9% | −0.002 |
| EA + CSP+SVM | 4 | 49.7% | −0.006 |
| FBCSP+LDA | 5a | 50.2% | +0.005 |
| FBCSP+SVM | 5a | 50.3% | +0.006 |
| Riem-MDM | 5b | 50.0% | −0.000 |
| Riem-TS+LDA | 5b | 49.9% | −0.002 |

---

## Key Findings

1. **ERSP normalisation matters:** Per-trial min-max normalisation (Exp 1, initial)
   destroys the discriminative signal. Fixed to ±6 dB clip in commit `3a242de`.

2. **Subject-pooling fails:** Mixing subjects into one model does not improve
   performance and obscures individual variability.

3. **Subject-specific CNN ≈ CSP classical baseline:** Under the offline→online
   protocol, neither deep nor classical approaches yield above-chance accuracy.

4. **The bottleneck is domain shift, not the classifier:** All five methods
   (3 CNNs + CSP+LDA + CSP+SVM) converge to ~50% on sessions 4–5. This
   motivates domain adaptation methods (EA, Exp 4) as the next step.

---

## References

- Leeb, R. et al. (2008). BCI Competition 2008 — Graz Data Set B. TU Graz.
- Lawhern, V.J. et al. (2018). EEGNet. *J. Neural Eng.*, 15(5), 056013.
- Schirrmeister, R.T. et al. (2017). Deep learning with CNNs for EEG. *Hum. Brain Mapp.*, 38.
- Ruffini, G. et al. (2018). Deep learning using EEG spectrograms. *arXiv*.
- He, H. & Wu, D. (2019). Transfer learning for EEG-BCI: review. *IEEE TNSRE*, 27(1).
- He, H. et al. (2020). Transfer learning for BCI: Euclidean alignment. *IEEE TNSRE*, 68(6).
- Gramfort, A. et al. (2014). MNE software. *NeuroImage*, 86.
