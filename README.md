# EEG Motor Imagery Classification — ERSP + CNN on BCI Competition IV-2b

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange?logo=pytorch)](https://pytorch.org/)
[![MNE](https://img.shields.io/badge/MNE--Python-1.6%2B-green)](https://mne.tools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Reproducible pipeline for **binary EEG motor imagery (MI) classification** (left hand vs. right hand) using **Event-Related Spectral Perturbation (ERSP) spectrograms** as input to lightweight **Convolutional Neural Network (CNN)** architectures.

This repository contains the full implementation developed as part of the Master's thesis:

> **"Classification of Hand Movement Intention using Deep Learning and ERSP Analysis of EEG Signals: Implementation and Evaluation of Existing Architectures"**
> Juan Carlos Guerrero Sierra
> Maestría en Ingeniería — Institución Universitaria de Envigado, Colombia, 2026
> Advisor: Hernán Darío Villota Bolaños

---

## Dataset

**BCI Competition IV – Dataset 2b** (Leeb et al., 2008)
- 9 subjects · 5 sessions · 3 electrodes (C3, Cz, C4) · 250 Hz
- Binary MI task: left hand (class 0) vs. right hand (class 1)
- Download: https://www.bbci.de/competition/iv/download/

> **Note:** Dataset files (`.gdf`) are not included due to licensing. Download them and place them in `data/raw/`.

---

## Repository Structure

The code in `src/` is organized by **pipeline stage**, following the data flow.
Each stage reads the previous stage's output from `data/processed/`.

```
eeg-motor-imagery-ersp-cnn/
├── README.md
├── EXPERIMENTS.md                 ← full experimental log: every result, fix and caveat
├── pyproject.toml · uv.lock · .python-version
├── data/
│   ├── raw/                       ← GDF files go here (B0101T.gdf ... B0905E.gdf)
│   └── processed/                 ← S??{T,E}-epo.fif (stage 1) and S??{T,E}-ersp.npz (stage 2)
├── notebooks/                     ← dataset exploration
├── src/
│   ├── config.py                  ← all parameters, shared by every stage
│   │
│   ├── p1_preprocessing/          ── PIPELINE 1: raw EEG → clean epochs
│   │   ├── preprocessing.py       ← GDF loading, 8–30 Hz FIR, ICA, artifact rejection, epoching
│   │   ├── eval_labels.py         ← true labels for sessions 4–5 (from MOABB)
│   │   └── verify_timeline.py     ← checks the trial timeline against raw event markers
│   │
│   ├── p2_ersp/                   ── PIPELINE 2: epochs → ERSP images (CNN branch only)
│   │   └── ersp.py                ← STFT → baseline-normalized dB → (3, 22, 32) image per trial
│   │
│   ├── p3_classification/         ── PIPELINE 3: train, evaluate, compare
│   │   ├── cnn/                   ← reads the ERSP images
│   │   │   ├── models/            ← SpectNet, EEGNet, ShallowConvNet
│   │   │   ├── dataset.py · augmentation.py
│   │   │   ├── train.py           ← one model pooled over the 9 subjects
│   │   │   └── train_subject_specific.py  ← one model per subject
│   │   ├── classical/             ← reads the epochs directly (no images)
│   │   │   ├── train_csp_lda.py · train_ea_csp_lda.py
│   │   │   ├── train_fbcsp.py     ← best single method
│   │   │   └── train_riemannian.py
│   │   └── evaluation/
│   │       ├── evaluate.py · evaluate_cv.py
│   │       ├── experiment_ensemble_validated.py  ← FBCSP + SpectNet, headline result
│   │       ├── experiment_ensemble.py            ← original version (weight tuned on test)
│   │       ├── diagnostic_stats_test.py          ← paired Wilcoxon between methods
│   │       └── generate_final_comparison.py      ← final comparison figures
│   │
│   └── analysis/                  ── outside the main pipeline (does not produce headline results)
│       ├── interpretability/      ← gradcam_spectnet.py, erd_characterization.py
│       ├── experiments/           ← AST, augmentation, C3−C4 channel, fine-tuning, resolution sweep
│       └── diagnostics/           ← scripts that found and confirmed the ERSP bugs, LR sweep
└── results/                       ← outputs (see table below)
```

---

## Pipelines

```
 data/raw/*.gdf
       │
 ① p1_preprocessing ──► data/processed/*-epo.fif ──────────────┐
       │                                                      │
 ② p2_ersp ──► data/processed/*-ersp.npz                       │
       │                                                      │
 ③ p3_classification/cnn                 p3_classification/classical
   SpectNet · EEGNet · ShallowConvNet      CSP · EA-CSP · FBCSP · Riemannian
       │                                                      │
       └──────────────► p3_classification/evaluation ◄────────┘
                        metrics · ensemble · Wilcoxon · figures
```

The classical methods skip stage ② entirely: they work on the filtered
epochs (log-variance or covariance features), not on images. Only the CNNs
use the ERSP images. The ensemble joins both branches.

> Several scripts in `analysis/` and `p3_classification/evaluation/`
> recompute ERSP images in memory from the `.fif` epochs instead of reading
> the cached `.npz` (to test other resolutions or channels). A change to
> `IMG_TIME_BINS` must be checked against them.

### Where to find what

| Question | Code | Results |
|---|---|---|
| How is the raw EEG cleaned? | `src/p1_preprocessing/preprocessing.py`, `src/config.py` | `results/figures/` (preprocessing figures) |
| Where do the evaluation labels come from? | `src/p1_preprocessing/eval_labels.py` | — |
| How is an ERSP image computed? | `src/p2_ersp/ersp.py` | `results/figures/` (grand-average ERSP) |
| What are the CNN architectures? | `src/p3_classification/cnn/models/` | — |
| Pooled CNN results (Exp 1) | `src/p3_classification/cnn/train.py`, `evaluation/evaluate.py` | `results/*_all.pth`, `results/metrics/`, `results/figures/` |
| Subject-specific CNN results (Exp 2) | `src/p3_classification/cnn/train_subject_specific.py` | `results/subject_specific/` |
| Classical baselines (Exp 3–5) | `src/p3_classification/classical/` | `results/csp_lda/`, `results/ea_csp_lda/`, `results/fbcsp/`, `results/riemannian/` |
| Cross-validation within sessions 1–3 | `src/p3_classification/evaluation/evaluate_cv.py` | `results/cv_within_train/` |
| Best result (ensemble) and its p-values | `src/p3_classification/evaluation/experiment_ensemble_validated.py`, `diagnostic_stats_test.py` | `EXPERIMENTS.md` → "Follow-up Experiments" |
| Does the model look at mu/beta over C3/C4? | `src/analysis/interpretability/gradcam_spectnet.py` | `results/figures/gradcam/` |
| Is the ERD/ERS statistically significant? | `src/analysis/interpretability/erd_characterization.py` | `results/figures/erd_characterization/` |
| Why 32 time bins and no blur? | `src/analysis/experiments/experiment_resolution_sweep.py` | `EXPERIMENTS.md` → "ERSP Resolution / Blur Sweep" |
| What bugs were found in the ERSP and how were they confirmed? | `src/analysis/diagnostics/` | `results/baseline_bug_before_fix/`, `EXPERIMENTS.md` → "Diagnostic Finding" sections |

---

## Installation

```bash
git clone https://github.com/jcgs1030/eeg-motor-imagery-ersp-cnn.git
cd eeg-motor-imagery-ersp-cnn

uv sync
```

This creates a virtual environment and installs all dependencies automatically.
**Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.**

---

## Quick Start

### 1. Place dataset files in `data/raw/`

```
B0101T.gdf  B0102T.gdf  B0103T.gdf  B0104E.gdf  B0105E.gdf
...
B0901T.gdf  B0902T.gdf  B0903T.gdf  B0904E.gdf  B0905E.gdf
```
9 subjects × 5 sessions = 45 files total.

### 2. Verify dataset
```bash
uv run preprocess --verify
```

### 3. Explore (notebook)
```bash
uv run jupyter notebook notebooks/01_dataset_exploration.ipynb
```

### 4. Full pipeline

```bash
# ① Preprocessing
uv run preprocess --subject all --suffix both

# ② ERSP images
uv run ersp --subject all --suffix both --plot

# ③ CNN branch
uv run train --model spectnet --all_subjects
uv run train --model eegnet --all_subjects
uv run train --model shallowconvnet --all_subjects
uv run python src/p3_classification/cnn/train_subject_specific.py

# ③ Classical branch
uv run python src/p3_classification/classical/train_csp_lda.py
uv run python src/p3_classification/classical/train_ea_csp_lda.py
uv run python src/p3_classification/classical/train_fbcsp.py
uv run python src/p3_classification/classical/train_riemannian.py

# ③ Evaluation and comparison
uv run evaluate
uv run python src/p3_classification/evaluation/experiment_ensemble_validated.py
uv run python src/p3_classification/evaluation/generate_final_comparison.py
```

Every script accepts `--help`. All scripts can be run by file path from the
repository root; `uv sync` installs `src` as a package, so imports resolve
from anywhere.

---

## Key Parameters (`src/config.py`)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Channels | C3, Cz, C4 | Only electrodes in BCI-IV-2b |
| Bandpass filter | 8–30 Hz (FIR, Hamming) | Mu and beta bands |
| Artifacts | ICA (FastICA, 3 components) + 100 µV rejection | Ocular/muscular artifacts |
| Epoch window | −3.0 to 4.0 s around the cue | Includes the full pre-cue fixation period |
| ERSP baseline | −1.0 to 0.0 s | Must be ≥ STFT window length (see "Known Methodological Findings") |
| Analysis window | −0.5 to 4.0 s | Part of the epoch turned into the image |
| STFT | Hann, 256 samples (1.024 s), hop 9 samples (36 ms, ~96.5% overlap) | ~1 Hz frequency resolution, 125 real time frames |
| ERSP range | 8–30 Hz (22 bins) | Motor-relevant frequencies |
| Normalization | Divisive baseline (dB), clipped to ±6 dB, scaled to [0, 1] | Comparable across trials |
| Image size | 22 × 32 per channel per trial | Reduced from 22 × 128, see `EXPERIMENTS.md` |
| Train / test | Sessions 1–3 / 4–5 | Standard BCI-IV-2b protocol |
| Optimizer | Adam, lr = 0.001, weight decay = 1e−4 | — |
| Early stopping | Patience = 20 on validation loss (15% of training set) | — |

---

## Architectures

| Architecture | Parameters (22 × 32 input) | Reference |
|---|---|---|
| **SpectNet** | 88,050 | Ruffini et al., arXiv, 2018 |
| **ShallowConvNet** | 36,242 | Schirrmeister et al., Hum. Brain Mapp., 2017 |
| **EEGNet** | 6,098 | Lawhern et al., J. Neural Eng., 2018 |

Input: `(batch, 3, 22, 32)`, i.e. 3 channels × 22 frequency bins × 32 time
steps. ShallowConvNet's temporal kernel and pooling lengths scale with the
number of time bins (25/75/15 at 128 bins → 6/19/4 at 32).

---

## Results

Full experimental narrative and per-experiment analysis in [`EXPERIMENTS.md`](EXPERIMENTS.md).

**Protocol:** Sessions 1–3 train / Sessions 4–5 test (official BCI-IV-2b split).  
**Metric:** Mean ± std across 9 subjects, subject-specific models.  
**Chance level:** 50% (balanced binary classification).

> **Critical fix:** Earlier results (~50%) were caused by broken evaluation labels in
> the GDF files. True labels retrieved from MOABB. See [`EXPERIMENTS.md`](EXPERIMENTS.md).

### Experiment 1 — CNN Subject-Pooled

| Model | Test accuracy, 22 × 128 images | Current checkpoint, 22 × 32 images |
|---|---|---|
| EEGNet | 70.0% | not retrained yet (checkpoint is still 22 × 128) |
| ShallowConvNet | 60.3% | 59.3% |
| **SpectNet** | **71.7%** | 68.0% |

The 22 × 128 column is the documented result in `EXPERIMENTS.md`. The pooled
checkpoints for SpectNet and ShallowConvNet were retrained after the switch to
32 time bins; the right column is the test accuracy stored in each checkpoint.

Three independent ERSP bugs were found and fixed here (see "Known
Methodological Findings" below): a too-short baseline window, a numerical
epsilon comparable in scale to the EEG power itself, and an STFT hop so
large that 110 of the 128 output time-bins were pure interpolation. Before
any fix, all three pooled models scored ~48–50% (chance). The baseline-window
fix alone pushed them to 70.7–74.5%; the epsilon fix then cost a few points
back and collapsed ShallowConvNet specifically to near-chance; the hop fix
partially recovered ground for SpectNet (70.0%→71.7%) while leaving
ShallowConvNet's collapse unchanged. See
[`EXPERIMENTS.md`](EXPERIMENTS.md#experiment-1--cnn-subject-pooled) for the
full four-stage before/after table.

### Experiment 2 — CNN Subject-Specific (22 × 32 images)

| Model | Accuracy | F1-score | Kappa |
|---|---|---|---|
| EEGNet | 68.6% ± 12.2% | 66.6% ± 16.1% | 0.372 |
| ShallowConvNet | 54.8% ± 7.4% | 53.3% ± 8.4% | 0.096 |
| SpectNet | 69.5% ± 12.9% | 69.1% ± 13.3% | 0.390 |

Versus 22 × 128 images (paired Wilcoxon, n = 9): SpectNet −1.3 pp (p = 0.074,
lower on 7 of 9 subjects), EEGNet −0.4 pp (p = 0.734), ShallowConvNet +0.3 pp
(p = 1.0). None is significant, but SpectNet's drop is a consistent trend.

### Experiment 3 — CSP Baselines

| Classifier | Accuracy | F1-score | Kappa |
|---|---|---|---|
| CSP + LDA | 71.0% ± 12.6% | 69.8% ± 13.3% | 0.419 |
| CSP + SVM | 72.5% ± 12.0% | 71.4% ± 13.1% | 0.450 |

### Experiment 4 — Euclidean Alignment + CSP

| Classifier | Accuracy | F1-score | Kappa |
|---|---|---|---|
| EA + CSP + LDA | 71.0% ± 12.6% | 69.8% ± 13.3% | 0.419 |
| EA + CSP + SVM | 72.5% ± 12.0% | 71.4% ± 13.1% | 0.450 |

### Experiment 5a — Filter Bank CSP (best result)

| Classifier | Accuracy | F1-score | Kappa |
|---|---|---|---|
| **FBCSP + LDA** | **75.9% ± 13.4%** | **75.5% ± 13.7%** | **0.519** |
| FBCSP + SVM | 73.6% ± 13.5% | 72.7% ± 14.5% | 0.472 |

### Experiment 5b — Riemannian Geometry

| Classifier | Accuracy | F1-score | Kappa |
|---|---|---|---|
| Riem-MDM | 70.8% ± 13.8% | 69.2% ± 15.4% | 0.416 |
| Riem-TS+LDA | 72.9% ± 13.4% | 72.0% ± 13.9% | 0.459 |

### Key Finding

**Best result: an FBCSP+LDA / SpectNet ensemble at 78.7% ± 12.8%** — a
weighted average of FBCSP's predict_proba and SpectNet pooled's softmax
output (weight chosen on a validation split from sessions 1-3, not on the
test set), significantly ahead of either method alone (paired Wilcoxon,
p=0.027 and p=0.008 respectively; n=9 subjects). **Best single method:
FBCSP+LDA at 75.9%**, consistent with the BCI-IV-2b literature for
3-channel classifiers and still ahead of every individual ERSP+CNN method
(best CNN: SpectNet pooled, 71.7% with 22 × 128 images). All three ERSP fixes were worth applying
for scientific correctness, but only the hop fix gave a CNN method a boost on
its own — see "Known Methodological Findings" below. High inter-subject
variance (±12–16%) reflects genuine differences in individual EEG motor
imagery responses (S04 > 90%, S02/S03 ~50–58%).

See [`results/figures/final_comparison_all_methods.png`](results/figures/final_comparison_all_methods.png)
for the full visual comparison including the ensemble, and
[`EXPERIMENTS.md`](EXPERIMENTS.md#follow-up-experiments--statistical-validation-ensembling-improvement-attempts)
for the detailed narrative (statistical validation, the ensemble, and the
ShallowConvNet improvement attempts).

---

## Known Methodological Findings — ERSP Computation (fixed and reprocessed, three times)

Three independent issues were found in the ERSP computation (`src/p2_ersp/ersp.py`)
and fixed in sequence, plus a purely cosmetic colormap change. All were
necessary for scientifically defensible, legible ERSP values; only one of
the three (the hop fix) improved classification accuracy. This is itself
worth reading as a finding, not just the fixes.

**1. Baseline window too short.** `BASELINE = (-0.5, 0.0)` was shorter than
the 1.024 s STFT window used to estimate it, so the baseline reference leaked
post-cue signal — flattening the ERD/ERS contrast. Confirmed with the full 3 s
pre-cue fixation period as baseline instead: the expected contralateral ERD
reappears (S04 alone: -2.87 dB; 9 subjects pooled: -0.98 dB, same direction).
See [`results/baseline_bug_before_fix/`](results/baseline_bug_before_fix/) for
the before/diagnostic/confirmation images. Fixed by extending
`EPOCH_TMIN`/`BASELINE` in `src/config.py`.

**2. Numerical epsilon comparable to signal scale.** A fixed `epsilon = 1e-12`
added for `log(0)` safety was the same order of magnitude as the actual EEG
power (MNE stores amplitude in volts, so squared STFT power sits in the
~1e-12 to 1e-27 range) — this silently biased every ERSP value strongly
negative, saturating 77-96% of the mu/beta bands at the ±6 dB clip floor.
Fixed by scaling the epsilon to each trial's own power range
(`eps = power.max() * 1e-12`), which brought the median ERSP value from an
implausible -12/-13 dB down to a literature-plausible -2 dB. Confirmed
visually with a classic Pfurtscheller-style ERD curve
(`results/figures/erd_curve_classic_grandaverage_C3.png`): beta-band grand
average now shows a clean -10 to -12% contralateral ERD.

**3. STFT hop too large relative to the output size.** `_resize_2d()` stretches
the raw STFT output to `(22, 128)` (now `(22, 32)`) via bilinear interpolation. The frequency
axis needed ~1.00x (22 real bins already). The time axis needed **7.11x**:
with the original `STFT_HOP=64` (256ms), the 4.5s analysis window produced
only 18 real STFT frames — the other ~110 output time-bins were pure
interpolation, giving an illusion of ~35ms temporal resolution when the real
resolution was ~256ms. Raised by the thesis advisor. Fixed by reducing
`STFT_HOP` to 9 samples (36ms, 125 real frames, ~1.02x interpolation).

**4. Colormap choice (visualisation only, no data/model change).** Even with
all three fixes above, the grand-average heatmap plotted with a smooth
diverging colormap (`RdYlBu_r`) still made the ERD hard to see at a glance.
Switched to `jet` (the classic EEGLAB-style ERSP convention) — its sharper
hue transitions make the 2-6 dB contralateral signal visible without further
explanation. See `src/analysis/diagnostics/diagnostic_db_heatmap.py` for a standalone script that
plots the grand-average in real (signed) dB units instead of the [0,1] scale
used for training.

**Net effect on classification:** the baseline-window fix alone took pooled
CNNs from chance (~48-50%) to 70.7-74.5%. The epsilon fix then cost 2-4 points
for EEGNet/SpectNet and collapsed ShallowConvNet to near-chance
(54.5% ± 6.9%, unrecovered by the hop fix). The hop fix was net-neutral-to-
positive, improving SpectNet pooled to 71.7% (its best result across all
three fixes) without hurting EEGNet. Full before/after tables and the working
hypotheses are in
[`EXPERIMENTS.md`](EXPERIMENTS.md#second-diagnostic-finding--ersp-epsilon-bias-fixed-and-reprocessed)
(epsilon) and
[`EXPERIMENTS.md`](EXPERIMENTS.md#third-diagnostic-finding--stft-hop--resize_2d-undersampling-fixed-and-reprocessed)
(hop).

---

## References

- Leeb, R. et al. (2008). *BCI Competition 2008 – Graz Data Set B*. Graz University of Technology.
- Lawhern, V.J. et al. (2018). *EEGNet: A compact convolutional neural network for EEG-based BCI*. J. Neural Eng., 15(5), 056013.
- Schirrmeister, R.T. et al. (2017). *Deep learning with convolutional neural networks for EEG decoding*. Hum. Brain Mapp., 38(11).
- Ruffini, G. et al. (2018). *Deep learning using EEG spectrograms for prognosis of neurodegeneration*. arXiv.
- Ang, K.K. et al. (2008). *Filter Bank Common Spatial Pattern (FBCSP) algorithm*. Proc. IEEE IJCNN.
- Barachant, A. et al. (2012). *Multiclass BCI classification by Riemannian geometry*. IEEE TBME, 59(4).
- Barachant, A. et al. (2013). *Classification of covariance matrices using a Riemannian-based kernel*. Neurocomputing, 112.
- He, H. & Wu, D. (2019). *Transfer learning for EEG-based BCI: A review*. IEEE TNSRE, 27(1).
- He, H. et al. (2020). *Transfer learning for BCI: A Euclidean space data alignment approach*. IEEE TNSRE, 68(6).
- Gramfort, A. et al. (2014). *MNE software for processing MEG and EEG data*. NeuroImage, 86.

---

## Citation

```bibtex
@mastersthesis{guerrero2026eeg,
  author  = {Guerrero Sierra, Juan Carlos},
  title   = {Classification of Hand Movement Intention using Deep Learning
             and ERSP Analysis of EEG Signals},
  school  = {Institución Universitaria de Envigado},
  year    = {2026},
  address = {Envigado, Colombia},
  note    = {https://github.com/jcgs1030/eeg-motor-imagery-ersp-cnn}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
