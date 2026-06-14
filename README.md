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

```
eeg-motor-imagery-ersp-cnn/
├── README.md
├── EXPERIMENTS.md             ← full experimental log with results and analysis
├── pyproject.toml
├── uv.lock
├── .python-version
├── data/
│   ├── raw/                   ← place your GDF files here (B0101T.gdf ... B0905E.gdf)
│   └── processed/             ← auto-generated epochs (.fif) and spectrograms (.npz)
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_gdf_visual_exploration.ipynb
├── src/
│   ├── config.py              ← all pipeline parameters (single source of truth)
│   ├── preprocessing.py       ← GDF loading, bandpass filter, epoching
│   ├── ersp.py                ← STFT-based ERSP spectrogram generation
│   ├── dataset.py             ← PyTorch Dataset / DataLoader
│   ├── train.py               ← CNN training — subject-pooled
│   ├── train_subject_specific.py  ← CNN training — one model per subject
│   ├── train_csp_lda.py       ← classical baseline: CSP + LDA / SVM
│   ├── train_ea_csp_lda.py    ← domain adaptation: EA + CSP + LDA / SVM
│   ├── evaluate.py            ← metrics, confusion matrix, model comparison
│   └── models/
│       ├── eegnet.py          ← EEGNet (Lawhern et al., 2018)
│       ├── shallowconvnet.py  ← ShallowConvNet (Schirrmeister et al., 2017)
│       └── spectnet.py        ← SpectNet (Ruffini et al., 2018)
└── results/
    ├── figures/               ← Exp 1: pooled CNN
    ├── metrics/
    ├── subject_specific/      ← Exp 2: subject-specific CNN
    │   ├── figures/
    │   └── metrics/
    ├── csp_lda/               ← Exp 3: CSP + LDA / SVM
    │   ├── figures/
    │   └── metrics/
    └── ea_csp_lda/            ← Exp 4: EA + CSP + LDA / SVM
        ├── figures/
        └── metrics/
```

---

## Pipeline Overview

```
BCI-IV-2b GDF files
        │
        ▼
1. Preprocessing       (src/preprocessing.py)
   ├─ Channels: C3, Cz, C4
   ├─ Bandpass: 8–30 Hz (FIR Hamming)
   ├─ Artifacts: ICA (FastICA, 3 components)
   └─ Epochs: −0.5 to 4.0 s
        │
        ▼
2. ERSP Generation     (src/ersp.py)
   ├─ STFT: Hann window, 256 samples, 75% overlap
   ├─ Range: 8–30 Hz → 22 bins
   ├─ Normalization: divisive baseline (dB)
   └─ Output: (3, 22, 128) tensor per trial
        │
        ▼
3. CNN Classification  (src/train.py + src/models/)
   ├─ EEGNet          ~2,300 parameters
   ├─ ShallowConvNet  ~47,000 parameters
   └─ SpectNet        ~1,500 parameters
        │
        ▼
4. Evaluation          (src/evaluate.py)
   ├─ Protocol: sessions 1–3 train | sessions 4–5 test
   ├─ Metrics: accuracy, kappa, F1, confusion matrix
   └─ Baselines: LDA, SVM+CSP
```

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
uv run preprocess --subject all --suffix both
uv run ersp --subject all --suffix both --plot
uv run train --model spectnet --all_subjects
uv run train --model eegnet --all_subjects
uv run train --model shallowconvnet --all_subjects
uv run evaluate
```

---

## Key Parameters (src/config.py)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Bandpass filter | 8–30 Hz | Mu and beta bands |
| Epoch window | −0.5 to 4.0 s | Includes pre-stimulus baseline |
| STFT window | 256 samples (1.024 s) | ~1 Hz frequency resolution |
| STFT overlap | 75% | Adequate temporal resolution |
| ERSP range | 8–30 Hz (22 bins) | Motor-relevant frequencies |
| Normalization | Divisive baseline (dB) | Relative to pre-stimulus |
| Image size | 22 × 128 px | Per channel, per trial |
| Train / test | Sessions 1–3 / 4–5 | Standard BCI-IV-2b protocol |
| Optimizer | Adam lr=0.001 | Weight decay = 1e−4 |
| Early stopping | Patience = 20 | On validation loss |

---

## Architectures

| Architecture | Parameters | Reference |
|---|---|---|
| **EEGNet** | ~2,300 | Lawhern et al., J. Neural Eng., 2018 |
| **ShallowConvNet** | ~47,000 | Schirrmeister et al., Hum. Brain Mapp., 2017 |
| **SpectNet** | ~1,500 | Ruffini et al., arXiv, 2018 |

All input: `(batch, 3, 22, 128)` — 3 channels × 22 freq. bins × 128 time steps.

---

## Results

Full experimental narrative and per-experiment analysis in [`EXPERIMENTS.md`](EXPERIMENTS.md).

**Protocol:** Sessions 1–3 (offline, no feedback) → train | Sessions 4–5 (online, with feedback) → test.  
**Metric:** Mean ± std across 9 subjects, subject-specific models (one model per subject).  
**Chance level:** 50% (balanced binary classification).

### Experiment 1 — CNN Subject-Pooled (all 9 subjects concatenated)

| Model | Accuracy | F1-score | Kappa |
|---|---|---|---|
| EEGNet | 48.3% | 47.9% | −0.033 |
| ShallowConvNet | 49.6% | 49.2% | −0.008 |
| SpectNet | 48.9% | 48.9% | −0.023 |

### Experiment 2 — CNN Subject-Specific

| Model | Accuracy | F1-score | Kappa |
|---|---|---|---|
| EEGNet | 49.8% ± 0.9% | 45.4% ± 6.9% | −0.005 ± 0.017 |
| ShallowConvNet | 50.2% ± 1.6% | 50.0% ± 1.7% | +0.003 ± 0.033 |
| SpectNet | 49.9% ± 1.8% | 47.1% ± 5.9% | −0.002 ± 0.036 |

### Experiment 3 — Classical Baseline: CSP

| Classifier | Accuracy | F1-score | Kappa |
|---|---|---|---|
| CSP + LDA | 49.9% ± 0.9% | 48.2% ± 2.4% | −0.002 ± 0.019 |
| CSP + SVM | 49.7% ± 1.1% | 48.1% ± 2.0% | −0.006 ± 0.023 |

### Experiment 4 — Domain Adaptation: Euclidean Alignment + CSP

| Classifier | Accuracy | F1-score | Kappa |
|---|---|---|---|
| EA + CSP + LDA | 49.9% ± 0.9% | 48.2% ± 2.4% | −0.002 ± 0.019 |
| EA + CSP + SVM | 49.7% ± 1.1% | 48.1% ± 2.0% | −0.006 ± 0.023 |

### Experiment 5a — Filter Bank CSP (FBCSP)

| Classifier | Accuracy | F1-score | Kappa |
|---|---|---|---|
| FBCSP + LDA | 50.2% ± 1.2% | 49.5% ± 1.6% | +0.005 ± 0.025 |
| FBCSP + SVM | 50.3% ± 0.4% | 48.9% ± 2.6% | +0.006 ± 0.009 |

### Experiment 5b — Riemannian Geometry (MDM / TS+LDA)

| Classifier | Accuracy | F1-score | Kappa |
|---|---|---|---|
| Riem-MDM | 50.0% ± 1.6% | 47.9% ± 3.7% | −0.000 ± 0.032 |
| Riem-TS+LDA | 49.9% ± 1.4% | 48.4% ± 2.1% | −0.002 ± 0.028 |

### Key Finding

All 11 methods across 5 experiments converge to chance level (~50%) on sessions 4–5.
The bottleneck is the **structural domain shift** between the offline training paradigm
(sessions 1–3, no feedback) and the online evaluation paradigm (sessions 4–5, with
visual feedback), not the choice of model or feature extraction method.

See [`results/riemannian/figures/comparison_all_methods.png`](results/riemannian/figures/comparison_all_methods.png)
for the full visual comparison across all experiments.

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
