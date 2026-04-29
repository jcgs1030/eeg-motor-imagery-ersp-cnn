# EEG Motor Imagery Classification — ERSP + CNN on BCI Competition IV-2b

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
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
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/               ← place your GDF files here (B01T.gdf ... B09E.gdf)
│   └── processed/         ← auto-generated epochs (.fif) and spectrograms (.npz)
├── notebooks/
│   └── 01_explore_dataset.ipynb
├── src/
│   ├── config.py          ← all pipeline parameters (single source of truth)
│   ├── preprocessing.py   ← GDF loading, bandpass filter, ICA, epoching
│   ├── ersp.py            ← STFT-based ERSP spectrogram generation
│   ├── dataset.py         ← PyTorch Dataset / DataLoader
│   ├── train.py           ← training loop with early stopping
│   ├── evaluate.py        ← metrics, confusion matrix, model comparison
│   └── models/
│       ├── __init__.py
│       ├── eegnet.py      ← EEGNet (Lawhern et al., 2018)
│       ├── shallowconvnet.py   ← ShallowConvNet (Schirrmeister et al., 2017)
│       └── spectnet.py    ← SpectNet (Ruffini et al., 2018)
└── results/
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
**Requires [uv](https://docs.astral.sh/uv/) and Python 3.10+.**

<details>
<summary>Alternative: pip</summary>

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```
</details>

---

## Quick Start

### 1. Place dataset files in `data/raw/`

```
B01T.gdf  B01E.gdf  ...  B09T.gdf  B09E.gdf
```

### 2. Verify dataset
```bash
uv run python src/preprocessing.py --verify
```

### 3. Explore (notebook)
```bash
uv run jupyter notebook notebooks/01_explore_dataset.ipynb
```

### 4. Full pipeline

```bash
uv run python src/preprocessing.py --subject all --suffix both
uv run python src/ersp.py --subject all --suffix both --plot
uv run python src/train.py --model spectnet --all_subjects
uv run python src/train.py --model eegnet --all_subjects
uv run python src/train.py --model shallowconvnet --all_subjects
uv run python src/evaluate.py
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

> To be completed after the experimental phase.

| Method | Mean Acc. (%) | Kappa | F1-score |
|--------|-------------|-------|----------|
| LDA | — | — | — |
| SVM + CSP | — | — | — |
| EEGNet | — | — | — |
| ShallowConvNet | — | — | — |
| SpectNet | — | — | — |

---

## References

- Leeb, R. et al. (2008). *BCI Competition 2008 – Graz Data Set B*. Graz University of Technology.
- Lawhern, V.J. et al. (2018). *EEGNet*. J. Neural Eng., 15(5), 056013.
- Schirrmeister, R.T. et al. (2017). *Deep learning with CNNs for EEG decoding*. Hum. Brain Mapp., 38(11).
- Ruffini, G. et al. (2018). *Deep learning using EEG spectrograms for RBD prognosis*. arXiv.
- Gramfort, A. et al. (2014). *MNE software for MEG and EEG data*. NeuroImage, 86.

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
