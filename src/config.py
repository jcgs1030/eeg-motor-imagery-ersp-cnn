"""
config.py
---------
Centralized parameters for the entire pipeline.
Modify here and the change propagates to all modules.
"""

from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent
DATA_RAW      = ROOT_DIR / "data" / "raw"
DATA_PROC     = ROOT_DIR / "data" / "processed"
RESULTS_DIR   = ROOT_DIR / "results"
FIGURES_DIR   = RESULTS_DIR / "figures"
METRICS_DIR   = RESULTS_DIR / "metrics"

# Create directories if they do not exist
for d in [DATA_PROC, FIGURES_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── BCI-IV-2b dataset ────────────────────────────────────────────────────────
N_SUBJECTS    = 9
SUBJECTS      = list(range(1, N_SUBJECTS + 1))   # [1, 2, ..., 9]

# File naming: B<subject><session><suffix>.gdf  e.g. B0101T.gdf, B0104E.gdf
# 9 subjects × 5 sessions = 45 files total
TRAIN_SUFFIX  = "T"   # sessions 1-3 (training)
EVAL_SUFFIX   = "E"   # sessions 4-5 (evaluation)

# Channels of interest (positions over the motor cortex)
CHANNELS      = ["C3", "Cz", "C4"]
N_CHANNELS    = len(CHANNELS)

# Dataset sampling frequency
SFREQ         = 250   # Hz

# BCI-IV-2b event codes
# Training sessions (1-3): offline paradigm
EVENT_LEFT    = 769   # left-hand imagery → class 0
EVENT_RIGHT   = 770   # right-hand imagery → class 1
# Evaluation sessions (4-5): online feedback paradigm uses different codes
EVENT_LEFT_ONLINE  = 781
EVENT_RIGHT_ONLINE = 783
EVENT_LABELS  = {769: 0, 770: 1, 781: 0, 783: 1}
CLASS_NAMES   = {0: "Left", 1: "Right"}

# ── Preprocessing ─────────────────────────────────────────────────────────────
# Band-pass filter (mu and beta bands associated with motor activity)
FILT_LOW      = 8.0   # Hz
FILT_HIGH     = 30.0  # Hz
FILT_METHOD   = "fir"  # FIR filter with Hamming window (MNE default)

# Epoching (temporal segmentation)
EPOCH_TMIN    = -0.5  # s before cue onset (baseline)
EPOCH_TMAX    =  4.0  # s after cue onset
BASELINE      = (-0.5, 0.0)   # window for baseline correction

# Epoch rejection threshold (maximum amplitude artefact)
REJECT_THRESH = 100e-6   # 100 µV on any channel

# ICA
ICA_N_COMPS   = 3     # equal to the number of EEG channels
ICA_METHOD    = "fastica"
ICA_SEED      = 42

# ── ERSP spectrogram ─────────────────────────────────────────────────────────
# Transform: STFT with Hann window
STFT_WIN_LEN  = 256   # samples (1.024 s at 250 Hz)
STFT_OVERLAP  = 0.75  # 75% overlap → hop of 64 samples
STFT_HOP      = int(STFT_WIN_LEN * (1 - STFT_OVERLAP))  # 64 samples

# Frequency range of ERSP images
ERSP_FMIN     = 8.0   # Hz (lower mu band)
ERSP_FMAX     = 30.0  # Hz (upper beta band)

# Output image dimensions (per channel, per trial)
# 22 frequency bins × 128 time steps
IMG_FREQ_BINS = 22
IMG_TIME_BINS = 128
IMG_SIZE      = (IMG_FREQ_BINS, IMG_TIME_BINS)

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
MAX_EPOCHS    = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 20       # early stopping on validation loss
VAL_SPLIT     = 0.15     # fraction of training set reserved for validation
RANDOM_SEED   = 42

# Compute device (CPU — no dedicated GPU assumed)
DEVICE        = "cpu"

# ── Available models ──────────────────────────────────────────────────────────
MODELS        = ["eegnet", "shallowconvnet", "spectnet"]

# ── Classical baselines ───────────────────────────────────────────────────────
BASELINES     = ["lda", "svm_csp"]

# ── Evaluation ────────────────────────────────────────────────────────────────
# BCI-IV-2b protocol: sessions 1-3 training, sessions 4-5 evaluation
TRAIN_SESSIONS = [1, 2, 3]
TEST_SESSIONS  = [4, 5]

# Metrics to report
METRICS       = ["accuracy", "precision", "recall", "f1", "kappa"]
