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
# Training sessions (1-3): offline paradigm — true class labels in the GDF
EVENT_LEFT    = 769   # left-hand imagery → class 0
EVENT_RIGHT   = 770   # right-hand imagery → class 1
# Evaluation sessions (4-5): GDF only contains generic event types, NOT labels.
# 781 = BCI feedback (continuous), 783 = cue onset (class unknown in the file).
# True labels are retrieved from MOABB via src/eval_labels.py and applied
# in chronological trial order during preprocessing.
EVENT_CUE_EVAL = 783  # cue onset marker in eval sessions — used for epoch alignment only
EVENT_LABELS   = {769: 0, 770: 1}   # training labels only
CLASS_NAMES    = {0: "Left", 1: "Right"}

# ── Preprocessing ─────────────────────────────────────────────────────────────
# Band-pass filter (mu and beta bands associated with motor activity)
FILT_LOW      = 8.0   # Hz
FILT_HIGH     = 30.0  # Hz
FILT_METHOD   = "fir"  # FIR filter with Hamming window (MNE default)

# Epoching (temporal segmentation)
# EPOCH_TMIN reaches back to the trial start (verified empirically in
# verify_timeline.py: trial_start -> cue onset = 3.000s exact, std=0, across
# all 45 GDF files), NOT just -0.5s. This gives the baseline reference window
# enough buffer before the epoch's own extraction edge to avoid STFT
# zero-padding contamination (see EXPERIMENTS.md "Diagnostic Finding" and its
# reprocessing follow-up). BASELINE is a clean 1s window, 2s away from that edge.
EPOCH_TMIN    = -3.0  # s before cue onset (full pre-cue fixation period)
EPOCH_TMAX    =  4.0  # s after cue onset
BASELINE      = (-1.0, 0.0)   # clean baseline reference window for ERSP normalisation

# ERSP analysis/output window — this is the time axis that actually gets
# resized to IMG_TIME_BINS and fed to the models. Unchanged from the original
# pipeline; only the raw extraction (EPOCH_TMIN above) was extended to fix the
# baseline measurement.
ANALYSIS_TMIN = -0.5
ANALYSIS_TMAX =  4.0

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
