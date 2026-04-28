"""
config.py
---------
Parámetros centralizados de todo el pipeline.
Modifica aquí y el cambio se propaga a todos los módulos.
"""

from pathlib import Path

# ── Rutas del proyecto ──────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent
DATA_RAW      = ROOT_DIR / "data" / "raw"
DATA_PROC     = ROOT_DIR / "data" / "processed"
RESULTS_DIR   = ROOT_DIR / "results"
FIGURES_DIR   = RESULTS_DIR / "figures"
METRICS_DIR   = RESULTS_DIR / "metrics"

# Crear directorios si no existen
for d in [DATA_PROC, FIGURES_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset BCI-IV-2b ───────────────────────────────────────────────────────
N_SUBJECTS    = 9
SUBJECTS      = list(range(1, N_SUBJECTS + 1))   # [1, 2, ..., 9]

# Nombres de archivo: B01T.gdf, B01E.gdf, etc.
TRAIN_SUFFIX  = "T"   # sesiones 1-3 (entrenamiento)
EVAL_SUFFIX   = "E"   # sesiones 4-5 (evaluación)

# Canales de interés (posiciones sobre la corteza motora)
CHANNELS      = ["C3", "Cz", "C4"]
N_CHANNELS    = len(CHANNELS)

# Frecuencia de muestreo del dataset
SFREQ         = 250   # Hz

# Códigos de eventos del BCI-IV-2b
EVENT_LEFT    = 769   # imaginación de mano izquierda → clase 0
EVENT_RIGHT   = 770   # imaginación de mano derecha   → clase 1
EVENT_LABELS  = {769: 0, 770: 1}
CLASS_NAMES   = {0: "Izquierda", 1: "Derecha"}

# ── Preprocesamiento ────────────────────────────────────────────────────────
# Filtro pasa banda (bandas mu y beta asociadas a actividad motora)
FILT_LOW      = 8.0   # Hz
FILT_HIGH     = 30.0  # Hz
FILT_METHOD   = "fir"  # filtro FIR con ventana Hamming (MNE default)

# Epoching (segmentación temporal)
EPOCH_TMIN    = -0.5  # s antes del onset del cue (línea base)
EPOCH_TMAX    =  4.0  # s después del onset del cue
BASELINE      = (-0.5, 0.0)   # ventana para corrección de línea base

# Rechazo de épocas con artefactos (umbral de amplitud máxima)
REJECT_THRESH = 100e-6   # 100 µV en cualquier canal

# ICA
ICA_N_COMPS   = 3     # igual al número de canales EEG
ICA_METHOD    = "fastica"
ICA_SEED      = 42

# ── Espectrograma ERSP ──────────────────────────────────────────────────────
# Transformada: STFT con ventana Hann
STFT_WIN_LEN  = 256   # muestras (1.024 s a 250 Hz)
STFT_OVERLAP  = 0.75  # 75% de solapamiento → paso de 64 muestras
STFT_HOP      = int(STFT_WIN_LEN * (1 - STFT_OVERLAP))  # 64 muestras

# Rango frecuencial de las imágenes ERSP
ERSP_FMIN     = 8.0   # Hz (banda mu inferior)
ERSP_FMAX     = 30.0  # Hz (banda beta superior)

# Dimensiones de salida de la imagen ERSP (por canal, por ensayo)
# 22 bins de frecuencia × 128 pasos temporales
IMG_FREQ_BINS = 22
IMG_TIME_BINS = 128
IMG_SIZE      = (IMG_FREQ_BINS, IMG_TIME_BINS)

# ── Entrenamiento ───────────────────────────────────────────────────────────
BATCH_SIZE    = 32
MAX_EPOCHS    = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 20       # early stopping sobre pérdida de validación
VAL_SPLIT     = 0.15     # fracción del conjunto de entrenamiento para validación
RANDOM_SEED   = 42

# Dispositivo de cómputo (CPU dado que no hay GPU dedicada)
DEVICE        = "cpu"

# ── Modelos disponibles ─────────────────────────────────────────────────────
MODELS        = ["eegnet", "shallowconvnet", "spectnet"]

# ── Líneas de base clásicas ─────────────────────────────────────────────────
BASELINES     = ["lda", "svm_csp"]

# ── Evaluación ──────────────────────────────────────────────────────────────
# Protocolo BCI-IV-2b: sesiones 1-3 entrenamiento, sesiones 4-5 evaluación
TRAIN_SESSIONS = [1, 2, 3]
TEST_SESSIONS  = [4, 5]

# Métricas a reportar
METRICS       = ["accuracy", "precision", "recall", "f1", "kappa"]
