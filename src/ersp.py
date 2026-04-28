"""
ersp.py
-------
Generación de espectrogramas ERSP (Event-Related Spectral Perturbation)
a partir de las épocas preprocesadas del BCI-IV-2b.

Uso:
    python src/ersp.py                        # genera todos los sujetos
    python src/ersp.py --subject 1            # solo sujeto 1
    python src/ersp.py --subject 1 --plot     # generar y visualizar
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from scipy.signal import stft

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROC, FIGURES_DIR, SUBJECTS, CHANNELS, N_CHANNELS, SFREQ,
    EPOCH_TMIN, EPOCH_TMAX,
    STFT_WIN_LEN, STFT_HOP,
    ERSP_FMIN, ERSP_FMAX,
    IMG_FREQ_BINS, IMG_TIME_BINS, IMG_SIZE,
    TRAIN_SUFFIX, EVAL_SUFFIX
)

mne.set_log_level("WARNING")


# ── Core ERSP ───────────────────────────────────────────────────────────────

def compute_ersp_image(epoch_data: np.ndarray,
                       baseline_data: np.ndarray,
                       sfreq: float = SFREQ) -> np.ndarray:
    """
    Calcula la imagen ERSP de un ensayo individual.

    ERSP(f, t) = 10 * log10( P(f, t) / P_baseline(f) )

    Parámetros
    ----------
    epoch_data    : array (n_samples,) — señal del ensayo completo (línea base + MI)
    baseline_data : array (n_samples,) — solo la ventana de línea base pre-estímulo
    sfreq         : frecuencia de muestreo

    Retorna
    -------
    ersp_img : array (IMG_FREQ_BINS, IMG_TIME_BINS) — imagen ERSP normalizada [0, 1]
    """
    # ── STFT del ensayo completo ──
    freqs, times_stft, Zxx = stft(
        epoch_data,
        fs=sfreq,
        window="hann",
        nperseg=STFT_WIN_LEN,
        noverlap=STFT_WIN_LEN - STFT_HOP,
        padded=True
    )
    power = np.abs(Zxx) ** 2  # (n_freqs_stft, n_times_stft)

    # ── STFT de la línea base ──
    _, _, Zxx_bl = stft(
        baseline_data,
        fs=sfreq,
        window="hann",
        nperseg=STFT_WIN_LEN,
        noverlap=STFT_WIN_LEN - STFT_HOP,
        padded=True
    )
    power_bl = np.abs(Zxx_bl) ** 2  # (n_freqs_stft, n_times_bl)
    baseline_mean = power_bl.mean(axis=-1, keepdims=True) + 1e-12  # evitar div/0

    # ── ERSP divisiva (en dB) ──
    ersp = 10 * np.log10(power / baseline_mean + 1e-12)

    # ── Seleccionar rango de frecuencias de interés ──
    freq_mask = (freqs >= ERSP_FMIN) & (freqs <= ERSP_FMAX)
    ersp_roi = ersp[freq_mask, :]    # (n_freq_roi, n_times_stft)

    # ── Redimensionar a IMG_SIZE con interpolación ──
    ersp_resized = _resize_2d(ersp_roi, IMG_FREQ_BINS, IMG_TIME_BINS)

    # ── Normalizar al rango [0, 1] ──
    vmin, vmax = ersp_resized.min(), ersp_resized.max()
    if vmax - vmin > 1e-8:
        ersp_norm = (ersp_resized - vmin) / (vmax - vmin)
    else:
        ersp_norm = np.zeros_like(ersp_resized)

    return ersp_norm.astype(np.float32)


def _resize_2d(arr: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Redimensiona un array 2D a (n_rows, n_cols) mediante interpolación lineal."""
    from scipy.ndimage import zoom
    zoom_r = n_rows / arr.shape[0]
    zoom_c = n_cols / arr.shape[1]
    return zoom(arr, (zoom_r, zoom_c), order=1)  # interpolación bilineal


# ── Procesamiento por sujeto ─────────────────────────────────────────────────

def generate_ersp_for_subject(subject: int, suffix: str,
                               save: bool = True) -> dict:
    """
    Genera los espectrogramas ERSP para todos los ensayos de un sujeto.

    Retorna
    -------
    result : dict con keys:
        'X'       : array (n_trials, n_channels, IMG_FREQ_BINS, IMG_TIME_BINS)
        'y'       : array (n_trials,) — etiquetas 0=izquierda, 1=derecha
        'subject' : número de sujeto
        'suffix'  : 'T' o 'E'
    """
    tag = f"S{subject:02d}{suffix}"
    epo_path = DATA_PROC / f"{tag}-epo.fif"

    if not epo_path.exists():
        raise FileNotFoundError(
            f"No se encontró {epo_path.name}. "
            f"Ejecuta primero: python src/preprocessing.py --subject {subject}"
        )

    print(f"\n  Generando ERSP para {tag}...")
    epochs = mne.read_epochs(str(epo_path), verbose=False)

    # Índices de tiempo para línea base y ventana de imaginación
    bl_start = 0
    bl_end   = int(abs(EPOCH_TMIN) * SFREQ)  # hasta t=0
    # (la señal completa incluye línea base + periodo de imaginación)

    X_list, y_list = [], []

    for cls_name, cls_label in [("left", 0), ("right", 1)]:
        if cls_name not in epochs.event_id:
            continue
        ep_data = epochs[cls_name].get_data()  # (n_trials, n_channels, n_times)

        for trial_idx in range(ep_data.shape[0]):
            trial_channels = []
            valid = True

            for ch_idx in range(ep_data.shape[1]):
                signal = ep_data[trial_idx, ch_idx, :]
                baseline = signal[:bl_end]

                ersp_img = compute_ersp_image(signal, baseline, sfreq=SFREQ)

                if ersp_img.shape != IMG_SIZE:
                    valid = False
                    break
                trial_channels.append(ersp_img)

            if valid and len(trial_channels) == ep_data.shape[1]:
                X_list.append(np.stack(trial_channels, axis=0))  # (C, F, T)
                y_list.append(cls_label)

    if not X_list:
        raise ValueError(f"No se pudieron generar espectrogramas para {tag}")

    X = np.array(X_list, dtype=np.float32)  # (N, C, F, T)
    y = np.array(y_list,  dtype=np.int64)   # (N,)

    n_left  = (y == 0).sum()
    n_right = (y == 1).sum()
    print(f"    ERSP generados: {len(X)} ensayos "
          f"(izquierda: {n_left}, derecha: {n_right}) — forma: {X.shape}")

    result = {"X": X, "y": y, "subject": subject, "suffix": suffix}

    if save:
        out_path = DATA_PROC / f"{tag}-ersp.npz"
        np.savez_compressed(str(out_path), X=X, y=y,
                            subject=subject, suffix=suffix)
        print(f"    Guardado en: {out_path.name}")

    return result


# ── Visualización ────────────────────────────────────────────────────────────

def plot_ersp_examples(subject: int, suffix: str = TRAIN_SUFFIX,
                       n_examples: int = 3, save_fig: bool = True):
    """
    Visualiza ejemplos de imágenes ERSP para un sujeto,
    mostrando n_examples ensayos de cada clase.
    """
    tag = f"S{subject:02d}{suffix}"
    npz_path = DATA_PROC / f"{tag}-ersp.npz"

    if not npz_path.exists():
        print(f"  No se encontró {npz_path.name}. "
              f"Ejecuta primero: python src/ersp.py --subject {subject}")
        return

    data = np.load(str(npz_path))
    X, y = data["X"], data["y"]

    idx_left  = np.where(y == 0)[0][:n_examples]
    idx_right = np.where(y == 1)[0][:n_examples]

    n_rows = 2 * N_CHANNELS  # 2 clases × 3 canales
    n_cols = n_examples
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows))
    fig.suptitle(
        f"Espectrogramas ERSP — Sujeto {subject:02d} "
        f"({'Entrenamiento' if suffix == 'T' else 'Evaluación'})\n"
        f"Filas: Izquierda (C3/Cz/C4) | Derecha (C3/Cz/C4) — "
        f"Eje X: tiempo | Eje Y: frecuencia (8–30 Hz)",
        fontsize=11, fontweight="bold"
    )

    cls_data = [(idx_left, "Izquierda", "#2C7BB6"),
                (idx_right, "Derecha",   "#D7191C")]
    row = 0
    for indices, cls_name, color in cls_data:
        for ch_i, ch_name in enumerate(CHANNELS[:N_CHANNELS]):
            for col, trial_idx in enumerate(indices):
                ax = axes[row, col] if n_cols > 1 else axes[row]
                img = X[trial_idx, ch_i, :, :]  # (F, T)
                im = ax.imshow(
                    img, aspect="auto", origin="lower",
                    cmap="RdYlBu_r",
                    extent=[EPOCH_TMIN, EPOCH_TMAX, ERSP_FMIN, ERSP_FMAX]
                )
                if col == 0:
                    ax.set_ylabel(f"{cls_name}\n{ch_name}\nFrecuencia (Hz)",
                                  fontsize=8)
                if row == n_rows - 1:
                    ax.set_xlabel("Tiempo (s)", fontsize=8)
                ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
                ax.tick_params(labelsize=7)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
            row += 1

    plt.tight_layout()

    if save_fig:
        fig_path = FIGURES_DIR / f"ersp_examples_{tag}.png"
        fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        print(f"    Figura guardada en: {fig_path.name}")
        plt.close(fig)
    else:
        plt.show()


def plot_ersp_average(subject: int, suffix: str = TRAIN_SUFFIX,
                      save_fig: bool = True):
    """
    Visualiza el ERSP promedio por clase y canal para un sujeto.
    """
    tag = f"S{subject:02d}{suffix}"
    npz_path = DATA_PROC / f"{tag}-ersp.npz"

    if not npz_path.exists():
        print(f"  No se encontró {npz_path.name}.")
        return

    data = np.load(str(npz_path))
    X, y = data["X"], data["y"]

    fig, axes = plt.subplots(N_CHANNELS, 2,
                             figsize=(10, 4 * N_CHANNELS))
    fig.suptitle(
        f"ERSP promedio por clase — Sujeto {subject:02d}\n"
        f"(columnas: Izquierda | Derecha — filas: C3, Cz, C4)",
        fontsize=11, fontweight="bold"
    )

    for ch_i, ch_name in enumerate(CHANNELS[:N_CHANNELS]):
        for cls_i, (cls_label, cls_name) in enumerate([(0, "Izquierda"), (1, "Derecha")]):
            ax = axes[ch_i, cls_i]
            mask = y == cls_label
            if mask.sum() == 0:
                ax.set_visible(False)
                continue
            avg_ersp = X[mask, ch_i, :, :].mean(axis=0)
            im = ax.imshow(
                avg_ersp, aspect="auto", origin="lower",
                cmap="RdYlBu_r", vmin=0, vmax=1,
                extent=[EPOCH_TMIN, EPOCH_TMAX, ERSP_FMIN, ERSP_FMAX]
            )
            ax.axvline(0, color="white", linewidth=1.0, linestyle="--")
            ax.axhspan(8, 13, alpha=0.15, color="cyan")    # banda mu
            ax.axhspan(14, 30, alpha=0.10, color="yellow") # banda beta
            ax.set_title(f"{ch_name} — {cls_name} (n={mask.sum()})", fontsize=9)
            ax.set_ylabel("Frecuencia (Hz)", fontsize=8)
            ax.set_xlabel("Tiempo (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label="ERSP norm.").ax.tick_params(labelsize=6)

    plt.tight_layout()

    if save_fig:
        fig_path = FIGURES_DIR / f"ersp_average_{tag}.png"
        fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        print(f"    Figura promedio guardada en: {fig_path.name}")
        plt.close(fig)
    else:
        plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generación de espectrogramas ERSP — BCI-IV-2b"
    )
    parser.add_argument("--subject", type=str, default="all",
                        help="Número de sujeto (1–9) o 'all'")
    parser.add_argument("--suffix", type=str, default="both",
                        choices=["T", "E", "both"])
    parser.add_argument("--plot", action="store_true",
                        help="Generar figuras de ejemplos y promedio")
    args = parser.parse_args()

    print("\n══════════════════════════════════════════════")
    print("  BCI-IV-2b — Módulo de Espectrogramas ERSP")
    print("══════════════════════════════════════════════")

    subjects = SUBJECTS if args.subject == "all" else [int(args.subject)]
    suffixes = ["T", "E"] if args.suffix == "both" else [args.suffix]

    for subj in subjects:
        for suf in suffixes:
            epo_path = DATA_PROC / f"S{subj:02d}{suf}-epo.fif"
            if not epo_path.exists():
                print(f"\n  S{subj:02d}{suf}: épocas no encontradas — "
                      f"ejecuta preprocessing.py primero.")
                continue
            try:
                generate_ersp_for_subject(subj, suf)
                if args.plot:
                    plot_ersp_examples(subj, suf)
                    plot_ersp_average(subj, suf)
            except Exception as e:
                print(f"\n  S{subj:02d}{suf}: ERROR — {e}")

    print("\n── Generación de ERSP completada ──\n")


if __name__ == "__main__":
    main()
