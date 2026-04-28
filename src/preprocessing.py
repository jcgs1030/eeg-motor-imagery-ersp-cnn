"""
preprocessing.py
----------------
Carga, filtrado, eliminación de artefactos y segmentación
de las señales EEG del dataset BCI Competition IV - Dataset 2b.

Uso desde terminal:
    python src/preprocessing.py --verify            # solo verificar archivos
    python src/preprocessing.py --subject 1         # procesar sujeto 1
    python src/preprocessing.py --subject all       # procesar todos los sujetos
    python src/preprocessing.py --subject 1 --plot  # procesar y graficar
"""

import argparse
import sys
from pathlib import Path

import mne
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend sin pantalla (compatible con servidor)
import matplotlib.pyplot as plt

# Añadir src/ al path para imports relativos
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_RAW, DATA_PROC, FIGURES_DIR,
    SUBJECTS, TRAIN_SUFFIX, EVAL_SUFFIX,
    CHANNELS, SFREQ,
    EVENT_LEFT, EVENT_RIGHT, EVENT_LABELS, CLASS_NAMES,
    FILT_LOW, FILT_HIGH, FILT_METHOD,
    EPOCH_TMIN, EPOCH_TMAX, BASELINE, REJECT_THRESH,
    ICA_N_COMPS, ICA_METHOD, ICA_SEED
)

# Silenciar logs de MNE (mantener solo warnings y errores)
mne.set_log_level("WARNING")


# ── Funciones de utilidad ───────────────────────────────────────────────────

def get_gdf_path(subject: int, suffix: str) -> Path:
    """Devuelve la ruta al archivo GDF de un sujeto."""
    return DATA_RAW / f"B{subject:02d}{suffix}.gdf"


def verify_dataset() -> bool:
    """
    Verifica que todos los archivos GDF del BCI-IV-2b están presentes.
    Retorna True si el dataset está completo.
    """
    print("\n── Verificando archivos del dataset BCI-IV-2b ──")
    missing = []
    for subj in SUBJECTS:
        for suffix in [TRAIN_SUFFIX, EVAL_SUFFIX]:
            path = get_gdf_path(subj, suffix)
            status = "OK" if path.exists() else "FALTA"
            symbol = "✓" if path.exists() else "✗"
            print(f"  {symbol} B{subj:02d}{suffix}.gdf  ({status})")
            if not path.exists():
                missing.append(path.name)

    if missing:
        print(f"\n  ADVERTENCIA: faltan {len(missing)} archivo(s).")
        print(f"  Descarga el dataset desde: https://www.bbci.de/competition/iv/download/")
        print(f"  y coloca los archivos en: {DATA_RAW}/")
        return False
    else:
        print(f"\n  Dataset completo: 18 archivos GDF encontrados en {DATA_RAW}/")
        return True


def load_raw(subject: int, suffix: str) -> mne.io.Raw:
    """
    Carga el archivo GDF de un sujeto y configura los canales EEG.

    Parámetros
    ----------
    subject : int
        Número de sujeto (1–9).
    suffix  : str
        'T' para entrenamiento, 'E' para evaluación.

    Retorna
    -------
    raw : mne.io.Raw
        Objeto Raw con datos cargados y canales configurados.
    """
    path = get_gdf_path(subject, suffix)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path.name}. "
            f"Verifica que el archivo esté en {DATA_RAW}/"
        )

    # Cargar con MNE (incluye canales EEG y EOG)
    raw = mne.io.read_raw_gdf(str(path), preload=True, verbose=False)

    # Seleccionar solo los canales de interés (C3, Cz, C4)
    # El BCI-IV-2b nombra los canales EEG como "EEG:C3", "EEG:Cz", "EEG:C4"
    # y los canales EOG como "EOG:ch01", "EOG:ch02", "EOG:ch03"
    available = raw.ch_names
    eeg_picks = [ch for ch in available if any(c in ch for c in CHANNELS)]

    if len(eeg_picks) == 0:
        # Fallback: tomar los primeros 3 canales si los nombres varían
        eeg_picks = available[:3]
        print(f"  Advertencia: canales no encontrados por nombre. "
              f"Usando: {eeg_picks}")

    # Mantener solo los canales EEG seleccionados
    raw.pick_channels(eeg_picks)

    # Renombrar canales al formato estándar si es necesario
    rename_map = {}
    for ch in raw.ch_names:
        for std in CHANNELS:
            if std in ch and ch != std:
                rename_map[ch] = std
    if rename_map:
        raw.rename_channels(rename_map)

    # Marcar tipo de canal como EEG
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

    return raw


def apply_filter(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Aplica filtro pasa banda FIR (8–30 Hz) para conservar
    las bandas mu y beta asociadas a la imaginación motora.
    """
    raw.filter(
        l_freq=FILT_LOW,
        h_freq=FILT_HIGH,
        method=FILT_METHOD,
        fir_window="hamming",
        verbose=False
    )
    return raw


def apply_ica(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Aplica ICA para eliminar artefactos oculares y musculares.
    Con 3 canales, ICA tiene capacidad limitada pero se aplica
    como paso de higiene de señal.
    """
    ica = mne.preprocessing.ICA(
        n_components=ICA_N_COMPS,
        method=ICA_METHOD,
        random_state=ICA_SEED,
        max_iter=200,
        verbose=False
    )
    ica.fit(raw, verbose=False)
    # Con 3 canales, no se excluye ningún componente automáticamente
    # (requeriría canales EOG separados para correlación)
    ica.apply(raw, verbose=False)
    return raw


def extract_epochs(raw: mne.io.Raw) -> mne.Epochs:
    """
    Segmenta la señal en épocas alineadas con los eventos
    de imaginación motora del BCI-IV-2b.

    Eventos:
        769 → mano izquierda (clase 0)
        770 → mano derecha   (clase 1)
    """
    # Extraer eventos del canal de estímulo
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Filtrar solo los eventos de interés (769 y 770)
    target_ids = {
        "left":  event_id.get(str(EVENT_LEFT),  event_id.get("769",  None)),
        "right": event_id.get(str(EVENT_RIGHT), event_id.get("770", None)),
    }
    # Eliminar IDs no encontrados
    target_ids = {k: v for k, v in target_ids.items() if v is not None}

    if not target_ids:
        # Intentar con claves numéricas directamente
        target_ids = {}
        for k, v in event_id.items():
            if "769" in str(k) or "left" in str(k).lower():
                target_ids["left"] = v
            elif "770" in str(k) or "right" in str(k).lower():
                target_ids["right"] = v

    if not target_ids:
        print(f"  Eventos disponibles: {event_id}")
        raise ValueError(
            "No se encontraron eventos de imaginación motora (769/770). "
            "Revisa los IDs de evento del archivo GDF."
        )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=target_ids,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=BASELINE,
        reject={"eeg": REJECT_THRESH},
        preload=True,
        verbose=False
    )

    return epochs


def process_subject(subject: int, suffix: str,
                    apply_ica_flag: bool = False,
                    save: bool = True) -> mne.Epochs:
    """
    Pipeline completo para un sujeto y una partición (T/E).

    Parámetros
    ----------
    subject       : número de sujeto (1–9)
    suffix        : 'T' (entrenamiento) o 'E' (evaluación)
    apply_ica_flag: si True, aplica ICA (puede ser lento en CPU)
    save          : si True, guarda las épocas en data/processed/

    Retorna
    -------
    epochs : mne.Epochs con las épocas limpias y etiquetadas
    """
    tag = f"S{subject:02d}{suffix}"
    print(f"\n  Procesando {tag}...")

    # 1. Cargar
    raw = load_raw(subject, suffix)
    print(f"    Cargado: {len(raw.ch_names)} canales, "
          f"{raw.n_times / raw.info['sfreq']:.1f} s, "
          f"{raw.info['sfreq']:.0f} Hz")

    # 2. Filtrar
    raw = apply_filter(raw)
    print(f"    Filtrado: {FILT_LOW}–{FILT_HIGH} Hz (FIR Hamming)")

    # 3. ICA (opcional, puede ser lento)
    if apply_ica_flag:
        raw = apply_ica(raw)
        print(f"    ICA aplicado ({ICA_N_COMPS} componentes)")

    # 4. Epoching
    epochs = extract_epochs(raw)
    n_total = len(epochs)
    n_left  = len(epochs["left"])  if "left"  in epochs.event_id else 0
    n_right = len(epochs["right"]) if "right" in epochs.event_id else 0
    print(f"    Épocas: {n_total} totales "
          f"(izquierda: {n_left}, derecha: {n_right})")

    # 5. Guardar
    if save:
        out_path = DATA_PROC / f"{tag}-epo.fif"
        epochs.save(str(out_path), overwrite=True, verbose=False)
        print(f"    Guardado en: {out_path.name}")

    return epochs


# ── Funciones de visualización ──────────────────────────────────────────────

def plot_subject_overview(subject: int, suffix: str = TRAIN_SUFFIX,
                          save_fig: bool = True):
    """
    Genera una figura de 4 paneles para un sujeto:
      (a) Señal cruda (primeros 10 s)
      (b) Señal filtrada (primeros 10 s)
      (c) Promedio de épocas por clase (canal C3)
      (d) Espectro de potencia por clase (canal C3)
    """
    tag = f"S{subject:02d}{suffix}"
    print(f"\n  Generando visualización para {tag}...")

    # Cargar raw y filtrado
    raw = load_raw(subject, suffix)
    raw_filt = raw.copy()
    apply_filter(raw_filt)

    # Épocas
    epochs = extract_epochs(raw_filt.copy())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"BCI-IV-2b — Sujeto {subject:02d} ({'Entrenamiento' if suffix == 'T' else 'Evaluación'})\n"
        f"Canal C3 | Filtro {FILT_LOW}–{FILT_HIGH} Hz | "
        f"Épocas: {len(epochs)} ({EPOCH_TMIN} a {EPOCH_TMAX} s)",
        fontsize=12, fontweight="bold"
    )

    times_raw = raw.times[:int(10 * SFREQ)]   # primeros 10 s
    ch_idx = 0  # primer canal (C3)

    # ── Panel (a): Señal cruda ──
    ax = axes[0, 0]
    data_raw = raw.get_data(picks=[ch_idx])[0, :int(10 * SFREQ)] * 1e6  # en µV
    ax.plot(times_raw, data_raw, color="#2C7BB6", linewidth=0.6)
    ax.set_title("(a) Señal EEG cruda — C3 (primeros 10 s)", fontsize=10)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud (µV)")
    ax.set_xlim([0, 10])
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel (b): Señal filtrada ──
    ax = axes[0, 1]
    data_filt = raw_filt.get_data(picks=[ch_idx])[0, :int(10 * SFREQ)] * 1e6
    ax.plot(times_raw, data_filt, color="#D7191C", linewidth=0.6)
    ax.set_title(f"(b) Señal filtrada {FILT_LOW}–{FILT_HIGH} Hz — C3", fontsize=10)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud (µV)")
    ax.set_xlim([0, 10])
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel (c): Promedio de épocas por clase ──
    ax = axes[1, 0]
    epoch_times = epochs.times
    colors_cls = {"left": "#2C7BB6", "right": "#D7191C"}
    for cls_name, color in colors_cls.items():
        if cls_name in epochs.event_id:
            ep_data = epochs[cls_name].get_data(picks=[ch_idx])[:, 0, :] * 1e6
            mean_ep = ep_data.mean(axis=0)
            std_ep  = ep_data.std(axis=0)
            ax.plot(epoch_times, mean_ep, color=color,
                    label=CLASS_NAMES.get(0 if cls_name == "left" else 1, cls_name),
                    linewidth=1.2)
            ax.fill_between(epoch_times, mean_ep - std_ep, mean_ep + std_ep,
                            color=color, alpha=0.15)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", label="Onset cue")
    ax.axhline(0, color="gray",  linewidth=0.4, linestyle=":")
    ax.set_title("(c) Promedio de épocas por clase — C3", fontsize=10)
    ax.set_xlabel("Tiempo relativo al onset (s)")
    ax.set_ylabel("Amplitud media (µV) ± DE")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel (d): Espectro de potencia por clase ──
    ax = axes[1, 1]
    for cls_name, color in colors_cls.items():
        if cls_name in epochs.event_id:
            ep_data = epochs[cls_name].get_data(picks=[ch_idx])[:, 0, :]
            # PSD con Welch sobre la ventana de imaginación (0–4 s)
            tmin_idx = int((0.0 - EPOCH_TMIN) * SFREQ)
            tmax_idx = int((4.0 - EPOCH_TMIN) * SFREQ)
            ep_mi = ep_data[:, tmin_idx:tmax_idx]
            from scipy.signal import welch
            freqs_w, psd = welch(ep_mi, fs=SFREQ, nperseg=256,
                                 noverlap=128, axis=-1)
            psd_db = 10 * np.log10(psd.mean(axis=0) + 1e-12)
            mask = (freqs_w >= 4) & (freqs_w <= 40)
            ax.plot(freqs_w[mask], psd_db[mask], color=color,
                    label=CLASS_NAMES.get(0 if cls_name == "left" else 1, cls_name),
                    linewidth=1.2)

    ax.axvspan(8, 13, alpha=0.1, color="green", label="Banda mu")
    ax.axvspan(14, 30, alpha=0.1, color="orange", label="Banda beta")
    ax.set_title("(d) Espectro de potencia por clase — C3 (0–4 s)", fontsize=10)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_fig:
        fig_path = FIGURES_DIR / f"overview_{tag}.png"
        fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        print(f"    Figura guardada en: {fig_path.name}")
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_all_subjects_summary(suffix: str = TRAIN_SUFFIX):
    """
    Genera una figura resumen con el número de épocas válidas
    por sujeto y clase, para detectar sujetos problemáticos.
    """
    print("\n── Resumen de épocas por sujeto ──")
    n_left_list, n_right_list = [], []
    subj_labels = []

    for subj in SUBJECTS:
        path = get_gdf_path(subj, suffix)
        if not path.exists():
            print(f"  S{subj:02d}: archivo no encontrado, omitido.")
            continue
        try:
            raw = load_raw(subj, suffix)
            apply_filter(raw)
            epochs = extract_epochs(raw)
            nl = len(epochs["left"])  if "left"  in epochs.event_id else 0
            nr = len(epochs["right"]) if "right" in epochs.event_id else 0
            n_left_list.append(nl)
            n_right_list.append(nr)
            subj_labels.append(f"S{subj:02d}")
            print(f"  S{subj:02d}: izquierda={nl:3d}, derecha={nr:3d}, total={nl+nr:3d}")
        except Exception as e:
            print(f"  S{subj:02d}: ERROR — {e}")

    if not subj_labels:
        print("  No hay datos para graficar.")
        return

    x = np.arange(len(subj_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_l = ax.bar(x - width/2, n_left_list,  width, label="Izquierda", color="#2C7BB6", alpha=0.85)
    bars_r = ax.bar(x + width/2, n_right_list, width, label="Derecha",   color="#D7191C", alpha=0.85)
    ax.bar_label(bars_l, padding=2, fontsize=8)
    ax.bar_label(bars_r, padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(subj_labels)
    ax.set_xlabel("Sujeto")
    ax.set_ylabel("Número de épocas válidas")
    ax.set_title(
        f"BCI-IV-2b — Épocas válidas por sujeto y clase\n"
        f"({'Entrenamiento (sesiones 1–3)' if suffix == 'T' else 'Evaluación (sesiones 4–5)'})",
        fontsize=11
    )
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    fig_path = FIGURES_DIR / f"epochs_summary_{suffix}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    print(f"\n  Figura resumen guardada en: {fig_path.name}")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocesamiento de señales EEG del BCI-IV-2b"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Solo verifica que los archivos GDF están presentes"
    )
    parser.add_argument(
        "--subject", type=str, default=None,
        help="Número de sujeto a procesar (1–9) o 'all' para todos"
    )
    parser.add_argument(
        "--suffix", type=str, default="T", choices=["T", "E", "both"],
        help="Partición a procesar: T (train), E (eval), both"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generar figuras de visualización"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Generar figura resumen de épocas por sujeto"
    )
    parser.add_argument(
        "--ica", action="store_true",
        help="Aplicar ICA para eliminación de artefactos (más lento)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n══════════════════════════════════════════════")
    print("  BCI-IV-2b — Módulo de Preprocesamiento")
    print("══════════════════════════════════════════════")

    # ── Verificación ──
    if args.verify or (args.subject is None and not args.summary):
        ok = verify_dataset()
        if args.verify:
            return

    # ── Resumen global ──
    if args.summary:
        suffixes = ["T", "E"] if args.suffix == "both" else [args.suffix]
        for s in suffixes:
            plot_all_subjects_summary(suffix=s)
        return

    # ── Procesar sujeto(s) ──
    if args.subject is not None:
        suffixes = ["T", "E"] if args.suffix == "both" else [args.suffix]

        if args.subject == "all":
            subjects = SUBJECTS
        else:
            try:
                subjects = [int(args.subject)]
            except ValueError:
                print(f"  ERROR: --subject debe ser un número 1–9 o 'all'")
                return

        print(f"\n── Procesando sujeto(s): {subjects} | Partición(es): {suffixes} ──")

        for subj in subjects:
            for suf in suffixes:
                path = get_gdf_path(subj, suf)
                if not path.exists():
                    print(f"\n  S{subj:02d}{suf}: archivo no encontrado, omitido.")
                    continue
                try:
                    process_subject(subj, suf, apply_ica_flag=args.ica)
                    if args.plot:
                        plot_subject_overview(subj, suf)
                except Exception as e:
                    print(f"\n  S{subj:02d}{suf}: ERROR — {e}")

    print("\n── Preprocesamiento completado ──\n")


if __name__ == "__main__":
    main()
