"""
evaluate.py
-----------
Evaluación comparativa de los modelos entrenados sobre el BCI-IV-2b.
Genera métricas completas, matriz de confusión y tabla resumen.

Uso:
    python src/evaluate.py                    # evaluar todos los modelos guardados
    python src/evaluate.py --model spectnet   # evaluar solo SpectNet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RESULTS_DIR, FIGURES_DIR, METRICS_DIR, SUBJECTS,
    N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS, DEVICE, CLASS_NAMES
)
from dataset import ERSPDataset
from models import get_model


@torch.no_grad()
def predict_all(model, dataset, device) -> tuple:
    """Obtiene predicciones y etiquetas reales para todo el dataset."""
    model.eval()
    all_preds, all_labels = [], []
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    for X, y in loader:
        X = X.to(device)
        preds = model(X).argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula el conjunto completo de métricas de clasificación."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa":     cohen_kappa_score(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, subj_tag: str):
    """Genera y guarda la matriz de confusión normalizada."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=list(CLASS_NAMES.values()),
        yticklabels=list(CLASS_NAMES.values()),
        ax=ax, cbar_kws={"label": "Proporción"}
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(f"Matriz de confusión — {model_name.upper()}\n"
                 f"(normalizada por fila, sujetos: {subj_tag})")
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"confusion_{model_name}_{subj_tag}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Matriz de confusión guardada: {fig_path.name}")


def evaluate_model(model_name: str, subjects: list = None,
                   device_str: str = DEVICE) -> dict:
    """
    Carga un modelo guardado y lo evalúa sobre el conjunto de prueba.
    """
    if subjects is None:
        subjects = SUBJECTS

    subj_tag = "all" if subjects == SUBJECTS else "-".join(map(str, subjects))
    device   = torch.device(device_str)

    model_path = RESULTS_DIR / f"{model_name}_{subj_tag}.pth"
    if not model_path.exists():
        print(f"  Modelo no encontrado: {model_path.name}")
        print(f"  Ejecuta primero: python src/train.py --model {model_name}")
        return {}

    # Cargar checkpoint
    ckpt = torch.load(str(model_path), map_location=device)

    # Instanciar modelo
    model = get_model(
        model_name,
        n_channels=N_CHANNELS,
        n_freq=IMG_FREQ_BINS,
        n_time=IMG_TIME_BINS,
        n_classes=2
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    # Dataset de prueba (sesiones 4-5)
    test_ds = ERSPDataset(subjects=subjects, suffix="E")

    y_pred, y_true = predict_all(model, test_ds, device)
    metrics = compute_metrics(y_true, y_pred)

    print(f"\n  ── {model_name.upper()} (sujetos: {subj_tag}) ──")
    print(f"  Exactitud:  {metrics['accuracy']:.1%}")
    print(f"  Precisión:  {metrics['precision']:.1%}")
    print(f"  Recall:     {metrics['recall']:.1%}")
    print(f"  F1-score:   {metrics['f1']:.1%}")
    print(f"  Kappa:      {metrics['kappa']:.3f}")

    plot_confusion_matrix(y_true, y_pred, model_name, subj_tag)

    return {"model": model_name, "subjects": subj_tag, **metrics,
            "n_params": ckpt.get("n_params", 0)}


def compare_all_models(subjects: list = None, device_str: str = DEVICE):
    """
    Evalúa todos los modelos guardados y genera una tabla comparativa.
    """
    if subjects is None:
        subjects = SUBJECTS

    model_names = ["spectnet", "eegnet", "shallowconvnet"]
    results = []

    print("\n══════════════════════════════════════════════")
    print("  Evaluación comparativa de modelos")
    print("══════════════════════════════════════════════")

    for name in model_names:
        r = evaluate_model(name, subjects, device_str)
        if r:
            results.append(r)

    if not results:
        print("\n  No hay modelos entrenados para comparar.")
        return

    # Tabla resumen
    df = pd.DataFrame(results)
    df = df.rename(columns={
        "model": "Modelo", "accuracy": "Exactitud",
        "precision": "Precisión", "recall": "Recall",
        "f1": "F1-score", "kappa": "Kappa", "n_params": "Parámetros"
    })

    # Formatear como porcentaje
    for col in ["Exactitud", "Precisión", "Recall", "F1-score"]:
        if col in df.columns:
            df[col] = df[col].map("{:.1%}".format)
    if "Kappa" in df.columns:
        df["Kappa"] = df["Kappa"].map("{:.3f}".format)
    if "Parámetros" in df.columns:
        df["Parámetros"] = df["Parámetros"].map("{:,}".format)

    print("\n\n  ══ Tabla resumen comparativa ══")
    print(df.to_string(index=False))

    # Guardar CSV
    subj_tag = "all" if subjects == SUBJECTS else "-".join(map(str, subjects))
    csv_path = METRICS_DIR / f"resultados_comparativos_{subj_tag}.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\n  Tabla guardada: {csv_path.name}")

    # Gráfico de barras comparativo
    df_raw = pd.DataFrame(results)
    _plot_comparison(df_raw, subj_tag)


def _plot_comparison(df: pd.DataFrame, subj_tag: str):
    """Gráfico de barras con exactitud y F1 por modelo."""
    metrics = ["accuracy", "f1", "kappa"]
    labels  = ["Exactitud", "F1-score", "Kappa"]
    colors  = ["#2C7BB6", "#D7191C", "#1A9641"]
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        if metric in df.columns:
            bars = ax.bar(x + i * width, df[metric], width,
                          label=label, color=color, alpha=0.85)
            ax.bar_label(bars, fmt="{:.1%}" if metric != "kappa" else "{:.3f}",
                         padding=2, fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df["model"].str.upper(), fontsize=10)
    ax.set_ylabel("Métrica")
    ax.set_ylim([0, 1.1])
    ax.set_title(
        f"Comparación de modelos CNN — BCI-IV-2b (sujetos: {subj_tag})\n"
        f"Sesiones 4-5 (evaluación)",
        fontsize=11
    )
    ax.legend()
    ax.axhline(0.5, color="gray", linewidth=0.6, linestyle="--",
               label="Azar (50%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    fig_path = FIGURES_DIR / f"comparacion_modelos_{subj_tag}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfico comparativo guardado: {fig_path.name}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluación de modelos CNN — BCI-IV-2b"
    )
    parser.add_argument("--model", type=str, default=None,
                        choices=["spectnet", "eegnet", "shallowconvnet"],
                        help="Modelo a evaluar (None = todos)")
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    if args.model:
        evaluate_model(args.model, args.subjects, args.device)
    else:
        compare_all_models(args.subjects, args.device)


if __name__ == "__main__":
    main()
