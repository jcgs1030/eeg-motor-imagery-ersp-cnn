"""
generate_final_comparison.py
-------------------------------
Visualises the four follow-up experiments requested after the ERSP fixes:
1. Full method comparison (extends results/riemannian/figures/comparison_all_methods.png
   with the pooled CNNs and the new FBCSP+SpectNet ensemble).
2. Per-subject paired comparison: FBCSP+LDA vs. SpectNet pooled vs. Ensemble.
3. ShallowConvNet learning-rate sweep.
4. Per-subject paired comparison: ShallowConvNet baseline vs. data augmentation.

Does not touch any production result file - saves to results/figures/.

Usage:
    python src/generate_final_comparison.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import SUBJECTS, FIGURES_DIR
from diagnostic_stats_test import per_subject_acc_pooled, load_per_subject_csv
from experiment_ensemble import compute_ensemble_per_subject

PALETTE = {
    "EEGNET": "#2C7BB6", "SHALLOWCONVNET": "#D7191C", "SPECTNET": "#1A9641",
    "CSP+LDA": "#E66100", "CSP+SVM": "#5D3A9B",
    "EA+CSP+LDA": "#0077BB", "EA+CSP+SVM": "#CC3311",
    "FBCSP+LDA": "#009988", "FBCSP+SVM": "#EE7733",
    "Riem-MDM": "#882255", "Riem-TS+LDA": "#44AA99",
    "ENSEMBLE": "#AA4499",
}


def fig1_full_comparison():
    """Bar chart: every method, including pooled CNNs and the ensemble."""
    methods = {}
    for m in ["eegnet", "shallowconvnet", "spectnet"]:
        methods[f"{m.upper()} (pooled)"] = per_subject_acc_pooled(m)

    classical_specs = [
        ("CSP+LDA", "results/csp_lda/metrics/per_subject_results.csv", "Classifier"),
        ("CSP+SVM", "results/csp_lda/metrics/per_subject_results.csv", "Classifier"),
        ("EA+CSP+LDA", "results/ea_csp_lda/metrics/per_subject_results.csv", "Classifier"),
        ("EA+CSP+SVM", "results/ea_csp_lda/metrics/per_subject_results.csv", "Classifier"),
        ("FBCSP+LDA", "results/fbcsp/metrics/per_subject_results.csv", "Classifier"),
        ("FBCSP+SVM", "results/fbcsp/metrics/per_subject_results.csv", "Classifier"),
        ("Riem-MDM", "results/riemannian/metrics/per_subject_results.csv", "Classifier"),
        ("Riem-TS+LDA", "results/riemannian/metrics/per_subject_results.csv", "Classifier"),
    ]
    for name, path, col in classical_specs:
        methods[name] = load_per_subject_csv(path, model_filter=name, model_col=col)

    for m in ["eegnet", "shallowconvnet", "spectnet"]:
        methods[f"{m.upper()} (subj-specific)"] = load_per_subject_csv(
            "results/subject_specific/metrics/per_subject_results.csv", model_filter=m)

    ens_rows = compute_ensemble_per_subject(weight=0.6)
    methods["ENSEMBLE (FBCSP+SpectNet)"] = {r["subject"]: r["acc_ens"] for r in ens_rows}

    stats = []
    for name, accs in methods.items():
        vals = np.array(list(accs.values()))
        stats.append({"method": name, "mean": vals.mean(), "std": vals.std()})
    df = pd.DataFrame(stats).sort_values("mean", ascending=False).reset_index(drop=True)

    def color_for(name):
        key = name.replace(" (pooled)", "").replace(" (subj-specific)", "").upper()
        if "ENSEMBLE" in name.upper():
            return PALETTE["ENSEMBLE"]
        return PALETTE.get(key, "gray")

    colors = [color_for(m) for m in df["method"]]
    hatches = ["//" if "pooled" in m else None for m in df["method"]]

    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(df))
    bars = ax.bar(x, df["mean"] * 100, 0.65, color=colors, alpha=0.88,
                  yerr=df["std"] * 100, capsize=3, error_kw={"linewidth": 1.0})
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
    for i, name in enumerate(df["method"]):
        if "ENSEMBLE" in name.upper():
            bars[i].set_edgecolor("black")
            bars[i].set_linewidth(2.2)

    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=7.5)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", label="Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], fontsize=8, rotation=28, ha="right")
    ax.set_ylabel("Accuracy (%) — mean ± std across 9 subjects")
    ax.set_ylim([0, 100])
    ax.set_title(
        "Comparación final de métodos — BCI-IV-2b (sesiones 4-5)\n"
        "Rayado = modelo pooled (un solo modelo, 9 sujetos) · Borde negro = ensamble (nuevo mejor resultado)",
        fontsize=11
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = FIGURES_DIR / "final_comparison_all_methods.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig2_ensemble_per_subject():
    """Slope plot: FBCSP, SpectNet pooled, Ensemble, one line per subject."""
    rows = compute_ensemble_per_subject(weight=0.6)
    rows = sorted(rows, key=lambda r: r["subject"])

    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [0, 1, 2]
    labels = ["FBCSP+LDA", "SpectNet\n(pooled)", "Ensemble\n(w=0.6)"]
    for r in rows:
        ys = [r["acc_fbcsp"] * 100, r["acc_cnn"] * 100, r["acc_ens"] * 100]
        color = "#44AA99" if ys[2] >= max(ys[0], ys[1]) else "#D7191C"
        ax.plot(xs, ys, marker="o", color=color, alpha=0.75, linewidth=1.6)
        ax.annotate(f"S{r['subject']:02d}", (xs[-1] + 0.05, ys[-1]), fontsize=8, va="center")

    means = [np.mean([r["acc_fbcsp"] for r in rows]) * 100,
             np.mean([r["acc_cnn"] for r in rows]) * 100,
             np.mean([r["acc_ens"] for r in rows]) * 100]
    ax.plot(xs, means, marker="s", color="black", linewidth=3, markersize=10,
            label="Media (9 sujetos)", zorder=5)
    for x, y in zip(xs, means):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("El ensamble iguala o supera al mejor de los dos métodos\n"
                "en la mayoría de los sujetos (verde = ensamble gana o empata)",
                fontsize=11)
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = FIGURES_DIR / "ensemble_per_subject.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig3_lr_sweep():
    """Known results from the lr sweep run on ShallowConvNet (see EXPERIMENTS.md)."""
    lrs = [1e-4, 3e-4, 1e-3]
    means = [53.3, 55.8, 54.5]
    stds = [3.9, 5.3, 6.5]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(range(len(lrs)), means, yerr=stds, marker="o", markersize=9,
               color="#D7191C", linewidth=2, capsize=5)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.annotate(f"{m:.1f}%", (i, m), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, fontweight="bold")
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", label="Azar (50%)")
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in lrs])
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Accuracy media (%), 9 sujetos")
    ax.set_ylim([40, 70])
    ax.set_title("ShallowConvNet — barrido de learning rate\n"
                "Sin tendencia clara: el LR no es la causa del colapso", fontsize=11)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = FIGURES_DIR / "shallowconvnet_lr_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig4_augmentation():
    """Known per-subject results from the augmentation experiment (see EXPERIMENTS.md)."""
    subjects = SUBJECTS
    baseline = [50.3, 47.9, 47.2, 67.8, 60.9, 51.1, 55.6, 59.1, 50.6]
    augmented = [53.4, 50.7, 45.3, 65.3, 60.9, 56.7, 55.6, 60.0, 58.4]

    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [0, 1]
    for subj, b, a in zip(subjects, baseline, augmented):
        color = "#44AA99" if a >= b else "#D7191C"
        ax.plot(xs, [b, a], marker="o", color=color, alpha=0.75, linewidth=1.6)
        ax.annotate(f"S{subj:02d}", (xs[-1] + 0.03, a), fontsize=8, va="center")

    mb, ma = np.mean(baseline), np.mean(augmented)
    ax.plot(xs, [mb, ma], marker="s", color="black", linewidth=3, markersize=10,
            label="Media (9 sujetos)", zorder=5)
    for x, y in zip(xs, [mb, ma]):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(["Sin aumentación", "Con aumentación\n(ruido + time-shift)"], fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("ShallowConvNet — efecto de la aumentación de datos\n"
                "+1.8pp, p=0.156 (no significativo, n=9) — prometedor, no concluyente",
                fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = FIGURES_DIR / "augmentation_per_subject.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating final comparison figures...\n")
    fig1_full_comparison()
    fig2_ensemble_per_subject()
    fig3_lr_sweep()
    fig4_augmentation()
    print("\nDone.")
