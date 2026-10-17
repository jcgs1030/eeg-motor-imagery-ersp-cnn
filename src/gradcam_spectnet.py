"""
gradcam_spectnet.py
-----------------------
Grad-CAM for SpectNet pooled, per advisor request: does the model actually
look at the mu/beta contralateral region (the same region the ERD/ERS
characterization in erd_characterization.py found to be statistically
significant), or is it picking up on something else entirely?

SpectNet mixes the 3 EEG channels (C3, Cz, C4) together from the very first
convolution (conv1 has in_channels=3), so a single Grad-CAM map taken at a
deeper layer cannot be split back out per input channel - it only answers
"where in time-frequency does the model look", not "which channel does it
look at". To answer the channel question too (central to contralaterality),
this script also computes a simple input-gradient saliency map directly on
the 3-channel input, which does preserve the channel axis.

Two complementary maps, per predicted class (only on correctly classified
trials, pooled across all 9 subjects' evaluation-session trials):

  1. Grad-CAM (Selvaraju et al., 2017) at the last conv layer (conv2, taken
     post-ReLU / pre-pool2, spatial resolution 21x29) - "where" in
     time-frequency the decision comes from, upsampled to the full 22x128
     image for overlay.
  2. Input-gradient saliency, |d(class score)/d(input)|, kept per channel
     (3, 22, 128) - "which channel" (C3/Cz/C4) and where, at full
     resolution, no upsampling needed.

Usage:
    python src/gradcam_spectnet.py
    python src/gradcam_spectnet.py --subjects 1 2 3   # faster subset
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import mne
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROC, FIGURES_DIR, SUBJECTS, CHANNELS, N_CHANNELS, SFREQ,
    IMG_FREQ_BINS, IMG_TIME_BINS, ANALYSIS_TMIN, ANALYSIS_TMAX,
    ERSP_FMIN, ERSP_FMAX,
)
from ersp import compute_ersp_image
from models import get_model
from train_fbcsp import EVENT_LABEL_MAP

mne.set_log_level("WARNING")

CLASS_NAMES = {0: "Left", 1: "Right"}
OUT_DIR = FIGURES_DIR / "gradcam"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Grad-CAM machinery ───────────────────────────────────────────────────────

class SpectNetGradCAM:
    """Hooks SpectNet's pool2 input (= relu(conv2(x)), the last conv
    activation before spatial pooling) to compute Grad-CAM, and exposes a
    plain input-gradient saliency method for per-channel attribution."""

    def __init__(self, model):
        self.model = model
        self._activation = None
        self._handle = model.pool2.register_forward_pre_hook(self._capture)

    def _capture(self, module, inputs):
        a = inputs[0]
        if a.requires_grad:
            a.retain_grad()
        self._activation = a

    def close(self):
        self._handle.remove()

    def gradcam(self, x: torch.Tensor, target_class: int) -> np.ndarray:
        """x: (1, C, F, T). Returns a (F, T) map in [0, 1], upsampled to the
        input's own spatial size."""
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[0, target_class]
        score.backward()

        grads = self._activation.grad          # (1, 16, Fc, Tc)
        acts  = self._activation.detach()       # (1, 16, Fc, Tc)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (1,1,Fc,Tc)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].numpy()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def channel_saliency(self, x: torch.Tensor, target_class: int) -> np.ndarray:
        """x: (1, C, F, T). Returns RAW |d(score)/d(input)|, shape (C, F, T) -
        NOT normalised per channel. Per-channel normalisation would rescale
        every channel to its own max and make it impossible to see whether
        one channel is actually more important than another - exactly the
        comparison this method exists to make. Normalise once, globally
        across channels, only at plot time."""
        x = x.clone().requires_grad_(True)
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[0, target_class]
        score.backward()
        return x.grad[0].abs().numpy()  # (C, F, T), raw scale


# ── Data ──────────────────────────────────────────────────────────────────

def load_subject_images(subject: int):
    """Full evaluation-session (E) trials for one subject, ERSP-encoded."""
    path = DATA_PROC / f"S{subject:02d}E-epo.fif"
    epochs = mne.read_epochs(str(path), verbose=False)
    X = epochs.get_data()
    code_to_label = {v: EVENT_LABEL_MAP[k] for k, v in epochs.event_id.items()}
    y = np.array([code_to_label[ev] for ev in epochs.events[:, 2]])

    imgs = np.zeros((len(X), N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS), dtype=np.float32)
    for i in range(len(X)):
        for ch in range(N_CHANNELS):
            imgs[i, ch] = compute_ersp_image(X[i, ch], sfreq=SFREQ)
    return imgs, y


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_gradcam(cam_mean: np.ndarray, cls_label: str, n_trials: int, out_path: Path):
    times = np.linspace(ANALYSIS_TMIN, ANALYSIS_TMAX, IMG_TIME_BINS)
    freqs = np.linspace(ERSP_FMIN, ERSP_FMAX, IMG_FREQ_BINS)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        cam_mean, aspect="auto", origin="lower", cmap="jet", vmin=0, vmax=1,
        extent=[times[0], times[-1], freqs[0], freqs[-1]]
    )
    ax.axvline(0, color="white", linewidth=1.0, linestyle="--")
    ax.axhspan(8, 13, alpha=0.12, color="white")
    ax.axhspan(14, 30, alpha=0.08, color="white")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Grad-CAM — SpectNet pooled — {cls_label}\n"
                f"Mean over {n_trials} correctly-classified trials (all 9 subjects)")
    plt.colorbar(im, ax=ax, label="Grad-CAM (normalised)")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_channel_saliency(sal_mean: np.ndarray, cls_label: str, n_trials: int, out_path: Path):
    times = np.linspace(ANALYSIS_TMIN, ANALYSIS_TMAX, IMG_TIME_BINS)
    freqs = np.linspace(ERSP_FMIN, ERSP_FMAX, IMG_FREQ_BINS)
    vmax = sal_mean.max()  # SAME scale for every channel - see channel_saliency() docstring

    fig, axes = plt.subplots(1, N_CHANNELS, figsize=(5 * N_CHANNELS, 4.2))
    for ch_i, ch_name in enumerate(CHANNELS[:N_CHANNELS]):
        ax = axes[ch_i]
        im = ax.imshow(
            sal_mean[ch_i], aspect="auto", origin="lower", cmap="jet", vmin=0, vmax=vmax,
            extent=[times[0], times[-1], freqs[0], freqs[-1]]
        )
        ax.axvline(0, color="white", linewidth=1.0, linestyle="--")
        ax.axhspan(8, 13, alpha=0.12, color="white")
        ax.axhspan(14, 30, alpha=0.08, color="white")
        ax.set_title(ch_name, fontsize=11)
        ax.set_xlabel("Time (s)")
        if ch_i == 0:
            ax.set_ylabel("Frequency (Hz)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Input-gradient saliency, per channel — SpectNet pooled — {cls_label}\n"
                f"Mean |d(score)/d(input)| over {n_trials} correctly-classified trials",
                fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM / saliency for SpectNet pooled")
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    args = parser.parse_args()

    ckpt = torch.load("results/spectnet_all.pth", map_location="cpu", weights_only=False)
    model = get_model("spectnet", n_channels=N_CHANNELS, n_freq=IMG_FREQ_BINS,
                      n_time=IMG_TIME_BINS, n_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    cam_tool = SpectNetGradCAM(model)

    cams = {0: [], 1: []}
    sals = {0: [], 1: []}

    for subj in args.subjects:
        imgs, y = load_subject_images(subj)
        print(f"  Subject S{subj:02d}: {len(y)} trials")
        for i in range(len(y)):
            x = torch.from_numpy(imgs[i:i + 1])
            with torch.no_grad():
                pred = model(x).argmax(dim=1).item()
            if pred != y[i]:
                continue  # explain correct decisions only
            cls = int(y[i])
            cams[cls].append(cam_tool.gradcam(x, cls))
            sals[cls].append(cam_tool.channel_saliency(x, cls))

    cam_tool.close()

    for cls, cls_label in CLASS_NAMES.items():
        n = len(cams[cls])
        print(f"\n{cls_label}: {n} correctly-classified trials pooled")
        if n == 0:
            continue
        cam_mean = np.mean(cams[cls], axis=0)
        sal_mean = np.mean(sals[cls], axis=0)
        plot_gradcam(cam_mean, cls_label, n, OUT_DIR / f"gradcam_{cls_label.lower()}.png")
        plot_channel_saliency(sal_mean, cls_label, n,
                              OUT_DIR / f"saliency_{cls_label.lower()}.png")

    print(f"\nFigures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
