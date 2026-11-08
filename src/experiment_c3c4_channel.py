"""
experiment_c3c4_channel.py
------------------------------
Per advisor request: add an explicit C3-C4 difference channel and check
whether it helps SpectNet (a) classify better and (b) show clearer
contralateral lateralization in its per-channel saliency - motivated by the
Grad-CAM finding (gradcam_spectnet.py) that per-channel saliency shows NO
lateralization between C3 and C4, even though the model clearly attends to
the correct mu-band time-frequency region (matching the statistically
significant ERD). The hypothesis: conv1 mixes all 3 channels together
immediately, which may be folding away the C3-vs-C4 relative contrast that
contralaterality predicts; an explicit difference channel hands the model
(and the saliency method) that contrast directly instead of requiring it to
be reconstructed implicitly.

This is exploratory - does not touch the production ersp.py pipeline or any
committed checkpoint. Builds a 4-channel (C3, Cz, C4, C3-C4) ERSP
representation on the fly from the existing epoch .fif files, trains a
SECOND, separate SpectNet pooled model (results/spectnet_c3c4_all.pth, not
overwriting results/spectnet_all.pth), and compares accuracy and per-channel
saliency against the original 3-channel model.

Usage:
    python src/experiment_c3c4_channel.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import mne
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROC, FIGURES_DIR, SUBJECTS, CHANNELS, SFREQ,
    IMG_FREQ_BINS, IMG_TIME_BINS, ANALYSIS_TMIN, ANALYSIS_TMAX,
    ERSP_FMIN, ERSP_FMAX,
    BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, VAL_SPLIT, RANDOM_SEED, DEVICE,
)
from ersp import compute_ersp_image
from models import get_model
from train_fbcsp import EVENT_LABEL_MAP
from gradcam_spectnet import SpectNetGradCAM
from diagnostic_stats_test import per_subject_acc_pooled

mne.set_log_level("WARNING")

N_CHANNELS_4 = 4
CH_NAMES_4 = ["C3", "Cz", "C4", "C3-C4"]
C3_IDX, CZ_IDX, C4_IDX = CHANNELS.index("C3"), CHANNELS.index("Cz"), CHANNELS.index("C4")

OUT_DIR = FIGURES_DIR / "c3c4_channel"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_4ch_images(subject: int, suffix: str):
    """4-channel ERSP images (C3, Cz, C4, C3-C4) for one subject/session group."""
    path = DATA_PROC / f"S{subject:02d}{suffix}-epo.fif"
    epochs = mne.read_epochs(str(path), verbose=False)
    X = epochs.get_data()  # (N, 3, T) in CHANNELS order
    code_to_label = {v: EVENT_LABEL_MAP[k] for k, v in epochs.event_id.items()}
    y = np.array([code_to_label[ev] for ev in epochs.events[:, 2]])

    imgs = np.zeros((len(X), N_CHANNELS_4, IMG_FREQ_BINS, IMG_TIME_BINS), dtype=np.float32)
    for i in range(len(X)):
        imgs[i, 0] = compute_ersp_image(X[i, C3_IDX], sfreq=SFREQ)
        imgs[i, 1] = compute_ersp_image(X[i, CZ_IDX], sfreq=SFREQ)
        imgs[i, 2] = compute_ersp_image(X[i, C4_IDX], sfreq=SFREQ)
        diff = X[i, C3_IDX] - X[i, C4_IDX]
        imgs[i, 3] = compute_ersp_image(diff, sfreq=SFREQ)
    return imgs, y


def build_pooled_arrays(subjects: list, suffix: str):
    all_imgs, all_y = [], []
    for subj in subjects:
        imgs, y = load_4ch_images(subj, suffix)
        all_imgs.append(imgs)
        all_y.append(y)
        print(f"    S{subj:02d}{suffix}: {len(y)} trials")
    return np.concatenate(all_imgs, axis=0), np.concatenate(all_y, axis=0)


def train_pooled_4ch(X_train_full, y_train_full, device_str=DEVICE):
    """Mirrors train.py's train_model(), but on precomputed 4-channel arrays
    held in memory instead of ERSPDataset/build_loaders (which only support
    the cached 3-channel npz files)."""
    device = torch.device(device_str)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    n = len(y_train_full)
    idx = np.random.RandomState(RANDOM_SEED).permutation(n)
    n_val = int(n * VAL_SPLIT)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = torch.from_numpy(X_train_full[train_idx])
    y_train = torch.from_numpy(y_train_full[train_idx]).long()
    X_val   = torch.from_numpy(X_train_full[val_idx])
    y_val   = torch.from_numpy(y_train_full[val_idx]).long()

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds   = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model("spectnet", n_channels=N_CHANNELS_4, n_freq=IMG_FREQ_BINS,
                      n_time=IMG_TIME_BINS, n_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val_loss, best_state, patience_count = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                val_loss += criterion(logits, yb).item() * len(yb)
                correct  += (logits.argmax(dim=1) == yb).sum().item()
                total    += len(yb)
        val_loss /= total
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= PATIENCE:
            print(f"    Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
            break

    model.load_state_dict(best_state)
    print(f"    Training done in {time.time()-t0:.0f}s")
    return model


@torch.no_grad()
def evaluate_per_subject(model, subjects, device_str=DEVICE):
    device = torch.device(device_str)
    accs = {}
    for subj in subjects:
        imgs, y = load_4ch_images(subj, "E")
        x = torch.from_numpy(imgs).to(device)
        pred = model(x).argmax(dim=1).cpu().numpy()
        accs[subj] = (pred == y).mean()
    return accs


def plot_channel_saliency_4ch(sal_mean, cls_label, n_trials, out_path):
    times = np.linspace(ANALYSIS_TMIN, ANALYSIS_TMAX, IMG_TIME_BINS)
    freqs = np.linspace(ERSP_FMIN, ERSP_FMAX, IMG_FREQ_BINS)
    vmax = sal_mean.max()  # SAME scale for all 4 channels - see channel_saliency() docstring

    fig, axes = plt.subplots(1, N_CHANNELS_4, figsize=(5 * N_CHANNELS_4, 4.2))
    for ch_i, ch_name in enumerate(CH_NAMES_4):
        ax = axes[ch_i]
        im = ax.imshow(
            sal_mean[ch_i], aspect="auto", origin="lower", cmap="jet", vmin=0, vmax=vmax,
            extent=[times[0], times[-1], freqs[0], freqs[-1]]
        )
        ax.axvline(0, color="white", linewidth=1.0, linestyle="--")
        ax.axhspan(8, 13, alpha=0.12, color="white")
        ax.axhspan(14, 30, alpha=0.08, color="white")
        ax.set_title(ch_name, fontsize=11,
                    fontweight="bold" if ch_name == "C3-C4" else "normal")
        ax.set_xlabel("Time (s)")
        if ch_i == 0:
            ax.set_ylabel("Frequency (Hz)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Input-gradient saliency, 4-channel SpectNet — {cls_label}\n"
                f"Mean |d(score)/d(input)| over {n_trials} correctly-classified trials",
                fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main():
    print("== Step 1: building 4-channel (C3, Cz, C4, C3-C4) ERSP arrays ==")
    print("  Training sessions (1-3):")
    X_train, y_train = build_pooled_arrays(SUBJECTS, "T")
    print(f"  Total training trials: {len(y_train)}")

    print("\n== Step 2: training SpectNet pooled, 4-channel input ==")
    model = train_pooled_4ch(X_train, y_train)

    ckpt_path = "results/spectnet_c3c4_all.pth"
    torch.save({"model_name": "spectnet", "n_channels": N_CHANNELS_4,
               "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}},
              ckpt_path)
    print(f"  Saved: {ckpt_path}")

    print("\n== Step 3: evaluating on test (sessions 4-5) ==")
    accs = evaluate_per_subject(model, SUBJECTS)
    for subj, acc in accs.items():
        print(f"  S{subj:02d}: {acc:.1%}")
    vals = np.array(list(accs.values()))

    # Baseline read from the CURRENT results/spectnet_all.pth checkpoint,
    # not a hardcoded number - a hardcoded literal here went stale the
    # moment the 3-channel model was retrained at a different resolution
    # (the 71.5% figure from EXPERIMENTS.md was at IMG_TIME_BINS=128;
    # the checkpoint and this script both now run at whatever is in
    # config.py, so this comparison always matches what's actually loaded).
    baseline_accs = per_subject_acc_pooled("spectnet")
    baseline_vals = np.array(list(baseline_accs.values()))

    print(f"\n  4-channel SpectNet pooled: {vals.mean():.1%} ± {vals.std():.1%}")
    print(f"  3-channel SpectNet pooled (baseline, current results/spectnet_all.pth): "
          f"{baseline_vals.mean():.1%} ± {baseline_vals.std():.1%}")
    print(f"  Difference: {100*(vals.mean()-baseline_vals.mean()):+.1f}pp")

    print("\n== Step 4: per-channel saliency with the 4-channel model ==")
    cam_tool = SpectNetGradCAM(model)
    sals = {0: [], 1: []}
    for subj in SUBJECTS:
        imgs, y = load_4ch_images(subj, "E")
        for i in range(len(y)):
            x = torch.from_numpy(imgs[i:i + 1])
            with torch.no_grad():
                pred = model(x).argmax(dim=1).item()
            if pred != y[i]:
                continue
            sals[int(y[i])].append(cam_tool.channel_saliency(x, int(y[i])))
    cam_tool.close()

    for cls, cls_label in {0: "Left", 1: "Right"}.items():
        n = len(sals[cls])
        if n == 0:
            continue
        sal_mean = np.mean(sals[cls], axis=0)
        plot_channel_saliency_4ch(sal_mean, cls_label, n,
                                  OUT_DIR / f"saliency_4ch_{cls_label.lower()}.png")

    print(f"\nFigures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
