"""
experiment_resolution_sweep.py
----------------------------------
Per advisor request: reduce the ERSP image resolution to cut the per-trial
"weight" (size) of the data, and apply a Gaussian blur before downsampling
to avoid aliasing - find which combination of (time resolution, blur sigma)
keeps accuracy while shrinking the image.

Frequency resolution is left untouched (IMG_FREQ_BINS=22 covering 8-30Hz is
already narrow; the time axis has much more slack - 128 bins over a 4.5s
window is far finer than the slow ERD/ERS dynamics it needs to represent).

Grid: TIME_BINS_GRID x SIGMA_GRID (9 combinations), SpectNet trained from
scratch on each, pooled across DEFAULT_SUBJECTS (same CPU-friendly 4-subject
subset used in experiment_ast.py) - full 9-subject retraining of the winning
combination is a separate, later step once one is chosen here.

Images are generated in memory from the epoch .fif files directly (never
written to data/processed/*.npz) - this never touches the production
22x128 dataset or any committed result file, so it is safe to run
alongside the normal pipeline.

Usage:
    python src/experiment_resolution_sweep.py
    python src/experiment_resolution_sweep.py --subjects 1 3 4 6
    python src/experiment_resolution_sweep.py --time_bins_grid 128 64 32 --sigma_grid 0 0.5 1.0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mne
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROC, N_CHANNELS, IMG_FREQ_BINS, SFREQ,
    BATCH_SIZE, MAX_EPOCHS, PATIENCE, WEIGHT_DECAY, LEARNING_RATE,
    VAL_SPLIT, RANDOM_SEED, DEVICE,
)
from ersp import compute_ersp_image
from models import get_model
from train_fbcsp import EVENT_LABEL_MAP

mne.set_log_level("ERROR")

DEFAULT_SUBJECTS = [1, 3, 4, 6]  # same CPU-friendly subset as experiment_ast.py
DEFAULT_TIME_BINS_GRID = [128, 64, 32]   # 128 = current production resolution
DEFAULT_SIGMA_GRID = [0.0, 0.5, 1.0]     # 0.0 = no blur (current pipeline)


def build_pooled_arrays(subjects: list, suffix: str, n_time_bins: int, sigma: float):
    """Generates ERSP images in memory for all trials of all `subjects` at
    (IMG_FREQ_BINS, n_time_bins) resolution with Gaussian blur `sigma`,
    pooled into one (N, C, F, T) / (N,) pair. Never touches disk."""
    all_X, all_y = [], []
    for subj in subjects:
        path = DATA_PROC / f"S{subj:02d}{suffix}-epo.fif"
        epochs = mne.read_epochs(str(path), verbose=False)
        X_raw = epochs.get_data()  # (N, C, T_samples)
        code_to_label = {v: EVENT_LABEL_MAP[k] for k, v in epochs.event_id.items()}
        y = np.array([code_to_label[ev] for ev in epochs.events[:, 2]])

        X = np.zeros((len(X_raw), N_CHANNELS, IMG_FREQ_BINS, n_time_bins), dtype=np.float32)
        for i in range(len(X_raw)):
            for ch in range(N_CHANNELS):
                X[i, ch] = compute_ersp_image(
                    X_raw[i, ch], sfreq=SFREQ,
                    n_freq_bins=IMG_FREQ_BINS, n_time_bins=n_time_bins, sigma=sigma,
                )
        all_X.append(X)
        all_y.append(y)
    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


def run_combo(X_train_full, y_train_full, X_test, y_test, n_time_bins: int,
             device_str: str = DEVICE):
    """Trains SpectNet from scratch at this resolution, evaluates once on
    the pooled test set at the same resolution."""
    device = torch.device(device_str)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    n = len(y_train_full)
    idx = np.arange(n)
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(idx)
    n_val = int(n * VAL_SPLIT)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train_full[train_idx]), torch.from_numpy(y_train_full[train_idx]).long())
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train_full[val_idx]), torch.from_numpy(y_train_full[val_idx]).long())
    test_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test).long())

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model("spectnet", n_channels=N_CHANNELS, n_freq=IMG_FREQ_BINS,
                      n_time=n_time_bins, n_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val_loss, best_state, patience_count, best_epoch = float("inf"), None, 0, 0
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        _train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _ = _eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_epoch     = epoch
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= PATIENCE:
            break
    model.load_state_dict(best_state)
    t_elapsed = time.time() - t0

    from sklearn.metrics import accuracy_score, cohen_kappa_score
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            preds.append(model(X.to(device)).argmax(dim=1).cpu().numpy())
            labels.append(y.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)

    return {
        "test_acc": accuracy_score(labels, preds),
        "test_kappa": cohen_kappa_score(labels, preds),
        "best_epoch": best_epoch,
        "train_time_s": t_elapsed,
        "n_params": model.count_parameters(),
    }


def main():
    parser = argparse.ArgumentParser(description="ERSP resolution / Gaussian-blur sweep (SpectNet)")
    parser.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--time_bins_grid", type=int, nargs="+", default=DEFAULT_TIME_BINS_GRID)
    parser.add_argument("--sigma_grid", type=float, nargs="+", default=DEFAULT_SIGMA_GRID)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    print(f"\n{'='*70}\n  ERSP resolution / Gaussian-blur sweep\n"
          f"  subjects={args.subjects}  time_bins={args.time_bins_grid}  sigma={args.sigma_grid}\n{'='*70}")

    results = []
    for n_time_bins in args.time_bins_grid:
        for sigma in args.sigma_grid:
            print(f"\n-- time_bins={n_time_bins}  sigma={sigma} --")
            t_data0 = time.time()
            X_train, y_train = build_pooled_arrays(args.subjects, "T", n_time_bins, sigma)
            X_test,  y_test  = build_pooled_arrays(args.subjects, "E", n_time_bins, sigma)
            t_data = time.time() - t_data0

            kb_per_trial = (N_CHANNELS * IMG_FREQ_BINS * n_time_bins * 4) / 1024.0

            r = run_combo(X_train, y_train, X_test, y_test, n_time_bins, args.device)
            r.update({
                "n_time_bins": n_time_bins, "sigma": sigma,
                "kb_per_trial": kb_per_trial, "data_gen_s": t_data,
                "n_train": len(y_train), "n_test": len(y_test),
            })
            results.append(r)
            print(f"    test_acc={r['test_acc']:.1%}  kappa={r['test_kappa']:.3f}  "
                  f"{kb_per_trial:.1f} KB/trial  "
                  f"(data_gen={t_data:.0f}s, train={r['train_time_s']:.0f}s, "
                  f"epoch={r['best_epoch']})")

    print(f"\n{'='*70}\n  Summary (sorted by test accuracy, best first)\n{'='*70}")
    print(f"  {'time_bins':>9} {'sigma':>6} {'KB/trial':>9} {'acc':>7} {'kappa':>7} "
          f"{'train_s':>8} {'epoch':>6}")
    for r in sorted(results, key=lambda r: -r["test_acc"]):
        print(f"  {r['n_time_bins']:>9} {r['sigma']:>6.1f} {r['kb_per_trial']:>9.1f} "
              f"{r['test_acc']:>6.1%} {r['test_kappa']:>7.3f} "
              f"{r['train_time_s']:>7.0f}s {r['best_epoch']:>6}")

    baseline = next((r for r in results if r["n_time_bins"] == 128 and r["sigma"] == 0.0), None)
    if baseline is not None:
        print(f"\n  Baseline (current production: 128 time bins, no blur): "
              f"acc={baseline['test_acc']:.1%}  {baseline['kb_per_trial']:.1f} KB/trial")


if __name__ == "__main__":
    main()
