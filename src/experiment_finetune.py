"""
experiment_finetune.py
-------------------------
Exploratory: does fine-tuning the SpectNet pooled checkpoint recover any of
the per-subject gap vs. subject-specific training, following the
transfer-learning approach reported in the BCI-IV literature (freeze the
convolutional feature extractor, fine-tune only the classifier head with a
low learning rate on a small amount of subject data)?

Two modes, both starting from the SAME pooled checkpoint
(results/spectnet_all.pth):

  --mode persubject (default)
      One fine-tuned model PER subject. Freezes conv1/conv2, fine-tunes
      fc1+fc2 only, using that subject's own training-session (T) data
      (~340 trials after the val split). Early stopping on that subject's
      own validation loss. Produces 9 specialised models.

  --mode global
      ONE model for all subjects. Continues training the pooled checkpoint
      (unfreezes everything) on the full pooled training set again, at a
      lower learning rate, for a few more epochs. This is NOT the same as
      the original pooled training - it starts from the already-converged
      checkpoint and just gives it a short low-LR continuation. Answers
      "fine-tuning sobre el promedio/pooled en vez de por sujeto".

Does not touch any committed result file.

Usage:
    python src/experiment_finetune.py --mode persubject
    python src/experiment_finetune.py --mode global
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SUBJECTS, N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS,
    BATCH_SIZE, WEIGHT_DECAY, RANDOM_SEED, DEVICE
)
from dataset import build_loaders, ERSPDataset
from models import get_model
from diagnostic_stats_test import per_subject_acc_pooled

FT_LR          = 1e-4   # conservative, per literature's "fine-tuning schedule"
FT_MAX_EPOCHS  = 60
FT_PATIENCE    = 10


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


@torch.no_grad()
def _predict_all(model, dataset, device):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    preds, labels = [], []
    for X, y in loader:
        preds.append(model(X.to(device)).argmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def _load_pooled_model(device):
    ckpt = torch.load("results/spectnet_all.pth", map_location="cpu", weights_only=False)
    model = get_model("spectnet", n_channels=N_CHANNELS, n_freq=IMG_FREQ_BINS,
                      n_time=IMG_TIME_BINS, n_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)


def finetune_per_subject(subjects, device_str=DEVICE):
    """Freeze conv1/conv2, fine-tune fc1+fc2 only, per subject."""
    device = torch.device(device_str)
    results = {}

    for subj in subjects:
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        model = _load_pooled_model(device)
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.conv2.parameters():
            p.requires_grad = False

        train_loader, val_loader, _ = build_loaders(
            subjects=[subj], batch_size=BATCH_SIZE, seed=RANDOM_SEED
        )

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable, lr=FT_LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        best_val_loss, best_state, patience_count = float("inf"), None, 0
        t0 = time.time()
        for epoch in range(1, FT_MAX_EPOCHS + 1):
            _train_epoch(model, train_loader, optimizer, criterion, device)
            vl_loss, _ = _eval_epoch(model, val_loader, criterion, device)
            if vl_loss < best_val_loss:
                best_val_loss  = vl_loss
                best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= FT_PATIENCE:
                break
        model.load_state_dict(best_state)

        test_ds = ERSPDataset(subjects=[subj], suffix="E")
        y_pred, y_true = _predict_all(model, test_ds, device)
        acc = (y_pred == y_true).mean()
        results[subj] = acc
        print(f"  S{subj:02d}: fine-tuned acc={acc:.1%}  ({time.time()-t0:.0f}s, "
              f"{epoch} epochs)")

    return results


def finetune_global(device_str=DEVICE, extra_epochs=30, lr=3e-5):
    """Continue training the pooled checkpoint, unfrozen, on the full pooled
    training set again at a low LR. One model, evaluated per subject."""
    device = torch.device(device_str)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = _load_pooled_model(device)
    train_loader, val_loader, _ = build_loaders(
        subjects=SUBJECTS, batch_size=BATCH_SIZE, seed=RANDOM_SEED
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val_loss, best_state, patience_count = float("inf"), None, 0
    for epoch in range(1, extra_epochs + 1):
        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = _eval_epoch(model, val_loader, criterion, device)
        print(f"  epoch {epoch:2d}: train_loss={tr_loss:.4f} train_acc={tr_acc:.1%} "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.1%}")
        if vl_loss < best_val_loss:
            best_val_loss  = vl_loss
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= FT_PATIENCE:
            break
    model.load_state_dict(best_state)

    results = {}
    for subj in SUBJECTS:
        test_ds = ERSPDataset(subjects=[subj], suffix="E")
        y_pred, y_true = _predict_all(model, test_ds, device)
        results[subj] = (y_pred == y_true).mean()
    return results


def main():
    parser = argparse.ArgumentParser(description="SpectNet fine-tuning experiment")
    parser.add_argument("--mode", choices=["persubject", "global"], default="persubject")
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    baseline = per_subject_acc_pooled("spectnet")

    if args.mode == "persubject":
        print(f"Fine-tuning SpectNet pooled, PER SUBJECT "
              f"(freeze conv1/conv2, lr={FT_LR}), subjects={args.subjects}\n")
        results = finetune_per_subject(args.subjects, args.device)
    else:
        print(f"Fine-tuning SpectNet pooled, GLOBAL continuation "
              f"(all layers unfrozen, low lr, full pooled set)\n")
        results = finetune_global(args.device)

    print("\n== Per-subject comparison: pooled baseline vs. fine-tuned ==")
    diffs = []
    for subj in sorted(results.keys()):
        b = baseline[subj]
        f = results[subj]
        diffs.append(f - b)
        print(f"  S{subj:02d}: pooled={b:.1%}  fine-tuned={f:.1%}  diff={100*(f-b):+.1f}pp")

    b_mean = np.mean([baseline[s] for s in results.keys()])
    f_mean = np.mean([results[s] for s in results.keys()])
    print(f"\n  Pooled baseline mean : {b_mean:.1%}")
    print(f"  Fine-tuned mean      : {f_mean:.1%}")
    print(f"  Mean diff            : {100*(f_mean-b_mean):+.1f}pp")


if __name__ == "__main__":
    main()
