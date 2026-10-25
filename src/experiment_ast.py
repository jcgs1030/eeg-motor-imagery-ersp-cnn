"""
experiment_ast.py
---------------------
Per advisor request: fine-tune AST (Audio Spectrogram Transformer,
Gong et al. 2021, Hugging Face `MIT/ast-finetuned-audioset-10-10-0.4593`)
on our ERSP images, and compare against SpectNet and FBCSP+LDA using the
same per-subject train(T)/test(E) split.

Architecture / adaptation notes
--------------------------------
AST is pretrained on single-channel, 128-mel-bin x 1024-frame audio
spectrograms (~1200 patches, 86M params) - our ERSP images are 3-channel,
22-freq-bin x 128-time-bin. Two adaptation choices were made, both driven by
being CPU-only in this environment (benchmarked: ~2s/sample forward alone at
AST's native 128x1024 resolution, CPU-bound regardless of batch size -
intractable for fine-tuning on this project's dataset size):

1. The 3 EEG channels (C3, Cz, C4) are stacked along the frequency axis into
   ONE single-channel "image" (time=128, freq=66), instead of running AST
   three times (once per channel) - one AST forward pass per trial instead
   of three.
2. A smaller patch grid is used (patch_size=8, frequency_stride=8,
   time_stride=8, vs. AST's native patch_size=16/stride=10) to keep the
   number of patches - and therefore attention cost, which is quadratic in
   sequence length - tractable (130 patches instead of ~1200). This means
   the patch-embedding convolution and the (now mismatched-size) position
   embeddings CANNOT be loaded from the pretrained checkpoint and start
   untrained; all 12 transformer encoder blocks (the bulk of the 86M
   parameters) still load pretrained AudioSet weights. Benchmarked at
   ~130-220ms/sample at this resolution, vs. ~2s/sample at native
   resolution - about 10-15x faster, the difference between infeasible and
   feasible on CPU.

Conservative fine-tuning (matches the "freeze most of the backbone"
practice already documented in EXPERIMENTS.md's literature review): only
the embeddings (necessarily, since reinitialised), the last `--unfreeze`
transformer blocks, and a new linear classifier head are trained; the
remaining pretrained blocks are frozen.

Due to the CPU cost above, this runs on a SUBSET of subjects (not all 9) -
clearly documented, not hidden. SpectNet and FBCSP+LDA comparison numbers
are the SAME subjects' existing per-subject test accuracy, for a fair,
like-for-like comparison at this smaller scale.

Usage:
    python src/experiment_ast.py                        # default subset
    python src/experiment_ast.py --subjects 1 3 4 6      # choose subjects
    python src/experiment_ast.py --max_epochs 10 --unfreeze 2
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
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_PROC, SFREQ, CHANNELS, N_CHANNELS, RANDOM_SEED, DEVICE
from ersp import compute_ersp_image
from models import get_model
from train_fbcsp import EVENT_LABEL_MAP
from diagnostic_stats_test import per_subject_acc_pooled, load_per_subject_csv

mne.set_log_level("WARNING")

AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
NUM_MEL_BINS   = 66   # 3 channels x 22 freq bins, stacked
MAX_LENGTH     = 128  # time bins, unchanged
PATCH_SIZE     = 8
FREQ_STRIDE    = 8
TIME_STRIDE    = 8

DEFAULT_SUBJECTS = [1, 3, 4, 6]  # mix of weak (S03), average (S01, S06), strong (S04) signal


def build_ast_model(unfreeze: int):
    from transformers import ASTConfig, ASTModel

    cfg = ASTConfig(num_mel_bins=NUM_MEL_BINS, max_length=MAX_LENGTH, patch_size=PATCH_SIZE,
                    frequency_stride=FREQ_STRIDE, time_stride=TIME_STRIDE)
    backbone = ASTModel.from_pretrained(AST_CHECKPOINT, config=cfg, ignore_mismatched_sizes=True)

    for p in backbone.parameters():
        p.requires_grad = False
    for p in backbone.embeddings.parameters():
        p.requires_grad = True   # reinitialised (size mismatch) - must be trained
    if unfreeze > 0:
        for p in backbone.layers[-unfreeze:].parameters():
            p.requires_grad = True
    for p in backbone.layernorm.parameters():
        p.requires_grad = True

    return backbone


class ASTClassifier(nn.Module):
    def __init__(self, unfreeze: int, n_classes: int = 2):
        super().__init__()
        self.backbone = build_ast_model(unfreeze)
        self.head = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.backbone(x)
        return self.head(out.pooler_output)


def load_stacked_images(subject: int, suffix: str):
    """Stacks C3/Cz/C4 ERSP images along the frequency axis into one
    (N, time=128, freq=66) single-channel array, AST's expected input
    layout (batch, time, freq)."""
    path = DATA_PROC / f"S{subject:02d}{suffix}-epo.fif"
    epochs = mne.read_epochs(str(path), verbose=False)
    X = epochs.get_data()  # (N, 3, T_raw)
    code_to_label = {v: EVENT_LABEL_MAP[k] for k, v in epochs.event_id.items()}
    y = np.array([code_to_label[ev] for ev in epochs.events[:, 2]])

    n = len(X)
    stacked = np.zeros((n, MAX_LENGTH, NUM_MEL_BINS), dtype=np.float32)
    for i in range(n):
        for ch in range(N_CHANNELS):
            img = compute_ersp_image(X[i, ch], sfreq=SFREQ)  # (freq=22, time=128)
            stacked[i, :, ch * 22:(ch + 1) * 22] = img.T       # -> (time=128, freq=22)
    return stacked, y


def build_pooled_arrays(subjects: list, suffix: str):
    all_x, all_y = [], []
    for subj in subjects:
        x, y = load_stacked_images(subj, suffix)
        all_x.append(x)
        all_y.append(y)
        print(f"    S{subj:02d}{suffix}: {len(y)} trials")
    return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)


def train_ast(X_train_full, y_train_full, unfreeze: int, max_epochs: int,
             patience: int, batch_size: int, lr: float, device_str=DEVICE):
    device = torch.device(device_str)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    n = len(y_train_full)
    idx = np.random.RandomState(RANDOM_SEED).permutation(n)
    n_val = max(1, int(n * 0.15))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train_full[train_idx]), torch.from_numpy(y_train_full[train_idx]).long())
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train_full[val_idx]), torch.from_numpy(y_train_full[val_idx]).long())
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ASTClassifier(unfreeze=unfreeze).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"    Trainable parameters: {sum(p.numel() for p in trainable):,} "
          f"/ {sum(p.numel() for p in model.parameters()):,} total")
    optimizer = optim.Adam(trainable, lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss, best_state, patience_count = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
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
        val_acc = correct / total
        elapsed = time.time() - t0
        print(f"    epoch {epoch:2d}: val_loss={val_loss:.4f} val_acc={val_acc:.1%} "
              f"({elapsed:.0f}s elapsed)")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"    Training done in {time.time()-t0:.0f}s")
    return model


@torch.no_grad()
def evaluate_per_subject(model, subjects, device_str=DEVICE):
    device = torch.device(device_str)
    model.eval()
    accs, kappas = {}, {}
    for subj in subjects:
        x, y = load_stacked_images(subj, "E")
        preds = []
        bs = 16
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(x[i:i + bs]).to(device)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
        preds = np.concatenate(preds)
        accs[subj] = (preds == y).mean()
        kappas[subj] = cohen_kappa_score(y, preds)
    return accs, kappas


def main():
    parser = argparse.ArgumentParser(description="AST fine-tuning on ERSP images")
    parser.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze", type=int, default=2,
                        help="Number of final transformer blocks to fine-tune (rest frozen)")
    args = parser.parse_args()

    print(f"\n{'='*60}\n  AST fine-tuning on ERSP images\n"
          f"  Subjects: {args.subjects} (subset - see module docstring)\n{'='*60}")

    print("\n== Step 1: building stacked (C3|Cz|C4) ERSP images ==")
    print("  Training sessions (1-3):")
    X_train, y_train = build_pooled_arrays(args.subjects, "T")
    print(f"  Total training trials: {len(y_train)}")

    print(f"\n== Step 2: fine-tuning AST (last {args.unfreeze} blocks + embeddings + head) ==")
    model = train_ast(X_train, y_train, args.unfreeze, args.max_epochs,
                      args.patience, args.batch_size, args.lr)

    print("\n== Step 3: evaluating on test (sessions 4-5) ==")
    ast_acc, ast_kappa = evaluate_per_subject(model, args.subjects)

    print("\n== Step 4: comparing against SpectNet pooled and FBCSP+LDA (same subjects) ==")
    spectnet_acc = per_subject_acc_pooled("spectnet")
    fbcsp_acc = load_per_subject_csv(
        "results/fbcsp/metrics/per_subject_results.csv", model_filter="FBCSP+LDA",
        model_col="Classifier")

    print(f"\n  {'Subject':>8} {'AST':>8} {'SpectNet':>10} {'FBCSP+LDA':>10}")
    for subj in args.subjects:
        print(f"  S{subj:02d}    {ast_acc[subj]:>7.1%} {spectnet_acc[subj]:>9.1%} "
              f"{fbcsp_acc[subj]:>9.1%}")

    ast_vals = np.array([ast_acc[s] for s in args.subjects])
    spectnet_vals = np.array([spectnet_acc[s] for s in args.subjects])
    fbcsp_vals = np.array([fbcsp_acc[s] for s in args.subjects])
    ast_kappa_vals = np.array([ast_kappa[s] for s in args.subjects])

    print(f"\n  == Summary (mean ± std over {len(args.subjects)} subjects - "
          f"too few for a formal significance test) ==")
    print(f"  AST (fine-tuned)  : {ast_vals.mean():.1%} ± {ast_vals.std():.1%}  "
          f"kappa={ast_kappa_vals.mean():.3f}")
    print(f"  SpectNet (pooled) : {spectnet_vals.mean():.1%} ± {spectnet_vals.std():.1%}")
    print(f"  FBCSP+LDA         : {fbcsp_vals.mean():.1%} ± {fbcsp_vals.std():.1%}")


if __name__ == "__main__":
    main()
