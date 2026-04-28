"""
dataset.py
----------
PyTorch Dataset para cargar los espectrogramas ERSP del BCI-IV-2b.
Compatible con DataLoader para entrenamiento por minibatches.

Uso:
    from dataset import ERSPDataset, build_loaders
    train_loader, val_loader = build_loaders(subjects=[1,2,3], suffix='T')
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_PROC, SUBJECTS, BATCH_SIZE, VAL_SPLIT, RANDOM_SEED,
    N_CHANNELS, IMG_FREQ_BINS, IMG_TIME_BINS, TRAIN_SUFFIX, EVAL_SUFFIX
)


class ERSPDataset(Dataset):
    """
    Dataset de espectrogramas ERSP para el BCI-IV-2b.

    Carga los archivos .npz generados por ersp.py y los sirve
    como tensores (X, y) para PyTorch.

    Parámetros
    ----------
    subjects : lista de enteros (1–9) con los sujetos a incluir
    suffix   : 'T' (entrenamiento/sesiones 1-3) o 'E' (evaluación/sesiones 4-5)
    transform: transformación opcional sobre las imágenes (data augmentation)
    """

    def __init__(self,
                 subjects: list = None,
                 suffix: str = TRAIN_SUFFIX,
                 transform=None):
        if subjects is None:
            subjects = SUBJECTS

        self.transform = transform
        X_all, y_all = [], []

        for subj in subjects:
            tag = f"S{subj:02d}{suffix}"
            npz_path = DATA_PROC / f"{tag}-ersp.npz"

            if not npz_path.exists():
                print(f"  Advertencia: {npz_path.name} no encontrado — sujeto omitido.")
                continue

            data = np.load(str(npz_path))
            X_all.append(data["X"])   # (N, C, F, T)
            y_all.append(data["y"])   # (N,)

        if not X_all:
            raise FileNotFoundError(
                "No se encontraron archivos ERSP. "
                "Ejecuta primero: python src/ersp.py"
            )

        self.X = torch.tensor(np.concatenate(X_all, axis=0), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate(y_all, axis=0), dtype=torch.long)

        n_left  = (self.y == 0).sum().item()
        n_right = (self.y == 1).sum().item()
        print(f"  Dataset cargado: {len(self.X)} imágenes "
              f"(izquierda: {n_left}, derecha: {n_right}) — "
              f"forma: {tuple(self.X.shape)}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    @property
    def n_classes(self) -> int:
        return int(self.y.max().item()) + 1

    @property
    def class_weights(self) -> torch.Tensor:
        """Pesos de clase inversos a la frecuencia (útil para datasets desbalanceados)."""
        counts = torch.bincount(self.y)
        weights = 1.0 / counts.float()
        weights = weights / weights.sum()
        return weights


def build_loaders(subjects: list = None,
                  suffix_train: str = TRAIN_SUFFIX,
                  suffix_test: str = EVAL_SUFFIX,
                  batch_size: int = BATCH_SIZE,
                  val_split: float = VAL_SPLIT,
                  seed: int = RANDOM_SEED):
    """
    Construye DataLoaders de entrenamiento, validación y prueba.

    Protocolo BCI-IV-2b:
        - Entrenamiento: sesiones 1-3 (suffix='T'), con val_split reservado para validación
        - Prueba: sesiones 4-5 (suffix='E')

    Parámetros
    ----------
    subjects     : lista de sujetos (None = todos)
    suffix_train : 'T' para sesiones de entrenamiento
    suffix_test  : 'E' para sesiones de evaluación
    batch_size   : tamaño del minibatch
    val_split    : fracción del conjunto de entrenamiento para validación
    seed         : semilla para reproducibilidad

    Retorna
    -------
    train_loader, val_loader, test_loader
    """
    if subjects is None:
        subjects = SUBJECTS

    print("\n── Cargando dataset de entrenamiento ──")
    train_full = ERSPDataset(subjects=subjects, suffix=suffix_train)

    print("\n── Cargando dataset de evaluación ──")
    test_ds = ERSPDataset(subjects=subjects, suffix=suffix_test)

    # División train / validación
    n_val   = int(len(train_full) * val_split)
    n_train = len(train_full) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        train_full, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    print(f"\n  Tamaños: train={n_train}, val={n_val}, test={len(test_ds)}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test rápido
    try:
        train_loader, val_loader, test_loader = build_loaders(subjects=[1])
        x_batch, y_batch = next(iter(train_loader))
        print(f"\n  Lote de entrenamiento: X={x_batch.shape}, y={y_batch.shape}")
    except FileNotFoundError as e:
        print(f"\n  {e}")
