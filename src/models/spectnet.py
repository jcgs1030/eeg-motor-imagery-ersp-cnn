"""
models/spectnet.py
------------------
SpectNet: CNN de 4 capas convolucionales para clasificación
de espectrogramas EEG (Ruffini et al., 2018).

Referencia:
    Ruffini G. et al., "Deep learning using EEG spectrograms for prognosis
    in idiopathic REM behavior disorder (RBD)", arXiv, 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectNet(nn.Module):
    """
    CNN ligera de 4 capas convolucionales diseñada para espectrogramas EEG.

    Arquitectura (adaptada de Ruffini et al. 2018):
        Conv2D(depth=8)  → ReLU → MaxPool(temporal)
        Conv2D(depth=16) → ReLU → MaxPool(temporal)
        Flatten
        FC(256) → Dropout(0.5)
        FC(n_classes)   → Softmax

    Entrada: (batch, n_channels, n_freq_bins, n_time_bins)
             e.g. (32, 3, 22, 128)
    Salida:  (batch, n_classes)
             e.g. (32, 2)
    """

    def __init__(self,
                 n_channels: int = 3,
                 n_freq: int = 22,
                 n_time: int = 128,
                 n_classes: int = 2,
                 dropout: float = 0.5):
        super().__init__()

        # ── Bloque convolucional 1 ──
        self.conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=8,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4))   # reducción temporal

        # ── Bloque convolucional 2 ──
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(2, 4),
            padding=(0, 0)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4))   # reducción temporal

        # Calcular tamaño de la capa plana
        self._flat_size = self._compute_flat_size(n_channels, n_freq, n_time)

        # ── Capas fully connected ──
        self.fc1     = nn.Linear(self._flat_size, 256)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2     = nn.Linear(256, n_classes)

    def _compute_flat_size(self, n_ch, n_freq, n_time) -> int:
        """Calcula el tamaño de salida de las capas convolucionales."""
        dummy = torch.zeros(1, n_ch, n_freq, n_time)
        with torch.no_grad():
            out = self._conv_forward(dummy)
        return int(out.numel())

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_forward(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test rápido del modelo
    model = SpectNet(n_channels=3, n_freq=22, n_time=128, n_classes=2)
    print(f"SpectNet — Parámetros entrenables: {model.count_parameters():,}")

    x = torch.randn(16, 3, 22, 128)
    out = model(x)
    print(f"Entrada: {x.shape}  →  Salida: {out.shape}")
