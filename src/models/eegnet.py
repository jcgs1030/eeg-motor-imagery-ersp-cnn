"""
models/eegnet.py
----------------
EEGNet adaptado para espectrogramas ERSP 2D.

Referencia original:
    Lawhern V.J. et al., "EEGNet: A compact convolutional neural network
    for EEG-based brain-computer interfaces", J. Neural Eng., 2018.

Adaptación:
    La entrada es un espectrograma ERSP (n_channels, n_freq, n_time)
    en lugar de la señal EEG cruda. Se sustituye la convolución
    temporal-espacial por una conv2D estándar sobre tiempo × frecuencia.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet adaptado para imágenes ERSP.

    Arquitectura:
        Conv2D temporal-frecuencial (F1 filtros)
        DepthwiseConv2D espacial (D filtros por canal)
        BatchNorm + ELU + AvgPool + Dropout
        SeparableConv2D (F2 filtros)
        BatchNorm + ELU + AvgPool + Dropout
        Flatten → FC(n_classes)

    Entrada: (batch, n_channels, n_freq_bins, n_time_bins)
    Salida:  (batch, n_classes)
    """

    def __init__(self,
                 n_channels: int = 3,
                 n_freq: int = 22,
                 n_time: int = 128,
                 n_classes: int = 2,
                 F1: int = 8,       # número de filtros temporales
                 D: int = 2,        # factor de profundidad (depthwise)
                 F2: int = 16,      # número de filtros separables
                 dropout: float = 0.5):
        super().__init__()
        F2 = F1 * D

        # ── Bloque 1: convolución 2D ──
        self.conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=F1,
            kernel_size=(1, 64),
            padding=(0, 32),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # ── Bloque 2: depthwise convolution ──
        self.dw_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_freq, 1),
            groups=F1,
            bias=False
        )
        self.bn2    = nn.BatchNorm2d(F1 * D)
        self.pool1  = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1  = nn.Dropout(p=dropout)

        # ── Bloque 3: separable convolution ──
        self.sep_conv = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 16),
            padding=(0, 8),
            bias=False
        )
        self.bn3   = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(p=dropout)

        # Calcular tamaño de salida
        self._flat_size = self._compute_flat_size(n_channels, n_freq, n_time)
        self.fc = nn.Linear(self._flat_size, n_classes)

    def _compute_flat_size(self, n_ch, n_freq, n_time) -> int:
        dummy = torch.zeros(1, n_ch, n_freq, n_time)
        with torch.no_grad():
            out = self._conv_forward(dummy)
        return int(out.numel())

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bloque 1
        x = self.bn1(self.conv1(x))
        # Bloque 2: depthwise
        x = F.elu(self.bn2(self.dw_conv(x)))
        x = self.drop1(self.pool1(x))
        # Bloque 3: separable
        x = F.elu(self.bn3(self.sep_conv(x)))
        x = self.drop2(self.pool2(x))
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_forward(x)
        return self.fc(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = EEGNet(n_channels=3, n_freq=22, n_time=128, n_classes=2)
    print(f"EEGNet — Parámetros entrenables: {model.count_parameters():,}")
    x = torch.randn(16, 3, 22, 128)
    out = model(x)
    print(f"Entrada: {x.shape}  →  Salida: {out.shape}")
