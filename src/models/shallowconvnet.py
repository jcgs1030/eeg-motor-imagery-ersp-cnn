"""
models/shallowconvnet.py
------------------------
ShallowConvNet adaptado para espectrogramas ERSP 2D.

Referencia original:
    Schirrmeister R.T. et al., "Deep learning with convolutional neural
    networks for EEG decoding and visualization", Hum. Brain Mapp., 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowConvNet(nn.Module):
    """
    ShallowConvNet adaptado para imágenes ERSP.

    Arquitectura:
        Conv2D (n_filters_time filtros, kernel temporal)
        Conv2D (profundidad espacial sobre frecuencia)
        Activación cuadrática → AvgPool → Log → Dropout
        Flatten → FC(n_classes)

    Entrada: (batch, n_channels, n_freq_bins, n_time_bins)
    Salida:  (batch, n_classes)
    """

    def __init__(self,
                 n_channels: int = 3,
                 n_freq: int = 22,
                 n_time: int = 128,
                 n_classes: int = 2,
                 n_filters_time: int = 40,
                 filter_time_length: int = 25,
                 pool_time_length: int = 75,
                 pool_time_stride: int = 15,
                 dropout: float = 0.5):
        super().__init__()

        # ── Convolución temporal ──
        self.conv_time = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_filters_time,
            kernel_size=(1, filter_time_length),
            bias=False
        )

        # ── Convolución frecuencial (espacial) ──
        self.conv_spat = nn.Conv2d(
            in_channels=n_filters_time,
            out_channels=n_filters_time,
            kernel_size=(n_freq, 1),
            bias=False
        )
        self.bn = nn.BatchNorm2d(n_filters_time, eps=1e-5, momentum=0.1)

        # ── Pooling ──
        self.pool = nn.AvgPool2d(
            kernel_size=(1, pool_time_length),
            stride=(1, pool_time_stride)
        )
        self.drop = nn.Dropout(p=dropout)

        self._flat_size = self._compute_flat_size(n_channels, n_freq, n_time)
        self.fc = nn.Linear(self._flat_size, n_classes)

    def _compute_flat_size(self, n_ch, n_freq, n_time) -> int:
        dummy = torch.zeros(1, n_ch, n_freq, n_time)
        with torch.no_grad():
            out = self._conv_forward(dummy)
        return int(out.numel())

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        # Activación cuadrática (característica de ShallowConvNet)
        x = x ** 2
        x = self.pool(x)
        # Log-activación + clamping para estabilidad numérica
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.drop(x)
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_forward(x)
        return self.fc(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ShallowConvNet(n_channels=3, n_freq=22, n_time=128, n_classes=2)
    print(f"ShallowConvNet — Parámetros entrenables: {model.count_parameters():,}")
    x = torch.randn(16, 3, 22, 128)
    out = model(x)
    print(f"Entrada: {x.shape}  →  Salida: {out.shape}")
