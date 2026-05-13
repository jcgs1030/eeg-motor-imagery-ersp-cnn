"""
models/eegnet.py
----------------
EEGNet adapted for 2D ERSP spectrograms.

Original reference:
    Lawhern V.J. et al., "EEGNet: A compact convolutional neural network
    for EEG-based brain-computer interfaces", J. Neural Eng., 2018.

Adaptation:
    Input is an ERSP spectrogram (n_channels, n_freq, n_time)
    instead of the raw EEG signal. The temporal-spatial convolution
    is replaced by a standard Conv2D over time × frequency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet adapted for ERSP images.

    Architecture:
        Conv2D temporal-frequency (F1 filters)
        DepthwiseConv2D spatial (D filters per channel)
        BatchNorm + ELU + AvgPool + Dropout
        SeparableConv2D (F2 filters)
        BatchNorm + ELU + AvgPool + Dropout
        Flatten → FC(n_classes)

    Input:  (batch, n_channels, n_freq_bins, n_time_bins)
    Output: (batch, n_classes)
    """

    def __init__(self,
                 n_channels: int = 3,
                 n_freq: int = 22,
                 n_time: int = 128,
                 n_classes: int = 2,
                 F1: int = 8,       # number of temporal filters
                 D: int = 2,        # depthwise depth multiplier
                 F2: int = 16,      # number of separable filters
                 dropout: float = 0.5):
        super().__init__()
        F2 = F1 * D

        # ── Block 1: Conv2D ──
        self.conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=F1,
            kernel_size=(1, 64),
            padding=(0, 32),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # ── Block 2: depthwise convolution ──
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

        # ── Block 3: separable convolution ──
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

        # Compute flattened output size
        self._flat_size = self._compute_flat_size(n_channels, n_freq, n_time)
        self.fc = nn.Linear(self._flat_size, n_classes)

    def _compute_flat_size(self, n_ch, n_freq, n_time) -> int:
        dummy = torch.zeros(1, n_ch, n_freq, n_time)
        with torch.no_grad():
            out = self._conv_forward(dummy)
        return int(out.numel())

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.bn1(self.conv1(x))
        # Block 2: depthwise
        x = F.elu(self.bn2(self.dw_conv(x)))
        x = self.drop1(self.pool1(x))
        # Block 3: separable
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
    print(f"EEGNet — Trainable parameters: {model.count_parameters():,}")
    x = torch.randn(16, 3, 22, 128)
    out = model(x)
    print(f"Input: {x.shape}  →  Output: {out.shape}")
