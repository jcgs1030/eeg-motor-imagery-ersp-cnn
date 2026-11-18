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
                 dropout: float = 0.5,
                 filter_time_length: int = None,
                 sep_filter_length: int = None,
                 pool1_length: int = None,
                 pool2_length: int = None):
        super().__init__()
        F2 = F1 * D

        # Temporal kernel and pooling lengths were tuned for 128 time bins
        # (64/16 kernels, 4/8 pools). Unless given explicitly they now scale
        # with n_time, the way ShallowConvNet's already did. At n_time=128 this
        # reproduces the original values exactly (64/16/4/8).
        #
        # Why this is needed (external review, M-7): with the 128-bin values
        # held fixed at n_time=32, the time axis collapsed to a SINGLE bin
        # before the classifier - pool2 output (16, 1, 1) instead of
        # (16, 1, 4) - so the model could not represent temporal structure at
        # all, and the 64-sample kernel was twice the entire input length.
        #
        # The pools are chosen to leave TARGET_FINAL_TIME bins before the
        # classifier, split as evenly as possible in powers of two; at
        # n_time=128 that split is exactly the original 4 then 8.
        TARGET_FINAL_TIME = 4
        scale = n_time / 128
        if filter_time_length is None:
            filter_time_length = max(1, round(64 * scale))
        if sep_filter_length is None:
            sep_filter_length = max(1, round(16 * scale))
        if pool1_length is None or pool2_length is None:
            total_pool = max(1, n_time // TARGET_FINAL_TIME)
            p1 = 1 << (max(0, total_pool.bit_length() - 1) // 2)
            if pool1_length is None:
                pool1_length = max(1, p1)
            if pool2_length is None:
                pool2_length = max(1, total_pool // pool1_length)

        self.filter_time_length = filter_time_length
        self.sep_filter_length  = sep_filter_length
        self.pool1_length       = pool1_length
        self.pool2_length       = pool2_length

        # ── Block 1: Conv2D ──
        self.conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=F1,
            kernel_size=(1, filter_time_length),
            padding=(0, filter_time_length // 2),
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
        self.pool1  = nn.AvgPool2d(kernel_size=(1, pool1_length))
        self.drop1  = nn.Dropout(p=dropout)

        # ── Block 3: separable convolution ──
        self.sep_conv = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, sep_filter_length),
            padding=(0, sep_filter_length // 2),
            bias=False
        )
        self.bn3   = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, pool2_length))
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
    for n_time in (128, 64, 32):
        model = EEGNet(n_channels=3, n_freq=22, n_time=n_time, n_classes=2)
        x = torch.randn(16, 3, 22, n_time)
        with torch.no_grad():
            feat = model._conv_forward(x)
        out = model(x)
        print(f"n_time={n_time:>3}  kernels={model.filter_time_length}/"
              f"{model.sep_filter_length}  pools={model.pool1_length}/"
              f"{model.pool2_length}  flat={feat.shape[1]:>3}  "
              f"params={model.count_parameters():,}  out={tuple(out.shape)}")
