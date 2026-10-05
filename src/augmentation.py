"""
augmentation.py
-----------------
Simple data augmentation transforms for ERSP tensors (3, IMG_FREQ_BINS,
IMG_TIME_BINS), applied only to the training split (never val/test).

Both transforms operate on the time axis / amplitude only - the frequency
axis (and therefore the 8-30Hz mu/beta band selection) is never touched or
resampled, per project constraint.

Usage:
    from augmentation import Compose, GaussianNoise, TimeShift
    transform = Compose([GaussianNoise(std=0.03), TimeShift(max_shift=6)])
"""

import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class GaussianNoise:
    """Additive Gaussian noise in the normalised [0,1] ERSP scale."""

    def __init__(self, std: float = 0.03):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.std
        return torch.clamp(x + noise, 0.0, 1.0)


class TimeShift:
    """Random circular shift along the time axis (last dim) only -
    frequency axis (dim -2) is never touched."""

    def __init__(self, max_shift: int = 6):
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=-1)
