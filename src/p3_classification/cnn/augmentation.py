"""
augmentation.py
-----------------
Simple data augmentation transforms for ERSP tensors (3, IMG_FREQ_BINS,
IMG_TIME_BINS), applied only to the training split (never val/test).

Both transforms operate on the time axis / amplitude only - the frequency
axis (and therefore the 8-30Hz mu/beta band selection) is never touched or
resampled, per project constraint.

Usage:
    from src.p3_classification.cnn.augmentation import Compose, GaussianNoise, TimeShift
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
    """Random shift along the time axis (last dim) only - frequency axis
    (dim -2) is never touched.

    The vacated edge is filled with `fill` instead of wrapping around
    (external review, L-4). `torch.roll` is a CIRCULAR shift: the end of
    the imagery period (where ERD is deepest) would reappear at the start
    of the image, a discontinuity that cannot occur in a real trial. At
    max_shift=6 over IMG_TIME_BINS=32 that wrapped band is up to ~19% of
    the image. fill=0.5 is the neutral value on the [0,1] ERSP scale
    (0 dB, i.e. no spectral change vs. baseline), so the shifted-in region
    reads as "no perturbation" rather than as inverted dynamics.
    """

    def __init__(self, max_shift: int = 6, fill: float = 0.5):
        self.max_shift = max_shift
        self.fill = fill

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
        if shift == 0:
            return x
        out = torch.full_like(x, self.fill)
        if shift > 0:
            # shift later in time: [0:n-shift] lands at [shift:n]
            out[..., shift:] = x[..., :-shift]
        else:
            out[..., :shift] = x[..., -shift:]
        return out
