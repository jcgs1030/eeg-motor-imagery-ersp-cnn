"""
tests/test_dataset_and_augmentation.py
--------------------------------------
Tests for the train/val split contract in build_loaders() and for the
augmentation transforms (external review, L-5 / C-3 / L-4).

The build_loaders tests need the generated *-ersp.npz files and are skipped
automatically when the dataset has not been built in this checkout.
"""

import numpy as np
import pytest
import torch

from src.config import DATA_PROC, SUBJECTS, VAL_SPLIT, IMG_FREQ_BINS, IMG_TIME_BINS
from src.p3_classification.cnn.augmentation import Compose, GaussianNoise, TimeShift

AVAILABLE = [s for s in SUBJECTS if (DATA_PROC / f"S{s:02d}T-ersp.npz").exists()]
needs_data = pytest.mark.skipif(
    not AVAILABLE,
    reason="no *-ersp.npz in data/processed - run src/p2_ersp/ersp.py first",
)


# ── build_loaders ────────────────────────────────────────────────────────────

@needs_data
def test_stratified_split_preserves_the_class_ratio():
    """The *-ersp.npz files are written class-sorted (all 'left' then all
    'right'), so an unstratified split over those indices is only balanced by
    luck. stratify=True must make it exact."""
    from src.p3_classification.cnn.dataset import build_loaders

    _, val_loader, _ = build_loaders(subjects=AVAILABLE, stratify=True)
    y_val = np.concatenate([y.numpy() for _, y in val_loader])
    frac_left = (y_val == 0).mean()
    assert abs(frac_left - 0.5) < 0.02, f"val split is {frac_left:.1%} left"


@needs_data
def test_train_and_val_splits_are_disjoint_and_cover_the_training_set():
    from src.p3_classification.cnn.dataset import build_loaders, ERSPDataset

    full = ERSPDataset(subjects=AVAILABLE, suffix="T")
    train_loader, val_loader, _ = build_loaders(subjects=AVAILABLE)

    train_idx = set(train_loader.dataset.indices)
    val_idx = set(val_loader.dataset.indices)

    assert train_idx.isdisjoint(val_idx), "a trial appears in both train and val"
    assert len(train_idx) + len(val_idx) == len(full)
    assert len(val_idx) == int(len(full) * VAL_SPLIT)


@needs_data
def test_test_loader_comes_from_the_evaluation_sessions_only():
    """The T/E protocol boundary: nothing from sessions 4-5 may reach the
    train or val loaders."""
    from src.p3_classification.cnn.dataset import build_loaders, ERSPDataset

    train_loader, val_loader, test_loader = build_loaders(subjects=AVAILABLE)
    n_e = len(ERSPDataset(subjects=AVAILABLE, suffix="E"))
    n_t = len(ERSPDataset(subjects=AVAILABLE, suffix="T"))

    assert len(test_loader.dataset) == n_e
    assert len(train_loader.dataset) + len(val_loader.dataset) == n_t


@needs_data
def test_images_have_the_configured_shape_and_range():
    from src.p3_classification.cnn.dataset import ERSPDataset

    ds = ERSPDataset(subjects=AVAILABLE[:1], suffix="T")
    x, y = ds[0]
    assert x.shape == (3, IMG_FREQ_BINS, IMG_TIME_BINS)
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0
    assert int(y) in (0, 1)


# ── augmentation ─────────────────────────────────────────────────────────────

def _ramp():
    """An image that increases monotonically along time, so any wrap-around
    is detectable as a non-monotonic jump."""
    t = torch.linspace(0.0, 1.0, IMG_TIME_BINS)
    return t.expand(3, IMG_FREQ_BINS, IMG_TIME_BINS).clone()


def test_time_shift_does_not_wrap_around():
    """Regression test for L-4: TimeShift used torch.roll, so the deepest-ERD
    end of the trial reappeared at the start of the image. The vacated edge
    must now be the neutral 0.5, and the surviving part must stay monotonic."""
    x = _ramp()
    shifted = TimeShift(max_shift=6)(x)

    row = shifted[0, 0]
    # the end of the ramp (values near 1.0) must never appear before the start
    assert not (row[0] > row[-1] > 0.0 and row[0] > 0.9), "looks like a circular roll"
    filled = (row == 0.5)
    if filled.any():
        # all filled positions must be contiguous and at one end
        idx = torch.nonzero(filled).flatten()
        assert idx[0] == 0 or idx[-1] == len(row) - 1


@pytest.mark.parametrize("shift", [1, 3, 6, -1, -3, -6])
def test_time_shift_preserves_content_and_fills_with_neutral(shift, monkeypatch):
    x = _ramp()
    monkeypatch.setattr(torch, "randint", lambda *a, **k: torch.tensor([shift]))
    out = TimeShift(max_shift=6)(x)

    n = IMG_TIME_BINS
    if shift > 0:
        torch.testing.assert_close(out[..., shift:], x[..., : n - shift])
        assert torch.all(out[..., :shift] == 0.5)
    else:
        torch.testing.assert_close(out[..., :shift], x[..., -shift:])
        assert torch.all(out[..., shift:] == 0.5)


def test_time_shift_never_touches_the_frequency_axis():
    """Project constraint: the 8-30 Hz frequency axis is never resampled or
    shifted, only the time axis."""
    x = torch.rand(3, IMG_FREQ_BINS, IMG_TIME_BINS)
    out = TimeShift(max_shift=6)(x)
    # a frequency-axis shift would change each time column's set of values
    assert out.shape == x.shape
    for col in range(IMG_TIME_BINS):
        c = out[0, :, col]
        if not torch.all(c == 0.5):
            assert any(torch.allclose(c, x[0, :, j]) for j in range(IMG_TIME_BINS))


def test_gaussian_noise_stays_in_the_unit_range():
    x = torch.rand(3, IMG_FREQ_BINS, IMG_TIME_BINS)
    out = GaussianNoise(std=0.5)(x)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_compose_applies_every_transform():
    x = _ramp()
    out = Compose([GaussianNoise(std=0.01), TimeShift(max_shift=6)])(x)
    assert out.shape == x.shape
    assert not torch.equal(out, x)
