"""
tests/test_ersp.py
------------------
Property tests for compute_ersp_image(), the function every CNN experiment
in the project depends on (external review, L-5).

These are pure-function tests on synthetic signals: they need no GDF file,
no .npz and no network access.
"""

import numpy as np
import pytest

from src.config import (
    IMG_FREQ_BINS, IMG_TIME_BINS, IMG_SIZE, SFREQ,
    EPOCH_TMIN, EPOCH_TMAX, BASELINE, ANALYSIS_TMIN, STFT_WIN_LEN,
)
from src.p2_ersp.ersp import compute_ersp_image, _to_db, DB_CLIP

EPOCH_LEN = int(round((EPOCH_TMAX - EPOCH_TMIN) * SFREQ)) + 1


def _t():
    """Time axis of one synthetic epoch, in seconds relative to cue onset."""
    return np.arange(EPOCH_LEN) / SFREQ + EPOCH_TMIN


def _sine(freq=10.0, amp=1e-5, envelope=None):
    """A mu-band sine over the full epoch, optionally amplitude-modulated."""
    t = _t()
    sig = amp * np.sin(2 * np.pi * freq * t)
    if envelope is not None:
        sig = sig * envelope(t)
    return sig.astype(np.float64)


def test_output_shape_matches_img_size():
    img = compute_ersp_image(_sine(), sfreq=SFREQ)
    assert img.shape == IMG_SIZE == (IMG_FREQ_BINS, IMG_TIME_BINS)


def test_output_is_float32_and_in_unit_range():
    img = compute_ersp_image(_sine(), sfreq=SFREQ)
    assert img.dtype == np.float32
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_stationary_signal_maps_to_the_neutral_value():
    """A signal whose power never changes has ERSP = 0 dB everywhere, which
    the [0,1] normalisation must place at exactly 0.5 (the 'no perturbation'
    value the CNN sees). This is the property that makes ERD (<0.5) and ERS
    (>0.5) comparable across trials."""
    img = compute_ersp_image(_sine(freq=12.0), sfreq=SFREQ)
    # The last ~4 of the 32 time bins are excluded: ANALYSIS_TMAX == EPOCH_TMAX,
    # so unlike the baseline side (which got a 2 s buffer via EPOCH_TMIN=-3.0)
    # the end of the analysis window sits right on the extracted trial's edge
    # and picks up the STFT's zero-padding taper. On a noiseless synthetic sine
    # that taper drives power to ~0 and saturates those columns at the -6 dB
    # clip; on real EEG the effect is much milder (42% vs ~35% of pixels
    # saturated in the last column, measured over all 9 subjects' T sessions).
    interior = img[:, 1:-4]
    assert np.allclose(interior, 0.5, atol=0.07), \
        f"stationary signal should sit at 0.5, got [{interior.min():.3f}, {interior.max():.3f}]"


def test_power_drop_after_cue_reads_as_erd():
    """Suppressing mu power after the cue must push the image BELOW 0.5
    (ERD), not above it - i.e. the sign convention is not inverted."""
    def envelope(t):
        return np.where(t < 0.0, 1.0, 0.2)   # 14 dB drop at cue onset

    img = compute_ersp_image(_sine(freq=10.0, envelope=envelope), sfreq=SFREQ)
    # Column index of a well-post-cue time point (~2 s after the cue).
    frac = (2.0 - ANALYSIS_TMIN) / (EPOCH_TMAX - ANALYSIS_TMIN)
    col = int(frac * IMG_TIME_BINS)
    assert img[:, col].mean() < 0.5, "post-cue power drop should read as ERD (< 0.5)"


def test_baseline_window_has_a_full_stft_window_of_buffer():
    """compute_ersp_image() measures its reference power over BASELINE. If
    BASELINE fell outside EPOCH_TMIN..EPOCH_TMAX the function would silently
    fall back to a single STFT frame; if it sat within one STFT window of the
    trial's leading edge its frames would be tainted by the zero-padding
    taper. That was the bug documented in EXPERIMENTS.md ("Baseline Window
    Contamination") and fixed by pulling EPOCH_TMIN back to -3.0 s - this
    pins the invariant so it cannot silently regress."""
    assert EPOCH_TMIN < BASELINE[0] < BASELINE[1] <= EPOCH_TMAX
    buffer_s = BASELINE[0] - EPOCH_TMIN
    assert buffer_s >= STFT_WIN_LEN / SFREQ, (
        f"baseline starts {buffer_s:.3f}s after the trial edge but the STFT "
        f"window is {STFT_WIN_LEN / SFREQ:.3f}s long"
    )


@pytest.mark.parametrize("n_time", [16, 32, 64, 128])
def test_requested_resolution_is_honoured(n_time):
    img = compute_ersp_image(_sine(), sfreq=SFREQ, n_time_bins=n_time)
    assert img.shape == (IMG_FREQ_BINS, n_time)


def test_sigma_zero_is_identical_to_the_default_path():
    """sigma=0 must short-circuit the Gaussian filter and stay byte-identical
    to the pre-sigma behaviour - the production dataset depends on it."""
    sig = _sine()
    assert np.array_equal(
        compute_ersp_image(sig, sfreq=SFREQ),
        compute_ersp_image(sig, sfreq=SFREQ, sigma=0.0),
    )


def test_to_db_inverts_the_normalisation():
    """_to_db() is used by every plot; it must be the exact inverse of the
    [0,1] scaling baked into compute_ersp_image()."""
    norm = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    db = _to_db(norm)
    assert np.allclose(db, [-DB_CLIP, -DB_CLIP / 2, 0.0, DB_CLIP / 2, DB_CLIP])


def test_db_clip_is_defined_once():
    """Regression test for the shadowing bug (external review, L-6): the
    local DB_CLIP redefinitions inside compute_ersp_image() /
    compute_ersp_image_wavelet() must stay removed, so changing the
    module-level constant really does change the normalisation."""
    import inspect
    from src.p2_ersp import ersp as ersp_mod

    for fn in (ersp_mod.compute_ersp_image, ersp_mod.compute_ersp_image_wavelet):
        src = inspect.getsource(fn)
        assert "DB_CLIP = " not in src, \
            f"{fn.__name__} redefines DB_CLIP locally, shadowing the module constant"
