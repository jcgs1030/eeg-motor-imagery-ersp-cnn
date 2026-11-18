"""
tests/test_eval_label_alignment.py
----------------------------------
Guards extract_epochs_eval()'s trial/label alignment (external review, L-5).

This is the single most consequential function in the pipeline: the
evaluation sessions' GDF files contain only a generic cue marker (783) with
no class information, so labels come from MOABB and are matched to trials by
POSITION. Getting that mapping wrong silently produces plausible-looking but
meaningless accuracies - exactly the bug recorded at the top of
EXPERIMENTS.md ("Critical Bug Fix - Evaluation Label Correction").

The subtle part is artifact rejection: MNE drops rejected epochs, so the
surviving trials must be re-indexed into the label array through
`epochs.selection`. These tests build a synthetic Raw with a known label
sequence and a deliberately over-threshold trial, so the re-indexing is
actually exercised rather than assumed.
"""

import numpy as np
import mne
import pytest

from src.config import (
    SFREQ, CHANNELS, EVENT_CUE_EVAL, EPOCH_TMIN, EPOCH_TMAX, REJECT_THRESH,
)
from src.p1_preprocessing.preprocessing import extract_epochs_eval

mne.set_log_level("ERROR")

N_TRIALS = 8
TRIAL_SPACING_S = 10.0
FIRST_CUE_S = 5.0


def _make_raw(spike_trial: int | None = None):
    """Synthetic 3-channel Raw with N_TRIALS '783' cue annotations.

    If `spike_trial` is given, that trial gets an amplitude well over
    REJECT_THRESH so MNE will drop it.
    """
    duration = FIRST_CUE_S + N_TRIALS * TRIAL_SPACING_S + 10.0
    n = int(duration * SFREQ)
    rng = np.random.RandomState(0)
    data = rng.randn(len(CHANNELS), n) * 1e-6  # ~1 uV, safely under threshold

    onsets = FIRST_CUE_S + np.arange(N_TRIALS) * TRIAL_SPACING_S

    if spike_trial is not None:
        centre = int(onsets[spike_trial] * SFREQ)
        data[:, centre:centre + 10] = 50 * REJECT_THRESH

    info = mne.create_info([str(c) for c in CHANNELS], SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(mne.Annotations(
        onset=onsets,
        duration=np.zeros(N_TRIALS),
        description=[str(EVENT_CUE_EVAL)] * N_TRIALS,
    ))
    return raw


def test_labels_are_applied_in_trial_order_when_nothing_is_rejected():
    labels = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    epochs = extract_epochs_eval(_make_raw(), labels)

    assert len(epochs) == N_TRIALS
    assert epochs.event_id == {"left": 1, "right": 2}
    # class 0 (left) -> code 1, class 1 (right) -> code 2
    np.testing.assert_array_equal(epochs.events[:, 2], np.where(labels == 0, 1, 2))


def test_rejected_trial_shifts_the_label_mapping_correctly():
    """The regression that matters: if a trial is dropped, every later trial's
    label must still come from its OWN position in the label array, not from
    a naive sequential walk (which would shift all later labels by one)."""
    labels = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    rejected = 3
    epochs = extract_epochs_eval(_make_raw(spike_trial=rejected), labels)

    assert len(epochs) == N_TRIALS - 1, "the over-threshold trial should be dropped"
    assert rejected not in epochs.selection

    expected = labels[epochs.selection]
    np.testing.assert_array_equal(epochs.events[:, 2], np.where(expected == 0, 1, 2))

    # And explicitly: this must NOT equal the naive (shifted) mapping.
    naive = labels[: len(epochs)]
    assert not np.array_equal(expected, naive), \
        "test is not exercising the re-indexing - pick a different rejected trial"


def test_cue_count_mismatch_is_a_hard_error():
    """A subject/session misalignment must fail loudly, never be truncated
    into a silently wrong result."""
    with pytest.raises(AssertionError, match="Cue count mismatch"):
        extract_epochs_eval(_make_raw(), np.array([0, 1, 0]))


def test_epoch_window_matches_the_configured_span():
    epochs = extract_epochs_eval(_make_raw(), np.zeros(N_TRIALS, dtype=int))
    assert epochs.tmin == pytest.approx(EPOCH_TMIN)
    assert epochs.tmax == pytest.approx(EPOCH_TMAX)
