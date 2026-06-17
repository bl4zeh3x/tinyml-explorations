"""Tests for features.py — extract_beat_features with morphology + RR features."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from features import extract_beat_features, FEATURE_NAMES


def test_output_shape(synthetic_beats_labels):
    beats, labels, rr = synthetic_beats_labels
    X, y, names = extract_beat_features(beats, labels, rr)

    assert X.shape == (len(beats), 24)  # 20 morphology + 4 RR
    assert y.shape == (len(beats),)
    assert len(names) == 24
    assert names == FEATURE_NAMES


def test_no_nan_or_inf(synthetic_beats_labels):
    beats, labels, rr = synthetic_beats_labels
    X, _, _ = extract_beat_features(beats, labels, rr)

    assert not np.isnan(X).any(), "Feature matrix contains NaN"
    assert not np.isinf(X).any(), "Feature matrix contains Inf"


def test_labels_preserved(synthetic_beats_labels):
    beats, labels, rr = synthetic_beats_labels
    _, y, _ = extract_beat_features(beats, labels, rr)

    assert len(y) == len(labels)
    assert set(np.unique(y).tolist()) == set(np.unique(labels).tolist())


def test_r_amplitude_is_centre_sample(synthetic_beats_labels):
    """r_amplitude (feature 0) must equal the exact centre sample of the beat."""
    beats, labels, rr = synthetic_beats_labels
    X, _, _ = extract_beat_features(beats, labels, rr)

    centre = beats.shape[1] // 2
    expected_r_amp = beats[:, centre]
    np.testing.assert_allclose(X[:, 0], expected_r_amp, rtol=1e-5)


def test_rr_features_appended_in_correct_position(synthetic_beats_labels):
    """The 4 RR features must land in the last 4 columns, unmodified from
    what was passed in — features.py must not transform them, only append."""
    beats, labels, rr = synthetic_beats_labels
    X, y_out, names = extract_beat_features(beats, labels, rr)

    assert names[-4:] == ["pre_rr_interval", "post_rr_interval", "rr_ratio", "local_rr_mean"]
    # Since no beats should be dropped for this clean synthetic fixture,
    # row order is preserved and the last 4 columns must equal the input rr array.
    np.testing.assert_allclose(X[:, -4:], rr, rtol=1e-5)


def test_deterministic(synthetic_beats_labels):
    beats, labels, rr = synthetic_beats_labels
    X1, y1, _ = extract_beat_features(beats, labels, rr)
    X2, y2, _ = extract_beat_features(beats, labels, rr)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_mismatched_row_counts_raises():
    """beats and rr_features must be row-aligned — a length mismatch is a
    real bug class (e.g. a dropped beat upstream not reflected in both
    arrays) and must fail loudly, not silently misalign."""
    beats = np.zeros((10, 360), dtype=np.float32)
    labels = np.zeros(10, dtype=np.int32)
    rr = np.zeros((9, 4), dtype=np.float32)  # deliberately wrong length

    with pytest.raises(ValueError):
        extract_beat_features(beats, labels, rr)
