"""Tests for features.py — extract_beat_features."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from features import extract_beat_features, FEATURE_NAMES


def test_output_shape(synthetic_beats_labels):
    beats, labels = synthetic_beats_labels
    X, y, names = extract_beat_features(beats, labels)

    assert X.shape == (len(beats), 20)
    assert y.shape == (len(beats),)
    assert len(names) == 20
    assert names == FEATURE_NAMES


def test_no_nan_or_inf(synthetic_beats_labels):
    beats, labels = synthetic_beats_labels
    X, _, _ = extract_beat_features(beats, labels)

    assert not np.isnan(X).any(), "Feature matrix contains NaN"
    assert not np.isinf(X).any(), "Feature matrix contains Inf"


def test_labels_preserved(synthetic_beats_labels):
    """Feature extraction must not silently drop or reorder labelled beats
    when input is well-formed (no malformed beats in the synthetic fixture)."""
    beats, labels = synthetic_beats_labels
    _, y, _ = extract_beat_features(beats, labels)

    assert len(y) == len(labels)
    assert set(np.unique(y).tolist()) == set(np.unique(labels).tolist())


def test_r_amplitude_is_centre_sample(synthetic_beats_labels):
    """r_amplitude (feature 0) must equal the exact centre sample of the beat —
    this is the contract the MCU-side feature extractor must also satisfy."""
    beats, labels = synthetic_beats_labels
    X, _, _ = extract_beat_features(beats, labels)

    centre = beats.shape[1] // 2
    expected_r_amp = beats[:, centre]
    np.testing.assert_allclose(X[:, 0], expected_r_amp, rtol=1e-5)


def test_deterministic(synthetic_beats_labels):
    """Same input must always produce the same output — required for the
    Python/C parity guarantee to mean anything."""
    beats, labels = synthetic_beats_labels
    X1, y1, _ = extract_beat_features(beats, labels)
    X2, y2, _ = extract_beat_features(beats, labels)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
