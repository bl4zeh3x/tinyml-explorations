"""
Shared pytest fixtures.

Synthetic beats are used for CI rather than real MIT-BIH data because:
  1. CI must not depend on a third-party server's uptime (physionet.org)
  2. Tests should complete in seconds, not minutes
  3. Class-specific morphology gives the model real (if easy) signal to learn,
     so these tests catch logic errors, not just crashes

Synthetic RR-interval features are generated alongside beats with plausible
physiological values (sinus rhythm ~0.6-1.0s intervals). Exact realism
doesn't matter for these fixtures — they exist to validate pipeline
plumbing (shapes, dtypes, no NaN), not to validate RR features' real-world
predictive value, which only a run against actual MIT-BIH data can show.
"""

import numpy as np
import pytest

WINDOW = 180
N_RR_FEATURES = 4

_CLASS_PARAMS = {
    0: dict(qrs_amp=3.0, qrs_width=8,  noise_level=0.15),
    1: dict(qrs_amp=2.2, qrs_width=10, noise_level=0.20),
    2: dict(qrs_amp=4.5, qrs_width=18, noise_level=0.25),
    3: dict(qrs_amp=3.5, qrs_width=13, noise_level=0.22),
    4: dict(qrs_amp=1.5, qrs_width=20, noise_level=0.35),
}


def _synthetic_beat(rng, qrs_amp, qrs_width, noise_level):
    n = 2 * WINDOW
    t = np.arange(n)
    centre = WINDOW
    qrs = qrs_amp * np.exp(-((t - centre) ** 2) / (2 * qrs_width ** 2))
    baseline = 0.05 * np.sin(2 * np.pi * t / n)
    noise = rng.normal(0, noise_level, n)
    beat = qrs + baseline + noise
    mu, sigma = beat.mean(), beat.std()
    return ((beat - mu) / (sigma if sigma > 1e-8 else 1.0)).astype(np.float32)


def _synthetic_rr_row(rng):
    """Plausible RR-interval feature row: pre_rr, post_rr, ratio, local_mean."""
    pre_rr = float(rng.uniform(0.6, 1.0))
    post_rr = float(rng.uniform(0.6, 1.0))
    ratio = pre_rr / post_rr
    local_mean = float(rng.uniform(0.6, 1.0))
    return np.array([pre_rr, post_rr, ratio, local_mean], dtype=np.float32)


def _make_dataset(rng, n_per_class):
    beats, labels, rr = [], [], []
    for label, params in _CLASS_PARAMS.items():
        for _ in range(n_per_class):
            beats.append(_synthetic_beat(rng, **params))
            labels.append(label)
            rr.append(_synthetic_rr_row(rng))
    beats = np.array(beats, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    rr = np.array(rr, dtype=np.float32)
    idx = rng.permutation(len(beats))
    return beats[idx], labels[idx], rr[idx]


@pytest.fixture(scope="session")
def synthetic_beats_labels():
    """Single pooled dataset (beats, labels, rr_features) — for testing
    feature extraction in isolation, where train/test distinction is
    irrelevant."""
    rng = np.random.default_rng(42)
    return _make_dataset(rng, n_per_class=60)


@pytest.fixture(scope="session")
def synthetic_train_test():
    """Two independently-drawn datasets simulating DS1 (train) / DS2 (test).
    Returns (beats_train, labels_train, rr_train, beats_test, labels_test, rr_test)."""
    rng_train = np.random.default_rng(42)
    rng_test  = np.random.default_rng(99)
    beats_train, labels_train, rr_train = _make_dataset(rng_train, n_per_class=60)
    beats_test,  labels_test,  rr_test  = _make_dataset(rng_test,  n_per_class=30)
    return beats_train, labels_train, rr_train, beats_test, labels_test, rr_test
