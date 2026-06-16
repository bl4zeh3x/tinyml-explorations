"""
Shared pytest fixtures.

Synthetic beats are used for CI rather than real MIT-BIH data because:
  1. CI must not depend on a third-party server's uptime (physionet.org)
  2. Tests should complete in seconds, not minutes
  3. Class-specific morphology gives the model real (if easy) signal to learn,
     so these tests catch logic errors, not just crashes

This is a deliberate engineering tradeoff: fast, deterministic, network-free
unit tests here; the real scientific run (`python main.py`) is a separate,
manual step that exercises the actual PhysioNet download.
"""

import numpy as np
import pytest

WINDOW = 180
N_PER_CLASS = 60  # small enough for fast CI, large enough for valid 5-fold CV


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


@pytest.fixture(scope="session")
def synthetic_beats_labels():
    """Deterministic synthetic dataset shaped exactly like real loader output."""
    rng = np.random.default_rng(42)
    class_params = {
        0: dict(qrs_amp=3.0, qrs_width=8,  noise_level=0.15),
        1: dict(qrs_amp=2.2, qrs_width=10, noise_level=0.20),
        2: dict(qrs_amp=4.5, qrs_width=18, noise_level=0.25),
        3: dict(qrs_amp=3.5, qrs_width=13, noise_level=0.22),
        4: dict(qrs_amp=1.5, qrs_width=20, noise_level=0.35),
    }
    beats, labels = [], []
    for label, params in class_params.items():
        for _ in range(N_PER_CLASS):
            beats.append(_synthetic_beat(rng, **params))
            labels.append(label)

    beats = np.array(beats, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    idx = rng.permutation(len(beats))
    return beats[idx], labels[idx]
