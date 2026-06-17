"""
ECG beat feature extraction.

24 features per beat, in four families:

  Amplitude   : R, Q, S peak values; peak-to-peak range; inter-peak ratios   (8)
  Statistical : mean, std, energy, skewness, kurtosis, zero-crossing rate    (6)
  Frequency   : FFT band powers (0-5, 5-15, 15-40, 40-100 Hz),
                dominant frequency, spectral centroid                       (6)
  Timing (RR) : pre-RR, post-RR, RR ratio, local rolling RR mean             (4)

Design rationale
-----------------
The first 20 features are computed from a single isolated 1-second waveform
window — the same computation an MCU would run from a single buffered beat.
The 4 RR-interval features come from a different physical signal: the
timestamps of consecutive R-peaks, supplied by data_loader.py from a single
record's chronological beat sequence. They are included because raw
morphology alone frequently cannot distinguish Supraventricular (S) beats
from Normal beats — what defines an S beat is often its PREMATURITY, a
timing fact rather than a shape fact. See README.md for the literature
basis. On an MCU, these come for free from the same R-peak detector that
already produces the window-extraction trigger.
"""

import numpy as np
from typing import List, Tuple

FS: int = 360  # MIT-BIH sampling frequency (Hz)

_MORPHOLOGY_FEATURE_NAMES: List[str] = [
    # Amplitude
    "r_amplitude", "global_max", "global_min", "peak_to_peak",
    "q_amplitude", "s_amplitude", "r_minus_q", "r_minus_s",
    # Statistical
    "mean", "std", "energy", "skewness", "kurtosis", "zero_crossing_rate",
    # Frequency
    "band_power_0_5hz", "band_power_5_15hz", "band_power_15_40hz", "band_power_40_100hz",
    "dominant_frequency", "spectral_centroid",
]

_RR_FEATURE_NAMES: List[str] = ["pre_rr_interval", "post_rr_interval", "rr_ratio", "local_rr_mean"]

FEATURE_NAMES: List[str] = _MORPHOLOGY_FEATURE_NAMES + _RR_FEATURE_NAMES


def extract_beat_features(
    beats: np.ndarray,
    labels: np.ndarray,
    rr_features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract feature vectors from normalised ECG beat segments + RR timing.

    Parameters
    ----------
    beats       : ndarray, shape (n, 2*window)  — z-score normalised beat windows
    labels      : ndarray, shape (n,)             — integer AAMI class labels
    rr_features : ndarray, shape (n, 4)           — [pre_rr, post_rr, ratio, local_mean]
                  per beat, from data_loader.py. Must be row-aligned with `beats`.

    Returns
    -------
    X            : ndarray, shape (n_valid, 24)
    y            : ndarray, shape (n_valid,)
    feature_names: list of 24 feature name strings
    """
    if len(beats) != len(rr_features):
        raise ValueError(
            f"beats ({len(beats)}) and rr_features ({len(rr_features)}) "
            f"must be row-aligned — they describe the same beats."
        )

    morphology = np.array([_single_beat(b) for b in beats], dtype=np.float32)
    combined = np.concatenate([morphology, rr_features.astype(np.float32)], axis=1)

    valid = ~np.any(np.isnan(combined) | np.isinf(combined), axis=1)
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"      ⚠  Dropped {n_dropped} beats with invalid features")

    return combined[valid], labels[valid], FEATURE_NAMES


# ── Private: single-beat morphology (unchanged from original 20-feature set) ──

def _single_beat(beat: np.ndarray) -> np.ndarray:
    n = len(beat)
    centre = n // 2

    q_window = beat[max(0, centre - 40): centre]
    s_window = beat[centre: min(n, centre + 40)]

    r_amp     = float(beat[centre])
    g_max     = float(beat.max())
    g_min     = float(beat.min())
    ptp       = g_max - g_min
    q_amp     = float(q_window.min()) if q_window.size else 0.0
    s_amp     = float(s_window.min()) if s_window.size else 0.0
    r_minus_q = r_amp - q_amp
    r_minus_s = r_amp - s_amp

    mu       = float(beat.mean())
    sigma    = float(beat.std())
    energy   = float(np.dot(beat, beat))
    centred  = beat - mu
    var      = float((centred ** 2).mean())
    skew     = float((centred ** 3).mean()) / (var ** 1.5 + 1e-9)
    kurt     = float((centred ** 4).mean()) / (var ** 2   + 1e-9)
    zcr      = float(np.diff(np.sign(beat)).astype(bool).sum())

    mag   = np.abs(np.fft.rfft(beat))
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    total = float(np.dot(mag, mag)) + 1e-9

    def _band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.dot(mag[mask], mag[mask])) / total

    bp_0_5    = _band(0,   5)
    bp_5_15   = _band(5,  15)
    bp_15_40  = _band(15, 40)
    bp_40_100 = _band(40, 100)

    dom_freq = float(freqs[np.argmax(mag[1:]) + 1]) if mag.size > 1 else 0.0
    sp_cent  = float(np.dot(freqs, mag) / (mag.sum() + 1e-9))

    return np.array([
        r_amp, g_max, g_min, ptp,
        q_amp, s_amp, r_minus_q, r_minus_s,
        mu, sigma, energy, skew, kurt, zcr,
        bp_0_5, bp_5_15, bp_15_40, bp_40_100,
        dom_freq, sp_cent,
    ], dtype=np.float32)
