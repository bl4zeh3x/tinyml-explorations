"""
ECG beat feature extraction.

Twenty interpretable features per beat, grouped into three families:

  Amplitude   : R, Q, S peak values; peak-to-peak range; inter-peak ratios
  Statistical : mean, std, energy, skewness, kurtosis, zero-crossing rate
  Frequency   : FFT band powers (0-5, 5-15, 15-40, 40-100 Hz),
                dominant frequency, spectral centroid

Design rationale
----------------
Raw waveforms are 360 samples × float32 ≈ 1.4 KB per beat.
These 20 scalar features reduce that to 80 bytes — small enough to compute
on a microcontroller at inference time from a new beat.  Features are chosen
to align with clinically documented discriminators of arrhythmia types.
"""

import numpy as np
from typing import List, Tuple

FS: int = 360  # MIT-BIH sampling frequency (Hz)

FEATURE_NAMES: List[str] = [
    # Amplitude
    "r_amplitude", "global_max", "global_min", "peak_to_peak",
    "q_amplitude", "s_amplitude", "r_minus_q", "r_minus_s",
    # Statistical
    "mean", "std", "energy", "skewness", "kurtosis", "zero_crossing_rate",
    # Frequency
    "band_power_0_5hz", "band_power_5_15hz", "band_power_15_40hz", "band_power_40_100hz",
    "dominant_frequency", "spectral_centroid",
]


def extract_beat_features(
    beats: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract feature vectors from normalised ECG beat segments.

    Parameters
    ----------
    beats  : ndarray, shape (n, 2*window)  — z-score normalised beat windows
    labels : ndarray, shape (n,)            — integer AAMI class labels

    Returns
    -------
    X            : ndarray, shape (n_valid, 20)
    y            : ndarray, shape (n_valid,)
    feature_names: list of 20 feature name strings
    """
    raw = np.array([_single_beat(b) for b in beats], dtype=np.float32)

    valid = ~np.any(np.isnan(raw) | np.isinf(raw), axis=1)
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"      ⚠  Dropped {n_dropped} beats with invalid features")

    return raw[valid], labels[valid], FEATURE_NAMES


# ── Private ───────────────────────────────────────────────────────────────────

def _single_beat(beat: np.ndarray) -> np.ndarray:
    n = len(beat)
    centre = n // 2  # R-peak position (data_loader centres on R-peak)

    # Neighbourhoods for Q and S waves (±40 samples ≈ 111 ms each side)
    q_window = beat[max(0, centre - 40): centre]
    s_window = beat[centre: min(n, centre + 40)]

    # ── Amplitude ─────────────────────────────────────────────────────────────
    r_amp     = float(beat[centre])
    g_max     = float(beat.max())
    g_min     = float(beat.min())
    ptp       = g_max - g_min
    q_amp     = float(q_window.min()) if q_window.size else 0.0
    s_amp     = float(s_window.min()) if s_window.size else 0.0
    r_minus_q = r_amp - q_amp
    r_minus_s = r_amp - s_amp

    # ── Statistical ───────────────────────────────────────────────────────────
    mu       = float(beat.mean())
    sigma    = float(beat.std())
    energy   = float(np.dot(beat, beat))
    centred  = beat - mu
    var      = float((centred ** 2).mean())
    skew     = float((centred ** 3).mean()) / (var ** 1.5 + 1e-9)
    kurt     = float((centred ** 4).mean()) / (var ** 2   + 1e-9)
    zcr      = float(np.diff(np.sign(beat)).astype(bool).sum())

    # ── Frequency (FFT) ───────────────────────────────────────────────────────
    mag   = np.abs(np.fft.rfft(beat))
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    total = float(np.dot(mag, mag)) + 1e-9

    def _band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.dot(mag[mask], mag[mask])) / total

    bp_0_5   = _band(0,   5)
    bp_5_15  = _band(5,  15)
    bp_15_40 = _band(15, 40)
    bp_40_100 = _band(40, 100)

    # Skip DC bin (index 0) so dominant_frequency is always > 0
    dom_freq  = float(freqs[np.argmax(mag[1:]) + 1]) if mag.size > 1 else 0.0
    sp_cent   = float(np.dot(freqs, mag) / (mag.sum() + 1e-9))

    return np.array([
        r_amp, g_max, g_min, ptp,
        q_amp, s_amp, r_minus_q, r_minus_s,
        mu, sigma, energy, skew, kurt, zcr,
        bp_0_5, bp_5_15, bp_15_40, bp_40_100,
        dom_freq, sp_cent,
    ], dtype=np.float32)
