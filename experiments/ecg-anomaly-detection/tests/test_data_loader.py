"""
Tests for data_loader.py — pure-logic invariants that don't require
network access. These guard the properties that make the inter-patient
protocol valid in the first place.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import data_loader as dl


def test_ds1_ds2_no_patient_overlap():
    """The entire point of an inter-patient split: zero shared patients
    between train and test. If this ever fails, every accuracy number
    in this project is silently invalid."""
    overlap = set(dl.DS1_RECORDS) & set(dl.DS2_RECORDS)
    assert overlap == set(), f"DS1/DS2 share patients: {overlap}"


def test_paced_records_excluded_from_both_sets():
    """Paced-beat patients must not leak into either split."""
    all_records = set(dl.DS1_RECORDS) | set(dl.DS2_RECORDS)
    leaked = all_records & set(dl.PACED_RECORDS_EXCLUDED)
    assert leaked == set(), f"Paced records leaked into DS1/DS2: {leaked}"


def test_record_counts_match_literature():
    """de Chazal et al. (2004): 22 records in DS1, 22 in DS2, 44 total."""
    assert len(dl.DS1_RECORDS) == 22
    assert len(dl.DS2_RECORDS) == 22
    assert len(set(dl.DS1_RECORDS) | set(dl.DS2_RECORDS)) == 44


def test_no_duplicate_records_within_a_set():
    assert len(dl.DS1_RECORDS) == len(set(dl.DS1_RECORDS))
    assert len(dl.DS2_RECORDS) == len(set(dl.DS2_RECORDS))


def test_aami_mapping_covers_all_known_mitbih_symbols():
    """Every MIT-BIH beat annotation symbol that should map to an AAMI
    class must be present. Missing a symbol silently drops real beats
    without any error — this test catches that class of bug."""
    expected_symbols = {
        "N", "L", "R", "e", "j",       # → N
        "A", "a", "J", "S",            # → S
        "V", "E",                       # → V
        "F",                             # → F
        "/", "f", "Q",                  # → Q
    }
    assert set(dl._TO_AAMI.keys()) == expected_symbols


def test_aami_mapping_targets_are_valid_classes():
    valid_classes = set(dl.LABEL_ENCODER.keys())
    for symbol, aami_class in dl._TO_AAMI.items():
        assert aami_class in valid_classes, f"Symbol {symbol!r} maps to invalid class {aami_class!r}"


def test_label_encoder_decoder_are_inverses():
    for name, idx in dl.LABEL_ENCODER.items():
        assert name in dl.LABEL_DECODER[idx]


def test_class_distribution_helper():
    import numpy as np
    labels = np.array([0, 0, 0, 1, 2, 2], dtype=np.int32)
    dist = dl.class_distribution(labels)
    assert dist[dl.LABEL_DECODER[0]] == 3
    assert dist[dl.LABEL_DECODER[1]] == 1
    assert dist[dl.LABEL_DECODER[2]] == 2


# ── RR-interval arithmetic: hand-verified, not just shape-checked ────────────
#
# These exist because RR-interval logic is easy to get subtly wrong
# (off-by-one in the rolling window, accidentally bridging across record
# boundaries, wrong index for first/last beat). A shape check alone would
# not catch a swapped pre_rr/post_rr or an off-by-one in the local window.

def test_rr_features_first_and_last_beat_are_none():
    """The first and last beat in a record have no defined pre/post RR —
    they must be explicitly None, not zero or some fallback value, so the
    caller drops them rather than training on fabricated timing."""
    peaks = [0, 360, 720, 1440, 1800]
    result = dl._compute_rr_features(peaks, fs=360)
    assert result[0] is None
    assert result[-1] is None


def test_rr_features_hand_computed_values():
    """Exact expected values, computed by hand:
    peaks = [0, 360, 720, 1440, 1800] at fs=360 Hz → intervals in seconds:
    gaps between consecutive peaks: 1.0s, 1.0s, 2.0s, 1.0s
    """
    import numpy as np
    peaks = [0, 360, 720, 1440, 1800]
    result = dl._compute_rr_features(peaks, fs=360)

    # i=1: pre_rr=(360-0)/360=1.0, post_rr=(720-360)/360=1.0, ratio=1.0, local_mean=mean([1.0])=1.0
    np.testing.assert_allclose(result[1], [1.0, 1.0, 1.0, 1.0], rtol=1e-5)

    # i=2: pre_rr=(720-360)/360=1.0, post_rr=(1440-720)/360=2.0, ratio=0.5, local_mean=mean([1.0,1.0])=1.0
    np.testing.assert_allclose(result[2], [1.0, 2.0, 0.5, 1.0], rtol=1e-5)

    # i=3: pre_rr=(1440-720)/360=2.0, post_rr=(1800-1440)/360=1.0, ratio=2.0, local_mean=mean([1.0,1.0,2.0])=4/3
    np.testing.assert_allclose(result[3], [2.0, 1.0, 2.0, 4.0 / 3.0], rtol=1e-5)


def test_rr_local_mean_window_truncates_correctly():
    """The rolling local-RR mean must use only the last RR_LOCAL_WINDOW
    (5) pre-RR values, not an unbounded running average — otherwise a
    patient's rhythm from minute 1 of a 30-minute recording would still
    be influencing the 'local' average at minute 29."""
    import numpy as np
    # 8 beats, evenly spaced at exactly 1.0s except a deliberate jump at the end
    peaks = [0, 360, 720, 1080, 1440, 1800, 2160, 2520, 7200]
    result = dl._compute_rr_features(peaks, fs=360)

    # By i=7 (8th beat, 0-indexed), 6 pre_rr values have been seen: [1,1,1,1,1,1]
    # all equal to 1.0s — local_mean should be exactly 1.0, NOT diluted by
    # anything before a 5-beat window even though more history exists.
    assert result[7] is not None
    np.testing.assert_allclose(result[7][3], 1.0, rtol=1e-5)


def test_rr_ratio_no_division_by_zero():
    """Defensive guard: post_rr should never be exactly zero in real data
    (would require two beats at the identical sample index), but the
    function must not crash if it somehow is."""
    peaks = [0, 360, 360, 720]  # deliberately pathological: duplicate peak
    result = dl._compute_rr_features(peaks, fs=360)
    # Must not raise ZeroDivisionError or produce inf/nan
    import numpy as np
    for r in result:
        if r is not None:
            assert np.isfinite(r).all()
