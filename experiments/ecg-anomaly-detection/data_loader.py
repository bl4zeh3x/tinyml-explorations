"""
MIT-BIH Arrhythmia Database loader — inter-patient protocol + RR-interval timing.

Uses the standard DS1/DS2 split from:
    de Chazal, P., O'Dwyer, M., & Reilly, R. B. (2004). Automatic
    classification of heartbeats using ECG morphology and heartbeat
    interval features. IEEE Transactions on Biomedical Engineering,
    51(7), 1196-1206.

In addition to single-beat morphology, this loader computes RR-interval
timing features (pre-RR, post-RR, their ratio, and a local rolling mean).
This matters because supraventricular ectopic (S) beats are frequently
near-identical to Normal beats in raw morphology — what distinguishes them
clinically is PREMATURITY, a timing fact a single isolated waveform window
cannot express. RR features are computed once per record and explicitly
reset at record boundaries: RR continuity must never bridge two different
patients' recordings.

Records 102, 104, 107, and 217 are excluded (paced-beat patients) — the
standard exclusion throughout the inter-patient literature. A direct
consequence: the Q (unclassifiable) AAMI class, drawn almost entirely from
paced beats, has near-zero representation under this protocol. This is a
documented property of the methodology, not a bug — see README.md.
"""

import wfdb
import numpy as np
from typing import Dict, List, Tuple

# ── The de Chazal et al. (2004) inter-patient split ───────────────────────────

DS1_RECORDS: List[str] = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
]

DS2_RECORDS: List[str] = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
]

PACED_RECORDS_EXCLUDED: List[str] = ["102", "104", "107", "217"]

assert set(DS1_RECORDS) & set(DS2_RECORDS) == set(), \
    "DS1 and DS2 must never share a patient — that is the entire point of an inter-patient split."

# ── AAMI EC57 beat taxonomy ────────────────────────────────────────────────────

_TO_AAMI: Dict[str, str] = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",  # Normal variants
    "A": "S", "a": "S", "J": "S", "S": "S",             # Supraventricular ectopic
    "V": "V", "E": "V",                                  # Ventricular ectopic
    "F": "F",                                            # Fusion
    "/": "Q", "f": "Q", "Q": "Q",                       # Unclassifiable (mostly paced)
}

LABEL_ENCODER: Dict[str, int] = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}
LABEL_DECODER: Dict[int, str] = {
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Ventricular (V)",
    3: "Fusion (F)",
    4: "Unclassifiable (Q)",
}

FS: int = 360      # MIT-BIH sampling frequency (Hz)
WINDOW: int = 180  # Samples each side of R-peak → 1-second total segment
RR_LOCAL_WINDOW: int = 5  # Beats used for the rolling local-RR mean

N_RR_FEATURES: int = 4
RR_FEATURE_NAMES: List[str] = ["pre_rr_interval", "post_rr_interval", "rr_ratio", "local_rr_mean"]


# ── Public API ────────────────────────────────────────────────────────────────

def load_interpatient_split(
    window: int = WINDOW,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the standard inter-patient train/test split, with RR-interval features.

    Returns
    -------
    beats_train, labels_train, rr_train : DS1 — training only
    beats_test,  labels_test,  rr_test  : DS2 — held out, never seen during
                                           training or model selection
    """
    print(f"      Loading DS1 (train, {len(DS1_RECORDS)} records) …")
    beats_train, labels_train, rr_train = _load_records(DS1_RECORDS, window)

    print(f"      Loading DS2 (test,  {len(DS2_RECORDS)} records) …")
    beats_test, labels_test, rr_test = _load_records(DS2_RECORDS, window)

    return beats_train, labels_train, rr_train, beats_test, labels_test, rr_test


def class_distribution(labels: np.ndarray) -> Dict[str, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {LABEL_DECODER[int(u)]: int(c) for u, c in zip(unique, counts)}


# ── Internals ─────────────────────────────────────────────────────────────────

def _load_records(records: List[str], window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    beats_out: List[np.ndarray] = []
    labels_out: List[int] = []
    rr_out: List[np.ndarray] = []

    for rec_id in records:
        try:
            record = wfdb.rdrecord(rec_id, pn_dir="mitdb", channels=[0])
            annot = wfdb.rdann(rec_id, "atr", pn_dir="mitdb")
        except Exception as exc:
            print(f"        ⚠  Skipping {rec_id}: {exc}")
            continue

        signal: np.ndarray = record.p_signal[:, 0].astype(np.float32)

        # Pass 1: collect ONLY genuine beat annotations, in chronological order.
        # Non-beat markers (rhythm-change '+', signal-quality '~', etc.) are
        # already excluded by the _TO_AAMI filter — this is what makes
        # consecutive entries here represent consecutive real heartbeats,
        # which is required for RR-interval arithmetic to be meaningful.
        valid_peaks: List[int] = []
        valid_aami: List[str] = []
        for peak, sym in zip(annot.sample, annot.symbol):
            aami = _TO_AAMI.get(sym)
            if aami is not None:
                valid_peaks.append(int(peak))
                valid_aami.append(aami)

        if len(valid_peaks) < 3:
            continue  # not enough beats in this record to compute RR features

        rr_features = _compute_rr_features(valid_peaks, FS)

        # Pass 2: for each beat with BOTH a valid waveform window AND valid
        # RR features (i.e. not the first/last beat in this record, where
        # pre_rr/post_rr would be undefined), extract the segment.
        for i, (peak, aami) in enumerate(zip(valid_peaks, valid_aami)):
            if rr_features[i] is None:
                continue  # first or last beat in this record

            start, end = peak - window, peak + window
            if start < 0 or end > len(signal):
                continue

            segment = _znorm(signal[start:end].copy())
            beats_out.append(segment)
            labels_out.append(LABEL_ENCODER[aami])
            rr_out.append(rr_features[i])

    if not beats_out:
        raise RuntimeError(
            "No beats were loaded. Confirm internet access — "
            "wfdb fetches from physionet.org on first run."
        )

    return (
        np.array(beats_out, dtype=np.float32),
        np.array(labels_out, dtype=np.int32),
        np.array(rr_out, dtype=np.float32),
    )


def _compute_rr_features(peaks: List[int], fs: int) -> List:
    """
    Compute [pre_rr, post_rr, rr_ratio, local_rr_mean] for each beat in a
    chronologically-ordered, single-record list of R-peak sample indices.

    Returns a list the same length as `peaks`; entries for the first and
    last beat are None (pre/post RR undefined at record boundaries — these
    beats are dropped by the caller, never carried across to another record).
    """
    n = len(peaks)
    out: List = [None] * n
    pre_rr_history: List[float] = []  # for the rolling local mean

    for i in range(1, n - 1):
        pre_rr = (peaks[i] - peaks[i - 1]) / fs
        post_rr = (peaks[i + 1] - peaks[i]) / fs

        pre_rr_history.append(pre_rr)
        window = pre_rr_history[-RR_LOCAL_WINDOW:]
        local_mean = float(np.mean(window))

        rr_ratio = pre_rr / post_rr if post_rr > 1e-6 else 1.0

        out[i] = np.array([pre_rr, post_rr, rr_ratio, local_mean], dtype=np.float32)

    return out


def _znorm(x: np.ndarray) -> np.ndarray:
    mu, sigma = x.mean(), x.std()
    return (x - mu) / (sigma if sigma > 1e-8 else 1.0)
