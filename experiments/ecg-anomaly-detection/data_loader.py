"""
MIT-BIH Arrhythmia Database loader.

Downloads ECG records programmatically from PhysioNet via the wfdb package,
segments the continuous signal into individual beats centred on annotated
R-peaks, and maps cardiologist labels to the 5-class AAMI EC57 taxonomy.

On first run, wfdb downloads and caches the selected records under mitdb/.
Subsequent runs read from cache — no network required.

Reference
---------
Moody G.B., Mark R.G. (2001). The impact of the MIT-BIH Arrhythmia Database.
IEEE Engineering in Medicine and Biology Magazine, 20(3), 45–50.
"""

import wfdb
import numpy as np
from typing import List, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────

# 15 records chosen to cover the full rhythm diversity of the MIT-BIH corpus
RECORDS: List[str] = [
    "100", "103", "105", "106", "108",
    "109", "111", "112", "113", "115",
    "116", "117", "118", "119", "121",
]

# MIT-BIH beat symbol → AAMI EC57 class mapping
_TO_AAMI: dict = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",  # Normal variants
    "A": "S", "a": "S", "J": "S", "S": "S",             # Supraventricular ectopic
    "V": "V", "E": "V",                                  # Ventricular ectopic
    "F": "F",                                            # Fusion
    "/": "Q", "f": "Q", "Q": "Q",                       # Unclassifiable
}

LABEL_ENCODER: dict = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}
LABEL_DECODER: dict = {
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Ventricular (V)",
    3: "Fusion (F)",
    4: "Unclassifiable (Q)",
}

FS: int     = 360   # MIT-BIH sampling frequency (Hz)
WINDOW: int = 180   # Samples each side of R-peak → 1-second total segment


# ── Public API ────────────────────────────────────────────────────────────────

def load_mitbih_records(
    records: List[str] = RECORDS,
    window: int = WINDOW,
    max_per_class: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Download and segment ECG beats from MIT-BIH.

    Parameters
    ----------
    records       : MIT-BIH record IDs to include.
    window        : Half-window in samples; each beat has 2*window samples.
    max_per_class : Per-class cap to limit class imbalance without discarding
                    all minority-class examples.

    Returns
    -------
    beats  : float32 array (n_beats, 2*window) — z-score normalised segments
    labels : int32   array (n_beats,)           — AAMI integer class labels
    """
    beats_out:   List[np.ndarray] = []
    labels_out:  List[int]        = []
    class_counts: dict            = {k: 0 for k in LABEL_ENCODER}

    for rec_id in records:
        try:
            record = wfdb.rdrecord(rec_id, pn_dir="mitdb", channels=[0])
            annot  = wfdb.rdann(rec_id, "atr", pn_dir="mitdb")
        except Exception as exc:
            print(f"       ⚠  Skipping {rec_id}: {exc}")
            continue

        signal: np.ndarray = record.p_signal[:, 0].astype(np.float32)

        for peak, sym in zip(annot.sample, annot.symbol):
            aami = _TO_AAMI.get(sym)
            if aami is None:
                continue

            if class_counts[aami] >= max_per_class:
                continue

            start, end = int(peak) - window, int(peak) + window
            if start < 0 or end > len(signal):
                continue

            segment = _znorm(signal[start:end].copy())
            beats_out.append(segment)
            labels_out.append(LABEL_ENCODER[aami])
            class_counts[aami] += 1

    if not beats_out:
        raise RuntimeError(
            "No beats were loaded. "
            "Confirm internet access — wfdb fetches from physionet.org on first run."
        )

    return (
        np.array(beats_out, dtype=np.float32),
        np.array(labels_out, dtype=np.int32),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _znorm(x: np.ndarray) -> np.ndarray:
    mu, sigma = x.mean(), x.std()
    return (x - mu) / (sigma if sigma > 1e-8 else 1.0)
