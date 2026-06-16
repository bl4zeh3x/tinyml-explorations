"""
ECG Anomaly Detection — TinyML Experiment

Target deployment: STM32 / Nordic nRF52 / ESP32 (Cortex-M class MCUs)
Objective       : Train an arrhythmia classifier that fits within the RAM and
                  flash constraints of a wearable ECG patch, then export it to
                  dependency-free C code ready for cross-compilation.

Run
---
    python main.py

First run downloads ~20 MB from PhysioNet (cached after that).
Results written to results/   Models written to models/
"""

import time
from data_loader import load_mitbih_records, LABEL_DECODER
from features    import extract_beat_features
from model       import train_and_evaluate
from visualizer  import generate_report


def main() -> None:
    t0 = time.perf_counter()

    _banner("ECG Anomaly Detection · TinyML Experiment")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    _step(1, "Loading MIT-BIH Arrhythmia Database from PhysioNet …")
    _note("First run downloads ~20 MB and caches locally in mitdb/")
    beats, labels = load_mitbih_records()

    unique, counts = _class_distribution(labels)
    _note(f"{len(beats):,} beats across {len(unique)} AAMI classes:")
    for cls, n in zip(unique, counts):
        print(f"           {LABEL_DECODER[cls]:30s}  {n:5,} samples")

    # ── 2. Features ───────────────────────────────────────────────────────────
    _step(2, "Extracting features …")
    X, y, feature_names = extract_beat_features(beats, labels)
    _note(f"Feature matrix: {X.shape[0]:,} × {X.shape[1]} ({X.nbytes / 1024:.1f} KB)")
    _note("Each beat: 360 raw samples → 20 scalar features → 80 bytes on MCU")

    # ── 3. Train ──────────────────────────────────────────────────────────────
    _step(3, "Training Reference, Edge, and Tiny models (5-fold CV) …")
    _note("Edge model targets Cortex-M (≤256 KB flash), Tiny targets ATmega328P")
    _, results = train_and_evaluate(X, y, feature_names)

    # ── 4. Report ─────────────────────────────────────────────────────────────
    _step(4, "Generating visualisations …")
    generate_report(results)

    elapsed = time.perf_counter() - t0
    _banner(
        f"Complete in {elapsed:.1f}s  |  "
        f"Ref acc {results['reference']['accuracy']:.3f}  |  "
        f"Edge acc {results['edge']['accuracy']:.3f}  |  "
        f"Tiny acc {results['tiny']['accuracy']:.3f}"
    )
    print("  C models → results/ecg_edge.c   results/ecg_tiny.c")
    print("  Figures  → results/*.png")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def _step(n: int, msg: str) -> None:
    print(f"\n[{n}/4] {msg}")


def _note(msg: str) -> None:
    print(f"       {msg}")


def _class_distribution(labels):
    import numpy as np
    unique, counts = np.unique(labels, return_counts=True)
    return unique.tolist(), counts.tolist()


if __name__ == "__main__":
    main()
