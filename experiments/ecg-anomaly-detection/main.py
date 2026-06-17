"""
ECG Anomaly Detection — TinyML Experiment

Methodology  : Inter-patient evaluation (de Chazal et al. 2004, IEEE TBME)
               Train on DS1 (22 patients), test on DS2 (22 disjoint patients).
               24 features per beat: 20 morphology/frequency + 4 RR-interval
               timing (the latter added specifically to give the model a
               chance at the S class, which morphology alone often cannot
               distinguish from Normal — see README.md).
Target deploy: STM32 / Nordic nRF52 / ESP32 (Cortex-M class MCUs)

Run
---
    python main.py

First run downloads ~50 MB from PhysioNet (44 records, cached after that).
Results written to results/   Models written to models/
"""

import time
from data_loader import load_interpatient_split, class_distribution
from features    import extract_beat_features
from model       import train_and_evaluate
from visualizer  import generate_report


def main() -> None:
    t0 = time.perf_counter()

    _banner("ECG Anomaly Detection · TinyML Experiment")
    _note("Methodology: de Chazal et al. (2004) inter-patient split")
    _note("Train = DS1 (22 patients) · Test = DS2 (22 disjoint patients)")
    _note("Features: 20 morphology/frequency + 4 RR-interval timing = 24 total")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    _step(1, "Loading MIT-BIH inter-patient split from PhysioNet …")
    _note("First run downloads ~50 MB (44 records) and caches locally")
    beats_train, labels_train, rr_train, beats_test, labels_test, rr_test = load_interpatient_split()

    _note(f"DS1 (train): {len(beats_train):,} beats")
    for cls, n in class_distribution(labels_train).items():
        print(f"           {cls:30s}  {n:6,} samples")
    _note(f"DS2 (test):  {len(beats_test):,} beats")
    for cls, n in class_distribution(labels_test).items():
        print(f"           {cls:30s}  {n:6,} samples")

    # ── 2. Features ───────────────────────────────────────────────────────────
    _step(2, "Extracting features (morphology + RR timing) …")
    X_train, y_train, feature_names = extract_beat_features(beats_train, labels_train, rr_train)
    X_test,  y_test,  _             = extract_beat_features(beats_test,  labels_test,  rr_test)
    _note(f"Train matrix: {X_train.shape[0]:,} × {X_train.shape[1]}")
    _note(f"Test matrix:  {X_test.shape[0]:,} × {X_test.shape[1]}")

    # ── 3. Train ──────────────────────────────────────────────────────────────
    _step(3, "Training Reference, Edge, and Tiny models on DS1 …")
    _note("Evaluating once on DS2 — held-out patients, never seen during training")
    _, results = train_and_evaluate(X_train, y_train, X_test, y_test, feature_names)

    # ── 4. Report ─────────────────────────────────────────────────────────────
    _step(4, "Generating visualisations …")
    generate_report(results)

    elapsed = time.perf_counter() - t0
    _banner(
        f"Complete in {elapsed:.1f}s  |  "
        f"Ref acc {results['reference']['accuracy']:.3f} "
        f"(F1 strict {results['reference']['f1_macro']:.3f} / "
        f"reliable {results['reference']['f1_macro_reliable']:.3f})"
    )
    print("  C models     → results/ecg_edge.c   results/ecg_tiny.c")
    print("  Figures      → results/*.png")
    print("  Full metrics → results/summary.json")
    print("  Per-model text reports → results/classification_report_<name>.txt")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def _step(n: int, msg: str) -> None:
    print(f"\n[{n}/4] {msg}")


def _note(msg: str) -> None:
    print(f"       {msg}")


if __name__ == "__main__":
    main()
