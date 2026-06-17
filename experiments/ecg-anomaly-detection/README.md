# ECG Anomaly Detection for Wearable Edge Devices

**Experiment 001 · tinyml-explorations**

A complete pipeline for training and exporting arrhythmia detection models
that run on microcontrollers — no OS, no Python, no ML library required on the device.

---

## What This Is

1. Extract 24 interpretable features per beat — 20 from single-beat
   morphology/frequency, 4 from R-R interval timing (96 bytes per inference on device)
2. Train **three** models across a complexity spectrum — Reference, Edge, Tiny
3. Export Edge and Tiny models to **dependency-free C code** via `m2cgen`
4. The generated `.c` files compile with `arm-none-eabi-gcc` for any Cortex-M target
5. A test suite verifies the compiled C produces **identical predictions** to the
   Python model it was exported from — not just that it compiles

When hardware arrives (nRF52840 or STM32L4), the edge model drops in directly.

---

## Methodology: Inter-Patient Evaluation

Follows the inter-patient protocol from de Chazal, O'Dwyer & Reilly (2004),
*IEEE Transactions on Biomedical Engineering*, 51(7), 1196-1206 — the
standard comparison baseline across this literature.

```
DS1 (train, 22 patients): 101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
                           122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
                           223, 230
DS2 (test,  22 patients): 100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
                           210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
                           233, 234
```

Models train on DS1 and are evaluated **once** on DS2 — patients never seen
during training. Records 102, 104, 107, and 217 (pacemaker patients) are
excluded from both sets, the standard exclusion in this literature.

### Why RR-interval timing features were added

The first version of this pipeline used 20 single-beat morphology and
frequency features only. On the real MIT-BIH inter-patient split, this
produced accuracy around 0.88 but macro F1 around 0.32 — a result with a
specific, identifiable cause: supraventricular ectopic (S) beats are
frequently near-identical to Normal beats in raw waveform shape. What defines
an S beat clinically is usually **prematurity** — it fires earlier than the
heart's expected rhythm — which is a timing fact a single isolated 1-second
waveform window cannot express.

Four RR-interval features were added: `pre_rr_interval`, `post_rr_interval`,
`rr_ratio`, and `local_rr_mean` (a 5-beat rolling average). These come from
the same R-peak timestamps that beat segmentation already requires, so they
cost nothing extra to obtain at inference time on an MCU, and they're
computationally trivial (subtraction and division) compared to the FFT
already being run for the frequency features.

This is a real, evidence-driven fix, not a fully solved problem: published
inter-patient baselines using far more sophisticated wave-boundary feature
engineering than this project still report SVEB positive predictivity
around 31% — this is a genuinely hard classification problem in this
literature, and the realistic ceiling for a feature-engineered (non-deep-
learning) model on this exact task is well below near-perfect.

### Two macro-F1 numbers, reported transparently

The AAMI "Q" class is drawn almost entirely from paced beats, which the
standard protocol excludes — so Q has near-zero training support under
this split (observed: 7-8 examples total). A macro average that includes
a near-unlearnable class mostly measures noise on that one class, not
overall model quality. This pipeline reports both:

- **`f1_macro`** — strict, every class present in the test set included
- **`f1_macro_reliable`** — restricted to classes with ≥30 training
  examples (excludes Q under the current data; the exact excluded set is
  always printed and saved, never silently assumed)

Per-class precision/recall/F1, with exact train/test sample counts, are
written to `results/summary.json` and `results/classification_report_<model>.txt`
after every run — this is necessary, not optional, since an earlier version
of this pipeline computed these numbers internally but never surfaced them,
which delayed diagnosing the original problem.

---

## Dataset

**MIT-BIH Arrhythmia Database** — PhysioNet, 44 recordings (DS1 + DS2),
360 Hz, expert-annotated. AAMI EC57 beat taxonomy (5 classes).
Downloads automatically on first run (~50 MB, cached locally by `wfdb`).

---

## Features (24 per beat)

**Amplitude** (8) — R, Q, S peak values; peak-to-peak range; R−Q, R−S differences
**Statistical** (6) — mean, std, energy, skewness, kurtosis, zero-crossing rate
**Frequency** (6) — FFT band powers (0–5, 5–15, 15–40, 40–100 Hz); dominant
frequency; spectral centroid
**Timing** (4) — pre-RR interval, post-RR interval, RR ratio, local rolling RR mean

The first 20 are computed from a single isolated waveform window — identical
to what an MCU would compute from one buffered beat. The 4 RR features come
from R-peak timestamps across consecutive beats in the same recording,
reset at every record boundary (RR continuity never bridges two patients).

---

## Models

| Model     | Architecture                      | Target Hardware        |
|-----------|------------------------------------|--------------------------|
| Reference | RandomForest 200 trees             | Desktop / cloud         |
| Edge      | RandomForest 10 trees, depth ≤ 8   | STM32L4 / nRF52840      |
| Tiny      | DecisionTree depth ≤ 6              | ATmega328P (Arduino)    |

Hyperparameters are fixed in advance based on deployment-size constraints,
not tuned against DS2.

---

## Results

*(run `python main.py` to reproduce — first run ~10–15 min including the
44-record PhysioNet download and feature extraction over ~100k beats)*

| Model     | DS2 Accuracy | F1 (strict) | F1 (reliable) | Nodes | C File |
|-----------|-------------:|------------:|---------------:|------:|---------|
| Reference |            — |           — |               — |     — | N/A |
| Edge      |            — |           — |               — |     — | `results/ecg_edge.c` |
| Tiny      |            — |           — |               — |     — | `results/ecg_tiny.c` |

Full per-class breakdown in `results/summary.json` — check it before
treating any single aggregate number as the complete picture.

---

## Verified Flash Footprint (Cortex-M4)

Measured with the real ARM GNU toolchain on the 24-feature models:

```
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -std=c99 -O2 -c ecg_tiny.c
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -std=c99 -O2 -c ecg_edge.c
arm-none-eabi-size *.o
```

| Model | `.text` size | Notes |
|-------|-------------|-------|
| Tiny  | 300 bytes   | Single `score()` function, no runtime deps |
| Edge  | 4,004 bytes | 10-tree ensemble, still sub-4KB |

Standalone function sizes — a full firmware image adds startup code, a
UART/BLE driver, and a main loop on top.

---

## Testing

```
tests/
├── conftest.py              # Synthetic beat + RR-feature generator
├── test_data_loader.py      # DS1/DS2 invariants + hand-verified RR arithmetic
├── test_features.py         # Shape, NaN/Inf, determinism, RR-feature integration
├── test_model.py            # Train/test isolation, per-class reporting, reliable-F1 mechanism
├── test_visualizer.py       # Figure generation, output-path isolation
└── test_c_export_parity.py  # Compiles exported C, asserts predictions == sklearn exactly
```

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Notable tests beyond basic plumbing checks: `test_rr_features_hand_computed_values`
verifies the RR-interval arithmetic against by-hand-calculated expected
values (not just shape checks — this class of timing logic is easy to get
subtly wrong with off-by-one errors at record boundaries), and
`test_thin_class_correctly_excluded_from_reliable_f1` verifies the
reliable-F1 exclusion mechanism directly via persisted `summary.json`
metadata, deterministically (every class's count is explicitly fixed in
this test, not left to a random split — an earlier version of this exact
test relied on a random 50/50 split that occasionally starved an
unintended class by chance, producing a flaky failure unrelated to any
real bug).

Tests run against synthetic data — fast (~7s), no dependency on PhysioNet's
uptime. The real experimental run (`python main.py`) is a separate, manual
step. CI runs automatically on every push via GitHub Actions.

---

## Setup and Run

```bash
git clone https://github.com/bl4zeh3x/tinyml-explorations
cd tinyml-explorations/experiments/ecg-anomaly-detection

conda create -n tinyml-ecg python=3.11 -y
conda activate tinyml-ecg
pip install -r requirements.txt

python main.py
```

---

## File Structure

```
ecg-anomaly-detection/
├── main.py                 # Pipeline orchestrator
├── data_loader.py          # DS1/DS2 split + beat segmentation + RR-interval timing
├── features.py             # Morphology (20) + RR (4) = 24 features per beat
├── model.py                # Train(DS1)/test(DS2), per-class reporting, C export
├── visualizer.py           # Confusion matrices, feature importance, tradeoff plot
├── tests/                  # pytest suite — see Testing section above
├── requirements.txt
├── requirements-dev.txt
└── results/                # Generated PNGs, C models, summary.json (tracked in git;
                             #  models/*.joblib and mitdb/ cache are gitignored)
```

---

## Next Steps

- [ ] Deploy `ecg_edge.c` to nRF52840 BLE development kit (hardware pending)
- [ ] Real-time inference from simulated serial stream
- [ ] Quantise features to INT8 to eliminate FPU requirement
- [ ] Benchmark actual inference latency on Cortex-M4 @ 64 MHz (cycle count)
- [ ] Investigate class-balancing strategies (oversampling DS1 minority
      classes) now that both the split and feature set are evidence-grounded

---

*Part of [tinyml-explorations](https://github.com/bl4zeh3x/tinyml-explorations)
— software-side TinyML experiments before hardware arrives.*
