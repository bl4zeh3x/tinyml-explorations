# ECG Anomaly Detection for Wearable Edge Devices

**Experiment 001 · tinyml-explorations**

A complete pipeline for training and exporting arrhythmia detection models
that run on microcontrollers — no OS, no Python, no ML library required on the device.

---

## What This Is

Standard ECG classification projects train large models on laptops and stop there.
This project treats the MCU deployment as a first-class constraint from the start:

1. Extract 20 interpretable features per beat (80 bytes per inference on device)
2. Train **three** models across a complexity spectrum — Reference, Edge, Tiny
3. Export Edge and Tiny models to **dependency-free C code** via `m2cgen`
4. The generated `.c` files compile with `arm-none-eabi-gcc` for any Cortex-M target
5. A test suite verifies the compiled C produces **identical predictions** to the
   Python model it was exported from — not just that it compiles

When hardware arrives (nRF52840 or STM32L4), the edge model drops in directly.

---

## Dataset

**MIT-BIH Arrhythmia Database** — PhysioNet
48 half-hour ECG recordings, 360 Hz, expert-annotated
Beat taxonomy follows AAMI EC57 standard (5 classes)

| Class | Description               | Count (capped) |
|-------|---------------------------|----------------|
| N     | Normal + Bundle-Branch    | ≤1000          |
| S     | Supraventricular ectopic  | ≤1000          |
| V     | Ventricular ectopic       | ≤1000          |
| F     | Fusion beat               | ≤1000          |
| Q     | Unclassifiable            | ≤1000          |

Data downloads automatically from PhysioNet on first run (~20 MB cached locally).

---

## Features (20 per beat)

Each 360-sample beat window is reduced to 20 scalars before classification.
This is the same computation that will run on the MCU at inference time.

**Amplitude** — R, Q, S peak values; peak-to-peak range; R−Q, R−S differences
**Statistical** — mean, std, energy, skewness, kurtosis, zero-crossing rate
**Frequency** — FFT band powers (0–5 Hz, 5–15 Hz, 15–40 Hz, 40–100 Hz);
dominant frequency; spectral centroid

---

## Models

| Model     | Architecture           | Nodes | Target Hardware        |
|-----------|------------------------|-------|------------------------|
| Reference | RandomForest 200 trees | ~2.5k | Desktop / cloud        |
| Edge      | RandomForest 10 trees  | ~150  | STM32L4 / nRF52840     |
| Tiny      | DecisionTree depth≤6   | ≤127  | ATmega328P (Arduino)   |

Node counts vary slightly per training run because trees use bootstrap
sampling; orders of magnitude shown are stable.

---

## Results

*(run `python main.py` to reproduce — first run ~5 min including PhysioNet download)*

| Model     | CV Accuracy | Macro F1 | C File             |
|-----------|-------------|----------|--------------------|
| Reference | —           | —        | N/A                |
| Edge      | —           | —        | `results/ecg_edge.c` |
| Tiny      | —           | —        | `results/ecg_tiny.c` |

*Populate after first run against real MIT-BIH data — see note below.*

> **On these numbers**: this table intentionally ships empty. Numbers from
> synthetic test data exist in this repo's test suite (`tests/`) purely to
> verify the pipeline's mechanics — they are not scientific results and do
> not belong here. The real accuracy figures come only from a run against
> the actual MIT-BIH database.

---

## Verified Flash Footprint (Cortex-M4)

Compiled with the real ARM GNU toolchain — not estimated:

```
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -std=c99 -O2 -c ecg_tiny.c
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -std=c99 -O2 -c ecg_edge.c
arm-none-eabi-size *.o
```

| Model | `.text` size | Notes |
|-------|-------------|-------|
| Tiny  | 292 bytes   | Single `score()` function, no runtime deps |
| Edge  | 3,740 bytes | 10-tree ensemble, still sub-4KB |

These are the standalone function sizes — a full deployable firmware image
adds startup code, a UART/BLE driver, and a main loop on top, but the model
itself contributes a negligible fraction of any MCU's flash budget.

---

## Testing

```
tests/
├── conftest.py              # Synthetic beat generator (shared fixture)
├── test_features.py         # Shape, NaN/Inf, determinism, R-peak contract
├── test_model.py            # CV correctness, model-size ordering, OOF-vs-in-sample guard
├── test_visualizer.py       # Figure generation, output-path isolation
└── test_c_export_parity.py  # Compiles exported C, asserts predictions == sklearn exactly
```

Run locally:
```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests run against synthetic data, not real MIT-BIH downloads — this keeps
CI fast (~15s) and independent of a third-party server's uptime. The real
experimental run (`python main.py`) is a separate, manual step.

CI runs automatically on every push via GitHub Actions (`.github/workflows/test.yml`).

---

## Setup and Run

```bash
# Clone the parent repo
git clone https://github.com/bl4zeh3x/tinyml-explorations
cd tinyml-explorations/experiments/ecg-anomaly-detection

# Create environment
conda create -n tinyml-ecg python=3.11 -y
conda activate tinyml-ecg
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

---

## File Structure

```
ecg-anomaly-detection/
├── main.py                 # Pipeline orchestrator
├── data_loader.py          # PhysioNet download + beat segmentation
├── features.py             # Feature extraction (runs identically on MCU)
├── model.py                # Training, CV evaluation, C export
├── visualizer.py           # Confusion matrices, feature importance, tradeoff plot
├── tests/                  # pytest suite — see Testing section above
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # + pytest, for running tests/
└── results/                # Generated PNGs + C models (tracked in git;
                             #  models/*.joblib and mitdb/ cache are gitignored)
```

---

## Why This Approach

Most ECG classifiers use CNNs or LSTMs that require hundreds of KB of RAM and
a floating-point unit. This project deliberately targets the other end of the
spectrum: interpretable features + tree-based models whose inference is a
sequence of comparisons — exactly what a Cortex-M0 does well.

The `m2cgen` export produces C that looks like this:

```c
void score(double * input, double * output) {
    double var0[5];
    if (input[19] <= 55.68) {
        if (input[14] <= 0.67) {
            memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
        } else { /* ... */ }
    } else { /* ... */ }
    memcpy(output, var0, 5 * sizeof(double));
}
```

No `malloc`, no floating-point libraries beyond what the MCU's FPU already
provides, no BLAS. The entire model is a sequence of comparisons.

---

## Next Steps

- [ ] Deploy `ecg_edge.c` to nRF52840 BLE development kit (hardware pending)
- [ ] Real-time inference from simulated serial stream
- [ ] Quantise features to INT8 to eliminate FPU requirement
- [ ] Benchmark actual inference latency on Cortex-M4 @ 64 MHz (cycle count, not just code size)

---

*Part of [tinyml-explorations](https://github.com/bl4zeh3x/tinyml-explorations)
— software-side TinyML experiments before hardware arrives.*
