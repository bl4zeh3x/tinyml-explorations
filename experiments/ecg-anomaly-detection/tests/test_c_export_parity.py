"""
Parity tests for the m2cgen C export.

These compile the generated C with gcc and verify that predictions from the
compiled binary EXACTLY match sklearn's predictions on the same feature rows.
This is the test that actually matters for this project: a C export that
compiles cleanly but predicts differently than the Python model it came from
is a silent, dangerous bug — exactly the kind of thing that looks fine in a
demo and fails in deployment.

Requires gcc. Skipped automatically if unavailable (e.g. minimal CI images).
"""

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from features import extract_beat_features

GCC_AVAILABLE = shutil.which("gcc") is not None


def _compile_and_run(c_file: Path, feature_rows: np.ndarray, tmp_path: Path) -> list:
    """Compile a small C harness against the exported model and return
    the predicted class index for each row, as computed by the compiled C."""
    rows_literal = ",\n".join(
        "{" + ", ".join(f"{v:.12f}" for v in row) + "}" for row in feature_rows
    )
    harness = textwrap.dedent(f"""
        #include <stdio.h>
        void score(double * input, double * output);
        int argmax(double *a, int n) {{
            int b = 0;
            for (int i = 1; i < n; i++) if (a[i] > a[b]) b = i;
            return b;
        }}
        int main() {{
            double rows[{len(feature_rows)}][{feature_rows.shape[1]}] = {{
                {rows_literal}
            }};
            double output[5];
            for (int r = 0; r < {len(feature_rows)}; r++) {{
                score(rows[r], output);
                printf("%d\\n", argmax(output, 5));
            }}
            return 0;
        }}
    """)
    harness_path = tmp_path / "harness.c"
    harness_path.write_text(harness)

    binary_path = tmp_path / "harness_bin"
    compile_result = subprocess.run(
        ["gcc", "-std=c99", str(harness_path), str(c_file), "-o", str(binary_path), "-lm"],
        capture_output=True, text=True,
    )
    assert compile_result.returncode == 0, f"C compilation failed:\n{compile_result.stderr}"

    run_result = subprocess.run([str(binary_path)], capture_output=True, text=True)
    assert run_result.returncode == 0, f"Compiled binary crashed:\n{run_result.stderr}"

    return [int(line) for line in run_result.stdout.strip().splitlines()]


@pytest.mark.skipif(not GCC_AVAILABLE, reason="gcc not available in this environment")
def test_edge_model_c_export_matches_sklearn(synthetic_beats_labels, tmp_path):
    import joblib
    import model as model_module

    beats, labels = synthetic_beats_labels
    X, y, feature_names = extract_beat_features(beats, labels)
    fitted_models, _ = model_module.train_and_evaluate(
        X, y, feature_names, output_dir=tmp_path
    )

    test_rows = X[:10]
    sklearn_preds = fitted_models["edge"].predict(test_rows).tolist()

    c_file = tmp_path / "results" / "ecg_edge.c"
    c_preds = _compile_and_run(c_file, test_rows, tmp_path)

    assert c_preds == sklearn_preds, (
        f"C/sklearn prediction mismatch for edge model.\n"
        f"sklearn: {sklearn_preds}\nC code:  {c_preds}"
    )


@pytest.mark.skipif(not GCC_AVAILABLE, reason="gcc not available in this environment")
def test_tiny_model_c_export_matches_sklearn(synthetic_beats_labels, tmp_path):
    import model as model_module

    beats, labels = synthetic_beats_labels
    X, y, feature_names = extract_beat_features(beats, labels)
    fitted_models, _ = model_module.train_and_evaluate(
        X, y, feature_names, output_dir=tmp_path
    )

    test_rows = X[:10]
    sklearn_preds = fitted_models["tiny"].predict(test_rows).tolist()

    c_file = tmp_path / "results" / "ecg_tiny.c"
    c_preds = _compile_and_run(c_file, test_rows, tmp_path)

    assert c_preds == sklearn_preds, (
        f"C/sklearn prediction mismatch for tiny model.\n"
        f"sklearn: {sklearn_preds}\nC code:  {c_preds}"
    )
