"""Tests for model.py — training, cross-validation, and C export."""

import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from features import extract_beat_features


@pytest.fixture(scope="module")
def trained_results(synthetic_beats_labels, tmp_path_factory):
    """Train once per test module — CV training is the slow part; reuse it."""
    import model as model_module

    tmp_dir = tmp_path_factory.mktemp("model_test_output")
    beats, labels = synthetic_beats_labels
    X, y, feature_names = extract_beat_features(beats, labels)
    fitted_models, results = model_module.train_and_evaluate(
        X, y, feature_names, output_dir=tmp_dir
    )

    return fitted_models, results, tmp_dir


def test_all_three_models_present(trained_results):
    _, results, _ = trained_results
    assert set(results.keys()) == {"reference", "edge", "tiny"}


def test_accuracy_above_random_baseline(trained_results):
    """5 classes → random baseline ≈ 0.20. Any model below ~0.5 signals a
    real bug (e.g. label misalignment), not just 'hard data'."""
    _, results, _ = trained_results
    for name, r in results.items():
        assert r["accuracy"] > 0.5, f"{name} accuracy {r['accuracy']:.3f} suspiciously low"


def test_model_size_ordering(trained_results):
    """Reference (200 trees) must have far more nodes than Edge (10 trees),
    which must have more than Tiny (1 tree, depth 6) — this is the entire
    point of the accuracy-vs-size tradeoff the project is built around."""
    _, results, _ = trained_results
    assert results["reference"]["n_nodes"] > results["edge"]["n_nodes"]
    assert results["edge"]["n_nodes"] > results["tiny"]["n_nodes"]
    assert results["tiny"]["n_nodes"] <= 127  # max nodes for a depth-6 binary tree


def test_c_files_generated(trained_results):
    _, _, tmp_dir = trained_results
    assert (tmp_dir / "results" / "ecg_edge.c").exists()
    assert (tmp_dir / "results" / "ecg_tiny.c").exists()
    assert not (tmp_dir / "results" / "ecg_reference.c").exists(), \
        "Reference model should NOT be exported — it's a baseline, not a deploy target"


def test_confusion_matrix_is_out_of_fold(trained_results):
    """Regression test for a real bug found during review: the confusion
    matrix must come from cross_val_predict (held-out), not clf.predict(X)
    after fitting on all of X — otherwise it silently shows in-sample
    accuracy while the headline number reports honest CV accuracy."""
    _, results, _ = trained_results
    ref = results["reference"]
    cm_accuracy = np.trace(ref["confusion_matrix"]) / ref["confusion_matrix"].sum()
    # OOF accuracy from the CM must be close to the reported CV accuracy.
    # A bug here would show cm_accuracy ≈ 1.0 regardless of reported accuracy.
    assert abs(cm_accuracy - ref["accuracy"]) < 0.05, (
        f"Confusion matrix accuracy ({cm_accuracy:.3f}) diverges from "
        f"reported CV accuracy ({ref['accuracy']:.3f}) — check for in-sample leakage"
    )
