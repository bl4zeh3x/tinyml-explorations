"""Tests for model.py — DS1/DS2 train/test evaluation, per-class reporting, C export."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from features import extract_beat_features


@pytest.fixture(scope="module")
def trained_results(synthetic_train_test, tmp_path_factory):
    import model as model_module

    tmp_dir = tmp_path_factory.mktemp("model_test_output")
    beats_train, labels_train, rr_train, beats_test, labels_test, rr_test = synthetic_train_test

    X_train, y_train, feature_names = extract_beat_features(beats_train, labels_train, rr_train)
    X_test, y_test, _ = extract_beat_features(beats_test, labels_test, rr_test)

    fitted_models, results = model_module.train_and_evaluate(
        X_train, y_train, X_test, y_test, feature_names, output_dir=tmp_dir
    )
    return fitted_models, results, tmp_dir


def test_all_three_models_present(trained_results):
    _, results, _ = trained_results
    assert set(results.keys()) == {"reference", "edge", "tiny"}


def test_accuracy_above_random_baseline(trained_results):
    _, results, _ = trained_results
    for name, r in results.items():
        assert r["accuracy"] > 0.5, f"{name} accuracy {r['accuracy']:.3f} suspiciously low"


def test_model_size_ordering(trained_results):
    _, results, _ = trained_results
    assert results["reference"]["n_nodes"] > results["edge"]["n_nodes"]
    assert results["edge"]["n_nodes"] > results["tiny"]["n_nodes"]
    assert results["tiny"]["n_nodes"] <= 127


def test_c_files_generated(trained_results):
    _, _, tmp_dir = trained_results
    assert (tmp_dir / "results" / "ecg_edge.c").exists()
    assert (tmp_dir / "results" / "ecg_tiny.c").exists()


def test_predictions_match_test_set_size(trained_results):
    _, results, _ = trained_results
    for name, r in results.items():
        assert len(r["y_true"]) == len(r["y_pred"])
        assert len(r["y_true"]) == 30 * 5  # synthetic_train_test: 30/class test


def test_train_and_test_support_reported_separately(trained_results):
    _, results, _ = trained_results
    ref = results["reference"]
    assert "train_support" in ref
    assert "test_support" in ref
    assert sum(ref["train_support"].values()) == 60 * 5
    assert sum(ref["test_support"].values())  == 30 * 5


def test_per_class_breakdown_present_and_complete(trained_results):
    """Regression guard for the exact observability gap found in review:
    per-class precision/recall/F1 must be present for every class, not
    just the aggregate accuracy/macro-F1 numbers."""
    _, results, _ = trained_results
    ref = results["reference"]
    assert "per_class" in ref
    assert len(ref["per_class"]) == 5  # all 5 AAMI classes in this synthetic fixture
    for cls_name, metrics in ref["per_class"].items():
        assert set(metrics.keys()) == {"train_n", "test_n", "precision", "recall", "f1"}


def test_thin_class_correctly_excluded_from_reliable_f1(tmp_path):
    """When one class is deliberately starved below MIN_RELIABLE_SAMPLES,
    it must be excluded from f1_macro_reliable's label set. This tests the
    exclusion MECHANISM directly via the persisted summary.json metadata.

    Every class's training count is explicitly controlled here (rather than
    derived from a random split of a shared fixture) specifically so this
    test is deterministic — an earlier version of this test used a random
    50/50 split, which by chance also pushed an unintended class below the
    threshold, causing a flaky failure that had nothing to do with a real bug.
    """
    import json
    import numpy as np
    from conftest import _synthetic_beat, _synthetic_rr_row, _CLASS_PARAMS
    import model as model_module

    rng = np.random.default_rng(123)

    # Explicit, generous counts for classes 0-3; class 4 deliberately thin.
    train_counts = {0: 80, 1: 80, 2: 80, 3: 80, 4: 5}
    test_counts  = {0: 40, 1: 40, 2: 40, 3: 40, 4: 10}

    def _build(counts):
        beats, labels, rr = [], [], []
        for label, n in counts.items():
            for _ in range(n):
                beats.append(_synthetic_beat(rng, **_CLASS_PARAMS[label]))
                labels.append(label)
                rr.append(_synthetic_rr_row(rng))
        return (np.array(beats, dtype=np.float32),
                np.array(labels, dtype=np.int32),
                np.array(rr, dtype=np.float32))

    beats_train, labels_train, rr_train = _build(train_counts)
    beats_test, labels_test, rr_test = _build(test_counts)

    X_train, y_train, fn = extract_beat_features(beats_train, labels_train, rr_train)
    X_test, y_test, _ = extract_beat_features(beats_test, labels_test, rr_test)

    _, results = model_module.train_and_evaluate(
        X_train, y_train, X_test, y_test, fn, output_dir=tmp_path
    )

    ref = results["reference"]
    assert ref["train_support"].get("Unclassifiable (Q)", 0) == 5
    assert ref["train_support"].get("Normal (N)", 0) == 80  # explicit, not left to chance

    summary = json.loads((tmp_path / "results" / "summary.json").read_text())
    excluded = summary["_metadata"]["thin_classes_excluded_from_reliable_f1"]
    assert excluded == ["Unclassifiable (Q)"], (
        f"Expected only the deliberately-starved class excluded, got: {excluded}"
    )
