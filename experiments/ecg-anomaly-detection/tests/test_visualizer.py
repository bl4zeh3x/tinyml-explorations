"""Tests for visualizer.py — figure generation under the train/test + RR-feature results format."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from features import extract_beat_features


@pytest.fixture(scope="module")
def results_and_dir(synthetic_train_test, tmp_path_factory):
    import model as model_module

    tmp_dir = tmp_path_factory.mktemp("viz_test_output")
    beats_train, labels_train, rr_train, beats_test, labels_test, rr_test = synthetic_train_test
    X_train, y_train, feature_names = extract_beat_features(beats_train, labels_train, rr_train)
    X_test, y_test, _ = extract_beat_features(beats_test, labels_test, rr_test)

    _, results = model_module.train_and_evaluate(
        X_train, y_train, X_test, y_test, feature_names, output_dir=tmp_dir
    )
    return results, tmp_dir


def test_generate_report_creates_all_figures(results_and_dir):
    from visualizer import generate_report

    results, tmp_dir = results_and_dir
    generate_report(results, output_dir=tmp_dir)

    results_dir = tmp_dir / "results"
    expected = ["confusion_matrices.png", "feature_importance.png", "accuracy_vs_size.png"]
    for fname in expected:
        path = results_dir / fname
        assert path.exists(), f"{fname} was not created"
        assert path.stat().st_size > 1000, f"{fname} exists but looks empty/corrupt"


def test_generate_report_does_not_pollute_real_results_dir(results_and_dir, tmp_path):
    from visualizer import generate_report

    results, _ = results_and_dir
    real_results_dir = Path(__file__).parent.parent / "results"
    before = set(real_results_dir.glob("*.png")) if real_results_dir.exists() else set()

    generate_report(results, output_dir=tmp_path)

    after = set(real_results_dir.glob("*.png")) if real_results_dir.exists() else set()
    assert before == after, "generate_report wrote into the real results/ directory during a test"
