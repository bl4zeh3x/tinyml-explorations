"""
Model training, evaluation, and edge export.

Three models trained across a complexity spectrum:

  Reference   RandomForest(200 trees)            — maximum accuracy baseline
  Edge        RandomForest(10 trees, depth ≤ 8)  — target: STM32 / nRF52840
  Tiny        DecisionTree(depth ≤ 6)            — target: ATmega328P (Arduino Uno)

Edge and Tiny models are exported to dependency-free C via m2cgen.
The generated .c files compile with any C99 toolchain, including arm-none-eabi-gcc.

References
----------
m2cgen: https://github.com/BayesWitnesses/m2cgen
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import m2cgen as m2c

_HERE = Path(__file__).parent


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cv_folds: int = 5,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train three models, evaluate via stratified CV, export edge models to C.

    Parameters
    ----------
    output_dir : Path, optional
        Root directory for results/ and models/ subfolders, and for the
        deployments/ecg-anomaly-detection/ C export. Defaults to this
        script's own directory (production behaviour). Tests pass an
        explicit tmp_path here so test runs never touch real output files.

    Returns
    -------
    models  : dict  name → fitted sklearn estimator (trained on full data)
    results : dict  name → evaluation metrics and arrays for plotting
    """
    base = Path(output_dir) if output_dir is not None else _HERE
    results_dir = base / "results"
    models_dir = base / "models"
    # Production layout: experiments/ecg-anomaly-detection/ → ../../deployments/...
    # Test layout (output_dir given): keep everything self-contained under base/.
    deploy_dir = (
        (base / "../../deployments/ecg-anomaly-detection").resolve()
        if output_dir is None
        else base / "deployments"
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_configs = {
        "reference": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=seed,
        ),
        "edge": RandomForestClassifier(
            n_estimators=10, max_depth=8, class_weight="balanced", n_jobs=-1, random_state=seed,
        ),
        "tiny": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=seed,
        ),
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fitted_models: Dict[str, Any] = {}
    results: Dict[str, Any] = {}

    for name, clf in model_configs.items():
        print(f"\n  ── {name.upper()} ──")

        # ── Cross-validated metrics ────────────────────────────────────────────
        cv_res   = cross_validate(clf, X, y, cv=cv,
                                  scoring=["accuracy", "f1_macro"], n_jobs=-1)
        acc_mean = float(cv_res["test_accuracy"].mean())
        acc_std  = float(cv_res["test_accuracy"].std())
        f1_mean  = float(cv_res["test_f1_macro"].mean())
        f1_std   = float(cv_res["test_f1_macro"].std())

        print(f"    Accuracy : {acc_mean:.3f} ± {acc_std:.3f}")
        print(f"    Macro F1 : {f1_mean:.3f} ± {f1_std:.3f}")

        # ── Out-of-fold predictions for confusion matrix ───────────────────────
        # Uses the SAME cv splits as cross_validate above (same seed + StratifiedKFold).
        # Honest held-out predictions — NOT in-sample, which would inflate the matrix
        # relative to the reported CV accuracy above.
        y_pred_oof = cross_val_predict(clf, X, y, cv=cv)
        cm         = confusion_matrix(y, y_pred_oof)
        report     = classification_report(y, y_pred_oof, zero_division=0)

        # ── Final fit on full dataset (for deployment) ─────────────────────────
        clf.fit(X, y)
        fitted_models[name] = clf
        n_nodes = _count_nodes(clf)

        results[name] = {
            "accuracy":           acc_mean,
            "accuracy_std":       acc_std,
            "f1_macro":           f1_mean,
            "f1_std":             f1_std,
            "y_true":             y,
            "y_pred":             y_pred_oof,   # OOF — consistent with reported metrics
            "confusion_matrix":   cm,
            "report":             report,
            "n_nodes":            n_nodes,
            "feature_importances": _get_importances(clf),
            "feature_names":      feature_names,
        }

        joblib.dump(clf, models_dir / f"{name}.joblib")

        if name in ("edge", "tiny"):
            _export_to_c(clf, name, feature_names, results_dir, deploy_dir)

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n  ── ACCURACY vs MODEL SIZE ──")
    print(f"  {'Model':<12} {'Accuracy':>10} {'Macro F1':>10} {'Nodes':>8}")
    for name in ("reference", "edge", "tiny"):
        r = results[name]
        print(f"  {name:<12} {r['accuracy']:>10.3f} {r['f1_macro']:>10.3f} {r['n_nodes']:>8,d}")

    summary = {
        k: {"accuracy": round(v["accuracy"], 4),
            "f1_macro":  round(v["f1_macro"],  4),
            "n_nodes":   v["n_nodes"]}
        for k, v in results.items()
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return fitted_models, results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _export_to_c(
    clf, name: str, feature_names: List[str], results_dir: Path, deploy_dir: Path,
) -> None:
    """Export a fitted sklearn model to dependency-free C inference code."""
    c_code = m2c.export_to_c(clf)

    header = (
        f"/* Auto-generated by m2cgen — DO NOT EDIT MANUALLY\n"
        f" *\n"
        f" * Model    : {name}\n"
        f" * Features : {', '.join(feature_names)}\n"
        f" *\n"
        f" * Usage\n"
        f" *   double output[N_CLASSES];\n"
        f" *   score(input_features, output);  // input_features: double[{len(feature_names)}]\n"
        f" *   int predicted = argmax(output, N_CLASSES);\n"
        f" *\n"
        f" * Compile\n"
        f" *   arm-none-eabi-gcc -std=c99 -O2 -c ecg_{name}.c -o ecg_{name}.o\n"
        f" */\n\n"
    )
    full_code = header + c_code

    local_path = results_dir / f"ecg_{name}.c"
    local_path.write_text(full_code, encoding="utf-8")
    print(f"    ✓ C model → {local_path}")

    try:
        deploy_dir.mkdir(parents=True, exist_ok=True)
        deploy_path = deploy_dir / f"ecg_{name}.c"
        deploy_path.write_text(full_code, encoding="utf-8")
        print(f"    ✓ Copied  → {deploy_path}")
    except Exception as exc:
        print(f"    ⚠  Could not write to {deploy_dir}: {exc}")


def _count_nodes(clf) -> int:
    if hasattr(clf, "estimators_"):
        return sum(int(e.tree_.node_count) for e in clf.estimators_)
    if hasattr(clf, "tree_"):
        return int(clf.tree_.node_count)
    return -1


def _get_importances(clf) -> np.ndarray:
    return clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.array([])
