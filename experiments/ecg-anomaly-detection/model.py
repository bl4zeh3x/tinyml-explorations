"""
Model training, evaluation, and edge export — inter-patient protocol.

Models are trained ONCE on DS1 (train) and evaluated ONCE on DS2 (test).
No cross-validation: the de Chazal inter-patient protocol specifies a
single fixed train/test split by patient. Hyperparameters are fixed in
advance based on deployment-size constraints, not tuned against DS2.

Two macro-F1 numbers are reported, both real, both saved:

  f1_macro            : strict, all classes present in y_test included.
  f1_macro_reliable    : restricted to classes with >= MIN_RELIABLE_SAMPLES
                         training examples. A class with single-digit
                         training support (e.g. Q, drawn almost entirely
                         from excluded paced-beat records) cannot be
                         meaningfully learned, and including it in an
                         aggregate average mostly measures noise, not
                         model quality. Excluding near-empty classes from
                         the headline metric is standard practice in this
                         literature when class presence is under ~1%.

This is reported transparently, not silently substituted: both numbers,
the threshold, and exactly which classes were excluded are always printed
and saved to summary.json.

Three models across a complexity spectrum:
  Reference   RandomForest(200 trees)            — maximum accuracy baseline
  Edge        RandomForest(10 trees, depth ≤ 8)  — target: STM32 / nRF52840
  Tiny        DecisionTree(depth ≤ 6)            — target: ATmega328P (Arduino Uno)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
import joblib
import m2cgen as m2c

_HERE = Path(__file__).parent
MIN_RELIABLE_SAMPLES = 30


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base = Path(output_dir) if output_dir is not None else _HERE
    results_dir = base / "results"
    models_dir = base / "models"
    deploy_dir = (
        (base / "../../deployments/ecg-anomaly-detection").resolve()
        if output_dir is None
        else base / "deployments"
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_support = _class_counts(y_train)
    test_support = _class_counts(y_test)
    all_labels = sorted(set(y_train.tolist()) | set(y_test.tolist()))

    reliable_labels = [lbl for lbl in all_labels if _raw_count(y_train, lbl) >= MIN_RELIABLE_SAMPLES]
    thin_labels = [lbl for lbl in all_labels if lbl not in reliable_labels]

    if thin_labels:
        from data_loader import LABEL_DECODER
        thin_named = {LABEL_DECODER[l]: _raw_count(y_train, l) for l in thin_labels}
        print(f"\n  ⚠  Classes below the {MIN_RELIABLE_SAMPLES}-sample reliability "
              f"threshold in training data: {thin_named}")
        print(f"      These are EXCLUDED from f1_macro_reliable below, but included "
              f"in the strict f1_macro and in the full per-class report.")

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

    fitted_models: Dict[str, Any] = {}
    results: Dict[str, Any] = {}

    for name, clf in model_configs.items():
        print(f"\n  ── {name.upper()} ──")

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        f1_macro_strict = float(f1_score(y_test, y_pred, average="macro",
                                          zero_division=0, labels=all_labels))
        f1_macro_reliable = float(f1_score(y_test, y_pred, average="macro",
                                            zero_division=0, labels=reliable_labels)) \
            if reliable_labels else float("nan")

        per_class_precision = precision_score(y_test, y_pred, average=None,
                                                zero_division=0, labels=all_labels)
        per_class_recall = recall_score(y_test, y_pred, average=None,
                                         zero_division=0, labels=all_labels)
        per_class_f1 = f1_score(y_test, y_pred, average=None,
                                 zero_division=0, labels=all_labels)

        cm = confusion_matrix(y_test, y_pred, labels=all_labels)
        report_text = classification_report(y_test, y_pred, zero_division=0, labels=all_labels)

        print(f"    DS2 accuracy             : {acc:.4f}")
        print(f"    DS2 macro F1 (strict)    : {f1_macro_strict:.4f}  (all {len(all_labels)} classes)")
        print(f"    DS2 macro F1 (reliable)  : {f1_macro_reliable:.4f}  "
              f"({len(reliable_labels)} classes with >= {MIN_RELIABLE_SAMPLES} train samples)")
        print(f"\n{_indent(report_text)}")

        fitted_models[name] = clf
        n_nodes = _count_nodes(clf)

        per_class_table = {
            _class_name(lbl): {
                "train_n": _raw_count(y_train, lbl),
                "test_n": _raw_count(y_test, lbl),
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
            }
            for lbl, p, r, f in zip(all_labels, per_class_precision, per_class_recall, per_class_f1)
        }

        results[name] = {
            "accuracy": acc,
            "f1_macro": f1_macro_strict,
            "f1_macro_reliable": f1_macro_reliable,
            "y_true": y_test,
            "y_pred": y_pred,
            "confusion_matrix": cm,
            "confusion_matrix_labels": all_labels,
            "report": report_text,
            "per_class": per_class_table,
            "train_support": train_support,
            "test_support": test_support,
            "n_nodes": n_nodes,
            "feature_importances": _get_importances(clf),
            "feature_names": feature_names,
        }

        joblib.dump(clf, models_dir / f"{name}.joblib")
        (results_dir / f"classification_report_{name}.txt").write_text(report_text)

        if name in ("edge", "tiny"):
            _export_to_c(clf, name, feature_names, results_dir, deploy_dir)

    print("\n  ── DS2 HELD-OUT TEST RESULTS ──")
    print(f"  {'Model':<12} {'Accuracy':>10} {'F1 strict':>10} {'F1 reliable':>12} {'Nodes':>8}")
    for name in ("reference", "edge", "tiny"):
        r = results[name]
        print(f"  {name:<12} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} "
              f"{r['f1_macro_reliable']:>12.4f} {r['n_nodes']:>8,d}")

    summary = {
        k: {
            "accuracy": round(v["accuracy"], 4),
            "f1_macro": round(v["f1_macro"], 4),
            "f1_macro_reliable": round(v["f1_macro_reliable"], 4) if v["f1_macro_reliable"] == v["f1_macro_reliable"] else None,
            "n_nodes": v["n_nodes"],
            "per_class": v["per_class"],
        }
        for k, v in results.items()
    }
    summary["_metadata"] = {
        "min_reliable_samples_threshold": MIN_RELIABLE_SAMPLES,
        "thin_classes_excluded_from_reliable_f1": [
            _class_name(l) for l in thin_labels
        ],
        "train_support": train_support,
        "test_support": test_support,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return fitted_models, results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _class_name(label: int) -> str:
    from data_loader import LABEL_DECODER
    return LABEL_DECODER[int(label)]


def _class_counts(y: np.ndarray) -> Dict[str, int]:
    unique, counts = np.unique(y, return_counts=True)
    return {_class_name(int(u)): int(c) for u, c in zip(unique, counts)}


def _raw_count(y: np.ndarray, label: int) -> int:
    return int(np.sum(y == label))


def _indent(text: str, prefix: str = "      ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _export_to_c(clf, name, feature_names, results_dir, deploy_dir) -> None:
    c_code = m2c.export_to_c(clf)
    header = (
        f"/* Auto-generated by m2cgen — DO NOT EDIT MANUALLY\n"
        f" *\n"
        f" * Model    : {name}\n"
        f" * Features : {', '.join(feature_names)}\n"
        f" * Trained  : DS1 (de Chazal et al. 2004 inter-patient split)\n"
        f" * Tested   : DS2 (held-out patients, never seen during training)\n"
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
