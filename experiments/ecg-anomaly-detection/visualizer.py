"""
Visualisation module.

Generates three figures saved to results/:
  confusion_matrices.png   — confusion matrices for all three models (side by side)
  feature_importance.png   — top-15 features from the reference model
  accuracy_vs_size.png     — accuracy and F1 as a function of model node count
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any

# AAMI class display names (short form for axis labels)
CLASS_LABELS = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
CLASS_FULL   = {
    0: "Normal", 1: "Supravent.", 2: "Ventricular", 3: "Fusion", 4: "Unknown"
}


def generate_report(results: Dict[str, Any], output_dir: Path = None) -> None:
    """Generate all figures from DS2 held-out test results.

    Parameters
    ----------
    output_dir : Path, optional
        Directory containing (or to create) results/. Defaults to this
        script's own directory. Tests pass an explicit tmp_path so test
        runs never write into the real results/ folder.
    """
    base = Path(output_dir) if output_dir is not None else Path(__file__).parent
    results_dir = base / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    _plot_confusion_matrices(results, results_dir)
    _plot_feature_importance(results["reference"], results_dir)
    _plot_accuracy_vs_size(results, results_dir)
    print(f"    Figures saved to {results_dir}/")


# ── Figures ───────────────────────────────────────────────────────────────────

def _plot_confusion_matrices(results: Dict[str, Any], results_dir: Path) -> None:
    model_order = ["reference", "edge", "tiny"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, name in zip(axes, model_order):
        r   = results[name]
        cm  = r["confusion_matrix"].astype(float)
        labels_present = sorted(np.unique(r["y_true"]).tolist())
        tick_labels = [CLASS_FULL.get(i, str(i)) for i in labels_present]

        # Row-normalise
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums

        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=tick_labels, yticklabels=tick_labels,
            ax=ax, vmin=0, vmax=1, linewidths=0.4,
        )
        ax.set_title(
            f"{name.upper()}\nAcc {r['accuracy']:.3f}  F1 {r['f1_macro']:.3f}",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True" if name == "reference" else "")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        "ECG Arrhythmia Classification — Normalised Confusion Matrices\n"
        "MIT-BIH Database · AAMI EC57 Classes · Inter-Patient Split (DS1 train → DS2 test)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(results_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importance(ref_results: Dict[str, Any], results_dir: Path) -> None:
    importances = ref_results["feature_importances"]
    names       = ref_results["feature_names"]

    if importances.size == 0:
        return

    idx  = np.argsort(importances)[::-1][:15]
    vals = importances[idx]
    lbls = [names[i].replace("_", " ").title() for i in idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(vals)), vals[::-1], color="steelblue", alpha=0.82)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(lbls[::-1], fontsize=9)
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title(
        "Feature Importance — Reference Model (200-tree RF)\n"
        "Top 15 of 20 engineered features",
        fontsize=11, fontweight="bold",
    )
    for i, (bar_val, rect) in enumerate(
        zip(vals[::-1], ax.patches)
    ):
        ax.text(
            bar_val + 0.001, rect.get_y() + rect.get_height() / 2,
            f"{bar_val:.3f}", va="center", fontsize=8,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(results_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_accuracy_vs_size(results: Dict[str, Any], results_dir: Path) -> None:
    order  = ["tiny", "edge", "reference"]
    labels = ["Tiny\n(DT d=6)", "Edge\n(RF 10)", "Reference\n(RF 200)"]
    accs   = [results[k]["accuracy"] for k in order]
    f1s    = [results[k]["f1_macro"] for k in order]
    nodes  = [results[k]["n_nodes"]  for k in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy vs model name
    x = np.arange(len(order))
    width = 0.35
    ax1.bar(x - width / 2, accs, width, label="Accuracy", color="steelblue", alpha=0.85)
    ax1.bar(x + width / 2, f1s,  width, label="Macro F1",  color="darkorange", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score")
    ax1.set_title("Accuracy vs Model Complexity", fontweight="bold")
    ax1.legend()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for xi, (a, f) in enumerate(zip(accs, f1s)):
        ax1.text(xi - width / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax1.text(xi + width / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=8)

    # Node count (log scale)
    ax2.bar(x, nodes, color="teal", alpha=0.80)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_yscale("log")
    ax2.set_ylabel("Total Nodes (log scale)")
    ax2.set_title("Model Size (Node Count)", fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for xi, n in enumerate(nodes):
        ax2.text(xi, n * 1.1, f"{n:,}", ha="center", fontsize=9)

    fig.suptitle(
        "TinyML Tradeoff — ECG Classifier\n"
        "Accuracy vs Deployability on Microcontrollers",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(results_dir / "accuracy_vs_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
