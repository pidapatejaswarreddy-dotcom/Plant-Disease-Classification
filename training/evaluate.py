"""Evaluate the trained model on the held-out TEST split and write figures/metrics.

  python training/evaluate.py          (uses models/, results/split_manifest.csv)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CLASS_NAMES_FILENAME, MODEL_FILENAME, MODEL_INFO_FILENAME  # noqa: E402
from src.dataset_utils import build_tf_dataset  # noqa: E402


def evaluate_model(model, test_df: pd.DataFrame, data_root: Path, class_names: list,
                   results_dir: Path, batch_size: int = 32) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 precision_recall_fscore_support)
    idx = {c: i for i, c in enumerate(class_names)}
    y_true = test_df["class_name"].map(idx).to_numpy()
    ds = build_tf_dataset([str(data_root / p) for p in test_df["rel_path"]], y_true, batch_size, training=False)
    probs = model.predict(ds, verbose=0)
    y_pred = probs.argmax(axis=1)
    k = min(3, len(class_names))
    topk = np.argsort(probs, axis=1)[:, -k:]
    labels = list(range(len(class_names)))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    _, _, wf1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).T.to_csv(results_dir / "classification_report.csv")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(results_dir / "confusion_matrix.csv")

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    short = [c.replace("___", ": ").replace("_", " ")[:30] for c in class_names]
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix on the test split (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=110)
    plt.close(fig)

    summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": "test", "num_test_images": int(len(test_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top3_accuracy": float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))])),
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f1),
        "weighted_f1": float(wf1), "per_class": report,
    }
    (results_dir / "evaluation_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_training_curves(history_df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(1, len(history_df) + 1)
    for ax, key, title in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        ax.plot(x, history_df[key], label="train")
        ax.plot(x, history_df["val_" + key], label="validation")
        if "stage" in history_df and history_df["stage"].nunique() > 1:
            first_ft = int(np.argmax(history_df["stage"].to_numpy() == 2)) + 1
            ax.axvline(first_ft - 0.5, color="grey", linestyle="--", label="fine-tuning starts")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    from training.prepare_dataset import resolve_dataset_location
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=str(PROJECT_ROOT / "models"))
    ap.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)
    model_dir, results_dir = Path(args.model_dir), Path(args.results_dir)
    manifest = results_dir / "split_manifest.csv"
    if not (model_dir / MODEL_FILENAME).is_file() or not manifest.is_file():
        print("ERROR: trained model or results/split_manifest.csv not found. Train first (training/train.py).",
              file=sys.stderr)
        return 1
    import keras
    class_names = json.loads((model_dir / CLASS_NAMES_FILENAME).read_text(encoding="utf-8"))
    model = keras.models.load_model(model_dir / MODEL_FILENAME, compile=False)
    df = pd.read_csv(manifest)
    test_df = df[df["split"] == "test"]
    root, _ = resolve_dataset_location(PROJECT_ROOT, args.data_dir, None)
    summary = evaluate_model(model, test_df, root, class_names, results_dir, args.batch_size)
    info_path = model_dir / MODEL_INFO_FILENAME
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["evaluation"] = {k: v for k, v in summary.items() if k != "per_class"}
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Test accuracy {summary['accuracy']:.4f} | top-3 {summary['top3_accuracy']:.4f} | "
          f"macro-F1 {summary['macro_f1']:.4f} on {summary['num_test_images']} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
