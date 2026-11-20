"""Train the MobileNetV2 plant-disease classifier (identical code path locally and on Colab).

  python training/train.py --epochs 10 --fine-tune-epochs 5

Outputs (written to --model-dir only after training AND evaluation finished successfully):
  plant_disease_model.keras, class_names.json, model_info.json
and to --results-dir: split_manifest.csv, training_history.csv/json, training_curves.png,
evaluation_results.json, classification_report.csv, confusion_matrix.csv/png.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import (ARCHITECTURE_ID, CLASS_NAMES_FILENAME, DATASET_NAME, DEFAULT_SEED,  # noqa: E402
                           IMG_SIZE, INFO_SCHEMA_VERSION, MODEL_FILENAME, MODEL_INFO_FILENAME,
                           PREPROCESSING_ID, SPLIT_FRACTIONS)
from src.dataset_utils import (DatasetError, assign_groups, build_tf_dataset,  # noqa: E402
                               compute_class_weights, load_leaf_map, make_splits,
                               subsample_per_class, verify_dataset)
from src.model_builder import (build_model, compile_model, freeze_base,  # noqa: E402
                               unfreeze_top_layers)
from training.evaluate import evaluate_model, plot_training_curves  # noqa: E402
from training.prepare_dataset import resolve_dataset_location  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None, help="class-folder root (default: from dataset/dataset_info.json)")
    ap.add_argument("--leaf-map", default=None)
    ap.add_argument("--model-dir", default=str(PROJECT_ROOT / "models"))
    ap.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--epochs", type=int, default=10, help="max epochs for the head-training stage")
    ap.add_argument("--fine-tune-epochs", type=int, default=0, help="max epochs for optional fine-tuning (0 = off)")
    ap.add_argument("--fine-tune-at", type=int, default=100, help="first MobileNetV2 layer index to unfreeze")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fine-tune-lr", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--weights", choices=["imagenet", "none"], default="imagenet",
                    help="'none' = random init (NOT transfer learning); only for pipeline checks")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="use only N images per class (pipeline check ONLY; recorded in model_info.json)")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--skip-hash", action="store_true", help="skip MD5 duplicate detection")
    ap.add_argument("--skip-readability-check", action="store_true")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    import keras
    import tensorflow as tf
    keras.utils.set_random_seed(args.seed)

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    device = f"GPU ({len(gpus)} device(s))" if gpus else "CPU"
    print(f"TensorFlow {tf.__version__} | Keras {keras.__version__} | training on {device}")
    if not gpus:
        print("No GPU detected: training on CPU is slow for the full dataset. Consider Google Colab (see README).")

    model_dir, results_dir = Path(args.model_dir), Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tmp = model_dir / "_training_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    # ---- dataset ---------------------------------------------------------------
    try:
        root, leaf_map_path = resolve_dataset_location(PROJECT_ROOT, args.data_dir, args.leaf_map)
        print(f"Dataset root: {root}")
        report = verify_dataset(root, check_readable=not args.skip_readability_check)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    from src.dataset_utils import discover_images
    df = discover_images(root)
    df = df[~df["rel_path"].isin(report["unreadable_files"])].reset_index(drop=True)
    if args.max_per_class:
        df = subsample_per_class(df, args.max_per_class, args.seed)
        print(f"WARNING: SUBSET MODE ({args.max_per_class} images/class). Pipeline check only; not a real model.")
    leaf_map = load_leaf_map(leaf_map_path)
    print("Grouping images (leaf map / photo id / identical files) to prevent split leakage ...")
    df, group_info = assign_groups(df, root, leaf_map, hash_files=not args.skip_hash)
    df = make_splits(df, SPLIT_FRACTIONS, args.seed)
    class_names = sorted(df["class_name"].unique())
    idx = {c: i for i, c in enumerate(class_names)}
    df["label"] = df["class_name"].map(idx)
    df[["rel_path", "class_name", "group", "split"]].to_csv(results_dir / "split_manifest.csv", index=False)
    split_sizes = {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")}
    print(f"{len(class_names)} classes | split sizes {split_sizes} | groups {group_info['num_groups']} "
          f"| leaf-map hits {group_info['leaf_map_hits']}")

    def ds_for(split, training):
        d = df[df["split"] == split]
        return build_tf_dataset([str(root / p) for p in d["rel_path"]], d["label"].to_numpy(),
                                args.batch_size, training, args.seed)

    train_ds, val_ds = ds_for("train", True), ds_for("val", False)
    class_weight = None if args.no_class_weights else compute_class_weights(
        df.loc[df["split"] == "train", "label"].to_numpy(), len(class_names))

    # ---- model -----------------------------------------------------------------
    pretrained = args.weights == "imagenet"
    try:
        model = build_model(len(class_names), weights="imagenet" if pretrained else None, dropout=args.dropout)
    except Exception as exc:
        print(f"ERROR: could not build the model with ImageNet weights ({exc}).\n"
              "Check your internet connection. Not falling back to random weights silently; "
              "use --weights none only for a deliberate pipeline check.", file=sys.stderr)
        return 1
    if pretrained:
        freeze_base(model)
    compile_model(model, args.lr)
    print(f"Parameters: {model.count_params():,} total | stage 1 trainable: "
          f"{sum(int(keras.ops.size(w)) for w in model.trainable_weights):,}")

    history_rows, stages = [], []
    best = {"val_loss": float("inf"), "stage": None}

    def run_stage(stage, epochs):
        cbs = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(str(tmp / f"checkpoint_stage{stage}.keras"), monitor="val_loss",
                                            save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, args.patience - 1)),
        ]
        h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight,
                      callbacks=cbs, verbose=2).history
        n = len(h["loss"])
        for e in range(n):
            history_rows.append({"stage": stage, "epoch_in_stage": e + 1, **{k: float(v[e]) for k, v in h.items()}})
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        stages.append({"stage": stage, "epochs_run": n, "val_loss_after_restore": float(val_loss),
                       "val_accuracy_after_restore": float(val_acc)})
        if val_loss < best["val_loss"]:
            best.update(val_loss=float(val_loss), stage=stage)
            model.save(tmp / MODEL_FILENAME)
            print(f"Stage {stage}: new best validation loss {val_loss:.4f} -> saved")

    run_stage(1, args.epochs)
    if args.fine_tune_epochs > 0:
        if not pretrained:
            print("Skipping fine-tuning: the base was trained from scratch (--weights none).")
        else:
            n_train = unfreeze_top_layers(model, args.fine_tune_at)
            compile_model(model, args.fine_tune_lr)
            print(f"Fine-tuning {n_train} base layers (lr={args.fine_tune_lr}).")
            run_stage(2, args.fine_tune_epochs)

    # ---- evaluate the SAVED artefact on the untouched test split ----------------
    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(results_dir / "training_history.csv", index=False)
    (results_dir / "training_history.json").write_text(hist_df.to_json(orient="records", indent=2), encoding="utf-8")
    plot_training_curves(hist_df, results_dir / "training_curves.png")
    saved = keras.models.load_model(tmp / MODEL_FILENAME, compile=False)
    test_df = df[df["split"] == "test"]
    summary = evaluate_model(saved, test_df, root, class_names, results_dir, args.batch_size)
    print(f"TEST accuracy {summary['accuracy']:.4f} | top-3 {summary['top3_accuracy']:.4f} | "
          f"macro-F1 {summary['macro_f1']:.4f} ({summary['num_test_images']} images)")

    # ---- publish model files (only now: training finished and was evaluated) ----
    stamp = datetime.now(timezone.utc)
    info = {
        "schema_version": INFO_SCHEMA_VERSION, "is_test_fixture": False,
        "model_version": "pdc-mobilenetv2-" + stamp.strftime("%Y%m%d-%H%M%S"),
        "trained_at": stamp.isoformat(timespec="seconds"),
        "architecture_id": ARCHITECTURE_ID, "preprocessing_id": PREPROCESSING_ID, "img_size": IMG_SIZE,
        "num_classes": len(class_names), "class_names": class_names,
        "pretrained_weights": "imagenet" if pretrained else "none",
        "training_mode": "subset" if args.max_per_class else "full", "max_per_class": args.max_per_class,
        "dataset": {"name": DATASET_NAME, "num_images_used": int(len(df)), "num_images_found": report["num_images"],
                    "unreadable_excluded": len(report["unreadable_files"]),
                    "grouping": group_info, "split_fractions": list(SPLIT_FRACTIONS)},
        "split_sizes": split_sizes,
        "config": {k: v for k, v in vars(args).items() if k not in ("data_dir", "model_dir", "results_dir", "leaf_map")},
        "stages": stages, "best_stage": best["stage"], "best_val_loss": best["val_loss"],
        "device": device, "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__, "keras_version": keras.__version__,
        "training_seconds": round(time.time() - t0, 1),
        "evaluation": {k: v for k, v in summary.items() if k != "per_class"},
    }
    shutil.copy2(tmp / MODEL_FILENAME, model_dir / (MODEL_FILENAME + ".part"))
    os.replace(model_dir / (MODEL_FILENAME + ".part"), model_dir / MODEL_FILENAME)
    (model_dir / CLASS_NAMES_FILENAME).write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (model_dir / MODEL_INFO_FILENAME).write_text(json.dumps(info, indent=2), encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Saved {MODEL_FILENAME}, {CLASS_NAMES_FILENAME}, {MODEL_INFO_FILENAME} to {model_dir}")
    print("Training finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
