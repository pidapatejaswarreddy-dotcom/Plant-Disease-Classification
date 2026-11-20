"""Loads the trained model + labels and checks that they are compatible with this app.

There is no fallback: if anything is missing or incompatible a ModelNotReadyError is
raised and the web app shows setup instructions instead of inventing predictions.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (ARCHITECTURE_ID, CLASS_NAMES_FILENAME, IMG_SIZE, INFO_SCHEMA_VERSION,
                        MODEL_FILENAME, MODEL_INFO_FILENAME, PREPROCESSING_ID)


class ModelNotReadyError(RuntimeError):
    """The trained model is missing or incompatible."""


@dataclass
class ModelBundle:
    model: Any
    class_names: list
    info: dict
    warnings: list = field(default_factory=list)
    is_test_fixture: bool = False
    lock: Any = field(default_factory=threading.Lock)


def _required_paths(model_dir: Path):
    return [model_dir / MODEL_FILENAME, model_dir / CLASS_NAMES_FILENAME, model_dir / MODEL_INFO_FILENAME]


def load_model_bundle(model_dir, allow_test_fixture: bool = False) -> ModelBundle:
    model_dir = Path(model_dir)
    missing = [p.name for p in _required_paths(model_dir) if not p.is_file()]
    if missing:
        raise ModelNotReadyError(f"Missing file(s) in the model folder: {', '.join(missing)}.")
    try:
        info = json.loads((model_dir / MODEL_INFO_FILENAME).read_text(encoding="utf-8"))
        class_names = json.loads((model_dir / CLASS_NAMES_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelNotReadyError(f"Could not read model metadata: {exc}") from exc

    is_fixture = bool(info.get("is_test_fixture"))
    if is_fixture and not allow_test_fixture:
        raise ModelNotReadyError("This model is marked as a test fixture and cannot be used by the app.")
    if info.get("schema_version") != INFO_SCHEMA_VERSION:
        raise ModelNotReadyError("model_info.json has an unsupported schema version.")
    if info.get("preprocessing_id") != PREPROCESSING_ID:
        raise ModelNotReadyError("The model was trained with a different preprocessing pipeline than this app uses.")
    if not is_fixture and info.get("architecture_id") != ARCHITECTURE_ID:
        raise ModelNotReadyError("The model architecture does not match the one this project trains.")
    if info.get("img_size") != IMG_SIZE:
        raise ModelNotReadyError(f"The model expects a different input size than {IMG_SIZE}x{IMG_SIZE}.")
    if (not isinstance(class_names, list) or len(class_names) < 2 or
            len(set(class_names)) != len(class_names) or
            not all(isinstance(c, str) for c in class_names)):
        raise ModelNotReadyError("class_names.json must be a list of at least two unique class names.")
    if info.get("class_names") != class_names:
        raise ModelNotReadyError("class_names.json does not match the class list recorded in model_info.json.")

    try:
        import keras
        model = keras.models.load_model(model_dir / MODEL_FILENAME, compile=False)
    except Exception as exc:
        raise ModelNotReadyError(f"The model file could not be loaded: {exc}") from exc

    if tuple(model.input_shape[1:]) != (IMG_SIZE, IMG_SIZE, 3):
        raise ModelNotReadyError(f"Unexpected model input shape {model.input_shape}.")
    if int(model.output_shape[-1]) != len(class_names):
        raise ModelNotReadyError(
            f"The model has {model.output_shape[-1]} outputs but class_names.json lists {len(class_names)} classes.")

    out = np.asarray(model(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32"), training=False))
    if out.shape != (1, len(class_names)) or not np.all(np.isfinite(out)) or abs(float(out.sum()) - 1.0) > 1e-3:
        raise ModelNotReadyError("The model failed the inference smoke test (output is not a valid probability vector).")

    warnings = []
    saved = str(info.get("keras_version", ""))
    if saved and saved.split(".")[:2] != str(keras.__version__).split(".")[:2]:
        warnings.append(f"Model was saved with Keras {saved}; this environment has Keras {keras.__version__}.")
    return ModelBundle(model=model, class_names=class_names, info=info, warnings=warnings,
                       is_test_fixture=is_fixture)


class ModelManager:
    """Lazy, thread-safe loader that notices when model files change on disk."""

    def __init__(self, model_dir, allow_test_fixture: bool = False):
        self.model_dir = Path(model_dir)
        self.allow_test_fixture = allow_test_fixture
        self._lock = threading.Lock()
        self._bundle = None
        self._error = None
        self._sig = None

    def _signature(self):
        sig = []
        for p in _required_paths(self.model_dir):
            try:
                st = p.stat()
                sig.append((p.name, st.st_mtime_ns, st.st_size))
            except OSError:
                sig.append((p.name, None, None))
        return tuple(sig)

    def get(self) -> ModelBundle:
        with self._lock:
            sig = self._signature()
            if sig == self._sig:
                if self._bundle is not None:
                    return self._bundle
                raise ModelNotReadyError(self._error)
            self._sig, self._bundle, self._error = sig, None, None
            try:
                self._bundle = load_model_bundle(self.model_dir, self.allow_test_fixture)
                return self._bundle
            except ModelNotReadyError as exc:
                self._error = str(exc)
                raise

    def status(self) -> dict:
        try:
            b = self.get()
            return {"ready": True, "message": "Model loaded.", "model_version": b.info.get("model_version"),
                    "num_classes": len(b.class_names), "warnings": b.warnings}
        except ModelNotReadyError as exc:
            return {"ready": False, "message": str(exc)}
