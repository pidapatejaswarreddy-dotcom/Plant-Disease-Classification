"""Shared fixtures.

IMPORTANT: the model built here is a tiny, untrained TEST FIXTURE that exists only inside
pytest temp folders and is flagged ``is_test_fixture``. The production loader refuses it;
only the test app (ALLOW_TEST_MODEL=True) may load it. It is never the real model.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import (CLASS_NAMES_FILENAME, IMG_SHAPE, INFO_SCHEMA_VERSION,  # noqa: E402
                           MODEL_FILENAME, MODEL_INFO_FILENAME, PREPROCESSING_ID)

FIXTURE_CLASSES = ["Apple___Apple_scab", "Tomato___Early_blight", "Tomato___healthy"]


def make_fixture_model_dir(path: Path, class_names=FIXTURE_CLASSES, **info_overrides) -> Path:
    import keras
    from keras import layers
    path.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(0)
    inp = keras.Input(shape=IMG_SHAPE)
    x = layers.Rescaling(1 / 127.5, offset=-1.0)(inp)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(len(class_names), activation="softmax")(x)
    keras.Model(inp, out).save(path / MODEL_FILENAME)
    (path / CLASS_NAMES_FILENAME).write_text(json.dumps(class_names))
    info = {"schema_version": INFO_SCHEMA_VERSION, "is_test_fixture": True, "model_version": "TEST-FIXTURE",
            "preprocessing_id": PREPROCESSING_ID, "architecture_id": "test_fixture", "img_size": 224,
            "num_classes": len(class_names), "class_names": class_names,
            "keras_version": keras.__version__}
    info.update(info_overrides)
    (path / MODEL_INFO_FILENAME).write_text(json.dumps(info))
    return path


def make_image_bytes(fmt="JPEG", size=(200, 160), seed=0, mode="RGB") -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr).convert(mode)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="session")
def fixture_model_dir(tmp_path_factory):
    return make_fixture_model_dir(tmp_path_factory.mktemp("fixture_model"))


@pytest.fixture()
def client(fixture_model_dir, tmp_path):
    from app import create_app
    app = create_app({"TESTING": True, "MODEL_DIR": fixture_model_dir, "RESULTS_DIR": tmp_path,
                      "ALLOW_TEST_MODEL": True})
    return app.test_client()


@pytest.fixture()
def client_no_model(tmp_path):
    from app import create_app
    app = create_app({"TESTING": True, "MODEL_DIR": tmp_path / "empty", "RESULTS_DIR": tmp_path})
    return app.test_client()
