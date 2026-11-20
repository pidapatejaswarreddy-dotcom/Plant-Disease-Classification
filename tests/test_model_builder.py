"""Tests the REAL architecture (MobileNetV2 head) with untrained random weights.

These models exist only inside pytest temp folders and are NOT trained: they verify shapes,
freezing logic and the save -> load -> production-compatibility-check -> predict path.
"""
import json

import numpy as np
import pytest

from src.constants import (ARCHITECTURE_ID, CLASS_NAMES_FILENAME, INFO_SCHEMA_VERSION, MODEL_FILENAME,
                           MODEL_INFO_FILENAME, PREPROCESSING_ID)
from src.model_builder import build_model, freeze_base, get_base, unfreeze_top_layers

CLASSES = ["Tomato___healthy", "Tomato___Late_blight", "Potato___Early_blight", "Apple___Apple_scab"]


@pytest.fixture(scope="module")
def untrained_model():
    return build_model(len(CLASSES), weights=None)


def test_architecture_shapes_and_layers(untrained_model):
    m = untrained_model
    assert m.input_shape == (None, 224, 224, 3) and m.output_shape == (None, len(CLASSES))
    names = [layer.name for layer in m.layers]
    for expected in ("aug_flip", "rescale_minus1_to_1", "mobilenetv2_base", "gap", "dropout", "predictions"):
        assert expected in names
    out = np.asarray(m(np.full((2, 224, 224, 3), 128.0, "float32"), training=False))
    assert out.shape == (2, len(CLASSES)) and np.allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_augmentation_is_inactive_at_inference(untrained_model):
    x = np.random.default_rng(0).uniform(0, 255, (1, 224, 224, 3)).astype("float32")
    a = np.asarray(untrained_model(x, training=False))
    b = np.asarray(untrained_model(x, training=False))
    assert np.array_equal(a, b)


def test_freeze_and_fine_tune_logic():
    m = build_model(len(CLASSES), weights=None)
    freeze_base(m)
    assert not get_base(m).trainable
    trainable = {w.path.split("/")[0] for w in m.trainable_weights}
    assert trainable == {"predictions"}
    n = unfreeze_top_layers(m, fine_tune_at=100)
    assert n > 0
    import keras
    base = get_base(m)
    assert all(not l.trainable for l in base.layers[:100])
    assert all(not l.trainable for l in base.layers if isinstance(l, keras.layers.BatchNormalization))


def test_real_architecture_roundtrip_through_production_loader(tmp_path, untrained_model):
    """Save exactly like train.py, then load with the PRODUCTION loader (fixtures not allowed)."""
    import keras
    from PIL import Image
    from src.model_loader import load_model_bundle
    from src.predictor import predict
    untrained_model.save(tmp_path / MODEL_FILENAME)
    (tmp_path / CLASS_NAMES_FILENAME).write_text(json.dumps(CLASSES))
    (tmp_path / MODEL_INFO_FILENAME).write_text(json.dumps({
        "schema_version": INFO_SCHEMA_VERSION, "is_test_fixture": False, "model_version": "TEST-ARCH-UNTRAINED",
        "architecture_id": ARCHITECTURE_ID, "preprocessing_id": PREPROCESSING_ID, "img_size": 224,
        "num_classes": len(CLASSES), "class_names": CLASSES, "keras_version": keras.__version__}))
    bundle = load_model_bundle(tmp_path, allow_test_fixture=False)
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (300, 260, 3), dtype=np.uint8))
    r = predict(bundle, img, top_k=3, low_conf_threshold=0.6)
    assert len(r["top"]) == 3 and r["top"][0]["class_name"] in CLASSES
    assert r["top"][0]["confidence"] >= r["top"][1]["confidence"] >= r["top"][2]["confidence"]
