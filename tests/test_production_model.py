"""Exercises the PRODUCTION inference path with the REAL trained model in models/ (or PDC_MODEL_DIR).

Skipped (not failed, not 'passed') when no genuine model is installed; the skip reason says so.
"""
import io
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("PDC_MODEL_DIR", ROOT / "models"))
HAVE_MODEL = (MODEL_DIR / "plant_disease_model.keras").is_file() and (MODEL_DIR / "model_info.json").is_file()

pytestmark = pytest.mark.skipif(not HAVE_MODEL, reason=f"no trained model found in {MODEL_DIR}")


def _find_real_image():
    info = ROOT / "dataset" / "dataset_info.json"
    if not info.is_file():
        return None
    root = ROOT / json.loads(info.read_text())["dataset_root"]
    for p in root.rglob("*.JPG"):
        return p
    return None


def test_real_model_loads_with_production_loader_and_predicts():
    from src.model_loader import load_model_bundle
    from src.predictor import predict
    from src.preprocessing import decode_image_bytes
    b = load_model_bundle(MODEL_DIR, allow_test_fixture=False)
    assert not b.is_test_fixture and b.info["is_test_fixture"] is False
    img_path = _find_real_image()
    if img_path is None:
        pytest.skip("dataset not present locally; cannot test on a genuine leaf image")
    r = predict(b, decode_image_bytes(img_path.read_bytes()))
    assert r["top"][0]["class_name"] in b.class_names
    assert abs(sum(t["confidence"] for t in r["top"]) - 1.0) <= 1.0 and 0 <= r["confidence"] <= 1


def test_real_model_through_flask_app():
    from app import create_app
    img_path = _find_real_image()
    if img_path is None:
        pytest.skip("dataset not present locally")
    app = create_app({"TESTING": True, "MODEL_DIR": MODEL_DIR})
    r = app.test_client().post("/api/predict", data={"image": (io.BytesIO(img_path.read_bytes()), "leaf.jpg")},
                               content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["predicted_class"]
