import io
import json

from conftest import make_image_bytes


def post_image(client, data, filename="leaf.jpg", url="/predict", ctype="image/jpeg"):
    return client.post(url, data={"image": (io.BytesIO(data), filename, ctype)}, content_type="multipart/form-data")


# ---------------------------------------------------------------- routes
def test_pages_render(client):
    for url in ("/", "/detect", "/about"):
        r = client.get(url)
        assert r.status_code == 200, url
    assert b"Detect Disease" in client.get("/").data


def test_status_endpoint(client):
    j = client.get("/status").get_json()
    assert j["ready"] is True and j["num_classes"] == 3


def test_unknown_route_is_404_without_trace(client):
    r = client.get("/nope")
    assert r.status_code == 404 and b"Traceback" not in r.data


def test_security_headers(client):
    h = client.get("/").headers
    assert h["X-Content-Type-Options"] == "nosniff" and h["X-Frame-Options"] == "DENY"
    assert "script-src 'self'" in h["Content-Security-Policy"]


# ---------------------------------------------------------------- inference with FIXTURE model
def test_predict_valid_upload_returns_result_page(client):
    r = post_image(client, make_image_bytes("JPEG"))
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "Prediction result" in html
    assert ("Model score" in html) or ("Inconclusive result" in html)
    assert "not</em> a calibrated probability" in html
    assert "Analyse another image" in html and "data:image/jpeg;base64," in html


def test_predict_png_and_api_json(client):
    r = post_image(client, make_image_bytes("PNG"), "leaf.png", "/api/predict", "image/png")
    assert r.status_code == 200
    j = r.get_json()
    assert j["predicted_class"] in {"Apple___Apple_scab", "Tomato___Early_blight", "Tomato___healthy"}
    confs = [t["confidence"] for t in j["top"]]
    assert confs == sorted(confs, reverse=True) and sum(confs) <= 1.0 + 1e-6
    assert 0 <= j["confidence"] <= 1 and len(j["top"]) == 3 and "calibrated" in j["note"]


def test_inference_is_deterministic(client):
    data = make_image_bytes("JPEG", seed=5)
    a = post_image(client, data, url="/api/predict").get_json()
    b = post_image(client, data, url="/api/predict").get_json()
    assert a == b


# ---------------------------------------------------------------- validation
def test_rejects_wrong_extension(client):
    r = post_image(client, make_image_bytes(), "leaf.gif")
    assert r.status_code == 400 and b"Unsupported file type" in r.data


def test_rejects_fake_image_with_valid_extension(client):
    r = post_image(client, b"<?php echo 1; ?>", "shell.jpg")
    assert r.status_code == 400 and b"could not be read" in r.data


def test_rejects_corrupted_and_empty(client):
    assert post_image(client, make_image_bytes()[:50]).status_code == 400
    assert post_image(client, b"").status_code == 400


def test_no_file_selected(client):
    r = client.post("/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400 and b"No file" in r.data


def test_oversize_upload_returns_413(client):
    client.application.config["MAX_CONTENT_LENGTH"] = 1024
    r = post_image(client, b"x" * 5000)
    assert r.status_code == 413 and b"too large" in r.data
    j = post_image(client, b"x" * 5000, url="/api/predict")
    assert j.status_code == 413 and j.get_json()["error"] == "file_too_large"


def test_path_traversal_filename_is_harmless_and_nothing_is_written(client, tmp_path):
    before = set(tmp_path.rglob("*"))
    r = post_image(client, make_image_bytes(), "../../etc/passwd.jpg")
    assert r.status_code == 200 and b"passwd.jpg" in r.data and b"../" not in r.data
    assert set(tmp_path.rglob("*")) == before


def test_results_route_whitelist(client):
    assert client.get("/results/../app.py").status_code == 404
    assert client.get("/results/evaluation_results.json").status_code == 404
    assert client.get("/results/%2e%2e%2fapp.py").status_code == 404
    assert client.get("/results/training_curves.png").status_code == 404  # whitelisted but file absent


def test_xss_in_filename_is_escaped(client):
    r = post_image(client, make_image_bytes(), "<script>alert(1)</script>.jpg")
    assert b"<script>alert(1)" not in r.data


# ---------------------------------------------------------------- missing model
def test_missing_model_shows_setup_page_and_never_predicts(client_no_model):
    for method, url in (("get", "/detect"), ("post", "/predict")):
        r = getattr(client_no_model, method)(url)
        assert r.status_code == 503
        assert b"Trained model required" in r.data and b"never invent" in r.data
        assert b"Model score" not in r.data
    api = post_image(client_no_model, make_image_bytes(), url="/api/predict")
    assert api.status_code == 503 and api.get_json()["error"] == "model_not_ready"
    assert client_no_model.get("/").status_code == 200
    assert client_no_model.get("/about").status_code == 200


def test_production_config_rejects_fixture_model(fixture_model_dir, tmp_path):
    from app import create_app
    app = create_app({"TESTING": True, "MODEL_DIR": fixture_model_dir, "RESULTS_DIR": tmp_path})
    r = app.test_client().get("/detect")
    assert r.status_code == 503 and b"test fixture" in r.data


def test_unexpected_error_hides_stack_trace(client, monkeypatch):
    import app as app_module
    def boom(*a, **k):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(app_module, "predict", boom)
    client.application.config["PROPAGATE_EXCEPTIONS"] = False
    r = post_image(client, make_image_bytes())
    assert r.status_code == 500 and b"secret internal detail" not in r.data and b"Traceback" not in r.data


def test_prediction_depends_on_the_image(client):
    """Guards against an inference path that ignores its input (constant output)."""
    import io as _io
    import numpy as np
    from PIL import Image

    def solid_noise(level, seed):
        rng = np.random.default_rng(seed)
        arr = np.clip(rng.normal(level, 40, (160, 160, 3)), 0, 255).astype("uint8")
        buf = _io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    dark = post_image(client, solid_noise(30, 1), "a.png", "/api/predict", "image/png").get_json()
    bright = post_image(client, solid_noise(225, 2), "b.png", "/api/predict", "image/png").get_json()
    assert dark["top"] != bright["top"]


def test_confident_branch_shows_description_guidance_and_status(client):
    """Force the non-inconclusive branch (threshold 0) and check what a confident result must show."""
    client.application.config["LOW_CONFIDENCE_THRESHOLD"] = 0.0
    html = post_image(client, make_image_bytes("JPEG", seed=9)).get_data(as_text=True)
    assert "Model score:" in html and "Inconclusive result" not in html
    assert "General guidance (educational only)" in html and "About this class" in html
    assert ("badge ok" in html) or ("badge bad" in html)
    assert "Other likely classes" in html
