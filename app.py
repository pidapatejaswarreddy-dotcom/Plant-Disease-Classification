"""Flask web application: Plant Disease Classification using Deep Learning.

Run:  python app.py   ->  http://127.0.0.1:5000
Uploads are validated, decoded and classified in memory; they are never written to disk.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import HTTPException, NotFound, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import Config
from src.class_info import parse_class_name
from src.model_loader import ModelManager, ModelNotReadyError
from src.predictor import predict
from src.preprocessing import (ImageValidationError, check_extension, decode_image_bytes,
                               make_thumbnail_data_uri)

log = logging.getLogger("plant_disease_app")

# Result files the app may serve from the results folder (whitelist; nothing else is reachable).
RESULT_FIGURES = {"dataset_class_distribution.png", "dataset_samples.png",
                  "training_curves.png", "confusion_matrix.png"}


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
    logging.basicConfig(level=logging.INFO)

    manager = ModelManager(app.config["MODEL_DIR"], app.config["ALLOW_TEST_MODEL"])
    app.extensions["model_manager"] = manager

    # ------------------------------------------------------------------ helpers
    def supported_plants():
        try:
            names = manager.get().class_names
        except ModelNotReadyError:
            return None
        grouped = {}
        for n in names:
            p = parse_class_name(n)
            grouped.setdefault(p["plant"], []).append(p["condition"])
        return dict(sorted(grouped.items()))

    def model_context():
        try:
            b = manager.get()
            return {"ready": True, "info": b.info, "warnings": b.warnings}
        except ModelNotReadyError as exc:
            return {"ready": False, "reason": str(exc), "info": None, "warnings": []}

    def setup_page(reason: str, status: int = 503):
        return render_template("setup_required.html", reason=reason), status

    def detect_page(error: str | None = None, status: int = 200):
        return render_template("detect.html", error=error, max_mb=app.config["MAX_UPLOAD_MB"],
                               model=model_context()), status

    def run_upload(file_storage):
        """Validate + classify an upload. Returns (result, thumbnail_uri, safe_name)."""
        if file_storage is None or not file_storage.filename:
            raise ImageValidationError("No file was selected. Please choose an image to upload.")
        safe_name = secure_filename(file_storage.filename) or "upload"
        check_extension(safe_name, app.config["ALLOWED_EXTENSIONS"])
        data = file_storage.read()
        image = decode_image_bytes(data)
        bundle = manager.get()
        result = predict(bundle, image, app.config["TOP_K"], app.config["LOW_CONFIDENCE_THRESHOLD"])
        return result, make_thumbnail_data_uri(image), safe_name, bundle

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return render_template("index.html", plants=supported_plants(), model=model_context())

    @app.get("/detect")
    def detect():
        try:
            manager.get()
        except ModelNotReadyError as exc:
            return setup_page(str(exc))
        return detect_page()

    @app.post("/predict")
    def predict_route():
        try:
            manager.get()
        except ModelNotReadyError as exc:
            return setup_page(str(exc))
        try:
            result, thumb, name, bundle = run_upload(request.files.get("image"))
        except ImageValidationError as exc:
            return detect_page(str(exc), 400)
        return render_template("result.html", result=result, thumbnail=thumb, filename=name,
                               model=model_context(), warnings=bundle.warnings)

    @app.get("/about")
    def about():
        res = Path(app.config["RESULTS_DIR"])
        return render_template(
            "about.html", model=model_context(),
            dataset_stats=_read_json(res / "dataset_stats.json"),
            evaluation=_read_json(res / "evaluation_results.json"),
            figures={f: (res / f).is_file() for f in RESULT_FIGURES})

    @app.get("/results/<path:name>")
    def results_file(name):
        if name not in RESULT_FIGURES:
            raise NotFound()
        return send_from_directory(Path(app.config["RESULTS_DIR"]).resolve(), name)

    # ------------------------------------------------------------------ JSON API
    @app.get("/status")
    def status():
        return jsonify(manager.status())

    @app.post("/api/predict")
    def api_predict():
        try:
            manager.get()
        except ModelNotReadyError as exc:
            return jsonify(error="model_not_ready", message=str(exc)), 503
        try:
            result, _thumb, _name, _b = run_upload(request.files.get("image"))
        except ImageValidationError as exc:
            return jsonify(error="invalid_image", message=str(exc)), 400
        return jsonify(
            predicted_class=result["prediction"]["class_name"], plant=result["prediction"]["plant"],
            condition=result["prediction"]["condition"], status=result["prediction"]["status"],
            confidence=result["confidence"], inconclusive=result["inconclusive"], top=result["top"],
            note="Softmax score; not a calibrated probability.")

    # ------------------------------------------------------------------ errors / headers
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_e):
        msg = f"The file is too large. Maximum size is {app.config['MAX_UPLOAD_MB']} MB."
        if request.path.startswith("/api/"):
            return jsonify(error="file_too_large", message=msg), 413
        return detect_page(msg, 413)

    @app.errorhandler(HTTPException)
    def http_error(e):
        if request.path.startswith("/api/"):
            return jsonify(error=e.name, message=e.description), e.code
        return render_template("error.html", code=e.code, title=e.name), e.code

    @app.errorhandler(Exception)
    def unexpected(e):
        log.exception("Unhandled error")  # details only in the server log, never in the page
        if request.path.startswith("/api/"):
            return jsonify(error="server_error", message="An unexpected error occurred."), 500
        return render_template("error.html", code=500, title="Something went wrong"), 500

    @app.after_request
    def security_headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; "
            "frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
        return resp

    return app


app = create_app()

if __name__ == "__main__":
    print(f"Open http://{Config.HOST}:{Config.PORT} in your browser (Ctrl+C to stop).")
    app.run(host=Config.HOST, port=Config.PORT, debug=False)
