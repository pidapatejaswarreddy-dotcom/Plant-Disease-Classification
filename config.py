"""Central configuration for the Flask web application.

Nothing here is secret. Paths are resolved relative to this file, so the project
can be extracted anywhere (no hard-coded user paths).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    # Where the trained model files live (override with the PDC_MODEL_DIR env var).
    MODEL_DIR = Path(os.environ.get("PDC_MODEL_DIR", BASE_DIR / "models"))
    # Where training/evaluation figures and JSON files live.
    RESULTS_DIR = Path(os.environ.get("PDC_RESULTS_DIR", BASE_DIR / "results"))

    # Upload limits. Files are processed in memory and are never written to disk.
    MAX_UPLOAD_MB = 8
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

    # Below this top-1 softmax score the app reports the result as "inconclusive".
    # This is a heuristic, NOT a calibrated probability threshold.
    LOW_CONFIDENCE_THRESHOLD = 0.60
    TOP_K = 3

    # Only the automated tests set this to True, to load a tiny throw-away fixture model.
    ALLOW_TEST_MODEL = False

    HOST = "127.0.0.1"
    PORT = int(os.environ.get("PORT", "5000"))
