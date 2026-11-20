"""Real inference: preprocess -> model -> top-k classes with their softmax scores."""
from __future__ import annotations

import numpy as np

from .class_info import get_class_details
from .preprocessing import preprocess_pil_image


def predict(bundle, pil_image, top_k: int = 3, low_conf_threshold: float = 0.60) -> dict:
    x = preprocess_pil_image(pil_image)
    with bundle.lock:
        probs = np.asarray(bundle.model(x, training=False))[0].astype("float64")
    order = np.argsort(probs)[::-1][:top_k]
    top = []
    for i in order:
        d = get_class_details(bundle.class_names[int(i)])
        top.append({"class_name": d["class_name"], "plant": d["plant"], "condition": d["condition"],
                    "status": d["status"], "confidence": float(probs[int(i)])})
    best = get_class_details(top[0]["class_name"])
    conf = top[0]["confidence"]
    return {
        "top": top,
        "prediction": best,
        "confidence": conf,
        "confidence_percent": round(conf * 100, 1),
        "inconclusive": conf < low_conf_threshold,
        "low_conf_threshold": low_conf_threshold,
        "model_version": bundle.info.get("model_version"),
    }
