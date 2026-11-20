"""Upload validation and image preprocessing.

The SAME resize operation (tf.image.resize, bilinear, antialias=True) is used for
training (``tf_load_image``) and for inference (``resize_for_model``). The rescale to
[-1, 1] lives inside the saved model, so the app never normalises by hand.
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .constants import IMG_SIZE

log = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = frozenset({"jpg", "jpeg", "png"})
ALLOWED_FORMATS = frozenset({"JPEG", "PNG"})
MIN_SIDE = 64
MAX_SIDE = 12000
MAX_PIXELS = 40_000_000
MIN_STD = 2.0  # reject (near-)uniform images


class ImageValidationError(ValueError):
    """Raised with a user-presentable message when an upload cannot be used."""


def check_extension(filename: str | None, allowed=ALLOWED_EXTENSIONS) -> str:
    ext = Path(filename or "").suffix.lower().lstrip(".")
    if ext not in allowed:
        raise ImageValidationError(
            "Unsupported file type. Please upload a JPG, JPEG or PNG image."
        )
    return ext


def decode_image_bytes(data: bytes, *, min_side=MIN_SIDE, max_side=MAX_SIDE,
                       max_pixels=MAX_PIXELS) -> Image.Image:
    """Decode untrusted bytes into an RGB PIL image or raise ImageValidationError."""
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            width, height = probe.size
            if fmt not in ALLOWED_FORMATS:
                raise ImageValidationError(
                    "The file content is not a JPEG or PNG image (the extension does not match).")
            if width < min_side or height < min_side:
                raise ImageValidationError(
                    f"The image is too small ({width}x{height}). Minimum is {min_side}x{min_side} pixels.")
            if width > max_side or height > max_side or width * height > max_pixels:
                raise ImageValidationError(
                    f"The image is too large ({width}x{height}). Please upload a smaller photo.")
            probe.verify()  # structural check; the image must be re-opened afterwards
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgba = img.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, (255, 255, 255))
                rgb.paste(rgba, mask=rgba.split()[-1])
            else:
                rgb = img.convert("RGB")
    except ImageValidationError:
        raise
    except Exception as exc:  # corrupted / truncated / decompression bomb / anything odd
        log.debug("Image decode failed: %r", exc)
        raise ImageValidationError(
            "The image could not be read. It may be corrupted or incomplete.") from exc

    small = np.asarray(rgb.resize((64, 64)), dtype=np.float32)
    if float(small.std()) < MIN_STD:
        raise ImageValidationError(
            "The image looks blank or uniform. Please upload a clear photo of a plant leaf.")
    return rgb


def pil_to_uint8(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def resize_for_model(rgb_uint8: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 (224,224,3) with values in 0..255 (inference path)."""
    import tensorflow as tf
    t = tf.convert_to_tensor(rgb_uint8, dtype=tf.uint8)
    t = tf.image.resize(t, (IMG_SIZE, IMG_SIZE), method="bilinear", antialias=True)
    return t.numpy().astype("float32")


def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    """PIL image -> model input batch of shape (1, 224, 224, 3)."""
    return resize_for_model(pil_to_uint8(img))[None, ...]


def tf_load_image(path):
    """Training path: file path tensor -> float32 (224,224,3), same resize as inference."""
    import tensorflow as tf
    data = tf.io.read_file(path)
    img = tf.io.decode_image(data, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear", antialias=True)
    return tf.cast(img, tf.float32)


def make_thumbnail_data_uri(img: Image.Image, max_side: int = 480) -> str:
    """Small JPEG data URI so the result page can show the upload without saving it."""
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
