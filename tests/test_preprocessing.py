import io

import numpy as np
import pytest
from PIL import Image

from conftest import make_image_bytes
from src import preprocessing as pp


def test_extension_check():
    assert pp.check_extension("leaf.JPG") == "jpg"
    assert pp.check_extension("leaf.png") == "png"
    for bad in ["leaf.gif", "leaf.exe", "leaf", "", None, "a.jpg.php"]:
        with pytest.raises(pp.ImageValidationError):
            pp.check_extension(bad)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_decode_valid(fmt):
    img = pp.decode_image_bytes(make_image_bytes(fmt))
    assert img.mode == "RGB" and img.size == (200, 160)


def test_decode_rgba_png_is_flattened_to_rgb():
    img = pp.decode_image_bytes(make_image_bytes("PNG", mode="RGBA"))
    assert img.mode == "RGB"


@pytest.mark.parametrize("data", [b"", b"not an image at all", b"\xff\xd8\xff\xe0garbage"])
def test_decode_rejects_garbage(data):
    with pytest.raises(pp.ImageValidationError):
        pp.decode_image_bytes(data)


def test_decode_rejects_truncated_image():
    data = make_image_bytes("PNG", size=(300, 300))
    with pytest.raises(pp.ImageValidationError):
        pp.decode_image_bytes(data[: len(data) // 2])


def test_decode_rejects_wrong_format_even_with_image_content():
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)).save(buf, format="GIF")
    with pytest.raises(pp.ImageValidationError, match="not a JPEG or PNG"):
        pp.decode_image_bytes(buf.getvalue())


def test_decode_rejects_too_small_and_too_large_dimensions():
    with pytest.raises(pp.ImageValidationError, match="too small"):
        pp.decode_image_bytes(make_image_bytes(size=(32, 32)))
    with pytest.raises(pp.ImageValidationError, match="too large"):
        pp.decode_image_bytes(make_image_bytes(size=(200, 200)), max_side=150)
    with pytest.raises(pp.ImageValidationError, match="too large"):
        pp.decode_image_bytes(make_image_bytes(size=(200, 200)), max_pixels=10_000)


def test_decode_rejects_blank_image():
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), (10, 10, 10)).save(buf, format="PNG")
    with pytest.raises(pp.ImageValidationError, match="blank"):
        pp.decode_image_bytes(buf.getvalue())


def test_exif_orientation_is_applied():
    arr = np.random.default_rng(1).integers(0, 255, (100, 200, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise when displayed
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    assert pp.decode_image_bytes(buf.getvalue()).size == (100, 200)


def test_preprocess_output_contract():
    img = pp.decode_image_bytes(make_image_bytes(size=(321, 257)))
    x = pp.preprocess_pil_image(img)
    assert x.shape == (1, 224, 224, 3) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 255.0  # raw 0..255; the model rescales internally


def test_training_and_inference_preprocessing_agree(tmp_path):
    """Same file through the training loader (tf) and the upload path (PIL -> tf) must match closely."""
    import tensorflow as tf
    path = tmp_path / "leaf.jpg"
    path.write_bytes(make_image_bytes("JPEG", size=(300, 260), seed=3))
    train_x = pp.tf_load_image(tf.constant(str(path))).numpy()
    infer_x = pp.preprocess_pil_image(pp.decode_image_bytes(path.read_bytes()))[0]
    assert train_x.shape == infer_x.shape == (224, 224, 3)
    diff = np.abs(train_x - infer_x)
    assert diff.mean() < 1.0 and diff.max() < 12.0, (diff.mean(), diff.max())


def test_thumbnail_data_uri():
    uri = pp.make_thumbnail_data_uri(pp.decode_image_bytes(make_image_bytes(size=(900, 700))))
    assert uri.startswith("data:image/jpeg;base64,")
