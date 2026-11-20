import json

import pytest

from conftest import make_fixture_model_dir
from src.constants import CLASS_NAMES_FILENAME, MODEL_INFO_FILENAME
from src.model_loader import ModelManager, ModelNotReadyError, load_model_bundle


def test_missing_model_raises(tmp_path):
    with pytest.raises(ModelNotReadyError, match="Missing"):
        load_model_bundle(tmp_path)


def test_fixture_is_refused_by_production_loader(fixture_model_dir):
    with pytest.raises(ModelNotReadyError, match="test fixture"):
        load_model_bundle(fixture_model_dir, allow_test_fixture=False)


def test_fixture_loads_only_when_explicitly_allowed(fixture_model_dir):
    b = load_model_bundle(fixture_model_dir, allow_test_fixture=True)
    assert b.is_test_fixture and len(b.class_names) == 3


def test_preprocessing_mismatch_rejected(tmp_path):
    d = make_fixture_model_dir(tmp_path / "m", preprocessing_id="something-else")
    with pytest.raises(ModelNotReadyError, match="preprocessing"):
        load_model_bundle(d, allow_test_fixture=True)


def test_class_count_mismatch_rejected(tmp_path):
    d = make_fixture_model_dir(tmp_path / "m")
    names = ["A___x", "B___y", "C___z", "D___w"]
    (d / CLASS_NAMES_FILENAME).write_text(json.dumps(names))
    info = json.loads((d / MODEL_INFO_FILENAME).read_text())
    info.update(class_names=names, num_classes=4)
    (d / MODEL_INFO_FILENAME).write_text(json.dumps(info))
    with pytest.raises(ModelNotReadyError, match="outputs"):
        load_model_bundle(d, allow_test_fixture=True)


def test_labels_info_disagreement_rejected(tmp_path):
    d = make_fixture_model_dir(tmp_path / "m")
    (d / CLASS_NAMES_FILENAME).write_text(json.dumps(["a___1", "b___2", "c___3"]))
    with pytest.raises(ModelNotReadyError, match="does not match"):
        load_model_bundle(d, allow_test_fixture=True)


def test_corrupt_model_file_rejected(tmp_path):
    d = make_fixture_model_dir(tmp_path / "m")
    (d / "plant_disease_model.keras").write_bytes(b"corrupt")
    with pytest.raises(ModelNotReadyError, match="could not be loaded"):
        load_model_bundle(d, allow_test_fixture=True)


def test_manager_picks_up_model_dropped_in_later(tmp_path, fixture_model_dir):
    import shutil
    target = tmp_path / "models"
    mgr = ModelManager(target, allow_test_fixture=True)
    assert mgr.status()["ready"] is False
    shutil.copytree(fixture_model_dir, target)
    assert mgr.status()["ready"] is True
