import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import dataset_utils as du
from src.class_info import get_class_details, parse_class_name


def make_dataset(root, classes=("Tomato___healthy", "Tomato___Early_blight"), n=30, duplicate_pairs=3):
    rng = np.random.default_rng(0)
    for c in classes:
        d = root / c
        d.mkdir(parents=True)
        for i in range(n):
            arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i:08d}-aaaa-bbbb-cccc-000000000000___X_{c[:3]} {i}.JPG")
        for i in range(duplicate_pairs):  # byte-identical copies under another name
            (d / f"dup{i}.JPG").write_bytes((d / f"{i:08d}-aaaa-bbbb-cccc-000000000000___X_{c[:3]} {i}.JPG").read_bytes())
    return root


def test_discover_and_verify(tmp_path):
    root = make_dataset(tmp_path)
    rep = du.verify_dataset(root)
    assert rep["num_classes"] == 2 and rep["num_images"] == 66 and rep["unreadable_files"] == []


def test_verify_detects_unreadable_and_missing(tmp_path):
    root = make_dataset(tmp_path)
    (root / "Tomato___healthy" / "broken.jpg").write_bytes(b"not a jpeg")
    assert du.verify_dataset(root)["unreadable_files"] == ["Tomato___healthy/broken.jpg"]
    with pytest.raises(du.DatasetError):
        du.verify_dataset(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(du.DatasetError):
        du.verify_dataset(tmp_path / "empty")


def test_split_has_no_leakage_and_is_deterministic(tmp_path):
    root = make_dataset(tmp_path)
    df = du.discover_images(root)
    df, info = du.assign_groups(df, root)
    a = du.make_splits(df, seed=1)
    b = du.make_splits(df, seed=1)
    assert a["split"].tolist() == b["split"].tolist()
    assert (a.groupby("group")["split"].nunique() == 1).all()
    # byte-identical duplicates share a group, hence a split
    for c in ("Tomato___healthy", "Tomato___Early_blight"):
        for i in range(3):
            orig = a[a.rel_path.str.endswith(f"{i:08d}-aaaa-bbbb-cccc-000000000000___X_{c[:3]} {i}.JPG") & (a.class_name == c)]
            dup = a[a.rel_path == f"{c}/dup{i}.JPG"]
            assert orig["split"].iloc[0] == dup["split"].iloc[0]
    assert set(a["split"]) == {"train", "val", "test"}


def test_leaf_map_groups_images_of_same_leaf(tmp_path):
    root = make_dataset(tmp_path, n=30, duplicate_pairs=0)
    df = du.discover_images(root)
    leaf_map = {f"x_tom {i}": ["Tomato___healthy:::1.0"] for i in range(10)}
    leaf_map.update({f"x_tom {i}".lower(): ["Tomato___healthy:::1.0"] for i in range(10)})
    df, info = du.assign_groups(df, root, leaf_map, hash_files=False)
    healthy = df[(df.class_name == "Tomato___healthy") & df.rel_path.str.contains("X_Tom ")]
    ids = {int(p.split(" ")[-1].split(".")[0]) for p in healthy.rel_path}
    assert info["leaf_map_hits"] >= 10
    same = df[(df.class_name == "Tomato___healthy") & df.rel_path.apply(lambda p: int(p.split(" ")[-1].split(".")[0]) < 10 if "X_Tom " in p else False)]
    assert same["group"].nunique() == 1 and ids


def test_check_no_leakage_raises():
    df = pd.DataFrame({"group": ["g", "g"], "split": ["train", "test"]})
    with pytest.raises(du.DatasetError):
        du.check_no_leakage(df)


def test_class_weights_balance():
    w = du.compute_class_weights([0] * 90 + [1] * 10, 2)
    assert w[1] > w[0]


def test_class_name_parsing_and_details():
    p = parse_class_name("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot")
    assert p["plant"] == "Corn (maize)" and p["status"] == "diseased"
    assert parse_class_name("Tomato___healthy")["status"] == "healthy"
    assert parse_class_name("Pepper,_bell___Bacterial_spot")["plant"] == "Pepper, bell"
    d = get_class_details("Potato___Late_blight")
    assert d["known"] and d["recommendations"]
    unk = get_class_details("Banana___Sigatoka")
    assert unk["known"] is False and "No curated description" in unk["description"]
