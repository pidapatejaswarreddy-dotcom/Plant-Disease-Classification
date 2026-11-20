"""Dataset discovery, verification, leakage-aware splitting and tf.data pipelines."""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

from .constants import DEFAULT_SEED, IMAGE_EXTENSIONS, SPLIT_FRACTIONS

SKIP_DIR_NAMES = {"__macosx", ".git", ".ipynb_checkpoints"}
CONTAINER_DIR_NAMES = {"train", "valid", "val", "validation", "test"}


class DatasetError(RuntimeError):
    """Raised when the dataset is missing, malformed or unreadable."""


# --------------------------------------------------------------------------- discovery
def discover_images(root) -> pd.DataFrame:
    """Return a DataFrame(rel_path, class_name) for every image below ``root``.

    The class is the name of the image's parent folder. Folders such as train/valid/test
    are treated as containers, so the same class found in several of them is merged.
    """
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(f"Dataset folder not found: {root}")
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = p.relative_to(root)
        if any(part.lower() in SKIP_DIR_NAMES or part.startswith(".") for part in rel.parts[:-1]):
            continue
        parent = p.parent.name
        if p.parent == root or parent.lower() in CONTAINER_DIR_NAMES:
            continue  # image not inside a class folder
        rows.append((rel.as_posix(), parent))
    return pd.DataFrame(rows, columns=["rel_path", "class_name"])


def verify_dataset(root, check_readable: bool = True, min_classes: int = 2,
                   min_images_per_class: int = 10) -> dict:
    """Verify structure, counts and (optionally) that every image opens. Raises DatasetError."""
    df = discover_images(root)
    if df.empty:
        raise DatasetError(f"No class folders with JPG/PNG images were found under {root}.")
    counts = df["class_name"].value_counts().sort_index()
    if len(counts) < min_classes:
        raise DatasetError(f"Only {len(counts)} class folder(s) found; expected a folder per class.")
    unreadable = []
    if check_readable:
        for rel in df["rel_path"]:
            try:
                with Image.open(Path(root) / rel) as im:
                    im.verify()
            except Exception:
                unreadable.append(rel)
    small = counts[counts < min_images_per_class]
    report = {
        "root": str(root),
        "num_classes": int(len(counts)),
        "num_images": int(len(df)),
        "images_per_class": {k: int(v) for k, v in counts.items()},
        "min_images_in_a_class": int(counts.min()),
        "max_images_in_a_class": int(counts.max()),
        "readability_checked": bool(check_readable),
        "unreadable_files": unreadable,
        "classes_below_minimum": {k: int(v) for k, v in small.items()},
    }
    if unreadable:
        report["warning"] = (f"{len(unreadable)} unreadable image(s) were found and will be "
                             "excluded from training.")
    if len(small):
        raise DatasetError(f"Classes with fewer than {min_images_per_class} images: {list(small.index)}")
    return report


# --------------------------------------------------------------------------- grouping
_UUID_PREFIX = re.compile(r"^[0-9a-fA-F\-]{36}___")


def image_identifier(filename: str) -> str:
    """Original-photo identifier, following the PlantVillage repository's own convention."""
    name = filename.replace("_final_masked", "")
    if "___" in name:
        name = name.split("___")[-1]
    name = name.split("copy")[0]
    name = re.sub(r"\.(jpg|jpeg|png)$", "", name, flags=re.I)
    return name.strip().lower()


def load_leaf_map(path):
    if path and Path(path).is_file():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def assign_groups(df: pd.DataFrame, root, leaf_map: dict | None = None,
                  hash_files: bool = True) -> tuple[pd.DataFrame, dict]:
    """Add a ``group`` column. Images that share a physical leaf (via leaf-map.json), an
    original-photo identifier, or identical bytes (MD5) get the same group and therefore
    always land in the same split."""
    uf = _UnionFind()
    info = {"leaf_map_used": leaf_map is not None, "leaf_map_hits": 0, "md5_hashing": hash_files}
    keys = []
    for rel, cls in zip(df["rel_path"], df["class_name"]):
        fname = Path(rel).name
        ident = image_identifier(fname)
        node = f"img::{rel}"
        uf.find(node)
        uf.union(node, f"id::{cls}::{ident}")
        if leaf_map:
            options = leaf_map.get(ident)
            if options:
                pick = next((o for o in options if cls in o), None) if len(options) > 1 else options[0]
                if pick:
                    uf.union(node, f"leaf::{pick}")
                    info["leaf_map_hits"] += 1
        keys.append(node)
    if hash_files:
        seen = {}
        for rel in df["rel_path"]:
            digest = _file_md5(Path(root) / rel)
            if digest in seen:
                uf.union(f"img::{rel}", f"img::{seen[digest]}")
            else:
                seen[digest] = rel
    out = df.copy()
    out["group"] = [uf.find(k) for k in keys]
    info["num_groups"] = int(out["group"].nunique())
    return out, info


# --------------------------------------------------------------------------- splitting
def subsample_per_class(df: pd.DataFrame, max_per_class: int, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    parts = [g.sample(n=min(len(g), max_per_class), random_state=seed)
             for _, g in df.groupby("class_name")]
    return pd.concat(parts).sort_values("rel_path").reset_index(drop=True)


def make_splits(df: pd.DataFrame, fractions=SPLIT_FRACTIONS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Stratified, group-aware train/val/test split. Needs a ``group`` column."""
    rng = random.Random(seed)
    names = ["train", "val", "test"]
    split_of_group = {}
    for cls, g in df.groupby("class_name"):
        sizes = g.groupby("group").size()
        groups = sorted(sizes.index)
        if len(groups) < 3:
            raise DatasetError(f"Class '{cls}' has only {len(groups)} independent group(s); need >= 3.")
        rng.shuffle(groups)
        groups.sort(key=lambda k: -sizes[k])  # place big groups first, ties stay shuffled
        total = int(sizes.sum())
        targets = [total * f for f in fractions]
        assigned = [0, 0, 0]
        for k in groups:
            deficits = [(targets[i] - assigned[i]) / targets[i] for i in range(3)]
            i = max(range(3), key=lambda j: deficits[j])
            split_of_group[k] = names[i]
            assigned[i] += int(sizes[k])
    out = df.copy()
    out["split"] = out["group"].map(split_of_group)
    check_no_leakage(out)
    return out


def check_no_leakage(df: pd.DataFrame) -> None:
    per_group = df.groupby("group")["split"].nunique()
    bad = per_group[per_group > 1]
    if len(bad):
        raise DatasetError(f"Leakage: {len(bad)} group(s) appear in more than one split.")
    if df["split"].isna().any():
        raise DatasetError("Some images were not assigned to a split.")


# --------------------------------------------------------------------------- tf.data
def compute_class_weights(labels, num_classes: int) -> dict:
    counts = pd.Series(labels).value_counts()
    n = len(labels)
    return {int(c): float(n / (num_classes * counts[c])) for c in counts.index}


def build_tf_dataset(paths, labels, batch_size: int, training: bool, seed: int = DEFAULT_SEED):
    import tensorflow as tf
    from .preprocessing import tf_load_image
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if training:
        ds = ds.shuffle(min(len(paths), 4096), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (tf_load_image(p), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
