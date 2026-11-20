"""Automatic PlantVillage download (Method A).

Source : https://github.com/spMohanty/PlantVillage-Dataset (colour images in raw/color)
Paper  : Hughes & Salathe (2015), arXiv:1511.08060; Mohanty, Hughes & Salathe (2016),
         Frontiers in Plant Science 7:1419, doi:10.3389/fpls.2016.01419
Licence: the repository ships no LICENSE file. Treat the data as research material,
         cite the papers, and do NOT redistribute it inside your own ZIP.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .constants import DATASET_REPO_URL

ZIP_URL = "https://codeload.github.com/spMohanty/PlantVillage-Dataset/zip/refs/heads/master"
MANUAL_HELP = """\
Automatic download failed. Provide the dataset manually instead:
  1. Obtain PlantVillage (colour images, one folder per class, 38 classes), e.g. from
     https://github.com/spMohanty/PlantVillage-Dataset  (folder raw/color)
     or a Kaggle copy of PlantVillage.
  2. Then run ONE of:
       python training/prepare_dataset.py --source zip    --zip   path\\to\\dataset.zip
       python training/prepare_dataset.py --source folder --folder path\\to\\extracted\\folder
"""


class DatasetDownloadError(RuntimeError):
    pass


def _count_class_dirs(color_dir: Path) -> int:
    return sum(1 for d in color_dir.iterdir() if d.is_dir()) if color_dir.is_dir() else 0


def _download_with_git(dest: Path) -> Path:
    if shutil.which("git") is None:
        raise DatasetDownloadError("git is not installed.")
    repo = dest / "PlantVillage-Dataset"
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    if not repo.exists():
        cmd = ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", DATASET_REPO_URL, str(repo)]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise DatasetDownloadError(f"git clone failed: {r.stderr.strip()[-300:]}")
    r = subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "raw/color"],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise DatasetDownloadError(f"git sparse-checkout failed: {r.stderr.strip()[-300:]}")
    return repo


def _download_with_zip(dest: Path) -> Path:
    tmp = Path(tempfile.mkstemp(suffix=".zip", dir=dest)[1])
    try:
        print("Downloading repository ZIP (large, can take a while)...")
        with urllib.request.urlopen(ZIP_URL, timeout=60) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out, 1 << 20)
        if not zipfile.is_zipfile(tmp):
            raise DatasetDownloadError("The downloaded ZIP is incomplete or corrupted.")
        repo = dest / "PlantVillage-Dataset"
        with zipfile.ZipFile(tmp) as z:
            for m in z.namelist():
                parts = m.split("/", 1)
                if len(parts) < 2:
                    continue
                rel = parts[1]
                if not (rel.startswith("raw/color/") or rel == "leaf-map.json"):
                    continue
                target = (repo / rel).resolve()
                if not str(target).startswith(str(repo.resolve())):
                    raise DatasetDownloadError("Unsafe path in ZIP.")
                if m.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(m) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        return repo
    finally:
        tmp.unlink(missing_ok=True)


def download_plantvillage(dest) -> tuple[Path, Path | None]:
    """Download PlantVillage colour images. Returns (class_root, leaf_map_path_or_None).

    Never assumes success: the result is checked for class folders before returning.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    errors = []
    for name, fn in (("git sparse checkout", _download_with_git), ("ZIP download", _download_with_zip)):
        try:
            repo = fn(dest)
            color = repo / "raw" / "color"
            n = _count_class_dirs(color)
            if n < 30:
                raise DatasetDownloadError(f"Expected ~38 class folders, found {n}.")
            leaf_map = repo / "leaf-map.json"
            return color, (leaf_map if leaf_map.is_file() else None)
        except Exception as exc:  # try the next method
            errors.append(f"{name}: {exc}")
            print(f"[{name}] failed: {exc}")
    raise DatasetDownloadError("; ".join(errors) + "\n" + MANUAL_HELP)
