"""Prepare and VERIFY the PlantVillage dataset (never assumes a download succeeded).

  python training/prepare_dataset.py --source auto
  python training/prepare_dataset.py --source zip    --zip    C:\\path\\to\\dataset.zip
  python training/prepare_dataset.py --source folder --folder C:\\path\\to\\extracted_folder

Writes dataset/dataset_info.json and, under results/: dataset_stats.json,
dataset_class_distribution.png and dataset_samples.png (real sample images).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import DATASET_NAME, IMAGE_EXTENSIONS  # noqa: E402
from src.dataset_download import DatasetDownloadError, MANUAL_HELP, download_plantvillage  # noqa: E402
from src.dataset_utils import (CONTAINER_DIR_NAMES, SKIP_DIR_NAMES, DatasetError,  # noqa: E402
                               discover_images, verify_dataset)

CITATION = ("Hughes & Salathe (2015) arXiv:1511.08060; Mohanty, Hughes & Salathe (2016) "
            "Frontiers in Plant Science 7:1419, doi:10.3389/fpls.2016.01419")


def _has_images(d: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in d.iterdir())


def find_class_root(folder: Path, max_depth: int = 4) -> Path:
    """Locate the folder that directly holds one sub-folder per class."""
    queue = [(Path(folder), 0)]
    while queue:
        d, depth = queue.pop(0)
        subs = [s for s in d.iterdir() if s.is_dir() and not s.name.startswith(".")
                and s.name.lower() not in SKIP_DIR_NAMES]
        if len([s for s in subs if _has_images(s)]) >= 2:
            return d
        if any(s.name.lower() in CONTAINER_DIR_NAMES for s in subs):
            return d
        if depth < max_depth:
            queue.extend((s, depth + 1) for s in subs)
    raise DatasetError(f"Could not find class folders (one folder per class, containing images) in {folder}.")


def safe_extract(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    base = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad:
            raise DatasetError(f"The ZIP is corrupted (first bad member: {bad}).")
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            if not str(target).startswith(str(base)):
                raise DatasetError("The ZIP contains an unsafe path and was not extracted.")
        z.extractall(dest)


def _rel_or_abs(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_dataset_location(project_root: Path, data_dir: str | None, leaf_map: str | None):
    """Return (dataset_root, leaf_map_path_or_None) from CLI args or dataset/dataset_info.json."""
    info_path = project_root / "dataset" / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    if data_dir:
        root = Path(data_dir)
    elif info.get("dataset_root"):
        root = Path(info["dataset_root"])
        if not root.is_absolute():
            root = project_root / root
    else:
        raise DatasetError("No dataset prepared yet. Run: python training/prepare_dataset.py --source auto")
    lm = leaf_map or info.get("leaf_map")
    lm_path = Path(lm) if lm else None
    if lm_path and not lm_path.is_absolute():
        lm_path = project_root / lm_path
    return root, (lm_path if lm_path and lm_path.is_file() else None)


def write_figures(root: Path, report: dict, results_dir: Path, seed: int = 42) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    counts = report["images_per_class"]
    names = list(counts)
    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.25)))
    ax.barh(names, [counts[n] for n in names], color="#40916c")
    ax.set_xlabel("Number of images")
    ax.set_title(f"{DATASET_NAME}: images per class ({report['num_images']} images, {report['num_classes']} classes)")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(results_dir / "dataset_class_distribution.png", dpi=110)
    plt.close(fig)

    df = discover_images(root)
    rng = random.Random(seed)
    classes = sorted(df["class_name"].unique())
    step = max(1, len(classes) // 24)
    chosen = classes[::step][:24]
    cols = 6
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, cls in zip(axes.ravel(), chosen):
        rel = rng.choice(list(df.loc[df["class_name"] == cls, "rel_path"]))
        with Image.open(root / rel) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(cls.replace("___", "\n").replace("_", " ")[:34], fontsize=6)
    fig.suptitle(f"Sample images from the dataset ({len(chosen)} of {len(classes)} classes)")
    fig.tight_layout()
    fig.savefig(results_dir / "dataset_samples.png", dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["auto", "zip", "folder"], required=True)
    ap.add_argument("--zip", help="path to a dataset ZIP (with --source zip)")
    ap.add_argument("--folder", help="path to an extracted dataset folder (with --source folder)")
    ap.add_argument("--leaf-map", help="optional leaf-map.json (leaf-level grouping to avoid split leakage)")
    ap.add_argument("--download-dir", default=str(PROJECT_ROOT / "dataset" / "_download"))
    ap.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--skip-readability-check", action="store_true")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    leaf_map = Path(args.leaf_map) if args.leaf_map else None
    try:
        if args.source == "auto":
            root, found_map = download_plantvillage(args.download_dir)
            leaf_map = leaf_map or found_map
        elif args.source == "zip":
            if not args.zip or not Path(args.zip).is_file():
                raise DatasetError("Please pass --zip PATH to an existing ZIP file.")
            dest = PROJECT_ROOT / "dataset" / "_extracted" / Path(args.zip).stem
            print(f"Extracting {args.zip} ...")
            safe_extract(Path(args.zip), dest)
            root = find_class_root(dest)
            leaf_map = leaf_map or next(iter(dest.rglob("leaf-map.json")), None)
        else:
            if not args.folder or not Path(args.folder).is_dir():
                raise DatasetError("Please pass --folder PATH to an existing folder.")
            root = find_class_root(Path(args.folder))
            leaf_map = leaf_map or next(iter(Path(args.folder).rglob("leaf-map.json")), None)
        print(f"Dataset root: {root}\nVerifying structure, counts and image readability ...")
        report = verify_dataset(root, check_readable=not args.skip_readability_check)
    except (DatasetError, DatasetDownloadError) as exc:
        print(f"\nERROR: {exc}\n\n{MANUAL_HELP}", file=sys.stderr)
        return 1

    stats = dict(report, dataset=DATASET_NAME, source_method=args.source, citation=CITATION,
                 verified_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 leaf_map_available=bool(leaf_map))
    (results_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    info = {"dataset_root": _rel_or_abs(Path(root)), "leaf_map": _rel_or_abs(Path(leaf_map)) if leaf_map else None,
            "method": args.source, "verified_at": stats["verified_at"],
            "num_classes": report["num_classes"], "num_images": report["num_images"]}
    (PROJECT_ROOT / "dataset").mkdir(exist_ok=True)
    (PROJECT_ROOT / "dataset" / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    write_figures(Path(root), report, results_dir)
    print(f"\nVerified {report['num_images']} images in {report['num_classes']} classes "
          f"(per-class min {report['min_images_in_a_class']}, max {report['max_images_in_a_class']}).")
    if report["unreadable_files"]:
        print(f"WARNING: {len(report['unreadable_files'])} unreadable file(s) will be excluded.")
    if not leaf_map:
        print("NOTE: leaf-map.json not found; splits will group by photo id and identical files only "
              "(see README, 'Data leakage').")
    print("Wrote dataset/dataset_info.json and results/dataset_*.{json,png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
