"""Copy genuine example leaf images for the demonstration into demo_images/.

Only images from the held-out TEST split are used (they were never seen in training),
taken from the real dataset on your machine. Nothing is generated or edited.

  python scripts/pick_demo_images.py            (1 healthy + 2 diseased)
  python scripts/pick_demo_images.py --healthy 2 --diseased 3
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.prepare_dataset import resolve_dataset_location  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--healthy", type=int, default=1)
    ap.add_argument("--diseased", type=int, default=2)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    manifest = Path(a.results_dir) / "split_manifest.csv"
    if not manifest.is_file():
        print("ERROR: results/split_manifest.csv not found. Train a model first (it is written by training/train.py).")
        return 1
    root, _ = resolve_dataset_location(ROOT, None, None)
    df = pd.read_csv(manifest)
    test = df[df["split"] == "test"].copy()
    test["healthy"] = test["class_name"].str.lower().str.endswith("___healthy")
    rng = random.Random(a.seed)
    out = ROOT / "demo_images"
    out.mkdir(exist_ok=True)
    picks = []
    for flag, n in ((True, a.healthy), (False, a.diseased)):
        pool = test[test["healthy"] == flag]
        classes = sorted(pool["class_name"].unique())
        rng.shuffle(classes)
        for cls in classes[:n]:
            picks.append(rng.choice(list(pool.loc[pool["class_name"] == cls, "rel_path"])))
    for rel in picks:
        cls = Path(rel).parent.name
        dest = out / f"{cls.replace(' ', '_')}__{Path(rel).name.replace(' ', '_')}"
        shutil.copy2(root / rel, dest)
        print(f"copied {dest.name}   (true label: {cls})")
    print(f"\n{len(picks)} genuine test-split images are in {out}")
    print("The true label is shown only so YOU know the ground truth; the app does not see it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
