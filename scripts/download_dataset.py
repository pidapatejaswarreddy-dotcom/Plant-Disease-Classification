"""Convenience wrapper: download + verify PlantVillage (same as prepare_dataset.py --source auto)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.prepare_dataset import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["--source", "auto"] + sys.argv[1:]))
