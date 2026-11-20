"""Generate colab/Plant_Disease_Classification_Training.ipynb from the repository's own source files.

The notebook embeds the exact files used for local training (src/*.py and training/*.py),
so Colab and Windows training share the same architecture and preprocessing. Run this
after editing any of those files:   python scripts/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "colab" / "Plant_Disease_Classification_Training.ipynb"
EMBEDDED = ["src/constants.py", "src/preprocessing.py", "src/model_builder.py", "src/dataset_utils.py",
            "src/dataset_download.py", "training/prepare_dataset.py", "training/evaluate.py", "training/train.py"]
PROJECT = "/content/pdc"


def lines(text):
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text.strip("\n"))}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines(text.strip("\n"))}


cells = [
    md("""# Plant Disease Classification: Google Colab training

Trains the **same MobileNetV2 transfer-learning model** as the local Windows workflow. The source files embedded below are generated from the project repository (`scripts/build_notebook.py`), so preprocessing and architecture are identical.

**Steps**
1. `Runtime > Change runtime type > T4 GPU` (recommended).
2. Run the cells from top to bottom. Nothing is faked: if a step fails, the notebook stops with the real error.
3. At the end, download `plant_disease_model.keras`, `class_names.json` and `model_info.json` and put them in the web app's `models/` folder.

Dataset: PlantVillage (Hughes & Salathe 2015; Mohanty, Hughes & Salathe 2016), downloaded from GitHub or supplied as a ZIP in your Google Drive. Check the dataset's terms before redistributing it."""),
    code("""import platform, sys
import tensorflow as tf
print("Python", platform.python_version(), "| TensorFlow", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPU devices:", gpus if gpus else "NONE - training will be very slow. Switch the runtime to GPU.")"""),
    md("## 1. Settings"),
    code(f"""# ---- edit these if you like ------------------------------------------------
USE_DRIVE = True                      # save the finished model files to Google Drive too
DRIVE_DIR = "/content/drive/MyDrive/plant_disease_project"
DATASET_SOURCE = "auto"               # "auto" = download from GitHub, "drive_zip" = use DRIVE_ZIP
DRIVE_ZIP = DRIVE_DIR + "/PlantVillage.zip"
EPOCHS = 10                           # head-training epochs (early stopping may stop sooner)
FINE_TUNE_EPOCHS = 5                  # 0 disables fine-tuning
BATCH_SIZE = 32
# -----------------------------------------------------------------------------
PROJECT = "{PROJECT}"
OUT = "/content/output"               # model + results are written here
import os, subprocess, sys
for d in (PROJECT + "/src", PROJECT + "/training", PROJECT + "/dataset", OUT + "/models", OUT + "/results"):
    os.makedirs(d, exist_ok=True)
for init in (PROJECT + "/src/__init__.py", PROJECT + "/training/__init__.py"):
    open(init, "a").close()

def run(cmd):
    \"\"\"Run a command and stream its output; raise if it fails.\"\"\"
    print("$", " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    if p.wait() != 0:
        raise RuntimeError("Command failed with exit code %d" % p.returncode)"""),
    code("""if USE_DRIVE or DATASET_SOURCE == "drive_zip":
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(DRIVE_DIR, exist_ok=True)"""),
    md("## 2. Project source files (generated from the repository; do not edit)"),
]
for rel in EMBEDDED:
    body = (ROOT / rel).read_text(encoding="utf-8")
    cells.append(code(f"%%writefile {PROJECT}/{rel}\n{body.rstrip(chr(10))}"))
cells += [
    md("## 3. Dataset: download (or load from Drive) and verify"),
    code("""args = [sys.executable, "training/prepare_dataset.py", "--results-dir", OUT + "/results",
        "--download-dir", "/content/dataset_download"]
if DATASET_SOURCE == "auto":
    args += ["--source", "auto"]
else:
    args += ["--source", "zip", "--zip", DRIVE_ZIP]
run(args)"""),
    code("""from IPython.display import Image, display
display(Image(OUT + "/results/dataset_samples.png"))
display(Image(OUT + "/results/dataset_class_distribution.png"))"""),
    md("## 4. Train (MobileNetV2 transfer learning, early stopping, checkpointing, optional fine-tuning)"),
    code("""run([sys.executable, "training/train.py",
     "--epochs", str(EPOCHS), "--fine-tune-epochs", str(FINE_TUNE_EPOCHS), "--batch-size", str(BATCH_SIZE),
     "--model-dir", OUT + "/models", "--results-dir", OUT + "/results"])"""),
    md("## 5. Results (computed on the held-out test split by the training script)"),
    code("""import json
res = json.load(open(OUT + "/results/evaluation_results.json"))
print({k: res[k] for k in ("num_test_images", "accuracy", "top3_accuracy", "macro_f1", "weighted_f1")})
display(Image(OUT + "/results/training_curves.png"))
display(Image(OUT + "/results/confusion_matrix.png"))"""),
    md("## 6. Download the model files"),
    code("""import shutil
print(open(OUT + "/models/model_info.json").read()[:600], "...")
archive = shutil.make_archive("/content/plant_disease_model_files", "zip", OUT + "/models")
if USE_DRIVE:
    shutil.copytree(OUT, DRIVE_DIR + "/output", dirs_exist_ok=True)
    print("Copied to Google Drive:", DRIVE_DIR + "/output")
from google.colab import files
files.download(archive)"""),
    md("""## 7. Use the model in the web application
Unzip the download and copy these three files into the project's `models/` folder (same folder as `.gitkeep`):

```
models/plant_disease_model.keras
models/class_names.json
models/model_info.json
```

Optionally copy the figures from `output/results/` (`training_curves.png`, `confusion_matrix.png`, `evaluation_results.json`, ...) into the project's `results/` folder so the About page can show them. Then start the app (`scripts\\run_windows.bat`). The app runs a compatibility check on load and refuses incompatible files; it never substitutes an untrained model."""),
]
nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"name": "Plant_Disease_Classification_Training.ipynb", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
NB_PATH.parent.mkdir(exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", NB_PATH, f"({NB_PATH.stat().st_size // 1024} KB, {len(cells)} cells)")
