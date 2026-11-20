# Plant Disease Classification using Deep Learning

A Flask web application that classifies plant-leaf photos with a **genuinely trained** MobileNetV2 (transfer learning) model
on the **PlantVillage** dataset. There are no fake predictions, no demo mode and no dummy model: if no trained model is
installed, the app says so and shows how to get one.

> **Important - this ZIP does not contain a trained model.** A real model has to be trained on the real dataset (many hours on a CPU; far
> faster on a GPU, for example in Colab - timings were not measured for this package). Follow *Step 4* below. Until then the app starts and shows setup instructions only.

---

## 1. What you need

| Item | Requirement |
|---|---|
| OS | Windows 10/11 (administrator rights are **not** needed) |
| Python | **3.12 (64-bit)**. Download from <https://www.python.org/downloads/windows/> -> "Install for me only", tick *Add python.exe to PATH* |
| Disk | about 6 GB free (TensorFlow and its dependencies, the ~1 GB dataset, working space) |
| Internet | for installing packages, downloading the dataset and the ImageNet MobileNetV2 weights |
| Editor | Visual Studio Code (only to open the folder and use its terminal) |

Why Python 3.12? The pinned `numpy 2.5.4` requires 3.12 or newer, and TensorFlow 2.21 publishes Windows wheels up to 3.13.
3.12 is the version this project was tested on.

## 2. Quick start on Windows (VS Code)

1. Extract `Plant_Disease_Classification_Project.zip` to a normal folder (e.g. `Documents`).
2. In VS Code: **File > Open Folder...** > choose the extracted `Plant-Disease-Classification` folder.
3. Open the terminal: **Terminal > New Terminal**.
4. Check Python:  `py -3.12 --version`  (should print `Python 3.12.x`).
5. Run setup: `scripts\setup_windows.bat`
6. Prepare the dataset and train **or** train on Colab (Step 4).
7. Start the app: `scripts\run_windows.bat` and open <http://127.0.0.1:5000>.
8. Upload a genuine leaf photo on **Detect Disease**.

### PowerShell alternative (instead of the .bat files)

```powershell
cd Plant-Disease-Classification
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python training\prepare_dataset.py --source auto
python training\train.py --epochs 10 --fine-tune-epochs 5
python app.py                          # then open http://127.0.0.1:5000
```

## 3. Dataset (PlantVillage)

Colour images of healthy and diseased leaves, one folder per class (38 classes, 14 crops, about 54,300 images).

**Method A - automatic download** (`scripts\train_windows.bat` does this for you, or run it yourself):

```
python training\prepare_dataset.py --source auto
```

It downloads only `raw/color` (about 0.9 GB) from <https://github.com/spMohanty/PlantVillage-Dataset> using `git` (sparse
checkout) and falls back to the repository ZIP if git is missing. Git for Windows: <https://git-scm.com/download/win>.
After downloading it **verifies** the folder structure, class folders, image counts and that every image opens, then writes
`dataset/dataset_info.json` and (in `results/`) `dataset_stats.json`, `dataset_class_distribution.png` and
`dataset_samples.png` (real sample images). If anything fails it stops and prints the manual instructions.

**Method B - manual dataset** (ZIP or extracted folder, one folder per class; `train/`/`valid/` layouts are merged):

```
python training\prepare_dataset.py --source zip    --zip    "C:\path\to\dataset.zip"
python training\prepare_dataset.py --source folder --folder "C:\path\to\extracted_folder"
```

**Source and licence.** PlantVillage: Hughes & Salathé (2015), arXiv:1511.08060; Mohanty, Hughes & Salathé
(2016), *Frontiers in Plant Science* 7:1419. The GitHub mirror contains no LICENSE file, so check the terms before
redistributing the data; this project does **not** ship the dataset. Please cite the papers in your report.

### Data leakage (important for honest results)
Many PlantVillage photos show the *same physical leaf*. Splitting images randomly would put near-duplicates in both train
and test and inflate accuracy. Here, images are first grouped by (1) leaf id from the repository's `leaf-map.json`
(about three quarters of the images), (2) original-photo id (also joins `...copy` variants) and (3) identical file content (MD5). Whole groups are
assigned to train / validation / test (70 / 15 / 15), stratified per class, and a check aborts if any group spans two
splits. **Limitation:** images without a leaf-map entry are grouped only by photo id, so some same-leaf leakage can remain;
test accuracy on this dataset is therefore still likely to be optimistic compared with real field photos.

## 4. Get a trained model (choose ONE workflow)

Both workflows run the **same code**: `src/model_builder.py` (architecture), `src/preprocessing.py` (resize), `training/train.py`.
Architecture: augmentation (training only) -> Rescaling to [-1, 1] -> MobileNetV2 (ImageNet weights) -> global average pooling
-> dropout -> softmax. Input 224x224. Loss: sparse categorical cross-entropy; optimiser: Adam; class weights; early stopping,
checkpointing, learning-rate reduction; optional fine-tuning of the top MobileNetV2 layers.

### Workflow A - local training on Windows

```
scripts\train_windows.bat                       (default: 10 epochs + 5 fine-tune epochs)
scripts\train_windows.bat --epochs 5 --fine-tune-epochs 0
```

* Training on a CPU over all ~54k images takes many hours. TensorFlow has had **no native Windows GPU support since 2.11**
  (use Colab, or WSL2 for GPU). The script prints whether a GPU was detected.
* For a quick, *clearly labelled* trial you can limit the data, e.g. `--max-per-class 300`. The model is then marked
  `training_mode: subset` in `model_info.json` and the app shows a warning banner on every result. Do not present that as
  the final model.
* The script never falls back to random weights silently: if the ImageNet weights cannot be downloaded it stops with an
  error. (`--weights none` exists only for deliberate pipeline checks and is recorded in `model_info.json`.)
* Model files are published to `models/` **only after** training finished and the saved model was evaluated on the test
  split: `plant_disease_model.keras`, `class_names.json`, `model_info.json`. Figures/metrics go to `results/`.

### Workflow B - Google Colab (GPU, recommended)

1. Upload `colab/Plant_Disease_Classification_Training.ipynb` to <https://colab.research.google.com> (File > Upload notebook).
2. `Runtime > Change runtime type > T4 GPU`, then run all cells top to bottom (optionally keeps results in Google Drive).
3. The last cell downloads `plant_disease_model_files.zip`. Unzip it and copy **three files** into the project's `models\` folder:
   `plant_disease_model.keras`, `class_names.json`, `model_info.json`.
4. Optional: copy `training_curves.png`, `confusion_matrix.png`, `evaluation_results.json`, `dataset_stats.json`,
   `dataset_samples.png`, `dataset_class_distribution.png` from the Colab `output/results` folder into the project's
   `results\` folder so the **About** page and the demonstration can show them.

The notebook embeds the repository's source files (generated by `scripts/build_notebook.py`; a test verifies they are
identical). **Model compatibility check:** on load, the app verifies the preprocessing and architecture identifiers, input size,
class list, output size and runs a forward-pass smoke test. An incompatible or incomplete model is rejected with a clear
message; nothing is substituted. If Colab's Keras minor version differs from your local one, a warning is shown.

## 5. Run the web app

```
scripts\run_windows.bat            (or: python app.py)
```
Open <http://127.0.0.1:5000>. Pages: **Home**, **Detect Disease**, **About Project** (dataset statistics, model provenance,
evaluation metrics and figures when available). A model dropped into `models\` is picked up without restarting.
JSON endpoint for testing: `POST /api/predict` (field `image`); status: `GET /status`.

What the app does with an upload: validates extension, size (8 MB), real image decoding (JPEG/PNG only), dimensions and
blank images; processes the image **in memory** (never saved to disk, so there is nothing to clean up and no path-traversal
surface); resizes to 224x224; runs the model; shows the top class, status (healthy/diseased from the class name), softmax
score, top-3 classes, a class description and cautious general guidance. Scores below 0.60 are reported as
*inconclusive* with no diagnosis. The score is **not a calibrated probability** (no calibration was implemented). The
guidance is educational and contains no pesticide prescriptions.

Not implemented on purpose: user accounts, a database and prediction history (they would add complexity and store user
images). OpenCV is not used (Pillow + TensorFlow ops are sufficient and keep installation smaller).

## 6. Tests

```
python -m pytest -q
```
Covers preprocessing, upload validation, routes, security headers, missing-model behaviour, the model-compatibility
checks, leakage-free splitting, the Colab notebook (valid JSON, compiles, embeds the same code) and inference. Tests that
need a *model* use tiny throw-away fixtures created inside pytest temp folders (flagged `is_test_fixture`, refused by the
production loader). `tests/test_production_model.py` exercises the real model in `models/` and is **skipped** until you
install one.

## 7. Demonstration

See `docs/DEMONSTRATION_GUIDE.md`. `python scripts\pick_demo_images.py` copies genuine **test-split** images (never seen
in training) to `demo_images\`.

## 8. Project structure

| Path | Purpose |
|---|---|
| `app.py`, `config.py` | Flask app factory, routes, security headers; configuration |
| `src/constants.py` | Shared constants (image size, preprocessing/architecture ids, file names) |
| `src/preprocessing.py` | Upload validation + the single resize function used in training and inference |
| `src/model_builder.py` | MobileNetV2 transfer-learning model, freezing / fine-tuning helpers |
| `src/dataset_utils.py` | Dataset discovery/verification, leakage-aware splits, tf.data pipeline, class weights |
| `src/dataset_download.py` | Automatic PlantVillage download (git sparse checkout, ZIP fallback) |
| `src/class_info.py` | Class-name parsing, cautious descriptions and general guidance |
| `src/model_loader.py`, `src/predictor.py` | Model loading + compatibility checks; real inference |
| `training/prepare_dataset.py` | Prepare + verify dataset (auto / zip / folder), statistics and figures |
| `training/train.py`, `training/evaluate.py` | Training pipeline; test-split evaluation, curves, confusion matrix |
| `colab/Plant_Disease_Classification_Training.ipynb` | Colab workflow (generated by `scripts/build_notebook.py`) |
| `scripts/*.bat` | Windows setup / train / run scripts; `download_dataset.py`, `pick_demo_images.py` |
| `templates/`, `static/` | Pages, CSS (green theme, responsive, dark-mode aware), JavaScript |
| `tests/` | Automated tests |
| `docs/DEMONSTRATION_GUIDE.md` | 5-10 minute faculty demonstration + viva notes |
| `models/`, `dataset/`, `results/` | Empty (`.gitkeep`); filled by you. Git-ignored. |

## 9. Limitations

* Only the trained classes can be recognised. Other plants/diseases/non-leaf photos still receive *some* class unless the
  score is low; no out-of-distribution detector is implemented.
* PlantVillage photos are lab-style (uniform background). Accuracy on field photos is usually much lower.
* Softmax scores are uncalibrated and often over-confident.
* Residual leakage is possible for images without leaf-map entries (see Section 3).
* Not a diagnostic tool; confirm with an agricultural expert.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `Python 3.12 is required` | Install Python 3.12 (64-bit) and re-run `scripts\setup_windows.bat` |
| `pip` fails on TensorFlow | Check you use 64-bit Python 3.12; enable Windows long paths if the error mentions path length; retry |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, or use the `.bat` scripts |
| Port 5000 busy | `set PORT=5050` then `python app.py` |
| App shows "Trained model required" | Train (Step 4) or copy the 3 model files into `models\`; the page states the exact reason |
| "preprocessing/architecture does not match" | The model was trained with a different version of this project; retrain with this code |
| Dataset download fails | Use Method B (manual ZIP/folder) |
| Training stops with an ImageNet-weights error | Connect to the internet (weights are fetched from storage.googleapis.com) and retry |
| Training killed / out of memory | Lower `--batch-size` (e.g. 16 or 8) |

## 11. Verification status of this package

See `VERIFICATION_REPORT.md` for exactly what was executed when this package was built, including what was **not** run
(training of the real model, Windows `.bat` scripts, Colab).
