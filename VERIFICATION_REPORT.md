# Verification report

Written when this package was built. It lists what was actually executed, what was not, and the known problems.
No metric below describes a deliverable model: **no real trained model is included** in this package.

## Environment used for verification
Linux sandbox (1 CPU core, 4 GB RAM, no GPU), Python 3.12.3, tensorflow-cpu 2.21.0, keras 3.15.1, numpy 2.5.4, pandas 3.0.6,
scikit-learn 1.9.1, matplotlib 3.11.2, Pillow 13.0.0, Flask 3.1.3, pytest 9.2.0. Packages were installed into the system
Python (not a fresh virtual environment). `pip install --dry-run -r requirements.txt` resolved without conflicts.
`requirements.txt` pins `tensorflow` (the CPU build on Windows); the sandbox used `tensorflow-cpu` of the same version.
PyPI metadata confirms Windows (win_amd64) wheels exist for tensorflow 2.21.0 (cp310-cp313), numpy 2.5.4 (cp312+),
pandas, scikit-learn, matplotlib and Pillow for cp312.

## Completed and tested
| Item | Result |
|---|---|
| Syntax / import of all Python files | OK (every test imports them) |
| Automated tests (`python -m pytest`) | **59 passed, 2 skipped** (the 2 skipped are `tests/test_production_model.py`: no genuine model installed) |
| Production-inference tests with the smoke-run model (`PDC_MODEL_DIR=/tmp/smoke/models`) | 2 passed (loader + Flask `/api/predict` on a genuine PlantVillage test-split image) |
| Real architecture save -> load -> production compatibility check -> predict (untrained weights, in tests) | passed |
| Colab notebook | valid nbformat-4 JSON; all code cells parse; embedded sources identical to the repository; embedded `train.py --help` runs from a clean Colab-style layout |
| Dataset: automatic download (git sparse checkout) | worked: 54,305 images in 38 classes verified readable in ~36 s (README of the source says 54,306) |
| Dataset: manual ZIP (nested train/valid layout, safe extraction) and manual folder methods | worked; ZIP-slip and corrupt ZIP rejected |
| Leakage-aware split | leaf-map grouping used (446 of 608 images in the smoke run had a leaf-map entry); no group spans two splits |
| Failure modes | missing dataset -> clean error, exit 1; ImageNet weights unreachable -> clean error, exit 1, **no model files published**, no silent fallback |
| Web app (Flask dev server, real HTTP) | routes 200/404 as intended; real JPEG uploads classified; wrong type / fake image -> 400; 9 MB -> 413; nothing written to disk; no tracebacks in the server log |
| Browser test (headless Chromium) | file validation, preview, reset, submit/result flow, About-page figures load, no horizontal overflow at 390 px, no CSP violations |
| Missing-model behaviour | setup page (HTTP 503) and JSON error; never a prediction |

## Pipeline-verification training run (NOT a usable model)
To prove that training -> evaluation -> saving -> serving works end to end, one real run was executed with
`--weights none --max-per-class 16 --epochs 3 --batch-size 8` (random initial weights, 608 images, split 380 / 114 / 114, CPU,
288 s, about 66-81 s per epoch). It is recorded in its own `model_info.json` as `training_mode: subset`,
`pretrained_weights: none`. Result on its 114 test images: accuracy 0.0263 (about chance for 38 classes), top-3 0.0789,
macro-F1 0.0013. **The network collapsed to a constant output** (verified: identical outputs for different images and for random
input; base features constant), which is expected for a randomly initialised MobileNetV2 trained for only 144 steps. The app
correctly reported every result as *inconclusive* and displayed the subset warning. This model was **not** packaged.

## NOT executed (cannot be claimed as working)
* **Training of the real model** (ImageNet transfer learning on the full dataset). ImageNet weights are blocked in the build
  sandbox (storage.googleapis.com) and one CPU core is not enough. Therefore: no real accuracy figures exist yet,
  stage-2 fine-tuning was only unit-tested (freezing logic), and the app's confident-result display was tested with a test
  fixture, not with a genuine model.
* **Windows**: the `.bat` scripts and the Windows installation were not run. The scripts were written with CRLF line endings and
  reviewed by hand only.
* **Google Colab**: the notebook was validated structurally; it was not executed on Colab.
* **GPU** training and Python versions other than 3.12.
* The ZIP-download fallback of `src/dataset_download.py` against the real repository: in the sandbox the repository ZIP stream
  ended early (~718 MB, no valid archive), so the code path that detects an incomplete ZIP was relevant but the full-download
  success path is unverified. The git path (preferred) is verified.

## Known limitations
See README section 9 (limited classes, lab-style images, uncalibrated scores, possible residual leakage, no OOD detection).
