# Faculty demonstration guide (about 8 minutes)

> **Rule for the demo:** show only what the application and your own training run really produced.
> Do not quote an accuracy figure that is not on the About page / in `results/evaluation_results.json`
> of **your** trained model, and do not show screenshots from anywhere else.

## 0. Before the demo (10 minutes, once)

1. Set up and train (or import from Colab) as described in the README. The app must show a model on the **About** page.
2. Pick genuine example images from the held-out test split (never seen during training):
   ```
   python scripts\pick_demo_images.py --healthy 1 --diseased 2
   ```
   This copies real PlantVillage test images into `demo_images\` and prints their true labels (for your notes).
3. Start the app (`scripts\run_windows.bat`) and open <http://127.0.0.1:5000>.
4. Optional: a real photo of a leaf from a plant you can photograph, to show how field photos behave (see step 8).

## 1. Script

| Min | Step | What to do / say |
|-----|------|------------------|
| 0:00 | **Start & Home page** | Show the app running in the browser. Read the one-line objective. Point out the "How it works" steps and the list of classes the installed model can recognise (read from the model, not hard-coded). |
| 1:00 | **Dataset** | Open **About Project**. Explain PlantVillage: labelled photos of healthy and diseased leaves, one folder per class. Show the dataset statistics (images, classes) and the sample-image grid and class-distribution chart, which were produced from the real files. Mention class imbalance. |
| 2:00 | **CNN / MobileNetV2** | Explain: a CNN learns visual filters; MobileNetV2 is a compact CNN pre-trained on ImageNet; we replace its classifier head with a softmax layer for our classes (transfer learning). Mention input 224x224 and that the rescaling is inside the model. |
| 3:00 | **Training & evaluation graphs** | Show the training/validation curves and the confusion matrix (About page, if present). Say what the numbers on screen are and that they were measured on a held-out test split. Point out where the model still confuses classes. |
| 4:00 | **Healthy leaf** | **Detect Disease** > choose a healthy image from `demo_images\` > preview appears > *Detect disease*. Show the loading state, the predicted plant/status, the score and the top-3 table. |
| 5:00 | **Diseased leaf** | Upload a diseased image. Show the prediction, description and the cautious general guidance. Compare with the true label you noted in step 0. If the model is wrong, say so openly and explain why that can happen. |
| 6:00 | **Robustness** | Upload an unsupported file (e.g. a `.txt` renamed to `.jpg`) to show validation, then a non-leaf photo to show the *Inconclusive* message when the score is low. Explain that a neural network can still be confidently wrong on non-leaf images. |
| 7:00 | **Limitations** | State clearly: only trained classes; lab-style PlantVillage images vs field photos; the score is an uncalibrated softmax value, not a probability of being correct; guidance is educational, not a diagnosis or a pesticide prescription. |
| 8:00 | **Questions** | Use the viva notes below. |

## 2. Using genuine example images

* Use `scripts\pick_demo_images.py` (test-split images from your dataset) or your own leaf photographs.
* Do not use images downloaded from random websites as "ground truth": you cannot verify their labels.
* Do not use training-split images for the demo: the model has seen them, so the result says nothing about generalisation.

## 3. Viva notes (short answers)

**Dataset.** PlantVillage colour images (about 54k) in 38 crop/disease classes (14 crops). Verified for structure, counts and readability before training. Classes are imbalanced (roughly 150 to 5,500 images), so training uses class weights.

**Preprocessing.** Every image is converted to RGB and resized to 224x224 with the same TensorFlow resize operation in training and in the web app. The MobileNetV2 scaling to [-1, 1] is a layer inside the saved model, so training and inference cannot drift apart.

**Avoiding data leakage.** Many PlantVillage photos show the same physical leaf. The split is made on *groups* (leaf ids from the dataset's `leaf-map.json`, original-photo ids, and identical-file hashes), stratified by class, so one leaf never appears in two splits. Remaining limitation: leaves missing from the leaf map are grouped only by photo id, so some leakage can remain.

**Architecture.** Augmentation layers (training only) -> Rescaling -> MobileNetV2 (ImageNet weights) -> global average pooling -> dropout -> dense softmax. Stage 1 trains only the new head; optional stage 2 fine-tunes the top layers at a low learning rate (BatchNorm layers stay frozen).

**Training.** Adam optimiser, sparse categorical cross-entropy, class weights, early stopping on validation loss with best-weights restore, learning-rate reduction on plateau and model checkpointing. The best model is saved as `.keras` together with `class_names.json` and `model_info.json` (provenance: dataset, split sizes, configuration, versions).

**Evaluation.** Performed once on the untouched test split of the *saved* model file: accuracy, top-3 accuracy, macro/weighted F1, per-class report, confusion matrix.

**Inference.** The Flask app validates the upload (type, size, real image decoding, dimensions), resizes it, runs the saved model and shows the top class and softmax score. If the score is below 0.60 the result is reported as inconclusive. Uploads are processed in memory and never stored.

**Why is the confidence not a probability?** Softmax outputs sum to 1 but neural networks are often over-confident; no calibration (e.g. temperature scaling) has been implemented or verified here.

**What would you improve?** Field-image training data, calibration, out-of-distribution (non-leaf) detection, leaf-level cross-validation, and a user-feedback loop reviewed by experts.
