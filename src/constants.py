"""Constants shared by training, evaluation and inference.

Keeping them in one dependency-free module guarantees that the local scripts, the
Google Colab notebook and the Flask app all use the same preprocessing/architecture.
"""

IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Identifies the exact preprocessing contract. The saved model takes RGB pixels in the
# range 0..255 (float32); resizing happens in preprocessing.py and the rescale to
# [-1, 1] (MobileNetV2 convention) is a layer INSIDE the saved model.
PREPROCESSING_ID = "tf-resize224-bilinear-antialias__in-model-rescale-1to1__v1"
ARCHITECTURE_ID = "mobilenetv2_1.00_224__gap_dropout_dense_softmax__v1"

MODEL_FILENAME = "plant_disease_model.keras"
CLASS_NAMES_FILENAME = "class_names.json"
MODEL_INFO_FILENAME = "model_info.json"
INFO_SCHEMA_VERSION = 1

DEFAULT_SEED = 42
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / validation / test
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

DATASET_NAME = "PlantVillage"
DATASET_REPO_URL = "https://github.com/spMohanty/PlantVillage-Dataset.git"
