"""MobileNetV2 transfer-learning model (used by local training AND the Colab notebook)."""
from __future__ import annotations

from .constants import IMG_SHAPE


def build_model(num_classes: int, weights: str | None = "imagenet", dropout: float = 0.2,
                augment: bool = True):
    """Return an uncompiled Keras model.

    Input : float32 RGB, 0..255, shape (224, 224, 3)
    Layers: [training-only augmentation] -> Rescaling(-1..1) -> MobileNetV2 -> GAP
            -> Dropout -> Dense(softmax)
    Keras random-augmentation layers are inactive at inference, so the saved model is
    deterministic when used for prediction.
    """
    import keras
    from keras import layers

    inputs = keras.Input(shape=IMG_SHAPE, name="image_rgb_0_255")
    x = inputs
    if augment:
        x = layers.RandomFlip("horizontal_and_vertical", name="aug_flip")(x)
        x = layers.RandomRotation(0.1, name="aug_rotation")(x)
        x = layers.RandomZoom(0.1, name="aug_zoom")(x)
        x = layers.RandomContrast(0.1, name="aug_contrast")(x)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale_minus1_to_1")(x)
    base = keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE, include_top=False, weights=weights, alpha=1.0,
        name="mobilenetv2_base")
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32",
                           name="predictions")(x)
    return keras.Model(inputs, outputs, name="plant_disease_mobilenetv2")


def get_base(model):
    return model.get_layer("mobilenetv2_base")


def freeze_base(model) -> None:
    get_base(model).trainable = False


def unfreeze_top_layers(model, fine_tune_at: int = 100) -> int:
    """Unfreeze base layers from index ``fine_tune_at`` on (BatchNorm stays frozen).

    Returns the number of trainable base layers.
    """
    import keras
    base = get_base(model)
    base.trainable = True
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= fine_tune_at and not isinstance(
            layer, keras.layers.BatchNormalization)
    return sum(1 for layer in base.layers if layer.trainable)


def compile_model(model, learning_rate: float) -> None:
    import keras
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
