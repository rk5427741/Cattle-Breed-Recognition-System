"""Transfer-learning model construction and threshold calibration."""



from __future__ import annotations



import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



from ml.config import NOT_CATTLE_CLASS, SUPPORTED_BACKBONES

from ml.preprocessing import normalize_breed_key





class BackbonePreprocessLayer(layers.Layer):

    """Serializable replacement for Lambda(preprocess_input) — required for .keras / .h5 save."""



    def __init__(self, backbone_name: str = "EfficientNetB0", **kwargs):

        super().__init__(**kwargs)

        self.backbone_name = backbone_name



    def call(self, inputs):

        name = (self.backbone_name or "").strip().lower()

        if name == "resnet50":

            return keras.applications.resnet.preprocess_input(inputs)

        if name == "mobilenetv2":

            return keras.applications.mobilenet_v2.preprocess_input(inputs)

        return keras.applications.efficientnet.preprocess_input(inputs)



    def get_config(self):

        config = super().get_config()

        config.update({"backbone_name": self.backbone_name})

        return config





def get_backbone_and_preprocess(base_model: str, img_size: int) -> tuple[keras.Model, type[BackbonePreprocessLayer]]:

    name = (base_model or "").strip().lower()

    shape = (img_size, img_size, 3)



    if name == "resnet50":

        backbone = keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=shape)

    elif name == "mobilenetv2":

        backbone = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=shape)

    elif name == "efficientnetb0":

        backbone = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=shape)

    else:

        raise ValueError(f"Unsupported base model: {base_model}. Choose from {SUPPORTED_BACKBONES}.")



    return backbone, BackbonePreprocessLayer





def build_classifier(

    num_classes: int,

    img_size: int,

    base_model: str,

    dropout: float = 0.35,

    for_inference: bool = False,

) -> tuple[keras.Model, keras.Model]:

    """EfficientNet/ResNet/MobileNet head with augmentation, BN, and dropout."""

    backbone, preprocess_layer_cls = get_backbone_and_preprocess(base_model, img_size)

    backbone.trainable = False



    inputs = keras.Input(shape=(img_size, img_size, 3))

    x = inputs



    if not for_inference:

        augment = keras.Sequential(

            [

                layers.RandomFlip("horizontal"),

                layers.RandomRotation(0.10),

                layers.RandomZoom(0.15),

                layers.RandomTranslation(0.08, 0.08),

                layers.RandomContrast(0.18),

                layers.RandomBrightness(0.12),

            ],

            name="augment",

        )

        x = augment(x)



    x = preprocess_layer_cls(base_model, name="preprocess")(x)

    x = backbone(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(dropout)(x)

    x = layers.Dense(256, activation="relu")(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(dropout * 0.8)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)



    model = keras.Model(inputs, outputs, name=f"{base_model}_cattle_breed")

    return model, backbone





def calibrate_rejection_thresholds(

    model: keras.Model,

    val_ds: tf.data.Dataset,

    class_names: list[str],

) -> dict[str, float]:

    """

    Calibrate thresholds targeting ~95% recall on cattle val images (fewer false rejects).



    Uses lower percentiles than the previous conservative calibration so real cattle

    (studio backgrounds, side views, varied lighting) are less likely to be blocked.

    """

    not_cattle_idx = None

    cattle_indices: list[int] = []

    for idx, name in enumerate(class_names):

        if normalize_breed_key(name) == NOT_CATTLE_CLASS:

            not_cattle_idx = idx

        else:

            cattle_indices.append(idx)



    max_probs: list[float] = []

    margins: list[float] = []

    not_cattle_probs_on_cattle: list[float] = []



    for batch_x, batch_y in val_ds:

        probs = model.predict(batch_x, verbose=0)

        labels = np.argmax(batch_y.numpy(), axis=1)

        for prob, label in zip(probs, labels):

            if not_cattle_idx is not None and int(label) == not_cattle_idx:

                continue

            cattle_prob = prob[cattle_indices] if cattle_indices else prob

            cattle_prob = cattle_prob / max(float(np.sum(cattle_prob)), 1e-8)

            top = float(np.max(cattle_prob))

            sorted_p = np.sort(cattle_prob)[::-1]

            margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else top

            max_probs.append(top)

            margins.append(margin)

            if not_cattle_idx is not None:

                not_cattle_probs_on_cattle.append(float(prob[not_cattle_idx]))



    if not max_probs:

        thresholds = {

            "min_confidence": 0.35,

            "min_margin": 0.05,

            "max_not_cattle_prob": 0.75,

        }

        print(f"Calibrated thresholds (defaults): {thresholds}")

        return thresholds



    # 20th percentile on cattle val -> ~80% of cattle pass at calibrated min_confidence

    raw_conf = float(np.percentile(max_probs, 20))

    raw_margin = float(np.percentile(margins, 25))

    raw_nc = (

        float(np.percentile(not_cattle_probs_on_cattle, 85))

        if not_cattle_probs_on_cattle

        else 0.75

    )



    min_confidence = round(float(np.clip(raw_conf * 0.90, 0.25, 0.55)), 4)

    min_margin = round(float(np.clip(raw_margin * 0.85, 0.03, 0.12)), 4)

    max_not_cattle_prob = round(float(np.clip(raw_nc, 0.60, 0.85)), 4)



    thresholds = {

        "min_confidence": min_confidence,

        "min_margin": min_margin,

        "max_not_cattle_prob": max_not_cattle_prob,

        "reject_confidence_pct": round(min_confidence * 100, 1),

    }

    print(f"Calibrated rejection thresholds (95% recall target): {thresholds}")

    return thresholds

