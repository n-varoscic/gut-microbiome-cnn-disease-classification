""" 
CNN model definitions for image-only and metadata-fusion architectures. 
The _image_branch is shared between both models. 
"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from config import LOSS_FN


# ── Shared taxa abundance image branch ───────────────────────────────────────────────────────

def _image_branch(img_input, conv_drop):
    """Build the shared image feature extractor.

    3× (Conv2D → BN → ReLU → MaxPool → SpatialDropout2D) → Flatten → Dense(256).

    Args:
        img_input : Keras Input tensor for the taxa image
        conv_drop : SpatialDropout2D rate

    Returns:
        Tensor after Dense(256) — ready to be used directly (image-only)
        or concatenated with a metadata embedding.
    """
    # Block 1: 32 filters
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.SpatialDropout2D(conv_drop)(x)

    # Block 2: 64 filters
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.SpatialDropout2D(conv_drop)(x)

    # Block 3: 128 filters (no SpatialDropout — Flatten follows directly)
    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)

    x = layers.Dense(256, activation='relu')(x)
    return x


# ── Model builders ────────────────────────────────────────────────────────────

def build_image_only_model(img_shape, lr=1e-4, conv_drop=0.25, dense_drop=0.5):
    """Build the image-only CNN (ablation baseline, no metadata).

    Args:
        img_shape  : tuple — (height, width, channels), e.g. (78, 78, 1)
        lr         : float — Adam learning rate
        conv_drop  : float — SpatialDropout2D rate in conv blocks
        dense_drop : float — Dropout rate after dense layers

    Returns:
        Compiled Keras Model with one input: 'taxa_image'.
    """
    img_input = keras.Input(shape=img_shape, name='taxa_image')

    x   = _image_branch(img_input, conv_drop)
    x   = layers.Dropout(dense_drop)(x)
    out = layers.Dense(64, activation='relu')(x)
    out = layers.Dropout(dense_drop)(out)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)

    model = Model(inputs=img_input, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=LOSS_FN,
        metrics=['accuracy', keras.metrics.AUC(name='auc')],
    )
    return model


def build_metadata_model(img_shape, meta_dim, lr=1e-4, conv_drop=0.25, dense_drop=0.5):
    """Build the metadata-fusion CNN (image branch + metadata branch + late fusion).

    Args:
        img_shape  : tuple — (height, width, channels)
        meta_dim   : int   — number of metadata features (e.g. 20)
        lr         : float — Adam learning rate
        conv_drop  : float — SpatialDropout2D rate in conv blocks
        dense_drop : float — Dropout rate after dense layers

    Returns:
        Compiled Keras Model with two inputs: 'taxa_image' and 'metadata'.
    """
    # Image branch
    img_input = keras.Input(shape=img_shape, name='taxa_image')
    x = _image_branch(img_input, conv_drop)
    x = layers.Dropout(dense_drop)(x)

    # METADATA BRANCH: progressively compress meta_dim features into 32-dim embedding
    meta_input = keras.Input(shape=(meta_dim,), name='metadata')
    m = layers.Dense(128, activation='relu')(meta_input)
    m = layers.Dropout(0.3)(m)
    m = layers.Dense(64, activation='relu')(m)
    m = layers.Dropout(0.3)(m)
    m = layers.Dense(32, activation='relu')(m)

    # LATE FUSION: concatenate image and metadata embeddings, then classify
    fused = layers.Concatenate(name='fusion')([x, m])
    out   = layers.Dense(64, activation='relu')(fused)
    out   = layers.Dropout(dense_drop)(out)
    out   = layers.Dense(1, activation='sigmoid', name='output')(out)

    model = Model(inputs=[img_input, meta_input], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=LOSS_FN,
        metrics=['accuracy', keras.metrics.AUC(name='auc')],
    )
    return model
