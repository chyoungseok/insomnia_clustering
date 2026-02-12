"""
Convolutional Autoencoder (CAE) architectures.

This module currently implements:
    - CAE_v02: A 2D convolutional autoencoder without max pooling.

The model is designed for scalogram inputs with shape
    (frequency_bins, time_steps, 1)

To modify or add new CAE versions, follow the same structure.
"""

from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    Flatten,
    Dense,
    Reshape,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


# ======================================================================
# CAE_v02: Convolutional Autoencoder (No MaxPooling)
# ======================================================================
def CAE_v02(
    vector_len=30,
    input_shape=(16, 2000, 1),
    init_filter_num=4,
    depth=1
):
    """
    Build a 2D Convolutional Autoencoder (Version 02).

    Characteristics:
        - No max pooling; uses stride-based downsampling.
        - Encoder and decoder depths are symmetrical.
        - Embedding vector is produced via dense layer (flattened bottleneck).
        - L2 regularization is applied to all convolution layers.

    Parameters
    ----------
    vector_len : int
        Size of the embedding vector (bottleneck).
    input_shape : tuple(int, int, int)
        Shape of input feature map (H, W, C).
    init_filter_num : int
        Number of filters in the first layer. Each subsequent depth
        increases filters by (init_filter_num * (i+1)).
    depth : int
        Number of encoder/decoder convolution blocks.

    Returns
    -------
    keras.Model
        Constructed autoencoder model.
    """
    # ------------------------------
    # Input
    # ------------------------------
    inputs = Input(shape=input_shape)
    x = inputs

    # ==========================================================
    # Encoder
    # ==========================================================
    for i in range(depth):
        block_name = f"enc_layer{i+1}_"

        # First Conv block (freq axis)
        x = Conv2D(
            filters=init_filter_num * (i + 1),
            kernel_size=(2, 2),
            strides=(1, 1),
            padding="same",
            activation="elu",
            kernel_regularizer=regularizers.l2(),
            name=block_name + "conv1"
        )(x)
        x = BatchNormalization(name=block_name + "bn1")(x)

        # Second Conv block (time axis, stride=2 → downsampling)
        x = Conv2D(
            filters=init_filter_num * (i + 1),
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="same",
            activation="elu",
            kernel_regularizer=regularizers.l2(),
            name=block_name + "conv2"
        )(x)
        x = BatchNormalization(name=block_name + "bn2")(x)

    # ==========================================================
    # Bottleneck (Embedding)
    # ==========================================================
    flat = Flatten()(x)

    # Embedding vector
    embedding = Dense(
        units=vector_len,
        name="embedding"
    )(flat)

    # Upscale back to encoder output shape
    up = Dense(
        units=x.shape[1] * x.shape[2] * x.shape[3]
    )(embedding)
    x = Reshape(
        (x.shape[1], x.shape[2], x.shape[3])
    )(up)

    # ==========================================================
    # Decoder
    # ==========================================================
    for i in range(depth):
        block_name = f"dec_layer{i+1}_"
        filters = init_filter_num * (depth - i)  # Reverse filter order

        # First transposed conv (upsampling)
        x = Conv2DTranspose(
            filters=filters,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="same",
            activation="elu",
            kernel_regularizer=regularizers.l2(),
            name=block_name + "convT1"
        )(x)
        x = BatchNormalization(name=block_name + "bn1")(x)

        # Second transposed conv
        x = Conv2DTranspose(
            filters=filters,
            kernel_size=(2, 2),
            strides=(1, 1),
            padding="same",
            activation="elu",
            kernel_regularizer=regularizers.l2(),
            name=block_name + "convT2"
        )(x)
        x = BatchNormalization(name=block_name + "bn2")(x)

    # ==========================================================
    # Final Output Layer (1-channel reconstruction)
    # ==========================================================
    outputs = Conv2D(
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation=None,
        kernel_regularizer=regularizers.l2(),
        name="last_layer"
    )(x)

    # Build Model
    return Model(inputs=inputs, outputs=outputs)
