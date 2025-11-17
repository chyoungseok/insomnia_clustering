"""
Utility functions for training the Convolutional Autoencoder (CAE).

This module provides:
    - Path configurations
    - Boolean parser for argument parsing
    - JSON save utility
    - Model initialization wrapper
    - Custom correlation metric (numpy + Keras)
    - Model compilation (optimizer, loss, callbacks)
    - Training helper (wrapper around model.fit)
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)

from autoencoders import CAE_v02


# ======================================================================
# Path Configuration
# ======================================================================
# NOTE: Do NOT modify unless directory structure changes.
path_main = "D:/USC/01_code/insomnia_clustering"
path_csv = "D:/USC/01_code/insomnia_clustering"
path_np_data = "D:/USC/01_code/insomnia_clustering/data"


# ======================================================================
# Argument Utilities
# ======================================================================
def str2bool(v):
    """
    Convert string input to boolean for argparse.

    Parameters
    ----------
    v : str or bool
        Input argument.

    Returns
    -------
    bool
        Converted boolean value.

    Raises
    ------
    argparse.ArgumentTypeError
        When invalid boolean value is provided.
    """
    if isinstance(v, bool):
        return v

    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_minimal_args(parser):
    """
    Add minimal arguments to a parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser

    Returns
    -------
    argparse.ArgumentParser
        Parser with additional arguments.
    """
    parser.add_argument("--only_test", type=str2bool, default=False)
    return parser


# ======================================================================
# JSON Save Utility
# ======================================================================
def save_json(params, experiment_group, experiment_subgroup, json_type):
    """
    Save parameter dictionary as a JSON file.

    Parameters
    ----------
    params : dict
        Parameter set to save.
    experiment_group : str
        Folder name (group level).
    experiment_subgroup : str
        File name (subgroup level).
    json_type : str
        Directory name inside 'results'.

    Returns
    -------
    None
    """
    save_dir = os.path.join(path_main, f"results/{experiment_group}/{json_type}")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{experiment_subgroup}.json")
    with open(save_path, "w") as f:
        json.dump(params, f, indent=4)


# ======================================================================
# Model Initialization
# ======================================================================
def init_model(experiment_group, model_params):
    """
    Initialize a model based on experiment group identifier.

    Parameters
    ----------
    experiment_group : str
        Name indicating which model version to initialize.
    model_params : dict
        Construction arguments passed to the CAE constructor.

    Returns
    -------
    keras.Model
        Initialized model instance.
    """
    if "v02" in experiment_group:
        return CAE_v02(**model_params)

    raise ValueError(f"Unknown experiment group: {experiment_group}")


# ======================================================================
# Custom Correlation Metric
# ======================================================================
def custom_corr(X_true, X_pred, median=False):
    """
    Compute correlation (Pearson) for each sample.

    Parameters
    ----------
    X_true : np.ndarray
        Ground-truth input.
    X_pred : np.ndarray
        Model reconstruction.
    median : bool, optional
        If True, return median instead of mean.

    Returns
    -------
    float or (float, list)
        Mean correlation, or (median, list of correlations).
    """
    corr_list = []

    for i in range(len(X_true)):
        corr_value = np.corrcoef(
            K.flatten(tf.squeeze(X_true[i])),
            K.flatten(tf.squeeze(X_pred[i]))
        )[0, 1]
        corr_list.append(corr_value)

    if median:
        return np.median(corr_list), corr_list

    return np.mean(corr_list)


def corr_keras(X_true, X_pred):
    """
    Keras wrapper for custom_corr to be used as a metric.

    Returns
    -------
    tf.Tensor
        Correlation score as TensorFlow tensor.
    """
    return tf.py_function(
        func=custom_corr,
        inp=[X_true, X_pred],
        Tout=tf.float32,
        name="corr"
    )


# ======================================================================
# Model Compilation
# ======================================================================
def compile_model(
    model,
    learning_rate,
    loss,
    experiment_group,
    experiment_subgroup,
    is_lr_reducer,
    is_early_stop
):
    """
    Compile the CAE model with optimizer, loss, metrics, and callbacks.

    Parameters
    ----------
    model : keras.Model
        CAE model instance.
    learning_rate : float
        Learning rate for Adam optimizer.
    loss : str
        Loss function name.
    experiment_group : str
        Group name for saving weight files.
    experiment_subgroup : str
        Weight file identifier.
    is_lr_reducer : bool
        Whether to include ReduceLROnPlateau.
    is_early_stop : bool
        Whether to include EarlyStopping.

    Returns
    -------
    model : keras.Model
        Compiled model.
    callbacks : list
        List of callbacks for training.
    """
    # Register custom metric
    get_custom_objects().update({"corr_keras": corr_keras})

    # Optimizer configuration (based on U-Sleep paper)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        decay=0.0
    )

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[loss, "corr_keras"]
    )

    # -------------------------
    # Set up callbacks
    # -------------------------
    weight_dir = os.path.join(path_main, f"model_weights/{experiment_group}")
    os.makedirs(weight_dir, exist_ok=True)

    weight_path = os.path.join(
        weight_dir, f"{experiment_subgroup}.h5"
    )
    checkpoint = ModelCheckpoint(
        weight_path,
        monitor="loss",
        save_best_only=True,
        verbose=False,
        mode="min",
    )

    callbacks = [checkpoint]

    if is_lr_reducer:
        lr_reducer = ReduceLROnPlateau(
            monitor="loss",
            factor=0.2,
            cooldown=10,
            patience=10,
            min_lr=1e-9,
            verbose=True,
            mode="min"
        )
        callbacks.append(lr_reducer)

    if is_early_stop:
        early_stop = EarlyStopping(
            monitor="loss",
            patience=50,
            restore_best_weights=True,
            verbose=True,
            mode="min"
        )
        callbacks.append(early_stop)

    return model, callbacks


# ======================================================================
# Training Function
# ======================================================================
def train_model(
    model,
    _input,
    _output,
    epochs,
    callbacks,
    batch_size,
    experiment_group,
    experiment_subgroup,
    loss
):
    """
    Wrapper for model.fit(), used for consistency and future extensions.

    Parameters
    ----------
    model : keras.Model
        CAE instance.
    _input : np.ndarray
        Input scalogram data.
    _output : np.ndarray
        Output (same as input for autoencoder).
    epochs : int
        Training epochs.
    callbacks : list
        List of callbacks.
    batch_size : int
        Batch size.
    experiment_group : str
        Group name.
    experiment_subgroup : str
        Subgroup name.
    loss : str
        Loss function name.

    Returns
    -------
    H : keras.callbacks.History
        Training history.
    model : keras.Model
        Trained model.
    """
    history = model.fit(
        _input,
        _output,
        epochs=epochs,
        callbacks=callbacks,
        batch_size=batch_size
    )

    return history, model
