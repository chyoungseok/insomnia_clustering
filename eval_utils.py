"""
Evaluation utilities for Convolutional Autoencoder (CAE) models.

This module provides:
    - Loss & correlation history plotting
    - Input vs reconstructed output visualization
    - Correlation distribution analysis (boxplot)
    - Model loading with encoder extraction
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model, Model

from train_utils import path_main, custom_corr, corr_keras


# ======================================================================
# Visualization Helpers
# ======================================================================
def plot_acc_loss(H, experiment_group, experiment_subgroup, loss="mse"):
    """
    Plot training loss and correlation metric history.

    Parameters
    ----------
    H : keras.callbacks.History
        Training history object.
    experiment_group : str
        Name of experiment group.
    experiment_subgroup : str
        Model identifier.
    loss : str
        Loss function name.

    Returns
    -------
    None
    """
    epochs = len(H.history[loss])
    num_subplot = 2

    plt.style.use("ggplot")
    plt.figure(figsize=(7 * num_subplot, 5))

    # ----------------------------------------------------------
    # 1. Loss plot
    # ----------------------------------------------------------
    plt.subplot(1, num_subplot, 1)
    plt.plot(np.arange(epochs), H.history[loss], label="train_loss")
    plt.title(f"{loss}_{experiment_subgroup}", fontsize=15)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(loss, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)

    # ----------------------------------------------------------
    # 2. Correlation plot
    # ----------------------------------------------------------
    plt.subplot(1, num_subplot, 2)
    plt.plot(np.arange(epochs), H.history["corr_keras"], label="train_corr")
    plt.title(f"corr_{experiment_subgroup}", fontsize=15)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Correlation", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    # Save result
    save_dir = os.path.join(path_main, f"results/{experiment_group}/plot_acc_loss")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{experiment_subgroup}.png"))
    plt.close()


def detail_modification(ax):
    """
    Apply consistent axis styling for scalogram plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to modify.

    Returns
    -------
    None
    """
    ax.set_ylabel("frequency [Hz]", fontsize=10)
    ax.set_xlabel("sleep period", fontsize=10)
    ax.yaxis.set_ticks(range(0, 16))
    ax.yaxis.set_ticklabels(
        [0.5, 0.7, 0.9, 1.2, 1.6, 2.1, 2.8, 3.8,
         5.0, 6.7, 8.9, 11.9, 15.8, 21.1, 28.1, 37.5],
        fontsize=10
    )
    ax.grid(False)


# ======================================================================
# Input–Output Comparison
# ======================================================================
def plot_i_o_compare(
    inputs,
    outputs,
    experiment_group,
    experiment_subgroup,
    num_samples=5,
    title_input="input",
    title_output="output"
):
    """
    Visualize comparison of input and reconstructed output scalograms.

    Parameters
    ----------
    inputs : np.ndarray
        Original scalograms.
    outputs : np.ndarray
        Reconstructed scalograms.
    experiment_group : str
        Group identifier.
    experiment_subgroup : str
        Experiment identifier.
    num_samples : int
        Number of samples to visualize.
    title_input : str
        Plot title for input.
    title_output : str
        Plot title for output.

    Returns
    -------
    None
    """
    # Ensure shape compatibility (N, H, W, C)
    if inputs.ndim < 4:
        inputs = inputs[:, :, np.newaxis, :]
        outputs = outputs[:, :, np.newaxis, :]

    inputs = inputs[:num_samples]
    outputs = outputs[:num_samples]

    fig, axes = plt.subplots(2, num_samples, figsize=(6 * num_samples, 10))

    for i in range(num_samples):
        corr = np.corrcoef(
            K.flatten(tf.squeeze(inputs[i])),
            K.flatten(tf.squeeze(outputs[i]))
        )[0, 1]

        ax_in = axes[0, i] if num_samples > 1 else axes[0]
        ax_out = axes[1, i] if num_samples > 1 else axes[1]

        # Input plot
        im_in = ax_in.imshow(inputs[i], cmap="hot", aspect=100, origin="lower")
        ax_in.set_title(f"{title_input}_{i+1}", fontsize=15)
        detail_modification(ax_in)
        fig.colorbar(im_in, shrink=0.5, ax=ax_in)

        # Output plot
        im_out = ax_out.imshow(outputs[i], cmap="hot", aspect=100, origin="lower")
        ax_out.set_title(f"{title_output}_{i+1} ({corr:.3f})", fontsize=15)
        detail_modification(ax_out)
        fig.colorbar(im_out, shrink=0.5, ax=ax_out)

    save_dir = os.path.join(path_main, f"results/{experiment_group}/plot_i_o")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{experiment_subgroup}.png"))
    plt.close()


# ======================================================================
# Model Loader
# ======================================================================
def load_my_model(path_weight):
    """
    Load a trained CAE model and extract the encoder.

    Parameters
    ----------
    path_weight : str
        Path to the HDF5 weight file.

    Returns
    -------
    aec : keras.Model
        Full autoencoder model.
    enc : keras.Model
        Encoder network producing latent embedding.
    """
    aec = load_model(
        path_weight,
        custom_objects={"corr_keras": corr_keras}
    )

    # Extract encoder (layer named "embedding")
    enc = Model(
        inputs=aec.input,
        outputs=aec.get_layer("embedding").output
    )
    return aec, enc


# ======================================================================
# Evaluation Wrapper
# ======================================================================
def eval_model(
    H,
    model,
    _input,
    loss,
    experiment_group,
    experiment_subgroup
):
    """
    Complete evaluation pipeline for a trained CAE model.

    It performs:
        1. Forward prediction
        2. Input–output visualization
        3. Correlation distribution (boxplot)
        4. Training history plots

    Parameters
    ----------
    H : keras.callbacks.History
        Training history.
    model : keras.Model
        Trained CAE model.
    _input : np.ndarray
        Scalogram input.
    loss : str
        Loss function used.
    experiment_group : str
        Group identifier.
    experiment_subgroup : str
        Subgroup identifier.

    Returns
    -------
    None
    """
    # ----------------------------------------------------------
    # 1. Reconstruction prediction
    # ----------------------------------------------------------
    pred = model.predict(_input)

    # ----------------------------------------------------------
    # 2. Input–output comparison plots
    # ----------------------------------------------------------
    plot_i_o_compare(
        _input, pred,
        experiment_group,
        experiment_subgroup
    )

    # ----------------------------------------------------------
    # 3. Correlation distribution
    # ----------------------------------------------------------
    median_corr, corr_list = custom_corr(_input, pred, median=True)

    plt.figure()
    ax = sns.boxplot(x=corr_list, boxprops={"alpha": 0.5})
    ax.set_xlabel("Correlation Coefficients", fontsize=15)
    ax.text(
        x=min(corr_list),
        y=-0.2,
        s=f"Mean corr = {np.mean(corr_list):.4f}",
        fontsize=14
    )
    ax.text(
        x=min(corr_list),
        y=-0.1,
        s=f"Median corr = {np.median(corr_list):.4f}",
        fontsize=14
    )

    save_dir = os.path.join(path_main, f"results/{experiment_group}/corr_boxplot")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{experiment_subgroup}.png"))
    plt.close()

    # ----------------------------------------------------------
    # 4. Training history plots
    # ----------------------------------------------------------
    plot_acc_loss(
        H=H,
        experiment_group=experiment_group,
        experiment_subgroup=experiment_subgroup,
        loss=loss
    )
