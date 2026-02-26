"""
Evaluation utilities for Convolutional Autoencoder (CAE) models.

This module provides:
    - Loss and correlation history plotting
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

from modules.train_utils import path_main, custom_corr, corr_keras


# ======================================================================
# Visualization Helpers
# ======================================================================
def plot_acc_loss(H, experiment_group, experiment_subgroup, loss="mse"):
    """Plot training loss and correlation history."""
    epochs = len(H.history[loss])
    num_subplot = 2

    plt.style.use("ggplot")
    plt.figure(figsize=(7 * num_subplot, 5))

    plt.subplot(1, num_subplot, 1)
    plt.plot(np.arange(epochs), H.history[loss], label="train_loss")
    plt.title(f"{loss}_{experiment_subgroup}", fontsize=15)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(loss, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)

    plt.subplot(1, num_subplot, 2)
    plt.plot(np.arange(epochs), H.history["corr_keras"], label="train_corr")
    plt.title(f"corr_{experiment_subgroup}", fontsize=15)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Correlation", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    save_dir = os.path.join(path_main, f"results/{experiment_group}/plot_acc_loss")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{experiment_subgroup}.png"))
    plt.close()


def detail_modification(ax):
    """Apply consistent axis styling for scalogram plots."""
    ax.set_ylabel("frequency [Hz]", fontsize=10)
    ax.set_xlabel("sleep period", fontsize=10)
    ax.yaxis.set_ticks(range(0, 16))
    ax.yaxis.set_ticklabels(
        [0.5, 0.7, 0.9, 1.2, 1.6, 2.1, 2.8, 3.8,
         5.0, 6.7, 8.9, 11.9, 15.8, 21.1, 28.1, 37.5],
        fontsize=10
    )
    ax.grid(False)


def _select_plot_channel(sample, channel_idx=0):
    """
    Convert one sample into a 2D map (freq x time) for plotting.
    Supports single-channel and multi-channel samples.
    """
    s = np.asarray(sample)
    s = np.squeeze(s)

    if s.ndim == 2:
        return s

    if s.ndim != 3:
        raise ValueError(f"Unsupported sample shape for plotting: {s.shape}")

    # (freq, time, channel)
    if s.shape[0] == 16 and s.shape[1] >= 100:
        ch = min(channel_idx, s.shape[2] - 1)
        return s[:, :, ch]

    # (freq, channel, time)
    if s.shape[0] == 16 and s.shape[2] >= 100:
        ch = min(channel_idx, s.shape[1] - 1)
        return s[:, ch, :]

    raise ValueError(f"Cannot infer axes for plotting shape: {s.shape}")


# ======================================================================
# Input/Output Comparison
# ======================================================================
def plot_i_o_compare(
    inputs,
    outputs,
    experiment_group,
    experiment_subgroup,
    num_samples=5,
    title_input="input",
    title_output="output",
    channel_idx=0,
):
    """Visualize comparison of input and reconstructed output scalograms."""
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

        in_map = _select_plot_channel(inputs[i], channel_idx=channel_idx)
        out_map = _select_plot_channel(outputs[i], channel_idx=channel_idx)

        im_in = ax_in.imshow(in_map, cmap="hot", aspect=100, origin="lower")
        ax_in.set_title(f"{title_input}_{i+1} (ch={channel_idx})", fontsize=15)
        detail_modification(ax_in)
        fig.colorbar(im_in, shrink=0.5, ax=ax_in)

        im_out = ax_out.imshow(out_map, cmap="hot", aspect=100, origin="lower")
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
    """Load a trained CAE model and extract the encoder."""
    aec = load_model(
        path_weight,
        custom_objects={"corr_keras": corr_keras}
    )

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
    experiment_subgroup,
    channel_idx=0,
):
    """Complete evaluation pipeline for a trained CAE model."""
    pred = model.predict(_input)

    plot_i_o_compare(
        _input,
        pred,
        experiment_group,
        experiment_subgroup,
        channel_idx=channel_idx,
    )

    median_corr, corr_list = custom_corr(_input, pred, median=True)

    plt.figure()
    ax = sns.boxplot(x=corr_list, boxprops={"alpha": 0.5})
    ax.set_xlabel("Correlation Coefficients", fontsize=15)
    ax.text(
        x=min(corr_list),
        y=-0.2,
        s=f"Mean corr = {np.mean(corr_list):.4f}",
        fontsize=14,
    )
    ax.text(
        x=min(corr_list),
        y=-0.1,
        s=f"Median corr = {np.median(corr_list):.4f}",
        fontsize=14,
    )

    save_dir = os.path.join(path_main, f"results/{experiment_group}/corr_boxplot")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{experiment_subgroup}.png"))
    plt.close()

    plot_acc_loss(
        H=H,
        experiment_group=experiment_group,
        experiment_subgroup=experiment_subgroup,
        loss=loss,
    )
