import os
import warnings
import argparse
import pandas as pd
import numpy as np

import modules.train_utils as train_utils
import modules.eval_utils as eval_utils
from modules.train_utils import path_csv, path_np_data, path_main

warnings.filterwarnings('ignore')


# ======================================================================
# Utility Functions
# ======================================================================
def get_df_experiment():
    """
    Load and preprocess experiment configuration CSV.

    Returns
    -------
    pd.DataFrame
        Preprocessed experiment configuration table.
    """
    df = pd.read_csv(
        os.path.join(path_csv, "00_experiments_13_2ch.csv"),
        index_col=0
    )

    def cast_rows_to_dtype(row_names, dtype):
        rows = [r for r in row_names if r in df.index]
        if rows:
            df.loc[rows] = df.loc[rows].astype(dtype)

    # Boolean rows
    bool_rows = ["is_lr_reducer", "is_early_stop", "verbose"]
    if all(r in df.index for r in bool_rows):
        df.loc[bool_rows] = df.loc[bool_rows].applymap(
            lambda x: True if x == "TRUE" else False
        )

    # String rows
    cast_rows_to_dtype(
        ["experiment_group", "experiment_subgroup", "loss"], str
    )

    # Integer rows
    int_rows = [
        "vector_len", "init_filter_num", "model_depth",
        "epochs", "batch_size", "n_channels"
    ]
    cast_rows_to_dtype(int_rows, int)

    # Float rows
    cast_rows_to_dtype(["learning_rate"], float)

    return df


def _normalize_scalogram_shape(arr, n_channels):
    """
    Normalize scalogram array to (N, 16, 2000, C).
    """
    arr = np.asarray(arr)

    # Common legacy format: (N, F, C, T, 1)
    if arr.ndim == 5 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 4:
        raise ValueError(f"Unsupported scalogram ndim: {arr.ndim}, shape={arr.shape}")

    # (N, F, T, C)
    if arr.shape[1] == 16 and arr.shape[2] == 2000:
        pass
    # (N, F, C, T)
    elif arr.shape[1] == 16 and arr.shape[3] == 2000:
        arr = np.transpose(arr, (0, 1, 3, 2))
    # (N, C, F, T)
    elif arr.shape[2] == 16 and arr.shape[3] == 2000:
        arr = np.transpose(arr, (0, 2, 3, 1))
    else:
        raise ValueError(f"Cannot infer axes for scalogram shape: {arr.shape}")

    if arr.shape[-1] < n_channels:
        raise ValueError(
            f"Requested n_channels={n_channels}, but data has only {arr.shape[-1]} channels"
        )

    arr = arr[..., :n_channels]
    return arr.astype(np.float32)


# ======================================================================
# Training Function
# ======================================================================
def train_CAE(
    experiment_group,
    experiment_subgroup,
    verbose,
    vector_len,
    init_filter_num,
    model_depth,
    learning_rate,
    is_lr_reducer,
    is_early_stop,
    loss,
    epochs,
    batch_size,
    n_channels=6,
):
    """
    End-to-end training pipeline for a Convolutional Autoencoder (CAE).
    """
    # --------------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------------
    train_utils.save_json(
        {"verbose": verbose, "n_channels": int(n_channels)},
        experiment_group,
        experiment_subgroup,
        "load_params"
    )
    
    if n_channels == 1:
        ch_mode = "single"
    elif n_channels == 2:
        ch_mode = "2ch"
    elif n_channels == 6:
        ch_mode = "6ch"

    scalograms = np.load(os.path.join(
        path_np_data, ch_mode, "scalogram_2000t_16f_healthy_insomnia.npy"
    ))
    scalograms_MR = np.load(os.path.join(
        path_np_data, ch_mode, "scalogram_scalograms_with_MR_only_insomnia.npy"
    ))

    scalograms = _normalize_scalogram_shape(scalograms, n_channels=n_channels)
    scalograms_MR = _normalize_scalogram_shape(scalograms_MR, n_channels=n_channels)

    # Concatenate two datasets
    scalograms = np.concatenate((scalograms, scalograms_MR), axis=0)
    print(f"Shape of final scalograms: {scalograms.shape}")

    # --------------------------------------------------------------
    # 2. Initialize CAE model
    # --------------------------------------------------------------
    model_params = {
        "vector_len": vector_len,
        "input_shape": tuple(scalograms.shape[1:]),
        "init_filter_num": init_filter_num,
        "depth": model_depth,
        "output_channels": int(n_channels),
    }
    train_utils.save_json(
        model_params, experiment_group, experiment_subgroup, "model_params"
    )
    model = train_utils.init_model(experiment_group, model_params)

    # --------------------------------------------------------------
    # 3. Compile model
    # --------------------------------------------------------------
    compile_params = {
        "experiment_group": experiment_group,
        "experiment_subgroup": experiment_subgroup,
        "learning_rate": learning_rate,
        "is_lr_reducer": is_lr_reducer,
        "is_early_stop": is_early_stop,
        "loss": loss,
    }
    train_utils.save_json(
        compile_params, experiment_group, experiment_subgroup, "compile_params"
    )
    model, callbacks = train_utils.compile_model(model, **compile_params)

    # --------------------------------------------------------------
    # 4. Train model
    # --------------------------------------------------------------
    train_params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "loss": loss,
    }
    train_utils.save_json(
        train_params, experiment_group, experiment_subgroup, "train_params"
    )

    H, model = train_utils.train_model(
        model=model,
        _input=scalograms,
        _output=scalograms,
        callbacks=callbacks,
        experiment_group=experiment_group,
        experiment_subgroup=experiment_subgroup,
        **train_params
    )

    # --------------------------------------------------------------
    # 5. Evaluate model
    # --------------------------------------------------------------
    eval_utils.eval_model(
        H=H,
        model=model,
        _input=scalograms,
        loss=loss,
        experiment_group=experiment_group,
        experiment_subgroup=experiment_subgroup,
        channel_idx=0,
    )


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = train_utils.parse_minimal_args(parser)
    args = parser.parse_args()

    df = get_df_experiment()
    df_train = df.loc["experiment_group":"batch_size", :]
    if "n_channels" in df.index:
        df_train = pd.concat([df_train, df.loc[["n_channels"], :]], axis=0)

    # Train for each experiment column
    for column in df.columns:
        config = df_train.loc[:, column].to_dict()

        # Ensure weight directory exists
        os.makedirs(
            os.path.join(path_main, f"model_weights/{config['experiment_group']}"),
            exist_ok=True
        )

        train_CAE(**config)
