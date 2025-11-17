import os
import warnings
import argparse
import pandas as pd
import numpy as np

import train_utils
import eval_utils
from train_utils import path_csv, path_np_data, path_main

warnings.filterwarnings('ignore')


# ======================================================================
# Utility Functions
# ======================================================================
def get_df_experiment():
    """
    Load and preprocess experiment configuration CSV.

    This function:
        - Reads the experiment configuration CSV file.
        - Converts boolean-like string entries to Python booleans.
        - Casts predefined rows to string, integer, or float types.

    Returns
    -------
    pd.DataFrame
        Preprocessed experiment configuration table.
    """
    df = pd.read_csv(
        os.path.join(path_csv, "00_experiments_12.csv"),
        index_col=0
    )

    def cast_rows_to_dtype(row_names, dtype):
        """Cast specific rows to a given dtype."""
        df.loc[row_names] = df.loc[row_names].astype(dtype)

    # Boolean rows
    bool_rows = ["is_lr_reducer", "is_early_stop", "verbose"]
    df.loc[bool_rows] = df.loc[bool_rows].applymap(
        lambda x: True if x == "TRUE" else False
    )

    # String rows
    cast_rows_to_dtype(
        ["experiment_group", "experiment_subgroup", "loss"], str
    )

    # Integer rows
    cast_rows_to_dtype(
        ["vector_len", "init_filter_num", "model_depth", "epochs",
         "batch_size"],
        int
    )

    # Float rows
    cast_rows_to_dtype(["learning_rate"], float)

    return df


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
    batch_size
):
    """
    End-to-end training pipeline for a Convolutional Autoencoder (CAE).

    This performs:
        1. Loading scalogram data.
        2. Initializing the CAE model.
        3. Compiling the model with optimizer and loss.
        4. Training the model.
        5. Running evaluation (visualization + corr metrics).

    Parameters
    ----------
    experiment_group : str
        Name of experiment group (folder structure).
    experiment_subgroup : str
        Name of experiment subgroup (model identifier).
    verbose : bool
        Whether to print loading logs.
    vector_len : int
        Size of embedding vector.
    init_filter_num : int
        Number of initial convolution filters.
    model_depth : int
        Number of encoder/decoder blocks.
    learning_rate : float
        Initial learning rate.
    is_lr_reducer : bool
        Whether to apply ReduceLROnPlateau.
    is_early_stop : bool
        Whether to apply EarlyStopping.
    loss : str
        Loss function name.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    """
    # --------------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------------
    train_utils.save_json(
        {"verbose": verbose},
        experiment_group,
        experiment_subgroup,
        "load_params"
    )

    scalograms = np.load(os.path.join(
        path_np_data, "scalogram_2000t_16f_healthy_insomnia.npy"
    ))
    scalograms_MR = np.load(os.path.join(
        path_np_data, "scalogram_scalograms_with_MR_only_insomnia.npy"
    ))

    # Concatenate two datasets
    scalograms = np.vstack((scalograms, scalograms_MR))
    print(f"Shape of final scalograms: {scalograms.shape}")

    # --------------------------------------------------------------
    # 2. Initialize CAE model
    # --------------------------------------------------------------
    model_params = {
        "vector_len": vector_len,
        "input_shape": (16, 2000, 1),
        "init_filter_num": init_filter_num,
        "depth": model_depth,
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
        experiment_subgroup=experiment_subgroup
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

    # Train for each experiment column
    for column in df.columns:
        config = df_train.loc[:, column].to_dict()

        # Ensure weight directory exists
        os.makedirs(
            os.path.join(path_main, f"model_weights/{config['experiment_group']}"),
            exist_ok=True
        )

        train_CAE(**config)
