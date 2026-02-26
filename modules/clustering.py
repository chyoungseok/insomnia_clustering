import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from modules.utils import path_main, path_np_data, path_csv
from modules import load, statistics
from modules import gen_df_MR
from modules import autoencoders
from modules import utils, eval_utils
from modules.stability_scheme1 import stability


def _resolve_channel_mode(channel_mode=None, n_channels=None):
    if channel_mode is not None:
        if channel_mode not in ["single", "2ch", "6ch"]:
            raise ValueError("channel_mode must be 'single', '2ch', or '6ch'")
        return channel_mode

    if n_channels == 1:
        return "single"
    if n_channels == 2:
        return "2ch"
    if n_channels == 6:
        return "6ch"

    return "6ch"


def _load_npy_with_fallback(primary_path, fallback_path=None):
    if os.path.exists(primary_path):
        return np.load(primary_path), primary_path

    if fallback_path is not None and os.path.exists(fallback_path):
        return np.load(fallback_path), fallback_path

    raise FileNotFoundError(
        f"Numpy file not found. primary='{primary_path}', fallback='{fallback_path}'"
    )


def get_scalo_hs(channel_mode="6ch"):
    channel_mode = _resolve_channel_mode(channel_mode=channel_mode)

    hs_fname = "scalogram_2000t_16f_healthy_insomnia.npy"
    hs_path_primary = os.path.join(path_np_data, channel_mode, hs_fname)
    hs_path_fallback = os.path.join(path_np_data, hs_fname)
    np_hs_ins, loaded_path = _load_npy_with_fallback(hs_path_primary, hs_path_fallback)
    print(f"Load healthy+insomnia scalograms from: {loaded_path}")

    df_demo = pd.read_csv('D:/USC/01_code/insomnia_clustering/csv_files/df_demo_2000t_16f_healthy_insomnia.csv', encoding='euc-kr') # index_col=0 금지
    df_demo_psm = statistics.get_df_demo_HI_psm()
    
    psgid_HS = df_demo_psm.loc[df_demo_psm.labels == 2, 'PSG study Number#'].values
    con_HS = df_demo['PSG study Number#'].isin(psgid_HS)
    idx_HS = df_demo[con_HS].index.to_list()
    scalogram_HS = np_hs_ins[idx_HS]

    return scalogram_HS


class class_clustering:
    """
    1. Get load_params
    2. Embeddings from autoencoder
    3. Clustering
    """

    def __init__(
        self,
        experiment_group,
        experiment_subgroup,
        clustering_method,
        num_K,
        random_state,
        include_MR,
        inclusion_option,
        check_inclusion_option=False,
        is_plot_scalogram=False,
        B=10,
        is_average_clustering=False,
        channel_mode="6ch",
    ):
        load_params, channel_mode = getLoadParams(
            experiment_group,
            experiment_subgroup,
            inclusion_option,
            check_inclusion_option,
            channel_mode=channel_mode,
        )

        scalograms, _load = getWholeScalogram(
            load_params,
            include_MR,
            channel_mode=channel_mode,
        )

        scalograms_hs = get_scalo_hs(channel_mode=channel_mode)
        scalograms_ins = scalograms

        embeddings = getEmbeddings(
            experiment_group,
            experiment_subgroup,
            scalograms_ins,
            scalograms_hs,
        )
        center, labels, _km = runClustering(
            clustering_method,
            num_K,
            embeddings,
            random_state,
            B=B,
            is_average_clustering=is_average_clustering,
        )

        df_MR = gen_df_MR.get_df_MR()
        df_MR_scalo_ins = df_MR.loc[df_MR.is_mr_scalo, :]
        list_org, list_add = gen_df_MR.get_id_with_MR_org_add(df_MR_scalo_ins)

        df_demo = _load.df_demo.copy()
        if len(list_add) > 0:
            df_demo["labels"] = labels[:-len(list_add)]
        else:
            df_demo["labels"] = labels
            
        # udate and save df_demo_HI_psm by adding new labels (여기)
        df_demo_HI_psm = statistics.get_df_demo_HI_psm()
        df_demo_HI_psm_updated = df_demo_HI_psm.copy()
        
        for idx in df_demo.index:
            con = df_demo_HI_psm_updated['PSG study Number#'].isin([idx])
            df_demo_HI_psm_updated.loc[con, 'labels'] = df_demo.loc[idx, 'labels']

        df_demo_HI_psm_updated.to_csv(os.path.join(path_csv, 'df_demo_HI_psm_updated_labels.csv'), index=False, encoding='euc-kr')

        if "G" in clustering_method.upper():
            experiment_subgroup = experiment_subgroup + "_G_K%d" % num_K
        elif "K" in clustering_method.upper():
            experiment_subgroup = experiment_subgroup + "_K_K%d" % num_K

        if check_inclusion_option:
            experiment_group = experiment_group + "_" + inclusion_option

        if is_plot_scalogram:
            utils.align_scalograms(
                embeddings=embeddings,
                idx_cluster=labels,
                centroids=center,
                scalograms=scalograms,
                num_K=num_K,
                experiment_group=experiment_group,
                experiment_subgroup=experiment_subgroup,
                num_align=1,
            )

            utils.align_scalograms(
                embeddings=embeddings,
                idx_cluster=labels,
                centroids=center,
                scalograms=scalograms,
                num_K=num_K,
                experiment_group=experiment_group,
                experiment_subgroup=experiment_subgroup,
            )

        self.scalograms = scalograms
        self.channel_mode = channel_mode
        self.is_average_clustering = is_average_clustering
        self._load = _load
        self.embeddings = embeddings
        self.center = center
        self.labels = labels
        self.df_demo = df_demo
        self.df_demo_HI_psm_updated = df_demo_HI_psm_updated
        self.df_MR = df_MR
        self.df_MR_scalo_ins = df_MR_scalo_ins
        self.list_org = list_org
        self.list_add = list_add
        self._km = _km


def getLoadParams(
    experiment_group,
    experiment_subgroup,
    inclusion_option,
    check_inclusion_option,
    channel_mode="6ch",
):
    path_json_input_param = os.path.join(
        path_main,
        "results/%s/%s/%s.json" % (experiment_group, "load_params", experiment_subgroup),
    )
    with open(path_json_input_param, "r") as json_file:
        raw_params = json.load(json_file)

    resolved_channel_mode = _resolve_channel_mode(
        channel_mode=channel_mode,
        n_channels=raw_params.get("n_channels", None),
    )

    # Keep only keys supported by load._load; ignore training-only keys like n_channels.
    load_params = {
        "path_scalogram": raw_params.get("path_scalogram", None),
        "reset": False,
        "inclusion_option": raw_params.get("inclusion_option", "only_insomnia"),
        "verbose": raw_params.get("verbose", True),
        "include_MR": raw_params.get("include_MR", "no_use"),
    }

    if check_inclusion_option:
        if load_params["inclusion_option"] != inclusion_option:
            print("Use healthy_insomnia pretrained model, but load only_insomnia data")
        load_params["inclusion_option"] = inclusion_option

    return load_params, resolved_channel_mode


def getWholeScalogram(load_params, include_MR, channel_mode="6ch", gen_np_scalo=True):
    """
    1. Create _load class using load_params.
    2. Optionally append additional MR scalograms.
    """
    channel_mode = _resolve_channel_mode(channel_mode=channel_mode)

    safe_params = load_params.copy()
    safe_params["reset"] = False

    _load = load._load(**safe_params)

    # Prefer channel-mode specific npy. Fallback to legacy root path.
    base_fname = _load.fname_scalogram
    path_primary = os.path.join(path_np_data, channel_mode, base_fname)
    path_fallback = os.path.join(path_np_data, base_fname)

    scalograms, loaded_path = _load_npy_with_fallback(path_primary, path_fallback)
    print(f"Load original scalograms from: {loaded_path}")

    if include_MR:
        if gen_np_scalo:
            gen_df_MR.gen_additional_scalogram_npy(verbose=True, channel_mode=channel_mode)

        mr_fname = "scalogram_scalograms_with_MR_only_insomnia.npy"
        mr_primary = os.path.join(path_np_data, channel_mode, mr_fname)
        mr_fallback = os.path.join(path_np_data, mr_fname)
        scalograms_MR, mr_loaded_path = _load_npy_with_fallback(mr_primary, mr_fallback)
        print(f"Load MR scalograms from: {mr_loaded_path}")

        scalograms = np.vstack((scalograms, scalograms_MR))

    _load.scalograms = scalograms
    print("Shape of whole scalograms: {}".format(scalograms.shape))
    return scalograms, _load


def calculate_z_score(patient_data, control_data):
    control_mean = np.mean(control_data, axis=0)
    control_std = np.std(control_data, axis=0)
    control_std_safe = np.where(control_std == 0, 1, control_std)
    return (patient_data - control_mean) / control_std_safe


def getEmbeddings(experiment_group, experiment_subgroup, scalograms_ins, scalograms_hs=None):
    path_weights = os.path.join(path_main, "model_weights/%s/%s.h5" % (experiment_group, experiment_subgroup))
    aec, enc = eval_utils.load_my_model(path_weights)

    embeddings_ins = enc.predict(scalograms_ins)

    if scalograms_hs is not None:
        embeddings_hs = enc.predict(scalograms_hs)
        embeddings_zscore = calculate_z_score(embeddings_ins, embeddings_hs)
    else:
        scaler = StandardScaler()
        embeddings_zscore = scaler.fit_transform(embeddings_ins)

    print("Shape of embeddings: {}".format(embeddings_zscore.shape))
    return embeddings_zscore


def runGMM(num_K, embeddings, random_state):
    _gmm = GaussianMixture(
        n_components=num_K,
        covariance_type="full",
        max_iter=10000,
        verbose=False,
        tol=1e-3,
        init_params="kmeans",
        random_state=random_state,
    )
    _gmm.fit(embeddings)

    center_gmm = _gmm.means_
    labels_gmm = _gmm.predict(embeddings)
    return center_gmm, labels_gmm


def runKMenas(num_K, embeddings, random_state):
    _km = KMeans(n_clusters=num_K, random_state=random_state, init="k-means++")
    _km.fit(embeddings)

    center_kmeans = _km.cluster_centers_
    labels_kmeans = _km.labels_
    return center_kmeans, labels_kmeans, _km


def unique_cluster_test(labels):
    return len(np.unique(labels)) < 2


def runClustering(clustering_method, num_K, embeddings, random_state, B, is_average_clustering=False):
    print("\n ----- Clustering start")

    _km = None
    if not is_average_clustering:
        iter_cnt = 0
        labels = [0, 0]
        while True:
            if "G" in clustering_method.upper():
                center, labels = runGMM(num_K, embeddings, random_state)
                print("       --> clustering method: Gaussian Mixture Model")
            elif "K" in clustering_method.upper():
                center, labels, _km = runKMenas(num_K, embeddings, random_state)
                print("       --> clustering method: K Means Clustering")
            else:
                raise ValueError(f"Unknown clustering method: {clustering_method}")

            if not unique_cluster_test(labels):
                break
            if iter_cnt > 50:
                break

            random_state += 1
            iter_cnt += 1
            print("case of uniqueness of clustering #%d" % iter_cnt)
    else:
        center, labels = average_clustering(org_data=embeddings, K=num_K, B=B)

    print(" ----- Clustering end")
    return center, labels, _km


def average_clustering(org_data, K, B, verbose=True):
    _stability = stability(org_data=org_data, K=K, B=B)

    center_ensemble = _stability._orgClustering.center.copy()

    list_bootClustering = _stability.list_bootClustering
    for b in tqdm(range(B), desc="ensemble_clustering ..."):
        center_ensemble += list_bootClustering[b].B2O_center
    center_ensemble = center_ensemble / (B + 1)

    label_ensemble = cdist(center_ensemble, org_data, metric="euclidean").argmin(axis=0)

    if verbose:
        print("shape of center_ensemble: {}".format(center_ensemble.shape))
        print("shape of label_ensemble: {}".format(label_ensemble.shape))
        print("size of each cluster: " + K * "%d, " % tuple(sum(label_ensemble == i) for i in range(K)))

    return center_ensemble, label_ensemble
