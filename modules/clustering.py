import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from modules.utils import path_main, path_np_data
from modules import load
from modules import gen_df_MR
from modules import autoencoders
from modules import utils
from modules.stability_scheme1 import stability
    
def get_scalo_hs():
    # df_demo_psm의 Healthy id 획득
    # df_demo에서 해당 id의 index 획득
    # scalogram_2000t_16f_healthy_insomnia.npy 에서 indexing
    
    np_hs_ins = np.load('../data/scalogram_2000t_16f_healthy_insomnia.npy')
    
    df_demo = pd.read_csv('../csv_files/df_demo_2000t_16f_healthy_insomnia.csv', encoding='euc-kr')
    df_demo_psm = pd.read_csv('../csv_files/df_demo_HI_psm.csv', encoding='euc-kr')

    # healthy group index 구하기
    psgid_HS = df_demo_psm.loc[df_demo_psm.labels == 2, 'PSG study Number#'].values
    con_HS = df_demo['PSG study Number#'].isin(psgid_HS)
    idx_HS = df_demo[con_HS].index.to_list()
    scalogram_HS = np_hs_ins[idx_HS]
    
    return scalogram_HS


class class_clustering():
    """
    1. Get load_params
    --> load the saved .json file which contains information used for input parameters of _load class in the function, 'getWholeScalogram'

    2. Embeddings from autoencoder
    --> Extract emeddings of scalograms using pretrained autoencoder
    --> input: scalograms of insomnia (subjects in df_demo (n=682) + additional subjects with MR features (n=13))

    3. Clustering
    - Apply clustering methods to data
    """
    def __init__(self, experiment_group, experiment_subgroup, clustering_method, num_K, 
                 random_state, include_MR, inclusion_option, check_inclusion_option=False, 
                 is_plot_scalogram=False, B=10, is_average_clustering=False):
        load_params = getLoadParams(experiment_group, experiment_subgroup, inclusion_option, check_inclusion_option)
        scalograms, _load = getWholeScalogram(load_params, include_MR)

        # scalograms_hs, scalograms_ins = split_scalo_hs_ins(scalograms=scalograms)
        scalograms_hs = get_scalo_hs()
        scalograms_ins = scalograms
        
        embeddings = getEmbeddings(experiment_group, experiment_subgroup, scalograms_ins, scalograms_hs)
        center, labels, _km = runClustering(clustering_method, num_K, embeddings, random_state, B=B, is_average_clustering=is_average_clustering) # clustering include additional scalograms with MR features

        df_MR = gen_df_MR.get_df_MR()
        # df_MR_scalo_ins = df_MR.loc[(df_MR.ISI >= 15) & df_MR.is_mr_scalo, ] # select the case with both insomnia and MR features
        df_MR_scalo_ins = df_MR.loc[df_MR.is_mr_scalo, ] # ISI 기반 subject select (x), is_mr_scalo만 고려
        list_org, list_add = gen_df_MR.get_id_with_MR_org_add(df_MR_scalo_ins)

        df_demo = _load.df_demo.copy() # get df_demo 
        df_demo['labels'] = labels[:-len(list_add)] # add 'labels' column

        # --- manual modification start 
        if 'G' in clustering_method.upper():
                # experiment_subgroup = experiment_subgroup.replace('K', 'G')
                experiment_subgroup = experiment_subgroup + '_G_K%d' % num_K
        elif 'K' in clustering_method.upper():
                experiment_subgroup = experiment_subgroup + '_K_K%d' % num_K

        if check_inclusion_option:
            experiment_group = experiment_group + '_' + inclusion_option
        # --- manual modification start 
            
        if is_plot_scalogram:
            # plot only centroid
            utils.align_scalograms(embeddings=embeddings,
                                idx_cluster=labels,
                                centroids=center,
                                scalograms=scalograms,
                                num_K=num_K,
                                experiment_group=experiment_group,
                                experiment_subgroup=experiment_subgroup,
                                num_align=1)
        
            # plot aligned scalograms
            utils.align_scalograms(embeddings=embeddings,
                                idx_cluster=labels,
                                centroids=center,
                                scalograms=scalograms,
                                num_K=num_K,
                                experiment_group=experiment_group,
                                experiment_subgroup=experiment_subgroup)

        self.scalograms = scalograms
        self.is_average_clustering = is_average_clustering
        self._load = _load
        self.embeddings = embeddings
        self.center = center
        self.labels = labels
        self.df_demo = df_demo
        self.df_MR = df_MR
        self.df_MR_scalo_ins = df_MR_scalo_ins
        self.list_org = list_org
        self.list_add = list_add
        self._km = _km


def getLoadParams(experiment_group, experiment_subgroup, inclusion_option, check_inclusion_option): 
    # .json으로 저장되어 있는 load_params 로드
    path_json_input_param = os.path.join(path_main, "results/%s/%s/%s.json" % (experiment_group, "load_params", experiment_subgroup))
    with open(path_json_input_param, 'r') as json_file:
        load_params = json.load(json_file)

    if check_inclusion_option:
        # 학습할 때, healthy_insomnia를 사용하더라도, check_inclusion_option을 사용하여 clustering에 사용할 subject scope를 지정할 수 있음
        # --> 지정된 subject scope를 반영하기 위하여 load_params의 inclusion_option 값을 변경

        if not load_params['inclusion_option'] == inclusion_option: 
            print("Use healthy_insomnia pretrained model, but load only_insomnia data")
        
        load_params['inclusion_option'] = inclusion_option

    return load_params

def getWholeScalogram(load_params, include_MR, gen_np_scalo=True):
    '''
    1. Create _load class using a input parameter, load_params

    2. if include_MR is True, load additional scalograms which also provide MR features
    - MR-containing scalograms consist of two part, original and additional part
    - a) original scalograms (n=5), already included in df_demo so that used for training autoencoder
    - b) additional scalograms (n=13), newly added, making them not utilized in training autoencoder but clustering

      *  Get df_MR
         --> use the external function gen_df_MR.get_df_MR()
         --> df_MR contains the PSG_id and MR features of subjects who simultaneously provide a scalogram
    '''
    load_params['reset'] = False
    # load original scaalograms
    _load = load._load(**load_params)
    scalograms = _load.scalograms

    if include_MR:
        # load additional scalograms with MR features from subjects of insomnia

        if gen_np_scalo:
            # np_data 경로에, additional scalograms (n=13)에 해당하는 .npy 생성
            gen_df_MR.gen_additional_scalogram_npy(verbose=True) 

        scalograms_MR = np.load(os.path.join(path_np_data, "scalogram_scalograms_with_MR_only_insomnia.npy"))
        scalograms = np.vstack((_load.scalograms, scalograms_MR))

    print("Shape of whole scalograms: {}".format(scalograms.shape))
    return scalograms, _load

def calculate_z_score(patient_data, control_data):
    control_mean = np.mean(control_data, axis=0)
    control_std = np.std(control_data, axis=0)
    control_std_safe = np.where(control_std == 0, 1, control_std)
    return (patient_data - control_mean) / control_std_safe

def getEmbeddings(experiment_group, experiment_subgroup, scalograms_ins, scalograms_hs=None):
    # load pretrained Convolutional Autoencoder
    path_weights = os.path.join(path_main, "model_weights/%s/%s.h5" % (experiment_group, experiment_subgroup))
    aec, enc = autoencoders.load_my_model(path_weights)

    # get embeddings from encoder
    embeddings_ins = enc.predict(scalograms_ins)
    
    if not(scalograms_hs is None):
        embeddings_hs = enc.predict(scalograms_hs)
        embeddings_zscore = calculate_z_score(embeddings_ins, embeddings_hs)
    else:
        scaler = StandardScaler()
        embeddings_zscore = scaler.fit_transform(embeddings_ins)
        
    print("Shape of embeddings: {}".format(embeddings_zscore.shape))

    return embeddings_zscore

def runGMM(num_K, embeddings, random_state):
    # Perform Gaussian Mixture Model Clustering
    _gmm = GaussianMixture(n_components=num_K,
                           covariance_type='full',
                           max_iter=10000,
                           verbose=False,
                           tol=1e-3,
                           init_params='kmeans',
                           random_state=random_state)
    _gmm.fit(embeddings)

    center_gmm = _gmm.means_
    labels_gmm = _gmm.predict(embeddings)

    return center_gmm, labels_gmm

def runKMenas(num_K, embeddings, random_state):
    # Perform K-Means Clustering
    _km = KMeans(n_clusters=num_K, random_state=random_state, init="k-means++")
    _km.fit(embeddings)

    center_kmeans = _km.cluster_centers_
    labels_kmeans = _km.labels_  

    return center_kmeans, labels_kmeans, _km

def unique_cluster_test(labels):
    if len(np.unique(labels)) < 2:
        return True
    else:
        return False
    
def runClustering(clustering_method, num_K, embeddings, random_state, B, is_average_clustering=False):
    print('\n ----- Clustering start')

    if not(is_average_clustering):
        iter = 0
        labels = [0, 0] # initial label
        while 1:
            if 'G' in clustering_method.upper():
                center, labels = runGMM(num_K, embeddings, random_state)
                print("       --> clustering method: Gaussian Mixture Model")
            elif 'K' in clustering_method.upper():
                center, labels, _km = runKMenas(num_K, embeddings, random_state)
                print("       --> clustering method: K Means Clustering")

            # Test that labels of clustering form uniquness 
            # uniqueness means only one cluster after clustering, which is not expected result
            if not unique_cluster_test(labels):
                break
            elif iter > 50:
                break
            else:
                random_state += 1 
                iter += 1
                print("case of uniqueness of clustering #%d" % iter)
    else:
        center, labels = average_clustering(org_data=embeddings, K=num_K, B=B) # average clustering

    print(' ----- Clustering end')
    return center, labels, _km

def unique_cluster_test(labels):
    if len(np.unique(labels)) < 2:
        return True
    else:
        return False

def average_clustering(org_data, K, B, verbose=True):
    """
    - orginal clustering과 모든 bootstrapped clustering으로부터 각 cluster center 추출
    - 모든 cluster center를 평균내어서 ensemble center 계산 --> center_ensemble
    - ensemble center기반으로 original data의 cluster label assign --> label_ensemble
    """
    _stability = stability(org_data=org_data, K=K, B=B)
    
    center_ensemble = _stability._orgClustering.center.copy() # center_ensemble 초기 선언 as cneter of orgClustering
    
    list_bootClustering = _stability.list_bootClustering
    for b in tqdm(range(B), desc='ensemble_clustering ...'):
        # list_bootClustering에 포함된 모든 bootClustering에 대해 반복
        center_ensemble += list_bootClustering[b].B2O_center # boot2org mapping이 적용된 center 추출
    center_ensemble = center_ensemble/(B+1)
    
    label_ensemble = cdist(center_ensemble, org_data, metric='euclidean').argmin(axis=0)
    
    if verbose:
        print("shape of center_ensemble: {}".format(center_ensemble.shape))
        print("shape of label_ensemble: {}".format(label_ensemble.shape))
        # print("size of each cluster: %d, %d, %d" % (sum(label_ensemble==0), sum(label_ensemble==1), sum(label_ensemble==2)))
        print("size of each cluster: " + K*"%d, " % tuple(sum(label_ensemble==i) for i in range(K)))
    
    return center_ensemble, label_ensemble
