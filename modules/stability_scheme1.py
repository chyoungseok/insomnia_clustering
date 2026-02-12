import os, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm
from scipy.spatial.distance import cdist


def explore_new_label(dist_btween_centers, nk):
    """

    """
    
    all_false = [False] * nk
    new_label = np.ones(nk) * -1 # B clustering의 기존 label을 어떤 label로 바꿔야 할지에 대한 정보
    # new_label[0] = 1 인 경우, 0번 cluster는 1번 으로 label 변경
    
    for i in range(nk):
        min_idx = dist_btween_centers.argmin() # 전체 value 중, minimum value의 location
        min_idx_i = min_idx // nk # min value가 위치한 row (i th cluster label in bootstrap data)
        min_idx_j = min_idx - min_idx_i * nk # min value가 위치한 column (j th cluster label in original data)
        
        if min_idx_j in new_label:
            # min_idx_j가 이미 등장한 경우, inf로 설정해줌으로써 min에 의해 탐지되지 않도록 함
            dist_btween_centers[min_idx_i, min_idx_j] = np.inf
            min_idx = dist_btween_centers.argmin()
            min_idx_i = min_idx // nk
            min_idx_j = min_idx - min_idx_i * nk
        
        new_label[min_idx_i] = min_idx_j
        
        # i-th cluster of boostrap data는 이미 배정됐으니, min 값 탐색 후보에서 제외
        # min 값 탐색 후보에서 제외하기 위하여 inf로 설정
        all_false[min_idx_i] = True
        dist_btween_centers[all_false] = np.inf 
    
    return new_label.astype(int)

def map_B_center_to_O_center(org_data, O_centers, O_lables, B_centers, B2O_mapping_method, nk):
    """
        Bootstrap data로부터 구한 center를 Original data의 center에 matching 시키기
        
        sim_method에 따라서 mapping method가 다름
    """
    if B2O_mapping_method == 'euclidean':
        distances = cdist(B_centers, O_centers, metric='euclidean') # distance matrix between two centers
        # for row i and column j,
        # [i, j] --> distance between i_th B_center and j_th O_center
        dist_btween_centers = distances.copy() # distances 복제
        
        
    elif B2O_mapping_method == 'jaccard':
         # original data와 B_centroids 중, 가장 거리가 짧은 centroid가 속한 cluster로 labeling
        o2b_labels = cdist(B_centers, org_data, metric='euclidean').argmin(axis=0)
        
        distances = np.ones((nk,nk)) # 두 클러스터 간 jaccard coefficients를 기록하는 matrix
        idx = np.arange(org_data.shape[0]) # data크기에 해당하는 index array
        for i in range(nk):
            for j in range(nk):
                O_set = set(idx[O_lables == i]) # cluster label이 i인 index from original clustering label
                o2b_set = set(idx[o2b_labels == j]) # cluster label이 j인 index from ob2 clustering label
                
                intersection = len(O_set.intersection(o2b_set)) # 교집합
                union = len(O_set.union(o2b_set)) # 합집합
                
                jaccard = float(intersection / union) # jaccard coefficients
                
                distances[i, j] = -jaccard # matrix update
                # minimum 값을 찾아가는 explore_new_label() 알고리즘에 맞추기 위하여 (-)를 취해줌
                # jaccard는 값이 클수록 similarity가 크기 때문
        dist_btween_centers = distances.copy()
                
    new_label = explore_new_label(dist_btween_centers=dist_btween_centers, nk=nk)
    new_center = B_centers[new_label]
            
    return  new_center

def cal_jaccard(org_set, o2b_set):
    return float(len(org_set.intersection(o2b_set)) / len(org_set.union(o2b_set)))    

class clustering_algs():
    def __init__(self, data, clst_alg, K, random_state=None) -> None:
        self.clst_alg = clst_alg
        
        if clst_alg == 'kmeans':
            _km = KMeans(n_clusters=K, random_state=random_state, init="k-means++")
            _km.fit(data)
            
            self.center = _km.cluster_centers_
            self.B2O_center = _km.cluster_centers_ * 0 # _bootClustering.B2O_center = mapped_center  을 통해 업데이트 되어야 함
            # stability_scheme1.py의 line88 참고
            self.labels = _km.labels_        
            self.data = data 



class stability():
    """ Input parameters
        - org_data : an original dataset
        - K : number of clusters (default=2)
        - clsg_alg : an algorithm to perform clustering (default="kmeans")
        - B : a total number of the bootstrapping (B=10)
        - sim_method : how to measure a similarity between samples (default="euclidean")
    """
    
    @staticmethod
    def getStabilities(stability_matrix, _orgClustering, K):
        '''
            input: stability matrix; shape=(B,n); B=number of bootstrapping; n=number of data
            
            output: list of stabilities
                [observation_wise, cluster_wise, overall]
        '''
        observation_wise = np.mean(stability_matrix, axis=0)
        cluster_labels = _orgClustering.labels
        cluster_wise = [observation_wise[cluster_labels == ki].mean() for ki in range(K)]
        overall = np.mean(observation_wise)
        return [observation_wise, cluster_wise, overall]

    
    @staticmethod
    def plot_K_optimization(data, max_K=9, B=5):
        Smins = []
        for k in range(2, max_K+1):
            _stability = stability(org_data=data, K=k, B=B)
            Smins.append(_stability.Smin)
            
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        plt.plot(range(2,max_K+1), Smins, marker='o', label="S$_{min}$");
        plt.legend()
        plt.ylabel("S$_{min}$", fontdict={'fontsize':15})
        plt.xlabel("K (number of clusters)", fontdict={'fontsize':15})
        plt.title("Optimize the K (Bootstrapping=%d times for each K)" % B)
        ax.set_xticks(range(2,max_K+1))

    @staticmethod
    def getJaccard_scheme2(_orgClustering, list_bootClustering, B, K):
        idx = np.arange(_orgClustering.data.shape[0])
        total_clustering = [_orgClustering] + list_bootClustering

        obs_jaccard_for_each_ref = np.empty((B+1, len(idx)))
        clust_jaccard_for_each_ref = np.empty((B+1, K))
        over_jaccard_for_each_ref = np.empty((B+1, 1))
        
        for r in range(B+1):
            ref_clustering = total_clustering[r]  # reference clustering 지정

            stability_matrix_jaccard = np.empty((B, len(idx)))
            for b in range(B):
                if b >= r:
                    b += 1

                r2b_labels = cdist(total_clustering[b].center, total_clustering[r].data, metric='euclidean').argmin(axis=0)  # reference data를 bootClustering의 center에 mapping
                
                for i in range(len(idx)):
                    temp_ref_label = ref_clustering.labels[i]  # x_i가 origianl clustering에서 가지는 label
                    temp_r2b_label = r2b_labels[i]  # x_i가 boot clustering에서 가지는 label

                    ref_set = set(idx[ref_clustering.labels == temp_ref_label])  # x_i와 같은 refCluster에 있는 data members
                    r2b_set = set(idx[r2b_labels == temp_r2b_label])  # x_i와 같은 bootCluster에 있는 data members

                    # Jaccard based stability
                    stability_matrix_jaccard[b if b < r else b-1, i] = cal_jaccard(ref_set, r2b_set)
            
            temp_ref_stabs = stability.getStabilities(stability_matrix_jaccard, _orgClustering=ref_clustering, K=K)
            obs_jaccard_for_each_ref[r] = temp_ref_stabs[0]
            clust_jaccard_for_each_ref[r] = temp_ref_stabs[1]
            over_jaccard_for_each_ref[r] = temp_ref_stabs[2]

        obs_jaccard_scheme2 = np.mean(obs_jaccard_for_each_ref, axis=0)
        clust_jaccard_scheme2 = np.mean(clust_jaccard_for_each_ref, axis=0)
        over_jaccard_scheme2 = np.mean(over_jaccard_for_each_ref)
        return [obs_jaccard_scheme2, clust_jaccard_scheme2, over_jaccard_scheme2]



    @staticmethod
    def getSmin_scheme2(_orgClustering, list_bootClustering, B, K):
        Smin_scheme2_for_each_ref = np.empty((B+1, 1)) 
        total_clustering = [_orgClustering] + list_bootClustering
        idx = np.arange(_orgClustering.data.shape[0])

        for r in range(B+1):
            ref_clustering = total_clustering[r]
            stability_matrix_cluster_wise_jaccard_scheme2 = np.empty((B, K))

            for b in range(B):
                if b >= r:
                    b += 1
                
                r2b_labels = cdist(total_clustering[b].center, total_clustering[r].data, metric='euclidean').argmin(axis=0)
                
                for k in range(K):
                    ref_set = set(idx[ref_clustering.labels == k])
                    if len(ref_set) == 0:
                        print("refset is zero %d" % k)
                        continue

                    cluster_wise_similarity = sum(
                        cal_jaccard(ref_set, set(idx[r2b_labels == r2b_labels[idx_for_tempK]])) 
                        for idx_for_tempK in ref_set
                    ) / len(ref_set)
                    
                    stability_matrix_cluster_wise_jaccard_scheme2[b if b < r else b-1, k] = cluster_wise_similarity
            
            Smin_scheme2_for_each_ref[r] = np.mean(np.min(stability_matrix_cluster_wise_jaccard_scheme2, axis=1))

        Smin_scheme2 = np.mean(Smin_scheme2_for_each_ref)
        return Smin_scheme2


            
    
    def __init__(self, org_data, K=2, B=10, B2O_mapping_method='jaccard', clst_alg='kmeans') -> None:
        self.org_data = org_data
        self.K = K
        self.B = B
        # self.sim_method = sim_method
        self.clst_alg = clst_alg
        self.B2O_mapping_method = B2O_mapping_method

        # Clustering for the original data
        _orgClustering = clustering_algs(data = org_data,
                                         clst_alg = clst_alg,
                                         K = K)
                                        #  sim_method = sim_method)
        
        # Clustering for the each of the bootstrapped data
        list_bootClustering = []
        stability_matrix_naive = np.empty((B, org_data.shape[0]))
        stability_matrix_jaccard = np.empty((B, org_data.shape[0]))
        stability_matrix_cluster_wise_jaccard = np.empty((B, K))
        for b in tqdm(range(B), desc="Bootstrapping for K=%d..." % K):
            resample_idx = np.random.choice(org_data.shape[0], size=org_data.shape[0], replace=True)
            boot_data = org_data[resample_idx, ]
            
            # clustering for the bootstrapped data
            _bootClustering = clustering_algs(data = boot_data,
                                              clst_alg = clst_alg,
                                              K = K,
                                              random_state = b)
                                            #   sim_method = sim_method,
                                              
            
            # mapping B_centers into O_centers
            mapped_center = map_B_center_to_O_center(org_data=org_data,
                                                     O_centers = _orgClustering.center,
                                                     O_lables = _orgClustering.labels,
                                                     B_centers = _bootClustering.center,
                                                     B2O_mapping_method = B2O_mapping_method,
                                                     nk = K)
            _bootClustering.B2O_center = mapped_center # B2O_center update          
            list_bootClustering.append(_bootClustering)
            
            o2b_labels = cdist(_bootClustering.center, org_data, metric='euclidean').argmin(axis=0) # original data를 bootClustering의 center에 mapping
            mapped_o2b_labels = cdist(mapped_center, org_data, metric='euclidean').argmin(axis=0) # original data를 mapped bootClustering의 center에 mapping
            idx = np.arange(org_data.shape[0])
            
            # get stability matrix for naive and jaccard based methods
            for i in range(org_data.shape[0]):
                temp_org_label = _orgClustering.labels[i] # x_i가 origianl clustering에서 가지는 label
                temp_o2b_label = o2b_labels[i] # x_i가 boot clustering에서 가지는 label
                temp_mapped_o2b_label = mapped_o2b_labels[i] # x_i가 mapped boot clustering에서 가지는 label
                
                org_set = set(idx[_orgClustering.labels == temp_org_label]) # x_i와 같은 orgCluster에 있는 data members
                o2b_set = set(idx[o2b_labels == temp_o2b_label]) # x_i와 같은 bootCluster에 있는 data members
                mapped_o2b_set = set(idx[mapped_o2b_labels == temp_mapped_o2b_label]) # x_i와 같은 o2bCluster에 있는 data members
                
                # Naive stability
                if len(org_set.intersection(mapped_o2b_set)) == len(org_set):
                    stability_matrix_naive[b, i] = 1
                else:
                    stability_matrix_naive[b, i] = 0
                    
                # Jaccard based stability
                stability_matrix_jaccard[b, i] = cal_jaccard(org_set, o2b_set)
                
            
            # Smin (scheme1)
            # Cluster-wise jaccard based stability 
            for k in range(K):
                cluster_wise_similarity = 0
                org_set = set(idx[_orgClustering.labels == k]) # k 번째 org cluster에 포함된 모든 sample들의 index
                for idx_for_tempK in org_set: # org_list의 모든 sample에 대한 반복문
                    temp_label = o2b_labels[idx_for_tempK] # temp sample의 o2b_label
                    o2b_set = set(idx[o2b_labels == temp_label]) # temp sample이 포함되어 있는 o2b cluster의 모든 sample의 index
                    
                    cluster_wise_similarity += cal_jaccard(org_set, o2b_set)
                
                stability_matrix_cluster_wise_jaccard[b, k] = cluster_wise_similarity/len(org_set)

        Smin_scheme1 = np.mean(np.min(stability_matrix_cluster_wise_jaccard, axis=1)) # calculate Smin_scheme1

        # jaccard (scheme2); observation, cluster, and overall level
        jaccard_stabs_scheme2 = self.getJaccard_scheme2(_orgClustering, list_bootClustering, B, K)

        # Smin (scheme2)
        Smin_scheme2 = self.getSmin_scheme2(_orgClustering, list_bootClustering, B, K)

        self._orgClustering = _orgClustering
        self.list_bootClustering = list_bootClustering
        self.stability_matrix_naive = stability_matrix_naive
        self.stability_matrix_jaccard = stability_matrix_jaccard
        self.stability_matrix_cluster_wise_jaccard = stability_matrix_cluster_wise_jaccard
        self.naive_stabs = self.getStabilities(stability_matrix_naive, _orgClustering, K=K)
        self.jaccard_stabs = self.getStabilities(stability_matrix_jaccard, _orgClustering, K=K)
        self.jaccard_stabs_scheme2 = jaccard_stabs_scheme2
        self.Smin_scheme1 = Smin_scheme1
        self.Smin_scheme2 = Smin_scheme2
        

def search_opt_K(latent_vector, B, minK=2, maxK=9, is_user_input=True, path_save=None, experiment_name=None):
    """
    Stability profile을 통해서, optimal clustering number (K) 탐색
    """
    opt_K = 3
    
    # cluster-wise stability 
    naive_stab = []
    jaccard_stab = []
    jaccard_stab_scheme2 = []
    Smins_scheme1 = []
    Smins_scheme2 = []
    for k in range(minK, maxK):
        _stability = stability(org_data=latent_vector, K=k, B=B)
        naive_stab.append(_stability.naive_stabs[2])
        jaccard_stab.append(_stability.jaccard_stabs[2])
        jaccard_stab_scheme2.append(_stability.jaccard_stabs_scheme2[2])
        Smins_scheme1.append(_stability.Smin_scheme1)
        Smins_scheme2.append(_stability.Smin_scheme2)
        
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(131)
    plt.plot(range(2, maxK), naive_stab, marker='o');
    plt.plot(range(2, maxK), jaccard_stab, color='orange', marker='o');
    plt.plot(range(2, maxK), jaccard_stab_scheme2, color='green', marker='o');
    plt.legend(("naive", "jaccard_scheme1", "jaccard_scheme2"))

    ax = fig.add_subplot(132)
    plt.plot(range(2,maxK), Smins_scheme1, marker='o', color='orange', label="S$_{min}$_(scheme1)");
    plt.legend()
    plt.ylabel("S$_{min}$_(scheme1)")
    plt.xlabel("K")
    
    ax = fig.add_subplot(133)
    plt.plot(range(2,maxK), Smins_scheme2, marker='o', color='green', label="S$_{min}$_(scheme2)");
    plt.legend()
    plt.ylabel("S$_{min}$_(scheme2)")
    plt.xlabel("K")
    
    if path_save != None:
        os.makedirs(path_save, exist_ok=True)
        plt.savefig(os.path.join(path_save, experiment_name))
    
    if is_user_input:
        plt.show()
        user_input = input("Decide the optimal 'K' to proceed: ")
        plt.close()
        opt_K = int(user_input)
        print("--> Your optimal K is %d !\n" % opt_K)
    
    
    return opt_K, jaccard_stab_scheme2, Smins_scheme1, Smins_scheme2
