import os, json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from modules.stability_scheme1 import stability

# path_main = os.getcwd().split('INS_clustering_v06')[0] + 'INS_clustering_v06'
path_main = 'D:/USC/01_code/insomnia_clustering'
path_csv = os.path.join(path_main, 'csv_files')
path_np_data = os.path.join(path_main, 'data')

cluster_ch = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def myPrint(_string, verbose=True):
    if verbose:
        print(_string)


def save_json(params, experiment_group, experiment_subgroup, jason_type):
    # 딕셔너리를 JSON 파일로 저장
    path_json = os.path.join(path_main, 'results/%s/%s' % (experiment_group, jason_type))
    os.makedirs(path_json, exist_ok=True)
    with open(os.path.join(path_json,
                           "%s.json" % experiment_subgroup),
              'w') as json_file:
        json.dump(params, json_file)


def getDistance_against_centroid(embeddings, centroid):
    distances = []
    for i in range(embeddings.shape[0]):
        distances.append(np.linalg.norm(embeddings[i] - centroid))
        
    return distances


def plot_scalogram(scalogram, ax, i=None, title=None):
    
    clim = (0.0, 1.5)

    im_input = ax.imshow(scalogram, cmap='hot', aspect='auto', clim=clim);
    ax.tick_params(bottom=False, top=False, left=False, right=False); 
    ax.set_xticklabels([]);
    ax.set_yticklabels([]);
    ax.invert_yaxis();
    
    if not(title==None) and (i==0):
        ax.text(200, 4,
                'centroid', size=25, 
                rotation='vertical',
                color='white',
                weight='bold')
    
    if not(i==None):       
        ax.text(9000, 17,
                "Cluster %s" % cluster_ch[i], size=25,
                weight='bold')
        
def plot_scalogram_custom(scalogram, ax, i=None, title=None, max_x=6):
    freqs = np.array([0.5, 0.7, 0.9, 1.2, 1.6, 2.1, 2.8, 3.8, 5.0, 6.7, 8.9, 11.9, 15.8, 21.1, 28.1, 37.5])
    freqs_half = freqs[np.arange(0, 16, 2)]
        
    clim = (0.0, 1.5)

    # 사용자 정의 색상 맵 생성
    # >> imshow의 cmap에 cmap을 전달 하면 됨
    # colors = ['#000000', '#7F3300', '#FF8C00', '#FFD700']
    # cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    time_bins = np.linspace(0, max_x, 2000)  # 시간 대역 설정 (0부터 6까지를 2000개로 나누어서)

    im_input = ax.imshow(scalogram, cmap='hot', aspect='auto', clim=clim, extent=[time_bins[0], time_bins[-1], 0, scalogram.shape[0]]);
    ax.tick_params(bottom=True, top=False, left=True, right=False); 
    # ax.set_xticklabels([]);
    ax.set_yticklabels([]);
    ax.invert_yaxis();
    ax.grid(False)
    ax.tick_params(axis='x', width=.9, labelsize=12)  # x축의 tick 두께 조절
    ax.spines['bottom'].set_linewidth(.9)  # x축의 두께 조절
    
    ax.yaxis.set_ticks(np.arange(0, 16, 2));
    ax.yaxis.set_ticklabels(freqs_half[::-1], fontsize=12);

    # colorbar 추가
    cbar = plt.colorbar(im_input, ax=ax, orientation='vertical')
    # cbar.set_label('Intensity', fontsize=20)  # colorbar에 대한 레이블 설정
    cbar.ax.tick_params(labelsize=15)  # colorbar 눈금의 폰트 크기 설정


def align_scalograms(embeddings, idx_cluster, centroids, scalograms, num_K, experiment_group, experiment_subgroup, num_align=10):
    idx_total = np.arange(embeddings.shape[0])
    if num_align != 1:
        fig, axs = plt.subplots(num_align, num_K, figsize=(30,13))
    else:
        fig, axs = plt.subplots(num_align, num_K, figsize=(8,3))

    for k in range(num_K):
        temp_scalograms = scalograms[idx_cluster == k]
        temp_embeddings = embeddings[idx_cluster == k]

        distances = getDistance_against_centroid(temp_embeddings, centroids[k])
        distances_argsort = np.argsort(distances) # np.argsort([1,3,2]) = [0, 2, 1]

        if temp_scalograms.shape[0] == 0:
            # k-th cluster에 포함된 sample의 개수가 0일 때, 그림을 그리지 않는다.
            continue


        if num_align == 1:
            temp_ax = axs[k]
            i = 0
            temp_idx = distances_argsort[0]
            print(temp_idx)
            plot_scalogram(scalogram=temp_scalograms[temp_idx],ax=temp_ax)
            temp_ax.grid(False)

            temp_ax.set_title("Cluster %s (n=%d)" % (cluster_ch[k], np.sum(idx_cluster==k)), fontsize=10, weight='bold')
                     
        else:
            for i in range(num_align):
                temp_ax = axs[i][k] # 현재 scalogram이 할당될 ax
                # temp_idx = (distances_argsort == int((len(distances)-1) - len(distances)*1/num_align*i))
                temp_idx = distances_argsort[int(len(distances)/num_align*i)]
                plot_scalogram(scalogram=temp_scalograms[temp_idx],ax=temp_ax)
                temp_ax.grid(False)
                
                if i == 0:
                    temp_ax.set_title("Cluster %s (n=%d)" % (cluster_ch[k], np.sum(idx_cluster==k)), fontsize=50, weight='bold')
                
                if k == 0:                
                    temp_ax.text(-0.07, 0.5, str(int(100/num_align*i))+"%", transform=temp_ax.transAxes, 
                        verticalalignment='center', size=25, color='black', weight='bold')
                
    fig.tight_layout()

    path_save = os.path.join(path_main, 'results/%s/align_scalograms' % (experiment_group))
    os.makedirs(path_save, exist_ok=True)
    plt.savefig(os.path.join(path_save, '%s.png' % experiment_subgroup))

def visualize_latent_space2(num_K, embeddings, labels, plot_only_pca=False):
    idx = np.arange(embeddings.shape[0])

    _stability = stability(org_data=embeddings, K=num_K, B=5)
    color_for_stab = _stability.jaccard_stabs[0]
    cmap = plt.cm.get_cmap('coolwarm')    
    
    # num_K에 따라서, 각 cluster에 할당된 sample의 index를 얻는다
    legends_character = ["A", "B", "C", "D", "E", "F", "G"]
    list_markers = ['o', '^']

    list_label = [] 
    for i in range(num_K):
        list_label.append(idx[_stability._orgClustering.labels == 0])
        
    # Initialize figure    
    fig, axs = plt.subplots(1, 2, figsize=(17, 5))
    fig.suptitle("Visualization of latent space")
    fig.tight_layout()
    
    # TSNE =======================================
    perplexity = min(30, embeddings.shape[0] - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(embeddings)
    
    for i in range(num_K):
        # scatter = axs[0].scatter(X_tsne[list_label[i], 0], X_tsne[list_label[i], 1], label='cluster %s' % legends_character[i])    
        scatter = axs[0].scatter(X_tsne[list_label[i], 0], X_tsne[list_label[i], 1], marker=list_markers[i], s=50, c=color_for_stab[list_label[i]], cmap=cmap, vmin=0, vmax=1)
    
    axs[0].set_title("t-SNE")
    axs[0].legend()
    # cbar = plt.colorbar(scatter)
    
    # PCA =======================================
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(embeddings)
    
    for i in range(num_K):
        # axs[1].scatter(X_pca[list_label[i], 0], X_pca[list_label[i], 1], label='cluster %s' % legends_character[i]) 
        scatter = axs[1].scatter(X_pca[list_label[i], 0], X_pca[list_label[i], 1], marker=list_markers[i], s=50, c=color_for_stab[list_label[i]], cmap=cmap, vmin=0, vmax=1)

    axs[1].set_title("PCA")
    axs[1].legend()
    cbar = plt.colorbar(scatter)

    if plot_only_pca:
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
        for i in range(num_K):
            ax.scatter(X_pca[list_label[i], 0], X_pca[list_label[i], 1], label='cluster %s' % legends_character[i]) 
            ax.set_title("PCA")
            ax.legend()

def visualize_latent_space(num_K, embeddings, labels, plot_only_pca=False):
    # num_K에 따라서, 각 cluster에 할당된 sample의 index를 얻는다
    legends_character = ["A", "B", "C", "D", "E", "F", "G"]
    list_markers = ['o', '^']
    
    list_label = [] 
    for i in range(num_K):
        list_label.append((labels==i))
        
    # Initialize figure    
    fig, axs = plt.subplots(1, 2, figsize=(17, 5))
    fig.suptitle("Visualization of latent space")
    fig.tight_layout()
    
    # TSNE =======================================
    perplexity = min(30, embeddings.shape[0] - 1)
    tsne = TSNE(n_components=3, random_state=0, perplexity=perplexity)
    X_tsne = tsne.fit_transform(embeddings)
    
    for i in range(num_K):
        axs[0].scatter(X_tsne[list_label[i], 0], X_tsne[list_label[i], 1], label='cluster %s' % legends_character[i], marker=list_markers[i])    
    
    axs[0].set_title("t-SNE")
    axs[0].legend()
    
    # PCA =======================================
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(embeddings)
    
    for i in range(num_K):
        axs[1].scatter(X_pca[list_label[i], 0], X_pca[list_label[i], 1], label='cluster %s' % legends_character[i], marker=list_markers[i]) 
    axs[1].set_title("PCA")
    axs[1].legend()

def visualize_latent_space(num_K, embeddings, labels, plot_only_pca=False):
    # stability 계산
    _stability = stability(org_data=embeddings, K=num_K, B=5)
    color_for_stab = _stability.jaccard_stabs[0]
    cmap = plt.cm.get_cmap('coolwarm') 


    # num_K에 따라서, 각 cluster에 할당된 sample의 index를 얻는다
    legends_character = ["A", "B", "C", "D", "E", "F", "G"]
    list_markers = ['o', '^']
    
    list_label = [] 
    for i in range(num_K):
        list_label.append((labels==i))
        
    # Initialize figure    
    fig, axs = plt.subplots(1, 2, figsize=(17, 5))
    fig.suptitle("Visualization of latent space")
    fig.tight_layout()
    
    # TSNE =======================================
    perplexity = min(30, embeddings.shape[0] - 1)
    tsne = TSNE(n_components=3, random_state=0, perplexity=perplexity)
    X_tsne = tsne.fit_transform(embeddings)
    
    for i in range(num_K):
        axs[0].scatter(X_tsne[list_label[i], 0], X_tsne[list_label[i], 1], label='cluster %s' % legends_character[i], marker=list_markers[i])    
    
    axs[0].set_title("t-SNE")
    axs[0].legend()
    
    # PCA =======================================
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(embeddings)
    
    for i in range(num_K):
        axs[1].scatter(X_pca[list_label[i], 0], X_pca[list_label[i], 1], label='cluster %s' % legends_character[i], marker=list_markers[i]) 
    axs[1].set_title("PCA")
    axs[1].legend()