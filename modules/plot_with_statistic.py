import pandas as pd

from scipy.stats import ttest_ind
from statannotations.Annotator import Annotator
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm.notebook import tqdm
import warnings
from modules import statistics
warnings.filterwarnings('ignore')

# Set the default font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

def addLabel_melt(df_demo):
    df = df_demo.copy()
    df["labels"] = df_demo["labels"]
    df.loc[df.labels==0, "cluster"] = "Cluster A"
    df.loc[df.labels==1, "cluster"] = "Cluster B"
    df.loc[df.labels==2, "cluster"] = "Cluster C"
    df.drop(columns='labels', inplace=True)

    df_melt = df.melt(var_name='feature_names',
                      value_name='vals',
                      value_vars=df.columns.to_list(),
                      id_vars=("cluster"))   
    
    df_melt['vals'] = df_melt['vals'].astype(float)

    return df_melt

def plot_each_feature3(feature, df_melt_feature, ax, dic_posthoc_p, yaxis_visible=True, is_legend=False, ylabel='', title=''):
    new_palette = ['#333333', '#666666', '#999999']  # Cluster C는 진한 회색, Cluster A는 중간 회색, Cluster B는 연한 회색으로 설정

    plotting_params = {"data": df_melt_feature,
                       "x": "cluster",
                       "y": 'vals',
                       "order": ['Cluster C', 'Cluster A', 'Cluster B'],
                       }

    pvals = dic_posthoc_p[feature]
    pairs = [('Cluster A', 'Cluster B'),
            ('Cluster B', 'Cluster C'),
            ('Cluster C', 'Cluster A')
            ]
    
    # seaborn으로 boxplot을 그릴 때, 배경을 흰색으로 설정하고, x축과 y축을 표시
    bar = sns.barplot(palette=new_palette, ax=ax, **plotting_params, linewidth=1.5, errorbar='se')
    
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.set_xlabel('Cluster')  # x축에 레이블 추가
    ax.set_ylabel(ylabel)     # y축에 레이블 추가
    ax.set_title(title)       # 그래프 제목 추가

    # bar = sns.boxplot(palette=new_palette, ax=ax, **plotting_params) # width 옵션은 작동 x, errcolor='white'

    if is_legend:
        legend_elements = [Patch(facecolor=new_palette[0], edgecolor=new_palette[0], label='Healthy'),
                           Patch(facecolor=new_palette[1], edgecolor=new_palette[1], label='INS A'),
                           Patch(facecolor=new_palette[2], edgecolor=new_palette[2], label='INS B')]
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.77,1))

def get_plot_params(df: pd.DataFrame, feature: str):
    df_feature = df.loc[:, [feature, 'labels']]
    df_feature_rm = statistics.rm_abnormal(df_feature, feature)
    df_melt_feature = addLabel_melt(df_feature_rm)

    # new_palette = ['#B0B0B0', '#1f77b4', '#76c7c0'] # grey set
    new_palette = ['#B0B0B0', '#cd5c5c', '#d2b48c'] # red set
    order = ['Cluster C', 'Cluster A', 'Cluster B']
    
    if len(df.labels.unique()) == 2:
        # two group case
        new_palette = ['#B0B0B0', '#cd5c5c']
        order = ['Cluster B', 'Cluster A']
        
    plotting_params = {"data": df_melt_feature,
                       "x": "cluster",
                       "y": 'vals',
                       "order": order,
                       "palette": new_palette,
                      }

    return plotting_params

def customize_ax(ax, feature, custom_ylabel=None, y_min=None, y_max=None, fontsize=15):
    ax.set_xlabel('');
    if not(custom_ylabel == None):
        ax.set_ylabel(custom_ylabel, fontsize=fontsize)
    else:
        ax.set_ylabel(make_feature_plot_txt(feature), fontsize=fontsize);
    ax.set_facecolor('white')
    sns.despine(ax=ax, top=True, right=True)
    ax.spines['bottom'].set_linewidth(.9)  # x축의 두께 조절
    ax.spines['left'].set_linewidth(.9)    # y축의 두께 조절
    ax.tick_params(axis='x', width=.9)  # x축의 tick 두께 조절
    ax.tick_params(axis='y', width=.9)  # y축의 tick 두께 조절
    
    groups = ['HS', 'SSM', "OSD"]
    if len(ax.get_xticklabels()) == 2:
        # two group case
        groups = ['HS', 'SSM']
    ax.set_xticklabels(groups, fontsize=fontsize)

    # y축 범위 설정
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    elif y_min is not None:
        ax.set_ylim(bottom=y_min)
    elif y_max is not None:
        ax.set_ylim(top=y_max)

    ax.tick_params(axis='y', labelsize=fontsize) # y-tick 레이블의 폰트 크기 조절

    # yticks = ax.get_yticks()
    # ax.set_yticklabels([f'{int(tick)}' for tick in yticks], fontsize=fontsize)
    
def make_feature_plot_txt(feature: str):
    if '_' in feature:
        return feature.split('_')[0] + ' ' + feature.split('_')[1]

def annot_stat(ax, plotting_params, feature, dic_posthoc_p, pairs=None, fontsize=15, p_threhold=0.05):
    pvals_hold = dic_posthoc_p[feature]
    pairs_hold = [('Cluster A', 'Cluster B'),
                  ('Cluster B', 'Cluster C'),
                  ('Cluster C', 'Cluster A')
                 ]

    if pairs == None:
        pairs = []
        pvals = []
        for i, pval in enumerate(pvals_hold):
            # pvalue가 유의한 pair만 추가
            # 유의하지 않은 pair는 plot에 표시하지 않게 하기 위함
            if pval < p_threhold:
                pairs.append(pairs_hold[i])
                pvals.append(pvals_hold[i])

    if len(pvals) < 1:
        # significant pair가 없는 경우, pvals와 pairs는 비어 있는 리스트임
        # 이 상태로 Annotator에 전달되면 오류 발생
        # 유의한 pair가 없는 경우 본 함수는 여기서 종료
        return 
        
    formatted_pvalues = [statistics.get_significance_asterisk(p) for p in pvals]
    
    annotator = Annotator(ax, pairs, verbose=False, loc='outside', **plotting_params);
    # annotator.line.set_linewidth(0.1)  # 선의 두께 조절
    annotator.set_pvalues(pvals);
    annotator.configure(loc="outside", fontsize=fontsize)
    annotator.set_custom_annotations(formatted_pvalues)
    annotator.annotate()

def stat_boxplot(df: pd.DataFrame, feature: str, ax, dic_posthoc_p, custom_ylabel=None, y_min=None, y_max=None, fontsize=15):
    sns.set(style='ticks', context='talk', font_scale=1.2, font='Helvetica')

    plotting_params = get_plot_params(df, feature)

    sns.boxplot(showfliers=False, # flier 표시 여부
                linewidth=0.5, # boxplot의 테두리 두께
                ax=ax,
                **plotting_params)

    customize_ax(ax, feature, custom_ylabel, y_min, y_max, fontsize=fontsize)
    annot_stat(ax, plotting_params, feature, dic_posthoc_p)

def stat_barplot(df: pd.DataFrame, feature: str, ax, dic_posthoc_p, custom_ylabel=None, y_min=None, y_max=None, fontsize=15):
    sns.set(style='ticks', context='talk', font_scale=1.2, font='Helvetica')

    plotting_params = get_plot_params(df, feature)

    sns.barplot(errorbar='se', # flier 표시 여부
                linewidth=0.5, # boxplot의 테두리 두께
                ax=ax,
                **plotting_params)
    
    for bar in ax.lines:
        bar.set_linewidth(1.0)  # 여기서 오류 막대의 두께를 설정

    customize_ax(ax, feature, custom_ylabel, y_min, y_max, fontsize=fontsize)
    annot_stat(ax, plotting_params, feature, dic_posthoc_p)