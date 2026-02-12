import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import scikit_posthocs as sp
from pingouin import pairwise_gameshowell
import pandas as pd
import numpy as np
from modules import load

def posthoc_tukey_hsd(df, feature):
    df = df.copy()
    # 정규성 (O), 등분산성 (O), equal sample size (O)
    tukey_hsd_result = pairwise_tukeyhsd(endog=df[feature], groups=df['labels'], alpha=0.05)
    tukey_hsd_p_values = pd.DataFrame(data=tukey_hsd_result._results_table.data[1:], columns=tukey_hsd_result._results_table.data[0])

    # Tukey's HSD p-values 추출 및 저장
    tukey_hsd_p_values.set_index(['group1', 'group2'], inplace=True)
    p_posthoc = [tukey_hsd_p_values.loc[(0, 1), 'p-adj'],
                 tukey_hsd_p_values.loc[(1, 2), 'p-adj'],
                 tukey_hsd_p_values.loc[(0, 2), 'p-adj']
                ]
    
    return p_posthoc

def posthoc_fisher_lsd(df: pd.DataFrame, feature: str):
    df = df.copy()
    df = load.rm_abnormal(df, [feature])
    # 정규성 (O), 등분산성 (O), equal sample size (X)
    unique_labels = np.sort(df['labels'].unique())
    p_values = {}

    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            group1 = df[df['labels'] == unique_labels[i]][feature]
            group2 = df[df['labels'] == unique_labels[j]][feature]
            t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=True)  # 등분산 가정
            p_values[f'{unique_labels[i]} vs {unique_labels[j]}'] = p_val
    
    p_values = pd.DataFrame(index=p_values.keys(), data=p_values.values()) 
    p_posthoc = [p_values.loc['0 vs 1'].values[0],
                 p_values.loc['1 vs 2'].values[0],
                 p_values.loc['0 vs 2'].values[0]]
    p_posthoc = [round(p, 4) for p in p_posthoc]
    return p_posthoc

def posthoc_scheffe(df: pd.DataFrame, feature: str):
    df = df.copy()
    # 정규성 (O), 등분산성 (O), equal sample size (X)
    scheffe_p_values = sp.posthoc_scheffe(df, val_col=feature, group_col='labels')
    
    scheffe_p_values.columns = scheffe_p_values.columns.astype(int)  # 컬럼 타입을 숫자로 변경
    p_posthoc = [scheffe_p_values.at[0, 1],
                 scheffe_p_values.at[1, 2],
                 scheffe_p_values.at[0, 2]
                ]
    p_posthoc = [round(p, 4) for p in p_posthoc]
    return p_posthoc

def posthoc_bonferroni(df: pd.DataFrame, feature: str):
    df = df.copy()
    # 정규성 (O), 등분산성 (O), equal sample size (X)
    mc_bonferroni = MultiComparison(df[feature], df['labels'])
    bonferroni_summary = mc_bonferroni.allpairtest(stats.ttest_ind, method='bonf')[0]
    bonferroni_summary_data = pd.DataFrame(data=bonferroni_summary.data[1:], columns=bonferroni_summary.data[0])
    bonferroni_p_values = bonferroni_summary_data[['group1', 'group2', 'pval_corr']]

    bonferroni_p_values.set_index(['group1', 'group2'], inplace=True)
    p_posthoc = [bonferroni_p_values.loc[(0, 1), 'pval_corr'],
                                bonferroni_p_values.loc[(1, 2), 'pval_corr'],
                                bonferroni_p_values.loc[(0, 2), 'pval_corr']
                                ]
    p_posthoc = [round(p, 4) for p in p_posthoc]
    return p_posthoc

def posthoc_games_howell(df: pd.DataFrame, feature: str):
    df = df.copy()
    # 정규성 (O), 등분산성 (O), equal sample size (X)
    posthoc = pairwise_gameshowell(data=df, dv=feature, between='labels')

    posthoc.set_index(['A', 'B'], inplace=True)  # 'A', 'B'는 기본적으로 비교 그룹의 라벨
    p_posthoc = [posthoc.loc[(0, 1), 'pval'],
                 posthoc.loc[(1, 2), 'pval'],
                 posthoc.loc[(0, 2), 'pval']
                 ]
    p_posthoc = [round(p, 4) for p in p_posthoc]
    return p_posthoc
