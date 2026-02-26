import pickle
import sys, os

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from modules.utils import path_csv, path_main
from modules import statistics, gen_df_MR, gen_df_SNSB

set_features_1 = ['sex', 'age', 'BMI', 'AHI']
set_features_2 = ['BDI', 'ESS', 'PSQI', 'SSS', 'ISI']
set_features_3 = ['TST', 'SOL', 'REM_latency', 'N2_latency', 'N1', 'N2', 'N3', 'REM', 'SE', 'WASO_rel', 'AI', 'REM_AI_h', 'NREM_h']
set_features_4 = ['sTST', 'ratio_TST', 'diff_TST', 'rd_TST', 'sSOL', 'ratio_SOL', 'diff_SOL', 'rd_SOL']


# # 클래스 인스턴스를 pickle 파일로 저장
# with open('../_clustering.pkl', 'wb') as f:
#     pickle.dump(_clustering, f)

# # pickle 파일에서 클래스 인스턴스 불러오기
# with open(os.path.join(path_main, '_clustering_zscore_by_HS.pkl'), 'rb') as f:
#     _clustering = pickle.load(f)
    
# with open(os.path.join(path_main, '_clustering.pkl'), 'rb') as f:
#     _clustering = pickle.load(f)

class demo_summary():
    def __init__(self, df_demo_HI_psm=None) -> None:
        if df_demo_HI_psm is None:
            df_demo_HI_psm = pd.read_csv(os.path.join(path_csv, 'df_demo_HI_psm_updated_labels.csv'), encoding='euc-kr', index_col=0)
        df_stat_demo_psm_age_sex, dic_posthoc_p_psm_age_sex, dic_posthoc_str_psm_age_sex = statistics.gen_df_stat(df_demo_HI_psm, 'demo', ['age', 'sex'], force_lm_anova=False, posthoc_method="scheffe")
        statistics.show_number_each_cluster(df_demo_HI_psm)

        self.df_demo_HI_psm = df_demo_HI_psm
        self.df_stat_demo_psm_age_sex = df_stat_demo_psm_age_sex
        self.dic_posthoc_p_psm_age_sex = dic_posthoc_p_psm_age_sex
        self.dic_posthoc_str_psm_age_sex = dic_posthoc_str_psm_age_sex

''' 2. MRI data '''
class mr_summary():
    def __init__(self, _clustering, psm_N=7, show_group_N=False, psm_seed=42, p_correct_method='fdr', posthoc_method='scheffe') -> None:
        # ======== Load MRI related data

        _, df_MR_bai_ins, df_MR = gen_df_MR.get_df_MR_thickness_and_bai(_clustering)
        df_mr_bai_healthy = gen_df_MR.get_df_mr_bai_normal()
        df_mr_bai_total = gen_df_MR.merge_df_mr_bai(df_mr_bai_ins=df_MR_bai_ins, df_mr_bai_norm=df_mr_bai_healthy)

        if (df_mr_bai_total.labels == 0).sum() > (df_mr_bai_total.labels == 1).sum():
            # ins A (n=14), ins B (n=5)를 ins A (n=5), ins B (n=14)로 변경
            df_mr_bai_total.labels.replace({0:1, 1:0}, inplace=True)

        # ======== Propensity Score Matching
        list_ids_ins = df_MR_bai_ins.index.to_list()
        list_ids_healthy_psm = list(statistics.psm_matching(features=["age", "sex"], df_demo_HI=df_mr_bai_total.copy(), multi_N=psm_N, seed=psm_seed))
        df_mr_bai_total_psm = df_mr_bai_total.loc[list_ids_ins + list_ids_healthy_psm, :]

        if show_group_N:
            statistics.show_number_each_cluster(df_mr_bai_total_psm)

        # ======== Statistical Analysis
        df_stat_bai_age_sex, dic_posthoc_p_age_sex, dic_posthoc_str_age_sex = statistics.gen_df_stat(df_mr_bai_total, df_type='bai', covariates=['age', 'sex'], force_lm_anova=False, p_correct_method=p_correct_method, posthoc_method=posthoc_method) 
        df_stat_bai_psm_no_cov, dic_posthoc_p_psm_no_cov, dic_posthoc_str_psm_no_cov = statistics.gen_df_stat(df_mr_bai_total_psm, df_type='bai', covariates=[], force_lm_anova=False) # no covariate
        df_stat_bai_psm_age_sex, dic_posthoc_p_psm_age_sex, dic_posthoc_str_psm_age_sex = statistics.gen_df_stat(df_mr_bai_total_psm, df_type='bai', covariates=['age', 'sex'], force_lm_anova=False, posthoc_method=posthoc_method, p_correct_method=p_correct_method)

        self.df_mr_bai_total = df_mr_bai_total
        self.list_ids_ins = list_ids_ins
        self.list_ids_healthy_psm = list_ids_healthy_psm
        self.df_mr_bai_total_psm = df_mr_bai_total_psm
        
        self.df_stat_bai_age_sex = df_stat_bai_age_sex
        self.df_stat_bai_psm_no_cov = df_stat_bai_psm_no_cov
        self.df_stat_bai_psm_age_sex = df_stat_bai_psm_age_sex
        
        self.dic_posthoc_p_age_sex = dic_posthoc_p_age_sex
        self.dic_posthoc_p_psm_age_sex = dic_posthoc_p_psm_age_sex
        self.dic_posthoc_str_psm_age_sex = dic_posthoc_str_psm_age_sex

''' 3. SNSB data '''
class snsb_summary():
    def __init__(self, _clustering, psm_N=7, show_group_N=False, psm_seed=42, p_correct_method='fdr', posthoc_method="scheffe") -> None:
        # === Load Cognitive score data ===
        df_SNSB = gen_df_SNSB.get_df_SNSB(_clustering)

        if (df_SNSB.labels == 0).sum() > (df_SNSB.labels == 1).sum():
            # ins A (n=14), ins B (n=5)를 ins A (n=5), ins B (n=14)로 변경
            # 이 과정을 해주지 않으면, statistic의 logistic regression 부분에서 singular matrix 관련 오류 발생
            df_SNSB.labels.replace({0:1, 1:0}, inplace=True) 

        # ======== Propensity Score Matching
        df_SNSB.index.name = 'PSG study Number#'
        list_ids_ins = df_SNSB.loc[df_SNSB.labels != 2].index.to_list()
        list_ids_healthy_psm = list(statistics.psm_matching(features=["age", "sex"], df_demo_HI=df_SNSB.copy(), multi_N=psm_N, seed=psm_seed))

        df_SNSB_psm = df_SNSB.loc[list_ids_ins + list_ids_healthy_psm, :]
        if show_group_N:
            statistics.show_number_each_cluster(df_SNSB_psm)

        # ======== Statistical Analysis
        df_stat_SNSB_age_sex, dic_posthoc_p_age_sex , dic_posthoc_str_age_sex = statistics.gen_df_stat(df_SNSB, df_type='SNSB', covariates=['age', 'sex'], force_lm_anova=False, posthoc_method=posthoc_method, p_correct_method=p_correct_method)
        df_stat_SNSB_psm_age_sex, dic_posthoc_p_psm_age_sex , dic_posthoc_str_psm_age_sex = statistics.gen_df_stat(df_SNSB_psm, df_type='SNSB', covariates=['age', 'sex'], force_lm_anova=False, posthoc_method=posthoc_method, p_correct_method=p_correct_method)

        self.df_SNSB = df_SNSB
        self.list_ids_ins = list_ids_ins
        self.list_ids_healthy_psm = list_ids_healthy_psm
        self.df_SNSB_psm = df_SNSB_psm
        self.df_stat_SNSB_age_sex = df_stat_SNSB_age_sex
        self.df_stat_SNSB_psm_age_sex = df_stat_SNSB_psm_age_sex
        self.dic_posthoc_p_psm_age_sex = dic_posthoc_p_psm_age_sex
        self.dic_posthoc_p_age_sex = dic_posthoc_p_age_sex
        self.dic_posthoc_str_psm_age_sex = dic_posthoc_str_psm_age_sex

# _demo_summary = demo_summary(_clustering)
# _mr_summary = mr_summary(_clustering)
# _snsb_summary = snsb_summary(_clustering)
