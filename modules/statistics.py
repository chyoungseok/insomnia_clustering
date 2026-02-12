import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
import scikit_posthocs as sp
from psmpy import PsmPy
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu, ttest_ind
from pingouin import welch_anova
import traceback

from modules import load, post_hoc_methods

def psm_matching(features: list, df_demo_HI: pd.DataFrame, merge_ins=True, multi_N=2, plot_result=False, seed=42):
    '''
    https://ysyblog.tistory.com/315
    https://blog.naver.com/l_e_e_sr/223103611816
    https://github.com/adriennekline/psmpy
    '''

    df_demo_HI = df_demo_HI.copy()

    if "PSG study Number#" in df_demo_HI.columns.to_list():
        pass
    else:
        df_demo_HI.reset_index(inplace=True)

    if merge_ins:
        df_demo_HI.loc[df_demo_HI.labels == 0, 'new_labels'] = 0
        df_demo_HI.loc[df_demo_HI.labels == 1, 'new_labels'] = 0
        df_demo_HI.loc[df_demo_HI.labels == 2, 'new_labels'] = 1

    df_demo_HI_selected = df_demo_HI.loc[:, ["PSG study Number#", "new_labels"] + features]
    df_demo_HI_selected.age = df_demo_HI_selected.age.astype(float)

    if multi_N==1:
        psm1 = PsmPy(df_demo_HI_selected, treatment='new_labels', indx='PSG study Number#')
        psm1.logistic_ps(balance = True)
        psm1.knn_matched(matcher='propensity_logit', replacement=False, caliper=None) # 기존 PSM
        matched_ids = psm1.matched_ids['matched_ID'].values
    else:
        psm1 = PsmPy(df_demo_HI_selected, treatment='new_labels', indx='PSG study Number#', seed=seed)
        psm1.logistic_ps(balance = True)
        psm1.knn_matched_12n(matcher='propensity_logit', how_many=multi_N) # 변경된 PSM --> how_many를 이용해서 선택하는 sample 수 커스텀 가능
        matched_ids = psm1.df_matched.loc[psm1.df_matched.new_labels==1, "PSG study Number#"].values
    # print(matched_ids)

    # Plot PSM results
    # psm1.plot_match(Tilot(title='Standardized Mean differences accross covariates before and after matching', before_color='#FCB754', after_color='#3EC8FB', save=False)    
    if plot_result:
        psm1.effect_size_plot(title='Standardized Mean differences accross covariates before and after matching', before_color='#FCB754', after_color='#3EC8FB', save=False)

    return matched_ids


def get_df_demo_HI_psm(_clustering):
    # healthy와 insomnia를 모두 포함하는 df_demo 로드 (ISI score 기반)
    _load_HI = load._load(path_scalogram=None,
                        reset=False,
                        inclusion_option='healthy_insomnia',
                        verbose=True)
    df_demo_HI = _load_HI.df_demo

    # Add cluster labels of only_insomnia subjects using _clustering.df_demo
    df_demo_HI['labels'] = 2 # set the labels of healthy as '2'
    df_demo_HI.loc[_clustering.df_demo.index, 'labels'] = _clustering.df_demo.labels # put labels from df_demo_I

    list_ids_ins = _clustering.df_demo.index.to_list()
    list_ids_healthy_psm = list(psm_matching(features=["age", "sex"], df_demo_HI=df_demo_HI.copy(), multi_N=1))

    df_demo_HI_psm = df_demo_HI.loc[list_ids_ins + list_ids_healthy_psm, :].copy()

    return df_demo_HI_psm

class ancova():
    """
    input으로 전달 받은 하나의 특정 feature에 대한 ANCOVA class
    """
    def __init__(self, df, feature, covariates=['age', 'sex', 'AHI'], force_lm_anova=False, force_kruskal=False, force_normality=False, posthoc_method='scheffe'):
        new_covariates = []
        for covariate in covariates:
            if not feature == covariate:
                new_covariates.append(covariate)

        self.mean, self.std = cal_mean_std(df, feature)
        self.mean_std_str = gen_mean_std_str(self.mean, self.std)

        list_data = []
        for i in sorted(df.labels.unique()):
            list_data.append(df.loc[df.labels==i, feature])

        self.df = df
        self.feature = feature
        self.covariates = new_covariates
        self.list_data = list_data
        self.normality = is_normality(list_data, force_normality)
        self.homoscedasticity = is_homoscedasticity(list_data)
        self.force_lm_anova = force_lm_anova
        self.force_kruskal = force_kruskal
        self.posthoc_method = posthoc_method

        if self.force_lm_anova and self.force_kruskal:
            raise ValueError("Select either lm_ancova or kruskal, not both.")

    def anova(self):
        df = self.df.copy()

        if self.force_lm_anova:
            self.normality = True
            self.homoscedasticity = True
        elif self.force_kruskal:
            self.normality = False
            self.homoscedasticity = False

        # self.normality = False # manual modification (force kruskal)

        if self.normality:
            if self.homoscedasticity:
                # 정규성 (O), 등분산성 (O) --> one way anova
                F_ancova, p_ancova = lm_ancova(feature=self.feature, covariates=self.covariates, df=self.df)
                self.type_anova = 'lm_ancova'
                self.F_ancova = F_ancova
                self.p_ancova = p_ancova
            elif not(self.homoscedasticity):
                # 정규성 (O), 등분산성 (x) --> welch anova
                anova = welch_anova(dv=self.feature, between='labels', data=self.df)
                self.type_anova = 'welch'
                self.F_ancova = anova.loc[0, 'F']
                self.p_ancova = anova.loc[0, 'p-unc']    
        else:
            # 정규성 (x), 등분산성 (x) --> Kruskal test
            kruskal_test = stats.kruskal(df[df["labels"] == 0][self.feature],
                                         df[df["labels"] == 1][self.feature],
                                         df[df["labels"] == 2][self.feature])
            num_K = len(self.list_data)
            kruskal_test = [0, 1]
            com = 'kruskal_test = stats.kruskal(' + num_K*'self.list_data[%s],' % tuple(range(num_K)) + ')'
                
            lcls = locals() # exec를 통해 할당한 변수를 가져오기 위한 처리
            exec(com, globals(), lcls)
            kruskal_test = lcls['kruskal_test']

            self.type_anova = 'kruskal'
            self.F_ancova = kruskal_test[0]
            self.p_ancova = kruskal_test[1]
            

    def post_hoc(self):
        df = self.df.copy()
        feature = self.feature
        comparisons = [(0, 1), (1, 2), (0, 2)]
        p_posthoc = []
        type_posthoc = None

        if self.normality:
            if self.homoscedasticity:
                if check_uniform_length(self.list_data):
                    # 정규성 (O), 등분산성 (O), equal sample size (O) --> Tukey's HSD
                    p_posthoc = post_hoc_methods.posthoc_tukey_hsd(df=self.df, feature=self.feature)
                    type_posthoc = 'tukey'
                else:
                    # 정규성 (O), 등분산성 (O), equal sample size (X) --> Fisher, Scheffe, Bonferroni
                    if 'fisher' in self.posthoc_method:            
                        p_posthoc = post_hoc_methods.posthoc_fisher_lsd(df=self.df, feature=self.feature)
                        type_posthoc = 'fisher'
                    elif 'scheffe' in self.posthoc_method:
                        p_posthoc = post_hoc_methods.posthoc_scheffe(df=self.df, feature=self.feature)
                        type_posthoc = 'scheffe'
                    elif 'bonf' in self.posthoc_method:
                        p_posthoc = post_hoc_methods.posthoc_bonferroni(df=self.df, feature=self.feature)
                        type_posthoc = 'bonferroni'
            else:
                # 정규성 (O), 등분산성 (X) --> Games-Howell post hoc
                p_posthoc = post_hoc_methods.posthoc_games_howell(df=self.df, feature=self.feature)
                type_posthoc = 'games_howell'
            
        else:
            # 정규성 (X) --> Kruskal-Wallis Test에 따른 post hoc 진행

            # Dunn's Test
            dunn_test = sp.posthoc_dunn(df, feature, 'labels', p_adjust=None)
            p_posthoc_dunn = [dunn_test.loc[0, 1], dunn_test.loc[1, 2], dunn_test.loc[0, 2]]
            type_posthoc = 'dunn'
        
            # Conover's Test
            conover_test = sp.posthoc_conover(a = df, val_col = feature, group_col = 'labels', p_adjust=None)
            conover_test = [conover_test.loc[0, 1], conover_test.loc[1, 2], conover_test.loc[0, 2]]

            # Nemenyi 검정
            nemenyi_test = sp.posthoc_nemenyi(df, feature, 'labels')
            nemenyi_test = [nemenyi_test.loc[0, 1], nemenyi_test.loc[1, 2], nemenyi_test.loc[0, 2]]

            p_posthoc = p_posthoc_dunn

        self.p_posthoc = p_posthoc
        self.post_hoc_str = gen_post_hoc_str(self)
        self.type_posthoc = type_posthoc

class categorical_ancova():
    def __init__(self, df, feature, covariates=['age', 'sex', 'AHI']): 
        F_ancova, p_ancova = chi_square(feature, df)

        new_covariates = []
        new_covariates = [covariate for covariate in covariates if not covariate == 'sex']
        # if feature == 'sex':
            
        #     covariates = ['age', 'AHI']

        self.mean, self.std = cal_mean_std(df, feature)
        self.mean_std_str = gen_mean_std_str(self.mean, self.std)

        list_data = []
        for i in sorted(df.labels.unique()):
            list_data.append(df.loc[df.labels==i, feature])

        self.df = df
        self.feature = feature
        self.covariates = new_covariates
        self.type_anova = 'chi_square'
        self.F_ancova = F_ancova
        self.p_ancova = p_ancova
        self.list_data = list_data

    def post_hoc(self):
        df = self.df.copy()
        
        # if self.p_ancova >= 0.05:
        #     p_posthoc = [1, 1, 1]
        #     F_posthoc = [0, 0, 0]

        comparisons = [(0, 1), (1, 2), (0, 2)]

        p_posthoc = []
        F_posthoc = []
        for group1, group2 in comparisons:
            # 두 그룹 데이터 필터링
            df_filtered = df[df['labels'].isin([group1, group2])]

            # 원-핫 인코딩으로 그룹 변수 처리
            df_filtered = pd.get_dummies(df_filtered, columns=['labels'], drop_first=True)

            # 로지스틱 회귀분석 모델 구성 및 적합
            model = sm.Logit(df_filtered[self.feature], df_filtered[['labels_%d' % group2] + self.covariates])  # Group_B: A 대비 B 그룹, Age: 나이 공변량
            result = model.fit(disp=0)

            p_posthoc.append(result.pvalues['labels_%d' % group2])
            F_posthoc.append(result.tvalues['labels_%d' % group2])
                
        self.p_posthoc = p_posthoc
        self.F_posthoc = F_posthoc
        self.type_posthoc = 'logistic'
        self.post_hoc_str = gen_post_hoc_str(self)

def gen_formula(feature, covariates):
    formula = "%s ~ C(labels)" % feature

    if len(covariates) == 0:
        pass
    else:
        for covariate in covariates:
            if covariate == feature:
                pass

            if covariate == 'sex':
                formula += " + C(%s)" % covariate
            else:
                formula += " + %s" % covariate
        
    return formula

def lm_ancova(feature: str, covariates: list, df: pd.DataFrame):
    # create formula
    formula = gen_formula(feature, covariates)
    # print(formula) # ISI ~ C(labels) + age + C(sex)

    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    F = anova_table.loc["C(labels)", "F"]
    p = anova_table.loc["C(labels)", "PR(>F)"]

    return F, p

def chi_square(feature: str, df: pd.DataFrame):
    cross_tab = pd.crosstab(df['labels'], df[feature]) # 교차 표 생성
    chi2, p_value, dof, expected = chi2_contingency(cross_tab) # 카이 제곱 검정 수행

    return chi2, p_value

def cal_mean_std(df: pd.DataFrame, feature: str):
    unique_labels = np.sort(df.labels.unique())

    mean = []
    std = []
    for label in unique_labels:
        df_temp =  df.loc[df.labels == label, feature]

        if feature != "sex":
            mean.append(df_temp.mean())
            std.append(df_temp.std())
        else:
            mean.append(1-df_temp.mean())
            std.append(0)

    return mean, std

def gen_mean_std_str(mean, std):
    list_str = []
    for temp_mean, temp_std in zip(mean, std):
        list_str.append("%.2f (%.2f)" % (temp_mean, temp_std))
    
    return list_str

def get_inequality(pval):
    if pval>=0.05:
        return (' , ')
    elif (pval>0.01) & (pval<=0.05):
        return (' < ')
    elif (pval>0.001) & (pval<0.01):
        return (' << ')
    else:
        return (' <<< ')
    
def get_significance_asterisk(pval, ns_hyphen=False):
    if pval>=0.05:
        if ns_hyphen:
            return (' ')
        else:
            return ('ns')
    elif (pval>=0.01) & (pval<0.05):
        return ('*')
    elif (pval>=0.001) & (pval<0.01):
        return ('**')
    else:
        return ('***')
    
def get_significance_cross(pval):
    if pval>=0.05:
        return ('ns')
    elif (pval>=0.01) & (pval<0.05):
        return ('✝')
    elif (pval>=0.001) & (pval<0.01):
        return ('✝✝')
    else:
        return ('✝✝✝')
    
def inverse_inequality(ineqaulity: str):
    if ',' in ineqaulity: # (수정) '.' --> ','
        return ineqaulity
    elif ' < ' in ineqaulity:
        return ' > '
    elif ' << ' in ineqaulity:
        return ' >> '
    else:
        return ' >>> '

def gen_post_hoc_str(_ancova):
    df = _ancova.df.copy()
    len_cluster = len(df.labels.unique())

    alphabets = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
    alphabets_temp = alphabets[:len_cluster]
    sorted_arg = np.argsort(_ancova.mean) # small -> big

    if not hasattr(_ancova, 'p_posthoc'):
        if hasattr(_ancova, 'p_ancova'): 
            # two group 비교 class도 본 함수를 사용 가능하도록 하는 과정
            _ancova.p_posthoc = [_ancova.p_ancova]

    list_inequality = []
    for i in range(len_cluster):
        groupA = sorted_arg[i]

        if not i == len_cluster-1:
            groupB = sorted_arg[i+1]
        else:
            groupB = sorted_arg[0]

        if (groupA + groupB) == 1: # 0, 1
            pval = _ancova.p_posthoc[0] 
        elif (groupA + groupB) == 3: # 1, 2
            pval = _ancova.p_posthoc[1]
        else: # 0, 2
            pval = _ancova.p_posthoc[2]
        
        list_inequality.append(get_inequality(pval))

    if len_cluster > 2:
        post_hoc_str = "%s%s%s%s%s (%s%s)" % (alphabets_temp[sorted_arg][0],
                                            list_inequality[0],
                                            alphabets_temp[sorted_arg][1],
                                            list_inequality[1],
                                            alphabets_temp[sorted_arg][2],
                                            inverse_inequality(list_inequality[2]),
                                            alphabets_temp[sorted_arg][0])
    else:
        post_hoc_str = "%s%s%s" % (alphabets_temp[sorted_arg][0],
                                   list_inequality[0],
                                   alphabets_temp[sorted_arg][1])
    return post_hoc_str

def gen_df_stat(df, df_type="demo", covariates=[], force_lm_anova=False, force_kruskal=False, force_normality=False, posthoc_method='scheffe', p_correct_method='FDR'):
    """
    df_type ('demo', 'thick', 'bai', 'snsb')
    """
    categorical_variables = ['sex', 'is_paradoxical_1', 'is_paradoxical_2', 'is_paradoxical_3', 'is_paradoxical_4']
    stat_demo = pd.DataFrame(columns=['A', 'B', 'C', 'Statistic', 'F', 'p', ' ', 'Post hoc', 'type_posthoc'])

    if 'demo' in df_type.lower():
        col_start = 2; col_end = -1
        force_normality = True
        # force_lm_anova = True
    elif 'thick' in df_type.lower():
        col_start = 1; col_end = None
    elif 'bai' in df_type.lower():
        col_start = 1; col_end = None
    elif 'snsb' in df_type.lower():
        col_start = 1; col_end = None
    elif 'eeg' in df_type.lower():
        col_start = 0; col_end = None
    
    dic_posthoc_p = {}
    dic_posthoc_str = {}
    for feature in df.columns.to_list()[col_start:col_end]:

        df_feature = load.rm_abnormal(df=df, list_feature=[feature], verbose=False) # remove abnormal values ex) NaN, '.'
        df_feature[feature] = df_feature[feature].astype(float)

        list_data = []
        for i in sorted(df_feature.labels.unique()):
            list_data.append(df_feature.loc[df_feature.labels==i, feature])
        
        if not feature in categorical_variables:
            _ancova = ancova(df=df_feature, feature=feature, covariates=covariates, force_lm_anova=force_lm_anova, force_kruskal=force_kruskal, force_normality=force_normality, posthoc_method=posthoc_method)
            _ancova.anova()
        else:
            _ancova = categorical_ancova(df=df_feature, feature=feature, covariates=covariates)
        _ancova.post_hoc()

        stat_demo.loc[feature, 'A'] = _ancova.mean_std_str[0]
        stat_demo.loc[feature, 'B'] = _ancova.mean_std_str[1]
        stat_demo.loc[feature, 'C'] = _ancova.mean_std_str[2]

        stat_demo.loc[feature, 'Statistic'] = _ancova.type_anova[0].upper()
        stat_demo.loc[feature, 'F'] = _ancova.F_ancova
        stat_demo.loc[feature, 'p'] = _ancova.p_ancova
        stat_demo.loc[feature, ' '] = get_significance_cross(_ancova.p_ancova) # significance 추가

        stat_demo.loc[feature, 'Post hoc'] = replace_posthoc_str(_ancova.post_hoc_str)
        stat_demo.loc[feature, 'type_posthoc'] = _ancova.type_posthoc
        dic_posthoc_p[feature] = _ancova.p_posthoc
        dic_posthoc_str[feature] = _ancova.post_hoc_str

    stat_demo['F'] = stat_demo['F'].apply(lambda x: f"{x:.2f}")
    stat_demo['p'] = stat_demo['p'].apply(lambda x: f"{x:.3f}") # 'p' column에 대하여 특정 소수점 아래자리까지 보여지게 formatting

    # ====== Correction for multiple comparison ======
    # p-value 리스트
    p_values = stat_demo['p'].astype(float)

    
    # 본페로니 교정
    if 'bonf' in p_correct_method.lower():
        bonferroni_corrected = multipletests(p_values, alpha=0.05, method='bonferroni')
        stat_demo['p_corrected'] = bonferroni_corrected[1]
        stat_demo['  '] = stat_demo['p_corrected'].apply(get_significance_cross)

    # con1 = table_2.p.astype(float) < 0.05
    # con2 = table_2.bonferroni_p >= 0.05
    # table_2.loc[con1 & con2,['HS', 'Para-I', 'Psy-I', 'Statistic', 'p', ' ', 'bonferroni_p', '  '] ]

    # 홀름-본페로니 교정
    elif 'holm' in p_correct_method.lower():
        holm_corrected = multipletests(p_values, alpha=0.05, method='holm')
        stat_demo['p_corrected'] = holm_corrected[1]
        stat_demo['  '] = stat_demo['p_corrected'].apply(get_significance_cross)

    # con1 = table_2.p.astype(float) < 0.05
    # con2 = table_2.holm_p >= 0.05
    # table_2.loc[con1 & con2,['HS', 'Para-I', 'Psy-I', 'Statistic', 'p', ' ', 'holm_p', '  '] ]

    # 벤저미니-호크버그 교정 (FDR correction)
    # --> ESS의 유의성만 손실
    # --> sSOL/SOL, (sSOL-SOL)/SOL 의 유의성만 손실
    elif 'fdr' in p_correct_method.lower():
        bh_corrected = multipletests(p_values, alpha=0.05, method='fdr_bh')
        stat_demo['p_corrected'] = bh_corrected[1]
        stat_demo['  '] = stat_demo['p_corrected'].apply(get_significance_cross)

    # con1 = table_2.p.astype(float) < 0.05
    # con2 = table_2.p_corrected >= 0.05
    # table_2.loc[con1 & con2,['HS', 'Para-I', 'Psy-I', 'Statistic', 'p', ' ', 'p_corrected', '  '] ]

    stat_demo['p_corrected'] = stat_demo['p_corrected'].apply(lambda x: f"{x:.3f}")
    
    # p value < 0.001 인 경우, '< 0.001'로 표기
    con_under_three_decimal = stat_demo.p_corrected.astype(float) < 0.001
    stat_demo.loc[con_under_three_decimal, 'p_corrected'] = '< 0.001'

    # n 수 row 추가
    stat_demo.loc['N', 'A'] = (df.labels==0).sum()
    stat_demo.loc['N', 'B'] = (df.labels==1).sum()
    stat_demo.loc['N', 'C'] = (df.labels==2).sum()
    stat_demo.loc['N', 'Statistic'] = ''
    stat_demo.loc['N', 'F'] = ''
    stat_demo.loc['N', 'p'] = ''
    stat_demo.loc['N', ' '] = ''
    stat_demo.loc['N', 'p_corrected'] = ''
    stat_demo.loc['N', '  '] = ''
    stat_demo.loc['N', 'Post hoc'] = ''
    stat_demo.loc['N', 'type_posthoc'] = ''
    stat_demo = pd.concat([stat_demo.loc[['N'], :], stat_demo.iloc[:-1, :]])
    stat_demo = stat_demo.loc[:, ['C', 'A', 'B', 'Statistic', 'p', 'p_corrected', '  ', 'Post hoc']]

    # Sex mean값 표기
    # - Male의 비율을 백분율로 표시 ex) 55
    if df_type != 'eeg':
        stat_demo = convert_sex_values_to_int_percentage(stat_demo, three_group=True)

    stat_demo.rename(columns={"A": "Subtype 1 (PDI)", "B": "Subtype 2 (PPI)", "C": "HS"}, inplace=True)
    

    return stat_demo, dic_posthoc_p, dic_posthoc_str

def show_number_each_cluster(df):
    print()
    print("Para: %d" % (df.labels == 0).sum())
    print("Non-para: %d" % (df.labels == 1).sum())
    print("GS (PI): %d" % (df.labels == 2).sum())
    
    # ✅ 어디서 호출됐는지 전체 호출 스택 출력
    # print("\n[DEBUG] Call stack:")
    # traceback.print_stack()
    # print("\n")

def is_normality(data: list, force_normality: bool):
    normality = True

    if not is_enough_sample(data):
        return False
    
    if force_normality:
        return True
    else:
        for temp_data in data:
            if len(temp_data) < 3:
                print("wow")
            normality = normality & (stats.shapiro(temp_data)[1]>0.05)
    
    return normality

def is_homoscedasticity(data: list):
    num_K = len(data)

    if not is_enough_sample(data):
        return False

    homoscedasticity = True
    # com = 'homoscedasticity = stats.levene(' + num_K*'data[%s],' % tuple(range(num_K)) + ')[1] > 0.05'
    com = 'homoscedasticity = stats.bartlett(' + num_K*'data[%s],' % tuple(range(num_K)) + ')[1] > 0.05'
        
    lcls = locals() # exec를 통해 할당한 변수를 가져오기 위한 처리
    exec(com, globals(), lcls)
    homoscedasticity = lcls['homoscedasticity']
    
    return homoscedasticity

def is_enough_sample(list_data: list) -> bool:
    for item in list_data:
        if len(item) < 3:
            return False
    return True

def rm_abnormal(df, feature):
    """
    - dataframe(df)의 feature column에서 abnormal value 제거
    - abnormal value의 종류는 'na', '-' 등 다양함
    """
    con_isna = df[feature].isna() 
    con_dot = df[feature]=='.'
    con_hyphen = df[feature]=='-'
    con_hyphen2 = df[feature]==' -  '
    con_hyphen3 = df[feature]==' -   '
    con_hyphen4 = df[feature]==' -     '
    con_hyphen5 = df[feature]==' - '
    con_hyphen6 = df[feature]=='-  '
    con_hyphen7 = df[feature]==' -      '
    con_hyphen8 = df[feature]=='  -  '

    con = con_isna | con_dot | con_hyphen | con_hyphen2 | con_hyphen3 | con_hyphen4 | con_hyphen5 | con_hyphen6 | con_hyphen7 | con_hyphen8

    df.loc[~con, feature] = df.loc[~con, feature].astype(float)
    
    df = df[~con]
    return df

class two_group_stat():
    """
    input으로 전달 받은 하나의 특정 feature에 대한 ANCOVA class
    """
    def __init__(self, df, feature, covariates=['age', 'sex', 'AHI'], force_mann=False, force_ttest=False, force_normality=False):
        # feature와 covariate가 중복되지 않도록 함
        new_covariates = []
        for covariate in covariates:
            if not feature == covariate:
                new_covariates.append(covariate)

        self.mean, self.std = cal_mean_std(df, feature)
        self.mean_std_str = gen_mean_std_str(self.mean, self.std)

        list_data = []
        for i in sorted(df.labels.unique()):
            list_data.append(df.loc[df.labels==i, feature])

        self.df = df
        self.feature = feature
        self.covariates = new_covariates
        self.list_data = list_data
        self.normality = is_normality(list_data, force_normality)
        self.force_mann = force_mann
        self.force_ttest = force_ttest
        self.force_normality = force_normality

    def run(self):
        # force_mann과 force_ttest가 서로 다른지 확인
        if all([self.force_mann, self.force_ttest]):
            raise ValueError("force_mann와 force_ttest는 서로 다른 값을 가져야 합니다.")
        
        if self.force_mann:
            self.normality = False
        elif self.force_ttest:
            self.normality = True

        if self.normality:
        # (예정) 정규성 만족하는 경우, t-test 진행

            t_statistic, p_value = ttest_ind(self.list_data[0], self.list_data[1])

            self.type_anova = 'T-test'
            self.F_ancova = t_statistic # 이전에 작성된 함수와 form을 맞추기 위하여 F_ancova라는 변수 이름 사용
            self.p_ancova = p_value # 이전에 작성된 함수와 form을 맞추기 위하여 p_ancova라는 변수 이름 사용
        else:
            # 정규성을 만족하지 않는 경우, Mann-Whitney U Test 진행

            u_statistic, p_value = mannwhitneyu(self.list_data[0], self.list_data[1])

            self.type_anova = 'Mann_Whitney'
            self.F_ancova = u_statistic # 이전에 작성된 함수와 form을 맞추기 위하여 F_ancova라는 변수 이름 사용
            self.p_ancova = p_value # 이전에 작성된 함수와 form을 맞추기 위하여 p_ancova라는 변수 이름 사용

def gen_df_stat_two_group(df, df_type="demo", force_mann=False, force_ttest=False, force_normality=False, show_statistic=False):
    """
    two group 통계 비교에 대한 df_stat을 생성하는 함수
    df_type ('demo', 'thick', 'bai', 'snsb')
    """
    categorical_variables = ['sex', 'is_paradoxical_1', 'is_paradoxical_2', 'is_paradoxical_3', 'is_paradoxical_4']
    stat_demo = pd.DataFrame(columns=['A', 'B', 'stat_method', 'statistic', 'p', ' ', 'p_corrected', '  '])

    if 'demo' in df_type.lower():
        col_start = 2; col_end = -1
        # force_lm_anova = True
    elif 'thick' in df_type.lower():
        col_start = 1; col_end = None
    elif 'bai' in df_type.lower():
        col_start = 1; col_end = None
    elif 'snsb' in df_type.lower():
        col_start = 1; col_end = None
    elif 'eeg' in df_type.lower():
        col_start = 0; col_end = None

    for feature in df.columns.to_list()[col_start:col_end]:

        df_feature = load.rm_abnormal(df=df, list_feature=[feature], verbose=False) # remove abnormal values ex) NaN, '.'
        df_feature[feature] = df_feature[feature].astype(float)

        if not feature in categorical_variables:
            _two_sample_test = two_group_stat(df=df_feature, feature=feature, force_mann=force_mann, force_ttest=force_ttest, force_normality=force_normality)
            _two_sample_test.run()
        else:
            _two_sample_test = categorical_ancova(df=df_feature, feature=feature)

        stat_demo.loc[feature, 'A'] = _two_sample_test.mean_std_str[0]
        stat_demo.loc[feature, 'B'] = _two_sample_test.mean_std_str[1]

        stat_demo.loc[feature, 'stat_method'] = _two_sample_test.type_anova[0].upper()
        stat_demo.loc[feature, 'statistic'] = _two_sample_test.F_ancova
        stat_demo.loc[feature, 'p'] = _two_sample_test.p_ancova
        stat_demo.loc[feature, ' '] = get_significance_asterisk(_two_sample_test.p_ancova) # significance 추가

    stat_demo['statistic'] = stat_demo['statistic'].apply(lambda x: f"{x:.2f}")
    stat_demo['p'] = stat_demo['p'].apply(lambda x: f"{x:.3f}") # 'p' column에 대하여 특정 소수점 아래자리까지 보여지게 formatting

    # ====== Correction for multiple comparison ======
    # p-value 리스트
    p_values = stat_demo['p'].astype(float)

    # 본페로니 교정
    bonferroni_corrected = multipletests(p_values, alpha=0.05, method='bonferroni')
    stat_demo['p_corrected'] = bonferroni_corrected[1]
    stat_demo['  '] = bonferroni_corrected[0]
    stat_demo['  '] = stat_demo['p_corrected'].apply(get_significance_asterisk)
    stat_demo['p_corrected'] = stat_demo['p_corrected'].apply(lambda x: f"{x:.3f}")

    # # 홀름-본페로니 교정
    # holm_corrected = multipletests(p_values, alpha=0.05, method='holm')
    # table_1['holm_p'] = holm_corrected[1]
    # table_1['holm_significant'] = holm_corrected[0]

    # 벤저미니-호크버그 교정 (FDR correction)
    # --> ESS의 유의성만 손실
    # bh_corrected = multipletests(p_values, alpha=0.05, method='fdr_bh')
    # stat_demo['p_corrected'] = bh_corrected[1]
    # stat_demo['  '] = bh_corrected[0]
    # stat_demo['  '] = stat_demo['p_corrected'].apply(get_significance_asterisk)
    # stat_demo['p_corrected'] = stat_demo['p_corrected'].apply(lambda x: f"{x:.3f}")

    # p value < 0.001 인 경우, '< 0.001'로 표기
    con_under_three_decimal = stat_demo.p_corrected.astype(float) < 0.001
    stat_demo.loc[con_under_three_decimal, 'p_corrected'] = '< 0.001'

    # n 수 row 추가
    stat_demo.loc['N', 'A'] = (df.labels==0).sum()
    stat_demo.loc['N', 'B'] = (df.labels==1).sum()
    stat_demo.loc['N', 'stat_method'] = ''
    stat_demo.loc['N', 'statistic'] = ''
    stat_demo.loc['N', 'p'] = ''
    stat_demo.loc['N', ' '] = ''
    stat_demo.loc['N', 'p_corrected'] = ''
    stat_demo.loc['N', '  '] = ''

    stat_demo = pd.concat([stat_demo.loc[['N'], :], stat_demo.iloc[:-1, :]])

    stat_demo = convert_sex_values_to_int_percentage(stat_demo, three_group=False)

    if show_statistic:
        stat_demo = stat_demo.loc[:, ['A', 'B', 'stat_method', 'statistic', 'p', 'p_corrected', '  ']] 
    else:
        stat_demo = stat_demo.loc[:, ['A', 'B', 'stat_method', 'p_corrected', '  ']]

    return stat_demo


def check_uniform_length(lists: list):
    # 내부 리스트들의 길이를 취득하여 set으로 변환
    # 모든 길이가 같다면 set의 길이는 1일 것
    return len(set(len(sublist) for sublist in lists)) == 1

def replace_posthoc_str(posthoc_str: str):
    posthoc_str = posthoc_str.replace('a', 'Para-I')
    posthoc_str = posthoc_str.replace('b', 'Psy-I')
    posthoc_str = posthoc_str.replace('c', 'HS')

    posthoc_str = posthoc_str.replace("HS", 'a')
    posthoc_str = posthoc_str.replace('Para-I', 'b')
    posthoc_str = posthoc_str.replace('Psy-I', 'c')
    return posthoc_str

def convert_sex_values_to_int_percentage(df: pd.DataFrame, three_group=True):
    df = df.copy()
    
    feature = None
    if 'sex' in df.index.to_list():
        feature = 'sex'
    elif 'Sex' in df.index.to_list():
        feature = 'Sex'
    elif 'Male, %' in df.index.to_list():
        feature = 'Male, %'

    if three_group:
        list_groups = ["C", "A", "B"]
    else:
        list_groups = ["B", "A"]
    
    df_sex = df.loc[[feature], list_groups]

    for col in df_sex.columns:
        df_sex[col] = df_sex[col].apply(lambda x: int((float(x.split()[0])) * 100) if isinstance(x, str) else x)

    df.loc[[feature], list_groups] = df_sex.values
    return df

def merge_N_Male(group_num, df_stat: pd.DataFrame, list_groups=None):
    """
    기존 df_stat에서는 N수와 Male % index가 따로 분리되어 있었음. 
    이를 합치는 함수
    """
    
    if isinstance(list_groups, type(None)):
        if group_num == 2:
            list_groups = ["HS", "Insomnia"]
        elif group_num == 3:
            list_groups = ["HS", "Subtype 1 (PDI)", "Subtype 2 (PPI)"]
    
    df_stat_N_Male = df_stat.loc[['N', 'Male, %'], :]  # [N, Male] index만 미리 따로 추출
    df_stat_N_Male.loc["N (male, %)", :] = '' # "N (male, %)" 라는 새로운 index를 생성하고, 모든 column의 값을 빈 값으로 초기화
    for group in list_groups:
        df_stat_N_Male.loc["N (male, %)", group] = "%d (%d)" %  (df_stat_N_Male.loc["N", group], df_stat_N_Male.loc["Male, %", group])

    df_stat_merged = pd.concat([df_stat_N_Male.loc[["N (male, %)"], :], df_stat.drop(index=['N', 'Male, %'])])

    return df_stat_merged

