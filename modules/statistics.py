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
        psm1.knn_matched(matcher='propensity_logit', replacement=False, caliper=None) # 疫꿸퀣??PSM
        matched_ids = psm1.matched_ids['matched_ID'].values
    else:
        psm1 = PsmPy(df_demo_HI_selected, treatment='new_labels', indx='PSG study Number#', seed=seed)
        psm1.logistic_ps(balance = True)
        psm1.knn_matched_12n(matcher='propensity_logit', how_many=multi_N) # 癰궰野껋럥留?PSM --> how_many????곸뒠??곴퐣 ?醫뤾문??롫뮉 sample ???뚣끉??? 揶쎛??
        matched_ids = psm1.df_matched.loc[psm1.df_matched.new_labels==1, "PSG study Number#"].values
    # print(matched_ids)

    # Plot PSM results
    # psm1.plot_match(Tilot(title='Standardized Mean differences accross covariates before and after matching', before_color='#FCB754', after_color='#3EC8FB', save=False)    
    if plot_result:
        psm1.effect_size_plot(title='Standardized Mean differences accross covariates before and after matching', before_color='#FCB754', after_color='#3EC8FB', save=False)

    return matched_ids


def get_df_demo_HI_psm():
    # healthy?? insomnia??筌뤴뫀紐???釉??롫뮉 df_demo 嚥≪뮆諭?(ISI score 疫꿸퀡而?
    _load_HI = load._load(path_scalogram=None,
                        reset=False,
                        inclusion_option='healthy_insomnia',
                        verbose=True)
    df_demo_HI = _load_HI.df_demo

    _load_INS = load._load(path_scalogram=None,
                           reset=False,
                           inclusion_option='only_insomnia',
                           channel_mode='single',
                           verbose=True)
    df_demo_INS = _load_INS.df_demo

    # Add cluster labels of only_insomnia subjects using _clustering.df_demo
    df_demo_HI['labels'] = 2 # set the labels of healthy as '2'
    df_demo_HI.loc[df_demo_INS.index, 'labels'] = 0 # put labels from df_demo_I

    list_ids_ins = df_demo_INS.index.to_list()
    list_ids_healthy_psm = list(psm_matching(features=["age", "sex"], df_demo_HI=df_demo_HI.copy(), multi_N=1))

    df_demo_HI_psm = df_demo_HI.loc[list_ids_ins + list_ids_healthy_psm, :].copy()

    return df_demo_HI_psm

class ancova():
    """
    input??곗쨮 ?袁⑤뼎 獄쏆룇? ??롪돌???諭??feature??????ANCOVA class
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
                # ?類?뇣??(O), ?源낇뀋?怨쀪쉐 (O) --> one way anova
                F_ancova, p_ancova, partial_eta2 = lm_ancova(feature=self.feature, covariates=self.covariates, df=self.df)
                self.type_anova = 'lm_ancova'
                self.F_ancova = F_ancova
                self.p_ancova = p_ancova
                self.effect_3group = partial_eta2
                self.effect_3group_type = 'partial_eta2'
            elif not(self.homoscedasticity):
                # ?類?뇣??(O), ?源낇뀋?怨쀪쉐 (x) --> welch anova
                anova = welch_anova(dv=self.feature, between='labels', data=self.df)
                self.type_anova = 'welch'
                self.F_ancova = anova.loc[0, 'F']
                self.p_ancova = anova.loc[0, 'p-unc']
                self.effect_3group = cal_welch_eta2(F=anova.loc[0, 'F'], ddof1=anova.loc[0, 'ddof1'], ddof2=anova.loc[0, 'ddof2'])
                self.effect_3group_type = 'eta2_welch'
        else:
            # ?類?뇣??(x), ?源낇뀋?怨쀪쉐 (x) --> Kruskal test
            kruskal_test = stats.kruskal(df[df["labels"] == 0][self.feature],
                                         df[df["labels"] == 1][self.feature],
                                         df[df["labels"] == 2][self.feature])
            num_K = len(self.list_data)
            kruskal_test = [0, 1]
            com = 'kruskal_test = stats.kruskal(' + num_K*'self.list_data[%s],' % tuple(range(num_K)) + ')'
                
            lcls = locals() # exec?????퉸 ?醫딅뼣??癰궰??? 揶쎛?紐꾩궎疫??袁る립 筌ｌ꼶??
            exec(com, globals(), lcls)
            kruskal_test = lcls['kruskal_test']

            self.type_anova = 'kruskal'
            self.F_ancova = kruskal_test[0]
            self.p_ancova = kruskal_test[1]
            self.effect_3group = cal_kruskal_epsilon2(H=kruskal_test[0], n_total=len(df), k_groups=len(self.list_data))
            self.effect_3group_type = 'epsilon2_kruskal'
            

    def post_hoc(self):
        df = self.df.copy()
        feature = self.feature
        comparisons = [(0, 1), (1, 2), (0, 2)]
        p_posthoc = []
        type_posthoc = None

        if self.normality:
            if self.homoscedasticity:
                if check_uniform_length(self.list_data):
                    # ?類?뇣??(O), ?源낇뀋?怨쀪쉐 (O), equal sample size (O) --> Tukey's HSD
                    p_posthoc = post_hoc_methods.posthoc_tukey_hsd(df=self.df, feature=self.feature)
                    type_posthoc = 'tukey'
                else:
                    # ?類?뇣??(O), ?源낇뀋?怨쀪쉐 (O), equal sample size (X) --> Fisher, Scheffe, Bonferroni
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
                # ?類?뇣??(O), ?源낇뀋?怨쀪쉐 (X) --> Games-Howell post hoc
                p_posthoc = post_hoc_methods.posthoc_games_howell(df=self.df, feature=self.feature)
                type_posthoc = 'games_howell'
            
        else:
            # ?類?뇣??(X) --> Kruskal-Wallis Test???怨뺚뀲 post hoc 筌욊쑵六?

            # Dunn's Test
            dunn_test = sp.posthoc_dunn(df, feature, 'labels', p_adjust=None)
            p_posthoc_dunn = [dunn_test.loc[0, 1], dunn_test.loc[1, 2], dunn_test.loc[0, 2]]
            type_posthoc = 'dunn'
        
            # Conover's Test
            conover_test = sp.posthoc_conover(a = df, val_col = feature, group_col = 'labels', p_adjust=None)
            conover_test = [conover_test.loc[0, 1], conover_test.loc[1, 2], conover_test.loc[0, 2]]

            # Nemenyi 野꺜??
            nemenyi_test = sp.posthoc_nemenyi(df, feature, 'labels')
            nemenyi_test = [nemenyi_test.loc[0, 1], nemenyi_test.loc[1, 2], nemenyi_test.loc[0, 2]]

            p_posthoc = p_posthoc_dunn

        self.p_posthoc = p_posthoc
        if self.type_anova == 'kruskal':
            self.pair_effect_posthoc = [
                cal_cliffs_delta(df.loc[df['labels'] == 0, feature], df.loc[df['labels'] == 1, feature]),
                cal_cliffs_delta(df.loc[df['labels'] == 1, feature], df.loc[df['labels'] == 2, feature]),
                cal_cliffs_delta(df.loc[df['labels'] == 0, feature], df.loc[df['labels'] == 2, feature]),
            ]
            self.pair_effect_type = 'cliffs_delta'
        elif self.type_anova == 'welch':
            self.pair_effect_posthoc = [
                cal_hedges_g(df.loc[df['labels'] == 0, feature], df.loc[df['labels'] == 1, feature]),
                cal_hedges_g(df.loc[df['labels'] == 1, feature], df.loc[df['labels'] == 2, feature]),
                cal_hedges_g(df.loc[df['labels'] == 0, feature], df.loc[df['labels'] == 2, feature]),
            ]
            self.pair_effect_type = 'hedges_g'
        else:
            self.pair_effect_posthoc = [
                cal_cohens_d(df.loc[df['labels'] == 0, feature], df.loc[df['labels'] == 1, feature]),
                cal_cohens_d(df.loc[df['labels'] == 1, feature], df.loc[df['labels'] == 2, feature]),
                cal_cohens_d(df.loc[df['labels'] == 0, feature], df.loc[df['labels'] == 2, feature]),
            ]
            self.pair_effect_type = 'cohens_d'
        self.post_hoc_str = gen_post_hoc_str(self)
        self.type_posthoc = type_posthoc

class categorical_ancova():
    def __init__(self, df, feature, covariates=['age', 'sex', 'AHI']): 
        F_ancova, p_ancova, effect_3group = chi_square(feature, df)

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
        self.effect_3group = effect_3group
        self.effect_3group_type = 'cramers_v'
        self.list_data = list_data

    def post_hoc(self):
        df = self.df.copy()
        
        # if self.p_ancova >= 0.05:
        #     p_posthoc = [1, 1, 1]
        #     F_posthoc = [0, 0, 0]

        comparisons = [(0, 1), (1, 2), (0, 2)]

        p_posthoc = []
        F_posthoc = []
        pair_effect_posthoc = []
        for group1, group2 in comparisons:
            # ??域밸챶竊??怨쀬뵠???袁り숲筌?
            df_filtered = df[df['labels'].isin([group1, group2])]

            # ?????紐꾪맜??뱀몵嚥?域밸챶竊?癰궰??筌ｌ꼶??
            df_filtered = pd.get_dummies(df_filtered, columns=['labels'], drop_first=True)

            # 嚥≪뮇???쎈뼓 ????브쑴苑?筌뤴뫀???닌딄쉐 獄??怨밸?
            model = sm.Logit(df_filtered[self.feature], df_filtered[['labels_%d' % group2] + self.covariates])  # Group_B: A ????B 域밸챶竊? Age: ??륁뵠 ?⑤벉???
            result = model.fit(disp=0)

            p_posthoc.append(result.pvalues['labels_%d' % group2])
            F_posthoc.append(result.tvalues['labels_%d' % group2])
            pair_effect_posthoc.append(np.exp(result.params['labels_%d' % group2]))
                
        self.p_posthoc = p_posthoc
        self.F_posthoc = F_posthoc
        self.pair_effect_posthoc = pair_effect_posthoc
        self.pair_effect_type = 'odds_ratio'
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
    ss_effect = anova_table.loc["C(labels)", "sum_sq"]
    ss_error = anova_table.loc["Residual", "sum_sq"]
    partial_eta2 = ss_effect / (ss_effect + ss_error)

    return F, p, partial_eta2

def cal_cohens_d(group1: pd.Series, group2: pd.Series):
    group1 = pd.to_numeric(group1, errors='coerce').dropna()
    group2 = pd.to_numeric(group2, errors='coerce').dropna()
    n1 = len(group1)
    n2 = len(group2)
    if (n1 < 2) or (n2 < 2):
        return np.nan
    sd1 = group1.std(ddof=1)
    sd2 = group2.std(ddof=1)
    pooled_var = ((n1 - 1) * (sd1 ** 2) + (n2 - 1) * (sd2 ** 2)) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return np.nan
    pooled_sd = np.sqrt(pooled_var)
    return abs((group1.mean() - group2.mean()) / pooled_sd)

def cal_hedges_g(group1: pd.Series, group2: pd.Series):
    group1 = pd.to_numeric(group1, errors='coerce').dropna()
    group2 = pd.to_numeric(group2, errors='coerce').dropna()
    n1 = len(group1)
    n2 = len(group2)
    if (n1 < 2) or (n2 < 2):
        return np.nan
    d = cal_cohens_d(group1, group2)
    if pd.isna(d):
        return np.nan
    df = n1 + n2 - 2
    if df <= 0:
        return np.nan
    J = 1 - (3 / (4 * df - 1))
    return abs(J * d)

def cal_kruskal_epsilon2(H: float, n_total: int, k_groups: int):
    if n_total <= k_groups:
        return np.nan
    return (H - k_groups + 1) / (n_total - k_groups)

def cal_welch_eta2(F: float, ddof1: float, ddof2: float):
    denominator = F * ddof1 + ddof2
    if denominator <= 0:
        return np.nan
    return (F * ddof1) / denominator

def cal_cliffs_delta(group1: pd.Series, group2: pd.Series):
    group1 = pd.to_numeric(group1, errors='coerce').dropna()
    group2 = pd.to_numeric(group2, errors='coerce').dropna()
    n1 = len(group1)
    n2 = len(group2)
    if (n1 == 0) or (n2 == 0):
        return np.nan
    u_stat, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    delta = (2 * u_stat) / (n1 * n2) - 1
    return abs(delta)

def chi_square(feature: str, df: pd.DataFrame):
    cross_tab = pd.crosstab(df['labels'], df[feature]) # 교차 표 생성
    chi2, p_value, dof, expected = chi2_contingency(cross_tab) # 카이 제곱 검정 수행
    n_total = cross_tab.to_numpy().sum()
    min_dim = min(cross_tab.shape[0] - 1, cross_tab.shape[1] - 1)
    if (n_total == 0) or (min_dim <= 0):
        effect_3group = np.nan
    else:
        effect_3group = np.sqrt(chi2 / (n_total * min_dim))

    return chi2, p_value, effect_3group
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
        return ('+')
    elif (pval>=0.001) & (pval<0.01):
        return ('++')
    else:
        return ('+++')
def inverse_inequality(ineqaulity: str):
    if ',' in ineqaulity: # (??륁젟) '.' --> ','
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
            # two group ??쑨??class??癰???λ땾??????揶쎛?館釉?袁⑥쨯 ??롫뮉 ?⑥눘??
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
    stat_demo = pd.DataFrame(columns=['A', 'B', 'C', 'Statistic', 'F', 'p', ' ', 'Post hoc', 'type_posthoc',
                                      'effect_3group', 'effect_3group_type',
                                      'effect_0_vs_1', 'effect_1_vs_2', 'effect_0_vs_2', 'effect_pair_type'])

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
        stat_demo.loc[feature, ' '] = get_significance_cross(_ancova.p_ancova) # significance ?곕떽?

        stat_demo.loc[feature, 'Post hoc'] = replace_posthoc_str(_ancova.post_hoc_str)
        stat_demo.loc[feature, 'type_posthoc'] = _ancova.type_posthoc
        pair_effect_posthoc = getattr(_ancova, 'pair_effect_posthoc', [np.nan, np.nan, np.nan])
        stat_demo.loc[feature, 'effect_3group'] = getattr(_ancova, 'effect_3group', np.nan)
        stat_demo.loc[feature, 'effect_3group_type'] = getattr(_ancova, 'effect_3group_type', '')
        stat_demo.loc[feature, 'effect_0_vs_1'] = pair_effect_posthoc[0]
        stat_demo.loc[feature, 'effect_1_vs_2'] = pair_effect_posthoc[1]
        stat_demo.loc[feature, 'effect_0_vs_2'] = pair_effect_posthoc[2]
        stat_demo.loc[feature, 'effect_pair_type'] = getattr(_ancova, 'pair_effect_type', '')
        dic_posthoc_p[feature] = _ancova.p_posthoc
        dic_posthoc_str[feature] = _ancova.post_hoc_str

    stat_demo['F'] = stat_demo['F'].apply(lambda x: f"{x:.2f}")
    stat_demo['p'] = stat_demo['p'].apply(lambda x: f"{x:.3f}") # 'p' column??????뤿연 ?諭?????땾???袁⑥삋?癒?봺繹먮슣? 癰귣똻肉э쭪?野?formatting

    stat_demo['effect_3group'] = pd.to_numeric(stat_demo['effect_3group'], errors='coerce').apply(lambda x: f"{x:.3f}" if pd.notna(x) else '')
    stat_demo['effect_0_vs_1'] = pd.to_numeric(stat_demo['effect_0_vs_1'], errors='coerce').apply(lambda x: f"{x:.3f}" if pd.notna(x) else '')
    stat_demo['effect_1_vs_2'] = pd.to_numeric(stat_demo['effect_1_vs_2'], errors='coerce').apply(lambda x: f"{x:.3f}" if pd.notna(x) else '')
    stat_demo['effect_0_vs_2'] = pd.to_numeric(stat_demo['effect_0_vs_2'], errors='coerce').apply(lambda x: f"{x:.3f}" if pd.notna(x) else '')
    # ====== Correction for multiple comparison ======
    # p-value ?귐딅뮞??
    p_values = stat_demo['p'].astype(float)

    
    # 癰귣챸?방에?뺣빍 ?대Ŋ??
    if 'bonf' in p_correct_method.lower():
        bonferroni_corrected = multipletests(p_values, alpha=0.05, method='bonferroni')
        stat_demo['p_corrected'] = bonferroni_corrected[1]
        stat_demo['  '] = stat_demo['p_corrected'].apply(get_significance_cross)

    # con1 = table_2.p.astype(float) < 0.05
    # con2 = table_2.bonferroni_p >= 0.05
    # table_2.loc[con1 & con2,['HS', 'Para-I', 'Psy-I', 'Statistic', 'p', ' ', 'bonferroni_p', '  '] ]

    # ????癰귣챸?방에?뺣빍 ?대Ŋ??
    elif 'holm' in p_correct_method.lower():
        holm_corrected = multipletests(p_values, alpha=0.05, method='holm')
        stat_demo['p_corrected'] = holm_corrected[1]
        stat_demo['  '] = stat_demo['p_corrected'].apply(get_significance_cross)

    # con1 = table_2.p.astype(float) < 0.05
    # con2 = table_2.holm_p >= 0.05
    # table_2.loc[con1 & con2,['HS', 'Para-I', 'Psy-I', 'Statistic', 'p', ' ', 'holm_p', '  '] ]

    # 甕겹끉?沃섎챶???紐낃쾿甕곌쑨???대Ŋ??(FDR correction)
    # --> ESS???醫롮벥?源낆춸 ?癒?뼄
    # --> sSOL/SOL, (sSOL-SOL)/SOL ???醫롮벥?源낆춸 ?癒?뼄
    elif 'fdr' in p_correct_method.lower():
        bh_corrected = multipletests(p_values, alpha=0.05, method='fdr_bh')
        stat_demo['p_corrected'] = bh_corrected[1]
        stat_demo['  '] = stat_demo['p_corrected'].apply(get_significance_cross)

    # con1 = table_2.p.astype(float) < 0.05
    # con2 = table_2.p_corrected >= 0.05
    # table_2.loc[con1 & con2,['HS', 'Para-I', 'Psy-I', 'Statistic', 'p', ' ', 'p_corrected', '  '] ]

    stat_demo['p_corrected'] = stat_demo['p_corrected'].apply(lambda x: f"{x:.3f}")
    
    # p value < 0.001 ??野껋럩?? '< 0.001'嚥???볥┛
    con_under_three_decimal = stat_demo.p_corrected.astype(float) < 0.001
    stat_demo.loc[con_under_three_decimal, 'p_corrected'] = '< 0.001'

    # n ??row ?곕떽?
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
    stat_demo.loc['N', 'effect_3group'] = ''
    stat_demo.loc['N', 'effect_3group_type'] = ''
    stat_demo.loc['N', 'effect_0_vs_1'] = ''
    stat_demo.loc['N', 'effect_1_vs_2'] = ''
    stat_demo.loc['N', 'effect_0_vs_2'] = ''
    stat_demo.loc['N', 'effect_pair_type'] = ''
    stat_demo = pd.concat([stat_demo.loc[['N'], :], stat_demo.iloc[:-1, :]])
    stat_demo = stat_demo.loc[:, ['C', 'A', 'B', 'Statistic', 'p', 'p_corrected', '  ',
                                  'effect_3group', 'effect_3group_type',
                                  'effect_0_vs_1', 'effect_1_vs_2', 'effect_0_vs_2', 'effect_pair_type',
                                  'Post hoc']]

    # Sex mean揶???볥┛
    # - Male????쑴???獄쏄퉭???ㅼ쨮 ??뽯뻻 ex) 55
    if df_type != 'eeg':
        stat_demo = convert_sex_values_to_int_percentage(stat_demo, three_group=True)

    stat_demo.rename(columns={"A": "Subtype 1 (PDI)", "B": "Subtype 2 (PPI)", "C": "HS"}, inplace=True)
    

    return stat_demo, dic_posthoc_p, dic_posthoc_str

def show_number_each_cluster(df):
    print()
    print("Para: %d" % (df.labels == 0).sum())
    print("Non-para: %d" % (df.labels == 1).sum())
    print("GS (PI): %d" % (df.labels == 2).sum())
    
    # ????逾???紐꾪뀱?癒?뮉筌왖 ?袁⑷퍥 ?紐꾪뀱 ??쎄문 ?곗뮆??
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
        
    lcls = locals() # exec?????퉸 ?醫딅뼣??癰궰??? 揶쎛?紐꾩궎疫??袁る립 筌ｌ꼶??
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
    - dataframe(df)??feature column?癒?퐣 abnormal value ??볤탢
    - abnormal value???ル굝履??'na', '-' ????쇰펶??
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
    input??곗쨮 ?袁⑤뼎 獄쏆룇? ??롪돌???諭??feature??????ANCOVA class
    """
    def __init__(self, df, feature, covariates=['age', 'sex', 'AHI'], force_mann=False, force_ttest=False, force_normality=False):
        # feature?? covariate揶쎛 餓λ쵎???? ??낅즲嚥???
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
        # force_mann??force_ttest揶쎛 ??뺤쨮 ??삘뀲筌왖 ?類ㅼ뵥
        if all([self.force_mann, self.force_ttest]):
            raise ValueError("force_mann?? force_ttest????뺤쨮 ??삘뀲 揶쏅???揶쎛?紐꾨튊 ??몃빍??")
        
        if self.force_mann:
            self.normality = False
        elif self.force_ttest:
            self.normality = True

        if self.normality:
        # (??됱젟) ?類?뇣??筌띾슣???롫뮉 野껋럩?? t-test 筌욊쑵六?

            t_statistic, p_value = ttest_ind(self.list_data[0], self.list_data[1])

            self.type_anova = 'T-test'
            self.F_ancova = t_statistic # ??곸읈???臾믨쉐????λ땾?? form??筌띿쉸?쎿묾??袁る릭??F_ancova??곕뮉 癰궰????已?????
            self.p_ancova = p_value # ??곸읈???臾믨쉐????λ땾?? form??筌띿쉸?쎿묾??袁る릭??p_ancova??곕뮉 癰궰????已?????
        else:
            # ?類?뇣?源놁뱽 筌띾슣???? ??낅뮉 野껋럩?? Mann-Whitney U Test 筌욊쑵六?

            u_statistic, p_value = mannwhitneyu(self.list_data[0], self.list_data[1])

            self.type_anova = 'Mann_Whitney'
            self.F_ancova = u_statistic # ??곸읈???臾믨쉐????λ땾?? form??筌띿쉸?쎿묾??袁る릭??F_ancova??곕뮉 癰궰????已?????
            self.p_ancova = p_value # ??곸읈???臾믨쉐????λ땾?? form??筌띿쉸?쎿묾??袁る릭??p_ancova??곕뮉 癰궰????已?????

def gen_df_stat_two_group(df, df_type="demo", force_mann=False, force_ttest=False, force_normality=False, show_statistic=False):
    """
    two group ??????쑨???????df_stat????밴쉐??롫뮉 ??λ땾
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
        stat_demo.loc[feature, ' '] = get_significance_asterisk(_two_sample_test.p_ancova) # significance ?곕떽?

    stat_demo['statistic'] = stat_demo['statistic'].apply(lambda x: f"{x:.2f}")
    stat_demo['p'] = stat_demo['p'].apply(lambda x: f"{x:.3f}") # 'p' column??????뤿연 ?諭?????땾???袁⑥삋?癒?봺繹먮슣? 癰귣똻肉э쭪?野?formatting

    # ====== Correction for multiple comparison ======
    # p-value ?귐딅뮞??
    p_values = stat_demo['p'].astype(float)

    # 癰귣챸?방에?뺣빍 ?대Ŋ??
    bonferroni_corrected = multipletests(p_values, alpha=0.05, method='bonferroni')
    stat_demo['p_corrected'] = bonferroni_corrected[1]
    stat_demo['  '] = bonferroni_corrected[0]
    stat_demo['  '] = stat_demo['p_corrected'].apply(get_significance_asterisk)
    stat_demo['p_corrected'] = stat_demo['p_corrected'].apply(lambda x: f"{x:.3f}")

    # # ????癰귣챸?방에?뺣빍 ?대Ŋ??
    # holm_corrected = multipletests(p_values, alpha=0.05, method='holm')
    # table_1['holm_p'] = holm_corrected[1]
    # table_1['holm_significant'] = holm_corrected[0]

    # 甕겹끉?沃섎챶???紐낃쾿甕곌쑨???대Ŋ??(FDR correction)
    # --> ESS???醫롮벥?源낆춸 ?癒?뼄
    # bh_corrected = multipletests(p_values, alpha=0.05, method='fdr_bh')
    # stat_demo['p_corrected'] = bh_corrected[1]
    # stat_demo['  '] = bh_corrected[0]
    # stat_demo['  '] = stat_demo['p_corrected'].apply(get_significance_asterisk)
    # stat_demo['p_corrected'] = stat_demo['p_corrected'].apply(lambda x: f"{x:.3f}")

    # p value < 0.001 ??野껋럩?? '< 0.001'嚥???볥┛
    con_under_three_decimal = stat_demo.p_corrected.astype(float) < 0.001
    stat_demo.loc[con_under_three_decimal, 'p_corrected'] = '< 0.001'

    # n ??row ?곕떽?
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
    # ??? ?귐딅뮞?紐껊굶??疫뀀챷?좂몴??띯뫀諭??뤿연 set??곗쨮 癰궰??
    # 筌뤴뫀諭?疫뀀챷?졾첎? 揶쏆늾?롳쭖?set??疫뀀챷???1??野?
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
    疫꿸퀣??df_stat?癒?퐣??N??? Male % index揶쎛 ?怨뺤쨮 ?브쑬???뤿선 ??됰??? 
    ??? ??뱁뒄????λ땾
    """
    
    if isinstance(list_groups, type(None)):
        if group_num == 2:
            list_groups = ["HS", "Insomnia"]
        elif group_num == 3:
            list_groups = ["HS", "Subtype 1 (PDI)", "Subtype 2 (PPI)"]
    
    df_stat_N_Male = df_stat.loc[['N', 'Male, %'], :]  # [N, Male] index筌?沃섎챶???怨뺤쨮 ?곕뗄??
    df_stat_N_Male.loc["N (male, %)", :] = '' # "N (male, %)" ??곕뮉 ??덉쨮??index????밴쉐??랁? 筌뤴뫀諭?column??揶쏅?????揶쏅??앮에??λ뜃由??
    for group in list_groups:
        df_stat_N_Male.loc["N (male, %)", group] = "%d (%d)" %  (df_stat_N_Male.loc["N", group], df_stat_N_Male.loc["Male, %", group])

    df_stat_merged = pd.concat([df_stat_N_Male.loc[["N (male, %)"], :], df_stat.drop(index=['N', 'Male, %'])])

    return df_stat_merged

