import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from statannotations.Annotator import Annotator
from scipy.stats import ttest_ind
from tqdm.notebook import tqdm

from modules.utils import path_csv
from modules import statistics

def replace_divide_str(df):
    df = df.copy()
    for feature in df.columns.to_list():
        if '/' in feature:
            df.rename(columns={feature: feature.split('/')[0] + '_' + feature.split('/')[1]}, inplace=True)
        
    return df

def get_starisk(pval):
    if pval>=0.05:
        return 'ns'
    elif (pval>0.01) & (pval<=0.05):
        return '*'
    elif (pval>0.001) & (pval<0.01):
        return '**'
    elif (pval>0.0001) & (pval<0.001):
        return '***'
    else:
        return '****'

class EEG_storage():
    def __init__(self, _demo_summary) -> None:
        df_BandPower_WholeNight = replace_divide_str(pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_WholeNight.csv", index_col=0))
        df_BandPower_WholeNight_Stage = replace_divide_str(pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_WholeNight_Stage.csv", index_col=0))
        df_BandPower_Quartile = replace_divide_str(pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_Quartile.csv", index_col=0))
        df_BandPower_Quartile_Stage = replace_divide_str(pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_Quartile_Stage.csv", index_col=0))
        df_BandPower_Half = replace_divide_str(pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_Half.csv", index_col=0))
        df_BandPower_Half_Stage = replace_divide_str(pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_Half_Stage.csv", index_col=0))

        df_BandPower_WholeNight_Stage = add_alpha_delta_ratio(df_BandPower_WholeNight_Stage.copy())
        df_BandPower_Quartile_Stage = add_alpha_delta_ratio(df_BandPower_Quartile_Stage.copy())
        
        list_df_eeg = [df_BandPower_Half, df_BandPower_Half_Stage, df_BandPower_WholeNight, df_BandPower_WholeNight_Stage, df_BandPower_Quartile, df_BandPower_Quartile_Stage]
        list_statistic_fname = ["statistic_Half", "statistic_Half_Stage", "statistic_WholeNight", "statistic_WholeNight_Stage", "statistic_Quartile", "statistic_Quartile_Stage"]
        df_demo_HI_psm = _demo_summary.df_demo_HI_psm.copy()

        session_dic_posthoc_p = {}
        list_df_eeg_out = []
        for temp_df, statistic_fname in tqdm(zip(list_df_eeg, list_statistic_fname)):
            # if statistic_fname == 'statistic_Half':
            #     pass
            df_eeg = temp_df
            df_eeg = df_eeg.loc[df_demo_HI_psm.index, :]
            df_eeg.loc[df_demo_HI_psm.index, 'labels'] = df_demo_HI_psm.labels
            list_df_eeg_out.append(df_eeg.copy())
            df_eeg.loc[df_demo_HI_psm.index, 'sex'] = df_demo_HI_psm.sex
            df_eeg.loc[df_demo_HI_psm.index, 'age'] = df_demo_HI_psm.age
            df_eeg.loc[df_demo_HI_psm.index, 'AHI'] = df_demo_HI_psm.AHI

            df_stat, dic_posthoc_p, _ = statistics.gen_df_stat(df_eeg, df_type='eeg', covariates=['age', 'sex'], force_lm_anova=False, posthoc_method='scheffe')
            df_stat.to_csv(path_csv + "/feature_eeg_power_rel_H_I/%s.csv" % statistic_fname)
            session_dic_posthoc_p[statistic_fname.split('statistic_')[1]] = dic_posthoc_p

        statistic_WholeNight = pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/statistic_WholeNight.csv", index_col=0)
        statistic_WholeNight_Stage = pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/statistic_WholeNight_Stage.csv", index_col=0)
        statistic_Half = pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/statistic_Half.csv", index_col=0)
        statistic_Half_Stage = pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/statistic_Half_Stage.csv", index_col=0)
        statistic_Quartile = pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/statistic_Quartile.csv", index_col=0)
        statistic_Quartile_Stage = pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/statistic_Quartile_Stage.csv", index_col=0)

        # self.df_BandPower_Half = df_BandPower_Half
        # self.df_BandPower_Half_Stage = df_BandPower_Half_Stage
        # self.df_BandPower_WholeNight = df_BandPower_WholeNight
        # self.df_BandPower_WholeNight_Stage = df_BandPower_WholeNight_Stage
        # self.df_BandPower_Quartile = df_BandPower_Quartile
        # self.df_BandPower_Quartile_Stage = df_BandPower_Quartile_Stage
        
        self.df_BandPower_Half = list_df_eeg_out[0]
        self.df_BandPower_Half_Stage = list_df_eeg_out[1]
        self.df_BandPower_WholeNight = list_df_eeg_out[2]
        self.df_BandPower_WholeNight_Stage = list_df_eeg_out[3]
        self.df_BandPower_Quartile = list_df_eeg_out[4]
        self.df_BandPower_Quartile_Stage = list_df_eeg_out[5]
        
        
        self.df_demo_HI_psm = df_demo_HI_psm

        self.session_dic_posthoc_p = session_dic_posthoc_p

        self.statistic_WholeNight = statistic_WholeNight
        self.statistic_WholeNight_Stage = statistic_WholeNight_Stage
        self.statistic_Half = statistic_Half
        self.statistic_Half_Stage = statistic_Half_Stage
        self.statistic_Quartile = statistic_Quartile
        self.statistic_Quartile_Stage = statistic_Quartile_Stage

    def addLabel_melt_QS(self):
        df = self.df_BandPower_Quartile_Stage.copy()
        # df.loc[self.df_demo_HI_psm.index, "labels"] = self.df_demo_HI_psm.labels
        df.loc[df.labels==0, "cluster"] = "Cluster A"
        df.loc[df.labels==1, "cluster"] = "Cluster B"
        df.loc[df.labels==2, "cluster"] = "Cluster C"
        df.drop(columns='labels', inplace=True)

        df_melt = df.melt(var_name='Freq. Band',
                        value_name='Relative Power',
                        value_vars=df.columns.to_list(),
                        id_vars=("cluster"))
        
        df_melt.loc[df_melt["Freq. Band"].str.contains('STAGE0'), "stage"] = 0
        df_melt.loc[df_melt["Freq. Band"].str.contains('STAGE1'), "stage"] = 1
        df_melt.loc[df_melt["Freq. Band"].str.contains('STAGE2'), "stage"] = 2
        df_melt.loc[df_melt["Freq. Band"].str.contains('STAGE3'), "stage"] = 3
        df_melt.loc[df_melt["Freq. Band"].str.contains('STAGE4'), "stage"] = 4
        df_melt.stage = df_melt.stage.astype(int)
        
        self.df_melt_QS = df_melt

    def plot_quartile_change(self,
                             fontsize=15,
                             ax=None,
                             feature='Delta_Q',
                             ex_feature: list=None,
                             ymin=0.798,
                             ymax=0.842,
                             y_ref_u=0.858,
                             y_space=0.004,
                             ylabel=None,
                             stage_selection=3):
        
        df_melt_QS = self.df_melt_QS.copy()
        df_melt_QS.dropna(subset=['Relative Power'], inplace=True)

        con_feature = df_melt_QS["Freq. Band"].str.contains(feature)
        if ex_feature is not None:
            for feature_this in ex_feature:
                con_ex_feature = ~df_melt_QS["Freq. Band"].str.contains(feature_this)
                con_feature = con_feature & con_ex_feature
        con_stage = df_melt_QS["stage"]==stage_selection

        df_melt_QS_feature = df_melt_QS.loc[con_feature & con_stage, :]
        self.df_melt_QS_feature = df_melt_QS_feature
        
        plt.rcParams["font.family"] = "Helvetica"
        sns.set(style='ticks', context='talk', font_scale=1.2, font='Helvetica')

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6,3))
        new_palette = ['#333333', '#666666', '#999999'] # grey set
        new_palette = ['#B0B0B0', '#cd5c5c', '#d2b48c'] # red set

        line = sns.lineplot(data=df_melt_QS_feature,
                            x="Freq. Band",
                            y="Relative Power",
                            hue="cluster",
                            hue_order=["Cluster C", "Cluster A", "Cluster B"],
                            errorbar=None,
                            ax=ax,
                            palette=new_palette,
                            )

        for line in ax.lines:  # ax.lines는 그래프의 Line2D 객체들의 리스트
            line.set_linestyle('--')
            line.set_linewidth(3)

        grouped_means = df_melt_QS_feature.groupby(["Freq. Band", "cluster"])["Relative Power"].mean().reset_index()

        sns.scatterplot(data=grouped_means,
                        x="Freq. Band",
                        y="Relative Power",
                        hue="cluster",
                        hue_order=["Cluster C", "Cluster A", "Cluster B"],
                        ax=ax,
                        palette=new_palette,
                        s=100,  # 점의 크기 조정
                        legend=False) 

        # ax.set_title("Relative $\Delta$ power in N3", fontsize=20)
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontsize=fontsize)
        ax.set_xlabel('')
        if ylabel is None:
            ax.set_ylabel("Relative $\delta$ power in N3", fontsize=fontsize)
        else:
            ax.set_ylabel(ylabel, fontsize=fontsize)
            
        ax.set_xlim([-0.5, 5])
        ax.set_ylim([ymin, ymax])

        ax.set_facecolor('white')
        sns.despine(ax=ax, top=True, right=True)
        ax.spines['bottom'].set_linewidth(.9)  # x축의 두께 조절
        ax.spines['left'].set_linewidth(.9)    # y축의 두께 조절
        ax.tick_params(axis='x', width=.9)  # x축의 tick 두께 조절
        ax.tick_params(axis='y', width=.9, labelsize=fontsize)  # y축의 tick 두께 조절
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5)) 

        ax.legend(["_no", "_no", "_no", "HS", "SSM", "OSD"], loc="center right", fontsize=fontsize, bbox_to_anchor=(1.0,0.5))

        # y_ref_u
        y_ref_b = y_ref_u * 0.995

        # 유니코드 문자를 지원하는 폰트 경로 설정 (예: Arial Unicode MS)
        # font_path = 'C:/Windows/Fonts/arial.ttf'
        # prop = fm.FontProperties(fname=font_path)
        # # 유니코드 문자를 지원하는 폰트 경로 설정 (예: DejaVu Sans)
        # font_path = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        # prop = fm.FontProperties(fname=font_path[font_select])  # 예시로 첫 번째 폰트를 사용

        p_sig = self.statistic_Quartile_Stage.loc[[f'{feature}_Q0_STAGE{stage_selection}', f'{feature}_Q1_STAGE{stage_selection}', f'{feature}_Q2_STAGE{stage_selection}', f'{feature}_Q3_STAGE{stage_selection}'], '  '].values
        for q in range(4):
            # ax.text(q, y_ref_u, "$\dagger$" * p_sig[q].__len__(), ha='center', va='top', fontsize=11) # ANCOVA significance
            if p_sig[q] == 'ns':
                anova_sig = ' '
            else:
                anova_sig = '#' * p_sig[q].__len__()
            ax.text(q, y_ref_u, anova_sig, ha='center', va='top', fontsize=11) # ANCOVA significance

            posthoc_p = self.session_dic_posthoc_p['Quartile_Stage'][f'{feature}_Q{q}_STAGE{stage_selection}']
            # Post hoc significance
            ax.text(q, y_ref_u - y_space, statistics.get_significance_asterisk(posthoc_p[0], ns_hyphen=True), ha='center', va='top', fontsize=11) # SSM vs OSD
            ax.text(q, y_ref_u - y_space*2, statistics.get_significance_asterisk(posthoc_p[1], ns_hyphen=True), ha='center', va='top', fontsize=11) # OSD vs HS
            ax.text(q, y_ref_u - y_space*3, statistics.get_significance_asterisk(posthoc_p[2], ns_hyphen=True), ha='center', va='top', fontsize=11) # SSM vs HS
        
        ax.text(3.3, y_ref_u, 'Multiple group comparison', ha='left', va='top', fontsize=9)
        ax.text(3.3, y_ref_u - y_space, 'SSM  vs  OSD', ha='left', va='top', fontsize=9)
        ax.text(3.3, y_ref_u - y_space*2, 'OSD  vs  HS', ha='left', va='top', fontsize=9)
        ax.text(3.3, y_ref_u - y_space*3, 'SSM  vs  HS', ha='left', va='top', fontsize=9)
    
    def plot_quartile_change_HS_SSM(self, fontsize=15,
                                    ax=None, 
                                    session_dic_posthoc_p=None,
                                    feature='Delta_Q', ex_feature: list=None,
                                    stage_selection=3, 
                                    ylabel=None, 
                                    y_ref_u=0.858,
                                    ymin=0.798,
                                    ymax=0.842,
                                    y_space=0.004,
                                    show_significance=True):
        df_melt_QS = self.df_melt_QS.copy()
        df_melt_QS = df_melt_QS[df_melt_QS.cluster.isin(['Cluster C', 'Cluster A'])].copy()

        con_feature = df_melt_QS["Freq. Band"].str.contains(feature)
        if ex_feature is not None:
            for feature_this in ex_feature:
                con_ex_feature = ~df_melt_QS["Freq. Band"].str.contains(feature_this)
                con_feature = con_feature & con_ex_feature
        con_stage = df_melt_QS["stage"]==stage_selection

        df_melt_QS_feature = df_melt_QS.loc[con_feature & con_stage, :]
        self.df_melt_QS_feature = df_melt_QS_feature

        plt.rcParams["font.family"] = "Helvetica"
        sns.set(style='ticks', context='talk', font_scale=1.2, font='Helvetica')

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6,3))
        new_palette = ['#B0B0B0', '#cd5c5c'] # red set

        line = sns.lineplot(data=df_melt_QS_feature,
                            x="Freq. Band",
                            y="Relative Power",
                            hue="cluster",
                            hue_order=["Cluster C", "Cluster A"],
                            errorbar=None,
                            ax=ax,
                            palette=new_palette,
                            )

        for line in ax.lines:  # ax.lines는 그래프의 Line2D 객체들의 리스트
            line.set_linestyle('--')
            line.set_linewidth(3)

        grouped_means = df_melt_QS_feature.groupby(["Freq. Band", "cluster"])["Relative Power"].mean().reset_index()

        sns.scatterplot(data=grouped_means,
                        x="Freq. Band",
                        y="Relative Power",
                        hue="cluster",
                        hue_order=["Cluster C", "Cluster A"],
                        ax=ax,
                        palette=new_palette,
                        s=100,  # 점의 크기 조정
                        legend=False) 

        # ax.set_title("Relative $\Delta$ power in N3", fontsize=20)
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontsize=fontsize)
        ax.set_xlabel('')
        ax.set_xlim([-0.5, 5])
        ax.set_ylim([ymin, ymax])
        
        if ylabel is None:
            ax.set_ylabel("Relative $\delta$ power in N3", fontsize=fontsize)
        else:
            ax.set_ylabel(ylabel, fontsize=fontsize)

        ax.set_facecolor('white')
        sns.despine(ax=ax, top=True, right=True)
        ax.spines['bottom'].set_linewidth(.9)  # x축의 두께 조절
        ax.spines['left'].set_linewidth(.9)    # y축의 두께 조절
        ax.tick_params(axis='x', width=.9)  # x축의 tick 두께 조절
        ax.tick_params(axis='y', width=.9, labelsize=fontsize)  # y축의 tick 두께 조절
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5)) 

        ax.legend(["_no", "_no", "HS", "SSM"], loc="center right", fontsize=fontsize, bbox_to_anchor=(1.0,0.5))

        # 유니코드 문자를 지원하는 폰트 경로 설정 (예: Arial Unicode MS)
        # font_path = 'C:/Windows/Fonts/arial.ttf'
        # prop = fm.FontProperties(fname=font_path)
        # # 유니코드 문자를 지원하는 폰트 경로 설정 (예: DejaVu Sans)
        # font_path = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        # prop = fm.FontProperties(fname=font_path[font_select])  # 예시로 첫 번째 폰트를 사용

        p_sig = self.statistic_Quartile_Stage.loc[[f'{feature}_Q0_STAGE{stage_selection}',
                                                   f'{feature}_Q1_STAGE{stage_selection}',
                                                   f'{feature}_Q2_STAGE{stage_selection}', 
                                                   f'{feature}_Q3_STAGE{stage_selection}'], '  '].values
        for q in range(4):
            posthoc_p = session_dic_posthoc_p['Quartile_Stage'][f'{feature}_Q{q}_STAGE{stage_selection}']
            # Post hoc significance
            if show_significance:
                ax.text(q, y_ref_u - y_space*3, statistics.get_significance_asterisk(posthoc_p, ns_hyphen=True), ha='center', va='top', fontsize=fontsize) # SSM vs HS

        if show_significance:
            ax.text(3.3, y_ref_u - y_space*3, 'HS  vs  SSM', ha='left', va='top', fontsize=fontsize)





def add_alpha_delta_ratio(df):
    if 'Alpha_0' in df.columns.tolist():
        df['alpha_delta_0'] = df['Alpha_0']/df['Delta_0']
        df['alpha_delta_1'] = df['Alpha_1']/df['Delta_1']
        df['alpha_delta_2'] = df['Alpha_2']/df['Delta_2']
        df['alpha_delta_3'] = df['Alpha_3']/df['Delta_3']
        df['alpha_delta_4'] = df['Alpha_4']/df['Delta_4']
        
    elif 'Alpha_Q0_STAGE0' in df.columns.tolist():
        df['alpha_delta_Q0_STAGE0'] = df['Alpha_Q0_STAGE0']/df['Delta_Q0_STAGE0']
        df['alpha_delta_Q0_STAGE1'] = df['Alpha_Q0_STAGE1']/df['Delta_Q0_STAGE1']
        df['alpha_delta_Q0_STAGE2'] = df['Alpha_Q0_STAGE2']/df['Delta_Q0_STAGE2']
        df['alpha_delta_Q0_STAGE3'] = df['Alpha_Q0_STAGE3']/df['Delta_Q0_STAGE3']
        df['alpha_delta_Q0_STAGE4'] = df['Alpha_Q0_STAGE4']/df['Delta_Q0_STAGE4']
        
        df['alpha_delta_Q1_STAGE0'] = df['Alpha_Q1_STAGE0']/df['Delta_Q1_STAGE0']
        df['alpha_delta_Q1_STAGE1'] = df['Alpha_Q1_STAGE1']/df['Delta_Q1_STAGE1']
        df['alpha_delta_Q1_STAGE2'] = df['Alpha_Q1_STAGE2']/df['Delta_Q1_STAGE2']
        df['alpha_delta_Q1_STAGE3'] = df['Alpha_Q1_STAGE3']/df['Delta_Q1_STAGE3']
        df['alpha_delta_Q1_STAGE4'] = df['Alpha_Q1_STAGE4']/df['Delta_Q1_STAGE4']
        
        df['alpha_delta_Q2_STAGE0'] = df['Alpha_Q2_STAGE0']/df['Delta_Q2_STAGE0']
        df['alpha_delta_Q2_STAGE1'] = df['Alpha_Q2_STAGE1']/df['Delta_Q2_STAGE1']
        df['alpha_delta_Q2_STAGE2'] = df['Alpha_Q2_STAGE2']/df['Delta_Q2_STAGE2']
        df['alpha_delta_Q2_STAGE3'] = df['Alpha_Q2_STAGE3']/df['Delta_Q2_STAGE3']
        df['alpha_delta_Q2_STAGE4'] = df['Alpha_Q2_STAGE4']/df['Delta_Q2_STAGE4']
        
        df['alpha_delta_Q3_STAGE0'] = df['Alpha_Q3_STAGE0']/df['Delta_Q3_STAGE0']
        df['alpha_delta_Q3_STAGE1'] = df['Alpha_Q3_STAGE1']/df['Delta_Q3_STAGE1']
        df['alpha_delta_Q3_STAGE2'] = df['Alpha_Q3_STAGE2']/df['Delta_Q3_STAGE2']
        df['alpha_delta_Q3_STAGE3'] = df['Alpha_Q3_STAGE3']/df['Delta_Q3_STAGE3']
        df['alpha_delta_Q3_STAGE4'] = df['Alpha_Q3_STAGE4']/df['Delta_Q3_STAGE4']
    return df





band_names = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma', 'DA', 'TA']
stage_to_str = {0: "WAKE",
                1: "N1",
                2: "N2",
                3: "N3",
                4: "REM"}
half_to_str = {0: "First",
               1: "Second"}
quartile_to_str = {0: "1st",
                   1: "2nd",
                   2: "3rd",
                   3: "4th"}

class Whole_Night():
    def __init__(self, df_demo_HI, df_bandpower, feature: str, ax, yaxis_visible: bool, is_legend: bool) -> None:
        _ancova = statistics.ancova(df=df_bandpower, feature=feature)
        _ancova.anova()
        _ancova.post_hoc()

        df_bandpower_feature = df_bandpower[[feature, 'labels']].copy()
        df_bandpower_feature = statistics.rm_abnormal(df_bandpower_feature, feature)

        df_melt_feature = Whole_Night.addLabel_melt_WN(df_bandpower=df_bandpower_feature)

        Whole_Night.plot_each_feature(feature, df_melt_feature, ax, _ancova, yaxis_visible=yaxis_visible, is_legend=is_legend)
                
        self.df_demo_HI = df_demo_HI
        self.df_bandpower = df_bandpower
        self.feature = feature
        self.df_melt_feature = df_melt_feature

    @staticmethod
    def addLabel_melt_WN(df_bandpower):
        df = df_bandpower.copy()
        df.loc[df.labels==0, "cluster"] = "Cluster A"
        df.loc[df.labels==1, "cluster"] = "Cluster B"
        df.loc[df.labels==2, "cluster"] = "Cluster C"

        df.drop(columns='labels', inplace=True)

        df = df.melt(var_name='Freq. Band',
                    value_name='Relative Power',
                    value_vars=df.columns.to_list(),
                    id_vars=("cluster"))

        return df

    @staticmethod
    def plot_each_feature(feature, df_melt_feature, ax, _ancova, yaxis_visible=True, is_legend=True):
        new_palette = ['#022534', '#08546C', '#A0BACC']

        plotting_params = {"data": df_melt_feature,
                           "x": "cluster",
                           "y": 'Relative Power',
                           "order": ['Cluster C', 'Cluster A', 'Cluster B'],
                           }
        
        pvals = _ancova.p_posthoc
        # C, A, B 순서에 맞게 pvalue 정렬
        pvals_0 = pvals[0]
        pvals_1 = pvals[1]
        pvals[0] = pvals[2]
        pvals[1] = pvals_0
        pvals[2] = pvals_1

        pairs = [('Cluster C', 'Cluster A'),
                 ('Cluster A', 'Cluster B'),
                 ('Cluster C', 'Cluster B')
                ]

        ax.set_ylim([np.min(_ancova.mean)*0.95, np.max(_ancova.mean)*1.05])

        bar = sns.barplot(palette=new_palette, errwidth=0, ax=ax, **plotting_params) # width 옵션은 작동 x

        annotator = Annotator(ax, pairs, verbose=False, **plotting_params);
        annotator.set_pvalues(pvals);
        annotator.annotate(line_offset=0.01, line_offset_to_group=0.05);

        ax.set_xlabel('');
        ax.set_ylabel('');
        ax.set_title(feature, fontsize=15)
        ax.set_xticklabels(["INS A", "INS B", "Healthy"], fontsize=12)
        ax.set_xticklabels("")
        ax.yaxis.set_visible(yaxis_visible)

        sns.despine(left=True, right=True, top=True, bottom=True)
        ax.grid(False)
        plt.rcParams['axes.facecolor'] = 'w'

        # 각 막대 맨 위에 숫자값 표시
        for p in bar.patches:
            # bar.annotate(format(p.get_height(), '.2f'), 
            #             (p.get_x() + p.get_width() / 2., p.get_height()*0.98), 
            #             ha = 'center', va = 'center', 
            #             xytext = (0, 9), 
            #             textcoords = 'offset points',
            #             fontweight='bold',  # 글꼴을 굵게 설정
            #             color='k',
            #             fontsize=12
            #             )
            
            # barplot 폭 변경
            # https://zephyrus1111.tistory.com/250
            width = 0.9
            x = p.get_x() # 막대 좌측 하단 x 좌표   
            old_width = p.get_width() # 기존 막대 폭  
            p.set_width(width) # 폭변경  
            p.set_x(x+(old_width-width)/2) # 막대 좌측 하단 x 좌표 업데이트

        
        from matplotlib.patches import Patch
        # if ('TA' in feature) | ('Theta/Alpha' in feature):
        if is_legend:
            legend_elements = [Patch(facecolor=new_palette[0], label='Healthy'),
                               Patch(facecolor=new_palette[1], label='INS A'),
                               Patch(facecolor=new_palette[2], label='INS B')]
            ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.4, 1))

class Whole_Night_Stage():
    def __init__(self) -> None:
        pass
