import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns

from modules.utils import path_csv
from modules import load, summary

df_demo_HI_psm = pd.read_csv(os.path.join(path_csv, 'df_demo_HI_psm_updated_labels_6ch.csv'), encoding="EUC-KR", index_col=0)

class paraInsCriteria():    
    @staticmethod
    def getCriteriaCondition(df, num_criteria):
        if num_criteria == 0:
            con = (((df['sSOL'])/df['SOL']) > 1.5)  # hours 단위이기 때문에, 분 단위로 변환 (*60)
            
        elif num_criteria == 1:
            con1 = df['sSOL'] < 30 # hours 단위이기 때문에, 분 단위로 변환 (*60)
            con2 = df['SE'] > 87
            con = con1 & con2
        
        elif num_criteria == 2:
            con1 = df['SE'] >= 90
            con2 = (df['TST'] - df['sTST']) >= 60
            con = con1 & con2
            
        elif num_criteria == 3:
            con1 = df['SE'] >= 80
            con2 = ((df['sSOL'] - df['SOL'])/df['SOL']) >= 0.2  # hours 단위이기 때문에, 분 단위로 변환 (*60)
            con3 = ((df['TST'] - df['sTST'])/df['TST']) >= 0.2
            con = con1 & con2 & con3
            
        elif num_criteria == 4:
            con1 = df['TST'] >= 360
            con2 = (df['age']<60) & (df['TST']>360) & (df['TST']<390) & (df['SE']>85)
            con3 = (df['age']>=60) & (df['TST']>360) & (df['TST']<390) & (df['SE']>80)
            con = con1 | con2 | con3
            
        elif num_criteria == 5:
            con1 = df['TST'] >= 390
            con2 = df['SOL'] < 30
            con3 = (df['TST'] - df['sTST']) >= 120
            con4 = ((df['sSOL'])/df['SOL'])*100 > 120 # hours 단위이기 때문에, 분 단위로 변환 (*60)
            con = con1 & con2 & con3 & con4
            
        elif num_criteria == 6:
            con = (df['TST'] - df['sTST']) >= 120
            
        elif num_criteria == 7:
            con1 = ((df['TST'] - df['sTST'])/df['TST']) >= 0.9
            con2 = df['TST'] >= 120
            con = con1 & con2
            
        elif num_criteria == 8:
            con = df['SE'] >= 85
            
        elif num_criteria == 9:
            con = df['TST'] > 360
            
        elif num_criteria == 10:
            con1 = (df['TST']>360) & (df['SE']>85)
            con2 = ((df['TST'] - df['sTST'])>60) | ((df['sSE'] - df['SE'])>=15)
            con = con1 & con2
            
        elif num_criteria == 11:
            con1 = (df['TST']>380) | (df['SE']>=80)
            con2 = ((df['sSOL'] - df['SOL'])>=60) | ((df['TST']-df['sTST'])>=60) | ((df['SE']-df['sSE'])>=15)  # hours 단위이기 때문에, 분 단위로 변환 (*60)
            con = con1 & con2
            
        elif num_criteria == 12: # exclude 되는 criteria
            con1 = df['TST']>390
            con2 = df['SE'] >= 85
            con3 = (df['sSE'] - df['SE']) >= 15
            con4 = (df['TST'] - df['sTST']) >= 60
            con = con1 & con2 & con3 & con4
            
        elif num_criteria == 13:
            con1 = df['TST'] >= 390
            con2 = df['SE'] >= 85
            con = con1 & con2
            
        elif num_criteria == 14:
            con1 = df['TST'] >= 360
            con2 = df['SOL'] <= 30
            con3 = df['WASO_min'] <= 30
            con = con1 & con2 & con3      

        return con   

class get_ParaIns_Criteria():
    def __init__(self, df_demo):
        df_demo = df_demo.copy()

        # select only insomnia patients (ISI >= 15) (X)
        # df_demo = df_demo.loc[df_demo.ISI >= 15, :]
        con_ins = (df_demo.labels != 2)

        # Replace 0 SOL with 0.1
        # SOL이 0인 경우, 나누기 오류 발생
        df_demo = df_demo.copy()
        df_demo.SOL = df_demo.SOL.astype(float)
        con = (df_demo['SOL'] == 0)
        df_demo.loc[con, 'SOL'] = 1
        
        # 'age', 'TST', 'sTST', 'SOL', 'sSOL', 'SE', 'WASO_min' 에 비정상 data가 있는 경우, 해당 subject 제거
        df_demo = load.rm_abnormal(df_demo, ['age', 'TST', 'sTST', 'SOL', 'sSOL', 'SE', 'WASO_min'], verbose=True)
            
        # subjective sleep efficiency (sSE) column 생성
        df_demo['sSE'] = df_demo['sTST'] / df_demo['TIB']
        
        for i in range(15):
            con = paraInsCriteria.getCriteriaCondition(df=df_demo, num_criteria=i)
            
            df_demo.loc[con & con_ins, 'criteria_%d' % i] = 0 # paradoxical insomnia (clustering 결과에서 '0' cluster가 paradoxical 특성을 보임)
            df_demo.loc[(~con) & con_ins, 'criteria_%d' % i] = 1 # insomnia
            df_demo.loc[~con_ins, 'criteria_%d' % i] = 2 # healthy
        
        df_demo.loc[:, 'criteria_0':'criteria_14'] = df_demo.loc[:, 'criteria_0':'criteria_14'].astype(int)

        self.df_demo = df_demo
    
def extract_top_bottom_criteria(df_corr):
    # float type으로 변환 
    df_corr['corr_val'] = df_corr['corr_val'].astype(float) 

    # NaN 값 제거
    df_clean = df_corr.dropna(subset=['corr_val'])

    # 'corr_val' 기준 상위 2개 추출
    top_2 = df_clean.nlargest(2, 'corr_val')['criteria_name']

    # 'corr_val' 기준 하위 2개 추출
    bottom_2 = df_clean.nsmallest(2, 'corr_val')['criteria_name']

    # 결과 반환
    return top_2.tolist(), bottom_2.tolist()

def plot_statistic_heatmap(df, df_p, ax, norm_type=None, ylabel_off=True):
    # Create a mask for cells where p-value >= 0.05
    mask = df_p >= 0.05
    
    # Create a custom colormap that maps the masked cells to light grey
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    
    # Draw the heatmap with the mask and custom colormap
    heatmap = sns.heatmap(df, annot=True, fmt=".2f", cmap=cmap, cbar=True, ax=ax,
                          cbar_kws={'label': 'T value'}, mask=mask, linewidths=.5, linecolor='black', annot_kws={"color": "black"},
                          vmin=-25, vmax=25)

    # Draw the dark grey cells on top of the heatmap
    # sns.heatmap(df, annot=False, fmt=".2f", cmap=mcolors.ListedColormap(['#E0E0E0']), # A9A9A9 C0C0C0 D3D3D3
    #             cbar=False, ax=ax, mask=~mask, linewidths=.5, linecolor='black')
    # 투명한 색상 정의
    rgba_color = (169/255, 169/255, 169/255, 0)
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_gray", [rgba_color, rgba_color])
    sns.heatmap(df, annot=False, fmt=".2f", cmap=cmap, cbar=False, ax=ax, mask=~mask, linewidths=.5, linecolor='black')

    if ylabel_off:
        ax.set_ylabel('')
    else:
        ax.set_ylabel('Feature')

    ax.set_xticklabels(["Criteria %d" % (i+1) for i in range(14)] + ["Clustering"], rotation=45, ha='right', fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=15)

    for spine in ["bottom", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.0)

    # Add text to colorbar (top and bottom)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.text(2.2, 0.97, 'PDI high', ha='center', va='bottom', fontsize=15, transform=cbar.ax.transAxes)
    cbar.ax.text(2.2, 0.025, 'PPI high', ha='center', va='top', fontsize=15, transform=cbar.ax.transAxes)

def plot_p_heatmap(df_p):
    # 사용자 정의 컬러맵 생성
    # 0.05 미만은 연한 빨강, 이상은 옅은 회색
    cmap = ListedColormap(['lightcoral', 'lightgray'])
    bounds = [0, 0.05, 1]
    norm = BoundaryNorm(bounds, cmap.N)

    # 히트맵 그리기
    plt.figure(figsize=(20, 10))
    ax = sns.heatmap(df_p, cmap=cmap, norm=norm, annot=True, fmt=".4f", linewidths=0.01, linecolor='black')
    ax.set_title("Significance Map (p < 0.05 in light red)")
    plt.show()

def get_criteria_names(exclude_index: list = []): 
    list_selected_criteria = []
    for i in  range(15):
        if i in exclude_index:
            continue
        temp_label = 'criteria_%d' % i # generate temp_label which is label for the current criteria of paradoxical insomnia 
        list_selected_criteria.append(temp_label)

    return list_selected_criteria

def gen_df_for_heatmap(list_criteria_names: list, list_df_stat: list, corrected_p=True):
    """
    Heatmap 생성을 위한 Dataframe 구축
    """
    df_F = pd.DataFrame(columns=list_criteria_names)
    df_p = pd.DataFrame(columns=list_criteria_names)
    df_F.index.name = 'feature'
    df_p.index.name = 'feature'

    features = summary.set_features_1 + summary.set_features_2 + ['bai'] + summary.set_features_3 + summary.set_features_4
    for feature in features:
        for i, temp_criteria in enumerate(list_criteria_names):
            df_F.loc[feature, temp_criteria] = list_df_stat[i].loc[feature, 'statistic']

            if corrected_p:
                if '< ' in list_df_stat[i].loc[feature, 'p_corrected']:
                    df_p.loc[feature, temp_criteria] = 0.01
                else:
                    df_p.loc[feature, temp_criteria] = list_df_stat[i].loc[feature, 'p_corrected']
            else:
                df_p.loc[feature, temp_criteria] = list_df_stat[i].loc[feature, 'p']
            
    df_F = df_F.astype(float)
    df_p = df_p.astype(float)

    return df_F, df_p

def normalization_z(df):
    return df.T.apply(stats.zscore).T # 균일한 비교를 위한 z-score normalization

def normalization_minmax(df):
    return df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)
