import os
import pandas as pd

from modules.utils import path_csv
from modules import gen_df_MR

df_demo = pd.read_csv(os.path.join(path_csv, 'df_init.csv'), index_col=0, encoding='EUC-KR')
features_cognitive = ['SNSB_Attention', 'SNSB_Language', 'SNSB_Visuospatial', 'SNSB_Verbal_memory', 'SNSB_Visual_memory', 'SNSB_Frontal_and_executive_function']

def get_df_SNSB(_clustering):

    # step1) df_MR과 SMC_insomnia_SNSB.xlsx에 있는 cognitive score는 같음
    # --> df_MR을 그대로 활용
    df_SNSB_1 = get_df_SNSB_1(_clustering)

    # step2) SMC_normal_SNSB.xlsx 
    df_SNSB_2 = get_df_SNSB_2()

    # step3) concat two df_SNSB
    df_SNSB = pd.concat([df_SNSB_1, df_SNSB_2])
    # df_SNSB.dropna(subset=['labels', 'ISI'], inplace=True) # label과 ISI에 NaN 값이 있는 경우 제외
    df_SNSB.dropna(subset=['labels'], inplace=True) # label에 NaN 값이 있는 경우 제외 (ISI 는 고려하지 않는다.)

    # # step4) replace nan index with healthy_0, healthy_1, ...
    df_SNSB = replace_nan_index(df_SNSB)

    df_SNSB.rename(columns={"Sex": "sex", "AGE": "age", 'Total A+H \nIndex(/h)': "AHI"}, inplace=True)

    # labels column을 int형으로 변환
    df_SNSB['labels'] = df_SNSB['labels'].astype(int)
    
    # 'sex' 변환 (1, 2) --> (0, 1)
    # logistic regression을 위해서, 종속변수의 형태는 0과 1로 구성되어야 함
    df_SNSB.loc[df_SNSB.sex == 1, 'sex'] = 0
    df_SNSB.loc[df_SNSB.sex == 2, 'sex'] = 1

    return df_SNSB

def rename_coloumns(df):
    df = df.copy()
    df.rename(columns = {#"Sex": "sex",
                         #"AGE": "age",
                         "Attention": "SNSB_Attention",
                         "Language": "SNSB_Language",
                         "Visuospatial": "SNSB_Visuospatial",
                         "Verbal memory": "SNSB_Verbal_memory",
                         "Visual memory": "SNSB_Visual_memory",
                         "Frontal and executive function": "SNSB_Frontal_and_executive_function",
                         }, inplace=True)
    
    return df

def get_df_SNSB_1(_clustering):
    df_MR = gen_df_MR.get_df_MR()
    df_MR.reset_index(inplace=True)
    df_MR.set_index(['PSG#'], inplace=True)
    # df_MR = rename_coloumns(df_MR)

    # labels setting for df_MR
    df_MR.loc[_clustering.list_org, 'labels'] = _clustering.df_demo.loc[_clustering.list_org, 'labels']
    df_MR.loc[_clustering.list_add, 'labels'] = _clustering.labels[-len(_clustering.list_add):]
    # df_MR.loc[df_MR.ISI < isi_cutoff, 'labels'] = 2 # labels for healthy, clustering label이 할당되었지만, isi_cutoff로 인해 healthy로 label이 변경될 수 있음 ex). 1 --> 2 or 0 --> 2
    # df_MR의 모든 subject는 insomnia임

    # list_mr_I = df_MR.loc[(df_MR.ISI >= isi_cutoff) & df_MR.is_mr_scalo, :].index.to_list() # clustering label이 바로 윗줄에서 할당되었지만, 설정된 isi_cutoff를 통한 insomnia 범주에 속하지 못하면 제외됨
    list_mr_I = df_MR.loc[df_MR.is_mr_scalo, :].index.to_list() # 어차피 모두 insomnia이므로, is_mr_scalo만 고려
    # list_mr_H = df_MR.loc[(df_MR.ISI < isi_cutoff), :].index.to_list()

    # df_SNSB_1 = df_MR.loc[list_mr_I + list_mr_H, ['labels', 'ISI', "Sex", "AGE", 'Total A+H \nIndex(/h)'] + features_cognitive]
    df_SNSB_1 = df_MR.loc[list_mr_I, ['labels', 'ISI', "Sex", "AGE", 'Total A+H \nIndex(/h)'] + features_cognitive]
    df_SNSB_1.index.name = ''

    return df_SNSB_1

def get_df_SNSB_2(isi_cutoff=15):
    df_SNSB_2 = pd.read_excel(os.path.join(path_csv, 'cognitive score', 'SMC_normal_SNSB.xlsx'), index_col=0)
    df_SNSB_2.index.name = ''

    # rename
    df_SNSB_2 = rename_coloumns(df_SNSB_2)

    # remove nan value of ISI
    # (수정) ISI 상관 없이 모든 subject 포함 (normal 군으로 간주)
    # df_SNSB_2 = df_SNSB_2.loc[~df_SNSB_2.ISI.isna(), ['ISI', "Sex", "AGE", 'Total A+H \nIndex(/h)'] + features_cognitive]
    df_SNSB_2 = df_SNSB_2.loc[:, ['ISI', "Sex", "AGE", 'Total A+H \nIndex(/h)'] + features_cognitive]

    # remove nan value for all the cognitive scores
    nan_att = df_SNSB_2.SNSB_Attention.isna()
    nan_lan = df_SNSB_2.SNSB_Language.isna()
    nan_vis = df_SNSB_2.SNSB_Visuospatial.isna()
    nan_verm = df_SNSB_2['SNSB_Verbal_memory'].isna()
    nan_vism = df_SNSB_2['SNSB_Visual_memory'].isna()
    nan_fro = df_SNSB_2['SNSB_Frontal_and_executive_function'].isna()
    con = nan_att & nan_lan & nan_vis & nan_verm & nan_vism & nan_fro
    # print("\nnumber of subjects who don't have any cognitive score: %d" % con.sum())
    df_SNSB_2 = df_SNSB_2.loc[~con, :]

    # 3-1. 중복 id 검사
    # df_SNSB_1의 subject가 df_SNSB_2에 중복되는지 확인
    #  - df_SNSB_1은 df_MR로부터 비롯되기 때문에, df_MR과 비교
    #  --> 결론) 중복되는 id 없음
    #  --> df_SNSB_1과 df_SNSB_2를 concat 하자 !
    df_MR = gen_df_MR.get_df_MR()
    # test_id_overlap(df_SNSB_2, df_MR)

    df_SNSB_2 = set_psg_id(df_SNSB_2) # PSG id 설정

    # df_SNSB_2.reset_index(inplace=True, drop=False)
    # df_SNSB_2.set_index(['PSG#'], inplace=True)

    # PE170439 제외 (OSA 환자)
    df_SNSB_2 = df_SNSB_2[df_SNSB_2['PSG#'] != 'PE170439']
    df_SNSB_2.drop('PSG#', axis=1, inplace=True)
    # df_SNSB_2.drop(index=['PE170439'], inplace=True)

    # add labels of '2' considering they are all healthy (isi<15)
    # --> df_SNSB_2는 SMC_normal_SNSB.xlsx의 dataframe이므로 애초에 모두 normal임
    # --> 따라서 isi_cutoff 없이 모두 사용
    # --> 따라서 모두 label을 2로 취해줌
    # df_SNSB_2.loc[df_SNSB_2.ISI < isi_cutoff, 'labels'] = 2 # (수정 전)
    df_SNSB_2.loc[:, 'labels'] = 2 # (수정 후)
    
    df_SNSB_2.index = df_SNSB_2.index.map(lambda x: int(x[1:]))
    
    return df_SNSB_2

def replace_nan_index(df):
    df = df.copy()

    new_index = df.index.tolist()

    # NaN 인덱스를 가진 위치를 찾아 새로운 인덱스를 할당합니다.
    nan_indices = [i for i, idx in enumerate(df.index) if pd.isna(idx)]
    for i, idx in enumerate(nan_indices):
        new_index[idx] = 'healthy_{}'.format(i)

    # 새로운 인덱스 리스트로 DataFrame의 인덱스를 업데이트합니다.
    df.index = new_index

    return df

def set_psg_id(df_SNSB):
    """ PSG id setting
    - df_SNSB에는 "LE001"과 같은 id가 포함되어 있지 않음
    - df_demo로부터 가져오기
    """
    df_SNSB = df_SNSB.copy()
 
    for id in df_SNSB.index:
        id_modi = int(id.split('s')[1])
        if id_modi in df_demo.환자번호.values: # df_demo에서 먼저 우선적으로 가져온다 
            df_SNSB.loc[id, "PSG#"] = df_demo.loc[df_demo.환자번호==id_modi,].index.values[0]

    return df_SNSB

def test_id_overlap(df1: pd.DataFrame, df2: pd.DataFrame):
    """ 
    - 두 dataframe의 index간에 중복되는 id가 있는지 비교
    
    **** df1, df2 구분 필수 ! ****
    - df1의 index 예시) 's00012123': str
    - df2의 index 예시) 21312: int
    """

    print("=======================")
    print("Start id overlap test !")

    cnt = 0
    for id in df1.index:
        id_modi = int(id.split('s')[1])
        # print(id_modi)
        if id_modi in df2.index.values:
            print(id_modi)
            cnt += 1
    print("=======================")
        
    
