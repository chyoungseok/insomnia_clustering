import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from modules.utils import path_csv, path_np_data
from modules import load

if not 'df_init.csv' in os.listdir(path_csv):
    raise ImportError('"No df_init.csv" ... please create.')
else:
    # print("Laod df_demo_2000t_16f_only_insomnia.csv ... Successful !!\n")
    print("Laod df_init.csv ... Successful !!\n")
    df_demo = pd.read_csv(os.path.join(path_csv, 'df_init.csv'), index_col=0, encoding='EUC-KR')

features_thickness = ['Sensorimotor', 'FPN', 'Dorsal', 'Ventral', 'Default', 'Salient', 'Language', 'Auditory', 'Visual', 'Limbic', 'AD_signature', 'Global']
features_bai = ['Sensorimotor_BAI', 'FPN_BAI', 'Dorsal_BAI', 'Ventral_BAI', 'Default_BAI', 'Salient_BAI', 'Language_BAI', 'Auditory_BAI', 'Visual_BAI', 'Limbic_BAI', 'AD_signature_BAI', 'Global_BAI']


def get_df_MR(df_demo=None, verbose=False):
    """ ** 주의 !! **
    본 함수에서 생성된 df_MR은 healthy와 insomnia를 모두 포함함
    본 함수에서 생성된 df_MR은 insomnia 만을 포함함 (2024.05.07)
        --> regional_bai.xlsx에는 임상의에 의해 불면증으로 진단된 환자들의 정보가 기록되어 있음 (ISI 점수와 무관함)
    """

    # insomnia_regional_BAIs.xlsx 
    df_regional_bai = load_regional_bai_xlsx(verbose=verbose)

    # SMC_MRI_INS_PSG_NP.xlsx 
    df_smc_psg = load_SMC_PSG_xlsx(verbose=verbose)

    # insomnia_regional_BAIs.xlsx의 NaN ISI 값 대체
    df_regional_bai = replace_nan_isi_regional_bai(df_regional_bai, df_smc_psg, verbose=verbose)
   
    # PSG id setting
    df_regional_bai = set_psg_id(df_regional_bai, df_smc_psg)

    # scalogram을 보유하고 있는 id만 선택
    list_org, list_add = get_id_with_MR_org_add(df_regional_bai)
    list_whole = list_org + list_add

    # is_mr_scalo column 추가
    df_regional_bai["is_mr_scalo"] = False
    for id in df_regional_bai["PSG#"].values:
        if id in list_whole:
            con = df_regional_bai["PSG#"] == id
            df_regional_bai.loc[con, "is_mr_scalo"] = True

    if verbose:
        print()
        print(f"Number of subjects who provide both a scalogram and MR features: {df_regional_bai.is_mr_scalo.sum()}")

    return df_regional_bai

def gen_additional_scalogram_npy(verbose=False, save_npy=True, channel_mode='6ch'):
    '''
    - path_np_data에, 'only_insomnia' subject에 해당하는 scalograms을 포함하는 numpy array 생성
    a) additional scalograms는 path_scalogram_w_MR에 .h5 형식으로 14개가 저장되어 있음
    b) (이전) 이 중, insomnia만 선택하면 n=13 (line 84)
    b) (변경) ISI를 기준으로 insomnia 선택 (x), is_mr_scalo만 고려 (line 85)
    c) n=13에 해당하는 scalogram 만을 npy 형식으로 path_np_data에 저장
    '''
    
    path_scalogram_w_MR = 'D:/USC/00_data/scalograms_with_MR'
    list_scalogram = os.listdir(path_scalogram_w_MR)
    id_scalogram = []

    for fname in list_scalogram:
        id = fname.split('.')[0]
        id_scalogram.append(id)
    id_scalogram = np.array(id_scalogram)

    df_demo_mr = pd.DataFrame(index=id_scalogram)

    print('\n ------ generating .npy scalograms which also provide MR features')
    print('     include all types of subject (healthy, insomnia ...)')
    scalograms_mr = load.load_scalograms(path_scalogram=path_scalogram_w_MR,
                                        df=df_demo_mr,
                                        resample_len=2000,
                                        channel_mode=channel_mode,
                                        verbose=True)
    print(' ------ generating done.')

    # import matplotlib.pyplot as plt
    # plt.imshow(scalograms_mr[1][:, :, 0], aspect='auto', origin='lower', cmap='hot')

    # ---- insomnia subject만 선택 start ---- #
    # (수정) df_MR에 있는 subject는 모두 insomnia 임
    df_MR = get_df_MR()
    df_MR_scalo_ins = df_MR.loc[df_MR.is_mr_scalo, ] # 모두 insomnia 이므로 별도의 ISI 기준 적용 x, is_mr_scalo만 고려

    arr_bool = [False] * len(id_scalogram)
    for i, id in enumerate(id_scalogram):
        if id in df_MR_scalo_ins['PSG#'].values:
            arr_bool[i] = True
    arr_bool = np.array(arr_bool)

    if verbose:
        print("\nNumber of subjects who are insomnia, providing both a scalogram and MR features: %d" % np.sum(arr_bool))
        print("--> {}".format(arr_bool))
        print("--> {}".format(id_scalogram[arr_bool]))
        print("--> not selected {}".format(id_scalogram[~arr_bool]))

    scalograms_mr = scalograms_mr[arr_bool] # scalogram numpy array에서 insomnia 환자만 선택
    # ---- insomnia subject만 선택 end ---- #
    
    if save_npy:
        npy_fname = 'scalogram_scalograms_with_MR_only_insomnia.npy'
        np.save(os.path.join(path_np_data, channel_mode, npy_fname), scalograms_mr)
        if verbose:
            print("\nSaving scalograms with MR features of only insomnia subjects as .npy file ... Successful !!")
            print(" --> %s" % os.path.join(path_np_data, channel_mode, npy_fname))

def get_id_with_MR_org_add(df_MR, verbose=False):
    " Get psg_id of subjects who provide both a scalogram and MR features"

    df_MR = df_MR.copy()
    # df_MR = get_df_MR()
    # df_MR = df_MR.loc[(df_MR.ISI >= 15) & df_MR.is_mr_scalo, ] # insomnia 선택 & with_scalo 선택
    # df_MR = df_MR.loc[~df_MR['PSG#'].isna(), ] # NaN 제거

    list_scalogram_org = os.listdir("D:/USC/00_data/scalograms/2000t_16f/")
    list_scalogram_add = os.listdir("D:/USC/00_data/scalograms_with_MR_resamled2000/")

    list_org = []
    list_add = []

    for id in df_MR.loc[~df_MR['PSG#'].isna(), 'PSG#'].values:
        if id+'.h5' in list_scalogram_org:
            list_org.append(id)
        elif id+'.h5' in list_scalogram_add:
            list_add.append(id)

    if verbose:
        print()
        print("[Number of scalograms that simultaneously provide MR features]")
        print("'Original', n = %d" % len(list_org))
        print("'Additional', n = %d" % len(list_add))

    return list_org, list_add

def rename_bai_coloumns(df_MR):
    df_MR = df_MR.copy()
    df_MR.rename(columns = {"Attention": "SNSB_Attention",
                            "Language": "SNSB_Language",
                            "Visuospatial": "SNSB_Visuospatial",
                            "Verbal memory": "SNSB_Verbal_memory",
                            "Visual memory": "SNSB_Visual_memory",
                            "Frontal and executive function": "SNSB_Frontal_and_executive_function",
                            "Ventral ": "Ventral",
                            "Language.1": "Language",
                            "AD signature": "AD_signature",
                            "Sensorimotor BAI": "Sensorimotor_BAI",
                            "FPN\nBAI": "FPN_BAI",
                            "Dorsal\nBAI": "Dorsal_BAI",
                            "Ventral \nBAI": "Ventral_BAI",
                            "Default\nBAI": "Default_BAI",
                            "Salient\nBAI": "Salient_BAI",
                            "Language\nBAI": "Language_BAI",
                            "Auditory\nBAI": "Auditory_BAI",
                            "Visual\nBAI": "Visual_BAI",
                            "Limbic\nBAI": "Limbic_BAI",
                            "AD signature\nBAI": "AD_signature_BAI",
                            "Global\nBAI": "Global_BAI"}, inplace=True)
    
    return df_MR

def load_regional_bai_xlsx(verbose=False):
    df_MR = pd.read_excel(os.path.join(path_csv, "insomnia_regional_BAIs.xlsx"), index_col=0)
    df_MR.index = df_MR.index.map(lambda x: int(x[1:])) # subject id 변환 ex) (str)'s00070431' --> (int)70431
    df_MR = rename_bai_coloumns(df_MR) # rename columns ex) "FPN\nBAI" --> "FPN BAI"

    if verbose:
        print("---- Inspection of 'insomnia_regional_BAIs.xlsx'")
        print("Number of insomnia subjects with MR data: %d" % len(df_MR))
        print("Number of subjects without ISI value (NaN): %d" % df_MR.ISI.isna().sum())
        print("    --> {}".format(df_MR.loc[df_MR.ISI.isna(), :].index.to_list()))
    
    return df_MR

def load_SMC_PSG_xlsx(verbose=False):
    if verbose:
        print()
        print("---- Inspection of 'SMC_MRI_INS_PSG_NP.xlsx'")
        
    df_MR = pd.read_excel(os.path.join(path_csv, "SMC_MRI_INS_PSG_NP.xlsx"), index_col=0)

    # remove unnecessary rows and columns
    df_MR.columns = df_MR.iloc[0, :] # 0번째 row를 column 으로 사용
    df_MR = df_MR.iloc[1:, :] # 0번째 row 제거
    df_MR.index.name = '' # index name 변경

    arr = [True] * len(df_MR) # 중간에 나누어진 section 관련 row 제거
    arr[27:30] = [False] * 3
    df_MR = df_MR.loc[arr, :]

    # select only 'SMC_no' and 'ISI' columns
    df_MR = df_MR[['SMC_no', 'ISI', 'PSG#']]
    df_MR.SMC_no = df_MR.SMC_no.map(lambda x: int(x)) # subject id 변환 ex) (str)'00070431' --> (int)70431
    df_MR = df_MR.loc[:, ~df_MR.columns.duplicated(keep='first')] # duplicated는 중복되면 True, 중복되지 않거나 keep column에 대해서는 True를 반환
    if verbose:
        print("Number of subjects 'with' ISI value: %d" % (~df_MR.ISI.isna()).sum())
        print("    --> {}".format(df_MR.loc[~df_MR.ISI.isna(), :].SMC_no.to_list()))
        print("    --> {}".format(df_MR.loc[~df_MR.ISI.isna(), :].ISI.to_list()))
    return df_MR

def replace_nan_isi_regional_bai(df_regional_bai, df_smc_psg, verbose=False):
    '''
    step 1) df_demo에 id가 일치하는 경우, df_demo의 ISI 값으로 먼저 대체
    step 2) df_demo에 id가 일치하지 않는 경우, SMC.xlsx의 ISI 값으로 대체
    '''
    
    ''' 아래는 scatch 단계에서의 흐름
    # 두 엘셀 파일에 저장된 ISI 값이 다르다 ex1) 3788595의 ISI 값이 다름 regional_BAI.xlsx (21) vs. SMC.xlsx (16)
    # --> df_demo에 기록된 ISI를 정답으로 보정 --> df_demo에는 3788595의 ISI 값이 21로 기록되어 있음
    # --> 따라서, regional_BAI.xlsx와 SMC.xlsx의 ISI 값이 다른 경우, SMC.xlsx의 값으로 교정
    # --> 이렇게 할 바에는 애초에 df_demo의 ISI 값으로 모두 보정하자

    # 일치하는 경우도 있음                 ex2) 21392875 --> regional_BAI.xlsx (23) vs. SMC.xlsx (23)
    # 일치하는 경우도 있음                 ex3) 35209336 --> regional_BAI.xlsx (21) vs. SMC.xlsx (21)
    '''

    df_regional_bai = df_regional_bai.copy()

    for id in df_regional_bai.index.to_list():
        if id in df_demo.환자번호.values: # step 1) df_demo에 id가 일치하는 경우, df_demo의 ISI 값으로 먼저 대체
            df_regional_bai.loc[id, 'ISI'] = df_demo.loc[df_demo.환자번호==id, 'ISI'].values

        elif id in df_smc_psg.SMC_no.values: # step 2) df_demo에 id가 일치하지 않는 경우, SMC.xlsx의 ISI 값으로 대체
            if not df_smc_psg.loc[df_smc_psg.SMC_no==id, 'ISI'].empty and not pd.isna(df_smc_psg.loc[df_smc_psg.SMC_no==id, 'ISI']).any():
                df_regional_bai.loc[id, 'ISI'] = df_smc_psg.loc[df_smc_psg.SMC_no==id, 'ISI'].values

    # regional_BAI.xlsx의 ISI NaN value 다시 체크
    if verbose:
        print()
        print("---- Re-inspection of 'insomnia_regional_BAIs.xlsx' after correction")
        print("Number of subjects with NaN value of ISI: %d" % df_regional_bai.ISI.isna().sum())

    if verbose:
        print("Number of insomnia subjects with MR data: %d" % len(df_regional_bai)) # regional_BAI.xlsx의 모든 subject는 insomnia 임

    """
    print('regional_BAI.xlsx %d' % df_regional_bai.loc[3788595, 'ISI'])
    print('SMC.xlsx %d' % df_smc_psg.loc[df_smc_psg.SMC_no==3788595, 'ISI'])
    print('df_demo %d' % df_demo.loc[df_demo['환자번호'] == 3788595, 'ISI'])
    """
    None

    return df_regional_bai

def set_psg_id(df_regional_bai, df_smc_psg):
    """ PSG id setting
    - df_regional_bai에는 "LE001"과 같은 id가 포함되어 있지 않음
    - df_demo와 df_smc_psg로부터 가져오기
    """

    df_regional_bai = df_regional_bai.copy()
    df_smc_psg = df_smc_psg.copy()

    for id in df_regional_bai.index.to_list():
        if id in df_demo.환자번호.values: # df_demo에서 먼저 우선적으로 가져온다 
            df_regional_bai.loc[id, "PSG#"] = df_demo.loc[df_demo.환자번호==id,].index.values[0]

        elif id in df_smc_psg.SMC_no.values:
            df_regional_bai.loc[id, "PSG#"] = df_smc_psg.loc[df_smc_psg.SMC_no==id, 'PSG#'].values[0]

    df_regional_bai = replace_nan_id(df_regional_bai)

    return df_regional_bai

def get_df_MR_thickness_and_bai(_clustering):
    df_MR = get_df_MR()
    df_MR.reset_index(inplace=True)
    df_MR.set_index(['PSG#'], inplace=True)
    df_MR.rename(columns={"Sex": "sex", "AGE": "age", 'Total A+H \nIndex(/h)': "AHI"}, inplace=True)
    
    # 'sex' 변환 (1, 2) --> (0, 1)
    # logistic regression을 위해서, 종속변수의 형태는 0과 1로 구성되어야 함
    df_MR.loc[df_MR.sex == 1, 'sex'] = 0
    df_MR.loc[df_MR.sex == 2, 'sex'] = 1

    # labels setting for df_MR
    df_MR.loc[_clustering.list_org, 'labels'] = _clustering.df_demo.loc[_clustering.list_org, 'labels']
    df_MR.loc[_clustering.list_add, 'labels'] = _clustering.labels[-len(_clustering.list_add):]
    # df_MR.loc[df_MR.ISI < isi_cutoff, 'labels'] = 2 # labels for healthy,  clustering label이 할당되었지만, isi_cutoff로 인해 healthy로 label이 변경될 수 있음 ex). 1 --> 2 or 0 --> 2
    # --> df_MR의 모든 subject는 어차피 insomnia 이므로, healthy를 정의하는 작업은 삭제

    df_MR.dropna(subset=['labels'], inplace=True) # label에 NaN 값이 있는 경우 제외

    # list_mr_I = df_MR.loc[(df_MR.ISI >= isi_cutoff) & df_MR.is_mr_scalo, :].index.to_list()
    list_mr_I = df_MR.loc[df_MR.is_mr_scalo, :].index.to_list() # scalogram을 보유한 인원만 선택
    # list_mr_H = df_MR.loc[(df_MR.ISI < isi_cutoff), :].index.to_list()

    # df_MR_thickness = df_MR.loc[list_mr_I + list_mr_H, ['labels', 'sex', 'age', 'AHI', 'ISI'] + features_thickness]
    df_MR_thickness = df_MR.loc[list_mr_I, ['labels', 'sex', 'age', 'AHI', 'ISI'] + features_thickness]
    # df_MR_thickness = replace_nan_id(df_MR_thickness)
    
    # df_MR_bai = df_MR.loc[list_mr_I + list_mr_H, ['labels', 'sex', 'age', 'AHI', 'ISI'] + features_bai]
    df_MR_bai = df_MR.loc[list_mr_I, ['labels', 'sex', 'age', 'AHI', 'ISI'] + features_bai]
    # df_MR_bai = replace_nan_id(df_MR_bai)

    return df_MR_thickness, df_MR_bai, df_MR

def replace_nan_id(df: pd.DataFrame):
    # 새 인덱스 목록 생성
    new_index = []
    nan_counter = 0
    for idx in df['PSG#']:
        if pd.isna(idx):  # 현재 인덱스가 NaN인 경우
            new_index.append(f"na_id_{nan_counter}")
            nan_counter += 1
        else:
            new_index.append(idx)
    df["PSG#"] = new_index  # 새 인덱스를 DataFrame에 할당
    return df

def get_df_mr_bai_normal():
    """
    normal group의 MRI BAI 정보가 저장된 df를 생성하는 함수
    """

    df = pd.read_excel(os.path.join(path_csv, "SMC_MRI brain age_T1_PSG_NP_composite_score_NC.xlsx"), index_col=0)

    # 0. 기본 frame 설정
    df.set_index(['ID'], drop=True, inplace=True)
    df.index = df.index.map(lambda x: int(x[1:])) # subject id 변환 ex) (str)'s00070431' --> (int)70431

    df.rename(columns={"Sex": "sex", "AGE": "age", 'Total A+H \nIndex(/h)': "AHI"}, inplace=True)
        
    # 1. 'sex' 변환
    # - (1, 2) --> (0, 1)
    # - 0: M, 1: F
    # - logistic regression을 위해서, 종속변수의 형태는 0과 1로 구성되어야 함
    df.loc[df.sex == 1, 'sex'] = 0
    df.loc[df.sex == 2, 'sex'] = 1

    # 2. Rename columns
    df = rename_bai_coloumns(df) # rename columns ex) "FPN\nBAI" --> "FPN BAI"

    # 3. Set a 'labels' column
    df.loc[:, 'labels'] = 2
    df = df.loc[:, ['labels', 'sex', 'age', 'AHI', 'ISI'] + features_bai]

    return df

def merge_df_mr_bai(df_mr_bai_ins: pd.DataFrame, df_mr_bai_norm: pd.DataFrame):

    """
    MR BAI가 저장되어 있는 두 데이터 프레임을 합치는 함수
    - df_mr_bai_ins: insomnia subjects의 df
    - df_mr_bai_norm: normal subjects의 df 
    """
    df_mr_bai_total = pd.concat([df_mr_bai_ins, df_mr_bai_norm])
    df_mr_bai_total.age = df_mr_bai_total.age.astype(int)
    df_mr_bai_total.labels = df_mr_bai_total.labels.astype(int)

    df_mr_bai_total.index.name = 'PSG study Number#'

    return df_mr_bai_total


