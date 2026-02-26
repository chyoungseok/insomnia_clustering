import os, glob, cv2, random, tempfile, zipfile
import h5py as h5
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules.utils import path_csv, path_np_data
from modules.utils import myPrint

class _load():    
    def __init__(self, path_scalogram=None, reset=False, inclusion_option='only_insomnia', channel_mode='6ch', verbose=True, include_MR='no_use'):

        # ****assert zone*****************
        assert inclusion_option in ['healthy', 'only_insomnia', 'healthy_insomnia', 'only_insomnia_isi10']
        # ********************************
        
        if path_scalogram == None:
            path_scalogram = 'D:/USC/00_data/scalograms/2000t_16f'

        fname_scalogram = 'scalogram_%s_%s.npy' % (path_scalogram.split('/')[-1], inclusion_option)
        fname_df_demo = "df_demo_%s_%s.csv" % (path_scalogram.split('/')[-1], inclusion_option)

        if reset:
            df_demo, scalograms, df_demo_pre_exclusion = run_pipeline(path_scalogram=path_scalogram,
                                                                      inclusion_option=inclusion_option,
                                                                      fname_df_demo=fname_df_demo,
                                                                      fname_scalogram=fname_scalogram,
                                                                      channel_mode=channel_mode,
                                                                      verbose=verbose)
        
        else:
            if (fname_df_demo in os.listdir(path_csv)) & (fname_scalogram in os.listdir(os.path.join(path_np_data, channel_mode))):
                df_demo = pd.read_csv(os.path.join(path_csv, fname_df_demo), index_col=0, encoding="EUC-KR")
                scalograms = np.load(os.path.join(path_np_data, channel_mode, fname_scalogram))
                myPrint("Load '%s' and '%s' %s" % (fname_df_demo, fname_scalogram, scalograms.shape ))
            
            else:
                df_demo, scalograms, df_demo_pre_exclusion = run_pipeline(path_scalogram=path_scalogram,
                                                                          inclusion_option=inclusion_option,
                                                                          fname_df_demo=fname_df_demo,
                                                                          fname_scalogram=fname_scalogram,
                                                                          channel_mode=channel_mode,
                                                                          verbose=verbose)
        
        self.df_demo = df_demo
        # self.df_demo_pre_exclusion = df_demo_pre_exclusion
        self.scalograms = scalograms
        self.path_scalogram = path_scalogram
        self.reset = reset
        self.inclusion_option = inclusion_option
        self.fname_scalogram = fname_scalogram
        self.fname_df_demo = fname_df_demo

'''
functions
'''

def _read_excel_safe(path, **kwargs):
    """
    Read xlsx robustly. Some files contain malformed custom document
    properties that crash openpyxl with a TypeError.
    """
    try:
        return pd.read_excel(path, **kwargs)
    except TypeError as e:
        msg = str(e)
        if "openpyxl.packaging.custom.StringProperty" not in msg:
            raise

        # Rebuild workbook without custom properties and retry.
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with zipfile.ZipFile(path, "r") as zin, zipfile.ZipFile(tmp_path, "w") as zout:
                for item in zin.infolist():
                    if item.filename == "docProps/custom.xml":
                        continue
                    zout.writestr(item, zin.read(item.filename))
            return pd.read_excel(tmp_path, **kwargs)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

def init_df_demo(verbose=True):
    df = _read_excel_safe(os.path.join(path_csv, "Brain age_PSG_raw_Total_200120(whole_data).xlsx"), index_col=0)

    df_demo = df.copy()
    df_demo.columns = df_demo.iloc[2, :] # column 이름 setting
    df_demo = df_demo.iloc[3:-3, 1:] # 쓰레기 row, column 제거
    df_demo.reset_index(inplace=True, drop=True) # index 초기화
    df_demo.index.names = ['index']; df_demo.columns.names = [''] # index와 column names 설정

    df_demo['PSG study Number#'] = df_demo['PSG study Number#'].str.upper() # subject id를 모두 대문자로 변환 
    df_demo.set_index('PSG study Number#', inplace=True, drop=True) # subject id를 index로 설정
    df_demo.age = -df_demo.age # age 값이 음수로 저장되어 있기 때문에 양수로 변환

    myPrint("INIT 1_Number of subjects in df_demo at first: %d" % len(df_demo), verbose=verbose)

    return df_demo

def find_subjects_with_scalograms(path_scalogram, df, verbose=True):
    df_demo = df.copy()

    fnames_scalogram = glob.glob(os.path.join(path_scalogram, '*.h5')) # 존재하는 모든 scalogram의 filenames

    selected_sub_id = []; selected_f_names = []
    for sub_id in df_demo.index.to_list():
        for fname in fnames_scalogram:
            if sub_id in fname:
                # sub_id가 f_name에 포함되는 경우,
                selected_sub_id.append(sub_id)
                selected_f_names.append(fname)

    assert len(selected_f_names) == len(set(selected_f_names)) # 중복되는 f_name이 있는지 조사

    df_demo = df_demo.loc[selected_sub_id, :] # df_demo 에서 scalogram을 가지는 사람만 선택

    myPrint("INIT 2_Number of subjects with scalograms: %d" % len(fnames_scalogram), verbose=verbose)
    myPrint("       Number of patients who provide both 'df_demo' and 'scalograms': %d" % len(selected_f_names), verbose=verbose)

    return df_demo

def add_bai_to_df_demo(df, verbose=True):
    df_demo = df.copy()

    df_bai = _read_excel_safe(os.path.join(path_csv, 'PSG_list2_insomnia_cys.xlsx'), sheet_name='IS', index_col=0)
    df_bai = df_bai.iloc[:, 0:2] # bai, age 정보만 선택

    sub_ids_org = df_demo.index.to_list()
    sub_ids_bai = df_bai.index.to_list()
    common_sub_ids = []

    for sub_id_bai in sub_ids_bai:
        # df_bai와 df_demo의 공통 sub_id 추출
        if sub_id_bai in sub_ids_org:
            common_sub_ids.append(sub_id_bai)
            
    df_demo.loc[common_sub_ids, 'bai'] = df_bai.loc[common_sub_ids, 'bai']

    
    myPrint("INIT 3_Number of patients who provide 'BAI': %d" % len(common_sub_ids), verbose=verbose)

    return df_demo
    
def add_bai_to_df_demo_healthy(df, verbose=True):
    df_demo = df.copy()

    df_bai = _read_excel_safe(os.path.join(path_csv, 'PSG_list2_healthy_cys.xlsx'), sheet_name='Healthy', index_col=0)
    df_bai = df_bai.iloc[:, 0:2] # bai, age 정보만 선택

    sub_ids_org = df_demo.index.to_list()
    sub_ids_bai = df_bai.index.to_list()
    common_sub_ids = []

    for sub_id_org in sub_ids_org:
        # df_bai와 df_demo의 공통 sub_id 추출
        if sub_id_org in sub_ids_bai:
            common_sub_ids.append(sub_id_org)
            
    df_demo.loc[common_sub_ids, 'bai'] = df_bai.loc[common_sub_ids, 'ba']
    
    myPrint("INIT 3_Number of patients who provide 'BAI': %d" % len(common_sub_ids), verbose=verbose)

    return df_demo

def select_columns(df, verbose=True):
    selected_features = ['이름', '환자번호', '성별', 'age', 'BMI', 'TST-Total Sleep time (min)', 'Sleep latency (min)', 'TIB - Total Recording time (min)',\
                        'N2 sleep\nlatency(min)', 'REM latency (min)', 'WASO (min)', 'WASO (%)',\
                        'STAGE 1/TST(%)', 'STAGE 2/TST(%)', 'STAGE 3+4/TST(%)', 'REM (%)', 'AI - Arousal index',\
                        'REM Arousal \nindex(h)', 'NREM Arousal \nindex(h)', 'Sleep Efficiency (%)',\
                        'AHI - Total index A+H', 'ODI(%)', 'BDI Total-2', 'ISI  Total-2',\
                        #'K-BDI2', \
                        'PSQI Total-2', 'SSS', 'ESS_total', '1-Subjective TST (hr)', '3-Subjective Sleep latency  (hr)', 'bai']

    convert_columns = {
        '성별': 'sex',
        'TST-Total Sleep time (min)': 'TST',
        'Sleep latency (min)': 'SOL',
        'TIB - Total Recording time (min)': 'TIB',
        'N2 sleep\nlatency(min)': 'N2_latency',
        'REM latency (min)': 'REM_latency',
        'WASO (min)': 'WASO_min',
        'WASO (%)': 'WASO_rel',
        'STAGE 1/TST(%)': 'N1',
        'STAGE 2/TST(%)': 'N2',
        'STAGE 3+4/TST(%)': 'N3',
        'REM (%)': 'REM',
        'AI - Arousal index': 'AI',
        'REM Arousal \nindex(h)': 'REM_AI_h',
        'NREM Arousal \nindex(h)': 'NREM_h',
        'Sleep Efficiency (%)': 'SE',
        'AHI - Total index A+H': 'AHI',
        'ODI(%)': 'ODI',
        'BDI Total-2': 'BDI',
        'ISI  Total-2': 'ISI',
        'PSQI Total-2': 'PSQI',
        'ESS_total': 'ESS',
        '1-Subjective TST (hr)': 'sTST',
        '3-Subjective Sleep latency  (hr)': 'sSOL',
        #"K-BDI2": "BDI"    
    }

    df_demo = df.copy()

    df_demo = df_demo.loc[:, selected_features] # select columns using designated features
    df_demo.rename(columns=convert_columns, inplace=True) # 복잡한 기존 column 이름을 간단한 형식으로 수정
    df_demo['sSOL'] = df_demo['sSOL'] * 60 # hour --> min 
    df_demo['sTST'] = df_demo['sTST'] * 60 # hour --> min

    df_demo.loc[df_demo.sex == 'M', 'sex'] = 0
    df_demo.loc[df_demo.sex == 'F', 'sex'] = 1

    myPrint("INIT 4 Select features; number of features=%d" % len(selected_features))

    return df_demo 

def remove_abnormal_ahi_isi(df, verbose=True):
    df_demo = df.copy()

    #   - ISI와 AHI는 subject inclusion의 중요한 기준으로 사용되기 때문에 두 값에 문제가 있는 경우 연구에서 제외
    df_demo = rm_abnormal(df_demo, ['ISI'], verbose=False)
    df_demo = rm_abnormal(df_demo, ['AHI'], verbose=False)
    myPrint("INIT 5_Number of patients after removal of abnormals in 'AHI' and 'ISI': %d" % len(df_demo), verbose=verbose)

    return df_demo

def addFeatures(df, verbose=True):
    df_new = df.copy()
    # add custom features (N2_latency/TST, REM_latency/TST)

    # rm abonormal for 'TST', 'N2_latency', 'REM_latency'
    df_new_TST_N2_latency = rm_abnormal(df_new, ['TST', 'N2_latency'], verbose=False)
    df_new_TST_REM_latency = rm_abnormal(df_new, ['TST', 'REM_latency'], verbose=False)

    # 'TST'와 'N2_latency'에 모두 abnormal value가 없는 subject에 대해서만 계산
    df_new_TST_N2_latency['N2_latency'] = df_new_TST_N2_latency['N2_latency'].astype(float)
    df_new_TST_N2_latency['N2_latency_rel'] = df_new_TST_N2_latency['N2_latency'] / df_new_TST_N2_latency['TST']

    # 'TST'와 'REM_latency'에 모두 abnormal value가 없는 subject에 대해서만 계산
    df_new_TST_REM_latency['REM_latency'] = df_new_TST_REM_latency['REM_latency'].astype(float)
    df_new_TST_REM_latency['REM_latency_rel'] = df_new_TST_REM_latency['REM_latency'] / df_new_TST_REM_latency['TST']

    # removal of abnormals가 적용되지 않는 원본 dataframe에 계산한 custom feature 추가
    # df_new_TST_N2_latency와 df_new_TST_REM_latency를 이용하여, abnormal value가 없는 subject에만 각 feature 저장
    # abnormal value가 있는 subject의 경우, custom feature 값이 NaN으로 기록될 것임
    df_new.loc[df_new_TST_N2_latency.index, 'N2_latency_rel'] = df_new_TST_N2_latency['N2_latency_rel']
    df_new.loc[df_new_TST_REM_latency.index, 'REM_latency_rel'] = df_new_TST_REM_latency['REM_latency_rel']

    myPrint("INIT 6_Add custom features to the df_demo", verbose=verbose)
    myPrint("       'N2_latency/TST' was added to the %d patients" % len(df_new_TST_N2_latency), verbose=verbose)
    myPrint("       'REM_latency/TST' was added to the %d patients" % len(df_new_TST_REM_latency), verbose=verbose)

    return df_new

def assign_paradoxical(df, verbose=True):
    df_new = df.copy()

    # Replace 0 SOL with 0.1
    # SOL이 0인 경우, 나누기 오류 발생
    df_new.SOL = df_new.SOL.astype(float)
    con = (df_new['SOL'] == 0)
    df_new.loc[con, 'SOL'] = 1

    # Remove abnormals
    myPrint("     - rm TST_sTST", verbose=False)
    df_new_TST_sTST = rm_abnormal(df_new, ['TST', 'sTST'], verbose=False).loc[:, ['TST', 'sTST']].astype(float)
    myPrint("     - rm TST_sTST_SE", verbose=False)
    df_new_TST_sTST_SE = rm_abnormal(df_new, ['TST', 'sTST', 'SE'], verbose=False).loc[:, ['TST', 'sTST', 'SE']].astype(float)
    myPrint("     - rrm SOL_sSOL", verbose=False)
    df_new_SOL_sSOL = rm_abnormal(df_new, ['SOL', 'sSOL'], verbose=False).loc[:, ['SOL', 'sSOL']].astype(float)
    myPrint("     - rrm TST_sTST_SOL_sSOL_SE", verbose=False)
    df_new_TST_sTST_SOL_sSOL_SE = rm_abnormal(df_new, ['TST', 'sTST', 'SOL', 'sSOL', 'SE'], verbose=False).loc[:, ['TST', 'sTST', 'SOL', 'sSOL', 'SE']].astype(float)
    myPrint("     - rrm TST_sTST_SOL_sSOL", verbose=False)
    df_new_TST_sTST_SOL_sSOL = rm_abnormal(df_new, ['TST', 'sTST', 'SOL', 'sSOL'], verbose=False).loc[:, ['TST', 'sTST', 'SOL', 'sSOL']].astype(float)

    # Add criteria
    # If satisfy the criteria, record 1, meaning paradoxical insomnia
    list_criteria = []
    list_criteria.append( (df_new_SOL_sSOL['sSOL']/df_new_SOL_sSOL['SOL']) > 1.5 )

    # list_criteria.append( (df_new_TST_sTST_SE['SE']>=90) & (df_new_TST_sTST_SE['TST']-df_new_TST_sTST_SE['sTST'])>=60 )

    list_criteria.append( (df_new_TST_sTST_SOL_sSOL_SE['SE']>80) &
                         (((df_new_TST_sTST_SOL_sSOL_SE['sSOL']-df_new_TST_sTST_SOL_sSOL_SE['SOL'])/df_new_TST_sTST_SOL_sSOL_SE['SOL'])>=0.2) &
                         (((df_new_TST_sTST_SOL_sSOL_SE['TST']-df_new_TST_sTST_SOL_sSOL_SE['sTST'])/df_new_TST_sTST_SOL_sSOL_SE['TST'])>=0.2) )
    
    list_criteria.append( (df_new_TST_sTST_SOL_sSOL['TST']>=390) &
                         (df_new_TST_sTST_SOL_sSOL['SOL']<30) &
                         ((df_new_TST_sTST_SOL_sSOL['TST']-df_new_TST_sTST_SOL_sSOL['sTST'])>=120) &
                         ((df_new_TST_sTST_SOL_sSOL['sSOL']/df_new_TST_sTST_SOL_sSOL['SOL'])>1.2) )
    
    list_criteria.append( (df_new_TST_sTST['TST']-df_new_TST_sTST['sTST'])>120 )

    # Add value
    # - value itself through the criteria
    list_criteria_val = []
    list_criteria_val.append( df_new_SOL_sSOL['sSOL']/df_new_SOL_sSOL['SOL'] )
    list_criteria_val.append( df_new_TST_sTST['sTST']/df_new_TST_sTST['TST'] )
    list_criteria_val.append( df_new_TST_sTST['TST']-df_new_TST_sTST['sTST'] )
    list_criteria_val.append( df_new_SOL_sSOL['sSOL']-df_new_SOL_sSOL['SOL'] )
    list_criteria_val.append( (df_new_SOL_sSOL['sSOL'] - df_new_SOL_sSOL['SOL'])/df_new_SOL_sSOL['SOL'] )
    list_criteria_val.append( (df_new_TST_sTST['TST'] - df_new_TST_sTST['sTST'])/df_new_TST_sTST['TST'] )

    list_criteria_val_name = ['ratio_SOL', 'ratio_TST', 'diff_TST', 'diff_SOL', 'rd_SOL', 'rd_TST']

    
    for i, criteria in enumerate(list_criteria):
        df_new.loc[criteria[criteria].index, 'is_paradoxical_%d' % (i+1)] = 1
        df_new.loc[criteria[~criteria].index, 'is_paradoxical_%d' % (i+1)] = 0

    for i, val in enumerate(list_criteria_val):
        df_new.loc[val.index, list_criteria_val_name[i]] = val 

    myPrint("INIT 7_Assign paradoxical criteria and values to the df_demo", verbose=verbose)   
    # for i in range(1,6):
    #     print(df_1['is_paradoxical_%d' % i].sum()){}
    #     print(df_1['is_paradoxical_%d' % i].isna().sum())

    return df_new

def subject_inclusion(df, isi_upper=14, isi_lower=0, ahi_upper=14, ahi_lower=0, inclusion_option=None, verbose=True):
    """
    - Include subjects whose ISI score is within [isi_lower, isi_upper] and AHI score is within [ahi_lower, ahi_upper]
        --> When all the parameters are set as default value, includes only healthy controls
    - Use inclusion_option, to select subjects specifically ("only_insomnia" or "healthy")
        
    - Ouput: df_inclusion
    """

    assert inclusion_option in ['healthy', 'only_insomnia', 'healthy_insomnia', 'only_insomnia_isi10', 'only_osa']

    df_inclusion = df.copy()

    if inclusion_option == 'healthy':
        pass
    elif inclusion_option == 'only_insomnia':
        isi_lower = 15
        isi_upper = 50        
    elif inclusion_option == 'healthy_insomnia':
        isi_upper = 50    
    elif inclusion_option == 'only_insomnia_isi10':
        isi_lower = 10
        isi_upper = 50
    elif inclusion_option == 'only_osa':
        ahi_lower = 15
        ahi_upper = 100


    # Include based on the given AHI range
    con_OSA_upper = df_inclusion['AHI'] <= ahi_upper
    con_OSA_lower = df_inclusion['AHI'] >= ahi_lower
    con_OSA = con_OSA_upper & con_OSA_lower

    # Include based on the given ISI range
    con_INS_upper = df_inclusion['ISI'] <= isi_upper # upper bound of insomnia inclusion
    con_INS_lower = df_inclusion['ISI'] >= isi_lower # lower bound of insomnia inclusion
    con_INS = con_INS_lower & con_INS_upper

    con = con_OSA & con_INS

    myPrint("INIT 8_Subject Inclusion")
    myPrint("       Number of included subjects: %d" % sum(con), verbose=verbose)
    
    df_inclusion = df_inclusion[con]

    return df_inclusion

def load_scalograms(path_scalogram, df, resample_len=None, inclusion_option=None, channel_mode='single', verbose=True, save_npy=False):
    df_demo = df.copy()

    assert channel_mode in ['single', '2ch', '6ch'], "channel_mode must be 'single', '2ch', or '6ch'"
    
    if channel_mode == 'single':
        myPrint("INIT 9_Load Selected Scalograms (Channel mode: single)", verbose=verbose)
        n_channels = 1
    elif channel_mode == '2ch':
        myPrint("INIT 9_Load Selected Scalograms (Channel mode: 2ch)", verbose=verbose)
        n_channels = 2
    elif channel_mode == '6ch':
        myPrint("INIT 9_Load Selected Scalograms (Channel mode: 6ch)", verbose=verbose)
        n_channels = 6

    scalograms = []
    for id in tqdm(df_demo.index.to_list(), desc='     load scalograms'):
        with h5.File(os.path.join(path_scalogram, id + '.h5'), 'r') as f:
            # original shape: (16, 7, 1, T)
            scal = np.array(f['scalogram'])

            # single: first channel only (index 0)
            # 6ch: first 6 channels (0~5), excluding sleep-stage channel
            scal = scal[:, :n_channels, 0, :]  # -> (16, C, T)
            scalograms.append(scal.astype(np.float32))

    if len(scalograms) == 0:
        return np.array([], dtype=np.float32)

    if not timeLen_unity(scalograms):
        print('run resampling...')
        arr_time_len = [temp_data.shape[-1] for temp_data in scalograms]
        if resample_len is None:
            resample_len = int(np.median(arr_time_len))

        resampled_data_list = []
        for temp_data in scalograms:  # (16, C, T)
            resized_ch_list = []
            for ch_idx in range(temp_data.shape[1]):
                resized = cv2.resize(
                    temp_data[:, ch_idx, :],             # (16, T)
                    (resample_len, 16),                  # (W, H)
                    interpolation=cv2.INTER_NEAREST
                )                                        # -> (16, resample_len)
                resized_ch_list.append(resized)

            # (16, resample_len, C)
            temp_resampled = np.stack(resized_ch_list, axis=-1).astype(np.float32)
            resampled_data_list.append(temp_resampled)

        scalograms = np.array(resampled_data_list, dtype=np.float32)
    else:
        # (N, 16, C, T) -> (N, 16, T, C)
        scalograms = np.array(
            [np.transpose(temp_data, (0, 2, 1)) for temp_data in scalograms],
            dtype=np.float32
        )

    myPrint("INIT 9_Load Selected Scalograms", verbose=verbose)
    myPrint("       Channel mode: {}".format(channel_mode), verbose=verbose)
    myPrint("       Shape of scalograms: {}".format(scalograms.shape), verbose=verbose)

    if save_npy:
        np.save(os.path.join(path_np_data, channel_mode, f'scalogram_2000t_16f_{inclusion_option}.npy'), scalograms)
        myPrint("       Scalograms are saved to '%s'" % os.path.join(path_np_data, channel_mode, f'scalogram_2000t_16f_{inclusion_option}.npy'), verbose=verbose)  
    return scalograms

def run_pipeline(path_scalogram, inclusion_option, fname_df_demo,  fname_scalogram, channel_mode='6ch',verbose=True):
    # 1. Init df_demo =================================================
    df_demo = init_df_demo(verbose=verbose)

    # 2. Select patients with scalograms ==============================
    df_demo = find_subjects_with_scalograms(path_scalogram=path_scalogram, df=df_demo, verbose=verbose)

    # 3. Add 'bai' column to the df_demo ==============================
    df_demo = add_bai_to_df_demo(df=df_demo, verbose=verbose)
    df_demo = add_bai_to_df_demo_healthy(df=df_demo, verbose=verbose)

    # 4. Select columns (features) ====================================
    df_demo = select_columns(df_demo, verbose=verbose)

    # 5. Remove abnormal values for (ISI and AHI) =====================
    df_demo = remove_abnormal_ahi_isi(df_demo, verbose=verbose)

    # 6. Add custom features ==========================================
    #   - N2 latency/TST, REM latency/TST
    #   - if there is any abnormal value, the patient with that value will be recorded as NaN in the column
    df_demo = addFeatures(df_demo, verbose=verbose)

    # 7. Assign paradoxical criteria ==================================
    df_demo = assign_paradoxical(df_demo, verbose=verbose)
    df_demo_pre_exclusion = df_demo.copy()

    # 8. Subject Exclusion ============================================
    df_demo = subject_inclusion(df_demo, inclusion_option=inclusion_option, verbose=verbose)

    # 9. Load Scalograms ==============================================
    scalograms = load_scalograms(path_scalogram, df_demo, channel_mode=channel_mode, verbose=verbose)

    df_demo.to_csv(os.path.join(path_csv, fname_df_demo), encoding="EUC-KR")
    np.save(os.path.join(path_np_data, fname_scalogram), scalograms)

    return df_demo, scalograms, df_demo_pre_exclusion

def rm_abnormal(df: pd.DataFrame, list_feature: list, verbose=True):
    
    df = df.copy() 
    
    for feature in list_feature:
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

        if verbose:
            print("     Number of abnormal values of %s: %d" % (feature, con.sum()))

    return df

def timeLen_unity(list_data):
    for i in range(len(list_data)):
        for j in range(len(list_data)):
            if list_data[i].shape[-1] != list_data[j].shape[-1]:
                return False
    
    return True

def resampling(list_data, len_custom=None, plot_result=False):
    # Get time length of each scalogram
    arr_time_len = []
    for temp_data in list_data:
        arr_time_len.append(temp_data.shape[-1])
    
    if len_custom == None:
        len_custom = int(np.median(arr_time_len))
    
    # Resampling for each scalogram
    resampled_data_list = []
    for temp_data in list_data:
        resized_nearest = cv2.resize(temp_data[:, 0, :], (len_custom, 16), interpolation=cv2.INTER_NEAREST)
        resampled_data_list.append(resized_nearest[:, :, np.newaxis])
        
    if plot_result:
        fig, axes = plt.subplots(1,4, figsize=(15, 3))
        cmap_style = 'jet'
        
        random_index = random.sample(list(range(len(list_data))), 4)
        
        for i, index in enumerate(random_index):
            im_output = axes[i].imshow(list_data[index][:,0,:], cmap=cmap_style, aspect=400,  origin='lower');
            axes[i].set_title("%dth Spectrogram before resize" % i)

        fig, axes = plt.subplots(1,4, figsize=(15, 3))
        for i, index in enumerate(random_index):
            im_output = axes[i].imshow(resampled_data_list[index], cmap=cmap_style, aspect=50,  origin='lower');
            axes[i].set_title("Spectrogram after resize")
        
    return resampled_data_list
