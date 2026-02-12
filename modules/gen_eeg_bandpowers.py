import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import os, sys, mne, yasa
from scipy.signal import welch
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from modules.utils import path_csv, path_main

# sampling rate
dic_sf = {
    'LE': 500,
    'LG': 500,
    'MR': 500,
    'PE': 200,
    'ST': 500
}

def match_len(eeg, hypno_up_int):
    if len(eeg) > len(hypno_up_int):
        eeg = eeg[:len(hypno_up_int)]
    elif len(eeg) < len(hypno_up_int):
        hypno_up_int = hypno_up_int[:len(eeg)]

    return eeg, hypno_up_int

def get_df_BandPower_WholeNight(df_BandPower_WholeNight, id, eeg, sf):
    bands_name = ["Delta", "Theta", "Alpha", "Sigma", "Beta", "Gamma"]

    df_BandPower_WholeNight.loc[id, bands_name] = yasa.bandpower(eeg, sf=sf, ch_names='C4A1', relative=True).loc[:, "Delta":"Gamma"].values[0]

    df_BandPower_WholeNight.loc[id, "Delta/Alpha"] = df_BandPower_WholeNight.loc[id, "Delta"] / df_BandPower_WholeNight.loc[id, "Alpha"]
    df_BandPower_WholeNight.loc[id, "Theta/Alpha"] = df_BandPower_WholeNight.loc[id, "Theta"] / df_BandPower_WholeNight.loc[id, "Alpha"]

    return df_BandPower_WholeNight

def get_df_BandPower_WholeNight_Stage(df_BandPower_whole_night_stage, id, eeg, hypno_int, hypno_up_int, sf):
    bands_name = ["Delta", "Theta", "Alpha", "Sigma", "Beta", "Gamma"]

    df_BandPower_0 = pd.DataFrame(index=[id], columns=bands_name) # bandpower for stage 0
    df_BandPower_1 = pd.DataFrame(index=[id], columns=bands_name) # bandpower for stage 1
    df_BandPower_2 = pd.DataFrame(index=[id], columns=bands_name) # bandpower for stage 2
    df_BandPower_3 = pd.DataFrame(index=[id], columns=bands_name) # bandpower for stage 3
    df_BandPower_4 = pd.DataFrame(index=[id], columns=bands_name) # bandpower for stage 4

    if 0 in hypno_int:
        df_BandPower_0.loc[id, bands_name] = yasa.bandpower(eeg, sf=sf, ch_names='C4A1', hypno=hypno_up_int, include=0, relative=True).loc[0, "Delta":"Gamma"].values # Wake
    else:
        df_BandPower_0.loc[id, bands_name] = [None] * 6 # 특정 sleep stage가 아예 없는 경우가 있음

    if 1 in hypno_int:
        df_BandPower_1.loc[id, bands_name] = yasa.bandpower(eeg, sf=sf, ch_names='C4A1', hypno=hypno_up_int, include=1, relative=True).loc[1, "Delta":"Gamma"].values # N1
    else:
        df_BandPower_1.loc[id, bands_name] = [None] * 6
    
    if 2 in hypno_int:
        df_BandPower_2.loc[id, bands_name] = yasa.bandpower(eeg, sf=sf, ch_names='C4A1', hypno=hypno_up_int, include=2, relative=True).loc[2, "Delta":"Gamma"].values # N2
    else:
        df_BandPower_2.loc[id, bands_name] = [None] * 6

    if 3 in hypno_int:
        df_BandPower_3.loc[id, bands_name] = yasa.bandpower(eeg, sf=sf, ch_names='C4A1', hypno=hypno_up_int, include=3, relative=True).loc[3, "Delta":"Gamma"].values # N3
    else:
        df_BandPower_3.loc[id, bands_name] = [None] * 6

    if 4 in hypno_int:
        df_BandPower_4.loc[id, bands_name] = yasa.bandpower(eeg, sf=sf, ch_names='C4A1', hypno=hypno_up_int, include=4, relative=True).loc[4, "Delta":"Gamma"].values # REM
    else:
        df_BandPower_4.loc[id, bands_name] = [None] * 6

    list_df_BandPower = [df_BandPower_0, df_BandPower_1, df_BandPower_2, df_BandPower_3, df_BandPower_4]
    for i in range(len(list_df_BandPower)):
        df_BandPower = list_df_BandPower[i]
        df_BandPower_whole_night_stage.loc[id, ["Delta_%d" % i, "Theta_%d" % i, "Alpha_%d" % i, "Sigma_%d" % i, "Beta_%d" % i, "Gamma_%d" % i ]] = df_BandPower.loc[id, :].values
        
        # ratio 계산
        if df_BandPower_whole_night_stage.loc[id, "Delta_%d" % i] == None:
            DA = None
            TA = None
        else:
            DA = df_BandPower_whole_night_stage.loc[id, "Delta_%d" % i] / df_BandPower_whole_night_stage.loc[id, "Alpha_%d" % i]
            TA = df_BandPower_whole_night_stage.loc[id, "Theta_%d" % i] / df_BandPower_whole_night_stage.loc[id, "Alpha_%d" % i]

        df_BandPower_whole_night_stage.loc[id, "Delta/Alpha_%d" % i] = DA
        df_BandPower_whole_night_stage.loc[id, "Theta/Alpha_%d" % i] = TA
    
    return df_BandPower_whole_night_stage
    
def get_df_BandPower_Quartile(df_BandPower_Quartile, id, eeg, hypno_int, sf):
    quratiles = [0, 1, 2, 3] # 수면 시간을 4등분했을 때, 각 등분을 의미
    
    for temp_quartile in quratiles:
        start_idx_hypno = int(int(hypno_int.size/4)*temp_quartile)
        end_idx_hypno = int(int(hypno_int.size/4)*(temp_quartile+1))

        start_idx_eeg = int(int(hypno_int.size/4)*30*sf*temp_quartile)
        end_idx_eeg = int(int(hypno_int.size/4)*30*sf*(temp_quartile+1))

        eeg_quartile = eeg[start_idx_eeg: end_idx_eeg]

        temp_coloumns = ["Delta_Q%d" % (temp_quartile),
                        "Theta_Q%d" % (temp_quartile),
                        "Alpha_Q%d" % (temp_quartile),
                        "Sigma_Q%d" % (temp_quartile),
                        "Beta_Q%d" % (temp_quartile),
                        "Gamma_Q%d" % (temp_quartile)]

        df_BandPower_Quartile.loc[id, temp_coloumns] = yasa.bandpower(eeg_quartile, sf=sf, ch_names="C4A1", relative=True).loc[:, "Delta":"Gamma"].values[0]

        # power ratio 계산
        df_BandPower_Quartile.loc[id,"Delta/Alpha_Q%d" % temp_quartile] = df_BandPower_Quartile.loc[id,"Delta_Q%d" % temp_quartile] / df_BandPower_Quartile.loc[id,"Alpha_Q%d" % temp_quartile]
        df_BandPower_Quartile.loc[id,"Theta/Alpha_Q%d" % temp_quartile] = df_BandPower_Quartile.loc[id,"Theta_Q%d" % temp_quartile] / df_BandPower_Quartile.loc[id,"Alpha_Q%d" % temp_quartile]

    return df_BandPower_Quartile

def get_df_BandPower_Quartile_Stage(df_BandPower_Quartile_Stage, id, eeg, hypno_int, hypno_up_int, sf):
    quratiles = [0, 1, 2, 3] # 수면 시간을 4등분했을 때, 각 등분을 의미
    stages = [0, 1, 2, 3, 4]

    hypno_len = len(hypno_int)

    for temp_quartile in quratiles:
        start_idx_hypno = int(int(hypno_int.size/4)*temp_quartile)
        end_idx_hypno = int(int(hypno_int.size/4)*(temp_quartile+1))

        start_idx_eeg = int(int(hypno_int.size/4)*30*sf*temp_quartile)
        end_idx_eeg = int(int(hypno_int.size/4)*30*sf*(temp_quartile+1))

        eeg_quartile = eeg[start_idx_eeg: end_idx_eeg]
        hypno_int_quqrtile = hypno_int[start_idx_hypno: end_idx_hypno]
        hypno_up_int_quartile = hypno_up_int[start_idx_eeg: end_idx_eeg]

        # eeg_quartile = eeg[int(hypno_len/4)*temp_quartile*30*sf : int(hypno_len/4)*(temp_quartile+1)*30*sf]
        # hypno_int_quqrtile = hypno_int[int(hypno_len/4)*temp_quartile : int(hypno_len/4)*(temp_quartile+1)]
        # hypno_up_int_quartile = hypno_up_int[int(hypno_len/4)*temp_quartile*30*sf : int(hypno_len/4)*(temp_quartile+1)*30*sf]

        for stage in stages:
            if stage in hypno_int_quqrtile:
                df_BandPower_Quartile_Stage.loc[id, ["Delta_Q%d_STAGE%d" % (temp_quartile, stage),
                                                     "Theta_Q%d_STAGE%d" % (temp_quartile, stage),
                                                     "Alpha_Q%d_STAGE%d" % (temp_quartile, stage),
                                                     "Sigma_Q%d_STAGE%d" % (temp_quartile, stage),
                                                     "Beta_Q%d_STAGE%d" % (temp_quartile, stage),
                                                     "Gamma_Q%d_STAGE%d" % (temp_quartile, stage)]] = \
                                                        yasa.bandpower(eeg_quartile,
                                                                       sf=sf,
                                                                       ch_names='C4A1',
                                                                       hypno=hypno_up_int_quartile,
                                                                       include=stage,
                                                                       relative=True).loc[stage, "Delta":"Gamma"].values[0]
                
                df_BandPower_Quartile_Stage.loc[id, "Delta/Alpha_Q%d_STAGE%d" % (temp_quartile, stage)] = df_BandPower_Quartile_Stage.loc[id, "Delta_Q%d_STAGE%d" % (temp_quartile, stage)] / df_BandPower_Quartile_Stage.loc[id, "Alpha_Q%d_STAGE%d" % (temp_quartile, stage)]
                df_BandPower_Quartile_Stage.loc[id, "Theta/Alpha_Q%d_STAGE%d" % (temp_quartile, stage)] = df_BandPower_Quartile_Stage.loc[id, "Theta_Q%d_STAGE%d" % (temp_quartile, stage)] / df_BandPower_Quartile_Stage.loc[id, "Alpha_Q%d_STAGE%d" % (temp_quartile, stage)]

            else:
                df_BandPower_Quartile_Stage.loc[id, ["Delta_Q%d_STAGE%d" % (temp_quartile, stage),
                                                     "Theta_Q%d_STAGE%d" % (temp_quartile, stage),
                                                     "Alpha_Q%d_STAGE%d" % (temp_quartile, stage),
                                                     "Sigma_Q%d_STAGE%d" % (temp_quartile, stage),
                                                     "Beta_Q%d_STAGE%d" % (temp_quartile, stage),
                                                     "Gamma_Q%d_STAGE%d" % (temp_quartile, stage)]] = [None] * 6
                
                df_BandPower_Quartile_Stage.loc[id, "Delta/Alpha_Q%d_STAGE%d" % (temp_quartile, stage)] = None
                df_BandPower_Quartile_Stage.loc[id, "Theta/Alpha_Q%d_STAGE%d" % (temp_quartile, stage)] = None
                
    return df_BandPower_Quartile_Stage

def get_df_BandPower_Half(df_BandPower_Half, id, eeg, hypno_int, sf):
    halves = [0, 1] # 수면 시간을 2등분했을 때, 각 등분을 의미

    for temp_half in halves:
        # eeg_half = eeg[int(eeg.size/2)*temp_half : int(eeg.size/2)*(temp_half+1)]
        start_idx_hypno = int(int(hypno_int.size/2)*temp_half)
        end_idx_hypno = int(int(hypno_int.size/2)*(temp_half+1))

        start_idx_eeg = int(int(hypno_int.size/2)*30*sf*temp_half)
        end_idx_eeg = int(int(hypno_int.size/2)*30*sf*(temp_half+1))

        eeg_half = eeg[start_idx_eeg: end_idx_eeg]

        temp_coloumns = ["Delta_Half%d" % (temp_half),
                        "Theta_Half%d" % (temp_half),
                        "Alpha_Half%d" % (temp_half),
                        "Sigma_Half%d" % (temp_half),
                        "Beta_Half%d" % (temp_half),
                        "Gamma_Half%d" % (temp_half)]

        df_BandPower_Half.loc[id, temp_coloumns] = yasa.bandpower(eeg_half, sf=sf, ch_names="C4A1", relative=True).loc[:, "Delta":"Gamma"].values[0]

        df_BandPower_Half.loc[id, "Delta/Alpha_Half%d" % temp_half] = df_BandPower_Half.loc[id, "Delta_Half%d" % temp_half] / df_BandPower_Half.loc[id, "Alpha_Half%d" % temp_half]
        df_BandPower_Half.loc[id, "Theta/Alpha_Half%d" % temp_half] = df_BandPower_Half.loc[id, "Theta_Half%d" % temp_half] / df_BandPower_Half.loc[id, "Alpha_Half%d" % temp_half]

    return df_BandPower_Half
   
def get_df_BandPower_Half_Stage(df_BandPower_Half_Stage, id, eeg, hypno_int, hypno_up_int, sf):
    halves = [0, 1]
    stages = [0, 1, 2, 3, 4]

    for half in halves:
        # print(half)

        start_idx_hypno = int(int(hypno_int.size/2)*half)
        end_idx_hypno = int(int(hypno_int.size/2)*(half+1))

        start_idx_eeg = int(int(hypno_int.size/2)*30*sf*half)
        end_idx_eeg = int(int(hypno_int.size/2)*30*sf*(half+1))

        eeg_half = eeg[start_idx_eeg: end_idx_eeg]
        hypno_int_half = hypno_int[start_idx_hypno: end_idx_hypno]
        hypno_up_int_half = hypno_up_int[start_idx_eeg: end_idx_eeg]

        for stage in stages:
            if stage in hypno_int_half:
                df_BandPower_Half_Stage.loc[id, ["Delta_Half%d_STAGE%d" % (half, stage),
                                                 "Theta_Half%d_STAGE%d" % (half, stage),
                                                 "Alpha_Half%d_STAGE%d" % (half, stage),
                                                 "Sigma_Half%d_STAGE%d" % (half, stage),
                                                 "Beta_Half%d_STAGE%d" % (half, stage),
                                                 "Gamma_Half%d_STAGE%d" % (half, stage)]] = \
                                                    yasa.bandpower(eeg_half, sf=sf, ch_names='C4A1', hypno=hypno_up_int_half, include=stage, relative=True).loc[stage, "Delta":"Gamma"].values[0]
                
                df_BandPower_Half_Stage.loc[id, "Delta/Alpha_Half%d_STAGE%d" % (half, stage)] = df_BandPower_Half_Stage.loc[id, "Delta_Half%d_STAGE%d" % (half, stage)] / df_BandPower_Half_Stage.loc[id, "Alpha_Half%d_STAGE%d" % (half, stage)]
                df_BandPower_Half_Stage.loc[id, "Theta/Alpha_Half%d_STAGE%d" % (half, stage)] = df_BandPower_Half_Stage.loc[id, "Theta_Half%d_STAGE%d" % (half, stage)] / df_BandPower_Half_Stage.loc[id, "Alpha_Half%d_STAGE%d" % (half, stage)]

            else:
                df_BandPower_Half_Stage.loc[id, ["Delta_Half%d_STAGE%d" % (half, stage),
                                                 "Theta_Half%d_STAGE%d" % (half, stage),
                                                 "Alpha_Half%d_STAGE%d" % (half, stage),
                                                 "Sigma_Half%d_STAGE%d" % (half, stage),
                                                 "Beta_Half%d_STAGE%d" % (half, stage),
                                                 "Gamma_Half%d_STAGE%d" % (half, stage)]] = [None] * 6
                
                df_BandPower_Half_Stage.loc[id, "Delta/Alpha_Half%d_STAGE%d" % (half, stage)] = None
                df_BandPower_Half_Stage.loc[id, "Theta/Alpha_Half%d_STAGE%d" % (half, stage)] = None
                
                
    return df_BandPower_Half_Stage


def main(df: pd.DataFrame):
    path_C4_preprocessed_hypnoMatched = path_main + '/data/C4_preprocessed_hypnoMatched/' # hypnogram과 time-length가 동일한 preprocessed EEG가 저장되는 경로
    path_hypnogram = path_main + '/data/C4A1_SOLcropped_hypnogram/'

    stages = (0, 1, 2, 3, 4)
    bands_name = ["Delta", "Theta", "Alpha", "Sigma", "Beta", "Gamma"]

    if 'df_BandPower_Half_Stage.csv' in os.listdir(path_csv + "/feature_eeg_power_rel_H_I"):
        df_BandPower_WholeNight = pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_WholeNight.csv", encoding="EUC-KR", index_col=0)
        df_BandPower_WholeNight_Stage = pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_WholeNight_Stage.csv", encoding="EUC-KR", index_col=0)
        df_BandPower_Quartile = pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_Quartile.csv", encoding="EUC-KR", index_col=0)
        df_BandPower_Quartile_Stage = pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_Quartile_Stage.csv", encoding="EUC-KR", index_col=0)
        df_BandPower_Half = pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_Half.csv", encoding="EUC-KR", index_col=0)
        df_BandPower_Half_Stage = pd.read_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_Half_Stage.csv", encoding="EUC-KR", index_col=0)

    else:
        df_BandPower_WholeNight = pd.DataFrame(index=df.index, columns=["Delta", "Theta", "Alpha", "Sigma", "Beta", "Gamma"])
        df_BandPower_WholeNight_Stage = pd.DataFrame(index=df.index)
        df_BandPower_Quartile = pd.DataFrame(index=df.index)
        df_BandPower_Quartile_Stage = pd.DataFrame(index=df.index)
        df_BandPower_Half = pd.DataFrame(index=df.index)
        df_BandPower_Half_Stage = pd.DataFrame(index=df.index)


    for id in tqdm(df.index.to_list()):
        # if (~df_BandPower_Half_Stage.loc[id, :].isna()).sum() > 0:
        #     continue

        fname = id + ".npy"
        sf = dic_sf[id[:2]]

        eeg = np.load(os.path.join(path_C4_preprocessed_hypnoMatched, fname)) # SOLcropped preprocessed EEG load & convert into the unit of uV
        hypno_int = np.load(os.path.join(path_hypnogram, id+".npy"))
        hypno_up_int = yasa.hypno_upsample_to_data(hypno=hypno_int, sf_hypno=(1/30), data=eeg, sf_data=sf, verbose='ERROR')
        eeg, hypno_up_int = match_len(eeg, hypno_up_int)

        # ======================= Whole Night =======================
        
        # relative를 False로 설정하여 absolute power 계산
        df_BandPower_WholeNight = get_df_BandPower_WholeNight(df_BandPower_WholeNight, id, eeg, sf)

    # ======================= Whole Night Stage =======================
    df_BandPower_WholeNight_Stage = get_df_BandPower_WholeNight_Stage(df_BandPower_WholeNight_Stage, id, eeg, hypno_int, hypno_up_int, sf)

    # ======================= Quartile =======================
    df_BandPower_Quartile = get_df_BandPower_Quartile(df_BandPower_Quartile, id, eeg, hypno_int, sf)

    # ======================= Quartile Stage =======================
    df_BandPower_Quartile_Stage = get_df_BandPower_Quartile_Stage(df_BandPower_Quartile_Stage, id, eeg, hypno_int, hypno_up_int, sf)

    # ======================= Half =======================
    df_BandPower_Half = get_df_BandPower_Half(df_BandPower_Half, id, eeg, hypno_int, sf)
    
    # ======================= Half Stage =======================
    df_BandPower_Half_Stage = get_df_BandPower_Half_Stage(df_BandPower_Half_Stage, id, eeg, hypno_int, hypno_up_int, sf)
 
    os.makedirs(path_csv + "/feature_eeg_power_rel_H_I", exist_ok=True)
    df_BandPower_WholeNight.to_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_WholeNight.csv", encoding="EUC-KR")
    df_BandPower_WholeNight_Stage.to_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_WholeNight_Stage.csv", encoding="EUC-KR")
    df_BandPower_Quartile.to_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_Quartile.csv", encoding="EUC-KR")
    df_BandPower_Quartile_Stage.to_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_Quartile_Stage.csv", encoding="EUC-KR")
    df_BandPower_Half.to_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_Half.csv", encoding="EUC-KR")
    df_BandPower_Half_Stage.to_csv(path_csv + "/feature_eeg_power_rel_H_I/df_BandPower_Half_Stage.csv", encoding="EUC-KR")

