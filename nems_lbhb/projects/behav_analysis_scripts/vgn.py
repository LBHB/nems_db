import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nems0 import db

import matplotlib as mpl

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
mpl.rcParams['font.family'] = 'Arial'

def import_data(runclass='VGN', animal="SQD",
                start_date="2024-02-01", stop_date=None, min_trials=10):
    startstr=""
    stopstr=""
    if start_date is not None:
        startstr = f"AND gPenetration.pendate>='{start_date}'"
    if stop_date is not None:
        stopstr = f"AND gPenetration.pendate<='{stop_date}'"
    sql = f"SELECT gCellMaster.cellid as session, gDataRaw.* FROM gDataRaw INNER JOIN gCellMaster on gDataRaw.masterid=gCellMaster.id" +\
        f" INNER JOIN gPenetration ON gPenetration.id=gCellMaster.penid" +\
        f" WHERE gDataRaw.runclass = '{runclass}' AND gDataRaw.cellid like '{animal}%' AND bad=0"+\
        f" {startstr} {stopstr} ORDER BY gDataRaw.id"

    d_rawfiles = db.pd_query(sql)
    all_data = []
    for i, r in d_rawfiles.iterrows():
        path = os.path.join(r['resppath'], r['parmfile'])
        triallog = os.path.join(path, 'trial_log.csv')
        try:
            d = pd.read_csv(triallog)
            if d.shape[0]<min_trials:
                assert 0==1
            # Constructs spatial config columns
            d['trial_duration'] = d['response_time'] - d['np_start']
            d['spatial_config'] = 'monaural'
            d.loc[d['s1_name'] == d['s2_name'], 'spatial_config'] = 'binaural'
            # Constructs pitch column
            import re
            def label_based_on_three_digit_number(text):
                if text is np.nan:
                    return "None"
                pattern = r'(\d{3})\.wav'
                match = re.search(pattern, text)
                if match:
                    extracted_number = match.group(1)
                    if extracted_number in ["106", "151", "201"]:
                        return extracted_number
                return "None"
            def get_vowel(text):
                vowels = ['AE','AH','EE','EH','IH','OO','UH','XX']
                text=str(text)
                if len(text)>=2:
                    v = text[:2]
                if v in vowels:
                    return v
                else:
                    return ""

            d.loc[(d['s1idx']==0) & (d['s1_name'].astype(str)=='nan'),'s1_name']="AE.828.1920.2500.106.wav"
            d.loc[(d['s2idx']==0) & (d['s2_name'].astype(str)=='nan'),'s2_name']="AE.828.1920.2500.106.wav"
            d['combined_stim'] = d['s1_name']
            d.loc[d['s1_name'].isna(), 'combined_stim'] = d['s2_name']
            d['pitch'] = d['combined_stim'].apply(label_based_on_three_digit_number)
            d['v1'] = d['s1_name'].apply(get_vowel)
            d['v2'] = d['s2_name'].apply(get_vowel)
            d['parmfile'] = r['parmfile']
            d['session_id'] = r['cellid']
            d['day'] = i+1

            interesting_data = d[
                ['day','parmfile','session_id','trial_number', 'v1','v2','pitch', 'snr', 'spatial_config',
                 'response', 'correct', 'response_time',
                 'trial_duration', 's1idx', 's2idx', 'trial_is_repeat']].copy()


            interesting_data['early'] = (interesting_data['response'] == 'early_np').astype(float)
            interesting_data['incorrect'] = ((interesting_data['response'] != 'early_np') & \
                                                 (interesting_data['correct'] == False)).astype(float)
            interesting_data['correct_val'] = (interesting_data['correct']).astype(float)
            interesting_data['early_nosepoke'] = (interesting_data['response'] == 'early_np').astype(bool)

            # Finds and labels remind trials
            interesting_data['prev_is_wrong'] = False
            interesting_data.loc[interesting_data.index > 0, 'prev_is_wrong'] = (
                    (interesting_data.iloc[:-1]['correct'] == False) &
                    (interesting_data.iloc[:-1]['response'] != 'early_np')).values
            interesting_data['remind_trial'] = interesting_data['prev_is_wrong'] & interesting_data[
                'trial_is_repeat']

            all_data.append(interesting_data)
        except:
            print(f'Missing file or too short {r["parmfile"]}')

    interesting_data_all = pd.concat(all_data, ignore_index=True)

    return interesting_data_all


f,ax = plt.subplots(2,1, figsize=(3,3), sharex=True)
animals = ['SDS','SQD']

for i,a in enumerate(animals):
    data_all = import_data(runclass="VGN", animal=a, start_date="2024-01-01")
    data_filtered = data_all.loc[data_all['remind_trial']==False]

    cpd = data_filtered.groupby(['day','v1'])['correct'].mean()
    cpd = cpd.unstack(-1)
    ax[i].plot(cpd.index,cpd['AE'], label='/ae/ (go)')
    ax[i].plot(cpd.index,cpd['EH'], label='/eh/ (go)')
    ax[i].plot(cpd.index,cpd['XX'], ls='--', label='noise (no-go)')
    ax[i].plot(cpd.index,cpd['OO'], ls='--', label='/oo/ (no-go)')
    ax[i].set_title(f"Animal {i+1}")
    ax[i].set_ylabel("Fraction correct")
ax[1].set_xlabel("Training day")
ax[1].legend(fontsize=6, frameon=False)
plt.tight_layout()
f.savefig('/auto/data/tmp/vgn_performance.pdf')
