import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nems0 import db
from nems0.utils import smooth

plt.ion()

def lemon_single_day(index = -1, XVARIABLE =''):
    sql = "SELECT * FROM gDataRaw where cellid like 'LMD%' AND training=1 AND bad=0 order by id"

    d_rawfiles = db.pd_query(sql)

    # filter out "bad" sessions
    good_sessions = np.isfinite(d_rawfiles['trials']) & (d_rawfiles['trials'] > 10)
    d_rawfiles = d_rawfiles.loc[good_sessions].reset_index(drop=True)

    plt.figure()
    plt.plot(d_rawfiles['trials'])

    # pick last file or file equal to index
    if index == -1:
        session_id = (d_rawfiles.shape[0] - 2)
    else:
        session_id = (d_rawfiles.shape[0] - (1 + index))

    r = d_rawfiles.loc[session_id]
    path = os.path.join(r['resppath'], r['parmfile'])
    triallog = os.path.join(path, 'trial_log.csv')

    d = pd.read_csv(triallog)
    d['spatial_config'] = 'same'
    d.loc[d['bg_channel'] == d['fg_channel'], 'spatial_config'] = 'diff'
    d.loc[d['bg_channel'] == -1, 'spatial_config'] = 'dichotic'

    # d = pd.read_csv(triallog)
    interesting_data = d[['trial_number', 'response', 'correct', 'response_time',
                          'snr', 'migrate_trial', 'spatial_config']].copy()

    interesting_data.iloc[0]

    # Plots a bar graph showing percent of YVARIABLE by XVARIABLE
    if XVARIABLE == '':
        XVARIABLE = 'migrate_trial'
    YVARIABLE = 'early_nosepoke'
    interesting_data['early_nosepoke'] = (interesting_data['response'] == 'early_np').astype(bool)
    series2 = interesting_data.groupby([XVARIABLE])[YVARIABLE].mean()
    series2.plot(kind='bar')
    plt.xlabel(XVARIABLE)
    plt.ylabel(YVARIABLE)
    plt.title(f'{YVARIABLE} by {XVARIABLE} for session {session_id}')
    plt.show()

    # Plots a bar graph showing the percent of YVARIABLE across group XVARIABLE
    # Specifically for non-early nosepoke trials
    XVARIABLE = ['migrate_trial', 'snr']
    YVARIABLE = 'correct'
    Non_nose_poke = interesting_data.loc[(interesting_data['early_nosepoke'] == False)]
    series3 = Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].mean()
    series3.plot(kind='bar')
    plt.axhline(y=1, linestyle='--')
    plt.axhline(y=0.5, linestyle='--')
    plt.ylim([0, 1])
    plt.xlabel(XVARIABLE)
    plt.ylabel(YVARIABLE)
    plt.title(f'Percent correct for session {r["cellid"]} without early_nps')
    plt.tight_layout()
    plt.show()

    print(Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].count())


# def lemon():


# def slippy_single_day(index = -1, XVARIABLE =''):


def slippy():
    sql = "SELECT * FROM gDataRaw where cellid like 'SLJ%' AND training=1 AND bad=0 order by id"

    d_rawfiles = db.pd_query(sql)

    # filter out "bad" sessions
    good_sessions = np.isfinite(d_rawfiles['trials']) & (d_rawfiles['trials'] > 10)
    d_rawfiles = d_rawfiles.loc[good_sessions].reset_index(drop=True)

    # plt.figure()
    # plt.plot(d_rawfiles['trials'])

    # Identify which experiments had variable snr
    # snr_indexes = np.arange(39, 53)
    real_task_indexes = [33]

    all_data = []

    for i in real_task_indexes:
        session_id = i

        r = d_rawfiles.loc[session_id]
        path = os.path.join(r['resppath'], r['parmfile'])
        triallog = os.path.join(path, 'trial_log.csv')

        d = pd.read_csv(triallog)
        d['spatial_config'] = 'same'
        d.loc[d['bg_channel'] == d['fg_channel'], 'spatial_config'] = 'diff'
        d.loc[d['bg_channel'] == -1, 'spatial_config'] = 'dichotic'

        interesting_data = d[
            ['trial_number', 'response', 'correct', 'response_time', 'hold_duration', 'snr', 'spatial_config']].copy()
        interesting_data['parmfile'] = r['parmfile']
        interesting_data['session_id'] = session_id
        interesting_data['day'] = r['cellid']
        all_data.append(interesting_data)

    interesting_data_all1 = pd.concat(all_data, ignore_index=True)

    interesting_data_all = interesting_data_all1.loc[(interesting_data_all1['day'] != 'SLJ043Ta')]

    interesting_data_all['early'] = (interesting_data_all['response'] == 'early_np').astype(float)
    interesting_data_all['incorrect'] = ((interesting_data_all['response'] != 'early_np') & \
                                         (interesting_data_all['correct'] == False)).astype(float)
    interesting_data_all['correct_val'] = (interesting_data_all['correct']).astype(float)

    session_avg = interesting_data_all.groupby('session_id').mean().reset_index()
    session_avg.groupby(['response'])['correct'].plot.bar()

    session_avg.columns

def vowel_single_day(name, index = -1, XVARIABLE = ''):
    if name == 'SQD':
        sql = "SELECT * FROM gDataRaw where cellid like 'SQD%' AND training=1 AND bad=0 order by id"
    elif name == 'SDS':
        sql = "SELECT * FROM gDataRaw where cellid like 'SDS%' AND training=1 AND bad=0 order by id"

    d_rawfiles = db.pd_query(sql)

    # filter out "bad" sessions
    good_sessions = np.isfinite(d_rawfiles['trials']) & (d_rawfiles['trials'] > 10)
    d_rawfiles = d_rawfiles.loc[good_sessions].reset_index(drop=True)

    plt.figure()
    plt.plot(d_rawfiles['trials'])

    # pick last file or file equal to index
    if index == -1:
        session_id = (d_rawfiles.shape[0] - 2)
    else:
        session_id = (d_rawfiles.shape[0] - (1 + index))

    r = d_rawfiles.loc[session_id]
    path = os.path.join(r['resppath'], r['parmfile'])
    triallog = os.path.join(path, 'trial_log.csv')

    d = pd.read_csv(triallog)

    # Constructs spatial config columns
    d['spatial_config'] = 'monaural'
    d.loc[d['s1_name'] == d['s2_name'], 'spatial_config'] = 'binaural'

    # Constructs pitch column
    import re
    def label_based_on_three_digit_number(text):
        pattern = r'(\d{3})'
        match = re.search(pattern, text)
        if match:
            extracted_number = match.group(1)
            if extracted_number in ["106", "151", "201"]:
                return extracted_number
        return "Not Matched"

    d['combined_stim'] = d['s1_name']
    d.loc[d['s1_name'].isna(), 'combined_stim'] = d['s2_name']
    d['pitch'] = d['combined_stim'].apply(label_based_on_three_digit_number)

    interesting_data = d[
        ['trial_number', 'response', 'correct', 'response_time', 'snr', 'spatial_config', 'pitch']].copy()


    # Plots a bar graph showing percent of YVARIABLE by XVARIABLE
    if XVARIABLE == '':
        XVARIABLE = ['spatial_config', 'pitch']
    YVARIABLE = 'early_nosepoke'
    interesting_data['early_nosepoke'] = (interesting_data['response'] == 'early_np').astype(bool)
    series2 = interesting_data.groupby(XVARIABLE)[YVARIABLE].mean()
    series2.plot(kind='bar')
    plt.xlabel(XVARIABLE)
    plt.ylabel(YVARIABLE)
    plt.title(f'{YVARIABLE} by {XVARIABLE} for session {r["cellid"]}')
    plt.tight_layout()
    plt.show()

    # Plots a bar graph showing the percent of YVARIABLE across group XVARIABLE
    # Specifically for non-early nosepoke trials
    if XVARIABLE == '':
        XVARIABLE = ['spatial_config', 'pitch']
    YVARIABLE = 'correct'
    Non_nose_poke = interesting_data.loc[(interesting_data['early_nosepoke'] == False)]
    series3 = Non_nose_poke.groupby(XVARIABLE)[YVARIABLE].mean()
    series3.plot(kind='bar')
    plt.axhline(y=0.5, linestyle='--')
    plt.ylim([0, 1])
    plt.xlabel(XVARIABLE)
    plt.ylabel(YVARIABLE)
    plt.title(f'Percent correct for session {r["cellid"]} without early_nps')
    plt.tight_layout()
    plt.show()

# def vowel():

if __name__ == "__main__":
    # Vowel columns: ['trial_number', 'response', 'correct', 'response_time', 'snr', 'spatial_config', 'pitch']
    #
    # XVARIABLE = ['spatial_config', 'pitch']
    # # vowel_single_day('SQD', index=3, XVARIABLE=XVARIABLE)
    # vowel_single_day('SQD', index=2, XVARIABLE=XVARIABLE)
    # vowel_single_day('SQD', index=0, XVARIABLE=XVARIABLE)

    # Slippy columns:
    #
    # slippy()

    # Lemon columns: ['trial_number', 'response', 'correct', 'response_time', 'snr', 'migrate_trial', 'spatial_config']
    #
    lemon_single_day(index = 1)