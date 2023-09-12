import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nems0 import db
from nems0.utils import smooth

plt.ion()

sql = "SELECT * FROM gDataRaw where cellid like 'LMD%' AND training=1 AND bad=0"

d_rawfiles=db.pd_query(sql)

# filter out "bad" sessions
good_sessions = np.isfinite(d_rawfiles['trials']) & (d_rawfiles['trials']>10)
d_rawfiles=d_rawfiles.loc[good_sessions].reset_index(drop=True)

plt.figure()
plt.plot(d_rawfiles['trials'])

# pick last file
session_id=44

r = d_rawfiles.loc[session_id]
path = os.path.join(r['resppath'], r['parmfile'])
triallog = os.path.join(path,'trial_log.csv')

d = pd.read_csv(triallog)
interesting_data = d[['trial_number','response','correct','response_time', 'snr']].copy()

plt.figure()

sm_win_size=20

early = (interesting_data['response']=='early_np').astype(float)
incorrect = ((interesting_data['response']!='early_np') & \
        (interesting_data['correct']==False)).astype(float)
incorrect_rate=smooth(incorrect,window_len=sm_win_size)

early_rate=smooth(early,window_len=sm_win_size)
correct_rate=smooth(d['correct'].astype(float),window_len=sm_win_size)
plt.plot(early_rate,label='early rate')
plt.plot(correct_rate,label='correct rate')
plt.plot(incorrect_rate,label='incorrect rate')
plt.legend()
plt.xlabel('Trial')
plt.ylabel('Rate')
plt.title(r['parmfile'])

all_data = []
for session_id,r in d_rawfiles.iterrows():

    path = os.path.join(r['resppath'], r['parmfile'])
    print(path)
    triallog = os.path.join(path,'trial_log.csv')

    d = pd.read_csv(triallog)
    interesting_data = d[['trial_number','response','correct','response_time','hold_duration']].copy()
    interesting_data['parmfile']=r['parmfile']
    interesting_data['session_id']=session_id
    interesting_data['day']=r['cellid']
    all_data.append(interesting_data)

interesting_data_all = pd.concat(all_data, ignore_index=True)
interesting_data_all['early']=(interesting_data_all['response']=='early_np').astype(float)
interesting_data_all['incorrect']=((interesting_data_all['response']!='early_np') & \
        (interesting_data_all['correct']==False)).astype(float)
interesting_data_all['correct_val']=(interesting_data_all['correct']).astype(float)

day_avg = interesting_data_all.groupby('trial_number').mean().reset_index()
session_avg = interesting_data_all.groupby('session_id').mean().reset_index()

plt.figure()

sm_win_size=10
incorrect_rate=smooth(day_avg['incorrect'],window_len=sm_win_size)
early_rate=smooth(day_avg['early'],window_len=sm_win_size)
correct_rate=smooth(day_avg['correct_val'],window_len=sm_win_size)

plt.plot(day_avg['trial_number'], early_rate,'.',label='early rate')
plt.plot(day_avg['trial_number'], correct_rate,'.',label='correct rate')
plt.plot(day_avg['trial_number'], incorrect_rate,'.',label='incorrect rate')
plt.legend()
plt.xlabel('Trial')
plt.ylabel('Rate')


session_avg[['early','incorrect','correct_val']].plot()
plt.xlabel('training session')