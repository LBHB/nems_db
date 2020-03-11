"""
Compare gain coefficients for on/off BF cells per file
Use stategain model results for this to get interpretable gain coefficients
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load MI data
dgain_307 = pd.read_csv('nems_lbhb/pupil_behavior_scripts/d_307_pup_fil_stategain.csv', index_col=0)
dgain_309 = pd.read_csv('nems_lbhb/pupil_behavior_scripts/d_309_pup_fil_stategain.csv', index_col=0)

# extract per cell pupil gain (this will get tossed below when doing per file stuff)
pgain_307 = dgain_307[(dgain_307.state_chan=='pupil') & (dgain_307.state_sig=='st.pup.fil')]
pgain_309 = dgain_309[(dgain_309.state_chan=='pupil') & (dgain_309.state_sig=='st.pup.fil')]

# set the cutoff for BF (in octaves from target)
cutoff = 0.5
snr_cutoff = 3

# load BF / SNR data
dBF_307 = pd.read_csv('nems_lbhb/pupil_behavior_scripts/d_307_tuning.csv', index_col=0)
dBF_307['cellid'] = dBF_307.index
dBF_309 = pd.read_csv('nems_lbhb/pupil_behavior_scripts/d_309_tuning.csv', index_col=0)
dBF_309['cellid'] = dBF_309.index

# load tar frequencies
dTF_307 = pd.read_csv('nems_lbhb/pupil_behavior_scripts/d_307_tar_freqs.csv', index_col=0)
dTF_309 = pd.read_csv('nems_lbhb/pupil_behavior_scripts/d_309_tar_freqs.csv', index_col=0)

# merge data frames
df_307 = dgain_307.merge(dTF_307, on=['cellid', 'state_chan_alt']).merge(dBF_307, on='cellid')
df_309 = dgain_309.merge(dTF_309, on=['cellid', 'state_chan_alt']).merge(dBF_309, on='cellid')
df_307['oct_diff'] = df_307['tar_freq'] / df_307['BF']
df_309['oct_diff'] = df_309['tar_freq'] / df_309['BF']

# keep only unique entries for the relevant plots below
df_307 = df_307[df_307.state_sig=='st.pup.fil']
df_309 = df_309[df_309.state_sig=='st.pup.fil']


# compare task gain for on / off BF cells
# ======================================
on_gain = df_307[(df_307.oct_diff <= cutoff) & (df_307.SNR > snr_cutoff)].groupby(by='cellid').mean()
off_gain = df_307[(df_307.oct_diff > cutoff) & (df_307.SNR > snr_cutoff)].groupby(by='cellid').mean()

task_sort_on_307 = on_gain.sort_values('g')
task_sort_off_307 = off_gain.sort_values('g')
pupil_sort_on_307 = pgain_307[pgain_307.cellid.isin(task_sort_on_307.index)].sort_values('g')
pupil_sort_off_307 = pgain_307[pgain_307.cellid.isin(task_sort_off_307.index)].sort_values('g')

f, ax = plt.subplots(2, 2, figsize=(16, 8), sharey=True)

ax[0, 0].set_title("pupil on BF")
ax[0, 0].bar(np.arange(0, pupil_sort_on_307.shape[0]), pupil_sort_on_307['g'])
ax[0, 0].set_ylabel('Pupil gain')
ax[1, 0].set_title("pupil off BF")
ax[1, 0].bar(np.arange(0, pupil_sort_off_307.shape[0]), pupil_sort_off_307['g'])
ax[1, 0].set_ylabel('Pupil gain')

ax[0, 1].set_title("task on BF")
ax[0, 1].bar(np.arange(0, task_sort_on_307.shape[0]), task_sort_on_307['g'])
ax[0, 1].set_ylabel('Task gain')
ax[1, 1].set_title("task off BF")
ax[1, 1].bar(np.arange(0, task_sort_off_307.shape[0]), task_sort_off_307['g'])
ax[1, 1].set_ylabel('Task gain')

f.tight_layout()
f.canvas.set_window_title('A1')

# IC
on_gain = df_309[(df_309.oct_diff <= cutoff) & (df_309.SNR > snr_cutoff)].groupby(by='cellid').mean()
off_gain = df_309[(df_309.oct_diff > cutoff) & (df_309.SNR > snr_cutoff)].groupby(by='cellid').mean()

task_sort_on_309 = on_gain.sort_values('g')
task_sort_off_309 = off_gain.sort_values('g')
pupil_sort_on_309 = pgain_309[pgain_309.cellid.isin(task_sort_on_309.index)].sort_values('g')
pupil_sort_off_309 = pgain_309[pgain_309.cellid.isin(task_sort_off_309.index)].sort_values('g')

f, ax = plt.subplots(2, 2, figsize=(16, 8), sharey=True)

ax[0, 0].set_title("pupil on BF")
ax[0, 0].bar(np.arange(0, pupil_sort_on_309.shape[0]), pupil_sort_on_309['g'])
ax[0, 0].set_ylabel('Pupil gain')
ax[1, 0].set_title("pupil off BF")
ax[1, 0].bar(np.arange(0, pupil_sort_off_309.shape[0]), pupil_sort_off_309['g'])
ax[1, 0].set_ylabel('Pupil gain')

ax[0, 1].set_title("task on BF")
ax[0, 1].bar(np.arange(0, task_sort_on_309.shape[0]), task_sort_on_309['g'])
ax[0, 1].set_ylabel('Task gain')
ax[1, 1].set_title("task off BF")
ax[1, 1].bar(np.arange(0, task_sort_off_309.shape[0]), task_sort_off_309['g'])
ax[1, 1].set_ylabel('Task gain')

f.tight_layout()
f.canvas.set_window_title('IC')

# compared matched cells
matched_307 = np.unique([c for c in df_307.cellid.unique() if (c in task_sort_on_307.index) & (c in task_sort_off_307.index)])
matched_309 = np.unique([c for c in df_309.cellid.unique() if (c in task_sort_on_309.index) & (c in task_sort_off_309.index)])

f, ax = plt.subplots(2, 2, figsize=(8, 8))

pupON_307 = pgain_307[pgain_307.cellid.isin(matched_307) & pgain_307.cellid.isin(task_sort_on_307.index)].groupby(by='cellid').mean()['g']
pupOFF_307 = pgain_307[pgain_307.cellid.isin(matched_307) & pgain_307.cellid.isin(task_sort_off_307.index)].groupby(by='cellid').mean()['g']
taskON_307 = df_307[df_307.cellid.isin(matched_307) & (df_307.oct_diff <= cutoff)].groupby(by='cellid').mean()['g']
taskOFF_307 = df_307[df_307.cellid.isin(matched_307) & (df_307.oct_diff > cutoff)].groupby(by='cellid').mean()['g']

ax[0, 0].set_title('Unique Pup. MI, A1')
ax[0, 0].scatter(pupON_307, pupOFF_307, color='k', edgecolor='white', s=50)
ax[0, 0].plot([-.5, .5], [-.5, .5], 'grey', linestyle='--')
ax[0, 0].axhline(0, linestyle='--', color='grey')
ax[0, 0].axvline(0, linestyle='--', color='grey')
ax[0, 0].set_xlabel('ON BF')
ax[0, 0].set_ylabel('OFF BF')


ax[0, 1].set_title('Unique Task MI, A1')
ax[0, 1].scatter(taskON_307, taskOFF_307, color='k', edgecolor='white', s=50)
ax[0, 1].plot([-.5, .5], [-.5, .5], 'grey', linestyle='--')
ax[0, 1].axhline(0, linestyle='--', color='grey')
ax[0, 1].axvline(0, linestyle='--', color='grey')
ax[0, 1].set_xlabel('ON BF')
ax[0, 1].set_ylabel('OFF BF')

pupON_309 = pgain_309[pgain_309.cellid.isin(matched_309) & pgain_309.cellid.isin(task_sort_on_309.index)].groupby(by='cellid').mean()['g']
pupOFF_309 = pgain_309[pgain_309.cellid.isin(matched_309) & pgain_309.cellid.isin(task_sort_off_309.index)].groupby(by='cellid').mean()['g']
taskON_309 = df_309[df_309.cellid.isin(matched_309) & (df_309.oct_diff <= cutoff)].groupby(by='cellid').mean()['g']
taskOFF_309 = df_309[df_309.cellid.isin(matched_309) & (df_309.oct_diff > cutoff)].groupby(by='cellid').mean()['g']

ax[1, 0].set_title('Unique Pup. MI, IC')
ax[1, 0].scatter(pupON_309, pupOFF_309, color='k', edgecolor='white', s=50)
ax[1, 0].plot([-.5, .5], [-.5, .5], 'grey', linestyle='--')
ax[1, 0].axhline(0, linestyle='--', color='grey')
ax[1, 0].axvline(0, linestyle='--', color='grey')
ax[1, 0].set_xlabel('ON BF')
ax[1, 0].set_ylabel('OFF BF')


ax[1, 1].set_title('Unique Task MI, IC')
ax[1, 1].scatter(taskON_309, taskOFF_309, color='k', edgecolor='white', s=50)
ax[1, 1].plot([-.5, .5], [-.5, .5], 'grey', linestyle='--')
ax[1, 1].axhline(0, linestyle='--', color='grey')
ax[1, 1].axvline(0, linestyle='--', color='grey')
ax[1, 1].set_xlabel('ON BF')
ax[1, 1].set_ylabel('OFF BF')

f.tight_layout()

plt.show()