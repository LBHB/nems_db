"""
No pupil data, so do this separately from 309 / 307
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

batches = [295]
# set the cutoff for BF (in octaves from target)
cutoff = 0.75
snr_cutoff = 0

for batch in batches:
    # load MI data
    dMI = pd.read_csv('nems_lbhb/pupil_behavior_scripts/d_{}_fil_stategain.csv'.format(batch), index_col=0)

    file_merge = dMI[dMI['state_sig']=='st.beh'][['cellid', 'state_chan_alt', 'MI']].merge(\
                    dMI[dMI['state_sig']=='st.beh0'][['cellid', 'state_chan_alt', 'MI']], on=['cellid', 'state_chan_alt'])
    file_merge['file_unique'] = file_merge['MI_x'] - file_merge['MI_y']
    dMI = dMI.merge(file_merge, on=['cellid'])

    # load BF / SNR data
    dBF = pd.read_csv('nems_lbhb/pupil_behavior_scripts/d_{}_tuning.csv'.format(batch), index_col=0)
    dBF['cellid'] = dBF.index

    # load tar frequencies
    dTF = pd.read_csv('nems_lbhb/pupil_behavior_scripts/d_{}_tar_freqs.csv'.format(batch), index_col=0)

    # merge results into single df for 307 and for 309
    df = dMI.merge(dTF, on=['cellid', 'state_chan_alt']).merge(dBF, on='cellid')
    df = df.drop(columns=['MI_x_x', 'MI_y_x', 'MI_x_y', 'MI_y_y'])

    # add column classifying octave sep. from target
    df['oct_diff'] = df['tar_freq'] / df['BF']

    # keep only unique entries for the relevant plots below
    df = df[df.state_sig=='st.pup.fil']

    # ============================ MAKE FIGURES =================================

    # 307, sorted histogram of unique file and unique pupil for on BF / off BF
    df_pup_sort_onBF = df[(df['oct_diff'] <= cutoff) & (df.SNR > snr_cutoff)].groupby(by='cellid').mean().sort_values('pupil_unique')
    df_pup_sort_offBF = df[(df['oct_diff'] > cutoff) & (df.SNR > snr_cutoff)].groupby(by='cellid').mean().sort_values('pupil_unique')

    df_task_sort_onBF = df[(df['oct_diff'] <= cutoff) & (df.SNR > snr_cutoff)].groupby(by='cellid').mean().sort_values('file_unique')
    df_task_sort_offBF = df[(df['oct_diff'] > cutoff) & (df.SNR > snr_cutoff)].groupby(by='cellid').mean().sort_values('file_unique')

    f, ax = plt.subplots(2, 2, figsize=(16, 8), sharey=True)

    ax[0, 0].set_title("pupil on BF")
    ax[0, 0].bar(np.arange(0, df_pup_sort_onBF.shape[0]), df_pup_sort_onBF['pupil_unique'])
    ax[0, 0].set_ylabel('MI pup. unique')
    ax[1, 0].set_title("pupil off BF")
    ax[1, 0].bar(np.arange(0, df_pup_sort_offBF.shape[0]), df_pup_sort_offBF['pupil_unique'])
    ax[1, 0].set_ylabel('MI pup. unique')

    ax[0, 1].set_title("task on BF")
    ax[0, 1].bar(np.arange(0, df_task_sort_onBF.shape[0]), df_task_sort_onBF['file_unique'])
    ax[0, 1].set_ylabel('MI beh. unique')
    ax[1, 1].set_title("task off BF")
    ax[1, 1].bar(np.arange(0, df_task_sort_offBF.shape[0]), df_task_sort_offBF['file_unique'])
    ax[1, 1].set_ylabel('MI beh. unique')

    f.tight_layout()
    f.canvas.set_window_title(batch)


    # ============================== find cells that were recorded for both on/off ======================
    on_cells = df[df['oct_diff'] <= cutoff].cellid 
    off_cells = df[df['oct_diff'] > cutoff].cellid 
    matched = np.unique([c for c in df.cellid.unique() if (c in on_cells.values) & (c in off_cells.values)])

    f, ax = plt.subplots(1, 2, figsize=(8, 4))

    pupON = df[df.cellid.isin(matched) & (df.oct_diff <= cutoff)].groupby(by='cellid').mean()['pupil_unique']
    pupOFF = df[df.cellid.isin(matched) & (df.oct_diff > cutoff)].groupby(by='cellid').mean()['pupil_unique']
    taskON = df[df.cellid.isin(matched) & (df.oct_diff <= cutoff)].groupby(by='cellid').mean()['file_unique']
    taskOFF = df[df.cellid.isin(matched) & (df.oct_diff > cutoff)].groupby(by='cellid').mean()['file_unique']

    ax[0].set_title('Unique Pup. MI, A1')
    ax[0].scatter(pupON, pupOFF, color='k', edgecolor='white', s=50)
    ax[0].plot([-.5, .5], [-.5, .5], 'grey', linestyle='--')
    ax[0].set_xlabel('ON BF')
    ax[0].set_ylabel('OFF BF')


    ax[1].set_title('Unique Task MI, A1')
    ax[1].scatter(taskON, taskOFF, color='k', edgecolor='white', s=50)
    ax[1].plot([-.5, .5], [-.5, .5], 'grey', linestyle='--')
    ax[1].set_xlabel('ON BF')
    ax[1].set_ylabel('OFF BF')

    f.tight_layout()

plt.show()