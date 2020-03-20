"""
No pupil data, so do this separately from 309 / 307
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

batch = 295
# set the cutoff for BF (in octaves from target)
cutoff = 0.4
snr_cutoff = 0

# load MI data
dMI = pd.read_csv('nems_lbhb/pupil_behavior_scripts/d_{}_fil_stategain.csv'.format(batch), index_col=0)
dMI['r'] = [np.float(r.strip('[]')) for r in dMI['r'].values]
dMI['r_se'] = [np.float(r.strip('[]')) for r in dMI['r_se'].values]
file_merge = dMI[dMI['state_sig']=='st.fil'][['cellid', 'state_chan_alt', 'MI', 'g', 'd', 'r', 'r_se']].merge(\
                dMI[dMI['state_sig']=='st.fil0'][['cellid', 'state_chan_alt', 'MI', 'g', 'd', 'r', 'r_se']], on=['cellid', 'state_chan_alt'])

file_merge['gain_unique'] = file_merge['g_x'] - file_merge['g_y']

file_merge['MI_unique'] = file_merge['MI_x'] - file_merge['MI_y']

file_merge['dc_unique'] = file_merge['d_x'] - file_merge['d_y']


sig_cells = file_merge[(file_merge['r_x'] - file_merge['r_y']) > (file_merge['r_se_x'] + file_merge['r_se_y'])].cellid.unique()

# load BF / SNR data
dBF = pd.read_csv('nems_lbhb/pupil_behavior_scripts/d_{}_tuning.csv'.format(batch), index_col=0)
dBF.index.name = 'cellid'

# load tar frequencies
dTF = pd.read_csv('nems_lbhb/pupil_behavior_scripts/d_{}_tar_freqs.csv'.format(batch), index_col=0)

# merge results into single df for 307 and for 309
df = file_merge.merge(dTF, on=['cellid', 'state_chan_alt'])
df.index = df.cellid
df = df.drop(columns=['cellid'])
df.index.name = 'cellid'
df = df.merge(dBF, on='cellid', how='left')

df = df.drop(columns=['MI_x', 'MI_y'])

# add column classifying octave sep. from target
df['oct_diff'] = df['tar_freq'] / df['BF']

#df['cellid'] = df.index

# ============================ MAKE FIGURES =================================

# histogram of MI / gain over all data
f, ax = plt.subplots(1, 3, figsize=(15, 5))

mi_bins = np.arange(-1, 1, 0.1)
ax[0].hist(df['MI_unique'], bins=mi_bins, color='white', edgecolor='k', rwidth=0.6, label='all cells')
ax[0].hist([df[df.index.isin(sig_cells) & (df.oct_diff>cutoff)]['MI_unique'],   
                df[df.index.isin(sig_cells) & (df.oct_diff<=cutoff)]['MI_unique']], bins=mi_bins, color=['red', 'blue'], 
                edgecolor='k', rwidth=0.6, label='OFF bf cells', histtype='barstacked')
ax[0].set_xlabel('MI', fontsize=10)
ax[0].set_ylabel('Number of neurons', fontsize=10)

gain_bins = np.arange(-2, 2, 0.2)
ax[1].hist(df['gain_unique'], bins=gain_bins, color='white', edgecolor='k', rwidth=0.6, label='all cells')
ax[1].hist([df[df.index.isin(sig_cells) & (df.oct_diff>cutoff)]['gain_unique'],   
                df[df.index.isin(sig_cells) & (df.oct_diff<=cutoff)]['gain_unique']], bins=gain_bins, color=['red', 'blue'], 
                edgecolor='k', rwidth=0.6, label='OFF bf cells', histtype='barstacked')
ax[1].set_xlabel('gain', fontsize=10)
ax[1].set_ylabel('Number of neurons', fontsize=10)


dc_bins = np.arange(-2, 2, 0.2)
ax[2].hist(df['dc_unique'], bins=dc_bins, color='white', edgecolor='k', rwidth=0.6, label='all cells')
ax[2].hist(df[df.index.isin(sig_cells)]['dc_unique'], bins=dc_bins, color='k', edgecolor='k', rwidth=0.6, label='sig behavior cells')
ax[2].hist([df[df.index.isin(sig_cells) & (df.oct_diff>cutoff)]['dc_unique'],   
                df[df.index.isin(sig_cells) & (df.oct_diff<=cutoff)]['dc_unique']], bins=dc_bins, color=['red', 'blue'], 
                edgecolor='k', rwidth=0.6, label=['OFF BF', 'ON BF'], histtype='barstacked')
ax[2].set_xlabel('DC offset', fontsize=10)
ax[2].set_ylabel('Number of neurons', fontsize=10)
ax[2].legend(frameon=False, fontsize=10)

f.tight_layout()


# ============================== find cells that were recorded for both on/off ======================
on_cells = df[df['oct_diff'] <= cutoff].index
off_cells = df[df['oct_diff'] > cutoff].index
matched = np.unique([c for c in df.index.unique() if (c in on_cells.values) & (c in off_cells.values)])

f, ax = plt.subplots(1, 3, figsize=(15, 5))

taskON = df[df.index.isin(matched) & (df.oct_diff <= cutoff)].groupby(by='cellid').mean()
taskOFF = df[df.index.isin(matched) & (df.oct_diff > cutoff)].groupby(by='cellid').mean()

ax[0].set_title('Unique Task MI, IC batch 295')
ax[0].scatter(taskON['MI_unique'], taskOFF['MI_unique'], color='grey', edgecolor='white', s=50)
ax[0].scatter(taskON[taskON.index.isin(sig_cells)]['MI_unique'], 
                taskOFF[taskON.index.isin(sig_cells)]['MI_unique'], color='k', edgecolor='white', s=50)
ax[0].plot([-1, 1], [-1, 1], 'grey', linestyle='--')
ax[0].axhline(0, linestyle='--', color='grey')
ax[0].axvline(0, linestyle='--', color='grey')
ax[0].set_xlim((-1, 1))
ax[0].set_ylim((-1, 1))
ax[0].set_xlabel('ON BF')
ax[0].set_ylabel('OFF BF')

ax[1].set_title('Gain, IC batch 295')
ax[1].scatter(taskON['gain_unique'], taskOFF['gain_unique'], color='grey', edgecolor='white', s=50)
ax[1].scatter(taskON[taskON.index.isin(sig_cells)]['gain_unique'], 
                taskOFF[taskON.index.isin(sig_cells)]['gain_unique'], color='k', edgecolor='white', s=50)
ax[1].plot([-1, 1], [-1, 1], 'grey', linestyle='--')
ax[1].axhline(0, linestyle='--', color='grey')
ax[1].axvline(0, linestyle='--', color='grey')
ax[1].set_xlim((-1, 1))
ax[1].set_ylim((-1, 1))
ax[1].set_xlabel('ON BF')
ax[1].set_ylabel('OFF BF')

ax[2].set_title('DC offset, IC batch 295')
ax[2].scatter(taskON['dc_unique'], taskOFF['dc_unique'], color='grey', edgecolor='white', s=50)
ax[2].scatter(taskON[taskON.index.isin(sig_cells)]['dc_unique'], 
                taskOFF[taskON.index.isin(sig_cells)]['dc_unique'], color='k', edgecolor='white', s=50)
ax[2].plot([-1, 1], [-1, 1], 'grey', linestyle='--')
ax[2].axhline(0, linestyle='--', color='grey')
ax[2].axvline(0, linestyle='--', color='grey')
ax[2].set_xlim((-1, 1))
ax[2].set_ylim((-1, 1))
ax[2].set_xlabel('ON BF')
ax[2].set_ylabel('OFF BF')

f.tight_layout()


plt.show()