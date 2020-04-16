"""
helper function to load and process state model results from the 
pupil behavior dump files e.g. d_307_sdexp_pup_fil.csv
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

def preprocess_stategain_dump(df_name, batch, full_model=None, p0=None, b0=None, shuf_model=None, octave_cutoff=0.5, r0_threshold=0,
                                 path='/auto/users/hellerc/code/nems_db/'):
    
    db_path = path
    cutoff = octave_cutoff
    
    # load model results data
    dMI = pd.read_csv(os.path.join(db_path, str(batch), df_name), index_col=0)
    try:
        dMI['r'] = [np.float(r.strip('[]')) for r in dMI['r'].values]
        dMI['r_se'] = [np.float(r.strip('[]')) for r in dMI['r_se'].values]
    except:
        pass

    # remove AMT cells
    dMI = dMI[~dMI.cellid.str.contains('AMT')]


    # get task model params / MI
    cols = ['cellid', 'state_chan_alt', 'MI', 'g', 'd', 'r', 'r_se']
    file_merge = dMI[(dMI['state_sig']==full_model) & (dMI['state_chan_alt'].str.contains('ACTIVE'))][cols].merge(\
                    dMI[(dMI['state_sig']==b0) & (dMI['state_chan_alt'].str.contains('ACTIVE'))][cols], \
                    on=['cellid', 'state_chan_alt'])

    file_merge['gain_task'] = file_merge['g_x']
    file_merge['MI_task'] = file_merge['MI_x'] - file_merge['MI_y']
    file_merge['dc_task'] = file_merge['d_x']

    # get sig task cells
    file_merge['sig_task'] = [True if ((file_merge.iloc[i]['r_x'] - file_merge.iloc[i]['r_y']) > 
                                            (file_merge.iloc[i]['r_se_x'] + file_merge.iloc[i]['r_se_y'])) else False for i in range(file_merge.shape[0])]

    # strip extraneous columns
    file_merge = file_merge.drop(columns=[c for c in file_merge.columns if ('_x' in c) | ('_y' in c)])

    # get pupil model params / MI
    pupil_merge = dMI[(dMI['state_sig']==full_model) & (dMI['state_chan_alt']=='pupil')][cols].merge(\
                    dMI[(dMI['state_sig']==p0) & (dMI['state_chan_alt']=='pupil')][cols], \
                    on=['cellid', 'state_chan_alt'])

    pupil_merge['gain_pupil'] = pupil_merge['g_x']
    pupil_merge['MI_pupil'] = pupil_merge['MI_x'] - pupil_merge['MI_y']
    pupil_merge['dc_pupil'] = pupil_merge['d_x']

    # get sig pupil cells
    pupil_merge['sig_pupil'] = [True if ((pupil_merge.iloc[i]['r_x'] - pupil_merge.iloc[i]['r_y']) > 
                                            (pupil_merge.iloc[i]['r_se_x'] + pupil_merge.iloc[i]['r_se_y'])) else False for i in range(pupil_merge.shape[0])]

    # strip extraneous columns
    pupil_merge = pupil_merge.drop(columns=[c for c in pupil_merge.columns if ('_x' in c) | ('_y' in c)])
    pupil_merge = pupil_merge.drop(columns=['state_chan_alt'])
    
    # =========================== get sig sensory cells ============================
    psth_cells = dMI[(dMI.state_sig==shuf_model) & (dMI.r > r0_threshold)].cellid.unique()

    # load BF / SNR data
    dBF = pd.read_csv(os.path.join(db_path, str(batch), 'd_tuning.csv'), index_col=0)
    dBF.index.name = 'cellid'

    # load tar frequencies
    dTF = pd.read_csv(os.path.join(db_path, str(batch), 'd_tar_freqs.csv').format(batch), index_col=0)

    # merge results into single df for 307 and for 309
    df = file_merge.merge(dTF, on=['cellid', 'state_chan_alt'])
    df = df.merge(pupil_merge, on=['cellid'])
    df.index = df.cellid
    df = df.drop(columns=['cellid'])

    df = df.merge(dBF, left_index=True, right_index=True)

    # add column classifying octave sep. from target
    df['oct_diff'] = abs(np.log2(df['tar_freq'] / df['BF']))

    # add column for on cells / off cells
    df['ON_BF'] = [True if df.iloc[i]['oct_diff']<=cutoff else False for i in range(df.shape[0])]
    df['OFF_BF'] = [True if df.iloc[i]['oct_diff']>cutoff else False for i in range(df.shape[0])]
    
    df['sig_psth'] = df.index.isin(psth_cells)
    
    return df
    
    
    
def preprocess_sdexp_dump(df_name, batch, full_model=None, p0=None, b0=None, shuf_model=None, octave_cutoff=0.5, r0_threshold=0,
                                 path='/auto/data/nems_db/results/'):
    db_path = path
    cutoff = octave_cutoff
    
    # load model results data
    dMI = pd.read_csv(os.path.join(db_path, str(batch), df_name), index_col=0)

    try:
        dMI['r'] = [np.float(r.strip('[]')) for r in dMI['r'].values]
        dMI['r_se'] = [np.float(r.strip('[]')) for r in dMI['r_se'].values]
    except:
        pass

    # remove AMT cells
    dMI = dMI[~dMI.cellid.str.contains('AMT')]
    
    # ========================== get sig overall state cells ==============================
    cols = ['cellid', 'state_chan_alt', 'r', 'r_se', 'isolation']
    state_merge = dMI[(dMI['state_sig']==full_model) & (dMI['state_chan_alt'].str.contains('ACTIVE|active', regex=True))][cols].merge(\
                    dMI[(dMI['state_sig']==shuf_model) & (dMI['state_chan_alt'].str.contains('ACTIVE|active', regex=True))][cols], \
                    on=['cellid', 'state_chan_alt'])    
    state_merge['sig_state'] = [True if ((state_merge.iloc[i]['r_x'] - state_merge.iloc[i]['r_y']) > 
                                           (state_merge.iloc[i]['r_se_x'] + state_merge.iloc[i]['r_se_y'])) else False for i in range(state_merge.shape[0])]
    # add SU column
    state_merge['SU'] = state_merge['isolation_x'] >= 95 

    # add overall and shuffled rpred
    state_merge['r_full'] = state_merge['r_x'].pow(2)
    state_merge['r_shuff'] = state_merge['r_y'].pow(2)
    
    # drop merge cols
    state_merge = state_merge.drop(columns=[c for c in state_merge.columns if ('_x' in c) | ('_y' in c)])
    
    # =================== get overall task model params / MI ===============================
    cols = ['cellid', 'state_chan_alt', 'MI', 'gain_mod', 'dc_mod', 'r', 'r_se']
    task_merge = dMI[(dMI['state_sig']==p0) & (dMI['state_chan_alt'].str.contains('ACTIVE|active', regex=True))][cols].merge(\
                    dMI[(dMI['state_sig']==shuf_model) & (dMI['state_chan_alt'].str.contains('ACTIVE|active', regex=True))][cols], \
                    on=['cellid', 'state_chan_alt'])

    task_merge['gain_task'] = task_merge['gain_mod_x'] - task_merge['gain_mod_y']
    task_merge['MI_task'] = task_merge['MI_x'] - task_merge['MI_y']
    task_merge['dc_task'] = task_merge['dc_mod_x'] - task_merge['dc_mod_y']

    task_merge['sig_task'] = [True if ((task_merge.iloc[i]['r_x'] - task_merge.iloc[i]['r_y']) > 
                                           (task_merge.iloc[i]['r_se_x'] + task_merge.iloc[i]['r_se_y'])) else False for i in range(task_merge.shape[0])]
    # add task rpred
    task_merge['r_task'] = task_merge['r_x'].pow(2)

    # strip extraneous columns
    task_merge = task_merge.drop(columns=[c for c in task_merge.columns if ('_x' in c) | ('_y' in c)])


    # =================== get unique task model params / MI ===============================
    cols = ['cellid', 'state_chan_alt', 'MI', 'gain_mod', 'dc_mod', 'r', 'r_se']
    utask_merge = dMI[(dMI['state_sig']==full_model) & (dMI['state_chan_alt'].str.contains('ACTIVE|active', regex=True))][cols].merge(\
                    dMI[(dMI['state_sig']==b0) & (dMI['state_chan_alt'].str.contains('ACTIVE|active', regex=True))][cols], \
                    on=['cellid', 'state_chan_alt'])

    utask_merge['gain_task_unique'] = utask_merge['gain_mod_x'] - utask_merge['gain_mod_y']
    utask_merge['MI_task_unique'] = utask_merge['MI_x'] - utask_merge['MI_y']
    utask_merge['dc_task_unique'] = utask_merge['dc_mod_x'] - utask_merge['dc_mod_y']

    utask_merge['sig_utask'] = [True if ((utask_merge.iloc[i]['r_x'] - utask_merge.iloc[i]['r_y']) > 
                                           (utask_merge.iloc[i]['r_se_x'] + utask_merge.iloc[i]['r_se_y'])) else False for i in range(utask_merge.shape[0])]

    # add unique task rpred
    utask_merge['r_task_unique'] = utask_merge['r_x'].pow(2) - utask_merge['r_y'].pow(2)

    # strip extraneous columns
    utask_merge = utask_merge.drop(columns=[c for c in utask_merge.columns if ('_x' in c) | ('_y' in c)])
        
    # ======================= get overal pupil model params / MI =========================
    pupil_merge = dMI[(dMI['state_sig']==b0) & (dMI['state_chan_alt']=='pupil')][cols].merge(\
                    dMI[(dMI['state_sig']==shuf_model) & (dMI['state_chan_alt']=='pupil')][cols], \
                    on=['cellid', 'state_chan_alt'])

    pupil_merge['gain_pupil'] = pupil_merge['gain_mod_x'] - pupil_merge['gain_mod_y']
    pupil_merge['MI_pupil'] = pupil_merge['MI_x'] - pupil_merge['MI_y']
    pupil_merge['dc_pupil'] = pupil_merge['dc_mod_x'] - pupil_merge['dc_mod_y']

    pupil_merge['sig_pupil'] = [True if ((pupil_merge.iloc[i]['r_x'] - pupil_merge.iloc[i]['r_y']) > 
                                            (pupil_merge.iloc[i]['r_se_x'] + pupil_merge.iloc[i]['r_se_y'])) else False for i in range(pupil_merge.shape[0])]

    # add pupil r pred
    pupil_merge['r_pupil'] = pupil_merge['r_x'].pow(2)

    # strip extraneous columns
    pupil_merge = pupil_merge.drop(columns=[c for c in pupil_merge.columns if ('_x' in c) | ('_y' in c)])
    pupil_merge = pupil_merge.drop(columns=['state_chan_alt'])


    # ======================= get unique pupil model params / MI =========================
    upupil_merge = dMI[(dMI['state_sig']==full_model) & (dMI['state_chan_alt']=='pupil')][cols].merge(\
                    dMI[(dMI['state_sig']==p0) & (dMI['state_chan_alt']=='pupil')][cols], \
                    on=['cellid', 'state_chan_alt'])

    upupil_merge['gain_pupil_unique'] = upupil_merge['gain_mod_x'] - upupil_merge['gain_mod_y']
    upupil_merge['MI_pupil_unique'] = upupil_merge['MI_x'] - upupil_merge['MI_y']
    upupil_merge['dc_pupil_unique'] = upupil_merge['dc_mod_x'] - upupil_merge['dc_mod_y']

    upupil_merge['sig_upupil'] = [True if ((upupil_merge.iloc[i]['r_x'] - upupil_merge.iloc[i]['r_y']) > 
                                            (upupil_merge.iloc[i]['r_se_x'] + upupil_merge.iloc[i]['r_se_y'])) else False for i in range(upupil_merge.shape[0])]

    # add unique pupil rpred
    upupil_merge['r_pupil_unique'] = upupil_merge['r_x'].pow(2) - upupil_merge['r_y'].pow(2)
    
    # strip extraneous columns
    upupil_merge = upupil_merge.drop(columns=[c for c in upupil_merge.columns if ('_x' in c) | ('_y' in c)])
    upupil_merge = upupil_merge.drop(columns=['state_chan_alt'])

    # =========================== get sig sensory cells ============================
    psth_cells = dMI[(dMI.state_sig==shuf_model) & (dMI.r > r0_threshold)].cellid.unique()

    # load BF / SNR data
    dBF = pd.read_csv(os.path.join(db_path, str(batch), 'd_tuning.csv'), index_col=0)
    dBF.index.name = 'cellid'

    # load tar frequencies
    dTF = pd.read_csv(os.path.join(db_path, str(batch), 'd_tar_freqs.csv'), index_col=0)

    # merge results into single df
    if 'beh' in full_model:
        # "target freq is meaningless because could change between files"
        df = task_merge.merge(utask_merge, on=['cellid', 'state_chan_alt'])
        df = df.merge(state_merge, on=['cellid', 'state_chan_alt'])
        df = df.merge(pupil_merge, on=['cellid'])
        df = df.merge(upupil_merge, on=['cellid'])

        df.index = df.cellid
        df = df.drop(columns=['cellid'])

        df = df.merge(dBF, left_index=True, right_index=True)

        df['oct_diff'] = np.nan
        df['ON_BF'] = True
        df['OFF_BF'] = True


    else:
        df = task_merge.merge(dTF, on=['cellid', 'state_chan_alt'])
        df = df.merge(utask_merge, on=['cellid', 'state_chan_alt'])
        df = df.merge(state_merge, on=['cellid', 'state_chan_alt'])
        df = df.merge(pupil_merge, on=['cellid'])
        df = df.merge(upupil_merge, on=['cellid'])
        df.index = df.cellid
        df = df.drop(columns=['cellid'])

        df = df.merge(dBF, left_index=True, right_index=True)

        # add column classifying octave sep. from target
        df['oct_diff'] = abs(np.log2(df['tar_freq'] / df['BF']))

        # add column for on cells / off cells
        df['ON_BF'] = [True if df.iloc[i]['oct_diff']<=cutoff else False for i in range(df.shape[0])]
        df['OFF_BF'] = [True if df.iloc[i]['oct_diff']>cutoff else False for i in range(df.shape[0])]

    df['sig_psth'] = df.index.isin(psth_cells)
    
    return df


def stripplot_df(df, fix_ylims=False, group_files=True):

    if group_files:
        data = df.groupby(by=['cellid', 'ON_BF']).mean().copy()
        data['ON_BF'] = data.index.get_level_values('ON_BF')
    else:
        data = df.copy()

    f, ax = plt.subplots(2, 3, figsize=(12, 8))

    # BEHAVIOR results
    # MI
    sns.stripplot(x='sig_task', y='MI_task', hue='ON_BF', data=data, dodge=True, ax=ax[0, 0])
    ax[0, 0].axhline(0, linestyle='--', color='k')
    pval = np.round(ss.ranksums(data[data['sig_task'] & data['ON_BF']]['MI_task'], data[data['sig_task'] & data['OFF_BF']]['MI_task']).pvalue, 3)
    on_median = np.round(data[data['sig_task'] & data['ON_BF']]['MI_task'].median(), 3)
    off_median = np.round(data[data['sig_task'] & data['OFF_BF']]['MI_task'].median(), 3)
    ax[0, 0].set_title('sig ON vs. OFF, pval: {0} \n'
                        'ON median: {1}, OFF median: {2}'.format(pval, on_median, off_median))

    # Gain
    sns.stripplot(x='sig_task', y='gain_task', hue='ON_BF', data=data, dodge=True, ax=ax[0, 1])
    ax[0, 1].axhline(0, linestyle='--', color='k')
    pval = np.round(ss.ranksums(data[data['sig_task'] & data['ON_BF']]['gain_task'], data[data['sig_task'] & data['OFF_BF']]['gain_task']).pvalue, 3)
    on_median = np.round(data[data['sig_task'] & data['ON_BF']]['gain_task'].median(), 3)
    off_median = np.round(data[data['sig_task'] & data['OFF_BF']]['gain_task'].median(), 3)
    ax[0, 1].set_title('sig ON vs. OFF, pval: {0} \n'
                        'ON median: {1}, OFF median: {2}'.format(pval, on_median, off_median))

    # DC
    sns.stripplot(x='sig_task', y='dc_task', hue='ON_BF', data=data, dodge=True, ax=ax[0, 2])
    ax[0, 2].axhline(0, linestyle='--', color='k')
    pval = np.round(ss.ranksums(data[data['sig_task'] & data['ON_BF']]['dc_task'], data[data['sig_task'] & data['OFF_BF']]['dc_task']).pvalue, 3)
    on_median = np.round(data[data['sig_task'] & data['ON_BF']]['dc_task'].median(), 3)
    off_median = np.round(data[data['sig_task'] & data['OFF_BF']]['dc_task'].median(), 3)
    ax[0, 2].set_title('sig ON vs. OFF, pval: {0} \n'
                        'ON median: {1}, OFF median: {2}'.format(pval, on_median, off_median))

    # PUPIL results
    # MI
    sns.stripplot(x='sig_pupil', y='MI_pupil', hue='ON_BF', data=data, dodge=True, ax=ax[1, 0])
    ax[1, 0].axhline(0, linestyle='--', color='k')
    pval = np.round(ss.ranksums(data[data['sig_pupil'] & data['ON_BF']]['MI_pupil'], data[data['sig_pupil'] & data['OFF_BF']]['MI_pupil']).pvalue, 3)
    on_median = np.round(data[data['sig_pupil'] & data['ON_BF']]['MI_pupil'].median(), 3)
    off_median = np.round(data[data['sig_pupil'] & data['OFF_BF']]['MI_pupil'].median(), 3)
    ax[1, 0].set_title('sig ON vs. OFF, pval: {0} \n'
                        'ON median: {1}, OFF median: {2}'.format(pval, on_median, off_median))

    # Gain
    sns.stripplot(x='sig_pupil', y='gain_pupil', hue='ON_BF', data=data, dodge=True, ax=ax[1, 1])
    ax[1, 1].axhline(0, linestyle='--', color='k')
    pval = np.round(ss.ranksums(data[data['sig_pupil'] & data['ON_BF']]['gain_pupil'], data[data['sig_pupil'] & data['OFF_BF']]['gain_pupil']).pvalue, 3)
    on_median = np.round(data[data['sig_pupil'] & data['ON_BF']]['gain_pupil'].median(), 3)
    off_median = np.round(data[data['sig_pupil'] & data['OFF_BF']]['gain_pupil'].median(), 3)
    ax[1, 1].set_title('sig ON vs. OFF, pval: {0} \n'
                        'ON median: {1}, OFF median: {2}'.format(pval, on_median, off_median))

    # DC
    sns.stripplot(x='sig_pupil', y='dc_pupil', hue='ON_BF', data=data, dodge=True, ax=ax[1, 2])
    ax[1, 2].axhline(0, linestyle='--', color='k')
    pval = np.round(ss.ranksums(data[data['sig_pupil'] & data['ON_BF']]['dc_pupil'], data[data['sig_pupil'] & data['OFF_BF']]['dc_pupil']).pvalue, 3)
    on_median = np.round(data[data['sig_pupil'] & data['ON_BF']]['dc_pupil'].median(), 3)
    off_median = np.round(data[data['sig_pupil'] & data['OFF_BF']]['dc_pupil'].median(), 3)
    ax[1, 2].set_title('sig ON vs. OFF, pval: {0} \n'
                        'ON median: {1}, OFF median: {2}'.format(pval, on_median, off_median))

    if fix_ylims:
        for a in ax.flatten():
            a.set_ylim((-1.5, 1.5))
            
    f.tight_layout()

    return f, ax


def compare_models(df1, df2, xlab=None, ylab=None):
    f, ax = plt.subplots(2, 3, figsize=(12, 8))

    ax[0, 0].set_title('MI task')
    ax[0, 0].plot([-1, 1], [-1, 1], 'k--')
    ax[0, 0].scatter(df1['MI_task'], df2['MI_task'], s=50, color='grey', edgecolor='white')
    ax[0, 0].set_xlabel(xlab)
    ax[0, 0].set_ylabel(ylab)
    ax[0, 0].axis('square')

    ax[0, 1].set_title('gain task')
    ax[0, 1].plot([-1, 1], [-1, 1], 'k--')
    ax[0, 1].scatter(df1['gain_task'], df2['gain_task'], s=50, color='grey', edgecolor='white')
    ax[0, 1].set_xlabel(xlab)
    ax[0, 1].set_ylabel(ylab)
    ax[0, 1].axis('square')

    ax[0, 2].set_title('DC task')
    ax[0, 2].plot([-1, 1], [-1, 1], 'k--')
    ax[0, 2].scatter(df1['dc_task'], df2['dc_task'], s=50, color='grey', edgecolor='white')
    ax[0, 2].set_xlabel(xlab)
    ax[0, 2].set_ylabel(ylab)
    ax[0, 2].axis('square')

    ax[1, 0].set_title('MI pupil')
    ax[1, 0].plot([-1, 1], [-1, 1], 'k--')
    ax[1, 0].scatter(df1['MI_pupil'], df2['MI_pupil'], s=50, color='grey', edgecolor='white')
    ax[1, 0].set_xlabel(xlab)
    ax[1, 0].set_ylabel(ylab)
    ax[1, 0].axis('square')

    ax[1, 1].set_title('gain pupil')
    ax[1, 1].plot([-1, 1], [-1, 1], 'k--')
    ax[1, 1].scatter(df1['gain_pupil'], df2['gain_pupil'], s=50, color='grey', edgecolor='white')
    ax[1, 1].set_xlabel(xlab)
    ax[1, 1].set_ylabel(ylab)
    ax[1, 1].axis('square')

    ax[1, 2].set_title('DC pupil')
    ax[1, 2].plot([-1, 1], [-1, 1], 'k--')
    ax[1, 2].scatter(df1['dc_pupil'], df2['dc_pupil'], s=50, color='grey', edgecolor='white')
    ax[1, 2].set_xlabel(xlab)
    ax[1, 2].set_ylabel(ylab)
    ax[1, 2].axis('square')

    f.tight_layout()

    return f, ax


