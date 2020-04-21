"""
helper function to load and process state model results from the 
pupil behavior dump files e.g. d_307_sdexp_pup_fil.csv
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as ss
import scipy.stats as st
from itertools import product

import nems_lbhb.stateplots as stateplots
import nems.plots.api as nplt
import common


def preprocess_stategain_dump(df_name, batch, full_model=None, p0=None, b0=None, shuf_model=None, octave_cutoff=0.5, r0_threshold=0,
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

    # ======================================= task unique results ================================================
    cols = ['cellid', 'state_chan_alt', 'MI', 'g', 'd', 'r', 'r_se']
    utask_merge = dMI[(dMI['state_sig']==full_model) & (dMI['state_chan_alt'].str.contains('ACTIVE|active', regex=True))][cols].merge(\
                    dMI[(dMI['state_sig']==b0) & (dMI['state_chan_alt'].str.contains('ACTIVE|active', regex=True))][cols], \
                    on=['cellid', 'state_chan_alt'])

    utask_merge['gain_task_unique'] = utask_merge['g_x']
    utask_merge['MI_task_unique'] = utask_merge['MI_x'] - utask_merge['MI_y']
    utask_merge['dc_task_unique'] = utask_merge['d_x']

    # get sig task cells
    utask_merge['sig_utask'] = [True if ((utask_merge.iloc[i]['r_x'] - utask_merge.iloc[i]['r_y']) > 
                                            (utask_merge.iloc[i]['r_se_x'] + utask_merge.iloc[i]['r_se_y'])) else False for i in range(utask_merge.shape[0])]

    # add unique task rpred
    utask_merge['r_task_unique'] = utask_merge['r_x'].pow(2) - utask_merge['r_y'].pow(2)

    # strip extraneous columns
    utask_merge = utask_merge.drop(columns=[c for c in utask_merge.columns if ('_x' in c) | ('_y' in c)])

    # ======================================= task overall results ================================================
    if p0 is not None:
        cols = ['cellid', 'state_chan_alt', 'MI', 'g', 'd', 'r', 'r_se']
        task_merge = dMI[(dMI['state_sig']==p0) & (dMI['state_chan_alt'].str.contains('ACTIVE|active', regex=True))][cols].merge(\
                        dMI[(dMI['state_sig']==shuf_model) & (dMI['state_chan_alt'].str.contains('ACTIVE|active', regex=True))][cols], \
                        on=['cellid', 'state_chan_alt'])

        task_merge['gain_task'] = task_merge['g_x']
        task_merge['MI_task'] = task_merge['MI_x'] - task_merge['MI_y']
        task_merge['dc_task'] = task_merge['d_x']

        # get sig task cells
        task_merge['sig_task'] = [True if ((task_merge.iloc[i]['r_x'] - task_merge.iloc[i]['r_y']) > 
                                                (task_merge.iloc[i]['r_se_x'] + task_merge.iloc[i]['r_se_y'])) else False for i in range(task_merge.shape[0])]

        task_merge['r_task'] = task_merge['r_x'].pow(2)

        # strip extraneous columns
        task_merge = task_merge.drop(columns=[c for c in task_merge.columns if ('_x' in c) | ('_y' in c)])
    else:
        # if no pupil in model, task_unique == task overall
        task_merge = utask_merge.copy()
        task_merge.columns = [c.replace('_unique', '') for c in task_merge.columns]
        task_merge.rename({'sig_utask': 'sig_task'})


    # ====================================== get pupil results ===================================================
    if p0 is not None:
        # =============================== pupil overall ==============================================
        pupil_merge = dMI[(dMI['state_sig']==full_model) & (dMI['state_chan_alt']=='pupil')][cols].merge(\
                        dMI[(dMI['state_sig']==p0) & (dMI['state_chan_alt']=='pupil')][cols], \
                        on=['cellid', 'state_chan_alt'])

        pupil_merge['gain_pupil'] = pupil_merge['g_x']
        pupil_merge['MI_pupil'] = pupil_merge['MI_x'] - pupil_merge['MI_y']
        pupil_merge['dc_pupil'] = pupil_merge['d_x']

        # get sig pupil cells
        pupil_merge['sig_pupil'] = [True if ((pupil_merge.iloc[i]['r_x'] - pupil_merge.iloc[i]['r_y']) > 
                                                (pupil_merge.iloc[i]['r_se_x'] + pupil_merge.iloc[i]['r_se_y'])) else False for i in range(pupil_merge.shape[0])]

        # add pupil r pred
        pupil_merge['r_pupil'] = pupil_merge['r_x'].pow(2)

        # strip extraneous columns
        pupil_merge = pupil_merge.drop(columns=[c for c in pupil_merge.columns if ('_x' in c) | ('_y' in c)])
        pupil_merge = pupil_merge.drop(columns=['state_chan_alt'])

        # =============================== pupil unique ==============================================
        upupil_merge = dMI[(dMI['state_sig']==full_model) & (dMI['state_chan_alt']=='pupil')][cols].merge(\
                        dMI[(dMI['state_sig']==p0) & (dMI['state_chan_alt']=='pupil')][cols], \
                        on=['cellid', 'state_chan_alt'])

        upupil_merge['gain_pupil_unique'] = upupil_merge['g_x']
        upupil_merge['MI_pupil_unique'] = upupil_merge['MI_x'] - upupil_merge['MI_y']
        upupil_merge['dc_pupil_unique'] = upupil_merge['d_x']

        # get sig pupil cells
        upupil_merge['sig_upupil'] = [True if ((upupil_merge.iloc[i]['r_x'] - upupil_merge.iloc[i]['r_y']) > 
                                                (upupil_merge.iloc[i]['r_se_x'] + upupil_merge.iloc[i]['r_se_y'])) else False for i in range(upupil_merge.shape[0])]
        
        # add unique pupil rpred
        upupil_merge['r_pupil_unique'] = upupil_merge['r_x'].pow(2) - upupil_merge['r_y'].pow(2)

        # strip extraneous columns
        upupil_merge = upupil_merge.drop(columns=[c for c in upupil_merge.columns if ('_x' in c) | ('_y' in c)])
        upupil_merge = upupil_merge.drop(columns=['state_chan_alt'])
    
    # =========================== get sig sensory cells ============================
    psth_cells = dMI[(dMI.state_sig==shuf_model) & (dMI.r > r0_threshold)].cellid.unique()

    # merge results into single df
    if 'beh' in full_model:
        # "target freq is meaningless because could change between files"
        df = task_merge.merge(utask_merge, on=['cellid', 'state_chan_alt'])
        df = df.merge(state_merge, on=['cellid', 'state_chan_alt'])
        if p0 is not None:
            df = df.merge(pupil_merge, on=['cellid'])
            df = df.merge(upupil_merge, on=['cellid'])

        df.index = df.cellid
        df = df.drop(columns=['cellid'])

        df = df.merge(dBF, left_index=True, right_index=True)

        df['oct_diff'] = np.nan
        df['ON_BF'] = True
        df['OFF_BF'] = True


    else:
        try:
            # load BF / SNR data
            dBF = pd.read_csv(os.path.join(db_path, str(batch), 'd_tuning.csv'), index_col=0)
            dBF.index.name = 'cellid'

            # load tar frequencies
            dTF = pd.read_csv(os.path.join(db_path, str(batch), 'd_tar_freqs.csv'), index_col=0)

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

        except FileNotFoundError:
            print('WARNING. Did not find tuning file(s) for this batch')
            df = task_merge.merge(utask_merge, on=['cellid', 'state_chan_alt'])
            if p0 is not None:
                df = df.merge(pupil_merge, on=['cellid'])
                df = df.merge(upupil_merge, on=['cellid'])

            
            df['oct_diff'] = np.nan
            df['ON_BF'] = True
            df['OFF_BF'] = True

            df.index = df.cellid
            df = df.drop(columns=['cellid'])

    df['sig_psth'] = df.index.isin(psth_cells)
    
    return df    
    
    
def preprocess_sdexp_dump(df_name, batch, full_model=None, p0=None, b0=None, shuf_model=None, octave_cutoff=0.5, r0_threshold=0,
                                 pas_model=False, path='/auto/data/nems_db/results/'):
    db_path = path
    cutoff = octave_cutoff
    
    # load model results data
    dMI = pd.read_csv(os.path.join(db_path, str(batch), df_name), index_col=0)

    if pas_model:
        task_regex = 'PASSIVE_1'
    else:
        task_regex = 'ACTIVE|active'

    try:
        dMI['r'] = [np.float(r.strip('[]')) for r in dMI['r'].values]
        dMI['r_se'] = [np.float(r.strip('[]')) for r in dMI['r_se'].values]
    except:
        pass

    # remove AMT cells
    dMI = dMI[~dMI.cellid.str.contains('AMT')]
    # ========================== get sig overall state cells ==============================
    cols = ['cellid', 'state_chan_alt', 'r', 'r_se', 'isolation']
    state_merge = dMI[(dMI['state_sig']==full_model) & (dMI['state_chan_alt'].str.contains(task_regex, regex=True))][cols].merge(\
                    dMI[(dMI['state_sig']==shuf_model) & (dMI['state_chan_alt'].str.contains(task_regex, regex=True))][cols], \
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

    # =================== get unique task model params / MI ===============================
    cols = ['cellid', 'state_chan_alt', 'MI', 'gain_mod', 'dc_mod', 'r', 'r_se']
    utask_merge = dMI[(dMI['state_sig']==full_model) & (dMI['state_chan_alt'].str.contains(task_regex, regex=True))][cols].merge(\
                    dMI[(dMI['state_sig']==b0) & (dMI['state_chan_alt'].str.contains(task_regex, regex=True))][cols], \
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
    
    # =================== get overall task model params / MI ===============================
    if p0 is not None:
        cols = ['cellid', 'state_chan_alt', 'MI', 'gain_mod', 'dc_mod', 'r', 'r_se']
        task_merge = dMI[(dMI['state_sig']==p0) & (dMI['state_chan_alt'].str.contains(task_regex, regex=True))][cols].merge(\
                        dMI[(dMI['state_sig']==shuf_model) & (dMI['state_chan_alt'].str.contains(task_regex, regex=True))][cols], \
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
    else:
        # if no pupil in model, task_unique == task overall
        task_merge = utask_merge.copy()
        task_merge.columns = [c.replace('_unique', '') for c in task_merge.columns]
        task_merge.rename({'sig_utask': 'sig_task'})
        
    # ======================= get overal pupil model params / MI =========================
    if p0 is not None:
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
    
    # merge results into single df
    if ('beh' in full_model) | pas_model:
        # "target freq is meaningless because could change between files"
        df = task_merge.merge(utask_merge, on=['cellid', 'state_chan_alt'])
        df = df.merge(state_merge, on=['cellid', 'state_chan_alt'])
        if p0 is not None:
            df = df.merge(pupil_merge, on=['cellid'])
            df = df.merge(upupil_merge, on=['cellid'])

        df.index = df.cellid
        df = df.drop(columns=['cellid'])

        # df = df.merge(dBF, left_index=True, right_index=True)

        df['oct_diff'] = np.nan
        df['ON_BF'] = True
        df['OFF_BF'] = True


    else:
        try:
            # load BF / SNR data
            dBF = pd.read_csv(os.path.join(db_path, str(batch), 'd_tuning.csv'), index_col=0)
            dBF.index.name = 'cellid'

            # load tar frequencies
            dTF = pd.read_csv(os.path.join(db_path, str(batch), 'd_tar_freqs.csv'), index_col=0)

            df = task_merge.merge(dTF, on=['cellid', 'state_chan_alt'])
            df = df.merge(utask_merge, on=['cellid', 'state_chan_alt'])
            df = df.merge(state_merge, on=['cellid', 'state_chan_alt'])
            try: 
                difficulty = pd.read_csv(os.path.join(db_path, str(batch), 'd_difficulty.csv'), index_col=0)
                df = df.merge(difficulty, on=['cellid', 'state_chan_alt'])
            except:
                pass
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

        except FileNotFoundError:
            print('WARNING. Did not find tuning file(s) for this batch')
            df = task_merge.merge(utask_merge, on=['cellid', 'state_chan_alt'])
            if p0 is not None:
                df = df.merge(pupil_merge, on=['cellid'])
                df = df.merge(upupil_merge, on=['cellid'])

            
            df['oct_diff'] = np.nan
            df['ON_BF'] = True
            df['OFF_BF'] = True

            df.index = df.cellid
            df = df.drop(columns=['cellid'])

    df['sig_psth'] = df.index.isin(psth_cells)
    
    return df


def stripplot_df(df, fix_ylims=False, hue='ON_BF', group_files=True):

    if group_files:
        data = df.groupby(by=['cellid', 'ON_BF']).mean().copy()
        data['ON_BF'] = data.index.get_level_values('ON_BF')
        dtypes = {'ON_BF': bool, 
                  'OFF_BF': bool,
                  'sig_upupil': bool, 
                  'sig_utask': bool,
                  'difficulty': bool}
        data = data.astype(dtypes)
    else:
        data = df.copy()

    f, ax = plt.subplots(2, 3, figsize=(12, 8))

    # BEHAVIOR results
    # MI
    sns.stripplot(x='sig_utask', y='MI_task', hue=hue, data=data, dodge=True, ax=ax[0, 0])
    ax[0, 0].axhline(0, linestyle='--', color='k')
    pval = np.round(st.ranksums(data[data['sig_utask'] & data['ON_BF']]['MI_task'], data[data['sig_utask'] & data['OFF_BF']]['MI_task']).pvalue, 3)
    on_median = np.round(data[data['sig_utask'] & data['ON_BF']]['MI_task'].median(), 3)
    off_median = np.round(data[data['sig_utask'] & data['OFF_BF']]['MI_task'].median(), 3)
    ax[0, 0].set_title('sig ON vs. OFF, pval: {0} \n'
                        'ON median: {1}, OFF median: {2}'.format(pval, on_median, off_median))

    # Gain
    sns.stripplot(x='sig_utask', y='gain_task', hue=hue, data=data, dodge=True, ax=ax[0, 1])
    ax[0, 1].axhline(0, linestyle='--', color='k')
    pval = np.round(st.ranksums(data[data['sig_utask'] & data['ON_BF']]['gain_task'], data[data['sig_utask'] & data['OFF_BF']]['gain_task']).pvalue, 3)
    on_median = np.round(data[data['sig_utask'] & data['ON_BF']]['gain_task'].median(), 3)
    off_median = np.round(data[data['sig_utask'] & data['OFF_BF']]['gain_task'].median(), 3)
    ax[0, 1].set_title('sig ON vs. OFF, pval: {0} \n'
                        'ON median: {1}, OFF median: {2}'.format(pval, on_median, off_median))

    # DC
    sns.stripplot(x='sig_utask', y='dc_task', hue=hue, data=data, dodge=True, ax=ax[0, 2])
    ax[0, 2].axhline(0, linestyle='--', color='k')
    pval = np.round(st.ranksums(data[data['sig_utask'] & data['ON_BF']]['dc_task'], data[data['sig_utask'] & data['OFF_BF']]['dc_task']).pvalue, 3)
    on_median = np.round(data[data['sig_utask'] & data['ON_BF']]['dc_task'].median(), 3)
    off_median = np.round(data[data['sig_utask'] & data['OFF_BF']]['dc_task'].median(), 3)
    ax[0, 2].set_title('sig ON vs. OFF, pval: {0} \n'
                        'ON median: {1}, OFF median: {2}'.format(pval, on_median, off_median))

    # PUPIL results
    # MI
    sns.stripplot(x='sig_upupil', y='MI_pupil', hue=hue, data=data, dodge=True, ax=ax[1, 0])
    ax[1, 0].axhline(0, linestyle='--', color='k')
    pval = np.round(st.ranksums(data[data['sig_upupil'] & data['ON_BF']]['MI_pupil'], data[data['sig_upupil'] & data['OFF_BF']]['MI_pupil']).pvalue, 3)
    on_median = np.round(data[data['sig_upupil'] & data['ON_BF']]['MI_pupil'].median(), 3)
    off_median = np.round(data[data['sig_upupil'] & data['OFF_BF']]['MI_pupil'].median(), 3)
    ax[1, 0].set_title('sig ON vs. OFF, pval: {0} \n'
                        'ON median: {1}, OFF median: {2}'.format(pval, on_median, off_median))

    # Gain
    sns.stripplot(x='sig_upupil', y='gain_pupil', hue=hue, data=data, dodge=True, ax=ax[1, 1])
    ax[1, 1].axhline(0, linestyle='--', color='k')
    pval = np.round(st.ranksums(data[data['sig_upupil'] & data['ON_BF']]['gain_pupil'], data[data['sig_upupil'] & data['OFF_BF']]['gain_pupil']).pvalue, 3)
    on_median = np.round(data[data['sig_upupil'] & data['ON_BF']]['gain_pupil'].median(), 3)
    off_median = np.round(data[data['sig_upupil'] & data['OFF_BF']]['gain_pupil'].median(), 3)
    ax[1, 1].set_title('sig ON vs. OFF, pval: {0} \n'
                        'ON median: {1}, OFF median: {2}'.format(pval, on_median, off_median))

    # DC
    sns.stripplot(x='sig_upupil', y='dc_pupil', hue=hue, data=data, dodge=True, ax=ax[1, 2])
    ax[1, 2].axhline(0, linestyle='--', color='k')
    pval = np.round(st.ranksums(data[data['sig_upupil'] & data['ON_BF']]['dc_pupil'], data[data['sig_upupil'] & data['OFF_BF']]['dc_pupil']).pvalue, 3)
    on_median = np.round(data[data['sig_upupil'] & data['ON_BF']]['dc_pupil'].median(), 3)
    off_median = np.round(data[data['sig_upupil'] & data['OFF_BF']]['dc_pupil'].median(), 3)
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


def aud_vs_state(df, nb=5, title=None, state_list=None, colors=['r','g','b','k']):
    """
    d = dataframe output by get_model_results_per_state_model()
    nb = number of bins
    """
    if state_list is None:
        state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
    
    f = plt.figure(figsize=(4,6))

    dr = df.copy()

    if len(state_list)==4:
        dr['bp_common'] = dr['r_full'] - df['r_task_unique'] - df['r_pupil_unique'] - dr['r_shuff']
        dr = dr.sort_values('r_shuff')
        mfull = dr[['r_shuff', 'r_full', 'bp_common', 'r_task_unique', 'r_pupil_unique', 'sig_state']].values

    elif len(state_list)==2:
        dr['bp_common'] = dr[state_list[1]]**2 - dr[state_list[0]]**2
        dr['b_unique'] = dr['bp_common']*0
        dr['p_unique'] = dr['bp_common']*0

        dr['bp_full'] = dr['b_unique'] + dr['p_unique'] + dr['bp_common']
        dr['null']=dr[state_list[0]]**2 * np.sign(dr[state_list[0]])
        dr['full']=dr[state_list[1]]**2 * np.sign(dr[state_list[1]])

        dr['sig']=((dp['r'][state_list[1]]-dp['r'][state_list[0]]) > \
             (dp['r_se'][state_list[1]]+
              dp['r_se'][state_list[0]]))
        dr['cellid'] = dp['r'][state_list[1]].index
        #dm = dr.loc[dr['sig'].values,['null','full','bp_common','p_unique','b_unique']]
        dm = dr.loc[:,['cellid','null','full','bp_common','b_unique','p_unique','sig']]
        dm = dm.sort_values(['null'])
        mfull=dm[['null','full','bp_common','b_unique','p_unique','sig']].values
        cellids=dm['cellid'].to_list()

        big_idx = mfull[:,1]-mfull[:,0]>0.2
        for i,b in enumerate(big_idx):
            if b:
                print('{} : {:.3f} - {:.3f}'.format(cellids[i],mfull[i,0],mfull[i,1]))

    if nb > 0:
        stepsize = mfull.shape[0]/nb
        mm=np.zeros((nb,mfull.shape[1]))
        for i in range(nb):
            x01=(mfull[:,0]>i/nb) & (mfull[:,0]<=(i+1)/nb)
            if np.sum(x01):
                mm[i,:]=np.nanmean(mfull[x01,:],axis=0)

        print(np.round(mm,3))

        m = mm.copy()
    else:
        # alt to look at each cell individually:
        m = mfull.copy()

    mall = np.nanmean(mfull, axis=0, keepdims=True)

    # remove sensory component, which swamps everything else
    mall = mall[:, 2:]
    mb=m[:,2:]

    ax1 = plt.subplot(3,1,1)
    stateplots.beta_comp(mfull[:,0],mfull[:,1],n1='State independent',n2='Full state-dep',
                         ax=ax1, highlight=mfull[:, -1], hist_range=[-0.1, 1])

    ax2 = plt.subplot(3,1,2)
    width=0.8
    mplots=np.concatenate((mall, mb), axis=0)
    ind = np.arange(mplots.shape[0])

    p1 = plt.bar(ind, mplots[:,0], width=width, color=colors[1])
    p2 = plt.bar(ind, mplots[:,1], width=width, bottom=mplots[:,0], color=colors[2])
    p3 = plt.bar(ind, mplots[:,2], width=width, bottom=mplots[:,0]+mplots[:,1], color=colors[3])
    plt.legend(('common','b-unique','p_unique'))
    if title is not None:
        plt.title(title)
    plt.xlabel('behavior-independent quintile')
    plt.ylabel('mean r2')

    ax3 = plt.subplot(3,1,3)
    d=(mfull[:,1]-mfull[:,0])
    stateplots.beta_comp(mfull[:,0], d, n1='State independent',n2='dep - indep',
                     ax=ax3, highlight=mfull[:,-1], hist_range=[-0.1, 1], markersize=4)
    ax3.plot([1,0], [0,1], 'k--', linewidth=0.5)
    r, p = st.pearsonr(mfull[:,0],d)
    plt.title('cc={:.3} p={:.4}'.format(r,p))


    f.tight_layout()

    return f


def hlf_analysis(df, state_list, pas_df=None, norm_sign=True, sig_cells_only=False, states=None, scatter_sig_cells=None):
    """
    Copied/modified version of mod_per_state.hlf_analysis. Rewritten by crh 04/17/2020
    """
    # figure out what cells show significant state effect. Can just use
    # pupil for this, so that there's one entry per cell (rtest is the same for all states)

    if states is None:
        states = ['ACTIVE_1','PASSIVE_1', 'ACTIVE_2', 'PASSIVE_2']
    
    da = df[df['state_chan']=='pupil']
    dp = pd.pivot_table(da, index='cellid',columns='state_sig',values=['r','r_se'])

    sig = (dp.loc[:, pd.IndexSlice['r', state_list[3]]] - dp.loc[:, pd.IndexSlice['r', state_list[0]]]) > \
            (dp.loc[:, pd.IndexSlice['r_se', state_list[3]]] + dp.loc[:, pd.IndexSlice['r_se', state_list[0]]])
    sig_cells = sig[sig].index

    dfull = df[df['state_sig']==state_list[3]]
    dpup = df[df['state_sig']==state_list[2]] 
    dbeh = df[df['state_sig']==state_list[1]] 
    dp = pd.pivot_table(dfull, index='cellid',columns='state_chan',values=['MI'])
    dp_beh = pd.pivot_table(dbeh, index='cellid',columns='state_chan',values=['MI'])
    dp0 = pd.pivot_table(dpup, index='cellid',columns='state_chan',values=['MI'])

    dMI = dp.loc[:, pd.IndexSlice['MI', states]]
    dMIbeh = dp_beh.loc[:, pd.IndexSlice['MI', states]]
    dMI0 = dp0.loc[:, pd.IndexSlice['MI', states]]
    dMIu = dMI - dMI0
    
    if pas_df is not None:
        dfull_pas = pas_df[pas_df['state_sig']=='st.pup.pas']
        dbeh_pas = pas_df[pas_df['state_sig']=='st.pup0.pas']
        dpup_pas = pas_df[pas_df['state_sig']=='st.pup.pas0'] 
        dp_pas = pd.pivot_table(dfull_pas, index='cellid',columns='state_chan_alt',values=['MI'])
        dp_beh_pas = pd.pivot_table(dbeh_pas, index='cellid',columns='state_chan_alt',values=['MI'])
        dp0_pas = pd.pivot_table(dpup_pas, index='cellid',columns='state_chan_alt',values=['MI'])

        dMI_pas = dp_pas.loc[:, pd.IndexSlice['MI', states]]
        dMI0_pas = dp0_pas.loc[:, pd.IndexSlice['MI', states]]
        dMIu_pas = dMI_pas - dMI0_pas
        dMI_pas = dp_beh_pas.loc[:, pd.IndexSlice['MI', states]]

    
    # add zeros for "PASSIVE_0" col
    dMI.at[:, pd.IndexSlice['MI', 'PASSIVE_0']] = 0
    dMI0.at[:, pd.IndexSlice['MI', 'PASSIVE_0']] = 0
    dMIu.at[:, pd.IndexSlice['MI', 'PASSIVE_0']] = 0

    active_idx = [c for c in dMI.columns.get_level_values('state_chan') if 'ACTIVE' in c]
    passive_idx = [c for c in dMI.columns.get_level_values('state_chan') if 'PASSIVE' in c]

    # force reorder the columns of all dataframes for the plot
    new_col_order = sorted(dMI.columns.get_level_values('state_chan'), key=lambda x: x[-1])
    new_cols = pd.MultiIndex.from_product([['MI'], new_col_order], names=[None, 'state_chan'])
    dMI = dMI.reindex(columns=new_cols, fill_value=0)
    dMI0 = dMI0.reindex(columns=new_cols, fill_value=0)
    dMIu = dMIu.reindex(columns=new_cols, fill_value=0)

    # define data to use for scatter plot
    if pas_df is not None:
        dMI_all = dMI_pas.copy()
        dMIu_all = dMIu_pas.copy()
    else:
        # dMI_all = dMI.copy()
        dMI_all = dMIbeh.copy()
        dMIu_all = dMIu.copy()

    if norm_sign:
        b = dMI.loc[:, pd.IndexSlice['MI', passive_idx]].mean(axis=1).fillna(0)
        dMI = dMI.subtract(b, axis=0)
        dMIu = dMIu.subtract(b, axis=0)
        dMI0 = dMI0.subtract(b, axis=0)
        sg = dMI.loc[:, pd.IndexSlice['MI', active_idx]].mean(axis=1) - \
                    dMI.loc[:, pd.IndexSlice['MI', passive_idx]].mean(axis=1)
        sg = sg.apply(np.sign)
        dMI = dMI.multiply(sg, axis=0)
        dMIu = dMIu.multiply(sg, axis=0)
        dMI0 = dMI0.multiply(sg, axis=0)

    # plot only significant state cells, with data for all state_chan conditions
    state_mask = (dMI.isna().sum(axis=1) == 0)
    cell_mask = dMI.index.isin(sig_cells)
    if sig_cells_only:
        dMI = dMI.loc[cell_mask & state_mask, :]
        dMI0 = dMI0.loc[cell_mask & state_mask, :]
        dMIu = dMIu.loc[cell_mask & state_mask, :]
    else:
        dMI = dMI.loc[state_mask, :]
        dMI0 = dMI0.loc[state_mask, :]
        dMIu = dMIu.loc[state_mask, :]

    total_cells = len(df.cellid.unique())
    sig_state_cells = len(sig_cells)
    stable_cells = state_mask.sum()

    f, ax = plt.subplots(2, 1, figsize=(8, 8))

    # scatter plot of raw post passive MI vs. unique post passive MI
    # e.g. does pupil account for some persistent effects?
    ax[0].scatter(dMI_all.loc[:, pd.IndexSlice['MI', 'PASSIVE_1']], 
                        dMIu_all.loc[:, pd.IndexSlice['MI', 'PASSIVE_1']], color='lightgrey', edgecolor='white', s=40, label='all cells')
    if scatter_sig_cells is None:
        pass
    else:
        # if sig cells, overlay colors on sig cells
        for category in scatter_sig_cells:
            sig_cells = scatter_sig_cells[category]
            if category == 'task_only':
                color = common.color_b
            elif category == 'pupil_only':
                color = common.color_p
            elif category == 'both':
                color = common.color_both
            elif category == 'task_or_pupil':
                color = common.color_either
            else:
                color = 'k'
            ax[0].scatter(dMI_all.loc[sig_cells, pd.IndexSlice['MI', 'PASSIVE_1']], 
                        dMIu_all.loc[sig_cells, pd.IndexSlice['MI', 'PASSIVE_1']], color=color, edgecolor='white', s=50, label=category)

    ax[0].legend()
    ax[0].set_xlabel('Pre vs. post MI, task only')
    ax[0].set_ylabel('Pre vs. post MI, task unique')
    ax[0].plot([-1, 1], [-1, 1], 'k--')
    ax[0].axhline(0, linestyle='--', color='k')
    ax[0].axvline(0, linestyle='--', color='k')
    ax[0].axis('square')

    # plot mean MI over cells for pupil, task unique, and overall state
    ax[1].set_title('total cells: {0}, \n state cells: {1}, \n stable across all blocks: {2}'.format(total_cells, sig_state_cells, stable_cells),
            fontsize=8)
    ax[1].set_title('Total cells going into average: {0}'.format(dMI.shape[0]))
    ax[1].plot(dMIu.mean(axis=0).values, '-', lw=2, color=common.color_b, marker='o', label='unique task')
    ax[1].plot(dMI.mean(axis=0).values, '--', lw=2, color=common.color_b, marker='o', label='overall')
    ax[1].plot(dMI0.mean(axis=0).values, '--', color=common.color_p, lw=2, marker='o', label='pupil')
    ax[1].legend()

    ax[1].axhline(0, linestyle='--', color='grey', lw=2)
    ax[1].set_ylabel('mean MI')
    ax[1].set_xticks(np.arange(dMI.shape[1]))
    ax[1].set_xticklabels(dMI.columns.get_level_values('state_chan'))
    ax[1].set_xlabel('behavioral block')

    f.tight_layout()

    return dMI, dMI0