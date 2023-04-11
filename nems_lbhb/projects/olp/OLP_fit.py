import numpy as np
import matplotlib.pyplot as plt
import nems0.db as nd  # NEMS database functions -- NOT celldb
import nems_lbhb.baphy as nb   # baphy-specific functions
import nems_lbhb.xform_wrappers as nw  # wrappers for calling nems code with database stuff
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems0.recording as recording
import numpy as np
import nems0.preprocessing as preproc
import nems0.metrics.api as nmet
import pickle as pl
import pandas as pd
import sys
import os
import re
import seaborn as sns
import itertools
import nems0.epoch as ep
import logging

import glob
import nems0.analysis.api
import nems0.modelspec as ms
import warnings
import pandas as pd
from nems_lbhb import baphy_io

import nems_lbhb.projects.olp.binaural_OLP_helpers as bnh
log = logging.getLogger(__name__)
import nems_lbhb.fitEllipse as fE

import nems0.db as nd  # NEMS database functions -- NOT celldb
import nems_lbhb.baphy as nb  # baphy-specific functions
import nems_lbhb.xform_wrappers as nw  # wrappers for calling nems code with database stuff
import nems0.recording as recording
import nems_lbhb.SPO_helpers as sp
import nems0.preprocessing as preproc
import nems0.metrics.api as nmet
import nems0.metrics.corrcoef
import copy
import nems0.epoch as ep
import scipy.stats as sst
from nems_lbhb.gcmodel.figures.snr import compute_snr
from nems0.preprocessing import generate_psth_from_resp
import logging
import nems_lbhb.projects.olp.OLP_helpers as ohel
import nems_lbhb.TwoStim_helpers as ts

from nems0.xform_helper import load_model_xform
import joblib as jl
from datetime import date


log = logging.getLogger(__name__)
from nems0 import db

def OLP_fit_weights(batch=340, parmfile=None, loadpath=None, savepath=None, filter=None,
                    cells=None, fs=100):
    '''Puts all the compontents that go into weight_df into one command. Gives you the option
    to save the resulting dataframe it returns. But mainly it gives you the option to either
    pass an entire batch to it or take a single parmfile and only use the cells from that.
    Added 2022_08_23 getting things together after CLT recordings.'''
    if not loadpath:
        if cells:
            # Give me the option of manually feeding it a list of cells I got from somewhere else
            cell_list = cells
        else:
            if parmfile:
                manager = BAPHYExperiment(parmfile)
                options = {'rasterfs': fs, 'stim': True, 'stimfmt': 'lenv', 'resp': True, 'recache': False}
                rec = manager.get_recording(**options)
                cell_list = rec.signals['resp'].chans
            else:
                cell_df = nd.get_batch_cells(batch)
                cell_list = cell_df['cellid'].tolist()
                if isinstance(filter, str):
                    cell_list = [cc for cc in cell_list if filter in cc]
                elif isinstance(filter, list):
                    if len(filter) <= 2:
                        cell_list = [cc for cc in cell_list if (filter[0] in cc) or (filter[-1] in cc)]
                    else:
                        cell_list = [cc for cc in cell_list if cc[:6] in filter]
                else:
                    raise ValueError(f"Can't filter by {filter}, must be a single string or list of two strings.")
                if len(cell_list) == 0:
                    raise ValueError(f"You did something wrong with your filter, there are no cells left.")

        cell_list = ohel.manual_fix_units(cell_list)  # So far only useful for two TBR cells

        # Gets some cell metrics
        metrics=[]
        for cellid in cell_list:
            cell_metric = calc_psth_metrics(batch, cellid)
            cell_metric.insert(loc=0, column='cellid', value=cellid)
            print(f"Adding cellid {cellid}.")
            metrics.append(cell_metric)
        df = pd.concat(metrics)
        df.reset_index()

        # Fits weights
        weight_df = fit_weights(df, batch, fs)

        # # Adds sound stats to the weight_df
        # sound_df = ohel.get_sound_statistics(weight_df, plot=False)
        # weight_df = ohel.add_sound_stats(weight_df, sound_df)

        # Adding 2022_09_14. If this doesn't work, figure it out, or revert to the lines above
        # Line below should add synthetic parameters to files that are too old to have that
        if 'synth_kind' not in weight_df:
            weight_df['synth_kind'] = 'A'
        # Line below should add binaural parameters to files that are too old to have that
        if 'kind' not in weight_df:
            weight_df['kind'] = '11'
        # Adds relative gain so that sound_df can be computed with it, but also because it's useful
        weight_df['BG_rel_gain'] = (weight_df.weightsA - weight_df.weightsB) / \
                                   (np.abs(weight_df.weightsB) + np.abs(weight_df.weightsA))
        weight_df['FG_rel_gain'] = (weight_df.weightsB - weight_df.weightsA) / \
                                   (np.abs(weight_df.weightsB) + np.abs(weight_df.weightsA))
        sound_df = ohel.get_sound_statistics_full(weight_df)
        weight_df = ohel.add_sound_stats(weight_df, sound_df)

        if savepath:
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            store = pd.HDFStore(savepath)
            df_store = copy.deepcopy(weight_df)
            store['df'] = df_store.copy()
            store.close()

    else:
        store = pd.HDFStore(loadpath)
        weight_df = store['df']
        store.close()

    return weight_df


def calc_psth_metrics(batch, cellid, parmfile=None, paths=None):
    start_win_offset = 0  # Time (in sec) to offset the start of the window used to calculate threshold, exitatory percentage, and inhibitory percentage
    if parmfile:
        manager = BAPHYExperiment(parmfile)
    else:
        manager = BAPHYExperiment(cellid=cellid, batch=batch)

    options = ohel.get_load_options(batch) #gets options that will include gtgram if batch=339
    rec = manager.get_recording(**options)

    area_df = db.pd_query(f"SELECT DISTINCT area FROM sCellFile where cellid like '{manager.siteid}%%'")
    area = area_df.area.iloc[0]

    if rec['resp'].epochs[rec['resp'].epochs['name'] == 'PASSIVE_EXPERIMENT'].shape[0] >= 2:
        rec = ohel.remove_olp_test(rec)

    rec['resp'] = rec['resp'].extract_channels([cellid])
    resp = copy.copy(rec['resp'].rasterize())
    rec['resp'].fs = 100

    norm_spont, SR, STD = ohel.remove_spont_rate_std(resp)
    params = ohel.get_expt_params(resp, manager, cellid)

    epcs = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PreStimSilence'].copy()
    ep2 = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PostStimSilence'].iloc[0].copy()
    params['prestim'], params['poststim'] = epcs.iloc[0]['end'], ep2['end'] - ep2['start']
    params['lenstim'] = ep2['end']

    stim_epochs = ep.epoch_names_matching(resp.epochs, 'STIM_')

    if paths and cellid[:3] == 'TBR':
        print(f"Deprecated, run on {cellid} though...")
        stim_epochs, rec, resp = ohel.path_tabor_get_epochs(stim_epochs, rec, resp, params)

    epoch_repetitions = [resp.count_epoch(cc) for cc in stim_epochs]
    full_resp = np.empty((max(epoch_repetitions), len(stim_epochs),
                          (int(params['lenstim']) * rec['resp'].fs)))
    full_resp[:] = np.nan
    for cnt, epo in enumerate(stim_epochs):
        resps_list = resp.extract_epoch(epo)
        full_resp[:resps_list.shape[0], cnt, :] = resps_list[:, 0, :]

    #Calculate a few metrics
    corcoef = ohel.calc_base_reliability(full_resp)
    avg_resp = ohel.calc_average_response(full_resp, params)
    snr = compute_snr(resp)

    #Grab and label epochs that have two sounds in them (no null)
    presil, postsil = int(params['prestim'] * rec['resp'].fs), int(params['poststim'] * rec['resp'].fs)
    twostims = resp.epochs[resp.epochs['name'].str.count('-0-1') == 2].copy()
    ep_twostim = twostims.name.unique().tolist()
    ep_twostim.sort()

    ep_names = resp.epochs[resp.epochs['name'].str.contains('STIM_')].copy()
    ep_names = ep_names.name.unique().tolist()
    ep_types = list(map(ohel.label_ep_type, ep_names))
    ep_synth_type = list(map(ohel.label_synth_type, ep_names))

    ep_df = pd.DataFrame({'name': ep_names, 'type': ep_types, 'synth_type': ep_synth_type})

    cell_df = []
    for cnt, stimmy in enumerate(ep_twostim):
        kind = ohel.label_pair_type(stimmy)
        synth_kind = ohel.label_synth_type(stimmy)
        # seps = (stimmy.split('_')[1], stimmy.split('_')[2])
        seps = list(re.findall("_(null|\d{2}.*)_(null|\d{2}.*)", stimmy)[0])
        BG, FG = seps[0].split('-')[0][2:], seps[1].split('-')[0][2:]

        Aepo, Bepo = 'STIM_' + seps[0] + '_null', 'STIM_null_' + seps[1]

        rAB = resp.extract_epoch(stimmy)
        rA, rB = resp.extract_epoch(Aepo), resp.extract_epoch(Bepo)

        fn = lambda x: np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - SR)
        rAsm = np.squeeze(np.apply_along_axis(fn, 2, rA))
        rBsm = np.squeeze(np.apply_along_axis(fn, 2, rB))
        rABsm = np.squeeze(np.apply_along_axis(fn, 2, rAB))

        rA_st, rB_st = rAsm[:, presil:-postsil], rBsm[:, presil:-postsil]
        rAB_st = rABsm[:, presil:-postsil]

        rAm, rBm = np.nanmean(rAsm, axis=0), np.nanmean(rBsm, axis=0)
        rABm = np.nanmean(rABsm, axis=0)

        AcorAB = np.corrcoef(rAm, rABm)[0, 1]  # Corr between resp to A and resp to dual
        BcorAB = np.corrcoef(rBm, rABm)[0, 1]  # Corr between resp to B and resp to dual

        A_FR, B_FR, AB_FR = np.nanmean(rA_st), np.nanmean(rB_st), np.nanmean(rAB_st)

        min_rep = np.min((rA.shape[0], rB.shape[0])) #only will do something if SoundRepeats==Yes
        lin_resp = np.nanmean(rAsm[:min_rep, :] + rBsm[:min_rep, :], axis=0)
        supp = np.nanmean(lin_resp - AB_FR)

        AcorLin = np.corrcoef(rAm, lin_resp)[0, 1]  # Corr between resp to A and resp to lin
        BcorLin = np.corrcoef(rBm, lin_resp)[0, 1]  # Corr between resp to B and resp to lin

        Apref, Bpref = AcorAB - AcorLin, BcorAB - BcorLin
        pref = Apref - Bpref

        # if params['Binaural'] == 'Yes':
        #     dA, dB = ohel.get_binaural_adjacent_epochs(stimmy)
        #
        #     rdA, rdB = resp.extract_epoch(dA), resp.extract_epoch(dB)
        #     rdAm = np.nanmean(np.squeeze(np.apply_along_axis(fn, 2, rdA))[:, presil:-postsil], axis=0)
        #     rdBm = np.nanmean(np.squeeze(np.apply_along_axis(fn, 2, rdB))[:, presil:-postsil], axis=0)
        #
        #     ABcordA = np.corrcoef(rABm, rdAm)[0, 1]  # Corr between resp to AB and resp to BG swap
        #     ABcordB = np.corrcoef(rABm, rdBm)[0, 1]  # Corr between resp to AB and resp to FG swap

        cell_df.append({'epoch': stimmy,
                        'kind': kind,
                        'synth_kind': synth_kind,
                        'BG': BG,
                        'FG': FG,
                        'AcorAB': AcorAB,
                        'BcorAB': BcorAB,
                        'AcorLin': AcorLin,
                        'BcorLin': BcorLin,
                        'Apref': Apref,
                        'Bpref': Bpref,
                        'pref': pref,
                        'combo_FR': AB_FR,
                        'bg_FR': A_FR,
                        'fg_FR': B_FR,
                        'supp': supp})

    cell_df = pd.DataFrame(cell_df)
    cell_df['SR'], cell_df['STD'] = SR, STD
    # cell_df['corcoef'], cell_df['avg_resp'], cell_df['snr'] = corcoef, avg_resp, snr
    cell_df.insert(loc=0, column='area', value=area)

    return cell_df




    # COMPUTE ALL FOLLOWING metrics using smoothed driven rate
    # est, val = rec.split_using_epoch_occurrence_counts(rec,epoch_regex='^STIM_')
    val = rec.copy()
    val['resp'] = val['resp'].rasterize()
    val['stim'] = val['stim'].rasterize()
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')

    # smooth and subtract SR
    fn = lambda x: np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - SR)
    val['resp'] = val['resp'].transform(fn)
    val['resp'] = ohel.add_stimtype_epochs(val['resp'])

    if val['resp'].count_epoch('REFERENCE'):
        epochname = 'REFERENCE'
    else:
        epochname = 'TRIAL'
    sts = val['resp'].epochs['start'].copy()
    nds = val['resp'].epochs['end'].copy()
    sts_rec = rec['resp'].epochs['start'].copy()
    val['resp'].epochs['end'] = val['resp'].epochs['start'] + params['prestim']
    ps = val['resp'].select_epochs([epochname]).as_continuous()
    ff = np.isfinite(ps)
    SR_av = ps[ff].mean() * resp.fs
    SR_av_std = ps[ff].std() * resp.fs

    # Compute max over single-voice trials
    val['resp'].epochs['end'] = nds
    val['resp'].epochs['start'] = sts
    val['resp'].epochs['start'] = val['resp'].epochs['start'] + params['prestim']
    TotalMax = np.nanmax(val['resp'].as_continuous())
    ps = np.hstack((val['resp'].extract_epoch('10').flatten(), val['resp'].extract_epoch('01').flatten()))
    SinglesMax = np.nanmax(ps)

    # Compute threshold, exitatory percentage, and inhibitory percentage
    prestim, poststim = params['prestim'], params['poststim']
    val['resp'].epochs['end'] = nds
    val['resp'].epochs['start'] = sts
    val['resp'].epochs['start'] = val['resp'].epochs['start'] + prestim + start_win_offset
    val['resp'].epochs['end'] = val['resp'].epochs['end'] - poststim
    thresh = np.array(((SR + SR_av_std) / resp.fs,
                       (SR - SR_av_std) / resp.fs))
    thresh = np.array((SR / resp.fs + 0.1 * (SinglesMax - SR / resp.fs),
                       (SR - SR_av_std) / resp.fs))
    # SR/resp.fs - 0.5 * (np.nanmax(val['resp'].as_continuous()) - SR/resp.fs)]

    types = ['10', '01', '20', '02', '11', '12', '21', '22']
    excitatory_percentage = {}
    inhibitory_percentage = {}
    Max = {}
    Mean = {}
    for _type in types:
        if _type in val['resp'].epochs.name.values:
            ps = val['resp'].extract_epoch(_type).flatten()
            ff = np.isfinite(ps)
            excitatory_percentage[_type] = (ps[ff] > thresh[0]).sum() / ff.sum()
            inhibitory_percentage[_type] = (ps[ff] < thresh[1]).sum() / ff.sum()
            Max[_type] = ps[ff].max() / SinglesMax
            Mean[_type] = ps[ff].mean()

    # Compute threshold, exitatory percentage, and inhibitory percentage just over onset time
    # restore times
    val['resp'].epochs['end'] = nds
    val['resp'].epochs['start'] = sts
    # Change epochs to stimulus onset times
    val['resp'].epochs['start'] = val['resp'].epochs['start'] + prestim
    val['resp'].epochs['end'] = val['resp'].epochs['start'] + prestim + .5
    excitatory_percentage_onset = {}
    inhibitory_percentage_onset = {}
    Max_onset = {}
    for _type in types:
        ps = val['resp'].extract_epoch(_type).flatten()
        ff = np.isfinite(ps)
        excitatory_percentage_onset[_type] = (ps[ff] > thresh[0]).sum() / ff.sum()
        inhibitory_percentage_onset[_type] = (ps[ff] < thresh[1]).sum() / ff.sum()
        Max_onset[_type] = ps[ff].max() / SinglesMax


        # find correlations between double and single-voice responses
    val['resp'].epochs['end'] = nds
    val['resp'].epochs['start'] = sts
    val['resp'].epochs['start'] = val['resp'].epochs['start'] + prestim
    rec['resp'].epochs['start'] = rec['resp'].epochs['start'] + prestim
    # over stim on time to end + 0.5
    val['linmodel'] = val['resp'].copy()
    val['linmodel']._data = np.full(val['linmodel']._data.shape, np.nan)
    types = ['11', '12', '21', '22']
    epcs = val['resp'].epochs[val['resp'].epochs['name'].str.contains('STIM')].copy()
    epcs['type'] = epcs['name'].apply(ohel.label_ep_type)
    names = [[n.split('_')[1], n.split('_')[2]] for n in epcs['name']]
    EA = np.array([n[0] for n in names])
    EB = np.array([n[1] for n in names])

    r_dual_B, r_dual_A, r_dual_B_nc, r_dual_A_nc  = {}, {}, {}, {}
    r_dual_B_bal, r_dual_A_bal = {}, {}
    r_lin_B, r_lin_A, r_lin_B_nc, r_lin_A_nc = {}, {}, {}, {}
    r_lin_B_bal, r_lin_A_bal = {}, {}

    N_ac = 200
    full_resp = rec['resp'].rasterize()
    full_resp = full_resp.transform(fn)
    for _type in types:
        inds = np.nonzero(epcs['type'].values == _type)[0]
        rA_st, rB_st, r_st, rA_rB_st = [], [], [], []
        init = True
        for ind in inds:
            # for each dual-voice response
            r = val['resp'].extract_epoch(epcs.iloc[ind]['name'])
            if np.any(np.isfinite(r)):
                print(epcs.iloc[ind]['name'])
                # Find the indicies of single-voice responses that match this dual-voice response
                indA = np.where((EA[ind] == EA) & (EB == 'null'))[0]
                indB = np.where((EB[ind] == EB) & (EA == 'null'))[0]
                if (len(indA) > 0) & (len(indB) > 0):
                    # from pdb import set_trace
                    # set_trace()
                    rA = val['resp'].extract_epoch(epcs.iloc[indA[0]]['name'])
                    rB = val['resp'].extract_epoch(epcs.iloc[indB[0]]['name'])
                    r_st.append(full_resp.extract_epoch(epcs.iloc[ind]['name'])[:, 0, :])
                    rA_st_ = full_resp.extract_epoch(epcs.iloc[indA[0]]['name'])[:, 0, :]
                    rB_st_ = full_resp.extract_epoch(epcs.iloc[indB[0]]['name'])[:, 0, :]
                    rA_st.append(rA_st_)
                    rB_st.append(rB_st_)
                    minreps = np.min((rA_st_.shape[0], rB_st_.shape[0]))
                    rA_rB_st.append(rA_st_[:minreps, :] + rB_st_[:minreps, :])
                    if init:
                        rA_ = rA.squeeze();
                        rB_ = rB.squeeze();
                        r_ = r.squeeze();
                        rA_rB_ = rA.squeeze() + rB.squeeze()
                        init = False
                    else:
                        rA_ = np.hstack((rA_, rA.squeeze()))
                        rB_ = np.hstack((rB_, rB.squeeze()))
                        r_ = np.hstack((r_, r.squeeze()))
                        rA_rB_ = np.hstack((rA_rB_, rA.squeeze() + rB.squeeze()))
                    val['linmodel'] = val['linmodel'].replace_epoch(epcs.iloc[ind]['name'], rA + rB, preserve_nan=False)
        ff = np.isfinite(r_) & np.isfinite(rA_) & np.isfinite(rB_)  # find places with data
        r_dual_A[_type] = np.corrcoef(rA_[ff], r_[ff])[0, 1]  # Correlation between response to A and response to dual
        r_dual_B[_type] = np.corrcoef(rB_[ff], r_[ff])[0, 1]  # Correlation between response to B and response to dual
        r_lin_A[_type] = np.corrcoef(rA_[ff], rA_rB_[ff])[
            0, 1]  # Correlation between response to A and response to linear 'model'
        r_lin_B[_type] = np.corrcoef(rB_[ff], rA_rB_[ff])[
            0, 1]  # Correlation between response to B and response to linear 'model'

        # correlations over single-trial data
        minreps = np.min([x.shape[0] for x in r_st])
        r_st = [x[:minreps, :] for x in r_st]
        r_st = np.concatenate(r_st, axis=1)
        rA_st = [x[:minreps, :] for x in rA_st]
        rA_st = np.concatenate(rA_st, axis=1)
        rB_st = [x[:minreps, :] for x in rB_st]
        rB_st = np.concatenate(rB_st, axis=1)
        rA_rB_st = [x[:minreps, :] for x in rA_rB_st]
        rA_rB_st = np.concatenate(rA_rB_st, axis=1)

        r_lin_A_bal[_type] = np.corrcoef(rA_st[0::2, ff].mean(axis=0), rA_rB_st[1::2, ff].mean(axis=0))[0, 1]
        r_lin_B_bal[_type] = np.corrcoef(rB_st[0::2, ff].mean(axis=0), rA_rB_st[1::2, ff].mean(axis=0))[0, 1]
        r_dual_A_bal[_type] = np.corrcoef(rA_st[0::2, ff].mean(axis=0), r_st[:, ff].mean(axis=0))[0, 1]
        r_dual_B_bal[_type] = np.corrcoef(rB_st[0::2, ff].mean(axis=0), r_st[:, ff].mean(axis=0))[0, 1]

        r_dual_A_nc[_type] = ohel.r_noise_corrected(rA_st, r_st)
        r_dual_B_nc[_type] = ohel.r_noise_corrected(rB_st, r_st)
        r_lin_A_nc[_type] = ohel.r_noise_corrected(rA_st, rA_rB_st)
        r_lin_B_nc[_type] = ohel.r_noise_corrected(rB_st, rA_rB_st)

        if _type == '11':
            r11 = nems0.metrics.corrcoef._r_single(r_st, 200, 0)
        elif _type == '12':
            r12 = nems0.metrics.corrcoef._r_single(r_st, 200, 0)
        elif _type == '21':
            r21 = nems0.metrics.corrcoef._r_single(r_st, 200, 0)
        elif _type == '22':
            r22 = nems0.metrics.corrcoef._r_single(r_st, 200, 0)
        # rac = _r_single(X, N)
        # r_ceiling = [nmet.r_ceiling(p, rec, 'pred', 'resp') for p in val_copy]

    # Things that used to happen only for _type is 'C' but still seem valid
    r_A_B = np.corrcoef(rA_[ff], rB_[ff])[0, 1]
    r_A_B_nc = r_noise_corrected(rA_st, rB_st)
    rAA = nems0.metrics.corrcoef._r_single(rA_st, 200, 0)
    rBB = nems0.metrics.corrcoef._r_single(rB_st, 200, 0)
    Np = 0
    rAA_nc = np.zeros(Np)
    rBB_nc = np.zeros(Np)
    hv = int(minreps / 2);
    for i in range(Np):
        inds = np.random.permutation(minreps)
        rAA_nc[i] = sp.r_noise_corrected(rA_st[inds[:hv]], rA_st[inds[hv:]])
        rBB_nc[i] = sp.r_noise_corrected(rB_st[inds[:hv]], rB_st[inds[hv:]])
    ffA = np.isfinite(rAA_nc)
    ffB = np.isfinite(rBB_nc)
    rAAm = rAA_nc[ffA].mean()
    rBBm = rBB_nc[ffB].mean()
    mean_nsA = rA_st.sum(axis=1).mean()
    mean_nsB = rB_st.sum(axis=1).mean()
    min_nsA = rA_st.sum(axis=1).min()
    min_nsB = rB_st.sum(axis=1).min()

    # Calculate correlation between linear 'model and dual-voice response, and mean amount of suppression, enhancement relative to linear 'model'
    r_fit_linmodel = {}
    r_fit_linmodel_NM = {}
    r_ceil_linmodel = {}
    mean_enh = {}
    mean_supp = {}
    EnhP = {}
    SuppP = {}
    DualAboveZeroP = {}
    resp_ = copy.deepcopy(rec['resp'].rasterize())
    resp_.epochs['start'] = sts_rec
    fn = lambda x: np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - SR / val['resp'].fs)
    resp_ = resp_.transform(fn)
    for _type in types:
        val_copy = copy.deepcopy(val)
        #        from pdb import set_trace
        #        set_trace()
        val_copy['resp'] = val_copy['resp'].select_epochs([_type])
        # Correlation between linear 'model' (response to A plus response to B) and dual-voice response
        r_fit_linmodel_NM[_type] = nmet.corrcoef(val_copy, 'linmodel', 'resp')
        # r_ceil_linmodel[_type] = nems0.metrics.corrcoef.r_ceiling(val_copy,rec,'linmodel', 'resp',exclude_neg_pred=False)[0]
        # Noise-corrected correlation between linear 'model' (response to A plus response to B) and dual-voice response
        r_ceil_linmodel[_type] = nems0.metrics.corrcoef.r_ceiling(val_copy, rec, 'linmodel', 'resp')[0]

        pred = val_copy['linmodel'].as_continuous()
        resp = val_copy['resp'].as_continuous()
        ff = np.isfinite(pred) & np.isfinite(resp)
        # cc = np.corrcoef(sp.smooth(pred[ff],3,2), sp.smooth(resp[ff],3,2))
        cc = np.corrcoef(pred[ff], resp[ff])
        r_fit_linmodel[_type] = cc[0, 1]

        prdiff = resp[ff] - pred[ff]
        mean_enh[_type] = prdiff[prdiff > 0].mean() * val['resp'].fs
        mean_supp[_type] = prdiff[prdiff < 0].mean() * val['resp'].fs

        # Find percent of time response is suppressed vs enhanced relative to what would be expected by a linear sum of single-voice responses
        # First, jacknife to find...
    #        Njk=10
    #        if _type is 'C':
    #            stims=['STIM_T+si464+si464','STIM_T+si516+si516']
    #        else:
    #            stims=['STIM_T+si464+si516', 'STIM_T+si516+si464']
    #        T=int(700+prestim*val['resp'].fs)
    #        Tps=int(prestim*val['resp'].fs)
    #        jns=np.zeros((Njk,T,len(stims)))
    #        for ns in range(len(stims)):
    #            for njk in range(Njk):
    #                resp_jn=resp_.jackknife_by_epoch(Njk,njk,stims[ns])
    #                jns[njk,:,ns]=np.nanmean(resp_jn.extract_epoch(stims[ns]),axis=0)
    #        jns=np.reshape(jns[:,Tps:,:],(Njk,700*len(stims)),order='F')
    #
    #        lim_models=np.zeros((700,len(stims)))
    #        for ns in range(len(stims)):
    #            lim_models[:,ns]=val_copy['linmodel'].extract_epoch(stims[ns])
    #        lim_models=lim_models.reshape(700*len(stims),order='F')
    #
    #        ff=np.isfinite(lim_models)
    #        mean_diff=(jns[:,ff]-lim_models[ff]).mean(axis=0)
    #        std_diff=(jns[:,ff]-lim_models[ff]).std(axis=0)
    #        serr_diff=np.sqrt(Njk/(Njk-1))*std_diff
    #
    #        thresh=3
    #        dual_above_zero = (jns[:,ff].mean(axis=0) > std_diff)
    #        sig_enh = ((mean_diff/serr_diff) > thresh) & dual_above_zero
    #        sig_supp = ((mean_diff/serr_diff) < -thresh)
    #        DualAboveZeroP[_type] = (dual_above_zero).sum()/len(mean_diff)
    #        EnhP[_type] = (sig_enh).sum()/len(mean_diff)
    #        SuppP[_type] = (sig_supp).sum()/len(mean_diff)

    #        time = np.arange(0, lim_models.shape[0])/ val['resp'].fs
    #        plt.figure();
    #        plt.plot(time,jns.mean(axis=0),'.-k');
    #        plt.plot(time,lim_models,'.-g');
    #        plt.plot(time[sig_enh],lim_models[sig_enh],'.r')
    #        plt.plot(time[sig_supp],lim_models[sig_supp],'.b')
    #        plt.title('Type:{:s}, Enh:{:.2f}, Sup:{:.2f}, Resp_above_zero:{:.2f}'.format(_type,EnhP[_type],SuppP[_type],DualAboveZeroP[_type]))
    #        from pdb import set_trace
    #        set_trace()
    #        a=2
    # thrsh=5
    #        EnhP[_type] = ((prdiff*val['resp'].fs) > thresh).sum()/len(prdiff)
    #        SuppP[_type] = ((prdiff*val['resp'].fs) < -thresh).sum()/len(prdiff)
    #    return val
    #    return {'excitatory_percentage':excitatory_percentage,
    #            'inhibitory_percentage':inhibitory_percentage,
    #            'r_fit_linmodel':r_fit_linmodel,
    #            'SR':SR, 'SR_std':SR_std, 'SR_av_std':SR_av_std}
    #
    return {'thresh': thresh * val['resp'].fs,
            'EP_A': excitatory_percentage['A'],
            'EP_B': excitatory_percentage['B'],
            #            'EP_C':excitatory_percentage['C'],
            'EP_I': excitatory_percentage['I'],
            'IP_A': inhibitory_percentage['A'],
            'IP_B': inhibitory_percentage['B'],
            #            'IP_C':inhibitory_percentage['C'],
            'IP_I': inhibitory_percentage['I'],
            'OEP_A': excitatory_percentage_onset['A'],
            'OEP_B': excitatory_percentage_onset['B'],
            #            'OEP_C':excitatory_percentage_onset['C'],
            'OEP_I': excitatory_percentage_onset['I'],
            'OIP_A': inhibitory_percentage_onset['A'],
            'OIP_B': inhibitory_percentage_onset['B'],
            #            'OIP_C':inhibitory_percentage_onset['C'],
            'OIP_I': inhibitory_percentage_onset['I'],
            'Max_A': Max['A'],
            'Max_B': Max['B'],
            #            'Max_C':Max['C'],
            'Max_I': Max['I'],
            'Mean_A': Mean['A'],
            'Mean_B': Mean['B'],
            #            'Mean_C':Mean['C'],
            'Mean_I': Mean['I'],
            'OMax_A': Max_onset['A'],
            'OMax_B': Max_onset['B'],
            #            'OMax_C':Max_onset['C'],
            'OMax_I': Max_onset['I'],
            'TotalMax': TotalMax * val['resp'].fs,
            'SinglesMax': SinglesMax * val['resp'].fs,
            #            'r_lin_C':r_fit_linmodel['C'],
            'r_lin_I': r_fit_linmodel['I'],
            #            'r_lin_C_NM':r_fit_linmodel_NM['C'],
            'r_lin_I_NM': r_fit_linmodel_NM['I'],
            #            'r_ceil_C':r_ceil_linmodel['C'],
            'r_ceil_I': r_ceil_linmodel['I'],
            #            'MEnh_C':mean_enh['C'],
            'MEnh_I': mean_enh['I'],
            #            'MSupp_C':mean_supp['C'],
            'MSupp_I': mean_supp['I'],
            #            'EnhP_C':EnhP['C'],
            #        'EnhP_I':EnhP['I'],
            #            'SuppP_C':SuppP['C'],
            #        'SuppP_I':SuppP['I'],
            #            'DualAboveZeroP_C':DualAboveZeroP['C'],
            #        'DualAboveZeroP_I':DualAboveZeroP['I'],
            #            'r_dual_A_C':r_dual_A['C'],
            'r_dual_A_I': r_dual_A['I'],
            #            'r_dual_B_C':r_dual_B['C'],
            'r_dual_B_I': r_dual_B['I'],
            #            'r_dual_A_C_nc':r_dual_A_nc['C'],
            'r_dual_A_I_nc': r_dual_A_nc['I'],
            #            'r_dual_B_C_nc':r_dual_B_nc['C'],
            'r_dual_B_I_nc': r_dual_B_nc['I'],
            #            'r_dual_A_C_bal':r_dual_A_bal['C'],
            'r_dual_A_I_bal': r_dual_A_bal['I'],
            #            'r_dual_B_C_bal':r_dual_B_bal['C'],
            'r_dual_B_I_bal': r_dual_B_bal['I'],
            #            'r_lin_A_C':r_lin_A['C'],
            'r_lin_A_I': r_lin_A['I'],
            #            'r_lin_B_C':r_lin_B['C'],
            'r_lin_B_I': r_lin_B['I'],
            #            'r_lin_A_C_nc':r_lin_A_nc['C'],
            'r_lin_A_I_nc': r_lin_A_nc['I'],
            #            'r_lin_B_C_nc':r_lin_B_nc['C'],
            'r_lin_B_I_nc': r_lin_B_nc['I'],
            #            'r_lin_A_C_bal':r_lin_A_bal['C'],
            'r_lin_A_I_bal': r_lin_A_bal['I'],
            #            'r_lin_B_C_bal':r_lin_B_bal['C'],
            'r_lin_B_I_bal': r_lin_B_bal['I'],
            'r_A_B': r_A_B,
            'r_A_B_nc': r_A_B_nc,
            'rAAm': rAAm, 'rBBm': rBBm,
            'rAA': rAA, 'rBB': rBB, 'rII': rII,
            #           'rCC':rCC
            'rAA_nc': rAA_nc, 'rBB_nc': rBB_nc,
            'mean_nsA': mean_nsA, 'mean_nsB': mean_nsB, 'min_nsA': min_nsA, 'min_nsB': min_nsB,
            'SR': SR, 'SR_std': SR_std, 'SR_av_std': SR_av_std,
            'norm_spont': norm_spont, 'spont_rate': spont_rate, 'params': params,
            'corcoef': corcoef, 'avg_resp': avg_resp, 'snr': snr,
            'pair_names': twostims, 'suppression': supp_array, 'FR': FR_array,
            'rec': rec,
            'animal': cellid[:3]}


def calc_psth_metrics_cuts(batch, cellid, parmfile=None, paths=None, cut_ids=None):
    start_win_offset = 0  # Time (in sec) to offset the start of the window used to calculate threshold, exitatory percentage, and inhibitory percentage
    if parmfile:
        manager = BAPHYExperiment(parmfile)
    else:
        manager = BAPHYExperiment(cellid=cellid, batch=batch)

    options = ohel.get_load_options(batch) #gets options that will include gtgram if batch=339
    rec = manager.get_recording(**options)

    area_df = db.pd_query(f"SELECT DISTINCT area FROM sCellFile where cellid like '{manager.siteid}%%'")
    area = area_df.area.iloc[0]

    if rec['resp'].epochs[rec['resp'].epochs['name'] == 'PASSIVE_EXPERIMENT'].shape[0] >= 2:
        # rec = ohel.remove_olp_test(rec)
        rec = remove_olp_test(rec)

    rec['resp'] = rec['resp'].extract_channels([cellid])
    resp = copy.copy(rec['resp'].rasterize())
    rec['resp'].fs = 100

    norm_spont, SR, STD = ohel.remove_spont_rate_std(resp)
    # params = ohel.get_expt_params(resp, manager, cellid)
    params = ohel.get_expt_params(resp, manager, cellid)


    epcs = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PreStimSilence'].copy()
    ep2 = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PostStimSilence'].iloc[0].copy()
    params['prestim'], params['poststim'] = epcs.iloc[0]['end'], ep2['end'] - ep2['start']
    params['lenstim'] = ep2['end']

    stim_epochs = ep.epoch_names_matching(resp.epochs, 'STIM_')

    if paths and cellid[:3] == 'TBR':
        print(f"Deprecated, run on {cellid} though...")
        stim_epochs, rec, resp = ohel.path_tabor_get_epochs(stim_epochs, rec, resp, params)

    epoch_repetitions = [resp.count_epoch(cc) for cc in stim_epochs]
    full_resp = np.empty((max(epoch_repetitions), len(stim_epochs),
                          (int(params['lenstim']) * rec['resp'].fs)))
    full_resp[:] = np.nan
    for cnt, epo in enumerate(stim_epochs):
        resps_list = resp.extract_epoch(epo)
        full_resp[:resps_list.shape[0], cnt, :] = resps_list[:, 0, :]

    #Calculate a few metrics
    corcoef = ohel.calc_base_reliability(full_resp)
    avg_resp = ohel.calc_average_response(full_resp, params)
    snr = compute_snr(resp)

    #Grab and label epochs that have two sounds in them (no null)
    presil, postsil = int(params['prestim'] * rec['resp'].fs), int(params['poststim'] * rec['resp'].fs)
    twostims = resp.epochs[resp.epochs['name'].str.count('-1') == 2].copy()
    ep_twostim = twostims.name.unique().tolist()
    ep_twostim.sort()

    ep_names = resp.epochs[resp.epochs['name'].str.contains('STIM_')].copy()
    ep_names = ep_names.name.unique().tolist()
    ep_types = list(map(ohel.label_ep_type, ep_names))
    ep_synth_type = list(map(ohel.label_synth_type, ep_names))

    ep_df = pd.DataFrame({'name': ep_names, 'type': ep_types, 'synth_type': ep_synth_type})

    cell_dff = []
    for cnt, stimmy in enumerate(ep_twostim):
        kind = ohel.label_pair_type(stimmy)
        # synth_kind = ohel.label_synth_type(stimmy)
        synth_kind = label_synth_type(stimmy)
        dynamic_kind = label_dynamic_ep_type(stimmy)
        # seps = (stimmy.split('_')[1], stimmy.split('_')[2])
        seps = list(re.findall("_(null|\d{2}.*)_(null|\d{2}.*)", stimmy)[0])
        BG, FG = seps[0].split('-')[0][2:], seps[1].split('-')[0][2:]

        Aepo, Bepo = 'STIM_' + seps[0] + '_null', 'STIM_null_' + seps[1]

        rAB = resp.extract_epoch(stimmy)
        rA, rB = resp.extract_epoch(Aepo), resp.extract_epoch(Bepo)

        fn = lambda x: np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - SR)
        rAsm = np.squeeze(np.apply_along_axis(fn, 2, rA))
        rBsm = np.squeeze(np.apply_along_axis(fn, 2, rB))
        rABsm = np.squeeze(np.apply_along_axis(fn, 2, rAB))

        rA_st, rB_st = rAsm[:, presil:-postsil], rBsm[:, presil:-postsil]
        rAB_st = rABsm[:, presil:-postsil]

        rAm, rBm = np.nanmean(rAsm, axis=0), np.nanmean(rBsm, axis=0)
        rABm = np.nanmean(rABsm, axis=0)

        AcorAB = np.corrcoef(rAm, rABm)[0, 1]  # Corr between resp to A and resp to dual
        BcorAB = np.corrcoef(rBm, rABm)[0, 1]  # Corr between resp to B and resp to dual

        A_FR, B_FR, AB_FR = np.nanmean(rA_st), np.nanmean(rB_st), np.nanmean(rAB_st)

        min_rep = np.min((rA.shape[0], rB.shape[0]))  # only will do something if SoundRepeats==Yes
        lin_resp = np.nanmean(rAsm[:min_rep, :] + rBsm[:min_rep, :], axis=0)
        supp = np.nanmean(lin_resp - AB_FR)

        AcorLin = np.corrcoef(rAm, lin_resp)[0, 1]  # Corr between resp to A and resp to lin
        BcorLin = np.corrcoef(rBm, lin_resp)[0, 1]  # Corr between resp to B and resp to lin

        Apref, Bpref = AcorAB - AcorLin, BcorAB - BcorLin
        pref = Apref - Bpref

        # If there are no cuts provided, just make one that takes everything.
        if not cut_ids:
            cut_ids = {'': np.full((int(params['lenstim'] * params['fs']),), True)}

        # Start the dict that becomes the df with universal things regardless of if cuts or not
        cell_dict = {'epoch': stimmy,
                        'kind': kind,
                        'synth_kind': synth_kind,
                        'dynamic_kind': dynamic_kind,
                        'BG': BG,
                        'FG': FG,
                        'AcorAB': AcorAB,
                        'BcorAB': BcorAB,
                        'AcorLin': AcorLin,
                        'BcorLin': BcorLin,
                        'pref': pref,
                        'Apref': Apref,
                        'Bpref': Bpref
                        }

        for lb, cut in cut_ids.items():
            cut_st = cut[presil:-postsil]
            rA_st_cut, rB_st_cut, rAB_st_cut = rA_st[:, cut_st], rB_st[:, cut_st], rAB_st[:, cut_st]
            rAsm_cut, rBsm_cut, rABsm_cut = rAsm[:, cut], rBsm[:, cut], rABsm[:, cut]

            # AcorAB = np.corrcoef(rAm_cut, rABm_cut)[0, 1]  # Corr between resp to A and resp to dual
            # BcorAB = np.corrcoef(rBm_cut, rABm_cut)[0, 1]  # Corr between resp to B and resp to dual

            A_FR, B_FR, AB_FR = np.nanmean(rA_st_cut), np.nanmean(rB_st_cut), np.nanmean(rAB_st_cut)

            min_rep = np.min((rA.shape[0], rB.shape[0])) #only will do something if SoundRepeats==Yes
            lin_resp = np.nanmean(rAsm_cut[:min_rep, :] + rBsm_cut[:min_rep, :], axis=0)
            supp = np.nanmean(lin_resp - AB_FR)

            # AcorLin = np.corrcoef(rAm_cut, lin_resp)[0, 1]  # Corr between resp to A and resp to lin
            # BcorLin = np.corrcoef(rBm_cut, lin_resp)[0, 1]  # Corr between resp to B and resp to lin

            # Apref, Bpref = AcorAB - AcorLin, BcorAB - BcorLin
            # pref = Apref - Bpref

            cell_dict[f"bg_FR{lb}"], cell_dict[f"fg_FR{lb}"], cell_dict[f"combo_FR{lb}"] = A_FR, B_FR, AB_FR
            # cell_dict[f"AcorAB{lb}"], cell_dict[f"BcorAB{lb}"] = AcorAB, BcorAB
            # cell_dict[f"AcorLin{lb}"], cell_dict[f"B_corLin{lb}"] = AcorLin, BcorLin
            # cell_dict[f"pref{lb}"], cell_dict[f"Apref{lb}"], cell_dict[f"Bpref{lb}"] = pref, Apref, Bpref
            cell_dict[f"supp{lb}"] = supp

        cell_dff.append(cell_dict)

        # if params['Binaural'] == 'Yes':
        #     dA, dB = ohel.get_binaural_adjacent_epochs(stimmy)
        #
        #     rdA, rdB = resp.extract_epoch(dA), resp.extract_epoch(dB)
        #     rdAm = np.nanmean(np.squeeze(np.apply_along_axis(fn, 2, rdA))[:, presil:-postsil], axis=0)
        #     rdBm = np.nanmean(np.squeeze(np.apply_along_axis(fn, 2, rdB))[:, presil:-postsil], axis=0)
        #
        #     ABcordA = np.corrcoef(rABm, rdAm)[0, 1]  # Corr between resp to AB and resp to BG swap
        #     ABcordB = np.corrcoef(rABm, rdBm)[0, 1]  # Corr between resp to AB and resp to FG swap

    cell_df = pd.DataFrame(cell_dff)
    cell_df['SR'], cell_df['STD'] = SR, STD
    # cell_df['corcoef'], cell_df['avg_resp'], cell_df['snr'] = corcoef, avg_resp, snr
    cell_df.insert(loc=0, column='area', value=area)

    return cell_df


def calc_psth_metrics_cuts(batch, cellid, parmfile=None, paths=None, cut_ids=None):
    '''2022_12_29. This is a version of the calc_psth_metrics that accounts for the fits that were snippets.'''
    start_win_offset = 0  # Time (in sec) to offset the start of the window used to calculate threshold, exitatory percentage, and inhibitory percentage
    if parmfile:
        manager = BAPHYExperiment(parmfile)
    else:
        manager = BAPHYExperiment(cellid=cellid, batch=batch)

    options = ohel.get_load_options(batch) #gets options that will include gtgram if batch=339
    rec = manager.get_recording(**options)

    area_df = db.pd_query(f"SELECT DISTINCT area FROM sCellFile where cellid like '{manager.siteid}%%'")
    area = area_df.area.iloc[0]

    # Search for things that start with FILE_
    if rec['resp'].epochs[rec['resp'].epochs['name'] == 'PASSIVE_EXPERIMENT'].shape[0] >= 2:
        print(f"Number of experiments is {rec['resp'].epochs[rec['resp'].epochs['name'] == 'PASSIVE_EXPERIMENT'].shape[0]}, which is more"
              f"than 1, going to remove the extra.")
        # rec = ohel.remove_olp_test(rec)
        rec = ohel.remove_olp_test_nonOLP(rec)
    else:
        print(f"Number of experiments is {rec['resp'].epochs[rec['resp'].epochs['name'] == 'PASSIVE_EXPERIMENT'].shape[0]}, all good.")



    rec['resp'] = rec['resp'].extract_channels([cellid])
    resp = copy.copy(rec['resp'].rasterize())
    rec['resp'].fs = 100

    norm_spont, SR, STD = ohel.remove_spont_rate_std(resp)
    params = ohel.get_expt_params(resp, manager, cellid)


    epcs = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PreStimSilence'].copy()
    ep2 = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PostStimSilence'].iloc[0].copy()
    params['prestim'], params['poststim'] = epcs.iloc[0]['end'], ep2['end'] - ep2['start']
    params['lenstim'] = ep2['end']

    stim_epochs = ep.epoch_names_matching(resp.epochs, 'STIM_')
    OLP_epochs = ep.epoch_names_matching(resp.epochs, '_OLP')


    if paths and cellid[:3] == 'TBR':
        print(f"Deprecated, run on {cellid} though...")
        stim_epochs, rec, resp = ohel.path_tabor_get_epochs(stim_epochs, rec, resp, params)

    epoch_repetitions = [resp.count_epoch(cc) for cc in stim_epochs]
    full_resp = np.empty((max(epoch_repetitions), len(stim_epochs),
                          (int(params['lenstim']) * rec['resp'].fs)))
    full_resp[:] = np.nan
    for cnt, epo in enumerate(stim_epochs):
        resps_list = resp.extract_epoch(epo)
        full_resp[:resps_list.shape[0], cnt, :] = resps_list[:, 0, :]

    #Calculate a few metrics
    corcoef = ohel.calc_base_reliability(full_resp)
    avg_resp = ohel.calc_average_response(full_resp, params)
    snr = compute_snr(resp)

    #Grab and label epochs that have two sounds in them (no null)
    presil, postsil = int(params['prestim'] * rec['resp'].fs), int(params['poststim'] * rec['resp'].fs)
    twostims = resp.epochs[(resp.epochs['name'].str.count('null') == 0) & (resp.epochs['name'].str.count('STIM_') == 1)].copy()
    ep_twostim = twostims.name.unique().tolist()
    ep_twostim.sort()

    ep_names = resp.epochs[resp.epochs['name'].str.contains('STIM_')].copy()
    ep_names = ep_names.name.unique().tolist()
    ep_types = list(map(ohel.label_ep_type, ep_names))
    ep_synth_type = list(map(ohel.label_synth_type, ep_names))

    ep_df = pd.DataFrame({'name': ep_names, 'type': ep_types, 'synth_type': ep_synth_type})

    cell_dff = []
    for cnt, stimmy in enumerate(ep_twostim):
        # kind = ohel.label_pair_type(stimmy)
        kind = ohel.label_ep_type(stimmy)
        # synth_kind = ohel.label_synth_type(stimmy)
        synth_kind = ohel.label_synth_type(stimmy)
        dynamic_kind = ohel.label_dynamic_ep_type(stimmy)
        # seps = (stimmy.split('_')[1], stimmy.split('_')[2])
        seps = list(re.findall("_(null|\d{2}.*)_(null|\d{2}.*)", stimmy)[0])
        BG, FG = seps[0].split('-')[0][2:], seps[1].split('-')[0][2:]

        Aepo, Bepo = 'STIM_' + seps[0] + '_null', 'STIM_null_' + seps[1]

        rAB = resp.extract_epoch(stimmy)
        rA, rB = resp.extract_epoch(Aepo), resp.extract_epoch(Bepo)

        fn = lambda x: np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - SR)
        rAsm = np.squeeze(np.apply_along_axis(fn, 2, rA))
        rBsm = np.squeeze(np.apply_along_axis(fn, 2, rB))
        rABsm = np.squeeze(np.apply_along_axis(fn, 2, rAB))

        rA_st, rB_st = rAsm[:, presil:-postsil], rBsm[:, presil:-postsil]
        rAB_st = rABsm[:, presil:-postsil]

        rAm, rBm = np.nanmean(rAsm, axis=0), np.nanmean(rBsm, axis=0)
        rABm = np.nanmean(rABsm, axis=0)

        AcorAB = np.corrcoef(rAm, rABm)[0, 1]  # Corr between resp to A and resp to dual
        BcorAB = np.corrcoef(rBm, rABm)[0, 1]  # Corr between resp to B and resp to dual

        A_FR, B_FR, AB_FR = np.nanmean(rA_st), np.nanmean(rB_st), np.nanmean(rAB_st)

        min_rep = np.min((rA.shape[0], rB.shape[0]))  # only will do something if SoundRepeats==Yes
        lin_resp = np.nanmean(rAsm[:min_rep, :] + rBsm[:min_rep, :], axis=0)
        supp = np.nanmean(lin_resp - AB_FR)

        AcorLin = np.corrcoef(rAm, lin_resp)[0, 1]  # Corr between resp to A and resp to lin
        BcorLin = np.corrcoef(rBm, lin_resp)[0, 1]  # Corr between resp to B and resp to lin

        Apref, Bpref = AcorAB - AcorLin, BcorAB - BcorLin
        pref = Apref - Bpref

        # If there are no cuts provided, just make one that takes everything.
        if not cut_ids:
            cut_ids = {'': np.full((int(params['lenstim'] * params['fs']),), True)}

        # Start the dict that becomes the df with universal things regardless of if cuts or not
        cell_dict = {'epoch': stimmy,
                        'kind': kind,
                        'synth_kind': synth_kind,
                        'dynamic_kind': dynamic_kind,
                        'BG': BG,
                        'FG': FG,
                        'AcorAB': AcorAB,
                        'BcorAB': BcorAB,
                        'AcorLin': AcorLin,
                        'BcorLin': BcorLin,
                        'pref': pref,
                        'Apref': Apref,
                        'Bpref': Bpref
                        }

        for lb, cut in cut_ids.items():
            cut_st = cut[presil:-postsil]
            rA_st_cut, rB_st_cut, rAB_st_cut = rA_st[:, cut_st], rB_st[:, cut_st], rAB_st[:, cut_st]
            rAsm_cut, rBsm_cut, rABsm_cut = rAsm[:, cut], rBsm[:, cut], rABsm[:, cut]

            # AcorAB = np.corrcoef(rAm_cut, rABm_cut)[0, 1]  # Corr between resp to A and resp to dual
            # BcorAB = np.corrcoef(rBm_cut, rABm_cut)[0, 1]  # Corr between resp to B and resp to dual

            A_FR, B_FR, AB_FR = np.nanmean(rA_st_cut), np.nanmean(rB_st_cut), np.nanmean(rAB_st_cut)

            min_rep = np.min((rA.shape[0], rB.shape[0])) #only will do something if SoundRepeats==Yes
            lin_resp = np.nanmean(rAsm_cut[:min_rep, :] + rBsm_cut[:min_rep, :], axis=0)
            supp = np.nanmean(lin_resp - AB_FR)

            # AcorLin = np.corrcoef(rAm_cut, lin_resp)[0, 1]  # Corr between resp to A and resp to lin
            # BcorLin = np.corrcoef(rBm_cut, lin_resp)[0, 1]  # Corr between resp to B and resp to lin

            # Apref, Bpref = AcorAB - AcorLin, BcorAB - BcorLin
            # pref = Apref - Bpref

            cell_dict[f"bg_FR{lb}"], cell_dict[f"fg_FR{lb}"], cell_dict[f"combo_FR{lb}"] = A_FR, B_FR, AB_FR
            # cell_dict[f"AcorAB{lb}"], cell_dict[f"BcorAB{lb}"] = AcorAB, BcorAB
            # cell_dict[f"AcorLin{lb}"], cell_dict[f"B_corLin{lb}"] = AcorLin, BcorLin
            # cell_dict[f"pref{lb}"], cell_dict[f"Apref{lb}"], cell_dict[f"Bpref{lb}"] = pref, Apref, Bpref
            cell_dict[f"supp{lb}"] = supp

        cell_dff.append(cell_dict)

        # if params['Binaural'] == 'Yes':
        #     dA, dB = ohel.get_binaural_adjacent_epochs(stimmy)
        #
        #     rdA, rdB = resp.extract_epoch(dA), resp.extract_epoch(dB)
        #     rdAm = np.nanmean(np.squeeze(np.apply_along_axis(fn, 2, rdA))[:, presil:-postsil], axis=0)
        #     rdBm = np.nanmean(np.squeeze(np.apply_along_axis(fn, 2, rdB))[:, presil:-postsil], axis=0)
        #
        #     ABcordA = np.corrcoef(rABm, rdAm)[0, 1]  # Corr between resp to AB and resp to BG swap
        #     ABcordB = np.corrcoef(rABm, rdBm)[0, 1]  # Corr between resp to AB and resp to FG swap

    cell_df = pd.DataFrame(cell_dff)
    cell_df['SR'], cell_df['STD'] = SR, STD
    # cell_df['corcoef'], cell_df['avg_resp'], cell_df['snr'] = corcoef, avg_resp, snr
    cell_df.insert(loc=0, column='area', value=area)

    return cell_df


def get_sep_stim_names(stim_name):
    '''Moved here 2022_12_29. From Luke's old stuff.'''
    seps = [m.start() for m in re.finditer('_(\d|n)', stim_name)]
    if len(seps) < 2 or len(seps) > 2:
        return None
    else:
        print(f"Returning {[stim_name[seps[0] + 1:seps[1]], stim_name[seps[1] + 1:]]}")
        return [stim_name[seps[0] + 1:seps[1]], stim_name[seps[1] + 1:]]


def calc_psth_weight_resp(row, do_plot=False, find_mse_confidence=False, fs=200, fit_type='Binaural'):
    print('load {}'.format(row.cellid))
    if fit_type == 'Binaural':
        fit_epoch = ['10', '01', '20', '02', '11', '12', '21', '22']
    elif fit_type == 'Synthetic':
        fit_epoch = ['N', 'C', 'T', 'S', 'U', 'M', 'A']
    else:
        fit_epoch = ['10', '01', '20', '02', '11', '12', '21', '22']
    modelspecs, est, val = load_TwoStim(int(row.batch),
                                        row.cellid,
                                        fit_epoch,
                                        None, fs=fs,
                                        get_est=False,
                                        get_stim=False)
    # smooth and subtract SR
    fn = lambda x: np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - row['SR'] / val[0]['resp'].fs)
    val[0]['resp'] = val[0]['resp'].transform(fn)

    print('calc weights')
    d = ts.calc_psth_weights_of_model_responses(val[0], signame='resp', do_plot=do_plot,
                                             find_mse_confidence=find_mse_confidence)
    d = {k + 'R': v for k, v in d.items()}
    for k, v in d.items():
        row[k] = v
    return row


def load_TwoStim(batch, cellid, fit_epochs, modelspec_name, loader='env100',
                 modelspecs_dir='/auto/users/luke/Code/nems/modelspecs', fs=100, get_est=True, get_stim=True,
                 paths=None):
    # load into a recording object
    if not get_stim:
        loadkey = 'ns.fs100'
    else:
        raise RuntimeError('Put stimuli in batch')
    manager = BAPHYExperiment(cellid=cellid, batch=batch)

    options = {'rasterfs': 100,
               'stim': False,
               'resp': True}
    rec = manager.get_recording(**options)

    rec['resp'].fs = fs
    rec['resp'] = rec['resp'].extract_channels([cellid])
    # ----------------------------------------------------------------------------
    # DATA PREPROCESSING
    #
    # GOAL: Split your data into estimation and validation sets so that you can
    #       know when your model exhibits overfitting.

    # Method #1: Find which stimuli have the most reps, use those for val
    #    if not get_stim:
    #        del rec.signals['stim']

    ##Added Greg 9/22/21 for

    stim_epochs = ep.epoch_names_matching(rec['resp'].epochs, 'STIM_')

    val = rec.copy()
    val['resp'] = val['resp'].rasterize()
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')

    if get_est:
        raise RuntimeError('Fix me')
        df0 = est['resp'].epochs.copy()
        df2 = est['resp'].epochs.copy()
        df0['name'] = df0['name'].apply(parse_stim_type)
        df0 = df0.loc[df0['name'].notnull()]
        df3 = pd.concat([df0, df2])

        est['resp'].epochs = df3
        est_sub = copy.deepcopy(est)
        est_sub['resp'] = est_sub['resp'].select_epochs(fit_epochs)
    else:
        est_sub = None

    df0 = val['resp'].epochs.copy()
    df2 = val['resp'].epochs.copy()
    # df0['name'] = df0['name'].apply(ts.parse_stim_type)
    if fit_epochs == ['10', '01', '20', '02', '11', '12', '21', '22']:
        df0['name'] = df0['name'].apply(ohel.label_ep_type)
    elif fit_epochs == ['N', 'C', 'T', 'S', 'U', 'M', 'A']:
        df0['name'] = df0['name'].apply(ohel.label_synth_type)

    df0 = df0.loc[df0['name'].notnull()]
    df3 = pd.concat([df0, df2])

    val['resp'].epochs = df3
    val_sub = copy.deepcopy(val)
    val_sub['resp'] = val_sub['resp'].select_epochs(fit_epochs)

    # ----------------------------------------------------------------------------
    # GENERATE SUMMARY STATISTICS

    if modelspec_name is None:
        return None, [est_sub], [val_sub]
    else:
        fit_epochs_str = "+".join([str(x) for x in fit_epochs])
        mn = loader + '_subset_' + fit_epochs_str + '.' + modelspec_name
        an_ = modelspecs_dir + '/' + cellid + '/' + mn
        an = glob.glob(an_ + '*')
        if len(an) > 1:
            warnings.warn('{} models found, loading an[0]:{}'.format(len(an), an[0]))
            an = [an[0]]
        if len(an) == 1:
            filepath = an[0]
            modelspecs = [ms.load_modelspec(filepath)]
            modelspecs[0][0]['meta']['modelname'] = mn
            modelspecs[0][0]['meta']['cellid'] = cellid
        else:
            raise RuntimeError('not fit')
        # generate predictions
        est_sub, val_sub = nems0.analysis.api.generate_prediction(est_sub, val_sub, modelspecs)
        est_sub, val_sub = nems0.analysis.api.generate_prediction(est_sub, val_sub, modelspecs)

        return modelspecs, est_sub, val_sub


def fit_weights(df, batch, fs=100):
    df['batch'] = batch

    df_fit = df[['cellid', 'SR', 'batch']].copy()
    df_fit = df_fit.drop_duplicates(subset=['cellid'])

    df0 = df_fit.apply(calc_psth_weight_resp, axis=1, fs=fs)

    def drop_get_error(row):
        row['weight_dfR'] = row['weight_dfR'].copy().drop(columns=['get_error', 'Efit'])
        return row

    df0 = df0.copy().drop(columns='get_nrmseR')
    df0 = df0.apply(drop_get_error, axis=1)

    weight_df = pd.concat(df0['weight_dfR'].values, keys=df0.cellid).reset_index(). \
        drop(columns='level_1')
    ep_names = [f"STIM_{aa}_{bb}" for aa, bb in zip(weight_df.namesA, weight_df.namesB)]
    weight_df = weight_df.drop(columns=['namesA', 'namesB'])
    weight_df['epoch'] = ep_names

    weights_df = pd.merge(right=weight_df, left=df, on=['cellid', 'epoch'])
    if df.shape[0] != weights_df.shape[0] or weight_df.shape[0] != weights_df.shape[0]:
        raise ValueError("Resulting weights_df does not match length of parts, some epochs were dropped.")

    return weights_df


def calc_psth_weights_of_model_responses_list(val, names, signame='resp',
                                              get_nrmse_fn=False, cuts=None):
    '''Moved from OLP_analysis_main on 2022_10_03. I don't know what this is for. It is possible
    it is from TwoStim Helpers and no longer has a use. There are no uses or implementations
    in my code, but it all looks like things I would use...'''
    sig1 = np.concatenate([val[signame].extract_epoch(n).squeeze()[cuts] for n in names[0]])
    sig2 = np.concatenate([val[signame].extract_epoch(n).squeeze()[cuts] for n in names[1]])
    # sig_SR=np.ones(sig1.shape)
    sigO = np.concatenate([val[signame].extract_epoch(n).squeeze()[cuts] for n in names[2]])

    # fsigs=np.vstack((sig1,sig2,sig_SR)).T
    fsigs = np.vstack((sig1, sig2)).T
    ff = np.all(np.isfinite(fsigs), axis=1) & np.isfinite(sigO)
    close_to_zero = np.array([np.allclose(fsigs[ff, i], 0, atol=1e-17) for i in (0, 1)])
    if all(close_to_zero):
        # Both input signals have all their values close to 0. Set weights to 0.
        weights = np.zeros(2)
        rank = 1
    elif any(close_to_zero):
        weights_, residual_sum, rank, singular_values = np.linalg.lstsq(np.expand_dims(fsigs[ff, ~close_to_zero], 1),
                                                                        sigO[ff], rcond=None)
        weights = np.zeros(2)
        weights[~close_to_zero] = weights_
    else:
        weights, residual_sum, rank, singular_values = np.linalg.lstsq(fsigs[ff, :], sigO[ff], rcond=None)
        # residuals = ((sigO[ff]-(fsigs[ff,:]*weights).sum(axis=1))**2).sum()

    # calc CC between weight model and actual response
    pred = np.dot(weights, fsigs[ff, :].T)
    cc = np.corrcoef(pred, sigO[ff])
    r_weight_model = cc[0, 1]

    # norm_factor = np.std(sigO[ff])
    norm_factor = np.mean(sigO[ff] ** 2)

    if rank == 1:
        min_nMSE = 1
        min_nRMSE = 1
    else:
        # min_nrmse = np.sqrt(residual_sum[0]/ff.sum())/norm_factor
        pred = np.dot(weights, fsigs[ff, :].T)
        min_nRMSE = np.sqrt(((sigO[ff] - pred) ** 2).mean()) / np.sqrt(
            norm_factor)  # minimim normalized root mean squared error
        min_nMSE = ((sigO[ff] - pred) ** 2).mean() / norm_factor  # minimim normalized mean squared error

    # create NMSE caclulator for later
    if get_nrmse_fn:
        def get_nrmse(weights=weights):
            pred = np.dot(weights, fsigs[ff, :].T)
            nrmse = np.sqrt(((pred - sigO[ff]) ** 2).mean(axis=-1)) / norm_factor
            return nrmse
    else:
        get_nrmse = np.nan

    weights[close_to_zero] = np.nan
    return weights, np.nan, min_nMSE, norm_factor, get_nrmse, r_weight_model#, get_error

### From OLP_analysis_main on 2022_10_03. I don't think this has any use, but stashing it here in
### case... If you don't miss it by the next time you come across this, just trash it.
# def get_sep_stim_names(stim_name):
#     seps = [m.start() for m in re.finditer('_(\d|n)', stim_name)]
#     if len(seps) < 2 or len(seps) > 2:
#         return None
#     else:
#         return [stim_name[seps[0] + 1:seps[1]], stim_name[seps[1] + 1:]]
#
# weight_list = []
# batch = 339
# fs = 100
# lfreq, hfreq, bins = 100, 24000, 48
# threshold = 0.75
# cell_df = nd.get_batch_cells(batch)
# cell_list = cell_df['cellid'].tolist()
# cell_list = ohel.manual_fix_units(cell_list) #So far only useful for two TBR cells
#
# fit_epochs = ['10', '01', '20', '02', '11', '12', '21', '22']
# loader = 'env100'
# modelspecs_dir = '/auto/users/luke/Code/nems/modelspecs'
#
# for cellid in cell_list:
#     loadkey = 'ns.fs100'
#     manager = BAPHYExperiment(cellid=cellid, batch=batch)
#     options = {'rasterfs': 100,
#                'stim': False,
#                'resp': True}
#     rec = manager.get_recording(**options)
#
#     #GET sound envelopes and get the indices for chopping?
#     expt_params = manager.get_baphy_exptparams()
#     ref_handle = expt_params[-1]['TrialObject'][1]['ReferenceHandle'][1]
#     FG_folder, fgidx = ref_handle['FG_Folder'], list(set(ref_handle['Foreground']))
#     fgidx.sort(key=int)
#     idxstr = [str(ff).zfill(2) for ff in fgidx]
#
#     fg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
#                            f'{FG_folder}/{ff}*.wav'))[0] for ff in idxstr]
#     fgname = [ff.split('/')[-1].split('.')[0].replace(' ', '') for ff in fg_paths]
#     ep_fg = [f"STIM_null_{ff}" for ff in fgname]
#
#     prebins = int(ref_handle['PreStimSilence'] * options['rasterfs'])
#     postbins = int(ref_handle['PostStimSilence'] * options['rasterfs'])
#     durbins = int(ref_handle['Duration'] * options['rasterfs'])
#     trialbins = durbins + postbins
#
#     env_cuts = {}
#     for nm, pth in zip(fgname, fg_paths):
#         sfs, W = wavfile.read(pth)
#         spec = gtgram(W, sfs, 0.02, 0.01, bins, lfreq, hfreq)
#
#         env = np.nanmean(spec, axis=0)
#         cutoff = np.max(env) * threshold
#
#         # aboves = np.squeeze(np.argwhere(env >= cutoff))
#         # belows = np.squeeze(np.argwhere(env < cutoff))
#
#         highs, lows, whole_thing = env >= cutoff, env < cutoff, env > 0
#         prestimFalse = np.full((prebins,), False)
#         poststimTrue = np.full((trialbins - len(env),), True)
#         poststimFalse = np.full((trialbins - len(env),), False)
#
#         full = np.concatenate((prestimFalse, np.full((trialbins,), True)))
#         aboves = np.concatenate((prestimFalse, highs, poststimFalse))
#         belows = np.concatenate((prestimFalse, lows, poststimFalse))
#         belows_post = np.concatenate((prestimFalse, lows, poststimTrue))
#
#         env_cuts[nm] = [full, aboves, belows, belows_post]
#
#         f, ax = plt.subplots(3, 1, sharex=True, sharey=True)
#         ax[0].plot(env)
#         ax[0].hlines(cutoff, 0, 100, ls=':')
#         ax[0].set_title(f"{nm}")
#         ax[1].plot(env[aboves])
#         ax[2].plot(env[belows])
#
#     rec['resp'].fs = fs
#     rec['resp'] = rec['resp'].extract_channels([cellid])
#     resp = copy.copy(rec['resp'].rasterize())
#
#     _, SR, _ = ohel.remove_spont_rate_std(resp)
#
#     stim_epochs = ep.epoch_names_matching(rec['resp'].epochs, 'STIM_')
#
#     val = rec.copy()
#     val['resp'] = val['resp'].rasterize()
#     val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')
#
#     est_sub = None
#
#     df0 = val['resp'].epochs.copy()
#     df2 = val['resp'].epochs.copy()
#     df0['name'] = df0['name'].apply(ohel.label_ep_type)
#     df0 = df0.loc[df0['name'].notnull()]
#     df3 = pd.concat([df0, df2])
#
#     val['resp'].epochs = df3
#     val_sub = copy.deepcopy(val)
#     val_sub['resp'] = val_sub['resp'].select_epochs(fit_epochs)
#
#     val = val_sub
#     fn = lambda x: np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - SR / rec['resp'].fs)
#     val['resp'] = val['resp'].transform(fn)
#
#     print(f'calc weights {cellid}')
#
#     #where twostims fit actually begins
#     epcs = val.epochs[val.epochs['name'].str.count('-0-1') >= 1].copy()
#     sepname = epcs['name'].apply(get_sep_stim_names)
#     epcs['nameA'] = [x[0] for x in sepname.values]
#     epcs['nameB'] = [x[1] for x in sepname.values]
#
#     # epochs with two sounds in them
#     epcs_twostim = epcs[epcs['name'].str.count('-0-1') == 2].copy()
#
#     A, B, AB, sepnames = ([], [], [], [])  # re-defining sepname
#     for i in range(len(epcs_twostim)):
#         if any((epcs['nameA'] == epcs_twostim.iloc[i].nameA) & (epcs['nameB'] == 'null')) \
#                 and any((epcs['nameA'] == 'null') & (epcs['nameB'] == epcs_twostim.iloc[i].nameB)):
#             A.append('STIM_' + epcs_twostim.iloc[i].nameA + '_null')
#             B.append('STIM_null_' + epcs_twostim.iloc[i].nameB)
#             AB.append(epcs_twostim['name'].iloc[i])
#             sepnames.append(sepname.iloc[i])
#
#     #Calculate weights
#     subsets = len(list(env_cuts.values())[0])
#     weights = np.zeros((2, len(AB), subsets))
#     Efit = np.zeros((5,len(AB), subsets))
#     nMSE = np.zeros((len(AB), subsets))
#     nf = np.zeros((len(AB), subsets))
#     r = np.zeros((len(AB), subsets))
#     cut_len = np.zeros((len(AB), subsets-1))
#     get_error=[]
#
#     for i in range(len(AB)):
#         names=[[A[i]],[B[i]],[AB[i]]]
#         Fg = names[1][0].split('_')[2].split('-')[0]
#         cut_list = env_cuts[Fg]
#
#         for ss, cut in enumerate(cut_list):
#             weights[:,i,ss], Efit[:,i,ss], nMSE[i,ss], nf[i,ss], _, r[i,ss], _ = \
#                     calc_psth_weights_of_model_responses_list(val, names,
#                                                               signame='resp', cuts=cut)
#             if ss != 0:
#                 cut_len[i, ss-1] = np.sum(cut)
#             # get_error.append(ge)
#
#     if subsets == 4:
#         weight_df = pd.DataFrame(
#             [epcs_twostim['nameA'].values, epcs_twostim['nameB'].values,
#              weights[0, :, 0], weights[1, :, 0], nMSE[:, 0], nf[:, 0], r[:, 0],
#              weights[0, :, 1], weights[1, :, 1], nMSE[:, 1], nf[:, 1], r[:, 1], cut_len[:,0],
#              weights[0, :, 2], weights[1, :, 2], nMSE[:, 2], nf[:, 2], r[:, 2], cut_len[:,1],
#              weights[0, :, 3], weights[1, :, 3], nMSE[:, 3], nf[:, 3], r[:, 3], cut_len[:,2],])
#         weight_df = weight_df.T
#         weight_df.columns = ['namesA', 'namesB', 'weightsA', 'weightsB', 'nMSE', 'nf', 'r',
#                              'weightsA_h', 'weightsB_h', 'nMSE_h', 'nf_h', 'r_h', 'h_idxs',
#                              'weightsA_l', 'weightsB_l', 'nMSE_l', 'nf_l', 'r_l', 'l_idxs',
#                              'weightsA_lp', 'weightsB_lp', 'nMSE_lp', 'nf_lp', 'r_lp', 'lp_idxs']
#         cols = ['namesA', 'namesB', 'weightsA', 'weightsB', 'nMSE']
#         print(weight_df[cols])
#
#         weight_df = weight_df.astype({'weightsA': float, 'weightsB': float,
#                                       'weightsA_h': float, 'weightsB_h': float,
#                                       'weightsA_l': float, 'weightsB_l': float,
#                                       'weightsA_lp': float, 'weightsB_lp': float,
#                                       'nMSE': float, 'nf': float, 'r': float,
#                                       'nMSE_h': float, 'nf_h': float, 'r_h': float,
#                                       'nMSE_l': float, 'nf_l': float, 'r_l': float,
#                                       'nMSE_lp': float, 'nf_lp': float, 'r_lp': float,
#                                       'h_idxs': float, 'l_idxs': float, 'lp_idxs': float})
#
#     else:
#         raise ValueError(f"Only {subsets} subsets. You got lazy and didn't make this part"
#                          f"flexible yet.")
#
#
#     weight_df.insert(loc=0, column='cellid', value=cellid)
#
#     weight_list.append(weight_df)
#
# weight_df0 = pd.concat(weight_list)
#
#
# ep_names = [f"STIM_{aa}_{bb}" for aa, bb in zip(weight_df0.namesA, weight_df0.namesB)]
# weight_df0 = weight_df0.drop(columns=['namesA', 'namesB'])
# weight_df0['epoch'] = ep_names
#
# weight_df0 = pd.merge(right=weight_df0, left=df, on=['cellid', 'epoch'])
# weight_df0['threshold'] = str(int(threshold * 100))
# if df.shape[0] != weights_df.shape[0] or weight_df.shape[0] != weights_df.shape[0]:
#     raise ValueError("Resulting weights_df does not match length of parts, some epochs were dropped.")
#
# ##load here.
# OLP_partialweights_db_path = '/auto/users/hamersky/olp_analysis/Binaural_OLP_full_partial_weights20.h5'  # weight + corr
# OLP_partialweights_db_path = '/auto/users/hamersky/olp_analysis/Binaural_OLP_full_partial_weights.h5'  # weight + corr
#
# part_weights = False
# if part_weights == True:
#     os.makedirs(os.path.dirname(OLP_partialweights_db_path),exist_ok=True)
#     store = pd.HDFStore(OLP_partialweights_db_path)
#     df_store=copy.deepcopy(weight_df0)
#     store['df'] = df_store.copy()
#     store.close()
#
# else:
#     store = pd.HDFStore(OLP_partialweights_db_path)
#     weight_df0=store['df']
#     store.close()


def get_parm_data(cellid,runclassid=128):
    '''2023_04_04. Added by SVD and amended by GRH. Takes a given cellid and finds the parmfiles for OLP runs. It then uses
    parameters of those files to label features of the parmfiles that will be useful later. Primarily, the goal is to get
    the run_kind parameter which indicates which runs are tests vs real OLPs so that in the main function I can only load
    the signal from the real OLPs'''
    sql=f"SELECT * FROM sCellFile WHERE cellid='{cellid}' AND runclassid={runclassid}"
    parmfiles = nd.pd_query(sql)
    parmfiles[['parmfile', 'run_kind', 'olp_type', 'RMS', 'ramp', 'SNR']] = ''

    for i,r in parmfiles.iterrows():
        ddata = nd.get_data_parms(rawid=r.rawid)
        parmdict = {r_['name']: r_['value'].strip() if type(r_['value']) is str else r_['value'] for i_,r_ in ddata.iterrows()}
        parmfiles['parmfile'] = r['stimpath']+r['stimfile']

        # Sort out if the run of OLP is a test, one of Stephen's (vowel), or a good. If good, get additional parameters for sorting.
        if parmdict.get('Ref_Combos','No') == 'No':
            parmfiles.loc[i,'run_kind'] = 'test'
        elif parmdict['Ref_BG_Folder'] == 'Background4':
            parmfiles.loc[i,'run_kind'] = 'vowel'
        else:
            parmfiles.loc[i,'run_kind']= 'real'

            # If it is a real run of OLP, then figure out what type (basic, binaural, dynamic, synthetic) it is
            if parmdict.get('Ref_Binaural', 'No') == 'Yes':
                parmfiles.loc[i,'olp_type'] = 'binaural'
            elif parmdict.get('Ref_Synthetic', 'No') == 'C-T-S-ST-STM' or parmdict.get('Ref_Synthetic', 'No') == 'C-T-S-ST':
                parmfiles.loc[i,'olp_type'] = 'synthetic'
            elif parmdict.get('Ref_Dynamic', 'No') == 'Yes':
                parmfiles.loc[i,'olp_type'] = 'dynamic'
            # This is before I add a dynamic parameter, back when dynamic was default, so if binaural and synthetic are set to
            # No, or not yet added (default to No), then set the Dynamic parameter to a random value that indicates Yes
            elif parmdict.get('Ref_Binaural', 'No') == 'No' and parmdict.get('Ref_Synthetic', 'No') == 'No' and parmdict.get('Ref_Dynamic', 'dne') == 'dne':
                parmfiles.loc[i,'olp_type'] = 'dynamic'
            # A situation after I've added the option to have dynamic but say No, which would make it the basic run
            elif parmdict['Ref_Binaural'] == 'No' and parmdict['Ref_Synthetic'] == 'No' and parmdict['Ref_Dynamic'] == 'No':
                parmfiles.loc[i,'olp_type'] = 'basic'
            else:
                raise ValueError(f"For {cellid}, {r['respfile']} none of your elifs to define 'olp_type' worked. "
                                 f"You either forgot something or this file is weird, check it out.")

            # Determine if ramp was used or not
            if parmdict.get('Ref_Ramp', 'No') == 'No':
                parmfiles.loc[i,'ramp'] = 'No'
            else:
                parmfiles.loc[i, 'ramp'] = parmdict['Ref_Ramp']

            # Determine if RMS balancing was used
            if parmdict.get('Ref_NormalizeRMS', 'No') == 'No':
                parmfiles.loc[i,'RMS'] = 'No'
            else:
                # Even in the case of me choosing "Some", it only has an effect when synthetic is chosen, otherwise, the code
                # just has everything RMS balanced.
                parmfiles.loc[i, 'RMS'] = 'Yes'

            # Determine if this is an SNR condition
            if parmdict.get('Ref_SNR', 0) == 0:
                parmfiles.loc[i,'SNR'] = 0
            else:
                parmfiles.loc[i, 'SNR'] = parmdict['Ref_SNR']

    return parmfiles



def OLP_fit_cell_pred_individual(cellid, batch, threshold=None, snip=None, pred=False, fit_epos='syn',
                                 fs=100):
    '''2023_04_06. Added many things. First off the loader that uses get_parm_data and gives you labeled parmfiles
    that are associated with different kinds of olp runs. The function will fit individually for all the good ones it
    finds and add one big, labeled df at the end that it'll save for every run associated with that unit. Also,
    the new area + layer thing is used.
    2022_12_29. Added this from the bigger funxtion to make one that works for each cellid and can
    divvy it up to the cluster.'''
    if fit_epos == 'bin':
        fit_epochs, lbl_fn = ['10', '01', '20', '02', '11', '12', '21', '22'], ohel.label_ep_type
    elif fit_epos == 'syn':
        fit_epochs, lbl_fn = ['N', 'C', 'T', 'S', 'U', 'M', 'A'], ohel.label_synth_type
    elif fit_epos == 'dyn':
        fit_epochs, lbl_fn = ['ff', 'fh', 'fn', 'hf', 'hh', 'hn', 'nf', 'nh'], ohel.label_dynamic_ep_type

    # cellid = 'PRN013c-235-1'
    parms = get_parm_data(cellid, runclassid=128)

    real_olps = parms.loc[parms['run_kind']=='real'].reset_index(drop=True)
    olp_parms = real_olps['parmfile'].to_list()

    area_info = baphy_io.get_depth_info(cellid=cellid)
    layer, area = area_info['layer'].values[0], area_info['area'].values[0]

    # baphy_io.get_depth_info(siteid='PRN013a')

    weight_df_list = []
    for cc, pf in enumerate(olp_parms):
        manager = BAPHYExperiment(cellid=cellid, parmfile=pf)
        # manager = BAPHYExperiment(cellid=cellid, batch=batch)

        area_df = db.pd_query(f"SELECT DISTINCT cellid,area FROM sCellFile where cellid like '{manager.siteid}%%'")

        if pred == True:
            modelname = "gtgram.fs100.ch18-ld.pop-norm.l1-sev.fOLP_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_prefit.b322.f.nf-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4"
            xf, ctx = load_model_xform(cellid=cellid, batch=batch, modelname=modelname)
            rec = ctx['val']
            rec['pred'].chans = rec['resp'].chans

        else:
            options = {'rasterfs': 100,
                       'stim': False,
                       'resp': True}
            rec = manager.get_recording(**options)

        expt_params = manager.get_baphy_exptparams()

        ref_handle = expt_params[0]['TrialObject'][1]['ReferenceHandle'][1]
        ref_handle = {key: val.strip() if type(val) is str else val for key, val in ref_handle.items()}

        prebins = int(ref_handle['PreStimSilence'] * fs)
        postbins = int(ref_handle['PostStimSilence'] * fs)
        durbins = int(ref_handle['Duration'] * fs)
        trialbins = durbins + postbins

        if threshold:
            lfreq, hfreq, bins = 100, 24000, 48
            FG_folder, fgidx = ref_handle['FG_Folder'], list(set(ref_handle['Foreground']))
            fgidx.sort(key=int)
            idxstr = [str(ff).zfill(2) for ff in fgidx]

            fg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                   f'{FG_folder}/{ff}*.wav'))[0] for ff in idxstr]
            fgname = [ff.split('/')[-1].split('.')[0].replace(' ', '') for ff in fg_paths]
            ep_fg = [f"STIM_null_{ff}" for ff in fgname]

            env_cuts = {}
            for nm, pth in zip(fgname, fg_paths):
                sfs, W = wavfile.read(pth)
                spec = gtgram(W, sfs, 0.02, 0.01, bins, lfreq, hfreq)

                env = np.nanmean(spec, axis=0)
                cutoff = np.max(env) * threshold

                highs, lows, whole_thing = env >= cutoff, env < cutoff, env > 0
                prestimFalse = np.full((prebins,), False)
                poststimTrue = np.full((trialbins - len(env),), True)
                poststimFalse = np.full((trialbins - len(env),), False)  ## Something is wrong here with the lengths

                full = np.concatenate((prestimFalse, np.full((trialbins,), True)))
                aboves = np.concatenate((prestimFalse, highs, poststimFalse))
                belows = np.concatenate((prestimFalse, lows, poststimFalse))
                belows_post = np.concatenate((prestimFalse, lows, poststimTrue))

                env_cuts[nm] = [full, aboves, belows, belows_post]

                f, ax = plt.subplots(3, 1, sharex=True, sharey=True)
                ax[0].plot(env)
                ax[0].hlines(cutoff, 0, 100, ls=':')
                ax[0].set_title(f"{nm}")
                ax[1].plot(env[highs])
                ax[2].plot(env[lows])

                cut_labels = ['', '_h', '_l', '_lp']

        if snip:
            start, dur = int(snip[0] * fs), int(snip[1] * fs)
            prestimFalse = np.full((prebins,), False)
            # poststimTrue = np.full((trialbins - len(env),), True)
            poststimFalse = np.full((trialbins - durbins), False)
            # if start == dur:
            #
            # else:
            end = durbins - start - dur
            goods = [False] * start + [True] * dur + [False] * end
            bads = [not ll for ll in goods]

            full = np.concatenate((prestimFalse, np.full((trialbins,), True)))
            goods = np.concatenate((prestimFalse, goods, poststimFalse))
            bads = np.concatenate((prestimFalse, bads, poststimFalse))
            full_nopost = np.concatenate((prestimFalse, np.full((durbins,), True), poststimFalse))
            cut_list = [full, goods, bads, full_nopost]
            cut_labels = ['', '_start', '_end', '_nopost']

        rec['resp'].fs = fs
        rec['resp'] = rec['resp'].extract_channels([cellid])
        resp = copy.copy(rec['resp'].rasterize())

        if pred == True:
            rec['pred'].fs = fs
            rec['pred'] = rec['pred'].extract_channels([cellid])

        _, SR, _ = ohel.remove_spont_rate_std(resp)

        stim_epochs = ep.epoch_names_matching(rec['resp'].epochs, 'STIM_')

        val = rec.copy()
        val['resp'] = val['resp'].rasterize()
        val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')

        est_sub = None

        df0 = val['resp'].epochs.copy()
        df2 = val['resp'].epochs.copy()
        # change this shit
        # if synth == True:
        #     df0['name'] = df0['name'].apply(ohel.label_synth_type)
        # else:
        # df0['name'] = df0['name'].apply(ohel.label_ep_type)
        df0['name'] = df0['name'].apply(lbl_fn)

        df0 = df0.loc[df0['name'].notnull()]
        df3 = pd.concat([df0, df2])

        val['resp'].epochs = df3
        val_sub = copy.deepcopy(val)
        val_sub['resp'] = val_sub['resp'].select_epochs(fit_epochs)

        val = val_sub
        fn = lambda x: np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - SR / rec['resp'].fs)
        val['resp'] = val['resp'].transform(fn)

        print(f'calc weights {cellid}')

        # where twostims fit actually begins
        ## This is where adapt to fitting only the half stimuli
        # The old way of doing this replaced with regex 2023_04_05
        # epcs = val.epochs[val.epochs['name'].str.count('-0-1') >= 1].copy()
        # sepname = epcs['name'].apply(get_sep_stim_names)
        # epcs['nameA'] = [x[0] for x in sepname.values]
        # epcs['nameB'] = [x[1] for x in sepname.values]

        epcs = val.epochs[val.epochs['name'].str.contains("_(null|\d{2}.*)_(null|\d{2}.*)", regex=True, na=False)].copy()
        sepname = epcs['name'].apply(get_sep_stim_names)
        epcs['nameA'] = [x[0] for x in sepname.values]
        epcs['nameB'] = [x[1] for x in sepname.values]

        # epochs with two sounds in them
        # epcs_twostim = epcs[epcs['name'].str.count('-0-1') == 2].copy()
        epcs_twostim = epcs[epcs['name'].str.contains("_(\d{2}.*)_(\d{2}.*)", regex=True, na=False)].copy()

        A, B, AB, sepnames = ([], [], [], [])  # re-defining sepname
        for i in range(len(epcs_twostim)):
            if any((epcs['nameA'] == epcs_twostim.iloc[i].nameA) & (epcs['nameB'] == 'null')) \
                    and any((epcs['nameA'] == 'null') & (epcs['nameB'] == epcs_twostim.iloc[i].nameB)):
                A.append('STIM_' + epcs_twostim.iloc[i].nameA + '_null')
                B.append('STIM_null_' + epcs_twostim.iloc[i].nameB)
                AB.append(epcs_twostim['name'].iloc[i])
                sepnames.append(sepname.iloc[i])

        # Calculate weights
        if snip:
            subsets = len(cut_list)
            signames = ['resp'] * subsets
            if pred == True:
                subsets = subsets * 2
                signames = signames + (['pred'] * len(signames))
                cut_labels = cut_labels + ([aa + '_pred' for aa in cut_labels])
                cut_list = cut_list * 2
        # Figure out this dumb envelope thing you made
        # else:
        #     subsets = len(list(env_cuts.values())[0])
        weights = np.zeros((2, len(AB), subsets))
        Efit = np.zeros((5, len(AB), subsets))
        nMSE = np.zeros((len(AB), subsets))
        nf = np.zeros((len(AB), subsets))
        r = np.zeros((len(AB), subsets))
        cut_len = np.zeros((len(AB), subsets - 1))

        for i in range(len(AB)):
            names = [[A[i]], [B[i]], [AB[i]]]
            for ss, (cut, sig) in enumerate(zip(cut_list, signames)):
                weights[:, i, ss], Efit[:, i, ss], nMSE[i, ss], nf[i, ss], _, r[i, ss] = \
                    calc_psth_weights_of_model_responses_list(val, names,
                                                                   signame=sig, cuts=cut)
                if ss != 0:
                    cut_len[i, ss - 1] = np.sum(cut)


        # If this part is working the above code is useless.
        # Makes a list of lists that iterates through the arrays you created, then flattens them in the next line
        big_list = [[weights[0, :, ee], weights[1, :, ee], nMSE[:, ee], nf[:, ee], r[:, ee]] for ee in range(len(cut_list))]
        flat_list = [item for sublist in big_list for item in sublist]
        small_list = [epcs_twostim['nameA'].values, epcs_twostim['nameB'].values]
        # Combines the lists into a format that is conducive to the dataframe format I want to make
        bigger_list = small_list + flat_list
        weight_df = pd.DataFrame(bigger_list)
        weight_df = weight_df.T

        # Automatically generates a list of column names based on the names of the subsets provided above
        column_labels1 = ['namesA', 'namesB']
        column_labels2 = [[f"weightsA{cl}", f"weightsB{cl}", f"nMSE{cl}", f"nf{cl}", f"r{cl}"] for cl in cut_labels]
        column_labels_flat = [item for sublist in column_labels2 for item in sublist]
        column_labels = column_labels1 + column_labels_flat
        # Renames the columns according to that list - should work for any scenario as long as you specific names above
        weight_df.columns = column_labels1 + column_labels_flat

        # Not sure why I need this, I guess some may not be floats, so just doing it
        col_dict = {ii: float for ii in column_labels_flat}
        weight_df = weight_df.astype(col_dict)

        weight_df.insert(loc=0, column='cellid', value=cellid)
        weight_df['fit_segment'] = f"{int(snip[0] * 1000)}-{int((snip[0] + snip[1]) * 1000)}"
        row = real_olps.iloc[cc]

        weight_df['olp_type'], weight_df['RMS'], weight_df['ramp'], weight_df['SNR'] = row['olp_type'], row['RMS'], row['ramp'], row['SNR']
        weight_df['area'], weight_df['layer'] = area, layer

        weight_df_list.append(weight_df)

    final_weight_df = pd.concat(weight_df_list)

    OLP_partialweights_db_path = f'/auto/users/hamersky/cache/{cellid}}'  # weight + corr
    # OLP_partialweights_db_path = f'/auto/users/hamersky/cache/{cellid}_{real_olps.iloc[cc]["olp_type"]}'  # weight + corr
    os.makedirs(os.path.dirname(OLP_partialweights_db_path), exist_ok=True)

    jl.dump(final_weight_df, OLP_partialweights_db_path)

    return final_weight_df, OLP_partialweights_db_path


def OLP_fit_partial_weights(batch, threshold=None, snip=None, pred=False, fit_epos='syn',
                            fs=100, filter_animal=None, filter_experiment=None, note=None):
    '''2022_12_28. This is an updated version of the OLP_fit that allows you to either use the sound
    envelope of the FG to fit the model (use threshold) or snip out different parts of the stimuli
    to fit (use snip, [start, length] ie [0, 0.5] for 0-500, 500-1000, and whole thing to be fit
    separately). Pred being toggled on only works if you are thusfar using batch 341 that has a
    separate signal in rec called 'pred' that is Stephen using NAT/BNT data to fit OLP. Usually
    leave fit_epos to syn, I think I have made it such that it doesn't matter which toggle that is
    on. Also has the option to do some basic filtering of cell_list that comes from the batch, either
    by animal or a certain experiment number. Note will simply add that string to the file that will
    be saved by this function.'''
    weight_list = []

    cell_df = nd.get_batch_cells(batch)
    cell_list = cell_df['cellid'].tolist()
    cell_list = ohel.manual_fix_units(cell_list)  # So far only useful for two TBR cells

    #Only CLT synth units
    if filter_animal:
        cell_list = [cell for cell in cell_list if cell.split('-')[0][:3]==filter_animal]
    if filter_experiment:
        if filter_experiment[0] == '>':
            cell_list = [cell for cell in cell_list if cell_list.split('-')[0][:3] > filter_experiment[1]]
    # cell_list = [cell for cell in cell_list if (cell.split('-')[0][:3]=='CLT') & (int(cell.split('-')[0][3:6]) < 26)]
    # cell_list = cell_list[:2]


    # loader = 'env100'
    # modelspecs_dir = '/auto/users/luke/Code/nems/modelspecs'

    for cellid in cell_list:

        if fit_epos == 'bin':
            fit_epochs, lbl_fn = ['10', '01', '20', '02', '11', '12', '21', '22'], ohel.label_ep_type
        elif fit_epos == 'syn':
            fit_epochs, lbl_fn = ['N', 'C', 'T', 'S', 'U', 'M', 'A'], ohel.label_synth_type
        elif fit_epos == 'dyn':
            fit_epochs, lbl_fn = ['ff', 'fh', 'fn', 'hf', 'hh', 'hn', 'nf', 'nh'], ohel.label_dynamic_ep_type

        manager = BAPHYExperiment(cellid=cellid, batch=batch)
        if pred == True:
            modelname = "gtgram.fs100.ch18-ld.pop-norm.l1-sev.fOLP_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_prefit.b322.f.nf-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4"
            xf, ctx = load_model_xform(cellid=cellid, batch=batch, modelname=modelname)
            rec = ctx['val']
            rec['pred'].chans = rec['resp'].chans

        else:
            loadkey = 'ns.fs100'
            options = {'rasterfs': 100,
                       'stim': False,
                       'resp': True}
            rec = manager.get_recording(**options)

        expt_params = manager.get_baphy_exptparams()
        whichparams = [aa for aa in range(len(expt_params)) if expt_params[aa]['runclass']=='OLP']
        if len(whichparams) == 1:
            ref_handle = expt_params[whichparams[0]]['TrialObject'][1]['ReferenceHandle'][1]
        else:
            print(f"There are {len(whichparams)} OLPs for {cellid}, using the last one.")
            ref_handle = expt_params[whichparams[-1]]['TrialObject'][1]['ReferenceHandle'][1]

        prebins = int(ref_handle['PreStimSilence'] * fs)
        postbins = int(ref_handle['PostStimSilence'] * fs)
        durbins = int(ref_handle['Duration'] * fs)
        trialbins = durbins + postbins

        if threshold:
            lfreq, hfreq, bins = 100, 24000, 48
            labels = ['', '_h', '_l', '_lp']
            FG_folder, fgidx = ref_handle['FG_Folder'], list(set(ref_handle['Foreground']))
            fgidx.sort(key=int)
            idxstr = [str(ff).zfill(2) for ff in fgidx]

            fg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                   f'{FG_folder}/{ff}*.wav'))[0] for ff in idxstr]
            fgname = [ff.split('/')[-1].split('.')[0].replace(' ', '') for ff in fg_paths]
            ep_fg = [f"STIM_null_{ff}" for ff in fgname]


            env_cuts = {}
            for nm, pth in zip(fgname, fg_paths):
                sfs, W = wavfile.read(pth)
                spec = gtgram(W, sfs, 0.02, 0.01, bins, lfreq, hfreq)

                env = np.nanmean(spec, axis=0)
                cutoff = np.max(env) * threshold

                # aboves = np.squeeze(np.argwhere(env >= cutoff))
                # belows = np.squeeze(np.argwhere(env < cutoff))

                highs, lows, whole_thing = env >= cutoff, env < cutoff, env > 0
                prestimFalse = np.full((prebins,), False)
                poststimTrue = np.full((trialbins - len(env),), True)
                poststimFalse = np.full((trialbins - len(env),), False) ## Something is wrong here with the lengths

                full = np.concatenate((prestimFalse, np.full((trialbins,), True)))
                aboves = np.concatenate((prestimFalse, highs, poststimFalse))
                belows = np.concatenate((prestimFalse, lows, poststimFalse))
                belows_post = np.concatenate((prestimFalse, lows, poststimTrue))

                env_cuts[nm] = [full, aboves, belows, belows_post]

                f, ax = plt.subplots(3, 1, sharex=True, sharey=True)
                ax[0].plot(env)
                ax[0].hlines(cutoff, 0, 100, ls=':')
                ax[0].set_title(f"{nm}")
                ax[1].plot(env[highs])
                ax[2].plot(env[lows])

                cut_labels = ['', '_h', '_l', '_lp']

        if snip:
            start, dur = int(snip[0]*fs), int(snip[1]*fs)
            prestimFalse = np.full((prebins,), False)
            # poststimTrue = np.full((trialbins - len(env),), True)
            poststimFalse = np.full((trialbins - durbins), False)
            end = durbins - start - dur
            goods = [False]*start + [True]*dur + [False]*end
            bads = [not ll for ll in goods]

            full = np.concatenate((prestimFalse, np.full((trialbins,), True)))
            goods = np.concatenate((prestimFalse, goods, poststimFalse))
            bads = np.concatenate((prestimFalse, bads, poststimFalse))
            full_nopost = np.concatenate((prestimFalse, np.full((durbins,), True), poststimFalse))
            cut_list = [full, goods, bads, full_nopost]
            cut_labels = ['', '_start', '_end', '_nopost']

        rec['resp'].fs = fs
        rec['resp'] = rec['resp'].extract_channels([cellid])
        resp = copy.copy(rec['resp'].rasterize())

        if pred == True:
            rec['pred'].fs = fs
            rec['pred'] = rec['pred'].extract_channels([cellid])

        _, SR, _ = ohel.remove_spont_rate_std(resp)

        val = rec.copy()
        val['resp'] = val['resp'].rasterize()
        val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')

        df0 = val['resp'].epochs.copy()
        df2 = val['resp'].epochs.copy()
        df0['name'] = df0['name'].apply(lbl_fn)

        df0 = df0.loc[df0['name'].notnull()]
        df3 = pd.concat([df0, df2])

        val['resp'].epochs = df3
        val_sub = copy.deepcopy(val)
        val_sub['resp'] = val_sub['resp'].select_epochs(fit_epochs)

        val = val_sub
        fn = lambda x: np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - SR / rec['resp'].fs)
        val['resp'] = val['resp'].transform(fn)

        print(f'calc weights {cellid}')

        #where twostims fit actually begins
        ## This is where adapt to fitting only the half stimuli
        epcs = val.epochs[val.epochs['name'].str.count('-0-1') >= 1].copy()
        sepname = epcs['name'].apply(get_sep_stim_names)
        epcs['nameA'] = [x[0] for x in sepname.values]
        epcs['nameB'] = [x[1] for x in sepname.values]

        # epochs with two sounds in them
        epcs_twostim = epcs[epcs['name'].str.count('-0-1') == 2].copy()

        A, B, AB, sepnames = ([], [], [], [])  # re-defining sepname
        for i in range(len(epcs_twostim)):
            if any((epcs['nameA'] == epcs_twostim.iloc[i].nameA) & (epcs['nameB'] == 'null')) \
                    and any((epcs['nameA'] == 'null') & (epcs['nameB'] == epcs_twostim.iloc[i].nameB)):
                A.append('STIM_' + epcs_twostim.iloc[i].nameA + '_null')
                B.append('STIM_null_' + epcs_twostim.iloc[i].nameB)
                AB.append(epcs_twostim['name'].iloc[i])
                sepnames.append(sepname.iloc[i])

        #Calculate weights
        if snip:
            subsets = len(cut_list)
            cuts_info = {cut_labels[i]: cut_list[i] for i in range(len(cut_list))}
            signames = ['resp'] * subsets
            if pred == True:
                subsets = subsets * 2
                signames = signames + (['pred'] * len(signames))
                cut_labels = cut_labels + ([aa + '_pred' for aa in cut_labels])
                cut_list = cut_list * 2
        # Figure out this dumb envelope thing you made
        # else:
        #     subsets = len(list(env_cuts.values())[0])
        weights = np.zeros((2, len(AB), subsets))
        Efit = np.zeros((5,len(AB), subsets))
        nMSE = np.zeros((len(AB), subsets))
        nf = np.zeros((len(AB), subsets))
        r = np.zeros((len(AB), subsets))
        cut_len = np.zeros((len(AB), subsets-1))

        # if synth:
        for i in range(len(AB)):
            names = [[A[i]], [B[i]], [AB[i]]]
            for ss, (cut, sig) in enumerate(zip(cut_list, signames)):
                weights[:, i, ss], Efit[:, i, ss], nMSE[i, ss], nf[i, ss], _, r[i, ss] = \
                    calc_psth_weights_of_model_responses_list(val, names,
                                                              signame=sig, cuts=cut)
                if ss != 0:
                    cut_len[i, ss - 1] = np.sum(cut)


        # Makes a list of lists that iterates through the arrays you created, then flattens them in the next line
        big_list = [[weights[0, :, ee], weights[1, :, ee], nMSE[:, ee], nf[:, ee], r[:, ee]] for ee in range(len(cut_list))]
        flat_list = [item for sublist in big_list for item in sublist]
        small_list = [epcs_twostim['nameA'].values, epcs_twostim['nameB'].values]
        #Combines the lists into a format that is conducive to the dataframe format I want to make
        bigger_list = small_list + flat_list
        weight_df = pd.DataFrame(bigger_list)
        weight_df = weight_df.T

        #Automatically generates a list of column names based on the names of the subsets provided above
        column_labels1 = ['namesA', 'namesB']
        column_labels2 = [[f"weightsA{cl}", f"weightsB{cl}", f"nMSE{cl}", f"nf{cl}", f"r{cl}"] for cl in cut_labels]
        column_labels_flat = [item for sublist in column_labels2 for item in sublist]
        column_labels = column_labels1 + column_labels_flat
        #Renames the columns according to that list - should work for any scenario as long as you specific names above
        weight_df.columns = column_labels1 + column_labels_flat

        #Not sure why I need this, I guess some may not be floats, so just doing it
        col_dict = {ii: float for ii in column_labels_flat}
        weight_df = weight_df.astype(col_dict)

        weight_df.insert(loc=0, column='cellid', value=cellid)
        weight_list.append(weight_df)

    weight_df0 = pd.concat(weight_list)

    ep_names = [f"STIM_{aa}_{bb}" for aa, bb in zip(weight_df0.namesA, weight_df0.namesB)]
    weight_df0 = weight_df0.drop(columns=['namesA', 'namesB'])
    weight_df0['epoch'] = ep_names

    weight_df0['fit_segment'] = f"{int(snip[0] * 1000)}-{int((snip[0] + snip[1]) * 1000)}"

    if not note:
        note = f'Batch{batch}'
    OLP_partialweights_db_path = \
        f'/auto/users/hamersky/olp_analysis/{date.today()}_{note}_{weight_df0.fit_segment.unique()[0]}_nometrics.h5'
    print(f"Saving as {OLP_partialweights_db_path}")
    os.makedirs(os.path.dirname(OLP_partialweights_db_path), exist_ok=True)
    store = pd.HDFStore(OLP_partialweights_db_path)
    df_store = copy.deepcopy(weight_df0)
    store['df'] = df_store.copy()
    store.close()

    return weight_df0, cuts_info
