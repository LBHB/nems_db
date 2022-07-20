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

import nems_lbhb.projects.olp.binaural_OLP_helpers as bnh
log = logging.getLogger(__name__)
import nems_lbhb.fitEllipse as fE

import nems0.db as nd  # NEMS database functions -- NOT celldb
import nems_lbhb.baphy as nb  # baphy-specific functions
import nems_lbhb.xform_wrappers as nw  # wrappers for calling nems code with database stuff
import nems0.recording as recording
import SPO_helpers as sp
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


log = logging.getLogger(__name__)
from nems import db

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
    ep_df = pd.DataFrame({'name': ep_names, 'type': ep_types})

    cell_df = []
    for cnt, stimmy in enumerate(ep_twostim):
        kind = ohel.label_pair_type(stimmy)
        seps = (stimmy.split('_')[1], stimmy.split('_')[2])
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


def calc_psth_weight_resp(row, do_plot=False, find_mse_confidence=False, fs=200, paths=None):
    print('load {}'.format(row.cellid))
    modelspecs, est, val = load_TwoStim(int(row.batch),
                                        row.cellid,
                                        ['10', '01', '20', '02', '11', '12', '21', '22'],
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
    df0['name'] = df0['name'].apply(ohel.label_ep_type)
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

