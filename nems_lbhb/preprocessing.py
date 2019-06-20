#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

nems_lbhb. state initializers

Created on Fri Aug 31 12:50:49 2018

@author: svd
"""
import logging
import re
import numpy as np

import nems.epoch as ep
import nems.signal as signal

log = logging.getLogger(__name__)

def append_difficulty(rec, **kwargs):

    newrec = rec.copy()

    newrec['puretone_trials'] = resp.epoch_to_signal('PURETONE_BEHAVIOR')
    newrec['puretone_trials'].chans = ['puretone_trials']
    newrec['easy_trials'] = resp.epoch_to_signal('EASY_BEHAVIOR')
    newrec['easy_trials'].chans = ['easy_trials']
    newrec['hard_trials'] = resp.epoch_to_signal('HARD_BEHAVIOR')
    newrec['hard_trials'].chans = ['hard_trials']


def mask_high_repetion_stims(rec, epoch_regex='^STIM_'):
    full_rec = rec.copy()
    stims = (full_rec.epochs['name'].value_counts() >= 8)
    stims = [stims.index[i] for i, s in enumerate(stims) if bool(re.search(epoch_regex, stims.index[i])) and s == True]
    if len(stims) == 0:
        raise ValueError("Fewer than min reps found for all stim")
        max_counts = full_rec.epochs['name'].value_counts().max()
        stims = (full_rec.epochs['name'].value_counts() >= max_counts)
    if 'mask' not in full_rec.signals.keys():
        full_rec = full_rec.create_mask(True)
    full_rec = full_rec.and_mask(stims)

    return full_rec


def pupil_mask(est, val, condition, balance):
    """
    Create pupil mask by epoch (use REF by default) - so entire epoch is
    classified as big or small. Perform the mask on both est and val sets
    separately. This is so that both test metrics and fit metrics are
    evaluated on the same class of data (big or small pupil)
    """
    full_est = est.copy()
    full_val = val.copy()
    new_est_val = []
    for i, r in enumerate([full_est, full_val]):
        pupil_data = r['pupil'].extract_epoch('REFERENCE')
        pupil_data = np.tile(np.nanmean(pupil_data, axis=-1),
                             [1, pupil_data.shape[-1]])[:, np.newaxis, :]
        pup_median = np.median(pupil_data.flatten()[~np.isnan(pupil_data.flatten())])

        if condition == 'large':
            mask = ((pupil_data > pup_median) & (~np.isnan(pupil_data)))
            op_mask = ((pupil_data <= pup_median) & (~np.isnan(pupil_data)))
        elif condition == 'small':
            mask = ((pupil_data <= pup_median) & (~np.isnan(pupil_data)))
            op_mask = ((pupil_data > pup_median) & (~np.isnan(pupil_data)))

        # perform AND mask with existing mask
        if 'mask' in est.signals:
            mask = (mask & r['mask'].extract_epoch('REFERENCE'))
            op_mask = (op_mask & r['mask'].extract_epoch('REFERENCE'))
        elif 'mask' not in est.signals:
            pass

        r['mask'] = r['mask'].replace_epochs({'REFERENCE': mask})

        if (i == 1) & (balance == True):
            # balance epochs between big / small pupil conditions for the val
            # set in order to make sure r_test for big / small pupil fits is
            # comparable
            log.info("balancing REF epochs between big and small pupil in val")
            s1 = r.copy()
            s2 = r.copy()
            s1 = s1.apply_mask(reset_epochs=True)
            s2['mask'] = s2['mask'].replace_epochs({'REFERENCE': op_mask})
            s2 = s2.apply_mask(reset_epochs=True)

            val = val.apply_mask(reset_epochs=True)
            stims = np.unique([ep for ep in val.epochs.name if 'STIM' in ep])
            ntot = len(stims)
            balanced_stims = []
            for stim in stims:
                big_reps = np.sum([str(ep) == stim for ep in s1.epochs.name])
                small_reps = np.sum([str(ep) == stim for ep in s2.epochs.name])
                if abs(big_reps - small_reps) <= 2:
                    balanced_stims.append(stim)
            balanced_stims = [str(ep) for ep in balanced_stims]
            log.info("keeping {0}/{1} val epochs".format(len(balanced_stims), ntot))

            r = r.and_mask(balanced_stims)

        new_est_val.append(r)

    return (new_est_val[0], new_est_val[1])


def mask_tor(rec):
    full_rec = rec.copy()
    eps = [ep for ep in full_rec.epochs.name if ('TOR' in ep) & ('FILE' in ep)]
    full_rec = full_rec.create_mask(True)
    full_rec = full_rec.and_mask(eps)

    return full_rec


def mask_runclass(rec, runclass="NAT"):
    full_rec = rec.copy()
    eps = [ep for ep in full_rec.epochs.name if ep.endswith(runclass) & ('FILE' in ep)]
    full_rec = full_rec.and_mask(eps)

    return full_rec


def mask_nat(rec):
    full_rec = rec.copy()
    eps = [ep for ep in full_rec.epochs.name if ('NAT' in ep) & ('FILE' in ep)]
    full_rec = full_rec.create_mask(True)
    full_rec = full_rec.and_mask(eps)

    return full_rec


def mask_subset_by_epoch(rec,epoch_list):
    full_rec = rec.copy()
    full_rec = full_rec.or_mask(epoch_list)
    return full_rec


def getPrePostSilence(sig):
    """
    Figure out Pre- and PostStimSilence (units of time bins) for a signal

    input:
        sig : Signal (required)
    returns
        PreStimSilence, PostStimSilence : integers

    """
    d = sig.get_epoch_bounds('PreStimSilence')
    if d.size > 0:
        PreStimSilence = np.mean(np.diff(d))
    else:
        PreStimSilence = 0

    PostStimSilence = 0
    d = sig.get_epoch_bounds('PostStimSilence')
    if d.size > 0:
        dd = np.diff(d)
        dd = dd[dd > 0]

        if dd.size > 0:
            PostStimSilence = np.min(dd)

    return PreStimSilence, PostStimSilence


def normalizePrePostSilence(rec, PreStimSilence=0.5, PostStimSilence=0.5):
    """
    Shorten pre- and post-stim silence to specified valeues

    input:
        sig : Signal (required)
        PreStimSilence : float
        PostStimSilence : float
    returns
        sig : modified signal

    """
    sig = rec.signals[list(rec.signals.keys())[0]]
    fs = sig.fs
    PreStimSilence0, PostStimSilence0 = getPrePostSilence(sig)
    epochs = sig.epochs.copy()

    if PreStimSilence0 != PreStimSilence:
        if PreStimSilence0 < PreStimSilence-1/fs:
            raise Warning('Adjusting PreStimSilence to be longer than in orginal signal')
        d = sig.get_epoch_bounds('PreStimSilence')
        for e in d:
            ee = (epochs['start'] == e[0])
            epochs.loc[ee, 'start'] = epochs.loc[ee, 'start'] + PreStimSilence0 - PreStimSilence

    if PostStimSilence0 != PostStimSilence:
        if PostStimSilence0 < PostStimSilence-1/fs:
            raise Warning('Adjusting PostStimSilence to be longer than in orginal signal')
        d = sig.get_epoch_bounds('PostStimSilence')
        for e in d:
            ee = (epochs['end'] == e[0])
            epochs.loc[ee, 'end'] = epochs.loc[ee, 'end'] - PostStimSilence0 + PostStimSilence

    new_rec = rec.copy()
    for k in rec.signals.keys():
        new_rec.signals[k].epochs = epochs

    return new_rec


def hi_lo_psth_jack(est=None, val=None, rec=None, **kwargs):

    for e, v in zip(est, val):
        r = hi_lo_psth(rec=e, **kwargs)
        e.add_signal(r['rec']['psth'])
        v.add_signal(r['rec']['psth'])

    return {'est': est, 'val': val}


def hi_lo_psth(rec=None, resp_signal='resp', state_signal='state',
               state_channel='pupil', psth_signal='psth',
               epoch_regex="^STIM_", smooth_resp=False, **kwargs):
    '''
    Like nems.preprocessing.generate_psth_from_resp() but generates two PSTHs,
    one each for periods when state_channel is higher or lower than its
    median.

    subtract spont rate based on pre-stim silence for ALL estimation data.

    if rec['mask'] exists, uses rec['mask'] == True to determine valid epochs
    '''

    newrec = rec.copy()
    resp = newrec[resp_signal].rasterize()
    presec, postsec = getPrePostSilence(resp)
    prebins=int(presec * resp.fs)
    postbins=int(postsec * resp.fs)

    state_chan_idx = newrec[state_signal].chans.index(state_channel)

    # extract relevant epochs
    epochs_to_extract = ep.epoch_names_matching(resp.epochs, epoch_regex)
    folded_matrices = resp.extract_epochs(epochs_to_extract,
                                          mask=newrec['mask'])
    folded_state = newrec[state_signal].extract_epochs(epochs_to_extract,
                                                       mask=newrec['mask'])
    for k, v in folded_state.items():
        m = np.nanmean(folded_state[k][:,state_chan_idx,:prebins],
                       axis=1, keepdims=True)
        folded_state[k][:,state_chan_idx,:] = m
#
#    # determine median of state variable for splitting
#    all_state=[]
#    for k, v in folded_state.items():
#        m = np.nanmean(folded_state[k][:,state_chan_idx,:], axis=1)
#        all_state.append(v[:,state_chan_idx,:])
#        folded_state[k] = m
#    all_state = np.concatenate(all_state, axis=0)
#    med = np.nanmedian(all_state)
#    print("median of state var {} : {}".format(state_channel, med))
    # compute spont rate during valid (non-masked) trials
    prestimsilence = resp.extract_epoch('PreStimSilence', mask=newrec['mask'])
    prestimstate = newrec[state_signal].extract_epoch('PreStimSilence',
                                                      mask=newrec['mask'])
#    if 'mask' in newrec.signals.keys():
#        prestimmask = np.tile(newrec['mask'].extract_epoch('PreStimSilence'),
#                              [1, nCells, 1])
#        prestimsilence[prestimmask == False] = np.nan
#        prestimstate[prestimmask[:,0,0] == False,:,:] = np.nan
    prestimstate = np.nanmean(prestimstate[:,state_chan_idx,:], axis=-1)
    med = np.nanmedian(prestimstate)
    print("median of pre state var {} : {}".format(state_channel, med))

    if len(prestimsilence.shape) == 3:
        spont_rate_lo = np.nanmean(prestimsilence[prestimstate<med,:,:], axis=(0, 2))
        spont_rate_hi = np.nanmean(prestimsilence[prestimstate>=med,:,:], axis=(0, 2))
    else:
        spont_rate_lo = np.nanmean(prestimsilence[prestimstate<med,:,:])
        spont_rate_hi = np.nanmean(prestimsilence[prestimstate>=med,:,:])


    # 2. Average over all reps of each stim and save into dict called psth.
    per_stim_psth_lo = dict()
    per_stim_psth_hi = dict()

    for k, v in folded_matrices.items():
        if smooth_resp:
            # replace each epoch (pre, during, post) with average
            v[:, :, :prebins] = np.nanmean(v[:, :, :prebins],
                                           axis=2, keepdims=True)
            v[:, :, prebins:(prebins+2)] = np.nanmean(v[:, :, prebins:(prebins+2)],
                                                      axis=2, keepdims=True)
            v[:, :, (prebins+2):-postbins] = np.nanmean(v[:, :, (prebins+2):-postbins],
                                                        axis=2, keepdims=True)
            v[:, :, -postbins:(-postbins+2)] = np.nanmean(v[:, :, -postbins:(-postbins+2)],
                                                          axis=2, keepdims=True)
            v[:, :, (-postbins+2):] = np.nanmean(v[:, :, (-postbins+2):],
                                                 axis=2, keepdims=True)

        hi = (folded_state[k][:,state_chan_idx,0] >= med)
        lo = np.logical_not(hi)
        per_stim_psth_hi[k] = np.nanmean(v[hi,:,:], axis=0) - spont_rate_hi[:, np.newaxis]
        per_stim_psth_lo[k] = np.nanmean(v[lo,:,:], axis=0) - spont_rate_lo[:, np.newaxis]

    # 3. Invert the folding to unwrap the psth into a predicted spike_dict by
    #   replacing all epochs in the signal with their average (psth)
    psthlo = resp.replace_epochs(per_stim_psth_lo)
    psthhi = resp.replace_epochs(per_stim_psth_hi)
    psth = psthlo.concatenate_channels([psthlo,psthhi])
    psth.name = 'psth'
    #print(per_stim_psth_lo[k].shape)
    #print(folded_state[k].shape)
    #state = newrec[state_signal].replace_epochs(folded_state)
    #print(state.shape)

    # add signal to the recording
    newrec.add_signal(psth)
    #newrec.add_signal(state)

    if smooth_resp:
        log.info('Replacing resp with smoothed resp')
        resp = resp.replace_epochs(folded_matrices)
        newrec.add_signal(resp)

    return {'rec': newrec}


def transform_stim_envelope(rec=None):
    '''
    Collapse over frequency channels
    '''
    newrec = rec.copy()
    stimSig = newrec['stim'].rasterize()
    stim = np.sum(stimSig.as_continuous(), 0)
    #stim = stim ** 2
    newrec['stim'] = stimSig._modified_copy(stim[np.newaxis, :])

    return newrec
