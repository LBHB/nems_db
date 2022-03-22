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
import copy
import nems.epoch as ep
import nems.signal as signal
import scipy.fftpack as fp
import scipy.signal as ss
from scipy.ndimage import gaussian_filter1d
import pickle
import pandas as pd

from nems.preprocessing import mask_incorrect, generate_average_sig, normalize_epoch_lengths
from nems.epoch import epoch_names_matching
import nems.db as nd


log = logging.getLogger(__name__)


def append_difficulty(rec, **kwargs):

    newrec = rec.copy()

    newrec['puretone_trials'] = resp.epoch_to_signal('PURETONE_BEHAVIOR')
    newrec['puretone_trials'].chans = ['puretone_trials']
    newrec['easy_trials'] = resp.epoch_to_signal('EASY_BEHAVIOR')
    newrec['easy_trials'].chans = ['easy_trials']
    newrec['hard_trials'] = resp.epoch_to_signal('HARD_BEHAVIOR')
    newrec['hard_trials'].chans = ['hard_trials']

def fix_cpn_epochs(rec, sequence_only=False, use_old=False, **kwargs):
    """
    Specialized preprocessor for CPN data to make the epoch names match more 
    "traditional" baphy format.

    Also, some specialized tweaking of epochs to work with NAT
    sounds decoding anlaysis

    crh - 05.08.2021
    """
    newrec = copy.deepcopy(rec)
    new_epochs = copy.deepcopy(newrec.epochs)

    # Name the "trial" pre/post differently to distinguish from each STIM
    prepost = new_epochs.name.str.startswith('PreStimSilence') | new_epochs.name.str.startswith('PostStimSilence')
    new_epochs.loc[prepost, 'name'] = ['TRIAL'+s for s in new_epochs[prepost].name]

    if not sequence_only:
        # strip the seq. epochs and sub pre/post stim
        new_epochs = new_epochs.loc[~new_epochs.name.str.contains('_sequence')]

        # remove "sub" labels -- sub masks finds SubPreStimSilence and SubPostStimSilence
        sub = new_epochs.name.str.startswith('Sub')
        new_epochs.loc[sub, 'name'] = [s.strip('Sub') for s in new_epochs[sub].name]

        # Clean up the actual sound epochs
        stim_mask = new_epochs.name.str.startswith('STIM_')
        new_epochs.loc[stim_mask, 'name'] = [s.split('context:')[0][:-1] for s in new_epochs[stim_mask].name]

        # Chop out first bin of each (to remove weird context effects) -- (and for the "dummy" prestim silence)
        one_bin = np.float(1 / rec['resp'].fs)
        new_epochs.loc[stim_mask, 'start'] = new_epochs.loc[stim_mask, 'start'].values + one_bin
        new_epochs.loc[sub, 'start'] = new_epochs.loc[sub, 'start'].values + one_bin
        # looks like was never actually dealing with prestim silence correctly, which resulted
        # in prestim silence epochs with negative length. Oops. Fixing that now -- crh 06.28.2021
        # reason this happened is that sub pre/post stim are length zero for this data, so I forgot
        # to update the end timestamp too.
        if not use_old:
            new_epochs.loc[sub, 'end'] = new_epochs.loc[sub, 'end'].values + one_bin

    else:
        new_epochs = new_epochs[~new_epochs.name.str.contains('_probe')]
        new_epochs = new_epochs[~new_epochs.name.str.startswith('SubPreStimSilence')]
        new_epochs = new_epochs[~new_epochs.name.str.startswith('SubPostStimSilence')]

        # remove "Trial" silence prefixes
        sub = new_epochs.name.str.startswith('TRIALP')
        new_epochs.loc[sub, 'name'] = [s.strip('TRIAL') for s in new_epochs[sub].name]

    # strip old references and make new ones
    ref = new_epochs.name.str.startswith('REFERENCE')
    new_epochs = new_epochs.loc[~ref]
    stim_mask = new_epochs.name.str.startswith('STIM_')
    stim_epochs = new_epochs.loc[stim_mask].copy()
    stim_epochs.name = 'REFERENCE'
    new_epochs = pd.concat([new_epochs, stim_epochs])

    # get rid of float point error duplicates, Thiw was an anoying bug
    new_epochs.loc[:, ['start', 'end']] = new_epochs.loc[:, ['start', 'end']].values.round(decimals=12)
    new_epochs.drop_duplicates(inplace=True)

    # clean up index
    new_epochs = new_epochs.sort_values(
            by=['start', 'end'], ascending=[1, 0]
            ).reset_index(drop=True)

    # assign to new recording
    # newrec.epochs = new_epochs
    for signal in newrec.signals.keys():
        newrec[signal].epochs = new_epochs

    return newrec

#
# BUNCH OF MASKING FUNCTIONS
#

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


def mask_evoked(rec):
    r = rec.copy()
    r = r.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
    return r


def mask_all_but_targets(rec, include_incorrect=True):
    """
    Specialized function for removing incorrect trials from data
    collected using baphy during behavior.

    """
    newrec = rec.copy()
    newrec['resp'] = newrec['resp'].rasterize()
    #newrec = normalize_epoch_lengths(newrec, resp_sig='resp', epoch_regex='TARGET',
    #                                include_incorrect=include_incorrect)
    if 'stim' in newrec.signals.keys():
        newrec['stim'] = newrec['stim'].rasterize()

    #newrec = newrec.or_mask(['TARGET'])
    #newrec = newrec.and_mask(['PASSIVE_EXPERIMENT', 'TARGET'])
    #newrec = newrec.and_mask(['REFERENCE','TARGET'])
    newrec = newrec.and_mask(['TARGET'])

    if not include_incorrect:
        newrec = mask_incorrect(newrec)

    # svd attempt to kludge this masking to work with a lot of code that assumes all relevant epochs are
    # called "REFERENCE"
    #import pdb;pdb.set_trace()
    for k in newrec.signals.keys():
        newrec[k].epochs.name = newrec[k].epochs.name.str.replace("TARGET", "REFERENCE")
    return newrec


def mask_all_but_reference_target(rec, include_incorrect=True, **ctx):
    """
    Specialized function for removing incorrect trials from data
    collected using baphy during behavior.

    """
    newrec = rec.copy()
    newrec['resp'] = newrec['resp'].rasterize()
    newrec = normalize_epoch_lengths(newrec, resp_sig='resp', 
                                     epoch_regex='^(STIM|TAR|REF|CAT)',
                                     include_incorrect=include_incorrect)
    if 'stim' in newrec.signals.keys():
        newrec['stim'] = newrec['stim'].rasterize()

    #newrec = newrec.and_mask(['PASSIVE_EXPERIMENT', 'TARGET'])
    newrec = newrec.and_mask(['REFERENCE','TARGET','CATCH'])
    #newrec = newrec.and_mask(['TARGET'])

    if not include_incorrect:
        newrec = mask_incorrect(newrec)

    # svd attempt to kludge this masking to work with a lot of code that assumes all relevant epochs are
    # called "REFERENCE"
    #import pdb;pdb.set_trace()
    #for k in newrec.signals.keys():
    #    newrec[k].epochs.name = newrec[k].epochs.name.str.replace("TARGET", "REFERENCE")

    return {'rec': newrec}


def pupil_mask(est, val, condition, evoked_only=True, balance=False):
    """
    Create pupil mask by epoch (use REF by default) - so entire epoch is
    classified as big or small. Perform the mask on both est and val sets
    separately. This is so that both test metrics and fit metrics are
    evaluated on the same class of data (big or small pupil)
    """
    raise Warning('deprecated???')
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
        if 'mask' in r.signals:
            mask = (mask & r['mask'].extract_epoch('REFERENCE'))
            op_mask = (op_mask & r['mask'].extract_epoch('REFERENCE'))
        elif 'mask' not in r.signals:
            pass

        r['mask'] = r['mask'].replace_epochs({'REFERENCE': mask})
        r=r.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
        
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
    Figure out Pre- and PostStimSilence (units of sec) for a signal

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
    print(f"Pre0={PreStimSilence0} Post0={PostStimSilence0}")
    if PreStimSilence0 != PreStimSilence:
        if PreStimSilence0 < PreStimSilence-1/fs:
            raise Warning('Adjusting PreStimSilence to be longer than in orginal signal')
        d = sig.get_epoch_bounds('PreStimSilence')
        for e in d:
            ee = (np.round(epochs['start']*fs) == np.round(e[0]*fs))
            epochs.loc[ee, 'start'] = epochs.loc[ee, 'start'] + PreStimSilence0 - PreStimSilence

    if PostStimSilence0 != PostStimSilence:
        if PostStimSilence0 < PostStimSilence-1/fs:
            raise Warning('Adjusting PostStimSilence to be longer than in orginal signal')
        d = sig.get_epoch_bounds('PostStimSilence')
        for e in d:
            ee = (np.round(epochs['end']*fs) == np.round(e[1]*fs))
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


def create_pupil_mask(rec, evoked_only=False, **options):
    """
    Returns new recording with mask specified by the following options params.
    Pupil mask will always be "and-ed" with existing mask

    options dictionary
    =========================================================================
        state: what pupil state to return ("big" or "small" or "rem")
            default: "big"
        method: how to divide states - mean, median, fraction etc.
            default: "median"
        fraction: tuple desribing which fraction of pupil (ex: (0, 20)) would be 0 to 20 percent
            default: None
        epoch: If set, will classify entire epochs as big or small
            default: None
        fs: If integer, will classify each sample (as a rate of fs)
            default: None
        collapse: If true, tile the mean of pupil across each epoch
            default: False
        rm_rem: If true, find (if file exists) period with rem and remove these from the mask
            default: True
        cutoff: used if method ==  'user_def_value'. Force the division to be greater than (or less
            than cutoff)
    """

    state = options.get('state', 'big')
    method = options.get('method', 'median')
    fraction = options.get('fraction', None)
    epoch = options.get('epoch', None)
    fs = options.get('fs', rec['resp'].fs)
    collapse = options.get('collapse', False)
    rm_rem = options.get("rm_rem", False)  # updated default behavior on 08.07.2020 by crh

    if fs > rec['resp'].fs:
        raise ValueError

    if (fraction is None) and (method == 'fraction'):
        raise ValueError

    log.info("creating pupil mask with options: {0}".format(options))

    # make sure there are no nans in pupil signal, if so, pad with the last value
    if np.any(np.isnan(rec['pupil'].as_continuous())):
        tr = copy.deepcopy(rec)
        pupil = tr['pupil'].as_continuous().squeeze()
        pupil_raw = tr['pupil'].as_continuous().squeeze()
        args = np.argwhere((np.isnan(pupil)))
        val = args[-1]
        pupil[args] = pupil[val]
        pupil_raw[args] = pupil_raw[val]
        rec['pupil'] = rec['pupil']._modified_copy(pupil[np.newaxis, :])
        rec['pupil'] = rec['pupil']._modified_copy(pupil_raw[np.newaxis, :])

    # one rec to apply mask to
    r = rec.copy()
    # the rec returned should be the same size as original rec
    newrec = rec.copy()
    if 'mask' not in r.signals.keys():
        r = r.create_mask(True)
        newrec = newrec.create_mask(True)

    # get rid of rem periods in the mask prior to creating pupil mask based on mean/median
    if rm_rem & (state != 'rem'):
        # mask only reference presentations in which there was no rem sleep
        try:
            rem_mask = r['rem'].as_continuous().astype(np.bool)
            current_mask = r['mask'].as_continuous()
            new_mask = (~rem_mask & current_mask)
            r['mask'] = r['mask']._modified_copy(new_mask)
            newrec['mask'] = newrec['mask']._modified_copy(new_mask)

            # make sure all refs are classified homogenously (if trail has any rem, call it rem)
            folded_rem_mask = r['mask'].extract_epoch('REFERENCE')
            reps = folded_rem_mask.shape[0]
            trial_len = folded_rem_mask.shape[-1]
            n_rem = []
            for r_ in range(reps):
                if (np.sum(folded_rem_mask[r_, :, :]) != trial_len) & (folded_rem_mask[r_, :, :].sum() > 0):
                    folded_rem_mask[r_, :, :] = False
                    n_rem.append(1)

            newrec['mask'] = newrec['mask'].replace_epochs({'REFERENCE': folded_rem_mask})
            r['mask'] = r['mask'].replace_epochs({'REFERENCE': folded_rem_mask})

            log.info("Removed {0} REFs with REM".format(sum(n_rem)))

        except:
            log.info("WARNING - Not removing REM periods. Does rem file exist for this recording?")

    elif state == 'rem':
        try:
            rem_mask = r['rem'].as_continuous().astype(np.bool)
            current_mask = r['mask'].as_continuous()
            new_mask = (rem_mask & current_mask)
            r['mask'] = r['mask']._modified_copy(new_mask)
            newrec['mask'] = newrec['mask']._modified_copy(new_mask)

            # make sure all refs are classified homogenously (if trail has any rem, call it rem)
            folded_rem_mask = r['mask'].extract_epoch('REFERENCE')
            reps = folded_rem_mask.shape[0]
            trial_len = folded_rem_mask.shape[-1]
            n_rem = []
            for r_ in range(reps):
                if (np.sum(folded_rem_mask[r_, :, :]) != trial_len) & (folded_rem_mask[r_, :, :].sum() > 0):
                    folded_rem_mask[r_, :, :] = False
                    n_rem.append(1)

            newrec['mask'] = newrec['mask'].replace_epochs({'REFERENCE': folded_rem_mask})
            r['mask'] = r['mask'].replace_epochs({'REFERENCE': folded_rem_mask})

            log.info("Removed {0} REFs without REM".format(sum(n_rem)))

        except:
            log.info("WARNING - Not masking REM periods")

        return r

    r = r.apply_mask(reset_epochs=True)
    # we want the returned rec to be the same size, so don't apply any masking
    #newrec = newrec.apply_mask(reset_epochs=True)  # why wasn't this here before? CRH 1/30/19
    #newrec = newrec.create_mask(True)

    # process/classify pupil here, based on mean/median/abs size
    # Go through each of the binning options and create the mask
    if (collapse == False) & (epoch is not None):
        log.info('binning pupil at fs: {0} within epochs: {1}'.format(fs, epoch))

        # In this case, fold pupil first based on the epoch(s)
        folded_pupil = r['pupil'].extract_epochs(epoch)

        # Now, repeat the mean of each bin w/in the epoch over the length
        # of that bin (resampling pupil within epochs to give more "coarse" classification
        for e in epoch:
            reps = folded_pupil[e].shape[0]
            samps = folded_pupil[e].shape[-1]
            nbins = int(round(samps / (rec['resp'].fs / fs)))
            bin_len = int(round(samps / nbins))
            for rep in np.arange(0, reps):
                for b in np.arange(0, nbins+1):
                    if b == nbins:
                        start = int(b * bin_len)
                        if not folded_pupil[e][rep, :, start:]:
                            # check for empty slice
                            pass
                        else:
                            mean_pup = np.nanmean(folded_pupil[e][rep, :, start:])
                            folded_pupil[e][rep, :, start:] = mean_pup
                    else:
                        start = int(b * bin_len)
                        stop = start + bin_len
                        mean_pup = np.nanmean(folded_pupil[e][rep, :, start:stop])
                        folded_pupil[e][rep, :, start:stop] = mean_pup

        # rebuild pupil signal
        r['pupil'] = r['pupil'].replace_epochs(folded_pupil)

    elif (collapse is True) & (epoch is not None):
        log.info('collapsing over all {0} epochs and tiling mean pupil per epoch'.format(epoch))
        # In this case, fold pupil first based on the epoch(s)
        folded_pupil = r['pupil'].extract_epochs(epoch)

        # Now, repeat the mean of each epoch over the whole epoch
        for e in epoch:
            reps = folded_pupil[e].shape[0]
            for rep in np.arange(0, reps):
                mean_pup = np.mean(folded_pupil[e][rep, :, :])
                folded_pupil[e][rep, :, :] = mean_pup

        # rebuild pupil signal
        r['pupil'] = r['pupil'].replace_epochs(folded_pupil)

    elif (fs == rec['resp'].fs) & (epoch is None):
        log.info('classifying pupil continuously (at each time point)')


    elif (fs != rec['resp'].fs) & (epoch is None):
        log.info('WARNING - this could lead to weird edge effects later on')
        # Don't know which epoch to collapse over, so just rebin on the
        # continuous trace
        samps = r['pupil'].as_continuous().shape[-1]
        nbins = int(round(samps / (rec['resp'].fs / fs)))
        bin_len = int(round(samps / nbins))
        pupil_trace = r['pupil'].as_continuous()
        for b in np.arange(0, nbins+1):
                    if b == nbins:
                        start = int(b * bin_len)
                        if not pupil_trace[0, start:]:
                            # check for empty slice
                            pass
                        else:
                            mean_pup = np.mean(pupil_trace[0, start:])
                            pupil_trace[0, start:] = mean_pup
                    else:
                        start = int(b * bin_len)
                        stop = start + bin_len
                        mean_pup = np.mean(pupil_trace[0, start:stop])
                        pupil_trace[0, start:stop] = mean_pup

        # rebuild pupil signal
        r['pupil'] = r['pupil']._modified_copy(pupil_trace)

    
    # get pupil divider based on new pupil signal
    if method == 'median':
        pupil_divider = np.median(r['pupil'].as_continuous())
    elif method == 'mean':
        pupil_divider = np.mean(r['pupil'].as_continuous())
    elif method == 'user_def_value':
        pupil_divider = options['cutoff']
    elif method == 'fraction':
        pupil_max = np.max(r['pupil'].as_continuous())
        upper_lim = fraction[1]
        lower_lim = fraction[0]
        # normalize pupil
        r['pupil'] = r['pupil']._modified_copy(r['pupil'].as_continuous() / pupil_max)
        state = None

    # create short mask (on the masked data - for ex, because of rem exclusion)
    if state == 'big':
        short_mask = r['pupil'].as_continuous() >= pupil_divider
    elif state == 'small':
        short_mask = r['pupil'].as_continuous() < pupil_divider
    elif method == 'fraction':
        short_mask = (r['pupil'].as_continuous() > lower_lim) & (r['pupil'].as_continuous() < upper_lim)
    else:
        raise ValueError

    # Now, go through the short mask and make it the correct length using
    # newrec's mask. short_mask size should be equal to current_mask sum
    current_mask = newrec['mask'].as_continuous().squeeze()  # this is the long mask (which has rem excluded)

    final_mask = np.zeros(current_mask.shape).astype(np.bool)

    j = 0
    # loop over the current mask (with the mask set to False when rem is True). 
    # only copy over the pupil mask (the short_mask) for periods when current mask is True
    for i, m in enumerate(current_mask):

        if m == True:
            if short_mask[0, j] == True:
                final_mask[i] = True
            j += 1

    final_mask = final_mask[np.newaxis, :]

    # Add the new mask to the recording
    newrec['mask'] = newrec['mask']._modified_copy(final_mask)
    if evoked_only:
        newrec=newrec.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
    
    return newrec


def bandpass_filter_resp(rec, low_c, high_c, data=None, signal='resp'):
    '''
    Bandpass filter resp. Return new recording with filtered resp.
    '''

    if low_c is None:
        low_c = 0
    if high_c is None:
        high_c = rec['resp'].fs

    fs = rec[signal].fs
    if data is None:
        newrec = rec.copy()
        #newrec = newrec.apply_mask(reset_epochs=True)
        newrec[signal] = rec[signal].rasterize()
        resp = newrec[signal]._data
    else:
        resp = data
    
    resp_filt = resp.copy()
    for n in range(resp.shape[0]):
        s = resp[n, :]
        resp_fft = fp.fft(s)
        w = fp.fftfreq(s.size, 1 / fs)
        inds = np.argwhere((w >= low_c) & (w <= high_c))
        inds2 = np.argwhere((w <= -low_c) & (w >= -high_c))
        m = np.zeros(w.shape)
        alpha = 0.1
        m[inds] = ss.tukey(len(inds), alpha)[:, np.newaxis]
        m[inds2] = ss.tukey(len(inds2), alpha)[:, np.newaxis]
        resp_cut = resp_fft * m
        resp_filt[n, :] = fp.ifft(resp_cut)

    if data is None:
        newrec[signal] = newrec[signal]._modified_copy(resp_filt)
        return newrec
    else:
        return resp_filt

def get_pupil_balanced_epochs(rec, rec_sp=None, rec_bp=None):
    """
    Given big/small pupil recordings return list of
    epochs that are balanced between the two.
    """
    all_epochs = np.unique([str(ep) for ep in rec.epochs.name if 'STIM_00' in ep]).tolist()

    if (rec_sp is None) | (rec_bp is None):
        pup_ops = {'state': 'big', 'epoch': ['REFERENCE'], 'collapse': True}
        rec_bp = create_pupil_mask(rec.copy(), **pup_ops)
        pup_ops['state']='small'
        rec_sp = create_pupil_mask(rec.copy(), **pup_ops)

        rec_bp = rec_bp.apply_mask(reset_epochs=True)
        rec_sp = rec_sp.apply_mask(reset_epochs=True)

    # get rid of pre/post stim silence
    #rec_bp = rec_bp.and_mask(['PostStimSilence'], invert=True)
    #rec_sp = rec_sp.and_mask(['PostStimSilence'], invert=True)
    #rec = rec.and_mask(['PostStimSilence'], invert=True)
    #rec_bp = rec_bp.apply_mask(reset_epochs=True)
    #rec_sp = rec_sp.apply_mask(reset_epochs=True)
    #rec = rec.apply_mask(reset_epochs=True)

    # find pupil matched epochs
    balanced_eps = []
    for ep in all_epochs:
        sp = rec_sp['resp'].extract_epoch(ep).shape[0]
        bp = rec_bp['resp'].extract_epoch(ep).shape[0]
        if len(all_epochs)==3:
            if abs(sp - bp) < 3:
                balanced_eps.append(ep)
        else:
            if sp==bp:
                balanced_eps.append(ep)

    if len(balanced_eps)==0:
        log.info("no balanced epochs at site {}".format(rec.name))

    else:
        log.info("found {0} balanced epochs:".format(len(balanced_eps)))

    return balanced_eps

def mask_pupil_balanced_epochs(rec):
    r = rec.copy()
    balanced_epochs = get_pupil_balanced_epochs(r)
    r = r.and_mask(balanced_epochs)
    return r

def add_pupil_mask(rec, state='big', mask_name='p_mask', evoked_only=True):
    '''
    Simply add a p_mask signal that's true where p > median.
    Does this on a "per ref" basis so that epochs aren't chopped up.
    '''
    r = rec.copy()
    ops = {'state': state, 'epoch': ['REFERENCE'], 'collapse': True} 
    bp = create_pupil_mask(r, evoked_only=evoked_only, **ops)
    r[mask_name] = bp['mask']
    return r

def pupil_large_small_masks(rec, evoked_only=True, ev_bins=0, split_per_stim=False, add_per_stim=False, custom_epochs=False, respsort=False, pupsort=False, reduce_mask=False, **kwargs):
    """
    Utility function for cc_norm fitter. Generates masking signals used by the fitter to make LV weights
      reproduce desired pattern of noise correlations in different conditions.
    By default, creates signals that mask trials according to whether pupil is smaller or larger than the mean.
    Added to whatever mask already exists in rec.
    Inputs: rec: Recording from NEMS
    Returns rec: Recording with new signals 'mask_large' and 'mask_small'
    Option: add_per_stim: if True, also returns signals 'mask_<epoch>' matching each epoch that 
       starts with "STIM_". Currently these masks span both large and small pupil
    """
    r = rec.copy()
    r = add_pupil_mask(r, state='big', mask_name='mask_large', evoked_only=evoked_only)
    r = add_pupil_mask(r, state='small', mask_name='mask_small', evoked_only=evoked_only)

    if custom_epochs:
        # special case, masking pupil per epoch/bin using a custom set of epochs.
        site = kwargs['meta']['cellid']
        batch = kwargs['meta']['batch']
        if (batch==322) | (batch==str(322)):
            batch = 289
        fn = f'/auto/users/hellerc/results/nat_pupil_ms/reliable_epochs/{batch}/{site}.pickle'
        log.info(f"Loading sorted epochs / bins for batch / site {batch}/{site} from {fn}")
        reliable_epochs = pickle.load(open(fn, "rb"))

        # Use the best <ev_bins> epochs, or default of 5
        if ev_bins == 0:
            ev_bins = 5

        mask_bins = []
        reliable = []
        for i in range(ev_bins):
            # add pupil masks for these individual bins
            (e, b) = reliable_epochs['sorted_epochs'][i]
            p = r['pupil'].extract_epoch(str(e), mask=r['mask'])[:, :, int(b)]
            idx = r['resp'].get_epoch_indices(e, mask=r['mask'])
            m = np.zeros(r['pupil']._data.shape[-1])
            m[idx[:, 0] + int(b)] = 1
            m = m.astype(bool)
            m_sm = (r['pupil']._data.squeeze() < np.median(p)) & m
            m_lg = (r['pupil']._data.squeeze() >= np.median(p)) & m
            r['mask_'+e+':'+b+'_sm'] = r['mask']._modified_copy((r['mask']._data * m_sm).astype(bool))
            r['mask_'+e+':'+b+'_lg'] = r['mask']._modified_copy((r['mask']._data * m_lg).astype(bool))
            log.info(f"{e} {b} sm={r['mask_'+e+':'+b+'_sm'].as_continuous().sum()} lg={r['mask_'+e+':'+b+'_lg'].as_continuous().sum()}")
            mask_bins.append((e, int(b)))
            reliable.append(reliable_epochs['reliable_mask'][i])

        r.meta['mask_bins'] = mask_bins
        #r.meta['reliable_bin'] = reliable

        return {'rec': r}

    if (ev_bins>=1) & (respsort):
        log.info(f"Sorting epochs by response magnitude and keeping the top {ev_bins} for lrg/sm splits")
        # sort epoch / bin combinations by size of mean population response
        # take the first ev_bins of these sorted epoch / bin combinations
        # 21.10.2021 -- using 'reset_epochs' here is kinda slow. Is there a better way?
        epoch_names = epoch_names_matching(r.apply_mask(reset_epochs=True)['resp'].epochs, '^STIM_')

        # for each epoch / bin, get the size of the response
        resp = r['resp'].extract_epochs(epoch_names, mask=r['mask'])
        rmag = np.array([resp[e].mean(axis=(0,1)) for e in epoch_names])
        
        # sort responses from greatest to smallest
        idx = np.unravel_index(np.argsort(rmag.ravel())[::-1], rmag.shape)

        # remove any spont bins
        pre,post = getPrePostSilence(r['resp'])
        prebins = int(pre*r['resp'].fs)
        postbins = int(post*r['resp'].fs)
        durbins = rmag.shape[-1]-prebins-postbins
        rridx = (idx[1] < prebins) | (idx[1] >= (prebins+durbins))
        epoch_idx = idx[0][~rridx]
        bin_idx = idx[1][~rridx]

        # extract corresponding epoch / bin number
        keep_epochs = np.array(epoch_names)[epoch_idx][:ev_bins]
        keep_bins = np.array(bin_idx)[:ev_bins]

        # for each epoch / bin, make a pupil mask
        mask_bins=[]
        for i, (e, b) in enumerate(zip(keep_epochs, keep_bins)):
            e = str(e)
            mask_bins.append((e, b))
            p = r['pupil'].extract_epoch(e, mask=r['mask'])
            eindices = r['resp'].get_epoch_indices(e, mask=r['mask'])
            p = np.squeeze(p.mean(axis=2))

            _m_lg = np.zeros(r['pupil'].shape)
            _m_sm = np.zeros(r['pupil'].shape)
            for j, ei in enumerate(eindices):
                if p[j] < np.median(p):
                    _m_sm[0, ei[0] + b] = 1
                else:
                    _m_lg[0, ei[0] + b] = 1

            r['mask_'+e+':'+str(b)+'_sm'] = r['mask']._modified_copy((r['mask']._data * _m_sm).astype(bool))
            r['mask_'+e+':'+str(b)+'_lg'] = r['mask']._modified_copy((r['mask']._data * _m_lg).astype(bool))
            log.info(f"{e} {b} sm={r['mask_'+e+':'+str(b)+'_sm'].as_continuous().sum()} lg={r['mask_'+e+':'+str(b)+'_lg'].as_continuous().sum()}")
        r.meta['mask_bins'] = mask_bins

    elif (ev_bins>=1) & (pupsort):
        log.info(f"Sorting epochs by normalized pupil size")
        # sort epoch / bin combinations by the range of normalized pupil range that they span
        # crh 14.11.2021
        epoch_names = epoch_names_matching(r.apply_mask(reset_epochs=True)['resp'].epochs, '^STIM_')

        # for each epoch / bin, get the size of the response
        pup = r['pupil_raw'].extract_epochs(epoch_names, mask=r['mask'])
        pmax = r['pupil_raw']._data.max()
        pup = {k: p / pmax for k, p in pup.items()}

        # sort by how close the mean is to 0.5
        pmdiff = np.array([np.abs(pup[e].mean(axis=(0,1)) - 0.5) for e in epoch_names])
        
        # sort diffs from smallest to greatest
        idx = np.unravel_index(np.argsort(pmdiff.ravel()), pmdiff.shape)

        # remove any spont bins
        pre,post = getPrePostSilence(r['resp'])
        prebins = int(pre*r['resp'].fs)
        postbins = int(post*r['resp'].fs)
        durbins = pmdiff.shape[-1]-prebins-postbins
        rridx = (idx[1] < prebins) | (idx[1] >= (prebins+durbins))
        epoch_idx = idx[0][~rridx]
        bin_idx = idx[1][~rridx]

        # extract corresponding epoch / bin number
        keep_epochs = np.array(epoch_names)[epoch_idx][:ev_bins]
        keep_bins = np.array(bin_idx)[:ev_bins]

        # for each epoch / bin, make a pupil mask
        mask_bins=[]
        for i, (e, b) in enumerate(zip(keep_epochs, keep_bins)):
            e = str(e)
            mask_bins.append((e, b))
            p = r['pupil'].extract_epoch(e, mask=r['mask'])
            eindices = r['resp'].get_epoch_indices(e, mask=r['mask'])
            p = np.squeeze(p.mean(axis=2))

            _m_lg = np.zeros(r['pupil'].shape)
            _m_sm = np.zeros(r['pupil'].shape)
            for j, ei in enumerate(eindices):
                if p[j] < np.median(p):
                    _m_sm[0, ei[0] + b] = 1
                else:
                    _m_lg[0, ei[0] + b] = 1

            r['mask_'+e+':'+str(b)+'_sm'] = r['mask']._modified_copy((r['mask']._data * _m_sm).astype(bool))
            r['mask_'+e+':'+str(b)+'_lg'] = r['mask']._modified_copy((r['mask']._data * _m_lg).astype(bool))
            log.info(f"{e} {b} sm={r['mask_'+e+':'+str(b)+'_sm'].as_continuous().sum()} lg={r['mask_'+e+':'+str(b)+'_lg'].as_continuous().sum()}")
        r.meta['mask_bins'] = mask_bins

    elif ev_bins>=1:
        # special case, find stim with high pupil variance, mask in individual bins from them
        # generate list of tuples in rec.meta with [(epoch, bin), ... ]
        epoch_names = epoch_names_matching(r['resp'].epochs, '^STIM_')
        epoch_names=list(r['resp'].extract_epochs(epoch_names, mask=r['mask']).keys())
        pstd=np.zeros(len(epoch_names))

        for i,e in enumerate(epoch_names):
            p = r['pupil'].extract_epoch(e, mask=r['mask'])
            # pstd[i] = p.mean(axis=0).std()
            # crh 05.07.2021. Think the previous line should be:
            pstd[i] = p.mean(axis=-1).std() # for variance across trials

        epoch_bins=p.shape[2]
        pre,post = getPrePostSilence(r['resp'])
        prebins = int(pre*r['resp'].fs)
        postbins = int(post*r['resp'].fs)
        durbins = epoch_bins-prebins-postbins
        pkeep = np.array([s>np.max(pstd)/3 for s in pstd])
        if pkeep.sum()<3:
            pkeep = np.array([s>np.max(pstd)/6 for s in pstd])
        if pkeep.sum()<3:
            pkeep=np.ones(len(epoch_names), dtype=bool)
        epoch_names=[epoch_names[j] for j,p in enumerate(pkeep) if p]
        pstd=pstd[pkeep]
        epoch_count=len(epoch_names)
        total_bins=durbins * epoch_count
        choose_bins = np.linspace(0,total_bins-1,ev_bins)
        choose_epoch=np.floor(choose_bins/durbins).astype(int)
        choose_bin = (choose_bins % durbins + prebins).astype(int)
        mask_bins=[]
        for i,eidx in enumerate(choose_epoch):
            e = epoch_names[eidx]
            mask_bins.append((e, int(choose_bin[i])))
            p = r['pupil'].extract_epoch(e, mask=r['mask'])
            b = r['resp'].get_epoch_indices(e, mask=r['mask'])
            p = np.squeeze(p.mean(axis=2))

            _m_lg = np.zeros(r['pupil'].shape)
            _m_sm = np.zeros(r['pupil'].shape)
            for j,_b in enumerate(b):
                if p[j]<np.median(p):
                    _m_sm[0,b[j][0]+choose_bin[i]] = 1
                else:
                    _m_lg[0,b[j][0]+choose_bin[i]] = 1

            r['mask_'+e+':'+str(choose_bin[i])+'_sm'] = r['mask']._modified_copy((r['mask']._data * _m_sm).astype(bool))
            r['mask_'+e+':'+str(choose_bin[i])+'_lg'] = r['mask']._modified_copy((r['mask']._data * _m_lg).astype(bool))
            log.info(f"{e} {choose_bin[i]} sm={r['mask_'+e+':'+str(choose_bin[i])+'_sm'].as_continuous().sum()} lg={r['mask_'+e+':'+str(choose_bin[i])+'_lg'].as_continuous().sum()}")
        r.meta['mask_bins'] = mask_bins
    
        #return {'rec': r}
    
    elif 0 & (ev_bins>0):
        _m = r['mask_large'].as_continuous()[0,:]
        _dm = np.concatenate(([0],np.diff(_m)>0)).astype(bool)
        r['mask_large'] = r['mask_large']._modified_copy(_dm)
        _m = r['mask_small'].as_continuous()[0,:]
        _dm = np.concatenate(([0],np.diff(_m)>0)).astype(bool)
        r['mask_small'] = r['mask_small']._modified_copy(_dm)
    
    elif add_per_stim or split_per_stim:
        e = epoch_names_matching(r['resp'].epochs, '^STIM_')
        #e[:3]

        epochs_to_mask=list(r['resp'].extract_epochs(e, mask=r['mask']).keys())
        for e in epochs_to_mask:
            _r = r.copy()
            _r = _r.and_mask(e)
            if evoked_only:
                _r=_r.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
            if ev_bins==1:
                _m = _r['mask'].as_continuous()[0,:]
                _dm = np.concatenate(([0],np.diff(_m.astype(float))>0)).astype(bool)
                _r = _r.and_mask(_dm)
                
            if split_per_stim:
                r['mask_'+e+'_lg'] = r['mask']._modified_copy(_r['mask']._data * r['mask_large']._data)
                r['mask_'+e+'_sm'] = r['mask']._modified_copy(_r['mask']._data * r['mask_small']._data)
            else:
                r['mask_'+e] = r['mask']._modified_copy(_r['mask']._data)

        #plt.figure()
        #plt.plot(rec['mask'].as_continuous()[0,:]-1.1)
        #for i,e in enumerate(epochs_to_mask):
        #    plt.plot(rec['mask_'+e].as_continuous()[0,:]+i*1.1)
        #import pdb;
        #pdb.set_trace()

    # at the end, AND the large and small masks with the selected epochs
    masks = [k for k in r.signals.keys()
             if (k.startswith("mask_") and k!="mask_small" and k!="mask_large")]
    m = np.zeros_like(r["mask"]._data)
    for i in masks:
        m += r[i]._data
    r['mask_small'] = r['mask_small']._modified_copy(data=r['mask_small']._data * m)
    r['mask_large'] = r['mask_large']._modified_copy(data=r['mask_large']._data * m)
    log.info(f"Masks trimmed: sm: {r['mask_small']._data.sum()}  lg: {r['mask_large']._data.sum()}")
    if reduce_mask:
        r['mask'] = r['mask']._modified_copy(data=r['mask_small']._data +r['mask_large']._data)
        log.info('Including main mask')
    return {'rec': r}


def mask_time_segments(rec, segment_count=8, evoked_only=True, **kwargs):
    """
    Utility function for cc_norm fitter. Generates masking signals used by the fitter to make LV weights
      reproduce desired pattern of noise correlations in different conditions.
    By default, creates signals that mask trials according to whether pupil is smaller or larger than the mean.
    Added to whatever mask already exists in rec.
    Inputs: rec: Recording from NEMS
    Returns rec: Recording with new signals 'mask_large' and 'mask_small'
    Option: add_per_stim: if True, also returns signals 'mask_<epoch>' matching each epoch that 
       starts with "STIM_". Currently these masks span both large and small pupil
    """
    r = rec.copy()
    
    if 'mask' in r.signals.keys():
        m = r['mask']._data.copy()
    else:
        m = np.ones((1,r['resp'].shape[1]))
    r = add_pupil_mask(r, state='big', mask_name='mask_large', evoked_only=evoked_only)
    r = add_pupil_mask(r, state='small', mask_name='mask_small', evoked_only=evoked_only)
    
    nbins = m.sum()
    validbins = np.where(m[0,:]>0)[0]
    chunksize = nbins / segment_count
    log.info(f"nbins/segmentcount: {nbins}/{segment_count} = {chunksize}")
    #log.info(f"{validbins[:100]}")
    for s in range(segment_count):
        _m = np.zeros_like(m)
        #log.info(f'{s}: {validbins[int(s*chunksize):int((s+1)*chunksize)]}')
        _m[0, validbins[int(s*chunksize):int((s+1)*chunksize)]]=1
        log.info(f'{s}: {_m.sum()}')
        r[f'mask_{s}'] = r['mask']._modified_copy(_m.astype(bool))
               
    return {'rec': r}


def movement_mask(rec, binsize=1, threshold=0.25, **kwargs):
    '''
    Use pupil_extras signals to mask the recording during periods of movement (eg blinks)
    binsize - window size in seconds over which to compute variance of eye position traces
    threshold - number of std of the variance trace over which data is excluded

    crh 06.18.2021
    '''
    if 'pupil_extras' not in rec.signals.keys():
        raise ValueError("Pupil extras do not exist for this recording. Old recording? Old pupil analysis?")
    r = rec.copy()
    # before masking, take raw signals and compute variance of an eyelid over a sliding window
    binsize = int(binsize * r['resp'].fs)
    signal = r['pupil_extras'].extract_channels(['eyelid_top_y'])._data 
    signal2 = r['pupil_extras'].extract_channels(['eyelid_bottom_y'])._data
    varsig = np.zeros(signal.shape)
    for i in range(varsig.shape[-1]):
        varsig[0, i] = np.var(signal[0, i:(i+binsize)])+np.var(signal2[0, i:(i+binsize)])
    varsig = np.roll(varsig, int(binsize/2))
    r['varsig'] = r['pupil']._modified_copy(varsig)
    threshold = np.std(varsig) * threshold
    mask = varsig >= threshold
    log.info(f"Found {mask.sum()} / {mask.shape[-1]} bins that violated movement threshold")

    # tile mask over generic epochs (REFERENCE, TARGET, CATCH)
    epochs = [e for e in ['REFERENCE', 'TARGET', 'CATCH'] if e in r['resp'].epochs.name.unique()]
    
    rec['var_mask'] = r['pupil']._modified_copy(mask)
    fm = rec['var_mask'].extract_epochs(epochs)
    # tile bool across full reference so not to split up epochs in weird ways
    fm = {k: np.concatenate([np.zeros(v[[i]].shape).astype(bool) if np.any(v[[i]]==True) else np.ones(v[[i]].shape).astype(bool) 
                        for i in range(v.shape[0])], axis=0) 
                        for k, v in fm.items()}
    rec['var_mask'] = rec['var_mask'].replace_epochs(fm)
    if 'mask' not in r.signals:
        r = r.create_mask(True)
    r['mask'] = r['mask']._modified_copy(r['mask']._data & rec['var_mask']._data)

    return {'rec': r}

def create_residual(rec, cutoff=None, shuffle=False, signal='psth_sp'):
    
    r = rec.copy()
    r['resp'] = r['resp'].rasterize()
    r['residual'] = r['resp']._modified_copy(r['resp']._data - r[signal]._data)
    
    if cutoff is not None:
        r = bandpass_filter_resp(r, low_c=cutoff, high_c=None, signal='residual')

    if shuffle:
        r['residual'] = r['residual'].shuffle_time(rand_seed=1)

    return r

def add_epoch_signal(rec):
    """
    Add new signal to recording with each channel being the 
    True/False mask for each STIM epoch
    """
    r = rec.copy()
    r['resp'] = r['resp'].rasterize()
    epochs = [e for e in r.epochs.name.unique() if 'STIM' in e]
    resp = r['resp']._data
    bins = r['resp'].extract_epoch(epochs[0]).shape[-1]
    epoch_signal = np.zeros((len(epochs) * bins, resp.shape[-1]))
    idx = 0
    for e in epochs:
        for b in range(bins):
            sig = r['resp'].epoch_to_signal(e)
            data = sig.extract_epoch(e)
        
            ran = np.arange(0, bins)
            data[:, :, (ran<b) | (ran>b)] = 0
            data[:, :, b] = 1 
            sig = sig._modified_copy(np.zeros((1, sig._data.shape[-1])))
            sig = sig.replace_epochs({e: data})
            epoch_signal[idx, :] = sig._data.squeeze()
            idx+=1

    r['stim_epochs'] = r['resp']._modified_copy(epoch_signal)
    r['stim_epochs'].name = 'stim_epochs'

    return r 


def add_meta(rec):
    from nems.signal import RasterizedSignal
    if type(rec['resp']) is not RasterizedSignal:
        rec['resp'] = rec['resp'].rasterize()

    ref_len = rec.apply_mask(reset_epochs=True)['resp'].extract_epoch('REFERENCE').shape[-1]

    rec.meta['ref_len'] = ref_len

    import charlieTools.noise_correlations as nc
    epochs = [e for e in rec.apply_mask(reset_epochs=True).epochs.name.unique() if 'STIM' in e]
    resp_dict = rec['resp'].extract_epochs(epochs)
    nc = nc.compute_rsc(resp_dict)
    idx = nc[nc['pval'] < 0.05].index
    idx = [(int(x.split('_')[0]), int(x.split('_')[1])) for x in idx]
    arr = np.zeros((rec['resp'].shape[0], rec['resp'].shape[0]))
    for i in idx:
        arr[i[0], i[1]] = 1

    rec.meta['sig_corr_pairs'] = arr
    
    return rec


def zscore_resp(rec, use_mask=False):
    r = rec.copy()
    r['resp'] = r['resp'].rasterize()
    zscore = r['resp']._data
    if use_mask:
        log.info("using recording mask to zscore")
        _zscore = r.apply_mask()['resp']._data
    else:
        _zscore = zscore.copy()
    zscore = (zscore.T - _zscore.mean()).T
    zscore = (zscore.T / _zscore.std(axis=-1)).T

    r['resp_raw'] = rec['resp']
    r['resp'] = r['resp']._modified_copy(zscore)

    return r


def state_resp_outer(rec, s='state', r='resp', smooth_window=5,
                     shuffle_interactions=False, **ctx):

    state = rec[s]._data
    state[1:,:] = state[1:,:] - np.min(state[1:,:], axis=1, keepdims=True)
    
    psth = generate_average_sig(rec[r],'psth','^(STIM|TAR|CAT)', mask=rec['mask'])
    p = psth._data
    resp = rec[r]._data - p
    #smwin = np.ones((1,smooth_window))/smooth_window
    #resp = ss.convolve2d(resp,smwin,'same')
    resp = gaussian_filter1d(resp,smooth_window,axis=1)
 
    new_state = state.copy()

    for i in range(resp.shape[0]):
        if shuffle_interactions:
            tsig1 = state[0:1,:]*resp[i:(i+1),:]
            tsig2 = rec[s]._modified_copy(data=state[1:,:]*resp[i:(i+1),:])
            tsig2 = tsig2.shuffle_time(rand_seed=i, mask=rec['mask'])._data 
            new_state = np.concatenate((new_state, tsig1, tsig2))
        else:
            new_state = np.concatenate((new_state, state*resp[i:(i+1),:]))

    ts=np.std(new_state, axis=1, keepdims=True)
    ts[ts==0]=1
    new_state[1,:] -= np.min(new_state[1,:]) # hard-coded assuming pupil
 
    new_rec = rec.copy()
    new_rec[s]=rec[s]._modified_copy(data=new_state/ts)

    return {'rec': new_rec}

