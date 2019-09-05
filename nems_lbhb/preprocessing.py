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


def create_pupil_mask(rec, **options):
    """
    Returns new recording with mask specified by the following options params.
    Pupil mask will always be "and-ed" with existing mask

    options dictionary
    =========================================================================
        state: what pupil state to return ("big" or "small" or "rem")
            default: "big"
        method: how to divide states - mean, median, K-nearest-neighbors, fraction etc.
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
    """

    state = options.get('state', 'big')
    method = options.get('method', 'median')
    fraction = options.get('fraction', None)
    epoch = options.get('epoch', None)
    fs = options.get('fs', rec['resp'].fs)
    collapse = options.get('collapse', False)
    rm_rem = options.get("rm_rem", True)
    use_cache = False

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

    if use_cache == False:
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
        elif method == 'cache':
            use_cache = True
            log.info('using cached pupil classifications')
        elif method == 'fraction':
            pupil_max = get_max_pupil(r['resp'].chans[0][:7])
            if np.isnan(pupil_max):
                pupil_max = r['pupil'].as_continuous().max()
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

        return newrec

    elif use_cache:
        # load cached pupil classifiers and use these to make pupil mask

        # load mask
        mask = pcu.load_pupil_labels(batch, rec.name[:7], rec['resp'].fs, state, True)

        # and mask with the current mask
        mask = (mask & newrec['mask']._data.squeeze())
        newrec['mask'] = newrec['mask']._modified_copy(mask[np.newaxis, :])

        # go through all the different epoch options for how to apply the mask
        if (collapse == False) & (epoch is not None):
            log.info('binning pupil at fs: {0} within epochs: {1}'.format(fs, epoch))
            # In this case, fold pupil first based on the epoch(s)
            folded_mask = newrec['mask'].extract_epochs(epoch)

            # Now, repeat the mean of each bin w/in the epoch over the length
            # of that bin (1 if greater than 0.5)
            for e in epoch:
                reps = folded_mask[e].shape[0]
                samps = folded_mask[e].shape[-1]
                nbins = int(round(samps / (rec['resp'].fs / fs)))
                bin_len = int(round(samps / nbins))
                for rep in np.arange(0, reps):
                    for b in np.arange(0, nbins+1):
                        if b == nbins:
                            start = int(b * bin_len)
                            if not folded_mask[e][rep, :, start:]:
                                # check for empty slice
                                pass
                            else:
                                mean_m = np.mean(folded_mask[e][rep, :, start:])
                                if mean_m >= 0.5:
                                    folded_mask[e][rep, :, start:] = 1
                                else:
                                    folded_mask[e][rep, :, start:] = 0

                        else:
                            start = int(b * bin_len)
                            stop = start + bin_len
                            mean_m = np.mean(folded_mask[e][rep, :, start:stop])
                            if mean_m >= 0.5:
                                folded_mask[e][rep, :, start:stop] = 1
                            else:
                                folded_mask[e][rep, :, start:stop] = 0

                # rebuild mask
                newrec['mask'] = newrec['mask'].replace_epochs(folded_mask)

        elif (collapse==True) & (epoch is not None):
            log.info('collapsing over all {0} epochs'.format(epoch))

            # In this case, fold mask first based on the epoch(s)
            folded_mask = newrec['mask'].extract_epochs(epoch)

            # Now, repeat the mean of each epoch over the whole epoch
            for e in epoch:
                reps = folded_mask[e].shape[0]
                for rep in np.arange(0, reps):
                    mean_m = np.mean(folded_mask[e][rep, :, :])
                    if mean_m >= 0.5:
                        folded_mask[e][rep, :, :] = 1
                    else:
                        folded_mask[e][rep, :, :] = 0

            # rebuild pupil signal
            newrec['mask'] = newrec['mask'].replace_epochs(folded_mask)


        elif (fs == rec['resp'].fs) & (epoch is None):
            log.info('classifying pupil continuously (at each time point)')
            # in this case, the loaded mask &'ed with the existing mask is all we need
            pass

        elif (fs != rec['resp'].fs) & (epoch is None):
            log.info('WARNING - this could lead to weird edge effects later on')

            # Don't know which epoch to collapse over, so just rebin on the
            # continuous trace
            samps = r['pupil']._data.shape[-1]
            nbins = int(round(samps / (rec['resp'].fs / fs)))
            bin_len = int(round(samps / nbins))
            mask_trace = newrec['mask']._data
            for b in np.arange(0, nbins+1):
                        if b == nbins:
                            start = int(b * bin_len)
                            if not mask_trace[0, start:]:
                                # check for empty slice
                                pass
                            else:
                                mean_m = np.mean(mask_trace[0, start:])
                                if mean_m >= 0.5:
                                    mask_trace[0, start:] = 1
                                else:
                                    mask_trace[0, start:] = 0
                        else:
                            start = int(b * bin_len)
                            stop = start + bin_len
                            mean_m = np.mean(mask_trace[0, start:stop])
                            if mean_m >= 0.5:
                                mask_trace[0, start:stop] = 1
                            else:
                                mask_trace[0, start:stop] = 0

            # rebuild mask signal
            newrec['mask'] = newrec['mask']._modified_copy(mask_trace)

        return newrec
