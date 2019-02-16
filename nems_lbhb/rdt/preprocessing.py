import re
import logging
import numpy as np
import pandas as pd
from nems.epoch import epoch_difference, epoch_union
from nems.epoch import epoch_difference, epoch_intersection

log = logging.getLogger(__name__)

def select_balanced_targets(epochs, rng):
    pattern = 'target_(\d+)_(random|repeating)_(dual|single)'
    epoch_counts = epochs.groupby('name').size()
    epoch_counts = epoch_counts.filter(regex=pattern)

    pattern = re.compile(pattern)
    result = []
    for name, count in epoch_counts.iteritems():
        target, phase, stream = pattern.match(name).groups()
        result.append({
            'target': target,
            'phase': phase,
            'stream': stream,
            'count': count,
        })
    result = pd.DataFrame(result).set_index(['target', 'stream', 'phase'])
    result = result.unstack('phase')

    keep = []
    for (target, stream), counts in result.iterrows():
        random, repeating = counts.values
        repeating_epoch_name = f'target_{target}_repeating_{stream}'
        m_repeating = epochs['name'] == repeating_epoch_name
        random_epoch_name = f'target_{target}_random_{stream}'
        m_random = epochs['name'] == random_epoch_name

        keep.append(epochs.loc[m_random])
        if random >= repeating:
            keep.append(epochs.loc[m_repeating])
            continue
        else:
            subset = epochs.loc[m_repeating].sample(random, random_state=rng)
            keep.append(subset)

    keep = pd.concat(keep, ignore_index=True)[['start', 'end']]
    return keep


def get_est_val_times(rec, balance_phase=False, njacks=5):
    rng = np.random.RandomState(0)

    epochs = rec.epochs
    est_times, val_times = get_est_val_times_by_sequence(rec, rng, njacks)

    if balance_phase:
        target_times = select_balanced_targets(epochs, rng)

        m = epochs['name'].str.contains('^repeating$')
        repeating_times = epochs.loc[m, ['start', 'end']].values
        for jj in range(njacks):
            # Remove the repeating phase from the dataset
            est_times[jj] = epoch_difference(est_times[jj], repeating_times)
            # Now, add back in selected targets from repeating phase
            est_times[jj] = epoch_union(est_times[jj], target_times)

            # Remove the repeating phase from the dataset
            val_times[jj] = epoch_difference(val_times[jj], repeating_times)
            # Now, add back in selected targets from repeating phase
            val_times[jj] = epoch_union(val_times[jj], target_times)

    return est_times, val_times


def split_est_val(rec, balance_phase=False, njacks=5, jackknifed_fit=False, **context):
    est_times, val_times = get_est_val_times(rec, balance_phase, njacks)
    dual_only=False
    if not(jackknifed_fit):
        est = select_times(rec, est_times[0], random_only=False, dual_only=dual_only)
        val = select_times(rec, val_times[0], random_only=False, dual_only=dual_only)
        rand = select_times(rec, est_times[0], random_only=True, dual_only=dual_only),

    else:
        est = rec.tile_views(njacks)
        val = rec.tile_views(njacks)
        rand = rec.tile_views(njacks)

        for jj in range(njacks):
            est = est.set_view(jj)
            est = select_times(est, est_times[jj], random_only=False, dual_only=dual_only)
            val = val.set_view(jj)
            val = select_times(val, val_times[jj], random_only=False, dual_only=dual_only)
            rand = rand.set_view(jj)
            rand = select_times(rand, est_times[jj], random_only=True, dual_only=dual_only)

    return {'rand': rand, 'est': est, 'val': val, 'jackknifed_fit': jackknifed_fit}

def get_est_val_times_by_sequence(rec, rng, njack=5):
    epochs = rec.epochs
    m = epochs['name'].str.match('^SEQUENCE')
    sequences = epochs.loc[m, 'name'].unique()
    valfrac = 1/njack

    s_map = {}
    for s in sequences:
        tid = rec['target_id_map'].extract_epoch(s).ravel()[0]
        tid = int(tid)
        is_ds = rec['dual_stream'].extract_epoch(s).ravel()[0]
        is_ds = bool(is_ds)
        s_map.setdefault((tid, is_ds), []).append(s)

    val_epochs, est_epochs = [], []
    for jj in range(njack):
        val_epochs.append([])
        est_epochs.append([])

    offset=0
    for v in s_map.values():
        offset+=1
        np.array(rng.shuffle(v))
        val_size = len(v) * valfrac
        for jj in range(njack):
            jjj = (jj+offset) % njack
            vset = np.zeros(len(v), dtype=bool)
            vv=np.arange(np.round(jjj*val_size), np.round((jjj+1)*val_size), dtype=int)
            vset[vv] = True
            #v_epochs = [v[i] if b for i,b in enumerate(vset)]
            #e_epochs = v[1-vset]
            for b,vv in zip(vset, v):
                if b:
                    val_epochs[jj].append(vv)
                else:
                    est_epochs[jj].append(vv)
    log.info(val_epochs)
    val_times, est_times = [], []
    for jj in range(njack):
        # This returns the times of the validation and estimation sequences
        #import pdb
        #pdb.set_trace()
        #m = epochs['name'].apply(lambda x: x in val_epochs[jj])
        m = np.concatenate([rec['resp'].get_epoch_indices(x) for x in val_epochs[jj]])
        val_times.append(m)

        #m = epochs['name'].apply(lambda x: x in est_epochs[jj])
        m = np.concatenate([rec['resp'].get_epoch_indices(x) for x in est_epochs[jj]])
        est_times.append(m)
        #est_times.append(epochs.loc[m][['start', 'end']].values)

    return est_times, val_times


def shuffle_streams(rec):
    fg = rec['fg'].as_continuous().copy()
    bg = rec['bg'].as_continuous().copy()
    i_all = np.arange(fg.shape[-1])
    n = round(fg.shape[-1]/2)
    np.random.shuffle(i_all)
    i_switch = i_all[:n]
    fg[:, i_switch], bg[:, i_switch] = bg[:, i_switch], fg[:, i_switch]

    s = rec['fg']
    rec['fg'] = s._modified_copy(fg)
    rec['bg'] = s._modified_copy(bg)
    return rec


def rdt_shuffle(rec, shuff_streams=False, shuff_rep=False, **context):
    rec0 = rec.copy()
    if shuff_streams:
        rec0 = shuffle_streams(rec0)
    if shuff_rep:
        x = rec0['repeating'].as_continuous().copy().T
        np.random.shuffle(x)
        rec0['repeating'] = rec0['repeating']._modified_copy(x.T)
        # ix = rec0['state'].chans.index('repeating')
        # xall = rec0['state'].as_continuous().copy()
        # x = xall[ix, :]
        # np.random.shuffle(x)
        # xall[ix, :] = x
        # rec0['state'] = rec0['state']._modified_copy(xall)
    return {'rec': rec0}


def select_times(rec, subset, random_only=True, dual_only=True):
    '''
    Parameters
    ----------
    rec : nems.recording.Recording
        The recording object.
    subset : Nx2 array
        Epochs representing the selected subset (e.g., from an est/val split).
    random_only : bool
        If True, return only the repeating portion of the subset
    dual_only : bool
        If True, return only the dual stream portion of the subset
    '''
    epochs = rec['stim'].epochs

    m_dual = epochs['name'] == 'dual'
    m_repeating = epochs['name'] == 'repeating'
    m_trial = epochs['name'] == 'TRIAL'

    #dual_epochs = epochs.loc[m_dual, ['start', 'end']].values
    #repeating_epochs = epochs.loc[m_repeating, ['start', 'end']].values
    #trial_epochs = epochs.loc[m_trial, ['start', 'end']].values
    dual_epochs = rec['stim'].get_epoch_indices(m_dual)
    repeating_epochs = rec['stim'].get_epoch_indices(m_repeating)
    trial_epochs = rec['stim'].get_epoch_indices(m_trial)

    if random_only:
        subset = epoch_difference(subset, repeating_epochs)

    if dual_only:
        subset = epoch_intersection(subset, dual_epochs)

    #return rec.select_times(subset)
    return rec.create_mask(subset)
