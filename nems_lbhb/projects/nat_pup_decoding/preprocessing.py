"""
Some very basic tools for matrix reshaping, used to get data 
in the correct shape for doing decoding analysis smoothly.

(and for some preprocessing normalization / est/val splitting steps)

CRH 04/10/2020
"""

from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis

import logging

log = logging.getLogger()

def dict_to_X(d):
    """
    Transform dictionary of spike counts (returned by nems.recording.extract_epochs)
    into a matrix of shape: (Neuron X Repetition X Stimulus X Time)
    """
    
    epochs = d.keys()
    # figure out min reps (if they're not the same for each stimulus)
    reps = [d[e].shape[0] for e in d.keys()]
    min_even_reps = np.min(reps)
    if min_even_reps % 2 == 0:
        pass
    else:
        min_even_reps -= 1

    if np.max(np.abs(np.diff(reps)))==0:
        for i, epoch in enumerate(epochs):
            r_epoch = d[epoch].transpose(1, 0, -1)[:, :, np.newaxis, :]
            if i == 0:
                X = r_epoch
            else:
                # stack on stimuli (epochs)
                X = np.append(X, r_epoch, axis=2)
    else:
        log.info(f"WARNING: Need to choose subset of reps for certain stim, since reps aren't even \n Max rep n: {max(reps)}, choosing {min_even_reps} from each stim to balance")
        for i, epoch in enumerate(epochs):
            # don't choose randomly bc then wouldn't line up with other data that gets tranformed
            choose = np.arange(min_even_reps)
            r_epoch = d[epoch].transpose(1, 0, -1)[:, choose, np.newaxis, :]
            if i == 0:
                X = r_epoch
            else:
                # stack on stimuli (epochs)
                X = np.append(X, r_epoch, axis=2)
    
    return X


def flatten_X(X):
    """
    Transform X matrix (Neuron X Stimulus X Repetition X Time) into 
    matrix of Neurons X Repetition * Stimulus * Time
    """
    if len(X.shape) != 4: raise ValueError("Input has unexpected shape")
    return X.reshape(X.shape[0], -1)


def fold_X(X_flat, nstim, nreps, nbins):
    """
    Invert the transformation done by cat_stimuli. 
        X_flat is shape (Neurons X Stimulus * Repetition * Time)
        and this fn returns matrix of shape (Neurons X Repetion X Stimulus X Time)
    """
    if len(X_flat.shape) != 2: raise ValueError("Input has unexpected shape")
    return X_flat.reshape(X_flat.shape[0], nreps, nstim, nbins)


def get_est_val_sets(X, pup_mask=None, njacks=10, est_equal_val=False):
    """
    njacks 50-50 splits of the data matrix X.

    X is shape Neurons X Reps X Stim.

    if est_equal_val, just return two copies of the raw data. This 
    is to be used for cases when too few reps exist for cross validation
    """
    nreps = X.shape[1]
    n_test_reps = int(nreps / 2)
    all_idx = np.arange(0, nreps)

    est = []
    val = []
    p_est = []
    p_val = []
    test_idx = []
    count = 0
    while count < njacks:
        ti = list(np.random.choice(all_idx, n_test_reps, replace=False))
        if ti not in test_idx:
            tri = list(set(all_idx) - set(ti))
        
            test_idx.append(ti)

            est.append(X[:, ti, :])
            val.append(X[:, tri, :])

            if pup_mask is not None:
                p_est.append(pup_mask[:, ti, :])
                p_val.append(pup_mask[:, tri, :])
        
            count += 1
        else:
            pass

    if pup_mask is not None:
        if est_equal_val:
            return [X], [X], [pup_mask], [pup_mask]
        else:
            return est, val, p_est, p_val
    else:
        if est_equal_val:
            return [X], [X]
        else:
            return est, val


def scale_est_val(est, val, full=None, mean=True, sd=True):
    """
    Scale est / val datasets and return normalized data.
    If mean true, mean center data
    If sd true, normalize by std

    val gets normalized according to stats measured in est

    29.10.2021 - add "full" option to scale the full recording acording to this
    est set too. Idea is that we can try projecting the full dataset into the same
    scaled space for pupil analysis, since otherwise pupil isn't guaranteed to have
    enough reps in each state, since the jackknife split just goes across all data and 
    splits 50-50
    """
    est_new = est.copy()
    val_new = val.copy()

    if len(est) != len(val):
        raise ValueError("len of est and val must be the same")

    if (len(est[0].shape) != 3) | (len(val[0].shape) != 3):
        raise ValueError("est / val elements must be shape neuron x rep x stim")

    if full is not None:
        if len(full.shape) == 3:
            full_new = []
            for i in range(len(est)):
                full_new.append(full.copy())

    sets = len(est)

    for i in range(sets):
        u = est[i].mean(axis=(1, 2), keepdims=True)
        std = est[i].std(axis=(1, 2), keepdims=True)

        if mean:
            est_new[i] = est_new[i] - u
            val_new[i] = val_new[i] - u
            if full is not None:
                full_new[i] = full_new[i] - u

        if sd:
            if np.any(std==0):
                std[np.where(std==0)] = 1
            est_new[i] = est_new[i] / std
            val_new[i] = val_new[i] / std
            if full is not None: 
                full_new[i] = full_new[i] / std

    if full is not None:
        return est_new, val_new, full_new
    else:
        return est_new, val_new


def get_first_pc_per_est(est, method='pca'):
    """
    est is a list of validation response matrices. For each matrix, 
    calculate the first PC of the noise (stimulus subtracted).
    Return a list with the PC weights for each val set.
    """
    # each el in val is shape (neuron x reps x stim)
    if method=='pca':
        pcs = []
        for e in est:
            residual = e - e.mean(axis=1, keepdims=True)
            pca = PCA(n_components=1)
            pca.fit(residual.reshape(residual.shape[0], -1).T)
            pcs.append(pca.components_)
        return pcs

    elif method=='fa':
        factors = []
        for e in est:
            residual = e - e.mean(axis=1, keepdims=True)
            fa = FactorAnalysis(n_components=1, random_state=0)
            fa.fit(residual.reshape(residual.shape[0], -1).T)
            factors.append(fa.components_ / np.linalg.norm(fa.components_))
        return factors


def get_pupil_range(X_pup, pmask):
    """
    pup_mask is a mask of shape X_pup that splits each stim bin in 
    X_pup in half based on the median pupil during those stim trials.
    Goal here is to return a metric describing what fraction of total pupil variance 
    is traversed during these trials.
        e.g. might have a given stimulus that was only presented when pupil was very small
            want to keep track of this because pupil-dep. results might be weaker for this 
            stimulus.
    """
    max_pup = X_pup.max()
    min_pup = X_pup.min()
    full_range = max_pup - min_pup

    df = pd.DataFrame()

    # for each stim:
    for stim in range(X_pup.shape[-1]):
        # what fraction of full range does med(big) - med(small) span?
        try:
            bp = X_pup[0, pmask[0, :, stim], stim]
            sp = X_pup[0, ~pmask[0, :, stim], stim]
            _range = np.median(bp) - np.median(sp)
            _range /= full_range

            # get min / max of each stim as fraction of total max
            _max = bp.max() / max_pup
            _min = sp.min() / max_pup

            # get variance in big / small (is it roughly balanced?)
            _bp_var = np.var(bp) / np.var(X_pup)
            _sp_var = np.var(sp) / np.var(X_pup)

            results = {'range': _range,
                    'max': _max,
                    'min': _min,
                    'bp_var': _bp_var,
                    'sp_var': _sp_var,
                    'stim': stim}
        except ValueError:
                # happens if/when pup_mask is empty for a stim. This should only be the case for specialized pupil spliting code
                    results = {'range': np.nan,
                    'max': np.nan,
                    'min': np.nan,
                    'bp_var': np.nan,
                    'sp_var': np.nan,
                    'stim': stim}
        
        df = df.append([results])
        #df = pd.concat([df,pd.DataFrame(results,index=[0])], ignore_index=True)

    # add overall results to df to compare across sites
    results = {'range': full_range,
               'max': max_pup,
               'min': min_pup,
               'bp_var': np.nan,
               'sp_var': np.nan,
               'stim': 'all'}
    df = df.append([results])
    #df = pd.concat([df,pd.DataFrame(results,index=[0])], ignore_index=True)
    return df
