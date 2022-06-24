import numpy as np
import copy

def generate_simulated_trials(r1, r2=None, keep_stats=[1, 2], var_first_order=True, cov=True, N=500):
    """
    Generate simulated data for r1 dictionary that preserves either 
    first order statistics, second order statistics, or both (keep_stats=[1], [2], [1, 2])

    If r2 is not None, automatically use the first/second order stats of r2 (based on keep_stats)
    to generate new r1 data.
        For example generate_simulated_trials(r1, r2, keep_stats=[1], N=500) returns:
            new r1 data with each neuron's mean/sd determined from r1 and 
            covariance determined by cov(r2)

    Note - can also pass array type for r1/r2. Should be shape:
        neurons x reps x epochs x bins
    """

    # determine which data to use for calculation of first / second order statistics
    if r2 is not None:
        if keep_stats==[1]:
            first_order = copy.deepcopy(r1) 
            second_order = copy.deepcopy(r2)
        if keep_stats==[1, 2]:
            first_order = copy.deepcopy(r1)
            second_order = copy.deepcopy(r1)
        if keep_stats==[2]:
            first_order = copy.deepcopy(r2)
            second_order = copy.deepcopy(r1)
    else:
        first_order = copy.deepcopy(r1)
        second_order = copy.deepcopy(r1)

    if type(r1) is dict:
        # get epoch list
        ep = list(r1.keys())
        # now, for each time bin, and neuron, compute statistics and simulate new data
        r_new = dict.fromkeys(ep)
        nbins = first_order[ep[0]].shape[-1]
        nNeurons = first_order[ep[0]].shape[1]
        for e in ep:
            r_new[e] = np.zeros((N, nNeurons, nbins))
            for b in range(0, nbins):
                u = first_order[e][:, :, b].mean(axis=0)       # mean of all neurons over all trials from the first order dataset
                cor = np.corrcoef(second_order[e][:, :, b].T)  # covariance of all neurons from the second order dataset
                cor[np.isnan(cor)] = 0
                var = np.var(first_order[e][:, :, b], axis=0)          # variance of single neurons in the first order dataset

                # determine the new covariance matrix by scaling cor appropriately based on var
                # For neuron pair i, j:
                # cov_new(i, j) = corr(i, j) * sqrt(var(i) * var(j))
                # cov_new = cor * rootV

                # sqrt of outerproduct of variances (rootV)
                rootV = np.sqrt(np.matmul(var[:, np.newaxis], var[np.newaxis, :]))
                cov_new = cor * rootV

                # simulate new data with mean u and covariance cov_new
                r_new[e][:, :, b] = np.random.multivariate_normal(u, cov_new, (N))
    elif type(r1) is np.ndarray:
        nepochs = first_order.shape[2]
        nbins = first_order.shape[3]
        nNeurons = first_order.shape[0]
        r_new = np.zeros((nNeurons, N, nepochs, nbins))
        for e in range(r1.shape[2]):
            for b in range(r1.shape[-1]):
                u = first_order[:, :, e, b].mean(axis=1)       # mean of all neurons over all trials from the first order dataset
                if cov:
                    cor = np.cov(second_order[:, :, e, b])
                    #np.fill_diagonal(cor, np.var(first_order[:, :, e, b], axis=1, ddof=1))
                    cov_new = cor
                else:
                    cor = np.corrcoef(second_order[:, :, e, b])  # covariance of all neurons from the second order dataset
                    cor[np.isnan(cor)] = 0
                    if var_first_order:
                        # use first order dataset to define single neuron variance
                        var = np.var(first_order[:, :, e, b], axis=1, ddof=1)          # variance of single neurons in the first order dataset
                    else:
                        # use second order dataset to define single neurons variance
                        var = np.var(second_order[:, :, e, b], axis=1, ddof=1)
                    # determine the new covariance matrix by scaling cor appropriately based on var
                    # For neuron pair i, j:
                    # cov_new(i, j) = corr(i, j) * sqrt(var(i) * var(j))
                    # cov_new = cor * rootV

                    # sqrt of outerproduct of variances (rootV)
                    rootV = np.sqrt(np.matmul(var[:, np.newaxis], var[np.newaxis, :]))
                    cov_new = cor * rootV

                # simulate new data with mean u and covariance cov_new
                r_new[:, :, e, b] = np.random.multivariate_normal(u, cov_new, (N)).T

    else:
        raise TypeError("Unexpected input datatype")

    return r_new


# ===================== Old functions ================================

def generate_uncorrelated_trials(rec, epochs=None, N=500):
    """
    For the data contained in the epochs specified, and rec['mask'],
    compute the mean/std of all reps of each unique stimulus bin. Using
    these params, generate many (500) pseudo trials. Because this is too
    many trials to shove back into the recording, we have to
    return a dict of numpy arrays (trials x neuron x time) where the
    keys are each unique epoch.

    Purpose is to generate pseudo data with zero correlations
    between neurons.

    CRH -- 09/17/2019
    """

    if type(rec) is not dict:
        if 'mask' not in rec.signals:
            rec = rec.create_mask(True)

        if epochs is not None:
            rec = rec.and_mask(epochs)
        else:
            epochs = np.unique([ep for ep in rec['resp'].epochs.name if 'STIM' in ep]).tolist()

        r = rec.apply_mask(reset_epochs=True)
        r = r.and_mask(['PostStimSilence'], invert=True)
        r = r.apply_mask(reset_epochs=True)

        data = r['resp'].extract_epochs(epochs)
    else:
        data = rec

    pseudo_data = dict.fromkeys(data.keys())
    n_units = data[list(data.keys())[0]].shape[1]
    for ep in data.keys():
        nsegs = data[ep].shape[-1]
        pseudo_data[ep] = np.zeros((N, n_units, nsegs))
        for s in range(nsegs):
            for n in range(0, n_units):
                u = data[ep][:, n, s].mean()
                sd = data[ep][:, n, s].std()
                pseudo_data[ep][:, n, s] = np.random.normal(u, sd, N)

    return pseudo_data


def generated_correlated_trials(r_state, r_all, N=500):
    """
    Generate simulated data with single neuron variance / mean calculated from
    all data (r_all) and covariance specified by r_state.
    For example, r_state could be dictionary of all big pupil trials and r_all
    is dictionary including all trials. Compute single neuron stats on r_all,
    compute covariance matrix on r_state, generate new data.

    r_state / r_all: dictionaries. Keys = epochs. Each value is np.array of
        shape: trial x neuron x time.
        neuron / time dimensions must match between the two dictionaries
    """

    if len(list(r_state.keys())) != len(list(r_all.keys())):
        raise ValueError("mismatched epochs between r_state and r_all")
    else:
        epochs = list(r_state.keys())

    if r_state[epochs[0]].shape[1] != r_all[epochs[0]].shape[1]:
        raise ValueError("mismatched neurons between r_state and r_all")

    if r_state[epochs[0]].shape[2] != r_all[epochs[0]].shape[2]:
        raise ValueError("mismatched trial length between r_state and r_all")

    new_data = dict.fromkeys(r_state.keys())
    for ep in epochs:
        for b in range(0, m_all.shape[-1]):
            # get mean of each neuron across all states
            u_all = r_all[ep][:, :, b].mean(axis=0)
            # get variance of each single neuron across all states
            cov_all = np.diagonal(np.cov(r_all[ep][:, :, b].T))
            # get covariance matrix for the current state (big/small pupil)
            cov_state = np.cov(r_state[ep][:, :, b].T)
            # replace state-dependent single neuron covariance with single
            # neuron variance across states
            np.fill_diagonal(cov_state, cov_all)
            # generate data
            new_data[ep][:, :, b] = np.random.multivariate_normal(u_all, cov_state, (N))

    return new_data
