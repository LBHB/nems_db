import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from nems_lbhb.xform_wrappers import generate_recording_uri
from nems.recording import load_recording
import nems.db as nd
from nems_lbhb.gcmodel.figures.utils import get_dataframes

log = logging.getLogger(__name__)


current_path = '/auto/users/jacob/notes/gc_rank3/autocorrelation/batch289_100hz_cutoff1000.pkl'
def load_batch_results(load_path):
    df = pd.read_pickle(load_path)
    return df


def autocorrelation_analysis(cellid, batch, sampling_rate=20, resp=None,
                             plot=False, add_noise=True):
    if resp is None:
        # load recording
        loadkey = 'ozgf.fs%d.ch18' % sampling_rate
        recording_uri = generate_recording_uri(cellid=cellid, batch=batch,
                                               loadkey=loadkey, stim=False)
        rec = load_recording(recording_uri)
        resp = rec['resp'].extract_channels([cellid]).as_continuous().flatten()

    # break into chunks of pre-stim silence, treat each one as a "trial"
    trials = []
    epochs = rec.epochs
    silence_epochs = epochs[epochs.name == 'PreStimSilence']
    shape = None
    for index, row in silence_epochs.iterrows():
        start = int(row['start']*sampling_rate)
        end = int(row['end']*sampling_rate)
        this_resp = resp[start:end]
        if shape is None:
            shape = this_resp.shape
        elif this_resp.shape != shape:
            # only use trials with same length
            continue
        trials.append(this_resp)

    # stack trials by bin within silence, e.g. first "column" is all bins
    # that were 0ms from the start of pre-silence period.
    stacked = np.vstack(trials).T

    if add_noise:
        # add noise (with values much smaller than spike counts)
        # so that sparse recordings don't get counted as 0 standard deviation
        # and cause nans, but non-sparse epochs will be mostly unchanged
        noise = np.random.rand(*stacked.shape)
        noise /= np.abs(noise.max())  # -1 to 1
        noise = (noise + 1)/200       # 0 to 0.01
        stacked += noise

    # compute pearson's R across trials for all pairs of bins.
    # e.g. correlate first bin to 2nd bin, and to 3rd bin, etc., then
    # repeat for all other bins.
    r = np.corrcoef(stacked)

    if plot:
        plt.figure()
        plt.imshow(r, aspect='auto')

    return r


def autocorrelation_batch_average(batch, sampling_rate=20, test_limit=None,
                                  add_noise=False, cutoff=1000, save_path=None):
    # do autocorrelation_analysis for each cell
    # then take average
    cells = nd.get_batch_cells(batch, as_list=True)
    rs = {}
    for c in cells[:test_limit]:
        r = autocorrelation_analysis(c, batch, sampling_rate,
                                     add_noise=add_noise)
        rs[c] = r

    # Not all correlation matrices are the same shape since some recordings
    # are longer than others, so have to pad with nans before we can make
    # a single array to average over
    max_dim = np.NINF
    for r in rs.values():
        a0, a1 = r.shape
        if a0 >= max_dim:   # only have to compare one dim since all square
            max_dim = a0

    padded_arrays = []
    for r in rs.values():
        a0, a1 = r.shape
        diff = max_dim - a0
        if diff > 0:
            padded = np.pad(r, ((0, diff), (0, diff)), 'constant',
                            constant_values=np.nan)
        else:
            padded = r
        padded_arrays.append(padded)

    # using full duration seems like it biases the cost function to
    # fit the asymptote since most of the time points are near there,
    # so try cutting off at a shorter time point.
    # (also, later time points have less data to support them)
    ms_bin_size = 1000/sampling_rate
    reduced_dim = int(cutoff / ms_bin_size)
    stacked = np.dstack(padded_arrays)
    reduced = stacked[:reduced_dim, :reduced_dim]
    mean_r = np.nanmean(reduced, axis=2)

    autocorr, bin_lags = autocorr_by_bin_lag(mean_r)
    times = bin_lags * ms_bin_size
    A, tau, B = autocorr_decay_fit(mean_r, sampling_rate=sampling_rate)
    d_exp = decaying_exponential(times, A, tau, B)

    results = {}
    results['cellid'] = ['mean']
    results['correlation_matrix'] = [mean_r]
    results['autocorrelation_fn'] = [autocorr]
    results['decaying_exp_fit'] = [d_exp]
    results['A'] = [A]
    results['tau'] = [tau]
    results['B'] = [B]
    for c, r in rs.items():
        autocorr, bin_lags = autocorr_by_bin_lag(r)
        times = bin_lags * ms_bin_size
        try:
            A, tau, B = autocorr_decay_fit(r, sampling_rate=sampling_rate)
            d_exp = decaying_exponential(times, A, tau, B)
        except ValueError:
            # nan in r
            A, tau, B = (np.nan, np.nan, np.nan)
            d_exp = np.full_like(autocorr, np.nan)
        except RuntimeError:
            # couldn't find solution
            A, tau, B = (np.nan, np.nan, np.nan)
            d_exp = np.full_like(autocorr, np.nan)

        results['cellid'].append(c)
        results['correlation_matrix'].append(mean_r)
        results['autocorrelation_fn'].append(autocorr)
        results['decaying_exp_fit'].append(d_exp)
        results['A'].append(A)
        results['tau'].append(tau)
        results['B'].append(B)

    df = pd.DataFrame.from_dict(results)
    df.set_index('cellid', inplace=True)

    if save_path is not None:
        df.to_pickle(save_path)

    return df


def autocorr_decay_fit(r, sampling_rate=20, maxiter=1000, tolerance=1e-12):
    # given a correlation matrix from autocorrelation_analysis,
    # fit a decaying exponential for autocorrelation as a function of
    # time lag k between bins, with the form: R(t) = A*(exp(-k/tau)+B)

    # Convert correlation matrix to a 1d array of mean auto correlation
    # for each time lag
    autocorr, bin_lags = autocorr_by_bin_lag(r)
    ms_bin_size = 1000/sampling_rate
    times = bin_lags*ms_bin_size

    # Fit to least squares between decaying exp and the autocorrelation data
    popt, pcov = curve_fit(decaying_exponential, times, autocorr)

    return popt


def autocorr_by_bin_lag(r):
    # select upper triangular portion of matrix, offset from diagonal by 1
    idx = np.triu_indices(r.shape[0], 1)
    flat_triangle = r[idx]
    bin_lags = np.arange(r.shape[0]-1) + 1
    ac = np.empty_like(bin_lags, dtype=np.float64)

    for i in (bin_lags-1):
        this_bin = []
        k = i
        j = bin_lags[-1]
        while j > i:
            #print('i: %s, k: %s' % (i,k))
            this_bin.append(flat_triangle[k])
            k += j
            j -= 1

        ac[i] = np.mean(this_bin)

    return np.array(ac), bin_lags


def decaying_exponential(t, A, tau, B):
    return A*(np.exp(-t/tau)+B)


# Uses df from batch analysis above
def tau_vs_model_performance(df, batch, gc, stp, LN, combined, bin_count=60):

    LN_cells, gc_cells, stp_cells, both_cells = _get_cells(df, batch, gc, stp,
                                                           LN, combined)
    _filter_cells(df, LN_cells, stp_cells, gc_cells, both_cells)
    taus = df['tau']
    tau_range, bins, bar_width, axis_locs = _setup_bar(taus, LN_cells, gc_cells,
                                                       stp_cells, both_cells,
                                                       bin_count)

    _stack_4_bar(taus, LN_cells, gc_cells, stp_cells, both_cells,
                     tau_range, axis_locs, bar_width, bins)


def tau_vs_contrast_window(df, batch, gc30, gc60, LN, combined, bin_count=60):
    tau_vs_model_performance(df, batch, gc30, gc60, LN, combined, bin_count)


def _get_cells(df, batch, gc, stp, LN, combined):
    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    cellids = df_r[LN] > df_e[LN]*2
    gc_LN_SE = (df_e[gc] + df_e[LN])
    stp_LN_SE = (df_e[stp] + df_e[LN])
    gc_cells = (cellids) & ((df_r[gc] - df_r[LN]) > gc_LN_SE)
    stp_cells = (cellids) & ((df_r[stp] - df_r[LN]) > stp_LN_SE)
    both_cells = gc_cells & stp_cells
    gc_cells = gc_cells & np.logical_not(both_cells)
    stp_cells = stp_cells & np.logical_not(both_cells)
    LN_cells = cellids & np.logical_not(gc_cells | stp_cells | both_cells)

    return LN_cells, gc_cells, stp_cells, both_cells


def _filter_cells(df, LN_cells, gc_cells, stp_cells, both_cells):
    taus = df['tau']
    nan_mask = np.isnan(taus)
    upper_mask = (taus > 1000)
    lower_mask = (taus <= 0)

    for c in df.index.values:
        for group in [LN_cells, gc_cells, stp_cells, both_cells]:
            if c not in group:
                group[c] = False

    for c in df.index.values:
        for group in [LN_cells, gc_cells, stp_cells, both_cells]:
            group[nan_mask | upper_mask | lower_mask] = False


def _setup_bar(taus, LN_cells, gc_cells, stp_cells, both_cells, bin_count):
    all_taus = taus[LN_cells | gc_cells | stp_cells | both_cells]
    max_tau = all_taus.max()
    min_tau = all_taus.min()
    tau_range = (min_tau, max_tau)
    bins = np.linspace(min_tau, max_tau, bin_count)
    bar_width = bins[1]-bins[0]
    axis_locs = bins[:-1]

    return tau_range, bins, bar_width, axis_locs


def _stack_4_bar(taus, LN_cells, gc_cells, stp_cells, both_cells,
                 tau_range, axis_locs, bar_width, bins):
    LN_raw = np.histogram(taus[LN_cells], range=tau_range, bins=bins)[0]
    gc_raw = np.histogram(taus[gc_cells], range=tau_range, bins=bins)[0]
    stp_raw = np.histogram(taus[stp_cells], range=tau_range, bins=bins)[0]
    both_raw = np.histogram(taus[both_cells], range=tau_range, bins=bins)[0]

    fig = plt.figure(figsize=figsize)
    plt.bar(axis_locs, LN_raw, width=bar_width, color='gray', alpha=0.8)
    plt.bar(axis_locs, gc_raw, width=bar_width, color='maroon', alpha=0.8,
            bottom=LN_raw)
    plt.bar(axis_locs, stp_raw, width=bar_width, color='teal', alpha=0.8,
            bottom=LN_raw+gc_raw)
    plt.bar(axis_locs, both_raw, width=bar_width, color='goldenrod', alpha=0.8,
            bottom=LN_raw+gc_raw+stp_raw)

    plt.legend(['LN', 'm1', 'm2', 'both'])
