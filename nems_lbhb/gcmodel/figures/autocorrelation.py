import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from nems_lbhb.xform_wrappers import generate_recording_uri
from nems.recording import load_recording
import nems.db as nd
from nems_lbhb.gcmodel.figures.utils import (get_dataframes,
                                             improved_cells_to_list)
from nems_lbhb.gcmodel.figures.definitions import *

log = logging.getLogger(__name__)
plt.rcParams.update(params)

current_path = '/auto/users/jacob/notes/gc_rank3/autocorrelation/batch289_100hz_cutoff1000_run2.pkl'
previous_path = '/auto/users/jacob/notes/gc_rank3/autocorrelation/batch289_100hz_cutoff1000.pkl'
def load_batch_results(load_path):
    df = pd.read_pickle(load_path)
    return df


def autocorrelation_analysis(cellid, batch, sampling_rate=20, resp=None,
                             plot=False, add_noise=False):
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
        noise = (noise + 1)/2000       # 0 to 0.001
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

    # A, tau, B
    bounds = ([0, 10, -np.inf],[np.inf, 4000, np.inf])
    p0 = [0.2, 100, 0.2]

    # Fit to least squares between decaying exp and the autocorrelation data
    popt, pcov = curve_fit(decaying_exponential, times, autocorr, p0=p0,
                           bounds=bounds)

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
def tau_vs_model_performance(df, batch, gc, stp, LN, combined, bin_count=30,
                             good_ln=0.0, log_tau=False):

    # either, neither, gc, stp, combined
    e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           good_ln=good_ln, as_lists=False)
    # want categories to be exclusive in this case
    n = (a & np.logical_not(g) & np.logical_not(s) & np.logical_not(c))
    g1 = (g & np.logical_not(s))
    s1 = (s & np.logical_not(g))
    c1 = (c & np.logical_not(g) & np.logical_not(s))

    taus = df['tau']
    if log_tau:
        taus = np.log(taus)
    _filter_cells(taus, n, g1, s1, c1, log_tau)

    tau_range, bins, bar_width, axis_locs = _setup_bar(taus, n, g1, s1, c1,
                                                       bin_count)

    if log_tau:
        label='Log(Tau (ms))'
    else:
        label='Tau (ms)'

    fig = _stack_4_bar(taus, n, g1, s1, c1, tau_range, axis_locs, bar_width,
                       bins, xlabel=label)
    plt.tight_layout()
    return fig


def tau_vs_contrast_window(df, batch, gc30, gc60, LN, combined, bin_count=30):
    tau_vs_model_performance(df, batch, gc30, gc60, LN, combined, bin_count)


def _filter_cells(taus, LN_cells, gc_cells, stp_cells, both_cells,
                  log_tau=False):
    nan_mask = np.isnan(taus)
    upper_mask = (taus > 2000)
    lower_mask = (taus <= 0)

    for c in taus.index.values:
        for group in [LN_cells, gc_cells, stp_cells, both_cells]:
            if c not in group:
                group[c] = False

    for c in taus.index.values:
        for group in [LN_cells, gc_cells, stp_cells, both_cells]:
            if log_tau:
                group[nan_mask] = False
            else:
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
                 tau_range, axis_locs, bar_width, bins, xlabel='Tau (ms)'):
    LN_raw = np.histogram(taus[LN_cells], range=tau_range, bins=bins)[0]
    gc_raw = np.histogram(taus[gc_cells], range=tau_range, bins=bins)[0]
    stp_raw = np.histogram(taus[stp_cells], range=tau_range, bins=bins)[0]
    both_raw = np.histogram(taus[both_cells], range=tau_range, bins=bins)[0]
    n_LN = np.sum(LN_cells)
    n_gc = np.sum(gc_cells)
    n_stp = np.sum(stp_cells)
    n_combined = np.sum(both_cells)
    n_cells = np.sum(np.isfinite(taus))

    fig = plt.figure()
    plt.bar(axis_locs, LN_raw, width=bar_width, color=model_colors['LN'],
            alpha=0.8)
    plt.bar(axis_locs, gc_raw, width=bar_width, color=model_colors['gc'],
            alpha=0.8, bottom=LN_raw)
    plt.bar(axis_locs, stp_raw, width=bar_width, color=model_colors['stp'],
            alpha=0.8, bottom=LN_raw+gc_raw)
    plt.bar(axis_locs, both_raw, width=bar_width, color=model_colors['combined'],
            alpha=0.8, bottom=LN_raw+gc_raw+stp_raw)
    plt.xlabel(xlabel)
    plt.ylabel('Cell count')
    plt.title("Autocorrelation by model improvement category\n"
              "n:  %d" % n_cells)
    plt.legend(['LN, %d' % n_LN, 'gc, %d' % n_gc, 'stp, %d' % n_stp,
                'combined, %d' % n_combined])
    return fig
