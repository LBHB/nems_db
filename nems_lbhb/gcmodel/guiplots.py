import copy

import numpy as np

import nems.modelspec as ms
from nems.plots.heatmap import plot_heatmap
from nems.plots.timeseries import timeseries_from_signals
from nems.plots.specgram import plot_spectrogram
from nems_lbhb.gcmodel.modules import _get_ctk_coefficients
import nems.utils as nu


def contrast_kernel_output(rec, modelspec, ax=None, title=None,
                           idx=0, channels=0, xlabel='Time', ylabel='Value',
                           **options):

    output = ms.evaluate(rec, modelspec, stop=idx+1)['ctpred']
    timeseries_from_signals([output], channels=channels, xlabel=xlabel,
                                 ylabel=ylabel, ax=ax, title=title)

    return ax


def contrast_kernel_heatmap(rec, modelspec, ax=None, title=None,
                            idx=0, channels=0, xlabel='Lag (s)',
                            ylabel='Channel In', **options):

    ctk_idx = nu.find_module('contrast_kernel', modelspec)
    phi = copy.deepcopy(modelspec[ctk_idx]['phi'])
    fn_kwargs = copy.deepcopy(modelspec[ctk_idx]['fn_kwargs'])
    old = ('auto_copy' in fn_kwargs)
    if old:
        fn_kwargs['use_phi'] = True

    # Remove duplicates from fn_kwargs (phi is more up to date)
    # to avoid argument collisions
    removals = []
    for k in fn_kwargs:
        if k in phi:
            removals.append(k)
    for k in removals:
        fn_kwargs.pop(k)

    strf, wc_coefs, fir_coefs = _get_ctk_coefficients(**fn_kwargs, **phi)

    # Show factorized coefficients on the edges to match up with
    # regular STRF
    cscale = np.nanmax(np.abs(strf.reshape(-1)))
    clim = [-cscale, cscale]
    wc_max = np.nanmax(np.abs(wc_coefs[:]))
    fir_max = np.nanmax(np.abs(fir_coefs[:]))
    wc_coefs = wc_coefs * (cscale / wc_max)
    fir_coefs = fir_coefs * (cscale / fir_max)
    n_inputs, _ = wc_coefs.shape
    nchans, ntimes = fir_coefs.shape
    gap = np.full([nchans + 1, nchans + 1], np.nan)
    horz_space = np.full([1, ntimes], np.nan)
    vert_space = np.full([n_inputs, 1], np.nan)
    top_right = np.concatenate([fir_coefs, horz_space], axis=0)
    top_left = np.concatenate([wc_coefs, vert_space], axis=1)
    bot = np.concatenate([top_left, strf], axis=1)
    top = np.concatenate([gap, top_right], axis=1)
    everything = np.concatenate([top, bot], axis=0)
    skip = nchans + 1

    plot_heatmap(everything, xlabel=xlabel, ylabel=ylabel, ax=ax, skip=skip,
                 clim=clim, title=title)

    return ax


def contrast_spectrogram(rec, modelspec, ax=None, title=None,
                         idx=0, channels=0, xlabel='Time', ylabel='Value',
                         **options):

    contrast = rec['contrast']
    array = contrast.as_continuous()
    ax = plot_spectrogram(array, ax=ax, fs=contrast.fs, **options)

    return ax
