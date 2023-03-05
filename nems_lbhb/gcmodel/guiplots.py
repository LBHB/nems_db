import copy

import numpy as np
import matplotlib.pyplot as plt

import nems0.modelspec as ms
from nems0.plots.heatmap import plot_heatmap
from nems0.plots.timeseries import timeseries_from_signals
from nems0.plots.specgram import plot_spectrogram
from nems_lbhb.gcmodel.modules import _get_ctk_coefficients
from nems0.modules.fir import fir_exp_coefficients, _offset_coefficients
from nems0.modules.weight_channels import gaussian_coefficients
import nems0.utils as nu
from nems0.gui.decorators import scrollable

@scrollable
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
    fs = rec['stim'].fs
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

    strf, wc_coefs, fir_coefs = _get_ctk_coefficients(**fn_kwargs, **phi, fs=fs)
    _strf_heatmap(strf, wc_coefs, fir_coefs, xlabel=xlabel, ylabel=ylabel,
                  ax=ax, title=title)

    return ax


def contrast_kernel_heatmap2(rec, modelspec, ax=None, title=None,
                             idx=0, channels=0, xlabel='Lag (s)',
                             ylabel='Channel In', **options):

    ct_idx = nu.find_module('contrast', modelspec)
    phi = copy.deepcopy(modelspec[ct_idx]['phi'])
    fn_kwargs = copy.deepcopy(modelspec[ct_idx]['fn_kwargs'])
    fs = rec['stim'].fs

    wc_kwargs = {k: phi[k] for k in ['mean', 'sd']}
    wc_kwargs['n_chan_in'] = fn_kwargs['n_channels']
    fir_kwargs = {k: phi[k] for k in ['tau', 'a', 'b', 's']}
    fir_kwargs['n_coefs'] = fn_kwargs['n_coefs']
    wc_coefs = gaussian_coefficients(**wc_kwargs)
    fir_coefs = fir_exp_coefficients(**fir_kwargs)
    if 'offsets' in phi:
        offsets = phi['offsets']
    elif 'offsets' in fn_kwargs:
        offsets = fn_kwargs['offsets']
    else:
        offsets = None
    if offsets is not None:
        fir_coefs = _offset_coefficients(fir_coefs, offsets, fs, pad_bins=True)

    wc_coefs = np.abs(wc_coefs).T
    fir_coefs = np.abs(fir_coefs)
    strf = wc_coefs @ fir_coefs

    # TODO: This isn't really doing the same operation as an STRF anymore
    #       so it may be better not to plot it this way in the future.
    _strf_heatmap(strf, wc_coefs, fir_coefs, xlabel=xlabel, ylabel=ylabel,
                  ax=ax, title=title)

    return ax


def summed_contrast(rec, modelspec, ax=None, title='Summed Contrast', idx=0,
                    channels=0, xlabel='Time (s)', ylabel='A.U.', **options):

    ctpred = rec.apply_mask()['ctpred']
    ax = timeseries_from_signals([ctpred], title=title, xlabel=xlabel,
                                 ylabel=ylabel, ax=ax)
    return ax


def _strf_heatmap(strf, wc_coefs, fir_coefs, xlabel='Lag (s)', ylabel='Channel',
                  ax=None, title=None):

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


@scrollable
def contrast_spectrogram(rec, modelspec, ax=None, title=None,
                         idx=0, channels=0, xlabel='Time', ylabel='Value',
                         **options):

    contrast = rec['contrast']
    array = contrast.as_continuous()
    ax = plot_spectrogram(array, ax=ax, fs=contrast.fs, **options)

    return ax
