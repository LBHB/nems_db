'''
Contains new NEMS modules specific to the contrast-dependent gain control model.

Functions:
---------
dynamic_sigmoid: reimplements dexp and logsig such that its parameters are
                 allowed to vary based on the output of the contrast STRF.

weight_channels: as normal weight_channels, but parameters are applied to
                 filters for both stim and contrast.

fir:             as normal fir.basic, but parameters are applied to
                 filters for both stim and contrast.

levelshift:      as normal levelshift, but shift is applied to the coefficients
                 of both the stim STRF and the contrast STRF.

'''
import logging

import numpy as np
from scipy.signal import convolve2d

from nems.modules.fir import (per_channel, do_coefficients, _offset_coefficients,
                              fir_exp_coefficients)
from nems.modules.weight_channels import gaussian_coefficients
from nems.modules.nonlinearity import _logistic_sigmoid, _double_exponential

log = logging.getLogger(__name__)


def dynamic_sigmoid(rec, i, o, c, base, amplitude, shift, kappa,
                    base_mod=np.nan, amplitude_mod=np.nan, shift_mod=np.nan,
                    kappa_mod=np.nan, eq='dexp', norm=False, alternate=False):

    static = False
    for p in [base_mod, amplitude_mod, shift_mod, kappa_mod]:
        try:
            if not np.isnan(p):
                break
        except TypeError:
            break
        static = True

    if not static and rec[c]:
        contrast = rec[c].as_continuous()

        if norm:
            contrast = contrast/np.nanmax(contrast)

        if np.isnan(base_mod):
            base_mod = base
        if alternate:
            b = base + base_mod*contrast
        else:
            b = base + (base_mod - base)*contrast

        if np.isnan(amplitude_mod):
            amplitude_mod = amplitude
        if alternate:
            a = amplitude + amplitude_mod*contrast
        else:
            a = amplitude + (amplitude_mod - amplitude)*contrast

        if np.isnan(shift_mod):
            shift_mod = shift
        if alternate:
            s = shift + shift_mod*contrast
        else:
            s = shift + (shift_mod - shift)*contrast

        if np.isnan(kappa_mod):
            kappa_mod = kappa
        if alternate:
            k = kappa + kappa_mod*contrast
        else:
            k = kappa + (kappa_mod - kappa)*contrast
    else:
        # If there's no ctpred yet (like during initialization),
        # or if mods are all nan, no need to do anything with contrast,
        # so just pass through base, amplitude, shift and kappa as-is.
        b = base
        a = amplitude
        s = shift
        k = kappa

    if eq.lower() in ['logsig', 'logistic_sigmoid', 'l']:
        fn = lambda x: _logistic_sigmoid(x, b, a, s, k)
    elif eq.lower() in ['dexp', 'double_exponential', 'd']:
        fn = lambda x: _double_exponential(x, b, a, s, k)
    else:
        # Not a recognized equation, do logistic_sigmoid by default.
        fn = lambda x: _logistic_sigmoid(x, b, a, s, k)

    return [rec[i].transform(fn, o)]


# Compatibility function for a plot routine in nems.plots.scatter
# TODO: Better solution for this?
def _dynamic_sigmoid(x, base, amplitude, shift, kappa, base_mod, amplitude_mod,
                     shift_mod, kappa_mod):

    return _double_exponential(x, base, amplitude, shift, kappa)


def weight_channels(rec, i, o, ci, co, n_chan_in, mean, sd,
                    compute_contrast=True, **kwargs):
    '''
    Parameters
    ----------
    rec : recording
        Recording to transform
    i : string
        Name of input signal
    o : string
        Name of output signal
    ci : string
        Name of input signal for contrast portion
    co : string
        Name of output signal for contrast portion
    compute_contrast : boolean
        Skip contrast portion if False
    mean : array-like (between 0 and 1)
        Centers of Gaussian channel weights
    sd : array-like
        Standard deviation of Gaussian channel weights

    '''
    coefficients = gaussian_coefficients(mean, sd, n_chan_in)
    fn = lambda x: coefficients @ x
    new_signals = [rec[i].transform(fn, o)]
    if compute_contrast:
        gc_fn = lambda x: np.abs(coefficients) @ x
        new_signals.append(rec[ci].transform(gc_fn, co))

    return new_signals


def fir(rec, i, o, ci, co, coefficients=[], compute_contrast=True):
    """
    apply fir filters of the same size in parallel. convolve in time, then
    sum across channels

    Parameters
    ----------
    rec : recording
        Recording to transform
    i : string
        Name of input signal
    o : string
        Name of output signal
    ci : string
        Name of input signal for contrast portion
    co : string
        Name of output signal for contrast portion
    coefficients : 2d array
        all coefficients matrix shape=channel X time lag, for which
        .shape[0] matched to the channel count of the input
    compute_contrast : boolean
        Skip contrast portion if False

    """
    fn = lambda x: per_channel(x, coefficients)
    new_signals = [rec[i].transform(fn, o)]
    if compute_contrast:
        gc_fn = lambda x: per_channel(x, np.abs(coefficients))
        new_signals.append(rec[ci].transform(gc_fn, co))

    return new_signals


def levelshift(rec, i, o, ci, co, level, compute_contrast=True,
               block_contrast=False):
    '''
    Parameters
    ----------
    rec : recording
        Recording to transform
    i : string
        Name of input signal
    o : string
        Name of output signal
    ci : string
        Name of input signal for contrast portion
    co : string
        Name of output signal for contrast portion
    level : a scalar to add to every element of the input signal.
    compute_contrast : boolean
        Skip contrast portion if False
    block_contrast : boolean
        Skip contrast portion if True
        Second control used to stop fitting process from turning this
        computation on, i.e. only apply levelshift to pred for the
        entire model.

    '''
    fn = lambda x: x + level
    new_signals = [rec[i].transform(fn, o)]
    if compute_contrast and not block_contrast:
        gc_fn = lambda x: x + np.abs(level)
        rec[ci].transform(gc_fn, co)
    return new_signals


# TODO: currently hard-coded to use gaussian coeffs for the auto_copy portion,
#       need to be smarter about that.
#       the problem: non-gaussian wc and fir both use 'coefficients' as the
#       phi name, so they would overwrite eachother using the current
#       basic_with_copy implementation.
def contrast_kernel(rec, i, o, wc_coefficients=None, fir_coefficients=None,
                    mean=None, sd=None, coefficients=None, f1s=None, taus=None,
                    delays=None, gains=None, use_phi=False, offsets=None,
                    compute_contrast=False, n_coefs=18, auto_copy=None,
                    fixed=False, offset=None):
    # auto_copy is no longer used directly, but is included in the keyword
    # arguments in order to load old versions of the model that have
    # not been re-run
    if auto_copy is not None:
        use_phi = True

    if compute_contrast:
        fs = rec[i].fs
        coeffs, _, _ = _get_ctk_coefficients(
                wc_coefficients, fir_coefficients, mean, sd, coefficients,
                f1s, taus, delays, gains, use_phi, n_coefs, offsets, offset, fs
                )
        fn = lambda x: per_channel(x, coeffs)
        return [rec[i].transform(fn, o)]
    else:
        # pass through vector of 0's until contrast is ready to be computed
        fn = lambda x: np.zeros((1, x.shape[-1]))
        return [rec[i].transform(fn, o)]


def _get_ctk_coefficients(wc_coefficients=None, fir_coefficients=None, mean=None,
                         sd=None, coefficients=None, f1s=None, taus=None,
                         delays=None, gains=None, use_phi=False, n_coefs=18,
                         offsets=None, offset=None, fs=None, **kwargs):

    if use_phi:
        wc_coeffs = gaussian_coefficients(mean, sd, n_coefs)
        if coefficients is None:
            fir_coeffs = do_coefficients(f1s=f1s, taus=taus,
                                               delays=delays,
                                               gains=gains, n_coefs=n_coefs)
        else:
            fir_coeffs = coefficients
    else:
        if (wc_coefficients is None) or (fir_coefficients is None):
            raise ValueError("contrast_kernel module was called without "
                             "wc or fir coefficients set.")
        wc_coeffs = wc_coefficients
        fir_coeffs = fir_coefficients

    if (offsets is not None) and (fs is not None):
        fir_coeffs = _offset_coefficients(fir_coeffs, offsets, fs)
    elif (offsets is None) and (offset is not None):
        # old model
        offset = int(offset)
        fir_coeffs = np.concatenate((np.zeros((fir_coeffs.shape[0], offset)),
                                     fir_coeffs[:, :-1*offset]), axis=1)

    coeffs = np.abs(wc_coeffs.T @ fir_coeffs)
    return coeffs, wc_coeffs.T, fir_coeffs


# TODO: May still want to cache the "contrast" signal somehow, even though
#       it's not really pure contrast anymore?
def contrast(rec, tau, a, b, s, mean, sd, i='stim', o='ctpred', c='contrast',
             offsets=0.0, n_channels=18, n_coefs=15, compute_contrast=False):

    if compute_contrast:
        wc_coeffs = gaussian_coefficients(mean, sd, n_channels)
        fir_coeffs = fir_exp_coefficients(tau, a, b, s, n_coefs=15)
        if not np.all(offsets == 0):
            fs = rec[i].fs
            fir_coeffs = _offset_coefficients(fir_coeffs, offsets, fs,
                                              pad_bins=True)
        wc_coeffs = np.abs(wc_coeffs)
        fir_coeffs = np.abs(fir_coeffs)

        def fn(x):
            weighted = wc_coeffs.T * x
            weighted[np.isnan(weighted)] = 0
            width = wc_coeffs.shape[0]
            history = fir_coeffs.shape[-1]
            zero_pad = np.zeros([width, history-1])
            filt = np.concatenate((zero_pad, fir_coeffs), axis=1)
            filt /= np.sum(fir_coeffs)

            mn = convolve2d(weighted, filt, mode='same', fillvalue=0)
            var = convolve2d(weighted ** 2, filt, mode='same', fillvalue=0) - mn**2
            ct = np.sqrt(var) / (mn*.99 + np.nanmax(mn)*0.01)
            ctpred = np.sum(ct, axis=0)
            ctpred = np.expand_dims(ctpred, axis=0)

            return ctpred

    else:
        # pass through zeros until ready to fit GC portion of the model
        fn = lambda x: np.zeros((1, x.shape[-1]))

    return [rec[i].transform(fn, o)]


def summed_contrast_kernel(rec, offsets, i='contrast', o='ctpred',
                           compute_contrast=False, **kwargs):
    # **kwargs is just to catch some unused fn_kargs that normally get passed
    # to contrast_kernel.
    fs = rec[i].fs
    if compute_contrast:
        def fn(x):
            summed = np.expand_dims(np.sum(x, axis=0), axis=0)
            if not np.all(offsets == 0):
                summed = _offset_coefficients(summed, offsets=offsets, fs=fs,
                                              pad_bins=False)

            return summed

    else:
        # pass through zeros until ready to fit GC portion of the model
        fn = lambda x: np.zeros((1, x.shape[-1]))

    return [rec[i].transform(fn, o)]
