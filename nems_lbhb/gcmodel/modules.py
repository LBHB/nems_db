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

import numpy as np

from nems.modules.fir import per_channel
from nems.modules.weight_channels import gaussian_coefficients
from nems.modules.nonlinearity import _logistic_sigmoid, _double_exponential


def dynamic_sigmoid(rec, i, o, c, base, amplitude, shift, kappa,
                    base_mod=np.nan, amplitude_mod=np.nan, shift_mod=np.nan,
                    kappa_mod=np.nan, eq='logsig'):

    static = False
    if np.all(np.isnan([base_mod, amplitude_mod, shift_mod, kappa_mod])):
        static = True

    if not static and rec[c]:
        contrast = rec[c].as_continuous()

        if np.isnan(base_mod):
            base_mod = base
        b = base + (base_mod - base)*contrast

        if np.isnan(amplitude_mod):
            amplitude_mod = amplitude
        a = amplitude + (amplitude_mod - amplitude)*contrast

        if np.isnan(shift_mod):
            shift_mod = shift
        s = shift + (shift_mod - shift)*contrast

        if np.isnan(kappa_mod):
            kappa_mod = kappa
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
    elif eq.lower() == ['dexp', 'double_exponential', 'd']:
        fn = lambda x: _double_exponential(x, b, a, s, k)
    else:
        # Not a recognized equation, do logistic_sigmoid by default.
        fn = lambda x: _logistic_sigmoid(x, b, a, s, k)

    return [rec[i].transform(fn, o)]


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


def levelshift(rec, i, o, ci, co, level, compute_contrast=True):
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

    '''
    fn = lambda x: x + level
    new_signals = [rec[i].transform(fn, o)]
    if compute_contrast:
        gc_fn = lambda x: x + np.abs(level)
        rec[ci].transform(gc_fn, co)
    return new_signals
