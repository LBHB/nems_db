'''
Functions for quantifying the strength of the gain control model's effect.
(WIP)

Functions:
----------
gc_magnitude (preferred so far)
gc_magnitude_with_ctpred

'''

import numpy as np

from nems.modules.nonlinearity import _logistic_sigmoid


def gc_magnitude(b, b_m, a, a_m, s, s_m, k, k_m):
    '''
    Compute the magnitude of the gain control response for a given set of
    dynamic_sigmoid parameters as the sum of the absolute differences
    between the high contrast and low contrast values.

    Parameters
    ----------
    (See dynamic_simgoid and nems.modules.nonlinearity._logistic_sigmoid)
    b : float
        base
    b_m : float
        base_mod
    a : float
        amplitude
    a_m : float
        amplitude_mod
    s : float
        shift
    s_m : float
        shift_mod
    k : float
        kappa
    k_m : float
        kappa_mod

    Returns
    -------
    mag : float

    '''
    mag = np.abs(b_m - b) + np.abs(a_m - a) + np.abs(k_m - k) + np.abs(s_m - s)
    return mag


def gc_magnitude_with_ctpred(ctpred, b, b_m, a, a_m, s, s_m, k, k_m):
    b = b + (b_m - b)*ctpred
    a = a + (a_m - a)*ctpred
    s = s + (s_m - s)*ctpred
    k = k + (k_m - k)*ctpred

    x_low = np.linspace(s[0]*-1, s[0]*3, 1000)

    # Can just use the first bin since they always start with silence
    b_low = b[0]
    a_low = a[0]
    s_low = s[0]
    k_low = k[0]

    some_contrast = ctpred[np.abs(ctpred - ctpred[0])/np.abs(ctpred[0]) > 0.02]
    high_contrast = ctpred > np.percentile(some_contrast, 50)
    b_high = np.median(b[high_contrast])
    a_high = np.median(a[high_contrast])
    s_high = np.median(s[high_contrast])
    k_high = np.median(k[high_contrast])

    x_high = np.linspace(s_high*-1, s_high*3, 1000)

    y_low = _logistic_sigmoid(x_low, b_low, a_low, s_low, k_low)
    y_high = _logistic_sigmoid(x_high, b_high, a_high, s_high, k_high)

    return np.mean(y_high - y_low)
