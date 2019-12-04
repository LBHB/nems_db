"""
modules/state.py

functions for applying state-related transformations
"""

import numpy as np

def _state_dexp(x, s, g, d, base, amplitude, kappa):
   # Apparently, numpy is VERY slow at taking the exponent of a negative number
    # https://github.com/numpy/numpy/issues/8233
    # The correct way to avoid this problem is to install the Intel Python Packages:
    # https://software.intel.com/en-us/distribution-for-python
    sg = g @ s
    sd = d @ s
    sg = base[0] + amplitude[0] * np.exp(-np.exp(np.array(-np.exp(kappa[0])) * sg))
    sd = base[1] + amplitude[1] * np.exp(-np.exp(np.array(-np.exp(kappa[1])) * sd))

    return sg * x + sd


def state_dexp(rec, i, o, s, g, d, base, amplitude, kappa):
    '''
    Parameters
    ----------
    i name of input
    o name of output signal
    s name of state signal
    g - gain to scale s by
    d - dc to offset by
    base, amplitude, kappa - parameters for dexp applied to each state channel
    '''

    fn = lambda x : _state_dexp(x, rec[s]._data, g, d, base, amplitude, kappa)

    return [rec[i].transform(fn, o)]


def _state_exp(x, s, g):
    
    sg = np.exp(g[:, 1:] @ s[1:, :])
    base = g[:, 0][:, np.newaxis] @ s[0, :][np.newaxis, :]

    return (sg * x) + base


def state_exp(rec, i, o, s, g):
    '''
    pure state gain model with exp (following Rabinowitz 2015) 
    r[o] = r[i] * exp(g * r[s] + b)
    
    i: input
    o: output
    s: state signal(s)
    g: weight(s)
    (b = baseline -- first dim on state signal is baseline,
        so first dim of g are the baseline weights)

    CRH 12/3/2019
    '''
    
    fn = lambda x : _state_exp(x, rec[s]._data, g)

    return [rec[i].transform(fn, o)]