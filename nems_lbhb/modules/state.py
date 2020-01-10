"""
modules/state.py

functions for applying state-related transformations
"""

import numpy as np
import nems_lbhb.preprocessing as preproc

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
    
    if g.shape[-1] > 1:
        sg = np.exp(g[:, 1:] @ s[1:, :])
        base = g[:, 0][:, np.newaxis] @ s[0, :][np.newaxis, :]
        return (sg * x) + base
    else:
        sg = np.exp(g @ s)
        return sg * x 


def state_exp(rec, i, o, s, g):
    '''
    pure state gain model with exp (following Rabinowitz 2015) 
    r[o] = r[i] * exp(g * r[s] + b)
    
    i: input
    o: output
    s: state signal(s)
    g: weight(s)

    CRH 12/3/2019
    '''
    
    fn = lambda x : _state_exp(x, rec[s]._data, g)

    return [rec[i].transform(fn, o)]


def _state_logsig(x, s, g, b, a):
    '''
    Gain is fixed to a max of 50 (this could be a free param)
    '''
    def fn(x):
        sig = a / (1 + np.exp(-x))
        return sig

    sg = fn(g @ s)

    return sg * x + b


def state_logsig(rec, i, o, s, g, b, a):
    '''
    State gain model with sigmoidal expansions/compression.
    r[o] = r[i] * sig(g * r[s])

    i: intput
    o: output
    s: state signal(s)
    g: weight(s)
    a: amplitude
    '''

    fn = lambda x: _state_logsig(x, rec[s]._data, g, b, a)

    return [rec[i].transform(fn, o)]

def _state_logsig_dcgain(x, s, g, d, b, a):
    '''
    Gain is fixed to a max of 50 (this could be a free param)
    '''
    def fn(x, a):
        sig = a / (1 + np.exp(-x))
        return sig

    sg = fn(g @ s, a[:, 0][:, np.newaxis])
    sd = fn(d @ s, a[:, 1][:, np.newaxis])

    return sg * x + sd + b

def state_logsig_dcgain(rec, i, o, s, g, d, b, a):
    '''
    State gain model with sigmoidal expansions/compression.
    r[o] = r[i] * sig(g * r[s])

    i: intput
    o: output
    s: state signal(s)
    g: weight(s)
    a: amplitude
    '''

    fn = lambda x: _state_logsig_dcgain(x, rec[s]._data, g, d, b, a)

    return [rec[i].transform(fn, o)]


def add_lv(rec, i, o, n, e):
    """
    Compute latent variable and add to state signals:
        projection of residual responses (resp minus current pred)
        onto encoding weights (e). Add a channel  of all 1's to the
        lv signal. This will be for offset in state models.
    
    i: 'resp'
    o: 'lv'
    e: encoding weights
    shuffle: bool (should you shuffle LV or not)
    """ 
    newrec = rec.copy()
    # CRH (12-13-2019) removing below code. 
    # Residual signal now gets created
    # (and shuffled) in preprocessing step
    #if cutoff is not None:
    #    # high pass filter resp before creating LV
    #    newrec = preproc.bandpass_filter_resp(newrec, low_c=cutoff, high_c=None)

    #res = newrec['resp'].rasterize()._data - newrec['pred']._data
    res = newrec['residual']._data
    lv = e.T @ res

    lv = np.concatenate((np.ones((1, lv.shape[-1])), lv), axis=0)

    # z-score lv? (to avoid pred blowing up)
    #lv = lv - lv.mean(axis=-1, keepdims=True)
    #if ~np.any(lv.std(axis=-1) == 0):
    #    lv = lv / lv.std(axis=-1, keepdims=True)

    lv_sig = newrec['resp'].rasterize()._modified_copy(lv)
    lv_sig.name = 'lv'
    nchans = e.shape[-1]
    lv_chans = []
    lv_chans.append('lv0')
    for c in range(nchans):
        lv_chans.append('lv{0}'.format(c+1))
    
    if len(n) > 0:
        if len(n) != nchans:
            raise ValueError("number of lv names must match number of LV chans!")
        for i, c in enumerate(lv_chans):
            if i != 0:
                # first chan is DC term, leave as lv0
                lv_chans[i] = 'lv_' + n[i-1]

    
    lv_sig.chans = lv_chans

    return [lv_sig]
