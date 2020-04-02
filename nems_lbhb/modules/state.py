"""
modules/state.py

functions for applying state-related transformations
"""

import numpy as np
import nems_lbhb.preprocessing as preproc

def _state_dexp(x, s, base_g, amplitude_g, kappa_g, offset_g, base_d, amplitude_d, kappa_d, offset_d):
    '''
     Apparently, numpy is VERY slow at taking the exponent of a negative number
     https://github.com/numpy/numpy/issues/8233
     The correct way to avoid this problem is to install the Intel Python Packages:
     https://software.intel.com/en-us/distribution-for-python

     "current" version of sdexp. separate kappa/amp/base phis for gain/dc.
     So all parameters (g, d, base_g, etc.) are the same shape.
     '''

    sg = base_g.T + amplitude_g.T * np.exp(-np.exp(np.array(-np.exp(kappa_g.T)) * (s - offset_g.T)))
    sd = base_d.T + amplitude_d.T * np.exp(-np.exp(np.array(-np.exp(kappa_d.T)) * (s - offset_d.T)))

    sg = sg.sum(axis=0)[np.newaxis, :]
    sd = sd.sum(axis=0)[np.newaxis, :]

    return sg * x + sd, sg, sd


def _state_dexp_old(x, s, g, d, base, amplitude, kappa):
    '''
     Apparently, numpy is VERY slow at taking the exponent of a negative number
     https://github.com/numpy/numpy/issues/8233
     The correct way to avoid this problem is to install the Intel Python Packages:
     https://software.intel.com/en-us/distribution-for-python

     "old" version of sdexp. kappa/amp/base dims = (n x 2) - (:, 0) for g and (:, 1) for d
     '''
    sg = g @ s
    sd = d @ s
    #sg = base[:, [0]] + amplitude[:, [0]] * np.exp(-np.exp(np.array(-np.exp(kappa[:, [0]])) * sg))
    #sd = base[:, [1]] + amplitude[:, [1]] * np.exp(-np.exp(np.array(-np.exp(kappa[:, [1]])) * sd))
    sg = base[[0], :] + amplitude[[0], :] * np.exp(-np.exp(np.array(-np.exp(kappa[[0], :])) * sg))
    sd = base[[1], :] + amplitude[[1], :] * np.exp(-np.exp(np.array(-np.exp(kappa[[1], :])) * sd))

    return sg * x + sd


def state_dexp(rec, i, o, s, g=None, d=None, base=None, amplitude=None, kappa=None,
                                    base_g=None, amplitude_g=None, kappa_g=None, offset_g=None,
                                    base_d=None, amplitude_d=None, kappa_d=None, offset_d=None):
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

    if (base_d is None) & (amplitude_d is None) & (kappa_d is None):
        fn = lambda x : _state_dexp_old(x, rec[s]._data, g, d, base, amplitude, kappa)
    else:
        fn = lambda x : _state_dexp(x, rec[s]._data, base_g, amplitude_g, kappa_g, offset_g,
                                 base_d, amplitude_d, kappa_d, offset_d)
    
    # kludgy backwards compatibility
    try:
        pred, gain, dc = rec[i].transform(fn, o)
        gain = pred._modified_copy(gain)
        gain.name = 'gain'
        dc = pred._modified_copy(dc)
        dc.name = 'dc'
        return [pred, gain, dc]
    except:
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


def add_lv(rec, i, o, n, cutoff, e):
    """
    Compute latent variable and add to state signals:
        projection of residual responses (resp minus current pred)
        onto encoding weights (e). Add a channel  of all 1's to the
        lv signal. This will be for offset in state models.

    i: signal to subtract from resp (pred or psth)
    o: 'lv'
    e: encoding weights
    shuffle: bool (should you shuffle LV or not)
    """
    newrec = rec.copy()

    # input can be pred, or psth.
    # If psth, subtract psth (use residual signal)
    # if pred, subtract pred to create residual
    # Any signal that you wish
    # to project down to your LV

    res = newrec['resp'].rasterize()._data - newrec[i].rasterize()._data

    if cutoff is not None:
        # highpass filter residuals
        res = preproc.bandpass_filter_resp(newrec, low_c=cutoff, high_c=None, data=res)

    lv = e.T @ res

    lv = np.concatenate((np.ones((1, lv.shape[-1])), lv), axis=0)

    # z-score lv? (to avoid pred blowing up)
    #lv = lv - lv.mean(axis=-1, keepdims=True)
    #if ~np.any(lv.std(axis=-1) == 0):
    #    lv = lv / lv.std(axis=-1, keepdims=True)

    lv_sig = newrec['resp'].rasterize()._modified_copy(lv)
    lv_sig.name = o
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


def _population_mod(x, r, s, g, d, gs, ds):

    if g is not None:
        #import pdb; pdb.set_trace()
        _rat = (r-x)/(x+(x==0))
        _rat[_rat>1]=1
        _rat[_rat<-0.5]=-0.5

        _g = g.copy()
        np.fill_diagonal(_g, 0)
        gd = _g.T @ _rat

        if gs is not None:
            y = x * (gs@s) * np.exp(gd)
        else:
            y = x * np.exp(gd)
    else:
        y = x.copy()

    if d is not None:
        _diff = r-x
        _d = d.copy()
        do = np.diag(_d)
        np.fill_diagonal(_d, 0)
        dd = _d.T @ _diff
        if ds is not None:
            y += (ds@s) * dd
        else:
            y += dd

    """
    sg = g @ s
    sd = d @ s
    sg = base[0] + amplitude[0] * np.exp(-np.exp(np.array(-np.exp(kappa[0])) * sg))
    sd = base[1] + amplitude[1] * np.exp(-np.exp(np.array(-np.exp(kappa[1])) * sd))
    """
    return y


def population_mod(rec, i, o, s=None, g=None, d=None, gs=None, ds=None):
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

    r = 'resp'
    fn = lambda x : _population_mod(x, rec[r]._data, rec[s]._data, g, d, gs, ds)

    return [rec[i].transform(fn, o)]
