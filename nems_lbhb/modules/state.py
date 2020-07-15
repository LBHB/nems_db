"""
modules/state.py

functions for applying state-related transformations
"""

import re
import logging
import numpy as np

import nems_lbhb.preprocessing as preproc
from nems.modules import NemsModule
from nems.registry import xmodule

log = logging.getLogger(__name__)


def _state_dexp(x, s, base_g, amplitude_g, kappa_g, offset_g, base_d, amplitude_d, kappa_d, offset_d):
    '''
     Apparently, numpy is VERY slow at taking the exponent of a negative number
     https://github.com/numpy/numpy/issues/8233
     The correct way to avoid this problem is to install the Intel Python Packages:
     https://software.intel.com/en-us/distribution-for-python

     "current" version of sdexp. separate kappa/amp/base phis for gain/dc.
     So all parameters (g, d, base_g, etc.) are the same shape.

     parameters are pred_inputs X state_inputs
        _sg = _bg + _ag * tf.exp(-tf.exp(-tf.exp(_kg * (tf.expand_dims(s,3) - _og))))
    '''

    n_inputs=base_g.shape[0]
    n_states=base_g.shape[1]
    _n = np.newaxis
    for i in range(n_inputs):
        _sg = np.sum(base_g[i,:,_n] + amplitude_g[i,:,_n] * 
                     np.exp(-np.exp(-np.exp(kappa_g[i,:,_n]) * (s - offset_g[i,:,_n]))), 
                     axis=0, keepdims=True)
        _sd = np.sum(base_d[i,:,_n] + amplitude_d[i,:,_n] * 
                     np.exp(-np.exp(-np.exp(kappa_d[i,:,_n]) * (s - offset_d[i,:,_n]))), 
                     axis=0, keepdims=True)
        if i == 0:
           sg = _sg
           sd = _sd
        else:
           sg = np.concatenate((sg, _sg), axis=0)
           sd = np.concatenate((sd, _sd), axis=0)

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
        pred, gain, dc = fn(rec[i]._data)
        pred = rec[i]._modified_copy(pred)
        pred.name = o
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


class sdexp_new(NemsModule):
    """
    Add a constant to a NEMS signal
    """
    def __init__(self, **options):
        """
        set options to defaults if not supplied. pass to super() to add to data_dict
        """
        options['fn'] = options.get('fn', str(self.__module__) + '.sdexp_new')
        options['tf_layer'] = options.get('nems_lbhb.tf.layers.sdexp_layer')
        options['fn_kwargs'] = options.get('fn_kwargs', {'i': 'pred', 'o': 'pred', 's': 'state',
                                                         'n_inputs': 1, 'chans': 1,
                                                         'state_type': 'both'})
        options['plot_fns'] = options.get('plot_fns',
                                          ['nems.plots.api.mod_output',
                                           'nems.plots.api.before_and_after',
                                           'nems.plots.api.pred_resp',
                                           'nems.plots.api.state_vars_timeseries',
                                           'nems.plots.api.state_vars_psth_all'])
        options['plot_fn_idx'] = options.get('plot_fn_idx', 3)
        options['bounds'] = options.get('bounds', {})
        super().__init__(**options)

    def description(self):
        """
        String description (include phi values?)
        """
        return "Pass state variable(s) through sigmoid then apply dc/gain to pred"

    @xmodule('sdexp2')
    def keyword(kw):
        '''
        Generate and register modulespec for the state_dexp

        Parameters
        ----------
        kw : str
            Expected format: r'^sdexp\.?(\d{1,})x(\d{1,})$'
            e.g., "sdexp.SxR" or "sdexp.S":
                S : number of state channels (required)
                R : number of channels to modulate (default = 1)
        Options
        -------
        None
        '''
        options = kw.split('.')
        pattern = re.compile(r'^(\d{1,})x(\d{1,})$')
        parsed = re.match(pattern, options[1])
        if parsed is None:
            # backward compatible parsing if R not specified
            pattern = re.compile(r'^(\d{1,})$')
            parsed = re.match(pattern, options[1])
        try:
            n_vars = int(parsed.group(1))
            if len(parsed.groups()) > 1:
                n_chans = int(parsed.group(2))
            else:
                n_chans = 1

        except TypeError:
            raise ValueError("Got TypeError when parsing stategain keyword.\n"
                             "Make sure keyword is of the form: \n"
                             "sdexp.{n_state_variables} \n"
                             "keyword given: %s" % kw)

        state = 'state'
        set_bounds = False
        # nl_state_chans = 1
        nl_state_chans = n_vars
        for o in options[2:]:
            if o == 'lv':
                state = 'lv'
            if o == 'bound':
                set_bounds = True
            # if o == 'snl':
            # state-specific non linearities (snl)
            # only reason this is an option is to allow comparison with old models
            # nl_state_chans = n_vars

        # init gain params
        zeros = np.zeros([n_chans, nl_state_chans])
        ones = np.ones([n_chans, nl_state_chans])
        base_mean_g = zeros.copy()
        base_sd_g = ones.copy()
        amp_mean_g = zeros.copy() + 0
        amp_sd_g = ones.copy() * 0.1
        amp_mean_g[:, 0] = 1  # (1 / np.exp(-np.exp(-np.exp(0)))) # so that gain = 1 for baseline chan
        kappa_mean_g = zeros.copy()
        kappa_sd_g = ones.copy() * 0.1
        offset_mean_g = zeros.copy()
        offset_sd_g = ones.copy() * 0.1

        # init dc params
        base_mean_d = zeros.copy()
        base_sd_d = ones.copy()
        amp_mean_d = zeros.copy() + 0
        amp_sd_d = ones.copy() * 0.1
        kappa_mean_d = zeros.copy()
        kappa_sd_d = ones.copy() * 0.1
        offset_mean_d = zeros.copy()
        offset_sd_d = ones.copy() * 0.1

        template = {
            'fn_kwargs': {'i': 'pred',
                          'o': 'pred',
                          's': state,
                          'n_inputs': n_chans,
                          'chans': n_vars,
                          'state_type': 'both'},
            'plot_fns': ['nems.plots.api.mod_output',
                         'nems.plots.api.before_and_after',
                         'nems.plots.api.pred_resp',
                         'nems.plots.api.state_vars_timeseries',
                         'nems.plots.api.state_vars_psth_all'],
            'plot_fn_idx': 3,
            'prior': {'base_g': ('Normal', {'mean': base_mean_g, 'sd': base_sd_g}),
                      'amplitude_g': ('Normal', {'mean': amp_mean_g, 'sd': amp_sd_g}),
                      'kappa_g': ('Normal', {'mean': kappa_mean_g, 'sd': kappa_sd_g}),
                      'offset_g': ('Normal', {'mean': offset_mean_g, 'sd': offset_sd_g}),
                      'base_d': ('Normal', {'mean': base_mean_d, 'sd': base_sd_d}),
                      'amplitude_d': ('Normal', {'mean': amp_mean_d, 'sd': amp_sd_d}),
                      'kappa_d': ('Normal', {'mean': kappa_mean_d, 'sd': kappa_sd_d}),
                      'offset_d': ('Normal', {'mean': offset_mean_d, 'sd': offset_sd_d})}
        }
        if set_bounds:
            template['bounds'] = {'base_g': (0, 10),
                                  'amplitude_g': (0, 10),
                                  'kappa_g': (None, None),
                                  'offset_g': (None, None),
                                  'base_d': (-10, 10),
                                  'amplitude_d': (-10, 10),
                                  'kappa_d': (None, None),
                                  'offset_d': (None, None)}

        return sdexp_new(**template)

    def eval(self, rec, i, o, s, g=None, d=None, base=None, amplitude=None, kappa=None,
             base_g=None, amplitude_g=None, kappa_g=None, offset_g=None,
             base_d=None, amplitude_d=None, kappa_d=None, offset_d=None, **kw_args):
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
            fn = lambda x: _state_dexp_old(x, rec[s]._data, g, d, base, amplitude, kappa)
        else:
            fn = lambda x: _state_dexp(x, rec[s]._data, base_g, amplitude_g, kappa_g, offset_g,
                                       base_d, amplitude_d, kappa_d, offset_d)

        # kludgy backwards compatibility
        try:
            pred, gain, dc = fn(rec[i]._data)
            pred = rec[i]._modified_copy(pred)
            pred.name = o
            gain = pred._modified_copy(gain)
            gain.name = 'gain'
            dc = pred._modified_copy(dc)
            dc.name = 'dc'
            return [pred, gain, dc]
        except:
            return [rec[i].transform(fn, o)]

    def tflayer(self):
        """
        layer definition for TF spec
        """
        #import tf-relevant code only here, to avoid dependency
        return []
