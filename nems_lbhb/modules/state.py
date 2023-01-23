"""
modules/state.py

functions for applying state-related transformations

sdexp2 keyword declared here using NemsModule class, can replace sdexp
"""

import re
import logging
import numpy as np

import nems_lbhb.preprocessing as preproc
from nems0.modules import NemsModule
from nems0.registry import xmodule

log = logging.getLogger(__name__)


def _state_dexp(x, s, base_g, amplitude_g, kappa_g, offset_g, base_d, amplitude_d, kappa_d, offset_d, per_channel=False):
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
    if per_channel:
        for i in range(n_states):
            _sg = base_g[[0],i,_n] + amplitude_g[[0],i,_n] * \
                         np.exp(-np.exp(-np.exp(kappa_g[[0],i,_n]) * (s[i] - offset_g[[0],i,_n])))
            _sd = base_d[[0],i,_n] + amplitude_d[[0],i,_n] * \
                         np.exp(-np.exp(-np.exp(kappa_d[[0],i,_n]) * (s[i] - offset_d[[0],i,_n])))
            if i == 0:
               sg = _sg
               sd = _sd
            else:
               sg = np.concatenate((sg, _sg), axis=0)
               sd = np.concatenate((sd, _sd), axis=0)
                
        #import pdb; pdb.set_trace()
    else:
        for i in range(n_inputs):
            _sg = np.sum(base_g[i,:,_n] + amplitude_g[i,:,_n] * 
                         np.exp(-np.exp(-np.exp(kappa_g[i,:,_n]) * (s[:n_states] - offset_g[i,:,_n]))),
                         axis=0, keepdims=True)
            _sd = np.sum(base_d[i,:,_n] + amplitude_d[i,:,_n] * 
                         np.exp(-np.exp(-np.exp(kappa_d[i,:,_n]) * (s[:n_states] - offset_d[i,:,_n]))),
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

def _stmod_dexp(x, chans, base, amplitude, kappa):
    '''
    Modulate state signal. Only model channels in "chans".
    '''
    xnew = x.copy()
    for chan in chans:
        xnew[chan, :] = base[[chan], :] + amplitude[[chan], :] * np.exp(-np.exp(np.array(-np.exp(kappa[[chan], :])) * x[[chan], :]))
    return xnew


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


class population_mod(NemsModule):
    """
    Scale each input by weighted sum of (true) responses on other channels, projected to separate latent variables,
    each scaled by state

    """
    def __init__(self, **options):
        """
        set options to defaults if not supplied. pass to super() to add to data_dict
        """
        options['fn'] = options.get('fn', str(self.__module__) + '.population_mod')
        options['tf_layer'] = options.get('tf_layer', 'nems_lbhb.tf.layers.population_mod')
        options['fn_kwargs'] = options.get('fn_kwargs', {'i': 'pred', 'o': 'pred', 's': 'state', 'r': 'resp',
                                                         'n_inputs': 1, 'n_state_vars': 1, 'state_type': 'dc'
                                                         })
        options['plot_fns'] = options.get('plot_fns',
                                          ['nems0.plots.api.mod_output',
                                           'nems0.plots.api.before_and_after',
                                           'nems0.plots.api.pred_resp',
                                           'nems0.plots.api.state_vars_timeseries',
                                           'nems0.plots.api.state_vars_psth_all'])
        options['plot_fn_idx'] = options.get('plot_fn_idx', 3)
        options['bounds'] = options.get('bounds', {})
        super().__init__(**options)

    def description(self):
        """
        String description (include phi values?)
        """
        return "Pass state variable(s) through sigmoid then apply dc/gain to pred"

    @xmodule('pmodxx')
    def keyword(kw):
        '''
        Generate and register modulespec for the state_dexp

        Parameters
        ----------
        kw : str
            Expected format: r'^pmod\.?(\d{1,})x(\d{1,})$'
            e.g., "pmod.SxR" or "pmod.S":
                S : number of state channels (required)
                R : number of channels to modulate, ie, number of neurons in pop model (default = 1)
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
            n_state_vars = int(parsed.group(1))
            if len(parsed.groups()) > 1:
                n_inputs = int(parsed.group(2))
            else:
                n_inputs = 1

        except TypeError:
            raise ValueError("Got TypeError when parsing pmod keyword.\n"
                             "Make sure keyword is of the form: \n"
                             "pmod.{n_state_variables} \n"
                             "keyword given: %s" % kw)

        state = 'state'
        resp = 'resp'
        set_bounds = False
        for o in options[2:]:
            if o == 'lv':
                state = 'lv'
            if o == 'bound':
                set_bounds = True
        state_type = 'dc'
        # init gain params
        z = np.zeros([n_state_vars, n_inputs, n_inputs])

        template = {
            'fn_kwargs': {'i': 'pred',
                          'o': 'pred',
                          's': state,
                          'r': resp,
                          'n_inputs': n_inputs,
                          'n_state_vars': n_state_vars,
                          'state_type': state_type},
            'plot_fns': ['nems0.plots.api.mod_output',
                         'nems0.plots.api.before_and_after',
                         'nems0.plots.api.pred_resp',
                         'nems0.plots.api.state_vars_timeseries',
                         'nems0.plots.api.state_vars_psth_all'],
            'plot_fn_idx': 3,
            'prior': {'coefficients': ('Normal', {'mean': z, 'sd': z+0.1})}
        }
        if set_bounds:
            pass
            #template['bounds'] = {}

        return population_mod(**template)

    def eval(self, rec, i, o, s, r, coefficients, **kw_args):
        '''

        Parameters
        ----------
        i name of input
        o name of output signal
        s name of state signal
        r name of response signal
        coefficients - weight matrix (S x R x R)

        '''
        resp = rec[r]._data
        state = rec[s]._data

        c = coefficients.copy()
        b = np.zeros((c.shape[2], c.shape[0]))
        for ii in range(c.shape[0]):
            b[:, ii] = np.diagonal(c[ii])
            np.fill_diagonal(c[ii], 0)

        lv = np.tensordot(c, resp, axes=(2, 0))
        lv = np.sum(np.expand_dims(rec[s]._data, 1) * lv, axis=0)
        lv0 = b @ state

        fn = lambda x: x + lv + lv0

        return [rec[i].transform(fn, o)]

    def tflayer(self):
        """
        layer definition for TF spec
        """
        #import tf-relevant code only here, to avoid dependency
        return []


class sdexp_new(NemsModule):
    """
    Add state-dependent modulation through exponential
    """
    def __init__(self, **options):
        """
        set options to defaults if not supplied. pass to super() to add to data_dict
        """
        options['fn'] = options.get('fn', str(self.__module__) + '.sdexp_new')
        options['tf_layer'] = options.get('tf_layer', 'nems_lbhb.tf.layers.sdexp_layer')
        options['fn_kwargs'] = options.get('fn_kwargs', {'i': 'pred', 'o': 'pred', 's': 'state',
                                                         'n_inputs': 1, 'chans': 1,
                                                         'state_type': 'both'})
        options['plot_fns'] = options.get('plot_fns',
                                          ['nems0.plots.api.mod_output',
                                           'nems0.plots.api.before_and_after',
                                           'nems0.plots.api.pred_resp',
                                           'nems0.plots.api.state_vars_timeseries',
                                           'nems0.plots.api.state_vars_psth_all'])
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
            raise ValueError("Got TypeError when parsing sdexp2 keyword.\n"
                             "Make sure keyword is of the form: \n"
                             "sdexp2.{n_state_variables} \n"
                             "keyword given: %s" % kw)

        state = 'state'
        set_bounds = False
        # nl_state_chans = 1
        nl_state_chans = n_vars
        per_channel=('per' in options[2:])
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
                          'per_channel': per_channel,
                          'state_type': 'both'},
            'plot_fns': ['nems0.plots.api.mod_output',
                         'nems0.plots.api.before_and_after',
                         'nems0.plots.api.pred_resp',
                         'nems0.plots.api.state_vars_timeseries',
                         'nems0.plots.api.state_vars_psth_all'],
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
        
        fn_kwargs = self.get('fn_kwargs')
        
        if (base_d is None) & (amplitude_d is None) & (kappa_d is None):
            fn = lambda x: _state_dexp_old(x, rec[s]._data, g, d, base, amplitude, kappa)
        else:
            fn = lambda x: _state_dexp(x, rec[s]._data, base_g, amplitude_g, kappa_g, offset_g,
                                       base_d, amplitude_d, kappa_d, offset_d, per_channel=fn_kwargs['per_channel'])

        # kludgy backwards compatibility
        try:
            p, gain, dc = fn(rec[i]._data)
            pred = rec[i]._modified_copy(p)
            pred.name = o
            gain = pred._modified_copy(gain)
            gain.name = 'gain'
            dc = pred._modified_copy(dc)
            dc.name = 'dc'
            # uncomment to save first-order pred for use by LV models.
            pred0 = rec[i]._modified_copy(p)
            pred0.name = 'pred0'
            return [pred, gain, dc, pred0]

            # uncomment to skip saving first-order pred for use by LV models.
            #return [pred, gain, dc]

        except:
            return [rec[i].transform(fn, o)]

    def tflayer(self):
        """
        layer definition for TF spec
        """
        #import tf-relevant code only here, to avoid dependency
        return []


class state_hinge(NemsModule):
    """
    Hinge function to capture non-monotonic state effects
    """
    def __init__(self, **options):
        """
        set options to defaults if not supplied. pass to super() to add to data_dict
        """
        options['fn'] = options.get('fn', str(self.__module__) + '.state_hinge')
        options['tf_layer'] = options.get('tf_layer', '')
        options['fn_kwargs'] = options.get('fn_kwargs', {'i': 'pred', 'o': 'pred', 's': 'state',
                                                         'n_inputs': 1, 'chans': 1,
                                                         'state_type': 'both'})
        options['plot_fns'] = ['nems0.plots.api.mod_output',
                    'nems0.plots.api.spectrogram_output',
                    'nems0.plots.api.before_and_after',
                    'nems0.plots.api.pred_resp',
                    'nems0.plots.api.state_vars_timeseries',
                    'nems0.plots.api.state_vars_psth_all']

        options['plot_fn_idx'] = options.get('plot_fn_idx', 5)
        options['bounds'] = options.get('bounds', {})
        super().__init__(**options)

    def description(self):
        """
        String description (include phi values?)
        """
        return "Pass state variable(s) through hinge then apply dc/gain to pred"

    @xmodule('sthinge')
    def keyword(kw):
        '''
        Generate and register modulespec for the sthinge

        Parameters
        ----------
        kw : str
            Expected format: r'^sthinge\.?(\d{1,})x(\d{1,})$'
            e.g., "sthinge.SxR" or "sthinge.S":
                S : number of state channels (required)
                R : number of channels to modulate (default = 1)
        Options
        -------
        None
        '''
        options = kw.split('.')
        in_out_pattern = re.compile(r'^(\d{1,})x(\d{1,})$')

        try:
            parsed = re.match(in_out_pattern, options[1])
            if parsed is None:
                # backward compatible parsing if R not specified
                n_vars = int(options[1])
                n_chans = 1

            else:
                n_vars = int(parsed.group(1))
                if len(parsed.groups())>1:
                    n_chans = int(parsed.group(2))
                else:
                    n_chans = 1

        except TypeError:
            raise ValueError("Got TypeError when parsing sthinge keyword.\n"
                             "Make sure keyword is of the form: \n"
                             "sthinge.{n_variables} or sthinge.{n_variables}x{n_chans} \n"
                             "keyword given: %s" % kw)
            
        zeros = np.zeros([n_chans, n_vars])
        ones = np.ones([n_chans, n_vars])
        g1_mean = zeros.copy()
        g1_mean[:, 0] = 1
        g1_sd = ones.copy() / 20
        d1_mean = zeros
        d1_sd = ones
        g2_mean = zeros.copy()
        g2_mean[:, 0] = 1
        g2_sd = ones.copy() / 20
        d2_mean = zeros
        d2_sd = ones
        x0_mean= zeros
        x0_sd = ones / 20
        
        fn_kwargs = {'i': 'pred', 'o': 'pred', 's': 'state', 'chans': n_vars, 'n_inputs': n_chans}
        
        if ('g' in options[2:]):
            fn_kwargs['state_type'] = 'gain_only'
            fn_kwargs['d1'] = d1_mean
            fn_kwargs['d2'] = d2_mean
            prior = {'g1': ('Normal', {'mean': g1_mean, 'sd': g1_sd}),
                     'g2': ('Normal', {'mean': g2_mean, 'sd': g2_sd}),
                     'x0': ('Normal', {'mean': x0_mean, 'sd': x0_sd})}
        elif ('d' in options[2:]):
            fn_kwargs['state_type'] = 'dc_only'
            fn_kwargs['g1'] = g1_mean
            fn_kwargs['g2'] = g2_mean
            prior = {'d1': ('Normal', {'mean': d1_mean, 'sd': d1_sd}),
                     'd2': ('Normal', {'mean': d2_mean, 'sd': d2_sd}),
                     'x0': ('Normal', {'mean': x0_mean, 'sd': x0_sd})}
        else:
            prior = {'d1': ('Normal', {'mean': d1_mean, 'sd': d1_sd}),
                     'd2': ('Normal', {'mean': d2_mean, 'sd': d2_sd}),
                     'g1': ('Normal', {'mean': g1_mean, 'sd': g1_sd}),
                     'g2': ('Normal', {'mean': g2_mean, 'sd': g2_sd}),
                     'x0': ('Normal', {'mean': x0_mean, 'sd': x0_sd})}

            
        if ('s' in options[2:]): # separate offset for spont than during evoked
            raise ValueError('include_spont not yet supported.')
        fn_kwargs['per_channel']=('per' in options[2:])
        
        fn_kwargs['fix_across_channels'] = 0
        if 'c1' in options[2:]:
            fn_kwargs['fix_across_channels'] = 1
        elif 'c2' in options[2:]:
            fn_kwargs['fix_across_channels'] = 2
        if 'lv' in options[2:]:
            fn_kwargs['s'] = 'lv'
        
        #If .o# is passed, fix gainoffset to #, initialize gain to 0.
        # y = (np.matmul(g, rec[s]._data) + offset) * x so .g1 will by initialize with no state-dependence
        gainoffset = 0
        bounds=None
        exclude_chans = None

        for op in options[2:]:
            if op.startswith('o'):
                num = op[1:].replace('\\', '')
                gainoffset = float(num)
                g_mean[:, 0] = 0
            elif op.startswith('b'):
                bounds_in = op[1:].replace('\\', '').replace('d', '.')
                bounds_in = bounds_in.split(':')
                bounds = tuple(np.full_like(g_mean,float(bound) - gainoffset) for bound in bounds_in)
            elif op.startswith("x"):
                fn_kwargs['exclude_chans'] = [int(x) for x in op[1:].split(',')]

        template = {
            'fn_kwargs': fn_kwargs,
            'plot_fn_idx': 5,
            'prior': prior
        }
        template['tf_layer'] = ''
        
        return state_hinge(**template)


    def eval(self, rec, i='pred', o='pred', s='state', g1=1, g2=1, x0=0, d1=0, d2=0, exclude_chans=None,
             per_channel=False, **kw_args):
        '''
        Parameters
        ----------
        i name of input
        o name of output signal
        s name of state signal
        g1,g2 - gain to scale s by
        d1,d2 - dc to offset by
        x0 -crossover between 1 and 2
        '''
        
        state = rec[s]._data
        if exclude_chans is not None:
            keepidx = [idx for idx in range(0, state.shape[0]) if idx not in exclude_chans]
            state = state[keepidx, :]

        if per_channel:
            def fn(x):
                s_ = state-x0.T
                sp = s_ * (s_>0)
                sn = -s_ * (s_<0)
                # gain applied point-wise to each state channel, split above and below x0
                return (-g1.T * sn + g2.T * sp) * x + (-d1.T * sn + d2.T * sp)

        else:
            def fn(x):
                s_ = state-x0.T
                sp = s_ * (s_>0)
                sn = -s_ * (s_<0)
                # gain to all state channels, split above and below x0
                return (np.matmul(-g1, sn) + np.matmul(g2, sp)) * x + np.matmul(-d1, sn) + np.matmul(d2, sp)
                
        return [rec[i].transform(fn, o)]

    def tflayer(self):
        """
        layer definition for TF spec
        """
        #import tf-relevant code only here, to avoid dependency
        return []

    
class lv_norm(NemsModule):
    """
    Add latent variable for normative models (matched cc)
    """
    def __init__(self, **options):
        """
        set options to defaults if not supplied. pass to super() to add to data_dict
        """
        options['fn'] = options.get('fn', str(self.__module__) + '.lv_norm')
        options['tf_layer'] = options.get('tf_layer', None)
        options['fn_kwargs'] = options.get('fn_kwargs', {'i': 'pred', 'o': 'pred', 's': 'state',
            'lv': 'lv', 'additive': False })
        options['plot_fns'] = options.get('plot_fns',
                                          ['nems0.plots.api.mod_output',
                                           'nems0.plots.api.before_and_after',
                                           'nems0.plots.api.pred_resp',
                                           'nems0.plots.api.state_vars_timeseries',
                                           'nems0.plots.api.state_vars_psth_all'])
        options['plot_fn_idx'] = options.get('plot_fn_idx', 3)
        options['bounds'] = options.get('bounds', {})
        super().__init__(**options)

    def description(self):
        """
        String description (include phi values?)
        """
        return "state-dependent LV modulation"

    @xmodule('lvnorm')
    def keyword(kw):
        '''
        Generate and register modulespec for the lv_norm module

        Parameters
        ----------
        kw : str
            Expected format: r'^lvnorm\.?(\d{1,})x(\d{1,})$'
            e.g., "lvnorm.SxR" or "lvnorm.S":
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
            n_states = int(parsed.group(1))
            if len(parsed.groups()) > 1:
                n_chans = int(parsed.group(2))
            else:
                n_chans = 1

        except TypeError:
            raise ValueError("Got TypeError when parsing lvnorm keyword.\n"
                             "Make sure keyword is of the form: \n"
                             "lvnorm.{n_state_variables}x{n_resp_chans} \n"
                             "keyword given: %s" % kw)

        state = 'state'
        set_bounds = False
        additive = False
        single_offset = False
        exclude_chans = None
        for o in options[2:]:
            if o == 'bound':
                set_bounds = True
            elif o == 'd':
                additive=True
            elif o == 'so':
                single_offset=True
            elif o.startswith('sm'):
                state = 'state_mod'
            elif o.startswith('x'):
                exclude_chans = [int(x) for x in o[1:].split(',')]

        # update number of state channels, if we're asking to exlude any
        if exclude_chans is not None:
            n_states = n_states - len(exclude_chans)

        # init gain/dc params
        mean_g = np.zeros([n_chans, n_states])
        sd_g = np.ones([n_chans, n_states])/10
        if single_offset:
            mean_d = np.zeros([1, n_states])
            sd_d = np.ones([1, n_states])/10
        else:
            mean_d = np.zeros([n_chans, n_states])
            sd_d = np.ones([n_chans, n_states])/10

        template = {
            'fn_kwargs': {'i': 'pred',
                          'o': 'pred',
                          's': state,
                          'lv': 'lv',
                          'additive': additive,
                          'n_inputs': n_chans,
                          'n_states': n_states,
                          'exclude_chans': exclude_chans},
            'plot_fns': ['nems0.plots.api.mod_output',
                         'nems0.plots.api.before_and_after',
                         'nems0.plots.api.pred_resp',
                         'nems0.plots.api.state_vars_timeseries',
                         'nems0.plots.api.state_vars_psth_all',
                         'nems0.plots.api.state_gain_parameters'],
            'plot_fn_idx': 5,
            'prior': {'g': ('Normal', {'mean': mean_g, 'sd': sd_g}),
                      'd': ('Normal', {'mean': mean_d, 'sd': sd_d})}
        }
        if set_bounds:
            template['bounds'] = {'g': (None, None),
                                  'd': (None, None)}

        return lv_norm(**template)

    def eval(self, rec, i, o, s, lv, g=None, d=None, additive=False, exclude_chans=None, **kw_args):
        '''
        Parameters
        ----------
        i name of input (baseline pred)
        o name of output signal
        s name of state signal
        lv - name of lv signal
        g - gain to scale s by (n_chan X n_state)
        d - dc to offset by
        '''

        lv = rec[lv].as_continuous()
        state = rec[s].as_continuous()
        # if excluding channels, update state now
        if exclude_chans is not None:
            keepidx = [idx for idx in range(0, state.shape[0]) if idx not in exclude_chans]
            state = state[keepidx, :]
        #import pdb; pdb.set_trace()
        def fn(x):
            x = x.copy()
            # faster(?): compute all scaling terms then apply at once (outside of loop)
            sf = np.zeros_like(x)
            for l in range(d.shape[1]):
                sf += (d[:,[l]] + g[:,[l]]*state[[l],:]) * lv[[l],:]
            
            x *= np.exp(sf)
            return x

        def fn_additive(x):
            x = x.copy()
            for l in range(d.shape[1]):
                x += (d[:,l:(l+1)] + g[:,l:(l+1)]*state[l:(l+1),:]) * lv[l:(l+1),:] 
            return x

        if additive:
            return [rec[i].transform(fn_additive, o)]
        else:
            return [rec[i].transform(fn, o)]

    def tflayer(self):
        """
        layer definition for TF spec
        """
        #import tf-relevant code only here, to avoid dependency
        return []


class indep_noise(NemsModule):
    """
    Add latent variable for normative models (matched cc)
    """
    def __init__(self, **options):
        """
        set options to defaults if not supplied. pass to super() to add to data_dict
        """
        options['fn'] = options.get('fn', str(self.__module__) + '.indep_noise')
        options['tf_layer'] = options.get('tf_layer', None)
        options['fn_kwargs'] = options.get('fn_kwargs', {'i': 'pred', 'o': 'pred', 's': 'state',
            'additive': False, 'indep': 'indep'})
        options['plot_fns'] = options.get('plot_fns',
                                          ['nems0.plots.api.mod_output',
                                           'nems0.plots.api.before_and_after',
                                           'nems0.plots.api.pred_resp',
                                           'nems0.plots.api.state_vars_timeseries',
                                           'nems0.plots.api.state_vars_psth_all'])
        options['plot_fn_idx'] = options.get('plot_fn_idx', 3)
        options['bounds'] = options.get('bounds', {})
        super().__init__(**options)

    def description(self):
        """
        String description (include phi values?)
        """
        return "Pass state variable(s) through sigmoid then apply dc/gain to pred"

    @xmodule('inoise')
    def keyword(kw):
        '''
        Generate and register modulespec for the lv_norm module

        Parameters
        ----------
        kw : str
            Expected format: r'^inoise\.?(\d{1,})x(\d{1,})$'
            e.g., "lvnorm.SxR" or "lvnorm.S":
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
            n_states = int(parsed.group(1))
            if len(parsed.groups()) > 1:
                n_chans = int(parsed.group(2))
            else:
                n_chans = 1

        except TypeError:
            raise ValueError("Got TypeError when parsing inoise keyword.\n"
                             "Make sure keyword is of the form: \n"
                             "inoise.{n_state_variables}x{n_resp_chans} \n"
                             "keyword given: %s" % kw)

        state = 'state'
        set_bounds = False
        additive = True
        exclude_chans = None
        poisson = False
        for o in options[2:]:
            if o == 'bound':
                set_bounds = True
            elif o == 'g':
                additive = False
            elif o == 'p':
                poisson = True
            elif o.startswith("sm"):
                state = "state_mod"
            elif o.startswith("x"):
                exclude_chans = [int(x) for x in o[1:].split(',')]

        # update number of state channels, if we're asking to exlude any
        if exclude_chans is not None:
            n_states = n_states - len(exclude_chans)

        # init gain/dc params
        mean_g = np.zeros([n_chans, n_states])
        sd_g = np.ones([n_chans, n_states])/10
        if poisson:
            mean_g[:,0]=0.5
        elif additive:
            mean_g[:,0]=0.5
        else:
            mean_g[:,0]=0.1

        template = {
            'fn_kwargs': {'i': 'pred',
                          'o': 'pred',
                          's': state,
                          'indep': 'indep',
                          'additive': additive,
                          'poisson': poisson,
                          'n_inputs': n_chans,
                          'n_states': n_states,
                          'exclude_chans': exclude_chans},
            'plot_fns': ['nems0.plots.api.mod_output',
                         'nems0.plots.api.before_and_after',
                         'nems0.plots.api.pred_resp',
                         'nems0.plots.api.state_vars_timeseries',
                         'nems0.plots.api.state_vars_psth_all',
                         'nems0.plots.api.state_gain_parameters'],
            'plot_fn_idx': 5,
            'prior': {'g': ('Normal', {'mean': mean_g, 'sd': sd_g})}
        }
        if set_bounds:
            template['bounds'] = {'g': (None, None)}

        return indep_noise(**template)

    def eval(self, rec, i, o, s, indep, g=None, additive=True, exclude_chans=None, poisson=False, **kw_args):
        '''
        Parameters
        ----------
        i name of input (baseline pred)
        o name of output signal
        s name of state signal
        indep - name of indep noise signal
        g - gain applied to state-mod indep noise for each unit
        additive - boolean: if True noise is additive, False multiplicative
        '''

        indep_noise = rec[indep].as_continuous()
        state = rec[s].as_continuous()
        # if excluding channels, update state now
        if exclude_chans is not None:
            keepidx = [idx for idx in range(0, state.shape[0]) if idx not in exclude_chans]
            state = state[keepidx, :]
        def fn_multiplicative(x):
            x = x * np.exp((g @ state[:g.shape[1],:]) * indep_noise)
            if poisson:
                x[x<0]=0
                rng = np.random.default_rng(2021)
                x = rng.poisson(x).astype(float) * 0.1 + x * 0.9
            return x

        def fn_additive(x):
            x = x + (g @ state[:g.shape[1],:]) * indep_noise
            if poisson:
                x[x<0]=0
                rng = np.random.default_rng(2021)
                x = rng.poisson(x).astype(float) * 0.1 + x * 0.9
            return x
        
        if additive:
            return [rec[i].transform(fn_additive, o)]
        else:
            return [rec[i].transform(fn_multiplicative, o)]


    def tflayer(self):
        """
        layer definition for TF spec
        """
        #import tf-relevant code only here, to avoid dependency
        return []


class save_prediction(NemsModule):
    """
    Save pred signal at this stage of model fit and add to recording.
    Index based on the module? e.g. saving after first module is
    pred0, after second is pred1. How to do that?
    """
    def __init__(self, **options):
        """
        set options to defaults if not supplied. pass to super() to add to data_dict
        """
        options['fn'] = options.get('fn', str(self.__module__) + '.save_prediction')
        options['tf_layer'] = options.get('tf_layer', None)
        options['fn_kwargs'] = options.get('fn_kwargs', {'i': 'pred', 'o': 'pred0'})
        options['bounds'] = options.get('bounds', {})
        super().__init__(**options)

    def description(self):
        """
        String description (include phi values?)
        """
        return "Save pred signal at the current stage of model fitting to be used later by another module e.g. fit_ccnorm uses pred0"

    @xmodule('spred')
    def keyword(kw):
        '''
        Generate and register modulespec for the save prediction module

        Parameters
        ----------
        kw : str
            Expected format: 'spred'
        Options
        -------
        None
        '''
        # note, dummy phi prior because otherwise will crash. This is pretty hacky
        template = {
            'fn_kwargs': {'i': 'pred',
                          'o': 'pred0'
                          },
            'plot_fns': [],
            'prior': {'d': ('Normal', {'mean': np.zeros(1), 'sd': np.ones(1)})}
        }
        return save_prediction(**template)

    def eval(self, rec, i, o, **kw_args):
        '''
        Parameters
        ----------
        i name of input (baseline pred)
        o name of output signal
        '''

        def fn_dummy(x):
            return x
        
        return [rec[i].transform(fn_dummy, o)]

    def tflayer(self):
        """
        layer definition for TF spec
        """
        #import tf-relevant code only here, to avoid dependency
        return []


class state_mod(NemsModule):
    """
    input / output: state -- modify state channels
    Specify nonlinearity/not (sdexp) for each channel.
    """
    def __init__(self, **options):
        """
        set options to defaults if not supplied. pass to super() to add to data_dict
        """
        options['fn'] = options.get('fn', str(self.__module__) + '.state_mod')
        options['tf_layer'] = options.get('tf_layer', None)
        options['fn_kwargs'] = options.get('fn_kwargs', {'i': 'state', 
                                                         'o': 'state_mod', 
                                                         'chans': []
                                                         }
                                                         )
        options['plot_fns'] = options.get('plot_fns', [])
        options['bounds'] = options.get('bounds', {})
        super().__init__(**options)

    def description(self):
        """
        String description (include phi values?)
        """
        return "Pass state variable(s) through sigmoid. Return new state_mod signal"

    @xmodule('stmod')
    def keyword(kw):
        '''
        Generate and register modulespec for the stmod module

        Parameters
        ----------
        kw : str
            Expected format: 
                e.g. stmod.0:2 (modulate state channels 0 through 2 - inclusive)
                e.g. stmod.0,1 (modulate only channels 0 and 1)
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

        options = kw.split('.')
        modchans = None 
        for option in options[1:]:
            if ("x" not in option):
                if "," in option:
                    modchans = [int(k)+1 for k in option.split(",")]
                else:
                    modchans = [int(option)+1]

        if modchans == None:
            raise ValueError("Must specify which")

        n_chans = n_vars #len(modchans)
        # init gain params
        zeros = np.zeros([n_chans, 1])
        ones = np.ones([n_chans, 1])
        base_mean = zeros.copy()
        base_sd = ones.copy()
        amp_mean = zeros.copy() + 0
        amp_sd = ones.copy() * 0.1
        amp_mean[:, 0] = 1 
        kappa_mean = zeros.copy()
        kappa_sd = ones.copy() * 0.1

        template = {
            'fn_kwargs': {'i': 'state',
                          'o': 'state_mod',
                          'chans': modchans},
            'plot_fns': [],
            'prior': {'base': ('Normal', {'mean': base_mean, 'sd': base_sd}),
                      'amplitude': ('Normal', {'mean': amp_mean, 'sd': amp_sd}),
                      'kappa': ('Normal', {'mean': kappa_mean, 'sd': kappa_sd})}
        }

        return state_mod(**template)

    def eval(self, rec, i, o, chans, base=None, amplitude=None, kappa=None, **kw_args):
        '''
        Parameters
        ----------
        i name of input
        o name of output signal
        chans which channels to modulate
        base, amplitude, kappa, offset - parameters for dexp applied to each state channel
        '''
        fn = lambda x: _stmod_dexp(x, chans, base, amplitude, kappa)
        return [rec[i].transform(fn, o)]


    def tflayer(self):
        """
        layer definition for TF spec
        """
        #import tf-relevant code only here, to avoid dependency
        return []