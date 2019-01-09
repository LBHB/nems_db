import numpy as np


def _global_gain_spec(n_targets):
    gain_mean = np.zeros(n_targets)
    gain_sd = np.ones(n_targets)
    template = {
        'fn': 'nems_lbhb.rdt.modules.global_gain',
        'fn_kwargs': {},
        'prior': {
            'gain': ('Normal', {'mean': gain_mean, 'sd': gain_sd}),
        }
    }
    return template


def _relative_gain_spec(n_targets):
    gain_mean = np.zeros(n_targets)
    gain_sd = np.ones(n_targets)
    template = {
        'fn': 'nems_lbhb.rdt.modules.relative_gain',
        'fn_kwargs': {},
        'plot_fns': ['nems.plots.api.spectrogram_from_epoch',
                     'nems.plots.api.pred_resp'],
        'plot_fn_idx': 0,
        'prior': {
            'fg_gain': ('Normal', {'mean': gain_mean, 'sd': gain_sd}),
            'bg_gain': ('Normal', {'mean': gain_mean, 'sd': gain_sd}),
            'single_gain': ('Normal', {'mean': gain_mean, 'sd': gain_sd}),
        }
    }
    return template


def _relative_gain_spec_generic(n_targets):
    gain_mean = np.zeros(n_targets)
    gain_sd = np.ones(n_targets)
    template = {
        'fn': 'nems_lbhb.rdt.modules.rdt_gain',
        'fn_kwargs': {},
        'plot_fns': ['nems.plots.api.pred_resp'],
        'plot_fn_idx': 0,
        'prior': {
            'fg_gain': ('Normal', {'mean': gain_mean, 'sd': gain_sd}),
            'bg_gain': ('Normal', {'mean': gain_mean, 'sd': gain_sd}),
            'single_gain': ('Normal', {'mean': gain_mean, 'sd': gain_sd}),
        }
    }
    return template


def rdtgain(kw):
    _, mode, n_targets = kw.split('.')
    n_targets = int(n_targets)
    if mode == 'global':
        return _global_gain_spec(n_targets)
    elif mode == 'relative':
        return _relative_gain_spec(n_targets)
    elif mode == 'gen':
        return _relative_gain_spec_generic(n_targets)
    else:
        raise ValueError("Unknown mode %s", mode)

def rdtmerge(kw):
    ops = kw.split('.')
    chans = 1
    i = 'fg+bg' # default

    for op in ops:
        if op == 'stim':
            i = 'fg+bg'
        elif op == 'resp':
            i = 'fg_pred+bg_pred'

    template = {
        'fn': 'nems_lbhb.rdt.modules.apply_gain',
        'fn_kwargs': {'i': i, 'o': 'pred'},
        'plot_fns': ['nems.plots.api.mod_output',
                     'nems.plots.api.pred_spectrogram'],
        'plot_fn_idx': 1,
        'prior': {'offset': ('Normal', {
                'mean': np.zeros((chans, 1)),
                'sd': np.ones((chans, 1))*2})}
    }
    return template


def rdtwc(kw):
    from nems.plugins import default_keywords
    kw = kw[3:]
    ms = default_keywords.wc(kw)
    del ms['fn_kwargs']['i']
    del ms['fn_kwargs']['o']
    ms['fn'] = 'nems_lbhb.rdt.weight_channels.gaussian'
    return ms


def rdtfir(kw):
    from nems.plugins import default_keywords
    kw = kw[3:]
    ms = default_keywords.fir(kw)
    del ms['fn_kwargs']['i']
    del ms['fn_kwargs']['o']
    ms['fn'] = 'nems_lbhb.rdt.fir.basic'
    return ms
