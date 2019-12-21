import copy

from nems.initializers import from_keywords
from nems.utils import find_module
import nems.xform_helper as xhelp
from nems.analysis.fit_basic import fit_basic
import nems.xforms as xforms

import matplotlib.pyplot as plt
import numpy as np


# Default modelnames for fitting to sims, rank2
stp = 'ozgf.fs100.ch18-ld-sev_dlog.f-wc.18x2.g-stp.2-fir.2x15-lvl.1-dexp.1_init.t3-basic.t5'
LN = 'ozgf.fs100.ch18-ld-sev_dlog.f-wc.18x2.g-fir.2x15-lvl.1-dexp.1_init.t3-basic.t5'
gc = 'ozgf.fs100.ch18-ld-contrast.ms30.cont.n.off0-sev_dlog.f-wc.18x2.g-fir.2x15-lvl.1-ctk3-dsig.d_gc2.PF.pt3.t5'
combined = 'ozgf.fs100.ch18-ld-contrast.ms30.cont.n.off0-sev_dlog.f-wc.18x2.g-stp.3-fir.2x15-lvl.1-ctk3-dsig.d_gc2.PF.pt3.t5'

_DEFAULT_CTX = None
_DEFAULT_CELL = 'TAR009d-42-1'
_DEFAULT_BATCH = 289
_DEFAULT_MODEL = 'ozgf.fs100.ch18-ld-contrast.ms30.cont.n.off0-sev_dlog.f-wc.18x3.g-fir.3x15-lvl.1-ctk3-dsig.d_gc2.PF'

def get_default_ctx():
    global _DEFAULT_CTX
    if _DEFAULT_CTX is None:
        _set_default_ctx()
    return copy.deepcopy(_DEFAULT_CTX)

def _set_default_ctx():
    xfspec, ctx = xhelp.load_model_xform(_DEFAULT_CELL, batch=_DEFAULT_BATCH,
                                         modelname=_DEFAULT_MODEL,
                                         eval_model=True)
    global _DEFAULT_CTX
    _DEFAULT_CTX = ctx


def build_toy_LN_cell():
    modelspec = from_keywords(LN.split('_')[1])
    return _set_LN_phi(modelspec)

def _set_LN_phi(modelspec):
    wc_idx2 = find_module('weight_channels', modelspec)
    fir_idx2 = find_module('fir', modelspec)
    modelspec[wc_idx2]['phi'] = {
            'mean': np.array([0.4, 0.5]),
            'sd': np.array([0.15, 0.15])
            }
    modelspec[fir_idx2]['phi'] = {
            'coefficients': np.array([
                    [0, -.125, -.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, .275, .15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ])
            }
    return _set_nonlinearity(modelspec)


def _set_nonlinearity(modelspec):
#    ctx = get_default_ctx()
#    est = ctx['est']
#    val = ctx['val']
#    eresp = est['resp'].as_continuous()
#    vresp = val['resp'].as_continuous()
#    min_resp = min(eresp.min(), vresp.min())
#    max_resp = max(eresp.max(), vresp.max())
#    epred = est['pred'].as_continuous()
#    vpred = val['pred'].as_continuous()
#    predrange = 2/(max(epred.max() - epred.min(),
#                       vpred.max() - vpred.min()) + 1)

    #base = np.array([min_resp])
    #amplitude = np.array([max_resp*0.5])
    #shift = np.array([0.5*(epred.mean() + vpred.mean())])
    #kappa = np.array([np.log(predrange)])
    base = np.array([0])
    amplitude = np.array([2])
    shift = np.array([0.275])
    kappa = np.array([2.5])

    dexp_idx = find_module('dexp', modelspec)
    if dexp_idx is None:
        # no dexp, assume dsig for gc instead
        dexp_idx = find_module('dynamic_sigmoid', modelspec)
    modelspec[dexp_idx]['phi'].update({
        'base': base, 'amplitude': amplitude,
        'shift': shift, 'kappa': kappa
        })

    return modelspec


def build_toy_gc_cell(base, amplitude, shift, kappa):
    modelspec = from_keywords(gc.split('_')[1])
    modelspec = _set_LN_phi(modelspec)
    return _set_gc_phi(modelspec, base, amplitude, shift, kappa)

def _set_gc_phi(modelspec, base, amplitude, shift, kappa):
    '''
    Parameters given as differences, e.g. kappa = -0.5 means set
    kappa_mod to be 0.5 less than kappa.

    '''
    dsig_idx = find_module('dynamic_sigmoid', modelspec)
    p = modelspec[dsig_idx]['phi']
    b = p['base']
    a = p['amplitude']
    s = p['shift']
    k = p['kappa']
    modelspec[dsig_idx]['phi'].update({
            'base_mod': b + base, 'amplitude_mod': a + amplitude,
            'shift_mod': s + shift, 'kappa_mod': k + kappa
            })
    return modelspec


def build_toy_stp_cell(u, tau):
    if not isinstance(u, np.ndarray):
        u = np.array(u)
    if not isinstance(tau, np.ndarray):
        tau = np.array(tau)
    modelspec = from_keywords(stp.split('_')[1])
    modelspec = _set_LN_phi(modelspec)
    stp_idx = find_module('stp', modelspec)
    modelspec[stp_idx]['phi'] = {'u': u, 'tau': tau}

    return modelspec


def build_toy_combined_cell(base, amplitude, shift, kappa, u, tau):
    modelspec = from_keywords(combined.split('_')[1])
    modelspec = _set_LN_phi(modelspec)
    stp_idx = find_module('stp', modelspec)
    modelspec[stp_idx]['phi'] = {'u': u, 'tau': tau}

    return _set_gc_phi(modelspec, base, amplitude, shift, kappa)


def fit_to_simulation(fit_model, simulation_spec):
    '''
    Parameters:
    -----------
    fit_model : str
        Modelname to fit to the simulation.
    simulation_spec : NEMS ModelSpec
        Modelspec to base simulation on.

    Returns:
    --------
    ctx : dict
        Xforms context. See nems.xforms.

    '''
    rec = get_default_ctx()['rec']
    ctk_idx = find_module('contrast_kernel', simulation_spec)
    if ctk_idx is not None:
        simulation_spec[ctk_idx]['fn_kwargs']['evaluate_contrast'] = True
    new_resp = simulation_spec.evaluate(rec)['pred']
    rec['resp'] = new_resp

    # replace ozgf and ld with ldm
    modelname = '-'.join(fit_model.split('-')[2:])
    xfspec = xhelp.generate_xforms_spec(modelname=modelname)
    ctx, _ = xforms.evaluate(xfspec, context={'rec': rec})

    return ctx


def compare_sim_fits(simulation_spec):
    stp_ctx = fit_to_simulation(stp, simulation_spec)
    gc_ctx = fit_to_simulation(gc, simulation_spec)
    LN_ctx = fit_to_simulation(LN, simulation_spec)

    simulation = stp_ctx['val']['resp'].as_continuous().flatten()
    stp_pred = stp_ctx['val']['pred'].as_continuous().flatten()
    gc_pred = gc_ctx['val']['pred'].as_continuous().flatten()
    LN_pred = LN_ctx['val']['pred'].as_continuous().flatten()

    return simulation, stp_pred, gc_pred, LN_pred


#def compare_to_sim(simulation, stp_pred, gc_pred, LN_pred,
#                   start=None, stop=None):
#    fig = plt.figure()
#    plt.plot(simulation)
#    plt.plot(stp_pred)
#    plt.plot(gc_pred)
#    plt.plot(LN_pred)
#    plt.xlim(start, stop)
#    plt.legend(["sim", "stp", "gc", "LN"])
#
#    return fig


def compare_sims(start=None, stop=None):
    # TODO: set up to compare on synthetic stimuli
    xfspec, ctx = xhelp.load_model_xform(_DEFAULT_CELL, _DEFAULT_BATCH,
                                         _DEFAULT_MODEL)
    val = ctx['val']
    gc_sim = build_toy_gc_cell(0, 0, 0, -0.5) #base, amp, shift, kappa
    gc_sim[-2]['fn_kwargs']['compute_contrast'] = True
    stp_sim = build_toy_stp_cell([0, 0.1], [0.08, 0.08]) #u, tau
    LN_sim = build_toy_LN_cell()

    stim = val['stim'].as_continuous()
    #resp = val['resp'].as_continuous().flatten()
    gc_val = gc_sim.evaluate(val)
    gc_sim.recording = gc_val
    gc_psth = gc_val['pred'].as_continuous().flatten()
    stp_val = stp_sim.evaluate(val)
    stp_sim.recording = stp_val
    stp_psth = stp_val['pred'].as_continuous().flatten()
    LN_val = LN_sim.evaluate(val)
    LN_sim.recording = LN_val
    LN_psth = LN_val['pred'].as_continuous().flatten()

    fig = plt.figure()
    plt.imshow(stim, aspect='auto', cmap='Greys',
               origin='lower', extent=(0, 1650, 2.0, 3.0))
    #plt.plot(resp, alpha=0.4, color='gray')
    plt.plot(LN_psth)
    plt.plot(gc_psth)
    plt.plot(stp_psth)
    plt.xlim(start,stop)
    plt.legend(['LN','GC','STP'])

    return fig
