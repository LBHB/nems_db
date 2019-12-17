from nems.initializers import from_keywords
from nems.utils import find_module
import nems.xform_helper as xhelp
from nems.analysis.fit_basic import fit_basic
import nems.xforms as xforms

import matplotlib.pyplot as plt
import numpy as np

from nems_lbhb.gcmodel.figures.definitions import *
_, gc, stp, LN, combined = default_args


def _phis_to_arrays(*phis):
    new_phis = []
    for phi in phis:
        if np.isscalar(phi):
            new_phis.append(np.array([[phi]]))
        else:
            new_phis.append(phi)
    return new_phis


def get_default_rec():
    xfspec, ctx = xhelp.load_model_xform('TAR010c-13-1', 289, LN,
                                         eval_model=True)
    return ctx['rec']


def build_toy_LN_cell():
    modelspec = from_keywords(LN.split('_')[1])
    # TODO: decie whether to tweak LN parameters
    return modelspec


def build_toy_gc_cell(base, amplitude, shift, kappa):
    modelspec = from_keywords(gc.split('_')[1])
    dsig_idx = find_module('dynamic_sigmoid', modelspec)
    modelspec[dsig_idx]['phi'] = {
            'base': base, 'amplitude': amplitude, 'shift': shift,
            'kappa': kappa
            }
    return modelspec


def build_toy_stp_cell(u, tau):
    u, tau = _phis_to_arrays(u, tau)
    modelspec = from_keywords(stp.split('_')[1])
    stp_idx = find_module('stp', modelspec)
    modelspec[stp_idx]['phi'] = {'u': u, 'tau': tau}
    return modelspec


def build_toy_combined_cell(base, amplitude, shift, kappa, u, tau):
    modelspec = from_keywords(combined.split('_')[1])
    dsig_idx = find_module('gc', modelspec)
    stp_idx = find_module('stp', modelspec)
    modelspec[dsig_idx]['phi'] = {
            'base': base, 'amplitude': amplitude, 'shift': shift,
            'kappa': kappa
            }
    modelspec[stp_idx]['phi'] = {'u': u, 'tau': tau}
    return modelspec


# Given a modelspec and a recording, generate stimulated response
# (as a firing rate) by assuming the model prediction to be ground truth.
def simulate_firing_rate(modelspec, rec, as_array=False):
    #stim = rec[source_signal]
    simulated_response = modelspec.evaluate(rec)['pred']
    if as_array:
        return simulated_response.as_continuous().flatten()
    else:
        return simulated_response


def fit_to_simulation(modelname, simulation_spec):
    # cell and batch are just used to get a stimulus set
    cellid = 'TAR010c-13-1'
    batch = 289
    xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname,
                                          eval_model=True)
    rec = ctx['rec']
    new_resp = simulate_firing_rate(simulation_spec, rec)
    rec['resp'] = new_resp

    # replace ozgf and ld with ldm
    modelname = '-'.join(modelname.split('-')[2:])
    xfspec = xhelp.generate_xforms_spec(modelname=modelname)
    xfspec, ctx = xforms.evaluate(xfspec, context={'rec': rec})

    simulation = ctx['val']['resp'].as_continuous().flatten()
    prediction = ctx['val']['pred'].as_continuous().flatten()
    plt.plot([prediction, simulation], legend=['pred', 'sim'])

    return ctx
