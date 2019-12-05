from nems.initializers import from_keywords
from nems.utils import find_module
import nems.xform_helper as xhelp
from nems.analysis import fit_basic
import matplotlib.pyplot as plt

from nems_lbhb.gcmodel.figures.definitions import *
_, gc, stp, LN, combined = default_args


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


def fit_to_simulation(modelname, simulation_spec, rec,
                      maxiter=1000, tolerance=1e-8):
    # TODO: this should work for LN and stp, but need to do something
    # more for contrast. either load in contrast manually and use different
    # fitter, or loop through xforms system to take care of all of that

    # don't want to mess with est-val split or converting spikes
    # to rate, so remove rec loader and sev from load string
    modelspec = from_keywords(modelname.split('_')[1])
    est, val = rec.split_using_epoch_occurrence_counts('STIM')
    new_est_resp = simulate_firing_rate(simulation_spec, val)
    new_val_resp = simulate_firing_rate(simulation_spec, est)
    new_est = est.copy()
    new_val = val.copy()
    new_est['resp'] = new_est_resp
    new_val['resp'] = new_val_resp
    fitted_model = fit_basic(new_est, modelspec)
    new_pred = fitted_model.evaluate(new_val)['pred'].as_continuous().flatten()
    new_resp = new_val_resp.as_continuous().flatten()

    plt.plot([new_pred, new_resp], legend=['pred', 'sim'])

    return fitted_model, val, new_val
