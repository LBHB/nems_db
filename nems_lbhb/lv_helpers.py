import nems
import numpy as np
import logging

log = logging.getLogger(__name__)

def fit_pupil_lv(modelspec, est, max_iter=1000, tolerance=1e-7,
              metric='nmse', IsReload=False, fitter='scipy_minimize',
              jackknifed_fit=False, random_sample_fit=False,
              n_random_samples=0, random_fit_subset=None,
              output_name='resp', **context):
    ''' 
    Copied from nems.xforms.fit_basic. Only modification is that we change
    the metric from standard nmse, as is used for fit_basic
    '''
    if IsReload:
        return {}
    metric_fn = lambda d: pup_nmse(d, 'pred', output_name, alpha=context['alpha'])
    fitter_fn = getattr(nems.fitters.api, fitter)
    fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}

    for fit_idx in range(modelspec.fit_count):
        for jack_idx, e in enumerate(est.views()):
            modelspec.jack_index = jack_idx
            modelspec.fit_index = fit_idx
            log.info("----------------------------------------------------")
            log.info("Fitting: fit %d/%d, fold %d/%d",
                     fit_idx + 1, modelspec.fit_count,
                     jack_idx + 1, modelspec.jack_count)
            modelspec = nems.analysis.api.fit_basic(
                    e, modelspec, fit_kwargs=fit_kwargs,
                    metric=metric_fn, fitter=fitter_fn)

    return {'modelspec': modelspec}


def pup_nmse(result, pred_name='pred', resp_name='resp', alpha=0):
    """
    When alpha = 0, this is just standard nmse of pred vs. resp.
    When alpha > 0, we add a pupil constraint on the variance of the
    latent variable. Alpha is a hyperparam, can range from 0 to 1
    """
    # standard nmse (following nems.metrics.mse.nmse)
    X1 = result[pred_name].as_continuous()
    X2 = result[resp_name].as_continuous()
    respstd = np.nanstd(X2)
    squared_errors = (X1-X2)**2
    mse = np.sqrt(np.nanmean(squared_errors))
    nmse = mse / respstd

    # add pupil constraint on latent variable
    lv = result['lv'].as_continuous()
    pupil = result['pupil'].as_continuous()
    u = np.median(pupil)
    high = lv[:, pupil[0] >= u]
    low = lv[:, pupil[0] < u]

    pup_cost = -abs(np.var(high) - np.var(low))
    
    if np.var(lv) != 0:
        pup_cost /= np.var(lv)

    cost = (alpha * pup_cost) + ((1 - alpha) * nmse)

    return cost