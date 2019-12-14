import nems
import nems_lbhb.preprocessing as preproc
import numpy as np
import matplotlib.pyplot as plt
import json
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
    log.info(metric)
    if IsReload:
        return {}
    if (metric == 'pup_nmse') | (metric == 'nmse'):
        metric_fn = lambda d: pup_nmse(d, 'pred', output_name, alpha=context['alpha'])
    elif metric == 'pup_nc_nmse':
        metric_fn = lambda d: pup_nc_nmse(d, 'pred', output_name, alpha=context['alpha'])
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
    # constrain lv to have pupil-dependent variance
    lv = result['lv'].as_continuous()
    pupil = result['pupil'].as_continuous()
    u = np.median(pupil)
    high = lv[:, pupil[0] >= u]
    low = lv[:, pupil[0] < u]

    pup_cost = -abs(np.var(high) - np.var(low))
    
    if np.var(lv) != 0:
        pup_cost /= np.var(lv)

    cost = (alpha * pup_cost) + ((1 - alpha) * nmse)
    
    if ~np.isfinite(cost):
        import pdb; pdb.set_trace()

    return cost


def pup_nc_nmse(result, pred_name='pred', resp_name='resp', alpha=0):
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

    # add pupil constraint so that pairwise correlations
    # are equal between large/small after removing model prediction
    p_mask = result['p_mask'].as_continuous().squeeze()
    residual_big = X2[:, p_mask] - X1[:, p_mask]
    residual_small = X2[:, ~p_mask] - X1[:, ~p_mask]
    nc_big = np.corrcoef(residual_big)
    nc_small = np.corrcoef(residual_small)
    pup_cost = abs(nc_big - nc_small).mean()

    cost = (alpha * pup_cost) + ((1 - alpha) * nmse)

    if ~np.isfinite(cost):
        import pdb; pdb.set_trace()

    return cost


def add_summary_statistics(est, val, modelspec, fn='standard_correlation',
                           rec=None, use_mask=True, **context):
    """
    Add additional summary statistics to meta for Results table. Will be 
    saved as json encoded string in NarfResults

    modelspec['meta']['extra_results'] = json.dumps({'my_statisitc': value})

    return modelspec
    """

    # save noise correlation statistics
    import charlieTools.noise_correlations as nc
    import charlieTools.preprocessing as cpreproc
    r = val.apply_mask(reset_epochs=True)

    # regress out model pred method 1
    log.info("regress out model prediction using method 1, compute noise correlations")
    r12 = r.copy()
    if 'lv' in r12.signals.keys():
        r12['lv'] = r12['lv']._modified_copy(r12['lv']._data[1, :][np.newaxis, :])
        r12 = cpreproc.regress_state(r12, state_sigs=['pupil', 'lv'], regress=['pupil', 'lv'])
    else:
        r12 = cpreproc.regress_state(r12, state_sigs=['pupil'], regress=['pupil'])

    # mask pupil
    pup_ops = {'state': 'big', 'epoch': ['REFERENCE'], 'collapse': True}
    big = preproc.create_pupil_mask(r12.copy(), **pup_ops)
    big = big.apply_mask(reset_epochs=True)
    pup_ops['state'] = 'small'
    small = preproc.create_pupil_mask(r12.copy(), **pup_ops)
    small = small.apply_mask(reset_epochs=True)

    epochs = np.unique([e for e in r12.epochs.name if 'STIM' in e]).tolist()
    r_dict = r12['resp'].extract_epochs(epochs)
    epochs = np.unique([e for e in big.epochs.name if 'STIM' in e]).tolist()
    big_dict = big['resp'].extract_epochs(epochs)
    epochs = np.unique([e for e in small.epochs.name if 'STIM' in e]).tolist()
    small_dict = small['resp'].extract_epochs(epochs)

    all_df1 = nc.compute_rsc(r_dict)
    big_df1 = nc.compute_rsc(big_dict)
    small_df1 = nc.compute_rsc(small_dict)   

    # regress out model pred method 2
    log.info("regress out model prediction using method 2, compute noise correlations")
    r12 = r.copy()
    mod_data = r12['resp']._data - r12['pred']._data + r12['psth_sp']._data
    r12['resp'] = r12['resp']._modified_copy(mod_data)

    # mask pupil
    pup_ops = {'state': 'big', 'epoch': ['REFERENCE'], 'collapse': True}
    big = preproc.create_pupil_mask(r12.copy(), **pup_ops)
    big = big.apply_mask(reset_epochs=True)
    pup_ops['state'] = 'small'
    small = preproc.create_pupil_mask(r12.copy(), **pup_ops)
    small = small.apply_mask(reset_epochs=True)

    epochs = np.unique([e for e in r12.epochs.name if 'STIM' in e]).tolist()
    r_dict = r12['resp'].extract_epochs(epochs)
    epochs = np.unique([e for e in big.epochs.name if 'STIM' in e]).tolist()
    big_dict = big['resp'].extract_epochs(epochs)
    epochs = np.unique([e for e in small.epochs.name if 'STIM' in e]).tolist()
    small_dict = small['resp'].extract_epochs(epochs)

    all_df2 = nc.compute_rsc(r_dict)
    big_df2 = nc.compute_rsc(big_dict)
    small_df2 = nc.compute_rsc(small_dict) 


    results = {'rsc_all': all_df1['rsc'].mean(), 'rsc_all_sem': all_df1['rsc'].sem(),
               'rsc_big': big_df1['rsc'].mean(), 'rsc_big_sem': big_df1['rsc'].sem(),
               'rsc_small': small_df1['rsc'].mean(), 'rsc_small_sem': small_df1['rsc'].sem(),
               'rsc_all2': all_df2['rsc'].mean(), 'rsc_all_sem2': all_df2['rsc'].sem(),
               'rsc_big2': big_df2['rsc'].mean(), 'rsc_big_sem2': big_df2['rsc'].sem(),
               'rsc_small2': small_df2['rsc'].mean(), 'rsc_small_sem2': small_df2['rsc'].sem()}
    modelspec['meta']['extra_results'] = json.dumps(results)


    return {'modelspec': modelspec}