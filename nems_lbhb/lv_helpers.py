import nems
import nems_lbhb.preprocessing as preproc
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import copy
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
        metric_fn = lambda d: pup_nmse(d, 'pred', output_name, **context)
    elif metric == 'pup_nc_nmse':
        metric_fn = lambda d: pup_nc_nmse(d, 'pred', output_name, **context)
    elif metric == 'pup_dep_LVs':
        metric_fn = lambda d: pup_dep_LVs(d, 'pred', output_name, **context)
    fitter_fn = getattr(nems.fitters.api, fitter)
    fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}

    if modelspec[0]['fn_kwargs']['step'] == True:
        modelspec[0]['fn_kwargs']['p_only'] = True
        tfit_kwargs = fit_kwargs.copy()
        tfit_kwargs['tolerance'] = 1e-5
        log.info('Fit first order pupil')
        a = context['alpha']
        context['alpha'] = 0.0
        metric_fn = lambda d: pup_dep_LVs(d, 'pred', output_name, **context)
        modelspec = nems.analysis.api.fit_basic(
            est, modelspec, fit_kwargs=tfit_kwargs,
            metric=metric_fn, fitter=fitter_fn)
        modelspec[0]['fn_kwargs']['p_only'] = False
        context['alpha'] = a

    # option to freeze first-order pupil:
    """
    modelspec[0]['fn_kwargs']['pd'] = modelspec.phi[0]['pd']
    del modelspec.phi[0]['pd']
    modelspec = nems.analysis.api.fit_basic(est, modelspec, fit_kwargs=fit_kwargs,
        metric=metric_fn, fitter=fitter_fn)
    modelspec.phi[0]['pd'] = modelspec[0]['fn_kwargs']['pd']
    del modelspec.phi[0]['fn_kwargs']['pd']
    """

    modelspec = nems.analysis.api.fit_basic(est, modelspec, fit_kwargs=fit_kwargs,
        metric=metric_fn, fitter=fitter_fn)

    return {'modelspec': modelspec}


def pup_nmse(result, pred_name='pred', resp_name='resp', **context):
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
    
    alpha = context['alpha']
    
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


def pup_nc_nmse(result, pred_name='pred', resp_name='resp', **context):
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

    alpha = context['alpha']

    # add pupil constraint so that pairwise correlations
    # are equal between large/small after removing model prediction
    p_mask = result['p_mask'].as_continuous().squeeze()
    residual_big = X2[:, p_mask] - X1[:, p_mask]
    residual_small = X2[:, ~p_mask] - X1[:, ~p_mask]
    
    if 'stim_epochs' in result.signals.keys():
        # for minimizing per stimulus
        epochs = result['stim_epochs']._data
        idx = np.argwhere(epochs.sum(axis=-1) != 0)
        epochs = epochs[idx, :].squeeze()
        if len(epochs.shape)==1:
            epochs = epochs[np.newaxis, :]
        delta_nc = []
        for i in range(epochs.shape[0]):
            s_mask = epochs[i, :].astype(np.bool)
            rb = X2[:, s_mask & p_mask] - X1[:, s_mask & p_mask]
            rs = X2[:, s_mask & ~p_mask] - X1[:, s_mask & ~p_mask]
            nc_big = np.corrcoef(rb)
            nc_small = np.corrcoef(rs)
            delta_nc.append(np.nanmean(abs(nc_big - nc_small)))
        pup_cost = np.mean(delta_nc)

    else:
        # for minimizing over ALL stimuli
        nc_big = np.corrcoef(residual_big)
        nc_small = np.corrcoef(residual_small)
        pup_cost = abs(nc_big - nc_small).mean()

    # additional pupil constraint to set first order mod index to zero
    #residual_big = X2[:, p_mask] - X3[:, p_mask]
    #residual_small = X2[:, ~p_mask] - X3[:, ~p_mask]
    #mi_mean = np.nanmean((residual_big - residual_small / residual_big + residual_small))
    #pup_cost2 = mi_mean

    cost = (alpha * pup_cost) + ((1 - alpha) * nmse)

    if ~np.isfinite(cost):
        import pdb; pdb.set_trace()

    return cost


def pup_dep_LVs(result, pred_name='pred', resp_name='resp', **context):
    '''
    For purely LV model. Constrain first LV (lv_slow) to correlate with pupil,
    second LV (lv_fast) to have variance that correlates with pupil.
    Weigh these constraints vs. minimizing nsme.

    Will also work if only have one or the other of the two LVs
    '''
    result = result.apply_mask()
    lv_chans = result['lv'].chans
    X1 = result[pred_name].as_continuous()
    X2 = result[resp_name].as_continuous()
    respstd = np.nanstd(X2)
    squared_errors = (X1-X2)**2
    mse = np.sqrt(np.nanmean(squared_errors))
    nmse = mse / respstd

    alpha = context['alpha']
    signed_correlation = context['signed_correlation']

    if len(lv_chans) > 3:
        raise ValueError("Not set up to handle greater than 2 LVs right now due to \
                        complications with hyperparameter specification")

    if type(alpha) is dict:
        # passed different hyperparameters for each of the LVs
        fast_alpha = alpha['fast_alpha']
        slow_alpha = alpha['slow_alpha']

        if (fast_alpha + slow_alpha) > 1:
                raise ValueError("Hyperparameter values must sum to < 1")

    else:
        fast_alpha = slow_alpha = alpha

    if ('lv_fast' not in lv_chans) & ('lv_slow' not in lv_chans):
        # don't know how to constrain LV(s), just minimizing nmse
        return nmse
    
    elif ('lv_fast' in lv_chans) & ('lv_slow' in lv_chans):
        ref_len = result.meta['ref_len']
        p = result['pupil']._data.reshape(-1, ref_len)
        
        fast_lv_chans = [c for c in lv_chans if 'lv_fast' in c]
        fast_cc = []
        p = np.mean(p, axis=-1)
        for c in fast_lv_chans:
            lv_fast = result['lv'].extract_channels([c])._data.reshape(-1, ref_len)
            lv_fast = np.std(lv_fast, axis=-1)
            if signed_correlation:
                cc = lv_corr_pupil(p, lv_fast)
            else:
                cc = -abs(lv_corr_pupil(p, lv_fast))
            fast_cc.append(cc)


        p = result['pupil']._data
        lv_slow = result['lv'].extract_channels(['lv_slow'])._data
        slow_cc = -abs(lv_corr_pupil(p, lv_slow))

        cost = (slow_alpha * slow_cc) + ((1 - (slow_alpha + fast_alpha)) * nmse)

        for i, c in enumerate(fast_lv_chans):
            cost += (fast_alpha * fast_cc[i])

        return cost

    elif ('lv_fast' in lv_chans):
        ref_len = result.meta['ref_len']
        p = result['pupil']._data.reshape(-1, ref_len)
        lv_fast = result['lv'].extract_channels(['lv_fast'])._data.reshape(-1, ref_len)
        
        p = np.mean(p, axis=-1)
        lv_fast = np.std(lv_fast, axis=-1)
        if signed_correlation:
            fast_cc = lv_corr_pupil(p, lv_fast)
        else:
            fast_cc = -abs(lv_corr_pupil(p, lv_fast))

        if np.sum(lv_fast)==0:
            fast_cc = 0

        cost = (fast_alpha * fast_cc) + ((1 - fast_alpha) * nmse)
        return cost

    elif ('lv_slow' in lv_chans):
        p = result['pupil']._data
        lv_slow = result['lv'].extract_channels(['lv_slow'])._data
        slow_cc = -abs(lv_corr_pupil(p, lv_slow))

        cost = (slow_alpha * slow_cc) + ((1 - slow_alpha) * nmse)
        return cost

def lv_corr_pupil(p, lv):
    """
    return correlation of pupil and lv
    """
    return np.corrcoef(p.squeeze(), lv.squeeze())[0, 1]

def lv_var_corr_pupil(p, lv, fs):
    """
    return corr. between var(lv) and p for sliding window
    """
    # 5 sec window
    win_size = int(5 * fs)
    p_roll = rolling_window(p, win_size).squeeze()
    lv_roll = rolling_window(lv, win_size).squeeze()

    p_mean = np.mean(p_roll, axis=-1)
    lv_var = np.var(lv_roll, axis=-1)

    return np.corrcoef(p_mean, lv_var)[0, 1]

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window +1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def add_summary_statistics(est, val, modelspec, fn='standard_correlation',
                           rec=None, use_mask=True, **context):
    """
    Add additional summary statistics to meta for Results table. Will be 
    saved as json encoded string in NarfResults

    modelspec['meta']['extra_results'] = json.dumps({'my_statisitc': value})

    return modelspec
    """
    r = val.copy()
    r = r.apply_mask(reset_epochs=True)
    ref_len = r['resp'].extract_epoch('REFERENCE').shape[-1]
    p = r['pupil']._data.reshape(-1, ref_len).mean(axis=-1)
    lv = r['lv']._data.reshape(-1, ref_len).std(axis=-1)

    cc = np.corrcoef(p, lv)[0, 1]

    results = {'lv_power_vs_pupil': cc}

    modelspec.meta['extra_results'] = json.dumps(results)

    return {'modelspec': modelspec}


def dc_lv_model(rec, ss, o, p_only, flvw, step, pd, lvd, d, lve):
    """
    fit lv model for N neurons
    :param rec:
    :param ss: subtracting signal name ('psth' or 'pred')
    :param o: name of output (typically 'pred')
    :param p_only: (if True) only fit first-order pupil
    :param flvw: (if True) force encoding and decoding weights for lv to be the same
    :param step: (if True) intialize fit with first order weights, then fit full model
    :param pd: first-order pupil weights (dc shift) - N x 1
    :param lvd: free parameter - latent variable decoding weights, also encoding
                weights if flvw==True
    :param d: free parameter - offset of first-order pred (N x 1)
    :param lve: free parameter - latent variable encoding weights (if flvw==False),
                otherwise not used
    :return: [lv, pred, residual] (residual= signal used to fit lv component)
    """
    resp = rec['resp'].rasterize()._data
    psth = rec['psth'].rasterize()._data
    psth_sp = rec['psth_sp'].rasterize()._data
    pupil = copy.deepcopy(rec['state'].extract_channels(['pupil']))._data

    pred1 = (pd @ pupil) + psth + d

    # define residual
    if ss == 'psth':
        residual = resp - psth_sp

    elif ss == 'pred':
        # do first order prediction first
        residual = resp - pred1

    # compute latent variable
    if flvw:
        lv = lvd.T @ residual
    else:
        lv = lve.T @ residual
    
    if p_only:
        pred = pred1
    else:
        # compute full prediction
        pred = psth + (pd @ pupil) + (lvd @ lv) + d

    lv = rec['pupil']._modified_copy(lv)
    lv.name = 'lv'
    lv.chans = ['lv_fast']
    residual = rec['pupil']._modified_copy(residual)
    residual.name = 'residual'
    pred = rec['resp'].rasterize()._modified_copy(pred)
    pred.name = 'pred'

    return [lv, residual, pred]

def gain_lv_model(rec, ss, o, g, p_only, flvw, step, pg, lvg, d, lve):

    resp = rec['resp'].rasterize()._data
    psth = rec['psth'].rasterize()._data
    psth_sp = rec['psth_sp'].rasterize()._data
    pupil = copy.deepcopy(rec['state'].extract_channels(['pupil']))._data

    pred1 = (g * psth) + ((pg @ pupil) * psth) + d
    # define residual
    if ss == 'psth':
        residual = resp - psth_sp
    elif ss == 'pred':       
        residual = resp - pred1

    # compute latent variable
    if flvw:
        lv = lvg.T @ residual
    else:
        lv = lve.T @ residual

    # compute full prediction
    if p_only:
        pred = pred1
    else:
        pred = (g * psth) + ((pg @ pupil) * psth) + ((lvg @ lv) * psth) + d

    lv = rec['pupil']._modified_copy(lv)
    lv.name = 'lv'
    lv.chans = ['lv_fast']
    residual = rec['pupil']._modified_copy(residual)
    residual.name = 'residual'
    pred = rec['resp'].rasterize()._modified_copy(pred)
    pred.name = 'pred'

    return [lv, residual, pred]

def full_lv_model(rec, ss, o, p_only, step, g, pg, lvg, pd, lvd, d, lve):

    resp = rec['resp'].rasterize()._data
    psth = rec['psth'].rasterize()._data
    psth_sp = rec['psth_sp'].rasterize()._data
    pupil = copy.deepcopy(rec['state'].extract_channels(['pupil']))._data

    pred1 = (g * psth) + ((pg @ pupil) * psth) + (pd @ pupil) + d
    # define residual
    if ss == 'psth':
        residual = resp - psth_sp
    elif ss == 'pred':
        residual = resp - pred1

    # compute latent variable
    lv = lve.T @ residual

    # compute full prediction
    if p_only:
        pred = pred1
    else:
        predg = (g * psth) + ((pg @ pupil) * psth) + ((lvg @ lv) * psth) + d
        preddc = (pd @ pupil) + (lvd @ lv)

        pred = predg + preddc

    lv = rec['pupil']._modified_copy(lv)
    lv.name = 'lv'
    lv.chans = ['lv_fast']
    residual = rec['pupil']._modified_copy(residual)
    residual.name = 'residual'
    pred = rec['resp'].rasterize()._modified_copy(pred)
    pred.name = 'pred'

    return [lv, residual, pred]