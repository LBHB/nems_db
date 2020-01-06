import nems
import nems_lbhb.preprocessing as preproc
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
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

    '''
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
    '''
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
            lv_fast = np.var(lv_fast, axis=-1)

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
        lv_fast = np.var(lv_fast, axis=-1)
        if signed_correlation:
            fast_cc = lv_corr_pupil(p, lv_fast)
        else:
            fast_cc = -abs(lv_corr_pupil(p, lv_fast))

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
    big = r12.copy()
    big['mask'] = big['p_mask']
    small = r12.copy()
    small['mask'] = small['p_mask']._modified_copy(~small['p_mask']._data)
    
    big = big.apply_mask(reset_epochs=True)
    small = small.apply_mask(reset_epochs=True)

    epochs = np.unique([e for e in r12.epochs.name if 'STIM' in e]).tolist()
    r_dict = r12['resp'].extract_epochs(epochs)
    big_dict = big['resp'].extract_epochs(epochs)
    small_dict = small['resp'].extract_epochs(epochs)

    all_df1 = nc.compute_rsc(r_dict)
    big_df1 = nc.compute_rsc(big_dict)
    small_df1 = nc.compute_rsc(small_dict)   
    
    # per stim
    perstim_df = pd.DataFrame(index=epochs, columns=['rsc', 'sem'])
    big_perstim = pd.DataFrame(index=epochs, columns=['rsc', 'sem'])
    small_perstim = pd.DataFrame(index=epochs, columns=['rsc', 'sem'])
    for e in epochs:
        r_dict = r12['resp'].extract_epochs(e)
        big_dict = big['resp'].extract_epochs(e)
        small_dict = small['resp'].extract_epochs(e)

        perstim_df.loc[e, 'rsc'] = nc.compute_rsc(r_dict)['rsc'].mean()
        big_perstim.loc[e, 'rsc'] = nc.compute_rsc(big_dict)['rsc'].mean()
        small_perstim.loc[e, 'rsc'] = nc.compute_rsc(small_dict)['rsc'].mean()

    # =========== regress out model pred method 2 ==================
    log.info("regress out model prediction using method 2, compute noise correlations")
    r12 = r.copy()
    mod_data = r12['resp']._data - r12['pred']._data + r12['psth_sp']._data
    r12['resp'] = r12['resp']._modified_copy(mod_data)

    # mask pupil
    big = r12.copy()
    big['mask'] = big['p_mask']
    small = r12.copy()
    small['mask'] = small['p_mask']._modified_copy(~small['p_mask']._data)

    big = big.apply_mask(reset_epochs=True)
    small = small.apply_mask(reset_epochs=True)

    epochs = np.unique([e for e in r12.epochs.name if 'STIM' in e]).tolist()
    r_dict = r12['resp'].extract_epochs(epochs)
    big_dict = big['resp'].extract_epochs(epochs)
    small_dict = small['resp'].extract_epochs(epochs)

    all_df2 = nc.compute_rsc(r_dict)
    big_df2 = nc.compute_rsc(big_dict)
    small_df2 = nc.compute_rsc(small_dict) 

    # per stim
    perstim2_df = pd.DataFrame(index=epochs, columns=['rsc', 'sem'])
    big_perstim2 = pd.DataFrame(index=epochs, columns=['rsc', 'sem'])
    small_perstim2 = pd.DataFrame(index=epochs, columns=['rsc', 'sem'])
    for e in epochs:
        r_dict = r12['resp'].extract_epochs(e)
        big_dict = big['resp'].extract_epochs(e)
        small_dict = small['resp'].extract_epochs(e)

        perstim2_df.loc[e, 'rsc'] = nc.compute_rsc(r_dict)['rsc'].mean()
        big_perstim2.loc[e, 'rsc'] = nc.compute_rsc(big_dict)['rsc'].mean()
        small_perstim2.loc[e, 'rsc'] = nc.compute_rsc(small_dict)['rsc'].mean()

    results = {'rsc_all': all_df1['rsc'].mean(), 'rsc_all_sem': all_df1['rsc'].sem(),
               'rsc_big': big_df1['rsc'].mean(), 'rsc_big_sem': big_df1['rsc'].sem(),
               'rsc_small': small_df1['rsc'].mean(), 'rsc_small_sem': small_df1['rsc'].sem(),
               'rsc_all2': all_df2['rsc'].mean(), 'rsc_all_sem2': all_df2['rsc'].sem(),
               'rsc_big2': big_df2['rsc'].mean(), 'rsc_big_sem2': big_df2['rsc'].sem(),
               'rsc_small2': small_df2['rsc'].mean(), 'rsc_small_sem2': small_df2['rsc'].sem(),
               'rsc_perstim_small': small_perstim['rsc'].mean(), 'rsc_perstim_small_sem': np.nan,
               'rsc_perstim_big': big_perstim['rsc'].mean(), 'rsc_perstim_big_sem': np.nan,
               'rsc_perstim_small2': small_perstim2['rsc'].mean(), 'rsc_perstim_small_sem2': np.nan,
               'rsc_perstim_big2': big_perstim2['rsc'].mean(), 'rsc_perstim_big_sem2': np.nan}

    modelspec['meta']['extra_results'] = json.dumps(results)


    return {'modelspec': modelspec}