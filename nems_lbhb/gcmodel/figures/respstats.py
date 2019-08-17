import os
import logging

import numpy as np
import scipy.stats as st
from scipy import ndimage
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

import nems.xform_helper as xhelp
import nems.db as nd
import nems.epoch as ep
from nems.plots.heatmap import _get_fir_coefficients, _get_wc_coefficients
from nems_lbhb.gcmodel.figures.utils import (improved_cells_to_list,
                                             get_filtered_cellids,
                                             get_dataframes,
                                             adjustFigAspect)
from nems_lbhb.gcmodel.figures.soundstats import silence_duration
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems.recording import load_recording
from nems.preprocessing import average_away_epoch_occurrences
from nems_lbhb.gcmodel.figures.definitions import *

log = logging.getLogger(__name__)
plt.rcParams.update(params)


def rate_by_batch(batch, cells=None, stat='max', fs=100):
    if cells is None:
        cells = nd.get_batch_cells(batch, as_list=True)
    rates = []
    for cellid in cells:
        # should be able to do this with just the recording somehow, but
        # I guess i'm missing a step. So for now just load the evaluated model
#        loadkey = 'ozgf.fs%d.ch18' % fs
#        recording_uri = generate_recording_uri(cellid=cellid, batch=batch,
#                                               loadkey=loadkey, stim=False)
#        rec = load_recording(recording_uri)
#        rec['resp'] = rec['resp'].rasterize()
#        avgrec = average_away_epoch_occurrences(rec)
#        resp = avgrec['resp'].extract_channels([cellid]).as_continuous().flatten()
#        #resp = rec['resp'].extract_channels([cellid])
#        #avgresp = generate_average_sig(resp)

        # using ln_dexp3 because it should be the fastest to evaluate,
        # but actual model doesn't matter since we only need response
        xfspec, ctx = xhelp.load_model_xform(cellid, batch, ln_dexp3)
        resp = ctx['val'].apply_mask()['resp'].as_continuous().flatten()
        if stat == 'max':
            raw_max = np.nanmax(resp)
            mean_3sd = np.nanmean(resp) + 3*np.nanstd(resp)
            max_rate = min(raw_max, mean_3sd)
            rates.append(max_rate)
        elif stat == 'spont':
            epochs = ctx['val'].apply_mask()['resp'].epochs
            stim_epochs = ep.epoch_names_matching(epochs, 'STIM_')
            pre_silence = silence_duration(epochs, 'PreStimSilence')
            silence_only = np.empty(0,)
            # cut out only the pre-stim silence portions
            for s in stim_epochs:
                row = epochs[epochs.name == s]
                pre_start = int(row['start'].values[0]*fs)
                stim_start = int((row['start'].values[0] + pre_silence)*fs)
                silence_only = np.append(silence_only, resp[pre_start:stim_start])

            spont_rate = np.nanmean(silence_only)
            rates.append(spont_rate)

        else:
            raise ValueError("unrecognized stat: use 'max' for maximum or "
                             "'spont' for spontaneous")


    results = {'cellid': cells, 'rate': rates}
    df = pd.DataFrame.from_dict(results)
    df.set_index('cellid', inplace=True)

    return df


def rate_vs_performance(batch, gc, stp, LN, combined, compare='gc',
                        se_filter=True, LN_filter=False, plot_stat='r_ceiling',
                        normalize_rates=False, load_path=None,
                        save_path=None, rate_stat='max', fs=100,
                        relative_performance=False):
    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
#    cellids, under_chance, less_LN = get_filtered_cellids(df_r, df_e, gc, stp,
#                                                          LN, combined,
#                                                          se_filter,
#                                                          LN_filter)

    e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           as_lists=False)

    if plot_stat == 'r_ceiling':
        plot_df = df_c
    else:
        plot_df = df_r

    if compare == 'gc':
        model = gc
        improved = g
        not_improved = (a & np.logical_not(g))
    elif compare == 'stp':
        model = stp
        improved = s
        not_improved = (a & np.logical_not(s))
    elif compare == 'combined':
        model = combined
        improved = c
        not_improved = (a & np.logical_not(c))
    else:
        raise ValueError('model comparison not recognized: %s' % compare)

    if load_path is None:
        df = rate_by_batch(batch, stat=rate_stat, fs=fs)
        if save_path is not None:
            df.to_pickle(save_path)
    else:
        df = pd.read_pickle(load_path)

    df_remove = []
    filter_remove = []
    if not (sorted(df['rate'].index.values) == sorted(improved.index.values)):
        log.warning('index mismatch in rate analysis, only keeping matched cells')
        for cell in improved.index.values:
            if cell not in df['rate'].index.values:
                filter_remove.append(cell)
        for cell in df['rate'].index.values:
            if cell not in improved.index.values:
                df_remove.append(cell)

        for cell in df_remove:
            df.drop(cell, inplace=True)
        for cell in filter_remove:
            improved.drop(cell, inplace=True)
            not_improved.drop(cell, inplace=True)
            a.drop(cell, inplace=True)
        if not sorted(df.index.values) == sorted(improved.index.values):
            raise ValueError('still have an index mismatch after fix')
    cellids = sorted(df.index.values.tolist())
    # also sort all indexes to make sure the values are in the same order
    plot_df.sort_index(inplace=True)
    df.sort_index(inplace=True)
    improved.sort_index(inplace=True)
    not_improved.sort_index(inplace=True)
    a.sort_index(inplace=True)

    imp_rates = df['rate'][improved].values.astype('float32')
    not_rates = df['rate'][not_improved].values.astype('float32')
    all_rates = df['rate'][a].values.astype('float32')
    n_imp = imp_rates.size
    n_not = not_rates.size
    n_all = all_rates.size

    imp_scores = plot_df[model][cellids][improved].values.astype('float32')
    not_scores = plot_df[model][cellids][not_improved].values.astype('float32')
    all_scores = plot_df[model][cellids][a].values.astype('float32')
    if normalize_rates:
        not_rates /= not_rates.max()
        imp_rates /= imp_rates.max()
        all_rates /= all_rates.max()
    if relative_performance:
        imp_LN = plot_df[LN][cellids][improved].values.astype('float32')
        not_LN = plot_df[LN][cellids][not_improved].values.astype('float32')
        all_LN = plot_df[LN][cellids][a].values.astype('float32')
        imp_scores = imp_scores - imp_LN
        not_scores = not_scores - not_LN
        all_scores = all_scores - all_LN

    r_all, p_all = st.pearsonr(all_rates, all_scores)
    r_imp, p_imp = st.pearsonr(imp_rates, imp_scores)
    r_not, p_not = st.pearsonr(not_rates, not_scores)

    fig = plt.figure()
    ax = plt.gca()
    imp_color = model_colors[compare]
    not_color = model_colors['LN']
    ax.scatter(imp_rates, imp_scores, color=imp_color, s=30,
               label='sig. imp.')
    ax.scatter(not_rates, not_scores, color=not_color, s=20, alpha=0.7,
               label='not imp.')
    ax.legend()
    adjustFigAspect(fig, aspect=1)
    plt.xlabel("%s rate\n"
               "normalized? %s" % (rate_stat, normalize_rates))
    plt.ylabel("%s\n"
               "relative to LN?  %s" % (plot_stat, relative_performance))

    title = ("%s vs %s model performance\n"
             "all -- r:  %.4f, p:  %.4E, n:  %d\n"
             "imp -- r:  %.4f, p:  %.4E, n:  %d\n"
             "not -- r:  %.4f, p:  %.4E, n:  %d"
             % (rate_stat, compare, r_all, p_all, n_all, r_imp, p_imp,
                n_imp, r_not, p_not, n_not))

    plt.title(title)
    plt.tight_layout()
    return fig


def strf_vs_resp_by_contrast(cellid, batch, modelname,
                             plot_stim=False, plot_contrast=False,
                             continuous=False, bin_count=40):

    xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)
    val = ctx['val'].apply_mask()
    pred = val['pred'].as_continuous().flatten()
    resp = val['resp'].as_continuous().flatten()
    stim = val['stim'].as_continuous()
    contrast = val['contrast'].as_continuous()
    summed_contrast = np.sum(contrast, axis=0)

    fig = plt.figure(figsize=(7,7))
    if continuous:
        plt.scatter(pred, resp, c=summed_contrast, alpha=0.75,
                    cmap=plt.get_cmap('plasma'))
    else:
        median_contrast = np.percentile(summed_contrast, 50)
        first_quartile = np.percentile(summed_contrast, 25)
        third_quartile = np.percentile(summed_contrast, 75)
        lower_contrast_mask = summed_contrast < first_quartile
        low_contrast_mask = ((summed_contrast >= first_quartile)
                             & (summed_contrast < median_contrast))
        med_contrast_mask = ((summed_contrast >= median_contrast)
                             & (summed_contrast < third_quartile))
        high_contrast_mask = (summed_contrast >= third_quartile)

        # NOTE: commenting out for now, just too messy to see much
        # plot the un-averaged data in the background
#        plt.scatter(pred[lower_contrast_mask], resp[lower_contrast_mask],
#                    color='gray', alpha=0.1, s=5)
#        plt.scatter(pred[low_contrast_mask], resp[low_contrast_mask],
#                    color='blue', alpha=0.1, s=5)
#        plt.scatter(pred[med_contrast_mask], resp[med_contrast_mask],
#                    color='#B089E1', alpha=0.1, s=5)
#        plt.scatter(pred[high_contrast_mask], resp[high_contrast_mask],
#                    color='red', alpha=0.1, s=5)

        # break each quartile into # bins & average, for
        # 4 * # total points, to smooth out the plots
        mean_pred, bin_masks = _binned_xvar(pred, bin_count)
        r_lower = _binned_yavg(resp, lower_contrast_mask, bin_masks)
        r_low = _binned_yavg(resp, low_contrast_mask, bin_masks)
        r_med = _binned_yavg(resp, med_contrast_mask, bin_masks)
        r_high = _binned_yavg(resp, high_contrast_mask, bin_masks)

        # then plot over raw data
        plasma = plt.get_cmap('plasma')
        c1, c2, c3, c4 = [plasma(n) for n in [.1, .4, .7, .9]]

        plt.scatter(mean_pred, r_lower, color=c1, s=40,
                    label='lower (binned avg)')
        plt.scatter(mean_pred, r_low, color=c2, s=40,
                    label='low')
        plt.scatter(mean_pred, r_med, color=c3, s=40,
                    label='medium')
        plt.scatter(mean_pred, r_high, color=c4, s=40,
                    label='high')

        plt.legend()

    plt.xlabel('linear model prediction')
    plt.ylabel('actual response')
    plt.title(cellid)
    if continuous:
        plt.colorbar()

    if plot_stim:
        _, (a1, a2, a3, a4) = plt.subplots(4, 1)
        clim = [np.nanmin(stim), np.nanmax(stim)]
        a1.imshow(stim[:, lower_contrast_mask], aspect='auto', origin='lower',
                  clim=clim)
        a2.imshow(stim[:, low_contrast_mask], aspect='auto', origin='lower',
                  clim=clim)
        a3.imshow(stim[:, med_contrast_mask], aspect='auto', origin='lower',
                  clim=clim)
        a4.imshow(stim[:, high_contrast_mask], aspect='auto', origin='lower',
                  clim=clim)
        a1.set_ylabel('lower')
        a1.get_xaxis().set_visible(False)
        a2.set_ylabel('low')
        a2.get_xaxis().set_visible(False)
        a3.set_ylabel('medium')
        a3.get_xaxis().set_visible(False)
        a4.set_ylabel('high')
        plt.title('stim')

    if plot_contrast:
        _, (a1, a2, a3, a4) = plt.subplots(4, 1)
        clim = [np.nanmin(contrast), np.nanmax(contrast)]
        a1.imshow(contrast[:, lower_contrast_mask], aspect='auto', origin='lower',
                  clim=clim)
        a2.imshow(contrast[:, low_contrast_mask], aspect='auto', origin='lower',
                  clim=clim)
        a3.imshow(contrast[:, med_contrast_mask], aspect='auto', origin='lower',
                  clim=clim)
        a4.imshow(contrast[:, high_contrast_mask], aspect='auto', origin='lower',
                  clim=clim)
        a1.set_ylabel('lower')
        a1.get_xaxis().set_visible(False)
        a2.set_ylabel('low')
        a2.get_xaxis().set_visible(False)
        a3.set_ylabel('medium')
        a3.get_xaxis().set_visible(False)
        a4.set_ylabel('high')
        plt.title('contrast')

    return fig


def _binned_xvar(x, bin_count):
    bin_edges = np.linspace(x.min(), x.max(), bin_count+1)
    midpoints = (bin_edges[:-1] + bin_edges[1:])/2
    bin_masks = []
    for i, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (x >= lower) & (x < upper)
        if i == (bin_count - 1):
            # last bin, allow equal final upper bound
            mask = mask | (x == upper)
        bin_masks.append(mask)

    return midpoints, bin_masks


def _binned_yavg(y, mask, bin_masks):
    binned_ys = [y[mask & m] for m in bin_masks]
    mean_y = np.array([np.mean(b) for b in binned_ys])
    return mean_y


def strf_vs_resp_batch(batch, modelname, save_path, test_limit=None,
                       continuous=False):
    cells = nd.get_batch_cells(batch, as_list=True)
    #plot_kwargs = {'alpha': 0.2, 's': 2}
    for cellid in cells[:test_limit]:
        try:
            fig = strf_vs_resp_by_contrast(cellid, batch, modelname,
                                           plot_stim=False, plot_contrast=False,
                                           continuous=continuous)
            full_path = os.path.join(save_path, str(batch), cellid)
            fig.savefig(full_path, format='pdf')
            plt.close(fig)
        except:
            # cell probably not fit for this model or batch
            print('error for cell: %s' % cellid)
            continue


def filtered_strf_vs_resp_batch(batch, gc, stp, LN, combined, strf, save_path,
                                good_ln=0.0, test_limit=None, stat='r_ceiling',
                                bin_count=40):

    e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined,
                                           good_ln=good_ln)
    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    if stat == 'r_ceiling':
        df = df_c
    else:
        df = df_r

    tags = ['either', 'neither', 'gc', 'stp', 'combined']
    for cells, tag in zip([e, a, g, s, c], tags):
        _strf_resp_sub_batch(cells[:test_limit], df, tag, stat, batch,
                             gc, stp, LN, combined, strf, save_path,
                             bin_count=bin_count)


def _strf_resp_sub_batch(cells, df, tag, stat, batch, gc, stp, LN,
                         combined, strf, save_path, bin_count):
    for cellid in cells:
        try:
            fig = strf_vs_resp_by_contrast(cellid, batch, strf, plot_stim=False,
                                           plot_contrast=False, continuous=False,
                                           bin_count=bin_count)
            gc_r = df[gc][cellid]
            stp_r = df[stp][cellid]
            LN_r = df[LN][cellid]
            combined_r = df[combined][cellid]
        except:
            # model probably not fit for that cell
            continue

        full_path = os.path.join(save_path, str(batch), tag, cellid)
        parent_directory = '/'.join(full_path.split('/')[:-1])
        if not os.path.exists(parent_directory):
            os.makedirs(parent_directory, mode=0o777)

        fig.suptitle("model performances, %s:\n"
                     "gc: %.4f  |stp: %.4f  |LN: %.4f  |comb.: %.4f"
                     % (stat, gc_r, stp_r, LN_r, combined_r))
        fig.savefig(full_path, format='pdf', dpi=fig.dpi)
        plt.close(fig)


def cf_from_LN_strf(cellid, batch, modelname, f_low=0.2, f_high=20,
                    nf=18, method='gaussian'):
    # calculate characteristic frequency based on STRF from LN model
    # assemble STRF from parameters, take absolute value,
    # sum over time, then find center of mass along frequency
    strf = get_strf(cellid, batch, modelname)
    abs_strf = np.abs(strf)
    # extra dim just added back in for plotting convenience
    summed_time = np.expand_dims(np.sum(abs_strf, axis=1), axis=-1)
    khz_freqs = np.logspace(np.log(f_low), np.log(f_high), num=nf, base=np.e)

    if method == 'com':
        # use center of mass
        com_bin, _ = ndimage.measurements.center_of_mass(summed_time)
        cf_bin = int(round(com_bin))
        cf = khz_freqs[cf_bin]

    elif method == 'softmax':
        # use maximum of probabilities generated by softmax transformation
        def softmax(x):
            return np.exp(x)/sum(np.exp(x))

        probabilities = softmax(summed_time.flatten())
        cf_bin = np.argmax(probabilities)
        cf = khz_freqs[cf_bin]

    else:
        # use mean of gaussian fit
        def fn(x, mu, sigma, a, s):
            exponent = -0.5*((x-mu)/sigma)**2
            y = a*(1/(sigma*np.sqrt(2*np.pi))) * np.exp(exponent) + s
            return y
        ydata = summed_time.flatten()
        xdata = np.arange(ydata.size)
        bounds = (0, np.array([nf-1, np.inf, np.inf, np.inf]))
        (mu, sigma, a, s), _ = curve_fit(fn, xdata, ydata, bounds=bounds)
        cf_bin = int(round(mu))
        cf = khz_freqs[cf_bin]

    return cf, cf_bin


def cf_batch(batch, modelname, save_path=None, load_path=None, f_low=0.2,
             f_high=20, nf=18, method='gaussian', test_limit=None):
    if load_path is None:
        cells = nd.get_batch_cells(batch, as_list=True)
        cfs = []
        cf_bins = []
        skipped = []
        for cellid in cells[:test_limit]:
            try:
                cf, cf_bin = cf_from_LN_strf(cellid, batch, modelname, f_low,
                                             f_high, nf, method)
                cfs.append(cf)
                cf_bins.append(cf_bin)
            except:
                # cell probably not fit for this model
                skipped.append(cellid)
                continue

        cellid_index = [c for c in cells[:test_limit] if c not in skipped]
        results = {'cellid': cellid_index, 'cf': cfs, 'cf_bin': cf_bins}
        df = pd.DataFrame.from_dict(results)
        df.set_index('cellid', inplace=True)
        if save_path is not None:
            df.to_pickle(save_path)
    else:
        df = pd.read_pickle(load_path)

    return df


def cf_batch_rank1(batch, modelname, save_path=None, load_path=None, f_low=0.2,
                   f_high=20, nf=18, test_limit=None):

    if load_path is not None:
        df = pd.read_pickle(load_path)
        return df

    cells = nd.get_batch_cells(batch, as_list=True)
    cfs = []
    cf_bins = []
    skipped = []
    for cellid in cells[:test_limit]:
        try:
            xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname,
                                                 eval_model=False)
        except:
            # cell probably not fit for this model
            skipped.append(cellid)
            continue

        modelspec = ctx['modelspec']
        # mult by nf b/c x vals in gaussian coeffs module are divided by
        # number of channels
        # max and min bounds b/c means outside of bin range are allowed
        mean = min(max(0, np.asscalar(modelspec.phi[1]['mean'])*nf), nf-1)
        khz_freqs = np.logspace(np.log(f_low), np.log(f_high), num=nf,
                                base=np.e)
        cf_bin = int(round(mean))
        cf = khz_freqs[cf_bin]
        cfs.append(cf)
        cf_bins.append(cf_bin)

    cellid_index = [c for c in cells[:test_limit] if c not in skipped]
    results = {'cellid': cellid_index, 'cf': cfs, 'cf_bin': cf_bins}
    df = pd.DataFrame.from_dict(results)
    df.set_index('cellid', inplace=True)
    if save_path is not None:
        df.to_pickle(save_path)

    return df


def plot_cf_on_strf(cellid, batch, modelname, cf_load_path=None, cf_kwargs={}):
    if cf_load_path is None:
        cf, cf_bin = cf_from_LN_strf(cellid, batch, modelname, **cf_kwargs)
    else:
        df = pd.read_pickle(cf_load_path)
        cf = df['cf'][cellid]
        cf_bin = df['cf_bin'][cellid]
    strf = get_strf(cellid, batch, modelname)

    fig = plt.figure()
    plt.imshow(strf, aspect='auto', origin='lower', cmap='jet')
    x_bins = np.arange(strf.shape[-1])
    plt.plot(x_bins, np.full_like(x_bins, cf_bin), color='black', linewidth=3)
    return fig



def compare_cf_methods(batch, modelname, rank1, save_path=None, cf_kwargs={},
                       load_paths={}, test_limit=None):
    # Try each of the 4 methods available for CF, plot them vs the STRF
    # for each cell in the batch, and save

    # Assemble results for each method
    if 'com' in load_paths:
        com_df = pd.read_pickle(load_paths['com'])
    else:
        com_df = cf_batch(batch, modelname, method='com',
                          test_limit=test_limit, **cf_kwargs)

    if 'gaussian' in load_paths:
        gaus_df = pd.read_pickle(load_paths['gaussian'])
    else:
        gaus_df = cf_batch(batch, modelname, method='gaussian',
                           test_limit=test_limit, **cf_kwargs)

    if 'softmax' in load_paths:
        sm_df = pd.read_pickle(load_paths['softmax'])
    else:
        sm_df = cf_batch(batch, modelname, method='softmax',
                         test_limit=test_limit, **cf_kwargs)

    if 'rank1' in load_paths:
        r1_df = pd.read_pickle(load_paths['rank1'])
    else:
        r1_df = cf_batch_rank1(batch, rank1, test_limit=test_limit,
                               **cf_kwargs)

    cells = nd.get_batch_cells(batch, as_list=True)
    for cellid in cells[:test_limit]:
        try:
            strf = get_strf(cellid, batch, modelname)
            com_bin = com_df['cf_bin'][cellid]
            gaus_bin = gaus_df['cf_bin'][cellid]
            sm_bin = sm_df['cf_bin'][cellid]
            r1_bin = r1_df['cf_bin'][cellid]
        except:
            # probably not fit for this cell
            continue

        fig = plt.figure(figsize=(9,7))

        # show pseudo frequency kernel alongside STRF, since that's
        # what the CF function actually  bases it's judgement on
        abs_strf = np.abs(strf)
        summed_time = np.expand_dims(np.sum(abs_strf, axis=1), axis=-1)
        normed_freq = summed_time / np.max(summed_time)
        residual = np.flip(normed_freq.flatten())
        normed_strf = strf / np.max(abs_strf)
        spacer = np.full_like(summed_time, np.nan)
        concatted = np.concatenate((spacer, spacer, normed_freq, spacer,
                                    normed_strf), axis=1)
        x_bins = np.arange(abs_strf.shape[-1]+4)
        y_bins = np.arange(normed_freq.size)

        plt.imshow(concatted, origin='lower', cmap='jet')
        plt.plot(x_bins, np.full_like(x_bins, com_bin), color='black',
                 linewidth=7, linestyle='-', label='CoM')
        plt.plot(x_bins, np.full_like(x_bins, sm_bin), color='white',
                 linewidth=5, linestyle='-', label='SoftMax')
        plt.plot(x_bins, np.full_like(x_bins, gaus_bin), color='black',
                 linewidth=4, linestyle='--', label='Gaussian')
        plt.plot(x_bins, np.full_like(x_bins, r1_bin), color='white',
                 linewidth=2, linestyle='--', label='Rank 1')
        plt.plot(residual, y_bins, color='black', linewidth=2)
        plt.plot(np.zeros_like(residual), y_bins, color='gray',
                 linestyle='--', linewidth=1)
        plt.plot(np.ones_like(residual), y_bins, color='gray',
                 linestyle='--', linewidth=1)
        plt.xlabel('Time Lag')
        plt.ylabel('Frequecy')
        plt.legend(framealpha=0.0)
        plt.title(cellid)
        plt.tight_layout()

        if save_path is not None:
            full_path = os.path.join(save_path, str(batch), cellid)
            parent_path = os.path.join(save_path, str(batch))
            if not os.path.exists(parent_path):
                os.makedirs(parent_path, mode=0o777)
            fig.savefig(full_path, format='pdf', dpi=fig.dpi)
            plt.close(fig)


def cf_distribution(batch, modelname, load_path=None, bins=60, cf_kwargs={},
                    plot_bins=False):
    if load_path is None:
        df = cf_batch(batch, modelname, load_path=load_path, **cf_kwargs)
    else:
        df = pd.read_pickle(load_path)
    plt.figure()
    if plot_bins:
        values = df['cf_bin'].values
    else:
        values = df['cf'].values
    plt.hist(values, bins=bins, color=[wsu_gray_light],
             edgecolor='black', linewidth=1)


def cf_vs_model_performance(batch, gc, stp, LN, combined, cf_load_path=None,
                            cf_kwargs={}, se_filter=True, LN_filter=False,
                            plot_stat='r_ceiling', include_LN=False,
                            include_combined=False, only_improvements=False):
    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    cellids, under_chance, less_LN = get_filtered_cellids(df_r, df_e, gc, stp,
                                                          LN, combined,
                                                          se_filter,
                                                          LN_filter)

    cellids = cellids
    if only_improvements:
        e, a, g, s, c = improved_cells_to_list(batch, gc, stp, LN, combined)
        gc_cells = list((set(cellids) & (set(e) | set(g))) - set(c) - set(s))
        stp_cells = list((set(cellids) & (set(e) | set(s))) - set(c) - set(g))
        n_gc = len(gc_cells)
        n_stp = len(stp_cells)
    else:
        gc_cells = stp_cells = cellids
        n_gc = n_stp = len(cellids)

    if plot_stat == 'r_ceiling':
        plot_df = df_c
    else:
        plot_df = df_r

    if cf_load_path is None:
        df = cf_batch(batch, LN, load_path=cf_load_path, **cf_kwargs)
    else:
        df = pd.read_pickle(cf_load_path)

    gc_cfs = df['cf'][gc_cells].values.astype('float32')
    stp_cfs = df['cf'][stp_cells].values.astype('float32')
    #ln_test = plot_df[LN][cellids].values.astype('float32')
    gc_test = plot_df[gc][gc_cells].values.astype('float32')
    stp_test = plot_df[stp][stp_cells].values.astype('float32')
    #combined_test = plot_df[combined][cellids].values.astype('float32')

    r_gc, p_gc = st.spearmanr(gc_cfs, gc_test)
    r_stp, p_stp = st.spearmanr(stp_cfs, stp_test)

    plt.figure()
#    if include_LN:
#        plt.scatter(cfs, ln_test, color='gray', alpha=0.5)
    plt.scatter(gc_cfs, gc_test, color='goldenrod', alpha=0.5)
    plt.scatter(stp_cfs, stp_test, color=wsu_crimson, alpha=0.5)
#    if include_combined:
#        plt.scatter(cfs, combined_test, color='purple', alpha=0.5)
    plt.xscale('log', basex=np.e)

    title = ("CF vs model performance\n"
             "gc -- rho:  %.4f, p:  %.4E, n:  %d\n"
             "stp -- rho:  %.4f, p:  %.4E, n:  %d"
             % (r_gc, p_gc, n_gc, r_stp, p_stp, n_stp))
    plt.title(title)


def get_strf(cellid, batch, modelname):
    xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname,
                                         eval_model=False)
    modelspec = ctx['modelspec']
    wc_coefs = _get_wc_coefficients(modelspec, idx=0)
    fir_coefs = _get_fir_coefficients(modelspec, idx=0)
    strf = wc_coefs.T @ fir_coefs

    return strf
