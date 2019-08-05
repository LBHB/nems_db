import os

import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt

import nems.xform_helper as xhelp
import nems.db as nd
import nems.epoch as ep
from nems_lbhb.gcmodel.figures.utils import (get_filtered_cellids,
                                             get_dataframes,
                                             adjustFigAspect)
from nems_lbhb.gcmodel.figures.soundstats import silence_duration
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems.recording import load_recording
from nems.preprocessing import average_away_epoch_occurrences

# TODO: analyses for separating model fits by:
#       max firing rate
#       spont rate
#       characteristic frequency

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


def rate_vs_performance(batch, gc, stp, LN, combined, se_filter=True,
                        LN_filter=False, plot_stat='r_ceiling',
                        test_limit=None, normalize_rates=False, load_path=None,
                        save_path=None, rate_stat='max', fs=100,
                        relative_performance=False):
    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    cellids, under_chance, less_LN = get_filtered_cellids(df_r, df_e, gc, stp,
                                                          LN, combined,
                                                          se_filter,
                                                          LN_filter)

    cellids = cellids[:test_limit]

    if plot_stat == 'r_ceiling':
        plot_df = df_c
    else:
        plot_df = df_r

    if load_path is None:
        rates = rate_by_batch(batch, cellids, stat=rate_stat, fs=fs)
        rates = rates['rate'][cellids].values.astype('float32')
        if save_path is not None:
            np.save(save_path, rates)
    else:
        rates = np.load(load_path)

    gc_test = plot_df[gc][cellids].values.astype('float32')
    stp_test = plot_df[stp][cellids].values.astype('float32')
    if normalize_rates:
        rates /= rates.max()
    if relative_performance:
        ln_test = plot_df[LN][cellids].values.astype('float32')
        gc_test = gc_test - ln_test
        stp_test = stp_test - ln_test

    r_gc, p_gc = st.pearsonr(rates, gc_test)
    r_stp, p_stp = st.pearsonr(rates, stp_test)

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.scatter(rates, gc_test, color='goldenrod', alpha=0.75, s=20,
               label='GC')
    ax.scatter(rates, stp_test, color=wsu_crimson, alpha=0.75, s=20,
               label='STP')
    ax.legend()
    adjustFigAspect(fig, aspect=1)
    plt.xlabel("%s rate\n"
               "normalized? %s" % (rate_stat, normalize_rates))
    plt.ylabel("%s\n"
               "relative to LN?  %s" % (plot_stat, relative_performance))
    plt.title("%s vs model performance\n"
              "gc -- r:  %.4f, p:  %.4E\n"
              "stp -- r:  %.4f, p:  %.4E"
              % (rate_stat, r_gc, p_gc, r_stp, p_stp))


def strf_vs_resp_by_contrast(cellid, batch, modelname, ax=None,
                             plot_stim=False, plot_contrast=False,
                             continuous=False):
    xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)
    val = ctx['val'].apply_mask()
    pred = val['pred'].as_continuous().flatten()
    resp = val['resp'].as_continuous().flatten()
    stim = val['stim'].as_continuous()
    contrast = val['contrast'].as_continuous()
    summed_contrast = np.sum(contrast, axis=0)

    if ax is None:
        plt.figure(figsize=figsize)
    else:
        plt.sca(ax)

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

        plt.scatter(pred[lower_contrast_mask], resp[lower_contrast_mask],
                    color='gray', **plot_kwargs)
        plt.scatter(pred[low_contrast_mask], resp[low_contrast_mask],
                    color='blue', **plot_kwargs)
        plt.scatter(pred[med_contrast_mask], resp[med_contrast_mask],
                    color='#B089E1', **plot_kwargs)
        plt.scatter(pred[high_contrast_mask], resp[high_contrast_mask],
                    color='red', **plot_kwargs)
        plt.legend(['lower', 'low', 'medium', 'high'])

    plt.xlabel('linear model prediction')
    plt.ylabel('actual response')
    plt.title('cellid:  %s\nbatch:  %s\nmodel:  %s' % (cellid, batch, modelname))
    if continuous:
        plt.colorbar()

    if plot_stim:
        _, (a1, a2, a3, a4) = plt.subplots(4, 1, figsize=figsize)
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
        _, (a1, a2, a3, a4) = plt.subplots(4, 1, figsize=figsize)
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


def strf_vs_resp_batch(batch, modelname, save_path, test_limit=None,
                       continuous=False):
    cells = nd.get_batch_cells(batch, as_list=True)
    #plot_kwargs = {'alpha': 0.2, 's': 2}
    for cellid in cells[:test_limit]:
        try:
            fig, ax = plt.subplots(1,1, figsize=figsize)
            strf_vs_resp_by_contrast(cellid, batch, modelname, ax=ax,
                                     plot_stim=False, plot_contrast=False,
                                     continuous=continuous)
            full_path = os.path.join(save_path, str(batch), cellid)
            fig.savefig(full_path, format='pdf')
            plt.close(fig)
        except:
            # cell probably not fit for this model or batch
            print('error for cell: %s' % cellid)
            continue
