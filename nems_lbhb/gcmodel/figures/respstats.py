import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nems.xform_helper as xhelp
import nems.db as nd
from nems_lbhb.gcmodel.figures.utils import (get_filtered_cellids,
                                             get_dataframes,
                                             adjustFigAspect)
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems.recording import load_recording
from nems.preprocessing import average_away_epoch_occurrences

# TODO: analyses for separating model fits by:
#       max firing rate
#       spont rate
#       characteristic frequency

def max_rate_by_batch(batch, sampling_rate=100):
    cells = nd.get_batch_cells(batch, as_list=True)
    maxes = []
    for cellid in cells:
        loadkey = 'ozgf.fs%d.ch18' % sampling_rate
        recording_uri = generate_recording_uri(cellid=cellid, batch=batch,
                                               loadkey=loadkey, stim=False)
        rec = load_recording(recording_uri)
        rec['resp'] = rec['resp'].rasterize()
        avgrec = average_away_epoch_occurrences(rec)
        resp = avgrec['resp'].extract_channels([cellid]).as_continuous().flatten()
        #resp = rec['resp'].extract_channels([cellid])
        #avgresp = generate_average_sig(resp)

        raw_max = np.nanmax(resp)
        mean_3sd = np.nanmean(resp) + 3*np.nanstd(resp)
        max_rate = min(raw_max, mean_3sd)
        maxes.append(max_rate)

    results = {'cellid': cells, 'max_rate': maxes}
    df = pd.DataFrame.from_dict(results)
    df.set_index('cellid', inplace=True)

    return df


def max_rate_vs_performance(batch, gc, stp, LN, combined, se_filter=True,
                            LN_filter=False, plot_stat='r_ceiling'):
    df_r, df_c, df_e = get_dataframes(batch, gc, stp, LN, combined)
    cellids, under_chance, less_LN = get_filtered_cellids(df_r, df_e, gc, stp,
                                                          LN, combined,
                                                          se_filter,
                                                          LN_filter)

    if plot_stat == 'r_ceiling':
        plot_df = df_c
    else:
        plot_df = df_r

    max_rates = max_rate_by_batch(batch)
    gc_test = plot_df[gc][cellids].values.astype('float32')
    stp_test = plot_df[stp][cellids].values.astype('float32')
    max_rates = max_rates[cellids].values.astype('float32')

    y_max = np.max(np.maximum([stp_test, gc_test]))
    y_min = np.min(np.minimum([stp_test, gc_test]))
    x_max = np.max(max_rates)
    x_min = np.min(max_rates)
    abs_max = max(np.abs(y_max), np.abs(x_max), np.abs(y_min), np.abs(x_min))
    abs_max *= 1.15

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.plot([0, 0], [1, 1], linewidth=2, linestyle='dashed',
            dashes=dash_spacing)
    ax.scatter(max_rates, gc_test, color='goldenrod', alpha=0.5, s=20,
               label='GC')
    ax.scatter(max_rates, stp_test, color=wsu_crimson, alpha=0.5, s=20,
               label='STP')
    ax.legend()
    ax.set_ylim(ymin=(-1)*abs_max, ymax=abs_max)
    ax.set_xlim(xmin=(-1)*abs_max, xmax=abs_max)
    adjustFigAspect(fig, aspect=1)


def strf_vs_resp_by_contrast(cellid, batch, modelname, ax=None,
                             plot_stim=False, plot_contrast=False,
                             plot_kwargs=None):
    xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)
    val = ctx['val'].apply_mask()
    pred = val['pred'].as_continuous().flatten()
    resp = val['resp'].as_continuous().flatten()
    stim = val['stim'].as_continuous()
    contrast = val['contrast'].as_continuous()
    summed_contrast = np.sum(contrast, axis=0)

    #median_contrast = np.median(summed_contrast)
    median_contrast = np.percentile(summed_contrast, 50)
    first_quartile = np.percentile(summed_contrast, 25)
    third_quartile = np.percentile(summed_contrast, 75)
    lower_contrast_mask = summed_contrast < first_quartile
    low_contrast_mask = ((summed_contrast >= first_quartile)
                         & (summed_contrast < median_contrast))
    med_contrast_mask = ((summed_contrast >= median_contrast)
                         & (summed_contrast < third_quartile))
    high_contrast_mask = (summed_contrast >= third_quartile)

    if ax is None:
        plt.figure(figsize=figsize)
    else:
        plt.sca(ax)
    if plot_kwargs is None:
        plot_kwargs = {'alpha': 0.5}
    plt.scatter(pred[lower_contrast_mask], resp[lower_contrast_mask],
                color='gray', **plot_kwargs)
    plt.scatter(pred[low_contrast_mask], resp[low_contrast_mask],
                color='blue', **plot_kwargs)
    plt.scatter(pred[med_contrast_mask], resp[med_contrast_mask],
                color='#B089E1', **plot_kwargs)
    plt.scatter(pred[high_contrast_mask], resp[high_contrast_mask],
                color='red', **plot_kwargs)
    plt.xlabel('linear model prediction')
    plt.ylabel('actual response')
    plt.legend(['lower', 'low', 'medium', 'high'])
    plt.title('cellid:  %s\nbatch:  %s\nmodel:  %s' % (cellid, batch, modelname))

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


def strf_vs_resp_batch(batch, modelname, save_path, test_limit=None):
    cells = nd.get_batch_cells(batch, as_list=True)
    #plot_kwargs = {'alpha': 0.2, 's': 2}
    for cellid in cells[:test_limit]:
        try:
            fig, ax = plt.subplots(1,1, figsize=figsize)
            strf_vs_resp_by_contrast(cellid, batch, modelname, ax=ax,
                                     plot_stim=False, plot_contrast=False)
            full_path = os.path.join(save_path, batch, cellid)
            fig.savefig(full_path, format='pdf')
            plt.close(fig)
        except:
            # cell probably not fit for this model or batch
            print('error for cell: %s' % cellid)
            continue
