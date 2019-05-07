#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:05:34 2018

@author: svd
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image

import nems.plots.api as nplt
import nems.xforms as xforms
import nems.xform_helper as xhelp
import nems.modelspec as ms
import nems.epoch as ep
import nems.preprocessing as preproc
from nems.metrics.state import state_mod_index
from nems.utils import find_module
import nems_lbhb.plots as lplt

font_size=8
params = {'legend.fontsize': font_size-2,
          'figure.figsize': (8, 6),
          'axes.labelsize': font_size,
          'axes.titlesize': font_size,
          'xtick.labelsize': font_size,
          'ytick.labelsize': font_size,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)

line_colors = {'actual_psth': (0,0,0),
               'predicted_psth': 'red',
               #'passive': (255/255, 133/255, 133/255),
               'passive': (216/255, 151/255, 212/255),
               #'active': (196/255, 33/255, 43/255),
               'active': (129/255, 64/255, 138/255),
               'false_alarm': (79/255, 114/255, 184/255),
               'miss': (183/255, 196/255, 229/255),
               'hit': (36/255, 49/255, 103/255),
               'pre': 'green',
               'post': (123/255, 104/255, 238/255),
               'pas1': 'green',
               'pas2': (153/255, 124/255, 248/255),
               'pas3': (173/255, 144/255, 255/255),
               'pas4': (193/255, 164/255, 255/255),
               'pas5': 'green',
               'pas6': (123/255, 104/255, 238/255),
               'hard': (196/255, 149/255, 44/255),
               'easy': (255/255, 206/255, 6/255),
               'puretone': (247/255, 223/255, 164/255),
               'large': (44/255, 125/255, 61/255),
               'small': (181/255, 211/255, 166/255)}
fill_colors = {'actual_psth': (.8,.8,.8),
               'predicted_psth': 'pink',
               #'passive': (226/255, 172/255, 185/255),
               'passive': (234/255, 176/255, 223/255),
               #'active': (244/255, 44/255, 63/255),
               'active': (163/255, 102/255, 163/255),
               'false_alarm': (107/255, 147/255, 204/255),
               'miss': (200/255, 214/255, 237/255),
               'hit': (78/255, 92/255, 135/255),
               'pre': 'green',
               'post': (123/255, 104/255, 238/255),
               'hard':  (229/255, 172/255, 57/255),
               'easy': (255/255, 225/255, 100/255),
               'puretone': (255/255, 231/255, 179/255),
               'large': (69/255, 191/255, 89/255),
               'small': (215/255, 242/255, 199/255)}


def beta_comp(beta1, beta2, n1='model1', n2='model2', hist_bins=20,
              hist_range=[-1, 1], title=None,
              highlight=None, ax=None, click_fun=None):
    """
    beta1, beta2 are T x 1 vectors
    scatter plot comparing beta1 vs. beta2
    histograms of marginals
    """

    beta1 = np.array(beta1).astype(float)
    beta2 = np.array(beta2).astype(float)

    nncells = np.isfinite(beta1) & np.isfinite(beta2)
    beta1 = beta1[nncells]
    beta2 = beta2[nncells]
    if highlight is not None:
        highlight = np.array(highlight).astype(float)
        highlight = highlight[nncells]

    if title is None:
        title = "{} v {}".format(n1,n2)

    if highlight is not None:
        title += " (n={}/{})".format(np.sum(highlight),len(highlight))

    # exclude cells without prepassive
    outcells = ((beta1 > hist_range[1]) | (beta1 < hist_range[0]) |
                (beta2 > hist_range[1]) | (beta2 < hist_range[0]))
    goodcells = (np.abs(beta1) > 0) | (np.abs(beta2) > 0)

    beta1[beta1 > hist_range[1]] = hist_range[1]
    beta1[beta1 < hist_range[0]] = hist_range[0]
    beta2[beta2 > hist_range[1]] = hist_range[1]
    beta2[beta2 < hist_range[0]] = hist_range[0]

    if highlight is None:
        set1 = goodcells.astype(bool)
        set2 = (1-goodcells).astype(bool)
    else:
        highlight = np.array(highlight)
        set1 = np.logical_and(goodcells, (highlight))
        set2 = np.logical_and(goodcells, (1-highlight))

    if ax is None:
        fh = plt.figure(figsize=(8, 6))

        ax = plt.subplot(2, 2, 3)
        exit_after_scatter=False
    else:
        plt.sca(ax)

        fh = plt.gcf()
        exit_after_scatter=True

    ##plt.axvline(0, color='k', linestyle='--', linewidth=0.5)
    #plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    #plt.plot(np.array(hist_range), np.array(hist_range), 'k--', linewidth=0.5)
    hist_range = np.array(hist_range)
    zz = np.zeros(2)

    outi = np.where(outcells)
    set2i = np.where(set2)
    set1i = np.where(set1)
    evs = set1i[0]

    ax.plot(hist_range, zz, 'k--',linewidth=0.5)
    ax.plot(zz, hist_range, 'k--',linewidth=0.5)
    ax.plot(hist_range,hist_range, 'k--',linewidth=0.5)

    ax.plot(beta1[set2], beta2[set2], '.', color='lightgray', markersize=6,
            markeredgecolor='white', markeredgewidth=0.5)
    ax.plot(beta1[set1], beta2[set1], 'k.', picker=5, markersize=10,
            markeredgecolor='white', markeredgewidth=0.5)
    ax.plot(beta1[outcells], beta2[outcells], '.', color='red', markeredgecolor='white',
            markeredgewidth=0.5, markersize=10)

    ax.set_aspect('equal', 'box')
    #plt.ylim(hist_range)
    #plt.xlim(hist_range)

    plt.xlabel("{} (m={:.3f})".format(n1, np.mean(beta1[goodcells])))
    plt.ylabel("{} (m={:.3f})".format(n2, np.mean(beta2[goodcells])))
    plt.title(title)
    nplt.ax_remove_box(ax)

    if click_fun is not None:
        def display_wrapper(event):
            i = evs[int(event.ind[0])]
            click_fun(i)

        fh.canvas.mpl_connect('pick_event', display_wrapper)


    if exit_after_scatter:
        return plt.gcf()

    ax = plt.subplot(2, 2, 1)
    plt.hist([beta1[set1], beta1[set2]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True,
             color=['black','lightgray'])
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta1[goodcells]),
                     np.mean(np.abs(beta1[goodcells]))))
    plt.xlabel(n1)
    nplt.ax_remove_box(ax)

    ax = plt.subplot(2, 2, 4)
    plt.hist([beta2[set1], beta2[set2]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True, orientation="horizontal",
             color=['black','lightgray'])
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta2[goodcells]),
                     np.mean(np.abs(beta2[goodcells]))))
    plt.xlabel(n2)
    nplt.ax_remove_box(ax)

    ax = plt.subplot(2, 2, 2)
#    plt.hist([(beta2[set1]-beta1[set1]) * np.sign(beta2[set1]),
#              beta2[set2]-beta1[set2] * np.sign(beta2[set2])],
#             bins=hist_bins-1, range=[hist_range[0]/2,hist_range[1]/2],
#             histtype='bar', stacked=True,
#             color=['black','lightgray'])

    # d = np.sort(np.sign(beta1[goodcells])*(beta2[goodcells]-beta1[goodcells]))
    d = np.sort(beta2[goodcells] - beta1[goodcells])
    plt.bar(np.arange(np.sum(goodcells)), d,
            color='black')
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta2[goodcells]),
                     np.mean(np.abs(beta2[goodcells]))))
    plt.ylabel('difference')

    plt.tight_layout()
    nplt.ax_remove_box(ax)

    old_title=fh.canvas.get_window_title()
    fh.canvas.set_window_title(old_title+': '+title)

    return fh


def display_png(event, cellids, path):
    ind = event.ind
    if len(ind) > 1:
        ind = [ind[0]]
    else:
        ind = ind
    cell1 = cellids[ind]
    print('cell1: {0}'.format(cell1))
    print(ind)
    # img = mpimg.imread(path+'/'+cell1[0]+'.png')
    # img = plt.imread(path+'/'+cell1[0]+'.png')
    img = Image.open(path+'/'+cell1[0]+'.png')
    img.show(img)


def beta_comp_from_folder(beta1='r_pup', beta2='r_beh',
                          n1='model1', n2='model2', hist_bins=20,
                          hist_range=[-1, 1], title='modelname/batch',
                          folder=None, highlight=None):

    if folder is None:
        raise ValueError('Must specify the results folder!')
    elif folder[-1] == '/':
        folder = folder[:-1]

    if highlight is not None:
        highlight = np.array(highlight)

    results = pd.read_csv(folder+'/results.csv')
    cellids = results['cellid'].values

    beta1 = results[beta1].values
    beta2 = results[beta2].values

    nncells = np.isfinite(beta1) & np.isfinite(beta2)
    beta1 = beta1[nncells]
    beta2 = beta2[nncells]

    # exclude cells without prepassive
    outcells = ((beta1 > hist_range[1]) | (beta1 < hist_range[0]) |
                (beta2 > hist_range[1]) | (beta2 < hist_range[0]))
    goodcells = (np.abs(beta1) > 0) | (np.abs(beta2) > 0)

    beta1[beta1 > hist_range[1]] = hist_range[1]
    beta1[beta1 < hist_range[0]] = hist_range[0]
    beta2[beta2 > hist_range[1]] = hist_range[1]
    beta2[beta2 < hist_range[0]] = hist_range[0]

    if highlight is None:
        set1 = goodcells.astype(bool)
        set2 = (1-goodcells).astype(bool)
    else:
        highlight = np.array(highlight)
        set1 = np.logical_and(goodcells, (highlight))
        set2 = np.logical_and(goodcells, (1-highlight))


    fh = plt.figure(figsize=(6, 6))

    plt.subplot(2, 2, 3)
    plt.plot(np.array(hist_range), np.array([0, 0]), 'k--')
    plt.plot(np.array([0, 0]), np.array(hist_range), 'k--')
    plt.plot(np.array(hist_range), np.array(hist_range), 'k--')
    plt.plot(beta1[set1], beta2[set1], 'k.', picker=3)
    plt.plot(beta1[set2], beta2[set2], '.', color='lightgray', picker=3)
    plt.plot(beta1[outcells], beta2[outcells], '.', color='red')
    plt.axis('equal')
    plt.axis('tight')

    plt.xlabel(n1)
    plt.ylabel(n2)
    plt.title(title)

    def display_wrapper(event):

        if sum(set2)==0:
            display_png(event, cellids[set1], folder)
        elif event.mouseevent.button==1:
            print("Left-click detected, displaying from 'highlighted' cells")
            display_png(event, cellids[set1], folder)
        elif event.mouseevent.button==3:
            print("Right-click detected, loading from 'non-highlighted' cells")
            display_png(event, cellids[set2], folder)

    fh.canvas.mpl_connect('pick_event', lambda event: display_wrapper(event))


    ax = plt.subplot(2, 2, 1)
    plt.hist([beta1[set1], beta1[set2]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True,
             color=['black','lightgray'])
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta1[goodcells]),
                     np.mean(np.abs(beta1[goodcells]))))
    plt.xlabel(n1)
    nplt.ax_remove_box(ax)

    ax = plt.subplot(2, 2, 4)
    plt.hist([beta2[set1], beta2[set2]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True, orientation="horizontal",
             color=['black','lightgray'])
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta2[goodcells]),
                     np.mean(np.abs(beta2[goodcells]))))
    plt.xlabel(n2)
    nplt.ax_remove_box(ax)

    ax = plt.subplot(2, 2, 2)
    plt.hist([(beta2[set1]-beta1[set1]) * np.sign(beta2[set1]),
              beta2[set2]-beta1[set2] * np.sign(beta2[set2])],
             bins=hist_bins-1, range=[hist_range[0]/2,hist_range[1]/2],
             histtype='bar', stacked=True,
             color=['black','lightgray'])

#    d=np.sort(np.sign(beta1[goodcells])*(beta2[goodcells]-beta1[goodcells]))
#    plt.bar(np.arange(np.sum(goodcells)), d,
#            color='black')
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta2[goodcells]),
                     np.mean(np.abs(beta2[goodcells]))))
    plt.ylabel('difference')

    plt.tight_layout()
    nplt.ax_remove_box(ax)

    old_title=fh.canvas.get_window_title()
    fh.canvas.set_window_title(old_title+': '+title)

    return fh


def beta_comp_cols(g, b, n1='A', n2='B', hist_bins=20,
                  hist_range=[-1,1], title='modelname/batch',
                  highlight=None):

    #exclude cells without prepassive
    goodcells=(np.abs(g[:,0]) > 0) & (np.abs(g[:,1])>0)

    if highlight is None:
        set1=goodcells
        set2=goodcells * 0
    else:
        set1 = np.logical_and(goodcells, (highlight))
        set2 = np.logical_and(goodcells, (1-highlight))

    plt.figure(figsize=(6,8))

    plt.subplot(3, 2, 1)
    plt.plot(np.array(hist_range), np.array([0, 0]), 'k--')
    plt.plot(np.array([0, 0]), np.array(hist_range), 'k--')
    plt.plot(g[set1, 0], g[set1, 1], 'k.')
    plt.plot(g[set2, 0], g[set2, 1], 'b.')
    plt.axis('equal')
    plt.axis('tight')

    plt.xlabel(n1)
    plt.ylabel(n2)
    plt.title(title)

    plt.subplot(3, 2, 3)
    plt.hist(g[goodcells,0],bins=hist_bins,range=hist_range)
    plt.title('mean={:.3f}'.format(np.mean(g[goodcells,0])))
    plt.xlabel(n1)

    plt.subplot(3, 2, 5)
    plt.hist(g[goodcells,1],bins=hist_bins,range=hist_range)
    plt.title('mean={:.3f}'.format(np.mean(g[goodcells,1])))
    plt.xlabel(n2)

    plt.subplot(3, 2, 2)
    plt.plot(np.array(hist_range), np.array([0, 0]), 'k--')
    plt.plot(np.array([0, 0]), np.array(hist_range), 'k--')
    plt.plot(b[set1, 0], b[set1, 1], 'k.')
    plt.plot(b[set2, 0], b[set2, 1], 'b.')
    plt.axis('equal')
    plt.axis('tight')

    plt.xlabel(n1)
    plt.ylabel(n2)
    plt.title('baseline')

    plt.subplot(3, 2, 4)
    plt.hist(b[goodcells, 0], bins=hist_bins, range=hist_range)
    plt.title('mean={:.3f}'.format(np.mean(b[goodcells, 0])))
    plt.xlabel(n1)

    plt.subplot(3, 2, 6)
    plt.hist(b[goodcells, 1], bins=hist_bins, range=hist_range)
    plt.xlabel(n2)
    plt.title('mean={:.3f}'.format(np.mean(b[goodcells, 1])))
    plt.tight_layout()


def model_split_psths(cellid, batch, modelname, state1 = 'pupil',
                      state2 = 'active', epoch='REFERENCE', state_colors=None,
                      psth_name = 'resp'):
    """
    state_colors : N x 2 list
       color spec for high/low lines in each of the N states
    """
    global line_colors
    global fill_colors

    xf, ctx = xhelp.load_model_xform(cellid, batch, modelname)

    rec = ctx['val'][0].apply_mask()
    fs = rec[psth_name].fs
    state_sig = 'state_raw'

    d = rec[psth_name].get_epoch_bounds('PreStimSilence')
    PreStimSilence = np.mean(np.diff(d)) - 0.5/fs
    d = rec[psth_name].get_epoch_bounds('PostStimSilence')
    if d.size > 0:
        PostStimSilence = np.min(np.diff(d)) - 0.5/fs
        dd = np.diff(d)
        dd = dd[dd > 0]
    else:
        dd = np.array([])
    if dd.size > 0:
        PostStimSilence = np.min(dd) - 0.5/fs
    else:
        PostStimSilence = 0

    chanidx=0
    full_psth = rec[psth_name]
    folded_psth = full_psth.extract_epoch(epoch)[:, [chanidx], :] * fs

    full_var1 = rec[state_sig].loc[state1]
    folded_var1 = np.squeeze(full_var1.extract_epoch(epoch))
    full_var2 = rec[state_sig].loc[state2]
    folded_var2 = np.squeeze(full_var2.extract_epoch(epoch))

    # compute the mean state for each occurrence
    g2 = (np.sum(np.isfinite(folded_var2), axis=1) > 0)
    m2 = np.zeros_like(g2, dtype=float)
    m2[g2] = np.nanmean(folded_var2[g2, :], axis=1)
    mean2 = np.nanmean(m2)
    gtidx2 = (m2 >= mean2) & g2
    ltidx2 = np.logical_not(gtidx2) & g2

    # compute the mean state for each occurrence
    g1 = (np.sum(np.isfinite(folded_var1), axis=1) > 0)
    m1 = np.zeros_like(g1, dtype=float)
    m1[g1] = np.nanmean(folded_var1[g1, :], axis=1)

    mean1 = np.nanmean(m1[gtidx2])
    std1 = np.nanstd(m1[gtidx2])

    gtidx1 = (m1 >= mean1-std1*3) & (m1 <= mean1+std1*1) & g1
    # ltidx1 = np.logical_not(gtidx1) & g1

    # highlow = response on epochs when state1 high and state2 low
    if (np.sum(ltidx2) == 0):
        low = np.zeros_like(folded_psth[0, :, :].T) * np.nan
        highlow = np.zeros_like(folded_psth[0, :, :].T) * np.nan
    else:
        low = np.nanmean(folded_psth[ltidx2, :, :], axis=0).T
        highlow = np.nanmean(folded_psth[gtidx1 & ltidx2, :, :], axis=0).T

    # highhigh = response on epochs when state high and state2 high
    if (np.sum(gtidx2) == 0):
        high = np.zeros_like(folded_psth[0, :, :].T) * np.nan
        highhigh = np.zeros_like(folded_psth[0, :, :].T) * np.nan
    else:
        high = np.nanmean(folded_psth[gtidx2, :, :], axis=0).T
        highhigh = np.nanmean(folded_psth[gtidx1 & gtidx2, :, :], axis=0).T

    legend = ('Lo', 'Hi')

    plt.figure()
    ax = plt.subplot(2,1,1)
    plt.plot(m1)
    plt.plot(m2)
    plt.plot(gtidx1+1.1)
    plt.legend((state1,state2,state1 + ' matched'))

    ax = plt.subplot(2,2,3)
    title = "{} all/ {}".format(state1, state2)
    nplt.timeseries_from_vectors([low, high], fs=fs, title=title, ax=ax,
                            legend=legend, time_offset=PreStimSilence,
                            ylabel="sp/sec")
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.plot(np.array([0, 0]), ylim, 'k--')
    ax.plot(np.array([xlim[1], xlim[1]])-PostStimSilence, ylim, 'k--')

    ax = plt.subplot(2,2,4)
    title = "{} matched/ {}".format(state1, state2)
    nplt.timeseries_from_vectors([highlow, highhigh], fs=fs, title=title, ax=ax,
                            legend=legend, time_offset=PreStimSilence,
                            ylabel="sp/sec")
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.plot(np.array([0, 0]), ylim, 'k--')
    ax.plot(np.array([xlim[1], xlim[1]])-PostStimSilence, ylim, 'k--')

    plt.tight_layout()


def model_per_time_wrapper(cellid, batch=307,
                           loader= "psth.fs20.pup-ld-",
                           fitter = "_jk.nf20-basic",
                           basemodel = "-ref-psthfr_stategain.S",
                           state_list=None, plot_halves=True,
                           colors=None):
    """
    batch = 307  # A1 SUA and MUA
    batch = 309  # IC SUA and MUA

    alternatives:
        basemodels = ["-ref-psthfr.s_stategain.S",
                      "-ref-psthfr.s_sdexp.S",
                      "-ref.a-psthfr.s_sdexp.S"]
        state_list = ['st.pup0.hlf0','st.pup0.hlf','st.pup.hlf0','st.pup.hlf']
        state_list = ['st.pup0.far0.hit0.hlf0','st.pup0.far0.hit0.hlf',
                      'st.pup.far.hit.hlf0','st.pup.far.hit.hlf']
        state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']
        
    """

    # pup vs. active/passive
    if state_list is None:
        state_list = ['st.pup0.hlf0','st.pup0.hlf','st.pup.hlf0','st.pup.hlf']
        #state_list = ['st.pup0.far0.hit0.hlf0','st.pup0.far0.hit0.hlf',
        #              'st.pup.far.hit.hlf0','st.pup.far.hit.hlf']
        #state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']

    modelnames = []
    contexts = []
    for i, s in enumerate(state_list):
        modelnames.append(loader + s + basemodel + fitter)

        xf, ctx = xhelp.load_model_xform(cellid, batch, modelnames[i],
                                         eval_model=False)
        ctx, l = xforms.evaluate(xf, ctx, start=0, stop=-2)

        ctx['val'] = preproc.make_state_signal(
            ctx['val'], state_signals=['each_half'], new_signalname='state_f')

        contexts.append(ctx)
        #import pdb;
        #pdb.set_trace()

    plt.figure()
    #if ('hlf' in state_list[0]) or ('fil' in state_list[0]):
    if plot_halves:
        files_only=True
    else:
        files_only=False
        
    for i, ctx in enumerate(contexts):

        rec = ctx['val'].apply_mask()
        modelspec = ctx['modelspec']
        epoch="REFERENCE"
        rec = ms.evaluate(rec, modelspec)
        if i == len(contexts)-1:
            ax = plt.subplot(len(contexts)+1, 1, 1)
            nplt.state_vars_timeseries(rec, modelspec, ax=ax)
            ax.set_title('{} {}'.format(cellid, modelnames[-1]))

        ax = plt.subplot(len(contexts)+1, 1, 2+i)
        nplt.state_vars_psth_all(rec, epoch, psth_name='resp',
                            psth_name2='pred', state_sig='state_f',
                            colors=colors, channel=None, decimate_by=1,
                            ax=ax, files_only=files_only, modelspec=modelspec)
        ax.set_ylabel(state_list[i])
        ax.set_xticks([])

    #plt.tight_layout()
    

def _model_step_plot(cellid, batch, modelnames, factors, state_colors=None):
    """
    state_colors : N x 2 list
       color spec for high/low lines in each of the N states
    """
    global line_colors
    global fill_colors

    modelname_p0b0, modelname_p0b, modelname_pb0, modelname_pb = \
       modelnames
    factor0, factor1, factor2 = factors

    xf_p0b0, ctx_p0b0 = xhelp.load_model_xform(cellid, batch, modelname_p0b0,
                                                  eval_model=False)
    # ctx_p0b0, l = xforms.evaluate(xf_p0b0, ctx_p0b0, stop=-2)

    ctx_p0b0, l = xforms.evaluate(xf_p0b0, ctx_p0b0, start=0, stop=-2)

    xf_p0b, ctx_p0b = xhelp.load_model_xform(cellid, batch, modelname_p0b,
                                                eval_model=False)
    ctx_p0b, l = xforms.evaluate(xf_p0b, ctx_p0b, start=0, stop=-2)

    xf_pb0, ctx_pb0 = xhelp.load_model_xform(cellid, batch, modelname_pb0,
                                                eval_model=False)
    #ctx_pb0['rec'] = ctx_p0b0['rec'].copy()
    ctx_pb0, l = xforms.evaluate(xf_pb0, ctx_pb0, start=0, stop=-2)

    xf_pb, ctx_pb = xhelp.load_model_xform(cellid, batch, modelname_pb,
                                              eval_model=False)
    #ctx_pb['rec'] = ctx_p0b0['rec'].copy()
    ctx_pb, l = xforms.evaluate(xf_pb, ctx_pb, start=0, stop=-2)

    # organize predictions by different models
    val = ctx_pb['val'][0].copy()

    # val['pred_p0b0'] = ctx_p0b0['val'][0]['pred'].copy()
    val['pred_p0b'] = ctx_p0b['val'][0]['pred'].copy()
    val['pred_pb0'] = ctx_pb0['val'][0]['pred'].copy()

    state_var_list = val['state'].chans

    pred_mod = np.zeros([len(state_var_list), 2])
    pred_mod_full = np.zeros([len(state_var_list), 2])
    resp_mod_full = np.zeros([len(state_var_list), 1])

    state_std = np.nanstd(val['state'].as_continuous(), axis=1, keepdims=True)
    for i, var in enumerate(state_var_list):
        if state_std[i]:
            # actual response modulation index for each state var
            resp_mod_full[i] = state_mod_index(val, epoch='REFERENCE',
                                               psth_name='resp', state_chan=var)

            mod2_p0b = state_mod_index(val, epoch='REFERENCE',
                                       psth_name='pred_p0b', state_chan=var)
            mod2_pb0 = state_mod_index(val, epoch='REFERENCE',
                                       psth_name='pred_pb0', state_chan=var)
            mod2_pb = state_mod_index(val, epoch='REFERENCE',
                                      psth_name='pred', state_chan=var)

            pred_mod[i] = np.array([mod2_pb-mod2_p0b, mod2_pb-mod2_pb0])
            pred_mod_full[i] = np.array([mod2_pb0, mod2_p0b])

    pred_mod_norm = pred_mod / (state_std + (state_std == 0).astype(float))
    pred_mod_full_norm = pred_mod_full / (state_std +
                                          (state_std == 0).astype(float))

    if 'each_passive' in factors:
        psth_names_ctl = ["pred_p0b"]
        factors.remove('each_passive')
        for v in state_var_list:
            if v.startswith('FILE_'):
                factors.append(v)
                psth_names_ctl.append("pred_pb0")
    else:
        psth_names_ctl = ["pred_p0b", "pred_pb0"]

    col_count = len(factors) - 1
    if state_colors is None:
        state_colors = [[None, None]]*col_count

    fh = plt.figure(figsize=(8,8))
    ax = plt.subplot(4, 1, 1)
    nplt.state_vars_timeseries(val, ctx_pb['modelspecs'][0],
                               state_colors=[s[1] for s in state_colors])
    ax.set_title("{}/{} - {}".format(cellid, batch, modelname_pb))
    ax.set_ylabel("{} r={:.3f}".format(factor0,
                  ctx_p0b0['modelspecs'][0][0]['meta']['r_test'][0]))
    nplt.ax_remove_box(ax)

    for i, var in enumerate(factors[1:]):
        if var.startswith('FILE_'):
           varlbl = var[5:]
        else:
           varlbl = var
        ax = plt.subplot(4, col_count, col_count+i+1)

        nplt.state_var_psth_from_epoch(val, epoch="REFERENCE",
                                       psth_name="resp",
                                       psth_name2=psth_names_ctl[i],
                                       state_chan=var, ax=ax,
                                       colors=state_colors[i])
        if i == 0:
            ax.set_ylabel("Control model")
            ax.set_title("{} ctl r={:.3f}"
                         .format(varlbl.lower(),
                                 ctx_p0b['modelspecs'][0][0]['meta']['r_test'][0]),
                         fontsize=6)
        else:
            ax.yaxis.label.set_visible(False)
            ax.set_title("{} ctl r={:.3f}"
                         .format(varlbl.lower(),
                                 ctx_pb0['modelspecs'][0][0]['meta']['r_test'][0]),
                         fontsize=6)
        if ax.legend_:
            ax.legend_.remove()
        ax.xaxis.label.set_visible(False)
        nplt.ax_remove_box(ax)

        ax = plt.subplot(4, col_count, col_count*2+i+1)
        nplt.state_var_psth_from_epoch(val, epoch="REFERENCE",
                                       psth_name="resp",
                                       psth_name2="pred",
                                       state_chan=var, ax=ax,
                                       colors=state_colors[i])
        if i == 0:
            ax.set_ylabel("Full Model")
        else:
            ax.yaxis.label.set_visible(False)
        if ax.legend_:
            ax.legend_.remove()

        if psth_names_ctl[i] == "pred_p0b":
            j=0
        else:
            j=1

        ax.set_title("r={:.3f} rawmod={:.3f} umod={:.3f}"
                     .format(ctx_pb['modelspecs'][0][0]['meta']['r_test'][0],
                             pred_mod_full_norm[i+1][j], pred_mod_norm[i+1][j]),
                     fontsize=6)

        if var == 'active':
            ax.legend(('pas', 'act'))
        elif var == 'pupil':
            ax.legend(('small', 'large'))
        elif var == 'PRE_PASSIVE':
            ax.legend(('act+post', 'pre'))
        elif var.startswith('FILE_'):
            ax.legend(('this', 'others'))
        nplt.ax_remove_box(ax)

    # EXTRA PANELS
    # figure out some basic aspects of tuning/selectivity for target vs.
    # reference:
    r = ctx_pb['rec']['resp']
    e = r.epochs
    fs = r.fs

    passive_epochs = r.get_epoch_indices("PASSIVE_EXPERIMENT")
    tar_names = ep.epoch_names_matching(e, "^TAR_")
    tar_resp={}
    for tarname in tar_names:
        t = r.get_epoch_indices(tarname)
        t = ep.epoch_intersection(t, passive_epochs)
        tar_resp[tarname] = r.extract_epoch(t) * fs

    # only plot tar responses with max SNR or probe SNR
    keys=[]
    for k in list(tar_resp.keys()):
        if k.endswith('0') | k.endswith('2'):
            keys.append(k)
    keys.sort()

    # assume the reference with most reps is the one overlapping the target
    groups = ep.group_epochs_by_occurrence_counts(e, '^STIM_')
    l = np.array(list(groups.keys()))
    hi = np.max(l)
    ref_name = groups[hi][0]
    t = r.get_epoch_indices(ref_name)
    t = ep.epoch_intersection(t, passive_epochs)
    ref_resp = r.extract_epoch(t) * fs

    t = r.get_epoch_indices('REFERENCE')
    t = ep.epoch_intersection(t, passive_epochs)
    all_ref_resp = r.extract_epoch(t) * fs

    prestimsilence = r.get_epoch_indices('PreStimSilence')
    prebins=prestimsilence[0,1]-prestimsilence[0,0]
    poststimsilence = r.get_epoch_indices('PostStimSilence')
    postbins=poststimsilence[0,1]-poststimsilence[0,0]
    durbins = ref_resp.shape[-1] - prebins

    spont = np.nanmean(all_ref_resp[:,0,:prebins])
    ref_mean = np.nanmean(ref_resp[:,0,prebins:durbins])-spont
    all_ref_mean = np.nanmean(all_ref_resp[:,0,prebins:durbins])-spont
    #print(spont)
    #print(np.nanmean(ref_resp[:,0,prebins:-postbins]))
    ax1=plt.subplot(4, 2, 7)
    ref_psth = [np.nanmean(ref_resp[:, 0, :], axis=0),
                np.nanmean(all_ref_resp[:, 0, :], axis=0)]
    ll = ["{} {:.1f}".format(ref_name, ref_mean),
          "all refs {:.1f}".format(all_ref_mean)]
    nplt.timeseries_from_vectors(ref_psth, fs=fs, legend=ll, ax=ax1,
                                 time_offset=prebins/fs)

    ax2=plt.subplot(4, 2, 8)
    ll = []
    tar_mean = np.zeros(np.max([2,len(keys)])) * np.nan
    tar_psth = []
    for ii, k in enumerate(keys):
        tar_psth.append(np.nanmean(tar_resp[k][:, 0, :], axis=0))
        tar_mean[ii] = np.nanmean(tar_resp[k][:, 0, prebins:durbins])-spont
        ll.append("{} {:.1f}".format(k, tar_mean[ii]))
    nplt.timeseries_from_vectors(tar_psth, fs=fs, legend=ll, ax=ax2,
                                 time_offset=prebins/fs)
    # plt.legend(ll, fontsize=6)

    ymin=np.min([ax1.get_ylim()[0], ax2.get_ylim()[0]])
    ymax=np.max([ax1.get_ylim()[1], ax2.get_ylim()[1]])
    ax1.set_ylim([ymin, ymax])
    ax2.set_ylim([ymin, ymax])
    nplt.ax_remove_box(ax1)
    nplt.ax_remove_box(ax2)

    plt.tight_layout()

    stats = {'cellid': cellid,
             'batch': batch,
             'modelnames': modelnames,
             'state_vars': state_var_list,
             'factors': factors,
             'r_test': np.array([
                     ctx_p0b0['modelspecs'][0][0]['meta']['r_test'][0],
                     ctx_p0b['modelspecs'][0][0]['meta']['r_test'][0],
                     ctx_pb0['modelspecs'][0][0]['meta']['r_test'][0],
                     ctx_pb['modelspecs'][0][0]['meta']['r_test'][0]
                     ]),
             'se_test': np.array([
                     ctx_p0b0['modelspecs'][0][0]['meta']['se_test'][0],
                     ctx_p0b['modelspecs'][0][0]['meta']['se_test'][0],
                     ctx_pb0['modelspecs'][0][0]['meta']['se_test'][0],
                     ctx_pb['modelspecs'][0][0]['meta']['se_test'][0]
                     ]),
             'r_floor': np.array([
                     ctx_p0b0['modelspecs'][0][0]['meta']['r_floor'][0],
                     ctx_p0b['modelspecs'][0][0]['meta']['r_floor'][0],
                     ctx_pb0['modelspecs'][0][0]['meta']['r_floor'][0],
                     ctx_pb['modelspecs'][0][0]['meta']['r_floor'][0]
                     ]),
             'pred_mod': pred_mod.T,
             'pred_mod_full': pred_mod_full.T,
             'pred_mod_norm': pred_mod_norm.T,
             'pred_mod_full_norm': pred_mod_full_norm.T,
             'g': np.array([
                     ctx_p0b0['modelspecs'][0][0]['phi']['g'],
                     ctx_p0b['modelspecs'][0][0]['phi']['g'],
                     ctx_pb0['modelspecs'][0][0]['phi']['g'],
                     ctx_pb['modelspecs'][0][0]['phi']['g']]),
             'b': np.array([
                     ctx_p0b0['modelspecs'][0][0]['phi']['d'],
                     ctx_p0b['modelspecs'][0][0]['phi']['d'],
                     ctx_pb0['modelspecs'][0][0]['phi']['d'],
                     ctx_pb['modelspecs'][0][0]['phi']['d']]),
             'ref_all_resp': all_ref_mean,
             'ref_common_resp': ref_mean,
             'tar_max_resp': tar_mean[0],
             'tar_probe_resp': tar_mean[1]
        }

    return fh, stats


def pb_model_plot(cellid='TAR010c-06-1', batch=301,
                  loader="psth.fs20", basemodel="stategain.S", fitter="basic.st.nf10"):
    """
    test for pupil-behavior interaction.
    loader : string
      can be 'psth' or 'psths'
    fitter : string
      can be 'basic-nf' or 'cd-nf'

    """
    global line_colors

    # modelname_p0b0 = loader + "20pup0beh0_stategain3_" + fitter
    # modelname_p0b = loader + "20pup0beh_stategain3_" + fitter
    # modelname_pb0 = loader + "20pupbeh0_stategain3_" + fitter
    # modelname_pb = loader + "20pupbeh_stategain3_" + fitter
    modelname_p0b0 = loader + "-ld-st.pup0.beh0-" + basemodel + "_" + fitter
    modelname_p0b = loader + "-ld-st.pup0.beh-" + basemodel + "_" + fitter
    modelname_pb0 = loader + "-ld-st.pup.beh0-" + basemodel + "_" + fitter
    modelname_pb = loader + "-ld-st.pup.beh-" + basemodel + "_" + fitter

    factor0 = "baseline"
    factor1 = "pupil"
    factor2 = "active"

    modelnames = [modelname_p0b0, modelname_p0b, modelname_pb0,
                  modelname_pb]
    factors = [factor0, factor1, factor2]
    state_colors = [[line_colors['small'], line_colors['large']],
                    [line_colors['passive'], line_colors['active']]]

    fh, stats = _model_step_plot(cellid, batch, modelnames, factors,
                                 state_colors=state_colors)

    return fh, stats


def bperf_model_plot(cellid='TAR010c-06-1', batch=307,
                     loader="psth.fs20.pup",
                     basemodel="ref-psthfr.s_stategain.S",
                     fitter="jk.nf10-init.st-basic"):
    """
    test for engagement-performance interaction.
    loader : string
      can be 'psth' or 'psths'
    fitter : string
      can be 'basic-nf' or 'cd-nf'

    """
    global line_colors

    modelname_p0b0 = loader + "-ld-st.pup.beh0.far0.hit0-" + basemodel + "_" + fitter
    modelname_p0b = loader + "-ld-st.pup.beh.far0.hit0-" + basemodel + "_" + fitter
    modelname_pb0 = loader + "-ld-st.pup.beh0.far.hit-" + basemodel + "_" + fitter
    modelname_pb = loader + "-ld-st.pup.beh.far.hit-" + basemodel + "_" + fitter

    factor0 = "baseline"
    factor1 = "active"
    factor2 = "hit"

    modelnames = [modelname_p0b0, modelname_p0b, modelname_pb0,
                  modelname_pb]
    factors = [factor0, factor1, factor2]
    state_colors = [[line_colors['passive'], line_colors['active']],
                    [line_colors['passive'], line_colors['active']],
                    [line_colors['passive'], line_colors['active']],
                    [line_colors['easy'], line_colors['hard']]]

    fh, stats = _model_step_plot(cellid, batch, modelnames, factors,
                                 state_colors=state_colors)

    return fh, stats


def pp_model_plot(cellid='TAR010c-06-1', batch=301,
                  loader="psth", basemodel="stategain.N", fitter="basic-nf"):
    """
    test for pre-post effects
    loader : string
      can be 'psth' or 'psths'
    fitter : string
      can be 'basic-nf' or 'cd-nf'
    """

    modelname_p0b0 = loader + "20pup0pre0beh_stategain4_" + fitter
    modelname_p0b = loader + "20pup0prebeh_stategain4_" + fitter
    modelname_pb0 = loader + "20puppre0beh_stategain4_" + fitter
    modelname_pb = loader + "20pupprebeh_stategain4_" + fitter

    factor0 = "baseline"
    factor1 = "pupil"
    factor2 = "PRE_PASSIVE"

    modelnames = [modelname_p0b0, modelname_p0b, modelname_pb0,
                  modelname_pb]
    factors = [factor0, factor1, factor2]
    state_colors = [[line_colors['small'], line_colors['large']],
                    [line_colors['pre'], line_colors['post']],
                    [line_colors['passive'], line_colors['active']]]

    fh, stats = _model_step_plot(cellid, batch, modelnames, factors,
                                 state_colors=state_colors)

    plt.tight_layout()

    return fh, stats


def ppas_model_plot(cellid='TAR010c-06-1', batch=301,
                    loader="psth.fs20", basemodel="stategain.S",
                    fitter="basic.st.nf10"):
    """
    test for pre-post effects -- passive only data
    loader : string
      can be 'psth' or 'psths'
    fitter : string
      can be 'basic-nf' or 'cd-nf'
    """

    # psth.fs20-st.pup0.pas0-pas_stategain.N_basic.st.nf10
    modelname_p0b0 = loader + "-ld-st.pup0.pas0-ref-pas-" + basemodel + "_" + fitter
    modelname_p0b = loader + "-ld-st.pup0.pas-ref-pas-" + basemodel + "_" + fitter
    modelname_pb0 = loader + "-ld-st.pup.pas0-ref-pas-" + basemodel + "_" + fitter
    modelname_pb = loader + "-ld-st.pup.pas-ref-pas-" + basemodel + "_" + fitter

    factor0 = "baseline"
    factor1 = "pupil"
    factor2 = "each_passive"

    modelnames = [modelname_p0b0, modelname_p0b, modelname_pb0,
                  modelname_pb]
    factors = [factor0, factor1, factor2]
    state_colors = [[line_colors['small'], line_colors['large']],
                    [line_colors['pas1'], line_colors['post']],
                    [line_colors['pas2'], line_colors['post']],
                    [line_colors['pas3'], line_colors['post']],
                    [line_colors['pas4'], line_colors['post']],
                    [line_colors['pas5'], line_colors['post']],
                    [line_colors['pas6'], line_colors['post']]]

    fh, stats = _model_step_plot(cellid, batch, modelnames, factors,
                                 state_colors=state_colors)

    plt.tight_layout()

    return fh, stats


def psth_per_file(rec):

    raise NotImplementedError

    resp = rec['resp'].rasterize()

    file_epochs = ep.epoch_names_matching(resp.epochs, "^FILE_")

    epoch_regex = "^STIM_"
    stim_epochs = ep.epoch_names_matching(resp.epochs, epoch_regex)

    r = []
    max_rep_id = np.zeros(len(file_epochs))
    for f in file_epochs:

        r.append(resp.as_matrix(stim_epochs, overlapping_epoch=f) * resp.fs)

    repcount = np.sum(np.isfinite(r[:, :, 0, 0]), axis=1)
    max_rep_id, = np.where(repcount == np.max(repcount))

    t = np.arange(r.shape[-1]) / resp.fs

    plt.figure()

    ax = plt.subplot(3, 1, 1)
    nplt.plot_spectrogram(s[max_rep_id[-1],0,:,:], fs=stim.fs, ax=ax,
                          title="cell {} - stim".format(cellid))

    ax = plt.subplot(3, 1, 2)
    nplt.raster(t,r[max_rep_id[-1],:,0,:], ax=ax, title='raster')

    ax = plt.subplot(3, 1, 3);
    nplt.psth_from_raster(t,r[max_rep_id[-1],:,0,:], ax=ax, title='raster',
                          ylabel='spk/s')

    plt.tight_layout()


def quick_pop_state_plot(modelspec=None, **ctx):
    
    modelname = modelspec.meta['modelname']
    cellid = modelspec.meta['cellid']
    rec = ctx['val'].apply_mask()
    stateidx = find_module('state', modelspec)
    wcidx = find_module('weight_channels', modelspec)

    g = modelspec.phi[stateidx]['g']
    d = modelspec.phi[stateidx]['d']
    w = modelspec.phi[wcidx]['coefficients'][0,:]
    s = rec['state'].as_continuous()
    s_med = np.median(s, axis=1)
    s_med[s_med==0] = 0.5
    s_med[s_med==1] = 0.5
    
    loidx = (s[1,:]<=s_med[1]) & (s[2,:]<=s_med[2])
    hi1idx=s[1,:]>s_med[1]
    hi2idx=s[2,:]>s_med[2]
    
    dm=d.copy()
    dm[:,0] = (d[:, 0] + d[:, 1] * np.mean(s[1, loidx]) + d[:, 2] * np.mean(s[2, loidx])) * w
    dm[:,1] = (d[:, 0] + d[:, 1] * np.mean(s[1, hi1idx]) + d[:, 2] * np.mean(s[2, loidx])) * w
    dm[:,2] = (d[:, 0] + d[:, 1] * np.mean(s[1, loidx]) + d[:, 2] * np.mean(s[2, hi2idx])) * w

    gm=g.copy()
    gm[:,0] = (g[:, 0] + g[:, 1] * np.mean(s[1, loidx]) + g[:, 2] * np.mean(s[2, loidx])) * w
    gm[:,1] = (g[:, 0] + g[:, 1] * np.mean(s[1, hi1idx]) + g[:, 2] * np.mean(s[2, loidx])) * w
    gm[:,2] = (g[:, 0] + g[:, 1] * np.mean(s[1, loidx]) + g[:, 2] * np.mean(s[2, hi2idx])) * w

    fh = plt.figure()
    ax = plt.subplot(3, 1, 1)
    nplt.state_vars_timeseries(rec, modelspec, ax=ax)
    ax.set_title('{} {}'.format(cellid, modelname))
    
    ax = plt.subplot(3, 1, 2)
    ax.plot(dm)
    plt.title('offset')
    plt.legend(('base','d_pup','d_act'))

    ax = plt.subplot(3, 1, 3)
    ax.plot(gm)
    plt.title('gain')
    ax.set_xticks(np.arange(len(rec['stim'].chans)))
    ax.set_xticklabels(rec['stim'].chans)

    return {}