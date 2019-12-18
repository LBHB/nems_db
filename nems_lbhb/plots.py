#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:05:34 2018

@author: svd
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
import scipy.ndimage.filters as sf

import nems.plots.api as nplt
import nems.xforms as xforms
import nems.xform_helper as xhelp
import nems.epoch as ep
import nems.modelspec as ms
from nems.utils import (find_module)
import nems.db as nd
import nems_lbhb.old_xforms.xforms as oxf
import nems_lbhb.old_xforms.xform_helper as oxfh
from nems.modules.weight_channels import gaussian_coefficients
from nems.modules.fir import da_coefficients

params = {'legend.fontsize': 6,
          'figure.figsize': (8, 6),
          'axes.labelsize': 8,
          'axes.titlesize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)


#def ax_remove_box(ax=None):
#    """
#    remove right and top lines from plot border
#    """
#    if ax is None:
#        ax = plt.gca()
#    ax.spines['right'].set_visible(False)
#    ax.spines['top'].set_visible(False)


def get_model_preds(cellid, batch, modelname):
    xf, ctx = xhelp.load_model_xform(cellid, batch, modelname,
                                     eval_model=False)
    ctx, l = xforms.evaluate(xf, ctx, stop=-1)
    #ctx, l = oxf.evaluate(xf, ctx, stop=-1)

    return xf, ctx


def compare_model_preds(cellid, batch, modelname1, modelname2,
                        max_pre=0.25, max_dur=1.0, stim_ids=None,
                        ctx1=None, ctx2=None):
    """
    compare prediction accuracy of two models on validation stimuli

    borrows a lot of functionality from nplt.quickplot()

    """
    if ctx1 is None:
        xf1, ctx1 = get_model_preds(cellid, batch, modelname1)
    if ctx2 is None:
        xf2, ctx2 = get_model_preds(cellid, batch, modelname2)
    colors = [[254/255, 15/255, 6/255],
              [217/255, 217/255, 217/255],
              [129/255, 201/255, 224/255],
              [128/255, 128/255, 128/255],
              [32/255, 32/255, 32/255]
              ]

    rec = ctx1['rec']
    val1 = ctx1['val']
    val2 = ctx2['val']

    stim = rec['stim'].rasterize()
    resp = rec['resp'].rasterize()
    pred1 = val1['pred']
    pred2 = val2['pred']
    fs = resp.fs

    d = resp.get_epoch_bounds('PreStimSilence')
    PreStimSilence = np.mean(np.diff(d))
    d = resp.get_epoch_bounds('PostStimSilence')
    PostStimSilence = np.mean(np.diff(d))

    epoch_regex = "^STIM_"
    stim_epochs = ep.epoch_names_matching(resp.epochs, epoch_regex)

    r = resp.as_matrix(stim_epochs)
    s = stim.as_matrix(stim_epochs)
    repcount = np.sum(np.isfinite(r[:, :, 0, 0]), axis=1)
    max_rep_id, = np.where(repcount == np.max(repcount))

    # keep a max of two stimuli
    if stim_ids is None:
        stim_ids = max_rep_id[:2]
    else:
        stim_ids = np.array(stim_ids)

    stim_count = len(stim_ids)
    # print(max_rep_id)

    # stim_i=max_rep_id[-1]
    # print("Max rep stim={} ({})".format(stim_i, stim_epochs[stim_i]))

    p1 = pred1.as_matrix(stim_epochs)
    p2 = pred2.as_matrix(stim_epochs)

    ms1 = ctx1['modelspec']
    ms2 = ctx2['modelspec']
    r_test1 = ms1.meta['r_test'][0]
    r_test2 = ms2.meta['r_test'][0]

    fh = plt.figure(figsize=(16, 6))

    # model 1 modules
    ax = plt.subplot(5, 4, 1)
    nplt.strf_timeseries(ms1, ax=ax, clim=None, show_factorized=True,
                         title="{}/{} rtest={:.3f}".format(cellid,modelname1,r_test1),
                         fs=resp.fs)
    nplt.ax_remove_box(ax)

    ax = plt.subplot(5, 4, 3)
    nplt.strf_timeseries(ms2, ax=ax, clim=None, show_factorized=True,
                      title="{}/{} rtest={:.3f}".format(cellid,modelname2,r_test2),
                      fs=resp.fs)
    nplt.ax_remove_box(ax)

    if find_module('stp', ms1):
        ax = plt.subplot(5, 4, 5)
        nplt.before_and_after_stp(ms1, sig_name='pred', ax=ax, title='',
                                  channels=0, xlabel='Time (s)', ylabel='STP',
                                  fs=resp.fs)
        nplt.ax_remove_box(ax)

    nlidx = find_module('double_exponential', ms1, find_all_matches=True)
    if len(nlidx):
        nlidx=nlidx[-1]
        fn1, fn2 = nplt.before_and_after_scatter(
                rec, ms1, nlidx, smoothing_bins=200,
                mod_name='double_exponential'
                )
        ax = plt.subplot(5, 4, 6)
        fn1(ax=ax)
        nplt.ax_remove_box(ax)

    # model 1 wc
    wcidx = find_module('weight_channels', ms1)
    if wcidx:
        ax = plt.subplot(5, 4, 2)
        try:
            coefs = ms1[wcidx]['phi']['coefficients']
            plt.imshow(coefs, clim=np.array([-1,1])*np.max(np.abs(coefs)), cmap='bwr')
            plt.xlabel('in')
            plt.ylabel('out')
            plt.colorbar()
        except:
            coefs = gaussian_coefficients(ms1[wcidx]['phi']['mean'],
                                          ms1[wcidx]['phi']['sd'],
                                          ms1[wcidx]['fn_kwargs']['n_chan_in'])
            coefs -= np.abs(np.min(coefs, axis=1, keepdims=True))
            coefs /= np.abs(np.sum(coefs, axis=1, keepdims=True))
            ax.set_prop_cycle(color=colors)
            plt.plot(coefs.T)
            plt.xlabel('in')
            plt.ylabel('gain')
        nplt.ax_remove_box(ax)

    # model 2 modules
    wcidx = find_module('weight_channels', ms2)
    if wcidx:
        ax = plt.subplot(5, 4, 4)
        try:
            coefs = ms2[wcidx]['phi']['coefficients']
            plt.imshow(coefs, clim=np.array([-1,1])*np.max(np.abs(coefs)), cmap='bwr')
            plt.xlabel('in')
            plt.ylabel('out')
            plt.colorbar()
        except:
            coefs = gaussian_coefficients(ms2[wcidx]['phi']['mean'],
                                          ms2[wcidx]['phi']['sd'],
                                          ms2[wcidx]['fn_kwargs']['n_chan_in'])
            coefs -= np.abs(np.min(coefs, axis=1, keepdims=True))
            coefs /= np.abs(np.sum(coefs, axis=1, keepdims=True))
            ax.set_prop_cycle(color=colors)
            plt.plot(coefs.T)
            plt.xlabel('in')
            plt.ylabel('gain')
        nplt.ax_remove_box(ax)

    if find_module('stp', ms2):
        ax = plt.subplot(5, 4, 7)
        nplt.before_and_after_stp(ms2, sig_name='pred', ax=ax, title='',
                                  channels=0, colors=colors, xlabel='Time (s)', ylabel='STP',
                                  fs=resp.fs)
        nplt.ax_remove_box(ax)

    nlidx = find_module('double_exponential', ms2, find_all_matches=True)
    if len(nlidx):
        nlidx=nlidx[-1]
        fn1, fn2 = nplt.before_and_after_scatter(
                rec, ms2, nlidx, smoothing_bins=200,
                mod_name='double_exponential'
                )
        ax = plt.subplot(5, 4, 8)
        fn1(ax=ax)
        nplt.ax_remove_box(ax)

    max_bins = int((PreStimSilence+max_dur)*fs)
    pre_cut_bins = int((PreStimSilence-max_pre)*fs)
    if pre_cut_bins < 0:
        pre_cut_bins = 0
    else:
        PreStimSilence = max_pre

    for i, stim_i in enumerate(stim_ids):

        ax = plt.subplot(5, 2, 5+i)
        if s.shape[2] <= 2:
            nplt.timeseries_from_vectors(
                    [s[stim_i, 0, 0, pre_cut_bins:max_bins],
                     s[max_rep_id[-1], 0, 1, pre_cut_bins:max_bins]],
                    fs=stim.fs, time_offset=PreStimSilence, ax=ax,
                    title="{}/{} rfit={:.3f}/{:.3f}".format(cellid,
                           stim_epochs[stim_i], r_test1, r_test2))
        else:
            nplt.plot_spectrogram(
                    s[stim_i, 0, :, pre_cut_bins:max_bins],
                    fs=stim.fs, time_offset=PreStimSilence, ax=ax,
                    title="{}/{} rfit={:.3f}/{:.3f}".format(
                            cellid, stim_epochs[stim_i], r_test1, r_test2))
        ax.get_xaxis().set_visible(False)
        nplt.ax_remove_box(ax)

        ax = plt.subplot(5, 2, 7+i)
        _r = r[stim_i, :, 0, pre_cut_bins:max_bins]
        t = np.arange(_r.shape[-1]) / resp.fs - PreStimSilence - 0.5/resp.fs
        nplt.raster(t, _r)
        ax.get_xaxis().set_visible(False)

        ax = plt.subplot(5, 2, 9+i)
        nplt.timeseries_from_vectors(
                [np.nanmean(_r, axis=0), p1[stim_i, 0, 0, pre_cut_bins:max_bins],
                 p2[stim_i, 0, 0, pre_cut_bins:max_bins]],
                fs=resp.fs, time_offset=PreStimSilence, ax=ax)
        nplt.ax_remove_box(ax)

    plt.tight_layout()
    return fh, ctx1, ctx2


def quick_pred_comp(cellid, batch, modelname1, modelname2,
                    ax=None, max_pre=0.25, max_dur=1.0, color1='orange',
                    color2='purple'):
    """
    compare prediction accuracy of two models on validation stimuli

    borrows a lot of functionality from nplt.quickplot()

    """
    ax0 = None
    if ax is None:
        ax = plt.gca()
    elif type(ax) is tuple:
        ax0=ax[0]
        ax=ax[1]

    xf1, ctx1 = get_model_preds(cellid, batch, modelname1)
    xf2, ctx2 = get_model_preds(cellid, batch, modelname2)

    ms1 = ctx1['modelspec']
    ms2 = ctx2['modelspec']
    r_test1 = ms1.meta['r_test'][0]
    r_test2 = ms2.meta['r_test'][0]

    rec = ctx1['rec']
    val1 = ctx1['val']
    val2 = ctx2['val']

    stim = val1['stim'].rasterize()
    resp = val1['resp'].rasterize()
    pred1 = val1['pred']
    pred2 = val2['pred']

    d = resp.get_epoch_bounds('PreStimSilence')
    PreStimSilence = np.mean(np.diff(d))
    d = resp.get_epoch_bounds('PostStimSilence')
    PostStimSilence = np.mean(np.diff(d))

    epoch_regex = "^STIM_"
    stim_epochs = ep.epoch_names_matching(resp.epochs, epoch_regex)

    r = resp.as_matrix(stim_epochs)
    s = stim.as_matrix(stim_epochs)
    p1 = pred1.as_matrix(stim_epochs)
    p2 = pred2.as_matrix(stim_epochs)
    fs = resp.fs

    repcount = np.sum(np.isfinite(r[:, :, 0, 0]), axis=1)
    max_rep_id, = np.where(repcount == np.max(repcount))

    # keep a max of two stimuli
    stim_ids = max_rep_id[:2]
    # stim_count = len(stim_ids)
    # print(max_rep_id)

    #stim_i=max_rep_id[-1]
    stim_i = stim_ids[0]
    # print("Max rep stim={} ({})".format(stim_i, stim_epochs[stim_i]))

    ds = 2
    max_bins = int((PreStimSilence+max_dur)*fs)
    pre_cut_bins = int((PreStimSilence-max_pre)*fs)
    if pre_cut_bins < 0:
        pre_cut_bins = 0
    else:
        PreStimSilence = max_pre

    if ax0 is not None:
        s1 = s[stim_i, 0, 0, pre_cut_bins:max_bins]
        s2 = s[stim_i, 0, 1, pre_cut_bins:max_bins]
        t = np.arange(len(s1))/fs - PreStimSilence
        ax0.plot(t, s1, color=(248/255, 153/255, 29/255))
        ax0.plot(t, s2, color=(65/255, 207/255, 221/255))

        #nplt.timeseries_from_vectors(
        #        [s[stim_i, 0, 0, :max_bins], s[stim_i, 0, 1, :max_bins]],
        #        fs=fs, time_offset=PreStimSilence, ax=ax0,
        #        title="{}".format(stim_epochs[stim_i]))
        nplt.ax_remove_box(ax0)

    lg = ("{:.3f}".format(r_test2), "{:.3f}".format(r_test1), 'act')

    _r = r[stim_i, :, 0, :]
    mr = np.nanmean(_r[:,pre_cut_bins:max_bins], axis=0) * fs
    pred1 = p1[stim_i, 0, 0, pre_cut_bins:max_bins] * fs
    pred2 = p2[stim_i, 0, 0, pre_cut_bins:max_bins] * fs

    if ds > 1:
        keepbins=int(np.floor(len(mr)/ds)*ds)
        mr = np.mean(np.reshape(mr[:keepbins], [-1, 2]), axis=1)
        pred1 = np.mean(np.reshape(pred1[:keepbins], [-1, 2]), axis=1)
        pred2 = np.mean(np.reshape(pred2[:keepbins], [-1, 2]), axis=1)
        fs = int(fs/ds)

    t = np.arange(len(mr))/fs - PreStimSilence

    ax.fill_between(t, np.zeros(t.shape), mr, facecolor='lightgray')
    ax.plot(t, pred1, color=color1)
    ax.plot(t, pred2, color=color2)

    ym = ax.get_ylim()
    ax.set_ylim(ym)
    ptext = "{}\n{:.3f}\n{:.3f}".format(cellid, r_test1, r_test2)
    ax.text(t[0], ym[1], cellid, fontsize=8, va='top')
    ax.text(t[0], ym[1]*.85, "{:.3f}".format(r_test1),
            fontsize=8, va='top', color=color1)
    ax.text(t[0], ym[1]*.7, "{:.3f}".format(r_test2),
            fontsize=8, va='top', color=color2)

    #yl=ax.get_ylim()
    #plt.ylim([yl[0], yl[1]*2])
    nplt.ax_remove_box(ax)

    return ax, ctx1, ctx2


def scatter_comp(beta1, beta2, n1='model1', n2='model2', hist_bins=20,
                 hist_range=[-1, 1], title='modelname/batch',
                 highlight=None):
    """
    beta1, beta2 are T x 1 vectors
    scatter plot comparing beta1 vs. beta2
    histograms of marginals
    """
    beta1 = np.array(beta1)
    beta2 = np.array(beta2)

    # exclude cells without prepassive
    outcells = ((beta1 > hist_range[1]) | (beta1 < hist_range[0]) |
                (beta2 > hist_range[1]) | (beta2 < hist_range[0]))
    goodcells = (np.abs(beta1) > 0) | (np.abs(beta2) > 0)

    beta1[beta1 > hist_range[1]] = hist_range[1]
    beta1[beta1 < hist_range[0]] = hist_range[0]
    beta2[beta2 > hist_range[1]] = hist_range[1]
    beta2[beta2 < hist_range[0]] = hist_range[0]

    if highlight is None:
        set1 = goodcells
        set2 = []
    else:
        highlight = np.array(highlight)
        set1 = np.logical_and(goodcells, (highlight))
        set2 = np.logical_and(goodcells, (1-highlight))

    fh = plt.figure(figsize=(8, 6))

    plt.subplot(2, 2, 3)
    plt.plot(beta1[outcells], beta2[outcells], '.', color='red')
    plt.plot(beta1[set2], beta2[set2], '.', color='lightgray')
    plt.plot(beta1[set1], beta2[set1], 'k.')
    plt.plot(np.array(hist_range), np.array([0, 0]), 'k--')
    plt.plot(np.array([0, 0]), np.array(hist_range), 'k--')
    plt.plot(np.array(hist_range), np.array(hist_range), 'k--')
    plt.axis('equal')
    plt.axis('tight')

    plt.xlabel(n1)
    plt.ylabel(n2)
    plt.title(title)

    plt.subplot(2, 2, 1)
    plt.hist([beta1[set1], beta1[set2]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True,
             color=['black', 'lightgray'])
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta1[goodcells]),
                     np.mean(np.abs(beta1[goodcells]))))
    plt.xlabel(n1)

    ax = plt.subplot(2, 2, 4)
    plt.hist([beta2[set1], beta2[set2]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True, orientation="horizontal",
             color=['black', 'lightgray'])
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta2[goodcells]),
                     np.mean(np.abs(beta2[goodcells]))))
    plt.xlabel(n2)

    ax = plt.subplot(2, 2, 2)
    plt.hist([(beta2[set1]-beta1[set1]) * np.sign(beta2[set1]),
              beta2[set2]-beta1[set2] * np.sign(beta2[set2])],
             bins=hist_bins-1, range=[-hist_range[1]/2, hist_range[1]/2],
             histtype='bar', stacked=True,
             color=['black', 'lightgray'])
    plt.title('mean={:.3f} sterr={:.3f}'.
              format(np.mean(beta2[goodcells]-beta1[goodcells]),
                     np.std(beta2[goodcells]-beta1[goodcells])/np.sqrt(np.sum(goodcells))))
    plt.xlabel('difference')

    plt.tight_layout()

    return fh


def plot_weights_64D(h, cellids, highlight_cellid=None, vmin=None, vmax=None, cbar=True,
                     overlap_method='offset', ax=None):

    '''
    given a weight vector, h, plot the weights on the appropriate electrode channel
    mapped based on the cellids supplied. Weight vector must be sorted the same as
    cellids. Channels without weights will be plotted as empty dots. For cases
    where there are more than one unit on a given electrode, additional units will
    be "offset" from the array geometry as additional electrodes.
    '''

    if type(cellids) is not np.ndarray:
        cellids = np.array(cellids)

    if type(h) is not np.ndarray:
        h = np.array(h)
        if vmin is None:
            vmin = np.min(h)
        if vmax is None:
            vmax = np.max(h)
    else:
        if vmin is None:
            vmin = np.min(h)
        if vmax is None:
            vmax = np.max(h)

     # Make a vector for each column of electrodes

    # left column + right column are identical
    lr_col = np.arange(0,21*0.25,0.25)  # 25 micron vertical spacing
    left_ch_nums = np.arange(3,64,3)
    right_ch_nums = np.arange(4,65,3)
    center_ch_nums = np.insert(np.arange(5, 63, 3),obj=slice(0,1),values =[1,2],axis=0)
    center_col = np.arange(-0.25,20.25*.25,0.25)-0.125
    ch_nums = np.hstack((left_ch_nums, center_ch_nums, right_ch_nums))
    sort_inds = np.argsort(ch_nums)

    l_col = np.vstack((np.ones(21)*-0.2,lr_col))
    r_col = np.vstack((np.ones(21)*0.2,lr_col))
    c_col = np.vstack((np.zeros(22),center_col))
    locations = np.hstack((l_col,c_col,r_col))[:,sort_inds]
    if ax is not None:
        plt.sca(ax)
    else:
        plt.figure()
    plt.scatter(locations[0,:],locations[1,:],facecolor='none',edgecolor='k',s=50)

    # Now, color appropriately
    electrodes = np.zeros(len(cellids))

    for i in range(0, len(cellids)):
        electrodes[i] = int(cellids[i].split("-")[-2])
    if highlight_cellid is not None:
        h_electrode = int(highlight_cellid.split("-")[-2])
    else:
        h_electrode = None

    # Add locations for cases where two or greater units on an electrode
    electrodes=list(electrodes-1)  # cellids labeled 1-64, python counts 0-63
    dupes = list(set([int(x) for x in electrodes if electrodes.count(x)>1]))

    if overlap_method == 'mean':
        print('averaging weights across electrodes with multiple units:')
        print([d+1 for d in dupes])
        uelectrodes=list(set(electrodes))
        uh=np.zeros(len(uelectrodes))
        for i,e in enumerate(uelectrodes):
            uh[i] = np.mean(h[electrodes==e])
        electrodes = uelectrodes
        h = uh
        dupes = list(set([int(x) for x in electrodes if electrodes.count(x)>1]))
    else:
        print('electrodes with multiple units:')
        print([d+1 for d in dupes])


    num_of_dupes = [electrodes.count(x) for x in electrodes]
    num_of_dupes = list(set([x for x in num_of_dupes if x>1]))
    #max_duplicates = np.max(np.array(num_of_dupes))
    dup_locations=np.empty((2,int(np.sum(num_of_dupes))*len(dupes)))
    max_=0
    count = 0
    x_shifts = dict.fromkeys([str(i) for i in dupes])
    for i in np.arange(0,len(dupes)):
        loc_x = locations[0,dupes[i]]

        x_shifts[str(dupes[i])]=[]
        x_shifts[str(dupes[i])].append(loc_x)

        n_dupes = electrodes.count(dupes[i])-1
        shift = 0
        for d in range(0,n_dupes):
            if loc_x < 0:
                shift -= 0.2
            elif loc_x == 0:
                shift += 0.4
            elif loc_x > 0:
                shift += 0.2

            m = shift
            if m > max_:
                max_=m

            x_shifts[str(dupes[i])].append(loc_x+shift)

            count += 1
    count+=len(dupes)
    dup_locations = np.empty((2, count))
    c=0
    h_dupes = []
    for k in x_shifts.keys():
        index = np.argwhere(np.array(electrodes) == int(k))
        for i in range(0, len(x_shifts[k])):
            dup_locations[0,c] = x_shifts[k][i]
            dup_locations[1,c] = locations[1,int(k)]
            h_dupes.append(h[index[i][0]])
            c+=1

    plt.scatter(dup_locations[0,:],dup_locations[1,:],facecolor='none',edgecolor='k',s=50)

    plt.axis('scaled')
    plt.xlim(-max_-.3,max_+.3)

    c_id = np.sort([int(x) for x in electrodes if electrodes.count(x)==1])
    electrodes = [int(x) for x in electrodes]

    # find the indexes of the unique cellids
    indexes = np.argwhere(np.array([electrodes.count(x) for x in electrodes])==1)
    indexes2 = np.argwhere(np.array([electrodes.count(x) for x in electrodes])!=1)
    indexes=[x[0] for x in indexes]
    indexes2=[x[0] for x in indexes2]

    # make an inverse mask of the unique indexes
    mask = np.ones(len(h),np.bool)
    mask[indexes]=0

    # plot the unique ones
    import matplotlib
    norm =matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    cmap = matplotlib.cm.jet
    mappable = matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap)
    mappable.set_array(h[indexes])
    #mappable.set_cmap('jet')
    colors = mappable.to_rgba(list(h[indexes]))
    plt.scatter(locations[:,c_id][0,:],locations[:,c_id][1,:],
                          c=colors,vmin=vmin,vmax=vmax,s=50,edgecolor='none')
    # plot the duplicates
    norm =matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    cmap = matplotlib.cm.jet
    mappable = matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap)
    mappable.set_array(h[mask])
    #mappable.set_cmap('jet')
    #colors = mappable.to_rgba(h[mask])
    colors = mappable.to_rgba(h_dupes)
    plt.scatter(dup_locations[0,:],dup_locations[1,:],
                          c=colors,vmin=vmin,vmax=vmax,s=50,edgecolor='none')
    if h_electrode is not None:
        print("h_electrode={}".format(h_electrode))
        plt.scatter(locations[0, h_electrode], locations[1, h_electrode],
                    facecolor='none', s=60, lw=2, edgecolor='red')

    if cbar is True:
        plt.colorbar(mappable)


def plot_mean_weights_64D(h=None, cellids=None, l4=None, vmin=None, vmax=None, title=None):

    # for case where given single array

    if type(h) is not list:
        h = [h]

    if type(cellids) is not list:
        cellids = [cellids]

    if type(l4) is not list:
        l4 = [l4]


    # create average h-vector, after applying appropriate shift and filling in missing
    # electrodes with nans

    l4_zero = 52 - 1 # align center of l4 with 52
    shift = np.subtract(l4,l4_zero)
    max_shift = shift[np.argmax(abs(shift))]
    h_mat_full = np.full((len(h), 64+abs(max_shift)), np.nan)

    for i in range(0, h_mat_full.shape[0]):

        if type(cellids[i]) is not np.ndarray:
            cellids[i] = np.array(cellids[i])

        s = shift[i]
        electrodes = np.zeros(len(cellids[i]))
        for j in range(0, len(cellids[i])):
            electrodes[j] = int(cellids[i][j][-4:-2])

        chans = (np.sort([int(x) for x in electrodes])-1) + abs(max_shift)

        chans = np.add(chans,s)

        h_mat_full[i,chans] = h[i]

    # remove outliers
    one_sd = np.nanstd(h_mat_full.flatten())
    print(one_sd)
    print('adjusted {0} outliers'.format(np.sum(abs(h_mat_full)>3*one_sd)))
    out_inds = np.argwhere(abs(h_mat_full)>3*one_sd)
    print(h_mat_full[out_inds[:,0], out_inds[:,1]])
    h_mat_full[abs(h_mat_full)>3*one_sd] = 2*one_sd*np.sign(h_mat_full[abs(h_mat_full)>3*one_sd])
    print(h_mat_full[out_inds[:,0], out_inds[:,1]])

    # Compute a sliding window averge of the weights
    h_means = np.nanmean(h_mat_full,0)
    h_mat = np.zeros(h_means.shape)
    h_mat_error = np.zeros(h_means.shape)
    for i in range(0, len(h_mat)):
        if i < 4:
            h_mat[i] = np.nanmean(h_means[0:i])
            h_mat_error[i] = np.nanstd(h_means[0:i])/np.sqrt(i)
        elif i > h_mat.shape[0]-4:
            h_mat[i] = np.nanmean(h_means[i:])
            h_mat_error[i] = np.nanstd(h_means[i:])/np.sqrt(len(h_means)-i)
        else:
            h_mat[i] = np.nanmean(h_means[(i-2):(i+2)])
            h_mat_error[i] = np.nanstd(h_means[(i-2):(i+2)])/np.sqrt(4)

    if vmin is None:
        vmin = np.nanmin(h_mat)
    if vmax is None:
        vmax = np.nanmax(h_mat)


    # Now plot locations for each site

    # left column + right column are identical
    el_shift = int(abs(max_shift)/3)
    tf=0
    while tf is 0:
        if el_shift%3 != 0:
            el_shift += 1
        elif max_shift>0 and max_shift<3:
            el_shift+=1
            tf=1
        else:
            tf=1
    while max_shift%3 != 0:
        if max_shift<0:
            max_shift-=1
        elif max_shift>=0:
            max_shift+=1

    lr_col = np.arange(0,(21+el_shift)*0.25,0.25)  # 25 micron vertical spacing
    left_ch_nums = np.arange(3,64+abs(max_shift),3)
    right_ch_nums = np.arange(4,65+abs(max_shift),3)
    center_ch_nums = np.insert(np.arange(5, 63+abs(max_shift), 3),obj=slice(0,1),values =[1,2],axis=0)
    center_col = np.arange(-0.25,(20.25+el_shift)*.25,0.25)-0.125
    ch_nums = np.hstack((left_ch_nums, center_ch_nums, right_ch_nums))
    sort_inds = np.argsort(ch_nums)

    l_col = np.vstack((np.ones(21+el_shift)*-0.2,lr_col))
    r_col = np.vstack((np.ones(21+el_shift)*0.2,lr_col))
    c_col = np.vstack((np.zeros(22+el_shift),center_col))

    if l_col.shape[1]!=len(left_ch_nums):
        left_ch_nums = np.concatenate((left_ch_nums,[left_ch_nums[-1]+3]))
    if r_col.shape[1]!=len(right_ch_nums):
        right_ch_nums = np.concatenate((right_ch_nums,[left_ch_nums[-1]+3]))
    if c_col.shape[1]!=len(center_ch_nums):
        center_ch_nums = np.concatenate((center_ch_nums,[left_ch_nums[-1]+3]))

    ch_nums = np.hstack((left_ch_nums, center_ch_nums, right_ch_nums))
    sort_inds = np.argsort(ch_nums)

    l_col = np.vstack((np.ones(21+el_shift)*-0.2,lr_col))
    r_col = np.vstack((np.ones(21+el_shift)*0.2,lr_col))
    c_col = np.vstack((np.zeros(22+el_shift),center_col))

    locations = np.hstack((l_col,c_col,r_col))[:,sort_inds]


    locations[1,:] = 100*(locations[1,:])
    locations[0,:] = 3000*(locations[0,:]*0.2)
    print(h_mat_full.shape)
    if h_mat.shape[0] != locations.shape[1]:
        diff = locations.shape[1] - h_mat.shape[0]
        h_mat_scatter = np.concatenate((h_mat_full, np.full((np.shape(h_mat_full)[0],diff), np.nan)),axis=1)
        h_mat = np.concatenate((h_mat, np.full(diff,np.nan)))
        h_mat_error = np.concatenate((h_mat_error, np.full(diff,np.nan)))

    if title is not None:
        plt.figure(title)
    else:
        plt.figure()
    plt.subplot(142)
    plt.title('mean weights per channel')
    plt.scatter(locations[0,:],locations[1,:],facecolor='none',edgecolor='k',s=50)

    indexes = [x[0] for x in np.argwhere(~np.isnan(h_mat))]
    # plot the colors
    import matplotlib
    norm =matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    cmap = matplotlib.cm.jet
    mappable = matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap)
    mappable.set_array(h_mat[indexes])
    colors = mappable.to_rgba(list(h_mat[indexes]))
    plt.scatter(locations[:,indexes][0,:],locations[:,indexes][1,:],
                          c=colors,vmin=vmin,vmax=vmax,s=50,edgecolor='none')
    plt.colorbar(mappable) #,orientation='vertical',fraction=0.04, pad=0.0)
    #plt.axis('scaled')
    plt.xlim(-500,500)
    plt.axis('off')

    # Add dashed line at "layer IV"
    plt.plot([-250, 250], [locations[1][l4_zero]+75, locations[1][l4_zero]+75],
             linestyle='-', color='k', lw=4,alpha=0.3)
    plt.plot([-250, 250], [locations[1][l4_zero]-75, locations[1][l4_zero]-75],
             linestyle='-', color='k', lw=4,alpha=0.3)

    # plot conditional density

    h_kde = h_mat.copy()
    sigma = 3
    h_kde[np.isnan(h_mat)]=0
    h_kde = sf.gaussian_filter1d(h_kde,sigma)
    h_kde_error = h_mat_error.copy()
    h_kde_error[np.isnan(h_mat)]=0
    h_kde_error = sf.gaussian_filter1d(h_kde_error,sigma)
    plt.subplot(141)
    plt.title('smoothed mean weights')
    plt.plot(-h_kde, locations[1,:],lw=3,color='k')
    plt.fill_betweenx(locations[1,:], -(h_kde+h_kde_error), -(h_kde-h_kde_error), alpha=0.3, facecolor='k')
    plt.axhline(locations[1][l4_zero]+75,color='k',lw=3,alpha=0.3)
    plt.axhline(locations[1][l4_zero]-75,color='k',lw=3,alpha=0.3)
    plt.axvline(0, color='k',linestyle='--',alpha=0.5)
    plt.ylabel('um (layer IV center at {0} um)'.format(int(locations[1][l4_zero])))
    #plt.xlim(-vmax, -vmin)
    for i in range(0, h_mat_scatter.shape[0]):
        plt.plot(-h_mat_scatter[i,:],locations[1,:],'.')
    #plt.axis('off')

    # plot binned histogram for each layer
    plt.subplot(222)
    l4_shift = locations[1][l4_zero]
    plt.title('center of layer IV: {0} um'.format(l4_shift))
    # 24 electrodes spans roughly 200um
    # shift by 18 (150um) each window
    width_string = '200um'
    width = 24
    step = 18
    sets = int(h_mat_full.shape[1]/step)+1
    print('number of {1} bins: {0}'.format(sets, width_string))

    si = 0
    legend_strings = []
    w = []
    for i in range(0, sets):
        if si+width > h_mat_full.shape[1]:
            w.append(h_mat_full[:,si:][~np.isnan(h_mat_full[:,si:])])
            plt.hist(w[i],alpha=0.5)
            legend_strings.append(str(int(100*si/3*0.25))+', '+str(int(100*h_mat_full.shape[1]/3*0.25))+'um')
            si+=step
        else:
            w.append(h_mat_full[:,si:(si+width)][~np.isnan(h_mat_full[:,si:(si+width)])])
            plt.hist(w[i],alpha=0.5)
            legend_strings.append(str(int(100*si/3*0.25))+', '+str(int(100*(si+width)/3*0.25))+'um')
            si+=step

    plt.legend(legend_strings[::-1])
    plt.xlabel('weight')
    plt.ylabel('counts per {0} bin'.format(width_string))

    plt.subplot(224)
    mw = []
    mw_error = []
    for i in range(0, sets):
        mw.append(np.nanmean(w[i]))
        mw_error.append(np.nanstd(w[i])/np.sqrt(len(w[i])))

    plt.bar(np.arange(0,sets), mw, yerr=mw_error, facecolor='k',alpha=0.5)
    plt.xticks(np.arange(0,sets), legend_strings, rotation=45)
    plt.xlabel('Window')
    plt.ylabel('Mean weight')

    plt.tight_layout()


def pop_weights(modelspec, rec=None, idx=None, sig="state", variable="g", prefix="",
                ax=None, title=None, **options):
    """
    :param modelspec: modelspec object
    :param rec: recording object
    :param idx: index into modelspec
    :param ax: axes to use for plot. if None, plot in new figure
    :return: axes where plotted
    """

    chans = rec[sig].chans
    if prefix == "":
        chan_match = [i for i in range(len(chans)) if (len(chans[i].split("-"))==3) and (chans[i][1]!='x')]
    else:
        chan_match = [i for i in range(len(chans)) if chans[i].startswith(prefix)]

    phi_mean = modelspec.phi_mean[idx][variable]
    phi_sem = modelspec.phi_sem[idx][variable]
    phi_z = phi_mean / phi_sem
    weights = [phi_z[0,i] for i in chan_match]
    cellids = [chans[i] for i in chan_match]
    print(cellids)

    if ax is None:
        plt.figure()
        ax=plt.subplot(111)
    highlight_cellid = modelspec.meta['cellid']
    plot_weights_64D(h=weights, cellids=cellids, highlight_cellid=highlight_cellid, ax=ax)

    return ax


def depth_analysis_64D(h, cellids, l4=None, depth_list=None, title=None):

    # for case where given single array
    if type(h) is not list:
        h = [h]
    if type(cellids) is not list:
        cellids = [cellids]
    if l4 is not None and type(l4) is not list:
        l4 = [l4]
    if (depth_list is not None) & (type(depth_list) is not list):
        depth_list = [depth_list]

    l4_zero = 48  # arbitrary - just used to align everything to center of layer four

    if depth_list is None:
        # Define depth for each electrode
        lr_col = np.arange(0,21*0.25,0.25)          # 25 micron vertical spacing
        center_col = np.arange(-0.25,20.25*.25,0.25)-0.125
        left_ch_nums = np.arange(3,64,3)
        right_ch_nums = np.arange(4,65,3)
        center_ch_nums = np.insert(np.arange(5, 63, 3),obj=slice(0,1),values =[1,2],axis=0)
        ch_nums = np.hstack((left_ch_nums, center_ch_nums, right_ch_nums))
        sort_inds = np.argsort(ch_nums)

        # define locations of all electrodes
        l_col = np.vstack((np.ones(21)*-0.2,lr_col))
        r_col = np.vstack((np.ones(21)*0.2,lr_col))
        c_col = np.vstack((np.zeros(22),center_col))
        locations = np.hstack((l_col,c_col,r_col))[:,sort_inds]

        chan_depth_weight=[]
        l4_depth = round(0.25*((l4_zero)/3)*100,2)
        l4_depth_ = round(0.25*((l4_zero)/3),2)
        # Assign each channel in each recording a depth
        for i in range(0, len(h)):
            chans = np.array([int(x[-4:-2]) for x in cellids[i]])
            l4_loc = locations[1,l4[i]]
            shift_by = l4_depth_ - l4_loc
            print('shift site {0} by {1} um'.format(cellids[i][0][:-5], round(shift_by*100,2)))
            depths = np.array([locations[1,c] for c in chans]) + shift_by
            w = h[i]
            fs = []
            for c in cellids[i]:
                try:
                    fs.append(nd.get_wft(c))
                except:
                    fs.append(-1)

            chan_depth_weight.append(pd.DataFrame(data=np.vstack((chans, depths, w, fs)).T,
                             columns=['chans','depths', 'weights', 'wft']))
    elif depth_list is not None:
        chan_depth_weight=[]
        l4_depth = l4_depth = 0.25*int((l4_zero)/3)*100
        for i in range(0, len(h)):
            chans = np.array([int(x[-4:-2]) for x in cellids[i]])
            depths = np.array(depth_list[i])
            w = h[i]
            fs = []
            for c in cellids[i]:
                try:
                    fs.append(nd.get_wft(c))
                except:
                    fs.append(-1)
            chan_depth_weight.append(pd.DataFrame(data=np.vstack((chans, depths, w, fs)).T,
                             columns=['chans','depths', 'weights', 'wft']))

    chan_depth_weight = pd.concat(chan_depth_weight)
    chan_depth_weight['depths'] = chan_depth_weight['depths']*100

    # shift depths so that top of layer four is at 400um and depths count down
    top_l4 = l4_depth + 100
    chan_depth_weight['depth_adjusted'] = chan_depth_weight['depths'] - top_l4 - 400
    mi = chan_depth_weight.min()['depths']
    if mi<0:
        chan_depth_weight['depths'] = chan_depth_weight['depths']+abs(mi)
        l4_depth += abs(mi)
    else:
        chan_depth_weight['depths'] = chan_depth_weight['depths']-mi
        l4_depth -= mi

    # bin for bar plot
    step_size = 100
    bin_size = 100
    wBinned = []
    wError = []
    w_fsBinned = []
    w_rsBinned = []
    w_fsError = []
    w_rsError = []
    xlabels = []

    start = int(chan_depth_weight.min()['depth_adjusted'])
    m = chan_depth_weight.max()['depths']
    nBins = int(m/step_size)+1
    nBins = int(np.floor((chan_depth_weight.max()['depth_adjusted'] -
                          chan_depth_weight.min()['depth_adjusted'])/step_size))
    end = int(start + nBins * step_size)

    fs_df = chan_depth_weight[chan_depth_weight['wft'] == 1]
    rs_df = chan_depth_weight[chan_depth_weight['wft'] != 1]

    for i in np.arange(start, end, step_size):
        w = chan_depth_weight[(chan_depth_weight['depth_adjusted']>i).values & (chan_depth_weight['depth_adjusted']<(i+bin_size)).values]
        mw = w.mean()['weights']
        sd = w.std()['weights']/np.sqrt(len(w['weights']))
        wBinned.append(mw)
        wError.append(sd)

        w = fs_df[(fs_df['depth_adjusted']>i).values & (fs_df['depth_adjusted']<(i+bin_size)).values]
        mw = w.mean()['weights']
        sd = w.std()['weights']/np.sqrt(len(w['weights']))
        w_fsBinned.append(mw)
        w_fsError.append(sd)

        w = rs_df[(rs_df['depth_adjusted']>i).values & (rs_df['depth_adjusted']<(i+bin_size)).values]
        mw = w.mean()['weights']
        sd = w.std()['weights']/np.sqrt(len(w['weights']))
        w_rsBinned.append(mw)
        w_rsError.append(sd)

        xlabels.append(str(i)+' - '+str(i+bin_size)+' um')

    # fine binning for sliding window
    step_size=5
    bin_size=50
    nWindows = int(m/step_size)
    depthBin = []
    m_sw = []
    e_sw = []
    for i in np.arange(start, end, step_size):
        w = chan_depth_weight[(chan_depth_weight['depth_adjusted']>(i)).values & (chan_depth_weight['depth_adjusted']<(i+bin_size)).values]
        mw = w.mean()['weights']
        sd = w.std()['weights']/np.sqrt(len(w['weights']))
        if ~np.isnan(sd):
            m_sw.append(mw)
            e_sw.append(sd)
            depthBin.append(np.mean([i, i+bin_size]))

    sigma = 1
    m_sw = sf.gaussian_filter1d(np.array(m_sw), sigma)
    e_sw = sf.gaussian_filter1d(np.array(e_sw), sigma)
    plt.figure()
    if title is not None:
        plt.suptitle(title)
    plt.subplot(121)
    plt.plot(-m_sw, depthBin, 'k-')
    for i in range(0, len(chan_depth_weight)):
        if chan_depth_weight.iloc[i]['wft']==1:
            plt.plot(-chan_depth_weight.iloc[i]['weights'], chan_depth_weight.iloc[i]['depth_adjusted'], color='r',marker='.')
        else:
            plt.plot(-chan_depth_weight.iloc[i]['weights'], chan_depth_weight.iloc[i]['depth_adjusted'], color='k',marker='.')

    plt.fill_betweenx(depthBin, -(e_sw+m_sw), e_sw+-m_sw ,alpha=0.3, facecolor='k')
    plt.axvline(0, color='k',linestyle='--')
    plt.axhline(-600, color='Grey', lw=2)
    plt.axhline(-400, color='Grey', lw=2)
    plt.ylabel('depth from surface (um)')
    plt.xlabel('weights')

    plt.subplot(222)
    plt.bar(np.arange(0, nBins), wBinned, yerr=wError,facecolor='Grey')

    plt.title('layer IV depth: {0}'.format(l4_depth))

    plt.subplot(224)
    plt.bar(np.arange(0, nBins, 1), w_fsBinned, width=0.4, yerr=w_fsError,facecolor='Red')
    plt.bar(np.arange(0.5, nBins, 1), w_rsBinned, width=0.4, yerr=w_rsError,facecolor='Black')
    plt.xticks(np.arange(0, nBins,1), xlabels, rotation=45)
    plt.legend(['fast-spiking', 'regular-spiking'])


def LN_plot(ctx, ax1=None, ax2=None, ax3=None, ax4=None):
    """
    compact summary plot for model fit to a single dim of a population subspace

    in 2-4 panels, show: pc load, timecourse plus STRF + static NL
    (skip the first two if their respective ax handles are None)

    """
    rec = ctx['val'][0].apply_mask()
    modelspec = ctx['modelspecs'][0]
    rec = ms.evaluate(rec, modelspec)
    cellid = modelspec[0]['meta']['cellid']
    fs = ctx['rec']['resp'].fs
    pc_idx = ctx['rec'].meta['pc_idx']

    if (ax1 is not None) and (pc_idx is not None):
        cellids=ctx['rec'].meta['cellid']
        h=ctx['rec'].meta['pc_weights'][pc_idx[0],:]
        max_w=np.max(np.abs(h))*0.75
        plt.sca(ax1)
        plot_weights_64D(h,cellids,vmin=-max_w,vmax=max_w)
        plt.axis('off')

    if ax2 is not None:
        r = ctx['rec']['resp'].extract_epoch('REFERENCE',
               mask=ctx['rec']['mask'])
        d = ctx['rec']['resp'].get_epoch_bounds('PreStimSilence')
        if len(d):
            PreStimSilence = np.mean(np.diff(d))
        else:
            PreStimSilence = 0
        prestimbins = int(PreStimSilence * fs)

        mr=np.mean(r,axis=0)
        spont=np.mean(mr[:,:prestimbins],axis=1,keepdims=True)
        mr-=spont
        mr /= np.max(np.abs(mr),axis=1,keepdims=True)
        tt=np.arange(mr.shape[1])/fs
        ax2.plot(tt-PreStimSilence, mr[0,:], 'k')
        # time bar
        ax2.plot(np.array([0,1]),np.array([1.1, 1.1]), 'k', lw=3)
        nplt.ax_remove_box(ax2)
        ax2.set_title(cellid)

    title="r_fit={:.3f} test={:.3f}".format(
            modelspec[0]['meta']['r_fit'][0],
            modelspec[0]['meta']['r_test'][0])

    nplt.strf_heatmap(modelspec, title=title, interpolation=(2,3),
                      show_factorized=False, fs=fs, ax=ax3)
    nplt.ax_remove_box(ax3)

    nl_mod_idx = find_module('nonlinearity', modelspec)
    nplt.nl_scatter(ctx['est'][0].apply_mask(), modelspec, nl_mod_idx, sig_name='pred',
                    compare='resp', smoothing_bins=60,
                    xlabel1=None, ylabel1=None, ax=ax4)

    sg_mod_idx = find_module('state', modelspec)
    if sg_mod_idx is not None:
        modelspec2 = copy.deepcopy(modelspec)
        g=modelspec2[sg_mod_idx]['phi']['g'][0,:]
        d=modelspec2[sg_mod_idx]['phi']['d'][0,:]

        modelspec2[nl_mod_idx]['phi']['amplitude'] *= 1+g[-1]
        modelspec2[nl_mod_idx]['phi']['base'] += d[-1]
        nplt.plot_nl_io(modelspec2[nl_mod_idx], ax4.get_xlim(), ax4)
        g=["{:.2f}".format(g) for g in list(modelspec[sg_mod_idx]['phi']['g'][0,:])]
        ts = "SG: " + " ".join(g)
        ax4.set_title(ts)

    nplt.ax_remove_box(ax4)

def LN_pop_plot(ctx):
    """
    compact summary plot for model fit to a single dim of a population subspace

    in 2-4 panels, show: pc load, timecourse plus STRF + static NL
    (skip the first two if their respective ax handles are None)

    """
    rec = ctx['val']
    modelspec = ctx['modelspec']
    rec = ms.evaluate(rec, modelspec)
    cellid = modelspec[0]['meta']['cellid']

    resp = rec['resp']
    stim = rec['stim']
    pred = rec['pred']
    fs = resp.fs

    fir_idx = find_module('fir', modelspec)
    wc_idx = find_module('weight_channels', modelspec, find_all_matches=True)

    chan_count = modelspec[wc_idx[-1]]['phi']['coefficients'].shape[1]
    cell_count = modelspec[wc_idx[-1]]['phi']['coefficients'].shape[0]
    filter_count = modelspec[fir_idx]['phi']['coefficients'].shape[0]
    bank_count = modelspec[fir_idx]['fn_kwargs']['bank_count']
    chan_per_bank = int(filter_count/bank_count)

    fig = plt.figure()
    for chanidx in range(chan_count):

        tmodelspec=copy.deepcopy(modelspec[:(fir_idx+1)])
        tmodelspec[fir_idx]['fn_kwargs']['bank_count']=1
        rr=slice(chanidx*chan_per_bank, (chanidx+1)*chan_per_bank)
        tmodelspec[wc_idx[0]]['phi']['mean'] = tmodelspec[wc_idx[0]]['phi']['mean'][rr]
        tmodelspec[wc_idx[0]]['phi']['sd'] = tmodelspec[wc_idx[0]]['phi']['sd'][rr]
        tmodelspec[fir_idx]['phi']['coefficients'] = \
                   tmodelspec[fir_idx]['phi']['coefficients'][rr,:]

        ax = fig.add_subplot(chan_count, 3, chanidx*3+1)
        nplt.strf_heatmap(tmodelspec, title=None, interpolation=(2,3),
                          show_factorized=False, fs=fs, ax=ax)
        nplt.ax_remove_box(ax)
        if chanidx < chan_count-1:
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('')
            plt.ylabel('')

    ax = fig.add_subplot(2, 3, 2)
    fcc = modelspec[fir_idx]['phi']['coefficients'].copy()
    fcc = np.reshape(fcc, (chan_per_bank, bank_count, -1))
    fcc = np.mean(fcc,axis=0)
    fcc_std = np.std(fcc,axis=1,keepdims=True)
    wcc = modelspec[wc_idx[-1]]['phi']['coefficients'].copy().T
    wcc *= fcc_std
    mm = np.std(wcc)*3
    im = ax.imshow(wcc, aspect='auto', clim=[-mm, mm], cmap='bwr')
    #plt.colorbar(im)
    plt.title(modelspec.meta['cellid'])
    nplt.ax_remove_box(ax)

    ax = fig.add_subplot(2, 3, 3)
    plt.plot(modelspec.meta['r_test'])
    plt.xlabel('cell')
    plt.ylabel('r test')
    nplt.ax_remove_box(ax)


    epoch_regex = '^STIM_'
    epochs_to_extract = ep.epoch_names_matching(rec.epochs, epoch_regex)
    epoch=epochs_to_extract[0]

    # or just plot the PSTH for an example stimulus
    raster = resp.extract_epoch(epoch)
    psth = np.mean(raster, axis=0)
    praster = pred.extract_epoch(epoch)
    ppsth = np.mean(praster, axis=0)
    spec = stim.extract_epoch(epoch)[0,:,:]
    trimbins=50
    if trimbins > 0:
        ppsth=ppsth[:,trimbins:]
        psth=psth[:,trimbins:]
        spec=spec[:,trimbins:]

    ax = plt.subplot(6, 2, 8)
    #nplt.plot_spectrogram(spec, fs=resp.fs, ax=ax, title=epoch)
    extent = [0.5/fs, (spec.shape[1]+0.5)/fs, 0.5, spec.shape[0]+0.5]
    im=ax.imshow(spec, origin='lower', interpolation='none',
                 aspect='auto', extent=extent)
    nplt.ax_remove_box(ax)
    plt.ylabel('stim')
    plt.xticks([])
    plt.colorbar(im)

    ax = plt.subplot(6, 2, 10)
    clim=(np.nanmin(psth),np.nanmax(psth)*.6)
    #nplt.plot_spectrogram(psth, fs=resp.fs, ax=ax, title="resp",
    #                      cmap='gray_r', clim=clim)
    #fig.colorbar(im, cax=ax, orientation='vertical')
    im=ax.imshow(psth, origin='lower', interpolation='none',
                 aspect='auto', extent=extent,
                 cmap='gray_r', clim=clim)
    nplt.ax_remove_box(ax)
    plt.ylabel('resp')
    plt.xticks([])
    plt.colorbar(im)

    ax = plt.subplot(6, 2, 12)
    clim=(np.nanmin(psth),np.nanmax(ppsth))
    im=ax.imshow(ppsth, origin='lower', interpolation='none',
                 aspect='auto', extent=extent,
                 cmap='gray_r', clim=clim)
    nplt.ax_remove_box(ax)
    plt.ylabel('pred')
    plt.colorbar(im)

#    if (ax1 is not None) and (pc_idx is not None):
#        cellids=ctx['rec'].meta['cellid']
#        h=ctx['rec'].meta['pc_weights'][pc_idx[0],:]
#        max_w=np.max(np.abs(h))*0.75
#        plt.sca(ax1)
#        plot_weights_64D(h,cellids,vmin=-max_w,vmax=max_w)
#        plt.axis('off')
#
#    if ax2 is not None:
#        r = ctx['rec']['resp'].extract_epoch('REFERENCE',
#               mask=ctx['rec']['mask'])
#        d = ctx['rec']['resp'].get_epoch_bounds('PreStimSilence')
#        if len(d):
#            PreStimSilence = np.mean(np.diff(d))
#        else:
#            PreStimSilence = 0
#        prestimbins = int(PreStimSilence * fs)
#
#        mr=np.mean(r,axis=0)
#        spont=np.mean(mr[:,:prestimbins],axis=1,keepdims=True)
#        mr-=spont
#        mr /= np.max(np.abs(mr),axis=1,keepdims=True)
#        tt=np.arange(mr.shape[1])/fs
#        ax2.plot(tt-PreStimSilence, mr[0,:], 'k')
#        # time bar
#        ax2.plot(np.array([0,1]),np.array([1.1, 1.1]), 'k', lw=3)
#        nplt.ax_remove_box(ax2)
#        ax2.set_title(cellid)

#    title="r_fit={:.3f} test={:.3f}".format(
#            modelspec[0]['meta']['r_fit'][0],
#            modelspec[0]['meta']['r_test'][0])


#    nl_mod_idx = find_module('nonlinearity', modelspec)
#    nplt.nl_scatter(rec, modelspec, nl_mod_idx, sig_name='pred',
#                    compare='resp', smoothing_bins=60,
#                    xlabel1=None, ylabel1=None, ax=ax4)
#
#    sg_mod_idx = find_module('state', modelspec)
#    if sg_mod_idx is not None:
#        modelspec2 = copy.deepcopy(modelspec)
#        g=modelspec2[sg_mod_idx]['phi']['g'][0,:]
#        d=modelspec2[sg_mod_idx]['phi']['d'][0,:]
#
#        modelspec2[nl_mod_idx]['phi']['amplitude'] *= 1+g[-1]
#        modelspec2[nl_mod_idx]['phi']['base'] += d[-1]
#        nplt.plot_nl_io(modelspec2[nl_mod_idx], ax4.get_xlim(), ax4)
#        g=["{:.2f}".format(g) for g in list(modelspec[sg_mod_idx]['phi']['g'][0,:])]
#        ts = "SG: " + " ".join(g)
#        ax4.set_title(ts)
#
#    nplt.ax_remove_box(ax4)
    return fig


def model_comp_pareto(modelnames=None, batch=0, modelgroups=None, goodcells=None,
                      offset=0.5, max=0.85, ax=None):

    if (modelnames is None) and (modelgroups is None):
        raise ValueError("Must specify modelnames list or modelgroups dict")
    elif modelgroups is None:
        modelgroups={'ALL': modelnames}
    modelnames=[]
    for k, m in modelgroups.items():
        modelnames.extend(m)

    dot_colors = ['k','b','r','g','purple','orange','lightblue']
    if ax is None:
        ax = plt.gca()

    if goodcells is None:
        cellids=None
    else:
        cellids=list(goodcells.index)
    b_ceiling = nd.batch_comp(batch, modelnames, cellids=cellids, stat='r_ceiling')
    b_n = nd.batch_comp(batch, modelnames, cellids=cellids, stat='n_parms')

    # find good cells
    if goodcells is None:
        b_test = nd.batch_comp(batch, modelnames, stat='r_test')
        b_se = nd.batch_comp(batch, modelnames, stat='se_test')
        b_goodcells = np.zeros_like(b_test)
        for i, m in enumerate(modelnames):
            td = b_test[[m]].join(b_se[[m]], rsuffix='_se')
            b_goodcells[:,i] = td[m] > 2*td[m+'_se']
        goodcells = np.sum(b_goodcells, axis=1)/(len(modelnames)*0.05) > 1
    #b_m = np.array((b_ceiling.loc[goodcells]**2).mean()[modelnames])
    b_m = np.array((b_ceiling.loc[goodcells]).mean()[modelnames])
    n_parms = np.array([np.mean(b_n[m]) for m in modelnames])

    #u_modelgroups = np.unique(modelgroups)
    #for i, g in enumerate(u_modelgroups):
    i=0
    for k, m in modelgroups.items():
        jj = [m0 in m for m0 in modelnames]
        modelset=[]
        for jjj in range(len(jj)):
            if jj[jjj]:
                modelset.append(modelnames[jjj])
        print("{} : {}".format(k, modelset))
        ax.plot(n_parms[jj], b_m[jj], '-', color=dot_colors[i])
        ax.plot(n_parms[jj], b_m[jj], '.', color=dot_colors[i], label=k)
        i+=1

    handles, labels = ax.get_legend_handles_labels()
    # reverse the order
    ax.legend(handles, labels, loc='lower right')

    ax.set_xlabel('Free parameters')
    ax.set_ylabel('Mean var explained')
    ax.set_ylim((offset-0.05, max))
    nplt.ax_remove_box(ax)

    return ax, b_ceiling



def lv_quickplot(rec, modelspec, ax=None, **options):
    """
    quick view of latent variable and the "encoding" weights
    """
    r = rec.apply_mask()
    
    f = plt.figure(figsize=(8, 4))
    lv = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    weights = plt.subplot2grid((1, 3), (0, 2), colspan=1)
    
    lv.plot(r['lv']._data.T)
    lv.legend(r['lv'].chans, fontsize=6)
    lv.set_xlabel('Time')
    lv.set_title('Latent Variable(s)')

    # figure out module index
    idx = [i for i in range(0, len(modelspec.modules)) if 'nems_lbhb.modules.state.add_lv' in modelspec.modules[i]['fn']][0]

    lim = np.max(abs(modelspec.phi[idx]['e'].squeeze()))
    bins = np.linspace(-lim, lim, 11)
    nLVs = modelspec.phi[idx]['e'].shape[-1]
    for i in range(nLVs):
        weights.hist(modelspec.phi[idx]['e'][:, i], bins=bins, alpha=0.5, edgecolor='k', label=r['lv'].chans[i+1])
    weights.legend(fontsize=8)
    weights.axvline(0, linestyle='--', color='r')
    weights.set_xlabel('Encoding weights')

    modelspecs = modelspec.modelspecname.split('-')
    f.suptitle(modelspecs[idx])
    f.tight_layout()


def state_logsig_plot(rec, modelspec, ax=None, **options):
    """
    quick view of first order model fit weight(s). Show fit for 
    'best' first order prediction.
    """

    r = rec.apply_mask()

    f = plt.figure()
    rp_fig = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    state_plot = plt.subplot2grid((2, 3), (1, 0), colspan=2)
    sig = plt.subplot2grid((2, 3), (0, 2), colspan=1)
    weights = plt.subplot2grid((2, 3), (1, 2), colspan=1)

    # figure out module index
    idx = [i for i in range(0, len(modelspec.modules)) if 'nems_lbhb.modules.state.state_logsig' in modelspec.modules[i]['fn']][0]

    best = np.argmax(modelspec.phi[idx]['g'][:, 1])
    best2 = np.argmin(modelspec.phi[idx]['g'][:, 1])
    cellid = r['resp'].chans[best]
    cellid2 = r['resp'].chans[best2]

    # prediction of "best" first order cell (biggest state effect, based on model fit)
    rp_fig.plot(r['resp']._data[best, :], color='grey', label='resp')
    rp_fig.plot(r['pred']._data[best, :], color='k', label='pred')
    rp_fig.legend()
    rp_fig.set_title(cellid)

    # plot state timeseries
    state_plot.plot(r['state']._data.T)
    state_plot.set_xlabel('Time')
    state_plot.set_title('state signals')
    
    # gain applied as function of pupil for this "best" cell
    s = r['state']._data
    g = modelspec.phi[idx]['g'][best, :]
    a = modelspec.phi[idx]['a'][best, 0]
    sg = g @ s
    sg = a / (1 + np.exp(-sg))
    sig.plot(s[1, :], sg, '.', color='k')
    sig.set_xlabel('state')
    sig.set_ylabel('linear gain applied')
    

    g = modelspec.phi[idx]['g'][best2, :]
    a = modelspec.phi[idx]['a'][best2, 0]
    sg = g @ s
    sg = a / (1 + np.exp(-sg))
    sig.plot(s[1, :], sg, '.', color='r')

    sig.legend(['{0} state gain (phi): {1}'.format(cellid, round(modelspec.phi[idx]['g'][best, 1], 3)), 
                '{0} state gain (phi): {1}'.format(cellid2, round(modelspec.phi[idx]['g'][best2, 1], 3))])
    sig.axhline(1, linestyle='--', color='grey')
    sig.axvline(0, linestyle='--', color='grey')
    

    # all model weights (pupil gain weights)
    lim = np.max(abs(modelspec.phi[idx]['g'][:, 1]))
    bins = np.linspace(-lim, lim, 11)
    weights.hist(modelspec.phi[idx]['g'][:, 1], bins=bins, color='lightgrey', edgecolor='k')
    weights.set_xlabel('state gain (phi)')
    weights.set_ylabel('n neurons')
    weights.axvline(0, linestyle='--', color='r')

    modelspecs = modelspec.modelspecname.split('-')
    f.suptitle(modelspecs[idx])
    f.tight_layout()


def lv_logsig_plot(rec, modelspec, ax=None, **options):
    """
    Quick view of latent variable model fit. 
    Compare encoding and decoding phi. 
    Encoding weights are purely linear weightings,
    decoding weights allow lv to pass through sigmoid first.
    """

    r = rec.apply_mask()

    f, ax = plt.subplots(1, 1)

    # simple scatter plot of encoding vs. decoding weights
    modelspecs = modelspec.modelspecname.split('-')
    idx = [i for i in range(0, len(modelspecs)) if 'lv.' in modelspecs[i]][0]
    e = modelspec.phi[idx]['e']
    idx = [i for i in range(0, len(modelspecs)) if 'lvlogsig.' in modelspecs[i]][0]
    g = modelspec.phi[idx]['g'][:, 1:]

    for i in range(e.shape[-1]):
        _e = e[:, i]
        _g = g[:, i]
        ulim = np.max(np.concatenate((e, g)))
        llim = np.min(np.concatenate((e, g)))
        ax.scatter(_e, _g, s=30, edgecolor='white', label=r['lv'].chans[i+1])

    ax.legend(fontsize=8)
    ax.plot([-1, 1], [-1, 1], color='grey', linestyle='--')
    ax.axhline(0, linestyle='--', color='grey')
    ax.axvline(0, linestyle='--', color='grey')
    ax.set_xlabel('Encoding weight')
    ax.set_ylabel('Decoding weight')
    ax.set_title(modelspecs[idx])

    f.tight_layout()