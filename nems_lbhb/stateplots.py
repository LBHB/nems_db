#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:05:34 2018

@author: svd
"""
import numpy as np
import matplotlib.pyplot as plt

import nems_db.xform_wrappers as nw
import nems.plots.api as nplt
import nems.xforms as xforms

params = {'legend.fontsize': 8,
          'figure.figsize': (8, 6),
         'axes.labelsize': 8,
         'axes.titlesize': 8,
         'xtick.labelsize': 8,
         'ytick.labelsize': 8}
plt.rcParams.update(params)

def beta_comp(beta1, beta2, n1='model1', n2='model2', hist_bins=20,
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
        set2 = goodcells * 0
    else:
        highlight = np.array(highlight)
        set1 = np.logical_and(goodcells, (highlight))
        set2 = np.logical_and(goodcells, (1-highlight))

    fh = plt.figure(figsize=(6, 6))

    plt.subplot(2, 2, 3)
    plt.plot(np.array(hist_range), np.array([0, 0]), 'k--')
    plt.plot(np.array([0, 0]), np.array(hist_range), 'k--')
    plt.plot(np.array(hist_range), np.array(hist_range), 'k--')
    plt.plot(beta1[set1], beta2[set1], 'k.')
    plt.plot(beta1[set2], beta2[set2], '.', color='lightgray')
    plt.plot(beta1[outcells], beta2[outcells], '.', color='red')
    plt.axis('equal')
    plt.axis('tight')

    plt.xlabel(n1)
    plt.ylabel(n2)
    plt.title(title)

    plt.subplot(2, 2, 1)
    plt.hist([beta1[set1], beta1[set2]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True,
             color=['black','lightgray'])
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta1[goodcells]),
                     np.mean(np.abs(beta1[goodcells]))))
    plt.xlabel(n1)

    ax = plt.subplot(2, 2, 4)
    plt.hist([beta2[set1], beta2[set2]], bins=hist_bins, range=hist_range,
             histtype='bar', stacked=True, orientation="horizontal",
             color=['black','lightgray'])
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta2[goodcells]),
                     np.mean(np.abs(beta2[goodcells]))))
    plt.xlabel(n2)

    ax = plt.subplot(2, 2, 2)
    plt.hist([(beta2[set1]-beta1[set1]) * np.sign(beta2[set1]),
              beta2[set2]-beta1[set2] * np.sign(beta2[set2])],
             bins=hist_bins-1, range=[hist_range[0]/2,hist_range[1]/2],
             histtype='bar', stacked=True,
             color=['black','lightgray'])
    plt.title('mean={:.3f} abs={:.3f}'.
              format(np.mean(beta2[goodcells]),
                     np.mean(np.abs(beta2[goodcells]))))
    plt.xlabel('difference')

    plt.tight_layout()

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


def _state_var_psth_from_epoch_difference(
        rec, epoch='REFERENCE', psth_name='resp', psth_name2='pred',
        state_sig='pupil'):

    full_psth = rec[psth_name]
    folded_psth = full_psth.extract_epoch(epoch)
    if psth_name2 is not None:
        full_psth2 = rec[psth_name2]
        folded_psth2 = full_psth2.extract_epoch(epoch)

    full_var = rec['state'].loc[state_sig]
    folded_var = np.squeeze(full_var.extract_epoch(epoch))

    # compute the mean state for each occurrence
    m = np.nanmean(folded_var, axis=1)

    # compute the mean state across all occurrences
    mean = np.nanmean(m)

    # low = response on epochs when state less than mean
    if np.sum(m < mean):
        low = np.nanmean(folded_psth[m < mean, :, :], axis=0).T
        low2 = np.nanmean(folded_psth2[m < mean, :, :], axis=0).T
    else:
        low = np.ones(folded_psth[0, :, :].shape).T * np.nan
        low2 = np.ones(folded_psth2[0, :, :].shape).T * np.nan

    # high = response on epochs when state less than mean
    title = state_sig
    high = np.nanmean(folded_psth[m >= mean, :, :], axis=0).T
    high2 = np.nanmean(folded_psth2[m >= mean, :, :], axis=0).T

    mod1 = np.sum(high - low) / np.sum(high + low)
    mod2 = np.sum(high2 - low2) / np.sum(high2 + low2)

    return mod1, mod2


def _model_step_plot_old(cellid, batch, modelnames, factors):

    modelname_p0b0, modelname_p0b, modelname_pb0, modelname_pb = \
       modelnames
    factor0, factor1, factor2 = factors

    xf_p0b0, ctx_p0b0 = nw.load_model_baphy_xform(cellid, batch, modelname_p0b0,
                                                  eval_model=False)
    # ctx_p0b0, l = xforms.evaluate(xf_p0b0, ctx_p0b0, stop=-2)

    ctx_p0b0, l = xforms.evaluate(xf_p0b0, ctx_p0b0, start=0, stop=1)

    xf_p0b, ctx_p0b = nw.load_model_baphy_xform(cellid, batch, modelname_p0b,
                                                eval_model=False)
    ctx_p0b['rec'] = ctx_p0b0['rec'].copy()
    ctx_p0b, l = xforms.evaluate(xf_p0b, ctx_p0b, start=1, stop=-2)

    xf_pb0, ctx_pb0 = nw.load_model_baphy_xform(cellid, batch, modelname_pb0,
                                                eval_model=False)
    ctx_pb0['rec'] = ctx_p0b0['rec'].copy()
    ctx_pb0, l = xforms.evaluate(xf_pb0, ctx_pb0, start=1, stop=-2)

    xf_pb, ctx_pb = nw.load_model_baphy_xform(cellid, batch, modelname_pb,
                                              eval_model=False)
    ctx_pb['rec'] = ctx_p0b0['rec'].copy()
    ctx_pb, l = xforms.evaluate(xf_pb, ctx_pb, start=1, stop=-2)

    val = ctx_pb['val'][0].copy()

    # val['pred_p0b0'] = ctx_p0b0['val'][0]['pred'].copy()
    val['pred_p0b'] = ctx_p0b['val'][0]['pred'].copy()
    val['pred_pb0'] = ctx_pb0['val'][0]['pred'].copy()

    state_var_list = val['state'].chans
    col_count = len(state_var_list)

    resp_mod = np.zeros([col_count, 2])
    pred_mod = np.zeros([col_count, 2])
    for i, var in enumerate(state_var_list):
        mod1_p0b, mod2_p0b = _state_var_psth_from_epoch_difference(
                val, epoch="REFERENCE", psth_name="resp",
                psth_name2="pred_p0b", state_sig=var)
        mod1_pb0, mod2_pb0 = _state_var_psth_from_epoch_difference(
                val, epoch="REFERENCE", psth_name="resp",
                psth_name2="pred_pb0", state_sig=var)
        mod1_pb, mod2_pb = _state_var_psth_from_epoch_difference(
                val, epoch="REFERENCE", psth_name="resp",
                psth_name2="pred", state_sig=var)

        resp_mod[i] = np.array([mod1_pb-mod1_p0b, mod1_pb-mod1_pb0])
        pred_mod[i] = np.array([mod2_pb-mod2_p0b, mod2_pb-mod2_pb0])

    fh = plt.figure()
    ax = plt.subplot(4, 1, 1)
    nplt.state_vars_timeseries(val, ctx_pb['modelspecs'][0])
    ax.set_title("{}/{} - {}".format(cellid, batch, modelname_pb))
    ax.set_ylabel("{} r={:.3f}".format(factor0,
                  ctx_p0b0['modelspecs'][0][0]['meta']['r_test']))

    for i, var in enumerate(state_var_list):
        ax = plt.subplot(4, col_count, col_count+i+1)
        nplt.state_var_psth_from_epoch(val, epoch="REFERENCE",
                                       psth_name="resp",
                                       psth_name2="pred_p0b",
                                       state_sig=var, ax=ax)
        if ax.legend_:
            ax.legend_.remove()
        ax.xaxis.label.set_visible(False)
        if i == 0:
            # ax.set_ylabel('Behavior-only', fontsize=10)
            ax.set_ylabel("{} r={:.3f}".format(factor1,
                          ctx_p0b['modelspecs'][0][0]['meta']['r_test']))
            ax.set_title("{} g={:.3f} b={:.3f}"
                         .format(var.lower(),
                                 ctx_p0b['modelspecs'][0][0]['phi']['g'][i],
                                 ctx_p0b['modelspecs'][0][0]['phi']['d'][i]))
        else:
            ax.yaxis.label.set_visible(False)
            ax.set_title("{} g={:.3f} b={:.3f} mod={:.2f}"
                         .format(var.lower(),
                                 ctx_p0b['modelspecs'][0][0]['phi']['g'][i],
                                 ctx_p0b['modelspecs'][0][0]['phi']['d'][i],
                                 pred_mod[i, 0]))

        ax = plt.subplot(4, col_count, col_count*2+i+1)
        nplt.state_var_psth_from_epoch(val, epoch="REFERENCE",
                                       psth_name="resp",
                                       psth_name2="pred_pb0",
                                       state_sig=var, ax=ax)
        if ax.legend_:
            ax.legend_.remove()
        ax.xaxis.label.set_visible(False)
        if i == 0:
            ax.set_ylabel("{} r={:.3f}".format(factor2,
                          ctx_pb0['modelspecs'][0][0]['meta']['r_test']))
            ax.set_title("{} g={:.3f} b={:.3f}"
                         .format(var.lower(),
                                 ctx_pb0['modelspecs'][0][0]['phi']['g'][i],
                                 ctx_pb0['modelspecs'][0][0]['phi']['d'][i]))
        else:
            ax.yaxis.label.set_visible(False)
            ax.set_title("{} g={:.3f} b={:.3f} mod={:.2f}"
                         .format(var.lower(),
                                 ctx_pb0['modelspecs'][0][0]['phi']['g'][i],
                                 ctx_pb0['modelspecs'][0][0]['phi']['d'][i],
                                 pred_mod[i, 1]))

        ax = plt.subplot(4, col_count, col_count*3+i+1)
        nplt.state_var_psth_from_epoch(val, epoch="REFERENCE",
                                       psth_name="resp",
                                       psth_name2="pred",
                                       state_sig=var, ax=ax)
        if i == 0:
            ax.set_ylabel("{} r={:.3f}".format('Full',
                          ctx_pb['modelspecs'][0][0]['meta']['r_test']))
        else:
            ax.yaxis.label.set_visible(False)
        if var == 'active':
            ax.legend(('pas', 'act'))
        ax.set_title("{} g={:.3f} b={:.3f}"
                     .format(var.lower(),
                             ctx_pb['modelspecs'][0][0]['phi']['g'][i],
                             ctx_pb['modelspecs'][0][0]['phi']['d'][i]))

    plt.tight_layout()

    stats = {'cellid': cellid,
             'batch': batch,
             'modelnames': modelnames,
             'state_vars': state_var_list,
             'factors': factors,
             'r_test': np.array([
                     ctx_p0b0['modelspecs'][0][0]['meta']['r_test'],
                     ctx_p0b['modelspecs'][0][0]['meta']['r_test'],
                     ctx_pb0['modelspecs'][0][0]['meta']['r_test'],
                     ctx_pb['modelspecs'][0][0]['meta']['r_test']
                     ]),
             'r_floor': np.array([
                     ctx_p0b0['modelspecs'][0][0]['meta']['r_floor'],
                     ctx_p0b['modelspecs'][0][0]['meta']['r_floor'],
                     ctx_pb0['modelspecs'][0][0]['meta']['r_floor'],
                     ctx_pb['modelspecs'][0][0]['meta']['r_floor']
                     ]),
             'pred_mod': pred_mod.T,
             'g': np.array([
                     ctx_p0b0['modelspecs'][0][0]['phi']['g'],
                     ctx_p0b['modelspecs'][0][0]['phi']['g'],
                     ctx_pb0['modelspecs'][0][0]['phi']['g'],
                     ctx_pb['modelspecs'][0][0]['phi']['g']]),
             'b': np.array([
                     ctx_p0b0['modelspecs'][0][0]['phi']['d'],
                     ctx_p0b['modelspecs'][0][0]['phi']['d'],
                     ctx_pb0['modelspecs'][0][0]['phi']['d'],
                     ctx_pb['modelspecs'][0][0]['phi']['d']])
    }

    return fh, stats


def _model_step_plot(cellid, batch, modelnames, factors):

    modelname_p0b0, modelname_p0b, modelname_pb0, modelname_pb = \
       modelnames
    factor0, factor1, factor2 = factors

    xf_p0b0, ctx_p0b0 = nw.load_model_baphy_xform(cellid, batch, modelname_p0b0,
                                                  eval_model=False)
    # ctx_p0b0, l = xforms.evaluate(xf_p0b0, ctx_p0b0, stop=-2)

    ctx_p0b0, l = xforms.evaluate(xf_p0b0, ctx_p0b0, start=0, stop=-2)

    xf_p0b, ctx_p0b = nw.load_model_baphy_xform(cellid, batch, modelname_p0b,
                                                eval_model=False)
    ctx_p0b, l = xforms.evaluate(xf_p0b, ctx_p0b, start=0, stop=-2)

    xf_pb0, ctx_pb0 = nw.load_model_baphy_xform(cellid, batch, modelname_pb0,
                                                eval_model=False)
    #ctx_pb0['rec'] = ctx_p0b0['rec'].copy()
    ctx_pb0, l = xforms.evaluate(xf_pb0, ctx_pb0, start=0, stop=-2)

    xf_pb, ctx_pb = nw.load_model_baphy_xform(cellid, batch, modelname_pb,
                                              eval_model=False)
    #ctx_pb['rec'] = ctx_p0b0['rec'].copy()
    ctx_pb, l = xforms.evaluate(xf_pb, ctx_pb, start=0, stop=-2)

    val = ctx_pb['val'][0].copy()

    # val['pred_p0b0'] = ctx_p0b0['val'][0]['pred'].copy()
    val['pred_p0b'] = ctx_p0b['val'][0]['pred'].copy()
    val['pred_pb0'] = ctx_pb0['val'][0]['pred'].copy()

    state_var_list = val['state'].chans

    resp_mod = np.zeros([len(state_var_list), 2])
    pred_mod = np.zeros([len(state_var_list), 2])
    pred_mod_full = np.zeros([len(state_var_list), 2])
    for i, var in enumerate(state_var_list):
        mod1_p0b, mod2_p0b = _state_var_psth_from_epoch_difference(
                val, epoch="REFERENCE", psth_name="resp",
                psth_name2="pred_p0b", state_sig=var)
        mod1_pb0, mod2_pb0 = _state_var_psth_from_epoch_difference(
                val, epoch="REFERENCE", psth_name="resp",
                psth_name2="pred_pb0", state_sig=var)
        mod1_pb, mod2_pb = _state_var_psth_from_epoch_difference(
                val, epoch="REFERENCE", psth_name="resp",
                psth_name2="pred", state_sig=var)

        resp_mod[i] = np.array([mod1_pb-mod1_p0b, mod1_pb-mod1_pb0])
        pred_mod[i] = np.array([mod2_pb-mod2_p0b, mod2_pb-mod2_pb0])
        pred_mod_full[i] = np.array([mod2_pb0, mod2_p0b])

    fh = plt.figure()
    ax = plt.subplot(3, 1, 1)
    nplt.state_vars_timeseries(val, ctx_pb['modelspecs'][0])
    ax.set_title("{}/{} - {}".format(cellid, batch, modelname_pb))
    ax.set_ylabel("{} r={:.3f}".format(factor0,
                  ctx_p0b0['modelspecs'][0][0]['meta']['r_test']))

    col_count = len(factors) - 1
    psth_names_ctl = ["pred_p0b", "pred_pb0"]

    for i, var in enumerate(factors[1:]):
        ax = plt.subplot(3, col_count, col_count+i+1)

        nplt.state_var_psth_from_epoch(val, epoch="REFERENCE",
                                       psth_name="resp",
                                       psth_name2=psth_names_ctl[i],
                                       state_sig=var, ax=ax)
        if i == 0:
            ax.set_ylabel("Control model")
            if ax.legend_:
                ax.legend_.remove()
            ax.set_title("{} pred by other vars r={:.3f}"
                         .format(var.lower(),
                                 ctx_p0b['modelspecs'][0][0]['meta']['r_test']))
        else:
            ax.yaxis.label.set_visible(False)
            ax.set_title("{} pred by other vars r={:.3f}"
                         .format(var.lower(),
                                 ctx_pb0['modelspecs'][0][0]['meta']['r_test']))
        ax.xaxis.label.set_visible(False)

        ax = plt.subplot(3, col_count, col_count*2+i+1)
        nplt.state_var_psth_from_epoch(val, epoch="REFERENCE",
                                       psth_name="resp",
                                       psth_name2="pred",
                                       state_sig=var, ax=ax)
        if i == 0:
            ax.set_ylabel("Full Model")
            if ax.legend_:
                ax.legend_.remove()
        else:
            ax.yaxis.label.set_visible(False)

        ax.set_title("{} r={:.3f} rawmod={:.3f} unqmod={:.3f}"
                     .format(var.lower(),
                             ctx_pb['modelspecs'][0][0]['meta']['r_test'],
                             pred_mod_full[i+1][i], pred_mod[i+1][i]))

        if var == 'active':
            ax.legend(('pas', 'act'))

    plt.tight_layout()

    stats = {'cellid': cellid,
             'batch': batch,
             'modelnames': modelnames,
             'state_vars': state_var_list,
             'factors': factors,
             'r_test': np.array([
                     ctx_p0b0['modelspecs'][0][0]['meta']['r_test'],
                     ctx_p0b['modelspecs'][0][0]['meta']['r_test'],
                     ctx_pb0['modelspecs'][0][0]['meta']['r_test'],
                     ctx_pb['modelspecs'][0][0]['meta']['r_test']
                     ]),
             'r_floor': np.array([
                     ctx_p0b0['modelspecs'][0][0]['meta']['r_floor'],
                     ctx_p0b['modelspecs'][0][0]['meta']['r_floor'],
                     ctx_pb0['modelspecs'][0][0]['meta']['r_floor'],
                     ctx_pb['modelspecs'][0][0]['meta']['r_floor']
                     ]),
             'pred_mod': pred_mod.T,
             'pred_mod_full': pred_mod_full.T,
             'g': np.array([
                     ctx_p0b0['modelspecs'][0][0]['phi']['g'],
                     ctx_p0b['modelspecs'][0][0]['phi']['g'],
                     ctx_pb0['modelspecs'][0][0]['phi']['g'],
                     ctx_pb['modelspecs'][0][0]['phi']['g']]),
             'b': np.array([
                     ctx_p0b0['modelspecs'][0][0]['phi']['d'],
                     ctx_p0b['modelspecs'][0][0]['phi']['d'],
                     ctx_pb0['modelspecs'][0][0]['phi']['d'],
                     ctx_pb['modelspecs'][0][0]['phi']['d']])
        }

    return fh, stats


def pb_model_plot(cellid='TAR010c-06-1', batch=301,
                  loader="psth", fitter="basic-nf"):
    """
    test for pupil-behavior interaction.
    loader : string
      can be 'psth' or 'psths'
    fitter : string
      can be 'basic-nf' or 'cd-nf'

    """

    modelname_p0b0 = loader + "20pup0beh0_stategain3_" + fitter
    modelname_p0b = loader + "20pup0beh_stategain3_" + fitter
    modelname_pb0 = loader + "20pupbeh0_stategain3_" + fitter
    modelname_pb = loader + "20pupbeh_stategain3_" + fitter

    factor0 = "baseline"
    factor1 = "pupil"
    factor2 = "active"

    modelnames = [modelname_p0b0, modelname_p0b, modelname_pb0,
                  modelname_pb]
    factors = [factor0, factor1, factor2]

    fh, stats = _model_step_plot(cellid, batch, modelnames, factors)

    return fh, stats


def pp_model_plot(cellid='TAR010c-06-1', batch=301,
                  loader="psth", fitter="basic-nf"):
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

    factor0 = "basline"
    factor1 = "pupil"
    factor2 = "PRE_PASSIVE"

    modelnames = [modelname_p0b0, modelname_p0b, modelname_pb0,
                  modelname_pb]
    factors = [factor0, factor1, factor2]

    fh, stats = _model_step_plot(cellid, batch, modelnames, factors)

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
