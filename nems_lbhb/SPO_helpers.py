#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:00:51 2018

@author: luke
"""
import random

import numpy as np
import matplotlib.pyplot as plt
import nems.db as nd  # NEMS database functions -- NOT celldb
import nems_lbhb.baphy as nb  # baphy-specific functions
import nems_lbhb.xform_wrappers as nw  # wrappers for calling nems code with database stuff
import nems.recording as recording
import nems.plots.api as nplt
import numpy as np
import nems
import nems.preprocessing as preproc
import nems.metrics.api as nmet
import pickle as pl
import pandas as pd
import sys
import os
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems_lbhb.TwoStim_helpers as ts
import nems.epoch as ep
import seaborn as sb
import scipy
import nems_db.params
import warnings
import itertools

sb.color_palette
sb.color_palette('colorblind')

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sb.color_palette('colorblind'))
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
pd.options.display.float_format = '{:,.3f}'.format
pd.set_option('precision', 3)
pd.set_option('display.expand_frame_repr', False)

sys.path.insert(0, '/auto/users/luke/Code/Python/Utilities')
import fitEllipse as fE


def parse_stim_type(stim_name):
    stim_sep = stim_name.split('+')
    if len(stim_sep) == 1:
        stim_type = None
    elif stim_sep[1] == 'null':
        stim_type = 'B'
    elif stim_sep[2] == 'null':
        stim_type = 'A'
    elif stim_sep[1] == stim_sep[2]:
        stim_type = 'C'
    else:
        stim_sep2 = stim_sep[2].split('to')
        if len(stim_sep2) == 1:
            stim_type = 'I'
        elif stim_sep[1] == stim_sep2[0]:
            stim_type = 'CtoI'
        else:
            raise RuntimeError(f"{stim_name} is not parseable")
    return stim_type


def add_stimtype_epochs(sig):
    df0 = sig.epochs.copy()
    df0['name'] = df0['name'].apply(parse_stim_type)
    df0 = df0.loc[df0['name'].notnull()]
    sig.epochs = pd.concat([sig.epochs, df0])
    return sig


def add_stimtype_CtoI_sub_epochs(sig, IncSwitchTime):
    PSS = sig.epochs[sig.epochs['name'] == 'PreStimSilence'].iloc[0]
    prestimtime = PSS['end'] - PSS['start']

    # Make coherent indicies for the time before stimuli switch
    df0 = sig.epochs.copy()
    df0 = df0[df0['name'] == 'CtoI']
    df0['end'] = df0['start'] + prestimtime + IncSwitchTime
    df0['name'] = 'C'

    # Make IafterC indicies for the time after stimuli switch
    df1 = sig.epochs.copy()
    df1 = df1[df1['name'] == 'CtoI']
    df1['start'] = df0['start'] + prestimtime + IncSwitchTime
    df1['name'] = 'IafterC'
    sig.epochs = pd.concat([sig.epochs, df0, df1])
    return sig


def scatterplot_print(x, y, names, ax=None, fn=None, fnargs={}, dv=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if 'marker' not in kwargs:
        kwargs['marker'] = '.'
    if 'linestyle' not in kwargs:
        kwargs['linestyle'] = 'none'
    good_inds = np.where(np.isfinite(x + y))[0]
    x = x[good_inds]
    y = y[good_inds]
    names = [names[g] for g in good_inds]
    if type(fn) is list:
        for i in range(len(fnargs)):
            if 'pth' in fnargs[i].keys():
                fnargs[i]['pth'] = [fnargs[i]['pth'][gi] for gi in good_inds]
    art, = ax.plot(x, y, picker=5, **kwargs)

    # art=ax.scatter(x,y,picker=5,**kwargs)

    def onpick(event):
        if event.artist == art:
            # ind = good_inds[event.ind[0]]
            ind = event.ind[0]
            print('onpick scatter: {}: {} ({},{})'.format(ind, names[ind], np.take(x, ind), np.take(y, ind)))
            if dv is not None:
                dv[0] = names[ind]
            if fn is None:
                print('fn is none?')
            elif type(fn) is list:
                for fni, fna in zip(fn, fnargs):
                    fni(names[ind], **fna)
                    # fni(names[ind],**fna,ind=ind)
            else:
                fn(names[ind], **fnargs)

    def on_plot_hover(event):

        for curve in ax.get_lines():
            if curve.contains(event)[0]:
                print('over {0}'.format(curve.get_gid()))

    ax.figure.canvas.mpl_connect('pick_event', onpick)
    return art

def scatterplot_print_df(dfx, dfy, varnames, dispname = 'pcellid', ax=None, fn=None, fnargs={}, dv=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if 'marker' not in kwargs:
        kwargs['marker'] = '.'
    if 'linestyle' not in kwargs:
        kwargs['linestyle'] = 'none'
    x = dfx[varnames[0]].values
    y = dfy[varnames[0]].values
    good_inds = np.where(np.isfinite(x + y))[0]
    x = x[good_inds]
    y = y[good_inds]
    names = list(dfx[dispname].values[good_inds])
    if type(fn) is list:
        for i in range(len(fnargs)):
            if 'pth' in fnargs[i].keys():
                fnargs[i]['pth'] = [fnargs[i]['pth'][gi] for gi in good_inds]
    art, = ax.plot(x, y, picker=5, **kwargs)

    # art=ax.scatter(x,y,picker=5,**kwargs)

    def onpick(event):
        if event.artist == art:
            # ind = good_inds[event.ind[0]]
            ind = event.ind[0]
            print('onpick scatter: {}: {} ({},{})'.format(ind, names[ind], np.take(x, ind), np.take(y, ind)))
            if dv is not None:
                dv[0] = names[ind]
            if fn is None:
                print('fn is none?')
            elif type(fn) is list:
                for fni, fna in zip(fn, fnargs):
                    if 'data_series_dict' in fna:
                        if fna['data_series_dict'] == 'dsx':
                            fna = dict(dfx.iloc[ind]) | fna
                        elif fna['data_series_dict'] == 'dsy':
                            fna = dict(dfy.iloc[ind]) | fna
                        else:
                            raise RuntimeError('data_series_dict must be either dsx or dsy')
                    fni(**fna)
                    # fni(names[ind],**fna,ind=ind)
            else:
                fn(names[ind], **fnargs)

    def on_plot_hover(event):

        for curve in ax.get_lines():
            if curve.contains(event)[0]:
                print('over {0}'.format(curve.get_gid()))

    ax.figure.canvas.mpl_connect('pick_event', onpick)
    return art


def load(rec, batch, cellid, meta, **context):
    eval_conds=[['A'],['B'],['C'],['I'],['A','B'],['C','I']]
    rec['resp'] = add_stimtype_epochs(rec['resp'])
    if len(ep.epoch_occurrences(rec.epochs, 'CtoI')) > 0:
        manager = BAPHYExperiment(batch=batch, cellid=cellid[0])
        exptparams = manager.get_baphy_exptparams()[0]
        meta['IncSwitchTime'] = exptparams['TrialObject'][1]['ReferenceHandle'][1]['IncSwitchTime']
        rec['resp'] = add_stimtype_CtoI_sub_epochs(rec['resp'], meta['IncSwitchTime'])
    else:
        meta['IncSwitchTime'] = None

    return {'rec':  rec, 'meta':meta, 'evaluation_conditions':eval_conds}

def split_by_occurrence_counts_SPO(rec, epoch_regex='^STIM',**context):
    if False:
        #Just going by rep numbers sometimes is wrong when a few extra reps were recorded
        N_per_epoch = ep.epoch_occurrences(rec.epochs, epoch_regex)
        est_mask = N_per_epoch < N_per_epoch.max() / 9 #makes sure Square stimuli go into val set
        epochs_for_est = N_per_epoch.index.values[est_mask]
        epochs_for_val = N_per_epoch.index.values[~est_mask]
        square_epochs = ep.epoch_names_matching(rec.epochs, '.*Square')
        #epochs_for_val = [ep for ep in epochs_for_val if ep not in square_epochs]
    else:
        #Just do it explicitly
        all_epochs = ep.epoch_names_matching(rec.epochs, '^STIM')
        epochs_for_val = ['STIM_T+si464+null', 'STIM_T+null+si464', 'STIM_T+si464+si464',
                      'STIM_T+si516+null', 'STIM_T+null+si516', 'STIM_T+si516+si516',
                      'STIM_T+si464+si516', 'STIM_T+si516+si464',
                      'STIM_T+si464+si464tosi516', 'STIM_T+si516+si516tosi464']
        square_epochs = ep.epoch_names_matching(rec.epochs, '.*Square')
        # epochs_for_val = epochs_for_val + square_epochs 10/7/21: Don't include square epochs in val, make a new recording for them later as needed
        epochs_for_est = set(all_epochs) - set(epochs_for_val) - set(square_epochs)
    est, val = rec.split_by_epochs(epochs_for_est, epochs_for_val)
    return {'est': est, 'val': val}

def mask_out_Squares(val, **context):
    square_epochs = ep.epoch_names_matching(val.epochs, '.*Square')
    return {'val': val.and_mask(epoch=square_epochs,invert=True)}


def add_coherence_as_state(rec, permute=False, baseline=True, **context):
    coh = rec['resp'].epoch_to_signal('C')
    inc = rec['resp'].epoch_to_signal('I')
    if permute:
        coh = coh.shuffle_time(rand_seed=0, mask=rec['mask'])
        inc = inc.shuffle_time(rand_seed=1, mask=rec['mask'])
    rec = preproc.concatenate_state_channel(rec, coh, state_signal_name='state', generate_baseline=baseline)
    rec = preproc.concatenate_state_channel(rec, inc, state_signal_name='state')
    rec = preproc.concatenate_state_channel(rec, coh, state_signal_name='state_raw', generate_baseline=baseline)
    rec = preproc.concatenate_state_channel(rec, inc, state_signal_name='state_raw')
    chans = ['Coherent', 'Incoherent']
    if baseline:
        chans.insert(0, 'baseline')
    rec.signals['state'].chans = chans
    rec.signals['state_raw'].chans = chans



def plot_all_vals_(modelspec, val, figures=None, IsReload=False, **context):
    if figures is None:
        figures = []
    if not IsReload:
        for i in range(len(val['resp'].chans)):
            fig = plot_all_vals(val[0], modelspec, IncSwitchTime = modelspec[0]['meta']['IncSwitchTime'],
                                channels=[i,i])
            # Needed to make into a Bytes because you can't deepcopy figures!
            figures.append(nplt.fig2BytesIO(fig))

    return {'figures': figures}


def plot_linear_and_weighted_psths_model(modelspec, val, rec, figures=None, IsReload=False, **context):
    if figures is None:
        figures = []
    if not IsReload:
        SR = 0  # SR=get_SR(rec)
        fig, _, _ = plot_linear_and_weighted_psths(val, SR, signame='pred', subset='C+I', addsig='resp')
        phi = modelspec[1]['phi']
        if 'g' in phi:
            g = phi['g'].copy()
            if g.shape[1] == 2:
                gn = np.hstack((np.full((2, 1), 1), g)) + 1
            else:
                gn = g / g[:, :1] + 1
            yl = fig.axes[0].get_ylim()
            th = fig.axes[0].text(fig.axes[0].get_xlim()[1], yl[1] + .2 * np.diff(yl),
                                  '     gain  \nA: {: .2f} \nB: {: .2f} \nA-B: {: .2f} '.format(
                                      gn[0][2], gn[1][2], gn[0][2] - gn[1][2]),
                                  verticalalignment='top', horizontalalignment='right')
            th2 = fig.axes[2].text(fig.axes[0].get_xlim()[1], yl[1] + 0 * np.diff(yl),
                                   '     gain  \nA: {: .2f} \nB: {: .2f} \nA-B: {: .2f} '.format(
                                       gn[0][1], gn[1][1], gn[0][1] - gn[1][1]),
                                   verticalalignment='top', horizontalalignment='right')
        fig.axes[0].set_title('{}: {}'.format(modelspec[0]['meta']['cellid'], modelspec[0]['meta']['modelname']))
        # Needed to make into a Bytes because you can't deepcopy figures!
        figures.append(nplt.fig2BytesIO(fig))

    return {'figures': figures}

def load_SPO(pcellid, fit_epochs, modelspec_name, loader='env100',
             modelspecs_dir='/auto/users/luke/Code/nems/modelspecs', fs=100, get_est=True, get_stim=True):
    import glob
    import nems.analysis.api
    import nems.modelspec as ms
    import warnings
    import nems.recording as recording
    import nems.preprocessing as preproc
    import pandas as pd
    import copy

    batch = 306

    if get_stim:
        loadkey = 'env.fs'
    else:
        loadkey = 'ns.fs100'

    # load into a recording object
    # recname = '/auto/data/nems_db/recordings/' + str(batch) + '/envelope0_fs100/' + pcellid +'.tgz'
    rec_file = nw.generate_recording_uri(pcellid, batch, loadkey=loadkey, force_old_loader=False)
    rec = recording.load_recording(rec_file)
    rec['resp'].fs = fs
    rec['resp'] = rec['resp'].extract_channels([pcellid])
    # ----------------------------------------------------------------------------
    # DATA PREPROCESSING
    #
    # GOAL: Split your data into estimation and validation sets so that you can
    #       know when your model exhibits overfitting.

    # Method #1: Find which stimuli have the most reps, use those for val
    # if not get_stim:
    #    del rec.signals['stim']
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')

    # Optional: Take nanmean of ALL occurrences of all signals
    if get_est:
        est = preproc.average_away_epoch_occurrences(est, epoch_regex='^STIM_')
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')

    if get_est:
        df0 = est['resp'].epochs.copy()
        df2 = est['resp'].epochs.copy()
        df0['name'] = df0['name'].apply(parse_stim_type)
        df0 = df0.loc[df0['name'].notnull()]
        df3 = pd.concat([df0, df2])

        est['resp'].epochs = df3
        est_sub = copy.deepcopy(est)
        est_sub['resp'] = est_sub['resp'].select_epochs(fit_epochs)
    else:
        est_sub = None

    df0 = val['resp'].epochs.copy()
    df2 = val['resp'].epochs.copy()
    df0['name'] = df0['name'].apply(parse_stim_type)
    df0 = df0.loc[df0['name'].notnull()]
    df3 = pd.concat([df0, df2])

    val['resp'].epochs = df3
    val_sub = copy.deepcopy(val)
    val_sub['resp'] = val_sub['resp'].select_epochs(fit_epochs)

    # ----------------------------------------------------------------------------
    # GENERATE SUMMARY STATISTICS

    if modelspec_name is None:
        return None, [est_sub], [val_sub]
    else:
        fit_epochs_str = "+".join([str(x) for x in fit_epochs])
        mn = loader + '_subset_' + fit_epochs_str + '.' + modelspec_name
        an_ = modelspecs_dir + '/' + pcellid + '/' + mn
        an = glob.glob(an_ + '*')
        if len(an) > 1:
            warnings.warn('{} models found, loading an[0]:{}'.format(len(an), an[0]))
            an = [an[0]]
        if len(an) == 1:
            filepath = an[0]
            modelspecs = [ms.load_modelspec(filepath)]
            modelspecs[0][0]['meta']['modelname'] = mn
            modelspecs[0][0]['meta']['cellid'] = pcellid
        else:
            raise RuntimeError('not fit')
        # generate predictions
        est_sub, val_sub = nems.analysis.api.generate_prediction(est_sub, val_sub, modelspecs)
        est_sub, val_sub = nems.analysis.api.generate_prediction(est_sub, val_sub, modelspecs)

        return modelspecs, est_sub, val_sub


def plot_all_vals(val, modelspec, signames=['resp', 'pred'], channels=[0, 0, 0], subset=None,
                  plot_singles_on_dual=False, IncSwitchTime=None):
    # NOTE TO SELF: Not sure why channels=[0,0,1]. Setting it as default, but when called by plot_linear_and_weighted_psths it should be [0,0,0]
    from nems.plots.timeseries import timeseries_from_epoch
    import matplotlib.pyplot as plt
    import numpy as np
    from cycler import cycler
    if val[signames[0]].count_epoch('REFERENCE'):
        epochname = 'REFERENCE'
    else:
        epochname = 'TRIAL'
    extracted = val[signames[0]].extract_epoch(epochname)
    finite_trial = [np.sum(np.isfinite(x)) > 0 for x in extracted]
    occurrences, = np.where(finite_trial)

    epochs = val[signames[0]].epochs
    epochs = epochs[epochs['name'] == epochname].iloc[occurrences]
    st_mask = val[signames[0]].epochs['name'].str.contains('ST')
    inds = []
    for index, row in epochs.iterrows():
        matchi = (val[signames[0]].epochs['start'] == row['start']) & (val[signames[0]].epochs['end'] == row['end'])
        matchi = matchi & st_mask
        inds.append(np.where(matchi)[0][0])

    names = val[signames[0]].epochs['name'].iloc[inds].tolist()

    A = [];
    B = [];
    for name in names:
        nm = name.split('+')
        A.append(nm[1])
        B.append(nm[2])

    if subset == 'C+I' and IncSwitchTime is not None:
        subset = 'CtoI+I'
    if subset is None:
        plot_order = ['STIM_T+si464+null', 'STIM_T+null+si464', 'STIM_T+si464+si464',
                      'STIM_T+si516+null', 'STIM_T+null+si516', 'STIM_T+si516+si516',
                      'STIM_T+si464+si516', 'STIM_T+si516+si464']
        if IncSwitchTime is not None:
            plot_order[2] = 'STIM_T+si464+si464tosi516'
            plot_order[5] = 'STIM_T+si516+si516tosi464'
    elif subset == 'C+I':
        plot_order = ['STIM_T+si464+si464', 'STIM_T+si516+si516',
                      'STIM_T+si464+si516', 'STIM_T+si516+si464']
    elif subset == 'CtoI+I':
        plot_order = ['STIM_T+si464+si464tosi516', 'STIM_T+si516+si516tosi464',
                      'STIM_T+si464+si516', 'STIM_T+si516+si464']
    elif subset == 'squares':
        plot_oder = []
        # square_epochs = ep.epoch_names_matching(val.epochs, '.*Square')
        plot_order = ['STIM_T+Square_0_2+null', 'STIM_T+null+Square_0_2', 'STIM_T+Square_0_2+Square_0_2',
                      'STIM_T+Square_0_2+Square_1_3', 'STIM_T+Square_1_3+Square_0_2']
    elif subset == 'squaresOverlap':
        plot_oder = []
        # square_epochs = ep.epoch_names_matching(val.epochs, '.*Square')
        plot_order = [['STIM_T+Square_0_2+null', 'STIM_T+null+Square_0_2'], ['STIM_T+Square_0_2+Square_0_2',
                                                                             'STIM_T+Square_0_2+Square_1_3'],
                      'STIM_T+Square_1_3+Square_0_2']

    # OVERWRITE PLOT ORDER TO BE WHAT YOU WANT:
    # plot_order=['STIM_T+si464+null', 'STIM_T+null+si464','STIM_T+si464+si464']

    plot_order.reverse()
    order = np.array([names.index(nm) for nm in plot_order])
    names_short = [
        n.replace('STIM_T+', '').replace('si464', '1').replace('si516', '2').replace('null', '_').replace('Square',
                                                                                                          'sq') for n in
        names]
    #    names2=sorted(names,key=lambda x: plot_order.index(x))

    #   idmap = dict((id,pos) for pos,id in enumerate(plot_order))

    sigs = [val[s] for s in signames]
    title = ''
    nplt = len(plot_order)
    gs_kw = dict(hspace=0, left=0.06, right=.99)
    fig, ax = plt.subplots(nrows=nplt, ncols=1, figsize=(10, 15), sharey=True, gridspec_kw=gs_kw)
    if signames == ['resp', 'lin_model']:
        [axi.set_prop_cycle(cycler('color', ['k', 'g']) + cycler(linestyle=['-', 'dotted']) + cycler(linewidth=[1, 2]))
         for axi in ax]
    else:
        [axi.set_prop_cycle(cycler('color', ['k', '#1f77b4', 'r']) + cycler('linestyle', ['-', '-', '--'])) for axi in
         ax]

    def minmax(x): return np.nanmin(x),np.nanmax(x)
    mm = [minmax(s.as_continuous()[channels[0], :]) for s in sigs]
    yl = [np.nanmin(np.array(mm)[:,0]), np.nanmax(np.array(mm)[:,1])]
    stimname = 'stim'  # LAS was this
    #stimname = 'resp'
    prestimtime = val[stimname].epochs.loc[0].end

    for i in range(nplt):
        timeseries_from_epoch(sigs, epochname, title=title,
                              occurrences=occurrences[order[i]], ax=ax[i], channels=channels, linestyle=None,
                              linewidth=None)
        if names_short[order[i]] in ['1+_', '2+_']:
            # timeseries_from_epoch([val['stim']], epochname, title=title,
            #             occurrences=occurrences[order[i]],ax=ax[i])

            ep = val[stimname].extract_epoch(names[order[i]]).squeeze()
            ep = 80 + 20 * np.log10(ep.T)
            ep = ep / ep.max() * yl[1]
            time_vector = np.arange(0, len(ep)) / val[stimname].fs
            ax[i].plot(time_vector - prestimtime, ep, '--', color='#ff7f0e')
        if plot_singles_on_dual:
            snA = names_short[order[i]][:2] + '_'
            snB = '_' + names_short[order[i]][1:]
            snA_ = names[names_short.index(snA)]
            epA = val['resp'].extract_epoch(snA_).squeeze()
            time_vector = np.arange(0, len(epA)) / val['resp'].fs
            ax[i].plot(time_vector - prestimtime, epA, '--', color=(1, .5, 0), linewidth=1.5)
            if 'to' in snB:
                snB_ = snB.split('to')
                snB_part1 = names[names_short.index(snB_[0])]
                epB = val['resp'].extract_epoch(snB_part1).squeeze()
                time = time_vector - prestimtime
                pi = time < IncSwitchTime
                ax[i].plot(time[pi], epB[pi], '--', color=(0, .5, 1), linewidth=1.5)

                snB_part2 = names[names_short.index('_+' + snB_[1])]
                epB = val['resp'].extract_epoch(snB_part2).squeeze()
                pi = time >= IncSwitchTime
                ax[i].plot(time[pi], epB[pi], '--', color=(.1, .7, 1), linewidth=1.5)
            else:
                snB_ = names[names_short.index(snB)]
                epB = val['resp'].extract_epoch(snB_).squeeze()
                ax[i].plot(time_vector - prestimtime, epB, '--', color=(0, .5, 1), linewidth=1.5)

        ax[i].set_ylabel(names_short[order[i]], rotation=0, horizontalalignment='right', verticalalignment='bottom')

    if modelspec is not None:
        ax[0].set_title('{}: {}'.format(sigs[0].chans[channels[0]], modelspec[0]['meta']['modelname']))
    [axi.get_xaxis().set_visible(False) for axi in ax[:-1]]
    [axi.get_yaxis().set_ticks([]) for axi in ax]
    [axi.get_legend().set_visible(False) for axi in ax[:-1]]
    [axi.set_xlim([.8 - 1, 4.5 - 1]) for axi in ax]
    yl_margin = .01 * (yl[1] - yl[0])
    [axi.set_ylim((yl[0] - yl_margin, yl[1] + yl_margin)) for axi in ax]
    [axi.set_ylim((yl[0] - yl_margin, yl[1] + yl_margin)) for axi in ax]
    if IncSwitchTime is not None:
        for i in range(nplt):
            if 'to' in names_short[order[i]]:
                ax[i].plot(np.repeat(IncSwitchTime, 2), (yl[0] - yl_margin, yl[1] + yl_margin), color=(.8, .8, .8))
    if plot_singles_on_dual:
        ls = ['resp A', 'resp B']
    else:
        ls = ['log(stim)']
    ax[nplt - 1].legend(signames + ls)
    return fig


def smooth(x, window_len=11, passes=2, window='flat'):
    import numpy as np
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)


    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = s
    for passnum in range(passes):
        y = np.convolve(w / w.sum(), y, mode='valid')
    return y


def export_all_vals(val, modelspec, signames=['resp', 'pred']):
    from nems.plots.timeseries import timeseries_from_epoch
    import matplotlib.pyplot as plt
    import numpy as np
    from cycler import cycler
    if val[signames[0]].count_epoch('REFERENCE'):
        epochname = 'REFERENCE'
    else:
        epochname = 'TRIAL'
    extracted = val[signames[0]].extract_epoch(epochname)
    finite_trial = [np.sum(np.isfinite(x)) > 0 for x in extracted]
    occurrences, = np.where(finite_trial)

    epochs = val[signames[0]].epochs
    epochs = epochs[epochs['name'] == epochname].iloc[occurrences]
    st_mask = val[signames[0]].epochs['name'].str.contains('ST')
    inds = []
    for index, row in epochs.iterrows():
        matchi = (val[signames[0]].epochs['start'] == row['start']) & (val[signames[0]].epochs['end'] == row['end'])
        matchi = matchi & st_mask
        inds.append(np.where(matchi)[0][0])

    names = val[signames[0]].epochs['name'].iloc[inds].tolist()

    A = [];
    B = [];
    for name in names:
        nm = name.split('+')
        A.append(nm[1])
        B.append(nm[2])

    plot_order = ['STIM_T+si464+null', 'STIM_T+null+si464', 'STIM_T+si464+si464',
                  'STIM_T+si516+null', 'STIM_T+null+si516', 'STIM_T+si516+si516',
                  'STIM_T+si464+si516', 'STIM_T+si516+si464']
    plot_order.reverse()
    order = np.array([names.index(nm) for nm in plot_order])
    names_short = [n.replace('STIM_T+', '').replace('si464', '1').replace('si516', '2').replace('null', '_') for n in
                   names]
    #    names2=sorted(names,key=lambda x: plot_order.index(x))

    #   idmap = dict((id,pos) for pos,id in enumerate(plot_order))

    nplt = len(occurrences)
    ep = []
    for i in range(nplt):
        ep.append(val['pred'].extract_epoch(names[order[i]]).squeeze())

    ep_ = val['resp'].fs * np.array(ep)
    dd = '/auto/users/luke/Code/nems/modelspecs/svd_fs_branch/'
    pth = dd + modelspec[0]['meta']['cellid'] + '/' + modelspec[0]['meta']['modelname']
    np.save(pth + '.npy', ep_)
    from pdb import set_trace;
    set_trace()


def calc_psth_metrics(batch, cellid, rec_file=None):
    import nems.db as nd  # NEMS database functions -- NOT celldb
    import nems_lbhb.baphy as nb  # baphy-specific functions
    import nems_lbhb.xform_wrappers as nw  # wrappers for calling nems code with database stuff
    import nems.recording as recording
    import numpy as np
    import nems.preprocessing as preproc
    import nems.metrics.api as nmet
    import nems.metrics.corrcoef
    import copy

    options = {}
    # options['cellid']=cellid
    # options['batch']=batch
    # options["stimfmt"] = "envelope"
    # options["chancount"] = 0
    # options["rasterfs"] = 100
    # rec_file=nb.baphy_data_path(options)

    # Get Baphy params
    manager = BAPHYExperiment(batch=batch, cellid=cellid)
    exptparams = manager.get_baphy_exptparams()[0]
    try:
        IncSwitchTime = exptparams['TrialObject'][1]['ReferenceHandle'][1]['IncSwitchTime']
        types = ['A', 'B', 'CtoI', 'I']
        types_2s = ['CtoI', 'I']
    except:
        IncSwitchTime = None
        types = ['A', 'B', 'C', 'I']
        types_2s = ['C', 'I']

    if rec_file is None:
        rec_file = nw.generate_recording_uri(cellid, batch, loadkey='ns.fs100',
                                             force_old_loader=False)  # 'was 'env.fs100'
    # uri = nb.baphy_load_recording_uri(cellid=cellid, batch=batch, **options)
    rec = recording.load_recording(rec_file)
    rec['resp'] = rec['resp'].extract_channels([cellid])
    rec['resp'].fs = 200

    epcs = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PreStimSilence'].copy()
    ep2 = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PostStimSilence'].iloc[0].copy()
    prestim = epcs.iloc[0]['end']
    poststim = ep2['end'] - ep2['start']

    spike_times = rec['resp']._data[cellid]
    count = 0
    for index, row in epcs.iterrows():
        count += np.sum((spike_times > row['start']) & (spike_times < row['end']))
    SR = count / (epcs['end'] - epcs['start']).sum()

    resp = rec['resp'].rasterize()
    resp = add_stimtype_epochs(resp)
    ps = resp.select_epochs(['PreStimSilence']).as_continuous()
    ff = np.isfinite(ps)
    SR_rast = ps[ff].mean() * resp.fs
    SR_std = ps[ff].std() * resp.fs

    # COMPUTE ALL FOLLOWING metrics using smoothed driven rate
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')
    epochs_to_extract = ep.epoch_names_matching(val.epochs, '^STIM_')
    folded_matrices = val['resp'].extract_epochs(epochs_to_extract)
    stds = np.array(0)
    for k in folded_matrices.keys():
        if np.sum(~np.isnan(folded_matrices[k])) > 0:
            stds_ = np.nanstd(folded_matrices[k], axis=0)
            stds = np.append(stds, stds_)
    AV_STD = np.mean(stds)
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')

    # smooth and subtract SR
    fn = lambda x: np.atleast_2d(smooth(x.squeeze(), 3, 2) - SR / resp.fs)
    val['resp'] = val['resp'].transform(fn)
    val['resp'] = add_stimtype_epochs(val['resp'])

    if val['resp'].count_epoch('REFERENCE'):
        epochname = 'REFERENCE'
    else:
        epochname = 'TRIAL'
    sts = val['resp'].epochs['start'].copy()
    nds = val['resp'].epochs['end'].copy()
    sts_rec = rec['resp'].epochs['start'].copy()
    val['resp'].epochs['end'] = val['resp'].epochs['start'] + prestim
    ps = val['resp'].select_epochs([epochname]).as_continuous()
    ff = np.isfinite(ps)
    SR_av = ps[ff].mean() * resp.fs
    SR_av_std = ps[ff].std() * resp.fs
    val['resp'].epochs['end'] = nds

    # restore times
    val['resp'].epochs['end'] = nds
    val['resp'].epochs['start'] = sts
    val['resp'].epochs['start'] = val['resp'].epochs['start'] + prestim
    TotalMax = np.nanmax(val['resp'].as_continuous())
    ps = np.hstack((val['resp'].extract_epoch('A').flatten(), val['resp'].extract_epoch('B').flatten()))
    SinglesMax = np.nanmax(ps)

    # Change epochs to stimulus steady-state times
    val['resp'].epochs['end'] = nds
    val['resp'].epochs['start'] = sts
    val['resp'].epochs['start'] = val['resp'].epochs['start'] + prestim + .5
    val['resp'].epochs['end'] = val['resp'].epochs['end'] - poststim

    thresh = np.array(((SR + SR_av_std) / resp.fs,
                       (SR - SR_av_std) / resp.fs))
    thresh = np.array((SR / resp.fs + 0.1 * (SinglesMax - SR / resp.fs),
                       (SR - SR_av_std) / resp.fs))
    # SR/resp.fs - 0.5 * (np.nanmax(val['resp'].as_continuous()) - SR/resp.fs)]

    excitatory_percentage = {}
    inhibitory_percentage = {}
    Max = {}
    Mean = {}
    for _type in types:
        ps = val['resp'].extract_epoch(_type).flatten()
        ff = np.isfinite(ps)
        excitatory_percentage[_type] = (ps[ff] > thresh[0]).sum() / ff.sum()
        inhibitory_percentage[_type] = (ps[ff] < thresh[1]).sum() / ff.sum()
        Max[_type] = ps[ff].max() / SinglesMax
        Mean[_type] = ps[ff].mean()

    # restore times
    val['resp'].epochs['end'] = nds
    val['resp'].epochs['start'] = sts
    # Change epochs to stimulus onset times
    val['resp'].epochs['start'] = val['resp'].epochs['start'] + prestim
    val['resp'].epochs['end'] = val['resp'].epochs['start'] + prestim + .5
    # types=['A','B','C','I']
    excitatory_percentage_onset = {}
    inhibitory_percentage_onset = {}
    Max_onset = {}
    for _type in types:
        ps = val['resp'].extract_epoch(_type).flatten()
        ff = np.isfinite(ps)
        excitatory_percentage_onset[_type] = (ps[ff] > thresh[0]).sum() / ff.sum()
        inhibitory_percentage_onset[_type] = (ps[ff] < thresh[1]).sum() / ff.sum()
        Max_onset[_type] = ps[ff].max() / SinglesMax

        # restore times
    val['resp'].epochs['end'] = nds
    val['resp'].epochs['start'] = sts
    val['resp'].epochs['start'] = val['resp'].epochs['start'] + prestim
    rec['resp'].epochs['start'] = rec['resp'].epochs['start'] + prestim
    # over stim on time to end + 0.5
    val['linmodel'] = val['resp'].copy()
    val['linmodel']._data = np.full(val['linmodel']._data.shape, np.nan)
    # types=['C','I']
    epcs = val['resp'].epochs[val['resp'].epochs['name'].str.contains('STIM')].copy()
    epcs['type'] = epcs['name'].apply(parse_stim_type)
    EA = np.array([n.split('+')[1] for n in epcs['name']])
    EB = np.array([n.split('+')[2] for n in epcs['name']])
    r_dual_B = {};
    r_dual_A = {};
    r_dual_B_nc = {};
    r_dual_A_nc = {};
    r_dual_B_bal = {};
    r_dual_A_bal = {}
    r_lin_B = {};
    r_lin_A = {};
    r_lin_B_nc = {};
    r_lin_A_nc = {};
    r_lin_B_bal = {};
    r_lin_A_bal = {}
    N_ac = 200
    full_resp = rec['resp'].rasterize()
    full_resp = full_resp.transform(fn)

    for _type in types_2s:
        inds = np.nonzero(epcs['type'].values == _type)[0]
        rA_st = [];
        rB_st = [];
        r_st = [];
        rA_rB_st = [];
        init = True
        for ind in inds:
            r = val['resp'].extract_epoch(epcs.iloc[ind]['name'])
            if np.any(np.isfinite(r)):
                print(epcs.iloc[ind]['name'])
                indA = np.where((EA[ind] == EA) & (EB == 'null'))[0]
                if _type == 'CtoI':
                    EBparts = EB[ind].split('to')
                    indB = np.where((EBparts[0] == EB) & (EA == 'null'))[0]
                else:
                    indB = np.where((EB[ind] == EB) & (EA == 'null'))[0]
                if (len(indA) > 0) & (len(indB) > 0):
                    rA = val['resp'].extract_epoch(epcs.iloc[indA[0]]['name'])
                    rB = val['resp'].extract_epoch(epcs.iloc[indB[0]]['name'])
                    r_st_ = full_resp.extract_epoch(epcs.iloc[ind]['name'])[:, 0, :]
                    if _type == 'CtoI':
                        rmi = np.arange(IncSwitchTime * resp.fs, r.shape[2])
                        r[:, :, rmi] = np.inf
                        r_st_[:, rmi] = np.inf
                    else:
                        rmi = []
                    r_st.append(r_st_)
                    rA_st_ = full_resp.extract_epoch(epcs.iloc[indA[0]]['name'])[:, 0, :]
                    rB_st_ = full_resp.extract_epoch(epcs.iloc[indB[0]]['name'])[:, 0, :]
                    rA_st.append(rA_st_)
                    rB_st.append(rB_st_)
                    minreps = np.min((rA_st_.shape[0], rB_st_.shape[0]))
                    rA_rB_st.append(rA_st_[:minreps, :] + rB_st_[:minreps, :])
                    if init:
                        rA_ = rA.squeeze();
                        rB_ = rB.squeeze();
                        r_ = r.squeeze();
                        rA_rB_ = rA.squeeze() + rB.squeeze()
                        init = False
                    else:
                        rA_ = np.hstack((rA_, rA.squeeze()))
                        rB_ = np.hstack((rB_, rB.squeeze()))
                        r_ = np.hstack((r_, r.squeeze()))
                        rA_rB_ = np.hstack((rA_rB_, rA.squeeze() + rB.squeeze()))
                    lin_model = rA + rB
                    lin_model[:, :, rmi] = np.inf
                    val['linmodel'] = val['linmodel'].replace_epoch(epcs.iloc[ind]['name'], lin_model,
                                                                    preserve_nan=False)
        ff = np.isfinite(r_) & np.isfinite(rA_) & np.isfinite(rB_)
        r_dual_A[_type] = np.corrcoef(rA_[ff], r_[ff])[0, 1]
        r_dual_B[_type] = np.corrcoef(rB_[ff], r_[ff])[0, 1]
        r_lin_A[_type] = np.corrcoef(rA_[ff], rA_rB_[ff])[0, 1]
        r_lin_B[_type] = np.corrcoef(rB_[ff], rA_rB_[ff])[0, 1]

        minreps = np.min([x.shape[0] for x in r_st + rA_st + rB_st])
        r_st = [x[:minreps, :] for x in r_st]
        r_st = np.concatenate(r_st, axis=1)
        rA_st = [x[:minreps, :] for x in rA_st]
        rA_st = np.concatenate(rA_st, axis=1)
        rB_st = [x[:minreps, :] for x in rB_st]
        rB_st = np.concatenate(rB_st, axis=1)
        rA_rB_st = [x[:minreps, :] for x in rA_rB_st]
        rA_rB_st = np.concatenate(rA_rB_st, axis=1)

        r_lin_A_bal[_type] = np.corrcoef(rA_st[0::2, ff].mean(axis=0), rA_rB_st[1::2, ff].mean(axis=0))[0, 1]
        r_lin_B_bal[_type] = np.corrcoef(rB_st[0::2, ff].mean(axis=0), rA_rB_st[1::2, ff].mean(axis=0))[0, 1]
        r_dual_A_bal[_type] = np.corrcoef(rA_st[0::2, ff].mean(axis=0), r_st[:, ff].mean(axis=0))[0, 1]
        r_dual_B_bal[_type] = np.corrcoef(rB_st[0::2, ff].mean(axis=0), r_st[:, ff].mean(axis=0))[0, 1]

        r_dual_A_nc[_type] = r_noise_corrected(rA_st[:, ff], r_st[:, ff])
        r_dual_B_nc[_type] = r_noise_corrected(rB_st[:, ff], r_st[:, ff])
        r_lin_A_nc[_type] = r_noise_corrected(rA_st[:, ff], rA_rB_st[:, ff])
        r_lin_B_nc[_type] = r_noise_corrected(rB_st[:, ff], rA_rB_st[:, ff])

        if ['C', 'CtoI'].count(_type) > 0:
            r_A_B = np.corrcoef(rA_[ff], rB_[ff])[0, 1]
            r_A_B_nc = r_noise_corrected(rA_st[:, ff], rB_st[:, ff])
            rAA = nems.metrics.corrcoef._r_single(rA_st[:, ff], 200, 0)
            rBB = nems.metrics.corrcoef._r_single(rB_st[:, ff], 200, 0)
            rCC = nems.metrics.corrcoef._r_single(r_st[:, ff], 200, 0)
            Np = 100
            rAA_nc = np.zeros(Np)
            rBB_nc = np.zeros(Np)
            hv = int(minreps / 2);
            for i in range(Np):
                inds = np.random.permutation(minreps)
                rAA_nc[i] = r_noise_corrected(rA_st[inds[:hv]], rA_st[inds[hv:]])
                rBB_nc[i] = r_noise_corrected(rB_st[inds[:hv]], rB_st[inds[hv:]])
            ffA = np.isfinite(rAA_nc)
            ffB = np.isfinite(rBB_nc)
            rAAm = rAA_nc[ffA].mean()
            rBBm = rBB_nc[ffB].mean()
            mean_nsA = rA_st.sum(axis=1).mean()
            mean_nsB = rB_st.sum(axis=1).mean()
            min_nsA = rA_st.sum(axis=1).min()
            min_nsB = rB_st.sum(axis=1).min()
        else:
            rII = nems.metrics.corrcoef._r_single(r_st, 200, 0)
        # rac = _r_single(X, N)
        # r_ceiling = [nmet.r_ceiling(p, rec, 'pred', 'resp') for p in val_copy]

    # Calculate correlation between linear 'model and dual-voice response, and mean amount of suppression, enhancement relative to linear 'model'
    r_fit_linmodel = {}
    r_fit_linmodel_NM = {}
    r_ceil_linmodel = {}
    mean_enh = {}
    mean_supp = {}
    EnhP = {}
    SuppP = {}
    DualAboveZeroP = {}
    resp_ = copy.deepcopy(rec['resp'].rasterize())
    resp_.epochs['start'] = sts_rec
    fn = lambda x: np.atleast_2d(smooth(x.squeeze(), 3, 2) - SR / val['resp'].fs)
    resp_ = resp_.transform(fn)
    for _type in types_2s:
        val_copy = copy.deepcopy(val)
        val_copy['resp'] = val_copy['resp'].select_epochs([_type])
        # Correlation between linear 'model' (response to A plus response to B) and dual-voice response
        r_fit_linmodel_NM[_type] = nmet.corrcoef(val_copy, 'linmodel', 'resp')
        # r_ceil_linmodel[_type] = nems.metrics.corrcoef.r_ceiling(val_copy,rec,'linmodel', 'resp',exclude_neg_pred=False)[0]
        # Noise-corrected correlation between linear 'model' (response to A plus response to B) and dual-voice response
        r_ceil_linmodel[_type] = nems.metrics.corrcoef.r_ceiling(val_copy, rec, 'linmodel', 'resp')[0]

        pred = val_copy['linmodel'].as_continuous()
        resp = val_copy['resp'].as_continuous()
        ff = np.isfinite(pred) & np.isfinite(resp)
        # cc = np.corrcoef(smooth(pred[ff],3,2), smooth(resp[ff],3,2))
        cc = np.corrcoef(pred[ff], resp[ff])
        r_fit_linmodel[_type] = cc[0, 1]

        prdiff = resp[ff] - pred[ff]
        mean_enh[_type] = prdiff[prdiff > 0].mean() * val['resp'].fs
        mean_supp[_type] = prdiff[prdiff < 0].mean() * val['resp'].fs

        Njk = 10
        if _type == 'C':
            stims = ['STIM_T+si464+si464', 'STIM_T+si516+si516']
        elif _type == 'CtoI':
            stims = ['STIM_T+si464+si464tosi516', 'STIM_T+si516+si516tosi464']
        else:
            stims = ['STIM_T+si464+si516', 'STIM_T+si516+si464']
        T = int(700 + prestim * val['resp'].fs)
        Tps = int(prestim * val['resp'].fs)
        jns = np.zeros((Njk, T, len(stims)))
        for ns in range(len(stims)):
            for njk in range(Njk):
                resp_jn = resp_.jackknife_by_epoch(Njk, njk, stims[ns])
                jns[njk, :, ns] = np.nanmean(resp_jn.extract_epoch(stims[ns]), axis=0)
        jns = np.reshape(jns[:, Tps:, :], (Njk, 700 * len(stims)), order='F')

        lim_models = np.zeros((700, len(stims)))
        for ns in range(len(stims)):
            lim_models[:, ns] = val_copy['linmodel'].extract_epoch(stims[ns])
        lim_models = lim_models.reshape(700 * len(stims), order='F')

        ff = np.isfinite(lim_models)
        mean_diff = (jns[:, ff] - lim_models[ff]).mean(axis=0)
        std_diff = (jns[:, ff] - lim_models[ff]).std(axis=0)
        serr_diff = np.sqrt(Njk / (Njk - 1)) * std_diff

        thresh = 3
        dual_above_zero = (jns[:, ff].mean(axis=0) > std_diff)
        sig_enh = ((mean_diff / serr_diff) > thresh) & dual_above_zero
        sig_supp = ((mean_diff / serr_diff) < -thresh)
        DualAboveZeroP[_type] = (dual_above_zero).sum() / len(mean_diff)
        EnhP[_type] = (sig_enh).sum() / len(mean_diff)
        SuppP[_type] = (sig_supp).sum() / len(mean_diff)

    #        time = np.arange(0, lim_models.shape[0])/ val['resp'].fs
    #        plt.figure();
    #        plt.plot(time,jns.mean(axis=0),'.-k');
    #        plt.plot(time,lim_models,'.-g');
    #        plt.plot(time[sig_enh],lim_models[sig_enh],'.r')
    #        plt.plot(time[sig_supp],lim_models[sig_supp],'.b')
    #        plt.title('Type:{:s}, Enh:{:.2f}, Sup:{:.2f}, Resp_above_zero:{:.2f}'.format(_type,EnhP[_type],SuppP[_type],DualAboveZeroP[_type]))
    #        from pdb import set_trace
    #        set_trace()
    #        a=2
    # thrsh=5
    #        EnhP[_type] = ((prdiff*val['resp'].fs) > thresh).sum()/len(prdiff)
    #        SuppP[_type] = ((prdiff*val['resp'].fs) < -thresh).sum()/len(prdiff)
    #    return val
    #    return {'excitatory_percentage':excitatory_percentage,
    #            'inhibitory_percentage':inhibitory_percentage,
    #            'r_fit_linmodel':r_fit_linmodel,
    #            'SR':SR, 'SR_std':SR_std, 'SR_av_std':SR_av_std}
    #
    return {'thresh': thresh * val['resp'].fs,
            'EP_A': excitatory_percentage['A'],
            'EP_B': excitatory_percentage['B'],
            'EP_C': excitatory_percentage[types_2s[0]],
            'EP_I': excitatory_percentage['I'],
            'IP_A': inhibitory_percentage['A'],
            'IP_B': inhibitory_percentage['B'],
            'IP_C': inhibitory_percentage[types_2s[0]],
            'IP_I': inhibitory_percentage['I'],
            'OEP_A': excitatory_percentage_onset['A'],
            'OEP_B': excitatory_percentage_onset['B'],
            'OEP_C': excitatory_percentage_onset[types_2s[0]],
            'OEP_I': excitatory_percentage_onset['I'],
            'OIP_A': inhibitory_percentage_onset['A'],
            'OIP_B': inhibitory_percentage_onset['B'],
            'OIP_C': inhibitory_percentage_onset[types_2s[0]],
            'OIP_I': inhibitory_percentage_onset['I'],
            'Max_A': Max['A'],
            'Max_B': Max['B'],
            'Max_C': Max[types_2s[0]],
            'Max_I': Max['I'],
            'Mean_A': Mean['A'],
            'Mean_B': Mean['B'],
            'Mean_C': Mean[types_2s[0]],
            'Mean_I': Mean['I'],
            'OMax_A': Max_onset['A'],
            'OMax_B': Max_onset['B'],
            'OMax_C': Max_onset[types_2s[0]],
            'OMax_I': Max_onset['I'],
            'TotalMax': TotalMax * val['resp'].fs,
            'SinglesMax': SinglesMax * val['resp'].fs,
            'r_lin_C': r_fit_linmodel[types_2s[0]],
            'r_lin_I': r_fit_linmodel['I'],
            'r_lin_C_NM': r_fit_linmodel_NM[types_2s[0]],
            'r_lin_I_NM': r_fit_linmodel_NM['I'],
            'r_ceil_C': r_ceil_linmodel[types_2s[0]],
            'r_ceil_I': r_ceil_linmodel['I'],
            'MEnh_C': mean_enh[types_2s[0]],
            'MEnh_I': mean_enh['I'],
            'MSupp_C': mean_supp[types_2s[0]],
            'MSupp_I': mean_supp['I'],
            'EnhP_C': EnhP[types_2s[0]],
            'EnhP_I': EnhP['I'],
            'SuppP_C': SuppP[types_2s[0]],
            'SuppP_I': SuppP['I'],
            'DualAboveZeroP_C': DualAboveZeroP[types_2s[0]],
            'DualAboveZeroP_I': DualAboveZeroP['I'],
            'r_dual_A_C': r_dual_A[types_2s[0]],
            'r_dual_A_I': r_dual_A['I'],
            'r_dual_B_C': r_dual_B[types_2s[0]],
            'r_dual_B_I': r_dual_B['I'],
            'r_dual_A_C_nc': r_dual_A_nc[types_2s[0]],
            'r_dual_A_I_nc': r_dual_A_nc['I'],
            'r_dual_B_C_nc': r_dual_B_nc[types_2s[0]],
            'r_dual_B_I_nc': r_dual_B_nc['I'],
            'r_dual_A_C_bal': r_dual_A_bal[types_2s[0]],
            'r_dual_A_I_bal': r_dual_A_bal['I'],
            'r_dual_B_C_bal': r_dual_B_bal[types_2s[0]],
            'r_dual_B_I_bal': r_dual_B_bal['I'],
            'r_lin_A_C': r_lin_A[types_2s[0]],
            'r_lin_A_I': r_lin_A['I'],
            'r_lin_B_C': r_lin_B[types_2s[0]],
            'r_lin_B_I': r_lin_B['I'],
            'r_lin_A_C_nc': r_lin_A_nc[types_2s[0]],
            'r_lin_A_I_nc': r_lin_A_nc['I'],
            'r_lin_B_C_nc': r_lin_B_nc[types_2s[0]],
            'r_lin_B_I_nc': r_lin_B_nc['I'],
            'r_lin_A_C_bal': r_lin_A_bal[types_2s[0]],
            'r_lin_A_I_bal': r_lin_A_bal['I'],
            'r_lin_B_C_bal': r_lin_B_bal[types_2s[0]],
            'r_lin_B_I_bal': r_lin_B_bal['I'],
            'r_A_B': r_A_B,
            'r_A_B_nc': r_A_B_nc,
            'rAAm': rAAm, 'rBBm': rBBm,
            'rAA': rAA, 'rBB': rBB, 'rCC': rCC, 'rII': rII,
            'rAA_nc': rAA_nc, 'rBB_nc': rBB_nc,
            'mean_nsA': mean_nsA, 'mean_nsB': mean_nsB, 'min_nsA': min_nsA, 'min_nsB': min_nsB,
            'SR': SR, 'SR_std': SR_std, 'SR_av_std': SR_av_std}


def r_noise_corrected(X, Y, N_ac=200):
    import nems.metrics.corrcoef
    Xac = nems.metrics.corrcoef._r_single(X, N_ac, 0)
    Yac = nems.metrics.corrcoef._r_single(Y, N_ac, 0)
    repcount = X.shape[0]
    rs = np.zeros((repcount, repcount))
    for nn in range(repcount):
        for mm in range(repcount):
            X_ = X[mm, :]
            Y_ = Y[nn, :]
            # remove all nans from pred and resp
            ff = np.isfinite(X_) & np.isfinite(Y_)

            if (np.sum(X_[ff]) != 0) and (np.sum(Y_[ff]) != 0):
                rs[nn, mm] = np.corrcoef(X_[ff], Y_[ff])[0, 1]
            else:
                rs[nn, mm] = 0
    # rs=rs[np.triu_indices(rs.shape[0],1)]
    # plt.figure(); plt.imshow(rs)
    return np.mean(rs) / (np.sqrt(Xac) * np.sqrt(Yac))


def calc_psth_metrics_orig(batch, cellid):
    import nems.db as nd  # NEMS database functions -- NOT celldb
    import nems_lbhb.baphy as nb  # baphy-specific functions
    import nems_lbhb.xform_wrappers as nw  # wrappers for calling nems code with database stuff
    import nems.recording as recording
    import numpy as np
    import nems.preprocessing as preproc
    import nems.metrics.api as nmet
    import copy

    options = {}
    options['cellid'] = cellid
    options['batch'] = batch
    options["stimfmt"] = "envelope"
    options["chancount"] = 0
    options["rasterfs"] = 100
    rec_file = nb.baphy_data_path(options)
    rec = recording.load_recording(rec_file)
    rec['resp'].fs = 200

    epcs = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PreStimSilence'].copy()
    spike_times = rec['resp']._data[options['cellid']]
    count = 0
    for index, row in epcs.iterrows():
        count += np.sum((spike_times > row['start']) & (spike_times < row['end']))
    SR = count / (epcs['end'] - epcs['start']).sum()

    resp = rec['resp'].rasterize()
    resp = add_stimtype_epochs(resp)
    ps = resp.select_epochs(['PreStimSilence']).as_continuous()
    ff = np.isfinite(ps)
    SR_rast = ps[ff].mean() * resp.fs
    SR_std = ps[ff].std() * resp.fs

    # COMPUTE ALL FOLLOWING metrics using smoothed driven rate
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')
    from pdb import set_trace
    set_trace()
    # smooth and subtract SR
    fn = lambda x: np.atleast_2d(smooth(x.squeeze(), 3, 2) - SR / resp.fs)
    val['resp'] = val['resp'].transform(fn)
    val['resp'] = add_stimtype_epochs(val['resp'])

    sts = val['resp'].epochs['start'].copy()
    nds = val['resp'].epochs['end'].copy()
    val['resp'].epochs['end'] = val['resp'].epochs['start'] + 1
    ps = val['resp'].select_epochs(['TRIAL']).as_continuous()
    ff = np.isfinite(ps)
    SR_av = ps[ff].mean() * resp.fs
    SR_av_std = ps[ff].std() * resp.fs
    val['resp'].epochs['end'] = nds

    # restore times
    val['resp'].epochs['end'] = nds
    val['resp'].epochs['start'] = sts
    val['resp'].epochs['start'] = val['resp'].epochs['start'] + 1
    TotalMax = np.nanmax(val['resp'].as_continuous())
    ps = np.hstack((val['resp'].extract_epoch('A').flatten(), val['resp'].extract_epoch('B').flatten()))
    SinglesMax = np.nanmax(ps)

    # Change epochs to stimulus ss times
    val['resp'].epochs['end'] = nds
    val['resp'].epochs['start'] = sts
    val['resp'].epochs['start'] = val['resp'].epochs['start'] + 1.5
    val['resp'].epochs['end'] = val['resp'].epochs['end'] - 0.5
    # types=['A','B','C','I']
    thresh = np.array(((SR + SR_av_std) / resp.fs,
                       (SR - SR_av_std) / resp.fs))
    thresh = np.array((SR / resp.fs + 0.1 * (SinglesMax - SR / resp.fs),
                       (SR - SR_av_std) / resp.fs))
    # SR/resp.fs - 0.5 * (np.nanmax(val['resp'].as_continuous()) - SR/resp.fs)]

    excitatory_percentage = {}
    inhibitory_percentage = {}
    Max = {}
    for _type in types:
        ps = val['resp'].extract_epoch(_type).flatten()
        ff = np.isfinite(ps)
        excitatory_percentage[_type] = (ps[ff] > thresh[0]).sum() / ff.sum()
        inhibitory_percentage[_type] = (ps[ff] < thresh[1]).sum() / ff.sum()
        Max[_type] = ps[ff].max() / SinglesMax

        # restore times
    val['resp'].epochs['end'] = nds
    val['resp'].epochs['start'] = sts
    # Change epochs to stimulus onset times
    val['resp'].epochs['start'] = val['resp'].epochs['start'] + 1
    val['resp'].epochs['end'] = val['resp'].epochs['start'] + 1.5
    # types=['A','B','C','I']
    excitatory_percentage_onset = {}
    inhibitory_percentage_onset = {}
    Max_onset = {}
    for _type in types:
        ps = val['resp'].extract_epoch(_type).flatten()
        ff = np.isfinite(ps)
        excitatory_percentage_onset[_type] = (ps[ff] > thresh[0]).sum() / ff.sum()
        inhibitory_percentage_onset[_type] = (ps[ff] < thresh[1]).sum() / ff.sum()
        Max_onset[_type] = ps[ff].max() / SinglesMax

        # restore times
    val['resp'].epochs['end'] = nds
    val['resp'].epochs['start'] = sts
    val['resp'].epochs['start'] = val['resp'].epochs['start'] + 1

    # over stim on time to end + 0.5
    val['linmodel'] = val['resp'].copy()
    val['linmodel']._data = np.full(val['linmodel']._data.shape, np.nan)
    # types=['CtoI','I']
    epcs = val['resp'].epochs[val['resp'].epochs['name'].str.contains('STIM')].copy()
    epcs['type'] = epcs['name'].apply(parse_stim_type)
    EA = np.array([n.split('+')[1] for n in epcs['name']])
    EB = np.array([n.split('+')[2] for n in epcs['name']])
    for _type in types_2s:
        inds = np.nonzero(epcs['type'].values == _type)[0]
        for ind in inds:
            r = val['resp'].extract_epoch(epcs.iloc[ind]['name'])
            if np.any(np.isfinite(r)):
                indA = np.where((EA[ind] == EA) & (EB == 'null'))[0]
                indB = np.where((EB[ind] == EB) & (EA == 'null'))[0]
                if (len(indA) > 0) & (len(indB) > 0):
                    rA = val['resp'].extract_epoch(epcs.iloc[indA[0]]['name'])
                    rB = val['resp'].extract_epoch(epcs.iloc[indB[0]]['name'])
                    val['linmodel'] = val['linmodel'].replace_epoch(epcs.iloc[ind]['name'], rA + rB, preserve_nan=False)

    r_fit_linmodel = {}
    mean_enh = {}
    mean_supp = {}
    EnhP = {}
    SuppP = {}
    for _type in types_2s:
        val_copy = copy.deepcopy(val)
        val_copy['resp'] = val_copy['resp'].select_epochs(_type)
        # r_fit_linmodel[_type] = nmet.corrcoef(val_copy, 'linmodel', 'resp')
        pred = val_copy['linmodel'].as_continuous()
        resp = val_copy['resp'].as_continuous()
        ff = np.isfinite(pred) & np.isfinite(resp)
        # cc = np.corrcoef(smooth(pred[ff],3,2), smooth(resp[ff],3,2))
        cc = np.corrcoef(pred[ff], resp[ff])
        r_fit_linmodel[_type] = cc[0, 1]

        prdiff = resp[ff] - pred[ff]
        mean_enh[_type] = prdiff[prdiff > 0].mean() * val['resp'].fs
        mean_supp[_type] = prdiff[prdiff < 0].mean() * val['resp'].fs

        resp = rec['resp'].rasterize()
        resp_jn = resp.jackknife_by_epoch(10, 0, 'STIM_T+si464+si464')
        resp_jn = rec.jackknife_by_epoch(10, 0, 'STIM_T+si464+si464')

        val['resp'].extract_epoch('STIM_T+si464+si464')

        resp_jn.np.zeros(900, 10)
        Njk = 10
        jns = np.zeros(900, Njk, ken(stims))
        for ns in range(len(stims)):
            for resp_jn in resp.jackknifes_by_epoch(10, stims[ns]):
                resp_jn[njk, :, ns] = resp_jn.extract_epoch(stims[ns]).mean(axis=1)

        thresh = 5
        EnhP[_type] = ((prdiff * val['resp'].fs) > thresh).sum() / len(prdiff)
        SuppP[_type] = ((prdiff * val['resp'].fs) < -thresh).sum() / len(prdiff)
    #    return val
    #    return {'excitatory_percentage':excitatory_percentage,
    #            'inhibitory_percentage':inhibitory_percentage,
    #            'r_fit_linmodel':r_fit_linmodel,
    #            'SR':SR, 'SR_std':SR_std, 'SR_av_std':SR_av_std}
    #
    return {'thresh': thresh * val['resp'].fs,
            'EP_A': excitatory_percentage['A'],
            'EP_B': excitatory_percentage['B'],
            'EP_C': excitatory_percentage['C'],
            'EP_I': excitatory_percentage['I'],
            'IP_A': inhibitory_percentage['A'],
            'IP_B': inhibitory_percentage['B'],
            'IP_C': inhibitory_percentage['C'],
            'IP_I': inhibitory_percentage['I'],
            'OEP_A': excitatory_percentage_onset['A'],
            'OEP_B': excitatory_percentage_onset['B'],
            'OEP_C': excitatory_percentage_onset['C'],
            'OEP_I': excitatory_percentage_onset['I'],
            'OIP_A': inhibitory_percentage_onset['A'],
            'OIP_B': inhibitory_percentage_onset['B'],
            'OIP_C': inhibitory_percentage_onset['C'],
            'OIP_I': inhibitory_percentage_onset['I'],
            'Max_A': Max['A'],
            'Max_B': Max['B'],
            'Max_C': Max['C'],
            'Max_I': Max['I'],
            'OMax_A': Max_onset['A'],
            'OMax_B': Max_onset['B'],
            'OMax_C': Max_onset['C'],
            'OMax_I': Max_onset['I'],
            'TotalMax': TotalMax * val['resp'].fs,
            'SinglesMax': SinglesMax * val['resp'].fs,
            'r_lin_C': r_fit_linmodel['C'],
            'r_lin_I': r_fit_linmodel['I'],
            'MEnh_C': mean_enh['C'],
            'MEnh_I': mean_enh['I'],
            'MSupp_C': mean_supp['C'],
            'MSupp_I': mean_supp['I'],
            'EnhP_C': EnhP['C'],
            'EnhP_I': EnhP['I'],
            'SuppP_C': SuppP['C'],
            'SuppP_I': SuppP['I'],
            'SR': SR, 'SR_std': SR_std, 'SR_av_std': SR_av_std}


def type_by_psth(row, prefix=''):
    t = ['X', 'X']
    thresh = .05
    if row[prefix + 'EP_A'] < thresh and row[prefix + 'IP_A'] < thresh:
        t[0] = 'O'
    elif row[prefix + 'EP_A'] >= thresh:
        t[0] = 'E'
    else:
        t[0] = 'I'
    if row[prefix + 'EP_B'] < thresh and row[prefix + 'IP_B'] < thresh:
        t[1] = 'O'
    elif row[prefix + 'EP_B'] >= thresh:
        t[1] = 'E'
    else:
        t[1] = 'I'

    if t.count('E') == 2:  # EE
        if row[prefix + 'EP_A'] > row[prefix + 'EP_B']:
            inds = np.array((0, 1))
        else:
            inds = np.array((1, 0))
    elif t.count('I') == 2:  # II
        if row[prefix + 'IP_A'] > row[prefix + 'IP_B']:
            inds = np.array((0, 1))
        else:
            inds = np.array((1, 0))
    elif t[0] == 'E' and t[1] == 'I':  # EI
        inds = np.array((0, 1))
    elif t[0] == 'I' and t[1] == 'E':  # IE
        inds = np.array((1, 0))
        t = ['E', 'I']
    elif t[0] == 'E' and t[1] == 'O':  # EO
        inds = np.array((0, 1))
    elif t[0] == 'O' and t[1] == 'E':  # OE
        inds = np.array((1, 0))
        t = ['E', 'O']
    elif t.count('O') == 2:  # OO
        if row[prefix + 'Max_A'] > row[prefix + 'Max_B']:
            inds = np.array((0, 1))
        else:
            inds = np.array((1, 0))
    else:
        # t = ['ERROR']
        # inds = None
        raise RuntimeError('Unknown type {}'.format(t))
    row[prefix + 'Rtype'] = ''.join(t)
    row[prefix + 'inds'] = inds
    # return pd.Series({'Rtype': ''.join(t), 'inds': inds})
    return row


def calc_psth_weights_of_model_responses_list(val, names, signame='pred', do_plot=False, find_mse_confidence=True,
                                              get_nrmse_fn=True):
    PSS = val[signame].epochs[val[signame].epochs['name'] == 'PreStimSilence'].iloc[0]
    prestimtime = PSS['end'] - PSS['start']
    REF = val[signame].epochs[val[signame].epochs['name'] == 'REFERENCE'].iloc[0]
    total_duration = REF['end'] - REF['start']
    POSS = val[signame].epochs[val[signame].epochs['name'] == 'PostStimSilence'].iloc[0]
    poststimtime = POSS['end'] - POSS['start']
    duration = total_duration - prestimtime - poststimtime
    post_duration_pad = .5
    time = np.arange(0, val[signame].extract_epoch(names[0][0]).shape[-1]) / val[signame].fs - prestimtime
    xc_win = (time > 0) & (time < (duration + post_duration_pad))
    # names = [ [n[0]] for n in names]
    sig1 = np.concatenate([val[signame].extract_epoch(n).squeeze()[xc_win] for n in names[0]])
    sig2 = np.concatenate([val[signame].extract_epoch(n).squeeze()[xc_win] for n in names[1]])
    # sig_SR=np.ones(sig1.shape)
    sigO = np.concatenate([val[signame].extract_epoch(n).squeeze()[xc_win] for n in names[2]])

    # fsigs=np.vstack((sig1,sig2,sig_SR)).T
    fsigs = np.vstack((sig1, sig2)).T
    ff = np.all(np.isfinite(fsigs), axis=1) & np.isfinite(sigO)
    close_to_zero = np.array([np.allclose(fsigs[ff, i], 0, atol=1e-17) for i in (0, 1)])
    if any(close_to_zero):
        weights_, residual_sum, rank, singular_values = np.linalg.lstsq(np.expand_dims(fsigs[ff, ~close_to_zero], 1),
                                                                        sigO[ff], rcond=None)
        weights = np.zeros(2)
        weights[~close_to_zero] = weights_
    else:
        weights, residual_sum, rank, singular_values = np.linalg.lstsq(fsigs[ff, :], sigO[ff], rcond=None)

        # calc CC
    sigF2 = np.dot(weights, fsigs[ff, :].T)
    cc = np.corrcoef(sigF2, sigO[ff])
    r_weight_model = cc[0, 1]

    norm_factor = np.std(sigO[ff])

    min_nrmse = np.sqrt(residual_sum[0] / ff.sum()) / norm_factor
    # create NMSE caclulator for later
    if get_nrmse_fn:
        def get_nrmse(weights=weights):
            sigF2 = np.dot(weights, fsigs[ff, :].T)
            nrmse = np.sqrt(((sigF2 - sigO[ff]) ** 2).mean(axis=-1)) / norm_factor
            return nrmse
    else:
        get_nrmse = np.nan

    if not find_mse_confidence:
        weights[close_to_zero] = np.nan
        return weights, np.nan, min_nrmse, norm_factor, get_nrmse, r_weight_model

    #    sigF=weights[0]*sig1 + weights[1]*sig2 + weights[2]
    #    plt.figure();
    #    plt.plot(np.vstack((sig1,sig2,sigO,sigF)).T)
    #    wA_ = np.linspace(-2, 4, 100)
    #    wB_ = np.linspace(-2, 4, 100)
    #    wA, wB = np.meshgrid(wA_,wB_)
    #    w=np.vstack((wA.flatten(),wB.flatten())).T
    #    sigF2=np.dot(w,fsigs[ff,:].T)
    #    mse = ((sigF2-sigO[ff].T) ** 2).mean(axis=1)
    #    mse = np.reshape(mse,(len(wA_),len(wB_)))
    #    plt.figure();plt.imshow(mse,interpolation='none',extent=[wA_[0],wA_[-1],wB_[0],wB_[-1]],origin='lower',vmax=.02,cmap='viridis_r');plt.colorbar()

    def calc_nrmse_matrix(margin, N=60, threshtype='ReChance'):
        # wsearcha=(-2, 4, 100)
        # wsearchb=wsearcha
        # margin=6
        if not hasattr(margin, "__len__"):
            margin = np.float(margin) * np.ones(2)
        wA_ = np.hstack((np.linspace(weights[0] - margin[0], weights[0], N),
                         (np.linspace(weights[0], weights[0] + margin[0], N)[1:])))
        wB_ = np.hstack((np.linspace(weights[1] - margin[1], weights[1], N),
                         (np.linspace(weights[1], weights[1] + margin[1], N)[1:])))
        wA, wB = np.meshgrid(wA_, wB_)
        w = np.stack((wA, wB), axis=2)
        nrmse = get_nrmse(w)
        # range_=mse.max()-mse.min()
        if threshtype == 'Absolute':
            thresh = nrmse.min() * np.array((1.4, 1.6))
            thresh = nrmse.min() * np.array((1.02, 1.04))
            As = wA[(nrmse < thresh[1]) & (nrmse > thresh[0])]
            Bs = wB[(nrmse < thresh[1]) & (nrmse > thresh[0])]
        elif threshtype == 'ReChance':
            thresh = 1 - (1 - nrmse.min()) * np.array((.952, .948))
            As = wA[(nrmse < thresh[1]) & (nrmse > thresh[0])]
            Bs = wB[(nrmse < thresh[1]) & (nrmse > thresh[0])]
        return nrmse, As, Bs, wA_, wB_

    if min_nrmse < 1:
        this_threshtype = 'ReChance'
    else:
        this_threshtype = 'Absolute'
    margin = 6
    As = np.zeros(0)
    Bs = np.zeros(0)
    attempt = 0
    did_estimate = False
    while len(As) < 20:
        attempt += 1
        if (attempt > 1) and (len(As) > 0) and (len(As) > 2) and (not did_estimate):
            margin = np.float(margin) * np.ones(2)
            m = np.abs(weights[0] - As).max() * 3
            if m == 0:
                margin[0] = margin[0] / 2
            else:
                margin[0] = m

            m = np.abs(weights[1] - Bs).max() * 3
            if m == 0:
                margin[1] = margin[1] / 2
            else:
                margin[1] = m
            did_estimate = True
        elif attempt > 1:
            margin = margin / 2
        if attempt > 1:
            print('Attempt {}, margin = {}'.format(attempt, margin))
        nrmse, As, Bs, wA_, wB_ = calc_nrmse_matrix(margin, threshtype=this_threshtype)

        if attempt == 8:
            print('Too many attempts, break')
            break

    try:
        efit = fE.fitEllipse(As, Bs)
        center = fE.ellipse_center(efit)
        phi = fE.ellipse_angle_of_rotation(efit)
        axes = fE.ellipse_axis_length(efit)

        epars = np.hstack((center, axes, phi))
    except:
        print('Error fitting ellipse: {}'.format(sys.exc_info()[0]))
        print(sys.exc_info()[0])
        epars = np.full([5], np.nan)
    #    idxA = (np.abs(wA_ - weights[0])).argmin()
    #    idxB = (np.abs(wB_ - weights[1])).argmin()
    if do_plot:
        plt.figure();
        plt.imshow(nrmse, interpolation='none', extent=[wA_[0], wA_[-1], wB_[0], wB_[-1]], origin='lower',
                   cmap='viridis_r');
        plt.colorbar()
        ph = plt.plot(weights[0], weights[1], Color='k', Marker='.')
        plt.plot(As, Bs, 'r.')

        if not np.isnan(epars).any():
            a, b = axes
            R = np.arange(0, 2 * np.pi, 0.01)
            xx = center[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
            yy = center[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)
            plt.plot(xx, yy, color='k')

    #    plt.figure();plt.plot(get_nrmse(weights=(xx,yy)))
    #    plt.figure();plt.plot(get_nrmse(weights=(As,Bs)))
    weights[close_to_zero] = np.nan
    return weights, epars, nrmse.min(), norm_factor, get_nrmse, r_weight_model


def calc_psth_weights_of_model_responses(val, signame='pred', do_plot=False, find_mse_confidence=True,
                                         get_nrmse_fn=True, exptparams=None):
    # weights_C=np.ones((2,3))
    # names=['STIM_T+si464+null','STIM_T+null+si464','STIM_T+si464+si464']
    # weights_C[0,:]=calc_psth_weights_of_model_responses_single(val,names)
    # names=['STIM_T+si516+null','STIM_T+null+si516','STIM_T+si516+si516']
    # weights_C[1,:]=calc_psth_weights_of_model_responses_single(val,names)

    namesC = [['STIM_T+si464+null', 'STIM_T+si516+null'],
              ['STIM_T+null+si464', 'STIM_T+null+si516'],
              ['STIM_T+si464+si464', 'STIM_T+si516+si516']]

    try:
        IncSwitchTime = exptparams['TrialObject'][1]['ReferenceHandle'][1]['IncSwitchTime']
        namesC[2] = ['STIM_T+si464+si464tosi516', 'STIM_T+si516+si516tosi464']
        window = [0, IncSwitchTime]  # Use this window to calculate weight model
    except:
        IncSwitchTime = None
        window = None

    weights_C, Efit_C, nrmse_C, nf_C, get_nrmse_C, r_C, get_error_C = ts.calc_psth_weights_of_model_responses_list(
        val, namesC, signame, do_plot=do_plot, find_mse_confidence=find_mse_confidence, get_nrmse_fn=get_nrmse_fn,
        window=window)
    if do_plot and find_mse_confidence:
        plt.title('Coherent, signame={}'.format(signame))

    names = [['STIM_T+si464+null', 'STIM_T+si516+null'],
             ['STIM_T+null+si516', 'STIM_T+null+si464'],
             ['STIM_T+si464+si516', 'STIM_T+si516+si464']]
    weights_I, Efit_I, nrmse_I, nf_I, get_nrmse_I, r_I, get_error_I = ts.calc_psth_weights_of_model_responses_list(
        val, names, signame, do_plot=do_plot, find_mse_confidence=find_mse_confidence, get_nrmse_fn=get_nrmse_fn)
    if do_plot and find_mse_confidence:
        plt.title('Incoherent, signame={}'.format(signame))

    D = locals()
    D = {k: D[k] for k in ('weights_C', 'Efit_C', 'nrmse_C', 'nf_C', 'get_nrmse_C', 'r_C', 'get_error_C',
                           'weights_I', 'Efit_I', 'nrmse_I', 'nf_I', 'get_nrmse_I', 'r_I', 'get_error_I')}
    return D
    # return weights_C, Efit_C, nmse_C, nf_C, get_mse_C, weights_I, Efit_I, nmse_I, nf_I, get_mse_I


def test(**kwds):
    print(kwds.keys())

def show_img(cellid, ax=None, ft=1, subset='A+B+C+I', modelspecname='dlog_fir2x15_lvl1_dexp1',
             loader='env.fs100-ld-sev-subset.A+B+C+I', fitter='fit_basic', pth=None,
             ind=None, modelname=None, fignum=0, batch=306, modelpath=None, **extras):
    ax_ = None
    if pth is None:
        print('pth is None')
        if ft == 0:
            pth = '/auto/users/luke/Projects/SPS/plots/NEMS/types/'
            pth = pth + cellid + '_env100_subset_' + subset + '.' + modelspecname + '_all_val+FIR.png'
        elif ft == 1:
            pth = '/auto/users/luke/Projects/SPS/plots/NEMS/types/PSTH/'
            pth = pth + cellid + '.png'
        elif ft == 11:
            pth = '/auto/users/luke/Projects/SPS/plots/NEMS/types/PSTH/Overlay/'
            pth = pth + cellid + '.png'
        elif ft == 2:
            pth = '/auto/users/luke/Projects/SPS/plots/NEMS/types/PSTH/'
            pth = pth + cellid + '.pickle'
        elif ft == 3:
            pth = '/auto/users/luke/Projects/SPS/plots/NEMS/normalization_branch/'
            # subset='A+B+C+I'
            # subset='I'
            pth = pth + cellid + '_env100_subset_' + subset + '.' + modelspecname + '.png'
            if type(ax) == list:
                print('list!')
                ax_ = ax
                ax = ax_[0]
                pth2 = pth.replace('.png', '_all_val.png')
            else:
                pth = pth.replace('.png', '_all_val.png')
        elif ft == 4:
            pth = '/auto/users/luke/Projects/SPS/plots/NEMS/svd_fs_branch/'
            # subset='A+B+C+I'
            # subset='I'
            # pth=pth + cellid + '_env.fs100-ld-sev-st.coh-subset.'+subset+'_'+modelspecname+'_all_val.png'
            # pth=pth + cellid + '_'+loader+'_'+modelspecname+'_all_val.png'
            if len(fitter) == 0:
                pth = pth + cellid + '_' + loader + '_' + modelspecname + '.png'
            else:
                pth = pth + cellid + '_' + loader + '_' + modelspecname + '_' + fitter + '.png'
            if type(ax) == list:
                print('list!')
                ax_ = ax
                ax = ax_[0]
                pth2 = pth.replace('.png', '_all_val.png')
            else:
                pth = pth.replace('.png', '_all_val.png')
        elif ft == 5:
                pth = os.path.join(nd.get_results_file(batch, [modelname], [cellid])['modelpath'][0],
                               'figure.{:04d}.png'.format(fignum))
        elif ft == 6:
            pth = '/auto/users/luke/Projects/SPS/plots/NEMS/types/PSTH/Overlay/{}.png'.format(cellid)

    else:
        pth = pth[ind]
        print('pth is {}, ind is {}'.format(pth, ind))
        if type(pth) == list:
            pth_ = pth;
            pth = pth_[0]
            pth2 = pth_[1]
            print('pth1 is {} in ind {}'.format(pth, ind))
            print('pth2 is {} in ind {}'.format(pth2, ind))
            ax_ = ax
            ax = ax_[0]
        else:
            print('{} in ind {}'.format(pth, ind))
            if type(ax) == list:
                print('list!')
                ax_ = ax
                ax = ax_[0]
                pth2 = pth.replace('.png', '_all_val.png')
            else:
                pth = pth.replace('.png', '_all_val.png')
    if type(ax) == list:
        print('ax is a list!')
        ax_ = ax
        ax = ax_[0]
        if 'cellids' in extras:
            ind = extras['cellids'].index(cellid)
            pth2 = pth.replace('0000.png',f'{ind+1:04d}.png')
        else:
            pth2 = pth.replace('.png', '_all_val.png')
    print(pth)
    if pth.split('.')[1] == 'pickle':
        ax.figure = pl.load(open(pth, 'rb'))
    elif ax is None:
        ax = display_image_in_actual_size(pth)
    else:
        im_data = plt.imread(pth)
        ax.clear()
        ax.imshow(im_data, interpolation='bilinear')
    ax.figure.canvas.draw()
    ax.figure.canvas.show()

    if type(ax_) == list:
        im_data = plt.imread(pth2)
        ax_[1].clear()
        print(pth2);
        print(ax_[1])
        ax_[1].imshow(im_data, interpolation='bilinear')
        ax_[1].figure.canvas.draw()
        ax_[1].figure.canvas.show()


def display_image_in_actual_size(im_path, ax=None):
    dpi = 150
    im_data = plt.imread(im_path)
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray', interpolation='bilinear')

    plt.show()

    return ax


def generate_weighted_model_signals(sig_in, weights, epcs_offsets, IncSwitchTime=None):
    sig_out = sig_in.copy()
    sig_out._data = np.full(sig_out._data.shape, np.nan)
    if IncSwitchTime is None:
        types = ['C', 'I']
    else:
        types = ['CtoI', 'I']
    epcs = sig_in.epochs[sig_in.epochs['name'].str.contains('STIM')].copy()
    epcs['type'] = epcs['name'].apply(parse_stim_type)
    orig_epcs = sig_in.epochs.copy()
    sig_in.epochs['start'] = sig_in.epochs['start'] + epcs_offsets[0]
    sig_in.epochs['end'] = sig_in.epochs['end'] + epcs_offsets[1]
    EA = np.array([n.split('+')[1] for n in epcs['name']])
    EB = np.array([n.split('+')[2] for n in epcs['name']])
    corrs = {}
    # print(epcs)
    # print(types)
    # print(weights)
    for _weights, _type in zip(weights, types):
        inds = np.nonzero(epcs['type'].values == _type)[0]
        for ind in inds:
            r = sig_in.extract_epoch(epcs.iloc[ind]['name'])
            if np.any(np.isfinite(r)):
                indA = np.where((EA[ind] == EA) & (EB == 'null'))[0]
                if (len(indA) > 0):
                    rA = sig_in.extract_epoch(epcs.iloc[indA[0]]['name'])
                if _type == 'CtoI':
                    EBparts = EB[ind].split('to')
                    indB = np.where((EBparts[0] == EB) & (EA == 'null'))[0]
                    indB2 = np.where((EBparts[1] == EB) & (EA == 'null'))[0]
                    if (len(indB) > 0) & (len(indB2) > 0):
                        rB = sig_in.extract_epoch(epcs.iloc[indB[0]]['name'])
                        rB2 = sig_in.extract_epoch(epcs.iloc[indB2[0]]['name'])
                        replace_inds = np.arange(IncSwitchTime * sig_in.fs, rB.shape[2])
                        rB[:, :, replace_inds] = rB2[:, :, replace_inds]
                else:
                    indB = np.where((EB[ind] == EB) & (EA == 'null'))[0]
                    if (len(indB) > 0):
                        rB = sig_in.extract_epoch(epcs.iloc[indB[0]]['name'])
                if (len(indA) > 0) & (len(indB) > 0):
                    sig_out = sig_out.replace_epoch(epcs.iloc[ind]['name'], _weights[0] * rA + _weights[1] * rB,
                                                    preserve_nan=False)

        ins = sig_in.extract_epochs(epcs.iloc[inds]['name'])
        if len(epcs.iloc[inds]['name']) == 0:
            from pdb import set_trace
            set_trace()
        ins = np.hstack([ins[k] for k in ins.keys()]).flatten()
        outs = sig_out.extract_epochs(epcs.iloc[inds]['name'])
        outs = np.hstack([outs[k] for k in outs.keys()]).flatten()
        ff = np.isfinite(ins) & np.isfinite(outs)
        cc = np.corrcoef(ins[ff], outs[ff])
        corrs[_type] = cc[0, 1]

    # Generate weighted model signals for square-wave stimuli
    sq_weights = weights[0]  # Use coherent weights for square wave comnbinations
    if np.sum(epcs['name'] == 'STIM_T+Square_0_2+Square_0_2') == 1:
        rA = sig_in.extract_epoch('STIM_T+Square_0_2+null')
        rB = sig_in.extract_epoch('STIM_T+null+Square_0_2')
        sig_out = sig_out.replace_epoch('STIM_T+Square_0_2+Square_0_2', sq_weights[0] * rA + sq_weights[1] * rB,
                                        preserve_nan=False)
    if np.sum(epcs['name'] == 'STIM_T+Square_0_2+Square_1_3') == 1:
        rA = sig_in.extract_epoch('STIM_T+Square_0_2+null')
        rB = sig_in.extract_epoch('STIM_T+null+Square_0_2')
        rB = np.pad(rB, [(0, 0), (0, 0), (sig_in.fs, 0)], 'constant')  # add 1s of zeros to the front
        rB = rB[:, :, :rA.shape[2]]
        sig_out = sig_out.replace_epoch('STIM_T+Square_0_2+Square_1_3', sq_weights[0] * rA + sq_weights[1] * rB,
                                        preserve_nan=False)
    if np.sum(epcs['name'] == 'STIM_T+Square_1_3+Square_0_2') == 1:
        rA = sig_in.extract_epoch('STIM_T+Square_0_2+null')
        rB = sig_in.extract_epoch('STIM_T+null+Square_0_2')
        rA = np.pad(rA, [(0, 0), (0, 0), (sig_in.fs, 0)], 'constant')  # add 1s of zeros to the front
        rA = rA[:, :, :rB.shape[2]]
        sig_out = sig_out.replace_epoch('STIM_T+Square_1_3+Square_0_2', sq_weights[0] * rA + sq_weights[1] * rB,
                                        preserve_nan=False)

    sig_in.epochs = orig_epcs
    sig_out.epochs = orig_epcs.copy()
    return sig_out, corrs


def plot_linear_and_weighted_psths(batch, cellid, weights=None, subset=None, rec_file=None, fs=200):
    # options = {}
    # options['cellid']=cellid
    # options['batch']=batch
    # options["stimfmt"] = "envelope"
    # options["chancount"] = 0
    # options["rasterfs"] = 100
    # rec_file=nb.baphy_data_path(options)

    # from pdb import set_trace
    # set_trace()
    manager = BAPHYExperiment(batch=batch, cellid=cellid)
    exptparams = manager.get_baphy_exptparams()[0]
    try:
        IncSwitchTime = exptparams['TrialObject'][1]['ReferenceHandle'][1]['IncSwitchTime']
    except:
        IncSwitchTime = None

    if subset == 'squares' or subset == 'squaresOverlap':
        try:
            if exptparams['TrialObject'][1]['ReferenceHandle'][1]['AddSquares'] == 0:
                return None, None, None
        except:
            return None, None, None

    if rec_file is None:
        rec_file = nw.generate_recording_uri(cellid, batch, loadkey='ns.fs100', force_old_loader=False)
    rec = recording.load_recording(rec_file)
    rec['resp'] = rec['resp'].extract_channels([cellid])
    rec['resp'].fs = fs

    epcs = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PreStimSilence'].copy()
    spike_times = rec['resp']._data[cellid]
    count = 0
    for index, row in epcs.iterrows():
        count += np.sum((spike_times > row['start']) & (spike_times < row['end']))
    SR = count / (epcs['end'] - epcs['start']).sum()

    # COMPUTE ALL FOLLOWING metrics using smoothed driven rate
    groups = ep.group_epochs_by_occurrence_counts(rec.epochs, '^STIM_')
    square_epochs = ep.epoch_occurrences(rec.epochs, 'Square')
    N_per_epoch = ep.epoch_occurrences(rec.epochs, '^STIM')
    est_mask = N_per_epoch < N_per_epoch.max() / 9
    epochs_for_est = N_per_epoch.index.values[est_mask]
    epochs_for_val = N_per_epoch.index.values[~est_mask]

    est, val = rec.split_by_epochs(epochs_for_est, epochs_for_val)
    # est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')

    # smooth and subtract SR
    fn = lambda x: np.atleast_2d(smooth(x.squeeze(), 3, 2) - SR / rec['resp'].fs)
    val['resp'] = val['resp'].transform(fn)
    val['resp'] = add_stimtype_epochs(val['resp'])

    lin_weights = [[1, 1], [1, 1]]
    epcs_offsets = [epcs['end'].iloc[0], 0]
    val['lin_model'], l_corrs = generate_weighted_model_signals(val['resp'], lin_weights, epcs_offsets,
                                                                IncSwitchTime=IncSwitchTime)
    if subset == 'squares' or subset == 'squaresOverlap':
        sigz = ['resp', 'lin_model']
        plot_singles_on_dual = False
        w_corrs = None
    elif weights is None:
        sigz = ['resp', 'lin_model']
        plot_singles_on_dual = True
        w_corrs = None
    else:
        val['weighted_model'], w_corrs = generate_weighted_model_signals(val['resp'], weights, epcs_offsets,
                                                                         IncSwitchTime=IncSwitchTime)
        sigz = ['resp', 'lin_model', 'weighted_model']
        plot_singles_on_dual = False
    fh = plot_all_vals(val, None, signames=sigz, channels=[0, 0, 0], subset=subset,
                       plot_singles_on_dual=plot_singles_on_dual, IncSwitchTime=IncSwitchTime)
    return fh, w_corrs, l_corrs


def calc_square_time_constants(row, fs=50, save_pth=None, do_plot=True):
    # options = {}
    # options['cellid']=cellid
    # options['batch']=batch
    # options["stimfmt"] = "envelope"
    # options["chancount"] = 0
    # options["rasterfs"] = 100
    # rec_file=nb.baphy_data_path(options)

    print('load {}'.format(row.name))
    metrics = {}
    manager = BAPHYExperiment(batch=int(row['batch']), cellid=row.name)
    exptparams = manager.get_baphy_exptparams()[0]
    try:
        IncSwitchTime = exptparams['TrialObject'][1]['ReferenceHandle'][1]['IncSwitchTime']
    except:
        IncSwitchTime = None

    rec_file = nw.generate_recording_uri(row.name, int(row['batch']), loadkey='ns.fs' + str(fs), force_old_loader=False)
    rec = recording.load_recording(rec_file)
    rec['resp'] = rec['resp'].extract_channels([row.name])
    # rec['resp'].fs = fs

    # est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')
    names = ['STIM_T+Square_0_2+null', 'STIM_T+null+Square_0_2', 'STIM_T+Square_0_2+Square_0_2',
             'STIM_T+Square_0_2+Square_1_3', 'STIM_T+Square_1_3+Square_0_2']
    epoch_data = rec['resp'].rasterize().extract_epochs(names)
    epoch_data = {key: np.squeeze(val) for key, val in epoch_data.items()}

    # Practicing making a dataframe of it to use Seaborn
    # dat = pd.DataFrame(epoch_data[names[0]].T).melt(var_name='Rep')

    fn = lambda x: smooth(x.squeeze(), 3, 2) - row['SR']
    epoch_means = {}
    epoch_sterrs = {}
    for name in names:
        epoch_means[name] = np.nanmean(epoch_data[name], axis=0) * fs
        epoch_sterrs[name] = np.nanstd(epoch_data[name], axis=0) / np.sqrt(epoch_data[name].shape[0]) * fs
        epoch_means[name] = fn(epoch_means[name])
        epoch_sterrs[name] = fn(epoch_sterrs[name])
        # epoch_means[name] = smooth(epoch_means[name],3,2)
        # epoch_sterrs[name] = smooth(epoch_sterrs[name], 3, 2)
    prestimtime = rec['resp'].epochs[rec['resp'].epochs['name'] == 'PreStimSilence']['end'].values[0]
    time = np.arange(0, len(
        epoch_means[names[0]])) / fs - prestimtime + 1 / fs / 2  # center time of time bines, re stim on time

    thresh = np.array((row['SR'] / fs + 0.1 * (row['SinglesMax'] - row['SR'] / fs),
                       (row['SR'] - row['SR_av_std']) / fs))
    excitatory_percentage = {}
    inhibitory_percentage = {}
    Max = {}
    Mean = {}
    twins = ((0, 2), (0, 2), (0, 2), (1, 2), (1, 2))
    # from pdb import set_trace;
    # set_trace()
    for _type, twin in zip(names, twins):
        ps = epoch_means[_type][np.all([time >= twin[0], time < twin[1]], 0)]
        # plt.figure();plt.plot(ps); plt.title(_type)
        ff = np.isfinite(ps)
        excitatory_percentage[_type] = (ps[ff] > thresh[0]).sum() / ff.sum()
        inhibitory_percentage[_type] = (ps[ff] < thresh[1]).sum() / ff.sum()
        Max[_type] = ps[ff].max() / row['SinglesMax']
        Mean[_type] = ps[ff].mean()

    metrics['SqEP_A'] = excitatory_percentage['STIM_T+Square_0_2+null']
    metrics['SqEP_B'] = excitatory_percentage['STIM_T+null+Square_0_2']
    metrics['SqEP_C'] = excitatory_percentage['STIM_T+Square_0_2+Square_0_2']
    metrics['SqEP_I'] = (
    excitatory_percentage['STIM_T+Square_0_2+Square_1_3'], excitatory_percentage['STIM_T+Square_1_3+Square_0_2'])
    metrics['SqIP_A'] = inhibitory_percentage['STIM_T+Square_0_2+null']
    metrics['SqIP_B'] = inhibitory_percentage['STIM_T+null+Square_0_2']
    metrics['SqIP_C'] = inhibitory_percentage['STIM_T+Square_0_2+Square_0_2']
    metrics['SqIP_I'] = (
    inhibitory_percentage['STIM_T+Square_0_2+Square_1_3'], inhibitory_percentage['STIM_T+Square_1_3+Square_0_2'])
    metrics['SqMax_A'] = Max['STIM_T+Square_0_2+null']
    metrics['SqMax_B'] = Max['STIM_T+null+Square_0_2']
    metrics['SqMax_C'] = Max['STIM_T+Square_0_2+Square_0_2']
    metrics['SqMax_I'] = (Max['STIM_T+Square_0_2+Square_1_3'], Max['STIM_T+Square_1_3+Square_0_2'])
    metrics['SqMean_A'] = Mean['STIM_T+Square_0_2+null']
    metrics['SqMean_B'] = Mean['STIM_T+null+Square_0_2']
    metrics['SqMean_C'] = Mean['STIM_T+Square_0_2+Square_0_2']
    metrics['SqMean_I'] = (Mean['STIM_T+Square_0_2+Square_1_3'], Mean['STIM_T+Square_1_3+Square_0_2'])
    metrics = type_by_psth(metrics, prefix='Sq')

    if do_plot:
        ph = []
        gs_kw = dict(hspace=0, left=0.06, right=.99)
        fh, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharey=True, gridspec_kw=gs_kw)
        ph.append(ax[2].plot(time, epoch_means[names[0]])[0])
        ax[2].fill_between(time, epoch_means[names[0]] - epoch_sterrs[names[0]] / 2,
                           epoch_means[names[0]] + epoch_sterrs[names[0]] / 2, alpha=.5)
        ph.append(ax[2].plot(time, epoch_means[names[1]])[0])
        ax[2].fill_between(time, epoch_means[names[1]] - epoch_sterrs[names[1]] / 2,
                           epoch_means[names[1]] + epoch_sterrs[names[1]] / 2, alpha=.5)
        ax[2].legend(ph[:2], names[:2])

        ph.append(ax[1].plot(time, epoch_means[names[2]], color='C2')[0])
        ax[1].fill_between(time, epoch_means[names[2]] - epoch_sterrs[names[2]] / 2,
                           epoch_means[names[2]] + epoch_sterrs[names[2]] / 2, alpha=.5, color='C2')
        ph.append(ax[1].plot(time, epoch_means[names[3]], color='C3')[0])
        ax[1].fill_between(time, epoch_means[names[3]] - epoch_sterrs[names[3]] / 2,
                           epoch_means[names[3]] + epoch_sterrs[names[3]] / 2, alpha=.5, color='C3')
        ax[1].legend(ph[2:4], names[2:4])

        ph.append(ax[0].plot(time, epoch_means[names[2]], color='C2')[0])
        ax[0].fill_between(time, epoch_means[names[2]] - epoch_sterrs[names[2]] / 2,
                           epoch_means[names[2]] + epoch_sterrs[names[2]] / 2, alpha=.5, color='C2')
        ph.append(ax[0].plot(time, epoch_means[names[4]], color='C4')[0])
        ax[0].fill_between(time, epoch_means[names[4]] - epoch_sterrs[names[4]] / 2,
                           epoch_means[names[4]] + epoch_sterrs[names[4]] / 2, alpha=.5, color='C4')
        ax[0].legend(ph[4:], [names[x] for x in [2, 4]])
        yl = ax[0].get_ylim()

    tcomp = np.nonzero(np.logical_and(time > 1, time <= 2))[0]
    pv = np.full(tcomp.shape, np.nan)
    df = np.full(tcomp.shape, np.nan)
    for i in range(len(tcomp)):
        st, pv[i] = scipy.stats.ttest_ind(epoch_data[names[2]][:, tcomp[i]], epoch_data[names[3]][:, tcomp[i]])
        df[i] = epoch_data[names[3]][:, tcomp[i]].mean() - epoch_data[names[2]][:, tcomp[i]].mean()

    ts_ss = epoch_means[names[2]][tcomp].mean()
    times = time[tcomp] - 1
    twovoice_diff = epoch_means[names[3]][tcomp] - ts_ss
    # params, fit_fn = fit_monoExp(times, twovoice_diff)
    params, fit_fn, mse = fit_diffTwoExp(times, twovoice_diff)
    metrics['Square_AtoA+B_amp'] = params[0]
    metrics['Square_AtoA+B_tc1'] = 1 / params[1]
    metrics['Square_AtoA+B_tc2'] = 1 / params[2]
    metrics['Square_AtoA+B_mse'] = mse
    metrics['Square_AtoA+B_Nmse'] = mse / row['SinglesMax']
    if np.isnan(params[0]):
        ax[1].text(1, yl[1], 'can\'t fit', va='top')
    elif do_plot:
        time_fit = np.arange(0, max(times), .01)
        fit_curve = fit_fn(time_fit, *params)
        # plt.figure()
        # plt.plot(times,twovoice_diff,'.')
        # plt.plot(time_fit, fit_curve, '--')
        ax[1].plot(time_fit + 1, fit_curve + ts_ss, '--', color='grey')
        ax[1].text(time_fit.mean() + 1, yl[1], r'$g= {:.0f},  \tau=[{:.0f}, {:.0f}]ms, nmse{:.2f}$'.format(params[0],
                                                                                                           1000 /
                                                                                                           params[1],
                                                                                                           1000 /
                                                                                                           params[2],
                                                                                                           mse / row[
                                                                                                               'SinglesMax']),
                   va='top')
        ax[1].plot(time_fit[[0, -1]] + 1, [ts_ss, ts_ss], color='C2')

    # plt.figure()
    # plt.plot(times, twovoice_diff, '.')
    # params, fit_fn = fit_monoExp(times, twovoice_diff)
    # fit_curve = fit_fn(time_fit, *params)
    # plt.plot(time_fit, fit_curve, '--',label='1exp')
    # params, fit_fn = fit_diffTwoExp(times, twovoice_diff)
    # fit_curve = fit_fn(time_fit, *params)
    # plt.plot(time_fit, fit_curve, '--',label='diffExp')
    # params, fit_fn, mse = fit_diffTwoExpBase(times, twovoice_diff)
    # fit_curve = fit_fn(time_fit, *params)
    # plt.plot(time_fit, fit_curve, '--', label='diffExp+baseline')
    # params, fit_fn, mse = fit_diffTwoExpTwoGainBase(times, twovoice_diff)
    # fit_curve = fit_fn(time_fit, *params)
    # plt.plot(time_fit, fit_curve, '--', label='diffExpTwoGain+baseline')
    # params, fit_fn, mse = fit_diffTwoExpSteady(times, twovoice_diff)
    # fit_curve = fit_fn(time_fit, *params)
    # plt.plot(time_fit, fit_curve, '--', label='diffExp+steady')
    # plt.legend()

    twovoice_diff = epoch_means[names[4]][tcomp] - ts_ss
    params, _, mse = fit_diffTwoExp(times, twovoice_diff)
    metrics['Square_BtoA+B_amp'] = params[0]
    metrics['Square_BtoA+B_tc'] = 1 / params[1]
    metrics['Square_BtoA+B_tc2'] = 1 / params[2]
    metrics['Square_BtoA+B_mse'] = mse
    metrics['Square_BtoA+B_Nmse'] = mse / row['SinglesMax']
    if np.isnan(params[0]):
        ax[0].text(1, yl[1], 'can\'t fit', va='top')
    elif do_plot:
        time_fit = np.arange(0, max(times), .01)
        fit_curve = fit_fn(time_fit, *params)
        ax[0].plot(time_fit + 1, fit_curve + ts_ss, '--', color='grey')
        ax[0].text(time_fit.mean() + 1, yl[1], r'$g= {:.0f},  \tau=[{:.0f}, {:.0f}]ms, mse{:.2f}$'.format(params[0],
                                                                                                          1000 / params[
                                                                                                              1],
                                                                                                          1000 / params[
                                                                                                              2],
                                                                                                          mse / row[
                                                                                                              'SinglesMax']),
                   va='top')
        ax[0].plot(time_fit[[0, -1]] + 1, [ts_ss, ts_ss], color='C2')

    # scipy.stats.bootstrap((epoch_data[names[3]][:, tcomp]-epoch_data[names[2]][:, tcomp].mean(),),np.mean,method='percentile')
    if do_plot:
        ax[0].set_ylim(yl)
        fh.axes[0].set_title('{}: SqRType: {}, Pri: {}, RType: {}, Pri: {}'.format(row.name, metrics['SqRtype'],
                                                                                   ['A', 'B'][metrics['Sqinds'][0]],
                                                                                   row['Rtype'],
                                                                                   ['A', 'B'][row['inds'][0]]))
        if save_pth is not None:
            fh.savefig(save_pth + row.name + '.png')
            with open(save_pth + row.name + '.pickle', 'wb') as handle:
                pl.dump(fh, handle)
            plt.close(fh)
    return fh, metrics


def fit_monoExp(x, y):
    monoExp = lambda x, m, t: m * np.exp(-t * x)
    if len(np.unique(y)) < 3:
        # y has less than 3 values, don't fit
        return [np.NaN, np.NaN], monoExp
    elif np.abs(y.max()) > np.abs(y.min()):
        # max greater than min, fit positive gain
        p0 = (max(y), 1 / .1)  # initial values
        bounds = ((0, 1 / 2), (2 * max(y), np.inf))
    else:
        # min greater than max, fit negative gain
        p0 = (min(y), 1 / .1)  # initial values
        bounds = ((2 * min(y), 1 / 2), (0, np.inf))
    try:
        params, cv = scipy.optimize.curve_fit(monoExp, x, y, p0, bounds=bounds)
    except RuntimeError as RTE:
        if str(RTE).count('Optimal parameters not found') > 0:
            print(RTE)
            params = [np.NaN, np.NaN]
        else:
            raise (RTE)
    return params, monoExp


def fit_diffTwoExp(x, y):
    diffTwoExp = lambda x, m, t1, t2: m * (np.exp(-t1 * x) - np.exp(-t2 * x))
    if len(np.unique(y)) < 3:
        # y has less than 3 values, don't fit
        return [np.NaN, np.NaN, np.NaN], diffTwoExp, np.NaN
    elif np.abs(y.max()) > np.abs(y.min()):
        # max greater than min, fit positive gain
        p0 = (max(y), 1 / .1, 1 / .01)  # initial values
        bounds = ((0, 1 / 2, 1 / .5), (2 * max(y), np.inf, np.inf))
    else:
        # min greater than max, fit negative gain
        p0 = (min(y), 1 / .1, 1 / .01)  # initial values
        bounds = ((2 * min(y), 1 / 2, 1 / .5), (0, np.inf, np.inf))
    try:
        params, cv = scipy.optimize.curve_fit(diffTwoExp, x, y, p0, bounds=bounds)
        mse = ((y - diffTwoExp(x, *params)) ** 2).mean()
    except RuntimeError as RTE:
        if str(RTE).count('Optimal parameters not found') > 0:
            print(RTE)
            params = [np.NaN, np.NaN, np.NaN]
            mse = np.NaN
        else:
            raise (RTE)
    return params, diffTwoExp, mse


def fit_diffTwoExpBase(x, y):
    diffTwoExpBase = lambda x, m, t1, t2, b: m * (np.exp(-t1 * x) - np.exp(-t2 * x)) + b
    if len(np.unique(y)) < 3:
        # y has less than 3 values, don't fit
        return [np.NaN, np.NaN], monoExp
    elif np.abs(y.max()) > np.abs(y.min()):
        # max greater than min, fit positive gain
        p0 = (max(y), 1 / .1, 1 / .01, y[round(.6 * len(y)):].mean())  # initial values
        bounds = ((0, 1 / 2, 1 / .5, 0), (2 * max(y), np.inf, np.inf, max(y)))
    else:
        # min greater than max, fit negative gain
        p0 = (min(y), 1 / .1, 1 / .01)  # initial values
        bounds = ((2 * min(y), 1 / 2, 1 / .5), (0, np.inf, np.inf))
    try:
        print('P0:{}'.format(p0))
        print('LB:{}'.format(bounds[0]))
        print('UB:{}'.format(bounds[1]))
        params, cv = scipy.optimize.curve_fit(diffTwoExpBase, x, y, p0, bounds=bounds)
        mse = ((y - diffTwoExpBase(x, *params)) ** 2).mean()
    except RuntimeError as RTE:
        if str(RTE).count('Optimal parameters not found') > 0:
            print(RTE)
            params = [np.NaN, np.NaN, np.NaN]
            mse = np.NaN
        else:
            raise (RTE)
    return params, diffTwoExpBase, mse


def fit_diffTwoExpSteady(x, y):
    diffTwoExp = lambda x, m, t1, t2, b: m * np.exp(-t1 * x) - (m + b) * np.exp(-t2 * x) + b
    if len(np.unique(y)) < 3:
        # y has less than 3 values, don't fit
        return [np.NaN, np.NaN], monoExp
    elif np.abs(y.max()) > np.abs(y.min()):
        # max greater than min, fit positive gain
        p0 = (max(y), 1 / .1, 1 / .01, y[round(.6 * len(y)):].mean())  # initial values
        bounds = ((0, 1 / 2, 1 / .5, 0), (2 * max(y), np.inf, np.inf, max(y)))
        bounds = ((0, 1 / 2, 1 / .5, 0), (np.inf, np.inf, np.inf, max(y)))
    else:
        # min greater than max, fit negative gain
        p0 = (min(y), 1 / .1, 1 / .01)  # initial values
        bounds = ((2 * min(y), 1 / 2, 1 / .5), (0, np.inf, np.inf))
    try:
        print('P0:{}'.format(p0))
        print('LB:{}'.format(bounds[0]))
        print('UB:{}'.format(bounds[1]))
        params, cv = scipy.optimize.curve_fit(diffTwoExp, x, y, p0, bounds=bounds)
        mse = ((y - diffTwoExp(x, *params)) ** 2).mean()
    except RuntimeError as RTE:
        if str(RTE).count('Optimal parameters not found') > 0:
            print(RTE)
            params = [np.NaN, np.NaN, np.NaN]
            mse = np.NaN
        else:
            raise (RTE)
    return params, diffTwoExp, mse


def fit_diffTwoExpTwoGainBase(x, y):
    diffTwoExp = lambda x, g1, g12diff, t1, t2, b: g1 * np.exp(-t1 * x) - (g1 + g12diff) * np.exp(-t2 * x) + b
    if len(np.unique(y)) < 3:
        # y has less than 3 values, don't fit
        return [np.NaN, np.NaN], monoExp
    elif np.abs(y.max()) > np.abs(y.min()):
        # max greater than min, fit positive gain
        p0 = (max(y), y[round(.6 * len(y)):].mean(), 1 / .15, 1 / .05, y[round(.6 * len(y)):].mean())  # initial values
        bounds = ((0, 0, 1 / 2, 1 / .5, 0), (2 * max(y), max(y), np.inf, np.inf, max(y)))
    else:
        # min greater than max, fit negative gain
        p0 = (min(y), 1 / .1, 1 / .01)  # initial values
        bounds = ((2 * min(y), 1 / 2, 1 / .5), (0, np.inf, np.inf))
    try:
        print('P0:{}'.format(p0))
        print('LB:{}'.format(bounds[0]))
        print('UB:{}'.format(bounds[1]))
        params, cv = scipy.optimize.curve_fit(diffTwoExp, x, y, p0, bounds=bounds)
        mse = ((y - diffTwoExp(x, *params)) ** 2).mean()
    except RuntimeError as RTE:
        if str(RTE).count('Optimal parameters not found') > 0:
            print(RTE)
            params = [np.NaN, np.NaN, np.NaN]
            mse = np.NaN
        else:
            raise (RTE)
    return params, diffTwoExp, mse


def dprime(x, y, axis=0):
    l1 = np.size(x, axis)
    l2 = np.size(y, axis)
    stdev = np.sqrt(((l2 - 1) * np.std(x, axis=axis) ** 2 + (l1 - 1) * np.std(y, axis=axis) ** 2) / (l1 + l2 - 2))
    return (np.mean(x, axis=axis) - np.mean(y, axis=axis)) / stdev


def permuation_test(vals, inds, statistic):
    l2 = len(inds[0])
    l1 = len(inds[1])
    g = list(np.sort(np.hstack((inds[0], inds[1]))))
    # pval = np.nan(vals.shape(1));
    # fvala = nan(1, size(values, 2));
    # exactN = np.math.factorial(l1 + l2) / np.math.factorial(l2) / np.math.factorial(l1)
    exactN = np.math.comb(l1 + l2, l1)
    N = 2000
    do_all = exactN < N

    if do_all:
        g2 = list(itertools.combinations(g, l2))
    else:
        rp = [random.sample(g, l1 + l2) for i in range(N)]
        g1 = [rp_[:l1] for rp_ in rp]
        g2 = [rp_[l1:] for rp_ in rp]

    fval = np.full(len(g2), 0)
    for i in range(len(g2)):
        if do_all:
            g1_ = list(set(g) - set(g2[i]))
        else:
            g1_ = g1[i]
        fval.append(statistic(vals[g1_], vals[list(g2[i])]))

    # if two_sided:
    #     pval(i) = sum((abs(fval) - abs(fvala(i))) > -10 ^ -10). / size(fval, 1);
    #     % pval(i) = sum((abs(fval) - abs(fvala(i))) > -10 ^ -10). / size(fval, 1);
    #     % sum((fval - fvala(i)) > -10 ^ -10). / size(fval, 1);
    #     Dp95(i) = prctile(abs(fval), 95);
    #     else
    #     pval(i) = sum((fval - fvala(i)) > -10 ^ -10). / size(fval, 1);
    #     Dp95(i) = prctile(fval, 95);
    #     end
    #
    #     Dpx(i) = prctile(abs(fval), 100 - 100 * pval(i));


def shuffle_along_axis(array, shuffle_axis, indie_axis=None, rng=None):
    '''
    shuffles in place an array along the selected axis or group of axis .
    :param array: nd-array
    :param shuffle_axis: int or int list. axis along which to perform the shuffle
    :param indie_axis: int or int list. shuffling will be done independently across positions in these axis.
    :rng: instance of numpy.random.default_rng(), if none is passed, a random seed is used to create one.
    :return: shuffled array of the same shape as input array.
    FROM Mateo
    '''

    # turn axis inputs into lists of ints.
    if isinstance(shuffle_axis, int):
        shuffle_axis = [shuffle_axis]
    if isinstance(indie_axis, int):
        indie_axis = [indie_axis]

    if rng is None:
        rng = np.random.default_rng()

    # reorder axis, first: indie_axis second: shuffle_axis, third: all other axis i.e. protected axis.
    other_axis = [x for x in range(array.ndim) if x not in indie_axis and x not in shuffle_axis]
    new_order = indie_axis + shuffle_axis + other_axis

    array = np.transpose(array, new_order)

    # if multiple axes are being shuffled together, reshapes  collapsing across the shuffle_axis
    # shape of independent chunks of the array, i, s, o , independent, shuffle, other.
    shape = array.shape
    i_shape = shape[0:len(indie_axis)]
    s_shape = (np.prod(shape[len(indie_axis):len(shuffle_axis) + len(indie_axis)], dtype=int),)
    o_shape = shape[-len(other_axis):] if len(other_axis) > 0 else ()

    new_shape = i_shape + s_shape + o_shape

    array = np.reshape(array, new_shape)

    if indie_axis is None:
        rng.shuffle(array)
    else:
        # slices the array along the independent axis
        # shuffles independently for each slice
        for ndx in np.ndindex(shape[:len(indie_axis)]):  # this is what takes for ever.
            rng.shuffle(array[ndx])

    # reshapes into original dimensions
    array = np.reshape(array, shape)

    # swap the axis back into original positions
    array = np.transpose(array, np.argsort(new_order))

    return array


def plot_psth_weights(df, ax=None, s='R', lengthscale=10, fnargs=None, fn=None, norm_method=None):
    if type(df) is pd.Series:
        df = pd.DataFrame(df).T
    if ax is None:
        ax = plt.gca()

    def center_on_C(weights):
        w = weights.copy()
        w = w - w[0, :]
        return w

    def center_on_I(weights):
        w = weights.copy()
        w = w - w[1, :]
        return w

    def no_norm(weights):
        return weights.copy()

    if norm_method is None:
        norm = no_norm
    else:
        norm = locals()[norm_method]
    R = np.arange(0, 2 * np.pi, 0.01)
    for index, row in df.iterrows():
        weights = np.vstack([row['weights_C' + s], row['weights_I' + s]])
        weights = norm(weights)
        ax.plot(weights[:, 0], weights[:, 1])

        try:
            efit = np.vstack((row['Efit_C' + s], row['Efit_I' + s]))
            efit[:, :2] = norm(efit[:, :2])

            if row['nrmse_CR'] < 1:
                ls = '-'
            else:
                ls = '--'

            centers = efit[0, 0:2]
            lengths = efit[0, 2:4].copy() / lengthscale
            phi = efit[0, 4]
            xx = centers[0] + lengths[0] * np.cos(R) * np.cos(phi) - lengths[1] * np.sin(R) * np.sin(phi)
            yy = centers[1] + lengths[0] * np.cos(R) * np.sin(phi) + lengths[1] * np.sin(R) * np.cos(phi)
            ax.plot(xx, yy, color='k', linewidth=.5, linestyle=ls)

            if row['nrmse_IR'] < 1:
                ls = '-'
            else:
                ls = '--'

            centers = efit[1, 0:2]
            lengths = efit[1, 2:4].copy() / lengthscale
            phi = efit[1, 4]
            xx = centers[0] + lengths[0] * np.cos(R) * np.cos(phi) - lengths[0] * np.sin(R) * np.sin(phi)
            yy = centers[1] + lengths[0] * np.cos(R) * np.sin(phi) + lengths[1] * np.sin(R) * np.cos(phi)
            ax.plot(xx, yy, color='r', linewidth=.5, linestyle=ls)
        except:
            pass
    # w=np.array(df['weights_C'+s].tolist())
    weights = np.swapaxes(np.stack((df['weights_C' + s].tolist(), df['weights_I' + s].tolist()), axis=2), 0, 2)
    weights = norm(weights)
    if df.index.name is None:
        names = df['cellid'].values.tolist()
    else:
        names = df.index.values.tolist()
    phc = scatterplot_print(weights[0, 0, :], weights[0, 1, :], names,
                            ax=ax, color='k', markersize=8, fn=fn, fnargs=fnargs)
    # w=np.array(df['weights_I'+s].tolist())
    phi = scatterplot_print(weights[1, 0, :], weights[1, 1, :], names,
                            ax=ax, color='r', markersize=8, fn=fn, fnargs=fnargs)

    ax.plot([0, 1], [0, 1], 'k')
    ax.plot([0, 0], [0, 1], 'k')
    ax.plot([0, 1], [0, 0], 'k')

    return phc, phi


def plot_weighted_psths_and_weightplot(row, weights, batch=306):
    weights2 = [w[row['inds']] for w in weights]
    fh, w_corrs, l_corrs = plot_linear_and_weighted_psths(batch, row.name, weights2)

    ax = fh.axes
    for ax_ in ax:
        pos = ax_.get_position()
        pos.y0 = pos.y0 - .08
        pos.y1 = pos.y1 - .08
        ax_.set_position(pos)
    axN = fh.add_axes([.3, .84, .4, .16])
    phcR, phiR = plot_psth_weights(row, ax=axN, lengthscale=1)
    # phcR,phiR=plot_psth_weights(row2,ax=ax,lengthscale=1)
    phc, phi = plot_psth_weights(row, ax=axN, s='')
    phc.set_marker('*');
    phi.set_marker('*')
    axN.plot(weights[0][0], weights[0][1], 'ok')
    axN.plot(weights[1][0], weights[1][1], 'or')
    axN.set_xlabel('Voice a')
    axN.set_ylabel('Voice b')
    meta_str = '{}\nRType: {}, Pri: {}'.format(row.name, row['Rtype'], ['A', 'B'][row['inds'][0]])
    corr_str = 'Weighted Correlations:\nC: {:.2f} ({:.2f})\nI : {:.2f} ({:.2f})'.format(
        w_corrs['C'], row['r_CR'], w_corrs['I'], row['r_IR'])
    xv = fh.axes[-1].get_xlim()[1]
    yv = fh.axes[-1].get_ylim()[1]
    fh.axes[-1].text(xv, yv, meta_str + '\n' + corr_str, verticalalignment='top')
    return fh


def calc_psth_weight_model(model, celldf=None, do_plot=False,
                           modelspecs_dir='/auto/users/luke/Code/nems/modelspecs/normalization_branch'):
    cellid = model['cellid']
    cell = celldf.loc[cellid]
    print('load {}, {}'.format(cellid, model['modelspecname']))
    modelspecs, est, val = load_SPO(cellid,
                                    ['A', 'B', 'C', 'I'],
                                    model['modelspecname'], fs=200,
                                    modelspecs_dir=modelspecs_dir)
    # smooth and subtract SR
    fn = lambda x: np.atleast_2d(smooth(x.squeeze(), 3, 2) - cell['SR'] / val[0]['resp'].fs)

    # fn = lambda x : np.atleast_2d(smooth(x.squeeze(), 3, 2)*val[0]['resp'].fs - row['SR'])
    val[0]['resp'] = val[0]['resp'].transform(fn)

    # calc SR of pred
    ps = est[0]['pred'].select_epochs(['PreStimSilence']).as_continuous()
    ff = np.isfinite(ps)
    SR_model = ps[ff].mean() * val[0]['pred'].fs

    fn = lambda x: np.atleast_2d(x.squeeze() - SR_model / val[0]['pred'].fs)
    val[0]['pred'] = val[0]['pred'].transform(fn)
    print('calc weights')
    # weights_CR_,weights_IR_=calc_psth_weights_of_model_responses(val[0],signame='resp')
    # weights_CR_,weights_IR_,Efit_CR_,Efit_IR_=calc_psth_weights_of_model_responses(val[0],signame='resp',do_plot=do_plot)
    # weights_CR_, Efit_C_, nmse_C, nf_C, get_mse_C, weights_I, Efit_I, nmse_I, nf_I, get_mse_I
    # d=calc_psth_weights_of_model_responses(val[0],signame='resp',do_plot=do_plot)
    # d={k+'R': v for k, v in d.items()}
    # for k, v in d.items():
    #    row[k]=v
    dat = calc_psth_weights_of_model_responses(val[0], do_plot=do_plot, find_mse_confidence=False, get_nrmse_fn=False)
    for k, v in dat.items():
        model[k] = v

    if cell['get_nrmse_IR'] is None:
        raise RuntimeError("Function cell['get_nrmse_IR'] is none.")
    else:
        model['LN_nrmse_ratio_I'] = (1 - cell['get_nrmse_IR'](model['weights_I'])) / (1 - cell['nrmse_IR'])
        model['LN_nrmse_ratio_C'] = (1 - cell['get_nrmse_CR'](model['weights_C'])) / (1 - cell['nrmse_CR'])
    return model


def calc_psth_weight_resp(row, do_plot=False, fs=200):
    print('load {}'.format(row.name))
    manager = BAPHYExperiment(batch=306, cellid=row.name)
    exptparams = manager.get_baphy_exptparams()[0]
    try:
        IncSwitchTime = exptparams['TrialObject'][1]['ReferenceHandle'][1]['IncSwitchTime']
        types = ['A', 'B', 'CtoI', 'I']
    except:
        types = ['A', 'B', 'C', 'I']

    modelspecs, est, val = load_SPO(row.name,
                                    types,
                                    None, fs=fs,
                                    get_est=False,
                                    get_stim=False)
    # smooth and subtract SR
    fn = lambda x: np.atleast_2d(smooth(x.squeeze(), 3, 2) - row['SR'] / val[0]['resp'].fs)

    # fn = lambda x : np.atleast_2d(sp.smooth(x.squeeze(), 3, 2)*val[0]['resp'].fs - row['SR'])
    val[0]['resp'] = val[0]['resp'].transform(fn)

    # fn = lambda x : np.atleast_2d(x.squeeze() - 0.02222993646993765)
    # val[0]['pred']=val[0]['pred'].transform(fn)
    print('calc weights')
    # weights_CR_,weights_IR_=sp.calc_psth_weights_of_model_responses(val[0],signame='resp')
    # weights_CR_,weights_IR_,Efit_CR_,Efit_IR_=sp.calc_psth_weights_of_model_responses(val[0],signame='resp',do_plot=do_plot)
    # weights_CR_, Efit_C_, nmse_C, nf_C, get_mse_C, weights_I, Efit_I, nmse_I, nf_I, get_mse_I
    d = calc_psth_weights_of_model_responses(val[0], signame='resp', do_plot=do_plot, exptparams=exptparams)
    d = {k + 'R': v for k, v in d.items()}
    for k, v in d.items():
        row[k] = v
    # Do these in SPO_model_metrics now?

    # d=sp.calc_psth_weights_of_model_responses(val[0],do_plot=do_plot,find_mse_confidence=False)
    # for k, v in d.items():
    #    row[k]=v

    # row['LN_nrmse_ratio_I']=(1-row['get_nrmse_IR'](row['weights_I'])) / (1 - row['nrmse_IR'])
    # row['LN_nrmse_ratio_C']=(1-row['get_nrmse_CR'](row['weights_C'])) / (1 - row['nrmse_CR'])
    return row


def calc_psth_weight_cell(cell, do_plot=False,
                          modelspecs_dir='/auto/users/luke/Code/nems/modelspecs/normalization_branch',
                          get_nrmse_only=False):
    cellid = cell.name
    if get_nrmse_only and (cell['get_nrmse_CR'] is not None) and (cell['get_nrmse_IR'] is not None):
        print('get_nrmse_CR and get_nrmse_IR already exist for {}, skipping'.format(cellid))
        return cell
    print('load {}'.format(cellid))

    modelspecs, est, val = load_SPO(cellid, ['A', 'B', 'C', 'I'], None, fs=200, get_est=False, get_stim=False)

    # smooth and subtract SR
    fn = lambda x: np.atleast_2d(smooth(x.squeeze(), 3, 2) - cell['SR'] / val[0]['resp'].fs)

    # fn = lambda x : np.atleast_2d(smooth(x.squeeze(), 3, 2)*val[0]['resp'].fs - row['SR'])
    val[0]['resp'] = val[0]['resp'].transform(fn)

    print('calc weights')
    # weights_CR_,weights_IR_=calc_psth_weights_of_model_responses(val[0],signame='resp')
    # weights_CR_,weights_IR_,Efit_CR_,Efit_IR_=calc_psth_weights_of_model_responses(val[0],signame='resp',do_plot=do_plot)
    # weights_CR_, Efit_C_, nmse_C, nf_C, get_mse_C, weights_I, Efit_I, nmse_I, nf_I, get_mse_I
    d = calc_psth_weights_of_model_responses(val[0], signame='resp', do_plot=do_plot,
                                             find_mse_confidence=(not get_nrmse_only))
    if get_nrmse_only:
        cell['get_nrmse_CR'] = d['get_nrmse_C']
        cell['get_nrmse_IR'] = d['get_nrmse_I']
    else:
        d = {k + 'R': v for k, v in d.items()}
        for k, v in d.items():
            cell[k] = v

    return cell

def load_modelspec_data(batch,cellids,modelnames):
    msp = []
    er = []
    for mod_i, m in enumerate(modelnames):

        try:
            print('Loading modelname: {}'.format(m))
            mds = nems_db.params._get_modelspecs(cellids, batch, m, multi='mean')
        except ValueError as error:
            er.append(error)
            if error.args[0][:28] == 'No result exists for:\nbatch:':
                print('No Results')
            else:
                raise (error)
        else:
            print('{} cells'.format(len(mds)))
            msp.extend(mds)
    dat = {}
    dat['loader'] = [mx[0]['meta']['recording'] for mx in msp]
    ss = []
    for mx in msp:
        s = mx[0]['meta']['recording'].split('subset')
        if len(s) == 1:
            ss.append('A+B+C+I')
        else:
            ss.append(s[1][1:].split('-')[0])
    dat['subset'] = ss
    dat['modelspecname'] = [mx[0]['meta']['modelspecname'] for mx in msp]
    dat['modelname'] = [mx[0]['meta']['modelname'] for mx in msp]
    dat['pcellid'] = [mx[0]['meta']['cellid'] for mx in msp]
    dat['cellid'] = [mx[0]['meta']['cellid'] for mx in msp]
    dat['modelpath'] = [mx[0]['meta']['modelpath'] for mx in msp]
    # dat['fitter'] = [mx[0]['meta']['fitter'] for mx in msp]  #not saved anymore?
    dat['fitkey'] = [mx[0]['meta']['fitkey'] for mx in msp]
    dat['svn_branch'] = ['svd_fsDB' for mx in msp]
    sg_un = []
    for mx in msp:
        gi = nems.utils.find_module('state_gain', mx)
        if gi is None:
            g = np.full((2, 3), np.NaN)
        else:
            g = mx.phi[gi]['g']
            if g.shape[1] == 2:
                g = np.hstack((np.full((2, 1), 1), g))
        sg_un.append(g)
    dat['sg_un'] = sg_un
    dat['sg'] = [g / g[:, :1] + 1 for g in dat['sg_un']]
    dat['sgCA'] = [sg[0, 1] for sg in dat['sg']]
    dat['sgCB'] = [sg[1, 1] for sg in dat['sg']]
    dat['sgIA'] = [sg[0, 2] for sg in dat['sg']]
    dat['sgIB'] = [sg[1, 2] for sg in dat['sg']]
    subsetnames = ['A', 'B', 'C', 'I', 'A+B', 'C+I']
    varnames = ['r_fit', 'r_test', 'r_ceiling', 'r_floor', 'mse_fit', 'mse_test', 'se_mse_test', 'se_mse_fit']
    varnames_no_ss = ['se_test', 'se_fit']
    for vname in varnames:
        for ssname in subsetnames:
            if vname in mx[0]['meta'][ssname].keys():
                dat[vname + '_' + ssname] = [mx[0]['meta'][ssname][vname] for mx in msp]
            else:
                warnings.warn(
                    '{} not a variable for {},\nssname {}. Setting to NaN'.format(vname, mx[0]['meta']['xfspec'],
                                                                                  ssname))
                dat[vname + '_' + ssname] = [np.NaN] * len(msp)
        # dat[vname] = [mx[0]['meta'][vname] for mx in msp]
    for vname in (varnames + varnames_no_ss):
        dat[vname] = [mx[0]['meta'][vname][0] for mx in msp]
    return pd.DataFrame(data=dat)


def load_modelspec_data_pop(batch,rep_cellids,pop_modelnames):
    msp = []
    er = []
    for mod_i, m in enumerate(pop_modelnames):

        try:
            print('Loading modelname: {}'.format(m))
            mds = nems_db.params._get_modelspecs(rep_cellids, batch, m, multi='mean')
        except ValueError as error:
            er.append(error)
            if error.args[0][:28] == 'No result exists for:\nbatch:':
                print('No Results')
            else:
                raise (error)
        else:
            print('{} cells'.format(len(mds)))
            msp.extend(mds)
    dat = {}
    dat['loader'] = [mx[0]['meta']['recording'] for mx in msp]
    ss = []
    for mx in msp:
        s = mx[0]['meta']['recording'].split('subset')
        if len(s) == 1:
            ss.append('A+B+C+I')
        else:
            ss.append(s[1][1:].split('-')[0])
    dat['subset'] = ss
    dat['modelspecname'] = [mx[0]['meta']['modelspecname'] for mx in msp]
    dat['modelname'] = [mx[0]['meta']['modelname'] for mx in msp]
    dat['popcellid'] = [mx[0]['meta']['cellid'] for mx in msp]
    dat['pcellid'] = [mx[0]['meta']['cellid'] for mx in msp]
    dat['cellids'] = [mx[0]['meta']['cellids'] for mx in msp]
    dat['modelpath'] = [mx[0]['meta']['modelpath'] for mx in msp]
    # dat['fitter'] = [mx[0]['meta']['fitter'] for mx in msp]  #not saved anymore?
    dat['fitkey'] = [mx[0]['meta']['fitkey'] for mx in msp]
    #dat['svn_branch'] = ['svd_fsDB' for mx in msp]
    # sg_un = []
    # for mx in msp:
    #     gi = nems.utils.find_module('state_gain', mx)
    #     if gi is None:
    #         g = np.full((2, 3), np.NaN)
    #     else:
    #         g = mx.phi[gi]['g']
    #         if g.shape[1] == 2:
    #             g = np.hstack((np.full((2, 1), 1), g))
    #     sg_un.append(g)
    # dat['sg_un'] = sg_un
    # dat['sg'] = [g / g[:, :1] + 1 for g in dat['sg_un']]
    # dat['sgCA'] = [sg[0, 1] for sg in dat['sg']]
    # dat['sgCB'] = [sg[1, 1] for sg in dat['sg']]
    # dat['sgIA'] = [sg[0, 2] for sg in dat['sg']]
    # dat['sgIB'] = [sg[1, 2] for sg in dat['sg']]
    dat2 = {}
    for k,v in dat.items():
        if k == 'pcellid':
            list_of_lists = [cid for vv, cid in zip(v, dat['cellids'])]
        else:
            list_of_lists  = [[vv]*len(cid) for vv, cid in zip(v,dat['cellids'])]
        dat2[k] = list(itertools.chain(*list_of_lists))
    dat2['cellid'] = dat2['pcellid']
    subsetnames = ['A', 'B', 'C', 'I', 'A+B', 'C+I']
    varnames = ['r_fit', 'r_test', 'r_ceiling', 'r_floor', 'mse_fit', 'mse_test', 'se_mse_test', 'se_mse_fit']
    varnames_no_ss = ['se_test', 'se_fit']
    for vname in varnames:
        for ssname in subsetnames:
            if vname in mx[0]['meta'][ssname].keys():
                list_of_lists = [mx[0]['meta'][ssname][vname] for mx in msp]
                vals_in_arrays = list(itertools.chain(*list_of_lists))
                dat2[vname + '_' + ssname] = [v[0] for v in vals_in_arrays]
            else:
                warnings.warn(
                    '{} not a variable for {},\nssname {}. Setting to NaN'.format(vname, mx[0]['meta']['xfspec'],
                                                                                  ssname))
                dat2[vname + '_' + ssname] = np.NaN
        # dat[vname] = [mx[0]['meta'][vname] for mx in msp]
    for vname in (varnames + varnames_no_ss):
        list_of_lists = [mx[0]['meta'][vname] for mx in msp]
        vals_in_arrays = list(itertools.chain(*list_of_lists))
        dat2[vname] = [v[0] for v in vals_in_arrays]
    dfm = pd.DataFrame(data=dat2)
    return dfm