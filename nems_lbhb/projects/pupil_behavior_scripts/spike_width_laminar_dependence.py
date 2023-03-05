#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:18:27 2018

@author: hellerc
"""
import nems0.db as nd
import nems_lbhb.plots as pl
import nems_lbhb.stateplots as stateplots

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# select analysis / model
plt.close('all')
PLOT_ABS = True
model_set = "pb_AC_307_psths_basic-nf"

# load model fits
t = pd.read_csv('/auto/users/svd/projects/pupil-behavior/'+model_set+'/results.csv')
t = t.set_index('cellid')
cellids = t.index
diff = t['r_pb']-t['r_p0b0']

# screen for signficant cellids
cellids_sig = cellids[diff > np.std(diff)]
sig_mod  = (diff > np.std(diff))

# filter mod results by significant cellids
t_filtered = t.T.filter(cellids_sig).T

# =================   Plot master summary figure of single cells  ==============
# for all cells (compare sig vs. not sig)
f = stateplots.beta_comp(t['pup_mod'], t['beh_mod'], n1='pupil', n2='active',
                           title='mod index', hist_range=[-0.4, 0.4],
                           highlight=sig_mod)
# just for significant cells (compare fs vs. rs)
fs = []
for cid in t_filtered.index:
    try:
        if nd.get_wft(cid)==1:
            fs.append(1)
        else:
            fs.append(0)
    except:
        fs.append(0)

f2 = stateplots.beta_comp(t_filtered['pup_mod'], t_filtered['beh_mod'], n1='pupil', n2='active',
                           title='mod index for FS vs. RS', hist_range=[-0.5, 0.5],
                           highlight=fs)

f3 = stateplots.beta_comp(t_filtered['beh_mod_pup0'], t_filtered['beh_mod'],
                           n1='active-nopup', n2='active',
                           title='FS vs RS unique mod', hist_range=[-0.5, 0.5],
                           highlight=fs)

# ====== plot summary figure of mod indexes as function of depth ==================
fit_sites = ['TAR010c', 'BRT026c', 'BRT033b', 'bbl102d']
h_pup = []
h_beh = []
h_pup_abs = []
h_beh_abs= []
cids = []
l4 = []
for s in fit_sites:
    h_beh.append(t.T.filter(regex=s).T['beh_mod'].values)
    h_pup.append(t.T.filter(regex=s).T['pup_mod'].values)
    h_beh_abs.append(t.T.filter(regex=s).T['beh_mod_pup0'].values)
    h_pup_abs.append(t.T.filter(regex=s).T['pup_mod_beh0'].values)
    cids.append(t.T.filter(regex=s).T.index)

    # seting layer 4 channels based on CSD (to be used by plotting function)
    if s is 'TAR010c':
        l4.append(49)
    elif s is 'BRT026c':
        l4.append(49)
    elif s is 'BRT033b':
        l4.append(46)
    elif s is 'bbl102d':
        l4.append(52)

fs_all = []
for cid in t.index:
    try:
        if nd.get_wft(cid)==1:
            fs_all.append(1)
        else:
            fs_all.append(0)
    except:
        fs_all.append(0)

if PLOT_ABS:
    pl.depth_analysis_64D([np.abs(h) for h in h_pup], cids, l4, title='pup_mod')
    pl.depth_analysis_64D([np.abs(h) for h in h_beh], cids, l4, title='beh_mod')
else:
    pl.depth_analysis_64D(h_pup, cids, l4, title='pup_mod')
    pl.depth_analysis_64D(h_beh, cids, l4, title='beh_mod')

# ===============================================================================

