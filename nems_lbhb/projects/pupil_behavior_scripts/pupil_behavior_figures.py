#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:59:38 2018

@author: svd
"""

import os
import sys
sys.path.append(os.path.abspath('/auto/users/svd/python/scripts/'))

import nems.db as nd
import nems_db.params
import numpy as np
import matplotlib.pyplot as plt
import nems_lbhb.stateplots as stateplots


# fittype = "basic-nf-shr"
#fittype = "cd-nf"
fittype = "basic-nf"
pretype = "psth"
#pretype = "psths"

batch = 309


if 1:
    modelname0 = pretype + "20pup0beh0_stategain3_" + fittype
    modelname = pretype + "20pupbeh_stategain3_" + fittype
    modelname2 = pretype + "20pup0beh_stategain3_" + fittype
    bins = 15

    factor_names = ['baseline', 'pupil', 'active']
    i1 = 1
    i2 = 2
    n1 = factor_names[i1] + ' gain'
    n2 = factor_names[i2] + ' gain'

elif 1:
    modelname0 = pretype + "20pup0beh0_stategain3_" + fittype
    modelname = pretype + "20pupprebeh_stategain4_" + fittype
    modelname2 = pretype + "20pup0prebeh_stategain4_" + fittype
    bins = 15
    factor_names = ['baseline', 'pupil', 'pre', 'active']
    i1 = 3
    i2 = 2
    n1 = factor_names[i1] + ' gain'
    n2 = factor_names[i2] + ' gain'

elif 0:
    batch = 305
    modelname0 = "psth20pre0beh0_stategain3_cd-nf"
    #modelname = "psth20prebeh_stategain3_cd-nf"
    modelname = "psth20pre0beh_stategain3_cd-nf"
    bins = 15
    hist_range = [-1, 1]
    n1 = 'active beta'
    i1 = 2
    n2 = 'pre-post beta'
    i2 = 1
    factor_names = ['baseline', 'pre', 'active']

if batch in [303, 309]:
    hist_range = [-0.5, 0.5]
    hist_range = [-1, 1]
else:
    hist_range = [-1, 1]

res = nd.batch_comp(batch, [modelname0, modelname])

d1 = nems_db.params.fitted_params_per_batch(
        batch, modelname,
        stats_keys=[])
d2 = nems_db.params.fitted_params_per_batch(
        batch, modelname2,
        stats_keys=[])

# parse modelname
kws = modelname.split("_")
modname = kws[1]
statecount = int(modname[-1])

g1 = np.zeros([0, statecount])
b1 = np.zeros([0, statecount])
g2 = np.zeros([0, statecount])
b2 = np.zeros([0, statecount])
sig = np.zeros(len(d1.columns))
ff = res[modelname] >= res[modelname0] + 0.0075

i = 0
for c in d1.columns:
    # print(c)
    # print(x)

    x = d1.loc['0--'+modname+'--g'][c]
    g1 = np.append(g1, np.reshape(x, [1, -1]), axis=0)
    y = d1.loc['0--'+modname+'--d'][c]
    b1 = np.append(b1, np.reshape(y, [1, -1]), axis=0)

    x = d2.loc['0--'+modname+'--g'][c]
    g2 = np.append(g2, np.reshape(x, [1, -1]), axis=0)
    y = d2.loc['0--'+modname+'--d'][c]
    b2 = np.append(b2, np.reshape(y, [1, -1]), axis=0)

    print("{0}: {1}={2:.3f} {3}={4:.3f}  {5}={6:.3f} {7}={8:.3f}".
          format(c, n1, g1[-1, i1], n2, g1[-1, i2],
                 n1, g2[-1, i1], n2, g2[-1, i2]))

    sig[i] = ff.loc[c]
    i += 1

plt.close('all')

title = '{}/{}'.format(modelname, batch)
stateplots.beta_comp(g1[:, i1], g1[:, i2],
                   n1=factor_names[i1], n2=factor_names[i2],
                   hist_bins=bins, hist_range=hist_range, title=title,
                   highlight=sig)

title = '{}/{}'.format(n2, batch)
stateplots.beta_comp(g1[:, i2], g2[:, i2],
                   n1=modelname, n2=modelname2,
                   hist_bins=bins, hist_range=hist_range, title=title,
                   highlight=sig)

