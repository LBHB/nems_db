#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:30:15 2018

@author: svd
"""
import numpy as np

import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
from nems.utils import escaped_split, escaped_join, get_setting
import nems.db as nd
import nems.gui.editors as gui
import nems.plots.api as nplt
import nems.epoch as epoch
import matplotlib.pyplot as plt

browse_results = False

cellid = 'chn022c-a2'
batch = 269
modelname = 'rdtld-rdtshf-rdtsev.j.10-rdtfmt_rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x2.g-fir.2x15-lvl.1_init-basic'

xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)

if browse_results:
    ex = gui.browse_xform_fit(ctx, xfspec)
else:
    ctx['modelspec'].quickplot()


r=ctx['rec']['resp']
dual = r.get_epoch_indices('dual')
rrnd9 = r.get_epoch_indices('Stim , 10 , Reference')
rnd9 = r.get_epoch_indices('Stim , 10 , Target')
rep9 = r.get_epoch_indices('Stim , 10 , TargetRep')
rep9 = r.get_epoch_indices('target_1_repeating_dual')

a = epoch.epoch_intersection(rep9, dual)

raster_rnd = r.extract_epoch('target_0')
raster_rep = r.extract_epoch('target_0_repeating_dual')
plt.figure()
plt.subplot(2,2,1)
i, j = np.where(raster_rnd[:,0,:])
i += 1
plt.plot(j, i,' k.')
plt.title('rand')

plt.subplot(2,2,2)
i, j = np.where(raster_rep[:,0,:])
i += 1
plt.plot(j, i,' k.')
plt.title('rep')

plt.subplot(2,2,3)
plt.plot(np.mean(raster_rnd[:,0,:],axis=0))
plt.subplot(2,2,4)
plt.plot(np.mean(raster_rep[:,0,:],axis=0))
