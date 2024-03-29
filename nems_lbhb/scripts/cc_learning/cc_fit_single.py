#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:30:15 2018

@author: svd
"""
import numpy as np

import nems0.modelspec as ms
import nems0.xforms as xforms
import nems0.xform_helper as xhelp
from nems0.utils import escaped_split, escaped_join, get_setting
import nems0.db as nd
import nems0.plots.api as nplt
import nems0.epoch as epoch
import matplotlib.pyplot as plt

TORC_MODEL = True

if TORC_MODEL:
    batch = 315

    # uncomment cell to analyze
    # cellid = "NMK003c-16-1"
    cellid = "NMK020c-29-1"

    # uncomment modelname to run

    # regress against pupil, shuffle file identity:
    # modelname = "psth.fs20.pup-ld-st.pup.fil0-ref.a-psthfr_stategain.S_jk.nf20.p-basic"

    # regress against pupil and file identity:
    modelname = "psth.fs20.pup-ld-st.pup.fil-ref.a-psthfr_stategain.S_jk.nf20.p-basic"
    modelname2 = "psth.fs20.pup-ld-st.pup.fil0-ref.a-psthfr_stategain.S_jk.nf20.p-basic"
    modelname = "psth.fs20.pup-ld-st.pup.hlf-ref.a-psthfr_stategain.S_jk.nf20.p-basic"
else:
    #alternative: TONE/TARGET only model

    batch=320

    # uncomment cell to analyze
    #cellid = "NMK003c-02-1"
    cellid = "NMK020c-29-1"
    modelname = "psth.fs20.pup-ld-st.pup.fil-tar-psthfr.tar_stategain.S_jk.nf20.p-basic"
    modelname2 = "psth.fs20.pup-ld-st.pup.fil0-tar-psthfr.tar_stategain.S_jk.nf20.p-basic"


browse_results = False

#save_file = xhelp.fit_model_xform(cellid, batch, modelname)
#xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)

xfspec, ctx = xhelp.fit_model_xform(cellid, batch, modelname,
                                    autoPlot=False, returnModel=True)
xfspec2, ctx2 = xhelp.fit_model_xform(cellid, batch, modelname2,
                                      autoPlot=False, returnModel=True)

modelspec = ctx['modelspec']
modelspec2 = ctx2['modelspec']
val = ctx['val']
r = val['resp']
state_channels = val['state'].chans
file_epochs = r.epochs.loc[r.epochs.name.str.startswith("FILE")]

modelspec[0]['plot_fn_idx'] = 5

if browse_results:
    import nems0.gui.editors as gui

    ex = gui.browse_xform_fit(ctx, xfspec)
else:
    modelspec.quickplot()

for i, s in enumerate(state_channels):
    # find name of current file
    print("{}: offset={:.3f} gain={:.3f} MI={:.3f} unique MI={:.3f}".format(
        s, modelspec.phi[0]['d'][0, i], modelspec.phi[0]['g'][0, i],
        modelspec.meta['state_mod'][i],
        modelspec.meta['state_mod'][i]-modelspec2.meta['state_mod'][i]))
