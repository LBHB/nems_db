#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:30:15 2018

@author: svd
"""

import matplotlib.pyplot as plt
import numpy as np
import os

#import nems0.recording
import nems.modelspec as ms
import nems.xforms as xforms
import nems.plots.api as nplt

#import nems_db.baphy as nb
import nems_db.db as nd
import nems_db.xform_wrappers as nw

import logging
log = logging.getLogger(__name__)

"""
#cellid = 'zee021e-c1'
cellid = 'TAR010c-18-1'
#cellid = 'BRT033b-01-1'

example_list=['TAR010c-06-1','TAR010c-22-1',
              'TAR010c-60-1','TAR010c-44-1',
              'bbl074g-a1','bbl081d-a1','BRT006d-a1','BRT007c-a1',
              'BRT009a-a1','BRT015c-a1','BRT016f-a1','BRT017g-a1']
cellid = 'zee019b-a1'
batch=271
modelname = "ozgf100ch18_wcg18x1_fir1x15_lvl1_dexp1_fit01"

#cellid = 'chn069b-d1'

cellid='BRT033b-12-1'
batch=301
modelname = "nostim20pupbeh_stategain3_fitpjk01"

cellid = 'TAR010c-18-1'
batch = 271
modelname = "ozgf100ch18_wcg18x2_fir2x15_lvl1_dexp1_fit01"

cellid = 'TAR010c-22-1'
batch = 301
modelname = "nostim20pupbeh_stategain3_fitpjk01"
cellid = 'TAR010c-18-1'
batch = 271
modelname = "ozgf100ch18_dlog_wcg18x1_fir1x15_lvl1_dexp1_fit01"

cellid = 'eno025c-c1'
batch = 271
modelname = "ozgf100ch18_dlog_wcg18x1_stp1_fir1x15_lvl1_dexp1_fit01"
#modelname = "ozgf100ch18_wcg18x1_stp1pc_fir1x15_lvl1_fit01"
"""
cellid='chn009b-a1'
batch=259
modelname = "env100_dlogn2_wccn2x2_stp2_fir2x15_lvl1_dexp1_fit01"




xfspec, ctx = nw.load_model_baphy_xform(cellid, batch, modelname,
                                        eval_model=False)

for xfa in xfspec:
    if xfa[0] == 'nems.initializers.from_keywords_as_list':
        pass
    else:
        ctx = xforms.evaluate_step(xfa, ctx)

modelspecs = ctx['modelspecs']
val = ctx['val'][0]
# nplt.plot_summary(val, modelspecs)

nplt.quickplot(ctx)
