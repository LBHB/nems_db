#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:30:15 2018

@author: svd
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import io
#from sklearn.decomposition import PCA

# import nems0.recording
import nems0.modelspec as ms
import nems0.xforms as xforms
import nems0.xform_helper as xhelp
import nems0.utils
from nems0.recording import load_recording
import nems0.preprocessing as preproc
import nems0.epoch as ep
import nems0.modelspec as ms
# import nems_lbhb.baphy as nb
#import nems0.db as nd
import nems_lbhb.xform_wrappers as nw

from nems import get_setting
from nems0.registry import KeywordRegistry
from nems0.plugins import default_keywords
from nems0.plugins import default_loaders
from nems0.plugins import default_initializers
from nems0.plugins import default_fitters
from nems0.signal import RasterizedSignal
import nems0.plots.api as nplt
from nems0.fitters.api import scipy_minimize
#from nems0.gui.recording_browser import browse_recording, browse_context

import logging
log = logging.getLogger(__name__)

batch = 310
cellid = 'BRT037b-06-1'
loadkey = "env.fs100.cst"

outpath = '/tmp/'

ctx = nw.baphy_load_wrapper(cellid=cellid, batch=batch,
                            loadkey=loadkey)

ctx.update(xforms.load_recordings(
        cellid=cellid, save_other_cells_to_state=True, **ctx))

# if one stim is repeated a lot, can use it as val
#ctx.update(xforms.split_by_occurrence_counts(
#    epoch_regex='^STIM_', **ctx))

# uncommenting will only allow signal correlations to help
# ctx.update(xforms.average_away_stim_occurrences(**ctx))

# TODO, why doesn't this work?
#ctx.update(xforms.split_at_time(fraction=0.8, **ctx))
ctx.update(xforms.use_all_data_for_est_and_val(**ctx))

modelspec_name='wc.2x2.c-fir.2x15-lvl.1-stategain.S'

# record some meta data for display and saving
meta = {'cellid': cellid, 'batch': batch,
        'modelname': modelspec_name, 'recording': cellid}

ctx.update(xforms.init_from_keywords(modelspec_name, meta=meta, **ctx))

ctx.update(xforms.fit_basic_init(**ctx))
ctx.update(xforms.fit_basic(**ctx))

ctx.update(xforms.predict(**ctx))

ctx.update(xforms.add_summary_statistics(**ctx))

ctx.update(xforms.plot_summary(**ctx))

# save results
# log.info('Saving modelspec(s) to {0} ...'.format(destination))
modelspecs = ctx['modelspecs']

destination = '/auto/data/nems_db/results/{0}/{1}/{2}/'.format(
        batch, cellid, ms.get_modelspec_longname(modelspecs[0]))
modelspecs[0][0]['meta']['modelpath'] = destination
modelspecs[0][0]['meta']['figurefile'] = destination+'figure.0000.png'
modelspecs[0][0]['meta'].update(meta)
#
# xforms.save_analysis(destination,
#                      recording=ctx['rec'],
#                      modelspecs=modelspecs,
#                      xfspec=xfspec,
#                      figures=ctx['figures'],
#                      log=log_xf)
# TODO : db results finalized?
#nd.update_results_table(modelspecs[0])


"""
xfspec.append(['nems0.xforms.average_away_stim_occurrences', {}])

# MODEL SPEC
# modelspecname = 'dlog_wcg18x1_stp1_fir1x15_lvl1_dexp1'
modelspecname = 'wc.18x1.g_fir.1x15_lvl.1'

meta = {'cellid': 'TAR010c-18-1', 'batch': 271, 'modelname': modelspecname}

xfspec.append(['nems0.xforms.init_from_keywords',
               {'keywordstring': modelspecname, 'meta': meta}])

xfspec.append(['nems0.xforms.fit_basic_init', {}])
xfspec.append(['nems0.xforms.fit_basic', {}])
# xfspec.append(['nems0.xforms.fit_basic_shrink', {}])
#xfspec.append(['nems0.xforms.fit_basic_cd', {}])
# xfspec.append(['nems0.xforms.fit_iteratively', {}])
xfspec.append(['nems0.xforms.predict',    {}])
# xfspec.append(['nems0.xforms.add_summary_statistics',    {}])
xfspec.append(['nems0.analysis.api.standard_correlation', {},
               ['est', 'val', 'modelspecs', 'rec'], ['modelspecs']])

# GENERATE PLOTS
xfspec.append(['nems0.xforms.plot_summary',    {}])

# actually do the fit
ctx, log_xf = xforms.evaluate(xfspec)

"""