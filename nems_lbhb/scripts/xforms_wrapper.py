#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:30:15 2018

@author: svd
"""

import matplotlib.pyplot as plt
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

import numpy as np
import os
import io
import logging

import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
from nems.utils import escaped_split, escaped_join
import nems.db as nd
from nems import get_setting
from nems.xform_helper import _xform_exists, generate_xforms_spec
from nems.registry import KeywordRegistry
from nems.plugins import (default_keywords, default_loaders,
                          default_initializers, default_fitters)
from nems.gui.recording_browser import browse_recording, browse_context
import nems.gui.editors as gui

log = logging.getLogger(__name__)

# NAT SINGLE NEURON
batch = 289
cellid = 'TAR009d-42-1'

# current "best" NEMS model
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x3.g-fir.3x15-dexp.1_init-basic"

modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x2-fir.2x15-relu.1_init-basic"
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x2-fir.2x15-relu.1_tf.n"
modelname = "ozgf.fs100.ch18-ld-norm-sev_fir.18x15-relu.1_tf.n"

autoPlot = True
saveInDB = False
save_data = False
browse_results = False

log.info('Initializing modelspec(s) for cell/batch %s/%d...',
         cellid, int(batch))

# Segment modelname for meta information
kws = escaped_split(modelname, '_')

modelspecname = escaped_join(kws[1:-1], '-')
loadkey = kws[0]
fitkey = kws[-1]

meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
        'loader': loadkey, 'fitkey': fitkey, 'modelspecname': modelspecname,
        'username': 'nems', 'labgroup': 'lbhb', 'public': 1,
        'githash': os.environ.get('CODEHASH', ''),
        'recording': loadkey}

if type(cellid) is list:
    meta['siteid'] = cellid[0][:7]

# registry_args = {'cellid': cellid, 'batch': int(batch)}
registry_args = {}
xforms_init_context = {'cellid': cellid, 'batch': int(batch)}

log.info("TODO: simplify generate_xforms_spec parameters")
xfspec = generate_xforms_spec(recording_uri=None, modelname=modelname,
                              meta=meta,  xforms_kwargs=registry_args,
                              xforms_init_context=xforms_init_context,
                              autoPlot=autoPlot)
log.info(xfspec)

# actually do the loading, preprocessing, fit
ctx, log_xf = xforms.evaluate(xfspec)

# save some extra metadata
modelspec = ctx['modelspec']

# this code may not be necessary any more.
#destination = '{0}/{1}/{2}/{3}'.format(
#    get_setting('NEMS_RESULTS_DIR'), batch, cellid, modelspec.get_longname())
if type(cellid) is list:
    destination = os.path.join(
        get_setting('NEMS_RESULTS_DIR'), str(batch),
        cellid[0][:7], modelspec.get_longname())
else:
    destination = os.path.join(
        get_setting('NEMS_RESULTS_DIR'), str(batch),
        cellid, modelspec.get_longname())
modelspec.meta['modelpath'] = destination
modelspec.meta['figurefile'] = os.path.join(destination, 'figure.0000.png')
modelspec.meta.update(meta)

# save results
log.info('Saving modelspec(s) to {0} ...'.format(destination))
if 'figures' in ctx.keys():
    figs = ctx['figures']
else:
    figs = []
save_data = xforms.save_analysis(destination,
                                 recording=ctx['rec'],
                                 modelspec=modelspec,
                                 xfspec=xfspec,
                                 figures=figs,
                                 log=log_xf)

# save in database as well
if saveInDB:
    # TODO : db results finalized?
    nd.update_results_table(modelspec)

if browse_results:
    ex = gui.browse_xform_fit(ctx, xfspec)
