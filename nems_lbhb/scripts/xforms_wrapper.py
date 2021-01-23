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
cellid = 'TAR009d-42-1'  #OG cell
cellid = 'TAR009d-33-2'  #pretty decent as well
cellid = 'TAR009d-20-1'  #good strf weird predictions
cellid = 'TAR009d-33-3'  #pretty decent as well

cellid = 'TAR009d-21-1'

# current "best" NEMS model
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x3.g-fir.3x15-dexp.1_init-basic"         #fit=0.661, test=0.674

modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x2-fir.2x15-relu.1_init-basic"           #fit=0.628, test=0.655
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x2-fir.2x15-relu.1_tf.n"                 #fit=0.678, test=0.680
modelname = "ozgf.fs100.ch18-ld-norm-sev_fir.18x15-relu.1_tf.n"                             #fit=0.615, test=0.653

#gregtrying things
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x2.g-fir.1x18x2-relu.2-wc.2x1_tf.n"      #fit=0.643, test=0.668
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x3.g-fir.1x18x3-relu.3-wc.3x1_tf.n"      #fit=0.677, test=0.676
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x4.g-fir.2x18x2-relu.2-wc.2x1_tf.n"      #fit=0.682, test=0.675
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x4.g-fir.2x18x2-relu.2-wc.2x1_init-tf.n" #fit=0.625, test=0.668
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x9.g-fir.3x18x3-relu.3-wc.3x1_-tf.n"     #fit=0.680, test=0.674
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x4-fir.2x18x2-relu.2-wc.2x1_tf.n"        #fit=0.688, test=0.691
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x4-fir.1x18x4-relu.4-wc.4x1_tf.n"        #fit=0.681, test=0.684

modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x16-fir.4x18x4-relu.4-wc.4x1_tf.n"       #fit=0.695, test=0.699
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x16.g-fir.4x18x4-relu.4-wc.4x1_tf.n"     #fit=0.679, test=0.669
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x16-fir.4x18x4-relu.4-wc.4x1_tf.n.rb10"  #fit=0.707, test=0.699
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x16-fir.4x18x4-relu.4-wc.4x1_init-basic" #fit=0.628, test=0.643
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x16-fir.4x18x4-relu.4-wc.4x1_init-tf.n"  #fit=0.589, test=0.631


modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x25-fir.5x18x5-relu.5-wc.5x1_tf.n"       #fit=0.693, test=0.697

modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x16-fir.8x18x2-relu.2-wc.2x1_tf.n"       #fit=0.687, test=0.683
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x16-fir.2x18x8-relu.8-wc.8x1_tf.n"       #fit=0.675, test=0.681

modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x12-fir.4x18x3-relu.3-wc.3x1_tf.n"       #fit=0.692, test=0.696
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x12-fir.3x18x4-relu.4-wc.4x1_tf.n"       #fit=0.692, test=0.698

modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x15-fir.5x18x3-relu.3-wc.3x1_tf.n"       #fit=0.693, test=0.694
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x15-fir.3x18x5-relu.5-wc.5x1_tf.n"       #fit=0.693, test=0.697

modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x16-fir.16x18x1-relu.1-wc.1x1_tf.n"      #fit=0.669, test=0.674

#try best tf above and compare to NEMS
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x16-fir.4x18x4-relu.4-wc.4x1_tf.n"       #fit=0.694, test=0.698
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x16-fir.16x18-relu.1_init-basic"         #fit=0.629, test=0.660
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x4.g-fir.4x15-dexp.1_init-basic"         #fit=0.650, test=0.683
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x4-fir.4x15-dexp.1_init-basic"           #fit=0.661, test=0.666


#play with Net.train() - optimize hyperparameters
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x16-fir.4x18x4-relu.4-wc.4x1_tf.n"
#learning rate = 0.01, max iter=1000, eval_interval=30 ---fit=0.693, test=0.698 --- DEFAULTS
#learning rate = 0.001 ---fit=0.615, test=0.655
#learning rate = 0.005 --- fit=0.685, test=0.681
#learning rate = 0.5 --- fit=0.6, test=0.6
#lr = 0.01, maxiter=2500 --- fit=0.687, test=0.688
#lr = 0.01, maxiter=5000 --- fit=0.688, test=0.689
#lr = 0.01, mi=1000, eval_interval = 10 --- fit=0.638, test=0.632
#lr = 0.01, mi=1000, eval_interval = 50 --- fit=0.693, test=0.699
#lr = 0.01, mi=1000, eval_interval = 75 --- fit=0.706, test=0.707  .704 .710 on round 2
#lr = 0.01, mi=1000, eval_interval = 100 --- fit=0.691, test=0.694
#lr = 0.01, mi=1000, eval_interval = 60 --- fit=0.691, test=0.694

#lr = 0.005, mi=1000, eval_interval = 75 --- fit=0.689, test=0.691
#lr = 0.01, mi=2500, eval_interval = 75 --- fit=0.686, test=0.684
modelname = "ozgf.fs100.ch18-ld-norm-sev_dlog-wc.18x16-fir.4x18x4-relu.4-wc.4x1_tf.n.rb10"


autoPlot = True
saveInDB = False
save_data = False
browse_results = False

log.info('Initializing modelspec(s) for cell/batch %s/%d...',
         cellid, int(batch))

# Segment modelname for meta information, specifically the parts between _s
kws = escaped_split(modelname, '_')

modelspecname = escaped_join(kws[1:-1], '-')
loadkey = kws[0]                                             #name of loader, samplying freq,channels
fitkey = kws[-1]                                             #what the fitter will be (in my case now NEMS or tf

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

###########################################################################
###########################################################################
###########################################################################

import nems.gui.editors as gui
import nems.xform_helper as xhelp

#interesting cell, I think I can see what's going on there too
#cellid="BRT038b-30-1"
#modelname="ozgf.fs100.ch18-ld-sev_dlog-wc.18x3-fir.1x15x3-relu.3-wc.3x1-lvl.1-dexp.1_tf.n.rb5"
#modelname="ozgf.fs100.ch18-ld-sev_dlog-wc.18x16.g-fir.4x15x4-relu.4-wc.4x1-lvl.1-dexp.1_tf.n.rb5"
#modelname='ozgf.fs100.ch18-ld-sev_dlog-wc.18x12-fir.4x15x3-relu.3-wc.3x1-lvl.1-dexp.1_tf.n.rb5'

cellid="BRT033b-12-3"
batch=308
modelname="ozgf.fs100.ch18-ld-sev_dlog-wc.18x16-fir.4x15x4-relu.4-wc.4x1-lvl.1-dexp.1_tf.n.rb5"
# modelname="ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-fir.2x15-relu.1-lvl.1-dexp.1_tf.n.rb5"

xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname)
ex = gui.browse_xform_fit(ctx, xfspec)

########################################################
########################################################
#########################################################
########################################################


import nems.db as nd

batch=308

performance_data = nd.pd_query("SELECT modelname,cellid,r_test,r_floor FROM Results WHERE batch={}".format(batch))
performance_data['significant']=performance_data['r_test'] > 2*performance_data['r_floor']
performance_data['zero_test']=(performance_data['r_test']==0)

performance_data.groupby(['modelname'])['r_test','significant','zero_test'].mean()
performance_data.groupby(['modelname'])['zero_test'].mean()

