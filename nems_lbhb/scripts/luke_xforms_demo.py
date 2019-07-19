#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:30:15 2018

@author: svd
"""
import os
os.environ['OPENBLAS_CORETYPE'] = 'sandybridge'
os.environ['OPENBLAS_VERBOSE'] = '2'
#import nems.recording
import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
import nems.utils

import matplotlib.pyplot as plt
import numpy as np
import io

#import nems_lbhb.baphy as nb
import nems.db as nd
import nems_lbhb.xform_wrappers as nw
from nems_lbhb.old_xforms.xform_wrappers import generate_recording_uri as ogru
import nems_lbhb.old_xforms.xform_helper as oxfh
import logging

logging.basicConfig(level='DEBUG')
log = logging.getLogger(__name__)


import nems.xforms as xforms
from nems import get_setting
from nems.registry import KeywordRegistry
from nems.plugins import default_keywords
from nems.plugins import default_loaders
from nems.plugins import default_initializers
from nems.plugins import default_fitters
from nems.gui.recording_browser import browse_recording, browse_context

def _xform_exists(xfspec, xform_fn):
    for xf in xfspec:
        if xf[0] == xform_fn:
            return True
        
batch = 289
cellid ='BRT026c-46-1'
modelname = "ozgf.fs100.ch18-ld-sev_dlog.f-wc.18x2.g-fir.2x15-lvl.1_init-basic"

batch=307
cellid="TAR010c-06-1"

modelname = "psth.fs20.pup-ld-st.pup0.beh0-ref-psthfr_stategain.S_jk.nf10-basic"
modelname = "psth.fs20.pup-ld-st.pup.beh-ref-psthfr_stategain.S_jk.nf10-basic"

batch=306
cellid='fre192b-03-1_6236-1701'
cellid='fre196b-35-1_1417-1233'
cellid='fre192b-26-1_6236-7017' #strongly affected by init.nl1 vs init.nl2
cellid='fre195b-18-1_774-242' # Unstable when using nems vs nemsenv
cellid='fre192c-39-1_7920-7049'
cellid='fre192b-29-1_6236-1701'
cellid='fre197c-76-1_1963-1024' # neg gains
cellid='fre196b-15-2_977-1270' #diff resutls for stategain L model baseline vs no baseline
modelname='env.fs100-SPOld-sev_dlog-fir.2x15-lvl.1-dexp.1_init.nl1-basic-SPOpf'
modelname='env.fs100-SPOld-sev_dlog-fir.2x15-lvl.1-dexp.1_init.nl2-basic-SPOpf.SDB'
modelname='env.fs100-SPOld-sev_dlog-wc.2x2.c-fir.2x15-lvl.1-dexp.1_init.nl1-basic-SPOpf.SDB'
modelname='env.fs100-SPOld-subset.A+B-sev_dlog-fir.2x15-lvl.1-dexp.1_init.nl1-basic-SPOpf.SDB'
modelname='SDB-env.fs100-SPOld-stSPO-sev-shuf.st_dlog-stategain.3x2.g-fir.2x15-lvl.1-dexp.1_init-SPOpf'
#modelname='SDB-env.fs100-SPOld-stSPO.nb-sev_dlog-stategain.2x2.g.o1-fir.2x15-lvl.1-dexp.1_init-basic-SPOpf'
#modelname='SDB-env.fs100-SPOld-stSPO.nb-sev_dlog-stategain.2x2.g.o1-fir.2x15-lvl.1-dexp.1_init-SPOpf'

#modelname='SDB-env.fs100-SPOld-stSPO-sev-shuf.st_dlog-stategain.3x2.g-fir.2x15-lvl.1-dexp.1_init-basic-SPOpf'
#modelname='SDB-env.fs200-SPOld-sev_dlog-fir.2x15-lvl.1-dexp.1_init-basic-SPOpf'
#modelname='SDB-env.fs100-SPOld-stSPO-sev_dlog-stategain.3x2.g-stp.2-fir.2x15-lvl.1-dexp.1_init-basic-SPOpf'
#Shuffled ver here

#modelname='env.fs100-SPOld-sev_dlog-fir.2x15-lvl.1-dexp.1_init'
#modelname='env.fs100-SPOld-sev_dlog-fir.2x15-lvl.1-dexp.1_init.nl2-basic-SPOpf'

autoPred = True
autoStats = True
autoPlot = False

saveToFile=False
saveInDB = False
browse_results = False

log.info('Initializing modelspec(s) for cell/batch %s/%d...',
         cellid, int(batch))

# Segment modelname for meta information
kws = modelname.split("_")
modelspecname = "-".join(kws[1:-1])
loadkey = kws[0]
fitkey = kws[-1]

meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
        'loader': loadkey, 'fitkey': fitkey, 'modelspecname': modelspecname,
        'username': 'nems', 'labgroup': 'lbhb', 'public': 1,
        'githash': os.environ.get('CODEHASH', ''),
        'recording': loadkey}

load_keywords, model_keywords, fit_keywords = modelname.split("_")

# xforms_kwargs = {'cellid': cellid, 'batch': int(batch)}
xforms_kwargs = {}
xforms_init_context = {'cellid': cellid, 'batch': int(batch),
                       'meta': meta, 'keywordstring': model_keywords}
xforms_lib = KeywordRegistry(**xforms_kwargs)
xforms_lib.register_modules([default_loaders, default_fitters,
                             default_initializers])
xforms_lib.register_plugins(get_setting('XFORMS_PLUGINS'))

keyword_lib = KeywordRegistry()
keyword_lib.register_module(default_keywords)
keyword_lib.register_plugins(get_setting('KEYWORD_PLUGINS'))

# Generate the xfspec, which defines the sequence of events
# to run through (like a packaged-up script)
xfspec = []

# 0) set up initial context
xfspec.append(['nems.xforms.init_context', xforms_init_context])

# 1) Load the data
xfspec.extend(xhelp._parse_kw_string(load_keywords, xforms_lib))

# 2) generate a modelspec
xfspec.append(['nems.xforms.init_from_keywords', {'registry': keyword_lib}])
#xfspec.append(['nems.xforms.init_from_keywords', {}])

# 3) fit the data
xfspec.extend(xhelp._parse_kw_string(fit_keywords, xforms_lib))

# 4) generate a prediction (optional)
if autoPred:
    if not _xform_exists(xfspec, 'nems.xforms.predict'):
        xfspec.append(['nems.xforms.predict', {}])

# 5) add some performance statistics (optional)
if autoStats:
    if not _xform_exists(xfspec, 'nems.xforms.add_summary_statistics'):
        xfspec.append(['nems.xforms.add_summary_statistics', {}])

# 5) generate plots
if autoPlot:
    if not _xform_exists(xfspec, 'nems.xforms.plot_summary'):
        log.info('Generating summary plot...')
        xfspec.append(['nems.xforms.plot_summary', {}])

# Create a log stream set to the debug level; add it as a root log handler
log_stream = io.StringIO()
ch = logging.StreamHandler(log_stream)
ch.setLevel(logging.DEBUG)
fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(fmt)
ch.setFormatter(formatter)
rootlogger = logging.getLogger()
rootlogger.addHandler(ch)

ctx = {}
for xfa in xfspec:
    ctx = xforms.evaluate_step(xfa, ctx)


if False:
    Ipred = np.where([xf[0]=='nems.xforms.predict' for xf in xfspec])[0][0]
    Ifit_init = np.where([xf[0]=='nems.xforms.fit_basic_init' for xf in xfspec])[0][0]
    Ifit = np.where([xf[0]=='nems.xforms.fit_basic' for xf in xfspec])[0][0]
    Isbc = np.where([xf[0]=='nems_lbhb.postprocessing.add_summary_statistics_by_condition' for xf in xfspec])[0][0]
    Ipav = np.where([xf[0]=='nems_lbhb.SPO_helpers.plot_all_vals_' for xf in xfspec])[0][0]
    ctx = {}
    for xfa in xfspec[:Ifit_init+1]:
        ctx = xforms.evaluate_step(xfa, ctx)
    ctxI=ctx.copy()
    
    ctx=ctxI.copy()
    ctx['modelspecs'][0] = init_dexp(ctx['est'], ctx['modelspecs'][0])
    #ctx['modelspecs'][0][3]['phi']['kappa'][0]=5
    #ctx['modelspecs'][0][3]['phi']['amplitude'][0]=.09
    xfi=list(range(Ifit_init+1,len(xfspec)))
    xfi.remove(Ifit); 
    xfi.remove(Isbc); xfi.remove(Ipav)
    #xfi.insert(0,Ifit_init)
    for xfa in [xfspec[i] for i in xfi]:
        ctx = xforms.evaluate_step(xfa, ctx)
    ctx['modelspecs'][0][0]['meta']['r_test']
    
    ctxF=ctx.copy()

# Close the log, remove the handler, and add the 'log' string to context
log.info('Done (re-)evaluating xforms.')
ch.close()
rootlogger.removeFilter(ch)

log_xf = log_stream.getvalue()

# save some extra metadata
modelspec = ctx['modelspec']
#
destination = '/auto/data/nems_db/results/{0}/{1}/{2}/'.format(
        batch, cellid, ms.get_modelspec_longname(modelspec))
modelspec['meta']['modelpath'] = destination
modelspec['meta']['figurefile'] = destination+'figure.0000.png'

# save results
if saveToFile:
    log.info('Saving modelspec(s) to {0} ...'.format(destination))
    xforms.save_analysis(destination,
                      recording=ctx['rec'],
                      modelspec=modelspec,
                      xfspec=xfspec,
                      figures=ctx['figures'],
                      log=log_xf)

# save in database as well
if saveInDB:
    # TODO : db results finalized?
    nd.update_results_table(modelspec)

if browse_results:
    aw = browse_context(ctx, signals=['stim', 'pred', 'resp'])
