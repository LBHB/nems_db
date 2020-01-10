#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:28:11 2018

@author: svd
"""

import os
import logging
import time
from itertools import groupby

import nems.modelspec as ms
import nems.xforms as xforms
from nems.xform_helper import _xform_exists
import nems.xform_helper as xhelp
import nems.db as nd
from nems import get_setting
from nems.utils import escaped_split, escaped_join
from nems.registry import KeywordRegistry
from nems.plugins import (default_keywords, default_loaders,
                          default_initializers, default_fitters)
from nems.gui.recording_browser import browse_recording, browse_context
import nems.gui.editors as gui

log = logging.getLogger(__name__)

batch=307
cellids = ['TAR010c-21-1','bbl102d-01-1', 'BRT026c-02-2', 'BRT033b-02-1',
           'BRT036b-29-1', 'BRT037b-13-2',
           ]
fit_alg = "basic"

#modelname="parm.fs40.pup-ld-st.beh.pup-ref-pca.psth.no_wc.15x1.g-fir.1x8-lvl.1-stategain.S-lvl.R_jk.nf5.o-initpop-iter.pop.T4,5,6"
modelname="parm.fs40.pup-ld-st.beh.pup-ref_wc.15x2.g-fir.1x8x2-lvl.2-wc.2xR-stategain.S_jk.nf5.o-init.st-tf.n"
modelname="parm.fs40.pup-ld-st.beh.pup-ref_wc.15x2.g-fir.2x8-lvl.1-stategain.S_jk.nf5.o-init.st-tf.n"

cellid = "TAR010c-21-1"
#cellid = 'BRT032e-15-1'


autoPlot = True
saveInDB = False
browse_results = False

log.info('Initializing modelspec(s) for cell/batch %s/%d...',
         cellid, int(batch))

# Segment modelname for meta information
kws = modelname.split("_")
old = False
if (len(kws) > 3) or ((len(kws) == 3) and kws[1].startswith('stategain') and not kws[1].startswith('stategain.')):
    # Check if modelname uses old format.
    log.info("Using old modelname format ... ")
    old = True
    modelspecname = '_'.join(kws[1:-1])
else:
    modelspecname = "-".join(kws[1:-1])
loadkey = kws[0]
fitkey = kws[-1]

meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
        'loader': loadkey, 'fitkey': fitkey, 'modelspecname': modelspecname,
        'username': 'nems', 'labgroup': 'lbhb', 'public': 1,
        'githash': os.environ.get('CODEHASH', ''),
        'recording': loadkey}

#recording_uri = nw.generate_recording_uri(cellid, batch, loadkey)
# code from
# xfspec = xhelp.generate_xforms_spec(recording_uri, modelname, meta)
"""
{'stim': 0, 'chancount': 0, 'pupil': 1, 'rasterfs': 20, 'rawid': None, 'cellid': 'BRT026c-15-1', 'pupil_median': 0, 'pertrial': 0, 'pupil_deblink': 1, 'stimfmt': 'parm', 'runclass': None, 'includeprestim': 1, 'batch': 307}
{'stimfmt': 'parm', 'chancount': 0, 'pupil': 1, 'rasterfs': 20, 'rawid': None, 'cellid': 'BRT026c-15-1', 'pupil_median': 0, 'pertrial': 0, 'pupil_deblink': 1, 'stim': 0, 'runclass': None, 'includeprestim': 1, 'batch': 307}
"""
#log.info('Initializing modelspec(s) for recording/model {0}/{1}...'
#         .format(recording_uri, modelname))
xforms_kwargs = {}
xforms_init_context = {'cellid': cellid, 'batch': int(batch)}
recording_uri = None
kw_kwargs ={}

# equivalent of xform_helper.generate_xforms_spec():

# parse modelname and assemble xfspecs for loader and fitter
load_keywords, model_keywords, fit_keywords = escaped_split(modelname, '_')
if recording_uri is not None:
    xforms_lib = KeywordRegistry(recording_uri=recording_uri, **xforms_kwargs)
else:
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
if xforms_init_context is None:
    xforms_init_context = {}
if kw_kwargs is not None:
     xforms_init_context['kw_kwargs'] = kw_kwargs
xforms_init_context['keywordstring'] = model_keywords
xforms_init_context['meta'] = meta
xfspec.append(['nems.xforms.init_context', xforms_init_context])

# 1) Load the data
xfspec.extend(xhelp._parse_kw_string(load_keywords, xforms_lib))

# 2) generate a modelspec
xfspec.append(['nems.xforms.init_from_keywords', {'registry': keyword_lib}])
#xfspec.append(['nems.xforms.init_from_keywords', {}])

# 3) fit the data
xfspec.extend(xhelp._parse_kw_string(fit_keywords, xforms_lib))

# 4) add some performance statistics
if not _xform_exists(xfspec, 'nems.xforms.predict'):
    xfspec.append(['nems.xforms.predict', {}])

# 5) add some performance statistics (optional)
if not _xform_exists(xfspec, 'nems.xforms.add_summary_statistics'):
    xfspec.append(['nems.xforms.add_summary_statistics', {}])

# 6) generate plots (optional)
if autoPlot and not _xform_exists(xfspec, 'nems.xforms.plot_summary'):
    xfspec.append(['nems.xforms.plot_summary', {}])

# equivalent of xforms.evaluate():
ctx = {}
for xfa in xfspec:
    ctx = xforms.evaluate_step(xfa, ctx)

# Close the log, remove the handler, and add the 'log' string to context
log.info('Done (re-)evaluating xforms.')

# save some extra metadata
modelspec = ctx['modelspec']

# save in database as well
if saveInDB:
    # TODO : db results finalized?
    nd.update_results_table(modelspec)

if browse_results:
    #from nems.gui.recording_browser import browse_context
    #aw = browse_context(ctx, rec='val', signals=['stim', 'pred', 'resp'])

    ex = gui.browse_xform_fit(ctx, xfspec)

"""
pause=0.01

user="svd"
force_rerun=True
executable_path="/auto/users/svd/bin/miniconda3/envs/nems2/bin/python"
script_path="/auto/users/svd/python/nems_db/nems_fit_single.py"

for i in range(len(cellids)):
    for m in modelnames:
        for pc_idx in range(10):

            cellid = "{}_P{}".format(cellids[i],pc_idx)

            queueid, res = nd.enqueue_single_model(cellid, batch, m,
                                    user=user, force_rerun=force_rerun,
                                    executable_path=executable_path,
                                    script_path=script_path)
            print("{}: {}".format(queueid,res))
            time.sleep(pause)

            # enqueue_single_model(
            #         cellid, batch, modelname, user=None,
            #         session=None,
            #         force_rerun=False, codeHash="master", jerbQuery='',
            #         executable_path=None, script_path=None)
"""
