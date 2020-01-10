#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:30:15 2018

@author: svd
"""

import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
from nems.utils import escaped_split, escaped_join

#import nems_lbhb.baphy as nb
import nems.db as nd
#import nems_lbhb.xform_wrappers as nw

import logging

log = logging.getLogger(__name__)

import nems.xforms as xforms
from nems import get_setting
from nems.registry import KeywordRegistry
from nems.plugins import (default_keywords, default_loaders,
                          default_initializers, default_fitters)
#from nems.gui.recording_browser import browse_recording, browse_context
#from nems.gui.editors import browse_xform_fit, EditorWidget


cellid = 'TAR010c-18-1'
batch = 289
modelname = 'ozgf.fs100.ch18.pup-load-st.pup_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_basic.nf5'

fit_model_xform(cellid, batch, modelname, autoPlot=True, saveInDB=False)

# NAT SINGLE NEURON
batch = 289
cellid ='BRT026c-46-1'
cellid = 'gus019d-b2'
cellid ='bbl104h-13-1'
cellid = "TAR010c-18-2"
modelname="ozgf.fs100.ch18-ld-contrast.ms250-sev_dlog.f-wc.18x2.g-fir.2x15-lvl.1-ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.l_init.c-basic"
#modelname="ozgf.fs100.ch18-ld-contrast-sev_dlog.f-wc.18x2.g-fir.2x15-lvl.1-ctwc.18x1.g-ctfir.1x15-ctlvl.1-dsig.d_init.c.t3-basic"
modelname = "ozgf.fs50.ch18.pup-ld-st.pup0_dlog.f-wc.18x1.g-fird.1x10-lvl.1-dexp.1-stategain.S_jk.nf5-init.st-basic"
modelname = "psth.fs4.pup-ld-hrc-st.pup_sdexp.S_jk.nf20-psthfr.j-basic"
modelname = "ozgf.fs50.ch18-ld-sev_dlog.f-wc.18x2.g-fir.2x10-lvl.1-relu_init-basic"
modelname = "ozgf.fs50.ch18-ld-sev_dlog-wc.18x2.g-fir.2x15-lvl.1-dexp.1_init-basic"
modelname = "ozgf.fs50.ch18-ld-sev_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_init-iter.T4,5.fi5"
modelname = "ozgf.fs50.ch18-ld-sev_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_init-basic"
modelname = "ozgf.fs100.ch18-ld-norm-sev_wc.18x1.g-fir.1x15-relu.1_init-iter.cd.T4,5.fi5"
modelname = "ozgf.fs100.ch18-ld-norm-sev_wc.18x1.g-fir.1x15-relu.1_tf.s3"
modelname = "ozgf.fs100.ch18-ld-norm-sev_wc.18x1.g-fir.1x15-relu.1_init-basic"

# NAT pupil + population model
cellid='TAR010c'
modelname = "ozgf.fs50.ch18.pop-loadpop.cc25-pca.cc2.no-tev.vv34_dlog-wc.18x4.g-fir.2x10x2-wc.2xR-lvl.R_popiter.T3,4.fi5"
modelname = "ozgf.fs50.ch18.pop-loadpop.cc25-pca.cc3.no-tev.vv34_dlog-wc.18x3.g-fir.1x10x3-wc.3xR-lvl.R-dexp.R_popiter.T3,4.fi5"
modelname = "ozgf.fs50.ch18.pop-loadpop.cc20-pca.cc2.no-tev.vv34_wc.18x4.g-dlog.c4-fir.2x10x2-wc.2xR-lvl.R_popiter.T3,4.fi5"
modelname = "ozgf.fs50.ch18.pop-loadpop.cc20-pca.cc2.no-tev.vv34_dlog-wc.18x4.g-fir.2x10x2-relu.2-wc.2xR-lvl.R_popiter.T3,4.fi5"

modelname = 'ns.fs100.pupcnn.eysp-ld-st.pup-hrc-psthfr-mod.r_sdexp.S_jk.nf10-basic'
modelname = 'ns.fs20.pupcnn.eysp-ld-st.pup-tor-hrc-psthfr-mod.r_sdexp.S_jk.nf10-basic'
batch = 314
cellid = 'AMT003c-11-1'

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

if old:
    recording_uri = ogru(cellid, batch, loadkey)
    xfspec = oxfh.generate_loader_xfspec(loadkey, recording_uri)
    xfspec.append(['nems_lbhb.old_xforms.xforms.init_from_keywords',
                   {'keywordstring': modelspecname, 'meta': meta}])
    xfspec.extend(oxfh.generate_fitter_xfspec(fitkey))
    xfspec.append(['nems.analysis.api.standard_correlation', {},
                   ['est', 'val', 'modelspecs', 'rec'], ['modelspecs']])
    if autoPlot:
        log.info('Generating summary plot ...')
        xfspec.append(['nems.xforms.plot_summary', {}])
else:
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
    xfspec.append(['nems.xforms.predict', {}])
    xfspec.append(['nems.xforms.add_summary_statistics', {}])

    # 5) generate plots
    if autoPlot:
        xfspec.append(['nems.xforms.plot_summary', {}])

# equivalent of xforms.evaluate():

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

# Close the log, remove the handler, and add the 'log' string to context
log.info('Done (re-)evaluating xforms.')
ch.close()
rootlogger.removeFilter(ch)

log_xf = log_stream.getvalue()

# save some extra metadata
modelspec = ctx['modelspec']

#results_dir = nems.get_setting('NEMS_RESULTS_DIR')
#destination = '{0}/{1}/{2}/{3}/'.format(
#        results_dir, batch, cellid, ms.get_modelspec_longname(modelspec))
#modelspec.meta['modelpath'] = destination
#modelspec.meta['figurefile'] = destination+'figure.0000.png'

# save results
# log.info('Saving modelspec(s) to {0} ...'.format(destination))
# xforms.save_analysis(destination,
#                      recording=ctx['rec'],
#                      modelspec=modelspec,
#                      xfspec=xfspec,
#                      figures=ctx['figures'],
#                      log=log_xf)
""""
import nems.plots.api as nplt

cellid='bbl086b-03-1'
batch=289
modelname="ozgf.fs50.ch18-ld-sev_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_init-basic"

d=nd.get_results_file(batch=batch, cellids=[cellid], modelnames=[modelname])

filepath = d['modelpath'][0] + '/'
xfspec, ctx = xforms.load_analysis(filepath, eval_model=False)

ctx, log_xf = xforms.evaluate(xfspec, ctx)

nplt.quickplot(ctx)
"""


# save in database as well
if saveInDB:
    # TODO : db results finalized?
    nd.update_results_table(modelspec)

if browse_results:
    aw = browse_context(ctx, signals=['stim', 'pred', 'resp'])

    #ex = EditorWidget(modelspec=ctx['modelspec'], rec=ctx['val'], xfspec=xf,
    #                  ctx=ctx, parent=self)
    #ex.show()