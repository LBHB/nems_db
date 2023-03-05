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

#import nems0.recording
import nems0.modelspec as ms
import nems0.xforms as xforms
import nems0.xform_helper as xhelp
from nems0.utils import escaped_split, escaped_join

#import nems_lbhb.baphy as nb
import nems0.db as nd
#import nems_lbhb.xform_wrappers as nw

import logging

log = logging.getLogger(__name__)

import nems0.xforms as xforms
from nems import get_setting
from nems0.registry import KeywordRegistry
from nems0.plugins import (default_keywords, default_loaders,
                          default_initializers, default_fitters)
#from nems0.gui.recording_browser import browse_recording, browse_context
#from nems0.gui.editors import browse_xform_fit, EditorWidget


cellid = 'TAR010c-18-1'
batch = 289
modelname = 'ozgf100ch18pup_dlog_wcg18x1_fir1x15_lvl1_dexp1_basic-nf'
modelname = 'ozgf.fs100.ch18.pup-load-st.pup_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_basic.nf5'


cellid = 'ley041l-a1'
batch = 303
#modelname = "evt20pupbehtarlic_firNx40_lvl1_stategain3_basic-nf"
#modelname = "evt.fs20.pupbehtarlic_fir.Nx40-lvl.1-stategain.3_basic.st.nf10"
modelname = "psth.fs20.pup-ld-st.pup.beh-evs.tar.lic_fir.Nx40-lvl.1-stategain.3_jk.nf5-init.st-basic"

#cellid = 'BRT036b-07-1'
cellid = 'TAR010c-06-1'
batch = 307
#modelname = "psth.fs20.pupbeh_stategain.3_basic.st.nf10"
modelname = "psth.fs20.s-st.pup0.pas-pas_stategain.N_basic.st.nf10"
modelname = "psth.fs20-st.pup.pas-pas_stategain.N_basic.st.nf10"
#modelname = "psth.fs20-st.pup.beh_sdexp.S_basic.st.nf10"
modelname = "psth20pupbeh_stategain3_basic-nf"
modelname = "psth.fs20-ld-st.pup.beh-ref-psthfr_stategain.S_jk.nf10-init.st-basic"
#modelname = "psth.fs20-ld-st.pup.pas-ref-pas-psthfr_stategain.N_jk.nf10-init.st-basic"
modelname = "psth.fs20.pup-ld-st.pbs.pev.beh-ref-psthfr_sdexp.S_jk.nf10-init.st-basic"

cellid = 'TAR009d-15-1'
batch = 289
cellid='bbl099g-52-1'
batch=271
#modelname = 'ozgf100ch18_dlog_wcg18x1_fir1x15_lvl1_dexp1_basic'
#modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_init-basic'
#modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_init-basic'
#modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-stp.2-fir.2x15-lvl.1-dexp.1_init-basic'
#modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_init-basic'
modelname = 'ozgf.fs50.ch18-ld-sev_dlog-wc.18x2.g-fir.2x15-lvl.1-dexp.1_init-basic'
modelname = 'ozgf.fs100.ch18.pup-ld-st.pup_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1-stategain.S_jk.nf5-init.st-basic'
modelname='ozgf.fs100.ch18-ld-contrast.ms100_dlog.f-wc.18x2.g-fir.2x15-lvl.1-ctwc.18x2.g-ctfir.2x15-ctlvl.1-dsig.d_sev-init.c.t3-basic.t5'

cellid  = "por020a-c1" # 'chn067b-b1'  # 'por016d-b1'
batch = 259
#modelname = 'env100_dlog_wcc2x1_fir1x15_lvl1_dexp1_basic'
#modelname = 'env.fs100-ld-sev_dlog-wc.2x1.c-fir.1x15-lvl.1-dexp.1_init-basic'
#modelname = 'env.fs100-ld-sev_dlog-fir.2x15-stp.1-lvl.1-dexp.1_init-basic'
modelname = 'env100_dlog_fir2x15_stp1_lvl1_dexp1_basic'
modelname = 'env.fs100-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic'
modelname = 'env.fs100-ld-sev_dlog.f-wc.2x2.c.n-stp.2-fir.2x15-lvl.1-dexp.1_init-basic'
modelname = 'env.fs200-ld-sev_dlog.f-fir.2x15-lvl.1-dexp.1_init-basic'
modelname = 'env.fs100-ld-sev_dlog.f-wc.2x3.c.n-stp.3-fir.3x15-lvl.1-dexp.1_init-basic'




cellid='bbl099g-52-1'
batch=271
cellid = 'TAR010c-58-2'
#cellid='bbl099g-52-1'
batch = 289
#modelname = 'ozgf100ch18_dlog_wcg18x1_fir1x15_lvl1_dexp1_basic'
#modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_init-basic'
#modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_init-basic'
#modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x2.g-stp.2-fir.2x15-lvl.1-dexp.1_init-basic'
#modelname = 'ozgf.fs100.ch18-ld-sev_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_init-basic'
modelname = 'ozgf.fs50.ch18-ld-sev_dlog-wc.18x2.g-fir.2x15-lvl.1-dexp.1_init-basic'
modelname = 'ozgf.fs100.ch18.pup-ld-st.pup_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1-stategain.S_jk.nf5-init.st-basic'
modelname='ozgf.fs100.ch18-ld-sev_dlog.f-wc.18x2.g-fir.2x15-lvl.1-logsig_init.t4-nestspec'
modelname='ozgf.fs100.ch18-ld-sev_dlog.f-wc.18x2.g-fir.2x15-lvl.1-logsig_init.t4-basic'
modelname='ozgf.fs100.ch18-ld-contrast.ms100_dlog.f-wc.18x2.g-fir.2x15-lvl.1-ctwc.18x2.g-ctfir.2x15-ctlvl.1-dsig.d_sev-init.c.t3-basic.t5'

batch = 275
cellid = 'eno012c-c1'
modelname = 'envm100beh_rep2_fir2x2x15_lvl2_mrg_state01-jkm'
modelname = 'env.fs100.m.beh_rep.2-fir.2x15x2-lvl.2-mrg_basic.st.nf5'
modelname = 'env.fs100-ld-st.beh-ref_fir.2x15-rep.2-lvl.2-mrg_jk.nf5-init.st-basic'
modelname = "env.fs100-ld-st.beh-ref_dlog.f-wc.2x1.c-fir.1x15-lvl.1-dexp.1_jk.nf10-init.st-basic"

batch=294
modelname = "psth.fs20.pup-ld-st.pup_stategain.S_jk.nf10-psthfr.j-basic"
modelname = "psth.fs20.pup-ld-st.pup_sw.2x2_jk.nf10-psthfr.j.hilo-basic"
cellid='eno048e-d1'
cellid="zee019e-b1"
cellid="eno050d-c1"
cellid="BOL005c-04-1"
modelname="psth.fs4.pup-ld-st.pup_stategain.S_jk.nf20-psthfr.j-basic"


cellid='bbl086b'
batch=289
chancount=3
bincount=10
modelname="ozgf.fs50.ch18.pup-ld-st.pup-pca.psth.no_dlog-wc.18x{0}.g-fir.1x{1}x{0}-stategain.2x{0}-wc.{0}xR-lvl.R_jk.nf5.o-popiter.T3,4,5,6.fi5".format(chancount, bincount)
modelname="ozgf.fs50.ch18.pup-ld-st.pup-pca.psth.no_dlog-wc.18x{0}.g-fir.1x{1}x{0}.fl-relu.{0}-stategain.2x{0}-wc.{0}xR-lvl.R_jk.nf5.o-popiter.T3,4,5,6.fi5".format(chancount, bincount)


# PTD PC population model
cellid='TAR010c'
batch=307
cellid="TAR010c-21-1_P0"
modelname="parm.fs40.pup-ld-st.pup.beh-ref-pca.psth-psthfr_wc.15x2.g-fir.2x8.fl-lvl.1-dexp.1-stategain.S_jk.nf5-init.st.psth-basic"

# NAT population model (no pupil)
cellid='TAR010c'
batch = 289
modelname="ozgf.fs50.ch18-ld-pca.psth.no-sev_dlog-wc.18x3.g-fir.1x10x3.fl-wc.3xR-lvl.R_popiter.T3,4,5.fi5"
modelname="ozgf.fs50.ch18-ld-pca.psth.no-sev_dlog-wc.18x3.g-fir.1x10x3.fl-relu.3-wc.3xR-lvl.R_popiter.pcf.T3,4,5.fi5"

# PTD old A1 data
batch=311
cellid="ele050b-a1"
cellid="min024b-c1"
cellid='ele026b-b1'
modelname="psth.fs20-ld-st.fil-ref-psthfr_stategain.S_jk.nf20-basic"

#batch, cellid = 269, 'btn144a-a1'
batch, cellid = 269, 'chn019a-a1'
#batch, cellid = 269, 'sti032b-b3'
#batch, cellid = 269, 'oys042c-d1'
#batch, cellid = 273, 'chn041d-b1'
#batch, cellid = 273, 'zee027b-c1'

#keywordstring = 'dlog-wc.18x1.g-fir.1x15-lvl.1'
#keywordstring = 'rdtwc.18x1.g-rdtfir.1x15-rdtgain.global.NTARGETS-lvl.1'
#keywordstring = 'rdtwc.18x1.g-rdtfir.1x15-rdtgain.relative.NTARGETS-lvl.1'
keywordstring =   'rdtgain.gen.NTARGETS-rdtmerge.stim-wc.18x1.g-fir.1x15-lvl.1'
#modelname = 'rdtld-rdtshf-rdtsev-rdtfmt_' + keywordstring + '_init.t5-basic'
modelname = 'rdtld-rdtshf-rdtfmt_' + keywordstring + '_jk.nf2-init.t5-basic'


#
# PTD PSTH + PUPIL + BEHAVIOR
batch=309
cellid="ley026g-a1"
batch=307
cellid="TAR010c-06-1"
cellid="BRT033b-02-1"
modelname = "psth.fs20.pup-ld-st.pup.fil-ref.a-psthfr_stategain.S_jk.nf10-basic"
modelname = "psth.fs20.pup-ld-st.pup0.ttp0-ref.a-psthfr.s_stategain.S_jk.nf10-basic"
modelname = "psth.fs20.pup-ld-st.pup.pas-ref-pas-psthfr.s_stategain.S_jk.nf10-basic"
modelname = "psth.fs20.pup-ld-st.pup.beh-psthfr.stimtar_stategain.S_jk.nf10.bt-basic"
modelname = "psth.fs20.pup-ld-st.pup.beh-ref-psthfr_stategain.S_jk.nf10-basic"

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
    xfspec.append(['nems0.analysis.api.standard_correlation', {},
                   ['est', 'val', 'modelspecs', 'rec'], ['modelspecs']])
    if autoPlot:
        log.info('Generating summary plot ...')
        xfspec.append(['nems0.xforms.plot_summary', {}])
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
    xfspec.append(['nems0.xforms.init_context', xforms_init_context])

    # 1) Load the data
    xfspec.extend(xhelp._parse_kw_string(load_keywords, xforms_lib))

    # 2) generate a modelspec
    xfspec.append(['nems0.xforms.init_from_keywords', {'registry': keyword_lib}])
    #xfspec.append(['nems0.xforms.init_from_keywords', {}])

    # 3) fit the data
    xfspec.extend(xhelp._parse_kw_string(fit_keywords, xforms_lib))

    # 4) add some performance statistics
    xfspec.append(['nems0.xforms.predict', {}])
    xfspec.append(['nems0.xforms.add_summary_statistics', {}])

    # 5) generate plots
    if autoPlot:
        xfspec.append(['nems0.xforms.plot_summary', {}])

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

#results_dir = nems0.get_setting('NEMS_RESULTS_DIR')
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
import nems0.plots.api as nplt

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