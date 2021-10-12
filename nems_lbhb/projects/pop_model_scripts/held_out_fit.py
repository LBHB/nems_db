import os
import logging
import copy
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import nems
import nems.db as nd
import nems.xform_helper as xhelp
import nems_lbhb.xform_wrappers as xwrap
import nems.xforms
import nems.epoch as ep
from nems.xforms import evaluate_step

from nems import get_setting
from nems.utils import escaped_split, escaped_join
from nems.registry import KeywordRegistry, xforms_lib, keyword_lib
from nems.plugins import (default_keywords, default_loaders, default_fitters,
                          default_initializers)
from nems.xform_helper import generate_xforms_spec

log = logging.getLogger(__name__)

# batch = 322
# all_sites = ['AMT003c','AMT005c', 'bbl099g','bbl104h','BRT026c',   # NAT 3 dataset
#              'BRT033b','BRT034f', 'BRT038b','BRT039c']
#
# # Get list of all cellids from dataset (not batch, b/c not all from batch are included in the fit)
# cells = []
# for site in all_sites:
#     site_cells = nd.get_batch_cells(batch, cellid=site, as_list=True)
#     cells.extend(site_cells)
#
# sites = ['bbl086b']
# LN = 'ozgf.fs100.ch18.pop-ld-cc.0.xx{}-norm.l1-popev_conv2d.4.7x3.rep3-wcn.30-relu.30-wc.30xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4'
# conv2d = 'ozgf.fs100.ch18.pop-ld-cc.0.xx{}-norm.l1-popev_wc.18x30.g-fir.1x20x30-relu.30.f-wc.30x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4'
# conv1dx2 = 'ozgf.fs100.ch18.pop-ld-cc.0.xx{}-norm.l1-popev_wc.18x30.g-fir.1x12x30-relu.30-wc.30x50-fir.1x8x50-relu.50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4'
# all_LN = [LN.format(site) for site in sites]
# all_conv2d = [conv2d.format(site) for site in sites]
# all_conv1dx2 = [conv1dx2.format(site) for site in sites]
# models_per_site = zip(all_LN, all_conv2d, all_conv1dx2)
#
#
# # just pick one for testing
# site = sites[0]
# modelname = all_conv1dx2[0]
#
# # Load model that was fit to all but one site1.2bguv
#
# # pick first cellid from a site that wasn't excluded
# for c in cells:
#     if not c.startswith(site):
#         cellid = c
#         break
# xfspec, ctx = xhelp.load_model_xform(cellid, batch, modelname, eval_model=False)


# Use first N layers of that model to transform input
# TODO: figure out a way to do this programmatically from the modelspec?
# stop_indices = {'LN': 0, 'conv2d': 0, 'conv1dx2': 6} # TODO: load LN and conv2d to check those
# stop_idx = 6  # TODO: temporarily hardcoded for conv1dx2
# modelspec = ctx['modelspec']

# Nope, going about this wrong. I don't want to copy the data,
# I just want to copy the parameter to an identical model that will
# be fit to the one held-out site. That should be much easier (I think).
# also avoids need to evaluate model, which speeds things up a lot.

# Snag though: it would be ideal if I could just have this script say:
# okay, copy all these parameters then queue up jobs on exacloud.
# But the only way I know of to get the paramter intot he TF model would be to copy
# directly to a new modelspec. Not a problem if i'm running locally, but would need
# to set up a pipeline to have exacloud pull that information (or send it with the
# queue request somehow), which does not sound trivial.
# So for now: do it locally.
# TODO: set up something to do this through exacloud. Could maybe add a kwarg to
# the queue function, for example, like: init_from_fit: {modelname, batch, site}

# Other snag: actually, I might need to copy data after all.
# Two options, can just copy parameters and then set those layers to
# be "frozen" somehow through TF, or if I can't figure that out I would
# have to "chop off" that part of the model and manually input the
# transformed stim at that point in the process to start the abbreviated model fit.

# idea: solve both problems, do whole fit at once (there's always a 1-1 of held-out part
#       to second fit anyway). Just need a new "fitter" keyword that would
#       swap out the recordings between fits, and would need to do all preprocessing steps
#       on both subsets plus store the held-out subset in ctx. but finish testing here
#       first to make sure everything is working.


# est_out = modelspec.evaluate(ctx['est'], stop=stop_idx)
# new_est = est_out.copy()
# new_est['stim'] = new_est['pred'].copy()
# new_est.pop('pred')
#val_out = modelspec.evaluate(ctx['val'], stop=stop_idx)
#new_input_est = est_out['pred'].as_continuous()
#new_input_val = val_out['pred'].as_continuous()

# NOTE: fit tolerances were reduced to e1 for faster testing, so don't just copy this back to notebook for full fit
# modelname = ("ozgf.fs100.ch18.pop-ld-norm.l1-popev-hc.AMT003c_"
#              "wc.18x30.g-fir.1x12x30-relu.30-wc.30x50-fir.1x8x50-relu.50-wc.50xR-lvl.R-dexp.R_"
#              "tfinit.n.lr1e1.et1-newtf.n.lr1e1-tfheld.TL6:9-tfinit.n.lr1e1.et1.TL6:9-newtf.n.lr1e1.TL6:9")
# modelname = ("ozgf.fs100.ch18.pop-ld-norm.l1-popev-hc.AMT003c_"
#              "conv2d.4.7x3.rep3-wcn.30-relu.30-wc.30xR-lvl.R-dexp.R_"
#              "tfinit.n.lr1e1.et1-newtf.n.lr1e1-tfheld.FL0:5-tfinit.n.lr1e1.et1.FL0:5-newtf.n.lr1e1.FL0:5")
# modelname = ("ozgf.fs100.ch18.pop-ld-norm.l1-popev-hc.0.msAMT003c_"
#              "conv2d.4.7x3.rep3-wcn.30-relu.30-wc.30xR-lvl.R-dexp.R_"
#              "tfinit.n.lr1e1.et1-newtf.n.lr1e1-tfheld.FL0:5.ms-tfinit.n.lr1e1.et1.FL0:5-newtf.n.lr1e1.FL0:5")
modelnames = ['ozgf.fs100.ch18.pop-ld-norm.l1-popev_wc.18x5.g-fir.1x15x5-relu.5.f-wc.5x10-fir.1x10x10-relu.10.f-wc.10x20-relu.20-wc.20xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
 'ozgf.fs100.ch18.pop-ld-norm.l1-popev_wc.18x10.g-fir.1x15x10-relu.10.f-wc.10x10-fir.1x10x10-relu.10.f-wc.10x20-relu.20-wc.20xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
 'ozgf.fs100.ch18.pop-ld-norm.l1-popev_wc.18x10.g-fir.1x15x10-relu.10.f-wc.10x20-fir.1x10x20-relu.20.f-wc.20x30-relu.30-wc.30xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
 'ozgf.fs100.ch18.pop-ld-norm.l1-popev_wc.18x20.g-fir.1x15x20-relu.20.f-wc.20x20-fir.1x10x20-relu.20.f-wc.20x40-relu.40-wc.40xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
 'ozgf.fs100.ch18.pop-ld-norm.l1-popev_wc.18x20.g-fir.1x15x20-relu.20.f-wc.20x40-fir.1x10x40-relu.40.f-wc.40x60-relu.60-wc.60xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
 'ozgf.fs100.ch18.pop-ld-norm.l1-popev_wc.18x30.g-fir.1x15x30-relu.30.f-wc.30x60-fir.1x10x60-relu.60.f-wc.60x80-relu.80-wc.80xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
 'ozgf.fs100.ch18.pop-ld-norm.l1-popev_wc.18x50.g-fir.1x15x50-relu.50.f-wc.50x70-fir.1x10x70-relu.70.f-wc.70x90-relu.90-wc.90xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
 'ozgf.fs100.ch18.pop-ld-norm.l1-popev_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4',
 'ozgf.fs100.ch18.pop-ld-norm.l1-popev_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x90-fir.1x10x90-relu.90.f-wc.90x120-relu.120-wc.120xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3-newtf.n.lr1e4'
]

batch = 322
cellid = 'NAT4'

# heldout
# modelname = ("ozgf.fs100.ch18.pop-ld-norm.l1-popev-hc.DRX007a.e65:128_"
#              + "conv2d.4.8x3-wcn.10-relu.10-wc.10xR-lvl.R-dexp.R_"
#              + "tfinit.n.lr1e1.et1.rb2.es20-newtf.n.lr1e1.es20-tfheld.FL0:3-tfinit.n.lr1e1.et1.es20-newtf.n.lr1e1")

# # matched site
modelname = ("ozgf.fs100.ch18.pop-ld-norm.l1-popev-hc.ms.DRX007a.e65:128_"
             + "conv2d.4.8x3-wcn.10-relu.10-wc.10xR-lvl.R-dexp.R_"
             + "tfinit.n.lr1e1.et1.rb10.es20-newtf.n.lr1e1.es20-tfheld.FL0:3-tfinit.n.lr1e1.et1.es20-newtf.n.lr1e1")

# heldout with cell count
# modelname = ("ozgf.fs100.ch18.pop-ld-norm.l1-popev-hc.ARM029a-mc.20_"
#              + "conv2d.4.8x3-wcn.10-relu.10-wc.10xR-lvl.R-dexp.R_"
#              + "tfinit.n.lr1e1.v.et1.rb2.es20.L2-newtf.n.lr1e1.es20-tfheld.FL0:3-tfinit.n.lr1e1.et1.es20-newtf.n.lr1e1")


batch = 323
modelname = 'ozgf.fs100.ch18.pop-loadpop-norm.l1-popev_conv2d.10.8x3.rep3-wcn.30-relu.30-wc.30xR-lvl.R-dexp.R_tfinit.n.lr1e1.et3.rb2.es20-newtf.n.lr1e1'
recording_uri = '/auto/data/nems_db/recordings/%s/NAT4_ozgf.fs100.ch18.tgz' % batch


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

registry_args = {}
xforms_init_context = {'cellid': cellid, 'batch': int(batch)}

log.info("TODO: simplify generate_xforms_spec parameters")
xfspec2 = generate_xforms_spec(recording_uri=recording_uri, modelname=modelname,
                              meta=meta, xforms_kwargs=registry_args,
                              xforms_init_context=xforms_init_context,
                              autoPlot=False)

# perform "normal" fit, with siteid held out
# context2 = {}
# for xfa in xfspec2[:-4]:
#     context2 = evaluate_step(xfa, context2)
#
# context3 = copy.deepcopy(context2)
# for xfa in xfspec2[-4:]:
#     context3 = evaluate_step(xfa, context3)

context2 = {}
for xfa in xfspec2[:-5]:
    context2 = evaluate_step(xfa, context2)

context3 = copy.deepcopy(context2)
context2.update(nems.xforms.predict(**context2))
context2.update(nems.xforms.add_summary_statistics(**context2))
rt2 = context2['modelspec'].meta['r_test']

for xfa in xfspec2[-5:]:
    context3 = evaluate_step(xfa, context3)
rt3 = context3['modelspec'].meta['r_test']

std_model = 'ozgf.fs100.ch18.pop-ld-norm.l1-popev_conv2d.4.8x3.rep3-wcn.40-relu.40-wc.40xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.rb5.es20-newtf.n.lr1e4.es20'
rt4 = nd.batch_comp(batch, [std_model], stat='r_test')
rt5 = rt4.reindex(context2['rec']['resp'].chans)

# fig = plt.figure()
# plt.scatter(rt2, rt5, c='black', s=2)
# plt.plot([0, 1], [0, 1], c='black', linestyle='dashed', linewidth=1)
# plt.xlabel('matched first fit')
# plt.ylabel('standard model')

#nd.update_results_table(context3['modelspec'])


#
# modelspec = context3['modelspec']
#
# cellids = modelspec.meta.get('cellids', [])
#
# if (type(cellids) is list) and len(cellids) > 1:
#
#     cell_name = cellids[0].split("-")[0]
#
# elif type(cellid) is list:
#     cell_name = cellid[0].split("-")[0]
# else:
#     cell_name = cellid
#
# if 'modelpath' not in modelspec.meta:
#     prefix = get_setting('NEMS_RESULTS_DIR')
#     destination = os.path.join(prefix, str(batch), cell_name, modelspec.get_longname())
#
#     log.info(f'Setting modelpath to "{destination}"')
#     modelspec.meta['modelpath'] = destination
#     modelspec.meta['figurefile'] = os.path.join(destination, 'figure.0000.png')
# else:
#     destination = modelspec.meta['modelpath']
#
# # figure out URI for location to save results (either file or http, depending on USE_NEMS_BAPHY_API)
# if get_setting('USE_NEMS_BAPHY_API'):
#     prefix = 'http://' + get_setting('NEMS_BAPHY_API_HOST') + ":" + str(get_setting('NEMS_BAPHY_API_PORT')) + \
#              '/results'
#     save_loc = str(batch) + '/' + cell_name + '/' + modelspec.get_longname()
#     save_destination = prefix + '/' + save_loc
#     # set the modelspec meta save locations to be the filesystem and not baphy
#     modelspec.meta['modelpath'] = get_setting('NEMS_RESULTS_DIR') + '/' + save_loc
#     modelspec.meta['figurefile'] = modelspec.meta['modelpath'] + '/' + 'figure.0000.png'
# else:
#     save_destination = destination
#
# nd.update_results_table(modelspec)
