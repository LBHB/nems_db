#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:30:15 2018

@author: svd
"""

import matplotlib.pyplot as plt
import numpy as np
import os

import nems.recording
import nems.modelspec as ms
import nems.xforms as xforms

import nems_db.baphy as nb
import nems_db.db as nd
import nems_db.xform_wrappers as nw

import logging
log = logging.getLogger(__name__)

#cellid = 'zee021e-c1'
cellid = 'BRT026c-05-1'
batch=289
modelname = "ozgf100ch18_wc18x1_fir1x15_lvl1_dexp1_fit01"

autoPlot=True
saveInDB=False

log.info('Initializing modelspec(s) for cell/batch {0}/{1}...'.format(cellid,batch))

# parse modelname
kws = modelname.split("_")
loader = kws[0]
modelspecname = "_".join(kws[1:-1])
fitter = kws[-1]

# generate xfspec, which defines sequence of events to load data,
# generate modelspec, fit data, plot results and save
xfspec = nw.generate_loader_xfspec(cellid,batch,loader)

xfspec.append(['nems.xforms.init_from_keywords', {'keywordstring': modelspecname}])

# parse the fit spec: Use gradient descent on whole data set(Fast)
if fitter == "fit01":
    # prefit strf
    log.info("Prefitting STRF without other modules...")
    xfspec.append(['nems.xforms.fit_basic_init', {}])
    xfspec.append(['nems.xforms.fit_basic', {}])
elif fitter == "fitjk01":

    log.info("n-fold fitting...")
    xfspec.append(['nems.xforms.split_for_jackknife', {'njacks': 5}])
    xfspec.append(['nems.xforms.fit_nfold', {}])

elif fitter == "fitpjk01":

    log.info("n-fold fitting...")
    xfspec.append(['nems.xforms.split_for_jackknife', {'njacks': 5}])
    xfspec.append(['nems.xforms.generate_psth_from_est_for_both_est_and_val_nfold',  {}])
    xfspec.append(['nems.xforms.fit_nfold', {}])

elif fitter == "fit02":
    # no pre-fit
    log.info("Performing full fit...")
    xfspec.append(['nems.xforms.fit_basic', {}])
else:
    raise ValueError('unknown fitter string')

xfspec.append(['nems.xforms.add_summary_statistics',    {}])

if autoPlot:
    # GENERATE PLOTS
    log.info('Generating summary plot...')
    xfspec.append(['nems.xforms.plot_summary',    {}])

# actually do the fit
ctx, log_xf = xforms.evaluate(xfspec)

# save some extra metadata
modelspecs=ctx['modelspecs']

if 'CODEHASH' in os.environ.keys():
    githash=os.environ['CODEHASH']
else:
    githash=""
meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
        'loader': loader, 'fitter': fitter, 'modelspecname': modelspecname,
        'username': 'svd', 'labgroup': 'lbhb', 'public': 1,
        'githash': githash, 'recording': loader}
if not 'meta' in modelspecs[0][0].keys():
    modelspecs[0][0]['meta'] = meta
else:
    modelspecs[0][0]['meta'].update(meta)
destination = '/auto/data/tmp/modelspecs/{0}/{1}/{2}/'.format(
        batch,cellid,ms.get_modelspec_longname(modelspecs[0]))
modelspecs[0][0]['meta']['modelpath']=destination
modelspecs[0][0]['meta']['figurefile']=destination+'figure.0000.png'

# save results

log.info('Saving modelspec(s) to {0} ...'.format(destination))
xforms.save_analysis(destination,
                     recording=ctx['rec'],
                     modelspecs=modelspecs,
                     xfspec=xfspec,
                     figures=ctx['figures'],
                     log=log_xf)

# save in database as well
if saveInDB:
    # TODO : db results
    nd.update_results_table(modelspecs[0])

# save some extra metadata
modelspecs=ctx['modelspecs']
val=ctx['val']

plt.figure();
plt.plot(val['resp'].as_continuous().T)
plt.plot(val['pred'].as_continuous().T)
plt.plot(val['state'].as_continuous().T/100)



