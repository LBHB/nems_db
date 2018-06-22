# -*- coding: utf-8 -*-
# wrapper code for fitting models

import os
import random
import re
import numpy as np
import matplotlib.pyplot as plt
import sys

import nems
import nems.initializers
import nems.epoch as ep
import nems.priors
import nems.preprocessing as preproc
import nems.modelspec as ms
import nems.plots.api as nplt
import nems.metrics.api
import nems.analysis.api
import nems.utils
import nems_db.baphy as nb
import nems_db.db as nd
from nems.recording import Recording
from nems.fitters.api import dummy_fitter, coordinate_descent, scipy_minimize
import nems.xforms as xforms
import nems.xform_helper as xhelp

import logging
log = logging.getLogger(__name__)


def get_recording_file(cellid, batch, options={}):

    # options["batch"] = batch
    # options["cellid"] = cellid
    uri = nb.baphy_data_path(cellid, batch, **options)

    return uri


def get_recording_uri(cellid, batch, options={}):

    opts = []
    for i, k in enumerate(options):
        if type(options[k]) is bool:
            opts.append(k+'='+str(int(options[k])))
        elif type(options[k]) is list:
            pass
        else:
            opts.append(k+'='+str(options[k]))
    optstring = "&".join(opts)

    url = "http://hyrax.ohsu.edu:3000/baphy/{0}/{1}?{2}".format(
                batch, cellid, optstring)
    return url


def generate_recording_uri(cellid, batch, loader):
    """
    figure out filename (or eventually URI) of pre-generated
    NEMS-format recording for a given cell/batch/loader string

    very baphy-specific. Needs to be coordinated with loader processing
    in nems.xform_helper
    """

    # TODO: A lot of the parsing is copy-pasted from nems_lbhb/loaders/,
    #       need to figure out which role goes where and delete the
    #       repeated code.

    options = {}

    def _parm_helper(fs, pupil):
        options = {'rasterfs': fs, 'stimfmt': 'parm',
                   'chancount': 0, 'stim': False}

        if pupil:
            pup_med = 2.0 if fs == 10 else 0.5
            options.update({'pupil': True, 'pupil_deblink': True,
                            'pupil_median': pup_med})
        else:
            options['pupil'] = False

        return options

    if loader.startswith('ozgf'):
        pattern = re.compile(r'^ozgf\.(\d{1,})ch(\d{1,})(\w*)?$')
        parsed = re.match(pattern, loader)
        # TODO: fs and chans useful for anything for the loader? They don't
        #       seem to be used here, only in the baphy-specific stuff.
        fs = parsed[1]
        chans = parsed[2]
        ops = parsed[3]
        pupil = ('pup' in ops)

        options = {'rasterfs': fs, 'includeprestim': True,
                   'stimfmt': 'ozgf', 'chancount': chans}

        if pupil:
            options.update({'pupil': True, 'stim': True, 'pupil_deblink': True,
                            'pupil_median': 2})

    elif loader.startswith('nostim'):
        pattern = re.compile(r'^nostim\.(\d{1,})(\w*)?$')
        parsed = re.match(pattern, loader)
        fs = parsed[1]
        ops = parsed[2]
        pupil = ('pup' in ops)

        options.update(_parm_helper(fs, pupil))

    elif loader.startswith('psth'):
        pattern = re.compile(r'^psth(s?)\.(\d{1,})(\w*)?$')
        parsed = re.match(pattern, loader)
        state = parsed[1]  # not relevant here?
        fs = parsed[2]
        ops = parsed[3]
        pupil = ('pup' in ops)

        options.update(_parm_helper(fs, pupil))

    elif loader.startswith('evt'):
        pattern = re.compile(r'^evt\.(\d{1,})(\w*)?$')
        parsed = re.match(pattern, loader)
        fs = parsed[1]
        ops = parsed[2]
        pupil = ('pup' in ops)

        options.update(_parm_helper(fs, pupil))

    elif loader.startswith('env'):
        pattern = re.compile(r'^env(m?)\.(\d{1,})$')
        parsed = re.match(pattern, loader)
        mask = parsed[1]  # not relevant here?
        fs = parsed[2]

        options.update({'rasterfs': fs, 'stimfmt': 'envelope', 'chancount': 0})

    else:
        raise ValueError('unknown loader string')

    # recording_uri = get_recording_uri(cellid, batch, options)
    recording_uri = get_recording_file(cellid, batch, options)

    return recording_uri


def fit_model_xforms_baphy(cellid, batch, modelname,
                           autoPlot=True, saveInDB=False):
    """
    Fits a single NEMS model using data from baphy/celldb
    eg, 'ozgf100ch18_wc18x1_lvl1_fir15x1_dexp1_fit01'
    generates modelspec with 'wc18x1_lvl1_fir15x1_dexp1'

    based on this function in nems/scripts/fit_model.py
       def fit_model(recording_uri, modelstring, destination):

     xfspec = [
        ['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
        ['nems.xforms.add_average_sig', {'signal_to_average': 'resp',
                                         'new_signalname': 'resp',
                                         'epoch_regex': '^STIM_'}],
        ['nems.xforms.split_by_occurrence_counts', {'epoch_regex': '^STIM_'}],
        ['nems.xforms.init_from_keywords', {'keywordstring': modelspecname}],
        ['nems.xforms.set_random_phi',  {}],
        ['nems.xforms.fit_basic',       {}],
        # ['nems.xforms.add_summary_statistics',    {}],
        ['nems.xforms.plot_summary',    {}],
        # ['nems.xforms.save_recordings', {'recordings': ['est', 'val']}],
        ['nems.xforms.fill_in_default_metadata',    {}],
    ]
    """

    log.info('Initializing modelspec(s) for cell/batch %s/%d...',
             cellid, int(batch))

    # Segment modelname for meta information
    kws = modelname.split("_")
    loader = kws[0].split('-')[0]
    modelspecname = "_".join(kws[1:-1])
    fitkey = kws[-1]

    meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
            'loader': loader, 'fitkey': fitkey, 'modelspecname': modelspecname,
            'username': 'nems', 'labgroup': 'lbhb', 'public': 1,
            'githash': os.environ.get('CODEHASH', ''),
            'recording': loader}

    recording_uri = generate_recording_uri(cellid, batch, loader)

    # generate xfspec, which defines sequence of events to load data,
    # generate modelspec, fit data, plot results and save
    xfspec = xhelp.generate_xforms_spec(recording_uri, modelname)

    # actually do the fit
    ctx, log_xf = xforms.evaluate(xfspec)

    # save some extra metadata
    modelspecs = ctx['modelspecs']

    destination = '/auto/data/nems_db/results/{0}/{1}/{2}/'.format(
            batch, cellid, ms.get_modelspec_longname(modelspecs[0]))
    modelspecs[0][0]['meta']['modelpath'] = destination
    modelspecs[0][0]['meta']['figurefile'] = destination+'figure.0000.png'

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
        # TODO : db results finalized?
        nd.update_results_table(modelspecs[0])

    return ctx


def load_model_baphy_xform(cellid, batch=271,
        modelname="ozgf100ch18_wcg18x2_fir15x2_lvl1_dexp1_fit01",
        eval_model=True):

    d = nd.get_results_file(batch, [modelname], [cellid])
    filepath = d['modelpath'][0]
    # Removed print statement here since load_analysis already does it.
    # Was causing a lot of log spam when loading many modelspecs.
    # -jacob 4-8-2018
    return xforms.load_analysis(filepath, eval_model=eval_model)


def load_batch_modelpaths(batch, modelnames, cellids=None, eval_model=True):
    d = nd.get_results_file(batch, [modelnames], cellids=cellids)
    return d['modelpath'].tolist()


def quick_inspect(cellid="chn020f-b1", batch=271,
        modelname="ozgf100ch18_wc18x1_fir15x1_lvl1_dexp1_fit01"):

    ctx = load_model_baphy_xform(cellid, batch, modelname, eval=True)

    modelspecs = ctx['modelspecs']
    est = ctx['est']
    val = ctx['val']
    nplt.plot_summary(val, modelspecs)

    return modelspecs, est, val


"""
# SPN example
cellid='btn144a-c1'
batch=259
modelname="env100_fir15x2_dexp1_fit01"

# A1 RDT example
cellid = 'zee021e-c1'
batch=269
modelname = "ozgf100ch18pt_wc18x1_fir15x1_lvl1_dexp1_fit01"
savepath = fit_model_baphy(cellid=cellid, batch=batch, modelname=modelname,
                           autoPlot=False, saveInDB=True)
modelspec,est,val=load_model_baphy(savepath)

# A1 VOC+pupil example
cellid = 'eno053f-a1'
batch=294
modelname = "ozgf100ch18pup_pup_psth_stategain2_fit02"
savepath = fit_model_baphy(cellid=cellid, batch=batch, modelname=modelname,
                           autoPlot=False, saveInDB=True)
modelspec,est,val=load_model_baphy(savepath)


# A1 NAT + pupil example
cellid = 'TAR010c-18-1'
batch=289
modelname = "ozgf100ch18_wcg18x2_fir15x2_lvl1_dexp1_fit01"

savepath = fit_model_baphy(cellid = cellid, batch=batch, modelname = modelname, autoPlot=True, saveInDB=True)
modelspec,est,val=load_model_baphy(savepath)

# IC NAT example
cellid = "bbl031f-a1"
batch=291
modelname = "ozgf100ch18_wc18x1_fir15x1_lvl1_dexp1_fit01"

savepath = fit_model_baphy(cellid = cellid, batch=batch, modelname = modelname, autoPlot=True, saveInDB=True)
modelspec,est,val=load_model_baphy(savepath)
"""


# A1 NAT + pupil example
"""
cellid = 'TAR010c-18-1'
batch=289
modelname = "ozgf100ch18pup_wcg18x1_fir1x15_lvl1_stategain2_fitjk01"
ctx=fit_model_xforms_baphy(cellid = cellid, batch=batch, modelname = modelname, autoPlot=True, saveInDB=True)

"""


# A1 NAT example
"""

cellid = 'zee019b-b1'
batch=271
modelname = "ozgf100ch18_dlog_wcg18x1_fir1x15_lvl1_dexp1_fit01"
ctx=fit_model_xforms_baphy(cellid = cellid, batch=batch, modelname = modelname, autoPlot=True, saveInDB=True)

savepath = fit_model_baphy(cellid = cellid, batch=batch, modelname = modelname, autoPlot=True, saveInDB=True)

modelspec,est,val=load_model_baphy(savepath)

"""

# A1 VOC+pup example
"""
cellid = 'eno052d-a1'
batch=294
modelname = "nostim10pup_stategain2_fitpjk01"
ctx=fit_model_xforms_baphy(cellid = cellid, batch=batch, modelname = modelname, autoPlot=True, saveInDB=True)
"""

"""
cellid = 'TAR010c-06-1'
batch=301
modelname = "nostim10pup_stategain2_fitpjk01"
ctx=fit_model_xforms_baphy(cellid = cellid, batch=batch, modelname = modelname, autoPlot=True, saveInDB=True)
"""
