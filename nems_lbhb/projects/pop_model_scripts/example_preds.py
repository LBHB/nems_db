import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import pandas as pd
import importlib

import matplotlib as mpl
params = {'axes.spines.right': False,
          'axes.spines.top': False,
          'legend.fontsize': 12,
          'axes.labelsize': 12,
          'axes.titlesize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
mpl.rcParams.update(params)

import nems
import nems.db as nd
import nems.xform_helper as xhelp
import nems_lbhb.xform_wrappers as xwrap
import nems.epoch as ep
from nems.xforms import evaluate_step
import nems_lbhb.baphy_io as io
from nems_lbhb import baphy_experiment
from nems.xform_helper import load_model_xform
from nems import xforms
from nems_lbhb.plots import scatter_bin_lin
from nems_lbhb.analysis import pop_models

from nems_lbhb.projects.pop_model_scripts.pop_model_utils import load_string_pop, fit_string_pop, load_string_single, fit_string_single,\
    POP_MODELS, SIG_TEST_MODELS, shortnames, shortnamesp, MODELGROUPS, ALL_FAMILY_MODELS


# load hi-res spectrogram
batch=322
cellid="DRX006b-128-2"


b = baphy_experiment.BAPHYExperiment(batch=batch, cellid=cellid)
tctx = {'rec': b.get_recording(loadkey="ozgf.fs100.ch64")}
tctx = xforms.evaluate_step(['nems.xforms.split_by_occurrence_counts', {'epoch_regex': '^STIM', 'keepfrac': 1.0}], tctx)
tctx = xforms.evaluate_step(['nems.xforms.average_away_stim_occurrences', {'epoch_regex': '^STIM'}], tctx)

example_models=[ALL_FAMILY_MODELS[3], ALL_FAMILY_MODELS[2], ALL_FAMILY_MODELS[0]]
example_shortnames=['ln_pop','conv1dx2+d','conv2dx3']

cellids = ["DRX006b-128-2", "ARM030a-40-2"]   # , "ARM030a-23-2"
figs = []
for cellid in cellids:
    # LN
    xf0,ctx0=load_model_xform(cellid=cellid,batch=batch,modelname=example_models[0])
    xf1,ctx1=load_model_xform(cellid=cellid,batch=batch,modelname=example_models[1],
                              eval_model=False)
    ctx1['val'] = ctx1['modelspec'].evaluate(rec=ctx0['val'].copy())

    xf2,ctx2=load_model_xform(cellid=cellid,batch=batch,modelname=example_models[2],
                              eval_model=False)
    ctx2['val'] = ctx2['modelspec'].evaluate(rec=ctx0['val'].copy())

    ctx0['val']['stim'] = tctx['val']['stim']
    figs.append(pop_models.model_pred_sum(
        [ctx0, ctx1, ctx2], cellid=cellid, rr=np.arange(150,600),
        predcolor=['orange', 'blue', 'darkgreen'], labels=example_shortnames))

