from os.path import basename, join
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d, butter, sosfilt
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_array
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import os
import datetime
#import tensorflow as tf
import itertools

from nems0 import db, epoch, initializers, xforms
import nems0.preprocessing as preproc
from nems0.utils import smooth, parse_kw_string
from nems_lbhb import xform_wrappers, baphy_io, baphy_experiment

from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.baphy_io import load_continuous_openephys, get_spike_info, get_depth_info
from nems_lbhb.plots import plot_waveforms_64D
from nems_lbhb.preprocessing import impute_multi
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models

from nems.layers import WeightChannels, FIR, LevelShift, \
    DoubleExponential, RectifiedLinear, ConcatSignals
from nems import Model
from nems.layers.base import Layer, Phi, Parameter
import nems.visualization.model as nplt
#import nems0.plots.api as nplt
from nems_lbhb.projects.freemoving.free_model import load_free_data, free_fit
from nems0.modules.nonlinearity import _dlog
from nems0.xform_helper import _xform_exists, load_model_xform, fit_model_xform
from nems_lbhb.projects.bignat.bnt_defaults import POP_MODELS, SIG_TEST_MODELS, shortnames, POP_MODELS_OLD

from nems0.registry import KeywordRegistry, xforms_lib, keyword_lib

from nems_lbhb.projects.bignat.bnt_tools import to_sparse4d, to_dense4d, data_subset, \
    get_submodel, save_submodel, btn_generator, load_bnt_recording, pc_fit
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems0.recording import load_recording
from nems0.epoch import epoch_names_matching
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb import baphy_io

from nems_lbhb.projects.bignat.bnt_defaults import POP_MODELS


batch=343

siteids, cellids = db.get_batch_sites(batch)

siteid = "PRN021a"
loadkey = "gtgram.fs100.ch18"

# get area info. This will be a superset of what the batch loads
d = baphy_io.get_depth_info(siteid=siteid)

uri = generate_recording_uri(cellid=siteid, batch=batch, loadkey=loadkey, recache=False)
rec = load_recording(uri)

epochs =rec['resp'].epochs

stim_epochs = epoch_names_matching(rec['resp'].epochs, "^STIM_")


#modelname="gtgram.fs100.ch18-ld-norm.l1-sev.fOLP_wc.18x1x70.g-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R_prefit.b322.f.nf-lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4"

#xfspec, ctx=fit_model_xform(cellid, batch, modelname, returnModel=True)

"""
manager = BAPHYExperiment(batch=341, cellid="ARM024a")
fs = 100
loadkey="gtgram.fs100.ch18"
rec = manager.get_recording(loadkey=loadkey,recache=True)

e = ep.epoch_names_matching(rec['resp'].epochs,"^STIM_")
e1=e[0]

e2='STIM_08Waterfall-0-1_04Gobble-0-1'
e3 ='STIM_cat199_rec1_horse_neighing_excerpt1.wav'

s1= rec['stim']._data[e1]
s2= rec['stim']._data[e2]
s3 =  rec['stim']._data[e3]

plt.figure()
plt.imshow(np.concatenate((s1,s2,s3), axis=1), aspect='auto', interpolation='none')
"""


"""
sitecount = 2
keywordstub = "wc.19x1x3-fir.10x1x3-wc.3"
modelspec, datasets = do_bnt_fit(sitecount, keywordstub, save_results=False)





exclude_sites =['CLT027c', 'CLT050c']
batch=343

siteids,cellids = db.get_batch_sites(343)
siteids = [s for s in siteids if s not in exclude_sites]

siteids = siteids[:sitecount]

print(siteids)

site_i = []
datasets = {}
for i, siteid in enumerate(siteids):
    datasets[siteid] = load_bnt_recording(
        siteid, batch=batch, loadkey="gtgram.fs100.ch18",
        pc_count=10, val_single=False)

cellids = [d['cellids'] for k,d in datasets.items()]
cellids = list(itertools.chain(*cellids))
cell_siteids = [c.split("-")[0] for c in cellids]

total_cells = len(cell_siteids)
total_pcs = int(len(siteids)*10)

model, ccpca = pc_fit(keywordstub, datasets)

"""