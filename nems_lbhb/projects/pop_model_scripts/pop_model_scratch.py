from pathlib import Path
import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as st

from nems_lbhb.projects.pop_model_scripts.pop_model_utils import (
    MODELGROUPS, POP_MODELGROUPS, HELDOUT, MATCHED, EQUIVALENCE_MODELS_SINGLE, EQUIVALENCE_MODELS_POP,
    POP_MODELS, ALL_FAMILY_POP,
    SIG_TEST_MODELS,
    get_significant_cells, snr_by_batch, NAT4_A1_SITES, NAT4_PEG_SITES, PLOT_STAT, DOT_COLORS, DOT_MARKERS, base_path,
    linux_user, ALL_FAMILY_MODELS, VERSION, count_fits, int_path, a1, peg
)
from nems0 import db
from nems0.xform_helper import fit_model_xform, load_model_xform

batch=322
siteids, cellids = db.get_batch_sites(batch)

load_string_pop = "ozgf.fs100.ch18.pop-loadpop-norm.l1-popev"  # 0.42552472088615917
load_string_pop = "gtgram.fs100.ch18.pop-loadpop-norm.l1-popev" #  0.424147
use_nems0=False
if use_nems0:
    # ozgf :  mean= 0.4203
    fit_string_stage1 = f"tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4"
    modelname=f"{load_string_pop}_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100.f-wc.100xR-lvl.R-dexp.R_{fit_string_stage1}"
else:
    fit_string_stage1 = f"lite.tf.init.lr1e3.t3-lite.tf.lr1e4"
    modelname=f"{load_string_pop}_wc.18x1x70.g-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100.f-wc.100xR-dexp.R_{fit_string_stage1}"

modelnames = [
    'gtgram.fs100.ch18.pop-loadpop-norm.l1-popev_wc.18x1x70.g-fir.15x1x70-relu.70.o.s-wc.70x1x80-fir.10x1x80-relu.80.o.s-wc.80x100-relu.100.o.s-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb10-lite.tf.lr1e4',
    'gtgram.fs100.ch18.pop-loadpop-norm.l1-popev_wc.18x1x70.g-fir.15x1x70-relu.70-wc.70x1x80-fir.10x1x80-relu.80-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb10-lite.tf.lr1e4',
    'gtgram.fs100.ch18.pop-loadpop-norm.l1-popev_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100.f-wc.100xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4',
    'ozgf.fs100.ch18.pop-loadpop-norm.l1-popev_wc.18x1x70.g-fir.15x1x70-relu.70.o.s-wc.70x1x80-fir.10x1x80-relu.80.o.s-wc.80x100-relu.100.o.s-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb10-lite.tf.lr1e4',
    'ozgf.fs100.ch18.pop-loadpop-norm.l1-popev_wc.18x1x70.g-fir.15x1x70-relu.70-wc.70x1x80-fir.10x1x80-relu.80-wc.80x100-relu.100-wc.100xR-dexp.R_lite.tf.init.lr1e3.t3.es20.rb10-lite.tf.lr1e4',
    'ozgf.fs100.ch18.pop-loadpop-norm.l1-popev_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100.f-wc.100xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4',
]

cellid = 'NAT4v2'
batch = 322

xf = [None] * len(modelnames)
ctx = [None] * len(modelnames)

for i,m in enumerate(modelnames):
    xf[i],ctx[i] = fit_model_xform(cellid, batch, m, returnModel=True)

r_test = np.stack([c['modelspec'].meta['r_test'] for c in ctx],axis=1)


