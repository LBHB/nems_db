import logging

import pandas as pd
import numpy as np
import getpass
from pathlib import Path
import datetime
import os
import matplotlib as mpl
mplparams = {
    'axes.spines.right': False,
    'axes.spines.top': False,
    'legend.frameon': False,
    'legend.fontsize': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'hatch.linewidth': 0.5
}
mpl.rcParams.update(mplparams)

import nems0
import nems0.epoch as ep
import nems0.db as nd
import nems_lbhb.xform_wrappers as xwrap

log = logging.getLogger(__name__)

# set to >1 to use l2-reg models
VERSION = 1

batch = 343

linux_user = getpass.getuser()
if linux_user=='svd':
    #figures_base_path = Path('/auto/users/svd/projects/pop_models/eps/')
    figures_base_path = Path('/auto/users/svd/docs/current/pop_coding/figures/')
    int_path = Path('/auto/users/svd/python/nems_db/nems_lbhb/projects/pop_model_scripts/intermediate_results/')
    base_path = figures_base_path
elif linux_user == 'luke':
    figures_base_path = Path('/auto/users/luke/Projects/SPS/plots/NEMS/pop_plots/')
    base_path = figures_base_path
else:
    figures_base_path = Path('/auto/users/jacob/notes/pop_model_figs/')
    int_path = Path('/auto/users/jacob/notes/new_equivalence_results/')
    date = str(datetime.datetime.now()).split(' ')[0]
    base_path = figures_base_path / date

if not base_path.is_dir():
    base_path.mkdir(parents=True, exist_ok=True)

# TODO: adjust figure sizes
if linux_user=='svd':
    sf=1
    single_column_shorter = (3.5, 2*sf)
    single_column_short = (3.5*sf, 2.5*sf)
    single_column_tall = (3.5*sf, 6*sf)
    column_and_half_vshort = (5*sf, 1.5*sf)
    column_and_half_short = (5*sf, 2.5*sf)
    column_and_half_tall = (5*sf, 5*sf)
    double_column_short = (7*sf, 3*sf)
    double_column_shorter = (7*sf, 2*sf)
    double_column_medium = (7*sf, 5*sf)
else:
    single_column_shorter = (3.5, 2)
    single_column_short = (3.5, 3)
    single_column_tall = (3.5, 6)
    column_and_half_vshort = (5, 1.5)
    column_and_half_short = (5, 3)
    column_and_half_tall = (5, 6)
    double_column_short = (7, 3)
    double_column_shorter = (7, 2)
    double_column_medium = (7, 5)
#inset = (1, 1)  # easier to just resize manually, making it this smaller makes things behave weirdly


####
# set version-specific fit strings
####

# load/fit prefix/suffix
load_string_pop = "gtgram.fs100.ch18-ld-norm.l1-sev"
load_string_single = "gtgram.fs100.ch18-ld-norm.l1-sev"

fit_string_pop = "lite.tf.mi1000.lr1e3.t6.es20"
fit_string_single = "lite.tf.mi1000.lr1e3.t6.es20"

# maybe better model? only slightly. fewer L1 units, more L2
cnn1dx2_alt = "wc.18x1x60.g-fir.15x1x60-relu.60.f-wc.60x1x90-fir.10x1x90-relu.90.f-wc.90x100-relu.100.s-wc.100xR-dexp.R"
cnn1dx2_alt2 = "wc.18x1x60.g-fir.15x1x60-relu.60.f-wc.60x1x90-fir.10x1x90-relu.90.f-wc.90x80-relu.80.s-wc.80xR-dexp.R"
cnn1dx2_alt3 = "wc.18x1x70.g-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x80-relu.80.s-wc.80xR-dexp.R"
cnn1dx2 = "wc.18x1x70.g-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100.s-wc.100xR-dexp.R"
cnn1dx2_2 = "wc.18x1x70.g-fir.15x1x70-relu.70.f-wc.70x1x80-fir.10x1x80-relu.80.f-wc.80x100-relu.100-wc.100xR-dexp.R"
cnn1d = "wc.18x1x80.g-fir.25x1x80-relu.80.f-wc.80x100-relu.100.s-wc.100xR-dexp.R"
lnpop = "wc.18x1x120.g-fir.25x1x120-wc.120xR-dexp.R"
cnnsingle="wc.18x1x6.g-fir.25x1x6-relu.6.f-wc.6x1-dexp.1"

# POP_MODELS: round 1, fit using cellid="NAT4" on exacloud
POP_MODELS = [
    f"{load_string_pop}_{cnn1dx2}_{fit_string_pop}", # c1dx2
    f"{load_string_pop}_{lnpop}_{fit_string_pop}", # LN_pop
    f"{load_string_pop}_{cnn1d}_{fit_string_pop}", # c1d
    f"{load_string_pop}_{cnn1dx2_2}_{fit_string_pop}", # c1dx2
]
# SIG_TEST_MODELS: round 2, fit using real single cellid in LBHB or exacloud. LBHB probably faster
SIG_TEST_MODELS = [
    f"{load_string_single}_{cnn1dx2}_{fit_string_single}", # c1dx2+d
    f"{load_string_single}_{lnpop}_{fit_string_single}", # LN_pop
    f"{load_string_single}_{cnnsingle}_{fit_string_single}"  # dnn1_single
]
shortnames = ['1D-CNN', 'POP-LN', 'single-CNN']


def list_pop_models_old():
    # load/fit prefix/suffix
    vsuffix = ''
    vsuffixp = ''
    vsuffixdnn = ''
    vsuffixc2d = vsuffix
    vsuffixpc2d = vsuffixp

    
    # DIFF FROM PLOSCB ANALYSIS : single-site loading, gtgram instead of ozgf, rb5 instead of rb10
    #load_string_pop = "ozgf.fs100.ch18.pop-loadpop-norm.l1-popev"
    load_string_pop = "gtgram.fs100.ch18-ld-norm.l1-sev"
    load_string_single = "gtgram.fs100.ch18-ld-norm.l1-sev"

    fit_string_pop =   f"tfinit.n.lr1e3.et3.rb5.es20-newtf.n.lr1e4"
    #fit_string_pop =   f"tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4{vsuffixp}"
    fit_string_nopre = f'tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4{vsuffixdnn}'
    fit_string_dnn =   f'prefit.m-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4{vsuffixdnn}'
    fit_string_single = f'prefit.f-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4{vsuffix}'

    fit_string_pop_c2d =   f"tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4{vsuffixpc2d}"
    fit_string_single_c2d = f'prefit.f-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4{vsuffixc2d}'

    # POP_MODELS: round 1, fit using cellid="NAT4" on exacloud
    POP_MODELS_OLD = [
        #f"{load_string_pop}_conv2d.10.8x3.rep3-wcn.90-relu.90-wc.90xR-lvl.R-dexp.R_{fit_string_pop_c2d}",  #c2d
        f"{load_string_pop}_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100.f-wc.100xR-lvl.R-dexp.R_{fit_string_pop}", # c1dx2+d
        f"{load_string_pop}_wc.18x120.g-fir.1x25x120-wc.120xR-lvl.R-dexp.R_{fit_string_pop}", # LN_pop
        f"{load_string_pop}_wc.18x100.g-fir.1x25x100-relu.100.f-wc.100x120-relu.120.f-wc.120xR-lvl.R-dexp.R_{fit_string_pop}", # c1d
        #f"{load_string_pop}_wc.18x4.g-fir.1x25x4-wc.4xR-lvl.R-dexp.R_{fit_string_pop}", # Low_dim
        #f"{load_string_single}_wc.18x6.g-fir.1x25x6-relu.6.f-wc.6x1-lvl.1-dexp.1_{fit_string_dnn}"  # dnn1_single
    ]
    return POP_MODELS_OLD

POP_MODELS_OLD = list_pop_models_old()