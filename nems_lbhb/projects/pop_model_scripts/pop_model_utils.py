import logging

import pandas as pd
import numpy as np
import getpass
from pathlib import Path
import datetime
import os
import matplotlib as mpl
params = {'axes.spines.right': False,
          'axes.spines.top': False,
          'legend.fontsize': 10,
          'axes.labelsize': 10,
          'axes.titlesize': 10,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
mpl.rcParams.update(params)

import nems
import nems.epoch as ep
import nems.db as nd
import nems_lbhb.xform_wrappers as xwrap

log = logging.getLogger(__name__)

# set to >1 to use l2-reg models
VERSION = 1

a1=322
peg=323

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
    single_column_short = (3.5*sf, 2.5*sf)
    single_column_tall = (3.5*sf, 6*sf)
    column_and_half_vshort = (5*sf, 1.5*sf)
    column_and_half_short = (5*sf, 2.5*sf)
    column_and_half_tall = (5*sf, 5*sf)

else:
    single_column_short = (3.5, 3)
    single_column_tall = (3.5, 6)
    column_and_half_vshort = (5, 1.5)
    column_and_half_short = (5, 3)
    column_and_half_tall = (5, 6)
#inset = (1, 1)  # easier to just resize manually, making it this smaller makes things behave weirdly


####
# set version-specific fit strings
####
if VERSION > 1:
    if VERSION == 2:
        # no longer used????
        vsuffix = '.ver2'
        vsuffixp = '.ver2'
        vsuffixdnn = ''
        vsuffixc2d = vsuffix
        vsuffixpc2d = vsuffixp
    elif VERSION == 3:
        vsuffix = '.l2:5-dstrf'
        vsuffixp = '.l2:5'
        vsuffixdnn = vsuffix
        vsuffixc2d = '.l2:4-dstrf'
        vsuffixpc2d = '.l2:4'
        #vsuffixc2d = vsuffix
        #vsuffixpc2d = vsuffixp

else:
    vsuffix = ''
    vsuffixp = ''
    vsuffixdnn = ''
    vsuffixc2d = vsuffix
    vsuffixpc2d = vsuffixp

# load/fit prefix/suffix
load_string_pop = "ozgf.fs100.ch18.pop-loadpop-norm.l1-popev"
load_string_single = "ozgf.fs100.ch18-ld-norm.l1-sev"

fit_string_pop =   f"tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4{vsuffixp}"
fit_string_nopre = f'tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4{vsuffixdnn}'
fit_string_dnn =   f'prefit.m-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4{vsuffixdnn}'
fit_string_single = f'prefit.f-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4{vsuffix}'

fit_string_pop_c2d =   f"tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4{vsuffixpc2d}"
fit_string_single_c2d = f'prefit.f-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4{vsuffixc2d}'

# POP_MODELS: round 1, fit using cellid="NAT4" on exacloud
POP_MODELS = [
    #f"{load_string_pop}_conv2d.10.8x3.rep3-wcn.90-relu.90-wc.90xR-lvl.R-dexp.R_{fit_string_pop_c2d}",  #c2d
    f"{load_string_pop}_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_{fit_string_pop}", # c1dx2+d
    f"{load_string_pop}_wc.18x120.g-fir.1x25x120-wc.120xR-lvl.R-dexp.R_{fit_string_pop}", # LN_pop
    #f"{load_string_pop}_wc.18x100.g-fir.1x25x100-relu.100.f-wc.100x120-relu.120.f-wc.120xR-lvl.R-dexp.R_{fit_string_pop}", # c1d
    #f"{load_string_pop}_wc.18x4.g-fir.1x25x4-wc.4xR-lvl.R-dexp.R_{fit_string_pop}", # Low_dim
    f"{load_string_single}_wc.18x6.g-fir.1x25x6-relu.6.f-wc.6x1-lvl.1-dexp.1_{fit_string_dnn}"  # dnn1_single
]
# SIG_TEST_MODELS: round 2, fit using real single cellid in LBHB or exacloud. LBHB probably faster
SIG_TEST_MODELS = [
    #f"{load_string_single}_conv2d.10.8x3.rep3-wcn.90-relu.90-wc.90xR-lvl.R-dexp.R_{fit_string_single_c2d}",  #c2d
    f"{load_string_single}_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_{fit_string_single}", # c1dx2+d
    f"{load_string_single}_wc.18x120.g-fir.1x25x120-wc.120xR-lvl.R-dexp.R_{fit_string_single}", # LN_pop
    #f"{load_string_single}_wc.18x100.g-fir.1x25x100-relu.100.f-wc.100x120-relu.120.f-wc.120xR-lvl.R-dexp.R_{fit_string_single}", # c1d
    #f"{load_string_single}_wc.18x4.g-fir.1x25x4-wc.4xR-lvl.R-dexp.R_{fit_string_single}", # Low_dim
    f"{load_string_single}_wc.18x6.g-fir.1x25x6-relu.6.f-wc.6x1-lvl.1-dexp.1_{fit_string_dnn}"  # dnn1_single
]
shortnames=['conv1d','conv1dx2','dnn-sing']
shortnamesp=[s+"_p" for s in shortnames]

# POP_MODELS: round 1, fit using cellid="NAT4" on exacloud
ALL_FAMILY_POP = [
    f"{load_string_pop}_conv2d.10.8x3.rep3-wcn.90-relu.90-wc.90xR-lvl.R-dexp.R_{fit_string_pop_c2d}",  #c2d
    f"{load_string_pop}_wc.18x100.g-fir.1x25x100-relu.100.f-wc.100x120-relu.120.f-wc.120xR-lvl.R-dexp.R_{fit_string_pop}", # c1d
    f"{load_string_pop}_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_{fit_string_pop}", # c1dx2+d
    f"{load_string_pop}_wc.18x120.g-fir.1x25x120-wc.120xR-lvl.R-dexp.R_{fit_string_pop}", # LN_pop
    f"{load_string_single}_wc.18x6.g-fir.1x25x6-relu.6.f-wc.6x1-lvl.1-dexp.1_{fit_string_dnn}"  # dnn1_single
]
ALL_FAMILY_MODELS = [
    f"{load_string_single}_conv2d.10.8x3.rep3-wcn.90-relu.90-wc.90xR-lvl.R-dexp.R_{fit_string_single_c2d}",  #c2d
    f"{load_string_single}_wc.18x100.g-fir.1x25x100-relu.100.f-wc.100x120-relu.120.f-wc.120xR-lvl.R-dexp.R_{fit_string_single}", # c1d
    f"{load_string_single}_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_{fit_string_single}", # c1dx2+d
    f"{load_string_single}_wc.18x120.g-fir.1x25x120-wc.120xR-lvl.R-dexp.R_{fit_string_single}", # LN_pop
    f"{load_string_single}_wc.18x6.g-fir.1x25x6-relu.6.f-wc.6x1-lvl.1-dexp.1_{fit_string_dnn}"  # dnn1_single
]

shortnames=['conv2d','conv1d','conv1dx2','ln-pop', 'dnn-sing']
#shortnames=['conv1d','conv1dx2','dnn-sing']


####
# A1 expanded dataset (v>=2)
####
if VERSION > 1:
    NAT4_A1_SITES, rep_cellids = nd.get_batch_sites(322, POP_MODELS[1])
else:
    NAT4_A1_SITES = [
        'ARM029a', 'ARM030a', 'ARM031a',
        'ARM032a', 'ARM033a',
        'CRD016d', 'CRD017c',
        'DRX006b.e1:64', 'DRX006b.e65:128',
        'DRX007a.e1:64', 'DRX007a.e65:128',
        'DRX008b.e1:64', 'DRX008b.e65:128',
    ]

NAT4_PEG_SITES = [
    'ARM017a', 'ARM018a', 'ARM019a', 'ARM021b', 'ARM022b', 'ARM023a',
    'ARM024a', 'ARM025a', 'ARM026b', 'ARM027a', 'ARM028b'
]


# For correlation histograms
EQUIVALENCE_MODELS_SINGLE = [
    f"{load_string_single}_conv2d.10.8x3.rep3-wcn.90-relu.90-wc.90xR-lvl.R-dexp.R_{fit_string_single_c2d}",  #c2d
    f"{load_string_single}_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_{fit_string_single}", # c1dx2+d
    f"{load_string_single}_wc.18x120.g-fir.1x25x120-wc.120xR-lvl.R-dexp.R_{fit_string_single}",  # LN_pop
]
EQUIVALENCE_MODELS_POP = [
    f"{load_string_pop}_conv2d.10.8x3.rep3-wcn.90-relu.90-wc.90xR-lvl.R-dexp.R_{fit_string_pop_c2d}",  #c2d
    f"{load_string_pop}_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_{fit_string_pop}", # c1dx2+d
    f"{load_string_pop}_wc.18x120.g-fir.1x25x120-wc.120xR-lvl.R-dexp.R_{fit_string_pop}",  # LN_pop
]



# TODO: make sure these match

# DNN_SINGLE_MODELS = [
#         f"{load_string_single}_wc.18x12.g-fir.1x25x12-relu.12.f-wc.12x1-lvl.1-dexp.1_prefit.h-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
#         f"{load_string_single}_wc.18x12.g-fir.1x25x12-relu.12.f-wc.12x1-lvl.1-dexp.1_prefit.m-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4"
# ]

DNN_SINGLE_MODELS = [f'{load_string_single}_wc.18x2.g-fir.1x25x2-relu.2.f-wc.2x1-lvl.1-dexp.1_{fit_string_nopre}',
    f'{load_string_single}_wc.18x3.g-fir.1x25x3-relu.3.f-wc.3x1-lvl.1-dexp.1_{fit_string_nopre}',
    f'{load_string_single}_wc.18x4.g-fir.1x25x4-relu.4.f-wc.4x1-lvl.1-dexp.1_{fit_string_nopre}',
    f'{load_string_single}_wc.18x6.g-fir.1x25x6-relu.6.f-wc.6x1-lvl.1-dexp.1_{fit_string_nopre}',
    f'{load_string_single}_wc.18x9.g-fir.1x25x9-relu.9.f-wc.9x1-lvl.1-dexp.1_{fit_string_nopre}',
    f'{load_string_single}_wc.18x12.g-fir.1x25x12-relu.12.f-wc.12x1-lvl.1-dexp.1_{fit_string_nopre}',
    f'{load_string_single}_wc.18x15.g-fir.1x25x15-relu.15.f-wc.15x1-lvl.1-dexp.1_{fit_string_nopre}',
    f'{load_string_single}_wc.18x18.g-fir.1x25x18-relu.18.f-wc.18x1-lvl.1-dexp.1_{fit_string_nopre}'
                    ]
DNN_SINGLE_STAGE2 = [m.replace("tfinit.n.lr1e3.et3.rb10.es20","prefit.m-tfinit.n.lr1e3.et3.es20") for m in DNN_SINGLE_MODELS]

LN_SINGLE_MODELS = [f"{load_string_single}_wc.18x{rank}.g-fir.{rank}x25-lvl.1-dexp.1_{fit_string_nopre}"
                    for rank in range(1,13)]

STP_SINGLE_MODELS =[
    f"{load_string_single}_wc.18x1.g-stp.1.q.s-fir.1x25-lvl.1-dexp.1_{fit_string_nopre}",
    f"{load_string_single}_wc.18x2.g-stp.2.q.s-fir.2x25-lvl.1-dexp.1_{fit_string_nopre}",
    f"{load_string_single}_wc.18x3.g-stp.3.q.s-fir.3x25-lvl.1-dexp.1_{fit_string_nopre}",
    f"{load_string_single}_wc.18x4.g-stp.4.q.s-fir.4x25-lvl.1-dexp.1_{fit_string_nopre}",
    f"{load_string_single}_wc.18x5.g-stp.5.q.s-fir.5x25-lvl.1-dexp.1_{fit_string_nopre}",
]

HELDOUT_pop = [m.replace("loadpop","loadpop.hs") for m in POP_MODELS]
MATCHED_pop = [m.replace("loadpop","loadpop.hm") for m in POP_MODELS]

HELDOUT = [m.replace("prefit.f","prefit.hs") for m in SIG_TEST_MODELS]
HELDOUT[-1] = f"{load_string_single}_wc.18x6.g-fir.1x25x6-relu.6.f-wc.6x1-lvl.1-dexp.1_prefit.h-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4{vsuffixdnn}"

MATCHED = [m.replace("prefit.f","prefit.hm") for m in SIG_TEST_MODELS]
MATCHED[-1] = f"{load_string_single}_wc.18x6.g-fir.1x25x6-relu.6.f-wc.6x1-lvl.1-dexp.1_prefit.m-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4{vsuffixdnn}"



fit_string2 = "tfinit.n.lr1e3.et3.rb10.es20.v-newtf.n.lr1e4.es20.v"
fit_string3 = "tfinit.n.lr1e3.et3.rb10.es20.L2-newtf.n.lr1e4.es20.L2"

PLOT_STAT = 'r_ceiling'
#PLOT_STAT = 'r_test'

DOT_COLORS = {'conv2d': 'darkgreen', 'LN': 'black', 'conv1d': 'lightblue', #'conv1dx2': 'purple',
              'conv1dx2+d': 'purple', 'conv1dx2+dd': 'yellow', 'conv1dx2+d2': 'magenta', 'conv1dx2+d3': 'gray',
              'LN_pop': 'orange', 'dnn1': 'lightgreen', 'dnn1_single': 'lightgreen', 'c1dx2-stp': 'red', #'STP': 'lightblue',
              'LN_2d': 'purple',
              'c1d2_input': 'blue',
              'c1d2_tiny': 'blue',
              'c1d2_output': 'blue',
              'c1d2_25h20': 'blue',
              'c1d2_25h160': 'blue',
              'c2d_num_filters': 'darkgreen',
              'c2d_filter_length': 'darkgreen',
              'c2d_filter_reps': 'darkgreen',
              'c2d_10f': 'darkgreen',
              'conv2d_v': 'darkgreen',
              'conv2d_L2': 'darkgreen',
              'stp': 'red',
              }

DOT_MARKERS = {#'conv1dx2': '^',
               'conv2d': 's', 'LN_pop': 'o', 'conv1d': 'o',
               'LN':'.', 'dnn1': 'v', 'dnn1_single': 'v', 'c1dx2-stp': '*', #'STP': 'x',
               'conv1dx2+d': '+', 'LN_2d': 'x',
               'c1d2_input': '^',
               'c1d2_tiny': '>',
               'c1d2_output': 'v',
               'c1d2_25h20': '<',
               'c1d2_25h160': 'o',
               'c2d_num_filters': '^',
               'c2d_filter_length': '>',
               'c2d_filter_reps': 'v',
               'c2d_10f': '<',
               'conv2d_v': 'o',
               'conv2d_L2': '+',
               'stp': '.',
            }


# CELL_COUNT_TEST = []
# for c in [50, 100]:
#     CELL_COUNT_TEST.extend([
#         f"{load_string}-mc.{c}_conv2d.4.8x3.rep3-wcn.40-relu.40-wc.40xR-lvl.R-dexp.R_{fit_string}",
#         f"{load_string}-mc.{c}_wc.18x80.g-fir.1x25x80-relu.80.f-wc.80x100-relu.100.f-wc.100xR-lvl.R-dexp.R_{fit_string}", # c1d
#         f"{load_string}-mc.{c}_wc.18x30.g-fir.1x15x30-relu.30.f-wc.30x60-fir.1x10x60-relu.60.f-wc.60x80-relu.80-wc.80xR-lvl.R-dexp.R_{fit_string}", # c1dx2+d
#     ])


# Build modelnames
MODELGROUPS = {}
POP_MODELGROUPS = {}

# LN ###################################################################################################################
params = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]#, 12, 14]
MODELGROUPS['LN'] = [f'{load_string_single}_wc.18x{p}.g-fir.{p}x25-lvl.1-dexp.1_{fit_string_nopre}' for p in params]
POP_MODELGROUPS['LN'] = [f'{load_string_single}_wc.18x{p}.g-fir.{p}x25-lvl.1-dexp.1_{fit_string_nopre}' for p in params]

params = [2,3,4,5]
MODELGROUPS['stp'] = [f'{load_string_single}_wc.18xR.g-stp.R.q.s-fir.1x12xR-lvl.R-dexp.R_{fit_string_nopre}'] + \
    [f'{load_string_single}_wc.18x{p}R.g-stp.{p}R.q.s-fir.{p}x12xR-lvl.R-dexp.R_{fit_string_nopre}' for p in params]

POP_MODELGROUPS['stp'] = MODELGROUPS['stp'].copy()

# LN_pop ###############################################################################################################
params = [4, 6, 10, 14, 30, 42, 60, 80, 100, 120, 150, 175, 200, 250, 300]
MODELGROUPS['LN_pop'] = [f'{load_string_single}_wc.18x{p}.g-fir.1x25x{p}-wc.{p}xR-lvl.R-dexp.R_{fit_string_single}' for p in params]
POP_MODELGROUPS['LN_pop'] = [f'{load_string_pop}_wc.18x{p}.g-fir.1x25x{p}-wc.{p}xR-lvl.R-dexp.R_{fit_string_pop}' for p in params]


# conv1d ###############################################################################################################
L1_L2 = [
    (5, 10), (10, 10), (10, 20), (20, 30),
    (30, 40), (30, 50), (40, 50), (50, 60),
    (60, 80), (80, 100), (100, 120), (120, 140),
    (140, 160), (170, 200), (200, 250), (230, 300)
]
MODELGROUPS['conv1d'] = [
    f"{load_string_single}_wc.18x{layer1}.g-fir.1x25x{layer1}-relu.{layer1}.f-"
    + f"wc.{layer1}x{layer2}-relu.{layer2}.f-wc.{layer2}xR-lvl.R-dexp.R_{fit_string_single}"
    for layer1, layer2 in L1_L2
]
POP_MODELGROUPS['conv1d'] = [
    f"{load_string_pop}_wc.18x{layer1}.g-fir.1x25x{layer1}-relu.{layer1}.f-"
    + f"wc.{layer1}x{layer2}-relu.{layer2}.f-wc.{layer2}xR-lvl.R-dexp.R_{fit_string_pop}"
    for layer1, layer2 in L1_L2
]


# conv1dx2+d ###########################################################################################################
L1_L2_L3 = [
    (5, 10, 20), (10, 10, 20), (10, 20, 30),
    (20, 20, 40), (20, 40, 60), (30, 60, 80), (50, 70, 90), (70, 80, 100),
    (70, 90, 120), (80, 100, 140), (90, 120, 160), (100, 140, 180),
    (120, 160, 220), (150, 200, 250), #(180, 250, 300)
]
MODELGROUPS['conv1dx2+d'] = [
    f"{load_string_single}_wc.18x{layer1}.g-fir.1x15x{layer1}-relu.{layer1}.f-wc.{layer1}x{layer2}-fir.1x10x{layer2}-"
    + f"relu.{layer2}.f-wc.{layer2}x{layer3}-relu.{layer3}-wc.{layer3}xR-lvl.R-dexp.R_{fit_string_single}"
    for layer1, layer2, layer3 in L1_L2_L3
]
POP_MODELGROUPS['conv1dx2+d'] = [
    f"{load_string_pop}_wc.18x{layer1}.g-fir.1x15x{layer1}-relu.{layer1}.f-wc.{layer1}x{layer2}-fir.1x10x{layer2}-"
    + f"relu.{layer2}.f-wc.{layer2}x{layer3}-relu.{layer3}-wc.{layer3}xR-lvl.R-dexp.R_{fit_string_pop}"
    for layer1, layer2, layer3 in L1_L2_L3
]

# vary input space, starting from 70x80x100
# inputs = [5, 10, 25, 40, 55, 70, 85, 100, 125, 150]
# MODELGROUPS['c1d2_input'] = [
#     f"{load_string}_wc.18x{layer1}.g-fir.1x15x{layer1}-relu.{layer1}.f-wc.{layer1}x80-fir.1x10x80-"
#     + f"relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_{fit_string}"
#     for layer1 in inputs
# ]

# vary output layer, starting from 70x80x100
# outputs = [20, 40, 60, 80, 100, 120, 140, 160]
# MODELGROUPS['c1d2_output'] = [
#     f"{load_string}_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-"
#     + f"relu.80.f-wc.80x{out}-relu.{out}-wc.{out}xR-lvl.R-dexp.R_{fit_string}"
#     for out in outputs
# ]


# # using input 5, output 20
# hidden = [20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 200, 300, 400, 500]
# MODELGROUPS['c1d2_5h20'] = [
#     f"{load_string}_wc.18x5.g-fir.1x15x5-relu.5.f-wc.5x{h}-fir.1x10x{h}-"
#     + f"relu.{h}.f-wc.{h}x20-relu.20-wc.20xR-lvl.R-dexp.R_{fit_string}"
#     for h in hidden
# ]

# using input 25, output 20
# hidden = [20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 200, 300, 400, 500]
# MODELGROUPS['c1d2_25h20'] = [
#     f"{load_string}_wc.18x25.g-fir.1x15x25-relu.25.f-wc.25x{h}-fir.1x10x{h}-"
#     + f"relu.{h}.f-wc.{h}x20-relu.20-wc.20xR-lvl.R-dexp.R_{fit_string}"
#     for h in hidden
# ]

# using "", output 160
# MODELGROUPS['c1d2_25h160'] = [
#     f"{load_string}_wc.18x25.g-fir.1x15x25-relu.25.f-wc.25x{h}-fir.1x10x{h}-"
#     + f"relu.{h}.f-wc.{h}x160-relu.160-wc.160xR-lvl.R-dexp.R_{fit_string}"
#     for h in hidden
# ]


# tiny version, for heldout comparison
# L1_L2_L3 = [
#     (5, 5, 5), (5, 10, 10), (5, 15, 10)
# ]
# MODELGROUPS['c1d2_tiny'] = [
#     f"{load_string}_wc.18x{layer1}.g-fir.1x15x{layer1}-relu.{layer1}.f-wc.{layer1}x{layer2}-fir.1x10x{layer2}-"
#     + f"relu.{layer2}.f-wc.{layer2}x{layer3}-relu.{layer3}-wc.{layer3}xR-lvl.R-dexp.R_{fit_string}"
#     for layer1, layer2, layer3 in L1_L2_L3
# ]


# conv2d ###############################################################################################################
# params = [
#     # (num_filters, filter_length, filter_width, layer_reps, dense_count)
#     (2, 8, 3, 3, 4), (4, 8, 3, 3, 8), (4, 8, 3, 3, 12), (4, 8, 3, 3, 20),
#     (4, 8, 3, 3, 40), (4, 8, 3, 3, 50), (4, 8, 3, 3, 70),
#     (4, 8, 3, 3, 90), (4, 8, 3, 3, 110), (4, 8, 3, 3, 130), (4, 8, 3, 3, 150),
#     (4, 8, 3, 3, 175), (4, 8, 3, 3, 200), (4, 8, 3, 3, 250), (4, 8, 3, 3, 300)
# ]
# MODELGROUPS['conv2d'] = [
#     f"{load_string}_conv2d.{num_filters}.{filter_length}x{filter_width}.rep{layer_reps}-wcn.{dense_count}-"
#     + f"relu.{dense_count}-wc.{dense_count}xR-lvl.R-dexp.R_{fit_string}"
#     for num_filters, filter_length, filter_width, layer_reps, dense_count in params
# ]


# try fixing with 10 filters
dense_counts = [4, 8, 12, 20, 40, 50, 70, 90, 110, 130, 150, 175, 200, 250, 300]#, 400]
MODELGROUPS['conv2d'] = [
    f"{load_string_single}_conv2d.10.8x3.rep3-wcn.{dense}-"
    + f"relu.{dense}-wc.{dense}xR-lvl.R-dexp.R_{fit_string_single_c2d}"
    for dense in dense_counts
]
POP_MODELGROUPS['conv2d'] = [
    f"{load_string_pop}_conv2d.10.8x3.rep3-wcn.{dense}-"
    + f"relu.{dense}-wc.{dense}xR-lvl.R-dexp.R_{fit_string_pop_c2d}"
    for dense in dense_counts
]

# MODELGROUPS['conv2d_v'] = [
#     f"{load_string}_conv2d.10.8x3.rep3-wcn.{dense}-"
#     + f"relu.{dense}-wc.{dense}xR-lvl.R-dexp.R_{fit_string2}"
#     for dense in dense_counts
# ]

# MODELGROUPS['conv2d_L2'] = [
#     f"{load_string}_conv2d.10.8x3.rep3-wcn.{dense}-"
#     + f"relu.{dense}-wc.{dense}xR-lvl.R-dexp.R_{fit_string3}"
#     for dense in dense_counts
# ]


# vary filter parameters, starting from (4, 8, 3, 3, 50)
# num_filters = [
#     # (num_filters, filter_length, filter_width, layer_reps, dense_count)
#     1, 2, 4, 6, 8, 10, 12, 14
# ]
# MODELGROUPS['c2d_num_filters'] = [
#     f"{load_string}_conv2d.{nf}.8x3.rep3-wcn.50-"
#     + f"relu.50-wc.50xR-lvl.R-dexp.R_{fit_string}"
#     for nf in num_filters
# ]

# filter_length = [
#     4, 6, 8, 10, 12, 14, 16
# ]
# MODELGROUPS['c2d_filter_length'] = [
#     f"{load_string}_conv2d.4.{fl}x3.rep3-wcn.50-"
#     + f"relu.50-wc.50xR-lvl.R-dexp.R_{fit_string}"
#     for fl in filter_length
# ]

# filter_reps = [
#     1, 2, 3, 4, 5
# ]
# MODELGROUPS['c2d_filter_reps'] = [
#     f"{load_string}_conv2d.4.8x3.rep{fr}-wcn.50-"
#     + f"relu.50-wc.50xR-lvl.R-dexp.R_{fit_string}"
#     for fr in filter_reps
# ]



# dnn1_single ##########################################################################################################
params = [2,3,4, 6, 9, 12, 15, 18]
MODELGROUPS['dnn1_single'] = [
    f"{load_string_single}_wc.18x{p}.g-fir.1x25x{p}-relu.{p}.f-wc.{p}x1-lvl.1-dexp.1_{fit_string_dnn}"
    for p in params
]
POP_MODELGROUPS['dnn1_single'] = [
    f"{load_string_single}_wc.18x{p}.g-fir.1x25x{p}-relu.{p}.f-wc.{p}x1-lvl.1-dexp.1_{fit_string_nopre}"
    for p in params
]
# _single flag tells pareto plot function not to divide by cells per site
# the double conv layer models were doing substantially worse per parameter for dnn1, so leaving them out for now.


# heldout models #######################################################################################################
# modelstrings = [m.split('_')[1] for m in SIG_TEST_MODELS]  # temporarily only conv1dx2  # TODO change me back
# layers = ['0:5', '0:5', '0:8', '0:2'] # ,'0:2']
# fs1, fs2 = fit_string.split('-')
# fs1 = fs1.replace('.rb10', '')  # don't do random init on second fit stage
# #MAX_CELL_COUNTS = [10, 20, 50, 100, 200, 300]
# MAX_CELL_COUNTS = [75, 100, 200, 300]
# #RANDOM_SEEDS = [0, 1, 2, 3, 4]
# RANDOM_SEEDS = [0]
#
# def build_truncated(sites, flatten=False):
#     heldout_modelnames = []
#     matched_modelnames = []
#
#     for siteid in sites:
#         heldout_model_stubs = [f"{load_string}-hc.{siteid}_{modelstring}_{fit_string}" for modelstring in modelstrings]
#         heldout_modelnames.extend(heldout_model_stubs)
#         matched_model_stubs = [f"{load_string}-hc.ms.{siteid}_{modelstring}_{fit_string}" for modelstring in modelstrings]
#         matched_modelnames.extend(matched_model_stubs)
#
#     # for n_cells in MAX_CELL_COUNTS:
#     #     this_count = []
#     #     for random_seed in RANDOM_SEEDS:
#     #         this_seed = []
#     #         for siteid in sites:
#     #             this_site = [f"{load_string}-hc.{siteid}-mc.{n_cells}.sd{random_seed}_{modelstring}_{fit_string}"
#     #                          for modelstring in modelstrings]
#     #             this_seed.append(this_site)
#     #         this_count.append(this_seed)
#     #     max_modelnames.append(this_count)
#
#     # if flatten:
#     #     max_modelnames = np.array(max_modelnames).flatten().tolist()
#
#     return heldout_modelnames, matched_modelnames
#
# h322, m322 = build_truncated(NAT4_A1_SITES, flatten=True)
# h323, m323 = build_truncated(NAT4_PEG_SITES, flatten=True)
# TRUNCATED = {322: h322, 323: h323}
# TRUNCATED_MATCHED = {322: m322, 323: m323}
#
#
# def build_heldout(sites):
#     modelnames = []
#     for siteid in sites:
#         heldout_model_stubs = [f"{load_string}-hc.{siteid}_{modelstring}_{fit_string}" for modelstring in modelstrings]
#
#         models1 = [m + f"-tfheld.FL{l}-" + fs1 + '-' + fs2 for m, l in zip(heldout_model_stubs, layers)]
#         modelnames.extend(models1)
#
#     return modelnames
#
# HELDOUT = {322: build_heldout(NAT4_A1_SITES), 323: build_heldout(NAT4_PEG_SITES)}
#
#
# # heldout max ##########################################################################################################
# def build_max(sites, flatten=False):
#     modelnames = []
#     for n_cells in MAX_CELL_COUNTS:
#         this_count = []
#         for random_seed in RANDOM_SEEDS:
#             this_seed = []
#             for siteid in sites:
#                 max_model_stubs = [f"{load_string}-hc.{siteid}-mc.{n_cells}.sd{random_seed}_{modelstring}_{fit_string}"
#                                    for modelstring in modelstrings]
#
#                 this_site = [m + f"-tfheld.FL{l}-" + fs1 + '-' + fs2 for m, l in zip(max_model_stubs, layers)]
#                 this_seed.append(this_site)
#             this_count.append(this_seed)
#         modelnames.append(this_count)
#
#     if flatten:
#         modelnames = np.array(modelnames).flatten().tolist()
#
#     return modelnames
#
# HELDOUT_MAX = {322: build_max(NAT4_A1_SITES, flatten=True), 323: build_max(NAT4_PEG_SITES, flatten=True)}
#
#
# def build_max_matched(sites, flatten=False):
#     modelnames = []
#     for n_cells in MAX_CELL_COUNTS:
#         this_count = []
#         for random_seed in RANDOM_SEEDS:
#             this_seed = []
#             for siteid in sites:
#                 max_model_stubs = [f"{load_string}-hc.ms.{siteid}-mc.{n_cells}.sd{random_seed}_{modelstring}_{fit_string}"
#                                    for modelstring in modelstrings]
#
#                 this_site = [m + f"-tfheld.FL{l}-" + fs1 + '-' + fs2 for m, l in zip(max_model_stubs, layers)]
#                 this_seed.append(this_site)
#             this_count.append(this_seed)
#         modelnames.append(this_count)
#
#     if flatten:
#         modelnames = np.array(modelnames).flatten().tolist()
#
#     return modelnames
#
# MATCHED_MAX = {322: build_max_matched(NAT4_A1_SITES, flatten=True), 323: build_max_matched(NAT4_PEG_SITES, flatten=True)}
#
#
# # match models #########################################################################################################
# def build_matched(sites):
#     site_models = []
#     random_models = []
#     for siteid in sites:
#         heldout_model_stubs = [f"{load_string}-hc.ms.{siteid}_{modelstring}_{fit_string}" for modelstring in modelstrings]
#
#         models2 = [m + f"-tfheld.FL{l}-" + fs1 + '-' + fs2 for m, l in zip(heldout_model_stubs, layers)]
#         site_models.extend(models2)
#
#     return site_models#, random_models
#
# s_322 = build_matched(NAT4_A1_SITES)
# s_323 = build_matched(NAT4_PEG_SITES)
# MATCHED_SITE = {322: s_322, 323: s_323}
# #MATCHED_RANDOM = {322: r_322, 323: r_323}
#
#
# # LN dummy heldout #####################################################################################################
# LN_HELDOUT = (f"{load_string}_wc.18x4R.g-fir.4x25xR-lvl.R-dexp.R_{fit_string}" + "-tfheld.same.FL0:1-" + fs1 + '-' + fs2)


#
# Partial est data models
#
# tentative: use best conv1dx2+d
half_test_modelspec = "wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R"
_tfitter = "tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4" + vsuffix
_tfitter_rb = fit_string_nopre

# test condition: take advantage of larger model population fit (hs: heldout), then fit single cell with half a dataset
# fit last layer on half the data, using prefit with the current site held-out, run per cell
modelname_half_prefit=[f"{load_string_single}.k10_{half_test_modelspec}_prefit.hs-{_tfitter}",
                       f"{load_string_single}.k15_{half_test_modelspec}_prefit.hs-{_tfitter}",
                       f"{load_string_single}.k25_{half_test_modelspec}_prefit.hs-{_tfitter}",
                       f"{load_string_single}.k50_{half_test_modelspec}_prefit.hs-{_tfitter}",
                       f"{load_string_single}_{half_test_modelspec}_prefit.hm-{_tfitter}"]

# control condition: fit pop model then single cell with half the data. hm/hhm: exclude matched to preserve balance with heldout
# fit held-out pop model with half the est data, run per site
modelname_half_pop=[f"ozgf.fs100.ch18.pop-loadpop.hm-norm.l1-popev.k10_{half_test_modelspec}_{_tfitter_rb}",
                    f"ozgf.fs100.ch18.pop-loadpop.hm-norm.l1-popev.k15_{half_test_modelspec}_{_tfitter_rb}",
                    f"ozgf.fs100.ch18.pop-loadpop.hm-norm.l1-popev.k25_{half_test_modelspec}_{_tfitter_rb}",
                    f"ozgf.fs100.ch18.pop-loadpop.hm-norm.l1-popev.k50_{half_test_modelspec}_{_tfitter_rb}",
                    f"ozgf.fs100.ch18.pop-loadpop.hm-norm.l1-popev_{half_test_modelspec}_{_tfitter_rb}"]

# then fit last layer on heldout cell with half the data (same est data as for modelname_half_prefit), run per cell
modelname_half_fullfit=[f"{load_string_single}.k10_{half_test_modelspec}_prefit.htm-{_tfitter}",
                        f"{load_string_single}.k15_{half_test_modelspec}_prefit.hfm-{_tfitter}",
                        f"{load_string_single}.k25_{half_test_modelspec}_prefit.hqm-{_tfitter}",
                        f"{load_string_single}.k50_{half_test_modelspec}_prefit.hhm-{_tfitter}",
                        f"{load_string_single}_{half_test_modelspec}_prefit.hm-{_tfitter}"]

# control condition: fit pop model then single cell with half the data. hm/hhm: exclude matched to preserve balance with heldout
# fit held-out pop model with half the est data, run per site
modelname_half_heldoutpop=[f"ozgf.fs100.ch18.pop-loadpop.hs-norm.l1-popev.k10_{half_test_modelspec}_{_tfitter_rb}",
                           f"ozgf.fs100.ch18.pop-loadpop.hs-norm.l1-popev.k15_{half_test_modelspec}_{_tfitter_rb}",
                           f"ozgf.fs100.ch18.pop-loadpop.hs-norm.l1-popev.k25_{half_test_modelspec}_{_tfitter_rb}",
                           f"ozgf.fs100.ch18.pop-loadpop.hs-norm.l1-popev.k50_{half_test_modelspec}_{_tfitter_rb}",
                           f"ozgf.fs100.ch18.pop-loadpop.hs-norm.l1-popev_{half_test_modelspec}_{_tfitter_rb}"]

# then fit last layer on heldout cell with half the data (same est data as for modelname_half_prefit), run per cell
modelname_half_heldoutfullfit=[f"{load_string_single}.k10_{half_test_modelspec}_prefit.hts-{_tfitter}",
                               f"{load_string_single}.k15_{half_test_modelspec}_prefit.hfs-{_tfitter}",
                               f"{load_string_single}.k25_{half_test_modelspec}_prefit.hqs-{_tfitter}",
                               f"{load_string_single}.k50_{half_test_modelspec}_prefit.hhs-{_tfitter}",
                               f"{load_string_single}_{half_test_modelspec}_prefit.hs-{_tfitter}"]

ln_half_models = [
    f'{load_string_single}.k15_wc.18x4.g-fir.4x25-lvl.1-dexp.1_{_tfitter_rb}',
    f'{load_string_single}.k25_wc.18x4.g-fir.4x25-lvl.1-dexp.1_{_tfitter_rb}',
    f'{load_string_single}.k50_wc.18x4.g-fir.4x25-lvl.1-dexp.1_{_tfitter_rb}',
    f'{load_string_single}_wc.18x4.g-fir.4x25-lvl.1-dexp.1_{_tfitter_rb}',
]
dnns_half_models = [
    f'{load_string_single}.k15_wc.18x12.g-fir.1x25x12-relu.12.f-wc.12x1-lvl.1-dexp.1_{_tfitter_rb}',
    f'{load_string_single}.k25_wc.18x12.g-fir.1x25x12-relu.12.f-wc.12x1-lvl.1-dexp.1_{_tfitter_rb}',
    f'{load_string_single}.k50_wc.18x12.g-fir.1x25x12-relu.12.f-wc.12x1-lvl.1-dexp.1_{_tfitter_rb}',
    f"{load_string_single}_wc.18x12.g-fir.1x25x12-relu.12.f-wc.12x1-lvl.1-dexp.1_prefit.m-{_tfitter}"
]

def count_fits(models, batch=None):
    if batch is None:
        batches=[322,323]
    else:
        batches=[batch]

    for batch in batches:
        df_r = nd.batch_comp(batch, models, stat='r_test')
        #print(f"BATCH {batch}:")
        for i,c in enumerate(df_r.columns):
            parts=c.split("_")
            if i==0:
                print(f"BATCH {batch} -- {parts[0]}_XX_{parts[2]}")
            print(f'{parts[1]}: {df_r[c].count()}')

def get_significant_cells(batch, models, as_list=False):
    df_r = nd.batch_comp(batch, models, stat='r_test')
    df_r.dropna(axis=0, how='any', inplace=True)
    df_r.sort_index(inplace=True)
    df_e = nd.batch_comp(batch, models, stat='se_test')
    df_e.dropna(axis=0, how='any', inplace=True)
    df_e.sort_index(inplace=True)
    df_f = nd.batch_comp(batch, models, stat='r_floor')
    df_f.dropna(axis=0, how='any', inplace=True)
    df_f.sort_index(inplace=True)

    masks = []
    for m in models:
        mask1 = df_r[m] > df_e[m] * 2
        mask2 = df_r[m] > df_f[m]
        mask = mask1 & mask2
        masks.append(mask)

    all_significant = masks[0]
    for m in masks[1:]:
        all_significant &= m

    if as_list:
        all_significant = all_significant[all_significant].index.values.tolist()

    return all_significant


def snr_by_batch(batch, loadkey, save_path=None, load_path=None, frac_total=True, rec=None, siteids=None):
    snrs = []
    cells = []

    if load_path is None:

        if rec is None:
            if siteids is None:
                cellids = nd.get_batch_cells(batch, as_list=True)
                siteids = list(set([c.split('-')[0] for c in cellids]))

            for site in siteids:
                rec_path = xwrap.generate_recording_uri(site, batch, loadkey=loadkey)
                rec = nems.recording.load_recording(rec_path)
                est, val = rec.split_using_epoch_occurrence_counts('^STIM_')
                val = val.apply_mask()
                for cellid in rec['resp'].chans:
                    resp = val['resp'].extract_channels([cellid])
                    snr = compute_snr(resp, frac_total=frac_total)
                    snrs.append(snr)
                    cells.append(cellid)
                    print(f"{cellid}: {snr:.3f}")
                    
        else:
            if isinstance(rec, str):
                rec = nems.recording.load_recording(rec)
            cellids = rec['resp'].chans
            est, val = rec.split_using_epoch_occurrence_counts('^STIM_')
            val = val.apply_mask()
            for cellid in cellids:
                log.info("computing SNR for cell: %s" % cellid)
                resp = val['resp'].extract_channels([cellid])
                snr = compute_snr(resp, frac_total=frac_total)
                snrs.append(snr)
            cells = cellids

        results = {'cellid': cells, 'snr': snrs}
        df = pd.DataFrame.from_dict(results)
        df.dropna(inplace=True)
        df.set_index('cellid', inplace=True)
        if save_path is not None:
            df.to_csv(save_path)

    else:
        df = pd.read_csv(load_path, index_col=0)

    return df


def compute_snr(resp, frac_total=True):
    epochs = resp.epochs
    stim_epochs = ep.epoch_names_matching(epochs, 'STIM_')
    resp_dict = resp.extract_epochs(stim_epochs)

    per_stim_snrs = []
    for stim, resp in resp_dict.items():
        resp = resp.squeeze()
        if resp.ndim == 1:
            # Only one stim rep, have to add back in axis for number of reps
            resp = np.expand_dims(resp, 0)
        products = np.dot(resp, resp.T)
        per_rep_snrs = []

        for i, _ in enumerate(resp):
            total_power = products[i,i]
            signal_powers = np.delete(products[i], i)
            if total_power>0:
                if frac_total:
                    rep_snr = np.nanmean(signal_powers)/total_power
                else:
                    rep_snr = np.nanmean(signal_powers/(total_power-signal_powers))

                per_rep_snrs.append(rep_snr)
        if len(per_rep_snrs)>0:
            per_stim_snrs.append(np.nanmean(per_rep_snrs))
    
    snr = np.nanmean(per_stim_snrs)
    if snr==0:
        import pdb; pdb.set_trace()
    # if np.sum(np.isnan(per_stim_snrs)) == len(per_stim_snrs):
    #     import pdb; pdb.set_trace()

    return np.nanmean(per_stim_snrs)


def get_rceiling_correction(batch):
    LN_model = MODELGROUPS['LN'][3]

    rceiling_ratios = []
    significant_cells = get_significant_cells(batch, SIG_TEST_MODELS, as_list=True)
    rtest = nd.batch_comp(batch, [LN_model], cellids=significant_cells, stat='r_test')
    rceiling = nd.batch_comp(batch, [LN_model], cellids=significant_cells, stat='r_ceiling')

    rceiling_ratios = rceiling[LN_model] / rtest[LN_model]
    rceiling_ratios.loc[rceiling_ratios < 1] = 1

    return rceiling_ratios


def set_equal_axes(ax, aspect=1):
    # Set same limits
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()

    ymin = min(ymin, xmin)
    ymax = max(ymax, xmax)
    xmin = ymin
    xmax = ymax

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_aspect(aspect='equal')
