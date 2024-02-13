import nems0.db as nd
import re
import itertools
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import copy
import nems_lbhb.projects.olp.OLP_helpers as ohel
import nems_lbhb.projects.olp.OLP_fit as ofit
import nems_lbhb.projects.olp.OLP_Synthetic_plot as osyn
import nems_lbhb.projects.olp.OLP_Binaural_plot as obip
import nems_lbhb.projects.olp.OLP_plot_helpers as oph
import nems_lbhb.projects.olp.OLP_figures as ofig
import nems_lbhb.projects.olp.OLP_plot_helpers as opl
import nems_lbhb.projects.olp.OLP_poster as opo
import scipy.ndimage.filters as sf
from scipy import stats
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems0.analysis.gammatone.gtgram import gtgram
from scipy.io import wavfile
import glob
import nems0.epoch as ep
from nems0 import db
import re
import nems_lbhb.SPO_helpers as sp
from nems0.xform_helper import load_model_xform
from datetime import date
import joblib as jl
from nems_lbhb import baphy_io
plt.rcParams['svg.fonttype'] = 'none'

sb.color_palette
sb.color_palette('colorblind')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sb.color_palette('colorblind'))
fs = 100

# big old naive dataset
path = '/auto/users/hamersky/olp_analysis/2023-09-22_batch344_0-500_final'
weight_df = jl.load(path)

# 2024_02_08. Addresses 'look at naive data only for kit fg'
filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=True)
kit_filt = filt.loc[(filt.FG == 'KitWhine') | (filt.FG == 'KitHigh') |
                    (filt.FG == 'Kit_Whine') | (filt.FG == 'Kit_High') | (filt.FG == 'Kit_Low')]
stat_dict = ofig.weight_summary_histograms_manuscript(kit_filt, bar=True, stat_plot='median')

filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=False)
kit_filt = filt.loc[(filt.FG=='KitWhine') | (filt.FG=='KitHigh') |
                    (filt.FG=='Kit_Whine') | (filt.FG=='Kit_High') | (filt.FG=='Kit_Low')]
ofig.all_filter_stats(kit_filt, xcol='bg_snr', ycol='fg_snr', snr_thresh=0.12, r_cut=0.4, increment=0.2,
                 fr_thresh=0.01, xx='resp')


# "naive" comparison
path = '/auto/users/hamersky/olp_analysis/2024-01-08_batch352_LEMON_OLP_standard'
path = '/auto/users/hamersky/olp_analysis/2024-02-08_batch352_LEMON_OLP_standard'
weight_df = jl.load(path)

# All Lemon "naive" weights and inclusion criteria
filt = get_olp_filter(weight_df, kind='vanilla', metric=True)
stat_dict = ofig.weight_summary_histograms_manuscript(filt, bar=True, stat_plot='median')


filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=False)
ofig.all_filter_stats(filt, xcol='bg_snr', ycol='fg_snr', snr_thresh=0.12, r_cut=0.4, increment=0.2,
                 fr_thresh=0.01, xx='resp')

# FG specifics
filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=True)
ferret_filt = filt.loc[(filt.FG == 'FightSqueak')]
ferret_filt = filt.loc[(filt.FG == 'Xylophone') | (filt.FG == 'WomanA')]
ferret_filt = filt.loc[(filt.FG == 'ferretb2001R')]
stat_dict = ofig.weight_summary_histograms_manuscript(ferret_filt, bar=True, stat_plot='median')

filt = ohel.get_olp_filter(weight_df, kind='vanilla', metric=False)
ferret_filt = filt.loc[(filt.FG == 'FightSqueak')]
ferret_filt = filt.loc[(filt.FG == 'Xylophone') | (filt.FG == 'WomanA')]
ferret_filt = filt.loc[(filt.FG == 'ferretb2001R')]
ofig.all_filter_stats(ferret_filt, xcol='bg_snr', ycol='fg_snr', snr_thresh=0.12, r_cut=0.4, increment=0.2,
                 fr_thresh=0.01, xx='resp')
