import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as sci
import statsmodels.formula.api as smf
import matplotlib.collections as clt
import re
import pylab as pl

from nems_lbhb.pupil_behavior_scripts.mod_per_state import get_model_results_per_state_model
from nems_lbhb.pupil_behavior_scripts.mod_per_state import aud_vs_state
from nems_lbhb.pupil_behavior_scripts.mod_per_state import hlf_analysis
from nems_lbhb.pupil_behavior_scripts.mod_per_state import beh_only_plot
from nems_lbhb.stateplots import model_per_time_wrapper, beta_comp
import common


# SPECIFY pup+beh models
state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
basemodel = "-ref-psthfr.s_sdexp.S"

#
# Figures 6A-B  - separate pup and beh effects

df = pd.read_csv('pup_beh_processed.csv')

# creating list of booleans to mask A1, IC, onBF and offBF out of big df
A1 = df['area']=='A1'
ICC = df['area']=='ICC'
ICX = df['area']=='ICX'
onBF = df['onBF']==True
offBF = df['onBF']==False
SU = df['SU']==True
sig_ubeh = df['sig_ubeh']==True
sig_upup = df['sig_upup']==True
sig_both = sig_ubeh & sig_upup
sig_state = df['sig_state']==True
sig_obeh = df['sig_obeh']==True
sig_oubeh = sig_ubeh | sig_obeh

# 6A
aud_vs_state(df.loc[A1], nb=5)

# 6B
aud_vs_state(df.loc[ICC | ICX], nb=5)


# Figures 6C-D  - beh only effects, bigger set of cells
# later figure -- beh only (ignore pupil, can use larger stim set)

dfb = pd.read_csv('beh_only_processed.csv')

# creating subdf with only rows that match conditions
is_active = (dfb['state_chan'] == 'active')
full_model = (dfb['state_sig'] == 'st.beh')
null_model = (dfb['state_sig'] == 'st.beh0')

A1 = dfb['area']=='A1'
IC = dfb['area']=='IC'
SU = dfb['SU']==True
sig_ubeh = dfb['sig_ubeh']==True
sig_upup = dfb['sig_upup']==True
sig_both = sig_ubeh & sig_upup
sig_state = dfb['sig_state']==True
sig_obeh = dfb['sig_obeh']==True
sig_oubeh = sig_ubeh | sig_obeh

print((dfb.loc[full_model & is_active & A1 & sig_state, 'MIbeh_only']).median())

if 0:
    # A1
    common.scat_states(dfb, x_model=null_model,
                y_model=full_model,
                beh_state=is_active,
                area=A1,
                sig_list=[~sig_state, sig_state],
                x_column='R2',
                y_column='R2',
                color_list=common.color_list,
                #highlight_cellids={'TAR010c-27-2':'red', 'TAR010c-06-1':'blue'},
                save=False,
                xlim=(0,1),
                ylim=(0,1),
                xlabel='state-independent R2',
                ylabel='state-dependent R2',
                title='A1')

    # IC
    common.scat_states(dfb, x_model=null_model,
                y_model=full_model,
                beh_state=is_active,
                area=IC,
                sig_list=[~sig_state, sig_state],
                x_column='R2',
                y_column='R2',
                color_list=common.color_list,
                #highlight_cellids={'TAR010c-27-2':'red', 'TAR010c-06-1':'blue'},
                save=False,
                xlim=(0,1),
                ylim=(0,1),
                xlabel='state-independent R2',
                ylabel='state-dependent R2',
                title='IC')

aud_vs_state(dfb.loc[A1], nb=5, state_list=['st.beh0','st.beh'])
aud_vs_state(dfb.loc[IC], nb=5, state_list=['st.beh0','st.beh'])


