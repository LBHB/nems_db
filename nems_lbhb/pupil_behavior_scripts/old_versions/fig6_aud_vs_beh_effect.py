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
import nems_lbhb.pupil_behavior_scripts.common as common
import nems_lbhb.pupil_behavior_scripts.helpers as helper

save_path = os.path.join(os.path.expanduser('~'),'docs/current/pupil_behavior/eps')
save_fig = False

# SPECIFY pup+beh models
state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
basemodel = "-ref-psthfr.s_sdexp.S"
basemodel2= "-ref-psthfr.s_stategain.S"

#
# Figures 6A-B  - separate pup and beh effects

#df = pd.read_csv('pup_beh_processed.csv')
df = pd.read_csv('pup_beh_processed'+basemodel+'.csv')

#xsubset = df.cellid.str.startswith('AMT018') | df.cellid.str.startswith('AMT020')
xsubset = df.cellid.str.startswith('AMT')
#xsubset = df.cellid.str.startswith('XXXXXX')

# creating list of booleans to mask A1, IC, onBF and offBF out of big df
A1 = (df['area']=='A1') & ~xsubset
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

df['r_se'] = df['r_se'].str.strip(to_strip="[]").astype(float)

# 6A
f = aud_vs_state(df.loc[A1], nb=5, colors=common.color_list, title='A1')
if save_fig:
    f.savefig(os.path.join(save_path,'fig6_tuning_vs_pup_beh_A1.pdf'))

# 6B
f = aud_vs_state(df.loc[ICC | ICX], nb=5, colors=common.color_list, title='IC')
if save_fig:
    f.savefig(os.path.join(save_path,'fig6_tuning_vs_pup_beh_IC.pdf'))


# Figures 6C-D  - beh only effects, bigger set of cells
# later figure -- beh only (ignore pupil, can use larger stim set)

#dfb = pd.read_csv('beh_only_processed.csv')
dfb = pd.read_csv('beh_only_processed'+basemodel2+'.csv')

# creating subdf with only rows that match conditions
is_active = (dfb['state_chan'] == 'active')
full_model = (dfb['state_sig'] == 'st.beh')
null_model = (dfb['state_sig'] == 'st.beh0')

#xsubset = dfb.cellid.str.startswith('AMT018') | dfb.cellid.str.startswith('AMT020') | dfb.cellid.str.startswith('oni015b-b1')
xsubset = dfb.cellid.str.startswith('AMT') | dfb.cellid.str.startswith('oni015b-b1')
#xsubset = dfb.cellid.str.startswith('XXXXXX')

# creating list of booleans to mask A1, IC, onBF and offBF out of big df
A1 = (dfb['area']=='A1') & ~xsubset
IC = dfb['area']=='IC'
SU = dfb['SU']==True
sig_ubeh = dfb['sig_ubeh']==True
sig_upup = dfb['sig_upup']==True
sig_both = sig_ubeh & sig_upup
sig_state = dfb['sig_state']==True
sig_obeh = dfb['sig_obeh']==True
sig_oubeh = sig_ubeh | sig_obeh

print((dfb.loc[full_model & is_active & A1 & sig_state, 'MIbeh_only']).median())

#dfb['r_se'] = dfb['r_se'].str.strip(to_strip="[]").astype(float)

f = aud_vs_state(dfb.loc[A1], nb=5, state_list=['st.beh0','st.beh'], colors=common.color_list, title='A1')
if save_fig:
    f.savefig(os.path.join(save_path,'fig6_tuning_vs_beh_only_A1.pdf'))

f = aud_vs_state(dfb.loc[IC], nb=5, state_list=['st.beh0','st.beh'], colors=common.color_list, title='IC')
if save_fig:
    f.savefig(os.path.join(save_path,'fig6_tuning_vs_beh_only_IC.pdf'))


