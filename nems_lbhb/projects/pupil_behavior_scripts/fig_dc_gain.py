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
#basemodel = "-ref-psthfr.s_sdexp.S"
basemodel = "-ref-psthfr.s_stategain.S"

#df.pd.read_csv('pup_beh_processed.csv')
df=pd.read_csv('pup_beh_processed'+basemodel+'.csv')

# creating subdf with only rows that match conditions
is_active = (df['state_chan'] == 'active')
is_pupil = (df['state_chan'] == 'pupil')
full_model = (df['state_sig'] == 'st.pup.beh')
null_model = (df['state_sig'] == 'st.pup0.beh0')
part_beh_model = (df['state_sig'] == 'st.pup0.beh')
part_pup_model = (df['state_sig'] == 'st.pup.beh0')

# creating list of booleans to mask A1, IC, onBF and offBF out of big df
very_sig=(df['r']>4*df['r_se'])
sig_any = df['sig_any']
sig_ubeh = df['sig_ubeh'] & sig_any
sig_upup = df['sig_upup'] & sig_any
sig_both = sig_ubeh & sig_upup
sig_state = df['sig_state'] & sig_any
sig_obeh = df['sig_obeh'] & sig_any
sig_oubeh = sig_ubeh | sig_obeh

A1 = (df['area'] == 'A1') & very_sig
ICC = (df['area'] == 'ICC') & very_sig
ICX = (df['area'] == 'ICX') & very_sig
onBF = (df['onBF'] == True)
offBF = (df['onBF'] == False)
SU = (df['SU'] == True)

fh, axs = plt.subplots(2, 2, figsize=(8,8))
area_str = ['A1','IC']
for i, area in enumerate([A1, ICC|ICX]):
    for j, var in enumerate(['d', 'g']):

        # common.color_list = [color_ns, color_either, color_b, color_p, color_both]
        common.scat_states(
            df, x_model=full_model, y_model=full_model,
            x_beh_state=is_active, y_beh_state=is_pupil,
            area=area,
            sig_list=[~sig_state, sig_state, sig_ubeh, sig_upup, sig_both],
            x_column=var, y_column=var,
            color_list=common.color_list,
            #highlight_cellids={'TAR010c-27-2':'red', 'TAR010c-06-1':'blue'},
            save=False,
            xlabel='task', ylabel='pupil',
            title=var + ' ' + area_str[i],
            xlim=(-1.5,1.5), ylim=(-1.5, 1.5),
            ax=axs[i][j])

fh, axs = plt.subplots(2, 2, figsize=(8,8))
area_str = ['A1','IC']
var_str = ['active','pupil']
for i, area in enumerate([A1, ICC|ICX]):
    for j, var in enumerate([is_active, is_pupil]):

        # common.color_list = [color_ns, color_either, color_b, color_p, color_both]
        common.scat_states(
            df, x_model=full_model, y_model=full_model,
            x_beh_state=var, y_beh_state=var,
            area=area,
            sig_list=[~sig_state, sig_state, sig_ubeh, sig_upup, sig_both],
            x_column='d', y_column='g',
            color_list=common.color_list,
            #highlight_cellids={'TAR010c-27-2':'red', 'TAR010c-06-1':'blue'},
            save=False,
            xlabel='baseline', ylabel='gain',
            title=var_str[j] + ' ' + area_str[i],
            xlim=(-1.5,1.5), ylim=(-1.5, 1.5),
            ax=axs[i][j])

