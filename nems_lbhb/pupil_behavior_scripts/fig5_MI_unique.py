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

#df = pd.read_csv('pup_beh_processed.csv')
df = pd.read_csv('pup_beh_processed'+basemodel+'.csv')

# creating subdf with only rows that match conditions
is_active = (df['state_chan'] == 'active')
is_pupil = (df['state_chan'] == 'pupil')
full_model = (df['state_sig'] == 'st.pup.beh')
null_model = (df['state_sig'] == 'st.pup0.beh0')
part_beh_model = (df['state_sig'] == 'st.pup0.beh')
part_pup_model = (df['state_sig'] == 'st.pup.beh0')

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
sig_any = df['sig_any']
sig_ubeh = df['sig_ubeh'] & sig_any
sig_upup = df['sig_upup'] & sig_any
sig_both = sig_ubeh & sig_upup
sig_state = df['sig_state'] & sig_any
sig_obeh = df['sig_obeh'] & sig_any
sig_oubeh = sig_ubeh | sig_obeh

fh, axs = plt.subplots(1, 2, figsize=(8,4))

# common.color_list = [color_ns, color_either, color_b, color_p, color_both]
common.scat_states(df, x_model=full_model,
            y_model=full_model,
            x_beh_state=is_active,
            area=A1,
            sig_list=[~sig_state, sig_state, sig_ubeh, sig_upup,sig_both],
            x_column='MIpup_unique',
            y_column='MIbeh_unique',
            color_list=common.color_list,
            #highlight_cellids={'TAR010c-27-2':'red', 'TAR010c-06-1':'blue'},
            save=False,
            xlabel='MI_AP task unique',
            ylabel='MI_LS pupil unique',
            title='A1',
            xlim=(-0.6,0.6),
            ylim=(-0.6,0.6),
            ax=axs[0])

# IC
common.scat_states(df, x_model=full_model,
            y_model=full_model,
            x_beh_state=is_active,
            area=(ICX | ICC),
            sig_list=[~sig_state, sig_state, sig_ubeh, sig_upup,sig_both],
            x_column='MIpup_unique',
            y_column='MIbeh_unique',
            color_list=common.color_list,
            #highlight_cellids={'TAR010c-27-2':'red', 'TAR010c-06-1':'blue'},
            save=False,
            xlabel='MI_AP task unique',
            ylabel='MI_LS pupil unique',
            title='ICC & ICX',
            xlim=(-0.3,0.4),
            ylim=(-0.3,0.4),
            ax=axs[1])