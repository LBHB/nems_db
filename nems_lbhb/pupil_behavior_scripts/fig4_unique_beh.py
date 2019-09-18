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
import nems.plots.api as nplt
import common


# SPECIFY pup+beh models
state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
basemodel = "-ref-psthfr.s_sdexp.S"

#df = pd.read_csv('pup_beh_processed.csv')
df = pd.read_csv('pup_beh_processed'+basemodel+'.csv')

xsubset = df.cellid.str.startswith('AMT018') | df.cellid.str.startswith('AMT020')
#xsubset = df.cellid.str.startswith('AMT020')
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

# creating subdf with only rows that match conditions
is_active = (df['state_chan'] == 'active')
is_pupil = (df['state_chan'] == 'pupil')
full_model = (df['state_sig'] == 'st.pup.beh')
null_model = (df['state_sig'] == 'st.pup0.beh0')
part_beh_model = (df['state_sig'] == 'st.pup0.beh')
part_pup_model = (df['state_sig'] == 'st.pup.beh0')

fh, axs = plt.subplots(3, 3, figsize=(12,12))


# Figure 4A
# A1
common.scat_states(df, x_model=full_model,
            y_model=full_model,
            x_beh_state=is_active,
            area=A1,
            sig_list=[~sig_state, sig_state, sig_ubeh, sig_upup,sig_both],
            x_column='MIbeh_only',
            y_column='MIbeh_unique',
            color_list=common.color_list,
            #highlight_cellids={'TAR010c-27-2':'red', 'TAR010c-06-1':'blue'},
            save=False,
            xlabel='MI task only (pupil ignored)',
            ylabel='MI task unique (pupil regressed out)',
            title='A1',
            xlim=(-0.7,0.7),
            ylim=(-0.7,0.7),
                   ax=axs[0,0])

# ICC
common.scat_states(df, x_model=full_model,
            y_model=full_model,
            x_beh_state=is_active,
            area=ICC,
            sig_list=[~sig_state, sig_state, sig_ubeh, sig_upup,sig_both],
            x_column='MIbeh_only',
            y_column='MIbeh_unique',
            color_list=common.color_list,
            #highlight_cellids={'TAR010c-27-2':'red', 'TAR010c-06-1':'blue'},
            save=False,
            xlabel='MI task only (pupil ignored)',
            ylabel='MI task unique (pupil regressed out)',
            title='ICC',
            xlim=(-0.4,0.6),
            ylim=(-0.4,0.6),
            marker='^',
                   ax=axs[0,1])

# ICX
common.scat_states(df, x_model=full_model,
            y_model=full_model,
            x_beh_state=is_active,
            area=ICX,
            sig_list=[~sig_state, sig_state, sig_ubeh, sig_upup,sig_both],
            x_column='MIbeh_only',
            y_column='MIbeh_unique',
            color_list=common.color_list,
            #highlight_cellids={'TAR010c-27-2':'red', 'TAR010c-06-1':'blue'},
            save=False,
            xlabel='MI task only (pupil ignored)',
            ylabel='MI task unique (pupil regressed out)',
            title='ICX',
            xlim=(-0.4,0.6),
            ylim=(-0.4,0.6),
                   ax=axs[0,2])


# Figure 4B

# sort based on MI
df_MI_unique_sorted = df.sort_values('MIbeh_unique')
df_MI_only_sorted = df.sort_values('MIbeh_only')

x_axis_A1 = list(range(len(df[full_model & is_active & A1])))
x_axis_ICC = list(range(len(df[full_model & is_active & ICC])))
x_axis_ICX = list(range(len(df[full_model & is_active & ICX])))
x_axis_IC = list(range(len(df[full_model & is_active & (ICC | ICX)])))

plt.sca(axs[1,0])
plt.bar(x_axis_A1, df_MI_only_sorted.loc[full_model & is_active & A1, 'MIbeh_only'], color = common.color_b, edgecolor = common.color_b)
plt.ylim((-0.7,0.7))
plt.xlabel('A1 units')
plt.ylabel('MI_task only (pupil ignored)')
plt.title('MI_task only (A1)')
#plt.savefig('MI_task_only_A1.pdf')
nplt.ax_remove_box(axs[1,0])

plt.sca(axs[2,0])
plt.bar(x_axis_A1, df_MI_unique_sorted.loc[full_model & is_active & A1, 'MIbeh_unique'], color = common.color_b,
        edgecolor = common.color_p, linewidth=0.5)
plt.ylim((-0.7,0.7))
plt.xlabel('A1 units')
plt.ylabel('MI_task unique (pupil regressed out)')
plt.title('MI_task unique (A1)')
#plt.savefig('MI_task_unique_A1.pdf')
nplt.ax_remove_box(axs[2,0])


plt.sca(axs[1,1])
plt.bar(x_axis_IC, df_MI_only_sorted.loc[full_model & is_active & (ICC | ICX), 'MIbeh_only'], color = common.color_b, edgecolor = common.color_b)
plt.ylim((-0.7,0.7))
plt.xlabel('IC units')
plt.ylabel('MI_task only (pupil ignored)')
plt.title('MI_task only (IC)')
#plt.savefig('MI_task_only_IC.pdf')
nplt.ax_remove_box(axs[1,1])


plt.sca(axs[2,1])
plt.bar(x_axis_IC, df_MI_unique_sorted.loc[full_model & is_active & (ICC | ICX), 'MIbeh_unique'], color = common.color_b,
        edgecolor = common.color_p, linewidth=0.5)
plt.ylim((-0.7,0.7))
plt.xlabel('IC units')
plt.ylabel('MI_task unique (pupil regressed out)')
plt.title('MI_task unique (IC)')
#plt.savefig('MI_task_unique_IC.pdf')
nplt.ax_remove_box(axs[2,1])

# To quantify differences in modulation without the confounding element of sign let's do
# (MItask only - MItask unique) * sign((MItask only+MItask unique)/2)
# and calculate how much that differs from zero

# A1

sign_A1 = np.sign((df.loc[full_model & is_active & A1, 'MIbeh_unique']+
                           df.loc[full_model & is_active & A1,'MIbeh_only'])/2)

signed_diff_A1 = (df.loc[full_model & is_active & A1, 'MIbeh_only']
                  -df.loc[full_model & is_active & A1, 'MIbeh_unique']) * sign_A1


signed_only_A1 = df.loc[full_model & is_active & A1, 'MIbeh_only'] * sign_A1

signed_unique_A1 = df.loc[full_model & is_active & A1, 'MIbeh_unique'] * sign_A1

diff_A1 = (df.loc[full_model & is_active & A1, 'MIbeh_only']
                  -df.loc[full_model & is_active & A1, 'MIbeh_unique'])

diff_A1_sig_state = (df.loc[full_model & is_active & A1 & sig_state, 'MIbeh_only']
                  -df.loc[full_model & is_active & A1 & sig_state, 'MIbeh_unique'])

# A1 onBF
signed_diff_A1_onBF = (df.loc[full_model & is_active & A1 & onBF,
                  'MIbeh_only']-df.loc[full_model & is_active & A1 & onBF,
                                         'MIbeh_unique']) * np.sign((df.loc[full_model & is_active & A1 & onBF,
                                                                          'MIbeh_unique']+df.loc[full_model & is_active & A1 & onBF,
                                                                                                 'MIbeh_only'])/2)

# A1 offBF
signed_diff_A1_offBF = (df.loc[full_model & is_active & A1 & offBF,
                  'MIbeh_only']-df.loc[full_model & is_active & A1 & offBF,
                                         'MIbeh_unique']) * np.sign((df.loc[full_model & is_active & A1 & offBF,
                                                                          'MIbeh_unique']+df.loc[full_model & is_active & A1 & offBF,
                                                                                                 'MIbeh_only'])/2)

# A1 SU
signed_diff_A1_SU = (df.loc[full_model & is_active & A1 & SU,
                  'MIbeh_only']-df.loc[full_model & is_active & A1 & SU,
                                         'MIbeh_unique']) * np.sign((df.loc[full_model & is_active & A1 & SU,
                                                                          'MIbeh_unique']+df.loc[full_model & is_active & A1 & SU,
                                                                                                 'MIbeh_only'])/2)

# A1 MU
signed_diff_A1_MU = (df.loc[full_model & is_active & A1 & ~SU,
                  'MIbeh_only']-df.loc[full_model & is_active & A1 & ~SU,
                                         'MIbeh_unique']) * np.sign((df.loc[full_model & is_active & A1 & ~SU,
                                                                          'MIbeh_unique']+df.loc[full_model & is_active & A1 & ~SU,
                                                                                                 'MIbeh_only'])/2)

######################
# IC all

sign_IC = np.sign((df.loc[full_model & is_active & (ICC | ICX), 'MIbeh_unique']+
                           df.loc[full_model & is_active & (ICC | ICX),'MIbeh_only'])/2)

signed_diff_IC = (df.loc[full_model & is_active & (ICC | ICX), 'MIbeh_only']
                  -df.loc[full_model & is_active & (ICC | ICX), 'MIbeh_unique']) * sign_IC


signed_only_IC = df.loc[full_model & is_active & (ICC | ICX), 'MIbeh_only'] * sign_IC

signed_unique_IC = df.loc[full_model & is_active & (ICC | ICX), 'MIbeh_unique'] * sign_IC

diff_IC = (df.loc[full_model & is_active & (ICC | ICX), 'MIbeh_only']
                  -df.loc[full_model & is_active & (ICC | ICX), 'MIbeh_unique'])

diff_IC_sig_state = (df.loc[full_model & is_active & (ICC | ICX) & sig_state, 'MIbeh_only']
                  -df.loc[full_model & is_active & (ICC | ICX) & sig_state, 'MIbeh_unique'])



# IC onBF
signed_diff_IC_onBF = (df.loc[full_model & is_active & (ICC | ICX) & onBF,
                  'MIbeh_only']-df.loc[full_model & is_active & (ICC | ICX) & onBF,
                                         'MIbeh_unique']) * np.sign((df.loc[full_model & is_active & (ICC | ICX) & onBF,
                                                                          'MIbeh_unique']+df.loc[full_model & is_active & (ICC | ICX) & onBF,
                                                                                                 'MIbeh_only'])/2)

# IC onBF
signed_diff_IC_offBF = (df.loc[full_model & is_active & (ICC | ICX) & offBF,
                  'MIbeh_only']-df.loc[full_model & is_active & (ICC | ICX) & offBF,
                                         'MIbeh_unique']) * np.sign((df.loc[full_model & is_active & (ICC | ICX) & offBF,
                                                                          'MIbeh_unique']+df.loc[full_model & is_active & (ICC | ICX) & offBF,
                                                                                                 'MIbeh_only'])/2)


# ICC
signed_diff_ICC = (df.loc[full_model & is_active & ICC,
                  'MIbeh_only']-df.loc[full_model & is_active & ICC,
                                         'MIbeh_unique']) * np.sign((df.loc[full_model & is_active & ICC,
                                                                          'MIbeh_unique']+df.loc[full_model & is_active & ICC,
                                                                                                 'MIbeh_only'])/2)

# ICX
signed_diff_ICX = (df.loc[full_model & is_active & ICX,
                  'MIbeh_only']-df.loc[full_model & is_active & ICX,
                                         'MIbeh_unique']) * np.sign((df.loc[full_model & is_active & ICX,
                                                                          'MIbeh_unique']+df.loc[full_model & is_active & ICX,
                                                                                                 'MIbeh_only'])/2)


# A1 SU
signed_diff_IC_SU = (df.loc[full_model & is_active & (ICC | ICX)  & SU,
                  'MIbeh_only']-df.loc[full_model & is_active & (ICC | ICX)  & SU,
                                         'MIbeh_unique']) * np.sign((df.loc[full_model & is_active & (ICC | ICX)  & SU,
                                                                          'MIbeh_unique']+df.loc[full_model & is_active & (ICC | ICX)  & SU,
                                                                                                 'MIbeh_only'])/2)

# A1 MU
signed_diff_IC_MU = (df.loc[full_model & is_active & (ICC | ICX)  & ~SU,
                  'MIbeh_only']-df.loc[full_model & is_active & (ICC | ICX)  & ~SU,
                                         'MIbeh_unique']) * np.sign((df.loc[full_model & is_active & (ICC | ICX)  & ~SU,
                                                                          'MIbeh_unique']+df.loc[full_model & is_active & (ICC | ICX)  & ~SU,
                                                                                                 'MIbeh_only'])/2)

print(signed_diff_A1.mean())
print(signed_diff_IC.mean())

print(signed_only_A1.mean())
print(signed_unique_A1.mean())

print(signed_only_IC.mean())
print(signed_unique_IC.mean())

ratio_A1 = (signed_unique_A1 / signed_only_A1).median()
ratio_IC = (signed_unique_IC / signed_only_IC).median()

print(ratio_A1)
print(ratio_IC)

print(diff_A1.median())
print(diff_IC.median())

print(diff_A1_sig_state.median())
print(diff_IC_sig_state.median())


