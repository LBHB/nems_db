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
from nems import get_setting
import nems.plots.api as nplt
import common
import helpers as helper

# set path to dump file
dump_path = get_setting('NEMS_RESULTS_DIR')

# SPECIFY models
dump_results = 'd_pup_beh_sdexp.csv'
model_string = 'st.pup.beh'
p0_model = 'st.pup0.beh'
b0_model = 'st.pup.beh0'
shuf_model = 'st.pup0.beh0' 

# set params for BF characterization and sig. sensory response threshold
octave_cutoff = 0.5
r0_threshold = 0
group_files = True

# import / preprocess model results
A1 = helper.preprocess_sdexp_dump(dump_results,
                                  batch=307,
                                  full_model=model_string,
                                  p0=p0_model,
                                  b0=b0_model,
                                  shuf_model=shuf_model,
                                  r0_threshold=r0_threshold,
                                  octave_cutoff=octave_cutoff,
                                  path=dump_path)
A1['area'] = 'A1'
IC = helper.preprocess_sdexp_dump(dump_results,
                                  batch=309,
                                  full_model=model_string,
                                  p0=p0_model,
                                  b0=b0_model,
                                  shuf_model=shuf_model,
                                  r0_threshold=r0_threshold,
                                  octave_cutoff=octave_cutoff,
                                  path=dump_path)
d_IC_area = pd.read_csv('IC_cells_area.csv', index_col=0)
IC = IC.merge(d_IC_area, on=['cellid'])

df = pd.concat([A1, IC])

if group_files & ('beh' not in model_string):
    area = df['area']
    df = df.groupby(by=['cellid', 'ON_BF']).mean()
    df['area'] = [area.loc[c] if type(area.loc[c]) is str else area.loc[c][0] for c in df.index.get_level_values('cellid')]
    
fh, axs = plt.subplots(3, 3, figsize=(12,12))

# Figure 4A
# A1
common.scat_states_crh(df, x_model='MI_task',
            y_model='MI_task_unique',
            area='A1',
            save=False,
            xlabel='MI task only (pupil ignored)',
            ylabel='MI task unique (pupil regressed out)',
            title='A1',
            xlim=(-0.7,0.7),
            ylim=(-0.7,0.7),
                   ax=axs[0,0])

# ICC
common.scat_states_crh(df, x_model='MI_task',
            y_model='MI_task_unique',
            area='ICC',
            save=False,
            xlabel='MI task only (pupil ignored)',
            ylabel='MI task unique (pupil regressed out)',
            title='ICC',
            xlim=(-0.4,0.6),
            ylim=(-0.4,0.6),
            marker='^',
                   ax=axs[0,1])

# ICX
common.scat_states_crh(df, x_model='MI_task',
            y_model='MI_task_unique',
            area='ICX',
            save=False,
            xlabel='MI task only (pupil ignored)',
            ylabel='MI task unique (pupil regressed out)',
            title='ICC',
            xlim=(-0.4,0.6),
            ylim=(-0.4,0.6),
            marker='o',
                   ax=axs[0,2])

# Figure 4B

# sort based on MI
df_MI_unique_sorted = df.sort_values('MI_task_unique')
df_MI_only_sorted = df.sort_values('MI_task')

x_axis_A1 = np.arange(0, (df.area=='A1').sum())
x_axis_ICC = np.arange(0, (df.area=='ICC').sum())
x_axis_ICX = np.arange(0, (df.area=='ICX').sum())
x_axis_IC = np.arange(0, df.area.isin(['ICC', 'ICX']).sum())

axs[1, 0].bar(x_axis_A1, df_MI_only_sorted.loc[df_MI_only_sorted.area=='A1', 'MI_task'], color = common.color_b, edgecolor = common.color_b)
axs[1, 0].set_ylim((-0.7,0.7))
axs[1, 0].set_xlabel('A1 units')
axs[1, 0].set_ylabel('MI_task only (pupil ignored)')
axs[1, 0].set_title('MI_task only (A1)')
#plt.savefig('MI_task_only_A1.pdf')
nplt.ax_remove_box(axs[1,0])


axs[2, 0].bar(x_axis_A1, df_MI_unique_sorted.loc[df_MI_unique_sorted.area=='A1', 'MI_task_unique'], color = common.color_b,
        edgecolor = common.color_p, linewidth=0.5)
axs[2, 0].set_ylim((-0.7,0.7))
axs[2, 0].set_xlabel('A1 units')
axs[2, 0].set_ylabel('MI_task unique (pupil regressed out)')
axs[2, 0].set_title('MI_task unique (A1)')
#plt.savefig('MI_task_unique_A1.pdf')
nplt.ax_remove_box(axs[2,0])


axs[1, 1].bar(x_axis_IC, df_MI_only_sorted.loc[df_MI_only_sorted.area.isin(['ICC', 'ICX']), 'MI_task'], 
                                color = common.color_b, edgecolor = common.color_b)
axs[1, 1].set_ylim((-0.7,0.7))
axs[1, 1].set_xlabel('IC units')
axs[1, 1].set_ylabel('MI_task only (pupil ignored)')
axs[1, 1].set_title('MI_task only (IC)')
#plt.savefig('MI_task_only_IC.pdf')
nplt.ax_remove_box(axs[1, 1])


axs[2, 1].bar(x_axis_IC, df_MI_unique_sorted.loc[df_MI_unique_sorted.area.isin(['ICC', 'ICX']), 'MI_task_unique'], 
                                color=common.color_b, edgecolor=common.color_p, linewidth=0.5)
axs[2, 1].set_ylim((-0.7,0.7))
axs[2, 1].set_xlabel('IC units')
axs[2, 1].set_ylabel('MI_task unique (pupil regressed out)')
axs[2, 1].set_title('MI_task unique (IC)')
#plt.savefig('MI_task_unique_IC.pdf')
nplt.ax_remove_box(axs[2,1])


# ===================================== Stats stuff ==============================================
# To quantify differences in modulation without the confounding element of sign let's do
# (MItask only - MItask unique) * sign((MItask only+MItask unique)/2)
# and calculate how much that differs from zero

# A1
sign_A1 = np.sign((df.loc[df.area=='A1', 'MI_task_unique'] +
                           df.loc[df.area=='A1','MI_task'])/2)

signed_diff_A1 = (df.loc[df.area=='A1', 'MI_task']
                  -df.loc[df.area=='A1', 'MI_task_unique']) * sign_A1

signed_only_A1 = df.loc[df.area=='A1', 'MI_task'] * sign_A1

signed_unique_A1 = df.loc[df.area=='A1', 'MI_task_unique'] * sign_A1

diff_A1 = (df.loc[df.area=='A1', 'MI_task']
                  -df.loc[df.area=='A1', 'MI_task_unique'])

diff_A1_sig_state = (df.loc[(df.area=='A1') & df['sig_state'], 'MI_task']
                  -df.loc[(df.area=='A1') & df['sig_state'], 'MI_task_unique'])

# A1 onBF
signed_diff_A1_onBF = (df.loc[(df.area=='A1') & df['ON_BF'],
                  'MI_task']-df.loc[(df.area=='A1') & df['ON_BF'],
                                         'MI_task_unique']) * np.sign((df.loc[(df.area=='A1') & df['ON_BF'],
                                                                          'MI_task_unique']+df.loc[(df.area=='A1') & df['ON_BF'],
                                                                                                 'MI_task'])/2)

# A1 offBF
signed_diff_A1_offBF = (df.loc[(df.area=='A1') & df['OFF_BF'],
                  'MI_task']-df.loc[(df.area=='A1') & df['OFF_BF'],
                                         'MI_task_unique']) * np.sign((df.loc[(df.area=='A1') & df['OFF_BF'],
                                                                          'MI_task_unique']+df.loc[(df.area=='A1') & df['OFF_BF'],
                                                                                                 'MI_task'])/2)

# A1 SU
signed_diff_A1_SU = (df.loc[(df.area=='A1') & df['SU'],
                  'MI_task']-df.loc[(df.area=='A1') & df['SU'],
                                         'MI_task_unique']) * np.sign((df.loc[(df.area=='A1') & df['SU'],
                                                                          'MI_task_unique']+df.loc[(df.area=='A1') & df['SU'],
                                                                                                 'MI_task'])/2)

# A1 MU
signed_diff_A1_MU = (df.loc[(df.area=='A1') & ~df['SU'],
                  'MI_task']-df.loc[(df.area=='A1') & ~df['SU'],
                                         'MI_task_unique']) * np.sign((df.loc[(df.area=='A1') & ~df['SU'],
                                                                          'MI_task_unique']+df.loc[(df.area=='A1') & ~df['SU'],
                                                                                                 'MI_task'])/2)

######################
# IC all

sign_IC = np.sign((df.loc[df.area.isin(['ICC', 'ICX']), 'MI_task_unique']+
                           df.loc[df.area.isin(['ICC', 'ICX']),'MI_task'])/2)

signed_diff_IC = (df.loc[df.area.isin(['ICC', 'ICX']), 'MI_task']
                  -df.loc[df.area.isin(['ICC', 'ICX']), 'MI_task_unique']) * sign_IC


signed_only_IC = df.loc[df.area.isin(['ICC', 'ICX']), 'MI_task'] * sign_IC

signed_unique_IC = df.loc[df.area.isin(['ICC', 'ICX']), 'MI_task_unique'] * sign_IC

diff_IC = (df.loc[df.area.isin(['ICC', 'ICX']), 'MI_task']
                  -df.loc[df.area.isin(['ICC', 'ICX']), 'MI_task_unique'])

diff_IC_sig_state = (df.loc[df.area.isin(['ICC', 'ICX']) & df['sig_state'], 'MI_task']
                  -df.loc[df.area.isin(['ICC', 'ICX']) & df['sig_state'], 'MI_task_unique'])



# IC onBF
signed_diff_IC_onBF = (df.loc[df.area.isin(['ICC', 'ICX']) & df['ON_BF'],
                  'MI_task']-df.loc[df.area.isin(['ICC', 'ICX']) & df['ON_BF'],
                                         'MI_task_unique']) * np.sign((df.loc[df.area.isin(['ICC', 'ICX']) & df['ON_BF'],
                                                                          'MI_task_unique']+df.loc[df.area.isin(['ICC', 'ICX']) & df['ON_BF'],
                                                                                                 'MI_task'])/2)

# IC onBF
signed_diff_IC_offBF = (df.loc[df.area.isin(['ICC', 'ICX']) & df['OFF_BF'],
                  'MI_task']-df.loc[df.area.isin(['ICC', 'ICX']) & df['OFF_BF'],
                                         'MI_task_unique']) * np.sign((df.loc[df.area.isin(['ICC', 'ICX']) & df['OFF_BF'],
                                                                          'MI_task_unique']+df.loc[df.area.isin(['ICC', 'ICX']) & df['OFF_BF'],
                                                                                                 'MI_task'])/2)


# ICC
signed_diff_ICC = (df.loc[(df.area=='ICC'),
                  'MI_task']-df.loc[(df.area=='ICC'),
                                         'MI_task_unique']) * np.sign((df.loc[(df.area=='ICC'),
                                                                          'MI_task_unique']+df.loc[(df.area=='ICC'),
                                                                                                 'MI_task'])/2)

# ICX
signed_diff_ICX = (df.loc[(df.area=='ICX'),
                  'MI_task']-df.loc[(df.area=='ICX'),
                                         'MI_task_unique']) * np.sign((df.loc[(df.area=='ICX'),
                                                                          'MI_task_unique']+df.loc[(df.area=='ICX'),
                                                                                                 'MI_task'])/2)


# IC SU
signed_diff_IC_SU = (df.loc[df.area.isin(['ICC', 'ICX'])  & df['SU'],
                  'MI_task']-df.loc[df.area.isin(['ICC', 'ICX'])  & df['SU'],
                                         'MI_task_unique']) * np.sign((df.loc[df.area.isin(['ICC', 'ICX'])  & df['SU'],
                                                                          'MI_task_unique']+df.loc[df.area.isin(['ICC', 'ICX'])  & df['SU'],
                                                                                                 'MI_task'])/2)

# IC MU
signed_diff_IC_MU = (df.loc[df.area.isin(['ICC', 'ICX'])  & ~df['SU'],
                  'MI_task']-df.loc[df.area.isin(['ICC', 'ICX'])  & ~df['SU'],
                                         'MI_task_unique']) * np.sign((df.loc[df.area.isin(['ICC', 'ICX'])  & ~df['SU'],
                                                                          'MI_task_unique']+df.loc[df.area.isin(['ICC', 'ICX'])  & ~df['SU'],
                                                                                                 'MI_task'])/2)

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


