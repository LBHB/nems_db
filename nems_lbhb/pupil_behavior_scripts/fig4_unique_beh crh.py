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
import nems_lbhb.pupil_behavior_scripts.common as common
import nems_lbhb.pupil_behavior_scripts.helpers as helper
from nems_lbhb.analysis.statistics import get_bootstrapped_sample, get_direct_prob

# set path to dump file
dump_path = get_setting('NEMS_RESULTS_DIR')
helper_path = os.path.dirname(helper.__file__)

save_path = os.path.join(os.path.expanduser('~'),'docs/current/pupil_behavior/eps')
save_fig = False

# SPECIFY models
USE_AFL=True
if USE_AFL:
    dump_results = 'd_pup_afl_sdexp.csv'
    model_string = 'st.pup.afl'
    p0_model = 'st.pup0.afl'
    b0_model = 'st.pup.afl0'
    shuf_model = 'st.pup0.afl0'
else:
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
    df['area'] = [area.loc[c] if type(area.loc[c]) is str else area.loc[c].iloc[0] for c in df.index.get_level_values('cellid')]
    df=df.reset_index()
    df.index = df.cellid
    
fh, axs = plt.subplots(2, 3, figsize=(7.5,5))

# Figure 4A
# A1
rr=(-0.55, 0.55)
common.scat_states_crh(df, x_model='MI_task',
            y_model='MI_task_unique',
            area='A1',
            save=False,
            xlabel='MI task only (pupil ignored)',
            ylabel='MI task unique (pupil regressed out)',
            title='A1',
            xlim=rr,
            ylim=rr,
            ax=axs[0,0], 
            bootstats=True)

# ICC
common.scat_states_crh(df, x_model='MI_task',
            y_model='MI_task_unique',
            area='ICC',
            save=False,
            xlabel='MI task only (pupil ignored)',
            ylabel='MI task unique (pupil regressed out)',
            title='ICC',
            xlim=rr,
            ylim=rr,
            marker='^',
            ax=axs[0,1],
            bootstats=True)

# ICX
common.scat_states_crh(df, x_model='MI_task',
            y_model='MI_task_unique',
            area='ICX',
            save=False,
            xlabel='MI task only (pupil ignored)',
            ylabel='MI task unique (pupil regressed out)',
            title='ICC+ICX',
            xlim=rr,
            ylim=rr,
            marker='o',
            ax=axs[0,1], 
            bootstats=True)

# Figure 4B

# sort based on MI
df_MI_unique_sorted = df.sort_values('MI_task_unique')
df_MI_only_sorted = df.sort_values('MI_task')

x_axis_A1 = np.arange(0, (df.area=='A1').sum())
x_axis_ICC = np.arange(0, (df.area=='ICC').sum())
x_axis_ICX = np.arange(0, (df.area=='ICX').sum())
x_axis_IC = np.arange(0, df.area.isin(['ICC', 'ICX']).sum())

axs[1, 0].bar(x_axis_A1, df_MI_only_sorted.loc[df_MI_only_sorted.area=='A1', 'MI_task'], color = common.color_b, edgecolor = common.color_b)
axs[1, 0].bar(x_axis_A1, df_MI_unique_sorted.loc[df_MI_unique_sorted.area=='A1', 'MI_task_unique'], color = common.color_b,
        edgecolor = common.color_p, linewidth=0.5)
axs[1, 0].set_ylim((-0.7,0.7))
axs[1, 0].set_xlabel('A1 units')
axs[1, 0].set_ylabel('MI_task only/unique')
#plt.savefig('MI_task_only_A1.pdf')
nplt.ax_remove_box(axs[1,0])


axs[1, 1].bar(x_axis_IC, df_MI_only_sorted.loc[df_MI_only_sorted.area.isin(['ICC', 'ICX']), 'MI_task'],
                                color = common.color_b, edgecolor = common.color_b)
axs[1, 1].bar(x_axis_IC, df_MI_unique_sorted.loc[df_MI_unique_sorted.area.isin(['ICC', 'ICX']), 'MI_task_unique'],
                                color=common.color_b, edgecolor=common.color_p, linewidth=0.5)
axs[1, 1].set_ylim((-0.7,0.7))
axs[1, 1].set_xlabel('IC units')
axs[1, 1].set_ylabel('MI_task only/unique')
#plt.savefig('MI_task_unique_IC.pdf')
nplt.ax_remove_box(axs[1,1])

axs[0, 2].bar(x_axis_ICC, df_MI_only_sorted.loc[df_MI_only_sorted.area.isin(['ICC']), 'MI_task'],
                                color = common.color_b, edgecolor = common.color_b)
axs[0, 2].bar(x_axis_ICC, df_MI_unique_sorted.loc[df_MI_unique_sorted.area.isin(['ICC']), 'MI_task_unique'],
                                color=common.color_b, edgecolor=common.color_p, linewidth=0.5)
axs[0, 2].set_ylim((-0.7,0.7))
axs[0, 2].set_xlabel('ICC units')
axs[0, 2].set_ylabel('MI_task only/unique')
nplt.ax_remove_box(axs[1,1])

axs[1, 2].bar(x_axis_ICX, df_MI_only_sorted.loc[df_MI_only_sorted.area.isin(['ICX']), 'MI_task'],
                                color = common.color_b, edgecolor = common.color_b)
axs[1, 2].bar(x_axis_ICX, df_MI_unique_sorted.loc[df_MI_unique_sorted.area.isin(['ICX']), 'MI_task_unique'],
                                color=common.color_b, edgecolor=common.color_p, linewidth=0.5)
axs[1, 2].set_ylim((-0.7,0.7))
axs[1, 2].set_xlabel('ICX units')
axs[1, 2].set_ylabel('MI_task only/unique')
nplt.ax_remove_box(axs[1,1])



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
unique_A1 = df.loc[(df.area=='A1') & df['sig_state'], 'MI_task_unique']

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
sign_ICC = np.sign((df.loc[df.area.isin(['ICC']), 'MI_task_unique']+
                           df.loc[df.area.isin(['ICC']),'MI_task'])/2)
sign_ICX = np.sign((df.loc[df.area.isin(['ICX']), 'MI_task_unique']+
                           df.loc[df.area.isin(['ICX']),'MI_task'])/2)

signed_diff_IC = (df.loc[df.area.isin(['ICC', 'ICX']), 'MI_task']
                  -df.loc[df.area.isin(['ICC', 'ICX']), 'MI_task_unique']) * sign_IC


signed_only_IC = df.loc[df.area.isin(['ICC', 'ICX']), 'MI_task'] * sign_IC
signed_unique_IC = df.loc[df.area.isin(['ICC', 'ICX']), 'MI_task_unique'] * sign_IC
signed_only_ICC = df.loc[df.area.isin(['ICC']), 'MI_task'] * sign_ICC
signed_unique_ICC = df.loc[df.area.isin(['ICC']), 'MI_task_unique'] * sign_ICC
signed_only_ICX = df.loc[df.area.isin(['ICX']), 'MI_task'] * sign_ICX
signed_unique_ICX = df.loc[df.area.isin(['ICX']), 'MI_task_unique'] * sign_ICX
unique_IC = df.loc[df.area.isin(['ICC', 'ICX']) & df['sig_state'], 'MI_task_unique']
unique_ICC = df.loc[df.area.isin(['ICC']) & df['sig_state'], 'MI_task_unique']
unique_ICX = df.loc[df.area.isin(['ICX']) & df['sig_state'], 'MI_task_unique']

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
signed_diff_IC_MU = (df.loc[df.area.isin(['ICC', 'ICX'])  & ~df['SU'], 'MI_task']
                     -df.loc[df.area.isin(['ICC', 'ICX'])  & ~df['SU'],
                             'MI_task_unique']) * np.sign((df.loc[df.area.isin(['ICC', 'ICX'])  & ~df['SU'],
                             'MI_task_unique']+df.loc[df.area.isin(['ICC', 'ICX'])  & ~df['SU'], 'MI_task'])/2)


ratio_A1 = 1 - (signed_unique_A1.mean() / signed_only_A1.mean())
ratio_IC = 1 - (signed_unique_IC.mean() / signed_only_IC.mean())
ratio_ICC = 1 - (signed_unique_ICC.mean() / signed_only_ICC.mean())
ratio_ICX = 1 - (signed_unique_ICX.mean() / signed_only_ICX.mean())

stat, p = sci.wilcoxon(signed_diff_A1)
print(f'A1 mean MI task only, unique: {signed_only_A1.median():.3f}, {signed_unique_A1.median():.3f}')
print(f'  signed delta task unique: stat={stat:.3f}, p={p:.4e}')
print(f'  Ratio: {ratio_A1:.3f}')
stat, p = sci.wilcoxon(signed_diff_IC)
print(f'IC mean MI task only, unique: {signed_only_IC.median():.3f}, {signed_unique_IC.median():.3f}')
print(f'  signed delta task unique: stat={stat:.3f}, p={p:.4e}')
print(f'  Ratio: {ratio_IC:.3f}')

# run the above two comparisons with bootstrapped test
np.random.seed(123)
signed_diff_A1_wSite = pd.DataFrame(signed_diff_A1, columns=['signed_diff'])
signed_diff_A1_wSite['siteid'] = [c[:7] for c in signed_diff_A1_wSite.index]
signed_diff_IC_wSite = pd.DataFrame(signed_diff_IC, columns=['signed_diff'])
signed_diff_IC_wSite['siteid'] = [c[:7] for c in signed_diff_IC_wSite.index]

a1 = {s: signed_diff_A1_wSite.loc[(signed_diff_A1_wSite.siteid==s), 'signed_diff'].values for s in signed_diff_A1_wSite.siteid.unique()}
a1 = get_bootstrapped_sample(a1, nboot=100)
p = get_direct_prob(a1, np.zeros(a1.shape[0]))[0]
print(f"\n A1 task only vs. task unique bootstrapped prob: {p}\n")
ic = {s: signed_diff_IC_wSite.loc[(signed_diff_IC_wSite.siteid==s), 'signed_diff'].values for s in signed_diff_IC_wSite.siteid.unique()}
ic = get_bootstrapped_sample(ic, nboot=100)
p = get_direct_prob(ic, np.zeros(ic.shape[0]))[0]
print(f"\n IC task only vs. task unique bootstrapped prob: {p}\n")

# split up ICC / ICX
stat, p = sci.wilcoxon(signed_diff_ICC)
print(f' ICC signed delta task unique: stat={stat:.3f}, p={p:.4e}')
print(f'  Ratio: {ratio_ICC:.3f}')
stat, p = sci.wilcoxon(signed_diff_ICX)
print(f' ICX signed delta task unique: stat={stat:.3f}, p={p:.4e}')
print(f'  Ratio: {ratio_ICX:.3f}')

stat, p = sci.ranksums(signed_unique_A1 / signed_only_A1,
                       signed_unique_IC / signed_only_IC)
print(f'A1 vs. IC rank sum ratio: stat={stat:.3f}, p={p:.4e}')
stat, p = sci.ranksums(signed_unique_ICC / signed_only_ICC,
                       signed_unique_ICX / signed_only_ICX)
print(f'ICC vs. ICX rank sum ratio: stat={stat:.3f}, p={p:.4e}')

# ICC vs. ICX comparison with bootstrap
signed_diff_ICC_wSite = pd.DataFrame(signed_diff_ICC, columns=['signed_diff'])
signed_diff_ICC_wSite['siteid'] = [c[:7] for c in signed_diff_ICC_wSite.index]
signed_diff_ICX_wSite = pd.DataFrame(signed_diff_ICX, columns=['signed_diff'])
signed_diff_ICX_wSite['siteid'] = [c[:7] for c in signed_diff_ICX_wSite.index]

icx = {s: signed_diff_ICX_wSite.loc[(signed_diff_ICX_wSite.siteid==s), 'signed_diff'].values for s in signed_diff_ICX_wSite.siteid.unique()}
icx = get_bootstrapped_sample(icx, nboot=100)
icc = {s: signed_diff_ICC_wSite.loc[(signed_diff_ICC_wSite.siteid==s), 'signed_diff'].values for s in signed_diff_ICC_wSite.siteid.unique()}
icc = get_bootstrapped_sample(icc, nboot=100)
p = get_direct_prob(icc, icx)[0]
print(f"\n ICX vs. ICC bootstrapped prob: {p}\n")

stat, p = sci.wilcoxon(unique_A1)
print(f'A1 u_mod_beh: n+={np.sum(unique_A1>0)}/{len(unique_A1)} med={np.median(unique_A1):.3f} Wilcoxon stat={stat:.3f}, p={p:.4e}')
stat, p = sci.wilcoxon(unique_IC)
print(f'IC u_mod_beh: n+={np.sum(unique_IC>0)}/{len(unique_IC)} med={np.median(unique_IC):.3f} Wilcoxon stat={stat:.3f}, p={p:.4e}')
stat, p = sci.wilcoxon(unique_ICC)
print(f'ICC u_mod_beh: n+={np.sum(unique_ICC>0)}/{len(unique_ICC)} med={np.median(unique_ICC):.3f} Wilcoxon stat={stat:.3f}, p={p:.4e}')
stat, p = sci.wilcoxon(unique_ICX)
print(f'ICX u_mod_beh: n+={np.sum(unique_ICX>0)}/{len(unique_ICX)} med={np.median(unique_ICX):.3f} Wilcoxon stat={stat:.3f}, p={p:.4e}')

stat, p = sci.ranksums(unique_A1,unique_IC)
print(f'A1 vs. IC ranksum: {stat:.3f}, p={p:.4e}')


"""
print(signed_diff_A1.mean())
print(signed_diff_IC.mean())

print(signed_only_A1.mean())
print(signed_unique_A1.mean())

print(signed_only_IC.mean())
print(signed_unique_IC.mean())


print(ratio_A1)
print(ratio_IC)

print(diff_A1.median())
print(diff_IC.median())

print(diff_A1_sig_state.median())
print(diff_IC_sig_state.median())
"""
if save_fig:
    fh.savefig(os.path.join(save_path, 'fig4_unique_beh.pdf'))

