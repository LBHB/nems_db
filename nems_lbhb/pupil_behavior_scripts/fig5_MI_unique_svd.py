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
from nems_lbhb.analysis.statistics import get_direct_prob, get_bootstrapped_sample

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
d_IC_area = pd.read_csv(os.path.join(helper_path,'IC_cells_area.csv'), index_col=0)
IC = IC.merge(d_IC_area, on=['cellid'])

df = pd.concat([A1, IC])

if group_files & ('beh' not in model_string):
    area = df['area']
    df = df.groupby(by=['cellid', 'ON_BF']).mean()
    df['area'] = [area.loc[c] if type(area.loc[c]) is str else area.loc[c][0] for c in df.index.get_level_values('cellid')]

# generate the plot
fh, axs = plt.subplots(1, 2, figsize=(5,2.5))

# common.color_list = [color_ns, color_either, color_b, color_p, color_both]
common.scat_states_crh(df, x_model='MI_pupil_unique',
            y_model='MI_task_unique',
            area='A1',
            save=False,
            xlabel='MI pupil unique',
            ylabel='MI task unique',
            title='A1',
            xlim=(-0.45,0.45),
            ylim=(-0.45,0.45),
            ax=axs[0],
            bootstats=True)

common.scat_states_crh(df, x_model='MI_pupil_unique',
            y_model='MI_task_unique',
            area='ICC',
            save=False,
            xlabel='MI pupil unique',
            ylabel='MI task unique',
            title='IC',
            xlim=(-0.45,0.45),
            ylim=(-0.45,0.45),
            ax=axs[1],
            bootstats=True)

common.scat_states_crh(df, x_model='MI_pupil_unique',
            y_model='MI_task_unique',
            area='ICX',
            save=False,
            xlabel='MI pupil unique',
            ylabel='MI task unique',
            title='IC',
            xlim=(-0.45,0.45),
            ylim=(-0.45,0.45),
            marker='v',
            ax=axs[1],
            bootstats=True)

# CRH adding scipy test for correlation significance -- it's in the ms, but code doesn't seem to exist?
a1cc, p = sci.pearsonr(df[df.area=='A1']['MI_task_unique'], df[df.area=='A1']['MI_pupil_unique'])     
print(f"A1 \n   correlation MI_task_unique vs. MI_pupil_unique: {round(a1cc, 3)}, {round(p, 3)}")       

iccc, p = sci.pearsonr(df[df.area.isin(['ICX', 'ICC'])]['MI_task_unique'], df[df.area.isin(['ICX', 'ICC'])]['MI_pupil_unique'])     
print(f"IC \n   correlation MI_task_unique vs. MI_pupil_unique: {round(iccc, 3)}, {round(p, 3)}")  

# test correlation using hierarchical bootstrap
np.random.seed(123)
print("Using hierarchical bootstrap:")
da1_task = {s: df.loc[(df.siteid==s) & (df.area=='A1'), 'MI_pupil_unique'].values for s in df[(df.area=='A1')].siteid.unique()}
da1_pupil = {s: df.loc[(df.siteid==s) & (df.area=='A1'), 'MI_task_unique'].values for s in df[(df.area=='A1')].siteid.unique()}
a1_boot_cc = get_bootstrapped_sample(da1_task, da1_pupil, metric='corrcoef', nboot=100)
p = 1 - get_direct_prob(a1_boot_cc, np.zeros(a1_boot_cc.shape[0]))[0]
print(f"A1 \n   correlation MI_task_unique vs. MI_pupil_unique: {round(a1cc, 3)}, {round(p, 5)}")  

dic_task = {s: df.loc[(df.siteid==s) & (df.area.isin(['ICC', 'ICX'])), 'MI_pupil_unique'].values for s in df[(df.area.isin(['ICC', 'ICX']))].siteid.unique()}
dic_pupil = {s: df.loc[(df.siteid==s) & (df.area.isin(['ICC', 'ICX'])), 'MI_task_unique'].values for s in df[(df.area.isin(['ICC', 'ICX']))].siteid.unique()}
ic_boot_cc = get_bootstrapped_sample(dic_task, dic_pupil, metric='corrcoef', nboot=100)
p = 1 - get_direct_prob(ic_boot_cc, np.zeros(ic_boot_cc.shape[0]))[0]
print(f"IC \n   correlation MI_task_unique vs. MI_pupil_unique: {round(iccc, 3)}, {round(p, 5)}")  

for s_area in ['A1', 'ICC|ICX', 'ICC', 'ICX']:
    for varname in ['MI_task_unique','MI_pupil_unique']:

        area = df.area.str.contains(s_area, regex=True) & df['sig_state']
        m=df.loc[area, varname].mean()
        stat,p = sci.wilcoxon(df.loc[area, varname].values)
        d = {s: df.loc[(df.siteid==s) & area, varname].values for s in df[area].siteid.unique()}
        bs = get_bootstrapped_sample(d, nboot=100)
        pboot = get_direct_prob(bs, np.zeros(bs.shape[0]))[0]
        npos=np.sum(df.loc[area, varname] > 0)
        n=len(df.loc[area, varname])
        print(f"{s_area} {varname}: mean={m:.3f} W={stat:.3f} p={p:.3e}, pboot={pboot:.3f}")
        print(f"  npos={npos}/{n}")

# A1 vs. IC change in task unique
a1 = {s: df.loc[(df.siteid==s) & (df.area=='A1'), 'MI_task_unique'].values for s in df[(df.area=='A1')].siteid.unique()}
ic = {s: df.loc[(df.siteid==s) & df.area.isin(['ICC', 'ICX']), 'MI_task_unique'].values for s in df[df.area.isin(['ICC', 'ICX'])].siteid.unique()}
a1 = get_bootstrapped_sample(a1, nboot=100)
ic = get_bootstrapped_sample(ic, nboot=100)
pboot = get_direct_prob(a1, ic)[0]
print(f"A1 task unique vs. IC task unique, pboot: {pboot}")

# fig S5 -- ICC / ICX task only vs. task unique


if save_fig:
    fh.savefig(os.path.join(save_path, 'fig5_MI_unique.pdf'))
