import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as sci
import statsmodels.formula.api as smf
import matplotlib.collections as clt
import re
import pylab as pl

import nems_lbhb.pupil_behavior_scripts.common as common
import nems_lbhb.pupil_behavior_scripts.helpers as helper
from nems_lbhb.analysis.statistics import get_bootstrapped_sample, get_direct_prob
from nems import get_setting

# set path to dump file
dump_path = get_setting('NEMS_RESULTS_DIR')

save_path = os.path.join(os.path.expanduser('~'),'docs/current/pupil_behavior/eps')
save_fig = False

# SPECIFY models
AFL = True
if AFL:
    dump_results = 'd_pup_afl_sdexp.csv'
    #dump_results = 'd_pup_afl_sdexp_ap1.csv'
    model_string = 'st.pup.afl'
    p0_model = 'st.pup0.afl'
    b0_model = 'st.pup.afl0'
    shuf_model = 'st.pup0.afl0'
else:
    dump_results = 'd_pup_fil_sdexp.csv'
    model_string = 'st.pup.fil'
    p0_model = 'st.pup0.fil'
    b0_model = 'st.pup.fil0'
    shuf_model = 'st.pup0.fil0'

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
    #df = df.groupby(by=['cellid', 'ON_BF']).mean()
    df = df.groupby(by=['cellid']).mean()
    df['area'] = [area.loc[c] if type(area.loc[c]) is str else area.loc[c].iloc[0] for c in df.index.get_level_values('cellid')]
    df['sig_task'] = df['sig_task'].astype(bool)
    df['sig_utask'] = df['sig_utask'].astype(bool)
    df['sig_pupil'] = df['sig_pupil'].astype(bool)
    df['sig_upupil'] = df['sig_upupil'].astype(bool)
    df['sig_state'] = df['sig_state'].astype(bool)
# Create figure

def donut_plot(area, unit_list, colors, savefigure=False):
    white_circle=plt.Circle((0,0), 0.7, color='white')
    plt.axis('equal')
    plt.pie(unit_list, colors=colors, labels=unit_list)
    p=plt.gcf()
    p.gca().add_artist(white_circle)
    plt.title(area)
    if savefigure:
        plt.savefig(area + '_donut.pdf')

print('A1 had {} units for which behavior alone sign modulated activity'.format(len(df.loc[(df.area=='A1') & df['sig_task']])))
print('A1 had {} units for which behavior unique sign modulated activity'.format(len(df.loc[(df.area=='A1') & df['sig_utask']])))
print('A1 had {} units for which behavior either only or unique sign modulated activity'.format(len(df.loc[(df.area=='A1') &
                                                                                                           (df['sig_task'] | df['sig_utask'])])))
print('IC had {} units for which behavior alone sign modulated activity'.format(len(df.loc[(df.area.isin(['ICC', 'ICX'])) & df['sig_task']])))
print('IC had {} units for which behavior unique sign modulated activity'.format(len(df.loc[(df.area.isin(['ICC', 'ICX']))& df['sig_utask']])))

# Fig 3A
A1_n_sig_both = df[(df.area=='A1') & df['sig_utask'] & df['sig_upupil']].shape[0]
A1_n_sig_ubeh = df[(df.area=='A1') & df['sig_utask']].shape[0] - A1_n_sig_both
A1_n_sig_upup = df[(df.area=='A1') & df['sig_upupil']].shape[0] - A1_n_sig_both
A1_n_sig_state = df[(df.area=='A1') & df['sig_state']].shape[0]
A1_n_sig_either = A1_n_sig_state - (A1_n_sig_both + A1_n_sig_ubeh + A1_n_sig_upup)

A1_n_total = df[df.area=='A1'].shape[0]
A1_n_not_sig = A1_n_total - (A1_n_sig_state)

A1_units = [A1_n_sig_ubeh, A1_n_sig_upup, A1_n_sig_both, A1_n_sig_either, A1_n_not_sig]

# IC

IC_n_sig_both = df[df.area.isin(['ICC', 'ICX']) & df['sig_utask'] & df['sig_upupil']].shape[0]
IC_n_sig_ubeh = df[df.area.isin(['ICC', 'ICX']) & df['sig_utask']].shape[0] - IC_n_sig_both
IC_n_sig_upup = df[df.area.isin(['ICC', 'ICX']) & df['sig_upupil']].shape[0] - IC_n_sig_both
IC_n_sig_state = df[df.area.isin(['ICC', 'ICX']) & df['sig_state']].shape[0]
IC_n_sig_either = IC_n_sig_state - (IC_n_sig_both + IC_n_sig_ubeh + IC_n_sig_upup)

IC_n_total = df[df.area.isin(['ICC', 'ICX'])].shape[0]
IC_n_not_sig = IC_n_total - (IC_n_sig_state)

IC_units = [IC_n_sig_ubeh, IC_n_sig_upup, IC_n_sig_both, IC_n_sig_either, IC_n_not_sig]

colors = [common.color_b, common.color_p, common.color_both, common.color_either, common.color_ns]

# Figure 3A
fh, axs = plt.subplots(3, 2, figsize=(5,7.5))

plt.sca(axs[0,0])
donut_plot('A1', A1_units, colors, savefigure=False)

plt.sca(axs[0,1])
donut_plot('IC', IC_units, colors, savefigure=False)

# Figure 3B

# A1 with colored units according to model significance
common.scat_states_crh(df, x_model='r_shuff',
            y_model='r_full',
            area='A1',
            save=False,
            xlim=(0,1),
            ylim=(0,1),
            xlabel='state-independent R2',
            ylabel='state-dependent R2',
            title='A1',
            ax=axs[1,0],
            bootstats=True)

# All IC with colored untis according to model significance
common.scat_states_crh(df, x_model='r_shuff',
            y_model='r_full',
            area='ICC|ICX',
            save=False,
            xlim=(0,1),
            ylim=(0,1),
            xlabel='state-independent R2',
            ylabel='state-dependent R2',
            title='IC',
            ax=axs[1,1],
            bootstats=True)

# Figure 3C
# A1
common.scat_states_crh(df, x_model='r_pupil_unique',
            y_model='r_task_unique',
            area='A1',
            save=False,
            xlabel='R2 pupil unique (task regressed out)',
            ylabel='R2 task unique (pupil regressed out)',
            title='A1',
            xlim=(-0.05,0.2),
            ylim=(-0.05,0.2),
            ax=axs[2,0],
            bootstats=True, nboots=1000)

# IC and ICX together
common.scat_states_crh(df, x_model='r_pupil_unique',
            y_model='r_task_unique',
            area='ICC|ICX',
            save=False,
            xlabel='R2 pupil unique (task regressed out)',
            ylabel='R2 task unique (pupil regressed out)',
            title='IC',
            xlim=(-0.05,0.2),
            ylim=(-0.05,0.2),
            ax=axs[2,1],
            bootstats=True, nboots=1000)

#plt.tight_layout()

# add statistical test to directly test if r_pupil/task is different between areas
np.random.seed(123)
df['siteid'] = [c[:7] for c in df.index]

# pupil test
da1 = {s: df.loc[(df.siteid==s) & (df.area=='A1'), 'r_pupil_unique'].values for s in df[(df.area=='A1')].siteid.unique()}
dic = {s: df.loc[(df.siteid==s) & df.area.isin(['ICC', 'ICX']), 'r_pupil_unique'].values for s in df[df.area.isin(['ICC', 'ICX'])].siteid.unique()}
a1 = get_bootstrapped_sample(da1, nboot=1000)
ic = get_bootstrapped_sample(dic, nboot=1000)
p = 1- get_direct_prob(a1, ic)[0]
print("\n")
print(f" Median r_pupil_unique IC: {df[df.area.isin(['ICC', 'ICX'])]['r_pupil_unique'].median()}\n"\
      f" Median r_pupil_unique A1: {df[df.area.isin(['A1'])]['r_pupil_unique'].median()}\n"\
      f" Bootstrapped probability A1 > IC: {p}\n")

# task test
da1 = {s: df.loc[(df.siteid==s) & (df.area=='A1'), 'r_task_unique'].values for s in df[(df.area=='A1')].siteid.unique()}
dic = {s: df.loc[(df.siteid==s) & df.area.isin(['ICC', 'ICX']), 'r_task_unique'].values for s in df[df.area.isin(['ICC', 'ICX'])].siteid.unique()}
a1 = get_bootstrapped_sample(da1, nboot=1000)
ic = get_bootstrapped_sample(dic, nboot=1000)
p = 1- get_direct_prob(a1, ic)[0]
print(f" Median r_task_unique IC: {df[df.area.isin(['ICC', 'ICX'])]['r_task_unique'].median()}\n"\
      f" Median r_task_unique A1: {df[df.area.isin(['A1'])]['r_task_unique'].median()}\n"\
      f" Bootstrapped probability A1 > IC: {p}")

# within area tests of pupil vs. task (using significant cells only)
print("Only significant cells included:\n")
sigmask = df.sig_state
da1 = {s: df.loc[(df.siteid==s) & (df.area=='A1') & sigmask, 'r_task_unique'].values -
                       df.loc[(df.siteid==s) & (df.area=='A1') & sigmask, 'r_pupil_unique'].values for s in df[(df.area=='A1')].siteid.unique()}
dic = {s: df.loc[(df.siteid==s) & df.area.isin(['ICC', 'ICX']) & sigmask, 'r_task_unique'].values - 
                       df.loc[(df.siteid==s) & df.area.isin(['ICC', 'ICX']) & sigmask, 'r_pupil_unique'].values 
                       for s in df[df.area.isin(['ICC', 'ICX'])].siteid.unique()}
a1 = get_bootstrapped_sample(da1, nboot=1000)
ic = get_bootstrapped_sample(dic, nboot=1000)
p = get_direct_prob(a1, np.zeros(a1.shape[0]))[0]
print(f"A1\n    r_pupil_unique vs. r_task_unique p-value: {p}")
p = get_direct_prob(ic, np.zeros(ic.shape[0]))[0]
print(f"IC\n    r_pupil_unique vs. r_task_unique p-value: {p}")
if save_fig:
    fh.savefig(os.path.join(save_path, 'fig3_r2_summ.pdf'))

