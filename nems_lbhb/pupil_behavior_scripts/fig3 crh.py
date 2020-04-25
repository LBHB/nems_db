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
from nems import get_setting

# set path to dump file
dump_path = get_setting('NEMS_RESULTS_DIR')

save_path = os.path.join(os.path.expanduser('~'),'docs/current/pupil_behavior/eps')
save_fig = False


# SPECIFY models
dump_results = 'd_pup_afl_sdexp.csv'
model_string = 'st.pup.afl'
p0_model = 'st.pup0.afl'
b0_model = 'st.pup.afl0'
shuf_model = 'st.pup0.afl0' 

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
            ax=axs[1,0])

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
            ax=axs[1,1])

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
                   ax=axs[2,0])

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
            ax=axs[2,1])

#plt.tight_layout()

if save_fig:
    fh.savefig(os.path.join(save_path, 'fig3_r2_summ.pdf'))

