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
            ax=axs[0])

common.scat_states_crh(df, x_model='MI_pupil_unique',
            y_model='MI_task_unique',
            area='ICC',
            save=False,
            xlabel='MI pupil unique',
            ylabel='MI task unique',
            title='IC',
            xlim=(-0.45,0.45),
            ylim=(-0.45,0.45),
            ax=axs[1])
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
            ax=axs[1])

if save_fig:
    fh.savefig(os.path.join(save_path, 'fig5_MI_unique.pdf'))
