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
save_fig = True

# ===================================================== pupil behavior data =======================================================
# SPECIFY models
USE_AFL = True
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
if 'stategain' in dump_results:
    preprocess_fn = helper.preprocess_stategain_dump
else:
    preprocess_fn = helper.preprocess_sdexp_dump

# import / preprocess model results
A1 = preprocess_fn(dump_results,
                    batch=307,
                    full_model=model_string,
                    p0=p0_model,
                    b0=b0_model,
                    shuf_model=shuf_model,
                    r0_threshold=r0_threshold,
                    octave_cutoff=octave_cutoff,
                    path=dump_path)
A1['area'] = 'A1'
IC = preprocess_fn(dump_results,
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

# 6A
norm_by_null=True
f = helper.aud_vs_state(df.loc[df.area=='A1'], nb=5, colors=common.color_list, title='A1', norm_by_null=norm_by_null)
if save_fig:
    f.savefig(os.path.join(save_path,'fig6_tuning_vs_pup_beh_A1.pdf'))

# 6B
f = helper.aud_vs_state(df.loc[df.area.isin(['ICC', 'ICX'])], nb=5, colors=common.color_list, title='IC', norm_by_null=norm_by_null)
if save_fig:
    f.savefig(os.path.join(save_path,'fig6_tuning_vs_pup_beh_IC.pdf'))


# ==================================================== behavior only data ===========================================================
# Figures 6C-D  - beh only effects, bigger set of cells
# later figure -- beh only (ignore pupil, can use larger stim set)

# SPECIFY models
USE_AFL = True
if USE_AFL:
    dump_results = 'd_afl_sdexp.csv'
    model_string = 'st.afl'
    p0_model = None
    b0_model = 'st.afl0'
    shuf_model = 'st.afl0'
else:
    dump_results = 'd_beh_sdexp.csv'
    model_string = 'st.beh'
    p0_model = None
    b0_model = 'st.beh0'
    shuf_model = 'st.beh0'

# set params for BF characterization and sig. sensory response threshold
octave_cutoff = 0.5
r0_threshold = 0
group_files = True
if 'stategain' in dump_results:
    preprocess_fn = helper.preprocess_stategain_dump
else:
    preprocess_fn = helper.preprocess_sdexp_dump

# import / preprocess model results
A1 = []
for batch in [307, 311, 312]:
    _A1 = preprocess_fn(dump_results,
                        batch=batch,
                        full_model=model_string,
                        p0=p0_model,
                        b0=b0_model,
                        shuf_model=shuf_model,
                        r0_threshold=r0_threshold,
                        octave_cutoff=octave_cutoff,
                        path=dump_path)
    _A1['area'] = 'A1'
    A1.append(_A1)
A1 = pd.concat(A1)

IC = []
for batch in [295, 313]:
    _IC = preprocess_fn(dump_results,
                        batch=batch,
                        full_model=model_string,
                        p0=p0_model,
                        b0=b0_model,
                        shuf_model=shuf_model,
                        r0_threshold=r0_threshold,
                        octave_cutoff=octave_cutoff,
                        path=dump_path)
    _IC['area'] = 'IC'
    IC.append(_IC)
IC = pd.concat(IC)

df = pd.concat([A1, IC])

if group_files & ('beh' not in model_string):
    area = df['area']
    df = df.groupby(by=['cellid']).mean()
    df['area'] = [area.loc[c] if type(area.loc[c]) is str else area.loc[c][0] for c in df.index.get_level_values('cellid')]

f = helper.aud_vs_state(df.loc[df.area=='IC'], nb=5, state_list=['st.afl0', 'st.afl'], colors=common.color_list, title='IC', norm_by_null=norm_by_null)
if save_fig:
    f.savefig(os.path.join(save_path,'fig6_tuning_vs_beh_only_IC.pdf'))

f = helper.aud_vs_state(df.loc[df.area=='A1'], nb=5, state_list=['st.afl0', 'st.afl'], colors=common.color_list, title='A1', norm_by_null=norm_by_null)
if save_fig:
    f.savefig(os.path.join(save_path,'fig6_tuning_vs_beh_only_A1.pdf'))

"""
"""