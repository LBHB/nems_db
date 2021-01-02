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
import nems.xform_helper as xhelp


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


# get all sites
sites = np.unique([s[:7] for s in df.index.get_level_values(0)])

# set column for siteid / batch
df['siteid'] = [s[:7] for s in df.index.get_level_values(0)]
df['batch'] = [307 if df['area'].iloc[i]=='A1' else 309 for i in range(0, df.shape[0])]
# for each site, compute the mean / se of r_pup_unique and r_task_unique
dfg = df[['r_task_unique', 'r_pupil_unique', 'siteid', 'batch']].groupby(by='siteid').mean()

# for each site, load recording and get pupil variance across ref stimuli (use nems mask from fitting)
modelname = 'psth.fs20.pup-ld-st.pup.beh-ref-psthfr_sdexp.S_jk.nf20-basic'
for site, batch in zip(dfg.index, dfg['batch']):
    cid = df[df.siteid==site].index.get_level_values(0)[0]
    xf, ctx = xhelp.load_model_xform(cid, batch, modelname, eval_model=True)