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


save_path = os.path.join(os.path.expanduser('~'),'docs/current/pupil_behavior/eps')
save_fig = True


# SPECIFY pup+beh models
state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
basemodel = "-ref-psthfr.s_sdexp.S"

use_hlf = False
if use_hlf:
    state_list = ['st.pup0.hlf0', 'st.pup0.hlf', 'st.pup.hlf0', 'st.pup.hlf']
    #state_list = ['st.pup0.far0.hit0.hlf0','st.pup0.far0.hit0.hlf',
    #              'st.pup.far.hit.hlf0','st.pup.far.hit.hlf']
    states = ['PASSIVE_0_A','PASSIVE_0_B', 'ACTIVE_1_A','ACTIVE_1_B',
              'PASSIVE_1_A','PASSIVE_1_B', 'ACTIVE_2_A','ACTIVE_2_B',
              'PASSIVE_2_A','PASSIVE_2_B']
    #states = ['PASSIVE_0_A','PASSIVE_0_B', 'ACTIVE_1_A','ACTIVE_1_B',
    #          'PASSIVE_1_A','PASSIVE_1_B']
else:
    state_list = ['st.pup0.fil0', 'st.pup0.fil', 'st.pup.fil0', 'st.pup.fil']
    states = ['PASSIVE_0',  'ACTIVE_1', 'PASSIVE_1',
              'ACTIVE_2', 'PASSIVE_2']

batch=307
df = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)
title = "{} {} Area A1 keep sgn".format(basemodel,state_list[-1],batch)
f307, dMI, dMI0 = hlf_analysis(df, state_list, title=title, norm_sign=True, states=states)

batch=309
df = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)
title = "{} {} Area IC keep sgn".format(basemodel,state_list[-1],batch)
f309, dMI, dMI0 = hlf_analysis(df, state_list, title=title, norm_sign=True, states=states)

