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
import helpers as helper
import common

dump_path = get_setting('NEMS_RESULTS_DIR')
basemodel = "-ref-psthfr.s_sdexp.S"
state_list = ['st.pup0.fil0', 'st.pup0.fil', 'st.pup.fil0', 'st.pup.fil']
states = ['PASSIVE_0',  'ACTIVE_1', 'PASSIVE_1',
        'ACTIVE_2', 'PASSIVE_2']

batch=307
A1 = pd.read_csv(os.path.join(dump_path, str(batch), 'd_pup_fil_sdexp.csv'))
# convert r values to numeric
try:
        A1['r'] = [np.float(r.strip('[]')) for r in A1['r'].values]
        A1['r_se'] = [np.float(r.strip('[]')) for r in A1['r_se'].values]
except:
        pass
A1 = A1[~A1.cellid.str.contains('AMT')]
batch=309
IC = pd.read_csv(os.path.join(dump_path, str(batch), 'd_pup_fil_sdexp.csv'))
# convert r values to numeric
try:
        IC['r'] = [np.float(r.strip('[]')) for r in IC['r'].values]
        IC['r_se'] = [np.float(r.strip('[]')) for r in IC['r_se'].values]
except:
        pass
IC = IC[~IC.cellid.str.contains('AMT')]

title = "{} {} Area A1 keep sgn".format(basemodel,state_list[-1],batch)
helper.hlf_analysis(A1, state_list, title=title, norm_sign=True, sig_cells_only=True, states=states)


title = "{} {} Area IC keep sgn".format(basemodel,state_list[-1],batch)
helper.hlf_analysis(IC, state_list, title=title, norm_sign=True, sig_cells_only=True, states=states)
