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

"""
batch = 307  # A1 SUA and MUA
batch = 309  # IC SUA and MUA

basemodels = ["-ref-psthfr.s_stategain.S",
              "-ref-psthfr.s_sdexp.S",
              "-ref.a-psthfr.s_sdexp.S"]
state_list = ['st.pup0.far0.hit0.hlf0', 'st.pup0.far0.hit0.hlf',
              'st.pup.far.hit.hlf0', 'st.pup.far.hit.hlf']
state_list = ['st.pup0.fil0', 'st.pup0.fil', 'st.pup.fil0', 'st.pup.fil']
"""
state_list = ['st.pup0.hlf0', 'st.pup0.hlf', 'st.pup.hlf0', 'st.pup.hlf']

cellids = ["TAR010c-06-1", "TAR010c-27-2"]
batch = 307

for cellid in cellids:
    model_per_time_wrapper(cellid, batch=307,
                               loader= "psth.fs20.pup-ld-",
                               fitter = "_jk.nf20-basic",
                               basemodel = "-ref-psthfr_stategain.S",
                               state_list=None, plot_halves=True)