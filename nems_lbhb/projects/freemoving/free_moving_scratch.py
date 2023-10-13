from os.path import basename, join
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter, zoom

import datetime
import os

from nems0 import db
import nems0.epoch as ep
import nems0.plots.api as nplt
from nems0.utils import smooth
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.plots import plot_waveforms_64D
from nems_lbhb.preprocessing import impute_multi
from nems_lbhb import baphy_io
from nems_lbhb.projects.freemoving.free_tools import compute_d_theta, \
    free_scatter_sum, dlc2dist
from nems_lbhb.projects.freemoving import free_model, free_vs_fixed_strfs
from nems.tools import json

# code to support dumping figures
#dt = datetime.date.today().strftime("%Y-%m-%d")
#figpath = f'/auto/users/svd/docs/current/grant/r21_free_moving/eps/{dt}/'
#os.makedirs(figpath, exist_ok=True)


# tested sites
siteid = 'PRN020a'
siteid = 'PRN010a'
siteid = 'PRN015a'
siteid = 'PRN034a'
siteid = 'PRN018a'
siteid = 'PRN022a'
siteid = 'PRN043a'
siteid = 'PRN051a'

# interesting sites
siteid = 'PRN067a' # ok both
siteid = 'PRN015a' # nice aud, single stream
siteid = 'PRN047a' # some of everything.
siteid = 'PRN074a' # ok both

siteid = 'SLJ021a'
siteid = 'PRN048a' # some of everything.


dlc_chans=8
rasterfs=50
batch=348
rec = free_model.load_free_data(siteid, batch=batch, rasterfs=rasterfs, dlc_chans=dlc_chans, compute_position=True)

modelopts={'dlc_memory': 4, 'acount': 12, 'dcount': 8, 'l2count': 24, 'cost_function': 'squared_error'}
model = free_model.free_fit(rec, shuffle='none', apply_hrtf=True, save_to_db=True, **modelopts)
model2 = free_model.free_fit(rec, shuffle='none', apply_hrtf=False, save_to_db=True, **modelopts)

for i,c in enumerate(model2.meta['cellids']):
    print(f"{i}: {c} {model.meta['r_test'][i,0]:.3f} {model2.meta['r_test'][i,0]:.3f} {rec.meta['depth'][i]}")
print(f"MEAN              {model.meta['r_test'].mean():.3f} {model2.meta['r_test'].mean():.3f}")

# scatter plot of free-moving position with example positions highlighted
f = free_vs_fixed_strfs.movement_plot(rec)

# dSTRFS for interesting units: PRN048a-269-1, PRN048a-285-2
#for out_channel in [20,22]:
for out_channel in [6, 8, 5]:
    cellid = rec['resp'].chans[out_channel]
    mdstrf, pc1, pc2, pc_mag = free_vs_fixed_strfs.dstrf_snapshots(rec, [model, model2], D=11, out_channel=out_channel)
    f = free_vs_fixed_strfs.dstrf_plots(rec, [model, model2], mdstrf, out_channel)


modelpath =model.meta['modelfile']
modeltest = json.load_model(modelpath)
dlc_count = rec['dlc'].shape[0]
input = {'stim': est['stim'].as_continuous().T, 'dlc': est['dlc'].as_continuous().T[:, :dlc_count]}

testpred = modeltest.predict(input)['prediction']
origpred = model.predict(input)['prediction']

