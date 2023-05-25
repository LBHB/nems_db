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
from nems_lbhb.motor.free_tools import compute_d_theta, \
    free_scatter_sum, dlc2dist

from nems_lbhb.projects.freemoving import free_model

dt = datetime.date.today().strftime("%Y-%m-%d")
figpath = f'/auto/users/svd/docs/current/grant/r21_free_moving/eps/{dt}/'
os.makedirs(figpath, exist_ok=True)


siteid = 'PRN020a'
siteid = 'PRN010a'
siteid = 'PRN015a'
siteid = 'PRN034a'
siteid = 'PRN018a'
siteid = 'PRN022a'
siteid = 'PRN043a'
siteid = 'PRN051a'

dlc_chans=10
rasterfs=50
rec = free_model.load_free_data(siteid, rasterfs=rasterfs, dlc_chans=dlc_chans)