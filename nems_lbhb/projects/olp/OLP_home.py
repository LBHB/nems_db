import nems_lbhb.projects.olp.OLP_plot_helpers as opl
import nems.db as nd
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import copy
import nems_lbhb.projects.olp.OLP_helpers as ohel
import nems_lbhb.projects.olp.OLP_fit as ofit
import nems_lbhb.projects.olp.OLP_Synthetic_plot as osyn
import nems_lbhb.projects.olp.OLP_Binaural_plot as obip

from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems.epoch as ep

sb.color_palette
sb.color_palette('colorblind')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sb.color_palette('colorblind'))
fs, paths = 100, None

batch = 328 #Ferret A1
batch = 329 #Ferret PEG
batch = 333 #Marmoset (HOD+TBR)
batch = 340 #All ferret OLP

# Add new filenames as you need to add things
filename = '_'
storepath = f'/auto/users/hamersky/olp_analysis/{filename}.h5'

# To fit whole batch and save it
weight_df = ofit.OLP_fit_weights(batch, savepath=storepath, filter=None)
# To fit only a specific parmfile and save it
weight_df = ofit.OLP_fit_weights(batch, parmfile=parmfile, savepath=storepath, filter=None)
# Alternate to parmfile loading is use keyword to get the number experiment you want
weight_df = ofit.OLP_fit_weights(batch, savepath=storepath, filter='CLT022')
# To filter by CLT Synthetic only, use a long list of experiment names
synths = [f'CLT0{cc}' for cc in range(27,54)]
weight_df = ofit.OLP_fit_weights(batch, savepath=storepath, filter=synths)
# To load
path = '/auto/users/hamersky/olp_analysis/Synthetic_Full.h5'
bin_path = = '/auto/users/hamersky/olp_analysis/Binaural_OLP_full_sound_stats.h5' #weight + corr
# This gets my big synthetic DF(NEED TO ADD CLT053)
weight_df = ofit.OLP_fit_weights(batch, loadpath=path)


# Plot simple comparison of sites and synethetics
osyn.plot_synthetic_weights(weight_df, areas=None, synth_show='A-')
