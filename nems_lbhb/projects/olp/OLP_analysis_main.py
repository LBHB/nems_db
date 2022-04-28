import nems_lbhb.TwoStim_helpers as ts
import nems_lbhb.projects.olp.OLP_plot_helpers as opl
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems.db as nd
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import copy
import nems_lbhb.projects.olp.OLP_helpers as ohel
import nems_lbhb.projects.olp.OLP_fit as ofit
import nems_lbhb.projects.olp.OLP_Binaural_plot as obip



sb.color_palette
sb.color_palette('colorblind')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sb.color_palette('colorblind'))
fs, paths = 100, None

# fit = False
fit = True
OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/Marmosets_OLP.h5'
OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/Marmosets_OLP_add_spont.h5'
OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/Marmosets_OLP_resp.h5'

batch = 328 #Ferret A1
batch = 329 #Ferret PEG
batch = 333 #Marmoset (HOD+TBR)
batch = 340 #All ferret OLP
batch = 339 #Binaural ferret OLP

if fit == True:
    cell_df = nd.get_batch_cells(batch)
    cell_list = cell_df['cellid'].tolist()
    cell_list = ohel.manual_fix_units(cell_list) #So far only useful for two TBR cells
    # cellid, parmfile = cell_list[4], None

    metrics=[]
    for cellid in cell_list[55:65]:
        cell_metric = ofit.calc_psth_metrics(batch, cellid)
        cell_metric.insert(loc=0, column='cellid', value=cellid)
        print(f"Adding cellid {cellid}.")
        metrics.append(cell_metric)

    df = pd.concat(metrics)
    df.reset_index()






cell = 'CLT007a-009-2'
bg, fg = 'Waterfall', 'Keys'


cells = obip.get_cell_names(df)

cell = 'CLT007a-009-2'

pairs = obip.get_pair_names(cell, df)

obip.plot_binaural_psths(df, cells, bg, fg, batch, save=True)
