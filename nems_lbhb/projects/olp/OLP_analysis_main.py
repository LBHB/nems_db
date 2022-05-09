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

fit = False
OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/Marmosets_OLP_add_spont.h5'
OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/Marmosets_OLP_resp.h5'
OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/Binaural_OLP_corr.h5'  #New for testig corr

batch = 328 #Ferret A1
batch = 329 #Ferret PEG
batch = 333 #Marmoset (HOD+TBR)
batch = 340 #All ferret OLP
batch = 339 #Binaural ferret OLP

if fit == True:
    cell_df = nd.get_batch_cells(batch)
    cell_list = cell_df['cellid'].tolist()
    cell_list = ohel.manual_fix_units(cell_list) #So far only useful for two TBR cells
    cell_list = [cc for cc in cell_list if cc[:6] == "CLT007"]
    cell_list = cell_list[:5]
    # cellid, parmfile = 'CLT007a-009-2', None

    metrics=[]
    for cellid in cell_list:
        cell_metric = ofit.calc_psth_metrics(batch, cellid)
        cell_metric.insert(loc=0, column='cellid', value=cellid)
        print(f"Adding cellid {cellid}.")
        metrics.append(cell_metric)

    df = pd.concat(metrics)
    df.reset_index()

    os.makedirs(os.path.dirname(OLP_cell_metrics_db_path),exist_ok=True)
    store = pd.HDFStore(OLP_cell_metrics_db_path)
    df_store=copy.deepcopy(df)
    store['df'] = df_store.copy()
    store.close()

else:
    store = pd.HDFStore(OLP_cell_metrics_db_path)
    df=store['df']
    store.close()

# df = df.query("cellid == 'CLT008a-006-2'")
df = df.query("cellid == 'CLT007a-002-1'")

weights = True
OLP_weights_db_path = '/auto/users/hamersky/olp_analysis/Binaural_OLP_full.h5' #weight + corr
if weights == True:
    df['batch'] = batch

    df_fit = df[['cellid', 'SR', 'batch']].copy()
    df_fit = df_fit.drop_duplicates(subset=['cellid'])

    df0 = df_fit.apply(ofit.calc_psth_weight_resp, axis=1, fs=fs)

    def drop_get_error(row):
        row['weight_dfR'] = row['weight_dfR'].copy().drop(columns=['get_error', 'Efit'])
        return row
    df0 = df0.copy().drop(columns='get_nrmseR')
    df0 = df0.apply(drop_get_error, axis=1)

    weight_df = pd.concat(df0['weight_dfR'].values, keys=df0.cellid).reset_index().\
        drop(columns='level_1')
    ep_names = [f"STIM_{aa}_{bb}" for aa, bb in zip(weight_df.namesA, weight_df.namesB)]
    weight_df = weight_df.drop(columns=['namesA', 'namesB'])
    weight_df['epoch'] = ep_names

    weights_df = pd.merge(right=weight_df, left=df, on=['cellid', 'epoch'])
    if df.shape[0] != weights_df.shape[0] or weight_df.shape[0] != weights_df.shape[0]:
        raise ValueError("Resulting weights_df does not match length of parts, some epochs were dropped.")

    weight_df = weights_df
    os.makedirs(os.path.dirname(OLP_weights_db_path),exist_ok=True)
    store = pd.HDFStore(OLP_weights_db_path)
    df_store=copy.deepcopy(weight_df)
    store['df'] = df_store.copy()
    store.close()

else:
    store = pd.HDFStore(OLP_weights_db_path)
    weight_df=store['df']
    store.close()




cell = 'CLT007a-009-2'
bg, fg = 'Waterfall', 'Keys'


cells = obip.get_cell_names(df)
cells['site'] = cells.cellid.str[:6]

a1df = df.loc[df.area == 'A1']
pegdf = df.loc[df.area == 'PEG']

a1 = cells.loc[cells.area == 'A1']
peg = cells.loc[cells.area == 'PEG']

site = 'CLT019'
sites = cells.loc[cells.site == site]
for ss in sites.cellid:
    print(ss)


cell = 'CLT007a-009-2'
cell = 'CLT007a-019-1'
cell = 'CLT019a-031-2'
cell = 'CLT019a-055-1'

pairs = obip.get_pair_names(cell, df)

bg = 'Rain'
fg = 'Bell'


obip.plot_binaural_psths(df, cell, bg, fg, batch, save=True, close=False)


check = df.loc[(df.cellid == cell) & (df.BG == bg) & (df.FG == fg)]


for ci in sites.cellid:
    obip.plot_binaural_psths(df, ci, bg, fg, batch, save=True, close=True)