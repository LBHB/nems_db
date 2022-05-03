#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:58:16 2020

@author: luke
"""
#import SPO_helpers as sp
import nems_lbhb.TwoStim_helpers as ts
import nems_lbhb.OLP_plot_helpers as opl
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems.db as nd
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import copy

sb.color_palette
sb.color_palette('colorblind')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sb.color_palette('colorblind'))
# OLP_cell_metrics_db_path='/auto/users/luke/Projects/OLP/NEMS/celldat_A1_v1.h5'

OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/Marmosets_OLP.h5'
batch, fs, paths = 333, 100, None

#Load my saved fit
store = pd.HDFStore(OLP_cell_metrics_db_path)
df = store['df']
store.close()

#Decide which cell:
#UPDATE THE LINE BELOW TO POINT TO THE FILE
#rec_file_dir='/auto/data/nems_db/recordings/306/'
#cellid='fre196b-21-1_1417-1233'; rf='fre196b_ec1d319ae74a2d790f3cbda73d46937e588bc791.tgz'
#cellid='fre197c-105-1_705-1024'; rf='fre197c_f94fb643b4cb6380b8eb3286fc30d908a1940ea2.tgz' #Neuron 1 on poster
#rec_file = rec_file_dir + rf

# batch=328
# if batch == 328:
#     OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/a1_new_celldat1.h5'
# if batch == 329:
#     OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/peg_new_celldat1.h5'
# cell_df=nd.get_batch_cells(batch)
# cell_list=cell_df['cellid'].tolist()
# fs=100
#
# cell_list = [cell for cell in cell_list if cell[:3] != 'HOD']
# cell_list = [cell for cell in cell_list if cell[:3] == 'TBR']
#
# cell_list = [cell for cell in cell_list if cell[:7] == 'ARM033a']
#cell_list=cell_df['cellid'].tolist()[-10:-8]
#cell_list=['ARM013b-03-1','ARM013b-04-1']

#cellid='ARM020a-05';


#options = {'rasterfs': 100, 'resp': True}
# load from parmfile(s)
#pf = ['full_path_to_mfile1', 'full_path_to_mfile2']
#pf=['/auto/data/daq/Armillaria/ARM020/ARM020a05_p_OLP.m']
#manager = BAPHYExperiment(parmfile=pf)
#rec = manager.get_recording(**options)
## load from cellid(or siteid)/batch
#cellid = 'TAR010c'
#batch = 307
#rec = manager.get_recording(**options)

# # This was to try and load using parmfiles, it never really worked but I spent time on it..
# cell_list, parm_list, rasterfs = [], [], 100
# cell_dict = dict()
# for parm in parmfiles:
#     count = 0
#     expt = BAPHYExperiment(parm)
#     rec = expt.get_recording(rasterfs=rasterfs, resp=True, stim=False)
#     resp = rec['resp'].rasterize()
#     chan = resp.chans
#     for ch in chan:
#         iso = int(nd.get_cell_files(ch).isolation.unique()[0])
#         # print(f"Channel: {ch} -- Isolation: {iso}")
#         if iso >= 95:
#             cell_list.append(ch)
#             print(f"Good Channel: {ch} -- Isolation: {iso}")
#             parm_list.append(parm)
#             count += 1
#     print(f"Kept {count} units of {len(chan)}")
    # parms = [parm] * len(cell_list)
    # parm_list_first.append(parms)
# cell_list = [item for sublist in cell_list_first for item in sublist]
# parm_list = [item for sublist in parm_list_first for item in sublist]
OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/Marmosets_OLP.h5'

paths = ['Background1', 'Foreground1']

batch = 333
cell_df=nd.get_batch_cells(batch)
cell_list=cell_df['cellid'].tolist()
fs=100

cell_list, title, paths = [cell for cell in cell_list if cell[:3] == 'HOD'], 'Hood', None
cell_list, title = [cell for cell in cell_list if cell[:3] == 'TBR'], 'Tabor'

# PSTH metrics that have to do with one stimulus at a time
if True:
    metrics=[]
    for cellid in cell_list:
        if paths:
            metrics_ = ts.calc_psth_metrics(batch, cellid, paths=paths)
        else:
            metrics_ = ts.calc_psth_metrics(batch, cellid)
        print('****rAAm: {} rBBm: {}'.format(metrics_['rAAm'], metrics_['rBBm']))
        metrics.append(metrics_)

    df=pd.DataFrame(data=metrics)
    df['modelspecname']='dlog_fir2x15_lvl1_dexp1'
    df['cellid']=cell_list
    df = df.set_index('cellid')

    df = df.apply(ts.type_by_psth, axis=1);
    df['batch']=batch

    if paths:
        df=df.apply(ts.calc_psth_weight_resp,axis=1,fs=fs, paths=paths)
    else:
        df = df.apply(ts.calc_psth_weight_resp, axis=1, fs=fs)

    # df2 = ts.calc_psth_weight_resp(df.iloc[0])  #apply to one cell by index number
    # df2 = ts.calc_psth_weight_resp(df.loc['ARM031a-39-1'])  #apply to one cell by name
    # df2 = ts.calc_psth_weight_resp(df.loc['ARM031a-39-1'],find_mse_confidence=False,do_plot=True)  #apply to one cell by name and plot

    os.makedirs(os.path.dirname(OLP_cell_metrics_db_path),exist_ok=True)
    store = pd.HDFStore(OLP_cell_metrics_db_path)
    df_store=copy.deepcopy(df)
    def drop_get_error(row):
        row['weight_dfR'] = row['weight_dfR'].copy().drop(columns='get_error')
        return row
    df_store=df_store.apply(drop_get_error,axis=1)
    df_store=df_store.drop(columns=['get_nrmseR', 'rec', ])
    cols_to_keep = df_store[['weight_dfR', 'pair_names', 'suppression', 'FR', 'animal',
    'modelspecname', 'corcoef', 'avg_resp', 'snr', 'batch', 'weightsR', 'EfitR', 'nMSER', 'nfR', 'rR', 'namesR',
     'namesAR', 'namesBR', 'WeightAgroupsR', 'WeightBgroupsR', 'norm_spont']]
    store['df'] = cols_to_keep.copy()
    store.close()
else:
    store = pd.HDFStore(OLP_cell_metrics_db_path)
    df=store['df']
    store.close()

cols=['EP_A','EP_B','IP_A','IP_B','SR','SR_av_std']
cols2=cols+['SinglesMax','MEnh_I','MSupp_I','Rtype']
df[cols2]

##subtract metrics
#check if my thresholds are reliably getting rid of the same 'bad' units
# the_data = df
# corco_df = df.loc[df['corcoef'] >= 0.1]
# bad_corco = df.loc[df['corcoef'] < 0.1]
# avg_df = df.loc[df['avg_resp'] >= 0.025]
# bad_avg = df.loc[df['avg_resp'] < 0.025]
# snr_df = df.loc[df['snr'] >= 0.03]
# bad_snr = df.loc[df['snr'] < 0.03]
#
# bad_corc_set = set(bad_corco.index)
# bad_avg_set = set(bad_avg.index)
# bad_snr_set = set(bad_snr.index)
# consistent_bad = bad_corc_set.intersection(bad_avg_set)
# bad_set = bad_corc_set.union(bad_avg_set)
# consistent_bad = bad_corc_set.intersection(bad_avg_set, bad_snr_set)
# bad_set = bad_corc_set.union(bad_avg_set, bad_snr_set)



#filter dataframe
df = df.loc[(df['corcoef'] >= 0.1) & (df['avg_resp'] >= 0.025)]

Wcols = ['namesA','namesB','weightsA','weightsB']
weight_df = pd.concat(df['weight_dfR'].values,keys=df.index)
BGgroups = pd.concat(df['WeightAgroupsR'].values,keys=df.index)
FGgroups = pd.concat(df['WeightBgroupsR'].values,keys=df.index)
animal_list = [anim[0][:3] for anim in weight_df.index]
single_animal = np.unique(animal_list) == 1
weight_df.insert(4, 'Animal', animal_list)

#Add suppression column to weights_df
supp_df = pd.DataFrame()
for cll in df.index:
    supp = df.loc[cll,'suppression']
    fr = df.loc[cll, 'FR']

    names = [ts.get_sep_stim_names(sn) for sn in df.loc[cll,'pair_names']]
    BGs, FGs = [rr[0] for rr in names], [qq[1] for qq in names]
    cell_df = pd.DataFrame({'suppression': supp,
                           'BG_FR': fr[:,0],
                           'FG_FR': fr[:,1],
                           'Combo_FR': fr[:,2],
                           'namesA': BGs,
                           'namesB': FGs,
                           'cellid': cll})
    supp_df = supp_df.append(cell_df)

supp_df = supp_df.set_index('cellid', append=True)
supp_df = supp_df.swaplevel(0,1)
supp_df = supp_df.set_index(['namesA','namesB'], append=True)
weight_df = weight_df.set_index(['namesA','namesB'], append=True)
joint = pd.concat([weight_df, supp_df], axis=1)
weight_df = joint.reset_index(['namesA','namesB'])


#Beginning plotting, these first lines are general and helpful
bins=np.arange(-2,2,.05)
if batch == 328:
    titles = 'A1'
if batch == 329:
    titles = 'PEG'
if batch == 333:
    if paths:
        if single_animal:
            titles = f"{title+' Subset'}"
        else:
            titles = f"{np.unique(animal_list)} Subset"
    else:
        titles = f"{np.unique(animal_list)}"
import seaborn as sns


# Filtered big df by certain epochs for sounds used with Hood
weight_df = opl.filter_epochs_by_file_names([1,1], weight_df=weight_df)

#Trying to make FR scatter and filter weight_df by a particular sound
kw = 'Tsik'
df_filtered, plotids, fnargs = opl.get_keyword_sound_type(kw, weight_df=weight_df,
                                                          single=True, exact=False)

#Two keywords to get only a single sound pair across all units
kw = 'Waterfall'
kw2 = 'Alarm'
df_filtered, plotids, fnargs = opl.get_keyword_sound_type(kw, weight_df=weight_df,
                                                          single=True, exact=True, kw2=kw2)

opl.split_psth_multiple_units(df_filtered, sortby='suppression', order='low', sigma=2)

opl.split_psth_highest_lowest(df_filtered, sortby='suppression', rows=18, sigma=2)

#If you just want standard scatter of BG weight v FG weight
plotids, df_filtered, fnargs = {'xcol': 'weightsA', 'ycol': 'weightsB', 'fn':opl.plot_psth}, weight_df.copy(), {'df_filtered': weight_df}


# Run this once you have plotids, df_filtered, fnargs defined based on what you want to plot
cellid_and_stim_strs= [index[0]+':'+nameA+':'+nameB for index,nameA,nameB in \
                      zip(df_filtered.index.values,
                          df_filtered['namesA'],df_filtered['namesB'])]

# Makes interactive plot, works with standard scatter or single keyword, untested with dual keyword
f, ax = plt.subplots(1,1)
phi=ts.scatterplot_print(df_filtered[plotids['xcol']].values,
                         df_filtered[plotids['ycol']].values,
                         cellid_and_stim_strs, plotids,
                         ax=ax,fn=plotids['fn'],fnargs=fnargs)
ax.set_title(f"{titles}")

# This is what I used to save PSTHs to /OLP PSTHs/ which was sorted by BG or FG sound
kws = ['Chimes', 'Gravel', 'Insect', 'Rain', 'Rock', 'Stream', 'Thunder', 'Waterfall', 'Waves', 'Wind', 'Alarm', 'Chirp', 'Shrill', 'Phee', 'Seep', 'Trill', 'TwitterA', 'TwitterB', 'Ek']
for kw in kws:
    df_filtered, plotids, fnargs = opl.get_keyword_sound_type(kw, weight_df=weight_df,
                                                              single=True)
    cellid_and_stim_strs= [index[0]+':'+nameA+':'+nameB for index,nameA,nameB in \
                          zip(df_filtered.index.values,
                              df_filtered['namesA'],df_filtered['namesB'])]

    for css in cellid_and_stim_strs:
        opl.psth_responses_by_kw(css, df_filtered, kw, plotids['sound_type'], sigma=2, save=True)
###

opl.scatter_weights(weight_df, titles)
opl.heatmap_weights(weight_df, titles)




#Colorful scatter plot of BG weights v FG weights
plt.figure()
for i in range(len(df)):
    plt.plot(df.iloc[i].weightsR[0,:],df.iloc[i].weightsR[1,:],'.')
plt.xlabel('Background Weights')
plt.ylabel('Foreground Weights')
plt.xlim((-2, 2))
plt.ylim((-2, 2))
plt.title(f"{titles}")
plt.gca().set_aspect(1)

#Colorful scatter plot of BG weights v FG weights - moved to opl
fig, ax = plt.subplots()
g = sns.scatterplot(x='weightsA', y='weightsB', data=weight_df, hue='Animal')
ax.set_title(f"{titles}")

plt.xlabel('Background Weights')
plt.ylabel('Foreground Weights')
plt.xlim((-2, 2))
plt.ylim((-2, 2))
plt.title(f"{titles}")
plt.gca().set_aspect(1)

#Heatmap version of BG/FG weight scatter
weights=np.concatenate(df.weightsR.values,axis=1)
weights=weights[:,~np.any(np.isnan(weights),axis=0)]
plt.figure();  plt.hist2d(weights[0,:],weights[1,:],bins=bins)
plt.xlim((-.5, 1.5)); plt.ylim((-.5, 1.5))
plt.gca().set_aspect(1)
plt.xlabel('Background Weights')
plt.ylabel('Foreground Weights')
plt.title(f"{titles}")

#Same plot as the previous one but using the weight dataframe - moved to opl
gi=~np.isnan(weight_df['weightsA']) & ~np.isnan(weight_df['weightsB'])
# weights=weights[:,~np.any(np.isnan(weights),axis=0)]
plt.figure(figsize=(5,5));  plt.hist2d(weight_df['weightsA'][gi],weight_df['weightsB'][gi],bins=bins)
plt.xlim((-.5, 1.5)); plt.ylim((-.5, 1.5))
plt.gca().set_aspect(1)
plt.xlabel('Background Weight')
plt.ylabel('Foreground Weight')
plt.title(f"{titles}")

#WARNING, LEGEND BACKWARDS???!
#plt.figure();  plt.hist(weights.T,bins=400,histtype='step')
#plt.legend(('Background','Foreground'))
#plt.xlim((-1, 2));

#Histogram of BG weights and FG weights
bins=np.arange(-2,2,.05)
plt.figure();
plt.hist(weights[0,:],bins=bins,histtype='step')
plt.hist(weights[1,:],bins=bins,histtype='step')
plt.legend(('Background','Foreground'))
plt.xlabel('Weight')
plt.title(f"{titles}")

#Histogram of difference between FG and BG weights
plt.figure(figsize=(5,5));  plt.hist(np.diff(weights,axis=0).T,bins=bins,histtype='step')
plt.xlim((-2, 2));
plt.xlabel('Paired Foreground - Background')
plt.title(f"{titles}")

## Plots based on differences
# Scatterplot of range of weights over constant bg
plt.figure(figsize=(5,5))
for cellid in BGgroups.index.levels[0]:
    plt.plot(BGgroups.loc[cellid]['weightsA']['range'], BGgroups.loc[cellid]['weightsB']['range'], '.')
plt.xlabel('Background Weight Range')
plt.ylabel('Foreground Weight Range')
plt.title('Weight ranges over a constant Background')
plt.xlim((0, 5))
plt.ylim((0, 5))
plt.gca().set_aspect(1)

#2-d hist of range of weights over constant fg and constant bg
gi=~np.isnan(BGgroups['weightsA']['range']) & ~np.isnan(BGgroups['weightsB']['range'])
f, ax = plt.subplots(1, 2, sharex=True, sharey=True,figsize=(8.7,4))
ax[0].hist2d(BGgroups[gi]['weightsA']['range'],\
           BGgroups[gi]['weightsB']['range'],\
           bins=np.linspace(0,5,40))
ax[0].plot((0,2),(0,2),'-k',linewidth=0.5)
gi=~np.isnan(FGgroups['weightsA']['range']) & ~np.isnan(FGgroups['weightsB']['range'])
ax[1].hist2d(FGgroups[gi]['weightsA']['range'],\
           FGgroups[gi]['weightsB']['range'],\
           bins=np.linspace(0,5,40))
ax[1].plot((0,2),(0,2),'-k',linewidth=0.5)
[axi.set_xlim((0,2)) for axi in ax]
[axi.set_ylim((0,2)) for axi in ax]
[axi.set_aspect('equal') for axi in ax];
ax[0].set_xlabel('Background Weight Range')
ax[0].set_ylabel('Foreground Weight Range')
ax[0].set_title('Over a constant Background')
ax[1].set_title('Over a constant Foreground')


#Marginal histograms of range of weights over constant fg and constant bg
bins=np.linspace(0,2,30)
f, ax = plt.subplots(1,2,figsize=(8.7,4))
gi=~np.isnan(BGgroups['weightsA']['range']) & ~np.isnan(BGgroups['weightsB']['range'])
N,_=np.histogram(BGgroups[gi]['weightsA']['range'],bins=bins)
N=N.astype(np.float32) /sum(N)*100
ax[0].step(np.insert(bins,0,0), np.insert(np.append(N,0),0,0),where='post',color='C0')
N,_=np.histogram(BGgroups[gi]['weightsB']['range'],bins=bins)
N=N.astype(np.float32) /sum(N)*100
ax[0].step(np.insert(bins,0,0), -1*np.insert(np.append(N,0),0,0),where='post',linestyle='--',color='C0')
gi=~np.isnan(FGgroups['weightsA']['range']) & ~np.isnan(FGgroups['weightsB']['range'])
N,_=np.histogram(FGgroups[gi]['weightsA']['range'],bins=bins)
N=N.astype(np.float32) /sum(N)*100
ax[0].step(np.insert(bins,0,0), np.insert(np.append(N,0),0,0),where='post',color='C1')
N,_=np.histogram(FGgroups[gi]['weightsB']['range'],bins=bins)
N=N.astype(np.float32) /sum(N)*100
ax[0].step(np.insert(bins,0,0), -1*np.insert(np.append(N,0),0,0),where='post',linestyle='--',color='C1')
ax[0].set_ylabel('Percent of Population')
ax[0].set_xlabel('Range of weights')
yl = np.array((-1,1))*np.max(np.abs(ax[0].get_ylim()))
ax[0].set_ylim(yl)
ytl=ax[0].get_yticklabels()
for this_ytl in ytl:
    if this_ytl._y < 0:
        this_ytl.set_text(this_ytl._text[1:]);
ax[0].set_yticklabels(ytl)
ax[0].legend(('Range of Bg over constant Bg','Range of Fg over constant Bg','Range of Bg over constant Fg','Range of Fg over constant Fg'),loc='upper right', bbox_to_anchor=(1.05,1.15))

#Histograms of difference in range of weights (fg-bg) over constant fg and constant bg
bins=np.linspace(-2,2,61)
gi=~np.isnan(BGgroups['weightsA']['range']) & ~np.isnan(BGgroups['weightsB']['range'])
N,_=np.histogram(BGgroups[gi]['weightsB']['range']-BGgroups[gi]['weightsA']['range'],bins=bins)
N=N.astype(np.float32) /sum(N)*100
ax[1].step(np.insert(bins,0,bins[0]), np.insert(np.append(N,0),0,0),where='post',color='C0')
gi=~np.isnan(FGgroups['weightsA']['range']) & ~np.isnan(FGgroups['weightsB']['range'])
N,_=np.histogram(FGgroups[gi]['weightsB']['range']-FGgroups[gi]['weightsA']['range'],bins=bins)
N=N.astype(np.float32) /sum(N)*100
ax[1].step(np.insert(bins,0,bins[0]), np.insert(np.append(N,0),0,0),where='post',color='C1')
ax[1].legend(('range(Fg) - range(Bg) over constant Bg','range(Fg) - range(Bg) over constant Fg'), bbox_to_anchor=(1.05,1.15))
ax[1].set_xlabel('Diff in range of weights (range(Fg)-range(Bg))')
ax[1].plot((0,0),(0,np.max(np.abs(ax[1].get_ylim()))),'k',linewidth=.5)
#ends here

#Get and plot error functions
err = weight_df.iloc[0]['get_error']()
squared_errors = np.zeros((len(err),len(weight_df)))
for i in range(len(weight_df)):
    err = weight_df.iloc[i]['get_error']()
    norm_factor = weight_df.iloc[i]['nf'] #mean of resp to Fg+Bg squared
    squared_errors[:,i] = err**2/norm_factor

time = np.arange(0, err.shape[-1]) / fs
plt.figure();
#plt.plot(time,squared_errors,linewidth=.5)
plt.plot(time,np.nanmean(squared_errors,axis=1),'k',LineWidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Normalized Squared Error')
plt.title(f"{titles}")

#To plot PSTHs and weight model for a given cell
cellid=cell_list[4];
row=df.loc[cellid]['weight_dfR'].iloc[0]
plt.figure();
err=row['get_error']()
time = np.arange(0, err.shape[-1]) / fs
plt.plot(time,row['get_error'](get_what='sigA'))
plt.plot(time,row['get_error'](get_what='sigB'))
plt.plot(time,row['get_error'](get_what='sigAB'))
plt.plot(time,row['get_error'](get_what='pred'))
plt.legend(('Bg','Fg','Both','Weight Model'))

#to plot error function
plt.figure();plt.plot(time,row['get_error']()/np.sqrt(row['nf']))

#PSTH plotting function:
def plot_psth(cellid_and_stim_str, weight_df=weight_df, plot_error=True):
    if plot_error:
        nr=2
    else:
        nr=2
    f, ax = plt.subplots(1,nr, figsize=(15,5))
    if nr==1: ax=[ax]
    cellid,stimA,stimB = cellid_and_stim_str.split(':')
    cell_df=weight_df.loc[cellid]
    this_cell_stim=cell_df[(cell_df['namesA']==stimA) & (cell_df['namesB']==stimB)].iloc[0]
    err=this_cell_stim['get_error']()
    time = np.arange(0, err.shape[-1]) / fs
    ax[0].plot(time,this_cell_stim['get_error'](get_what='sigA'))
    ax[0].plot(time,this_cell_stim['get_error'](get_what='sigB'))
    ax[0].plot(time,this_cell_stim['get_error'](get_what='sigAB'))
    ax[0].plot(time,this_cell_stim['get_error'](get_what='pred'))
    ax[0].plot(time, this_cell_stim['get_error'](get_what='sigA') +
               this_cell_stim['get_error'](get_what='sigB'), linestyle=":",
               color='black')
    ax[0].legend(('Bg, weight={:.2f}'.format(this_cell_stim.weightsA),
      'Fg, weight={:.2f}'.format(this_cell_stim.weightsB),
      'Both',
      'Weight Model, r={:.2f}'.format(this_cell_stim.r),
      'Linear Sum'))
    ax[0].set_title(f"{cellid_and_stim_str} sup:{this_cell_stim['suppression']}")
    
    if plot_error:
        ax[1].plot(time,this_cell_stim['get_error']()/np.sqrt(this_cell_stim['nf']))

    bg = this_cell_stim['get_error'](get_what='sigA')[:100]
    fg = this_cell_stim['get_error'](get_what='sigB')[:100]
    ab = this_cell_stim['get_error'](get_what='sigAB')[:100]
    binweights = pd.DataFrame({'bg': bg, 'fg': fg, 'combo': ab})
    # fig,ax = plt.subplots()
    a = ax[1].scatter(bg, fg, c=ab, cmap='inferno', s=15)
    ax[1].set_xlabel('r(BG)'), ax[1].set_ylabel('r(FG)')
    f.colorbar(a)
    # plt.gca().set_aspect(1)
    f.tight_layout()


weight_df = weight_df.dropna()
#Make interactive scatterplot of weights
cellid_and_str_strs= [index[0]+':'+nameA+':'+nameB for index,nameA,nameB in \
                      zip(weight_df.index.values,
                          weight_df['namesA'],weight_df['namesB'])]

f, ax = plt.subplots(1,1)
fnargs={'plot_error':False}
phi=ts.scatterplot_print(weight_df['weightsA'].values,
                         weight_df['weightsB'].values,
                         cellid_and_str_strs,
                         ax=ax,fn=plot_psth,fnargs=fnargs)
ax.set_title(f"{titles}")


#Check and plot the exclusion metrics
snrs, coefs, avgs = df['snr'], df['corcoef'], df['avg_resp']
fig, ax = plt.subplots(2,2)
ax = ax.ravel('F')
ax[0].hist(snrs.values, bins=75), ax[0].set_title('SNR', fontweight='bold')
ax[1].hist(coefs.values, bins=75), ax[1].set_title('Corr Coef', fontweight='bold')
ax[2].hist(avgs.values, bins=75), ax[2].set_title('Average Response', fontweight='bold')
fig.suptitle(f'Batch {batch}', fontweight='bold')

###########################################################################
#Make interactive scatterplot of weights - UNFINISHED######################
X = df['corcoef']
Y = df['avg_resp']
idx_df = df
cellid_and_str_strs= [index[0]+':'+nameA+':'+nameB for index,nameA,nameB in \
                      zip(weight_df.index.values,
                          weight_df['namesA'],weight_df['namesB'])]
# coefs = [df['corcoef'][cell.split(':')[0]] for cell in cellid_and_str_strs]
# avgs = [df['avg_resp'][cell.split(':')[0]] for cell in cellid_and_str_strs]
# s2nrs = [df['snr'][cell.split(':')[0]] for cell in cellid_and_str_strs]
#
# cellid_and_str_strs = [orig+':'+coe+':'+avg+':'+s2 for orig,coe,avg,s2 in \
#                        zip(cellid_and_str_strs, str(coefs), str(avgs), str(s2nrs))]

f, ax = plt.subplots(1,1)
fnargs={'plot_error':False}
phi=ts.scatterplot_print(weight_df['weightsA'].values,
                         weight_df['weightsB'].values,
                         cellid_and_str_strs,
                         ax=ax,fn=plot_psth,fnargs=fnargs)


def plot_psth(cellid_and_stim_str, weight_df=weight_df, df=df, plot_error=True):
    if plot_error:
        nr = 2
    else:
        nr = 1
    f, ax = plt.subplots(nr, 1)
    if nr == 1: ax = [ax]
    cellid, stimA, stimB = cellid_and_stim_str.split(':')
    cell_df = weight_df.loc[cellid]
    this_cell_stim = cell_df[(cell_df['namesA'] == stimA) & (cell_df['namesB'] == stimB)].iloc[0]
    err = this_cell_stim['get_error']()
    time = np.arange(0, err.shape[-1]) / fs
    ax[0].plot(time, this_cell_stim['get_error'](get_what='sigA'))
    ax[0].plot(time, this_cell_stim['get_error'](get_what='sigB'))
    ax[0].plot(time, this_cell_stim['get_error'](get_what='sigAB'))
    ax[0].plot(time, this_cell_stim['get_error'](get_what='pred'))
    ax[0].legend(('Bg, weight={:.2f}'.format(this_cell_stim.weightsA),
                  'Fg, weight={:.2f}'.format(this_cell_stim.weightsB),
                  'Both',
                  'Weight Model, r={:.2f}'.format(this_cell_stim.r)))
    ax[0].set_title(f"{cellid_and_stim_str} -\ncorcoef: {df['corcoef'][cellid]} - "
                    f"average response: {df['avg_resp'][cellid]} - snr: {df['snr'][cellid]}")

    if plot_error:
        ax[1].plot(time, this_cell_stim['get_error']() / np.sqrt(this_cell_stim['nf']))
##########################################################################

##Plot weights by individual sound
import seaborn as sns
import joblib as jl
from pathlib import Path

cell_list = df.index.values.tolist()

sound_weights = pd.DataFrame()

for cellid in cell_list:
    bb = df.namesAR[cellid]
    ff = df.namesBR[cellid]
    ww = df.weightsR[cellid]

    bgs = [bg.split('_', 1)[1].replace('_null', '').split('-', 1)[0] for bg in bb]
    fgs = [fg.split('_', 1)[1].replace('null_', '').split('-', 1)[0] for fg in ff]
    bgs_idx = [int(bg[:2]) for bg in bgs]
    fgs_idx = [int(fg[:2]) for fg in fgs]

    bg_weights = pd.DataFrame(
        {'name': bgs,
         'type': 'bg',
         'idx': bgs_idx,
         'weight': ww[0, :],
         'animal': cellid[:3]
         })
    fg_weights = pd.DataFrame(
        {'name': fgs,
         'type': 'fg',
         'idx': fgs_idx,
         'weight': ww[1, :],
         'animal': cellid[:3]
         })

    bgfg_weight = pd.concat([bg_weights, fg_weights], ignore_index=True)
    sound_weights = sound_weights.append(bgfg_weight, ignore_index=True)
sound_weights = sound_weights.sort_values(['type', 'animal', 'idx'], ascending=[True, True, True])

# # saving function I don't really know why I added in here a while back
# PEG_path = Path('/auto/users/hamersky/olp_analysis/PEG_new_weights')
# A1_path = Path('/auto/users/hamersky/olp_analysis/A1_new_weights')
# if PEG_path.parent.exists() is False and batch == 329:
#     PEG_path.parent.mkdir()
# if A1_path.parent.exists() is False and batch == 328:
#     A1_path.parent.mkdir()
# if batch == 329:
#     jl.dump(sound_weights, PEG_path)
# if batch == 328:
#     jl.dump(sound_weights, A1_path)
#
# if batch == 329:
#     peg_sound_weights = jl.load(PEG_path)
# if batch == 328:
#     a1_sound_weights = jl.load(A1_path)
# if batch == 333:
#     TBR_sound_weights = sound_weights

if sound_weights['animal'].nunique() > 1:
    animals = list(sound_weights['animal'].unique())
    sound_weights_list = [sound_weights.loc[sound_weights['animal'] == ani, :] for ani in animals]
else:
    sound_weights_list = [sound_weights]

for sound_dfs in sound_weights_list:
    fig, ax = plt.subplots()
    if batch == 329:
        g = sns.stripplot(x='name', y='weight', hue='type', data=peg_sound_weights, ax=ax)
        ax.set_ylim(-3, 2.5)
        ax.set_title('PEG', fontweight='bold')
        mean_weights = peg_sound_weights.groupby(by='name').agg('mean')
        type_weights = peg_sound_weights.groupby(by='type').agg('mean')
    if batch == 328:
        g = sns.stripplot(x='name', y='weight', hue='type', data=a1_sound_weights, ax=ax)
        ax.set_ylim(-3, 2.5)
        ax.set_title('A1', fontweight='bold')
        mean_weights = a1_sound_weights.groupby(by='name').agg('mean')
        type_weights = a1_sound_weights.groupby(by='type').agg('mean')
    if batch == 333:
        g = sns.stripplot(x='name', y='weight', hue='type', data=sound_dfs, ax=ax)
        ax.set_ylim(-3, 2.5)
        ax.set_title(f"{sound_dfs['animal'].unique()[0]}", fontweight='bold')
        mean_weights = sound_dfs.groupby(by='name').agg('mean')
        type_weights = sound_dfs.groupby(by='type').agg('mean')

    bg_count = sound_dfs.loc[sound_dfs['type'] == 'bg']['name'].nunique()
    fg_count = sound_dfs.loc[sound_dfs['type'] == 'fg']['name'].nunique()

    labels = [e.get_text() for e in plt.gca().get_xticklabels()]
    ticks = plt.gca().get_xticks()
    w = 0.4
    for cnt, sound_name in enumerate(labels):
        plt.hlines(mean_weights.loc[sound_name]['weight'],
                   ticks[cnt] - w, ticks[cnt] + w, lw=3, color='black')
    plt.hlines(type_weights.loc[type_weights.index == 'bg']['weight'],
               -0.5, bg_count - 0.5, lw=2, linestyles=':', color='deepskyblue')
    plt.hlines(type_weights.loc[type_weights.index == 'fg']['weight'],
               bg_count - 0.5, (bg_count+fg_count) - 0.5, lw=2, linestyles=':',
               color='yellowgreen')
    g.set_xticklabels(g.get_xticklabels(), rotation=60)
    ax.set_ylim(bottom=-0.75, top=1.75)


#weights vs suppression quick plots with regression
plt.figure(figsize=(5,5))
g = sns.regplot(x='weightsA', y='suppression', data=weight_df, color='black')
g.set_xlabel('BG Weight')
g.set_title(f'{titles} -- BG', fontweight='bold')
plt.figure(figsize=(5,5))
g = sns.regplot(x='weightsB', y='suppression', data=weight_df, color='black')
g.set_xlabel('FG Weight')
g.set_title(f'{titles} -- FG', fontweight='bold')


#Plots all weights for each cell -- Messy and doesn't look good, maybe order it
fig, ax = plt.subplots()
weighties = copy.copy(weight_df)
cell_weights = weighties.reset_index()
g = sns.stripplot(x='cellid', y='weightsA', color='deepskyblue', data=cell_weights, ax=ax)
g = sns.stripplot(x='cellid', y='weightsB', color='yellowgreen', data=cell_weights, ax=ax)
ax.set_ylim(2, -1)
ax.set_title(f'{titles}', fontweight='bold')

bee = pd.DataFrame({'cellid': cell_weights['cellid'],
                    'type': 'bg',
                    'weight': cell_weights['weightsA']})

eff = pd.DataFrame({'cellid': cell_weights['cellid'],
                    'type': 'fg',
                    'weight': cell_weights['weightsB']})
beef = bee.append(eff)

grouped = beef.groupby('cellid').agg('mean')
sorted = grouped.sort_values('weight')
sortorder = sorted.index


fig, ax = plt.subplots()
g = sns.stripplot(x='cellid', y='weight', hue='type', data=beef, ax=ax, order=sortorder)
ax.set_ylim(-1, 2)
ax.set_title(f'{titles}', fontweight='bold')

groupie = beef.groupby(['cellid','type']).agg('mean')
groupie = groupie.reset_index()
pivy = groupie.pivot(index='cellid', columns='type', values='weight')
fig, ax = plt.subplots()
g = sns.scatterplot(x='bg', y='fg', data=pivy)
plt.gca().set_aspect(1)
mini = min(pivy.min())
maxi = max(pivy.max())
ax.set_ylim(mini,maxi)
ax.set_xlim(mini,maxi)
ax.plot([mini,maxi],[mini,maxi], linestyle=':', color='black')
ax.set_title(f'{titles}', fontweight='bold')




#Regression
import statsmodels.formula.api as smf
import statsmodels.api as sm

reg_df = weight_df.reset_index(['cellid'])
reg_df = sm.add_constant(reg_df)

results = smf.ols(formula='suppression ~ C(cellid) + weightsA + '
                          'weightsB + const', data=reg_df).fit()

#Do with shuffles
results = pd.DataFrame()
shuffles = [None, 'neuron', 'weightsA', 'weightsB']
shuffles = ['weightsA', 'weightsB', 'neuron', None]
rr, res = {}, {}
regres = pd.DataFrame()
for shuf in shuffles:
    reg_results = neur_stim_reg(reg_df, shuf)
    rr[shuf] = reg_results.rsquared
    res[shuf] = reg_results
    regres['results'], regres['r'] = reg_results, reg_results.rsquared

#Plot small line plot of rsquare with different shuffle conditions
fig,ax = plt.subplots()
ax.plot(rr.values(), linestyle='-', color='black')
ax.set_ylabel('R_squared', fontweight='bold', size=12)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['WeightBG\nShuffled', 'WeightFG\nShuffled',
                    'Neuron\nShuffled', 'Full\nModel'],
                   fontweight='bold', size=8)
ax.set_title(f"{titles}", fontweight='bold', size=15)

#Function for implementing each shuffle
def neur_stim_reg(reg_df, shuffle=None):
    if not shuffle:
        results = smf.ols(formula='suppression ~ C(cellid) + weightsA + '
                                  'weightsB + const', data=reg_df).fit()
    if shuffle == 'neuron':
        neur_shuff = reg_df.copy()
        neur_shuff['cellid'] = neur_shuff['cellid'].iloc[np.random.choice(
            np.arange(reg_df.shape[0]),reg_df.shape[0], replace=False)].values
        results = smf.ols(formula='suppression ~ C(cellid) + weightsA + '
                                  'weightsB + const', data=neur_shuff).fit()
    if shuffle == 'weightsA':
        A_shuff = reg_df.copy()
        A_shuff['weightsA'] = A_shuff['weightsA'].iloc[np.random.choice(
            np.arange(reg_df.shape[0]),reg_df.shape[0], replace=False)].values
        results = smf.ols(formula='suppression ~ C(cellid) + weightsA + '
                                  'weightsB + const', data=A_shuff).fit()
    if shuffle == 'weightsB':
        B_shuff = reg_df.copy()
        B_shuff['weightsB'] = B_shuff['weightsB'].iloc[np.random.choice(
            np.arange(reg_df.shape[0]),reg_df.shape[0], replace=False)].values
        results = smf.ols(formula='suppression ~ C(cellid) + weightsA + '
                                  'weightsB + const', data=B_shuff).fit()

    # reg_results = _regression_results(results, shuffle)
    return results



##I'm here vvv trying to make dataframe of relevant reg results
def _regression_results(results, shuffle):
    intercept = results.params.loc[results.params.index.str.contains('Intercept')].values
    int_err = results.bse.loc[results.bse.index.str.contains('Intercept')].values
    int_conf = results.conf_int().loc[results.conf_int().index.str.contains('Intercept')].values[0]
    neuron_coeffs = results.params.loc[results.params.index.str.contains('cellid')].values
    neuron_coeffs = np.concatenate(([0], neuron_coeffs))
    stim_coeffs = results.params.loc[results.params.index.str.contains('stimulus')].values
    stim_coeffs = np.concatenate(([0], stim_coeffs))
    neur_coeffs = neuron_coeffs + intercept + stim_coeffs.mean()
    stim_coeffs = stim_coeffs + intercept + neuron_coeffs.mean()
    coef_list = np.concatenate((neur_coeffs, stim_coeffs))

    neuron_err = results.bse.loc[results.bse.index.str.contains('neuron')].values
    stim_err = results.bse.loc[results.bse.index.str.contains('stimulus')].values
    neuron_err = np.concatenate((int_err, neuron_err))
    stim_err = np.concatenate((int_err, stim_err))
    err_list = np.concatenate((neuron_err, stim_err))

    neur_low_conf = results.conf_int()[0].loc[results.conf_int().index.str.contains('neuron')].values
    neur_low_conf = np.concatenate(([int_conf[0]], neur_low_conf))
    stim_low_conf = results.conf_int()[0].loc[results.conf_int().index.str.contains('stimulus')].values
    stim_low_conf = np.concatenate(([int_conf[0]], stim_low_conf))
    low_list = np.concatenate((neur_low_conf, stim_low_conf))

    neur_high_conf = results.conf_int()[1].loc[results.conf_int().index.str.contains('neuron')].values
    neur_high_conf = np.concatenate(([int_conf[1]], neur_high_conf))
    stim_high_conf = results.conf_int()[1].loc[results.conf_int().index.str.contains('stimulus')].values
    stim_high_conf = np.concatenate(([int_conf[1]], stim_high_conf))
    high_list = np.concatenate((neur_high_conf, stim_high_conf))

    neur_list = ['neuron'] * len(neur_coeffs)
    stim_list = ['stimulus'] * len(stim_coeffs)
    name_list = np.concatenate((neur_list, stim_list))

    if shuffle == None:
        shuffle = 'full'
    shuff_list = [f"{shuffle}"] * len(name_list)
    site_list = [f"{params['experiment']}"] * len(name_list)
    r_list = [f"{np.round(results.rsquared,4)}"] * len(name_list)

    name_list_actual = list(params['good_units'])
    name_list_actual.extend(params['pairs'])

    reg_results = pd.DataFrame(
        {'name': name_list_actual,
         'id': name_list,
         'site': site_list,
         'shuffle': shuff_list,
         'coeff': coef_list,
         'error': err_list,
         'conf_low': low_list,
         'conf_high': high_list,
         'rsquare': r_list
        })

    return reg_results


def plot_psth(cellid_and_stim_str, weight_df=weight_df, plot_error=True):
    # Version of popup psth plot that includes spectrograms below psth
    from nems.analysis.gammatone.gtgram import gtgram
    from scipy.io import wavfile

    f = plt.figure(figsize=(15, 9))
    psth = plt.subplot2grid((4, 5), (0, 0), rowspan=2, colspan=3)
    specA = plt.subplot2grid((4, 5), (2, 0), rowspan=1, colspan=3)
    specB = plt.subplot2grid((4, 5), (3, 0), rowspan=1, colspan=3)
    scat = plt.subplot2grid((4, 5), (0, 3), rowspan=2, colspan=2)

    ax = [psth, specA, specB, scat]

    cellid, stimA, stimB = cellid_and_stim_str.split(':')
    cell_df = weight_df.loc[cellid]
    this_cell_stim = cell_df[(cell_df['namesA'] == stimA) & (cell_df['namesB'] == stimB)].iloc[0]
    err = this_cell_stim['get_error']()
    time = np.arange(0, err.shape[-1]) / fs
    ax[0].plot(time, this_cell_stim['get_error'](get_what='sigA'))
    ax[0].plot(time, this_cell_stim['get_error'](get_what='sigB'))
    ax[0].plot(time, this_cell_stim['get_error'](get_what='sigAB'))
    ax[0].plot(time, this_cell_stim['get_error'](get_what='pred'))
    ax[0].plot(time, this_cell_stim['get_error'](get_what='sigA') +
               this_cell_stim['get_error'](get_what='sigB'), linestyle=":",
               color='black')
    ax[0].legend(('Bg, weight={:.2f}'.format(this_cell_stim.weightsA),
                  'Fg, weight={:.2f}'.format(this_cell_stim.weightsB),
                  'Both',
                  'Weight Model, r={:.2f}'.format(this_cell_stim.r),
                  'Linear Sum'))
    ax[0].set_title(f"{cellid_and_stim_str} sup:{this_cell_stim['suppression']}")

    AA, BB = this_cell_stim['namesA'].split('-')[0], this_cell_stim['namesB'].split('-')[0]

    animal_id = cellid_and_str_strs[0].split(':')[0][:3]
    if animal_id == 'HOD':
        folder_ids = [1,1]
    elif animal_id == 'TBR':
        folder_ids = [2,3]
    elif animal_id == 'ARM':
        folder_ids = [1,2]

    bg_path = f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/Background{folder_ids[0]}/{AA}.wav'
    fg_path = f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/Foreground{folder_ids[1]}/{BB}.wav'

    xf = 100
    low, high = ax[0].get_xlim()
    low, high = low * xf, high * xf
    ticks = ax[0].get_xticklabels()

    sfs, W = wavfile.read(bg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[1].imshow(spec, aspect='auto', origin='lower', extent=[0,spec.shape[1], 0, spec.shape[0]])
    ax[1].set_xlim(low, high)
    ax[1].set_xticks([0, 20, 40, 60, 80]), ax[1].set_yticks([])
    ax[1].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8]), ax[1].set_yticklabels([])
    ax[1].spines['top'].set_visible(False), ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False), ax[1].spines['right'].set_visible(False)

    sfs, W = wavfile.read(fg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[2].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]])
    ax[2].set_xlim(low, high)
    ax[2].set_xticks([0, 20, 40, 60, 80]), ax[2].set_yticks([])
    ax[2].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8]), ax[2].set_yticklabels([])
    ax[2].spines['top'].set_visible(False), ax[2].spines['bottom'].set_visible(False)
    ax[2].spines['left'].set_visible(False), ax[2].spines['right'].set_visible(False)

    bg = this_cell_stim['get_error'](get_what='sigA')[:100]
    fg = this_cell_stim['get_error'](get_what='sigB')[:100]
    ab = this_cell_stim['get_error'](get_what='sigAB')[:100]
    a = ax[3].scatter(bg, fg, c=ab, cmap='inferno', s=15)
    ax[3].set_xlabel('r(BG)'), ax[3].set_ylabel('r(FG)')
    f.colorbar(a)
    f.tight_layout()


#Make interactive scatterplot of weights

weight_df = A1_weight_df.loc[(A1_weight_df['r'] >= 0.87)]

cellid_and_str_strs= [index[0]+':'+nameA+':'+nameB for index,nameA,nameB in \
                      zip(weight_df.index.values,
                          weight_df['namesA'],weight_df['namesB'])]

f, ax = plt.subplots(1,1)
fnargs={'plot_error':False}
phi=ts.scatterplot_print(weight_df['weightsA'].values,
                         weight_df['weightsB'].values,
                         cellid_and_str_strs,
                         ax=ax,fn=plot_psth,fnargs=fnargs)
ax.set_title(f"{titles}")


#Manicure by BG
def sound_df(bg=None, fg=None, weight_df=weight_df):
    some = weight_df
    if bg:
        bgs = tuple(["{:02d}".format(bb) for bb in bg])
        some = some.loc[some.namesA.str.startswith(bgs)]
    if fg:
        fgs = tuple(["{:02d}".format(ff) for ff in fg])
        some = some.loc[some.namesB.str.startswith(fgs)]
    if not bg:
        bgs = 'All'
    if not fg:
        fgs = 'All'

    return some, bgs, fgs


def interactive_plot(bg=None, fg=None, weight_df=weight_df):
    some, bgs, fgs = sound_df(bg=bg, fg=fg, weight_df=weight_df)
    cellid_and_str_strs= [index[0]+':'+nameA+':'+nameB for index,nameA,nameB in \
                          zip(some.index.values,
                              some['namesA'],some['namesB'])]

    f, ax = plt.subplots(1,1)
    fnargs={'plot_error':False}
    phi=ts.scatterplot_print(some['weightsA'].values,
                             some['weightsB'].values,
                             cellid_and_str_strs,
                             ax=ax,fn=plot_psth,fnargs=fnargs)
    ax.set_title(f"{titles} - BG {bgs} - FG {fgs}")

interactive_plot(bg=[1], fg=None)


#small figure adding for DAC, just compares average BG/FG for A1 and PEG
#on same bar graph to condense
from scipy.stats import ttest_ind
A1s, PEGs = A1_sound_weights, PEG_sound_weights

A1s['area'] = 'A1'
PEGs['area'] = 'PEG'

allweights = pd.concat([A1s, PEGs])


g = sns.stripplot(x='type', y='weight', hue='type', data=A1s, ax=ax)
ax.set_ylim(-3, 2.5)
ax.set_title('PEG', fontweight='bold')
mean_weights = peg_sound_weights.groupby(by='name').agg('mean')
type_weights = peg_sound_weights.groupby(by='type').agg('mean')

fig, ax = plt.subplots()

g = sns.pointplot(x='area', y='weight', hue='type', data=allweights, ax=ax, ci=68,
                  capsize=0.05, join=False, dodge=0.1)

cat1 = A1s[A1s['type'] == 'bg']
cat2 = A1s[A1s['type'] == 'fg']
ttestA1 = ttest_ind(cat1['weight'], cat2['weight'])

cat1 = PEGs[PEGs['type'] == 'bg']
cat2 = PEGs[PEGs['type'] == 'fg']
ttestPEG = ttest_ind(cat1['weight'], cat2['weight'])


##Making heat map thing for all units
exp_df = weight_df.loc['ARM031a-45-3']
err = exp_df['get_error']()


cell_df = weight_df.loc[cellid]
this_cell_stim = cell_df[(cell_df['namesA'] == stimA) & (cell_df['namesB'] == stimB)].iloc[0]
err = this_cell_stim['get_error']()
time = np.arange(0, err.shape[-1]) / fs


#Final plot_psth to go with interactive scatter, style taken from OLP_psth_plot.plot_responses()

def plot_psth(cellid_and_stim_str, weight_df=weight_df, plot_error=True):
    # Version of popup psth plot that includes spectrograms below psth
    from nems.analysis.gammatone.gtgram import gtgram
    from scipy.io import wavfile
    import scipy.ndimage.filters as sf

    cellid, stimA, stimB = cellid_and_stim_str.split(':')
    cell_df = weight_df.loc[cellid]
    this_cell_stim = cell_df[(cell_df['namesA'] == stimA) & (cell_df['namesB'] == stimB)].iloc[0]
    animal_id = cellid[:3]
    weightBG, weightFG = this_cell_stim['weightsA'], this_cell_stim['weightsB']

    if animal_id == 'HOD' or animal_id == 'TBR':
        batch = 333
    elif animal_id == 'ARM':
        # got to code in for batch to differentiate between A1 and PEG batches,
        # where can I get that info above?
        batch = 0

    fs=100
    expt = BAPHYExperiment(cellid=cellid, batch=batch)
    rec = expt.get_recording(rasterfs=fs, resp=True, stim=False)
    resp = rec['resp'].rasterize()

    BG, FG = stimA.split('-')[0], stimB.split('-')[0]
    epochs = [f'STIM_{BG}-0-1_null', f'STIM_null_{FG}-0-1', f'STIM_{BG}-0-1_{FG}-0-1']
            # , f'STIM_{BG}-0.5-1_{FG}-0-1']

    resp = resp.extract_channels([cellid])
    r = resp.extract_epochs(epochs)

    f = plt.figure(figsize=(12, 9))
    psth = plt.subplot2grid((4, 3), (0, 0), rowspan=2, colspan=3)
    specA = plt.subplot2grid((4, 3), (2, 0), rowspan=1, colspan=3)
    specB = plt.subplot2grid((4, 3), (3, 0), rowspan=1, colspan=3)
    ax = [psth, specA, specB]

    prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
    time = (np.arange(0, r[epochs[0]].shape[-1]) / fs ) - prestim

    bg_alone, fg_alone = epochs[0], epochs[1]
    r_mean = {e:np.squeeze(np.nanmean(r[e], axis=0)) for e in epochs}
    r_mean['Weight Model'] = (r_mean[bg_alone] * weightBG) + (r_mean[fg_alone] * weightFG)
    r_mean['Linear Sum'] = r_mean[bg_alone] + r_mean[fg_alone]

    colors = ['deepskyblue', 'yellowgreen', 'lightcoral', 'dimgray', 'black']
    styles = ['-', '-', '-', '-', ':']

    for e, c, s in zip(r_mean.keys(), colors, styles):
        ax[0].plot(time, sf.gaussian_filter1d(r_mean[e], sigma=1)
             * fs, color=c, linestyle=s, label=e)
    ax[0].legend(('Bg, weight={:.2f}'.format(this_cell_stim.weightsA),
                  'Fg, weight={:.2f}'.format(this_cell_stim.weightsB),
                  'Both',
                  'Weight Model, r={:.2f}'.format(this_cell_stim.r),
                  'Linear Sum'))
    ax[0].set_title(f"{cellid_and_stim_str} sup:{this_cell_stim['suppression']}")
    ymin, ymax = ax[0].get_ylim()
    ax[0].vlines([0,1.0], ymin, ymax, color='black', lw=0.75, ls='--')
    ax[0].vlines(0.5, ymax*0.5, ymax, color='black', lw=0.75, ls=':')
    ax[0].set_xlim((-prestim * 0.5), (1 + (prestim * 0.75)))

    #parts for spectrograms now
    if animal_id == 'HOD':
        folder_ids = [1,1]
    elif animal_id == 'TBR':
        folder_ids = [2,3]
    elif animal_id == 'ARM':
        folder_ids = [1,2]

    bg_path = f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/' \
              f'Background{folder_ids[0]}/{BG}.wav'
    fg_path = f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/' \
              f'Foreground{folder_ids[1]}/{FG}.wav'

    xf = 100
    low, high = ax[0].get_xlim()
    low, high = low * xf, high * xf

    sfs, W = wavfile.read(bg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[1].imshow(spec, aspect='auto', origin='lower', extent=[0,spec.shape[1], 0, spec.shape[0]])
    ax[1].set_xlim(low, high)
    ax[1].set_xticks([0, 20, 40, 60, 80]), ax[1].set_yticks([])
    ax[1].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8]), ax[1].set_yticklabels([])
    ax[1].spines['top'].set_visible(False), ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False), ax[1].spines['right'].set_visible(False)

    sfs, W = wavfile.read(fg_path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[2].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]])
    ax[2].set_xlim(low, high)
    ax[2].set_xticks([0, 20, 40, 60, 80]), ax[2].set_yticks([])
    ax[2].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8]), ax[2].set_yticklabels([])
    ax[2].spines['top'].set_visible(False), ax[2].spines['bottom'].set_visible(False)
    ax[2].spines['left'].set_visible(False), ax[2].spines['right'].set_visible(False)

weight_df = weight_df.dropna()
#Make interactive scatterplot of weights


cellid_and_str_strs= [index[0]+':'+nameA+':'+nameB for index,nameA,nameB in \
                      zip(weight_df.index.values,
                          weight_df['namesA'],weight_df['namesB'])]

f, ax = plt.subplots(1,1)
fnargs = {'xcol': 'weightsA', 'ycol': 'weightsB', 'fn':plot_psth}
phi=ts.scatterplot_print(weight_df['weightsA'].values,
                         weight_df['weightsB'].values,
                         cellid_and_str_strs,
                         ax=ax,fn=plot_psth,fnargs={})
ax.set_title(f"{titles}")



weight_dff = weight_df

# Filtered big df by certain epochs for sounds used with Hood
weight_df = filter_epochs_by_file_names([1,1], weight_df=weight_df)
df_filtered = {}

#Trying to make FR scatter and filter weight_df by a particular sound
kw = 'Wind'
df_filtered, plotids, fnargs = get_keyword_sound_type(kw, weight_df=weight_df)

#If you just want standard scatter of BG weight v FG weight
plotids, df_filtered, fnargs = {'xcol': 'weightsA', 'ycol': 'weightsB', 'fn':plot_psth}, weight_df.copy(), {'weight_df': weight_df}


# Run this once you have plotids, df_filtered, fnargs defined based on what you want to plot
cellid_and_stim_strs= [index[0]+':'+nameA+':'+nameB for index,nameA,nameB in \
                      zip(df_filtered.index.values,
                          df_filtered['namesA'],df_filtered['namesB'])]
f, ax = plt.subplots(1,1)
phi=ts.scatterplot_print(df_filtered[plotids['xcol']].values,
                         df_filtered[plotids['ycol']].values,
                         cellid_and_stim_strs, plotids,
                         ax=ax,fn=plotids['fn'],fnargs=fnargs)
ax.set_title(f"{titles}")


def plot_single_psth(cellid_and_stim_str, sound_type, df_filtered=df_filtered):
    # Version of popup psth plot that includes spectrograms below psth
    from nems.analysis.gammatone.gtgram import gtgram
    from scipy.io import wavfile
    import scipy.ndimage.filters as sf
    from scipy import stats
    import glob

    cellid, stimA, stimB = cellid_and_stim_str.split(':')
    cell_df = df_filtered.loc[cellid]
    this_cell_stim = cell_df[(cell_df['namesA'] == stimA) & (cell_df['namesB'] == stimB)].iloc[0]
    animal_id = cellid[:3]
    # weightBG, weightFG = this_cell_stim['weightsA'], this_cell_stim['weightsB']
    # frBG, frFG = this_cell_stim['BG_FR'], this_cell_stim['FG_FR']

    if animal_id == 'HOD' or animal_id == 'TBR':
        batch = 333
    elif animal_id == 'ARM':
        # got to code in for batch to differentiate between A1 and PEG batches,
        # where can I get that info above?
        batch = 0

    fs=100
    expt = BAPHYExperiment(cellid=cellid, batch=batch)
    rec = expt.get_recording(rasterfs=fs, resp=True, stim=False)
    resp = rec['resp'].rasterize()

    BG, FG = stimA.split('-')[0], stimB.split('-')[0]
    if sound_type == 'BG':
        epochs = f'STIM_{BG}-0-1_null'
        color = 'deepskyblue'
    elif sound_type == 'FG':
        epochs = f'STIM_null_{FG}-0-1'
        color = 'yellowgreen'
    else:
        raise ValueError(f"sound_type must be 'BG' or 'FG', {sound_type} is invalid.")

    resp = resp.extract_channels([cellid])
    r = resp.extract_epochs(epochs)

    sem = np.squeeze(stats.sem(r[epochs], axis=0, nan_policy='omit'))
    r_mean = np.squeeze(np.nanmean(r[epochs], axis=0))

    f = plt.figure(figsize=(12, 6))
    psth = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=3)
    spec = plt.subplot2grid((3, 3), (2, 0), rowspan=1, colspan=3)
    ax = [psth, spec]

    prestim = resp.epochs[resp.epochs['name'] == 'PreStimSilence'].copy().iloc[0]['end']
    time = (np.arange(0, r[epochs].shape[-1]) / fs) - prestim

    ax[0].plot(time, sf.gaussian_filter1d(r_mean, sigma=1)
         * fs, color=color, label=epochs)

    ax[0].fill_between(time, sf.gaussian_filter1d((r_mean - sem) * fs, sigma=1),
                    sf.gaussian_filter1d((r_mean + sem) * fs, sigma=1),
                    alpha=0.3, color='grey')
    ax[0].legend((f"{sound_type}, weight={np.around(this_cell_stim['weightsA'],2)}\n"
                  f"{sound_type}, firing rate={this_cell_stim['BG_FR']}",
                  ' '))
    ax[0].set_title(f"{cellid_and_stim_str}")
    ymin, ymax = ax[0].get_ylim()
    ax[0].vlines([0,1.0], ymin, ymax, color='black', lw=0.75, ls='--')
    ax[0].vlines(0.5, ymax*0.5, ymax, color='black', lw=0.75, ls=':')
    ax[0].set_xlim((-prestim * 0.5), (1 + (prestim * 0.75)))

    #parts for spectrograms now
    if animal_id == 'HOD':
        folder_ids = [1,1]
    elif animal_id == 'TBR':
        folder_ids = [2,3]
    elif animal_id == 'ARM':
        folder_ids = [1,2]

    BG, FG = int(BG[:2]), int(FG[:2])


    if sound_type == 'BG':
        path = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                           f'Background{folder_ids[0]}/*.wav'))[BG - 1]
    elif sound_type =='FG':
        path = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/' \
                           f'Foreground{folder_ids[1]}/*.wav'))
    else:
        raise ValueError(f"sound_type must be 'BG' or 'FG', {sound_type} is invalid.")

    xf = 100
    low, high = ax[0].get_xlim()
    low, high = low * xf, high * xf

    sfs, W = wavfile.read(path)
    spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
    ax[1].imshow(spec, aspect='auto', origin='lower', extent=[0,spec.shape[1], 0, spec.shape[0]])
    ax[1].set_xlim(low, high)
    ax[1].set_xticks([0, 20, 40, 60, 80]), ax[1].set_yticks([])
    ax[1].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8]), ax[1].set_yticklabels([])
    ax[1].spines['top'].set_visible(False), ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False), ax[1].spines['right'].set_visible(False)

def filter_epochs_by_file_names(pathidx, weight_df=weight_df):
    # Takes an array of two numbers [BG, FG] corresponding with which respective file in OLP
    # you want to grab epochs from. Accounts for differences in my naming schemes ('_' v ' ')
    import glob

    bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                           f'Background{pathidx[0]}/*.wav'))
    bg_names_spaces = [bb.split('/')[-1].split('.')[0][2:] for bb in bg_dir]
    bg_names_nospace = [bb.replace('_', '') for bb in bg_names_spaces if '_' in bb]
    bg_names = bg_names_spaces + bg_names_nospace

    fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                           f'Foreground{pathidx[1]}/*.wav'))
    fg_names_spaces = [ff.split('/')[-1].split('.')[0][2:] for ff in fg_dir]
    fg_names_nospace = [ff.replace('_', '') for ff in fg_names_spaces if '_' in ff]
    fg_names = fg_names_spaces + fg_names_nospace

    bool_bg = weight_df['namesA'].str.contains('|'.join(bg_names))
    bool_fg = weight_df['namesB'].str.contains('|'.join(fg_names))
    filtered_df = weight_df[bool_bg & bool_fg]

    return filtered_df


def get_keyword_sound_type(kw, weight_df=weight_df, pathidx=[2, 3]):
    # When plotting FR vs weights this will find which sound type (BG/FG) your keyword
    # belongs to to return some info to pass to the plotting function
    import glob

    bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Background{pathidx[0]}/*.wav'))
    bg_names_spaces = [bb.split('/')[-1].split('.')[0][2:] for bb in bg_dir]
    bg_names_underscore = [bb.replace(' ', '_') for bb in bg_names_spaces if ' ' in bb]
    bg_names_nospace = [bb.replace(' ', '') for bb in bg_names_spaces if ' ' in bb]
    bg_names = bg_names_spaces + bg_names_underscore + bg_names_nospace

    fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                        f'Foreground{pathidx[1]}/*.wav'))
    fg_names_spaces = [ff.split('/')[-1].split('.')[0][2:] for ff in fg_dir]
    fg_names_underscore = [ff.replace(' ', '_') for ff in fg_names_spaces if ' ' in ff]
    fg_names_nospace = [ff.replace(' ', '') for ff in fg_names_spaces if ' ' in ff]
    fg_names = fg_names_spaces + fg_names_underscore + fg_names_nospace

    if any(kw in nameb for nameb in bg_names) & any(kw in namef for namef in fg_names):
        raise ValueError(f"Your keyword '{kw}' is in BG and FG, be more specific.")
    elif any(kw in nameb for nameb in bg_names):
        if sum([kw in name for name in bg_names_spaces]) > 1:
            print(f"Caution: keyword '{kw}' is found multiple times in BG list, consider being more specific.")
        kw_info = {'sound_type': 'BG', 'xcol': 'BG_FR', 'ycol': 'weightsA', 'keyword': kw}
        df_filtered = weight_df[weight_df['namesA'].str.contains(kw)]
        fn_args = {'df_filtered': df_filtered, 'sound_type': 'BG'}
        print(f"Keyword '{kw}' is a BG. Filtering dataframe.")
    elif any(kw in namef for namef in fg_names):
        if sum([kw in name for name in fg_names_spaces]) > 1:
            print(f"Caution: keyword '{kw}' is found multiple times in FG list, consider being more specific.")
        kw_info = {'sound_type': 'FG', 'xcol': 'FG_FR', 'ycol': 'weightsB', 'keyword': kw}
        df_filtered = weight_df[weight_df['namesB'].str.contains(kw)]
        fn_args = {'df_filtered': df_filtered, 'sound_type': 'FG'}
        print(f"Keyword '{kw}' is a FG. Filtering dataframe.")
    else:
        raise ValueError(f"Your keyword '{kw}' is in neither sound type.")
    kw_info['fn'] = plot_single_psth

    return df_filtered, kw_info, fn_args





###Checking if sounds between Tabor and Hood are the same, they are not.
from pathlib import Path
hdidx,tbridx = 1, 1
for hod, tbr in zip(range(10),range(10)):
    bg_hood_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                               f'Background1/*.wav'))
    bg_tabor_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                               f'Background2/*.wav'))
    bg_hood, bg_tabor = bg_hood_dir[hod], bg_tabor_dir[tbr]
    name_hood, name_tabor = bg_hood.split('/')[-1], bg_tabor.split('/')[-1]
    sfshood, Whood = wavfile.read(bg_hood)
    sfstabor, Wtabor = wavfile.read(bg_tabor)

    spechood = gtgram(Whood, sfshood, 0.02, 0.01, 48, 100, 24000)
    spectabor = gtgram(Wtabor, sfstabor, 0.02, 0.01, 48, 100, 24000)


    fig, ax = plt.subplots(2,1)
    fig.set_figheight(9), fig.set_figwidth(15)
    ax[0].imshow(spechood, aspect='auto', origin='lower')
    ax[0].set_title(f"{bg_hood}")
    ymin, ymax = ax[0].get_ylim()
    ax[0].vlines(50, ymin, ymax, color='white', lw=0.75, ls=':')
    ax[1].imshow(spectabor, aspect='auto', origin='lower')
    ax[1].set_title(f"{bg_tabor}")
    ax[1].vlines(50, ymin, ymax, color='white', lw=0.75, ls=':')


    path = f"/home/hamersky/PSTHcompHODTBR/BGs/"
    # if os.path.isfile(path):
    Path(path).mkdir(parents=True, exist_ok=True)

    plt.savefig(path + f"{name_hood}.png")
    plt.close()

for hod, tbr in zip(range(10),range(5,15,1)):
    fg_hood_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                               f'Foreground1/*.wav'))
    fg_tabor_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                               f'Foreground3/*.wav'))
    fg_hood, fg_tabor = fg_hood_dir[hod], fg_tabor_dir[tbr]
    name_hood, name_tabor = fg_hood.split('/')[-1], fg_tabor.split('/')[-1]
    sfshood, Whood = wavfile.read(fg_hood)
    sfstabor, Wtabor = wavfile.read(fg_tabor)

    spechood = gtgram(Whood, sfshood, 0.02, 0.01, 48, 100, 24000)
    spectabor = gtgram(Wtabor, sfstabor, 0.02, 0.01, 48, 100, 24000)


    fig, ax = plt.subplots(2,1)
    fig.set_figheight(9), fig.set_figwidth(15)
    ax[0].imshow(spechood, aspect='auto', origin='lower')
    ax[0].set_title(f"{fg_hood}")
    ymin, ymax = ax[0].get_ylim()
    ax[0].vlines(50, ymin, ymax, color='white', lw=0.75, ls=':')
    ax[1].imshow(spectabor, aspect='auto', origin='lower')
    ax[1].set_title(f"{fg_tabor}")
    ax[1].vlines(50, ymin, ymax, color='white', lw=0.75, ls=':')


    path = f"/home/hamersky/PSTHcompHODTBR/FGs/"
    # if os.path.isfile(path):
    Path(path).mkdir(parents=True, exist_ok=True)

    plt.savefig(path + f"{name_hood}.png")
    plt.close()


    if animal_id == 'HOD':
        folder_ids = [1,1]
    elif animal_id == 'TBR':
        folder_ids = [2,3]
    elif animal_id == 'ARM':
        folder_ids = [1,2]

    BG, FG = int(BG[:2]), int(FG[:2])


    if sound_type == 'BG':
        path = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                           f'Background{folder_ids[0]}/*.wav'))[BG - 1]
    elif sound_type =='FG':
        path = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/' \
                           f'Foreground{folder_ids[1]}/*.wav'))
    else:
        raise ValueError(f"sound_type must be 'BG' or 'FG', {sound_type} is invalid.")

    xf = 100
