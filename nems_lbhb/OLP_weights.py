#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:58:16 2020

@author: luke
"""
#import SPO_helpers as sp
import nems_lbhb.TwoStim_helpers as ts
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


#Decide which cell:
#UPDATE THE LINE BELOW TO POINT TO THE FILE
#rec_file_dir='/auto/data/nems_db/recordings/306/'
#cellid='fre196b-21-1_1417-1233'; rf='fre196b_ec1d319ae74a2d790f3cbda73d46937e588bc791.tgz'
#cellid='fre197c-105-1_705-1024'; rf='fre197c_f94fb643b4cb6380b8eb3286fc30d908a1940ea2.tgz' #Neuron 1 on poster
#rec_file = rec_file_dir + rf

batch=329
if batch == 328:
    OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/a1_new_celldat1.h5'
if batch == 329:
    OLP_cell_metrics_db_path = '/auto/users/hamersky/olp_analysis/peg_new_celldat1.h5'
cell_df=nd.get_batch_cells(batch)
cell_list=cell_df['cellid'].tolist()
cell_list = [cell for cell in cell_list if cell[:3] != 'HOD']
fs=100
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


# PSTH metrics that have to do with one stimulus at a time
if True:
    metrics=[]
    for cellid in cell_list:
        metrics_=ts.calc_psth_metrics(batch,cellid)
        print('****rAAm: {} rBBm: {}'.format(metrics_['rAAm'],metrics_['rBBm']))
        metrics.append(metrics_)
    
    df=pd.DataFrame(data=metrics)
    df['modelspecname']='dlog_fir2x15_lvl1_dexp1'
    df['cellid']=cell_list
    df = df.set_index('cellid')
    
    df = df.apply(ts.type_by_psth, axis=1);
    df['batch']=batch
    
    df=df.apply(ts.calc_psth_weight_resp,axis=1,fs=fs)

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
    store['df']=df_store.drop(columns=['get_nrmseR'])
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

Wcols=['namesA','namesB','weightsA','weightsB']
weight_df = pd.concat(df['weight_dfR'].values,keys=df.index)
BGgroups = pd.concat(df['WeightAgroupsR'].values,keys=df.index)
FGgroups = pd.concat(df['WeightBgroupsR'].values,keys=df.index)

#Add suppression column to weights_df
supp_df = pd.DataFrame()
for cll in df.index:
    supp = df.loc[cll,'suppression']
    names = [ts.get_sep_stim_names(sn) for sn in df.loc[cll,'pair_names']]
    BGs, FGs = [rr[0] for rr in names], [qq[1] for qq in names]
    cell_df = pd.DataFrame({'suppression': supp,
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

#Heatmap version of BG/FG weight scatter
weights=np.concatenate(df.weightsR.values,axis=1)
weights=weights[:,~np.any(np.isnan(weights),axis=0)]
plt.figure();  plt.hist2d(weights[0,:],weights[1,:],bins=bins)
plt.xlim((-.5, 1.5)); plt.ylim((-.5, 1.5))
plt.gca().set_aspect(1)
plt.xlabel('Background Weights')
plt.ylabel('Foreground Weights')
plt.title(f"{titles}")

#Same plot as the previous one but using the weight dataframe
gi=~np.isnan(weight_df['weightsA']) & ~np.isnan(weight_df['weightsB'])
# weights=weights[:,~np.any(np.isnan(weights),axis=0)]
plt.figure();  plt.hist2d(weight_df['weightsA'][gi],weight_df['weightsB'][gi],bins=bins)
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
plt.figure();  plt.hist(np.diff(weights,axis=0).T,bins=bins,histtype='step')
plt.xlim((-2, 2));
plt.xlabel('Paired Foreground - Background')
plt.title(f"{titles}")

## Plots based on differences
# Scatterplot of range of weights over constant bg
plt.figure()
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
        nr=1
    f, ax = plt.subplots(nr,1)
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
    ax[0].legend(('Bg, weight={:.2f}'.format(this_cell_stim.weightsA),
      'Fg, weight={:.2f}'.format(this_cell_stim.weightsB),
      'Both',
      'Weight Model, r={:.2f}'.format(this_cell_stim.r)))
    ax[0].set_title(cellid_and_stim_str)
    
    if plot_error:
        ax[1].plot(time,this_cell_stim['get_error']()/np.sqrt(this_cell_stim['nf']))

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
         'weight': ww[0, :]
         })
    fg_weights = pd.DataFrame(
        {'name': fgs,
         'type': 'fg',
         'idx': fgs_idx,
         'weight': ww[1, :]
         })

    bgfg_weight = pd.concat([bg_weights, fg_weights], ignore_index=True)
    sound_weights = sound_weights.append(bgfg_weight, ignore_index=True)
sound_weights = sound_weights.sort_values(['type', 'idx'], ascending=[True, True])

PEG_path = Path('/auto/users/hamersky/olp_analysis/PEG_new_weights')
A1_path = Path('/auto/users/hamersky/olp_analysis/A1_new_weights')
if PEG_path.parent.exists() is False and batch == 329:
    PEG_path.parent.mkdir()
if A1_path.parent.exists() is False and batch == 328:
    A1_path.parent.mkdir()
if batch == 329:
    jl.dump(sound_weights, PEG_path)
if batch == 328:
    jl.dump(sound_weights, A1_path)

if batch == 329:
    peg_sound_weights = jl.load(PEG_path)
if batch == 328:
    a1_sound_weights = jl.load(A1_path)

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

bg_count = sound_weights.loc[sound_weights['type'] == 'bg']['name'].nunique()
fg_count = sound_weights.loc[sound_weights['type'] == 'fg']['name'].nunique()

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
plt.figure()
g = sns.regplot(x='weightsA', y='suppression', data=weight_df, color='black')
g.set_xlabel('BG Weight')
g.set_title(f'{titles} -- BG', fontweight='bold')
plt.figure()
g = sns.regplot(x='weightsB', y='suppression', data=weight_df, color='black')
g.set_xlabel('FG Weight')
g.set_title(f'{titles} -- FG', fontweight='bold')


#Plots all weights for each cell -- Messy and doesn't look good, maybe order it
fig, ax = plt.subplots()
weighties = copy.copy(weight_df)
cell_weights = weighties.reset_index()
g = sns.stripplot(x='cellid', y='weightsA', color='deepskyblue', data=cell_weights, ax=ax)
g = sns.stripplot(x='cellid', y='weightsB', color='yellowgreen', data=cell_weights, ax=ax)
ax.set_ylim(-3, 2.5)
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
rr = {}
for shuf in shuffles:
    reg_results = neur_stim_reg(reg_df, shuf)
    rr[shuf] = reg_results.rsquared

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