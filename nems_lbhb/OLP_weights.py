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
sb.color_palette 
sb.color_palette('colorblind')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sb.color_palette('colorblind')) 
OLP_cell_metrics_db_path='/auto/users/luke/Projects/OLP/NEMS/celldat_A1_v1.h5'

#Decide which cell:
#UPDATE THE LINE BELOW TO POINT TO THE FILE
#rec_file_dir='/auto/data/nems_db/recordings/306/'
#cellid='fre196b-21-1_1417-1233'; rf='fre196b_ec1d319ae74a2d790f3cbda73d46937e588bc791.tgz'
#cellid='fre197c-105-1_705-1024'; rf='fre197c_f94fb643b4cb6380b8eb3286fc30d908a1940ea2.tgz' #Neuron 1 on poster
#rec_file = rec_file_dir + rf

batch=328
cell_df=nd.get_batch_cells(batch)
cell_list=cell_df['cellid'].tolist()
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
    store['df']=df.drop(columns=['get_nrmseR','get_error'])
    store.close()
else:
    store = pd.HDFStore(OLP_cell_metrics_db_path)
    df=store['df']
    store.close()
    

cols=['EP_A','EP_B','IP_A','IP_B','SR','SR_av_std']
cols2=cols+['SinglesMax','MEnh_I','MSupp_I','Rtype']
df[cols2]


Wcols=['namesA','namesB','weightsA','weightsB']
weight_df = pd.concat(df['weight_dfR'].values,keys=df.index)
BGgroups = pd.concat(df['WeightAgroupsR'].values,keys=df.index)
FGgroups = pd.concat(df['WeightBgroupsR'].values,keys=df.index)

plt.figure()
for i in range(len(df)):
    plt.plot(df.iloc[i].weightsR[0,:],df.iloc[i].weightsR[1,:],'.')
plt.xlabel('Background Weights')
plt.ylabel('Foreground Weights')
plt.xlim((-2, 2))
plt.ylim((-2, 2))
plt.gca().set_aspect(1)

#weights=np.concatenate(df.weightsR.values,axis=1)
#weights=weights[:,~np.any(np.isnan(weights),axis=0)]
#plt.figure();  plt.hist2d(weights[0,:],weights[1,:],bins=200)
#plt.xlim((-.5, 1.5)); plt.ylim((-.5, 1.5))
#plt.gca().set_aspect(1)
#plt.xlabel('Background Weights')
#plt.ylabel('Foreground Weights')

#Same plot as the previous one but using the weight dataframe
gi=~np.isnan(weight_df['weightsA']) & ~np.isnan(weight_df['weightsB'])
weights=weights[:,~np.any(np.isnan(weights),axis=0)]
plt.figure();  plt.hist2d(weight_df['weightsA'][gi],weight_df['weightsB'][gi],bins=200)
plt.xlim((-.5, 1.5)); plt.ylim((-.5, 1.5))
plt.gca().set_aspect(1)
plt.xlabel('Background Weight')
plt.ylabel('Foreground Weight')

#WARNING, LEGEND BACKWARDS???!
#plt.figure();  plt.hist(weights.T,bins=400,histtype='step')
#plt.legend(('Background','Foreground'))
#plt.xlim((-1, 2));

bins=np.arange(-2,2,.05)
plt.figure();
plt.hist(weights[0,:],bins=bins,histtype='step')
plt.hist(weights[1,:],bins=bins,histtype='step')
plt.legend(('Background','Foreground'))
plt.xlabel('Weight')

plt.figure();  plt.hist(np.diff(weights,axis=0).T,bins=400,histtype='step')
plt.xlim((-2, 2));
plt.xlabel('Paired Foreground - Background')

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

#To plot PSTHs and weight model
cellid='ARM031a-33-1';
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

#from pdb import set_trace
#set_trace() 