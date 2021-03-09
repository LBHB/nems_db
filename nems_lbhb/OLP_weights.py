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

OLP_cell_metrics_db_path='/auto/users/luke/Projects/OLP/NEMS/celldat1.h5'

#Decide which cell:
#UPDATE THE LINE BELOW TO POINT TO THE FILE
#rec_file_dir='/auto/data/nems_db/recordings/306/'
#cellid='fre196b-21-1_1417-1233'; rf='fre196b_ec1d319ae74a2d790f3cbda73d46937e588bc791.tgz'
#cellid='fre197c-105-1_705-1024'; rf='fre197c_f94fb643b4cb6380b8eb3286fc30d908a1940ea2.tgz' #Neuron 1 on poster
#rec_file = rec_file_dir + rf

batch=328
cell_df=nd.get_batch_cells(batch)
#cell_list=cell_df['cellid'].tolist()[-2:]
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
    
    os.makedirs(os.path.dirname(OLP_cell_metrics_db_path),exist_ok=True)
    store = pd.HDFStore(OLP_cell_metrics_db_path)
    store['df']=df
    store.close()
else:
    store = pd.HDFStore(OLP_cell_metrics_db_path)
    df=store['df']
    store.close()
    

cols=['EP_A','EP_B','IP_A','IP_B','SR','SR_av_std']
df[cols+['SinglesMax','MEnh_I','MSupp_I','Rtype','inds']]


df=df.apply(ts.calc_psth_weight_resp,axis=1)   

