#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:59:38 2018

@author: svd
"""

import os
import sys
import pandas as pd
import scipy.signal as ss

import numpy as np
import matplotlib.pyplot as plt
import nems_lbhb.stateplots as stateplots
import nems_lbhb.xform_wrappers as nw
import nems.db as nd
import nems_db.params

import nems.recording as recording
import nems.epoch as ep
import nems.plots.api as nplt
import nems.modelspec as ms

# User parameters:
loader = "psth.fs20.pup-ld-"
fitter = "_jk.nf10-init.st-basic"
basemodel = "-ref-psthfr.s_stategain.S"

batch = 307

#statevars = ['pup','beh','far.hit']
#statevars0 = ['pup0','beh0','far0.hit0']
statevars = ['pup','beh']
statevars0 = ['pup0','beh0']

sv_len=len(statevars)
if sv_len==2:
    sv_pairs = [(0,1),(1,0)]
else:
    sv_pairs = [(0,1),(1,2),(2,0)]

states = []
for i,s in enumerate(statevars0):
    st = statevars.copy()
    st[i]=s
    states.append("st."+".".join(st))
states.append("st."+".".join(statevars))
modelnames = [loader + s + basemodel + fitter for s in states]

outpath = "{}/selectivity_{}/".format(
        '/auto/users/svd/docs/current/pupil_behavior',batch)
csv_file = outpath+"selectivity.csv"
d_tuning = pd.read_csv(csv_file)
d_tuning = d_tuning.sort_values('cellid')
tvr = (d_tuning['tar_onset'] - d_tuning['ref_onset']) / \
   (d_tuning['tar_onset'] + d_tuning['ref_onset'])
tvr = (d_tuning['tar_sust'] - d_tuning['ref_sust']) / \
   (d_tuning['tar_sust'] + d_tuning['ref_sust'])

meta=['r_test', 'se_test', 'state_mod']
stats_keys=[]
state_mod = []
r_test = []
se_test = []

celldata = nd.get_batch_cells(batch=batch)
cellids = celldata['cellid'].tolist()

for mod_i, m in enumerate(modelnames):

    d = nems_db.params.fitted_params_per_batch(batch, m, meta=meta,
                                               stats_keys=stats_keys)
    d = d.reindex(sorted(d.columns), axis=1)
    state_mod.append(np.stack(d.loc['meta--state_mod']))
    r_test.append(np.stack(d.loc['meta--r_test']))
    se_test.append(np.stack(d.loc['meta--se_test']))


cellids = list(d.columns)
state_mod = np.stack(state_mod)
r_test = np.stack(r_test)
se_test = np.stack(se_test)

u_state_mod = state_mod[[sv_len],:] - state_mod[:sv_len, :]
u_r_test = r_test[[sv_len],:] - r_test[:sv_len,:]
u_r_good = u_r_test > se_test[:sv_len,:]

plt.close('all')
plt.figure()


for i,p in enumerate(sv_pairs):
    ax = plt.subplot(len(sv_pairs),3,i*3+1)
    stateplots.beta_comp(u_r_test[p[0],:], u_r_test[p[1],:],
                         n1=statevars[p[0]], n2=statevars[p[1]],
                         title='u r_test', hist_range=[-0.05, 0.15],
                         ax=ax, highlight=(u_r_good[p[0],:] | u_r_good[p[1],:]))

    ax = plt.subplot(len(sv_pairs),3,i*3+2)
    stateplots.beta_comp(u_state_mod[p[0],:,p[0]+1], u_state_mod[p[1],:,p[1]+1],
                         n1=statevars[p[0]], n2=statevars[p[1]],
                         title='u state_mod', hist_range=[-0.6, 0.6],
                         ax=ax, highlight=(u_r_good[p[0],:] | u_r_good[p[1],:]))
#    plt.plot(tvr[np.logical_not(u_r_good[p[0],:])],
#             state_mod[-1,np.logical_not(u_r_good[p[0],:]),p[0]+1],
#             '.', color='LightGray')
#    plt.plot(tvr[u_r_good[p[0],:]], state_mod[-1,u_r_good[p[0],:],p[0]+1], 'k.')
#    plt.xlabel('tvr')
#    plt.ylabel("raw state_mod " + statevars[p[0]])

    ax = plt.subplot(len(sv_pairs),3,i*3+3)
    plt.plot(tvr[np.logical_not(u_r_good[p[0],:])],
             u_state_mod[p[0],np.logical_not(u_r_good[p[0],:]),p[0]+1],
             '.', color='LightGray')
    plt.plot(tvr[u_r_good[p[0],:]], u_state_mod[p[0],u_r_good[p[0],:],p[0]+1], 'k.')

    plt.xlabel('tvr')
    plt.ylabel("u_state_mod " + statevars[p[0]])
plt.tight_layout()

d = pd.DataFrame()
d['cellid']=cellids
d['p_state_mod']=u_state_mod[0,:,1]
d['b_state_mod']=u_state_mod[1,:,2]

pmod = u_state_mod[0,:,1]
bmod = u_state_mod[1,:,2]

ii=[]
for i in np.argsort(np.array(bmod)):
    if u_r_good[1,i]:
        ii.append(i)

r = np.array([np.array(eval(_r.replace('nan','np.nan'))) for _r in d_tuning['ref_mean']])
t = np.array([np.array(eval(_t.replace('nan','np.nan'))) for _t in d_tuning['tar_mean']])

spont=np.nanmean(np.abs(np.concatenate((r[:,:30],t[:,:30]),axis=1)),
                                        axis=1, keepdims=True)


r-=spont
t-=spont

r=ss.decimate(r,2,axis=1)
t=ss.decimate(t,2,axis=1)

t=t[:,:r.shape[1]]
m=np.nanmax(np.abs(np.concatenate((r,t),axis=1)),axis=1)
m=np.reshape(m,(-1, 1))
r /= m
t /= m
plt.figure()
plt.subplot(1,3,1)
plt.imshow(r[ii,:], aspect='auto', clim=[-1, 1])
plt.title('ref')
plt.subplot(1,3,2)
plt.imshow(t[ii,:], aspect='auto', clim=[-1, 1])
plt.title('tar')
plt.subplot(1,3,3)
plt.imshow(t[ii,:]-r[ii,:], aspect='auto', clim=[-1, 1])
plt.title('tar-ref')

plt.figure()
plt.plot(np.array(tvr)[ii])
plt.plot(pmod[ii])
plt.plot(bmod[ii])