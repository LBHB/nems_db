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
basemodel = "-ref.a-psthfr.s_stategain.S"

batch = 307

outpath = "{}/selectivity_{}/".format(
        '/auto/users/svd/docs/current/pupil_behavior',batch)

statevars = ['pup','fil']
statevars0 = ['pup0','fil0']

sv_len=len(statevars)
if sv_len==2:
    sv_pairs = [(0,1),(1,0)]
else:
    sv_pairs = [(0,1),(1,2),(2,0)]

states = ["st."+".".join(statevars)]
for i,s in enumerate(statevars0):
    st = statevars.copy()
    st[i]=s
    states.append("st."+".".join(st))
modelnames = [loader + s + basemodel + fitter for s in states]


celldata = nd.get_batch_cells(batch=batch)
cellids = celldata['cellid'].tolist()

d = pd.DataFrame(columns=['cellid','state_chan','MI','MI_fil0','MI_pup0','g','d',
                          'r','r_fil0','r_pup0','r_se','r_se_fil0','r_se_pup0'])
for mod_i, m in enumerate(modelnames):
    print('Loading ', m)
    modelspecs = nems_db.params._get_modelspecs(cellids, batch, m, multi='mean')
    for modelspec in modelspecs:
        c = modelspec[0]['meta']['cellid']
        dc = modelspec[0]['phi']['d']
        gain = modelspec[0]['phi']['g']
        meta = ms.get_modelspec_metadata(modelspec)
        state_mod = meta['state_mod']
        state_mod_se = meta['se_state_mod']
        state_chans = meta['state_chans']
        for j, sc in enumerate(state_chans):
            ii = ((d['cellid'] == c) & (d['state_chan'] == sc))
            if np.sum(ii)==0:
                d = d.append({'cellid': c, 'state_chan': sc, 'g': gain[0,j],
                              'd': dc[0,j], 'MI': state_mod[j]},
                            ignore_index=True)
                ii = ((d['cellid'] == c) & (d['state_chan'] == sc))
            if mod_i == 0:
                d.loc[ii, 'MI'] = state_mod[j]
                d.loc[ii, 'r'] = meta['r_test'][0]
                d.loc[ii, 'r_se'] = meta['se_test'][0]
            elif mod_i == 1:
                d.loc[ii, 'MI_pup0'] = state_mod[j]
                d.loc[ii, 'r_pup0'] = meta['r_test'][0]
                d.loc[ii, 'r_se_pup0'] = meta['se_test'][0]
            else:
                d.loc[ii, 'MI_fil0'] = state_mod[j]
                d.loc[ii, 'r_fil0'] = meta['r_test'][0]
                d.loc[ii, 'r_se_fil0'] = meta['se_test'][0]


csv_file = outpath+"selectivity.csv"
d_tuning = pd.read_csv(csv_file)
d_tuning = d_tuning.sort_values('cellid')
tvr = (d_tuning['tar_onset'] - d_tuning['ref_onset']) / \
   (d_tuning['tar_onset'] + d_tuning['ref_onset'])
tvr = (d_tuning['tar_sust'] - d_tuning['ref_sust']) / \
   (d_tuning['tar_sust'] + d_tuning['ref_sust'])
   
   
#save the  data frame
path = '/auto/users/daniela/code/python_ACIC_proj'
d.to_csv(os.path.join(path,'all_nems_data.csv'))

"""
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
"""