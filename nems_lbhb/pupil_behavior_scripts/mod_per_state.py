#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:59:38 2018

@author: svd

Call this to get the state-dep results:

d = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)

Special order requred for state_list (which is part of modelname that
defines the state variables):

state_list = [both shuff, pup_shuff, other_shuff, full_model]

Here's how to set parameters:

batch = 307  # A1 SUA and MUA
batch = 309  # IC SUA and MUA
batch = 295  # old (Slee) IC data
batch = 311  # A1 old (SVD) data -- on BF
batch = 312  # A1 old (SVD) data -- off BF

# pup vs. active/passive
state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
basemodel = "-ref-psthfr.s_sdexp.S"
d = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)
d.to_csv('d_fil_307.csv')

# pup vs. active/passive with spont+baseline separated
batch = 309  # IC SUA and MUA
state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
basemodel = "-ref.e-psthfr.s_stategain.S.s"
d = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)
d.to_csv('d_sbg_309.csv')

# fil only
state_list = ['st.fil0','st.fil']
basemodel = "-ref-psthfr.s_stategain.S"
loader = "psth.fs20-ld-"
d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                      basemodel=basemodel, loader=loader)
d.to_csv('d_307_pb.csv')

# example modelnames:
psth.fs20.pup-ld-st.fil0-ref-psthfr.s_sdexp.S_jk.nf20-basic
psth.fs20-ld-st.fil-ref-psthfr.s_sdexp.S_jk.nf20-basic

# beh only
batch = 311  # A1 old (SVD) data -- on BF
batch = 305  # IC PTD data (DS)
state_list = ['st.beh0','st.beh']
basemodel = "-ref-psthfr.s_stategain.S"
loader = "psth.fs20-ld-"
fitter = "_jk.nf20-basic"
d = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                      basemodel=basemodel, loader=loader)

# pup vs. per file
state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']
basemodel = "-ref-psthfr.s_sdexp.S"
d = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)

# pup vs. per 1/2 file
state_list = ['st.pup0.hlf0','st.pup0.hlf','st.pup.hlf0','st.pup.hlf']
basemodel = "-ref-psthfr.s_sdexp.S"
d = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)

# pup vs. performance
state_list = ['st.pup0.beh.far0.hit0','st.pup0.beh.far.hit',
              'st.pup.beh.far0.hit0','st.pup.beh.far.hit']
basemodel = "-ref.a-psthfr.s_sdexp.S"
d = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)

# pup vs. pre/post passive
state_list = ['st.pup0.pas0','st.pup0.pas',
              'st.pup.pas0','st.pup.pas']
basemodel = "-ref-pas-psthfr.s_sdexp.S"
d = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)

"""

import os
import sys
import pandas as pd
import scipy.signal as ss
import scipy.stats as st

import numpy as np
import matplotlib.pyplot as plt
import nems_lbhb.stateplots as stateplots
import nems.db as nd
import nems_db.params

import nems.recording as recording
import nems.epoch as ep
import nems.plots.api as nplt
import nems.modelspec as ms


def get_model_results_per_state_model(batch=307, state_list=None,
                      loader = "psth.fs20.pup-ld-",
                      fitter = "_jk.nf20-basic",
                      basemodel = "-ref-psthfr.s_sdexp.S"):
    """
    loader = "psth.fs20.pup-ld-"
    fitter = "_jk.nf20-basic"
    basemodel = "-ref-psthfr.s_sdexp.S"
    state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']

    d=get_model_results_per_state_model(batch=307, state_list=state_list,
                                        loader=loader,fitter=fitter,
                                        basemodel=basemodel)

    state_list defaults to
       ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
    """

    if state_list is None:
        state_list = ['st.pup0.beh0', 'st.pup0.beh',
                      'st.pup.beh0', 'st.pup.beh']

    modelnames = [loader + s + basemodel + fitter for s in state_list]

    celldata = nd.get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()
    isolation = [nd.get_isolation(cellid=c, batch=batch).loc[0, 'min_isolation'] for c in cellids]

    if state_list[-1].endswith('fil') or state_list[-1].endswith('pas'):
        include_AP = True
    else:
        include_AP = False

    d = pd.DataFrame(columns=['cellid', 'modelname', 'state_sig',
                              'state_chan', 'MI', 'isolation',
                              'r', 'r_se', 'd', 'g', 'sp', 'state_chan_alt'])

    for mod_i, m in enumerate(modelnames):
        print('Loading modelname: ', m)
        modelspecs = nems_db.params._get_modelspecs(cellids, batch, m, multi='mean')

        for modelspec in modelspecs:
            meta = ms.get_modelspec_metadata(modelspec)
            c = meta['cellid']
            iso = isolation[cellids.index(c)]
            state_mod = meta['state_mod']
            state_mod_se = meta['se_state_mod']
            state_chans = meta['state_chans']
            dc = modelspec[0]['phi']['d']
            gain = modelspec[0]['phi']['g']
            sp = modelspec[0]['phi'].get('sp', np.zeros(gain.shape))
            if dc.ndim > 1:
                dc = dc[0, :]
                gain = gain[0, :]
                sp = sp[0, :]
            a_count = 0
            p_count = 0
            for j, sc in enumerate(state_chans):
                r = {'cellid': c, 'state_chan': sc, 'modelname': m,
                     'isolation': iso,
                     'state_sig': state_list[mod_i],
                     'g': gain[j], 'd': dc[j], 'sp': sp[j],
                     'MI': state_mod[j],
                     'r': meta['r_test'][0], 'r_se': meta['se_test'][0]}
                d = d.append(r, ignore_index=True)
                l = len(d) - 1

                if include_AP and sc.startswith("FILE_"):
                    siteid = c.split("-")[0]
                    fn = "%" + sc.replace("FILE_","") + "%"
                    sql = "SELECT * FROM gDataRaw WHERE cellid=%s" +\
                       " AND parmfile like %s"
                    dcellfile = nd.pd_query(sql, (siteid, fn))
                    if dcellfile.loc[0]['behavior'] == 'active':
                        a_count += 1
                        d.loc[l,'state_chan_alt'] = "ACTIVE_{}".format(a_count)
                    else:
                        p_count += 1
                        d.loc[l,'state_chan_alt'] = "PASSIVE_{}".format(p_count)
                else:
                    d.loc[l,'state_chan_alt'] = d.loc[l,'state_chan']



    #d['r_unique'] = d['r'] - d['r0']
    #d['MI_unique'] = d['MI'] - d['MI0']

    return d



def get_model_results(batch=307, state_list=None,
                      loader = "psth.fs20.pup-ld-",
                      fitter = "_jk.nf20-basic",
                      basemodel = "-ref-psthfr.s_sdexp.S"):
    """
    loader = "psth.fs20.pup-ld-"
    fitter = "_jk.nf20-basic"
    basemodel = "-ref-psthfr.s_sdexp.S"

    state_list defaults to
       ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
    """

    if state_list is None:
        state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']

    modelnames = [loader + s + basemodel + fitter for s in state_list]

    celldata = nd.get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()

    d = pd.DataFrame(columns=['cellid','modelname','state_sig','state_sig0',
                              'state_chan','MI',
                              'r','r_se','d','g','MI0','r0','r0_se'])

    for mod_i, m in enumerate(modelnames):
        print('Loading modelname: ', m)
        modelspecs = nems_db.params._get_modelspecs(cellids, batch, m, multi='mean')

        for modelspec in modelspecs:
            meta = ms.get_modelspec_metadata(modelspec)
            c = meta['cellid']
            state_mod = meta['state_mod']
            state_mod_se = meta['se_state_mod']
            state_chans = meta['state_chans']
            dc = modelspec[0]['phi']['d']
            gain = modelspec[0]['phi']['g']
            for j, sc in enumerate(state_chans):
                ii = ((d['cellid'] == c) & (d['state_chan'] == sc))
                if np.sum(ii) == 0:
                    r = {'cellid': c, 'state_chan': sc}
                    d = d.append(r, ignore_index=True)
                    ii = ((d['cellid'] == c) & (d['state_chan'] == sc))
                if mod_i == 3:
                    # full model
                    d.loc[ii, ['modelname', 'state_sig', 'g', 'd', 'MI',
                               'r', 'r_se']] = \
                       [m, state_list[mod_i], gain[0, j], dc[0, j],
                        state_mod[j], meta['r_test'][0], meta['se_test'][0]]
                elif (mod_i == 1) & (sc == 'pupil'):
                    # pupil shuffled model
                    d.loc[ii, ['state_sig0', 'MI0', 'r0', 'r0_se']] = \
                       [state_list[mod_i], state_mod[j],
                        meta['r_test'][0], meta['se_test'][0]]
                elif (mod_i == 0) & (sc == 'baseline'):
                    d.loc[ii, ['state_sig0', 'r0', 'r0_se']] = \
                       [state_list[mod_i],
                        meta['r_test'][0], meta['se_test'][0]]
                elif (mod_i == 2) & (sc not in ['baseline', 'pupil']):
                    # pupil shuffled model
                    d.loc[ii, ['state_sig0', 'MI0', 'r0', 'r0_se']] = \
                       [state_list[mod_i], state_mod[j],
                        meta['r_test'][0], meta['se_test'][0]]

    d['r_unique'] = d['r'] - d['r0']
    d['MI_unique'] = d['MI'] - d['MI0']

    return d


def hlf_analysis(df, state_list, title=None, norm_sign=True, states=None):
    """
    df: dataframe output by get_model_results_per_state_model()
    state_list: list of state keywords used to generate df. e.g.:

    state_list = ['st.pup0.hlf0','st.pup0.hlf','st.pup.hlf0','st.pup.hlf']
    basemodel = "-ref-psthfr.s_sdexp.S"
    batch=307
    df = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)

    state_list = ['st.pup0.fil0','st.pup0.fil','st.pup.fil0','st.pup.fil']
    basemodel = "-ref-psthfr.s_sdexp.S"
    batch=307
    df = get_model_results_per_state_model(batch=batch, state_list=state_list, basemodel=basemodel)
    """

    # figure out what cells show significant state ef
    da = df[df['state_chan']=='pupil']
    dp = pd.pivot_table(da, index='cellid',columns='state_sig',values=['r','r_se'])
    #dp = da.pivot(index='cellid',columns='state_sig',values=['r','r_se'])
    #dp = da.pivot(index='cellid',columns='state_sig',values=['r'])

    dr = dp['r'].copy()
    dr['b_unique'] = dr[state_list[3]]**2 - dr[state_list[2]]**2
    dr['p_unique'] = dr[state_list[3]]**2 - dr[state_list[1]]**2
    dr['bp_common'] = dr[state_list[3]]**2 - dr[state_list[0]]**2 - dr['b_unique'] - dr['p_unique']
    dr['bp_full'] = dr['b_unique']+dr['p_unique']+dr['bp_common']
    dr['null']=dr[state_list[0]]**2 * np.sign(dr[state_list[0]])
    dr['full']=dr[state_list[3]]**2 * np.sign(dr[state_list[3]])

    #dr['sig']=((dp['r'][state_list[1]]-dp['r'][state_list[0]]) >
    #     (dp['r_se'][state_list[1]]+dp['r_se'][state_list[0]]))
    dr['sig']=((dp['r'][state_list[3]]-dp['r'][state_list[2]]) >
         (dp['r_se'][state_list[3]]+dp['r_se'][state_list[2]]))

    dfull = df[df['state_sig']==state_list[3]]
    dpup = df[df['state_sig']==state_list[2]]

    dp = pd.pivot_table(dfull, index='cellid',columns='state_chan',values=['MI'])
    dp0 = pd.pivot_table(dpup, index='cellid',columns='state_chan',values=['MI'])
    if states is not None:
        pass
    elif state_list[-1].endswith('fil'):
        states = ['PASSIVE_0',  'ACTIVE_1','PASSIVE_1',  'ACTIVE_2']
    else:
        states = ['PASSIVE_0_A','PASSIVE_0_B', 'ACTIVE_1_A','ACTIVE_1_B',
                  'PASSIVE_1_A','PASSIVE_1_B', 'ACTIVE_2_A','ACTIVE_2_B',
                  'PASSIVE_2_A','PASSIVE_2_B']
        #states = ['PASSIVE_0_A','PASSIVE_0_B', 'ACTIVE_1_A','ACTIVE_1_B',
        #          'PASSIVE_1_A','PASSIVE_1_B']
    dMI=dp['MI'].loc[:,states]
    dMI0=dp0['MI'].loc[:,states]

    MI = dMI.values
    MI0 = dMI0.values
    MIu = MI - MI0
    sig = dr['sig'].values

    if state_list[-1].endswith('fil'):
        # weigh post by 1/2 since pre is fixed at 0 and should get 1/2 weight
        ff = np.isfinite(MI[:,-1]) & sig
        ffall = np.isfinite(MI[:,-1])
        MI[np.isnan(MI)] = 0
        MIu[np.isnan(MIu)] = 0
        MI0[np.isnan(MI0)] = 0
        a = [1, 3]
        #sg = np.sign(MI[:,1:2]/2 + MI[:,3:4]/2 - MI[:,2:3]/2)
        #sg = np.sign(MI[:,1:2] - MI[:,2:3]/2)
    else:
        ff = np.isfinite(MI[:,-1]) & sig
        ffall = np.isfinite(MI[:,-1])
        MI[np.isnan(MI)] = 0
        MIu[np.isnan(MIu)] = 0
        MI0[np.isnan(MI0)] = 0
        #MI[:,0]=0
        #MIu[:,0]=0
        #MI0[:,0]=0

        if len(states) >= 8:
            a = [2, 3, 6, 7]
        else:
            a = [2, 3]

    p = np.zeros(MI.shape[1], dtype=bool)
    p[a] = True
    n = ~p
    #n[0] = False
    print('p: ', p)
    print('n: ', n)
    if norm_sign:
        b = np.mean(MI[:, n], axis=1, keepdims=True)
        MI -= b
        MIu -= b
        MI0 -= b
        sg = np.sign(np.mean(MI[:, p], axis=1, keepdims=True) -
                     np.mean(MI[:, n], axis=1, keepdims=True))
        MI *= sg
        MIu *= sg
        MI0 *= sg

    MIall = MI[ffall,:]
    MIuall = MIu[ffall,:]
    MI0all = MI0[ffall,:]
    MI = MI[ff,:]
    MIu = MIu[ff,:]
    MI0 = MI0[ff,:]

    plt.figure(figsize=(8,8))
    plt.subplot(3,1,1)
    plt.plot(MI.T, linewidth=0.5)
    plt.ylabel('raw MI per cell')
    plt.xticks(np.arange(len(states)),states)
    if title is None:
        plt.title('{} (n={}/{} sig MI)'.format(state_list[-1],np.sum(ff),np.sum(ffall)))
    else:
        plt.title('{} (n={}/{} sig MI)'.format(title,np.sum(ff),np.sum(ffall)))

    plt.subplot(3,1,2)
    plt.plot(MIu.T, linewidth=0.5)
    plt.ylabel('unique MI per cell')
    plt.xticks(np.arange(len(states)),states)

    plt.subplot(3,1,3)
    plt.plot(np.nanmean(MIu, axis=0), 'r-', linewidth=2)
    plt.plot(np.nanmean(MI, axis=0), 'r--', linewidth=2)
    #plt.plot(np.mean(MIuall, axis=0), 'r--', linewidth=1)
    plt.plot(np.nanmean(MI0, axis=0), 'b--', linewidth=1)
    plt.legend(('MIu','MIraw','MIpup'))
    plt.plot(np.nanmean(MIu, axis=0), 'k.', linewidth=2)
    plt.plot(np.nanmean(MI, axis=0), 'k.', linewidth=2)
    #plt.plot(np.mean(MIuall, axis=0), 'r--', linewidth=1)
    plt.plot(np.nanmean(MI0, axis=0), 'k.', linewidth=1)
    #plt.plot(np.mean(MI0all, axis=0), 'b--', linewidth=1)
    plt.plot(np.arange(len(states)), np.zeros(len(states)), 'k--', linewidth=1)
    plt.ylabel('mean MI')
    plt.xticks(np.arange(len(states)), states)
    plt.xlabel('behavioral block')

    plt.tight_layout()

    return dMI, dMI0


def hlf_wrapper(use_hlf=True):
    """
    batch = 307  # A1 SUA and MUA
    batch = 309  # IC SUA and MUA
    """

    # pup vs. active/passive
    if use_hlf:
        state_list = ['st.pup0.hlf0', 'st.pup0.hlf', 'st.pup.hlf0', 'st.pup.hlf']
        #state_list = ['st.pup0.far0.hit0.hlf0','st.pup0.far0.hit0.hlf',
        #              'st.pup.far.hit.hlf0','st.pup.far.hit.hlf']
        states = ['PASSIVE_0_A','PASSIVE_0_B', 'ACTIVE_1_A','ACTIVE_1_B',
                  'PASSIVE_1_A','PASSIVE_1_B', 'ACTIVE_2_A','ACTIVE_2_B',
                  'PASSIVE_2_A','PASSIVE_2_B']
        #states = ['PASSIVE_0_A','PASSIVE_0_B', 'ACTIVE_1_A','ACTIVE_1_B',
        #          'PASSIVE_1_A','PASSIVE_1_B']
    else:
        state_list = ['st.pup0.fil0', 'st.pup0.fil', 'st.pup.fil0', 'st.pup.fil']
        states = ['PASSIVE_0',  'ACTIVE_1', 'PASSIVE_1',
                  'ACTIVE_2', 'PASSIVE_2']
    basemodels = ["-ref-psthfr.s_sdexp.S"]
    #basemodels = ["-ref-psthfr.s_sdexp.S","-ref-psthfr.s_sdexp.S"]
    #basemodels = ["-ref.a-psthfr.s_sdexp.S"]
    batches = [307, 309]
    basemodel = basemodels[0]
    batch = batches[0]

    plt.close('all')
    for batch in batches:
        for basemodel in basemodels:
            df = get_model_results_per_state_model(
                    batch=batch, state_list=state_list, basemodel=basemodel)
            title = "{} {} batch {} keep sgn".format(basemodel,state_list[-1],batch)
            hlf_analysis(df, state_list, title=title, norm_sign=True, states=states);


def aud_vs_state(df, nb=5, title=None, state_list=None, colors=['r','g','b','k']):
    """
    d = dataframe output by get_model_results_per_state_model()
    nb = number of bins
    """
    if state_list is None:
        state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']

    f = plt.figure(figsize=(4,6))

    da = df[df['state_chan']=='active']

    dp = da.pivot(index='cellid',columns='state_sig',values=['r','r_se'])

    dr = dp['r'].copy()

    if len(state_list)==4:
        dr['b_unique'] = dr[state_list[3]]**2 - dr[state_list[2]]**2
        dr['p_unique'] = dr[state_list[3]]**2 - dr[state_list[1]]**2
        dr['bp_common'] = dr[state_list[3]]**2 - dr[state_list[0]]**2 - dr['b_unique'] - dr['p_unique']
        dr['bp_full'] = dr['b_unique'] + dr['p_unique'] + dr['bp_common']
        dr['null']=dr[state_list[0]]**2 * np.sign(dr[state_list[0]])
        dr['full']=dr[state_list[3]]**2 * np.sign(dr[state_list[3]])

        dr['sig']=((dp['r'][state_list[3]]-dp['r'][state_list[0]]) > \
             (dp['r_se'][state_list[3]]+
              dp['r_se'][state_list[0]]))

        #dm = dr.loc[dr['sig'].values,['null','full','bp_common','p_unique','b_unique']]
        dm = dr.loc[:,['null','full','bp_common','b_unique','p_unique','sig']]
        dm = dm.sort_values(['null'])
        mfull=dm[['null','full','bp_common','b_unique','p_unique','sig']].values

    elif len(state_list)==2:
        dr['bp_common'] = dr[state_list[1]]**2 - dr[state_list[0]]**2
        dr['b_unique'] = dr['bp_common']*0
        dr['p_unique'] = dr['bp_common']*0

        dr['bp_full'] = dr['b_unique'] + dr['p_unique'] + dr['bp_common']
        dr['null']=dr[state_list[0]]**2 * np.sign(dr[state_list[0]])
        dr['full']=dr[state_list[1]]**2 * np.sign(dr[state_list[1]])

        dr['sig']=((dp['r'][state_list[1]]-dp['r'][state_list[0]]) > \
             (dp['r_se'][state_list[1]]+
              dp['r_se'][state_list[0]]))
        dr['cellid'] = dp['r'][state_list[1]].index
        #dm = dr.loc[dr['sig'].values,['null','full','bp_common','p_unique','b_unique']]
        dm = dr.loc[:,['cellid','null','full','bp_common','b_unique','p_unique','sig']]
        dm = dm.sort_values(['null'])
        mfull=dm[['null','full','bp_common','b_unique','p_unique','sig']].values
        cellids=dm['cellid'].to_list()

        big_idx = mfull[:,1]-mfull[:,0]>0.2
        for i,b in enumerate(big_idx):
            if b:
                print('{} : {:.3f} - {:.3f}'.format(cellids[i],mfull[i,0],mfull[i,1]))


    if nb > 0:
        stepsize = mfull.shape[0]/nb
        mm=np.zeros((nb,mfull.shape[1]))
        for i in range(nb):
            #x0=int(np.floor(i*stepsize))
            #x1=int(np.floor((i+1)*stepsize))
            #mm[i,:]=np.mean(m[x0:x1,:],axis=0)
            x01=(mfull[:,0]>i/nb) & (mfull[:,0]<=(i+1)/nb)
            if np.sum(x01):
                mm[i,:]=np.nanmean(mfull[x01,:],axis=0)

        print(np.round(mm,3))

        m = mm.copy()
    else:
        # alt to look at each cell individually:
        m = mfull.copy()

    mall = np.nanmean(mfull, axis=0, keepdims=True)

    # remove sensory component, which swamps everything else
    mall = mall[:, 2:]
    mb=m[:,2:]

    ax1 = plt.subplot(3,1,1)
    stateplots.beta_comp(mfull[:,0],mfull[:,1],n1='State independent',n2='Full state-dep',
                         ax=ax1, highlight=dm['sig'], hist_range=[-0.1, 1])

    ax2 = plt.subplot(3,1,2)
    width=0.8
    #ind = m[:,0]
    mplots=np.concatenate((mall, mb), axis=0)
    ind = np.arange(mplots.shape[0])

    p1 = plt.bar(ind, mplots[:,0], width=width, color=colors[1])
    p2 = plt.bar(ind, mplots[:,1], width=width, bottom=mplots[:,0], color=colors[2])
    p3 = plt.bar(ind, mplots[:,2], width=width, bottom=mplots[:,0]+mplots[:,1], color=colors[3])
    plt.legend(('common','b-unique','p_unique'))
    if title is not None:
        plt.title(title)
    plt.xlabel('behavior-independent quintile')
    plt.ylabel('mean r2')

    ax3 = plt.subplot(3,1,3)
    d=(mfull[:,1]-mfull[:,0])#/(1-np.abs(mfull[:,0]))
    stateplots.beta_comp(mfull[:,0], d, n1='State independent',n2='dep - indep',
                     ax=ax3, highlight=dm['sig'], hist_range=[-0.1, 1], markersize=4)
    ax3.plot([1,0], [0,1], 'k--', linewidth=0.5)
    r, p = st.pearsonr(mfull[:,0],d)
    plt.title('cc={:.3} p={:.4}'.format(r,p))

    #ind = np.arange(mb.shape[0])
    ##ind = m[:,0]
    #p1 = plt.plot(ind, mb[:,0])
    #p2 = plt.plot(ind, mb[:,1]+mb[:,0])
    #p3 = plt.plot(ind, mb[:,2]+mb[:,0]+mb[:,1])
    #plt.legend(('common','p_unique','b-unique'))
    #plt.xlabel('behavior-independent quintile')
    #plt.ylabel('mean r2')

    plt.tight_layout()
    return f


def aud_vs_state_wrapper(batches=None, pupil=True):
    """
    batches includes any of...

    active/passive only (pupil=True)
      batch = 305  # IC SUA
      batch = 313  # IC SUA and MUA
      batch = 311  # A1 SUA and MUA onBF

    pupil + active/passive  (pupil=False)
      batch = 307  # A1 SUA and MUA (pup)
      batch = 309  # IC SUA and MUA (pup)

"""
    if batches is None:
        # IC / A1 SUA and MUA (pup)
        batches = [309, 307]

    if pupil:
        # pup vs. active/passive
        state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
        basemodel = "-ref-psthfr.s_sdexp.S"
        loader = "psth.fs20.pup-ld-"
    else:
        # active/passive only
        state_list = ['st.beh0','st.beh']
        basemodel = "-ref-psthfr.s_stategain.S"
        loader = "psth.fs20-ld-"

    #plt.close('all')
    for bi, batch in enumerate(batches):
        df = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                               basemodel=basemodel, loader=loader)

        ax1, ax2, ax3 = aud_vs_state(df, nb=5, title='batch {}'.format(batch),
                                     state_list=state_list)
        ax2.set_ylim([0,.1])
        ax3.set_ylim([0,.1])
        


def beh_only_plot(batch=311):

    # 311 = A1 old (SVD) data -- on BF
    state_list = ['st.beh0', 'st.beh']
    basemodel = "-ref-psthfr.s_stategain.S"
    loader = "psth.fs20-ld-"
    fitter = "_jk.nf20-basic"
    df = get_model_results_per_state_model(batch=batch, state_list=state_list,
                                          basemodel=basemodel, fitter = "_jk.nf20-basic",
                                          loader=loader)
    da = df[df['state_chan']=='active']

    dp = da.pivot(index='cellid',columns='state_sig',
                  values=['r', 'r_se', 'MI', 'g', 'd'])

    dr = dp['r'].copy()
    dr['sig']=((dp['r'][state_list[1]]-dp['r'][state_list[0]]) > \
         (dp['r_se'][state_list[1]]+dp['r_se'][state_list[0]]))

    g = dp['g'].copy()
    d = dp['d'].copy()
    ggood = np.isfinite(g['st.beh'])
    stateplots.beta_comp(d.loc[ggood, 'st.beh'], g.loc[ggood, 'st.beh'],
                         n1='Baseline', n2='Gain',
                         title="Baseline/gain: batch {}".format(batch),
                         highlight=dr.loc[ggood, 'sig'], hist_range=[-1, 1])

    MI = dp['MI'].copy()
    migood = np.isfinite(MI['st.beh'])
    stateplots.beta_comp(MI.loc[migood, 'st.beh0'], MI.loc[migood, 'st.beh'],
                         n1='State independent', n2='State-dep',
                         title="MI: batch {}".format(batch),
                         highlight=dr.loc[migood, 'sig'], hist_range=[-0.5, 0.5])

    return df