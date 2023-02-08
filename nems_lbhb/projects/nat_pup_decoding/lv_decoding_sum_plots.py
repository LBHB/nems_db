

import os
import pickle
import importlib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from nems0 import xform_helper
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models
import nems_lbhb.projects.nat_pup_decoding.do_decoding as decoding

import nems0.epoch as ep
from nems0 import db
from nems0.plots.state import cc_comp
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems0.recording import load_recording
from os.path import basename, join
from nems_lbhb.projects.nat_pup_decoding.plotting import compute_ellipse

ALL_SITES = ['BOL005c', 'BOL006b', 'bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
         'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
         'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b',
         'DRX006b.e1:64', 'DRX006b.e65:128',
         'DRX007a.e1:64', 'DRX007a.e65:128',
         'DRX008b.e1:64', 'DRX008b.e65:128',
         'CRD016d', 'CRD017c',
         'TNC008a','TNC009a', 'TNC010a', 'TNC012a', 'TNC013a', 'TNC014a',
         'TNC015a', 'TNC016a', 'TNC017a', 'TNC018a', 'TNC020a']
CPN_SITES = ['AMT020a', 'AMT026a', 'ARM029a', 'ARM031a',
       'ARM032a', 'ARM033a', 'CRD018d',
       'TNC006a', 'TNC008a', 'TNC009a', 'TNC010a', 'TNC012a',
       'TNC013a', 'TNC014a', 'TNC015a', 'TNC016a', 'TNC017a', 'TNC018a',
       'TNC020a', 'TNC021a', 'TNC043a', 'TNC044a', 'TNC045a'] # 'TNC019a',

figpath = '/auto/users/svd/docs/current/pupil_pop/2023_02_08/'

#batch = 322
batch = 331
use_sqrt = True
siteids_, cellids = db.get_batch_sites(batch=batch)

if batch == 322:
    siteids = [s for s, c in zip(siteids_, cellids) if s in ALL_SITES]
    cellids = [c for s, c in zip(siteids_, cellids) if s in ALL_SITES]

    states = ['st.pup+r3+s0,1,2,3','st.pup+r3+s1,2,3','st.pup+r3+s2,3','st.pup+r3+s3','st.pup+r3']
    modelnames = [f"psth.fs4.pup-ld-norm.sqrt-hrc-psthfr.z-pca.cc1.no.p-{s}-plgsm.p2-aev" + \
                  "_stategain.2xR.x2,3,4-spred-lvnorm.5xR.so.x1,2-inoise.5xR.x1,3,4" + \
                  "_tfinit.xx0.n.lr1e4.cont.et4.i20000-lvnoise.r2-aev-ccnorm.t4.f0"
                  for s in states]
elif batch == 331:
    siteids = [s for s, c in zip(siteids_, cellids) if s in CPN_SITES]
    cellids = [c for s, c in zip(siteids_, cellids) if s in CPN_SITES]

    states = ['st.pup+r3+s0,1,2,3','st.pup+r3+s1,2,3','st.pup+r3+s2,3','st.pup+r3+s3','st.pup+r3']
    if use_sqrt:

        # V2 -- dc offset in state before multiply by LV -- avoid double-negative.
        modelnames = [f"psth.fs4.pup-ld-norm.sqrt-epcpn-hrc-psthfr.z-pca.cc1.no.p-{s}-plgsm.p2-aev" + \
                      "_stategain.2xR.x2,3,4-spred-lvnorm.5xR.so.x1,2-inoise.5xR.x1,3,4" + \
                      "_tfinit.xx0.n.lr1e4.cont.et4.i20000-lvnoise.r2-aev-ccnorm.t4.f0.V2"
                      for s in states]
        # lvnorm without so
        modelnames = [f"psth.fs4.pup-ld-norm.sqrt-epcpn-hrc-psthfr.z-pca.cc1.no.p-{s}-plgsm.p2-aev" + \
                      "_stategain.2xR.x2,3,4-spred-lvnorm.5xR.so.x1,2-inoise.5xR.x1,3,4" + \
                      "_tfinit.xx0.n.lr1e4.cont.et4.i20000-lvnoise.r2-aev-ccnorm.t4.f0"
                      for s in states]

    else:
        # lvnorm without so, no sqrt norm
        modelnames = [f"psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{s}-plgsm.p2-aev" + \
                      "_stategain.2xR.x2,3,4-spred-lvnorm.5xR.so.x1,2-inoise.5xR.x1,3,4" + \
                      "_tfinit.xx0.n.lr1e4.cont.et4.i20000-lvnoise.r2-aev-ccnorm.t4.f0"
                      for s in states]

print("siteids:", siteids)
print("cellids:", cellids)

#importlib.reload(decoding)
PLOT_EACH_SITE=False
if PLOT_EACH_SITE:
    for cellid in cellids:
        ctx, tdr_pred, tdr_resp = decoding.load_decoding_set(cellid, batch, modelnames, force_recompute=False)

        f, ax = plt.subplots(5,2,figsize=(4,10), sharex='col', sharey='col')
        for midx in range(5):
            if midx == 0:
                md = np.max([np.max(tdr_resp.numeric_results['bp_dp']),
                             np.max(tdr_pred[midx].numeric_results['bp_dp'])])/2

            ax[midx, 0].plot([0, md*2], [0, md*2], 'k--')

            dtemp = tdr_resp.numeric_results.merge(tdr_pred[midx].numeric_results,
                                                   how='inner',left_index=True, right_index=True,
                                                   suffixes=('_a','_p'))
            ax[midx, 0].scatter(dtemp['sp_dp_a'], dtemp['bp_dp_a'],s=2)
            ax[midx, 0].scatter(dtemp['sp_dp_p'], dtemp['bp_dp_p'],s=2)

            ax[midx, 1].plot([-md, md], [-md, md], 'k--')
            a = dtemp['bp_dp_p']-dtemp['sp_dp_p']
            b = dtemp['bp_dp_a']-dtemp['sp_dp_a']
            cc=np.corrcoef(a, b)[0, 1]
            E = np.sqrt(np.sum((a-b)**2))/np.sqrt(np.sum(b**2))
            ax[midx, 1].scatter(a, b, s=3)
            ax[midx, 1].set_title(f"cc={cc:.3f}  E={E:.3f}")
        ax[0, 0].set_title(f"{db.get_siteid(cellid)} {batch} sqrt={use_sqrt}")

#
# load all decoding results
tdr_pred = []
ctx = {}
for cellid,siteid in zip(cellids, siteids):
    ctx[siteid], tdr_pred_, tdr_resp_ = decoding.load_decoding_set(cellid, batch, modelnames, hist_norm=True)
    tdr_resp_.numeric_results['siteid']=siteid
    tdr_resp_.numeric_results['state']='actual'
    tdr_resp_.numeric_results['stateid']=-1
    mean_act_dp = (tdr_resp_.numeric_results['sp_dp']+tdr_resp_.numeric_results['bp_dp'])/2
    tdr_resp_.numeric_results['mean_act_dp'] = mean_act_dp
    tdr_pred.append(tdr_resp_.numeric_results)
    for i,(s,t) in enumerate(zip(states, tdr_pred_)):
        t.numeric_results['siteid']=siteid
        t.numeric_results['state']=s
        t.numeric_results['stateid']=i
        t.numeric_results['mean_act_dp']=mean_act_dp
        tdr_pred.append(t.numeric_results)

tdr_pred = pd.concat(tdr_pred)
tdr_pred['deltad'] = tdr_pred['bp_dp'] - tdr_pred['sp_dp']

#
# scatter plot for ALL sites
#plt.close('all')
state_labels=['all shuffled', '+pup 1st order', '+pup inoise', '+1st pup LV', '+2nd pup LV']
state_colors=[(50/255,50/255,50/255), (255/255,127/255,15/255),
              (214/255,38/255,39/255), (148/255,103/255,189/255),
              (100/255,90/255,128/255)]
f, ax = plt.subplots(len(states)+1, 3, figsize=(2.25*3, 2.5*(len(states)+1)), sharex='col', sharey='col')
for midx in range(len(states)):

    rr = (tdr_pred['state']=='actual')#  & ((tdr_pred['r1mag_test']<1) | (tdr_pred['r2mag_test']<1))
    pp = (tdr_pred['state']==states[midx])#  & ((tdr_pred['r1mag_test']<1) | (tdr_pred['r2mag_test']<1))

    if midx==0:
        md = 150
        a, b = tdr_pred.loc[rr, 'sp_dp'], tdr_pred.loc[rr, 'bp_dp']
        g = (a < md) & (b < md)
        a = a[g]
        b = b[g]
        ax[0, 0].plot([0, md], [0, md], '--', color='lightgray')
        ax[0, 0].scatter(a[::5], b[::5], s=1, color='k')
        ax[0, 0].set_ylabel('Large pupil d-prime')
        ax[0, 0].set_xlabel('Small pupil d-prime')
        ax[0, 1].set_visible(False)
        ax[0, 2].set_visible(False)

    ax[midx+1, 0].plot([0,md], [0,md], '--', color='lightgray')
    a,b = tdr_pred.loc[rr, 'sp_dp'], tdr_pred.loc[rr, 'bp_dp']
    g = (a < md) & (b < md)
    a = a[g]
    b = b[g]
    ax[midx+1, 0].scatter(a[::5], b[::5], s=1, color='lightgray')
    a,b = tdr_pred.loc[pp, 'sp_dp'], tdr_pred.loc[pp, 'bp_dp']
    g = (a < md) & (b < md)
    a = a[g]
    b = b[g]
    ax[midx+1, 0].scatter(a[::5], b[::5], s=1, color=state_colors[midx])
    ax[midx+1, 0].set_title(f"{state_labels[midx]}")
    ax[midx+1, 0].set_ylabel('Large pupil d-prime')
    ax[midx+1, 0].set_xlabel('Small pupil d-prime')

    a = (tdr_pred.loc[pp, 'bp_dp']-tdr_pred.loc[pp, 'sp_dp'])
    b = (tdr_pred.loc[rr, 'bp_dp']-tdr_pred.loc[rr, 'sp_dp'])
    cc=np.corrcoef(a,b)[0,1]
    E = np.sqrt(np.sum((a-b)**2))/np.sqrt(np.sum(b**2))
    mmin,mmax = -50,100
    g = (a>mmin) & (b>mmin) & (a<mmax) & (b<mmax)
    a=a[g]
    b=b[g]
    ax[midx+1, 1].plot([mmin,mmax],[mmin,mmax], '--', color='lightgray')
    ax[midx+1, 1].scatter(a[::10], b[::10], s=1, color=state_colors[midx])
    ax[midx+1, 1].set_title(f"cc={cc:.3f}  E={E:.3f}")
    ax[midx+1, 1].set_ylabel('Large actual-model d-prime')
    ax[midx+1, 1].set_xlabel('Small actual-model d-prime')

    a = (tdr_pred.loc[pp, 'bp_dp']-tdr_pred.loc[pp, 'sp_dp'])  / (tdr_pred.loc[pp, 'bp_dp']+tdr_pred.loc[pp, 'sp_dp'])
    b = (tdr_pred.loc[rr, 'bp_dp']-tdr_pred.loc[rr, 'sp_dp']) / (tdr_pred.loc[rr, 'bp_dp']+tdr_pred.loc[rr, 'sp_dp'])
    cc=np.corrcoef(a,b)[0,1]
    E = np.sqrt(np.sum((a-b)**2))/np.sqrt(np.sum(b**2))
    ax[midx+1, 2].plot([-1,1],[-1,1],'--', color='lightgray')
    ax[midx+1, 2].scatter(a[::10],b[::10],s=1, color=state_colors[midx])
    ax[midx+1, 2].set_title(f"cc={cc:.3f}  E={E:.3f}")
    ax[midx+1, 2].set_ylabel('Large actual-model norm')
    ax[midx+1, 2].set_xlabel('Small actual-model norm')


modelspec = modelnames[-1].split('_')[1]
figfile = (f"{figpath}dp_all_{modelspec}_sqrt{use_sqrt}.pdf").replace(",","")
f.suptitle(modelspec)
plt.tight_layout()

print(figfile)
f.savefig(figfile)

#
# compare mean per site

dfsum = pd.DataFrame(columns=['siteid','stateid','state_label',
                              'E','cc','Enorm','ccnorm',
                              'sp_dp','bp_dp','sp_dp_act','bp_dp_act'])
for s,siteid in enumerate(CPN_SITES): # (siteids):
    for midx in range(len(states)):
        rr = (tdr_pred['state']=='actual') & (tdr_pred['siteid']==siteid)# & (tdr_pred['mean_act_dp']>1)
        pp = (tdr_pred['state']==states[midx]) & (tdr_pred['siteid']==siteid)# & (tdr_pred['mean_act_dp']>1)

        # uncomment for jackknife version-- doesn't weigh sites properly
        #rr = (tdr_pred['state']=='actual') & (tdr_pred['siteid']!=siteid)# & (tdr_pred['mean_act_dp']>1)
        #pp = (tdr_pred['state']==states[midx]) & (tdr_pred['siteid']!=siteid)# & (tdr_pred['mean_act_dp']>1)

        # delta dprime
        a = (tdr_pred.loc[pp, 'bp_dp']-tdr_pred.loc[pp, 'sp_dp'])
        b = (tdr_pred.loc[rr, 'bp_dp']-tdr_pred.loc[rr, 'sp_dp'])
        cc=np.corrcoef(a,b)[0,1]
        E = np.sqrt(np.sum((a-b)**2))/np.sqrt(np.sum(b**2))

        # normed delta dprime
        a = (tdr_pred.loc[pp, 'bp_dp']-tdr_pred.loc[pp, 'sp_dp'])  / (tdr_pred.loc[pp, 'bp_dp']+tdr_pred.loc[pp, 'sp_dp'])
        b = (tdr_pred.loc[rr, 'bp_dp']-tdr_pred.loc[rr, 'sp_dp']) / (tdr_pred.loc[rr, 'bp_dp']+tdr_pred.loc[rr, 'sp_dp'])
        ccnorm=np.corrcoef(a,b)[0,1]
        Enorm = np.sqrt(np.sum((a-b)**2))/np.sqrt(np.sum(b**2))

        sp_dp = (tdr_pred.loc[pp, 'sp_dp']).median()
        bp_dp = (tdr_pred.loc[pp, 'bp_dp']).median()
        sp_dp_act = (tdr_pred.loc[rr, 'sp_dp']).median()
        bp_dp_act = (tdr_pred.loc[rr, 'bp_dp']).median()
        #print(f"{midx} {sp_dp} {bp_dp} {sp_dp_act} {bp_dp_act}")
        dfsum.loc[len(dfsum)]=[siteid, midx, state_labels[midx], E, cc, Enorm, ccnorm, sp_dp, bp_dp, sp_dp_act, bp_dp_act]

dfsum.groupby('stateid').mean()

from scipy.stats import ttest_rel

for s in range(0,4):
    a = dfsum.loc[dfsum.stateid==s,'cc']
    b = dfsum.loc[dfsum.stateid==(s+1),'cc']
    T, p = ttest_rel(a,b)
    print("CC:", state_labels[s], state_labels[s+1],T,p)
    a = dfsum.loc[dfsum.stateid==s,'ccnorm']
    b = dfsum.loc[dfsum.stateid==(s+1),'ccnorm']
    T, p = ttest_rel(a,b)
    print("CC norm:", state_labels[s], state_labels[s+1],T,p)

#mean_dp = tdr_pred.groupby(['stateid','siteid']).median()
#mean_dp = mean_dp.reset_index()
#mean_dp.groupby('stateid').mean()[['sp_dp','bp_dp','deltad']]

sp_dp_mean = dfsum.pivot_table('sp_dp',index='siteid', columns='stateid').mean()
bp_dp_mean = dfsum.pivot_table('bp_dp',index='siteid', columns='stateid').mean()
sp_dp_sem = dfsum.pivot_table('sp_dp',index='siteid', columns='stateid').sem()
bp_dp_sem = dfsum.pivot_table('bp_dp',index='siteid', columns='stateid').sem()
sp_dp_act_mean = dfsum.pivot_table('sp_dp_act',index='siteid', columns='stateid').mean()
bp_dp_act_mean = dfsum.pivot_table('bp_dp_act',index='siteid', columns='stateid').mean()
act_offset = np.mean(bp_dp_mean+sp_dp_mean)/2 - (bp_dp_act_mean+sp_dp_act_mean)/2

d = dfsum.pivot_table('ccnorm',index='siteid', columns='stateid')
cc_mean = d.median().values
cc_sem = d.std().values / np.sqrt(d.shape[0]-1)

f,ax = plt.subplots(2,1, figsize=(4,4), sharex='col')

ax[0].plot(state_labels, sp_dp_mean, color='darkgreen')
ax[0].plot(state_labels, bp_dp_mean, color='darkblue')
ax[0].plot(state_labels, sp_dp_act_mean+act_offset, '--', color='darkgreen')
ax[0].plot(state_labels, bp_dp_act_mean+act_offset, '--', color='darkblue')
ax[0].set_ylabel("Mean d-prime")
ax[0].legend(('Small pred', 'Large pred', 'Small act', 'Large act'), frameon=False)

ax[1].plot(state_labels, cc_mean, color='lightgray')
for i in range(len(state_labels)):
    ax[1].plot(i, cc_mean[i], 'o', color=state_colors[i])
    ax[1].errorbar(i, cc_mean[i], cc_sem[i], color=state_colors[i], lw=2)

ax[1].set_ylabel("CC(predicted, actual)")
plt.tight_layout()

figfile = (f"{figpath}dp_sum_{modelspec}_sqrt{use_sqrt}.pdf").replace(",","")
f.savefig(figfile)

f,ax = plt.subplots(1,1, figsize=(3,3))
ax.scatter(d.values[:,1], d.values[:,3], color='k', s=10)
ax.plot([0,0.9], [0,0.9], '--', color='lightgray', lw=1)
ax.set_ylim([0,0.9])
ax.set_xlim([0,0.9])
ax.set_xlabel('First-order only model')
ax.set_ylabel('Pupil latent variable model')
ax.set_title('Model vs. actual d-prime correlation')
plt.tight_layout()

figfile = (f"{figpath}dp_sum_comp_{modelspec}_sqrt{use_sqrt}.pdf").replace(",","")
f.savefig(figfile)

example_site = 'ARM033a' # too few cells
example_site = 'TNC017a' # very low dprime
example_site = 'TNC013a' # no first-order effect on decoding?
example_site = 'AMT020a' # LV does not help match d
example_site = 'TNC021a' # small decoding changes, though pattern is nice
example_site = 'TNC016a' # very low dprime
example_site = 'TNC010a' # too few cells
example_site = 'TNC020a' #
example_site = 'CRD018d' # pretty nice.

cellid = [c for s,c in zip(siteids, cellids) if s==example_site][0]
modelname = modelnames[3]

xf, ctx = xform_helper.load_model_xform(cellid, batch, modelname)

ctx['IsReload']=False

from nems0.plots import state
import importlib

#plt.close('all')
importlib.reload(state)

figfile = (f"{figpath}cc_comp_{example_site}.pdf").replace(",","")
res = state.cc_comp(saveto=figfile, **ctx)

# example site scatter plots
f, ax = plt.subplots(len(states)+1, 2, figsize=(2.25*2, 2.5*(len(states)+1)), sharex='col', sharey='col')
for midx in range(len(states)):

    rr = (tdr_pred['state']=='actual') & (tdr_pred['siteid']==example_site)
    pp = (tdr_pred['state']==states[midx]) & (tdr_pred['siteid']==example_site)

    if midx == 0:
        a, b = tdr_pred.loc[rr, 'sp_dp'], tdr_pred.loc[rr, 'bp_dp']
        if b.max()>150:
            md = 150
            emin, emax = -50, 100
        else:
            md = 100
            emin, emax = -25, 75

        g = (a < md) & (b < md)
        a = a[g]
        b = b[g]
        ax[0, 0].plot([0, md], [0, md], '--', color='lightgray')
        ax[0, 0].scatter(a, b, s=2, color='k')
        ax[0, 0].set_ylabel('Large pupil d-prime')
        ax[0, 0].set_xlabel('Small pupil d-prime')
        ax[0, 1].set_visible(False)

    ax[midx+1, 0].plot([0,md], [0,md], '--', color='lightgray')
    a, b = tdr_pred.loc[rr, 'sp_dp'], tdr_pred.loc[rr, 'bp_dp']
    g = (a < md) & (b < md)
    a = a[g]
    b = b[g]
    ax[midx+1, 0].scatter(a, b, s=2, color='lightgray')
    a, b = tdr_pred.loc[pp, 'sp_dp'], tdr_pred.loc[pp, 'bp_dp']
    g = (a < md) & (b < md)
    a = a[g]
    b = b[g]
    ax[midx+1, 0].scatter(a, b, s=2, color=state_colors[midx])
    ax[midx+1, 0].set_title(f"{state_labels[midx]}")
    ax[midx+1, 0].set_ylabel('Large pupil d-prime')
    ax[midx+1, 0].set_xlabel('Small pupil d-prime')

    a = (tdr_pred.loc[pp, 'bp_dp']-tdr_pred.loc[pp, 'sp_dp'])
    b = (tdr_pred.loc[rr, 'bp_dp']-tdr_pred.loc[rr, 'sp_dp'])
    cc = np.corrcoef(a,b)[0,1]
    E = np.sqrt(np.sum((a-b)**2))/np.sqrt(np.sum(b**2))
    g = (a > emin) & (b > emin) & (a < emax) & (b <emax)
    a = a[g]
    b = b[g]
    ax[midx+1, 1].plot([emin, emax], [emin, emax], '--', color='lightgray')
    ax[midx+1, 1].scatter(a, b, s=2, color=state_colors[midx])
    ax[midx+1, 1].set_title(f"cc={cc:.3f}  E={E:.3f}")
    ax[midx+1, 1].set_ylabel('Large actual-model d-prime')
    ax[midx+1, 1].set_xlabel('Small actual-model d-prime')

modelspec = modelnames[-1].split('_')[1]
f.suptitle(example_site + " " + modelspec)
plt.tight_layout()

figfile = (f"{figpath}dp_{example_site}_{modelspec}_sqrt{use_sqrt}.pdf").replace(",","")
f.savefig(figfile)



rec = ctx['val']
epochs = ep.epoch_names_matching(rec['resp'].epochs, "^STIM_")
r = rec['resp'].extract_epochs(epochs, mask=rec['mask'])


n=len(r.keys()) * 3
cmap = plt.cm.get_cmap('viridis')  # type: matplotlib.colors.ListedColormap
color = np.array(cmap.colors)
color = color[np.linspace(0,255,n).astype(int)] # type: list

#plt.figure()
#plt.plot(rec['resp']._data.std(axis=1))

#plt.close('all')

c1,c2,highlight_bin = 21,26,1
c1,c2,highlight_bin = 21,19,1
c1,c2,highlight_bin = 21,0,1
c1,c2,highlight_bin = 10,26,0
c1,c2,highlight_bin = 11,26,0

f,ax=plt.subplots(figsize=(3,3))
cc=0
for k,v_ in r.items():
    for i in range(v_.shape[2]):
        v=v_  #**2
        j = np.random.randn(v.shape[0],2)/40+1
        e1 = compute_ellipse(v[:, c1, i], v[:, c2, i])
        nc = np.corrcoef(v[:,c1,i], v[:,c2,i])[0,1]
        m1,m2 = v[:,c1,i].mean(), v[:,c2,i].mean()
        if i == highlight_bin:
            ax.plot(v[:,c1,i]*j[:,0],v[:,c2,i]*j[:,1],'.', color=color[cc],
                    markersize=2)
            ax.plot(e1[0], e1[1], lw=2, color=color[cc])
            ax.text(m1,m2,f"{nc:.3f}",va='center',ha='center',color=color[cc])
        else:
            ax.plot(e1[0], e1[1], lw=1, color='lightgray')
        cc+=1
ax.set_title(f"{c1},{c2} highlight={highlight_bin}")
ax.set_aspect('equal')
ax.set_xlabel('Cell 1 (sqrt(spikes/sec)')
ax.set_ylabel('Cell 2 (sqrt(spikes/sec)')
#ax.set_xlim([-0.5,5.5])
#ax.set_ylim([-0.5,5.5])
figfile = (f"{figpath}perstimcorr_{example_site}_{c1}_{c2}_sqrt{use_sqrt}.pdf").replace(",","")
f.savefig(figfile)



