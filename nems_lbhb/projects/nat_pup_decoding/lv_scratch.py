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

from nems0 import db
from nems0.utils import smooth
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems0.recording import load_recording
from os.path import basename, join

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
       'TNC019a', 'TNC020a', 'TNC021a', 'TNC043a', 'TNC044a', 'TNC045a']

figpath = '/auto/users/svd/docs/current/pupil_pop/2023_01_26/'

batch = 331
siteids, cellids = db.get_batch_sites(batch=batch)
print(siteids)

use_sqrt=True
states = ['st.pup+r3+s0,1,2,3', 'st.pup+r3+s1,2,3', 'st.pup+r3+s2,3',
          'st.pup+r3+s3', 'st.pup+r3']
if use_sqrt:
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
if 0:
    #importlib.reload(decoding)
    for cellid in ['ARM029a-09-6']:  #cellids[:5]:
        ctx, tdr_pred, tdr_resp = decoding.load_decoding_set(cellid, batch, modelnames, force_recompute=False)

        f, ax = plt.subplots(5,2,figsize=(4,10), sharex='col', sharey='col')
        for midx in range(5):
            if midx == 0:
                md = np.max([np.max(tdr_resp.numeric_results['bp_dp']),
                             np.max(tdr_pred[midx].numeric_results['bp_dp'])])/2

            ax[midx, 0].plot([0, md*2], [0, md*2], 'k--')
            ax[midx, 0].scatter(tdr_resp.numeric_results['sp_dp'], tdr_resp.numeric_results['bp_dp'],s=3)
            ax[midx, 0].scatter(tdr_pred[midx].numeric_results['sp_dp'], tdr_pred[midx].numeric_results['bp_dp'],s=3)

            ax[midx, 1].plot([-md, md], [-md, md], 'k--')
            a = tdr_pred[midx].numeric_results['bp_dp']-tdr_pred[midx].numeric_results['sp_dp']
            b = tdr_resp.numeric_results['bp_dp']-tdr_resp.numeric_results['sp_dp']
            cc=np.corrcoef(a, b)[0, 1]
            E = np.sqrt(np.sum((a-b)**2))/np.sqrt(np.sum(b**2))
            ax[midx, 1].scatter(a, b, s=3)
            ax[midx, 1].set_title(f"cc={cc:.3f}  E={E:.3f}")
        ax[0, 0].set_title(f"{db.get_siteid(cellid)} sqrt={use_sqrt}")

if 1:
    loadkey = "psth.fs4.pup"
    cellid = 'TNC014a'
    force_recache = True

    uri = generate_recording_uri(cellid=cellid, batch=batch, loadkey=loadkey, recache=force_recache)
    rec = load_recording(uri)

    f, ax = plt.subplots(2, 1, figsize=(15, 5), sharex=True)

    ax[0].plot(rec['pupil']._data[0, :])
    rr = smooth(rec['resp'].rasterize()._data.copy())
    rr /= rr.max(axis=1, keepdims=True)
    ax[1].imshow(rr, cmap='gray_r', aspect='auto', interpolation='none')

if 0:
    import importlib
    importlib.reload(decoding)

    for cellid in cellids:
        ctx, tdr_pred, tdr_resp = decoding.load_decoding_set(cellid, batch, modelnames, hist_norm=False, force_recompute=False)

    cellid = 'TNC020a-001-1'
    ctx, tdr_pred, tdr_resp = decoding.load_decoding_set(cellid, batch, modelnames, hist_norm=False, force_recompute=False)

    #modelnames=[modelnames[-1]]
    #c=[{}]*len(modelnames)
    #for midx, m in enumerate(modelnames):
    #    xfspec, c[midx] = xform_helper.load_model_xform(cellid, batch, m)
    #
    #    tdr_pred_alt = decoding.do_decoding_analysis(lv_model=True, hist_norm=True, **c[midx])
    #    tdr_pred = tdr_pred + [tdr_pred_alt]

    f,ax = plt.subplots(len(tdr_pred),2,figsize=(3,1.5*len(tdr_pred)), sharex='col',sharey='col')
    for midx in range(len(tdr_pred)):
        if midx==0:
            md = np.max([np.max(tdr_resp.numeric_results['bp_dp']),
                         np.max(tdr_pred[midx].numeric_results['bp_dp'])])/2

        ax[midx,0].plot([0,md*2],[0,md*2],'k--')
        ax[midx,0].scatter(tdr_resp.numeric_results['sp_dp'], tdr_resp.numeric_results['bp_dp'],s=3)
        ax[midx,0].scatter(tdr_pred[midx].numeric_results['sp_dp'], tdr_pred[midx].numeric_results['bp_dp'],s=3)

        ax[midx,1].plot([-md,md],[-md,md],'k--')
        a = tdr_pred[midx].numeric_results['bp_dp']-tdr_pred[midx].numeric_results['sp_dp']
        b = tdr_resp.numeric_results['bp_dp']-tdr_resp.numeric_results['sp_dp']
        cc=np.corrcoef(a,b)[0,1]
        E = np.sqrt(np.sum((a-b)**2))/np.sqrt(np.sum(b**2))
        ax[midx,1].scatter(a,b,s=3)
        ax[midx,1].set_title(f"cc={cc:.3f}  E={E:.3f}")
    ax[0,0].set_title(db.get_siteid(cellid));


