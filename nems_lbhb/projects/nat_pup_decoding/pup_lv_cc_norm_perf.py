import numpy as np
import os
import io
import logging
import time
import matplotlib.pyplot as plt
import sys, importlib
import statsmodels.api as sm

import nems0.modelspec as ms
import nems0.xforms as xforms
import nems0.xform_helper as xhelp
from nems0.utils import escaped_split, escaped_join, smooth
from nems import get_setting
from nems0.xform_helper import _xform_exists, load_model_xform, fit_model_xform
from nems0.registry import KeywordRegistry, xforms_lib, keyword_lib
from nems0.plugins import (default_keywords, default_loaders,
                          default_initializers, default_fitters)
from nems_lbhb.projects.pop_model_scripts.pop_model_utils import POP_MODELS, SIG_TEST_MODELS
from nems import db
from nems0.recording import load_recording
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems import epoch as ep
from charlieTools.nat_sounds_ms.decoding import plot_stimulus_pair
from nems_lbhb import stateplots
from nems0.analysis import fit_ccnorm

log = logging.getLogger(__name__)

ALL_SITES = ['BOL005c', 'BOL006b', 'bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
            'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
            'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b',
            'DRX006b.e1:64', 'DRX006b.e65:128',
            'DRX007a.e1:64', 'DRX007a.e65:128',
            'DRX008b.e1:64', 'DRX008b.e65:128',
            'CRD016d', 'CRD017c',
            'TNC008a','TNC009a', 'TNC010a', 'TNC012a', 'TNC013a', 'TNC014a', 'TNC015a', 'TNC016a', 'TNC017a', 'TNC018a', 'TNC020a']

site, batch = "TAR010c", 322
site, batch = "AMT021b", 331
site, batch = "TNC011a", 331
site, batch = "CRD019b", 331
site, batch = "AMT020a", 331

cellid = [c for c in db.get_batch_cells(batch).cellid if site in c][0]


modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                 "_stategain.2xR.x1,3,4-spred-lvnorm.5xR.so.x2,4-inoise.5xR.x2,3" + \
                 "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"
states = ['st.pca0.pup+r2+s0,1,2', 'st.pca.pup+r2+s0,1,2',
          'st.pca.pup+r2+s1,2', 'st.pca.pup+r2+s2', 'st.pca.pup+r2', 'st.pca0.pup+r2']
modelnames=[modelname_base.format(s) for s in states]


xf_list=[]
ctx_list=[]
for modelname in modelnames:
    xf,ctx=load_model_xform(cellid=cellid,batch=batch,modelname=modelname)
    xf_list.append(xf)
    ctx_list.append(ctx)

if len(states)==6:
    baseidx=4
else:
    baseidx=3

use_ctx = {'modelspec': ctx_list[baseidx]['modelspec'].copy, 'IsReload': True,
           'est': ctx_list[baseidx]['est'].copy()}
for midx, xf, ctx in zip(range(len(xf_list)), xf_list, ctx_list):
    #importlib.reload(fit_ccnorm)
    use_ctx['modelspec'] = ctx['modelspec'].copy()
    use_ctx['est']['pred'] = ctx['est']['pred']
    group_idx, group_cc, conditions = fit_ccnorm.compute_cc_matrices(shared_pcs=3, verbose=False, signal_for_cc='resp',**use_ctx)
    group_idx, group_cc_pred, conditions = fit_ccnorm.compute_cc_matrices(shared_pcs=3, verbose=False, signal_for_cc='pred',**use_ctx)

    cols = 4
    rows = int(np.ceil(len(group_cc)))
    f, ax = plt.subplots(rows, cols, figsize=(4, 2*rows))
    ax = ax.flatten()
    i = 0
    mm = np.max(np.abs(np.stack(group_cc)))

    gpp = group_cc_pred[-1]
    for g, g_pred, cond in zip(group_cc, group_cc_pred, conditions):
        ax[i*4].imshow(g, cmap='bwr', clim=[-mm,mm], origin='lower', interpolation='none')
        ax[i*4+1].imshow(g_pred, cmap='bwr', clim=[-mm,mm], origin='lower', interpolation='none')
        ax[i*4+2].imshow(g-g_pred, cmap='bwr', clim=[-mm,mm], origin='lower', interpolation='none')
        ax[i * 4 + 2].set_title(f"E={np.std(g - g_pred):.3f}")
        ax[i * 4 + 3].imshow(g - gpp, cmap='bwr', clim=[-mm, mm], origin='lower', interpolation='none')
        ax[i * 4 + 3].set_title(f"rand E={np.std(g - gpp):.3f}")
        ax[i*4].set_title(cond)
        i += 1
        gpp = g_pred
    f.suptitle(f"{cellid} {states[midx]}", fontsize=7)

# plt.close('all')
# rec=ctx_list[0]['rec'].apply_mask()
# plt.figure()
# plt.plot(rec['state']._data[1:3,:1000].T)
# rec=ctx_list[-2]['rec'].apply_mask()
# plt.figure()
# plt.plot(rec['state']._data[1:3,:1000].T)

"""
val=ctx['val'].copy()
resp = val['resp'].rasterize()
epoch_regex="^STIM_"
epochs = ep.epoch_names_matching(resp.epochs,regex_str=epoch_regex)

input_name = 'pred0'
pred0 = val[input_name].extract_epochs(epochs, mask=val['mask'])
pred = val['pred'].extract_epochs(epochs, mask=val['mask'])
resp = val['resp'].extract_epochs(epochs, mask=val['mask'])
pupil = val['pupil'].extract_epochs(epochs, mask=val['mask'])
pmedian=np.nanmedian(val['pupil'].as_continuous())

epochs=list(resp.keys())
epochs


#
#ncells, nreps, nstim, nbins = X.shape

e1 = epochs[0]
i1 = 0
e2 = epochs[1]
i2 = 0

X_raw = np.stack((resp[e1][:, :, i1], resp[e2][:, :, i2]), axis=2)[:, :, :, np.newaxis]
X = np.stack((pred[e1][:, :, i1], pred[e2][:, :, i2]), axis=2)[:, :, :, np.newaxis]
X_pup = np.stack((pupil[e1][:, :, i1], pupil[e2][:, :, i2]), axis=2)[:, :, :, np.newaxis]
X_raw = np.swapaxes(X_raw, 0, 1)
X = np.swapaxes(X, 0, 1)
X_pup = np.swapaxes(X_pup, 0, 1)

for i in range(2):
    pm = np.argsort(X_pup[0, :, i, 0])
    X[:, :, i, :] = X[:, pm, i, :]
    X_raw[:, :, i, :] = X_raw[:, pm, i, :]
    X_pup[:, :, i, :] = X_pup[:, pm, i, :]

pup_mask = X_pup > pmedian

X_raw.shape



f,ax=plt.subplots(2,2)
c=0
plot_stimulus_pair(X=X_raw, X_pup=X_pup, X_raw=X_raw, pup_mask=pup_mask,
                   ellipse=True, pup_split=True, ax=ax[c,0])
plot_stimulus_pair(X=X, X_pup=X_pup, X_raw=X_raw, pup_mask=pup_mask,
                   ellipse=True, pup_split=True, ax=ax[c,1])
ax[c,0].set_ylabel(f"({i1},{i2}")
"""