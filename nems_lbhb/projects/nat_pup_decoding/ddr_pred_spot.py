import numpy as np
import os
import io
import logging
import time
import matplotlib.pyplot as plt
import sys, importlib
import statsmodels.api as sm

import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
from nems.utils import escaped_split, escaped_join, smooth
from nems import get_setting
from nems.xform_helper import _xform_exists, load_model_xform, fit_model_xform
from nems.registry import KeywordRegistry, xforms_lib, keyword_lib
from nems.plugins import (default_keywords, default_loaders,
                          default_initializers, default_fitters)
from nems_lbhb.projects.pop_model_scripts.pop_model_utils import POP_MODELS, SIG_TEST_MODELS
from nems import db
from nems.recording import load_recording
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems import epoch as ep
from charlieTools.nat_sounds_ms.decoding import plot_stimulus_pair

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

cellid = [c for c in db.get_batch_cells(batch).cellid if site in c][0]

if batch == 331:
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.md.t5.f0.ss3"
    states = ['st.pca0.pup+r1+s0,1','st.pca.pup+r1+s0,1','st.pca.pup+r1+s1','st.pca.pup+r1']
    resp_modelname = f"psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{states[-1]}-plgsm.p2-aev-rd.resp"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i20-lvnoise.r4-aev-ccnorm.md.t1.f0.ss3"
else:
    modelname_base = "psth.fs4.pup-ld-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i20000-lvnoise.r4-aev-ccnorm.md.t5.f0.ss3"
    states = ['st.pca0.pup+r1+s0,1','st.pca.pup+r1+s0,1','st.pca.pup+r1+s1','st.pca.pup+r1']
    resp_modelname = f"psth.fs4.pup-ld-hrc-psthfr.z-pca.cc1.no.p-{states[-1]}-plgsm.p2-aev-rd.resp"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i20-lvnoise.r4-aev-ccnorm.md.t1.f0.ss3"

modelnames=[resp_modelname] + [modelname_base.format(s) for s in states]

xf,ctx=load_model_xform(cellid=cellid,batch=batch,modelname=modelnames[-1])


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
i2 = 2

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
