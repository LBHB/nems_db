import numpy as np
import os
import io
import logging
import time
import matplotlib.pyplot as plt
import sys, importlib

import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
from nems.utils import escaped_split, escaped_join, smooth
from nems import get_setting
from nems.xform_helper import _xform_exists, load_model_xform, fit_model_xform
from nems.registry import KeywordRegistry, xforms_lib, keyword_lib
from nems_lbhb.projects.pop_model_scripts.pop_model_utils import POP_MODELS, SIG_TEST_MODELS
from nems import db
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models

log = logging.getLogger(__name__)

# A1/PEG LV models
batch = 331
#batch = 322

states = ['st.pca0.pup+r1+s0,1', 'st.pca.pup+r1+s0,1',
          'st.pca.pup+r1+s1', 'st.pca.pup+r1']

if batch == 331:
    ## batch 331- CPN (need epcpn keyword)
    # batch 331 - pred
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss3"
    # batch 331 - actual data decoding
    resp_modelname = f"psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{states[-1]}-plgsm.p2-aev-rd.resp"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i20-lvnoise.r4-aev-ccnorm.md.t1.f0.ss3"
elif batch==322:
    ## batch 322- NAT
    modelname_base = "psth.fs4.pup-ld-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.md.t5.f0.ss3"

    # batch 322 - actual data decoding
    resp_modelname = f"psth.fs4.pup-ld-hrc-psthfr.z-pca.cc1.no.p-{states[-1]}-plgsm.p2-aev-rd.resp"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i20-lvnoise.r4-aev-ccnorm.md.t1.f0.ss3"
else:
    raise ValueError('batch not implemented')

#modelnames=[resp_modelname]
modelnames=[resp_modelname]+[modelname_base.format(s) for s in states]

siteids, cellids = db.get_batch_sites(batch)

#siteids=[siteids[0]]

force_rerun = False

GPU_job = False
run_in_lbhb = True

if run_in_lbhb:
    executable_path = '/home/svd/bin/miniconda3/envs/nems_cpu/bin/python'
    script_path = '/auto/users/svd/python/nems/scripts/fit_single.py'
    GPU_job=int(GPU_job)
    # first models, run locally so that recordings get generated.
    r = db.enqueue_models(siteids, batch, modelnames, executable_path=executable_path,
                          force_rerun=force_rerun,script_path=script_path, GPU_job=GPU_job, user="svd")
    for a, b in r:
        print(a, b)

else:
    # exacloud


    # exacloud queue settings:
    exa_executable_path = '/home/users/davids/anaconda3/envs/nems/bin/python'
    exa_script_path = '/home/users/davids/nems/scripts/fit_single.py'
    ssh_key = '/home/svd/.ssh/id_rsa'
    user = "davids"
    lbhb_user = "svd"

    enqueue_exacloud_models(
        cellist=siteids, batch=batch, modellist=modelnames,
        user=lbhb_user, linux_user=user, force_rerun=force_rerun,
        executable_path=exa_executable_path, script_path=exa_script_path, useGPU=GPU_job)


