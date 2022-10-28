import numpy as np
import os
import io
import logging
import time
import matplotlib.pyplot as plt
import sys, importlib

import nems0.modelspec as ms
import nems0.xforms as xforms
import nems0.xform_helper as xhelp
from nems0.utils import escaped_split, escaped_join, smooth
from nems import get_setting
from nems0.xform_helper import _xform_exists, load_model_xform, fit_model_xform
from nems0.registry import KeywordRegistry, xforms_lib, keyword_lib
from nems_lbhb.projects.pop_model_scripts.pop_model_utils import POP_MODELS, SIG_TEST_MODELS
from nems import db
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models
from nems_lbhb.projects.nat_pup_decoding.ddr_pred_site import parse_modelname_base

log = logging.getLogger(__name__)

# A1/PEG LV models
batch = 331
batch = 322

short_set=False
if short_set:
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.er5-aev-rd" + \
                     "_stategain.2xR.x1,3,4-spred-lvnorm.5xR.so.x2,3-inoise.5xR.x2,4" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.er5-aev-rd" + \
                     "_stategain.2xR.x1,3,4-spred-lvnorm.5xR.so.x2,3-inoise.5xR.x2,4" + \
                     "_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.er5-aev-rd" + \
                     "_stategain.2xR.x1,3,4-spred-lvnorm.5xR.so.x2,3-inoise.5xR.x2,4" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.w.ss3"
elif batch == 331:
    ## batch 331- CPN (need epcpn keyword)
    # batch 331 - pred
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss3"
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et5.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss3"
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc2.no.p-{0}-plgsm.p2-aev-rd"+\
                "_stategain.2xR.x1,2,4-spred-lvnorm.SxR.so.x3-inoise.SxR.x4"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss3"
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss2"
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                     "_stategain.2xR.x1,3,4-spred-lvnorm.5xR.so.x2,4-inoise.5xR.x2,3" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                     "_stategain.2xR.x1,3,4,5-spred-lvnorm.6xR.so.x2,4-inoise.6xR.x2,3,5" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                     "_stategain.2xR.x1,3,4,5-spred-lvnorm.6xR.so.x2,4-inoise.6xR.x2,3,5" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss3"
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                     "_stategain.2xR.x1,3,4-spred-lvnorm.5xR.so.x2,4-inoise.5xR.x2,3" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss3"

    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                     "_stategain.2xR.x1,3,4-spred-lvnorm.5xR.so.x2,3-inoise.5xR.x2,4" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                     "_stategain.2xR.x1,3,4-spred-lvnorm.5xR.so.x2,3-inoise.5xR.x2,4" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t3d5.f0.ss3"
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                     "_stategain.2xR.x1,3,4-spred-lvnorm.5xR.so.x2,3-inoise.5xR.x2,4" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t3.f0.ss3"

    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                     "_stategain.2xR.x1,3,4-spred-lvnorm.5xR.so.x2,3-inoise.5xR.x2,4" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.w.ss3"
    modelname_base = "psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                     "_stategain.2xR.x1,3,4-spred-lvnorm.5xR.so.x2,3-inoise.5xR.x2,4" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"
    
    # adding in 3 face motor pca state channels for first-order only.
    modelname_base = "psth.fs4.pup.fpca3-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                     "_stategain.5xR.x1,3,4-spred-lvnorm.8xR.so.x2,3,5,6,7-inoise.8xR.x2,4,5,6,7" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"
    # 3 facepcs plus 2 pupil dims for LV, no pca
    modelname_base = "psth.fs4.pup.fpca3-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                     "_stategain.5xR.x2,3,4-spred-lvnorm.8xR.so.x1,2,5,6,7-inoise.8xR.x1,3,4,5,6,7" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"
    # 3 facepcs plus 2 pupil dims for LV
    modelname_base = "psth.fs4.pup.fpca3-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                     "_stategain.5xR.x1,3,4,5-spred-lvnorm.9xR.so.x2,3,6,7,8-inoise.9xR.x2,4,5,6,7,8" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"
    modelname_base="exp331"
    # batch 331 - actual data decoding
    resp_modelname = f"psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-{'st.pca.pup+r1'}-plgsm.p2-aev-rd.resp"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i20-lvnoise.r4-aev-ccnorm.md.t1.f0.ss3"
elif batch==322:
    ## batch 322- NAT
    modelname_base = "psth.fs4.pup-ld-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t5.f0.ss3"  #-ccnorm.md.t5.f0.ss3
    modelname_base = "psth.fs4.pup-ld-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"
    
    # adding in 3 face motor pca state channels for first-order only.
    modelname_base = "psth.fs4.pup.fpca3-ld-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                     "_stategain.5xR.x1,3,4-spred-lvnorm.8xR.so.x2,3,5,6,7-inoise.8xR.x2,4,5,6,7" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"
    # 3 facepcs plus 2 pupil dims for LV
    modelname_base="exp322"
    modelname_base = "psth.fs4.pup.fpca3-ld-hrc-psthfr.z-pca.cc1.no.p-{0}-plgsm.p2-aev-rd" + \
                     "_stategain.5xR.x1,3,4,5-spred-lvnorm.9xR.so.x2,3,6,7,8-inoise.9xR.x2,4,5,6,7,8" + \
                     "_tfinit.xx0.n.lr1e4.cont.et4.i50000-lvnoise.r8-aev-ccnorm.t4.f0.ss3"

    # batch 322 - actual data decoding
    resp_modelname = f"psth.fs4.pup-ld-hrc-psthfr.z-pca.cc1.no.p-{'st.pca.pup+r1'}-plgsm.p2-aev-rd.resp"+\
                "_stategain.2xR.x1,3-spred-lvnorm.4xR.so.x2-inoise.4xR.x3"+\
                "_tfinit.xx0.n.lr1e4.cont.et4.i20-lvnoise.r4-aev-ccnorm.md.t1.f0.ss3"
else:
    raise ValueError('batch not implemented')

modelnames, states = parse_modelname_base(modelname_base, batch)
#modelnames=[modelnames[1]]
siteids, cellids = db.get_batch_sites(batch)

#siteids=[siteids[0]]
priority=2

force_rerun = False

GPU_job = False
run_in_lbhb = True

if run_in_lbhb:
    executable_path = '/home/svd/bin/miniconda3/envs/nems_cpu/bin/python'
    script_path = '/auto/users/svd/python/nems/scripts/fit_single.py'
    GPU_job=int(GPU_job)
    # first models, run locally so that recordings get generated.
    r = db.enqueue_models(siteids, batch, modelnames, executable_path=executable_path,
                          force_rerun=force_rerun,script_path=script_path, GPU_job=GPU_job, user="svd", priority=priority)
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


