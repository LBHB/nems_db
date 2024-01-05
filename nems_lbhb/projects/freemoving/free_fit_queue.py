from pathlib import Path
from scipy.io import wavfile
import matplotlib.pyplot as plt

from nems_lbhb.baphy_experiment import BAPHYExperiment

from nems0.analysis.gammatone.gtgram import gtgram
import nems0.epoch as ep
from nems0 import db
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models

batch = 348
siteids, cellids = db.get_batch_sites(batch)
#siteids=['PRN048a']

executable_path = '/auto/users/svd/python/nems_db/nems_lbhb/projects/freemoving/bash_fit_wrapper'
#script_path = '/auto/users/svd/python/nems_db/nems_lbhb/projects/freemoving/free_moving_fit_script.py'
script_path = '/auto/users/svd/python/nems_db/nems_lbhb/projects/freemoving/free_moving_fit_script.py'
GPU_job=True

modelnames = ['sh.none-hrtf.True__nmse',
              'sh.none-hrtf.False__nmse',
              'sh.dlc-hrtf.True__nmse',
              'sh.dlc-hrtf.False__nmse',
              'sh.stim-hrtf.True__nmse',
              ]

dlc_count=10
dlc1 = 40
strf_channels=20
rasterfs = 50

dlc_memory=4
acount=20
dcount=10
l2count=30
tcount=acount+dcount
input_count = 36


sep_kw = f'wcst.Nx1x{acount}.i-wcdl.{dlc_count}x1x{dcount}.i-first.8x1x{acount}-firdl.{dlc_memory}x1x{dcount}-cat-relu.{tcount}.o.s'
aud_kw = f'wc.{tcount}x1x{l2count}-fir.4x1x{l2count}-relu.{l2count}.o.s-wc.{l2count}xR-relu.R.o.s'
model_kw_old = sep_kw + '-' + aud_kw

hrtf_kw = f'wcdl.{dlc_count}x{dlc1}.i-relud.{dlc1}.o.s-wcdl.{dlc1}x10-relud.10.o.s-wcdl.10x5-relud.5.o.s-wcdl.5x{input_count}-sigd.{input_count}.s.g-mult'
aud_kw = f'wch.Nx1x{strf_channels}-fir.10x1x{strf_channels}-relu.{strf_channels}.o.s-wc.{strf_channels}x1x{l2count}-fir.10x1x{l2count}-relu.{l2count}.o.s-wc.{l2count}xR-relu.R.o.s'
model_kw_new = hrtf_kw + '-' + aud_kw

load_kw_shuff = f"free.fs{rasterfs}.ch18-norm.l1-fev-shuf.dlc"
load_kw = f"free.fs{rasterfs}.ch18-norm.l1-fev"
load_kw_hrtf = f"free.fs{rasterfs}.ch18-norm.l1-fev.hrtf"
load_kw_hrtf_shuff = f"free.fs{rasterfs}.ch18-norm.l1-fev.hrtf-shuf.dlc"
fit_kw = "lite.tf.cont.init.lr1e3.t3-lite.tf.cont.lr1e4"

modelnames=["_".join([load_kw,model_kw_old,fit_kw]),
            "_".join([load_kw,model_kw_new,fit_kw]),
            "_".join([load_kw_shuff, model_kw_old, fit_kw]),
            "_".join([load_kw_shuff, model_kw_new, fit_kw]),
            "_".join([load_kw_hrtf, model_kw_old, fit_kw]),
            "_".join([load_kw_hrtf_shuff, model_kw_old, fit_kw]),
            ]

force_rerun = False
run_in_lbhb = False

if run_in_lbhb:
    # first models, run locally so that recordings get generated.
    r = db.enqueue_models(siteids, batch, modelnames, executable_path=executable_path,
                          script_path=script_path, GPU_job=GPU_job, user="svd",
                          linux_user='svd', force_rerun=force_rerun)
    for a,b in r:
        print(a,b)
else:
    # exacloud

    # exacloud queue settings:
    exa_executable_path = '/home/users/davids/miniconda3/envs/nems/bin/python'
    exa_script_path = '/home/users/davids/nems_db/scripts/nems0_scripts/fit_single.py'
    ssh_key = '/home/svd/.ssh/id_rsa'
    user = "davids"
    lbhb_user = "svd"

    enqueue_exacloud_models(
        cellist=siteids, batch=batch, modellist=modelnames,
        user=lbhb_user, linux_user=user, force_rerun=force_rerun,
        executable_path=exa_executable_path, script_path=exa_script_path, useGPU=GPU_job)

