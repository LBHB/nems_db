from pathlib import Path
from scipy.io import wavfile
import matplotlib.pyplot as plt

from nems_lbhb.baphy_experiment import BAPHYExperiment

from nems0.analysis.gammatone.gtgram import gtgram
import nems0.epoch as ep
from nems0 import db
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models

batch = 348
rasterfs = 50

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

dlc_count=12
dlc1 = 40
strf_channels=20

dlc_memory=3
acount=25
dcount=12
l2count=30
tcount=acount+dcount
input_count = 36

# choose whether or not allow .o.s in intermediate relus
ros="" # ros=".o.s" # ros=""
# regularize wc layers? L2, 10^-4
reg=".l2:4"
#reg=""
sep_kw = f'wcst.Nx1x{acount}.i{reg}-wcdl.{dlc_count}x1x{dcount}.i{reg}-first.8x1x{acount}-firdl.{dlc_memory}x1x{dcount}.nc1-cat-relu.{tcount}{ros}'
aud_kw = f'wc.{tcount}x1x{l2count}{reg}-fir.4x1x{l2count}-relu.{l2count}{ros}-wc.{l2count}xR{reg}-relu.R.o.s'
model_kw_old = sep_kw + '-' + aud_kw

hrtf_kw = f'wcdl.{dlc_count}x{dlc1}.i-relud.{dlc1}{ros}-wcdl.{dlc1}x10-relud.10.o.s-wcdl.10x5-relud.5.o.s-wcdl.5x{input_count}-sigd.{input_count}.s.g-mult'
aud_kw = f'wch.Nx1x{strf_channels}-fir.10x1x{strf_channels}-relu.{strf_channels}-wc.{strf_channels}x1x{l2count}-fir.10x1x{l2count}-relu.{l2count}-wc.{l2count}xR-relu.R.o.s'
model_kw_new = hrtf_kw + '-' + aud_kw

model_kw_ln = f'wc.Nx1x{l2count}-fir.10x1x{l2count}-wc.{l2count}xR-relu.R.o.s'

# dlc effects from stategaindl (normal wc keywords for stim-->pred, wcdl/-s words to handle the dlc-->state path)
sep_kw = f'wcdl.{dlc_count}x1x{dcount}.i.s{reg}-firs.{dlc_memory}x1x{dcount}.nc1-relus.{dcount}{ros}-wcs.{dcount}x{dcount}{reg}-relus.{dcount}{ros}'
aud_kw = f'wc.Nx1x{acount}.i{reg}-fir.8x1x{acount}-relu.{acount}{ros}-wc.{acount}x1x{l2count}{reg}-fir.4x1x{l2count}-relu.{l2count}{ros}-wc.{l2count}xR{reg}-stategain.{dcount+1}xR-relu.R.o.s'
model_kw_sg = sep_kw + '-' + aud_kw

load_kw_shuff = f"free.fs{rasterfs}.ch18-norm.l1-fev-shuf.dlc"
load_kw = f"free.fs{rasterfs}.ch18-norm.l1-fev"
load_kw_hrtf = f"free.fs{rasterfs}.ch18-norm.l1-fev.hrtf"
load_kw_hrtf_shuff = f"free.fs{rasterfs}.ch18-norm.l1-fev.hrtf-shuf.dlc"
load_kw_hrtfae_shuff = f"free.fs{rasterfs}.ch18-norm.l1-fev.hrtfae-shuf.dlc"

jkn=6
load_kw_jk = f"free.fs{rasterfs}.ch18-norm.l1-fev.jk{jkn}"
load_kw_shuff_jk = f"free.fs{rasterfs}.ch18-norm.l1-fev.jk{jkn}-shuf.dlc"
load_kw_hrtf_jk = f"free.fs{rasterfs}.ch18-norm.l1-fev.hrtf.jk{jkn}"
load_kw_hrtf_shuff_jk = f"free.fs{rasterfs}.ch18-norm.l1-fev.hrtf.jk{jkn}-shuf.dlc"
load_kw_hrtfae_shuff_jk = f"free.fs{rasterfs}.ch18-norm.l1-fev.hrtfae.jk{jkn}-shuf.dlc"

rbn=3
fit_kw = "lite.tf.cont.init.lr1e3.t3-lite.tf.cont.lr1e4"
fit_kw_jk = f"lite.tf.cont.init.lr1e3.t3.rb{rbn}-lite.tf.cont.lr1e4.t5e4"  # .t3"

modelnames=[
#    "_".join([load_kw,model_kw_new,fit_kw]),
#    "_".join([load_kw_shuff, model_kw_new, fit_kw]),
#    "_".join([load_kw_jk, model_kw_new, fit_kw_jk]),
    "_".join([load_kw_hrtf_jk, model_kw_ln, fit_kw_jk]),
    "_".join([load_kw_shuff_jk, model_kw_old, fit_kw_jk]),
    "_".join([load_kw_hrtf_jk, model_kw_old, fit_kw_jk]),
    "_".join([load_kw_hrtf_shuff_jk, model_kw_old, fit_kw_jk]),
    "_".join([load_kw_hrtf_jk, model_kw_sg, fit_kw_jk]),
    "_".join([load_kw_hrtf_shuff_jk, model_kw_sg, fit_kw_jk]),
    "_".join([load_kw_hrtfae_shuff_jk, model_kw_old, fit_kw_jk]),
]

force_rerun = False
run_in_lbhb = False
mock_run = False

if mock_run:
    shortnames = [
        # 'HRTF+DLC new nojk',
        # 'HRTF+DLC new',
        # 'HRTF+Dsh new',
        'HRTF+LN',
        'Dsh old',
        'HRTF+DLC old',
        'HRTF+Dsh old',
        'HRTF+DLC sg',
        'HRTF+Dsh sg',
        'HRTFae+Dsh old',
    ]
    modelname = modelnames[3]
    modelname2 = modelnames[6]

    for i, m in enumerate(modelnames):
        if m == modelname2:
            print(f'**{i:2d} {shortnames[i]:12s}  {m}')
        elif m == modelname:
            print(f'* {i:2d} {shortnames[i]:12s}  {m}')
        else:
            print(f'  {i:2d} {shortnames[i]:12s}  {m}')

elif run_in_lbhb:
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

