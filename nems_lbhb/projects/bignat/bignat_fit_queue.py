from pathlib import Path
from scipy.io import wavfile
import matplotlib.pyplot as plt

from nems_lbhb.baphy_experiment import BAPHYExperiment

from nems0.analysis.gammatone.gtgram import gtgram
import nems0.epoch as ep
from nems0 import db
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models

batch=343
siteids, cellids = db.get_batch_sites(batch)
#siteids=['PRN048a']


load_kw = 'gtgram.fs100.ch18-ld-norm.l1-sev'
fit_kw = 'lite.tf.init.lr1e3.t3.es20.jk3-lite.tf.lr1e4.t5e4-dstrf.d15.t43.p5.ss'
modelnames = [
    f'{load_kw}_wc.18x1x70.g-fir.15x1x70-relu.70.f-wc.70x1x80.l2:4-fir.10x1x80-relu.80.f-wc.80x100.l2:4-relu.100.s-wc.100xR.l2:4-dexp.R_{fit_kw}',
    f'{load_kw}_wc.18x1x120.g-fir.25x1x120-wc.120xR.l2:4-dexp.R_{fit_kw}',
    f'{load_kw}_wc.18x1x6.g-fir.25x1x6-relu.6.f-wc.6x1-dexp.1_{fit_kw}'
]
shortnames = ['CNN 1d','LN','CNN single']

modelnames = [modelnames[1]]
#siteids = ["CLT028c"]

executable_path = '/auto/users/svd/python/nems_db/nems_lbhb/projects/freemoving/bash_fit_wrapper'
#script_path = '/auto/users/svd/python/nems_db/nems_lbhb/projects/freemoving/free_moving_fit_script.py'
script_path = '/auto/users/svd/python/nems_db/scripts/fit_single.py'
GPU_job=True

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

