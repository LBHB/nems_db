from pathlib import Path
from scipy.io import wavfile
import matplotlib.pyplot as plt

from nems_lbhb.baphy_experiment import BAPHYExperiment

from nems0.analysis.gammatone.gtgram import gtgram
import nems0.epoch as ep
from nems0 import db
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models

batch = 338
siteids, cellids = db.get_batch_sites(batch)

#enqueue_models(celllist, batch, modellist, force_rerun=False,
#                   user="nems", codeHash="master", jerbQuery='',
#                   executable_path=None, script_path=None,
#                   priority=1, GPU_job=0, reserve_gb=0)

executable_path = '/home/svd/bin/miniconda3/envs/nems2/bin/python'
script_path = '/auto/users/svd/python/nems_db/scripts/nems0_scripts/fit_single.py'
GPU_job=True

rank=8

modelnames = [
    f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_wc.Nx1x{rank}-fir.20x1x{rank}-wc.{rank}xR-dexp.R_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4",
]


force_rerun = False
run_in_lbhb = False

if run_in_lbhb:
    # first models, run locally so that recordings get generated.
    r = db.enqueue_models(siteids[:3], batch, modelnames, executable_path=executable_path,
                          script_path=script_path, GPU_job=GPU_job, user="svd")
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

