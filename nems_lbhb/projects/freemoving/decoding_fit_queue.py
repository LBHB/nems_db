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

executable_path = '/auto/users/svd/python/nems_db/nems_lbhb/projects/freemoving/bash_fit_wrapper'
script_path = '/auto/users/svd/python/nems_db/nems_lbhb/projects/freemoving/free_moving_fit_script.py'
GPU_job=True

modelnames = ['sh.none-hrtf.True__nmse',
              'sh.none-hrtf.False__nmse',
              'sh.dlc-hrtf.True__nmse',
              'sh.dlc-hrtf.False__nmse',
              'sh.stim-hrtf.True__nmse',
              ]
force_rerun = False
run_in_lbhb = True

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
    exa_executable_path = '/home/users/davids/anaconda3/envs/nems/bin/python'
    exa_script_path = '/home/users/davids/nems_db/scripts/nems0_scripts/fit_single.py'
    ssh_key = '/home/svd/.ssh/id_rsa'
    user = "davids"
    lbhb_user = "svd"

    enqueue_exacloud_models(
        cellist=siteids, batch=batch, modellist=modelnames,
        user=lbhb_user, linux_user=user, force_rerun=force_rerun,
        executable_path=exa_executable_path, script_path=exa_script_path, useGPU=GPU_job)

