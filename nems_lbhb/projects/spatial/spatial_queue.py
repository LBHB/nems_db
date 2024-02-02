from pathlib import Path
from scipy.io import wavfile
import matplotlib.pyplot as plt

import nems0.epoch as ep
from nems0 import db
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models

singlecell=True
if singlecell:
    batch2 = 353
    batch1 = 338
    modelnames2 = [
        'gtgram.fs100.ch18.bin6-ld-norm.l1-sev_LN.10xNx3_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4.t3e5',
        ]
    modelnames1 = [
        'gtgram.fs100.ch18.bin100-ld-hrtf-norm.l1-sev_LN.10xNx3_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4.t3e5',
        ]
    cellids = db.get_batch_cells(batch2, as_list=True)

    # exacloud queue settings:
    exa_executable_path = '/home/users/davids/miniconda3/envs/nems/bin/python'
    exa_script_path = '/home/users/davids/nems_db/scripts/nems0_scripts/fit_single.py'
    ssh_key = '/home/svd/.ssh/id_rsa'
    user = "davids"
    lbhb_user = "svd"

    GPU_job=False
    force_rerun = False

    enqueue_exacloud_models(
        cellist=cellids, batch=batch1, modellist=modelnames,
        user=lbhb_user, linux_user=user, force_rerun=force_rerun,
        executable_path=exa_executable_path, script_path=exa_script_path, useGPU=GPU_job)
    enqueue_exacloud_models(
        cellist=cellids, batch=batch2, modellist=modelnames2,
        user=lbhb_user, linux_user=user, force_rerun=force_rerun,
        executable_path=exa_executable_path, script_path=exa_script_path, useGPU=GPU_job)
else:
    batch = 338
    #batch = 353
    siteids, cellids = db.get_batch_sites(batch)

    rank=20
    rank2=30
    if batch==338:
        modelnames = [
            f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_LNpop.20xNxRx{rank}.l2_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4",
            f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_LNpop.20xNxRx{rank2}.l2_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4",
            f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_LNpop.20xNxRx{rank2}.l2:4_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4",
            f"gtgram.fs100.ch18.bin100-ld-hrtf-norm.l1-sev_LNpop.20xNxRx2.i.l2_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4",
        ]
    elif batch==353:
        modelnames = [
            f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_LNpop.20xNxRx{rank}.l2_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4",
            f"gtgram.fs100.ch18.bin6-ld-norm.l1-sev_LNpop.20xNxRx{rank}.l2_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4",
            f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_LNpop.20xNxRx{rank2}.l2_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4",
            f"gtgram.fs100.ch18.bin6-ld-norm.l1-sev_LNpop.20xNxRx{rank2}.l2_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4",
            f"gtgram.fs100.ch18.bin100-ld.pop-hrtf-norm.l1-sev_LNpop.20xNxRx{rank2}.l2:4_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4",
            f"gtgram.fs100.ch18.bin6-ld-norm.l1-sev_LNpop.20xNxRx{rank2}.l2:4_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4",
            f"gtgram.fs100.ch18.bin100-ld-hrtf-norm.l1-sev_LNpop.20xNxRx2.i.l2_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4",
            f"gtgram.fs100.ch18.bin6-ld-norm.l1-sev_LNpop.20xNxRx2.i.l2_lite.tf.init.lr1e3.t3.es20-lite.tf.lr1e4",
    ]

    GPU_job=True

    force_rerun = False
    run_in_lbhb = False

    if run_in_lbhb:
        executable_path = '/home/svd/bin/miniconda3/envs/nems2/bin/python'
        script_path = '/auto/users/svd/python/nems_db/scripts/nems0_scripts/fit_single.py'

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

