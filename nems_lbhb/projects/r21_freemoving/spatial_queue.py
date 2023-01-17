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

executable_path = '/home/svd/bin/miniconda3/envs/tfg/bin/python'
script_path = '/auto/users/svd/python/nems/scripts/fit_single.py'
GPU_job=True

modelnames=[
    "gtgram.fs100.ch18-ld.pop-norm.l1-sev_wc.Nx60-fir.1x20x60-wc.60xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    "gtgram.fs100.ch18.bin-ld.pop-norm.l1-sev_wc.Nx60-fir.1x20x60-wc.60xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    "gtgram.fs100.ch18-ld.pop-norm.l1-sev_wc.Nx40-fir.1x20x40-relu.40.f-wc.40x60-relu.60.f-wc.60xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    "gtgram.fs100.ch18.bin-ld.pop-norm.l1-sev_wc.Nx40-fir.1x20x40-relu.40.f-wc.40x60-relu.60.f-wc.60xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    "gtgram.fs100.ch18-ld.pop-norm.l1-sev_wc.Nx40-fir.1x20x40-relu.40.f-wc.40x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    "gtgram.fs100.ch18.bin-ld.pop-norm.l1-sev_wc.Nx40-fir.1x20x40-relu.40.f-wc.40x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4"
]

modelnames=[
    "gtgram.fs100.ch18-ld.pop-norm.l1-sev_wc.Nx50-fir.1x20x50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    "gtgram.fs100.ch18.bin-ld.pop-norm.l1-sev_wc.Nx50-fir.1x20x50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    "gtgram.fs100.ch18-ld.pop-norm.l1-sev_wc.Nx40-fir.1x25x40-relu.40.f-wc.40x60-relu.60.f-wc.60xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    "gtgram.fs100.ch18.bin-ld.pop-norm.l1-sev_wc.Nx40-fir.1x25x40-relu.40.f-wc.40x60-relu.60.f-wc.60xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
]
modelnames=[
    "gtgram.fs100.ch18-ld.pop-norm.l1-sev_wc.Nx50-fir.1x25x50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    "gtgram.fs100.ch18.bin-ld.pop-norm.l1-sev_wc.Nx50-fir.1x25x50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    "gtgram.fs100.ch18-ld.pop-norm.l1-sev_wc.Nx30-fir.1x25x30-relu.30.f-wc.30x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    "gtgram.fs100.ch18.bin-ld.pop-norm.l1-sev_wc.Nx30-fir.1x25x30-relu.30.f-wc.30x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
]
modelnames=[
    "gtgram.fs100.ch18-ld.pop-norm.l1-sev_wc.Nx50-fir.1x25x50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    "gtgram.fs100.ch18.bin-ld.pop-norm.l1-sev_wc.Nx50-fir.1x25x50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    "gtgram.fs100.ch18-ld.pop-norm.l1-sev_wc.Nx30-fir.1x25x30-relu.30.f-wc.30x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
    "gtgram.fs100.ch18.bin-ld.pop-norm.l1-sev_wc.Nx30-fir.1x25x30-relu.30.f-wc.30x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
]
modelnames=[
    "gtgram.fs100.ch18-ld.pop-norm.l1-sev_wc.Nx40-fir.1x25x40-relu.40.f-wc.40x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4",
    "gtgram.fs100.ch18.bin-ld.pop-norm.l1-sev_wc.Nx40-fir.1x25x40-relu.40.f-wc.40x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4",
    "gtgram.fs100.ch18-ld.pop-norm.l1-sev_wc.Nx30-fir.1x25x30-relu.30.f-wc.30x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4",
    "gtgram.fs100.ch18.bin-ld.pop-norm.l1-sev_wc.Nx30-fir.1x25x30-relu.30.f-wc.30x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4",
]
modelnames=[
    "gtgram.fs100.ch18.mono-ld.pop-norm.l1-sev_wc.Nx40-fir.1x25x40-relu.40.f-wc.40x60-relu.60.f-wc.60xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4",
    "gtgram.fs100.ch18.mono-ld.pop-norm.l1-sev_wc.Nx40-fir.1x25x40-relu.40.f-wc.40x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4",
    "gtgram.fs100.ch18.mono-ld.pop-norm.l1-sev_wc.Nx30-fir.1x25x30-relu.30.f-wc.30x50-relu.50.f-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4",
]
modelnames=[
    "gtgram.fs100.ch18.mono-ld.pop-norm.l1-sev_wc.Nx50-fir.1x25x50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4",
    "gtgram.fs100.ch18-ld.pop-norm.l1-sev_wc.Nx50-fir.1x25x50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4",
    "gtgram.fs100.ch18.bin10-ld.pop-norm.l1-sev_wc.Nx50-fir.1x25x50-wc.50xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4"
]

model_base = "wc.Nx40-fir.1x15x40-relu.40.f-wc.40x40-fir.1x10x40-relu.40.f-wc.40x50-relu.50.f-wc.50xR-lvl.R-dexp.R"
model_base = "wc.Nx40-fir.1x25x40-relu.40.f-wc.40x50-relu.50.f-wc.50xR-lvl.R-dexp.R"
model_base = "wc.Nx50-fir.1x25x50-wc.50xR-lvl.R-dexp.R"
modelnames = [
    f"gtgram.fs100.ch18.mono-ld.pop-norm.l1-sev_{model_base}_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4",
    f"gtgram.fs100.ch18-ld.pop-norm.l1-sev_{model_base}_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4",
    f"gtgram.fs100.ch18.bin10-ld.pop-norm.l1-sev_{model_base}_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4",
    f"gtgram.fs100.ch18.bin6-ld.pop-norm.l1-sev_{model_base}_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4",
    f"gtgram.fs100.ch18.bin10-ld.pop-norm.l1-sev.mono_{model_base}_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4",
    f"gtgram.fs100.ch18.bin10-ld.pop-norm.l1-sev.bin_{model_base}_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4",
    f"gtgram.fs100.ch18.bin10-ld.pop-norm.l1-sev.match_{model_base}_tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4.l2:4",
]
force_rerun = False
run_in_lbhb = False

if run_in_lbhb:
    # first models, run locally so that recordings get generated.
    r = db.enqueue_models(siteids, batch, modelnames, executable_path=executable_path,
                          script_path=script_path, GPU_job=GPU_job, user="svd")
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

