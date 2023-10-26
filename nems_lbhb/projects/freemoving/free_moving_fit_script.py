
import os
import sys
import logging
from pathlib import Path
import subprocess

from os.path import basename, join
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter, zoom

from nems0 import db
import nems0.epoch as ep
import nems0.plots.api as nplt
from nems0.utils import smooth
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.plots import plot_waveforms_64D
from nems_lbhb.preprocessing import impute_multi
from nems_lbhb import baphy_io
from nems_lbhb.projects.freemoving.free_tools import compute_d_theta, \
    free_scatter_sum, dlc2dist
from nems_lbhb.projects.freemoving import free_model, free_vs_fixed_strfs
from nems.tools import json

log = logging.getLogger(__name__)

force_SDB = True
try:
    if 'SDB-' in sys.argv[3]:
        force_SDB = True
except:
    pass
if force_SDB:
    os.environ['OPENBLAS_VERBOSE'] = '2'
    os.environ['OPENBLAS_CORETYPE'] = 'sandybridge'

import nems0.xform_helper as xhelp
import nems0.utils
from nems0.uri import save_resource
from nems0 import get_setting

if force_SDB:
    log.info('Setting OPENBLAS_CORETYPE to sandybridge')

try:
    import nems0.db as nd

    db_exists = True
except Exception as e:
    # If there's an error import nems0.db, probably missing database
    # dependencies. So keep going but don't do any database stuff.
    print("Problem importing nems0.db, can't update tQueue")
    print(e)
    db_exists = False

if __name__ == '__main__':

    if 'QUEUEID' in os.environ:
        queueid = os.environ['QUEUEID']
        nems0.utils.progress_fun = nd.update_job_tick
        if 'SLURM_JOB_ID' in os.environ:
            jobid = os.environ['SLURM_JOB_ID']
            nd.update_job_pid(jobid)
            nd.update_startdate()
            comment = ' '.join(sys.argv[1:])
            update_comment = ['sacctmgr', '-i', 'modify', 'job', f'jobid={jobid}', 'set', f'Comment="{comment}"']
            subprocess.run(update_comment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log.info(f'Set comment string to: "{comment}"')
    else:
        queueid = 0

    if queueid:
        log.info("Starting QUEUEID={}".format(queueid))
        nd.update_job_start(queueid)
        log.info("HOSTNAME={}".format(os.environ.get('HOSTNAME', 'unknown')))

    if len(sys.argv) < 4:
        print('syntax: fit_spatial_site.py siteid batch modelname')
        exit(-1)

    siteid = sys.argv[1]
    batch = int(sys.argv[2])
    modelname = sys.argv[3]

    dlc_chans = 8
    rasterfs = 50
    cost_function = 'squared_error'
    parms = modelname.split("_")
    loadparms = parms[0].split('-')
    loadops = {'apply_hrtf': True}
    if len(parms)>2:
        cost_function = parms[2]

    for op in loadparms:
        k = op.split(".")[0]
        v = op.split(".")[1]
        if k=='sh':
            loadops['shuffle'] = v
        elif k=='hrtf':
            if v.lower()=='true':
                loadops['apply_hrtf'] = True
            elif v.lower()=='false':
                loadops['apply_hrtf'] = False
    log.info(f"site/bactch/model: {siteid}/{batch}/{modelname}")
    log.info(f"{loadops}")
    modelopts = {'dlc_memory': 4, 'acount': 12, 'dcount': 8, 'l2count': 24, 'cost_function': cost_function}

    rec = free_model.load_free_data(siteid, batch=batch, rasterfs=rasterfs,
                                    dlc_chans=dlc_chans, compute_position=True)

    model = free_model.free_fit(rec, save_to_db=True, **loadops, **modelopts)

    ctx = free_model.free_split_rec(rec, apply_hrtf=loadops['apply_hrtf'])
    rec = ctx['rec']
    est = ctx['est'].apply_mask()
    val = ctx['val'].apply_mask()

    pc_mags = []
    mdstrfs = []
    for out_channel in range(rec['resp'].shape[0]):
        mdstrf, pc1, pc2, pc_mag = free_vs_fixed_strfs.dstrf_snapshots(rec, [model], D=11, out_channel=out_channel, pc_count=5)
        pc_mags.append(pc_mag)  # unit x model x didx x pc
        mdstrfs.append(mdstrf)   # unit x model x didx x frequency x lag

    pc_mags = np.stack(pc_mags, axis=0)
    mdstrfs = np.stack(mdstrfs, axis=0)
    # difference between front and back dstrfs
    fbdiff = mdstrfs[:,:,2,:,:]-mdstrfs[:,:,0,:,:]
    fbmod = fbdiff.std(axis=(2,3))/(mdstrfs[:,:,0,:,:].std(axis=(2,3))+mdstrfs[:,:,2,:,:].std(axis=(2,3)))*2
    cellids = rec['resp'].chans
    r_test = model.meta['r_test']
    r_floor = model.meta['r_floor']

    outpath = model.meta['modelpath']
    dfile = os.path.join(outpath, 'dstrf.npz')
    log.info(f"Saving dstrf data to {dfile}")
    np.savez(dfile, pc_mags=pc_mags, mdstrfs=mdstrfs, fbmod=fbmod,
             cellids=cellids, r_test=r_test, r_floor=r_floor, modelname=modelname)

    log.info("Done with fit.")

    # Mark completed in the queue. Note that this should happen last thing!
    # Otherwise the job might still crash after being marked as complete.
    if db_exists & bool(queueid):
        nd.update_job_complete(queueid)

        if 'SLURM_JOB_ID' in os.environ:
            # need to copy the job log over to the queue log dir
            log_file_dir = Path.home() / 'job_history'
            log_file = list(log_file_dir.glob(f'*jobid{os.environ["SLURM_JOB_ID"]}_log.out'))
            if len(log_file) == 1:
                log_file = log_file[0]
                log.info(f'Found log file: "{str(log_file)}"')
                log.info('Copying log file to queue log repo.')

                with open(log_file, 'r') as f:
                    log_data = f.read()

                dst_prefix = r'http://' + get_setting('NEMS_BAPHY_API_HOST') + ":" + str(
                    get_setting('NEMS_BAPHY_API_PORT'))
                dst_loc = dst_prefix + '/queuelog/' + str(queueid)
                save_resource(str(dst_loc), data=log_data)

    # code to support dumping figures
    #dt = datetime.date.today().strftime("%Y-%m-%d")
    #figpath = f'/auto/users/svd/docs/current/grant/r21_free_moving/eps/{dt}/'
    #os.makedirs(figpath, exist_ok=True)


