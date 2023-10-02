
import os
import sys
import logging
from pathlib import Path
import subprocess
log = logging.getLogger(__name__)

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

signals_dir = Path(nems0.NEMS_PATH) / 'recordings'
modelspecs_dir = Path(nems0.NEMS_PATH) / 'modelspecs'
import importlib
from nems_lbhb.projects.spatial import STRFfunction

importlib.reload(STRFfunction)

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

    parms = modelname.split("_")
    loadparms = parms[0].split('-')
    loadops = {op.split(".")[0]: op.split(".")[1] for op in loadparms}
    modelopts = {'dlc_memory': 4, 'acount': 12, 'dcount': 8, 'l2count': 24, 'cost_function': 'squared_error'}

    rec = free_model.load_free_data(siteid, batch=batch, rasterfs=rasterfs, dlc_chans=dlc_chans, compute_position=True)

    model = free_model.free_fit(rec, save_to_db=True, **loadops, **modelopts)

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


# tested sites
siteid = 'PRN020a'
siteid = 'PRN010a'
siteid = 'PRN015a'
siteid = 'PRN034a'
siteid = 'PRN018a'
siteid = 'PRN022a'
siteid = 'PRN043a'
siteid = 'PRN051a'

# interesting sites
siteid = 'PRN067a' # ok both
siteid = 'PRN015a' # nice aud, single stream
siteid = 'PRN047a' # some of everything.
siteid = 'PRN074a' # ok both

siteid = 'SLJ021a'
siteid = 'PRN048a'  # some of everything.


batch=348
df = db.pd_query(f"SELECT DISTINCT modelname,modelfile FROM Results WHERE cellid like '{siteid}%' and batch={batch}")

dlc_chans=8
rasterfs=50
batch=348
rec = free_model.load_free_data(siteid, batch=batch, rasterfs=rasterfs, dlc_chans=dlc_chans, compute_position=True)

modelopts={'dlc_memory': 4, 'acount': 20, 'dcount': 10, 'l2count': 24, 'cost_function': 'squared_error'}
model = free_model.free_fit(rec, shuffle='none', apply_hrtf=True, save_to_db=True, **modelopts)
model2 = free_model.free_fit(rec, shuffle='none', apply_hrtf=False, save_to_db=True, **modelopts)

for i,c in enumerate(model2.meta['cellids']):
    print(f"{i}: {c} {model.meta['r_test'][i,0]:.3f} {model2.meta['r_test'][i,0]:.3f} {rec.meta['depth'][i]}")
print(f"MEAN              {model.meta['r_test'].mean():.3f} {model2.meta['r_test'].mean():.3f}")

# scatter plot of free-moving position with example positions highlighted
f = free_vs_fixed_strfs.movement_plot(rec)

ctx1=free_model.free_split_rec(rec, apply_hrtf=True)
ctx2=free_model.free_split_rec(rec, apply_hrtf=False)
est1 = ctx1['est'].apply_mask()
est2 = ctx2['est'].apply_mask()

# dSTRFS for interesting units: PRN048a-269-1, PRN048a-285-2
#for out_channel in [20,22]:
for out_channel in [6, 8, 5]:
    cellid = rec['resp'].chans[out_channel]
    mdstrf, pc1, pc2, pc_mag = free_vs_fixed_strfs.dstrf_snapshots(rec, [model, model2], D=11, out_channel=out_channel)
    f = free_vs_fixed_strfs.dstrf_plots(rec, [model, model2], mdstrf, out_channel)


modelpath = model.meta['modelfile']
modeltest = json.load_model(modelpath)
dlc_count = rec['dlc'].shape[0]



input = {'stim': val['stim'].as_continuous().T, 'dlc': val['dlc'].as_continuous().T[:, :dlc_count]}

testpred = modeltest.predict(input)['prediction']
origpred = model.predict(input)['prediction']

plt.figure()
plt.plot(testpred[:1000,6])
plt.plot(origpred[:1000,6])

f,ax=plt.subplots(2,1)
ax[0].imshow(est1['stim']._data[:,:500])
ax[1].imshow(est2['stim']._data[:,:500])


