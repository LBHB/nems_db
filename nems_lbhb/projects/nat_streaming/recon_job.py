#!/usr/bin/env python3

# This script runs xhelp.fit_model_xform from the command line

import os
import sys
import logging
from pathlib import Path
import subprocess
import nems0.xform_helper as xhelp
import nems0.utils
from nems0.uri import save_resource
from nems0 import get_setting
from nems_lbhb.projects.nat_streaming.recon_tools import nmse, corrcoef, recon_site_stim, get_cluster_data

log = logging.getLogger(__name__)

from pathlib import Path

import nems0.uri

# CONFIGURATION

# figure out data and results paths:
signals_dir = Path(nems0.NEMS_PATH) / 'recordings'
modelspecs_dir = Path(nems0.NEMS_PATH) / 'modelspecs'

force_SDB = True
try:
    if 'SDB-' in sys.argv[3]:
        force_SDB = True
except:
    pass
if force_SDB:
    os.environ['OPENBLAS_VERBOSE'] = '2'
    os.environ['OPENBLAS_CORETYPE'] = 'sandybridge'

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

    # leftovers from some industry standard way of parsing inputs

    # parser = argparse.ArgumentParser(description='Generetes the topic vector and block of an author')
    # parser.add_argument('action', metavar='ACTION', type=str, nargs=1, help='action')
    # parser.add_argument('updatecount', metavar='COUNT', type=int, nargs=1, help='pubid count')
    # parser.add_argument('offset', metavar='OFFSET', type=int, nargs=1, help='pubid offset')
    # args = parser.parse_args()
    # action=parser.action[0]
    # updatecount=parser.updatecount[0]
    # offset=parser.offset[0]

    if 'QUEUEID' in os.environ:
        queueid = os.environ['QUEUEID']
        nems0.utils.progress_fun = nd.update_job_tick

    else:
        queueid = 0

    if queueid:
        log.info("Starting QUEUEID={}".format(queueid))
        nd.update_job_start(queueid)
        log.info("HOSTNAME={}".format(os.environ.get('HOSTNAME', 'unknown')))

    if len(sys.argv) < 4:
        print('syntax: recon_job.py siteid batch estim cluster_count groupby')
        exit(-1)

    siteid = sys.argv[1]
    batch = int(sys.argv[2])
    estim = sys.argv[3]
    cluster_count = int(sys.argv[4])
    groupby = sys.argv[5]
    shuffle_count = 11
    modeltype = 'LN'
    df_recon = recon_site_stim(siteid, estim, cluster_count=cluster_count,
                               batch=batch, modeltype=modeltype, groupby=groupby,
                               shuffle_count=shuffle_count, force_rerun=False)

    log.info("Done with recon.")

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

                dst_prefix = r'http://' + get_setting('NEMS_BAPHY_API_HOST') + ":" + str(get_setting('NEMS_BAPHY_API_PORT'))
                dst_loc = dst_prefix + '/queuelog/' + str(queueid)
                save_resource(str(dst_loc), data=log_data)
