#!/usr/bin/env python3

# This script runs xhelp.fit_model_xform from the command line

import os
import sys
from pathlib import Path
import subprocess
import time
import io
import logging
import matplotlib.pyplot as plt
import itertools

log = logging.getLogger(__name__)

force_SDB=True
try:
    if 'SDB-' in sys.argv[3]:
        force_SDB=True
except:
    pass
if force_SDB:
    os.environ['OPENBLAS_VERBOSE'] = '2'
    os.environ['OPENBLAS_CORETYPE'] = 'sandybridge'

import nems0.xform_helper as xhelp
import nems0.utils
from nems0.utils import get_setting
from nems0.uri import save_resource
from nems_lbhb.projects.bignat.bnt_tools import to_sparse4d, to_dense4d, data_subset, \
    get_submodel, save_submodel, btn_generator, load_bnt_recording, pc_fit
from nems0 import db, epoch, initializers, xforms
from nems.tools import json
import nems0.plots.api as n0plt

if force_SDB:
    log.info('Setting OPENBLAS_CORETYPE to sandybridge')


if __name__ == '__main__':

    # leftovers from some industry standard way of parsing inputs

    #parser = argparse.ArgumentParser(description='Generetes the topic vector and block of an author')
    #parser.add_argument('action', metavar='ACTION', type=str, nargs=1, help='action')
    #parser.add_argument('updatecount', metavar='COUNT', type=int, nargs=1, help='pubid count')
    #parser.add_argument('offset', metavar='OFFSET', type=int, nargs=1, help='pubid offset')
    #args = parser.parse_args()
    #action=parser.action[0]
    #updatecount=parser.updatecount[0]
    #offset=parser.offset[0]

    if 'QUEUEID' in os.environ:
        queueid = os.environ['QUEUEID']
        nems0.utils.progress_fun = db.update_job_tick

        if 'SLURM_JOB_ID' in os.environ:
            jobid = os.environ['SLURM_JOB_ID']
            db.update_job_pid(jobid)
            db.update_startdate()
            comment = ' '.join(sys.argv[1:])
            update_comment = ['sacctmgr', '-i', 'modify', 'job', f'jobid={jobid}', 'set', f'Comment="{comment}"']
            subprocess.run(update_comment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log.info(f'Set comment string to: "{comment}"')

    else:
        queueid = 0

    if queueid:
        log.info("Starting QUEUEID={}".format(queueid))
        db.update_job_start(queueid)
        log.info("HOSTNAME={}".format(os.environ.get('HOSTNAME','unknown')))

    if len(sys.argv) < 4:
        print('syntax: fit_single cellid batch modelname')
        exit(-1)

    sitecount = int(sys.argv[1])
    batch = 343
    keywordstub = sys.argv[3]

    log.info(f"Running do_bnt_fit({sitecount},{modelname})")
    savefile = do_bnt_fit(sitecount, modelname, save_results=True)

    log.info("Done with fit.")

    # Mark completed in the queue. Note that this should happen last thing!
    # Otherwise the job might still crash after being marked as complete.
    if db_exists & bool(queueid):
        db.update_job_complete(queueid)

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
