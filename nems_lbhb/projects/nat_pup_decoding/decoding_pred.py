#!/usr/bin/env python3

# This script runs do_decoding_analysis using a previously fit
# LV model from the command line

import os
import sys
import logging
from pathlib import Path
import subprocess
log = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt

import nems0.xform_helper as xhelp
import nems0.utils
from nems0.uri import save_resource
from nems0 import get_setting

from nems0 import xform_helper
from nems_lbhb.projects.nat_pup_decoding import do_decoding

force_SDB=True
try:
    if 'SDB-' in sys.argv[3]:
        force_SDB=True
except:
    pass
if force_SDB:
    os.environ['OPENBLAS_VERBOSE'] = '2'
    os.environ['OPENBLAS_CORETYPE'] = 'sandybridge'


if force_SDB:
    log.info('Setting OPENBLAS_CORETYPE to sandybridge')

try:
    from nems0 import db
    db_exists = True
except Exception as e:
    # If there's an error import nems0.db, probably missing database
    # dependencies. So keep going but don't do any database stuff.
    print("Problem importing nems0.db, can't update tQueue")
    print(e)
    db_exists = False

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
        print('syntax: decoding_pred cellid batch modelname')
        exit(-1)

    #try:
    cellid = sys.argv[1]
    batch = int(sys.argv[2])
    modelname = sys.argv[3]
    #except:
    #    cellid = 'AMT020a-02-1'
    #    batch = 331
    #    modelname = 'psth.fs4.pup-ld-epcpn-hrc-psthfr.z-pca.cc1.no.p-st.pup+r3+s3-plgsm.p2-aev_stategain.2xR.x2,3,4-spred-lvnorm.5xR.so.x1,2-inoise.5xR.x1,3,4_tfinit.xx0.n.lr1e4.cont.et4.i20000-lvnoise.r2-aev-ccnorm.t4.f0'

    try:
        use_signal = sys.argv[4]
    except:
        use_signal = 'pred'

    hist_norm = True
    log.info("Running do_decoding.do_decoding_analysis({0},{1},{2})".format(cellid, batch, modelname))
    xfspec, ctx = xform_helper.load_model_xform(cellid, batch, modelname)

    tdr_pred = do_decoding.do_decoding_analysis(lv_model=True, hist_norm=hist_norm, **ctx)
    tdr_resp = do_decoding.do_decoding_analysis(lv_model=False, hist_norm=hist_norm, **ctx)
    dtemp = tdr_resp.numeric_results.merge(tdr_pred.numeric_results,
                                           how='inner',left_index=True, right_index=True,
                                           suffixes=('_a','_p'))
    f, ax = plt.subplots(1, 2, figsize=(6, 3), sharex='col', sharey='col')
    md = np.max([np.max(dtemp['bp_dp_a']), np.max(dtemp['bp_dp_p'])])/2

    ax[0].plot([0, md*2], [0, md*2], 'k--')
    ax[0].scatter(dtemp['sp_dp_a'], dtemp['bp_dp_a'],s=3)
    ax[0].scatter(dtemp['sp_dp_p'], dtemp['bp_dp_p'],s=3)

    a = dtemp['bp_dp_p'] - dtemp['sp_dp_p']
    b = dtemp['bp_dp_a'] - dtemp['sp_dp_a']
    cc = np.corrcoef(a, b)[0, 1]
    E = np.sqrt(np.sum((a-b)**2))/np.sqrt(np.sum(b**2))
    ax[1].plot([-md, md], [-md, md],'k--')
    ax[1].scatter(a, b, s=3)

    ax[0].set_title(db.get_siteid(cellid))
    ax[1].set_title(f"cc={cc:.3f}  E={E:.3f}")
    fig_filepath = ctx['modelspec'].meta['modelpath'] + '/figure.0001.png'
    f.savefig(fig_filepath)
    log.info("Done with decoding analysis.")

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



if 0:

    ALL_SITES = ['BOL005c', 'BOL006b', 'bbl086b', 'bbl099g', 'bbl104h', 'BRT026c', 'BRT034f',  'BRT036b', 'BRT038b',
             'BRT039c', 'TAR010c', 'TAR017b', 'AMT005c', 'AMT018a', 'AMT019a',
             'AMT020a', 'AMT021b', 'AMT023d', 'AMT024b',
             'DRX006b.e1:64', 'DRX006b.e65:128',
             'DRX007a.e1:64', 'DRX007a.e65:128',
             'DRX008b.e1:64', 'DRX008b.e65:128',
             'CRD016d', 'CRD017c',
             'TNC008a','TNC009a', 'TNC010a', 'TNC012a', 'TNC013a', 'TNC014a',
             'TNC015a', 'TNC016a', 'TNC017a', 'TNC018a', 'TNC020a']

    batch = 322
    siteids_,cellids = db.get_batch_sites(batch=batch)

    siteids = [s for s,c in zip(siteids_,cellids) if s in ALL_SITES]
    cellids = [c for s,c in zip(siteids_,cellids) if s in ALL_SITES]

    print(siteids)
    states = ['st.pup+r3+s0,1,2,3','st.pup+r3+s1,2,3','st.pup+r3+s2,3','st.pup+r3+s3','st.pup+r3']
    modelnames = [f"psth.fs4.pup-ld-norm.sqrt-hrc-psthfr.z-pca.cc1.no.p-{s}-plgsm.p2-aev" + \
                  "_stategain.2xR.x2,3,4-spred-lvnorm.5xR.so.x1,2-inoise.5xR.x1,3,4" + \
                  "_tfinit.xx0.n.lr1e4.cont.et4.i20000-lvnoise.r2-aev-ccnorm.t4.f0"
                  for s in states]

    for cellid in cellids:
        for modelname in modelnames:

            args = f"{cellid} {batch} {modelname}"
            user = 'svd'
            note = f"{args.replace(' ','/')}/decoding_pred"
            executable_path='/home/svd/bin/miniconda3/envs/nems_cpu/bin/python'
            script_path='/auto/users/svd/python/nems_db/nems_lbhb/projects/nat_pup_decoding/decoding_pred.py'

            r = db.add_job_to_queue([args], note, user=user,
                                    executable_path=executable_path,
                                    script_path=script_path)
            for q,m in r:
                print(m)

