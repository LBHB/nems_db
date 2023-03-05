import time

import numpy as np
import matplotlib.pyplot as plt

import nems
import nems0.db as nd
from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models
import nems0.plots.api as nplt
import nems_lbhb.stateplots as sp
from nems_lbhb.baphy import baphy_load_recording_file
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems_lbhb.baphy_io as io

# from pop_model_utils import (MODELGROUPS, HELDOUT, MATCHED_SITE, LN_HELDOUT, NAT4_A1_SITES,
#                              NAT4_PEG_SITES, HELDOUT_MAX, MATCHED_MAX, CELL_COUNT_TEST, TRUNCATED, TRUNCATED_MATCHED)
from pop_model_utils import MODELGROUPS, POP_MODELGROUPS, NAT4_A1_SITES, NAT4_PEG_SITES


FORCE_RERUN = True
EXACLOUD_SETTINGS = {
    'user': 'jacob',
    'linux_user': 'penningj',
    'executable_path': '/home/users/penningj/python-envs/nems_env/bin/python3',
    'script_path': '/home/users/penningj/code/NEMS/scripts/fit_single.py',
    'force_rerun': FORCE_RERUN,
    #'ssh_key': '/home/jacob/.ssh/id_rsa'
}

# for SVD variables
# executable_path = '/home/users/davids/anaconda3/envs/nems/bin/python'
# script_path = '/home/users/davids/nems/scripts/fit_single.py'
# ssh_key = '/auto/data/exa_keys/davids/id_rsa'
# user = "davids"


def _queue_fits(batch, modelnames, iterator):
    for siteid in iterator:
        for modelname in modelnames:
            do_fit=True
            if not FORCE_RERUN:
                d = nd.pd_query("SELECT * FROM Results WHERE cellid like %s and modelname=%s and batch=%s",
                                params=(siteid+"%", modelname, batch))
                if len(d) > 0:
                    do_fit=False
                    print(f'Fit exists for {siteid} {batch} {modelname}')
            if do_fit:
                enqueue_exacloud_models(cellist=[siteid], batch=batch, modellist=[modelname], useGPU=True,
                                        **EXACLOUD_SETTINGS)


def fit_pop_models(batch):
    modelnames = []
    for k, v in POP_MODELGROUPS.items():
       if ('_single' not in k) and ('_exploration' not in k) and (k != 'LN'):
           modelnames.extend(v)

    iterator = ['NAT4']
    _queue_fits(batch, modelnames, iterator)

    return modelnames


def second_fit_pop_models(batch, start_from=None, test_count=None):
    all_cellids = nd.get_batch_cells(batch, as_list=True)
    if batch == 322:
        sites = NAT4_A1_SITES
    else:
        sites = NAT4_PEG_SITES
    cellids = [c for c in all_cellids if np.any([c.startswith(s.split('.')[0]) for s in sites])]

    modelnames = []
    for k, v in MODELGROUPS.items():
       if ('_single' not in k) and ('_exploration' not in k) and (k != 'LN'):
           modelnames.extend(v)
    iterator = cellids

    for siteid in iterator:
        for modelname in modelnames[start_from:test_count]:
            do_fit=True
            if not FORCE_RERUN:
                d = nd.pd_query("SELECT * FROM Results WHERE cellid like %s and modelname=%s and batch=%s",
                                params=(siteid+"%", modelname, batch))
                if len(d) > 0:
                    do_fit=False
                    print(f'Fit exists for {siteid} {batch} {modelname}')
            if do_fit:
                nd.enqueue_models(celllist=[siteid], batch=batch, modellist=[modelname], user="jacob",
                                  #executable_path='/auto/users/jacob/bin/anaconda3/envs/jacob_nems/bin/python',
                                  executable_path='/auto/users/svd/bin/miniconda3/envs/tf/bin/python',
                                  script_path='/auto/users/jacob/bin/anaconda3/envs/jacob_nems/nems/scripts/fit_single.py')

    return modelnames


def fit_LN_models(batch):
    modelnames = MODELGROUPS['LN']
    if batch == 322:
        sites = NAT4_A1_SITES
    else:
        sites = NAT4_PEG_SITES
    iterator = sites
    _queue_fits(batch, modelnames, iterator)

    return modelnames


def fit_dnn_single(batch, sites):
    modelnames = MODELGROUPS['dnn1_single']
    rec = nems0.recording.load_recording('/auto/data/nems_db/recordings/%s/NAT4_ozgf.fs100.ch18.tgz' % batch)
    cellids = rec['resp'].chans
    iterator = cellids
    _queue_fits(batch, modelnames, iterator)

    return modelnames


def fit_heldout_analysis(batch):
    modelnames = (HELDOUT[batch] + MATCHED_SITE[batch]# + [LN_HELDOUT] + HELDOUT_MAX[batch] + MATCHED_MAX[batch]
                  + TRUNCATED[batch] + TRUNCATED_MATCHED[batch])
    # modelnames = HELDOUT_MAX[batch] + MATCHED_MAX[batch]
    iterator = ['NAT4']
    _queue_fits(batch, modelnames, iterator)

    return modelnames

def fit_cell_counts(batch):
    modelnames = CELL_COUNT_TEST
    iterator = ['NAT4']
    _queue_fits(batch, modelnames, iterator)

    return modelnames


########################################################################################################################
######################        RUN FITS       ###########################################################################
########################################################################################################################

a1 = 322
peg = 323

#m1 = fit_pop_models(a1)
#m2 = fit_pop_models(peg)
#m3 = fit_dnn_single(a1, NAT4_A1_SITES)
#m4 = fit_dnn_single(peg, NAT4_PEG_SITES)
#m5 = fit_heldout_analysis(a1)
#m6 = fit_heldout_analysis(peg)
#m7 = fit_cell_counts(a1)
#m8 = fit_LN_models(a1)
#m9 = fit_LN_models(peg)
m10 = second_fit_pop_models(a1, start_from=2, test_count=None)
m11 = second_fit_pop_models(peg, start_from=2, test_count=None)  # TODO

# some additional test models, add variable learning rate and L2 reg. separately and together
# modelnames = [
#     'ozgf.fs100.ch18.pop-ld-norm.l1-popev_conv2d.10.8x3.rep3-wcn.300-relu.300-wc.300xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.rb10.es20.v-newtf.n.lr1e4.es20.v',
#     'ozgf.fs100.ch18.pop-ld-norm.l1-popev_conv2d.10.8x3.rep3-wcn.300-relu.300-wc.300xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.rb10.es20.v.L2-newtf.n.lr1e4.es20.v.L2',
#     #'ozgf.fs100.ch18.pop-ld-norm.l1-popev_conv2d.10.8x3.rep3-wcn.300-relu.300-wc.300xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.rb10.es20.L2-newtf.n.lr1e4.es20.L2',
# ]
#modelnames = MODELGROUPS['conv2d']
#iterator = ['NAT4']
#_queue_fits(a1, modelnames, iterator)
#_queue_fits(peg, modelnames, iterator)
