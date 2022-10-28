import nems0.db as nd
from nems0 import get_setting

from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models

from nems_lbhb.projects.pop_model_scripts.pop_model_utils import load_string_pop, fit_string_pop, load_string_single, fit_string_single,\
    POP_MODELS, SIG_TEST_MODELS, shortnames, shortnamesp, ALL_FAMILY_MODELS, ALL_FAMILY_POP, get_significant_cells, \
    VERSION, HELDOUT, MATCHED, HELDOUT_pop, MATCHED_pop, \
    DNN_SINGLE_MODELS, DNN_SINGLE_STAGE2, LN_SINGLE_MODELS,\
    NAT4_A1_SITES, NAT4_PEG_SITES, MODELGROUPS, POP_MODELGROUPS, count_fits, \
    ALL_TRUNC_MODELS, ALL_TRUNC_POP

# parameters for adding to queue
batches = [322, 323]   # ,334]  # 334 is merged A1+PEG megabatch

force_rerun = False
lbhb_user = "svd"

# exacloud settings:
executable_path_exa = '/home/users/davids/anaconda3/envs/nems/bin/python'
script_path_exa = '/home/users/davids/nems_db/scripts/nems0_scripts/fit_single.py'
ssh_key = '/home/svd/.ssh/id_rsa'
user = "davids"

time_limit_gpu=14
time_limit_cpu=2
reserve_gb=0

modelname_filter = POP_MODELS[1]
mfb = {322: modelname_filter,
       323: modelname_filter.replace('.ver2','')}

# ROUND 1, all families pop
if 0:
    modelnames = ALL_FAMILY_POP[:-1]
    modelnames = ALL_TRUNC_POP
    useGPU = True

    for batch in batches:
        if useGPU and (batch==322):
            c = ['NAT4v2']
        elif useGPU:
            c = ['NAT4']
        else:
            c = nd.batch_comp(modelnames=[modelname_filter], batch=batch).index.to_list()
        enqueue_exacloud_models(
            cellist=c, batch=batch, modellist=modelnames,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun, priority=1,
            executable_path=executable_path_exa, script_path=script_path_exa, useGPU=useGPU)

if 0:
    # dnn single, round 1
    modelnames = DNN_SINGLE_MODELS[:5]

    # ln single, only 1 round
    modelnames = LN_SINGLE_MODELS[:10]

    # dnn single, round 2
    modelnames = DNN_SINGLE_STAGE2[:5]

    useGPU = False
    for batch in batches:
        cellids = nd.batch_comp(modelnames=[mfb[batch]], batch=batch).index.to_list()
        #cellids = [c for c in cellids if c.startswith("TNC")]
        enqueue_exacloud_models(
                cellist=cellids, batch=batch, modellist=modelnames,
                user=lbhb_user, linux_user=user, force_rerun=force_rerun,
                executable_path=executable_path_exa, script_path=script_path_exa, useGPU=useGPU)

#
# round 2, all families single
#
if 0:
    # round 2 all family models
    modelnames = ALL_FAMILY_MODELS
    modelnames = [ALL_FAMILY_MODELS[0]]
    modelnames = ALL_TRUNC_MODELS

    useGPU = False
    for batch in batches:
        c = nd.batch_comp(modelnames=[modelname_filter], batch=batch).index.to_list()
        #c = [_c for _c in c if _c.startswith("TNC")]
        enqueue_exacloud_models(
            cellist=c, batch=batch, modellist=modelnames,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun,
            executable_path=executable_path_exa, script_path=script_path_exa, useGPU=useGPU)


#
# MATCHED/HELDOUT Round 1 - POP, excluding single sites
#

batch_sites = {322: NAT4_A1_SITES, 323: NAT4_PEG_SITES}

if 0:
    modelnames = MATCHED_pop[:-1] + HELDOUT_pop[:-1]
    useGPU = True
    for batch in batches:
        c = batch_sites[batch]
        enqueue_exacloud_models(
            cellist=c, batch=batch, modellist=modelnames,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun,
            executable_path=executable_path_exa, script_path=script_path_exa, useGPU=useGPU)

#
# MATCHED/HELDOUT Round 2 - SINGLE, prefit with excluded sites
#
if 0:
    modelnames = MATCHED + HELDOUT
    useGPU = False
    for batch in batches:
        c = nd.batch_comp(modelnames=[modelname_filter], batch=batch).index.to_list()
        enqueue_exacloud_models(
            cellist=c, batch=batch, modellist=modelnames,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun,
            executable_path=executable_path_exa, script_path=script_path_exa, useGPU=useGPU)

# POP MODELS FOR PARETO PLOT
if 0:
    modelnames = sum([POP_MODELGROUPS[k] for k in POP_MODELGROUPS if k not in ['LN','stp','dnn1_single']], [])
    useGPU = True
    for batch in batches:
        if (VERSION > 1) and (batch==322):
            c = ['NAT4v2']
        elif useGPU:
            c = ['NAT4']
        enqueue_exacloud_models(
            cellist=c, batch=batch, modellist=modelnames,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun,
            time_limit=time_limit_gpu,
            executable_path=executable_path_exa, script_path=script_path_exa, useGPU=useGPU)

    # single-cell fits
    modelnames = sum([POP_MODELGROUPS[k] for k in POP_MODELGROUPS if k in ['LN','stp','dnn1_single']], [])
    useGPU = False
    for batch in batches:
        c = nd.batch_comp(modelnames=[modelname_filter], batch=batch).index.to_list()
        enqueue_exacloud_models(
            cellist=c, batch=batch, modellist=modelnames,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun,
            time_limit=time_limit_cpu,
            executable_path=executable_path_exa, script_path=script_path_exa, useGPU=useGPU)

# STAGE 2 MODELS FOR PARETO PLOT
if 0:
    modelnames = sum([MODELGROUPS[k] for k in MODELGROUPS if k not in ['LN','stp']], [])
    #modelnames=MODELGROUPS['conv2d']

    useGPU = False
    for batch in batches:
        c = nd.batch_comp(modelnames=[modelname_filter], batch=batch).index.to_list()
        enqueue_exacloud_models(
            cellist=c, batch=batch, modellist=modelnames,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun,
            time_limit=time_limit_cpu,
            executable_path=executable_path_exa, script_path=script_path_exa, useGPU=useGPU)


# PARTIAL EST MODELS STAGE 1
if 0:
    from nems_lbhb.projects.pop_model_scripts.pop_model_utils import (
        modelname_half_pop, modelname_half_prefit, modelname_half_fullfit,
        modelname_half_heldoutpop, modelname_half_heldoutfullfit)
    modelnames = modelname_half_pop + modelname_half_heldoutpop
    useGPU = True
    for batch in batches:
        c = batch_sites[batch]
        enqueue_exacloud_models(
            cellist=c, batch=batch, modellist=modelnames,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun,
            executable_path=executable_path_exa, script_path=script_path_exa, useGPU=useGPU)

# PARTIAL EST MODELS STAGE 2
if 0:
    from nems_lbhb.projects.pop_model_scripts.pop_model_utils import (
        modelname_half_pop, modelname_half_prefit, modelname_half_fullfit,
        modelname_half_heldoutpop, modelname_half_heldoutfullfit)

    modelnames = modelname_half_prefit + modelname_half_fullfit + modelname_half_heldoutfullfit
    useGPU = False
    for batch in batches:
        c = nd.batch_comp(modelnames=[modelname_filter], batch=batch).index.to_list()
        enqueue_exacloud_models(
            cellist=c, batch=batch, modellist=modelnames,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun,
            executable_path=executable_path_exa, script_path=script_path_exa, useGPU=useGPU)



#  scratch space
#
if 0:
    modelnames = ALL_FAMILY_POP[:-1]
    modelnames = [m.replace("l2:4","l2:5") for m in modelnames]
    useGPU = True

    for batch in batches:
        if useGPU and (VERSION > 1) and (batch==322):
            c = ['NAT4v2']
        elif useGPU:
            c = ['NAT4']
        else:
            c = nd.batch_comp(modelnames=[modelname_filter], batch=batch).index.to_list()
        enqueue_exacloud_models(
            cellist=c, batch=batch, modellist=modelnames,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun,
            executable_path=executable_path_exa, script_path=script_path_exa, useGPU=useGPU)

    cellid='ARM029a-04-1'
    batch=322

    modelnames = ALL_FAMILY_MODELS[:-1]
    modelnames = [m.replace("l2:4","l2:5") for m in modelnames]

    useGPU = False
    for batch in batches:
        cellids = nd.batch_comp(modelnames=[modelname_filter], batch=batch).index.to_list()
        #cellids = [c for c in cellids if c.startswith("TNC")]
        enqueue_exacloud_models(
            cellist=cellids, batch=batch, modellist=modelnames,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun,
            executable_path=executable_path_exa, script_path=script_path_exa, useGPU=useGPU)


