import nems.db as nd
from nems import get_setting

from nems_lbhb.exacloud.queue_exacloud_job import enqueue_exacloud_models

from nems_lbhb.projects.pop_model_scripts.pop_model_utils import load_string_pop, fit_string_pop, load_string_single, fit_string_single,\
    POP_MODELS, SIG_TEST_MODELS, shortnames, shortnamesp, ALL_FAMILY_MODELS, ALL_FAMILY_POP, get_significant_cells, \
    VERSION2_FLAG, HELDOUT, MATCHED, HELDOUT_pop, MATCHED_pop, \
    DNN_SINGLE_MODELS, DNN_SINGLE_STAGE2, LN_SINGLE_MODELS, STP_SINGLE_MODELS,\
    NAT4_A1_SITES, NAT4_PEG_SITES, MODELGROUPS, POP_MODELGROUPS

# parameters for adding to queue
if VERSION2_FLAG:
    batches = [322]
else:
    batches = [322, 323]
    # ,334]  # 334 is merged A1+PEG megabatch

force_rerun = False
lbhb_user = "svd"

# exacloud settings:
executable_path_exa = '/home/users/davids/anaconda3/envs/nems/bin/python'
script_path_exa = '/home/users/davids/nems/scripts/fit_single.py'
ssh_key = '/home/svd/.ssh/id_rsa'
user = "davids"

modelname_filter = POP_MODELS[2]

# ROUND 1, all families pop
if 0:
    modelnames = ALL_FAMILY_POP[:-1]
    useGPU = True

    for batch in batches:
        if useGPU and VERSION2_FLAG:
            c = ['NAT4v2']
        elif useGPU:
            c = ['NAT4']
        else:
            c = nd.batch_comp(modelnames=[modelname_filter], batch=batch).index.to_list()
        enqueue_exacloud_models(
            cellist=c, batch=batch, modellist=modelnames,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun,
            executable_path=executable_path_exa, script_path=script_path_exa, useGPU=useGPU)

if 0:
    # dnn single, round 1
    modelnames = DNN_SINGLE_MODELS

    # ln single, only 1 round
    modelnames = LN_SINGLE_MODELS

    # dnn single, round 2
    modelnames = DNN_SINGLE_STAGE2

    useGPU = False
    for batch in batches:
        cellids = nd.batch_comp(modelnames=[modelname_filter], batch=batch).index.to_list()
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
if VERSION2_FLAG:
    batch_sites = {322: NAT4_A1_SITES}
else:
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

    modelnames = MATCHED_pop[-1:]
    useGPU = False
    for batch in batches:
        c = nd.batch_comp(modelnames=[modelname_filter], batch=batch).index.to_list()
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
        if VERSION2_FLAG:
            c = ['NAT4v2']
        elif useGPU:
            c = ['NAT4']
        enqueue_exacloud_models(
            cellist=c, batch=batch, modellist=modelnames,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun,
            executable_path=executable_path_exa, script_path=script_path_exa, useGPU=useGPU)

    # single-cell fits
    modelnames = sum([POP_MODELGROUPS[k] for k in POP_MODELGROUPS if k in ['LN','stp','dnn1_single']], [])
    useGPU = False
    for batch in batches:
        c = nd.batch_comp(modelnames=[modelname_filter], batch=batch).index.to_list()
        enqueue_exacloud_models(
            cellist=c, batch=batch, modellist=modelnames,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun,
            executable_path=executable_path_exa, script_path=script_path_exa, useGPU=useGPU)

# STAGE 2 MODELS FOR PARETO PLOT
if 0:
    modelnames = sum([MODELGROUPS[k] for k in MODELGROUPS if k not in ['LN','stp']], [])
    useGPU = False
    for batch in batches:
        c = nd.batch_comp(modelnames=[modelname_filter], batch=batch).index.to_list()
        enqueue_exacloud_models(
            cellist=c, batch=batch, modellist=modelnames,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun,
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
    modelnames = [m.replace("tfinit.n.lr1e3.et3.rb10.es20",
                            "tfinit.n.lr1e3.et3.rb10.es20.l2,4") for m in modelnames]
    useGPU = True

    for batch in batches:
        if useGPU and VERSION2_FLAG:
            c = ['NAT4v2']
        elif useGPU:
            c = ['NAT4']
        else:
            c = nd.batch_comp(modelnames=[modelname_filter], batch=batch).index.to_list()
        enqueue_exacloud_models(
            cellist=c, batch=batch, modellist=modelnames,
            user=lbhb_user, linux_user=user, force_rerun=force_rerun,
            executable_path=executable_path_exa, script_path=script_path_exa, useGPU=useGPU)


