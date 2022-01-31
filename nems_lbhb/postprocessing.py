# LBHB-specific post-processors
# WARNING: Changes to ctx made by functions in this file won't get saved whenthe model is run by fit_model_xform
import logging

import nems.db as nd
import nems.xforms
log = logging.getLogger(__name__)



def add_summary_statistics_by_condition(**context):
    return nems.xforms.add_summary_statistics_by_condition(**context)
    #LAS: This is here for backwards compatibility for old models.
    # For new models the keyword 'ebc' (evaluate by condition) adds nems.xforms.add_summary_statistics_by_condition


def run_decoding_analysis(IsReload=False, **kwargs):
    """
    Specialized postprocessor to queue decoding analysis for the model pred data
    """
    if IsReload:
        log.info("Reload, skipping rda")
        return {}
    modelname = kwargs['meta']['modelname']
    # figure out movement keywords
    threshold = 25
    window = 1
    ops = modelname.split('-')
    for o in ops:
        if o.startswith('mvm'):
            parms = o.split('.')
            for p in parms:
                if p.startswith('t'):
                    threshold = int(p[1:])
                elif p.startswith('w'):
                    window = int(p[1:])
    
    # "base dprime" analysis
    mn = f'dprime_mvm-{threshold}-{window}_jk10_zscore_nclvz_fixtdr2-fa'
    
    # noise dims
    noise = [-1, 0, 1, 2, 3, 4, 5, 6]
    modellist = []
    for n in noise:
        if n > 0:
            modellist.append(mn+f'_noiseDim-{n}')
        elif n == -1:
            modellist.append(mn+f'_noiseDim-dU')
        else:
            modellist.append(mn)

    # append lv modelname
    modellist = [mn+f'_model-LV-{modelname}' for mn in modellist]

    script = '/auto/users/hellerc/code/projects/nat_pupil_ms/dprime_new/cache_dprime.py'
    python_path = '/auto/users/hellerc/anaconda3/envs/lbhb/bin/python'

    nd.enqueue_models(celllist=[kwargs['meta']['cellid'][:7]],
                    batch=kwargs['meta']['batch'],
                    modellist=modellist,
                    executable_path=python_path,
                    script_path=script,
                    user='hellerc',
                    force_rerun=True,
                    reserve_gb=2)
    log.info('Queued decoding analysis')
    return {}

