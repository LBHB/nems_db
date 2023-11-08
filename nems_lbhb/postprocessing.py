# LBHB-specific post-processors
# WARNING: Changes to ctx made by functions in this file won't get saved whenthe model is run by fit_model_xform
import logging

import nems0.db as nd
import nems0.xforms

import nems_lbhb.projects.nat_pup_decoding.do_decoding as decoding

log = logging.getLogger(__name__)



def add_summary_statistics_by_condition(**context):
    return nems0.xforms.add_summary_statistics_by_condition(**context)
    #LAS: This is here for backwards compatibility for old models.
    # For new models the keyword 'ebc' (evaluate by condition) adds nems0.xforms.add_summary_statistics_by_condition


def run_decoding(use_pred, **ctx):
    """
    Run decoding analysis on LV model results.
    For now, just use hardcoded dprime analysis options. These could be modified with the 
    rd keyword in the future.

    CRH 2022.05.21
    """
    if use_pred:
        log.info("Running decoding analysis on pred data")
    else:
        log.info("Running decoding analysis on raw data")
    success = decoding.do_decoding_analysis(lv_model=use_pred, **ctx)
    # decoding results get saved in their own file, so don't really need to return anything
    if success == 0:
        return ctx
    else:
        raise ValueError("Decoding analysis failed")


def run_decoding_analysis(IsReload=False, **kwargs):
    """
    Specialized postprocessor to queue decoding analysis for the model pred data
    """
    raise DeprecationWarning("Use 'run_decoding'. It is cleaner.")
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

def save_pred_signal(**ctx):
    """
    Saves the model prediction for the validation set
    """
    rec = ctx['val'].copy()
    rec.signals = {key:sig for key, sig in rec.signals.items() if key == 'pred'}
    rec.signal_views = [rec.signals]

    savefile = ctx['modelspec'].meta['modelpath'] + '/' + 'prediction.tar.gz'
    rec.save(str(savefile))

    return ctx


def dstrf_pca(rec, est, val, modelspec, model_list=None,
              D=15, timestep=3 **ctx)

    if model_list is None:
        model_list = [modelspec]
    cellids = rec['resp'].chans

    t_indexes = np.arange(time_step, val['stim'].shape[1], timestep)
    log.info(f"Computing dSTRF at {len(t_indexes)} timepoints,timestep={timestep}")

    out_channels = np.arange(len(cellids))
    stim = {'input': val['stim'].as_continuous().T}
    dstrfs = []
    for mi, m in enumerate(model_list):
        d = m.dstrf(stim, D=D, out_channels=out_channels, t_indexes=t_indexes, reset_backend=False)
        dstrfs.append(d['input'])

    dstrf = np.stack(dstrfs, axis=1)
    s = np.std(dstrf, axis=(2, 3, 4), keepdims=True)
    dstrf /= s
    dstrf /= np.max(np.abs(dstrf)) * 0.9

    mdstrf = dstrf.mean(axis=1, keepdims=True)
    sdstrf = dstrf.std(axis=1, keepdims=True)
    sdstrf[sdstrf == 0] = 1
    mzdstrf = shrinkage(mdstrf, sdstrf, sigrat=0.75)

    mdstrf /= np.max(np.abs(mdstrf)) * 0.9
    mzdstrf /= np.max(np.abs(mzdstrf)) * 0.9
    pc_count = 7
    dpc, dpc_mag = dtools.compute_dpcs(mdstrf[:, 0], pc_count=pc_count)
    dpcz, dpc_magz = dtools.compute_dpcs(mzdstrf[:, 0], pc_count=pc_count)

    imopts = {'cmap': 'bwr', 'vmin': -1, 'vmax': 1, 'origin': 'lower',
              'interpolation': 'none'}
    imoptsz = {'cmap': 'bwr', 'origin': 'lower',
               'interpolation': 'none'}
    # plt.close('all')
    f, ax = plt.subplots(len(out_channels), pc_count, figsize=(pc_count, len(out_channels)), sharex=True, sharey=True)
    f2, ax2 = plt.subplots(len(out_channels), pc_count, figsize=(pc_count, len(out_channels)), sharex=True, sharey=True)
    for oi, oc in enumerate(out_channels):
        for di in range(pc_count):
            d = dpc[oi, di]
            d = d / np.max(np.abs(d)) / dpc_mag[0, oi] * dpc_mag[di, oi]
            ax[oi, di].imshow(d, **imopts)
            d = dpcz[oi, di]
            d = d / np.max(np.abs(d)) / dpc_magz[0, oi] * dpc_magz[di, oi]
            ax2[oi, di].imshow(d, **imopts)

