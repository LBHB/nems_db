# -*- coding: utf-8 -*-
# code for loading baphy-specific data into xforms models. fit_wrappers
# should have migrated out to xhelp!

import os
import random
import re
import numpy as np
import matplotlib.pyplot as plt
import sys

import nems
import nems.initializers
import nems.epoch as ep
import nems.priors
import nems.modelspec as ms
import nems.plots.api as nplt
import nems.metrics.api
import nems.analysis.api
import nems.utils
import nems_lbhb.baphy as nb
import nems.db as nd
from nems.recording import Recording, load_recording
from nems.fitters.api import dummy_fitter, coordinate_descent, scipy_minimize
import nems.xforms as xforms
import nems.xform_helper as xhelp
from nems_lbhb.old_xforms.xform_wrappers import generate_recording_uri as ogru
import nems_lbhb.old_xforms.xforms as oxf
import nems_lbhb.old_xforms.xform_helper as oxfh
import nems_lbhb.baphy_io as io
from nems import get_setting
from nems_lbhb.baphy_experiment import BAPHYExperiment


import logging
log = logging.getLogger(__name__)


def _matching_cells(batch=289, siteid=None, alt_cells_available=None,
                    cell_count=None, best_cells=False, manual_cell_list=None):

    if batch==289:
       pmodelname = "ozgf.fs100.ch18-ld-sev_dlog-wc.18x3.g-fir.3x15-lvl.1-dexp.1_init-basic"
    else:
       pmodelname = "ozgf.fs100.ch18.pop-ld-norm.l1-popev_wc.18x4R.g-fir.4x25xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.rb5.es20-newtf.n.lr1e4.es20"

    single_perf = nd.batch_comp(batch=batch, modelnames=[pmodelname], stat='r_test')
    if alt_cells_available is not None:
        all_cells = alt_cells_available
    else:
        all_cells = list(single_perf.index)
    log.info("Batch: %d", batch)
    log.info("Per-cell modelname: %s", pmodelname)
    if manual_cell_list is None:
        cellid = [c for c in all_cells if c.split("-")[0]==siteid]
    else:
        cellid = manual_cell_list
    this_perf=np.array([single_perf[single_perf.index==c][pmodelname].values[0] for c in cellid])

    if cell_count is None:
        pass
    elif best_cells:
        sidx = np.argsort(this_perf)
        keepidx=(this_perf >= this_perf[sidx[-cell_count]])
        cellid=list(np.array(cellid)[keepidx])
        this_perf = this_perf[keepidx]
    else:
        cellid=cellid[:cell_count]
        this_perf = this_perf[:cell_count]

    out_cellid = [c for c in all_cells if c.split("-")[0]!=siteid]
    out_perf=np.array([single_perf[single_perf.index==c][pmodelname].values[0]
                       if c in single_perf.index else 0.0
                       for c in out_cellid])

    alt_cellid=[]
    alt_perf=[]
    for i, c in enumerate(cellid):
        d = np.abs(out_perf-this_perf[i])
        w = np.argmin(d)
        alt_cellid.append(out_cellid[w])
        alt_perf.append(out_perf[w])
        out_perf[w]=100 # prevent cell from getting picked again
    log.info("Rand matched cellids: %s", alt_cellid)
    log.info("Mean actual: %.3f", np.mean(this_perf))
    log.info(this_perf)
    log.info("Mean rand: %.3f", np.mean(np.array(alt_perf)))
    log.info(np.array(alt_perf))

    return cellid, this_perf, alt_cellid, alt_perf


def pop_selector(recording_uri_list, batch=None, cellid=None,
                 rand_match=False, cell_count=20, best_cells=False,
                 whiten=True, meta={}, manual_cellids=None, **context):

    rec = load_recording(recording_uri_list[0])

    if type(cellid) is list:
        #if len(cellid[0].split("-"))>1:
        #    manual_cellids = cellid.copy()

        # convert back to siteid
        cellid = cellid[0].split("-")[0]
        
    # Can't do these steps w/o access to LBHB database,
    # so have to skip this step when fitting models locally.
    # TODO: handle this better. Ideally, cellids should just be saved
    # as metadata when recording is generated by baphy.
    if manual_cellids is None:
        log.info('pop_selector: %s', cellid)
        cellid, this_perf, alt_cellid, alt_perf = _matching_cells(
            batch=batch, siteid=cellid, alt_cells_available=rec['resp'].chans,
            cell_count=cell_count, best_cells=best_cells)

        if rand_match == True:
            rec['resp'] = rec['resp'].extract_channels(alt_cellid)
        elif rand_match == 'both':
            rec['resp'] = rec['resp'].extract_channels(cellid + alt_cellid)
        else:
            rec['resp'] = rec['resp'].extract_channels(cellid)

    else:
        # index by count instead of string label
        cellid = manual_cellids[:cell_count]
        if len(manual_cellids) > cell_count:
            log.info("pop_selector: truncating manual_cellids to cell_count")
        if rand_match:
            raise NotImplementedError("pop_selector: random match not implemented for manual cellids")
        else:
            rec['resp'] = rec['resp'].extract_channels(cellid)


    if whiten:
        # normalize mean and std of each channel
        d=rec['resp'].as_continuous().copy()
        d -= np.mean(d, axis=1, keepdims=True)
        d /= np.std(d, axis=1, keepdims=True)
        d -= np.min(d, axis=1, keepdims=True)
        rec['resp'] = rec['resp']._modified_copy(data=d)

    # preserve "actual" cellids for saving to database
    rec.meta['cellid'] = cellid
    del meta['cellid']
    meta['cellids'] = cellid
    return {'rec': rec, 'meta': meta}


def split_pop_rec_by_mask(rec, **contex):

    emask = rec['mask_est']
    emask.name = 'mask'
    vmask = emask._modified_copy(1-emask._data)
    est = rec.copy()
    est.add_signal(emask)
    val=rec.copy()
    val.add_signal(vmask)

    return {'est': est, 'val': val}


def select_cell_count(rec, cell_count, seed_mod=0, exclusions=None, **context):

    cell_set = rec['resp'].chans
    random.seed(12345 + seed_mod)

    if cell_count == 0:
        cell_count = len(cell_set)
    random_selection = random.sample(cell_set, cell_count)
    rec['resp'] = rec['resp'].extract_channels(random_selection)

    if 'mask_est' in rec.signals:
        rec['mask_est'].chans = random_selection
    meta = context['meta']
    meta['cellids'] = random_selection

    return {'rec': rec, 'meta': meta}


def holdout_cells(rec, est, val, exclusions, meta, seed_mod=0, match_to_site=None, **context):
    rec_cells = est['resp'].chans
    batch = int(meta['batch'])
    random.seed(12345 + seed_mod)
    if isinstance(exclusions, int):
        # pick random subset to exclude
        if match_to_site is not None:
            if exclusions == 0:
                if ':' in match_to_site:
                    cellid_options = {'batch': batch, 'cellid': match_to_site, 'rawid': None}
                    cells_to_extract, _ = io.parse_cellid(cellid_options)
                    cell_count = len(cells_to_extract)
                    manual_cell_list = cells_to_extract
                else:
                    cell_count = len(nd.get_batch_cells(batch, cellid=match_to_site, as_list=True))
                    manual_cell_list = None
            else:
                cell_count = exclusions

            cellid, this_perf, alt_cellid, alt_perf = _matching_cells(
                batch=batch, siteid=match_to_site, alt_cells_available=rec['resp'].chans, cell_count=cell_count,
                manual_cell_list=manual_cell_list
            )
            exclusions = alt_cellid
        else:
            exclusions = random.sample(rec_cells, exclusions)
    # else: exclusions should be a list of siteids to exclude
    if match_to_site is None:
        updated_exclusions = []
        for e in exclusions:
            if ':' in e:
                # Have to parse DRX siteids that contain subsite specifications like e1:64
                cellid_options = {'batch': batch, 'cellid': e, 'rawid': None}
                cells_to_extract, _ = io.parse_cellid(cellid_options)
                updated_exclusions.extend(cells_to_extract)
            else:
                updated_exclusions.append(e)

        cell_set = [c for c in rec_cells if not np.any([c.startswith(x) for x in updated_exclusions])]
        holdout_set = list(set(rec_cells) - set(cell_set))
    else:
        cell_set = list(set(rec_cells) - set(exclusions))
        holdout_set = exclusions

    est, holdout_est = _get_holdout_recs(est, cell_set, holdout_set)
    val, holdout_val = _get_holdout_recs(val, cell_set, holdout_set)
    # also have to do rec b/c init from keywords uses it for some checks
    rec, holdout_rec = _get_holdout_recs(rec, cell_set, holdout_set)

    meta['cellids'] = cell_set
    meta['holdout_cellids'] = holdout_set
    meta['matched_site'] = match_to_site

    if exclusions is not None:
        meta['excluded_cellids'] = exclusions

    return {'est': est, 'val': val, 'holdout_est': holdout_est, 'holdout_val': holdout_val,
            'rec': rec, 'holdout_rec': holdout_rec, 'meta': meta}


def _get_holdout_recs(rec, cell_set, holdout_set) -> object:
    holdout_rec = rec.copy()
    rec['resp'] = rec['resp'].extract_channels(cell_set)
    holdout_rec['resp'] = holdout_rec['resp'].extract_channels(holdout_set)

    if 'mask_est' in rec.signals:
        rec['mask_est'].chans = cell_set
        holdout_rec['mask_est'].chans = holdout_set

    return rec, holdout_rec


def switch_to_heldout_data(holdout_est, holdout_val, holdout_rec, meta, modelspec, trainable_layers=None,
                           use_matched_site=False, **context):
    '''Make heldout data the "primary" for final fit. Requires `holdout_cells` during preprocessing.'''
    if use_matched_site:
        site = meta['matched_site']
        batch = meta['batch']
        cellids = nd.get_batch_cells(batch, cellid=site, as_list=True)
        meta['cellids'] = cellids
    else:
        meta['cellids'] = meta['holdout_cellids']
    modelspec.meta['cellids'] = meta['holdout_cellids']

    # Reinitialize trainable layers so that .R options are adjusted to new cell count
    temp_ms = nems.initializers.from_keywords(meta['modelspecname'], rec=holdout_rec, input_name=context['input_name'],
                                              output_name=context['output_name'])
    temp_ms[0].pop('meta')  # don't overwrite metadata in first module
    if trainable_layers is None:
        trainable_layers = list(range(len(temp_ms)))
    for i in trainable_layers:
        modelspec[i].update(temp_ms[i])  # overwrite phi, kwargs, etc

    return {'est': holdout_est, 'val': holdout_val, 'rec': holdout_rec, 'modelspec': modelspec, 'meta': meta}


def pop_file(stimfmt='ozgf', batch=None,
             rasterfs=50, chancount=18, siteid=None, **options):

    siteid = siteid.split("-")[0]
    if ((batch==272) and (siteid=='none')) or (siteid in ['bbl086b','TAR009d','TAR010c','TAR017b']):
        subsetstr = "NAT1"
    elif siteid in ['none', 'NAT3', 'AMT003c','AMT005c','AMT018a','AMT020a','AMT023d',
                    'bbl099g','bbl104h',
                    'BRT026c','BRT032e','BRT033b','BRT034f','BRT037b','BRT038b','BRT039c',
                    'AMT031a','AMT032a']:
        # Should use NAT3 as siteid going forward for better readability,
        # but left other options here for backwards compatibility.
        subsetstr = "NAT3"
    elif siteid == 'NAT4':
        subsetstr = "NAT4"
    else:
        raise ValueError('site not known for popfile')

    use_API = get_setting('USE_NEMS_BAPHY_API')

    uri_path = '/auto/data/nems_db/recordings/'
    recname="{}_{}.fs{}.ch{}".format(subsetstr, stimfmt, rasterfs, chancount)
    data_file = '{}{}/{}.tgz'.format(uri_path, batch, recname)

    if use_API:
        p, f = os.path.split(data_file)
        host = 'http://'+get_setting('NEMS_BAPHY_API_HOST')+":"+str(get_setting('NEMS_BAPHY_API_PORT'))
        recording_uri = host + '/recordings/' + str(batch) + '/' + f
    else:
        recording_uri = data_file

    return recording_uri


def generate_recording_uri(cellid=None, batch=None, loadkey=None,
                           siteid=None, force_old_loader=False, **options):
    """
    required parameters (passed through to nb.baphy_data_path):
        cellid: string or list
            string can be a valid cellid or siteid
            list is a list of cellids from a single(?) site
        batch: integer

    figure out filename (or eventually URI) of pre-generated
    NEMS-format recording for a given cell/batch/loader string

    very baphy-specific. Needs to be coordinated with loader processing
    in nems.xform_helper
    """
    # remove any preprocessing keywords in the loader string.
    if '-' in loadkey:
        loader = nems.utils.escaped_split(loadkey, '-')[0]
    else:
        loader = loadkey
    log.info('loader=%s',loader)

    ops = loader.split(".")

    # updates some some defaults
    options.update({'rasterfs': 100, 'chancount': 0})
    load_pop_file = False

    for op in ops:
        if op=='ozgf':
            options['stimfmt'] = 'ozgf'
        elif op=='parm':
            options['stimfmt'] = 'parm'
        elif op=='ll':
            options['stimfmt'] = 'll'
        elif op=='env':
            options['stimfmt'] = 'envelope'
        elif op in ['nostim','psth','ns', 'evt']:
            options.update({'stim': False, 'stimfmt': 'parm'})

        elif op.startswith('fs'):
            options['rasterfs'] = int(op[2:])
        elif op.startswith('ch'):
            options['chancount'] = int(op[2:])

        elif op.startswith('fmap'):
            options['facemap'] = int(op[4:])

        elif op=='pup':
            options.update({'pupil': True, 'rem': 1})
            #options.update({'pupil': True, 'pupil_deblink': True,
            #                'pupil_deblink_dur': 1,
            #                'pupil_median': 0, 'rem': 1})
        elif op=='rem':
            options['rem'] = True

        elif 'eysp' in ops:
            options['pupil_eyespeed'] = True
        elif op.startswith('pop'):
            load_pop_file = True
        elif op == 'voc':
            options.update({'runclass': 'VOC'})

    if 'stimfmt' not in options.keys():
        raise ValueError('Valid stim format (ozgf, psth, parm, env, evt) not specified in loader='+loader)
    if (options['stimfmt']=='ozgf') and (options['chancount'] <= 0):
        raise ValueError('Stim format ozgf requires chancount>0 (.chNN) in loader='+loader)

    if int(batch) in [263,294]:
        options["runclass"] = "VOC"

    if siteid is not None:
        options['siteid'] = siteid

    options["batch"] = batch
    options["cellid"] = cellid

    if load_pop_file:
        recording_uri = pop_file(siteid=cellid, **options)
    elif force_old_loader: # | (batch==307):
        log.info("Using 'old' baphy.py loader")
        recording_uri, _ = nb.baphy_load_recording_uri(**options)
    else:
        manager = BAPHYExperiment(batch=batch, cellid=cellid)
        recording_uri = manager.get_recording_uri(**options)

    return recording_uri


def baphy_load_wrapper(cellid=None, batch=None, loadkey=None,
                       siteid=None, normalize=False, options={}, **context):
    # check for special pop signal code
    pc_idx = None
    if type(cellid) is str:
        cc=cellid.split("_")
        if (len(cc) > 1) and (cc[1][0]=="P"):
            pc_idx=[int(cc[1][1:])]
            cellid=cc[0]
        elif (len(cellid.split('+')) > 1):
            # list of cellids (specified in model queue by separating with '_')
            cellid = cellid.split('+')

    recording_uri = generate_recording_uri(cellid=cellid, batch=batch,
                                           loadkey=loadkey, siteid=siteid, **options)

    # update the cellid in context so that we don't have to parse the cellid
    # again in xforms
    t_ops = options.copy()
    t_ops['cellid'] = cellid
    t_ops['batch'] = batch
    cells_to_extract, _ = io.parse_cellid(t_ops)
    context = {'recording_uri_list': [recording_uri], 'cellid': cells_to_extract}

    if pc_idx is not None:
        context['pc_idx'] = pc_idx

    return context

def load_existing_pred(cellid=None, siteid=None, batch=None, modelname_existing=None, **kwargs):
    """
    designed to be called by xforms keyword loadpred 
    cellid/siteid - one or the other required
    batch - required
    default modelname_existing = "psth.fs4.pup-ld-st.pup-hrc-psthfr-aev_sdexp2.SxR_newtf.n.lr1e4.cont.et5.i50000"
    
    makes new signal 'pred0' from evaluated 'pred', returns in updated rec
    returns ctx-compatible dict {'rec': nems.Recording, 'input_name': 'pred0'}
    """
    if (batch is None):
        raise ValueError("must specify cellid/siteid and batch")

    if cellid is None:
        if siteid is None:
            raise ValueError("must specify cellid/siteid and batch")
        d = nd.pd_query("SELECT batch,cellid FROM Batches WHERE batch=%s AND cellid like %s",
                        (batch, siteid+"%",))
        cellid=d['cellid'].values[0]
    elif type(cellid) is list:
        cellid = cellid[0]

    if modelname_existing is None:
        #modelname_existing = "psth.fs4.pup-ld-st.pup-hrc-psthfr-aev_sdexp2.SxR_newtf.n.lr1e4.cont"
        modelname_existing = "psth.fs4.pup-ld-st.pup-hrc-psthfr-aev_sdexp2.SxR_newtf.n.lr1e4.cont.et5.i50000"

    xf,ctx = xhelp.load_model_xform(cellid, batch, modelname_existing)
    for k in ctx['val'].signals.keys():
        if k not in ctx['rec'].signals.keys():
           ctx['rec'].signals[k] = ctx['val'].signals[k].copy()
    s = ctx['rec']['pred'].copy()
    s.name='pred0'
    ctx['rec'].add_signal(s)

    #return {'rec': ctx['rec'],'val': ctx['val'],'est': ctx['est']}
    return {'rec': ctx['rec'], 'input_name': 'pred0'}

def model_pred_comp(cellid, batch, modelnames, occurrence=None,
                    pre_dur=None, dur=None):
    """
    return ccd12, ccd13, ccd23
    """

    modelcount = len(modelnames)
    epoch = 'REFERENCE'
    c = 0
    plot_colors = ['lightgray', 'g', 'lightgray', 'r', 'lightgray', 'b']
    legend = ['act','LN','act','GC','act','STP']
    times = []
    values = []
    r_test = []
    for i, m in enumerate(modelnames):
        xf, ctx = load_model_baphy_xform(cellid, batch, m)

        val = ctx['val'][0]

        if i == 0:
            d = val['resp'].get_epoch_bounds('PreStimSilence')
            if len(d):
                PreStimSilence = np.mean(np.diff(d))
            else:
                PreStimSilence = 0
            if pre_dur is None:
                pre_dur = PreStimSilence

            if occurrence is not None:
                # Get values from specified occurrence and channel
                extracted = val['resp'].extract_epoch(epoch)
                r_vector = extracted[occurrence][c]
            else:
                r_vector = val['resp'].as_continuous()[0, :]

            validbins = np.isfinite(r_vector)
            r_vector = nems.utils.smooth(r_vector[validbins], 7)
            r_vector = r_vector[3:-3]

            # Convert bins to time (relative to start of epoch)
            # TODO: want this to be absolute time relative to start of data?
            time_vector = np.arange(0, len(r_vector)) / val['resp'].fs - PreStimSilence

            # limit time range if specified
            good_bins = (time_vector >= -pre_dur)
            if dur is not None:
                good_bins[time_vector > dur] = False

        if occurrence is not None:
            extracted = val['pred'].extract_epoch(epoch)
            p_vector = extracted[occurrence][c]
        else:
            p_vector = val['pred'].as_continuous()
            p_vector = p_vector[0, validbins]

        times.append(time_vector[good_bins])
        values.append(r_vector[good_bins] + i)
        times.append(time_vector[good_bins])
        values.append(p_vector[good_bins] + i)

        r_test.append(ctx['modelspec'].meta['r_test'][0])

    times_all = times
    values_all = values

    cc12 = np.corrcoef(values_all[0], values_all[1])[0, 1]
    cc13 = np.corrcoef(values_all[0], values_all[2])[0, 1]
    cc23 = np.corrcoef(values_all[1], values_all[2])[0, 1]
    ccd23 = np.corrcoef(values_all[1]-values_all[0],
                        values_all[2]-values_all[0])[0, 1]

#    ccd12 = np.corrcoef(values_all[0]-values_all[3],
#                        values_all[1]-values_all[3])[0, 1]
#    ccd13 = np.corrcoef(values_all[0]-values_all[3],
#                        values_all[2]-values_all[3])[0, 1]
#    ccd23 = np.corrcoef(values_all[1]-values_all[3],
#                        values_all[2]-values_all[3])[0, 1]

    print("CC LN-GC: {:.3f}  LN-STP: {:.3f} STP-GC: {:.3f}".format(
            cc12,cc13,cc23))
#    print("CCd LN-GC: {:.3f}  LN-STP: {:.3f} STP-GC: {:.3f}".format(
#            ccd12,ccd13,ccd23))

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    if occurrence is not None:
        extracted = val['stim'].extract_epoch(epoch)
        sg = extracted[occurrence]
    else:
        sg = val['stim'].as_continuous()
        sg = sg[:, validbins]
    sg = sg[:, good_bins]
    nplt.plot_spectrogram(sg, val['resp'].fs, ax=ax,
                          title='{} Stim {}'.format(cellid, occurrence),
                          time_offset=pre_dur, cmap='gist_yarg')

    ax = plt.subplot(2, 1, 2)
    title = 'Preds LN {:.3f} GC {:.3f} STP {:.3f} /CC LN-GC: {:.3f}  LN-STP: {:.3f} STP-GC: {:.3f} dSTP-dGC: {:.3f}'.format(
            r_test[0],r_test[1],r_test[2],cc12,cc13,cc23,ccd23)
    nplt.plot_timeseries(times_all, values_all, ax=ax, legend=legend,
                         title=title, colors=plot_colors)

    plt.tight_layout()

    return cc12, cc13, cc23, ccd23


def load_batch_modelpaths(batch, modelnames, cellids=None, eval_model=True):
    d = nd.get_results_file(batch, [modelnames], cellids=cellids)
    return d['modelpath'].tolist()


def quick_inspect(cellid="chn020f-b1", batch=271,
                  modelname="ozgf100ch18_wc18x1_fir15x1_lvl1_dexp1_fit01"):
    """
    DEPRECATED? pretty much replaced by xhelp.load_model_xform()
    """
    xf, ctx = load_model_baphy_xform(cellid, batch, modelname, eval_model=True)

    modelspec = ctx['modelspec']
    est = ctx['est']
    val = ctx['val']
    nplt.quickplot(ctx)

    return modelspec, est, val
