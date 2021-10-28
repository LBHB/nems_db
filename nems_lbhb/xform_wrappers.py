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
       pmodelname = "ozgf.fs100.ch18.pop-ld-norm.l1-popev_wc.18x4R.g-fir.4x25xR-lvl.R-dexp.R_tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4"

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

    out_cellid = [c for c in all_cells if c not in cellid]#c.split("-")[0]!=siteid]
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
                 whiten=True, meta={}, manual_cellids=None, holdout=None,
                 **context):

    if (cellid == "ALLCELLS"):
        rec_list = [load_recording(uri) for uri in recording_uri_list]
        for r in rec_list:
            if 'cellids' not in r.meta.keys():
                channels = r['resp'].chans
                cellids = [str(c) for c in channels]
                r.meta['cellids'] = cellids

        return {'rec': rec_list[0], 'rec_list': rec_list}
    
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
        
    if holdout is not None:
        all_cells = rec['resp'].chans
        siteid=cellid
        log.info('pop_selector: %s holdout: %s', cellid, holdout)
        this_cellid, this_perf, alt_cellid, alt_perf = _matching_cells(
            batch=batch, siteid=siteid, alt_cells_available=all_cells,
            cell_count=None, best_cells=None)
        
        if holdout == 'site':
            cellids = [c for c in all_cells if c not in this_cellid]
        elif holdout == 'matched':
            cellids = [c for c in all_cells if c not in alt_cellid]

        # fake the cellids so that entries get saved to celldb for them
        meta['cellids'] = this_cellid
        
        meta['holdout'] = holdout
        meta['holdout_cellids'] = this_cellid
        meta['matched_cellids'] = alt_cellid
        meta['matched_site'] = siteid
        rec['resp'] = rec['resp'].extract_channels(cellids)
        
        return {'rec': rec, 'meta': meta}
    
    elif manual_cellids is None:
        log.info('pop_selector: %s', cellid)
        cellid, this_perf, alt_cellid, alt_perf = _matching_cells(
            batch=batch, siteid=cellid, alt_cells_available=rec['resp'].chans,
            cell_count=cell_count, best_cells=best_cells)
        
        if len(cellid)==0:
            cellid = rec['resp'].chans
        elif rand_match == True:
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


def split_pop_rec_by_mask(rec, rec_list=None, keepfrac=1, **context):

    if rec_list is None:
        rec_list = [rec]
        return_reclist = False
    else:
        rec=rec_list[0]
        return_reclist = True
    est_list = []
    val_list = []
    for rec in rec_list:
        emask = rec['mask_est']
        emask.name = 'mask'
        vmask = emask._modified_copy(1-emask._data)
        if keepfrac<1:
            d=emask._data[0,:].copy()
            nmask=d.sum()
            nkeep=int(np.ceil(nmask*keepfrac/150)*150)
            m=np.argwhere(d)
            print(d.sum())
            d[m[nkeep][0]:]=0
            log.info(f'reducing emask by {keepfrac} {nkeep}/{nmask}')
            emask=emask._modified_copy(data=d)
        est = rec.copy()
        est.add_signal(emask)
        val = rec.copy()
        val.add_signal(vmask)
        
        est_list.append(est)
        val_list.append(val)
    if return_reclist:
        return {'est': est_list[0], 'val': val_list[0], 'est_list': est_list, 'val_list': val_list}
    else:
        return {'est': est, 'val': val}


def select_cell_count(rec, cell_count, seed_mod=0, exclusions=None, **context):

    cell_set = rec['resp'].chans
    random.seed(12345 + seed_mod)

    if cell_count == 0:
        cell_count = len(cell_set)

    rec['resp'] = rec['resp'].extract_channels(random_selection)

    if 'mask_est' in rec.signals:
        rec['mask_est'].chans = random_selection
    meta = context['meta']
    meta['cellids'] = random_selection

    return {'rec': rec, 'meta': meta}


def max_cells(rec, est, val, meta, n_cells, seed_mod=0, **context):
    '''
    Similar to holdout_cells, but for fitting up to n_cells and does not separately save cells that are not removed.
    '''

    rec_cells = est['resp'].chans
    random.seed(12345 + seed_mod)

    if meta['matched_site'] is None:
        keep_these_cells = random.sample(rec_cells, n_cells)
    else:
        # if ':' in matched_site:
        #     cellid_options = {'batch': meta['batch'], 'cellid': matched_site, 'rawid': None}
        #     cells_to_extract, _ = io.parse_cellid(cellid_options)
        #     site_cells = cells_to_extract
        # else:
        #     site_cells = [c for c in rec_cells if c.startswith(matched_site)]
        matched_cells = meta['matched_cellids']
        n_site = len(matched_cells)

        if n_cells <= n_site:
            keep_these_cells = random.sample(matched_cells, n_cells)
        else:
            keep_these_cells = random.sample(rec_cells, n_cells-n_site) + matched_cells

    est, keep_these_est = _get_holdout_recs(est, keep_these_cells, None)
    val, keep_these_val = _get_holdout_recs(val, keep_these_cells, None)
    rec, keep_these_rec = _get_holdout_recs(rec, keep_these_cells, None)

    meta['cellids'] = keep_these_cells

    return {'est': est, 'val': val, 'rec': rec, 'meta': meta}


def holdout_cells(rec, est, val, site, meta, exclude_matched_cells=False, **context):

    batch = int(meta['batch'])
    rec_cells = est['resp'].chans

    if ':' in site:
        cellid_options = {'batch': batch, 'cellid': site, 'rawid': None}
        cellids, _ = io.parse_cellid(cellid_options)
        cell_count = len(cellids)
    else:
        cellids = [c for c in rec_cells if c.startswith(site)]
        cell_count = len(cellids)

    _, _, alt_cellid, _ = _matching_cells(
                # alt_cells_available = rec_cells   # causes problems if not all cells in rec have been fit
                batch=batch, siteid=site, alt_cells_available=None, cell_count=cell_count,
                manual_cell_list=cellids
    )

    matched_set = alt_cellid
    holdout_set = cellids

    cell_set1 = list(set(rec_cells) - set(matched_set))
    cell_set2 = list(set(rec_cells) - set(holdout_set))

    # first rec all except holdout_set, second rec only holdout_set
    est1, holdout_est = _get_holdout_recs(est, cell_set1, holdout_set)
    val1, holdout_val = _get_holdout_recs(val, cell_set1, holdout_set)
    rec1, holdout_rec = _get_holdout_recs(rec, cell_set1, holdout_set)

    # first rec all except matched_set, second rec only matched_set
    est2, matched_est = _get_holdout_recs(est, cell_set2, matched_set)
    val2, matched_val = _get_holdout_recs(val, cell_set2, matched_set)
    rec2, matched_rec = _get_holdout_recs(rec, cell_set2, matched_set)

    if exclude_matched_cells:
        cell_set = cell_set1
        new_est, new_val, new_rec = (est1, val1, rec1)
    else:
        cell_set = cell_set2
        new_est, new_val, new_rec = (est2, val2, rec2)

    meta['cellids'] = cell_set
    meta['holdout_cellids'] = holdout_set
    meta['matched_cellids'] = matched_set
    meta['matched_site'] = site

    return {'est': new_est, 'val': new_val, 'holdout_est': holdout_est, 'holdout_val': holdout_val,
            'rec': new_rec, 'holdout_rec': holdout_rec, 'meta': meta, 'matched_est': matched_est,
            'matched_val': matched_val, 'matched_rec': matched_rec}


def _get_holdout_recs(rec, cell_set, holdout_set=None) -> object:
    rec = rec.copy()
    holdout_rec = rec.copy()
    rec['resp'] = rec['resp'].extract_channels(cell_set)
    if holdout_set is not None:
        holdout_rec['resp'] = holdout_rec['resp'].extract_channels(holdout_set)

    if 'mask_est' in rec.signals:
        rec['mask_est'].chans = cell_set
        if holdout_set is not None:
            holdout_rec['mask_est'].chans = holdout_set

    return rec, holdout_rec


def switch_to_heldout_data(meta, modelspec, freeze_layers=None, use_matched_recording=False, use_same_recording=False,
                           IsReload=False, **context):
    '''Make heldout data the "primary" for final fit. Requires `holdout_cells` during preprocessing.'''

    if use_matched_recording:
        new_est = context['matched_est']
        new_val = context['matched_val']
        new_rec = context['matched_rec']
        cellids = meta['holdout_cellids']  # this intentionally doesn't match the recording, for easier comparisons
    elif use_same_recording:
        # for dummy LN version, just resets parameters for frozen layers
        new_est = context['est']
        new_val = context['val']
        new_rec = context['rec']
        cellids = meta['cellids']
    else:
        new_est = context['holdout_est']
        new_val = context['holdout_val']
        new_rec = context['holdout_rec']
        cellids = meta['holdout_cellids']

    meta['cellids'] = cellids
    modelspec.meta['cellids'] = meta['cellids']

    if IsReload:
        log.info('Skipping reinitialization of modelspec on reload')
    else:
        # Reinitialize trainable layers so that .R options are adjusted to new cell count
        temp_ms = nems.initializers.from_keywords(meta['modelspecname'], rec=new_rec, input_name=context['input_name'],
                                                  output_name=context['output_name'])
        temp_ms[0].pop('meta')  # don't overwrite metadata in first module
        all_idx = list(range(len(temp_ms)))
        if freeze_layers is None:
            freeze_layers = all_idx
        for i in all_idx:
            if i not in freeze_layers:
                modelspec[i].update(temp_ms[i])  # overwrite phi, kwargs, etc

    return {'est': new_est, 'val': new_val, 'rec': new_rec, 'modelspec': modelspec, 'meta': meta,
            'freeze_layers': freeze_layers}


def pop_file(stimfmt='ozgf', batch=None, cellid=None,
             rasterfs=50, chancount=18, siteid=None, loadkey=None, **options):

    siteid = siteid.split("-")[0]
    subsetstr=[]
    sitelist=[]
    if siteid == 'ALLCELLS':
        if (batch in [322]):
            subsetstr = ["NAT4", "NAT3", "NAT1"]
        elif (batch in [323]):
            subsetstr = ["NAT4"]
        elif (batch in [333]):
            runclass="OLP"
            sql="SELECT sRunData.cellid,gData.svalue,gData.rawid FROM sRunData INNER JOIN" +\
                    " sCellFile ON sRunData.cellid=sCellFile.cellid " +\
                    " INNER JOIN gData ON" + \
                    " sCellFile.rawid=gData.rawid AND gData.name='Ref_Combos'" +\
                    " AND gData.svalue='Manual'" +\
                    " INNER JOIN gRunClass on gRunClass.id=sCellFile.runclassid" +\
                    f" WHERE sRunData.batch={batch} and gRunClass.name='{runclass}'"
            d = nd.pd_query(sql)

            d['siteid'] = d['cellid'].apply(nd.get_siteid)
            sitelist = d['siteid'].unique()
        else:
            raise ValueError(f'ALLCELLS not supported for batch {batch}')
    elif ((batch==272) and (siteid=='none')) or (siteid in ['bbl086b','TAR009d','TAR010c','TAR017b']):
        subsetstr = ["NAT1"]
    elif siteid in ['none', 'NAT3', 'AMT003c','AMT005c','AMT018a','AMT020a','AMT023d',
                    'bbl099g','bbl104h',
                    'BRT026c','BRT032e','BRT033b','BRT034f','BRT037b','BRT038b','BRT039c',
                    'AMT031a','AMT032a']:
        # Should use NAT3 as siteid going forward for better readability,
        # but left other options here for backwards compatibility.
        subsetstr = ["NAT3"]
    elif (batch in [322,323]) or (siteid == 'NAT4'):
        subsetstr = [siteid]
    else:
        raise ValueError('site not known for popfile')
    use_API = get_setting('USE_NEMS_BAPHY_API')

    uri_root = '/auto/data/nems_db/recordings/'
    
    recording_uri_list = []
    log.info("TRUNCATING AT FIVE RECORDINGS")
    for s in sitelist[:5]:
        recording_uri = generate_recording_uri(batch=batch, cellid=s, stimfmt=stimfmt,
                     rasterfs=rasterfs, chancount=chancount, **options)
        log.info(f'loading {recording_uri}')
        #if use_API:
        #    host = 'http://'+get_setting('NEMS_BAPHY_API_HOST')+":"+str(get_setting('NEMS_BAPHY_API_PORT'))
        #    recording_uri = host + '/recordings/' + str(batch) + '/' + recname + '.tgz'
        #else:
        #    recording_uri = '{}{}/{}.tgz'.format(uri_root, batch, recname)
        recording_uri_list.append(recording_uri)
    for s in subsetstr:
        recname=f"{s}_{stimfmt}.fs{rasterfs}.ch{chancount}"
        log.info(f'loading {recname}')
        #data_file = '{}{}/{}.tgz'.format(uri_root, batch, recname)

        if use_API:
            host = 'http://'+get_setting('NEMS_BAPHY_API_HOST')+":"+str(get_setting('NEMS_BAPHY_API_PORT'))
            recording_uri = host + '/recordings/' + str(batch) + '/' + recname + '.tgz'
        else:
            recording_uri = '{}{}/{}.tgz'.format(uri_root, batch, recname)
        recording_uri_list.append(recording_uri)
    if len(subsetstr)==1:
        return recording_uri
    else:
        return recording_uri_list


def generate_recording_uri(cellid=None, batch=None, loadkey="",
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
    if loader != '':
        log.info('loader=%s',loader)

    ops = loader.split(".")

    # updates some some defaults
    options['rasterfs'] = options.get('rasterfs', 100)
    options['chancount'] = options.get('chancount', 0)
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
    elif force_old_loader: #  | (batch==316):
        log.info("Using 'old' baphy.py loader")
        recording_uri, _ = nb.baphy_load_recording_uri(**options)
    else:
        manager = BAPHYExperiment(batch=batch, cellid=cellid)
        recording_uri = manager.get_recording_uri(**options)

    return recording_uri


def baphy_load_wrapper(cellid=None, batch=None, loadkey=None,
                       siteid=None, normalize=False, options={}, **context):
    # check for special pop signal code
    # DEPRECATED AND TAKEN CARE OF IN xform_helper ???
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
    if type(recording_uri) is list:
        recording_uri_list=recording_uri
    else:
        recording_uri_list=[recording_uri]
        
    # update the cellid in context so that we don't have to parse the cellid
    # again in xforms
    t_ops = options.copy()
    t_ops['cellid'] = cellid
    t_ops['batch'] = batch
    cells_to_extract, _ = io.parse_cellid(t_ops)
    context = {'recording_uri_list': recording_uri_list, 'cellid': cells_to_extract}

    if pc_idx is not None:
        context['pc_idx'] = pc_idx

    return context


##
## STUFF BELOW HERE CAN BE DELTED AND/OR MOVED?
##

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
    s.name = 'pred0'
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
