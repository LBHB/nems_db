
import logging
from os.path import basename, join

import matplotlib.pyplot as plt
import numpy as np

from nems0 import db
from nems0.utils import smooth
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.baphy_io import load_continuous_openephys, get_spike_info, get_depth_info
from nems_lbhb.preprocessing import impute_multi
from nems.layers import WeightChannels, FIR, LevelShift, \
    DoubleExponential, RectifiedLinear, ConcatSignals, WeightChannelsGaussian
from nems import Model
from nems.tools import json
from nems.layers.base import Layer, Phi, Parameter
from nems0.recording import load_recording
import nems.visualization.model as nplt
from nems0.modules.nonlinearity import _dlog
from nems_lbhb.projects.freemoving.free_tools import stim_filt_hrtf, compute_d_theta, \
    free_scatter_sum, dlc2dist
from nems0.epoch import epoch_names_matching
from nems0.metrics.api import r_floor
from nems0 import xforms
from nems.preprocessing import (indices_by_fraction, split_at_indices, JackknifeIterator)
from nems0.registry import xform, scan_for_kw_defs
from nems_lbhb.plugins.lbhb_loaders import _load_dict

from nems.registry import layer, keyword_lib
from nems0.registry import xform, scan_for_kw_defs

log = logging.getLogger(__name__)

@layer('wcdl')
def wcdl(keyword):
    k = keyword.replace('wcdl','wc')
    wc = keyword_lib[k]
    options = keyword.split('.')
    if 'i' in options:
        wc.input = 'dlc'
    else:
        wc.input = 'hrtf'
    wc.output = 'hrtf'
    return wc

@layer('firdl')
def firdl(keyword):
    k = keyword.replace('firdl','fir')
    fir = keyword_lib[k]
    fir.input = 'hrtf'
    fir.output = 'hrtf'
    return fir

@layer('wcst')
def wcst(keyword):
    k = keyword.replace('wcst','wc')
    wc = keyword_lib[k]
    options = keyword.split('.')
    if 'i' in options:
        wc.input = 'input'
    else:
        wc.input = 'stim'
    wc.output = 'stim'
    return wc

@layer('first')
def first(keyword):
    k = keyword.replace('first','fir')
    fir = keyword_lib[k]
    fir.input = 'stim'
    fir.output = 'stim'
    return fir

@layer('wch')
def wch(keyword):
    k = keyword.replace('wch','wc')
    wc = keyword_lib[k]
    wc.input = 'hstim'
    return wc

@layer('relud')
def relud(keyword):
    k = keyword.replace('relud','relu')
    relu = keyword_lib[k]
    relu.input = 'hrtf'
    relu.output = 'hrtf'
    return relu

@layer('sigd')
def sigd(keyword):
    k = keyword.replace('sigd','sig')
    sig = keyword_lib[k]
    sig.input = 'hrtf'
    sig.output = 'hrtf'
    return sig

@xform()
def free(loadkey, cellid=None, batch=None, siteid=None, **options):
    d = _load_dict(loadkey, cellid, batch)
    d['siteid']=cellid
    del d['cellid']
    xfspec = [['nems_lbhb.projects.freemoving.free_model.load_free_data', d]]
    return xfspec

@xform()
def fev(keyword):
    ops = keyword.split('.')[1:]
    d={}
    if 'hrtf' in ops:
        d['apply_hrtf']=True

    xfspec = [['nems_lbhb.projects.freemoving.free_model.free_split_rec', d]]
    return xfspec

def load_free_data(siteid, cellid=None, batch=None, rasterfs=50, runclassid=132,
                   recache=False, dlc_chans=10, dlc_threshold=0.2, compute_position=False,
                   meta=None, **context):

    sitenum = int(siteid[3:6])
    if batch==347:
        mono = False
        loadkey = f'gtgram.fs{rasterfs}.ch18.dlc.bin'
    else:
        mono = False
        loadkey = f'gtgram.fs{rasterfs}.ch18.dlc'

    log.info(f"{siteid}/{batch}/{loadkey}")
    recording_uri = generate_recording_uri(batch=batch, loadkey=loadkey, cellid=siteid)
    rec = load_recording(recording_uri)

    try:
        df_siteinfo = get_depth_info(siteid=siteid)
        a1cellids = df_siteinfo.loc[(df_siteinfo['area']=='A1') | (df_siteinfo['area']=='BS') |
                                    (df_siteinfo['area']=='PEG')].index.to_list()
    except:
        df_siteinfo = db.pd_query(f"SELECT DISTINCT cellid,area FROM sCellFile WHERE cellid like '{siteid}%%'" +
                         " AND area in ('A1','PEG','AC','BS')")
        a1cellids = df_siteinfo['cellid'].to_list()

    #if cellid is not None:
    #    a1cellids=[cellid]
    """
    sql = f"SELECT distinct stimpath,stimfile from sCellFile where cellid like '{siteid}%%' and runclassid={runclassid}"
    dparm = db.pd_query(sql)
    parmfile = [r.stimpath + r.stimfile for i, r in dparm.iterrows()]

    ## load the recording
    ex = BAPHYExperiment(parmfile=parmfile)
    # print(ex.experiment, ex.openephys_folder, ex.openephys_tarfile, ex.openephys_tarfile_relpath)

    if mono:
        extops = {'mono': True}
    else:
        extops = {}
    rec = ex.get_recording(resp=True, stim=True, stimfmt='gtgram', channels=18,
                           dlc=True, recache=recache, rasterfs=rasterfs,
                           dlc_threshold=dlc_threshold, fill_invalid='interpolate', **extops)
    """

    log.info('Imputing missing DLC values')
    rec = impute_multi(rec, sig='dlc', empty_values=np.nan, keep_dims=dlc_chans)['rec']
    #rec = dlc2dist(rec, smooth_win=2, keep_dims=dlc_chans)
    dlc_data = rec['dlc'][:, :]
    dlc_valid = np.sum(np.isfinite(dlc_data), axis=0, keepdims=True) > 0

    rec['dlc_valid'] = rec['dlc']._modified_copy(data=dlc_valid, chans=['dlc_valid'])
    log.info(f"DLC valid bins: {rec['dlc_valid'].as_continuous().sum()}/{rec['dlc_valid'].shape[1]}")

    rec['stim'] = rec['stim'].rasterize()
    if mono:
        rec['stim'] = rec['stim']._modified_copy(data=rec['stim']._data[:18, :])

    if compute_position:
        rec2 = stim_filt_hrtf(rec, hrtf_format='az', smooth_win=2,
                              f_min=200, f_max=20000, channels=18)['rec']
        # get angle to each speaker and scale -1 to 1
        theta = rec2['disttheta'].as_continuous()[[1, 3], :] / 180
        chans = rec['dlc'].chans + ['th1','th2']
        rec['disttheta']=rec2['disttheta']
        d = np.concatenate((rec['dlc']._data, theta), axis=0)
        rec['dlc']=rec['dlc']._modified_copy(data=d, chans=chans)


    rec['resp'] = rec['resp'].rasterize()

    cid = [i for i, c in enumerate(rec['resp'].chans) if c in a1cellids]
    cellids = [c for i, c in enumerate(rec['resp'].chans) if c in a1cellids]

    rec['resp'] = rec['resp'].extract_channels(cellids)
    rec.meta['siteid'] = siteid
    rec.meta['cellids'] = cellids

    try:
        rec.meta['depth'] = np.array([df_siteinfo.loc[c, 'depth'] for c in cellids])
        #rec.meta['sw'] = np.array([df_siteinfo.loc[c, 'sw'] for c in cellids])
        rec.meta['sw'] = np.ones(len(cellids)) * 1
    except:
        rec.meta['depth'] = np.array([float(c.split("-")[-2]) for c in cellids])
        rec.meta['sw'] = np.ones(len(cellids)) * 100
    if meta is None:
        meta={}
    meta['cellids']=cellids
    meta['siteid']=siteid
    return {'rec': rec, 'meta': meta}

def free_split_rec(rec, apply_hrtf=True, **context):

    if apply_hrtf:
        log.info('Applying HRTF')
        rec = stim_filt_hrtf(rec, hrtf_format='az', smooth_win=2,
                             f_min=200, f_max=20000, channels=18)['rec']
    elif rec['stim'].shape[0]==18:
        log.info('Stacking on noise to control for HRTF')
        stim2 = rec['stim'].shuffle_time(rand_seed=500)
        rec['stim'] = rec['stim'].concatenate_channels([rec['stim'], stim2])

    # log compress and normalize stim
    #fn = lambda x: _dlog(x, -1)
    #rec['stim'] = rec['stim'].transform(fn, 'stim')
    #rec['stim'] = rec['stim'].normalize('minmax')
    #rec['resp'] = rec['resp'].normalize('minmax')

    OLD_MASK = False
    if OLD_MASK:
        # epoch_regex = "^STIM_"
        # est, val = rec.split_using_epoch_occurrence_counts(epoch_regex=epoch_regex)
        # est = preproc.average_away_epoch_occurrences(est, epoch_regex=epoch_regex)
        # val = preproc.average_away_epoch_occurrences(val, epoch_regex=epoch_regex)
        est = rec.jackknife_mask_by_epoch(5, 0, 'REFERENCE', invert=False)
        val = rec.jackknife_mask_by_epoch(5, 0, 'REFERENCE', invert=True)

        est = est.and_mask(est['dlc_valid'].as_continuous()[0,:])
        val = val.and_mask(val['dlc_valid'].as_continuous()[0,:])
    else:
        val_epochs = epoch_names_matching(rec['resp'].epochs, "^STIM_00")
        val = rec.copy()
        val = val.create_mask(val_epochs).and_mask(val['dlc_valid'].as_continuous()[0, :])
        mask = val['mask'].as_continuous()[0, :]
        est = rec.copy()
        est = est.create_mask(~mask).and_mask(est['dlc_valid'].as_continuous()[0, :])

    return {'rec': rec, 'est': est, 'val': val}


def free_fit(rec, shuffle="none", apply_hrtf=True, dlc_memory=4,
             acount=20, dcount=10, l2count=30, cost_function='squared_error',
             save_to_db=False, jack_count=None, **options):
    """
    special case: if dlc_memory=1 add a delay line with shuffled data to
    the dlc signal to match the parameter count for dlc_memory=4
    """

    siteid = rec.meta['siteid']
    cellids = rec['resp'].chans
    log.info("Splitting rec into est, val")
    ctx = free_split_rec(rec, apply_hrtf=apply_hrtf)

    rec = ctx['rec']
    est = ctx['est'].apply_mask()
    val = ctx['val'].apply_mask()

    log.info(f"est resp: {est['resp'].shape} stim: {est['stim'].shape} dlc: {est['dlc'].shape}")
    log.info(f"val resp: {val['resp'].shape} stim: {val['stim'].shape} dlc: {val['dlc'].shape}")
    dlc_count = rec['dlc'].shape[0]
    
    if jack_count is not None:
        # undo est/val breakdown so that full dataset can be jackknifed

        est = rec.create_mask(rec['dlc_valid'].as_continuous()[0, :]).apply_mask()
        val = rec.create_mask(rec['dlc_valid'].as_continuous()[0, :]).apply_mask()

    if shuffle=='none':
        input = {'stim': est['stim'].as_continuous().T, 'dlc': est['dlc'].as_continuous().T[:, :dlc_count]}
        test_input = {'stim': val['stim'].as_continuous().T, 'dlc': val['dlc'].as_continuous().T[:, :dlc_count]}
    elif shuffle=='dlc':
        input = {'stim': est['stim'].as_continuous().T,
                 'dlc': est['dlc'].shuffle_time(rand_seed=1000).as_continuous().T[:, :dlc_count]}
        test_input = {'stim': val['stim'].as_continuous().T,
                      'dlc': val['dlc'].shuffle_time(rand_seed=1000).as_continuous().T[:, :dlc_count]}
    elif shuffle=='stim':
        input = {'stim': est['stim'].shuffle_time(rand_seed=1000).as_continuous().T,
                 'dlc': est['dlc'].as_continuous().T[:, :dlc_count]}
        test_input = {'stim': val['stim'].shuffle_time(rand_seed=1000).as_continuous().T,
                      'dlc': val['dlc'].as_continuous().T[:, :dlc_count]}

    if dlc_memory == 1:
        d = [input['dlc']] + [est['dlc'].shuffle_time(rand_seed=100).as_continuous().T]
        input['dlc'] = np.concatenate(d, axis=1)
        d = [test_input['dlc']] + [val['dlc'].shuffle_time(rand_seed=100).as_continuous().T]
        test_input['dlc'] = np.concatenate(d, axis=1)
        log.info('delay line with shuffled dlc')

    target = est['resp'].as_continuous().T
    test_target = val['resp'].as_continuous().T

    # number of auditory filters, dlc filters, L2 filters
    #acount, dcount, l2count = 16, 8, 30
    #acount, dcount, l2count = 12, 4, 24

    tcount = acount+dcount
    cellcount = len(cellids)
    input_count = est['stim'].shape[0]
    dlc_count = input['dlc'].shape[1]

    modelstring = f"wc.Nx1x{acount}-fir.8x1x{acount}-wc.Dx1x{dcount}.dlc-fir.{dlc_memory}x1x{dcount}.dlc-concat.space-relu.{tcount}.f" +\
        f"-wc.{tcount}x1x{l2count}-fir.4x1x{l2count}-relu.{l2count}.f-wc.{l2count}xR-relu.R"
    if dcount > 0:
        layers = [
            WeightChannels(shape=(input_count, 1, acount), input='stim', output='prediction'),
            WeightChannels(shape=(dlc_count, 1, dcount), input='dlc', output='space'),
            FIR(shape=(8, 1, acount), input='prediction', output='prediction'),
            FIR(shape=(dlc_memory, 1, dcount), input='space', output='space'),
            ConcatSignals(input=['prediction','space'], output='prediction'),
            RectifiedLinear(shape=(tcount,), input='prediction', output='prediction',
                            no_offset=True, no_shift=True),
            WeightChannels(shape=(tcount, 1, l2count), input='prediction', output='prediction'),
            FIR(shape=(4, 1, l2count), input='prediction', output='prediction'),
            RectifiedLinear(shape=(1, l2count), input='prediction', output='prediction',
                            no_offset=True, no_shift=True),
            WeightChannels(shape=(l2count, cellcount), input='prediction', output='prediction'),
            #DoubleExponential(shape=(1, cellcount), input='prediction', output='prediction'),
            RectifiedLinear(shape=(1, cellcount), input='prediction', output='prediction',
                            no_offset=False, no_shift=False),
        ]
    else:
        layers = [
            WeightChannels(shape=(input_count, 1, acount), input='stim', output='prediction'),
            FIR(shape=(15, 1, acount), input='prediction', output='prediction'),
            WeightChannels(shape=(tcount, cellcount), input='prediction', output='prediction'),
            LevelShift(shape=(1, cellcount), input='prediction', output='prediction'),
        ]
    fitter = 'tf'
    fitter_options = {'cost_function': cost_function,  # 'nmse'
                      'early_stopping_tolerance': 1e-3,
                      'validation_split': 0,
                      'learning_rate': 1e-2, 'epochs': 3000
                      }
    fitter_options2 = {'cost_function': cost_function,
                      'early_stopping_tolerance': 1e-4,
                      'validation_split': 0,
                      'learning_rate': 1e-3, 'epochs': 8000
                      }

    model = Model(layers=layers)
    model.name = f'sh.{shuffle}-hrtf.{apply_hrtf}_{modelstring}_{cost_function.replace("_","")}'
    model = model.sample_from_priors()
    model = model.sample_from_priors()
    model = model.sample_from_priors()
    log.info(f'Site: {siteid}')
    log.info(f'Model: {model.name}')


    if jack_count is not None:
        fit_set = JackknifeIterator(input, target=target, samples=jack_count, axis=0)
        # to do ... get this to work 
        log.info('Fit stage 1: without static output nonlinearity')
        model.layers[-1].skip_nonlinearity()
        model_fit_list = fit_set.get_fitted_jackknifes(model, backend=fitter,
                          fitter_options=fitter_options)

        for model in model_fit_list:
            model.layers[-1].unskip_nonlinearity()

        log.info('Fit stage 2: with static output nonlinearity')
        model_fit_list = fit_set.get_fitted_jackknifes(model_fit_list, backend=fitter,
                          fitter_options=fitter_options)

        # predict responses for each jk validation set and recombine into
        # a single prediction that matches the size of the orginal target
        dpred = fit_set.get_predicted_jackknifes(model_fit_list)
        prediction = dpred['prediction']
        model = model_fit_list[0]
        fit_pred = model.predict(input=input)['prediction']

    else:
        log.info('Fit stage 1: without static output nonlinearity')
        model.layers[-1].skip_nonlinearity()
        model = model.fit(input=input, target=target, backend=fitter,
                          fitter_options=fitter_options)
        model.layers[-1].unskip_nonlinearity()
        log.info('Fit stage 2: with static output nonlinearity')
        model = model.fit(input=input, target=target, backend=fitter,
                          verbose=0, fitter_options=fitter_options2)

        fit_pred = model.predict(input=input)
        prediction = model.predict(input=test_input)
        if type(prediction) is dict:
            fit_pred = fit_pred['prediction']
            prediction = prediction['prediction']

    est['pred']=est['resp']._modified_copy(data=fit_pred.T)
    val['pred']=val['resp']._modified_copy(data=prediction.T)

    fit_cc = np.array([np.corrcoef(fit_pred[:, i], target[:, i])[0, 1] for i in range(cellcount)])
    cc = np.array([np.corrcoef(prediction[:, i], test_target[:, i])[0, 1] for i in range(cellcount)])
    rf = r_floor(X1mat=prediction.T, X2mat=test_target.T)

    model.meta['fit_predxc'] = fit_cc
    model.meta['predxc'] = cc
    model.meta['prediction'] = prediction
    model.meta['resp'] = test_target
    model.meta['siteid'] = siteid
    model.meta['batch'] = rec.meta['batch']
    model.meta['modelname'] = model.name
    model.meta['cellids'] = est['resp'].chans
    model.meta['cellid'] = siteid
    model.meta['r_test'] = cc[:, np.newaxis]
    model.meta['r_fit'] = fit_cc[:, np.newaxis]
    model.meta['r_floor'] = rf[:, np.newaxis]

    if save_to_db:
        log.info('Saving to disk and db')
        destination = xforms.save_lite(model)
        model.meta['modelfile'] = join(destination,'modelspec.json')
        model.meta['modelpath'] = destination
        db.save_results(model)

    ctx = {'rec': rec, 'est': est, 'val': val, 'modelspec': model}
    if jack_count is not None:
        for i,m in enumerate(model_list):
            m.meta.update(model.meta)
            m.meta['jack_index']=i
        ctx['model_list'] = model_fit_list
    return ctx



def compare_models(rec, model1, model2):
    depth = rec.meta['depth']
    sw = rec.meta['sw']
    cc1 = model1.meta['r_test']
    cc2 = model2.meta['r_test']
    target = model1.meta['resp']
    prediction1 = model1.meta['prediction']
    prediction2 = model2.meta['prediction']

    imopts = {'aspect': 'auto', 'origin': 'lower', 'interpolation': 'none'}

    f, ax = plt.subplots(4, 3, figsize=(12, 8))

    ax[0, 0].imshow(smooth(prediction1[:2000, :].T), **imopts)
    ax[0, 0].set_title(rec.meta['siteid'])

    ax[1, 0].imshow(smooth(target[:2000, :].T), **imopts)

    nplt.plot_strf(model1.layers[1], model1.layers[0], ax=ax[0, 1])
    ax[0, 1].set_title('model1 stim FIR')
    nplt.plot_strf(model1.layers[3], model1.layers[2], ax=ax[1, 1])
    ax[1, 1].set_title('model1 dlc FIR')
    nplt.plot_strf(model2.layers[1], model2.layers[0], ax=ax[0, 2])
    ax[0, 2].set_title('model2 stim FIR')
    nplt.plot_strf(model2.layers[3], model2.layers[2], ax=ax[1, 2])
    ax[1, 2].set_title('model2 dlc FIR')

    ax[2, 0].plot(depth[sw > 0.4], cc1[sw > 0.4], '.', label='dlc-RS')
    ax[2, 0].plot(depth[sw > 0.4], cc2[sw > 0.4], '.', label='dlcsh-RS')
    ax[2, 0].plot(depth[sw <= 0.4], cc1[sw <= 0.4], '.', label='dlc-NS')
    ax[2, 0].plot(depth[sw <= 0.4], cc2[sw <= 0.4], '.', label='dlcsh-NS')
    ax[2, 0].set_title(f"pred cc1={np.mean(cc1):.3} cc2={np.mean(cc2):.3}")
    ax[2, 0].set_xlabel('depth from L3-L4 border (um)')
    ax[2, 0].legend(fontsize=8)

    ax[3, 0].plot(depth[sw > 0.4], cc1[sw > 0.4] - cc2[sw > 0.4], '-', label='RS')
    ax[3, 0].plot(depth[sw <= 0.4], cc1[sw <= 0.4] - cc2[sw <= 0.4], '-', label='NS')
    ax[3, 0].set_xlabel('depth from L3-L4 border (um)')
    ax[3, 0].set_ylabel('dlc diff')
    ax[3, 0].legend(fontsize=8)

    ax[2, 1].plot(model1.layers[-2].coefficients.T);
    ax[2, 2].plot(model2.layers[-2].coefficients.T);
    plt.tight_layout()

    return f




def free_fit_two(rec, **options):
    # TODO : DELETE ME?  Redundant with free_fit and compare_models
    siteid = rec.meta['siteid']

    cellids = rec['resp'].chans
    epoch_regex = "^STIM_"

    #est, val = rec.split_using_epoch_occurrence_counts(epoch_regex=epoch_regex)
    #est = preproc.average_away_epoch_occurrences(est, epoch_regex=epoch_regex)
    #val = preproc.average_away_epoch_occurrences(val, epoch_regex=epoch_regex)
    est = rec.jackknife_mask_by_epoch(5, 0, 'REFERENCE', invert=False)
    val = rec.jackknife_mask_by_epoch(5, 0, 'REFERENCE', invert=True)

    est = est.and_mask(est['dlc_valid'].as_continuous()[0,:])
    val = val.and_mask(val['dlc_valid'].as_continuous()[0,:])

    est = est.apply_mask()
    val = val.apply_mask()
    print(est['resp'].shape, est['stim'].shape, est['dlc'].shape)

    acount=8
    dcount=4
    tcount = acount+dcount
    l2count = 8
    cellcount = len(cellids)
    input_count = rec['stim'].shape[0]
    dlc_count = rec['dlc'].shape[0]

    if dcount > 0:
        layers = [
            WeightChannels(shape=(input_count, 1, acount), input='stim', output='prediction'),
            FIR(shape=(8, 1, acount), input='prediction', output='prediction'),
            WeightChannels(shape=(dlc_count, 1, dcount), input='dlc', output='space'),
            FIR(shape=(4, 1, dcount), input='space', output='space'),
            ConcatSignals(input=['prediction','space'], output='prediction'),
            RectifiedLinear(shape=(1, tcount), input='prediction', output='prediction',
                            no_offset=False, no_shift=False),
            WeightChannels(shape=(tcount, l2count), input='prediction', output='prediction'),
            RectifiedLinear(shape=(1, l2count), input='prediction', output='prediction',
                            no_offset=False, no_shift=False),
            WeightChannels(shape=(l2count, cellcount), input='prediction', output='prediction'),
            DoubleExponential(shape=(1, cellcount), input='prediction', output='prediction'),
        ]
    else:
        layers = [
            WeightChannels(shape=(input_count, 1, acount), input='stim', output='prediction'),
            FIR(shape=(15, 1, acount), input='prediction', output='prediction'),
            WeightChannels(shape=(tcount, cellcount), input='prediction', output='prediction'),
            LevelShift(shape=(1, cellcount), input='prediction', output='prediction'),
        ]

    fitter = 'tf'
    if fitter == 'scipy':
        fitter_options = {'cost_function': 'nmse', 'options': {'ftol':  1e-4, 'gtol': 1e-4, 'maxiter': 100}}
    else:
        fitter_options = {'cost_function': 'nmse',
                          'early_stopping_delay': 5,
                          'early_stopping_patience': 10,
                          'early_stopping_tolerance': 1e-3,
                          'validation_split': 0,
                          'learning_rate': 1e-2, 'epochs': 2000
                          }
        fitter_options2 = {'cost_function': 'nmse',
                          'early_stopping_delay': 5,
                          'early_stopping_patience': 10,
                          'early_stopping_tolerance': 1e-5,
                          'validation_split': 0,
                          'learning_rate': 5e-3, 'epochs': 2000
                          }

    model = Model(layers=layers)
    model = model.sample_from_priors()
    model = model.sample_from_priors()
    model2 = model.copy()

    input1 = {'stim': est['stim'].as_continuous().T, 'dlc': est['dlc'].as_continuous().T[:, :dlc_count]}
    input2 = {'stim': est['stim'].as_continuous().T, 'dlc': est['dlcsh'].as_continuous().T[:, :dlc_count]}
    target = est['resp'].as_continuous().T

    model.layers[-1].skip_nonlinearity()
    model = model.fit(input=input1, target=target, backend=fitter, fitter_options=fitter_options)
    model.layers[-1].unskip_nonlinearity()
    model = model.fit(input=input1, target=target, backend=fitter, fitter_options=fitter_options2)

    model2.layers[-1].skip_nonlinearity()
    model2 = model2.fit(input=input2, target=target, backend=fitter, fitter_options=fitter_options)
    model2.layers[-1].unskip_nonlinearity()
    model2 = model2.fit(input=input2, target=target, backend=fitter, fitter_options=fitter_options2)

    test_input1 = {'stim': val['stim'].as_continuous().T, 'dlc': val['dlc'].as_continuous().T[:, :dlc_count]}
    test_input2 = {'stim': val['stim'].as_continuous().T, 'dlc': val['dlcsh'].as_continuous().T[:, :dlc_count]}
    test_target = val['resp'].as_continuous().T

    fit_pred1 = model.predict(input=input1)
    fit_pred2 = model2.predict(input=input2)
    prediction1 = model.predict(input=test_input1)
    prediction2 = model2.predict(input=test_input2)
    if type(prediction1) is dict:
        fit_pred1 = fit_pred1['prediction']
        fit_pred2 = fit_pred2['prediction']
        prediction1 = prediction1['prediction']
        prediction2 = prediction2['prediction']

    fit_cc1 = np.array([np.corrcoef(fit_pred1[:, i], target[:, i])[0, 1] for i in range(cellcount)])
    fit_cc2 = np.array([np.corrcoef(fit_pred2[:, i], target[:, i])[0, 1] for i in range(cellcount)])
    cc1 = np.array([np.corrcoef(prediction1[:, i], test_target[:, i])[0, 1] for i in range(cellcount)])
    cc2 = np.array([np.corrcoef(prediction2[:, i], test_target[:, i])[0, 1] for i in range(cellcount)])

    model.meta['fit_predxc'] = fit_cc1
    model2.meta['fit_predxc'] = fit_cc2
    model.meta['predxc'] = cc1
    model2.meta['predxc'] = cc2
    model.meta['prediction'] = prediction1
    model2.meta['prediction'] = prediction2
    model.meta['resp'] = test_target
    model2.meta['resp'] = test_target

    try:
        depth = rec.meta['depth']
        sw = rec.meta['sw']
    except:
        depth = rec.meta['depth']
        sw = rec.meta['sw']

    imopts = {'aspect': 'auto', 'origin': 'lower', 'interpolation': 'none'}

    f, ax = plt.subplots(4, 3, figsize=(12, 8))

    ax[0, 0].imshow(smooth(prediction1[:2000, :].T), **imopts)
    ax[0, 0].set_title(siteid)

    ax[1, 0].imshow(smooth(target[:2000, :].T), **imopts)

    nplt.plot_strf(model.layers[1], model.layers[0], ax=ax[0, 1])
    nplt.plot_strf(model.layers[3], model.layers[2], ax=ax[1, 1])
    nplt.plot_strf(model2.layers[1], model2.layers[0], ax=ax[0, 2])
    nplt.plot_strf(model2.layers[3], model2.layers[2], ax=ax[1, 2])

    ax[2, 0].plot(depth[sw > 0.4], cc1[sw > 0.4], '.', label='dlc-RS')
    ax[2, 0].plot(depth[sw > 0.4], cc2[sw > 0.4], '.', label='dlcsh-RS')
    ax[2, 0].plot(depth[sw <= 0.4], cc1[sw <= 0.4], '.', label='dlc-NS')
    ax[2, 0].plot(depth[sw <= 0.4], cc2[sw <= 0.4], '.', label='dlcsh-NS')
    ax[2, 0].set_title(f"pred cc1={np.mean(cc1):.3} cc2={np.mean(cc2):.3}")
    ax[2, 0].set_xlabel('depth from L3-L4 border (um)')
    ax[2, 0].legend()

    ax[3, 0].plot(depth[sw > 0.4], cc1[sw > 0.4] - cc2[sw > 0.4], '.', label='RS')
    ax[3, 0].plot(depth[sw <= 0.4], cc1[sw <= 0.4] - cc2[sw <= 0.4], '.', label='NS')
    ax[3, 0].set_xlabel('depth from L3-L4 border (um)')
    ax[3, 0].set_ylabel('dlc diff')
    ax[3, 0].legend()

    ax[2, 1].plot(model.layers[-2].coefficients.T);
    ax[2, 2].plot(model.layers[-2].coefficients.T);
    plt.tight_layout()

    return model, model2, f

