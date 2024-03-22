"""xforms library

This module contains standard transformations ("xforms") applied sequentially during a NEMS
fitting process. Custom xforms can be developed as long as they adhere to the required syntax.

"""
import copy
import glob
import io
import logging
import os
import socket

import matplotlib.pyplot as plt
import numpy as np

import nems0
import nems0.epoch as ep
import nems0.db as nd
import nems0.metrics.api as metrics
import nems0.plots.api as nplt
import nems0.preprocessing as preproc
import nems0.priors as priors
from nems0 import analysis
from nems0 import initializers as init
from nems0 import modelspec as ms
from nems0.fitters.api import scipy_minimize
from nems0.modelspec import set_modelspec_metadata, get_modelspec_metadata, \
    get_modelspec_shortname
from nems0.recording import load_recording, Recording
from nems0.registry import xform, scan_for_kw_defs
from nems0.signal import RasterizedSignal
from nems0.uri import save_resource, load_resource
from nems0.utils import (get_setting, iso8601_datestring, find_module,
                         recording_filename_hash, get_default_savepath, lookup_fn_at,
                         progress_fun)

log = logging.getLogger(__name__)

# REGISTRY SETUP
# scan in plugins dir by default
scan_for_kw_defs(os.path.join(get_setting('NEMS_DIR'),'nems0','plugins'))

# populate the registry as specified in config settings
scan_for_kw_defs(get_setting('LIB_PLUGINS'))
scan_for_kw_defs(get_setting('KEYWORD_PLUGINS'))
scan_for_kw_defs(get_setting('XFORMS_PLUGINS'))

# DEPRECATED - migrates to import and decorators
#xforms_lib.register_modules([default_loaders, default_fitters, default_initializers])
#xforms_lib.register_plugins(get_setting('XFORMS_PLUGINS'))

# DEPRECATED?
#xforms = {}  # A mapping of kform keywords to xform 2-tuplets (2 element lists)

def defxf(keyword, xformspec):
    """
    Adds xformspec to the xforms keyword dictionary.
    A helper function so not every keyword mapping has to be in a single
    file and part of a very large single multiline dict.
    """
    raise DeprecationWarning('Deprecated?')
    if keyword in xforms:
        raise ValueError("Keyword already defined! Choose another name.")
    xforms[keyword] = xformspec



def evaluate_step(xfa, context={}, verbose=True):
    """
    Take one step in evaluation of xforms sequence.
    :param xfa: list of 2 or 4 elements specifying function to be evaluated on this step
                and relevant args for that function.
                xfa[0] : string of python path to function evaluated on this step. e.g.,
                         `nems0.xforms.load_recording_wrapper`
                xfa[1] : dictionary of args to pass to xfa[0]
                xfa[2] : optional (DEPRECATED?), indicates context-in keys (if xfa[0] returns a tuple rather than context dict)
                xfa[3] : optional (DEPRECATED?), context-out keys
    :param context: xforms context prior to evaluating this step, combined with xfa[1] to
                    provide input to xfa[0]
    :return: context_out dictionary updated with output of xfa[0]
    """
    if not(len(xfa) == 2 or len(xfa) == 4):
        raise ValueError('Got non 2- or 4-tuple for xform: {}'.format(xfa))
    xf = xfa[0]
    xfargs = xfa[1]
    if len(xfa) > 2:
        # backward compatibility
        a = [k.replace("modelspecs", "modelspec") for k in xfa[2]]
        context_in = {k: context[k] for k in a}
    else:
        context_in = context
    if len(xfa) > 3:
        a = [k.replace("modelspecs", "modelspec") for k in xfa[3]]
        context_out_keys = a
    else:
        context_out_keys = []

    # Load relevant function into lib path
    fn = lookup_fn_at(xf)

    # Run the xf
    if verbose:
        log.info('Evaluating: {}'.format(xf))

    # Check for collisions; more to avoid confusion than for correctness:
    # (except for init_context, which can update)
    #if not 'init_context' in xf:
    #    for k in xfargs:
    #        if k in context_in:
    #            m = 'xf arg {} overlaps with context: {}'.format(k, xf)
    #            raise ValueError(m)
    for k in xfargs:
        if (k in context_in) and verbose:
            log.info('xf argument %s overlaps with existing context key: %s', k, xf)

    # Merge args into context, and make a deepcopy so that mutation
    # inside xforms will not be propagated unless the arg is returned.
    args = copy.deepcopy(context_in)
    args['verbose'] = verbose
    #args = {}
    #for k,v in context_in.items():
    #    log.info(f"deep copying {k}")
    #    args[k] = copy.deepcopy(v)

    args.update(**xfargs)
    #merged_args = {**xfargs, **context_in}
    #args = copy.deepcopy(merged_args)

    new_context = fn(**args)
    if len(context_out_keys):
        if type(new_context) is tuple:
            # print(new_context)
            new_context = {k: new_context[i] for i, k
                           in enumerate(context_out_keys)}
        elif len(context_out_keys) == 1:
            new_context = {context_out_keys[0]: new_context}
        else:
            raise ValueError('len(context_out_keys) needs to match '
                             'number of outputs from xf fun')
    # Use the new context for the next step
    if type(new_context) is not dict:
        raise ValueError('xf did not return a context dict: {}'.format(xf))
    context_out = {**context, **new_context}

    return context_out


def evaluate(xformspec, context={}, start=0, stop=None, skip_postprocess=True, verbose=True):
    '''
    Similar to modelspec.evaluate, but for xformspecs, which is a list of
    2-element lists of function and keyword arguments dict. Each XFORM must
    return a dictionary of the explicit changes made to the context, which
    is the dict that is passed from xform to xform.

    Also, this function wraps every logging call and saves it in a log
    that is the second value returned by this function.
    '''
    context = copy.deepcopy(context)  # Create a new starting context

    # Create a log stream set to the debug level; add it as a root log handler
    log_stream = io.StringIO()
    ch = logging.StreamHandler(log_stream)
    ch.setLevel(logging.DEBUG)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    ch.setFormatter(formatter)
    rootlogger = logging.getLogger()
    rootlogger.addHandler(ch)

    # Evaluate the xforms
    for xfa in xformspec[start:stop]:
        if not(skip_postprocess) or not('postprocess' in xfa[0]):
            context = evaluate_step(xfa, context, verbose=verbose)

    # Close the log, remove the handler, and add the 'log' string to context
    if verbose:
        log.info('Done (re-)evaluating xforms.')
    ch.close()
    rootlogger.removeFilter(ch)
    logstring = log_stream.getvalue()
    context['log'] = logstring

    return context, logstring


###############################################################################
# Stuff below this line are useful resuable components.
# See xforms_test.py for how to use it.


###############################################################################
##################       LOADERS / MODELSPEC STUFF    #########################
###############################################################################


def init_context(kw_kwargs=None, **context):
    #if kw_kwargs is None:
    #    keyword_lib = KeywordRegistry()
    #else:
    #    keyword_lib = KeywordRegistry(**kw_kwargs)

    #keyword_lib.register_module(default_keywords)
    #keyword_lib.register_plugins(get_setting('KEYWORD_PLUGINS'))
    #context['registry'] = keyword_lib

    return context


def load_recording_wrapper(load_command=None, exptid="RECORDING", cellid=None,
                           save_cache=True, IsReload=False, modelspecs=None,
                           modelspec=None, **context):
    """
    generic wrapper for loading recordings
    :param load_command: string pointing to relevant load command, eg "my.lib.load_fun"
        load_command should be able to take **context as an input and return
        a dictionary of relevant signals. d : {'stim': X, 'resp': Y, 'state': S} etc.
        X and Y can be in three different forms:
           1. 2D, with M x T and N x T matrices. Ie, the number of channels can differ
              between them but they should have the same times.
           TODO: Support for other stim/resp matrix formats:
           2. 3D, M x T and N x R x T. R corresponds to repetitions of the same X, X will be
              tiled R times to match the length of Y
           3. 4D M x S x T and N x R x S x T or N x S x T. S stimuli were repeated R times
              (or once) X will be unwrapped to M*S x T and tiled if R>1
        additional special dictionary entries:
          d['<signame>_labels'] - list of strings, one to label each row of stim, resp, etc
          d['fs'] - sampling rate for all signals
          d['epochs'] - dataframe with 'start' 'end' 'name' columns for important events.
          d['meta'] - meta data that will be included in rec.meta. regardless, rec.meta
            will be initialized with rec.meta = context
    :param save_cache
        if true the recording will be saved to a NEMS native recording file. context will
        also be used to generate a hash that determines the file name. future calls will
        then generate a hash and load a matching recording from the cache if it exists.
        cached recordings are stored in nems0.get_config(NEMS_RECORDINGS_DIR)
        if batch is not None, it will be saved in "<batch>/" subdirectory
        filename will be "<cellid>_<hash>.tgz"
    :param context['batch'] (optional)
        numerical identifier for grouping the recording with other recordings
    :param cellid
        string identifier for the signals being modeled. eg the name of the cell or
        the experiment (if multiple cells are being modeled at once). will extract
        that chan from rec['resp'].chans after loading (but not saved in cache)
    :param context: dictionary of parameters/metadata that will be passed through to load_command

    :return: rec - NEMS recording object

    TODO: option to re-cache
    """
    data_file = recording_filename_hash(exptid, context, nems0.get_setting('NEMS_RECORDINGS_DIR'))
    if os.path.exists(data_file):
        log.info("Loading cached file %s", data_file)
        rec = load_recording(data_file)
    else:

        fn = lookup_fn_at(load_command)

        data = fn(exptid=exptid, **context)

        signals = {}
        fs = data.get('fs',100)
        meta = data.get('meta', {})
        epochs = data.get('epochs', None)
        for k in data.keys():
            if k in ['fs', 'meta', 'epochs']:
                pass
            elif k.endswith('_labels'):
                # these wil
                pass
            else:
                signals[k] = RasterizedSignal(fs, data[k], k, exptid, epochs=epochs,
                                              chans=data.get(k+'_labels'))

        rec = Recording(signals)
        rec.meta = meta.copy()
        rec.meta.update(context)
        rec.meta["exptid"] = exptid

        if save_cache:
            rec.save(data_file)

    # if cellid specified, select only that channel
    if cellid is not None:
        if cellid in rec['resp'].chans:
            log.info("match found, extracting channel from rec")
            rec['resp'] = rec['resp'].extract_channels([cellid])
            rec.meta["cellid"] = cellid

    return {'rec': rec}


def load_recordings(recording_uri_list=None, normalize=False, cellid=None,
                    save_other_cells_to_state=None, input_name='stim',
                    output_name='resp', meta={}, **context):
    '''
    Load one or more recordings into memory given a list of URIs.
    '''
    try:
        rec = load_recording(recording_uri_list[0])
    except:
        log.info(f'saved recording not found. Generating uri')
        cellid=meta['cellid']
        batch=meta['batch']
        loadkey = meta['loader'].split('-')[0]
        from nems_lbhb import xform_wrappers
        uri = xform_wrappers.generate_recording_uri(cellid=cellid, batch=batch, loadkey=loadkey)
        rec = load_recording(uri)

    other_recordings = [load_recording(uri) for uri in recording_uri_list[1:]]
    if other_recordings:
        rec.concatenate_recordings(other_recordings)

    if normalize and 'stim' in rec.signals.keys():
        log.info('Normalizing stim')
        rec['stim'] = rec['stim'].rasterize().normalize('minmax')
    if context.get('verbose', True):
        log.info('Extracting cellid(s) {}'.format(cellid))
    if (cellid is None) or (cellid in ['none', 'NAT3', 'NAT4', 'ALLCELLS']):
        # No cellid specified, use all channels
        channels = rec[output_name].chans
        if len(channels) == 0:
            # no channels were specified for recording, so use default indices
            cellids = [str(i) for i in range(rec[output_name].shape[0])]
        else:
            # Channel names should already be strings, but just incase
            cellids = [str(c) for c in channels]
        meta['cellids'] = cellids

    elif type(cellid) is str:
        if len(cellid.split('-')) == 1:
            # siteid given instead of cellid or list of cellids
            siteid = cellid
            cellids = [c for c in rec[output_name].chans if siteid in c]
            meta['cellids'] = cellids
            meta['siteid'] = siteid
        else:
            meta['cellids'] = [cellid]

    else:
        # Assume a list of cellids was given
        meta['cellids'] = cellid

    if (save_other_cells_to_state is not None):

        if (type(save_other_cells_to_state) is str) & (save_other_cells_to_state!='state'):
            excluded_cells = rec['resp'].chans.copy()
            pop_var = save_other_cells_to_state
        else:
            excluded_cells = list(set(rec['resp'].chans) - set(meta['cellids']))
            pop_var = 'state'
        if (len(excluded_cells) > 0):
            s = rec['resp'].extract_channels(excluded_cells, name=pop_var).rasterize()
            rec.add_signal(s)

            if pop_var == 'state':
                rec['state_raw'] = rec['state']._modified_copy(rec['state']._data, name='state_raw')
            # else:
            #     raise ValueError('pop_var {} unknown'.format(pop_var))

    rec['resp'] = rec['resp'].extract_channels(meta['cellids'])

    rec = preproc.generate_stim_from_epochs(rec, new_signal_name='epoch_onsets',
                                            epoch_regex='TRIAL', onsets_only=True)

    return {'rec': rec, 'input_name': input_name, 'output_name': output_name,
            'meta': meta}


def none(**context):
    '''
    Does nothing. Useful when xforms expects at least one keywords to exist
    but nothing is needed. For example, if recordings are manually preprocessed
    and added to the context instead of using a loader.

    '''
    return {}


def init_from_keywords(keywordstring, meta={}, IsReload=False,
                       registry=None, rec=None, rec_list=None, input_name='stim',
                       output_name='resp', **context):
    if not IsReload:
        modelspec = init.from_keywords(keyword_string=keywordstring,
                                       meta=meta, registry=registry, rec=rec, rec_list=rec_list,
                                       input_name=input_name,
                                       output_name=output_name)
        modelspec.meta['engine'] = 'nems0'
    else:
        modelspec=context['modelspec']

        # backward compatibility? Maybe can delete?
        if modelspec is not None:
            if modelspec.meta.get('input_name', None) is None:
                modelspec.meta['input_name'] = input_name
                modelspec.meta['output_name'] = output_name

    return {'modelspec': modelspec}


def load_modelspecs(modelspecs, uris, IsReload=False, **context):
    '''
    i.e. Load a modelspec from a specific place. This is not
    the same as reloading a model for later inspection; it would be more
    appropriate when doing something complicated with several different
    models.
    '''
    if not IsReload:
        modelspec = ms.ModelSpec([load_resource(uri) for uri in uris])
    return {'modelspec': modelspec}


def set_random_phi(modelspecs, IsReload=False, **context):
    ''' Starts all modelspecs at random phi sampled from the priors. '''
    if not IsReload:
        for fit_idx in modelspec.fit_count:
            modelspec.fit_index = fit_idx
            for i, m in enumerate(modelspec):
                modelspec[i] = priors.set_random_phi(m)
    return {'modelspec': modelspec}


def fill_in_default_metadata(rec, modelspec, IsReload=False, **context):
    '''
    Sets any uninitialized metadata to defaults that should help us
    find it in nems_db again. (fitter, recording, date, etc)
    '''
    if not IsReload:
        # Add metadata to help you reload this state later
        meta = get_modelspec_metadata(modelspec)
        if 'fitter' not in meta:
            set_modelspec_metadata(modelspec, 'fitter', 'None')
        if 'fit_time' not in meta:
            set_modelspec_metadata(modelspec, 'fitter', 'None')
        if 'recording' not in meta:
            recname = rec.name if rec else 'None'
            set_modelspec_metadata(modelspec, 'recording', recname)
        if 'recording_uri' not in meta:
            uri = rec.uri if rec and rec.uri else 'None'
            set_modelspec_metadata(modelspec, 'recording_uri', uri)
        if 'date' not in meta:
            set_modelspec_metadata(modelspec, 'date', iso8601_datestring())
        if 'hostname' not in meta:
            set_modelspec_metadata(modelspec, 'hostname',
                                   socket.gethostname())
    return {'modelspec': modelspec}


def only_best_modelspec(modelspecs, metakey='r_test', comparison='greatest',
                        IsReload=False, **context):
    '''
    Collapses a list of modelspecs so that it only contains the modelspec
    with the highest given meta metric.
    '''
    if not IsReload:
        # TODO: Not the fastest way to do this but probably doesn't matter
        #       since it's only done once per fit.

        # TODO: Make this allow for variable number of top specs by
        #       updating ms function to sort then pick top n
        return {'modelspecs': ms.get_best_modelspec(modelspecs, metakey,
                                                    comparison)}
    else:
        return {}


def sort_modelspecs(modelspecs, metakey='r_test', order='descending',
                    IsReload=False, **context):
    '''
    Sorts modelspecs according to the specified metakey and order.
    '''
    if not IsReload:
        return {'modelspecs': ms.sort_modelspecs(modelspecs, metakey, order)}
    else:
        return {}


###############################################################################
#######################     NEMS-LITE SPECIFIC     ############################
###############################################################################


def init_nems_keywords(keywordstring, meta=None, IsReload=False,
                       **context):
    """NEMS-LITE specific model initialization"""
    from nems import Model
    if IsReload:
        modelspec = context['modelspec']
    else:
        if meta is None:
            meta = {}
        keywordstring0 = keywordstring
        keywordstring = init.fill_keyword_string_values(keywordstring, **context)
        log.info(f'modelspec: {keywordstring}')
        modelspec = Model.from_keywords(keywordstring)
        #modelspec = modelspec.sample_from_priors()
        modelspec.meta.update(meta.copy())
        modelspec.meta['engine'] = 'nems-lite'
        modelspec.meta['keywordstring'] = keywordstring0
        cellid = meta.get('cellid', 'NAT4v2')

        name = f"{cellid}/{meta['batch']}/{meta['modelname']}"
        modelspec.name = name.replace(":","")

    return {'modelspec': modelspec}


def lite_input_dict(modelspec, rec, epoch_name="", add_batch_dim=False,
                    input_name='stim', output_name='resp', extra_signals=[]):

    all_inputs, all_outputs = modelspec.get_io_names()
    existing_inputs = [i for i in all_inputs if i in rec.signals.keys()]
    if ('state' in rec.signals.keys()) and ('state' not in existing_inputs):
        existing_inputs.append('state')
    existing_inputs.extend(extra_signals)

    if len(epoch_name) > 0:
        # generate batches with epoch
        X = {'input': np.moveaxis(rec.apply_mask()[input_name].extract_epoch("REFERENCE"), -1, 1)}
        for i in existing_inputs:
            X[i] = np.moveaxis(rec.apply_mask()[input_name].extract_epoch("REFERENCE"), -1, 1)
        Y = np.moveaxis(rec.apply_mask()[output_name].extract_epoch("REFERENCE"), -1, 1)
    else:
        # single batch/epoch -- good for behavior data
        X = {'input': np.moveaxis(rec.apply_mask()[input_name].as_continuous(), -1, 0)}
        for i in existing_inputs:
            X[i] = np.moveaxis(rec.apply_mask()[i].as_continuous(), -1, 0)
        Y = np.moveaxis(rec.apply_mask()[output_name].as_continuous(), -1, 0)
        if add_batch_dim:
            X = {k: v[np.newaxis, :, :] for k, v in X.items()}
            Y = Y[np.newaxis, :, :]

    return X, Y

def fit_lite(modelspec=None, est=None, modelspec_list=None,
             input_name='stim', output_name='resp', IsReload=False,
             cost_function='nmse', learning_rate=1e-3, tolerance=1e-4, max_iter=5000, backend='scipy',
             validation_split=0.0, early_stopping_patience=150, early_stopping_delay=100, rand_count=1,
             jackknife_count=0, epoch_name="REFERENCE",
             initialize_nl=False, freeze_layers=None, shuffle_batches=False, skip_init=False, **context):
    """
    fit_wrapper modified to work with nems-lite
    :param modelspec: modelspec to fit (nems-lite format)
    :param est:  nems-format Recording with fit data
    :param input_name:
    :param output_name:
    :param IsReload:    [False] if True, skip fit and return without doing anything
    :param cost_function:
    :param learning_rate:
    :param tolerance:
    :param max_iter:
    :param backend:
    :param validation_split:
    :param early_stopping_patience:
    :param early_stopping_delay:
    :param rand_count:
    :param context:  pass-through dictionary into fitter
    :return: results = xforms context dictionary update
    """
    if IsReload:
        if jackknife_count>0:
            return {'jackknifed_fit': True, 'jackknife_count': jackknife_count}
        else:
            return {}

    if (modelspec is None) or (est is None):
        raise ValueError("Inputs modelspec and est required")

    from nems.preprocessing import JackknifeIterator
    from nems.models.dataset import DataSet

    # convert signal matrices to nems-lite format
    if backend == 'scipy':

        X_est = np.moveaxis(est.apply_mask()[input_name].as_continuous(), -1, 0)
        Y_est = np.moveaxis(est.apply_mask()[output_name].as_continuous(), -1, 0)
        if 'state' in est.signals.keys():
            S_est = np.moveaxis(est.apply_mask()['state'].as_continuous(), -1, 0)
        else:
            S_est = None
        fitter_options = {'cost_function': 'nmse',
                          'options': {'ftol': tolerance, 'gtol': tolerance / 10, 'maxiter': max_iter}}
        log.info(f"{fitter_options}")
        try:
            modelspec.layers[-1].skip_nonlinearity()
            fit_stage_1 = True
        except:
            log.info('No NL to exclude from stage 1 fit')
            fit_stage_1 = False

        if fit_stage_1:
            if rand_count>1:
                log.info(f'({backend}) Fitting without NL (rand_inits={rand_count}...')
                modelspec_copies = modelspec.sample_from_priors(5)
                modelspec_copies = [m.fit(input=X_est, target=Y_est, state=S_est,
                                    backend=backend, freeze_layers=freeze_layers, fitter_options=fitter_options)
                                    for m in modelspec_copies]
                E = np.array([m.results.final_error for m in modelspec_copies])
                E0 = np.array([m.results.initial_error[0] for m in modelspec_copies])
                best_i = np.argmax(E)
                log.info(f'Init E: {E0} Final E: {E} best={E[best_i]}')
                modelspec=modelspec_copies[best_i]
            else:
                log.info(f'({backend}) Fitting without NL ...')
                modelspec = modelspec.fit(input=X_est, target=Y_est, state=S_est,
                                          backend=backend, freeze_layers=freeze_layers, fitter_options=fitter_options)

            log.info(f'({backend}) Now fitting with NL ...')
            modelspec.layers[-1].unskip_nonlinearity()

        modelspec = modelspec.fit(input=X_est, target=Y_est, state=S_est,
                                  backend=backend, freeze_layers=freeze_layers, fitter_options=fitter_options)

    elif backend == 'tf':
        # convert signal matrices to nems-lite format
        X_est, Y_est = lite_input_dict(modelspec, est, input_name=input_name,
                                       output_name=output_name, epoch_name=epoch_name,
                                       add_batch_dim=True)

        if False:
            log.info("Adding a tiny bit of noise to X_est")
            X_est = X_est + np.random.randn(*X_est.shape) / 10000
            
        if jackknife_count>0:
            ja = int(Y_est.shape[0] <= 1)  # jk on axis with more th
            fit_set = JackknifeIterator(X_est, target=Y_est,
                                        samples=jackknife_count, axis=ja)
            log.info(f"Jackknifes: {jackknife_count}")
            E = np.zeros((rand_count, jackknife_count))
            E0 = np.zeros((rand_count, jackknife_count))
        else:
            fit_set = DataSet(X_est, target=Y_est)
            E = np.zeros(rand_count)
            E0 = np.zeros(rand_count)
            
        log.info(f"Dataset size X_est: {X_est['input'].shape} Y_est: {Y_est.shape}")
        batch_size = None #  X_est.shape[0]  # or None or bigger?
        log.info(f"Batch size: {batch_size}")

        fitter_options = {'cost_function': cost_function, 'early_stopping_delay': early_stopping_delay,
                          'early_stopping_patience': early_stopping_patience,
                          'early_stopping_tolerance': tolerance,
                          'validation_split': validation_split,
                          'learning_rate': learning_rate, 'epochs': max_iter,
                          'shuffle': shuffle_batches,
                          }
        log.info(f"LR: {learning_rate:.2e} Tol: {tolerance:.2e} Cost: {cost_function}")

        prephi = modelspec.layers[-1].get_parameter_values(as_dict=True)
        if initialize_nl:
            try:
                modelspec.layers[-1].skip_nonlinearity()
                fit_stage_1 = True
            except:
                log.info('No NL to exclude from stage 1 fit')
                fit_stage_1 = False
        else:
            fit_stage_1=False
        if fit_stage_1:
            #from nems.layers import ShortTermPlasticity
            # for i, l in enumerate(modelspec.layers):
            #    if isinstance(l, ShortTermPlasticity):
            #        log.info(f'Freezing parameters for layer {i}: {l.name}')
            #        modelspec.layers[i].freeze_parameters()
            if rand_count > 1:
                log.info(f'Initializing {rand_count} model instances based on priors')
                modelspec_copies = [modelspec.copy()] + modelspec.sample_from_priors(rand_count-1)
            elif skip_init:
                log.info('Using inherited initial conditions for model parameters')
                modelspec_copies = [modelspec.copy()]
                rand_count = 1
            else:
                log.info('Initializing based on model priors')
                modelspec_copies = [modelspec.sample_from_priors()]
                rand_count = 1

            for mi, m in enumerate(modelspec_copies):
                log.info(f'** ({backend}) Fitting without NL - initialization {mi+1}/{rand_count} ...')

                if jackknife_count <= 1:
                    m = m.fit(input=X_est, target=Y_est, backend=backend,
                              freeze_layers=freeze_layers, fitter_options=fitter_options,
                              batch_size=batch_size, verbose=1, progress_fun=progress_fun)
                    m.layers[-1].unskip_nonlinearity()
                    # special case - set initial conditions for NL layer parameters
                    m = init.init_nl_lite(m, X_est, Y_est)
                    # idea (bad one?) to only fit static NL?
                    # for i in range(len(m.layers) - 1):
                    #    m.layers[i].freeze_parameters()
                    log.info(f'** ({backend}) Fitting with NL - initialization {mi + 1}/{rand_count} ...')
                    modelspec_copies[mi] = m.fit(
                        input=X_est, target=Y_est, backend=backend,
                        freeze_layers=freeze_layers, fitter_options=fitter_options,
                        batch_size=batch_size, verbose=0, progress_fun=progress_fun)
                    E[mi] = modelspec_copies[mi].results.final_error
                    E0[mi] = modelspec_copies[mi].results.initial_error[0]
                else:
                    model_fit_list = fit_set.get_fitted_jackknifes(
                        m, backend=backend,
                        freeze_layers=freeze_layers, fitter_options=fitter_options,
                        batch_size=batch_size, verbose=1, progress_fun=progress_fun)
                    for i, d in enumerate(fit_set):
                        m = model_fit_list[i]
                        m.layers[-1].unskip_nonlinearity()
                        model_fit_list[i] = init.init_nl_lite(m, d.inputs['input'], d.targets['target'])

                    log.info(f'** ({backend}) Fitting with NL - jacks {jackknife_count} initialization {mi + 1}/{rand_count} ...')

                    model_fit_list = fit_set.get_fitted_jackknifes(
                        model_fit_list, backend=backend,
                        freeze_layers=freeze_layers, fitter_options=fitter_options,
                        batch_size=batch_size, verbose=1, progress_fun=progress_fun)

                    modelspec_copies[mi] = model_fit_list
                    for ji, m in enumerate(model_fit_list):
                        E[mi,ji] = m.results.final_error
                        E0[mi,ji] = m.results.initial_error[0]

            if rand_count > 1:
                if jackknife_count>1:
                    best_i = np.argmin(E,axis=0)
                    modelspec=[]
                    for ji in range(jackknife_count):
                        log.info(f'JK {ji}/{jackknife_count} Final E: {np.round(E[:,ji],4)} best={best_i[ji]} ({E[best_i[ji],ji]})')
                        modelspec.append(modelspec_copies[best_i[ji]][ji])
                else:
                    best_i = np.argmin(E)
                    log.info(f'Init E: {E0}  Final E: {E} best={best_i} ({E[best_i]})')
                    modelspec = modelspec_copies[best_i]
            else:
                modelspec = modelspec_copies[0]
                
            if jackknife_count>1:
                modelspec_list = modelspec
                modelspec = modelspec_list[0]

            # is this necessary for anything?
            for i in range(len(modelspec.layers)):
                modelspec.layers[i].unfreeze_parameters()

        else:
            log.info(f'({backend}) Fitting all layers (lr={learning_rate} tol={tolerance:.3e}) ...')
            if jackknife_count<=1:

                log.info('Using inherited model parameters')
                modelspec = modelspec.fit(
                    input=X_est, target=Y_est, backend=backend,
                    freeze_layers=freeze_layers, fitter_options=fitter_options,
                    batch_size=batch_size, progress_fun=progress_fun)
            else:
                if modelspec_list is not None:
                    log.info(f'Jackknifes {jackknife_count} starting with list')
                    modelspec = modelspec_list
                else:
                    log.info(f'Jackknifes {jackknife_count} starting single init model')

                modelspec_list = fit_set.get_fitted_jackknifes(
                    modelspec, backend=backend,
                    freeze_layers=freeze_layers, fitter_options=fitter_options,
                    batch_size=batch_size, verbose=1, progress_fun=progress_fun)
                modelspec = modelspec_list[0]

        postphi = modelspec.layers[-1].get_parameter_values(as_dict=True)
        modelspec.meta['prephi'] = prephi
        modelspec.meta['postphi'] = postphi
        modelspec.meta['n_parms'] = modelspec.results.n_parameters
        
        modelspec.backend = None
        if jackknife_count > 1:
            for j, m in enumerate(modelspec_list):
                m.backend = None
                m.meta.update(modelspec.meta)
            return {'modelspec': modelspec, 'modelspec_list': modelspec_list,
                    'jackknifed_fit': True, 'jackknife_count': jackknife_count,
                    }
        else:
            return {'modelspec': modelspec}


def fit_lite_per_cell(modelspec=None, est=None, input_name='stim', output_name='resp', IsReload=False,
             cost_function='nmse', learning_rate=1e-3, tolerance=1e-5, max_iter=100, backend='scipy',
             validation_split=0.0, early_stopping_patience=20, early_stopping_delay=100,
             chop_layers=2, epoch_name="REFERENCE",
             **context):
    """
    Wrapper to loop through all jackknifes, fits and output slices (if/when >1 of any)
       for a modelspec, calling fit_function to fit each.
    Modified to work with nems-lite
    :param modelspec:  modelspec to fit
    :param est:  nems-format Recording with fit data
    :param IsReload:    [False] if True, skip fit and return without doing anything
    :param context:  pass-through dictionary into fitter
    :return: results = xforms context dictionary update
    """
    if IsReload:
        return {}

    if (modelspec is None) or (est is None):
        raise ValueError("Inputs modelspec and est required")

    from nems import Model

    save_params = [l.get_parameter_values(as_dict=True) for l in modelspec.layers]
    save_layers = range(len(save_params) - chop_layers, len(save_params))

    cell_count = est['resp'].shape[0]

    modelspecname = modelspec.meta['modelspecname']

    est1 = est.copy()
    est1['resp'] = est1['resp'].extract_channels([est1['resp'].chans[0]])
    keywordstring = init.fill_keyword_string_values(modelspecname, rec=est1)
    log.info(f'sliced modelspec: {keywordstring}')
    tmodel = Model.from_keywords(keywordstring)

    for i, d in enumerate(save_params):
        if i < save_layers[0]:
            tmodel.layers[i].set_parameter_values(d, ignore_bounds=True)
            tmodel.layers[i].freeze_parameters()

    for cid in range(cell_count):
        log.info("")
        log.info(f"Fitting single cell: ({cid}) {est['resp'].chans[cid]}")
        # initialized with parameter values for this cell
        for i, s in enumerate(save_layers):
            d = {}
            for k in save_params[s].keys():
                if save_params[s][k].ndim == 1:
                    d[k] = save_params[s][k][[cid]]
                else:
                    d[k] = save_params[s][k][:, [cid]]

            tmodel.layers[s].set_parameter_values(d, ignore_bounds=True)

        est1 = est.copy()
        est1['resp'] = est1['resp'].extract_channels([est1['resp'].chans[cid]])
        X_est = np.moveaxis(est1.apply_mask()[input_name].extract_epoch("REFERENCE"), -1, 1)
        Y_est = np.moveaxis(est1.apply_mask()[output_name].extract_epoch("REFERENCE"), -1, 1)

        if 'state' in est.signals.keys():
            S_est = np.moveaxis(est.apply_mask()['state'].extract_epoch("REFERENCE"), -1, 1)
        else:
            S_est = None

        fitter_options = {'cost_function': cost_function, 'early_stopping_delay': early_stopping_delay,
                          'early_stopping_patience': early_stopping_patience,
                          'early_stopping_tolerance': tolerance,
                          'validation_split': validation_split,
                          'learning_rate': learning_rate, 'epochs': max_iter,
                          }
        tmodel = tmodel.fit(
            input=X_est, target=Y_est, state=S_est, backend=backend,
            fitter_options=fitter_options, batch_size=None)

        # put parameters back into main model
        for i, s in enumerate(save_layers):
            d = tmodel.layers[s].get_parameter_values(as_dict=True)
            for k in save_params[s].keys():
                #print(f"updating {k}[{cid}] from {save_params[s][k][[cid]]} to {d[k]}")
                if save_params[s][k].ndim == 1:
                    save_params[s][k][[cid]] = d[k]
                else:
                    save_params[s][k][:, [cid]] = d[k]

    for i, s in enumerate(save_layers):
        modelspec.layers[s].set_parameter_values(save_params[s], ignore_bounds=True)

    modelspec.backend = None

    return {'modelspec': modelspec}


def predict_lite(modelspec, est, val,
                 input_name='stim', output_name='resp',
                 jackknifed_fit=False, modelspec_list=None,
                 IsReload=False, **context):

    # convert signal matrices to nems-lite format
    # single batch/epoch -- good for behavior data
    X_est, Y_est = lite_input_dict(modelspec, est, input_name=input_name, output_name=output_name, epoch_name="")
    X_val, Y_val = lite_input_dict(modelspec, val, input_name=input_name, output_name=output_name, epoch_name="")

    fit_pred = modelspec.predict(X_est)
    prediction = modelspec.predict(X_val)
    if type(prediction) is dict:
        fit_pred = fit_pred['output']
        prediction = prediction['output']
    edata = np.zeros(est['resp'].shape) * np.nan
    vdata = np.zeros(val['resp'].shape) * np.nan
    if 'mask' in est.signals.keys():
        edata[:, est['mask']._data[0,:].astype(bool)]=fit_pred.T
        vdata[:, val['mask']._data[0,:].astype(bool)] = prediction.T
    else:
        edata = fit_pred.T
        vdata = prediction.T
    est['pred']=est[output_name]._modified_copy(data=edata)
    val['pred']=val[output_name]._modified_copy(data=vdata)

    if jackknifed_fit:
        from nems.preprocessing import JackknifeIterator
        from nems.models.dataset import DataSet

        jackknife_count = len(modelspec_list)
        ja = int(Y_est.shape[0] <= 1)  # jk on axis with more th
        test_set = JackknifeIterator(X_est, target=Y_est,
                                    samples=jackknife_count, axis=ja)
        #log.info(f"Jackknifes: {jackknife_count}")
        dataset = test_set.get_predicted_jackknifes(modelspec_list)

        try:
            m = est['mask'].as_continuous()[0,:]
        except:
            m = np.ones(est[output_name].shape[-1], dtype=bool)

        s = list(dataset['output'].T.shape)[:-1]+[len(m)]
        d = np.zeros(s)
        d[:,m] = dataset['output'].T
        est['pred'] = est[output_name]._modified_copy(data=d)
        if est['pred'].shape==val['pred'].shape:
            val['pred'] = val[output_name]._modified_copy(data=d)

        for k,v in dataset.items():
            if k not in ['output','target','stim','dlc']:
                # don't overwrite signals from original recording (kludgy)
                s = list(v.T.shape)[:-1] + [len(m)]
                d = np.zeros(s)
                d[:, m] = v.T
                est[k] = est[output_name]._modified_copy(data=d)
                if est['pred'].shape == val['pred'].shape:
                    val[k] = val[output_name]._modified_copy(data=d)

    return {'val': val, 'est': est}


def plot_lite(modelspec, val, input_name='stim', output_name='resp', IsReload=False,
              figures=None, figures_to_load=None, **context):

    from nems.visualization import model

    if figures is None:
        figures = []
    if IsReload:
        if figures_to_load is not None:
            figures.extend([load_resource(f) for f in figures_to_load])
        return {'figures': figures}

    # convert signal matrices to nems-lite format
    X_val, Y_val = lite_input_dict(modelspec, val, input_name=input_name, output_name=output_name, epoch_name="")
    fig = model.plot_model(
        modelspec, X_val, target=Y_val, sampling_rate=val[output_name].fs)

    if fig is not None:
        # Needed to make into a Bytes because you can't deepcopy figures!
        figures.append(nplt.fig2BytesIO(fig))
    else:
        log.info("plot_model returned None instead of figure handle?")

    return {'figures': figures}


def save_lite(modelspec=None, xfspec=None, log=None, figures=[], IsReload=False,
              modelspec_list=None, **ctx):
    from nems.tools import json

    if IsReload:
        return modelspec.meta['modelpath']

    if get_setting('USE_NEMS_BAPHY_API'):
        prefix = 'http://'+get_setting('NEMS_BAPHY_API_HOST')+":"+str(get_setting('NEMS_BAPHY_API_PORT')) + '/results/'
    else:
        prefix = get_setting('NEMS_RESULTS_DIR')
    batch = modelspec.meta.get('batch', 0)
    cellid = modelspec.meta.get('cellid', 'cell')
    basepath = os.path.join(prefix, str(batch), cellid)

    # use nems-lite model path name
    filepath = json.generate_model_filepath(modelspec, basepath=basepath)
    destination = os.path.dirname(filepath)
    modelspec.meta['modelpath'] = destination
    modelspec.meta['figurefile'] = os.path.join(destination, 'figure0000.png')
    # call nems-lite JSON encoder
    data = json.nems_to_json(modelspec)
    save_resource(os.path.join(destination, 'modelspec.json'), data=data)
    if modelspec_list is not None:
        for i,m in enumerate(modelspec_list):
            #log.info(f'Saving modelspec_list {i}')
            data = json.nems_to_json(m)
            save_resource(os.path.join(destination, f'modelspec_list{i:04d}.json'), data=data)

    for number, figure in enumerate(figures):
        fig_uri = os.path.join(destination, 'figure.{:04d}.png'.format(number))
        #log.info('saving figure %d to %s', number, fig_uri)
        save_resource(fig_uri, data=figure)

    if log is not None:
        save_resource(os.path.join(destination, 'log.txt'), data=log)
    if xfspec is not None:
        save_resource(os.path.join(destination, 'xfspec.json'), json=xfspec)
    return destination





###############################################################################
#########################     PREPROCESSORS     ###############################
###############################################################################


def normalize_stim(rec=None, sig='stim', norm_method='meanstd', **context):
    """
    Normalize each channel of rec[sig] according to norm_method
    :param rec:  NEMS recording
    :param norm_method:  string {'meanstd', 'minmax'}
    :param context: pass-through for other variables in xforms context dictionary that aren't used.
    :return: copy(?) of rec with updated signal.
    """
    if sig in rec.signals.keys():
        rec[sig] = rec.copy()[sig].rasterize().normalize(norm_method)
    else:
        log.info(f'Signal {sig} not in recording, skipping normalize')
    return {'rec': rec}


def normalize_sig(rec=None, rec_list=None, sig='stim', norm_method='meanstd', log_compress='None',
                  chop_channels=0, verbose=True, **context):
    """
    Normalize each channel of rec[sig] according to norm_method
    :param rec:  NEMS recording
    :param norm_method:  string {'meanstd', 'minmax'}
    :param context: pass-through for other variables in xforms context dictionary that aren't used.
    :return: copy(?) of rec with updated signal.
    """
    if rec_list is None:
        rec_list = [rec]
        return_reclist = False
    else:
        rec=rec_list[0]
        return_reclist = True
    new_rec_list = []
    b,g = None, None
    if sig in rec.signals.keys():
        for i, r in enumerate(rec_list):
            newrec = r.copy()
            s = newrec[sig].rasterize()
            if norm_method=='sqrt':
                if verbose:
                    log.info(f'xforms.normalize_sig({norm_method}): {sig}')
                newrec[sig] = s.normalize_sqrt(mask=newrec['mask'])
            else:
                if log_compress != 'None':
                   from nems0.modules.nonlinearity import _dlog
                   fn = lambda x: _dlog(x, -log_compress)
                   s=s.transform(fn, sig)
                newrec[sig] = s.normalize(norm_method, b=b, g=g, mask=newrec['mask'])
                new_rec_list.append(newrec)
                if (sig=='stim') and (i==0):
                    b=newrec[sig].norm_baseline
                    g=newrec[sig].norm_gain
                if (sig=='resp') and (chop_channels>0):
                    keepchans = newrec[sig].chans[-chop_channels:]
                    log.info(f'Keep chans: {keepchans}')
                    newrec[sig]=newrec[sig].extract_channels(chans=keepchans)
                if verbose:
                    log.info(f'xforms.normalize_sig({norm_method}): {sig} b={newrec[sig].norm_baseline.mean()}, g={newrec[sig].norm_gain.mean()}, dlog(..., -{log_compress})')
            
        if return_reclist:
            return {'rec': rec_list[0], 'rec_list': new_rec_list}
        else:
            return {'rec': newrec}
    else:
        log.info(f'Signal {sig} not in recording, skipping normalize')
        return {}


@xform()
def const(kw):
    ops = kw.split('.')[1:]
    d = {}
    for op in ops:
        d['sig']=op

    xfspec = [['nems0.xforms.concat_constant', d]]

    return xfspec

    
def concat_constant(rec, rec_list=None, sig='stim', use_mean_val=True, **context):
    """
    add a constant channel (all 1s) to rec[stim]
    """
    
    if rec_list is None:
        rec_list = [rec]
        return_reclist = False
    else:
        rec=rec_list[0]
        return_reclist = True
    new_rec_list = []
    
    for i, r in enumerate(rec_list):
        if sig in r.signals.keys():
            newrec = r.copy()
            if (i==0) & use_mean_val:
                m = np.nanmean(newrec[sig].rasterize()._data)
            else:
                m = 1
            s = newrec[sig].rasterize()
            sh = s.shape
            ch = s.chans
            d = np.concatenate([s._data, np.ones((1, sh[1]))*m], axis=0)
            if s.chans is None:
                chans = [f"STIM{i}" for i in range(sh[0])] + ['C']
            else:
                chans = s.chans + ['C']
            newrec[sig] = s._modified_copy(data=d, chans=chans)
            log.info(f"Added const to signal {sig}, shape: {newrec[sig].shape}")
        new_rec_list.append(newrec)
    
    if return_reclist:
        return {'rec': rec_list[0], 'rec_list': new_rec_list}
    else:
        return {'rec': newrec}


def add_average_sig(rec, signal_to_average, new_signalname, epoch_regex,
                    **context):
    rec = preproc.add_average_sig(rec,
                                  signal_to_average=signal_to_average,
                                  new_signalname=new_signalname,
                                  epoch_regex=epoch_regex)
    return {'rec': rec}


def remove_all_but_correct_references(rec, **context):
    '''
    find REFERENCE epochs spanned by either PASSIVE_EXPERIMENT or
    HIT_TRIAL epochs. remove all other segments from signals in rec
    '''
    rec = preproc.remove_invalid_segments(rec)

    return {'rec': rec}


def mask_all_but_correct_references(rec, balance_rep_count=False,
                                    include_incorrect=False, generate_evoked_mask=False,
                                    **context):
    '''
    find REFERENCE epochs spanned by either PASSIVE_EXPERIMENT or
    HIT_TRIAL epochs. mask out all other segments from signals in rec
    '''
    rec = preproc.mask_all_but_correct_references(
            rec, balance_rep_count=balance_rep_count,
            include_incorrect=include_incorrect, generate_evoked_mask=generate_evoked_mask)

    return {'rec': rec}


def mask_all_but_targets(rec, **context):
    '''
    find TARGET epochs all behaviors/outcomes
    '''
    rec = preproc.mask_all_but_targets(rec)

    return {'rec': rec}


def mask_incorrect(rec, **options):
    '''
    Create mask removing incorrect trials
    '''
    rec = preproc.mask_incorrect(rec, **options)

    return {'rec': rec}


def generate_psth_from_resp(rec, epoch_regex='^(STIM_|TAR_|CAT_|REF_)', use_as_input=True,
                            smooth_resp=False, channel_per_stim=False, mean_zero=False, **context):
    '''
    generate PSTH prediction from rec['resp'] (before est/val split). Could
    be considered "cheating" b/c predicted PSTH then is based on data in
    val set, but this is because we're interested in testing state effects,
    not sensory coding models. The appropriate control, however is to run
    generate_psth_from_est_for_both_est_and_val_nfold on each nfold est/val
    split.
    '''
    rec = preproc.generate_psth_from_resp(rec, epoch_regex=epoch_regex,
                                          smooth_resp=smooth_resp,
                                          channel_per_stim=channel_per_stim,
                                          mean_zero=mean_zero)
    if use_as_input:
        return {'rec': rec, 'input_name': 'psth'}
    else:
        return {'rec': rec}


def generate_psth_from_est_for_both_est_and_val_nfold(
        est, val, epoch_regex='^STIM_', mean_zero=False, **context):
    '''
    generate PSTH prediction for each set
    '''
    est_out, val_out = \
        preproc.generate_psth_from_est_for_both_est_and_val_nfold(est, val,
                                          mean_zero=mean_zero)
    return {'est': est_out, 'val': val_out}


def make_state_signal(rec, state_signals=['pupil'], permute_signals=[], generate_signals=[],
                      new_signalname='state', sm_win_len=180, **context):

    rec = preproc.make_state_signal(rec, state_signals=state_signals,
                                    permute_signals=permute_signals,
                                    generate_signals=generate_signals,
                                    new_signalname=new_signalname, sm_win_len=sm_win_len)

    return {'rec': rec}


def concatenate_input_channels(rec, input_signals=[], **context):
    input_name = context.get('input_name', 'stim')
    rec = preproc.concatenate_input_channels(rec, input_signals=input_signals,
                                             input_name=input_name)

    return {'rec': rec}

def make_mod_signal(rec, signal='resp'):
    """
    Make new signal called mod that can be used for calculating an unbiased
    mod_index
    """
    new_rec = rec.copy()
    psth = new_rec['psth_sp']
    resp = new_rec[signal]
    mod_data = resp.as_continuous() - psth.as_continuous()
    mod = psth._modified_copy(mod_data)
    mod.name = 'mod'
    new_rec.add_signal(mod)

    return new_rec

@xform()
def sev(kw):
    ops = kw.split('.')[1:]
    parms={'epoch_regex': '^STIM'}
    continuous=False
    for op in ops:
        if op=='seq':
            parms['epoch_regex'] = '^STIM_se'
        elif op == 'a':
            parms['est_all']=True
        elif op == 'cont':
            continuous = True
        elif op == 'mono':
            parms['selection'] = 'mono'
        elif op == 'bin':
            parms['selection'] = 'bin'
        elif op == 'match':
            parms['selection'] = 'match'
        elif op.startswith("k"):
            parms['keepfrac'] = int(op[1:]) / 100
        elif op.startswith("f"):
            parms['filemask'] = op[1:]
        else:
            parms['epoch_regex'] = op
    
    xfspec = [['nems0.xforms.split_by_occurrence_counts', parms]]
    if not continuous:
        xfspec.append(['nems0.xforms.average_away_stim_occurrences', parms])
    return xfspec

def split_by_occurrence_counts(rec, epoch_regex='^STIM_', rec_list=None, filemask=None, keepfrac=1, **context):

    if rec_list is None:
        rec_list = [rec]
        return_reclist = False
    else:
        rec=rec_list[0]
        return_reclist = True
    est_list = []
    val_list = []

    for rec in rec_list:
        est, val = rec.split_using_epoch_occurrence_counts(
            epoch_regex=epoch_regex, keepfrac=keepfrac, filemask=filemask, **context)
        est_list.append(est)
        val_list.append(val)

    if return_reclist:
        c = {'est': est_list[0], 'val': val_list[0], 'est_list': est_list, 'val_list': val_list}
    else:
        c = {'est': est, 'val': val}

    if context.get('selection', '') in ['mono','bin','match']:
        c['eval_binaural'] = True
    return c

@xform()
def tev(kw):
    ops = kw.split('.')[1:]

    valfrac = 0.1
    for op in ops:
        if op.startswith("vv"):
            valfrac=int(op[2:]) / 1000
        elif op.startswith("v"):
            valfrac=int(op[1:]) / 100

    xfspec = [['nems0.xforms.split_at_time', {'valfrac': valfrac}]]

    return xfspec


@xform()
def avgreps(kw):
    ops = kw.split('.')[1:]
    epoch_regex = '^STIM'
    if 'seq' in ops:
        epoch_regex = '^STIM_se'
    for op in ops:
        epoch_regex = op

    xfspec = [['nems0.xforms.average_away_stim_occurrences', {'epoch_regex': epoch_regex}]]

    return xfspec


def average_away_stim_occurrences(est=None, val=None, rec=None,
                                  est_list=None, val_list=None, rec_list=None,
                                  epoch_regex='^STIM', **context):
    if rec_list is None:
        rec_list = [rec]
        est_list = [est]
        val_list = [val]
        return_reclist = False
    else:
        return_reclist = True

    if (est is not None):
        new_est_list=[]
        new_val_list=[]
        for est, val in zip(est_list, val_list):
            est = preproc.average_away_epoch_occurrences(est, epoch_regex=epoch_regex)
            val = preproc.average_away_epoch_occurrences(val, epoch_regex=epoch_regex)
            new_est_list.append(est)
            new_val_list.append(val)
        if return_reclist:
            return {'est': new_est_list[0], 'val': new_val_list[0],
                    'est_list': new_est_list, 'val_list': new_val_list}
        else:
            return {'est': est, 'val': val}
    else:
        new_rec_list=[]
        for rec in rec_list:
            rec = preproc.average_away_epoch_occurrences(rec, epoch_regex=epoch_regex)
            new_rec_list.append(rec)

        if return_reclist:
            return {'rec': new_rec_list[0], 'rec_list': new_rec_list}
        else:
            return {'rec': rec}

    # mask out nan periods
#    d=np.isfinite(est['resp'].as_continuous()[[0],:])
#    log.info('found %d non-nans in est', np.sum(d))
#    est=est.create_mask()
#    est['mask']=est['mask']._modified_copy(d)
#
#    d=np.isfinite(val['resp'].as_continuous()[[0],:])
#    log.info('found %d non-nans  in val', np.sum(d))
#    val=val.create_mask()
#    val['mask']=val['mask']._modified_copy(d)




def average_away_stim_occurrences_rec(rec, epoch_regex='^STIM_', **context):
    rec = preproc.average_away_epoch_occurrences(rec, epoch_regex=epoch_regex)
    return {'rec': rec}


def split_val_and_average_reps(rec, epoch_regex='^STIM_', **context):
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex=epoch_regex)
    est = preproc.average_away_epoch_occurrences(est, epoch_regex=epoch_regex)
    val = preproc.average_away_epoch_occurrences(val, epoch_regex=epoch_regex)

    return {'est': est, 'val': val}


def use_all_data_for_est_and_val(rec, **context):
    est = rec.copy()
    val = rec.copy()
    signals = ['resp', 'stim']
    signals = [s for s in rec.signals.keys()]
    for s in signals:
        rec[s] = rec[s].rasterize()
        est[s] = est[s].rasterize()
        val[s] = val[s].rasterize()

    return {'rec': rec, 'est': est, 'val': val}


def split_for_jackknife(rec, modelspecs=None, epoch_name='REFERENCE',
                        njacks=10, IsReload=False, **context):

    est_out, val_out, modelspecs_out = \
        preproc.split_est_val_for_jackknife(rec, modelspecs=modelspecs,
                                            epoch_name=epoch_name,
                                            njacks=njacks, IsReload=IsReload)
    if IsReload:
        return {'est': est_out, 'val': val_out}
    else:
        return {'est': est_out, 'val': val_out, 'modelspecs': modelspecs_out}


def mask_for_jackknife(rec, modelspec=None, epoch_name=None,
                       epoch_regex='(REFERENCE|TARGET|CATCH|ITI)',
                       by_time=False, njacks=10, IsReload=False,
                       allow_partial_epochs=False, **context):

    _rec = rec.copy()
    _rec['resp'] = _rec['resp'].rasterize()
    if 'stim' in _rec.signals.keys():
        _rec['stim'] = _rec['stim'].rasterize()
    if epoch_name is not None:
        epoch_regex=epoch_name
    elif epoch_regex is None:
        epoch_regex=epoch_name
    
    if by_time != True:
        est_out, val_out, modelspec_out = \
            preproc.mask_est_val_for_jackknife(_rec, modelspec=modelspec,
                                               epoch_name=epoch_name, epoch_regex=epoch_regex,
                                               njacks=njacks,
                                               allow_partial_epochs=allow_partial_epochs,
                                               IsReload=IsReload)
    else:
        est_out, val_out, modelspec_out = \
            preproc.mask_est_val_for_jackknife_by_time(_rec, modelspec=modelspec,
                                                       njacks=njacks,
                                                       IsReload=IsReload)

    if IsReload:
        return {'est': est_out, 'val': val_out}
    else:
        return {'est': est_out, 'val': val_out,
                'jackknifed_fit': True, 'modelspec': modelspec_out}


def jack_subset(est, val, modelspec=None, IsReload=False,
                keep_only=1, **context):

    if keep_only == 1:
        est = est.views(view_range=0)[0]
        val = val.views(view_range=0)[0]
        est['resp'] = est['resp'].rasterize()
        val['resp'] = val['resp'].rasterize()
        if 'stim' in est.signals.keys():
            est['stim'] = est['stim'].rasterize()
            val['stim'] = val['stim'].rasterize()

    else:
        est = est.views(keep_only)[0]
        val = val.views(keep_only)[0]

    if modelspec is not None:
        modelspec_out = modelspec.copy(jack_index=keep_only)
        modelspec_out.fit_index = 0

    if IsReload:
        return {'est': est, 'val': val, 'jackknifed_fit': False}
    else:
        return {'est': est, 'val': val, 'modelspec': modelspec_out, 'jackknifed_fit': False}


###############################################################################
######################        INITIALIZERS         ############################
###############################################################################


def fit_basic_init(modelspec, est, tolerance=10**-5.5, max_iter=1500, metric='nmse',
                   IsReload=False, norm_fir=False, nl_kw={},
                   output_name='resp', **context):
    '''
    Initialize modelspecs in a way that avoids getting stuck in
    local minima.

    written/optimized to work for (dlog)-wc-(stp)-fir-(dexp) architectures
    optional modules in (parens)
    '''
    # only run if fitting
    if IsReload:
        return {}

    if isinstance(metric, str):
        metric_fn = lambda d: getattr(metrics, metric)(d, 'pred', output_name)
    else:
        metric_fn = metric
    modelspec = nems0.initializers.prefit_LN(
            est, modelspec,
            analysis_function=analysis.api.fit_basic,
            fitter=scipy_minimize, metric=metric_fn,
            tolerance=tolerance, max_iter=max_iter, norm_fir=norm_fir,
            nl_kw=nl_kw)
    return {'modelspec': modelspec}

def fit_basic_subset(modelspec, est, metric='nmse', output_name='resp',
                     IsReload=False, **context):
    '''
    Initialize modelspecs in a way that avoids getting stuck in
    local minima.

    written/optimized to work for (dlog)-wc-(stp)-fir-(dexp) architectures
    optional modules in (parens)
    '''
    # only run if fitting
    if IsReload:
        return {}

    if isinstance(metric, str):
        if metric == 'pup_dep_LV':
            metric_fn = lambda d: getattr(metrics, metric)(d, 'pred', output_name, **context)
        else:
            metric_fn = lambda d: getattr(metrics, metric)(d, 'pred', output_name)
    else:
        metric_fn = metric
    modelspec = nems0.initializers.prefit_subset(
            est, modelspec, metric=metric_fn, **context)
    return {'modelspec': modelspec}

"""
    # TODO : merge JK and non-JK code if possible.
    if jackknifed_fit:
        nfolds = est.view_count
        if modelspec.jack_count < est.view_count:
            modelspec.tile_jacks(nfolds)
            # TODO replace with tile_jacks
            #  allow nested loop for multiple fits (init conditions) within a jackknife
            #  initialize each jackknife with the same random ICs?
        for fit_idx in range(modelspec.fit_count):
            for jack_idx, e in enumerate(est.views()):
                modelspec.fit_index = fit_idx
                log.info("----------------------------------------------------")
                log.info("Init fitting model fit %d/%d, fold %d/%d",
                         fit_idx+1, modelspec.fit_count,
                         jack_idx + 1, modelspec.jack_count)

                modelspec = nems0.initializers.prefit_LN(
                        e, modelspec.set_jack(jack_idx),
                        analysis_function=analysis.api.fit_basic,
                        fitter=scipy_minimize, metric=metric_fn,
                        tolerance=tolerance, max_iter=700, norm_fir=norm_fir,
                        nl_kw=nl_kw)
    else:
        #import pdb
        #pdb.set_trace()
        for fit_idx in range(modelspec.fit_count):
            log.info("Init fitting model instance %d/%d", fit_idx + 1, modelspec.fit_count)
            modelspec = nems0.initializers.prefit_LN(
                    est, modelspec.set_fit(fit_idx),
                    analysis_function=analysis.api.fit_basic,
                    fitter=scipy_minimize, metric=metric_fn,
                    tolerance=tolerance, max_iter=700, norm_fir=norm_fir,
                    nl_kw=nl_kw)
"""

def _set_zero(x):
    """ fill x with zeros, except preserve nans """
    y = x.copy()
    if y.shape[0]>1:
        y[0,np.isfinite(y[0,:])] = 1
        y[1:,np.isfinite(y[0,:])] = 0
    else:
        y[np.isfinite(y)] = 0
    return y


def fit_state_init(modelspec, est, tolerance=10**-5.5, max_iter=1500, metric='nmse',
                   IsReload=False, norm_fir=False, nl_kw = {},
                   fit_sig='resp', output_name='resp', **context):

    '''
    Initialize modelspecs in an attempt to avoid getting stuck in
    local minima. Remove state replication/merging first.

    written/optimized to work for (dlog)-wc-(stp)-fir-(dexp) architectures
    optional modules in (parens)

    assumption -- est['state'] signal is being used for merge
    '''
    if IsReload:
        return {}

    metric_fn = lambda d: getattr(metrics, metric)(d, 'pred', output_name)

    # set state to 0 for all timepoints so that only first filterbank
    # is used
    dc = est.copy()
    dc['state'] = dc['state'].transform(_set_zero, 'state')
    if fit_sig != 'resp':
        log.info("Subbing %s for resp signal", fit_sig)
        dc['resp'] = dc[fit_sig]
    modelspec = nems0.initializers.prefit_LN(
            dc, modelspec,
            analysis_function=analysis.api.fit_basic,
            fitter=scipy_minimize, metric=metric_fn,
            tolerance=tolerance, max_iter=max_iter, norm_fir=norm_fir,
            nl_kw=nl_kw)

    # fit a bit more to settle in STP variables and anything else
    # that might have been excluded
    # SVD disabling to speed up
    #fit_kwargs = {'tolerance': tolerance/2, 'max_iter': 500}
    #modelspec = analysis.api.fit_basic(
    #        dc, modelspec, fit_kwargs=fit_kwargs, metric=metric_fn,
    #        fitter=scipy_minimize)

    rep_idx = find_module('replicate_channels', modelspec)
    mrg_idx = find_module('merge_channels', modelspec)
    if rep_idx is not None:
        repcount = modelspec[rep_idx]['fn_kwargs']['repcount']
        for j in range(rep_idx+1, mrg_idx):
            # assume all phi
            log.debug(modelspec[j]['fn'])
            if 'phi' in modelspec[j].keys():
                for phi in modelspec[j]['phi'].keys():
                    s = modelspec[j]['phi'][phi].shape
                    setcount = int(s[0] / repcount)
                    log.debug('phi[%s] setcount=%d', phi, setcount)
                    snew = np.ones(len(s))
                    snew[0] = repcount
                    new_v = np.tile(modelspec[j]['phi'][phi][:setcount, ...],
                                    snew.astype(int))
                    log.debug(new_v)
                    modelspec[j]['phi'][phi] = new_v

    return {'modelspec': modelspec}


###############################################################################
########################       FITTERS / ANALYSES      ########################
###############################################################################


def fit_basic(modelspec, est, max_iter=1000, tolerance=1e-7,
              metric='nmse', IsReload=False, fitter='scipy_minimize',
              jackknifed_fit=False, random_sample_fit=False,
              n_random_samples=0, random_fit_subset=None,
              output_name='resp', **context):
    ''' A basic fit that optimizes every input modelspec. '''

    if IsReload:
        return {}
    metric_fn = lambda d: getattr(metrics, metric)(d, 'pred', output_name)
    fitter_fn = getattr(nems0.fitters.api, fitter)
    fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}

    if modelspec.jack_count < est.view_count:
        raise Warning('modelspec.jack_count does not match est.view_count')
        # modelspec.tile_jacks(nfolds)
    for fit_idx in range(modelspec.fit_count):
        for jack_idx, e in enumerate(est.views()):
            modelspec.jack_index = jack_idx
            modelspec.fit_index = fit_idx
            log.info("----------------------------------------------------")
            log.info("Fitting: fit %d/%d, fold %d/%d (tol=%.2e, max_iter=%d)",
                     fit_idx + 1, modelspec.fit_count,
                     jack_idx + 1, modelspec.jack_count, tolerance, max_iter)
            modelspec = analysis.api.fit_basic(
                    e, modelspec, fit_kwargs=fit_kwargs,
                    metric=metric_fn, fitter=fitter_fn)

    return {'modelspec': modelspec}


def reverse_correlation(modelspec, est, IsReload=False, jackknifed_fit=False,
                input_name='stim', output_name='resp', **context):
    ''' Perform basic normalized reverse correlation between input '''

    if not IsReload:

        if jackknifed_fit:
            nfolds = est.view_count
            if modelspec.jack_count < est.view_count:
                modelspec.tile_jacks(nfolds)
            for fit_idx in range(modelspec.fit_count):
                for jack_idx, e in enumerate(est.views()):
                    modelspec.fit_index = fit_idx
                    modelspec.jack_index = jack_idx
                    log.info("----------------------------------------------------")
                    log.info("Fitting: fit %d/%d, fold %d/%d",
                             fit_idx + 1, modelspec.fit_count,
                             jack_idx + 1, modelspec.jack_count)

                    modelspec = analysis.api.reverse_correlation(
                            e, modelspec, input_name)

        else:
            # standard single shot
            for fit_idx in range(modelspec.fit_count):
                modelspec = analysis.api.reverse_correlation(
                    est, modelspec.set_fit(fit_idx), input_name)

    return {'modelspec': modelspec}


def fit_iteratively(modelspec, est, tol_iter=100, fit_iter=20, IsReload=False,
                    module_sets=None, invert=False, tolerances=[1e-4, 1e-5, 1e-6, 1e-7],
                    metric='nmse', fitter='scipy_minimize', fit_kwargs={},
                    jackknifed_fit=False, output_name='resp', **context):
    if IsReload:
        return {}

    fitter_fn = getattr(nems0.fitters.api, fitter)
    metric_fn = lambda d: getattr(metrics, metric)(d, 'pred', output_name)

    if modelspec.jack_count < est.view_count:
        raise Warning('modelspec.jack_count does not match est.view_count')
        modelspec.tile_jacks(nfolds)

    for fit_idx in range(modelspec.fit_count):
        for jack_idx, e in enumerate(est.views()):
            modelspec.jack_index = jack_idx
            modelspec.fit_index = fit_idx
            log.info("----------------------------------------------------")
            log.info("Iter fitting: fit %d/%d, fold %d/%d",
                     fit_idx + 1, modelspec.fit_count,
                     jack_idx + 1, modelspec.jack_count)
            modelspec = analysis.api.fit_iteratively(
                        e, modelspec, fit_kwargs=fit_kwargs,
                        fitter=fitter_fn, module_sets=module_sets,
                        invert=invert, tolerances=tolerances,
                        tol_iter=tol_iter, fit_iter=fit_iter,
                        metric=metric_fn)

    return {'modelspec': modelspec}


def fit_wrapper(modelspec, est=None, fit_function='nems0.analysis.api.fit_basic',
                fit_slices_start=None, IsReload=False, **context):
    """
    Wrapper to loop through all jackknifes, fits and output slices (if/when >1 of any)
       for a modelspec, calling fit_function to fit each
    :param modelspec:  modelspec to fit
    :param est:  nems-format Recording with fit data
    :param fit_function:  string lib path to fit function
    :param fit_slices_start: [None] If integer, iterate through output channels and
       fit each separately. start_at is set to fit_slices_start
       (assumes freezing common layers or will make for an interesting fit!)
    :param IsReload:    [False] if True, skip fit and return without doing anything
    :param context:  pass-through dictionary into fitter
    :return: results = xforms context dictionary update
    """

    if IsReload:
        return {}

    if (modelspec is None) or (est is None):
        raise ValueError("Inputs modelspec and est required")

    if modelspec.jack_count < est.view_count:
        raise Warning('modelspec.jack_count does not match est.view_count')
        modelspec.tile_jacks(nfolds)

    # load function into path
    fn = lookup_fn_at(fit_function)

    for fit_idx in range(modelspec.fit_count):
        for jack_idx, e in enumerate(est.views()):
            modelspec.jack_index = jack_idx
            modelspec.fit_index = fit_idx
            log.info("----------------------------------------------------")
            log.info("Fitting: %s, fit %d/%d, fold %d/%d", fit_function,
                     fit_idx + 1, modelspec.fit_count,
                     jack_idx + 1, modelspec.jack_count)

            if fit_slices_start is not None:
                output_name = modelspec.meta['output_name']
                output_count = est[output_name].shape[0]
                if fit_slices_start <= 0:
                    # neg, adjust relative to len(modelspec)
                    slice_at = len(modelspec)+fit_slices_start
                else:
                    slice_at = fit_slices_start
                context['freeze_layers'] = list(range(slice_at))
                for slice_idx in range(output_count):
                    log.info("Fitting: %s, fit %d/%d, fold %d/%d SLICE %d/%d",
                             fit_function,
                             fit_idx + 1, modelspec.fit_count,
                             jack_idx + 1, modelspec.jack_count,
                             slice_idx, output_count)

                    slice_channels = np.array([slice_idx])
                    modelspec_sliced, est_sliced = init.modelspec_slice_output_layers(modelspec, est, slice_channels, slice_at=slice_at)
                    results = fn(modelspec=modelspec_sliced, est=est_sliced, **context)
                    modelspec = init.modelspec_restore_sliced_output(
                        results['modelspec'], est_sliced, slice_channels,
                        slice_at, modelspec, est)
                results['modelspec'] = modelspec
            else:
                results = fn(modelspec=modelspec, est=e, **context)

                # compatible with direct modelspec return or xforms-ese dictionary
                if type(results) is dict:
                    if 'modelspec' in results.keys():
                        modelspec = results['modelspec']
                else:
                    modelspec = results
                    results = {'modelspec', modelspec}

    return results


def fit_nfold(modelspecs, est, tolerance=1e-7, max_iter=1000,
              IsReload=False, metric='nmse', fitter='scipy_minimize',
              analysis='fit_basic', tolerances=None, module_sets=None,
              tol_iter=100, fit_iter=20, output_name='resp', **context):
    ''' fitting n fold, one from each entry in est '''
    raise Warning("DEPRECATED?")
    if not IsReload:
        metric = lambda d: getattr(metrics, metric)(d, 'pred', 'resp')
        fitter_fn = getattr(nems0.fitters.api, fitter)
        fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}
        if fitter == 'coordinate_descent':
            fit_kwargs['step_size'] = 0.1
        modelspecs = analysis.api.fit_nfold(
                est, modelspecs, fitter=fitter_fn,
                fit_kwargs=fit_kwargs, analysis=analysis,
                tolerances=tolerances, module_sets=module_sets,
                tol_iter=tol_iter, fit_iter=fit_iter)

    return {'modelspecs': modelspecs}


def fit_n_times_from_random_starts(modelspecs, est, ntimes, subset,
                                   analysis='fit_basic', basic_kwargs={},
                                   IsReload=False, **context):
    ''' Self explanatory. '''
    raise Warning ('This is deprecated. Replaced by set_random_phi and analysis.test_prediction.pick_best_phi')
    if not IsReload:
        if len(modelspecs) > 1:
            raise NotImplementedError('I only work on 1 modelspec')

        modelspecs = analysis.api.fit_from_priors(
                est, modelspecs[0], ntimes=ntimes, subset=subset,
                analysis=analysis, basic_kwargs=basic_kwargs
                )

    return {'modelspecs': modelspecs}


###############################################################################
########################         SAVE / SUMMARY        ########################
###############################################################################


def save_recordings(modelspec, est, val, **context):
    # TODO: Save the recordings somehow?
    return {'modelspec': modelspec}


def predict(modelspec, est, val, est_list=None, val_list=None, jackknifed_fit=False,
            use_mask=True, **context):
    # TODO: Add statistics to metadata of every modelspec
    if (val_list is None):
        est, val = analysis.api.generate_prediction(est, val, modelspec, jackknifed_fit=jackknifed_fit, use_mask=use_mask)
        modelspec.recording = val
        return {'val': val, 'est': est, 'modelspec': modelspec}
    else:
        for cellidx,est,val in zip(range(len(est_list)),est_list,val_list):
            modelspec.set_cell(cellidx)
            est, val = analysis.api.generate_prediction(est, val, modelspec, jackknifed_fit=jackknifed_fit, use_mask=use_mask)
            modelspec.recording = val
            est_list[cellidx] = est
            val_list[cellidx] = val
        modelspec.set_cell(0)
        return {'val': val_list[0], 'est': est_list[0], 'est_list': est_list, 'val_list': val_list, 'modelspec': modelspec}


def add_summary_statistics(est, val, modelspec, est_list=None, val_list=None, rec_list=None, fn='standard_correlation',
                           rec=None, use_mask=True, eval_binaural=False, IsReload=False, **context):
    '''
    standard_correlation: average all correlation metrics and add
                          to first modelspec only.
    correlation_per_model: evaluate correlation metrics separately for each
                           modelspec and save results in each modelspec
    '''
    if IsReload:
        return {}

    corr_fn = getattr(analysis.api, fn)
    model_engine = modelspec.meta.get('engine', 'nems0')
    if est_list is None:
        est_list=[est]
        val_list=[val]
        rec_list=[rec]

    for cellidx,est,val,rec in zip(range(len(est_list)),est_list,val_list,rec_list):
        if len(est_list)>1:
            modelspec.set_cell(cellidx)
            log.info(f'cell_index: {cellidx}')
        modelspec = corr_fn(est, val, modelspec=modelspec, rec=rec, use_mask=use_mask)

        if eval_binaural:
            val_epochs = ep.epoch_names_matching(val['resp'].epochs, "^STIM_")
            mono_epochs = [e for e in val_epochs if 'NULL' in e]
            bin_epochs = [e for e in val_epochs if 'NULL' not in e]
            val = val.create_mask(mono_epochs, mask_name='mono_mask')
            val = val.create_mask(bin_epochs, mask_name='bin_mask')
            modelspec_mono = modelspec.copy()
            # don't pass rec so set to None and skip r_ceiling calc for subset evals
            modelspec_mono = corr_fn(est, val, modelspec=modelspec_mono, use_mask='mono_mask')
            modelspec_bin = modelspec.copy()
            modelspec_bin = corr_fn(est, val, modelspec=modelspec_bin, use_mask='bin_mask')
            modelspec.meta['r_test_mono'] = modelspec_mono.meta['r_test']
            modelspec.meta['r_test_bin'] = modelspec_bin.meta['r_test']

        if model_engine == 'nems-lite':
            pass
        elif find_module('lv_norm', modelspec) is not None:
            log.info('add_summary_statistics: lv_norm model, skipping all MI calculations')
            
        elif find_module('state', modelspec) is not None:
            if ('state_raw' not in val.signals.keys()):
                pass
            else:
                log.info('Skipping jackknife MI calculations')
                s = metrics.state_mod_index(val, epoch='REFERENCE', psth_name='pred',
                                            state_sig='state_raw', state_chan=[])
                j_s, ee = metrics.j_state_mod_index(val, epoch='REFERENCE', psth_name='pred',
                                                    state_sig='state_raw', state_chan=[], njacks=10)
                #j_s = s
                #ee = np.zeros(j_s.shape)
                modelspec.meta['state_mod'] = s
                modelspec.meta['j_state_mod'] = j_s
                modelspec.meta['se_state_mod'] = ee
                modelspec.meta['state_chans'] = val['state'].chans

                # adding gain / DC mod_indexes (if the corresponding signals exist -- new sdexp models)
                if 'gain' in val.signals.keys():
                    log.info('Saving gain/DC MI')
                    g = metrics.state_mod_index(val, epoch='REFERENCE', psth_name='gain', divisor=1,
                                    state_sig='state_raw', state_chan=[])
                    dc = metrics.state_mod_index(val, epoch='REFERENCE', psth_name='dc', divisor=1,
                                    state_sig='state_raw', state_chan=[])
                    modelspec.meta['state_mod_gain'] = g
                    modelspec.meta['state_mod_dc'] = dc

                # try using resp
                s = metrics.state_mod_index(val, epoch='REFERENCE', psth_name='resp',
                                    state_sig='state_raw', state_chan=[])
                #j_s, ee = metrics.j_state_mod_index(val, epoch='REFERENCE', psth_name='resp',
                #                    state_sig='state_raw', state_chan=[], njacks=10)
                j_s = s
                ee = np.zeros(j_s.shape)
                modelspec.meta['state_mod_r'] = s
                modelspec.meta['j_state_mod_r'] = j_s
                modelspec.meta['se_state_mod_r'] = ee

                # try using the "mod" signal (if it exists) which is calculated
                if ('mod' in modelspec.meta['modelname']) and ('mod' in val.signals.keys()):
                    s = metrics.state_mod_index(val, epoch='REFERENCE',
                                                    psth_name='mod', divisor='resp',
                                                    state_sig='state_raw', state_chan=[])
                    #j_s, ee = metrics.j_state_mod_index(val, epoch='REFERENCE',
                    #                                psth_name='mod', divisor='resp',
                    #                                state_sig='state_raw', state_chan=[],
                    #                                njacks=10)
                    j_s = s
                    ee = np.zeros(j_s.shape)

                    modelspec.meta['state_mod_m'] = s
                    modelspec.meta['j_state_mod_m'] = j_s
                    modelspec.meta['se_state_mod_m'] = ee

    # figure out spont rate storing in modelspec.meta
    resp = rec['resp']
    try:
        prestimsilence = resp.extract_epoch('PreStimSilence', mask=rec['mask'])
        if prestimsilence.shape[-1] > 0:
            if len(prestimsilence.shape) == 3:
                spont_rate = np.nanmean(prestimsilence, axis=(0, 2))
            else:
                spont_rate = np.nanmean(prestimsilence)
        elif prestimsilence.shape[0]>0:
            log.info('Zero prestimsilence?')
            spont_rate=0
        else:
            try:
                prestimsilence = resp.extract_epoch('TRIALPreStimSilence')
                if len(prestimsilence.shape) == 3:
                    spont_rate = np.nanmean(prestimsilence, axis=(0, 2))
                else:
                    spont_rate = np.nanmean(prestimsilence)
            except:
                raise ValueError("Can't find prestim silence to use for PSTH calculation")
    except:
        # special case where the epochs included in mask don't have PreStimSilence,
        # so we get it elsewhere. Designed for CPN data...
        try:
            prestimsilence = resp.extract_epoch('TRIALPreStimSilence')
            if len(prestimsilence.shape) == 3:
                spont_rate = np.nanmean(prestimsilence, axis=(0, 2))
            else:
                spont_rate = np.nanmean(prestimsilence)
        except:
            raise ValueError("Can't find prestim silence to use for PSTH calculation")
    r=resp.extract_epoch('REFERENCE')
    evoked_mean = r.mean(axis=2).mean(axis=0)-spont_rate

    modelspec.meta['spont_mean']=spont_rate
    modelspec.meta['evoked_mean']=evoked_mean
    if model_engine == 'nems0':
        modelspec.set_cell(0)
    log.info(f"r_test={modelspec.meta['r_test']}")
    return {'modelspec': modelspec}


def add_summary_statistics_by_condition(est, val, modelspec, evaluation_conditions, rec=None,
                                        use_mask=True, **context):
    modelspec = analysis.api.standard_correlation_by_epochs(est,val,modelspec=modelspec,
            epochs_list=evaluation_conditions,rec=rec, use_mask=use_mask)
    return {'modelspec': modelspec}

def plot_summary(modelspec, val, IsReload=False,
                 figures_to_load=None, time_range = None, **context):
    # CANNOT initialize figures=[] in optional args our you will create a bug

    figures = context.get('figures', [])
    if IsReload:
        if figures_to_load is not None:
            figures.extend([load_resource(f) for f in figures_to_load])

    elif modelspec.meta.get('fitter', 'basic')=='ccnorm':
        from nems0.plots.state import cc_comp
        return cc_comp(modelspec=modelspec, val=val, **context)
    else:
        fig = modelspec.quickplot(time_range=time_range)
        # Needed to make into a Bytes because you can't deepcopy figures!
        figures.append(nplt.fig2BytesIO(fig))

    return {'figures': figures}


###############################################################################
########################            FLAGS              ########################
###############################################################################


def use_metric(metric='nmse_shrink', IsReload=False, **context):
    if not IsReload:
        return {'metric': metric}
    else:
        return {}


def jackknifed_fit(IsReload=False, **context):
    if not IsReload:
        return {'jackknifed_fit': True}
    else:
        return {}

def fast_eval(modelspec, est, **context):
    modelspec.fast_eval_on(est)

    return {'modelspec': modelspec}


def random_sample_fit(ntimes=10, subset=None, IsReload=False, **context):
    if not IsReload:
        return {'random_sample_fit': True, 'n_random_samples': ntimes,
                'random_fit_subset': subset}
    else:
        return {}


# TODO: Perturb around the modelspec to get confidence intervals

# TODO: Use simulated annealing (Slow, arguably gets stuck less often)
# modelspecs = analysis.fit_basic(est, modelspec,
#                                   fitter=nems0.fitter.annealing)

# TODO: Use Metropolis algorithm (Very slow, gives confidence interval)
# modelspecs = analysis.fit_basic(est, modelspec,
#                                   fitter=nems0.fitter.metropolis)

# TODO: Use 10-fold cross-validated evaluation
# fitter = partial(nems0.cross_validator.cross_validate_wrapper, gradient_descent, 10)
# modelspecs = analysis.fit_cv(est, modelspec, folds=10)


###############################################################################
##################        XFORMS UTILITIES             ########################
###############################################################################


def load_xform(uri):
    """
    Loads and returns xform saved as a JSON.
    """
    xform = load_resource(uri)
    return xform


def xfspec_shortname(xformspec):
    """
    Given an xformspec, makes a shortname for it.
    """
    n = len('nems0.xforms.')
    fn_names = [xf[n:] for xf, xfa in xformspec]
    name = ".".join(fn_names)
    return name


def tree_path(recording, modelspecs, xfspec):
    '''
    Returns a relative path (excluding filename, host, port) for URIs.
    Editing this function edits the path in the file tree of every
    file saved!
    '''

    xformname = xfspec_shortname(xfspec)
    modelname = get_modelspec_shortname(modelspecs[0])
    recname = recording.name  # Or from rec.uri???
    meta = get_modelspec_metadata(modelspecs[0])
    date = meta.get('date', iso8601_datestring())

    path = '/' + recname + '/' + modelname + '/' + xformname + '/' + date + '/'

    return path


def save_analysis(destination, recording, modelspec, xfspec=[], figures=[],
                  log="", add_tree_path=False, update_meta=True, save_rec=False):
    '''Save an analysis file collection to a particular destination.'''
    if add_tree_path:
        treepath = tree_path(recording, [modelspec], xfspec)
        base_uri = os.path.join(destination, treepath)
    else:
        base_uri = destination

    if destination is None:
        destination = get_default_savepath(modelspec)
        base_uri = destination

    if update_meta:
        modelspec.meta['modelpath'] = base_uri
        modelspec.meta['figurefile'] = os.path.join(base_uri,'figure.0000.png')

    base_uri = base_uri if base_uri[-1] == '/' else base_uri + '/'
    xfspec_uri = base_uri + 'xfspec.json'  # For attaching to modelspecs

    number=0
    for cc in range(modelspec.cell_count):
        for ff in range(modelspec.fit_count):
            for jj in range(modelspec.jack_count):
                m = modelspec.copy()
                m.jack_index = jj
                m.fit_index = ff
                m.cell_index = cc
                set_modelspec_metadata(m, 'xfspec', xfspec_uri)
                set_modelspec_metadata(m, 'shape', modelspec.raw.shape)
                save_resource(base_uri + 'modelspec.{:04d}.json'.format(number), json=m[:])
                number+=1
    for number, figure in enumerate(figures):
        fig_uri = base_uri + 'figure.{:04d}.png'.format(number)
        #log.info('saving figure %d to %s', number, fig_uri)
        save_resource(fig_uri, data=figure)
    save_resource(base_uri + 'log.txt', data=log)
    save_resource(xfspec_uri, json=xfspec)

    if save_rec:
        # TODO: copy actual recording file
        rec_uri = base_uri + 'recording.tgz'
        recording.save(rec_uri)

    return {'savepath': base_uri}


def save_from_context(destination, xfspec, ctx, log_xf):
    '''
    As save_analysis, but accepts uses xforms context in place of
    modelspec, recording, figures and log.
    IS THIS EVER USED?
    '''

    modelspec = context['modelspec']
    log = context['log']
    figures = context['figures']
    xfspec_uri = os.path.join(destination, 'xfspec.json')
    log_uri = os.path.join(destination, 'log.txt')

    for number in range(modelspec.jack_count):
        m = modelspec.copy()
        m.jack_index = number
        set_modelspec_metadata(m, 'xfspec', xfspec_uri)
        save_resource(os.path.join(destination, 'modelspec.{:04d}.json'.format(number)), json=m[:])
    for number, figure in enumerate(figures):
        save_resource(os.path.join(destination, 'figure.{:04d}.png'.format(number)), data=figure)
    save_resource(log_uri, data=log)
    save_resource(xfspec_uri, json=xfspec)

    return {'savepath': destination}


def save_context(destination, ctx, xfspec=[]):
    '''
    Save entire context
    :param destination:
    :param ctx: NEMS context dictionary
    :param xfspec: options xfspec list (default [])
    :return:
    '''

    save_analysis(destination, recording=ctx['rec'],
                  modelspec=ctx['modelspec'], xfspec=xfspec,
                  figures=ctx.get('figures'), log=ctx.get('log', ''))

    _ctx=ctx.copy()
    del _ctx['modelspec']
    del _ctx['figures']
    klist = list(_ctx.keys())
    for k in klist:
        v=_ctx[k]
        if type(v) is Recording:
            print(k, "RECORDING")
            v.save(os.path.join(destination, k + '.tgz'))
            del _ctx[k]

    ctx_uri = os.path.join(destination, 'ctx.json')
    save_resource(ctx_uri, json=_ctx)

    return {'savepath': destination}


def load_context(filepath):
    """
    recreate saved context without any model eval, no xforms required
    :param filepath:
    :return:
    """
    ctx_uri = os.path.join(filepath, 'ctx.json')
    ctx = load_resource(ctx_uri)

    xfspec, _ctx = load_analysis(filepath, eval_model=False)
    ctx.update(_ctx)

    for tgz in glob.glob(os.path.join(filepath, '*.tgz')):
        k, ext = os.path.splitext((os.path.basename(tgz)))
        ctx[k] = load_recording(tgz)

    # hard-coded. Could this be implemented less kludgily?
    ctx['modelspec'].recording = ctx['val']

    return xfspec, ctx


def load_analysis(filepath, eval_model=True, only=None, verbose=True):
    """
    load xforms spec and context dictionary for a model fit
    :param filepath: URI of saved xforms model
    :param eval_model: if True, re-evaluates all steps. Time consuming but gives an exact copy of the
    original context
    :param only: either an int, usually 0, to evaluate the first step of loading a recording, or a slice
    object, which gives more flexibility over what steps of the original xfspecs to run again.
    :return: (xfspec, ctx) tuple
    """
    if verbose:
        log.info('Loading xfspec and context from %s ...', filepath)
        
    def _path_join(*args):
        if os.name == 'nt':
            # deal with problems on Windows OS
            path = "/".join(args)
        else:
            path = os.path.join(*args)
        return path

    xfspec = load_xform(_path_join(filepath, 'xfspec.json'))

    is_nems_lite = any(['init_nems_keywords' in x[0] for x in xfspec])

    mspaths = []
    mslistpaths = []
    figures_to_load = []
    logstring = ''
    for file in os.listdir(filepath):
        #log.info(file)
        if file.startswith("modelspec_list"):
            mslistpaths.append(_path_join(filepath, file))
        elif file.startswith("modelspec"):
            mspaths.append(_path_join(filepath, file))
        elif file.startswith("figure"):
            figures_to_load.append(_path_join(filepath, file))
        elif file.startswith("log"):
            logpath = _path_join(filepath, file)
            with open(logpath) as logfile:
                logstring = logfile.read()
    mspaths.sort()  # make sure we're in alphanumeric order!
    mslistpaths.sort()  # make sure we're in alphanumeric order!

    if is_nems_lite:
        from nems.tools.json import load_model
        ctx = {'modelspec': load_model(mspaths[0])}
        if len(mslistpaths) > 0:
            #for m in mslistpaths:
            #    print(m)
            ctx['modelspec_list'] = [load_model(m) for m in mslistpaths]
    else:
        ctx = load_modelspecs([], uris=mspaths, IsReload=False)
    ctx['IsReload'] = True
    ctx['figures_to_load'] = figures_to_load
    ctx['log'] = logstring

    if 'recording.tgz' in os.listdir(filepath):
        # TODO: change rec path in modelspec
        pass

    if eval_model:
        ctx, log_xf = evaluate(xfspec, ctx, verbose=verbose)
    elif only is not None:
        # Useful for just loading the recording without doing
        # any subsequent evaluation.
        if isinstance(only, int):
            ctx, log_xf = evaluate([xfspec[only]], ctx)
        elif isinstance(only, slice):
            ctx, log_xf = evaluate(xfspec[only], ctx)

    return xfspec, ctx


def regenerate_figures(batch, modelnames, cellids=None):
    '''
    Regenerate quickplot figures for a given modelname and batch.
    Intended to be used in cases where plot code has been updated and
    needs to be included in existing models, but a re-fit is not necessary.

    '''
    if cellids is None:
        cells = nd.get_batch_cells(batch)['cellid'].values.tolist()
    else:
        cells = cellids
    r = nd.get_results_file(batch, modelnames, cellids=cellids)

    for m in modelnames:
        for c in cells:
            try:
                filepath = r[r.cellid == c][r.modelname == m]['modelpath'].iat[0] + '/'
            except IndexError:
                # result doesn't exist
                continue
            xfspec, ctx = load_analysis(filepath, eval_model=True)
            fig = ctx['modlespec'].quickplot()
            fig_bytes = nplt.fig2BytesIO(fig)
            plt.close('all')
            save_resource(filepath + 'figure.0000.png', data=fig_bytes)
            # TODO: also have to tell


###############################################################################
##################        CONTEXT UTILITIES             #######################
###############################################################################


def evaluate_context(ctx, rec_key='val', rec_idx=0, mspec_idx=0, start=None,
                     stop=None):
    rec = ctx[rec_key][0]
    mspec = ctx['modelspec']
    mspec.fit_index = mspec_idx
    return ms.evaluate(rec, mspec, start=start, stop=stop)


def get_meta(ctx, mspec_idx=0, mod_idx=0):
    if (mspec_idx > 0) or (mod_idx >0):
        raise ValueError("meta indexing not supported")
    return ctx['modelspec'].meta


def get_modelspec(ctx, mspec_idx=0):
    m = ctx['modelspec'].copy()
    m.fit_index = mspec_idx
    return m


def get_module(ctx, val, key='index', mspec_idx=0, find_all_matches=False):
    mspec = ctx['modelspec']
    mspec.fit_index = mspec_idx
    if key in ['index', 'idx', 'i']:
        return mspec[val]
    else:
        i = find_module(val, mspec, find_all_matches=find_all_matches, key=key)
        return mspec[i]


def plot_heatmap(ctx, signal_name, cutoff=None, rec_key='val', rec_idx=0,
                 mspec_idx=0, start=None, stop=None):

    array = get_signal_as_array(ctx, signal_name, cutoff, rec_key, rec_idx,
                                mspec_idx, start, stop)
    array = array[:, ~np.all(np.isnan(array), axis=0)]
    plt.imshow(array, aspect='auto')


def plot_timeseries(ctx, signal_name, cutoff=None, rec_key='val', rec_idx=0,
                    mspec_idx=0, start=None, stop=None):

    array = get_signal_as_array(ctx, signal_name, cutoff, rec_key, rec_idx,
                                mspec_idx, start, stop)
    plt.plot(array.T)


def get_signal_as_array(ctx, signal_name, cutoff=None, rec_key='val',
                        rec_idx=0, mspec_idx=0, start=None, stop=None):

    rec = evaluate_context(ctx, rec_key=rec_key, rec_idx=rec_idx,
                           mspec_idx=mspec_idx, start=start, stop=stop)

    array = rec[signal_name].as_continuous()
    if cutoff is not None:
        if isinstance(cutoff, int):
            array = array[:, :cutoff]
        elif isinstance(cutoff, tuple):
            array = array[:, cutoff[0]:cutoff[1]]
        else:
            raise ValueError("cutoff must be an integer (:cutoff)"
                             "or a tuple (cutoff[0]:cutoff[1])")

    return array


###############################################################################
########################          UNUSED?        ##############################
###############################################################################


def fit_random_subsets(modelspecs, est, nsplits,
                       IsReload=False, **context):
    ''' Randomly sample parts of the data? Wait, HOW DOES THIS WORK? TODO?'''
    if not IsReload:
        if len(modelspecs) > 1:
            raise ValueError('I only work on 1 modelspec')
        modelspecs = analysis.api.fit_random_subsets(est,
                                                          modelspecs[0],
                                                          nsplits=nsplits)
    return {'modelspecs': modelspecs}


def fit_equal_subsets(modelspecs, est, nsplits,
                      IsReload=False, **context):
    ''' Divide the data into nsplits equal pieces and fit each one.'''
    if not IsReload:
        if len(modelspecs) > 1:
            raise ValueError('I only work on 1 modelspec')
        modelspecs = analysis.api.fit_subsets(est,
                                                   modelspecs[0],
                                                   nsplits=nsplits)
    return {'modelspecs': modelspecs}


def fit_jackknifes(modelspecs, est, njacks,
                   IsReload=False, **context):
    ''' Jackknife the data, fit on those, and make predictions from those.'''
    if not IsReload:
        if len(modelspecs) > 1:
            raise ValueError('I only work on 1 modelspec')
        modelspecs = analysis.api.fit_jackknifes(est,
                                                      modelspecs[0],
                                                      njacks=njacks)
    return {'modelspecs': modelspecs}


def fit_module_sets(modelspecs, est, max_iter=1000, IsReload=False,
                    module_sets=None, invert=False, tolerance=1e-4,
                    fitter=scipy_minimize, fit_kwargs={}, **context):

    if not IsReload:
        if len(modelspecs) > 1:
            raise NotImplementedError("Not supported for multiple modelspecs")
        modelspecs = [
                analysis.api.fit_module_sets(
                        est, modelspec, fit_kwargs=fit_kwargs,
                        fitter=fitter, module_sets=module_sets,
                        invert=invert, tolerance=tolerance,
                        max_iter=max_iter)[0]
                for modelspec in modelspecs
                ]
    return {'modelspecs': modelspecs}


def split_at_time(rec, valfrac=0.1, rec_list=None, **context):

    if rec_list is None:
        rec_list = [rec]
        return_reclist = False
    else:
        rec=rec_list[0]
        return_reclist = True

    est_list = []
    val_list = []

    for rec in rec_list:
        rec['resp'] = rec['resp'].rasterize()
        rec['stim'] = rec['stim'].rasterize()
        est, val = rec.split_at_time(fraction=valfrac)
        est_list.append(est)
        val_list.append(val)

    if return_reclist:
        return {'est': est_list[0], 'val': val_list[0], 'est_list': est_list, 'val_list': val_list}
    else:
        return {'est': est, 'val': val}
