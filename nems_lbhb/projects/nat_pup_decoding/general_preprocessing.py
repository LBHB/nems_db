import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import nems.xforms as xforms
import nems.db as nd
import logging
from nems.recording import Recording
from nems_lbhb.preprocessing import create_pupil_mask
import sys
import os
import scipy.fftpack as fp
import scipy.signal as ss
import copy
from nems_lbhb.preprocessing import create_pupil_mask

log = logging.getLogger(__name__)


def generate_state_corrected_psth(batch=None, modelname=None, cellids=None, siteid=None, movement_mask=False,
                                        gain_only=False, dc_only=False, cache_path=None, recache=False):
    """
    Modifies the exisiting recording so that psth signal is the prediction specified
    by the modelname. Designed with stategain models in mind. CRH.

    If the model doesn't exist already in /auto/users/hellerc/results/, this
    will go ahead and fit the model and save it in /auto/users/hellerc/results.

    If the fit dir (from xforms) exists, simply reload the result and call this psth.
    """
    if siteid is None:
        raise ValueError("must specify siteid!")
    if cache_path is not None:
        fn = cache_path + siteid + '_{}.tgz'.format(modelname.split('.')[1])
        if gain_only:
            fn = fn.replace('.tgz', '_gonly.tgz')
        if 'mvm' in modelname:
            fn = fn.replace('.tgz', '_mvm.tgz')
        if (os.path.isfile(fn)) & (recache == False):
            rec = Recording.load(fn)
            return rec
        else:
            # do the rest of the code
            pass

    if batch is None or modelname is None:
        raise ValueError('Must specify batch and modelname!')
    results_table = nd.get_results_file(batch, modelnames=[modelname])
    preds = []
    ms = []
    for cell in cellids:
        log.info(cell)
        try:
            p = results_table[results_table['cellid']==cell]['modelpath'].values[0]
            if os.path.isdir(p):
                xfspec, ctx = xforms.load_analysis(p)
                preds.append(ctx['val'])
                ms.append(ctx['modelspec'])
            else:
                sys.exit('Fit for {0} does not exist'.format(cell))
        except:
            log.info("WARNING: fit doesn't exist for cell {0}".format(cell))

    # Need to add a check to make sure that the preds are the same length (if
    # multiple cellids). This could be violated if one cell for example existed
    # in a prepassive run but the other didn't and so they were fit differently
    file_epochs = []

    for pr in preds:
        file_epochs += [ep for ep in pr.epochs.name if ep.startswith('FILE')]

    unique_files = np.unique(file_epochs)
    shared_files = []
    for f in unique_files:
        if np.sum([1 for file in file_epochs if file == f]) == len(preds):
            shared_files.append(str(f))
        else:
            # this rawid didn't span all cells at the requested site
            pass

    # mask all file epochs for all preds with the shared file epochs
    # and adjust epochs
    if (int(batch) == 307) | (int(batch) == 294):
        for i, p in enumerate(preds):
            preds[i] = p.and_mask(shared_files)
            preds[i] = preds[i].apply_mask(reset_epochs=True)

    sigs = {}
    for i, p in enumerate(preds):
        if gain_only:
            # update phi
            mspec = ms[i]
            not_gain_keys = [k for k in mspec[0]['phi'].keys() if '_g' not in k]
            for k in not_gain_keys:
                mspec[0]['phi'][k] = np.append(mspec[0]['phi'][k][0, 0], np.zeros(mspec[0]['phi'][k].shape[-1]-1))[np.newaxis, :]
            pred = mspec.evaluate(p)['pred']
        elif dc_only:
            mspec = ms[i]
            not_dc_keys = [k for k in mspec[0]['phi'].keys() if '_d' not in k]
            for k in not_dc_keys:
                mspec[0]['phi'][k] = np.append(mspec[0]['phi'][k][0, 0], np.zeros(mspec[0]['phi'][k].shape[-1]-1))[np.newaxis, :]
            pred = mspec.evaluate(p)['pred']
        else:
            pred = p['pred'] 
        if i == 0:           
            new_psth_sp = p['psth_sp']
            new_psth = pred
            new_resp = p['resp'].rasterize()

        else:
            try:
                new_psth_sp = new_psth_sp.concatenate_channels([new_psth_sp, p['psth_sp']])
                new_psth = new_psth.concatenate_channels([new_psth, pred])
                new_resp = new_resp.concatenate_channels([new_resp, p['resp'].rasterize()])
            except ValueError:
                import pdb; pdb.set_trace()

    new_pup = preds[0]['pupil']
    sigs['pupil'] = new_pup

    if 'pupil_raw' in preds[0].signals.keys():
        sigs['pupil_raw'] = preds[0]['pupil_raw']

    if 'mask' in preds[0].signals:
        new_mask = preds[0]['mask']
        sigs['mask'] = new_mask
    else:
        mask_rec = preds[0].create_mask(True)
        new_mask = mask_rec['mask']
        sigs['mask'] = new_mask

    if 'rem' in preds[0].signals.keys():
        rem = preds[0]['rem']
        sigs['rem'] = rem

    if 'pupil_eyespeed' in preds[0].signals.keys():
        new_eyespeed = preds[0]['pupil_eyespeed']
        sigs['pupil_eyespeed'] = new_eyespeed

    new_psth_sp.name = 'psth_sp'
    new_psth.name = 'psth'
    new_resp.name = 'resp'
    sigs['psth_sp'] = new_psth_sp
    sigs['psth'] = new_psth
    sigs['resp'] = new_resp

    new_rec = Recording(sigs, meta=preds[0].meta)

    # make sure mask is cast to bool
    new_rec['mask'] = new_rec['mask']._modified_copy(new_rec['mask']._data.astype(bool))

    if cache_path is not None:
        log.info('caching {}'.format(fn))
        new_rec.save_targz(fn)

    return new_rec


def regress_pupil(rec):
    """
    ** pupil specific function - used for batch 289**
    ** `regress_state` is an equivalent function, but allows you to fit multiple to regression
    to > 1 state variabel and choose which to regress out **

    Remove first-order effect on the trial-trial variability for each neuron

    1) Compute compute psth in each bin by averaging over the whole recording
    2) At each time bin, model residuals as linear function of pupil
    3) Subtract off prediction
    4) Add the corrected residuals back to the overall psth
    """
    r = copy.deepcopy(rec)
    ep = np.unique([ep for ep in r.epochs.name if 'STIM_00' in ep]).tolist()

    r_st = r['resp'].extract_epochs(ep)
    p_st = r['pupil'].extract_epochs(ep)
    r_psth = r_st.copy()
    r_new = r_st.copy()
    for e in ep:
        m = r_st[e].mean(axis=0)
        r_psth[e] = np.tile(m, [r_st[e].shape[0], 1, 1])
        # for each stim bin
        for b in range(r_st[e].shape[-1]):
            # for each neuron, regress out pupil effects
            for n in range(r_psth[e].shape[1]):
                X = p_st[e][:, :, b].squeeze()
                y = r_st[e][:, n, b] - r_psth[e][:, n, b]
                reg = LinearRegression()
                reg.fit(X[:, np.newaxis], y[:, np.newaxis])
                y_pred = reg.predict(X[:, np.newaxis])
                y_new_residual = y - y_pred.squeeze()
                r_new[e][:, n, b] = r_psth[e][:, n, b] + y_new_residual

    r['resp'] = r['resp'].replace_epochs(r_new)

    return r

def regress_state(rec, state_sigs=['behavior', 'pupil'], regress=None):
    """
    Remove first order effects of given state variable(s). Idea is to model all state
    variables, for example, pupil and behavior, then just choose to remove the first order
    effects of one or the other, or both.

    1) Compute compute psth in each bin by averaging over the whole recording
    2) At each time bin, model residuals as linear function of state variable(s)
    3) Subtract off prediction computed using the coef(s) for the "regress" states
    4) Add the corrected residuals back to the overall psth
    """
    if regress is not None:
        log.info(DeprecationWarning('regress argument is deprecated. Always regress out all state signals'))

    r = copy.deepcopy(rec)
    ep = np.unique([ep for ep in r.epochs.name if ('STIM' in ep) | ('TAR_' in ep)]).tolist()
    
    r_st = r['resp'].extract_epochs(ep)
    state_signals = dict.fromkeys(state_sigs)
    for s in state_sigs:
        if s == 'pupil':
            state_signals[s] = r['pupil'].extract_epochs(ep)
        elif s == 'behavior':
            r_beh_mask = r.create_mask(True)
            r_beh_mask = r_beh_mask.and_mask(['ACTIVE_EXPERIMENT'])
            state_signals[s] = r_beh_mask['mask'].extract_epochs(ep)
        elif s == 'lv':
            state_signals[s] = r['lv'].extract_epochs(ep)
        else:
            raise ValueError("No case set up for {}".format(s))

    r_psth = r_st.copy()
    r_new = r_st.copy()
    for e in ep:
        m = r_st[e].mean(axis=0)
        r_psth[e] = np.tile(m, [r_st[e].shape[0], 1, 1])
        # for each stim bin
        for b in range(r_st[e].shape[-1]):
            # for each neuron, regress out state effects
            for n in range(r_psth[e].shape[1]):
                for i, s in enumerate(state_sigs):
                    if i == 0:
                        X = state_signals[s][e][:, :, b]
                    else:
                        X = np.concatenate((X, state_signals[s][e][:, :, b]), axis=-1)

                y = r_st[e][:, n, b] - r_psth[e][:, n, b]
                reg = LinearRegression()

                # zscore if std of state signal not 0
                X = X - X.mean(axis=0)
                nonzero_sigs = np.argwhere(X.std(axis=0)!=0).squeeze()
                if nonzero_sigs.shape != (0,):
                    X = X[:, nonzero_sigs] / X[:, nonzero_sigs].std(axis=0)
                    if len(X.shape) == 1:
                        X = X[:, np.newaxis]
                    if (np.any(np.isnan(y)) | np.any(np.isnan(X))):
                        log.info(f"Found nans in data for bin {b}, epoch: {e}. Not regressing out state")
                        model_coefs = np.zeros(X.shape[-1])
                        intercept = 0
                    else:
                        reg.fit(X, y[:, np.newaxis])
                        model_coefs = reg.coef_
                        intercept = reg.intercept_
                    y_pred = np.matmul(X, model_coefs.T) + intercept
                    y_new_residual = y - y_pred.squeeze()
                    r_new[e][:, n, b] = r_psth[e][:, n, b] + y_new_residual
                else:
                    # state signal has 0 std so nothing to regress out
                    r_new[e][:, n, b] = r_psth[e][:, n, b] + y                

    r['resp'] = r['resp'].replace_epochs(r_new)

    return r


def regress_state2(rec, state_sigs=['behavior', 'pupil'], regress=['pupil']):
    """
    Same as the above function (and will likely replace it), except that now, instead of fitting
    for each stimulus independently, fit the residuals of all the data. I think this is important
    in order to fully sample state.
    """
    r = rec.copy()
    r = generate_psth(r)

    # get state signals
    state_signals = dict.fromkeys(state_sigs)
    for s in state_sigs:
        if s == 'pupil':
            state_signals[s] = r['pupil']._data
        elif s == 'behavior':
            r_beh_mask = r.create_mask(True)
            r_beh_mask = r_beh_mask.and_mask(['ACTIVE_EXPERIMENT'])
            state_signals[s] = r_beh_mask['mask']._data
        elif s == 'lv':
            state_signals[s] = r['lv']._data
        else:
            raise ValueError("No case set up for {}".format(s))

    # build state matrix
    for i, s in enumerate(state_sigs):
        if i == 0:
            X = state_signals[s]
        else:
            X = np.concatenate((X, state_signals[s]), axis=0).T

    r_new = r['resp']._data.copy()
    for n in range(r_new.shape[0]):
        # fit regression model for each neuron
        y = r['resp']._data[n, :] - r['psth']._data[n, :]
        reg = LinearRegression()

        X = X - X.mean(axis=0)
        X = X / X.std(axis=0)

        reg.fit(X, y[:, np.newaxis])

        # compute predictions (of residuals)
        pred = reg.predict(X)

        # compute new residual (regressing out state)
        new_residual = y - pred.T

        # add new residual (w/o state modulation) back to psth
        r_new[n, :] = r['psth']._data[n, :] + new_residual.squeeze()

    r['resp'] = r['resp']._modified_copy(r_new)

    return r


def bandpass_filter_resp(rec, low_c, high_c, fs=None, alpha=None, boxcar=True):
    '''
    Bandpass filter resp. Return new recording with filtered resp.
    '''

    if low_c is None:
        low_c = 0
    if high_c is None:
        high_c = rec['resp'].fs

    if (type(rec) is np.ndarray) & (fs is not None):
        resp_filt = rec.copy()
        resp = rec.copy()
    elif (type(rec) is np.ndarray) & (fs is None):
        raise ValueError("must give sampling rate")
    else:
        newrec = rec.copy()
        fs = rec['resp'].fs
        resp = rec['resp'].rasterize()._data
        resp_filt = resp.copy()
    for n in range(resp.shape[0]):
        s = resp[n, :]
        resp_fft = fp.fft(s)
        w = fp.fftfreq(s.size, 1 / fs)
        inds = np.argwhere((w >= low_c) & (w <= high_c))
        inds2 = np.argwhere((w <= -low_c) & (w >= -high_c))
        m = np.zeros(w.shape)
        if alpha is None:
            alpha = 0.001
        if boxcar:
            alpha = 0
        m[inds] = ss.tukey(len(inds), alpha)[:, np.newaxis]
        m[inds2] = ss.tukey(len(inds2), alpha)[:, np.newaxis]
        resp_cut = resp_fft * m
        resp_filt[n, :] = fp.ifft(resp_cut)

    if type(rec) is np.ndarray:
        return resp_filt

    else:
        newrec['resp'] = newrec['resp']._modified_copy(resp_filt)

        return newrec


def sliding_window(x, fs, window_size, step_size):
    """
    Transform 1 x time vector X into sliding bins of length window_size (in seconds), 
    taking steps of step_size (in seconds). If step_size == window_size, bins are non-overlapping.

    return:
        t: center time of each bin
        Xw: matrix of shape nbins x window_size
    """

    # convert window_size and step_size to bins
    window_size = int(fs * window_size)
    step_size = int(fs * step_size)

    if step_size > window_size:
        raise ValueError("Step size must be smaller than window size")

    nbins = int(x.shape[-1] / step_size)

    Xw = np.zeros((nbins, window_size))
    t = np.zeros((nbins))
    start_idx = 0
    for b in range(nbins):
        end_idx = start_idx + window_size
        _d = x[0, start_idx:end_idx]
        _t = ((end_idx + start_idx) / 2) / fs
        try:
            Xw[b, :] = _d
            t[b] = _t
        except:
            # no more full window_lengths left in x. Break from loop.
            break

        start_idx += step_size

    # truncate non-filled bins of Xw and t
    t = t[:b]
    Xw = Xw[:b, :]

    return t, Xw







def highpass_filter_response(rec, cutoff):
    '''
    will apply mask and filter all data in resp. cutoff in Hz.
    '''
    newrec = rec.copy()
    newrec = newrec.apply_mask(reset_epochs=True)
    resp = newrec['resp'].extract_epoch('REFERENCE')
    fs = rec['resp'].fs
    resp_filt = resp.copy()
    # filter each neuron/trial independently
    for rep in range(resp.shape[0]):
        for n in range(resp.shape[1]):
            s = resp[rep, n, :]
            resp_fft = fp.fft(s)
            w = fp.fftfreq(s.size, 1 / fs)
            inds = np.argwhere((w >= cutoff))
            inds2 = np.argwhere((w <= -cutoff))
            m = np.zeros(w.shape)
            alpha = 0.4
            m[inds] = ss.tukey(len(inds), alpha)[:, np.newaxis]
            m[inds2] = ss.tukey(len(inds2), alpha)[:, np.newaxis]
            resp_cut = resp_fft * m
            resp_filt[rep, n, :] = fp.ifft(resp_cut)

    newrec['resp'] = newrec['resp'].replace_epochs({'REFERENCE': resp_filt})

    return newrec


def zscore_data(rec):
    """
    Z-score data and return new recording with each neuron having mean 0, std 1
    """
    newrec = rec.copy()
    resp = newrec['resp']._data

    u = np.mean(resp, axis=-1)
    resp -= u

    std = np.std(resp, axis=-1)
    resp /= std

    newrec['resp'] = newrec['resp']._modified_copy(resp)

    return newrec

def get_pupil_balanced_epochs(rec, rec_sp=None, rec_bp=None):
    """
    Given big/small pupil recordings return list of
    epochs that are balanced between the two.
    """
    all_epochs = np.unique([str(ep) for ep in rec.epochs.name if 'STIM_00' in ep]).tolist()

    if (rec_sp is None) | (rec_bp is None):
        pup_ops = {'state': 'big', 'epoch': ['REFERENCE'], 'collapse': True}
        rec_bp = create_pupil_mask(rec.copy(), **pup_ops)
        pup_ops['state']='small'
        rec_sp = create_pupil_mask(rec.copy(), **pup_ops)

        rec_bp = rec_bp.apply_mask(reset_epochs=True)
        rec_sp = rec_sp.apply_mask(reset_epochs=True)

    # get rid of pre/post stim silence
    #rec_bp = rec_bp.and_mask(['PostStimSilence'], invert=True)
    #rec_sp = rec_sp.and_mask(['PostStimSilence'], invert=True)
    #rec = rec.and_mask(['PostStimSilence'], invert=True)
    #rec_bp = rec_bp.apply_mask(reset_epochs=True)
    #rec_sp = rec_sp.apply_mask(reset_epochs=True)
    #rec = rec.apply_mask(reset_epochs=True)

    # find pupil matched epochs
    balanced_eps = []
    for ep in all_epochs:
        sp = rec_sp['resp'].extract_epoch(ep).shape[0]
        bp = rec_bp['resp'].extract_epoch(ep).shape[0]
        if len(all_epochs)==3:
            if abs(sp - bp) < 3:
                balanced_eps.append(ep)
        else:
            if sp==bp:
                balanced_eps.append(ep)

    if len(balanced_eps)==0:
        log.info("no balanced epochs at site {}".format(rec.name))

    else:
        log.info("found {0} balanced epochs:".format(len(balanced_eps)))

    return balanced_eps

def get_pupil_balanced_torcs(rec, mask1, mask2, balanced=True, present=False):
    """
    Used to get the torcs that are present/balanced between both masks
    """
    all_epochs = np.unique([str(ep) for ep in rec.epochs.name if 'TORC' in ep]).tolist()

    # find pupil matched epochs
    balanced_eps = []
    for ep in all_epochs:
        try:
            sp = rec['resp'].extract_epoch(ep, mask=mask1).shape[0]
            bp = rec['resp'].extract_epoch(ep, mask=mask2).shape[0]
            if balanced == True:
                if (abs(sp - bp) < 2) & (bp > 0) & (sp > 0):
                    balanced_eps.append(ep)
            elif present == True:
                # just make sure epoch exists in both
                if ((bp > 0) & (sp > 0)):
                    balanced_eps.append(ep)
        except:
            # epoch doesn't exist inside one of the masks.
            pass

    if len(balanced_eps)==0:
        log.info("no balanced epochs at site {}".format(rec.name))

    else:
        log.info("found {0} balanced epochs:".format(len(balanced_eps)))

    return balanced_eps


def generate_psth(rec):
    """
    Generate a psth contained in the mask and tile it across the whole passed recording, r,
    using replace epochs
    """
    r = rec.copy()
    if 'mask' in r.signals.keys():
        r_mask = r.apply_mask(reset_epochs=True).copy()
    else:
        r_mask = r.copy()
    # get epochs inside the masked data
    ep = np.unique([e for e in r_mask.epochs.name if ('STIM' in e) | ('TAR_' in e)]).tolist()

    # for each epoch, compute psth for the masked data, rep it for the number of trials in all the data
    r_psth = dict.fromkeys(ep)
    r_psth_sem = dict.fromkeys(ep)
    for e in ep:
        mask_psth = r_mask['resp'].extract_epoch(e).mean(axis=0)[np.newaxis, :, :]
        nreps = r['resp'].extract_epoch(e).shape[0]
        mask_psth = np.tile(mask_psth, [nreps, 1, 1])
        r_psth[e] = mask_psth

        psth_sem = r_mask['resp'].extract_epoch(e).std(axis=0)[np.newaxis, :, :]
        nCells = r['resp'].shape[0]
        psth_sem /= np.tile(np.sqrt(nreps), [1, nCells, 1])
        r_psth_sem[e] = np.tile(psth_sem, [nreps, 1, 1])

    # finally, replace epochs in the orignal recording with the masked psth
    r['psth'] = r['resp'].replace_epochs(r_psth)
    r['psth_sem'] = r['resp'].replace_epochs(r_psth_sem)

    return r

def create_ptd_masks(rec, act_pup_range=2):
    """
    Create active behavior mask, passive mask, passive big pupil mask, and passive small pupil mask.
    return new recording with the four masks as signals.

    Modified 2/20/2020, CRH. Now, define large pupil as pupil matched to active (w/in 2sd). 
    Small pupil is the (smaller tail) leftover.

    Modified 07/13/2020, CRH. Now, specify how many sd from the mean of active counts as big pupil.
        default is still 2.
    """

    r = rec.copy()
    r = r.create_mask(True)

    act_mask = r.and_mask(['HIT_TRIAL'])['mask']
    pass_mask = r.and_mask(['PASSIVE_EXPERIMENT'])['mask']
    miss_mask = r.and_mask(['MISS_TRIAL'])['mask']

    # define the cutoff as two sd less than the mean of active pupil
    cutoff = r['pupil']._data[act_mask._data].mean() - (act_pup_range * r['pupil']._data[act_mask._data].std())

    options = {'state': 'big', 'method': 'user_def_value', 'cutoff': cutoff, 'collapse': True, 'epoch': ['REFERENCE', 'TARGET']}
    pass_big_mask = create_pupil_mask(r.and_mask(['PASSIVE_EXPERIMENT']), **options)['mask']
    options['state'] = 'small'
    pass_small_mask = create_pupil_mask(r.and_mask(['PASSIVE_EXPERIMENT']), **options)['mask']

    r['a_mask'] = r['mask']._modified_copy(act_mask._data)
    r['p_mask'] = r['mask']._modified_copy(pass_mask._data)
    r['miss_mask'] = r['mask']._modified_copy(miss_mask._data)
    r['pb_mask'] = r['mask']._modified_copy(pass_big_mask._data)
    r['ps_mask'] = r['mask']._modified_copy(pass_small_mask._data)

    return r


def zscore_dict(d, d_norm=None):
    # d_norm is optional - used to compute the mean/std if passed (for example if
    # d is a subset of d_norm and want to normalize to ALL data)

    if d_norm is None:
        d_norm = d

    epochs = d_norm.keys()
    # get mean per neuron
    # get std per neuron
    for i, e in enumerate(epochs):
        if i == 0:
            r = d_norm[e]
        else:
            r = np.concatenate((r, d_norm[e]), axis=0)
    m = r.transpose([0, 2, 1]).reshape(-1, r.shape[1]).mean()
    std = r.transpose([0, 2, 1]).reshape(-1, r.shape[1]).std()

    # normalize dict
    epochs = d.keys()
    for i, e in enumerate(epochs):
        reps = d[e].shape[0]
        bins = d[e].shape[-1]
        cells = d[e].shape[1]
        d[e] = (d[e].transpose([0, 2, 1]).reshape(-1, r.shape[1]) - m).reshape(reps, bins, cells).transpose([0, 2, 1])
        d[e] = (d[e].transpose([0, 2, 1]).reshape(-1, r.shape[1]) / std).reshape(reps, bins, cells).transpose([0, 2, 1])

    return d


def zscore_per_stim(d1, d2, with_std=True):
    d = d1.copy()
    if d2 is None:
        d_norm = d
    elif d2 is not None:
        d_norm = d2.copy()

    for k in d_norm.keys():
        reps = d[k].shape[0]
        m = np.tile(d_norm[k].mean(axis=0), [reps, 1, 1])
        std = np.tile(d_norm[k].std(axis=0), [reps, 1, 1])
        std[std==0] = 1
        d[k] = d[k] - m
        if with_std:
            d[k] = d[k] / std

    return d

def pca_reduce_dimensionality(dict_proj, npcs=1, dict_fit=None):
    """
    Compute pcs on psth of d_fit and then project all data from d dict onto the first npcs.
    Return the new dict with reduced dimensionality and the variance explained by npcs
    """
    d = dict_proj.copy()
    if dict_fit is None:
        d_fit = d
    else:
        d_fit = dict_fit.copy()

    epochs = d_fit.keys()
    for i, e in enumerate(epochs):
        if i == 0:
            r = d_fit[e].mean(axis=0)
        else:
            r = np.concatenate((r, d_fit[e].mean(axis=0)), axis=-1)

    # perform PCA
    pca = PCA(n_components=npcs)
    log.info("PCA dict shape: {}".format(r.shape))
    pca = pca.fit(r.T)

    # normalize variance of components
    # pca.components_ = (pca.components_.T / pca.explained_variance_).T

    epochs = d.keys()
    for e in epochs:
        reps = d[e].shape[0]
        bins = d[e].shape[-1]
        cells = d[e].shape[1]
        d[e] = np.matmul(d[e].transpose([-1, 0, 1]).reshape(reps*bins, cells), pca.components_[:npcs, :].T).reshape(bins, reps, npcs).transpose([1, -1, 0])

    return d, np.sum(pca.explained_variance_ratio_[:npcs])


def get_balanced_rep_counts(d1, d2):
    """
    For each epoch, in d1/d2, make sure number of presentations are
    balanced between the two.
    """
    epochs = d1.keys()
    new_dict = {}
    for ep in epochs:
        n1 = d1[ep].shape[0]
        n2 = d2[ep].shape[0]
        log.info("epoch: {0} \n d1 has {1} reps, d2 has {2} reps".format(ep, n1, n2))
        if n1 > n2:
            reps = np.random.choice(np.arange(0, n1), n2)
            new_dict[ep] = np.concatenate((d1[ep][reps, :, :], d2[ep]), axis=0)
        elif n2 > n1:
            reps = np.random.choice(np.arange(0, n2), n1)
            new_dict[ep] = np.concatenate((d1[ep], d2[ep][reps, :, :]), axis=0)
        else:
            new_dict[ep] = np.concatenate((d1[ep], d2[ep]), axis=0)

    return new_dict


def downsample_raster(psth, nbins):
    chunks = psth.reshape(nbins, -1)
    psth = np.array([x.sum() for x in chunks])
    return psth

def downsample_error(sem, nbins):
    # error propogation for +/- operation
    chunks = sem.reshape(nbins, -1)
    sem = np.array([np.sqrt(sum(x**2)) for x in chunks])
    return sem