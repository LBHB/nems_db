"""
Functions to take the place of "cache_dprime.py" - they run as postprocessing step to
LV models and the typical entry point would be to give ctx / maybe other options.
"""
"""
Procedure:
    1) Load data
    2) Generate list of stimulus pairs
    3) Make pupil mask, classify pupil for each stim pair
    4) Generate est/val sets (each est/val dataset should be shape Neuron X Rep X Stim)
    5) Preprocess est set (apply same preprocessing to val)
    --- For each stim pair ---
        6) Dimensionality reduction (on est, apply same to val)
        7) Compute dprime, save metrics
        8) Split data into large / small pupil, based on pupil mask generated at the beginning
"""
import numpy as np
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import os
import pandas as pd 
import pickle
import sys

import nems_lbhb.projects.nat_pup_decoding.preprocessing as nat_preproc
import nems_lbhb.projects.nat_pup_decoding.decoding as decoding
import nems_lbhb.projects.nat_pup_decoding.dim_reduction as dr

import nems
import nems_lbhb.baphy as nb
import nems.db as nd
import logging

log = logging.getLogger(__name__)

def do_decoding_analysis(ctx):
    """
    Meant to replace CRH's personal code for doing decoding analysis.
    Basically the only this that needed to change is to allow the decoding.load_site 
    function to take a ctx (an already loaded recording). So then load_site just 
    reshapes the data accordingly and returns same stuff as always
    """
    # pull site / batch out of context
    site = ctx["meta"]["cellid"]
    batch = ctx["meta"]["batch"]
    modelname = "decoding"

    # ================================ SET RNG STATE ===================================
    np.random.seed(123)

    # ============================== SAVE PARAMETERS ===================================
    # define save directory
    path = nems.get_setting("NEMS_RESULTS_DIR")

    # ========================== Define analysis options ========================
    # these are harcoded right now, but should be editable in the future
    njacks = 10
    zscore = True
    pup_ops = None
    shuffle_trials = False
    regress_pupil = False
    use_xforms = False

    # old simulation stuff (brute force approach)
    sim1 = False
    sim2 = False
    sim12 = False
    sim_tdr_space = False  # perform the simulation in the TDR space. piggy-back on jackknifes for this. e.g. for each jackknife, do a new simulation
                        # this is kludgy. may want to do more iterations. If so, gonna need to rethink the modelname / est val creation etc.

    # default for this function (that's why we're running it afer model fit)
    lv_model = True       # perform decoding on the simulated, lv model predictions
    lv_str = ctx["meta"]["modelname"]

    pca_ops = None
    do_pls = False
    do_PCA = False
    pca_lv = False
    nc_lv = False
    nc_lv_z = False
    fix_tdr2 = False
    ddr2_method = 'fa'  # use FactorAnalysis to ID the noise axis, by default. Can also do 'pca'
    gain_only = False
    dc_only = False
    est_equal_val = False  # for low rep sites where can't perform cross-validation
    n_noise_axes = 0    # whether or not to go beyond TDR = 2 dimensions. e.g. if this is 2, Will compute a 4D TDR space (2 more noise dimensions)
    loocv = False    # leave-one-out cross validation
    exclude_low_fr = False
    threshold = None
    movement_mask = False
    use_old_cpn = False
    all_pup = False
    fa_model = False # simulate data from FA
    fa_sim = False

    if do_pls:
        log.info("Also running PLS dimensionality reduction for N components. Will be slower")
        raise DeprecationWarning("Updates have been made since this was last used. Make sure behavior is as expected")
    elif do_PCA:
        log.info("Also running trial averaged PCA dimensionality reduction for N components.")
        raise DeprecationWarning("Updates have been made since this was last used. Make sure behavior is as expected")
    else:
        log.info("Only performing TDR dimensionality reduction. No PLS or PCA")

    # ================ load LV information for this site =======================
    # can't compare these axes if we've reduced dimensionality
    if pca_ops is None:
        if pca_lv:
            fn = '/auto/users/hellerc/results/nat_pupil_ms/LV/pca_regression_lvs.pickle'
            # load results from pickle file
            with open(fn, 'rb') as handle:
                lv_results = pickle.load(handle)
            beta1 = lv_results[site+str(batch)]['beta1']
            beta2 = lv_results[site+str(batch)]['beta2']
        elif nc_lv:
            log.info("loading LVs from NC method using raw responses")
            fn = '/auto/users/hellerc/results/nat_pupil_ms/LV/nc_based_lvs.pickle'
            # load results from pickle file
            with open(fn, 'rb') as handle:
                lv_results = pickle.load(handle)
            beta1 = lv_results[site+str(batch)]['beta1']
            beta2 = lv_results[site+str(batch)]['beta2']
        elif nc_lv_z:
            log.info("loading LVs from NC method using z-scored responses")
            fn = '/auto/users/hellerc/results/nat_pupil_ms/LV/nc_zscore_lvs.pickle'
            # load results from pickle file
            with open(fn, 'rb') as handle:
                lv_results = pickle.load(handle)
            beta1 = lv_results[site+str(batch)]['beta1']
            beta2 = lv_results[site+str(batch)]['beta2']
        else:
            beta1=None
            beta2=None
    else:
        beta1 = None
        beta2 = None

    # ================================= load recording ==================================
    X, sp_bins, X_pup, pup_mask, epochs = decoding.load_site(site=site, batch=batch, 
                                        ctx=ctx,
                                        pca_ops=pca_ops,
                                        pup_ops=pup_ops,
                                        regress_pupil=regress_pupil,
                                        gain_only=gain_only,
                                        dc_only=dc_only,
                                        use_xforms=use_xforms,
                                        return_epoch_list=True,
                                        exclude_low_fr=exclude_low_fr,
                                        threshold=threshold,
                                        mask_movement=movement_mask,
                                        use_old_cpn=use_old_cpn)
    ncells = X.shape[0]
    nreps_raw = X.shape[1]
    nstim = X.shape[2]
    nbins = X.shape[3]
    sp_bins = sp_bins.reshape(1, sp_bins.shape[1], nstim * nbins)
    nstim = nstim * nbins

    # =========================== generate a list of stim pairs ==========================
    all_combos = list(combinations(range(nstim), 2))
    spont_bins = np.argwhere(sp_bins[0, 0, :])
    spont_combos = [c for c in all_combos if (c[0] in spont_bins) & (c[1] in spont_bins)]
    ev_ev_combos = [c for c in all_combos if (c[0] not in spont_bins) & (c[1] not in spont_bins)]
    spont_ev_combos = [c for c in all_combos if (c not in ev_ev_combos) & (c not in spont_combos)]

    # get list of epoch combos as a tuple (in the same fashion as above)
    epochs_bins = np.concatenate([[e+'_'+str(k) for k in range(nbins)] for e in epochs])
    epochs_str_combos = list(combinations(epochs_bins, 2))

    # =================================== simulate =======================================
    # update X to simulated data if specified. Else X = X_raw.
    # point of this is so that decoding axes / TDR space doesn't change for simulation (or for xforms predicted data)
    # should make results easier to interpret. CRH 06.04.2020
    X_raw = X.copy()
    pup_mask_raw = pup_mask.copy()
    meta = None
    if (sim1 | sim2 | sim12) & (not sim_tdr_space):
        X, pup_mask = decoding.simulate_response(X, pup_mask, sim_first_order=sim1,
                                                            sim_second_order=sim2,
                                                            sim_all=sim12,
                                                            ntrials=5000)
    elif lv_model:
        # get lv model predictions from context (ctx["val"]["pred"])
        # then evaluate decoding with predictions
        X, _, _, pup_mask, _ = decoding.load_site(site=site, batch=batch, 
                                        ctx=ctx,
                                        use_pred=True,
                                        pca_ops=pca_ops,
                                        pup_ops=pup_ops,
                                        regress_pupil=regress_pupil,
                                        gain_only=gain_only,
                                        dc_only=dc_only,
                                        use_xforms=use_xforms,
                                        return_epoch_list=True,
                                        exclude_low_fr=exclude_low_fr,
                                        threshold=threshold,
                                        mask_movement=movement_mask,
                                        use_old_cpn=use_old_cpn)

    elif sim_tdr_space:
        log.info("Performing simulations within TDR space. Unique simulation per each jackknife")

    elif fa_model:
        log.info(f"Simulate data based on factor analysis model with sim={fa_sim} for this dataset")
        big_psth = np.stack([X.reshape(ncells, nreps_raw, nstim)[:, pup_mask.reshape(1, nreps_raw, nstim)[0, :, s], s].mean(axis=1) for s in range(nstim)]).T.reshape(ncells, X.shape[2], nbins)[:, np.newaxis, :, :]
        small_psth = np.stack([X.reshape(ncells, nreps_raw, nstim)[:, pup_mask.reshape(1, nreps_raw, nstim)[0, :, s]==False, s].mean(axis=1) for s in range(nstim)]).T.reshape(ncells, X.shape[2], nbins)[:, np.newaxis, :, :]
        X, pup_mask = decoding.load_FA_model(site, batch, big_psth, small_psth, sim=fa_sim, nreps=5000)
    
    # another option for surrogates based on pop metrics
    # how to do this? Generate data elsewhere, then load? 
    # that way can compute pop metrics / pairwise metrics on each
    # set and then just load them here for decoding

    else:
        pass

    nreps = X.shape[1]

    # =============================== reshape data ===================================
    # reshape mask to match data
    pup_mask = pup_mask.reshape(1, nreps, nstim)
    pup_mask_raw = pup_mask_raw.reshape(1, nreps_raw, nstim)
    # reshape X (and X_raw)
    X = X.reshape(ncells, nreps, nstim)
    X_raw = X_raw.reshape(ncells, nreps_raw, nstim)
    # reshape X_pup
    X_pup = X_pup.reshape(1, nreps_raw, nstim)

    # ===================== decide if / how we should shuffle data ===================
    if shuffle_trials:
        # shuffle trials per neuron within state, so don't break any overall gain changes between large / small pupil
        if np.any(X!=X_raw):
            raise ValueError("Right now, we're just set up to do this for raw data. Not sure how you'd do it for sim data")
        else:
            log.info("Shuffing trials within pupil condition to break state-dependent changes in correlation")
            X = decoding.shuffle_trials(X, pup_mask)
            X_raw = X.copy()

    # ============================== get pupil variance ==================================
    # figure out pupil variance per stimulus (this always happens on raw data... X_pup and pup_mask_raw)
    pupil_range = nat_preproc.get_pupil_range(X_pup, pup_mask_raw)

    # =========================== generate list of est/val sets ==========================
    # also generate list of est / val for the raw data. Because of random number seed, it's 
    # critical that this happens first, and doesn't happend twice (if simulation is False)
    log.info("Generate list of {0} est / val sets".format(njacks))

    # generate raw est/val sets
    np.random.seed(123)
    est_raw, val_raw, p_est_raw, p_val_raw = nat_preproc.get_est_val_sets(X_raw, pup_mask=pup_mask_raw, njacks=njacks, est_equal_val=est_equal_val)
    nreps_train_raw = est_raw[0].shape[1]
    nreps_test_raw = val_raw[0].shape[1]

    # check if data was simulated. If so, then generate the est / val sets for this data
    xraw_equals_x = False
    if (X.shape == X_raw.shape):
        if np.all(X_raw == X):
            xraw_equals_x = True
            est = est_raw.copy()
            val = val_raw.copy()
            p_est = p_est_raw.copy()
            p_val = p_val_raw.copy()
        else:
            est, val, p_est, p_val = nat_preproc.get_est_val_sets(X, pup_mask=pup_mask, njacks=njacks)
    else:
        est, val, p_est, p_val = nat_preproc.get_est_val_sets(X, pup_mask=pup_mask, njacks=njacks)

    nreps_train = est[0].shape[1]
    nreps_test = val[0].shape[1]

    # determine number of dim reduction components (bounded by ndim in dataset) 
    # force to less than 10, for speed purposes.
    components = np.min([ncells, nreps_train, 10])

    # ============================ preprocess est / val sets =============================
    if zscore:
        log.info("z-score est / val sets")
        est, val, fullX = nat_preproc.scale_est_val(est, val, full=X)
        est_raw, val_raw, fullX_raw = nat_preproc.scale_est_val(est_raw, val_raw, full=X_raw)
    else:
        # just center data
        log.info("center est / val sets")
        est, val, fullX = nat_preproc.scale_est_val(est, val, full=X, sd=False)
        est_raw, val_raw, fullX_raw = nat_preproc.scale_est_val(est_raw, val_raw, full=X_raw, sd=False)

    # =========================== if fix tdr 2 =======================================
    # calculate first noise PC for each val set, use this to define TDR2, rather
    # than stimulus specific first noise PC (this method seems too noisy). Always
    # use raw data for this.
    if fix_tdr2:
        if ddr2_method=='nclv':
            log.info("Loading cached delta noise correlation axis as ddr2 noise axis")
            lvdict = pickle.load(open('/auto/users/hellerc/results/nat_pupil_ms/LV/nc_zscore_lvs.pickle', 'rb'))
            nc_ax = lvdict[site+str(batch)]['beta2'].T
            tdr2_axes = [nc_ax] * len(val)
        else:
            log.info("Finding first noise dimension for each est set using raw data")
            tdr2_axes = nat_preproc.get_first_pc_per_est(est_raw, method=ddr2_method)
    else:
        tdr2_axes = [None] * len(val)
    # set up data frames to save results (wait to preallocate space on first
    # iteration, because then we'll have the columns)
    temp_pca_results = pd.DataFrame()
    temp_pls_results = pd.DataFrame()
    temp_tdr_results = pd.DataFrame()
    pls_index = range(len(all_combos) * njacks * (components-2))
    pca_index = range(len(all_combos) * njacks)
    tdr_index = range(len(all_combos) * njacks)
    pca_idx = 0
    pls_idx = 0
    tdr_idx = 0
    # ============================== Loop over stim pairs ================================
    for stim_pair_idx, (ecombo, combo) in enumerate(zip(epochs_str_combos, all_combos)):
        # print every 500th pair. Don't want to overwhelm log
        if (stim_pair_idx % 500) == 0:
            log.info("Analyzing stimulus pair {0} / {1}".format(stim_pair_idx, len(all_combos)))
        if combo in spont_combos:
            category = 'spont_spont'
        elif combo in spont_ev_combos:
            category = 'spont_evoked'
        elif combo in ev_ev_combos:
            category = 'evoked_evoked'

        for ev_set in range(njacks):
            X_train = est[ev_set][:, :, [combo[0], combo[1]]] 
            X_test = val[ev_set][:, :, [combo[0], combo[1]]]
            _Xfull = fullX[ev_set][:, :, [combo[0], combo[1]]]
            if all_pup:
                # use all data for big / small pupil
                ptrain_mask = pup_mask.copy()[:, :, [combo[0], combo[1]]]
                ptest_mask = pup_mask.copy()[:, :, [combo[0], combo[1]]]

            else:
                ptrain_mask = p_est[ev_set][:, :, [combo[0], combo[1]]]
                ptest_mask = p_val[ev_set][:, :, [combo[0], combo[1]]]
            tdr2_axis = tdr2_axes[ev_set]

            # if ptrain_mask is all False, just set it to None so
            # that we don't attempt to run the bp/sp analysis. This is a kludgy 
            # way to deal with cases where we selected a pupil window for which there 
            # was not valid data (or enough valid data) in this experiment
            if (np.sum(ptrain_mask[:, :, 0]) < 5) | (np.sum(ptrain_mask[:, :, 1]) < 5):
                log.info(f"For combo: {combo}, ev_set: {ev_set} not enough data. stim1: {ptrain_mask[:, :, 0].sum()}, stim2: {ptrain_mask[:, :, 1].sum()}")
                ptrain_mask = None

            xtrain = nat_preproc.flatten_X(X_train[:, :, :, np.newaxis])
            xtest = nat_preproc.flatten_X(X_test[:, :, :, np.newaxis])
            xfull = nat_preproc.flatten_X(_Xfull[:, :, :, np.newaxis])

            # define raw data
            if xraw_equals_x:
                raw_data = None
            else:
                # define data to be used for tdr decomposition
                X_train_raw = est_raw[ev_set][:, :, [combo[0], combo[1]]] 
                xtrain_raw = nat_preproc.flatten_X(X_train_raw[:, :, :, np.newaxis])
                raw_data = (xtrain_raw, nreps_train_raw)

            # ============================== TDR ANALYSIS ==============================
            # custom dim reduction onto plane defined by dU and first PC of noise covariance (+ additional noise axes)
            if sim_tdr_space:
                # simulate data *after* after projecting into TDR space.
                try:
                    if not loocv:
                        _tdr_results = decoding.do_tdr_dprime_analysis(xtrain,
                                                        xtest,
                                                        nreps_train,
                                                        nreps_test,
                                                        tdr_data=raw_data,
                                                        n_additional_axes=n_noise_axes,
                                                        sim1=sim1,
                                                        sim2=sim2,
                                                        sim12=sim12,
                                                        beta1=beta1,
                                                        beta2=beta2,
                                                        tdr2_axis=tdr2_axis,
                                                        fullpup=all_pup,
                                                        fullX=xfull,
                                                        ptrain_mask=ptrain_mask,
                                                        ptest_mask=ptest_mask)
                    else:
                        raise NotImplementedError("WIP -- loocv for simulations")
                except:
                    log.info("Can't perform analysis for stimulus combo: {0}".format(combo))
                    _tdr_results = {}
            else:
                if not loocv:
                    _tdr_results = decoding.do_tdr_dprime_analysis(xtrain,
                                                                xtest,
                                                                nreps_train,
                                                                nreps_test,
                                                                tdr_data=raw_data,
                                                                n_additional_axes=n_noise_axes,
                                                                beta1=beta1,
                                                                beta2=beta2,
                                                                tdr2_axis=tdr2_axis,
                                                                fullpup=all_pup,
                                                                fullX=xfull,                                                            
                                                                ptrain_mask=ptrain_mask,
                                                                ptest_mask=ptest_mask)
                else:
                    # use leave-one-out cross validation
                    _tdr_results = decoding.do_tdr_dprime_analysis_loocv(xtrain,
                                                                        nreps_train,
                                                                        tdr_data=raw_data,
                                                                        n_additional_axes=n_noise_axes,
                                                                        beta1=beta1,
                                                                        beta2=beta2,
                                                                        tdr2_axis=tdr2_axis,
                                                                        pmask=ptrain_mask)
                
            _tdr_results.update({
                'n_components': 2+n_noise_axes,
                'jack_idx': ev_set,
                'combo': combo,
                'e1': ecombo[0],
                'e2': ecombo[1],
                'category': category,
                'site': site
            })
            # preallocate space for subsequent iterations
            if tdr_idx == 0:
                if 'bp_dp' not in _tdr_results.keys():
                    bp_cols = ['bp_dp', 'bp_evals', 'bp_dU_mag', 'bp_dU_dot_evec',
                                        'bp_cos_dU_wopt', 'bp_dU_dot_evec_sq', 'bp_evec_snr', 
                                        'bp_cos_dU_evec']
                    for bp_col in bp_cols:
                        if bp_col in ['bp_dU_dot_evec', 'bp_dU_dot_evec_sq', 'bp_evec_snr', 'bp_cos_dU_evec']:
                            _tdr_results[bp_col] = np.nan * np.ones((1, 2+n_noise_axes))
                        elif bp_col == 'bp_evals':
                            _tdr_results[bp_col] = np.nan * np.ones(2+n_noise_axes)
                        elif bp_col == 'bp_cos_dU_wopt':
                            _tdr_results[bp_col] = np.nan * np.ones((1, 1))
                        else:
                            _tdr_results[bp_col] = np.nan # add place holder in case the first didn't have data for this
                if 'sp_dp' not in _tdr_results.keys():
                    sp_cols = ['sp_dp', 'sp_evals', 'sp_dU_mag', 'sp_dU_dot_evec',
                                        'sp_cos_dU_wopt', 'sp_dU_dot_evec_sq', 'sp_evec_snr', 
                                        'sp_cos_dU_evec']
                    for sp_col in sp_cols:
                        if sp_col in ['sp_dU_dot_evec', 'sp_dU_dot_evec_sq', 'sp_evec_snr', 'sp_cos_dU_evec']:
                            _tdr_results[sp_col] = np.nan * np.ones((1, 2+n_noise_axes))
                        elif sp_col == 'sp_evals':
                            _tdr_results[sp_col] = np.nan * np.ones(2+n_noise_axes)
                        elif sp_col == 'sp_cos_dU_wopt':
                            _tdr_results[sp_col] = np.nan * np.ones((1, 1))
                        else:
                            _tdr_results[sp_col] = np.nan # add place holder in case the first didn't have data for this

                temp_tdr_results = temp_tdr_results.append([_tdr_results])
                tdr_results = pd.DataFrame(index=tdr_index, columns=temp_tdr_results.columns)
                tdr_results.loc[tdr_idx] = temp_tdr_results.iloc[0].values
                temp_tdr_results = pd.DataFrame()

            else:
                temp_tdr_results = temp_tdr_results.append([_tdr_results])
                tdr_results.loc[tdr_idx, temp_tdr_results.keys()] = temp_tdr_results.iloc[0].values
                temp_tdr_results = pd.DataFrame()
            tdr_idx += 1

            # ============================== PCA ANALYSIS ===============================
            if do_PCA:
                _pca_results = decoding.do_pca_dprime_analysis(xtrain, 
                                                            xtest, 
                                                            nreps_train,
                                                            nreps_test,
                                                            ptrain_mask=ptrain_mask,
                                                            ptest_mask=ptest_mask)
                _pca_results.update({
                    'n_components': 2,
                    'jack_idx': ev_set,
                    'combo': combo,
                    'e1': ecombo[0],
                    'e2': ecombo[1],
                    'category': category,
                    'site': site
                })
                # preallocate space for subsequent iterations
                if pca_idx == 0:
                    temp_pca_results = temp_pca_results.append([_pca_results])
                    pca_results = pd.DataFrame(index=pca_index, columns=temp_pca_results.columns)
                    pca_results.loc[pca_idx] = temp_pca_results.iloc[0].values
                    temp_pca_results = pd.DataFrame()

                else:
                    temp_pca_results = temp_pca_results.append([_pca_results])
                    pca_results.loc[pca_idx] = temp_pca_results.iloc[0].values
                    temp_pca_results = pd.DataFrame()
                pca_idx += 1

            if do_pls:
                # ============================== PLS ANALYSIS ===============================
                for n_components in range(2, components):

                    _pls_results = decoding.do_pls_dprime_analysis(xtrain, 
                                                                xtest, 
                                                                nreps_train,
                                                                nreps_test,
                                                                ptrain_mask=ptrain_mask,
                                                                ptest_mask=ptest_mask,
                                                                n_components=n_components)
                    _pls_results.update({
                        'n_components': n_components,
                        'jack_idx': ev_set,
                        'combo': combo,
                        'e1': ecombo[0],
                        'e2': ecombo[1],
                        'category': category,
                        'site': site
                    })
                
                    # preallocate space for subsequent iterations
                    if pls_idx == 0:
                        temp_pls_results = temp_pls_results.append([_pls_results])
                        pls_results = pd.DataFrame(index=pls_index, columns=temp_pls_results.columns)
                        pls_results.loc[pls_idx] = temp_pls_results.iloc[0].values
                        temp_pls_results = pd.DataFrame()

                    else:
                        temp_pls_results = temp_pls_results.append([_pls_results])
                        pls_results.loc[pls_idx] = temp_pls_results.iloc[0].values
                        temp_pls_results = pd.DataFrame()

                    pls_idx += 1

    
    # convert columns to str
    tdr_results.loc[:, 'combo'] = ['{0}_{1}'.format(c[0], c[1]) for c in tdr_results.combo.values]
    if do_PCA:
        pca_results.loc[:, 'combo'] = ['{0}_{1}'.format(c[0], c[1]) for c in pca_results.combo.values]
    if do_pls:
        pls_results.loc[:, 'combo'] = ['{0}_{1}'.format(c[0], c[1]) for c in pls_results.combo.values]

    # get mean pupil range for each combo
    log.info('Computing mean pupil range for each pair of stimuli')
    combo_to_tup = lambda x: (int(x.split('_')[0]), int(x.split('_')[1])) 
    combos = pd.Series(tdr_results['combo'].values).apply(combo_to_tup)
    pr = pupil_range
    get_mean = lambda x: (pr[pr.stim==x[0]]['range'] + pr[pr.stim==x[1]]['range']) / 2
    pr_range = combos.apply(get_mean)
    tdr_results['mean_pupil_range'] = pr_range.values

    # convert to correct dtypes
    tdr_results = decoding.cast_dtypes(tdr_results)
    if do_PCA:
        pca_results = decoding.cast_dtypes(pca_results)
    if do_pls:
        pls_results = decoding.cast_dtypes(pls_results)

    # collapse over results to save disk space by packing into "DecodingResults object"
    log.info("Compressing results into DecodingResults object... ")
    tdr_results = decoding.DecodingResults(tdr_results, pupil_range=pupil_range)
    if do_PCA:
        pca_results = decoding.DecodingResults(pca_results, pupil_range=pupil_range)
    if do_pls:
        pls_results = decoding.DecodingResults(pls_results, pupil_range=pupil_range)

    if meta is not None:
        if 'mask_bins' in meta.keys():
            tdr_results.meta['mask_bins'] = meta['mask_bins']

    # save results
    modelname = modelname.replace('*', '_')

    log.info("Saving results to {}".format(os.path.join(path, str(batch))))
    if not os.path.isdir(os.path.join(path, str(batch))):
        os.mkdir(os.path.join(path, str(batch)))
    if not os.path.isdir(os.path.join(path, str(batch), site)):
        os.mkdir(os.path.join(path, str(batch), site))

    tdr_results.save_json(os.path.join(path, str(batch), site, modelname+'_TDR.json'))
    tdr_results.save_pickle(os.path.join(path, str(batch), site, modelname+'_TDR.pickle'))

    if do_PCA:
        pca_results.save_pickle(os.path.join(path, str(batch), site, modelname+'_PCA.pickle'))

    if do_pls:
        pls_results.save_pickle(os.path.join(path, str(batch), site, modelname+'_PLS.pickle'))


    return 0