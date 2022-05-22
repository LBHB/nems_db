"""
decoding tools for natural sounds analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import itertools
import pickle
import os
import copy

import nems.db as nd
import nems_lbhb.baphy as nb
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems.xform_helper import load_model_xform
from nems.preprocessing import resp_to_pc
import nems_lbhb.preprocessing as nems_preproc
import nems_lbhb.baphy_io as io

import nems_lbhb.projects.nat_pup_decoding.preprocessing as nat_preproc
import nems_lbhb.projects.nat_pup_decoding.dim_reduction as dr
import nems_lbhb.projects.nat_pup_decoding.general_preprocessing as preproc
import nems_lbhb.projects.nat_pup_decoding.simulate_data as simulate
import nems_lbhb.projects.nat_pup_decoding.plotting as cplt

import hashlib

import logging

log = logging.getLogger()

# ================================= Results Object to hold analysis results ===============================================
class DecodingResults():
    """
    Idea is to do a little preprocessing of jackknifed results and to make extracting of 
    results for a given recording site easier.
    """

    def __init__(self, df=None, pupil_range=None):
        """
        Init with the results data frame returned at the end of the dprime analysis
        Dprime analysis is pairwise, so each analysis returns results for many pairs of 
        stimuli and for many jackknifes and for many different dimensionalities (e.g. if
        doing PLS dim reduction, might have results in 2D space, 3D space etc.)
        """
        self.meta = {} # instantiate empty dictionary for miscellaneuos meta data to get stashed
        if df is not None:
            # set attributes with list of unique stimulus combinations / categories
            self.all_stimulus_pairs = df.combo.unique().tolist()
            self.spont_stimulus_pairs = df[df['category']=='spont_spont'].combo.unique().tolist()
            self.spont_evoked_stimulus_pairs = df[df['category']=='spont_evoked'].combo.unique().tolist()
            self.evoked_stimulus_pairs = df[df['category']=='evoked_evoked'].combo.unique().tolist()
            mapping = {a: (b, c) for a, b ,c in zip(df.combo, df.e1, df.e2)}
            self.mapping = mapping  # used to get true epoch names

            # set attribute for number of unique dimensionality reductions
            self.unique_subspace_dims = df.n_components.unique().tolist()

            # get df columns
            self.numeric_keys = df.select_dtypes(include=['float32', 'float64']).columns
            self.object_keys = df.select_dtypes(include=['object']).columns

            # get number of jacknnifes
            self.n_jacks = df.jack_idx.values.astype(np.float).max() + 1

            
            # Collapse over jackknifes. Save mean / sd of all numeric params.
            log.info("Collapsing results over jackknifes")
            # returns a dataframe for numeric results
            self.numeric_results = self._collapse_numeric(df) 
            
            # objects = arrays (e.g. arrays of eigenvectors, evals, decoding vectors).
            # returns a dictionary of data frames
            self.array_results = self._collapse_objects(df)

            # map evoked_stimulus_pairs to index

            # modify the above results by removing collapsing across spont repeats
            log.info("Consolidating results by combining all spont bins")
            try:
                self._consolidate_spont_results()
            except:
                log.info("Couldn't consolidate spont")

            # add dataframe detailing pupil results for each stim at this site
            self.pupil_range = pupil_range


    def _collapse_numeric(self, df):
        """
        Collapse numeric results over jackknifes. Return a dataframe with
        mean / sd for each stimulus pair / subspace dimensionality
        """
        grouped = df.groupby(by=['jack_idx', 'combo', 'n_components'])
        num_cols = self.numeric_keys
        mean = grouped[num_cols].mean().groupby(by=['combo', 'n_components']).mean()
        sem = grouped[num_cols].mean().groupby(by=['combo', 'n_components']).sem()
        sem.columns = [c+'_sem' for c in sem.columns]
        return pd.concat([mean, sem], axis=1)


    def _collapse_objects(self, df):
        categories = ['jack_idx', 'combo', 'n_components']
        obj_cols = self.object_keys
        unique_results = list(itertools.product(self.all_stimulus_pairs, self.unique_subspace_dims))
        # have to loop, because not straightforward to mean over categories in the 
        # case of objects
        results = {}
        index = pd.MultiIndex.from_tuples(unique_results, names=['combo', 'n_components'])
        for obj in obj_cols:
            log.info("collapsing results for object: {}".format(obj))
            _df = df[[obj]+categories]
            results[obj] = pd.DataFrame(index=index, columns=['mean', 'sem'])
            for idx in index:
                try:
                    x = np.concatenate([np.expand_dims(arr, 0) for arr in _df[(_df.combo==idx[0]) & (_df.n_components==idx[1])][obj].values], 0)
                    if 'evecs' in obj:
                        # reflect vectors so that there's no sign ambiguity when averaging
                        x = reflect_eigenvectors(x)
                    mean = np.nanmean(x, axis=0)
                    nanslice = [slice(0, x.shape[0], 1)] + [0] * (x.ndim - 1)
                    nanslice = tuple(nanslice)
                    sem = np.nanstd(x, axis=0) / np.sqrt(np.isfinite(x[nanslice].squeeze()).sum())
                    results[obj].loc[idx]['mean'] = mean
                    results[obj].loc[idx]['sem'] = sem
                except:
                    log.info("Failed trying to concatenate object: {}".format(obj))

        return results


    def _consolidate_spont_results(self):
        """
        Idea is that there are many spont vs. spont comparisons, and for each sound
        there are many spont comparisons. In reality, there should we just one
        spont. vs. spont result and one spont vs. evoked result (per evoked sound)

        Perform the consolidation by modifying the already created numeric results df and
        object results dict / dfs. Propogate error using sem_new = np.sqrt(sem1**2 + sem2**2 + ...)
        """

        # SPONT vs. SPONT

        # 1) deal with numeric results for spont spont
        df = self.numeric_results.copy()
        mean_cols = [c for c in df.columns if '_sem' not in c]
        err_cols = [c for c in df.columns if '_sem' in c]

        spont_spont_mean = df.loc[pd.IndexSlice[self.spont_stimulus_pairs, :], :][mean_cols].groupby(by='n_components').mean()
        spont_spont_sem = df.loc[pd.IndexSlice[self.spont_stimulus_pairs, :], :][err_cols].groupby(by='n_components').apply(error_prop)
        spont_spont = pd.concat([spont_spont_mean, spont_spont_sem], axis=1)
        new_idx = pd.MultiIndex.from_tuples([pd.Categorical(('spont_spont', n_components)) 
                                for n_components in spont_spont.index], names=['combo', 'n_components'])
        spont_spont.set_index(new_idx, inplace=True)

        # drop individual spont_spont pairs from master df
        df = df[~df.index.get_level_values('combo').isin(self.spont_stimulus_pairs)]

        # add new spont results to df
        df = spont_spont.append(df)
        self.numeric_results = df.copy()


        # 2) deal with array results for spont_spont
        for obj in self.object_keys:
            df = self.array_results[obj].copy()
            sp_df = df.loc[pd.IndexSlice[self.spont_stimulus_pairs, :], :]

            if 'evecs' in obj:
                m = [np.nanmean(reflect_eigenvectors(x), axis=0) for x in [np.stack([a for a in arr[1]['mean'].values]) for arr in sp_df.groupby('n_components')]]
                sem = [error_prop(x, axis=0) for x in [np.stack([a for a in arr[1]['sem'].values]) for arr in sp_df.groupby('n_components')]]
            else:
                m = [np.nanmean(x, axis=0) for x in [np.stack([a for a in arr[1]['mean'].values]) for arr in sp_df.groupby('n_components')]]
                sem = [error_prop(x, axis=0) for x in [np.stack([a for a in arr[1]['sem'].values]) for arr in sp_df.groupby('n_components')]]
            
            components = [arr[0] for arr in sp_df.groupby('n_components')]
            new_idx = pd.MultiIndex.from_tuples([pd.Categorical(('spont_spont', n_components)) 
                                for n_components in components], names=['combo', 'n_components'])
            new_df = pd.DataFrame(index=new_idx, columns=['mean', 'sem'])
            new_df['mean'] = m
            new_df['sem'] = sem

            df = df[~df.index.get_level_values('combo').isin(self.spont_stimulus_pairs)]
            df = new_df.append(df)
            
            self.array_results[obj] = df.copy()

        self.spont_stimulus_pairs = ['spont_spont']


        # SPONT vs. EVOKED
        df = self.numeric_results.copy()
        unique_evoked_bins = np.unique([[c.split('_')[0], c.split('_')[1]] for c in self.evoked_stimulus_pairs])

        # 1) deal with numeric results
        new_sp_ev_pairs = []
        for stim in unique_evoked_bins:
            # get all spont / evoked combos
            sp_ev = np.unique([c for c in self.spont_evoked_stimulus_pairs if stim in c])
            m = df.loc[pd.IndexSlice[sp_ev, :], :][mean_cols].groupby(by='n_components').mean()
            sem = df.loc[pd.IndexSlice[sp_ev, :], :][err_cols].groupby(by='n_components').apply(error_prop)
            sp_ev_df = pd.concat([m, sem], axis=1)
            new_idx = pd.MultiIndex.from_tuples([pd.Categorical(('spont_{}'.format(stim), n_components)) 
                                for n_components in sp_ev_df.index], names=['combo', 'n_components']) 
            sp_ev_df.set_index(new_idx, inplace=True)
            df = sp_ev_df.append(df)
            new_sp_ev_pairs.append('spont_{}'.format(stim))

        # remove inividual spont_evoked pairs 
        df = df[~df.index.get_level_values('combo').isin(self.spont_evoked_stimulus_pairs)] 

        # save updated dataframe for numeric results
        self.numeric_results = df.copy()

        # 2) deal with object results
        for obj in self.object_keys:
            df = self.array_results[obj].copy()
            for stim in unique_evoked_bins:
                sp_ev = np.unique([c for c in self.spont_evoked_stimulus_pairs if stim in c])
                sp_df = df.loc[pd.IndexSlice[sp_ev, :], :]

                if 'evecs' in obj:
                    m = [np.nanmean(reflect_eigenvectors(x), axis=0) for x in [np.stack([a for a in arr[1]['mean'].values]) for arr in sp_df.groupby('n_components')]]
                    sem = [error_prop(x, axis=0) for x in [np.stack([a for a in arr[1]['sem'].values]) for arr in sp_df.groupby('n_components')]]
                else:
                    m = [np.nanmean(x, axis=0) for x in [np.stack([a for a in arr[1]['mean'].values]) for arr in sp_df.groupby('n_components')]]
                    sem = [error_prop(x, axis=0) for x in [np.stack([a for a in arr[1]['sem'].values]) for arr in sp_df.groupby('n_components')]]
                components = [arr[0] for arr in sp_df.groupby('n_components')]
                new_idx = pd.MultiIndex.from_tuples([pd.Categorical(('spont_{}'.format(stim), n_components)) 
                                    for n_components in components], names=['combo', 'n_components'])
                new_df = pd.DataFrame(index=new_idx, columns=['mean', 'sem'])
                new_df['mean'] = m
                new_df['sem'] = sem

                df = df[~df.index.get_level_values('combo').isin(self.spont_evoked_stimulus_pairs)]
                df = new_df.append(df)
                self.array_results[obj] = df

        # update self.spont_evoked_stimulus_pairs
        self.spont_evoked_stimulus_pairs = new_sp_ev_pairs       

        # no need to return anything... just update object attributes

    def get_epochbins(self, epochbins):
        '''
        Get all stimulus pairs possible built from epochbins.
        Epochbins is a list of tuples of form (STIM_00xx, bin)

        return the df index using self.mapping
        '''
        # use mapping to figure out the "self.combos" that correspond to
        # each of the epochbins
        return idx


    def get_result(self, name, stim_pair=None, n_components=None):

        if name in self.numeric_keys:
            if stim_pair is None:
                m = self.numeric_results.index.get_level_values('n_components') == n_components
                return [self.numeric_results.loc[m, name],
                        self.numeric_results.loc[m, name+'_sem']]
            elif n_components is None:
                if type(stim_pair) is str:
                    m = self.numeric_results.index.get_level_values('combo') == stim_pair
                else:
                    m = self.numeric_results.index.get_level_values('combo').isin(stim_pair)
                return [self.numeric_results.loc[m, name],
                        self.numeric_results.loc[m, name+'_sem']]
            else:
                m1 = self.numeric_results.index.get_level_values('n_components') == n_components
                if type(stim_pair) is str:
                    m2 = self.numeric_results.index.get_level_values('combo') == stim_pair
                else:
                    m2 = self.numeric_results.index.get_level_values('combo').isin(stim_pair)
                m = m1 & m2
                return [self.numeric_results.loc[m, name],
                        self.numeric_results.loc[m, name+'_sem']]

        elif name in self.object_keys:
            if stim_pair is None:
                return [self.array_results[name].loc[pd.IndexSlice[:, n_components], 'mean'], 
                        self.array_results[name].loc[pd.IndexSlice[:, n_components], 'sem']]
            elif n_components is None:
                return [self.array_results[name].loc[pd.IndexSlice[stim_pair, :], 'mean'], 
                        self.array_results[name].loc[pd.IndexSlice[stim_pair, :], 'sem']]
            else:
                return [self.array_results[name].loc[pd.IndexSlice[stim_pair, n_components], 'mean'], 
                        self.array_results[name].loc[pd.IndexSlice[stim_pair, n_components], 'sem']]


        else:
            raise ValueError("unknown result column") 

    
    def slice_array_results(self, name, stim_pair, n_components, idx=None):
        """
        Specific fn to slice into a given index of the array result. For example,
        evecs_test might be a 2x2 array for each stim_pair, when n_components=2.
        If we only want the first eigenvector, we'd say:
            slice_array_results('evecs_test', None, 2, idx=[None, 0])
        """
        result = self.get_result(name, stim_pair, n_components)

        if type(result[0]) is np.ndarray:
            mean = self._slice_array(result[0], idx)
            sem = self._slice_array(result[1], idx)

        else:
            mean = self._slice_series(result[0], idx)
            sem = self._slice_series(result[1], idx)

        return [mean, sem]

    
    def _slice_array(self, x, idx=None):
        if idx is None:
            return x
        else:
            idx = np.array(idx)
            if np.any(idx == None):
                none_index = np.argwhere(idx == None)
                for ni in none_index[:, 0]:
                    idx[ni] = slice(0, x.shape[ni])
            idx = tuple(idx)
            return x[idx]

    def _slice_series(self, x, idx=None):
        if idx is None:
            return x

        else:
            return x.apply(lambda x: x[idx])
            #newx = x.copy()
            #newcol = [self._slice_array(_x, idx) for _x in x.values]
            #newx[x.index] = pd.Series(newcol)
            #return newx

    
    def save_pickle(self, fn):
        log.info("Saving pickle to {}".format(fn))
        try:
            with open(fn, 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            nn = hashlib.md5(os.path.split(fn)[1].strip('.pickle').encode()).hexdigest() + '.pickle'
            fn = os.path.join(os.path.split(fn)[0], nn)
            log.info(f"Hashing model name because model name is too long. Saveing to {fn}")
            with open(fn, 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        log.info('Success!')

    def load_results(self, fn, cache_path=None, recache=False):
        if (not os.path.isfile(fn)) & (cache_path is None):
            # try to find a hashed version
            nn = hashlib.md5(os.path.split(fn)[1].strip('.pickle').encode()).hexdigest() + '.pickle'
            fn = os.path.join(os.path.split(fn)[0], nn)
            if (not os.path.isfile(fn)):
                raise FileNotFoundError

        # look for locally cached results
        if cache_path is not None:
            cache_file = os.path.join(cache_path, os.path.sep.join(fn.split(os.path.sep)[-4:]))
        
            if os.path.isfile(cache_file) & (recache==False):
                log.info("loading pickle from {}".format(cache_file))
                with open(fn, 'rb') as handle:
                    data = pickle.load(handle, protocol=pickle.HIGHEST_PROTOCOL)
                return data

            if (not os.path.isfile(cache_file)) | recache:
                # load fn, then cache it
                log.info("loading pickle from {}".format(fn))
                with open(fn, 'rb') as handle:
                    data = pickle.load(handle, protocol=pickle.HIGHEST_PROTOCOL)
                log.info("caching pickle to {0}".format(cache_file))
                try:
                    with open(cache_file, 'wb') as handle:
                        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except FileNotFoundError:
                    os.mkdir(os.path.split(cache_file)[0])
                    with open(cache_file, 'wb') as handle:
                        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

                return data
        else:
            log.info("loading pickle from {}".format(fn))
            with open(fn, 'rb') as handle:
                data = pickle.load(handle)
            return data

    def save_json(self, fn):
        raise DeprecationWarning("Don't use this anymore")
        log.info("json serializing DecodingResults object to {}".format(fn))
        js_string = jsonpickle.encode(self)
        try:
            with open(fn, 'w') as handle:
                json.dump(js_string, handle)
        except OSError:
            nn = hashlib.md5(os.path.split(fn)[1].strip('.pickle').encode()).hexdigest() + '.pickle'
            fn = os.path.join(os.path.split(fn)[0], nn)
            log.info(f"Hashing model name because model name is too long. Saveing to {fn}")
            with open(fn, 'w') as handle:
                json.dump(js_string, handle)
    
    def load_json(self, fn):
        raise DeprecationWarning("Don't use this anymore")
        if not os.path.isfile(fn):
            raise FileNotFoundError
        log.info("loading json string from {}".format(fn))
        with open(fn, 'r') as handle:
            js_string = json.load(handle)
        return jsonpickle.decode(js_string)
        


def reflect_eigenvectors(x):
    """
    Need a special function for taking the mean of eigenvectors because we care about direction here.
    So, for example, the mean of [1, 0] and [-1, 0] should be [1, 0], not [0, 0].
    In order to do this, force cosines of all vectors in x to be postivie relative to a random
    vector in the space.

    First dim of x must be the number of vectors to be averaged. 
    Last dim of x is the indexer of eigenvectors. e.g. first eigenvector is x[0, :, 0]
    """
    # random reference vector
    xnew = x.copy()
    for v in range(x.shape[-1]):
        cum_sum = x[0, :, v]
        cum_sum /= np.linalg.norm(cum_sum)
        for i in np.arange(1, x.shape[0]):   
            if np.any(np.isnan(x[i, :, v])):
                xnew[i, :, v] = x[i, :, v]
            else:
                cos = cum_sum.dot(x[i, :, v])
                if cos > 0:
                    cum_sum += x[i, :, v]
                    cum_sum /= np.linalg.norm(cum_sum)

                else:
                    cum_sum += np.negative(x[i, :, v])
                    cum_sum /= np.linalg.norm(cum_sum)
                    xnew[i, :, v] = np.negative(x[i, :, v])
  
    return xnew


def unit_vectors(x):
    """
    xdim = number components x nvectors
    return matrix of same shape where each vector in range xdim.shape[-1] 
    is norm 1
    """
    xnew = x.copy()
    for v in range(x.shape[-1]):
        if np.linalg.norm(x[:, v]) > 0:
            xnew[:, v] = x[:, v] / np.linalg.norm(x[:, v])
    return xnew


def error_prop(x, axis=0):
    """
    Error propagation function.
    """
    nanslice = [0] * (x.ndim)
    nanslice[axis] = slice(0, x.shape[0], 1)
    nanslice = tuple(nanslice)
    if type(x) is not np.ndarray:
        return np.sqrt(x.pow(2).sum(axis=axis)) / np.isfinite(x.values[nanslice]).sum()
    else:
        return np.sqrt(np.nansum(x**2, axis=axis)) / np.isfinite(x[nanslice]).sum()


# =============================================== random helpers ==========================================================
# assortment of helper functions to clean up cache script.
def do_tdr_dprime_analysis(xtrain, xtest, nreps_train, nreps_test, tdr_data=None, n_additional_axes=0,
                                    beta1=None, beta2=None, tdr2_axis=None, 
                                    fullX=None, fullpup=False,
                                    ptrain_mask=None, ptest_mask=None, 
                                    sim1=False, sim2=False, sim12=False, verbose=False):
        """
        perform TDR (custom dim reduction): project into 2D space defined by dU and first noise PC
        compute dprime and associated metrics
        return results in a dictionary

        NEW: 06.04.2020 - Can perform simulation here, in TDR space
        """

        if sim1 | sim2 | sim12:
            raise DeprecationWarning("Needs to be re-evaluated. Have change stuff. Will this still work? crh - 29.10.2021")
            # simulate data. If pupil mask is specified, use this to created simulated trials.
            if ptrain_mask is not None:
                xtest_sim = nat_preproc.fold_X(xtest, nreps=nreps_test, nstim=2, nbins=1)
                pmask_sim = ptest_mask[:, :, :, np.newaxis]
                x_sim = []
                pup_mask_sim = []
                for s in range(xtest_sim.shape[2]):
                    _x_sim, _pup_mask_sim = simulate_response(xtest_sim[:, :, [s], :], pmask_sim[:, :, [s], :], 
                                                                            sim_first_order=sim1,
                                                                            sim_second_order=sim2,
                                                                            sim_all=sim12,
                                                                            nreps=5000,
                                                                            suppress_log=True)
                    x_sim.append(_x_sim)
                    pup_mask_sim.append(_pup_mask_sim)
                x_sim = np.concatenate(x_sim, axis=2)
                pup_mask_sim = np.concatenate(pup_mask_sim, axis=2)
                # pull out simulated test set, that's balanced over pupil conditions
                # do this by taking every other trial
                # only simulate the "test" set. This way decoding axis always the same as raw data analysis
                xtest = x_sim[:, ::2, :, 0]
                nreps_test = xtest.shape[1]
                xtest = xtest.reshape(xtest.shape[0], -1)
                ptest_mask = pup_mask_sim[:, ::2, :, 0]
                
            else:
                raise NotImplementedError("Can't do simulations without specifying a pupil mask. TODO: update decoding.simulate_response to handle this")
        
        if n_additional_axes==-1:
            naa = 0
        else:
            naa = n_additional_axes
        tdr = dr.TDR(tdr2_init=tdr2_axis, n_additional_axes=naa)
        if tdr_data is None:
            Y = dr.get_one_hot_matrix(ncategories=2, nreps=nreps_train)
            tdr.fit(xtrain.T, Y.T)
        else:
            Y = dr.get_one_hot_matrix(ncategories=2, nreps=tdr_data[1])
            tdr.fit(tdr_data[0].T, Y.T)

        if n_additional_axes==-1:
            tdr.weights = tdr.weights[[0], :]

        tdr_weights = tdr.weights

        xtrain_tdr = (xtrain.T @ tdr_weights.T).T
        xtest_tdr = (xtest.T @ tdr_weights.T).T

        xtrain_tdr = nat_preproc.fold_X(xtrain_tdr, nreps=nreps_train, nstim=2, nbins=1).squeeze(3)
        xtest_tdr = nat_preproc.fold_X(xtest_tdr, nreps=nreps_test, nstim=2, nbins=1).squeeze(3)

        # we use this if we're doing "cross validation" with full data being projected.
        # this would just be to boost the n's for low rep sites.
        if (fullX is not None) & (fullpup):
            fullX_tdr = (fullX.T @ tdr_weights.T).T
            fullX_tdr = nat_preproc.fold_X(fullX_tdr, nreps=nreps_train+nreps_test, nstim=2, nbins=1).squeeze(3)

        # perform training set decoding analysis (this is so that decoding axis is always defined 
        # using the raw data, not simulated data)
        tdr_train_var = np.var(xtrain_tdr.T @ tdr_weights)  / np.var(xtrain)

        tdr_dp_train, tdr_wopt_train, tdr_evals_train, tdr_evecs_train, evec_sim_train, tdr_dU_train = \
                                compute_dprime(xtrain_tdr[:, :, 0], xtrain_tdr[:, :, 1])

        # compute dprime metrics diag decoder
        if n_additional_axes >= 0:
            tdr_dp_train_diag, tdr_wopt_train_diag, _, _, _, x = \
                                    compute_dprime(xtrain_tdr[:, :, 0], xtrain_tdr[:, :, 1], diag=True)
        else:
            tdr_dp_train_diag = np.nan; tdr_wopt_train_diag = np.nan

        # now, compute dprime on the test set
        tdr_test_var = np.var(xtest_tdr.T @ tdr_weights)  / np.var(xtest)

        # compute dprime metrics raw 
        tdr_dp_test, tdr_wopt_test, tdr_evals_test, tdr_evecs_test, evec_sim_test, tdr_dU_test = \
                                compute_dprime(xtest_tdr[:, :, 0], xtest_tdr[:, :, 1], wopt=tdr_wopt_train)

        # overwrite test decoder with training, since it's fixed
        tdr_wopt_test = tdr_wopt_train

        # compute dprime diag on test set
        if n_additional_axes >= 0:
            tdr_dp_test_diag, tdr_wopt_test_diag, _, _, _, _ = \
                                    compute_dprime(xtest_tdr[:, :, 0], xtest_tdr[:, :, 1], diag=True)
        else:
            tdr_dp_test_diag = np.nan; tdr_wopt_test_diag = np.nan

        # caculate additional metrics (as long as we decoded in a space > 1D)
        if n_additional_axes >= 0:
            dU_mag_train         = np.linalg.norm(tdr_dU_train)
            dU_dot_evec_train    = tdr_dU_train.dot(tdr_evecs_train)
            cos_dU_wopt_train    = abs(unit_vectors(tdr_dU_train.T).T.dot(unit_vectors(tdr_wopt_train)))
            dU_dot_evec_sq_train = tdr_dU_train.dot(tdr_evecs_train) ** 2
            evec_snr_train       = dU_dot_evec_sq_train / tdr_evals_train
            cos_dU_evec_train    = abs(unit_vectors(tdr_dU_train.T).T.dot(tdr_evecs_train))

            dU_mag_test         = np.linalg.norm(tdr_dU_test)
            dU_dot_evec_test    = tdr_dU_test.dot(tdr_evecs_test)
            cos_dU_wopt_test    = abs(unit_vectors(tdr_dU_test.T).T.dot(unit_vectors(tdr_wopt_test)))
            dU_dot_evec_sq_test = tdr_dU_test.dot(tdr_evecs_test) ** 2
            evec_snr_test       = dU_dot_evec_sq_test / tdr_evals_test
            cos_dU_evec_test    = abs(unit_vectors(tdr_dU_test.T).T.dot(tdr_evecs_test))

            # project eigenvectors, dU, and wopt back into N-dimensional space
            # in order to investigate which neurons contribute to signal vs. noise
            # (think this just needs to be done for train set)
            dU_all = tdr_dU_train.dot(tdr.weights)
            wopt_all = tdr_wopt_train.T.dot(tdr.weights).T
            evecs_all = tdr_evecs_train.dot(tdr.weights).T

            # also do for test (except for wopt which is defined on train)
            dU_all_test = tdr_dU_test.dot(tdr.weights)
            evecs_all_test = tdr_evecs_test.dot(tdr.weights).T

            # calculate angle between responses
            # in raw response space
            _r = nat_preproc.fold_X(xtest, nreps=nreps_test, nstim=2, nbins=1).squeeze(3)
            r1 = _r[:, :, 0].mean(axis=-1) / np.linalg.norm(_r[:, :, 0].mean(axis=-1)) 
            r2 = _r[:, :, 1].mean(axis=-1) / np.linalg.norm(_r[:, :, 1].mean(axis=-1)) 
            cos_r1r2_test = np.abs(r1.dot(r2))
            _r = nat_preproc.fold_X(xtrain, nreps=nreps_train, nstim=2, nbins=1).squeeze(3)
            r1 = _r[:, :, 0].mean(axis=-1) / np.linalg.norm(_r[:, :, 0].mean(axis=-1)) 
            r2 = _r[:, :, 1].mean(axis=-1) / np.linalg.norm(_r[:, :, 1].mean(axis=-1))             
            cos_r1r2_train = np.abs(r1.dot(r2))

            # in dDR space
            r1 = xtest_tdr[:, :, 0].mean(axis=-1) / np.linalg.norm(xtest_tdr[:, :, 0].mean(axis=-1)) 
            r2 = xtest_tdr[:, :, 1].mean(axis=-1) / np.linalg.norm(xtest_tdr[:, :, 1].mean(axis=-1)) 
            cos_ddr_r1r2_test = np.abs(r1.dot(r2))
            r1 = xtrain_tdr[:, :, 0].mean(axis=-1) / np.linalg.norm(xtrain_tdr[:, :, 0].mean(axis=-1)) 
            r2 = xtrain_tdr[:, :, 1].mean(axis=-1) / np.linalg.norm(xtrain_tdr[:, :, 1].mean(axis=-1)) 
            cos_ddr_r1r2_train = np.abs(r1.dot(r2))

            # calculate the response magnitude for each stimulus
            _r = nat_preproc.fold_X(xtest, nreps=nreps_test, nstim=2, nbins=1).squeeze(3)
            r1mag_test = np.linalg.norm(_r[:, :, 0].mean(axis=-1)) 
            r2mag_test = np.linalg.norm(_r[:, :, 1].mean(axis=-1))
            _r = nat_preproc.fold_X(xtrain, nreps=nreps_train, nstim=2, nbins=1).squeeze(3)
            r1mag_train = np.linalg.norm(_r[:, :, 0].mean(axis=-1)) 
            r2mag_train = np.linalg.norm(_r[:, :, 1].mean(axis=-1))

            # pack results into dictionary to return
            results = {
                'dp_opt_test': tdr_dp_test, 
                'dp_diag_test': tdr_dp_test_diag,
                'wopt_test': tdr_wopt_test,
                'wdiag_test': tdr_wopt_test_diag,
                'var_explained_test': tdr_test_var,
                'evals_test': tdr_evals_test, 
                'evecs_test': tdr_evecs_test, 
                'evec_sim_test': evec_sim_test,
                'dU_test': tdr_dU_test,
                'dp_opt_train': tdr_dp_train, 
                'dp_diag_train': tdr_dp_train_diag, 
                'wopt_train': tdr_wopt_train, 
                'wdiag_train': tdr_wopt_train_diag, 
                'var_explained_train': tdr_train_var,
                'evals_train': tdr_evals_train, 
                'evecs_train': tdr_evecs_train, 
                'evec_sim_train': evec_sim_train,
                'dU_train': tdr_dU_train, 
                'dU_mag_test': dU_mag_test, 
                'dU_dot_evec_test': dU_dot_evec_test, 
                'cos_dU_wopt_test': cos_dU_wopt_test, 
                'dU_dot_evec_sq_test': dU_dot_evec_sq_test, 
                'evec_snr_test': evec_snr_test, 
                'cos_dU_evec_test': cos_dU_evec_test,
                'dU_mag_train': dU_mag_train, 
                'dU_dot_evec_train': dU_dot_evec_train, 
                'cos_dU_wopt_train': cos_dU_wopt_train, 
                'dU_dot_evec_sq_train': dU_dot_evec_sq_train, 
                'evec_snr_train': evec_snr_train, 
                'cos_dU_evec_train': cos_dU_evec_train,
                'dU_all': dU_all,
                'wopt_all': wopt_all,
                'evecs_all': evecs_all,
                'dU_all_test': dU_all_test,
                'evecs_all_test': evecs_all_test,
                'cos_r1r2_test': cos_r1r2_test,
                'cos_r1r2_train': cos_r1r2_train,
                'cos_ddr_r1r2_test': cos_ddr_r1r2_test,
                'cos_ddr_r1r2_train': cos_ddr_r1r2_train,
                'r1mag_train': r1mag_train,
                'r2mag_train': r2mag_train,
                'r1mag_test': r1mag_test,
                'r2mag_test': r2mag_test
            }

            if beta1 is not None:
                wopt_norm = wopt_all / np.linalg.norm(wopt_all)
                beta1_dU = abs(beta1.T.dot(tdr_weights[0, :])[0])
                try:
                    beta1_tdr2 = abs(beta1.T.dot(tdr_weights[1, :])[0])
                except:
                    beta1_tdr2 = np.nan
                beta1_wopt = abs(beta1.T.dot(wopt_norm)[0][0])
                beta1_tdr = beta1.T.dot(tdr_weights.T)   
                beta1_mag = np.linalg.norm(beta1_tdr) 
                beta1_tdr = beta1_tdr / beta1_mag

                # center xtest for each stim
                xcenter = xtest_tdr - xtest_tdr.mean(axis=1, keepdims=True)
                xcenter = xcenter.reshape(2+n_additional_axes, -1)    

                beta1_lambda = np.var(xcenter.T.dot(beta1_tdr.T)) # @ beta1_tdr)
                dU_dot_beta1_sq = tdr_dU_test.dot(beta1_tdr.T)[0][0]**2
                beta1_snr = dU_dot_beta1_sq / beta1_lambda

                dU_dot_beta1 = abs((tdr_dU_test / np.linalg.norm(tdr_dU_test)).dot(beta1_tdr.T))[0][0]

                results.update({
                    'beta1_dot_dU': beta1_dU,
                    'beta1_dot_tdr2': beta1_tdr2,
                    'beta1_dot_wopt': beta1_wopt,
                    'beta1_lambda': beta1_lambda,
                    'beta1_mag': beta1_mag,
                    'dU_dot_beta1_sq': dU_dot_beta1_sq,
                    'beta1_snr':  beta1_snr,
                    'cos_dU_beta1': dU_dot_beta1
                })  

            if beta2 is not None:
                wopt_norm = wopt_all / np.linalg.norm(wopt_all)
                beta2_dU = abs(beta2.T.dot(tdr_weights[0, :])[0])
                try:
                    beta2_tdr2 = abs(beta2.T.dot(tdr_weights[1, :])[0])
                except:
                    beta2_tdr2 = np.nan
                beta2_wopt = abs(beta2.T.dot(wopt_norm)[0][0])
                beta2_tdr = beta2.T.dot(tdr_weights.T)   
                beta2_mag = np.linalg.norm(beta2_tdr) 
                beta2_tdr = beta2_tdr / beta2_mag
                
                # center xtest for each stim
                xcenter = xtest_tdr - xtest_tdr.mean(axis=1, keepdims=True)
                xcenter = xcenter.reshape(2+n_additional_axes, -1)    

                beta2_lambda = np.var(xcenter.T.dot(beta2_tdr.T)) # @ beta2_tdr)
                dU_dot_beta2_sq = tdr_dU_test.dot(beta2_tdr.T)[0][0]**2
                beta2_snr = dU_dot_beta2_sq / beta2_lambda

                dU_dot_beta2 = abs((tdr_dU_test / np.linalg.norm(tdr_dU_test)).dot(beta2_tdr.T))[0][0]

                results.update({
                    'beta2_dot_dU': beta2_dU,
                    'beta2_dot_tdr2': beta2_tdr2,
                    'beta2_dot_wopt': beta2_wopt,
                    'beta2_lambda': beta2_lambda,
                    'beta2_mag': beta2_mag,
                    'dU_dot_beta2_sq': dU_dot_beta2_sq,
                    'beta2_snr':  beta2_snr,
                    'cos_dU_beta2': dU_dot_beta2
                })  
        else:
            # just doing dU decoding, so all the axes / alignement is meaningless.
            # just save dprime
            results = {
                'dp_opt_train': tdr_dp_train,
                'dp_opt_test': tdr_dp_test
            }

        # deal with large / small pupil data
        if ptrain_mask is not None:
            if fullpup:
                # perform analysis for big / small pupil data too. Using all data. So not a "true" jackknife.
                A_bp = fullX_tdr[:, ptest_mask[0, :, 0], 0]
                A_sp = fullX_tdr[:, ~ptest_mask[0, :, 0], 0]
                B_bp = fullX_tdr[:, ptest_mask[0, :, 1], 1]
                B_sp = fullX_tdr[:, ~ptest_mask[0, :, 1], 1]
            else:
                # perform analysis for big / small pupil data too. Only on test set.
                A_bp = xtest_tdr[:, ptest_mask[0, :, 0], 0]
                A_sp = xtest_tdr[:, ~ptest_mask[0, :, 0], 0]
                B_bp = xtest_tdr[:, ptest_mask[0, :, 1], 1]
                B_sp = xtest_tdr[:, ~ptest_mask[0, :, 1], 1]

            # get dprime / dU
            bp_dprime, _, _, _, _, bp_dU = compute_dprime(A_bp, B_bp, wopt=tdr_wopt_train)
            sp_dprime, _, _, _, _, sp_dU = compute_dprime(A_sp, B_sp, wopt=tdr_wopt_train)

            if n_additional_axes >= 0:
                # get pupil-dependent variance along the prinicple noise axes (analogous to lambda)
                # point is to compare the variance along these PCs between large / small pupil
                # NEED TO CENTER THE DATA FOR THIS!!! UPDATE 10-03-2020, CRH
                big = np.concatenate([A_bp - A_bp.mean(axis=1, keepdims=True), B_bp - B_bp.mean(axis=1, keepdims=True)], axis=-1)
                small = np.concatenate([A_sp - A_sp.mean(axis=1, keepdims=True), B_sp - B_sp.mean(axis=1, keepdims=True)], axis=-1)
                bp_lambda = np.var(big.T.dot(tdr_evecs_test), axis=0)
                sp_lambda = np.var(small.T.dot(tdr_evecs_test), axis=0)

                # compute additional metrics
                bp_dU_mag         = np.linalg.norm(bp_dU)
                bp_dU_dot_evec    = bp_dU.dot(tdr_evecs_test)
                bp_cos_dU_wopt    = abs(unit_vectors(bp_dU.T).T.dot(unit_vectors(tdr_wopt_train)))
                bp_dU_dot_evec_sq = bp_dU.dot(tdr_evecs_test) ** 2
                bp_evec_snr       = bp_dU_dot_evec_sq / bp_lambda
                bp_cos_dU_evec    = abs(unit_vectors(bp_dU.T).T.dot(tdr_evecs_test))

                sp_dU_mag         = np.linalg.norm(sp_dU)
                sp_dU_dot_evec    = sp_dU.dot(tdr_evecs_test)
                sp_cos_dU_wopt    = abs(unit_vectors(sp_dU.T).T.dot(unit_vectors(tdr_wopt_train)))
                sp_dU_dot_evec_sq = sp_dU.dot(tdr_evecs_test) ** 2
                sp_evec_snr       = sp_dU_dot_evec_sq / sp_lambda
                sp_cos_dU_evec    = abs(unit_vectors(sp_dU.T).T.dot(tdr_evecs_test))

                results.update({
                    'bp_dp': bp_dprime,
                    'bp_evals': bp_lambda,
                    'bp_dU_mag': bp_dU_mag,
                    'bp_dU_dot_evec': bp_dU_dot_evec,
                    'bp_cos_dU_wopt': bp_cos_dU_wopt,
                    'bp_dU_dot_evec_sq': bp_dU_dot_evec_sq,
                    'bp_evec_snr': bp_evec_snr,
                    'bp_cos_dU_evec': bp_cos_dU_evec,
                    'sp_dp': sp_dprime,
                    'sp_evals': sp_lambda,
                    'sp_dU_mag': sp_dU_mag,
                    'sp_dU_dot_evec': sp_dU_dot_evec,
                    'sp_cos_dU_wopt': sp_cos_dU_wopt,
                    'sp_dU_dot_evec_sq': sp_dU_dot_evec_sq,
                    'sp_evec_snr': sp_evec_snr,
                    'sp_cos_dU_evec': sp_cos_dU_evec
                })
            else:
                results.update({
                    'bp_dp': bp_dprime,
                    'sp_dp': sp_dprime
                })


        if not verbose:
            return results
        else:
            return results, tdr_weights


def do_tdr_dprime_analysis_loocv(X, nreps, tdr_data=None, n_additional_axes=0,
                                    beta1=None, beta2=None, tdr2_axis=None, 
                                    pmask=None, sim1=False, sim2=False, sim12=False, 
                                    verbose=False):
        """
        perform TDR (custom dim reduction): project into 2D space defined by dU and first noise PC
        compute dprime and associated metrics
        return results in a dictionary

        Same as above function in spirit, but no train/test set here. Instead, use leave one out cross-val
        """

        if sim1 | sim2 | sim12:
            raise NotImplementedError("WIP - simulations not yet implemented for loocv")
            # simulate data. If pupil mask is specified, use this to created simulated trials.
            if ptrain_mask is not None:
                xtest_sim = nat_preproc.fold_X(xtest, nreps=nreps_test, nstim=2, nbins=1)
                pmask_sim = ptest_mask[:, :, :, np.newaxis]
                x_sim = []
                pup_mask_sim = []
                for s in range(xtest_sim.shape[2]):
                    _x_sim, _pup_mask_sim = simulate_response(xtest_sim[:, :, [s], :], pmask_sim[:, :, [s], :], 
                                                                            sim_first_order=sim1,
                                                                            sim_second_order=sim2,
                                                                            sim_all=sim12,
                                                                            nreps=5000,
                                                                            suppress_log=True)
                    x_sim.append(_x_sim)
                    pup_mask_sim.append(_pup_mask_sim)
                x_sim = np.concatenate(x_sim, axis=2)
                pup_mask_sim = np.concatenate(pup_mask_sim, axis=2)
                # pull out simulated test set, that's balanced over pupil conditions
                # do this by taking every other trial
                # only simulate the "test" set. This way decoding axis always the same as raw data analysis
                xtest = x_sim[:, ::2, :, 0]
                nreps_test = xtest.shape[1]
                xtest = xtest.reshape(xtest.shape[0], -1)
                ptest_mask = pup_mask_sim[:, ::2, :, 0]
                
            else:
                raise NotImplementedError("Can't do simulations without specifying a pupil mask. TODO: update decoding.simulate_response to handle this")
        

        # initalize arrays to hold results for each loocv step
        ndim = 2
        if n_additional_axes is not None: ndim += n_additional_axes
        # overall results
        tdr_dp         = np.zeros(nreps)
        tdr_dp_diag    = np.zeros(nreps)
        tdr_wopt       = np.zeros((ndim, nreps))
        tdr_evals      = np.zeros((ndim, nreps))
        tdr_evecs      = np.zeros((ndim, ndim, nreps))
        evec_sim       = np.zeros(nreps)
        tdr_dU         = np.zeros((ndim, nreps))
        dU_mag         = np.zeros(nreps)
        dU_dot_evec    = np.zeros((ndim, nreps))
        cos_dU_wopt    = np.zeros(nreps)
        dU_dot_evec_sq = np.zeros((ndim, nreps))
        evec_snr       = np.zeros((ndim, nreps))
        cos_dU_evec    = np.zeros((ndim, nreps)) 
        dU_all         = np.zeros((X.shape[0], nreps))
        wopt_all       = np.zeros((X.shape[0], nreps))
        evecs_all      = np.zeros((X.shape[0], ndim, nreps))

        # beta1 results
        # not currently using these, don't save...

        # beta2 results
        beta2_dU        = np.zeros(nreps)
        beta2_tdr2      = np.zeros(nreps)
        beta2_wopt      = np.zeros(nreps)
        beta2_tdr       = np.zeros((ndim, nreps))
        beta2_mag       = np.zeros(nreps)
        beta2_lambda    = np.zeros(nreps)
        dU_dot_beta2_sq = np.zeros(nreps)
        beta2_snr       = np.zeros(nreps)
        dU_dot_beta2    = np.zeros(nreps)

        # big/small pupil results
        bp_dU             = np.zeros((ndim, nreps))
        sp_dU             = np.zeros((ndim, nreps))
        bp_lambda         = np.zeros((ndim, nreps))
        sp_lambda         = np.zeros((ndim, nreps)) 
        bp_dU_mag         = np.zeros(nreps) 
        bp_dU_dot_evec    = np.zeros((ndim, nreps))  
        bp_cos_dU_wopt    = np.zeros(nreps)  
        bp_dU_dot_evec_sq = np.zeros((ndim, nreps)) 
        bp_evec_snr       = np.zeros((ndim, nreps)) 
        bp_cos_dU_evec    = np.zeros((ndim, nreps))  
        sp_dU_mag         = np.zeros(nreps) 
        sp_dU_dot_evec    = np.zeros((ndim, nreps))
        sp_cos_dU_wopt    = np.zeros(nreps)  
        sp_dU_dot_evec_sq = np.zeros((ndim, nreps))  
        sp_evec_snr       = np.zeros((ndim, nreps)) 
        sp_cos_dU_evec    = np.zeros((ndim, nreps))  

        # init arrays to hold projected points for "test" dprime calculations
        popt = np.zeros((2, nreps))
        pdiag = np.zeros((2, nreps))

        # For big/small, will remove nans for these, since each left out trial can be large or small, not both.
        popt_bp  = np.nan * np.zeros((2, nreps))
        popt_sp  = np.nan * np.zeros((2, nreps))
        pdiag_bp = np.nan * np.zeros((2, nreps))
        pdiag_sp = np.nan * np.zeros((2, nreps))

        xfold = nat_preproc.fold_X(X, nreps=nreps, nstim=2, nbins=1).squeeze()
        for tr in np.arange(0, xfold.shape[1]):
            trials = np.array([i for i in np.arange(0, xfold.shape[1]) if i!=tr])
            _xfold = xfold[:, trials, :]
            _X = nat_preproc.flatten_X(_xfold[:, :, :, np.newaxis])
            tdr = dr.TDR(tdr2_init=tdr2_axis, n_additional_axes=n_additional_axes)

            if tdr_data is None:
                Y = dr.get_one_hot_matrix(ncategories=2, nreps=nreps-1)
                tdr.fit(_X.T, Y.T)
            else:
                Y = dr.get_one_hot_matrix(ncategories=2, nreps=tdr_data[1]-1)
                tdr.fit(tdr_data[:, trials, :].T, Y.T)
            tdr_weights = tdr.weights

            x_tdr = (_X.T @ tdr_weights.T).T

            x_tdr = nat_preproc.fold_X(x_tdr, nreps=nreps-1, nstim=2, nbins=1).squeeze()

            # perform decoding analysis (this is so that decoding axis is always defined 
            # using the raw data, not simulated data)

            # for each trial (rep) do analysis
            tdr_var = np.var(x_tdr.T @ tdr_weights)  / np.var(_X)

            tdr_dp[tr], tdr_wopt[:, [tr]], tdr_evals[:, tr], tdr_evecs[:, :, tr], evec_sim[tr], tdr_dU[:, tr] = \
                                    compute_dprime(x_tdr[:, :, 0], x_tdr[:, :, 1])

            # compute dprime metrics diag decoder
            tdr_dp_diag[tr], tdr_diag, _, _, _, x = \
                                    compute_dprime(x_tdr[:, :, 0], x_tdr[:, :, 1], diag=True)

            # now, project left out test point into TDR space and onto decoding axes
            popt[:, tr] = [(xfold[:, [tr], 0].T @ tdr.weights.T).dot(tdr_wopt[:, [tr]])[0][0], \
                                   (xfold[:, [tr], 1].T @ tdr.weights.T).dot(tdr_wopt[:, [tr]])[0][0]]
            pdiag[:, tr] = [(xfold[:, [tr], 0].T @ tdr.weights.T).dot(tdr_diag)[0][0], \
                                   (xfold[:, [tr], 1].T @ tdr.weights.T).dot(tdr_diag)[0][0]]
            # caculate additional metrics
            dU_mag[tr]            = np.linalg.norm(tdr_dU[:, tr])
            dU_dot_evec[:, tr]    = tdr_dU[:, tr].dot(tdr_evecs[:, :, tr])
            cos_dU_wopt[tr]       = abs(unit_vectors(tdr_dU[:, [tr]]).T @ unit_vectors(tdr_wopt[:, [tr]]))[0][0]
            dU_dot_evec_sq[:, tr] = tdr_dU[:, tr].dot(tdr_evecs[:, :, tr]) ** 2
            evec_snr[:, tr]       = dU_dot_evec_sq[:, tr] / tdr_evals[:, tr]
            cos_dU_evec[:, tr]    = abs(unit_vectors(tdr_dU[:, [tr]]).T @ tdr_evecs[:, :, tr])

            # project eigenvectors, dU, and wopt back into N-dimensional space
            # in order to investigate which neurons contribute to signal vs. noise
            dU_all[:, tr] = tdr_dU[:, tr].dot(tdr.weights)
            wopt_all[:, tr] = tdr_wopt[:, tr].T.dot(tdr.weights).T
            evecs_all[:, :, tr] = tdr_evecs[:, :, tr].dot(tdr.weights).T

            if beta1 is not None:
                pass
                '''
                wopt_norm = wopt_all[:, tr] / np.linalg.norm(wopt_all[:, tr])
                beta1_dU = abs(beta1.T.dot(tdr_weights[0, :])[0])
                beta1_tdr2 = abs(beta1.T.dot(tdr_weights[1, :])[0])
                beta1_wopt = abs(beta1.T.dot(wopt_norm)[0][0])
                beta1_tdr = beta1.T.dot(tdr_weights.T)   
                beta1_mag = np.linalg.norm(beta1_tdr) 
                beta1_tdr = beta1_tdr / beta1_mag

                # center xtest for each stim
                xcenter = x_tdr - x_tdr.mean(axis=1, keepdims=True)
                xcenter = xcenter.reshape(2+n_additional_axes, -1)    

                beta1_lambda = np.var(xcenter.T.dot(beta1_tdr.T)) # @ beta1_tdr)
                dU_dot_beta1_sq = tdr_dU[:, tr].dot(beta1_tdr.T)[0][0]**2
                beta1_snr = dU_dot_beta1_sq / beta1_lambda

                dU_dot_beta1 = abs((tdr_dU[:, tr] / np.linalg.norm(tdr_dU[:, tr])).dot(beta1_tdr.T))[0][0]
                '''
            if beta2 is not None:
                wopt_norm = wopt_all[:, tr] / np.linalg.norm(wopt_all[:, tr])
                beta2_dU[tr] = abs(beta2.T.dot(tdr_weights[0, :])[0])
                beta2_tdr2[tr] = abs(beta2.T.dot(tdr_weights[1, :])[0])
                beta2_wopt[tr] = abs(beta2.T.dot(wopt_norm)[0])
                _beta2_tdr = beta2.T.dot(tdr_weights.T)   
                beta2_mag[tr] = np.linalg.norm(_beta2_tdr) 
                beta2_tdr[:, tr] = _beta2_tdr / beta2_mag[tr]
                
                # center xtest for each stim
                xcenter = x_tdr - x_tdr.mean(axis=1, keepdims=True)
                xcenter = xcenter.reshape(2+n_additional_axes, -1)    

                beta2_lambda[tr] = np.var(xcenter.T.dot(beta2_tdr[:, tr].T)) # @ beta2_tdr)
                dU_dot_beta2_sq[tr] = tdr_dU[:, tr].dot(beta2_tdr[:, tr].T)**2
                beta2_snr[tr] = dU_dot_beta2_sq[tr] / beta2_lambda[tr]

                dU_dot_beta2[tr] = abs((tdr_dU[:, tr] / np.linalg.norm(tdr_dU[:, tr])).dot(beta2_tdr[:, tr].T))

            # deal with large / small pupil data
            if pmask is not None:
                # perform analysis for big / small pupil data too.

                # use the "train" data in order to estimate angles etc. in diff pupil states.
                # For dprime, only need to project the left out point onto the axis and label as big / small
                A_bp = x_tdr[:, pmask[0, trials, 0], 0]
                A_sp = x_tdr[:, ~pmask[0, trials, 0], 0]
                B_bp = x_tdr[:, pmask[0, trials, 1], 1]
                B_sp = x_tdr[:, ~pmask[0, trials, 1], 1]

                # get dprime / dU
                _, _, _, _, _, bp_dU[:, tr] = compute_dprime(A_bp, B_bp, wopt=tdr_wopt[:, [tr]])
                _, _, _, _, _, sp_dU[:, tr] = compute_dprime(A_sp, B_sp, wopt=tdr_wopt[:, [tr]])

                # project single point
                if pmask[0, tr, 0]:
                    popt_bp[:, tr] = popt[:, tr]
                    pdiag_bp[:, tr] = pdiag[:, tr]
                else:
                    popt_sp[:, tr] = popt[:, tr]
                    pdiag_sp[:, tr] = pdiag[:, tr]

                # get pupil-dependent variance along the prinicple noise axes (analogous to lambda)
                # point is to compare the variance along these PCs between large / small pupil
                big = np.concatenate([A_bp, B_bp], axis=-1)
                small = np.concatenate([A_sp, B_sp], axis=-1)
                bp_lambda[:, tr] = np.var(big.T.dot(tdr_evecs[:, :, tr]), axis=0)
                sp_lambda[:, tr] = np.var(small.T.dot(tdr_evecs[:, :, tr]), axis=0)

                # compute additional metrics
                bp_dU_mag[tr]            = np.linalg.norm(bp_dU[:, tr])
                bp_dU_dot_evec[:, tr]    = bp_dU[:, tr].dot(tdr_evecs[:, :, tr])
                bp_cos_dU_wopt[tr]       = abs(unit_vectors(bp_dU[:, [tr]]).T.dot(unit_vectors(tdr_wopt[:, [tr]])))[0][0]
                bp_dU_dot_evec_sq[:, tr] = bp_dU[:, tr].dot(tdr_evecs[:, :, tr]) ** 2
                bp_evec_snr[:, tr]       = bp_dU_dot_evec_sq[:, tr] / bp_lambda[:, tr]
                bp_cos_dU_evec[:, tr]    = abs(unit_vectors(bp_dU[:, [tr]]).T.dot(tdr_evecs[:, :, tr]))

                sp_dU_mag[tr]            = np.linalg.norm(sp_dU[:, tr])
                sp_dU_dot_evec[:, tr]    = sp_dU[:, tr].dot(tdr_evecs[:, :, tr])
                sp_cos_dU_wopt[tr]       = abs(unit_vectors(sp_dU[:, [tr]]).T.dot(unit_vectors(tdr_wopt[:, [tr]])))[0][0]
                sp_dU_dot_evec_sq[:, tr] = sp_dU[:, tr].dot(tdr_evecs[:, :, tr]) ** 2
                sp_evec_snr[:, tr]       = sp_dU_dot_evec_sq[:, tr] / sp_lambda[:, tr]
                sp_cos_dU_evec[:, tr]    = abs(unit_vectors(sp_dU[:, [tr]]).T.dot(tdr_evecs[:, :, tr]))

        # compute test dprimes using the projected left out points from each method
        dptest = ((np.mean(popt[0, :]) - np.mean(popt[1, :])) / np.sqrt(np.mean([np.var(popt[0, :]), np.var(popt[1, :])]))) ** 2
        dptest_diag = ((np.mean(pdiag[0, :]) - np.mean(pdiag[1, :])) / np.sqrt(np.mean([np.var(pdiag[0, :]), np.var(pdiag[1, :])]))) ** 2

        # pack results into dictionary to return
        results = {
            'dp_opt_test': dptest, 
            'dp_diag_test': dptest_diag,
            'wopt_test': np.mean(tdr_wopt, axis=-1),
            'evals_test': np.mean(tdr_evals, axis=-1), 
            'evecs_test': np.mean(tdr_evecs, axis=-1), 
            'evec_sim_test': np.mean(evec_sim),
            'dU_test': np.mean(tdr_dU, axis=-1), 
            'dp_opt_train': np.mean(tdr_dp), 
            'dp_diag_train': np.mean(tdr_dp_diag), 
            'wopt_train': np.mean(tdr_wopt, axis=-1), 
            'evals_train': np.mean(tdr_evals, axis=-1), 
            'evecs_train': np.mean(tdr_evecs, axis=-1), 
            'evec_sim_train': np.mean(evec_sim),
            'dU_train': np.mean(tdr_dU, axis=-1), 
            'dU_mag_test': np.mean(dU_mag), 
            'dU_dot_evec_test': np.mean(dU_dot_evec, axis=-1), 
            'cos_dU_wopt_test': np.mean(cos_dU_wopt, axis=-1), 
            'dU_dot_evec_sq_test': np.mean(dU_dot_evec_sq, axis=-1), 
            'evec_snr_test': np.mean(evec_snr, axis=-1), 
            'cos_dU_evec_test': np.mean(cos_dU_evec, axis=-1), 
            'dU_mag_train': np.mean(dU_mag, axis=-1),  
            'dU_dot_evec_train': np.mean(dU_dot_evec, axis=-1),  
            'cos_dU_wopt_train': np.mean(cos_dU_wopt, axis=-1),   
            'dU_dot_evec_sq_train': np.mean(dU_dot_evec_sq, axis=-1),  
            'evec_snr_train': np.mean(evec_snr, axis=-1),  
            'cos_dU_evec_train': np.mean(cos_dU_evec, axis=-1),
            'dU_all': np.mean(dU_all, axis=-1),
            'wopt_all': np.mean(wopt_all, axis=-1),
            'evecs_all': np.mean(evecs_all, axis=-1),
            'dU_all_test': np.mean(dU_all, axis=-1), 
            'evecs_all_test': np.mean(evecs_all, axis=-1),
        }

        if beta1 is not None:
            pass
            #results.update({
            #    'beta1_dot_dU': beta1_dU,
            #    'beta1_dot_tdr2': beta1_tdr2,
            #    'beta1_dot_wopt': beta1_wopt,
            #    'beta1_lambda': beta1_lambda,
            #    'beta1_mag': beta1_mag,
            #    'dU_dot_beta1_sq': dU_dot_beta1_sq,
            #    'beta1_snr':  beta1_snr,
            #    'cos_dU_beta1': dU_dot_beta1
            #})  

        if beta2 is not None:
            results.update({
                'beta2_dot_dU': np.mean(beta2_dU, axis=-1),
                'beta2_dot_tdr2': np.mean(beta2_tdr2, axis=-1),
                'beta2_dot_wopt': np.mean(beta2_wopt, axis=-1),
                'beta2_lambda': np.mean(beta2_lambda, axis=-1),
                'beta2_mag': np.mean(beta2_mag, axis=-1),
                'dU_dot_beta2_sq': np.mean(dU_dot_beta2_sq, axis=-1),
                'beta2_snr':  np.mean(beta2_snr, axis=-1),
                'cos_dU_beta2': np.mean(dU_dot_beta2, axis=-1)
            })  
        
        if pmask is not None:
            bp_dp = ((np.mean(popt_bp[0, ~np.isnan(popt_bp[0, :])]) - np.mean(popt_bp[1, ~np.isnan(popt_bp[0, :])])) / \
                    np.sqrt(np.mean([np.var(popt_bp[0, ~np.isnan(popt_bp[0, :])]), np.var(popt_bp[1, ~np.isnan(popt_bp[0, :])])]))) ** 2
            bp_dp_diag = ((np.mean(pdiag_bp[0, ~np.isnan(popt_bp[0, :])]) - np.mean(pdiag_bp[1, ~np.isnan(popt_bp[0, :])])) / \
                    np.sqrt(np.mean([np.var(pdiag_bp[0, ~np.isnan(popt_bp[0, :])]), np.var(pdiag_bp[1, ~np.isnan(popt_bp[0, :])])]))) ** 2
            sp_dp = ((np.mean(popt_sp[0, ~np.isnan(popt_sp[0, :])]) - np.mean(popt_sp[1, ~np.isnan(popt_sp[0, :])])) / \
                    np.sqrt(np.mean([np.var(popt_sp[0, ~np.isnan(popt_sp[0, :])]), np.var(popt_sp[1, ~np.isnan(popt_sp[0, :])])]))) ** 2
            sp_dp_diag = ((np.mean(pdiag_sp[0, ~np.isnan(popt_sp[0, :])]) - np.mean(pdiag_sp[1, ~np.isnan(popt_sp[0, :])])) / \
                    np.sqrt(np.mean([np.var(pdiag_sp[0, ~np.isnan(popt_sp[0, :])]), np.var(pdiag_sp[1, ~np.isnan(popt_sp[0, :])])]))) ** 2
            results.update({
                'bp_dp': bp_dp,
                'bp_dp_diag': bp_dp_diag,
                'bp_evals': np.mean(bp_lambda, axis=-1),
                'bp_dU_mag': np.mean(bp_dU_mag, axis=-1),
                'bp_dU_dot_evec': np.mean(bp_dU_dot_evec, axis=-1),
                'bp_cos_dU_wopt': np.mean(bp_cos_dU_wopt, axis=-1),
                'bp_dU_dot_evec_sq': np.mean(bp_dU_dot_evec_sq, axis=-1),
                'bp_evec_snr': np.mean(bp_evec_snr, axis=-1),
                'bp_cos_dU_evec': np.mean(bp_cos_dU_evec, axis=-1),
                'sp_dp': sp_dp,
                'sp_dp_diag': sp_dp_diag,
                'sp_evals': np.mean(sp_lambda, axis=-1),
                'sp_dU_mag': np.mean(sp_dU_mag, axis=-1),
                'sp_dU_dot_evec': np.mean(sp_dU_dot_evec, axis=-1),
                'sp_cos_dU_wopt': np.mean(sp_cos_dU_wopt, axis=-1),
                'sp_dU_dot_evec_sq': np.mean(sp_dU_dot_evec_sq, axis=-1),
                'sp_evec_snr': np.mean(sp_evec_snr, axis=-1),
                'sp_cos_dU_evec': np.mean(sp_cos_dU_evec, axis=-1)
            })

        if not verbose:
            return results
        else:
            return results, tdr_weights


def do_pca_dprime_analysis(xtrain, xtest, nreps_train, nreps_test, ptrain_mask=None, ptest_mask=None):
        """
        perform pca on the trail averaged response, project into 2D space, compute dprime and associated metrics
        return results in a dictionary
        """
        
        # perform trial averaged PCA (bc pairwise comparison, can only do PCA dim = 2)
        ncells = xtrain.shape[0]
        xfold = nat_preproc.fold_X(xtrain, nreps=nreps_train, nstim=2, nbins=1)
        xtrain_trial_average = xfold.mean(axis=1)[:, np.newaxis, :, np.newaxis]
        xtrain_trial_average = xtrain_trial_average.reshape(ncells, -1)

        pca = PCA(n_components=2)
        pca.fit(xtrain_trial_average.T)
        pca_weights = pca.components_

        xtrain_pca = (xtrain.T @ pca_weights.T).T
        xtest_pca = (xtest.T @ pca_weights.T).T

        xtrain_pca = nat_preproc.fold_X(xtrain_pca, nreps=nreps_train, nstim=2, nbins=1).squeeze()
        xtest_pca = nat_preproc.fold_X(xtest_pca, nreps=nreps_test, nstim=2, nbins=1).squeeze()

        pca_train_var = np.var(xtrain_pca.T @ pca_weights)  / np.var(xtrain)
        pca_test_var = np.var(xtest_pca.T @ pca_weights)  / np.var(xtest)

        # compute dprime metrics raw 
        pca_dp_train, pca_wopt_train, pca_evals_train, pca_evecs_train, evec_sim_train, pca_dU_train = \
                                compute_dprime(xtrain_pca[:, :, 0], xtrain_pca[:, :, 1])
        pca_dp_test, pca_wopt_test, pca_evals_test, pca_evecs_test, evec_sim_test, pca_dU_test = \
                                compute_dprime(xtest_pca[:, :, 0], xtest_pca[:, :, 1], wopt=pca_wopt_train)

        # overwrite test decoder with training, since it's fixed
        pca_wopt_test = pca_wopt_train

        # compute dprime metrics diag decoder
        pca_dp_train_diag, pca_wopt_train_diag, _, _, _, x = \
                                compute_dprime(xtrain_pca[:, :, 0], xtrain_pca[:, :, 1], diag=True)
        pca_dp_test_diag, pca_wopt_test_diag, _, _, _, _ = \
                                compute_dprime(xtest_pca[:, :, 0], xtest_pca[:, :, 1], diag=True)

        # caculate additional metrics
        dU_mag_train         = np.linalg.norm(pca_dU_train)
        dU_dot_evec_train    = pca_dU_train.dot(pca_evecs_train)
        cos_dU_wopt_train    = abs(unit_vectors(pca_dU_train.T).T.dot(unit_vectors(pca_wopt_train)))
        dU_dot_evec_sq_train = pca_dU_train.dot(pca_evecs_train) ** 2
        evec_snr_train       = dU_dot_evec_sq_train / pca_evals_train
        cos_dU_evec_train    = abs(unit_vectors(pca_dU_train.T).T.dot(pca_evecs_train))

        dU_mag_test         = np.linalg.norm(pca_dU_test)
        dU_dot_evec_test    = pca_dU_test.dot(pca_evecs_test)
        cos_dU_wopt_test    = abs(unit_vectors(pca_dU_test.T).T.dot(unit_vectors(pca_wopt_test)))
        dU_dot_evec_sq_test = pca_dU_test.dot(pca_evecs_test) ** 2
        evec_snr_test       = dU_dot_evec_sq_test / pca_evals_test
        cos_dU_evec_test    = abs(unit_vectors(pca_dU_test.T).T.dot(pca_evecs_test))

        # pack results into dictionary to return
        results = {
            'dp_opt_test': pca_dp_test, 
            'dp_diag_test': pca_dp_test_diag,
            'wopt_test': pca_wopt_test,
            'wdiag_test': pca_wopt_test_diag,
            'var_explained_test': pca_test_var,
            'evals_test': pca_evals_test, 
            'evecs_test': pca_evecs_test, 
            'dU_test': pca_dU_test,
            'dp_opt_train': pca_dp_train, 
            'dp_diag_train': pca_dp_train_diag, 
            'wopt_train': pca_wopt_train, 
            'wdiag_train': pca_wopt_train_diag, 
            'var_explained_train': pca_train_var,
            'evals_train': pca_evals_train, 
            'evecs_train': pca_evecs_train, 
            'dU_train': pca_dU_train, 
            'dU_mag_test': dU_mag_test, 
            'dU_dot_evec_test': dU_dot_evec_test, 
            'cos_dU_wopt_test': cos_dU_wopt_test, 
            'dU_dot_evec_sq_test': dU_dot_evec_sq_test, 
            'evec_snr_test': evec_snr_test, 
            'cos_dU_evec_test': cos_dU_evec_test,
            'dU_mag_train': dU_mag_train, 
            'dU_dot_evec_train': dU_dot_evec_train, 
            'cos_dU_wopt_train': cos_dU_wopt_train, 
            'dU_dot_evec_sq_train': dU_dot_evec_sq_train, 
            'evec_snr_train': evec_snr_train, 
            'cos_dU_evec_train': cos_dU_evec_train
        }

        # deal with large / small pupil data
        if ptrain_mask is not None:
            # perform analysis for big / small pupil data too. Only on test set.
            A_bp = xtest_pca[:, ptest_mask[0, :, 0], 0]
            A_sp = xtest_pca[:, ~ptest_mask[0, :, 0], 0]
            B_bp = xtest_pca[:, ptest_mask[0, :, 1], 1]
            B_sp = xtest_pca[:, ~ptest_mask[0, :, 1], 1]

            # get dprime / dU
            bp_dprime, _, _, _, _, bp_dU = compute_dprime(A_bp, B_bp, wopt=pca_wopt_train)
            sp_dprime, _, _, _, _, sp_dU = compute_dprime(A_sp, B_sp, wopt=pca_wopt_train)

            # get pupil-dependent variance along the prinicple noise axes (analogous to lambda)
            # point is to compare the variance along these PCs between large / small pupil
            big = np.concatenate([A_bp, B_bp], axis=-1)
            small = np.concatenate([A_sp, B_sp], axis=-1)
            bp_lambda = np.var(big.T.dot(pca_evecs_test), axis=0)
            sp_lambda = np.var(small.T.dot(pca_evecs_test), axis=0)

            # compute additional metrics
            bp_dU_mag         = np.linalg.norm(bp_dU)
            bp_dU_dot_evec    = bp_dU.dot(pca_evecs_test)
            bp_cos_dU_wopt    = abs(unit_vectors(bp_dU.T).T.dot(unit_vectors(pca_wopt_train)))
            bp_dU_dot_evec_sq = bp_dU.dot(pca_evecs_test) ** 2
            bp_evec_snr       = bp_dU_dot_evec_sq / bp_lambda
            bp_cos_dU_evec    = abs(unit_vectors(bp_dU.T).T.dot(pca_evecs_test))

            sp_dU_mag         = np.linalg.norm(sp_dU)
            sp_dU_dot_evec    = sp_dU.dot(pca_evecs_test)
            sp_cos_dU_wopt    = abs(unit_vectors(sp_dU.T).T.dot(unit_vectors(pca_wopt_train)))
            sp_dU_dot_evec_sq = sp_dU.dot(pca_evecs_test) ** 2
            sp_evec_snr       = sp_dU_dot_evec_sq / sp_lambda
            sp_cos_dU_evec    = abs(unit_vectors(sp_dU.T).T.dot(pca_evecs_test))

            results.update({
                'bp_dp': bp_dprime,
                'bp_evals': bp_lambda,
                'bp_dU_mag': bp_dU_mag,
                'bp_dU_dot_evec': bp_dU_dot_evec,
                'bp_cos_dU_wopt': bp_cos_dU_wopt,
                'bp_dU_dot_evec_sq': bp_dU_dot_evec_sq,
                'bp_evec_snr': bp_evec_snr,
                'bp_cos_dU_evec': bp_cos_dU_evec,
                'sp_dp': sp_dprime,
                'sp_evals': sp_lambda,
                'sp_dU_mag': sp_dU_mag,
                'sp_dU_dot_evec': sp_dU_dot_evec,
                'sp_cos_dU_wopt': sp_cos_dU_wopt,
                'sp_dU_dot_evec_sq': sp_dU_dot_evec_sq,
                'sp_evec_snr': sp_evec_snr,
                'sp_cos_dU_evec': sp_cos_dU_evec
            })

        return results


def do_pls_dprime_analysis(xtrain, xtest, nreps_train, nreps_test, ptrain_mask=None, ptest_mask=None, n_components=2):
        """
        perform pca on the trail averaged response, project into 2D space, compute dprime and associated metrics
        return results in a dictionary
        """
        try:
            Y = dr.get_one_hot_matrix(ncategories=2, nreps=nreps_train)
            pls = PLSRegression(n_components=n_components, max_iter=500, tol=1e-7)
            pls.fit(xtrain.T, Y.T)
            pls_weights = pls.x_weights_
            nan_pad = False
        except np.linalg.LinAlgError:
            # deflated matrix on this iteration of NIPALS was ~0. e.g. the overall matrix rank may have been 6, but
            # by the time it gets to this iteration, the last couple of indpendent dims are so small, that matrix
            # is essentially 0 and PLS can't converge.
            log.info("PLS can't converge. No more dimensions in the deflated matrix. Pad with nan and continue. \n"
                            "jack_idx: {0} \n"
                            "n_components: {1} \n "
                            "stim category: {2} \n "
                            "stim combo: {3}".format(ev_set, n_components, category, combo))
            nan_pad = True

        if not nan_pad:
            xtrain_pls = (xtrain.T @ pls_weights).T
            xtest_pls = (xtest.T @ pls_weights).T

            if np.linalg.matrix_rank(xtrain_pls) < n_components:
                # add one more check - this will cause singular matrix. dprime fn handles this,
                # but to prevent a barrage of log messages, check here to prevent even attempting dprime calc.
                # what this means is that the last dim(s) of x_weights are 0. i.e. there is no more explainable 
                # information about Y in X.
                nan_pad = True

            else:
                nan_pad = False
                xtrain_pls = nat_preproc.fold_X(xtrain_pls, nreps=nreps_train, nstim=2, nbins=1).squeeze()
                xtest_pls = nat_preproc.fold_X(xtest_pls, nreps=nreps_test, nstim=2, nbins=1).squeeze()
    
                pls_train_var = np.var(xtrain_pls.T @ pls_weights.T)  / np.var(xtrain)
                pls_test_var = np.var(xtest_pls.T @ pls_weights.T)  / np.var(xtest)

                # compute dprime metrics raw 
                pls_dp_train, pls_wopt_train, pls_evals_train, pls_evecs_train, pls_dU_train = \
                                        compute_dprime(xtrain_pls[:, :, 0], xtrain_pls[:, :, 1])
                pls_dp_test, pls_wopt_test, pls_evals_test, pls_evecs_test, pls_dU_test = \
                                        compute_dprime(xtest_pls[:, :, 0], xtest_pls[:, :, 1], wopt=pls_wopt_train)

                # override test decoder as train decoder
                pls_wopt_test = pls_wopt_train

                # compute dprime metrics diag decoder
                pls_dp_train_diag, pls_wopt_train_diag, _, _, _ = \
                                        compute_dprime(xtrain_pls[:, :, 0], xtrain_pls[:, :, 1], diag=True)
                pls_dp_test_diag, pls_wopt_test_diag, _, _, _ = \
                                        compute_dprime(xtest_pls[:, :, 0], xtest_pls[:, :, 1], diag=True)

                # caculate additional metrics
                dU_mag_train         = np.linalg.norm(pls_dU_train)
                dU_dot_evec_train    = pls_dU_train.dot(pls_evecs_train)
                cos_dU_wopt_train    = abs(unit_vectors(pls_dU_train.T).T.dot(unit_vectors(pls_wopt_train)))
                dU_dot_evec_sq_train = pls_dU_train.dot(pls_evecs_train) ** 2
                evec_snr_train       = dU_dot_evec_sq_train / pls_evals_train
                cos_dU_evec_train    = abs(unit_vectors(pls_dU_train.T).T.dot(pls_evecs_train))

                dU_mag_test         = np.linalg.norm(pls_dU_test)
                dU_dot_evec_test    = pls_dU_test.dot(pls_evecs_test)
                cos_dU_wopt_test    = abs(unit_vectors(pls_dU_test.T).T.dot(unit_vectors(pls_wopt_test)))
                dU_dot_evec_sq_test = pls_dU_test.dot(pls_evecs_test) ** 2
                evec_snr_test       = dU_dot_evec_sq_test / pls_evals_test
                cos_dU_evec_test    = abs(unit_vectors(pls_dU_test.T).T.dot(pls_evecs_test))

                # pack results into dictionary to return
                results = {
                    'dp_opt_test': pls_dp_test, 
                    'dp_diag_test': pls_dp_test_diag,
                    'wopt_test': pls_wopt_test,
                    'wdiag_test': pls_wopt_test_diag,
                    'var_explained_test': pls_test_var,
                    'evals_test': pls_evals_test, 
                    'evecs_test': pls_evecs_test, 
                    'dU_test': pls_dU_test,
                    'dp_opt_train': pls_dp_train, 
                    'dp_diag_train': pls_dp_train_diag, 
                    'wopt_train': pls_wopt_train, 
                    'wdiag_train': pls_wopt_train_diag, 
                    'var_explained_train': pls_train_var,
                    'evals_train': pls_evals_train, 
                    'evecs_train': pls_evecs_train, 
                    'dU_train': pls_dU_train, 
                    'dU_mag_test': dU_mag_test, 
                    'dU_dot_evec_test': dU_dot_evec_test, 
                    'cos_dU_wopt_test': cos_dU_wopt_test, 
                    'dU_dot_evec_sq_test': dU_dot_evec_sq_test, 
                    'evec_snr_test': evec_snr_test, 
                    'cos_dU_evec_test': cos_dU_evec_test,
                    'dU_mag_train': dU_mag_train, 
                    'dU_dot_evec_train': dU_dot_evec_train, 
                    'cos_dU_wopt_train': cos_dU_wopt_train, 
                    'dU_dot_evec_sq_train': dU_dot_evec_sq_train, 
                    'evec_snr_train': evec_snr_train, 
                    'cos_dU_evec_train': cos_dU_evec_train
                }


        if nan_pad:
            evec_nan = np.nan * np.ones((n_components, n_components))
            eval_nan = np.nan * np.ones(n_components)
            dU_nan = np.nan * np.ones((1, n_components))
            wopt_nan = np.nan * np.ones((n_components, 1)) 

            # pack results into dictionary to return
            results = {
                'dp_opt_test': np.nan, 
                'dp_diag_test': np.nan,
                'wopt_test': wopt_nan,
                'wdiag_test': wopt_nan,
                'var_explained_test': np.nan,
                'evals_test': eval_nan, 
                'evecs_test': evec_nan, 
                'dU_test': dU_nan,
                'dp_opt_train': np.nan, 
                'dp_diag_train': np.nan, 
                'wopt_train': wopt_nan, 
                'wdiag_train': wopt_nan, 
                'var_explained_train': np.nan,
                'evals_train': eval_nan, 
                'evecs_train': evec_nan, 
                'dU_train': dU_nan, 
                'dU_mag_test': np.nan, 
                'dU_dot_evec_test': dU_nan, 
                'cos_dU_wopt_test': np.nan, 
                'dU_dot_evec_sq_test': dU_nan, 
                'evec_snr_test': dU_nan, 
                'cos_dU_evec_test': dU_nan,
                'dU_mag_train': np.nan, 
                'dU_dot_evec_train': dU_nan, 
                'cos_dU_wopt_train': np.nan, 
                'dU_dot_evec_sq_train': dU_nan, 
                'evec_snr_train': dU_nan, 
                'cos_dU_evec_train': dU_nan
            }

        if (ptrain_mask is not None) & (nan_pad == False):
            # perform analysis for big / small pupil data too. Only on test set.
            A_bp = xtest_pls[:, ptest_mask[0, :, 0], 0]
            A_sp = xtest_pls[:, ~ptest_mask[0, :, 0], 0]
            B_bp = xtest_pls[:, ptest_mask[0, :, 1], 1]
            B_sp = xtest_pls[:, ~ptest_mask[0, :, 1], 1]

            # get dprime / dU
            bp_dprime, _, _, _, bp_dU = compute_dprime(A_bp, B_bp, wopt=pls_wopt_train)
            sp_dprime, _, _, _, sp_dU = compute_dprime(A_sp, B_sp, wopt=pls_wopt_train)

            # get pupil-dependent variance along the prinicple noise axes (analogous to lambda)
            # point is to compare the variance along these PCs between large / small pupil
            big = np.concatenate([A_bp, B_bp], axis=-1)
            small = np.concatenate([A_sp, B_sp], axis=-1)
            bp_lambda = np.var(big.T.dot(pls_evecs_test), axis=0)
            sp_lambda = np.var(small.T.dot(pls_evecs_test), axis=0)

            # compute additional metrics
            bp_dU_mag         = np.linalg.norm(bp_dU)
            bp_dU_dot_evec    = bp_dU.dot(pls_evecs_test)
            bp_cos_dU_wopt    = abs(unit_vectors(bp_dU.T).T.dot(unit_vectors(pls_wopt_train)))
            bp_dU_dot_evec_sq = bp_dU.dot(pls_evecs_test) ** 2
            bp_evec_snr       = bp_dU_dot_evec_sq / bp_lambda
            bp_cos_dU_evec    = abs(unit_vectors(bp_dU.T).T.dot(pls_evecs_test))

            sp_dU_mag         = np.linalg.norm(sp_dU)
            sp_dU_dot_evec    = sp_dU.dot(pls_evecs_test)
            sp_cos_dU_wopt    = abs(unit_vectors(sp_dU.T).T.dot(unit_vectors(pls_wopt_train)))
            sp_dU_dot_evec_sq = sp_dU.dot(pls_evecs_test) ** 2
            sp_evec_snr       = sp_dU_dot_evec_sq / sp_lambda
            sp_cos_dU_evec    = abs(unit_vectors(sp_dU.T).T.dot(pls_evecs_test))

            results.update({
                'bp_dp': bp_dprime,
                'bp_evals': bp_lambda,
                'bp_dU_mag': bp_dU_mag,
                'bp_dU_dot_evec': bp_dU_dot_evec,
                'bp_cos_dU_wopt': bp_cos_dU_wopt,
                'bp_dU_dot_evec_sq': bp_dU_dot_evec_sq,
                'bp_evec_snr': bp_evec_snr,
                'bp_cos_dU_evec': bp_cos_dU_evec,
                'sp_dp': sp_dprime,
                'sp_evals': sp_lambda,
                'sp_dU_mag': sp_dU_mag,
                'sp_dU_dot_evec': sp_dU_dot_evec,
                'sp_cos_dU_wopt': sp_cos_dU_wopt,
                'sp_dU_dot_evec_sq': sp_dU_dot_evec_sq,
                'sp_evec_snr': sp_evec_snr,
                'sp_cos_dU_evec': sp_cos_dU_evec
            })
        
        elif (ptrain_mask is not None) & (nan_pad == True):
            results.update({
                'bp_dp': np.nan,
                'bp_evals': eval_nan,
                'bp_dU_mag': np.nan,
                'bp_dU_dot_evec': dU_nan,
                'bp_cos_dU_wopt': np.nan,
                'bp_dU_dot_evec_sq': dU_nan,
                'bp_evec_snr': dU_nan,
                'bp_cos_dU_evec': dU_nan,
                'sp_dp': np.nan,
                'sp_evals': eval_nan,
                'sp_dU_mag': np.nan,
                'sp_dU_dot_evec': dU_nan,
                'sp_cos_dU_wopt': np.nan,
                'sp_dU_dot_evec_sq': dU_nan,
                'sp_evec_snr': dU_nan,
                'sp_cos_dU_evec': dU_nan
            })
        else:
            pass

        return results


def cast_dtypes(df):
    dtypes = {'dp_opt_test': 'float32',
              'dp_diag_test': 'float32',
              'wopt_test': 'object',
              'wdiag_test': 'object',
              'var_explained_test': 'float32',
              'evals_test': 'object',
              'evecs_test': 'object',
              'evec_sim_test': 'float32',
              'dU_test': 'object',
              'dp_opt_train': 'float32',
              'dp_diag_train': 'float32',
              'wopt_train': 'object',
              'wdiag_train': 'object',
              'var_explained_train': 'float32',
              'evals_train': 'object',
              'evecs_train': 'object',
              'evec_sim_train': 'float32',
              'dU_train': 'object',
              'dU_mag_test': 'float32',
              'dU_dot_evec_test': 'object',
              'cos_dU_wopt_test': 'float32',
              'dU_dot_evec_sq_test': 'object',
              'evec_snr_test': 'object', 
              'cos_dU_evec_test': 'object',
              'dU_mag_train': 'float32',
              'dU_dot_evec_train': 'object',
              'cos_dU_wopt_train': 'float32',
              'dU_dot_evec_sq_train': 'object',
              'evec_snr_train': 'object', 
              'cos_dU_evec_train': 'object',
              'beta1_dot_dU': 'float32',
              'beta2_dot_dU': 'float32',
              'beta1_dot_tdr2': 'float32',
              'beta2_dot_tdr2': 'float32',
              'beta1_dot_wopt': 'float32',
              'beta2_dot_wopt': 'float32',
              'beta1_lambda': 'float32',
              'beta2_lambda': 'float32',
              'beta1_mag': 'float32',
              'beta2_mag': 'float32',
              'dU_dot_beta1_sq': 'float32',
              'dU_dot_beta2_sq':'float32',
              'beta1_snr':'float32',
              'beta2_snr':  'float32',
              'cos_dU_beta1': 'float32',
              'cos_dU_beta2': 'float32',
              'dU_all': 'object',
              'wopt_all': 'object',
              'evecs_all': 'object',
              'cos_r1r2_test': 'float32',
              'cos_r1r2_train': 'float32',
              'cos_ddr_r1r2_test': 'float32',
              'cos_ddr_r1r2_train': 'float32',
              'r1mag_train': 'float32',
              'r2mag_train': 'float32',
              'r1mag_test': 'float32',
              'r2mag_test': 'float32',
              'dU_all_test': 'object',
              'evecs_all_test': 'object',
              'bp_dp': 'float32',
              'bp_dp_diag': 'float32',
              'bp_evals': 'object',
              'bp_dU_mag': 'float32',
              'bp_dU_dot_evec': 'object',
              'bp_cos_dU_wopt': 'float32',
              'bp_dU_dot_evec_sq': 'object',
              'bp_evec_snr': 'object',
              'bp_cos_dU_evec': 'object',
              'sp_dp': 'float32',
              'sp_dp_diag': 'float32',
              'sp_evals': 'object',
              'sp_dU_mag': 'float32',
              'sp_dU_dot_evec': 'object',
              'sp_cos_dU_wopt': 'float32',
              'sp_dU_dot_evec_sq': 'object',
              'sp_evec_snr': 'object',
              'sp_cos_dU_evec': 'object',
              'mean_pupil_range': 'float32',
              'category': 'category',
              'jack_idx': 'category',
              'n_components': 'category',
              'combo': 'category',
              'e1': 'category',
              'e2': 'category',
              'site': 'category'}
    dtypes_new = {k: v for k, v in dtypes.items() if k in df.columns}
    df = df.astype(dtypes_new)
    return df


# ================================== Analysis functions to compute dprime =================================================

def compute_dprime(A, B, diag=False, wopt=None):
    """
    Compute discriminability between matrix A and matrix B
    where both are shape N neurons X N reps.

    Return:
        dprime ** 2 (with sign preserved... so that we can evaluate consitency of sign btwn est/val sets)
        decoding axis (with norm 1)
        evals: (of mean covariance matrix)
        evecs: (of mean covariance matrix)
        evec_sim: similarity of first eigenvector between the two stimuli, A and B
        dU: <A> - <B>
    """
    if (A.shape[0] > A.shape[1]) & (wopt is None):
        raise ValueError("Number of dimensions greater than number of observations. Unstable")

    if A.shape[0] != B.shape[0]:
        raise ValueError("Number of dimensions do not match between conditions")

    if diag:
        dprime, wopt, evals, evecs, evec_sim, dU = _dprime_diag(A, B)

    elif A.shape[0]==1:
        # single dimension
        dprime = _dprime_single_dim(A, B)
        wopt = np.nan; evals = np.nan; evecs = np.nan; evec_sim = np.nan; dU = np.nan
        return dprime, wopt, evals, evecs, evec_sim, dU

    else:
        dprime, wopt, evals, evecs, evec_sim, dU = _dprime(A, B, wopt=wopt)

    return dprime, wopt, evals, evecs, evec_sim, dU 

def _dprime_single_dim(A, B):
    """
    Simplest case of single neuron (dimension) dprime -- classic signal detection theory
    u1 - u2 / sqrt(0.5 * (sig1 + sig2))
    """
    num = A.mean() - B.mean()
    den = np.sqrt(0.5 * (A.var() + B.var()))
    if num == 0:
        return 0
    elif den == 0:
        return np.inf
    else:
        return (num / den) ** 2

def _dprime(A, B, wopt=None):
    """
    See Rumyantsev et. al 2020, Nature for nice derivation
    """

    sigA = np.cov((A - A.mean(axis=-1, keepdims=True)))
    sigB = np.cov((B - B.mean(axis=-1, keepdims=True)))

    usig = 0.5 * (sigA + sigB)
    u_vec = (A.mean(axis=-1) - B.mean(axis=-1))[np.newaxis, :]    

    try:
        valA, vecA = np.linalg.eig(sigA)
        valB, vecB = np.linalg.eig(sigB)
        evec_sim = abs(vecB[:, np.argsort(valB)[::-1][0]].dot(vecA[:, np.argsort(valA)[::-1][0]]))
    except:
        evec_sim = np.nan

    if wopt is not None:
        wopt_train = wopt / np.linalg.norm(wopt)
        A = A.T.dot(wopt_train).T
        B = B.T.dot(wopt_train).T

        usig_ = 0.5 * (np.cov((A - A.mean(axis=-1, keepdims=True))) + np.cov((B - B.mean(axis=-1, keepdims=True))))
        u_vec_ = (A.mean(axis=-1) - B.mean(axis=-1))[np.newaxis, :]
    
    try:
        if wopt is not None:
            wopt = (1 / usig_) * u_vec_
            dp2 = np.matmul(u_vec_, wopt)[0][0]
            try:
                # if wopt is passed, could still compute dpirme but can't compute 
                # evecs/ evals
                evals, evecs = np.linalg.eig(usig)
                # make sure evals / evecs are sorted
                idx_sort = np.argsort(evals)[::-1]
                evals = evals[idx_sort]
                evecs = evecs[:, idx_sort]
            except:
                wopt = np.nan * np.ones((A.shape[0], 1))
                evals = np.nan * np.ones((A.shape[0], ))
                evecs = np.nan * np.ones((A.shape[0], A.shape[0]))

        else:
            inv = np.linalg.inv(usig)
            wopt = inv @ u_vec.T
            dp2 = np.matmul(u_vec, wopt)[0][0]

            evals, evecs = np.linalg.eig(usig)
            # make sure evals / evecs are sorted
            idx_sort = np.argsort(evals)[::-1]
            evals = evals[idx_sort]
            evecs = evecs[:, idx_sort]

    except:
        log.info('WARNING, Singular Covariance, dprime infinite, set to np.nan')
        wopt_nan = np.nan * np.ones((A.shape[0], 1))
        evals_nan = np.nan * np.ones((A.shape[0], ))
        evecs_nan = np.nan * np.ones((A.shape[0], A.shape[0]))
        u_vec_nan =  np.nan * np.ones((1, A.shape[0]))
        return np.nan, wopt_nan, evals_nan, evecs_nan, np.nan, u_vec_nan

    return dp2, wopt, evals, evecs, evec_sim, u_vec


def _dprime_diag(A, B):
    """
    See Rumyantsev et. al 2020, Nature  and Averbeck 2006, JNP for nice derivations.
        Note typo in Rumyantsev though!
    """

    sigA = np.cov((A - A.mean(axis=-1, keepdims=True)))
    sigB = np.cov((B - B.mean(axis=-1, keepdims=True)))

    usig = 0.5 * (sigA + sigB)
    u_vec = (A.mean(axis=-1) - B.mean(axis=-1))[np.newaxis, :]    

    try:
        valA, vecA = np.linalg.eig(sigA)
        valB, vecB = np.linalg.eig(sigB)
        evec_sim = abs(vecB[:, np.argsort(valB)[::-1][0]].dot(vecA[:, np.argsort(valA)[::-1][0]]))
    except:
        evec_sim = np.nan

    u_vec = (A.mean(axis=-1) - B.mean(axis=-1))[np.newaxis, :]

    try:
        # get diagonal covariance matrix
        usig_diag = np.zeros(usig.shape)
        np.fill_diagonal(usig_diag, np.diagonal(usig))

        # compute numerator
        numerator = (u_vec @ np.linalg.inv(usig_diag) @ u_vec.T) ** 2

        # compute denominator
        denominator = u_vec @ np.linalg.inv(usig_diag) @ usig @ np.linalg.inv(usig_diag) @ u_vec.T
        denominator = denominator[0][0]
    except np.linalg.LinAlgError:
        log.info('WARNING, Singular Covariance, dprime infinite, set to np.nan')
        wopt_nan = np.nan * np.ones((A.shape[0], 1))
        evals_nan = np.nan * np.ones((A.shape[0], ))
        evecs_nan = np.nan * np.ones((A.shape[0], A.shape[0]))
        u_vec_nan =  np.nan * np.ones((1, A.shape[0]))
        return np.nan, wopt_nan, evals_nan, evecs_nan, np.nan, u_vec_nan

    dp2 = (numerator / denominator).squeeze()

    evals, evecs = np.linalg.eig(usig)
    # make sure evals / evecs are sorted
    idx_sort = np.argsort(evals)[::-1]
    evals = evals[idx_sort]
    evecs = evecs[:, idx_sort]

    # best decoding axis ignoring correlations (reduces to direction of u_vec)
    wopt_diag = np.linalg.inv(usig_diag) @ u_vec.T

    return dp2, wopt_diag, evals, evecs, evec_sim, u_vec


# ================================= Data Loading Utils ========================================
def load_site(site, batch, ctx=None, use_pred=False, pca_ops=None, pup_ops=None,
                    regress_pupil=False, gain_only=False, dc_only=False, deflate_residual_dim=None, 
                    use_xforms=False, xforms_modelname=None, xforms_signal='pred', reshuf=False, 
                    return_epoch_list=False, mask_movement=False,
                    exclude_low_fr=False, threshold=None, 
                    use_old_cpn=False,
                    special=False, verbose=False):
    """
    Loads recording and does some standard preprocessing for nat sounds decoding analysis
        e.g. masks validation set and removes post stim silence.
        (does some more weird stuff for CPN (Mateo Data))
    
    Returns full spike count matrix, X, and a spike count matrix with just spont data

    If deflate_residual is not None (is a vector/matrix), the project residuals onto 
    thie set of dimenions to make a reduced rank matrix. Subtract this off 
    from residuals.

    crh - 6.9.2021
    Note that xforms_modelname is not related to use_xforms. use_xforms is just for regressing out first order pupil,
    while if xforms_modelname is not None, this will load the pred for that modelstring
    reshuf will reshuffle the signals in val['indep'] and val['lv'] -- these are random noise for LV models.

    20.10.2021
    Add special pupil options (pup_ops) used for specifying the "big" pupil mask.
    For example, if pup_ops[0] = 'even_split', pup_ops[1]=0, and pup_ops[2]=1/3 then "big pupil" will be defined as the bottom 1/3 of pupil values per stim.
    if pup_ops[0] = 'max_norm', we have to do some more hairy balancing stuff. In this case we want all pupil where it was less than 33% of its
    max across all data. For some experiments / animals, this might mean zero-percent of data.
    pup_ops[1] = lower quant limit
    pup_ops[2] = upper quant limit

    2022.05.2022
    Add "ctx". If ctx is passed, we already have the recording. Don't need to load. 
    Just get it out of ctx["val"]
    if use_pred = True, then load the prediction, not the raw response
    """
    if ctx is None:
        if 0: #batch in [289, 294, 322]:
            options = {'cellid': site, 'rasterfs': 4, 'batch': batch, 'pupil': True, 'stim': False}
            if batch == 294:
                options['runclass'] = 'VOC'
            if batch == 322:
                options['batch'] = 289
            rec = nb.baphy_load_recording_file(**options)
            rec['resp'] = rec['resp'].rasterize()
        elif batch in [289, 294, 322]:
            manager = BAPHYExperiment(cellid=site, batch=batch)
            options = {'rasterfs': 4, 'resp': True, 'stim': False, 'pupil': True, 'pupil_variable_name': 'area'}
            rec = manager.get_recording(**options)
            rec['resp'] = rec['resp'].rasterize()
            if batch == 294:
                stims = [s for s in rec['resp'].epochs.name.unique() if ('STIM_' in s) & ('Pips' not in s)]
                rec = rec.and_mask(stims)
                rec = rec.apply_mask(reset_epochs=True)

        elif batch in [331]:
            # CPN data from Mateo. Need to do some kludging with epochs
            manager = BAPHYExperiment(cellid=site, batch=batch)
            options = {'rasterfs': 4, 'resp': True, 'stim': False, 'pupil': True, 'pupil_variable_name': 'area'}
            rec = manager.get_recording(**options)
            rec['resp'] = rec['resp'].rasterize()
            rec = nems_preproc.fix_cpn_epochs(rec, use_old=use_old_cpn)

            if mask_movement != False:
                if type(mask_movement) is tuple:
                    log.info(f"Masking recording using threshold: {mask_movement[0]} and binsize: {mask_movement[1]}")
                    rec = nems_preproc.movement_mask(rec, threshold=mask_movement[0], binsize=mask_movement[1])['rec']
                else:
                    rec = nems_preproc.movement_mask(rec)['rec']
        
        else:
            raise ValueError(f"Unknown batch: {batch}")
            
    else:
        # ctx was passed, get recording directly
        rec = ctx["val"].copy()

    if use_pred: 
        signal = "pred"
    else:
        signal = "resp"
    
    if 'cells_to_extract' in rec.meta.keys():
        if rec.meta['cells_to_extract'] is not None:
            log.info("Extracting cellids: {0}".format(rec.meta['cells_to_extract']))
            rec[signal] = rec[signal].extract_channels(rec.meta['cells_to_extract'])

    if xforms_modelname is not None:
        try:
            xf, ctx = load_model_xform(cellid=site, batch=batch, modelname=xforms_modelname)
        except:
            cellid = [c for c in nd.get_batch_cells(batch).cellid if site in c][0]
            xf, ctx = load_model_xform(cellid=cellid, batch=batch, modelname=xforms_modelname)
        rec = ctx['val'].copy()
        if reshuf:
            for s in ['indep', 'lv']:
                try:
                    n_chans, T = rec[s].shape
                    rec[s] = rec[s]._modified_copy(np.random.randn(n_chans, T))
                    rec = ctx['modelspec'].evaluate(rec=rec)
                except:
                    log.info(f"Trying to reshuffle {s} but signal doesn't exist in val")
                    pass
        rec[signal] = rec[xforms_signal].copy()


    # make sure mask is a bool
    if 'mask' in rec.signals.keys():
        rec['mask'] = rec['mask']._modified_copy(rec['mask']._data.astype(bool))

    if batch in [294, 331]:
        epochs = [epoch for epoch in rec[signal].epochs.name.unique() if 'STIM_' in epoch]
    else:
        epochs = [epoch for epoch in rec[signal].epochs.name.unique() if 'STIM_00' in epoch]
    rec = rec.and_mask(epochs)


    if pca_ops is not None:
        ctx = resp_to_pc(rec, pc_idx=None, pc_count=pca_ops['pc_count'], pc_source=pca_ops['pc_source'],
                              overwrite_resp=True, compute_power='no', whiten=pca_ops['whiten'])
        rec = ctx['rec']

    # regress out pupil, if specified
    if regress_pupil:
        if not use_xforms:
            log.info('Removing first order pupil')
            rec = preproc.regress_state(rec, state_sigs=['pupil'])
        elif use_xforms:
            log.info('Removing first order pupil by subtracting xforms model prediction')
            cellid = rec['resp'].chans
            xforms_modelname = 'ns.fs4.pup-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic'
            if batch == 294:
                xforms_modelname = xforms_modelname.replace('pup-ld', 'pup.voc-ld')
            if batch == 331:
                xforms_modelname = xforms_modelname.replace('-hrc', '-epcpn-hrc')
            rec_path = f'/auto/users/hellerc/results/nat_pupil_ms/pr_recordings/{batch}/'
            rec = preproc.generate_state_corrected_psth(batch=batch, modelname=xforms_modelname, cellids=cellid, 
                                        siteid=site, gain_only=gain_only, dc_only=dc_only,
                                        cache_path=rec_path, recache=False)
            mod_data = rec[signal]._data - rec['psth']._data + rec['psth_sp']._data
            rec[signal] = rec[signal]._modified_copy(mod_data)
        
        #if batch in [331]:
        #    rec = nems_preproc.fix_cpn_epochs(rec)
        # this happens in the xforms model
    
    if deflate_residual_dim is not None:
            log.info('Reducing rank of residuals by deflating out given dimenions')
            cellid = rec[signal].chans
            xforms_modelname = 'ns.fs4.pup-ld-st.pup-hrc-psthfr_sdexp.SxR.bound_jk.nf10-basic'
            if batch == 294:
                xforms_modelname = xforms_modelname.replace('pup-ld', 'pup.voc-ld')
            rec_path = f'/auto/users/hellerc/results/nat_pupil_ms/pr_recordings/{batch}/'
            rec = preproc.generate_state_corrected_psth(batch=batch, modelname=xforms_modelname, cellids=cellid, 
                                        siteid=site,
                                        cache_path=rec_path, recache=False)
            residual = rec[signal]._data - rec['psth_sp']._data
            reduced_rank = residual.T.dot(deflate_residual_dim)[:, np.newaxis] @ deflate_residual_dim[np.newaxis]
            residual = residual - reduced_rank.T
            mod_data = rec['psth_sp']._data + residual
            rec[signal] = rec[signal]._modified_copy(mod_data)
    
    # remove post stim silence (keep prestim so that can get a baseline dprime on each sound)
    rec = rec.and_mask(['PostStimSilence'], invert=True)

    # remove low firing rate cells
    if exclude_low_fr:
        if threshold is not None:
            thresh = threshold
        else:
            thresh = 1
        data = rec.apply_mask()[signal]._data
        m = data.mean(axis=-1) * rec[signal].fs
        keep = np.argwhere(m > thresh)
        cids = np.array(rec[signal].chans)[keep]
        log.info(f"Removing {len(rec[signal].chans)-len(cids)} low FR units under threshold of {thresh}Hz")
        for s in [signal, 'pred', 'psth', 'psth_sp']:
            if s in rec.signals.keys():
                rec[s] = rec[s].extract_channels(list(cids))
    
    resp_dict = rec[signal].extract_epochs(epochs, mask=rec['mask'], allow_incomplete=True)
    spont_signal = rec[signal].epoch_to_signal('PreStimSilence')
    sp_dict = spont_signal.extract_epochs(epochs, mask=rec['mask'], allow_incomplete=True)
    pup_dict = rec['pupil'].extract_epochs(epochs, mask=rec['mask'], allow_incomplete=True)

    # create response matrix, X
    X = nat_preproc.dict_to_X(resp_dict)
    X_sp = nat_preproc.dict_to_X(sp_dict)
    X_pup = nat_preproc.dict_to_X(pup_dict)

    # save epoch names
    epoch_names = epochs

    # if batch is 331, we don't have traditional PreStimSilence
    # instead, extract mean spont rate per neuron for special plotting
    # stuff
    if batch == 331:
        tstimsilence = rec[signal].extract_epoch('TRIALPreStimSilence')
        spont_per_neuron = np.mean(tstimsilence, axis=(0, 2))

    # make pupil mask
    reps = X_pup.shape[1]
    epochs = X_pup.shape[2]
    bins = X_pup.shape[3]
    X_pup = X_pup.reshape(1, reps, epochs * bins)
    # 20.10.2021 -- adding some funky options here to do different pupil splitting
    if pup_ops is None:
        pup_mask = X_pup >= np.tile(np.median(X_pup, axis=1), [1, X_pup.shape[1], 1])
    elif pup_ops[0]=='even_split':
        # doing even split per stim in order to keep things balanced. However, this maybe isn't the "best" approach
        # because the bottom third for one stim / animal could be very different from another.
        lq = pup_ops[1]; uq = pup_ops[2]
        ll = np.tile(np.quantile(X_pup, lq, axis=1), [1, X_pup.shape[1], 1])
        ul = np.tile(np.quantile(X_pup, uq, axis=1), [1, X_pup.shape[1], 1])
        pup_mask = (X_pup >= ll) & (X_pup <= ul)

    elif pup_ops[0]=='max_norm':
        # this is probably the "better" approach, but a bit more tricky to get the analysis working
        # if there's no data for stim, max sure pup_mask gets set to all False so that we don't try to do
        # "bp/sp" analysis later on
        pnorm = X_pup / X_pup.max()
        lq = pup_ops[1]; uq = pup_ops[2]
        pup_mask = (pnorm >= lq) & (pnorm <= uq)
        
    X_pup = X_pup.reshape(1, reps, epochs, bins)
    pup_mask = pup_mask.reshape(1, reps, epochs, bins)

    if verbose:
        return X, X_sp, X_pup, pup_mask, X_raw, pup_mask_raw
    if return_epoch_list:
        return X, X_sp, X_pup, pup_mask, epoch_names
    if special:
        return X, X_sp, X_pup, pup_mask, epoch_names, spont_per_neuron
    else:
        return X, X_sp, X_pup, pup_mask


def fix_cpn_epochs(rec, manager):
    raise DeprecationWarning("Use the copy of this function in nems_lbhb.preprocessing!!")
    newrec = copy.deepcopy(rec)
    # figure out which file has AllPermutations
    parms = manager.get_baphy_exptparams()
    seqStruct = [p['TrialObject'][1]['ReferenceHandle'][1]['SequenceStructure'] for p in parms]
    try:
        goodfiles = np.argwhere(np.array(seqStruct)=='AllPermutations').squeeze()
        files = [e for e in rec.epochs.name if e.startswith('FILE')]
        gfiles = files[goodfiles]
        newrec = newrec.and_mask(gfiles)
        newrec = newrec.apply_mask(reset_epochs=True)

        # now that it's masked, fixed the epochs
        new_epochs = copy.deepcopy(newrec.epochs)

        # Name the "trial" pre/post differently to distinguish from each STIM
        prepost = new_epochs.name.str.startswith('PreStimSilence') | new_epochs.name.str.startswith('PostStimSilence')
        new_epochs.at[prepost, 'name'] = ['TRIAL'+s for s in new_epochs[prepost].name]

        # strip the seq. epochs and sub pre/post stim
        new_epochs = new_epochs[~new_epochs.name.str.contains('_sequence')]

        # remove "sub" labels
        sub = new_epochs.name.str.startswith('Sub')
        new_epochs.at[sub, 'name'] = [s.strip('Sub') for s in new_epochs[sub].name]

        # Clean up the actual sound epochs 
        stim_mask = new_epochs.name.str.startswith('STIM_')
        new_epochs.at[stim_mask, 'name'] = [s.split('context:')[0][:-1] for s in new_epochs[stim_mask].name]

        # Chop out first bin of each (to remove weird context effects) -- (and for the "dummy" prestim silence) 
        one_bin = np.float(1 / rec['resp'].fs)
        new_epochs.at[stim_mask, 'start'] = new_epochs.loc[stim_mask, 'start'].values + one_bin
        new_epochs.at[sub, 'start'] = new_epochs.loc[sub, 'start'].values + one_bin
        
        # clean up index
        new_epochs = new_epochs.reindex()

        # assign to new recording
        for signal in newrec.signals.keys():
            newrec[signal].epochs = new_epochs

        return newrec

    except:
        raise ValueError(f"Couldn't find file with AllPermuations for the CPN recording: {newrec.name}")


def simulate_response(X, pup_mask, sim_first_order=False,
                                   sim_second_order=False,
                                   sim_all=False,
                                   var_first_order=True,
                                   nreps=5000,
                                   suppress_log=False):
    reps = X.shape[1]
    epochs = X.shape[2]
    bins = X.shape[3]

    # simulate data, if specified
    if sim_first_order:
        if not suppress_log:
            log.info("Simulating only first order changes between large and small pupil")
        # simulate first order difference only between large/small pupil
        # fix second order stats to the overall data
        xtemp = X.reshape(X.shape[0], reps, epochs * bins)
        pmasktemp = pup_mask.reshape(1, reps, epochs * bins)
        X_big = np.stack([xtemp[:, pmasktemp[0, :, i], i] for i in range(epochs * bins)], axis=-1)
        X_big = X_big.reshape(-1, X_big.shape[1], epochs, bins)
        X_small = np.stack([xtemp[:, ~pmasktemp[0, :, i], i] for i in range(epochs * bins)], axis=-1)
        X_small = X_small.reshape(-1, X_small.shape[1], epochs, bins)

        # simulate
        X_big_sim = simulate.generate_simulated_trials(X_big, X, keep_stats=[1], var_first_order=var_first_order, N=nreps)
        X_small_sim = simulate.generate_simulated_trials(X_small, X, keep_stats=[1], var_first_order=var_first_order, N=nreps)
    
        X = np.concatenate((X_big_sim, X_small_sim), axis=1)
        p_mask = np.ones((1,) + X_big_sim.shape[1:]).astype(np.bool)
        pup_mask = np.concatenate((p_mask, ~p_mask), axis=1)

    elif sim_second_order:
        if not suppress_log:
            log.info("simulating only second order change between large and small pupil")
        # simulate first order difference only between large/small pupil
        # fix second order stats to the overall data
        xtemp = X.reshape(X.shape[0], reps, epochs * bins)
        pmasktemp = pup_mask.reshape(1, reps, epochs * bins)
        X_big = np.stack([xtemp[:, pmasktemp[0, :, i], i] for i in range(epochs * bins)], axis=-1)
        X_big = X_big.reshape(-1, X_big.shape[1], epochs, bins)
        X_small = np.stack([xtemp[:, ~pmasktemp[0, :, i], i] for i in range(epochs * bins)], axis=-1)
        X_small = X_small.reshape(-1, X_small.shape[1], epochs, bins)

        # simulate
        X_big_sim = simulate.generate_simulated_trials(X_big, X, keep_stats=[2], var_first_order=var_first_order, N=nreps)
        X_small_sim = simulate.generate_simulated_trials(X_small, X, keep_stats=[2], var_first_order=var_first_order, N=nreps)
    
        X = np.concatenate((X_big_sim, X_small_sim), axis=1)
        p_mask = np.ones((1,) + X_big_sim.shape[1:]).astype(np.bool)
        pup_mask = np.concatenate((p_mask, ~p_mask), axis=1)

    elif sim_all:
        if not suppress_log:
            log.info("simulating both first and second order change between large and small pupil")
        # simulate first order and second order statistics of data
        xtemp = X.reshape(X.shape[0], reps, epochs * bins)
        pmasktemp = pup_mask.reshape(1, reps, epochs * bins)
        X_big = np.stack([xtemp[:, pmasktemp[0, :, i], i] for i in range(epochs * bins)], axis=-1)
        X_big = X_big.reshape(-1, X_big.shape[1], epochs, bins)
        X_small = np.stack([xtemp[:, ~pmasktemp[0, :, i], i] for i in range(epochs * bins)], axis=-1)
        X_small = X_small.reshape(-1, X_small.shape[1], epochs, bins)

        # simulate
        X_big_sim = simulate.generate_simulated_trials(X_big, X, keep_stats=[1, 2], var_first_order=var_first_order, N=nreps)
        X_small_sim = simulate.generate_simulated_trials(X_small, X, keep_stats=[1, 2], var_first_order=var_first_order, N=nreps)
    
        X = np.concatenate((X_big_sim, X_small_sim), axis=1)
        p_mask = np.ones((1,) + X_big_sim.shape[1:]).astype(np.bool)
        pup_mask = np.concatenate((p_mask, ~p_mask), axis=1) 

    return X, pup_mask


def shuffle_trials(X, pmask):
    """
    Specialized function to shuffle trials for each stimulus, per neuron, within a given pupil state.
    Idea is to break within pupil correlations, but preserve overall gain changes etc. between pupil
    states.
    
    X should be shape neuron X reps X stim
    pmask should be shape 1 X reps X stim (true where pupil = big)
    """       
    Xnew = X.copy()
    if X.shape[0] > pmask[0, :, 0].sum():
        log.info("WARNING - fewer reps than neurons w/in a pupil state, shuffling won't be complete")
    for stim in range(X.shape[-1]):
        nbig = pmask[0, :, stim].sum()
        bigidx = np.argwhere(pmask[0, :, stim]).squeeze()
        nsmall = (~pmask[0, :, stim]).sum()
        smallidx = np.argwhere(~pmask[0, :, stim]).squeeze()
        for n in range(X.shape[0]):
            big_shuf_idx = np.random.choice(bigidx, nbig, replace=False)
            small_shuf_idx = np.random.choice(smallidx, nsmall, replace=False)
            Xnew[n, bigidx, stim] = X[n, big_shuf_idx, stim]
            Xnew[n, smallidx, stim] = X[n, small_shuf_idx, stim]
    return Xnew


def load_xformsModel(site, batch, signal='pred', modelstring=None, return_meta=False):
    """
    Load xforms model prediction. Note, some of these predictions (e.g. normative LV Models)
    tile the response across time, so the size is bigger. In these cases, we need to create a 
    new pupil mask for this "new" data.
    """
    # below is deprecated. modelname batch should match decoding batch now.
    #if batch==289:
    #    batch=322
    try:
        xf, ctx = load_model_xform(site, batch, modelname=modelstring)
    except:
        log.info("try loading using cellid instead of site")
        if '.e' in site:
            ops = {'cellid': site, 'batch': batch}
            ops = io.parse_cellid(ops)
            cellid = ops[0][0]
            xf, ctx = load_model_xform(cellid, batch, modelname=modelstring)
        else:
            xf, ctx = load_model_xform([c for c in nd.get_batch_cells(batch).cellid if site in c][0], batch, modelname=modelstring)

    # TODO - do we always want the "pred" signal?
    rec = ctx['val'].copy()

    # remove post stim silence (keep prestim so that can get a baseline dprime on each sound)
    rec = rec.and_mask(['PostStimSilence'], invert=True)
    if (batch == 294) | (batch==331):
        epochs = [epoch for epoch in rec.epochs.name.unique() if 'STIM_' in epoch]
    else:
        epochs = [epoch for epoch in rec.epochs.name.unique() if 'STIM_00' in epoch]
    rec = rec.and_mask(epochs)

    resp_dict = rec[signal].extract_epochs(epochs, mask=rec['mask'], allow_incomplete=True)
    spont_signal = rec[signal].epoch_to_signal('PreStimSilence')
    sp_dict = spont_signal.extract_epochs(epochs, mask=rec['mask'], allow_incomplete=True)
    pup_dict = rec['pupil'].extract_epochs(epochs, mask=rec['mask'], allow_incomplete=True)

    # create response matrix, X
    X = nat_preproc.dict_to_X(resp_dict)
    X_pup = nat_preproc.dict_to_X(pup_dict)

    # make pupil mask
    reps = X_pup.shape[1]
    epochs = X_pup.shape[2]
    bins = X_pup.shape[3]
    X_pup = X_pup.reshape(1, reps, epochs * bins)
    pup_mask = X_pup >= np.tile(np.median(X_pup, axis=1), [1, X_pup.shape[1], 1])
    X_pup = X_pup.reshape(1, reps, epochs, bins)
    pup_mask = pup_mask.reshape(1, reps, epochs, bins)
    
    if return_meta:
        return X, pup_mask, ctx['rec'].meta
    else:
        return X, pup_mask


# ================================= Plotting Utilities =========================================


def plot_stimulus_pair(site='SITE', batch=0, pair=(0,1),
                       X=None, X_raw=None, X_pup=None, pup_mask=None,
                       colors=['red', 'blue'], axlabs=['dim1', 'dim2'],
                        ylim=(None, None), xlim=(None, None), ellipse=False, 
                        pup_cmap=False, lv_axis=None, lv_ax_name='LV axis', ax_length=1, 
                        xforms_modelname=None, xforms_signal='pred', reshuf=False, mask_movement=False,
                        ax=None, pup_split=False, title_string=None, s=10):
    """
    Given a site / stimulus pair, load data, run dprime analysis on all data for the pair
     (no test / train), plot results

     specialized keys to be passed into load_site for loading xforms models:
        xforms_modelname=None, xforms_signal='pred', reshuf=False
    """

    # load raw data if not provided
    if X_raw is None:
        X_raw, sp_bins, X_pup, pup_mask = load_site(site=site, batch=batch,
                                        sim_first_order=False,
                                        sim_second_order=False,
                                        sim_all=False,
                                        mask_movement=mask_movement,
                                        xforms_modelname=None,
                                        regress_pupil=False)
    if (X is None) and (xforms_modelname is not None):
        X, sp_bins, X_pup, pup_mask = load_site(site=site, batch=batch, 
                                        sim_first_order=False, 
                                        sim_second_order=False,
                                        sim_all=False,
                                        xforms_modelname=xforms_modelname,
                                        xforms_signal=xforms_signal,
                                        reshuf=reshuf,
                                        regress_pupil=False)
    elif X is None:
        X = X_raw.copy()

    ncells = X.shape[0]
    nreps = X.shape[1]
    nstim = X.shape[2]
    nbins = X.shape[3]
    X = X.reshape(ncells, nreps, nstim * nbins)

    nreps_raw = X_raw.shape[1]
    X_raw = X_raw.reshape(ncells, nreps_raw, nstim * nbins)
    X_pup = X_pup.reshape(1, nreps, nstim * nbins)
    pup_mask = pup_mask.reshape(1, nreps, nstim * nbins)
    #sp_bins = sp_bins.reshape(1, sp_bins.shape[1], nstim * nbins)
    nstim = nstim * nbins

    reps = X.shape[1]
    reps_raw = X_raw.shape[1]

    X, _ = nat_preproc.scale_est_val([X], [X])
    X_raw, _ = nat_preproc.scale_est_val([X_raw], [X_raw])
    X = X[0]
    X_raw = X_raw[0]

    tdr_axis = nat_preproc.get_first_pc_per_est([X_raw], method='fa')

    X = X[:, :, [pair[0], pair[1]]]
    X_raw = X_raw[:, :, [pair[0], pair[1]]]
    X_pup = X_pup[:, :, [pair[0], pair[1]]]
    pup_mask = pup_mask[:, :, [pair[0], pair[1]]]

    Xflat = nat_preproc.flatten_X(X[:, :, :, np.newaxis])
    Xflat_raw = nat_preproc.flatten_X(X_raw[:, :, :, np.newaxis])
    pmask_flat = nat_preproc.flatten_X(pup_mask[:, :, :, np.newaxis])
    tdr_results, tdr_weights = do_tdr_dprime_analysis(Xflat_raw,
                                                      Xflat,
                                                      reps_raw,
                                                      reps,
                                                      tdr2_axis=tdr_axis[0],
                                                      ptrain_mask=None,
                                                      ptest_mask=None,
                                                      verbose = True)

    if pup_split:
        tdr_results_lg, _ = do_tdr_dprime_analysis(Xflat_raw,
                                                   Xflat[:,pmask_flat[0,:]],
                                                   reps_raw,
                                                   int(np.sum(pmask_flat[0,:])/2),
                                                   tdr2_axis=tdr_axis[0],
                                                   ptrain_mask=None,
                                                   ptest_mask=None,
                                                   verbose = True)
        tdr_results_sm, _ = do_tdr_dprime_analysis(Xflat_raw,
                                                   Xflat[:,~pmask_flat[0,:]],
                                                   reps_raw,
                                                   int(np.sum(~pmask_flat[0,:])/2),
                                                   tdr2_axis=tdr_axis[0],
                                                   ptrain_mask=None,
                                                   ptest_mask=None,
                                                   verbose = True)
        dprimelg = tdr_results_lg['dp_opt_test']
        dprimesm = tdr_results_sm['dp_opt_test']

    weights = tdr_weights
    dprime = tdr_results['dp_opt_test']
    evec = tdr_results['evecs_test'][:, 0]
    evals = tdr_results['evals_test'][0]  
    wopt = tdr_results['wopt_test']
    cos_du = tdr_results['cos_dU_evec_test'][0, 0]

    #print("\n\n\n\n\n\n")
    #print(tdr_weights)
    #print("\n\n\n\n\n\n")
    
    # project all data onto the mean tdr axes
    Xflat = Xflat.T.dot(weights.T).T
    X = nat_preproc.fold_X(Xflat, nreps=reps, nstim=2, nbins=1)

    removed = 0
    if xlim[1] is not None:
        xmask1 = (X[0, :, :] < xlim[1]).squeeze().sum(axis=-1) == 2
        xmask2 = (X[0, :, :] > xlim[0]).squeeze().sum(axis=-1) == 2
        xmask = xmask1 & xmask2
        removed += (xmask==False).sum()
        X = X[:, xmask, :]
        X_pup = X_pup[:, xmask, :]
        pup_mask = pup_mask[:, xmask, :]

    if ylim[1] is not None:
        ymask1 = (X[1, :, :] < ylim[1]).squeeze().sum(axis=-1) == 2
        ymask2 = (X[1, :, :] > ylim[0]).squeeze().sum(axis=-1) == 2
        ymask = ymask1 & ymask2
        removed += (ymask==False).sum()
        X = X[:, ymask, :]
        X_pup = X_pup[:, ymask, :]
        pup_mask = pup_mask[:, ymask, :]

    log.info("Removing {0} / {1} reps due to ax limits".format(removed, reps))

    # plot results
    # center X, for the sake of visualization. Doesn't affect dprime
    X = X - X.mean(axis=(1,2), keepdims=True)  # X.mean(axis=2, keepdims=True)

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(4, 4))

    if pup_split:
        puplg=pup_mask
        pupsm=~pup_mask
        ax.scatter(X[0, puplg[0,:,0], 0], X[1, puplg[0,:,0], 0], s=s, color=colors[0], edgecolor='white')
        ax.scatter(X[0, puplg[0,:,1], 1], X[1, puplg[0,:,1], 1], s=s, color=colors[1], edgecolor='white')
        ax.scatter(X[0, pupsm[0,:,0], 0], X[1, pupsm[0,:,0], 0], s=s, color='lightgray', edgecolor='white')
        ax.scatter(X[0, pupsm[0,:,1], 1], X[1, pupsm[0,:,1], 1], s=s, color='lightgray', edgecolor='white')
    elif pup_cmap:
        ax.scatter(X[0, :, 0, 0], X[1, :, 0, 0], s=s, c=X_pup[0, :, 0], cmap='Reds', edgecolor='white')
        ax.scatter(X[0, :, 1, 0], X[1, :, 1, 0], s=s, c=X_pup[0, :, 1], cmap='Blues', edgecolor='white')
    else:
        ax.scatter(X[0, :, 0], X[1, :, 0], s=s, color=colors[0], edgecolor='white')
        ax.scatter(X[0, :, 1], X[1, :, 1], s=s, color=colors[1], edgecolor='white')

    if ellipse:
        if pup_split:
            e1 = cplt.compute_ellipse(X[0, puplg[0,:,0], 0], X[1, puplg[0,:,0], 0])
            e2 = cplt.compute_ellipse(X[0, puplg[0,:,1], 1], X[1, puplg[0,:,1], 1])
            ax.plot(e1[0], e1[1], lw=2, color=colors[0])
            ax.plot(e2[0], e2[1], lw=2, color=colors[1])
            e1 = cplt.compute_ellipse(X[0, pupsm[0,:,0], 0], X[1, pupsm[0,:,0], 0])
            e2 = cplt.compute_ellipse(X[0, pupsm[0,:,1], 1], X[1, pupsm[0,:,1], 1])
            ax.plot(e1[0], e1[1], lw=2, color='lightgray')
            ax.plot(e2[0], e2[1], lw=2, color='lightgray')

        else:
            e1 = cplt.compute_ellipse(X[0, :, 0], X[1, :, 0])
            e2 = cplt.compute_ellipse(X[0, :, 1], X[1, :, 1])
            ax.plot(e1[0], e1[1], lw=2, color=colors[0])
            ax.plot(e2[0], e2[1], lw=2, color=colors[1])

    # plot first noise PC
    # scale evecs for plotting
    evec = evec * ax_length
    ax.plot([0, evec[0]], [0, evec[1]], 'k', lw=2, label=r"$\mathbf{e}_{1}$")
    ev1 = np.negative(evec)
    ax.plot([0, ev1[0]], [0, ev1[1]], 'k', lw=2)

    # plot wopt
    wopt = (wopt / np.linalg.norm(wopt)) * ax_length
    ax.plot([0, wopt[0]], [0, wopt[1]], 'grey', lw=2, label=r"$\mathbf{w}_{opt}$")
    wopt1 = np.negative(wopt)
    ax.plot([0, wopt1[0]], [0, wopt1[1]], 'grey', lw=2)

    if lv_axis is not None:
        lv_axis = lv_axis.T.dot(weights.T).T
        lv_axis = (lv_axis / np.linalg.norm(lv_axis)) * ax_length
        ax.plot([0, lv_axis[0]], [0, lv_axis[1]], 'magenta', lw=2, label=lv_ax_name)
        lv_axis = np.negative(lv_axis)
        ax.plot([0, lv_axis[0]], [0, lv_axis[1]], 'magenta', lw=2)

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    ax.set_xlabel(axlabs[0])
    ax.set_ylabel(axlabs[1])
    ax.legend(frameon=False)
    if title_string is not None:
        ax.set_title(f"{title_string}")
    elif pup_split:
        ax.set_title(f"dplg={dprimelg:.1f} dpsm={dprimesm:.1f} delta={(dprimelg-dprimesm):.1f}")
    else:
        ax.set_title(r"$d'^{2} = %s$" 
                        "\n"
                        r"$|cos(\theta_{\Delta \mathbf{\mu}, \mathbf{e}_{1}})| = %s$" % (round(dprime, 2), round(cos_du, 2)))   

    return ax


def plot_pair(est, val, nreps_train, nreps_test, train_pmask=None, test_pmask=None, el_only=False):
    
    # unfold data
    if train_pmask is None:
        train = nat_preproc.fold_X(est, nreps=nreps_train, nstim=2, nbins=1).squeeze()
        test = nat_preproc.fold_X(val, nreps=nreps_test, nstim=2, nbins=1).squeeze()
    else:
        train = nat_preproc.fold_X(est, nreps=nreps_train, nstim=2, nbins=1).squeeze()
        test = nat_preproc.fold_X(val, nreps=nreps_test, nstim=2, nbins=1).squeeze()
        Atrain = train[:, :, 0]
        Atest = test[:, :, 0]
        Btrain = train[:, :, 1]
        Btest = test[:, :, 1]

        Atrain_bp = Atrain[:, train_pmask[0, :, 0]]
        Atrain_sp = Atrain[:, ~train_pmask[0, :, 0]]
        Atest_bp = Atest[:, test_pmask[0, :, 0]]
        Atest_sp = Atest[:, ~test_pmask[0, :, 0]]

        Btrain_bp = Btrain[:, train_pmask[0, :, 1]]
        Btrain_sp = Btrain[:, ~train_pmask[0, :, 1]]
        Btest_bp = Btest[:, test_pmask[0, :, 1]]
        Btest_sp = Btest[:, ~test_pmask[0, :, 1]]

    # get stats
    dp_train, wopt_train, evals_train, evecs_train, dU_train, _ = \
                    compute_dprime(train[:, :, 0], train[:, :, 1])
    dp_test, wopt_test, evals_test, evecs_test, dU_test, _ = \
                    compute_dprime(test[:, :, 0], test[:, :, 1], wopt=wopt_train)

    wopt_unit = wopt_train / np.linalg.norm(wopt_train) * np.std(train)

    f, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    
    # plot train data
    ax[0].set_title(r"Train, $d'={}$".format(round(dp_train**(1/2), 3)))
    if train_pmask is None:
        ax[0].scatter(train[0, :, 0], train[1, :, 0], edgecolor='white', color='r', label='A')
        ax[0].scatter(train[0, :, 1], train[1, :, 1], edgecolor='white', color='b', label='B')
    else:
        if el_only:
            e = cplt.compute_ellipse(Atrain_bp[0, :], Atrain_bp[1, :])
            ax[0].plot(e[0, :], e[1, :], color='r', label='A, big pup')
            e = cplt.compute_ellipse(Btrain_bp[0, :], Btrain_bp[1, :])
            ax[0].plot(e[0, :], e[1, :], color='b', label='B, big pup')
            e = cplt.compute_ellipse(Atrain_sp[0, :], Atrain_sp[1, :])
            ax[0].plot(e[0, :], e[1, :], color='r', alpha=0.5, label='A, small pup')
            e = cplt.compute_ellipse(Btrain_sp[0, :], Btrain_sp[1, :])
            ax[0].plot(e[0, :], e[1, :], color='b', alpha=0.5, label='B, small pup')


        else:
            ax[0].scatter(Atrain_bp[0, :], Atrain_bp[1, :], color='r', label='A, big pup')
            ax[0].scatter(Btrain_bp[0, :], Btrain_bp[1, :], color='b', label='B, big pup')
            ax[0].scatter(Atrain_sp[0, :], Atrain_sp[1, :], facecolor='none', color='r', label='A, sm. pup')
            ax[0].scatter(Btrain_sp[0, :], Btrain_sp[1, :], facecolor='none', color='b', label='B, sm. pup')
    
    # plot vectors
    ax[0].plot([0, wopt_unit[0, 0]], [0, wopt_unit[1, 0]], 'k-', lw=2, label=r'$w_{opt}$')

    ax[0].set_xlabel('PLS 1')
    ax[0].set_ylabel('PLS 2')
    ax[0].axis('square')

    ax[0].legend(frameon=False)

    # plot test data
    ax[1].set_title(r"Test, $d'={}$".format(round(dp_test**(1/2), 3)))
    if test_pmask is None:
        ax[1].scatter(test[0, :, 0], test[1, :, 0], edgecolor='white', color='r', label='A')
        ax[1].scatter(test[0, :, 1], test[1, :, 1], edgecolor='white', color='b', label='B')
    else:
        if el_only:
            e = cplt.compute_ellipse(Atest_bp[0, :], Atest_bp[1, :])
            ax[1].plot(e[0, :], e[1, :], color='r', label='A, big pup')
            e = cplt.compute_ellipse(Btest_bp[0, :], Btest_bp[1, :])
            ax[1].plot(e[0, :], e[1, :], color='b', label='B, big pup')
            e = cplt.compute_ellipse(Atest_sp[0, :], Atest_sp[1, :])
            ax[1].plot(e[0, :], e[1, :], color='r', alpha=0.5, label='A, small pup')
            e = cplt.compute_ellipse(Btest_sp[0, :], Btest_sp[1, :])
            ax[1].plot(e[0, :], e[1, :], color='b', alpha=0.5, label='B, small pup')

        else:
            ax[1].scatter(Atest_bp[0, :], Atest_bp[1, :], color='r', label='A, big pup')
            ax[1].scatter(Btest_bp[0, :], Btest_bp[1, :], color='b', label='B, big pup')
            ax[1].scatter(Atest_sp[0, :], Atest_sp[1, :], facecolor='none', color='r', label='A, sm. pup')
            ax[1].scatter(Btest_sp[0, :], Btest_sp[1, :], facecolor='none', color='b', label='B, sm. pup')
        
    ax[1].set_xlabel('PLS 1')
    ax[1].set_ylabel('PLS 2')
    ax[1].axis('square')

    return f


# ========================== nems db utils ===================================
def get_max_pupil(site, force_new=True, rasterfs=4):
    '''
    For a site, find all mfiles where pupil has been analyzed and return the max value across all
    '''
    sql = f"SELECT respfile, parmfile FROM gDataRaw WHERE cellid like '%%{site}%%' and eyewin=2"
    results = nd.pd_query(sql)
    path = results['respfile'].apply(lambda x: x.split('raw/')[0])
    mfiles = (path + results['parmfile']).values.tolist()

    # majority wins vote for if we load the pup.mat analyses vs. cnn analyses (in cases where both exist). this should be rare
    old_pup = []
    new_pup = []
    for i, mfile in enumerate(mfiles):
        if os.path.exists(mfile.replace(results['parmfile'].iloc[i], f"sorted/{results['parmfile'].iloc[i].replace('.m', '.pickle')}")):
            new_pup.append(mfile)
        else:
            old_pup.append(mfile)
    if len(new_pup)>=len(old_pup):
        mfiles = new_pup
    elif force_new & (len(new_pup)>0):
        mfiles = new_pup
    else:
        mfiles = old_pup
    mfiles = np.sort(mfiles).tolist()
    log.info(mfiles)
    manager = BAPHYExperiment(parmfile=mfiles)
    rec = manager.get_recording(**{'rasterfs': rasterfs, 'pupil': True, 'resp': False, 'stim': False, 'pupil_variable_name': 'area'})

    return rec['pupil']._data.max()


def load_FA_model(site, batch, big_psth, small_psth, sim=1, fa_model="factor_analysis_pca_evoked", nreps=2000):
    """
    pretty specialized code to load the results of factor analysis model
    and generate data based on this.

    generate nreps big reps and nreps small reps per stimulus

    sim:
        1 = change in gain only
        2 = change in indep only (fixing absolute covariance)
        3 = change in indep only (fixing relative covariance - so off-diagonals change)
        4 = change in everything (full FA simulation)
        # extras:
        5 = set off-diag to zero, only change single neuron var.
        6 = set off-diag to zero, fix single neuorn var
    """
    np.random.seed(123)
    # load the model results. This hardcoding is a bit kludgy
    path = f"/auto/users/hellerc/results/nat_pupil_ms/factor_analysis/{batch}/{site}/"
    filename = f"{fa_model}.pickle"
    with open(path + filename, 'rb') as handle:
        results = pickle.load(handle)

    nstim = big_psth.shape[2]
    nbins = big_psth.shape[3]
    ncells = big_psth.shape[0]

    # reshape / squish psths
    big_psth = big_psth[:, 0, :, :].reshape(ncells, nstim*nbins)
    small_psth = small_psth[:, 0, :, :].reshape(ncells, nstim*nbins)

    Xsim_big = np.zeros((ncells, nreps, nstim*nbins))
    Xsim_small = np.zeros((ncells, nreps, nstim*nbins))


    if sim==1:
        cov_big = results["final_fit"]["fa_big.sigma_ind"] + results["final_fit"]["fa_big.sigma_shared"]
        cov_small = results["final_fit"]["fa_big.sigma_ind"] + results["final_fit"]["fa_big.sigma_shared"]
    elif sim==2:
        # absolute covariance fixed, but fraction shared variance can change
        cov_big = results["final_fit"]["fa_big.sigma_ind"] + results["final_fit"]["fa_big.sigma_shared"]
        cov_small = results["final_fit"]["fa_small.sigma_ind"] + results["final_fit"]["fa_big.sigma_shared"]
    elif sim==3:
        # relative covariance fixed, i.e. fraction shared variance can stays the same but absolute covariance can change
        cov_big = results["final_fit"]["fa_big.sigma_ind"] + results["final_fit"]["fa_big.sigma_shared"]
        cov_small = results["final_fit"]["fa_small.sigma_ind"] + results["final_fit"]["fa_big.sigma_shared"]
        # force small to have same corr. coef. as cov_big
        norm = np.sqrt(np.diag(cov_big)[:, np.newaxis] @ np.diag(cov_big)[np.newaxis, :])
        corr_big = cov_big / norm # normalize covariance
        var = np.diag(cov_small) # variance of small pupil          
        rootV = np.sqrt(var[:, np.newaxis] @ var[np.newaxis, :])
        cov_small = corr_big * rootV # cov small has same (normalized) correlations as cov_big, but variance like cov_small
    elif sim==4:
        cov_big = results["final_fit"]["fa_big.sigma_ind"] + results["final_fit"]["fa_big.sigma_shared"]
        cov_small = results["final_fit"]["fa_small.sigma_ind"] + results["final_fit"]["fa_small.sigma_shared"]
    elif sim==5:
        # diag matrix, entries change between large and small
        cov_big = results["final_fit"]["fa_big.sigma_ind"]
        cov_small = results["final_fit"]["fa_small.sigma_ind"]
    elif sim==6:
        # diad matrix, entries fixed to big pupil between states
        cov_big = results["final_fit"]["fa_big.sigma_ind"]
        cov_small = results["final_fit"]["fa_big.sigma_ind"]

    for s in range(big_psth.shape[-1]):
        _cb = cov_big.copy()
        _cs = cov_small.copy()
        Xsim_big[:, :, s]= np.random.multivariate_normal(big_psth[:, s], cov=_cb, size=nreps).T
        Xsim_small[:, :, s]= np.random.multivariate_normal(small_psth[:, s], cov=_cs, size=nreps).T
    
    # stack into signal matrix and make a new pupil mask to correspond
    X = np.concatenate((Xsim_big, Xsim_small), axis=1).reshape(ncells, int(nreps*2), nstim, nbins)
    pbig = np.ones((1, nreps, nstim, nbins)).astype(bool)
    psmall = np.zeros((1, nreps, nstim, nbins)).astype(bool)
    pup_mask = np.concatenate((pbig, psmall), axis=1)

    return X, pup_mask

def load_full_FA_model(site, batch, X, pup_mask, big_psth, small_psth, use_indep=False, fix_between_states=False, nreps=2000):
    """
    similar to the previous function, but here the idea is that we use the model that was fit across all data so that 
    factors are shared between the two states. 

    In this case, to get the state-dependent simulations we first project the raw (state-dependent) data onto the FA components 
    then compute our own covariance matrices.

    This is only relevant for the case where we're simulating everything (not just indep. noise) since we need a state-dependent model
    of indep. noise in
    """

    raise NotImplementedError("Haven't finished implementation for this.")

    # load the model results. This hardcoding is a bit kludgy
    path = f"/auto/users/hellerc/results/nat_pupil_ms/factor_analysis/{batch}/{site}/"
    filename = "factor_analysis.pickle"
    with open(path + filename, 'rb') as handle:
        results = pickle.load(handle)

    components = results["final_fit"]["fa_all.components_"]

    nstim = big_psth.shape[2]
    nbins = big_psth.shape[3]
    ncells = big_psth.shape[0]

    # reshape / squish psths
    big_psth = big_psth[:, 0, :, :].reshape(ncells, nstim*nbins)
    small_psth = small_psth[:, 0, :, :].reshape(ncells, nstim*nbins)

    Xsim_big = np.zeros((ncells, nreps, nstim*nbins))
    Xsim_small = np.zeros((ncells, nreps, nstim*nbins))
    if use_indep:
        # no correlations
        cov_big = results["final_fit"]["fa_big.sigma_ind"]
        cov_small = results["final_fit"]["fa_small.sigma_ind"]
    else:
        cov_big = results["final_fit"]["fa_big.sigma_full"]
        cov_small = results["final_fit"]["fa_small.sigma_full"]
    
    # use same covariance between large and small conditions
    if fix_between_states:
        cov_small = cov_big.copy()

    for s in range(big_psth.shape[-1]):
        _cb = cov_big.copy()
        _cs = cov_small.copy()
        Xsim_big[:, :, s]= np.random.multivariate_normal(big_psth[:, s], cov=_cb, size=nreps).T
        Xsim_small[:, :, s]= np.random.multivariate_normal(small_psth[:, s], cov=_cs, size=nreps).T
    
    # stack into signal matrix and make a new pupil mask to correspond
    X = np.concatenate((Xsim_big, Xsim_small), axis=1).reshape(ncells, int(nreps*2), nstim, nbins)
    pbig = np.ones((1, nreps, nstim, nbins)).astype(bool)
    psmall = np.zeros((1, nreps, nstim, nbins)).astype(bool)
    pup_mask = np.concatenate((pbig, psmall), axis=1)

    return X, pup_mask