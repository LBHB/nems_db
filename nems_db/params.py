import logging
import os
import copy

import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from nems_lbhb.xform_wrappers import load_batch_modelpaths
import nems0.db as nd
import nems0.modelspec as ms
from nems0.uri import load_resource
from nems0.utils import find_common
from nems0 import get_setting

log = logging.getLogger(__name__)


def fitted_params_per_batch(batch, modelname, mod_key='id', limit=None,
                            multi='mean', meta=['r_test', 'r_fit', 'se_test'],
                            stats_keys=['mean', 'std', 'sem', 'max', 'min'],
                            manual_cellids=None):

    if manual_cellids is not None:
        cellids = manual_cellids
    else:
        celldata = nd.get_batch_cells(batch=batch)
        cellids = celldata['cellid'].tolist()
        if limit is not None:
            cellids = cellids[:limit]

    return fitted_params_per_cell(cellids, batch, modelname, mod_key=mod_key,
                                  stats_keys=stats_keys, meta=meta,
                                  multi=multi)


def fitted_params_per_cell(cellids, batch, modelname, mod_key='id',
                           meta=['r_test', 'r_fit', 'se_test'], multi='mean',
                           stats_keys=['mean', 'std', 'sem', 'max', 'min']):
    '''
    Valid meta keys for LBHB (not exhaustive):
        r_test, r_fit, r_ceiling, r_floor, cellid, batch ... etc
    Valid stats_keys:
        mean, std, sem, max, min
    Valid 'multi' options (for dealing with multi-modelspec fits):
        mean (default), first, all (work in progress)
    '''

    # query nems_db results to get a list of modelspecs
    # (should end up with one modelspec per cell)
    modelspecs = _get_modelspecs(cellids, batch, modelname, multi=multi)
    if multi == 'all':
        raise NotImplementedError
        # Flatten sublists of modelspecs
        modelspecs = [m for ms in modelspecs for m in ms]

    stats = ms.summary_stats(modelspecs, mod_key=mod_key,
                             meta_include=meta, stats_keys=stats_keys)
    index = list(stats.keys())
    try:
        columns = [m[0].get('meta').get('cellid') for m in modelspecs]
    except:
        log.warning("Couldn't use cellids from modelspecs, using cellids "
                    "from function parameters instead.")
        columns = cellids

    data = {}
    current = columns[0]
    counter = 1
    for i, c in enumerate(columns):
        # Ensure unique column names, will have duplicates if multi='all'
        # and there were multi-fit models included.
        if i == 0:
            pass
        elif c == current:
            columns[i] = '%s<%d>'%(c, counter)
            counter += 1
        else:
            current = c
            counter = 1

        for k in index:
            val = ms.try_scalar(stats[k]['values'][i])
            if c in data.keys():
                data[c].append(val)
            else:
                data[c] = [val]

    if stats_keys:
        for s in reversed(stats_keys):
            columns.insert(0, s)
            data[s] = []
            for k in index:
                data[s].append(stats[k][s])

    return pd.DataFrame(data=data, index=index, columns=columns)


def get_batch_modelspecs(batch, modelname, mod_key='id', limit=None, multi='mean'):
    celldata = nd.get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()
    if limit is not None:
        cellids = cellids[:limit]
    return _get_modelspecs(cellids, batch, modelname, multi=multi)


def _get_modelspecs(cellids, batch, modelname, multi='mean'):
    filepaths = load_batch_modelpaths(batch, modelname, cellids,
                                      eval_model=False)
    speclists = []
    for path in filepaths:
        mspaths = []
        path = path.replace('http://hyrax.ohsu.edu:3003/','/auto/data/nems_db/')
        if get_setting('NEMS_RESULTS_DIR').startswith("/Volumes"):
            path = path.replace('/auto/', '/Volumes/')
        for file in os.listdir(path):
            if file.startswith("modelspec"):
                mspaths.append(os.path.join(path, file))
        speclists.append([load_resource(p) for p in mspaths])

    modelspecs = []
    for m in speclists:
        if len(m) > 1:
            if multi == 'first':
                this_mspec = m[0]
            elif multi == 'all':
                this_mspec = m
            elif multi == 'mean':
                stats = ms.summary_stats(m)
                temp_spec = copy.deepcopy(m[0])
                phis = [m['phi'] for m in temp_spec]
                for p in phis:
                    for k in p:
                        for s in stats:
                            if s.endswith('--'+k):
                                p[k] = stats[s]['mean']
                for m, p in zip(temp_spec, phis):
                    m['phi'] = p
                this_mspec = temp_spec
            else:
                log.warning("Couldn't interpret <multi> parameter. Got: %s,\n"
                            "Expected one of: 'mean, first, random, all'.\n"
                            "Using first modelspec instead.", multi)
                this_mspec = m[0]
        else:
            this_mspec = m[0]

        modelspecs.append(ms.ModelSpec([this_mspec]))

    return modelspecs


# https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-
# to-theoretical-ones-with-scipy-python
def plot_parameter(p, dists=None, num_bins=100, title=None):
    # TODO: Get this to work for arrays like
    #       fir coefficients
    if dists is None:
        dists = ['norm', 'beta', 'gamma', 'pareto', 'halfnorm']

    fig, ax = plt.subplots()
    ax.hist(p, bins=num_bins, normed=True, alpha=0.6)
    ylim = ax.get_ylim()

    lower = np.min(p)
    upper = np.max(p)
    # Values to show PDF for
    x = np.linspace(lower, upper, num_bins)

    for i, d in enumerate(dists):
        dist = getattr(st, d, None)
        if dist is None:
            raise ValueError("No such distribution in scipy.stats: %s" % d)
        args = dist.fit(p)
        pdf = dist.pdf(x, *args[:-2], loc=args[-2], scale=args[-1])
        ax.plot(x, pdf, label=d)
        ax.set_xlim(lower, upper)
        ax.set_ylim(ylim)
        log.info('\nFrom {}, distribution: {}'.format(title, d))
        if args[:-2]:
            log.info('a, b (shape): {}'.format(args[:-2]))
        log.info('loc: {}, scale: {}'.format(args[-2], args[-1]))

    ax.legend(loc='best')
    if title is not None:
        ax.set_title(title)

    return fig


def plot_all_params(df, dists=None, num_bins=100, dtype='float32',
                    only_scalars=True):
    params = df.index.tolist()
    arrays = []
    names = []
    for p in params:
        val = df.loc[p]

        if not np.isscalar(val.iat[0]):
            if only_scalars:
                log.info("<only_scalars> was True, skipping non-scalar"
                         " parameter: %s" % p)
                pass
            else:
                #print("\nfor parameter %s, non-scalar" % p)
                combined = np.array(val.tolist())
                #print("\ncombined was: %s" % combined)
                n = combined[0].shape[0]
                flattened = np.concatenate(combined, axis=0)
                #print("\nflattened was: %s" % flattened)
                per_n = flattened.reshape(n, -1).T.tolist()
                #print("\nper_n was: %s" % per_n)
                for i, _ in enumerate(per_n):
                    names.append('%s_index_%d' % (p, i))
                arrays.extend(per_n)
        else:
            a = np.array(val).astype(dtype)
            arrays.append(a)
            names.append(p)

    figs = [plot_parameter(a, dists=dists, num_bins=num_bins, title=p)
            for a, p in zip(arrays, names)]
    return figs


def _flatten_param_names(df):
    # TODO: repeats some code from plot params, could probably
    #       make this more efficient.
    params = df.index.tolist()
    flat_params = []
    for p in params:
        vals = df.loc[p]
        if not np.isscalar(vals):
            vals = np.array(vals)
            flat_vals = vals.flatten()
            param = ['%s%d'%(p,i) for i,_ in enumerate(flat_vals)]
        else:
            param = p
        flat_params.append(param)
    return flat_params


def _fit_distribution(array, dist_type):
    dist_class = getattr(st, dist_type, None)
    if dist_class is None:
        raise ValueError("Not a valid distribution type. Expected a string"
                         " name like 'norm' to import from scipy.stats."
                         "Got: %s", dist_type)

    return dist_class.fit(array)


def param_scatter_batch(batch, model1, model2, param, multi='mean',
                        limit=None, mod_key='id'):
    celldata = nd.get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()
    if limit is not None:
        cellids = cellids[:limit]
    param_scatter(cellids, batch, model1, model2, param, multi=multi,
                  mod_key=mod_key)


def param_scatter(cellids, batch, model1, model2, param, multi='mean',
                  mod_key='id'):
    df1 = fitted_params_per_cell(cellids, batch, model1, mod_key=mod_key,
                                 multi=multi)
    df2 = fitted_params_per_cell(cellids, batch, model2, mod_key=mod_key,
                                 multi=multi)
    vec1 = _get_param_values(df1, param)
    vec2 = _get_param_values(df2, param)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(vec1, vec2)
    ax.axis('equal')
    short, pre, suff = find_common([model1, model2])
    title = "%s\nmodel prefix: %s, suffix: %s" % (param, pre, suff)
    ax.set_title(title)
    ax.set_xlabel(short[0])
    ax.set_ylabel(short[1])
    fig.show()


def _get_param_values(df, key):
    try:
        vector = df.loc[key].values
    except KeyError:
        for k in df.index.values:
            if key in k:
                key = k
                log.info("Updating parameter key from %s to %s", key, k)
                break
        vector = df.loc[key].values
    return vector
