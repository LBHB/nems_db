import logging
import os

import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from nems_db.xform_wrappers import load_batch_modelpaths
import nems_db.db as nd
import nems.modelspec as ms
from nems.uri import load_resource

log = logging.getLogger(__name__)


def fitted_params_per_batch(batch, modelname, include_stats=True,
                            mod_key='id', limit=None):
    celldata = nd.get_batch_cells(batch=batch)
    cellids = celldata['cellid'].tolist()
    if limit is not None:
        cellids = cellids[:limit]
    return fitted_params_per_cell(cellids, batch, modelname,
                                  include_stats=include_stats, mod_key=mod_key)


def fitted_params_per_cell(cellids, batch, modelname, include_stats=True,
                           mod_key='id'):
    # query nems_db results to get a list of modelspecs
    # (should end up with one modelspec per cell)
    modelspecs = _get_modelspecs(cellids, batch, modelname)
    stats = ms.summary_stats(modelspecs, mod_key=mod_key)

    index = list(stats.keys())
    #index_perf = ['r_fit', 'r_test']
    index_perf=[]
    try:
        columns = [m[0].get('meta').get('cellid') for m in modelspecs]
    except:
        columns = cellids
    data = {}
    for i, c in enumerate(cellids):
        for k in index:
            val = ms.try_scalar(stats[k]['values'][i])
            if c in data.keys():
                data[c].append(val)
            else:
                data[c] = [val]
        for k in index_perf:
            val = modelspecs[i][0]['meta'][k]
            data[c].append(val)

    if include_stats:
        columns.insert(0, 'std')
        data['std'] = []
        columns.insert(0, 'mean')
        data['mean'] = []
        for k in index:
            data['std'].append(stats[k]['std'])
            data['mean'].append(stats[k]['mean'])
        for k in index_perf:
            data['std'].append(0)
            data['mean'].append(0)

    index.extend(index_perf)

    return pd.DataFrame(data=data, index=index, columns=columns)


def _get_modelspecs(cellids, batch, modelname):
    filepaths = load_batch_modelpaths(batch, modelname, cellids,
                                      eval_model=False)
    speclists = []
    for path in filepaths:
        mspaths = []
        for file in os.listdir(path):
            if file.startswith("modelspec"):
                mspaths.append(os.path.join(path, file))
        speclists.append([load_resource(p) for p in mspaths])

    modelspecs = []
    for m in speclists:
        if len(m) > 1:
            # TODO: Take mean? only want one modelspec per cell
            #       Or just use first mspec with warning?
            #       Or could just use all of the mspecs and extend
            #       instead of append
    #                stats = ms.summary_stats(mspecs)
    #                temp_spec = copy.deepcopy(mspecs[0])
    #                phis = [m['phi'] for m in temp_spec]
    #                # TODO: This is awful. Make this better.
    #                for p in phis:
    #                    for k in p:
    #                        for s in stats:
    #                            if k in s:
    #                                p[k] = stats[s]['mean']
    #                for m, p in zip(temp_spec, phis):
    #                    m['phi'] = p
    #                this_mspec = temp_spec
            #this_mspec = mspecs
            log.warning("Fit had multiple modelspecs, using first one")
        else:
            pass
        modelspecs.append(m[0])

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
