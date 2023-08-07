"""
Some routines for clustering neurons according to their correlation matrix
"""

import numpy as np
import pandas as pd

import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

def cluster_corr(corr_array, count=None, threshold=1.75, use_abs=False, return_indices=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to each other

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    if use_abs:
        pairwise_distances = sch.distance.pdist(np.abs(corr_array))
    else:
        pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    if count is not None:
        idx_to_cluster_array = sch.fcluster(linkage, count,
                                            criterion='maxclust')
    else:
        cluster_distance_threshold = pairwise_distances.max() / threshold
        idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    if return_indices:
        return idx, idx_to_cluster_array

    return corr_array[idx, :][:, idx]



def hierarchical_cluster(corr_array, threshold=10, labels=None, verbose=False):
    np.fill_diagonal(corr_array, 1)
    #corr_array[corr_array>0.9999] = 1
    dissimilarity = 1 - abs(corr_array)
    dissimilarity[dissimilarity < 0] = 0
    Z = linkage(squareform(dissimilarity), 'complete')

    #idx_to_cluster_array = fcluster(Z, threshold, criterion='distance')
    idx_to_cluster_array = fcluster(Z, threshold, criterion='maxclust')
    # renumber the clusters
    uidx = np.unique(idx_to_cluster_array)

    idx = np.argsort(idx_to_cluster_array)
    cluster_count = idx_to_cluster_array.max()

    if verbose:
        f,ax = plt.subplots(2,1)
        dendrogram(Z, labels=labels, orientation='top',
                   leaf_rotation=90, ax=ax[0])
        ax[0].axhline(threshold, linestyle='--')
        ax[0].set_title(f"Threshold={threshold} N={cluster_count}")
        sc_sorted = corr_array[idx, :][:, idx]
        cluster_n = np.zeros(cluster_count)
        for c in range(cluster_count):
            cluster_n[c]=(idx_to_cluster_array==c+1).sum()

        ax[1].imshow(sc_sorted, aspect='auto', cmap='gray_r', interpolation='none',
                         origin='lower', vmin=0, vmax=1)
        ax[1].vlines(np.cumsum(cluster_n)[:-1]-0.5, -0.5, sc_sorted.shape[1]-0.5, lw=0.5, color='r')
        ax[1].hlines(np.cumsum(cluster_n)[:-1]-0.5, -0.5, sc_sorted.shape[1]-0.5, lw=0.5, color='r')

    return idx, idx_to_cluster_array
