from scipy.spatial.distance import squareform
from fastcluster import linkage
import numpy as np

def seriation(Z, N, cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))


def compute_serial_matrix(dist_mat, method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage

def get_RDM(input_data):
    '''
    To estimate the representational dissimilarity matrix (RDM) for data X,
    where dissimilarity for time pair (t1,t2) = 1 - corrcoef( X[t1], X[t2] )
        input:
            - input data: timebin X dimensions
        output:
            - flattened upper triangular matrix of the RDM
            - RDM
            - Order of time indices to sort the RDM matrix to get "nice" looking RDMs

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''

    dist_tol = 1e-10
    #RDM = (1 - np.corrcoef(input_data))
    
    # Compute the mean squared errors using vectorized operations
    reshaped_matrix = input_data[:, np.newaxis, :]
    RDM = np.mean((reshaped_matrix - input_data) ** 2, axis=2)**0.5

    RDM = ((RDM + RDM.T)/2) # make it perfectly symmetric
    RDM[RDM<dist_tol] = 0 # make the diagonal = 0
    RDM_sorted, res_order, res_linkage = compute_serial_matrix(RDM, 'ward')

    return RDM[np.triu_indices(RDM.shape[0], k=1)], RDM, res_order

if __name__=='main':
    num_time_bins = 500
    num_dims = 20
    original_resp = np.random.randn(num_time_bins, num_dims)  # Time X Dims
    
    RDM_utri_1, RDM_1, res_order_1 =get_RDM(original_resp + np.random.randn(num_time_bins,num_dims))
    RDM_utri_2, RDM_2, res_order_2 =get_RDM(original_resp + np.random.randn(num_time_bins,num_dims))

    print(f"Theory=.25 | true={np.corrcoef(RDM_utri_1, RDM_utri_2)[0,1]}") # theory .25 because equal RMS