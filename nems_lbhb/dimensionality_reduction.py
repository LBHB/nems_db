import numpy as np


class npPCA:
    """
    numpy implementation of PCA. Should behave exactly the same
    as sklearn.decomposition.PCA
    """
    def __init__(self, n_components=None):
        """
        n_components - number of PCA components to return
        """
        self.n_components = n_components

    def fit(self, X):
        """
        X - numpy array, shape observations X variables
        """

        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        # make sure X is centered for computing covariance
        X_center = X - X.mean(axis=0, keepdims=True)

        # compute covariance matrix
        cov = np.cov(X_center.T)

        # compute evecs / evals
        ev, eg = np.linalg.eig(cov)

        # sort evecs
        sort_args = np.argsort(ev)[::-1]
        self.components = eg[:, sort_args][:, 0:n_components].T
        ev = ev[sort_args]

        # compute variance explained ratio
        self.var_explained_ratio = ev[0:n_components] / np.sum(ev)

        # compute PC transform (projection of X back onto PCs)
        self.X_transform = np.matmul(X_center, self.components.T)



def get_noise_projection(rec, epochs=None):
    """
    Compute first PC of pooled noise data and project data onto this axis.

    If epochs is None, pool data over all epochs within recording mask. 
        Else, only pool data across the specified epochs.

    return new recording with new signal "noise_projection"
    """
    # copy recording
    r = rec.copy()
    
    # get list of stim epochs
    if epochs is None:
        epochs = [e for e in rec.epochs.name.unique() if ('STIM_' in e) | ('TAR_' in e)]
    elif type(epochs) is not list:
        epochs = list(epochs)

    # extract resp dictionary
    R = rec['resp'].extract_epochs(epochs, mask=rec['mask'])

    # concatenate mean centered data for each stimulus
    R_center = []
    ncells = rec['resp'].shape[0]
    for e in R.keys():
        _resp = R[e].reshape(ncells, -1)
        _resp_center = _resp - _resp.mean(axis=-1)
        R_center.append(_resp_center)

    R_center = np.concatenate(R_center, axis=-1)

    # do PCA on the pooled data
    pca = npPCA(n_components=1)
    pca.fit(R_center.T)

    # finally, project the RAW trace back onto this axis
    resp = r['resp'].as_continuous()
    projection = np.matmul(resp.T, pca.components.T).T

    # create projection signal and add to new recording
    r['noise_projection'] = r['resp']._modified_copy(projection)

    return r

