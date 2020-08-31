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



def get_noise_projection(rec, epochs=None, collapse=False):
    """
    Compute first PC of pooled noise data and project data onto this axis.

    If epochs is None, pool data over all epochs within recording mask. 
        Else, only pool data across the specified epochs.
    
    If collapse=True, average over all bins in each epoch. e.g. single data point
    for each epoch repetition.

    ***NOTE***
    This function will compute the axis only using the resp data contained in
    rec['mask'], but it will project *all* resp data onto the resulting axis.
    Thus, if you'd like to exclude, for example, PreStimSilence from the axis estimation,
    do this in a preprocessing step by creating the appropriate mask signal. Alternatively,
    you can make custom epochs in a preprocessing step and then just pass these as
    fn arguments.

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
    R = rec['resp'].extract_epochs(epochs, mask=rec['mask'], allow_incomplete=True)

    # concatenate mean centered data for each stimulus
    R_center = []
    ncells = rec['resp'].shape[0]
    for e in R.keys():
        if not collapse:
            _resp = R[e].transpose([1, 0, 2]).reshape(ncells, -1)
        else:
            _resp = R[e].transpose([1, 0, 2]).mean(axis=-1, keepdims=True).reshape(ncells, -1)

        _resp_center = _resp - _resp.mean(axis=-1, keepdims=True)
        R_center.append(_resp_center)

    R_center = np.concatenate(R_center, axis=-1)

    # do PCA on the pooled data
    pca = npPCA(n_components=1)
    pca.fit(R_center.T)

    # finally, project the RAW trace back onto this axis
    resp = r['resp'].as_continuous()
    projection = np.matmul(resp.T, pca.components.T).T

    # rectify the projection just for "readibility" (bc sign of PC is arbitrary) 
    if np.mean(projection) < 1:
        projection = -1 * projection

    # create projection signal and add to new recording
    r['noise_projection'] = r['resp']._modified_copy(projection)

    return r


def get_discrimination_projection(rec, epoch1='TARGET', epoch2='REFERENCE', collapse=False):
    """
    Projection resp data onto the axis: mean(epoch1) minus mean(epoch2)

    epoch1: Name of first epoch 
    epoch2: Name of second epoch 

    If collapse=True, average over all bins in each epoch. e.g. single data point
        for each epoch repetition.

    ***NOTE***
    This function will compute the axis only using the resp data contained in
    rec['mask'], but it will project *all* resp data onto the resulting axis.
    Thus, if you'd like to exclude, for example, PreStimSilence from the axis estimation,
    do this in a preprocessing step by creating the appropriate mask signal. Alternatively,
    you can make custom epochs in a preprocessing step and then just pass these as
    fn arguments.

    Some common uses:
        define a TARGET vs. REFERENCE discrimination axis (default)
        define a HIT vs. MISS discimination axis, i.e. a choice axis
                This example requires some careful thinking about how
                you build your mask signal e.g. you probably only want
                to define this axis my comparing HIT vs. MISS acitivty in
                a certain TARGET window. Alternatively, you could define 
                two new specialized epoch names, add these to the rec.epochs, 
                then pass these as epoch1/epoch2 (see demo_pop_projections.py for an example).

    Return new recording with new signal "epoch1_vs_epoch2_projection"
    """
    # copy recording
    r = rec.copy()
    ncells = r['resp'].shape[0]

    # extract response matrices
    r1 = r['resp'].extract_epoch(epoch1, mask=r['mask'], allow_incomplete=True)
    r2 = r['resp'].extract_epoch(epoch2, mask=r['mask'], allow_incomplete=True)

    if collapse:
        r1 = r1.mean(axis=-1, keepdims=True)
        r2 = r2.mean(axis=-1, keepdims=True)
    
    r1 = r1.transpose([1, 0, 2]).reshape(ncells, -1)
    r2 = r2.transpose([1, 0, 2]).reshape(ncells, -1)

    # define discrimination axis (delta mu)
    disc_axis = r1.mean(axis=-1, keepdims=True) - r2.mean(axis=-1, keepdims=True)
    disc_axis /= np.linalg.norm(disc_axis)

    # project full response onto axis
    projection = np.matmul(rec['resp'].as_continuous().T, disc_axis)

    # rectify the projection just for "readibility" (bc sign of axis is arbitrary) 
    if np.mean(projection) < 1:
        projection = -1 * projection

    signal_name = "{0}_vs_{1}_projection".format(epoch1, epoch2)
    r[signal_name] = rec['resp']._modified_copy(projection.T)

    return r
    

