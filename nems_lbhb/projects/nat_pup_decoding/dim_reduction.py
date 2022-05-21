"""
Tools for reducing dimensionality of natural sounds data prior to decoding.

Implementing PCA and PLS for dimensionality reduction. 

Meant to deal with trial limitation issues (e.g. where n_trials << n_neurons)

CRH 04/10/2020
"""
import numpy as np
from sklearn.decomposition import PCA
import scipy.signal as ss 

def get_one_hot_matrix(ncategories, nreps):
    # build Y matrix of one hot vectors
    Y = np.zeros((ncategories, ncategories * nreps))
    for stim in range(ncategories):
        yt = np.zeros((nreps, ncategories))
        yt[:, stim] = 1
        yt = yt.reshape(1, -1)
        Y[stim, :] = yt
    return Y


class TDR():
    """
    Custom dimensionality reduction method. 
    Finds axis 1 = difference between centroids of two stimuli
    Finds first PC of noise covariance matrix
    Then, find axis 2, the axis orthogonal to axis1 that completes the plane
    defined by PC1 and axis 1
    """
    def __init__(self, tdr2_init=None, n_additional_axes=None):
        '''
        if tdr2_init is NOT none, then use this to define TDR2 relative to dU
        if n_additional_axes is not None, add this many more "noise" dimensions.
            - this means the first n PCs of the noise data outside of the original 2D TDR space
        '''
        self.tdr2_init = tdr2_init
        self.n_additional_axes = n_additional_axes
        return None

    def fit(self, x, y):
        """
        x is shape observation x dim (neuron)
        y is one-hot vector of shape observation x 2 and specifies the ID of each stimulus
        """

        # get stim A and stim B matrices
        A = x[y[:,0]==1, :]
        B = x[y[:,1]==1, :]

        # get dU
        dU = A.mean(axis=0, keepdims=True) - B.mean(axis=0, keepdims=True)
        dU = dU / np.linalg.norm(dU)

        # get first PC of mean centered data
        if self.tdr2_init is None:
            pca = PCA(n_components=1)
            A0 = A - A.mean(axis=0, keepdims=True)
            B0 = B - B.mean(axis=0, keepdims=True)
            Xcenter = np.concatenate((A0, B0), axis=0)
            pca.fit(Xcenter)
            noise_axis = pca.components_
        else:
            noise_axis = self.tdr2_init

        # figure out the axis that spans the plane with dU
        noise_on_dec = (np.dot(noise_axis, dU.T)) * dU
        orth_ax = noise_axis - noise_on_dec
        orth_ax /= np.linalg.norm(orth_ax)

        weights = np.concatenate((dU, orth_ax), axis=0)

        if self.n_additional_axes is not None:
            # remove TDR projection
            A0 = A - A.mean(axis=0, keepdims=True)
            B0 = B - B.mean(axis=0, keepdims=True)
            Xcenter = np.concatenate((A0, B0), axis=0)
            Xresidual = Xcenter - Xcenter.dot(weights.T).dot(weights)

            # find n additional axes, orthogonal to TDR plane
            pca = PCA(n_components=self.n_additional_axes)
            pca.fit(Xresidual)
            noise_weights = pca.components_
            weights = np.concatenate((weights, noise_weights), axis=0)


        self.weights = weights

class PLS:
    """
    NIPALS implementation of partial least squares regression.
    
    Has an additional option (specific to NAT sounds project) 
    to convert the projection of X into power at a given 
    frequency band. Idea is that there are dimensions with high 
    frequency power that correlate with pupil size. 
    """

    def __init__(self, n_components=None, fs=None, low=None, high=None, max_iter=100, tol=1e-7):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.low = low
        self.high = high
        self.fs = fs

    
    def fit(self, X, y):
        '''
        Perform PLS regression, add result attributes
        '''

        if (self.fs is not None) & (self.low is not None) & (self.high is not None):
            # perform specialized PLS regression
            self.fit_power(X, y)

        else:

            x_alg = X.copy()
            y_alg = y.copy()

            max_iter = self.max_iter
            tol = self.tol

            xdim = x_alg.shape[0]
            ydim = y_alg.shape[0]
            nobs = x_alg.shape[1]

            x_weights = np.zeros((xdim, xdim))
            x_loadings = np.zeros((xdim, xdim))
            x_scores = np.zeros((nobs, xdim))
            y_weights = np.zeros((ydim, xdim))
            y_loadings = np.zeros((ydim, xdim))
            y_scores = np.zeros((nobs, xdim))
            for i in range(0, xdim):
                u = y_alg[0, :][np.newaxis, :]
                cost = 1
                iteration = 0
                while (cost > tol) & (iteration < max_iter):
                    # ==== X block ====
                    w = (u @ x_alg.T) / (u @ u.T) # "regressing" X against Y
                    w = w / np.linalg.norm(w)
                    if iteration != 0:
                        t_last = t
                    else: 
                        t_last = np.nan
                    t = (x_alg.T @ w.T)            # project X onto w

                    if np.any(~np.isnan(t_last)):
                        # when t (the score) stops changing, w is optimized to maximize covariation
                        # between t and y (keeping in mind that t is a projection of X).
                        # i.e. on each iteration, w is shifting towards the direction of maximal subspace alignment
                        # between X and Y. t is the subspace of X and u is the subspace of Y.
                        cost = np.linalg.norm(t_last - t) 
                    else: 
                        cost = 2*tol

                    # ==== Y block ==== 
                    q = (t.T @ y_alg.T) / (t.T @ t)  # "regressing" Y against the score of X (projection on pls component)
                    u = (q @ y_alg) / (q @ q.T)

                    iteration+=1

                # calculate X loadings
                p = (t.T @ x_alg.T) / (t.T @ t)  # regression slope of every column in X onto t (scores)

                # compute new x_alg / y_alg (residuals)
                x_alg = x_alg - (t @ p).T
                y_alg = y_alg - (t @ q).T

                # save values
                x_scores[:, i] = t.squeeze()
                y_scores[:, i] = u.squeeze()

                x_weights[:, i] = w.squeeze()
                y_weights[:, i] = q.squeeze()

                x_loadings[:, i] = p.squeeze()
                y_loadings[:, i] = q.squeeze()

            if (iteration >= max_iter) & (cost > tol):
                print("did not converge")

            # set object attributes
            self.x_scores = x_scores
            self.y_scores = y_scores
            
            self.x_weights = x_weights
            self.y_weights = y_weights

            self.x_loadings = x_loadings
            self.y_loadings = y_loadings


    def fit_power(self, X, y):
        """
        specialized PLS fit. Fit power in projection of x to y.
        y must be one-dimensional for this!
        """

        x_alg = X.copy()
        y_alg = y.copy()

        # transform x_alg into power
        f, t, s = ss.spectrogram(x_alg.squeeze(), fs=self.fs)

        # figure out which channels of spectrogram to keep
        f_idx = np.argwhere((f > self.low) & (f < self.high))
        x_alg = s[:, f_idx, :].sum(axis=1).squeeze()
        
        nobs = x_alg.shape[-1]

        # downsample y to match
        y_alg = ss.resample(y_alg, nobs, axis=-1)

        max_iter = self.max_iter
        tol = self.tol

        xdim = x_alg.shape[0]
        ydim = y_alg.shape[0]

        if ydim > 1:
            raise ValueError("Dependent variable must be 1-D for specialized PLS fit")

        if self.n_components is not None:
            ncomp = self.n_components
        else:
            ncomp = xdim

        # preallocate space
        x_weights = np.zeros((xdim, ncomp))
        x_loadings = np.zeros((xdim, ncomp))
        x_scores = np.zeros((nobs, ncomp))
        for i in range(0, ncomp):
            u = y_alg[0, :][np.newaxis, :]
            cost = 1
            iteration = 0
            while (cost > tol) & (iteration < max_iter):
                # ==== X block ====
                w = (u @ x_alg.T) / (u @ u.T) # "regressing" X against Y
                w = w / np.linalg.norm(w)
                if iteration != 0:
                    t_last = t
                else: 
                    t_last = np.nan
                t = (x_alg.T @ w.T)            # project X onto w

                if np.any(~np.isnan(t_last)):
                    # when t (the score) stops changing, w is optimized to maximize covariation
                    # between t and y (keeping in mind that t is a projection of X).
                    # i.e. on each iteration, w is shifting towards the direction of maximal subspace alignment
                    # between X and Y. t is the subspace of X and u is the subspace of Y.
                    cost = np.linalg.norm(t_last - t) 
                else: 
                    cost = 2*tol

                # ==== Y block ==== 
                # no Y block, because one-dimensional

                iteration+=1

            # calculate X loadings
            p = (t.T @ x_alg.T) / (t.T @ t)  # regression slope of every column in X onto t (scores)

            # compute new x_alg / y_alg (residuals)
            x_alg = x_alg - (t @ p).T

            # save values
            x_scores[:, i] = t.squeeze()

            x_weights[:, i] = w.squeeze()

            x_loadings[:, i] = p.squeeze()

        if (iteration >= max_iter) & (cost > tol):
            print("did not converge")

        # set object attributes
        self.x_scores = x_scores
        self.y_scores = None
        
        self.x_weights = x_weights
        self.y_weights = None

        self.x_loadings = x_loadings
        self.y_loadings = None