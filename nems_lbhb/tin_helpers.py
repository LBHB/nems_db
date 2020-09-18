import numpy as np
import pandas as pd

def sort_targets(targets):
    """
    sort target epoch strings by freq, then by snr, then by targets tag (N1, N2 etc.)
    """
    f = []
    snrs = []
    labs = []
    for t in targets:
        f.append(int(t.strip('TAR_').strip('CAT_').split('+')[0]))
        snr = t.split('+')[1].split('dB')[0]
        if snr=='Inf': snr=np.inf
        elif snr=='-Inf': snr=-np.inf
        else: snr=int(snr)
        snrs.append(snr)
        try:
            labs.append(int(t.split('+')[-1].split(':N')[-1]))
        except:
            labs.append(np.nan)
    tar_df = pd.DataFrame(data=np.stack([f, snrs, labs]).T, columns=['freq', 'snr', 'n']).sort_values(by=['freq', 'snr', 'n'])
    sidx = tar_df.index
    return np.array(targets)[sidx].tolist()


def get_snrs(targets):
    """
    return list of snrs for each target
    """
    snrs = []
    for t in targets:
        snr = t.split('+')[1].split('dB')[0]
        if snr=='Inf': snr=np.inf 
        elif snr=='-Inf': snr=-np.inf
        else: snr=int(snr)
        snrs.append(snr)
    return snrs


def get_tar_freqs(targets):
    """
    return list of target freqs
    """
    return [int(t.split('+')[0]) for t in targets]


def compute_ellipse(x, y):
    inds = np.isfinite(x) & np.isfinite(y)
    x= x[inds]
    y = y[inds]
    data = np.vstack((x, y))
    mu = np.mean(data, 1)
    data = data.T - mu
    D, V = np.linalg.eig(np.divide(np.matmul(data.T, data), data.shape[0] - 1))
    # order = np.argsort(D)[::-1]
    # D = D[order]
    # V = abs(V[:, order])
    t = np.linspace(0, 2 * np.pi, 100)
    e = np.vstack((np.sin(t), np.cos(t)))  # unit circle
    VV = np.multiply(V, np.sqrt(D))  # scale eigenvectors
    e = np.matmul(VV, e).T + mu  # project circle back to orig space
    e = e.T
    return e