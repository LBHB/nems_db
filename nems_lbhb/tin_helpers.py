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
