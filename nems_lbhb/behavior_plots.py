"""
Helper plotting functions for behavior data.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_RT_histogram(rts, DI=None, bins=None, ax=None):
    """
    rts:    reaction times dictionary. keys are epochs, values are list of RTs
    DI:     dict with each target's DI. Vals get added to legend
    bins:   either int or range to specify bins for the histogram. If int, will use 
                that many bins between 0 and 2 sec
    ax:     default is None. If not None, make the plot on the specified axis
    """
    if bins is None:
        bins = np.arange(0, 2, 0.1)
    
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))

    skeys = np.sort(list(rts.keys()))

    # Now, for each soundID (reference, target1, target2 etc.),
    # create histogram
    for k in skeys:
        counts, xvals = np.histogram(rts[k], bins=bins)
        if DI is not None:
            try:
                _di = round(DI[k], 2)
            except:
                _di = 'N/A'
        else:
            _di = 'N/A'
        n = len(rts[k])
        ax.step(xvals[:-1], np.cumsum(counts) / len(rts[k]), 
                    label=f'{k}, DI: {_di}, n: {n}', lw=1)
    
    ax.legend(fontsize=6, frameon=False)
    ax.set_xlabel('Reaction time (s)', fontsize=8)
    ax.set_ylabel('Cummulative Probability', fontsize=8)

    return ax
