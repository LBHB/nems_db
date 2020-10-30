"""
Helper plotting functions for behavior data.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_RT_histogram(rts, DI=None, bins=None, ax=None, cmap=None, lw=1):
    """
    rts:    reaction times dictionary. keys are epochs, values are list of RTs
    DI:     dict with each target's DI. Vals get added to legend
    bins:   either int or range to specify bins for the histogram. If int, will use 
                that many bins between 0 and 2 sec
    ax:     default is None. If not None, make the plot on the specified axis
    cmap:   mpl iterable colormap generator (see tin_helpers.make_tbp_colormaps)
    """
    if bins is None:
        bins = np.arange(0, 2, 0.1)
    
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))

    skeys = rts.keys()

    # Now, for each soundID (reference, target1, target2 etc.),
    # create histogram
    for i, k in enumerate(skeys):
        counts, xvals = np.histogram(rts[k], bins=bins)
        if DI is not None:
            try:
                _di = round(DI[k], 2)
            except:
                _di = 'N/A'
        else:
            _di = 'N/A'
        n = len(rts[k])
        if cmap is not None:
            color = cmap(i)
        else:
            color = None
        ax.step(xvals[:-1], np.cumsum(counts) / len(rts[k]), 
                    label=f'{k}, DI: {_di}, n: {n}', lw=lw, color=color)
    
    ax.legend(frameon=False)
    ax.set_xlabel('Reaction time (s)')
    ax.set_ylabel('Cummulative Probability')

    return ax
