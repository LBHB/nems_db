import pylab as pl
import numpy as np

import nems0.modelspec


def plot_gain(rec, modelspec, n_targets):
    rec = nems0.modelspec.evaluate(rec, modelspec)

    def get_mean(r, signal, epoch_name):
        e = r[signal].extract_epoch(epoch_name)
        return np.nanmean(e, axis=0).T

    phi = modelspec[-3]['phi']

    figures = []

    for i in range(n_targets):
        f, axes = pl.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 5))
        axes[0, 1].plot(get_mean(rec, 'resp', f'target_{i}_repeating_single'), 'k')
        axes[0, 1].plot(get_mean(rec, 'pred', f'target_{i}_repeating_single'), 'r')
        axes[0, 0].plot(get_mean(rec, 'resp', f'target_{i}_random_single'), 'k')
        axes[0, 0].plot(get_mean(rec, 'pred', f'target_{i}_random_single'), 'r')
        axes[1, 1].plot(get_mean(rec, 'resp', f'target_{i}_repeating_dual'), 'k')
        axes[1, 1].plot(get_mean(rec, 'pred', f'target_{i}_repeating_dual'), 'r')
        axes[1, 0].plot(get_mean(rec, 'resp', f'target_{i}_random_dual'), 'k')
        axes[1, 0].plot(get_mean(rec, 'pred', f'target_{i}_random_dual'), 'r')

        bg = np.exp(phi['bg_gain'][i])
        fg = np.exp(phi['fg_gain'][i])
        ss = np.exp(phi['ss_gain'][i])

        axes[0, 1].text(1.1, 0.5, f'SS: {ss:.2f}', transform=axes[0, 1].transAxes)
        axes[1, 1].text(1.1, 0.5, f'BG: {bg:.2f}\nFG: {fg:.2f}', transform=axes[1, 1].transAxes)

        axes[0, 0].set_title('random phase')
        axes[0, 1].set_title('repeating phase')
        axes[0, 0].set_ylabel('single stream')
        axes[1, 0].set_ylabel('dual stream')

        figures.append(f)

    return figures
