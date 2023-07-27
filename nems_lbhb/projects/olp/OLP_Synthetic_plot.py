import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
import copy
import nems_lbhb.projects.olp.OLP_helpers as ohel
import nems_lbhb.projects.olp.OLP_fit as ofit
from scipy import stats
from nems_lbhb.baphy_experiment import BAPHYExperiment
import glob

import nems0.db as nd
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import copy
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems_lbhb.projects.olp.OLP_helpers as ohel
import nems_lbhb.projects.olp.OLP_fit as ofit
import nems_lbhb.projects.olp.OLP_Synthetic_plot as osyn
import nems_lbhb.projects.olp.OLP_Binaural_plot as obip
import nems_lbhb.projects.olp.OLP_plot_helpers as oph
import nems_lbhb.projects.olp.OLP_figures as ofig
import nems_lbhb.projects.olp.OLP_plot_helpers as opl
import nems_lbhb.projects.olp.OLP_poster as opo
import scipy.ndimage.filters as sf
from scipy import stats
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems0.analysis.gammatone.gtgram import gtgram
from scipy.io import wavfile
import glob
import nems0.epoch as ep
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def plot_synthetic_weights(weight_df, plotA='weightsA', plotB='weightsB', areas=None, thresh=0.03, quads=3,
                           synth_show=None, r_cut=None, title=None):
    '''2022_09_21. Added model fit filter (r_cut)
    Plot a bar graph comparing the BG and FG weights for the different synthetic conditions. Can
    specify if you want one or both areas and also which combination of synthetic conditions you
    want to plot, as described by a list of the strings for their codes. If you want to plot all minus
    the control for the control (A), simply use A- as the synth_show.'''
    quad, threshold = ohel.quadrants_by_FR(weight_df, threshold=thresh, quad_return=quads)
    if len(plotA.split('_')) > 1:
        r_name = f"r_{plotA.split('_')[-1]}"
    else:
        r_name = 'r'

    if r_cut:
        quad = quad.dropna(axis=0, subset=r_name)
        quad = quad.loc[quad[r_name] >= r_cut]
    #Create aliases for the kinds so I can dumbly swap them out all so they can be in the right order
    alias = {'A': '1', 'N': '2', 'M': '3', 'U': '4', 'S': '5', 'T': '6', 'C': '7'}
    kind_alias = {'1': 'Non-RMS Norm\nNatural', '2': 'RMS Norm\nNatural', '3': 'Spectrotemporal\nModulation',
                  '4': 'Spectro-\ntemporal', '5': 'Spectral', '6': 'Temporal', '7': 'Cochlear'}
    kind_alias = {'A': 'Non-RMS Norm Natural', 'N': 'RMS Norm Natural', 'M': 'Spectrotemporal\nModulation',
                  'U': 'Spectrotemporal', 'S': 'Spectral', 'T': 'Temporal', 'C': 'Cochlear'}


    # Shortcut to typing in all the actual conditions minus the control of the control
    if synth_show == 'A-':
        synth_show = ['N', 'M', 'U', 'S', 'T', 'C']

    # This let's you only show certain synthetic kinds, but it will always be ordered in the same way
    if synth_show:
        quad = quad.loc[quad['synth_kind'].isin(synth_show)]
        width = len(synth_show) + 1
    else:
        width = 8

    # If you just want one area it can do that, or a list of areas. If you do nothing it'll plot
    # as many areas as are represented in the df (should only be two at most...)
    if isinstance(areas, str):
        fig, axes = plt.subplots(1, len(synth_show), figsize=(width*2,4))
        areas = [areas]
    elif isinstance(areas, list):
        fig, axes = plt.subplots(len(areas), len(synth_show), figsize=(width*2,4*len(areas)), sharey=True)
    else:
        fig, axes = plt.subplots(len(weight_df.area.unique()), (len(synth_show)),
                                 figsize=(width*2,4*(len(weight_df.area.unique()))), sharey=True)
        areas = weight_df.area.unique().tolist()
    axes = np.ravel(axes)

    if len(areas) > 1:
        synth_show_full = synth_show * 2
        area_show = [[nn] * len(synth_show) for nn in areas]
        area_show = [item for sublist in area_show for item in sublist]
    else:
        synth_show_full = synth_show
        area_show = [areas] * len(synth_show)

    for cnt, (ax, ar, syn) in enumerate(zip(axes, area_show, synth_show_full)):
        area_df = quad.loc[quad.area == ar]
        # Extract only the relevant columns for plotting right now
        to_plot = area_df.loc[:,['synth_kind', plotA, plotB]].copy()
        to_plot = to_plot.loc[to_plot.synth_kind == syn]
        # Sort them by the order of kinds so it'll plot in an order that I want to see
        # to_plot['sort'] = to_plot['synth_kind']
        # to_plot = to_plot.replace(alias).sort_values('sort').drop('sort', axis=1).copy()
        to_plot = to_plot.melt(id_vars='synth_kind', value_vars=[plotA, plotB], var_name='weight_kind',
                     value_name='weights').replace({plotA:'BG', plotB:'FG'}).replace(kind_alias)

        # Plot
        sb.barplot(ax=ax, x="synth_kind", y="weights", hue="weight_kind", data=to_plot, ci=68, estimator=np.mean)
        if cnt == 0 or cnt == len(synth_show):
            ax.set_ylabel(f"{ar}\n\nweights", fontweight='bold', fontsize=10)
        else:
            ax.set_ylabel('')
        ax.set_xticklabels(''), ax.set_xlabel('')
        ax.legend(loc='upper right')
        if cnt < len(synth_show):
            ax.set_title(f"{kind_alias[syn]}", fontweight='bold', fontsize=10)
        fig.suptitle(f"{title} - {plotA}, {plotB}")


def plot_ramp_comparison(weight_df, thresh=0.03, quads=3):
    '''Plot weights of synthetic groups divided by sites that were pre-click removal. This will not get much use... but
    it's here.'''
    quad, threshold = ohel.quadrants_by_FR(weight_df, threshold=thresh, quad_return=quads)
    # Create aliases for the kinds so I can dumbly swap them out all so they can be in the right order
    alias = {'A': '1', 'N': '2', 'C': '3', 'T': '4', 'S': '5', 'U': '6', 'M': '7'}
    kind_alias = {'1': 'Non-RMS\nNatural', '2': ' RMS \nNatural', '3': 'Cochlear',
                  '4': 'Temporal', '5': 'Spectral', '6': 'Spectro-\ntemporal', '7': 'Spectrotemporal\nModulation'}
    to_plot = quad.sort_values('cellid').copy()
    to_plot['ramp'] = to_plot.cellid.str[3:6]
    to_plot['ramp'] = pd.to_numeric(to_plot['ramp'])
    # Gets rid of PEG sites so I'm comparing A1 to A1, this is a dumb function.
    to_plot = to_plot.loc[to_plot.ramp < 46]
    # Makes a new column that labels the early synthetic sites with click
    to_plot['has_click'] = np.logical_and(to_plot['ramp'] <= 37, to_plot['ramp'] >= 27)
    test_plot = to_plot.loc[:, ['synth_kind', 'weightsA', 'weightsB', 'has_click']].copy()
    test_plot['sort'] = test_plot['synth_kind']
    test_plot = test_plot.replace(alias).sort_values('sort').drop('sort', axis=1).copy()
    test_plot = test_plot.melt(id_vars=['synth_kind', 'has_click'], value_vars=['weightsA', 'weightsB'],
                               var_name='weight_kind',
                               value_name='weights').replace({'weightsA': 'BG', 'weightsB': 'FG'}).replace(kind_alias)
    test_plot['has_click'] = test_plot['has_click'].astype(str)
    test_plot = test_plot.replace({'True': ' - With Click', 'False': ' - Clickless'})
    test_plot['kind'] = test_plot['weight_kind'] + test_plot['has_click']
    test_plot = test_plot.drop(labels=['has_click'], axis=1)
    fig, axes = plt.subplots(1, 1, figsize=(8, 4))
    sb.barplot(ax=axes, x="synth_kind", y="weights", hue="kind", data=test_plot, ci=68, estimator=np.mean)
    axes.set_xlabel('')
    axes.legend(loc='upper right')
    axes.set_ylabel('Model Weights', fontsize=8, fontweight='bold')


def relative_gain_synthetic_histograms(weight_df, thresh=0.03, quads=3, synth_show=None):
    '''Will plot however many subplots of the synthetic conditions you define using synth_show
    next to one another showing relative gain and telling you what percentage of cells are
    suppressed in each condition. Passing 'A-' will show everything minus the non-RMS control.
    Moved from main function 2022_08_29.'''
    quad, _ = ohel.quadrants_by_FR(weight_df, threshold=thresh, quad_return=quads)

    # Create aliases for the kinds so I can dumbly swap them out all so they can be in the right order
    alias = {'A': '1', 'N': '2', 'C': '3', 'T': '4', 'S': '5', 'U': '6', 'M': '7'}
    kind_alias = {'1': 'Non-RMS Norm\nNatural', '2': 'RMS Norm\nNatural', '3': 'Cochlear',
                  '4': 'Temporal', '5': 'Spectral', '6': 'Spectro-\ntemporal', '7': 'Spectrotemporal\nModulation'}

    # Shortcut to typing in all the actual conditions minus the control of the control
    if synth_show == 'A-':
        synth_show = ['N', 'C', 'T', 'S', 'U', 'M']

    # This let's you only show certain synthetic kinds, but it will always be ordered in the same way
    if synth_show:
        # If you only put one condition to show it still needs to be a list
        if isinstance(synth_show, str):
            synth_show = [synth_show]
        quad = quad.loc[quad['synth_kind'].isin(synth_show)]
        width = len(synth_show) + 1
    else:
        width = 8
        synth_show = quad.synth_kind.unique().tolist()

    # Extract only the relevant columns for plotting right now
    to_plot = quad.loc[:, ['synth_kind', 'weightsA', 'weightsB', 'area']].copy()
    # Sort them by the order of kinds so it'll plot in an order that I want to see
    to_plot['sort'] = to_plot['synth_kind']
    to_plot = to_plot.replace(alias).sort_values('sort').drop('sort', axis=1).copy()
    # Put the dataframe into a format that can be plotted easily
    to_plot = to_plot.replace(kind_alias)

    fig, axes = plt.subplots(1, len(synth_show), figsize=(4+len(synth_show), 4), sharey=True)

    # if you only put one condition to show axes won't be a list and can't be iterated
    if ~isinstance(axes, list):
        axes = [axes]

    for (ax, synth) in zip(axes, to_plot.synth_kind.unique().tolist()):
        kind_df = to_plot.loc[to_plot.synth_kind == synth]

        a1_df = kind_df.loc[kind_df.area == 'A1']
        peg_df = kind_df.loc[kind_df.area == 'PEG']

        a1_rel_weight = (a1_df.weightsB - a1_df.weightsA) / (a1_df.weightsB + a1_df.weightsA)
        a1_supps = [cc for cc in a1_rel_weight if cc < 0]
        a1_percent_supp = np.around((len(a1_supps) / len(a1_rel_weight)) * 100, 1)

        peg_rel_weight = (peg_df.weightsB - peg_df.weightsA) / (peg_df.weightsB + peg_df.weightsA)
        peg_supps = [cc for cc in peg_rel_weight if cc < 0]
        peg_percent_supp = np.around((len(peg_supps) / len(peg_rel_weight)) * 100, 1)

        rel_weight = (kind_df.weightsB - kind_df.weightsA) / (kind_df.weightsB + kind_df.weightsA)
        supps = [cc for cc in rel_weight if cc < 0]
        percent_supp = np.around((len(supps) / len(rel_weight)) * 100, 1)

        edges = np.arange(-2, 2, .05)
        na, xa = np.histogram(a1_rel_weight, bins=edges)
        na = na / na.sum() * 100
        ax.hist(xa[:-1], xa, weights=na, histtype='step', color='violet', lw=1)

        nb, xb = np.histogram(peg_rel_weight, bins=edges)
        nb = nb / nb.sum() * 100
        ax.hist(xb[:-1], xb, weights=nb, histtype='step', color='wheat', lw=1)

        nc, xc = np.histogram(rel_weight, bins=edges)
        nc = nc / nc.sum() * 100
        ax.hist(xc[:-1], xc, weights=nc, histtype='step', color='black', lw=1)

        ax.legend((f"A1 - %supp: {a1_percent_supp}", f"PEG - %supp: {peg_percent_supp}",
                     f"All - %supp: {percent_supp}"), fontsize=5)
        ax.set_xlabel('Relative\nGain', fontsize=6, fontweight='bold')
        ax.set_title(synth, fontsize=8, fontweight='bold')

    axes[0].set_ylabel('Percent of Cells', fontsize=7, fontweight='bold')
    ymin, ymax = ax.get_ylim()
    for aa in axes:
        aa.vlines(0, ymin, ymax, ls=':', lw=0.5)


def synthetic_relative_gain_comparisons_specs(df, bg, fg, snr_threshold=0.12, thresh=0.03, quads=3, area='A1',
                                              batch=340, synth_show=None, r_cut=None, rel_cut=2.5,
                                              figsize=(12,20)):
    '''Made 2022_09_08. Makes the big figure on panel 5 of my 2022 NGP Poster. It takes a dataframe
    and a list of the synthetic code letters and vertically arranges them as histograms of relative
    gain, all aligned on the same zero so you can ideally see the shift. You also can put in a BG
    and FG name (with spaces in the name if it has!) and it will show the corresponding spectrograms
    for the synthetic conditions nextdoor to the relative gain plot. It's a fun figure.'''
    if thresh:
        quad, _ = ohel.quadrants_by_FR(df, threshold=thresh, quad_return=quads)
    if snr_threshold:
        quad = df.loc[(df.bg_snr >= snr_threshold) & (df.fg_snr >= snr_threshold)]
    quad = quad.loc[quad.area==area]
    if r_cut:
        quad = quad.dropna(axis=0, subset='r')
        quad = quad.loc[quad.r >= r_cut]
    quad = quad.copy()

    # Create aliases for the kinds so I can dumbly swap them out all so they can be in the right order
    kind_alias = {'A': 'Non-RMS Norm\nNatural', 'N': 'Natural', 'M': 'Spectrotemporal\nModulation',
                  'U': 'Spectro-\ntemporal', 'S': 'Spectral', 'T': 'Temporal', 'C': 'Cochlear'}
    kind_dict = {'M': 'SpectrotemporalMod', 'U': 'Spectrotemporal', 'T': 'Temporal',
                  'S': 'Spectral', 'C': 'Cochlear'}

    # Shortcut to typing in all the actual conditions minus the control of the control
    if synth_show == 'A-':
        synth_show = ['N', 'C', 'T', 'S', 'U', 'M']
    if isinstance(synth_show, str):
        synth_show = [synth_show]

    lens = len(synth_show)
    hists, bgs, fgs = [], [], []
    # fig, axes = plt.subplots(len(synth_show), 3, figsize=(8, len(synth_show)*2))
    # fig, axes = plt.subplots(figsize=(8, lens*2))
    fig, axes = plt.subplots(figsize=figsize)
    for aa in range(lens):
        hist = plt.subplot2grid((lens*5, 15), (0+(aa*5), 11), rowspan=4, colspan=4)
        bgsp = plt.subplot2grid((lens*5, 15), (1+(aa*5), 0), rowspan=3, colspan=4)
        fgsp = plt.subplot2grid((lens*5, 15), (1+(aa*5), 5), rowspan=3, colspan=4)
        hists.append(hist), bgs.append(bgsp), fgs.append(fgsp)
    ax = hists + bgs + fgs

    ymins, ymaxs, xmins, xmaxs = [], [], [], []
    for qq in range(lens):
        to_plot = quad.loc[quad.synth_kind==synth_show[qq]].copy()
        # Calculate relative gain and percent suppressed
        rel_gain = (to_plot.weightsB - to_plot.weightsA) / \
                   (to_plot.weightsB + to_plot.weightsA)
        # rel_gain = to_plot.FG_rel_gain

        if rel_cut:
            rel_gain = rel_gain.loc[rel_gain <= rel_cut]
            rel_gain = rel_gain.loc[rel_gain >= -rel_cut]

        # Plot
        supps = [cc for cc in rel_gain if cc < 0]
        percent_supp = np.around((len(supps) / len(rel_gain)) * 100, 1)
        print(percent_supp)
        # Filter dataframe to get rid of the couple with super weird, big or small weights
        rel = rel_gain.loc[rel_gain <= 2.5]
        rel = rel.loc[rel >= -2.5]

        sups = [cc for cc in rel_gain if cc < 0]
        enhs = [cc for cc in rel_gain if cc >= 0]

        sup_edges = np.arange(-1.4, 0.1, 0.05)
        enh_edges = np.arange(0, 1.5, 0.05)
        na, xa = np.histogram(sups, bins=sup_edges)
        nb, xb = np.histogram(enhs, bins=enh_edges)
        aa = na / (na.sum() + nb.sum()) * 100
        bb = nb / (na.sum() + nb.sum()) * 100

        ax[qq].hist(xa[:-1], xa, weights=aa, histtype='step', color='tomato', fill=True)
        ax[qq].hist(xb[:-1], xb, weights=bb, histtype='step', color='dodgerblue', fill=True)

        ymin, ymax = ax[qq].get_ylim()
        ymaxs.append(ymax)
        # # Plot
        # p = sb.distplot(rel_gain, bins=50, color='black', norm_hist=True, kde=True, ax=ax[qq],
        #                 kde_kws=dict(linewidth=0.5))
        #
        # # Change color of suppressed to red, enhanced to blue
        # for rectangle in p.patches:
        #     if rectangle.get_x() < 0:
        #         rectangle.set_facecolor('tomato')
        # for rectangle in p.patches:
        #     if rectangle.get_x() >= 0:
        #         rectangle.set_facecolor('dodgerblue')
        # # This might have to be change if I use %, which I want to, but density is between 0-1, cleaner
        # ax[qq].set_yticks([0,1])
        # ax[qq].set_yticklabels([0,1])
        # All axes match natural. Would need to be changed if natural cuts for some reason.
        if qq == 0:
            ax[qq].set_ylabel('Percent of cells', fontweight='bold', fontsize=8)
            xmin, xmax = ax[qq].get_xlim()
            ax[qq].set_title(f'Area: {area}\nnr >= {r_cut}, quads={quads}\nrel_cut={rel_cut}', fontsize=7,
                             fontweight='bold', loc='right')
        else:
            ax[qq].set_xlim(xmin, xmax)
            ax[qq].set_ylabel('')

        if qq == (lens-1):
            ax[qq].set_xlabel('Relative Gain', fontweight='bold', fontsize=10)
        ax[qq].text((xmin+(np.abs(xmin)*0.1)), 0.75, f"Percent\nSuppression:\n{percent_supp}", fontsize=6)

        # Spectrogram parts
        manager = BAPHYExperiment(cellid=quad.iloc[0].cellid, batch=batch)
        folder_ids = [int(manager.get_baphy_exptparams()[-1]['TrialObject'][1]['ReferenceHandle'][1]['BG_Folder'][-1]),
                int(manager.get_baphy_exptparams()[-1]['TrialObject'][1]['ReferenceHandle'][1]['FG_Folder'][-1])]

        if synth_show[qq]=='A' or synth_show[qq]=='N':
            bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                f'Background{folder_ids[0]}/*.wav'))
            fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                f'Foreground{folder_ids[1]}/*.wav'))
        else:
            bg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                f'Background{folder_ids[0]}/{kind_dict[synth_show[qq]]}/*.wav'))
            fg_dir = glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                f'Foreground{folder_ids[1]}/{kind_dict[synth_show[qq]]}/*.wav'))

        bg_path = [bb for bb in bg_dir if bg in bb]
        fg_path = [ff for ff in fg_dir if fg in ff]

        if len(bg_path)==0 or len(fg_path)==0:
            raise ValueError(f"Your BG {bg} or FG {fg} aren't in there. Maybe add a space if it needs.")

        paths = [bg_path, fg_path]
        # 1 and 2 because that is how much will get added to do the different axes
        for ww in range(1,3):
            sfs, W = wavfile.read(paths[ww-1][0])
            spec = gtgram(W, sfs, 0.02, 0.01, 48, 100, 16000)
            qqq = qq + (ww * lens) # get you to correct axes
            ax[qqq].imshow(spec, aspect='auto', origin='lower', extent=[0,spec.shape[1], 0, spec.shape[0]],
                     cmap='gray_r')
            ax[qqq].set_xticks([]), ax[qqq].set_yticks([])
            ax[qqq].set_xticklabels([]), ax[qqq].set_yticklabels([])
            ax[qqq].spines['top'].set_visible(True), ax[qqq].spines['bottom'].set_visible(True)
            ax[qqq].spines['left'].set_visible(True), ax[qqq].spines['right'].set_visible(True)
            if ww == 1:
                ax[qqq].set_ylabel(f"{kind_alias[synth_show[qq]]}", fontsize=12, fontweight='bold',
                                   horizontalalignment='center', rotation=90, labelpad=40,
                                   verticalalignment='center')
            if qq == 0 and ww == 1:
                ax[qqq].set_title(f"BG: {bg}", fontweight='bold', fontsize=10)
            elif qq == 0 and ww == 2:
                ax[qqq].set_title(f"FG: {fg}", fontweight='bold', fontsize=10)
            # if qq == (lens - 1):
            #     ax[qqq].set_xlabel('Time (s)', fontweight='bold', fontsize=10)

    maxymax = np.max(ymaxs)
    for qq in range(len(synth_show)):
        ax[qq].set_ylim(0, maxymax)
        ax[qq].vlines(0, 0, maxymax, color='black', ls = '--', lw=1)


def synthetic_relative_gain_comparisons(df, snr_threshold=0.12, thresh=None, quads=3, area='PEG',
                                              synth_show=None, r_cut=None, rel_cut=2.5):
    '''Updated 2022_10_03. Took out the spectrograms from the original function.
    Made 2022_09_08. Makes the big figure on panel 5 of my 2022 NGP Poster. It takes a dataframe
    and a list of the synthetic code letters and vertically arranges them as histograms of relative
    gain, all aligned on the same zero so you can ideally see the shift. You also can put in a BG
    and FG name (with spaces in the name if it has!) and it will show the corresponding spectrograms
    for the synthetic conditions nextdoor to the relative gain plot. It's a fun figure.'''
    if thresh:
        quad, _ = ohel.quadrants_by_FR(df, threshold=thresh, quad_return=quads)
    if snr_threshold:
        quad = df.loc[(df.bg_snr >= snr_threshold) & (df.fg_snr >= snr_threshold)]
    if r_cut:
        quad = quad.dropna(axis=0, subset='r')
        quad = quad.loc[quad.r >= r_cut]
    if area:
        quad = quad.loc[quad.area == area]
    quad = quad.copy()

    # Create aliases for the kinds so I can dumbly swap them out all so they can be in the right order
    kind_dict = {'A': 'Non-RMS Norm\nNatural', 'N': 'Natural', 'M': 'Spectrotemporal\nModulation',
                  'U': 'Spectro-\ntemporal', 'S': 'Spectral', 'T': 'Temporal', 'C': 'Cochlear'}

    # Shortcut to typing in all the actual conditions minus the control of the control
    if synth_show == 'A-':
        synth_show = ['N', 'M', 'U', 'S', 'T', 'C']
    if isinstance(synth_show, str):
        synth_show = [synth_show]

    fig, axes = plt.subplots(len(synth_show), 1, figsize=(4, len(synth_show)*4))

    ymaxs = []
    for qq, (ax, syn) in enumerate(zip(axes, synth_show)):
        to_plot = quad.loc[quad.synth_kind==syn].copy()

        # Calculate percent suppressed
        rel_gain = (to_plot.weightsB - to_plot.weightsA) / \
                              (to_plot.weightsB + to_plot.weightsA)
        # rel_gain = to_plot.FG_rel_gain

        if rel_cut:
            rel_gain = rel_gain.loc[rel_gain <= rel_cut]
            rel_gain = rel_gain.loc[rel_gain >= -rel_cut]

        # Plot
        supps = [cc for cc in rel_gain if cc < 0]
        percent_supp = np.around((len(supps) / len(rel_gain)) * 100, 1)
        print(percent_supp)
        # Filter dataframe to get rid of the couple with super weird, big or small weights
        rel = rel_gain.loc[rel_gain <= 2.5]
        rel = rel.loc[rel >= -2.5]

        sups = [cc for cc in rel_gain if cc < 0]
        enhs = [cc for cc in rel_gain if cc >= 0]

        sup_edges = np.arange(-1.4, 0.1, 0.05)
        enh_edges = np.arange(0, 1.5, 0.05)
        na, xa = np.histogram(sups, bins=sup_edges)
        nb, xb = np.histogram(enhs, bins=enh_edges)
        aa = na / (na.sum() + nb.sum()) * 100
        bb = nb / (na.sum() + nb.sum()) * 100

        ax.hist(xa[:-1], xa, weights=aa, histtype='step', color='tomato', fill=True)
        ax.hist(xb[:-1], xb, weights=bb, histtype='step', color='dodgerblue', fill=True)

        ymin, ymax = ax.get_ylim()
        ymaxs.append(ymax)
        # This might have to be change if I use %, which I want to, but density is between 0-1, cleaner
        # ax.set_yticks([0,1])
        # ax.set_yticklabels([0,1])
        # All axes match natural. Would need to be changed if natural cuts for some reason.
        if qq == 0:
            ax.set_ylabel('Percent of cells', fontweight='bold', fontsize=8)
            xmin, xmax = ax.get_xlim()
        else:
            ax.set_xlim(xmin, xmax)
            ax.set_ylabel('')

        if qq == (len(synth_show)-1):
            ax.set_xlabel('Relative Gain', fontweight='bold', fontsize=10)
        else:
            ax.set_xticklabels([])
        ax.text((xmin+(np.abs(xmin)*0.1)), 0.75, f"Percent\nSuppression:\n{percent_supp}\nn={len(to_plot)}", fontsize=6)

        ax.set_ylabel(f"{kind_dict[syn]}", fontweight='bold', fontsize=10)

    maxymax = np.max(ymaxs)
    for ax in axes:
        ax.set_ylim(0, maxymax)
        ax.vlines(0, 0, maxymax, color='black', ls = '--', lw=1)

    fig.suptitle(f'Area: {area}\nr >= {r_cut}, quads={quads}\nrel_cut={rel_cut}, thresh={thresh}',
                 fontsize=7, fontweight='bold', ha='right', x=1)
    fig.tight_layout()


def sound_metric_scatter_all_synth(df, x_metrics, y_metric, x_labels, area='A1', threshold=0.03,
                         jitter=[0.25,0.2,0.03],
                         quad_return=3, metric_filter=None, synth_show=['A', 'N', 'M', 'U', 'S', 'T', 'C'],
                         bin_kind='11', title_text='', r_cut=None):
    '''Made from ofig.sound_metric_scatter 2022_09_21. Basically the same except plots all of the
    synthetic combinations you specify with those metrics. Also, if you input 'power' as a metric, it
    will know whether or not to plot max_power or RMS_power depending on what synth type you are looking
    at. Also can do cut offs of model fit with r_cut.
    Makes a series of scatterplots that compare a stat of the sounds to some metric of data. In
    a usual situation it would be Tstationariness, bandwidth, and Fstationariness compared to relative
    gain. Can also be compared to weights.
    y_metric refers to the FIRST one it will input, for relative_gain this is not an issue. If you want
    to differentiate between weights the sound affects in others vs how that sound is weighted itself,
    input the one as it relates to BG, so 'weightsB' will be 'how that sound effects others' and will
    know to make the metric 'weightsA' for the FGs, for example.
    When inputting x_metric names, always make it a list. All entries should be found in the df being
    passed, but you should remove the BG_ or FG_ prefix.
    Made into a function from OLP_analysis_main on 2022_09_07'''
    quad, _ = ohel.quadrants_by_FR(df, threshold=threshold, quad_return=quad_return)
    quad = quad.loc[(quad.area==area) & (quad.kind==bin_kind)]
    if r_cut:
        quad = quad.dropna(axis=0, subset='r')
        quad = quad.loc[quad.r >= r_cut]
    quad = quad.copy()

    # Create aliases for the kinds so I can dumbly swap them out all so they can be in the right order
    kind_alias = {'A': 'Non-RMS Norm\nNatural', 'N': 'Natural', 'M': 'Spectrotemporal\nModulation',
                  'U': 'Spectro-\ntemporal', 'S': 'Spectral', 'T': 'Temporal', 'C': 'Cochlear'}

    # Shortcut to typing in all the actual conditions minus the control of the control
    if synth_show == 'A-':
        synth_show = ['N', 'M', 'U', 'S', 'T', 'C']

    # This let's you only show certain synthetic kinds, but it will always be ordered in the same way
    if synth_show:
        # If you only put one condition to show it still needs to be a list
        if isinstance(synth_show, str):
            synth_show = [synth_show]
        quad = quad.loc[quad['synth_kind'].isin(synth_show)]
        width = len(synth_show) + 1
    else:
        width = 8
        synth_show = quad.synth_kind.unique().tolist()

    # I use 2.5 for relative gain, I'm sure weights have one too...
    if metric_filter:
        quad = quad.loc[quad[y_metric] <= metric_filter]
        quad = quad.loc[quad[y_metric] >= -metric_filter]

    if y_metric=='BG_rel_gain':
        y_metric2, title, ylabel = 'FG_rel_gain', 'Relative Gain', 'Relative Gain'
    elif y_metric=='weightsB':
        y_metric2, title, ylabel = 'weightsA', 'How this sound effects a concurrent sound', 'Weight'
    elif y_metric=='weightsA':
        y_metric2, title, ylabel = 'weightsB', 'How this sound itself is weighted', 'Weight'
    else:
        y_metric2, title, ylabel = y_metric, y_metric, y_metric

    fig, axes = plt.subplots(len(x_metrics), len(synth_show),
                             figsize=(len(synth_show)*2, len(x_metrics)*2))
    axes = np.ravel(axes, 'F')

    for scnt, ss in enumerate(synth_show):
        to_plot = quad.loc[quad.synth_kind==ss]
        if scnt == 0:
            ff = 0

        for cnt, met in enumerate(x_metrics):

            # Add a column that is the data for that metric, but jittered, for viewability
            if cnt == 0 and scnt==0:
                ax = axes[0]
                aa = 0
            else:
                ax = axes[aa + (ff * scnt)]

            # If 'power' is given as a metric, make it only plot max_power for 'A'
            if met=='power':
                if ss=='A':
                    met = 'max_power'
                else:
                    met = 'RMS_power'

            to_plot[f'jitter_BG_{met}'] = to_plot[f'BG_{met}'] + np.random.normal(0, jitter[cnt], len(to_plot))
            to_plot[f'jitter_FG_{met}'] = to_plot[f'FG_{met}'] + np.random.normal(0, jitter[cnt], len(to_plot))
            # Do the plotting
            sb.scatterplot(x=f'jitter_BG_{met}', y=y_metric, data=to_plot, ax=ax, s=2, color='cornflowerblue')
            sb.scatterplot(x=f'jitter_FG_{met}', y=y_metric2, data=to_plot, ax=ax, s=2, color='olivedrab')
            if cnt in range(0,len(x_metrics)):
                ax.set_ylabel(ylabel, fontweight='bold', fontsize=8)
            else:
                ax.set_ylabel('')
            ax.set_xlabel(f"{x_labels[cnt]}", fontsize=8, fontweight='bold')

            # Run a regression
            Y = np.concatenate((to_plot[y_metric].values, to_plot[y_metric2].values))
            X = np.concatenate((to_plot[f'BG_{met}'].values, to_plot[f'FG_{met}'].values))
            reg = stats.linregress(X, Y)
            x = np.asarray(ax.get_xlim())
            y = reg.slope * x + reg.intercept
            ax.plot(x, y, color='darkgrey', label=f"slope: {reg.slope:.3f}\n"
                                                     f"coef: {reg.rvalue:.3f}\n"
                                                     f"p = {reg.pvalue:.3f}")
            ax.legend()

            if cnt==0:
                ax.set_title(f"{kind_alias[ss]}", fontsize=10, fontweight='bold')

            aa += 1

    fig.suptitle(f"{title} - {title_text} - r >= {r_cut}", fontweight='bold', fontsize=10)
    fig.tight_layout()


def sound_stats_comp_scatter(sound_df, stat, main='N', comp=['M', 'U', 'S', 'T', 'C'], label=False):
    '''2022_09_23. Takes a sound_df and a list of stats (must be columns in that df) and will
    plot a scatter comparing those metrics of the synthetic condition given by main against
    all of the synthetic conditions listed in comp. There is no reason it shouldn't be able
    to take only a single one for stat and comp and just give you something really simple too.'''
    main_df = sound_df.loc[sound_df.synth_kind == main]
    if isinstance(stat, str):
        stat = [stat]

    # Create aliases for the kinds so I can dumbly swap them out all so they can be in the right order
    kind_alias = {'A': 'Non-RMS Norm\nNatural', 'N': 'Natural', 'M': 'Spectrotemporal\nModulation',
                  'U': 'Spectro-\ntemporal', 'S': 'Spectral', 'T': 'Temporal', 'C': 'Cochlear'}

    fig, axes = plt.subplots(len(stat), len(comp), figsize=(len(comp) * 4, len(stat) * 4))
    axes = np.ravel(axes)

    for cntt, st in enumerate(stat):
        if cntt == 0:
            ff = 0

        mins, maxs, axs = [], [], []
        for cnt, cc in enumerate(comp):
            if cnt == 0 and cntt == 0:
                ax = axes[0]
                aa = 0
            else:
                ax = axes[aa + (ff * cntt)]

            comp_df = sound_df.loc[sound_df.synth_kind == cc]
            to_plot = main_df[['name', 'type', st]]
            to_plot[f"comp"] = list(comp_df[st])
            sb.scatterplot(x=st, y='comp', data=to_plot, ax=ax, s=10, hue='type')
            ax.set_xlabel(kind_alias[main], fontsize=10, fontweight='bold')
            ax.set_ylabel(kind_alias[cc], fontsize=10, fontweight='bold')
            ax.set_title(st, fontsize=12, fontweight='bold')
            # ax.set_aspect("equal")

            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            mins.append(xmin), maxs.append(xmax)
            mins.append(ymin), maxs.append(ymax)
            axs.append((aa + (ff * cntt) - 1))

            # if you turn it on, it will label all the dots so you can see who's the weirdo
            if label:
                high = np.max([ymax, xmax])
                offset = high * 0.03
                xs, ys, ls = to_plot[st].to_list(), to_plot.comp.to_list(), to_plot.name.to_list()

                for ii, nn in enumerate(ls):
                    ax.annotate(nn, (xs[ii] + 0.2, ys[ii]), fontsize=4)

            aa += 1

        little = np.min(mins)
        big = np.max(maxs)
        for bb in axs:
            axes[bb + 1].set_xlim(little, big)
            axes[bb + 1].set_ylim(little, big)
            axes[bb + 1].plot([little, big], [little, big], color='darkgrey')

    fig.tight_layout()


def checkout_mods(sound_num, weight_df, thresh=0.03, quads=3, r_cut=None, area=None):
    '''2022_09_28. Takes a number from the list of names (you have to run it once first with
    a random number to get the indexes printed out) and will plot some modulation specs along
    the degraded synthetic sounds. Nice for browsing.'''
    quad, threshold = ohel.quadrants_by_FR(weight_df, threshold=thresh, quad_return=quads)
    if area:
        quad = quad.loc[quad.area == area]
    if r_cut:
        quad = quad.dropna(axis=0, subset='r')
        quad = quad.loc[quad.r >= r_cut]

    lfreq, hfreq, bins = 100, 24000, 48
    cid, btch = weight_df.cellid.iloc[0], weight_df.batch.iloc[0]
    manager = BAPHYExperiment(cellid=cid, batch=btch)
    expt_params = manager.get_baphy_exptparams()
    ref_handle = expt_params[-1]['TrialObject'][1]['ReferenceHandle'][1]
    BG_folder, FG_folder = ref_handle['BG_Folder'], ref_handle['FG_Folder']

    bbs = list(set([bb.split('_')[1][:2] for bb in weight_df.epoch]))
    ffs = list(set([ff.split('_')[2][:2] for ff in weight_df.epoch]))
    bbs.sort(key=int), ffs.sort(key=int)

    # synths = list(weight_df.synth_kind.unique())
    kind_dict = {'M': 'SpectrotemporalMod', 'U': 'Spectrotemporal', 'T': 'Temporal',
                 'S': 'Spectral', 'C': 'Cochlear'}

    # sound_num = 27

    fig, axes = plt.subplots(3, 12, figsize=(20, 5))
    ax = np.ravel(axes, 'F')
    synths = ['N', 'M', 'U', 'S', 'T', 'C']
    dd = 0
    for syn in synths:
        # This is getting the mean rel gain for each sound (FG rel gain for FGs, etc)
        synth_df = quad.loc[quad.synth_kind == syn].copy()
        bg_df = synth_df[['BG', 'BG_rel_gain']]
        fg_df = synth_df[['FG', 'FG_rel_gain']]

        bg_mean = bg_df.groupby(by='BG').agg(mean=('BG_rel_gain', np.mean)).reset_index(). \
            rename(columns={'BG': 'short_name'})
        fg_mean = fg_df.groupby(by='FG').agg(mean=('FG_rel_gain', np.mean)).reset_index(). \
            rename(columns={'FG': 'short_name'})
        mean_df = pd.concat([bg_mean, fg_mean])

        # This is just loading the sounds and stuffs
        if syn == 'A' or syn == 'N':
            bg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                   f'{BG_folder}/{bb}*.wav'))[0] for bb in bbs]
            fg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                   f'{FG_folder}/{ff}*.wav'))[0] for ff in ffs]
        else:
            bg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                   f'{BG_folder}/{kind_dict[syn]}/{bb}*.wav'))[0] for bb in bbs]
            fg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                   f'{FG_folder}/{kind_dict[syn]}/{ff}*.wav'))[0] for ff in ffs]

        paths = bg_paths + fg_paths
        bgname = [bb.split('/')[-1].split('.')[0] for bb in bg_paths]
        fgname = [ff.split('/')[-1].split('.')[0] for ff in fg_paths]
        names = bgname + fgname

        Bs, Fs = ['BG'] * len(bgname), ['FG'] * len(fgname)
        labels = Bs + Fs

        cnt, sn, pth, ll = sound_num, names[sound_num], paths[sound_num], labels[sound_num]
        sn = sn.split('_')[0]
        sn = sn.replace(' ', '')

        sfs, W = wavfile.read(pth)
        spec = gtgram(W, sfs, 0.02, 0.01, bins, lfreq, hfreq)

        # 2022_09_23 Adding power spectrum stats
        temp = np.abs(np.fft.fft(spec, axis=1))
        freq = np.abs(np.fft.fft(spec, axis=0))

        temp_ps = np.sum(np.abs(np.fft.fft(spec, axis=1)), axis=0)
        freq_ps = np.sum(np.abs(np.fft.fft(spec, axis=0)), axis=1)

        ax[dd].imshow(spec, origin='lower')
        ax[dd].set_title(f"{sn} - {syn}")
        dd += 1
        if dd == 1:
            ax[dd].set_ylabel('Temporal P.S.')
        ax[dd].imshow(temp, origin='lower')
        dd += 1
        if dd == 2:
            ax[dd].set_ylabel('Freq P.S.')
        ax[dd].imshow(freq, origin='lower')
        dd += 1
        ax[dd].spines['top'].set_visible(False), ax[dd].spines['bottom'].set_visible(False)
        ax[dd].spines['left'].set_visible(False), ax[dd].spines['right'].set_visible(False)
        ax[dd].set_yticks([]), ax[dd].set_xticks([])
        qq = np.nanmean(synth_df.loc[synth_df[ll] == sn[2:], f"{ll}_rel_gain"])
        ax[dd].set_title(f"{qq:.2f}")
        dd += 1
        ax[dd].plot(temp_ps[1:])
        ax[dd].set_title(f"{temp_ps[1:].std()}")
        dd += 1
        ax[dd].plot(freq_ps[1:])
        ax[dd].set_title(f"{freq_ps[1:].std()}")
        dd += 1

    fig.tight_layout()

    return pd.DataFrame(names)


def checkout_mods_tidier(sound_num, df, show=['N', 'M', 'U', 'S', 'T', 'C'],
                         thresh=0.03, quads=3, r_cut=None, area=None):
    '''2022_10_24 Updated to make tidier. Takes a number from the list of names (you have to run it once first with
    a random number to get the indexes printed out) and will plot some modulation specs along
    the degraded synthetic sounds. Nice for browsing.'''
    quad, threshold = ohel.quadrants_by_FR(df, threshold=thresh, quad_return=quads)
    if area:
        quad = quad.loc[quad.area == area]
    if r_cut:
        quad = quad.dropna(axis=0, subset='r')
        quad = quad.loc[quad.r >= r_cut]

    lfreq, hfreq, bins = 100, 24000, 48
    cid, btch = df.cellid.iloc[0], df.batch.iloc[0]
    manager = BAPHYExperiment(cellid=cid, batch=btch)
    expt_params = manager.get_baphy_exptparams()
    ref_handle = expt_params[-1]['TrialObject'][1]['ReferenceHandle'][1]
    BG_folder, FG_folder = ref_handle['BG_Folder'], ref_handle['FG_Folder']

    bbs = list(set([bb.split('_')[1][:2] for bb in df.epoch]))
    ffs = list(set([ff.split('_')[2][:2] for ff in df.epoch]))
    bbs.sort(key=int), ffs.sort(key=int)

    kind_dict = {'M': 'SpectrotemporalMod', 'U': 'Spectrotemporal', 'T': 'Temporal',
                 'S': 'Spectral', 'C': 'Cochlear'}
    kind_alias = {'A': 'Non-RMS Norm Natural', 'N': 'RMS Norm Natural', 'M': 'Spectrotemporal Modulation',
                  'U': 'Spectrotemporal', 'S': 'Spectral (-temp)', 'T': 'Temporal (-spec)', 'C': 'Cochlear (-spec, -temp)'}

    fig, axes = plt.subplots(3, len(show), figsize=(len(show)*3, 6))
    ax = np.ravel(axes, 'F')
    dd = 0
    for syn in show:
        # This is getting the mean rel gain for each sound (FG rel gain for FGs, etc)
        synth_df = quad.loc[quad.synth_kind == syn].copy()
        bg_df = synth_df[['BG', 'BG_rel_gain']]
        fg_df = synth_df[['FG', 'FG_rel_gain']]

        bg_mean = bg_df.groupby(by='BG').agg(mean=('BG_rel_gain', np.mean)).reset_index(). \
            rename(columns={'BG': 'short_name'})
        fg_mean = fg_df.groupby(by='FG').agg(mean=('FG_rel_gain', np.mean)).reset_index(). \
            rename(columns={'FG': 'short_name'})
        mean_df = pd.concat([bg_mean, fg_mean])

        # This is just loading the sounds and stuffs
        if syn == 'A' or syn == 'N':
            bg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                   f'{BG_folder}/{bb}*.wav'))[0] for bb in bbs]
            fg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                   f'{FG_folder}/{ff}*.wav'))[0] for ff in ffs]
        else:
            bg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                   f'{BG_folder}/{kind_dict[syn]}/{bb}*.wav'))[0] for bb in bbs]
            fg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                   f'{FG_folder}/{kind_dict[syn]}/{ff}*.wav'))[0] for ff in ffs]

        paths = bg_paths + fg_paths
        bgname = [bb.split('/')[-1].split('.')[0] for bb in bg_paths]
        fgname = [ff.split('/')[-1].split('.')[0] for ff in fg_paths]
        names = bgname + fgname

        Bs, Fs = ['BG'] * len(bgname), ['FG'] * len(fgname)
        labels = Bs + Fs

        cnt, sn, pth, ll = sound_num, names[sound_num], paths[sound_num], labels[sound_num]
        sn = sn.split('_')[0]
        sn = sn.replace(' ', '')

        sfs, W = wavfile.read(pth)
        spec = gtgram(W, sfs, 0.02, 0.01, bins, lfreq, hfreq)

        # 2022_09_23 Adding power spectrum stats
        temp = np.abs(np.fft.fft(spec, axis=1))
        freq = np.abs(np.fft.fft(spec, axis=0))

        temp_ps = np.sum(np.abs(np.fft.fft(spec, axis=1)), axis=0)
        freq_ps = np.sum(np.abs(np.fft.fft(spec, axis=0)), axis=1)

        if dd == 0:
            ax[dd].set_ylabel(sn, fontsize=10, fontweight='bold')
        ax[dd].imshow(spec, aspect='auto', origin='lower', extent=[0, spec.shape[1], 0, spec.shape[0]],
                     cmap='gray_r')
        qq = np.nanmean(synth_df.loc[synth_df[ll] == sn[2:], f"{ll}_rel_gain"])
        ax[dd].set_title(f"{qq:.2f}")
        ax[dd].set_title(f"{kind_alias[syn]}\nRel Gain: {qq:.2f}", fontweight='bold', fontsize=8)
        ax[dd].spines['top'].set_visible(True), ax[dd].spines['right'].set_visible(True)
        ax[dd].set_xticks([]), ax[dd].set_yticks([])

        dd += 1
        if dd == 1:
            ax[dd].set_ylabel('Temporal P.S.', fontsize=8, fontweight='bold')
        ax[dd].plot(temp_ps[1:])
        ax[dd].set_title(f"{temp_ps[1:].std():.2f}")

        dd += 1
        if dd == 2:
            ax[dd].set_ylabel('Freq P.S.', fontsize=8, fontweight='bold')
        ax[dd].plot(freq_ps[1:])
        ax[dd].set_title(f"{freq_ps[1:].std():.2f}")

        dd += 1

    fig.tight_layout()

    return pd.DataFrame(names)


def checkout_mods_cleaner(sound_num, weight_df, synths=['N', 'M', 'S', 'T', 'C'], thresh=0.03, quads=3, r_cut=None, area=None):
    '''2022_11_01. Updated to show full modulation power spectrum and just cleaned up display for shownig Sam things.
    2022_09_28. Takes a number from the list of names (you have to run it once first with
    a random number to get the indexes printed out) and will plot some modulation specs along
    the degraded synthetic sounds. Nice for browsing.'''
    quad, threshold = ohel.quadrants_by_FR(weight_df, threshold=thresh, quad_return=quads)
    if area:
        quad = quad.loc[quad.area == area]
    if r_cut:
        quad = quad.dropna(axis=0, subset='r')
        quad = quad.loc[quad.r >= r_cut]

    lfreq, hfreq, bins = 100, 24000, 48
    cid, btch = weight_df.cellid.iloc[0], weight_df.batch.iloc[0]
    manager = BAPHYExperiment(cellid=cid, batch=btch)
    expt_params = manager.get_baphy_exptparams()
    ref_handle = expt_params[-1]['TrialObject'][1]['ReferenceHandle'][1]
    BG_folder, FG_folder = ref_handle['BG_Folder'], ref_handle['FG_Folder']

    bbs = list(set([bb.split('_')[1][:2] for bb in weight_df.epoch]))
    ffs = list(set([ff.split('_')[2][:2] for ff in weight_df.epoch]))
    bbs.sort(key=int), ffs.sort(key=int)

    # synths = list(weight_df.synth_kind.unique())
    kind_dict = {'M': 'SpectrotemporalMod', 'U': 'Spectrotemporal', 'T': 'Temporal',
                 'S': 'Spectral', 'C': 'Cochlear'}
    kind_alias = {'A': 'Non-RMS Norm Natural', 'N': 'RMS Norm\nNatural', 'M': 'Spectrotemporal\nModulation',
                  'U': 'Spectrotemporal', 'S': 'Spectral\n(-temp)', 'T': 'Temporal\n(-spec)', 'C': 'Cochlear\n(-spec, -temp)'}

    # sound_num = 27

    fig, axes = plt.subplots(3, len(synths)*2, figsize=(20, 5))
    ax = np.ravel(axes, 'F')
    dd = 0
    for syn in synths:
        # This is getting the mean rel gain for each sound (FG rel gain for FGs, etc)
        synth_df = quad.loc[quad.synth_kind == syn].copy()
        bg_df = synth_df[['BG', 'BG_rel_gain']]
        fg_df = synth_df[['FG', 'FG_rel_gain']]

        bg_mean = bg_df.groupby(by='BG').agg(mean=('BG_rel_gain', np.mean)).reset_index(). \
            rename(columns={'BG': 'short_name'})
        fg_mean = fg_df.groupby(by='FG').agg(mean=('FG_rel_gain', np.mean)).reset_index(). \
            rename(columns={'FG': 'short_name'})
        mean_df = pd.concat([bg_mean, fg_mean])

        # This is just loading the sounds and stuffs
        if syn == 'A' or syn == 'N':
            bg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                   f'{BG_folder}/{bb}*.wav'))[0] for bb in bbs]
            fg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                   f'{FG_folder}/{ff}*.wav'))[0] for ff in ffs]
        else:
            bg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                   f'{BG_folder}/{kind_dict[syn]}/{bb}*.wav'))[0] for bb in bbs]
            fg_paths = [glob.glob((f'/auto/users/hamersky/baphy/Config/lbhb/SoundObjects/@OverlappingPairs/'
                                   f'{FG_folder}/{kind_dict[syn]}/{ff}*.wav'))[0] for ff in ffs]

        paths = bg_paths + fg_paths
        bgname = [bb.split('/')[-1].split('.')[0] for bb in bg_paths]
        fgname = [ff.split('/')[-1].split('.')[0] for ff in fg_paths]
        names = bgname + fgname

        Bs, Fs = ['BG'] * len(bgname), ['FG'] * len(fgname)
        labels = Bs + Fs

        cnt, sn, pth, ll = sound_num, names[sound_num], paths[sound_num], labels[sound_num]
        sn = sn.split('_')[0]
        sn = sn.replace(' ', '')

        sfs, W = wavfile.read(pth)
        spec = gtgram(W, sfs, 0.02, 0.01, bins, lfreq, hfreq)

        # 2022_09_23 Adding power spectrum stats
        temp = np.abs(np.fft.fft(spec, axis=1))
        freq = np.abs(np.fft.fft(spec, axis=0))

        temp_ps = np.sum(np.abs(np.fft.fft(spec, axis=1)), axis=0)
        freq_ps = np.sum(np.abs(np.fft.fft(spec, axis=0)), axis=1)
        mod = np.sqrt(np.fft.fftshift(np.abs(np.fft.fft2(spec))))

        ax[dd].imshow(spec, origin='lower', cmap='gray_r')
        qq = np.nanmean(synth_df.loc[synth_df[ll] == sn[2:], f"{ll}_rel_gain"])
        ax[dd].set_title(f"{kind_alias[syn]}\nRG: {qq:.2f}", fontsize=8, fontweight='bold')
        ax[dd].set_yticks([]), ax[dd].set_xticks([])
        if dd == 0:
            ax[dd].set_ylabel(f"{ll} -\n{sn}", fontsize=8, fontweight='bold')
        dd += 1
        if dd == 1:
            ax[dd].set_ylabel('Temporal\nP.S.', fontsize=8, fontweight='bold')
        ax[dd].imshow(np.sqrt(temp), origin='lower', cmap='gray_r')
        ax[dd].set_yticks([]), ax[dd].set_xticks([])
        dd += 1
        if dd == 2:
            ax[dd].set_ylabel('Spectral\nP.S.', fontsize=8, fontweight='bold')
        ax[dd].imshow(np.sqrt(freq), origin='lower', cmap='gray_r')
        ax[dd].set_yticks([]), ax[dd].set_xticks([])
        dd += 1
        # ax[dd].spines['top'].set_visible(False), ax[dd].spines['bottom'].set_visible(False)
        # ax[dd].spines['left'].set_visible(False), ax[dd].spines['right'].set_visible(False)
        width_mod = int(np.floor(mod.shape[1] / 2))
        height_mod = int(np.floor(mod.shape[0] / 2))
        ax[dd].imshow(mod[height_mod:, width_mod:], origin='lower', cmap='gray_r')
        ax[dd].set_yticks([]), ax[dd].set_xticks([])
        ax[dd].set_title("Modspec", fontsize=7, fontweight='bold')
        # qq = np.nanmean(synth_df.loc[synth_df[ll] == sn[2:], f"{ll}_rel_gain"])
        # ax[dd].set_title(f"{qq:.2f}")
        dd += 1
        ax[dd].plot(temp_ps[1:])
        ax[dd].set_title(f"{temp_ps[1:].std():.2f}", fontweight='bold')
        dd += 1
        ax[dd].plot(freq_ps[1:])
        ax[dd].set_title(f"{freq_ps[1:].std():.2f}", fontweight='bold')
        dd += 1

    fig.tight_layout()

    return pd.DataFrame(names)


def rel_gain_synth_scatter(df, show=['N','M','U','S','T','C'], thresh=0.03,
                           quads=3, r_cut=None, area=None):
    '''2022_09_29. Currently an exploratory figure that could easily be adapted to something
    more presentable. Anyway, it plots BG and FG relative gains separately across the
    specified synthetic conditions as a lineplot so you can see the nice trends. The labels
    on the lines are usable for now but a bit cumbersome to read in their current form.'''
    quad, threshold = ohel.quadrants_by_FR(df, threshold=thresh, quad_return=quads)
    if r_cut:
        quad = quad.dropna(axis=0, subset='r')
        quad = quad.loc[quad.r >= r_cut]
    if area:
        quad = quad.loc[quad.area == area]
    # fig, axes = plt.subplots(1, len(show), figsize=(len(show*4), 5))
    fig, ax = plt.subplots(1, 2, figsize=(16,12))

    # Create aliases for the kinds so I can dumbly swap them out all so they can be in the right order
    alias = {'A': '1', 'N': '2', 'M': '3', 'U': '4', 'S': '5', 'T': '6', 'C': '7'}
    kind_alias = {'1': 'Non-RMS Norm\nNatural', '2': 'RMS Norm\nNatural', '3': 'Spectrotemporal\nModulation',
                  '4': 'Spectro-\ntemporal', '5': 'Spectral', '6': 'Temporal', '7': 'Cochlear'}

    if isinstance(show, str):
        show = [show]
    to_plot = quad.loc[quad.synth_kind.isin(show)]

    bg_df = to_plot[['BG','BG_rel_gain', 'synth_kind']]
    bg_plot = bg_df.groupby(['BG', 'synth_kind'])['BG_rel_gain'].mean().reset_index()
    bg_plot['sort'] = bg_plot['synth_kind']
    bg_plot = bg_plot.replace(alias).sort_values('sort').drop('sort', axis=1).copy()
    bg_plot = bg_plot.replace(kind_alias)
    bg_plot = bg_plot.rename(columns={'BG_rel_gain': 'Relative Gain'})
    # palette = sb.color_palette(['deepskyblue'], len(bg_plot.BG.unique()))
    sb.lineplot(data=bg_plot, x='synth_kind', y='Relative Gain', hue='BG', ax=ax[0],
                legend=False) #, palette=palette)

    backend = list(bg_plot.synth_kind.unique())[-1]
    label_df = bg_plot.loc[bg_plot.synth_kind==backend]
    xs, ys, ls = [len(show)-1] *len(label_df), label_df['Relative Gain'].to_list(), label_df['BG'].to_list()
    for ii, nn in enumerate(ls):
        ax[0].annotate(nn, (xs[ii] + 0.2, ys[ii]), fontsize=6)
    ax[0].set_ylabel('Relative Gain', fontweight='bold', fontsize=10)
    ax[0].set_xlabel('')
    ax[0].set_title('Backgrounds', fontweight='bold', fontsize=10)

    fg_df = to_plot[['FG', 'FG_rel_gain', 'synth_kind']]
    fg_plot = fg_df.groupby(['FG', 'synth_kind'])['FG_rel_gain'].mean().reset_index()
    fg_plot['sort'] = fg_plot['synth_kind']
    fg_plot = fg_plot.replace(alias).sort_values('sort').drop('sort', axis=1).copy()
    fg_plot = fg_plot.replace(kind_alias)
    fg_plot = fg_plot.rename(columns={'FG_rel_gain': 'Relative Gain'})
    # palette = sb.color_palette(['yellowgreen'], len(fg_plot.FG.unique()))
    sb.lineplot(data=fg_plot, x='synth_kind', y='Relative Gain', hue='FG', ax=ax[1],
                legend=False) #, palette=palette)

    backend = list(fg_plot.synth_kind.unique())[-1]
    label_df = fg_plot.loc[fg_plot.synth_kind == backend]
    xs, ys, ls = [len(show) - 1] * len(label_df), label_df['Relative Gain'].to_list(), label_df['FG'].to_list()
    for ii, nn in enumerate(ls):
        ax[1].annotate(nn, (xs[ii] + 0.2, ys[ii]), fontsize=6)
    ax[1].set_ylabel('')
    ax[1].set_xlabel('')
    ax[1].set_title('Foregrounds', fontweight='bold', fontsize=10)

    fig.suptitle(f"r >= {r_cut} - Quadrant {quads} - Area: {area}")
    fig.tight_layout()


def rel_gain_synth_scatter_single(df, show=['N', 'M', 'U', 'S', 'T', 'C'], thresh=0.03,
                                  quads=3, r_cut=None, area=None):
    '''2022_09_29. Currently an exploratory figure that could easily be adapted to something
    more presentable. Anyway, it plots BG and FG relative gains separately across the
    specified synthetic conditions as a lineplot so you can see the nice trends. The labels
    on the lines are usable for now but a bit cumbersome to read in their current form.'''
    quad, threshold = ohel.quadrants_by_FR(df, threshold=thresh, quad_return=quads)
    if r_cut:
        quad = quad.dropna(axis=0, subset='r')
        quad = quad.loc[quad.r >= r_cut]
    if area:
        quad = quad.loc[quad.area == area]

    fig, ax = plt.subplots(1, 1, figsize=(5, 10))

    # Create aliases for the kinds so I can dumbly swap them out all so they can be in the right order
    alias = {'A': '1', 'N': '2', 'M': '3', 'U': '4', 'S': '5', 'T': '6', 'C': '7'}
    kind_alias = {'1': 'Non-RMS Norm\nNatural', '2': 'RMS Norm\nNatural', '3': 'Spectrotemporal\nModulation',
                  '4': 'Spectro-\ntemporal', '5': 'Spectral', '6': 'Temporal', '7': 'Cochlear'}

    if isinstance(show, str):
        show = [show]
    to_plot = quad.loc[quad.synth_kind.isin(show)]

    bg_df = to_plot[['BG', 'BG_rel_gain', 'synth_kind']]
    bg_plot = bg_df.groupby(['BG', 'synth_kind'])['BG_rel_gain'].mean().reset_index()
    bg_plot['sort'] = bg_plot['synth_kind']
    bg_plot = bg_plot.replace(alias).sort_values('sort').drop('sort', axis=1).copy()
    bg_plot = bg_plot.replace(kind_alias)
    bg_plot = bg_plot.rename(columns={'BG_rel_gain': 'Relative Gain'})
    palette = sb.color_palette(['deepskyblue'], len(bg_plot.BG.unique()))
    sb.lineplot(data=bg_plot, x='synth_kind', y='Relative Gain', hue='BG', ax=ax,
                legend=False, palette=palette, lw=1)

    backend = list(bg_plot.synth_kind.unique())[-1]
    label_df = bg_plot.loc[bg_plot.synth_kind == backend]
    xs, ys, ls = [len(show) - 1] * len(label_df), label_df['Relative Gain'].to_list(), label_df['BG'].to_list()
    for ii, nn in enumerate(ls):
        ax.annotate(nn, (xs[ii] + 0.2, ys[ii]), fontsize=6)

    bg_mean = bg_df.groupby(['synth_kind'])['BG_rel_gain'].mean().reset_index()
    bg_mean['sort'] = bg_mean['synth_kind']
    bg_mean = bg_mean.replace(alias).sort_values('sort').drop('sort', axis=1).copy()
    bg_mean = bg_mean.replace(kind_alias)
    bg_mean['BG'] = 'All'
    palette = sb.color_palette(['steelblue'], 1)
    sb.lineplot(data=bg_mean, x='synth_kind', y='BG_rel_gain', ax=ax, hue='BG',
                legend=False, palette=palette, lw=2)

    fg_df = to_plot[['FG', 'FG_rel_gain', 'synth_kind']]
    fg_plot = fg_df.groupby(['FG', 'synth_kind'])['FG_rel_gain'].mean().reset_index()
    fg_plot['sort'] = fg_plot['synth_kind']
    fg_plot = fg_plot.replace(alias).sort_values('sort').drop('sort', axis=1).copy()
    fg_plot = fg_plot.replace(kind_alias)
    fg_plot = fg_plot.rename(columns={'FG_rel_gain': 'Relative Gain'})
    palette = sb.color_palette(['yellowgreen'], len(fg_plot.FG.unique()))
    sb.lineplot(data=fg_plot, x='synth_kind', y='Relative Gain', hue='FG', ax=ax,
                legend=False, palette=palette, lw=1)

    backend = list(fg_plot.synth_kind.unique())[-1]
    label_df = fg_plot.loc[fg_plot.synth_kind == backend]
    xs, ys, ls = [len(show) - 1] * len(label_df), label_df['Relative Gain'].to_list(), label_df['FG'].to_list()
    for ii, nn in enumerate(ls):
        ax.annotate(nn, (xs[ii] + 0.2, ys[ii]), fontsize=6)

    fg_mean = fg_df.groupby(['synth_kind'])['FG_rel_gain'].mean().reset_index()
    fg_mean['sort'] = fg_mean['synth_kind']
    fg_mean = fg_mean.replace(alias).sort_values('sort').drop('sort', axis=1).copy()
    fg_mean = fg_mean.replace(kind_alias)
    fg_mean['FG'] = 'All'
    palette = sb.color_palette(['olivedrab'], 1)
    sb.lineplot(data=fg_mean, x='synth_kind', y='FG_rel_gain', ax=ax, hue='FG',
                legend=False, palette=palette, lw=2)

    ax.set_ylabel('Relative Gain', fontweight='bold', fontsize=10)
    ax.set_xlabel('')
    xmin, xmax = ax.get_xlim()
    ax.hlines(0, xmin, xmax, ls='--', color='black', lw=1)

    fig.suptitle(f"r >= {r_cut} - Quadrant {quads} - FR Thresh: {thresh} - Area: {area}")
    fig.tight_layout()


def synth_scatter_metrics(df, first='temp_ps_std', second='freq_ps_std', metric='rel_gain',
                                  show=['N', 'M', 'U', 'S', 'T', 'C'], thresh=0.03,
                                  quads=3, r_cut=None, ref=None, area=None):
    '''2022_10_05. Makes three comparisons of FG and BG from the weight df. The defaults are what
    I needed it for, but I guess you could put other things in if you're feeling wild. Will plot
    three side by side panels of how a given metric changes for each sound over the synthetic
    sound progression. With average line. if you add a ref letter, it'll normalize everything.'''
    quad, threshold = ohel.quadrants_by_FR(df, threshold=thresh, quad_return=quads)
    if r_cut:
        quad = quad.dropna(axis=0, subset='r')
        quad = quad.loc[quad.r >= r_cut]
    if area:
        quad = quad.loc[quad.area==area]
    # so down the road it can't compare the others to something not displaying.
    if ref:
        if ref not in show:
            raise ValueError(f"Your reference {ref} is not in {show}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 10))

    # Create aliases for the kinds so I can dumbly swap them out all so they can be in the right order
    alias = {'A': '1', 'N': '2', 'M': '3', 'U': '4', 'S': '5', 'T': '6', 'C': '7'}
    kind_alias = {'1': 'Non-RMS Norm\nNatural', '2': 'RMS Norm\nNatural', '3': 'Spectrotemporal\nModulation',
                  '4': 'Spectro-\ntemporal', '5': 'Spectral', '6': 'Temporal', '7': 'Cochlear'}

    if isinstance(show, str):
        show = [show]
    to_plot = quad.loc[quad.synth_kind.isin(show)]

    yss = [first, second, metric]

    bg_df = to_plot[['BG', f'BG_{first}', f'BG_{second}', f'BG_{metric}', 'synth_kind']]
    bg_plot = bg_df.groupby(['BG', 'synth_kind', f'BG_{first}', f'BG_{second}'])[f'BG_{metric}'].mean().reset_index()
    bg_plot = bg_plot.rename(columns={'BG':'sound_name'})
    if ref:
        def normfunc(gdf):
            gdf = gdf.drop(columns=['sound_name']).set_index('synth_kind').T.copy()
            for col in gdf.columns:
                if col == ref: continue
                gdf[col] = gdf[col] / gdf[ref]
            gdf[ref] = gdf[ref] / gdf[ref]
            return gdf.T

        n_synths = bg_plot.groupby('sound_name').agg(count=('synth_kind', lambda x: pd.Series.nunique(x)))
        bad = n_synths.query(f'count < {len(show)}').index
        bg_plot = bg_plot.loc[~bg_plot.sound_name.isin(bad),:]
        bg_plot = bg_plot.groupby('sound_name').apply(normfunc).reset_index()

    bg_plot = bg_plot.rename(columns={'sound_name': 'BG'})
    bg_plot['sort'] = bg_plot['synth_kind']
    bg_plot = bg_plot.replace(alias).sort_values('sort').drop('sort', axis=1).copy()
    bg_plot = bg_plot.replace(kind_alias)
    # bg_plot = bg_plot.rename(columns={'BG_rel_gain': 'Relative Gain'})
    paletteBG = sb.color_palette(['deepskyblue'], len(bg_plot.BG.unique()))

    fg_df = to_plot[['FG', f'FG_{first}', f'FG_{second}', f'FG_{metric}', 'synth_kind']]
    fg_plot = fg_df.groupby(['FG', 'synth_kind', f'FG_{first}', f'FG_{second}'])[f'FG_{metric}'].mean().reset_index()
    fg_plot = fg_plot.rename(columns={'FG': 'sound_name'})
    if ref:
        n_synths = fg_plot.groupby('sound_name').agg(count=('synth_kind', lambda x: pd.Series.nunique(x)))
        bad = n_synths.query(f'count < {len(show)}').index
        fg_plot = fg_plot.loc[~fg_plot.sound_name.isin(bad),:]
        fg_plot = fg_plot.groupby('sound_name').apply(normfunc).reset_index()
    fg_plot = fg_plot.rename(columns={'sound_name': 'FG'})
    fg_plot['sort'] = fg_plot['synth_kind']
    fg_plot = fg_plot.replace(alias).sort_values('sort').drop('sort', axis=1).copy()
    fg_plot = fg_plot.replace(kind_alias)
    # bg_plot = bg_plot.rename(columns={'BG_rel_gain': 'Relative Gain'})
    paletteFG = sb.color_palette(['yellowgreen'], len(fg_plot.FG.unique()))

    for yy, ax in zip(yss, axes):
        sb.lineplot(data=bg_plot, x='synth_kind', y=f'BG_{yy}', hue='BG', ax=ax,
                    legend=False, palette=paletteBG, lw=1)

        backend = list(bg_plot.synth_kind.unique())[-1]
        label_df = bg_plot.loc[bg_plot.synth_kind == backend]
        xs, ys, ls = [len(show) - 1] * len(label_df), label_df[f'BG_{yy}'].to_list(), label_df['BG'].to_list()
        for ii, nn in enumerate(ls):
            ax.annotate(nn, (xs[ii] + 0.2, ys[ii]), fontsize=6)

        sb.lineplot(data=fg_plot, x='synth_kind', y=f'FG_{yy}', hue='FG', ax=ax,
                    legend=False, palette=paletteFG, lw=1)

        backend = list(fg_plot.synth_kind.unique())[-1]
        label_df = fg_plot.loc[fg_plot.synth_kind == backend]
        xs, ys, ls = [len(show) - 1] * len(label_df), label_df[f'FG_{yy}'].to_list(), label_df['FG'].to_list()
        for ii, nn in enumerate(ls):
            ax.annotate(nn, (xs[ii] + 0.2, ys[ii]), fontsize=6)

        if ref:
            bg_mean = bg_plot.groupby(['synth_kind'])[f'BG_{yy}'].mean().reset_index()
        else:
            bg_mean = bg_df.groupby(['synth_kind'])[f'BG_{yy}'].mean().reset_index()
        bg_mean['sort'] = bg_mean['synth_kind']
        bg_mean = bg_mean.replace(alias).sort_values('sort').drop('sort', axis=1).copy()
        bg_mean = bg_mean.replace(kind_alias)
        bg_mean['BG'] = 'All'
        paletteBG2 = sb.color_palette(['steelblue'], 1)
        sb.lineplot(data=bg_mean, x='synth_kind', y=f'BG_{yy}', ax=ax, hue='BG',
                    legend=False, palette=paletteBG2, lw=2)

        if ref:
            fg_mean = fg_plot.groupby(['synth_kind'])[f'FG_{yy}'].mean().reset_index()
        else:
            fg_mean = fg_df.groupby(['synth_kind'])[f'FG_{yy}'].mean().reset_index()
        fg_mean['sort'] = fg_mean['synth_kind']
        fg_mean = fg_mean.replace(alias).sort_values('sort').drop('sort', axis=1).copy()
        fg_mean = fg_mean.replace(kind_alias)
        fg_mean['FG'] = 'All'
        paletteFG2 = sb.color_palette(['olivedrab'], 1)
        sb.lineplot(data=fg_mean, x='synth_kind', y=f'FG_{yy}', ax=ax, hue='FG',
                    legend=False, palette=paletteFG2, lw=2)

        ax.set_ylabel(yy, fontweight='bold', fontsize=10)
        ax.set_xlabel('')

    fig.suptitle(f"r >= {r_cut} - Quadrant {quads} - FR Thresh: {thresh} - Area: {area}")
    fig.tight_layout()


def synthetic_summary_relative_gain_bar(weight_df0, condition=''):
    '''2022_11_23. This generates the summary of the synthetic histograms for FG relative gain and plots them as a
    horizontal bar graph and returns the statistics that are Bonferroni corrected. It is currently not flexible to
    include different synthetic conditions but that's fine for now.'''
    M = weight_df0.loc[weight_df0.synth_kind == 'M']
    S = weight_df0.loc[weight_df0.synth_kind == 'S']
    T = weight_df0.loc[weight_df0.synth_kind == 'T']
    C = weight_df0.loc[weight_df0.synth_kind == 'C']

    MS = stats.ttest_ind(M[f'FG_rel_gain{condition}'], S[f'FG_rel_gain{condition}']).pvalue
    MT = stats.ttest_ind(M[f'FG_rel_gain{condition}'], T[f'FG_rel_gain{condition}']).pvalue
    SC = stats.ttest_ind(S[f'FG_rel_gain{condition}'], C[f'FG_rel_gain{condition}']).pvalue
    TC = stats.ttest_ind(T[f'FG_rel_gain{condition}'], C[f'FG_rel_gain{condition}']).pvalue
    stat_dict = {'MS': MS * 4, 'MT': MT * 4, 'SC': SC * 4, 'TC': TC * 4}

    fig, ax = plt.subplots(1, 1, figsize=(8, 9))

    ax.barh(y=['Cochlear', 'Temporal', 'Spectral', 'Spectrotemporal\nModulation'],
            width=[np.nanmean(C['FG_rel_gain']), np.nanmean(T['FG_rel_gain']),
                   np.nanmean(S['FG_rel_gain']), np.nanmean(M['FG_rel_gain'])],
            label=f'Total (n=)', color='dimgrey', linestyle='None', height=0.9)  # , marker=symbols[cnt])

    ax.errorbar(y=['Cochlear', 'Temporal', 'Spectral', 'Spectrotemporal\nModulation'],
                x=[np.nanmean(C[f'FG_rel_gain{condition}']), np.nanmean(T[f'FG_rel_gain{condition}']),
                   np.nanmean(S[f'FG_rel_gain{condition}']), np.nanmean(M[f'FG_rel_gain{condition}'])],
                xerr=[stats.sem(C[f'FG_rel_gain{condition}']), stats.sem(T[f'FG_rel_gain{condition}']),
                      stats.sem(S[f'FG_rel_gain{condition}']), stats.sem(M[f'FG_rel_gain{condition}'])],
                yerr=None, color='black', linestyle='None')

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines['left'].set_visible(False), ax.spines['right'].set_visible(True)

    ax.tick_params(axis='x', labelsize=14)
    ax.set_xlabel("Relative FG Suppresion", fontsize=18, fontweight='bold')
    ax.set_yticklabels(['Cochlear', 'Temporal', 'Spectral', 'Spectrotemporal\nModulation'], fontsize=14,
                       fontweight='bold')
    fig.tight_layout()

    return stat_dict


def synthetic_summary_relative_gain_multi_bar(weight_df0, suffixes=['', '_start', '_end']):
    '''2022_11_30. Same thing as the original but it takes suffixes and adjusts the size based on that.
    2022_11_23. This generates the summary of the synthetic histograms for FG relative gain and plots them as a
    horizontal bar graph and returns the statistics that are Bonferroni corrected. It is currently not flexible to
    include different synthetic conditions but that's fine for now.'''
    N = weight_df0.loc[weight_df0.synth_kind == 'N']
    M = weight_df0.loc[weight_df0.synth_kind == 'M']
    S = weight_df0.loc[weight_df0.synth_kind == 'S']
    T = weight_df0.loc[weight_df0.synth_kind == 'T']
    C = weight_df0.loc[weight_df0.synth_kind == 'C']

    stat_dict = {}
    for ss in suffixes:
        MS = stats.ttest_ind(M[f'FG_rel_gain{ss}'], S[f'FG_rel_gain{ss}']).pvalue
        MT = stats.ttest_ind(M[f'FG_rel_gain{ss}'], T[f'FG_rel_gain{ss}']).pvalue
        SC = stats.ttest_ind(S[f'FG_rel_gain{ss}'], C[f'FG_rel_gain{ss}']).pvalue
        TC = stats.ttest_ind(T[f'FG_rel_gain{ss}'], C[f'FG_rel_gain{ss}']).pvalue
        stat_dict[f'MS{ss}'], stat_dict[f'MT{ss}'], stat_dict[f'SC{ss}'], stat_dict[f'TC{ss}'] = MS*4, MT*4, SC*4, TC*4

    fig, axes = plt.subplots(1, len(suffixes), figsize=(len(suffixes*4), 5), sharex=True)

    for ax, ss in zip(axes, suffixes):
        ax.barh(y=['Cochlear', 'Temporal', 'Spectral', 'Spectrotemporal\nModulation', 'Natural'],
                width=[np.nanmean(C[f'FG_rel_gain{ss}']), np.nanmean(T[f'FG_rel_gain{ss}']),
                       np.nanmean(S[f'FG_rel_gain{ss}']), np.nanmean(M[f'FG_rel_gain{ss}']), np.nanmean(N[f'FG_rel_gain{ss}'])],
                label=f'Total (n=)', color='dimgrey', linestyle='None', height=0.9)  # , marker=symbols[cnt])

        ax.errorbar(y=['Cochlear', 'Temporal', 'Spectral', 'Spectrotemporal\nModulation', 'Natural'],
                    x=[np.nanmean(C[f'FG_rel_gain{ss}']), np.nanmean(T[f'FG_rel_gain{ss}']),
                       np.nanmean(S[f'FG_rel_gain{ss}']), np.nanmean(M[f'FG_rel_gain{ss}']), np.nanmean(N[f'FG_rel_gain{ss}'])],
                    xerr=[stats.sem(C[f'FG_rel_gain{ss}']), stats.sem(T[f'FG_rel_gain{ss}']),
                          stats.sem(S[f'FG_rel_gain{ss}']), stats.sem(M[f'FG_rel_gain{ss}']), stats.sem(N[f'FG_rel_gain{ss}'])],
                    yerr=None, color='black', linestyle='None')

        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.spines['left'].set_visible(False), ax.spines['right'].set_visible(True)

        ax.tick_params(axis='x', labelsize=10)
        ax.set_xlabel("Relative FG Suppresion", fontsize=10, fontweight='bold')
        ax.set_yticklabels(['Cochlear', 'Temporal', 'Spectral', 'Spectrotemporal\nModulation', 'Natural'], fontsize=10,
                           fontweight='bold')
        ax.set_title(f"{ss} Ref: {weight_df0.filt_by.unique()[0]} - n={len(C)}", fontsize=10, fontweight='bold')
    fig.tight_layout()

    return stat_dict


def synthetic_summary_weight_multi_bar(df, suffixes=['', '_start', '_end'], show=['N','M', 'S', 'T', 'C'], figsize=None):
    '''2022_12_05. Same thing as the other except it is veritcal and plots the weights separately instead of relative gain.
    Same thing as the original but it takes suffixes and adjusts the size based on that.
    2022_11_23. This generates the summary of the synthetic histograms for FG relative gain and plots them as a
    horizontal bar graph and returns the statistics that are Bonferroni corrected. It is currently not flexible to
    include different synthetic conditions but that's fine for now.'''
    synth_dict = {ss: df.loc[df.synth_kind == ss] for ss in show}
    ns = len(synth_dict[show[0]])

    kind_dict = {'A': 'Non-RMS Norm\nNatural', 'N': 'Natural', 'M': 'Spectrotemporal\nModulation',
                  'U': 'Spectro-\ntemporal', 'S': 'Spectral', 'T': 'Temporal', 'C': 'Cochlear'}
    suff_dict = {'': 'Full Fit', '_start': '0-0.5s Fit', '_end': '0.5-1s Fit'}

    if figsize:
        fig, axes = plt.subplots(1, len(suffixes), figsize=figsize, sharey=True)
    else:
        fig, axes = plt.subplots(1, len(suffixes), figsize=(len(suffixes*4), 5), sharey=True)

    if len(suffixes) == 1:
        axes = [axes]

    to_plot_x = np.arange(len(show))
    x_labels = [f"{kind_dict[dd]}" for dd in show]
    width = 0.2

    for ax, ss in zip(axes, suffixes):

        to_plot_y1 = [np.nanmean(synth_dict[dd][f'weightsA{ss}']) for dd in show]
        sem1 = [stats.sem(synth_dict[dd][f'weightsA{ss}']) for dd in show]
        to_plot_y2 = [np.nanmean(synth_dict[dd][f'weightsB{ss}']) for dd in show]
        sem2 = [stats.sem(synth_dict[dd][f'weightsB{ss}']) for dd in show]

        ax.bar(x=to_plot_x-width, height=to_plot_y1, width=width*2, color='deepskyblue', label='BG', linestyle=None)
        ax.errorbar(x=to_plot_x-width, y=to_plot_y1, yerr=sem1, xerr=None, color='black', linestyle='None')
        ax.bar(x=to_plot_x+width, height=to_plot_y2, width=width*2, color='yellowgreen', label='FG', linestyle=None)
        ax.errorbar(x=to_plot_x+width, y=to_plot_y2, yerr=sem2, xerr=None, color='black', linestyle='None')

        ax.set_ylabel("Weights", fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold', rotation=60)
        ax.set_title(f"{suff_dict[ss]}", fontsize=10, fontweight='bold')
        ax.legend(loc='upper right')
    fig.suptitle(f"Area: {df.area.unique()[0]} - By: {df.filt_by.unique()[0]} - (n={ns})", fontsize=10, fontweight='bold')
    fig.tight_layout()

    stat_dict = {}
    for ss in suffixes:
        N, M, S, T, C = df.loc[df.synth_kind == 'N'], df.loc[df.synth_kind == 'M'], df.loc[df.synth_kind == 'S'], \
                        df.loc[df.synth_kind == 'T'], df.loc[df.synth_kind == 'C']
        NM_A = stats.ttest_ind(N[f'weightsA{ss}'], M[f'weightsA{ss}']).pvalue
        MS_A = stats.ttest_ind(M[f'weightsA{ss}'], S[f'weightsA{ss}']).pvalue
        MT_A = stats.ttest_ind(M[f'weightsA{ss}'], T[f'weightsA{ss}']).pvalue
        SC_A = stats.ttest_ind(S[f'weightsA{ss}'], C[f'weightsA{ss}']).pvalue
        TC_A = stats.ttest_ind(T[f'weightsA{ss}'], C[f'weightsA{ss}']).pvalue

        NM_B = stats.ttest_ind(N[f'weightsB{ss}'], M[f'weightsB{ss}']).pvalue
        MS_B = stats.ttest_ind(M[f'weightsB{ss}'], S[f'weightsB{ss}']).pvalue
        MT_B = stats.ttest_ind(M[f'weightsB{ss}'], T[f'weightsB{ss}']).pvalue
        SC_B = stats.ttest_ind(S[f'weightsB{ss}'], C[f'weightsB{ss}']).pvalue
        TC_B = stats.ttest_ind(T[f'weightsB{ss}'], C[f'weightsB{ss}']).pvalue

        N = stats.ttest_ind(N[f'weightsA{ss}'], N[f'weightsB{ss}']).pvalue
        M = stats.ttest_ind(M[f'weightsA{ss}'], M[f'weightsB{ss}']).pvalue
        S = stats.ttest_ind(S[f'weightsA{ss}'], S[f'weightsB{ss}']).pvalue
        T = stats.ttest_ind(T[f'weightsA{ss}'], T[f'weightsB{ss}']).pvalue
        C = stats.ttest_ind(C[f'weightsA{ss}'], C[f'weightsB{ss}']).pvalue

        stat_dict[f'NM_A{ss}'], stat_dict[f'MS_A{ss}'], stat_dict[f'MT_A{ss}'], stat_dict[f'SC_A{ss}'], \
            stat_dict[f'TC_A{ss}'] = NM_A * 4, MS_A * 4, MT_A * 4, SC_A * 4, TC_A * 4
        stat_dict[f'NM_B{ss}'], stat_dict[f'MS_B{ss}'], stat_dict[f'MT_B{ss}'], stat_dict[f'SC_B{ss}'], \
            stat_dict[f'TC_B{ss}'] = NM_B * 4, MS_B * 4, MT_B * 4, SC_B * 4, TC_B * 4
        stat_dict[f'N{ss}'], stat_dict[f'M{ss}'], stat_dict[f'S{ss}'], stat_dict[f'T{ss}'], \
            stat_dict[f'C{ss}'] = N * 4, M * 4, S * 4, T * 4, C * 4

    return stat_dict


def plot_cc_cuts(df, sn, type, power_thresh=0.2, percent_lims=[10, 90], sk='N'):
    '''2023_05_19. Mostly for testing the spectral statistic method, but it is a cute figure. Takes a dataframe,
    usually filter it by the binaural, dynamic, SNR conditions first, as well as keeping the olp_type=synthetic
    first. Also, area/layer. Anyhow, it takes a dataframe and a name of a sound, use sound_df.FG.unique()
    or sound_df.BG.unique() with your dataframe to get the keys that would work, and label it FG or BG for type.
    The power_thresh isn't super  useful anymore but the percent_lims should do the trick.'''
    lfreq, hfreq, bins = 100, 24000, 48
    # sk = bgs.synth_kind.unique()[0]
    imopts_cc = {'origin': 'lower', 'interpolation': 'none', 'cmap': 'inferno', 'aspect': 'auto'}
    imopts_spec = {'origin': 'lower', 'interpolation': 'none', 'cmap': 'gray_r', 'aspect': 'auto'}

    filt_df = df.loc[df.synth_kind==sk]
    sound = filt_df[[f'{type}', f'{type}_path', f'{type}_rel_gain_all', 'synth_kind']]
    sound = sound.drop_duplicates(subset=[f'{type}_path'])
    sound = sound.sort_values(by=f'{type}_rel_gain_all')

    row = sound.loc[sound[f'{type}'] == sn]

    name, path, gain = row[f'{type}'].values[0], row[f'{type}_path'].values[0], \
                       np.around(row[f'{type}_rel_gain_all'], 3).values[0]
    sfs, W = wavfile.read(path)
    spec = gtgram(W, sfs, 0.02, 0.01, bins, lfreq, hfreq)

    freq_mean = np.nanmean(spec, axis=1)
    biggun = np.max(freq_mean)
    x_freq = np.logspace(np.log2(lfreq), np.log2(hfreq), num=bins, base=2)
    csm = np.cumsum(freq_mean)
    big = np.max(csm)
    norm_mean = freq_mean/biggun

    lower, upper = percent_lims[0] / 100, percent_lims[1] / 100
    bin_high = np.abs(csm - (big * upper)).argmin()
    bin_low = np.abs(csm - (big * lower)).argmin()
    bandwidth = np.log2(x_freq[bin_high] / x_freq[bin_low])

    power_mask = norm_mean >= power_thresh
    change_mask = power_mask[:-1] != power_mask[1:]
    change_idx = np.where(power_mask[:-1] != power_mask[1:])[0]

    if np.all(power_mask == True):
        binup, bindown = 0, bins
    else:
        if power_mask[0] == True:
            binup = 0
        else:
            binup = change_idx[0]

        if power_mask[-1] == True:
            bindown = bins
        else:
            bindown = change_idx[-1]
    bandw = np.log2(x_freq[bindown] / x_freq[binup])

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    ax = np.ravel(axes)
    ax[0].imshow(np.sqrt(spec), **imopts_spec)
    ax[1].plot(norm_mean)
    ax[1].set_ylabel('Norm Freq Mean')
    ax[1].set_title(f'Spec Mean: thresh={power_thresh}')
    ax[2].plot(csm, color='yellowgreen')
    ax[2].set_title(f'CSM - {percent_lims[0]}%-{percent_lims[1]}%')
    ax[2].vlines([bin_low,bin_high], 0, big, color='black', ls=':')
    ax[1].vlines([binup,bindown], 0, 1, color='black', ls=':')
    ax[0].hlines([binup,bindown], 0, spec.shape[1], color='blue', ls=':')
    ax[0].hlines([bin_high,bin_low], 0, spec.shape[1], color='green', ls=':')
    ax[3].imshow(np.sqrt(spec[binup:bindown, :]), **imopts_spec)
    ax[4].imshow(np.sqrt(spec[bin_low:bin_high, :]), **imopts_spec)
    ax[0].set_title(f'{type}: {name} - gain: {gain}', fontsize=8, fontweight='bold')
    ax[1].hlines([power_thresh], 0, spec.shape[0], color='red', ls='--')
    ax[0].set_yticks([0, binup, bindown, bin_low, bin_high, len(x_freq)])
    ax[0].set_yticklabels([int(x_freq[0]), int(x_freq[binup]), int(x_freq[bindown]), int(x_freq[bin_low]),
                           int(x_freq[bin_high]), int(x_freq[-1])])
    ax[3].set_title(f"Bandwidth: {bandw:.2f}")
    ax[4].set_title(f"Bandwidth: {bandwidth:.2f}")

    cc = np.corrcoef(spec)
    # cpow = cc[np.triu_indices(bins, k=1)].mean()
    cpow = cc[np.triu_indices(bins, k=1)].mean()
    ax[5].imshow(cc, **imopts_cc)
    ax[5].set_title(f'CC: {cpow:.3f}', fontsize=7, fontweight='bold')
    ax[5].set_yticks([]), ax[5].set_xticks([])

    # Done by power threshold
    cut_spec = spec[binup:bindown, :]
    ccc = np.corrcoef(cut_spec)
    ccpow = ccc[np.triu_indices(cut_spec.shape[0], k=1)].mean()

    # Done by CSM
    cut_spec_csm = spec[bin_low:bin_high, :]
    cccsm = np.corrcoef(cut_spec_csm)
    cccsmpow = cccsm[np.triu_indices(cut_spec_csm.shape[0], k=1)].mean()

    ax[3].set_yticks([0, cut_spec.shape[0]])
    ax[3].set_yticklabels([int(x_freq[binup]), int(x_freq[bindown])])
    ax[8].imshow(ccc, **imopts_cc)
    ax[8].set_title(f'CC: {ccpow:.3f}', fontsize=7, fontweight='bold')
    ax[8].set_yticks([]), ax[8].set_xticks([])

    ax[4].set_yticks([0, cut_spec_csm.shape[0]])
    ax[4].set_yticklabels([int(x_freq[bin_low]), int(x_freq[bin_high])])
    ax[9].imshow(cccsm, **imopts_cc)
    ax[9].set_title(f'CC: {cccsmpow:.3f}', fontsize=7, fontweight='bold')
    ax[9].set_yticks([]), ax[9].set_xticks([])

    ax[6].spines['left'].set_visible(False), ax[6].spines['bottom'].set_visible(False)
    ax[6].set_yticks([]), ax[6].set_xticks([])
    ax[7].spines['left'].set_visible(False), ax[7].spines['bottom'].set_visible(False)
    ax[7].set_yticks([]), ax[7].set_xticks([])


def plot_spec_cc(df, type, percent_lims=[10,90], sk='N'):
    '''2023_05_19. Makes a big display of all of the sounds in a given BG/FG category, organized by ascending
    relative gain, and plots the spectrogram cut by the given bandwidth parameter (percent_lims) and then plots
    the spectral correlation matric next to it. Includes gain, cc, bandwidth, and cc if you didn't filter the
    spectrogram by bandwidth as a check.'''
    lfreq, hfreq, bins = 100, 24000, 48
    imopts_cc = {'origin': 'lower', 'interpolation': 'none', 'cmap': 'inferno', 'aspect': 'auto'}
    imopts_spec = {'origin': 'lower', 'interpolation': 'none', 'cmap': 'gray_r', 'aspect': 'auto'}

    filt_df = df.loc[df.synth_kind==sk]
    sound = filt_df[[f'{type}', f'{type}_path', f'{type}_rel_gain_all', 'synth_kind']]
    sound = sound.drop_duplicates(subset=[f'{type}_path'])
    sound = sound.sort_values(by=f'{type}_rel_gain_all')

    width = int(np.ceil(len(sound) / 7)) * 2
    leftovers = width*7 - len(sound)*2

    fig, axes = plt.subplots(7, width, figsize=(10, 12))
    ax = np.ravel(axes)
    cnt = 0
    for rr in range(len(sound)):
        name, path, gain = sound.iloc[rr][f'{type}'], sound.iloc[rr][f'{type}_path'], \
                           np.around(sound.iloc[rr][f'{type}_rel_gain_all'], 3)
        sfs, W = wavfile.read(path)
        spec = gtgram(W, sfs, 0.02, 0.01, bins, lfreq, hfreq)

        freq_mean = np.nanmean(spec, axis=1)
        x_freq = np.logspace(np.log2(lfreq), np.log2(hfreq), num=bins, base=2)
        csm = np.cumsum(freq_mean)
        big = np.max(csm)

        lower, upper = percent_lims[0] / 100, percent_lims[1] / 100
        bin_high = np.abs(csm - (big * upper)).argmin()
        bin_low = np.abs(csm - (big * lower)).argmin()
        bandwidth = np.log2(x_freq[bin_high] / x_freq[bin_low])

        cut_spec = spec[bin_low:bin_high, :]
        cc = np.corrcoef(cut_spec)
        cpow = cc[np.triu_indices(cut_spec.shape[0], k=1)].mean()

        cc_full = np.corrcoef(spec)
        cpow_full = cc_full[np.triu_indices(spec.shape[0], k=1)].mean()

        ax[rr+cnt].imshow(np.sqrt(cut_spec), **imopts_spec)
        ax[rr+cnt].set_yticks([0, cut_spec.shape[0]])
        ax[rr+cnt].set_yticklabels([int(x_freq[bin_low]), int(x_freq[bin_high])])
        ax[rr+cnt+1].imshow(cc, **imopts_cc)
        ax[rr+cnt+1].set_yticks([0, cut_spec.shape[0]-1]), ax[rr+cnt+1].set_xticks([])
        ax[rr+cnt+1].set_yticklabels([bin_low, bin_high-1])
        ax[rr+cnt].set_title(f'{name} | gain: {gain}\nbw: {bandwidth:.1f} | fcc: {cpow_full:.3f}', fontsize=8) #, fontweight='bold')
        ax[rr+cnt+1].set_title(f'CC: {cpow:.3f}', fontsize=8, fontweight='bold')
        ax[rr+cnt].spines['top'].set_visible(True), ax[rr+cnt].spines['right'].set_visible(True)
        ax[rr+cnt+1].spines['top'].set_visible(True), ax[rr+cnt+1].spines['right'].set_visible(True)

        cnt+=1

    ax[0].set_ylabel('Freq (Hz)', fontsize=6)

    for aa in range(leftovers):
        ax[-aa-1].spines['left'].set_visible(False), ax[-aa-1].spines['bottom'].set_visible(False)
        ax[-aa-1].set_yticks([]), ax[-aa-1].set_xticks([])

    fig.suptitle(f'{type} - {sk}\n ', fontweight='bold', fontsize=10)
    fig.tight_layout()


def sound_metric_scatter_bgfg_sep(df, x_metrics, x_labels=None, fr_thresh=0.03, r_cut=None):
    '''2023_05_24. Takes a sound df, must already be filtered by singular area, and the layers you want, and
    it will plot separately the mean relative gain for each sound across the different synthetic conditions. BG and FG
    will be on different rows for ease of viewing, but the axes will be shared so you can see the different spaces
    they occupy.'''

    df = df.loc[(df.bg_FR >= fr_thresh) & (df.fg_FR >= fr_thresh)]
    df = df.loc[(df.kind=='11') & (df.dyn_kind=='ff') & (df.SNR==0) & (df.olp_type=='synthetic')]

    if r_cut:
        df = df.dropna(axis=0, subset='r')
        df = df.loc[df.r >= r_cut]
    df = df.copy()

    # Get rid of when the weights are unrealistic
    weight_lim = [0, 2]
    df = df.loc[((df[f'weightsA'] >= weight_lim[0]) & (df[f'weightsA'] <= weight_lim[1])) &
                        ((df[f'weightsB'] >= weight_lim[0]) & (df[f'weightsB'] <= weight_lim[1]))]

    ylim_max = np.max([df.FG_rel_gain_avg.max(), df.FG_rel_gain_avg.max()])
    ylim_min = np.min([df.BG_rel_gain_avg.min(), df.FG_rel_gain_avg.min()])

    fig, axes = plt.subplots(2, len(x_metrics), figsize=(12, 5), sharey=True)
    ax = axes.ravel()

    count = 0
    for ll in ['BG', 'FG']:
        for cnt, met in enumerate(x_metrics):

            xlim_max = np.max([df[f'BG_{met}'].max(), df[f'FG_{met}'].max()])
            xlim_min = np.min([df[f'BG_{met}'].min(), df[f'FG_{met}'].min()])
            xlim_min = xlim_min - np.abs(xlim_max)*0.05

            to_plot = df[[f'{ll}', f'{ll}_rel_gain_avg', 'synth_kind', f'{ll}_{met}']]
            to_plot = to_plot.drop_duplicates(subset=[f'{ll}', 'synth_kind'])
            met_av = to_plot.groupby('synth_kind', as_index=False)[f'{ll}_{met}'].mean()
            gain_av = to_plot.groupby('synth_kind', as_index=False)[f'{ll}_rel_gain_avg'].mean()

            if ll == 'BG':
                colors = {'N':'cornflowerblue', 'M':'royalblue', 'T':'blue', 'S':'darkblue', 'C':'black'}
            elif ll == 'FG':
                colors = {'N': 'yellowgreen', 'M': 'forestgreen', 'T': 'green', 'S': 'darkgreen', 'C': 'black'}

            for key, val in colors.items():
                sb.scatterplot(x=f'{ll}_{met}', y=f'{ll}_rel_gain_avg', data=to_plot.loc[to_plot.synth_kind==key],
                               ax=ax[cnt+count], s=24, color=val, label=key)

                sb.scatterplot(x=f'{ll}_{met}', y=ylim_max*1.1, data=met_av.loc[met_av.synth_kind==key],
                               ax=ax[cnt+count], marker='d', color=val, s=100)
                sb.scatterplot(x=xlim_max*1.1, y=f'{ll}_rel_gain_avg', data=gain_av.loc[gain_av.synth_kind==key],
                               ax=ax[cnt+count], marker='<', color=val, s=200)

            ax[cnt+count].set_ylim(ylim_min, ylim_max*1.1), ax[cnt+count].set_xlim(xlim_min,xlim_max*1.1)
            ax[cnt+count].set_ylabel(''), ax[cnt+count].set_xlabel('')
            ax[count].set_ylabel(f'Relative Gain', fontweight='bold', fontsize=10)
            if count != 0:
                if x_labels:
                    ax[cnt + count].set_xlabel(f'{x_labels[cnt]}', fontweight='bold', fontsize=10)
                else:
                    ax[cnt + count].set_ylabel(f'{met}', fontweight='bold', fontsize=10)

        count += len(x_metrics)

    fig.suptitle(f'{df.area.unique()[0]} - FR threshold: {fr_thresh} - r > {r_cut}', fontsize=12, fontweight='bold')


def sound_metric_scatter_combined(df, x_metrics, x_labels=None, synth_show=['N', 'M', 'S', 'T', 'C'],
                                  fr_thresh=0.03, r_cut=None, suffix=''):
    '''2023_05_30. Added the line where it gets the sound_df after doing the filters you've applied. This makes
    more sense than how I did it before, because it calculated averages before filtering based on your criteria,
    which was dumb.

    2023_05_30. Finished. Takes a dataframe and the x_metrics you give it (must be already existing in sound_df,
    and will plot on the top row the metrics scattered and separated by BG/FG, showing their degradation over
    synthetic conditions. The bottom will take the same, but remove the BG/FG distinction and plot regression
    lines of the metric versus the relative gain, even though regression line may not be the best metric.

    2023_05_24. Takes a sound df, must already be filtered by singular area, and the layers you want, and
    it will plot separately the mean relative gain for each sound across the different synthetic conditions. BG and FG
    will be on different rows for ease of viewing, but the axes will be shared so you can see the different spaces
    they occupy.'''

    if suffix == '_start' or suffix == '_end':
        df = df.loc[(df.bg_FR_start >= fr_thresh) & (df.fg_FR_start >= fr_thresh)
                    & (df.bg_FR_end >= fr_thresh) & (df.fg_FR_end >= fr_thresh)]
    else:
        df = df.loc[(df.bg_FR >= fr_thresh) & (df.fg_FR >= fr_thresh)]

    df = df.loc[(df.kind == '11') & (df.dyn_kind == 'ff') & (df.SNR == 0) & (df.olp_type == 'synthetic')]

    if r_cut:
        if suffix == '_start' or suffix == '_end':
            df = df.loc[(df['r_start'] >= r_cut) & (df['r_end'] >= r_cut)]
        else:
            df = df.dropna(axis=0, subset='r')
            df = df.loc[df.r >= r_cut]
    df = df.copy()

    # Get rid of when the weights are unrealistic
    weight_lim = [0, 2]
    if suffix == '_start' or suffix == '_end':
        df = df.loc[((df[f'weightsA_start'] >= weight_lim[0]) & (df[f'weightsA_start'] <= weight_lim[1])) &
                    ((df[f'weightsB_start'] >= weight_lim[0]) & (df[f'weightsB_start'] <= weight_lim[1])) &
                    ((df[f'weightsA_end'] >= weight_lim[0]) & (df[f'weightsA_end'] <= weight_lim[1])) &
                    ((df[f'weightsB_end'] >= weight_lim[0]) & (df[f'weightsB_end'] <= weight_lim[1]))]
    else:
        df = df.loc[((df[f'weightsA'] >= weight_lim[0]) & (df[f'weightsA'] <= weight_lim[1])) &
                    ((df[f'weightsB'] >= weight_lim[0]) & (df[f'weightsB'] <= weight_lim[1]))]

    df = ohel.get_sound_statistics_from_df(df, percent_lims=[15, 85], append=True)

    ylim_max = np.max([df[f'BG_rel_gain_avg{suffix}'].max(), df[f'FG_rel_gain_avg{suffix}'].max()])
    ylim_min = np.min([df[f'BG_rel_gain_avg{suffix}'].min(), df[f'FG_rel_gain_avg{suffix}'].min()])

    fig, axes = plt.subplots(2, len(x_metrics), figsize=(12, 5 * len(['BG', 'FG'])), sharey=True)
    ax = axes.ravel()

    count = 0
    for ll in ['BG', 'FG']:
        for cnt, met in enumerate(x_metrics):

            xlim_max = np.max([df[f'BG_{met}{suffix}'].max(), df[f'FG_{met}{suffix}'].max()])
            xlim_min = np.min([df[f'BG_{met}{suffix}'].min(), df[f'FG_{met}{suffix}'].min()])
            xlim_min = xlim_min - np.abs(xlim_max) * 0.05

            to_plot = df[[f'{ll}', f'{ll}_rel_gain_avg{suffix}', 'synth_kind', f'{ll}_{met}{suffix}']]
            to_plot = to_plot.drop_duplicates(subset=[f'{ll}', 'synth_kind'])
            met_av = to_plot.groupby('synth_kind', as_index=False)[f'{ll}_{met}{suffix}'].mean()
            gain_av = to_plot.groupby('synth_kind', as_index=False)[f'{ll}_rel_gain_avg{suffix}'].mean()

            if ll == 'BG':
                colors = {'N': 'cornflowerblue', 'M': 'royalblue', 'T': 'blue', 'S': 'darkblue', 'C': 'black'}
            elif ll == 'FG':
                colors = {'N': 'yellowgreen', 'M': 'forestgreen', 'T': 'green', 'S': 'darkgreen', 'C': 'darkslategrey'}

            for key, val in colors.items():
                if cnt == 0:
                    plot = sb.scatterplot(x=f'{ll}_{met}{suffix}', y=f'{ll}_rel_gain_avg{suffix}',
                                          data=to_plot.loc[to_plot.synth_kind == key],
                                          ax=ax[cnt + count], s=24, color=val, label=f'{key}, {ll}')

                else:
                    sb.scatterplot(x=f'{ll}_{met}{suffix}', y=f'{ll}_rel_gain_avg{suffix}',
                                   data=to_plot.loc[to_plot.synth_kind == key],
                                   ax=ax[cnt + count], s=24, color=val)

                sb.scatterplot(x=f'{ll}_{met}{suffix}', y=ylim_max * 1.1, data=met_av.loc[met_av.synth_kind == key],
                               ax=ax[cnt + count], marker='v', color=val, s=150)
                sb.scatterplot(x=xlim_max * 1.1, y=f'{ll}_rel_gain_avg{suffix}',
                               data=gain_av.loc[gain_av.synth_kind == key],
                               ax=ax[cnt + count], marker='<', color=val, s=150)

                plt.setp(plot.get_legend().get_texts(), fontsize='5')
                plt.show()

            ax[cnt + count].set_ylim(ylim_min, ylim_max * 1.1), ax[cnt + count].set_xlim(xlim_min, xlim_max * 1.1)
            ax[cnt + count].set_ylabel(''), ax[cnt + count].set_xlabel('')
            ax[count].set_ylabel(f'Relative Gain', fontweight='bold', fontsize=10)

    count += len(x_metrics)

    greys = cm.get_cmap('viridis', 12)
    cols = greys(np.linspace(0, 0.85, len(synth_show))).tolist()
    cols.reverse()
    # synth_show.reverse()

    colors = dict(zip(synth_show, cols))

    for cnt, met in enumerate(x_metrics):

        xlim_max = np.max([df[f'BG_{met}{suffix}'].max(), df[f'FG_{met}{suffix}'].max()])
        xlim_min = np.min([df[f'BG_{met}{suffix}'].min(), df[f'FG_{met}{suffix}'].min()])
        xlim_min = xlim_min - np.abs(xlim_max) * 0.05

        to_plot_bg = df[[f'BG', f'BG_rel_gain_avg{suffix}', 'synth_kind', f'BG_{met}{suffix}']]
        to_plot_fg = df[[f'FG', f'FG_rel_gain_avg{suffix}', 'synth_kind', f'FG_{met}{suffix}']]
        to_plot_bg = to_plot_bg.drop_duplicates(subset=[f'BG', 'synth_kind'])
        to_plot_fg = to_plot_fg.drop_duplicates(subset=[f'FG', 'synth_kind'])
        to_plot_bg, to_plot_fg = to_plot_bg.drop(labels=['BG'], axis=1), to_plot_fg.drop(labels=['FG'], axis=1)

        to_plot_bg = to_plot_bg.rename(
            columns={f'BG_rel_gain_avg{suffix}': 'rel_gain', f'BG_{met}{suffix}': f'{met}{suffix}'})
        to_plot_fg = to_plot_fg.rename(
            columns={f'FG_rel_gain_avg{suffix}': 'rel_gain', f'FG_{met}{suffix}': f'{met}{suffix}'})

        to_plot = pd.concat([to_plot_bg, to_plot_fg])

        met_av = to_plot.groupby('synth_kind', as_index=False)[f'{met}{suffix}'].mean()
        # met_std_av = to_plot.groupby('synth_kind', as_index=False)[f'{ll}_{met}'].mean()

        to_plot_list = []
        for key, val in colors.items():
            synth_to_plot = to_plot.loc[to_plot.synth_kind == key]

            sb.scatterplot(x=f'{met}{suffix}', y=f'rel_gain', data=synth_to_plot, ax=ax[cnt + count], s=24, color=val)

            sb.scatterplot(x=f'{met}{suffix}', y=ylim_max * 1.1, data=met_av.loc[met_av.synth_kind == key],
                           ax=ax[cnt + count], marker='v', color=val, s=150)

            # plt.setp(plot.get_legend().get_texts(), fontsize='5')
            # plt.show()
            to_plot_list.append(synth_to_plot)

        for co, (key, val) in enumerate(colors.items()):
            # Run a regression
            for_reg = to_plot_list[co]
            Y = for_reg['rel_gain'].values
            X = for_reg[f'{met}{suffix}'].values
            reg = stats.linregress(X, Y)
            x = np.asarray([xlim_min, xlim_max])
            y = reg.slope * x + reg.intercept
            ax[cnt + count].plot(x, y, color=val, label=f"{key} | coef: {reg.rvalue:.3f}\n"
                                                        f"p = {reg.pvalue:.3f} | n={len(for_reg)}")
            ax[cnt + count].legend()

        ax[cnt + count].set_ylim(ylim_min, ylim_max * 1.1), ax[cnt + count].set_xlim(xlim_min, xlim_max * 1.1)
        ax[cnt + count].set_ylabel(''), ax[cnt + count].set_xlabel('')
        ax[count].set_ylabel(f'Relative Gain', fontweight='bold', fontsize=10)

        if x_labels:
            ax[cnt + count].set_xlabel(f'{x_labels[cnt]}', fontweight='bold', fontsize=10)
        else:
            ax[cnt + count].set_ylabel(f'{met}{suffix}', fontweight='bold', fontsize=10)

    fig.suptitle(f'{df.area.unique()[0]} - FR threshold: {fr_thresh} - r > {r_cut} - n={len(df)} - {suffix}',
                 fontsize=12, fontweight='bold')


def synthetic_sound_metric_scatters(filt, x_metrics, x_labels=None, synth_show=['N', 'M', 'S', 'T', 'C'],
                                  suffix=''):
    '''2023_07_27. Reworked osyn.sound_metric_scatter_combined() so that the rows are the different areas and the
    columns are the metrics you give it.'''
    areas = filt.area.unique().tolist()

    area_sound_dict = {}
    for aa in areas:
        area_df = filt.loc[filt.area==aa]
        print(f'Getting your {aa} sound_df.')
        area_sound_dict[aa] = ohel.get_sound_statistics_from_df(area_df, percent_lims=[15, 85], append=True)
    sound_df = pd.concat(list(area_sound_dict.values()))

    ylim_max = np.max([sound_df[f'BG_rel_gain_avg{suffix}'].max(), sound_df[f'FG_rel_gain_avg{suffix}'].max()])
    ylim_min = np.min([sound_df[f'BG_rel_gain_avg{suffix}'].min(), sound_df[f'FG_rel_gain_avg{suffix}'].min()])

    fig, axes = plt.subplots(len(areas), len(x_metrics), figsize=(len(x_metrics) *4, len(areas) *5), sharey=True, sharex='col')
    ax = axes.ravel()

    count = 0
    for ar in areas:
        area_df = area_sound_dict[ar]
        for ll in ['BG', 'FG']:
            for cnt, met in enumerate(x_metrics):

                xlim_max = np.max([area_df[f'BG_{met}{suffix}'].max(), area_df[f'FG_{met}{suffix}'].max()])
                xlim_min = np.min([area_df[f'BG_{met}{suffix}'].min(), area_df[f'FG_{met}{suffix}'].min()])
                xlim_min = xlim_min - np.abs(xlim_max) * 0.05

                to_plot = area_df[[f'{ll}', f'{ll}_rel_gain_avg{suffix}', 'synth_kind', f'{ll}_{met}{suffix}']]
                to_plot = to_plot.drop_duplicates(subset=[f'{ll}', 'synth_kind'])
                met_av = to_plot.groupby('synth_kind', as_index=False)[f'{ll}_{met}{suffix}'].mean()
                gain_av = to_plot.groupby('synth_kind', as_index=False)[f'{ll}_rel_gain_avg{suffix}'].mean()

                if ll == 'BG':
                    colors = {'N': 'cornflowerblue', 'M': 'royalblue', 'T': 'blue', 'S': 'darkblue', 'C': 'black'}
                elif ll == 'FG':
                    colors = {'N': 'yellowgreen', 'M': 'forestgreen', 'T': 'green', 'S': 'darkgreen', 'C': 'darkslategrey'}

                for key, val in colors.items():
                    if cnt == 0:
                        plot = sb.scatterplot(x=f'{ll}_{met}{suffix}', y=f'{ll}_rel_gain_avg{suffix}',
                                              data=to_plot.loc[to_plot.synth_kind == key],
                                              ax=ax[cnt + count], s=24, color=val, label=f'{key}, {ll}')
                        ax[cnt+count].set_title(f"{ar}: n={int(len(area_df)/len(synth_show))}", fontweight='bold', loc='left', fontsize=12)

                    else:
                        sb.scatterplot(x=f'{ll}_{met}{suffix}', y=f'{ll}_rel_gain_avg{suffix}',
                                       data=to_plot.loc[to_plot.synth_kind == key],
                                       ax=ax[cnt + count], s=24, color=val)

                    sb.scatterplot(x=f'{ll}_{met}{suffix}', y=ylim_max * 1.1, data=met_av.loc[met_av.synth_kind == key],
                                   ax=ax[cnt + count], marker='v', color=val, s=150)
                    sb.scatterplot(x=xlim_max * 1.1, y=f'{ll}_rel_gain_avg{suffix}',
                                   data=gain_av.loc[gain_av.synth_kind == key],
                                   ax=ax[cnt + count], marker='<', color=val, s=150)
                    ax[cnt+count].xaxis.set_tick_params(labelbottom=True)

                    plt.setp(plot.get_legend().get_texts(), fontsize='5')
                    plt.show()

                # ax[cnt + count].set_ylim(ylim_min * 1.1, ylim_max * 1.1), ax[cnt + count].set_xlim(xlim_min, xlim_max * 1.1)
                ax[cnt + count].set_ylabel(''), ax[cnt + count].set_xlabel('')
                ax[count].set_ylabel(f'Relative Gain', fontweight='bold', fontsize=12)

                if x_labels:
                    ax[cnt+count].xaxis.set_label(True)
                    ax[cnt + count].set_xlabel(f'{x_labels[cnt]}', fontweight='bold', fontsize=12)
                else:
                    ax[cnt + count].set_ylabel(f'{met}{suffix}', fontweight='bold', fontsize=12)
        count += len(x_metrics)


def sound_metric_scatter_combined_flanks(df, x_metrics, x_labels=None, synth_show=['N', 'M', 'S', 'T', 'C'],
                                  fr_thresh=0.03, r_cut=None, suffix=''):
    '''2023_05_30. This does the same thing as sound_metric_scatter_combined, but instead of using quadrant3
    it compares the instances where only one sound elicits a response and the other does not.

    2023_05_24. Takes a sound df, must already be filtered by singular area, and the layers you want, and
    it will plot separately the mean relative gain for each sound across the different synthetic conditions. BG and FG
    will be on different rows for ease of viewing, but the axes will be shared so you can see the different spaces
    they occupy.'''
    df = df.loc[(df.kind == '11') & (df.dyn_kind == 'ff') & (df.SNR == 0) & (df.olp_type == 'synthetic')]

    if r_cut:
        df = df.dropna(axis=0, subset='r')
        df = df.loc[df[f'r{suffix}'] >= r_cut]
    df = df.copy()

    # Get rid of when the weights are unrealistic
    weight_lim = [0, 2]
    if suffix == '_start' or suffix == '_end':
        df = df.loc[((df[f'weightsA_start'] >= weight_lim[0]) & (df[f'weightsA_start'] <= weight_lim[1])) &
                    ((df[f'weightsB_start'] >= weight_lim[0]) & (df[f'weightsB_start'] <= weight_lim[1])) &
                    ((df[f'weightsA_end'] >= weight_lim[0]) & (df[f'weightsA_end'] <= weight_lim[1])) &
                    ((df[f'weightsB_end'] >= weight_lim[0]) & (df[f'weightsB_end'] <= weight_lim[1]))]
    else:
        df = df.loc[((df[f'weightsA'] >= weight_lim[0]) & (df[f'weightsA'] <= weight_lim[1])) &
                    ((df[f'weightsB'] >= weight_lim[0]) & (df[f'weightsB'] <= weight_lim[1]))]

    df_bg = df.loc[(np.abs(df[f'bg_FR{suffix}']) <= fr_thresh) & (df[f'fg_FR{suffix}'] >= fr_thresh)]
    df_fg = df.loc[(df[f'bg_FR{suffix}'] >= fr_thresh) & (np.abs(df[f'fg_FR{suffix}']) <= fr_thresh)]

    df_bg = ohel.get_sound_statistics_from_df(df_bg, percent_lims=[15,85], append=True)
    df_fg = ohel.get_sound_statistics_from_df(df_fg, percent_lims=[15,85], append=True)

    ylim_max = np.max([df_bg[f'BG_rel_gain_avg{suffix}'].max(), df_fg[f'FG_rel_gain_avg{suffix}'].max()])
    ylim_min = np.min([df_bg[f'BG_rel_gain_avg{suffix}'].min(), df_fg[f'FG_rel_gain_avg{suffix}'].min()])

    fig, axes = plt.subplots(2, len(x_metrics), figsize=(12, 5 * len(['BG', 'FG'])), sharey=True)
    ax = axes.ravel()

    count = 0
    for cnt, met in enumerate(x_metrics):

        xlim_max = np.max([df_bg[f'BG_{met}{suffix}'].max(), df_fg[f'FG_{met}{suffix}'].max()])
        xlim_min = np.min([df_bg[f'BG_{met}{suffix}'].min(), df_fg[f'FG_{met}{suffix}'].min()])
        xlim_min = xlim_min - np.abs(xlim_max) * 0.05

        for ll, dff in zip(['BG', 'FG'], [df_bg, df_fg]):

            to_plot = dff[[f'{ll}', f'{ll}_rel_gain_avg{suffix}', 'synth_kind', f'{ll}_{met}{suffix}']]
            to_plot = to_plot.drop_duplicates(subset=[f'{ll}', 'synth_kind'])
            met_av = to_plot.groupby('synth_kind', as_index=False)[f'{ll}_{met}{suffix}'].mean()
            gain_av = to_plot.groupby('synth_kind', as_index=False)[f'{ll}_rel_gain_avg{suffix}'].mean()

            if ll == 'BG':
                colors = {'N': 'cornflowerblue', 'M': 'royalblue', 'T': 'blue', 'S': 'darkblue', 'C': 'black'}
            elif ll == 'FG':
                colors = {'N': 'yellowgreen', 'M': 'forestgreen', 'T': 'green', 'S': 'darkgreen', 'C': 'darkslategrey'}

            for key, val in colors.items():
                if cnt == 0:
                    plot = sb.scatterplot(x=f'{ll}_{met}{suffix}', y=f'{ll}_rel_gain_avg{suffix}',
                                          data=to_plot.loc[to_plot.synth_kind == key],
                                          ax=ax[cnt + count], s=24, color=val, label=f'{key}, {ll}')

                else:
                    sb.scatterplot(x=f'{ll}_{met}{suffix}', y=f'{ll}_rel_gain_avg{suffix}',
                                   data=to_plot.loc[to_plot.synth_kind == key],
                                   ax=ax[cnt + count], s=24, color=val)

                sb.scatterplot(x=f'{ll}_{met}{suffix}', y=ylim_max * 1.1, data=met_av.loc[met_av.synth_kind == key],
                               ax=ax[cnt + count], marker='v', color=val, s=150)
                sb.scatterplot(x=xlim_max * 1.1, y=f'{ll}_rel_gain_avg{suffix}',
                               data=gain_av.loc[gain_av.synth_kind == key],
                               ax=ax[cnt + count], marker='<', color=val, s=150)

                plt.setp(plot.get_legend().get_texts(), fontsize='5')
                plt.show()

            ax[cnt + count].set_ylim(ylim_min, ylim_max * 1.1), ax[cnt + count].set_xlim(xlim_min, xlim_max * 1.1)
            ax[cnt + count].set_ylabel(''), ax[cnt + count].set_xlabel('')
            ax[count].set_ylabel(f'Relative Gain', fontweight='bold', fontsize=10)

    count += len(x_metrics)

    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    greys = cm.get_cmap('viridis', 12)
    cols = greys(np.linspace(0, 0.85, len(synth_show))).tolist()
    cols.reverse()
    # synth_show.reverse()

    colors = dict(zip(synth_show, cols))

    for cnt, met in enumerate(x_metrics):

        xlim_max = np.max([df_bg[f'BG_{met}{suffix}'].max(), df_fg[f'FG_{met}{suffix}'].max()])
        xlim_min = np.min([df_bg[f'BG_{met}{suffix}'].min(), df_fg[f'FG_{met}{suffix}'].min()])
        xlim_min = xlim_min - np.abs(xlim_max) * 0.05

        to_plot_bg = df_bg[[f'BG', f'BG_rel_gain_avg{suffix}', 'synth_kind', f'BG_{met}{suffix}']]
        to_plot_fg = df_fg[[f'FG', f'FG_rel_gain_avg{suffix}', 'synth_kind', f'FG_{met}{suffix}']]
        to_plot_bg = to_plot_bg.drop_duplicates(subset=[f'BG', 'synth_kind'])
        to_plot_fg = to_plot_fg.drop_duplicates(subset=[f'FG', 'synth_kind'])
        to_plot_bg, to_plot_fg = to_plot_bg.drop(labels=['BG'], axis=1), to_plot_fg.drop(labels=['FG'], axis=1)

        to_plot_bg = to_plot_bg.rename(
            columns={f'BG_rel_gain_avg{suffix}': 'rel_gain', f'BG_{met}{suffix}': f'{met}{suffix}'})
        to_plot_fg = to_plot_fg.rename(
            columns={f'FG_rel_gain_avg{suffix}': 'rel_gain', f'FG_{met}{suffix}': f'{met}{suffix}'})

        to_plot = pd.concat([to_plot_bg, to_plot_fg])

        met_av = to_plot.groupby('synth_kind', as_index=False)[f'{met}{suffix}'].mean()
        # met_std_av = to_plot.groupby('synth_kind', as_index=False)[f'{ll}_{met}'].mean()

        to_plot_list = []
        for key, val in colors.items():
            synth_to_plot = to_plot.loc[to_plot.synth_kind == key]
            # if cnt==0:
            #     plot = sb.scatterplot(x=f'{met}', y=f'rel_gain', data=synth_to_plot,
            #                           ax=ax[cnt+count], s=24, color=val, label=f'{key}')
            #
            # else:
            sb.scatterplot(x=f'{met}{suffix}', y=f'rel_gain', data=synth_to_plot, ax=ax[cnt + count], s=24, color=val)

            sb.scatterplot(x=f'{met}{suffix}', y=ylim_max * 1.1, data=met_av.loc[met_av.synth_kind == key],
                           ax=ax[cnt + count], marker='v', color=val, s=150)

            # plt.setp(plot.get_legend().get_texts(), fontsize='5')
            # plt.show()
            to_plot_list.append(synth_to_plot)

        for co, (key, val) in enumerate(colors.items()):
            # Run a regression
            for_reg = to_plot_list[co]
            Y = for_reg['rel_gain'].values
            X = for_reg[f'{met}{suffix}'].values
            reg = stats.linregress(X, Y)
            x = np.asarray([xlim_min, xlim_max])
            y = reg.slope * x + reg.intercept
            ax[cnt + count].plot(x, y, color=val, label=f"{key} | coef: {reg.rvalue:.3f}\n"
                                                        f"p = {reg.pvalue:.3f} | n={len(for_reg)}")
            ax[cnt + count].legend()

        ax[cnt + count].set_ylim(ylim_min, ylim_max * 1.1), ax[cnt + count].set_xlim(xlim_min, xlim_max * 1.1)
        ax[cnt + count].set_ylabel(''), ax[cnt + count].set_xlabel('')
        ax[count].set_ylabel(f'Relative Gain', fontweight='bold', fontsize=10)

        if x_labels:
            ax[cnt + count].set_xlabel(f'{x_labels[cnt]}', fontweight='bold', fontsize=10)
        else:
            ax[cnt + count].set_ylabel(f'{met}{suffix}', fontweight='bold', fontsize=10)

    fig.suptitle(f'{df.area.unique()[0]} - FR threshold: {fr_thresh} - r > {r_cut} - '
                 f'bg_n={len(df_bg)}, fg_n={len(df_fg)} - flank quads - '
                 f'{suffix}',
                 fontsize=12, fontweight='bold')


def synthetic_rel_gain_summary(filt, synth_show=['M', 'S', 'T', 'C']):
    '''2023_07_25. A worse version of osyn.synthetic_summary_relative_gain_all_areas(). Scatter not bar.'''
    fig, ax = plt.subplots(1, 1, figsize=(5,8))
    cnt_list = []
    for cnt, syn in enumerate(synth_show):
        x_pos = cnt+1
        syn_df = filt.loc[filt.synth_kind==syn]
        a1, peg = syn_df.loc[syn_df.area=='A1'], syn_df.loc[syn_df.area=='PEG']
        a1_mean, peg_mean = a1.FG_rel_gain.mean(), peg.FG_rel_gain.mean()
        a1_sem, peg_sem = a1.FG_rel_gain.sem(), peg.FG_rel_gain.sem()

        ax.scatter(x_pos-0.1, a1_mean, color='indigo')
        ax.scatter(x_pos+0.1, peg_mean, color='maroon')
        ax.errorbar(x_pos-0.1, y=a1_mean, yerr=a1_sem, ls='none', color='indigo', capsize=5)
        ax.errorbar(x_pos+0.1, y=peg_mean, yerr=peg_sem, ls='none', color='maroon', capsize=5)
        cnt_list.append(x_pos)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, 0)
    ax.xaxis.tick_top()
    ax.spines['top'].set_visible(True), ax.spines['bottom'].set_visible(False)
    ax.set_xticks(cnt_list)
    ax.set_xticklabels(synth_show, fontsize=10, fontweight='bold')
    ax.set_ylabel('Relative Gain', fontsize=12, fontweight='bold')


def synthetic_summary_relative_gain_all_areas(filt, synth_show=['M', 'S', 'T', 'C'], mult_comp=1):
    '''2023_07_25. Takes a dataframe that has been filtered using ohel.filter_across_synths() and a list of
    conditions to plot and will make a summary horizontal bar plot of the average relative gains. Use
    mult_comp not being default 1 if you want to make multiple comparisons in each area.'''
    kind_dict = {'A': 'Non-RMS Norm\nNatural', 'N': 'Natural', 'M': 'Spectrotemporal\nModulation',
                  'U': 'Spectro-\ntemporal', 'S': 'Spectral', 'T': 'Temporal', 'C': 'Cochlear'}

    # Grab the areas that are present in the dataframe, separate the dataframe by area, and make a list of
    # dictionaries of each synth condition in synth_show for each area
    areas = filt.area.unique().tolist()
    area_dicts = {dd:filt.loc[filt.area==dd] for dd in areas}
    synth_dicts = [{f'{ar}_{syn}':area_dicts[ar].loc[area_dicts[ar].synth_kind == syn] for syn in synth_show} for ar in areas]

    # Get all comparisons possible given the synth_show parameter
    c = list(itertools.combinations(synth_show, 2))
    stat_combos = [''.join(dd) for dd in c]

    # Calculate individual stats for each synthetic combination and area and make one big dict to return
    stat_dict = {}
    for cnt, ar in enumerate(areas):
        sd = synth_dicts[cnt]
        for ss in stat_combos:
            one, two = ss[0], ss[1]
            stat_dict[f'{ar}_{ss}'] = stats.wilcoxon(sd[f'{ar}_{one}']['FG_rel_gain'],
                                                      sd[f'{ar}_{two}']['FG_rel_gain']).pvalue * mult_comp

    synth_show.reverse()
    ylabels = [kind_dict[kk] for kk in synth_show]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    y_pos_list = []
    for cnt, ss in enumerate(synth_show):
        y_pos = cnt+1
        if cnt==0:
            ax.barh(y=y_pos+0.2, width=synth_dicts[0][f'A1_{ss}']['FG_rel_gain'].mean(), color='violet',
                    linestyle='None', height=0.4, label='A1')
            ax.barh(y=y_pos-0.2, width=synth_dicts[1][f'PEG_{ss}']['FG_rel_gain'].mean(), color='coral',
                    linestyle='None', height=0.4, label='PEG')
        else:
            ax.barh(y=y_pos + 0.2, width=synth_dicts[0][f'A1_{ss}']['FG_rel_gain'].mean(), color='violet',
                    linestyle='None', height=0.4)
            ax.barh(y=y_pos - 0.2, width=synth_dicts[1][f'PEG_{ss}']['FG_rel_gain'].mean(), color='coral',
                    linestyle='None', height=0.4)
        ax.errorbar(y=y_pos+0.2, x=synth_dicts[0][f'A1_{ss}']['FG_rel_gain'].mean(), elinewidth=2, capsize=4,
                    xerr=synth_dicts[0][f'A1_{ss}']['FG_rel_gain'].sem(), color='black', linestyle='None', yerr=None)
        ax.errorbar(y=y_pos-0.2, x=synth_dicts[1][f'PEG_{ss}']['FG_rel_gain'].mean(), elinewidth=2, capsize=4,
                    xerr=synth_dicts[1][f'PEG_{ss}']['FG_rel_gain'].sem(), color='black', linestyle='None', yerr=None)

        ax.legend(fontsize=10)
        y_pos_list.append(y_pos)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines['left'].set_visible(False), ax.spines['right'].set_visible(True)

    ax.set_yticks(y_pos_list)
    ax.set_yticklabels(ylabels, fontsize=10, fontweight='bold')

    ax.tick_params(axis='x', labelsize=10)
    ax.set_xlabel("Relative Gain", fontsize=12, fontweight='bold')

    # ax.set_title(f"{ss} Ref: {weight_df0.filt_by.unique()[0]} - n={len(C)}", fontsize=10, fontweight='bold')
    fig.tight_layout()

    return stat_dict
