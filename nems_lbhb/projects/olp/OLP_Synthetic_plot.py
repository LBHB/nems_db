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


def plot_synthetic_weights(weight_df, areas=None, thresh=0.03, quads=3, synth_show=None, r_cut=None):
    '''2022_09_21. Added model fit filter (r_cut)
    Plot a bar graph comparing the BG and FG weights for the different synthetic conditions. Can
    specify if you want one or both areas and also which combination of synthetic conditions you
    want to plot, as described by a list of the strings for their codes. If you want to plot all minus
    the control for the control (A), simply use A- as the synth_show.'''
    quad, threshold = ohel.quadrants_by_FR(weight_df, threshold=thresh, quad_return=quads)
    if r_cut:
        quad = quad.dropna(axis=0, subset='r')
        quad = quad.loc[quad.r >= r_cut]
    #Create aliases for the kinds so I can dumbly swap them out all so they can be in the right order
    alias = {'A': '1', 'N': '2', 'C': '3', 'T': '4', 'S': '5', 'U': '6', 'M': '7'}
    kind_alias = {'1': 'Non-RMS Norm\nNatural', '2':' RMS Norm\nNatural', '3':'Cochlear',
                  '4': 'Temporal', '5': 'Spectral', '6': 'Spectro-\ntemporal', '7': 'Spectrotemporal\nModulation'}

    # Shortcut to typing in all the actual conditions minus the control of the control
    if synth_show == 'A-':
        synth_show = ['N', 'C', 'T', 'S', 'U', 'M']

    # This let's you only show certain synthetic kinds, but it will always be ordered in the same way
    if synth_show:
        quad = quad.loc[quad['synth_kind'].isin(synth_show)]
        width = len(synth_show) + 1
    else:
        width = 8

    # If you just want one area it can do that, or a list of areas. If you do nothing it'll plot
    # as many areas as are represented in the df (should only be two at most...)
    if isinstance(areas, str):
        fig, axes = plt.subplots(1, 1, figsize=(width,4))
        areas = [areas]
    elif isinstance(areas, list):
        fig, axes = plt.subplots(len(areas), 1, figsize=(width,4*len(areas)), sharey=True)
    else:
        fig, axes = plt.subplots(len(weight_df.area.unique()), 1,
                                 figsize=(width,4*(len(weight_df.area.unique()))), sharey=True)
        areas = weight_df.area.unique().tolist()

    for (ax, area) in zip(axes, areas):
        area_df = quad.loc[quad.area == area]
        # Extract only the relevant columns for plotting right now
        to_plot = area_df.loc[:,['synth_kind', 'weightsA', 'weightsB']].copy()
        # Sort them by the order of kinds so it'll plot in an order that I want to see
        to_plot['sort'] = to_plot['synth_kind']
        to_plot = to_plot.replace(alias).sort_values('sort').drop('sort', axis=1).copy()
        # Put the dataframe into a format that can be plotted easily
        to_plot = to_plot.melt(id_vars='synth_kind', value_vars=['weightsA', 'weightsB'], var_name='weight_kind',
                     value_name='weights').replace({'weightsA':'BG', 'weightsB':'FG'}).replace(kind_alias)

        # Plot
        sb.barplot(ax=ax, x="synth_kind", y="weights", hue="weight_kind", data=to_plot, ci=68, estimator=np.mean)
        ax.set_title(area, fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.legend(loc='upper right')
        ax.set_ylabel('Model Weights', fontsize=8, fontweight='bold')


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


def synthetic_relative_gain_comparisons_specs(df, bg, fg, thresh=0.03, quads=3, area='A1', batch=340,
                                              synth_show=None, r_cut=None, rel_cut=2.5):
    '''Made 2022_09_08. Makes the big figure on panel 5 of my 2022 NGP Poster. It takes a dataframe
    and a list of the synthetic code letters and vertically arranges them as histograms of relative
    gain, all aligned on the same zero so you can ideally see the shift. You also can put in a BG
    and FG name (with spaces in the name if it has!) and it will show the corresponding spectrograms
    for the synthetic conditions nextdoor to the relative gain plot. It's a fun figure.'''
    quad, _ = ohel.quadrants_by_FR(df, threshold=thresh, quad_return=quads)
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
    fig, axes = plt.subplots(figsize=(8, lens*2))
    fig, axes = plt.subplots(figsize=(12, 18))
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
        supps = [cc for cc in rel_gain if cc < 0]
        percent_supp = np.around((len(supps) / len(rel_gain)) * 100, 1)
        print(percent_supp)

        if rel_cut:
            rel_gain = rel_gain.loc[rel_gain <= rel_cut]
            rel_gain = rel_gain.loc[rel_gain >= -rel_cut]

        # Plot
        p = sb.distplot(rel_gain, bins=50, color='black', norm_hist=True, kde=True, ax=ax[qq],
                        kde_kws=dict(linewidth=0.5))

        ymin, ymax = ax[qq].get_ylim()
        ax[qq].vlines(0, ymin, ymax, color='black', ls = '--', lw=1)
        # Change color of suppressed to red, enhanced to blue
        for rectangle in p.patches:
            if rectangle.get_x() < 0:
                rectangle.set_facecolor('tomato')
        for rectangle in p.patches:
            if rectangle.get_x() >= 0:
                rectangle.set_facecolor('dodgerblue')
        # This might have to be change if I use %, which I want to, but density is between 0-1, cleaner
        ax[qq].set_yticks([0,1])
        ax[qq].set_yticklabels([0,1])
        # All axes match natural. Would need to be changed if natural cuts for some reason.
        if qq == 0:
            ax[qq].set_ylabel('Density', fontweight='bold', fontsize=8)
            xmin, xmax = ax[qq].get_xlim()
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
                                   horizontalalignment='center', rotation=0, labelpad=40,
                                   verticalalignment='center')
            if qq == 0 and ww == 1:
                ax[qqq].set_title(f"BG: {bg}", fontweight='bold', fontsize=10)
            elif qq == 0 and ww == 2:
                ax[qqq].set_title(f"FG: {fg}", fontweight='bold', fontsize=10)
            # if qq == (lens - 1):
            #     ax[qqq].set_xlabel('Time (s)', fontweight='bold', fontsize=10)


def synthetic_relative_gain_comparisons(df, thresh=0.03, quads=3, area=None,
                                              synth_show=None, r_cut=None, rel_cut=2.5):
    '''Updated 2022_10_03. Took out the spectrograms from the original function.
    Made 2022_09_08. Makes the big figure on panel 5 of my 2022 NGP Poster. It takes a dataframe
    and a list of the synthetic code letters and vertically arranges them as histograms of relative
    gain, all aligned on the same zero so you can ideally see the shift. You also can put in a BG
    and FG name (with spaces in the name if it has!) and it will show the corresponding spectrograms
    for the synthetic conditions nextdoor to the relative gain plot. It's a fun figure.'''
    quad, _ = ohel.quadrants_by_FR(df, threshold=thresh, quad_return=quads)
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

    for qq, (ax, syn) in enumerate(zip(axes, synth_show)):
        to_plot = quad.loc[quad.synth_kind==syn].copy()

        # Calculate percent suppressed
        rel_gain = (to_plot.weightsB - to_plot.weightsA) / \
                              (to_plot.weightsB + to_plot.weightsA)
        supps = [cc for cc in rel_gain if cc < 0]
        percent_supp = np.around((len(supps) / len(rel_gain)) * 100, 1)
        print(percent_supp)

        if rel_cut:
            rel_gain = rel_gain.loc[rel_gain <= rel_cut]
            rel_gain = rel_gain.loc[rel_gain >= -rel_cut]

        # Plot
        p = sb.distplot(rel_gain, bins=50, color='black', norm_hist=True, kde=True, ax=ax,
                        kde_kws=dict(linewidth=0.5))

        ymin, ymax = ax.get_ylim()
        ax.vlines(0, ymin, ymax, color='black', ls = '--', lw=1)
        # Change color of suppressed to red, enhanced to blue
        for rectangle in p.patches:
            if rectangle.get_x() < 0:
                rectangle.set_facecolor('tomato')
        for rectangle in p.patches:
            if rectangle.get_x() >= 0:
                rectangle.set_facecolor('dodgerblue')
        # This might have to be change if I use %, which I want to, but density is between 0-1, cleaner
        ax.set_yticks([0,1])
        ax.set_yticklabels([0,1])
        # All axes match natural. Would need to be changed if natural cuts for some reason.
        if qq == 0:
            ax.set_ylabel('Density', fontweight='bold', fontsize=8)
            xmin, xmax = ax.get_xlim()
        else:
            ax.set_xlim(xmin, xmax)
            ax.set_ylabel('')

        if qq == (len(synth_show)-1):
            ax.set_xlabel('Relative Gain', fontweight='bold', fontsize=10)
        else:
            ax.set_xticklabels([])
        ax.text((xmin+(np.abs(xmin)*0.1)), 0.75, f"Percent\nSuppression:\n{percent_supp}", fontsize=6)

        ax.set_ylabel(f"{kind_dict[syn]}", fontweight='bold', fontsize=10)

    fig.suptitle(f'Area: {area}, r >= {r_cut}, quads={quads}, rel_cut={rel_cut}, thresh={thresh}',
                 fontsize=7, fontweight='bold')
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