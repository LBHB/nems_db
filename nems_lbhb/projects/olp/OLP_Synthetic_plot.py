import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
import copy
import nems_lbhb.projects.olp.OLP_helpers as ohel
import nems_lbhb.projects.olp.OLP_fit as ofit


def plot_synthetic_weights(weight_df, areas=None, thresh=0.03, quads=3, synth_show=None):
    '''Plot a bar graph comparing the BG and FG weights for the different synthetic conditions. Can
    specify if you want one or both areas and also which combination of synthetic conditions you
    want to plot, as described by a list of the strings for their codes. If you want to plot all minus
    the control for the control (A), simply use A- as the synth_show.'''
    quad, threshold = ohel.quadrants_by_FR(weight_df, threshold=thresh, quad_return=quads)

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
                                              synth_show=None):
    '''Made 2022_09_08. Makes the big figure on panel 5 of my 2022 NGP Poster. It takes a dataframe
    and a list of the synthetic code letters and vertically arranges them as histograms of relative
    gain, all aligned on the same zero so you can ideally see the shift. You also can put in a BG
    and FG name (with spaces in the name if it has!) and it will show the corresponding spectrograms
    for the synthetic conditions nextdoor to the relative gain plot. It's a fun figure.'''
    quad, _ = ohel.quadrants_by_FR(df, threshold=thresh, quad_return=quads)
    quad = quad.loc[quad.area==area]

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

        rel_gain = rel_gain.loc[rel_gain <= 2.5]
        rel_gain = rel_gain.loc[rel_gain >= -2.5]

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

