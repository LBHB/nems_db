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