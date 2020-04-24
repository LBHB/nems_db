import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as sci
import statsmodels.formula.api as smf
import matplotlib.collections as clt
import re
import pylab as pl

from nems_lbhb.pupil_behavior_scripts.mod_per_state import get_model_results_per_state_model
from nems_lbhb.pupil_behavior_scripts.mod_per_state import aud_vs_state
from nems_lbhb.pupil_behavior_scripts.mod_per_state import hlf_analysis
from nems_lbhb.stateplots import model_per_time_wrapper, beta_comp
import nems.plots.api as nplt

params = {'legend.fontsize': 6,
          'figure.figsize': (8, 6),
          'axes.labelsize': 6,
          'axes.titlesize': 6,
          'xtick.labelsize': 6,
          'ytick.labelsize': 6,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)

color_b = '#C768D8'
color_p = '#47BF55'
#color_p = '#4ED163'
color_both = '#000000'
color_either = '#595959'
color_ns = '#BFBFBF'


color_list=[color_ns, color_either, color_b, color_p, color_both]

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 100)


def scat_states(df,
                x_model,
                y_model,
                x_beh_state,
                y_beh_state=None,
                area=None,
                sig_list=None,
                x_column=None,
                y_column=None,
                color_list=None,
                highlight_cellids={},
                hue=False,
                save=False,
                pup_state=None,
                xlabel=None,
                ylabel=None,
                title=None,
                xlim=None,
                ylim=None,
                marker='o',
                ax=None):
    '''This function makes a scatter plots of identified arguments.
    sig_list = ~sig_state, sig_state, sig_ubeh, sig_upup, sig_both]
    color_list = ['#D3D3D3', '#595959', '#82418B', '#2E7E3E', '#000000']

    TODO: make sig_list and color_list into a dict
    highlight_cellid has to be a dict 'cellid':'color'
    add hue if istead of plotting color for stat sign units,
    you want to plot different color based on values in a column (eg onBF)'''

    if ax is None:
        plt.figure(figsize=(6, 6))
        ax=plt.gca()
    else:
        plt.sca(ax)

    if y_beh_state is None:
        y_beh_state = x_beh_state

    # need a slope and c to fix the position of line
    if xlim is not None:
        xlim = xlim
        ylim = ylim
        slope = 1
        c = xlim[0]

        x_min = xlim[0]
        x_max = xlim[1]
        y_min, y_max = c, c + slope * (x_max - x_min)
        plt.plot([x_min, x_max], [y_min, y_max], linewidth=0.5, linestyle='--', color='k')

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)

    plt.axvline(0, linestyle='--', linewidth=0.5, color='k')
    plt.axhline(0, linestyle='--', linewidth=0.5, color='k')

    if hue:
        sns.scatterplot(x=df.loc[x_model & x_beh_state & area, x_column].tolist(),
                        y=df.loc[y_model & y_beh_state & area, y_column].tolist(),
                        s=200, hue=df.loc[x_model & x_beh_state & area, hue],
                        marker=marker, edgecolors='white', linewidth=0.5)

    elif pup_state:
        # plot not significant units
        plt.scatter(x=df.loc[x_model & pup_state & area & sig_list[0], x_column].tolist(),
                    y=df.loc[y_model & x_beh_state & area & sig_list[0], y_column].tolist(),
                    s=150, color=color_list[0], marker=marker, edgecolors='white', linewidth=0.5)

        # plot significant state units
        plt.scatter(x=df.loc[x_model & pup_state & area & sig_list[1], x_column].tolist(),
                    y=df.loc[y_model & x_beh_state & area & sig_list[1], y_column].tolist(),
                    s=200, color=color_list[1], marker=marker, edgecolors='white', linewidth=0.5)

        # plot significant unique behavior
        plt.scatter(x=df.loc[x_model & pup_state & area & sig_list[2], x_column].tolist(),
                    y=df.loc[y_model & x_beh_state & area & sig_list[2], y_column].tolist(),
                    s=200, color=color_list[2], marker=marker, edgecolors='white', linewidth=0.5)

        # plot significant unique pupil
        plt.scatter(x=df.loc[x_model & pup_state & area & sig_list[3], x_column].tolist(),
                    y=df.loc[y_model & x_beh_state & area & sig_list[3], y_column].tolist(),
                    s=200, color=color_list[3], marker=marker, edgecolors='white', linewidth=0.5)

        # plot significant unique both
        plt.scatter(x=df.loc[x_model & pup_state & area & sig_list[4], x_column].tolist(),
                    y=df.loc[y_model & x_beh_state & area & sig_list[4], y_column].tolist(),
                    s=200, color=color_list[4], marker=marker, edgecolors='white', linewidth=0.5)

    else:
        #import pdb
        #pdb.set_trace()

        # iterate: not significant units, sig state, sig u beh, sig u pup, sig u both
        for i, sig in enumerate(sig_list):
            x = df.loc[x_model & x_beh_state & area & sig, x_column].values
            y = df.loc[y_model & y_beh_state & area & sig, y_column].values
            x = np.clip(x, xlim[0], xlim[1])
            y = np.clip(y, ylim[0], ylim[1])
            x_outlier = (x <= xlim[0]) | (x >= xlim[1])
            y_outlier = (y <= xlim[0]) | (y >= xlim[1])

            if i == 0:
                s = 75
            else:
                s = 100

            # plot current group
            plt.scatter(x=x, y=y, s=s,
                        color=color_list[i], marker=marker, edgecolors='white', linewidth=0.5)

    # plot a cellid (e.g. TAR010c-27-2 (A1 behavior cell) or TAR010c-06-1 (A1 pupil cell)) with special color

    if type(highlight_cellids) is not dict:
        raise Exception('highlight_cellids has got to be a dict!')
    else:
        for cellid, color in highlight_cellids.items():
            plt.scatter(x=df.loc[x_model & x_beh_state & area & (df['cellid'] == cellid), x_column].tolist(),
                        y=df.loc[y_model & y_beh_state & area & (df['cellid'] == cellid), y_column].tolist(),
                        s=200, color=color, marker=marker, edgecolors='white', linewidth=0.5)
    ax.set_aspect('equal', 'box')
    nplt.ax_remove_box(ax)

    if save:
        plt.savefig(title + ylabel + xlabel + '.pdf')


def scat_states_crh(df,
                x_model,
                y_model,
                area=None,
                colors=None,
                highlight_cellids={},
                pup_state=False,
                hue=False,
                save=False,
                xlabel=None,
                ylabel=None,
                title=None,
                xlim=None,
                ylim=None,
                marker='o', marker_size=15,
                ax=None):
    """
    This function makes a scatter plots of identified arguments.
    sig_list = ~sig_state, sig_state, sig_ubeh, sig_upup, sig_both]
    color_list = ['#D3D3D3', '#595959', '#82418B', '#2E7E3E', '#000000']

    crh copy of scat_states 04/16/2020. Tweak some fn arguments to make a little 
    more user friendly, I think.

    params:
        df                - pandas dataframe with results
        x_model           - string name of x column
        y_model           - string name of y column
        area              - string (A1, ICC, or ICX)
        colors            - list of length 5 (color of not sig cells, sig state, sig beh, sig pup, sig both). 
                                If none, set to defaults.
        highlight_cellids - Dict of cellid / color pairs. If specified, will highlight these cellids
        xlabel            - string, label of X axis
        ylabel            - string, label of Y axis
        title             - string, title of axis
        xlim              - tuple, limits of x axis
        ylim              - tuple, limits of y axis
        ax                - axis object on which to make the scatter plot
        marker            - matplotlib marker
        save              - bool, if True, save pdf of figure
        hue               - string column name, if specified, use this column to determine groups 
                                (using seaborn grouping of dataframe on this column)
    """

    # generate sig_list (a list of boolean masks for each of the following conditions:
    #       not sig cells, sig state, sig task, sig pup, sig both)
    sig_list = [(~df['sig_state'] & ~df['sig_upupil'] & ~df['sig_utask']),
                (df['sig_state'] & ~df['sig_upupil'] & ~df['sig_utask']),
                (df['sig_utask'] & ~df['sig_upupil']),
                (~df['sig_utask'] & df['sig_upupil']),
                (df['sig_utask'] & df['sig_upupil'])]

    # generate area mask
    if area is not None:
        area = df.area.str.contains(area, regex=True)
    else:
        area = np.ones(df.shape[0]).astype(bool)

    if ax is None:
        plt.figure(figsize=(6, 6))
        ax=plt.gca()
    else:
        plt.sca(ax)
    
    if colors is None:
        colors = color_list

    # need a slope and c to fix the position of line
    if xlim is not None:
        xlim = xlim
        ylim = ylim
        slope = 1
        c = xlim[0]

        x_min = xlim[0]
        x_max = xlim[1]
        y_min, y_max = c, c + slope * (x_max - x_min)
        plt.plot([x_min, x_max], [y_min, y_max], linewidth=0.5, linestyle='--', color='k', dashes=(4,2))

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)

    plt.axvline(0, linestyle='--', linewidth=0.5, color='k', dashes=(4,2))
    plt.axhline(0, linestyle='--', linewidth=0.5, color='k', dashes=(4,2))

    if hue:
        sns.scatterplot(x=df.loc[x_beh_state & area, x_column].tolist(),
                        y=df.loc[y_beh_state & area, y_column].tolist(),
                        s=marker_size, hue=df.loc[x_model & x_beh_state & area, hue],
                        marker=marker, edgecolors='white', linewidth=0.5)

    elif pup_state:
        # plot not significant units
        plt.scatter(x=df.loc[x_model & pup_state & area & sig_list[0], x_column].tolist(),
                    y=df.loc[y_model & x_beh_state & area & sig_list[0], y_column].tolist(),
                    s=marker_size, color=colors[0], marker=marker, edgecolors='white', linewidth=0.5)

        # plot significant state units
        plt.scatter(x=df.loc[x_model & pup_state & area & sig_list[1], x_column].tolist(),
                    y=df.loc[y_model & x_beh_state & area & sig_list[1], y_column].tolist(),
                    s=marker_size, color=colors[1], marker=marker, edgecolors='white', linewidth=0.5)

        # plot significant unique behavior
        plt.scatter(x=df.loc[x_model & pup_state & area & sig_list[2], x_column].tolist(),
                    y=df.loc[y_model & x_beh_state & area & sig_list[2], y_column].tolist(),
                    s=marker_size, color=colors[2], marker=marker, edgecolors='white', linewidth=0.5)

        # plot significant unique pupil
        plt.scatter(x=df.loc[x_model & pup_state & area & sig_list[3], x_column].tolist(),
                    y=df.loc[y_model & x_beh_state & area & sig_list[3], y_column].tolist(),
                    s=marker_size, color=colors[3], marker=marker, edgecolors='white', linewidth=0.5)

        # plot significant unique both
        plt.scatter(x=df.loc[x_model & pup_state & area & sig_list[4], x_column].tolist(),
                    y=df.loc[y_model & x_beh_state & area & sig_list[4], y_column].tolist(),
                    s=marker_size, color=colors[4], marker=marker, edgecolors='white', linewidth=0.5)

    else:
        # iterate: not significant units, sig state, sig u beh, sig u pup, sig u both
        for i, sig in enumerate(sig_list):
            x = df.loc[area & sig, x_model].values
            y = df.loc[area & sig, y_model].values
            out_ix = (x <= xlim[0]) | (x >= xlim[1]) |  (y <= xlim[0]) | (y >= xlim[1])
            x0, y0 = x, y
            x = np.clip(x, xlim[0], xlim[1])
            y = np.clip(y, ylim[0], ylim[1])

            if i == 0:
                s = 75
            else:
                s = 100

            # plot current group
            plt.scatter(x=x, y=y, s=marker_size,
                        color=colors[i], marker=marker, edgecolors='white', linewidth=0.25)
            for _x,_y,_x0,_y0 in zip(x[out_ix], y[out_ix], x0[out_ix], y0[out_ix]):
                plt.text(_x,_y,f'({_x0:.2f},{_y0:.2f})',fontsize=5, color=colors[i])

    # plot a cellid (e.g. TAR010c-27-2 (A1 behavior cell) or TAR010c-06-1 (A1 pupil cell)) with special color

    if type(highlight_cellids) is not dict:
        raise ValueError('highlight_cellids has got to be a dict!')
    else:
        for cellid, color in highlight_cellids.items():
            plt.scatter(x=df.loc[x_model & x_beh_state & area & (df['cellid'] == cellid), x_column].tolist(),
                        y=df.loc[y_model & y_beh_state & area & (df['cellid'] == cellid), y_column].tolist(),
                        s=200, color=color, marker=marker, edgecolors='white', linewidth=0.5)
    ax.set_aspect('equal', 'box')
    nplt.ax_remove_box(ax)

    if save:
        plt.savefig(title + ylabel + xlabel + '.pdf')


def fix_TBD_onBF(df):
    '''This function takes the monster dataframe and adds True to the column 'onBF' when 'ACTIVE_1_tardist' is
    within 0.5 --> half an octave between unit BF and target frequency in ACTIVE 1'''
    BF_TBD = (df['onBF']=='TBD')
    df.loc[BF_TBD, 'onBF'] = df.loc[BF_TBD, 'ACTIVE_1_tardist'].map(lambda x: abs(x)<=0.5)
    return df
