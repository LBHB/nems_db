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
                beh_state,
                area,
                sig_list,
                x_column,
                y_column,
                color_list,
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
        sns.scatterplot(x=df.loc[x_model & beh_state & area, x_column].tolist(),
                        y=df.loc[y_model & beh_state & area, y_column].tolist(),
                        s=200, hue=df.loc[x_model & beh_state & area, hue],
                        marker=marker, edgecolors='white', linewidth=0.5)

    elif pup_state:
        # plot not significant units
        plt.scatter(x=df.loc[x_model & pup_state & area & sig_list[0], x_column].tolist(),
                    y=df.loc[y_model & beh_state & area & sig_list[0], y_column].tolist(),
                    s=150, color=color_list[0], marker=marker, edgecolors='white', linewidth=0.5)

        # plot significant state units
        plt.scatter(x=df.loc[x_model & pup_state & area & sig_list[1], x_column].tolist(),
                    y=df.loc[y_model & beh_state & area & sig_list[1], y_column].tolist(),
                    s=200, color=color_list[1], marker=marker, edgecolors='white', linewidth=0.5)

        # plot significant unique behavior
        plt.scatter(x=df.loc[x_model & pup_state & area & sig_list[2], x_column].tolist(),
                    y=df.loc[y_model & beh_state & area & sig_list[2], y_column].tolist(),
                    s=200, color=color_list[2], marker=marker, edgecolors='white', linewidth=0.5)

        # plot significant unique pupil
        plt.scatter(x=df.loc[x_model & pup_state & area & sig_list[3], x_column].tolist(),
                    y=df.loc[y_model & beh_state & area & sig_list[3], y_column].tolist(),
                    s=200, color=color_list[3], marker=marker, edgecolors='white', linewidth=0.5)

        # plot significant unique both
        plt.scatter(x=df.loc[x_model & pup_state & area & sig_list[4], x_column].tolist(),
                    y=df.loc[y_model & beh_state & area & sig_list[4], y_column].tolist(),
                    s=200, color=color_list[4], marker=marker, edgecolors='white', linewidth=0.5)

    else:
        #import pdb
        #pdb.set_trace()

        # iterate: not significant units, sig state, sig u beh, sig u pup, sig u both
        for i, sig in enumerate(sig_list):
            x = df.loc[x_model & beh_state & area & sig, x_column].values
            y = df.loc[y_model & beh_state & area & sig, y_column].values
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
            plt.scatter(x=df.loc[x_model & beh_state & area & (df['cellid'] == cellid), x_column].tolist(),
                        y=df.loc[y_model & beh_state & area & (df['cellid'] == cellid), y_column].tolist(),
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
