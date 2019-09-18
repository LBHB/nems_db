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
from nems_lbhb.pupil_behavior_scripts.mod_per_state import beh_only_plot
from nems_lbhb.stateplots import model_per_time_wrapper, beta_comp
import common



def donut_plot(area, unit_list, colors, savefigure=False):
    white_circle=plt.Circle((0,0), 0.7, color='white')
    plt.axis('equal')
    plt.pie(unit_list, colors=colors, labels=unit_list)
    p=plt.gcf()
    p.gca().add_artist(white_circle)
    plt.title(area)
    if savefigure:
        plt.savefig(area + '_donut.pdf')


# SPECIFY pup+beh models
state_list = ['st.pup0.beh0','st.pup0.beh','st.pup.beh0','st.pup.beh']
basemodel = "-ref-psthfr.s_sdexp.S"

#df = pd.read_csv('pup_beh_processed.csv')
df = pd.read_csv('pup_beh_processed'+basemodel+'.csv')

# creating subdf with only rows that match conditions
is_active = (df['state_chan'] == 'active')
is_pupil = (df['state_chan'] == 'pupil')
full_model = (df['state_sig'] == 'st.pup.beh')
null_model = (df['state_sig'] == 'st.pup0.beh0')
part_beh_model = (df['state_sig'] == 'st.pup0.beh')
part_pup_model = (df['state_sig'] == 'st.pup.beh0')

xsubset = df.cellid.str.startswith('AMT018') | df.cellid.str.startswith('AMT020')
#xsubset = df.cellid.str.startswith('AMT020')
#xsubset = df.cellid.str.startswith('XXXXXX')

# creating list of booleans to mask A1, IC, onBF and offBF out of big df
A1 = (df['area']=='A1') & ~xsubset
ICC = df['area']=='ICC'
ICX = df['area']=='ICX'
onBF = df['onBF']==True
offBF = df['onBF']==False
SU = df['SU']==True
sig_any = df['sig_any']
sig_ubeh = df['sig_ubeh'] & sig_any
sig_upup = df['sig_upup'] & sig_any
sig_both = sig_ubeh & sig_upup
sig_state = df['sig_state'] & sig_any
sig_obeh = df['sig_obeh'] & sig_any
sig_oubeh = sig_ubeh | sig_obeh

print((df.loc[full_model & is_active & A1 & sig_oubeh, 'MIbeh_only']).median())
print((df.loc[full_model & is_active & A1 & sig_oubeh, 'MIbeh_unique']).median())

if 0:
    a = df.loc[full_model & is_active & A1 & sig_oubeh, 'MIbeh_only']
    b = df.loc[full_model & is_active & A1 & sig_oubeh, 'MIbeh_unique']
    plt.hist(a, bins=30, color=common.color_b, histtype=u'step')
    plt.hist(b, bins=15, color=common.color_p, histtype=u'step')
    plt.xlabel('MI')
    plt.ylabel('count')
    plt.title('A1')
    plt.axvline(0, linestyle='--', linewidth=0.5, color='k')
    #plt.savefig('A1_MIcomp.pdf')

print('A1 had {} units for which behavior alone sign modulated activity'.format(len(df.loc[full_model & is_active & A1 & sig_obeh])))
print('A1 had {} units for which behavior unique sign modulated activity'.format(len(df.loc[full_model & is_active & A1 & sig_ubeh])))
print('A1 had {} units for which behavior either only or unique sign modulated activity'.format(len(df.loc[full_model & is_active & A1 &
                                                                                                           sig_oubeh])))
print('IC had {} units for which behavior alone sign modulated activity'.format(len(df.loc[full_model & is_active & (ICC | ICX) & sig_obeh])))
print('IC had {} units for which behavior unique sign modulated activity'.format(len(df.loc[full_model & is_active & (ICC | ICX) & sig_ubeh])))

print((df.loc[full_model & is_active & (ICC | ICX) & sig_oubeh, 'MIbeh_only']).median())
print((df.loc[full_model & is_active & (ICC | ICX) & sig_oubeh, 'MIbeh_unique']).median())

if 0:
    plt.hist(df.loc[full_model & is_active & (ICC | ICX) & sig_oubeh, 'MIbeh_only'], color=common.color_b, bins=23, histtype=u'step')
    plt.hist(df.loc[full_model & is_active & (ICC | ICX) & sig_oubeh, 'MIbeh_unique'], color=common.color_p, bins=12, histtype=u'step')
    plt.axvline(0, linestyle='--', linewidth=0.5, color='k')
    plt.xlabel('MI')
    plt.ylabel('count')
    plt.title('ICC & ICX')
    #plt.savefig('IC_MIcomp.pdf')

da1 = df[(df['state_chan']=='active') & A1]
dp = pd.pivot_table(da1, index='cellid',columns='state_sig',values=['r','r_se'])

# Fig 3A

A1_n_sig_both = len(df[~xsubset & full_model & is_active & A1 & sig_both])
A1_n_sig_ubeh = len(df[~xsubset & full_model & is_active & A1 & sig_ubeh]) - A1_n_sig_both
A1_n_sig_upup = len(df[~xsubset & full_model & is_active & A1 & sig_upup]) - A1_n_sig_both
A1_n_sig_state = len(df[~xsubset & full_model & is_active & A1 & sig_state])
A1_n_sig_either = A1_n_sig_state - (A1_n_sig_both + A1_n_sig_ubeh + A1_n_sig_upup)

A1_n_total = len(df[~xsubset & full_model & is_active & A1 & sig_any])
A1_n_not_sig = A1_n_total - (A1_n_sig_state)

A1_units = [A1_n_sig_ubeh, A1_n_sig_upup, A1_n_sig_both, A1_n_sig_either, A1_n_not_sig]

if 0:
    A1x_n_sig_both = len(df[cellsubset & full_model & is_active & A1 & sig_both])
    A1x_n_sig_ubeh = len(df[cellsubset & full_model & is_active & A1 & sig_ubeh]) - A1x_n_sig_both
    A1x_n_sig_upup = len(df[cellsubset & full_model & is_active & A1 & sig_upup]) - A1x_n_sig_both
    A1x_n_sig_state = len(df[cellsubset & full_model & is_active & A1 & sig_state])
    A1x_n_sig_either = A1x_n_sig_state - (A1x_n_sig_both + A1x_n_sig_ubeh + A1x_n_sig_upup)

    A1x_n_total = len(df[cellsubset & full_model & is_active & A1 & sig_any])
    A1x_n_not_sig = A1x_n_total - (A1x_n_sig_state)

    A1x_units = [A1x_n_sig_ubeh, A1x_n_sig_upup, A1x_n_sig_both, A1x_n_sig_either, A1x_n_not_sig]

# IC

IC_n_sig_both = len(df[full_model & is_active & (ICC | ICX) & sig_both])
IC_n_sig_ubeh = len(df[full_model & is_active & (ICC | ICX) & sig_ubeh]) - IC_n_sig_both
IC_n_sig_upup = len(df[full_model & is_active & (ICC | ICX) & sig_upup]) - IC_n_sig_both
IC_n_sig_state = len(df[full_model & is_active & (ICC | ICX) & sig_state])
IC_n_sig_either = IC_n_sig_state - (IC_n_sig_both + IC_n_sig_ubeh + IC_n_sig_upup)

IC_n_total = len(df[full_model & is_active & (ICC | ICX) & sig_any])
IC_n_not_sig = IC_n_total - (IC_n_sig_state)

IC_units = [IC_n_sig_ubeh, IC_n_sig_upup, IC_n_sig_both, IC_n_sig_either, IC_n_not_sig]

ICc_n_sig_both = len(df[full_model & is_active & (ICC) & sig_both])
ICc_n_sig_ubeh = len(df[full_model & is_active & (ICC) & sig_ubeh]) - ICc_n_sig_both
ICc_n_sig_upup = len(df[full_model & is_active & (ICC) & sig_upup]) - ICc_n_sig_both
ICc_n_sig_state = len(df[full_model & is_active & (ICC) & sig_state])
ICc_n_sig_either = ICc_n_sig_state - (ICc_n_sig_both + ICc_n_sig_ubeh + ICc_n_sig_upup)
ICc_n_total = len(df[full_model & is_active & (ICC) & sig_any])
ICc_n_not_sig = ICc_n_total - (ICc_n_sig_state)
ICc_units = [ICc_n_sig_ubeh, ICc_n_sig_upup, ICc_n_sig_both, ICc_n_sig_either, ICc_n_not_sig]

ICx_n_sig_both = len(df[full_model & is_active & (ICX) & sig_both])
ICx_n_sig_ubeh = len(df[full_model & is_active & (ICX) & sig_ubeh]) - ICx_n_sig_both
ICx_n_sig_upup = len(df[full_model & is_active & (ICX) & sig_upup]) - ICx_n_sig_both
ICx_n_sig_state = len(df[full_model & is_active & (ICX) & sig_state])
ICx_n_sig_either = ICx_n_sig_state - (ICx_n_sig_both + ICx_n_sig_ubeh + ICx_n_sig_upup)
ICx_n_total = len(df[full_model & is_active & (ICX) & sig_any])
ICx_n_not_sig = ICx_n_total - (ICx_n_sig_state)
ICx_units = [ICx_n_sig_ubeh, ICx_n_sig_upup, ICx_n_sig_both, ICx_n_sig_either, ICx_n_not_sig]

colors = [common.color_b, common.color_p, common.color_both, common.color_either, common.color_ns]

if 0:
    # ICC/ICx breakdown - no difference, though N is kinda small for ICc
    fh, axs = plt.subplots(1, 2, figsize=(8,4))
    plt.sca(axs[0])
    donut_plot('ICc', ICc_units, colors, savefigure=False)

    plt.sca(axs[1])
    donut_plot('ICx', ICx_units, colors, savefigure=False)

# Figure 3A
fh, axs = plt.subplots(3, 2, figsize=(8,12))

plt.sca(axs[0,0])
donut_plot('A1', A1_units, colors, savefigure=False)

plt.sca(axs[0,1])
donut_plot('IC', IC_units, colors, savefigure=False)

# Figure 3B

# A1 with colored units according to model significance
common.scat_states(df, x_model=null_model,
            y_model=full_model,
            x_beh_state=is_active,
            area=A1,
            sig_list=[~sig_state, sig_state, sig_ubeh, sig_upup, sig_both],
            x_column='R2',
            y_column='R2',
            color_list=common.color_list,
            #highlight_cellids={'TAR010c-27-2':'red', 'TAR010c-06-1':'blue'},
            save=False,
            xlim=(0,1),
            ylim=(0,1),
            xlabel='state-independent R2',
            ylabel='state-dependent R2',
            title='A1',
            ax=axs[1,0])

# All IC with colored untis according to model significance
common.scat_states(df, x_model=null_model,
            y_model=full_model,
            x_beh_state=is_active,
            area=(ICC | ICX),
            sig_list=[~sig_state, sig_state, sig_ubeh, sig_upup, sig_both],
            x_column='R2',
            y_column='R2',
            color_list=common.color_list,
            save=False,
            xlim=(0,1),
            ylim=(0,1),
            xlabel='state-independent R2',
            ylabel='state-dependent R2',
            title='ICC & ICX',
                   ax=axs[1,1])

# Figure 3C
# A1
common.scat_states(df, x_model=full_model,
            y_model=full_model,
            x_beh_state=is_active,
            area=A1,
            sig_list=[~sig_state, sig_state, sig_ubeh, sig_upup,sig_both],
            x_column='R2pup_unique',
            y_column='R2beh_unique',
            color_list=common.color_list,
            #highlight_cellids={'TAR010c-27-2':'red', 'TAR010c-06-1':'blue'},
            save=False,
            xlabel='R2 pupil unique (task regressed out)',
            ylabel='R2 task unique (pupil regressed out)',
            title='A1',
            xlim=(-0.05,0.2),
            ylim=(-0.05,0.2),
                   ax=axs[2,0])

# IC and ICX together
common.scat_states(df, x_model=full_model,
            y_model=full_model,
            x_beh_state=is_active,
            area=(ICC | ICX),
            sig_list=[~sig_state, sig_state, sig_ubeh, sig_upup,sig_both],
            x_column='R2pup_unique',
            y_column='R2beh_unique',
            color_list=common.color_list,
            save=False,
            xlabel='R2 pupil unique (task regressed out)',
            ylabel='R2 task unique (pupil regressed out)',
            title='IC',
            xlim=(-0.05,0.2),
            ylim=(-0.05,0.2),
            ax=axs[2,1])


