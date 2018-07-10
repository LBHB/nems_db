#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:59:38 2018

@author: svd
"""

import os
import sys
import pandas as pd

import nems_db.params
import numpy as np
import matplotlib.pyplot as plt
import nems_lbhb.stateplots as stateplots
import nems_db.xform_wrappers as nw
import nems_db.db as nd

import nems.recording as recording
import nems.epoch as ep
import nems.plots.api as nplt

sys.path.append(os.path.abspath('/auto/users/svd/python/scripts/'))


def plot_save_examples(batch, compare, loader, basemodel, fitter, RELOAD=False):

    if batch in [301, 307]:
        area = "AC"
    else:
        area = "IC"

    d = nd.get_batch_cells(batch)
    cellids = list(d['cellid'])

    stats_list = []
    root_path = '/auto/users/svd/projects/pupil-behavior'

    modelset = '{}_{}_{}_{}_{}_{}'.format(compare, area,
                                       batch, loader, basemodel, fitter)
    out_path = '{}/{}/'.format(root_path, modelset)

    if os.access(root_path, os.W_OK) and not(os.path.exists(out_path)):
        os.makedirs(out_path)

    datafile = out_path + 'results.csv'
    plt.close('all')

    if (not RELOAD) and (not os.path.isfile(datafile)):
        RELOAD = True
        print('datafile not found, reloading')

    if RELOAD:
        for cellid in cellids:
            if compare == "pb":
                fh, stats = stateplots.pb_model_plot(
                        cellid, batch, loader=loader, basemodel=basemodel,
                        fitter=fitter)
            elif compare == "ppas":
                fh, stats = stateplots.ppas_model_plot(
                        cellid, batch, loader=loader, basemodel=basemodel,
                        fitter=fitter)
            else:
                fh, stats = stateplots.pp_model_plot(
                        cellid, batch, loader=loader, basemodel=basemodel,
                        fitter=fitter)

            # fh2 = stateplots.pp_model_plot(cellid,batch)
            stats_list.append(stats)
            if os.access(out_path, os.W_OK):
                fh.savefig(out_path+cellid+'.pdf')
                fh.savefig(out_path+cellid+'.png')
            plt.close(fh)

        col_names = ['cellid', 'r_p0b0', 'r_p0b', 'r_pb0', 'r_pb',
                     'e_p0b0', 'e_p0b', 'e_pb0', 'e_pb',
                     'rf_p0b0', 'rf_p0b', 'rf_pb0', 'rf_pb',
                     'r_pup', 'r_beh', 'pup_mod', 'beh_mod',
                     'pup_mod_n', 'beh_mod_n',
                     'pup_mod_beh0', 'beh_mod_pup0',
                     'pup_mod_beh0_n', 'beh_mod_pup0_n']
        df = pd.DataFrame(columns=col_names)

        for stats in stats_list:
            df0 = pd.DataFrame([[stats['cellid'],
                                 stats['r_test'][0], stats['r_test'][1],
                                 stats['r_test'][2], stats['r_test'][3],
                                 stats['se_test'][0], stats['se_test'][1],
                                 stats['se_test'][2], stats['se_test'][3],
                                 stats['r_floor'][0], stats['r_floor'][1],
                                 stats['r_floor'][2], stats['r_floor'][3],
                                 stats['r_test'][3]-stats['r_test'][1],
                                 stats['r_test'][3]-stats['r_test'][2],
                                 stats['pred_mod'][0, 1], stats['pred_mod'][1, 2],
                                 stats['pred_mod_norm'][0, 1],
                                 stats['pred_mod_norm'][1, 2],
                                 stats['pred_mod_full'][0, 1],
                                 stats['pred_mod_full'][1, 2],
                                 stats['pred_mod_full_norm'][0, 1],
                                 stats['pred_mod_full_norm'][1, 2]
                                 ]], columns=col_names)
            df = df.append(df0)
        df.set_index(['cellid'], inplace=True)
        if os.access(out_path, os.W_OK):
            df.to_csv(datafile)
    else:
        # load cached dataframe
        df = pd.read_csv(datafile, index_col=0)

    sig_mod = list(df['r_pb']-df['e_pb'] > df['r_p0b0'] + df['e_p0b0'])
    if compare == "pb":
        alabel="active"
    elif compare == "ppas":
        alabel="each passive"
    else:
        alabel="pre/post"

    mi_bounds = [-0.4, 0.4]

    fh1 = stateplots.beta_comp(df['r_pup'], df['r_beh'], n1='pupil', n2=alabel,
                               title=modelset+' unique pred', hist_range=[-0.1, 0.1],
                               highlight=sig_mod)
    fh2 = stateplots.beta_comp(df['pup_mod_n'], df['beh_mod'],
                               n1='pupil', n2=alabel,
                               title=modelset+' mod index', hist_range=mi_bounds,
                               highlight=sig_mod)
    fh3 = stateplots.beta_comp(df['beh_mod_pup0'], df['beh_mod'],
                               n1=alabel+'-nopup', n2=alabel,
                               title=modelset+' unique mod', hist_range=mi_bounds,
                               highlight=sig_mod)

    #fh4 = stateplots.beta_comp(df['r_beh'], df['beh_mod'], n1='pred', n2='mod',
    #                           title='behavior', hist_range=[-0.4, 0.4])
    #fh5 = stateplots.beta_comp(df['r_pup'], df['pup_mod'], n1='pred', n2='mod',
    #                           title='pupil', hist_range=[-0.1, 0.1])

    if os.access(out_path, os.W_OK):
        fh1.savefig(out_path+'summary_pred.pdf')
        fh2.savefig(out_path+'summary_mod.pdf')
        fh3.savefig(out_path+'summary_mod_ctl.pdf')

# BEGIN main code

# User parameters:
RELOAD = False
loader = "psth.fs20"
fitter = "jk.nf10-init.st-basic"

batches = [301, 303, 307, 309]
#basemodels = ["psthfr_sdexp.S", "psthfr_stategain.S",
#              "psthfr.s_sdexp.S", "psthfr.s_stategain.S"]
basemodels = ["psthfr_sdexp.S", "psthfr_stategain.S"]
comparisons = ["pb", "ppas"]

for batch in batches:
    for basemodel in basemodels:
        for compare in comparisons:
            print("plot_save_examples({},{},{},{},{})".format(
                    batch, compare, loader, basemodel, fitter))
            plot_save_examples(batch, compare, loader, basemodel,
                               fitter, RELOAD)