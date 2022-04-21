import nems_lbhb.TwoStim_helpers as ts
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems.db as nd
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import copy
import numpy as np
import SPO_helpers as sp
import nems.preprocessing as preproc
import nems.metrics.api as nmet
import nems.metrics.corrcoef
import copy
import nems.epoch as ep
import scipy.stats as sst
from nems_lbhb.gcmodel.figures.snr import compute_snr
from nems.preprocessing import generate_psth_from_resp
from nems.gui.recording_browser import browse_recording
from nems.signal import concatenate_channels

# fitting stuff
from nems import initializers
from nems.fitters.api import scipy_minimize
import nems

import logging

log = logging.getLogger(__name__)

sb.color_palette
sb.color_palette('colorblind')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sb.color_palette('colorblind'))

def fit_bgfg_model(batch, site):
    cell_df = nd.get_batch_cells(batch)
    cellid = [cell for cell in cell_df['cellid'].tolist() if cell[:7] == site][0]
    fs = 100

    manager = BAPHYExperiment(cellid=cellid, batch=batch)
    options = {'rasterfs': 100,
               'stim': False,
               'resp': True}
    rec = manager.get_recording(**options)
    newrec = ts.generate_psth_from_resp_bgfg(rec, manager)

    rec = newrec.copy()
    rec['resp'] = rec['resp'].rasterize()

    bgfg_psth_signal = rec['psth'].concatenate_channels((rec['psth_bg'], rec['psth_fg']))
    bgfg_psth_signal.name = 'psth_bgfg'
    rec.add_signal(bgfg_psth_signal)

    epoch_regex = '^STIM'
    rec = nems.preprocessing.average_away_epoch_occurrences(rec, epoch_regex=epoch_regex)
    # mask out epochs with "null" in the name
    ep = nems.epoch.epoch_names_matching(rec['psth'].epochs, '^STIM')
    for e in ep:
        if ('null' not in e) and ('0.5' not in e):
            print(e)
            rec = rec.or_mask(e)

    est = rec.copy()
    val = rec.copy()

    outputcount = rec['psth'].shape[0]
    inputcount = outputcount * 2

    insignal = 'psth_bgfg'
    outsignal = 'psth_sp'

    modelspec_name = f'wc.{inputcount}x{outputcount}-lvl.{outputcount}'

    # record some meta data for display and saving
    meta = {'cellid': site,
            'batch': 1,
            'modelname': modelspec_name,
            'recording': est.name
            }
    modelspec = initializers.from_keywords(modelspec_name, meta=meta, input_name=insignal, output_name=outsignal)

    init_weights = np.eye(outputcount,outputcount)
    init_weights = np.concatenate((init_weights,init_weights), axis=1)
    modelspec[0]['phi']['coefficients'] = init_weights/2

    # RUN AN ANALYSIS
    # GOAL: Fit your model to your data, producing the improved modelspecs.
    #       Note that: nems.analysis.* will return a list of modelspecs, sorted
    #       in descending order of how they performed on the fitter's metric.

    # then fit full nonlinear model
    fit_kwargs={'tolerance': 1e-5, 'max_iter': 100000}
    modelspec = nems.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize,
                                            fit_kwargs=fit_kwargs)

    # GENERATE SUMMARY STATISTICS
    print('Generating summary statistics ...')

    # generate predictions
    est, val = nems.analysis.api.generate_prediction(est, val, modelspec)

    # evaluate prediction accuracy
    modelspec = nems.analysis.api.standard_correlation(est, val, modelspec)

    print("Performance: r_fit={0:.3f} r_test={1:.3f}".format(
            modelspec.meta['r_fit'][0][0],
            modelspec.meta['r_test'][0][0]))

    ctx = {'modelspec': modelspec, 'rec': rec, 'val': val, 'est': est}
    xfspec=[]

#import nems.gui.editors as gui
#gui.browse_xform_fit(ctx, xfspec)


    f,ax=plt.subplots(4,1, figsize=(12,6))
    cellnumber=6
    dur=2000
    r=val.apply_mask()
    ax[0].plot(r['pred'].as_continuous()[cellnumber,:dur])
    ax[0].plot(r['psth_sp'].as_continuous()[cellnumber,:dur])
    ax[1].plot(r['psth_fg'].as_continuous()[cellnumber,:dur])
    ax[2].plot(r['psth_bg'].as_continuous()[cellnumber,:dur])
    ax[3].plot(r['mask'].as_continuous()[0,:dur])

    #plt.legend(('pred','actual','mask'))

    plt.figure()
    plt.imshow(modelspec.phi[0]['coefficients'])
    plt.colorbar()

    return modelspec, val, r



# aw = browse_recording(val, ['psth_sp','pred', 'psth_bg', 'psth_fg'], cellid='ARM017a-01-10')



#
# batch=329
# cell_df=nd.get_batch_cells(batch)
# cell_list=cell_df['cellid'].tolist()
# fs=100
#
# cell_list = [cell for cell in cell_list if cell[:3] != 'HOD']
# # cell_list = [cell for cell in cell_list if cell[:7] == 'ARM026b']
# cell_dict = {cell[0:7]:cell for cell in cell_list}
#
# rec_dict = dict()
# for site, cell in cell_dict.items():
#     manager = BAPHYExperiment(cellid=cell, batch=batch)
#     options = {'rasterfs': 100,
#                'stim': False,
#                'resp': True}
#     rec = manager.get_recording(**options)
#     rec_dict[site] = ts.generate_psth_from_resp_bgfg(rec, manager)
#
# cellid='ARM026b'
# rec=rec_dict[cellid].copy()
# rec['resp']=rec['resp'].rasterize()
#
# bgfg_psth_signal = rec['psth'].concatenate_channels((rec['psth_bg'], rec['psth_fg']))
# bgfg_psth_signal.name = 'psth_bgfg'
# rec.add_signal(bgfg_psth_signal)
#
# epoch_regex = '^STIM'
# rec = nems.preprocessing.average_away_epoch_occurrences(rec, epoch_regex=epoch_regex)
# # mask out epochs with "null" in the name
# ep = nems.epoch.epoch_names_matching(rec['psth'].epochs, '^STIM')
# for e in ep:
#     if ('null' not in e) and ('0.5' not in e):
#         print(e)
#         rec = rec.or_mask(e)
#
# est=rec.copy()
# val=rec.copy()
#
# outputcount=rec['psth'].shape[0]
# inputcount=outputcount*2
#
# insignal='psth_bgfg'
# outsignal='psth_sp'
#
# modelspec_name = f'wc.{inputcount}x{outputcount}-lvl.{outputcount}'
#
# # record some meta data for display and saving
# meta = {'cellid': cellid,
#         'batch': 1,
#         'modelname': modelspec_name,
#         'recording': est.name
#         }
# modelspec = initializers.from_keywords(modelspec_name, meta=meta, input_name=insignal, output_name=outsignal)
#
# init_weights = np.eye(outputcount,outputcount)
# init_weights = np.concatenate((init_weights,init_weights), axis=1)
# modelspec[0]['phi']['coefficients'] = init_weights/2
#
# # RUN AN ANALYSIS
#
# # GOAL: Fit your model to your data, producing the improved modelspecs.
# #       Note that: nems.analysis.* will return a list of modelspecs, sorted
# #       in descending order of how they performed on the fitter's metric.
#
# # then fit full nonlinear model
# fit_kwargs={'tolerance': 1e-5, 'max_iter': 100000}
# modelspec = nems.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize,
#                                         fit_kwargs=fit_kwargs)
#
# # GENERATE SUMMARY STATISTICS
# print('Generating summary statistics ...')
#
# # generate predictions
# est, val = nems.analysis.api.generate_prediction(est, val, modelspec)
#
# # evaluate prediction accuracy
# modelspec = nems.analysis.api.standard_correlation(est, val, modelspec)
#
# print("Performance: r_fit={0:.3f} r_test={1:.3f}".format(
#         modelspec.meta['r_fit'][0][0],
#         modelspec.meta['r_test'][0][0]))
#
# ctx = {'modelspec': modelspec, 'rec': rec, 'val': val, 'est': est}
# xfspec=[]
#
# #import nems.gui.editors as gui
# #gui.browse_xform_fit(ctx, xfspec)
#
#
# f,ax=plt.subplots(4,1, figsize=(12,6))
# cellnumber=3
# dur=2000
# r=val.apply_mask()
# ax[0].plot(r['pred'].as_continuous()[cellnumber,:dur])
# ax[0].plot(r['psth_sp'].as_continuous()[cellnumber,:dur])
# ax[1].plot(r['psth_fg'].as_continuous()[cellnumber,:dur])
# ax[2].plot(r['psth_bg'].as_continuous()[cellnumber,:dur])
# ax[3].plot(r['mask'].as_continuous()[0,:dur])
#
# #plt.legend(('pred','actual','mask'))
#
# plt.figure()
# plt.imshow(modelspec.phi[0]['coefficients'])
# plt.colorbar()
#
#
#
# aw = browse_recording(val, ['psth_sp','pred', 'psth_bg', 'psth_fg'], cellid='ARM017a-01-10')