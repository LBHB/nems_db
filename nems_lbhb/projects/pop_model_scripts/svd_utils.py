import logging

import pandas as pd
import numpy as np

import nems
import nems0.epoch as ep
import nems0.db as nd
import nems_lbhb.xform_wrappers as xwrap

log = logging.getLogger(__name__)


load_string_pop = "ozgf.fs100.ch18.pop-loadpop-norm.l1-popev"
fit_string_pop = "tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4"

load_string_single = "ozgf.fs100.ch18-ld-norm.l1-sev"
fit_string_single = 'prefit.f-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4'  # maybe add ".et4" ?

# POP_MODELS: round 1, fit using cellid="NAT4" on exacloud
POP_MODELS = [
    f"{load_string_pop}_conv2d.10.8x3.rep3-wcn.90-relu.90-wc.90xR-lvl.R-dexp.R_{fit_string_pop}",  #c2d
    f"{load_string_pop}_wc.18x100.g-fir.1x25x100-relu.100.f-wc.100x120-relu.120.f-wc.120xR-lvl.R-dexp.R_{fit_string_pop}", # c1d
    f"{load_string_pop}_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_{fit_string_pop}", # c1dx2+d
    f"{load_string_pop}_wc.18x120.g-fir.1x25x120-wc.120xR-lvl.R-dexp.R_{fit_string_pop}", # LN_pop
    f'{load_string_single}_wc.18x6.g-fir.1x25x6-relu.6.f-wc.6x1-lvl.1-dexp.1_tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4', # dnn-single
]

SIG_TEST_MODELS = [
    f"{load_string_single}_conv2d.10.8x3.rep3-wcn.90-relu.90-wc.90xR-lvl.R-dexp.R_{fit_string_single}",  #c2d
    f"{load_string_single}_wc.18x100.g-fir.1x25x100-relu.100.f-wc.100x120-relu.120.f-wc.120xR-lvl.R-dexp.R_{fit_string_single}", # c1d
    f"{load_string_single}_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_{fit_string_single}", # c1dx2+d
    f"{load_string_single}_wc.18x120.g-fir.1x25x120-wc.120xR-lvl.R-dexp.R_{fit_string_single}", # LN_pop
    f'{load_string_single}_wc.18x6.g-fir.1x25x6-relu.6.f-wc.6x1-lvl.1-dexp.1_prefit.m-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4', # dnn-single
]

modelnames = POP_MODELS

shortnames=['conv2d','conv1d','conv1dx2','ln-pop', 'dnn-sing']
shortnamesp=[s+"_p" for s in shortnames]



fit_string_nopre = 'tfinit.n.lr1e3.et3.rb10.es20-newtf.n.lr1e4'
fit_string_dnn = 'prefit.m-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4'


# For correlation histograms
EQUIVALENCE_MODELS_SINGLE = [
    f"{load_string_single}_conv2d.10.8x3.rep3-wcn.90-relu.90-wc.90xR-lvl.R-dexp.R_{fit_string_single}",  #c2d
    f"{load_string_single}_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_{fit_string_single}", # c1dx2+d
    f"{load_string_single}_wc.18x120.g-fir.1x25x120-wc.120xR-lvl.R-dexp.R_{fit_string_single}",  # LN_pop
]
EQUIVALENCE_MODELS_POP = [
    f"{load_string_pop}_conv2d.10.8x3.rep3-wcn.90-relu.90-wc.90xR-lvl.R-dexp.R_{fit_string_pop}",  #c2d
    f"{load_string_pop}_wc.18x70.g-fir.1x15x70-relu.70.f-wc.70x80-fir.1x10x80-relu.80.f-wc.80x100-relu.100-wc.100xR-lvl.R-dexp.R_{fit_string_pop}", # c1dx2+d
    f"{load_string_pop}_wc.18x120.g-fir.1x25x120-wc.120xR-lvl.R-dexp.R_{fit_string_pop}",  # LN_pop
]

HELDOUT = [m.replace("prefit.f", "prefit.hs") for m in SIG_TEST_MODELS[:-1]]  # leave out dnn, which is last
dnn_single_held = SIG_TEST_MODELS[-1].replace('prefit.m', 'prefit.h')
HELDOUT.append(dnn_single_held)
MATCHED = [m.replace("prefit.f", "prefit.hm") for m in SIG_TEST_MODELS[:-1]]
dnn_single_matched = SIG_TEST_MODELS[-1]
MATCHED.append(dnn_single_matched)


# TODO: make sure these match

# DNN_SINGLE_MODELS = [
#         f"{load_string_single}_wc.18x12.g-fir.1x25x12-relu.12.f-wc.12x1-lvl.1-dexp.1_prefit.h-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4",
#         f"{load_string_single}_wc.18x12.g-fir.1x25x12-relu.12.f-wc.12x1-lvl.1-dexp.1_prefit.m-tfinit.n.lr1e3.et3.es20-newtf.n.lr1e4"
# ]


fit_string2 = "tfinit.n.lr1e3.et3.rb10.es20.v-newtf.n.lr1e4.es20.v"
fit_string3 = "tfinit.n.lr1e3.et3.rb10.es20.L2-newtf.n.lr1e4.es20.L2"

PLOT_STAT = 'r_ceiling'

DOT_COLORS = {'conv2d': 'darkgreen', 'LN': 'black', 'conv1d': 'lightblue', #'conv1dx2': 'purple',
              'conv1dx2+d': 'purple', 'conv1dx2+dd': 'yellow', 'conv1dx2+d2': 'magenta', 'conv1dx2+d3': 'gray',
              'LN_pop': 'orange', 'dnn1': 'lightgreen', 'dnn1_single': 'lightgreen', 'c1dx2-stp': 'red', #'STP': 'lightblue',
              'LN_2d': 'purple',
              'c1d2_input': 'blue',
              'c1d2_tiny': 'blue',
              'c1d2_output': 'blue',
              'c1d2_25h20': 'blue',
              'c1d2_25h160': 'blue',
              'c2d_num_filters': 'darkgreen',
              'c2d_filter_length': 'darkgreen',
              'c2d_filter_reps': 'darkgreen',
              'c2d_10f': 'darkgreen',
              'conv2d_v': 'darkgreen',
              'conv2d_L2': 'darkgreen'
              }

DOT_MARKERS = {#'conv1dx2': '^',
               'conv2d': 's', 'LN_pop': 'o', 'conv1d': 'o',
               'LN':'.', 'dnn1': 'v', 'dnn1_single': 'v', 'c1dx2-stp': '*', #'STP': 'x',
               'conv1dx2+d': '+', 'LN_2d': 'x',
               'c1d2_input': '^',
               'c1d2_tiny': '>',
               'c1d2_output': 'v',
               'c1d2_25h20': '<',
               'c1d2_25h160': 'o',
               'c2d_num_filters': '^',
               'c2d_filter_length': '>',
               'c2d_filter_reps': 'v',
               'c2d_10f': '<',
               'conv2d_v': 'o',
               'conv2d_L2': '+',
            }


# CELL_COUNT_TEST = []
# for c in [50, 100]:
#     CELL_COUNT_TEST.extend([
#         f"{load_string}-mc.{c}_conv2d.4.8x3.rep3-wcn.40-relu.40-wc.40xR-lvl.R-dexp.R_{fit_string}",
#         f"{load_string}-mc.{c}_wc.18x80.g-fir.1x25x80-relu.80.f-wc.80x100-relu.100.f-wc.100xR-lvl.R-dexp.R_{fit_string}", # c1d
#         f"{load_string}-mc.{c}_wc.18x30.g-fir.1x15x30-relu.30.f-wc.30x60-fir.1x10x60-relu.60.f-wc.60x80-relu.80-wc.80xR-lvl.R-dexp.R_{fit_string}", # c1dx2+d
#     ])

NAT4_A1_SITES = [
 'ARM029a', 'ARM030a', 'ARM031a',
 'ARM032a', 'ARM033a',
 'CRD016d', 'CRD017c',
 'DRX006b.e1:64', 'DRX006b.e65:128',
 'DRX007a.e1:64', 'DRX007a.e65:128',
 'DRX008b.e1:64', 'DRX008b.e65:128',
]

NAT4_PEG_SITES = [
    'ARM017a', 'ARM018a', 'ARM019a', 'ARM021b', 'ARM022b', 'ARM023a',
    'ARM024a', 'ARM025a', 'ARM026b', 'ARM027a', 'ARM028b'
    ]

# Build modelnames
MODELGROUPS = {}
POP_MODELGROUPS = {}

# LN ###################################################################################################################
params = [1, 2, 3, 4, 6, 8, 9, 10]#, 12, 14]
MODELGROUPS['LN'] = [f'{load_string_single}_wc.18x{p}.g-fir.{p}x25-lvl.1-dexp.1_{fit_string_nopre}' for p in params]
POP_MODELGROUPS['LN'] = [f'{load_string_single}_wc.18x{p}.g-fir.{p}x25-lvl.1-dexp.1_{fit_string_nopre}' for p in params]


# LN_pop ###############################################################################################################
params = [4, 6, 10, 14, 30, 42, 60, 80, 100, 120, 150, 175, 200, 250, 300]
MODELGROUPS['LN_pop'] = [f'{load_string_single}_wc.18x{p}.g-fir.1x25x{p}-wc.{p}xR-lvl.R-dexp.R_{fit_string_single}' for p in params]
POP_MODELGROUPS['LN_pop'] = [f'{load_string_pop}_wc.18x{p}.g-fir.1x25x{p}-wc.{p}xR-lvl.R-dexp.R_{fit_string_pop}' for p in params]


# conv1d ###############################################################################################################
L1_L2 = [
    (5, 10), (10, 10), (10, 20), (20, 30),
    (30, 40), (30, 50), (40, 50), (50, 60),
    (60, 80), (80, 100), (100, 120), (120, 140),
    (140, 160), (170, 200), (200, 250), (230, 300)
]
MODELGROUPS['conv1d'] = [
    f"{load_string_single}_wc.18x{layer1}.g-fir.1x25x{layer1}-relu.{layer1}.f-"
    + f"wc.{layer1}x{layer2}-relu.{layer2}.f-wc.{layer2}xR-lvl.R-dexp.R_{fit_string_single}"
    for layer1, layer2 in L1_L2
]
POP_MODELGROUPS['conv1d'] = [
    f"{load_string_pop}_wc.18x{layer1}.g-fir.1x25x{layer1}-relu.{layer1}.f-"
    + f"wc.{layer1}x{layer2}-relu.{layer2}.f-wc.{layer2}xR-lvl.R-dexp.R_{fit_string_pop}"
    for layer1, layer2 in L1_L2
]


# conv1dx2+d ###########################################################################################################
L1_L2_L3 = [
    (5, 10, 20), (10, 10, 20), (10, 20, 30),
    (20, 20, 40), (20, 40, 60), (30, 60, 80), (50, 70, 90), (70, 80, 100),
    (70, 90, 120), (80, 100, 140), (90, 120, 160), (100, 140, 180),
    (120, 160, 220), (150, 200, 250), #(180, 250, 300)
]
MODELGROUPS['conv1dx2+d'] = [
    f"{load_string_single}_wc.18x{layer1}.g-fir.1x15x{layer1}-relu.{layer1}.f-wc.{layer1}x{layer2}-fir.1x10x{layer2}-"
    + f"relu.{layer2}.f-wc.{layer2}x{layer3}-relu.{layer3}-wc.{layer3}xR-lvl.R-dexp.R_{fit_string_single}"
    for layer1, layer2, layer3 in L1_L2_L3
]
POP_MODELGROUPS['conv1dx2+d'] = [
    f"{load_string_pop}_wc.18x{layer1}.g-fir.1x15x{layer1}-relu.{layer1}.f-wc.{layer1}x{layer2}-fir.1x10x{layer2}-"
    + f"relu.{layer2}.f-wc.{layer2}x{layer3}-relu.{layer3}-wc.{layer3}xR-lvl.R-dexp.R_{fit_string_pop}"
    for layer1, layer2, layer3 in L1_L2_L3
]



# dnn1_single ##########################################################################################################
params = [4, 6, 9, 12, 15, 18]
MODELGROUPS['dnn1_single'] = [
    f"{load_string_single}_wc.18x{p}.g-fir.1x25x{p}-relu.{p}.f-wc.{p}x1-lvl.1-dexp.1_{fit_string_dnn}"
    for p in params
]
POP_MODELGROUPS['dnn1_single'] = [
    f"{load_string_single}_wc.18x{p}.g-fir.1x25x{p}-relu.{p}.f-wc.{p}x1-lvl.1-dexp.1_{fit_string_pop}"
    for p in params
]
# _single flag tells pareto plot function not to divide by cells per site
# the double conv layer models were doing substantially worse per parameter for dnn1, so leaving them out for now.

# try fixing with 10 filters
dense_counts = [4, 8, 12, 20, 40, 50, 70, 90, 110, 130, 150, 175, 200, 250, 300]#, 400]
MODELGROUPS['conv2d'] = [
    f"{load_string_single}_conv2d.10.8x3.rep3-wcn.{dense}-"
    + f"relu.{dense}-wc.{dense}xR-lvl.R-dexp.R_{fit_string_single}"
    for dense in dense_counts
]
POP_MODELGROUPS['conv2d'] = [
    f"{load_string_pop}_conv2d.10.8x3.rep3-wcn.{dense}-"
    + f"relu.{dense}-wc.{dense}xR-lvl.R-dexp.R_{fit_string_pop}"
    for dense in dense_counts
]



def get_significant_cells(batch, models, as_list=False):

    df_r = nd.batch_comp(batch, models, stat='r_test')
    df_r.dropna(axis=0, how='any', inplace=True)
    df_r.sort_index(inplace=True)
    df_e = nd.batch_comp(batch, models, stat='se_test')
    df_e.dropna(axis=0, how='any', inplace=True)
    df_e.sort_index(inplace=True)
    df_f = nd.batch_comp(batch, models, stat='r_floor')
    df_f.dropna(axis=0, how='any', inplace=True)
    df_f.sort_index(inplace=True)

    masks = []
    for m in models:
        mask1 = df_r[m] > df_e[m] * 2
        mask2 = df_r[m] > df_f[m]
        mask = mask1 & mask2
        masks.append(mask)

    all_significant = masks[0]
    for m in masks[1:]:
        all_significant &= m

    if as_list:
        all_significant = all_significant[all_significant].index.values.tolist()

    return all_significant


def snr_by_batch(batch, loadkey, save_path=None, load_path=None, frac_total=True, rec=None, siteids=None):
    snrs = []
    cells = []
    if load_path is None:

        if rec is None:
            if siteids is None:
                cellids = nd.get_batch_cells(batch, as_list=True)
                siteids = list(set([c.split('-')[0] for c in cellids]))

            for site in siteids:
                rec_path = xwrap.generate_recording_uri(site, batch, loadkey=loadkey)
                rec = nems0.recording.load_recording(rec_path)
                est, val = rec.split_using_epoch_occurrence_counts('^STIM_')
                for cellid in rec['resp'].chans:
                    resp = val.apply_mask()['resp'].extract_channels([cellid])
                    snr = compute_snr(resp, frac_total=frac_total)
                    snrs.append(snr)
                    cells.append(cellid)

        else:
            if isinstance(rec, str):
                rec = nems0.recording.load_recording(rec)
            cellids = rec['resp'].chans
            est, val = rec.split_using_epoch_occurrence_counts('^STIM_')
            for cellid in cellids:
                log.info("computing SNR for cell: %s" % cellid)
                resp = val.apply_mask()['resp'].extract_channels([cellid])
                snr = compute_snr(resp, frac_total=frac_total)
                snrs.append(snr)
            cells = cellids

        results = {'cellid': cells, 'snr': snrs}
        df = pd.DataFrame.from_dict(results)
        df.dropna(inplace=True)
        df.set_index('cellid', inplace=True)
        if save_path is not None:
            df.to_pickle(save_path)

    else:
        df = pd.read_pickle(load_path)

    return df


def compute_snr(resp, frac_total=True):
    epochs = resp.epochs
    stim_epochs = ep.epoch_names_matching(epochs, 'STIM_')
    resp_dict = resp.extract_epochs(stim_epochs)

    per_stim_snrs = []
    for stim, resp in resp_dict.items():
        resp = resp.squeeze()
        if resp.ndim == 1:
            # Only one stim rep, have to add back in axis for number of reps
            resp = np.expand_dims(resp, 0)
        products = np.dot(resp, resp.T)
        per_rep_snrs = []

        for i, _ in enumerate(resp):
            total_power = products[i,i]
            signal_powers = np.delete(products[i], i)
            if frac_total:
                rep_snr = np.nanmean(signal_powers)/total_power
            else:
                rep_snr = np.nanmean(signal_powers/(total_power-signal_powers))

            per_rep_snrs.append(rep_snr)
        per_stim_snrs.append(np.nanmean(per_rep_snrs))

    # if np.sum(np.isnan(per_stim_snrs)) == len(per_stim_snrs):
    #     import pdb; pdb.set_trace()

    return np.nanmean(per_stim_snrs)


def get_rceiling_correction(batch):
    LN_model = MODELGROUPS['LN'][3]

    rceiling_ratios = []
    significant_cells = get_significant_cells(batch, SIG_TEST_MODELS, as_list=True)
    rtest = nd.batch_comp(batch, [LN_model], cellids=significant_cells, stat='r_test')
    rceiling = nd.batch_comp(batch, [LN_model], cellids=significant_cells, stat='r_ceiling')

    rceiling_ratios = rceiling[LN_model] / rtest[LN_model]
    rceiling_ratios.loc[rceiling_ratios < 1] = 1

    return rceiling_ratios
