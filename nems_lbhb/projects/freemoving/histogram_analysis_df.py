import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter
from nems0 import db, preprocessing
import nems0.epoch as ep
import nems0.plots.api as nplt
from nems0.utils import smooth
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.plots import plot_waveforms_64D
from nems_lbhb.preprocessing import impute_multi
from nems_lbhb import baphy_io
from nems_lbhb.projects.freemoving.free_tools import compute_d_theta, \
    free_scatter_sum, dlc2dist
from nems import Model
from nems.layers import WeightChannels, FIR, DoubleExponential, LevelShift, ReLU
from nems.layers.base import Layer, Phi, Parameter
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from nems_lbhb.projects.freemoving.decoder_tools import spatial_tc_2d, dlc_to_tcpos, tc_stability, cell_spatial_info, decode2d_forloop, decode2d, dist_occ_hist, dist_tc, trial_2d_tc, target_lickspout_epochs, all_trials_plot, dvt_allcells_plot

runclassid = 132
rasterfs = 100
# sql = f"SELECT distinct stimpath,stimfile from sCellFile where cellid like '{siteid}%%' and runclassid={runclassid}"
#dparm = db.pd_query(sql)

batch = 348
siteids, cellids = db.get_batch_sites(batch)

# sql = f"SELECT distinct left(cellid,7) as siteid,stimpath,stimfile from sCellFile where runclassid={runclassid}"
# dallfiles = db.pd_query(sql)
# siteids = dallfiles['siteid'].unique().tolist()
print('loading dataframe..')
hist_df = pd.read_pickle('/auto/users/wingertj/data/spatial_hist_df.pkl')
df_list = []
df_list.append(hist_df)
siteids.reverse()
for siteid in siteids:
    if siteid in hist_df.siteid.unique():
        print(f"{siteid} already in dataframe")
        continue
    else:
        print(f"{siteid} not in dataframe...generating entry")

        sql = f"SELECT count(cellid) as cellcount,stimpath,stimfile from sCellFile where cellid like '{siteid}%%' AND runclassid={runclassid} AND area='A1' group by stimpath,stimfile"
        dparminfo = db.pd_query(sql)

        parmfile = [r.stimpath+r.stimfile for i,r in dparminfo.iterrows()]
        cellids=None

        # else:
        #     parmfile = ["/auto/data/daq/Prince/PRN015/PRN015a01_a_NTD",
        #                 "/auto/data/daq/Prince/PRN015/PRN015a02_a_NTD"]
        #     cellids = None

        ## load the recording
        try:
            ex = BAPHYExperiment(parmfile=parmfile, cellid=cellids)
            print(ex.experiment, ex.openephys_folder, ex.openephys_tarfile, ex.openephys_tarfile_relpath)

            recache = False

            # load recording
            # rec = ex.get_recording(resp=True, stim=True, stimfmt='gtgram',
            #                        dlc=True, recache=recache, rasterfs=rasterfs,
            #                        dlc_threshold=0.2, fill_invalid='interpolate')
            # rec = ex.get_recording(resp=True, stim=True, stimfmt='gtgram',
            #                        dlc=True, recache=False, rasterfs=rasterfs,
            #                        dlc_threshold=0.2, fill_invalid='interpolate')
            rec = ex.get_recording(resp=True, stim=False,
                                   dlc=True, recache=False, rasterfs=rasterfs,
                                   dlc_threshold=0.2, fill_invalid='interpolate')
        except Exception as e:
            print(f"Problem loading {siteid}")
            continue

        # generate 'dist' signal from dlc signal from approximately lickspout
        rec = dlc2dist(rec, ref_x0y0=[470, 90], smooth_win=5, norm=False, verbose=False)

        # grab A1 units
        try:
            depth_info = baphy_io.get_depth_info(siteid=siteid)
            A1_units = depth_info.loc[depth_info['area']== 'A1'].index.tolist()
            # grab unit names for all units in resp that have depth info in A1
            A1_in_rec = [chan for chan in rec['resp'].chans if chan in A1_units]
            if len(A1_in_rec) == 0:
                A1_in_rec = [chan for chan in rec['resp'].chans if chan[:7]+chan[15:] in A1_units]

            # compute PSTH for repeated stimuli
            epoch_regex = "^STIM_"
            rec['resp'] = rec['resp'].extract_channels(A1_in_rec)
            rec['resp'] = rec['resp'].rasterize()
            rec['psth'] = preprocessing.generate_average_sig(rec['resp'], 'psth', epoch_regex=epoch_regex)
        except:
            raise ValueError("A1 units don't match units in rec?")

        # get tc and xy pos for each cell
        tc, xy, xy_edges = spatial_tc_2d(rec)

        cell_si = cell_spatial_info(rec, xy, shuffle_dlc=False)

        tc_12_cc, tc1, tc2 = tc_stability(rec, tc, xy, shuffle_dlc=False)

        # try to load permutation data if already exists
        try:
            shuffled_df = pd.read_pickle(f"/auto/users/wingertj/data/{siteid}_hist_noise_estimate.pkl")
            print(f"{siteid}_hist_noise_estimate already exists")
        except:
            print(f"{siteid}_hist_noise_estimate does not exist...creating")
            shuffled_data = []
            for shuffle in range(100):

                cell_si_shuffle = cell_spatial_info(rec, xy, shuffle_dlc=True)

                tc_12_cc_shuffle, tc1_shuffle, tc2_shuffle = tc_stability(rec, tc, xy, shuffle_dlc=True)

                cellids = list(cell_si_shuffle.keys())
                shuffle_dict = {'cellids': cellids,
                                'si': [cell_si_shuffle[cellid] for cellid in cellids],
                                'scc': [tc_12_cc_shuffle[cellid] for cellid in cellids],
                                'tc1': [tc1_shuffle[cellid] for cellid in cellids],
                                'tc2': [tc2_shuffle[cellid] for cellid in cellids],
                                'shuffle_idx': [shuffle for cellid in cellids],
                                 }
                shuffled_data.append(pd.DataFrame(shuffle_dict))
            shuffled_df = pd.concat(shuffled_data)
            shuffled_df['sig_threshold_si'] = ''
            shuffled_df['p_si'] = ''
            shuffled_df['sig_threshold_scc'] = ''
            shuffled_df['p_scc'] = ''
            for cell in shuffled_df['cellids']:
                nulldistsi = np.append(shuffled_df.loc[shuffled_df['cellids'] == cell, 'si'].values, cell_si[cell])
                nulldistscc = np.append(shuffled_df.loc[shuffled_df['cellids'] == cell, 'scc'].values, tc_12_cc[cell])
                # two tailed permutation test
                alpha = 0.05
                significance_threshold = 1-alpha
                threshold_si = np.quantile(abs(nulldistsi), q=significance_threshold)
                p_si = sum(abs(nulldistsi) >= abs(cell_si[cell])) / len(nulldistsi)
                threshold_scc = np.quantile(abs(nulldistscc), q=significance_threshold)
                p_scc = sum(abs(nulldistscc) >= abs(tc_12_cc[cell])) / len(nulldistscc)

                shuffled_df.loc[shuffled_df['cellids'] == cell,'sig_threshold_si'] = threshold_si
                shuffled_df.loc[shuffled_df['cellids'] == cell, 'p_si'] = p_si

                shuffled_df.loc[shuffled_df['cellids'] == cell,'sig_threshold_scc'] = threshold_scc
                shuffled_df.loc[shuffled_df['cellids'] == cell, 'p_scc'] = p_scc

            shuffled_df.to_pickle(f"/auto/users/wingertj/data/{siteid}_hist_noise_estimate.pkl")
        # plot 3 most stable and 3 least stable cells
        #
        # sorted_tc_cc = sorted(tc_12_cc.items(), key=lambda x:x[1], reverse=True)
        # sorted_tc_cc = dict(sorted_tc_cc)
        #
        # best_3_ss = list(sorted_tc_cc.keys())[:3]
        # worst_3_ss = list(sorted_tc_cc.keys())[-3:]
        # f, ax = plt.subplots(6,3, layout='tight', figsize=(7,10))
        # for i, cell in enumerate(best_3_ss+worst_3_ss):
        #     ax[i, 0].imshow(tc[cell])
        #     ax[i, 1].imshow(tc1[cell])
        #     ax[i, 1].set_title(f"1st half: {cell}\n scc:{sorted_tc_cc[cell]}")
        #     ax[i, 2].imshow(tc2[cell])
        #     ax[i, 2].set_title(f"2nd half: {cell}\n scc:{sorted_tc_cc[cell]}")
        #
        # # plot 3 cells with highest spatial info and 3 with least
        # sorted_si = sorted(cell_si.items(), key=lambda x:x[1], reverse=True)
        # sorted_si = dict(sorted_si)
        #
        # best_3_si = list(sorted_si.keys())[:3]
        # worst_3_si = list(sorted_si.keys())[-3:]
        # f, ax = plt.subplots(6,3, layout='tight', figsize=(7,10))
        # for i, cell in enumerate(best_3_si+worst_3_si):
        #     ax[i, 0].imshow(tc[cell])
        #     ax[i, 0].set_title(f"cell: {cell}\n si:{sorted_si[cell]}")
        #     ax[i, 1].imshow(tc1[cell])
        #     ax[i, 1].set_title(f"1st half: {cell}\n scc:{sorted_tc_cc[cell]}")
        #     ax[i, 2].imshow(tc2[cell])
        #     ax[i, 2].set_title(f"2nd half: {cell}\n scc:{sorted_tc_cc[cell]}")
        d = { 'siteid': [siteid for i in range(len(rec['resp'].chans))],
              'cellid': [cell for cell in rec['resp'].chans],
              'tc': [tc[cell] for cell in rec['resp'].chans],
              'tc1': [tc1[cell] for cell in rec['resp'].chans],
              'tc2': [tc2[cell] for cell in rec['resp'].chans],
              'si': [cell_si[cell] for cell in rec['resp'].chans],
              'scc': [tc_12_cc[cell] for cell in rec['resp'].chans],
              'layer': [depth_info['layer'][cell[:7]+cell[15:]] for cell in rec['resp'].chans],
              'depth': [depth_info['depth'][cell[:7]+cell[15:]] for cell in rec['resp'].chans],
              'iso': [depth_info['iso'][cell[:7]+cell[15:]] for cell in rec['resp'].chans],
              'scc_threshold': [shuffled_df.loc[shuffled_df['cellids'] == cell,'sig_threshold_scc'].values[0] for cell in rec['resp'].chans],
              'si_threshold': [shuffled_df.loc[shuffled_df['cellids'] == cell, 'sig_threshold_si'].values[0] for cell in
                                rec['resp'].chans],
              'si_p': [shuffled_df.loc[shuffled_df['cellids'] == cell, 'p_si'].values[0] for cell in
                               rec['resp'].chans],
              'scc_p': [shuffled_df.loc[shuffled_df['cellids'] == cell, 'p_scc'].values[0] for cell in
                       rec['resp'].chans],

        }

        df_list.append(pd.DataFrame(d))
df = pd.concat(df_list)
df.to_pickle("/auto/users/wingertj/data/spatial_hist_df.pkl")
bp = []