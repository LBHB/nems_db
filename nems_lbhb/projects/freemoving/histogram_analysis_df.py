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
import nems_lbhb.projects.freemoving.decoder_tools as dec

runclassid = 132
rasterfs = 100
# sql = f"SELECT distinct stimpath,stimfile from sCellFile where cellid like '{siteid}%%' and runclassid={runclassid}"
#dparm = db.pd_query(sql)

batch = 348
siteids, cellids = db.get_batch_sites(batch)

# sql = f"SELECT distinct left(cellid,7) as siteid,stimpath,stimfile from sCellFile where runclassid={runclassid}"
# dallfiles = db.pd_query(sql)
# siteids = dallfiles['siteid'].unique().tolist()
print('Generating dataframe..')
hist_df = pd.DataFrame()
df_list = []
siteids.reverse()
for siteid in siteids:
    print(f"{siteid} not in dataframe...generating entry")

    sql = f"SELECT count(cellid) as cellcount,stimpath,stimfile from sCellFile where cellid like '{siteid}%%' AND runclassid={runclassid} AND area='A1' group by stimpath,stimfile"
    dparminfo = db.pd_query(sql)

    parmfile = [r.stimpath+r.stimfile for i,r in dparminfo.iterrows()]
    cellids=None

    ## load the recording
    try:
        ex = BAPHYExperiment(parmfile=parmfile, cellid=cellids)
        print(ex.experiment, ex.openephys_folder, ex.openephys_tarfile, ex.openephys_tarfile_relpath)

        recache = False
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
    tc, xy, stab = dec.spatial_tc_jackknifed(rec)

    # try to load permutation data if already exists
    try:
        shuffled_df = pd.read_pickle(f"/auto/users/wingertj/data/{siteid}_hist_noise_estimate.pkl")
        print(f"{siteid}_hist_noise_estimate already exists")
    except:
        print(f"{siteid}_hist_noise_estimate does not exist...creating")
        shuffled_data = []
        for shuffle in range(100):

            cell_si_shuffle = dec.cell_spatial_info(rec, xy, shuffle_dlc=True)

            tc_12_cc_shuffle, tc1_shuffle, tc2_shuffle = dec.tc_stability(rec, tc, xy, shuffle_dlc=True)

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
            nulldistscc = np.append(shuffled_df.loc[shuffled_df['cellids'] == cell, 'scc'].values, stab[cell])
            # two tailed permutation test
            alpha = 0.05
            significance_threshold = 1-alpha
            threshold_si = np.quantile(abs(nulldistsi), q=significance_threshold)
            p_si = sum(abs(nulldistsi) >= abs(cell_si[cell])) / len(nulldistsi)
            threshold_scc = np.quantile(abs(nulldistscc), q=significance_threshold)
            p_scc = sum(abs(nulldistscc) >= abs(stab[cell])) / len(nulldistscc)

            shuffled_df.loc[shuffled_df['cellids'] == cell,'sig_threshold_si'] = threshold_si
            shuffled_df.loc[shuffled_df['cellids'] == cell, 'p_si'] = p_si

            shuffled_df.loc[shuffled_df['cellids'] == cell,'sig_threshold_scc'] = threshold_scc
            shuffled_df.loc[shuffled_df['cellids'] == cell, 'p_scc'] = p_scc

        shuffled_df.to_pickle(f"/auto/users/wingertj/data/{siteid}_hist_noise_estimate.pkl")

    d = { 'siteid': [siteid for i in range(len(rec['resp'].chans))],
          'cellid': [cell for cell in rec['resp'].chans],
          'tc': [tc[cell] for cell in rec['resp'].chans],
          'si': [cell_si[cell] for cell in rec['resp'].chans],
          'scc': [stab[cell] for cell in rec['resp'].chans],
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
df.to_pickle("/auto/users/wingertj/data/hist_jackknifed_1000perm_df.pkl")
bp = []