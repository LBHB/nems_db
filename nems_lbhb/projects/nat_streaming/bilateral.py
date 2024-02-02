import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
from nems_lbhb import baphy_experiment, baphy_io, xform_wrappers
from nems0 import recording, epoch
from nems_lbhb.gcmodel.figures.snr import compute_snr, compute_snr_multi
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import compress
from scipy.stats import permutation_test

### BNB dual probe parmfiles ###
parmfile = '/auto/data/daq/SlipperyJack/SLJ015/SLJ015d07_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ016/SLJ016a04_p_BNB.m'
# crummy FTC - parmfile = '/auto/data/daq/SlipperyJack/SLJ017/SLJ017a11_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ018/SLJ018c02_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ019/SLJ019a12_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ020/SLJ020c02_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ021/SLJ021a06_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ022/SLJ022b03_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ023/SLJ023a07_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ025/SLJ025a03_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ026/SLJ026a07_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ030/SLJ030a03_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ032/SLJ032a02_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ033/SLJ033a05_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ033/SLJ033a10_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ036/SLJ036a06_p_BNB.m'

### BNT dual probe parmfiles ###
# diotic - low stim num - high rep - 18s
# left probe/A mostly in HC -parmfile = '/auto/data/daq/SlipperyJack/SLJ004/SLJ004c10_p_BNT.m'
# diotic - low stim num - high rep - 6s
parmfile = '/auto/data/daq/SlipperyJack/SLJ015/SLJ015d08_p_BNT.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ016/SLJ016a05_p_BNT.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ021/SLJ021a09_p_BNT.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ022/SLJ022b04_p_BNT.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ025/SLJ025a04_p_BNT.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ026/SLJ026a16_p_BNT.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ030/SLJ030a06_p_BNT.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ032/SLJ032a04_p_BNT.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ036/SLJ036a03_p_BNT.m'

# diotic - regular diotic BNT
# left probe/A mostly in HC - parmfile = '/auto/data/daq/SlipperyJack/SLJ004/SLJ004c11_p_BNT.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ016/SLJ016a06_p_BNT.m'
# crummy FTC - parmfile = '/auto/data/daq/SlipperyJack/SLJ017/SLJ017a12_p_BNT.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ019/SLJ019a13_p_BNT.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ021/SLJ021a12_p_BNT.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ023/SLJ023a09_p_BNT.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ026/SLJ026a13_p_BNT.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ033/SLJ033a06_p_BNT.m'

# mono
parmfile = '/auto/data/daq/SlipperyJack/SLJ032/SLJ032a06_p_BNT.m'

### binaural OLP dual probe parmfiles ###
parmfile = '/auto/data/daq/SlipperyJack/SLJ018/SLJ018c01_p_OLP.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ020/SLJ020c01_p_OLP.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ026/SLJ026a12_p_OLP.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ033/SLJ033a07_p_OLP.m'




# crummy FTC on both probes.
parmfile = '/auto/data/daq/SlipperyJack/SLJ017/SLJ017a11_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ017/SLJ017a12_p_BNT.m'


# maybe co-tuned FTC??? weirdly high correlations
parmfile = '/auto/data/daq/SlipperyJack/SLJ021/SLJ021a06_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ021/SLJ021a12_p_BNT.m'


# weak FTC. maybe overlap. increase cc?
parmfile = '/auto/data/daq/SlipperyJack/SLJ026/SLJ026a07_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ026/SLJ026a13_p_BNT.m'

parmfile = '/auto/data/daq/SlipperyJack/SLJ033/SLJ033a05_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ033/SLJ033a06_p_BNT.m'

# no tuning in probe B
parmfile = '/auto/data/daq/SlipperyJack/SLJ016/SLJ016a04_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ016/SLJ016a06_p_BNT.m'

# tuning unclear?
parmfile = '/auto/data/daq/SlipperyJack/SLJ003/SLJ003a04_p_BNT.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ004/SLJ004c11_p_BNT.m'

# different tuning? decrease??
parmfile = '/auto/data/daq/SlipperyJack/SLJ019/SLJ019a10_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ019/SLJ019a13_p_BNT.m'


# current parmfile
parmfile = '/auto/data/daq/SlipperyJack/SLJ032/SLJ032a04_p_BNT.m'


# list of low stim count 6s BNTs with bilateral data
BNT_parmfiles = ['/auto/data/daq/SlipperyJack/SLJ015/SLJ015d08_p_BNT.m',
                 '/auto/data/daq/SlipperyJack/SLJ016/SLJ016a05_p_BNT.m',
                 '/auto/data/daq/SlipperyJack/SLJ021/SLJ021a09_p_BNT.m',
                 '/auto/data/daq/SlipperyJack/SLJ022/SLJ022b04_p_BNT.m',
                 '/auto/data/daq/SlipperyJack/SLJ025/SLJ025a04_p_BNT.m',
                 '/auto/data/daq/SlipperyJack/SLJ026/SLJ026a16_p_BNT.m',
                 '/auto/data/daq/SlipperyJack/SLJ030/SLJ030a06_p_BNT.m',
                 '/auto/data/daq/SlipperyJack/SLJ032/SLJ032a04_p_BNT.m',
                 '/auto/data/daq/SlipperyJack/SLJ036/SLJ036a03_p_BNT.m']
# BNT_parmfiles = ['/auto/data/daq/SlipperyJack/SLJ025/SLJ025a04_p_BNT.m']

### quick functions ###

def signal_corr_AB(respA, respB, snrA, snrB):

     # get stimulus epochs
     stim_epochs=epoch.epoch_names_matching(respA.epochs,"^STIM_")

     # split stim_epochs into mono vs bilateral stims
     if len(stim_epochs)>1:
          stim_epochs=epoch.epoch_names_matching(respA.epochs,"^STIM_00") + \
              epoch.epoch_names_matching(respA.epochs,"^STIM_NULL:1:0\+00")

          stim_epochs1=[s for s in stim_epochs if 'NULL' in s]
          stim_epochs2=[s for s in stim_epochs if 'NULL' not in s]
          bi_label=['mono','bilateral']
          stim_epochs = [stim_epochs1, stim_epochs2]
     else:
          stim_epochs=[stim_epochs]
          bi_label=['mono']

     # signal correlations
     AA_sc = []
     AB_sc = []
     BB_sc = []
     for stim_eps, lbl in zip(stim_epochs, bi_label):
          AA = []
          AB = []
          BB = []
          for s in stim_eps:
               # extract stim epochs
               vA = respA.extract_epoch(s)
               vB = respB.extract_epoch(s)

               # take PSTH of each neuron
               vA = vA.mean(axis=0)
               vB = vB.mean(axis=0)

               # subtract mean firing rate for each cell
               vA -= vA.mean(axis=1, keepdims=True)
               vB -= vB.mean(axis=1, keepdims=True)

               # find std dev
               sA = vA.std(axis=1, keepdims=True)
               sB = vB.std(axis=1, keepdims=True)

               # divide by standard deviation
               vA /= (sA + (sA == 0))
               vB /= (sB + (sB == 0))

               # take length of samples
               N = vB.shape[1]

               # take dot prodouct of AA, AB, BB to compute pairwise correlations and append them to lists
               AA.append((vA @ vA.T) / N)
               AB.append((vA @ vB.T) / N)
               BB.append((vB @ vB.T) / N)

          # take mean signal correlation across stims
          AA_avg = np.mean(np.stack(AA, axis=2), axis=2)
          AB_avg = np.mean(np.stack(AB, axis=2), axis=2)
          BB_avg = np.mean(np.stack(BB, axis=2), axis=2)
          np.fill_diagonal(AA_avg, 0)
          np.fill_diagonal(BB_avg, 0)

          # append for each signal category
          AA_sc.append(AA_avg)
          AB_sc.append(AB_avg)
          BB_sc.append(BB_avg)

     # create array of cell pair names that matches AA, BB, AB
     AA_pairs = np.full((len(respA.chans), len(respA.chans), 2), fill_value='', dtype=object)
     for i in range(len(respA.chans)):
          for j in range(len(respA.chans)):
               AA_pairs[i, j, 0] = respA.chans[i]
               AA_pairs[i, j, 1] = respA.chans[j]
     BB_pairs = np.full((len(respB.chans), len(respB.chans), 2), fill_value='', dtype=object)
     for i in range(len(respB.chans)):
          for j in range(len(respB.chans)):
               BB_pairs[i, j, 0] = respB.chans[i]
               BB_pairs[i, j, 1] = respB.chans[j]
     AB_pairs = np.full((len(respA.chans), len(respB.chans), 2), fill_value='', dtype=object)
     for i in range(len(respA.chans)):
          for j in range(len(respB.chans)):
               AB_pairs[i, j, 0] = respA.chans[i]
               AB_pairs[i, j, 1] = respB.chans[j]

     bi_pairs = [AA_pairs, AB_pairs, BB_pairs]

     # create array of snr values that matches AA, BB, AB
     AA_snr = np.full((len(snrA), len(snrA), 2), fill_value=np.nan)
     for i in range(len(snrA)):
          for j in range(len(snrA)):
               AA_snr[i, j, 0] = snrA[i]
               AA_snr[i, j, 1] = snrA[j]
     BB_snr = np.full((len(snrB), len(snrB), 2), fill_value=np.nan)
     for i in range(len(snrB)):
          for j in range(len(respB.chans)):
               BB_snr[i, j, 0] = snrB[i]
               BB_snr[i, j, 1] = snrB[j]
     AB_snr = np.full((len(snrA), len(snrB), 2), fill_value=np.nan)
     for i in range(len(snrA)):
          for j in range(len(snrB)):
               AB_snr[i, j, 0] = snrA[i]
               AB_snr[i, j, 1] = snrB[j]

     bi_snr = [AA_snr, AB_snr, BB_snr]

     return AA_sc, AB_sc, BB_sc, bi_label, bi_pairs, bi_snr

def noise_corr_AB(respA, respB, snrA, snrB):

     # get stimulus epochs
     stim_epochs=epoch.epoch_names_matching(respA.epochs,"^STIM_")

     # split stim_epochs into mono vs bilateral stims
     if len(stim_epochs)>1:
          stim_epochs=epoch.epoch_names_matching(respA.epochs,"^STIM_00") + \
              epoch.epoch_names_matching(respA.epochs,"^STIM_NULL:1:0\+00")

          stim_epochs1=[s for s in stim_epochs if 'NULL' in s]
          stim_epochs2=[s for s in stim_epochs if 'NULL' not in s]
          bi_label=['mono','bilateral']
          stim_epochs = [stim_epochs1, stim_epochs2]
     else:
          stim_epochs=[stim_epochs]
          bi_label=['mono']

     # signal correlations
     AA_sc = []
     AB_sc = []
     BB_sc = []
     for stim_eps, lbl in zip(stim_epochs, bi_label):
          AA = []
          AB = []
          BB = []
          for s in stim_eps:
               # extract stim epochs
               rA = respA.extract_epoch(s)
               rB = respB.extract_epoch(s)

               # subtract mean auditory response to look at noise correlations
               vA = rA-rA.mean(axis=0,keepdims=True)
               vB = rB-rB.mean(axis=0,keepdims=True)

               # change to neurons by trials by time
               vA=vA.transpose([1, 0, 2])
               vB = vB.transpose([1, 0, 2])

               # concatenate trials
               vA=np.reshape(vA, [vA.shape[0], -1])
               vB = np.reshape(vB, [vB.shape[0], -1])

               # subtract mean firing rate for each cell
               vA -= vA.mean(axis=1, keepdims=True)
               vB -= vB.mean(axis=1, keepdims=True)

               # find std dev
               sA = vA.std(axis=1, keepdims=True)
               sB = vB.std(axis=1, keepdims=True)

               # divide by standard deviation
               vA /= (sA + (sA == 0))
               vB /= (sB + (sB == 0))

               # take length of samples
               N = vB.shape[1]

               # take dot prodouct of AA, AB, BB to compute pairwise correlations and append them to lists
               AA.append((vA @ vA.T) / N)
               AB.append((vA @ vB.T) / N)
               BB.append((vB @ vB.T) / N)

          # take mean signal correlation across stims
          AA_avg = np.mean(np.stack(AA, axis=2), axis=2)
          AB_avg = np.mean(np.stack(AB, axis=2), axis=2)
          BB_avg = np.mean(np.stack(BB, axis=2), axis=2)
          np.fill_diagonal(AA_avg, 0)
          np.fill_diagonal(BB_avg, 0)

          # append for each signal category
          AA_sc.append(AA_avg)
          AB_sc.append(AB_avg)
          BB_sc.append(BB_avg)

     # create array of cell pair names that matches AA, BB, AB
     AA_pairs = np.full((len(respA.chans), len(respA.chans), 2), fill_value='', dtype=object)
     for i in range(len(respA.chans)):
          for j in range(len(respA.chans)):
               AA_pairs[i, j, 0] = respA.chans[i]
               AA_pairs[i, j, 1] = respA.chans[j]
     BB_pairs = np.full((len(respB.chans), len(respB.chans), 2), fill_value='', dtype=object)
     for i in range(len(respB.chans)):
          for j in range(len(respB.chans)):
               BB_pairs[i, j, 0] = respB.chans[i]
               BB_pairs[i, j, 1] = respB.chans[j]
     AB_pairs = np.full((len(respA.chans), len(respB.chans), 2), fill_value='', dtype=object)
     for i in range(len(respA.chans)):
          for j in range(len(respB.chans)):
               AB_pairs[i, j, 0] = respA.chans[i]
               AB_pairs[i, j, 1] = respB.chans[j]

     bi_pairs = [AA_pairs, AB_pairs, BB_pairs]

     # create array of snr values that matches AA, BB, AB
     AA_snr = np.full((len(snrA), len(snrA), 2), fill_value=np.nan)
     for i in range(len(snrA)):
          for j in range(len(snrA)):
               AA_snr[i, j, 0] = snrA[i]
               AA_snr[i, j, 1] = snrA[j]
     BB_snr = np.full((len(snrB), len(snrB), 2), fill_value=np.nan)
     for i in range(len(snrB)):
          for j in range(len(respB.chans)):
               BB_snr[i, j, 0] = snrB[i]
               BB_snr[i, j, 1] = snrB[j]
     AB_snr = np.full((len(snrA), len(snrB), 2), fill_value=np.nan)
     for i in range(len(snrA)):
          for j in range(len(snrB)):
               AB_snr[i, j, 0] = snrA[i]
               AB_snr[i, j, 1] = snrB[j]

     bi_snr = [AA_snr, AB_snr, BB_snr]

     return AA_sc, AB_sc, BB_sc, bi_label, bi_pairs, bi_snr

### for each site perform permutation test on mean difference between within pairwise correlations vs between hemi
def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def gen_corr_dataframe(sAA, sAB, sBB, nAA, nAB, nBB, slabels, bi_pairs, bi_snr, parmfile, siteid, A_area, B_area):

     import pandas as pd
     # get probe 1 and probe 2 unit names and area for AA, BB, AB
     #AA
     linear_cellid_AA_1 = bi_pairs[0][:, :, 0][np.triu_indices(bi_pairs[0][:, :, 0].shape[0], k=1)]
     linear_area_AA_1 = [A_area for i in range(len(linear_cellid_AA_1))]
     linear_cellid_AA_2 = bi_pairs[0][:, :, 1][np.triu_indices(bi_pairs[0][:, :, 1].shape[0], k=1)]
     linear_area_AA_2 = [A_area for i in range(len(linear_cellid_AA_1))]
     linear_cellpairs_AA = [c1 + "," + c2 for c1, c2 in zip(linear_cellid_AA_1, linear_cellid_AA_2)]
     linear_probepair_AA = ['AA' for i in range(len(linear_cellpairs_AA))]
     #BB
     linear_cellid_BB_1 = bi_pairs[2][:, :, 0][np.triu_indices(bi_pairs[2][:, :, 0].shape[0], k=1)]
     linear_area_BB_1 = [B_area for i in range(len(linear_cellid_BB_1))]
     linear_cellid_BB_2 = bi_pairs[2][:, :, 1][np.triu_indices(bi_pairs[2][:, :, 1].shape[0], k=1)]
     linear_area_BB_2 = [B_area for i in range(len(linear_cellid_BB_2))]
     linear_cellpairs_BB = [c1 + "," + c2 for c1, c2 in zip(linear_cellid_BB_1, linear_cellid_BB_2)]
     linear_probepair_BB = ['BB' for i in range(len(linear_cellpairs_BB))]
     #AB
     linear_cellid_AB_1 = bi_pairs[1][:, :, 0].flatten()
     linear_area_AB_1 = [A_area for i in range(len(linear_cellid_AB_1))]
     linear_cellid_AB_2 = bi_pairs[1][:, :, 1].flatten()
     linear_area_AB_2 = [B_area for i in range(len(linear_cellid_AB_2))]
     linear_cellpairs_AB = [c1 + "," + c2 for c1, c2 in zip(linear_cellid_AB_1, linear_cellid_AB_2)]
     linear_probepair_AB = ['AB' for i in range(len(linear_cellpairs_AB))]

     # linearize SNR
     #AA
     linear_snr_AA_1 = bi_snr[0][:, :, 0][np.triu_indices(bi_snr[0][:, :, 0].shape[0], k=1)]
     linear_snr_AA_2 = bi_snr[0][:, :, 1][np.triu_indices(bi_snr[0][:, :, 1].shape[0], k=1)]

     #BB
     linear_snr_BB_1 = bi_snr[2][:, :, 0][np.triu_indices(bi_snr[2][:, :, 0].shape[0], k=1)]
     linear_snr_BB_2 = bi_snr[2][:, :, 1][np.triu_indices(bi_snr[2][:, :, 1].shape[0], k=1)]

     #AB
     linear_snr_AB_1 = bi_snr[1][:, :, 0].flatten()
     linear_snr_AB_2 = bi_snr[1][:, :, 1].flatten()


     # linearize pairwise correlations for mono and bi stims
     linear_sAA = []
     linear_sBB = []
     linear_sAB = []
     linear_nAA = []
     linear_nBB = []
     linear_nAB = []
     for i in range(len(slabels)):
          linear_sAA.append(sAA[i][np.triu_indices(sAA[i].shape[0], k=1)])
          linear_sBB.append(sBB[i][np.triu_indices(sBB[i].shape[0], k=1)])
          linear_sAB.append(sAB[i].flatten())
          linear_nAA.append(nAA[i][np.triu_indices(nAA[i].shape[0], k=1)])
          linear_nBB.append(nBB[i][np.triu_indices(nBB[i].shape[0], k=1)])
          linear_nAB.append(nAB[i].flatten())

     # package data into
     d = {'cell_pair': np.concatenate((linear_cellpairs_AA, linear_cellpairs_AB, linear_cellpairs_BB)),
          'cell_1': np.concatenate((linear_cellid_AA_1, linear_cellid_AB_1, linear_cellid_BB_1)),
          'cell_2': np.concatenate((linear_cellid_AA_2, linear_cellid_AB_2, linear_cellid_BB_2)),
          'area_1': np.concatenate((linear_area_AA_1, linear_area_AB_1, linear_area_BB_1)),
          'area_2': np.concatenate((linear_area_AA_2, linear_area_AB_2, linear_area_BB_2)),
          'probe_pair': np.concatenate((linear_probepair_AA, linear_probepair_AB, linear_probepair_BB)),
          'signal_corr_mono': np.concatenate((linear_sAA[0], linear_sAB[0], linear_sBB[0])),
          'signal_corr_bi': np.concatenate((linear_sAA[1], linear_sAB[1], linear_sBB[1])),
          'noise_corr_mono': np.concatenate((linear_nAA[0], linear_nAB[0], linear_nBB[0])),
          'noise_corr_bi': np.concatenate((linear_nAA[1], linear_nAB[1], linear_nBB[1])),
          'cell_1_snr': np.concatenate((linear_snr_AA_1, linear_snr_AB_1, linear_snr_BB_1)),
          'cell_2_snr': np.concatenate((linear_snr_AA_2, linear_snr_AB_2, linear_snr_BB_2)),
          'siteid': [siteid for i in range(len(np.concatenate((linear_cellpairs_AA, linear_cellpairs_AB, linear_cellpairs_BB))))],
          'parmfile': [parmfile for i in range(len(np.concatenate((linear_cellpairs_AA, linear_cellpairs_AB, linear_cellpairs_BB))))],
             }

     corr_dataframe = pd.DataFrame(data=d)

     return corr_dataframe

# random number generator for permutation
rng = np.random.default_rng()

# create dataframe
sAA_all = {}
sBB_all = {}
sAB_all = {}
nAA_all = {}
nBB_all = {}
nAB_all = {}
sAA_all_high = {}
sBB_all_high = {}
sAB_all_high = {}
nAA_all_high = {}
nBB_all_high = {}
nAB_all_high = {}
parmfile_snr = {}
parmfile_area = {}
badfiles = []
parmfile_dfs = []
for parmfile in BNT_parmfiles:

     ex = baphy_experiment.BAPHYExperiment(parmfile=[parmfile])
     loadkey="psth.fs10"

     rec = ex.get_recording(loadkey=loadkey)
     #.sortparameters.Kilosort_load_completed_job_params

     resp=rec['resp'].rasterize()


     # grab A1 units only
     try:
          siteid = parmfile[-17:-10]
          depth_info = baphy_io.get_depth_info(siteid=siteid)
          A1_units = depth_info.loc[depth_info.isin({'area':["A1", 'PEG']})['area']].index.tolist()
          A_units = [unit for unit in A1_units if "-A-" in unit]
          B_units = [unit for unit in A1_units if "-B-" in unit]
          A_area = depth_info['area'][A_units[0]]
          B_area = depth_info['area'][B_units[0]]
          parmfile_area[parmfile] = {"A":A_area, "B":B_area}

          A1_units = [unit[-8:] for unit in A1_units]
          # grab unit names for all units in resp that have depth info in A1
          A_chans = [chan for chan in resp.chans if '-A-' in chan if chan[-8:] in A1_units]
          B_chans = [chan for chan in resp.chans if '-B-' in chan if chan[-8:] in A1_units]
          # A_chans = [c for c in resp.chans if '-A-' in c]
          # B_chans = [c for c in resp.chans if '-B-' in c]
          respA = resp.extract_channels(A_chans)
          respB = resp.extract_channels(B_chans)
     except:
         print(f"{parmfile} depth info not working?...skipping")
         badfiles.append(parmfile)
         continue

     # compute auditory responsiveness
     snrA = compute_snr_multi(respA)
     snrB = compute_snr_multi(respB)

     parmfile_snr[parmfile] = [{a_chan: asnr for a_chan, asnr in list(zip(respA.chans, snrA))}, {b_chan: bsnr for b_chan, bsnr in list(zip(respB.chans, snrB))}]

     highA = snrA >= 0.4
     lowA = snrA < 0.4
     highB = snrB >= 0.4
     lowB = snrB < 0.4

     stim_epochs=epoch.epoch_names_matching(resp.epochs,"^STIM_")

     # split stim_epochs into mono vs bilateral stims
     if len(stim_epochs)>1:
          stim_epochs=epoch.epoch_names_matching(resp.epochs,"^STIM_00") + \
              epoch.epoch_names_matching(resp.epochs,"^STIM_NULL:1:0\+00")

          stim_epochs1=[s for s in stim_epochs if 'NULL' in s]
          stim_epochs2=[s for s in stim_epochs if 'NULL' not in s]
          bi_label=['mono','bilateral']
          stim_epochs = [stim_epochs1, stim_epochs2]
     else:
          stim_epochs=[stim_epochs]
          bi_label=['mono']

     snr_bins = np.linspace(0, 1, 10, endpoint=False)
     snrA_hist, bin_edges = np.histogram(snrA, bins=snr_bins)
     snrB_hist, bin_edges = np.histogram(snrB, bins=snr_bins)

     # for i in range(sum(highA)):
     #      f, ax = plt.subplots(2,1, sharex=True)
     #      good_unit = np.where(highA)[0][i]
     #      psthA = np.concatenate([respA.extract_epoch(e)[:, good_unit, :] for e in stim_epochs[0]], axis=1)
     #      ax[0].plot(psthA.mean(axis=0))
     #      ax[1].imshow(psthA, origin='lower', aspect='auto', interpolation=None)
     #      ax[0].set_title(f"{snrA[good_unit]:.2f}")

     # quick plot of goob/bad snr units between sites
     f, ax = plt.subplots(4, 3, figsize=(7,5), sharex='col')
     ax[1, 0].stairs(snrA_hist/sum(snrA_hist)*100, edges=bin_edges, fill=True)
     ax[3, 0].stairs(snrB_hist/sum(snrB_hist)*100, edges=bin_edges, fill=True)
     dividerA = make_axes_locatable(ax[1,0])
     dividerB = make_axes_locatable(ax[3,0])
     Adiv = dividerA.append_axes("left", size="5%", pad=0.5)
     Adiv.set_visible(False)
     Bdiv = dividerB.append_axes("left", size="5%", pad=0.5)
     Bdiv.set_visible(False)
     row_space = np.linspace(0, 1, 21)
     ProbeA = "A"
     ProbeB = "B"
     plt.figtext(0.01, row_space[13], s=f"ProbeA - {parmfile_area[parmfile][ProbeA]}", rotation='vertical', fontsize=12)
     plt.figtext(0.01, row_space[4], s=f"ProbeB - {parmfile_area[parmfile][ProbeB]}", rotation='vertical', fontsize=12)
     ax[3, 0].set_xlabel("SNR")
     ax[3, 0].set_ylabel("Percent of units")
     ax[1, 0].set_ylabel("Percent of units")
     good_unitA = np.where(highA)[0][np.random.randint(len(np.where(highA)[0]), size=1)[0]]
     good_unitB = np.where(highB)[0][np.random.randint(len(np.where(highB)[0]), size=1)[0]]
     psthA = np.concatenate([respA.extract_epoch(e)[:, good_unitA, :] for e in stim_epochs[0]], axis=1)
     psthB = np.concatenate([respB.extract_epoch(e)[:, good_unitB, :] for e in stim_epochs[0]], axis=1)
     bad_unitA = np.where(lowA)[0][np.random.randint(len(np.where(lowA)[0]), size=1)[0]]
     bad_unitB = np.where(lowB)[0][np.random.randint(len(np.where(lowB)[0]), size=1)[0]]
     psthA_low = np.concatenate([respA.extract_epoch(e)[:, bad_unitA, :] for e in stim_epochs[0]], axis=1)
     psthB_low = np.concatenate([respB.extract_epoch(e)[:, bad_unitB, :] for e in stim_epochs[0]], axis=1)
     ax[0,1].plot(psthA.mean(axis=0), color='black')
     ax[0,1].set_ylabel("firing rate")
     ax[1,1].imshow(psthA, origin='lower', aspect='auto', interpolation=None)
     ax[1,1].set_ylabel("trials")
     ax[0,1].set_title(f"SNR - {snrA[good_unitA]:.2f}")
     ax[2,1].plot(psthB.mean(axis=0), color='black')
     ax[2,1].set_ylabel("firing rate")
     ax[3,1].imshow(psthB, origin='lower', aspect='auto', interpolation=None)
     ax[3,1].set_ylabel("trials")
     ax[2,1].set_title(f"SNR - {snrB[good_unitB]:.2f}")

     ax[0,2].plot(psthA_low.mean(axis=0), color='purple')
     ax[1,2].imshow(psthA_low, origin='lower', aspect='auto', interpolation=None)
     ax[0,2].set_title(f"SNR - {snrA[bad_unitA]:.2f}")
     ax[2,2].plot(psthB_low.mean(axis=0), color='purple')
     ax[3,2].imshow(psthB_low, origin='lower', aspect='auto', interpolation=None)
     ax[2,2].set_title(f"SNR - {snrB[bad_unitB]:.2f}")

     ax[3, 0].axvline(snrB[good_unitB], color='black')
     ax[3, 0].axvline(snrB[bad_unitB], color='purple')
     ax[1, 0].axvline(snrA[good_unitA], color='black')
     ax[1, 0].axvline(snrA[bad_unitA], color='purple')
     ax[0, 0].remove()
     ax[2, 0].remove()
     f.suptitle(parmfile[-17:])
     plt.tight_layout()

     respA_high = respA.extract_channels(list(compress(respA.chans, highA)))
     respB_high = respB.extract_channels(list(compress(respB.chans, highB)))

     sAA_high, sAB_high, sBB_high, slabels, bi_pairs, bi_snr = signal_corr_AB(respA_high, respB_high, snrA, snrB)
     nAA_high, nAB_high, nBB_high, nlabels, bi_pairs, bi_snr = noise_corr_AB(respA_high, respB_high, snrA, snrB)

     sAA, sAB, sBB, slabels, bi_pairs, bi_snr = signal_corr_AB(respA, respB, snrA, snrB)
     nAA, nAB, nBB, nlabels, bi_pairs, bi_snr = noise_corr_AB(respA, respB, snrA, snrB)

     parmfile_dfs.append(gen_corr_dataframe(sAA, sAB, sBB, nAA, nAB, nBB, slabels, bi_pairs, bi_snr, parmfile, siteid, A_area, B_area))

     sAA_all[parmfile] = sAA
     sBB_all[parmfile] = sBB
     sAB_all[parmfile] = sAB
     nAA_all[parmfile] = nAA
     nBB_all[parmfile] = nBB
     nAB_all[parmfile] = nAB
     sAA_all_high[parmfile] = sAA_high
     sBB_all_high[parmfile] = sBB_high
     sAB_all_high[parmfile] = sAB_high
     nAA_all_high[parmfile] = nAA_high
     nBB_all_high[parmfile] = nBB_high
     nAB_all_high[parmfile] = nAB_high

good_files = [parmfile for parmfile in BNT_parmfiles if not parmfile in badfiles]

corr_df = pd.concat(parmfile_dfs)


# f, ax = plt.subplots(len(good_files), 5, figsize=(8,5))
# all_pairs = []
# for parmfile in good_files:
#      for i, label in enumerate(slabels):
#           if label == 'mono':
#                all_pairs.append(np.concatenate(
#                (sAA_all_high[parmfile][i][np.triu_indices(sAA_all_high[parmfile][i].shape[0], k=1)],
#                 sBB_all_high[parmfile][i][np.triu_indices(sBB_all_high[parmfile][i].shape[0], k=1)],
#                 sAB_all_high[parmfile][i].flatten())))
# all_pairs = np.concatenate(all_pairs)
# min_all = all_pairs.min()
# max_all = all_pairs.max()
# abs_max = np.abs(all_pairs).max()
#
# for j, parmfile in enumerate(good_files):
#      for i, label in enumerate(slabels):
#           if label == 'mono':
#                # set image options
#                imopts = {'cmap': 'gray', 'origin': 'lower', 'vmin': -abs_max, 'vmax': abs_max}
#                ax[j, 0].imshow(sAA_all_high[parmfile][i], **imopts)
#                ax[j, 1].imshow(sBB_all_high[parmfile][i], **imopts)
#                ax[j, 2].imshow(sAB_all_high[parmfile][i], **imopts)
#
#                # make a histogram of pairwise cc
#                bins = np.linspace(min_all, max_all, 21)
#                AA_hist, hist_bins = np.histogram(sAA_all_high[parmfile][i][np.triu_indices(sAA_all_high[parmfile][i].shape[0], k=1)], bins=bins)
#                AB_hist = np.histogram(sAB_all_high[parmfile][i].flatten(), bins=bins)[0]
#                BB_hist = np.histogram(sBB_all_high[parmfile][i][np.triu_indices(sBB_all_high[parmfile][i].shape[0], k=1)], bins=bins)[0]
#                # probability density
#                pdf_AA = AA_hist / sum(AA_hist)
#                pdf_BB = BB_hist / sum(BB_hist)
#                pdf_AB = AB_hist / sum(AB_hist)
#
#                # cummulative density
#                cdf_AA = np.cumsum(pdf_AA)
#                cdf_BB = np.cumsum(pdf_BB)
#                cdf_AB = np.cumsum(pdf_AB)
#
#                # h = np.stack(h, axis=1).astype(float)
#                # h /= h.sum(axis=0, keepdims=True)
#                ax[j, 3].plot(hist_bins[1:], cdf_AA)
#                ax[j, 3].plot(hist_bins[1:], cdf_BB)
#                ax[j, 3].plot(hist_bins[1:], cdf_AB)
#                ax[j, 3].legend(("AA", "BB", "AB"))
#
#                same_hemisphere = np.concatenate(
#                     (nAA_all_high[parmfile][i][np.triu_indices(nAA_all_high[parmfile][i].shape[0], k=1)],
#                      nBB_all_high[parmfile][i][np.triu_indices(nBB_all_high[parmfile][i].shape[0], k=1)]))
#                between_hemisphere = nAB_all_high[parmfile][i].flatten()
#                res = permutation_test((same_hemisphere, between_hemisphere), statistic, n_resamples=1000,
#                                       vectorized=True, alternative='two-sided',
#                                       random_state=rng)
#                print("true statistic " + str(res.statistic))
#                print("pvalue " + str(res.pvalue))
#                ax[j, 4].hist(res.null_distribution, bins=50)
#                ax[j, 4].text(0, 45, s = f"pvalue:{np.round(res.pvalue, decimals = 3)}")
#                ax[j, 4].set_title("Permutation distribution of test statistic")
#                ax[j, 4].set_xlabel("Value of Statistic")
#                ax[j, 4].set_ylabel("Frequency")
#                ax[j, 4].axvline(res.statistic)
#           else:
#                continue
# plt.tight_layout()
#
# f, ax = plt.subplots(len(good_files), 5, figsize=(8,5))
# all_pairs = []
# for parmfile in good_files:
#      for i, label in enumerate(slabels):
#           if label == 'mono':
#                all_pairs.append(np.concatenate(
#                (nAA_all_high[parmfile][i][np.triu_indices(nAA_all_high[parmfile][i].shape[0], k=1)],
#                 nBB_all_high[parmfile][i][np.triu_indices(nBB_all_high[parmfile][i].shape[0], k=1)],
#                 nAB_all_high[parmfile][i].flatten())))
# all_pairs = np.concatenate(all_pairs)
# min_all = all_pairs.min()
# max_all = all_pairs.max()
# abs_max = np.abs(all_pairs).max()
#
# for j, parmfile in enumerate(good_files):
#      for i, label in enumerate(slabels):
#           if label == 'mono':
#                # set image options
#                imopts = {'cmap': 'gray', 'origin': 'lower', 'vmin': -abs_max, 'vmax': abs_max}
#                ax[j, 0].imshow(nAA_all_high[parmfile][i], **imopts)
#                ax[j, 1].imshow(nBB_all_high[parmfile][i], **imopts)
#                ax[j, 2].imshow(nAB_all_high[parmfile][i], **imopts)
#
#                # make a histogram of pairwise cc
#                bins = np.linspace(min_all, max_all, 21)
#                AA_hist, hist_bins = np.histogram(nAA_all_high[parmfile][i][np.triu_indices(nAA_all_high[parmfile][i].shape[0], k=1)], bins=bins)
#                AB_hist = np.histogram(nAB_all_high[parmfile][i].flatten(), bins=bins)[0]
#                BB_hist = np.histogram(nBB_all_high[parmfile][i][np.triu_indices(nBB_all_high[parmfile][i].shape[0], k=1)], bins=bins)[0]
#                # probability density
#                pdf_AA = AA_hist / sum(AA_hist)
#                pdf_BB = BB_hist / sum(BB_hist)
#                pdf_AB = AB_hist / sum(AB_hist)
#
#                # cummulative density
#                cdf_AA = np.cumsum(pdf_AA)
#                cdf_BB = np.cumsum(pdf_BB)
#                cdf_AB = np.cumsum(pdf_AB)
#
#                # h = np.stack(h, axis=1).astype(float)
#                # h /= h.sum(axis=0, keepdims=True)
#                ax[j, 3].plot(hist_bins[1:], cdf_AA)
#                ax[j, 3].plot(hist_bins[1:], cdf_BB)
#                ax[j, 3].plot(hist_bins[1:], cdf_AB)
#                ax[j, 3].legend(("AA", "BB", "AB"))
#
#                same_hemisphere = np.concatenate(
#                     (nAA_all_high[parmfile][i][np.triu_indices(nAA_all_high[parmfile][i].shape[0], k=1)],
#                      nBB_all_high[parmfile][i][np.triu_indices(nBB_all_high[parmfile][i].shape[0], k=1)]))
#                between_hemisphere = nAB_all_high[parmfile][i].flatten()
#                res = permutation_test((same_hemisphere, between_hemisphere), statistic, n_resamples=1000,
#                                       vectorized=True, alternative='two-sided',
#                                       random_state=rng)
#                print("true statistic " + str(res.statistic))
#                print("pvalue " + str(res.pvalue))
#                ax[j, 4].hist(res.null_distribution, bins=50)
#                ax[j, 4].text(0, 45, s = f"pvalue:{np.round(res.pvalue, decimals = 3)}")
#                ax[j, 4].set_title("Permutation distribution of test statistic")
#                ax[j, 4].set_xlabel("Value of Statistic")
#                ax[j, 4].set_ylabel("Frequency")
#                ax[j, 4].axvline(res.statistic)
#           else:
#                continue
# plt.tight_layout()

# Signal and noise correlations histograms and permutation tests all parmfiles
f, ax = plt.subplots(len(good_files), 4, figsize=(10,8), sharex='col')
all_pairs = []
for parmfile in good_files:
     for i, label in enumerate(slabels):
          if label == 'mono':
               all_pairs.append(np.concatenate(
               (nAA_all_high[parmfile][i][np.triu_indices(nAA_all_high[parmfile][i].shape[0], k=1)],
                nBB_all_high[parmfile][i][np.triu_indices(nBB_all_high[parmfile][i].shape[0], k=1)],
                nAB_all_high[parmfile][i].flatten())))
all_pairs = np.concatenate(all_pairs)
min_all = all_pairs.min()
max_all = all_pairs.max()
abs_max = np.abs(all_pairs).max()

for j, parmfile in enumerate(good_files):
     for i, label in enumerate(slabels):
          if label == 'mono':
               # plot siteid on left and areas
               divider = make_axes_locatable(ax[j, 0])
               div = divider.append_axes("left", size="5%", pad=0.5)
               xax = div.get_xaxis()
               xax.set_visible(False)
               div.set_yticklabels([])
               div.set_yticks([])
               ProbeA='A'
               ProbeB='B'
               div.set_ylabel(f"{parmfile[-17:-10]} \n {parmfile_area[parmfile][ProbeA]}-{parmfile_area[parmfile][ProbeB]}")

               # make a histogram of pairwise cc
               bins = np.linspace(min_all, max_all, 21)
               nAA_pairs = nAA_all_high[parmfile][i][np.triu_indices(nAA_all_high[parmfile][i].shape[0], k=1)]
               nBB_pairs = nBB_all_high[parmfile][i][np.triu_indices(nBB_all_high[parmfile][i].shape[0], k=1)]
               nAB_pairs = nAB_all_high[parmfile][i].flatten()
               AA_hist, hist_bins = np.histogram(nAA_pairs, bins=bins)
               AB_hist = np.histogram(nAB_pairs, bins=bins)[0]
               BB_hist = np.histogram(nBB_pairs, bins=bins)[0]
               # probability density
               pdf_AA = AA_hist / sum(AA_hist)
               pdf_BB = BB_hist / sum(BB_hist)
               pdf_AB = AB_hist / sum(AB_hist)

               # cummulative density
               cdf_AA = np.cumsum(pdf_AA)
               cdf_BB = np.cumsum(pdf_BB)
               cdf_AB = np.cumsum(pdf_AB)

               # h = np.stack(h, axis=1).astype(float)
               # h /= h.sum(axis=0, keepdims=True)
               ax[j, 0].plot(hist_bins[1:], cdf_AA, color="red")
               ax[j, 0].plot(hist_bins[1:], cdf_BB, color='maroon')
               ax[j, 0].plot(hist_bins[1:], cdf_AB, color="blue")
               ax[j, 0].legend((f"AA n={len(nAA_pairs)}", f"BB n={len(nBB_pairs)}", f"AB n={len(nAB_pairs)}"))
               ax[j, 0].set_xlabel("Noise correlation")
               ax[j, 0].set_ylabel("CDF")

               same_hemisphere = np.concatenate(
                    (nAA_all_high[parmfile][i][np.triu_indices(nAA_all_high[parmfile][i].shape[0], k=1)],
                     nBB_all_high[parmfile][i][np.triu_indices(nBB_all_high[parmfile][i].shape[0], k=1)]))
               between_hemisphere = nAB_all_high[parmfile][i].flatten()
               res = permutation_test((same_hemisphere, between_hemisphere), statistic, n_resamples=1000,
                                      vectorized=True, alternative='two-sided',
                                      random_state=rng)
               print("true statistic " + str(res.statistic))
               print("pvalue " + str(res.pvalue))
               ax[j, 1].hist(res.null_distribution, bins=50, color="grey")
               ax[j, 1].text(0, 45, s = f"pvalue:{np.round(res.pvalue, decimals = 3)}")
               ax[j, 1].set_title("Null distribution - mean diff (within - between)")
               ax[j, 1].set_xlabel("Mean difference")
               ax[j, 1].set_ylabel("Frequency")
               ax[j, 1].axvline(res.statistic, color="black")

               # make a histogram of pairwise cc
               bins = np.linspace(min_all, max_all, 21)
               sAA_pairs = sAA_all_high[parmfile][i][np.triu_indices(sAA_all_high[parmfile][i].shape[0], k=1)]
               sBB_pairs = sBB_all_high[parmfile][i][np.triu_indices(sBB_all_high[parmfile][i].shape[0], k=1)]
               sAB_pairs = sAB_all_high[parmfile][i].flatten()
               AA_hist, hist_bins = np.histogram(sAA_pairs, bins=bins)
               AB_hist = np.histogram(sAB_pairs, bins=bins)[0]
               BB_hist = np.histogram(sBB_pairs, bins=bins)[0]

               # probability density
               pdf_AA = AA_hist / sum(AA_hist)
               pdf_BB = BB_hist / sum(BB_hist)
               pdf_AB = AB_hist / sum(AB_hist)

               # cummulative density
               cdf_AA = np.cumsum(pdf_AA)
               cdf_BB = np.cumsum(pdf_BB)
               cdf_AB = np.cumsum(pdf_AB)

               # h = np.stack(h, axis=1).astype(float)
               # h /= h.sum(axis=0, keepdims=True)
               ax[j, 2].plot(hist_bins[1:], cdf_AA, color="red")
               ax[j, 2].plot(hist_bins[1:], cdf_BB, color='maroon')
               ax[j, 2].plot(hist_bins[1:], cdf_AB, color="blue")
               ax[j, 2].legend((f"AA n={len(sAA_pairs)}", f"BB n={len(sBB_pairs)}", f"AB n={len(sAB_pairs)}"))
               ax[j, 2].set_xlabel("Signal correlation")
               ax[j, 2].set_ylabel("CDF")

               same_hemisphere = np.concatenate(
                    (sAA_all_high[parmfile][i][np.triu_indices(sAA_all_high[parmfile][i].shape[0], k=1)],
                     sBB_all_high[parmfile][i][np.triu_indices(sBB_all_high[parmfile][i].shape[0], k=1)]))
               between_hemisphere = sAB_all_high[parmfile][i].flatten()
               res = permutation_test((same_hemisphere, between_hemisphere), statistic, n_resamples=1000,
                                      vectorized=True, alternative='two-sided',
                                      random_state=rng)
               print("true statistic " + str(res.statistic))
               print("pvalue " + str(res.pvalue))
               ax[j, 3].hist(res.null_distribution, bins=50, color="grey")
               ax[j, 3].text(0, 45, s = f"pvalue:{np.round(res.pvalue, decimals = 3)}")
               ax[j, 3].set_title("Null distribution - mean diff (within - between)")
               ax[j, 3].set_xlabel("Value of Statistic")
               ax[j, 3].set_ylabel("Frequency")
               ax[j, 3].axvline(res.statistic, color="black")
          else:
               continue
plt.tight_layout()

# repeat above to check and see if same result with dataframe
high_corr_df = corr_df.query('cell_1_snr > 0.4 and cell_2_snr > 0.4')
f, ax = plt.subplots(len(good_files), 4, figsize=(10,8), sharex='col')
f1, ax1 = plt.subplots(len(good_files), 4, figsize=(10,8), sharex='col')
figs = [f, f1]
axs = [ax, ax1]
signal = ['signal_corr_mono', 'signal_corr_bi']
noise = ['noise_corr_mono', 'noise_corr_bi']
min_all = min([high_corr_df['noise_corr_mono'].min(), high_corr_df['noise_corr_bi'].min()])
max_all = min([high_corr_df['noise_corr_mono'].max(), high_corr_df['noise_corr_bi'].max()])
for i, parmfile in enumerate(good_files):
     temp_parm = parmfile
     # create temp df for the current parmfile
     parm_df = high_corr_df.query('parmfile == @parmfile')

     for j in range(len(figs)):
          cax = axs[j]
          cfig = figs[j]
          csignal = signal[j]
          cnoise = noise[j]
          # plot siteid on left y axis
          # plot siteid on left and areas
          divider = make_axes_locatable(cax[i, 0])
          div = divider.append_axes("left", size="5%", pad=0.5)
          xax = div.get_xaxis()
          xax.set_visible(False)
          div.set_yticklabels([])
          div.set_yticks([])
          ProbeA = 'A'
          ProbeB = 'B'
          div.set_ylabel(f"{parmfile[-17:-10]} \n {parmfile_area[parmfile][ProbeA]}-{parmfile_area[parmfile][ProbeB]}")

          # make a histogram of pairwise cc
          bins = np.linspace(min_all, max_all, 21)
          nAA_pairs = parm_df[parm_df['probe_pair'] == 'AA'][cnoise].to_numpy()
          nBB_pairs = parm_df[parm_df['probe_pair'] == 'BB'][cnoise].to_numpy()
          nAB_pairs = parm_df[parm_df['probe_pair'] == 'AB'][cnoise].to_numpy()
          AA_hist, hist_bins = np.histogram(nAA_pairs, bins=bins)
          AB_hist = np.histogram(nAB_pairs, bins=bins)[0]
          BB_hist = np.histogram(nBB_pairs, bins=bins)[0]
          # probability density
          pdf_AA = AA_hist / sum(AA_hist)
          pdf_BB = BB_hist / sum(BB_hist)
          pdf_AB = AB_hist / sum(AB_hist)

          # cummulative density
          cdf_AA = np.cumsum(pdf_AA)
          cdf_BB = np.cumsum(pdf_BB)
          cdf_AB = np.cumsum(pdf_AB)

          cax[i, 0].plot(hist_bins[1:], cdf_AA, color="red")
          cax[i, 0].plot(hist_bins[1:], cdf_BB, color='maroon')
          cax[i, 0].plot(hist_bins[1:], cdf_AB, color="blue")
          cax[i, 0].legend((f"AA n={len(nAA_pairs)}", f"BB n={len(nBB_pairs)}", f"AB n={len(nAB_pairs)}"))
          cax[i, 0].set_xlabel(cnoise)
          cax[i, 0].set_ylabel("CDF")

          # permutation test difference between AA+BB and AB
          same_hemisphere = np.concatenate((nAA_pairs, nBB_pairs))
          between_hemisphere = nAB_pairs
          res = permutation_test((same_hemisphere, between_hemisphere), statistic, n_resamples=1000,
                                 vectorized=True, alternative='two-sided',
                                 random_state=rng)
          print("true statistic " + str(res.statistic))
          print("pvalue " + str(res.pvalue))
          cax[i, 1].hist(res.null_distribution, bins=50, color="grey")
          cax[i, 1].text(0, 45, s=f"pvalue:{np.round(res.pvalue, decimals=3)}")
          cax[i, 1].set_title("Null distribution - mean diff (within - between)")
          cax[i, 1].set_xlabel("Mean difference")
          cax[i, 1].set_ylabel("Frequency")
          cax[i, 1].axvline(res.statistic, color="black")

          # repeat above for signal
          # make a histogram of pairwise cc
          bins = np.linspace(min_all, max_all, 21)
          sAA_pairs = parm_df[parm_df['probe_pair'] == 'AA'][csignal].to_numpy()
          sBB_pairs = parm_df[parm_df['probe_pair'] == 'BB'][csignal].to_numpy()
          sAB_pairs = parm_df[parm_df['probe_pair'] == 'AB'][csignal].to_numpy()
          AA_hist, hist_bins = np.histogram(sAA_pairs, bins=bins)
          AB_hist = np.histogram(sAB_pairs, bins=bins)[0]
          BB_hist = np.histogram(sBB_pairs, bins=bins)[0]
          # probability density
          pdf_AA = AA_hist / sum(AA_hist)
          pdf_BB = BB_hist / sum(BB_hist)
          pdf_AB = AB_hist / sum(AB_hist)

          # cummulative density
          cdf_AA = np.cumsum(pdf_AA)
          cdf_BB = np.cumsum(pdf_BB)
          cdf_AB = np.cumsum(pdf_AB)

          cax[i, 2].plot(hist_bins[1:], cdf_AA, color="red")
          cax[i, 2].plot(hist_bins[1:], cdf_BB, color='maroon')
          cax[i, 2].plot(hist_bins[1:], cdf_AB, color="blue")
          cax[i, 2].legend((f"AA n={len(sAA_pairs)}", f"BB n={len(sBB_pairs)}", f"AB n={len(sAB_pairs)}"))
          cax[i, 2].set_xlabel(csignal)
          cax[i, 2].set_ylabel("CDF")

          # permutation test difference between AA+BB and AB
          same_hemisphere = np.concatenate((sAA_pairs, sBB_pairs))
          between_hemisphere = sAB_pairs
          res = permutation_test((same_hemisphere, between_hemisphere), statistic, n_resamples=1000,
                                 vectorized=True, alternative='two-sided',
                                 random_state=rng)
          print("true statistic " + str(res.statistic))
          print("pvalue " + str(res.pvalue))
          cax[i, 3].hist(res.null_distribution, bins=50, color="grey")
          cax[i, 3].text(0, 45, s=f"pvalue:{np.round(res.pvalue, decimals=3)}")
          cax[i, 3].set_title("Null distribution - mean diff (within - between)")
          cax[i, 3].set_xlabel("Mean difference")
          cax[i, 3].set_ylabel("Frequency")
          cax[i, 3].axvline(res.statistic, color="black")

f.tight_layout()
f1.tight_layout()

# compare bi vs mono stimuli both within hemisphere and between hemisphere
high_corr_df = corr_df.query('cell_1_snr > 0.4 and cell_2_snr > 0.4')
f, ax = plt.subplots(len(good_files), 4, figsize=(10,8), sharex='col')
f1, ax1 = plt.subplots(len(good_files), 4, figsize=(10,8), sharex='col')
figs = [f, f1]
axs = [ax, ax1]
min_all = min([high_corr_df['noise_corr_mono'].min(), high_corr_df['noise_corr_bi'].min()])
max_all = min([high_corr_df['noise_corr_mono'].max(), high_corr_df['noise_corr_bi'].max()])
# make all histogram bins the same
bins = np.linspace(min_all, max_all, 21)
wvb = ['within', 'between']
for i, parmfile in enumerate(good_files):
     temp_parm = parmfile
     # create temp df for the current parmfile
     parm_df = high_corr_df.query('parmfile == @parmfile')

     for j, connection in enumerate(wvb):
          cax = axs[j]
          cfig = figs[j]
          # plot siteid on left y axis
          # plot siteid on left and areas
          divider = make_axes_locatable(cax[i, 0])
          div = divider.append_axes("left", size="5%", pad=0.5)
          xax = div.get_xaxis()
          xax.set_visible(False)
          div.set_yticklabels([])
          div.set_yticks([])
          ProbeA = 'A'
          ProbeB = 'B'
          div.set_ylabel(f"{parmfile[-17:-10]} \n {parmfile_area[parmfile][ProbeA]}-{parmfile_area[parmfile][ProbeB]}")

          if connection == 'within':
               # noise correlations mono
               nAA_mono = parm_df[parm_df['probe_pair'] == 'AA']['noise_corr_mono'].to_numpy()
               nBB_mono = parm_df[parm_df['probe_pair'] == 'BB']['noise_corr_mono'].to_numpy()
               n_mono = np.concatenate((nAA_mono, nBB_mono))
               # noise correlations bi
               nAA_bi = parm_df[parm_df['probe_pair'] == 'AA']['noise_corr_bi'].to_numpy()
               nBB_bi = parm_df[parm_df['probe_pair'] == 'BB']['noise_corr_bi'].to_numpy()
               n_bi = np.concatenate((nAA_bi, nBB_bi))
               # signal correlations mono
               sAA_mono = parm_df[parm_df['probe_pair'] == 'AA']['signal_corr_mono'].to_numpy()
               sBB_mono = parm_df[parm_df['probe_pair'] == 'BB']['signal_corr_mono'].to_numpy()
               s_mono = np.concatenate((sAA_mono, sBB_mono))
               # signal correlations bi
               sAA_bi = parm_df[parm_df['probe_pair'] == 'AA']['signal_corr_bi'].to_numpy()
               sBB_bi = parm_df[parm_df['probe_pair'] == 'BB']['signal_corr_bi'].to_numpy()
               s_bi = np.concatenate((sAA_bi, sBB_bi))
          elif connection == 'between':
               # noise correlations mono
               n_mono = parm_df[parm_df['probe_pair'] == 'AB']['noise_corr_mono'].to_numpy()
               # noise correlations bi
               n_bi = parm_df[parm_df['probe_pair'] == 'AB']['noise_corr_bi'].to_numpy()
               # signal correlations mono
               s_mono = parm_df[parm_df['probe_pair'] == 'AB']['signal_corr_mono'].to_numpy()
               # signal correlations bi
               s_bi = parm_df[parm_df['probe_pair'] == 'AB']['signal_corr_bi'].to_numpy()

          # noise correlations mono vs bi stim
          mono_hist, hist_bins = np.histogram(n_mono, bins=bins)
          bi_hist, hist_bins = np.histogram(n_bi, bins=bins)

          # probability density
          pdf_mono = mono_hist / sum(mono_hist)
          pdf_bi = bi_hist / sum(bi_hist)

          # cummulative density
          cdf_mono = np.cumsum(pdf_mono)
          cdf_bi = np.cumsum(pdf_bi)

          cax[i, 0].plot(hist_bins[1:], cdf_mono, color="red")
          cax[i, 0].plot(hist_bins[1:], cdf_bi, color="blue")
          cax[i, 0].legend((f"mono n={len(n_mono)}", f"bi n={len(n_bi)}"))
          cax[i, 0].set_xlabel(f"noise corr {connection}")
          cax[i, 0].set_ylabel("CDF")

          # permutation test difference between AA+BB and AB
          res = permutation_test((n_mono, n_bi), statistic, n_resamples=1000,
                                 vectorized=True, alternative='two-sided',
                                 random_state=rng)
          print("true statistic " + str(res.statistic))
          print("pvalue " + str(res.pvalue))
          cax[i, 1].hist(res.null_distribution, bins=50, color="grey")
          cax[i, 1].text(0, 45, s=f"pvalue:{np.round(res.pvalue, decimals=3)}")
          cax[i, 1].set_title("Null distribution - mean diff (within - between)")
          cax[i, 1].set_xlabel("Mean difference")
          cax[i, 1].set_ylabel("Frequency")
          cax[i, 1].axvline(res.statistic, color="black")

          # repeat above for signal
          # signal correlations mono vs bi stim
          mono_hist, hist_bins = np.histogram(s_mono, bins=bins)
          bi_hist, hist_bins = np.histogram(s_bi, bins=bins)

          # probability density
          pdf_mono = mono_hist / sum(mono_hist)
          pdf_bi = bi_hist / sum(bi_hist)

          # cummulative density
          cdf_mono = np.cumsum(pdf_mono)
          cdf_bi = np.cumsum(pdf_bi)

          cax[i, 2].plot(hist_bins[1:], cdf_mono, color="red")
          cax[i, 2].plot(hist_bins[1:], cdf_bi, color="blue")
          cax[i, 2].legend((f"mono n={len(s_mono)}", f"bi n={len(s_bi)}"))
          cax[i, 2].set_xlabel(f"signal corr {connection}")
          cax[i, 2].set_ylabel("CDF")

          # permutation test difference between AA+BB and AB
          res = permutation_test((s_mono, s_bi), statistic, n_resamples=1000,
                                 vectorized=True, alternative='two-sided',
                                 random_state=rng)
          print("true statistic " + str(res.statistic))
          print("pvalue " + str(res.pvalue))
          cax[i, 3].hist(res.null_distribution, bins=50, color="grey")
          cax[i, 3].text(0, 45, s=f"pvalue:{np.round(res.pvalue, decimals=3)}")
          cax[i, 3].set_title("Null distribution - mean diff (within - between)")
          cax[i, 3].set_xlabel("Mean difference")
          cax[i, 3].set_ylabel("Frequency")
          cax[i, 3].axvline(res.statistic, color="black")

f.tight_layout()
f1.tight_layout()

# TODO - plot signal vs noise correlation scatter for both within and between areas
# f, ax = plt.subplots(len(good_files), 4, figsize=(10,8), sharex='col')
# for j, parmfile in enumerate(good_files):
#      for i, label in enumerate(slabels):
#           if label == 'mono':
#                # plot siteid on left and areas
#                divider = make_axes_locatable(ax[j, 0])
#                div = divider.append_axes("left", size="5%", pad=0.5)
#                xax = div.get_xaxis()
#                xax.set_visible(False)
#                div.set_yticklabels([])
#                div.set_yticks([])
#                div.set_ylabel(f"{parmfile[-17:-10]} - {parmfile_area[parmfile]}")
#
#                # make a histogram of pairwise cc
#                bins = np.linspace(min_all, max_all, 21)
#                nAA_pairs = nAA_all_high[parmfile][i][np.triu_indices(nAA_all_high[parmfile][i].shape[0], k=1)]
#                nBB_pairs = nBB_all_high[parmfile][i][np.triu_indices(nBB_all_high[parmfile][i].shape[0], k=1)]
#                nAB_pairs = nAB_all_high[parmfile][i].flatten()
#                AA_hist, hist_bins = np.histogram(nAA_pairs, bins=bins)
#                AB_hist = np.histogram(nAB_pairs, bins=bins)[0]
#                BB_hist = np.histogram(nBB_pairs, bins=bins)[0]
#                # probability density
#                pdf_AA = AA_hist / sum(AA_hist)
#                pdf_BB = BB_hist / sum(BB_hist)
#                pdf_AB = AB_hist / sum(AB_hist)
#
#                # cummulative density
#                cdf_AA = np.cumsum(pdf_AA)
#                cdf_BB = np.cumsum(pdf_BB)
#                cdf_AB = np.cumsum(pdf_AB)
#
#                # h = np.stack(h, axis=1).astype(float)
#                # h /= h.sum(axis=0, keepdims=True)
#                ax[j, 0].plot(hist_bins[1:], cdf_AA, color="red")
#                ax[j, 0].plot(hist_bins[1:], cdf_BB, color='maroon')
#                ax[j, 0].plot(hist_bins[1:], cdf_AB, color="blue")
#                ax[j, 0].legend((f"AA n={len(nAA_pairs)}", f"BB n={len(nBB_pairs)}", f"AB n={len(nAB_pairs)}"))
#                ax[j, 0].set_xlabel("Noise correlation")
#                ax[j, 0].set_ylabel("CDF")
#
#                same_hemisphere = np.concatenate(
#                     (nAA_all_high[parmfile][i][np.triu_indices(nAA_all_high[parmfile][i].shape[0], k=1)],
#                      nBB_all_high[parmfile][i][np.triu_indices(nBB_all_high[parmfile][i].shape[0], k=1)]))
#                between_hemisphere = nAB_all_high[parmfile][i].flatten()
#                res = permutation_test((same_hemisphere, between_hemisphere), statistic, n_resamples=1000,
#                                       vectorized=True, alternative='two-sided',
#                                       random_state=rng)
#                print("true statistic " + str(res.statistic))
#                print("pvalue " + str(res.pvalue))
#                ax[j, 1].hist(res.null_distribution, bins=50, color="grey")
#                ax[j, 1].text(0, 45, s = f"pvalue:{np.round(res.pvalue, decimals = 3)}")
#                ax[j, 1].set_title("Null distribution - mean diff (within - between)")
#                ax[j, 1].set_xlabel("Mean difference")
#                ax[j, 1].set_ylabel("Frequency")
#                ax[j, 1].axvline(res.statistic, color="black")
#
#                # make a histogram of pairwise cc
#                bins = np.linspace(min_all, max_all, 21)
#                sAA_pairs = sAA_all_high[parmfile][i][np.triu_indices(sAA_all_high[parmfile][i].shape[0], k=1)]
#                sBB_pairs = sBB_all_high[parmfile][i][np.triu_indices(sBB_all_high[parmfile][i].shape[0], k=1)]
#                sAB_pairs = sAB_all_high[parmfile][i].flatten()
#                AA_hist, hist_bins = np.histogram(sAA_pairs, bins=bins)
#                AB_hist = np.histogram(sAB_pairs, bins=bins)[0]
#                BB_hist = np.histogram(sBB_pairs, bins=bins)[0]
#
#                # probability density
#                pdf_AA = AA_hist / sum(AA_hist)
#                pdf_BB = BB_hist / sum(BB_hist)
#                pdf_AB = AB_hist / sum(AB_hist)
#
#                # cummulative density
#                cdf_AA = np.cumsum(pdf_AA)
#                cdf_BB = np.cumsum(pdf_BB)
#                cdf_AB = np.cumsum(pdf_AB)
#
#                # h = np.stack(h, axis=1).astype(float)
#                # h /= h.sum(axis=0, keepdims=True)
#                ax[j, 2].plot(hist_bins[1:], cdf_AA, color="red")
#                ax[j, 2].plot(hist_bins[1:], cdf_BB, color='maroon')
#                ax[j, 2].plot(hist_bins[1:], cdf_AB, color="blue")
#                ax[j, 2].legend((f"AA n={len(sAA_pairs)}", f"BB n={len(sBB_pairs)}", f"AB n={len(sAB_pairs)}"))
#                ax[j, 2].set_xlabel("Signal correlation")
#                ax[j, 2].set_ylabel("CDF")
#
#                same_hemisphere = np.concatenate(
#                     (sAA_all_high[parmfile][i][np.triu_indices(sAA_all_high[parmfile][i].shape[0], k=1)],
#                      sBB_all_high[parmfile][i][np.triu_indices(sBB_all_high[parmfile][i].shape[0], k=1)]))
#                between_hemisphere = sAB_all_high[parmfile][i].flatten()
#                res = permutation_test((same_hemisphere, between_hemisphere), statistic, n_resamples=1000,
#                                       vectorized=True, alternative='two-sided',
#                                       random_state=rng)
#                print("true statistic " + str(res.statistic))
#                print("pvalue " + str(res.pvalue))
#                ax[j, 3].hist(res.null_distribution, bins=50, color="grey")
#                ax[j, 3].text(0, 45, s = f"pvalue:{np.round(res.pvalue, decimals = 3)}")
#                ax[j, 3].set_title("Null distribution - mean diff (within - between)")
#                ax[j, 3].set_xlabel("Value of Statistic")
#                ax[j, 3].set_ylabel("Frequency")
#                ax[j, 3].axvline(res.statistic, color="black")
#           else:
#                continue
# plt.tight_layout()

# for parmfile in good_files:
#      for i, label in enumerate(slabels):
#           if label == 'mono':
#                same_hemisphere = np.concatenate(
#                (nAA_all_high[parmfile][i][np.triu_indices(nAA_all_high[parmfile][i].shape[0], k=1)],
#                 nBB_all_high[parmfile][i][np.triu_indices(nBB_all_high[parmfile][i].shape[0], k=1)]))
#                between_hemisphere = nAB_all_high[parmfile][i].flatten()
#                res = permutation_test((same_hemisphere, between_hemisphere), statistic, n_resamples=1000,
#                        vectorized=True, alternative='two-sided',
#                        random_state=rng)
#                print("true statistic " + str(res.statistic))
#                print("pvalue " + str(res.pvalue))
#                f, ax = plt.subplots(1,1, figsize=(5,5))
#                ax.hist(res.null_distribution, bins=50)
#                ax.set_title("Permutation distribution of test statistic")
#                ax.set_xlabel("Value of Statistic")
#                ax.set_ylabel("Frequency")
#                ax.axvline(res.statistic)
#
# all_pairs = np.concatenate(all_pairs)
#
# f, ax = plt.subplots(3, 3, figsize=(5, 7))
# # signal correlations for high SNR neurons
# for stim_eps, lbl in zip(stim_epochs, bi_label):
#      AA = []
#      AB = []
#      BB = []
#      for s in stim_eps:
#           # extract stim epochs
#           vA = respA_high.extract_epoch(s)
#           vB = respB_high.extract_epoch(s)
#
#           # take PSTH of each neuron
#           vA = vA.mean(axis=0)
#           vB = vB.mean(axis=0)
#
#           # # change to neurons by trials by time
#           # vA = vA.transpose([1, 0, 2])
#           # vB = vB.transpose([1, 0, 2])
#           #
#           # # concatenate trials
#           # vA = np.reshape(vA, [vA.shape[0], -1])
#           # vB = np.reshape(vB, [vB.shape[0], -1])
#
#           # subtract mean firing rate for each cell
#           vA -= vA.mean(axis=1, keepdims=True)
#           vB -= vB.mean(axis=1, keepdims=True)
#
#           # find std dev
#           sA = vA.std(axis=1, keepdims=True)
#           sB = vB.std(axis=1, keepdims=True)
#
#           # divide by standard deviation
#           vA /= (sA + (sA == 0))
#           vB /= (sB + (sB == 0))
#
#           # take length of samples
#           N = vB.shape[1]
#
#           # take dot prodouct of AA, AB, BB to compute pairwise correlations and append them to lists
#           AA.append((vA @ vA.T) / N)
#           AB.append((vA @ vB.T) / N)
#           BB.append((vB @ vB.T) / N)
#
#      # quick plot to check that noise corr looks similar across stimuli
#      # f2, ax1 = plt.subplots(3, 4)
#      # for i in range(len(AA)):
#      #      cmat = AA[i]
#      #      np.fill_diagonal(cmat, 0)
#      #      ax1[0, i].imshow(cmat)
#      # for i in range(len(AB)):
#      #      cmat = AB[i]
#      #      np.fill_diagonal(cmat, 0)
#      #      ax1[1, i].imshow(cmat)
#      # for i in range(len(BB)):
#      #      cmat = BB[i]
#      #      np.fill_diagonal(cmat, 0)
#      #      ax1[2, i].imshow(cmat)
#
#      AA = np.mean(np.stack(AA, axis=2), axis=2)
#      AB = np.mean(np.stack(AB, axis=2), axis=2)
#      BB = np.mean(np.stack(BB, axis=2), axis=2)
#      np.fill_diagonal(AA, 0)
#      np.fill_diagonal(BB, 0)
#
#      if lbl == 'mono':
#           imopts = {'cmap': 'gray', 'origin': 'lower', 'vmin': -np.abs(AA).max(), 'vmax': np.abs(AA).max()}
#           ax[1, 0].imshow(AA, **imopts)
#           ax[1, 0].set_title('A x A')
#           ax[1, 0].set_ylabel('Probe A unit')
#           ax[1, 1].imshow(AB, **imopts)
#           ax[1, 1].set_title('A x B')
#           ax[2, 1].imshow(BB, **imopts)
#           ax[2, 1].set_title('B x B')
#           ax[2, 1].set_xlabel('Probe B unit')
#
#      # ax[0,0].plot(AA.std(axis=0))
#      # ax[0,1].plot(AB.std(axis=0))
#      # ax[1,2].plot(AB.std(axis=1),np.arange(AB.shape[0]))
#      # ax[2,2].plot(BB.std(axis=1),np.arange(BB.shape[0]))
#      ax[0, 0].plot(AA.mean(axis=0), label=lbl)
#      ax[0, 1].plot(AB.mean(axis=0))
#      ax[1, 2].plot(AB.mean(axis=1), np.arange(AB.shape[0]))
#      ax[2, 2].plot(BB.mean(axis=1), np.arange(BB.shape[0]))
#
#      bins = np.linspace(-0.5, 0.6, 21)
#
#      # make a histogram of pairwise cc
#      h = [np.histogram(AA[np.triu_indices(AA.shape[0], k=1)], bins=bins)[0],
#           np.histogram(AB.flatten(), bins=bins)[0],
#           np.histogram(BB[np.triu_indices(BB.shape[0], k=1)], bins=bins)[0]]
#      h = np.stack(h, axis=1).astype(float)
#      h /= h.sum(axis=0, keepdims=True)
#      ax[2, 0].plot((bins[1:] + bins[:-1]) / 2, h)
#      ax[2, 0].legend(('AA', 'AB', 'BB'))
#      f.suptitle(os.path.basename(parmfile) + " " + lbl)
#
# ax[0, 0].legend()
# plt.tight_layout()
#
#
# f, ax = plt.subplots(3, 3, figsize=(5,7))
#
# # quick noise correlations run
# # loop through mono or bilateral stim epochs - should be the same if we subtract auditory response?
# for stim_eps, lbl in zip(stim_epochs, bi_label):
#      AA=[]
#      AB=[]
#      BB=[]
#      for s in stim_eps:
#           # extract stim epochs
#           rA=respA.extract_epoch(s)
#           rB=respB.extract_epoch(s)
#
#           # subtract mean auditory response to look at noise correlations?
#           vA = rA-rA.mean(axis=0,keepdims=True)
#           vB = rB-rB.mean(axis=0,keepdims=True)
#
#           # change to neurons by trials by time
#           vA=vA.transpose([1, 0, 2])
#           vB = vB.transpose([1, 0, 2])
#
#           # concatenate trials
#           vA=np.reshape(vA, [vA.shape[0], -1])
#           vB = np.reshape(vB, [vB.shape[0], -1])
#
#           # subtract mean firing rate
#           vA-=vA.mean(axis=1, keepdims=True)
#           vB -= vB.mean(axis=1, keepdims=True)
#
#           # find std dev
#           sA = vA.std(axis=1, keepdims=True)
#           sB = vB.std(axis=1, keepdims=True)
#
#           # divide by standard deviation
#           vA/= (sA+(sA==0))
#           vB/=(sB+(sB==0))
#
#           # take length of samples
#           N=vB.shape[1]
#
#           # take dot prodouct of AA, AB, BB to compute pairwise correlations and append them to lists
#           AA.append((vA @ vA.T)/N)
#           AB.append((vA @ vB.T)/N)
#           BB.append((vB @ vB.T)/N)
#
#      # quick plot to check that noise corr looks similar across stimuli
#      # f2, ax1 = plt.subplots(3, 4)
#      # for i in range(len(AA)):
#      #      cmat = AA[i]
#      #      np.fill_diagonal(cmat, 0)
#      #      ax1[0, i].imshow(cmat)
#      # for i in range(len(AB)):
#      #      cmat = AB[i]
#      #      np.fill_diagonal(cmat, 0)
#      #      ax1[1, i].imshow(cmat)
#      # for i in range(len(BB)):
#      #      cmat = BB[i]
#      #      np.fill_diagonal(cmat, 0)
#      #      ax1[2, i].imshow(cmat)
#
#      AA=np.mean(np.stack(AA, axis=2),axis=2)
#      AB=np.mean(np.stack(AB, axis=2),axis=2)
#      BB=np.mean(np.stack(BB, axis=2),axis=2)
#      np.fill_diagonal(AA,0)
#      np.fill_diagonal(BB,0)
#
#      if lbl=='mono':
#           imopts={'cmap':'gray', 'origin':'lower', 'vmin': -0.25, 'vmax': 0.25}
#           ax[1,0].imshow(AA, **imopts)
#           ax[1,0].set_title('A x A')
#           ax[1,0].set_ylabel('Probe A unit')
#           ax[1,1].imshow(AB, **imopts)
#           ax[1,1].set_title('A x B')
#           ax[2,1].imshow(BB, **imopts)
#           ax[2,1].set_title('B x B')
#           ax[2,1].set_xlabel('Probe B unit')
#
#      #ax[0,0].plot(AA.std(axis=0))
#      #ax[0,1].plot(AB.std(axis=0))
#      #ax[1,2].plot(AB.std(axis=1),np.arange(AB.shape[0]))
#      #ax[2,2].plot(BB.std(axis=1),np.arange(BB.shape[0]))
#      ax[0,0].plot(AA.mean(axis=0), label=lbl)
#      ax[0,1].plot(AB.mean(axis=0))
#      ax[1,2].plot(AB.mean(axis=1),np.arange(AB.shape[0]))
#      ax[2,2].plot(BB.mean(axis=1),np.arange(BB.shape[0]))
#
#      bins=np.linspace(-1, 1, 21)
#
#      h = [np.histogram(AA[np.triu_indices(AA.shape[0], k=1)], bins=bins)[0],
#           np.histogram(AB.flatten(), bins=bins)[0],
#           np.histogram(BB[np.triu_indices(BB.shape[0], k=1)], bins=bins)[0]]
#      h = np.stack(h, axis=1).astype(float)
#      h /= h.sum(axis=0, keepdims=True)
#      ax[2,0].plot((bins[1:]+bins[:-1])/2,h)
#      ax[2,0].legend(('AA','AB','BB'))
#      f.suptitle(os.path.basename(parmfile) +" " + lbl)
#
#
# ax[0,0].legend()
# plt.tight_layout()
#
# f, ax = plt.subplots(3, 3)
#
# # quick signal correlations
# # loop through mono or bilateral stim epochs - should be the same if we subtract auditory response?
# for stim_eps, lbl in zip(stim_epochs, bi_label):
#      AA = []
#      AB = []
#      BB = []
#      for s in stim_eps:
#           # extract stim epochs
#           rA = respA.extract_epoch(s)
#           rB = respB.extract_epoch(s)
#
#           # subtract mean auditory response to look at noise correlations?
#           vA = rA # - rA.mean(axis=0, keepdims=True)
#           vB = rB # - rB.mean(axis=0, keepdims=True)
#
#           # # take PSTH of each neuron
#           # vA = vA.mean(axis=0)
#           # vB = vB.mean(axis=0)
#
#           # change to neurons by trials by time
#           vA = vA.transpose([1, 0, 2])
#           vB = vB.transpose([1, 0, 2])
#
#           # concatenate trials
#           vA = np.reshape(vA, [vA.shape[0], -1])
#           vB = np.reshape(vB, [vB.shape[0], -1])
#
#           # subtract mean firing rate for each cell
#           vA -= vA.mean(axis=1, keepdims=True)
#           vB -= vB.mean(axis=1, keepdims=True)
#
#           # find std dev
#           sA = vA.std(axis=1, keepdims=True)
#           sB = vB.std(axis=1, keepdims=True)
#
#           # divide by standard deviation
#           vA /= (sA + (sA == 0))
#           vB /= (sB + (sB == 0))
#
#           # take length of samples
#           N = vB.shape[1]
#
#           # take dot prodouct of AA, AB, BB to compute pairwise correlations and append them to lists
#           AA.append((vA @ vA.T) / N)
#           AB.append((vA @ vB.T) / N)
#           BB.append((vB @ vB.T) / N)
#
#      # quick plot to check that noise corr looks similar across stimuli
#      # f2, ax1 = plt.subplots(3, 4)
#      # for i in range(len(AA)):
#      #      cmat = AA[i]
#      #      np.fill_diagonal(cmat, 0)
#      #      ax1[0, i].imshow(cmat)
#      # for i in range(len(AB)):
#      #      cmat = AB[i]
#      #      np.fill_diagonal(cmat, 0)
#      #      ax1[1, i].imshow(cmat)
#      # for i in range(len(BB)):
#      #      cmat = BB[i]
#      #      np.fill_diagonal(cmat, 0)
#      #      ax1[2, i].imshow(cmat)
#
#      AA = np.mean(np.stack(AA, axis=2), axis=2)
#      AB = np.mean(np.stack(AB, axis=2), axis=2)
#      BB = np.mean(np.stack(BB, axis=2), axis=2)
#      np.fill_diagonal(AA, 0)
#      np.fill_diagonal(BB, 0)
#
#      if lbl == 'mono':
#           imopts = {'cmap': 'gray', 'origin': 'lower', 'vmin': -np.abs(AA).max(), 'vmax': np.abs(AA).max()}
#           ax[1, 0].imshow(AA, **imopts)
#           ax[1, 0].set_title('A x A')
#           ax[1, 0].set_ylabel('Probe A unit')
#           ax[1, 1].imshow(AB, **imopts)
#           ax[1, 1].set_title('A x B')
#           ax[2, 1].imshow(BB, **imopts)
#           ax[2, 1].set_title('B x B')
#           ax[2, 1].set_xlabel('Probe B unit')
#
#      # ax[0,0].plot(AA.std(axis=0))
#      # ax[0,1].plot(AB.std(axis=0))
#      # ax[1,2].plot(AB.std(axis=1),np.arange(AB.shape[0]))
#      # ax[2,2].plot(BB.std(axis=1),np.arange(BB.shape[0]))
#      ax[0, 0].plot(AA.mean(axis=0), label=lbl)
#      ax[0, 1].plot(AB.mean(axis=0))
#      ax[1, 2].plot(AB.mean(axis=1), np.arange(AB.shape[0]))
#      ax[2, 2].plot(BB.mean(axis=1), np.arange(BB.shape[0]))
#
#      bins = np.linspace(-0.5, 0.5, 21)
#
#      # make a histogram of pairwise cc
#      h = [np.histogram(AA[np.triu_indices(AA.shape[0], k=1)], bins=bins)[0],
#           np.histogram(AB.flatten(), bins=bins)[0],
#           np.histogram(BB[np.triu_indices(BB.shape[0], k=1)], bins=bins)[0]]
#      h = np.stack(h, axis=1).astype(float)
#      h /= h.sum(axis=0, keepdims=True)
#      ax[2, 0].plot((bins[1:] + bins[:-1]) / 2, h)
#      ax[2, 0].legend(('AA', 'AB', 'BB'))
#      f.suptitle(os.path.basename(parmfile) + " " + lbl)
#
# ax[0, 0].legend()
# plt.tight_layout()

from nems_lbhb.plots import ftc_heatmap

f,ax =plt.subplots(1,2)

mua=False
probe='B'
fs=100
smooth_win=5
resp_len=0.1
siteid=ex.siteid.split("_")[0][:7]
ftc_heatmap(siteid, mua=mua, probe='A', fs=fs,
       smooth_win=smooth_win, ax=ax[0])
ax[0].set_title('Probe A')
ftc_heatmap(siteid, mua=mua, probe='B', fs=fs,
       smooth_win=smooth_win, ax=ax[1])
ax[1].set_title('Probe B')
f.suptitle(siteid)

ll = []

