import os

import matplotlib.colors
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
import nems
from nems.models import Model
from nems.layers import WeightChannels, LevelShift, DoubleExponential, FiniteImpulseResponse, RectifiedLinear
from nems.preprocessing import (
    indices_by_fraction, split_at_indices, JackknifeIterator)
from nems.tools.json import save_model
from matplotlib.cm import get_cmap
from pathlib import Path

#plot params
font_size=8
params = {'legend.fontsize': font_size-2,
          'figure.figsize': (8, 6),
          'axes.labelsize': font_size,
          'axes.titlesize': font_size,
          'axes.spines.right': False,
          'axes.spines.top': False,
          'xtick.labelsize': font_size,
          'ytick.labelsize': font_size,
          'pdf.fonttype': 42,
          'ps.fonttype': 42}
plt.rcParams.update(params)

# model save path
model_save_path = Path('/auto/users/wingertj/models/')
# dataframe save path
data_save_path = Path('/auto/users/wingertj/data/')

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
BNT_regular_parmfiles = ['/auto/data/daq/SlipperyJack/SLJ016/SLJ016a06_p_BNT.m',
                         '/auto/data/daq/SlipperyJack/SLJ019/SLJ019a13_p_BNT.m',
                         '/auto/data/daq/SlipperyJack/SLJ021/SLJ021a12_p_BNT.m',
                         '/auto/data/daq/SlipperyJack/SLJ023/SLJ023a09_p_BNT.m',
                         '/auto/data/daq/SlipperyJack/SLJ026/SLJ026a13_p_BNT.m',
                         '/auto/data/daq/SlipperyJack/SLJ033/SLJ033a06_p_BNT.m']
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


def subspace_matrices(respA, respB, source='largest', stim='both', svn=['signal', 'noise']):
     import random

     # how to pick source and target areas - if source is largest set n to largest and m to smallest population
     # else allow 0 to specify first resp and 1 to stand for second resp to set as source - n and other resp set to m

     if source == 'largest':
          # assign larger population to n and smaller to m
          if len(respA.chans) >= len(respB.chans):
               n_resp = respA
               m_resp = respB
          else:
               n_resp = respB
               m_resp = respA
     elif source == 0:
          n_resp = respA
          m_resp = respB
     elif source == 1:
          n_resp = respB
          m_resp = respA
     else:
          raise ValueError("Unrecognized source input")

     # get random chan numbers equal to size N called s for subsample
     # if n != 2xgreater than m then split n in half and subsample m as well
     if len(n_resp.chans) >= 2*len(m_resp.chans):
          # get random population of source equal to target population for predictions - s2
          s2_inds = random.sample(range(len(n_resp.chans)), len(m_resp.chans))
          s2_inds.sort()
          # use remaining source population for model input - s1
          s1_inds = [i for i in range(len(n_resp.chans)) if i not in s2_inds]
          s1_chans = np.take(n_resp.chans, s1_inds)
          s2_chans = np.take(n_resp.chans, s2_inds)

          # target inds and chans are equal to that of m_resp
          t_inds = range(len(m_resp.chans))
          t_chans = np.take(m_resp.chans, t_inds)

     else:
          print("source population is not 2x greater than target...subsample source and target to half of source")
          # get random population of source equal to target population for predictions - s2
          s2_inds = random.sample(range(len(n_resp.chans)), int(np.floor(len(n_resp.chans)/2)))
          s2_inds.sort()
          # use remaining source population for model input - s1
          s1_inds = [i for i in range(len(n_resp.chans)) if i not in s2_inds]
          s1_chans = np.take(n_resp.chans, s1_inds)
          s2_chans = np.take(n_resp.chans, s2_inds)

          # target inds and chans are equal to that of m_resp
          t_inds = random.sample(range(len(m_resp.chans)), int(np.floor(len(n_resp.chans)/2)))
          t_chans = np.take(m_resp.chans, t_inds)


     # n and m are full data where n is source and m is target population
     n_mats = []
     m_mats = []

     # (s1 is for source input, s2 is for source prediction), and t is target
     s1_mats = []
     s2_mats = []
     t_mats = []

     # get stimulus epochs
     stim_epochs=epoch.epoch_names_matching(n_resp.epochs,"^STIM_")

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

     # check if mono vs binaural or both is requested
     if stim == 'both':
          stims= [stim for l in stim_epochs for stim in l]
     elif stim == 'mono':
          stims = stim_epochs[0]
     elif stim == 'bilateral':
          stims = stim_epochs[0]
     else:
          raise ValueError("Unrecognized flag for stims")

     for i, response in enumerate(svn):
          n_mat = []
          m_mat = []
          for s in stims:
               if response == 'signal':
                    # extract stim epochs
                    vn = n_resp.extract_epoch(s)
                    vm = m_resp.extract_epoch(s)

                    # subtract mean firing rate for each cell
                    vn -= vn.mean(axis=1, keepdims=True)
                    vm -= vm.mean(axis=1, keepdims=True)

                    # change to neurons by trials by time
                    vn = vn.transpose([1, 0, 2])
                    vm = vm.transpose([1, 0, 2])

                    # concatenate trials
                    vn = np.reshape(vn, [vn.shape[0], -1])
                    vm = np.reshape(vm, [vm.shape[0], -1])

                    # append signals for each stim to list
                    n_mat.append(vn.T)
                    m_mat.append(vm.T)

               elif response == 'noise':

                    # extract stim epochs
                    rn = n_resp.extract_epoch(s)
                    rm = m_resp.extract_epoch(s)

                    # subtract mean auditory response to look at noise correlations
                    vn = rn - rn.mean(axis=0, keepdims=True)
                    vm = rm - rm.mean(axis=0, keepdims=True)

                    # change to neurons by trials by time
                    vn = vn.transpose([1, 0, 2])
                    vm = vm.transpose([1, 0, 2])

                    # concatenate trials
                    vn = np.reshape(vn, [vn.shape[0], -1])
                    vm = np.reshape(vm, [vm.shape[0], -1])

                    # subtract mean firing rate for each cell
                    vn -= vn.mean(axis=1, keepdims=True)
                    vm -= vm.mean(axis=1, keepdims=True)

                    # append signals for each stim to list
                    n_mat.append(vn.T)
                    m_mat.append(vm.T)

          n_mats.append(np.concatenate(n_mat, axis=0))
          m_mats.append(np.concatenate(m_mat, axis=0))
          s1_mats.append(np.take(np.concatenate(n_mat, axis=0), indices=s1_inds, axis=1))
          s2_mats.append(np.take(np.concatenate(n_mat, axis=0), indices=s2_inds, axis=1))
          t_mats.append(np.take(np.concatenate(m_mat, axis=0), indices=t_inds, axis=1))

     return s1_mats, s1_inds, s1_chans, s2_mats, s2_inds, s2_chans, t_mats, t_inds, t_chans

def subspace_plot_R01_layout(psthA, psthB, dfs, df_labels, parmfile_area, plot_type='internal', cmap_diverging = [], cmap_qualitative = []):
     """
     :param psthA: represenative psth for one probe
     :param psthB: representative psth for another probe
     :param dfs: list of dfs (signal vs noise or mono vs dichotic)
     :param parmfile_area: dictionary of parmfiles and area for probe {A: 'A1', B: 'PEG'}
     :param type: comparison to be made. 'internal': compare AA vs AB for each dataframe, 'external': compare df1 vs df2 AB
     :param cmap_diverging: list of cmaps used to define colors for each dataframe. diverging used for compairing AA vs AB
     :param cmap_qualitative: list of cmaps used to define colors for each dataframe. qualitative used for regional comparisons.
     :return:
     """
     # pick color maps for plotting
     if cmap_diverging:
          diverging_cmaps = cmap_diverging
     else:
          diverging_cmaps = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'coolwarm', 'bwr', 'seismic']
     if cmap_qualitative:
          qualitative_cmaps = cmap_qualitative
     else:
          qualitative_cmaps = ['Paired', 'Accent', 'Dark2', 'Set1', 'tab10', 'tab20', 'tab20b','tab20c']
     site_models = [model_name.split('_') for model_name in dfs[0]['model_names']]
     siteids = list(set([siteid[0][4:] for siteid in site_models]))
     ranks = list(set([int(siteid[1][4:]) for siteid in site_models]))
     jackknifes = list(set([jack for jack in dfs[0]['jackknife']]))
     if plot_type == 'internal':
          f, ax = plt.subplot_mosaic(
               [['.', 'B', 'C', 'D'], ['A1', 'B', 'C', 'D'], ['A2', 'B', 'C', 'D'], ['A3', 'B', 'C', 'D'],
                ['.', 'B', 'C', 'D']], width_ratios=[1.5, 1, 1, 1], height_ratios=[1, 0.75, 0.75, 0.75, 1],
               gridspec_kw={'hspace': 0.75, 'wspace': 0.75}, figsize=(8, 3.5))
          imopts = {'origin': 'lower', 'aspect': 'auto', 'cmap': 'Greys'}
          ax['A1'].imshow(stim_spec, **imopts, interpolation='gaussian')
          ax['A2'].imshow(psthA, **imopts, interpolation='None', vmax=min_val)
          ax['A3'].imshow(psthB, **imopts, interpolation='None', vmax=min_val)
          ticks = np.arange(0, len(psthA[0, :]), 2 * rasterfs)
          tick_labels = np.arange(0, len(psthA[0, :]) / rasterfs, 2 * 1)
          ax['A3'].set_xticks(ticks)
          ax['A3'].set_xticklabels(tick_labels)
          ax['A1'].sharex(ax['A3'])
          ax['A2'].sharex(ax['A3'])
          plt.setp(ax['A1'].get_xticklabels(), visible=False)
          plt.setp(ax['A2'].get_xticklabels(), visible=False)
          ax['A3'].set_xlabel("time (s)")
          ax['A1'].set_ylabel("channels")
          ax['A2'].set_ylabel("trials")
          ax['A3'].set_ylabel("trials")
          ax['A1'].set_title(stim_epochs2[0], fontsize=6)
          ax['A2'].set_title(good_A_chans[0], fontsize=6)
          ax['A3'].set_title(good_B_chans[0], fontsize=6)
          for df_num in range(len(dfs)):
               df_label = df_labels[df_num]
               site_performance_AA = []
               site_area_AB = []
               site_performance_AB = []
               site_rank_performance_AA = []
               site_rank_performance_AB = []
               for siteid in siteids:
                    site_area_AB.append(
                         ['-'.join([area['A'], area['B']]) for parm, area in parmfile_area.items() if siteid in parm][0])
                    site_df = dfs[df_num][dfs[df_num]['model_names'].str.contains(siteid)]
                    mean_rank_performance_AA = []
                    mean_rank_performance_AB = []
                    rank_performance_AA = []
                    rank_performance_AB = []
                    for rank in ranks:
                         if site_df['model_names'].str.contains('bilateral').values[0]:
                              rank_AA_df = site_df[site_df['model_names'].str.contains(f'Rank{str(rank)}_bilateral_AA')]
                              rank_AB_df = site_df[site_df['model_names'].str.contains(f'Rank{str(rank)}_bilateral_AB')]
                              stim_type = 'bilateral'
                         elif site_df['model_names'].str.contains('mono').values[0]:
                              rank_AA_df = site_df[site_df['model_names'].str.contains(f'Rank{str(rank)}_mono_AA')]
                              rank_AB_df = site_df[site_df['model_names'].str.contains(f'Rank{str(rank)}_mono_AB')]
                              stim_type = 'mono'
                         else:
                              rank_AA_df = site_df[site_df['model_names'].str.contains(f'Rank{str(rank)}_AA')]
                              rank_AB_df = site_df[site_df['model_names'].str.contains(f'Rank{str(rank)}_AB')]
                              stim_type = ''
                         mean_rank_performance_AA.append(np.nanmean(rank_AA_df['mean_model_performance']))
                         mean_rank_performance_AB.append(np.nanmean(rank_AB_df['mean_model_performance']))
                         rank_performance_AA.append(rank_AA_df['mean_model_performance'])
                         rank_performance_AB.append(rank_AB_df['mean_model_performance'])
                    site_performance_AA.append(mean_rank_performance_AA)
                    site_performance_AB.append(mean_rank_performance_AB)
                    site_rank_performance_AA.append(rank_performance_AA)
                    site_rank_performance_AB.append(rank_performance_AB)
               site_performance_AA = np.array(site_performance_AA)[:, :10]
               site_performance_AB = np.array(site_performance_AB)[:, :10]
               semAA = np.nanstd(site_performance_AA, axis=0, ddof=1) / np.sqrt(
                    np.sum(~np.isnan(site_performance_AA), axis=0))
               semAB = np.nanstd(site_performance_AB, axis=0, ddof=1) / np.sqrt(
                    np.sum(~np.isnan(site_performance_AB), axis=0))
               try:
                    cmap = get_cmap(diverging_cmaps[df_num + 1])
                    import matplotlib
                    if type(cmap) == matplotlib.colors.LinearSegmentedColormap:
                         AA_color = cmap(0)
                         AB_color = cmap(256)
                    else:
                         raise ValueError("color map is not LinearSegmentedColormap")
               except:
                    print("cmap is not matplotlib colormap...trying color names")
                    try:
                         AA_color = diverging_cmaps[0 + df_num*2]
                         AB_color = diverging_cmaps[1 + df_num*2]
                    except:
                         raise ValueError("Not enough colors listed. Check diverging cmap is linearsegementedcolormap or list of named colors")
               ax['B'].plot(ranks[:10], site_performance_AA[:, :].mean(axis=0), color=AA_color, linewidth=3, label=f'within-{df_label}-{stim_type}')
               ax['B'].plot(ranks[:10], site_performance_AB[:, :].mean(axis=0), color=AB_color, linewidth=3,
                            label=f'between-{df_label}-{stim_type}')
               ax['B'].fill_between(ranks[:10], site_performance_AA.mean(axis=0) - semAA,
                                    site_performance_AA.mean(axis=0) + semAA, alpha=0.2, color=AA_color)
               ax['B'].fill_between(ranks[:10], site_performance_AB.mean(axis=0) - semAB,
                                    site_performance_AB.mean(axis=0) + semAB, alpha=0.2, color=AB_color)
               ax['B'].set_ylabel('model performance')
               ax['B'].set_xlabel('model rank')
               ax['B'].legend()
               ax['B'].set_box_aspect(1)

               site_area_combos = set(site_area_AB)
               site_dimensionality_AA = []
               site_dimensionality_AB = []
               cmap = get_cmap(qualitative_cmaps[df_num])
               if cmap.N < 256:
                    if cmap.N < len(site_area_combos):
                         raise ValueError("Not enough colors in colormap for number of areas")
                    else:
                     area_colors = {aname: cmap(i) for i, aname in enumerate(site_area_combos)}
               else:
                    crange = np.linspace(0, 1, len(site_area_combos))
                    area_colors = {aname: cmap(crange[i]) for i, aname in enumerate(site_area_combos)}
               for sitenum in range(len(siteids)):
                    site_rank_AA = np.array(site_rank_performance_AA[sitenum])
                    site_rank_AB = np.array(site_rank_performance_AB[sitenum])
                    semAA = np.nanstd(site_rank_AA, axis=1, ddof=1) / np.sqrt(
                         np.sum(~np.isnan(site_rank_AA), axis=1))
                    semAB = np.nanstd(site_rank_AB, axis=1, ddof=1) / np.sqrt(
                         np.sum(~np.isnan(site_rank_AB), axis=1))
                    semAA = semAA[:10]
                    semAB = semAB[:10]
                    site_dimensionality_AA.append(
                         np.where(site_performance_AA[sitenum, :] > (site_performance_AA[sitenum, -1] - semAA[-1]))[0][
                              0] + 1)
                    site_dimensionality_AB.append(
                         np.where(site_performance_AB[sitenum, :] > (site_performance_AB[sitenum, -1] - semAB[-1]))[0][
                              0] + 1)
               site_dimensionality_AA = np.array(site_dimensionality_AA)
               site_dimensionality_AB = np.array(site_dimensionality_AB)
               for area in site_area_combos:
                    current_area = np.array([s_area == area for s_area in site_area_AB])
                    area_color = [area_colors[area] for t in current_area if t == True]
                    ax['C'].scatter(site_dimensionality_AA[current_area], site_dimensionality_AB[current_area], s=6,
                                    c=area_color, label=f"{area}-{df_label}-{stim_type}")
               ax['C'].legend()
               lims = (0, 10)
               ticks = np.arange(lims[0], lims[1], 2)
               ax['C'].set_ylim(lims)
               ax['C'].set_xlim(lims)
               ax['C'].set_yticks(ticks)
               ax['C'].set_xticks(ticks)
               ax['C'].set_xlabel("predictive dimensions \n - within")
               ax['C'].set_ylabel("predictive dimensions \n - between")
               ax['C'].plot(np.arange(lims[0], lims[1] + 1, 1), np.arange(lims[0], lims[1] + 1, 1), '--', color='grey')
               ax['C'].set_box_aspect(1)

               # plot performance within vs between hemispheres for number of predictive dimensions
               site_dim_performanceAA = np.array(
                    [site_performance_AA[si, di] for si, di in enumerate([9 for i in site_dimensionality_AA])])
               site_dim_performanceAB = np.array(
                    [site_performance_AB[si, di] for si, di in enumerate([9 for i in site_dimensionality_AB])])
               # site_dim_performanceAB_s = np.array([site_performance_AB_s[si, di] for si, di in enumerate([9 for i in site_dimensionality_AB_s])])
               for area in site_area_combos:
                    current_area = np.array([s_area == area for s_area in site_area_AB])
                    area_color_n = [area_colors[area] for t in current_area if t == True]
                    ax['D'].scatter(site_dim_performanceAA[current_area], site_dim_performanceAB[current_area], s=6,
                                    c=area_color_n, label=f"{area}-{df_label}-{stim_type}")
               max_perf = np.round(np.max(np.concatenate((site_dim_performanceAA, site_dim_performanceAB))), decimals=1)
               dlims = (0, max_perf + 0.1)
               dticks = np.arange(dlims[0], dlims[1] + 0.1, 0.2)
               ax['D'].set_ylim(dlims)
               ax['D'].set_xlim(dlims)
               ax['D'].set_yticks(dticks)
               ax['D'].set_xticks(dticks)
               ax['D'].set_xlabel("performance \n - within")
               ax['D'].set_ylabel("performance \n - between")
               ax['D'].plot(np.arange(dlims[0], dlims[1] + 0.1, 0.1), np.arange(dlims[0], dlims[1] + 0.1, 0.1), '--',
                            color='grey')
               ax['D'].set_box_aspect(1)
               plt.tight_layout()

     elif plot_type == 'external':
          if len(dfs) > 2:
               raise ValueError("Number of dataframes for external comparison needs to be less than 2")
          f, ax = plt.subplot_mosaic(
               [['.', 'B', 'C', 'D'], ['A1', 'B', 'C', 'D'], ['A2', 'B', 'C', 'D'], ['A3', 'B', 'C', 'D'],
                ['.', 'B', 'C', 'D']], width_ratios=[1.5, 1, 1, 1], height_ratios=[1, 0.75, 0.75, 0.75, 1],
               gridspec_kw={'hspace': 0.75, 'wspace': 0.75}, figsize=(8, 3.5))
          imopts = {'origin': 'lower', 'aspect': 'auto', 'cmap': 'Greys'}
          ax['A1'].imshow(stim_spec, **imopts, interpolation='gaussian')
          ax['A2'].imshow(psthA, **imopts, interpolation='None', vmax=min_val)
          ax['A3'].imshow(psthB, **imopts, interpolation='None', vmax=min_val)
          ticks = np.arange(0, len(psthA[0, :]), 2 * rasterfs)
          tick_labels = np.arange(0, len(psthA[0, :]) / rasterfs, 2 * 1)
          ax['A3'].set_xticks(ticks)
          ax['A3'].set_xticklabels(tick_labels)
          ax['A1'].sharex(ax['A3'])
          ax['A2'].sharex(ax['A3'])
          plt.setp(ax['A1'].get_xticklabels(), visible=False)
          plt.setp(ax['A2'].get_xticklabels(), visible=False)
          ax['A3'].set_xlabel("time (s)")
          ax['A1'].set_ylabel("channels")
          ax['A2'].set_ylabel("trials")
          ax['A3'].set_ylabel("trials")
          ax['A1'].set_title(stim_epochs2[0], fontsize=6)
          ax['A2'].set_title(good_A_chans[0], fontsize=6)
          ax['A3'].set_title(good_B_chans[0], fontsize=6)
          site_performance_AA_1 = []
          site_performance_AA_2 = []
          site_area_AB = []
          site_performance_AB_1 = []
          site_performance_AB_2 = []
          site_rank_performance_AA = []
          site_rank_performance_AB = []
          site_rank_performance_AA_s = []
          site_rank_performance_AB_s = []
          for siteid in siteids:
               site_area_AB.append(
                    ['-'.join([area['A'], area['B']]) for parm, area in parmfile_area.items() if
                     siteid in parm][0])
               site_df = dfs[df_num][dfs[df_num]['model_names'].str.contains(siteid)]
               # site_signal_df = subspace_mono_df[subspace_mono_df['model_names'].str.contains(siteid)]
               mean_rank_performance_AA_n = []
               mean_rank_performance_AB_n = []
               rank_performance_AA_n = []
               rank_performance_AB_n = []
               # mean_rank_performance_AA_s = []
               # mean_rank_performance_AB_s = []
               # rank_performance_AA_s = []
               # rank_performance_AB_s = []

               for rank in ranks:
                    rank_AA_df_n = site_noise_df[
                         site_noise_df['model_names'].str.contains(f'Rank{str(rank)}_AA')]
                    rank_AB_df_n = site_noise_df[
                         site_noise_df['model_names'].str.contains(f'Rank{str(rank)}_AB')]
                    # rank_AA_df_s = site_signal_df[site_signal_df['model_names'].str.contains(f'Rank{str(rank)}_mono_AA')]
                    # rank_AB_df_s = site_signal_df[site_signal_df['model_names'].str.contains(f'Rank{str(rank)}_mono_AB')]
                    mean_rank_performance_AA_n.append(np.nanmean(rank_AA_df_n['mean_model_performance']))
                    mean_rank_performance_AB_n.append(np.nanmean(rank_AB_df_n['mean_model_performance']))
                    rank_performance_AA_n.append(rank_AA_df_n['mean_model_performance'])
                    rank_performance_AB_n.append(rank_AB_df_n['mean_model_performance'])
                    # mean_rank_performance_AA_s.append(np.nanmean(rank_AA_df_s['mean_model_performance']))
                    # mean_rank_performance_AB_s.append(np.nanmean(rank_AB_df_s['mean_model_performance']))
                    # rank_performance_AA_s.append(rank_AA_df_s['mean_model_performance'])
                    # rank_performance_AB_s.append(rank_AB_df_s['mean_model_performance'])
               site_performance_AA_n.append(mean_rank_performance_AA_n)
               site_performance_AB_n.append(mean_rank_performance_AB_n)
               site_rank_performance_AA.append(rank_performance_AA_n)
               site_rank_performance_AB.append(rank_performance_AB_n)
               # site_performance_AA_s.append(mean_rank_performance_AA_s)
               # site_performance_AB_s.append(mean_rank_performance_AB_s)
               # site_rank_performance_AA_s.append(rank_performance_AA_s)
               # site_rank_performance_AB_s.append(rank_performance_AB_s)
          site_performance_AA_n = np.array(site_performance_AA_n)[:, :10]
          site_performance_AB_n = np.array(site_performance_AB_n)[:, :10]
          # site_performance_AA_s = np.array(site_performance_AA_s)[:, :10]
          # site_performance_AB_s = np.array(site_performance_AB_s)[:, :10]
          semAA_n = np.nanstd(site_performance_AA_n, axis=0, ddof=1) / np.sqrt(
               np.sum(~np.isnan(site_performance_AA_n), axis=0))
          semAB_n = np.nanstd(site_performance_AB_n, axis=0, ddof=1) / np.sqrt(
               np.sum(~np.isnan(site_performance_AB_n), axis=0))
          # semAA_s = np.nanstd(site_performance_AA_s, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(site_performance_AA_s), axis=0))
          # semAB_s = np.nanstd(site_performance_AB_s, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(site_performance_AB_s), axis=0))
          ax['B'].plot(ranks[:10], site_performance_AA_n[:, :].mean(axis=0), color='red', linewidth=3,
                       label='within-noise')
          ax['B'].plot(ranks[:10], site_performance_AB_n[:, :].mean(axis=0), color='blue', linewidth=3,
                       label='between-noise')
          ax['B'].fill_between(ranks[:10], site_performance_AA_n.mean(axis=0) - semAA_n,
                               site_performance_AA_n.mean(axis=0) + semAA_n, alpha=0.2, color='red')
          ax['B'].fill_between(ranks[:10], site_performance_AB_n.mean(axis=0) - semAB_n,
                               site_performance_AB_n.mean(axis=0) + semAB_n, alpha=0.2, color='blue')
          # ax['B'].plot(ranks[:10], site_performance_AA_s[:, :].mean(axis=0), color='orange', linewidth=3, label='within-signal')
          # ax['B'].plot(ranks[:10], site_performance_AB_s[:, :].mean(axis=0), color='cyan', linewidth=3, label='between-signal')
          # ax['B'].fill_between(ranks[:10], site_performance_AA_s.mean(axis=0) - semAA_s, site_performance_AA_s.mean(axis=0) + semAA_s, alpha=0.2, color ='orange')
          # ax['B'].fill_between(ranks[:10], site_performance_AB_s.mean(axis=0) - semAB_s, site_performance_AB_s.mean(axis=0) + semAB_s, alpha=0.2, color ='cyan')
          ax['B'].set_ylabel('model performance')
          ax['B'].set_xlabel('model rank')
          ax['B'].legend()
          ax['B'].set_box_aspect(1)

          site_dimensionality_AA_n = []
          site_dimensionality_AB_n = []
          # site_dimensionality_AA_s = []
          # site_dimensionality_AB_s = []

          cmap = get_cmap('Dark2')
          cmap = get_cmap('viridis')
          crange = np.linspace(0, 1, 3)
          cmap_s = get_cmap('Set2')
          colors = cmap.colors
          colors_s = cmap_s.colors
          site_area_combos = set(site_area_AB)
          area_colors = {'A1-A1': cmap(crange[0]), 'A1-PEG': cmap(crange[1]), 'PEG-PEG': cmap(crange[2]),
                         'PEG-A1': colors[3]}
          area_colors_s = {'A1-A1': colors_s[0], 'A1-PEG': colors_s[1], 'PEG-PEG': colors_s[2],
                           'PEG-A1': colors_s[3]}
          for sitenum in range(len(siteids)):
               site_rank_AA = np.array(site_rank_performance_AA[sitenum])
               site_rank_AB = np.array(site_rank_performance_AB[sitenum])
               # site_rank_AA_s = np.array(site_rank_performance_AA_s[sitenum])
               # site_rank_AB_s = np.array(site_rank_performance_AB_s[sitenum])
               semAA_n = np.nanstd(site_rank_AA, axis=1, ddof=1) / np.sqrt(
                    np.sum(~np.isnan(site_rank_AA), axis=1))
               semAB_n = np.nanstd(site_rank_AB, axis=1, ddof=1) / np.sqrt(
                    np.sum(~np.isnan(site_rank_AB), axis=1))
               # semAA_s = np.nanstd(site_rank_AA_s, axis=1, ddof=1) / np.sqrt(
               #      np.sum(~np.isnan(site_rank_AA_s), axis=1))
               # semAB_s = np.nanstd(site_rank_AB_s, axis=1, ddof=1) / np.sqrt(
               #      np.sum(~np.isnan(site_rank_AB_s), axis=1))
               semAA_n = semAA_n[:10]
               semAB_n = semAB_n[:10]
               # semAA_s = semAA_s[:10]
               # semAB_s = semAB_s[:10]
               site_dimensionality_AA_n.append(
                    np.where(site_performance_AA_n[sitenum, :] > (
                                 site_performance_AA_n[sitenum, -1] - semAA_n[-1]))[0][
                         0] + 1)
               site_dimensionality_AB_n.append(
                    np.where(site_performance_AB_n[sitenum, :] > (
                                 site_performance_AB_n[sitenum, -1] - semAB_n[-1]))[0][
                         0] + 1)
               # site_dimensionality_AA_s.append(np.where(site_performance_AA_s[sitenum, :] > (site_performance_AA_s[sitenum, -1]-semAA_s[-1]))[0][0] + 1)
               # site_dimensionality_AB_s.append(np.where(site_performance_AB_s[sitenum, :] > (site_performance_AB_s[sitenum, -1] - semAB_s[-1]))[0][0] + 1)

          site_dimensionality_AA_n = np.array(site_dimensionality_AA_n)
          site_dimensionality_AB_n = np.array(site_dimensionality_AB_n)
          # site_dimensionality_AA_s = np.array(site_dimensionality_AA_s)
          # site_dimensionality_AB_s = np.array(site_dimensionality_AB_s)
          for area in site_area_combos:
               current_area = np.array([s_area == area for s_area in site_area_AB])
               area_color_n = [area_colors[area] for t in current_area if t == True]
               # area_color_s = [area_colors_s[area] for t in current_area if t == True]
               ax['C'].scatter(site_dimensionality_AA_n[current_area], site_dimensionality_AB_n[current_area],
                               s=6,
                               c=area_color_n, label=f"{area}-noise")
               # ax['C'].scatter(site_dimensionality_AA_s[current_area], site_dimensionality_AB_s[current_area], s=6,
               #                 c=area_color_s, label=f"{area}-signal")
          ax['C'].legend()
          lims = (0, 10)
          ticks = np.arange(lims[0], lims[1], 2)
          ax['C'].set_ylim(lims)
          ax['C'].set_xlim(lims)
          ax['C'].set_yticks(ticks)
          ax['C'].set_xticks(ticks)
          ax['C'].set_xlabel("predictive dimensions \n - within")
          ax['C'].set_ylabel("predictive dimensions \n - between")
          ax['C'].plot(np.arange(lims[0], lims[1] + 1, 1), np.arange(lims[0], lims[1] + 1, 1), '--',
                       color='grey')
          ax['C'].set_box_aspect(1)

          # plot performance within vs between hemispheres for number of predictive dimensions
          site_dim_performanceAA_n = np.array(
               [site_performance_AA_n[si, di] for si, di in enumerate([9 for i in site_dimensionality_AA_n])])
          site_dim_performanceAB_n = np.array(
               [site_performance_AB_n[si, di] for si, di in enumerate([9 for i in site_dimensionality_AB_n])])
          # site_dim_performanceAB_s = np.array([site_performance_AB_s[si, di] for si, di in enumerate([9 for i in site_dimensionality_AB_s])])
          for area in site_area_combos:
               current_area = np.array([s_area == area for s_area in site_area_AB])
               area_color_n = [area_colors[area] for t in current_area if t == True]
               ax['D'].scatter(site_dim_performanceAA_n[current_area], site_dim_performanceAB_n[current_area],
                               s=6,
                               c=area_color_n, label=area)
          max_perf = np.round(np.max(np.concatenate((site_dim_performanceAA_n, site_dim_performanceAB_n))),
                              decimals=1)
          dlims = (0, max_perf + 0.1)
          dticks = np.arange(dlims[0], dlims[1] + 0.1, 0.2)
          ax['D'].set_ylim(dlims)
          ax['D'].set_xlim(dlims)
          ax['D'].set_yticks(dticks)
          ax['D'].set_xticks(dticks)
          ax['D'].set_xlabel("performance \n - within")
          ax['D'].set_ylabel("performance \n - between")
          ax['D'].plot(np.arange(dlims[0], dlims[1] + 0.1, 0.1), np.arange(dlims[0], dlims[1] + 0.1, 0.1),
                       '--',
                       color='grey')
          ax['D'].set_box_aspect(1)
          plt.tight_layout()

### crap code pulled from bilateral.py to get area and psth plots ###
parmfile_area = {}
badfiles = []
for parmfile in BNT_parmfiles:
     ex = baphy_experiment.BAPHYExperiment(parmfile=[parmfile])
     loadkey="psth.fs10"
     rasterfs = 10
     rec = ex.get_recording(loadkey=loadkey)
     # rec = ex.get_recording(loadkey=loadkey, stim='gtgram')
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
parmfile_area_bnt = {}
badfiles_bnt = []
for parmfile in BNT_regular_parmfiles:
     ex = baphy_experiment.BAPHYExperiment(parmfile=[parmfile])
     loadkey="psth.fs10"
     rasterfs = 10
     rec = ex.get_recording(loadkey=loadkey)
     # rec = ex.get_recording(loadkey=loadkey, stim='gtgram')
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
          parmfile_area_bnt[parmfile] = {"A":A_area, "B":B_area}

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
         badfiles_bnt.append(parmfile)
         continue
# load dataframe
subspace_noise_df = pd.read_pickle(data_save_path/'subspace_df.pkl')
subspace_signal_df = pd.read_pickle(data_save_path/'subspace_df_signal.pkl')
# # dichotic binaural bnt
# split bnt dataframes because code to save collapsed mono and bilateral stims together
subspace_bnt_signal_df = pd.read_pickle(data_save_path/'subspace_df_signal_mono.pkl')
subspace_signal_bi_df = subspace_bnt_signal_df[subspace_bnt_signal_df['stim'] == 'bilateral']
subspace_signal_mono_df = subspace_bnt_signal_df[subspace_bnt_signal_df['stim'] == 'mono']
subspace_bnt_noise_df = pd.read_pickle(data_save_path/'subspace_df_noise_mono.pkl')
subspace_noise_bi_df = subspace_bnt_noise_df[subspace_bnt_noise_df['stim'] == 'bilateral']
subspace_noise_mono_df = subspace_bnt_noise_df[subspace_bnt_noise_df['stim'] == 'mono']

site_models = [model_name.split('_') for model_name in subspace_noise_df['model_names']]
siteids = list(set([siteid[0][4:] for siteid in site_models]))
ranks = list(set([int(siteid[1][4:]) for siteid in site_models]))
jackknifes = list(set([jack for jack in subspace_noise_df['jackknife']]))
# for each site plot rank vs model performance
### plot represenative dual hemisphere site ###
parmfile ='/auto/data/daq/SlipperyJack/SLJ036/SLJ036a03_p_BNT.m'
ex = baphy_experiment.BAPHYExperiment(parmfile=[parmfile])
rasterfs = 50
stim_chans = 64
rec = ex.get_recording(resp=True, rasterfs=rasterfs, recache=False, stim=True, stimfmt='gtgram', channels=stim_chans)
# rec = ex.get_recording(loadkey=loadkey, stim='gtgram')
# .sortparameters.Kilosort_load_completed_job_params
resp = rec['resp'].rasterize()
stim = rec['stim']
# grab A1 units only
siteid = parmfile[-17:-10]
depth_info = baphy_io.get_depth_info(siteid=siteid)
A1_units = depth_info.loc[depth_info.isin({'area': ["A1", 'PEG']})['area']].index.tolist()
A_units = [unit for unit in A1_units if "-A-" in unit]
B_units = [unit for unit in A1_units if "-B-" in unit]
A_area = depth_info['area'][A_units[0]]
B_area = depth_info['area'][B_units[0]]
A1_units = [unit[-8:] for unit in A1_units]
# grab unit names for all units in resp that have depth info in A1
A_chans = [chan for chan in resp.chans if '-A-' in chan if chan[-8:] in A1_units]
B_chans = [chan for chan in resp.chans if '-B-' in chan if chan[-8:] in A1_units]
respA = resp.extract_channels(A_chans)
respB = resp.extract_channels(B_chans)
# compute auditory responsiveness - SNR
snrA = compute_snr_multi(respA)
snrB = compute_snr_multi(respB)
highA = snrA >= 0.15
highB = snrB >= 0.15
stim_epochs = epoch.epoch_names_matching(resp.epochs, "^STIM_")
# split stim_epochs into mono vs bilateral stims
if len(stim_epochs) > 1:
     stim_epochs = epoch.epoch_names_matching(resp.epochs, "^STIM_00") + \
                   epoch.epoch_names_matching(resp.epochs, "^STIM_NULL:1:0\+00")
     stim_epochs1 = [s for s in stim_epochs if 'NULL' in s]
     stim_epochs2 = [s for s in stim_epochs if 'NULL' not in s]
     bi_label = ['mono', 'bilateral']
     stim_epochs = [stim_epochs1, stim_epochs2]
else:
     stim_epochs = [stim_epochs]
     bi_label = ['mono']
# grab a random high snr unit to plot
random_A_ind = np.random.randint(len(np.where(highA)[0]), size=1)[0]
random_B_ind = np.random.randint(len(np.where(highB)[0]), size=1)[0]
good_unitA = np.where(highA)[0][random_A_ind]
good_unitB = np.where(highB)[0][random_B_ind]
random_A_chan = respA.chans[good_unitA]
random_B_chan = respB.chans[good_unitB]
good_B_chans = ['SLJ036a03_p_BNT.m-B-142-1', 'SLJ036a03_p_BNT.m-B-030-2']
good_A_chans = ['SLJ036a03_p_BNT.m-A-263-1']
good_unitA = [i for i, ch in enumerate(respA.chans) if ch == good_A_chans[0]]
good_unitB = [i for i, ch in enumerate(respB.chans) if ch == good_B_chans[0]]
psthA = respA.extract_epoch(stim_epochs2[0])[:, good_unitA[0], :]
psthB = respB.extract_epoch(stim_epochs2[0])[:, good_unitB[0], :]
min_val = np.min([np.min(psthA[np.nonzero(psthA)]), np.min(psthB[np.nonzero(psthB)])])
max_val = np.max([psthA.max(), psthB.max()])
stim_spec = stim._data[stim_epochs2[0]][:int(stim_chans/2), :]

# test plotting functions
dfs_bnt_signal = [subspace_signal_mono_df, subspace_signal_bi_df]
dfs_bnt_noise = [subspace_noise_mono_df, subspace_noise_bi_df]
mono_svn = [subspace_signal_mono_df, subspace_noise_mono_df]
bi_svn = [subspace_signal_bi_df, subspace_noise_bi_df]

# signal predictions mono vs bi
subspace_plot_R01_layout(psthA, psthB, dfs_bnt_signal, ['signal', 'signal'], parmfile_area_bnt, plot_type='internal',
                         cmap_diverging = ['blue', 'red', 'teal', 'salmon'], cmap_qualitative = ['viridis', 'Dark2'])
# noise predictions mono vs bi
subspace_plot_R01_layout(psthA, psthB, dfs_bnt_noise, ['noise', 'noise'], parmfile_area_bnt, plot_type='internal',
                         cmap_diverging = ['blue', 'red', 'teal', 'salmon'], cmap_qualitative = ['viridis', 'Dark2'])
# signal vs noise mono
subspace_plot_R01_layout(psthA, psthB, mono_svn, ['signal', 'noise'], parmfile_area_bnt, plot_type='external',
                         cmap_diverging = ['blue', 'red', 'teal', 'salmon'], cmap_qualitative = ['viridis', 'Dark2'])
# signal vs noise bi
subspace_plot_R01_layout(psthA, psthB, bi_svn, ['signal', 'noise'], parmfile_area_bnt, plot_type='external',
                         cmap_diverging = ['blue', 'red', 'teal', 'salmon'], cmap_qualitative = ['viridis', 'Dark2'])


## plot for between signal vs noise
f, ax = plt.subplot_mosaic([['.', 'B', 'C', 'D'], ['A1', 'B', 'C', 'D'], ['A2', 'B', 'C', 'D'], ['A3', 'B', 'C', 'D'], ['.', 'B', 'C', 'D']], width_ratios=[1.5,1,1,1], height_ratios=[1, 0.75, 0.75, 0.75, 1], gridspec_kw={'hspace':0.75, 'wspace':0.75}, figsize=(8,3.5))
# gridspec_kw={'hspace':0.01, 'wspace':0.5}
imopts = {'origin': 'lower', 'aspect':'auto', 'cmap':'Greys'}
ax['A1'].imshow(stim_spec, **imopts, interpolation='gaussian')
ax['A2'].imshow(psthA, **imopts, interpolation='None', vmax=min_val)
ax['A3'].imshow(psthB, **imopts, interpolation='None', vmax=min_val)
ticks = np.arange(0, len(psthA[0, :]), 2*rasterfs)
tick_labels = np.arange(0, len(psthA[0, :])/rasterfs, 2*1)
# ax['A1'].set_xticks(ticks)
# ax['A1'].set_xticklabels(tick_labels)
# ax['A2'].set_xticks(ticks)
# ax['A2'].set_xticklabels(tick_labels)
ax['A3'].set_xticks(ticks)
ax['A3'].set_xticklabels(tick_labels)
# ax['A1'].get_shared_x_axes().joined(ax['A1'], ax['A3'])
# ax['A2'].get_shared_x_axes().joined(ax['A2'], ax['A3'])
ax['A1'].sharex(ax['A3'])
ax['A2'].sharex(ax['A3'])
plt.setp(ax['A1'].get_xticklabels(), visible=False)
plt.setp(ax['A2'].get_xticklabels(), visible=False)
ax['A3'].set_xlabel("time (s)")
ax['A1'].set_ylabel("channels")
ax['A2'].set_ylabel("trials")
ax['A3'].set_ylabel("trials")
ax['A1'].set_title(stim_epochs2[0], fontsize=6)
ax['A2'].set_title(good_A_chans[0], fontsize=6)
ax['A3'].set_title(good_B_chans[0], fontsize=6)

site_performance_AA_n = []
# site_performance_AA_s = []
site_area_AB = []
site_performance_AB_n = []
# site_performance_AB_s = []
site_rank_performance_AA = []
site_rank_performance_AB = []
# site_rank_performance_AA_s = []
# site_rank_performance_AB_s = []
for siteid in siteids:
     site_area_AB.append(['-'.join([area['A'], area['B']]) for parm, area in parmfile_area.items() if siteid in parm][0])
     site_noise_df = subspace_noise_df[subspace_noise_df['model_names'].str.contains(siteid)]
     # site_signal_df = subspace_mono_df[subspace_mono_df['model_names'].str.contains(siteid)]
     mean_rank_performance_AA_n = []
     mean_rank_performance_AB_n = []
     rank_performance_AA_n = []
     rank_performance_AB_n = []
     # mean_rank_performance_AA_s = []
     # mean_rank_performance_AB_s = []
     # rank_performance_AA_s = []
     # rank_performance_AB_s = []

     for rank in ranks:
          rank_AA_df_n = site_noise_df[site_noise_df['model_names'].str.contains(f'Rank{str(rank)}_AA')]
          rank_AB_df_n = site_noise_df[site_noise_df['model_names'].str.contains(f'Rank{str(rank)}_AB')]
          # rank_AA_df_s = site_signal_df[site_signal_df['model_names'].str.contains(f'Rank{str(rank)}_mono_AA')]
          # rank_AB_df_s = site_signal_df[site_signal_df['model_names'].str.contains(f'Rank{str(rank)}_mono_AB')]
          mean_rank_performance_AA_n.append(np.nanmean(rank_AA_df_n['mean_model_performance']))
          mean_rank_performance_AB_n.append(np.nanmean(rank_AB_df_n['mean_model_performance']))
          rank_performance_AA_n.append(rank_AA_df_n['mean_model_performance'])
          rank_performance_AB_n.append(rank_AB_df_n['mean_model_performance'])
          # mean_rank_performance_AA_s.append(np.nanmean(rank_AA_df_s['mean_model_performance']))
          # mean_rank_performance_AB_s.append(np.nanmean(rank_AB_df_s['mean_model_performance']))
          # rank_performance_AA_s.append(rank_AA_df_s['mean_model_performance'])
          # rank_performance_AB_s.append(rank_AB_df_s['mean_model_performance'])
     site_performance_AA_n.append(mean_rank_performance_AA_n)
     site_performance_AB_n.append(mean_rank_performance_AB_n)
     site_rank_performance_AA.append(rank_performance_AA_n)
     site_rank_performance_AB.append(rank_performance_AB_n)
     # site_performance_AA_s.append(mean_rank_performance_AA_s)
     # site_performance_AB_s.append(mean_rank_performance_AB_s)
     # site_rank_performance_AA_s.append(rank_performance_AA_s)
     # site_rank_performance_AB_s.append(rank_performance_AB_s)
site_performance_AA_n = np.array(site_performance_AA_n)[:, :10]
site_performance_AB_n = np.array(site_performance_AB_n)[:, :10]
# site_performance_AA_s = np.array(site_performance_AA_s)[:, :10]
# site_performance_AB_s = np.array(site_performance_AB_s)[:, :10]
semAA_n = np.nanstd(site_performance_AA_n, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(site_performance_AA_n), axis=0))
semAB_n = np.nanstd(site_performance_AB_n, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(site_performance_AB_n), axis=0))
# semAA_s = np.nanstd(site_performance_AA_s, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(site_performance_AA_s), axis=0))
# semAB_s = np.nanstd(site_performance_AB_s, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(site_performance_AB_s), axis=0))
ax['B'].plot(ranks[:10], site_performance_AA_n[:, :].mean(axis=0), color='red', linewidth=3, label='within-noise')
ax['B'].plot(ranks[:10], site_performance_AB_n[:, :].mean(axis=0), color='blue', linewidth=3, label='between-noise')
ax['B'].fill_between(ranks[:10], site_performance_AA_n.mean(axis=0) - semAA_n, site_performance_AA_n.mean(axis=0) + semAA_n, alpha=0.2, color ='red')
ax['B'].fill_between(ranks[:10], site_performance_AB_n.mean(axis=0) - semAB_n, site_performance_AB_n.mean(axis=0) + semAB_n, alpha=0.2, color ='blue')
# ax['B'].plot(ranks[:10], site_performance_AA_s[:, :].mean(axis=0), color='orange', linewidth=3, label='within-signal')
# ax['B'].plot(ranks[:10], site_performance_AB_s[:, :].mean(axis=0), color='cyan', linewidth=3, label='between-signal')
# ax['B'].fill_between(ranks[:10], site_performance_AA_s.mean(axis=0) - semAA_s, site_performance_AA_s.mean(axis=0) + semAA_s, alpha=0.2, color ='orange')
# ax['B'].fill_between(ranks[:10], site_performance_AB_s.mean(axis=0) - semAB_s, site_performance_AB_s.mean(axis=0) + semAB_s, alpha=0.2, color ='cyan')
ax['B'].set_ylabel('model performance')
ax['B'].set_xlabel('model rank')
ax['B'].legend()
ax['B'].set_box_aspect(1)

site_dimensionality_AA_n = []
site_dimensionality_AB_n = []
# site_dimensionality_AA_s = []
# site_dimensionality_AB_s = []

cmap = get_cmap('Dark2')
cmap = get_cmap('viridis')
crange = np.linspace(0,1,3)
cmap_s = get_cmap('Set2')
colors = cmap.colors
colors_s = cmap_s.colors
site_area_combos = set(site_area_AB)
area_colors = {'A1-A1': cmap(crange[0]), 'A1-PEG': cmap(crange[1]), 'PEG-PEG': cmap(crange[2]), 'PEG-A1': colors[3]}
area_colors_s = {'A1-A1': colors_s[0], 'A1-PEG': colors_s[1], 'PEG-PEG': colors_s[2], 'PEG-A1': colors_s[3]}
for sitenum in range(len(siteids)):
     site_rank_AA = np.array(site_rank_performance_AA[sitenum])
     site_rank_AB = np.array(site_rank_performance_AB[sitenum])
     # site_rank_AA_s = np.array(site_rank_performance_AA_s[sitenum])
     # site_rank_AB_s = np.array(site_rank_performance_AB_s[sitenum])
     semAA_n = np.nanstd(site_rank_AA, axis=1, ddof=1) / np.sqrt(
          np.sum(~np.isnan(site_rank_AA), axis=1))
     semAB_n = np.nanstd(site_rank_AB, axis=1, ddof=1) / np.sqrt(
          np.sum(~np.isnan(site_rank_AB), axis=1))
     # semAA_s = np.nanstd(site_rank_AA_s, axis=1, ddof=1) / np.sqrt(
     #      np.sum(~np.isnan(site_rank_AA_s), axis=1))
     # semAB_s = np.nanstd(site_rank_AB_s, axis=1, ddof=1) / np.sqrt(
     #      np.sum(~np.isnan(site_rank_AB_s), axis=1))
     semAA_n = semAA_n[:10]
     semAB_n = semAB_n[:10]
     # semAA_s = semAA_s[:10]
     # semAB_s = semAB_s[:10]
     site_dimensionality_AA_n.append(np.where(site_performance_AA_n[sitenum, :] > (site_performance_AA_n[sitenum, -1]-semAA_n[-1]))[0][0] + 1)
     site_dimensionality_AB_n.append(np.where(site_performance_AB_n[sitenum, :] > (site_performance_AB_n[sitenum, -1] - semAB_n[-1]))[0][0] + 1)
     # site_dimensionality_AA_s.append(np.where(site_performance_AA_s[sitenum, :] > (site_performance_AA_s[sitenum, -1]-semAA_s[-1]))[0][0] + 1)
     # site_dimensionality_AB_s.append(np.where(site_performance_AB_s[sitenum, :] > (site_performance_AB_s[sitenum, -1] - semAB_s[-1]))[0][0] + 1)

site_dimensionality_AA_n = np.array(site_dimensionality_AA_n)
site_dimensionality_AB_n = np.array(site_dimensionality_AB_n)
# site_dimensionality_AA_s = np.array(site_dimensionality_AA_s)
# site_dimensionality_AB_s = np.array(site_dimensionality_AB_s)
for area in site_area_combos:
     current_area = np.array([s_area == area for s_area in site_area_AB])
     area_color_n = [area_colors[area] for t in current_area if t == True]
     # area_color_s = [area_colors_s[area] for t in current_area if t == True]
     ax['C'].scatter(site_dimensionality_AA_n[current_area], site_dimensionality_AB_n[current_area], s=6,
                     c=area_color_n, label=f"{area}-noise")
     # ax['C'].scatter(site_dimensionality_AA_s[current_area], site_dimensionality_AB_s[current_area], s=6,
     #                 c=area_color_s, label=f"{area}-signal")
ax['C'].legend()
lims = (0,10)
ticks = np.arange(lims[0], lims[1], 2)
ax['C'].set_ylim(lims)
ax['C'].set_xlim(lims)
ax['C'].set_yticks(ticks)
ax['C'].set_xticks(ticks)
ax['C'].set_xlabel("predictive dimensions \n - within")
ax['C'].set_ylabel("predictive dimensions \n - between")
ax['C'].plot(np.arange(lims[0], lims[1]+1, 1), np.arange(lims[0], lims[1]+1, 1), '--', color='grey')
ax['C'].set_box_aspect(1)

# plot performance within vs between hemispheres for number of predictive dimensions
site_dim_performanceAA_n = np.array([site_performance_AA_n[si, di] for si, di in enumerate([9 for i in site_dimensionality_AA_n])])
site_dim_performanceAB_n = np.array([site_performance_AB_n[si, di] for si, di in enumerate([9 for i in site_dimensionality_AB_n])])
# site_dim_performanceAB_s = np.array([site_performance_AB_s[si, di] for si, di in enumerate([9 for i in site_dimensionality_AB_s])])
for area in site_area_combos:
     current_area = np.array([s_area == area for s_area in site_area_AB])
     area_color_n = [area_colors[area] for t in current_area if t == True]
     ax['D'].scatter(site_dim_performanceAA_n[current_area], site_dim_performanceAB_n[current_area], s=6, c=area_color_n, label=area)
max_perf = np.round(np.max(np.concatenate((site_dim_performanceAA_n, site_dim_performanceAB_n))), decimals=1)
dlims = (0,max_perf+0.1)
dticks = np.arange(dlims[0], dlims[1]+0.1, 0.2)
ax['D'].set_ylim(dlims)
ax['D'].set_xlim(dlims)
ax['D'].set_yticks(dticks)
ax['D'].set_xticks(dticks)
ax['D'].set_xlabel("performance \n - within")
ax['D'].set_ylabel("performance \n - between")
ax['D'].plot(np.arange(dlims[0], dlims[1]+0.1, 0.1), np.arange(dlims[0], dlims[1]+0.1, 0.1), '--', color='grey')
ax['D'].set_box_aspect(1)
plt.tight_layout()

# plot for diotic vs dichotic
f, ax = plt.subplot_mosaic([['.', 'B', 'C', 'D'], ['A1', 'B', 'C', 'D'], ['A2', 'B', 'C', 'D'], ['A3', 'B', 'C', 'D'], ['.', 'B', 'C', 'D']], width_ratios=[1.5,1,1,1], height_ratios=[1, 0.75, 0.75, 0.75, 1], gridspec_kw={'hspace':0.75, 'wspace':0.75}, figsize=(8,3.5))
# gridspec_kw={'hspace':0.01, 'wspace':0.5}
imopts = {'origin': 'lower', 'aspect':'auto', 'cmap':'Greys'}
ax['A1'].imshow(stim_spec, **imopts, interpolation='gaussian')
ax['A2'].imshow(psthA, **imopts, interpolation='None', vmax=min_val)
ax['A3'].imshow(psthB, **imopts, interpolation='None', vmax=min_val)
ticks = np.arange(0, len(psthA[0, :]), 2*rasterfs)
tick_labels = np.arange(0, len(psthA[0, :])/rasterfs, 2*1)
# ax['A1'].set_xticks(ticks)
# ax['A1'].set_xticklabels(tick_labels)
# ax['A2'].set_xticks(ticks)
# ax['A2'].set_xticklabels(tick_labels)
ax['A3'].set_xticks(ticks)
ax['A3'].set_xticklabels(tick_labels)
# ax['A1'].get_shared_x_axes().joined(ax['A1'], ax['A3'])
# ax['A2'].get_shared_x_axes().joined(ax['A2'], ax['A3'])
ax['A1'].sharex(ax['A3'])
ax['A2'].sharex(ax['A3'])
plt.setp(ax['A1'].get_xticklabels(), visible=False)
plt.setp(ax['A2'].get_xticklabels(), visible=False)
ax['A3'].set_xlabel("time (s)")
ax['A1'].set_ylabel("channels")
ax['A2'].set_ylabel("trials")
ax['A3'].set_ylabel("trials")
ax['A1'].set_title(stim_epochs2[0], fontsize=6)
ax['A2'].set_title(good_A_chans[0], fontsize=6)
ax['A3'].set_title(good_B_chans[0], fontsize=6)

site_performance_AA_n = []
site_performance_AA_s = []
site_area_AB = []
site_performance_AB_n = []
site_performance_AB_s = []
for siteid in siteids:
     site_area_AB.append(['-'.join([area['A'], area['B']]) for parm, area in parmfile_area.items() if siteid in parm][0])
     site_noise_df = subspace_bilateral_df[subspace_bilateral_df['model_names'].str.contains(siteid)]
     site_signal_df = subspace_mono_df[subspace_mono_df['model_names'].str.contains(siteid)]
     mean_rank_performance_AA_n = []
     mean_rank_performance_AB_n = []
     mean_rank_performance_AA_s = []
     mean_rank_performance_AB_s = []
     for rank in ranks:
          rank_AA_df_n = site_noise_df[site_noise_df['model_names'].str.contains(f'Rank{str(rank)}_bilateral_AA')]
          rank_AB_df_n = site_noise_df[site_noise_df['model_names'].str.contains(f'Rank{str(rank)}_bilateral_AB')]
          rank_AA_df_s = site_signal_df[site_signal_df['model_names'].str.contains(f'Rank{str(rank)}_mono_AA')]
          rank_AB_df_s = site_signal_df[site_signal_df['model_names'].str.contains(f'Rank{str(rank)}_mono_AB')]
          mean_rank_performance_AA_n.append(np.nanmean(rank_AA_df_n['mean_model_performance']))
          mean_rank_performance_AB_n.append(np.nanmean(rank_AB_df_n['mean_model_performance']))
          mean_rank_performance_AA_s.append(np.nanmean(rank_AA_df_s['mean_model_performance']))
          mean_rank_performance_AB_s.append(np.nanmean(rank_AB_df_s['mean_model_performance']))
     site_performance_AA_n.append(mean_rank_performance_AA_n)
     site_performance_AB_n.append(mean_rank_performance_AB_n)
     site_performance_AA_s.append(mean_rank_performance_AA_s)
     site_performance_AB_s.append(mean_rank_performance_AB_s)
site_performance_AA_n = np.array(site_performance_AA_n)[:, :10]
site_performance_AB_n = np.array(site_performance_AB_n)[:, :10]
site_performance_AA_s = np.array(site_performance_AA_s)[:, :10]
site_performance_AB_s = np.array(site_performance_AB_s)[:, :10]
semAA_n = np.nanstd(site_performance_AA_n, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(site_performance_AA_n), axis=0))
semAB_n = np.nanstd(site_performance_AB_n, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(site_performance_AB_n), axis=0))
semAA_s = np.nanstd(site_performance_AA_s, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(site_performance_AA_s), axis=0))
semAB_s = np.nanstd(site_performance_AB_s, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(site_performance_AB_s), axis=0))
ax['B'].plot(ranks[:10], site_performance_AA_n[:, :].mean(axis=0), color='red', linewidth=3, label='within-bilateral')
ax['B'].plot(ranks[:10], site_performance_AB_n[:, :].mean(axis=0), color='blue', linewidth=3, label='between-bilateral')
ax['B'].fill_between(ranks[:10], site_performance_AA_n.mean(axis=0) - semAA_n, site_performance_AA_n.mean(axis=0) + semAA_n, alpha=0.2, color ='red')
ax['B'].fill_between(ranks[:10], site_performance_AB_n.mean(axis=0) - semAB_n, site_performance_AB_n.mean(axis=0) + semAB_n, alpha=0.2, color ='blue')
ax['B'].plot(ranks[:10], site_performance_AA_s[:, :].mean(axis=0), color='orange', linewidth=3, label='within-mono')
ax['B'].plot(ranks[:10], site_performance_AB_s[:, :].mean(axis=0), color='cyan', linewidth=3, label='between-mono')
ax['B'].fill_between(ranks[:10], site_performance_AA_s.mean(axis=0) - semAA_s, site_performance_AA_s.mean(axis=0) + semAA_s, alpha=0.2, color ='orange')
ax['B'].fill_between(ranks[:10], site_performance_AB_s.mean(axis=0) - semAB_s, site_performance_AB_s.mean(axis=0) + semAB_s, alpha=0.2, color ='cyan')
ax['B'].set_ylabel('model performance')
ax['B'].set_xlabel('model rank')
ax['B'].legend()
ax['B'].set_box_aspect(1)

site_dimensionality_AA_n = []
site_dimensionality_AB_n = []
site_dimensionality_AA_s = []
site_dimensionality_AB_s = []

cmap = get_cmap('Dark2')
cmap_s = get_cmap('Set2')
colors = cmap.colors
colors_s = cmap_s.colors
site_area_combos = set(site_area_AB)
area_colors = {'A1-A1': colors[0], 'A1-PEG': colors[1], 'PEG-PEG': colors[2], 'PEG-A1': colors[3]}
area_colors_s = {'A1-A1': colors_s[0], 'A1-PEG': colors_s[1], 'PEG-PEG': colors_s[2], 'PEG-A1': colors_s[3]}
for sitenum in range(len(siteids)):
     site_dimensionality_AA_n.append(np.where(site_performance_AA_n[sitenum, :] > (site_performance_AA_n[sitenum, -1]-semAA_n[-1]))[0][0] + 1)
     site_dimensionality_AB_n.append(np.where(site_performance_AB_n[sitenum, :] > (site_performance_AB_n[sitenum, -1] - semAB_n[-1]))[0][0] + 1)
     site_dimensionality_AA_s.append(np.where(site_performance_AA_s[sitenum, :] > (site_performance_AA_s[sitenum, -1]-semAA_s[-1]))[0][0] + 1)
     site_dimensionality_AB_s.append(np.where(site_performance_AB_s[sitenum, :] > (site_performance_AB_s[sitenum, -1] - semAB_s[-1]))[0][0] + 1)

site_dimensionality_AA_n = np.array(site_dimensionality_AA_n)
site_dimensionality_AB_n = np.array(site_dimensionality_AB_n)
site_dimensionality_AA_s = np.array(site_dimensionality_AA_s)
site_dimensionality_AB_s = np.array(site_dimensionality_AB_s)
for area in site_area_combos:
     current_area = np.array([s_area == area for s_area in site_area_AB])
     area_color_n = [area_colors[area] for t in current_area if t == True]
     area_color_s = [area_colors_s[area] for t in current_area if t == True]
     ax['C'].scatter(site_dimensionality_AA_n[current_area], site_dimensionality_AB_n[current_area], s=6,
                     c=area_color_n, label=f"{area}-bilateral")
     ax['C'].scatter(site_dimensionality_AA_s[current_area], site_dimensionality_AB_s[current_area], s=6,
                     c=area_color_s, label=f"{area}-mono")
ax['C'].legend()
lims = (0,8)
ticks = np.arange(lims[0], lims[1], 2)
ax['C'].set_ylim(lims)
ax['C'].set_xlim(lims)
ax['C'].set_yticks(ticks)
ax['C'].set_xticks(ticks)
ax['C'].set_xlabel("predictive dimensions \n - within")
ax['C'].set_ylabel("predictive dimensions \n - between")
ax['C'].plot(np.arange(lims[0], lims[1]+1, 1), np.arange(lims[0], lims[1]+1, 1), '--', color='grey')
ax['C'].set_box_aspect(1)

# plot performance within vs between hemispheres for number of predictive dimensions
site_dim_performanceAA_n = np.array([site_performance_AA_n[si, di] for si, di in enumerate(site_dimensionality_AA_n)])
site_dim_performanceAB_n = np.array([site_performance_AB_n[si, di] for si, di in enumerate(site_dimensionality_AB_n)])
site_dim_performanceAB_s = np.array([site_performance_AB_s[si, di] for si, di in enumerate(site_dimensionality_AB_s)])
for area in site_area_combos:
     current_area = np.array([s_area == area for s_area in site_area_AB])
     area_color_n = [area_colors[area] for t in current_area if t == True]
     ax['D'].scatter(site_dim_performanceAB_n[current_area], site_dim_performanceAB_s[current_area], s=6, c=area_color_n, label=area)
max_perf = np.round(np.max(np.concatenate((site_dim_performanceAB_n, site_dim_performanceAB_s))), decimals=1)
dlims = (0,max_perf+0.1)
dticks = np.arange(dlims[0], dlims[1], 0.2)
ax['D'].set_ylim(dlims)
ax['D'].set_xlim(dlims)
ax['D'].set_yticks(dticks)
ax['D'].set_xticks(dticks)
ax['D'].set_xlabel("performance \n - between - bi")
ax['D'].set_ylabel("performance \n - between - mono")
ax['D'].plot(np.arange(dlims[0], dlims[1]+0.1, 0.1), np.arange(dlims[0], dlims[1]+0.1, 0.1), '--', color='grey')
ax['D'].set_box_aspect(1)
plt.tight_layout()

# plt.savefig(data_save_path/'subspace_fig.pdf', dpi=600, format='pdf')
plt.savefig(data_save_path/'subspace_fig_noise_updated.pdf', dpi=600, format='pdf')
bp = []

