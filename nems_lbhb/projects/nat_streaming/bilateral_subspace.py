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
import nems
from nems.models import Model
from nems.layers import WeightChannels, LevelShift, DoubleExponential, FiniteImpulseResponse, RectifiedLinear
from nems.preprocessing import (
    indices_by_fraction, split_at_indices, JackknifeIterator)
from nems.tools.json import save_model

from pathlib import Path

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

# random number generator for permutation
rng = np.random.default_rng()

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

# create dataframe
badfiles = []
model_dataframes = []
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

     # filter high SNR units
     highA = snrA >= 0.3
     lowA = snrA < 0.2
     highB = snrB >= 0.3
     lowB = snrB < 0.2

     respA_high = respA.extract_channels(list(compress(respA.chans, highA)))
     respB_high = respB.extract_channels(list(compress(respB.chans, highB)))

     s1, s1_inds, s1_chans, s2, s2_inds, s2_chans, t, t_inds, t_chans = subspace_matrices(respA, respB,
                                                                                          source='largest',
                                                                                          stim='both',
                                                                                          svn=['noise', 'signal'])
     # s1_mono, s1_inds_mono, s1_chans_mono, s2_mono, s2_inds_mono, s2_chans_mono, t_mono, t_inds_mono, t_chans_mono = (
     #      subspace_matrices(respA, respB, source='largest', stim='mono', svn=['noise', 'signal']))

     # subspace analysis

     cost_function = 'squared_error'
     fitter = 'tf'

     # input
     X = s1[1]
     # target same side
     Y = s2[1]
     # target opposite side
     Y2 = t[1]

     input_count = X.shape[1]
     output_count = Y.shape[1]

     # store model predictions and true data for each jackknife to plot performance
     for i in range(12):
          acount = i + 1  # rank of model
          layers = [
               WeightChannels(shape=(input_count, acount), input='input', output='target'),
               WeightChannels(shape=(acount, output_count), input='target', output='target'),
               LevelShift(shape=(1, output_count), input='target', output='target'),
          ]
          fitter = 'tf'
          fitter_options = {'cost_function': cost_function,  # 'nmse'
                            'early_stopping_tolerance': 1e-3,
                            'validation_split': 0,
                            'learning_rate': 1e-2, 'epochs': 3000
                            }
          fitter_options2 = {'cost_function': cost_function,
                             'early_stopping_tolerance': 1e-4,
                             'validation_split': 0,
                             'learning_rate': 1e-3, 'epochs': 8000
                             }

          model = Model(layers=layers)
          model.name = f'Site{siteid}_Rank{acount}'
          model = model.sample_from_priors()
          model = model.sample_from_priors()
          print(f'Model: {model.name}')

          # only inlcude this if we add a nonlinearity at the output (Relu or Differenceofexp)
          # log.info('Fit stage 1: without static output nonlinearity')
          # model.layers[-1].skip_nonlinearity()
          # model = model.fit(input=input, target=target, backend=fitter,
          #                   fitter_options=fitter_options)
          # model.layers[-1].unskip_nonlinearity()
          # log.info('Fit stage 2: with static output nonlinearity')

          ###  do i need to fit a model before jackkniffing? ###
          # model_AA = model.fit(input=X, target=Y, backend=fitter,
          #                   verbose=0, fitter_options=fitter_options)
          # model_AB = model.fit(input=X, target=Y2, backend=fitter,
          #                   verbose=0, fitter_options=fitter_options)
          jack_samples = 10
          jackknife_iterator = JackknifeIterator(input=X, target=Y, samples=jack_samples, axis=0)
          jackknife_iteratorAB = JackknifeIterator(input=X, target=Y2, samples=jack_samples, axis=0)
          #
          # jackknife_iterator.reset_iter()
          # model_fit_list_AA=[]
          # for j in jackknife_iterator:
          #      model_fit_list_AA.append(model.fit(input=j['input'], target=j['target'], backend=fitter,
          #                           verbose=0, fitter_options=fitter_options))
          ###

          # We can then fit this iterator directly and return a list of fitted model
          # This will fit range(0, samples) models with given masks before returning a list of fitted models
          jackknife_iterator.reset_iter()
          model_fit_list_AA = jackknife_iterator.get_fitted_jackknifes(model, backend=fitter,
                                                                       verbose=0, fitter_options=fitter_options2)
          jackknife_inputs_AA = [jackknife_iterator.get_inverse_jackknife(X, jackknife_iterator.mask_list[j]) for j in
                                 range(jack_samples)]
          jackknife_targets_AA = [jackknife_iterator.get_inverse_jackknife(Y, jackknife_iterator.mask_list[j]) for j in
                                  range(jack_samples)]

          jackknife_iteratorAB.reset_iter()
          model_fit_list_AB = jackknife_iteratorAB.get_fitted_jackknifes(model, backend=fitter,
                                                                         verbose=0, fitter_options=fitter_options2)
          jackknife_inputs_AB = [jackknife_iteratorAB.get_inverse_jackknife(X, jackknife_iteratorAB.mask_list[j]) for j
                                 in
                                 range(jack_samples)]
          jackknife_targets_AB = [jackknife_iterator.get_inverse_jackknife(Y2, jackknife_iteratorAB.mask_list[j]) for j
                                  in
                                  range(jack_samples)]

          # test on inverse jackknifes and return predictions
          jack_predicts_AA = []
          jack_predicts_AB = []
          for j in range(jack_samples):
               jack_predicts_AA.append(model_fit_list_AA[j].predict(jackknife_inputs_AA[j]))
               jack_predicts_AB.append(model_fit_list_AB[j].predict(jackknife_inputs_AB[j]))

          # save models to /auto/users/wingertj/models
          for j, mod in enumerate(model_fit_list_AA):
               save_model(mod, model_save_path / f"{mod.name}_AA_Jackknife{j+1}")
          for j, mod in enumerate(model_fit_list_AB):
               save_model(mod, model_save_path / f"{mod.name}_AB_Jackknife{j+1}")

          jack_cc_AA = []
          jack_cc_AB = []
          for j in range(jack_samples):
               jack_cc_AA.append(
                    np.array([np.corrcoef(jack_predicts_AA[j][:, i], jackknife_targets_AA[j][:, i])[0, 1] for i in range(output_count)]))
               jack_cc_AB.append(
                    np.array([np.corrcoef(jack_predicts_AB[j][:, i], jackknife_targets_AB[j][:, i])[0, 1] for i in range(output_count)]))

          d = {
               'model_names': [f'{mod.name}_AA_Jackknife{j + 1}' for j, mod in enumerate(model_fit_list_AA)] + [f'{mod.name}_AB_Jackknife{j + 1}' for j, mod in enumerate(model_fit_list_AB)],
               'model_save_path': [model_save_path / f"{mod.name}_AA_Jackknife{j+1}" for j, mod in enumerate(model_fit_list_AA)] + [model_save_path / f"{mod.name}_AB_Jackknife{j+1}" for j, mod in enumerate(model_fit_list_AB)],
               'jackknife': [j+1 for j in range(jack_samples)] + [j+1 for j in range(jack_samples)],
               'mean_model_performance': [np.mean(jack_cc_AA[i]) for i in range(jack_samples)] + [np.mean(jack_cc_AB[i]) for i in range(jack_samples)],
               'cell_model_performance': jack_cc_AA + jack_cc_AB,
               'model_test_input': jackknife_inputs_AA + jackknife_inputs_AB,
               'model_test_targets': jackknife_targets_AA + jackknife_targets_AB,
               'model_test_predictions': jack_predicts_AA + jack_predicts_AB
          }
          rank_dataframe = pd.DataFrame(data=d)
          model_dataframes.append(rank_dataframe)

subspace_dataframe = pd.concat(model_dataframes)
subspace_dataframe.to_pickle(data_save_path/'subspace_df_signal.pkl')


# cost_function='squared_error'
# fitter='tf'
#
# # X = nn[0][np.setdiff1d(np.arange(nn[0].shape[0]),sindexs),:].T
# # input
# X = s1[0]
# # target same side
# Y = s2[0]
# # target opposite side
# Y2= t[0]
#
# input_count=X.shape[1]
# output_count=Y.shape[1]
#
# rank_predictions_AA = []
# rank_models_AA = []
# rank_predictions_AB = []
# rank_models_AB = []
# rank_targets_AA = []
# rank_targets_AB = []
#
# for i in range(12):
#      acount = i+1  # rank of model
#      layers = [
#           WeightChannels(shape=(input_count, acount), input='input', output='target'),
#           WeightChannels(shape=(acount, output_count), input='target', output='target'),
#           LevelShift(shape=(1, output_count), input='target', output='target'),
#      ]
#      fitter = 'tf'
#      fitter_options = {'cost_function': cost_function,  # 'nmse'
#                        'early_stopping_tolerance': 1e-3,
#                        'validation_split': 0,
#                        'learning_rate': 1e-2, 'epochs': 3000
#                        }
#      fitter_options2 = {'cost_function': cost_function,
#                         'early_stopping_tolerance': 1e-4,
#                         'validation_split': 0,
#                         'learning_rate': 1e-3, 'epochs': 8000
#                         }
#
#      model = Model(layers=layers)
#      model.name = f'Site{siteid}_Rank{acount}'
#      model = model.sample_from_priors()
#      model = model.sample_from_priors()
#      print(f'Model: {model.name}')
#
#
#      # only inlcude this if we add a nonlinearity at the output (Relu or Differenceofexp)
#      # log.info('Fit stage 1: without static output nonlinearity')
#      # model.layers[-1].skip_nonlinearity()
#      # model = model.fit(input=input, target=target, backend=fitter,
#      #                   fitter_options=fitter_options)
#      # model.layers[-1].unskip_nonlinearity()
#      #log.info('Fit stage 2: with static output nonlinearity')
#
#      ###  do i need to fit a model before jackkniffing? ###
#      # model_AA = model.fit(input=X, target=Y, backend=fitter,
#      #                   verbose=0, fitter_options=fitter_options)
#      # model_AB = model.fit(input=X, target=Y2, backend=fitter,
#      #                   verbose=0, fitter_options=fitter_options)
#      jack_samples = 10
#      jackknife_iterator = JackknifeIterator(input=X, target=Y, samples=jack_samples, axis=0)
#      jackknife_iteratorAB = JackknifeIterator(input=X, target=Y2, samples=jack_samples, axis=0)
#      #
#      # jackknife_iterator.reset_iter()
#      # model_fit_list_AA=[]
#      # for j in jackknife_iterator:
#      #      model_fit_list_AA.append(model.fit(input=j['input'], target=j['target'], backend=fitter,
#      #                           verbose=0, fitter_options=fitter_options))
#      ###
#
#      # We can then fit this iterator directly and return a list of fitted model
#      # This will fit range(0, samples) models with given masks before returning a list of fitted models
#      jackknife_iterator.reset_iter()
#      model_fit_list_AA = jackknife_iterator.get_fitted_jackknifes(model, backend=fitter,
#                                verbose=0, fitter_options=fitter_options2)
#      jackknife_inputs_AA = [jackknife_iterator.get_inverse_jackknife(X, jackknife_iterator.mask_list[j]) for j in
#                              range(jack_samples)]
#      jackknife_targets_AA = [jackknife_iterator.get_inverse_jackknife(Y, jackknife_iterator.mask_list[j]) for j in
#                              range(jack_samples)]
#
#      jackknife_iteratorAB.reset_iter()
#      model_fit_list_AB = jackknife_iteratorAB.get_fitted_jackknifes(model, backend=fitter,
#                                verbose=0, fitter_options=fitter_options2)
#      jackknife_inputs_AB= [jackknife_iteratorAB.get_inverse_jackknife(X, jackknife_iteratorAB.mask_list[j]) for j in
#                                 range(jack_samples)]
#      jackknife_targets_AB = [jackknife_iterator.get_inverse_jackknife(Y2, jackknife_iteratorAB.mask_list[j]) for j in
#                              range(jack_samples)]
#
#      # test on inverse jackknifes and return predictions
#      jack_predicts_AA = []
#      jack_predicts_AB = []
#      for j in range(jack_samples):
#           jack_predicts_AA.append(model_fit_list_AA[j].predict(jackknife_inputs_AA[j]))
#           jack_predicts_AB.append(model_fit_list_AB[j].predict(jackknife_inputs_AB[j]))
#      # Predictions can be done across the entire list of fitted models, using our sets of masks as well
#      # jackknife_iterator.reset_iter()
#      # prediction_set = jackknife_iterator.get_predicted_jackknifes(model_fit_list_AA)
#      #
#      # jackknife_iteratorAB.reset_iter()
#      # prediction_setAB = jackknife_iteratorAB.get_predicted_jackknifes(model_fit_list_AB)
#
#      # predictionsAA = prediction_set['prediction']
#      # targetsAA = prediction_set['target']
#      # predictionsAB = prediction_setAB['prediction']
#      # targetsAB = prediction_setAB['target']
#
#      rank_predictions_AA.append(jack_predicts_AA)
#      rank_predictions_AB.append(jack_predicts_AB)
#      rank_targets_AA.append(jackknife_targets_AA)
#      rank_targets_AB.append(jackknife_targets_AB)
#
#      rank_models_AA.append(model_fit_list_AA)
#      rank_models_AB.append(model_fit_list_AB)
#
#      # save models to /auto/users/wingertj/models
#      for j, mod in enumerate(model_fit_list_AA):
#           save_model(mod, model_save_path/f"{mod.name}_Jackknife{j}")
#
# rank_cc_AA = []
# rank_cc_AB = []
# for r in range(9):
#      rank_cc_AA.append(np.array([np.corrcoef(rank_predictions_AA[r][:, i], Y[:, i])[0, 1] for i in range(output_count)]))
#      rank_cc_AB.append(np.array([np.corrcoef(rank_predictions_AB[r][:, i], Y2[:, i])[0, 1] for i in range(output_count)]))
#
# f, ax = plt.subplots(2, 1)
# ax[0].scatter(np.arange(9)+1, [np.mean(rank_cc_AA[i]) for i in range(9)])
# ax[1].scatter(np.arange(9)+1, [np.mean(rank_cc_AB[i]) for i in range(9)])



# fit_pred = model.predict(input=input)
# prediction = model.predict(input=test_input)
# if type(prediction) is dict:
#      fit_pred = fit_pred['prediction']
#      prediction = prediction['prediction']
#
# fit_cc = np.array([np.corrcoef(fit_pred[:, i], target[:, i])[0, 1] for i in range(cellcount)])
# cc = np.array([np.corrcoef(prediction[:, i], test_target[:, i])[0, 1] for i in range(cellcount)])
# rf = r_floor(X1mat=prediction.T, X2mat=test_target.T)
#
# model.meta['fit_predxc'] = fit_cc
# model.meta['predxc'] = cc
# model.meta['prediction'] = prediction
# model.meta['resp'] = test_target
# model.meta['siteid'] = siteid
# model.meta['batch'] = rec.meta['batch']
# model.meta['modelname'] = model.name
# model.meta['cellids'] = est['resp'].chans
# model.meta['r_test'] = cc[:, np.newaxis]
# model.meta['r_fit'] = fit_cc[:, np.newaxis]
# model.meta['r_floor'] = rf[:, np.newaxis]


