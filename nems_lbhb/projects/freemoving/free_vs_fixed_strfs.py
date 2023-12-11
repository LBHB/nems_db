#stardard imports
import os
import numpy as np
import logging
import sys
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.ndimage import zoom

import nems0.utils
from nems0 import db
import nems0.preprocessing as preproc
import nems_lbhb.projects.freemoving.free_tools
from nems_lbhb.projects.freemoving import free_model, free_tools
from nems0.epoch import epoch_names_matching
import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, DoubleExponential, RectifiedLinear
from nems import visualization
from nems.tools import dstrf as dtools

log = logging.getLogger(__name__)

imopts_dstrf = {'origin': 'lower',
                'interpolation': 'none',
                'cmap': 'bwr',
                'aspect': 'auto'}
outpath = '/auto/data/nems_db/results/svd/strfs'

filepath = Path(os.path.dirname(__file__)) / 'dlc_coordinates.npy'
didx=np.load(filepath)

def load_rec(siteid, batch):
    dlc_chans = 8
    rasterfs = 50
    rec = free_model.load_free_data(siteid, batch=batch, rasterfs=rasterfs,
                                    dlc_chans=dlc_chans, compute_position=True)
    return rec


def fit_strf(stim, resp):
    N = stim.shape[1]
    model = Model()
    model.add_layers(
        WeightChannels(shape=(N, 3)),  # 18 spectral channels->2 composite channels
        FiniteImpulseResponse(shape=(15, 3)),  # 15 taps, 2 spectral channels
        RectifiedLinear(shape=(1,), no_shift=False, no_offset=False)  # static nonlinearity, 1 output
    )
    model.name = "Rank2LNSTRF"

    # random initial conditions
    model = model.sample_from_priors()

    # optimize model parameters
    options = {'cost_function': 'squared_error', 'early_stopping_delay': 100, 'early_stopping_patience': 50,
               'early_stopping_tolerance': 1e-3, 'validation_split': 0,
               'learning_rate': 5e-3, 'epochs': 2000}
    fitted_model = model.fit(stim, resp, fitter_options=options, backend='tf',
                             verbose=0)

    pred_test = fitted_model.predict(stim)
    r_fit = np.corrcoef(pred_test[:,0], resp[:,0])[0,1]
    fitted_model.meta['r_fit'] = r_fit
    return fitted_model

def fit_strfs(rec, verbose=False):

    rec = free_tools.stim_filt_hrtf(rec, hrtf_format='az', smooth_win=2,
                                    f_min=200, f_max=20000, channels=18)['rec']

    sig = 'dlc'
    bntepochs = epoch_names_matching(rec[sig].epochs, "^FILE_.*BNT")
    ntdepochs = epoch_names_matching(rec[sig].epochs, "^FILE_.*NTD")
    log.info(f"BNT epochs: {bntepochs}")
    log.info(f"NTD epochs: {ntdepochs}")

    epoch_regex = "^STIM_"
    r3 = rec.create_mask(ntdepochs)
    r3 = rec.and_mask(r3['dlc_valid'].as_continuous()[0, :])
    estntd, valntd = r3.split_using_epoch_occurrence_counts(epoch_regex=epoch_regex)
    estntd = estntd.and_mask(ntdepochs)
    valntd = valntd.and_mask(ntdepochs)

    r3 = rec.create_mask(bntepochs)
    estbnt, valbnt = r3.split_using_epoch_occurrence_counts(epoch_regex=epoch_regex)

    estbnt = preproc.average_away_epoch_occurrences(estbnt, epoch_regex=epoch_regex)
    valbnt = preproc.average_away_epoch_occurrences(valbnt, epoch_regex=epoch_regex)

    if verbose:
        f,ax=plt.subplots(2,1)
        dv=np.where(rec['dlc_valid'].as_continuous()[0,:])[0]
        dlc = rec['dlc'].as_continuous()
        ax[0].plot(dlc[0:2,dv[::100]].T)
        ax[1].plot(estbnt['mask'].as_continuous()[0,::100])
        ax[1].plot(valbnt['mask'].as_continuous()[0,::100]+1)
        ax[1].plot(estntd['mask'].as_continuous()[0,::100]+2)
        ax[1].plot(valntd['mask'].as_continuous()[0,::100]+3)

    cellids = rec['resp'].chans
    bnt_models=[]
    ntd_models=[]

    for cid, cellid in enumerate(cellids):
        spectrogram_fit = estbnt.apply_mask()['stim'].as_continuous().T
        response_fit = estbnt.apply_mask()['resp'].as_continuous().T[:, [cid]]
        spectrogram_test = valbnt.apply_mask()['stim'].as_continuous().T
        response_test = valbnt.apply_mask()['resp'].as_continuous().T[:, [cid]]

        bnt_models.append(fit_strf(spectrogram_fit, response_fit))
        ntd_models.append(
            fit_strf(estntd.apply_mask()['stim'].as_continuous().T,
                     estntd.apply_mask()['resp'].as_continuous().T[:,[cid]]))

    rowcount = int(np.ceil(len(cellids)/4))

    f, ax = plt.subplots(rowcount, 8, figsize=(12,rowcount*1.75),
                         sharex=True, sharey=True)
    ax1 = np.concatenate((ax[:,0], ax[:,2], ax[:,4], ax[:,6]))
    ax2 = np.concatenate((ax[:,1], ax[:,3], ax[:,5], ax[:,7]))

    for i, (c, a1, a2) in enumerate(zip(cellids, ax1, ax2)):
        visualization.simple_strf(bnt_models[i], ax=a1)
        visualization.simple_strf(ntd_models[i], ax=a2)
        a1.axhline(17.5, linestyle='--', lw=0.5, color='black')
        a2.axhline(17.5, linestyle='--', lw=0.5, color='black')

        xl = a1.get_xlim()
        a1.text(xl[1], 1, 'R', ha='right')
        a1.text(xl[1], 18, 'L', ha='right')
        a1.text(xl[1], 35, f'{bnt_models[i].meta["r_fit"]:.3f}', ha='right', va='top')
        a2.text(xl[1], 35, f'{ntd_models[i].meta["r_fit"]:.3f}', ha='right', va='top')
        a1.set_ylabel(c)
        a2.set_ylabel('')
        if i<len(cellids):
            a1.set_xlabel('')
            a2.set_xlabel('')

    ax1[0].set_title('BNT')
    ax2[0].set_title('NTD')
    f.suptitle(rec.meta['siteid'])
    plt.tight_layout()

    return f

def adjust_didx(dlc, didx):
    # didx : posidx, dlcchan, lag
    x, y = dlc[:, 0], dlc[:, 1]
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    ynp = np.percentile(y, 90)
    xnp = np.percentile(x[(x>0.05) & (y>ynp)], 75)
    ysp = np.percentile(y, 15)
    xsp = np.percentile(x[(y<ysp)], 50)
    xnp_=didx[0, -1, 0]
    ynp_=didx[0, -1, 1]
    xsp_=didx[2, -1, 0]
    ysp_=didx[2, -1, 1]
    didx_new = didx.copy()
    log.info(f"old (xnp,ynp): {xnp_:.3f},{ynp_:.3f} (xsp,ysp): {xsp_:.3f},{ysp_:.3f}")
    log.info(f"new (xnp,ynp): {xnp:.3f},{ynp:.3f} (xsp,ysp): {xsp:.3f},{ysp:.3f}")
    didx_new[:,:,0:8:2] = (didx[:,:,0:8:2]-xnp_)/(xsp_-xnp_)*(xsp-xnp)+xnp
    didx_new[:,:,1:8:2] = (didx[:,:,1:8:2]-ysp_)/(ynp_-ysp_)*(ynp-ysp)+ysp

    return didx_new


def movement_plot(rec, T1=260, T2=280, t_indexes=None):

    speaker1_x0y0 = 1.0, -0.8
    speaker2_x0y0 = 0.0, -0.8
    fs = rec['dlc'].fs
    dlc = rec['dlc'].as_continuous().T

    ctx = free_model.free_split_rec(rec, apply_hrtf=True)
    rec2 = ctx['rec']
    if t_indexes is None:
        t_indexes=np.array([270.2, 272, 273, 274.94])
    d_indexes=[13100, 13209, 13300, 13748]  #
    d_indexes=(np.array(t_indexes)*fs).astype(int)
    t1, t2 = int(T1*fs), int(T2*fs)

    didx_ = adjust_didx(dlc, didx)
    #didx_ = didx

    f1 = plt.figure()
    plt.scatter(dlc[::10, 0], dlc[::10, 1], s=2, color='lightgray')
    for i in range(len(didx_)):
        # compute distance and angle to each speaker
        # code pasted in from free_tools
        d1, theta1, vel, rvel, d_fwd, d_lat = free_tools.compute_d_theta(
            didx_[i].T, fs=fs, smooth_win=0.1, ref_x0y0=speaker1_x0y0)
        d2, theta2, vel, rvel, d_fwd, d_lat = free_tools.compute_d_theta(
            didx_[i].T, fs=fs, smooth_win=0.1, ref_x0y0=speaker2_x0y0)
        #log.info(f"{i} {didx[i,-1,:2]} d1={d1[0,-1]} th1= {d2[0,-1]}")
        plt.plot(didx_[i, -1, 2], didx_[i, -1, 3], "o", color='blue')
        plt.plot(didx_[i, -1, 0], didx_[i, -1, 1], "o", color='red')
        plt.text(didx_[i, -1, 0], didx_[i, -1, 1],
                 f"{i}: d,th1=({d1[0,-1]:.1f},{theta1[0,-1]:.0f})\n  d,th2=({d2[0,-1]:.1f},{theta2[0,-1]:.0f})",
                 va='center')
        plt.plot(didx_[i, -6:, 0], didx_[i, -6:, 1], color='darkblue', lw=1)
    plt.gca().invert_yaxis()
    plt.title(rec.meta['siteid'], fontsize=12)

    f2=plt.figure(figsize=(10,5))
    ax=f2.add_subplot(1, 2, 1)
    ax.scatter(dlc[::10, 0], dlc[::10, 1], s=2, color='lightgray')
    ax.plot(dlc[t1:t2, 0], dlc[t1:t2,1], lw=0.5, color='gray')
    ax.invert_yaxis()

    ax1=f2.add_subplot(4,2,2)
    ax1.plot(np.arange(t1, t2)/fs, dlc[t1:t2, 0:2])
    ax1.legend(('Nose x', 'Nose y'), fontsize=8, frameon=False)
    #ax1.invert_yaxis()
    xl = ax1.get_xlim()
    yl = ax1.get_ylim()

    ax2=f2.add_subplot(4,2,4)
    s_ = rec['stim']._data[:,t1:t2].copy()
    s2_ = rec2['stim']._data[:,t1:t2].copy()
    print(s_.max(),s2_.max())
    e=rec['stim'].epochs
    tarepochs=e.loc[e['name'].str.startswith("TAR_") & (e['start']>T1) & (e['end']<T2)]
    for i,r in tarepochs.iterrows():
        tar1,tar2 = int(r['start']*fs)-t1,int(r['end']*fs)-t1
        print(tar1,tar2,s_.shape)
        s_[5,tar1:tar2]+=0.1
        s2_[5,tar1:tar2]+=0.25
        s2_[5+18,tar1:tar2]+=0.25
    s_max=s_.max()
    s2_max=s2_.max()

    ax2.imshow(s_[:18,:]**1.5, extent=[t1/fs, t2/fs, 18.5,0.5], cmap='gray_r', vmax=s_max*0.4)
    ax2.imshow(s_[18:,:]**1.5, extent=[t1/fs, t2/fs, 37.5,19.5], cmap='gray_r', vmax=s_max*0.4)
    ax2.set_ylabel('R speaker - L speaker')
    ax2.set_xlim(xl)
    ax2.set_ylim([0,38])

    ax3=f2.add_subplot(4,2,6)
    ax3.imshow(s2_[:18,:]**1.5, extent=[t1/fs, t2/fs, 18.5,0.5], cmap='gray_r', vmax=s2_max)
    ax3.imshow(s2_[18:,:]**1.5, extent=[t1/fs, t2/fs, 37.5,19.5], cmap='gray_r', vmax=s2_max)
    ax3.set_ylabel('R ear - L ear')
    ax3.set_xlim(xl)
    ax3.set_ylim([0,38])

    ax4=f2.add_subplot(4,2,8)
    r_ = rec2['resp']._data[:,t1:t2]
    ax4.imshow(rec2['resp']._data[:,t1:t2], extent=[t1/fs, t2/fs, 0.5,rec2['resp'].shape[0]+0.5],
               vmax=r_.max()*.75, cmap='gray_r')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Unit')
    ax4.set_xlim(xl)

    for i, t in enumerate(d_indexes[:4]):
        print(dlc[t,0],dlc[t,1])
        ax.scatter(dlc[t,0],dlc[t,1],s=20, color='black')
        ax.scatter(dlc[t,0],dlc[t,1],s=10, color='yellow')
        ax.text(dlc[t,0]+0.02,dlc[t,1]-0.01,f"{i+1}",va='center',ha='left',fontsize=10)
        ax1.axvline(t/fs, color='red', lw=1)
        ax1.text(t/fs, yl[1]-0.05, f"{i+1}",va='bottom',ha='center',fontsize=10)
        ax2.axvline(t / fs, color='red', lw=1)
        ax3.axvline(t / fs, color='red', lw=1)
    f2.suptitle(rec.meta['siteid'])

    plt.tight_layout()

    
    return f1,f2


###
### DSTRF stuff
###

def dstrf_snapshots(rec, model_list, D=11, out_channel=0, time_step=85,
                    snr_threshold=5, pc_count=3, reset_backend=False, input_name = 'input'):
    """
    compute mean dSTRF for a single cell at standardized positions
    by "freezing" the DLC signal and computing the dSTRF for a bunch of stimuli
    """
    t_indexes = np.arange(time_step, rec['stim'].shape[1], time_step)
    dlc = rec['dlc'].as_continuous().T
    log.info(f"Computing dSTRF at {len(t_indexes)} timepoints, {dlc.shape[1]} DLC channels, t_step={time_step}")
    if rec.meta['batch'] in [346, 347]:
        dicount=didx.shape[0]
    else:
        dicount=4

    dstrf = {}
    mdstrf = np.zeros((len(model_list), dicount, rec['stim'].shape[0], D))
    pc1 = np.zeros((len(model_list), dicount, rec['stim'].shape[0], D))
    pc2 = np.zeros((len(model_list), dicount, rec['stim'].shape[0], D))
    pc_mag_all = np.zeros((len(model_list), dicount, pc_count))
    for di in range(dicount):
        dlc1 = dlc.copy()
        dcount = dlc1.shape[1]
        didx_ = adjust_didx(dlc, didx)
        #didx_ = didx

        for t in t_indexes:
            dlc1[(t-didx_.shape[1]+1):(t+1), :] = didx_[di,:,:dcount]
        log.info(f"DLC values: {np.round(didx[di,-1,:dcount],3)}")
        #log.info(f'di={di} Applying HRTF for frozen DLC coordinates')
        #rec2 = rec.copy()
        #rec2['dlc'] = rec2['dlc']._modified_copy(data=dlc1.T)
        #rec2 = free_tools.stim_filt_hrtf(rec2, hrtf_format='az', smooth_win=2,
        #                                 f_min=200, f_max=20000, channels=18)['rec']
        for mi, m in enumerate(model_list):
            stim = {input_name: rec['stim'].as_continuous().T, 'dlc': dlc1}
            dstrf[di] = m.dstrf(stim, D=D, out_channels=[out_channel], t_indexes=t_indexes, reset_backend=reset_backend)

            d = dstrf[di][input_name][0, :, :, :]

            if snr_threshold is not None:
                d = np.reshape(d, (d.shape[0], d.shape[1] * d.shape[2]))
                md = d.mean(axis=0, keepdims=True)
                e = np.std(d - md, axis=1) / np.std(md)
                if (e > snr_threshold).sum() > 0:
                    log.info(f"Removed {(e > snr_threshold).sum()}/{len(d)} noisy dSTRFs for PCA calculation")

                d = dstrf[di][input_name][0, (e <= snr_threshold), :, :]
                dstrf[di][input_name]=d[np.newaxis,:,:,:]
            mdstrf[mi, di, :, :] = d.mean(axis=0)
    
            # svd attempting to make compatible with new format of compute_pcs
            try:
                if (d.size>0) and (d.std()>0):
                    #pc, pc_mag = dtools.compute_dpcs(d[np.newaxis, :, :, :], pc_count=pc_count)
                    dpc = dtools.compute_dpcs(dstrf[di], pc_count=pc_count)
                    pc=dpc[input_name]['pcs']
                    pc_mag=dpc[input_name]['pc_mag']

                    pc1[mi, di, :, :] = pc[0, 0, :, :] * pc_mag[0, 0]
                    pc2[mi, di, :, :] = pc[0, 1, :, :] * pc_mag[1, 0]
                    pc_mag_all[mi, di, :] = pc_mag[:, 0]
            except:
                log.info('FAILED TO COMPUTE PCS. SETTING TO ZERO.')
    return mdstrf, pc1, pc2, pc_mag_all


def dstrf_plots(model_list, dstrf, out_channel, rec=None, cellid='cell', fs=50, interpolation_factor=None, flip_time=True,
                labels=None):
    if rec is not None:
        cellid = rec['resp'].chans[out_channel]
        fs = rec['resp'].fs
    if interpolation_factor is not None:
        fs=fs*interpolation_factor

    if labels is None:
        labels = ['HRTF+DLC', 'HRTF', 'DLC']

    f, ax = plt.subplots(len(model_list), dstrf.shape[1] + 1, figsize=(10, 8), sharex=True, sharey=True)
    for mi, m in enumerate(model_list):
        mmax = np.max(np.abs(dstrf[mi, :]))
        for di in range(dstrf.shape[1]):
            d = dstrf[mi, di]
            if interpolation_factor is not None:
                d = zoom(np.concatenate([d, np.zeros([d.shape[0], 1])], axis=1), interpolation_factor)[:,1:-1]
                #d=zoom(d, interpolation_factor, mode='constant')
            if flip_time:
                d=np.fliplr(d)
            mm = int(d.shape[0]/2)
            ax[mi, di].imshow(d[:mm,:], extent=[-0.5/fs, (d.shape[1]-0.5)/fs, -0.5, mm-0.5], 
                              vmin=-mmax, vmax=mmax, **imopts_dstrf)
            ax[mi, di].imshow(d[mm:,:], extent=[-0.5/fs, (d.shape[1]-0.5)/fs, mm+0.5, mm*2 + 0.5], 
                              vmin=-mmax, vmax=mmax, **imopts_dstrf)
            #ax[mi, di].imshow(dstrf[mi, di], vmin=-mmax, vmax=mmax, **imopts_dstrf)
            ax[mi, di].axhline(mm, color='k', ls='--', lw=0.5)
            if mi == len(model_list) - 1:
                ax[mi, di].set_xlabel(f"di={di}", fontsize=9)

        ax[mi, 0].text(0, 2, 'R')
        ax[mi, 0].text(0, mm+2, 'L')
        
        d = dstrf[mi, 2] - dstrf[mi, 0]
        if interpolation_factor is not None:
            d=zoom(np.concatenate([d,np.zeros([d.shape[0],1])], axis=1), interpolation_factor)[:,1:-1]
        if flip_time:
            d=np.fliplr(d)
        mm = int(d.shape[0]/2)
        mmax *= 0.8
        ax[mi, -1].imshow(d[:mm,:], extent=[-0.5/fs, (d.shape[1]-0.5)/fs, -0.5, mm-0.5], 
                          vmin=-mmax, vmax=mmax, **imopts_dstrf)
        ax[mi, -1].imshow(d[mm:,:], extent=[-0.5/fs, (d.shape[1]-0.5)/fs, mm+0.5, mm*2 + 0.5], 
                          vmin=-mmax, vmax=mmax, **imopts_dstrf)
        #ax[mi, -1].imshow(dstrf[mi, 2] - dstrf[mi, 0], vmin=-mmax, vmax=mmax, **imopts_dstrf)
        ax[mi, -1].axhline(mm, color='k', ls='--', lw=0.5)
        if mi == len(model_list) - 1:
            ax[mi, -1].set_xlabel(f"Front-back", fontsize=9)

        # ax[mi,3].set_title(f"{np.round(dlc[didx,:4],2)}")
        ax[mi, 2].set_title(f"{cellid} - {m.name}", fontsize=9)
        ax[mi, 0].set_ylabel(f"{labels[mi]} - r={m.meta['r_test'][out_channel, 0]:.3}", fontsize=9)
    plt.tight_layout()

    return f

def pop_models(rec, skip_dstrf=False, **model_opts):
    
    model = free_model.free_fit(rec, shuffle='none', save_to_db=True, **model_opts)
    model2 = free_model.free_fit(rec, shuffle='dlc', save_to_db=True, **model_opts)
    model3 = free_model.free_fit(rec, shuffle='stim', save_to_db=True, **model_opts)
    model4 = free_model.free_fit(rec, shuffle='none', apply_hrtf=False, save_to_db=True, **model_opts)
    model5 = free_model.free_fit(rec, shuffle='dlc', apply_hrtf=False, save_to_db=True, **model_opts)

    depth = rec.meta['depth']
    di = np.argsort(depth)
    siteid = rec.meta['siteid']
    batch = rec.meta['batch']

    f, ax = plt.subplots(2, 1, figsize=(8, 6))
    labels = ['space+vel', 'hrtf']
    for i, m in enumerate([model, model2]):
        ax[0].plot(depth[di], m.meta['r_test'][di] - model5.meta['r_test'][di], label=labels[i])
    ax[0].set_ylabel('improvement')
    ax[0].legend(fontsize=10)

    labels = ['full', 'y hrtf+no dlc', 'no aud+y dlc', 'no hrtf+y dlc', 'no hrtf+no dlc']
    ls = ['-', '-', ':', '-', '--']
    for i, m in enumerate([model, model2, model3, model4, model5]):
        ax[1].plot(depth[di], m.meta['r_test'][di], linestyle=ls[i],
                   label=f"{labels[i]} r={np.nanmean(m.meta['r_test']):.3f}")
    ax[1].set_ylabel('r_test')
    ax[1].set_xlabel('depth')
    ax[1].legend(fontsize=10)
    f.suptitle(siteid)
    outfile = f'{outpath}/{siteid}_{batch}_predcomp.png'
    log.info(f'Saving predictions to {outfile}')
    f.patch.set_facecolor('white')
    f.savefig(outfile, format='png')

    if skip_dstrf:
        return f

    f = movement_plot(rec)
    outfile = f'{outpath}/dstrf_{batch}/{siteid}_position.png'
    log.info(f'Saving position plot to {outfile}')
    f.patch.set_facecolor('white')
    f.savefig(outfile, format='png')

    good_channels = np.where(model.meta['r_test'][:,0]>0.15)[0]
    for i,out_channel in enumerate(good_channels):
        cellid = rec['resp'].chans[out_channel]
        log.info(f'Computing dSTRFs for {cellid} ({i}/{len(good_channels)}')
        mdstrf, pc1, pc2, pc_mag = dstrf_snapshots(rec, [model, model2, model4], D=11, out_channel=out_channel)
        f = dstrf_plots(rec, [model, model2, model4], mdstrf, out_channel)
        outfile = f'{outpath}/dstrf_{batch}/{cellid}_mdstrf.pdf'
        log.info(f'Saving mean dstrf plot to {outfile}')
        #f.patch.set_facecolor('white')
        f.savefig(outfile, format='pdf')
        f = dstrf_plots(rec, [model, model2, model4], pc1, out_channel)
        outfile = f'{outpath}/dstrf_{batch}/{cellid}_pc1.pdf'
        log.info(f'Saving dstrf pc1 plot to {outfile}')
        #f.patch.set_facecolor('white')
        f.savefig(outfile, format='pdf')

    return f

def fit_all_sites(batch=348):
    siteids, cellids = db.get_batch_sites(batch)
    for siteid in siteids:
        rec = load_rec(siteid, batch)
        pop_models(rec, skip_dstrf=True, acount=12, dcount=8, l2count=24,
                   cost_function='squared_error')

if __name__ == '__main__':

    if 'QUEUEID' in os.environ:
        queueid = os.environ['QUEUEID']
        nems0.utils.progress_fun = nd.update_job_tick

        if 'SLURM_JOB_ID' in os.environ:
            jobid = os.environ['SLURM_JOB_ID']
            nd.update_job_pid(jobid)
            nd.update_startdate()
            comment = ' '.join(sys.argv[1:])
            update_comment = ['sacctmgr', '-i', 'modify', 'job', f'jobid={jobid}', 'set', f'Comment="{comment}"']
            subprocess.run(update_comment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log.info(f'Set comment string to: "{comment}"')
    else:
        queueid = 0

    if queueid:
        log.info("Starting QUEUEID={}".format(queueid))
        nd.update_job_start(queueid)
        log.info("HOSTNAME={}".format(os.environ.get('HOSTNAME','unknown')))

    if len(sys.argv) < 3:
        print('syntax: free_vs_fixed_strfs siteid batch cmd<strf>')
        exit(-1)

    siteid = sys.argv[1]
    batch = int(sys.argv[2])
    if len(sys.argv)>=4:
        cmd=sys.argv[3]
    else:
        cmd = 'strf'
    log.info(f'batch: {batch} site: {siteid} cmd: {cmd}')

    if cmd == 'strf':
        rec = load_rec(siteid, batch)
        f = fit_strfs(rec)

        outfile=f'{outpath}/{siteid}_{batch}.png'
        log.info(f'Saving STRFs to {outfile}')
        f.savefig(outfile, format='png')
    elif cmd == 'dstrf':
        rec = load_rec(siteid, batch)
        pop_models(rec, acount=12, dcount=8, l2count=24)
    elif cmd == 'cnn':
        rec = load_rec(siteid, batch)
        pop_models(rec, skip_dstrf=True, acount=12, dcount=8, l2count=24,
                   cost_function='squared_error')

    else:
        log.info(f"Unknown command {cmd}")
