#stardard imports
import os
import numpy as np
import logging
import sys
import matplotlib.pyplot as plt

import nems0.utils
import nems0.preprocessing as preproc
import nems_lbhb.projects.freemoving.free_tools
from nems_lbhb.projects.freemoving import free_model, free_tools
from nems0.epoch import epoch_names_matching
import nems
from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, DoubleExponential, RectifiedLinear
from nems import visualization

log = logging.getLogger(__name__)

imopts_dstrf = {'origin': 'lower',
                'interpolation': 'none',
                'cmap': 'bwr',
                'aspect': 'auto'}
outpath = '/auto/users/svd/projects/free_moving/strfs'

didx = np.array([[[0.117, 0.698, 0.116, 0.651, 0.154, 0.654, 0.055, 0.646],
        [0.116, 0.697, 0.115, 0.649, 0.153, 0.652, 0.054, 0.645],
        [0.116, 0.696, 0.115, 0.647, 0.152, 0.65 , 0.053, 0.645],
        [0.116, 0.696, 0.115, 0.647, 0.152, 0.65 , 0.052, 0.645],
        [0.116, 0.696, 0.115, 0.647, 0.152, 0.65 , 0.052, 0.645],
        [0.116, 0.695, 0.116, 0.647, 0.154, 0.651, 0.052, 0.644],
        [0.116, 0.695, 0.116, 0.647, 0.154, 0.651, 0.052, 0.644],
        [0.116, 0.695, 0.116, 0.647, 0.154, 0.651, 0.052, 0.644],
        [0.116, 0.695, 0.114, 0.647, 0.153, 0.65 , 0.051, 0.643],
        [0.116, 0.695, 0.114, 0.647, 0.153, 0.65 , 0.051, 0.643],
        [0.116, 0.695, 0.114, 0.647, 0.152, 0.65 , 0.051, 0.643]],
       [[0.53 , 0.453, 0.477, 0.494, 0.468, 0.448, 0.507, 0.546],
        [0.53 , 0.453, 0.477, 0.494, 0.468, 0.448, 0.507, 0.546],
        [0.53 , 0.453, 0.477, 0.494, 0.468, 0.448, 0.507, 0.546],
        [0.53 , 0.453, 0.477, 0.494, 0.468, 0.448, 0.507, 0.546],
        [0.537, 0.444, 0.489, 0.486, 0.478, 0.439, 0.517, 0.538],
        [0.547, 0.433, 0.503, 0.475, 0.491, 0.428, 0.529, 0.526],
        [0.561, 0.427, 0.515, 0.462, 0.502, 0.417, 0.547, 0.509],
        [0.575, 0.409, 0.529, 0.44 , 0.513, 0.397, 0.567, 0.481],
        [0.588, 0.379, 0.544, 0.41 , 0.526, 0.37 , 0.59 , 0.443],
        [0.591, 0.373, 0.547, 0.404, 0.528, 0.365, 0.593, 0.437],
        [0.592, 0.372, 0.547, 0.404, 0.529, 0.365, 0.593, 0.437]],
       [[0.751, 0.122, 0.747, 0.178, 0.703, 0.155, 0.807, 0.2  ],
        [0.751, 0.122, 0.746, 0.178, 0.703, 0.156, 0.807, 0.2  ],
        [0.751, 0.122, 0.746, 0.178, 0.703, 0.155, 0.807, 0.2  ],
        [0.751, 0.123, 0.744, 0.178, 0.702, 0.154, 0.807, 0.197],
        [0.751, 0.123, 0.742, 0.178, 0.7  , 0.153, 0.808, 0.195],
        [0.751, 0.123, 0.742, 0.178, 0.7  , 0.153, 0.808, 0.195],
        [0.751, 0.123, 0.742, 0.178, 0.7  , 0.152, 0.808, 0.196],
        [0.751, 0.123, 0.742, 0.177, 0.7  , 0.152, 0.807, 0.197],
        [0.75 , 0.124, 0.74 , 0.178, 0.7  , 0.153, 0.806, 0.199],
        [0.749, 0.124, 0.738, 0.179, 0.7  , 0.154, 0.805, 0.2  ],
        [0.749, 0.124, 0.738, 0.179, 0.7  , 0.154, 0.805, 0.2  ]],
       [[0.268, 0.204, 0.313, 0.209, 0.306, 0.249, 0.326, 0.131],
        [0.268, 0.205, 0.313, 0.209, 0.306, 0.249, 0.326, 0.131],
        [0.269, 0.205, 0.314, 0.209, 0.307, 0.249, 0.326, 0.131],
        [0.27 , 0.205, 0.314, 0.209, 0.307, 0.249, 0.326, 0.131],
        [0.26 , 0.208, 0.303, 0.213, 0.296, 0.252, 0.314, 0.133],
        [0.244, 0.214, 0.286, 0.22 , 0.28 , 0.257, 0.295, 0.136],
        [0.188, 0.252, 0.226, 0.254, 0.225, 0.291, 0.229, 0.185],
        [0.153, 0.276, 0.187, 0.275, 0.19 , 0.313, 0.187, 0.217],
        [0.152, 0.276, 0.187, 0.275, 0.19 , 0.313, 0.187, 0.217],
        [0.152, 0.276, 0.187, 0.275, 0.19 , 0.313, 0.186, 0.216],
        [0.153, 0.276, 0.187, 0.276, 0.19 , 0.313, 0.186, 0.215]],
       [[0.5  , 0.2  , 0.5  , 0.25 , 0.46 , 0.25 , 0.53 , 0.25 ],
        [0.5  , 0.2  , 0.5  , 0.25 , 0.46 , 0.25 , 0.53 , 0.25 ],
        [0.5  , 0.2  , 0.5  , 0.25 , 0.46 , 0.25 , 0.53 , 0.25 ],
        [0.5  , 0.2  , 0.5  , 0.25 , 0.46 , 0.25 , 0.53 , 0.25 ],
        [0.5  , 0.2  , 0.5  , 0.25 , 0.46 , 0.25 , 0.53 , 0.25 ],
        [0.5  , 0.2  , 0.5  , 0.25 , 0.46 , 0.25 , 0.53 , 0.25 ],
        [0.5  , 0.2  , 0.5  , 0.25 , 0.46 , 0.25 , 0.53 , 0.25 ],
        [0.5  , 0.2  , 0.5  , 0.25 , 0.46 , 0.25 , 0.53 , 0.25 ],
        [0.5  , 0.2  , 0.5  , 0.25 , 0.46 , 0.25 , 0.53 , 0.25 ],
        [0.5  , 0.2  , 0.5  , 0.25 , 0.46 , 0.25 , 0.53 , 0.25 ],
        [0.5  , 0.2  , 0.5  , 0.25 , 0.46 , 0.25 , 0.53 , 0.25 ]]])

def fit_strf(stim,resp):
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


def pop_models(rec):
    model = free_model.free_fit(rec, shuffle='none', dlc_memory=4)
    model2 = free_model.free_fit(rec, shuffle='dlc', dlc_memory=4)
    model3 = free_model.free_fit(rec, shuffle='stim', dlc_memory=4)
    model4 = free_model.free_fit(rec, shuffle='none', apply_hrtf=False, dlc_memory=4)
    model5 = free_model.free_fit(rec, shuffle='dlc', apply_hrtf=False, dlc_memory=4)

    depth = rec.meta['depth']
    di = np.argsort(depth)

    f, ax = plt.subplots(2, 1, figsize=(8, 6))
    labels = ['space+vel', 'hrtf']
    for i, m in enumerate([model, model2]):
        ax[0].plot(depth[di], m.meta['r_test'][di] - model5.meta['r_test'][di], label=labels[i])
    ax[0].set_ylabel('improvement')
    ax[0].legend(fontsize=10);

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
    log.info(f'Saving STRFs to {outfile}')
    f.patch.set_facecolor('white')
    f.savefig(outfile, format='png')

    f = movement_plot(rec)
    outfile = f'{outpath}/dstrf_{batch}/{siteid}_position.png'
    log.info(f'Saving position plot to {outfile}')
    f.patch.set_facecolor('white')
    f.savefig(outfile, format='png')

    good_channels = np.where(model.meta['r_test'][:,0]>0.15)[0]
    for i,out_channel in enumerate(good_channels):
        cellid = rec['resp'].chans[out_channel]
        log.info(f'Computing dSTRFs for {cellid} ({i}/{len(good_channels)}')
        mdstrf = dstrf_snapshots(rec, [model, model2, model4], D=12, out_channel=out_channel)
        f = dstrf_plots(rec, [model, model2, model4], mdstrf, out_channel)
        outfile = f'{outpath}/dstrf_{batch}/{cellid}_dstrf.png'
        log.info(f'Saving dstrf plot to {outfile}')
        f.patch.set_facecolor('white')
        f.savefig(outfile, format='png')

    return f

def movement_plot(rec):
    speaker1_x0y0 = 1.0, -0.8
    speaker2_x0y0 = 0.0, -0.8
    fs = rec['dlc'].fs

    dlc = rec['dlc'].as_continuous()[:8, :].T
    f=plt.figure()
    plt.scatter(dlc[::10, 0], dlc[::10, 1], s=2, color='lightgray')
    for i in range(len(didx)):

        # compute distance and angle to each speaker
        # code pasted in from free_tools
        d1, theta1, vel, rvel, d_fwd, d_lat = free_tools.compute_d_theta(
            didx[i].T, fs=fs, smooth_win=0.1, ref_x0y0=speaker1_x0y0)
        d2, theta2, vel, rvel, d_fwd, d_lat = free_tools.compute_d_theta(
            didx[i].T, fs=fs, smooth_win=0.1, ref_x0y0=speaker2_x0y0)
        #log.info(f"{i} {didx[i,-1,:2]} d1={d1[0,-1]} th1= {d2[0,-1]}")
        plt.plot(didx[i, -1, 2], didx[i, -1, 3], "o", color='blue')
        plt.plot(didx[i, -1, 0], didx[i, -1, 1], "o", color='red')
        plt.text(didx[i, -1, 0], didx[i, -1, 1],
                 f"{i}: d,th1=({d1[0,-1]:.1f},{theta1[0,-1]:.0f})\n  d,th2=({d2[0,-1]:.1f},{theta2[0,-1]:.0f})",
                 va='center')
        plt.plot(didx[i, -6:, 0], didx[i, -6:, 1], color='darkblue', lw=1)
    plt.gca().invert_yaxis()
    plt.title(rec.meta['siteid'], fontsize=12)
    return f

def dstrf_snapshots(rec, model_list, D=11, out_channel=0):
    t_indexes = np.arange(100, rec['stim'].shape[1], 100)
    dlc = rec['dlc'].as_continuous()[:8, :].T

    if rec.meta['batch'] in [346,347]:
        dicount=4
    else:
        dicount=5
    dstrf = {}
    mdstrf = np.zeros((len(model_list), dicount, rec['stim'].shape[0], D))
    for di in range(dicount):
        dlc1 = dlc.copy()
        for t in t_indexes:
            dlc1[(t - didx.shape[1] + 1):(t + 1), :] = didx[di]

        log.info(f'di={di} Applying HRTF for frozen DLC coordinates')
        rec2 = rec.copy()
        rec2['dlc'] = rec2['dlc']._modified_copy(data=dlc1.T)
        rec2 = free_tools.stim_filt_hrtf(rec2, hrtf_format='az', smooth_win=2,
                                                                       f_min=200, f_max=20000, channels=18)['rec']

        for mi, m in enumerate(model_list):
            if mi in [0, 1]:
                # stim with hrtf
                stim = {'stim': rec2['stim'].as_continuous().T, 'dlc': dlc1}
            else:
                stim = {'stim': rec['stim'].as_continuous().T, 'dlc': dlc1}
            dstrf[di] = m.dstrf(stim, D=D, out_channels=[out_channel], t_indexes=t_indexes)
            mdstrf[mi, di, :, :] = dstrf[di]['stim'][0, :, :, :].mean(axis=0)

    return mdstrf

def dstrf_plots(rec, model_list, mdstrf, out_channel):
    cellid = rec['resp'].chans[out_channel]
    labels = ['HRTF+DLC', 'HRTF', 'DLC']

    f, ax = plt.subplots(len(model_list), len(didx) + 1, figsize=(10, 8), sharex=True, sharey=True)
    for mi, m in enumerate(model_list):
        mmax = np.max(np.abs(mdstrf[mi, :]))
        for di in range(mdstrf.shape[1]):
            ax[mi, di].imshow(mdstrf[mi, di], vmin=-mmax, vmax=mmax, **imopts_dstrf)
            ax[mi, di].axhline(17.5, color='k', ls='--', lw=0.5)
            if mi == len(model_list) - 1:
                ax[mi, di].set_xlabel(f"di={di}", fontsize=12)
        mm = np.max(np.abs(mdstrf[mi, 2] - mdstrf[mi, 0]))

        ax[mi, 0].text(0, 20, 'L')
        ax[mi, 0].text(0, 2, 'R')

        ax[mi, -1].imshow(mdstrf[mi, 2] - mdstrf[mi, 0], vmin=-mmax / 2, vmax=mmax / 2, **imopts_dstrf)
        ax[mi, -1].axhline(17.5, color='k', ls='--', lw=0.5)
        if mi == len(model_list) - 1:
            ax[mi, -1].set_xlabel(f"Front-back", fontsize=12)

        # ax[mi,3].set_title(f"{np.round(dlc[didx,:4],2)}")
        ax[mi, 2].set_title(f"{cellid} - {m.name}", fontsize=12)
        ax[mi, 0].set_ylabel(f"{labels[mi]} - r={m.meta['r_test'][out_channel, 0]:.3}", fontsize=12)
    plt.tight_layout()
    return f

def load_rec(siteid, batch):
    dlc_chans = 8
    rasterfs = 50
    rec = free_model.load_free_data(siteid, batch=batch, rasterfs=rasterfs, dlc_chans=dlc_chans)

    rec = free_tools.stim_filt_hrtf(rec, hrtf_format='az', smooth_win=2,
                                                                  f_min=200, f_max=20000, channels=18)['rec']
    return rec


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
        pop_models(rec)
    else:
        log.info(f"Unknown command {cmd}")