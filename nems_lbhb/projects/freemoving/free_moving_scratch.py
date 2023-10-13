from os.path import basename, join
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter, zoom
import importlib

import datetime
import os

from nems0 import db
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
from nems_lbhb.projects.freemoving import free_model, free_vs_fixed_strfs
from nems.tools import json
from nems_lbhb.baphy_io import load_continuous_openephys, get_spike_info, get_depth_info

# code to support dumping figures
#dt = datetime.date.today().strftime("%Y-%m-%d")
#figpath = f'/auto/users/svd/docs/current/grant/r21_free_moving/eps/{dt}/'
#os.makedirs(figpath, exist_ok=True)

# tested sites
siteid = 'PRN020a'
siteid = 'PRN010a'
siteid = 'PRN015a'
siteid = 'PRN034a'
siteid = 'PRN018a'
siteid = 'PRN022a'
siteid = 'PRN043a'
siteid = 'PRN051a'

# interesting sites
siteid = 'PRN067a' # ok both
siteid = 'PRN015a' # nice aud, single stream
siteid = 'PRN047a' # some of everything.
siteid = 'PRN074a' # ok both

siteid = 'SLJ021a'
siteid = 'PRN009a'
siteid = 'PRN048a' # some of everything.

batch=348
siteids, cellids = db.get_batch_sites(batch=batch)

dlc_chans=8
rasterfs=50

#ANALYSIS = 'summary'
#ANALYSIS = 'fit'
#ANALYSIS = 'dstrf example'
#ANALYSIS = 'scratch'
ANALYSIS='movement diagram'
#ANALYSIS= 'pop stats'

cost_function='nmse'
#cost_function='squarederror'

modelnamemask = f"wc.Nx1x12-fir.8x1x12-wc.Dx1x8.dlc-fir.4x1x8.dlc-concat.space-relu.20.f-wc.20x1x24-fir.4x1x24-relu.24.f-wc.24xR-relu.R_{cost_function}"

figpath = '/home/svd/Documents/onedrive/presentations/rochester_2023/eps'

if ANALYSIS=='summary':
    # copied from free_model_scratch.ipynb
    depthinfo = [get_spike_info(siteid=s).reset_index() for s in siteids]
    depthinfo = pd.concat(depthinfo, ignore_index=True)

    modelnames = [f'sh.none-hrtf.True_{modelnamemask}', f'sh.none-hrtf.False_{modelnamemask}', f'sh.dlc-hrtf.True_{modelnamemask}',
                  f'sh.dlc-hrtf.False_{modelnamemask}', f'sh.stim-hrtf.True_{modelnamemask}']
    shortnames = ['DLC/HRTF', 'DCL/no HRTF', 'no DLC/HRTF', 'no DLC/no HRTF', 'DLC/no Aud']
    d = db.batch_comp(batch=batch, modelnames=modelnames, shortnames=shortnames).reset_index()
    d_floor = db.batch_comp(batch=batch, modelnames=modelnames, stat='r_floor', shortnames=shortnames).reset_index()
    d_good = (d.iloc[:, 1:6] > d_floor.iloc[:, 1:6]).sum(axis=1)
    cnngood = (d.iloc[:, 1:6] > d_floor.iloc[:, 1:6])

    d = d.merge(depthinfo, how='inner', left_on='cellid', right_on='index')
    d['depth'] = d['depth'].astype(float)
    d['iso'] = d['iso'].astype(float)
    depth_valid = (d['depth'] > -800) & (d['depth'] < 1600)
    d['narrow']=d['sw']<0.35
    d_good = (d_good >= 3) & (d.iloc[:, 1].values > d.iloc[:, [2, 3, 4, 5]].values.max(axis=1) / 1.5)
    d = d.loc[d_good & depth_valid]
    print(d_good.sum(), depth_valid.sum(), (d_good & depth_valid).sum(),
          (d_good & depth_valid).sum() / depth_valid.sum(), d_good.sum() / len(d_good))

    f, ax = plt.subplots()
    d.loc[d_good].groupby('area')[shortnames].mean().T.plot.bar(ax=ax);
    plt.tight_layout()
    f.savefig(f"{figpath}/model_area_mean_rtest.pdf")

    f, ax = plt.subplots()
    d.loc[d_good].groupby('narrow')[shortnames].mean().T.plot.bar(ax=ax);
    #d.groupby('area')[shortnames].mean().T.plot.bar(ax=ax);
    plt.tight_layout()
    f.savefig(f"{figpath}/model_sw_mean_rtest.pdf")

    f, ax = plt.subplots(figsize=(12, 4))
    d.groupby('siteid')[[shortnames[0], shortnames[4]]].mean().T.plot.bar(ax=ax);
    f.savefig(f"{figpath}/mean_per_site.pdf")

    f = plt.figure()
    plt.hist(d.loc[d['sw']<1,'sw'].values, bins=15, color="black", lw=0)
    plt.xlabel('Spike width (ms)')
    plt.ylabel('Number of units')
    f.savefig(f"{figpath}/spike_width_hist.pdf")

    #d_=d.loc[d['sw']<0.3].copy()
    # d_ = d.loc[d['area']=='A1'].copy()
    d_ = d.copy()

    d_['depth'] = d_['depth'].astype(float)
    d_['depth_group'] = np.round(d_['depth'].astype(float) / 75) * 75
    b1 = 'no DLC/HRTF'
    b2 = 'no DLC/no HRTF'
    d_['space'] = d_['DLC/HRTF'] - d_[b1]
    d_['space_mi'] = d_['space'] / (d_['DLC/HRTF'] + d_[b1])
    d_['space2'] = d_[b1] - d_[b2]
    d_['space_mi2'] = d_['space2'] / (d_[b1] + d_[b2])
    d_['mean_Full_NoSpace'] = (d_['DLC/HRTF'] + d_[b2]) / 2

    d_['frAud'] = d_['no DLC/no HRTF'] ** 2 / d_['DLC/HRTF'] ** 2
    d_['frHRTF'] = d_['no DLC/HRTF'] ** 2 / d_['DLC/HRTF'] ** 2
    d_['frDLC'] = d_['DCL/no HRTF'] ** 2 / d_['DLC/HRTF'] ** 2
    d_['frNoAud'] = d_['DLC/no Aud'] ** 2 / d_['DLC/HRTF'] ** 2

    dg = d_.groupby('depth_group').mean()
    de = d_.groupby('depth_group').sem().fillna(0)
    dc = d_.groupby('depth_group').count()

    f, ax = plt.subplots(2, 3, figsize=(14, 5))

    dg[['DLC/HRTF', 'no DLC/HRTF', 'no DLC/no HRTF', 'DLC/no Aud']].plot(ax=ax[0, 0])
    ax[0, 0].legend(fontsize=9, frameon=False)
    ax[0, 0].set_ylabel('mean pred corr')

    d_.plot.scatter('depth', 'space', s=2, ax=ax[0, 1])
    ax[0, 1].axhline(0, color='black', linestyle='--', lw=0.5)
    dg.plot.scatter('depth', 'space', ax=ax[0, 1], color='red')
    ax[0, 2].axhline(0, color='black', linestyle='--', lw=0.5)
    d_.plot.scatter(b2, 'space2', ax=ax[0, 2], s=10, color='black')
    ax[1, 0].axhline(dg['space_mi'].median(), color='black', linestyle='--', lw=0.5)
    ax[1, 0].errorbar(dg.index, dg['space_mi'], de['space_mi'], label='+DLC')
    ax[1, 0].errorbar(dg.index, dg['space_mi2'], de['space_mi2'], label='+HRTF')
    ax[1, 0].set_ylabel('Modulation index (MI)')
    ax[1, 0].legend(frameon=False, fontsize=7)
    ax[1, 0].set_xlabel('Depth from L3/4 boundary (um)')
    ax[1, 1].plot([dg.index.min(), dg.index.max()], [1, 1], color='k', label='Posture+HRTF')
    ax[1, 1].errorbar(dg.index, dg['frDLC'], de['frDLC'], label='Posture (no HRTF)')
    ax[1, 1].errorbar(dg.index, dg['frHRTF'], de['frHRTF'], label='HRTF (no Posture)')
    ax[1, 1].errorbar(dg.index, dg['frAud'], de['frAud'], label='Aud (no HRTF/no Posture)')
    # ax[1,1].errorbar(dg.index, dg['frNoAud'], de['frAud'], label='frac NoAud')
    ax[1, 1].axhline(0, color='black', lw=0.5, linestyle='--')
    ax[1, 1].legend(fontsize=9, frameon=False)
    ax[1, 1].set_ylabel('Fraction explained variance')
    ax[1, 1].set_xlabel('Depth from L3/4 boundary (um)')
    ax[1, 2].plot(dc.index, dc['space'])
    ax[1, 2].set_ylabel('counts')
    ax[1, 2].set_xlabel('Depth from L3/4 boundary (um)')
    f.suptitle(modelnamemask)
    plt.tight_layout()

    outfile = f'{figpath}/depth_summary_{cost_function}.pdf'
    print(f'Saving plot to {outfile}')
    f.savefig(outfile, format='pdf')


elif ANALYSIS=='fit':
    rec = free_model.load_free_data(siteid, batch=batch, rasterfs=rasterfs, dlc_chans=dlc_chans, compute_position=True)
    modelopts={'dlc_memory': 4, 'acount': 12, 'dcount': 8, 'l2count': 24, 'cost_function': 'squared_error'}
    modelopts={'dlc_memory': 4, 'acount': 12, 'dcount': 8, 'l2count': 24, 'cost_function': 'nmse'}

    models = [None, None, None]
    loadops = {'shuffle': 'none', 'apply_hrtf': True}
    models[0] = free_model.free_fit(rec, save_to_db=False, **loadops, **modelopts)

    loadops = {'shuffle': 'none', 'apply_hrtf': False}
    models[1] = free_model.free_fit(rec, save_to_db=False, **loadops, **modelopts)

    loadops = {'shuffle': 'dlc', 'apply_hrtf': True}
    models[2] = free_model.free_fit(rec, save_to_db=False, **loadops, **modelopts)

    # scatter plot of free-moving position with example positions highlighted
    f1,f2 = free_vs_fixed_strfs.movement_plot(rec)
    f1.savefig(f"{figpath}/position_scatter_{siteid}.pdf")
    f2.savefig(f"{figpath}/position_spect_{siteid}.pdf")

    ctx1 = free_model.free_split_rec(rec, apply_hrtf=True)
    ctx2 = free_model.free_split_rec(rec, apply_hrtf=False)
    est1 = ctx1['est'].apply_mask()
    est2 = ctx2['est'].apply_mask()
    val1 = ctx1['est'].apply_mask()
    val2 = ctx2['est'].apply_mask()

    # dSTRFS for interesting units: PRN048a-269-1, PRN048a-285-2
    labels = ['HRTF+DLC', 'DLC', 'HRTF']
    #for out_channel in [20,22]:
    for out_channel in [6, 8]:  # 5
        cellid = rec['resp'].chans[out_channel]
        mdstrf, pc1, pc2, pc_mag = free_vs_fixed_strfs.dstrf_snapshots(val2, models, D=11, out_channel=out_channel)
        f = free_vs_fixed_strfs.dstrf_plots(models, mdstrf, out_channel, rec=val2, labels=labels)

    for i,c in enumerate(models[0].meta['cellids']):
        print(f"{i}: {c} {models[0].meta['r_test'][i,0]:.3f} {models[1].meta['r_test'][i,0]:.3f} {models[2].meta['r_test'][i,0]:.3f} {rec.meta['depth'][i]}")
    print(f"MEAN              {models[0].meta['r_test'].mean():.3f} {models[1].meta['r_test'].mean():.3f} {models[2].meta['r_test'].mean():.3f}")

elif ANALYSIS=='movement diagram':

    modelnamemask=f"wc.Nx1x12-fir.8x1x12-wc.Dx1x8.dlc-fir.4x1x8.dlc-concat.space-relu.20.f-wc.20x1x24-fir.4x1x24-relu.24.f-wc.24xR-relu.R_{cost_function}"

    df = db.pd_query(f"SELECT DISTINCT modelname,modelfile FROM Results WHERE batch={batch} AND cellid like '{siteid}%' and modelname like '%{modelnamemask}'")

    modelfiles = [df.loc[df['modelname'].str.contains('sh.none-hrtf.True'), 'modelfile'].values[0],
                  df.loc[df['modelname'].str.contains('sh.none-hrtf.False'), 'modelfile'].values[0],
                  df.loc[df['modelname'].str.contains('sh.dlc-hrtf.True'), 'modelfile'].values[0]]

    rec = free_model.load_free_data(siteid, batch=batch, rasterfs=rasterfs, dlc_chans=dlc_chans, compute_position=True)
    models = [json.load_model(f) for f in modelfiles]

    # scatter plot of free-moving position with example positions highlighted
    f1,f2 = free_vs_fixed_strfs.movement_plot(rec)
    f1.savefig(f"{figpath}/position_scatter_{siteid}.pdf")
    f2.savefig(f"{figpath}/position_spect_{siteid}.pdf")

elif ANALYSIS=='pop stats':
    dfs = []
    for siteid in siteids:
        print(siteid + '...')
        # for a single siteid...
        df = db.pd_query(f"SELECT DISTINCT modelname,modelfile FROM Results WHERE batch={batch} AND cellid like '{siteid}%' and modelname like '%{modelnamemask}'")

        modelfiles = [df.loc[df['modelname'].str.contains('sh.none-hrtf.True'), 'modelfile'].values[0],
                      df.loc[df['modelname'].str.contains('sh.none-hrtf.False'), 'modelfile'].values[0],
                      df.loc[df['modelname'].str.contains('sh.dlc-hrtf.True'), 'modelfile'].values[0]]
        models = [json.load_model(f) for f in modelfiles]

        dstrffiles = [f.replace('modelspec.json', 'dstrf.npz') for f in modelfiles]
        dstrfdata = [np.load(f) for f in dstrffiles]

        mdstrfs = np.concatenate([d['mdstrfs'] for d in dstrfdata], axis=1)
        fbdiff = mdstrfs[:, :, 2, :, :] - mdstrfs[:, :, 0, :, :]

        #fbmod = fbdiff.std(axis=(2, 3)) / (
        #            mdstrfs[:, :, 0, :, :].std(axis=(2, 3)) + mdstrfs[:, :, 2, :, :].std(axis=(2, 3))).mean(axis=1,keepdims=True)
        fbmod = np.concatenate([d['fbmod'] for d in dstrfdata], axis=1)/2
        r_test = np.concatenate([d['r_test'] for d in dstrfdata], axis=1)
        pc_mag = np.concatenate([d['pc_mags'] for d in dstrfdata], axis=1)
        pc_rat = pc_mag[:,:,:,0] / pc_mag.sum(axis=3)
        pc_rat_mean=pc_rat.mean(axis=2)
        mdstrfs = np.concatenate([d['mdstrfs'] for d in dstrfdata], axis=1)
        cellids=models[0].meta['cellids']

        dfsite = pd.DataFrame({'cellid': cellids, 'siteid': siteid,
                               'rDH': r_test[:,0], 'rD': r_test[:,1], 'rH': r_test[:,2],
                               'fbDH': fbmod[:,0], 'fbD': fbmod[:,1], 'fbH': fbmod[:,2],
                               'pcDH': pc_rat_mean[:, 0], 'pcD': pc_rat_mean[:, 1], 'pcH': pc_rat_mean[:, 2],
                               })
        dfs.append(dfsite)

    dfsite = pd.concat(dfs, ignore_index=True)
    dfsite = dfsite[dfsite.rDH+dfsite.rD>0.2]

    f, ax = plt.subplots(2,3, figsize=(6, 4)) # , sharex='col', sharey='col')

    dfsite.plot.scatter(x='rH', y='rDH', s=3, ax=ax[0,0])
    ax[0,0].plot([0,0.7],[0,0.7],'k--', lw=0.5)
    dfsite.plot.scatter(x='fbH', y='fbDH', s=3, ax=ax[0,1])
    ax[0,1].plot([0,0.8],[0,0.8],'k--', lw=0.5)
    dfsite.plot.scatter(x='pcH', y='pcDH', s=3, ax=ax[0,2])
    ax[0,2].plot([0.2,0.8],[0.2,0.8],'k--', lw=0.5)

    dfsite.plot.scatter(x='rD', y='rDH', s=3, ax=ax[1,0])
    ax[1,0].plot([0,0.7],[0,0.7],'k--', lw=0.5)
    dfsite.plot.scatter(x='fbD', y='fbDH', s=3, ax=ax[1,1])
    ax[1,1].plot([0,0.8],[0,0.8],'k--', lw=0.5)
    dfsite.plot.scatter(x='pcD', y='pcDH', s=3, ax=ax[1,2])
    ax[1,2].plot([0.2,0.8],[0.2,0.8],'k--', lw=0.5)

    plt.tight_layout()
    f.savefig(f"{figpath}/sum_dstrf_stats.pdf")

elif ANALYSIS=='dstrf example':

    df = db.pd_query(f"SELECT DISTINCT modelname,modelfile FROM Results WHERE batch={batch} AND cellid like '{siteid}%' and modelname like '%{modelnamemask}'")

    modelfiles = [df.loc[df['modelname'].str.contains('sh.none-hrtf.True'), 'modelfile'].values[0],
                  df.loc[df['modelname'].str.contains('sh.none-hrtf.False'), 'modelfile'].values[0],
                  df.loc[df['modelname'].str.contains('sh.dlc-hrtf.True'), 'modelfile'].values[0]]
    models = [json.load_model(f) for f in modelfiles]

    dstrffiles = [f.replace('modelspec.json', 'dstrf.npz') for f in modelfiles]
    dstrfdata = [np.load(f) for f in dstrffiles]
    fbmod = np.concatenate([d['fbmod'] for d in dstrfdata], axis=1)
    r_test = np.concatenate([d['r_test'] for d in dstrfdata], axis=1)
    pc_mag = np.concatenate([d['pc_mags'] for d in dstrfdata], axis=1)
    pc_rat = pc_mag[:,:,:,0] / pc_mag.sum(axis=3)
    pc_rat_mean=pc_rat.mean(axis=2)
    mdstrfs = np.concatenate([d['mdstrfs'] for d in dstrfdata], axis=1)
    cellids=models[0].meta['cellids']

    for i,c in enumerate(cellids):
        #print(f"{i}: {c} {models[0].meta['r_test'][i,0]:6.3f} {models[1].meta['r_test'][i,0]:6.3f} {models[2].meta['r_test'][i,0]:6.3f}")
        print(f"{i}: {c} {fbmod[i, 0]:6.3f} {fbmod[i, 1]:6.3f} {fbmod[i, 2]:6.3f} {r_test[i, 0]:6.3f} {r_test[i, 1]:6.3f} {r_test[i, 2]:6.3f} {pc_rat_mean[i, 0]:6.3f} {pc_rat_mean[i, 1]:6.3f} {pc_rat_mean[i, 2]:6.3f}")
    print(f"MEAN             {fbmod[:, 0].mean():6.3f} {fbmod[:, 1].mean():6.3f} {fbmod[:, 2].mean():6.3f} {r_test[:, 0].mean():6.3f} {r_test[:, 1].mean():6.3f} {r_test[:, 2].mean():6.3f}")

    labels = ['HRTF+DLC', 'DLC', 'HRTF']
    importlib.reload(free_vs_fixed_strfs)
    for out_channel in [i for i, r in enumerate(r_test[:,0]) if r>0.2]:
        f = free_vs_fixed_strfs.dstrf_plots(models, mdstrfs[out_channel], out_channel, interpolation_factor=2, cellid=cellids[out_channel], fs=50, labels=labels)
        f.savefig(f"{figpath}/dstrf_example_{cellids[out_channel]}.pdf")

elif ANALYSIS=='scratch':
    modelnamemask="wc.Nx1x12-fir.8x1x12-wc.Dx1x8.dlc-fir.4x1x8.dlc-concat.space-relu.20.f-wc.20x1x24-fir.4x1x24-relu.24.f-wc.24xR-relu.R_squarederror"

    df = db.pd_query(f"SELECT DISTINCT modelname,modelfile FROM Results WHERE batch={batch} AND cellid like '{siteid}%' and modelname like '%{modelnamemask}'")

    modelfiles = [df.loc[df['modelname'].str.contains('sh.none-hrtf.True'), 'modelfile'].values[0],
                  df.loc[df['modelname'].str.contains('sh.none-hrtf.False'), 'modelfile'].values[0],
                  df.loc[df['modelname'].str.contains('sh.dlc-hrtf.True'), 'modelfile'].values[0]]

    rec = free_model.load_free_data(siteid, batch=batch, rasterfs=rasterfs, dlc_chans=dlc_chans, compute_position=True)
    models = [json.load_model(f) for f in modelfiles]

    ctx1 = free_model.free_split_rec(rec, apply_hrtf=True)
    ctx2 = free_model.free_split_rec(rec, apply_hrtf=False)
    est1 = ctx1['est'].apply_mask()
    est2 = ctx2['est'].apply_mask()
    val1 = ctx1['est'].apply_mask()
    val2 = ctx2['est'].apply_mask()

    # dSTRFS for interesting units: PRN048a-269-1, PRN048a-285-2
    labels = ['HRTF+DLC', 'DLC', 'HRTF']
    #for out_channel in [20,22]:
    pc_mags = []
    mdstrfs = []
    for out_channel in range(rec['resp'].shape[0]):
        cellid = rec['resp'].chans[out_channel]
        mdstrf, pc1, pc2, pc_mag = free_vs_fixed_strfs.dstrf_snapshots(rec, models, D=11, out_channel=out_channel, pc_count=5)
        if out_channel in [5,6,8]:   # [5]
            f = free_vs_fixed_strfs.dstrf_plots(models, mdstrf, out_channel, rec=rec, labels=labels)
        pc_mags.append(pc_mag)  # unit x model x didx x pc
        mdstrfs.append(mdstrf)   # unit x model x didx x frequency x lag
    pc_mags = np.stack(pc_mags, axis=0)
    mdstrfs = np.stack(mdstrfs, axis=0)
    fbdiff = mdstrfs[:,:,2,:,:]-mdstrfs[:,:,0,:,:]
    fbmod = fbdiff.std(axis=(2,3))/(mdstrfs[:,:,0,:,:].std(axis=(2,3))+mdstrfs[:,:,2,:,:].std(axis=(2,3)))*2
    cellids = rec['resp'].chans
    r_test = models[0].meta['r_test']
    r_floor = models[0].meta['r_floor']

    outpath = model.meta['modelpath']
    dfile = os.path.join(outpath, 'dstrf.npz')
    np.savez(dfile, pc_mags=pc_mags, mdstrfs=mdstrfs, fbmod=fbmod,
             cellids=cellids, r_test=r_test, r_floor=r_floor, modelname=modelname)

    res = np.load(dfile)
    print(pc_mags.std(), res['pc_mags'].std(), (pc_mags-res['pc_mags']).std())


