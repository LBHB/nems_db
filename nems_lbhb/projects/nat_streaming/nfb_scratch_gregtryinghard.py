from os.path import basename, join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter, zoom
import importlib
import datetime
import os
import nems0.db as nd

from nems0 import db
import nems0.epoch as ep
from nems0.utils import smooth
from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems_lbhb.projects.olp.OLP_helpers as ohel
import nems_lbhb.projects.olp.OLP_fit as ofit
import nems_lbhb.SPO_helpers as sp
import copy
import nems0.preprocessing as preproc
import joblib as jl
from nems_lbhb import baphy_io



# code to support dumping figures
#dt = datetime.date.today().strftime("%Y-%m-%d")
#figpath = f'/auto/users/svd/docs/current/grant/r21_free_moving/eps/{dt}/'
#os.makedirs(figpath, exist_ok=True)

# tested sites
# siteid="LMD004a"
batch=349
rasterfs = 100
PreStimSilence = 1.0
cache_path = 'cache_behavior'

cell_df = nd.get_batch_cells(batch)
cell_list = cell_df['cellid'].tolist()
sites = list(set([dd[:7] for dd in cell_list]))
sites.sort()
sites.remove('LMD002a')
for site in sites:
    cells = [dd for dd in cell_list if dd[:7]==site]
    if 0:
        runclass = "'NFB'"

        dfiles = db.pd_query(
            f"SELECT * FROM gDataRaw WHERE bad=0 and training=0 and runclass in ({runclass}) and cellid='{site}'")
        parmfiles = [os.path.join(r['resppath'], r['parmfile']) for i, r in dfiles.iterrows()]

        manager = BAPHYExperiment(parmfile=parmfiles[:1])
    else:
        manager = BAPHYExperiment(batch=batch, cellid=site)

    rec = manager.get_recording(**{'rasterfs': rasterfs, 'resp': True, 'stim': False},
                                recache=False)

    if (site == 'LMD002a') or (site == 'LMD007a'):
        stim_epochs = ep.epoch_names_matching(rec['resp'].epochs, '^STIM_')
        bg_uniques = list(set([ee.split('_')[1] for ee in stim_epochs if ee.split('_')[1] != 'null']))
        fg_uniques = list(set([ee.split('_')[2] for ee in stim_epochs if ee.split('_')[2] != 'null']))

        shortest_bg = np.min([int(aa.split('-')[2]) for aa in bg_uniques])
        shortest_fg = np.min([int(aa.split('-')[2]) for aa in fg_uniques])

        for row, epo in enumerate(rec['resp'].epochs.name):
            if epo[:5] == 'STIM_':
                if epo.split('_')[1] != 'null':
                    bg_len = int(epo.split('_')[1].split('-')[2])
                else:
                    bg_len = 'null'
                if epo.split('_')[2] != 'null':
                    fg_len = int(epo.split('_')[2].split('-')[2])
                else:
                    fg_len = 'null'

                epo_parts = epo.split('_')
                if (bg_len != shortest_bg) and (bg_len != 'null'):
                    bg_parts = epo.split('_')[1].split('-')
                    bg_parts[2] = str(shortest_bg)
                    new_bg = '-'.join(bg_parts)
                    epo_parts[1] = new_bg
                if fg_len != shortest_fg and (fg_len != 'null'):
                    fg_parts = epo.split('_')[2].split('-')
                    fg_parts[2] = str(shortest_fg)
                    new_fg = '-'.join(fg_parts)
                    epo_parts[2] = new_fg
                new_str = '_'.join(epo_parts)
                rec['resp'].epochs.at[row, 'name'] = new_str

    rec=rec.create_mask('ACTIVE_EXPERIMENT', mask_name='mask_active')
    rec=rec.create_mask('PASSIVE_EXPERIMENT', mask_name='mask_passive')

    rec['resp'] = rec['resp'].rasterize()
    resp = rec['resp']
    epochs=resp.epochs

    expt_params = manager.get_baphy_exptparams()
    ref_handle = expt_params[0]['TrialObject'][1]['ReferenceHandle'][1]
    ref_handle = {key: vall.strip() if type(vall) is str else vall for key, vall in ref_handle.items()}

    prebins, postbins = int(ref_handle['PreStimSilence']*rasterfs), int(ref_handle['PostStimSilence']*rasterfs)

    stim_epochs=ep.epoch_names_matching(epochs, '^STIM_')
    dstim = pd.DataFrame({'epoch': stim_epochs, 'fg': '', 'bg': '', 'fgc': 0, 'bgc': 0, 'snr': 0})
    for i,r in dstim.iterrows():
        e=dstim.loc[i,'epoch']
        s = e.split('_')[1:]
        b = s[0].split("-")
        f = s[1].split("-")
        if len(f)<5:
            s_snr='0'
        else:
            s_snr=f[4][:-2]
        if s_snr[0]=='n':
            snr = -float(s_snr[1:])
        else:
            snr = float(s_snr)

        if (snr>=-50) & (f[0].upper()!='NULL'):
            dstim.loc[i,'fg']=f[0]
            dstim.loc[i,'fgc']=int(f[3])
        else:
            dstim.loc[i,'fg']='NULL'
            dstim.loc[i,'fgc']=1
        if (snr<50) & (b[0].upper()!='NULL'):
            dstim.loc[i,'bg']=b[0]
            dstim.loc[i,'bgc']=int(b[3])
            dstim.loc[i, 'snr'] = snr
        else:
            dstim.loc[i,'bg']='NULL'
            dstim.loc[i,'bgc']=1
            if snr>50:
                dstim.loc[i,'snr']=snr-100
            else:
                dstim.loc[i,'snr']=snr

    fg_unique, bg_unique = dstim['fg'].unique().tolist(), dstim['bg'].unique().tolist()
    snr_unique = dstim['snr'].unique().tolist()
    fg_unique.remove('NULL')
    bg_unique.remove('NULL')
    if -100 in snr_unique:
        snr_unique.remove(-100)
    fc, bc = [1,2], [1,2]
    conds = np.array(np.meshgrid(snr_unique, fc, bc)).T.reshape(-1,3)

    triads = []
    cc=0
    for i,f in enumerate(fg_unique):
        for j,b in enumerate(bg_unique):
            for k,c in enumerate(conds):
                print(f"{f} {b} {c}")
                snr,fc,bc = c
                fgbg = dstim.loc[(dstim.snr==snr) & (dstim.fgc==fc) & (dstim.bgc==bc) &
                               (dstim.fg==f) & (dstim.bg==b),'epoch'].values[0]
                fg = dstim.loc[(dstim.snr==snr) & (dstim.fgc==fc) &
                               (dstim.fg==f) & (dstim.bg=='NULL'),'epoch'].values[0]
                bg = dstim.loc[(dstim.bgc==bc) &
                               (dstim.fg=='NULL') & (dstim.bg==b),'epoch'].values[0]
                triads.append(pd.DataFrame({'f': f,'b':b, 'snr': snr, 'fc': fc, 'bc': bc,
                                      'fg': fg, 'bg': bg, 'fgbg': fgbg  }, index=[cc]))
                cc+=1
    d = pd.concat(triads)

    weight_df_list = []
    for ee, cid in enumerate(resp.chans):
        print(f'Running cellid {cid}: {ee+1}/{len(resp.chans)}')
        val = rec.copy()
        val['resp'] = val['resp'].extract_channels([cid])
        # val['resp'] = val['resp'].rasterize()
        # Gather some stats on the recording, SR really is the only important one I think.
        norm_spont, SR, STD = ohel.remove_spont_rate_std(val['resp'])

        try:
            area_info = baphy_io.get_depth_info(cellid=cid)
            layer, area = area_info['layer'].values[0], area_info['area'].values[0]
            print('I found area info!')
        except:
            area_info = db.pd_query(f"SELECT DISTINCT area FROM sCellFile where cellid like '{manager.siteid}%%'")
            layer, area = 'NA', area_info['area'].iloc[0]

        bads = []
        for eeecount,eee in enumerate(d.fgbg.to_list()):
            did_it_happen_active = val['resp'].get_epoch_indices(eee, mask=val['mask_active']).shape[0]
            did_it_happen_passive = val['resp'].get_epoch_indices(eee, mask=val['mask_passive']).shape[0]
            print(f'For {eee}, active={did_it_happen_active}--passive={did_it_happen_passive}.')
            if (did_it_happen_passive == 0) or (did_it_happen_active == 0):
                # if eeecount == 0:
                # print(f'For {eee}, active={did_it_happen_active}--passive={did_it_happen_passive}.')
                bads.append(eee)
            # print(f'For {eee}, active={did_it_happen_active}--passive={did_it_happen_passive}.')
        print(f'I found {len(bads)}/{len(d.fgbg.to_list())} bad epochs that did not happen in active and passive.')
        d = d.loc[~d['fgbg'].isin(bads)]

        A, B, AB = d.bg.to_list(), d.fg.to_list(), d.fgbg.to_list()
        subsets = len(val.signals) - 1
        weights = np.zeros((2, len(AB), subsets))
        Efit = np.zeros((5, len(AB), subsets))
        nMSE = np.zeros((len(AB), subsets))
        nf = np.zeros((len(AB), subsets))
        r = np.zeros((len(AB), subsets))

        # Goes through all two stim epochs and gets the matching null ones to run through the actual
        # weight fitting function, which will then positionally populate assorted matrices
        mask_list = list(val.signals.keys())[1:]
        mask_labels = ['_' + ww.split('_')[1] for ww in mask_list]
        signames = ['resp', 'resp']
        stim_dff = pd.DataFrame()
        for i in range(len(AB)):
            names = [[A[i]], [B[i]], [AB[i]]]
            stimmy_df = []
            for ss, (mask, sig) in enumerate(zip(mask_list, signames)):
                weights[:, i, ss], Efit[:, i, ss], nMSE[i, ss], nf[i, ss], _, r[i, ss], cell_deef = \
                    calc_psth_weights_of_model_responses_list_behavior(val, names,
                                                              signame=sig, maskname=mask)
                stimmy_df.append(cell_deef)
            stim_df = pd.merge(stimmy_df[0], stimmy_df[1], on='fgbg')
            stim_dff = pd.concat([stim_dff, stim_df], axis=0)

        stim_dff['SR'], stim_dff['STD'], stim_dff['batch'] = SR, STD, batch
        stim_dff.insert(loc=0, column='layer', value=layer)
        stim_dff.insert(loc=0, column='area', value=area)

        # Makes a list of lists that iterates through the arrays you created, then flattens them in the next line
        big_list = [[weights[0, :, ee], weights[1, :, ee], nMSE[:, ee], nf[:, ee], r[:, ee]] for ee in
                    range(len(mask_list))]
        flat_list = [item for sublist in big_list for item in sublist]
        small_list = [d['fgbg'].values]
        # Combines the lists into a format that is conducive to the dataframe format I want to make
        bigger_list = small_list + flat_list
        weight_df = pd.DataFrame(bigger_list)
        weight_df = weight_df.T

        # Automatically generates a list of column names based on the names of the subsets provided above
        column_labels1 = ['fgbg']
        column_labels2 = [[f"weightsA{cl}", f"weightsB{cl}", f"nMSE{cl}", f"nf{cl}", f"r{cl}"] for cl in mask_labels]
        column_labels_flat = [item for sublist in column_labels2 for item in sublist]
        # column_labels = column_labels1 + column_labels_flat
        # Renames the columns according to that list - should work for any scenario as long as you specific names above
        weight_df.columns = column_labels1 + column_labels_flat

        # Not sure why I need this, I guess some may not be floats, so just doing it
        col_dict = {ii: float for ii in column_labels_flat}
        weight_df = weight_df.astype(col_dict)

        # Insert epoch column so you can merge weight_df and cell_df on this.
        # weight_df.insert(loc=0, column='fgbg', value=d.fgbg)

        # Add relative gain metric for all the fits
        for ss in mask_labels:
            weight_df[f'FG_rel_gain{ss}'] = (weight_df[f'weightsB{ss}'] - weight_df[f'weightsA{ss}']) / \
                                            (np.abs(weight_df[f'weightsB{ss}']) + np.abs(
                                                weight_df[f'weightsA{ss}']))
            weight_df[f'BG_rel_gain{ss}'] = (weight_df[f'weightsA{ss}'] - weight_df[f'weightsB{ss}']) / \
                                            (np.abs(weight_df[f'weightsA{ss}']) + np.abs(
                                                weight_df[f'weightsB{ss}']))
        # Merge the informational dataframe (cell_df) with the one with the data (weight_df)
        merge = pd.merge(stim_dff, weight_df, on="fgbg")
        merge = pd.merge(d, merge, on='fgbg')
        # merge = merge.drop(['BG', 'FG'], axis='columns')

        merge.insert(loc=0, column='cellid', value=cid)
        # merge['fit_segment'] = f"{int(snip[0] * 1000)}-{int((snip[0] + snip[1]) * 1000)}"
        merge['animal'] = cid[:3]

        weight_df_list.append(merge)

    final_weight_df = pd.concat(weight_df_list)

    OLP_partialweights_db_path = f'/auto/users/hamersky/{cache_path}/{site}'  # weight + corr

    # OLP_partialweights_db_path = f'/auto/users/hamersky/cache/{cellid}_{real_olps.iloc[cc]["olp_type"]}'  # weight + corr
    os.makedirs(os.path.dirname(OLP_partialweights_db_path), exist_ok=True)

    jl.dump(final_weight_df, OLP_partialweights_db_path)

    # return final_weight_df, OLP_partialweights_db_path




def calc_psth_weights_of_model_responses_list_behavior(val, names, signame='resp',
                                              get_nrmse_fn=False, maskname=None, start_bins=80, keep_bins=120):
    '''fdfa'''
    good_bins = keep_bins+start_bins
    sig1 = val[signame].extract_epoch(names[0][0], mask=val[maskname])[:, 0, start_bins:good_bins].mean(axis=0) #BG
    sig2 = val[signame].extract_epoch(names[1][0], mask=val[maskname])[:, 0, start_bins:good_bins].mean(axis=0) #FG
    sigO = val[signame].extract_epoch(names[2][0], mask=val[maskname])[:, 0, start_bins:good_bins].mean(axis=0) #BGFG

    # fsigs=np.vstack((sig1,sig2,sig_SR)).T
    fsigs = np.vstack((sig1, sig2)).T
    ff = np.all(np.isfinite(fsigs), axis=1) & np.isfinite(sigO)
    close_to_zero = np.array([np.allclose(fsigs[ff, i], 0, atol=1e-17) for i in (0, 1)])
    if all(close_to_zero):
        # Both input signals have all their values close to 0. Set weights to 0.
        weights = np.zeros(2)
        rank = 1
    elif any(close_to_zero):
        weights_, residual_sum, rank, singular_values = np.linalg.lstsq(np.expand_dims(fsigs[ff, ~close_to_zero], 1),
                                                                        sigO[ff], rcond=None)
        weights = np.zeros(2)
        weights[~close_to_zero] = weights_
    else:
        weights, residual_sum, rank, singular_values = np.linalg.lstsq(fsigs[ff, :], sigO[ff], rcond=None)
        # residuals = ((sigO[ff]-(fsigs[ff,:]*weights).sum(axis=1))**2).sum()

    # calc CC between weight model and actual response
    pred = np.dot(weights, fsigs[ff, :].T)
    cc = np.corrcoef(pred, sigO[ff])
    r_weight_model = cc[0, 1]

    # norm_factor = np.std(sigO[ff])
    norm_factor = np.mean(sigO[ff] ** 2)

    if rank == 1:
        min_nMSE = 1
        min_nRMSE = 1
    else:
        # min_nrmse = np.sqrt(residual_sum[0]/ff.sum())/norm_factor
        pred = np.dot(weights, fsigs[ff, :].T)
        min_nRMSE = np.sqrt(((sigO[ff] - pred) ** 2).mean()) / np.sqrt(
            norm_factor)  # minimim normalized root mean squared error
        min_nMSE = ((sigO[ff] - pred) ** 2).mean() / norm_factor  # minimim normalized mean squared error

    # create NMSE caclulator for later
    if get_nrmse_fn:
        def get_nrmse(weights=weights):
            pred = np.dot(weights, fsigs[ff, :].T)
            nrmse = np.sqrt(((pred - sigO[ff]) ** 2).mean(axis=-1)) / norm_factor
            return nrmse
    else:
        get_nrmse = np.nan

    weights[close_to_zero] = np.nan

    ml = '_' + maskname.split('_')[1]
    rA = val[signame].extract_epoch(names[0][0], mask=val[maskname])[:, 0, start_bins:good_bins]  # BG
    rB = val[signame].extract_epoch(names[1][0], mask=val[maskname])[:, 0, start_bins:good_bins]  # FG
    rAB = val[signame].extract_epoch(names[2][0], mask=val[maskname])[:, 0, start_bins:good_bins]  # BGFG
    snr_list_AB_A_B = ofit.compute_epoch_snr([rAB, rA, rB])
    cell_deef = {'fgbg': [names[2][0]],
                 f'combo_snr{ml}': [snr_list_AB_A_B[0]],
                 f'bg_snr{ml}': [snr_list_AB_A_B[1]],
                 f'fg_snr{ml}': [snr_list_AB_A_B[2]]}
    cell_def = pd.DataFrame.from_dict(cell_deef)

    return weights, np.nan, min_nMSE, norm_factor, get_nrmse, r_weight_model, cell_def#, get_error



batch = 349
cellid = 'LMD014a-A-391-1'
epoch = 'STIM_cat185rec1hairdryerexcerpt1-0-4-1_ferretb4004R-0-4-1-0dB'
rasterfs = 100

cellid = 'LMD004a-A-279-1'
epoch = 'STIM_cat129rec1jackhammerexcerpt1-0-4-1_ferretb1001R-0-4-1-0dB'

cellid = 'LMD014a-A-256-1'
epoch = 'STIM_cat23rec1beesbuzzingexcerpt1-0-4-1_ferretb4004R-0-4-1-0dB'
'

rec, d = load_behavior_recording(cellid[:7])
dff, val = get_cell_fit_info(rec, d, cellid, epoch)
plot_active_passive_psth(dff, val, smooth=True)

def plot_active_passive_psth(dff, val, smooth=True, keep_bins=200, rasterfs=100, prestimsil=1.0):
    mask_list = list(val.signals.keys())[1:]
    mask_labels = ['_' + dd.split('_')[1] for dd in mask_list]

    fig, ax = plt.subplots(2, 1, figsize=(12,8), sharex=True, sharey=True)

    for ss, mask in enumerate(mask_list):
        AA = val['resp'].extract_epoch(dff.bg.values[0], mask=val[mask])[:, 0, :keep_bins]  # BG
        BB = val['resp'].extract_epoch(dff.fg.values[0], mask=val[mask])[:, 0, :keep_bins]  # FG
        AB = val['resp'].extract_epoch(dff.fgbg.values[0], mask=val[mask])[:, 0, :keep_bins]  # BGFG

        time = (np.arange(0, AB.shape[1]) / 100)
        time = time - prestimsil

        if smooth:
            ax[ss].plot(time, sf.gaussian_filter1d(AA.mean(axis=0), sigma), color='deepskyblue', label='BG')
            ax[ss].plot(time, sf.gaussian_filter1d(BB.mean(axis=0), sigma), color='yellowgreen', label='FG')
            ax[ss].plot(time, sf.gaussian_filter1d(AB.mean(axis=0), sigma), color='dimgrey', label='BG+FG')
        else:
            ax[ss].plot(time, AA.mean(axis=0), label='BG', color='deepskyblue')
            ax[ss].plot(time, BB.mean(axis=0), label='FG', color='yellowgreen')
            ax[ss].plot(time, AB.mean(axis=0), label='BG+FG', color='dimgrey')

        ax[ss].legend()
        bgw = np.around(dff[f'weightsA{mask_labels[ss]}'].values[0], 3)
        fgw = np.around(dff[f'weightsB{mask_labels[ss]}'].values[0], 3)
        rr = np.around(dff[f'r{mask_labels[ss]}'].values[0], 3)
        bgsnr = np.around(dff[f'bg_snr{mask_labels[ss]}'].values[0], 3)
        fgsnr = np.around(dff[f'fg_snr{mask_labels[ss]}'].values[0], 3)

        ax[ss].set_title(f"{mask_labels[ss]}, wbg={bgw}, wfg={fgw}, r={rr}\nbg_snr={bgsnr}, fg_snr={fgsnr}", fontsize=10)
        ax[ss].set_xticks(np.arange(time[0], time[-1], prestimsil))
        ymin, ymax = ax[ss].get_ylim()
        ax[ss].vlines([0], ymin, ymax, colors='black', linestyles=':')
        ax[ss].set_ylim(ymin, ymax)
        ax[ss].set_ylabel('spk/s', fontweight='bold', fontsize=10)

    ax[-1].set_xlabel('Time (s)', fontweight='bold', fontsize=10)
    fig.suptitle(f'cellid: {dff.cellid.values[0]}\nepoch: {dff.fgbg.values[0]}', fontweight='bold', fontsize=12)



def get_cell_fit_info(rec, d, cellid, epoch, keep_bins=200):
    val = rec.copy()
    val['resp'] = val['resp'].extract_channels([cellid])

    norm_spont, SR, STD = ohel.remove_spont_rate_std(val['resp'])

    area_info = baphy_io.get_depth_info(cellid=cellid)
    layer, area = area_info['layer'].values[0], area_info['area'].values[0]

    did_it_happen_active = val['resp'].get_epoch_indices(epoch, mask=val['mask_active']).shape[0]
    did_it_happen_passive = val['resp'].get_epoch_indices(epoch, mask=val['mask_passive']).shape[0]
    print(f'Epoch {epoch} was played\n{did_it_happen_active}x actively and {did_it_happen_passive}x passively.')

    row = d.loc[d.fgbg==epoch]
    names = [[row['bg'].values[0]], [row['fg'].values[0]], [row['fgbg'].values[0]]]
    mask_list = list(val.signals.keys())[1:]
    mask_labels = ['_' + dd.split('_')[1] for dd in mask_list]
    weights = np.zeros((2, len(val.signals) - 1))
    r = np.zeros((len(val.signals) - 1))

    stimmy_df = []
    for ss, mask in enumerate(mask_list):
        weights[:, ss], _, _, _, _, r[ss], cell_deef = \
            calc_psth_weights_of_model_responses_list_behavior(val, names,
                                                               signame='resp', maskname=mask, keep_bins=keep_bins)
        stimmy_df.append(cell_deef)

    stim_df = pd.merge(stimmy_df[0], stimmy_df[1], on='fgbg')
    stim_df['SR'], stim_df['area'], stim_df['layer'] = SR, area, layer



    # Makes a list of lists that iterates through the arrays you created, then flattens them in the next line
    big_list = [[weights[0, ee], weights[1, ee], r[ee]] for ee in
                range(len(mask_list))]
    flat_list = [item for sublist in big_list for item in sublist]
    weight_df = pd.DataFrame(flat_list)
    weight_df = weight_df.T
    column_labels2 = [[f"weightsA{cl}", f"weightsB{cl}", f"r{cl}"] for cl in mask_labels]
    weight_df.columns = [item for sublist in column_labels2 for item in sublist]

    # Not sure why I need this, I guess some may not be floats, so just doing it
    col_dict = {ii: float for ii in column_labels_flat}
    weight_df = weight_df.astype(col_dict)

    # Add relative gain metric for all the fits
    for ss in mask_labels:
        weight_df[f'FG_rel_gain{ss}'] = (weight_df[f'weightsB{ss}'] - weight_df[f'weightsA{ss}']) / \
                                        (np.abs(weight_df[f'weightsB{ss}']) + np.abs(
                                            weight_df[f'weightsA{ss}']))
        weight_df[f'BG_rel_gain{ss}'] = (weight_df[f'weightsA{ss}'] - weight_df[f'weightsB{ss}']) / \
                                        (np.abs(weight_df[f'weightsA{ss}']) + np.abs(
                                            weight_df[f'weightsB{ss}']))
    # Merge the informational dataframe (cell_df) with the one with the data (weight_df)
    merge = pd.concat([stim_df, weight_df], axis=1)
    full_merge = pd.merge(d, merge, on='fgbg')
    full_merge.insert(loc=0, column='cellid', value=cellid)

    return full_merge, val


def load_behavior_recording(siteid, batch=349, rasterfs=100):
    manager = BAPHYExperiment(batch=batch, cellid=siteid)
    rec = manager.get_recording(**{'rasterfs': rasterfs, 'resp': True, 'stim': False}, recache=False)
    rec=rec.create_mask('ACTIVE_EXPERIMENT', mask_name='mask_active')
    rec=rec.create_mask('PASSIVE_EXPERIMENT', mask_name='mask_passive')

    rec['resp'] = rec['resp'].rasterize()
    resp = rec['resp']
    epochs=resp.epochs

    expt_params = manager.get_baphy_exptparams()
    ref_handle = expt_params[0]['TrialObject'][1]['ReferenceHandle'][1]
    ref_handle = {key: vall.strip() if type(vall) is str else vall for key, vall in ref_handle.items()}

    stim_epochs=ep.epoch_names_matching(epochs, '^STIM_')
    dstim = pd.DataFrame({'epoch': stim_epochs, 'fg': '', 'bg': '', 'fgc': 0, 'bgc': 0, 'snr': 0})
    for i,r in dstim.iterrows():
        e=dstim.loc[i,'epoch']
        s = e.split('_')[1:]
        b = s[0].split("-")
        f = s[1].split("-")
        if len(f)<5:
            s_snr='0'
        else:
            s_snr=f[4][:-2]
        if s_snr[0]=='n':
            snr = -float(s_snr[1:])
        else:
            snr = float(s_snr)

        if (snr>=-50) & (f[0].upper()!='NULL'):
            dstim.loc[i,'fg']=f[0]
            dstim.loc[i,'fgc']=int(f[3])
        else:
            dstim.loc[i,'fg']='NULL'
            dstim.loc[i,'fgc']=1
        if (snr<50) & (b[0].upper()!='NULL'):
            dstim.loc[i,'bg']=b[0]
            dstim.loc[i,'bgc']=int(b[3])
            dstim.loc[i, 'snr'] = snr
        else:
            dstim.loc[i,'bg']='NULL'
            dstim.loc[i,'bgc']=1
            if snr>50:
                dstim.loc[i,'snr']=snr-100
            else:
                dstim.loc[i,'snr']=snr

    fg_unique, bg_unique = dstim['fg'].unique().tolist(), dstim['bg'].unique().tolist()
    snr_unique = dstim['snr'].unique().tolist()
    fg_unique.remove('NULL')
    bg_unique.remove('NULL')
    if -100 in snr_unique:
        snr_unique.remove(-100)
    fc, bc = [1,2], [1,2]
    conds = np.array(np.meshgrid(snr_unique, fc, bc)).T.reshape(-1,3)

    triads = []
    cc=0
    for i,f in enumerate(fg_unique):
        for j,b in enumerate(bg_unique):
            for k,c in enumerate(conds):
                print(f"{f} {b} {c}")
                snr,fc,bc = c
                fgbg = dstim.loc[(dstim.snr==snr) & (dstim.fgc==fc) & (dstim.bgc==bc) &
                               (dstim.fg==f) & (dstim.bg==b),'epoch'].values[0]
                fg = dstim.loc[(dstim.snr==snr) & (dstim.fgc==fc) &
                               (dstim.fg==f) & (dstim.bg=='NULL'),'epoch'].values[0]
                bg = dstim.loc[(dstim.bgc==bc) &
                               (dstim.fg=='NULL') & (dstim.bg==b),'epoch'].values[0]
                triads.append(pd.DataFrame({'f': f,'b':b, 'snr': snr, 'fc': fc, 'bc': bc,
                                      'fg': fg, 'bg': bg, 'fgbg': fgbg  }, index=[cc]))
                cc+=1
    d = pd.concat(triads)

    return rec, d



triadcount=len(d)


# plt.close('all')
smwin=3
T=int((PreStimSilence+2)*rasterfs)
lw=0.75
pstr={1: 'C', 2: 'I'}
colors = ['deepskyblue', 'yellowgreen', 'dimgray']

# interesting cids: LMD004a00_a_NFB - 59, 52, 68, 25
#cidlist=[2,3,6,7,8,9,13,15,18,19,20,21,22,25,28,31,32,35,41,42,48,49,50,52,55,59,60,61,63,65 , 68,73,74,82,89]
cidlist = np.arange(len(resp.chans))
# plt.close('all')
cellcount = len(cidlist)
rows=int(np.ceil(np.sqrt(cellcount)))
cols= int(np.ceil(cellcount/rows))
f,ax=plt.subplots(rows,cols, sharex=True, sharey=True)
ax=ax.flatten()
r = d.loc[0]
print(r.fg)
print(r.bg)
print(r.fgbg)
for i,a in enumerate(ax[:cellcount]):
    cid=cidlist[i]
    rfg = resp.extract_epoch(r['fg'])[:,cid,:T].mean(axis=0)
    rbg = resp.extract_epoch(r['bg'])[:,cid,:T].mean(axis=0)
    rfgbg = resp.extract_epoch(r['fgbg'])[:,cid,:T].mean(axis=0)
    tt = np.arange(len(rfg))/rasterfs - PreStimSilence
    mm = np.max(np.concatenate([rfg, rbg, rfgbg]))
    a.plot(tt,smooth(rfg/mm,smwin),lw=lw, label='fg', color=colors[1])
    a.plot(tt,smooth(rbg/mm,smwin),lw=lw,label='bg', color=colors[0])
    a.plot(tt,smooth(rfgbg/mm,smwin),lw=lw,label='fg+bg', color=colors[2])
    a.set_title(f"CH {cid}", fontsize=7)
a.legend(frameon=False)
plt.tight_layout()

cid=144

f,ax=plt.subplots(2, 2, sharex=True, sharey=True, figsize=(15,8))
ax=ax.flatten()
smwin=6
lw=1.5
eps = [0, 8]

for i, epochy in enumerate(eps):
    r = d.iloc[epochy]
    for j, m in enumerate(['mask_passive', 'mask_active']):
        try:
            rfg = resp.extract_epoch(r['fg'], mask=rec[m])[:,cid,:T].mean(axis=0)
            rbg = resp.extract_epoch(r['bg'], mask=rec[m])[:,cid,:T].mean(axis=0)
            rfgbg = resp.extract_epoch(r['fgbg'], mask=rec[m])[:,cid,:T].mean(axis=0)
            tt = np.arange(len(rfg))/rasterfs - PreStimSilence
            prebins = (tt<0).sum()
            if (i==0) & (j==0):
                spont = np.mean(rfg[:prebins]+rbg[:prebins]+rfgbg[:prebins])/3
            ax[i*2+j].plot(tt,smooth(rfg,smwin),lw=lw, label='FG', color=colors[1])
            ax[i*2+j].plot(tt,smooth(rbg,smwin),lw=lw,label='BG', color=colors[0])
            ax[i*2+j].plot(tt,smooth(rfgbg,smwin),lw=lw,label='BG+FG', color=colors[2])
            ax[i*2+j].axhline(y=spont,ls='--',lw=lw,color='gray')
            ax[i*2+j].axvline(x=0,ls=':',lw=lw,color='gray')
            ax[i*2+j].set_xlim(-0.35, 2)
        except:
            ax[i*2+j].set_axis_off()
    ax[i * 2].set_title(f"{r.f[6:]} {r.b[10:16]} snr={r.snr},f={pstr[r.fc]},b={pstr[r.bc]}")
ax[i * 2].legend()
f.suptitle(f"{cid} {resp.chans[cid]}")
plt.tight_layout()


#cols=int(np.ceil(np.sqrt(triadcount)))
cols=4
rows= int(np.ceil(triadcount/cols))
f,ax=plt.subplots(rows,cols*2, sharex=True, sharey=True)
ax=ax.flatten()

for i,r in d.iterrows():
    for j,m in enumerate(['mask_passive','mask_active']):
        try:
            rfg = resp.extract_epoch(r['fg'], mask=rec[m])[:,cid,:T].mean(axis=0)
            rbg = resp.extract_epoch(r['bg'], mask=rec[m])[:,cid,:T].mean(axis=0)
            rfgbg = resp.extract_epoch(r['fgbg'], mask=rec[m])[:,cid,:T].mean(axis=0)
            tt = np.arange(len(rfg))/rasterfs - PreStimSilence
            prebins = (tt<0).sum()
            if (i==0) & (j==0):
                spont = np.mean(rfg[:prebins]+rbg[:prebins]+rfgbg[:prebins])/3
            ax[i*2+j].plot(tt,smooth(rfg,smwin),lw=lw, label='fg', color=colors[1])
            ax[i*2+j].plot(tt,smooth(rbg,smwin),lw=lw,label='bg', color=colors[0])
            ax[i*2+j].plot(tt,smooth(rfgbg,smwin),lw=lw,label='fg+bg', color=colors[2])
            ax[i*2+j].axhline(y=spont,ls='--',lw=lw,color='gray')
        except:
            ax[i*2+j].set_axis_off()
    ax[i*2].set_title(f"{r.f[6:]} {r.b[10:16]} snr={r.snr},f={pstr[r.fc]},b={pstr[r.bc]}")
ax[i*2].legend()
f.suptitle(f"{cid} {resp.chans[cid]}")
plt.tight_layout()

cchisnr = (d.fc==1) & (d.bc==1) & (d.snr==0)
dcch = d.loc[cchisnr].reset_index()
f,ax=plt.subplots(2,len(dcch))
lateonset=int((PreStimSilence+0.2)*rasterfs)

for i,r in dcch.iterrows():
    efg=r['fg']
    ebg=r['bg']
    rfgp=resp.extract_epoch(efg, mask=rec['mask_passive']).mean(axis=0)
    rbgp=resp.extract_epoch(ebg, mask=rec['mask_passive']).mean(axis=0)
    rfga=resp.extract_epoch(efg, mask=rec['mask_active'], allow_empty=True).mean(axis=0)
    rbga=resp.extract_epoch(ebg, mask=rec['mask_active'], allow_empty=True).mean(axis=0)
    ax[0,i].plot(rfgp.mean(axis=0))
    ax[0,i].plot(rfga.mean(axis=0))
    ax[0,i].set_title(r['f'])
    ax[0,i].axvline(x=lateonset, ls='--')
    ax[1,i].plot(rbgp.mean(axis=0))
    ax[1,i].plot(rbga.mean(axis=0))
    ax[1,i].set_title(r['b'])
    ax[1,i].axvline(x=lateonset, ls='--')

i=0
ef=dcch.loc[0]