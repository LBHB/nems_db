import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nems_lbhb.baphy_experiment import BAPHYExperiment
import nems0.epoch as ep

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

def get_rec_epochs(parmfile=None, fs=100, rec=None):
    '''Takes a parmfile and outputs a dataframe containing every epoch that was played in the recording,
    with columns giving you the concurrent epoch name, BG and FG alone, and then what synthetic
    and binaural condition the epoch'''

    if rec is None:
        manager = BAPHYExperiment(parmfile)
        options = {'rasterfs': fs, 'stim': False, 'resp': True}
        rec = manager.get_recording(**options)

    stim_epochs = ep.epoch_names_matching(rec['resp'].epochs, 'STIM_')
    twostims = [epo for epo in stim_epochs if 'null' not in epo]

    epoch_df = pd.DataFrame({'BG + FG': twostims})
    epoch_df['BG'], epoch_df['FG'], epoch_df['Synth Type'], epoch_df['Binaural Type'] = \
        zip(*epoch_df['BG + FG'].apply(get_stim_type))

    return epoch_df


def get_stim_type(ep_name):
    '''Labels epochs that have two stimuli based on what kind of synthetic sound it is and what
    binaural condition it is.
    Synth: N = Normal RMS, C = Cochlear, T = Temporal, S = Spectral, U = Spectrotemporal, M = spectrotemporal
    modulation, A = Non-RMS normalized unsynethic
    Binaural: 11 = BG+FG Contra, 12 = BG Contra - FG Ipsi, 21, BG Ipsi - FG Contra, 22 = BG+FG Ipsi'''
    synth_dict = {'N': 'Unsynthetic', 'C': 'Cochlear', 'T': 'Temporal', 'S': 'Spectral',
                  'U': 'Spectrotemporal', 'M': 'Spectemp Modulation', 'A': 'Non-RMS Unsynthetic'}
    binaural_dict = {'11': 'BG Contra, FG Contra', '12': 'BG Contra, FG Ipsi',
                     '21': 'BG Ipsi, FG Contra', '22': 'BG Ipsi, FG Contra'}

    if len(ep_name.split('_')) == 3 and ep_name[:5] == 'STIM_':
        seps = (ep_name.split('_')[1], ep_name.split('_')[2])
        bg_ep, fg_ep = f"STIM_{seps[0]}_null", f"STIM_null_{seps[1]}"

        #get synth type
        if len(seps[0].split('-')) >= 5:
            synth_kind = seps[0].split('-')[4]
        else:
            synth_kind = 'A'

        #get binaural type
        if len(seps[0].split('-')) >= 4:
            bino_kind = seps[0].split('-')[3] + seps[1].split('-')[3]
        else:
            bino_kind = '11'

        synth_type = synth_dict[synth_kind]
        bino_type = binaural_dict[bino_kind]

    else:
        synth_type, bino_type = None, None
        bg_ep, fg_ep = f"STIM_{ep_name.split('_')[1]}_null", f"STIM_null_{ep_name.split('_')[2]}"

    return bg_ep, fg_ep, synth_type, bino_type

def r_ceiling(ra,rb):
    """
    compute a noise-corrected correlation between two rasters (compute correlation between subsets of
    trials between ra and rb, then normalize by correlation between subsets of trials in ra). Note this is
    asymetric, normalizing by ra's noise ceiling!
    TODO: figure out if there's a better way to normalize
    :param ra: Trial X Time raster
    :param rb: another Trial X Time raster
    :return: rceil - noise-corrected correlation coefficient. Maybe >1 b/c noise, but doesn't seem to get
    there in practice
    """
    # do corrcoeff calculation ourselves in parallel across subsamples, faster than manny corrcoef calls.
    ra = ra.copy() - ra.mean(axis=1, keepdims=True)
    rb = rb.copy() - rb.mean(axis=1, keepdims=True)

    # average across sets of half the trials
    halfcount=int(ra.shape[0]/2)
    for h in range(1,halfcount):
        ra += np.roll(ra,h,axis=0)
        rb += np.roll(rb, h, axis=0)

    #then shift rb so a different set of half trials is aligned with ra
    rb = np.roll(rb, halfcount, axis=0).copy()

    #shift ra and rb similarly for within-channel noise ceiling calculation
    ra2 = np.roll(ra, halfcount, axis=0)
    rb2 = np.roll(rb, halfcount, axis=0)

    cab = (ra * rb).mean(axis=1)
    caabb = np.sqrt(np.abs(ra*ra).mean(axis=1)*np.abs(rb*rb).mean(axis=1))
    caabb[caabb == 0] = 1
    ccab = cab.mean()/caabb.mean()

    caa = (ra * ra2).mean(axis=1)
    caaaa = np.sqrt(np.abs(ra*ra).mean(axis=1)*np.abs(ra2*ra2).mean(axis=1))
    caaaa[caaaa == 0] = 1
    ccaa = caa.mean()/caaaa.mean()

    # currently not normalizing by noise in rb
    #cbb = (rb * rb2).mean(axis=1)
    #cbbbb = np.sqrt(np.abs(rb*rb).mean(axis=1)*np.abs(rb2*rb2).mean(axis=1))
    #cbbbb[cbbbb == 0] = 1
    #ccbb = cbb.mean()/cbbbb.mean()

    rceil = ccab / ccaa
    if np.isnan(rceil):
        return 0
    else:
        return rceil

if __name__ == '__main__':
    parmfile = '/auto/data/daq/Clathrus/CLT039/CLT039c11_p_OLP.m'
    parmfile = '/auto/data/daq/Clathrus/CLT040/CLT040c07_p_OLP.m'
    parmfile = '/auto/data/daq/Clathrus/CLT041/CLT041c11_p_OLP.m'
    basename = os.path.basename(parmfile)

    manager = BAPHYExperiment(parmfile)
    fs = 50
    options = {'rasterfs': fs, 'stim': True, 'stimfmt': 'lenv', 'resp': True, 'recache': False}
    rec = manager.get_recording(**options)

    epoch_df = get_rec_epochs(rec=rec)
    resp = rec['resp'].rasterize()
    #resp = resp._modified_copy(data=resp._data.mean(axis=0,keepdims=True), chans=["MUA"])
    stim = rec['stim'].rasterize()

    cols = ['BG', 'FG', 'BG + FG']
    offset = 1
    dlist = []
    for c, cellid in enumerate(resp.chans):
        print(cellid)
        ed = epoch_df.copy()
        ed['cellid']=cellid
        for i, r in ed.iterrows():
            epochs = list(r[cols])
            sbg = stim.extract_epoch(epochs[0])[:, 0, :(-offset)]
            sfg = stim.extract_epoch(epochs[1])[:, 0, :(-offset)]
            rbg = resp.extract_epoch(epochs[0])[:, c, offset:]
            rfg = resp.extract_epoch(epochs[1])[:, c, offset:]
            rbgfg = resp.extract_epoch(epochs[2])[:,c , offset:]

            if (rbg.sum() > 0) & (rfg.sum() > 0) & (rbgfg.sum() > 0):
                cbb = r_ceiling(rbg, sbg)
                cff = r_ceiling(rfg, sfg)

                crfb = r_ceiling(rfg, rbg)
                crbbfl = r_ceiling(rbg, rbg+rfg)
                crfbfl = r_ceiling(rfg, rbg+rfg)

                crbbf = r_ceiling(rbg, rbgfg)
                crfbf = r_ceiling(rfg, rbgfg)
                ed.loc[i,['cbb','cff', 'crfb', 'cc(BG,sum)', 'cc(FG,sum)', 'cc(BG,FGBG)', 'cc(FG,FGBG)']] = \
                    [cbb, cff, crfb, crbbfl, crfbfl, crbbf, crfbf]
            else:
                ed.loc[i,['cbb','cff', 'crfb', 'cc(FG,sum)', 'cc(BG,sum)', 'cc(FG,FGBG)', 'cc(BG,FGBG)']] = 0
        ed['Eb']=np.sum(ed['cbb'])>0
        ed['Ef']=np.sum(ed['cff'])>0
        ed['E']=np.sum(ed['cbb']+ed['cff'])>0
        dlist.append(ed)
    epoch_df_all = pd.concat(dlist).reset_index()

    epoch_df_all['cff+cbb'] = (epoch_df_all['cff']+epoch_df_all['cbb'])/2
    epoch_df_all['cff-cbb'] = epoch_df_all['cff']-epoch_df_all['cbb']
    epoch_df_all['(crfbf-crbbf)'] = (epoch_df_all['cc(FG,sum)']-epoch_df_all['cc(BG,sum)'])
    epoch_df_all['FG cc(both-sum)'] = (epoch_df_all['cc(FG,FGBG)']-epoch_df_all['cc(FG,sum)'])
    epoch_df_all['BG cc(both-sum)'] = (epoch_df_all['cc(BG,FGBG)']-epoch_df_all['cc(BG,sum)'])
    epoch_df_all['dd'] = (epoch_df_all['FG cc(both-sum)']-epoch_df_all['BG cc(both-sum)'])
    epoch_df_all['ddraw'] = (epoch_df_all['cc(FG,FGBG)']-epoch_df_all['cc(BG,FGBG)'])

    plt.close('all')
    types = {'N': 'Natural RMS', 'U': 'Spectrotemporal', 'S': 'Spectral', 'T': 'Temporal', 'C': 'Cochlear'}
    #types = {'N': 'Natural RMS', 'U': 'Spectrotemporal', 'S': 'Spectral', 'T': 'Temporal', 'C': 'Cochlear'}
    #types ={'A': 'Natural max', 'N': 'Natural RMS', 'C': 'Cochlear', 'T': 'Temporal',
    #         'S': 'Spectral', 'U': 'Spectrotemporal', 'M': 'SpectrotemporalMod',
    #         }

    f, axs = plt.subplots(4, len(types), figsize=(10, 6))

    for t,ax in zip(types.keys(),axs.T):
        # both E
        #nat_epoch_df = epoch_df_all.loc[(epoch_df_all['cff'] > 0.1) & (epoch_df_all['cbb'] > 0.1) & epoch_df_all['BG + FG'].str.endswith(t)].reset_index()

        # both I
        #nat_epoch_df = epoch_df_all.loc[(epoch_df_all['cff']<-0.1) & (epoch_df_all['cbb']<-0.1) & epoch_df_all['BG + FG'].str.endswith(t)].reset_index()

        # both respond
        nat_epoch_df = epoch_df_all.loc[(np.abs(epoch_df_all['cff'])>0.1) & (np.abs(epoch_df_all['cbb'])>0.1) & epoch_df_all['BG + FG'].str.endswith(t)].reset_index()

        ax[0].set_title(f'{basename} {types[t]}')

        nat_epoch_df[['cff','cbb']].plot(lw=0.5,ax=ax[0])
        ax[0].plot([0,len(nat_epoch_df)],[0,0],'--', color='gray', lw=0.5)
        ax[0].set_ylim([-1.0, 1.0])

        nat_epoch_df[['cc(FG,sum)','cc(BG,sum)']].plot(lw=0.5,ax=ax[1])
        ax[1].plot([0,len(nat_epoch_df)],[0,0],'--', color='gray', lw=0.5)
        ax[1].set_ylim([-0.2, 1.0])

        nat_epoch_df[['FG cc(both-sum)','BG cc(both-sum)']].plot(lw=0.5,ax=ax[2])
        #nat_epoch_df[['cc(FG,FGBG)','cc(BG,FGBG)']].plot(lw=0.5,ax=ax[2])
        ax[2].plot([0,len(nat_epoch_df)],[0,0],'--', color='gray', lw=0.5)
        ax[2].set_ylim([-1.0, 0.4])

        k = 'dd'
        #k = 'ddraw'
        nat_epoch_df[[k]].plot(lw=0.5,ax=ax[3])
        ax[3].plot([0,len(nat_epoch_df)],[0,0],'--', color='gray', lw=0.5)
        ax[3].set_ylim([-1.0, 1.0])
        m = nat_epoch_df[k].mean()
        e = nat_epoch_df[k].std()
        ax[3].plot([0,len(nat_epoch_df)],[m-e,m-e],'r--', lw=0.5)
        ax[3].plot([0,len(nat_epoch_df)],[m+e,m+e],'r--', lw=0.5)
        ax[3].set_title(f"m+e={m:.2f}+{e:.2f}")
    plt.tight_layout()

    c = 0
    cellid=resp.chans[c]
    epoch_idx = 10
    f, ax = plt.subplots(3, len(types), figsize=(12, 6), sharey=True)

    for i, acol in enumerate(ax.T):
        nat_epoch_df = epoch_df_all.loc[(epoch_df_all.cellid == cellid) &
                                        epoch_df_all['BG + FG'].str.endswith(list(types.keys())[i])].reset_index()
        epochs = list(nat_epoch_df.loc[epoch_idx, cols])
        for a, e, k in zip(acol, epochs, cols):
            s = stim.extract_epoch(e)[0,0,:]
            r = resp.extract_epoch(e).mean(axis=0)[c,:]
            a.plot(s)
            a.plot(r)
            a.set_ylabel(k)
        acol[0].set_title(epochs[2])
        acol[1].set_title(f"BG={nat_epoch_df.loc[epoch_idx,'cc(BG,FGBG)']:.2f},FG={nat_epoch_df.loc[epoch_idx,'cc(FG,FGBG)']:.2f}")
        acol[2].set_title(f"BG={nat_epoch_df.loc[epoch_idx,'BG cc(both-sum)']:.2f},FG={nat_epoch_df.loc[epoch_idx,'FG cc(both-sum)']:.2f}")

    plt.tight_layout()


    f, axs = plt.subplots(4, len(types))

    for t,ax in zip(types.keys(),axs.T):
        nat_epoch_df = epoch_df_all.loc[(epoch_df_all.cellid==cellid) & epoch_df_all['BG + FG'].str.endswith(t)].reset_index()

        nat_epoch_df[['cff', 'cbb']].plot(lw=0.5, ax=ax[0])
        ax[0].plot([0, len(nat_epoch_df)], [0, 0], '--', color='gray', lw=0.5)
        ax[0].set_ylim([-0.2, 1])

        nat_epoch_df[['cc(FG,sum)', 'cc(BG,sum)']].plot(lw=0.5, ax=ax[1])
        ax[1].plot([0, len(nat_epoch_df)], [0, 0], '--', color='gray', lw=0.5)
        ax[1].set_ylim([-0.2, 1])

        nat_epoch_df[['FG cc(both-sum)', 'BG cc(both-sum)']].plot(lw=0.5, ax=ax[2])
        ax[2].plot([0, len(nat_epoch_df)], [0, 0], '--', color='gray', lw=0.5)
        ax[2].set_ylim([-0.8, 0.5])

        nat_epoch_df[['dd']].plot(lw=0.5, ax=ax[3])
        ax[3].plot([0, len(nat_epoch_df)], [0, 0], '--', color='gray', lw=0.5)
        ax[3].set_ylim([-0.8, 0.8])
        m = nat_epoch_df['dd'].mean()
        e = nat_epoch_df['dd'].std()
        ax[3].plot([0, len(nat_epoch_df)], [m - e, m - e], 'r--', lw=0.5)
        ax[3].plot([0, len(nat_epoch_df)], [m + e, m + e], 'r--', lw=0.5)
        ax[0].set_title(f'{cellid} {types[t]}')
