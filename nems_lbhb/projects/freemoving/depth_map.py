import matplotlib.pyplot as plt
import numpy as np
import json
import os
import datetime

from laminar_tools.lfp import lfp
from nems_lbhb.baphy_io import load_continuous_openephys, get_spike_info, get_depth_info
from nems0 import db


def plot_depth_map(siteid, batch=343):
    probeid = 'A'

    d_raw = db.pd_query(f"SELECT * FROM gDataRaw where cellid like '{siteid}%%' AND runclass='BNB' AND not(bad) AND not(isnull(depthinfo))")

    # use first match
    rawid = d_raw.loc[0,'id']
    highlight_cellids = []

    df_siteinfo = get_spike_info(siteid=siteid) # , save_to_db=True)
    #df_siteinfo=df_siteinfo.loc[(df_siteinfo.area=='BS') | (df_siteinfo.area=='A1')]

    df_siteinfo=df_siteinfo.reset_index()
    df_siteinfo['channel']=df_siteinfo['index'].apply(lambda x: int(x.split("-")[-2]))

    d=db.pd_query(f'SELECT * FROM gDataRaw where id={rawid}')
    d_labeled_depth = json.loads(d.loc[0,'depthinfo'])
    probe = 'Probe'+probeid
    try:
        d_labeled_depth = d_labeled_depth[probe]
    except:
        print('using top level of d_labeled_depth')


    if siteid=='PRN015a':
        parmfile = ['/auto/data/daq/Prince/PRN015/PRN015b04_p_BNB.m']
        rawid=146865
        highlight_cellids=[ "PRN015a-315-1",
                            "PRN015a-282-1",
                            "PRN015a-225-1"]

    elif siteid == 'PRN020a':
        parmfile = ['/auto/data/daq/Prince/PRN020/PRN020b02_p_BNB.m']
        rawid=146960
        highlight_cellids = []

    elif siteid == 'PRN034a':
        parmfile = ['/auto/data/daq/Prince/PRN034/PRN034a16_p_BNB.m']
        rawid=147117
        highlight_cellids = [ ]

    else:
        parmfile = [d_raw.loc[0,'resppath'] + d_labeled_depth['parmfile']]

    # load heatmaps of CSD and PSD
    left_csd, power, freqs, windowtime, rasterfs, column_xy_sorted, column_xy, channel_xy, coh_mat, erp, probe = \
        lfp.parmfile_event_lfp(parmfile)

    csd_crop=left_csd[0][:,80:]  # remove excess pre-stim silence
    power_crop = power[0][:,:]
    chans = column_xy_sorted[0]   # a channel that shows up in each CSD/PSD row
    csd_crop[np.isnan(csd_crop)] = 0  # replace nans with zeros to make it look nice

    # figure out how to align user-specified depths with the CSD/PSD images
    landmarks = {k:v for (k,v) in d_labeled_depth['landmarkPosition'].items() if d_labeled_depth['landmarkBoolean'][k]==True}
    channel_info = d_labeled_depth['channel info']
    um_per_pix = int(column_xy[0][chans[2]][1]) - int(column_xy[0][chans[1]][1])
    c34_pix = landmarks.get('3/4',0)
    gui_depth = np.array([channel_info[c][2] for c in chans if c in channel_info.keys()])
    gui_depth_pix = np.arange(len(chans))

    if gui_depth.max()==1050:
        print("UCLA probe adjust depth pix")
        offsetpix = gui_depth[0]/um_per_pix
        df_siteinfo['depth_adjusted'] = df_siteinfo['depth0']/um_per_pix - offsetpix
    else:
        df_siteinfo['depth_adjusted'] = df_siteinfo['depth0']

    # position of one column on probe (tip = 0), corresponding to channels in chans
    depths0 = np.array([float(column_xy[0][c][1]) for c in chans])
    gui_depth_adjust = channel_info[chans[0]][2] - gui_depth_pix[0]

    ylims = [(c34_pix+0.5)*um_per_pix, (c34_pix-len(chans)-0.5)*um_per_pix]


    # plt.close('all')
    f, ax = plt.subplots(1, 2, sharey=True)

    # plot CSD
    mm = np.max(np.abs(csd_crop))
    ax[0].imshow(csd_crop, origin='lower', aspect='auto', vmin=-mm, vmax=mm,
                 extent=[-0.1, 0.5]+ylims)
    #extent=[-0.1, 0.5, depths[0], depths[-1] ])
    for k,v in landmarks.items():
        ax[0].axhline((c34_pix-v)*um_per_pix, color='darkgray', linestyle='--')

    # overlay units at each depth
    nar_thr=0.35
    marker_dict={'BS': '.', '3': '^', '13': '^', '4': 'o', '44': 'o', '5': 'v',
                 '56': 'v', 'WM': '.'}

    for i,r in df_siteinfo.iterrows():
        depth_rel34 = (c34_pix-r['depth_adjusted'])*um_per_pix  #  - gui_depth_adjust
        marker = marker_dict.get(r['layer'], '.')
        sz=4
        if r['index'] in highlight_cellids:
            ax[0].plot(0.3+np.random.randn(1)/25, depth_rel34, marker, color='w')
        elif r['area'] in ['A1', 'PEG', 'BS']:
            if r['sw']>nar_thr:
                ax[0].plot(0.3 + np.random.randn(1) / 25, depth_rel34, marker, markersize=sz, color='black')
            else:
                ax[0].plot(0.3 + np.random.randn(1) / 25, depth_rel34, marker, markersize=sz, color='firebrick')
        else:
            ax[0].plot(0.3+np.random.randn(1)/25, depth_rel34, marker, markersize=sz, color='lightgray')
    ax[0].set_ylabel('Depth from L3/4')
    ax[0].set_title(f"{siteid} - CSD")

    """
    if gui_depth_pix.max() > 200:
        print(f"old format channel_info?")
        if gui_depth_pix.max() == 1050:
            gui_depth_pix = gui_depth_pix/50-2
    c34_idx = np.argmin(np.abs(gui_depth_pix-landmarks['3/4']))
    c34_depth0 = depths0[c34_idx]
    c34_chan = chans[c34_idx]
    landmark_chans = {k: chans[np.argmin(np.abs(gui_depth_pix-v))] for k,v in landmarks.items() if v>0}
    landmark_idx = {k: np.argmin(np.abs(gui_depth_pix-v)) for k,v in landmarks.items() if v>0}
    landmark_depths = {k: c34_depth0 - depths0[np.argmin(np.abs(gui_depth_pix-v))] for k,v in landmarks.items() if v>0}
    depths = -depths0 + c34_depth0

    #landmark_depths = {k: depths[int(v)]  for k,v in landmarks.items() if v>0}
    #layerBorders = {'BS/1': -800, '3/4': 0, '4/5':200, '6/WM':800, 'WM/HC':2700}
    """

    layern = df_siteinfo.groupby('layer')[['index']].count()
    cts=",".join([f"{i}: {r['index']}" for i,r in layern.iterrows()])
    ax[0].set_xlabel(cts)

    # in separate panel, display PSD
    norm_power = power_crop / np.max(power_crop, axis=0, keepdims=True)
    ax[1].imshow(norm_power, origin='lower', aspect='auto',
                 extent=[0, norm_power.shape[1]] + ylims)
    ax[1].set_title('LFP norm power')

    # overlay lines indicating depth boundaries
    for k,v in landmarks.items():
        ax[1].axhline((c34_pix-v)*um_per_pix, color='darkgray', linestyle='--')
        ax[1].text(0,(c34_pix-v)*um_per_pix,k,color='darkgray', ha='right',va='center')

    # save to pdf
    dt = datetime.date.today().strftime("%Y-%m-%d")
    #figpath = f'/auto/users/svd/docs/current/grant/r21_free_moving/eps/{dt}/'
    figpath = '/home/svd/Documents/onedrive/projects/subspace_models/'
    os.makedirs(figpath, exist_ok=True)

    f.savefig(f'{figpath}depth_map_{siteid}.pdf')

if __name__ == '__main__':
    siteid='PRN015a'
    siteid='PRN020a'
    siteid='PRN034a'
    siteid='PRN033a'
    siteid='PRN043a'
    siteid='PRN059a'
    siteid='PRN050a'
    siteid='PRN057a'
    siteid='CLT047c'
    siteid='TNC047a'
    siteid='PRN048a'
    siteid='PRN064a'
    siteid="CLT039c"
    siteid="LMD022a"
    siteid="LMD047a"

    batch=343
    #siteids, cellids = db.get_batch_sites(batch)
    #for siteid in ['PRN018a']:
    #    plot_depth_map(siteid, batch=batch)

    siteid="CLT033c"
    siteid='PRN018a'
    siteid="CLT040c"
    siteid="PRN043a"
    siteid="PRN013c"
    #plot_depth_map(siteid, batch=batch)

    siteid, batch="TAR017b", 322
    siteid, batch="PRN020a", 343
    siteid, batch="ARM025a", 319
    plot_depth_map(siteid, batch=batch)

