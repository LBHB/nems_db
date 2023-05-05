import matplotlib.pyplot as plt
import numpy as np
import json
import os
import datetime

from laminar_tools.lfp import lfp
from nems_lbhb.baphy_io import load_continuous_openephys, get_spike_info, get_depth_info
from nems0 import db

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

d_raw = db.pd_query(f"SELECT * FROM gDataRaw where cellid like '{siteid}%%' AND runclass='BNB' AND not(bad) AND not(isnull(depthinfo))")

# use first match
rawid = d_raw.loc[0,'id']
parmfile = [d_raw.loc[0,'resppath'] + d_raw.loc[0,'parmfile']]
highlight_cellids = []

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


df_siteinfo = get_spike_info(siteid=siteid, save_to_db=True)
#df_siteinfo=df_siteinfo.loc[(df_siteinfo.area=='BS') | (df_siteinfo.area=='A1')]

df_siteinfo=df_siteinfo.reset_index()
df_siteinfo['channel']=df_siteinfo['index'].apply(lambda x: int(x[8:11]))

d=db.pd_query(f'SELECT * FROM gDataRaw where id={rawid}')

d_labeled_depth = json.loads(d.loc[0,'depthinfo'])
landmarks = d_labeled_depth['landmarkPosition']

left_csd, power, freqs, windowtime, rasterfs, column_xy_sorted, column_xy, channel_xy, coh_mat, erp, probe = \
    lfp.parmfile_event_lfp(parmfile)

s = slice(52,95)
s = slice(0,None)
csd_crop=left_csd[s,80:]
power_crop = power[s,:]
chans = column_xy_sorted[s]

depths0 = np.array([float(column_xy[c][1]) for c in chans])
channums = np.array(chans).astype(int)

dmin, dmax = df_siteinfo['depth'].min(), df_siteinfo['depth'].max()
cmin, cmax = df_siteinfo['channel'].max(), df_siteinfo['channel'].min()

d0max = float(channel_xy[str(cmax)][1])
d0min = float(channel_xy[str(cmin)][1])



# position of one column on probe (tip = 0)
all_depths0 = np.array([float(column_xy[c][1]) for c in column_xy_sorted])

c34_depth0 = all_depths0[int(np.round(landmarks['3/4']))]
depths = -depths0 + c34_depth0

landmark_depths = {k: depths[int(v)]  for k,v in landmarks.items() if v>0}
layerBorders = {'BS/1': -800, '3/4': 0, '4/5':200, '6/WM':800, 'WM/HC':2700}
df_siteinfo['depths_rel34'] = -df_siteinfo['depth0'] + c34_depth0

f,ax = plt.subplots(1, 2, sharey=True)

ax[0].imshow(csd_crop, origin='lower', aspect='auto', extent=[-0.1, 0.5, depths[0], depths[-1] ])
for i,r in df_siteinfo.iterrows():
    if r['index'] in highlight_cellids:
        ax[0].plot(0.3+np.random.randn(1)/15, r['depths_rel34'], 'w.')
    elif r['area'] in ['A1','PEG','BS']:
        ax[0].plot(0.3+np.random.randn(1)/15, r['depths_rel34'], 'k.')
    else:
        ax[0].plot(0.3+np.random.randn(1)/15, r['depths_rel34'], '.', color='darkred')
ax[0].set_ylabel('Depth from L3/4')
ax[0].set_title(f"{siteid} - CSD")

norm_power = power_crop / np.max(power_crop, axis=0, keepdims=True)

ax[1].imshow(norm_power, origin='lower', aspect='auto', extent=[0, norm_power.shape[1], depths[0], depths[-1] ])
ax[1].set_title('LFP norm power')

for k,v in landmark_depths.items():
    if v<depths.max():
        ax[0].axhline(v, color='r')
        ax[1].axhline(v, color='r')
        ax[1].text(0,v,k,color='r', ha='right',va='center')

dt = datetime.date.today().strftime("%Y-%m-%d")
figpath = f'/auto/users/svd/docs/current/grant/r21_free_moving/eps/{dt}/'
os.makedirs(figpath, exist_ok=True)

f.savefig(f'{figpath}depth_map_{siteid}.pdf')
