import matplotlib.pyplot as plt
import numpy as np
import json

from laminar_tools.lfp import lfp
from nems_lbhb.baphy_io import load_continuous_openephys, get_spike_info, get_depth_info
from nems0 import db

parmfile = ['/auto/data/daq/Prince/PRN015/PRN015b04_p_BNB.m']
siteid='PRN015a'
rawid=146865
highlight_cellids=[ "PRN015a-315-1",
                        "PRN015a-282-1",
                        "PRN015a-225-1"]
df_siteinfo = get_spike_info(siteid=siteid, save_to_db=True)
df_siteinfo=df_siteinfo.loc[(df_siteinfo.area=='BS') | (df_siteinfo.area=='A1')]

df_siteinfo=df_siteinfo.reset_index()
df_siteinfo['channel']=df_siteinfo['index'].apply(lambda x: int(x[8:11]))

d=db.pd_query(f'SELECT * FROM gDataRaw where id={rawid}')

d_labeled_depth = json.loads(d.loc[0,'depthinfo'])
landmarks = d_labeled_depth['landmarkPosition']

left_csd_padded, power, freqs, windowtime, rasterfs, column_xy_sorted, channel_xy, coh_mat, erp = \
    lfp.parmfile_event_lfp(parmfile)

s = slice(52,95)
csd_crop=left_csd_padded[s,80:]
power_crop = power[s,:]
chans = column_xy_sorted[s]

depths0 = np.array([float(channel_xy[c][1]) for c in chans])
channums = np.array(chans).astype(int)

dmin, dmax = df_siteinfo['depth0'].min(), df_siteinfo['depth'].max()
cmin, cmax = df_siteinfo['channel'].max(), df_siteinfo['channel'].min()

d0max = 3820-float(channel_xy[str(cmax)][1])
d0min = 3820-float(channel_xy[str(cmin)][1])

adjust = d0max-dmax
depths = 3820-depths0 - adjust

all_depths0 = np.array([float(channel_xy[c][1]) for c in column_xy_sorted])
all_depths = 3820-all_depths0 - adjust
landmark_depths = {k: all_depths[int(v)] for k,v in landmarks.items()}
layerBorders = {'BS/1': -800, '3/4': 0, '4/5':200, '6/WM':800, 'WM/HC':2700}
f,ax = plt.subplots(1,2)

ax[0].imshow(csd_crop, origin='lower', aspect='auto', extent=[-0.1, 0.5, depths[0], depths[-1] ])
for i,r in df_siteinfo.iterrows():
    if r['index'] in highlight_cellids:
        ax[0].plot(0.3+np.random.randn(1)/20, 3820-r['depth0']-adjust, 'w.')
    else:
        ax[0].plot(0.3+np.random.randn(1)/20, 3820-r['depth0']-adjust, 'k.')

for k,v in landmark_depths.items():
    if v<depths.max():
        ax[0].plot([-0.1, 0.5], [v, v], 'r')
ax[0].set_title(siteid)

ax[1].imshow(power_crop, origin='lower', aspect='auto')
figpath = '/auto/users/svd/docs/current/grant/r21_free_moving/eps/'
f.savefig(f'{figpath}depth_map_{siteid}.pdf')
