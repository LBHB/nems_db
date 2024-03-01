import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nems_lbhb import baphy_io, baphy_experiment
from nems_lbhb.xform_wrappers import baphy_load_wrapper, generate_recording_uri
from nems0.recording import load_recording
from nems0 import db

#site_list = ['CLT007a', 'CLT014a', 'PRN007a', 'PRN014b', 'PRN015b', 'PRN023a']
site_list = ['CLT007a', 'CLT008a', 'CLT009a', 'CLT011a', 'CLT012a',
             'PRN014b', 'PRN015b']

d = [baphy_io.get_spike_info(siteid=s, save_to_db=True) for s in site_list]
d = pd.concat(d)

#recs = []
#for c in site_list:
#    sql = f"SELECT distinct stimpath, stimfile FROM sCellFile WHERE cellid like '{c}%%' AND runclassid={131}"
#    dff = db.pd_query(sql)
#    parmfile = [r.stimpath+r.stimfile for i, r in dff.iterrows()]
#    manager = baphy_experiment.BAPHYExperiment(parmfile=parmfile)
#    recording_uri = manager.get_recording_uri(loadkey="gtgram.fs50.ch18")
#    recs.append(load_recording(recording_uri))

#f,ax = plt.subplots(2,2)
a1 = (d['layer']!='WM')

#plt.close("all")
#x='iso'
x = 'sw'
j = sns.jointplot(data=d, x=x, y='depth',
                  xlim=[0.1,1.0], ylim=[3000, -1000], hue='siteid')
j.ax_joint.plot([0.1, 0.8], [800, 800], 'k--')

plt.figure()
bad=0
for i,r in d.loc[d.siteid=='CLT007a'].iterrows():
    x0=r['sw']
    y0=r['depth']/1000
    mwf = -r['mwf']/10 + y0
    try:
        t = np.arange(len(mwf))/2000 + x0
        plt.plot(t, mwf, lw=0.5)
    except:
        bad+=1
        print(f"{bad} mwf: {i}, {r['depth']}")
plt.ylim([3.0, -1.2])
