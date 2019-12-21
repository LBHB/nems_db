import numpy as np
import os
import io
import logging

import nems.modelspec as ms
import nems.xforms as xforms
import nems.xform_helper as xhelp
from nems.utils import escaped_split, escaped_join
import nems.db as nd
from nems import get_setting
from nems.xform_helper import _xform_exists
from nems.registry import KeywordRegistry
from nems.plugins import (default_keywords, default_loaders,
                          default_initializers, default_fitters)
from nems.gui.recording_browser import browse_recording, browse_context
import nems.gui.editors as gui
import matplotlib.pyplot as plt
from nems.plots.api import ax_remove_box

import nems_lbhb.rdt.io as rio
from nems_lbhb.baphy import baphy_data_path
from nems.recording import load_recording
import seaborn as sns

log = logging.getLogger(__name__)

#batch, cellid, targetid, ylim = 269, 'chn022c-a2', '10', [0,50]
#batch, cellid, targetid, ylim = 269, 'chn019a-a1', '04', [0,100]
#batch, cellid, targetid, ylim = 273, 'oys058c-d1', '01', [0,50]
batch, cellid, targetid, ylim = 273, 'chn041d-b1', '07', [0,25]

options = {
    'cellid': cellid,
    'batch': batch,
    'rasterfs': 100,
    'includeprestim': 1,
    'stimfmt': 'ozgf',
    'chancount': 18,
    'pupil': 0,
    'stim': 1,
    'pertrial': 1,
    'runclass': 'RDT',
    'recache': False,
}

recording_uri = baphy_data_path(**options)

rec=load_recording(recording_uri)

resp = rec['resp']
respfast = resp.copy()
respfast.fs = 1000
respfast=respfast.rasterize()
resp=resp.rasterize()

ep = resp.epochs

firsttar = (ep['name'].str.startswith('Stim , '+targetid+'+')  & ep['name'].str.endswith('Target'))
s,e = ep['start'][firsttar].values, ep['end'][firsttar].values
s = (s*resp.fs).astype(int)
e = (e*resp.fs).astype(int)
b = np.array([s,e]).T
dur = int((e-s).mean())

maxreps=4
fig = plt.figure(figsize=(10,4))
for i in range(maxreps):
    plt.subplot(2,maxreps,i+1)
    raster = respfast.extract_epoch((b + dur*i)*10)[:,0,:]
    psth = np.mean(resp.extract_epoch(b + dur*i)[:,0,:], axis=0)

    x,y = np.where(raster)
    plt.plot(y,x,'.',color='black')
    plt.title('{} {} rep {}'.format(cellid,targetid,i))
    ax_remove_box(plt.gca())

    plt.subplot(2,maxreps,i+maxreps+1)
    plt.plot(psth*100,color='black')
    plt.ylim(ylim)
    ax_remove_box(plt.gca())

sns.despine(fig, offset=10)
plt.tight_layout()

fig.savefig('/Users/svd/Documents/current/RDT/nems/raster_'+cellid+'_tarid_'+targetid+'.pdf')

