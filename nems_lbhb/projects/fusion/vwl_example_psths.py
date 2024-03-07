import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nems0 import db
import nems0.epoch as ep
from nems0.recording import load_recording
import nems0.preprocessing as preproc
from nems0.plots.api import raster
from nems0.utils import smooth

from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment

from os.path import basename, join

rasterfs = 100

siteid = "LMD026a"
siteid = "LMD022a"
siteid = "LMD034a"
siteid = "LMD039a"
channels = [4,5]
siteid = "LMD019a"
channels = [33,36,38,47,48,49,53,54,57,58,60,65,68,69,71,73]
channels = [33,38,48,49,60,65,68,69,71,73]
channels = [33,38,48,71,73]

sql = f"SELECT DISTINCT stimpath, stimfile FROM sCellFile WHERE cellid like '{siteid}%' AND runclassid=125"
d = db.pd_query(sql)
parmfiles = [join(r['stimpath'],r['stimfile']) for i,r in d.iterrows()]

ex = BAPHYExperiment(parmfile=parmfiles)
rec = ex.get_recording(stim=False, resp=True, rasterfs=rasterfs)

resp = rec['resp'].rasterize()
#channels = np.arange(resp.shape[0])

stim_epochs = ep.epoch_names_matching(resp.epochs,"^STIM_")
stim_epochs
use_epochs = [
    'STIM_AH-0-0.24-2_AH-0-0.24-1-0dB',
    'STIM_EE-0-0.24-1_EE-0-0.24-2-0dB',
    #'STIM_AE-0-0.24-1_AE-0-0.24-2-0dB',
    'STIM_EH-0-0.24-1_EH-0-0.24-2-0dB',
]
probe_epoch='STIM_AH-0-0.24-1_EE-0-0.24-2-0dB'
short_titles=['/ah/-/ah/','/ee/-/ee/','/eh/-/eh/']

mses = np.zeros((len(use_epochs),len(channels)))
for os in np.arange(0, len(channels), 10, dtype=int):
    colcount=np.min([10,len(channels)])
    f, ax = plt.subplots(len(use_epochs), colcount, figsize=(colcount*0.75,3), sharex=True, sharey=True)
    for channel in range(10):
        if channel+os<len(channels):
            for i,e in enumerate(use_epochs):
                cid = int(channels[channel+os])
                #resp.plot_raster(epoch=e, channel=int(channels[channel+os]), ax=ax[i,channel])
                #resp.plot_epoch_avg(epoch=e, channel=cid,
                #                    ax=ax[i,channel], prestimsilence=0.1, color='gray', lw=0.5)
                #resp.plot_epoch_avg(epoch=probe_epoch, channel=cid,
                #                    ax=ax[i,channel], prestimsilence=0.1, color='k', lw=0.5)

                rref = resp.extract_epoch(e)[:,cid,:].mean(axis=0)*resp.fs
                rprobe = resp.extract_epoch(probe_epoch)[:,cid,:].mean(axis=0)*resp.fs
                mses[i,channel] = np.std(rref-rprobe)/np.sqrt(np.mean(rprobe**2))
                tt=np.arange(len(rref))/resp.fs*1000
                ax[i,channel].plot(rref,color='gray',lw=0.5)
                ax[i,channel].plot(rprobe,color='k',lw=0.5)
                if channel==0:
                    ax[i,0].set_ylabel(short_titles[i], color='gray')
                else:
                    ax[i,channel].set_ylabel('')
                ax[i,channel].set_title('')
                ax[i,channel].set_xlabel('')
            ax[0,channel].set_title(f"u{channels[channel+os]}")
    yl = ax[0, 0].get_ylim()
    for channel in range(10):
        if channel+os<len(channels):
            for i,e in enumerate(use_epochs):
                if mses[i,channel] == mses[:,channel].min():
                    col='r'
                else:
                    col='gray'
                ax[i,channel].text(0, yl[1], f"{mses[i,channel]:.2f}", fontsize=8, va='top', color=col)

    ax[-1,2].set_xlabel(f"Time (ms) - Probe (/ah/-/eh/) response in black")
    plt.tight_layout()

f.savefig(f'/auto/data/tmp/vwl_examples_{siteid}.pdf')
