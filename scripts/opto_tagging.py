from nems_lbhb.baphy_experiment import BAPHYExperiment
import matplotlib.pyplot as plt
import numpy as np
import nems_lbhb.baphy_io as io
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

parmfile = '/auto/data/daq/Armillaria/ARM005/ARM005e04_p_NON.m'
animal = 'Armillaria'
rasterfs = 1000
recache = False
options = {'resp': True, 'rasterfs': rasterfs}

manager = BAPHYExperiment(parmfile=parmfile)
tend = 0.2

rec = manager.get_recording(recache=recache, **options)
rec['resp'] = rec['resp'].rasterize()

prestim = rec['resp'].extract_epoch('PreStimSilence').shape[-1] / rasterfs
m = rec.copy().and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
poststim = (rec['resp'].extract_epoch('REFERENCE', mask=m['mask'], allow_incomplete=True).shape[-1] / rasterfs) + prestim
lim = (-prestim, tend ) #(rec['resp'].extract_epoch('REFERENCE').shape[-1] / rasterfs) - prestim)

s = 1
n = int(len(rec['resp'].chans) / 4) + 1
f, ax = plt.subplots(n, 5, figsize=(14, n * 2))
a = ax.flatten()

for i, unit in enumerate(rec['resp'].chans):
    opt_data = rec['resp'].epoch_to_signal('LIGHTON')
    r = rec['resp'].extract_channels([unit]).extract_epoch('REFERENCE').squeeze()
    opt_mask = opt_data.extract_epoch('REFERENCE').mean(axis=(1,2)) > 0
    opt_s_stop = (np.argwhere(np.diff(opt_data.extract_epoch('REFERENCE')[opt_mask, :, :][0].squeeze())) + 1) / rasterfs
    st = np.where(r[opt_mask, :])
    a[i].scatter((st[1] / rasterfs) - prestim, st[0], s=s, color='b')
    offset = st[0].max()
    st = np.where(r[~opt_mask, :])
    a[i].scatter((st[1] / rasterfs) - prestim, st[0]+offset, s=s, color='grey')
    for ss in opt_s_stop:
        a[i].axvline(ss - prestim, linestyle='--', color='darkorange')
    
    a[i].axvline(0, linestyle='--', color='lime')
    a[i].axvline(poststim - prestim, linestyle='--', color='lime')

    a[i].set_title(unit)
    a[i].set_xlabel('Time (s)')
    a[i].set_ylabel('Rep')

    a[i].set_xlim(lim[0], lim[1])

    # add inset for mwf
    ax2 = plt.axes([0,0,1,1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(a[i], [0.5,0.5,0.5,0.5])
    ax2.set_axes_locator(ip)
    mwf = io.get_mean_spike_waveform(unit, animal)
    ax2.plot(mwf, color='red')
    ax2.axis('off')

f.tight_layout()

plt.show()