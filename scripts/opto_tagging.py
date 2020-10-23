from nems_lbhb.baphy_experiment import BAPHYExperiment
import matplotlib.pyplot as plt
import numpy as np

parmfile = '/auto/data/daq/Armillaria/ARM004/ARM004d02_p_NON.m'
rasterfs = 1000
recache = True
options = {'resp': True, 'rasterfs': rasterfs}

manager = BAPHYExperiment(parmfile=parmfile)

rec = manager.get_recording(recache=recache, **options)
rec['resp'] = rec['resp'].rasterize()

prestim = rec['resp'].extract_epoch('PreStimSilence').shape[-1] / rasterfs
m = rec.copy().and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
poststim = (rec['resp'].extract_epoch('REFERENCE', mask=m['mask'], allow_incomplete=True).shape[-1] / rasterfs) + prestim
lim = (-prestim, (rec['resp'].extract_epoch('REFERENCE').shape[-1] / rasterfs) - prestim)

s = 1
unit = 'ARM004d-40-1'
n = int(len(rec['resp'].chans) / 4) + 1
f, ax = plt.subplots(n, 4, figsize=(14, n * 2))
a = ax.flatten()

for i, unit in enumerate(rec['resp'].chans):
    opt_data = rec['resp'].epoch_to_signal('LIGHTON')
    r = rec['resp'].extract_channels([unit]).extract_epoch('REFERENCE').squeeze()
    opt_mask = opt_data.extract_epoch('REFERENCE').mean(axis=(1,2)) > 0
    opt_s_stop = np.argwhere(np.diff(opt_data.extract_epoch('REFERENCE')[opt_mask, :, :][0].squeeze())) / rasterfs
    st = np.where(r[opt_mask, :])
    a[i].scatter((st[1] / rasterfs) - prestim, st[0], s=s, color='b')
    offset = st[0].max()
    st = np.where(r[~opt_mask, :])
    a[i].scatter((st[1] / rasterfs) - prestim, st[0]+offset, s=s, color='grey')
    for ss in opt_s_stop:
        a[i].axvline(ss, linestyle='--', color='darkorange')
    
    a[i].axvline(prestim, linestyle='--', color='lime')
    a[i].axvline(poststim, linestyle='--', color='lime')

    a[i].set_title(unit)
    a[i].set_xlabel('Time (s)')
    a[i].set_ylabel('Rep')

    a[i].set_xlim(lim[0], lim[1])

f.tight_layout()

plt.show()