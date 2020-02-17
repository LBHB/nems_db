import matplotlib.pyplot as plt
import numpy as np

from nems_lbhb.io import BAPHYExperiment

batch = 307
siteid = 'TAR010c'
rawid = (123675, 123676, 123677, 123681)

manager = BAPHYExperiment(batch=batch, siteid=siteid, rawid=rawid) 
options = {'rasterfs': 20, 'resp': True, 'pupil': True, 'stim': False,
            'batch': batch, 'siteid': siteid, 'rawid': rawid}
r = manager.get_recording(**options)

import nems_lbhb.baphy as nb
from nems.recording import Recording
uri = nb.baphy_load_recording_uri(recache=False, **options)
r2 = Recording.load(uri)

r['resp'] = r['resp'].rasterize().extract_channels(r2['resp'].chans)
r2['resp'] = r2['resp'].rasterize()

# get behavior performance
performance = manager.get_behavior_performance(**options)
print('Performance over all trials: ')
print(performance)

# get performance only over the first 50 behavior trials
trial_range = np.arange(0, 50)
performance = manager.get_behavior_performance(trials=trial_range, **options)
print('Performance over all first 50 trials: ')
print(performance)

# plot trial averaged TARGET pupil on HIT_TRIALS and MISS_TRIALS
hit1 = r.and_mask(['HIT_TRIAL'])['mask']
miss1 = r.and_mask(['MISS_TRIAL'])['mask']
hit_pupil = r['pupil'].extract_epoch('TARGET', mask=hit1) 
miss_pupil = r['pupil'].extract_epoch('TARGET', mask=miss1) 

f, ax = plt.subplots(1, 2)

t = np.arange(0, hit_pupil.mean(axis=0).shape[-1] / options['rasterfs'], 1 / options['rasterfs'])
m = hit_pupil.mean(axis=0).squeeze()
sem = hit_pupil.std(axis=0).squeeze() / np.sqrt(hit_pupil.shape[0])
ax[0].plot(t, m, color='purple')
ax[0].fill_between(t, m-sem, m+sem, color=ax[0].get_lines()[-1].get_color(), alpha=0.4, lw=0, label='hit')
m = miss_pupil.mean(axis=0).squeeze()
sem = miss_pupil.std(axis=0).squeeze() / np.sqrt(miss_pupil.shape[0])
ax[0].plot(t, m, color='blue')
ax[0].fill_between(t, m-sem, m+sem, color=ax[0].get_lines()[-1].get_color(), alpha=0.4, lw=0, label='miss')
ax[0].legend()
ax[0].set_ylabel('pupil size')
ax[0].set_label('time')
ax[0].set_title('BAPHYExp loader')

hit2 = r2.and_mask(['HIT_TRIAL'])['mask']
miss2 = r2.and_mask(['MISS_TRIAL'])['mask']
hit_pupil = r2['pupil'].extract_epoch('TARGET', mask=hit2) 
miss_pupil = r2['pupil'].extract_epoch('TARGET', mask=miss2) 

t = np.arange(0, hit_pupil.mean(axis=0).shape[-1] / options['rasterfs'], 1 / options['rasterfs'])
m = hit_pupil.mean(axis=0).squeeze()
sem = hit_pupil.std(axis=0).squeeze() / np.sqrt(hit_pupil.shape[0])
ax[1].plot(t, m, color='purple')
ax[1].fill_between(t, m-sem, m+sem, color=ax[1].get_lines()[-1].get_color(), alpha=0.4, lw=0, label='hit')
m = miss_pupil.mean(axis=0).squeeze()
sem = miss_pupil.std(axis=0).squeeze() / np.sqrt(miss_pupil.shape[0])
ax[1].plot(t, m, color='blue')
ax[1].fill_between(t, m-sem, m+sem, color=ax[1].get_lines()[-1].get_color(), alpha=0.4, lw=0, label='miss')
ax[1].legend()
ax[1].set_ylabel('pupil size')
ax[1].set_label('time')
ax[1].set_title('old loader')

f.tight_layout()

# plot example REFERENCE raster for hit / miss trials, use masks from above
f, ax = plt.subplots(1, 2, figsize=(8, 4))
cellid = 16
hit_raster = r['resp'].extract_epoch('TARGET', mask=hit1)[:, cellid, :] 
miss_raster = r['resp'].extract_epoch('TARGET', mask=miss1)[:, cellid, :]  

t = np.arange(0, hit_raster.mean(axis=0).shape[-1] / options['rasterfs'], 1 / options['rasterfs'])
m = hit_raster.mean(axis=0).squeeze()
sem = hit_raster.std(axis=0).squeeze() / np.sqrt(hit_raster.shape[0])
ax[0].plot(t, m, color='purple')
ax[0].fill_between(t, m-sem, m+sem, color=ax[0].get_lines()[-1].get_color(), alpha=0.4, lw=0, label='hit')
m = miss_raster.mean(axis=0).squeeze()
sem = miss_raster.std(axis=0).squeeze() / np.sqrt(miss_raster.shape[0])
ax[0].plot(t, m, color='blue')
ax[0].fill_between(t, m-sem, m+sem, color=ax[0].get_lines()[-1].get_color(), alpha=0.4, lw=0, label='miss')
ax[0].legend()
ax[0].set_ylabel('spike count')
ax[0].set_label('time')
ax[0].set_title('BAPHYExp loader')

hit_raster = r2['resp'].extract_epoch('TARGET', mask=hit2)[:, cellid, :] 
miss_raster = r2['resp'].extract_epoch('TARGET', mask=miss2)[:, cellid, :]  

t = np.arange(0, hit_raster.mean(axis=0).shape[-1] / options['rasterfs'], 1 / options['rasterfs'])
m = hit_raster.mean(axis=0).squeeze()
sem = hit_raster.std(axis=0).squeeze() / np.sqrt(hit_raster.shape[0])
ax[1].plot(t, m, color='purple')
ax[1].fill_between(t, m-sem, m+sem, color=ax[1].get_lines()[-1].get_color(), alpha=0.4, lw=0, label='hit')
m = miss_raster.mean(axis=0).squeeze()
sem = miss_raster.std(axis=0).squeeze() / np.sqrt(miss_raster.shape[0])
ax[1].plot(t, m, color='blue')
ax[1].fill_between(t, m-sem, m+sem, color=ax[1].get_lines()[-1].get_color(), alpha=0.4, lw=0, label='miss')
ax[1].legend()
ax[1].set_ylabel('spike count')
ax[1].set_label('time')
ax[1].set_title('old loader')

f.tight_layout()


plt.show()