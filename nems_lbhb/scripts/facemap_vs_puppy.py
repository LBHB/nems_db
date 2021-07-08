import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss

from nems_lbhb.baphy_experiment import BAPHYExperiment

parmfile = '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_05_25_BVT_1.m'

options = {'pupil': True, 'rasterfs': 100}

manager = BAPHYExperiment(parmfile=parmfile)
rec = manager.get_recording(**options)
pupil = rec['pupil']._data.squeeze()

# load facemap analysis
fmap_fn = '/auto/data/daq/Cordyceps/training2020/Cordyceps_2020_05_25_facemap.npy'
fmap_results = np.load(fmap_fn, allow_pickle=True).item()

fmap_pupil = fmap_results['pupil'][0]['area']

# resample nems pupil to match length
pupil = ss.resample(pupil, fmap_pupil.shape[0])

# plot results
f = plt.figure(figsize=(12, 4))
p1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
p2 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
scat = plt.subplot2grid((2, 3), (0, 2), rowspan=2)

p1.plot(pupil)
p1.set_title('lbhb results')
p1.set_ylabel('Pupil size')

p2.plot(fmap_pupil)
p2.set_title('facemap results')
p2.set_ylabel('Pupil size')

scat.scatter(fmap_pupil, pupil, s=10, color='k', alpha=0.2)
scat.set_xlabel('facemap')
scat.set_ylabel('lbhb')

f.tight_layout()

plt.show()
