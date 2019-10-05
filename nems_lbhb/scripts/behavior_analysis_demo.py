import nems_lbhb.behavior as beh
import matplotlib.pyplot as plt
import numpy as np
parmfile = '/auto/data/daq/Drechsler/training2019/Drechsler_2019_10_03_BVT_4.m'  
options = {}
# plot an example RT histogram
RT = beh.get_RT(parmfile, **options)
HR = beh.get_HR_FAR(parmfile, **options)

bins = np.arange(0, 1, 0.05)
f, ax = plt.subplots(1, 1)
for k in RT.keys(): 
    counts, xvals = np.histogram(RT[k], bins=bins)

    ax.step(xvals[:-1], np.cumsum(counts) / len(RT[k]) * HR[k], label=k)

ax.legend()
ax.set_xlabel('Reaction time')

plt.show()
