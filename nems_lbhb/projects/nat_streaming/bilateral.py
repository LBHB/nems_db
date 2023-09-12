import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nems_lbhb import baphy_experiment, xform_wrappers
from nems0 import recording, epoch

parmfile = '/auto/data/daq/SlipperyJack/SLJ032/SLJ032a02_p_BNB.m'


# crummy FTC on both probes.
parmfile = '/auto/data/daq/SlipperyJack/SLJ017/SLJ017a11_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ017/SLJ017a12_p_BNT.m'


# maybe co-tuned FTC??? weirdly high correlations
parmfile = '/auto/data/daq/SlipperyJack/SLJ021/SLJ021a06_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ021/SLJ021a12_p_BNT.m'


# weak FTC. maybe overlap. increase cc?
parmfile = '/auto/data/daq/SlipperyJack/SLJ026/SLJ026a07_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ026/SLJ026a13_p_BNT.m'

parmfile = '/auto/data/daq/SlipperyJack/SLJ033/SLJ033a05_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ033/SLJ033a06_p_BNT.m'

# no tuning in probe B
parmfile = '/auto/data/daq/SlipperyJack/SLJ016/SLJ016a04_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ016/SLJ016a06_p_BNT.m'

# tuning unclear?
parmfile = '/auto/data/daq/SlipperyJack/SLJ003/SLJ003a04_p_BNT.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ004/SLJ004c11_p_BNT.m'

# different tuning? decrease??
parmfile = '/auto/data/daq/SlipperyJack/SLJ019/SLJ019a10_p_BNB.m'
parmfile = '/auto/data/daq/SlipperyJack/SLJ019/SLJ019a13_p_BNT.m'

ex = baphy_experiment.BAPHYExperiment(parmfile=[parmfile])
loadkey="psth.fs10"

rec = ex.get_recording(loadkey=loadkey)
#.sortparameters.Kilosort_load_completed_job_params

resp=rec['resp'].rasterize()

A_chans = [c for c in resp.chans if '-A-' in c]
B_chans = [c for c in resp.chans if '-B-' in c]
respA = resp.extract_channels(A_chans)
respB = resp.extract_channels(B_chans)

stim_epochs=epoch.epoch_names_matching(resp.epochs,"^STIM_")

if len(stim_epochs)>1:
     stim_epochs=epoch.epoch_names_matching(resp.epochs,"^STIM_00") + \
         epoch.epoch_names_matching(resp.epochs,"^STIM_NULL:1:0\+00")

     stim_epochs1=[s for s in stim_epochs if 'NULL' in s]
     stim_epochs2=[s for s in stim_epochs if 'NULL' not in s]
     bi_label=['mono','bilateral']
     stim_epochs = [stim_epochs1, stim_epochs2]
else:
     stim_epochs=[stim_epochs]
     bi_label=['mono']


f, ax = plt.subplots(3, 3)

for stim_epochs, lbl in zip(stim_epochs, bi_label):
     AA=[]
     AB=[]
     BB=[]
     for s in stim_epochs:

          rA=respA.extract_epoch(s)
          rB=respB.extract_epoch(s)

          vA = rA-rA.mean(axis=0,keepdims=True)
          vB = rB-rB.mean(axis=0,keepdims=True)
          vA=vA.transpose([1,0,2])
          vA=np.reshape(vA,[vA.shape[0], -1])
          vA-=vA.mean(axis=1, keepdims=True)
          s = vA.std(axis=1, keepdims=True)
          vA/= (s+(s==0))
          vB=vB.transpose([1,0,2])
          vB=np.reshape(vB,[vB.shape[0], -1])
          vB-=vB.mean(axis=1, keepdims=True)
          s = vB.std(axis=1, keepdims=True)
          vB/=(s+(s==0))
          N=vB.shape[1]

          AA.append((vA @ vA.T)/N)
          AB.append((vA @ vB.T)/N)
          BB.append((vB @ vB.T)/N)

     AA=np.mean(np.stack(AA, axis=2),axis=2)
     AB=np.mean(np.stack(AB, axis=2),axis=2)
     BB=np.mean(np.stack(BB, axis=2),axis=2)
     np.fill_diagonal(AA,0)
     np.fill_diagonal(BB,0)

     if lbl=='mono':
          imopts={'cmap':'gray', 'origin':'lower', 'vmin': -0.25, 'vmax': 0.25}
          ax[1,0].imshow(AA, **imopts)
          ax[1,0].set_title('A x A')
          ax[1,0].set_ylabel('Probe A unit')
          ax[1,1].imshow(AB, **imopts)
          ax[1,1].set_title('A x B')
          ax[2,1].imshow((vB @ vB.T)/N, **imopts)
          ax[2,1].set_title('B x B')
          ax[2,1].set_xlabel('Probe B unit')

     #ax[0,0].plot(AA.std(axis=0))
     #ax[0,1].plot(AB.std(axis=0))
     #ax[1,2].plot(AB.std(axis=1),np.arange(AB.shape[0]))
     #ax[2,2].plot(BB.std(axis=1),np.arange(BB.shape[0]))
     ax[0,0].plot(AA.mean(axis=0), label=lbl)
     ax[0,1].plot(AB.mean(axis=0))
     ax[1,2].plot(AB.mean(axis=1),np.arange(AB.shape[0]))
     ax[2,2].plot(BB.mean(axis=1),np.arange(BB.shape[0]))

     bins=np.linspace(-1, 1, 21)

     h = [np.histogram(AA[np.triu_indices(AA.shape[0], k=1)], bins=bins)[0],
          np.histogram(AB.flatten(), bins=bins)[0],
          np.histogram(BB[np.triu_indices(BB.shape[0], k=1)], bins=bins)[0]]
     h = np.stack(h, axis=1).astype(float)
     h /= h.sum(axis=0, keepdims=True)
     ax[2,0].plot((bins[1:]+bins[:-1])/2,h)
     ax[2,0].legend(('AA','AB','BB'))
     f.suptitle(os.path.basename(parmfile) +" " + lbl)

ax[0,0].legend()
plt.tight_layout()



from nems_lbhb.plots import ftc_heatmap

f,ax =plt.subplots(1,2)

mua=False
probe='B'
fs=100
smooth_win=5
resp_len=0.1
siteid=ex.siteid.split("_")[0][:7]
ftc_heatmap(siteid, mua=mua, probe='A', fs=fs,
            smooth_win=smooth_win, ax=ax[0])
ax[0].set_title('Probe A')
ftc_heatmap(siteid, mua=mua, probe='B', fs=fs,
            smooth_win=smooth_win, ax=ax[1])
ax[1].set_title('Probe B')
f.suptitle(siteid)
