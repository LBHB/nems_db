
import numpy as np
import matplotlib.pyplot as plt

#for PCA
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from nems import db
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.motor import face_tools

import importlib
importlib.reload(face_tools)
plt.close('all')

batch = 324
siteid="CLT021a"
df = db.get_batch_cell_data(batch=batch, cellid=siteid)
parmfiles = list(df.reset_index()['parm'].unique())
vid_paths = [f.replace('.m','.lick.avi') for f in parmfiles]
f = face_tools.summary_plot(vid_paths[1:2])

f.savefig('/auto/users/svd/projects/free_moving/face_pca.pdf')


"""
#parmfile = '/auto/data/daq/Clathrus/training2022/Clathrus_2022_01_11_TBP_1.m'
#parmfile = '/auto/data/daq/Clathrus/CLT011/CLT011a05_a_TBP.m'

parmfile1 = '/auto/data/daq/Clathrus/CLT021/CLT021a04_a_TBP.m'
experiment = BAPHYExperiment(parmfile=parmfile1)
rec1 = experiment.get_recording(rasterfs=30, recache=False, dlc=True,
                               resp=True, stim=False, dlc_threshold=0.25)

#parmfile2 = '/auto/data/daq/Clathrus/CLT021/CLT021a05_p_TBP.m'
parmfile2 = '/auto/data/daq/Clathrus/CLT021/CLT021a10_p_NON.m'
experiment = BAPHYExperiment(parmfile=parmfile2)
rec2 = experiment.get_recording(rasterfs=30, recache=False, dlc=True,
                               resp=True, stim=False, dlc_threshold=0.25)

# find lick events
_d=rec1['dlc']._data.copy()
d1 = rec1['dlc']._modified_copy(data=_d-_d.mean(axis=1, keepdims=True))
_d=rec2['dlc']._data.copy()
d2 = rec2['dlc']._modified_copy(data=_d-_d.mean(axis=1, keepdims=True))
dlc = d1.concatenate_time([d1,d2])

r1=rec1['resp'].rasterize()
r2=rec2['resp'].rasterize()
r1len=r1.shape[1]
rsig = r1.concatenate_time([r1,r2])

resp = rsig._data
data_array = dlc._data.copy()

fps = dlc.fs
bp_coords = dlc.chans



imp = SimpleImputer(missing_values=np.nan, strategy='mean') #replace NaN values with mean
filled = imp.fit_transform(data_array.T)
scaler = StandardScaler() #normalize to mean
X = scaler.fit_transform(filled)

pca = PCA() #create PC object
Xpca = pca.fit_transform(X) #fit model + apply dimensionality reduction
cov = pca.get_covariance() #compute covariance matrix

if 0:
    # cov matrix
    plt.figure()
    plt.imshow(cov,extent=[0,14,0,14]);
    plt.xticks(np.arange(14), bp_coords, rotation=70);
    plt.yticks(np.arange(14), bp_coords);
    plt.colorbar()
    plt.title('facial marker covariance matrix');

if 0:
    #plot eigenvectors
    plt.figure(figsize=(8,4))
    plt.plot(bp_coords, pca.components_[:,:2])
    plt.legend(['PC1', 'PC2'], loc=(1.04,0.5))
    plt.axhline(y=0,color='k',linestyle='--')
    plt.xticks(rotation=60);
    plt.ylabel('score')
    plt.xlabel('marker')
    plt.title('facial marker eigenvectors');

#get PC reconstructions
mean_vals = np.nanmean(filled.T, axis=1, keepdims=True)
movement1 = face_tools.invert_PCA(pca,filled, Xpca=Xpca, pc_index=0, scale_factor=2)
movement2 = face_tools.invert_PCA(pca,filled, Xpca=Xpca, pc_index=1, scale_factor=2)

#get lick times
show_sec = 30
show_frames=int(show_sec*dlc.fs)
t=np.arange(show_frames)/dlc.fs
task_data=dlc.epochs
lick_events = task_data.loc[(task_data.name=='LICK') & (task_data.start<show_sec)].copy()
lick_events['y']=10

f, ax = plt.subplots(2,1,figsize=(12,4), sharex=True)

ax[0].plot(Xpca[:show_frames,:2]);
for l in lick_events['start'].index:
    ax[0].axvline(lick_events['start'][l]*fps, color='g', linewidth=1, alpha=0.25)
#ax[0].plot(lick_events['start']*fps,lick_events['y'],'r.') #spout licks as detected by sensor
ax[0].title.set_text('first 2 PCs over time')
ax[0].set_xlabel('frame')
ax[0].set_ylabel('score')
ax[0].legend(['PC1','PC2','spout lick detected'], loc=(1.01,0.3));

im=ax[1].imshow(resp[:,:show_frames], vmax=2,
                interpolation='none', aspect='auto', cmap='gray_r')

r_norm = scaler.fit_transform(resp.T)
pc_norm = scaler.fit_transform(Xpca)

xc1 = r_norm[:r1len,:].T @ pc_norm[:r1len,:]/r1len
xc2 = r_norm[r1len:,:].T @ pc_norm[r1len:,:]/(r_norm.shape[1]-r1len)
f,ax = plt.subplots(1,2)
ax[0].imshow(xc1, clim=[-0.1,0.1])
ax[1].imshow(xc2, clim=[-0.1,0.1])


plt.figure()
plt.plot(Xpca[:,:3])

plt.figure()
plt.plot(data_array[::2,:].T)

"""