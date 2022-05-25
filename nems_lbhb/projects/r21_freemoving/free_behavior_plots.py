
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#for PCA
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.motor import face_tools

from nems_lbhb import baphy_io, plots

training_path='/auto/data/dlc/free_top-svd-2022-04-25/videos/'
dlcfilepaths = [training_path+"20220422-123303DLC_resnet50_free_topApr25shuffle1_110000.h5",
              training_path+"20220422-125147DLC_resnet50_free_topApr25shuffle1_110000.h5",
              training_path+"20220422-131219DLC_resnet50_free_topApr25shuffle1_110000.h5"]
data=[]
xppm, yppm = 600, 600
x0, y0 = 1.0, -0.8
for i,d in enumerate(dlcfilepaths):
    data_array, list_bodyparts = baphy_io.load_dlc_trace(
        d, dlc_threshold=0.6, fill_invalid=np.nan, return_raw=True, verbose=True)
    headx=np.nanmean(data_array[[0,2,4],:],axis=0)/xppm
    heady=np.nanmean(data_array[[1,3,5],:],axis=0)/yppm - 0.05
    headx[(heady<0) | (heady>0.7)]=np.nan
    heady[(heady<0) | (heady>0.7)]=np.nan
    dx=(np.mean(data_array[[2,4],:],axis=0)-data_array[0,:])/xppm
    dy=(np.mean(data_array[[3,5],:],axis=0)-data_array[1,:])/yppm
    theta=np.arctan2(dy, dx)
    theta-=np.pi/2
    theta[theta<-np.pi]=theta[theta<-np.pi]+2*np.pi
    dist=np.sqrt((headx-x0)**2+(heady-y0)**2)
    velocity = np.concatenate([np.diff(dist),[np.nan]])
    dist[(dist<0.8) | (dist>1.4) | (np.abs(velocity)>0.1)]=np.nan
    proc_data=np.stack([headx, heady, dx, dy, dist, theta, velocity], axis=0)
    proc_bodyparts=['head_x','head_y','dir_x','dir_y', 'dist', 'theta']
    data.append(proc_data)

training_path2='/auto/data/dlc/FerretTest3/'
dlcfilepaths2 = [training_path2+"Test_2021_10_28_NAT_1.lick.originalDLC_resnet101_Ferret Test 3Oct29shuffle1_50000.h5",
              training_path2+"Test_2021_10_28_NAT_2.lick.originalDLC_resnet101_Ferret Test 3Oct29shuffle1_50000.h5",
              training_path2+"Test_2021_10_28_NAT_3.lick.originalDLC_resnet101_Ferret Test 3Oct29shuffle1_50000.h5"]
data2=[]
list_bodyparts2=['right_x','right_y','left_x','left_y',
                 'front_x','front_y','rear_x','rear_y']
xppm, yppm = 300, 300
x0, y0 = 1.0, -0.8
for i,d in enumerate(dlcfilepaths2):
    data_array, _ = baphy_io.load_dlc_trace(
        d, dlc_threshold=0.6, fill_invalid=np.nan, return_raw=True, verbose=True)
    headx=np.nanmean(data_array[[0,2,4],:],axis=0)/xppm - 0.05
    heady=0.7-np.nanmean(data_array[[1,3,5],:],axis=0)/yppm
    headx[(heady<0) | (heady>0.7)]=np.nan
    heady[(heady<0) | (heady>0.7)]=np.nan
    dx=(np.mean(data_array[[0,2],:],axis=0)-data_array[4,:])/xppm
    dy=-(np.mean(data_array[[1,3],:],axis=0)-data_array[5,:])/yppm
    theta=np.arctan2(dy, dx)
    theta-=np.pi/2
    theta[theta<-np.pi]=theta[theta<-np.pi]+2*np.pi
    dist=np.sqrt((headx-x0)**2+(heady-y0)**2)
    velocity = np.concatenate([np.diff(dist),[np.nan]])
    dist[(dist<0.8) | (dist>1.4) | (np.abs(velocity)>0.1)]=np.nan
    proc_data=np.stack([headx, heady, dx, dy, dist, theta, velocity], axis=0)
    proc_bodyparts=['head_x','head_y','dir_x','dir_y', 'dist', 'theta']

    data2.append(proc_data)

def circular_hist(d, N=20, ax=None):
    theta = np.linspace(-np.pi, np.pi, N+1)
    theta_centers=theta[1:]
    theta_edges = theta+(theta[1]-theta[0])/2
    radii, _ = np.histogram(d[np.isfinite(d)], bins=theta_edges)
    width = theta[1]-theta[0]
    colors = plt.cm.viridis(radii / 10.)
    if ax is None:
        ax = plt.subplot(projection='polar')

    ax.bar(theta_centers, radii, width=width, bottom=0.0, color=colors)

    plt.show()

f,ax = plt.subplots(2,3,figsize=(5,2.5), sharex='col')
for i in range(len(data)):
    ax[0,0].plot(data[i][0, :], data[i][1, :], lw=0.5)
ax[0,0].invert_yaxis()
ax[0,0].set_title('Go/no-go task')
ax[0,0].set_ylabel('Y position (m)')

dall=np.concatenate(data,axis=1)
sns.histplot(x=dall[4,:], bins=20, stat='percent', ax=ax[0,1])
sns.histplot(x=dall[5,:], bins=20, stat='percent', ax=ax[0,2])

for i in range(len(data2)):
    ax[1,0].plot(data2[i][0, :], data2[i][1, :], lw=0.5)

dall=np.concatenate(data2,axis=1)
sns.histplot(x=dall[4,:], bins=20, stat='percent', ax=ax[1,1])
sns.histplot(x=dall[5,:], bins=20, stat='percent', ax=ax[1,2])

ax[1,0].set_title('Free behavior')
ax[1,0].invert_yaxis()
ax[1,0].set_ylabel('Y position (m)')
ax[1,0].set_xlabel('X position (m)')
ax[1,1].set_xlabel('Dist. from speaker (m)')
ax[1,2].set_xlabel('Bearing (deg)')

plt.tight_layout()
f.savefig('/auto/users/svd/projects/free_moving/dlc_examples.pdf')
#plt.figure()
#circular_hist(dall[5,:])