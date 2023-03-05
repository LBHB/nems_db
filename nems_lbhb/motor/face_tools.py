#stardard imports
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg

#for PCA
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb import baphy_io
import nems_lbhb.plots as nplt

#function definitions
def invert_PCA(pca, X, Xpca=None, pc_index=0, scale_factor=1):
    '''
    X: imputed data
    Xpca: PCA data
    pc_index: which PC
    '''
    if Xpca is None:
        Xpca = pca.fit_transform(X) #fit model + apply dimensionality reduction

    print(Xpca.shape, X.shape)
    PC = Xpca[:,[pc_index]] * scale_factor
    evec = pca.components_[[pc_index],:].T
    print(evec.shape, PC.shape)
    movement = (evec @ PC.T) * np.nanstd(X.T, axis=1, keepdims=True) + \
               np.nanmean(X.T, axis=1, keepdims=True)

    return movement

def PC_trials(Xpca, pc_index, trialtype, task_data, pre_sec=2, post_sec=2, fps=30):
    '''
    Extract PC values from a window of time around trial start.
    inputs:
        Xpca: array of PCA data
        pc_index: index of PC to extract data from
        trialtype: trial outcome ID as str (e.g. 'HIT_TRIAL', 'MISS_TRIAL')
        pre_sec: number of seconds before trial start, when to begin extracting data
        post_sec: number of seconds after trial start, when to stop extracting data
        fps: frames per second
    outputs:
        pretrial_pc: array of PC values from pre_sec before trial onset to post_sec after trial onset
    '''

    PC = Xpca[:,[pc_index]].T

    trials = task_data.loc[(task_data.name==trialtype)].copy()
    if trials.iloc[0,1]<pre_sec:
        trials=trials.iloc[1:,:] #remove first trial if not enough pre-trial time
    num_trials = len(trials)

    num_pre_frames = pre_sec * fps
    num_post_frames = post_sec * fps
    trial_frames = (trials['start'].values*fps).astype(int)
    pre_frames = (trial_frames - num_pre_frames)
    post_frames = (trial_frames + num_post_frames)

    pretrial_pc = np.zeros((num_trials, num_pre_frames+num_post_frames))
    for n in range(num_trials):
        pretrial_pc[n] = PC[0,pre_frames[n]:post_frames[n]]

    return pretrial_pc

def my_absolute_mean_difference(a,b):
    """
    Helper function for flexible_permutation_test, compute the absolute mean difference between a and b
    """
    m = np.abs(a.mean()-b.mean())
    return m


def flexible_permutation_test(a, b, myfunc, N=100, verbose=False):
    """
    Permutation to test whether the the value of myfunc(a,b) is significantly greater
    than expected by chance.
    inputs:
       a, b: 1-D distributions of data
       myfunc: user-defined function such that stat=myfunc(a,b) returns a scalar value
            stat, which can be evaluated over permutations of a and b
       N: number of permutations
       verbose: plot a histrogram of the shuffled data, helpful for debugging.
    outputs:
       p: p-value, how likely the actual value occurred by chance
    """
    # first, compute the shuffled mean differences
    shuffled_distribution = np.zeros(N)
    for i in range(N):
        all_data = np.append(a, b)
        s = np.random.permutation(all_data)
        a_shuffled = s[:len(a)]
        b_shuffled = s[len(a):]
        shuffled_distribution[i] = myfunc(a_shuffled,b_shuffled)

    # now compute p
    actual_value = myfunc(a,b)
    effective_distribution = np.append(shuffled_distribution, actual_value)

    total_N = len(effective_distribution)
    greater_equal_N = np.sum(np.abs(effective_distribution) >= actual_value)
    p = greater_equal_N / total_N

    if verbose:
        plt.figure()
        plt.hist(shuffled_distribution, label='shuffled')
        plt.axvline(actual_value, color='r', label='acutal')
        plt.xlabel('statistic')
        plt.ylabel('count')
        plt.legend()
        #print('p = {}'.format(p))
        print('p = %.3f' % p)


def load_dlc_results(dlcfilepath, dlc_threshold=0.25):
    dataframe = pd.read_hdf(dlcfilepath)
    scorer = dataframe.columns.get_level_values(0)[0]
    bodyparts = dataframe[scorer].columns.get_level_values(0)

    num_frames = dataframe.shape[0]
    names_bodyparts = list(bodyparts.unique(level=0))
    num_bodyparts = len(names_bodyparts)

    data_array = np.zeros((num_bodyparts*2, num_frames))

    list_bodyparts = []

    for i,bp in enumerate(names_bodyparts):
        x = dataframe[scorer][bp]['x'].values
        y = dataframe[scorer][bp]['y'].values
        threshold_check = dataframe[scorer][bp]['likelihood'].values > dlc_threshold
        x[~threshold_check] = np.nan
        y[~threshold_check] = np.nan
        data_array[2*i] = x
        data_array[2*i+1] = y

        list_bodyparts.append(bp+"_x")
        list_bodyparts.append(bp+"_y")

    return data_array


def summary_plot(vid_paths):
    if type(vid_paths) is str:
        vid_paths=[vid_paths]

    vids=[]
    for video_file in vid_paths:
        vid_dir, base_file = os.path.split(video_file)
        vids.append(base_file)
        animal_path, penname = os.path.split(vid_dir)
        daq_path, animal = os.path.split(animal_path)
        path_sorted = os.path.join(vid_dir, 'sorted')

    vid_paths = [os.path.join(vid_dir, v) for v in vids]
    output_aliased = [os.path.join(path_sorted, v.replace(".avi",".dlc.h5")) for v in vids]
    print(output_aliased)

    dlc_list=[]
    for dlcfilepath in output_aliased:
        dlc, bodyparts = baphy_io.load_dlc_trace(
            dlcfilepath, return_raw=True, dlc_threshold=0.5)
        dlc_list.append(dlc)

    dlc = np.concatenate(dlc_list, axis=1)
    print(dlc.shape)
    scaler = StandardScaler() #normalize to mean
    X = scaler.fit_transform(dlc.T)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean') #replace NaN values with mean
    X = imp.fit_transform(X)

    pca = PCA() #create PC object
    Xpca = pca.fit_transform(X) #fit model + apply dimensionality reduction
    cov = pca.get_covariance() #compute covariance matrix

    # get a sample frame
    vid_dir, base_file = os.path.split(vid_paths[0])

    frameidx = 350
    tmp_frame_folder='/tmp/'
    frame_file = tmp_frame_folder + f'frame1.jpg'
    t = frameidx * (1 / 30)
    os.system(f"ffmpeg -y -ss {t} -i {vid_paths[0]} -vframes 1 {frame_file}")
    frame = mpimg.imread(frame_file)

    frame = frame[:,:,0].astype(float)
    frame[frame < 50] = 50

    plt.figure()
    plt.plot(Xpca[:1000,0:2])

    mv0 = invert_PCA(pca, dlc.T, Xpca=Xpca, pc_index=0, scale_factor=2)
    mv1 = invert_PCA(pca, dlc.T, Xpca=Xpca, pc_index=1, scale_factor=2)
    mv2 = invert_PCA(pca, dlc.T, Xpca=Xpca, pc_index=2, scale_factor=2)

    print(dlc.std(axis=1))
    print(mv0.std(axis=1))

    f, ax = plt.subplots(2,2,figsize=(8,6), sharex=True, sharey=True)
    ax = ax.flatten()

    ax[0].imshow(frame,clim=[-255,255+128],cmap='gray')

    for i in range(0, len(bodyparts),2):
        ax[0].scatter(dlc[i,::4],dlc[i+1,::4], s=3, label=bodyparts[i].replace("_x",""), alpha=0.25)
    ax[0].legend(frameon=False, fontsize=6)

    ax[1].imshow(frame,clim=[-64,255+64],cmap='gray')
    ax[2].imshow(frame,clim=[-64,255+64],cmap='gray')
    ax[3].imshow(frame,clim=[-64,255+64],cmap='gray')

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(0,len(bodyparts),2):
        aa=np.argmin(mv0[i,:])
        bb=np.argmax(mv0[i,:])
        lab=bodyparts[i].replace("_x","")
        ax[1].plot([mv0[i,aa],mv0[i,bb]],[mv0[i+1,aa],mv0[i+1,bb]], lw=2, color=cycle[int(i/2)], label=lab)
        ax[2].plot([mv1[i,aa],mv1[i,bb]],[mv1[i+1,aa],mv1[i+1,bb]], lw=2, color=cycle[int(i/2)], label=lab)
        ax[3].plot([mv2[i,aa],mv2[i,bb]],[mv2[i+1,aa],mv2[i+1,bb]], lw=2, color=cycle[int(i/2)], label=lab)
    ax[2].legend(frameon=False, fontsize=6)
    ax[1].set_title('pc 1')
    ax[2].set_title('pc 2')
    ax[3].set_title('pc 3')
    ax[1].set_ylim([40,220])
    ax[1].set_xlim([20,frame.shape[1]-21])
    ax[1].invert_yaxis()
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    f.suptitle(base_file)
    return f