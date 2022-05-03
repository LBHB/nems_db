#run in deeplabcut environment (tensorflow 2.4, cuda 11.0, etc)
import deeplabcut as dlc
import os
import glob

"""
very helpful tutorial/notes on refinement
https://guillermohidalgogadea.com/openlabnotebook/refining-your-dlc-model/
"""

#change if using a different DLC model
path_config='/auto/data/dlc/multivid-CLL-2022-01-14/config.yaml'

penname='CLT020'
if penname=='CLT011':
    #leave blank to analyze ALL compressed vids in vid_dir; otherwise enter list of vid names
    #vids=['CLT011a07_p_TBP.lick.avi']
    #          'CLT011a05_a_TBP.lick.avi',
    vids=['CLT011a04_p_TBP.lick.avi',
          'CLT011a07_p_TBP.lick.avi',
          'CLT011a08_a_TBP.lick.avi',
          'CLT011a09_p_TBP.lick.avi']

elif penname == 'CLT009':
    vids=['CLT009a09_a_TBP.lick.avi']
elif penname == 'CLT007':
    vids=['CLT007a04_a_TBP.lick.avi']
elif penname == 'CLT020':
    vids=['CLT020a04_p_TBP.lick.avi',
          'CLT020a05_a_TBP.lick.avi',
          'CLT020a06_p_TBP.lick.avi',
          'CLT020a07_a_TBP.lick.avi',
          'CLT020a08_p_NON.lick.avi']

vid_dir=f'/auto/data/daq/Clathrus/{penname}/'
path_sorted = os.path.join(vid_dir, 'sorted')

if not os.path.exists(path_sorted):
    os.makedirs(path_sorted)
    print('Created sorted directory in', vid_dir)

if not vids: #get list of all compressed videos in training folder if no vids given as input
    for f in os.listdir(vid_dir):
        if f.endswith('lick.avi'): #non-compressed vids end in lick.original.avi
            vids.append(f)

print('Number of videos to be analyzed:', len(vids))
print('Results will be saved at', path_sorted)

vid_paths = [os.path.join(vid_dir, v) for v in vids]
output_aliased = [os.path.join(path_sorted, v.replace(".avi",".dlc.h5")) for v in vids]
if 1:
    dlc.analyze_videos(path_config, vid_paths, videotype='avi', destfolder=path_sorted)

    for v,a in zip(vids, output_aliased):
        # Get list of all files only in the given directory
        list_of_files = filter( os.path.isfile,
                                glob.glob(os.path.join(path_sorted, v.replace(".avi","")) + '*.h5') )
        # Sort list of files based on last modification time in ascending order
        list_of_files = sorted( list_of_files,
                                key = os.path.getmtime)

        os.system(f"ln -s {list_of_files[-1]} {a}")

if 0:
    dlc.create_labeled_video(path_config, vid_paths, videotype='avi', destfolder=path_sorted)

if 0:
    # identify "bad" frames and save in training set
    dlc.extract_outlier_frames(path_config, vid_paths, destfolder=path_sorted, automatic=True)

    # gui to relabel the bad frames
    dlc.refine_labels(path_config)

if 0:
    dlc.merge_datasets(path_config)
    dlc.create_training_dataset(path_config, net_type='resnet_50', augmenter_type='imgaug')

if 0:
    # before running, update pose_cfg.yaml to use last snapshot from previous iteration as initial condition
    # (rather than starting over from visnet)
    dlc.train_network(path_config, shuffle=1, displayiters=100)
