#run in deeplabcut environment (tensorflow 2.4, cuda 11.0, etc)
import deeplabcut as dlc
import os
import glob
import nems0.db as db

def dlc2nems(siteid=None, vids=None, suffix=".lick.avi",
             site_path="/auto/data/daq/Clathrus/CLT011/",
             path_config='/auto/data/dlc/multivid-CLL-2022-01-14/config.yaml'
             ):

    #change path_config if using a different DLC model
    if siteid is not None:
        dfiles = db.pd_query(f"SELECT * FROM gDataRaw WHERE not(bad) AND cellid='{siteid}'")
        site_path = dfiles.at[0,'resppath']
    elif site_path is None:
        raise ValueError("siteid or site_path required.")

    if vids is None:
        vids = []
        # get list of all videos in site folder if no vids given as input
        for f in os.listdir(site_path):
            if f.endswith(suffix): #non-compressed vids end in lick.original.avi
                vids.append(f)

    path_sorted = os.path.join(site_path, 'sorted')

    if not os.path.exists(path_sorted):
        os.makedirs(path_sorted)
        print('Created sorted directory in', site_path)

    print('Number of videos to be analyzed:', len(vids))
    print('Results will be saved in:', path_sorted)

    vid_paths = [os.path.join(site_path, v) for v in vids]
    output_aliased = [os.path.join(path_sorted, v.replace(".avi",".dlc.h5")) for v in vids]

    if 0:
        # STEP 1. Train DNN
        # before running, update pose_cfg.yaml to use last snapshot from previous iteration as initial condition
        # (rather than starting over from visnet)
        dlc.train_network(path_config, shuffle=1, displayiters=500)

    if 1:
        # STEP 2. extract feature values from video
        dlc.analyze_videos(path_config, vid_paths, videotype='avi', destfolder=path_sorted)

        for v,a in zip(vids, output_aliased):
            # Get list of all files only in the given directory
            list_of_files = filter( os.path.isfile,
                                    glob.glob(os.path.join(path_sorted, v.replace(".avi","")) + '*.h5') )
            # Sort list of files based on last modification time in ascending order
            list_of_files = sorted( list_of_files,
                                    key = os.path.getmtime)
            if not os.path.exists(a):
                os.system(f"ln -s {list_of_files[-1]} {a}")

    if 0:
        # STEP 3. opitonal, generate a video with features labeled
        dlc.create_labeled_video(path_config, vid_paths, videotype='avi', destfolder=path_sorted)

    if 0:
        # STEP 4. evaluate and decide if necessary to refine
        # identify "bad" frames and save in training set
        dlc.extract_outlier_frames(path_config, vid_paths, destfolder=path_sorted, automatic=True)

        # gui to relabel the bad frames
        dlc.refine_labels(path_config)

    if 0:
        # STEP 5. add the new labels to a new training set, go back to STEP 1 to refine the model
        dlc.merge_datasets(path_config)
        dlc.create_training_dataset(path_config, net_type='resnet_50', augmenter_type='imgaug')

