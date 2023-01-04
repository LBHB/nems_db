"""
process face video in dlc, ripped off of pupil_fit_script

very helpful tutorial/notes on refinement
https://guillermohidalgogadea.com/openlabnotebook/refining-your-dlc-model/

run in deeplabcut environment (tensorflow 2.4, cuda 11.0, etc) that
also has NEMS and nems_db installed

syntax:
   face_fit_script <avipath>/<avibase>.avi> [<dlcmodelconfig filepath>]

if dlcmodelconfig not provided, use default from nems_dlc_settings.py

results saved in <avipath>/sorted/<avibase>.dlc.h5
"""

import os
import glob
import sys

import deeplabcut as dlc
import numpy as np

import nems0.db as nd
import nems0

import nems_lbhb.motor.nems_dlc_settings as ds

import logging
log = logging.getLogger(__name__)

try:
    import nems0.db as nd
    db_exists = True
except Exception as e:
    # If there's an error import nems0.db, probably missing database
    # dependencies. So keep going but don't do any database stuff.
    print("Problem importing nems0.db, can't update tQueue")
    print(e)
    db_exists = False

if __name__ == '__main__':

    if 'QUEUEID' in os.environ:
        import nems0.utils
        queueid = int(os.environ['QUEUEID'])
        nems0.utils.progress_fun = nd.update_job_tick
    else:
        queueid = 0

    if db_exists & queueid>0:
        log.info("Starting QUEUEID={}".format(queueid))
        nd.update_job_start(queueid)

    # figure out filenames and paths to process
    video_file = sys.argv[1]
    create_labeled_video = False

    # figure out action(s) to perform
    if len(sys.argv) > 3:
        action_list = sys.argv[2].split(",")
    else:
        action_list = ['fit','summary']

    # location of DLC model
    # change if using a different DLC model
    if len(sys.argv) > 3:
        path_config = sys.argv[3]
    else:
        path_config = ds.DEFAULT_DLC_MODEL

    video_base = os.path.basename(video_file)
    if video_base.startswith('recording'):
        # create softlink in parent dir
        video_path = os.path.dirname(video_file)
        old_video_file = video_file
        video_file = video_path+'.avi'
        if os.path.exists(video_file):
            pass
        else:
            os.symlink(old_video_file, video_file)
    elif ~os.path.exists(video_file):
        # see if this is expected softlink
        orig_guess = video_file.replace('.avi','/recording.avi')
        if os.path.exists(orig_guess):
            os.symlink(orig_guess, video_file)

    vid_dir, base_file = os.path.split(video_file)
    animal_path, penname = os.path.split(vid_dir)
    daq_path, animal = os.path.split(animal_path)
    path_sorted = os.path.join(vid_dir, 'sorted')

    vids = [base_file]
    if not os.path.exists(path_sorted):
        os.makedirs(path_sorted)
        log.info(f'Created sorted directory in {vid_dir}')

    log.info(f'Number of videos to be analyzed: {len(vids)}')
    log.info(f'Results will be saved in: {path_sorted}')

    vid_paths = [os.path.join(vid_dir, v) for v in vids]
    output_aliased = [os.path.join(path_sorted, v.replace(".avi",".dlc.h5")) for v in vids]

    if 'fit' in action_list:
        dlc.analyze_videos(path_config, vid_paths, videotype='avi', destfolder=path_sorted)

        for v, a in zip(vids, output_aliased):
            # Get list of all files only in the given directory
            list_of_files = filter( os.path.isfile,
                                    glob.glob(os.path.join(path_sorted, v.replace(".avi","")) + '*.h5') )
            # Sort list of files based on last modification time in ascending order
            list_of_files = sorted( list_of_files,
                                    key = os.path.getmtime)

            os.system(f"ln -s {list_of_files[-1]} {a}")

    if 'video' in action_list:
        log.info('Creating labeled video')
        dlc.create_labeled_video(path_config, vid_paths, videotype='avi', destfolder=path_sorted)

    if 'summary' in action_list:
        from nems_lbhb.motor import face_tools

        fig=face_tools.summary_plot(vid_paths)

    if 'refine' in action_list:
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

    log.info(f"face_fit_script complete db_exists={db_exists} qid={queueid}")

    if db_exists & (queueid > 0):
        log.info('Marking job complete in celldb')
        nd.update_job_complete(queueid)


