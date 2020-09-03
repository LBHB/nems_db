"""
Simple helper function to batch queue pupil jobs
"""

import nems.db as nd
import nems_db
import sys
import os
import pickle
import numpy as np
import scipy.io 
import logging
log = logging.getLogger(__name__)

script_path = os.path.split(os.path.split(nems_db.__file__)[0])[0]
script_path = os.path.join(script_path, 'nems_lbhb', 'pup_py', 'pupil_fit_script.py')

def queue_pupil_jobs(pupilfiles, modeldate='Current', animal='All', python_path=None, username='nems', force_rerun=True):

    if python_path is None:
        python_path = sys.executable  # path to python (can be set manually, but defaults to active python running

    for fn in pupilfiles:
        # add job to queue
        nd.add_job_to_queue([fn, modeldate, animal], note="Pupil Job: {}".format(fn),
                            executable_path=python_path, user=username,
                            force_rerun=force_rerun, script_path=script_path, GPU_job=1)


def mark_complete(pupilfiles):
    """
    Save predictions for all files and update celldb to mark these analyses as complete.
    
    !!!!!!!!!!!!!!!! Take care using this!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    It's a good idea to first use pupil_browser to validate the results
    of the fit. However, if you collect many videos in a row and the hardware set up 
    doesn't change, it's likely you can just perform the pupil_browser QC on the first 
    video in the sequence and then batch process the rest of the videos using this function.
    """
    for vf in pupilfiles:
        video_name = os.path.splitext(os.path.split(vf)[-1])[0]

        fn = video_name + '.pickle'
        fn_mat = video_name + '.mat'
        fp = os.path.split(vf)[0]
        save_path = os.path.join(fp, 'sorted', fn)
        # for matlab loading
        mat_fn = os.path.join(fp, fn_mat)

        sorted_dir = os.path.split(save_path)[0]

        if os.path.isdir(sorted_dir) != True:
            # create sorted directory and force to be world writeable
            os.system("mkdir {}".format(sorted_dir))
            os.system("chmod a+w {}".format(sorted_dir))
            print("created new directory {0}".format(sorted_dir))
        else:
            pass

        try:
            # load predictions
            pred_path = os.path.join(sorted_dir, fn.replace('.pickle', '_pred.pickle'))
            with open(pred_path, 'rb') as fp:
                save_dict = pickle.load(fp)

            # No excluded frames options for batch saving
            save_dict['cnn']['excluded_frames'] = []

            x_diff = np.diff(save_dict['cnn']['x'])
            y_diff = np.diff(save_dict['cnn']['y'])
            d = np.sqrt((x_diff ** 2) + (y_diff ** 2))
            d[-1] = 0
            d = np.concatenate((d, np.zeros(1)))
            save_dict['cnn']['eyespeed'] = d
            with open(save_path, 'wb') as fp:
                    pickle.dump(save_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

            scipy.io.savemat(mat_fn, save_dict)

            # finally, update celldb to mark pupil as analyzed
            get_file1 = "SELECT eyecalfile from gDataRaw where eyecalfile='{0}'".format(self.raw_video)
            out1 = nd.pd_query(get_file1)
            if out1.shape[0]==0:
                # try the L:/ path
                og_video_path = self.raw_video.replace('/auto/data/daq/', 'L:/')
                sql = "UPDATE gDataRaw SET eyewin=2 WHERE eyecalfile='{}'".format(og_video_path)
            else:
                sql = "UPDATE gDataRaw SET eyewin=2 WHERE eyecalfile='{}'".format(self.raw_video)
            nd.sql_command(sql)

            log.info("Saved analysis successfully for {}".format(video_name))
        
        except:
            log.info("No saved pupil analysis {}".format(os.path.join(sorted_dir, fn.replace('.pickle', '_pred.pickle'))))
