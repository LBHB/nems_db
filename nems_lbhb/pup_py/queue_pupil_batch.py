"""
Simple helper function to batch queue pupil jobs
"""

import nems.db as nd
import nems_db
import sys
import os

script_path = os.path.split(os.path.split(nems_db.__file__)[0])[0]
script_path = os.path.join(script_path, 'nems_lbhb', 'pup_py', 'pupil_fit_script.py')

def queue_pupil_jobs(pupilfiles, modeldate='Current', python_path=None, username='nems', force_rerun=True):

    if python_path is None:
        python_path = sys.executable  # path to python (can be set manually, but defaults to active python running

    for fn in pupilfiles:
        # add job to queue
        nd.add_job_to_queue([fn, modeldate], note="Pupil Job: {}".format(fn),
                            executable_path=py_path, user=username,
                            force_rerun=force_rerun, script_path=script_path, GPU_job=1)