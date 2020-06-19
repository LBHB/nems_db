"""
Path settings for where to store training data and model results
"""
import os

# Directory where raw video data is stored
ROOT_VIDEO_DIRECTORY = '/auto/data/daq/'

# Directories for saving model results etc.
ROOT_DIRECTORY = '/auto/users/hellerc/pup_py_testing/' #'/auto/data/nems_db/pup_py/'
TRAIN_DATA_PATH = os.path.join(ROOT_DIRECTORY, 'training_data/')
TMP_SAVE = os.path.join(ROOT_DIRECTORY, 'tmp/') 