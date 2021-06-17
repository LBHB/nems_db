"""
Path settings for where to store training data and model results
"""
import os

ROOT_VIDEO_DIRECTORY = '/auto/data/daq/'

ROOT_DIRECTORY = '/auto/data/nems_db/pup_dev/'
TRAIN_DATA_PATH = os.path.join(ROOT_DIRECTORY, 'training_data/')
TMP_SAVE = os.path.join(ROOT_DIRECTORY, 'tmp/')
TMP_TRAIN = os.path.join(ROOT_DIRECTORY, 'tmp_train/')