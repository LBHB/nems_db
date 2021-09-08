"""
Path settings for where to store training data and model results
"""
import os

try:
    import pupil_settings_local as psl
except ImportError:
    psl = None

if hasattr(psl, 'ROOT_VIDEO_DIRECTORY'):
    ROOT_VIDEO_DIRECTORY = psl.ROOT_VIDEO_DIRECTORY
else:
    ROOT_VIDEO_DIRECTORY = '/auto/data/daq/'

if hasattr(psl, 'ROOT_DIRECTORY'):
    ROOT_DIRECTORY = psl.ROOT_DIRECTORY
else:
    ROOT_DIRECTORY = '/auto/data/nems_db/pup_dev/'

if hasattr(psl, 'TRAIN_DATA_PATH'):
    TRAIN_DATA_PATH = psl.TRAIN_DATA_PATH
else:
    TRAIN_DATA_PATH = os.path.join(ROOT_DIRECTORY, 'training_data/')

if hasattr(psl, 'TMP_SAVE'):
    TMP_SAVE = psl.TMP_SAVE
else:
    TMP_SAVE = os.path.join(ROOT_DIRECTORY, 'tmp/')

if hasattr(psl, 'TMP_TRAIN'):
    TMP_TRAIN = psl.TMP_TRAIN
else:
    TMP_TRAIN = os.path.join(ROOT_DIRECTORY, 'tmp_train/')

