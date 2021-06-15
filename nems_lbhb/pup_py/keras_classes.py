import numpy as np
import os
import pickle
import keras
import nems_lbhb.pup_py.utils as ut

import nems_db
nems_db_path = nems_db.__path__[0]
import sys
sys.path.append(os.path.join(nems_db_path, 'nems_lbhb/pup_py/'))
import pupil_settings as ps

from batch_norm import NORM_FACTORS

# define global variables for data
path = ps.TRAIN_DATA_PATH  #'/auto/data/nems_db/pup_py/training_data/'
data_frames = os.listdir(path)

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, batch_size=32, image_dim=(224, 224), n_parms=13, n_channels=3, shuffle=True,
                 augment_minibatches=False):
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_parms = n_parms
        self.image_dim = image_dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augment_minibatches
        self.on_epoch_end()

    def on_epoch_end(self):
        'Update index after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, listIDs_temp):
        'Generate batch size samples of data'
        X = np.empty((self.batch_size,) + self.image_dim + (self.n_channels,))
        y = np.empty((self.batch_size, self.n_parms))

        for i, ID in enumerate(listIDs_temp):
            # load frame
            frame_file = data_frames[ID]
            with open(path+'{0}'.format(frame_file), 'rb') as fp:
                current_frame = pickle.load(fp)

            # load labels (ellipse params)
            Y0_in = current_frame['ellipse_zack']['Y0_in']
            X0_in = current_frame['ellipse_zack']['X0_in']
            long_axis = current_frame['ellipse_zack']['b'] * 2
            short_axis = current_frame['ellipse_zack']['a'] * 2
            phi = current_frame['ellipse_zack']['phi']
            # eyelid keypoints
            lx = current_frame['ellipse_zack']['eyelid_left_x']
            ly = current_frame['ellipse_zack']['eyelid_left_y']
            tx = current_frame['ellipse_zack']['eyelid_top_x']
            ty = current_frame['ellipse_zack']['eyelid_top_y']
            rx = current_frame['ellipse_zack']['eyelid_right_x']
            ry = current_frame['ellipse_zack']['eyelid_right_y']
            bx = current_frame['ellipse_zack']['eyelid_bottom_x']
            by = current_frame['ellipse_zack']['eyelid_bottom_y']
            y[i, :] = np.asarray([Y0_in, X0_in, long_axis, short_axis, phi,
                            lx, ly, tx, ty, rx, ry, bx, by])

            if self.augment:
                # randomly augment mini-batches by performing transformations on each image
                # see utils.augment for more detauls
                im, y[i, ] = ut.augment(current_frame['frame'], y[i])
                scale_fact, im = ut.resize(im)

            else:
                scale_fact, im = ut.resize(current_frame['frame'], size=self.image_dim)

            y[i, ] = np.asarray([y[i, 0] * scale_fact[1], y[i, 1] * scale_fact[0], y[i, 2] * scale_fact[1],
                                y[i, 3] * scale_fact[0], y[i, 4],
                                y[i, 5] * scale_fact[0],
                                y[i, 6] * scale_fact[1],
                                y[i, 7] * scale_fact[0],
                                y[i, 8] * scale_fact[1],
                                y[i, 9] * scale_fact[0],
                                y[i, 10] * scale_fact[1],
                                y[i, 11] * scale_fact[0],
                                y[i, 12] * scale_fact[1],
                                ])

            im = keras.applications.densenet.preprocess_input(im)

            # this stuff is still a WIP. Trying to normalize parameters for the fit so one isn't weighted more heavily than others            
            # normalize the params to live between 0 and 1
            for j, (_, v) in enumerate(NORM_FACTORS.items()):
                # these should be sorted in the same order as y, so no need to use the
                # dict keys in NORM_FACTORS
                y[i, j] = (y[i, j] / v[2]) #/ v[1]
            '''
            y[i, 0] = y[i, 0] / self.image_dim[1]
            y[i, 1] = y[i, 1] / self.image_dim[1]
            y[i, 2] = y[i, 2] / (self.image_dim[1] / 2)
            y[i, 3] = y[i, 3] / (self.image_dim[1] / 2)
            y[i, 4] = (y[i, 4] / (np.pi)) 
            y[i, 5:] = y[i, 5:] / self.image_dim[1]

            y[i, :] = y[i, :] * 100
            '''
            X[i, ] = np.tile(np.expand_dims(im, -1), [1, 1, 3])

        return X, y

    def __len__(self):
        'Denote the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[(index * self.batch_size):((index+1) * self.batch_size)]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

def train(model, epochs=1):
    raise DeprecationWarning("Don't *think* this is used for anything anymore... delete?")
    # data path
    path = 'training_data/data/'
    training_files = os.listdir(path)
    n_training_files = len(training_files)

    params = {
        'batch_size': 16,
        'image_dim': (224, 224),
        'n_parms': 5,
        'n_channels': 3,
        'shuffle': True,
    }

    # define train indexes
    train = [i[0] for i in np.argwhere([True if 'AMT004b' not in i else False for i in training_files])]

    # define validation indexes
    test = [i[0] for i in np.argwhere([True if 'AMT004b' in i else False for i in training_files])]

    partition = {
        'train': train,
        'validation': test
    }

    training_generator = DataGenerator(partition['train'], **params)
    validation_generator = DataGenerator(partition['validation'], **params)

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=6, epochs=epochs)

    return model
