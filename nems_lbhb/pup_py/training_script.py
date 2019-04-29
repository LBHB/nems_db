import numpy as np
import os
import sys
from keras.applications.densenet import DenseNet201
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import keras
import nems
import nems.db as nd
import datetime
sys.path.append('/auto/users/hellerc/code/nems/nems_db/nems_lbhb/pup_py/')
import keras_classes as kc

import logging
log = logging.getLogger(__name__)

try:
    import nems.db as nd
    db_exists = True
except Exception as e:
    # If there's an error import nems.db, probably missing database
    # dependencies. So keep going but don't do any database stuff.
    print("Problem importing nems.db, can't update tQueue")
    print(e)
    db_exists = False

if __name__ == '__main__':

    if 'QUEUEID' in os.environ:
        queueid = os.environ['QUEUEID']
        nems.utils.progress_fun = nd.update_job_tick

    else:
        queueid = 0

    if queueid:
        log.info("Starting QUEUEID={}".format(queueid))
        nd.update_job_start(queueid)

    # project directory
    project_dir = '/auto/data/nems_db/pup_py/'
    # data path
    path = '/auto/data/nems_db/pup_py/training_data/'
    training_files = os.listdir(path)
    n_training_files = len(training_files)

    # get current date/time so that we can save the model results in the correct place
    dt = datetime.datetime.now().isoformat()
    os.system('mkdir {0}old_model_fits/{1}'.format('/auto/data/nems_db/pup_py/', dt))

    load_from_past = False
    # what iteration is this for the current model. Only matters if load_from_past = True
    nrun = 2
    update_lr = True
    # should train / val sets be randomly select (set control to False)
    controlled = False

    if load_from_past:
        model_to_load = '{0}default_trained_model.hdf5'.format('/auto/data/nems_db/pup_py/')
    else:
        model_to_load = None

    params = {
        'batch_size': 16,
        'image_dim': (224, 224),
        'n_parms': 5,
        'n_channels': 3,
        'shuffle': True,
        'augment_minibatches': True
        }

    # To make val/train specific to certain videos or not
    if controlled:
        # define train indexes
        train = [i[0] for i in np.argwhere([True if ('AMT004b08' not in i) else False for i in training_files])]

        # define validation indexes
        test = [i[0] for i in np.argwhere([True if ('AMT004b08' in i) else False for i in training_files])]

    else:
        train = np.random.choice(np.arange(0, n_training_files), int(n_training_files * 0.7), replace=False).tolist()
        test = [i for i in np.arange(0, n_training_files) if i not in train]

    partition = {
        'train': train,
        'validation': test
        }

    training_generator = kc.DataGenerator(partition['train'], **params)
    params['augment_minibatches'] = False
    validation_generator = kc.DataGenerator(partition['validation'], **params)

    if load_from_past:
        model = keras.models.load_model(model_to_load)

        clr = keras.backend.eval(model.optimizer.lr)
        print("current learning rate is {0}".format(clr))
        if update_lr:
            nlr = clr / 10
            model.compile(loss='mean_squared_error',
                          optimizer=keras.optimizers.Adam(lr=nlr))
        else:
            model.compile(loss='mean_squared_error',
                          optimizer='adam')
        print("new learning rate is {0}".format(keras.backend.eval(model.optimizer.lr)))

        callback = keras.callbacks.ModelCheckpoint(
            filepath='{0}old_model_fits/dt/weights.{epoch:02d}-{val_loss:.2f}.hdf5'.format(project_dir),
            monitor='val_loss', save_best_only=False)

        history = model.fit_generator(generator=training_generator,
                                      validation_data=validation_generator,
                                      callbacks=[callback],
                                      use_multiprocessing=True,
                                      workers=6, epochs=500)

        np.save('{0}old_model_fits/{1}/val_loss{0}'.format(project_dir, dt, nrun), np.array(history.history['val_loss']))
        np.save('{0}old_model_fits/{1}/train_loss{0}'.format(project_dir, dt, nrun), np.array(history.history['loss']))

    else:

        # define keras model
        base_model = DenseNet201(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        x = base_model.output
        # add global average pooling layer
        x = GlobalAveragePooling2D()(x)
        # add output layer
        predictions = Dense(5, activation='linear')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        nlayers = len(model.layers)

        for i in range(nlayers-100):
            if type(model.layers[i]) == keras.layers.BatchNormalization:
                pass
            else:
                model.layers[i].trainable = False

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=0.001))

        # define callback to save the best weights based on val
        callback = keras.callbacks.ModelCheckpoint(filepath=project_dir + 'old_model_fits/' + dt + '/' +
                                                   'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                   monitor='val_loss', save_best_only=False)

        # fit keras model
        history = model.fit_generator(generator=training_generator,
                                      validation_data=validation_generator,
                                      callbacks=[callback],
                                      use_multiprocessing=True,
                                      workers=6, epochs=500)

        np.save('{0}old_model_fits/{1}/val_loss'.format(project_dir, dt), np.array(history.history['val_loss']))
        np.save('{0}old_model_fits/{1}/train_loss'.format(project_dir, dt), np.array(history.history['loss']))
        print('finished fit')

        # find the new "best fit" and save this as the default trained model
        files = os.listdir("{0}old_model_fits/{1}/".format(project_dir, dt))
        vals = [0]
        for i, f in enumerate(files):
            if "weights" not in f:
                pass
            else:
                if i == 0:
                    val = np.float(f.split('-')[-1].split('.')[0])
                    mod = f
                else:
                    if np.float(f.split('-')[-1].split('.')[0]) < val:
                        mod = f

        default_name = project_dir + 'default_trained_model.hdf5'
        old_default = project_dir + 'default_trained_model_{0}.hdf5'.format(dt)
        
        # rename the old default model with the date on which it was replaced!
        if os.path.isfile(default_name):
            log.info("replacing old default model with new fit. Renaming previous default to {0}".format(old_default))
            os.system("mv {0} {1}".format(default_name, old_default))
        
        # Now save the new current default model            
        os.system("cp {0} {1}".format("{0}old_model_fits/{1}/{2}".format(project_dir, dt, mod), default_name))

        if queueid:
            nd.update_job_complete(queueid)
