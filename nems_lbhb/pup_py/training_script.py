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
import nems_db
nems_db_path = nems_db.__path__[0]
sys.path.append(os.path.join(nems_db_path, 'nems_lbhb/pup_py/'))
import keras_classes as kc
import pupil_settings as ps

# set up tensorflow session so that it is not "greedy"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)


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
   

    # check for sys arguments to determine how to fit model / where to save results
    if len(sys.argv) > 1:
        # get the animal / animal video key for figuring out which videos to use for training
        animal_name = sys.argv[2]
        video_code = sys.argv[3]
        # project directory
        project_dir = ps.ROOT_DIRECTORY  #'/auto/data/nems_db/pup_py/'
        # data path
        path = ps.TRAIN_DATA_PATH  #'/auto/data/nems_db/pup_py/training_data/'
        training_files = [t for t in os.listdir(path) if video_code in t]
        n_training_files = len(training_files)
        training_epochs = 500  # used to be 500. Make this a user def param?

        # get current date/time so that we can save the model results in the correct place
        dt = datetime.datetime.now().isoformat()
        this_model_directory = 'animal_specific_fits/{}'.format(animal_name)
        
        if os.path.isdir('{0}{1}/'.format('/auto/data/nems_db/pup_py/', this_model_directory)):
            os.system('mkdir {0}{1}/{2}'.format('/auto/data/nems_db/pup_py/', this_model_directory, dt))
        else:
            os.system('mkdir {0}{1}/'.format('/auto/data/nems_db/pup_py/', this_model_directory))
            os.system('mkdir {0}{1}/{2}'.format('/auto/data/nems_db/pup_py/', this_model_directory, dt))

    else:
        video_code = None
        # project directory
        project_dir = ps.ROOT_DIRECTORY  #'/auto/data/nems_db/pup_py/'
        # data path
        path = ps.TRAIN_DATA_PATH  #'/auto/data/nems_db/pup_py/training_data/'
        training_files = os.listdir(path)
        n_training_files = len(training_files)
        training_epochs = 250  # used to be 500. Make this a user def param?

        # get current date/time so that we can save the model results in the correct place
        dt = datetime.datetime.now().isoformat()
        this_model_directory = 'old_model_fits'
        os.system('mkdir {0}{1}/{2}'.format('/auto/data/nems_db/pup_py/', this_model_directory, dt))

    load_from_past = False
    # what iteration is this for the current model. Only matters if load_from_past = True
    nrun = 2
    update_lr = True
    # should train / val sets be randomly select (set control to False)
    controlled = False

    if load_from_past:
        model_to_load = os.path.join(ps.ROOT_DIRECTORY, 'default_trained_model.hdf5')  
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
            filepath='{0}{1}/dt/weights.{epoch:02d}-{val_loss:.2f}.hdf5'.format(project_dir, this_model_directory),
            monitor='val_loss', save_best_only=False)

        history = model.fit_generator(generator=training_generator,
                                      validation_data=validation_generator,
                                      callbacks=[callback],
                                      use_multiprocessing=True,
                                      workers=6, epochs=training_epochs)

        np.save('{0}{1}/{2}/val_loss{3}'.format(project_dir, this_model_directory, dt, nrun), np.array(history.history['val_loss']))
        np.save('{0}{1}/{2}/train_loss{3}'.format(project_dir, this_model_directory, dt, nrun), np.array(history.history['loss']))

    else:

        # define keras model
        base_model = DenseNet201(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        x = base_model.output
        # add global average pooling layer
        x = GlobalAveragePooling2D()(x)
        # add output layer
        #predictions = Dense(5, activation='linear')(x)
        # 5 ellipse params + 8 new params for the eyelid keypoints
        predictions = Dense(13, activation='linear')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        nlayers = len(model.layers)

        for i in range(nlayers-100):
            if type(model.layers[i]) == keras.layers.BatchNormalization:
                pass
            else:
                model.layers[i].trainable = False

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=0.001))

        # define callback to save the best weights based on val
        callback = keras.callbacks.ModelCheckpoint(filepath=project_dir + this_model_directory+'/' + dt + '/' +
                                                   'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                   monitor='val_loss', save_best_only=False)

        # fit keras model
        history = model.fit_generator(generator=training_generator,
                                      validation_data=validation_generator,
                                      callbacks=[callback],
                                      use_multiprocessing=True,
                                      workers=6, epochs=training_epochs)

        np.save('{0}{1}/{2}/val_loss'.format(project_dir, this_model_directory, dt), np.array(history.history['val_loss']))
        np.save('{0}{1}/{2}/train_loss'.format(project_dir, this_model_directory, dt), np.array(history.history['loss']))
        log.info('finished fit')

        # find the new "best fit" and save this as the default trained model
        files = os.listdir("{0}{1}/{2}/".format(project_dir, this_model_directory, dt))
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
                        val = np.float(f.split('-')[-1].split('.')[0])
                        mod = f

        if video_code is not None:
            try:
                old_date = os.listdir(project_dir + this_model_directory + '/default_trained_model/')[0]
                name = os.listdir(project_dir + this_model_directory + '/default_trained_model/{0}'.format(old_date))[0]
                default_name = project_dir + this_model_directory + '/default_trained_model/{0}/{1}'.format(old_date, name)
            except FileNotFoundError:
                os.system('mkdir {0}{1}/default_trained_model/'.format('/auto/data/nems_db/pup_py/', this_model_directory))
                default_name = None # no default exists yet
            except IndexError:
                default_name = None
        else:
            old_date = os.listdir(project_dir + 'default_trained_model/')[0]
            name = os.listdir(project_dir + 'default_trained_model/{0}'.format(old_date))[0]
            default_name = project_dir + 'default_trained_model/{0}/{1}'.format(old_date, name)

        # delete the current model from defaults (it is still saved in the
        # parent (probably "old_model_fits") folder under the date it was fit on), along with
        # the training data from that date
        if default_name is not None:
            if os.path.isfile(default_name):
                log.info("replacing old default model with new fit...")
                if video_code is not None:
                    os.system("rm -r {0}".format(project_dir + this_model_directory + '/default_trained_model/' + old_date))
                else:
                    os.system("rm -r {0}".format(project_dir + 'default_trained_model/' + old_date))

        # Now save the new best model as the current default model
        backup_loc = "{0}{1}/{2}/{3}".format(project_dir, this_model_directory, dt, mod)
        if video_code is not None:
            default_loc = "{0}/default_trained_model/{1}/{2} ".format(project_dir + this_model_directory, dt, mod)
            direc = "{0}/default_trained_model/{1}/".format(project_dir + this_model_directory, dt)
        else:
            default_loc = "{0}default_trained_model/{1}/{2} ".format(project_dir, dt, mod)
            direc = "{0}default_trained_model/{1}/".format(project_dir, dt)
        os.system("mkdir {}".format(direc))
        os.system("cp {0} {1}".format(backup_loc, default_loc))

        # purge all non-saved model weights
        log.info("Purging all other 'non-optimal' model fits...")
        for i, f in enumerate(files):
            if f != mod:
                os.system("rm {0}{1}/{2}/{3}".format(project_dir, this_model_directory, dt, f))

        # Finally, copy the current training data into this directory as well
        log.info("Copying training data into model fit folder...")
        if video_code is not None:
            os.system("cp -r {0}training_data/{1} {2}{3}/{4}/".format(project_dir, video_code+'*', project_dir, this_model_directory, dt))
        else:
            os.system("cp -r {0}training_data/ {1}old_model_fits/{2}/".format(project_dir, project_dir, dt))

        if queueid:
            nd.update_job_complete(queueid)
