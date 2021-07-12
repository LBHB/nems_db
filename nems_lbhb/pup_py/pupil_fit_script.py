import keras
import nems.db as nd
import nems_lbhb.pup_py.utils as ut
import numpy as np
import av
import pickle
import sys
import os
import nems

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)

import nems_db
nems_db_path = nems_db.__path__[0]
sys.path.append(os.path.join(nems_db_path, 'nems_lbhb/pup_py/'))
import pupil_settings as ps
from batch_norm import get_batch_norm_params

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

    # perform pupil fit
    video_file = sys.argv[1]
    modelname = sys.argv[2]
    species, animal = sys.argv[3].split('_')

    # load the keras model 
    project_dir = os.path.join(ps.ROOT_DIRECTORY, species+'/') 
    if (modelname == 'current') | (modelname == 'Current'):
        if (animal != '') & (animal != 'None') & (animal != 'All') & (animal != None):
            this_model_dir = 'animal_specific_fits/{}/'.format(animal)
            default_date = os.listdir(project_dir + this_model_dir + 'default_trained_model/')[0]
            name = os.listdir(project_dir + this_model_dir + 'default_trained_model/{0}'.format(default_date))[0]
            modelpath = project_dir + this_model_dir + 'default_trained_model/{0}/{1}'.format(default_date, name)
        else:
            default_date = os.listdir(project_dir + 'default_trained_model/')[0]
            name = os.listdir(project_dir + 'default_trained_model/{0}'.format(default_date))[0]
            modelpath = project_dir + 'default_trained_model/{0}/{1}'.format(default_date, name)
    else:
        # load an older model fit for this pupil video. Probably not used often, but nice 
        # to have the option...
        date = modelname
        datefolder = os.listdir(project_dir + 'old_model_fits/' + date)
        modelname = [m for m in datefolder if 'weights' in m][0]
        modelpath = project_dir + 'old_model_fits/{0}/{1}'.format(date, modelname)

    NORM_FACTORS = get_batch_norm_params(species)

    model = keras.models.load_model(modelpath)

    path = os.path.split(video_file)[0] 

    if os.path.isdir(path):
        pass
    else:
        raise ValueError("can't find pupil video")

    save_path = os.path.join(path, 'sorted')
    video = video_file
    results_name = os.path.split(video_file)[-1].split('.')[0] + '_pred.pickle'

    log.info("saving analysis to: {0}".format(os.path.join(save_path, results_name)))

    # define empty lists to hold params (cnn)
    a_cnn = []
    b_cnn = []
    x_cnn = []
    y_cnn = []
    phi_cnn = []
    lx=[];ly=[];tx=[];ty=[];rx=[];ry=[];bx=[];by=[]

    # reopen the video with pyav
    container = av.open(video)
    video_stream = [s for s in container.streams][0]

    for i, packet in enumerate(container.demux(video_stream)):
        if i % 1000 == 0:
            log.info("frame: {0}...".format(i))
            nd.update_job_tick()
        try:
            frame = packet.decode()[0]

            frame_ = np.asarray(frame.to_image().convert('LA'))
            frame_ = frame_[:, :-10, :]
            frame_ = frame_ - np.min(frame_)

            if frame_.shape[-1] > 1:
                frame_ = frame_[:, :, 0]

            # use the model to predict the ellipse params (cnn)

            # resize the image
            size = (224, 224)
            sf, im = ut.resize(frame_, size=size)

            # normalize the image
            im = keras.applications.densenet.preprocess_input(im)

            # get ellipse params
            ellipse_parms = model.predict(np.tile(im[np.newaxis, :, :, np.newaxis], [1, 1, 1, 3]))[0]

            # rescale parms correctly (based on the resizing and normalizing that was done for the fit)
            # undo normalization (WIP - CRH 1/30/19 -- update on 6/15/2021, using z-scores per param)
            for j, (_, v) in enumerate(NORM_FACTORS.items()):
                # these should be sorted in the same order as y, so no need to use the
                # dict keys in NORM_FACTORS
                ellipse_parms[j] = (ellipse_parms[j] * v[2]) #+ v[0]
            ellipse_parms /= 100
            '''
            ellipse_parms /= 100
            ellipse_parms[0] = ellipse_parms[0] * size[0]
            ellipse_parms[1] = ellipse_parms[1] * size[0]
            ellipse_parms[2] = ellipse_parms[2] * (size[0] / 2)
            ellipse_parms[3] = ellipse_parms[3] * (size[0] / 2)

            ellipse_parms[4] = ellipse_parms[4] * np.pi
            ellipse_parms[5:] = ellipse_parms[5:] * size[0] # eyelid keypoints
            '''

            # Finally, undo image scaling and save
            y_cnn.append(ellipse_parms[0] / sf[1])
            x_cnn.append(ellipse_parms[1] / sf[0])
            b_cnn.append(ellipse_parms[2] / sf[1] / 2)
            a_cnn.append(ellipse_parms[3] / sf[0] / 2)
            phi_cnn.append(ellipse_parms[4])
            lx.append(ellipse_parms[5] / sf[0])
            ly.append(ellipse_parms[6] / sf[1])
            tx.append(ellipse_parms[7] / sf[0])
            ty.append(ellipse_parms[8] / sf[1])
            rx.append(ellipse_parms[9] / sf[0])
            ry.append(ellipse_parms[10] / sf[1])
            bx.append(ellipse_parms[11] / sf[0])
            by.append(ellipse_parms[12] / sf[1])
        except:
            log.info("video decoding failed for frame {0}. Frame dropped? Pad with nans for now...".format(i))

            y_cnn.append(np.nan)
            x_cnn.append(np.nan)
            b_cnn.append(np.nan)
            a_cnn.append(np.nan)
            phi_cnn.append(np.nan)
            lx.append(np.nan)
            ly.append(np.nan)
            tx.append(np.nan)
            ty.append(np.nan)
            rx.append(np.nan)
            ry.append(np.nan)
            bx.append(np.nan)
            by.append(np.nan)
    results = {
        'cnn': {
            'a': np.array(a_cnn),
            'b': np.array(b_cnn),
            'x': np.array(x_cnn),
            'y': np.array(y_cnn),
            'phi': np.array(phi_cnn),
            'eyelid_left_x': lx,
            'eyelid_left_y': ly,
            'eyelid_top_x': tx,
            'eyelid_top_y': ty,
            'eyelid_right_x': rx,
            'eyelid_right_y': ry,
            'eyelid_bottom_x': bx,
            'eyelid_bottom_y': by
        },
        'cnn_modelpath': modelpath
    }

    if os.path.isdir(save_path) != True:
        os.system("mkdir {}".format(save_path))
        os.system("chmod a+w {}".format(save_path))
        print("created new directory {0}".format(save_path))

    # make sure the directory is writeable
    os.system("chmod a+w {}".format(save_path))

    save_file = os.path.join(save_path, results_name)

    # write the results
    with open(save_file, 'wb') as fp:
                pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)

    log.info("finished fit")

    if queueid:
            nd.update_job_complete(queueid)
