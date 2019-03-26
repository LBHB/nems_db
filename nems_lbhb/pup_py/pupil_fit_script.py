import keras
import utils as ut
import numpy as np
import av
import pickle
import sys
import os
import nems
import nems.db as nd

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
    animal = sys.argv[1]
    filename = sys.argv[2]

    # load the keras model (this is hardcoded rn but should be flexible at some point
    model = keras.models.load_model('/auto/data/nems_db/pup_py/default_trained_model.hdf5')

    path = '/auto/data/daq/{0}/{1}/'.format(animal, filename[:6])
    if os.path.isdir(path):
        pass
    else:
        try:
            path = '/auto/data/daq/{0}/training2019/{1}'.format(animal, filename+'.m')
            if os.path.isfile(path):
                path = '/auto/data/daq/{0}/training2019/'.format(animal)
            else:
                raise ValueError("can't find pupil video")
        except:
            path = '/auto/data/daq/{0}/training2018/{1}'.format(animal, filename+'.m')
            if os.path.isfile(path):
                path = '/auto/data/daq/{0}/training2018/'.format(animal)
            else:
                raise ValueError("can't find pupil video")

    save_path = path + 'sorted/'
    video = path + filename + '.mj2'

    # define empty lists to hold params (cnn)
    a_cnn = []
    b_cnn = []
    x_cnn = []
    y_cnn = []
    phi_cnn = []

    # reopen the video with pyav
    container = av.open(video)
    video_stream = [s for s in container.streams][0]

    for i, packet in enumerate(container.demux(video_stream)):
        if i % 1000 == 0:
            log.info("frame: {0}...".format(i))
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
            # undo normalization (WIP - CRH 1/30/19)
            # ellipse_parms[0] = ellipse_parms[0] / 100 * size[0]
            # ellipse_parms[1] = ellipse_parms[1] / 100 * size[0]
            # ellipse_parms[2] = ellipse_parms[2] / 100 * size[0]
            # ellipse_parms[3] = ellipse_parms[3] / 100 * size[0]

            # ellipse_parms[4] = ((ellipse_parms[4] / 100) - 0.4) * (2 * np.pi)

            # undo scaling and save
            y_cnn.append(ellipse_parms[0] / sf[1])
            x_cnn.append(ellipse_parms[1] / sf[0])
            b_cnn.append(ellipse_parms[2] / sf[1] / 2)
            a_cnn.append(ellipse_parms[3] / sf[0] / 2)
            phi_cnn.append(ellipse_parms[4])

        except:
            print("video decoding failed for frame {0}. Frame dropped? Pad with nans for now...".format(i))

            y_cnn.append(np.nan)
            x_cnn.append(np.nan)
            b_cnn.append(np.nan)
            a_cnn.append(np.nan)
            phi_cnn.append(np.nan)

    results = {
        'cnn': {
            'a': np.array(a_cnn),
            'b': np.array(b_cnn),
            'x': np.array(x_cnn),
            'y': np.array(y_cnn),
            'phi': np.array(phi_cnn)
        }
    }

    if os.path.isdir(save_path) != True:
        os.system("mkdir {}".format(sorted_dir))
        os.system("chmod a+w {}".format(sorted_dir))
        print("created new directory {0}".format(sorted_dir))

    save_file = save_path + filename + '_pred.pickle'

    # write the results
    with open(save_file, 'wb') as fp:
                pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print("finished fit")

    if queueid:
            nd.update_job_complete(queueid)
