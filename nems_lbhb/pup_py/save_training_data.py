from nems_lbhb.pup_py.utils import loadmat
import av
import pims
import numpy as np
import pickle

animal = 'Shaggyink'
site = 'SIK015'
vid_name = 'SIK015c09_p_TON.mj2'
mat_struct = 'SIK015c09_p_TON.pup.mat'
buffer = 10
threshold_max = 60
n_frames = 50

path = '/auto/data/daq/{0}/{1}/'.format(animal, site)

saved_train_data = '/auto/data/nems_db/pup_py/training_data/{0}'.format(vid_name[:-4])

video = path + vid_name

fit = loadmat(path + mat_struct)

# open video
v = pims.PyAVReaderIndexed(video)

# figure out how many frames in video
nframes = len(v)

# reopen the video with pyav
v = av.open(video)

frames = np.random.choice(np.arange(0, nframes), n_frames)
frames = np.sort(frames)
max_frame = np.max(frames)

# read in the first frame
for i, frame in enumerate(v.decode()):

    if i in frames:
        print('frame: {0}'.format(i))
        frame1 = np.asarray(frame.to_image().convert('LA'))
        frame1 = frame1[:, :-10, :]
        frame1 = frame1 - np.min(frame1)
        default = fit['pupil_data']['params']['default']
        a = fit['pupil_data']['results'][default]['ellipse'][i]

        if a.long_axis == 0:
            # set default value if no ellipse was fit for this frame
            a.a = 1
            a.b = 1
            a.X0 = 1
            a.X0_in = 1
            a.X0 = 1
            a.Y0 = 1
            a.Y0_in = 1
            a.long_axis = a.b * 2
            a.short_axis = a.a * 2
            a.phi = 0

        results_dict = {'frame': frame1[:, :, 0],
                        'ellipse_zack': {
                            'a': a.a,
                            'b': a.b,
                            'X0_in': a.X0_in,
                            'Y0_in': a.Y0_in,
                            'phi': a.phi
                        }
        }

        with open(saved_train_data+'{0}.pickle'.format(i), 'wb') as fp:
            pickle.dump(results_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    elif i > max_frame:
        break

    elif (i != max_frame) | (i not in frames):
        continue
