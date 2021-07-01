import pickle
import numpy as np
import os
import sys
import nems_db
nems_db_path = nems_db.__file__.split('/nems_db/__init__.py')[0]
sys.path.append(os.path.join(nems_db_path, 'nems_lbhb/pup_py/'))
import pupil_settings as ps
import nems_lbhb.pup_py.utils as ut
# load all training video frames so that we can get the mean / sd of each ellipse paramter.
# this way we can z-score the paramters so that they're evenly weighted during the fit


def get_batch_norm_params(species):
    training_files = os.listdir(os.path.join(ps.ROOT_DIRECTORY, species, 'training_data/'))
    y = np.zeros((len(training_files), 13))
    for i, f in enumerate(training_files):
        fp = open(os.path.join(ps.ROOT_DIRECTORY, species, 'training_data', f), "rb")
        current_frame = pickle.load(fp)

        scale_fact, im = ut.resize(current_frame['frame'], size=(224, 224))
        
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

    # now, get the normalization factors for each parm in format:
    # key: (mean, sd)
    keys = [
        'Y0_in',
        'X0_in',
        'b',
        'a',
        'phi',
        'eyelid_left_x',
        'eyelid_left_y',
        'eyelid_top_x',
        'eyelid_top_y',
        'eyelid_right_x',
        'eyelid_right_y',
        'eyelid_bottom_x',
        'eyelid_bottom_y'
    ]
    NORM_FACTORS = {}
    for i, k in enumerate(keys):
        m = y[:, i].mean()
        sd = y[:, i].std()
        ma = y[:, i].max()
        NORM_FACTORS[k] = (m, sd, ma)
    return NORM_FACTORS