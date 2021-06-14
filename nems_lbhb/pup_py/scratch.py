'''
Helper to manipulate saved training data (if necessary)
'''
import os
import sys
import nems_db
nems_db_path = nems_db.__path__[0]
sys.path.append(os.path.join(nems_db_path, 'nems_lbhb/pup_py/'))
import nems_lbhb.pup_py.pupil_settings as ps
import pickle
import numpy as np
# realized we need to reorder the key points for the eyelid
# do this here, rather than "resaving" all the video frames
frames = os.listdir(ps.TRAIN_DATA_PATH)
for frame in frames:
    filename = os.path.join(ps.TRAIN_DATA_PATH, frame)
    with open(f'{filename}', 'rb') as fp:
            frame_data = pickle.load(fp)
            fp.close()
    edgepoints = [
        [frame_data['ellipse_zack']['eyelid_left_x'], frame_data['ellipse_zack']['eyelid_left_y']],
        [frame_data['ellipse_zack']['eyelid_top_x'], frame_data['ellipse_zack']['eyelid_top_y']],
        [frame_data['ellipse_zack']['eyelid_right_x'], frame_data['ellipse_zack']['eyelid_right_y']],
        [frame_data['ellipse_zack']['eyelid_bottom_x'], frame_data['ellipse_zack']['eyelid_bottom_y']]
    ]
    xlocs = np.array(edgepoints)[:,0]
    sidx = np.argsort(xlocs)
    left = sidx[0]
    right = sidx[3]
    top = sidx[1]
    bottom = sidx[2]
    if edgepoints[top][1]>edgepoints[bottom][1]:
        pass
    else:
        top = sidx[2]
        bottom = sidx[1]
    edgepoints = np.array(edgepoints)
    edgepoints = [edgepoints[left], edgepoints[top], edgepoints[right], edgepoints[bottom]]

    newdata = frame_data.copy()
    newdata['ellipse_zack']['eyelid_left_x'] = edgepoints[0][0]
    newdata['ellipse_zack']['eyelid_left_y'] = edgepoints[0][1]
    newdata['ellipse_zack']['eyelid_top_x'] = edgepoints[1][0]
    newdata['ellipse_zack']['eyelid_top_y'] = edgepoints[1][1]
    newdata['ellipse_zack']['eyelid_right_x'] = edgepoints[2][0]
    newdata['ellipse_zack']['eyelid_right_y'] = edgepoints[2][1]
    newdata['ellipse_zack']['eyelid_bottom_x'] = edgepoints[3][0]
    newdata['ellipse_zack']['eyelid_bottom_y'] = edgepoints[3][1]

    with open(f"{filename}", 'wb') as fp:
        pickle.dump(newdata, fp, protocol=pickle.HIGHEST_PROTOCOL)

