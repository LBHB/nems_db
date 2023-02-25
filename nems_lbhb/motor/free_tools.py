import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter

from nems0 import db
import nems0.epoch as ep
import nems0.plots.api as nplt
from nems0.utils import smooth
from nems_lbhb.xform_wrappers import generate_recording_uri
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems_lbhb.plots import plot_waveforms_64D
from nems_lbhb.preprocessing import impute_multi
from nems_lbhb import baphy_io

def compute_d_theta(dlc_data, ref_x0y0=None, smooth_win=1.5, fs=1,
                    egocentric_velocity=False, verbose=False):
    """
    use nose and headpost position from DLC to compute distance and angle
    to a (sound) source. units of d preserved. pixels?
    :param dlc_data: np.array
    N x T matrix of DLC signals. Assume
        dlc_data[0:2, :] = (x,y) position of nose
        dlc_data[2:4, :] = (x,y) position of headpost (head center)
    :param imput_missing: bool
        if True fill in missing DLC values first -- NOT SUPPORTED
    :param ref_x0y0: Tuple
        speaker position. default =(600,-500)
    :param smooth_win: float
        SD of Gaussian smoothing prior to velocity computation
    :param fs: int
        sampling rate so vel values are in sec^-1 units (default 1)
    :return:  d, theta, dvel,  theta_vel, d_fwd, d_lat: 1XT np.arrays of distance and angle (CW) from
        sound source and their derivatives
        d_fwd, d_lat = (nose-ward, right-ward) linear velocity
    """
    if ref_x0y0 is None:
        # spout position
        #x0, y0 = 470, 90

        # speaker position (approx)
        x0, y0 = 600, -500
    else:
        x0, y0 = ref_x0y0

    d = np.sqrt((dlc_data[[0], :] - x0) ** 2 + (dlc_data[[1], :] - y0) ** 2)

    # head to speaker angle CW from y axis toward speaker (negative y)
    mx = (dlc_data[[2], :] + dlc_data[[0], :]) / 2
    my = (dlc_data[[3], :] + dlc_data[[1], :]) / 2
    dx0 = mx - x0
    dy0 = my - y0
    theta0 = np.arctan2(dy0, dx0)
    theta0 -= np.pi / 2
    # theta0[theta0 < -np.pi] = (theta0[theta0 < -np.pi]+2*np.pi)

    # head rotation CW from y axis pointing toward speaker (negative y)
    dx = dlc_data[[2], :] - dlc_data[[0], :]
    dy = dlc_data[[3], :] - dlc_data[[1], :]
    theta = np.arctan2(dy, dx)
    theta -= np.pi / 2
    # theta[theta < -np.pi] = (theta[theta < -np.pi] + 2*np.pi)

    # angle from speaker = head angle minus speaker angle
    theta = theta - theta0
    theta[theta < -np.pi] = (theta[theta < -np.pi] + 2 * np.pi)
    theta[theta > np.pi] = (theta[theta > np.pi] - 2 * np.pi)
    theta *= 180 / np.pi

    #v = np.concatenate([np.diff(dlc_data[0:2, :], axis=1),
    #                    np.zeros((2, 1))], axis=1)
    d_vel = np.concatenate([np.diff(smooth(d, smooth_win), axis=1), np.array([[0]])], axis=1) * fs
    theta_vel = np.concatenate([np.diff(smooth(theta,smooth_win), axis=1),np.array([[0]])], axis=1)*fs

    # egocentric velocity -- (fwd, lateral)
    # (deltax,deltay) nose-headpost
    v = np.concatenate([np.diff(smooth(dlc_data[0:2, :], smooth_win), axis=1),
                        np.zeros((2, 1))], axis=1) * fs

    dhead = dlc_data[0:2, :] - dlc_data[2:4, :]
    lenhead = np.sqrt(np.sum(dhead**2,axis=0,keepdims=True))
    u_head = dhead/lenhead
    d_fwd = np.sum(v * u_head, axis=0, keepdims=True)
    d_vec = d_fwd * u_head
    l_vec = v - d_vec
    u_lat = np.concatenate([u_head[[1],:], -u_head[[0],:]], axis=0)
    d_lat = np.sum(v * u_lat, axis=0, keepdims=True)

    if verbose:
        plt.close('all')
        plt.figure()
        t = 0
        for t in range(1000, 1100, 1):
            plt.plot(dlc_data[[2, 0], t], dlc_data[[3, 1], t], color='lightgray', lw=0.5)
            plt.plot(dlc_data[0, t], dlc_data[1, t], 'ko', markersize=2)
            plt.arrow(dlc_data[0, t], dlc_data[1, t], v[0, t], v[1, t],
                      color='r', width=0.005,
                      length_includes_head=True, head_width=0.01)
            plt.arrow(dlc_data[0, t], dlc_data[1, t], d_vec[0, t], d_vec[1, t],
                      color='g', width=0.005,
                      length_includes_head=True, head_width=0.01)
            plt.arrow(dlc_data[0, t], dlc_data[1, t], l_vec[0, t], l_vec[1, t],
                      color='b', width=0.005,
                      length_includes_head=True, head_width=0.01)
            # plt.plot([dlc_data[0,t],dlc_data[0,t]+d_vec[0,t]],[dlc_data[1,t],dlc_data[1,t]+d_vec[1,t]],'g-',lw=1)

        plt.gca().set_aspect('equal')
        # plt.plot(d[0,::20],theta[0,::20],'.')

    return d, theta, d_vel, theta_vel, d_fwd, d_lat

def dlc2dist(rec, rasterfs=1, norm=False, **d_theta_opts):
    """
    transform dlc signal into a new dist signal and add to recording
    :param rec:
    :return:
    """
    # get DLC data
    dlc_data = rec['dlc'][:, :]
    newrec = rec.copy()

    # fill in missing values where possible (when other values exist at that time)
    newrec = impute_multi(newrec, sig='dlc', empty_values=np.nan, norm=norm)['rec']
    dlc_data_imp = newrec['dlc'][:, :]
    rasterfs = newrec['dlc'].fs

    if d_theta_opts.get('verbose', False):
        f, ax = plt.subplots(4,1)

        for i, a in enumerate(ax):
            a.plot(dlc_data_imp[(i*2):((i+1)*2),2000:6000].T, color='lightgray')
            a.plot(dlc_data[(i*2):((i+1)*2),2000:6000].T)
            l = rec['dlc'].chans[i*2].split("_")[0]
            a.set_ylabel(l)
        ax[-1].set_xlabel('sample number')
        f.suptitle('Results of imputation (gray is imputed data)')

    d, theta, vel, rvel, d_fwd, d_lat = compute_d_theta(dlc_data_imp, fs=rasterfs, **d_theta_opts)

    dist = np.concatenate((d, theta, vel, rvel, d_fwd, d_lat), axis=0)
    newrec['dist'] = newrec['dlc']._modified_copy(
        data=dist, chans=['d', 'theta', 'v', 'v_theta', 'v_fwd', 'v_lat'])

    return newrec

def free_scatter_sum(rec):
    """
    summarize spatial and velocity distributions with scatter
    :param rec: recording with dist signal added
    :return: f: handle to new figure
    """
    f,ax = plt.subplots(1,4,figsize=(12,3))
    ax[0].plot(rec['dlc'][0,::20],500+rec['dlc'][1,::20],'.',markersize=2)
    ax[0].plot(rec['dlc'][2,::20],500+rec['dlc'][3,::20],'.',markersize=2)
    ax[0].invert_yaxis()
    ax[0].set_xlabel('X position (pixels)', fontsize=10)
    ax[0].set_ylabel('Y position (pixels)', fontsize=10)
    ax[0].legend(('front','back'), fontsize=10)

    ax[1].plot(rec['dist'][1,::20],rec['dist'][0,::20],'.',markersize=2)
    ax[1].set_xlabel('Angle from speaker (deg clockwise)', fontsize=10)
    ax[1].set_ylabel('Distance from speaker (pix)', fontsize=10)
    ax[1].invert_yaxis()

    ax[2].plot(rec['dist'][3,::20],rec['dist'][2,::20],'.',markersize=2)
    ax[2].set_xlabel('Rotational velocity (deg clockwise/sec)', fontsize=10)
    ax[2].set_ylabel('Velocity from speaker (pix/sec)', fontsize=10)
    ax[2].set_xlim([-100,100])
    ax[2].set_ylim([-100,100])
    ax[2].invert_yaxis()

    ax[3].plot(rec['dist'][5,::20],rec['dist'][4,::20],'.',markersize=2)
    ax[3].set_xlabel('Fwd velocity (pix/sec)', fontsize=10)
    ax[3].set_ylabel('Rightward velocity (pix/sec)', fontsize=10)
    ax[3].set_xlim([-150,150])
    ax[3].set_ylim([-150,150])
    plt.tight_layout()

    return f

