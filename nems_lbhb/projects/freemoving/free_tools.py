import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import logging

from scipy import interpolate

from nems0.analysis.gammatone.filters import centre_freqs
from nems0.utils import smooth
from nems_lbhb.preprocessing import impute_multi
from nems_lbhb import runclass

log = logging.getLogger(__name__)

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
    theta *= -180 / np.pi

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

def dlc2dist(rec, rasterfs=1, norm=False, keep_dims=None, **d_theta_opts):
    """
    transform dlc signal into a new dist signal and add to recording
    :param rec:
    :return:
    """
    # get DLC data
    dlc_data = rec['dlc'][:, :]
    newrec = rec.copy()

    # fill in missing values where possible (when other values exist at that time)
    newrec = impute_multi(newrec, sig='dlc', empty_values=np.nan,
                          norm=norm, keep_dims=keep_dims)['rec']
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

def stim_filt_hrtf(rec, signal='stim', hrtf_format='az', smooth_win=2,
                   f_min=200, f_max=20000, channels=None, **ctx):

    # require (stacked) binaural stim
    if channels is None:
        channels = int(rec[signal].shape[0]/2)
    rasterfs = rec[signal].fs
    stimcount = int(rec[signal].shape[0]/channels)
    log.info(f"HRTF: {signal} is {channels} x {stimcount}")

    L0, R0, c, A = load_hrtf(format=hrtf_format, fmin=f_min, fmax=f_max, num_freqs=channels)

    # assume dlc has already been imputed and normalized to (0,1)
    if 'dlc' in rec.signals.keys():
        dlc_data_imp = rec['dlc'][:, :]
    else:
        T=rec['resp'].shape[1]
        dlc_data_imp = np.repeat(np.array([[0.5,0.0,0.5,0.1]]).T,T,axis=1)

    speaker1_x0y0 = 1.0, -0.8
    speaker2_x0y0 = 0.0, -0.8

    d1, theta1, vel, rvel, d_fwd, d_lat = compute_d_theta(
        dlc_data_imp, fs=rasterfs, smooth_win=smooth_win, ref_x0y0=speaker1_x0y0)
    d2, theta2, vel, rvel, d_fwd, d_lat = compute_d_theta(
        dlc_data_imp, fs=rasterfs, smooth_win=smooth_win, ref_x0y0=speaker2_x0y0)

    # fall-off over distance
    # -- currently hard-coded 5 db difference from front to back
    dist_atten = 6
    log.info(f"Imposing distance attenuation={dist_atten} dB ")
    gaind1 = -(d1 - 1) * dist_atten
    gaind2 = -(d2 - 1) * dist_atten

    f = interpolate.interp1d(A, R0, axis=1, fill_value="extrapolate")
    gainr1 = f(theta1)[:, 0, :] + gaind1
    gainr2 = f(theta2)[:, 0, :] + gaind2
    g = interpolate.interp1d(A, L0, axis=1, fill_value="extrapolate")
    gainl1 = g(theta1)[:, 0, :] + gaind1
    gainl2 = g(theta2)[:, 0, :] + gaind2

    s1 = rec[signal].as_continuous()[:channels, :]
    if stimcount>1:
        s2 = rec[signal].as_continuous()[channels:, :]
    else:
        s2 = np.zeros_like(s1)

    # dB = 10*log10(P2/P1)
    # so, to scale the gtgrams:
    #   P2 = 10^(dB/10) * P1
    #r12 = s1 * 10 ** (gainr1 / 10) + s2 * 10 ** (gainr2 / 10)
    #l12 = s1 * 10 ** (gainl1 / 10) + s2 * 10 ** (gainl2 / 10)
    r12 = np.sqrt((s1**2) * 10 ** (gainr1 / 10) + (s2**2) * 10 ** (gainr2 / 10))
    l12 = np.sqrt((s1**2) * 10 ** (gainl1 / 10) + (s2**2) * 10 ** (gainl2 / 10))

    binaural_stim = np.concatenate([r12,l12], axis=0)
    newrec = rec.copy()
    newrec[signal] = newrec[signal].rasterize()
    newrec[signal] = newrec[signal]._modified_copy(data=binaural_stim)
    newrec['disttheta'] = newrec[signal]._modified_copy(
        data=np.concatenate([d1,theta1,d2,theta2],axis=0),
        chans=['d1','theta1','d2','theta2'])
    return {'rec': newrec}


def load_hrtf(format='az', fmin=200, fmax=20000, num_freqs=18):
    """
    load HRFT and map to center frequencies of a gtgram fitlerbank
    TODO: support for elevation, cleaner HRTF
    :param format: str - has to be 'az' (default)
    :param fmin: default 200
    :param fmax: default 20000
    :param num_freqs: default 18
    :return: L0, R0, c, A -- tuple
            L0: Left ear HRTF,
            R0: Right ear HRTF,
            c: frequency corresponding to each row (axis = 0),
            A: azimuth corresponding to each column (axis =1)
    """

    c = np.sort(centre_freqs(fmax*2, num_freqs, fmin, fmax))
    #libpath = Path('/auto/users/svd/python/nems_db/nems_lbhb/projects/freemoving/')
    libpath = Path(os.path.dirname(__file__))
    if format == 'az':
        filepath = libpath / 'hrtf_az.csv'
        arr = np.loadtxt(filepath, delimiter=",", dtype=float)

        A = np.unique(arr[:, 0])
        F = np.unique(arr[:, 1])
        L = np.reshape(arr[:, 2], [len(A), len(F)]).T
        R = np.reshape(arr[:, 3], [len(A), len(F)]).T

        f = interpolate.interp1d(F, L, axis=0)
        g = interpolate.interp1d(F, R, axis=0)
        L0 = f(c)
        R0 = g(c)
        if np.max(np.abs(A))<180:
            A = np.concatenate([[-180], A, [180]])
            L180 = np.mean(L0[:,[0, -1]], axis=1, keepdims=True)
            L0 = np.concatenate([L180, L0, L180], axis=1)
            R180 = np.mean(R0[:,[0, -1]], axis=1, keepdims=True)
            R0 = np.concatenate([R180, R0, R180], axis=1)

        #f,ax=plt.subplots(1,2)
        #ax[0].imshow(L0, origin='lower', extent=[A[0],A[-1],c[0],c[-1]], aspect='auto')
        #im=ax[1].imshow(R0, origin='lower', extent=[A[0],A[-1],c[0],c[-1]], aspect='auto')
        #plt.colorbar(im, ax=ax[1])
        #f,ax=plt.subplots(1,2)
        #ax[0].imshow(L, origin='lower', extent=[A[0],A[-1],F[0],F[-1]], aspect='auto')
        #ax[1].imshow(R, origin='lower', extent=[A[0],A[-1],F[0],F[-1]], aspect='auto')
    elif format == 'az_el':
        filepath = libpath / 'hrtf_az_el.csv'
        arr = np.loadtxt(filepath, delimiter=",", dtype=float)
        # D(cc,:) = [azimuths(a) elevations(e) f(fi) left_full(fi, a, e) right_full(fi,a,e)];
        A = np.unique(arr[:, 0])
        E = np.unique(arr[:, 1])
        F = np.unique(arr[:, 2])
        L = np.reshape(arr[:, 3], [len(E), len(A), len(F)])
        R = np.reshape(arr[:, 4], [len(E), len(A), len(F)])
        f = interpolate.interp1d(F, L, axis=2)
        g = interpolate.interp1d(F, R, axis=2)
        L0 = f(c)
        R0 = g(c)

        return L0,R0,c,A,E
    else:
        raise ValueError(f'Only az or az_el HRTF currently supported')

    return L0, R0, c, A


## Routines for az+el HRTF

def rotate_via_numpy(xy, radians):
    """Use numpy to build a rotation matrix and take the dot product."""
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, xy)

    return m

def rotate_point_around_line(point, x1, x2, theta):
    """
    points, x1 and x2 are coordinates in 3D
    theta is rotation angle in radians
    """
    
    # Convert points to NumPy arrays
    point = np.array(point)
    x1 = np.array(x1)
    x2 = np.array(x2)

    # Calculate the rotation axis
    rotation_axis = x2 - x1
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Calculate the Rodrigues' rotation formula
    cross_product = np.cross(rotation_axis, point)
    rotated_point = (
        point * np.cos(theta) +
        np.cross(rotation_axis, point) * np.sin(theta) +
        rotation_axis * np.dot(rotation_axis, point) * (1 - np.cos(theta))
    )

    return rotated_point

def ptdist(x,y):
    """
    euclidean distance between vectors x and y
    """
    return np.sqrt(np.mean((x-y)**2))


def get_animal_head_coordinates(animal='SLJ'):
    
    if animal=='PRN':
        # PRN left chimney , coordinates from JW & JS
        xyz=np.array([[ 0,    0, -53,  25],
                      [-36.4, 0,  -5.4, -3],
                      [-14,   0, 11.3,  -23]])
        # PRN right chimney
        xyz=np.array([[ 0,    0, 25,  -36],
                      [-36.4, 0, -3,  1],
                      [-14,   0,  -23, 11.3]])
        xyz=np.array([[ 0,    0, 25,  -38],
                      [-34.4, 0, -3,  2],
                      [-14,   0,  -23, 11.3]])
        xyz=np.array([[ 0,    0, 25,  -45],
                      [-42, 0, -15,  -3],
                      [0,   0,  -9, 30]])
    elif animal == 'LMD':
        # LMD
        xyz=np.array([[0, 0, 52, -52],
                      [-34, 0, -2.3, -2.3],
                      [-17.25, 0, 18.75, 18.75]])
    elif aniaml == 'SLJ':
        # (x,y,z) x 4
        # SLJ - 2 chimneys
        xyz=np.array([[ 0,    0,  51.9, -53,],
                      [-36.4, 0, -6.1,  -5.4],
                      [-14,   0,  13.1, 11.3]])
        animal="SLJ"
    else:
        raise ValueError(f"Unknown animal {animal}")
        
    return xyz


def generate_tilt_yaw_lookup(animal='SLJ', tilts=None, yaws=None):
    """
    tilts - list or array in radians
    yaws - list of array in radians
    """
    
    if tilts is None:
        tilts=np.arange(20,-32.5,-2.5)/180*np.pi
    if yaws is None:
        yaws=np.arange(-30,32.5,2.5)/180*np.pi
    N=len(tilts)
    N2=len(yaws)
    print(f"tilts: {N} yaws: {N2}")
    tt=np.zeros((N,N2))
    yy=np.zeros((N,N2))
    pp=np.zeros((N,N2,2,4))
    xyz = get_animal_head_coordinates(animal)
    
    for i, tilt in enumerate(tilts):
        for j, yaw in enumerate(yaws):
            X=xyz.copy()
            X=xyz.copy()

            #X[1:,:] = rotate_via_numpy(X[1:,:],tilt)
            #X[0::2,:] = rotate_via_numpy(X[0::2,:],-yaw)
            X2=np.stack([rotate_point_around_line(X[:,i], [0,0,0], [1,0,0], -tilt) for i in range(4)], axis=1)
            X3=np.stack([rotate_point_around_line(X2[:,i], X2[:,0], X2[:,1], -yaw) for i in range(4)], axis=1)

            xy = -X3[:2,:]        
            xy = xy / np.abs(xy[1,0])

            pp[i,j] = xy
            tt[i,j]=tilt/np.pi*180
            yy[i,j]=yaw/np.pi*180

    tilts=tt.flatten()
    yaws=yy.flatten()
    pp = np.reshape(pp, (-1, 2, 4))
    
    return tilts,yaws,pp

def compute_rotations(dlc, tilts=None, yaws=None, pp=None, animal='SLJ'):
    
    if pp is None:
        tilts,yaws,pp = generate_tilt_yaw_lookup(animal='SLJ', tilts=tilts, yaws=yaws)
        
    aty=np.zeros((3,dlc.shape[1]))
    Emin=np.zeros(dlc.shape[1]) * np.nan
    for i in range(dlc.shape[1]):
        d = dlc[:,i]
        x=d[0:8:2]
        y=-d[1:8:2]

        delta=[x[1]-x[0], y[1]-y[0]]
        angle = np.arctan(delta[0]/delta[1])
        if y[0]<y[1]:
            angle=angle-np.pi
        angle_save = (angle+np.pi) % (2*np.pi) - np.pi
        xy = rotate_via_numpy(np.stack((x,y),axis=1).T, -angle)
        xy-=xy[:,[1]]
        xy = np.array(xy) / xy[1,0] # normed
        d_ = xy
        d_ = d_[np.newaxis,:,:].copy()
        d = pp-d_

        E=np.sum(d**2,axis=(1,2))
        aa2 = np.argmin(E)
        Emin[i]=E[aa2]
        aty[:,i]=np.array([angle_save*180/np.pi, tilts[aa2],yaws[aa2]])

    return aty, Emin