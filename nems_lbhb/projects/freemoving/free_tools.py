import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import logging

from scipy import interpolate

from nems0.analysis.gammatone.filters import centre_freqs
from nems0.utils import smooth
from nems_lbhb.preprocessing import impute_multi
from nems_lbhb import runclass, baphy_io

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
                   f_min=200, f_max=20000, channels=None, verbose=False,
                   **ctx):

    # require (stacked) binaural stim
    if channels is None:
        channels = int(rec[signal].shape[0]/2)
    rasterfs = rec[signal].fs
    stimcount = int(rec[signal].shape[0]/channels)
    log.info(f"HRTF: {signal} is {channels} x {stimcount}")

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

    if hrtf_format=='az':
        L0, R0, c, A = load_hrtf(format=hrtf_format, fmin=f_min, fmax=f_max, num_freqs=channels)

        f = interpolate.interp1d(A, R0, axis=1, fill_value="extrapolate")
        gainr1 = f(theta1)[:, 0, :] + gaind1
        gainr2 = f(theta2)[:, 0, :] + gaind2
        g = interpolate.interp1d(A, L0, axis=1, fill_value="extrapolate")
        gainl1 = g(theta1)[:, 0, :] + gaind1
        gainl2 = g(theta2)[:, 0, :] + gaind2
    elif hrtf_format=='az_el':
        L, R, Fr, Az, El = load_hrtf(format='az_el', fmin=200, fmax=20000, num_freqs=18)
        animal = rec.name[:3]
        if animal not in ['PRN','SLJ','LMD']:
            raise ValueError(f"can't get valid animal abbreviate from rec.name {rec.name}")
        elif rec.name[:5] in ["PRN00", "PRN01", "PRN02", "PRN03"]:
            animal="PRN-L"
        log.info(f"Loading posture map for {animal}.")
        xy1, xy2 = az_el_map(dlc_data_imp, animal=animal, speakers='both')
        xy1[1,xy1[1,:]>50]=50
        xy1[1,xy1[1,:]<-60]=-60
        xy2[1,xy2[1,:]>50]=50
        xy2[1,xy2[1,:]<-60]=-60

        A,E = np.meshgrid(Az, El)
        inp = np.stack([A.flatten(),E.flatten()],axis=1)
        outR = np.reshape(R, [-1, R.shape[2]])
        fR=interpolate.LinearNDInterpolator(inp,outR)
        outL = np.reshape(L, [-1, L.shape[2]])
        fL=interpolate.LinearNDInterpolator(inp,outL)
        gainr1 = fR(xy1[0,:], xy1[1,:]).T + gaind1
        gainl1 = fL(xy1[0,:], xy1[1,:]).T + gaind1
        gainr2 = fR(xy2[0,:], xy2[1,:]).T + gaind2
        gainl2 = fL(xy2[0,:], xy2[1,:]).T + gaind2
        if verbose:
            gainR = np.concatenate([gainr1,gainr2],axis=0)
            gainL = np.concatenate([gainl1,gainl2],axis=0)
            # plt.close('all')
            f,ax = plt.subplots(5,1, sharex=True)
            t1,t2=500,5000
            ax[0].imshow(gainR[:,t1:t2:10], interpolation='none', origin='lower')
            ax[1].plot(xy1[:,t1:t2:10].T)
            ax[2].plot(dlc_data_imp[[0,1],t1:t2:10].T)
            ax[3].plot(theta1[0,t1:t2:10])
            ax[4].imshow(gainL[:,t1:t2:10], interpolation='none', origin='lower')

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
    load HRTF and map to center frequencies of a gtgram fitlerbank
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
        if np.max(np.abs(A))<180:
            A = np.concatenate([[-180], A, [180]])
            L180 = np.mean(L0[:,[0, -1]], axis=1, keepdims=True)
            L0 = np.concatenate([L180, L0, L180], axis=1)
            R180 = np.mean(R0[:,[0, -1]], axis=1, keepdims=True)
            R0 = np.concatenate([R180, R0, R180], axis=1)

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
    
    if animal in ['PRN', 'PRN-R']:
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

        # front, back, L, R  x  (LR, FB, UD)
        # based on SLJ
        xyz = np.array([[  0, 0,  25, -50],
                        [-48, 0, -15, 0],
                        [  0, 0, -13,  25]])
        # worked out empirically
        xyz = np.array([[0, 0, 25, -50],
                        [-42, 0, -15, -3],
                        [0, 0, -9, 30]])
    elif animal == 'PRN-L':
        # PRN left chimney only
        # flipped from PRN-R (note sign change for x, not for y,z)
        xyz = np.array([[0, 0, 50, -25],
                        [-42, 0, -3, -15],
                        [0, 0, 30, -9]])

    elif animal == 'LMD':
        # LMD
        xyz=np.array([[0, 0, 52, -52],
                      [-34, 0, -2.3, -2.3],
                      [-17.25, 0, 18.75, 18.75]])
    elif animal == 'SLJ':
        # (x,y,z) x 4
        # SLJ - 2 chimneys
        xyz=np.array([[ 0,    0,  51.9, -53,],
                      [-36.4, 0, -6.1,  -5.4],
                      [-14,   0,  13.1, 11.3]])
        xyz = np.array([[   0,   0,  50, -50],
                        [ -48,   0, -14, -14],
                        [   0,   0,  25,  25]])
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
        tilts=np.arange(20,-37.5,-2.5)/180*np.pi
    if yaws is None:
        yaws=np.arange(-30,32.5,2.5)/180*np.pi
    N=len(tilts)
    N2=len(yaws)
    log.info(f"generate_tilt_yaw_lookup -- tilts: {N} yaws: {N2}")
    tt=np.zeros((N,N2))
    yy=np.zeros((N,N2))
    pp=np.zeros((N,N2,2,4))
    xyz = get_animal_head_coordinates(animal)
    
    for i, tilt in enumerate(tilts):
        for j, yaw in enumerate(yaws):
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

def compute_rotations(dlc, tilts=None, yaws=None, pp=None):
    """
    :param dlc:   raw dlc matrix (D x time)
    :param tilts: tilt of each template
    :param yaws:  yaw of each template
    :param pp:  templates
    :return:
       aty: 3 x T  [Az, El, Yaw] for each timepoint
       Emin: error of template match for each timepoint
    """
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
        aty[:,i]=np.array([angle_save*180/np.pi, tilts[aa2], yaws[aa2]])

    return aty, Emin

def az_el_map(dlc, animal='PRN', speakers='both', camera_adjust=True):

    tilts, yaws, pp = generate_tilt_yaw_lookup(animal=animal, tilts=None, yaws=None)
    aty, Emin = compute_rotations(dlc, tilts=tilts, yaws=yaws, pp=pp)

    # load HRTF
    #L, R, Fr, Az, El = load_hrtf(format='az_el', fmin=200, fmax=20000, num_freqs=18)

    speaker1_x0y0 = 1.0, -0.8  #R
    speaker2_x0y0 = 0.0, -0.8  #L

    # compute (d,theta) for each speaker location
    d1, theta1, _, _, _, _ = compute_d_theta(dlc, ref_x0y0=speaker1_x0y0)
    d2, theta2, _, _, _, _ = compute_d_theta(dlc, ref_x0y0=speaker2_x0y0)

    log.info(f"Computed rotations for {aty.shape} frames, d, theta for {d1.shape} frames")

    if camera_adjust:
        # adjust for camera angle
        x0, y0, z0 = 0.5, 0.375, 1.5  # meters?

        x = dlc[0:8:2, :]
        y = dlc[1:8:2, :]

        # print("post", x.max(), y.max())
        cameraoffset = np.array([x0 - x[1,:], y0 - y[1,:]])
        delta = np.array([x[0,:] - x[1,:], y[0,:] - y[1,:]])
        proj = np.sum(cameraoffset*delta, axis=0) / (delta[0,:] ** 2 + delta[1,:] ** 2) * delta + np.array([x[1,:], y[1,:]])
        sideoffset = ((proj[0,:] - x0) ** 2 + (proj[1,:] - y0) ** 2) ** 0.5
        fwdoffset = ((proj[0,:] - x[1]) ** 2 + (proj[1,:] - y[1,:]) ** 2) ** 0.5

        # is distance from center to L ear bigger than distance to L ear?
        Lcloser = (x0-x[2,:])**2+(y0-y[2,:])**2 < (x0-x[3,:])**2+(y0-y[3,:])**2
        sideoffset[Lcloser] = -sideoffset[Lcloser]
        nosecloser = (proj[0,:] - x[1, :]) ** 2 + (proj[1,:] - y[1, :]) ** 2 > (proj[0,:] - x[0, :]) ** 2 + (proj[1,:] - y[0, :]) ** 2
        fwdoffset[nosecloser] = -fwdoffset[nosecloser]

        fwangle = np.arctan(fwdoffset / z0) * 180 / np.pi
        sdangle = np.arctan(sideoffset / z0) * 180 / np.pi

        # T = aty.shape[1]
        # fwangle=np.zeros(T)
        # sdangle=np.zeros(T)
        # for i in range(T):
        #     x = dlc[0:8:2,i]
        #     y = dlc[1:8:2,i]
        #     #print("post", x.max(), y.max())
        #     cameraoffset = np.array([x0 - x[1], y0 - y[1]])
        #
        #     delta = np.array([x[0] - x[1], y[0] - y[1]])
        #     proj = np.dot(cameraoffset, delta) / (delta[0] ** 2 + delta[1] ** 2) * delta + np.array([x[1], y[1]])
        #     sideoffset = ((proj[0] - x0) ** 2 + (proj[1] - y0) ** 2) ** 0.5
        #     fwdoffset = ((proj[0] - x[1]) ** 2 + (proj[1] - y[1]) ** 2) ** 0.5
        #     if ptdist(np.array([x0, y0]), np.array([x[2], y[2]])) < ptdist(np.array([x0, y0]), np.array([x[3], y[3]])):
        #         sideoffset = -sideoffset
        #     if ptdist(proj, np.array([x[1], y[1]])) > ptdist(proj, np.array([x[0], y[0]])):
        #         fwdoffset = -fwdoffset
        #     fwangle[i] = np.arctan(fwdoffset / z0) * 180 / np.pi
        #     sdangle[i] = np.arctan(sideoffset / z0) * 180 / np.pi
        aty[1,:] += fwangle
        aty[2,:] += sdangle
        log.info(f"Adjusted for camera angle.")

    # raw speaker positions based on azimuth (theta) and tilt (aty[1,:])
    xy1r = np.stack([theta1[0,:], -aty[1,:]], axis=0)
    xy2r = np.stack([theta2[0,:], -aty[1,:]], axis=0)

    # make sure yaw rotation is correct when facing away from speakers
    wd1 = xy1r[0,:] < -90
    wu1 = xy1r[0,:]  > 90
    xy1r[0,wd1] = xy1r[0,wd1] + 180
    xy1r[0,wu1] = xy1r[0,wu1] - 180
    wd2 = xy2r[0,:] < -90
    wu2 = xy2r[0,:]  > 90
    xy2r[0,wd2] = xy2r[0,wd2] + 180
    xy2r[0,wu2] = xy2r[0,wu2] - 180

    # rotate to account for yaw
    yawrad = -aty[2, :] / 180 * np.pi
    c, s = np.cos(yawrad), np.sin(yawrad)
    x1 = (np.stack([c,s],axis=0) * xy1r).sum(axis=0)
    y1 = (np.stack([-s,c],axis=0) * xy1r).sum(axis=0)
    xy1 = np.stack([x1,y1], axis=0)
    x2 = (np.stack([c,s],axis=0) * xy2r).sum(axis=0)
    y2 = (np.stack([-s,c],axis=0) * xy2r).sum(axis=0)
    xy2 = np.stack([x2,y2], axis=0)
    #np.concatenate([xy1,xy2],axis=0)[:,0]

    # undo adjustment for facing away
    xy1[0, wd1] = xy1[0, wd1] + 180
    xy1[0, wu1] = xy1[0, wu1] - 180
    xy2[0, wd2] = xy2[0, wd2] + 180
    xy2[0, wu2] = xy2[0, wu2] - 180

    xy1 = (xy1 + 180) % 360 - 180
    xy2 = (xy2 + 180) % 360 - 180

    # f,ax=plt.subplots(1,2)
    # ax[0].plot(xy1r[1,:])
    # ax[0].plot(xy1[1,:])
    # ax[1].plot(xy2r[1,:])
    # ax[1].plot(xy2[1,:])
    return xy1, xy2

def plot_frame(t, dlc, aty, frame_file=None, ax=None):
    from matplotlib import image

    # load image as pixel array
    if frame_file is None:
        frame_file = f'/auto/data/tmp/{animal}/frame{t:05d}.jpg'

    im = image.imread(frame_file)
    if ax is None:
        f, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(im)

    d = dlc.copy()
    #if np.nanmax(dlc) < 2:
    #    d *= im.shape[1]
    #x0, y0, z0 = 320, 240, 975
    x0, y0, z0 = 0.5, 0.375, 1.5  # meters?

    x = d[0:8:2]
    y = d[1:8:2]
    cameraoffset = np.array([x0 - x[1], y0 - y[1]])

    delta = np.array([x[0] - x[1], y[0] - y[1]])
    proj = np.dot(cameraoffset, delta) / (delta[0] ** 2 + delta[1] ** 2) * delta + np.array([x[1], y[1]])
    sideoffset = ((proj[0] - x0) ** 2 + (proj[1] - y0) ** 2) ** 0.5
    fwdoffset = ((proj[0] - x[1]) ** 2 + (proj[1] - y[1]) ** 2) ** 0.5
    if ptdist(np.array([x0, y0]), np.array([x[2], y[2]])) < ptdist(np.array([x0, y0]), np.array([x[3], y[3]])):
        sideoffset = -sideoffset
    if ptdist(proj, np.array([x[1], y[1]])) > ptdist(proj, np.array([x[0], y[0]])):
        fwdoffset = -fwdoffset
    fwangle = np.arctan(fwdoffset / z0) * 180 / np.pi
    sdangle = np.arctan(sideoffset / z0) * 180 / np.pi

    sf = im.shape[1]
    ax.plot(x*sf, y*sf, lw=3)
    ax.plot(x[2]*sf, y[2]*sf, 'o', color='blue')
    ax.plot(x[3]*sf, y[3]*sf, 'o', color='red')
    ax.plot([x0*sf, x[1]*sf], [y0*sf, y[1]*sf])
    ax.plot([x0*sf, proj[0]*sf], [y0*sf, proj[1]*sf])
    # print(t,ptdist(np.array([x0,y0]),np.array([x[2],y[2]])), ptdist(np.array([x0,y0]),np.array([x[3],y[3]])),(x[2],y[2]),(x[3],y[3]))
    # print(t,ptdist(np.array([x0,y0]),np.array([x[2],y[2]])), ptdist(np.array([x0,y0]),np.array([x[3],y[3]])),(x[2],y[2]),(x[3],y[3]))
    tty = aty[1:].copy()
    if tty[0] > 0:
        ts = f"{tty[0]:.0f} U"
    else:
        ts = f"{tty[0]:.0f} D"
    if tty[1] > 0:
        ys = f"{tty[1]:.0f} R"
    else:
        ys = f"{tty[1]:.0f} L"

    ax.set_title(f"t={t} tilt: {ts}, yaw: {ys} fw: {fwangle:.0f} sd: {sdangle:.0f}");

def validate_hrtf2(animal="PRN", dlc_file=None, video_file=None,
                   test_frames=None, verbose=False):
    if (dlc_file is None) or (video_file is None):
        if animal == "LMD":
            dlc_file = '/auto/data/daq/LemonDisco/LMD004/sorted/LMD004a07_a_NFB.dlc.h5'
            video_file = '/auto/data/daq/LemonDisco/LMD004/LMD004a07_a_NFB.avi'
        elif animal == "PRN":
            dlc_file = '/auto/data/daq/Prince/PRN048/sorted/PRN048a02_a_NTD.dlc.h5'
            video_file = '/auto/data/daq/Prince/PRN048/PRN048a02_a_NTD.avi'
        elif animal == "SLJ":
            dlc_file = '/auto/data/daq/SlipperyJack/SLJ019/sorted/SLJ019a05_a_NTD.dlc.h5'
            video_file = '/auto/data/daq/SlipperyJack/SLJ019/SLJ019a05_a_NTD.avi'
        else:
            raise ValueError(f"unknown animal {animal}")

    dlc, bodyparts = baphy_io.load_dlc_trace(dlc_file, return_raw=True)

    if np.nanmax(dlc) > 700:
        dlc /= 1280
    else:
        dlc /= 640
    print(dlc_file)
    print("Animal:", animal, "-- DLC shape:", dlc.shape, "max", np.nanmax(dlc))
    print(get_animal_head_coordinates(animal))
    if test_frames is None:
        test_frames = np.arange(500,np.min([8000,dlc.shape[1]]),500)
    frame_path = f'/auto/data/tmp/dlc/{Path(video_file).stem}/'
    os.makedirs(frame_path, exist_ok=True)
    for t in test_frames:
        frame_file = f"{frame_path}frame{t:05d}.jpg"
        #print(os.path.isfile(frame_file), frame_file)
        if os.path.isfile(frame_file) == False:
            # cmd = f"ffmpeg -i {video_file} -ss {t/30} -vframes 1 /tmp/frame{t:05d}.jpg"
            cmd = f"ffmpeg -i {video_file} -vf select='eq(n\,{t})' -vsync 0 {frame_file}"
            os.system(cmd)

    if verbose:
        tilts0 = np.arange(20, -30, -10) / 180 * np.pi
        yaws0 = np.arange(-10, 15, 10) / 180 * np.pi
        N = len(tilts0)
        N2 = len(yaws0)
        tilts, yaws, pp = generate_tilt_yaw_lookup(animal=animal, tilts=tilts0, yaws=yaws0)
        f, ax = plt.subplots(N, N2, sharex=True, sharey=True, figsize=(8, 8))
        cc = 0
        for i, tilt in enumerate(tilts0):
            for j, yaw in enumerate(yaws0):
                xy = pp[cc]
                # ax[i,j].plot(xy0[0,:].T,xy0[1,:].T, color='gray')
                ax[i, j].plot(xy[0, :].T, xy[1, :].T)
                ax[i, j].set_title(f"{np.round(xy[0, 2:],1)}, {np.round(xy[1, 2:],1)}")
                if i == N - 1:
                    if yaw > 0:
                        ax[i, j].set_xlabel(f"{yaw / np.pi * 180:0.0f} R")
                    else:
                        ax[i, j].set_xlabel(f"{yaw / np.pi * 180:0.0f} L")
                cc += 1
            if tilt > 0:
                ax[i, 0].set_ylabel(f"{tilt / np.pi * 180:0.0f} U")
            else:
                ax[i, 0].set_ylabel(f"{tilt / np.pi * 180:0.0f} D")
        plt.tight_layout()


    tilts, yaws, pp = generate_tilt_yaw_lookup(animal=animal, tilts=None, yaws=None)
    aty, Emin = compute_rotations(dlc[:, test_frames], tilts=tilts, yaws=yaws, pp=pp)

    if verbose:
        f, ax = plt.subplots(1, 1, figsize=(10, 8))
        ty = np.zeros((2, dlc.shape[1]))
        ty2 = np.zeros((2, dlc.shape[1]))
        Emin = np.zeros(dlc.shape[1]) * np.nan
        for i in range(0, np.min([dlc.shape[1],10000]), 100):
            d = dlc[:, i]
            x = d[0:8:2]
            y = -d[1:8:2]
            # ax[0].plot(x,y)

            delta = [x[1] - x[0], y[1] - y[0]]
            angle = np.arctan(delta[0] / delta[1])
            if y[0] > y[1]:
                xy = rotate_via_numpy(np.stack((x, y), axis=1).T, -angle)
            else:
                xy = rotate_via_numpy(np.stack((x, y), axis=1).T, np.pi - angle)
            xy -= xy[:, [1]]
            xy = np.array(xy) / xy[1, 0]  # normed
            d_ = xy
            d_ = d_[np.newaxis, :, :].copy()
            d = pp - d_

            E = np.sum(d ** 2, axis=(1, 2))
            aa2 = np.argmin(E)
            Emin[i] = E[aa2]
            ty2[:, i] = np.array([tilts[aa2], yaws[aa2]])
            if Emin[i] < 1:
                ax.plot(xy[0, :].T + yaws[aa2], xy[1, :].T + tilts[aa2], color='gray')
            else:
                ax.plot(xy[0, :].T + yaws[aa2], xy[1, :].T + tilts[aa2], color='red')

        ax.set_xlabel('yaw')
        ax.set_ylabel('tilt')


    # load HRTF
    L, R, Fr, Az, El = load_hrtf(format='az_el', fmin=200, fmax=20000, num_freqs=18)

    speaker1_x0y0 = 1.0, -0.8
    speaker2_x0y0 = 0.0, -0.8

    # compute (d,theta) for entire vector to allow smoothing
    d1, theta1, _, _, _, _ = compute_d_theta(dlc, ref_x0y0=speaker1_x0y0)
    d2, theta2, _, _, _, _ = compute_d_theta(dlc, ref_x0y0=speaker2_x0y0)
    # extract target frames
    d1=d1[:,test_frames]
    theta1=theta1[:,test_frames]
    d2=d2[:,test_frames]
    theta2=theta2[:,test_frames]

    print(f"Computed rotations for {aty.shape} frames, d,theta for {d1.shape} frames")
    xy1, xy2 = az_el_map(dlc[:, test_frames], animal=animal, speakers='both')

    for i, f in enumerate(test_frames):
        #plt.close('all')
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))

        frame_file = f"{frame_path}frame{f:05d}.jpg"
        plot_frame(f, dlc[:,f], aty[:,i], frame_file=frame_file, ax=ax[0])

        fi = 12
        fi2 = 4
        ax[1].imshow(R[:, :, fi], extent=[Az[0], Az[-1], El[0], El[-1]])
        ax[2].imshow(L[:, :, fi2], extent=[Az[0], Az[-1], El[0], El[-1]])
        if f==5500:
            print(f"f={f}")
        x = np.array([theta1[0, i], theta2[0, i]])
        xwrap = x.copy()
        wd=xwrap<-90
        wu=xwrap>90
        xwrap[wd]=xwrap[wd]+180
        xwrap[wu]=xwrap[wu]-180
        y = np.array([-aty[1, i], -aty[1, i]])
        xy0 = np.stack((xwrap, y), axis=0)
        xy = rotate_via_numpy(xy0, -aty[2, i] / 180 * np.pi)
        xy[0,wd]=xy[0,wd]-180
        xy[0,wu]=xy[0,wu]+180
        xy[1, :] = (xy[1, :] + 180) % 360 - 180
        for a in ax[1:3]:
            a.plot(x[0],y[0],'o',color='gray')
            a.plot(x[1],y[1],'o',color='gray')
            a.plot(xy[0, 0], xy[1, 0], 'o', color='red', label='R spkr')
            a.plot(xy[0, 1], xy[1, 1], 'o', color='blue', label='L spkr')
            a.text(xy1[0,i], xy1[1,i],'R')
            a.text(xy2[0,i], xy2[1,i], 'L')
        ax[1].set_title(f"f={Fr[fi]:.0f} Hz, R ear th=[{theta1[0, i]:.0f},{theta2[0, i]:.0f}] aty={np.round(aty[:, i], 1)}")
        ax[2].set_title(f"f={Fr[fi2]:.0f} Hz, L ear")
        ax[2].legend()
    return aty
