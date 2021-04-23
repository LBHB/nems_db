from math import cos, sin, radians
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from nems import db


def penetration_map(sites, equal_aspect=False, flip_X=False, flatten=False, landmarks=None):
    """
    Plots a 3d map of the list of specified sites, displaying the best frequency as color, and the brain region as
    maker type (NA: circle, A1: triangle, PEG: square).
    The site location, brain area and best frequency are extracted from celldb, specifically from the penetration (for
    coordinates and rotations) and the penetration (for area and best frequency) sites. If no coordinates are found the
    site is ignored. no BF is displayed as an empty marker.
    The values in the plot increase in direction Posterior -> Anterior, Medial -> Lateral and Ventral -> Dorsal. The
    antero posterior (X) axis can be flipped with the according parameter.
    :param sites: list of str specifying sites, with the format "ABC001a"
    :param equal_aspect: boolean. whether to constrain the data to a cubic/square space i.e. equal dimensions in XYZ/XY
    :flip_X: Boolean. whether to flip the direction labels for the antero-posterior (X) axis. The default is A > P .
    Y. Lateral > Medial, Z. Dorsal > Ventral.
    :flatten: Boolean. PCA 2d projection. Work in progress.
    :landmarks: dict of vectors, where the key specifies the landmark name, and the vector has the values
    [x0, y0, z0, x, y, z, tilt, rot]. If the larnmark name is 'OccCrest' or 'MidLine' uses the AP and ML values as zeros
    respectively.
    :return: matplotlib figure
    """
    area_marker = {'NA': 'o', 'A1': '^', 'PEG': 's'}

    coordinates = list()
    best_frequencies = list()
    areas = list()
    good_sites = list()

    # get values from cell db and transforms into coordinates
    for pp, site in enumerate(sites):
        # gets the penetrations MT coordinates
        penetration = site[0:6]
        coord_querry = "select ecoordinates from gPenetration where penname=%s"
        raw_coords = db.pd_query(coord_querry, params=(penetration,)).iloc[0, 0]
        all_coords = np.asarray(raw_coords.split(',')).astype(float).squeeze()

        # Dimensions X:Antero-Posterior Y:Medio-Lateral Z:Dorso-Ventral
        MT_0 = all_coords[0:3]
        MT = all_coords[3:6]
        tilt = radians(all_coords[6])
        rotation = radians(all_coords[7])

        # rejects sites with no coordinates
        no_ref = np.all(MT_0 == 0)
        no_val = np.all(MT == 0)
        if no_ref or no_val:
            print(f'skipping penetration {penetration}, No coordinates specified')
            continue

        # defines tilt and rotation matrices, tilt around X axis, rotation around Z axis
        tilt_mat = np.asarray([[1, 0, 0],
                               [0, cos(tilt), sin(tilt)],
                               [0, -sin(tilt), cos(tilt)]])
        rot_mat = np.asarray([[cos(rotation), sin(rotation), 0],
                              [-sin(rotation), cos(rotation), 0],
                              [0, 0, 1]])

        # calculates the relative MT coordinates and rotates.
        MT_rel_rot = rot_mat @ tilt_mat @ (MT - MT_0)

        # get the first value of BF from cellDB TODO this is a temporary cludge, in the future, with all BFs set, a more elegant approach is required
        BF_querry = "select bf, area from gCellMaster where siteid=%s"
        try:
            raw_BF, raw_area = db.pd_query(BF_querry, params=(site,)).iloc[0, :]
        except:
            raw_BF = None
            raw_area = None
        # Sanitize best frequency in case of missing values
        if raw_BF is None:
            print(f'site {site} has undefined best frequency')
            BF = 0
        else:
            BF = int(raw_BF.split(',')[0])
            if BF == 0:
                print(f'site {site} has undefined best frequency')

        # Sanitizes region in case of missing values
        if raw_area is None:
            print(f'site {site} has undefined region')
            area = 'NA'
        else:
            area = raw_area.split(',')[0]
            if area == '':
                print(f'site {site} has undefined region')
                area = 'NA'
            elif area not in ('A1', 'PEG'):
                print(f'site {site} has unrecognized region: {area}')
                area = 'NA'

        coordinates.append(MT_rel_rot)
        best_frequencies.append(BF)
        areas.append(area)
        good_sites.append(site)

    # adds manual landmarks specified in dictionary
    if landmarks is not None:
        X0 = []
        Y0 = []
        for landname, all_coords in landmarks.items():
            all_coords = np.asarray(all_coords)
            MT_0 = all_coords[0:3]
            MT = all_coords[3:6]
            tilt = radians(all_coords[6])
            rotation = radians(all_coords[7])

            # defines tilt and rotation matrices, tilt around X axis, rotation around Z axis
            tilt_mat = np.asarray([[1, 0, 0],
                                   [0, cos(tilt), sin(tilt)],
                                   [0, -sin(tilt), cos(tilt)]])
            rot_mat = np.asarray([[cos(rotation), sin(rotation), 0],
                                  [-sin(rotation), cos(rotation), 0],
                                  [0, 0, 1]])

            # calculates the relative MT coordinates and rotates.
            MT_rel_rot = rot_mat @ tilt_mat @ (MT - MT_0)

            coordinates.append(MT_rel_rot)
            best_frequencies.append(0)
            areas.append('NA')
            # pads with spaces and holds 3 letter, for consistent naming with sites
            good_sites.append(f'   {landname[0:3]}')

            #saves values as zero reference if correct landname
            if landname == 'OccCrest': X0.append(MT_rel_rot[0])
            if landname == 'MidLine': Y0.append(MT_rel_rot[1])



    coordinates = np.stack(coordinates, axis=1)
    best_frequencies = np.asarray(best_frequencies)
    areas = np.asarray(areas)
    good_sites = np.asarray(good_sites)

    # centers data and transforms cm to mm
    center = np.mean(coordinates, axis=1)

    # uses landmarks as zero values if any
    if landmarks is not None:
        if X0: center[0] = X0[0]
        if Y0: center[1] = Y0[0]

    coordinates = coordinates - center[:, None]
    coordinates = coordinates * 10

    # defines BF colormap range if valid best frequencies are available.
    vmax = best_frequencies.max() if best_frequencies.max() > 0 else 32000
    vmin = best_frequencies[best_frequencies != 0].min() if best_frequencies.min() > 0 else 100

    if flatten is False:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if equal_aspect:
            X, Y, Z = coordinates
            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')


        for area in set(areas):
            coord_subset = coordinates[:, areas == area]
            BF_subset = best_frequencies[areas == area]
            site_subset = good_sites[areas == area]

            X, Y, Z = coord_subset
            p = ax.scatter(X, Y, Z, s=100, marker=area_marker[area], edgecolor='black',
                           c=BF_subset, cmap='inferno',
                           norm=colors.LogNorm(vmin=vmin, vmax=vmax))

            for coord, site in zip(coord_subset.T, site_subset):
                x, y, z = coord
                ax.text(x, y, z, site[3:6])

        # formats axis
        ax.set_xlabel('anterior posterior (mm)')
        ax.set_ylabel('Medial Lateral (mm)')
        ax.set_zlabel('Dorsal ventral (mm)')

        fig.canvas.draw()
        xlab = [item.get_text() for item in ax.get_xticklabels() if item.get_text() != '']
        if flip_X:
            xlab[0] = 'A'
            xlab[-1] = 'P'
        else:
            xlab[0] = 'P'
            xlab[-1] = 'A'
        _ = ax.set_xticklabels(xlab)

        ylab = [item.get_text() for item in ax.get_yticklabels() if item.get_text() != '']
        ylab[0] = 'M'
        ylab[-1] = 'L'
        _ = ax.set_yticklabels(ylab)

        zlab = [item.get_text() for item in ax.get_zticklabels() if item.get_text() != '']
        zlab[0] = 'V'
        zlab[-1] = 'D'
        _ = ax.set_zticklabels(zlab)

    elif flatten is True:
        # flattens doing a PCA over the Y and Z dimensions, i.e. medio-lateral and dorso-ventral.
        # this keeps the anteroposterior orientations to help locate the flattened projection
        pc1 = PCA().fit_transform(coordinates[1:,:].T)[:,0]

        # uses midline zero as pc1 zero
        if landmarks is not None:
            pc1 = pc1 - pc1[np.argwhere(coordinates[1,:] == 0)].squeeze()

        flat_coords = np.stack((coordinates[0,:], pc1), axis=0)

        if flip_X is True:
            flat_coords = flat_coords * np.array([[-1],[ 1]])


        fig, ax = plt.subplots()

        for area in set(areas):
            flat_coords_subset = flat_coords[:, areas == area]
            BF_subset = best_frequencies[areas == area]
            site_subset = good_sites[areas == area]

            X, Y = flat_coords_subset
            p = ax.scatter(X, Y, s=100, marker=area_marker[area], edgecolor='black',
                           c=BF_subset, cmap='inferno',
                           norm=colors.LogNorm(vmin=vmin, vmax=vmax))

            for coord, site in zip(flat_coords_subset.T, site_subset):
                x, y = coord
                ax.text(x, y, site[3:6])

        # formats axis
        if equal_aspect:
            ax.axis('equal')
        ax.set_xlabel('anterior posterior (mm)')
        ax.set_ylabel('1PC_YZ (mm)')



    cbar = fig.colorbar(p)
    cbar.ax.set_ylabel('BF (Hz)', rotation=-90, va="top")

    return fig, coordinates

# # test_set
# sites = ['ARM012d', 'ARM013b', 'ARM014b', 'ARM015b', 'ARM016c', 'ARM017a', 'ARM018a', 'ARM019a', 'ARM020a',
#          'ARM021b', 'ARM022b', 'ARM023a', 'ARM024a', 'ARM025a', 'ARM026b', 'ARM027a', 'ARM028b', 'ARM029a', 'ARM030a']
#
# fig, coords = penetration_map(sites, cubic=False, flip_X=True, flatten=False)
# fig, coords = penetration_map(sites, cubic=False, flip_X=True, flatten=True)

# test landmarks
# ref = [0.84,6.91,3.93]
# tr = [49,0]
# sites = []
# landmarks = {'MidLine'     : ref+[1.45,5.75,3.49]+tr,
#              'OccCrest': ref+[0.045,6.39,4.26]+tr,
#              'Occ_Crest_in' : ref+[0.352,6.39,4.26]+tr}
# fig, coords = penetration_map(sites, equal_aspect=True, flip_X=False, flatten=True, landmarks=landmarks)
# plt.show()
# 
'''
ref = [0.91, 5.27, 4.99]
tr = [42,0]
sites = ['JLY002', 'JLY003', 'JLY004', 'JLY007d', 'JLY008', 'JLY009b', 'JLY010b', 'JLY011c', 'JLY012d', 'JLY013c', 'JLY014d']
landmarks = {'MidLine'     : ref+[1.384, 4.53, 4.64]+tr,
              'OccCrest': ref+[0.076, 5.27, 5.28]+tr,
              'Occ_Crest_in' : ref+[0.490, 5.27, 5.28]+tr}
fig, coords = penetration_map(sites, equal_aspect=True, flip_X=False, flatten=True, landmarks=None)
fig.axes[0].grid()
plt.show()
'''