from math import cos, sin, radians
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from nems import db


def penetration_map(sites, cubic=False, flip_X=False, flatten=False):
    """
    Plots a 3d mapt of the list of specified sites, displaying the best frequency as color, and the brain region as
    maker type (NA: circle, A1: triangle, PEG: square).
    The site location, brain area and best frequency are extracted from celldb, specifically from the penetration (for
    coordinates and rotations) and the penetration (for area and best frequency) sites. If no coordinates are found the
    site is ignored. no BF is displayed as an empty marker.
    The values in the plot increase in direction Posterior -> Anterior, Medial -> Lateral and Ventral -> Dorsal. The
    antero posterior (X) axis can be flipped with the according parameter.
    :param sites: list of str specifying sites, with the format "ABC001a"
    :param cubic: boolean. whether to constrain the data to a cubic space i.e. equal dimensions in XYZ
    :flip_X: Boolean. whether to flip the direction labels for the antero-posterior (X) axis. The default is A > P .
    Y. Lateral > Medial, Z. Dorsal > Ventral.
    :flatten: Boolean. PCA 2d projection. Work in progress.
    :return: matplotlib figure
    """
    area_marker = {'NA': 'o', 'A1': '^', 'PEG': 's'}

    coordinates = list()
    best_frequencies = list()
    areas = list()
    good_sites = list()

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
        raw_BF, raw_area = db.pd_query(BF_querry, params=(site,)).iloc[0, :]

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

    coordinates = np.stack(coordinates, axis=1)
    best_frequencies = np.asarray(best_frequencies)
    areas = np.asarray(areas)
    good_sites = np.asarray(good_sites)

    # centers around the mean and transforms cm to mm
    coordinates = coordinates - np.mean(coordinates, axis=1)[:, None]
    coordinates = coordinates * 10

    # defines BF colormap range if valid best frequencies are available.
    vmax = best_frequencies.max() if best_frequencies.max() > 0 else 32000
    vmin = best_frequencies[best_frequencies != 0].min() if best_frequencies.min() > 0 else 100

    if flatten is False:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if cubic:
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
        ax.set_xlabel('anterior posterior (mm)')
        ax.set_ylabel('1PC_YZ (mm)')



    cbar = fig.colorbar(p)
    cbar.ax.set_ylabel('BF (Hz)', rotation=-90, va="top")

    return fig, coordinates


# test_set
sites = ['ARM012d', 'ARM013b', 'ARM014b', 'ARM015b', 'ARM016c', 'ARM017a', 'ARM018a', 'ARM019a', 'ARM020a',
         'ARM021b', 'ARM022b', 'ARM023a', 'ARM024a', 'ARM025a', 'ARM026b', 'ARM027a', 'ARM028b', 'ARM029a', 'ARM030a']

# fig, coords = penetration_map(sites, cubic=False, flip_X=True, flatten=False)
fig, coords = penetration_map(sites, cubic=False, flip_X=True, flatten=True)

print('hola')
