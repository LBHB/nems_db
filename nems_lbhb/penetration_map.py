from nems import db
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from math import cos, sin, radians


sites = ['ARM012d', 'ARM013b', 'ARM014b', 'ARM015b', 'ARM016c', 'ARM017a', 'ARM018a', 'ARM019a', 'ARM020a',
        'ARM021b', 'ARM022b', 'ARM023a', 'ARM024a', 'ARM025a', 'ARM026b', 'ARM027a', 'ARM028b', 'ARM029a']
flip_ax = [1, 1, 1] # x y z
cubic_box=False
area_marker = {'NA': 'o', 'A1': '^', 'PEG': 's'}

coordinates = list()
best_frequencies = list()
areas = list()
good_sites = list()

for pp, site in enumerate(sites):
    # gets the penetrations MT coordinates
    penetration = site[0:6]
    coord_querry= "select ecoordinates from gPenetration where penname=%s"
    raw_coords = db.pd_query(coord_querry, params=(penetration,)).iloc[0, 0]
    all_coords = np.asarray(raw_coords.split(',')).astype(float).squeeze()

    # Dimensions X:Antero-Posterior Y:Medio-Lateral Z:Dorso-Ventral
    MT_0 = all_coords[0:3]
    MT = all_coords[3:6]
    tilt = radians(all_coords[6])
    rotation = radians(all_coords[7])

    # rejects sites with no coordinates
    no_ref = np.all(MT_0==0)
    no_val = np.all(MT==0)
    if no_ref or no_val:
        print(f'skipping penetration {penetration}, No coordinates specified')
        continue

    # defines tilt and rotation matrices, tilt around X axis, rotation around Z axis
    tilt_mat = np.asarray([[1, 0        , 0         ],
                           [0, cos(tilt) , sin(tilt)],
                           [0, -sin(tilt), cos(tilt)]])
    rot_mat = np.asarray([[cos(rotation) , sin(rotation), 0],
                          [-sin(rotation), cos(rotation) , 0],
                          [0             , 0             , 1]])

    # calculates the relative MT coordinates and rotates.
    MT_rel = (MT - MT_0) * flip_ax
    MT_rel_rot = rot_mat @ tilt_mat @ MT_rel


    # get the first value of BF from cellDB TODO this is a temporary cludge, in the future, with all BFs set, a more elegant approach is required
    BF_querry ="select bf, area from gCellMaster where siteid=%s"
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


    coordinates.append(MT_rel_rot)
    best_frequencies.append(BF)
    areas.append(area)
    good_sites.append(site)



coordinates = np.stack(coordinates, axis=1)
best_frequencies = np.asarray(best_frequencies)
areas = np.asarray(areas)
good_sites = np.asarray(good_sites)

# finds the center of mass and recenters the data
coordinates = coordinates - np.mean(coordinates, axis=1)[:,None]
#original coordinates are in cm, transforms to mm
coordinates = coordinates * 10


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

if cubic_box:
    X, Y, Z = coordinates
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')


vmax = best_frequencies.max() if best_frequencies.max() > 0 else 32000

for area in set(areas):
    coord_subset = coordinates[:, areas==area]
    BF_subset = best_frequencies[areas==area]
    site_subset = good_sites[areas==area]

    X, Y, Z = coord_subset
    p = ax.scatter(X, Y, Z, s=100, marker=area_marker[area], edgecolor='black',
                   c=BF_subset, cmap='inferno',
                   norm=colors.LogNorm(vmin=100, vmax=vmax))

    for coord, site in zip(coord_subset.T, site_subset):
        x, y, z = coord
        ax.text(x, y, z, site[3:6])


cbar = fig.colorbar(p)
cbar.ax.set_ylabel('BF (Hz)', rotation=-90, va="top")


# formats axis
ax.set_xlabel('anterior posterior (mm)')
ax.set_ylabel('Medial Lateral (mm)')
ax.set_zlabel('Dorsal ventral (mm)')

xlab = [item.get_text() for item in ax.get_xticklabels() if item.get_text() != '']
xlab[0] = 'P'; xlab[-1] = 'A'
_=ax.set_xticklabels(xlab)

ylab = [item.get_text() for item in ax.get_yticklabels() if item.get_text() != '']
ylab[0] = 'L'; ylab[-1] = 'M'
_=ax.set_yticklabels(ylab)

zlab = [item.get_text() for item in ax.get_zticklabels() if item.get_text() != '']
zlab[0] = 'V'; zlab[-1] = 'D'
_=ax.set_zticklabels(zlab)
