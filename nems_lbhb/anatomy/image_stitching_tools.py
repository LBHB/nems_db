import czifile as cz
import matplotlib
matplotlib.use('TkAgg')
from nems_lbhb import SettingXML as xml
import xml.etree.ElementTree as ET
import numpy as np
import glob
from czitools import metadata_tools as czimd
import os
from pathlib import Path
import numpy as np
import imutils
import cv2
from PIL import Image
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from functools import partial
from tkinter import filedialog
import random
import tifffile as tif
import json

class image_stitcher:
    def __init__(self):
        self.image_dir = None
        self.image1 = None
        self.image2 = None
        self.image2_metadata = None
        self.image1_metadata = None

    def set_images(self, hist, chans):
        self.hist = hist
        self.chans = chans
        image_dir = filedialog.askdirectory()
        self.image_dir = image_dir
        im1, czi1, metadata1, name1, im_path1 = open_image_selector(impath=self.image_dir, hist=self.hist, chans=self.chans)
        self.image1 = im1.astype(dtype="uint8")
        self.image1_metadata = {"czi": czi1, "metadata":metadata1, "name":name1, "path":im_path1, 'hist_method': hist, 'chans': chans}
        im2, czi2, metadata2, name2, im_path2 = open_image_selector(impath=self.image_dir, hist=self.hist, chans=self.chans)
        self.image2 = im2.astype(dtype="uint8")
        self.image2_metadata = {"czi": czi2, "metadata": metadata2, "name": name2, "path": im_path2, 'hist_method': hist, 'chans': chans}
        self.aligned_rgb = None
        self.aligned_bgr = None
        self.aligned_name = "Stitched"
        self.plot_setup()
        self.plot_stitched_setup()
        self.src_points = []
        self.src_desc = []
        self.dst_points = []
        self.dst_desc = []
        self.mask1 = []
        self.mask2 = []
        self.im1_upper_corner = []
        self.im1_lower_corner = []
        self.im2_upper_corner = []
        self.im2_lower_corner = []
        h1, w1 = self.image1.shape
        h2, w2 = self.image2.shape
        ms = max([h1,w1,h2,w2])
        self.borders = [ms,ms,ms,ms]
        self.plot_images()

    def rotation(self, image_index, theta):
        if image_index == 0:
            self.image1 = crop_image_only_outside(imutils.rotate(self.image1, theta), tol=0)
        if image_index == 1:
            self.image2 = crop_image_only_outside(imutils.rotate(self.image2, theta), tol=0)
        self.update_plots()

    def plot_setup(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        plt.ion()
        self.fig = fig
        self.image1_ax = ax[0]
        self.image1_ax.set_title(self.image1_metadata['name'])
        self.image2_ax = ax[1]
        self.image2_ax.set_title(self.image2_metadata['name'])
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    def plot_stitched_setup(self):
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        plt.ion()
        self.fig_stiched = fig
        self.image_stitched_ax = ax
        self.image_stitched_ax.set_title(self.aligned_name)
        plt.show(block=False)
        plt.pause(0.001)

    def plot_images(self):
        cspace = np.linspace(0, 1, len(self.src_points)+1)
        cmap = plt.get_cmap('turbo')
        self.image1_ax.imshow(self.image1, cmap='gray')
        self.image2_ax.imshow(self.image2, cmap='gray')
        self.image1_ax.axis("off")
        self.image2_ax.axis("off")
        self.image1_ax.set_title(self.image1_metadata['name'])
        self.image2_ax.set_title(self.image2_metadata['name'])


        im1_x = [pt[0][0] for pt in self.src_points]
        im1_y = [pt[0][1] for pt in self.src_points]
        pt_colors = [cmap(cspace[i]) for i in range(len(self.src_points))]
        self.image1_ax.scatter(im1_x, im1_y, marker='o', color=pt_colors)
        im2_x = [pt[0][0] for pt in self.dst_points]
        im2_y = [pt[0][1] for pt in self.dst_points]
        self.image2_ax.scatter(im2_x, im2_y, marker='o', color=pt_colors)
        plt.show(block=False)
        plt.pause(0.001)

    def move_features(self):
        cspace = np.linspace(0, 1, len(self.src_points)+1)
        cmap = plt.get_cmap('turbo')
        f, ax = plt.subplots(1,2)
        plt.show(block=False)
        ax[0].imshow(self.image1, cmap='gray')
        im1_x = [pt[0][0] for pt in self.src_points]
        im1_y = [pt[0][1] for pt in self.src_points]
        pt_colors = [cmap(cspace[i]) for i in range(len(self.src_points))]
        ax[1].imshow(self.image2, cmap='gray')
        im2_x = [pt[0][0] for pt in self.dst_points]
        im2_y = [pt[0][1] for pt in self.dst_points]
        im1_scatter = ax[0].scatter(im1_x, im1_y, marker='o', color=pt_colors)
        self.im1_scatter = im1_scatter
        im2_scatter = ax[1].scatter(im2_x, im2_y, marker='o', color=pt_colors)
        self.im2_scatter = im2_scatter
        ds1 = DraggableScatter(im1_scatter, self)
        ds2 = DraggableScatter(im2_scatter, self)
        plt.show(block=True)


    def plot_stitched(self):
        self.image_stitched_ax.set_title(self.aligned_name)
        if self.aligned_rgb is not None:
            self.image_stitched_ax.imshow(self.aligned_rgb, cmap='gray')
            self.image_stitched_ax.axis("off")

    def update_plots(self):
        self.image1_ax.cla()
        self.image2_ax.cla()
        self.image_stitched_ax.cla()
        self.plot_stitched()
        self.plot_images()

    def run_sift(self, sift_opts, debug):
        src_pts, dst_pts, image1_masked, image2_masked, borders =align_sift(self.image1, self.image2, masks=[self.mask1, self.mask2], sift_opts=sift_opts, max_rotation_angle=None, borders=self.borders, sift_debug = debug)
        self.borders = borders
        self.src_points = src_pts
        self.src_desc = []
        self.dst_points = dst_pts
        self.dst_desc = []
        self.update_plots()

    def run_align(self):
        aligned_rgb, aligned_bgr = sift_stitch(self.image1, self.image2, self.src_points, self.dst_points, borders=self.borders)
        self.aligned_rgb = aligned_rgb
        self.aligned_bgr = aligned_bgr
        self.update_plots()

    def mask_select(self):
        # drawtype is 'box' or 'line' or 'none'
        toggle_selector.RS1 = RectangleSelector(self.image1_ax, partial(line_select_callback, self),
                                               useblit=True,
                                               button=[1, 3],  # don't use middle button
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)
        toggle_selector.RS2 = RectangleSelector(self.image2_ax, partial(line_select_callback, self),
                                               useblit=True,
                                               button=[1, 3],  # don't use middle button
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)
        self.fig.canvas.mpl_connect('key_press_event', toggle_selector)

    def mask_set(self):
        # mask center 1/3 of each image
        h, w = self.image1.shape
        mask1 = np.zeros(self.image1.shape[:2], dtype="uint8")
        cv2.rectangle(mask1, (int(self.im1_upper_corner[0]), int(self.im1_upper_corner[1])), (int(self.im1_lower_corner[0]), int(self.im1_lower_corner[1])), 255, -1)
        h, w = self.image2.shape
        mask2 = np.zeros(self.image2.shape[:2], dtype="uint8")
        cv2.rectangle(mask2, (int(self.im2_upper_corner[0]), int(self.im2_upper_corner[1])), (int(self.im2_lower_corner[0]), int(self.im2_lower_corner[1])), 255, -1)
        self.mask1 = mask1
        self.mask2 = mask2
        fm, axm = plt.subplots(2,1)
        plt.ion()
        axm[0].imshow(mask1)
        axm[1].imshow(mask2)
        plt.show(block=True)
        plt.pause(0.001)

    def clear_features(self):
        self.src_points = []
        self.dst_points = []
        self.src_desc = []
        self.dst_desc = []
        self.update_plots()

    def add_point(self):
        w, h = self.image1.shape
        rw = random.randint(int(w/4), int(w/4)*3)
        rh = random.randint(int(h / 4), int(h / 4) * 3)
        pt =  np.float32([(rw, rh)]).reshape(-1, 1, 2)
        try:
            self.dst_points = np.append(self.dst_points, pt, axis=0)
            self.src_points = np.append(self.src_points, pt, axis=0)
        except:
            self.src_points = pt
            self.dst_points = pt
        self.update_plots()

    def rm_point(self):
        self.src_points = self.src_points[:-1]
        self.dst_points = self.dst_points[:-1]
        self.update_plots()

    def save_stitched(self, save_path, save_name):
        if save_name.endswith('.tiff'):
            imname = (Path(save_path) / save_name).as_posix()
            data = self.aligned_rgb
            if self.image1_metadata['metadata']:
                source = self.image1_metadata['path']
                xscale = self.image1_metadata['metadata'].scale.X
                yscale = self.image1_metadata['metadata'].scale.Y
            metadata = dict(metadata_source=source, xscale=xscale, yscale=yscale)
            metadata = json.dumps(metadata)
            tifffile.imsave(imname, data, description=metadata)

            # with tifffile.TiffFile('microscope.tif') as tif:
            #     data = tif.asarray()
            #     metadata = tif[0].image_description
            # metadata = json.loads(metadata.decode('utf-8'))
            # print(data.shape, data.dtype, metadata['microscope'])
        else:
            imname = (Path(save_path)/save_name).as_posix()
            cv2.imwrite(filename=imname, img=self.aligned_rgb)


class DraggableScatter():

    epsilon = 40

    def __init__(self, scatter, stitcher):
        self.stitcher = stitcher
        self.scatter = scatter
        self._ind = None
        self.ax = scatter.axes
        self.canvas = self.ax.figure.canvas
        self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas.mpl_connect('close_event', self.update_stitcher)

    def get_ind_under_point(self, event):
        xy = np.asarray(self.scatter.get_offsets())
        xyt = self.ax.transData.transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]

        d = np.sqrt((xt - event.x)**2 + (yt - event.y)**2)
        ind = d.argmin()

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        if event.button != 1:
            return
        self._ind = None

    def motion_notify_callback(self, event):
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        if event.inaxes == self.stitcher.im1_scatter.axes:
            print("axes 0")
            print(f"x: {x}, y: {y}")
            print(self._ind)
            model_points = self.stitcher.src_points
            model_points[self._ind] = (event.xdata, event.ydata)
            self.stitcher.src_points = model_points
        elif event.inaxes == self.stitcher.im2_scatter.axes:
            print("axes 1")
            print(f"x: {x}, y: {y}")
            print(self._ind)
            model_points = self.stitcher.dst_points
            model_points[self._ind] = (event.xdata, event.ydata)
            self.stitcher.dst_points = model_points
        xy = np.asarray(self.scatter.get_offsets())
        xy[self._ind] = np.array([x, y])
        self.scatter.set_offsets(xy)
        self.canvas.draw_idle()
    def update_stitcher(self, event):
        self.stitcher.update_plots()


def line_select_callback(self, eclick, erelease):
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    print(" The button you used were: %s %s" % (eclick.button, erelease.button))
    if eclick.inaxes is self.image1_ax:
        self.im1_upper_corner = [x1, y1]
        self.im1_lower_corner = [x2, y2]
    elif eclick.inaxes is self.image2_ax:
        self.im2_upper_corner = [x1, y1]
        self.im2_lower_corner = [x2, y2]

def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)

def constrain_rotation(matches, kp1, kp2, max_rotation_angle=10, reset=True):
    # Filter matches based on rotation angle
    filtered_matches = []
    for m in matches:

        # Compute the absolute difference in angles
        angle_diff = np.abs(kp1[m.queryIdx].angle - kp2[m.trainIdx].angle)
        # Keep the match only if the angle difference is below the threshold
        if angle_diff < max_rotation_angle:
            # kp1[m.queryIdx].angle = 0.0
            # kp2[m.trainIdx].angle = 0.0
            filtered_matches.append(m)
    return filtered_matches, kp1, kp2

def crop_image_only_outside(img,tol=0):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img>tol
    if img.ndim==3:
        mask = mask.all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return img[row_start:row_end,col_start:col_end]

def constrain_size(matches, kp1, kp2, size_ratio=0.9):
    # Filter matches based on size ratio
    filtered_matches = []
    for m in matches:
        match_size = [kp1[m.queryIdx].size, kp2[m.trainIdx].size]
        lg_feat = max(match_size)
        sm_feat = min(match_size)
        if sm_feat >= lg_feat*size_ratio:
            filtered_matches.append(m)
    return filtered_matches

def constrain_shift(matches, kp1, kp2, pixel_error=30):
    # Get corresponding points in both images
    src_pts = np.float32([kp1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    ydiffs = dst_pts[:, 0, 1] - src_pts[:, 0, 1]
    xdiffs = dst_pts[:, 0, 0] - src_pts[:, 0, 0]
    # Further constrain matches based on deviation from median
    y_median= np.median(ydiffs)
    x_median= np.median(xdiffs)

    # Filter matches based on y-coordinate difference sign
    filtered_ymatches = []
    for m in matches:
        if (y_median-pixel_error <= (dst_pts[matches.index(m), 0, 1] - src_pts[matches.index(m), 0, 1])) and ((dst_pts[matches.index(m), 0, 1] - src_pts[matches.index(m), 0, 1]) <= y_median+pixel_error):
            filtered_ymatches.append(m)

    filtered_matches = []
    for m in filtered_ymatches:
        if (x_median-pixel_error <= (dst_pts[matches.index(m), 0, 0] - src_pts[matches.index(m), 0, 0])) and ((dst_pts[matches.index(m), 0, 0] - src_pts[matches.index(m), 0, 0]) <= x_median+pixel_error):
            filtered_matches.append(m)

    return filtered_matches


def constrain_dst_separation(matches, kp2, dist=50):
    """
    filter matches based on distance between feature points to prevent clumping

    :param matches:
    :param kp2:
    :param dist:
    :return: filtered matches
    """
    filtered_matches = []
    for i, m in enumerate(matches):
        if i == 0:
            filtered_matches.append(m)
            continue
        dst_pt = kp2[m.trainIdx].pt
        pt_distances = np.array([np.linalg.norm(np.array(dst_pt) - np.array(kp2[m1.trainIdx].pt))for m1 in filtered_matches])
        filtered_matches.append(m) if np.all(pt_distances > dist) else None

    return filtered_matches

def blend_images(image1, image2):
    # Use simple averaging to blend the images
    return cv2.addWeighted(image1, 0.5, image2, 0.5, 0)


def add_border(image, top, bottom, left, right, color=(0, 0, 0)):
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

def align_sift(image1, image2, masks=[], sift_opts={}, max_rotation_angle=10, borders=[1000, 1000, 1000, 1000], sift_debug=False):

    # add a border to image2 so that after aligning the first image overhang is not cut off
    if len(masks[0]) != 0:
        for i in range(len(masks)):
            mask = masks[i]
            if i == 0:
                image1_masked = cv2.bitwise_and(image1, image1, mask=mask)
            elif i == 1:
                image2_masked = cv2.bitwise_and(image2, image2, mask=mask)
                image2_masked = add_border(image2_masked, borders[0], borders[1], borders[2], borders[3],
                                           color=(0, 0, 0))
            else:
                print("too many masks, only first two used for masking")
    else:
        image1_masked = image1
        image2_masked = image2
        image2_masked = add_border(image2_masked, borders[0], borders[1], borders[2], borders[3], color=(0, 0, 0))

    image2 = add_border(image2, borders[0], borders[1], borders[2], borders[3], color=(0, 0, 0))

    if not (sift_opts):
        sift_opts = {
            'Sigma': 1.6,
            'Edge Threshold': 10,
            'Contrast Threshold': 0.04
        }

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # some knobs on the feature detector
    # gaussian filter used for image blurring - default = 1.6
    sift.setSigma(sift_opts['Sigma'])
    print(f"Sigma: {sift.getSigma()}")
    # # edge threshold - used to filter out edge-like features - larger equals less filtered out/more features
    sift.setEdgeThreshold(sift_opts['Edge Threshold'])
    print(f"Edge Threshold: {sift.getEdgeThreshold()}")
    # # contrast threshold - larger equals more stringent and less features
    sift.setContrastThreshold(sift_opts['Contrast Threshold'])
    print(f"Contrast Threshold: {sift.getContrastThreshold()}")

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(image1_masked, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2_masked, None)

    # Use FLANN to find matches
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Filter matches based on Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append(m)

    # Constrain matches based on rotation angle
    if max_rotation_angle != None:
        good_matches, keypoints1, keypoints2 = constrain_rotation(good_matches, keypoints1, keypoints2, max_rotation_angle, reset=True)

    # Constrain matches based on feature size
    good_matches = constrain_size(good_matches, keypoints1, keypoints2, size_ratio=0.9)

    # constrain matches based on similarity of shift direction
    good_matches = constrain_shift(good_matches, keypoints1, keypoints2)

    # Set the minimum distance between keypoints
    good_matches = constrain_dst_separation(good_matches, keypoints2, dist=50)

    # min_distance = 8  # Adjust this value based on your requirement

    # Filter matches based on spatial distance
    # filtered_matches = []
    # for match in filtered_matches:
    #     pt1 = keypoints1[match.queryIdx].pt
    #     pt2 = keypoints2[match.trainIdx].pt
    #     distance = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    #
    #     if distance > min_distance:
    #         filtered_matches.append(match)

    # Draw the constrained matches (optional)
    lineThickness = 6
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchesThickness=lineThickness)

    # plot matches
    if sift_debug == True:
        # Convert BGR images to RGB for Matplotlib
        img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
        # Display the images using Matplotlib
        plt.figure(figsize=(12, 6))
        plt.imshow(img_matches_rgb)
        plt.title('Constrained Matches')

    # Get corresponding points in both images
    src_pts = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

    dst_pts = np.float32([(pt[0][0]-borders[0], pt[0][1]-borders[0]) for pt in dst_pts]).reshape(-1, 1, 2)

    return src_pts, dst_pts, image1_masked, image2_masked, borders

def sift_stitch(image1, image2, src_pts, dst_pts, borders, blend=False, debug=False, motion='affine'):
    #undo border removal for plotting
    dst_pts = np.float32([(pt[0][0] + borders[0], pt[0][1] + borders[0]) for pt in dst_pts]).reshape(-1, 1, 2)
    # add same border that was used in sift
    image2 = add_border(image2, borders[0], borders[1], borders[2], borders[3],
                                           color=(0, 0, 0))

    # Use RANSAC to find the homography matrix
    if motion=='warp':
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp both images
        h2, w2 = image2.shape[:2]
        result1 = cv2.warpPerspective(image1, H, (w2, h2))
        result_mask = ~(result1 == 0)
        result1_extended = cv2.warpPerspective(image1, H, (w2, h2))
        result2 = image2

    if motion=='affine':
        H, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        # Warp both images
        h2, w2 = image2.shape[:2]
        result1 = cv2.warpAffine(image1, H, (w2, h2))
        result_mask = ~(result1 == 0)
        result1_extended = cv2.warpAffine(image1, H, (w2, h2))
        result2 = image2

    # Combine the warped image and the second image
    if blend == True:
        blended_overlap = blend_images(result1, result2)
        result1_extended[:blended_overlap.shape[:2][0], :blended_overlap.shape[:2][1]] = blended_overlap
    else:
        image2[result_mask] = 0
        result1_extended = image2+result1
    # Convert BGR result to RGB for Matplotlib
    result_rgb = cv2.cvtColor(result1_extended, cv2.COLOR_BGR2RGB)

    result_rgb = crop_image_only_outside(result_rgb, tol=0)
    result1_extended = crop_image_only_outside(result1_extended, tol=0)

    # Display the stitched image using Matplotlib
    if debug == True:
        plt.figure(figsize=(8, 8))
        plt.imshow(result_rgb)
        plt.title('Stitched Image')

    return result_rgb, result1_extended

def equalize_image(im, hist):
    if hist == 'STD':
        imstd = np.std(im)
        immean = np.mean(im)
        f = interp1d((immean - 2 * imstd, immean + 2 * imstd), (0, 255), fill_value=(0, 255), bounds_error=False)
        # rescale image
        shape = im.shape
        scaledim = f(im.ravel()).reshape(shape)
    elif hist == 'CLAHE':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        scaledim = clahe.apply(im)
    elif hist == 'Equalize':
        scaledim = cv2.equalizeHist(im)

    return scaledim

def open_image_selector(impath, hist='Equalize', chans=0):
    hist_types = ['Equalize', 'CLAHE', 'STD']
    if ':' in chans:
        start,stop = chans.split(':')
        chans = range(int(start),int(stop))
    else:
        chans = [int(ch) for ch in chans]
    if hist in hist_types:
        imgpath = filedialog.askopenfilename(initialdir=impath, title="Select File",
                                             filetypes=[("all files", "*.*")])
        image_name = os.path.basename(imgpath)
        # Do something with the selected file path, for example, print it
        if image_name[-4:] == '.czi':
            czi = cz.CziFile(imgpath)
            metadata = czimd.CziMetadata(imgpath)
            img_sample = cz.imread(imgpath)
            img_sample = img_sample[:, chans, :, :, :]
            _, chan_num, x, y, _ = img_sample.shape
            channels = []
            for channel in range(chan_num):
                channel_raster = img_sample[0, channel, :, :, 0]
                channels.append(channel_raster)
            if len(channels) > 1:
                im = np.zeros((x,y, 3))
                imchs = np.stack(channels, axis=2)
                im[:, :, :chan_num] = imchs
                im = im.astype(np.uint8)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            else:
                im = np.stack(channels, axis=2).astype(np.uint8)
                im = np.squeeze(im, axis=2)
            scaledim = equalize_image(im, hist)
        else:
            image = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            scaledim = image
            czi = None
            metadata = None
    else:
        raise ValueError('hist transformation type not recognized')

    return scaledim, czi, metadata, image_name, imgpath