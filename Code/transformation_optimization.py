from scipy.optimize import lsq_linear
import numpy as np
import os
import sys
import datetime
from GPSPhoto import gpsphoto
import pymap3d
import utils


def linear_affine_transformation_optimization(images_path, matching_image_pair, matching_points, points_num):
    print('---------------linear affine transformation optimization---------------')
    images_path_list = os.listdir(images_path)
    images_num = len(images_path_list)
    matching_image_pair_num = matching_image_pair.shape[0]

    # select reference image
    ref_img_idx = generate_ref_img(images_path)

    # solve linear optimization problem (A*X=b)
    A = np.zeros((matching_image_pair_num * points_num * 2, images_num * 6))
    b = np.zeros((matching_image_pair_num * points_num * 2))

    # optimization bounds
    lower_bounds = [-np.inf] * (6 * images_num)
    upper_bounds = [np.inf] * (6 * images_num)

    lower_bounds[ref_img_idx * 6 + 0] = 1 - 1e-4
    lower_bounds[ref_img_idx * 6 + 1] = 0 - 1e-4
    lower_bounds[ref_img_idx * 6 + 2] = 0 - 1e-4
    lower_bounds[ref_img_idx * 6 + 3] = 0 - 1e-4
    lower_bounds[ref_img_idx * 6 + 4] = 1 - 1e-4
    lower_bounds[ref_img_idx * 6 + 5] = 0 - 1e-4

    upper_bounds[ref_img_idx * 6 + 0] = 1 + 1e-4
    upper_bounds[ref_img_idx * 6 + 1] = 0 + 1e-4
    upper_bounds[ref_img_idx * 6 + 2] = 0 + 1e-4
    upper_bounds[ref_img_idx * 6 + 3] = 0 + 1e-4
    upper_bounds[ref_img_idx * 6 + 4] = 1 + 1e-4
    upper_bounds[ref_img_idx * 6 + 5] = 0 + 1e-4

    # construct A and b based on the matching point alignment error
    for pairwise_match_idx in range(matching_image_pair_num):
        img1_idx = int(matching_image_pair[pairwise_match_idx, 0])
        img2_idx = int(matching_image_pair[pairwise_match_idx, 1])
        x0 = (matching_points[pairwise_match_idx][0, :])
        y0 = (matching_points[pairwise_match_idx][1, :])
        x1 = (matching_points[pairwise_match_idx][2, :])
        y1 = (matching_points[pairwise_match_idx][3, :])
        for pt_idx in range(points_num):
            A[pairwise_match_idx * points_num * 2 + pt_idx * 2, img1_idx * 6 + 0] = x0[pt_idx]
            A[pairwise_match_idx * points_num * 2 + pt_idx * 2, img1_idx * 6 + 1] = y0[pt_idx]
            A[pairwise_match_idx * points_num * 2 + pt_idx * 2, img1_idx * 6 + 2] = 1
            A[pairwise_match_idx * points_num * 2 + pt_idx * 2, img2_idx * 6 + 0] = -x1[pt_idx]
            A[pairwise_match_idx * points_num * 2 + pt_idx * 2, img2_idx * 6 + 1] = -y1[pt_idx]
            A[pairwise_match_idx * points_num * 2 + pt_idx * 2, img2_idx * 6 + 2] = -1
            b[pairwise_match_idx * points_num * 2 + pt_idx * 2] = 0

            A[pairwise_match_idx * points_num * 2 + pt_idx * 2 + 1, img1_idx * 6 + 3] = x0[pt_idx]
            A[pairwise_match_idx * points_num * 2 + pt_idx * 2 + 1, img1_idx * 6 + 4] = y0[pt_idx]
            A[pairwise_match_idx * points_num * 2 + pt_idx * 2 + 1, img1_idx * 6 + 5] = 1
            A[pairwise_match_idx * points_num * 2 + pt_idx * 2 + 1, img2_idx * 6 + 3] = -x1[pt_idx]
            A[pairwise_match_idx * points_num * 2 + pt_idx * 2 + 1, img2_idx * 6 + 4] = -y1[pt_idx]
            A[pairwise_match_idx * points_num * 2 + pt_idx * 2 + 1, img2_idx * 6 + 5] = -1
            b[pairwise_match_idx * points_num * 2 + pt_idx * 2 + 1] = 0

    # start optimization
    start_time = datetime.datetime.now()
    res = lsq_linear(A, b, bounds=(lower_bounds, upper_bounds), method='trf', tol=1e-10, lsq_solver=None,
                     lsmr_tol=None, max_iter=50, verbose=2)
    print('linear affine transformation optimization, running time: {}'.format(datetime.datetime.now() - start_time))

    # reconstruct the affine transformation vector
    X = res.x.copy()
    affine_transformation_vec = np.zeros(images_num * 9)
    for img_idx in range(images_num):
        affine_transformation_vec[img_idx * 9 + 0] = X[img_idx * 6 + 0]
        affine_transformation_vec[img_idx * 9 + 1] = X[img_idx * 6 + 1]
        affine_transformation_vec[img_idx * 9 + 2] = X[img_idx * 6 + 2]
        affine_transformation_vec[img_idx * 9 + 3] = X[img_idx * 6 + 3]
        affine_transformation_vec[img_idx * 9 + 4] = X[img_idx * 6 + 4]
        affine_transformation_vec[img_idx * 9 + 5] = X[img_idx * 6 + 5]
        affine_transformation_vec[img_idx * 9 + 6] = 0
        affine_transformation_vec[img_idx * 9 + 7] = 0
        affine_transformation_vec[img_idx * 9 + 8] = 1

    return affine_transformation_vec


def generate_ref_img(images_path):
    min_distance_GPS = sys.maxsize
    images_name = os.listdir(images_path)
    img_GPS, _ = GPS_img(images_path)
    datum = np.argmin(np.sum((img_GPS - np.mean(img_GPS, axis=0)) ** 2, axis=1))
    mean_lat_lon = np.mean(img_GPS[:, 0:2], axis=0)

    for img_idx in range(len(images_name)):
        gps_distance = utils.get_gps_distance(img_GPS[img_idx, 0], img_GPS[img_idx, 1], mean_lat_lon[0],
                                              mean_lat_lon[1])
        if gps_distance < min_distance_GPS:
            min_distance_GPS = gps_distance
            datum = img_idx
    return datum


def GPS_img(images_path):
    images_path_list = os.listdir(images_path)
    images_num = len(images_path_list)
    img_GPS = np.zeros((images_num, 3))
    img_xyz = np.zeros((images_num, 3))

    for img_idx in range(images_num):
        # get GPS data
        gps = gpsphoto.getGPSData("{0}/{1}".format(images_path, images_path_list[img_idx]))
        p_num_Lat = gps["Latitude"]
        p_num_Lon = gps["Longitude"]
        p_num_Alt = gps["Altitude"]
        img_GPS[img_idx, :] = [p_num_Lat, p_num_Lon, p_num_Alt]

    for img_idx in range(images_num):
        # convert GPS coordinate to ENU coordinate (take mean value of GPS as reference point)
        img_xyz[img_idx, :] = pymap3d.geodetic2enu(img_GPS[img_idx, 0], img_GPS[img_idx, 1], img_GPS[img_idx, 2],
                                                   int(np.mean(img_GPS[:, 0])), int(np.mean(img_GPS[:, 1])),
                                                   int(np.mean(img_GPS[:, 2])))
    return img_GPS, img_xyz
