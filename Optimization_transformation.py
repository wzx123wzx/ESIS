from scipy.optimize import lsq_linear
import numpy as np
import os
import sys
import datetime
from GPSPhoto import gpsphoto
from scipy.optimize import least_squares
import pymap3d
import cv_utils


def lin_optimization_affine(images_path, matching_image_pair, matching_points, points_num):
    print('---------------linear affine transformation optimization---------------')
    images_path_list = os.listdir(images_path)
    picture_num = len(images_path_list)
    matching_image_pair_num = len(matching_image_pair)

    ref_img_idx = generate_ref_img(images_path)
    # ref_img_idx = 5
    # ref_img_idx = generate_ref_img_v1(images_path)
    A_tf = np.zeros((matching_image_pair_num * points_num * 2, picture_num * 6))
    b_tf = np.zeros((matching_image_pair_num * points_num * 2))
    lb_tf = [-np.inf] * (6 * picture_num)
    ub_tf = [np.inf] * (6 * picture_num)

    lb_tf[ref_img_idx * 6 + 0] = 1 - 1e-4
    lb_tf[ref_img_idx * 6 + 1] = 0 - 1e-4
    lb_tf[ref_img_idx * 6 + 2] = 0 - 1e-4
    lb_tf[ref_img_idx * 6 + 3] = 0 - 1e-4
    lb_tf[ref_img_idx * 6 + 4] = 1 - 1e-4
    lb_tf[ref_img_idx * 6 + 5] = 0 - 1e-4

    ub_tf[ref_img_idx * 6 + 0] = 1 + 1e-4
    ub_tf[ref_img_idx * 6 + 1] = 0 + 1e-4
    ub_tf[ref_img_idx * 6 + 2] = 0 + 1e-4
    ub_tf[ref_img_idx * 6 + 3] = 0 + 1e-4
    ub_tf[ref_img_idx * 6 + 4] = 1 + 1e-4
    ub_tf[ref_img_idx * 6 + 5] = 0 + 1e-4
    for match_index in range(matching_image_pair_num):
        p_num_1 = int(matching_image_pair[match_index, 0])
        p_num_2 = int(matching_image_pair[match_index, 1])
        x0 = (matching_points[match_index][0, :])
        y0 = (matching_points[match_index][1, :])
        x1 = (matching_points[match_index][2, :])
        y1 = (matching_points[match_index][3, :])
        for pt_idx in range(points_num):
            A_tf[match_index * points_num * 2 + pt_idx * 2, p_num_1 * 6 + 0] = x0[pt_idx]
            A_tf[match_index * points_num * 2 + pt_idx * 2, p_num_1 * 6 + 1] = y0[pt_idx]
            A_tf[match_index * points_num * 2 + pt_idx * 2, p_num_1 * 6 + 2] = 1
            A_tf[match_index * points_num * 2 + pt_idx * 2, p_num_2 * 6 + 0] = -x1[pt_idx]
            A_tf[match_index * points_num * 2 + pt_idx * 2, p_num_2 * 6 + 1] = -y1[pt_idx]
            A_tf[match_index * points_num * 2 + pt_idx * 2, p_num_2 * 6 + 2] = -1
            b_tf[match_index * points_num * 2 + pt_idx * 2] = 0

            A_tf[match_index * points_num * 2 + pt_idx * 2 + 1, p_num_1 * 6 + 3] = x0[pt_idx]
            A_tf[match_index * points_num * 2 + pt_idx * 2 + 1, p_num_1 * 6 + 4] = y0[pt_idx]
            A_tf[match_index * points_num * 2 + pt_idx * 2 + 1, p_num_1 * 6 + 5] = 1
            A_tf[match_index * points_num * 2 + pt_idx * 2 + 1, p_num_2 * 6 + 3] = -x1[pt_idx]
            A_tf[match_index * points_num * 2 + pt_idx * 2 + 1, p_num_2 * 6 + 4] = -y1[pt_idx]
            A_tf[match_index * points_num * 2 + pt_idx * 2 + 1, p_num_2 * 6 + 5] = -1
            b_tf[match_index * points_num * 2 + pt_idx * 2 + 1] = 0
    start_time = datetime.datetime.now()
    tf_aff_obj = lsq_linear(A_tf, b_tf, bounds=(lb_tf, ub_tf), method='trf', tol=1e-10, lsq_solver=None, lsmr_tol=None,
                            max_iter=50, verbose=2)
    print('Linear affine transformation optimization, running time: {}'.format(datetime.datetime.now() - start_time))
    tf_vec = tf_aff_obj.x.copy()
    res_vec = abs(np.dot(A_tf, tf_vec))
    affine_transformation_vec = np.zeros(picture_num * 9)
    for p_num in range(picture_num):
        affine_transformation_vec[p_num * 9 + 0] = tf_vec[p_num * 6 + 0]
        affine_transformation_vec[p_num * 9 + 1] = tf_vec[p_num * 6 + 1]
        affine_transformation_vec[p_num * 9 + 2] = tf_vec[p_num * 6 + 2]
        affine_transformation_vec[p_num * 9 + 3] = tf_vec[p_num * 6 + 3]
        affine_transformation_vec[p_num * 9 + 4] = tf_vec[p_num * 6 + 4]
        affine_transformation_vec[p_num * 9 + 5] = tf_vec[p_num * 6 + 5]
        affine_transformation_vec[p_num * 9 + 6] = 0
        affine_transformation_vec[p_num * 9 + 7] = 0
        affine_transformation_vec[p_num * 9 + 8] = 1
    return affine_transformation_vec


def nolin_optimization_projective(images_name, initialization, m_aff, m_tr, m_proj, matching_image_pair,
                                  matching_points, points_num):
    num_images = len(images_name)
    H_0 = initialization
    lower_bounds = [-np.inf] * (9 * num_images)
    upper_bounds = [np.inf] * (9 * num_images)
    for img_idx in range(num_images):
        lower_bounds[0 + 9 * img_idx] = H_0[0 + 9 * img_idx] - m_aff
        lower_bounds[1 + 9 * img_idx] = H_0[1 + 9 * img_idx] - m_aff
        lower_bounds[2 + 9 * img_idx] = H_0[2 + 9 * img_idx] - m_tr
        lower_bounds[3 + 9 * img_idx] = H_0[3 + 9 * img_idx] - m_aff
        lower_bounds[4 + 9 * img_idx] = H_0[4 + 9 * img_idx] - m_aff
        lower_bounds[5 + 9 * img_idx] = H_0[5 + 9 * img_idx] - m_tr
        lower_bounds[6 + 9 * img_idx] = H_0[6 + 9 * img_idx] - m_proj
        lower_bounds[7 + 9 * img_idx] = H_0[7 + 9 * img_idx] - m_proj
        lower_bounds[8 + 9 * img_idx] = H_0[8 + 9 * img_idx] - m_proj

        upper_bounds[0 + 9 * img_idx] = H_0[0 + 9 * img_idx] + m_aff
        upper_bounds[1 + 9 * img_idx] = H_0[1 + 9 * img_idx] + m_aff
        upper_bounds[2 + 9 * img_idx] = H_0[2 + 9 * img_idx] + m_tr
        upper_bounds[3 + 9 * img_idx] = H_0[3 + 9 * img_idx] + m_aff
        upper_bounds[4 + 9 * img_idx] = H_0[4 + 9 * img_idx] + m_aff
        upper_bounds[5 + 9 * img_idx] = H_0[5 + 9 * img_idx] + m_tr
        upper_bounds[6 + 9 * img_idx] = H_0[6 + 9 * img_idx] + m_proj
        upper_bounds[7 + 9 * img_idx] = H_0[7 + 9 * img_idx] + m_proj
        upper_bounds[8 + 9 * img_idx] = H_0[8 + 9 * img_idx] + m_proj
    start_time = datetime.datetime.now()
    res = least_squares(residuals_function_proj, H_0, bounds=(lower_bounds, upper_bounds), verbose=2,
                        ftol=5e-2, args=(matching_image_pair, matching_points, points_num))
    print('Nonlinear projective transformation optimization, running time: {}'
          .format(datetime.datetime.now() - start_time))
    projective_transformation_vec = res.x
    return projective_transformation_vec


def residuals_function_proj(X, matching_image_pair, matching_points, points_num):
    # projective transformation error function
    residuals = []
    for pairwise_match_idx in range(matching_image_pair.shape[0]):
        img_idx_1 = int(matching_image_pair[pairwise_match_idx, 0])
        img_idx_2 = int(matching_image_pair[pairwise_match_idx, 1])
        x0 = (matching_points[pairwise_match_idx][0, :])
        y0 = (matching_points[pairwise_match_idx][1, :])
        x1 = (matching_points[pairwise_match_idx][2, :])
        y1 = (matching_points[pairwise_match_idx][3, :])
        matching_points_img1 = np.stack([x0, y0, np.ones(len(x0))])
        matching_points_img2 = np.stack([x1, y1, np.ones(len(x0))])
        H1 = X[img_idx_1 * 9:img_idx_1 * 9 + 9]
        H1 = H1.reshape(3, 3)
        H2 = X[img_idx_2 * 9:img_idx_2 * 9 + 9]
        H2 = H2.reshape(3, 3)
        warped_matching_points_img1 = np.matmul(H1, matching_points_img1)
        warped_matching_points_img2 = np.matmul(H2, matching_points_img2)
        for pt_idx in range(points_num):
            warped_pt1 = [warped_matching_points_img1[0, pt_idx], warped_matching_points_img1[1, pt_idx]]
            warped_pt2 = [warped_matching_points_img2[0, pt_idx], warped_matching_points_img2[1, pt_idx]]
            residuals.append((warped_pt1[0] - warped_pt2[0]) ** 2 + (warped_pt1[1] - warped_pt2[1]) ** 2)
    return residuals


def get_jacobians(X, matching_image_pair, matching_points, points_num):
    jacobians = []
    num_images = int(len(X) / 9)
    for pairwise_match_idx in range(matching_image_pair.shape[0]):
        img_idx1 = matching_image_pair[pairwise_match_idx, 0]
        img_idx2 = matching_image_pair[pairwise_match_idx, 1]
        H1 = X[img_idx1 * 9:img_idx1 * 9 + 9]
        H1 = H1.reshape(3, 3)
        H2 = X[img_idx2 * 9:img_idx2 * 9 + 9]
        H2 = H2.reshape(3, 3)
        x0 = (matching_points[pairwise_match_idx][0, :])
        y0 = (matching_points[pairwise_match_idx][1, :])
        x1 = (matching_points[pairwise_match_idx][2, :])
        y1 = (matching_points[pairwise_match_idx][3, :])
        matching_points_img1 = np.stack([x0, y0, np.ones(len(x0))])
        matching_points_img2 = np.stack([x1, y1, np.ones(len(x1))])
        warped_matching_points_img1 = np.matmul(H1, matching_points_img1)
        warped_matching_points_img2 = np.matmul(H2, matching_points_img2)
        for pt_idx in range(points_num):
            warped_pt1_origin = warped_matching_points_img1[:, pt_idx]
            warped_pt2_origin = warped_matching_points_img2[:, pt_idx]
            warped_pt1_div = warped_pt1_origin / warped_pt1_origin[2]
            warped_pt2_div = warped_pt2_origin / warped_pt2_origin[2]
            diff_x = warped_pt1_div[0] - warped_pt2_div[0]
            diff_y = warped_pt1_div[1] - warped_pt2_div[1]
            # jacobians for Homo1
            # H11
            rond_x = x0[pt_idx] / warped_pt1_origin[2]
            rond_y = 0
            jac_H1_11 = 2 * diff_x * rond_x + 2 * diff_y * rond_y
            # H12
            rond_x = y0[pt_idx] / warped_pt1_origin[2]
            rond_y = 0
            jac_H1_12 = 2 * diff_x * rond_x + 2 * diff_y * rond_y
            # H13
            rond_x = 1 / warped_pt1_origin[2]
            rond_y = 0
            jac_H1_13 = 2 * diff_x * rond_x + 2 * diff_y * rond_y

            # H21
            rond_x = 0
            rond_y = x0[pt_idx] / warped_pt1_origin[2]
            jac_H1_21 = 2 * diff_x * rond_x + 2 * diff_y * rond_y
            # H22
            rond_x = 0
            rond_y = y0[pt_idx] / warped_pt1_origin[2]
            jac_H1_22 = 2 * diff_x * rond_x + 2 * diff_y * rond_y
            # H23
            rond_x = 0
            rond_y = 1 / warped_pt1_origin[2]
            jac_H1_23 = 2 * diff_x * rond_x + 2 * diff_y * rond_y

            # H31
            rond_x = -x0[pt_idx] * warped_pt1_origin[0] / (warped_pt1_origin[2] ** 2)
            rond_y = -x0[pt_idx] * warped_pt1_origin[1] / (warped_pt1_origin[2] ** 2)
            jac_H1_31 = 2 * diff_x * rond_x + 2 * diff_y * rond_y
            # H32
            rond_x = -y0[pt_idx] * warped_pt1_origin[0] / (warped_pt1_origin[2] ** 2)
            rond_y = -y0[pt_idx] * warped_pt1_origin[1] / (warped_pt1_origin[2] ** 2)
            jac_H1_32 = 2 * diff_x * rond_x + 2 * diff_y * rond_y
            # H33
            rond_x = -1 * warped_pt1_origin[0] / (warped_pt1_origin[2] ** 2)
            rond_y = -1 * warped_pt1_origin[1] / (warped_pt1_origin[2] ** 2)
            jac_H1_33 = 2 * diff_x * rond_x + 2 * diff_y * rond_y

            # jacobians for Homo2
            # H11
            rond_x = x1[pt_idx] / warped_pt2_origin[2]
            rond_y = 0
            jac_H2_11 = 2 * diff_x * rond_x + 2 * diff_y * rond_y
            # H12
            rond_x = y1[pt_idx] / warped_pt2_origin[2]
            rond_y = 0
            jac_H2_12 = 2 * diff_x * rond_x + 2 * diff_y * rond_y
            # H13
            rond_x = 1 / warped_pt2_origin[2]
            rond_y = 0
            jac_H2_13 = 2 * diff_x * rond_x + 2 * diff_y * rond_y

            # H21
            rond_x = 0
            rond_y = x1[pt_idx] / warped_pt2_origin[2]
            jac_H2_21 = 2 * diff_x * rond_x + 2 * diff_y * rond_y
            # H22
            rond_x = 0
            rond_y = y1[pt_idx] / warped_pt2_origin[2]
            jac_H2_22 = 2 * diff_x * rond_x + 2 * diff_y * rond_y
            # H23
            rond_x = 0
            rond_y = 1 / warped_pt2_origin[2]
            jac_H2_23 = 2 * diff_x * rond_x + 2 * diff_y * rond_y

            # H31
            rond_x = -x1[pt_idx] * warped_pt2_origin[0] / (warped_pt2_origin[2] ** 2)
            rond_y = -x1[pt_idx] * warped_pt2_origin[1] / (warped_pt2_origin[2] ** 2)
            jac_H2_31 = 2 * diff_x * rond_x + 2 * diff_y * rond_y
            # H32
            rond_x = -y1[pt_idx] * warped_pt2_origin[0] / (warped_pt2_origin[2] ** 2)
            rond_y = -y1[pt_idx] * warped_pt2_origin[1] / (warped_pt2_origin[2] ** 2)
            jac_H2_32 = 2 * diff_x * rond_x + 2 * diff_y * rond_y
            # H33
            rond_x = -1 * warped_pt2_origin[0] / (warped_pt2_origin[2] ** 2)
            rond_y = -1 * warped_pt2_origin[1] / (warped_pt2_origin[2] ** 2)
            jac_H2_33 = 2 * diff_x * rond_x + 2 * diff_y * rond_y

            jac = np.zeros(9 * num_images)
            jac[img_idx1 * 9 + 0] = jac_H1_11
            jac[img_idx1 * 9 + 1] = jac_H1_12
            jac[img_idx1 * 9 + 2] = jac_H1_13
            jac[img_idx1 * 9 + 3] = jac_H1_21
            jac[img_idx1 * 9 + 4] = jac_H1_22
            jac[img_idx1 * 9 + 5] = jac_H1_23
            jac[img_idx1 * 9 + 6] = jac_H1_31
            jac[img_idx1 * 9 + 7] = jac_H1_32
            jac[img_idx1 * 9 + 8] = jac_H1_33

            jac[img_idx2 * 9 + 0] = jac_H2_11
            jac[img_idx2 * 9 + 1] = jac_H2_12
            jac[img_idx2 * 9 + 2] = jac_H2_13
            jac[img_idx2 * 9 + 3] = jac_H2_21
            jac[img_idx2 * 9 + 4] = jac_H2_22
            jac[img_idx2 * 9 + 5] = jac_H2_23
            jac[img_idx2 * 9 + 6] = jac_H2_31
            jac[img_idx2 * 9 + 7] = jac_H2_32
            jac[img_idx2 * 9 + 8] = jac_H2_33

            jacobians.append(jac)
    return np.array(jacobians)


def generate_ref_img(images_path):
    min_distance_GPS = sys.maxsize
    images_name = os.listdir(images_path)
    img_GPS, img_xyz = GPS_img(images_path)
    datum = np.argmin(np.sum((img_GPS - np.mean(img_GPS, axis=0)) ** 2, axis=1))
    mean_lat_lon = np.mean(img_GPS[:, 0:2], axis=0)
    for img_idx in range(len(images_name)):
        gps_distance = cv_utils.get_gps_distance(img_GPS[img_idx, 0], img_GPS[img_idx, 1], mean_lat_lon[0],
                                                 mean_lat_lon[1])
        if gps_distance < min_distance_GPS:
            min_distance_GPS = gps_distance
            datum = img_idx
    return datum


def GPS_img(images_path):
    images_path_list = os.listdir(images_path)
    picture_num = len(images_path_list)
    img_GPS = np.zeros((picture_num, 3))
    img_xyz = np.zeros((picture_num, 3))
    for p_num in range(picture_num):
        gps = gpsphoto.getGPSData("{0}/{1}".format(images_path, images_path_list[p_num]))
        p_num_Lat = gps["Latitude"]
        p_num_Lon = gps["Longitude"]
        p_num_Alt = gps["Altitude"]
        img_GPS[p_num, :] = [p_num_Lat, p_num_Lon, p_num_Alt]
    for p_num in range(picture_num):
        # convert GPS coordinate to ENU coordinate (take mean value of GPS as reference point)
        img_xyz[p_num, :] = pymap3d.geodetic2enu(img_GPS[p_num, 0], img_GPS[p_num, 1], img_GPS[p_num, 2],
                                                 int(np.mean(img_GPS[:, 0])), int(np.mean(img_GPS[:, 1])),
                                                 int(np.mean(img_GPS[:, 2])))
    return img_GPS, img_xyz


def nolin_optimization_projective_rs_2016(images_path, images_name, initialization, matching_image_pair,
                                          matching_points, points_num, omega=4000):
    # implementation of Mosaicking of Unmanned Aerial Vehicle Imagery in
    # the Absence of Camera Poses
    num_images = len(images_name)
    ref_img_idx = generate_ref_img(images_path)

    lower_bounds = [-np.inf] * (9 * num_images)
    upper_bounds = [np.inf] * (9 * num_images)
    lower_bounds[0 + 9 * ref_img_idx] = 1 - 1e-4
    lower_bounds[1 + 9 * ref_img_idx] = -1e-4
    lower_bounds[2 + 9 * ref_img_idx] = -1e-4
    lower_bounds[3 + 9 * ref_img_idx] = -1e-4
    lower_bounds[4 + 9 * ref_img_idx] = 1 - 1e-4
    lower_bounds[5 + 9 * ref_img_idx] = -1e-4
    lower_bounds[6 + 9 * ref_img_idx] = -1e-4
    lower_bounds[7 + 9 * ref_img_idx] = -1e-4
    lower_bounds[8 + 9 * ref_img_idx] = 1 - 1e-4

    upper_bounds[0 + 9 * ref_img_idx] = 1 + 1e-4
    upper_bounds[1 + 9 * ref_img_idx] = 1e-4
    upper_bounds[2 + 9 * ref_img_idx] = 1e-4
    upper_bounds[3 + 9 * ref_img_idx] = 1e-4
    upper_bounds[4 + 9 * ref_img_idx] = 1 + 1e-4
    upper_bounds[5 + 9 * ref_img_idx] = 1e-4
    upper_bounds[6 + 9 * ref_img_idx] = 1e-4
    upper_bounds[7 + 9 * ref_img_idx] = 1e-4
    upper_bounds[8 + 9 * ref_img_idx] = 1 + 1e-4

    start_time = datetime.datetime.now()

    res = least_squares(residuals_function_proj_rs_2016, initialization, bounds=(lower_bounds, upper_bounds), verbose=2,
                        ftol=1e-2, args=(num_images, matching_image_pair, matching_points, points_num, omega))
    print('Nonlinear projective transformation optimization, running time: {}'
          .format(datetime.datetime.now() - start_time))
    projective_transformation_vec = res.x
    return projective_transformation_vec


def residuals_function_proj_rs_2016(X, num_images, matching_image_pair, matching_points, points_num, omega):
    # projective transformation error function
    residuals = []
    for pairwise_match_idx in range(matching_image_pair.shape[0]):
        img_idx_1 = int(matching_image_pair[pairwise_match_idx, 0])
        img_idx_2 = int(matching_image_pair[pairwise_match_idx, 1])
        x0 = (matching_points[pairwise_match_idx][0, :])
        y0 = (matching_points[pairwise_match_idx][1, :])
        x1 = (matching_points[pairwise_match_idx][2, :])
        y1 = (matching_points[pairwise_match_idx][3, :])
        matching_points_img1 = np.stack([x0, y0, np.ones(len(x0))])
        matching_points_img2 = np.stack([x1, y1, np.ones(len(x0))])
        H1 = X[img_idx_1 * 9:img_idx_1 * 9 + 9]
        H1 = H1.reshape(3, 3)
        H2 = X[img_idx_2 * 9:img_idx_2 * 9 + 9]
        H2 = H2.reshape(3, 3)
        warped_matching_points_img1 = np.matmul(H1, matching_points_img1)
        warped_matching_points_img2 = np.matmul(H2, matching_points_img2)
        for pt_idx in range(points_num):
            warped_pt1 = [warped_matching_points_img1[0, pt_idx], warped_matching_points_img1[1, pt_idx]]
            warped_pt2 = [warped_matching_points_img2[0, pt_idx], warped_matching_points_img2[1, pt_idx]]
            residuals.append((warped_pt1[0] - warped_pt2[0]) ** 2 + (warped_pt1[1] - warped_pt2[1]) ** 2)
        for img_idx in range(num_images):
            H = X[img_idx * 9:img_idx * 9 + 9]
            H = H.reshape(3, 3)
            reg_idx = (H[0, 0] * H[0, 1] + H[1, 0] * H[1, 1]) ** 2 + (H[0, 0] ** 2 + H[1, 0] ** 2 - 1) ** 2 \
                      + (H[0, 1] ** 2 + H[1, 1] ** 2 - 1) ** 2 + (H[2, 0] ** 2 + H[2, 1]) ** 2
            residuals.append(omega * reg_idx)
    return residuals
