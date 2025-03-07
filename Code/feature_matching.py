import multiprocessing
import pickle
import numpy as np
from skimage.measure import ransac
from skimage.transform import AffineTransform
import matplotlib.pyplot as plt
import os
import cv2
import constant

# device
cores_num_to_use = constant.device.cores_num_to_use

# sift
max_sift_points_num = constant.sift.feature_extraction.max_sift_points_num
sift_point_filter_flag = constant.sift.feature_extraction.sift_point_filter_flag
min_sift_points_num = constant.sift.feature_extraction.min_sift_points_num

perc_next_match = constant.sift.feature_matching.perc_next_match
do_ransac = constant.sift.feature_matching.do_ransac
ransac_max_trials = constant.sift.feature_matching.ransac_max_trials
ransac_min_samples = constant.sift.feature_matching.ransac_min_samples
ransac_residual_threshold = constant.sift.feature_matching.ransac_residual_threshold
ransac_min_points_num = constant.sift.feature_matching.ransac_min_points_num
ransac_stop_probability = constant.sift.feature_matching.ransac_stop_probability
ransac_save_ransac_result = constant.sift.feature_matching.ransac_save_ransac_result

path_ransac_result = constant.path.path_ransac_result


def feature_matching_helper(args):
    # load sift detection and feature matching
    images_path = args[0]
    matching_image_pair_idx = args[1]
    sift_data_path = args[2]
    images_name = os.listdir(images_path)
    img1_idx = matching_image_pair_idx[0]
    img2_idx = matching_image_pair_idx[1]
    kp1 = pickle.load(open("{0}/{1}/{2}".format(sift_data_path, images_name[img1_idx], 'kp.pickle'), "rb+"))
    desc1 = pickle.load(open("{0}/{1}/{2}".format(sift_data_path, images_name[img1_idx], 'desc.pickle'), "rb+"))
    kp2 = pickle.load(open("{0}/{1}/{2}".format(sift_data_path, images_name[img2_idx], 'kp.pickle'), "rb+"))
    desc2 = pickle.load(open("{0}/{1}/{2}".format(sift_data_path, images_name[img2_idx], 'desc.pickle'), "rb+"))
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    matching_points = np.zeros((len(matches), 4))
    point_index = 0

    for i, (m1, m2) in enumerate(matches):
        if m1.distance < perc_next_match * m2.distance:
            # the smaller the Euclidean distance between two eigenvectors, the higher the match.
            pt1 = kp1[m1.queryIdx]
            pt2 = kp2[m1.trainIdx]
            matching_points[point_index, 0] = pt1[0]
            matching_points[point_index, 1] = pt1[1]
            matching_points[point_index, 2] = pt2[0]
            matching_points[point_index, 3] = pt2[1]
            point_index = point_index + 1
    matching_points = matching_points[0:point_index, :]

    if point_index < min_sift_points_num:
        return False
    else:
        return matching_points


def feature_matching_helper_ransac(args):
    images_path = args[0]
    matching_image_pair_idx = args[1]
    sift_data_path = args[2]
    images_name = os.listdir(images_path)
    img1_idx = matching_image_pair_idx[0]
    img2_idx = matching_image_pair_idx[1]
    kp1 = pickle.load(open("{0}/{1}/{2}".format(sift_data_path, images_name[img1_idx], 'kp.pickle'), "rb+"))
    desc1 = pickle.load(open("{0}/{1}/{2}".format(sift_data_path, images_name[img1_idx], 'desc.pickle'), "rb+"))
    kp2 = pickle.load(open("{0}/{1}/{2}".format(sift_data_path, images_name[img2_idx], 'kp.pickle'), "rb+"))
    desc2 = pickle.load(open("{0}/{1}/{2}".format(sift_data_path, images_name[img2_idx], 'desc.pickle'), "rb+"))
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    matching_points = np.zeros((len(matches), 4))
    point_index = 0

    for i, (m1, m2) in enumerate(matches):
        if m1.distance < perc_next_match * m2.distance:
            pt1 = kp1[m1.queryIdx]
            pt2 = kp2[m1.trainIdx]
            matching_points[point_index, 0] = pt1[0]
            matching_points[point_index, 1] = pt1[1]
            matching_points[point_index, 2] = pt2[0]
            matching_points[point_index, 3] = pt2[1]
            point_index = point_index + 1
    matching_points = matching_points[0:point_index, :]

    if point_index <= 20:
        return False
    else:
        matching_points_ransac = RANSAC_matching_points(images_path, matching_points, matching_image_pair_idx)
        return matching_points_ransac


def parallel_feature_matching(images_path, matching_image_pair, sift_data_path):
    # parallel sift feature matching + RANSAC matching feature point filtering
    if do_ransac:
        args = []
        for pairwise_idx in range(len(matching_image_pair)):
            args.append((images_path, matching_image_pair[pairwise_idx], sift_data_path))
        processes = multiprocessing.Pool(cores_num_to_use)
        results = processes.map(feature_matching_helper_ransac, args)
        processes.close()
    else:
        args = []
        for pairwise_idx in range(len(matching_image_pair)):
            args.append((images_path, matching_image_pair[pairwise_idx], sift_data_path))
        processes = multiprocessing.Pool(cores_num_to_use)
        results = processes.map(feature_matching_helper, args)
        processes.close()
    results_update = [result for result in results if result is not False]
    valid_list = []
    for idx in range(len(results)):
        if results[idx] is not False:
            valid_list.append(idx)
    matching_image_pair_update = matching_image_pair[valid_list, :]
    return results_update, matching_image_pair_update


def RANSAC_matching_points(images_path, matching_points_origin, matching_image_pair):
    # RANSAC matching feature point filtering
    x0 = matching_points_origin[:, 0]
    y0 = matching_points_origin[:, 1]
    x1 = matching_points_origin[:, 2]
    y1 = matching_points_origin[:, 3]
    src = np.stack([x0, y0]).T
    dst = np.stack([x1, y1]).T

    model, inliers = ransac((src, dst), AffineTransform, min_samples=ransac_min_samples,
                            residual_threshold=ransac_residual_threshold, max_trials=ransac_max_trials,
                            stop_probability=ransac_stop_probability)
    x0_ransac = []
    y0_ransac = []
    x1_ransac = []
    y1_ransac = []
    for idx in range(len(x0)):
        if inliers[idx]:
            x0_ransac.append(x0[idx])
            y0_ransac.append(y0[idx])
            x1_ransac.append(x1[idx])
            y1_ransac.append(y1[idx])

    x0_ransac = np.array(x0_ransac)
    y0_ransac = np.array(y0_ransac)
    x1_ransac = np.array(x1_ransac)
    y1_ransac = np.array(y1_ransac)

    matching_points_ransac = np.zeros((4, len(x0_ransac)))
    matching_points_ransac[0, :] = x0_ransac
    matching_points_ransac[1, :] = y0_ransac
    matching_points_ransac[2, :] = x1_ransac
    matching_points_ransac[3, :] = y1_ransac

    if len(x0_ransac) > ransac_min_points_num:
        if ransac_save_ransac_result:
            images_name = os.listdir(images_path)
            img1 = cv2.imread("{0}/{1}".format(images_path, images_name[matching_image_pair[0]]))
            img2 = cv2.imread("{0}/{1}".format(images_path, images_name[matching_image_pair[1]]))
            fig, ax = plt.subplots()
            ax.axis('off')
            ax.imshow(np.hstack((img1, img2)), cmap='gray')
            ax.plot(np.vstack((src[inliers, 0], dst[inliers, 0] + img1.shape[1])),
                    np.vstack((src[inliers, 1], dst[inliers, 1])), '-r')
            plt.savefig('{}/{}-{}.jpg'.format(path_ransac_result, images_name[matching_image_pair[0]],
                                              images_name[matching_image_pair[1]]))
            plt.close()
        return matching_points_ransac
    else:
        return False
