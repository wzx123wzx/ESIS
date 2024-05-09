import os
import cv2
import multiprocessing
import pickle
import cv_utils
import numpy as np
import RANSAC


def matching_points_origin_top(matching_points_origin, points_num):
    # select top points_num inliers
    match_num = len(matching_points_origin)
    matching_points_top = []
    for match_index in range(match_num):
        matching_points_origin_match_index = matching_points_origin[match_index]
        matching_points_top.append(np.zeros((4, points_num)))
        matching_points_top[match_index][0, :] = matching_points_origin_match_index[0, 0:points_num]
        matching_points_top[match_index][1, :] = matching_points_origin_match_index[1, 0:points_num]
        matching_points_top[match_index][2, :] = matching_points_origin_match_index[2, 0:points_num]
        matching_points_top[match_index][3, :] = matching_points_origin_match_index[3, 0:points_num]

    return matching_points_top


def feature_detecting_helper(args):
    # SIFT feature detection and save results (pickle)
    images_path = args[0]
    images_name = args[1]
    img_idx = args[2]
    max_sift_point_num = args[3]
    sift_point_filter_flag = args[4]
    SIFT_data_path = args[5]
    img = cv2.imread("{0}/{1}".format(images_path, images_name[img_idx]))
    kp, desc = cv_utils.get_SIFT_points(img, max_sift_point_num, sift_point_filter_flag)
    kp_picklable = []
    for p in kp:
        kp_picklable.append(p.pt)  # only coordinates are stored
    if not os.path.exists("{0}/{1}".format(SIFT_data_path, images_name[img_idx])):
        os.mkdir("{0}/{1}".format(SIFT_data_path, images_name[img_idx]))
    file = open("{0}/{1}/{2}".format(SIFT_data_path, images_name[img_idx], 'kp.pickle'), 'wb')
    pickle.dump(kp_picklable, file)
    file.close()
    file = open("{0}/{1}/{2}".format(SIFT_data_path, images_name[img_idx], 'desc.pickle'), 'wb')
    pickle.dump(desc, file)
    file.close()
    return


def parallel_feature_detecting(images_path, images_name, max_sift_point_num, sift_point_filter_flag, SIFT_data_path,
                               cores_num_to_use):
    # parallel SIFT detection
    args = []

    for img_idx in range(len(images_name)):
        args.append((images_path, images_name, img_idx, max_sift_point_num, sift_point_filter_flag, SIFT_data_path))
    processes = multiprocessing.Pool(cores_num_to_use)
    results = processes.map(feature_detecting_helper, args)
    processes.close()
    return results


def feature_matching_helper(args):
    # load SIFT detection and feature matching
    images_path = args[0]
    matching_image_pair_idx = args[1]
    SIFT_data_path = args[2]
    perc_next_match = args[3]
    min_points_num = args[4]
    images_name = os.listdir(images_path)
    img_idx1 = matching_image_pair_idx[0]
    img_idx2 = matching_image_pair_idx[1]
    kp1 = pickle.load(open("{0}/{1}/{2}".format(SIFT_data_path, images_name[img_idx1], 'kp.pickle'), "rb+"))
    desc1 = pickle.load(open("{0}/{1}/{2}".format(SIFT_data_path, images_name[img_idx1], 'desc.pickle'), "rb+"))
    kp2 = pickle.load(open("{0}/{1}/{2}".format(SIFT_data_path, images_name[img_idx2], 'kp.pickle'), "rb+"))
    desc2 = pickle.load(open("{0}/{1}/{2}".format(SIFT_data_path, images_name[img_idx2], 'desc.pickle'), "rb+"))
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    matching_points = np.zeros((len(matches), 4))
    point_index = 0

    for i, (m1, m2) in enumerate(matches):
        if m1.distance < perc_next_match * m2.distance:
            # the smaller the Euclidean distance between two eigenvectors,
            # the higher the match.
            pt1 = kp1[m1.queryIdx]
            pt2 = kp2[m1.trainIdx]
            matching_points[point_index, 0] = pt1[0]
            matching_points[point_index, 1] = pt1[1]
            matching_points[point_index, 2] = pt2[0]
            matching_points[point_index, 3] = pt2[1]
            point_index = point_index + 1
    matching_points = matching_points[0:point_index, :]
    if point_index < min_points_num:
        return False
    else:
        return matching_points


def feature_matching_helper_ransac(args):
    images_path = args[0]
    matching_image_pair_idx = args[1]
    SIFT_data_path = args[2]
    perc_next_match = args[3]
    ransac_min_samples = args[4]
    ransac_residual_threshold = args[5]
    ransac_max_trials = args[6]
    min_points_num = args[7]
    ransac_stop_probability = args[8]
    ransac_save_ransac_result = args[9]
    path_ransac_results = args[10]
    images_name = os.listdir(images_path)
    img_idx1 = matching_image_pair_idx[0]
    img_idx2 = matching_image_pair_idx[1]
    kp1 = pickle.load(open("{0}/{1}/{2}".format(SIFT_data_path, images_name[img_idx1], 'kp.pickle'), "rb+"))
    desc1 = pickle.load(open("{0}/{1}/{2}".format(SIFT_data_path, images_name[img_idx1], 'desc.pickle'), "rb+"))
    kp2 = pickle.load(open("{0}/{1}/{2}".format(SIFT_data_path, images_name[img_idx2], 'kp.pickle'), "rb+"))
    desc2 = pickle.load(open("{0}/{1}/{2}".format(SIFT_data_path, images_name[img_idx2], 'desc.pickle'), "rb+"))
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
        matching_points_ransac = RANSAC.RANSAC_matching_points(images_path, matching_points,
                                                               min_samples=ransac_min_samples,
                                                               residual_threshold=ransac_residual_threshold,
                                                               max_trials=ransac_max_trials,
                                                               matching_image_pair=matching_image_pair_idx,
                                                               save_ransac_result=ransac_save_ransac_result,
                                                               min_points_num=min_points_num,
                                                               dir_ransac_results=path_ransac_results,
                                                               stop_probability=ransac_stop_probability)
        return matching_points_ransac


def parallel_feature_matching(images_path, matching_image_pair, SIFT_data_path, perc_next_match,
                              cores_num_to_use,
                              do_ransac, ransac_min_samples, ransac_residual_threshold, ransac_max_trials,
                              min_sift_points_num, min_ransac_points_num, ransac_stop_probability,
                              ransac_save_ransac_result, path_ransac_results):
    # parallel SIFT feature matching + RANSAC feature points filtering
    if do_ransac:
        args = []
        for pairwise_idx in range(len(matching_image_pair)):
            args.append((images_path, matching_image_pair[pairwise_idx], SIFT_data_path, perc_next_match,
                         ransac_min_samples, ransac_residual_threshold, ransac_max_trials, min_ransac_points_num,
                         ransac_stop_probability, ransac_save_ransac_result, path_ransac_results))
        processes = multiprocessing.Pool(cores_num_to_use)
        results = processes.map(feature_matching_helper_ransac, args)
        processes.close()
    else:
        args = []
        for pairwise_idx in range(len(matching_image_pair)):
            args.append((images_path, matching_image_pair[pairwise_idx], SIFT_data_path, perc_next_match,
                         min_sift_points_num))
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
