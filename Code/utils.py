import os
import sys
import numpy as np
import cv2
import math


def white_bck(input_mosaic, mosaic_valid):
    output_mosaic = 255 * np.ones(input_mosaic.shape, dtype=np.uint8)
    output_mosaic[mosaic_valid] = input_mosaic[mosaic_valid]
    return output_mosaic


def get_gps_distance(lat1, lon1, lat2, lon2):
    if lat1 == lat2 and lon1 == lon2:
        return sys.maxsize
    else:
        phi1 = math.radians(lat1)
        lambda1 = math.radians(lon1)
        phi2 = math.radians(lat2)
        lambda2 = math.radians(lon2)
        r = 6371e3

        a = math.sin((phi2 - phi1) / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * (
                math.sin((lambda2 - lambda1) / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return r * c


def get_sift_points(img, max_sift_number, filter_flag):
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(img, None)

    if filter_flag is not True:
        kp = kp[: min(len(kp), max_sift_number)]
        desc = desc[: min(len(kp), max_sift_number)]
    elif filter_flag is True:
        # sort according to the response value
        sorted_id = sorted(range(len(kp)), key=lambda x: -kp[x].response)
        kp = list(kp)
        kp = (sorted(kp, key=lambda x: -x.response))
        desc_new = desc.copy()
        for i in range(len(kp)):
            desc_new[i, :] = desc[sorted_id[i], :]
        desc = desc_new
        kp = kp[: min(len(kp), max_sift_number)]
        desc = desc[: min(len(kp), max_sift_number)]

    return kp, desc


def calculate_projection_RMSE(transform_vec, matching_image_pair, matching_points):
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
        H1 = transform_vec[img_idx_1 * 9:img_idx_1 * 9 + 9]
        H1 = H1.reshape(3, 3)
        H2 = transform_vec[img_idx_2 * 9:img_idx_2 * 9 + 9]
        H2 = H2.reshape(3, 3)
        warped_matching_points_img1 = np.matmul(H1, matching_points_img1)
        warped_matching_points_img2 = np.matmul(H2, matching_points_img2)
        warped_matching_points_img1 = warped_matching_points_img1 / warped_matching_points_img1[2, :]
        warped_matching_points_img2 = warped_matching_points_img2 / warped_matching_points_img2[2, :]
        residuals.append((warped_matching_points_img1[0, :] - warped_matching_points_img2[0, :]) ** 2)
        residuals.append((warped_matching_points_img1[1, :] - warped_matching_points_img2[1, :]) ** 2)

    return math.sqrt(np.mean(residuals))


def index_image(images_path, img_name):
    # return image index from image name
    images_path_list = os.listdir(images_path)
    img_idx = 0

    for path_idx in range(len(images_path_list)):
        if images_path_list[path_idx] == img_name:
            img_idx = path_idx
            break

    return img_idx
