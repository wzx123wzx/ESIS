import cv2
import os
import sys
import numpy as np
import multiprocessing
import copy
import constant

# device
cores_num_to_use = constant.device.cores_num_to_use


def calculate_warped_img_helper(args):
    img_idx, mosaic_h, mosaic_w, tf_vec, img, isimg = args
    H = tf_vec[img_idx * 9:img_idx * 9 + 9]
    H = H.reshape(3, 3)
    warped_img_idx = cv2.warpPerspective(img, H, (mosaic_w, mosaic_h))
    if isimg:
        warped_img_idx = warped_img_idx.astype(np.uint8)
    else:
        warped_img_idx = warped_img_idx == 1
    return warped_img_idx


def parallel_calculate_warped_img(images_path, images_name, stitching_img_corner_coords, img_h, img_w, mosaic_h,
                                  mosaic_w):
    num_images = len(images_name)

    # generate new transformation vector (ensure minimum of corner coordinates of warped images is 0)
    stitching_transform_vec = np.zeros(num_images * 9)
    pts1 = np.float32([[0, 0], [800, 0], [800, 600], [0, 600]])
    for img_idx in range(num_images):
        pts2 = np.float32([
            [stitching_img_corner_coords[images_name[img_idx]]['UL'][0],
             stitching_img_corner_coords[images_name[img_idx]]['UL'][1]],
            [stitching_img_corner_coords[images_name[img_idx]]['UR'][0],
             stitching_img_corner_coords[images_name[img_idx]]['UR'][1]],
            [stitching_img_corner_coords[images_name[img_idx]]['LR'][0],
             stitching_img_corner_coords[images_name[img_idx]]['LR'][1]],
            [stitching_img_corner_coords[images_name[img_idx]]['LL'][0],
             stitching_img_corner_coords[images_name[img_idx]]['LL'][1]],
        ]
        )
        H = cv2.findHomography(pts1, pts2)[0]
        stitching_transform_vec[img_idx * 9:img_idx * 9 + 9] = H.flatten()

    args = []
    for img_idx in range(len(images_name)):
        args.append((img_idx, mosaic_h, mosaic_w, stitching_transform_vec,
                     cv2.imread("{0}/{1}".format(images_path, images_name[img_idx])), True))
    processes = multiprocessing.Pool(cores_num_to_use)
    warped_img = processes.map(calculate_warped_img_helper, args)
    processes.close()

    args = []
    for img_idx in range(len(images_name)):
        args.append((img_idx, mosaic_h, mosaic_w, stitching_transform_vec, np.ones((img_h, img_w)), False))
    processes = multiprocessing.Pool(cores_num_to_use)
    warped_img_valid = processes.map(calculate_warped_img_helper, args)
    processes.close()

    return stitching_transform_vec, np.array(warped_img), np.array(warped_img_valid)


def get_mosaic_size(images_path, transform_vec):
    # calculate corner coordinates of warped images and define size of mosaic
    num_images = int(len(transform_vec) / 9)
    images_name = os.listdir(images_path)
    img = cv2.imread("{0}/{1}".format(images_path, images_name[0]))
    x = img.shape[1]
    y = img.shape[0]
    UL_ref = [0, 0, 1]
    UR_ref = [x, 0, 1]
    LR_ref = [x, y, 1]
    LL_ref = [0, y, 1]
    img_corner_coords = {}
    for img_idx in range(num_images):
        H = transform_vec[img_idx * 9:img_idx * 9 + 9]
        H = H.reshape(3, 3)
        UL = np.matmul(H, UL_ref)
        UL = UL / UL[2]
        UL = UL[:2]

        UR = np.matmul(H, UR_ref)
        UR = UR / UR[2]
        UR = UR[:2]

        LR = np.matmul(H, LR_ref)
        LR = LR / LR[2]
        LR = LR[:2]

        LL = np.matmul(H, LL_ref)
        LL = LL / LL[2]
        LL = LL[:2]

        img_corner_coords[images_name[img_idx]] = {}
        img_corner_coords[images_name[img_idx]]['UL'] = UL
        img_corner_coords[images_name[img_idx]]['UR'] = UR
        img_corner_coords[images_name[img_idx]]['LR'] = LR
        img_corner_coords[images_name[img_idx]]['LL'] = LL

    min_x = sys.maxsize
    max_x = 0
    min_y = sys.maxsize
    max_y = 0

    for i in img_corner_coords:
        img_coords = img_corner_coords[i]

        if img_coords["UL"][0] > max_x:
            max_x = img_coords["UL"][0]
        if img_coords["UR"][0] > max_x:
            max_x = img_coords["UR"][0]
        if img_coords["LL"][0] > max_x:
            max_x = img_coords["LL"][0]
        if img_coords["LR"][0] > max_x:
            max_x = img_coords["LR"][0]

        if img_coords["UL"][1] > max_y:
            max_y = img_coords["UL"][1]
        if img_coords["UR"][1] > max_y:
            max_y = img_coords["UR"][1]
        if img_coords["LL"][1] > max_y:
            max_y = img_coords["LL"][1]
        if img_coords["LR"][1] > max_y:
            max_y = img_coords["LR"][1]

        if img_coords["UL"][0] < min_x:
            min_x = img_coords["UL"][0]
        if img_coords["UR"][0] < min_x:
            min_x = img_coords["UR"][0]
        if img_coords["LL"][0] < min_x:
            min_x = img_coords["LL"][0]
        if img_coords["LR"][0] < min_x:
            min_x = img_coords["LR"][0]

        if img_coords["UL"][1] < min_y:
            min_y = img_coords["UL"][1]
        if img_coords["UR"][1] < min_y:
            min_y = img_coords["UR"][1]
        if img_coords["LL"][1] < min_y:
            min_y = img_coords["LL"][1]
        if img_coords["LR"][1] < min_y:
            min_y = img_coords["LR"][1]

    mosaic_size = (round(max_x - min_x), round(max_y - min_y))

    stitching_img_corner_coords = copy.deepcopy(img_corner_coords)
    for img_idx in range(num_images):
        stitching_img_corner_coords[images_name[img_idx]]['UL'][0] = \
            stitching_img_corner_coords[images_name[img_idx]]['UL'][0] - min_x
        stitching_img_corner_coords[images_name[img_idx]]['UR'][0] = \
            stitching_img_corner_coords[images_name[img_idx]]['UR'][0] - min_x
        stitching_img_corner_coords[images_name[img_idx]]['LR'][0] = \
            stitching_img_corner_coords[images_name[img_idx]]['LR'][0] - min_x
        stitching_img_corner_coords[images_name[img_idx]]['LL'][0] = \
            stitching_img_corner_coords[images_name[img_idx]]['LL'][0] - min_x

        stitching_img_corner_coords[images_name[img_idx]]['UL'][1] = \
            stitching_img_corner_coords[images_name[img_idx]]['UL'][1] - min_y
        stitching_img_corner_coords[images_name[img_idx]]['UR'][1] = \
            stitching_img_corner_coords[images_name[img_idx]]['UR'][1] - min_y
        stitching_img_corner_coords[images_name[img_idx]]['LR'][1] = \
            stitching_img_corner_coords[images_name[img_idx]]['LR'][1] - min_y
        stitching_img_corner_coords[images_name[img_idx]]['LL'][1] = \
            stitching_img_corner_coords[images_name[img_idx]]['LL'][1] - min_y

    return img_corner_coords, stitching_img_corner_coords, mosaic_size, min_x, max_x, min_y, max_y


def generate_mosaic(images_path, warped_img, mosaic_size):
    images_name = os.listdir(images_path)
    mosaic = np.zeros((mosaic_size[1], mosaic_size[0], 3))

    for img_idx in range(len(images_name)):
        warped_img_idx = warped_img[img_idx]
        mosaic[mosaic == 0] = warped_img_idx[mosaic == 0]
    mosaic.astype(np.uint8)
    return mosaic
