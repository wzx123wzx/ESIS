# Multi-frame Graph-cut based on superpixel segmentation
import cv2
import numpy as np
import multiprocessing
import pygco
from skimage.segmentation import slic, mark_boundaries
import datetime
import utils
import constant

# device
cores_num_to_use = constant.device.cores_num_to_use


def Multi_frame_graph_cut(images_name, transform_vec, warped_img, warped_img_valid, img_h, img_w, mosaic_h, mosaic_w,
                          compactness, num_pixels_per_superpixel, path_results, max_coverage):
    print('---------------multi-frame graph-cut---------------')
    num_images = len(images_name)

    # calculate center of warped images
    start_time = datetime.datetime.now()
    warped_img_center = parallel_calculate_warped_img_center(num_images, transform_vec, img_h, img_w,
                                                             mosaic_h, mosaic_w)
    print('parallel calculate the center of the warped images, running time: {}'.format(
        datetime.datetime.now() - start_time))

    # calculate valid region of mosaic
    sum_valid = np.sum(warped_img_valid, axis=0)
    mosaic_valid_mask = np.where(sum_valid > 0, True, False)
    cv2.imwrite('{}/mosaic_valid_mask.png'.format(path_results), 255 * mosaic_valid_mask.astype(np.uint8))
    num_valid_pixels = np.sum(mosaic_valid_mask)

    # generate mosaic fused by weight constructed by distance to boundary
    start_time = datetime.datetime.now()
    mosaic_dist = blend_dist(num_images, transform_vec, warped_img, img_h, img_w, mosaic_h, mosaic_w, mosaic_valid_mask)
    print('initial weighted blending, running time: {}'.format(datetime.datetime.now() - start_time))
    cv2.imwrite('{}/mosaic_fusion_dist.png'.format(path_results), mosaic_dist)

    # superpixel segmentation (index from 1)
    slic_segments = slic(mosaic_dist, n_segments=round(num_valid_pixels / num_pixels_per_superpixel),
                         compactness=compactness, start_label=1)
    slic_segments, num_slic = index_slic_segments(slic_segments, mosaic_valid_mask)

    # calculate range of overlapping image index for each superpixel
    start_time = datetime.datetime.now()
    mosaic_sp_mask, range_sp, num_sp = parallel_calculate_mosaic_sp_mask(slic_segments, num_slic, warped_img_valid,
                                                                         warped_img_center, max_coverage)
    superpixel_segmentation = mark_boundaries(mosaic_dist, mosaic_sp_mask, color=(1, 1, 0))
    cv2.imwrite('{}/superpixel_segmentation.png'.format(path_results), 255 * superpixel_segmentation)
    superpixel_segmentation_boundary = np.all(superpixel_segmentation == np.array((1, 1, 0)), axis=2)
    print('refine superpixel segmentation and calculate image index range for each superpixel,'
          ' running time: {}'.format(datetime.datetime.now() - start_time))

    # calculate adjacent map of superpixel
    start_time = datetime.datetime.now()
    pairwise_sp = parallel_calculate_pairwise_sp(mosaic_sp_mask, num_sp)
    print('generate adjacent matrix of superpixel, running time: {}'.format(datetime.datetime.now() - start_time))

    # calculate smooth cost for each overlapping image pair
    start_time = datetime.datetime.now()
    overlap_img_pair = generate_overlap_img_pair_sp_based(pairwise_sp, range_sp)
    pairwise_smooth_cost = parallel_calculate_img_diff(overlap_img_pair, warped_img, warped_img_valid)

    print('calculate the smooth term cost for all overlapping image pairs,'
          ' running time: {}'.format(datetime.datetime.now() - start_time))

    # calculate smooth cost for each adjacent superpixel pair
    start_time = datetime.datetime.now()
    num_sp = pairwise_sp.shape[0]
    smooth_cost = calculate_smooth_cost_sp(pairwise_sp, mosaic_sp_mask, num_sp,
                                           pairwise_smooth_cost, range_sp, overlap_img_pair)
    print('calculate the smooth cost for each adjacent superpixel pair,'
          ' running time: {}'.format(datetime.datetime.now() - start_time))

    # multi-label optimization
    start_time = datetime.datetime.now()
    edges = calculate_graph_edges(pairwise_sp, smooth_cost)
    ref_inf_num = np.max(edges[:, 2])
    unary_cost = calculate_graph_unary_cost(num_images, num_sp, range_sp, 5 * ref_inf_num)
    pairwise_cost = calculate_graph_pairwise_cost(num_images, overlap_img_pair)
    optimal_sp = pygco.cut_from_graph(edges=edges, unary_cost=unary_cost, pairwise_cost=pairwise_cost,
                                      n_iter=10000, algorithm='expansion')
    print('superpixel level multiple label optimization, running time: {}'.format(datetime.datetime.now() - start_time))

    # show the superpixel partition results
    sp_segment_result = show_optimal_sp(num_images, mosaic_sp_mask, superpixel_segmentation_boundary, optimal_sp)
    sp_segment_result = utils.white_bck(sp_segment_result, mosaic_valid_mask)
    start_time = datetime.datetime.now()
    mask_all_img = parallel_generate_mask_img(mosaic_sp_mask, num_sp, optimal_sp, num_images, mosaic_h, mosaic_w)
    mosaic_graph_cut = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
    composition_mask = np.zeros((mosaic_h, mosaic_w), dtype=np.int32)
    for img_idx in range(num_images):
        warped_img_idx = warped_img[img_idx]
        mask_img = mask_all_img[img_idx]
        mosaic_graph_cut[mask_img] = warped_img_idx[mask_img]
        composition_mask[mask_img] = img_idx
    mosaic_graph_cut = utils.white_bck(mosaic_graph_cut, mosaic_valid_mask)
    print('generate mosaic fused by optimal superpixel distribution,'
          ' running time: {}'.format(datetime.datetime.now() - start_time))
    cv2.imwrite('{}/mosaic_graph_cut.png'.format(path_results), mosaic_graph_cut)
    cv2.imwrite('{}/optimal_superpixel_segmentation.png'.format(path_results), sp_segment_result)
    global_seamline = extract_global_seamline(num_images, mosaic_graph_cut, composition_mask)
    mosaic_with_global_seamline = mosaic_graph_cut
    mosaic_with_global_seamline[global_seamline] = np.array([0, 0, 255])
    cv2.imwrite('{}/mosaic_with_global_seamline.png'.format(path_results), mosaic_with_global_seamline)
    return mosaic_sp_mask, optimal_sp, sp_segment_result, mosaic_graph_cut


def extract_global_seamline(num_images, mosaic, composition_mask):
    mosaic_valid = composition_mask != num_images
    mosaic_boundary = extract_boundary(mosaic_valid)
    mosaic_boundary = cv2.dilate(mosaic_boundary.astype(np.uint8), np.ones((3, 3), dtype=np.uint8))
    mosaic_boundary = mosaic_boundary == 1
    global_seamline = mark_boundaries(mosaic, composition_mask, color=(1, 1, 0))
    global_seamline = np.all(global_seamline == np.array((1, 1, 0)), axis=2)
    global_seamline = global_seamline & ~mosaic_boundary
    return global_seamline


def extract_boundary(mask):
    sz1, sz2 = mask.shape

    BR = (mask & ~(np.concatenate((mask[:, 1:], np.zeros((sz1, 1), dtype=bool)), axis=1)))  # 右边界
    BL = (mask & ~(np.concatenate((np.zeros((sz1, 1), dtype=bool), mask[:, :-1]), axis=1)))  # 左边界
    BD = (mask & ~(np.concatenate((mask[1:, :], np.zeros((1, sz2), dtype=bool)), axis=0)))  # 下边界
    BU = (mask & ~(np.concatenate((np.zeros((1, sz2), dtype=bool), mask[:-1, :]), axis=0)))  # 上边界

    boundary_mask = BR | BL | BD | BU
    return boundary_mask


def select_KNN_helper(args):
    # select KNN range of superpixel
    sp_idx, range_sp, mosaic_sp_mask, warped_img_center, max_coverage = args
    sp_mask = mosaic_sp_mask == sp_idx + 1
    center_coord = calculate_region_center(sp_mask)
    dist = np.sum((warped_img_center[range_sp - 1] - center_coord) ** 2, axis=1)
    nearset_idx = np.argpartition(dist, max_coverage)[:max_coverage]
    check_vec = range_sp[nearset_idx]
    return sp_idx, check_vec


def parallel_select_KNN_for_range_sp(mosaic_sp_mask, range_sp, max_coverage, warped_img_center):
    num_sp = len(range_sp)
    range_sp_arr = np.zeros((num_sp, max_coverage), dtype=int)
    args = []

    for sp_idx, range_sp_idx in enumerate(range_sp):
        if range_sp_idx.shape[0] > max_coverage:
            args.append((sp_idx, range_sp_idx, mosaic_sp_mask, warped_img_center, max_coverage))
        else:
            range_sp_arr[sp_idx, :range_sp_idx.shape[0]] = range_sp_idx

    processes = multiprocessing.Pool(cores_num_to_use)
    results = processes.map(select_KNN_helper, args)
    processes.close()

    for sp_idx, check_vec in results:
        range_sp_arr[sp_idx, :] = check_vec
    return range_sp_arr


def calculate_range_sp(slic_segments, num_slic, warped_img_valid):
    range_sp_results = []
    for current_sp_idx in range(1, num_slic + 1):
        sp_mask = slic_segments == current_sp_idx
        sp_check = warped_img_valid[:, sp_mask]
        current_range_sp = np.all(sp_check, axis=1)
        current_range_sp = np.where(current_range_sp)[0]  # index from 0, all region of superpixel in valid region
        range_sp_results.append((current_sp_idx, current_range_sp))
    return range_sp_results


def parallel_calculate_mosaic_sp_mask(slic_segments, num_slic, warped_img_valid, warped_img_center, max_coverage):
    args = []
    for sp_idx in range(1, num_slic + 1):
        args.append((sp_idx, slic_segments, warped_img_valid))

    range_sp_results = calculate_range_sp(slic_segments, num_slic, warped_img_valid)
    across_sp_segment_results = refine_across_sp(slic_segments, num_slic, range_sp_results, warped_img_valid)
    range_sp = []
    sp = []

    # superpixels that do not cross valid regions
    for current_sp_idx, current_range_sp_idx in range_sp_results:
        if len(current_range_sp_idx) > 0:
            sp.append(current_sp_idx)
            range_sp.append(current_range_sp_idx + 1)  # index from 1

    # superpixels that cross valid regions
    next_sp_idx = num_slic + 1
    for across_sp_idx, across_mask_sp_idx, refined_across_sp_idx_set, refined_range_across_sp_set \
            in across_sp_segment_results:
        across_sp_mask = slic_segments == across_sp_idx
        slic_segments[across_sp_mask] = across_mask_sp_idx[across_sp_mask] + next_sp_idx
        for segment_idx in range(len(refined_across_sp_idx_set)):
            sp.append(refined_across_sp_idx_set[segment_idx] + next_sp_idx)
            range_sp.append(np.array([refined_range_across_sp_set[segment_idx]]))
        next_sp_idx = next_sp_idx + len(refined_across_sp_idx_set)

    # index superpixels
    num_sp = len(sp)
    mosaic_sp_mask = np.zeros_like(slic_segments, dtype=int)
    for idx, sp_idx in enumerate(sp):
        mosaic_sp_mask[slic_segments == sp_idx] = idx + 1  # index from 1

    # select KNN
    range_sp_KNN = parallel_select_KNN_for_range_sp(mosaic_sp_mask, range_sp, max_coverage, warped_img_center)

    return mosaic_sp_mask, range_sp_KNN, num_sp


def refine_across_sp(slic_segments, num_slic, list_range_sp, warped_img_valid):
    across_sp_segment_results = []
    for sp_idx in range(1, num_slic + 1):
        if len(list_range_sp[sp_idx - 1][1]) == 0:
            current_across_sp_idx = sp_idx
            across_sp_mask = slic_segments == current_across_sp_idx
            sp_mask_left = across_sp_mask.copy()
            range_current_across_sp = warped_img_valid[:, across_sp_mask]
            range_current_across_sp = np.any(range_current_across_sp, axis=1)
            range_current_across_sp = np.where(range_current_across_sp)[0]
            next_sp_idx = 1
            new_across_sp_mask = np.zeros(across_sp_mask.shape, dtype=int)

            new_sp = []
            new_range_sp = []
            for img_idx in range_current_across_sp:
                if ~np.any(sp_mask_left):
                    break
                valid_mask = sp_mask_left & warped_img_valid[img_idx]
                if np.any(valid_mask):
                    new_across_sp_mask[valid_mask] = next_sp_idx
                    new_sp.append(next_sp_idx)
                    new_range_sp.append(img_idx + 1)
                    next_sp_idx += 1
                    sp_mask_left[valid_mask] = False
            across_sp_segment_results.append((current_across_sp_idx, new_across_sp_mask, new_sp, new_range_sp))
    return across_sp_segment_results


def generate_mask_img_helper(args):
    mosaic_sp_mask, num_sp, optimal_sp, img_idx, mosaic_h, mosaic_w = args
    mask_img = np.zeros((mosaic_h, mosaic_w), dtype=bool)
    for sp_idx in range(1, num_sp + 1):
        if optimal_sp[sp_idx - 1] == img_idx:
            sp_mask = mosaic_sp_mask == sp_idx
            mask_img = mask_img | sp_mask
    return mask_img


def parallel_generate_mask_img(mosaic_sp_mask, num_sp, optimal_sp, num_images, mosaic_h, mosaic_w):
    args = []
    for img_idx in range(num_images):
        args.append((mosaic_sp_mask, num_sp, optimal_sp, img_idx, mosaic_h, mosaic_w))

    processes = multiprocessing.Pool(cores_num_to_use)
    results = processes.map(generate_mask_img_helper, args)
    processes.close()

    return results


def index_slic_segments(slic_segments, valid_mask):
    # index superpixels
    slic_segments[~valid_mask] = 0  # drop invalid data
    output_slic_segments = np.zeros(slic_segments.shape, dtype=int)
    all_sp_idx = np.unique(slic_segments)
    all_sp_idx = all_sp_idx[all_sp_idx != 0]
    num_sp = len(all_sp_idx)
    # index: from 1 to num_sp
    for sp_idx in range(num_sp):
        output_slic_segments[slic_segments == all_sp_idx[sp_idx]] = sp_idx + 1

    return output_slic_segments, num_sp


def blend_dist(num_images, transform_vec, warped_img, img_h, img_w, mosaic_h, mosaic_w, mosaic_valid_mask):
    # weighted fusion based on distance to boundary
    mosaic_valid_3d = np.expand_dims(mosaic_valid_mask, axis=-1)
    mosaic_valid_3d = np.repeat(mosaic_valid_3d, 3, axis=-1)
    row_dist = np.minimum(np.arange(img_h), img_h - np.arange(img_h))[:, np.newaxis]
    col_dist = np.minimum(np.arange(img_w), img_w - np.arange(img_w))[np.newaxis, :]
    mask_dist_2d = np.minimum(row_dist, col_dist)
    mask_dist = np.zeros((img_h, img_w, 3))
    mask_dist[:, :, 0] = mask_dist_2d
    mask_dist[:, :, 1] = mask_dist_2d
    mask_dist[:, :, 2] = mask_dist_2d

    mask_weight = np.zeros((mosaic_h, mosaic_w, 3))
    mosaic_dist = np.zeros((mosaic_h, mosaic_w, 3))
    for img_idx in range(num_images):
        H = transform_vec[img_idx * 9:img_idx * 9 + 9]
        H = H.reshape(3, 3)
        current_mask = cv2.warpPerspective(mask_dist, H, (mosaic_w, mosaic_h))
        current_img = warped_img[img_idx]
        current_img = np.multiply(current_img, current_mask)
        mosaic_dist = mosaic_dist + current_img
        mask_weight = mask_weight + current_mask
    mosaic_dist = np.divide(mosaic_dist, mask_weight + 1e-8)
    mosaic_dist = np.multiply(mosaic_dist, mosaic_valid_3d)
    mosaic_dist = mosaic_dist.astype(np.uint8)
    mosaic_dist = utils.white_bck(mosaic_dist, mosaic_valid_mask)
    return mosaic_dist


def calculate_warped_img_center_helper(args):
    img_center, img_transform_vec, mosaic_h, mosaic_w = args
    H = img_transform_vec.reshape(3, 3)
    img_center_warped = cv2.warpPerspective(img_center, H, (mosaic_w, mosaic_h))
    center_loc = np.where(img_center_warped != 0)
    return np.array([center_loc[0][0], center_loc[1][0]])


def parallel_calculate_warped_img_center(num_images, transform_vec, img_h, img_w, mosaic_h, mosaic_w):
    # calculate center of warped images
    img_center = np.zeros((img_h, img_w), dtype=np.uint8)
    img_center[round(img_h / 2) - 2:round(img_h / 2) + 2, round(img_w / 2) - 2:round(img_w / 2) + 2] = 1
    args = []
    for img_idx in range(num_images):
        args.append((img_center, transform_vec[img_idx * 9:img_idx * 9 + 9], mosaic_h, mosaic_w))
    processes = multiprocessing.Pool(cores_num_to_use)
    list_warped_img_center = processes.map(calculate_warped_img_center_helper, args)
    processes.close()

    warped_img_center = np.array(list_warped_img_center)
    return warped_img_center


def calculate_img_diff_helper(args):
    # calculate difference of two images and smooth cost
    # compared to paper, we found color difference is generate better result in most cases
    warped_img1, warped_img2, warped_img1_valid, warped_img2_valid = args
    overlap_mask = warped_img1_valid & warped_img2_valid
    # intensity difference
    warped_img1_int = warped_img1.astype(np.int16)
    warped_img2_int = warped_img2.astype(np.int16)

    img_color_diff = np.sum(np.abs(warped_img1_int - warped_img2_int), axis=2)

    # gradient difference
    # warped_img1_gray = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY)
    # warped_img2_gray = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY)
    # warped_img1_grad = np.abs(cv2.Sobel(warped_img1_gray, cv2.CV_64F, dx=1, dy=1, ksize=3))
    # warped_img2_grad = np.abs(cv2.Sobel(warped_img2_gray, cv2.CV_64F, dx=1, dy=1, ksize=3))
    #
    # img_grad_diff = (warped_img1_grad + warped_img2_grad) / 2 + np.abs(warped_img1_grad - warped_img2_grad)

    # saliency difference
    # saliency_obj = cv2.saliency.StaticSaliencySpectralResidual_create()
    # _, warped_img1_saliencymap = saliency_obj.computeSaliency(warped_img1)
    # _, warped_img2_saliencymap = saliency_obj.computeSaliency(warped_img2)
    # img_sal_diff = (warped_img1_saliencymap + warped_img2_saliencymap) / 2

    # combine
    # img_diff = (img_grad_diff + img_color_diff) * img_sal_diff / 5  # 0-255
    img_diff = img_color_diff / 3
    img_diff = img_diff * overlap_mask
    img_diff = img_diff.astype(np.uint8)
    return img_diff


def parallel_calculate_img_diff(overlap_img_pair, warped_img, warped_img_valid):
    args = [(warped_img[img_idx1],
             warped_img[img_idx2],
             warped_img_valid[img_idx1],
             warped_img_valid[img_idx2],
             )
            for img_idx1, img_idx2 in overlap_img_pair]

    processes = multiprocessing.Pool(cores_num_to_use)
    pairwise_smooth_cost = processes.map(calculate_img_diff_helper, args)
    processes.close()
    return np.array(pairwise_smooth_cost)


def calculate_smooth_cost_sp(pairwise_sp, mosaic_sp_mask, num_sp, pairwise_smooth_cost, range_sp, overlap_img_pair):
    smooth_cost = np.zeros((num_sp, num_sp), dtype=np.int32)
    sp_idx1_vec, sp_idx2_vec = np.nonzero(pairwise_sp)
    for adj_idx in range(len(sp_idx1_vec)):
        sp_idx1 = sp_idx1_vec[adj_idx]  # index from 0
        sp_idx2 = sp_idx2_vec[adj_idx]
        range_sp1 = range_sp[sp_idx1, :][range_sp[sp_idx1, :] != 0]
        range_sp2 = range_sp[sp_idx2, :][range_sp[sp_idx2, :] != 0]
        mask1 = mosaic_sp_mask == sp_idx1 + 1
        mask2 = mosaic_sp_mask == sp_idx2 + 1
        current_smooth_cost = 0
        for img_idx1 in range_sp1:  # index from 1
            for img_idx2 in range_sp2:
                if img_idx1 != img_idx2:
                    if img_idx2 > img_idx1:
                        if [img_idx1 - 1, img_idx2 - 1] in overlap_img_pair:
                            overlap_img_pair_idx = overlap_img_pair.index([img_idx1 - 1, img_idx2 - 1])
                            current_smooth_cost_map = pairwise_smooth_cost[overlap_img_pair_idx]
                            current_smooth_cost = max(np.mean(current_smooth_cost_map[mask1 | mask2]),
                                                      current_smooth_cost)
                            smooth_cost[sp_idx1, sp_idx2] = current_smooth_cost
                            smooth_cost[sp_idx2, sp_idx1] = current_smooth_cost
                    else:
                        if [img_idx2 - 1, img_idx1 - 1] in overlap_img_pair:
                            overlap_img_pair_idx = overlap_img_pair.index([img_idx2 - 1, img_idx1 - 1])
                            current_smooth_cost_map = pairwise_smooth_cost[overlap_img_pair_idx]
                            current_smooth_cost = max(np.mean(current_smooth_cost_map[mask1 | mask2]),
                                                      current_smooth_cost)
                            smooth_cost[sp_idx1, sp_idx2] = current_smooth_cost
                            smooth_cost[sp_idx2, sp_idx1] = current_smooth_cost
    return smooth_cost


def calculate_pairwise_sp_helper(args):
    sp_idx, mosaic_sp_mask = args
    current_sp_mask = (mosaic_sp_mask == sp_idx).astype(np.uint8)  # index from 1
    dilated_current_sp_mask = cv2.dilate(current_sp_mask, np.ones((3, 3), dtype=np.uint8))
    boundary_current_sp_mask = dilated_current_sp_mask - current_sp_mask
    cur_adj = np.unique(np.setdiff1d(boundary_current_sp_mask * mosaic_sp_mask, [0, sp_idx]))
    return sp_idx, cur_adj


def parallel_calculate_pairwise_sp(mosaic_sp_mask, num_sp):
    # adjacent matrix
    pairwise_sp = np.zeros((num_sp, num_sp), dtype=bool)
    args = []
    for sp_idx in range(1, num_sp + 1):
        args.append((sp_idx, mosaic_sp_mask))
    processes = multiprocessing.Pool(cores_num_to_use)
    results = processes.map(calculate_pairwise_sp_helper, args)
    processes.close()
    for sp_idx, sp_idx_adj in results:
        pairwise_sp[sp_idx - 1, sp_idx_adj - 1] = True
        pairwise_sp[sp_idx_adj - 1, sp_idx - 1] = True
    pairwise_sp = np.triu(pairwise_sp)  # upper triangular matrix
    return pairwise_sp


def generate_overlap_img_pair_sp_based(pairwise_sp, range_sp):
    # generate overlap img pair based on adjacent matrix of superpixel
    overlap_img_pair = []
    sp_idx1_vec, sp_idx2_vec = np.nonzero(pairwise_sp)
    for adj_idx in range(len(sp_idx1_vec)):
        sp_idx1 = sp_idx1_vec[adj_idx]  # index from 0
        sp_idx2 = sp_idx2_vec[adj_idx]
        range_sp1 = range_sp[sp_idx1, :][range_sp[sp_idx1, :] != 0]
        range_sp2 = range_sp[sp_idx2, :][range_sp[sp_idx2, :] != 0]
        if range_sp1.shape[0] > 1 or range_sp2.shape[0] > 1:
            for img_idx1 in range_sp1:  # index from 1
                for img_idx2 in range_sp2:
                    if img_idx1 != img_idx2:
                        if [img_idx1 - 1, img_idx2 - 1] not in overlap_img_pair \
                                and [img_idx2 - 1, img_idx1 - 1] not in overlap_img_pair:
                            if img_idx2 > img_idx1:
                                overlap_img_pair.append([img_idx1 - 1, img_idx2 - 1])
                            else:
                                overlap_img_pair.append([img_idx2 - 1, img_idx1 - 1])

    return overlap_img_pair


def show_optimal_sp(num_images, mosaic_sp_mask, superpixel_segmentation_boundary, optimal_sp):
    # displays the partition results based on superpixel segmentation
    R = np.random.randint(low=0, high=255, size=num_images)
    G = np.random.randint(low=0, high=255, size=num_images)
    B = np.random.randint(low=0, high=255, size=num_images)
    sp_result = np.zeros((mosaic_sp_mask.shape[0], mosaic_sp_mask.shape[1], 3), dtype=np.uint8)
    num_sp = len(optimal_sp)
    for sp_idx in range(1, num_sp + 1):
        img_idx = optimal_sp[sp_idx - 1]
        sp_result[mosaic_sp_mask == sp_idx] = optimal_sp[sp_idx - 1] + 1
        sp_result[mosaic_sp_mask == sp_idx] = np.array([R[img_idx], G[img_idx], B[img_idx]])
    sp_result[superpixel_segmentation_boundary] = np.array([0, 0, 255])
    return sp_result


def calculate_region_center(mask):
    # calculate center of region
    valid_i, valid_j = np.nonzero(mask)
    center_coord = np.array([np.round(np.mean(valid_i)), np.round(np.mean(valid_j))], dtype=int)
    return center_coord


def calculate_graph_edges(pairwise_sp, smooth_cost):
    edges = np.array(np.where(pairwise_sp), dtype=np.int32).T
    weight = smooth_cost[edges[:, 0], edges[:, 1]].reshape(-1, 1).astype(np.int32)
    edges = np.ascontiguousarray(edges)
    edges = np.hstack([edges, weight])
    edges = edges[edges[:, 2] != 0]
    return edges


def calculate_graph_unary_cost(num_images, num_sp, range_sp, inf_num):
    unary_cost = inf_num * np.ones((num_sp, num_images), dtype=np.int32)
    for sp_idx in range(1, num_sp + 1):
        current_range_sp = range_sp[sp_idx - 1, :]
        current_range_sp = current_range_sp[current_range_sp != 0]
        for img_idx in current_range_sp:
            unary_cost[sp_idx - 1, img_idx - 1] = 0
    unary_cost = np.ascontiguousarray(unary_cost)
    return unary_cost


def calculate_graph_pairwise_cost(num_images, overlap_img_pair):
    pairwise_cost = 1 * np.ones((num_images, num_images), dtype=np.int32)
    for idx in range(len(overlap_img_pair)):
        img_idx_1 = overlap_img_pair[idx][0]
        img_idx_2 = overlap_img_pair[idx][1]
        pairwise_cost[img_idx_1, img_idx_2] = 1
        pairwise_cost[img_idx_2, img_idx_1] = 1
    np.fill_diagonal(pairwise_cost, 0)
    return pairwise_cost
