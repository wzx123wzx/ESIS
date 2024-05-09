# Multi-frame Graph-cut based on superpixel segmentation
import cv2
import numpy as np
import multiprocessing
import pygco
from skimage.segmentation import slic, mark_boundaries
from skimage import measure
import datetime
import cv_utils


def Multi_frame_graph_cut(images_name, tf_vec, warped_img, warped_img_valid, img_h, img_w, mosaic_h, mosaic_w,
                          compactness, num_pixels_per_superpixel, Folder_settings, cores_num_to_use,
                          max_coverage=4):
    print('---------------multi-frame graph-cut---------------')
    num_images = len(images_name)

    # calculate center of warped images
    start_time = datetime.datetime.now()
    warped_img_center = parallel_img_center_warp(img_h, img_w, num_images, tf_vec, mosaic_h, mosaic_w,
                                                 cores_num_to_use=cores_num_to_use)
    print('parallel generate the center of the warped images, running time: {}'.format(
        datetime.datetime.now() - start_time))

    # calculate valid region of mosaic
    sum_valid = np.sum(warped_img_valid, axis=0)
    mosaic_valid = np.where(sum_valid > 0, True, False)
    num_valid_pixels = np.sum(mosaic_valid)
    np.save('{}/mosaic_valid.npy'.format(Folder_settings.path_results), mosaic_valid)

    # generate mosaic fused by weight constructed by distance to boundary
    start_time = datetime.datetime.now()
    mosaic_dist = blend_dist(warped_img, num_images, img_h, img_w, tf_vec, mosaic_h, mosaic_w, mosaic_valid)
    print('weighted blending, running time: {}'.format(datetime.datetime.now() - start_time))
    cv2.imwrite('{}/mosaic_fusion_dist.jpg'.format(Folder_settings.path_results), mosaic_dist)

    # superpixel segmentation
    slic_segments = slic(mosaic_dist, n_segments=round(num_valid_pixels / num_pixels_per_superpixel),
                         compactness=compactness, start_label=1)  # index from 1
    slic_segments, num_slic = index_mosaic_sp(slic_segments, mosaic_valid)

    # calculate range of overlapping image index for each superpixel
    start_time = datetime.datetime.now()
    mosaic_sp, range_sp, num_sp = parallel_calculate_mosaic_sp_range_sp(slic_segments, num_slic,
                                                                        warped_img_valid, warped_img_center,
                                                                        max_coverage, cores_num_to_use)
    out = mark_boundaries(mosaic_dist, mosaic_sp, color=(1, 1, 0))
    cv2.imwrite('{}/superpixel_mosaic.jpg'.format(Folder_settings.path_results), 255 * out)
    slic_boundary = np.all(out == np.array((1, 1, 0)), axis=2)
    print('refine superpixel segmentation and calculate image index range for each superpixel,'
          ' running time: {}'.format(datetime.datetime.now() - start_time))

    # calculate adjacent map of superpixel
    start_time = datetime.datetime.now()
    adj_sp = parallel_calculate_sp_adjacency(mosaic_sp, num_sp, cores_num_to_use=cores_num_to_use)
    print('generate adjacent matrix of superpixel, running time: {}'.format(datetime.datetime.now() - start_time))

    # calculate smooth cost for each overlapping image pair
    start_time = datetime.datetime.now()

    overlap_img_pair = generate_overlap_img_pair_sp_based(adj_sp, range_sp)

    pairwise_smooth_cost = parallel_calculate_img_dif_sal(overlap_img_pair, warped_img,
                                                          warped_img_valid,
                                                          cores_num_to_use=cores_num_to_use)

    print('calculate the smoothing term cost for different overlapping image pair,'
          ' running time: {}'.format(datetime.datetime.now() - start_time))

    # calculate smooth cost for each adjacent superpixel pair
    start_time = datetime.datetime.now()
    num_sp = adj_sp.shape[0]

    smooth_cost = calculate_smooth_cost_sp(adj_sp, mosaic_sp, num_sp,
                                           pairwise_smooth_cost, range_sp,
                                           overlap_img_pair)
    print('calculate the smoothing cost for each adjacent superpixel pair,'
          ' running time: {}'.format(datetime.datetime.now() - start_time))

    # calculate data cost
    inf_num = int(1e6)
    data_cost = inf_num * np.ones((num_sp, num_images), dtype=np.int32)
    for sp_idx in range(1, num_sp + 1):
        current_range_sp = range_sp[sp_idx - 1, :]
        current_range_sp = current_range_sp[current_range_sp != 0]
        for img_idx in current_range_sp:
            data_cost[sp_idx - 1, img_idx - 1] = 0

    # multi-label optimization
    start_time = datetime.datetime.now()
    sp_edges = np.array(np.where(adj_sp), dtype=np.int32).T
    smooth_weight = smooth_cost[sp_edges[:, 0], sp_edges[:, 1]].reshape(-1, 1)
    sp_edges = np.ascontiguousarray(sp_edges)

    sp_edges = np.hstack([sp_edges, smooth_weight])
    data_cost = np.ascontiguousarray(data_cost)
    sp_pairwise_cost = np.zeros((num_images, num_images), dtype=np.int32)
    optimal_sp = pygco.cut_from_graph(edges=sp_edges, unary_cost=data_cost, pairwise_cost=sp_pairwise_cost,
                                      n_iter=10000,
                                      algorithm='swap')
    print('multiple label (superpixel) optimization, running time: {}'.format(datetime.datetime.now() - start_time))

    # show the superpixel partition results
    sp_segment_result = show_optimal_sp(num_images, mosaic_sp, slic_boundary, optimal_sp)
    sp_segment_result = cv_utils.white_bck(sp_segment_result, mosaic_valid)
    start_time = datetime.datetime.now()
    mask_all_img = parallel_generate_mask_img(mosaic_sp, num_sp, optimal_sp, num_images, mosaic_h, mosaic_w,
                                              cores_num_to_use=cores_num_to_use)
    mosaic_graph_cut = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
    for img_idx in range(num_images):
        warped_img_idx = warped_img[img_idx]
        mask_img = mask_all_img[img_idx]
        mosaic_graph_cut[mask_img] = warped_img_idx[mask_img]
    mosaic_graph_cut = cv_utils.white_bck(mosaic_graph_cut, mosaic_valid)
    print('generate mosaic fused by optimal superpixel distribution,'
          ' running time: {}'.format(datetime.datetime.now() - start_time))
    cv2.imwrite('{}/mosaic_graph_cut.jpg'.format(Folder_settings.path_results), mosaic_graph_cut)
    cv2.imwrite('{}/optimal_sp_segment.jpg'.format(Folder_settings.path_results), sp_segment_result)
    return mosaic_sp, optimal_sp, sp_segment_result, mosaic_graph_cut


def select_KNN_helper(args):
    # select KNN range of superpixel
    sp_idx, range_sp, mosaic_sp, warped_img_center, max_coverage = args
    sp_mask = mosaic_sp == sp_idx + 1
    center_coord = calculate_region_center(sp_mask)
    dist = np.sum((warped_img_center[range_sp - 1] - center_coord) ** 2, axis=1)
    nearset_idx = np.argpartition(dist, max_coverage)[:max_coverage]
    check_vec = range_sp[nearset_idx]
    return sp_idx, check_vec


def parallel_select_KNN_for_range_sp(mosaic_sp, range_sp, max_coverage, warped_img_center, cores_num_to_use):
    num_sp = len(range_sp)
    range_sp_arr = np.zeros((num_sp, max_coverage), dtype=int)
    args = []

    for sp_idx, range_sp_idx in enumerate(range_sp):
        if range_sp_idx.shape[0] > max_coverage:
            args.append((sp_idx, range_sp_idx, mosaic_sp, warped_img_center, max_coverage))
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


def parallel_calculate_mosaic_sp_range_sp(slic_segments, num_slic, warped_img_valid, warped_img_center,
                                          max_coverage, cores_num_to_use):
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
    mosaic_sp_out = np.zeros_like(slic_segments, dtype=int)
    for idx, sp_idx in enumerate(sp):
        mosaic_sp_out[slic_segments == sp_idx] = idx + 1  # index from 1

    # select KNN
    range_sp_out = parallel_select_KNN_for_range_sp(mosaic_sp_out, range_sp, max_coverage, warped_img_center,
                                                    cores_num_to_use)

    return mosaic_sp_out, range_sp_out, num_sp


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
    mosaic_sp, num_sp, optimal_sp, img_idx, mosaic_h, mosaic_w = args
    mask_img = np.zeros((mosaic_h, mosaic_w), dtype=bool)
    for sp_idx in range(1, num_sp + 1):
        if optimal_sp[sp_idx - 1] == img_idx:
            sp_mask = mosaic_sp == sp_idx
            mask_img = mask_img | sp_mask
    return mask_img


def parallel_generate_mask_img(mosaic_sp, num_sp, optimal_sp, num_images, mosaic_h, mosaic_w, cores_num_to_use):
    args = []
    for img_idx in range(num_images):
        args.append((mosaic_sp, num_sp, optimal_sp, img_idx, mosaic_h, mosaic_w))

    processes = multiprocessing.Pool(cores_num_to_use)
    results = processes.map(generate_mask_img_helper, args)
    processes.close()

    return results


def index_mosaic_sp(mosaic_sp, mosaic_valid):
    # index superpixels
    mosaic_sp[~mosaic_valid] = 0  # drop invalid data
    mosaic_sp_out = np.zeros(mosaic_sp.shape, dtype=int)
    all_sp_idx = np.unique(mosaic_sp)
    all_sp_idx = all_sp_idx[all_sp_idx != 0]
    num_sp = len(all_sp_idx)
    for sp_idx in range(num_sp):
        mosaic_sp_out[mosaic_sp == all_sp_idx[sp_idx]] = sp_idx + 1

    return mosaic_sp_out, num_sp


def blend_dist(warped_img, num_images, img_h, img_w, tf_vec, mosaic_h, mosaic_w, mosaic_valid):
    # weighted fusion based distance to boundary
    mosaic_valid_3d = np.zeros((mosaic_h, mosaic_w, 3))
    mosaic_valid_3d[:, :, 0] = mosaic_valid
    mosaic_valid_3d[:, :, 1] = mosaic_valid
    mosaic_valid_3d[:, :, 2] = mosaic_valid

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
        H = tf_vec[img_idx * 9:img_idx * 9 + 9]
        H = H.reshape(3, 3)
        current_mask = cv2.warpPerspective(mask_dist, H, (mosaic_w, mosaic_h))
        current_img = warped_img[img_idx]
        current_img = np.multiply(current_img, current_mask)
        mosaic_dist = mosaic_dist + current_img
        mask_weight = mask_weight + current_mask
    mosaic_dist = np.divide(mosaic_dist, mask_weight)
    mosaic_dist = np.multiply(mosaic_dist, mosaic_valid_3d)
    mosaic_dist = mosaic_dist.astype(np.uint8)
    mosaic_dist = cv_utils.white_bck(mosaic_dist, mosaic_valid)
    return mosaic_dist


def img_center_warp_helper(args):
    img_center, img_tf_vec, mosaic_h, mosaic_w = args
    H = img_tf_vec.reshape(3, 3)
    img_center_warped = cv2.warpPerspective(img_center, H, (mosaic_w, mosaic_h))
    center_loc = np.where(img_center_warped != 0)
    return np.array([center_loc[0][0], center_loc[1][0]])


def parallel_img_center_warp(img_h, img_w, num_images, tf_vec, mosaic_h, mosaic_w, cores_num_to_use):
    # calculate center of warped images
    img_center = np.zeros((img_h, img_w), dtype=np.uint8)
    img_center[round(img_h / 2) - 2:round(img_h / 2) + 2, round(img_w / 2) - 2:round(img_w / 2) + 2] = 1
    args = []
    for img_idx in range(num_images):
        args.append((img_center, tf_vec[img_idx * 9:img_idx * 9 + 9], mosaic_h, mosaic_w))
    processes = multiprocessing.Pool(cores_num_to_use)
    list_warped_img_center = processes.map(img_center_warp_helper, args)
    processes.close()

    warped_img_center = np.array(list_warped_img_center)
    return warped_img_center


def calculate_img_dif_sal_helper(args):
    # calculate difference of two images and smooth cost
    warped_img1, warped_img2, warped_img1_valid, warped_img2_valid = args
    overlap_mask = warped_img1_valid & warped_img2_valid
    # intensity difference
    warped_img1_int = warped_img1.astype(np.int16)
    warped_img2_int = warped_img2.astype(np.int16)

    img_color_diff = np.sum(np.abs(warped_img1_int - warped_img2_int), axis=2)

    # gradient difference
    warped_img1_gray = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY)
    warped_img2_gray = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY)
    warped_img1_grad = np.abs(cv2.Sobel(warped_img1_gray, cv2.CV_64F, dx=1, dy=1, ksize=3))
    warped_img2_grad = np.abs(cv2.Sobel(warped_img2_gray, cv2.CV_64F, dx=1, dy=1, ksize=3))

    img_grad_diff = (warped_img1_grad + warped_img2_grad) / 2 + np.abs(warped_img1_grad - warped_img2_grad)

    # saliency difference
    saliency_obj = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, warped_img1_saliencymap = saliency_obj.computeSaliency(warped_img1)
    _, warped_img2_saliencymap = saliency_obj.computeSaliency(warped_img2)
    img_sal_diff = (warped_img1_saliencymap + warped_img2_saliencymap) / 2

    # combine
    img_diff = (img_grad_diff + img_color_diff) * img_sal_diff / 5  # 0-255
    img_diff = img_diff * overlap_mask
    img_diff = img_diff.astype(np.uint8)
    return img_diff


def parallel_calculate_img_dif_sal(overlap_img_pair, warped_img, warped_img_valid, cores_num_to_use):
    args = [(warped_img[img_idx1],
             warped_img[img_idx2],
             warped_img_valid[img_idx1],
             warped_img_valid[img_idx2],
             )
            for img_idx1, img_idx2 in overlap_img_pair]

    processes = multiprocessing.Pool(cores_num_to_use)
    pairwise_smooth_cost = processes.map(calculate_img_dif_sal_helper, args)
    processes.close()
    return np.array(pairwise_smooth_cost)


def calculate_smooth_cost_sp(adj_sp, mosaic_sp, num_sp, pairwise_smooth_cost, range_sp, overlap_img_pair):
    smooth_cost = np.zeros((num_sp, num_sp), dtype=np.int32)
    sp_idx1_vec, sp_idx2_vec = np.nonzero(adj_sp)
    for adj_idx in range(len(sp_idx1_vec)):
        sp_idx1 = sp_idx1_vec[adj_idx]  # index from 0
        sp_idx2 = sp_idx2_vec[adj_idx]
        range_sp1 = range_sp[sp_idx1, :][range_sp[sp_idx1, :] != 0]
        range_sp2 = range_sp[sp_idx2, :][range_sp[sp_idx2, :] != 0]
        mask1 = mosaic_sp == sp_idx1 + 1
        mask2 = mosaic_sp == sp_idx2 + 1
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


def parallel_calculate_warped_img(images_path, images_name, tf_vec, img_h, img_w, mosaic_h, mosaic_w,
                                  cores_num_to_use, isimg):
    # isimg:True  warped image
    # isimg:False valid region of warped image
    args = []
    if isimg:
        for img_idx in range(len(images_name)):
            args.append((img_idx, mosaic_h, mosaic_w, tf_vec,
                         cv2.imread("{0}/{1}".format(images_path, images_name[img_idx])), isimg))
    else:
        for img_idx in range(len(images_name)):
            args.append((img_idx, mosaic_h, mosaic_w, tf_vec, np.ones((img_h, img_w)), isimg))
    processes = multiprocessing.Pool(cores_num_to_use)
    if isimg:
        warped_img = processes.map(calculate_warped_img_helper, args)
        processes.close()
        return np.array(warped_img)
    else:
        warped_img_valid = processes.map(calculate_warped_img_helper, args)
        processes.close()
        return np.array(warped_img_valid)


def calculate_sp_adjacency_helper(args):
    sp_idx, mosaic_sp = args
    current_sp_mask = (mosaic_sp == sp_idx).astype(np.uint8)  # index from 1
    dilated_current_sp_mask = cv2.dilate(current_sp_mask, np.ones((3, 3), dtype=np.uint8))
    boundary_current_sp_mask = dilated_current_sp_mask - current_sp_mask
    cur_adj = np.unique(np.setdiff1d(boundary_current_sp_mask * mosaic_sp, [0, sp_idx]))
    return sp_idx, cur_adj


def parallel_calculate_sp_adjacency(mosaic_sp, num_sp, cores_num_to_use):
    # adjacent matrix
    adj_sp = np.zeros((num_sp, num_sp), dtype=bool)
    args = []
    for sp_idx in range(1, num_sp + 1):
        args.append((sp_idx, mosaic_sp))
    processes = multiprocessing.Pool(cores_num_to_use)
    results = processes.map(calculate_sp_adjacency_helper, args)
    processes.close()
    for sp_idx, sp_idx_adj in results:
        adj_sp[sp_idx - 1, sp_idx_adj - 1] = True
        adj_sp[sp_idx_adj - 1, sp_idx - 1] = True
    adj_sp = np.triu(adj_sp)  # upper triangular matrix
    return adj_sp


def generate_overlap_img_pair_sp_based(adj_sp, range_sp):
    # generate overlap img pair based on adjacent matrix of superpixel
    overlap_img_pair = []
    sp_idx1_vec, sp_idx2_vec = np.nonzero(adj_sp)
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


def show_optimal_sp(num_images, mosaic_sp, slic_boundary, optimal_sp):
    # displays the partition results based on superpixel segmentation
    R = np.random.randint(low=0, high=255, size=num_images)
    G = np.random.randint(low=0, high=255, size=num_images)
    B = np.random.randint(low=0, high=255, size=num_images)
    sp_result = np.zeros((mosaic_sp.shape[0], mosaic_sp.shape[1], 3), dtype=np.uint8)
    num_sp = len(optimal_sp)
    for sp_idx in range(1, num_sp + 1):
        img_idx = optimal_sp[sp_idx - 1]
        sp_result[mosaic_sp == sp_idx] = optimal_sp[sp_idx - 1] + 1
        sp_result[mosaic_sp == sp_idx] = np.array([R[img_idx], G[img_idx], B[img_idx]])
    sp_result[slic_boundary] = np.array([0, 0, 255])
    return sp_result


def calculate_region_center(mask):
    # calculate center of region
    valid_i, valid_j = np.nonzero(mask)
    center_coord = np.array([np.round(np.mean(valid_i)), np.round(np.mean(valid_j))], dtype=int)
    return center_coord


def calculate_region_adj(mask1, mask2):
    # check whether the area is connected
    mask3 = mask1 | mask2
    mask_labels = measure.label(mask3, connectivity=1)
    num_labels = max(mask_labels)
    if num_labels == 1:
        adj_flag = True
    else:
        adj_flag = False
    return adj_flag
