# Efficient Superpixel-based Seamline Detection for Large-scale Image Stitching
# wzx123wzx
import cv2
from feature_extraction import parallel_feature_extraction
from feature_matching import parallel_feature_matching
import matching_points_selection
from matching_image_pair_generation import get_all_4NN_neighbors, matching_image_index_pair
import utils
import image_stitching
from image_scaling import parallel_image_scaling
import numpy as np
from transformation_optimization import linear_affine_transformation_optimization
import time
import datetime
import constant
import os
import Graph_cut
import sys
# Configuration
# dataset
images_path = constant.dataset.images_path
img_h = constant.dataset.img_h
img_w = constant.dataset.img_w
scale = constant.dataset.scale

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

selected_points_num = constant.sift.points_selection.selected_points_num

# superpixel segmentation
slic_compactness = constant.superpixel_segmentation.slic_compactness
slic_num_pixels_per_superpixel = constant.superpixel_segmentation.slic_num_pixels_per_superpixel
max_coverage = constant.superpixel_segmentation.max_coverage

# device
cores_num_to_use = constant.device.cores_num_to_use

# path
path_scaled_images = constant.path.path_scaled_images
path_results = constant.path.path_results
path_sift_extraction_result = constant.path.path_sift_extraction_result
path_sift_matching_result = constant.path.path_sift_matching_result

if __name__ == '__main__':
    t_start = time.time()
    images_path = images_path
    images_name = os.listdir(images_path)
    constant.make_all_folders()
    original = sys.stdout
    log_file = open('{}/logs.txt'.format(path_results), "w")
    sys.stdout = log_file

    # dataset scaling
    print('---------------image scaling---------------')
    print('number of images: {}'.format(len(images_name)))
    start_time = datetime.datetime.now()
    parallel_image_scaling(images_path, scale, path_scaled_images)
    print('parallel image scaling, running time: {}'.format(datetime.datetime.now() - start_time))

    # matching image pair generation
    print('---------------matching image pair generation---------------')
    start_time = datetime.datetime.now()
    matching_image_name_pair = get_all_4NN_neighbors(images_path)
    matching_image_pair = matching_image_index_pair(images_path, matching_image_name_pair)
    print('number of matching image pair: {}'.format(len(matching_image_pair)))
    np.save('{}/matching_image_pair.npy'.format(path_sift_matching_result), matching_image_pair)
    print('generate matching image pair, running time: {}'.format(datetime.datetime.now() - start_time))

    # parallel feature extraction and save results
    print('---------------feature extraction---------------')
    start_time = datetime.datetime.now()
    parallel_feature_extraction(images_path=path_scaled_images, images_name=images_name,
                                max_sift_point_num=max_sift_points_num, sift_point_filter_flag=sift_point_filter_flag,
                                sift_data_path=path_sift_extraction_result)

    print('parallel feature extraction, running time: {}'.format(datetime.datetime.now() - start_time))

    # load sift extraction result and feature matching
    print('---------------feature matching---------------')
    start_time = datetime.datetime.now()
    matching_points_origin, matching_image_pair = parallel_feature_matching(
        images_path=path_scaled_images, matching_image_pair=matching_image_pair,
        sift_data_path=path_sift_extraction_result)
    print('parallel feature matching, running time: {}'.format(datetime.datetime.now() - start_time))

    # select sift points for constructing our optimization function
    print('---------------matching points selection---------------')
    start_time = datetime.datetime.now()
    matching_points_selected = matching_points_selection.matching_points_origin_top(
        matching_points_origin=matching_points_origin, points_num=selected_points_num)
    np.save('{}/matching_points_selected.npy'.format(path_sift_matching_result), matching_points_selected)
    print('select sift matching points, running time: {}'.format(datetime.datetime.now() - start_time))

    # affine transformation optimization
    print('---------------transformation optimization---------------')
    affine_transformation_vec = linear_affine_transformation_optimization(
        images_path, matching_image_pair, matching_points_selected, selected_points_num)

    # calculate projection RMSE
    Proj_RMSE_affine = utils.calculate_projection_RMSE(affine_transformation_vec, matching_image_pair,
                                                       matching_points_selected)

    print('projection RMSE of affine transformation: {}'.format(Proj_RMSE_affine))

    np.save('{}/affine_transformation_vec.npy'.format(path_results),
            affine_transformation_vec)

    # stitching
    print('---------------stitch aligned images---------------')
    start_time = datetime.datetime.now()
    transformation_vec = affine_transformation_vec
    img_corner_coords, stitching_img_corner_coords, mosaic_size, min_x, max_x, min_y, max_y = image_stitching.get_mosaic_size(
        path_scaled_images,
        transformation_vec)
    stitching_transformation_vec, warped_img, warped_img_valid = image_stitching.parallel_calculate_warped_img(
        path_scaled_images, images_name,
        stitching_img_corner_coords,
        img_h=int(img_h * scale),
        img_w=int(img_w * scale),
        mosaic_h=mosaic_size[1],
        mosaic_w=mosaic_size[0])

    mosaic = image_stitching.generate_mosaic(path_scaled_images, warped_img, mosaic_size)
    cv2.imwrite('{}/mosaic.png'.format(path_results), mosaic)
    print('stitch aligned images, running time: {}'.format(datetime.datetime.now() - start_time))

    # multi-frame graph-cut
    start_time = datetime.datetime.now()
    mosaic_sp, optimal_sp, sp_result, mosaic_graph_cut = Graph_cut.Multi_frame_graph_cut(
        images_name, stitching_transformation_vec, warped_img, warped_img_valid,
        img_h=int(img_h * scale),
        img_w=int(img_w * scale),
        mosaic_h=mosaic_size[1], mosaic_w=mosaic_size[0],
        compactness=slic_compactness,
        num_pixels_per_superpixel=slic_num_pixels_per_superpixel,
        path_results=path_results,
        max_coverage=max_coverage)
    print('superpixel based multi-frame graph-cut, running time: {}'.format(datetime.datetime.now() - start_time))

    t_end = time.time()
    print('total running time: {}s'.format((t_end - t_start)))
    print('done')
    sys.stdout = original
    log_file.close()
