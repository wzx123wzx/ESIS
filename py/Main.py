# Large-scale Image Stitching based on Global Registration Optimization and Multi-frame Graph-cut at Superpixel level
# Wang zx
import sys
import cv2
import Feature_matching
import Matching_image_pair_generation as Field_generation
import cv_utils
import image_stitching
import image_downsampling
import numpy as np
import Optimization_transformation as opt_tf
import time
import datetime
import Settings_manager
import Folder_manager
import os
import Graph_cut

if __name__ == '__main__':
    t_start = time.time()
    images_path = "E:/stitching_Dataset/4thAveReservoir"
    images_name = os.listdir(images_path)
    Folder_settings = Folder_manager.init_setting()
    Folder_manager.mk_all_folders()
    Settings = Settings_manager.init_setting(images_path)
    original = sys.stdout
    log_file = open('{}/logs.txt'.format(Folder_settings.path_results), "w")
    sys.stdout = log_file
    print('---------------image downsampling, feature detecting and matching---------------')
    # dataset downsampling
    print('number of images: {}'.format(len(images_name)))
    start_time = datetime.datetime.now()
    image_downsampling.parallel_image_downsampling(images_path, scale_factor=Settings.scale,
                                                   downsampled_images_path=Folder_settings.path_images_downsampled,
                                                   cores_num_to_use=Settings.cores_num_to_use)
    print('Parallel image downsampling, running time: {}'.format(datetime.datetime.now() - start_time))

    # Matching image pair generation
    start_time = datetime.datetime.now()
    matching_image_name_pair = Field_generation.get_all_neighbors_4NN(images_path)
    matching_image_pair = Field_generation.matching_image_index_pair(images_path, matching_image_name_pair)
    print('Generate matching image pair, running time: {}'.format(datetime.datetime.now() - start_time))

    # Parallel feature detection and save results
    start_time = datetime.datetime.now()
    Feature_matching.parallel_feature_detecting(images_path=Folder_settings.path_images_downsampled,
                                                images_name=images_name,
                                                max_sift_point_num=Settings.max_sift_points_num,
                                                sift_point_filter_flag=Settings.sift_point_filter_flag,
                                                SIFT_data_path=Folder_settings.path_SIFT_detecting_result,
                                                cores_num_to_use=Settings.cores_num_to_use)
    print('Parallel feature detecting, running time: {}'.format(datetime.datetime.now() - start_time))

    # load sift detection and feature matching
    start_time = datetime.datetime.now()
    matching_points_origin, matching_image_pair = Feature_matching.parallel_feature_matching(
        images_path=Folder_settings.path_images_downsampled, matching_image_pair=matching_image_pair,
        SIFT_data_path=Folder_settings.path_SIFT_detecting_result, cores_num_to_use=Settings.cores_num_to_use,
        perc_next_match=Settings.perc_next_match, do_ransac=Settings.do_ransac,
        ransac_max_trials=Settings.ransac_max_trials, ransac_min_samples=Settings.ransac_min_samples,
        ransac_residual_threshold=Settings.ransac_residual_threshold, min_sift_points_num=Settings.min_sift_points_num,
        min_ransac_points_num=Settings.ransac_min_points_num, ransac_stop_probability=Settings.ransac_stop_probability,
        ransac_save_ransac_result=Settings.ransac_save_ransac_result,
        path_ransac_results=Folder_settings.path_SIFT_matching_result)
    print('number of matching image pair: {}'.format(len(matching_image_pair)))
    print('Parallel feature matching, running time: {}'.format(datetime.datetime.now() - start_time))
    np.save('{}/matching_image_pair.npy'.format(Folder_settings.path_SIFT_matching_result), matching_image_pair)

    # select sift points for constructing our optimization function
    matching_points_selected = Feature_matching.matching_points_origin_top(
        matching_points_origin=matching_points_origin, points_num=20)
    np.save('{}/matching_points_selected.npy'.format(Folder_settings.path_SIFT_matching_result),
            matching_points_selected)

    # affine transformation optimization

    aff_tf_vec = opt_tf.lin_optimization_affine(images_path, matching_image_pair,
                                                matching_points_selected, points_num=Settings.selected_points_num)
    RMSE_aff = cv_utils.calculate_RMSE(aff_tf_vec, matching_image_pair, matching_points_selected,
                                       points_num=Settings.selected_points_num)
    print('RMSE: {}'.format(RMSE_aff))
    np.save('{}/optimal_affine_transformation.npy'.format(Folder_settings.path_results),
            aff_tf_vec)

    # stitching
    tf_vec = aff_tf_vec
    tf_vec, mosaic_size = image_stitching.get_mosaic_size(Folder_settings.path_images_downsampled, tf_vec)
    warped_img = image_stitching.parallel_calculate_warped_img(Folder_settings.path_images_downsampled, images_name,
                                                               tf_vec,
                                                               img_h=int(Settings.image_size[0] * Settings.scale),
                                                               img_w=int(Settings.image_size[1] * Settings.scale),
                                                               mosaic_h=mosaic_size[1], mosaic_w=mosaic_size[0],
                                                               cores_num_to_use=Settings.cores_num_to_use,
                                                               isimg=True)
    warped_img_valid = image_stitching.parallel_calculate_warped_img(Folder_settings.path_images_downsampled,
                                                                     images_name,
                                                                     tf_vec,
                                                                     img_h=int(Settings.image_size[0] * Settings.scale),
                                                                     img_w=int(Settings.image_size[1] * Settings.scale),
                                                                     mosaic_h=mosaic_size[1], mosaic_w=mosaic_size[0],
                                                                     cores_num_to_use=Settings.cores_num_to_use,
                                                                     isimg=False)
    np.save('warped_img.npy', warped_img)
    np.save('warped_img_valid.npy', warped_img_valid)
    mosaic = image_stitching.generate_mosaic(Folder_settings.path_images_downsampled, warped_img, mosaic_size)
    cv2.imwrite('{}/mosaic.jpg'.format(Folder_settings.path_results), mosaic)

    # Multi-frame graph-cut
    start_time = datetime.datetime.now()
    mosaic_sp, optimal_sp, sp_result, mosaic_graph_cut = Graph_cut.Multi_frame_graph_cut(
        images_name, tf_vec,
        warped_img=warped_img,
        warped_img_valid=warped_img_valid,
        img_h=int(Settings.image_size[0] * Settings.scale),
        img_w=int(Settings.image_size[1] * Settings.scale),
        mosaic_h=mosaic_size[1], mosaic_w=mosaic_size[0],
        compactness=Settings.slic_compactness,
        num_pixels_per_superpixel=Settings.slic_num_pixels_per_superpixel,
        Folder_settings=Folder_settings,
        cores_num_to_use=Settings.cores_num_to_use,
        max_coverage=Settings.max_coverage)
    print('Superpixel based multi-frame graph-cut, running time: {}'.format(datetime.datetime.now() - start_time))

    t_end = time.time()
    print('Total running time: {} s'.format((t_end - t_start)))
    print('done')
    sys.stdout = original
    log_file.close()
