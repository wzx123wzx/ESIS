# Multiple image stitching code (python)
# Wang zx
# 2023.12
import pickle
import sys
from tee import Tee
import cv2
import os
import Feature_matching
import Matching_image_pair_generation as Field_generation
import cv_utils
import image_stitching
import Multi_frame_graph_cut as MF_blend
import image_downsampling
import numpy as np
import Optimization_transformation as opt_tf
import time
import datetime
import Settings_manager
import Folder_manager
import Multi_frame_graph_cut_v4 as test
import frame_to_frame_graph_cut
import Multiple_frame_graph_cut_voronoi_diagram as test_voronoi

if __name__ == '__main__':
    t_start = time.time()
    # log_file = open('log.txt', 'w')
    # tee = Tee(sys.stdout, log_file)
    images_path = "E:/stitching_Dataset/DroneMapper_Golf9_May2016"
    Folder_settings = Folder_manager.init_setting()
    Folder_manager.mk_all_folders()
    Settings = Settings_manager.init_setting(images_path)

    # dataset downsampling
    start_time = datetime.datetime.now()
    image_downsampling.parallel_image_downsampling(images_path, scale_factor=Settings.scale,
                                                   downsampled_images_path=Folder_settings.path_images_downsampled,
                                                   cores_num_to_use=Settings.cores_num_to_use)
    print('Parallel image downsampling, running time: {}'.format(datetime.datetime.now() - start_time))

    # Matching image pair generation
    # start_time = datetime.datetime.now()
    # images_name = sorted(os.listdir(images_path))
    # matching_image_name_pair = Field_generation.get_all_neighbors_4NN(images_path)
    # matching_image_pair = Field_generation.matching_image_index_pair(images_path, matching_image_name_pair)
    # print('Generate matching image pair, running time: {}'.format(datetime.datetime.now() - start_time))

    # # Parallel feature detection and save results
    # start_time = datetime.datetime.now()
    # Feature_matching.parallel_feature_detecting(images_path=Folder_settings.path_images_downsampled,
    #                                             images_name=images_name,
    #                                             max_sift_point_num=Settings.max_sift_points_num,
    #                                             sift_point_filter_flag=Settings.sift_point_filter_flag,
    #                                             SIFT_data_path=Folder_settings.path_SIFT_detecting_result,
    #                                             cores_num_to_use=Settings.cores_num_to_use)
    # print('Parallel feature detecting, running time: {}'.format(datetime.datetime.now() - start_time))

    # load SIFT detection and feature matching
    # start_time = datetime.datetime.now()
    # matching_points_origin, matching_image_pair = Feature_matching.parallel_feature_matching(
    #     images_path=Folder_settings.path_images_downsampled, matching_image_pair=matching_image_pair,
    #     SIFT_data_path=Folder_settings.path_SIFT_detecting_result, cores_num_to_use=Settings.cores_num_to_use,
    #     perc_next_match=Settings.perc_next_match, do_ransac=Settings.do_ransac,
    #     ransac_max_trials=Settings.ransac_max_trials, ransac_min_samples=Settings.ransac_min_samples,
    #     ransac_residual_threshold=Settings.ransac_residual_threshold, min_sift_points_num=Settings.min_sift_points_num,
    #     min_ransac_points_num=Settings.ransac_min_points_num, ransac_stop_probability=Settings.ransac_stop_probability,
    #     ransac_save_ransac_result=Settings.ransac_save_ransac_result,
    #     path_ransac_results=Folder_settings.path_SIFT_matching_result)
    # print('Parallel feature matching, running time: {}'.format(datetime.datetime.now() - start_time))
    # np.save('{}/matching_image_pair.npy'.format(Folder_settings.path_SIFT_matching_result), matching_image_pair)

    # evenly distributed matching points generation
    # start_time = datetime.datetime.now()
    # matching_points_origin_uni = Feature_matching.matching_points_origin_uni(
    #     matching_points_origin=matching_points_origin, points_num=20, mesh_x=4, mesh_y=5)
    # print(
    #     'Generate uniform distribution matching points, running time: {}'.format(datetime.datetime.now() - start_time))
    # np.save('{}/matching_points_origin_uni.npy'.format(Folder_settings.path_SIFT_matching_result),
    #         matching_points_origin_uni)
    # matching_points_origin_uni = Feature_matching.matching_points_origin_top(matching_points_origin, 20)

    # affine transformation optimization
    # aff_tf_vec = opt_tf.lin_optimization_affine(images_path, matching_image_pair,
    #                                             matching_points_origin_uni, points_num=Settings.selected_points_num)
    # np.save('{}/optimal_affine_transformation.npy'.format(Folder_settings.path_transformation_optimization_result),
    #         aff_tf_vec)
    # np.save('aff_tf_vec.npy', aff_tf_vec)
    # RMSE_aff = cv_utils.calculate_RMSE(aff_tf_vec, matching_image_pair, matching_points_origin_uni,
    #                                    points_num=Settings.selected_points_num)
    # print(RMSE_aff)

    # projective transformation optimization
    # proj_tf_vec = opt_tf.nolin_optimization_projective(images_name, initialization=aff_tf_vec,
    #                                                    m_aff=1e-3, m_tr=1e-3, m_proj=1e-3,
    #                                                    matching_image_pair=matching_image_pair,
    #                                                    matching_points=matching_points_origin_uni,
    #                                                    points_num=Settings.selected_points_num)
    # np.save('{}/optimal_projective_transformation.npy'.format(Folder_settings.path_transformation_optimization_result),
    #         proj_tf_vec)
    #
    # RMSE_proj = cv_utils.calculate_RMSE(proj_tf_vec, matching_image_pair, matching_points_origin_uni,
    #                         points_num=Settings.selected_points_num)

    # stitching
    # aff_tf_vec = np.load('aff_tf_vec.npy')
    aff_tf_vec = np.load('Linux_results/max_coverage_2/10000/Golf/optimal_affine_transformation.npy')
    tf_vec = aff_tf_vec
    tf_vec, mosaic_size = image_stitching.get_mosaic_size(Folder_settings.path_images_downsampled, tf_vec)
    # mosaic = image_stitching.generate_mosaic(Folder_settings.path_images_downsampled, tf_vec, mosaic_size)
    # cv2.imwrite('mosaic.jpg', mosaic)
    with open('Linux_results/max_coverage_2/10000/Golf/images_name.pkl', 'rb') as f:
        images_name = pickle.load(f)
    frame_to_frame_graph_cut.parallel_calculate_save_warped_img(Folder_settings.path_images_downsampled,
                                                                images_name, tf_vec, mosaic_size[1], mosaic_size[0], 6)
    warped_img_valid = test.parallel_calculate_warped_img(images_path, images_name, tf_vec, 600, 800, mosaic_size[1],
                                                          mosaic_size[0],
                                                          cores_num_to_use=6, isimg=False)
    sum_valid = np.sum(warped_img_valid, axis=0)
    mosaic_valid = np.where(sum_valid > 0, True, False)
    np.save('mosaic_valid.npy', mosaic_valid)

    # Multi-frame graph-cut based on superpixel segmentation
    # mosaic_sp, optimal_sp, sp_result, mosaic_graph_cut = MF_blend.multi_frame_graph_cut_GCO_based(
    #     Folder_settings.path_images_downsampled, images_name, tf_vec,
    #     img_h=int(Settings.image_size[0] * Settings.scale),
    #     img_w=int(Settings.image_size[1] * Settings.scale),
    #     mosaic_h=mosaic_size[1], mosaic_w=mosaic_size[0],
    #     compactness=Settings.slic_compactness,
    #     num_pixels_per_superpixel=Settings.slic_num_pixels_per_superpixel,
    #     Folder_settings=Folder_settings,
    #     cores_num_to_use=Settings.cores_num_to_use,
    #     max_coverage=Settings.max_coverage)
    # cv2.imwrite('mosaic_graph_cut.jpg', mosaic_graph_cut)

    # mosaic_sp, optimal_sp, sp_result, mosaic_graph_cut = test.multi_frame_graph_cut(
    #     Folder_settings.path_images_downsampled, images_name, tf_vec,
    #     img_h=int(Settings.image_size[0] * Settings.scale),
    #     img_w=int(Settings.image_size[1] * Settings.scale),
    #     mosaic_h=mosaic_size[1], mosaic_w=mosaic_size[0],
    #     compactness=Settings.slic_compactness,
    #     num_pixels_per_superpixel=Settings.slic_num_pixels_per_superpixel,
    #     Folder_settings=Folder_settings,
    #     cores_num_to_use=Settings.cores_num_to_use,
    #     max_coverage=Settings.max_coverage)
    # cv2.imwrite('mosaic_graph_cut.jpg', mosaic_graph_cut)

    # mosaic_sp, optimal_sp, sp_result, mosaic_graph_cut = test.multi_frame_graph_cut(
    #     Folder_settings.path_images_downsampled, images_name, tf_vec,
    #     img_h=int(Settings.image_size[0] * Settings.scale),
    #     img_w=int(Settings.image_size[1] * Settings.scale),
    #     mosaic_h=mosaic_size[1], mosaic_w=mosaic_size[0],
    #     compactness=Settings.slic_compactness,
    #     num_pixels_per_superpixel=Settings.slic_num_pixels_per_superpixel,
    #     Folder_settings=Folder_settings,
    #     cores_num_to_use=Settings.cores_num_to_use,
    #     max_coverage=Settings.max_coverage)
    # cv2.imwrite('mosaic_graph_cut.jpg', mosaic_graph_cut)
    # cv2.imwrite('sp_result.jpg', sp_result)

    # labelled_mask, mosaic_vor = test_voronoi.multi_frame_graph_cut_vor(
    #     Folder_settings.path_images_downsampled, images_name, tf_vec,
    #     img_h=int(Settings.image_size[0] * Settings.scale),
    #     img_w=int(Settings.image_size[1] * Settings.scale),
    #     mosaic_h=mosaic_size[1], mosaic_w=mosaic_size[0],
    #     cores_num_to_use=Settings.cores_num_to_use)
    # cv2.imwrite('mosaic_vor.jpg', mosaic_vor)
    # np.save('labelled_mask.npy', labelled_mask)

    # start_time = datetime.datetime.now()
    # frame_to_frame_graph_cut_result = frame_to_frame_graph_cut.frame_to_frame_graph_cut(
    #     Folder_settings.path_images_downsampled, images_name, tf_vec,
    #     600, 800, mosaic_size[1], mosaic_size[0], cores_num_to_use=Settings.cores_num_to_use)
    # cv2.imwrite('frame_to_frame_graph_cut_result.jpg', frame_to_frame_graph_cut_result)
    # print('Frame-to-frame graph-cut, running time: {}'.format(datetime.datetime.now() - start_time))

    # calculate warped images
    # start_time = datetime.datetime.now()
    # warped_img = test.parallel_calculate_warped_img(images_path, images_name, tf_vec, 600, 800, mosaic_size[1],
    #                                                 mosaic_size[0],
    #                                                 cores_num_to_use=6, isimg=True)
    # print('Generate warped images, running time: {}'.format(datetime.datetime.now() - start_time))
    # np.save('warped_img.npy', warped_img)
    # start_time = datetime.datetime.now()

    # print('Generate valid region of warped images, running time: {}'.format(datetime.datetime.now() - start_time))

    # log_file.close()
    # tee.close()
    t_end = time.time()
    print('Total running time: {} s'.format((t_end - t_start)))
    print('done')
