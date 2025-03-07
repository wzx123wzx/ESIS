import os
import shutil


class dataset:
    images_path = '/home/kb249/wzx/Dataset/Gregg'
    scale = 0.2
    img_h = 3000
    img_w = 4000


class sift:
    class feature_extraction:
        max_sift_points_num = 10000
        sift_point_filter_flag = True
        min_sift_points_num = 30

    class feature_matching:
        perc_next_match = 0.7
        do_ransac = True
        ransac_max_trials = 100
        ransac_min_samples = 4  # minimum number of matching point pair to estimate transformation
        ransac_residual_threshold = 10
        ransac_min_points_num = 30
        ransac_stop_probability = 0.95
        ransac_save_ransac_result = False

    class points_selection:
        selected_points_num = 20

class superpixel_segmentation:
    slic_compactness = 10  # large value tend to make superpixel square
    slic_num_pixels_per_superpixel = 3000
    max_coverage = 2


class device:
    cores_num_to_use = 6


class path:
    path_scaled_images = 'scaled_images'
    path_results = 'results'
    path_sift_extraction_result = '{}/sift_extraction_result'.format(path_results)
    path_sift_matching_result = '{}/sift_matching_result'.format(path_results)
    path_ransac_result = '{}/ransac_result'.format(path_results)


def make_all_folders():
    if not os.path.exists(path.path_results):
        os.mkdir(path.path_results)
    else:
        shutil.rmtree(path.path_results)
        os.mkdir(path.path_results)

    if not os.path.exists(path.path_scaled_images):
        os.mkdir(path.path_scaled_images)
    else:
        shutil.rmtree(path.path_scaled_images)
        os.mkdir(path.path_scaled_images)

    if not os.path.exists(path.path_sift_extraction_result):
        os.mkdir(path.path_sift_extraction_result)
    else:
        shutil.rmtree(path.path_sift_extraction_result)
        os.mkdir(path.path_sift_extraction_result)

    if not os.path.exists(path.path_sift_matching_result):
        os.mkdir(path.path_sift_matching_result)
    else:
        shutil.rmtree(path.path_sift_matching_result)
        os.mkdir(path.path_sift_matching_result)

    if not os.path.exists(path.path_ransac_result):
        if sift.feature_matching.ransac_save_ransac_result:
            os.mkdir(path.path_ransac_result)
    else:
        shutil.rmtree(path.path_sift_matching_result)
        os.mkdir(path.path_sift_matching_result)
