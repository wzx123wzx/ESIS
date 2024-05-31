import os
import cv2


# parameters for code
def init_setting(image_path):
    settings = Configurations(image_path)
    return settings


class Configurations:
    def __init__(self, images_path):
        # basic parameters
        self.images_path = images_path
        images_name = os.listdir(images_path)
        img_tmp = cv2.imread("{}/{}".format(images_path, images_name[0]))
        self.scale = 0.2
        self.image_size = img_tmp.shape
        self.cores_num_to_use = 6
        # parameters for feature detecting
        self.max_sift_points_num = 50000
        self.sift_point_filter_flag = True
        self.min_sift_points_num = 30
        # parameters for feature matching
        self.perc_next_match = 0.7
        self.do_ransac = True
        self.ransac_max_trials = 100
        self.ransac_min_samples = 4  # minimum number of matching point pair to estimate transformation
        self.ransac_residual_threshold = 3
        self.ransac_min_points_num = 30
        self.ransac_stop_probability = 0.95
        self.ransac_save_ransac_result = False
        # parameters for matching points selection
        self.selected_points_num = 20
        self.mesh_x = 4
        self.mesh_y = 5
        # parameters for multi-frame graph cut
        self.slic_compactness = 10  # large value tend to make superpixel square
        self.slic_num_pixels_per_superpixel = 1000  # tend to set a large value
        self.max_coverage = 2
