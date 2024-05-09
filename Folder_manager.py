import os
import shutil

global settings


class Configurations:
    def __init__(self):
        self.path_images_downsampled = 'images_downsampled'
        self.path_results = 'results'
        self.path_SIFT_detecting_result = '{}/SIFT_detection_result'.format(self.path_results)
        self.path_SIFT_matching_result = '{}/SIFT_matching_result'.format(self.path_results)


def init_setting():
    global settings
    settings = Configurations()
    return settings


def mk_all_folders():
    if not os.path.exists(settings.path_results):
        os.mkdir(settings.path_results)
    else:
        shutil.rmtree(settings.path_results)
        os.mkdir(settings.path_results)

    if not os.path.exists(settings.path_images_downsampled):
        os.mkdir(settings.path_images_downsampled)
    else:
        shutil.rmtree(settings.path_images_downsampled)
        os.mkdir(settings.path_images_downsampled)

    if not os.path.exists(settings.path_SIFT_detecting_result):
        os.mkdir(settings.path_SIFT_detecting_result)
    else:
        shutil.rmtree(settings.path_SIFT_detecting_result)
        os.mkdir(settings.path_SIFT_detecting_result)

    if not os.path.exists(settings.path_SIFT_matching_result):
        os.mkdir(settings.path_SIFT_matching_result)
    else:
        shutil.rmtree(settings.path_SIFT_matching_result)
        os.mkdir(settings.path_SIFT_matching_result)

