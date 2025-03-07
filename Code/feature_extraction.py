import os
import cv2
import multiprocessing
import pickle
import utils
import constant

# device
cores_num_to_use = constant.device.cores_num_to_use


def feature_extraction_helper(args):
    # sift feature extraction and save results (pickle)
    images_path = args[0]
    images_name = args[1]
    img_idx = args[2]
    max_sift_point_num = args[3]
    sift_point_filter_flag = args[4]
    sift_data_path = args[5]
    img = cv2.imread("{0}/{1}".format(images_path, images_name[img_idx]))
    kp, desc = utils.get_sift_points(img, max_sift_point_num, sift_point_filter_flag)
    kp_list = []

    for p in kp:
        kp_list.append(p.pt)  # only coordinates are stored

    if not os.path.exists("{0}/{1}".format(sift_data_path, images_name[img_idx])):
        os.mkdir("{0}/{1}".format(sift_data_path, images_name[img_idx]))
    file = open("{0}/{1}/{2}".format(sift_data_path, images_name[img_idx], 'kp.pickle'), 'wb')
    pickle.dump(kp_list, file)
    file.close()
    file = open("{0}/{1}/{2}".format(sift_data_path, images_name[img_idx], 'desc.pickle'), 'wb')
    pickle.dump(desc, file)
    file.close()

    return


def parallel_feature_extraction(images_path, images_name, max_sift_point_num, sift_point_filter_flag, sift_data_path):
    # parallel sift extraction
    args = []
    for img_idx in range(len(images_name)):
        args.append((images_path, images_name, img_idx, max_sift_point_num, sift_point_filter_flag, sift_data_path))

    processes = multiprocessing.Pool(cores_num_to_use)
    results = processes.map(feature_extraction_helper, args)
    processes.close()
    return results
