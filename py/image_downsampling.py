import cv2
import os
import multiprocessing


def image_downsampling_helper(args):
    # image downsampling
    images_path = args[0]
    img_idx = args[1]
    scale_factor = args[2]
    downsampled_image_path = args[3]
    images_name = os.listdir(images_path)
    img = cv2.imread("{0}/{1}".format(images_path, images_name[img_idx]))
    img_downsampling = cv2.resize(
        img,
        (
            int(scale_factor * img.shape[1]),
            int(scale_factor * img.shape[0]),
        ),
    )
    cv2.imwrite(downsampled_image_path, img_downsampling)
    return


def parallel_image_downsampling(images_path, scale_factor, downsampled_images_path, cores_num_to_use):
    # parallel image downsampling
    images_name = os.listdir(images_path)
    args = []
    for img_idx in range(len(images_name)):
        args.append((images_path, img_idx, scale_factor, "{0}/{1}".format(downsampled_images_path, images_name[img_idx])))
    processes = multiprocessing.Pool(cores_num_to_use)
    processes.map(image_downsampling_helper, args)
    processes.close()
    return
