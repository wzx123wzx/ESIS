import cv2
import os
import multiprocessing
import constant

# device
cores_num_to_use = constant.device.cores_num_to_use


def image_scaling_helper(args):
    # image scaling
    images_path = args[0]
    img_idx = args[1]
    scale_factor = args[2]
    scaled_image_path = args[3]
    images_name = os.listdir(images_path)
    img = cv2.imread("{0}/{1}".format(images_path, images_name[img_idx]))
    scaled_img = cv2.resize(
        img,
        (
            int(scale_factor * img.shape[1]),
            int(scale_factor * img.shape[0]),
        ),
    )
    cv2.imwrite(scaled_image_path, scaled_img)
    return


def parallel_image_scaling(images_path, scale, scaled_images_path):
    # parallel image scaling
    images_name = os.listdir(images_path)
    args = []
    for img_idx in range(len(images_name)):
        args.append((images_path, img_idx, scale, "{0}/{1}".format(scaled_images_path, images_name[img_idx])))
    processes = multiprocessing.Pool(cores_num_to_use)
    processes.map(image_scaling_helper, args)
    processes.close()
    return
