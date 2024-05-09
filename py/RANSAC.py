import numpy as np
from skimage.measure import ransac
from skimage.transform import AffineTransform
import matplotlib.pyplot as plt
import os
import cv2


# RANSAC

def RANSAC_matching_points(images_path, matching_points_origin, min_samples, residual_threshold, max_trials,
                           matching_image_pair, min_points_num, stop_probability, dir_ransac_results,
                           save_ransac_result=False):
    # RANSAC matching points filtering
    x0 = matching_points_origin[:, 0]
    y0 = matching_points_origin[:, 1]
    x1 = matching_points_origin[:, 2]
    y1 = matching_points_origin[:, 3]
    src_points = np.stack([x0, y0]).T
    dst_points = np.stack([x1, y1]).T

    model, inliers = ransac((src_points, dst_points), AffineTransform, min_samples=min_samples,
                            residual_threshold=residual_threshold, max_trials=max_trials,
                            stop_probability=stop_probability)
    x0_ransac = []
    y0_ransac = []
    x1_ransac = []
    y1_ransac = []
    for idx in range(len(x0)):
        if inliers[idx]:
            x0_ransac.append(x0[idx])
            y0_ransac.append(y0[idx])
            x1_ransac.append(x1[idx])
            y1_ransac.append(y1[idx])

    x0_ransac = np.array(x0_ransac)
    y0_ransac = np.array(y0_ransac)
    x1_ransac = np.array(x1_ransac)
    y1_ransac = np.array(y1_ransac)

    matching_points_ransac = np.zeros((4, len(x0_ransac)))
    matching_points_ransac[0, :] = x0_ransac
    matching_points_ransac[1, :] = y0_ransac
    matching_points_ransac[2, :] = x1_ransac
    matching_points_ransac[3, :] = y1_ransac
    if len(x0_ransac) > min_points_num:
        if save_ransac_result:
            images_name = os.listdir(images_path)
            img1 = cv2.imread("{0}/{1}".format(images_path, images_name[matching_image_pair[0]]))
            img2 = cv2.imread("{0}/{1}".format(images_path, images_name[matching_image_pair[1]]))
            fig, ax = plt.subplots()
            ax.axis('off')
            ax.imshow(np.hstack((img1, img2)), cmap='gray')
            ax.plot(np.vstack((src_points[inliers, 0], dst_points[inliers, 0] + img1.shape[1])),
                    np.vstack((src_points[inliers, 1], dst_points[inliers, 1])), '-r')
            plt.savefig('{}/{}-{}.jpg'.format(dir_ransac_results, images_name[matching_image_pair[0]],
                                                                  images_name[matching_image_pair[1]]))
            plt.close()
        return matching_points_ransac
    else:
        return False
