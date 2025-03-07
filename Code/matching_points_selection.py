import numpy as np


def matching_points_origin_top(matching_points_origin, points_num):
    # select top inliers
    match_num = len(matching_points_origin)
    matching_points_top = []
    for match_index in range(match_num):
        matching_points_origin_match_index = matching_points_origin[match_index]
        matching_points_top.append(np.zeros((4, points_num)))
        matching_points_top[match_index][0, :] = matching_points_origin_match_index[0, 0:points_num]
        matching_points_top[match_index][1, :] = matching_points_origin_match_index[1, 0:points_num]
        matching_points_top[match_index][2, :] = matching_points_origin_match_index[2, 0:points_num]
        matching_points_top[match_index][3, :] = matching_points_origin_match_index[3, 0:points_num]

    return matching_points_top
