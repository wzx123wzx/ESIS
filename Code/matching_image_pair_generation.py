import numpy as np
import os
import sys
import math
from GPSPhoto import gpsphoto
import utils


def get_GPS_coords(images_path):
    # returns the GPS information of the dataset (dict)
    image_path_list = os.listdir(images_path)
    GPS_coords = {}
    for image_name in image_path_list:
        gps = gpsphoto.getGPSData("{}/{}".format(images_path, image_name))
        GPS_coords[image_name] = {}
        if "Latitude" in gps:
            GPS_coords[image_name]["lat"] = gps["Latitude"]
            GPS_coords[image_name]["lon"] = gps["Longitude"]
            GPS_coords[image_name]["alt"] = gps["Altitude"]
        else:
            GPS_coords[image_name]["lat"] = None
            GPS_coords[image_name]["lon"] = None
            GPS_coords[image_name]["alt"] = None

    return GPS_coords


def select_4NN(image_name, coords):
    # return the nearest 4 images of current image (4 directions)
    uni_vec_1 = np.array([1, 0])
    uni_vec_2 = np.array([0, 1])
    uni_vec_3 = np.array([-1, 0])
    uni_vec_4 = np.array([0, -1])
    lat_ref = coords[image_name]["lat"]
    lon_ref = coords[image_name]["lon"]
    neighbor_1 = []
    neighbor_2 = []
    neighbor_3 = []
    neighbor_4 = []
    np.seterr(divide='ignore', invalid='ignore')
    for image_name_match in coords.keys():
        if image_name_match is not image_name:
            uni_vec = np.array([coords[image_name_match]["lat"] - lat_ref,
                                coords[image_name_match]["lon"] - lon_ref])
            uni_vec = uni_vec / np.linalg.norm(uni_vec)
            angle_1 = math.acos(np.sum(uni_vec * uni_vec_1))
            angle_2 = math.acos(np.sum(uni_vec * uni_vec_2))
            angle_3 = math.acos(np.sum(uni_vec * uni_vec_3))
            angle_4 = math.acos(np.sum(uni_vec * uni_vec_4))
            if coords[image_name_match]["lat"] > lat_ref and abs(angle_1) < math.pi / 4:
                neighbor_1.append(image_name_match)
            if coords[image_name_match]["lon"] > lon_ref and abs(angle_2) < math.pi / 4:
                neighbor_2.append(image_name_match)
            if coords[image_name_match]["lat"] < lat_ref and abs(angle_3) < math.pi / 4:
                neighbor_3.append(image_name_match)
            if coords[image_name_match]["lon"] < lon_ref and abs(angle_4) < math.pi / 4:
                neighbor_4.append(image_name_match)

    neighbor_4NN = []
    dist1 = {}
    min_dist = sys.maxsize
    image_name_match = 0
    for i in range(0, len(neighbor_1)):
        p_num_ten = neighbor_1[i]
        dist1[p_num_ten] = utils.get_gps_distance(lat_ref, lon_ref,
                                                  coords[p_num_ten]["lat"],
                                                  coords[p_num_ten]["lon"])
        if dist1[p_num_ten] < min_dist:
            min_dist = dist1[p_num_ten]
            image_name_match = p_num_ten
    if min_dist is not sys.maxsize:
        neighbor_4NN.append(image_name_match)

    dist2 = {}
    min_dist = sys.maxsize
    image_name_match = 0
    for i in range(0, len(neighbor_2)):
        p_num_ten = neighbor_2[i]
        dist2[p_num_ten] = utils.get_gps_distance(lat_ref, lon_ref,
                                                  coords[p_num_ten]["lat"],
                                                  coords[p_num_ten]["lon"])
        if dist2[p_num_ten] < min_dist:
            min_dist = dist2[p_num_ten]
            image_name_match = p_num_ten
    if min_dist is not sys.maxsize:
        neighbor_4NN.append(image_name_match)

    dist3 = {}
    min_dist = sys.maxsize
    image_name_match = 0
    for i in range(0, len(neighbor_3)):
        p_num_ten = neighbor_3[i]
        dist3[p_num_ten] = utils.get_gps_distance(lat_ref, lon_ref,
                                                  coords[p_num_ten]["lat"],
                                                  coords[p_num_ten]["lon"])
        if dist3[p_num_ten] < min_dist:
            min_dist = dist3[p_num_ten]
            image_name_match = p_num_ten
    if min_dist is not sys.maxsize:
        neighbor_4NN.append(image_name_match)

    dist4 = {}
    min_dist = sys.maxsize
    image_name_match = 0
    for i in range(0, len(neighbor_4)):
        p_num_ten = neighbor_4[i]
        dist4[p_num_ten] = utils.get_gps_distance(lat_ref, lon_ref,
                                                  coords[p_num_ten]["lat"],
                                                  coords[p_num_ten]["lon"])
        if dist4[p_num_ten] < min_dist:
            min_dist = dist4[p_num_ten]
            image_name_match = p_num_ten
    if min_dist is not sys.maxsize:
        neighbor_4NN.append(image_name_match)

    return neighbor_4NN


def get_all_4NN_neighbors(images_path):
    matching_image_name_pair = []
    images_name = os.listdir(images_path)
    GPS_coords = get_GPS_coords(images_path)
    for image_name in images_name:
        neighbor_4NN = select_4NN(image_name, GPS_coords)
        for neighbor_image in neighbor_4NN:
            current_match_image_pair = [image_name, neighbor_image]
            if not repeat_check(current_match_image_pair, matching_image_name_pair):
                matching_image_name_pair.append(current_match_image_pair)

    return matching_image_name_pair


def repeat_check(match_image_pair_check, matching_image_pair):
    repeat_flag = False
    for i in range(0, len(matching_image_pair)):
        if match_image_pair_check[0] == matching_image_pair[i][0] and match_image_pair_check[1] == \
                matching_image_pair[i][1]:
            repeat_flag = True
        if match_image_pair_check[1] == matching_image_pair[i][0] and match_image_pair_check[0] == \
                matching_image_pair[i][1]:
            repeat_flag = True
    return repeat_flag


def index_image(images_path, img_name):
    # return image index from image name
    images_path_list = os.listdir(images_path)
    img_idx = 0

    for path_idx in range(len(images_path_list)):
        if images_path_list[path_idx] == img_name:
            img_idx = path_idx
            break

    return img_idx


def matching_image_index_pair(images_path, matching_image_name_pair):
    match_num = len(matching_image_name_pair)
    matching_image_pair = np.zeros((match_num, 2), dtype=int)
    for match_index in range(match_num):
        img1_name = matching_image_name_pair[match_index][0]
        img1_index = index_image(images_path, img1_name)
        img2_name = matching_image_name_pair[match_index][1]
        img2_index = index_image(images_path, img2_name)
        matching_image_pair[match_index, 0] = int(img1_index)
        matching_image_pair[match_index, 1] = int(img2_index)
    return matching_image_pair
