# 3D-TC2 implementation based on MotionNet
# Defense against poisoned data

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import imageio
import argparse
import json
import pickle

from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from data.data_utils import voxelize_occupy, calc_displace_vector, point_in_hull_fast
from model import MotionNet, MotionNetMGDA, FeatEncoder


color_map = {0: 'c', 1: 'm', 2: 'k', 3: 'y', 4: 'r'}
cat_names = {0: 'bg', 1: 'bus', 2: 'ped', 3: 'bike', 4: 'other'}
box_cat_names = {0:['bus','car'], 1:['pedestrian'],2:['bicycle'],3:['truck', 'construction_vehicle', 'trailer', 'motorcycle', 'traffic_cone', 'barrier']}
        
def check_box_cat_id(label):
    for bid in box_cat_names:
        for cat in box_cat_names[bid]:
            if label == cat:
                # print('matched label: ',cat)
                return bid

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def points_2D_in_box(box: 'Box', points: np.ndarray, wlh_factor: float = 1.0):
    """
    Checks whether points are inside the box.
    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579
    :param box: <Box>.
    :param points: <np.float: 2, n>.
    :param wlh_factor: Inflates or deflates the box.
    :return: <np.bool: n, >.
    """
    corners = box.corners(wlh_factor=wlh_factor)

    p1 = corners[:, 0]
    p_x = corners[:, 4]
    p_y = corners[:, 1]
    # p_z = corners[:, 3]

    i = p_x - p1
    j = p_y - p1
    # k = p_z - p1

    v = points - p1.reshape((-1, 1)) 

    iv = np.dot(i, v)
    jv = np.dot(j, v)
    # kv = np.dot(k, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    # mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
    # mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
    mask = np.logical_and(mask_x, mask_y)

    return mask

def vis_scene_data(nuscenes_path=None, nuscenes_version='v1.0-trainval', which_scene=0, max_seq_num=10,
                   begin_frame=0, frame_skip=3, trained_model_path=None, img_save_dir=None, which_model='MotionNet',
                   use_adj_frame_pred=True, use_motion_state_pred_masking=True, disp=True, curr_sample_token=None):
    """
    Visualize the scene data.

    nuscenes_path: the path to the nuScenes dataset
    nuscenes_version: the dataset version ['v1.0-trainval'/'v1.0-mini']
    which_scene: for which we want to visualize
    max_seq_num: how many frames want to predict and visualize
    begin_frame: for this scene, from which frame we want to start our prediction
    frame_skip: how many future frames we want to skip. This is used for generated preprocessed bev data.
    trained_model_path: the path to the pretrained model
    img_save_dir: the directory for saving the predicted image
    which_model: which network ['MotionNet'/'MotionNetMGDA']
    use_adj_frame_pred: whether to predict the relative offsets between two adjacent frames
    use_motion_state_pred_masking: whether to threshold the prediction with motion state estimation results
    disp: whether to immediately show the predicted results
    """
    if nuscenes_path is None:
        raise ValueError("Should specify the nuScenes data path.")

    nusc = NuScenes(version=nuscenes_version, dataroot=nuscenes_path, verbose=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nsweeps_back = 20
    nsweeps_forward = 20
    num_frame_skipped = 0

    voxel_size = (0.25, 0.25, 0.4)
    area_extents = np.array([[-32., 32.], [-32., 32.], [-3, 2]])

    sample_cnt = 1
    class_map = {'vehicle.car': 1, 'vehicle.bus.rigid': 1, 'vehicle.bus.bendy': 1, 'human.pedestrian': 2,
                 'vehicle.bicycle': 3}  # background: 0, other: 4

    curr_scene = nusc.scene[which_scene]

    first_sample_token = curr_scene['first_sample_token']
    last_sample_token = curr_scene['last_sample_token']
    curr_sample = nusc.get('sample', first_sample_token)
    # curr_sample = nusc.get('sample', curr_sample_token)
    curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])
    # curr_sample_data = nusc.get('sample_data', curr_sample_data_token)    

    has_reached_last_keyframe = False
    seq_num = 0

    # Load pre-trained network weights
    loaded_models = list()
    if which_model == "MotionNet":
        model = MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=13)

        model = nn.DataParallel(model)
        checkpoint = torch.load(trained_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        loaded_models = [model]
    else:
        model_encoder = FeatEncoder()
        model_head = MotionNetMGDA(out_seq_len=20, motion_category_num=2)

        model_encoder = nn.DataParallel(model_encoder)
        model_head = nn.DataParallel(model_head)

        checkpoint = torch.load(trained_model_path)
        model_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model_head.load_state_dict(checkpoint['head_state_dict'])

        model_encoder = model_encoder.to(device)
        model_head = model_head.to(device)

        loaded_models = [model_encoder, model_head]
    print("Loaded pretrained model {}".format(which_model))

    # poisoned outputs for defense
    id_ref = 0 # car
    with open('detection/results_mark_attack_pp_car.pkl', 'rb') as f:
        detector_mark = pickle.load(f)
    # with open('detection/results_mark_attack_second_car.pkl', 'rb') as f:
    #     detector_mark = pickle.load(f)

    # id_ref = 2 # cyl
    # with open('detection/results_mark_attack_pp_cyl.pkl', 'rb') as f:
    #     detector_mark = pickle.load(f)
    # with open('detection/results_mark_attack_second_cyl.pkl', 'rb') as f:
    #      detector_mark = pickle.load(f)

    # id_ref = 1 # ped
    # with open('detection/results_mark_attack_pp_ped.pkl', 'rb') as f:
    #     detector_mark = pickle.load(f)
    # with open('detection/results_mark_attack_second_ped.pkl', 'rb') as f:
    #     detector_mark = pickle.load(f)

    print('current scene:',curr_scene['name'])
    img_save_dir = check_folder(os.path.join(img_save_dir, curr_scene['name']))

    while curr_sample_data['next'] != '':
        # print('curr_sample_data_token:',curr_sample_data['token'])
        # print('curr_sample_token:',curr_sample_data['sample_token'])
        if has_reached_last_keyframe:
            break

        # has reached the final keyframe
        if curr_sample_data['token'] == last_sample_token:
            has_reached_last_keyframe = True

        # Skip current keyframe if possible
        if num_frame_skipped > 0 and sample_cnt % (num_frame_skipped + 1) == 0:
            sample_cnt += 1
            curr_sample_data = nusc.get('sample_data', curr_sample_data['next'])
            continue

        # Get the synchronized point clouds
        all_pc, all_times,future_sample_data_token,future_sample_token,future_is_key_frame = LidarPointCloud.from_file_multisweep_bf_sample_data(nusc, curr_sample_data,
                                                                                nsweeps_back=nsweeps_back,
                                                                                nsweeps_forward=nsweeps_forward,id_ref=id_ref)

        # Store point cloud of each sweep
        pc = all_pc.points
        _, sort_idx = np.unique(all_times, return_index=True)
        unique_times = all_times[np.sort(sort_idx)]  # Preserve the item order in unique_times
        num_sweeps = len(unique_times)
        print('number_sweeps:',num_sweeps)

        # Make sure we have sufficient past and future sweeps
        if num_sweeps != (nsweeps_back + nsweeps_forward):
            sample_cnt += 1
            curr_sample_data = nusc.get('sample_data', curr_sample_data['next'])
            continue # only when num_sweeps = 40, move on

        # Prepare data dictionary for visualization
        save_data_dict = dict()

        for tid in range(num_sweeps):
            _time = unique_times[tid]
            points_idx = np.where(all_times == _time)[0]
            _pc = pc[:, points_idx]
            save_data_dict['pc_' + str(tid)] = _pc

        save_data_dict['times'] = unique_times
        save_data_dict['num_sweeps'] = num_sweeps

        # Get the synchronized bounding boxes
        # First, we need to iterate all the instances, and then retrieve their corresponding bounding boxes
        num_instances = 0  # The number of instances within this sample
        corresponding_sample_token = curr_sample_data['sample_token']
        corresponding_sample_data_token = curr_sample_data['token']
        curr_is_key_frame = curr_sample_data['is_key_frame'] # is current frame the key frame
        corresponding_sample_rec = nusc.get('sample', corresponding_sample_token)

        # Get bounding boxes from detectors (nusc format, global coordinates)
        # Get reference pose and timestamp
        ref_pose_rec = nusc.get('ego_pose', curr_sample_data['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', curr_sample_data['calibrated_sensor_token'])

        # Get the bounding boxes in the future frame
        target_box = list()
        # if future_is_key_frame:
        #     for p in pointpillar['results'][future_sample_token]:
        #         box = Box(center=p['translation'], size=p['size'], orientation=Quaternion(p['rotation']))
        #         box.label=p['detection_name']
        #         # Move box to ego vehicle coord system
        #         box.translate(-np.array(ref_pose_rec['translation']))
        #         box.rotate(Quaternion(ref_pose_rec['rotation']).inverse)
        #         # Move box to sensor coord system
        #         box.translate(-np.array(ref_cs_rec['translation']))
        #         box.rotate(Quaternion(ref_cs_rec['rotation']).inverse)
        #         target_box.append(box)
        #     print('Transformed {} detector boxes for MotionNet future_sample {}:'.format(len(target_box),future_sample_token) )

        if future_is_key_frame:
            for cat in detector_mark:
                for p in cat:
                    if p.sample_token == future_sample_token:
                        box = Box(center=p.translation, size=p.size, orientation=Quaternion(p.rotation))
                        box.label=p.detection_name
                        box.name = p.attribute_name
                        box.token = p.sample_token
                        # Move box to ego vehicle coord system
                        box.translate(-np.array(ref_pose_rec['translation']))
                        box.rotate(Quaternion(ref_pose_rec['rotation']).inverse)
                        # Move box to sensor coord system
                        box.translate(-np.array(ref_cs_rec['translation']))
                        box.rotate(Quaternion(ref_cs_rec['rotation']).inverse)
                        # convert back would also work
                        # 
                        target_box.append(box)
            print('Transformed {} detector boxes for MotionNet future_sample {}:'.format(len(target_box),future_sample_token) )

        for ann_token in corresponding_sample_rec['anns']:
            ann_rec = nusc.get('sample_annotation', ann_token)
            category_name = ann_rec['category_name']

            flag = False
            for c, v in class_map.items():
                if category_name.startswith(c):
                    save_data_dict['category_' + str(num_instances)] = v
                    flag = True
                    break
            if not flag:
                save_data_dict['category_' + str(num_instances)] = 4  # Other category

            instance_token = ann_rec['instance_token']

            # get all boxes in 40 frames
            instance_boxes, instance_all_times, i_, j_ = LidarPointCloud. \
                get_instance_boxes_multisweep_sample_data(nusc, curr_sample_data,
                                                          instance_token,
                                                          nsweeps_back=nsweeps_back,
                                                          nsweeps_forward=nsweeps_forward) 

            assert np.array_equal(unique_times, instance_all_times), "The sweep and instance times are not consistent!"
            assert num_sweeps == len(instance_boxes), "The number of instance boxes does not match that of sweeps!"

            # Each row corresponds to a box annotation; the column consists of box center, box size, and quaternion
            box_data = np.zeros((len(instance_boxes), 3 + 3 + 4), dtype=np.float32)
            box_data.fill(np.nan)
            for r, box in enumerate(instance_boxes):
                if box is not None:
                    row = np.concatenate([box.center, box.wlh, box.orientation.elements])
                    box_data[r] = row[:]

            # Save the box data for current instance
            save_data_dict['instance_boxes_' + str(num_instances)] = box_data
            num_instances += 1

        save_data_dict['num_instances'] = num_instances

        if seq_num < begin_frame:
            seq_num += 1
            sample_cnt += 1
            print("Finish loading sequence sample {}".format(seq_num))
            continue

        # ------------------------------------ Visualization ------------------------------------
        # -- The following code is simply borrowed from gen_data.py and currently not optimized
        num_sweeps = save_data_dict['num_sweeps']
        times = save_data_dict['times']
        num_past_sweeps = len(np.where(times >= 0)[0])
        num_future_sweeps = len(np.where(times < 0)[0])
        assert num_past_sweeps + num_future_sweeps == num_sweeps, "The number of sweeps is incorrect!"


        # Load point cloud
        pc_list = []

        for i in range(num_sweeps):
            pc = save_data_dict['pc_' + str(i)]
            pc_list.append(pc.T)

        # for all 40 prepared frames
        # Reorder the pc, and skip sample frames if wanted
        tmp_pc_list_1 = pc_list[0:num_past_sweeps:(frame_skip + 1)]
        tmp_pc_list_1 = tmp_pc_list_1[::-1]
        tmp_pc_list_2 = pc_list[(num_past_sweeps + frame_skip)::(frame_skip + 1)]
        pc_list = tmp_pc_list_1 + tmp_pc_list_2

        # after skipping, only 10 pcs remains in the pc_list, 5 for the past(including the current one), 5 for the future
        num_past_pcs = len(tmp_pc_list_1)
        num_future_pcs = len(tmp_pc_list_2)

        # Voxelize the input point clouds, and compute the ground truth displacement vectors
        padded_voxel_points_list = list()  # This contains the compact representation of voxelization, as in the paper

        for i in range(num_past_pcs):
            res = voxelize_occupy(pc_list[i], voxel_size=voxel_size, extents=area_extents)
            padded_voxel_points_list.append(res)

        # Compile the batch of voxels for the past pcs, so that they can be fed into the network
        padded_voxel_points = torch.from_numpy(np.stack(padded_voxel_points_list, axis=0))

        # Finally, generate the ground-truth displacement field corresponding to the pc_list
        all_disp_field_gt, all_valid_pixel_maps, non_empty_map, pixel_cat_map \
            = gen_2d_grid_gt_for_visualization(save_data_dict, grid_size=voxel_size[0:2], reordered=True,
                                               extents=area_extents, frame_skip=frame_skip)

        bev_input_data = (padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps, non_empty_map,
                          pixel_cat_map, num_past_pcs, num_future_pcs)

        if future_is_key_frame:
            vis_model_per_sample_data(bev_input_data, save_data_dict, frame_skip=frame_skip, loaded_models=loaded_models,
                                  voxel_size=voxel_size, which_model=which_model, model_path=trained_model_path,
                                  img_save_dir=img_save_dir, use_adj_frame_pred=use_adj_frame_pred, disp=disp,
                                  use_motion_state_pred_masking=use_motion_state_pred_masking, frame_idx=seq_num,corresponding_sample_token=corresponding_sample_token, future_sample_token = future_sample_token,  future_sample_data_token = future_sample_data_token, target_box = target_box,id_ref=id_ref)
                                                
        seq_num += 1
        print("Finish loading sequence sample {}".format(seq_num))

        sample_cnt += 1
        curr_sample_data = nusc.get('sample_data', curr_sample_data['next'])

        if seq_num - begin_frame >= max_seq_num:
            break


def gen_2d_grid_gt_for_visualization(data_dict: dict, grid_size: np.array, extents: np.array = None,
                                     frame_skip: int = 0, reordered: bool = False, proportion_thresh: float = 0.5,
                                     category_num: int = 5, one_hot_thresh: float = 0.8):
    """
    Generate the 2d grid ground-truth for the input point cloud.
    The ground-truth is: the displacement vectors of the occupied pixels in BEV image.
    The displacement is computed w.r.t the current time and the future time

    The difference between this function and gen_2d_grid_gt: the input is "data_dict" instead of sample file path.

    :param data_dict: The dictionary storing point cloud data and annotations
    :param grid_size: The size of each pixel
    :param extents: The extents of the point cloud on the 2D xy plane. Shape (3, 2)
    :param frame_skip: The number of sample frames that need to be skipped
    :param reordered: Whether need to reorder the results, so that the first element corresponds to the oldest record.
    :param proportion_thresh: Within a given pixel, only when the proportion of foreground points exceeds this threshold
        will we compute the displacement vector for this pixel.
    :param category_num: The number of categories for points.
    :param one_hot_thresh: When the proportion of the majority points within a cell exceeds this threshold, we
        compute the (hard) one-hot category vector for this cell, otherwise compute the soft category vector.

    :return: The ground-truth displacement field. Shape (num_sweeps, image height, image width, 2).
    """
    num_sweeps = data_dict['num_sweeps']
    times = data_dict['times']
    num_past_sweeps = len(np.where(times >= 0)[0])
    num_future_sweeps = len(np.where(times < 0)[0])
    assert num_past_sweeps + num_future_sweeps == num_sweeps, "The number of sweeps is incorrect!"

    pc_list = []

    for i in range(num_sweeps):
        pc = data_dict['pc_' + str(i)]
        pc_list.append(pc.T)

    # Retrieve the instance boxes
    num_instances = data_dict['num_instances']
    instance_box_list = list()
    instance_cat_list = list()  # for instance categories

    for i in range(num_instances):
        instance = data_dict['instance_boxes_' + str(i)]
        category = data_dict['category_' + str(i)]
        instance_box_list.append(instance)
        instance_cat_list.append(category)

    # ----------------------------------------------------
    # Filter and sort the reference point cloud
    refer_pc = pc_list[0]
    refer_pc = refer_pc[:, 0:3]

    if extents is not None:
        if extents.shape != (3, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents.shape))

        filter_idx = np.where((extents[0, 0] < refer_pc[:, 0]) & (refer_pc[:, 0] < extents[0, 1]) &
                              (extents[1, 0] < refer_pc[:, 1]) & (refer_pc[:, 1] < extents[1, 1]) &
                              (extents[2, 0] < refer_pc[:, 2]) & (refer_pc[:, 2] < extents[2, 1]))[0]
        refer_pc = refer_pc[filter_idx]

    # -- Discretize pixel coordinates to given quantization size
    discrete_pts = np.floor(refer_pc[:, 0:2] / grid_size).astype(np.int32)

    # -- Use Lex Sort, sort by x, then y
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    sorted_order = np.lexsort((y_col, x_col))

    refer_pc = refer_pc[sorted_order]
    discrete_pts = discrete_pts[sorted_order]

    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1])))

    # -- The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # -- Sort unique indices to preserve order
    unique_indices.sort()
    pixel_coords = discrete_pts[unique_indices]

    # -- Number of points per voxel, last voxel calculated separately
    num_points_in_pixel = np.diff(unique_indices)
    num_points_in_pixel = np.append(num_points_in_pixel, discrete_pts.shape[0] - unique_indices[-1])

    # -- Compute the minimum and maximum voxel coordinates
    if extents is not None:
        min_pixel_coord = np.floor(extents.T[0, 0:2] / grid_size)
        max_pixel_coord = np.ceil(extents.T[1, 0:2] / grid_size) - 1
    else:
        min_pixel_coord = np.amin(pixel_coords, axis=0)
        max_pixel_coord = np.amax(pixel_coords, axis=0)

    # -- Get the voxel grid dimensions
    num_divisions = ((max_pixel_coord - min_pixel_coord) + 1).astype(np.int32)

    # -- Bring the min voxel to the origin
    pixel_indices = (pixel_coords - min_pixel_coord).astype(int)
    #pixel_indices = (pixel_coords - min_pixel_coord)
    # print(min_pixel_coord,max_pixel_coord)
    # ----------------------------------------------------

    # ----------------------------------------------------
    # Get the point cloud subsets, which are inside different instance bounding boxes
    refer_box_list = list()
    refer_pc_idx_per_bbox = list()
    points_category = np.zeros(refer_pc.shape[0], dtype=np.int)  # store the point categories

    for i in range(num_instances):
        instance_cat = instance_cat_list[i]
        instance_box = instance_box_list[i]
        instance_box_data = instance_box[0]
        assert not np.isnan(instance_box_data).any(), "In the keyframe, there should not be NaN box annotation!"

        tmp_box = Box(center=instance_box_data[:3], size=instance_box_data[3:6],
                      orientation=Quaternion(instance_box_data[6:]))
        idx = point_in_hull_fast(refer_pc[:, 0:3], tmp_box)
        refer_pc_idx_per_bbox.append(idx)
        refer_box_list.append(tmp_box)

        points_category[idx] = instance_cat

    if len(refer_pc_idx_per_bbox) > 0:
        refer_pc_idx_inside_box = np.concatenate(refer_pc_idx_per_bbox).tolist()
    else:
        refer_pc_idx_inside_box = []
    refer_pc_idx_outside_box = set(range(refer_pc.shape[0])) - set(refer_pc_idx_inside_box)
    refer_pc_idx_outside_box = list(refer_pc_idx_outside_box)

    # Compute pixel (cell) categories
    pixel_cat = np.zeros([unique_indices.shape[0], category_num], dtype=np.float32)
    most_freq_info = []

    for h, v in enumerate(zip(unique_indices, num_points_in_pixel)):
        pixel_elements_categories = points_category[v[0]:v[0] + v[1]]
        elements_freq = np.bincount(pixel_elements_categories, minlength=category_num)
        assert np.sum(elements_freq) == v[1], "The frequency count is incorrect."

        elements_freq = elements_freq / float(v[1])
        most_freq_cat, most_freq = np.argmax(elements_freq), np.max(elements_freq)
        most_freq_info.append([most_freq_cat, most_freq])

        if most_freq >= one_hot_thresh:
            one_hot_cat = np.zeros(category_num, dtype=np.float32)
            one_hot_cat[most_freq_cat] = 1.0
            pixel_cat[h] = one_hot_cat
        else:
            pixel_cat[h] = elements_freq  # we use soft category probability vector.

    pixel_cat_map = np.zeros((num_divisions[0], num_divisions[1], category_num), dtype=np.float32)
    pixel_cat_map[pixel_indices[:, 0], pixel_indices[:, 1]] = pixel_cat[:]

    # Set the non-zero pixels to 1.0
    # Note that the non-zero pixels correspond to the foreground and background objects
    non_empty_map = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.float32)
    non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0

    # Compute the displacement vectors w.r.t. the other sweeps
    all_disp_field_gt_list = list()
    zero_disp_field = np.zeros((num_divisions[0], num_divisions[1], 2), dtype=np.float32)
    all_disp_field_gt_list.append(zero_disp_field)

    all_valid_pixel_maps_list = list()  # valid pixel map will be used for masking the computation of loss
    all_valid_pixel_maps_list.append(non_empty_map)

    # -- Skip some frames if necessary
    past_part = list(range(0, num_past_sweeps, frame_skip + 1))
    future_part = list(range(num_past_sweeps + frame_skip, num_sweeps, frame_skip + 1))
    frame_considered = np.asarray(past_part + future_part)

    for i in frame_considered[1:]:
        curr_disp_vectors = np.zeros_like(refer_pc, dtype=np.float32)
        curr_disp_vectors.fill(np.nan)
        curr_disp_vectors[refer_pc_idx_outside_box, ] = 0.0

        # First, for each instance, compute the corresponding points displacement.
        for j in range(num_instances):
            instance_box = instance_box_list[j]
            instance_box_data = instance_box[i]  # This is for the i-th sweep

            if np.isnan(instance_box_data).any():  # It is possible that in this sweep there is no annotation
                continue

            tmp_box = Box(center=instance_box_data[:3], size=instance_box_data[3:6],
                          orientation=Quaternion(instance_box_data[6:]))
            pc_in_bbox_idx = refer_pc_idx_per_bbox[j]
            disp_vectors = calc_displace_vector(refer_pc[pc_in_bbox_idx], refer_box_list[j], tmp_box)

            curr_disp_vectors[pc_in_bbox_idx] = disp_vectors[:]

        # Second, compute the mean displacement vector and category for each non-empty pixel
        disp_field = np.zeros([unique_indices.shape[0], 2], dtype=np.float32)  # we only consider the 2D field

        # We only compute loss for valid pixels where there are corresponding box annotations between two frames
        valid_pixels = np.zeros(unique_indices.shape[0], dtype=np.bool)

        for h, v in enumerate(zip(unique_indices, num_points_in_pixel)):

            # Only when the number of majority points exceeds predefined proportion, we compute
            # the displacement vector for this pixel. Otherwise, We consider it is background (possibly ground plane)
            # and has zero displacement.
            pixel_elements_categories = points_category[v[0]:v[0] + v[1]]
            most_freq_cat, most_freq = most_freq_info[h]

            if most_freq >= proportion_thresh:
                most_freq_cat_idx = np.where(pixel_elements_categories == most_freq_cat)[0]
                most_freq_cat_disp_vectors = curr_disp_vectors[v[0]:v[0] + v[1], :3]
                most_freq_cat_disp_vectors = most_freq_cat_disp_vectors[most_freq_cat_idx]

                if np.isnan(most_freq_cat_disp_vectors).any():  # contains invalid disp vectors
                    valid_pixels[h] = 0.0
                else:
                    mean_disp_vector = np.mean(most_freq_cat_disp_vectors, axis=0)
                    disp_field[h] = mean_disp_vector[0:2]  # ignore the z direction

                    valid_pixels[h] = 1.0

        # Finally, assemble to a 2D image
        disp_field_sparse = np.zeros((num_divisions[0], num_divisions[1], 2), dtype=np.float32)
        disp_field_sparse[pixel_indices[:, 0], pixel_indices[:, 1]] = disp_field[:]

        valid_pixel_map = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.float32)
        valid_pixel_map[pixel_indices[:, 0], pixel_indices[:, 1]] = valid_pixels[:]

        all_disp_field_gt_list.append(disp_field_sparse)
        all_valid_pixel_maps_list.append(valid_pixel_map)

    all_disp_field_gt_list = np.stack(all_disp_field_gt_list, axis=0)
    all_valid_pixel_maps_list = np.stack(all_valid_pixel_maps_list, axis=0)

    if reordered:
        num_past = len(past_part)
        all_disp_field_gt_list[0:num_past] = all_disp_field_gt_list[(num_past - 1)::-1]
        all_valid_pixel_maps_list[0:num_past] = all_valid_pixel_maps_list[(num_past - 1)::-1]

    return all_disp_field_gt_list, all_valid_pixel_maps_list, non_empty_map, pixel_cat_map


def vis_model_per_sample_data(bev_input_data, data_dict, frame_skip=3, voxel_size=(0.25, 0.25, 0.4),
                              loaded_models=None, which_model="MotionNet", model_path=None, img_save_dir=None,
                              use_adj_frame_pred=False, use_motion_state_pred_masking=False, frame_idx=0, disp=True,corresponding_sample_token='', future_sample_token='',future_sample_data_token='', target_box=list(),id_ref=0):
    """
    Visualize the prediction (ie, displacement field) results.

    bev_input_data: the preprocessed sparse bev data
    data_dict: a dictionary storing the point cloud data and annotations
    frame_skip: how many frames we want to skip for future frames
    voxel_size: the size of each voxel
    loaded_models: the model which has loaded the pretrained weights
    which_model: which model to apply ['MotionNet'/'MotionNetMGDA']
    model_path: the path to the pretrained model
    img_save_dir: the directory for saving the predicted image
    use_adj_frame_pred: whether to predict the relative offsets between two adjacent frames
    use_motion_state_pred_masking: whether to threshold the prediction with motion state estimation results
    frame_idx: used for specifying the name of saved image frames
    disp: whether to immediately show the predicted results
    corresponding_sample_token: current sample token
    future_sample_token: future sample token to predict
    target_box: detected (from detectors like pointpillar and secondet) bounding boxes for target sample
    """

    if model_path is None:
        raise ValueError("Need to specify saved model path.")
    if img_save_dir is None:
        raise ValueError("Need to specify image save path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig, ax = plt.subplots(1, 3, figsize=(20, 8))

    # Load pre-trained network weights
    if which_model == "MotionNet":
        model = loaded_models[0]
    else:
        model_encoder = loaded_models[0]
        model_head = loaded_models[1]

    # Prepare data for the network
    padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps,\
        non_empty_map, pixel_cat_map_gt, num_past_pcs, num_future_pcs = bev_input_data

    #prepare the past 5 pcs extracting from 20 past pcs
    padded_voxel_points = torch.unsqueeze(padded_voxel_points, 0).to(device)

    # Make prediction
    if which_model == "MotionNet":
        model.eval()
    else:
        model_encoder.eval()
        model_head.eval()

    with torch.no_grad():
        if which_model == "MotionNet":
            disp_pred, cat_pred, motion_pred = model(padded_voxel_points)
        else:
            shared_feats = model_encoder(padded_voxel_points)
            disp_pred, cat_pred, motion_pred = model_head(shared_feats)

        disp_pred = disp_pred.cpu().numpy() # prediction for 20 future pcs
        disp_pred = np.transpose(disp_pred, (0, 2, 3, 1))
        cat_pred = np.squeeze(cat_pred.cpu().numpy(), 0)
        # print('>>>>>>>>>>>>>>>>>>>>')
        # print(len(disp_pred),len(cat_pred),len(motion_pred))

        #=============save the initial prediction result===================
        # import codecs,json
        # disp_pred_save = disp_pred.tolist()
        # cat_pred_save = cat_pred.tolist()
        # motion_pred_save = motion_pred.tolist()
        # res_path = str(img_save_dir) + '/disp_pred' + str(frame_idx) + '.json'
        # json.dump(disp_pred_save, codecs.open(res_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        # res_path = str(img_save_dir) + '/cat_pred' + str(frame_idx) + '.json'
        # json.dump(cat_pred_save, codecs.open(res_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        # res_path = str(img_save_dir) + '/motion_pred' + str(frame_idx) + '.json'
        # json.dump(motion_pred_save, codecs.open(res_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


        if use_adj_frame_pred:  # The prediction are the displacement between adjacent frames
            for c in range(1, disp_pred.shape[0]):
                disp_pred[c, ...] = disp_pred[c, ...] + disp_pred[c - 1, ...]

        if use_motion_state_pred_masking:
            motion_pred_numpy = motion_pred.cpu().numpy()
            motion_pred_numpy = np.argmax(motion_pred_numpy, axis=1)
            motion_mask = motion_pred_numpy == 0

            cat_pred_numpy = np.argmax(cat_pred, axis=0)
            cat_mask = np.logical_and(cat_pred_numpy == 0, non_empty_map == 1)
            cat_mask = np.expand_dims(cat_mask, 0)

            cat_weight_map = np.ones_like(motion_pred_numpy, dtype=np.float32)
            cat_weight_map[motion_mask] = 0.0
            cat_weight_map[cat_mask] = 0.0
            cat_weight_map = cat_weight_map[:, :, :, np.newaxis]  # (1, h, w. 1)

            disp_pred = disp_pred * cat_weight_map

    # ------------------------- Visualization -------------------------
    # --- Load the point cloud data and annotations ---
    #num_sweeps = 40, num_past_sweeps = 20, num_future_sweeps = 20
    num_sweeps = data_dict['num_sweeps'] 
    times = data_dict['times']
    num_past_sweeps = len(np.where(times >= 0)[0])
    num_future_sweeps = len(np.where(times < 0)[0])
    # print('num_past/future_sweeps',num_past_sweeps,num_future_sweeps)
    assert num_past_sweeps + num_future_sweeps == num_sweeps, "The number of sweeps is incorrect!"

    # Load point cloud
    pc_list = []

    for i in range(num_sweeps):
        pc = data_dict['pc_' + str(i)]
        pc_list.append(pc)

    # Reorder the pc, and skip sample frames if wanted
    tmp_pc_list_1 = pc_list[0:num_past_sweeps:(frame_skip + 1)] # In past 20 frames, for every 4 frames, choose 1 frame (skip 3 frames) 
    tmp_pc_list_1 = tmp_pc_list_1[::-1]
    tmp_pc_list_2 = pc_list[(num_past_sweeps + frame_skip)::(frame_skip + 1)]# In future 20 frames, for every 4 frames, choose 1 frame (skip 3 frames) 
    pc_list = tmp_pc_list_1 + tmp_pc_list_2 # in all 10 pcs, 5 in the past(5th is the curr frame), 5 in the future
    num_past_pcs = len(tmp_pc_list_1) # num_past_pcs = 5 frames

    # Load box annotations, and reorder and skip some annotations if wanted
    num_instances = data_dict['num_instances']
    instance_box_list = list()

    for i in range(num_instances):
        instance = data_dict['instance_boxes_' + str(i)]

        # Reorder the boxes
        tmp_instance = np.zeros((len(pc_list), instance.shape[1]), dtype=np.float32)
        tmp_instance[(num_past_pcs - 1)::-1] = instance[0:num_past_sweeps:(frame_skip + 1)]
        tmp_instance[num_past_pcs:] = instance[(num_past_sweeps + frame_skip)::(frame_skip + 1)]
        instance = tmp_instance[:]

        instance_box_list.append(instance)

    # Draw the LIDAR and quiver plots
    # The distant points are very sparse and not reliable. We do not show them.
    border_meter = 4
    border_pixel = border_meter * 4
    x_lim = [-(32 - border_meter), (32 - border_meter)]
    y_lim = [-(32 - border_meter), (32 - border_meter)]

    # We only show the cells having one-hot category vectors
    max_prob = np.amax(pixel_cat_map_gt, axis=-1)
    filter_mask = max_prob == 1.0
    pixel_cat_map = np.argmax(pixel_cat_map_gt, axis=-1) + 1  # category starts from 1 (background), etc
    pixel_cat_map = (pixel_cat_map * non_empty_map * filter_mask).astype(np.int)

    cat_pred = np.argmax(cat_pred, axis=0) + 1
    cat_pred = (cat_pred * non_empty_map * filter_mask).astype(np.int)

    # --- Visualization ---
    idx = num_past_pcs - 1
    points = pc_list[idx] # idx = 4,

    ax[0].scatter(points[0, :], points[1, :], c=points[2, :], s=1)
    ax[0].set_xlim(x_lim[0], x_lim[1]) #[-28,28]
    ax[0].set_ylim(y_lim[0], y_lim[1])
    ax[0].axis('on')
    ax[0].set_aspect('equal')
    # ax[0].title.set_text('LIDAR data ')
    ax[0].title.set_text('sample_token {}'.format(corresponding_sample_token))

    # Plot quiver. We only show non-empty vectors. Plot each category.
    field_gt = all_disp_field_gt[-1] # the 10th ground truth, also the 20th future frames
    idx_x = np.arange(field_gt.shape[0])
    idx_y = np.arange(field_gt.shape[1])
    idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
    qk = [None] * len(color_map)  # for quiver key

    # check if the predicted cells are in the bounding box of the same category
    sample_MO_tp_GT_tp = np.zeros((len(box_cat_names),), dtype=int)
    sample_MO_tp_GT_fp = np.zeros((len(box_cat_names),), dtype=int)
    sample_MO_fp_GT_tp = np.zeros((len(box_cat_names),), dtype=int)
    sample_MO_fp_GT_fp = np.zeros((len(box_cat_names),), dtype=int)

    for box in target_box:
        # position requiements
        X_re = box.center[0]>=-8 and box.center[0]<=8
        Y_re = box.center[1]>= 8 and box.center[1]<=30
        pos_re = X_re and Y_re
        box_cat_id = check_box_cat_id(box.label)
        cat_re = box_cat_id==id_ref
        
        if pos_re and cat_re:
            num_of_each_category = np.zeros((len(color_map),), dtype=int)
            for k in range(len(color_map)):
                # ------------------------ Ground-truth ------------------------
                mask = pixel_cat_map == (k + 1)
                # For cells with very small movements, we threshold them to be static
                field_gt_norm = np.linalg.norm(field_gt, ord=2, axis=-1)  # out: (h, w)
                thd_mask = field_gt_norm <= 0.4
                field_gt[thd_mask, :] = 0

                # Get the displacement field
                min_pixel_coord = -128
                X = (idx_x[mask] + min_pixel_coord) * voxel_size[0]
                Y = (idx_y[mask] + min_pixel_coord) * voxel_size[1]
                U = field_gt[:, :, 0][mask]
                V = field_gt[:, :, 1][mask]

                qk[k] = ax[1].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color=color_map[k])
                ax[1].quiverkey(qk[k], X=0.0 + k/5.0, Y=1.1, U=20, label=cat_names[k], labelpos='E')
                ax[1].set_xlim((border_pixel + min_pixel_coord) * voxel_size[0], (field_gt.shape[0] - border_pixel + min_pixel_coord) * voxel_size[0])
                ax[1].set_ylim((border_pixel + min_pixel_coord) * voxel_size[1], (field_gt.shape[1] - border_pixel + min_pixel_coord) * voxel_size[1])
                ax[1].set_aspect('equal')
                ax[1].title.set_text('Ground-truth')
                ax[1].axis('on')

                # ------------------------ Prediction ------------------------
                # Show the prediction results. We show the cells corresponding to the non-empty one-hot gt cells.
                mask_pred = cat_pred == (k + 1)
                field_pred = disp_pred[-1]  # Show last prediction, the 20-th future frame


                # For cells with very small movements, we threshold them to be static
                field_pred_norm = np.linalg.norm(field_pred, ord=2, axis=-1)  # out: (h, w)
                thd_mask = field_pred_norm <= 0.4
                field_pred[thd_mask, :] = 0

                # We use the same indices as the ground-truth, since we are currently focused on the foreground
                X_pred = (idx_x[mask_pred] + min_pixel_coord) * voxel_size[0]
                Y_pred = (idx_y[mask_pred] + min_pixel_coord) * voxel_size[1]
                U_pred = field_pred[:, :, 0][mask_pred]
                V_pred = field_pred[:, :, 1][mask_pred]

                #for type k, scatter all predtected cells
                X_cell = X_pred+U_pred
                Y_cell = Y_pred+V_pred
                ax[2].scatter(X_cell, Y_cell, c=color_map[k], s=1)
                ax[2].set_xlim((border_pixel+ min_pixel_coord) * voxel_size[0], (field_pred.shape[0] - border_pixel+ min_pixel_coord) * voxel_size[0]) #Set the x-axis view limits
                ax[2].set_ylim((border_pixel+ min_pixel_coord) * voxel_size[1], (field_pred.shape[1] - border_pixel+ min_pixel_coord) * voxel_size[1]) #Set the y-axis view limits
                # ax[2].set_xlim(-128,128)
                # ax[2].set_ylim(-128,128)
                ax[2].set_aspect('equal')
                ax[2].title.set_text('future_sample_data_token:{}'.format(future_sample_data_token))
                ax[2].axis('on')

                # go through all predicted cells in cat k 
                Z_cell = np.zeros(len(X_cell))
                pred_cells = []
                pred_cells.append(X_cell) 
                pred_cells.append(Y_cell)
                pred_cells.append(Z_cell)
                points_in_box = points_2D_in_box(box,pred_cells) # or use points_3D_in_box, given z_cell = box.h
                pts_counts = 0
                for pib in points_in_box:
                    if pib == True:
                        pts_counts += 1

                num_of_each_category[k] = pts_counts
            
            max_cell_cat_id = 0
            pts_compare = 0
            # for cell_cat_id in range(len(color_map)):
            #     if cell_cat_id > 0: # exclude bg cells
            #         if num_of_each_category[cell_cat_id] > pts_compare:
            #             pts_compare = num_of_each_category[cell_cat_id]
            #             max_cell_cat_id = cell_cat_id
            #             print('max_cell_cat:', cat_names[max_cell_cat_id])
            for cell_cat_id in range(len(color_map)):
                if num_of_each_category[cell_cat_id] > pts_compare:
                    # if cell_cat_id > 0: # exclude bg cells. 
                    pts_compare = num_of_each_category[cell_cat_id]
                    max_cell_cat_id = cell_cat_id
            print('max_cell_cat:', cat_names[max_cell_cat_id]) # if the box only contains bg cells, max_cell_cat_id = 0
            
            if max_cell_cat_id == box_cat_id+1: # if box label and max cell cat match
                box.render(ax[2], colors=('g', 'g', 'g'), linewidth=1)
                if box.name == 'TP':
                    scene_MO_tp_GT_tp[box_cat_id] += 1
                if box.name == 'FP':
                    scene_MO_tp_GT_fp[box_cat_id] += 1
            else:
                box.render(ax[2], colors=('r', 'r', 'r'), linewidth=2)
                if box.name == 'TP':
                    scene_MO_fp_GT_tp[box_cat_id] += 1
                if box.name == 'FP':
                    scene_MO_fp_GT_fp[box_cat_id] += 1

            fusion_matrix[0].append(scene_MO_tp_GT_tp[box_cat_id])
            fusion_matrix[1].append(scene_MO_tp_GT_fp[box_cat_id])
            fusion_matrix[2].append(scene_MO_fp_GT_tp[box_cat_id])
            fusion_matrix[3].append(scene_MO_fp_GT_fp[box_cat_id])
            
            # print at box level
            print('box_label:',cat_names[box_cat_id+1])
            print('curr sum of boxes: scene_MO_tp_GT_tp:{}, scene_MO_tp_GT_fp:{}, scene_MO_fp_GT_tp:{}, scene_MO_fp_GT_fp:{} )'.format(scene_MO_tp_GT_tp[box_cat_id],scene_MO_tp_GT_fp[box_cat_id],scene_MO_fp_GT_tp[box_cat_id],scene_MO_fp_GT_fp[box_cat_id]))
    #-------add ground-truth bounding boxes on the raw data and predictions----------
    for j in range(num_instances):
        inst = instance_box_list[j]

        curr_box_data = inst[idx]
        if np.isnan(curr_box_data).any():
            continue

        # add MotionNet current gt boxes on raw lidar data
        box = Box(center=curr_box_data[0:3], size=curr_box_data[3:6], orientation=Quaternion(curr_box_data[6:]))
        box.render(ax[0])
        # box.render(ax[1])

        future_box_data = inst[idx+num_future_pcs]
        if np.isnan(future_box_data).any():
            continue

        # add MotionNet future gt boxes on predictions
        box = Box(center=future_box_data[0:3], size=future_box_data[3:6], orientation=Quaternion(future_box_data[6:]))
        # box.render(ax[0], colors=('b', 'b', 'b'), linewidth=2)
        box.render(ax[1], colors=('b', 'b', 'b'), linewidth=1)
        # box.render(ax[2], colors=('b', 'b', 'b'), linewidth=1)

    # if future_sample_token == '00889f8a9549450aa2f32cf310a3e305':
    plt.savefig(os.path.join(img_save_dir, future_sample_token + '.png'))
    print('saved prediction for {}'.format(future_sample_token))
    print("finish sample {}".format(frame_idx))

 
    if disp:
        plt.pause(0.02)
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    plt.close(fig)

    if disp:
        plt.show()

def gen_scene_prediction_video(images_dir, output_dir, out_format='mp4'):
    images = [im for im in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, im))
              and im.endswith('.png')]

    num_images = len(images)

    if out_format == 'gif':
        save_gif_path = os.path.join(output_dir, 'result.gif')
        with imageio.get_writer(save_gif_path, mode='I', fps=20) as writer:
            for i in range(num_images):
                image_file = os.path.join(images_dir, str(i) + '.png')
                image = imageio.imread(image_file)
                writer.append_data(image)

                print("Appending image {}".format(i))
    else:
        save_mp4_path = os.path.join(output_dir, 'result.mp4')
        with imageio.get_writer(save_mp4_path, fps=15, quality=10, pixelformat='yuvj444p') as writer:
            for i in range(num_images):
                image_file = os.path.join(images_dir, str(i) + '.png')
                image = imageio.imread(image_file)
                writer.append_data(image)

                print("Appending image {}".format(i))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default=None, type=str, help='The path to nuScenes dataset')
    parser.add_argument('-v', '--version', default='v1.0-trainval', type=str, help='The version of nuScenes dataset')
    parser.add_argument('-l', '--savepath', default=None, type=str, help='Directory for saving the generated images')
    parser.add_argument('-n', '--nframe', default=10, type=int, help='The number of frames to be generated')
    parser.add_argument('-s', '--scene', default=5, type=int, help='Which scene')
    parser.add_argument('--net', default='MotionNet', type=str, help='Which network [MotionNet/MotionNetMGDA]')
    parser.add_argument('--modelpath', default=None, type=str, help='Path to the pretrained model')
    parser.add_argument('--beginframe', default=0, type=int, help='From which frame we start predicting')
    parser.add_argument('--format', default='gif', type=str, help='The output animation format [gif/mp4]')

    parser.add_argument('--video', action='store_true', help='Whether to generate images or [gif/mp4]')
    parser.add_argument('--adj', action='store_false', help='Whether predict the relative offset between frames')
    parser.add_argument('--disp', action='store_true', help='Whether to immediately show the images')
    parser.add_argument('--jitter', action='store_false', help='Whether to apply jitter suppression')
    parser.add_argument('--csdt',default=None,type=str,help='curr_sample_data_token')
    parser.add_argument('--cst',default=None,type=str,help='curr_sample_token')

    args = parser.parse_args()

    gen_prediction_frames = not args.video
    if_disp = args.disp
    image_save_dir = check_folder(args.savepath)

    if gen_prediction_frames:
        if not if_disp:
            matplotlib.use('AGG')

        scene_MO_tp_GT_tp = np.zeros((len(box_cat_names),), dtype=int)
        scene_MO_tp_GT_fp = np.zeros((len(box_cat_names),), dtype=int)
        scene_MO_fp_GT_tp = np.zeros((len(box_cat_names),), dtype=int)
        scene_MO_fp_GT_fp = np.zeros((len(box_cat_names),), dtype=int)
        fusion_matrix = [[],[],[],[]]

        nframe = 500
        for scene in range(10):
            vis_scene_data(nuscenes_path=args.data, nuscenes_version=args.version, img_save_dir=image_save_dir,
                           trained_model_path=args.modelpath, which_model=args.net, which_scene=scene,
                           frame_skip=3, max_seq_num=nframe, use_adj_frame_pred=args.adj,
                           disp=if_disp, use_motion_state_pred_masking=args.jitter, begin_frame=args.beginframe,curr_sample_token=args.cst)
            
        fm = np.array(fusion_matrix)
        TPs = fm[0]+fm[3]
        GTs = fm[0]+fm[1]+fm[2]+fm[3]
        # recall = TPs/GTs # curr recall
        recall = TPs/GTs[-1] # normal recall
        box_num = np.arange(len(recall))
        print('final recall: ',recall[-1])

        fig_rec, ax_rec = plt.subplots()
        ax_rec.plot(box_num, recall)
        ax_rec.set(xlabel='box num', ylabel='recall',title='recall - box number')

        img_save_dir = check_folder(os.path.join(image_save_dir))
        fig_rec.savefig(os.path.join(img_save_dir,'recall_all_scenes.png'))

        res_path = os.path.join(img_save_dir, 'results_recall.pkl')
        with open(res_path, 'wb') as f:
            pickle.dump(recall, f)     

    else:
        frames_dir = os.path.join('/media/pwu/Data/3D_data/nuscene/logs/images_job_talk', image_save_dir)
        save_gif_dir = os.path.join('/media/pwu/Data/3D_data/nuscene/logs/images_job_talk', image_save_dir)
        gen_scene_prediction_video(args.savepath, args.savepath, out_format='gif')

