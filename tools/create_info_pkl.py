# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.data_converter import nuscenes_converter as nuscenes_converter


map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def get_gt(info, disable_valid_point_filter=False):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes):
            continue
        if (not disable_valid_point_filter and
                ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels


def get_gt_with_tokens(info, disable_valid_point_filter=False):
    """Generate gt labels and aligned annotation tokens from info.

    Returns:
        tuple[list, list, list, list]: gt_boxes, gt_labels, gt_ann_tokens,
        gt_instance_tokens
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    gt_ann_tokens = list()
    gt_instance_tokens = list()
    for ann_info in info['ann_infos']:
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes):
            continue
        if (not disable_valid_point_filter and
                ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
        gt_ann_tokens.append(ann_info['token'])
        gt_instance_tokens.append(ann_info['instance_token'])
    return gt_boxes, gt_labels, gt_ann_tokens, gt_instance_tokens


def should_disable_valid_point_filter(metadata):
    sample_timestamp_to_sec = float(metadata.get('sample_timestamp_to_sec', 1e-6))
    return sample_timestamp_to_sec >= 1e-5


def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       can_bus_path=None,
                       max_sweeps=10,
                       train_half=False,
                       only_split='all'):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path,
        info_prefix,
        version=version,
        max_sweeps=max_sweeps,
        train_half=train_half,
        can_bus_path=can_bus_path,
        only_split=only_split)


def add_ann_adj_info(extra_tag,
                     dataroot,
                     nuscenes_version,
                     train_half=False,
                     only_split='all'):
    # nuscenes_version = 'v1.0-trainval'
    nuscenes = NuScenes(nuscenes_version, dataroot)
    if train_half:
        data_set = ['half_train']
    else:
        data_set = ['train', 'val']
    if only_split == 'train':
        data_set = [split for split in data_set if split in ('train', 'half_train')]
    elif only_split == 'val':
        data_set = [split for split in data_set if split == 'val']
    for set in data_set:
        dataset = pickle.load(
            open('%s/%s_infos_%s.pkl' % (dataroot, extra_tag, set), 'rb'))
        metadata = dataset.get('metadata', {})
        disable_valid_point_filter = should_disable_valid_point_filter(
            metadata)
        sample_timestamp_to_sec = float(
            metadata.get('sample_timestamp_to_sec', 1e-6))
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])
            ann_infos = list()
            for ann in sample['anns']:
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes_converter._box_velocity_with_timestamp_scale(
                    nuscenes, ann_info['token'], sample_timestamp_to_sec)
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)
            dataset['infos'][id]['ann_infos'] = ann_infos
            gt_boxes, gt_labels, gt_ann_tokens, gt_instance_tokens = \
                get_gt_with_tokens(
                    dataset['infos'][id],
                    disable_valid_point_filter=disable_valid_point_filter)
            dataset['infos'][id]['ann_infos'] = (gt_boxes, gt_labels)
            dataset['infos'][id]['gt_ann_tokens'] = gt_ann_tokens
            dataset['infos'][id]['gt_instance_tokens'] = gt_instance_tokens
            dataset['infos'][id]['scene_token'] = sample['scene_token']
            dataset['infos'][id]['prev'] = sample['prev']
            scene = nuscenes.get('scene', sample['scene_token'])
            dataset['infos'][id]['occ_path'] = \
                '%s/gts/%s/%s'%(dataroot, scene['name'], info['token'])
        with open('%s/%s_infos_%s.pkl' % (dataroot, extra_tag, set),
                  'wb') as fid:
            pickle.dump(dataset, fid)

def parse_args():
    parser = argparse.ArgumentParser(description='Create BEVDet/STCOcc data infos')
    parser.add_argument(
        '--root-path',
        default='./data/v2xreal_nuscenes',
        help='dataset root path')
    parser.add_argument(
        '--extra-tag',
        default='v2xreal',
        help='prefix used in generated info filenames')
    parser.add_argument(
        '--version',
        default='v1.0-trainval',
        help='nuScenes version, e.g. v1.0-trainval or v1.0-mini')
    parser.add_argument(
        '--max-sweeps',
        type=int,
        default=0,
        help='number of sweeps used when building infos')
    parser.add_argument(
        '--train-half',
        action='store_true',
        help='only generate half-train split')
    parser.add_argument(
        '--can-bus-path',
        default=None,
        help='optional CAN bus path')
    parser.add_argument(
        '--only-split',
        choices=['all', 'train', 'val'],
        default='all',
        help='only rebuild the requested split pkl')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_half = args.train_half
    train_version = args.version
    root_path = args.root_path
    extra_tag = args.extra_tag
    can_bus_path = args.can_bus_path

    nuscenes_data_prep(
        train_half=train_half,
        root_path=root_path,
        info_prefix=extra_tag,
        version=train_version,
        can_bus_path=can_bus_path,
        max_sweeps=args.max_sweeps,
        only_split=args.only_split)

    print('add_ann_infos')
    add_ann_adj_info(
        extra_tag,
        dataroot=root_path,
        nuscenes_version=train_version,
        train_half=train_half,
        only_split=args.only_split)
