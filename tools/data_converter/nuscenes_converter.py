# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from functools import lru_cache
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

from mmdet3d.core.bbox import points_cam2img
from mmdet3d.datasets import NuScenesDataset
from projects.RoadOcc.preprocess.occflow_utils import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_DYNAMIC_CLASS_NAMES,
    DEFAULT_EMPTY_IDX,
    compute_box_index_bounds,
    compute_voxel_grid_meta,
    evaluate_flow_consistency_against_future_occ,
    get_class_name_to_idx,
    get_dynamic_class_indices,
    mask_to_points_ego,
    quat_to_yaw,
    select_consistency_support_mask,
    transform_global_boxes_to_ego,
)


nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

RAW_CATEGORY_TO_OCC = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
}
OCC_CLASS_NAME_TO_IDX = get_class_name_to_idx(DEFAULT_CLASS_NAMES)
OCC_DYNAMIC_CLASS_INDICES = get_dynamic_class_indices(
    DEFAULT_CLASS_NAMES, DEFAULT_DYNAMIC_CLASS_NAMES)


def _infer_sample_timestamp_to_sec(nusc: NuScenes,
                                   fallback: float = 1e-6) -> float:
    """Infer the timestamp unit used by the dataset.

    nuScenes stores sample timestamps in microseconds, but the V2XReal-exported
    metadata in this repo uses a coarser unit. For V2X-Real, the official paper
    and project page describe the raw LiDAR streams as 10 Hz, but the released
    nuScenes-style ``sample`` timeline is effectively 2 Hz: consecutive
    ``sample.timestamp`` values differ by 5000 and should map to 0.5 seconds,
    not 0.05 seconds. We infer the scale from a few positive timestamp deltas to
    avoid hardcoding dataset names.
    """
    deltas = []
    for sample in nusc.sample:
        prev_token = sample.get('prev', '')
        if not prev_token:
            continue
        prev_sample = nusc.get('sample', prev_token)
        delta = float(sample['timestamp'] - prev_sample['timestamp'])
        if delta > 0:
            deltas.append(delta)
        if len(deltas) >= 128:
            break
    if not deltas:
        return float(fallback)
    median_delta = float(np.median(np.asarray(deltas, dtype=np.float64)))
    candidate_scales = np.asarray([1e-6, 1e-5, 1e-4], dtype=np.float64)
    # V2X-Real raw sensors run at 10 Hz, but the released synchronized sample
    # sequence is subsampled to 2 Hz. We therefore prefer the timestamp scale
    # whose median sample interval is closest to 0.5 seconds.
    target_keyframe_dt = 0.5
    candidate_dts = median_delta * candidate_scales
    best_index = int(np.argmin(np.abs(candidate_dts - target_keyframe_dt)))
    best_scale = float(candidate_scales[best_index])
    if candidate_dts[best_index] <= 0.0:
        return float(fallback)
    return best_scale


def _box_velocity_with_timestamp_scale(
        nusc: NuScenes,
        sample_annotation_token: str,
        sample_timestamp_to_sec: float,
        max_time_diff: float = 15.0,
        outlier_speed_thresh: float = 80.0,
        disagreement_ratio: float = 3.0,
        tiny_car_planar_max: float = 1.2,
        tiny_car_speed_thresh: float = 20.0,
        triplet_jump_speed_thresh: float = 80.0,
        triplet_bridge_speed_thresh: float = 60.0,
        triplet_jump_ratio_thresh: float = 2.5,
        triplet_mid_error_thresh: float = 3.0,
        isolated_match_center_thresh: float = 5.5,
        isolated_match_size_thresh: float = 2.0,
        isolated_match_yaw_thresh: float = 1.57,
        isolated_match_cost_margin: float = 0.5,
        isolated_family_mismatch_thresh: float = 0.55,
        isolated_family_fallback_thresh: float = 0.40,
        isolated_family_topk: int = 2,
        occ_speed_gate_thresh: float = 20.0,
        occ_min_support_voxels: int = 6,
        occ_score_margin: float = 0.10) -> np.ndarray:
    """Estimate annotation velocity with configurable timestamp scale.

    The V2XReal-exported metadata occasionally contains broken prev/next links.
    We therefore prefer one-sided velocities when the two neighbors disagree
    strongly, instead of always using the centered difference from official
    nuScenes.
    """
    @lru_cache(maxsize=None)
    def _sample_time_sec(sample_token: str) -> float:
        return float(nusc.get('sample', sample_token)['timestamp']) * float(
            sample_timestamp_to_sec)

    @lru_cache(maxsize=None)
    def _annotation_category(annotation_token: str) -> str:
        annotation = nusc.get('sample_annotation', annotation_token)
        return str(annotation['category_name'])

    @lru_cache(maxsize=None)
    def _sample_record(sample_token: str) -> dict:
        return nusc.get('sample', sample_token)

    @lru_cache(maxsize=None)
    def _sample_scene_name(sample_token: str) -> str:
        sample = _sample_record(sample_token)
        scene = nusc.get('scene', sample['scene_token'])
        return str(scene['name'])

    @lru_cache(maxsize=None)
    def _sample_occ_semantics(sample_token: str):
        scene_name = _sample_scene_name(sample_token)
        occ_path = osp.join(
            nusc.dataroot, 'gts', scene_name, sample_token, 'labels.npz')
        if not osp.exists(occ_path):
            return None
        with np.load(occ_path) as occ_data:
            semantics = occ_data.get('semantics')
            if semantics is None:
                return None
            return np.asarray(semantics, dtype=np.uint8)

    @lru_cache(maxsize=None)
    def _sample_grid_meta(sample_token: str):
        semantics = _sample_occ_semantics(sample_token)
        if semantics is None:
            return None
        return compute_voxel_grid_meta(semantics.shape)

    @lru_cache(maxsize=None)
    def _sample_ego_pose(sample_token: str):
        sample = _sample_record(sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        sd_record = nusc.get('sample_data', lidar_token)
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
        return pose_record['rotation'], pose_record['translation']

    family_to_categories = {
        'pedestrian': (
            'human.pedestrian.adult',
            'human.pedestrian.child',
            'human.pedestrian.construction_worker',
            'human.pedestrian.police_officer',
        ),
        'bicycle': ('vehicle.bicycle',),
        'motorcycle': ('vehicle.motorcycle',),
        'car': ('vehicle.car',),
        'truck': ('vehicle.truck',),
        'trailer': ('vehicle.trailer',),
        'bus': ('vehicle.bus.bendy', 'vehicle.bus.rigid'),
        'construction': ('vehicle.construction',),
    }
    family_prototypes = {
        'pedestrian': np.array([0.75, 0.65, 1.75], dtype=np.float32),
        'bicycle': np.array([1.55, 0.97, 1.69], dtype=np.float32),
        'motorcycle': np.array([0.92, 0.73, 1.73], dtype=np.float32),
        'car': np.array([4.70, 2.10, 1.70], dtype=np.float32),
        'truck': np.array([5.90, 2.20, 2.05], dtype=np.float32),
        'trailer': np.array([12.0, 2.8, 3.5], dtype=np.float32),
        'bus': np.array([12.65, 3.10, 3.44], dtype=np.float32),
        'construction': np.array([6.0, 2.5, 3.0], dtype=np.float32),
    }
    category_to_family = {
        category: family
        for family, categories in family_to_categories.items()
        for category in categories
    }

    def _annotation_yaw(annotation_record: dict) -> float:
        return float(Quaternion(annotation_record['rotation']).yaw_pitch_roll[0])

    def _normalize_yaw(yaw: float) -> float:
        return float((yaw + np.pi) % (2.0 * np.pi) - np.pi)

    def _relative_size_distance(size: np.ndarray, prototype: np.ndarray) -> float:
        return float(np.linalg.norm((size - prototype) / np.maximum(prototype, 1.0)))

    def _candidate_categories_for_isolated_match(
            target_record: dict) -> Tuple[str, ...]:
        target_category = target_record['category_name']
        target_family = category_to_family.get(target_category, None)
        if target_family is None:
            return (target_category,)
        target_size = np.asarray(target_record['size'], dtype=np.float32)
        primary_distance = _relative_size_distance(
            target_size, family_prototypes[target_family])
        if primary_distance <= float(isolated_family_mismatch_thresh):
            return (target_category,)

        ranked_families = sorted(
            family_prototypes.keys(),
            key=lambda family: _relative_size_distance(
                target_size, family_prototypes[family]))
        candidate_categories = [target_category]
        for family in ranked_families:
            if family == target_family:
                continue
            distance = _relative_size_distance(
                target_size, family_prototypes[family])
            if distance > float(isolated_family_fallback_thresh):
                continue
            candidate_categories.extend(family_to_categories[family])
            if len(candidate_categories) >= 1 + int(isolated_family_topk):
                break
        return tuple(dict.fromkeys(candidate_categories))

    def _match_isolated_neighbor(
            target_record: dict,
            neighbor_sample_token: str) -> dict:
        if not neighbor_sample_token:
            return None
        neighbor_sample = nusc.get('sample', neighbor_sample_token)
        target_center = np.asarray(target_record['translation'], dtype=np.float32)
        target_size = np.asarray(target_record['size'], dtype=np.float32)
        target_yaw = _annotation_yaw(target_record)
        allowed_categories = _candidate_categories_for_isolated_match(
            target_record)

        def _select_best_candidate(category_names: Tuple[str, ...]) -> dict:
            candidates = []
            for ann_token in neighbor_sample['anns']:
                ann_record = nusc.get('sample_annotation', ann_token)
                if ann_record['category_name'] not in category_names:
                    continue
                center_dist = float(
                    np.linalg.norm(
                        np.asarray(ann_record['translation'], dtype=np.float32) -
                        target_center))
                if center_dist > float(isolated_match_center_thresh):
                    continue
                size_dist = float(
                    np.linalg.norm(
                        np.asarray(ann_record['size'], dtype=np.float32) -
                        target_size))
                if size_dist > float(isolated_match_size_thresh):
                    continue
                yaw_dist = abs(
                    _normalize_yaw(_annotation_yaw(ann_record) - target_yaw))
                if yaw_dist > float(isolated_match_yaw_thresh):
                    continue
                cost = center_dist + 0.5 * size_dist + 0.3 * yaw_dist
                candidates.append((cost, ann_record))
            if not candidates:
                return None
            candidates.sort(key=lambda item: item[0])
            if (len(candidates) > 1 and
                    (candidates[1][0] - candidates[0][0]) <
                    float(isolated_match_cost_margin)):
                return None
            return candidates[0][1]

        primary_match = _select_best_candidate((target_record['category_name'],))
        if primary_match is not None:
            return primary_match
        if len(allowed_categories) <= 1:
            return None
        return _select_best_candidate(allowed_categories)

    @lru_cache(maxsize=None)
    def _annotation_occ_support(annotation_token: str):
        annotation_record = nusc.get('sample_annotation', annotation_token)
        occ_class_name = RAW_CATEGORY_TO_OCC.get(
            annotation_record['category_name'], 'ignore')
        if occ_class_name not in DEFAULT_DYNAMIC_CLASS_NAMES:
            return None
        semantics = _sample_occ_semantics(annotation_record['sample_token'])
        grid_meta = _sample_grid_meta(annotation_record['sample_token'])
        if semantics is None or grid_meta is None:
            return None

        pose_rotation, pose_translation = _sample_ego_pose(
            annotation_record['sample_token'])
        translation = np.asarray(
            annotation_record['translation'], dtype=np.float32)
        size = np.asarray(annotation_record['size'], dtype=np.float32)
        rotation = np.asarray(annotation_record['rotation'], dtype=np.float32)
        box_global = np.array([[
            translation[0],
            translation[1],
            translation[2],
            size[1],
            size[0],
            size[2],
            quat_to_yaw(rotation),
        ]], dtype=np.float32)
        box_ego = transform_global_boxes_to_ego(
            box_global,
            ego2global_rotation=pose_rotation,
            ego2global_translation=pose_translation,
        )
        bounds = compute_box_index_bounds(box_ego[0], grid_meta)
        if bounds is None:
            return None
        x0, x1, y0, y1, z0, z1 = bounds
        center_x, center_y, center_z, size_x, size_y, size_z, yaw = [
            float(value) for value in box_ego[0][:7]
        ]
        sub_semantics = semantics[x0:x1, y0:y1, z0:z1]
        xs = grid_meta.x_centers[x0:x1][:, None, None] - center_x
        ys = grid_meta.y_centers[y0:y1][None, :, None] - center_y
        zs = grid_meta.z_centers[z0:z1][None, None, :] - center_z
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        local_x = cos_yaw * xs + sin_yaw * ys
        local_y = -sin_yaw * xs + cos_yaw * ys
        half_z = max(size_z * 0.5, 1e-3)
        inside_mask = (
            (np.abs(local_x) <= size_x * 0.5 + 1e-4) &
            (np.abs(local_y) <= size_y * 0.5 + 1e-4) &
            (np.abs(zs) <= half_z + 1e-4))
        if not inside_mask.any():
            return None
        class_idx = OCC_CLASS_NAME_TO_IDX[occ_class_name]
        support_mask, support_name = select_consistency_support_mask(
            sub_semantics,
            inside_mask=inside_mask,
            class_idx=class_idx,
            dynamic_class_indices=OCC_DYNAMIC_CLASS_INDICES,
            empty_idx=DEFAULT_EMPTY_IDX,
            min_voxels=occ_min_support_voxels,
        )
        support_points = mask_to_points_ego(
            grid_meta.x_centers[x0:x1],
            grid_meta.y_centers[y0:y1],
            grid_meta.z_centers[z0:z1],
            support_mask,
        )
        if support_points.shape[0] <= 0:
            return None
        return dict(
            sample_token=annotation_record['sample_token'],
            class_idx=class_idx,
            support_name=support_name,
            support_points=support_points,
            pose_rotation=pose_rotation,
            pose_translation=pose_translation,
            grid_meta=grid_meta,
        )

    def _occ_reject_high_speed_candidate(
            annotation_token: str,
            velocity_global: np.ndarray,
            neighbor_key: str) -> bool:
        speed = float(np.linalg.norm(velocity_global[:2]))
        if speed <= float(occ_speed_gate_thresh):
            return False
        current_support = _annotation_occ_support(annotation_token)
        if current_support is None:
            return False
        annotation_record = nusc.get('sample_annotation', annotation_token)
        neighbor_token = annotation_record.get(neighbor_key, '')
        if not neighbor_token:
            return False
        neighbor_record = nusc.get('sample_annotation', neighbor_token)
        future_semantics = _sample_occ_semantics(neighbor_record['sample_token'])
        future_grid_meta = _sample_grid_meta(neighbor_record['sample_token'])
        future_pose = _sample_ego_pose(neighbor_record['sample_token'])
        if future_semantics is None or future_grid_meta is None:
            return False

        current_time = _sample_time_sec(annotation_record['sample_token'])
        neighbor_time = _sample_time_sec(neighbor_record['sample_token'])
        horizon_sec = abs(neighbor_time - current_time)
        if horizon_sec <= 0.0 or horizon_sec > max_time_diff:
            return False

        current_rotation = Quaternion(
            current_support['pose_rotation']).rotation_matrix
        flow_ego = current_rotation.T @ np.asarray(
            velocity_global, dtype=np.float32).reshape(3)
        flow_xy = flow_ego[:2].astype(np.float32)
        if neighbor_key == 'prev':
            flow_xy = -flow_xy

        moving_consistency = evaluate_flow_consistency_against_future_occ(
            support_points_ego=current_support['support_points'],
            flow_xy=flow_xy,
            horizon_sec=float(horizon_sec),
            current_ego2global_rotation=current_support['pose_rotation'],
            current_ego2global_translation=current_support['pose_translation'],
            future_ego2global_rotation=future_pose[0],
            future_ego2global_translation=future_pose[1],
            future_semantics=future_semantics,
            grid_meta=future_grid_meta,
            class_idx=current_support['class_idx'],
            dynamic_class_indices=OCC_DYNAMIC_CLASS_INDICES,
            empty_idx=DEFAULT_EMPTY_IDX)
        zero_consistency = evaluate_flow_consistency_against_future_occ(
            support_points_ego=current_support['support_points'],
            flow_xy=np.zeros(2, dtype=np.float32),
            horizon_sec=float(horizon_sec),
            current_ego2global_rotation=current_support['pose_rotation'],
            current_ego2global_translation=current_support['pose_translation'],
            future_ego2global_rotation=future_pose[0],
            future_ego2global_translation=future_pose[1],
            future_semantics=future_semantics,
            grid_meta=future_grid_meta,
            class_idx=current_support['class_idx'],
            dynamic_class_indices=OCC_DYNAMIC_CLASS_INDICES,
            empty_idx=DEFAULT_EMPTY_IDX)

        if moving_consistency.valid_count <= 0 or zero_consistency.valid_count <= 0:
            return False
        zero_has_evidence = (
            zero_consistency.same_class_ratio >= 0.05 or
            zero_consistency.dynamic_ratio >= 0.10 or
            zero_consistency.occupied_ratio >= 0.30)
        return bool(
            zero_has_evidence and
            zero_consistency.score >= (
                moving_consistency.score + float(occ_score_margin)) and
            moving_consistency.same_class_ratio < 0.20 and
            moving_consistency.dynamic_ratio < 0.35)

    current = nusc.get('sample_annotation', sample_annotation_token)
    has_prev = current['prev'] != ''
    has_next = current['next'] != ''
    planar_size = np.asarray(current['size'][:2], dtype=np.float32)
    is_tiny_car = (
        _annotation_category(sample_annotation_token) == 'vehicle.car' and
        planar_size.size >= 2 and
        float(np.max(planar_size)) < float(tiny_car_planar_max)
    )

    current_translation = np.asarray(current['translation'], dtype=np.float32)
    current_time = _sample_time_sec(current['sample_token'])
    if not has_prev and not has_next:
        current_sample = nusc.get('sample', current['sample_token'])
        matched_neighbor_velocities = []
        for neighbor_key in ('prev', 'next'):
            neighbor_sample_token = current_sample.get(neighbor_key, '')
            matched_neighbor = _match_isolated_neighbor(
                current, neighbor_sample_token)
            if matched_neighbor is None:
                continue
            neighbor_time = _sample_time_sec(matched_neighbor['sample_token'])
            time_diff = abs(neighbor_time - current_time)
            if time_diff <= 0.0 or time_diff > max_time_diff:
                continue
            neighbor_translation = np.asarray(
                matched_neighbor['translation'], dtype=np.float32)
            if neighbor_key == 'next':
                pos_diff = neighbor_translation - current_translation
            else:
                pos_diff = current_translation - neighbor_translation
            matched_neighbor_velocities.append(pos_diff / time_diff)
        if not matched_neighbor_velocities:
            return np.zeros(3, dtype=np.float32)
        if len(matched_neighbor_velocities) == 1:
            return matched_neighbor_velocities[0].astype(np.float32)
        return np.mean(
            np.stack(matched_neighbor_velocities, axis=0),
            axis=0).astype(np.float32)

    if has_prev and has_next:
        if is_tiny_car:
            prev_record = nusc.get('sample_annotation', current['prev'])
            next_record = nusc.get('sample_annotation', current['next'])
            prev_time = _sample_time_sec(prev_record['sample_token'])
            next_time = _sample_time_sec(next_record['sample_token'])
            dt_prev = current_time - prev_time
            dt_next = next_time - current_time
            if dt_prev > 0.0 and dt_next > 0.0:
                prev_xy = np.asarray(prev_record['translation'][:2], dtype=np.float32)
                current_xy = np.asarray(current['translation'][:2], dtype=np.float32)
                next_xy = np.asarray(next_record['translation'][:2], dtype=np.float32)
                disp_prev_cur = float(np.linalg.norm(current_xy - prev_xy))
                disp_cur_next = float(np.linalg.norm(next_xy - current_xy))
                disp_prev_next = float(np.linalg.norm(next_xy - prev_xy))
                v_prev_cur = disp_prev_cur / dt_prev
                v_cur_next = disp_cur_next / dt_next
                v_prev_next = disp_prev_next / (dt_prev + dt_next)
                alpha = dt_prev / (dt_prev + dt_next)
                interp_xy = prev_xy + alpha * (next_xy - prev_xy)
                mid_err = float(np.linalg.norm(current_xy - interp_xy))
                jump_ratio = min(v_prev_cur, v_cur_next) / max(v_prev_next, 1.0)
                suspicious_triplet = (
                    (
                        v_prev_cur > float(triplet_jump_speed_thresh) and
                        v_cur_next > float(triplet_jump_speed_thresh) and
                        (
                            v_prev_next < float(triplet_bridge_speed_thresh) or
                            jump_ratio > float(triplet_jump_ratio_thresh)
                        )
                    ) or (
                        v_prev_cur > (1.5 * float(triplet_jump_speed_thresh)) and
                        v_cur_next > (1.5 * float(triplet_jump_speed_thresh)) and
                        mid_err > float(triplet_mid_error_thresh)
                    )
                )
                if suspicious_triplet:
                    return np.zeros(3, dtype=np.float32)

    neighbor_velocities = []
    prefer_zero = False
    for neighbor_key in ('prev', 'next'):
        neighbor_token = current[neighbor_key]
        if not neighbor_token:
            continue
        neighbor = nusc.get('sample_annotation', neighbor_token)
        neighbor_time = _sample_time_sec(neighbor['sample_token'])
        time_diff = abs(neighbor_time - current_time)
        if time_diff <= 0.0 or time_diff > max_time_diff:
            continue
        neighbor_translation = np.asarray(
            neighbor['translation'], dtype=np.float32)
        if neighbor_key == 'next':
            pos_diff = neighbor_translation - current_translation
        else:
            pos_diff = current_translation - neighbor_translation
        velocity_candidate = pos_diff / time_diff
        if (
                is_tiny_car and
                float(np.linalg.norm(velocity_candidate[:2])) >
                float(tiny_car_speed_thresh)):
            prefer_zero = True
            continue
        if _occ_reject_high_speed_candidate(
                sample_annotation_token,
                velocity_candidate,
                neighbor_key):
            prefer_zero = True
            continue
        neighbor_velocities.append(velocity_candidate)

    if not neighbor_velocities:
        if prefer_zero:
            return np.zeros(3, dtype=np.float32)
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    if len(neighbor_velocities) == 1:
        return neighbor_velocities[0].astype(np.float32)

    speeds = np.asarray([
        np.linalg.norm(velocity[:2]) for velocity in neighbor_velocities
    ], dtype=np.float32)
    reasonable_mask = speeds <= float(outlier_speed_thresh)
    if reasonable_mask.sum() == 1:
        return neighbor_velocities[int(np.flatnonzero(reasonable_mask)[0])].astype(
            np.float32)
    if reasonable_mask.sum() == 0:
        return neighbor_velocities[int(np.argmin(speeds))].astype(np.float32)

    min_speed = float(np.minimum.reduce(speeds))
    max_speed = float(np.maximum.reduce(speeds))
    if max_speed / max(min_speed, 1.0) > float(disagreement_ratio):
        return neighbor_velocities[int(np.argmin(speeds))].astype(np.float32)
    return np.mean(
        np.stack(neighbor_velocities, axis=0), axis=0).astype(np.float32)


def create_nuscenes_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          train_half=False,
                          can_bus_path=None,
                          max_sweeps=10,
                          only_split='all'):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    sample_timestamp_to_sec = _infer_sample_timestamp_to_sec(nusc)
    print(f'Using sample_timestamp_to_sec={sample_timestamp_to_sec:.1e} for '
          f'{info_prefix}')
    # Check if can_bus directory exists before creating NuScenesCanBus
    nusc_can_bus = None
    if can_bus_path is not None:
        can_bus_full_path = osp.join(can_bus_path, 'can_bus')
        if osp.exists(can_bus_full_path):
            nusc_can_bus = NuScenesCanBus(dataroot=can_bus_path)
        else:
            print(f"Warning: CAN bus directory not found at {can_bus_full_path}. Skipping CAN bus data.")
    else:
        print("Warning: can_bus_path is None. Skipping CAN bus data.")
    from data_converter.splits import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits["train"]
        val_scenes = splits["val"]
    elif version == 'v1.0-test':
        train_scenes = splits["test"]
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits["mini_train"]
        val_scenes = splits["mini_val"]
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])
    if train_half:
        train_scenes = list(train_scenes)
        train_scenes = train_scenes[:len(train_scenes)//2]
        train_scenes = set(train_scenes)

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc,
        train_scenes=train_scenes,
        val_scenes=val_scenes,
        test=test,
        nusc_can_bus=nusc_can_bus,
        max_sweeps=max_sweeps,
        sample_timestamp_to_sec=sample_timestamp_to_sec,
        only_split=only_split)

    metadata = dict(
        version=version,
        sample_timestamp_to_sec=sample_timestamp_to_sec)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        if only_split in ('all', 'train'):
            if train_half:
                info_path = osp.join(
                    root_path, '{}_infos_half_train.pkl'.format(info_prefix))
            else:
                info_path = osp.join(
                    root_path, '{}_infos_train.pkl'.format(info_prefix))
            mmcv.dump(data, info_path)
        if only_split in ('all', 'val') and not train_half:
            data['infos'] = val_nusc_infos
            info_val_path = osp.join(root_path,
                                     '{}_infos_val.pkl'.format(info_prefix))
            mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes

def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    # If nusc_can_bus is None, return zeros
    if nusc_can_bus is None:
        return np.zeros(18)
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except Exception:
        return np.zeros(18)  # server scenes do not have can bus information.
    if not pose_list:
        return np.zeros(18)
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for pose in pose_list:
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose

    # Work on a copy so repeated calls do not mutate the cached CAN bus data.
    pose_record = dict(last_pose)
    pose_record.pop('utime', None)  # unused
    pos = pose_record.pop('pos')
    rotation = pose_record.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for value in pose_record.values():
        if isinstance(value, (list, tuple, np.ndarray)):
            can_bus.extend(value)
        else:
            can_bus.append(value)
    can_bus.extend([0., 0.])
    return np.array(can_bus)

def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         nusc_can_bus=None,
                         test=False,
                         max_sweeps=10,
                         sample_timestamp_to_sec=1e-6,
                         only_split='all'):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []

    for sample in mmcv.track_iter_progress(nusc.sample):
        in_train_split = sample['scene_token'] in train_scenes
        if only_split == 'train' and not in_train_split:
            continue
        if only_split == 'val' and in_train_split:
            continue
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)
        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'can_bus': can_bus,
            'sweeps': [],
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain image information per frame, handle missing cameras gracefully
        camera_types = [
            'CAM_FRONT',
            # 'CAM_FRONT_RIGHT',
            # 'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_LEFT',
            'CAM_RIGHT',
        ]
        for cam in camera_types:
            if cam not in sample['data']:
                print(f"Warning: Camera {cam} not found in sample data. Skipping.")
                continue
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array([
                _box_velocity_with_timestamp_scale(
                    nusc, token, sample_timestamp_to_sec)[:2]
                for token in sample['anns']
            ])
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # we need to convert box size to
            # the format of our lidar coordinate system
            # which is x_size, y_size, z_size (corresponding to l, w, h)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag

            if 'lidarseg' in nusc.table_names:
                info['pts_semantic_mask_path'] = osp.join(
                    nusc.dataroot,
                    nusc.get('lidarseg', lidar_token)['filename'])

        if in_train_split:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str, optional): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool, optional): Whether to export mono3d annotation.
            Default: True.
    """
    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    nusc_infos = mmcv.load(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # info_2d_list = []
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d)
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/nuscenes/')
                    [-1],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = _box_velocity_with_timestamp_scale(
                nusc, box.token, sample_timestamp_to_sec)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): file name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NuScenesDataset.NameMapping:
        return None
    cat_name = NuScenesDataset.NameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec
