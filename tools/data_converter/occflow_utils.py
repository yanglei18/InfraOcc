#!/usr/bin/env python3
from __future__ import annotations

import colorsys
import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_TRAIN_PKL = REPO_ROOT / 'data' / 'infraocc' / 'infraocc_infos_train.pkl'
DEFAULT_VAL_PKL = REPO_ROOT / 'data' / 'infraocc' / 'infraocc_infos_val.pkl'
DEFAULT_SOURCE_OCC_ROOT = REPO_ROOT / 'data' / 'infraocc' / 'gts'
DEFAULT_OUTPUT_OCC_ROOT = REPO_ROOT / 'data' / 'infraocc' / 'occflow_gts'
DEFAULT_METADATA_ROOT = REPO_ROOT / 'data' / 'infraocc' / 'v1.0-trainval'
# InfraOcc raw sensors are captured at 10 Hz, but the released nuScenes-style
# sample timeline is effectively 2 Hz. Consecutive sample timestamps differ by
# 5000 and should therefore map to 0.5 seconds.
DEFAULT_SAMPLE_TIMESTAMP_TO_SEC = 1e-4
DEFAULT_POINT_CLOUD_RANGE = np.array(
    [-64.0, -64.0, -4.8, 64.0, 64.0, 1.6], dtype=np.float32)
DEFAULT_CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free'
]
DEFAULT_DYNAMIC_CLASS_NAMES = [
    'bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'truck'
]
DEFAULT_EMPTY_IDX = DEFAULT_CLASS_NAMES.index('free')


@dataclass(frozen=True)
class VoxelGridMeta:
    point_cloud_range: np.ndarray
    voxel_size: np.ndarray
    grid_shape: Tuple[int, int, int]
    x_centers: np.ndarray
    y_centers: np.ndarray
    z_centers: np.ndarray


@dataclass(frozen=True)
class TemporalAnnotationIndex:
    sample_to_annotations: Mapping[str, Tuple[dict, ...]]
    annotation_by_token: Mapping[str, dict]
    sample_timestamp_by_token: Mapping[str, int]
    sample_next_by_token: Mapping[str, str]
    instance_to_class_name: Mapping[str, str]


@dataclass(frozen=True)
class MotionConsistency:
    support_count: int
    valid_count: int
    same_class_ratio: float
    dynamic_ratio: float
    occupied_ratio: float
    valid_ratio: float
    score: float


@dataclass(frozen=True)
class ConnectivityFlowRepair:
    repaired_flow: np.ndarray
    repaired_support_mask: np.ndarray
    strict_support_voxels: int
    repaired_support_voxels: int
    repaired_components: int


def load_infos(pkl_path: str | Path) -> List[dict]:
    with Path(pkl_path).open('rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        if 'infos' in obj:
            return list(obj['infos'])
        if 'data_list' in obj:
            return list(obj['data_list'])
    if isinstance(obj, list):
        return list(obj)
    raise ValueError(f'Unsupported pkl format: {type(obj)}')


@lru_cache(maxsize=2)
def load_temporal_annotation_index(metadata_root: str | Path = DEFAULT_METADATA_ROOT) -> TemporalAnnotationIndex:
    metadata_root = resolve_path(metadata_root, base_dir=REPO_ROOT).resolve()
    with (metadata_root / 'sample.json').open('r', encoding='utf-8') as f:
        sample_records = json.load(f)
    with (metadata_root / 'sample_annotation.json').open('r', encoding='utf-8') as f:
        annotation_records = json.load(f)
    with (metadata_root / 'instance.json').open('r', encoding='utf-8') as f:
        instance_records = json.load(f)
    with (metadata_root / 'category.json').open('r', encoding='utf-8') as f:
        category_records = json.load(f)

    category_token_to_name = {
        record['token']: record['name'] for record in category_records
    }
    raw_to_occ = {
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
    instance_to_class_name = {}
    for record in instance_records:
        category_name = category_token_to_name.get(record['category_token'], 'ignore')
        instance_to_class_name[record['token']] = raw_to_occ.get(category_name, 'ignore')

    sample_timestamp_by_token = {
        record['token']: int(record['timestamp']) for record in sample_records
    }
    sample_next_by_token = {
        record['token']: normalize_class_name(record.get('next', ''))
        for record in sample_records
    }
    annotation_by_token = {
        record['token']: record for record in annotation_records
    }
    sample_to_annotations: Dict[str, List[dict]] = {}
    for record in annotation_records:
        sample_to_annotations.setdefault(record['sample_token'], []).append(record)

    return TemporalAnnotationIndex(
        sample_to_annotations={
            sample_token: tuple(records)
            for sample_token, records in sample_to_annotations.items()
        },
        annotation_by_token=annotation_by_token,
        sample_timestamp_by_token=sample_timestamp_by_token,
        sample_next_by_token=sample_next_by_token,
        instance_to_class_name=instance_to_class_name,
    )


def resolve_path(path: str | Path, base_dir: str | Path = REPO_ROOT) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    path_str = str(path)
    if path_str.startswith('./'):
        path = Path(path_str[2:])
    return Path(base_dir) / path


def resolve_occ_path(occ_path: str | Path) -> Path:
    return resolve_path(occ_path, base_dir=REPO_ROOT)


def build_output_occ_path(
        occ_path: str | Path,
        output_root: str | Path = DEFAULT_OUTPUT_OCC_ROOT,
        source_root: str | Path = DEFAULT_SOURCE_OCC_ROOT) -> Path:
    occ_path = resolve_occ_path(occ_path).resolve()
    source_root = resolve_path(source_root, base_dir=REPO_ROOT).resolve()
    try:
        rel_path = occ_path.relative_to(source_root)
    except ValueError:
        path_parts = occ_path.parts
        if 'gts' not in path_parts:
            raise
        rel_path = Path(*path_parts[path_parts.index('gts') + 1:])
    return resolve_path(output_root, base_dir=REPO_ROOT) / rel_path


def normalize_class_name(name: object) -> str:
    if isinstance(name, bytes):
        return name.decode('utf-8')
    return str(name)


def get_class_name_to_idx(class_names: Sequence[str]) -> Dict[str, int]:
    return {normalize_class_name(name): index for index, name in enumerate(class_names)}


def get_dynamic_class_indices(
        class_names: Sequence[str],
        dynamic_class_names: Sequence[str]) -> Tuple[int, ...]:
    class_name_to_idx = get_class_name_to_idx(class_names)
    return tuple(class_name_to_idx[name] for name in dynamic_class_names)


def compute_voxel_grid_meta(
        semantics_shape: Sequence[int],
        point_cloud_range: Sequence[float] = DEFAULT_POINT_CLOUD_RANGE) -> VoxelGridMeta:
    point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
    grid_shape = tuple(int(dim) for dim in semantics_shape)
    voxel_size = (point_cloud_range[3:] - point_cloud_range[:3]) / np.asarray(
        grid_shape, dtype=np.float32)
    x_centers = point_cloud_range[0] + (np.arange(grid_shape[0], dtype=np.float32) + 0.5) * voxel_size[0]
    y_centers = point_cloud_range[1] + (np.arange(grid_shape[1], dtype=np.float32) + 0.5) * voxel_size[1]
    z_centers = point_cloud_range[2] + (np.arange(grid_shape[2], dtype=np.float32) + 0.5) * voxel_size[2]
    return VoxelGridMeta(
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size.astype(np.float32),
        grid_shape=grid_shape,
        x_centers=x_centers,
        y_centers=y_centers,
        z_centers=z_centers)


def sanitize_velocity(velocity_xy: Sequence[float], max_speed: float) -> np.ndarray:
    velocity_xy = np.asarray(velocity_xy, dtype=np.float32).reshape(2)
    if not np.isfinite(velocity_xy).all():
        return np.zeros(2, dtype=np.float32)
    speed = float(np.linalg.norm(velocity_xy))
    if speed <= 0.0:
        return velocity_xy
    if speed > max_speed:
        velocity_xy = velocity_xy * (max_speed / speed)
    return velocity_xy.astype(np.float32)


def quat_to_rot_matrix(quaternion: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32).reshape(4)
    w, x, y, z = quaternion
    norm_sq = float(np.dot(quaternion, quaternion))
    if norm_sq < 1e-8:
        return np.eye(3, dtype=np.float32)
    scale = 2.0 / norm_sq
    wx, wy, wz = scale * w * x, scale * w * y, scale * w * z
    xx, xy, xz = scale * x * x, scale * x * y, scale * x * z
    yy, yz, zz = scale * y * y, scale * y * z, scale * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=np.float32)


def quat_to_yaw(quaternion: Sequence[float]) -> float:
    w, x, y, z = [float(value) for value in quaternion]
    return float(np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)))


def normalize_yaw(yaw: float) -> float:
    return float(np.arctan2(np.sin(yaw), np.cos(yaw)))


def transform_boxes_velocity_lidar_to_ego(
        gt_boxes: np.ndarray,
        gt_velocity: np.ndarray,
        lidar2ego_rotation: Sequence[float],
        lidar2ego_translation: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).copy()
    gt_velocity = np.asarray(gt_velocity, dtype=np.float32).copy()
    if gt_boxes.size == 0:
        return gt_boxes, gt_velocity
    rotation = quat_to_rot_matrix(lidar2ego_rotation)
    translation = np.asarray(lidar2ego_translation, dtype=np.float32).reshape(3)
    yaw_offset = quat_to_yaw(lidar2ego_rotation)
    gt_boxes[:, :3] = (rotation @ gt_boxes[:, :3].T).T + translation[None, :]
    gt_boxes[:, 6] = gt_boxes[:, 6] + yaw_offset
    velocity_3d = np.concatenate(
        [gt_velocity, np.zeros((gt_velocity.shape[0], 1), dtype=np.float32)],
        axis=1)
    velocity_ego = (rotation @ velocity_3d.T).T
    gt_velocity = velocity_ego[:, :2]
    return gt_boxes, gt_velocity


def transform_global_boxes_to_ego(
        gt_boxes_global: np.ndarray,
        ego2global_rotation: Sequence[float],
        ego2global_translation: Sequence[float]) -> np.ndarray:
    gt_boxes_global = np.asarray(gt_boxes_global, dtype=np.float32).copy()
    if gt_boxes_global.size == 0:
        return gt_boxes_global
    global_to_ego_rotation = quat_to_rot_matrix(ego2global_rotation).T
    global_to_ego_translation = -global_to_ego_rotation @ np.asarray(
        ego2global_translation, dtype=np.float32).reshape(3)
    yaw_offset = quat_to_yaw(ego2global_rotation)
    gt_boxes_global[:, :3] = (
        global_to_ego_rotation @ gt_boxes_global[:, :3].T).T + global_to_ego_translation[None, :]
    gt_boxes_global[:, 6] = gt_boxes_global[:, 6] - yaw_offset
    return gt_boxes_global


def transform_points_ego_to_global(
        points_ego: np.ndarray,
        ego2global_rotation: Sequence[float],
        ego2global_translation: Sequence[float]) -> np.ndarray:
    points_ego = np.asarray(points_ego, dtype=np.float32)
    if points_ego.size == 0:
        return points_ego.reshape(-1, 3)
    rotation = quat_to_rot_matrix(ego2global_rotation)
    translation = np.asarray(ego2global_translation, dtype=np.float32).reshape(3)
    return (rotation @ points_ego.T).T + translation[None, :]


def transform_points_global_to_ego(
        points_global: np.ndarray,
        ego2global_rotation: Sequence[float],
        ego2global_translation: Sequence[float]) -> np.ndarray:
    points_global = np.asarray(points_global, dtype=np.float32)
    if points_global.size == 0:
        return points_global.reshape(-1, 3)
    rotation_t = quat_to_rot_matrix(ego2global_rotation).T
    translation = np.asarray(ego2global_translation, dtype=np.float32).reshape(3)
    return (rotation_t @ (points_global - translation[None, :]).T).T


def follow_future_sample_token(
        sample_token: str,
        sample_next_by_token: Mapping[str, str],
        steps: int) -> str:
    next_token = normalize_class_name(sample_token)
    for _ in range(max(int(steps), 0)):
        if not next_token:
            return ''
        next_token = normalize_class_name(sample_next_by_token.get(next_token, ''))
    return next_token


def build_annotation_boxes_global(
        annotation_records: Sequence[Mapping[str, object]],
        instance_to_class_name: Mapping[str, str],
        dynamic_class_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    dynamic_class_names = set(normalize_class_name(name) for name in dynamic_class_names)
    boxes_global = []
    class_names = []
    for record in annotation_records:
        class_name = instance_to_class_name.get(record['instance_token'], 'ignore')
        if class_name not in dynamic_class_names:
            continue
        translation = np.asarray(record['translation'], dtype=np.float32)
        size = np.asarray(record['size'], dtype=np.float32)
        rotation = np.asarray(record['rotation'], dtype=np.float32)
        boxes_global.append(np.array([
            translation[0],
            translation[1],
            translation[2],
            size[1],
            size[0],
            size[2],
            quat_to_yaw(rotation),
        ], dtype=np.float32))
        class_names.append(class_name)
    if not boxes_global:
        return np.zeros((0, 7), dtype=np.float32), []
    return np.stack(boxes_global, axis=0), class_names


def match_gt_boxes_to_dynamic_annotations(
        gt_boxes_ego: np.ndarray,
        gt_names: Sequence[object],
        ann_boxes_ego: np.ndarray,
        ann_class_names: Sequence[str],
        center_weight: float = 1.0,
        size_weight: float = 0.5,
        yaw_weight: float = 0.3,
        max_center_distance: float = 3.0) -> Dict[int, int]:
    assignments: Dict[int, int] = {}
    if gt_boxes_ego.size == 0 or ann_boxes_ego.size == 0:
        return assignments
    gt_names = [normalize_class_name(name) for name in gt_names]
    remaining_ann_indices = set(range(len(ann_boxes_ego)))
    for gt_index, (gt_box, gt_name) in enumerate(zip(gt_boxes_ego, gt_names)):
        candidate_indices = [
            ann_index for ann_index in remaining_ann_indices
            if ann_class_names[ann_index] == gt_name
        ]
        if not candidate_indices:
            continue
        best_index = None
        best_cost = None
        for ann_index in candidate_indices:
            ann_box = ann_boxes_ego[ann_index]
            center_dist = float(np.linalg.norm(gt_box[:3] - ann_box[:3]))
            if center_dist > max_center_distance:
                continue
            size_dist = float(np.linalg.norm(gt_box[3:6] - ann_box[3:6]))
            yaw_dist = abs(normalize_yaw(float(gt_box[6] - ann_box[6])))
            cost = (
                center_weight * center_dist +
                size_weight * size_dist +
                yaw_weight * yaw_dist)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_index = ann_index
        if best_index is None:
            continue
        assignments[gt_index] = best_index
        remaining_ann_indices.remove(best_index)
    return assignments


def estimate_annotation_temporal_speed(
        annotation_record: Mapping[str, object],
        annotation_by_token: Mapping[str, Mapping[str, object]],
        sample_timestamp_by_token: Mapping[str, int],
        timestamp_to_sec: float = DEFAULT_SAMPLE_TIMESTAMP_TO_SEC) -> Optional[float]:
    current_translation = np.asarray(annotation_record['translation'], dtype=np.float32)
    current_sample_token = annotation_record['sample_token']
    current_timestamp = sample_timestamp_by_token.get(current_sample_token)
    if current_timestamp is None:
        return None
    speeds = []
    for neighbor_key in ('prev', 'next'):
        neighbor_token = annotation_record.get(neighbor_key, '')
        if not neighbor_token:
            continue
        neighbor_record = annotation_by_token.get(neighbor_token)
        if neighbor_record is None:
            continue
        neighbor_sample_token = neighbor_record['sample_token']
        neighbor_timestamp = sample_timestamp_by_token.get(neighbor_sample_token)
        if neighbor_timestamp is None or neighbor_timestamp == current_timestamp:
            continue
        neighbor_translation = np.asarray(neighbor_record['translation'], dtype=np.float32)
        dt = abs(float(neighbor_timestamp - current_timestamp)) * float(timestamp_to_sec)
        if dt <= 1e-6:
            continue
        displacement = np.linalg.norm(neighbor_translation[:2] - current_translation[:2])
        speeds.append(float(displacement / dt))
    if not speeds:
        return None
    return max(speeds)


def estimate_annotation_temporal_velocity_ego(
        annotation_record: Mapping[str, object],
        annotation_by_token: Mapping[str, Mapping[str, object]],
        sample_timestamp_by_token: Mapping[str, int],
        ego2global_rotation: Sequence[float],
        timestamp_to_sec: float = DEFAULT_SAMPLE_TIMESTAMP_TO_SEC) -> Optional[np.ndarray]:
    current_translation = np.asarray(annotation_record['translation'], dtype=np.float32)
    current_sample_token = annotation_record['sample_token']
    current_timestamp = sample_timestamp_by_token.get(current_sample_token)
    if current_timestamp is None:
        return None
    rotation_t = quat_to_rot_matrix(ego2global_rotation).T
    velocity_xy_list = []
    for neighbor_key in ('prev', 'next'):
        neighbor_token = annotation_record.get(neighbor_key, '')
        if not neighbor_token:
            continue
        neighbor_record = annotation_by_token.get(neighbor_token)
        if neighbor_record is None:
            continue
        neighbor_sample_token = neighbor_record['sample_token']
        neighbor_timestamp = sample_timestamp_by_token.get(neighbor_sample_token)
        if neighbor_timestamp is None or neighbor_timestamp == current_timestamp:
            continue
        neighbor_translation = np.asarray(
            neighbor_record['translation'], dtype=np.float32)
        dt = abs(float(neighbor_timestamp - current_timestamp)) * float(timestamp_to_sec)
        if dt <= 1e-6:
            continue
        if neighbor_key == 'next':
            displacement_global = neighbor_translation - current_translation
        else:
            displacement_global = current_translation - neighbor_translation
        velocity_global = displacement_global / dt
        velocity_ego = (rotation_t @ velocity_global.reshape(3, 1)).reshape(3)
        velocity_xy_list.append(velocity_ego[:2])
    if not velocity_xy_list:
        return None
    return np.mean(np.stack(velocity_xy_list, axis=0), axis=0).astype(np.float32)


def analyze_annotation_triplet_jump(
        annotation_record: Mapping[str, object],
        annotation_by_token: Mapping[str, Mapping[str, object]],
        sample_timestamp_by_token: Mapping[str, int],
        timestamp_to_sec: float = DEFAULT_SAMPLE_TIMESTAMP_TO_SEC) -> Optional[Dict[str, float]]:
    prev_token = normalize_class_name(annotation_record.get('prev', ''))
    next_token = normalize_class_name(annotation_record.get('next', ''))
    if not prev_token or not next_token:
        return None
    prev_record = annotation_by_token.get(prev_token)
    next_record = annotation_by_token.get(next_token)
    if prev_record is None or next_record is None:
        return None

    current_sample_token = normalize_class_name(annotation_record['sample_token'])
    prev_sample_token = normalize_class_name(prev_record['sample_token'])
    next_sample_token = normalize_class_name(next_record['sample_token'])
    current_timestamp = sample_timestamp_by_token.get(current_sample_token)
    prev_timestamp = sample_timestamp_by_token.get(prev_sample_token)
    next_timestamp = sample_timestamp_by_token.get(next_sample_token)
    if current_timestamp is None or prev_timestamp is None or next_timestamp is None:
        return None

    dt_prev = (float(current_timestamp - prev_timestamp)) * float(timestamp_to_sec)
    dt_next = (float(next_timestamp - current_timestamp)) * float(timestamp_to_sec)
    if dt_prev <= 1e-6 or dt_next <= 1e-6:
        return None

    prev_xy = np.asarray(prev_record['translation'][:2], dtype=np.float32)
    current_xy = np.asarray(annotation_record['translation'][:2], dtype=np.float32)
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
    return dict(
        v_prev_cur=float(v_prev_cur),
        v_cur_next=float(v_cur_next),
        v_prev_next=float(v_prev_next),
        mid_err=float(mid_err),
        jump_ratio=float(jump_ratio),
    )


def select_consistency_support_mask(
        sub_semantics: np.ndarray,
        inside_mask: np.ndarray,
        class_idx: int,
        dynamic_class_indices: Sequence[int],
        empty_idx: int,
        min_voxels: int = 1) -> Tuple[np.ndarray, str]:
    candidate_masks = [
        ('same_class', inside_mask & (sub_semantics == class_idx)),
        ('dynamic', inside_mask & np.isin(sub_semantics, dynamic_class_indices)),
        ('occupied', inside_mask & (sub_semantics != empty_idx) & (sub_semantics != 255)),
        ('inside_box', inside_mask),
    ]
    best_name = 'inside_box'
    best_mask = inside_mask
    best_count = int(inside_mask.sum())
    for mask_name, mask in candidate_masks:
        voxel_count = int(mask.sum())
        if voxel_count >= int(min_voxels):
            return mask, mask_name
        if voxel_count > best_count:
            best_name = mask_name
            best_mask = mask
            best_count = voxel_count
    return best_mask, best_name


def mask_to_points_ego(
        x_centers: np.ndarray,
        y_centers: np.ndarray,
        z_centers: np.ndarray,
        mask: np.ndarray) -> np.ndarray:
    voxel_indices = np.nonzero(mask)
    if len(voxel_indices[0]) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack([
        x_centers[voxel_indices[0]],
        y_centers[voxel_indices[1]],
        z_centers[voxel_indices[2]],
    ], axis=1).astype(np.float32)


def sample_semantics_at_ego_points(
        points_ego: np.ndarray,
        semantics: np.ndarray,
        grid_meta: VoxelGridMeta) -> Tuple[np.ndarray, np.ndarray]:
    points_ego = np.asarray(points_ego, dtype=np.float32).reshape(-1, 3)
    if points_ego.size == 0:
        return np.zeros((0,), dtype=np.uint8), np.zeros((0,), dtype=bool)
    voxel_indices = np.floor(
        (points_ego - grid_meta.point_cloud_range[:3][None, :]) /
        grid_meta.voxel_size[None, :]).astype(np.int32)
    valid_mask = (
        (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < grid_meta.grid_shape[0]) &
        (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < grid_meta.grid_shape[1]) &
        (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < grid_meta.grid_shape[2]))
    labels = np.full((points_ego.shape[0],), 255, dtype=np.uint8)
    if valid_mask.any():
        valid_indices = voxel_indices[valid_mask]
        labels[valid_mask] = semantics[
            valid_indices[:, 0],
            valid_indices[:, 1],
            valid_indices[:, 2],
        ]
    return labels, valid_mask


def evaluate_flow_consistency_against_future_occ(
        support_points_ego: np.ndarray,
        flow_xy: Sequence[float],
        horizon_sec: float,
        current_ego2global_rotation: Sequence[float],
        current_ego2global_translation: Sequence[float],
        future_ego2global_rotation: Sequence[float],
        future_ego2global_translation: Sequence[float],
        future_semantics: np.ndarray,
        grid_meta: VoxelGridMeta,
        class_idx: int,
        dynamic_class_indices: Sequence[int],
        empty_idx: int) -> MotionConsistency:
    support_points_ego = np.asarray(support_points_ego, dtype=np.float32).reshape(-1, 3)
    if support_points_ego.size == 0:
        return MotionConsistency(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    shifted_points = support_points_ego.copy()
    shifted_points[:, :2] += np.asarray(flow_xy, dtype=np.float32).reshape(1, 2) * float(horizon_sec)
    points_global = transform_points_ego_to_global(
        shifted_points,
        ego2global_rotation=current_ego2global_rotation,
        ego2global_translation=current_ego2global_translation,
    )
    future_points_ego = transform_points_global_to_ego(
        points_global,
        ego2global_rotation=future_ego2global_rotation,
        ego2global_translation=future_ego2global_translation,
    )
    labels, valid_mask = sample_semantics_at_ego_points(
        future_points_ego,
        semantics=future_semantics,
        grid_meta=grid_meta,
    )
    support_count = int(support_points_ego.shape[0])
    valid_count = int(valid_mask.sum())
    if valid_count <= 0:
        return MotionConsistency(support_count, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    valid_labels = labels[valid_mask]
    same_class_ratio = float((valid_labels == class_idx).mean())
    dynamic_ratio = float(np.isin(valid_labels, dynamic_class_indices).mean())
    occupied_ratio = float(
        np.logical_and(valid_labels != empty_idx, valid_labels != 255).mean())
    valid_ratio = float(valid_count) / max(float(support_count), 1.0)
    score = same_class_ratio + 0.35 * dynamic_ratio + 0.1 * occupied_ratio
    return MotionConsistency(
        support_count=support_count,
        valid_count=valid_count,
        same_class_ratio=same_class_ratio,
        dynamic_ratio=dynamic_ratio,
        occupied_ratio=occupied_ratio,
        valid_ratio=valid_ratio,
        score=score,
    )


def compute_box_index_bounds(
        box: Sequence[float],
        grid_meta: VoxelGridMeta,
        z_scale: float = 1.0) -> Optional[Tuple[int, int, int, int, int, int]]:
    center_x, center_y, center_z, size_x, size_y, size_z, yaw = [
        float(value) for value in box[:7]
    ]
    cos_yaw = abs(np.cos(yaw))
    sin_yaw = abs(np.sin(yaw))
    half_x = 0.5 * (cos_yaw * size_x + sin_yaw * size_y)
    half_y = 0.5 * (sin_yaw * size_x + cos_yaw * size_y)
    half_z = 0.5 * size_z * max(float(z_scale), 1e-6)

    pc_range = grid_meta.point_cloud_range
    voxel_size = grid_meta.voxel_size
    grid_x, grid_y, grid_z = grid_meta.grid_shape
    x0 = max(0, int(np.floor((center_x - half_x - pc_range[0]) / voxel_size[0])))
    x1 = min(grid_x, int(np.ceil((center_x + half_x - pc_range[0]) / voxel_size[0])))
    y0 = max(0, int(np.floor((center_y - half_y - pc_range[1]) / voxel_size[1])))
    y1 = min(grid_y, int(np.ceil((center_y + half_y - pc_range[1]) / voxel_size[1])))
    z0 = max(0, int(np.floor((center_z - half_z - pc_range[2]) / voxel_size[2])))
    z1 = min(grid_z, int(np.ceil((center_z + half_z - pc_range[2]) / voxel_size[2])))
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return None
    return x0, x1, y0, y1, z0, z1


def rasterize_dynamic_boxes_to_occflow(
        semantics: np.ndarray,
        gt_boxes: np.ndarray,
        gt_names: Sequence[object],
        gt_velocity: np.ndarray,
        valid_flag: Optional[Sequence[bool]],
        stationary_mask: Optional[Sequence[bool]],
        grid_meta: VoxelGridMeta,
        class_name_to_idx: Mapping[str, int],
        dynamic_class_names: Sequence[str] = DEFAULT_DYNAMIC_CLASS_NAMES,
        empty_idx: int = DEFAULT_EMPTY_IDX,
        max_speed: float = 60.0,
        future_semantics: Optional[np.ndarray] = None,
        future_horizon_sec: Optional[float] = None,
        current_ego2global_rotation: Optional[Sequence[float]] = None,
        current_ego2global_translation: Optional[Sequence[float]] = None,
        future_ego2global_rotation: Optional[Sequence[float]] = None,
        future_ego2global_translation: Optional[Sequence[float]] = None,
        next_occ_score_margin: float = 0.05,
        next_occ_stationary_max_speed: float = 2.0,
        next_occ_min_support_voxels: int = 8,
        z_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    dynamic_class_names = [normalize_class_name(name) for name in dynamic_class_names]
    dynamic_class_indices = tuple(class_name_to_idx[name] for name in dynamic_class_names)
    filled_semantics = np.asarray(semantics, dtype=np.uint8).copy()
    flow = np.zeros(semantics.shape + (2,), dtype=np.float32)
    assigned_mask = np.zeros(semantics.shape, dtype=bool)
    assign_score = np.full(semantics.shape, np.inf, dtype=np.float32)
    stats: Dict[str, int] = dict(
        boxes_total=int(len(gt_boxes)),
        boxes_valid=0,
        boxes_dynamic=0,
        boxes_assigned=0,
        boxes_stationary=0,
        boxes_stationary_by_next_occ=0,
        boxes_skipped_invalid_velocity=0,
        boxes_next_occ_evaluated=0,
        boxes_next_occ_supported=0,
        boxes_next_occ_ambiguous=0,
        voxels_assigned=0,
        voxels_stationary=0,
    )
    valid_flag = np.asarray(valid_flag, dtype=bool) if valid_flag is not None else None
    stationary_mask = (
        np.asarray(stationary_mask, dtype=bool)
        if stationary_mask is not None else np.zeros(len(gt_boxes), dtype=bool))

    for box_index, (box, name, velocity_xy) in enumerate(
            zip(gt_boxes, gt_names, gt_velocity)):
        if valid_flag is not None and box_index < len(valid_flag) and not bool(valid_flag[box_index]):
            continue
        stats['boxes_valid'] += 1
        class_name = normalize_class_name(name)
        if class_name not in dynamic_class_names:
            continue
        stats['boxes_dynamic'] += 1
        velocity_xy = sanitize_velocity(velocity_xy, max_speed=max_speed)
        if not np.isfinite(velocity_xy).all():
            stats['boxes_skipped_invalid_velocity'] += 1
            continue

        bounds = compute_box_index_bounds(box, grid_meta, z_scale=z_scale)
        if bounds is None:
            continue
        x0, x1, y0, y1, z0, z1 = bounds
        center_x, center_y, center_z, size_x, size_y, size_z, yaw = [
            float(value) for value in box[:7]
        ]
        sub_source_semantics = semantics[x0:x1, y0:y1, z0:z1]

        xs = grid_meta.x_centers[x0:x1][:, None, None] - center_x
        ys = grid_meta.y_centers[y0:y1][None, :, None] - center_y
        zs = grid_meta.z_centers[z0:z1][None, None, :] - center_z
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        local_x = cos_yaw * xs + sin_yaw * ys
        local_y = -sin_yaw * xs + cos_yaw * ys
        half_z = size_z * 0.5 * max(float(z_scale), 1e-6)
        inside_mask = (
            (np.abs(local_x) <= size_x * 0.5 + 1e-4) &
            (np.abs(local_y) <= size_y * 0.5 + 1e-4) &
            (np.abs(zs) <= half_z + 1e-4))
        if not inside_mask.any():
            continue

        score = (
            (local_x / max(size_x * 0.5, 1e-3)) ** 2 +
            (local_y / max(size_y * 0.5, 1e-3)) ** 2 +
            (zs / max(half_z, 1e-3)) ** 2).astype(np.float32)
        sub_assign_score = assign_score[x0:x1, y0:y1, z0:z1]
        is_stationary = bool(
            stationary_mask[min(box_index, len(stationary_mask) - 1)]
        ) if len(stationary_mask) > 0 else False
        if (
                future_semantics is not None and
                future_horizon_sec is not None and
                float(future_horizon_sec) > 0.0 and
                current_ego2global_rotation is not None and
                current_ego2global_translation is not None and
                future_ego2global_rotation is not None and
                future_ego2global_translation is not None):
            class_idx = class_name_to_idx[class_name]
            support_mask, _ = select_consistency_support_mask(
                sub_source_semantics,
                inside_mask=inside_mask,
                class_idx=class_idx,
                dynamic_class_indices=dynamic_class_indices,
                empty_idx=empty_idx,
                min_voxels=next_occ_min_support_voxels,
            )
            support_points = mask_to_points_ego(
                grid_meta.x_centers[x0:x1],
                grid_meta.y_centers[y0:y1],
                grid_meta.z_centers[z0:z1],
                support_mask,
            )
            if support_points.shape[0] >= int(next_occ_min_support_voxels):
                moving_consistency = evaluate_flow_consistency_against_future_occ(
                    support_points_ego=support_points,
                    flow_xy=velocity_xy,
                    horizon_sec=float(future_horizon_sec),
                    current_ego2global_rotation=current_ego2global_rotation,
                    current_ego2global_translation=current_ego2global_translation,
                    future_ego2global_rotation=future_ego2global_rotation,
                    future_ego2global_translation=future_ego2global_translation,
                    future_semantics=future_semantics,
                    grid_meta=grid_meta,
                    class_idx=class_idx,
                    dynamic_class_indices=dynamic_class_indices,
                    empty_idx=empty_idx,
                )
                zero_consistency = evaluate_flow_consistency_against_future_occ(
                    support_points_ego=support_points,
                    flow_xy=np.zeros(2, dtype=np.float32),
                    horizon_sec=float(future_horizon_sec),
                    current_ego2global_rotation=current_ego2global_rotation,
                    current_ego2global_translation=current_ego2global_translation,
                    future_ego2global_rotation=future_ego2global_rotation,
                    future_ego2global_translation=future_ego2global_translation,
                    future_semantics=future_semantics,
                    grid_meta=grid_meta,
                    class_idx=class_idx,
                    dynamic_class_indices=dynamic_class_indices,
                    empty_idx=empty_idx,
                )
                if moving_consistency.valid_count > 0 or zero_consistency.valid_count > 0:
                    stats['boxes_next_occ_evaluated'] += 1
                    zero_has_evidence = (
                        zero_consistency.same_class_ratio >= 0.05 or
                        zero_consistency.dynamic_ratio >= 0.10)
                    if (
                            not is_stationary and
                            float(np.linalg.norm(velocity_xy)) <= float(next_occ_stationary_max_speed) and
                            zero_has_evidence and
                            zero_consistency.score >= (
                                moving_consistency.score + float(next_occ_score_margin))):
                        is_stationary = True
                        stats['boxes_stationary_by_next_occ'] += 1
                    elif moving_consistency.score >= zero_consistency.score:
                        stats['boxes_next_occ_supported'] += 1
                    else:
                        stats['boxes_next_occ_ambiguous'] += 1
        update_mask = inside_mask & (score < sub_assign_score)
        if not update_mask.any():
            continue

        sub_flow = flow[x0:x1, y0:y1, z0:z1]
        sub_semantics = filled_semantics[x0:x1, y0:y1, z0:z1]
        sub_assigned_mask = assigned_mask[x0:x1, y0:y1, z0:z1]
        class_idx = class_name_to_idx[class_name]
        sub_flow[update_mask] = 0.0 if is_stationary else velocity_xy
        sub_semantics[update_mask] = class_idx
        sub_assigned_mask[update_mask] = True
        sub_assign_score[update_mask] = score[update_mask]
        flow[x0:x1, y0:y1, z0:z1] = sub_flow
        filled_semantics[x0:x1, y0:y1, z0:z1] = sub_semantics
        assigned_mask[x0:x1, y0:y1, z0:z1] = sub_assigned_mask
        assign_score[x0:x1, y0:y1, z0:z1] = sub_assign_score
        stats['boxes_assigned'] += 1
        stats['voxels_assigned'] += int(update_mask.sum())
        if is_stationary:
            stats['boxes_stationary'] += 1
            stats['voxels_stationary'] += int(update_mask.sum())

    return filled_semantics, flow, assigned_mask, stats


def repair_flow_by_connected_components(
        raw_semantics: np.ndarray,
        assigned_mask: np.ndarray,
        flow: np.ndarray,
        dynamic_class_indices: Sequence[int],
        connectivity: int = 1,
        empty_idx: int = DEFAULT_EMPTY_IDX) -> ConnectivityFlowRepair:
    raw_semantics = np.asarray(raw_semantics)
    assigned_mask = np.asarray(assigned_mask, dtype=bool)
    flow = np.asarray(flow, dtype=np.float32)

    repaired_flow = flow.copy()
    repaired_support_mask = assigned_mask.copy()
    connectivity = max(1, min(int(connectivity), raw_semantics.ndim))
    structure = ndimage.generate_binary_structure(raw_semantics.ndim, connectivity)
    repaired_components = 0

    for class_idx in tuple(int(index) for index in dynamic_class_indices):
        raw_class_mask = raw_semantics == class_idx
        if not raw_class_mask.any():
            continue
        seed_mask = assigned_mask & raw_class_mask
        if not seed_mask.any():
            continue

        component_labels, num_components = ndimage.label(raw_class_mask, structure=structure)
        for component_id in range(1, int(num_components) + 1):
            component_mask = component_labels == component_id
            component_seed_mask = component_mask & seed_mask
            if not component_seed_mask.any():
                continue

            repaired_components += 1
            if component_seed_mask.all():
                repaired_support_mask[component_mask] = True
                continue

            component_coords = np.argwhere(component_mask)
            mins = component_coords.min(axis=0)
            maxs = component_coords.max(axis=0) + 1
            slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))
            component_mask_bbox = component_mask[slices]
            component_seed_bbox = component_seed_mask[slices]
            flow_bbox = repaired_flow[slices]

            _, nearest_indices = ndimage.distance_transform_edt(
                ~component_seed_bbox,
                return_indices=True,
            )
            target_positions = np.where(component_mask_bbox)
            nearest_x = nearest_indices[0][target_positions]
            nearest_y = nearest_indices[1][target_positions]
            nearest_z = nearest_indices[2][target_positions]
            flow_bbox[target_positions] = flow_bbox[
                nearest_x,
                nearest_y,
                nearest_z,
            ]
            repaired_flow[slices] = flow_bbox
            repaired_support_mask[component_mask] = True

    raw_dynamic_mask = np.isin(raw_semantics, tuple(int(index) for index in dynamic_class_indices))
    repaired_support_mask &= raw_dynamic_mask
    return ConnectivityFlowRepair(
        repaired_flow=repaired_flow.astype(np.float32, copy=False),
        repaired_support_mask=repaired_support_mask,
        strict_support_voxels=int(assigned_mask.sum()),
        repaired_support_voxels=int(repaired_support_mask.sum()),
        repaired_components=int(repaired_components),
    )


def _reshape_blocks(arr: np.ndarray, downscale: int) -> np.ndarray:
    if downscale == 1:
        return arr[..., None]
    size_x, size_y, size_z = arr.shape[:3]
    ds = int(downscale)
    block_shape = (
        size_x // ds, ds,
        size_y // ds, ds,
        size_z // ds, ds)
    if arr.ndim == 3:
        return arr.reshape(block_shape).transpose(0, 2, 4, 1, 3, 5).reshape(
            size_x // ds, size_y // ds, size_z // ds, ds ** 3)
    if arr.ndim == 4 and arr.shape[-1] == 2:
        return arr.reshape(block_shape + (2,)).transpose(
            0, 2, 4, 1, 3, 5, 6).reshape(
                size_x // ds, size_y // ds, size_z // ds, ds ** 3, 2)
    raise ValueError(f'Unsupported block reshape shape: {arr.shape}')


def downsample_semantics(
        semantics: np.ndarray,
        downscale: int,
        empty_idx: int = DEFAULT_EMPTY_IDX,
        priority_class_indices: Sequence[int] = ()) -> np.ndarray:
    semantics = np.asarray(semantics)
    if downscale == 1:
        return semantics.astype(np.uint8, copy=False)
    blocks = _reshape_blocks(semantics, downscale)
    empty_count = (blocks == empty_idx).sum(axis=-1)
    counts = np.stack(
        [(blocks == class_idx).sum(axis=-1) for class_idx in range(empty_idx)],
        axis=-1)
    majority = counts.argmax(axis=-1).astype(np.uint8)
    empty_threshold = int(np.ceil(0.95 * (downscale ** 3)))
    result = np.where(empty_count >= empty_threshold, empty_idx, majority)
    priority_indices = tuple(
        int(index) for index in priority_class_indices
        if 0 <= int(index) < empty_idx)
    if priority_indices:
        priority_counts = counts[..., list(priority_indices)]
        priority_local = priority_counts.argmax(axis=-1)
        priority_count = priority_counts.max(axis=-1)
        priority_class = np.asarray(priority_indices, dtype=np.uint8)[priority_local]
        result = np.where(priority_count > 0, priority_class, result)
    return result.astype(np.uint8)


def downsample_flow(
        flow: np.ndarray,
        semantics: np.ndarray,
        downscale: int,
        dynamic_class_indices: Sequence[int]) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)
    semantics = np.asarray(semantics)
    if downscale == 1:
        return flow.astype(np.float16)
    flow_blocks = _reshape_blocks(flow, downscale)
    semantic_blocks = _reshape_blocks(semantics, downscale)
    dynamic_class_indices = tuple(int(index) for index in dynamic_class_indices)
    if not dynamic_class_indices:
        return np.zeros(flow_blocks.shape[:3] + (2,), dtype=np.float16)

    dynamic_mask = np.zeros_like(semantic_blocks, dtype=bool)
    for class_idx in dynamic_class_indices:
        dynamic_mask |= semantic_blocks == class_idx
    has_dynamic = dynamic_mask.any(axis=-1)
    class_counts = np.stack(
        [(semantic_blocks == class_idx).sum(axis=-1) for class_idx in dynamic_class_indices],
        axis=-1)
    dominant_local = class_counts.argmax(axis=-1)
    dominant_class = np.asarray(dynamic_class_indices, dtype=np.int64)[dominant_local]
    dominant_mask = (semantic_blocks == dominant_class[..., None]) & dynamic_mask
    flow_sum = (flow_blocks * dominant_mask[..., None]).sum(axis=-2)
    count = dominant_mask.sum(axis=-1)[..., None]
    downsampled = np.divide(
        flow_sum,
        np.maximum(count, 1),
        out=np.zeros_like(flow_sum, dtype=np.float32),
        where=count > 0)
    downsampled[~has_dynamic] = 0.0
    return downsampled.astype(np.float16)


def save_occflow_bundle(
        output_occ_path: str | Path,
        semantics: np.ndarray,
        flow: np.ndarray,
        dynamic_class_indices: Sequence[int],
        empty_idx: int = DEFAULT_EMPTY_IDX,
        downscales: Sequence[int] = (1, 2, 4, 8)) -> None:
    output_occ_path = Path(output_occ_path)
    output_occ_path.mkdir(parents=True, exist_ok=True)
    semantics = np.asarray(semantics, dtype=np.uint8)
    flow = np.asarray(flow, dtype=np.float32)
    np.savez_compressed(
        output_occ_path / 'labels.npz',
        semantics=semantics,
        flow=flow.astype(np.float16))
    for downscale in downscales:
        if int(downscale) == 1:
            continue
        ms_semantics = downsample_semantics(
            semantics,
            downscale,
            empty_idx=empty_idx,
            priority_class_indices=dynamic_class_indices)
        ms_flow = downsample_flow(flow, semantics, downscale, dynamic_class_indices)
        np.savez_compressed(
            output_occ_path / f'labels_1_{int(downscale)}.npz',
            semantics=ms_semantics,
            flow=ms_flow)


def group_infos_by_scene(infos: Iterable[Mapping[str, object]]) -> Dict[str, List[Mapping[str, object]]]:
    scene_infos: Dict[str, List[Mapping[str, object]]] = {}
    for info in infos:
        scene_token = normalize_class_name(info.get('scene_token', 'unknown_scene'))
        scene_infos.setdefault(scene_token, []).append(info)
    for scene_token in scene_infos:
        scene_infos[scene_token].sort(key=lambda item: int(item.get('timestamp', 0)))
    return scene_infos


def orient_xy_grid_to_image(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    if grid.ndim == 2:
        return grid.transpose(1, 0)[::-1, ::-1]
    if grid.ndim == 3:
        return grid.transpose(1, 0, 2)[::-1, ::-1, :]
    return grid


def _window_sum_2d(values: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    radius = max(int(radius), 0)
    if radius <= 0:
        return values.copy()
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode='constant')
    height, width = values.shape
    window_sum = np.zeros((height, width), dtype=np.float32)
    for offset_x in range(2 * radius + 1):
        for offset_y in range(2 * radius + 1):
            window_sum += padded[
                offset_x:offset_x + height,
                offset_y:offset_y + width,
            ]
    return window_sum


def fill_flow_bev_holes(
        flow_bev: np.ndarray,
        valid_mask: np.ndarray,
        fill_iterations: int = 0,
        neighbor_radius: int = 1,
        min_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    flow_bev = np.asarray(flow_bev, dtype=np.float32).copy()
    valid_mask = np.asarray(valid_mask, dtype=bool).copy()
    fill_iterations = max(int(fill_iterations), 0)
    if fill_iterations <= 0:
        return flow_bev, valid_mask

    for _ in range(fill_iterations):
        valid_float = valid_mask.astype(np.float32)
        neighbor_count = _window_sum_2d(valid_float, radius=neighbor_radius)
        fill_mask = np.logical_and(~valid_mask, neighbor_count >= float(min_neighbors))
        if not fill_mask.any():
            break
        flow_sum_x = _window_sum_2d(
            flow_bev[..., 0] * valid_float, radius=neighbor_radius)
        flow_sum_y = _window_sum_2d(
            flow_bev[..., 1] * valid_float, radius=neighbor_radius)
        denom = np.maximum(neighbor_count, 1e-6)
        flow_bev[..., 0][fill_mask] = flow_sum_x[fill_mask] / denom[fill_mask]
        flow_bev[..., 1][fill_mask] = flow_sum_y[fill_mask] / denom[fill_mask]
        valid_mask[fill_mask] = True
    return flow_bev, valid_mask


def project_dynamic_flow_to_bev(
        flow: np.ndarray,
        semantics: np.ndarray,
        dynamic_class_indices: Sequence[int],
        fill_iterations: int = 0,
        neighbor_radius: int = 1,
        min_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    flow = np.asarray(flow, dtype=np.float32)
    semantics = np.asarray(semantics)
    dynamic_mask = np.zeros_like(semantics, dtype=bool)
    for class_idx in dynamic_class_indices:
        dynamic_mask |= semantics == int(class_idx)
    has_dynamic = dynamic_mask.any(axis=2)
    if not has_dynamic.any():
        return np.zeros(semantics.shape[:2] + (2,), dtype=np.float32), has_dynamic

    reversed_mask = dynamic_mask[:, :, ::-1]
    top_rev = np.argmax(reversed_mask, axis=2)
    top_idx = dynamic_mask.shape[2] - 1 - top_rev
    top_idx = np.where(has_dynamic, top_idx, 0).astype(np.int64)
    flow_bev = np.take_along_axis(flow, top_idx[..., None, None], axis=2)[..., 0, :]
    flow_bev[~has_dynamic] = 0.0
    flow_bev, has_dynamic = fill_flow_bev_holes(
        flow_bev,
        has_dynamic,
        fill_iterations=fill_iterations,
        neighbor_radius=neighbor_radius,
        min_neighbors=min_neighbors,
    )
    return flow_bev.astype(np.float32), has_dynamic


def make_occ_color_map(class_names: Sequence[str] = DEFAULT_CLASS_NAMES) -> np.ndarray:
    palette = {
        'others': (0, 0, 0),
        'barrier': (50, 120, 255),
        'bicycle': (203, 192, 255),
        'bus': (0, 255, 255),
        'car': (245, 150, 0),
        'construction_vehicle': (255, 255, 0),
        'motorcycle': (0, 127, 255),
        'pedestrian': (0, 0, 255),
        'traffic_cone': (150, 240, 255),
        'trailer': (0, 60, 135),
        'truck': (240, 32, 160),
        'driveable_surface': (255, 0, 255),
        'other_flat': (137, 137, 139),
        'sidewalk': (75, 0, 75),
        'terrain': (80, 240, 150),
        'manmade': (250, 230, 230),
        'vegetation': (0, 175, 0),
        'free': (255, 255, 255),
    }
    return np.array(
        [palette.get(str(name), (180, 180, 180)) for name in class_names],
        dtype=np.uint8)


def project_occ_to_bev_labels(semantics: np.ndarray, empty_idx: int) -> np.ndarray:
    semantics = np.asarray(semantics)
    if semantics.ndim != 3:
        raise ValueError(f'Expected [X, Y, Z] semantics, got {semantics.shape}')
    _, _, depth = semantics.shape
    semantics_valid = semantics != empty_idx
    depth_score = np.arange(depth, dtype=np.float32).reshape(1, 1, depth)
    selected = np.argmax(depth_score * semantics_valid, axis=2)
    return np.take_along_axis(semantics, selected[..., None], axis=2)[..., 0].astype(np.int32)


def resize_image_nearest(image: np.ndarray, image_size_hw: Tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(image))
    return np.asarray(
        image.resize((int(image_size_hw[1]), int(image_size_hw[0])), resample=Image.NEAREST))


def render_occ_bev_rgb(
        semantics: np.ndarray,
        empty_idx: int = DEFAULT_EMPTY_IDX,
        color_map: Optional[np.ndarray] = None,
        image_size_hw: Tuple[int, int] = (800, 800)) -> np.ndarray:
    if color_map is None:
        color_map = make_occ_color_map(DEFAULT_CLASS_NAMES)
    occ_bev = project_occ_to_bev_labels(semantics, empty_idx=empty_idx)
    invalid_mask = np.logical_or(occ_bev < 0, occ_bev >= len(color_map))
    occ_bev = np.clip(occ_bev, 0, len(color_map) - 1)
    occ_bev_vis = color_map[occ_bev]
    occ_bev_vis[invalid_mask] = np.array([255, 255, 255], dtype=np.uint8)
    occ_bev_vis = orient_xy_grid_to_image(occ_bev_vis)[..., :3]
    return resize_image_nearest(occ_bev_vis, image_size_hw=image_size_hw)


def flow_to_rgb(vx: np.ndarray, vy: np.ndarray, max_magnitude: Optional[float] = None) -> np.ndarray:
    magnitude = np.sqrt(vx ** 2 + vy ** 2)
    if max_magnitude is None:
        max_magnitude = float(np.max(magnitude))
    max_magnitude = max(float(max_magnitude), 1e-6)
    angle = np.arctan2(vy, vx)
    hue = (angle + np.pi) / (2.0 * np.pi)
    saturation = np.clip(magnitude / max_magnitude, 0.0, 1.0)
    value = np.ones_like(saturation)
    hsv = np.stack((hue, saturation, value), axis=-1)
    rgb = np.apply_along_axis(lambda item: colorsys.hsv_to_rgb(*item), -1, hsv)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
