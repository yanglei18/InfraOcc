import os
import sys
from os import path as osp

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config


_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def load_config(config_path):
    return Config.fromfile(config_path)


def get_split_cfg(cfg, split):
    if split not in cfg.data:
        raise KeyError(f'Unknown split {split!r}. Available: {list(cfg.data.keys())}')
    split_cfg = cfg.data[split].copy()
    split_cfg.test_mode = True
    return split_cfg


def resize_semantics(label, target_shape):
    label_tensor = torch.as_tensor(label, dtype=torch.float32)[None, None]
    resized = F.interpolate(
        label_tensor,
        size=tuple(int(dim) for dim in target_shape),
        mode='nearest')
    return resized[0, 0].cpu().numpy().astype(np.int64)


def flatten_results(results):
    flattened = []
    seen = set()
    for result in results:
        indices = result['index']
        predictions = result['occ_results']
        analysis_payloads = result.get('analysis_payload')
        for entry_index, sample_index in enumerate(indices):
            if sample_index in seen:
                continue
            seen.add(sample_index)
            flattened.append(dict(
                index=int(sample_index),
                occ_results=np.asarray(predictions[entry_index]),
                analysis_payload=(
                    analysis_payloads[entry_index]
                    if analysis_payloads is not None else None),
            ))
    flattened.sort(key=lambda item: item['index'])
    return flattened


def load_results(results_path):
    return flatten_results(mmcv.load(results_path))


def update_confusion(confusion, prediction, target, ignore_index=255):
    prediction = np.asarray(prediction, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    valid_mask = (
        (target != ignore_index)
        & (target >= 0)
        & (target < confusion.shape[0]))
    if not np.any(valid_mask):
        return
    prediction = prediction[valid_mask]
    target = target[valid_mask]
    bincount = np.bincount(
        target * confusion.shape[1] + prediction,
        minlength=confusion.size)
    confusion += bincount.reshape(confusion.shape)


def confusion_to_iou(confusion):
    confusion = confusion.astype(np.float64, copy=False)
    intersection = np.diag(confusion)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - intersection
    iou = np.zeros(confusion.shape[0], dtype=np.float64)
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid] * 100.0
    return iou


def load_occ_gt(dataset, index):
    info = dataset.data_infos[index]
    occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'), allow_pickle=True)
    return occ_gt['semantics']


def class_names_from_cfg(cfg):
    occupancy_head = getattr(cfg.model, 'occupancy_head', None)
    class_names = None
    if occupancy_head is not None:
        class_names = occupancy_head.get('class_names', None)
    if class_names is None:
        class_names = getattr(cfg, 'occ_class_names', None)
    if class_names is None:
        class_names = getattr(cfg, 'class_names', None)
    if class_names is None:
        raise KeyError(
            'Unable to infer class names from config. Expected one of '
            'cfg.model.occupancy_head.class_names, cfg.occ_class_names, or cfg.class_names.')
    return list(class_names)


def group_indices_from_cfg(cfg):
    class_names = class_names_from_cfg(cfg)
    occupancy_head = getattr(cfg.model, 'occupancy_head', {})
    empty_idx = int(occupancy_head.get('empty_idx', class_names.index('free')))
    static_names = list(occupancy_head.get('static_class_names', []))
    dynamic_names = list(occupancy_head.get('dynamic_class_names', []))
    if not static_names:
        static_names = list(getattr(cfg, 'static_occ_class_names', []))
    if not dynamic_names:
        dynamic_names = list(getattr(cfg, 'dynamic_occ_class_names', []))
    if not dynamic_names:
        dynamic_names = [
            name for name in
            ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'truck']
            if name in class_names
        ]
    if not static_names:
        static_names = [
            name for name in class_names
            if name not in set(dynamic_names) and name != 'free'
        ]
    static_indices = [class_names.index(name) for name in static_names]
    dynamic_indices = [class_names.index(name) for name in dynamic_names]
    return dict(
        class_names=class_names,
        empty_idx=empty_idx,
        static_indices=static_indices,
        dynamic_indices=dynamic_indices,
    )


def build_ego_frame_reanchor_matrix(translation, roll_deg=0.0, pitch_deg=0.0,
                                    yaw_deg=0.0):
    translation = tuple(float(value) for value in translation)
    if len(translation) != 3:
        raise ValueError('translation must contain three values.')

    roll = np.deg2rad(float(roll_deg))
    pitch = np.deg2rad(float(pitch_deg))
    yaw = np.deg2rad(float(yaw_deg))

    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_r, -sin_r],
        [0.0, sin_r, cos_r],
    ], dtype=np.float32)
    rot_y = np.array([
        [cos_p, 0.0, sin_p],
        [0.0, 1.0, 0.0],
        [-sin_p, 0.0, cos_p],
    ], dtype=np.float32)
    rot_z = np.array([
        [cos_y, -sin_y, 0.0],
        [sin_y, cos_y, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    rotation = rot_z @ rot_y @ rot_x

    reanchor = np.eye(4, dtype=np.float32)
    reanchor[:3, :3] = rotation
    reanchor[:3, 3] = np.asarray(translation, dtype=np.float32)
    return reanchor


def warp_semantics_by_reanchor(label, point_cloud_range, reanchor_mat,
                               empty_idx=255):
    label = np.asarray(label)
    if label.ndim != 3:
        raise ValueError(f'Expected a 3D voxel grid, got shape {label.shape}.')

    x_min, y_min, _, x_max, y_max, _ = [float(value) for value in point_cloud_range]
    size_x, size_y, size_z = [int(dim) for dim in label.shape]
    if size_x <= 0 or size_y <= 0 or size_z <= 0:
        return label.copy()

    voxel_x = (x_max - x_min) / float(size_x)
    voxel_y = (y_max - y_min) / float(size_y)
    x_centers = x_min + (np.arange(size_x, dtype=np.float32) + 0.5) * voxel_x
    y_centers = y_min + (np.arange(size_y, dtype=np.float32) + 0.5) * voxel_y
    target_x, target_y = np.meshgrid(x_centers, y_centers, indexing='ij')

    target_xyz1 = np.stack([
        target_x,
        target_y,
        np.zeros_like(target_x, dtype=np.float32),
        np.ones_like(target_x, dtype=np.float32),
    ], axis=-1)
    source_xyz1 = target_xyz1 @ reanchor_mat.T

    source_x = (source_xyz1[..., 0] - x_min) / voxel_x - 0.5
    source_y = (source_xyz1[..., 1] - y_min) / voxel_y - 0.5
    valid_mask = (
        (source_x >= 0.0) & (source_x <= max(size_x - 1, 0))
        & (source_y >= 0.0) & (source_y <= max(size_y - 1, 0)))

    source_x = np.rint(source_x).astype(np.int64)
    source_y = np.rint(source_y).astype(np.int64)
    source_x = np.clip(source_x, 0, max(size_x - 1, 0))
    source_y = np.clip(source_y, 0, max(size_y - 1, 0))

    warped = np.full_like(label, fill_value=int(empty_idx))
    for z_index in range(size_z):
        sampled = label[source_x, source_y, z_index]
        sampled = sampled.astype(label.dtype, copy=False)
        sampled = np.where(valid_mask, sampled, int(empty_idx))
        warped[:, :, z_index] = sampled
    return warped
