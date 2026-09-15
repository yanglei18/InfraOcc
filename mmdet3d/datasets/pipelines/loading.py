# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from zipfile import BadZipFile

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES

from torchvision.transforms.functional import rotate


def _warp_bev_feature_map(feature_map, aug_xy, fill_value):
    if feature_map.dim() != 4:
        raise ValueError(
            f'Expected feature map with shape [B, C, H, W], got {feature_map.shape}.'
        )

    batch_size, _, height, width = feature_map.shape
    device = feature_map.device
    dtype = feature_map.dtype

    center_x = (float(width) - 1.0) * 0.5
    center_y = (float(height) - 1.0) * 0.5
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing='ij')
    target_xy = torch.stack([
        grid_x - center_x,
        grid_y - center_y,
    ], dim=-1).reshape(1, height * width, 2).expand(batch_size, -1, -1)

    inverse_aug = torch.inverse(aug_xy.to(device=device, dtype=dtype))
    source_xy = torch.bmm(target_xy, inverse_aug.transpose(1, 2))
    source_xy[..., 0] += center_x
    source_xy[..., 1] += center_y

    norm_x = (source_xy[..., 0] / max(center_x, 1.0)) - 1.0
    norm_y = (source_xy[..., 1] / max(center_y, 1.0)) - 1.0
    sampling_grid = torch.stack([norm_x, norm_y],
                                dim=-1).view(batch_size, height, width, 2)

    warped = F.grid_sample(
        feature_map,
        sampling_grid,
        mode='nearest',
        padding_mode='zeros',
        align_corners=True)
    if float(fill_value) != 0.0:
        valid_mask = ((sampling_grid[..., 0].abs() <= 1.0)
                      & (sampling_grid[..., 1].abs() <= 1.0))
        warped = torch.where(valid_mask[:, None, :, :], warped,
                             warped.new_full((1, ), float(fill_value)))
    return warped


def _warp_bev_tensor_by_bda(tensor, bda_mat, fill_value=0):
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    aug_xy = bda_mat[:, :2, :2].to(device=tensor.device, dtype=torch.float32)
    warped = _warp_bev_feature_map(
        tensor.to(dtype=torch.float32), aug_xy=aug_xy, fill_value=fill_value)
    warped = warped.to(dtype=tensor.dtype)
    if squeeze_batch:
        return warped[0]
    return warped


def _warp_semantics_by_reanchor(label,
                                point_cloud_range,
                                reanchor_mat,
                                empty_idx=255):
    label = np.asarray(label)
    if label.ndim != 3:
        raise ValueError(f'Expected a 3D voxel grid, got shape {label.shape}.')

    x_min, y_min, _, x_max, y_max, _ = [
        float(value) for value in point_cloud_range
    ]
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
    ],
                           axis=-1)
    source_xyz1 = target_xyz1 @ np.asarray(reanchor_mat, dtype=np.float32).T

    source_x = (source_xyz1[..., 0] - x_min) / voxel_x - 0.5
    source_y = (source_xyz1[..., 1] - y_min) / voxel_y - 0.5
    valid_mask = ((source_x >= 0.0) & (source_x <= max(size_x - 1, 0))
                  & (source_y >= 0.0) & (source_y <= max(size_y - 1, 0)))

    source_x = np.rint(source_x).astype(np.int64)
    source_y = np.rint(source_y).astype(np.int64)
    source_x = np.clip(source_x, 0, max(size_x - 1, 0))
    source_y = np.clip(source_y, 0, max(size_y - 1, 0))

    warped = np.full_like(label, fill_value=int(empty_idx))
    for z_index in range(size_z):
        sampled = label[source_x, source_y, z_index]
        sampled = sampled.astype(label.dtype, copy=False)
        warped[..., z_index] = np.where(valid_mask, sampled, empty_idx)
    return warped


def _warp_flow_by_reanchor(flow, point_cloud_range, reanchor_mat):
    """Reanchor flow locations and rotate/scale their displacement vectors."""
    flow = np.asarray(flow)
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError(
            f'Expected flow with shape [X, Y, Z, 2], got {flow.shape}.')

    x_min, y_min, _, x_max, y_max, _ = [
        float(value) for value in point_cloud_range
    ]
    size_x, size_y, size_z = [int(dim) for dim in flow.shape[:3]]
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
    ],
                           axis=-1)
    source_xyz1 = target_xyz1 @ np.asarray(reanchor_mat, dtype=np.float32).T
    source_x = (source_xyz1[..., 0] - x_min) / voxel_x - 0.5
    source_y = (source_xyz1[..., 1] - y_min) / voxel_y - 0.5
    valid_mask = ((source_x >= 0.0) & (source_x <= max(size_x - 1, 0))
                  & (source_y >= 0.0) & (source_y <= max(size_y - 1, 0)))
    source_x = np.clip(
        np.rint(source_x).astype(np.int64), 0, max(size_x - 1, 0))
    source_y = np.clip(
        np.rint(source_y).astype(np.int64), 0, max(size_y - 1, 0))

    warped = flow[source_x, source_y, ...].astype(np.float32, copy=False)
    warped = np.where(valid_mask[..., None, None], warped, 0.0)
    affine = np.asarray(reanchor_mat, dtype=np.float32)[:2, :2]
    warped = np.matmul(warped, affine.T)
    return warped.astype(flow.dtype, copy=False)


def _warp_flow_by_bda(flow, bda_mat):
    """Warp flow voxels with BDA and transform their XY vectors."""
    flow = np.asarray(flow)
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError(
            f'Expected flow with shape [X, Y, Z, 2], got {flow.shape}.')

    # [X, Y, Z, 2] -> one BEV feature map per z slice, [Z, 2, Y, X].
    flow_tensor = torch.from_numpy(flow.astype(np.float32,
                                               copy=False)).permute(
                                                   2, 3, 1, 0).contiguous()
    if bda_mat.dim() == 2:
        bda_mat = bda_mat.unsqueeze(0)
    aug_xy = bda_mat[:, :2, :2].to(dtype=torch.float32)
    if aug_xy.shape[0] == 1 and flow_tensor.shape[0] != 1:
        aug_xy = aug_xy.expand(flow_tensor.shape[0], -1, -1)
    warped = _warp_bev_feature_map(flow_tensor, aug_xy=aug_xy, fill_value=0.0)
    vectors = warped.permute(0, 2, 3, 1)
    vectors = torch.einsum('zyxc,zdc->zyxd', vectors, aug_xy)
    return vectors.permute(2, 1, 0, 3).cpu().numpy().astype(
        flow.dtype, copy=False)


def _maybe_reanchor_occ_flow(results, flow, point_cloud_range=None):
    reanchor = results.get('ego_frame_reanchoring')
    if reanchor is None or reanchor.get('matrix') is None:
        return flow
    if point_cloud_range is None:
        point_cloud_range = results.get('point_cloud_range')
    if point_cloud_range is None:
        point_cloud_range = results.get('pc_range')
    if point_cloud_range is None:
        return flow
    return _warp_flow_by_reanchor(
        flow,
        point_cloud_range=point_cloud_range,
        reanchor_mat=reanchor['matrix'])


def _maybe_reanchor_occ_labels(results,
                               semantics,
                               empty_idx,
                               point_cloud_range=None,
                               mask_key=None):
    reanchor = results.get('ego_frame_reanchoring')
    if reanchor is None or reanchor.get('matrix') is None:
        return semantics

    if point_cloud_range is None:
        point_cloud_range = results.get('point_cloud_range')
    if point_cloud_range is None:
        point_cloud_range = results.get('pc_range')
    if point_cloud_range is None:
        return semantics

    warped = _warp_semantics_by_reanchor(
        semantics,
        point_cloud_range=point_cloud_range,
        reanchor_mat=reanchor['matrix'],
        empty_idx=empty_idx)

    if mask_key is not None and mask_key in results:
        warped_mask = _warp_semantics_by_reanchor(
            np.asarray(results[mask_key], dtype=np.uint8),
            point_cloud_range=point_cloud_range,
            reanchor_mat=reanchor['matrix'],
            empty_idx=0)
        results[mask_key] = warped_mask.astype(bool)

    return warped


@PIPELINES.register_module()
class LoadOccGTFromFile(object):

    def __init__(
        self,
        scale_1_2=False,
        scale_1_4=False,
        scale_1_8=False,
        load_mask=False,
        load_flow=False,
        flow_gt_path=None,
        ignore_invisible=False,
        group_list=None,
        point_cloud_range=None,
        dynamic_class_indices=None,
    ):
        self.scale_1_2 = scale_1_2
        self.scale_1_4 = scale_1_4
        self.scale_1_8 = scale_1_8
        self.ignore_invisible = ignore_invisible
        self.group_list = group_list
        self.load_mask = load_mask
        self.load_flow = load_flow
        self.flow_gt_path = flow_gt_path
        self.point_cloud_range = (
            tuple(float(value) for value in point_cloud_range)
            if point_cloud_range is not None else None)
        self.dynamic_class_indices = tuple(
            int(index) for index in (dynamic_class_indices or ()))

    def _resolve_flow_label_path(self, occ_gt_path, label_name='labels.npz'):
        if self.flow_gt_path is None:
            return os.path.join(occ_gt_path, label_name)

        occ_gt_path = os.path.abspath(occ_gt_path)
        flow_root = os.path.abspath(self.flow_gt_path)
        occ_parts = os.path.normpath(occ_gt_path).split(os.sep)
        if 'gts' in occ_parts:
            rel_path = os.path.join(*occ_parts[occ_parts.index('gts') + 1:])
        else:
            rel_path = os.path.basename(occ_gt_path)
        return os.path.join(flow_root, rel_path, label_name)

    @staticmethod
    def _downsample_semantics(semantics,
                              downscale,
                              empty_cls_idx,
                              priority_class_indices=()):
        if downscale == 1:
            return semantics
        h, w, d = semantics.shape
        ds = downscale
        hs, ws, ds_z = h // ds, w // ds, d // ds
        blocks = semantics.reshape(hs, ds, ws, ds, ds_z, ds).transpose(
            0, 2, 4, 1, 3, 5).reshape(hs, ws, ds_z, ds**3)
        empty_count = (blocks == empty_cls_idx).sum(axis=-1)
        counts = np.stack([(blocks == cls_id).sum(axis=-1)
                           for cls_id in range(empty_cls_idx)],
                          axis=-1)
        majority = counts.argmax(axis=-1).astype(np.uint8)
        empty_threshold = int(np.ceil(0.95 * (ds**3)))
        result = np.where(empty_count >= empty_threshold, empty_cls_idx,
                          majority)
        priority_indices = tuple(
            int(index) for index in priority_class_indices
            if 0 <= int(index) < empty_cls_idx)
        if priority_indices:
            priority_counts = counts[..., list(priority_indices)]
            priority_local = priority_counts.argmax(axis=-1)
            priority_count = priority_counts.max(axis=-1)
            priority_class = np.asarray(
                priority_indices, dtype=np.uint8)[priority_local]
            result = np.where(priority_count > 0, priority_class, result)
        return result.astype(np.uint8, copy=False)

    @staticmethod
    def _downsample_flow(flow, semantics, downscale, dynamic_class_indices):
        if downscale == 1:
            return np.asarray(flow, dtype=np.float16)
        flow = np.asarray(flow, dtype=np.float32)
        semantics = np.asarray(semantics)
        h, w, d = semantics.shape
        ds = int(downscale)
        hs, ws, ds_z = h // ds, w // ds, d // ds
        flow_blocks = flow.reshape(hs, ds, ws, ds, ds_z, ds, 2).transpose(
            0, 2, 4, 1, 3, 5, 6).reshape(hs, ws, ds_z, ds**3, 2)
        semantic_blocks = semantics.reshape(
            hs, ds, ws, ds, ds_z,
            ds).transpose(0, 2, 4, 1, 3, 5).reshape(hs, ws, ds_z, ds**3)
        dynamic_indices = tuple(int(index) for index in dynamic_class_indices)
        if not dynamic_indices:
            return np.zeros((hs, ws, ds_z, 2), dtype=np.float16)
        dynamic_mask = np.zeros_like(semantic_blocks, dtype=bool)
        for class_index in dynamic_indices:
            dynamic_mask |= semantic_blocks == class_index
        class_counts = np.stack([(semantic_blocks == class_index).sum(axis=-1)
                                 for class_index in dynamic_indices],
                                axis=-1)
        dominant_local = class_counts.argmax(axis=-1)
        dominant_class = np.asarray(dynamic_indices)[dominant_local]
        dominant_mask = (semantic_blocks
                         == dominant_class[..., None]) & dynamic_mask
        flow_sum = (flow_blocks * dominant_mask[..., None]).sum(axis=-2)
        count = dominant_mask.sum(axis=-1)[..., None]
        downsampled = np.divide(
            flow_sum,
            np.maximum(count, 1),
            out=np.zeros_like(flow_sum),
            where=count > 0)
        downsampled[~dynamic_mask.any(axis=-1)] = 0.0
        return downsampled.astype(np.float16)

    @staticmethod
    def _downsample_mask(mask, downscale):
        if downscale == 1:
            return mask
        h, w, d = mask.shape
        ds = downscale
        hs, ws, ds_z = h // ds, w // ds, d // ds
        blocks = mask.reshape(hs, ds, ws, ds, ds_z,
                              ds).transpose(0, 2, 4, 1, 3,
                                            5).reshape(hs, ws, ds_z, ds**3)
        return (blocks.max(axis=-1) > 0).astype(mask.dtype)

    def _load_or_build_ms_occ(self, label_path, downscale, load_mask):
        if os.path.exists(label_path):
            try:
                with np.load(label_path) as labels:
                    saved_priority = tuple(
                        int(index)
                        for index in labels['dynamic_priority_class_indices']
                    ) if 'dynamic_priority_class_indices' in labels.files else None
                    priority_matches = (
                        not self.dynamic_class_indices
                        or saved_priority == self.dynamic_class_indices)
                    mask_available = (
                        not load_mask or 'mask_camera' in labels.files)
                    if priority_matches and mask_available:
                        return {key: labels[key] for key in labels.files}
            except (BadZipFile, EOFError, OSError, ValueError):
                # A cache created by an interrupted or older non-atomic
                # writer is safe to rebuild from the full-resolution label.
                pass

        full_path = os.path.join(os.path.dirname(label_path), 'labels.npz')
        with np.load(full_path) as full_labels:
            semantics = full_labels['semantics']
            valid_semantics = semantics[semantics != 255]
            empty_cls_idx = (
                int(np.max(valid_semantics)) if valid_semantics.size else 255)
            ms_semantics = self._downsample_semantics(
                semantics,
                downscale=downscale,
                empty_cls_idx=empty_cls_idx,
                priority_class_indices=self.dynamic_class_indices)

            save_kwargs = dict(semantics=ms_semantics)
            if self.dynamic_class_indices:
                save_kwargs['dynamic_priority_class_indices'] = np.asarray(
                    self.dynamic_class_indices, dtype=np.int64)
            if 'flow' in full_labels.files:
                save_kwargs['flow'] = self._downsample_flow(
                    full_labels['flow'], semantics, downscale,
                    self.dynamic_class_indices)
            if load_mask and 'mask_camera' in full_labels.files:
                try:
                    ms_mask = self._downsample_mask(
                        full_labels['mask_camera'], downscale=downscale)
                    save_kwargs['mask_camera'] = ms_mask
                except Exception:
                    # Some datasets store mask_camera as object arrays; ignore
                    # it for auto-generated multi-scale labels in this fallback
                    # path.
                    pass

        label_dir = os.path.dirname(label_path)
        os.makedirs(label_dir, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(
            dir=label_dir,
            prefix='.{}.'.format(os.path.basename(label_path)),
            suffix='.tmp.npz',
            delete=False)
        temp_path = temp_file.name
        temp_file.close()
        try:
            np.savez_compressed(temp_path, **save_kwargs)
            os.replace(temp_path, label_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        if load_mask and 'mask_camera' not in save_kwargs:
            return dict(semantics=ms_semantics)
        return save_kwargs

    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        occ_gt_label = os.path.join(occ_gt_path, "labels.npz")
        occ_gt_label_1_2 = os.path.join(occ_gt_path, "labels_1_2.npz")
        occ_gt_label_1_4 = os.path.join(occ_gt_path, "labels_1_4.npz")
        occ_gt_label_1_8 = os.path.join(occ_gt_path, "labels_1_8.npz")

        occ_labels = np.load(occ_gt_label)

        semantics = occ_labels['semantics']
        empty_idx = int(np.max(semantics[semantics != 255])) if np.any(
            semantics != 255) else 255
        if self.load_mask:
            voxel_mask = occ_labels['mask_camera']
            results['voxel_mask_camera'] = voxel_mask.astype(bool)
            if self.ignore_invisible:
                semantics[voxel_mask == 0] = 255
        semantics = _maybe_reanchor_occ_labels(
            results,
            semantics,
            empty_idx=empty_idx,
            point_cloud_range=self.point_cloud_range,
            mask_key='voxel_mask_camera' if self.load_mask else None)
        results['voxel_semantic'] = semantics

        if self.load_flow:
            flow_label_path = self._resolve_flow_label_path(occ_gt_path)
            if not os.path.exists(flow_label_path):
                raise FileNotFoundError(
                    f'Flow GT file not found: {flow_label_path}')
            flow_labels = np.load(flow_label_path)
            if 'flow' not in flow_labels:
                raise KeyError(
                    f'flow is missing from flow GT file: {flow_label_path}')
            results['voxel_occflows'] = _maybe_reanchor_occ_flow(
                results,
                flow_labels['flow'].astype(np.float32),
                point_cloud_range=self.point_cloud_range)

        if self.scale_1_2:
            occ_labels_1_2 = self._load_or_build_ms_occ(
                occ_gt_label_1_2, downscale=2, load_mask=self.load_mask)
            semantics_1_2 = occ_labels_1_2['semantics']
            empty_idx_1_2 = int(np.max(
                semantics_1_2[semantics_1_2 != 255])) if np.any(
                    semantics_1_2 != 255) else 255

            if self.load_mask:
                voxel_mask = occ_labels_1_2['mask_camera']
                if self.ignore_invisible:
                    semantics_1_2[voxel_mask == 0] = 255
                results['voxel_mask_camera_1_2'] = voxel_mask
            semantics_1_2 = _maybe_reanchor_occ_labels(
                results,
                semantics_1_2,
                empty_idx=empty_idx_1_2,
                point_cloud_range=self.point_cloud_range,
                mask_key='voxel_mask_camera_1_2' if self.load_mask else None)
            results['voxel_semantic_1_2'] = semantics_1_2
            if self.load_flow:
                flow_label_path_1_2 = self._resolve_flow_label_path(
                    occ_gt_path, label_name='labels_1_2.npz')
                if not os.path.exists(flow_label_path_1_2):
                    raise FileNotFoundError(
                        f'Flow GT file not found: {flow_label_path_1_2}')
                flow_labels_1_2 = np.load(flow_label_path_1_2)
                if 'flow' not in flow_labels_1_2:
                    raise KeyError(
                        f'flow is missing from flow GT file: {flow_label_path_1_2}'
                    )
                results['voxel_occflows_1_2'] = _maybe_reanchor_occ_flow(
                    results,
                    flow_labels_1_2['flow'].astype(np.float32),
                    point_cloud_range=self.point_cloud_range)

        if self.scale_1_4:
            occ_labels_1_4 = self._load_or_build_ms_occ(
                occ_gt_label_1_4, downscale=4, load_mask=self.load_mask)
            semantics_1_4 = occ_labels_1_4['semantics']
            empty_idx_1_4 = int(np.max(
                semantics_1_4[semantics_1_4 != 255])) if np.any(
                    semantics_1_4 != 255) else 255

            if self.load_mask:
                voxel_mask = occ_labels_1_4['mask_camera']
                if self.ignore_invisible:
                    semantics_1_4[voxel_mask == 0] = 255
                results['voxel_mask_camera_1_4'] = voxel_mask
            semantics_1_4 = _maybe_reanchor_occ_labels(
                results,
                semantics_1_4,
                empty_idx=empty_idx_1_4,
                point_cloud_range=self.point_cloud_range,
                mask_key='voxel_mask_camera_1_4' if self.load_mask else None)
            results['voxel_semantic_1_4'] = semantics_1_4
            if self.load_flow:
                flow_label_path_1_4 = self._resolve_flow_label_path(
                    occ_gt_path, label_name='labels_1_4.npz')
                if not os.path.exists(flow_label_path_1_4):
                    raise FileNotFoundError(
                        f'Flow GT file not found: {flow_label_path_1_4}')
                flow_labels_1_4 = np.load(flow_label_path_1_4)
                if 'flow' not in flow_labels_1_4:
                    raise KeyError(
                        f'flow is missing from flow GT file: {flow_label_path_1_4}'
                    )
                results['voxel_occflows_1_4'] = _maybe_reanchor_occ_flow(
                    results,
                    flow_labels_1_4['flow'].astype(np.float32),
                    point_cloud_range=self.point_cloud_range)

        if self.scale_1_8:
            occ_labels_1_8 = self._load_or_build_ms_occ(
                occ_gt_label_1_8, downscale=8, load_mask=self.load_mask)
            semantics_1_8 = occ_labels_1_8['semantics']
            empty_idx_1_8 = int(np.max(
                semantics_1_8[semantics_1_8 != 255])) if np.any(
                    semantics_1_8 != 255) else 255

            if self.load_mask:
                voxel_mask = occ_labels_1_8['mask_camera']
                if self.ignore_invisible:
                    semantics_1_8[voxel_mask == 0] = 255
                results['voxel_mask_camera_1_8'] = voxel_mask
            semantics_1_8 = _maybe_reanchor_occ_labels(
                results,
                semantics_1_8,
                empty_idx=empty_idx_1_8,
                point_cloud_range=self.point_cloud_range,
                mask_key='voxel_mask_camera_1_8' if self.load_mask else None)
            results['voxel_semantic_1_8'] = semantics_1_8
            if self.load_flow:
                flow_label_path_1_8 = self._resolve_flow_label_path(
                    occ_gt_path, label_name='labels_1_8.npz')
                if not os.path.exists(flow_label_path_1_8):
                    raise FileNotFoundError(
                        f'Flow GT file not found: {flow_label_path_1_8}')
                flow_labels_1_8 = np.load(flow_label_path_1_8)
                if 'flow' not in flow_labels_1_8:
                    raise KeyError(
                        f'flow is missing from flow GT file: {flow_label_path_1_8}'
                    )
                results['voxel_occflows_1_8'] = _maybe_reanchor_occ_flow(
                    results,
                    flow_labels_1_8['flow'].astype(np.float32),
                    point_cloud_range=self.point_cloud_range)

        return results


@PIPELINES.register_module()
class RepairDetectionLabelsFromOcc(object):
    """Align auxiliary detection labels with dynamic occupancy support."""

    def __init__(self,
                 point_cloud_range,
                 dynamic_class_to_source,
                 min_support_voxels=2,
                 min_support_ratio=0.5):
        self.point_cloud_range = np.asarray(
            point_cloud_range, dtype=np.float32)
        self.dynamic_class_to_source = {
            int(occ_class): int(source_class)
            for occ_class, source_class in dynamic_class_to_source.items()
        }
        # Only source annotations that are already dynamic candidates may be
        # repaired.  Otherwise an overlapping barrier or other static box
        # could be relabeled as a dynamic occupancy class.
        self.dynamic_source_classes = frozenset(
            self.dynamic_class_to_source.values())
        self.min_support_voxels = int(min_support_voxels)
        self.min_support_ratio = float(min_support_ratio)

    def __call__(self, results):
        semantics = results.get('voxel_semantic')
        boxes = results.get('gt_bboxes_3d')
        labels = results.get('gt_labels_3d')
        if semantics is None or boxes is None or labels is None:
            return results
        if hasattr(boxes, 'data'):
            boxes = boxes.data
        if not hasattr(boxes, 'tensor'):
            return results

        semantics = np.asarray(semantics)
        box_array = boxes.tensor.detach().cpu().numpy()
        label_array = np.asarray(labels.detach().cpu() if torch.
                                 is_tensor(labels) else labels).reshape(-1)
        if semantics.ndim != 3 or box_array.ndim != 2:
            return results
        if box_array.shape[0] != label_array.shape[0]:
            return results

        voxel_size = (
            (self.point_cloud_range[3:] - self.point_cloud_range[:3]) /
            np.asarray(semantics.shape, dtype=np.float32))
        grid_centers = [
            self.point_cloud_range[axis] +
            (np.arange(semantics.shape[axis], dtype=np.float32) + 0.5) *
            voxel_size[axis] for axis in range(3)
        ]
        repaired = np.full(label_array.shape, -1, dtype=np.int64)
        dynamic_occ_classes = np.asarray(
            sorted(self.dynamic_class_to_source), dtype=np.int64)

        for box_index, box in enumerate(box_array):
            if int(label_array[box_index]) not in self.dynamic_source_classes:
                continue
            center_z = float(box[2] + box[5] * 0.5)
            x0 = max(
                0,
                int(
                    np.floor(
                        (box[0] - box[3] * 0.5 - self.point_cloud_range[0]) /
                        voxel_size[0])))
            x1 = min(
                semantics.shape[0],
                int(
                    np.ceil(
                        (box[0] + box[3] * 0.5 - self.point_cloud_range[0]) /
                        voxel_size[0])) + 1)
            y0 = max(
                0,
                int(
                    np.floor(
                        (box[1] - box[4] * 0.5 - self.point_cloud_range[1]) /
                        voxel_size[1])))
            y1 = min(
                semantics.shape[1],
                int(
                    np.ceil(
                        (box[1] + box[4] * 0.5 - self.point_cloud_range[1]) /
                        voxel_size[1])) + 1)
            z0 = max(
                0,
                int(
                    np.floor(
                        (center_z - box[5] * 0.5 - self.point_cloud_range[2]) /
                        voxel_size[2])))
            z1 = min(
                semantics.shape[2],
                int(
                    np.ceil(
                        (center_z + box[5] * 0.5 - self.point_cloud_range[2]) /
                        voxel_size[2])) + 1)
            if x0 >= x1 or y0 >= y1 or z0 >= z1:
                continue

            xs = grid_centers[0][x0:x1, None, None] - box[0]
            ys = grid_centers[1][None, y0:y1, None] - box[1]
            zs = grid_centers[2][None, None, z0:z1] - center_z
            cosine, sine = np.cos(box[6]), np.sin(box[6])
            local_x = cosine * xs + sine * ys
            local_y = -sine * xs + cosine * ys
            inside = ((np.abs(local_x) <= box[3] * 0.5 + 1e-4) &
                      (np.abs(local_y) <= box[4] * 0.5 + 1e-4) &
                      (np.abs(zs) <= box[5] * 0.5 + 1e-4))
            values = semantics[x0:x1, y0:y1, z0:z1][inside]
            values = values[np.isin(values, dynamic_occ_classes)]
            if values.size < self.min_support_voxels:
                continue
            counts = np.asarray([
                np.count_nonzero(values == occ_class)
                for occ_class in dynamic_occ_classes
            ])
            winner = int(dynamic_occ_classes[int(counts.argmax())])
            if counts.max() / max(values.size, 1) < self.min_support_ratio:
                continue
            repaired[box_index] = self.dynamic_class_to_source[winner]

        results['gt_labels_3d'] = torch.as_tensor(
            repaired,
            dtype=torch.long,
            device=labels.device if torch.is_tensor(labels) else None)
        return results


@PIPELINES.register_module()
class LoadOccGTFromFileOpenOcc(object):

    def __init__(self,
                 scale_1_2=False,
                 scale_1_4=False,
                 scale_1_8=False,
                 load_ray_mask=False):
        self.scale_1_2 = scale_1_2
        self.scale_1_4 = scale_1_4
        self.scale_1_8 = scale_1_8
        self.load_ray_mask = load_ray_mask

    def __call__(self, results):
        gts_occ_gt_path = results['occ_gt_path']

        occ_ray_mask_path = gts_occ_gt_path.replace('gts',
                                                    'openocc_v2_ray_mask')
        occ_ray_mask = os.path.join(occ_ray_mask_path, 'labels.npz')
        occ_ray_mask_1_2 = os.path.join(occ_ray_mask_path, 'labels_1_2.npz')
        occ_ray_mask_1_4 = os.path.join(occ_ray_mask_path, 'labels_1_4.npz')
        occ_ray_mask_1_8 = os.path.join(occ_ray_mask_path, 'labels_1_8.npz')

        occ_gt_path = gts_occ_gt_path.replace('gts', 'openocc_v2')
        occ_gt_label = os.path.join(occ_gt_path, "labels.npz")
        occ_gt_label_1_2 = os.path.join(occ_gt_path, "labels_1_2.npz")
        occ_gt_label_1_4 = os.path.join(occ_gt_path, "labels_1_4.npz")
        occ_gt_label_1_8 = os.path.join(occ_gt_path, "labels_1_8.npz")
        occ_labels = np.load(occ_gt_label)

        semantics = occ_labels['semantics']
        flow = occ_labels['flow']

        if self.scale_1_2:
            occ_labels_1_2 = np.load(occ_gt_label_1_2)
            semantics_1_2 = occ_labels_1_2['semantics']
            flow_1_2 = occ_labels_1_2['flow']
            results['voxel_semantic_1_2'] = semantics_1_2
            results['voxel_occflows_1_2'] = flow_1_2
            if self.load_ray_mask:
                ray_mask_1_2 = np.load(occ_ray_mask_1_2)
                ray_mask_1_2 = ray_mask_1_2['ray_mask2']
                results['ray_mask_1_2'] = ray_mask_1_2
        if self.scale_1_4:
            occ_labels_1_4 = np.load(occ_gt_label_1_4)
            semantics_1_4 = occ_labels_1_4['semantics']
            flow_1_4 = occ_labels_1_4['flow']
            results['voxel_semantic_1_4'] = semantics_1_4
            results['voxel_occflows_1_4'] = flow_1_4
            if self.load_ray_mask:
                ray_mask_1_4 = np.load(occ_ray_mask_1_4)
                ray_mask_1_4 = ray_mask_1_4['ray_mask2']
                results['ray_mask_1_4'] = ray_mask_1_4
        if self.scale_1_8:
            occ_labels_1_8 = np.load(occ_gt_label_1_8)
            semantics_1_8 = occ_labels_1_8['semantics']
            flow_1_8 = occ_labels_1_8['flow']
            results['voxel_semantic_1_8'] = semantics_1_8
            results['voxel_occflows_1_8'] = flow_1_8
            if self.load_ray_mask:
                ray_mask_1_8 = np.load(occ_ray_mask_1_8)
                ray_mask_1_8 = ray_mask_1_8['ray_mask2']
                results['ray_mask_1_8'] = ray_mask_1_8

        if self.load_ray_mask:
            ray_mask = np.load(occ_ray_mask)
            ray_mask = ray_mask['ray_mask2']
            results['ray_mask'] = ray_mask

        results['voxel_semantic'] = semantics
        results['voxel_occflows'] = flow

        return results


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class PointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config
        self.index = 0
        self.num_cam = 6
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)

    def vis_depth_img(self, img, depth):
        depth = depth.cpu().numpy()
        img = img.permute(1, 2, 0).cpu().numpy()
        img = img * self.std + self.mean
        img = np.array(img, dtype=np.uint8)
        invalid_y, invalid_x, invalid_c = np.where(img == 0)
        depth[invalid_y, invalid_x] = 0
        y, x = np.where(depth != 0)
        plt.figure()
        plt.imshow(img)
        plt.scatter(x, y, c=depth[y, x], cmap='rainbow_r', alpha=0.5, s=2)
        plt.show()
        self.index = self.index + 1

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)

        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]

        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        # prev process info
        points_source = results.get('points_for_depth', results['points'])
        points_lidar = points_source.tensor[:, :3]
        imgs, sensor2egos, ego2globals, cam2imgs, post_augs, bda = results[
            'img_inputs']
        lidar2imgs = results['lidar2img']
        nt, c, h, w = imgs.shape
        num_cams = len(results['cam_names'])
        if num_cams <= 0 or nt % num_cams:
            raise ValueError(
                'PointToMultiViewDepth requires an integral number of frames '
                f'per camera, got {nt} images for {num_cams} cameras.')
        t_frame = nt // num_cams

        # store list
        depth_maps = []  # process result

        vis_index = 0
        for cid in range(len(results['cam_names'])):
            lidar2img = lidar2imgs[cid]

            # project lidar point to img plane
            points_img = lidar2img @ torch.cat(
                [points_lidar.T,
                 torch.ones((1, points_lidar.shape[0]))],
                dim=0)
            points_img = points_img.permute(1, 0)
            points_img = torch.cat([
                points_img[:, :2] / points_img[:, 2].unsqueeze(1),
                points_img[:, 2].unsqueeze(1)
            ],
                                   dim=1)

            # get corresponding depth value
            depth_map = self.points2depthmap(points_img, h, w)

            # store
            depth_maps.append(depth_map)

            # vis depth img to check the correctness
            # self.vis_depth_img(imgs[cid*t_frame], depth_map)

        results['gt_depth'] = torch.stack(depth_maps)
        return results


def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    to_rgb = True
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class PrepareImageInputs(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        opencv_pp=False,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.opencv_pp = opencv_pp

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        if not self.opencv_pp:
            img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
        if self.opencv_pp:
            img = self.img_transform_core_opencv(img, post_rot, post_tran,
                                                 crop)

        copy_img = img.copy()
        invalid_index = np.where(np.array(copy_img) == 0)

        return img, post_rot, post_tran, invalid_index

    def img_transform_core_opencv(self, img, post_rot, post_tran, crop):
        img = np.array(img).astype(np.float32)
        img = cv2.warpAffine(
            img,
            np.concatenate([post_rot, post_tran.reshape(2, 1)], axis=1),
            (crop[2] - crop[0], crop[3] - crop[1]),
            flags=cv2.INTER_LINEAR)
        return img

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        keep_ratio_with_pad = bool(
            self.data_config.get('keep_ratio_with_pad', False))
        if keep_ratio_with_pad:
            # Roadside cameras often place useful traffic context higher in the
            # frame. In this mode we first discard a fixed top strip, then fit
            # the remaining image to the target height and pad the width so the
            # surviving content is preserved instead of being cropped again.
            crop_top_ratio = float(self.data_config.get('crop_top_ratio', 0.0))
            crop_top_ratio = float(np.clip(crop_top_ratio, 0.0, 0.95))
            visible_height = max(float(H) * (1.0 - crop_top_ratio), 1.0)
            resize = float(fH) / visible_height
            resize_dims = (
                max(int(round(float(W) * resize)), 1),
                max(int(round(float(H) * resize)), 1),
            )
            newW, newH = resize_dims
            crop_h = max(int(newH - fH), 0)
            crop_w = int(np.floor((float(newW) - float(fW)) / 2.0))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            if self.is_train:
                flip = self.data_config['flip'] and np.random.choice([0, 1])
                rotate = np.random.uniform(*self.data_config['rot'])
                if self.data_config.get('vflip', False) and np.random.choice(
                    [0, 1]):
                    rotate += 180
            else:
                flip = False if flip is None else flip
                rotate = 0
            return resize, resize_dims, crop, flip, rotate

        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            random_crop_height = self.data_config.get('random_crop_height',
                                                      False)
            if random_crop_height:
                crop_h = int(
                    np.random.uniform(max(0.3 * newH, newH - fH), newH - fH))
            else:
                crop_h = int(
                    (1 - np.random.uniform(*self.data_config['crop_h'])) *
                    newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
            if self.data_config.get('vflip', False) and np.random.choice(
                [0, 1]):
                rotate += 180
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor_transforms(self, cam_info, cam_name):
        # get sensor2ego
        sensor2ego = transform_matrix(
            translation=cam_info['cams'][cam_name]['sensor2ego_translation'],
            rotation=Quaternion(
                cam_info['cams'][cam_name]['sensor2ego_rotation']))
        sensor2ego = torch.from_numpy(sensor2ego).to(torch.float32)

        ego2sensor = transform_matrix(
            translation=cam_info['cams'][cam_name]['sensor2ego_translation'],
            rotation=Quaternion(
                cam_info['cams'][cam_name]['sensor2ego_rotation']),
            inverse=True)
        ego2sensor = torch.from_numpy(ego2sensor).to(torch.float32)

        # get sensorego2global
        ego2global = transform_matrix(
            translation=cam_info['cams'][cam_name]['ego2global_translation'],
            rotation=Quaternion(
                cam_info['cams'][cam_name]['ego2global_rotation']))
        ego2global = torch.from_numpy(ego2global).to(torch.float32)

        global2ego = transform_matrix(
            translation=cam_info['cams'][cam_name]['ego2global_translation'],
            rotation=Quaternion(
                cam_info['cams'][cam_name]['ego2global_rotation']),
            inverse=True)
        global2ego = torch.from_numpy(global2ego).to(torch.float32)

        return sensor2ego, ego2global, ego2sensor, global2ego

    def get_lidar_transformation(self, results):
        # get lidar2ego
        lidar2lidarego = transform_matrix(
            translation=results['curr']['lidar2ego_translation'],
            rotation=Quaternion(results['curr']['lidar2ego_rotation']),
        )
        lidar2lidarego = torch.from_numpy(lidar2lidarego).to(torch.float32)

        # get ego2lidar
        lidarego2lidar = transform_matrix(
            translation=results['curr']['lidar2ego_translation'],
            rotation=Quaternion(results['curr']['lidar2ego_rotation']),
            inverse=True)
        lidarego2lidar = torch.from_numpy(lidarego2lidar).to(torch.float32)

        # get ego2global
        lidarego2global = transform_matrix(
            translation=results['curr']['ego2global_translation'],
            rotation=Quaternion(results['curr']['ego2global_rotation']),
        )
        lidarego2global = torch.from_numpy(lidarego2global).to(torch.float32)

        return lidar2lidarego, lidarego2lidar, lidarego2global

    def photo_metric_distortion(self, img, pmd):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        if np.random.rand() > pmd.get('rate', 1.0):
            return img

        img = np.array(img).astype(np.float32)
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,' \
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        # random brightness
        if np.random.randint(2):
            delta = np.random.uniform(-pmd['brightness_delta'],
                                      pmd['brightness_delta'])
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            if np.random.randint(2):
                alpha = np.random.uniform(pmd['contrast_lower'],
                                          pmd['contrast_upper'])
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if np.random.randint(2):
            img[..., 1] *= np.random.uniform(pmd['saturation_lower'],
                                             pmd['saturation_upper'])

        # random hue
        if np.random.randint(2):
            img[..., 0] += np.random.uniform(-pmd['hue_delta'],
                                             pmd['hue_delta'])
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if np.random.randint(2):
                alpha = np.random.uniform(pmd['contrast_lower'],
                                          pmd['contrast_upper'])
                img *= alpha

        # randomly swap channels
        if np.random.randint(2):
            img = img[..., np.random.permutation(3)]
        return Image.fromarray(img.astype(np.uint8))

    def get_inputs(self, results, flip=None, scale=None):
        # get cam_names
        cam_names = self.data_config['cams']

        # get store list
        imgs = []
        (sensor2egos, ego2globals, ego2sensors, global2egos, cam2imgs, post_augs,
         lidar2imgs, ego2lidars)  =\
            [], [], [], [], [], [], [], []

        # get lidar-related transformation
        lidar2lidarego, lidarego2lidar, lidarego2global = self.get_lidar_transformation(
            results)

        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # get cam-related transformation
            cam2img = torch.eye(4)
            cam2img[:3, :3] = torch.tensor(
                cam_data['cam_intrinsic'][:3, :3], dtype=torch.float32)
            sensor2ego, ego2global, ego2sensor, global2ego = self.get_sensor_transforms(
                results['curr'], cam_name)

            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2, invalid_index = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 4x4
            post_aug = torch.eye(4)
            post_aug[:2, :2] = post_rot2
            post_aug[:2, 2] = post_tran2

            # get lidar2img
            lidar2img = cam2img @ ego2sensor @ global2ego @ lidarego2global @ lidar2lidarego
            lidar2img = post_aug @ lidar2img

            if self.is_train and self.data_config.get('pmd', None) is not None:
                img = self.photo_metric_distortion(img,
                                                   self.data_config['pmd'])

            imgs.append(self.normalize_img(img))

            # adjacent frame use the same aug with current frame
            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    if self.opencv_pp:
                        img_adjacent = \
                            self.img_transform_core_opencv(
                                img_adjacent,
                                post_rot[:2, :2],
                                post_tran[:2],
                                crop)
                    else:
                        img_adjacent = self.img_transform_core(
                            img_adjacent,
                            resize_dims=resize_dims,
                            crop=crop,
                            flip=flip,
                            rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))

            cam2imgs.append(cam2img)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            ego2sensors.append(ego2sensor)
            global2egos.append(global2ego)
            post_augs.append(post_aug)
            lidar2imgs.append(lidar2img)
        ego2lidars.append(lidarego2lidar)

        if self.sequential:
            for adj_info in results['adjacent']:
                # for convenience
                cam2imgs.extend(cam2imgs[:len(cam_names)])
                post_augs.extend(post_augs[:len(cam_names)])

                # align
                for cam_name in cam_names:
                    sensor2ego, ego2global, ego2sensor, global2ego = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)
                    ego2sensors.append(ego2sensor)
                    global2egos.append(global2ego)

        imgs = torch.stack(imgs)
        # sensor2egos and ego2globals containes current and adjacent frame information
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        ego2sensors = torch.stack(ego2sensors)
        global2egos = torch.stack(global2egos)
        # cam2imgs and post_augs only contain current frame information
        cam2imgs = torch.stack(cam2imgs)
        post_augs = torch.stack(post_augs)
        # lidar2imgs and ego2lidars only contain current frame information
        lidar2imgs = torch.stack(lidar2imgs)
        ego2lidars = torch.stack(ego2lidars)

        # store
        results['cam_names'] = cam_names
        results['sensor2sensorego'] = sensor2egos
        results['sensorego2global'] = ego2globals
        results['sensorego2sensor'] = ego2sensors
        results['global2sensorego'] = global2egos
        results['lidar2img'] = lidar2imgs
        results['ego2lidar'] = ego2lidars

        return (imgs, sensor2egos, ego2globals, cam2imgs, post_augs)

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadAnnotations(object):

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']
        gt_boxes = np.array(gt_boxes)
        gt_labels = np.array(gt_labels)
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = LiDARInstance3DBoxes(
            gt_boxes, box_dim=gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels
        return results


@PIPELINES.register_module()
class BEVAug(object):

    def __init__(self,
                 bda_aug_conf,
                 classes,
                 is_train=True,
                 transform_boxes=False):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes
        self.empty_idx = len(classes) - 1
        self.transform_boxes = bool(transform_boxes)

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
            translation_std = self.bda_aug_conf.get('tran_lim',
                                                    [0.0, 0.0, 0.0])
            tran_bda = np.random.normal(scale=translation_std, size=3).T
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
            tran_bda = np.zeros((1, 3), dtype=np.float32)
        return rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda

    def bev_transform(self, rotate_angle, scale_ratio, flip_dx, flip_dy,
                      tran_bda):
        # get rotation matrix
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])

        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        return rot_mat

    def voxel_transform(self, results, rotate_bda, scale_bda, flip_dx, flip_dy,
                        bda_mat):
        flow_keys = (
            'voxel_occflows',
            'voxel_occflows_1_2',
            'voxel_occflows_1_4',
            'voxel_occflows_1_8',
        )
        apply_dense_bda = (
            abs(float(rotate_bda)) > 1e-4
            or abs(float(scale_bda) - 1.0) > 1e-4)

        if flip_dx and not apply_dense_bda:
            results['voxel_semantic'] = results['voxel_semantic'][::-1,
                                                                  ...].copy()
            if 'voxel_semantic_1_2' in results:
                results['voxel_semantic_1_2'] = results[
                    'voxel_semantic_1_2'][::-1, ...].copy()
            if 'voxel_semantic_1_4' in results:
                results['voxel_semantic_1_4'] = results[
                    'voxel_semantic_1_4'][::-1, ...].copy()
            if 'voxel_semantic_1_8' in results:
                results['voxel_semantic_1_8'] = results[
                    'voxel_semantic_1_8'][::-1, ...].copy()
            for flow_key in flow_keys:
                if flow_key not in results:
                    continue
                results[flow_key] = results[flow_key][::-1, ...].copy()
                results[flow_key][..., 0] = -results[flow_key][..., 0]
                results[flow_key][..., 0][results[flow_key][...,
                                                            0] == -255] = 255

        if flip_dy and not apply_dense_bda:
            results['voxel_semantic'] = results['voxel_semantic'][:, ::-1,
                                                                  ...].copy()
            if 'voxel_semantic_1_2' in results:
                results['voxel_semantic_1_2'] = results[
                    'voxel_semantic_1_2'][:, ::-1, ...].copy()
            if 'voxel_semantic_1_4' in results:
                results['voxel_semantic_1_4'] = results[
                    'voxel_semantic_1_4'][:, ::-1, ...].copy()
            if 'voxel_semantic_1_8' in results:
                results['voxel_semantic_1_8'] = results[
                    'voxel_semantic_1_8'][:, ::-1, ...].copy()
            for flow_key in flow_keys:
                if flow_key not in results:
                    continue
                results[flow_key] = results[flow_key][:, ::-1, ...].copy()
                results[flow_key][..., 1] = -results[flow_key][..., 1]
                results[flow_key][..., 1][results[flow_key][...,
                                                            1] == -255] = 255

        if not apply_dense_bda:
            return results

        semantic_keys = (
            'voxel_semantic',
            'voxel_semantic_1_2',
            'voxel_semantic_1_4',
            'voxel_semantic_1_8',
        )
        for semantic_key in semantic_keys:
            if semantic_key not in results:
                continue
            semantic = torch.from_numpy(
                np.asarray(results[semantic_key],
                           dtype=np.float32)).permute(2, 1, 0)
            semantic = _warp_bev_tensor_by_bda(
                semantic, bda_mat, fill_value=self.empty_idx)
            results[semantic_key] = semantic.permute(
                2, 1, 0).cpu().numpy().astype(np.uint8)

        for flow_key in flow_keys:
            if flow_key not in results:
                continue
            results[flow_key] = _warp_flow_by_bda(results[flow_key], bda_mat)

        return results

    def box_transform(self, results, bda_mat):
        """Apply the same BEV affine transform to detection boxes.

        Occupancy labels are transformed in ``voxel_transform``.  Detection
        supervision must follow that transform as well, otherwise the new
        CenterHead would receive boxes in the unaugmented coordinate frame.
        """
        boxes = results.get('gt_bboxes_3d')
        if boxes is None or len(boxes) == 0:
            return results

        bda_mat = bda_mat.to(
            dtype=boxes.tensor.dtype, device=boxes.tensor.device)
        affine = bda_mat[:3, :3]
        translation = bda_mat[:3, 3]
        tensor = boxes.tensor.clone()
        tensor[:, :3] = tensor[:, :3] @ affine.t() + translation

        # BEVDet's BDA uses a uniform scale. Preserve the box dimensions under
        # that scale while keeping height and yaw in the box convention.
        scale = torch.linalg.norm(affine[0, :2])
        tensor[:, 3:6] = tensor[:, 3:6] * scale
        if tensor.shape[1] >= 7:
            yaw_vector = torch.stack(
                [torch.cos(tensor[:, 6]),
                 torch.sin(tensor[:, 6])], dim=1)
            yaw_vector = yaw_vector @ affine[:2, :2].t()
            tensor[:, 6] = torch.atan2(yaw_vector[:, 1], yaw_vector[:, 0])
        if tensor.shape[1] >= 9:
            tensor[:, 7:9] = tensor[:, 7:9] @ affine[:2, :2].t()
        results['gt_bboxes_3d'] = boxes.new_box(tensor)
        return results

    def __call__(self, results):
        # sample bda augmentation
        rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda = self.sample_bda_augmentation(
        )
        if 'bda_aug' in results:
            flip_dx, flip_dy = results['bda_aug']['flip_dx'], results[
                'bda_aug']['flip_dy']

        # get bda matrix
        bda_rot = self.bev_transform(rotate_bda, scale_bda, flip_dx, flip_dy,
                                     tran_bda)
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        bda_mat[:3, :3] = bda_rot
        bda_mat[:3, 3] = torch.from_numpy(tran_bda)

        # do voxel transformation
        results = self.voxel_transform(
            results,
            rotate_bda=rotate_bda,
            scale_bda=scale_bda,
            flip_dx=flip_dx,
            flip_dy=flip_dy,
            bda_mat=bda_mat.unsqueeze(0))

        if self.transform_boxes:
            results = self.box_transform(results, bda_mat)

        if 'img_inputs' in results:
            imgs, sensor2egos, ego2globals, cam2imgs, post_augs = results[
                'img_inputs']
            results['img_inputs'] = (imgs, sensor2egos, ego2globals, cam2imgs,
                                     post_augs, bda_mat)
        else:
            results['bda_mat'] = bda_mat

        return results
