"""Shared occupancy-flow dataset pipeline components for RoadOcc-style experiments."""

import os

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.core.points import get_points_type
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines.formating import DefaultFormatBundle3D
from mmdet3d.datasets.pipelines.loading import (BEVAug, LoadOccGTFromFile,
                                                LoadPointsFromFile,
                                                _maybe_reanchor_occ_flow,
                                                _maybe_reanchor_occ_labels)
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from torchvision.transforms.functional import rotate

class MultiFramePointsFromFiles(LoadPointsFromFile):
    """Load current/adjacent LiDAR frames and align them to occupancy coordinates."""

    def __init__(self,
                 num_adj_frames=None,
                 sweeps_num=1,
                 filter_zero_points=True,
                 normalize_fourth_channel=True,
                 fourth_channel_log_max=8192.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_adj_frames = num_adj_frames
        self.sweeps_num = max(int(sweeps_num), 1)
        self.filter_zero_points = filter_zero_points
        self.normalize_fourth_channel = normalize_fourth_channel
        self.fourth_channel_log_max = float(fourth_channel_log_max)

    def _build_points(self, points):
        attribute_dims = None
        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1)
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
        return points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)

    def _load_single_frame_points(self, pts_filename):
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        if self.filter_zero_points and points.shape[1] >= 3:
            points = points[np.any(points[:, :3] != 0, axis=1)]
        if self.normalize_fourth_channel and points.shape[1] >= 4:
            points = points.copy()
            denom = np.log1p(self.fourth_channel_log_max)
            if denom > 0:
                points[:, 3] = np.log1p(
                    np.clip(points[:, 3], a_min=0, a_max=None)) / denom
        return points

    def _transform_to_current_lidar(self, points, src_info, dst_info):
        if self._is_same_frame(src_info, dst_info):
            return points
        point_aug = points.copy()
        from mmdet3d.datasets.utils import nuscenes_get_rt_matrix
        lidar_to_lidar = nuscenes_get_rt_matrix(
            src_info, dst_info, 'lidar', 'lidar')
        xyz1 = np.concatenate(
            [point_aug[:, :3],
             np.ones((point_aug.shape[0], 1), dtype=point_aug.dtype)],
            axis=1)
        point_aug[:, :3] = (xyz1 @ lidar_to_lidar.T)[:, :3]
        return point_aug

    @staticmethod
    def _append_time_lag(points, time_lag):
        time_column = np.full((points.shape[0], 1), time_lag, dtype=points.dtype)
        return np.concatenate([points, time_column], axis=1)

    @staticmethod
    def _is_same_frame(src_info, dst_info):
        if src_info is dst_info:
            return True
        src_token = src_info.get('token') if src_info is not None else None
        dst_token = dst_info.get('token') if dst_info is not None else None
        if src_token and dst_token:
            return src_token == dst_token
        return src_info.get('lidar_path') == dst_info.get('lidar_path')

    @staticmethod
    def _get_lidar_path(info, default_path=None):
        if info is None:
            return default_path
        return info.get('lidar_path', info.get('data_path', default_path))

    @staticmethod
    def _get_timestamp_seconds(info, default_timestamp=None):
        if info is None or 'timestamp' not in info:
            return float(default_timestamp)
        return float(info['timestamp']) / 1e6

    @staticmethod
    def _build_current_ego2lidar(current_info):
        ego2lidar = transform_matrix(
            translation=current_info['lidar2ego_translation'],
            rotation=Quaternion(current_info['lidar2ego_rotation']),
            inverse=True)
        return torch.from_numpy(ego2lidar).to(torch.float32).unsqueeze(0)

    @staticmethod
    def _extract_bda_mat(results):
        bda_mat = results.get('bda_mat')
        if bda_mat is None:
            img_inputs = results.get('img_inputs')
            if (isinstance(img_inputs, (list, tuple)) and img_inputs and
                    torch.is_tensor(img_inputs[-1])):
                bda_mat = img_inputs[-1]
        if bda_mat is None:
            return None
        if bda_mat.dim() == 3:
            bda_mat = bda_mat[0]
        return bda_mat.to(torch.float32)

    @staticmethod
    def _transform_points_to_occ(points, ego2lidar, bda_mat):
        if ego2lidar is None:
            return points
        lidar2ego = torch.inverse(ego2lidar).cpu().numpy()
        point_aug = points.copy()
        xyz1 = np.concatenate(
            [point_aug[:, :3],
             np.ones((point_aug.shape[0], 1), dtype=point_aug.dtype)],
            axis=1)
        point_aug[:, :3] = (xyz1 @ lidar2ego.T)[:, :3]
        if bda_mat is not None:
            bda_np = bda_mat.cpu().numpy()
            xyz1_occ = np.concatenate(
                [point_aug[:, :3],
                 np.ones((point_aug.shape[0], 1), dtype=point_aug.dtype)],
                axis=1)
            point_aug[:, :3] = (xyz1_occ @ bda_np.T)[:, :3]
        return point_aug

    def _transform_sweep_to_frame(self, points, sweep_info, frame_info):
        if ('sensor2lidar_rotation' in sweep_info and
                'sensor2lidar_translation' in sweep_info):
            point_aug = points.copy()
            point_aug[:, :3] = point_aug[:, :3] @ np.asarray(
                sweep_info['sensor2lidar_rotation']).T
            point_aug[:, :3] += np.asarray(sweep_info['sensor2lidar_translation'])
            return point_aug
        return self._transform_to_current_lidar(points, sweep_info, frame_info)

    def _collect_sweep_infos(self, frame_info, fallback_infos):
        required_history = max(self.sweeps_num - 1, 0)
        if required_history == 0:
            return []
        sweep_infos = list(frame_info.get('sweeps', []))[:required_history]
        if len(sweep_infos) < required_history:
            sweep_infos.extend(
                fallback_infos[:required_history - len(sweep_infos)])
        return sweep_infos

    def _build_frame_points(self,
                            base_points,
                            frame_info,
                            current_info,
                            current_timestamp,
                            fallback_infos):
        frame_timestamp = self._get_timestamp_seconds(
            frame_info, default_timestamp=current_timestamp)
        frame_points = [
            self._append_time_lag(base_points, current_timestamp - frame_timestamp)
        ]
        for sweep_info in self._collect_sweep_infos(frame_info, fallback_infos):
            sweep_path = self._get_lidar_path(sweep_info)
            if sweep_path is None:
                continue
            sweep_points = self._load_single_frame_points(sweep_path)
            sweep_points = self._transform_sweep_to_frame(
                sweep_points, sweep_info, frame_info)
            sweep_timestamp = self._get_timestamp_seconds(
                sweep_info, default_timestamp=frame_timestamp)
            frame_points.append(self._append_time_lag(
                sweep_points, current_timestamp - sweep_timestamp))
        merged_frame_points = np.concatenate(frame_points, axis=0)
        if not self._is_same_frame(frame_info, current_info):
            merged_frame_points = self._transform_to_current_lidar(
                merged_frame_points, frame_info, current_info)
        return merged_frame_points

    def __call__(self, results):
        current_info = results['curr']
        current_points = self._load_single_frame_points(results['pts_filename'])
        current_timestamp = float(results['timestamp'])
        adjacent_infos = results.get('adjacent', [])
        output_adjacent_infos = adjacent_infos
        if self.num_adj_frames is not None:
            output_adjacent_infos = adjacent_infos[:self.num_adj_frames]
        frame_infos = [current_info] + output_adjacent_infos
        frame_points = []
        for frame_index, frame_info in enumerate(frame_infos):
            if frame_index == 0:
                base_points = current_points
            else:
                base_points = self._load_single_frame_points(
                    self._get_lidar_path(frame_info))
            fallback_infos = adjacent_infos[frame_index:]
            frame_points.append(self._build_frame_points(
                base_points, frame_info, current_info, current_timestamp,
                fallback_infos))
        ego2lidar = self._build_current_ego2lidar(current_info)
        bda_mat = self._extract_bda_mat(results)
        merged_points = np.concatenate(frame_points, axis=0)
        merged_points = self._transform_points_to_occ(
            merged_points, ego2lidar[0], bda_mat)
        results['points_for_depth'] = self._build_points(current_points)
        results['points_frame_splits'] = [
            int(points.shape[0]) for points in frame_points]
        results['points'] = self._build_points(merged_points)
        results['ego2lidar'] = ego2lidar
        return results



@PIPELINES.register_module()
class LoadMultiFramePointsFromFiles(MultiFramePointsFromFiles):
    """Compatibility alias used by RoadOcc configs."""


@PIPELINES.register_module()
class ALOccLoadMultiFramePointsFromFiles(MultiFramePointsFromFiles):
    """Compatibility alias used by ALOcc configs."""


@PIPELINES.register_module()
class LetOccFlowLoadMultiFramePointsFromFiles(MultiFramePointsFromFiles):
    """Compatibility alias used by LetOccFlow configs."""


@PIPELINES.register_module()
class CRTFusionLoadMultiFramePointsFromFiles(MultiFramePointsFromFiles):
    """Compatibility alias used by CRTFusion configs."""

@PIPELINES.register_module()
class LidarBEVAug(BEVAug):
    def __call__(self, results):
        rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda = self.sample_bda_augmentation()
        bda_rot = self.bev_transform(rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda)
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        bda_mat[:3, :3] = bda_rot
        bda_mat[:3, 3] = torch.from_numpy(tran_bda)
        results = self.voxel_transform(results, flip_dx=flip_dx, flip_dy=flip_dy)
        results['bda_mat'] = bda_mat
        return results

def _get_bda_mat_from_results(results, device, dtype):
    bda_mat = results.get('bda_mat')
    if bda_mat is None:
        img_inputs = results.get('img_inputs')
        if isinstance(img_inputs, (list, tuple)) and len(img_inputs) > 0 and torch.is_tensor(img_inputs[-1]):
            bda_mat = img_inputs[-1]
    if bda_mat is None:
        return torch.eye(4, device=device, dtype=dtype)
    bda_mat = bda_mat.to(device=device, dtype=dtype)
    if bda_mat.dim() == 3:
        bda_mat = bda_mat[0]
    return bda_mat


def _get_ego2lidar_from_results(results, device, dtype):
    ego2lidar = results.get('ego2lidar')
    if ego2lidar is None:
        return None
    ego2lidar = ego2lidar.to(device=device, dtype=dtype)
    if ego2lidar.dim() == 3:
        ego2lidar = ego2lidar[0]
    return ego2lidar


@PIPELINES.register_module()
class PreparePointsForOcc(object):
    def __call__(self, results):
        points = results.get('points')
        if points is None:
            return results

        points_tensor = points.tensor
        if 'points_for_depth' not in results:
            results['points_for_depth'] = points.new_point(points_tensor.clone())

        ego2lidar = _get_ego2lidar_from_results(
            results, points_tensor.device, points_tensor.dtype)
        if ego2lidar is None:
            return results

        lidar2ego = torch.inverse(ego2lidar)
        bda_mat = _get_bda_mat_from_results(
            results, points_tensor.device, points_tensor.dtype)

        xyz1 = torch.cat(
            [points_tensor[:, :3], torch.ones_like(points_tensor[:, :1])], dim=1)
        points_occ = points_tensor.clone()
        points_occ[:, :3] = (xyz1 @ lidar2ego.t())[:, :3]

        xyz1_occ = torch.cat(
            [points_occ[:, :3], torch.ones_like(points_occ[:, :1])], dim=1)
        points_occ[:, :3] = (xyz1_occ @ bda_mat.t())[:, :3]
        results['points'] = points.new_point(points_occ)
        return results

@PIPELINES.register_module()
class ALOccImageInputAdapter:
    """Expose RoadOcc camera tensors through ALOcc's original input contract."""

    def __init__(self, num_cams, strict_current_frame=False):
        self.num_cams = int(num_cams)
        self.strict_current_frame = bool(strict_current_frame)

    def __call__(self, results):
        raw_inputs = results['img_inputs']
        if len(raw_inputs) != 6:
            raise ValueError('ALOccImageInputAdapter expects RoadOcc inputs with BDA.')
        imgs, sensor2egos, ego2globals, cam2imgs, post_augs, bda_mat = raw_inputs
        expected_images = self.num_cams * (
            1 if self.strict_current_frame else 2)
        if imgs.shape[0] != expected_images:
            raise ValueError(
                f'ALOcc adapter expects {expected_images} camera-major '
                f'images, got {imgs.shape[0]}.')
        if cam2imgs.shape[0] < self.num_cams:
            raise ValueError(
                f'Expected {self.num_cams} current cameras, got {cam2imgs.shape[0]}.')
        if (not self.strict_current_frame and
                sensor2egos.shape[0] < self.num_cams * 2):
            raise ValueError('ALOCC stereo mode requires one adjacent frame per camera.')
        if 'gt_depth' in results and results['gt_depth'].shape[0] != self.num_cams:
            raise ValueError(
                'ALOCC depth supervision must provide one current-frame map '
                f'per camera, expected {self.num_cams} but got '
                f'{results["gt_depth"].shape[0]}.')

        # Keep the same coordinate convention as RoadOcc's
        # BEVDetStereoForwardProjection: camera extrinsics are expressed in
        # the current key-frame ego coordinate.  For the current frame this is
        # usually identical to raw sensor2ego, but the explicit computation
        # makes the adapter robust to any per-camera ego pose differences.
        keyego2global = ego2globals[:self.num_cams, ...][0:1]
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = (
            global2keyego @ ego2globals[:self.num_cams].double()
            @ sensor2egos[:self.num_cams].double()).float()

        rots = sensor2keyegos[:, :3, :3].contiguous()
        trans = sensor2keyegos[:, :3, 3].contiguous()
        intrins = cam2imgs[:self.num_cams, :3, :3].contiguous()
        post_rots = post_augs[:self.num_cams, :3, :3].contiguous()
        # RoadOcc/BEVDet stores image-space augmentation as a 3x3 homogeneous
        # matrix embedded in the top-left of the 4x4 ``post_aug`` tensor.  The
        # 2D translation is therefore column 2, not the 4x4 transform column 3.
        post_trans = torch.zeros_like(post_augs[:self.num_cams, :3, 3])
        post_trans[:, :2] = post_augs[:self.num_cams, :2, 2]
        post_trans = post_trans.contiguous()
        results['img_inputs'] = (
            imgs, rots, trans, intrins, post_rots, post_trans, bda_mat[:3, :3])
        if self.strict_current_frame:
            results.pop('aux_cam_params', None)
            results.pop('adj_aux_cam_params', None)
        else:
            results['aux_cam_params'] = (
                sensor2egos[:self.num_cams].contiguous(),
                ego2globals[:self.num_cams].contiguous())
            results['adj_aux_cam_params'] = (
                sensor2egos[self.num_cams:self.num_cams * 2].contiguous(),
                ego2globals[self.num_cams:self.num_cams * 2].contiguous())
        return results


@PIPELINES.register_module()
class ALOccBEVAug:
    """The BDA wrapper used by the official ALOcc pipeline.

    :class:`ALOccLoadOccupancy` applies the sampled transform to V2X-Real
    targets after loading them in their physical ``[x, y, z]`` layout.
    """

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.classes = classes
        self.is_train = is_train

    def _sample(self):
        if self.is_train:
            return (
                np.random.uniform(*self.bda_aug_conf['rot_lim']),
                np.random.uniform(*self.bda_aug_conf['scale_lim']),
                np.random.uniform() < self.bda_aug_conf['flip_dx_ratio'],
                np.random.uniform() < self.bda_aug_conf['flip_dy_ratio'])
        return 0.0, 1.0, False, False

    @staticmethod
    def _matrix(rotate_angle, scale_ratio, flip_dx, flip_dy):
        angle = torch.tensor(rotate_angle / 180.0 * np.pi)
        rot = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        scale = torch.diag(torch.tensor([scale_ratio, scale_ratio, scale_ratio]))
        flip = torch.eye(3)
        if flip_dx:
            flip = flip @ torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
        if flip_dy:
            flip = flip @ torch.diag(torch.tensor([1.0, -1.0, 1.0]))
        return flip @ (scale @ rot)

    def __call__(self, results):
        rotate_bda, scale_bda, flip_dx, flip_dy = self._sample()
        if 'bda_aug' in results:
            flip_dx = bool(results['bda_aug']['flip_dx'])
            flip_dy = bool(results['bda_aug']['flip_dy'])
        bda_rot = self._matrix(rotate_bda, scale_bda, flip_dx, flip_dy)
        bda_mat = torch.eye(4, dtype=bda_rot.dtype)
        bda_mat[:3, :3] = bda_rot
        img_inputs = results['img_inputs']
        if len(img_inputs) != 5:
            raise ValueError('ALOccBEVAug must run before ALOccImageInputAdapter.')
        results['img_inputs'] = (*img_inputs, bda_mat)
        # Match RoadOcc's BEVAug side channel.  Several shared helpers first
        # look for ``results['bda_mat']`` and only then fall back to
        # ``img_inputs[-1]``; exposing it here keeps points/depth/diagnostics
        # on the exact same BEV augmentation contract.
        results['bda_mat'] = bda_mat
        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy
        results['rotate_bda'] = rotate_bda
        results['scale_bda'] = scale_bda
        return results


@PIPELINES.register_module()
class ALOccLoadOccupancy:
    """Load V2X-Real targets in the physical view-transformer coordinates."""

    def __init__(self,
                 flow_gt_path,
                 load_flow=True,
                 ignore_nonvisible=True,
                 fix_void=False,
                 flow_mask=False,
                 flow_class_indices=None,
                 dynamic_class_indices=None,
                 mask='mask_camera',
                 point_cloud_range=None,
                 downsample_factor=1):
        self.flow_gt_path = flow_gt_path
        self.load_flow = bool(load_flow)
        self.ignore_nonvisible = bool(ignore_nonvisible)
        self.fix_void = bool(fix_void)
        self.flow_mask = bool(flow_mask)
        self.flow_class_indices = (
            None if flow_class_indices is None else
            tuple(int(index) for index in flow_class_indices))
        self.dynamic_class_indices = tuple(
            int(index) for index in (
                dynamic_class_indices
                if dynamic_class_indices is not None
                else (self.flow_class_indices or ())))
        self.mask = mask
        self.point_cloud_range = (
            tuple(float(value) for value in point_cloud_range)
            if point_cloud_range is not None else None)
        self.downsample_factor = int(downsample_factor)
        if self.downsample_factor < 1:
            raise ValueError('downsample_factor must be >= 1.')

    def _flow_path(self, occ_path):
        parts = os.path.normpath(os.path.abspath(occ_path)).split(os.sep)
        if 'gts' in parts:
            relative = os.path.join(*parts[parts.index('gts') + 1:])
        else:
            relative = os.path.basename(occ_path)
        return os.path.join(os.path.abspath(self.flow_gt_path), relative, 'labels.npz')

    @staticmethod
    def _to_alocc(occupancy):
        # V2X-Real stores ``semantics[x, y, z]`` in the same physical BEV
        # convention as ALOcc's [X, Y, Z] view-transformer output. The
        # released ALOcc rotation/flip is specific to OpenOcc/FBOcc labels.
        return occupancy.clone().contiguous()

    @staticmethod
    def _flow_to_alocc(flow):
        # Flow components are the V2X-Real physical (x, y) displacement.
        return flow.clone().contiguous()

    @staticmethod
    def _rotate_occupancy(occupancy, angle, fill):
        return rotate(occupancy.permute(2, 0, 1), -float(angle), fill=fill).permute(1, 2, 0)

    def _downsample_semantics(self, semantics, empty_idx):
        return LoadOccGTFromFile._downsample_semantics(
            semantics,
            downscale=self.downsample_factor,
            empty_cls_idx=empty_idx,
            priority_class_indices=self.dynamic_class_indices)

    def _downsample_mask(self, mask):
        return LoadOccGTFromFile._downsample_mask(
            mask.astype(np.uint8, copy=False),
            downscale=self.downsample_factor).astype(bool)

    def _downsample_flow(self, flow, semantics):
        return LoadOccGTFromFile._downsample_flow(
            flow,
            semantics,
            downscale=self.downsample_factor,
            dynamic_class_indices=self.dynamic_class_indices).astype(
                np.float32, copy=False)

    def __call__(self, results):
        label_path = os.path.join(results['occ_gt_path'], 'labels.npz')
        labels = np.load(label_path)
        raw_occupancy_np = labels['semantics'].copy()
        occupancy_np = raw_occupancy_np.copy()
        visible_mask_np = labels[self.mask].copy().astype(bool)
        occupancy_original_np = occupancy_np.copy()
        if self.ignore_nonvisible:
            occupancy_np[~visible_mask_np] = 255

        if self.downsample_factor != 1:
            valid_for_empty = raw_occupancy_np[raw_occupancy_np != 255]
            empty_idx_for_downsample = (
                int(valid_for_empty.max()) if valid_for_empty.size else 255)
            occupancy_np = self._downsample_semantics(
                occupancy_np, empty_idx_for_downsample)
            occupancy_original_np = self._downsample_semantics(
                occupancy_original_np, empty_idx_for_downsample)
            visible_mask_np = self._downsample_mask(visible_mask_np)

        # Match RoadOcc's optional ego-frame reanchoring path.  The current
        # ALOcc config does not insert such a transform, so this is a no-op for
        # the normal bs=2 run, but it keeps the adapter correct when sharing
        # RoadOcc robustness/data-augmentation pipelines.
        valid = occupancy_np[occupancy_np != 255]
        empty_idx = int(valid.max()) if valid.size else 255
        valid_original = occupancy_original_np[occupancy_original_np != 255]
        empty_idx_original = (
            int(valid_original.max()) if valid_original.size else empty_idx)
        temp_mask_key = '_alocc_visible_mask'
        results[temp_mask_key] = visible_mask_np
        occupancy_np = _maybe_reanchor_occ_labels(
            results,
            occupancy_np,
            empty_idx=empty_idx,
            point_cloud_range=self.point_cloud_range,
            mask_key=temp_mask_key)
        visible_mask_np = np.asarray(
            results.pop(temp_mask_key, visible_mask_np)).astype(bool)
        occupancy_original_np = _maybe_reanchor_occ_labels(
            results,
            occupancy_original_np,
            empty_idx=empty_idx_original,
            point_cloud_range=self.point_cloud_range)

        occupancy = torch.from_numpy(occupancy_np.copy()).long()
        visible_mask = torch.from_numpy(visible_mask_np.copy()).bool()
        occupancy_original = torch.from_numpy(
            occupancy_original_np.copy()).long()

        flow = None
        if self.load_flow:
            flow_path = self._flow_path(results['occ_gt_path'])
            with np.load(flow_path) as flow_labels:
                if 'flow' not in flow_labels:
                    raise KeyError(f'flow is missing from {flow_path}')
                flow_np = flow_labels['flow'].astype(np.float32, copy=True)
                if self.downsample_factor != 1:
                    flow_np = self._downsample_flow(
                        flow_np, raw_occupancy_np)
                flow_np = _maybe_reanchor_occ_flow(
                    results,
                    flow_np,
                    point_cloud_range=self.point_cloud_range)
                flow = torch.from_numpy(flow_np.copy())

        occupancy = self._to_alocc(occupancy)
        occupancy_original = self._to_alocc(occupancy_original)
        visible_mask = self._to_alocc(visible_mask.to(torch.uint8)).bool()
        if flow is not None:
            flow = self._flow_to_alocc(flow)
            if self.flow_mask:
                if self.flow_class_indices is None:
                    # Preserve the upstream OpenOcc convention for source
                    # configurations that do not provide an explicit class
                    # mapping.
                    flow_mask = occupancy < 8
                else:
                    # OpenOcc's ``occupancy < 8`` convention is tied to its
                    # label ordering. V2X-Real puts truck at index 10, so an
                    # explicit shared dynamic set is required to supervise
                    # every evaluated moving class and no static zero flow.
                    flow_mask = torch.zeros_like(occupancy, dtype=torch.bool)
                    for class_index in self.flow_class_indices:
                        flow_mask |= occupancy == class_index
                flow[~flow_mask] = float('inf')

        if self.fix_void:
            occupancy[occupancy < 255] += 1
            occupancy_original[occupancy_original < 255] += 1

        angle = float(results['rotate_bda'])
        if angle != 0.0:
            occupancy = self._rotate_occupancy(occupancy, angle, 255)
            occupancy_original = self._rotate_occupancy(occupancy_original, angle, 255)
            visible_mask = self._rotate_occupancy(visible_mask.to(torch.uint8), angle, 0).bool()
            if flow is not None:
                flow = rotate(
                    flow.permute(3, 2, 0, 1), -angle,
                    fill=float('inf')).permute(2, 3, 1, 0)

        if results['flip_dx']:
            # V2X-Real labels are [x, y, z], matching the BDA matrix's
            # physical x axis. The upstream ALOcc loader used [y, x, z]
            # labels and therefore flipped the opposite dimension.
            occupancy = torch.flip(occupancy, [0])
            occupancy_original = torch.flip(occupancy_original, [0])
            visible_mask = torch.flip(visible_mask, [0])
            if flow is not None:
                flow = torch.flip(flow, [0])
                flow[..., 0] = -flow[..., 0]
        if results['flip_dy']:
            occupancy = torch.flip(occupancy, [1])
            occupancy_original = torch.flip(occupancy_original, [1])
            visible_mask = torch.flip(visible_mask, [1])
            if flow is not None:
                flow = torch.flip(flow, [1])
                flow[..., 1] = -flow[..., 1]

        results['gt_occupancy'] = occupancy
        results['gt_occupancy_ori'] = occupancy_original
        results['visible_mask'] = visible_mask
        results['visible_mask_bev'] = (occupancy == 255).sum(-1)
        if flow is not None:
            results['gt_occ_flow'] = flow
        return results


@PIPELINES.register_module()
class ALOccFormatBundle3D(DefaultFormatBundle3D):
    """Format the ALOcc target tensors exactly as stackable batch fields."""

    def __call__(self, results):
        results = super().__call__(results)
        for key in ('gt_occupancy', 'gt_occupancy_ori', 'gt_occ_flow', 'gt_depth'):
            if key in results and results[key] is not None:
                results[key] = DC(to_tensor(results[key]), stack=True)
        return results

@PIPELINES.register_module()
class CRTFusionImageInputAdapter:
    """Expose the current RoadOcc camera frame through CRT-Fusion's contract."""

    def __init__(self, num_cams, strict_current_frame=False):
        self.num_cams = int(num_cams)
        self.strict_current_frame = bool(strict_current_frame)

    def __call__(self, results):
        raw_inputs = results['img_inputs']
        if len(raw_inputs) != 6:
            raise ValueError('CRTFusionImageInputAdapter expects RoadOcc BDA inputs.')
        imgs, sensor2egos, ego2globals, cam2imgs, post_augs, bda_mat = raw_inputs
        if imgs.shape[0] % self.num_cams != 0:
            raise ValueError(
                f'Expected a whole temporal group for {self.num_cams} cameras, '
                f'got {imgs.shape[0]} images.')
        queue_length = imgs.shape[0] // self.num_cams
        if queue_length < 1:
            raise ValueError('CRTFusionImageInputAdapter received no camera images.')
        if self.strict_current_frame and queue_length != 1:
            raise ValueError(
                'Strict current-frame CRT-Fusion expects exactly one image '
                f'per camera, got {queue_length} temporal groups.')

        # PrepareImageInputs orders images camera-major as
        # [cam0(t), cam0(t-1), cam1(t), cam1(t-1), ...], while calibration is
        # frame-major. CRT-Fusion's current-frame encoder must therefore take
        # the first image from every camera temporal group.
        current_imgs = imgs.view(self.num_cams, queue_length, *imgs.shape[1:])[:, 0]

        rots = sensor2egos[:self.num_cams, :3, :3].contiguous()
        trans = sensor2egos[:self.num_cams, :3, 3].contiguous()
        intrins = cam2imgs[:self.num_cams, :3, :3].contiguous()
        # ``PrepareImageInputs`` stores its 2D affine transform as a 4x4
        # homogeneous matrix with image translation in column 2.  CRT-Fusion's
        # view transformer follows the original BEVDet contract instead: the
        # linear transform and translation are supplied separately.  Passing
        # the homogeneous matrix as ``post_rots`` makes get_geometry multiply
        # the image translation by depth and badly corrupts lift-splat geometry.
        post_rots = torch.eye(
            3, dtype=post_augs.dtype, device=post_augs.device).unsqueeze(0).repeat(
                self.num_cams, 1, 1)
        post_rots[:, :2, :2] = post_augs[:self.num_cams, :2, :2]
        post_trans = torch.zeros(
            self.num_cams, 3,
            dtype=post_augs.dtype,
            device=post_augs.device)
        post_trans[:, :2] = post_augs[:self.num_cams, :2, 2]
        post_rots = post_rots.contiguous()
        post_trans = post_trans.contiguous()
        bda = bda_mat[:3, :3].contiguous()
        results['img_inputs'] = (
            current_imgs.contiguous(), rots, trans, intrins,
            post_rots, post_trans, bda)

        # CRT-Fusion stores the preceding BEV in the augmented frame. These
        # metadata flags make its original history transform use the same BDA.
        transform_types = []
        if bda[0, 0] < 0:
            results['pcd_vertical_flip'] = True
            transform_types.append('VF')
        if bda[1, 1] < 0:
            results['pcd_horizontal_flip'] = True
            transform_types.append('HF')
        results['transformation_3d_flow'] = transform_types
        return results


@PIPELINES.register_module()
class CRTFusionFormatBundle3D(DefaultFormatBundle3D):
    """Format dense RoadOcc supervision required by the adapted task head."""

    def __call__(self, results):
        results = super().__call__(results)
        for key in ('gt_depth', 'voxel_semantic', 'voxel_occflows'):
            if key in results and results[key] is not None:
                results[key] = DC(to_tensor(results[key]), stack=True)
        return results
