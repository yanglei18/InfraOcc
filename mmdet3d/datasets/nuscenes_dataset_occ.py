# Copyright (c) OpenMMLab. All rights reserved.
import os
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.utils import print_log
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from mmdet3d.datasets.occ_metrics import Metric_mIoU
from mmdet3d.datasets.utils import nuscenes_get_rt_matrix
from .next_frame_metrics import (evaluate_dynamic_motion_state_iou,
                                 evaluate_next_frame_dynamic_iou_samples)
from .nuscenes_ego_pose_loader import nuScenesDataset

occ3d_colors_map = np.array([
    [0, 0, 0, 255],  # 0 undefined
    [255, 158, 0, 255],  # 1 car  orange
    [0, 0, 230, 255],  # 2 pedestrian  Blue
    [47, 79, 79, 255],  # 3 sign  Darkslategrey
    [220, 20, 60, 255],  # 4 CYCLIST  Crimson
    [255, 69, 0, 255],  # 5 traiffic_light  Orangered
    [255, 140, 0, 255],  # 6 pole  Darkorange
    [233, 150, 70, 255],  # 7 construction_cone  Darksalmon
    [255, 61, 99, 255],  # 8 bycycle  Red
    [112, 128, 144, 255],  # 9 motorcycle  Slategrey
    [222, 184, 135, 255],  # 10 building Burlywood
    [0, 175, 0, 255],  # 11 vegetation  Green
    [165, 42, 42, 255],  # 12 trunk  nuTonomy green
    [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
    [75, 0, 75, 255],  # 14 walkable, sidewalk
    [255, 0, 0, 255],  # 15 unobsrvd
    [0, 0, 0, 0],  # 16 undefined
    [0, 0, 0, 0],  # 16 undefined
])

openocc_colors_map = np.array([
    [0, 150, 245],  # car                  blue         √
    [160, 32, 240],  # truck                purple       √
    [135, 60, 0],  # trailer              brown        √
    [255, 255, 0],  # bus                  yellow       √
    [0, 255, 255],  # construction_vehicle cyan         √
    [255, 192, 203],  # bicycle              pink         √
    [255, 127, 0],  # motorcycle           dark orange  √
    [255, 0, 0],  # pedestrian           red          √
    [255, 240, 150],  # traffic_cone         light yellow
    [255, 120, 50],  # barrier              orange
    [255, 0, 255],  # driveable_surface    dark pink
    [139, 137, 137],  # other_flat           dark red
    [75, 0, 75],  # sidewalk             dard purple
    [150, 240, 80],  # terrain              light green
    [230, 230, 250],  # manmade              white
    [0, 175, 0],  # vegetation           green
    [255, 255, 255],  # Free                 White
])


@DATASETS.register_module()
class NuScenesDatasetOccupancy(NuScenesDataset):

    def __init__(self,
                 lidar_sweeps_num=1,
                 miou_ignore_class_names=None,
                 miou_ignore_class_indices=None,
                 miou_dynamic_class_names=None,
                 miou_static_class_names=None,
                 point_cloud_range=None,
                 ray_eval_config=None,
                 next_frame_eval_config=None,
                 dynamic_motion_eval_config=None,
                 **kwargs):
        self.lidar_sweeps_num = max(int(lidar_sweeps_num), 1)
        self.miou_ignore_class_names = list(miou_ignore_class_names or [])
        self.miou_ignore_class_indices = list(miou_ignore_class_indices or [])
        self.miou_dynamic_class_names = list(miou_dynamic_class_names or [])
        self.miou_static_class_names = list(miou_static_class_names or [])
        self.point_cloud_range = (
            list(point_cloud_range) if point_cloud_range is not None else None)
        self.ray_eval_config = dict(ray_eval_config or {})
        self.next_frame_eval_config = dict(next_frame_eval_config or {})
        self.dynamic_motion_eval_config = dict(dynamic_motion_eval_config
                                               or {})
        self.next_frame_eval_enabled = bool(
            self.next_frame_eval_config.get('enabled', False))
        self.next_frame_timestamp_scale = float(
            self.next_frame_eval_config.get('timestamp_scale', 1e-4))
        self.dynamic_motion_eval_enabled = bool(
            self.dynamic_motion_eval_config.get('enabled', False))
        self.dynamic_motion_speed_threshold = float(
            self.dynamic_motion_eval_config.get('speed_threshold', 2.0))
        self.dynamic_motion_fast_speed_threshold = float(
            self.dynamic_motion_eval_config.get('fast_speed_threshold', 5.0))
        if (self.dynamic_motion_fast_speed_threshold
                <= self.dynamic_motion_speed_threshold):
            raise ValueError('dynamic motion fast_speed_threshold must exceed '
                             'speed_threshold.')
        self.virtual_ray_origin_mode = self.ray_eval_config.get(
            'virtual_origin_mode', 'center')
        self.virtual_ray_origin_z = float(
            self.ray_eval_config.get('virtual_origin_z', 0.0))
        self.virtual_ray_near_clip = float(
            self.ray_eval_config.get('near_clip', 3.0))
        self.virtual_ray_repeat = max(
            int(self.ray_eval_config.get('repeat', 1)), 1)
        super().__init__(**kwargs)
        self._token_to_index = {
            info['token']: index
            for index, info in enumerate(self.data_infos)
        }

    def _resolve_occ_label_path(self, info, dataset_name):
        occ_path = info['occ_path']
        if dataset_name == 'openocc':
            occ_path = occ_path.replace('gts', 'openocc_v2')
        return os.path.join(occ_path, 'labels.npz')

    def _resolve_flow_label_path(self, info, dataset_name):
        if self.flow_gt_path:
            occ_root = Path(info['occ_path']).resolve()
            occ_parts = os.path.normpath(str(occ_root)).split(os.sep)
            if 'gts' in occ_parts:
                rel_path = os.path.join(*occ_parts[occ_parts.index('gts') +
                                                   1:])
            else:
                rel_path = os.path.basename(str(occ_root))
            candidate = os.path.join(
                os.path.abspath(self.flow_gt_path), rel_path, 'labels.npz')
            if os.path.exists(candidate):
                return candidate
        return self._resolve_occ_label_path(info, dataset_name)

    def _next_data_info(self, index):
        """Return the following keyframe when it belongs to the same scene."""
        next_index = int(index) + 1
        if next_index >= len(self.data_infos):
            return None
        current_info = self.data_infos[index]
        next_info = self.data_infos[next_index]
        if next_info.get('scene_token') != current_info.get('scene_token'):
            return None
        if int(next_info.get('timestamp',
                             0)) <= int(current_info.get('timestamp', 0)):
            return None
        return next_info

    def _dynamic_class_indices_for_eval(self):
        class_names = list(self.CLASSES) if self.CLASSES is not None else []
        dynamic_names = self.miou_dynamic_class_names or [
            'bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'truck'
        ]
        class_to_index = {
            name: index
            for index, name in enumerate(class_names)
        }
        dynamic_class_indices = [
            class_to_index[name] for name in dynamic_names
            if name in class_to_index
        ]
        if not dynamic_class_indices:
            raise ValueError(
                'Dynamic evaluation requires dynamic class names that exist '
                'in dataset classes.')
        return class_names, dynamic_class_indices

    def _build_virtual_lidar_origin(self):
        pc_range = np.asarray(
            self.point_cloud_range if self.point_cloud_range is not None else
            [-64.0, -64.0, -4.8, 64.0, 64.0, 1.6],
            dtype=np.float32)

        if self.virtual_ray_origin_mode != 'center':
            raise NotImplementedError(
                f'Unsupported virtual origin mode: {self.virtual_ray_origin_mode}'
            )

        origin_z = np.clip(
            self.virtual_ray_origin_z,
            pc_range[2],
            pc_range[5],
        )
        origin = np.array([
            (pc_range[0] + pc_range[3]) * 0.5,
            (pc_range[1] + pc_range[4]) * 0.5,
            origin_z,
        ],
                          dtype=np.float32)
        if self.virtual_ray_repeat == 1:
            return torch.from_numpy(origin[None, :])
        return torch.from_numpy(
            np.repeat(origin[None, :], self.virtual_ray_repeat, axis=0))

    def _build_lidar_sweeps(self, index):
        info = self.data_infos[index]
        required_history = max(self.lidar_sweeps_num - 1, 0)
        if required_history == 0:
            return []

        raw_sweeps = list(info.get('sweeps', []))
        if raw_sweeps:
            return raw_sweeps[:required_history]

        sweeps = []
        cursor = index - 1
        while cursor >= 0 and len(sweeps) < required_history:
            candidate = self.data_infos[cursor]
            if candidate['scene_token'] != info['scene_token']:
                break
            sweeps.append(candidate)
            cursor -= 1
        return sweeps

    def _clone_info_with_lidar_sweeps(self, index):
        info = dict(self.data_infos[index])
        info['sweeps'] = self._build_lidar_sweeps(index)
        return info

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self._clone_info_with_lidar_sweeps(index)
        input_dict = dict(
            index=index,
            sample_idx=info['token'],
            prev_sample_idx=info.get('prev', ''),
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
            can_bus=info['can_bus'],
            scene_name=info['occ_path'].split('/')[-2],
            curr=info,
        )

        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']

        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    lidar2cam_r = np.linalg.inv(
                        cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.
                            shape[1]] = intrinsic
                    lidar2img_rt = viewpad @ lidar2cam_rt.T
                    lidar2img_rts.append(lidar2img_rt)
                input_dict.update(
                    dict(img_filename=image_paths, lidar2img=lidar2img_rts))
                if not self.test_mode:
                    input_dict['ann_info'] = self.get_ann_info(index)
            else:
                assert 'bevdet' in self.img_info_prototype
                if '4d' in self.img_info_prototype:
                    info_adj_list = [
                        self._clone_info_with_lidar_sweeps(
                            self._token_to_index[adj_info['token']])
                        for adj_info in self.get_adj_info(
                            self.data_infos[index], index)
                    ]
                    input_dict.update(dict(adjacent=info_adj_list))

        # RoadOcc uses the same adjacent-frame selection for camera and lidar.
        # Lidar-only mode still needs adjacent sample infos even though it does
        # not consume image metadata.
        if 'adjacent' not in input_dict and self.multi_adj_frame_id_cfg is not None:
            assert self.img_info_prototype is not None and '4d' in self.img_info_prototype
            input_dict['adjacent'] = [
                self._clone_info_with_lidar_sweeps(
                    self._token_to_index[adj_info['token']]) for adj_info in
                self.get_adj_info(self.data_infos[index], index)
            ]

        if self.use_sequence_group_flag:
            input_dict['sample_index'] = index
            input_dict['sequence_group_idx'] = self.flag[index]
            input_dict['start_of_sequence'] = index == 0 or self.flag[
                index - 1] != self.flag[index]
            if not self.test_mode and self.bda_aug_conf is not None and input_dict[
                    'start_of_sequence']:
                flip_dx = np.random.uniform(
                ) < self.bda_aug_conf['flip_dx_ratio']
                flip_dy = np.random.uniform(
                ) < self.bda_aug_conf['flip_dy_ratio']
                input_dict['bda_aug'] = dict(flip_dx=flip_dx, flip_dy=flip_dy)
            input_dict['nuscenes_get_rt_matrix'] = dict(
                lidar2ego_rotation=self.data_infos[index]
                ['lidar2ego_rotation'],
                lidar2ego_translation=self.data_infos[index]
                ['lidar2ego_translation'],
                ego2global_rotation=self.data_infos[index]
                ['ego2global_rotation'],
                ego2global_translation=self.data_infos[index]
                ['ego2global_translation'],
            )
            if not input_dict['start_of_sequence']:
                input_dict['curr_to_prev_lidar_rt'] = torch.FloatTensor(
                    nuscenes_get_rt_matrix(self.data_infos[index],
                                           self.data_infos[index - 1], 'lidar',
                                           'lidar'))
                input_dict['prev_lidar_to_global_rt'] = torch.FloatTensor(
                    nuscenes_get_rt_matrix(self.data_infos[index - 1],
                                           self.data_infos[index], 'lidar',
                                           'global'))
                input_dict['curr_to_prev_ego_rt'] = torch.FloatTensor(
                    nuscenes_get_rt_matrix(self.data_infos[index],
                                           self.data_infos[index - 1], 'ego',
                                           'ego'))
            else:
                input_dict['curr_to_prev_lidar_rt'] = torch.eye(4).float()
                input_dict['prev_lidar_to_global_rt'] = torch.FloatTensor(
                    nuscenes_get_rt_matrix(self.data_infos[index],
                                           self.data_infos[index], 'lidar',
                                           'global'))
                input_dict['curr_to_prev_ego_rt'] = torch.FloatTensor(
                    nuscenes_get_rt_matrix(self.data_infos[index],
                                           self.data_infos[index], 'ego',
                                           'ego'))
            input_dict['global_to_curr_lidar_rt'] = torch.FloatTensor(
                nuscenes_get_rt_matrix(self.data_infos[index],
                                       self.data_infos[index], 'global',
                                       'lidar'))

        rotation = Quaternion(input_dict['curr']['ego2global_rotation'])
        translation = input_dict['curr']['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        input_dict['can_bus'] = can_bus

        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        if 'pts_semantic_mask_path' in self.data_infos[index]:
            input_dict['pts_semantic_mask_path'] = self.data_infos[index][
                'pts_semantic_mask_path']
        input_dict['seg_label_mapping'] = self.SegLabelMapping
        return input_dict

    def evaluate_rayioU(self, results, logger=None, dataset_name='openocc'):
        use_virtual_ray_origin = dataset_name not in ('openocc', 'occ3d')

        if self.eval_show and self.work_dir:
            mmcv.mkdir_or_exist(self.work_dir)

        pred_sems, gt_sems = [], []
        pred_flows, gt_flows = [], []
        lidar_origins = []
        data_index = []

        print('\nStarting Evaluation...')
        processed_set = set()
        for index, result in enumerate(results):
            data_id = result['index']
            for i, id in enumerate(data_id):
                if id in processed_set: continue
                processed_set.add(id)

                pred_sem = result['occ_results'][i]

                if 'flow_results' not in result:
                    pred_flow = np.zeros(pred_sem.shape + (2, ))
                else:
                    pred_flow = result['flow_results'][i]

                data_index.append(id)
                pred_sems.append(pred_sem)
                pred_flows.append(pred_flow)

        nusdata = None
        virtual_origin_tensor = None
        if use_virtual_ray_origin:
            virtual_origin_tensor = self._build_virtual_lidar_origin()
            print_log(
                'Use virtual ray origin for dataset "{}": mode={}, origin={}, '
                'near_clip={:.2f}m, repeat={}'.format(
                    dataset_name,
                    self.virtual_ray_origin_mode,
                    virtual_origin_tensor[0].tolist(),
                    self.virtual_ray_near_clip,
                    self.virtual_ray_repeat,
                ),
                logger=logger)
        else:
            nusc = NuScenes('v1.0-trainval', self.data_root)
            nusdata = nuScenesDataset(nusc, 'val')

        max_valid_index = len(self.data_infos)
        missing_origin_tokens = []
        valid_pred_sems, valid_pred_flows = [], []
        for pred_sem, pred_flow, index in zip(pred_sems, pred_flows,
                                              data_index):
            if index >= max_valid_index:
                continue
            info = self.data_infos[index]
            sample_token = info.get('token')
            if use_virtual_ray_origin:
                output_origin_tensor = virtual_origin_tensor
            else:
                output_origin_tensor = None
                if sample_token is not None:
                    output_origin_tensor = nusdata.get_output_origin_by_sample_token(
                        sample_token)
                if output_origin_tensor is None:
                    missing_origin_tokens.append(
                        sample_token
                        if sample_token is not None else f'index:{index}')
                    continue

            valid_pred_sems.append(pred_sem)
            valid_pred_flows.append(pred_flow)

            occ_path = self._resolve_occ_label_path(info, dataset_name)
            with np.load(occ_path, allow_pickle=True) as occ_gt:
                gt_semantics = occ_gt['semantics'].astype(np.uint8)

            flow_path = self._resolve_flow_label_path(info, dataset_name)
            if dataset_name == 'occ3d':
                gt_flow = np.zeros(
                    gt_semantics.shape + (2, ), dtype=np.float16)
            elif os.path.exists(flow_path):
                with np.load(flow_path, allow_pickle=True) as flow_gt_npz:
                    if 'flow' in flow_gt_npz:
                        gt_flow = flow_gt_npz['flow'].astype(np.float16)
                    else:
                        gt_flow = np.zeros(
                            gt_semantics.shape + (2, ), dtype=np.float16)
            else:
                gt_flow = np.zeros(
                    gt_semantics.shape + (2, ), dtype=np.float16)

            gt_sems.append(gt_semantics)
            gt_flows.append(gt_flow)
            lidar_origins.append(output_origin_tensor.unsqueeze(0))

        pred_sems = valid_pred_sems
        pred_flows = valid_pred_flows

        if missing_origin_tokens:
            preview = ', '.join(map(str, missing_origin_tokens[:5]))
            print_log(
                'Skipped {} samples in rayIoU evaluation because lidar origins '
                'could not be aligned by sample token. First missing tokens: {}'
                .format(len(missing_origin_tokens), preview),
                logger=logger)
        if not pred_sems:
            print_log(
                'rayIoU evaluation skipped because no samples could be aligned '
                'with lidar origins.',
                logger=logger)
            return {
                'rayiou': float('nan'),
                'mave': float('nan'),
                'direct_mave': float('nan'),
                'dynamic_match_recall': float('nan'),
                'occ_score': float('nan'),
            }

        # visualization
        # if self.eval_show:
        #     for index in range(len(data_index)):
        #         if index >= len(self.data_infos):
        #             break
        #         info = self.data_infos[data_index[index]]
        #         if dataset_name == 'openocc':
        #             occ_bev_vis = self.vis_occ(pred_sems[index], color_map=openocc_colors_map, empty_idx=16)
        #             occ_bev_vis_gt = self.vis_occ(gt_sems[index], color_map=openocc_colors_map, empty_idx=16)
        #         elif dataset_name == 'occ3d':
        #             occ_bev_vis = self.vis_occ(pred_sems[index], color_map=occ3d_colors_map, empty_idx=17)
        #             occ_bev_vis_gt = self.vis_occ(gt_sems[index], color_map=occ3d_colors_map, empty_idx=17)
        #         scene_token = info['token']
        #         occ_bev_vis = np.concatenate([occ_bev_vis, occ_bev_vis_gt], axis=1)
        #         cv2.imwrite(os.path.join(self.work_dir, f'{scene_token}.png'), occ_bev_vis)

        if dataset_name == 'openocc':
            from mmdet3d.datasets.ray_metrics_openocc import \
                main as ray_based_miou_openocc
            miou, mave, occ_score = ray_based_miou_openocc(
                pred_sems,
                gt_sems,
                pred_flows,
                gt_flows,
                lidar_origins,
                logger=logger)
            extra_flow_metrics = {}
        else:
            from .ray_metrics_occ3d import main as ray_based_miou_occ3d
            class_names = list(
                self.CLASSES) if self.CLASSES is not None else None
            flow_class_names = [
                name
                for name in ('bicycle', 'bus', 'car', 'construction_vehicle',
                             'motorcycle', 'pedestrian', 'trailer', 'truck')
                if class_names is None or name in class_names
            ]
            point_cloud_range = (
                self.point_cloud_range if self.point_cloud_range is not None
                else [-64.0, -64.0, -4.8, 64.0, 64.0, 1.6])
            miou, mave, occ_score, extra_flow_metrics = ray_based_miou_occ3d(
                pred_sems,
                gt_sems,
                pred_flows,
                gt_flows,
                lidar_origins,
                logger=logger,
                point_cloud_range=point_cloud_range,
                class_names=class_names,
                flow_class_names_override=flow_class_names,
                near_clip=self.virtual_ray_near_clip
                if use_virtual_ray_origin else 0.0,
            )

        eval_dict = {
            'rayiou':
            miou * 100.0,
            'mave':
            mave,
            'direct_mave':
            extra_flow_metrics.get('direct_mave', float('nan')),
            'dynamic_match_recall':
            (extra_flow_metrics.get('dynamic_match_recall', float('nan')) *
             100.0),
            'occ_score':
            occ_score * 100.0,
        }
        return eval_dict

    def evaluate_miou(self, results, logger=None, dataset_name='openocc'):
        pred_sems = []
        data_index = []
        num_classes = 17 if dataset_name == 'openocc' else 18
        empty_idx = num_classes - 1

        miou_metric = Metric_mIoU(
            num_classes=num_classes,
            use_lidar_mask=False,
            use_image_mask=False,
            logger=logger,
            class_names=(list(self.CLASSES) if self.CLASSES is not None
                         and len(self.CLASSES) == num_classes else None),
            ignore_class_names=self.miou_ignore_class_names,
            ignore_class_indices=self.miou_ignore_class_indices,
            dynamic_class_names=self.miou_dynamic_class_names or None,
            static_class_names=self.miou_static_class_names or None,
        )
        geom_tp = 0
        geom_fp = 0
        geom_fn = 0
        print('\nStarting Evaluation...')
        processed_set = set()
        for result in results:
            data_id = result['index']
            for i, id in enumerate(data_id):
                if id in processed_set:
                    continue
                processed_set.add(id)
                pred_sem = result['occ_results'][i]
                data_index.append(id)
                pred_sems.append(pred_sem)
        for index, pred_semantics in tqdm(
                list(zip(data_index, pred_sems)), total=len(data_index)):
            if index >= len(self.data_infos):
                continue
            info = self.data_infos[index]
            occ_path = self._resolve_occ_label_path(info, dataset_name)
            occ_gt = np.load(occ_path, allow_pickle=True)
            gt_semantics = occ_gt['semantics']
            pred_semantics = np.asarray(pred_semantics)
            if pred_semantics.shape != gt_semantics.shape:
                if pred_semantics.ndim == 3:
                    pred_t = torch.from_numpy(pred_semantics).float()[None,
                                                                      None]
                    pred_t = F.interpolate(
                        pred_t, size=gt_semantics.shape, mode='nearest')
                    pred_semantics = pred_t[0,
                                            0].cpu().numpy().astype(np.uint8)
                else:
                    pred_semantics = np.reshape(pred_semantics,
                                                gt_semantics.shape)
            if dataset_name == 'occ3d':
                mask_camera = occ_gt['mask_camera'].astype(bool)
            else:
                mask_camera = None
            miou_metric.add_batch(pred_semantics, gt_semantics, None,
                                  mask_camera)

            valid_mask = (gt_semantics >= 0) & (gt_semantics < num_classes)
            if mask_camera is not None:
                valid_mask &= mask_camera
            gt_occ = (gt_semantics != empty_idx) & valid_mask
            pred_occ = (pred_semantics != empty_idx) & valid_mask
            geom_tp += np.logical_and(gt_occ, pred_occ).sum()
            geom_fp += np.logical_and(~gt_occ, pred_occ).sum()
            geom_fn += np.logical_and(gt_occ, ~pred_occ).sum()

        _, miou, _, ret_dict, _ = miou_metric.count_miou()
        geom_iou = round(geom_tp / max(geom_tp + geom_fp + geom_fn, 1) * 100,
                         2)
        print_log(
            f'===> geometric IoU of {len(data_index)} samples: {geom_iou}',
            logger=logger)
        eval_dict = {
            'miou': miou,
            'geom_iou': geom_iou,
        }
        if 'dynamic_mIoU' in ret_dict:
            eval_dict['dynamic_miou'] = float(ret_dict['dynamic_mIoU'])
        if 'static_mIoU' in ret_dict:
            eval_dict['static_miou'] = float(ret_dict['static_mIoU'])
        return eval_dict

    def evaluate_next_frame_dynamic(self,
                                    results,
                                    logger=None,
                                    dataset_name='openocc'):
        """Evaluate dynamic occupancy after transport to the next keyframe.

        This metric is intentionally opt-in because it loads a second occupancy
        label for every evaluated sample. It measures the temporal-addressing
        claim directly: predicted dynamic voxels are advanced by predicted flow
        and then compared with the following keyframe in its own ego frame.
        """
        if dataset_name == 'occ3d':
            return {
                'next_dynamic_iou': float('nan'),
                'next_dynamic_samples': 0
            }

        predictions = {}
        for result in results:
            data_ids = result['index']
            for local_index, data_index in enumerate(data_ids):
                data_index = int(data_index)
                if data_index in predictions or 'flow_results' not in result:
                    continue
                predictions[data_index] = (
                    np.asarray(result['occ_results'][local_index]),
                    np.asarray(result['flow_results'][local_index]),
                )

        class_names, dynamic_class_indices = self._dynamic_class_indices_for_eval(
        )
        empty_idx = len(class_names) - 1

        skipped = dict(missing_prediction=0, missing_next=0, missing_label=0)

        def iter_next_frame_samples():
            for data_index in tqdm(
                    sorted(predictions), desc='Next-frame dynamic evaluation'):
                if data_index < 0 or data_index >= len(self.data_infos):
                    skipped['missing_prediction'] += 1
                    continue
                next_info = self._next_data_info(data_index)
                if next_info is None:
                    skipped['missing_next'] += 1
                    continue
                next_path = self._resolve_occ_label_path(
                    next_info, dataset_name)
                if not os.path.exists(next_path):
                    skipped['missing_label'] += 1
                    continue
                with np.load(next_path, allow_pickle=True) as labels:
                    next_semantics = labels['semantics'].astype(np.uint8)
                pred_semantics, pred_flow = predictions[data_index]
                if pred_semantics.shape != next_semantics.shape:
                    raise ValueError(
                        'next-frame evaluation requires final occupancy predictions '
                        f'to match GT shape, got {pred_semantics.shape} and '
                        f'{next_semantics.shape} at dataset index {data_index}.'
                    )
                if pred_flow.shape[:3] != pred_semantics.shape:
                    raise ValueError(
                        'next-frame evaluation requires final flow predictions to '
                        f'match occupancy shape, got {pred_flow.shape} and '
                        f'{pred_semantics.shape} at dataset index {data_index}.'
                    )
                current_info = self.data_infos[data_index]
                delta_t = (
                    int(next_info['timestamp']) -
                    int(current_info['timestamp'])) * \
                    self.next_frame_timestamp_scale
                if delta_t <= 0.0:
                    skipped['missing_next'] += 1
                    continue
                yield (pred_semantics, pred_flow, next_semantics, current_info,
                       next_info, delta_t)

        metrics = evaluate_next_frame_dynamic_iou_samples(
            samples=iter_next_frame_samples(),
            point_cloud_range=(self.point_cloud_range
                               if self.point_cloud_range is not None else
                               [-64.0, -64.0, -4.8, 64.0, 64.0, 1.6]),
            dynamic_class_indices=dynamic_class_indices,
            empty_idx=empty_idx,
        )
        if metrics['samples'] == 0:
            print_log(
                'next-frame dynamic evaluation skipped: no valid consecutive predictions.',
                logger=logger)
            return {
                'next_dynamic_iou': float('nan'),
                'next_dynamic_samples': 0
            }
        print_log(
            'NextDynamicIoU@next-frame: {:.2f}% over {} samples '
            '(binary={:.2f}%, ghost={:.2f}%; skipped: prediction={}, '
            'next={}, label={})'.format(metrics['next_dynamic_iou'],
                                        metrics['samples'],
                                        metrics['next_dynamic_binary_iou'],
                                        metrics['next_dynamic_ghost_rate'],
                                        skipped['missing_prediction'],
                                        skipped['missing_next'],
                                        skipped['missing_label']),
            logger=logger)
        return {
            'next_dynamic_iou': metrics['next_dynamic_iou'],
            'next_dynamic_binary_iou': metrics['next_dynamic_binary_iou'],
            'next_dynamic_ghost_rate': metrics['next_dynamic_ghost_rate'],
            'next_dynamic_samples': metrics['samples'],
        }

    def evaluate_dynamic_motion_diagnostics(self,
                                            results,
                                            logger=None,
                                            dataset_name='openocc'):
        """Evaluate dynamic occupancy split by flow-derived motion state."""
        if dataset_name == 'occ3d':
            return {
                'moving_dynamic_iou': float('nan'),
                'still_dynamic_iou': float('nan'),
                'slow_dynamic_iou': float('nan'),
                'medium_dynamic_iou': float('nan'),
                'fast_dynamic_iou': float('nan'),
                'dynamic_ghost_rate': float('nan'),
                'dynamic_motion_samples': 0,
            }

        predictions = {}
        for result in results:
            data_ids = result['index']
            for local_index, data_index in enumerate(data_ids):
                data_index = int(data_index)
                if data_index in predictions or 'flow_results' not in result:
                    continue
                predictions[data_index] = (
                    np.asarray(result['occ_results'][local_index]),
                    np.asarray(result['flow_results'][local_index]),
                )

        _, dynamic_class_indices = self._dynamic_class_indices_for_eval()
        pred_sems, pred_flows, gt_sems, gt_flows, valid_masks = [], [], [], [], []
        skipped = dict(
            missing_prediction=0, missing_label=0, missing_flow=0, shape=0)
        for data_index in sorted(predictions):
            if data_index < 0 or data_index >= len(self.data_infos):
                skipped['missing_prediction'] += 1
                continue
            info = self.data_infos[data_index]
            occ_path = self._resolve_occ_label_path(info, dataset_name)
            flow_path = self._resolve_flow_label_path(info, dataset_name)
            if not os.path.exists(occ_path):
                skipped['missing_label'] += 1
                continue
            if not os.path.exists(flow_path):
                skipped['missing_flow'] += 1
                continue
            with np.load(occ_path, allow_pickle=True) as labels:
                gt_semantics = labels['semantics'].astype(np.uint8)
                valid_mask = labels['mask_camera'].astype(bool) \
                    if 'mask_camera' in labels.files else None
            with np.load(flow_path, allow_pickle=True) as flow_labels:
                if 'flow' not in flow_labels.files:
                    skipped['missing_flow'] += 1
                    continue
                gt_flow = flow_labels['flow'].astype(np.float32)
            pred_semantics, pred_flow = predictions[data_index]
            if (pred_semantics.shape != gt_semantics.shape
                    or pred_flow.shape[:3] != gt_semantics.shape
                    or gt_flow.shape[:3] != gt_semantics.shape):
                skipped['shape'] += 1
                continue
            pred_sems.append(pred_semantics)
            pred_flows.append(pred_flow)
            gt_sems.append(gt_semantics)
            gt_flows.append(gt_flow)
            valid_masks.append(valid_mask)

        if not pred_sems:
            print_log(
                'dynamic motion diagnostics skipped: no valid flow-labelled '
                'predictions.',
                logger=logger)
            return {
                'moving_dynamic_iou': float('nan'),
                'still_dynamic_iou': float('nan'),
                'slow_dynamic_iou': float('nan'),
                'medium_dynamic_iou': float('nan'),
                'fast_dynamic_iou': float('nan'),
                'dynamic_ghost_rate': float('nan'),
                'dynamic_motion_samples': 0,
            }

        metrics = evaluate_dynamic_motion_state_iou(
            pred_semantics_list=pred_sems,
            pred_flow_list=pred_flows,
            gt_semantics_list=gt_sems,
            gt_flow_list=gt_flows,
            valid_mask_list=valid_masks,
            dynamic_class_indices=dynamic_class_indices,
            speed_threshold=self.dynamic_motion_speed_threshold,
            fast_speed_threshold=self.dynamic_motion_fast_speed_threshold,
        )
        print_log(
            'DynamicMotionDiagnostics: moving={:.2f}%, still={:.2f}%, '
            'ghost={:.2f}% over {} samples at speed>={:.2f}m/s '
            '(skipped: prediction={}, label={}, flow={}, shape={})'.format(
                metrics['moving_dynamic_iou'], metrics['still_dynamic_iou'],
                metrics['dynamic_ghost_rate'],
                metrics['dynamic_motion_samples'],
                metrics['dynamic_motion_speed_threshold'],
                skipped['missing_prediction'], skipped['missing_label'],
                skipped['missing_flow'], skipped['shape']),
            logger=logger)
        print_log(
            'DynamicSpeedDiagnostics: slow={:.2f}%, medium={:.2f}%, '
            'fast={:.2f}% over {} samples at bins <{:.2f}/'
            '[{:.2f},{:.2f})/>={:.2f}m/s'.format(
                metrics['slow_dynamic_iou'], metrics['medium_dynamic_iou'],
                metrics['fast_dynamic_iou'], metrics['dynamic_motion_samples'],
                metrics['dynamic_motion_speed_threshold'],
                metrics['dynamic_motion_speed_threshold'],
                metrics['dynamic_motion_fast_speed_threshold'],
                metrics['dynamic_motion_fast_speed_threshold']),
            logger=logger)
        return {
            'moving_dynamic_iou': metrics['moving_dynamic_iou'],
            'still_dynamic_iou': metrics['still_dynamic_iou'],
            'slow_dynamic_iou': metrics['slow_dynamic_iou'],
            'medium_dynamic_iou': metrics['medium_dynamic_iou'],
            'fast_dynamic_iou': metrics['fast_dynamic_iou'],
            'dynamic_ghost_rate': metrics['dynamic_ghost_rate'],
            'dynamic_motion_samples': metrics['dynamic_motion_samples'],
        }

    def evaluate(self,
                 occ_results,
                 logger=None,
                 runner=None,
                 show_dir=None,
                 **eval_kwargs):
        temporal_gap_marked = any('temporal_gap_eval_valid' in info
                                  for info in self.data_infos)
        if temporal_gap_marked:
            filtered_results = []
            for result in occ_results:
                data_ids = result.get('index', [])
                if len(data_ids) != 1:
                    raise ValueError(
                        'Matched-anchor temporal-gap evaluation requires '
                        'samples_per_gpu=1.')
                data_index = int(data_ids[0])
                if (0 <= data_index < len(self.data_infos)
                        and self.data_infos[data_index].get(
                            'temporal_gap_eval_valid', False)):
                    filtered_results.append(result)
            print_log(
                'Temporal-gap matched-anchor evaluation: scoring {} of {} '
                'inference tokens.'.format(
                    len(filtered_results), len(self.data_infos)),
                logger=logger)
            occ_results = filtered_results
        if self.eval_metric == 'rayiou':
            return self.evaluate_rayioU(
                occ_results, logger, dataset_name=self.dataset_name)
        elif self.eval_metric == 'miou':
            return self.evaluate_miou(
                occ_results, logger, dataset_name=self.dataset_name)
        elif self.eval_metric in ('temporal_gap', 'miou+next'):
            eval_dict = {}
            eval_dict.update(
                self.evaluate_miou(
                    occ_results, logger, dataset_name=self.dataset_name))
            if self.next_frame_eval_enabled:
                eval_dict.update(
                    self.evaluate_next_frame_dynamic(
                        occ_results, logger, dataset_name=self.dataset_name))
            return eval_dict
        elif self.eval_metric in ('all', 'miou+rayiou', 'combined'):
            eval_dict = {}
            eval_dict.update(
                self.evaluate_miou(
                    occ_results, logger, dataset_name=self.dataset_name))
            eval_dict.update(
                self.evaluate_rayioU(
                    occ_results, logger, dataset_name=self.dataset_name))
            if self.next_frame_eval_enabled:
                eval_dict.update(
                    self.evaluate_next_frame_dynamic(
                        occ_results, logger, dataset_name=self.dataset_name))
            if self.dynamic_motion_eval_enabled:
                eval_dict.update(
                    self.evaluate_dynamic_motion_diagnostics(
                        occ_results, logger, dataset_name=self.dataset_name))
            return eval_dict

    def vis_occ(self, semantics, empty_idx, color_map=None):
        # simple visualization of result in BEV
        semantics_valid = np.logical_not(semantics == empty_idx)
        d = np.arange(16).reshape(1, 1, 16)
        d = np.repeat(d, 352, axis=0)
        d = np.repeat(d, 352, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(
            semantics_torch, dim=2, index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        occ_bev_vis = color_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(352, 352, 3)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis, (352, 352))

        occ_bev_vis = cv2.resize(occ_bev_vis, (600, 600))
        occ_bev_vis = cv2.cvtColor(occ_bev_vis, cv2.COLOR_BGR2RGB)
        return occ_bev_vis
