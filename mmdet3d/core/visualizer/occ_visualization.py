import colorsys
import fcntl
import json
import math
import os
import re
import shutil
import tempfile
import warnings
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

IMAGE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
DEFAULT_IMAGE_SIZE = (600, 600)

OCC_VISUALIZATION_TYPES = OrderedDict([
    ('camera_grid', 'camera_grid'),
    ('lidar_bev', 'lidar_bev'),
    ('lidar_history_bev', 'lidar_history_bev'),
    ('bev_feat', 'bev_feat_final'),
    ('occ_bev', 'occ_bev'),
    ('occ_pred_bev', 'occ_pred_bev'),
    ('occ_static_bev', 'occ_static_bev'),
    ('dynamic_candidate_bev', 'dynamic_candidate_bev'),
    ('dynamic_occ_bev', 'dynamic_occ_bev'),
    ('temporal_topk_bev', 'temporal_topk_bev'),
    ('footprint_token_bev', 'footprint_token_bev'),
    ('transition_gate_bev', 'transition_gate_bev'),
    ('canonical_ptr_bev', 'canonical_ptr_bev'),
    ('occ_flow_bev', 'occ_flow_bev'),
    ('stage_occ_flow_bev', 'stage_occ_flow_bev'),
    ('sparse_warp_bev', 'sparse_warp_bev'),
    ('occ_gt_bev', 'occ_gt_bev'),
    ('occ_fg_bg_prior_bev', 'occ_fg_bg_prior_bev'),
    ('occ_3d', 'occ_3d'),
    ('occ_pred_3d', 'occ_pred_3d'),
    ('occ_gt_3d', 'occ_gt_3d'),
    ('depth_grid', 'depth_grid'),
    ('modality_feat_grid', 'modality_feat_grid'),
    ('recursive_occ_bev', 'recursive_occ_bev'),
    ('prior_static_bias_bev', 'prior_static_bias_bev'),
])

_TYPE_TITLE = {
    'camera_grid': 'Camera Grid',
    'lidar_bev': 'Current LiDAR BEV',
    'lidar_history_bev': 'Aligned Multi-Frame LiDAR BEV',
    'bev_feat': 'Final BEV Feature',
    'occ_bev': 'Pred vs GT Occupancy BEV',
    'occ_pred_bev': 'Pred Occupancy BEV',
    'occ_static_bev': 'Static Occupancy BEV',
    'dynamic_candidate_bev': 'Dynamic Candidate BEV',
    'dynamic_occ_bev': 'Dynamic Occupancy BEV',
    'temporal_topk_bev': 'Temporal Top-K BEV',
    'footprint_token_bev': 'PTR Footprint Token BEV',
    'transition_gate_bev': 'PTR Transition Gate BEV',
    'canonical_ptr_bev': 'Canonical PTR Pyramid BEV',
    'occ_flow_bev': 'Occupancy Flow BEV',
    'stage_occ_flow_bev': 'Stage Occupancy Flow BEV',
    'sparse_warp_bev': 'Sparse Warp BEV',
    'occ_gt_bev': 'GT Occupancy BEV',
    'occ_fg_bg_prior_bev': 'Pred Occupancy vs Static Prior BEV',
    'occ_3d': 'Pred vs GT Occupancy 3D',
    'occ_pred_3d': 'Pred Occupancy 3D',
    'occ_gt_3d': 'GT Occupancy 3D',
    'depth_grid': 'Depth Grid',
    'modality_feat_grid': 'Modality Feature Summary',
    'recursive_occ_bev': 'Recursive Occupancy BEV',
    'prior_static_bias_bev': 'Prior Static Bias BEV',
}

_PLACEHOLDER_BG = (245, 245, 245)
_PLACEHOLDER_FG = (48, 48, 48)
_LEGACY_OCC_VISUALIZATION_DIRS = (
    'recursive_feat_grid',
    'occ_stage_bev',
)
_SPARSE_WARP_META_CACHE = {}
_SAME_TOKEN_MANIFEST_NAME = 'same_token_manifest.jsonl'


def _sample_meta_cache_key(sample_meta):
    if not isinstance(sample_meta, dict):
        return None
    sample_key = sample_meta.get('sample_idx', None)
    if sample_key in (None, ''):
        sample_key = sample_meta.get('token', None)
    if sample_key in (None, ''):
        return None
    return str(sample_key)


_WARP_STATUS_OUTSIDE = 0
_WARP_STATUS_EMPTY = 1
_WARP_STATUS_OCCUPIED_OTHER = 2
_WARP_STATUS_DYNAMIC_OTHER = 3
_WARP_STATUS_SAME_CLASS = 4

_WARP_HIT_COLORS = {
    _WARP_STATUS_SAME_CLASS: (64, 176, 64),
    _WARP_STATUS_DYNAMIC_OTHER: (255, 208, 64),
    _WARP_STATUS_OCCUPIED_OTHER: (220, 72, 72),
    _WARP_STATUS_EMPTY: (220, 72, 72),
}
_WARP_ZERO_BETTER_COLOR = (160, 160, 160)


def infer_epoch_from_checkpoint_path(checkpoint_path):
    if not checkpoint_path:
        return 0
    match = re.search(r'epoch_(\d+)', os.path.basename(checkpoint_path))
    if match:
        return int(match.group(1))
    return 0


def load_occ_gt_from_meta(img_meta):
    occ_gt_path = img_meta.get('occ_gt_path') or img_meta.get('occ_path')
    if not occ_gt_path:
        return None
    label_path = os.path.join(occ_gt_path, 'labels.npz')
    if not os.path.exists(label_path):
        return None
    occ_label = np.load(label_path, allow_pickle=True)
    return occ_label['semantics'].astype(np.uint8)


def _resolve_occ_label_path(img_meta,
                            label_name='labels.npz',
                            flow_gt_root=None):
    occ_gt_path = img_meta.get('occ_gt_path') or img_meta.get('occ_path')
    if not occ_gt_path:
        return None

    if flow_gt_root is None:
        return os.path.join(occ_gt_path, label_name)

    occ_gt_path_abs = os.path.abspath(occ_gt_path)
    flow_root_abs = os.path.abspath(flow_gt_root)
    occ_parts = os.path.normpath(occ_gt_path_abs).split(os.sep)
    if 'gts' in occ_parts:
        rel_path = os.path.join(*occ_parts[occ_parts.index('gts') + 1:])
    else:
        rel_path = os.path.basename(occ_gt_path_abs)
    return os.path.join(flow_root_abs, rel_path, label_name)


def _stage_label_name(scale_name):
    return 'labels.npz' if scale_name == '1_1' else f'labels_{scale_name}.npz'


def load_stage_occ_gt_from_meta(img_meta, scale_name):
    label_path = _resolve_occ_label_path(
        img_meta, label_name=_stage_label_name(scale_name))
    if label_path is None or not os.path.exists(label_path):
        return None
    occ_label = np.load(label_path, allow_pickle=True)
    if 'semantics' not in occ_label:
        return None
    return occ_label['semantics'].astype(np.uint8)


def load_occ_flow_from_meta(img_meta, flow_gt_root=None):
    label_path = _resolve_occ_label_path(
        img_meta, label_name='labels.npz', flow_gt_root=flow_gt_root)
    if label_path is None or not os.path.exists(label_path):
        return None
    occ_label = np.load(label_path, allow_pickle=True)
    if 'flow' not in occ_label:
        return None
    return occ_label['flow'].astype(np.float32)


def load_stage_occ_flow_from_meta(img_meta, scale_name, flow_gt_root=None):
    label_path = _resolve_occ_label_path(
        img_meta,
        label_name=_stage_label_name(scale_name),
        flow_gt_root=flow_gt_root)
    if label_path is None or not os.path.exists(label_path):
        return None
    occ_label = np.load(label_path, allow_pickle=True)
    if 'flow' not in occ_label:
        return None
    return occ_label['flow'].astype(np.float32)


def make_occ_color_map(class_names):
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
        [palette.get(name, (180, 180, 180)) for name in class_names],
        dtype=np.uint8)


def build_occ_visualization_dirs(base_dir, enabled_types=None):
    enabled_types = list(OCC_VISUALIZATION_TYPES.keys()
                         ) if enabled_types is None else list(enabled_types)
    if 'modality_feat_grid' in enabled_types and 'bev_feat' in enabled_types:
        enabled_types = [
            type_name for type_name in enabled_types if type_name != 'bev_feat'
        ]
    split_dirs = {}
    for split in ('train', 'test'):
        split_dir = os.path.join(base_dir, split)
        for legacy_dir in _LEGACY_OCC_VISUALIZATION_DIRS:
            legacy_path = os.path.join(split_dir, legacy_dir)
            if os.path.isdir(legacy_path):
                shutil.rmtree(legacy_path)
        bev_feat_path = os.path.join(split_dir, 'bev_feat')
        if 'bev_feat' not in enabled_types and os.path.isdir(bev_feat_path):
            shutil.rmtree(bev_feat_path)
        type_dirs = {}
        for type_name in enabled_types:
            if type_name not in OCC_VISUALIZATION_TYPES:
                continue
            save_dir = os.path.join(split_dir, type_name)
            os.makedirs(save_dir, exist_ok=True)
            type_dirs[type_name] = save_dir
        split_dirs[split] = type_dirs
    return split_dirs


def _append_same_token_manifest(output_dirs, split, epoch, vis_index,
                                scene_name, sample_token, artifact_paths):
    """Append one complete visualization group to the split manifest.

    Paths stay relative to the split directory so manifests remain valid when a
    work directory is moved or archived.  A file lock keeps each JSONL record
    intact if multiple visualization producers happen to share a directory.
    """
    if not sample_token or split not in output_dirs or not output_dirs[split]:
        return None

    split_dirs = tuple(output_dirs[split].values())
    try:
        split_root = os.path.commonpath(
            tuple(os.path.dirname(path) for path in split_dirs))
    except ValueError:
        return None
    if not split_root:
        return None

    artifacts = OrderedDict()
    for artifact_name, artifact_path in artifact_paths.items():
        try:
            artifact_exists = (artifact_path
                               and os.path.isfile(artifact_path)
                               and os.path.getsize(artifact_path) > 0)
        except OSError:
            artifact_exists = False
        if not artifact_exists:
            continue
        artifacts[str(artifact_name)] = os.path.relpath(
            artifact_path, split_root)
    if not artifacts:
        return None

    record = {
        'schema_version': 1,
        'sample_token': str(sample_token),
        'scene_name': str(scene_name),
        'split': str(split),
        'epoch': int(epoch),
        'vis_index': int(vis_index),
        'record_key': f'{split}:{int(epoch):03d}:{int(vis_index):06d}',
        'artifacts': artifacts,
    }
    manifest_path = os.path.join(split_root, _SAME_TOKEN_MANIFEST_NAME)
    payload = (json.dumps(record, sort_keys=True, separators=(',', ':')) +
               '\n').encode('utf-8')
    try:
        fd = os.open(manifest_path,
                     os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            offset = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise OSError(
                        'Could not append occupancy visualization manifest')
                offset += written
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
    except OSError as exc:
        warnings.warn(
            f'Could not append same-token visualization manifest: {exc}',
            RuntimeWarning)
        return None
    return manifest_path


def add_info_banner(image_bgr, lines):
    if image_bgr is None:
        return None
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_height = 22
    top_pad = 12
    bottom_pad = 10
    banner_h = top_pad + bottom_pad + line_height * len(lines)

    banner = np.full((banner_h, image_bgr.shape[1], 3),
                     _PLACEHOLDER_BG,
                     dtype=np.uint8)
    for line_index, line in enumerate(lines):
        y = top_pad + (line_index + 1) * line_height - 6
        cv2.putText(
            banner,
            str(line),
            (12, y),
            font,
            font_scale,
            _PLACEHOLDER_FG,
            thickness,
            cv2.LINE_AA,
        )
    return np.concatenate([banner, image_bgr], axis=0)


def create_placeholder_image(lines, message, size=DEFAULT_IMAGE_SIZE):
    width, height = size
    image = np.full((height, width, 3), _PLACEHOLDER_BG, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (width - 1, height - 1), (180, 180, 180), 1)
    cv2.putText(
        image,
        message,
        (30, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        _PLACEHOLDER_FG,
        2,
        cv2.LINE_AA,
    )
    return add_info_banner(image, lines)


def _create_placeholder_panel(message, size=DEFAULT_IMAGE_SIZE):
    width, height = size
    image = np.full((height, width, 3), _PLACEHOLDER_BG, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (width - 1, height - 1), (180, 180, 180), 1)
    cv2.putText(
        image,
        message,
        (30, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        _PLACEHOLDER_FG,
        2,
        cv2.LINE_AA,
    )
    return image


def _orient_xy_grid_to_image(grid):
    grid = np.asarray(grid)
    if grid.ndim == 2:
        return grid.transpose(1, 0)[::-1, ::-1]
    if grid.ndim == 3:
        return grid.transpose(1, 0, 2)[::-1, ::-1, :]
    return grid


def _orient_yx_grid_to_image(grid):
    grid = np.asarray(grid)
    if grid.ndim == 2:
        return grid[::-1, ::-1]
    if grid.ndim == 3:
        return grid[::-1, ::-1, :]
    return grid


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if hasattr(value, 'tensor'):
        return value.tensor.detach().cpu().numpy()
    return np.asarray(value)


def _to_numpy_occ(value, sample_index):
    if value is None:
        return None
    value = _to_numpy(value)
    if value is None:
        return None
    if value.ndim == 4:
        value = value[sample_index]
    return np.asarray(value).astype(np.uint8)


def _to_numpy_flow(value, sample_index):
    value = _to_numpy(value)
    if value is None:
        return None
    if value.ndim == 5:
        value = value[sample_index]
    # Dataset artifacts may retain a third (vertical) velocity component.
    # BEV/PTR transport is defined in XY, consistently with
    # ``_build_canonical_ptr_target(..., gt_flow[..., :2])``.
    if value.ndim != 4 or value.shape[-1] < 2:
        return None
    return np.asarray(value[..., :2], dtype=np.float32)


def _to_numpy_prior(value, sample_index, target_shape=None):
    if value is None:
        return None
    value = _to_numpy(value)
    if value is None:
        return None
    if value.ndim == 4:
        value = value[sample_index]
    value = np.asarray(value)
    if value.ndim != 3:
        return None
    if target_shape is not None and value.shape != target_shape:
        transposed = value.transpose(2, 1, 0)
        if transposed.shape == target_shape:
            value = transposed
    return (value > 0.5).astype(np.uint8)


def _extract_img_tensor(img_inputs):
    if img_inputs is None:
        return None
    if torch.is_tensor(img_inputs):
        return img_inputs
    if isinstance(img_inputs, (list, tuple)):
        if len(img_inputs) == 0:
            return None
        if torch.is_tensor(img_inputs[0]):
            return img_inputs[0]
        return _extract_img_tensor(img_inputs[0])
    return None


def _infer_depth_batch_size(pred_depth, gt_depth=None, img_inputs=None):
    pred_depth = _to_numpy(pred_depth)
    if pred_depth is None or pred_depth.ndim != 4:
        return None

    gt_depth = _to_numpy(gt_depth)
    if gt_depth is not None and gt_depth.ndim == 4:
        return int(gt_depth.shape[0])

    img_tensor = _extract_img_tensor(img_inputs)
    if img_tensor is not None:
        if img_tensor.dim() == 5:
            return int(img_tensor.shape[0])
        if img_tensor.dim() == 4:
            return 1
    return None


def _select_pred_depth_sample(pred_depth,
                              gt_depth=None,
                              img_inputs=None,
                              sample_index=0):
    pred_depth = _to_numpy(pred_depth)
    if pred_depth is None:
        return None

    if pred_depth.ndim == 5:
        if sample_index >= pred_depth.shape[0]:
            return None
        return pred_depth[sample_index]

    if pred_depth.ndim == 4:
        batch_size = _infer_depth_batch_size(
            pred_depth, gt_depth=gt_depth, img_inputs=img_inputs)
        if batch_size is not None and batch_size > 0 and pred_depth.shape[
                0] % batch_size == 0:
            num_panels = pred_depth.shape[0] // batch_size
            pred_depth = pred_depth.reshape(batch_size, num_panels,
                                            *pred_depth.shape[1:])
            if sample_index >= pred_depth.shape[0]:
                return None
            return pred_depth[sample_index]
    return pred_depth


def _select_current_camera_indices(total_slots,
                                   camera_num_frame=1,
                                   max_panels=4):
    total_slots = max(int(total_slots), 0)
    camera_num_frame = max(int(camera_num_frame), 1)
    if total_slots == 0:
        return []
    if total_slots % camera_num_frame == 0:
        num_cams = total_slots // camera_num_frame
        return [
            cam_id * camera_num_frame
            for cam_id in range(min(num_cams, max_panels))
        ]
    return list(range(min(max_panels, total_slots)))


def infer_camera_panel_size(img_inputs, sample_index=0):
    img_tensor = _extract_img_tensor(img_inputs)
    if img_tensor is None:
        return None
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.unsqueeze(0)
    if img_tensor.dim() != 5 or sample_index >= img_tensor.shape[0]:
        return None
    _, _, _, height, width = img_tensor.shape
    return int(width), int(height)


def _denormalize_image(image_tensor):
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * IMAGE_STD[None, None, :] + IMAGE_MEAN[None, None, :]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _make_labeled_placeholder(panel_size, label, value=235):
    width, height = panel_size
    panel = np.full((height, width, 3), value, dtype=np.uint8)
    cv2.putText(
        panel,
        str(label),
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _render_camera_panels(img_inputs,
                          sample_index=0,
                          camera_num_frame=1,
                          cam_names=None,
                          panel_size=None,
                          label_suffix=None,
                          max_panels=4):
    img_tensor = _extract_img_tensor(img_inputs)
    if img_tensor is None:
        return []
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.unsqueeze(0)
    if img_tensor.dim() != 5 or sample_index >= img_tensor.shape[0]:
        return []

    sample_imgs = img_tensor[sample_index]
    total_slots = sample_imgs.shape[0]
    # PrepareImageInputs stores temporal images camera-by-camera.  In the
    # BEVDet4D stereo setup the caller historically leaves camera_num_frame at
    # its default value, so infer the actual frame count from the camera list.
    if (int(camera_num_frame) == 1 and cam_names
            and total_slots % len(cam_names) == 0):
        camera_num_frame = total_slots // len(cam_names)
    current_indices = _select_current_camera_indices(
        total_slots,
        camera_num_frame=camera_num_frame,
        max_panels=max_panels)
    if not current_indices:
        return []

    labels = cam_names or [
        f'View {index}' for index in range(len(current_indices))
    ]
    panels = []
    for panel_index, img_index in enumerate(current_indices[:max_panels]):
        panel = _denormalize_image(sample_imgs[img_index])
        if panel_size is not None and (panel.shape[1],
                                       panel.shape[0]) != panel_size:
            panel = cv2.resize(
                panel, panel_size, interpolation=cv2.INTER_LINEAR)
        label = labels[panel_index] if panel_index < len(
            labels) else f'View {panel_index}'
        if label_suffix:
            label = f'{label} | {label_suffix}'
        cv2.putText(
            panel,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    return panels


def render_camera_grid(img_inputs,
                       sample_index,
                       camera_num_frame=1,
                       cam_names=None,
                       max_panels=4):
    panels = _render_camera_panels(
        img_inputs,
        sample_index=sample_index,
        camera_num_frame=camera_num_frame,
        cam_names=cam_names,
        max_panels=max_panels)
    if not panels:
        return None
    columns = 3 if max_panels > 4 else 2
    target_count = max(columns, int(max_panels))
    while len(panels) < target_count:
        panels.append(
            np.full_like(panels[0], 235) if panels else np.full(
                (256, 512, 3), 235, dtype=np.uint8))
    rows = [
        cv2.hconcat(panels[start:start + columns])
        for start in range(0, target_count, columns)
    ]
    return cv2.vconcat(rows)


def _detection_sample_value(value, sample_index):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if sample_index >= len(value):
            return None
        return _to_numpy(value[sample_index])
    value = _to_numpy(value)
    if value is None:
        return None
    if value.ndim >= 1:
        return value[sample_index]
    return value


def _center_box_corners(boxes):
    """Return BEV box corners for CenterHead boxes in occupancy coordinates."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    boxes = boxes.reshape(-1, boxes.shape[-1])
    dx = np.maximum(boxes[:, 3], 1e-3) * 0.5
    dy = np.maximum(boxes[:, 4], 1e-3) * 0.5
    local = np.stack([
        np.stack([dx, dy], axis=1),
        np.stack([dx, -dy], axis=1),
        np.stack([-dx, -dy], axis=1),
        np.stack([-dx, dy], axis=1),
    ],
                     axis=1)
    cosine = np.cos(boxes[:, 6])
    sine = np.sin(boxes[:, 6])
    rotation = np.stack([
        np.stack([cosine, -sine], axis=1),
        np.stack([sine, cosine], axis=1),
    ],
                        axis=1)
    return np.matmul(local, rotation.transpose(0, 2, 1)) + boxes[:, None, :2]


def _undo_bda_for_center_boxes(boxes, bda_mat, sample_index):
    if boxes is None:
        return None
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    if boxes.size == 0 or bda_mat is None:
        return boxes.reshape(-1, boxes.shape[-1] if boxes.ndim > 1 else 10)
    matrix = _to_numpy(bda_mat)
    if matrix.ndim == 3:
        if sample_index >= matrix.shape[0]:
            return boxes
        matrix = matrix[sample_index]
    if matrix.shape != (4, 4):
        return boxes
    inverse = np.linalg.inv(matrix).astype(np.float32)
    affine = inverse[:3, :3]
    homogeneous = np.concatenate(
        [boxes[:, :3],
         np.ones((boxes.shape[0], 1), dtype=np.float32)], axis=1)
    boxes[:, :3] = (homogeneous @ inverse.T)[:, :3]
    scale = float(np.linalg.norm(affine[0, :2]))
    boxes[:, 3:6] *= max(scale, 1e-6)
    if boxes.shape[1] >= 7:
        yaw_vector = np.stack(
            [np.cos(boxes[:, 6]), np.sin(boxes[:, 6])], axis=1)
        yaw_vector = yaw_vector @ affine[:2, :2].T
        boxes[:, 6] = np.arctan2(yaw_vector[:, 1], yaw_vector[:, 0])
    if boxes.shape[1] >= 9:
        boxes[:, 7:9] = boxes[:, 7:9] @ affine[:2, :2].T
    return boxes


def _center_detection_metadata(feature_payload):
    detection = feature_payload.get('center_detection')
    if not isinstance(detection, dict):
        return None
    class_names = list(
        feature_payload.get('center_detection_class_names') or [])
    source_indices = list(
        feature_payload.get('center_detection_source_indices') or [])
    source_to_name = {
        int(source): class_names[index]
        for index, source in enumerate(source_indices)
        if index < len(class_names)
    }
    return detection, class_names, source_indices, source_to_name


def _center_detection_label(label, class_names, source_to_name, source=False):
    label = int(label)
    if source:
        return source_to_name.get(label, str(label))
    return class_names[label] if 0 <= label < len(class_names) else str(label)


def render_center_detection_bev(feature_payload,
                                point_cloud_range,
                                sample_index=0,
                                image_size=(720, 720)):
    """Render a clean BEV diagnostic for CenterHead GT and predictions."""
    if not isinstance(feature_payload, dict):
        return None
    metadata = _center_detection_metadata(feature_payload)
    if metadata is None:
        return None
    detection, class_names, _, source_to_name = metadata
    boxes = _detection_sample_value(detection.get('boxes'), sample_index)
    scores = _detection_sample_value(detection.get('scores'), sample_index)
    gt_boxes = _detection_sample_value(
        feature_payload.get('center_detection_gt_boxes'), sample_index)
    gt_labels = _detection_sample_value(
        feature_payload.get('center_detection_gt_labels'), sample_index)
    gt_boxes = (
        np.asarray(gt_boxes, dtype=np.float32).reshape(-1, gt_boxes.shape[-1])
        if gt_boxes is not None and np.asarray(gt_boxes).size else np.zeros(
            (0, 10), dtype=np.float32))
    gt_labels = np.asarray(gt_labels).reshape(
        -1) if gt_labels is not None else np.zeros((0, ))
    boxes = (
        np.asarray(boxes, dtype=np.float32).reshape(-1, boxes.shape[-1])
        if boxes is not None and np.asarray(boxes).size else np.zeros(
            (0, 10), dtype=np.float32))
    scores = np.asarray(scores).reshape(
        -1) if scores is not None else np.zeros((0, ))
    # The auxiliary head supervises only its configured dynamic source
    # classes.  Hide unrelated detection annotations from this diagnostic.
    if gt_boxes.shape[0] and gt_labels.shape[0]:
        gt_mask = np.asarray(
            [int(label) in source_to_name for label in gt_labels])
        gt_boxes = gt_boxes[gt_mask]
        gt_labels = gt_labels[gt_mask]

    width, height = [int(value) for value in image_size]
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    pc_range = np.asarray(point_cloud_range, dtype=np.float32)
    x_min, y_min, x_max, y_max = pc_range[0], pc_range[1], pc_range[
        3], pc_range[4]

    def project_points(points):
        pixels = np.asarray(points, dtype=np.float32).copy()
        # Occupancy BEV rendering uses [x, y] -> image[y, x] with both
        # spatial axes reversed.  CenterHead is [H=y, W=x], so use the same
        # physical orientation for its boxes and heatmap.
        pixels[...,
               0] = (x_max - pixels[..., 0]) / max(x_max - x_min, 1e-6) * width
        pixels[..., 1] = (y_max - pixels[..., 1]) / max(y_max - y_min,
                                                        1e-6) * height
        return np.round(pixels).astype(np.int32)

    # Use a physical 10 m grid so the plot can be checked independently of
    # the network heatmap or predictions.
    grid_color = (225, 225, 225)
    axis_color = (175, 175, 175)
    for x_value in np.arange(np.ceil(x_min / 10.0) * 10.0, x_max + 1.0, 10.0):
        line = project_points(
            np.array([[x_value, y_min], [x_value, y_max]], dtype=np.float32))
        color = axis_color if abs(x_value) < 1e-4 else grid_color
        cv2.line(canvas, tuple(line[0]), tuple(line[1]), color, 1, cv2.LINE_AA)
    for y_value in np.arange(np.ceil(y_min / 10.0) * 10.0, y_max + 1.0, 10.0):
        line = project_points(
            np.array([[x_min, y_value], [x_max, y_value]], dtype=np.float32))
        color = axis_color if abs(y_value) < 1e-4 else grid_color
        cv2.line(canvas, tuple(line[0]), tuple(line[1]), color, 1, cv2.LINE_AA)

    if gt_boxes.shape[0] and gt_labels.shape[0]:
        corners = project_points(_center_box_corners(gt_boxes))
        fill_color = (96, 190, 255)
        for index, polygon in enumerate(corners):
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [polygon.reshape(-1, 1, 2)], fill_color)
            canvas = cv2.addWeighted(overlay, 0.48, canvas, 0.52, 0.0)
            cv2.polylines(canvas, [polygon.reshape(-1, 1, 2)], True,
                          (0, 125, 255), 1, cv2.LINE_AA)

    # Draw predictions last so their thin blue outlines remain visible over GT.
    threshold = float(detection.get('score_threshold', 0.05))
    pred_mask = scores >= threshold
    if boxes.shape[0] and pred_mask.shape[0] == boxes.shape[
            0] and pred_mask.any():
        pred_corners = project_points(_center_box_corners(boxes[pred_mask]))
        for polygon in pred_corners:
            cv2.polylines(canvas, [polygon.reshape(-1, 1, 2)], True,
                          (220, 24, 24), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (8, 8), (width - 8, height - 8), (70, 70, 70), 2)
    cv2.putText(canvas, 'CenterHead | GT orange | Pred blue | 10 m grid',
                (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (32, 32, 32), 1,
                cv2.LINE_AA)
    cv2.putText(
        canvas, f'point_cloud_range: x[{x_min:.0f},{x_max:.0f}] '
        f'y[{y_min:.0f},{y_max:.0f}] m', (16, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 80, 80), 1, cv2.LINE_AA)
    return canvas


def _draw_center_boxes_on_camera(image,
                                 boxes,
                                 labels,
                                 scores,
                                 ego2img,
                                 class_names,
                                 source_to_name,
                                 color,
                                 source_labels=False,
                                 score_threshold=0.0,
                                 draw_labels=False,
                                 line_width=1):
    if boxes is None:
        return image
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, boxes.shape[-1])
    labels = np.asarray(labels).reshape(
        -1) if labels is not None else np.zeros((0, ))
    scores = np.asarray(scores).reshape(-1) if scores is not None else np.ones(
        (len(boxes), ))
    matrix = np.asarray(ego2img, dtype=np.float32).reshape(4, 4)
    corners = _center_box_corners(boxes)
    for index, box_corners in enumerate(corners):
        if index >= len(labels) or index >= len(
                scores) or scores[index] < score_threshold:
            continue
        points = np.vstack([box_corners, box_corners]).astype(np.float32)
        points = np.concatenate(
            [points, np.zeros((8, 1), dtype=np.float32)], axis=1)
        # Add the upper and lower four corners using the box height.
        half_height = max(float(boxes[index, 5]) * 0.5, 1e-3)
        points[:4, 2] = boxes[index, 2] + half_height
        points[4:, :2] = box_corners
        points[4:, 2] = boxes[index, 2] - half_height
        points_4d = np.concatenate(
            [points, np.ones((8, 1), dtype=np.float32)], axis=1)
        projected = points_4d @ matrix.T
        depth = projected[:, 2]
        if not np.any(depth > 0.1):
            continue
        depth = np.clip(depth, 1e-4, None)
        projected = projected[:, :2] / depth[:, None]
        projected = np.round(projected).astype(np.int32)
        edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
                 (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
        for start, end in edges:
            cv2.line(image, tuple(projected[start]), tuple(projected[end]),
                     color, line_width, cv2.LINE_AA)
        visible = projected[np.isfinite(projected).all(axis=1)]
        if draw_labels and visible.size:
            anchor = tuple(np.mean(visible, axis=0).astype(np.int32))
            label = _center_detection_label(
                labels[index],
                class_names,
                source_to_name,
                source=source_labels)
            if not source_labels:
                label = f'{label} {float(scores[index]):.2f}'
            cv2.putText(image, label, (anchor[0] + 3, anchor[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return image


def render_center_detection_camera(feature_payload,
                                   img_inputs,
                                   sample_meta,
                                   sample_index=0,
                                   camera_num_frame=1,
                                   cam_names=None,
                                   bda_mat=None):
    """Render predicted and GT CenterHead boxes over the four camera views."""
    if not isinstance(feature_payload, dict):
        return None
    if hasattr(sample_meta, 'data'):
        sample_meta = sample_meta.data
    metadata = _center_detection_metadata(feature_payload)
    if metadata is None:
        return None
    detection, class_names, _, source_to_name = metadata
    boxes = _detection_sample_value(detection.get('boxes'), sample_index)
    labels = _detection_sample_value(detection.get('labels'), sample_index)
    scores = _detection_sample_value(detection.get('scores'), sample_index)
    gt_boxes = _detection_sample_value(
        feature_payload.get('center_detection_gt_boxes'), sample_index)
    gt_labels = _detection_sample_value(
        feature_payload.get('center_detection_gt_labels'), sample_index)
    threshold = float(detection.get('score_threshold', 0.05))
    boxes = _undo_bda_for_center_boxes(boxes, bda_mat, sample_index)
    gt_boxes = _undo_bda_for_center_boxes(gt_boxes, bda_mat, sample_index)
    panels = _render_camera_panels(
        img_inputs,
        sample_index=sample_index,
        camera_num_frame=camera_num_frame,
        cam_names=cam_names)
    if not panels:
        return None
    lidar2img = sample_meta.get('lidar2img') if isinstance(sample_meta,
                                                           dict) else None
    if lidar2img is None:
        return None
    lidar2img = _to_numpy(lidar2img)
    # ``lidar2img`` stores only the current frame's four cameras, while the
    # image tensor also contains adjacent temporal frames.
    current_indices = list(range(min(4, lidar2img.shape[0])))
    # Detection GT and CenterHead predictions use the occupancy/ego frame.
    # ``lidar2img`` starts from the raw lidar frame, so compose the explicit
    # ego-to-lidar transform before projecting either set of boxes.
    ego2lidar = _to_numpy(sample_meta.get('ego2lidar'))
    if ego2lidar is None:
        ego2lidar = np.eye(4, dtype=np.float32)
    elif ego2lidar.ndim == 3:
        ego2lidar = ego2lidar[0]
    ego2lidar = np.asarray(ego2lidar, dtype=np.float32).reshape(4, 4)
    gt_labels = np.asarray(gt_labels).reshape(
        -1) if gt_labels is not None else np.zeros((0, ))
    if gt_boxes is not None and len(gt_boxes) and len(gt_labels):
        gt_mask = np.asarray(
            [int(label) in source_to_name for label in gt_labels])
        gt_boxes = gt_boxes[gt_mask]
        gt_labels = gt_labels[gt_mask]
    for panel_index, panel in enumerate(panels):
        if panel_index >= len(current_indices):
            break
        matrix = lidar2img[current_indices[panel_index]] @ ego2lidar
        _draw_center_boxes_on_camera(
            panel,
            gt_boxes,
            gt_labels,
            None,
            matrix,
            class_names,
            source_to_name, (0, 125, 255),
            source_labels=True,
            line_width=1)
        _draw_center_boxes_on_camera(
            panel,
            boxes,
            labels,
            scores,
            matrix,
            class_names,
            source_to_name, (220, 24, 24),
            source_labels=False,
            score_threshold=threshold,
            line_width=1)
    while len(panels) < 4:
        panels.append(np.full_like(panels[0], 235))
    return cv2.vconcat([cv2.hconcat(panels[:2]), cv2.hconcat(panels[2:4])])


def _depth_bin_values(depth_channels, depth_bin_config):
    if depth_bin_config is not None and len(depth_bin_config) >= 3:
        depth_min, depth_max, depth_step = [
            float(value) for value in depth_bin_config[:3]
        ]
        depth_values = np.arange(
            depth_min, depth_max, depth_step, dtype=np.float32)
        if depth_values.size < depth_channels:
            depth_values = depth_min + np.arange(
                depth_channels, dtype=np.float32) * depth_step
        return depth_values[:depth_channels]
    return np.arange(depth_channels, dtype=np.float32)


def _infer_depth_channels(pred_depth=None, depth_bin_config=None):
    pred_depth = _to_numpy(pred_depth)
    if pred_depth is not None:
        if pred_depth.ndim == 4:
            return int(pred_depth.shape[1])
        if pred_depth.ndim == 3:
            return int(pred_depth.shape[0])
    if depth_bin_config is not None and len(depth_bin_config) >= 3:
        return int(
            np.arange(
                float(depth_bin_config[0]),
                float(depth_bin_config[1]),
                float(depth_bin_config[2]),
                dtype=np.float32).shape[0])
    return None


def _pred_depth_to_map(pred_depth, depth_bin_config):
    pred_depth = np.asarray(pred_depth, dtype=np.float32)
    if pred_depth.ndim != 3:
        return None
    depth_values = _depth_bin_values(pred_depth.shape[0], depth_bin_config)
    if depth_values.shape[0] != pred_depth.shape[0]:
        return None
    return np.tensordot(depth_values, pred_depth, axes=(0, 0))


def _downsample_gt_depth_maps(gt_depth, downsample=1):
    gt_depth = np.asarray(gt_depth, dtype=np.float32)
    if gt_depth.ndim == 2:
        gt_depth = gt_depth[None, ...]
    if gt_depth.ndim != 3:
        return None

    downsample = max(int(downsample), 1)
    if downsample == 1:
        return gt_depth

    num_cam, height, width = gt_depth.shape
    target_h = (height // downsample) * downsample
    target_w = (width // downsample) * downsample
    if target_h <= 0 or target_w <= 0:
        return gt_depth

    gt_depth = gt_depth[:, :target_h, :target_w]
    gt_depth = gt_depth.reshape(
        num_cam,
        target_h // downsample,
        downsample,
        target_w // downsample,
        downsample,
    )
    gt_depth = gt_depth.transpose(0, 1, 3, 2,
                                  4).reshape(num_cam, target_h // downsample,
                                             target_w // downsample,
                                             downsample * downsample)
    gt_depth_tmp = np.where(gt_depth == 0.0, 1e5, gt_depth)
    gt_depth = gt_depth_tmp.min(axis=-1)
    gt_depth[gt_depth >= 1e5] = 0.0
    return gt_depth.astype(np.float32)


def _loss_aligned_gt_depth_maps(gt_depth,
                                depth_bin_config,
                                downsample=1,
                                sid=False,
                                depth_channels=None,
                                target_shape=None):
    gt_depth = _downsample_gt_depth_maps(gt_depth, downsample=downsample)
    if gt_depth is None:
        return None

    depth_channels = _infer_depth_channels(
        pred_depth=None, depth_bin_config=depth_bin_config
    ) if depth_channels is None else int(depth_channels)
    if depth_channels is None or depth_channels <= 0:
        return [
            np.asarray(gt_depth[index], dtype=np.float32)
            for index in range(gt_depth.shape[0])
        ]

    depth_values = _depth_bin_values(depth_channels, depth_bin_config)
    if depth_values.shape[0] != depth_channels:
        return None

    if depth_bin_config is None or len(depth_bin_config) < 3:
        depth_maps = gt_depth.astype(np.float32)
    else:
        depth_min = float(depth_bin_config[0])
        depth_max = float(depth_bin_config[1])
        depth_step = float(depth_bin_config[2])
        if not sid:
            depth_labels = (gt_depth -
                            (depth_min - depth_step)) / max(depth_step, 1e-6)
        else:
            safe_gt = np.maximum(gt_depth, 1e-6)
            sid_scale = np.log(
                max((depth_max - 1.0) / max(depth_min, 1e-6), 1.0 + 1e-6))
            depth_labels = np.log(safe_gt) - np.log(max(depth_min, 1e-6))
            if depth_channels > 1:
                depth_labels = depth_labels * (depth_channels - 1) / max(
                    sid_scale, 1e-6)
            depth_labels = depth_labels + 1.0

        valid = (depth_labels < depth_channels + 1) & (depth_labels >= 0.0)
        depth_labels = np.where(valid, depth_labels, 0.0).astype(np.int64)
        depth_maps = np.zeros_like(gt_depth, dtype=np.float32)
        valid = depth_labels > 0
        if np.any(valid):
            depth_maps[valid] = depth_values[np.clip(depth_labels[valid] - 1,
                                                     0, depth_channels - 1)]

    if target_shape is not None and tuple(
            depth_maps.shape[1:]) != tuple(target_shape):
        resized_maps = []
        target_h, target_w = int(target_shape[0]), int(target_shape[1])
        for index in range(depth_maps.shape[0]):
            resized_maps.append(
                cv2.resize(
                    depth_maps[index], (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST))
        depth_maps = np.stack(resized_maps, axis=0)

    return [
        np.asarray(depth_maps[index], dtype=np.float32)
        for index in range(depth_maps.shape[0])
    ]


def _render_depth_panel(depth_map,
                        panel_label,
                        panel_size=(704, 256),
                        depth_min=None,
                        depth_max=None):
    depth_map = np.asarray(depth_map, dtype=np.float32)
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    if depth_min is None:
        depth_min = float(np.min(
            depth_map[valid_mask])) if np.any(valid_mask) else 0.0
    if depth_max is None:
        depth_max = float(np.max(
            depth_map[valid_mask])) if np.any(valid_mask) else depth_min + 1.0
    depth_max = max(depth_max, depth_min + 1e-6)

    normalized = np.zeros_like(depth_map, dtype=np.uint8)
    if np.any(valid_mask):
        clipped = np.clip((depth_map - depth_min) / (depth_max - depth_min),
                          0.0, 1.0)
        normalized[valid_mask] = (clipped[valid_mask] * 255).astype(np.uint8)

    panel = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    panel[~valid_mask] = np.array([235, 235, 235], dtype=np.uint8)
    panel = cv2.resize(panel, panel_size, interpolation=cv2.INTER_NEAREST)
    cv2.putText(
        panel,
        str(panel_label),
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _compose_quad_grid(panels):
    panels = list(panels)
    if not panels:
        return None
    while len(panels) < 4:
        panels.append(np.full_like(panels[0], 235))
    return cv2.vconcat([
        cv2.hconcat(panels[:2]),
        cv2.hconcat(panels[2:4]),
    ])


def _compose_panel_row(panels, title=None):
    panels = [panel for panel in panels if panel is not None]
    if not panels:
        return None
    row = cv2.hconcat(panels)
    if not title:
        return row
    return add_info_banner(row, [str(title)])


def _compose_panel_grid(panels, title=None):
    grid = _compose_quad_grid(panels)
    if grid is None:
        return None
    if not title:
        return grid
    return _draw_panel_title(grid, title)


def render_depth_grid(pred_depth,
                      gt_depth,
                      img_inputs=None,
                      sample_index=0,
                      cam_names=None,
                      depth_bin_config=None,
                      camera_num_frame=1,
                      panel_size=None,
                      gt_depth_mode='raw',
                      gt_depth_downsample=1,
                      gt_depth_sid=False,
                      max_panels=4):
    pred_depth = _select_pred_depth_sample(
        pred_depth,
        gt_depth=gt_depth,
        img_inputs=img_inputs,
        sample_index=sample_index)
    gt_depth = _to_numpy(gt_depth)

    if gt_depth is not None and gt_depth.ndim == 4:
        gt_depth = gt_depth[sample_index]

    pred_panel_maps = None
    if pred_depth is not None:
        if pred_depth.ndim == 3:
            pred_depth = pred_depth[None, ...]
        if pred_depth.ndim == 4:
            pred_panel_maps = [
                _pred_depth_to_map(pred_depth[index], depth_bin_config)
                for index in range(pred_depth.shape[0])
            ]

    gt_panel_maps_raw = None
    gt_panel_maps_loss = None
    if gt_depth is not None:
        if gt_depth.ndim == 2:
            gt_depth = gt_depth[None, ...]
        if gt_depth.ndim == 3:
            if gt_depth_mode in ('raw', 'both'):
                gt_panel_maps_raw = [
                    np.asarray(gt_depth[index], dtype=np.float32)
                    for index in range(gt_depth.shape[0])
                ]
            if gt_depth_mode in ('loss', 'both'):
                target_shape = None
                depth_channels = _infer_depth_channels(
                    pred_depth=pred_depth, depth_bin_config=depth_bin_config)
                if pred_depth is not None:
                    if pred_depth.ndim == 4:
                        target_shape = pred_depth.shape[-2:]
                    elif pred_depth.ndim == 3:
                        target_shape = pred_depth.shape[-2:]
                gt_panel_maps_loss = _loss_aligned_gt_depth_maps(
                    gt_depth,
                    depth_bin_config=depth_bin_config,
                    downsample=gt_depth_downsample,
                    sid=gt_depth_sid,
                    depth_channels=depth_channels,
                    target_shape=target_shape)

    image_panels = _render_camera_panels(
        img_inputs,
        sample_index=sample_index,
        camera_num_frame=camera_num_frame,
        cam_names=cam_names,
        panel_size=panel_size,
        label_suffix='Image',
        max_panels=max_panels)

    camera_count = max(
        len(image_panels),
        0 if pred_panel_maps is None else len(pred_panel_maps),
        0 if gt_panel_maps_raw is None else len(gt_panel_maps_raw),
        0 if gt_panel_maps_loss is None else len(gt_panel_maps_loss),
        0 if cam_names is None else len(cam_names))
    camera_count = min(max(camera_count, 0), max(int(max_panels), 1))
    if camera_count == 0:
        return None

    depth_min = None
    depth_max = None
    if depth_bin_config is not None and len(depth_bin_config) >= 2:
        depth_min = float(depth_bin_config[0])
        depth_max = float(depth_bin_config[1])

    if panel_size is None:
        if image_panels:
            panel_size = (int(image_panels[0].shape[1]),
                          int(image_panels[0].shape[0]))
        else:
            panel_maps = pred_panel_maps
            if panel_maps is None:
                panel_maps = gt_panel_maps_loss
            if panel_maps is None:
                panel_maps = gt_panel_maps_raw
            if panel_maps is None or panel_maps[0] is None:
                return None
            panel_size = (int(panel_maps[0].shape[1]),
                          int(panel_maps[0].shape[0]))

    labels = cam_names or [f'View {index}' for index in range(camera_count)]

    def get_image_panel(panel_index):
        label = labels[panel_index] if panel_index < len(
            labels) else f'View {panel_index}'
        if panel_index < len(image_panels):
            return image_panels[panel_index]
        return _make_labeled_placeholder(panel_size, f'{label} | Image')

    def get_depth_panel(panel_maps, panel_index, suffix):
        label = labels[panel_index] if panel_index < len(
            labels) else f'View {panel_index}'
        if panel_maps is None or panel_index >= len(
                panel_maps) or panel_maps[panel_index] is None:
            return _make_labeled_placeholder(panel_size, f'{label} | {suffix}')
        return _render_depth_panel(
            panel_maps[panel_index],
            f'{label} | {suffix}',
            panel_size=panel_size,
            depth_min=depth_min,
            depth_max=depth_max)

    image_rows = []
    gt_raw_rows = []
    gt_loss_rows = []
    pred_rows = []
    columns = 3 if camera_count > 4 else 2
    for row_start in range(0, camera_count, columns):
        image_row = []
        gt_raw_row = []
        gt_loss_row = []
        pred_row = []
        for panel_index in range(row_start,
                                 min(row_start + columns, camera_count)):
            image_row.append(get_image_panel(panel_index))
            if gt_panel_maps_raw is not None:
                gt_raw_row.append(
                    get_depth_panel(gt_panel_maps_raw, panel_index, 'GT(raw)'))
            if gt_panel_maps_loss is not None:
                gt_loss_row.append(
                    get_depth_panel(gt_panel_maps_loss, panel_index,
                                    'GT(loss)'))
            pred_row.append(
                get_depth_panel(pred_panel_maps, panel_index, 'Pred'))
        while len(image_row) < columns:
            pad_index = row_start + len(image_row)
            label = labels[pad_index] if pad_index < len(
                labels) else f'View {pad_index}'
            image_row.append(
                _make_labeled_placeholder(panel_size, f'{label} | Image'))
            if gt_panel_maps_raw is not None:
                gt_raw_row.append(
                    _make_labeled_placeholder(panel_size,
                                              f'{label} | GT(raw)'))
            if gt_panel_maps_loss is not None:
                gt_loss_row.append(
                    _make_labeled_placeholder(panel_size,
                                              f'{label} | GT(loss)'))
            pred_row.append(
                _make_labeled_placeholder(panel_size, f'{label} | Pred'))
        image_rows.append(cv2.hconcat(image_row))
        if gt_raw_row:
            gt_raw_rows.append(cv2.hconcat(gt_raw_row))
        if gt_loss_row:
            gt_loss_rows.append(cv2.hconcat(gt_loss_row))
        pred_rows.append(cv2.hconcat(pred_row))
    row_images = image_rows + gt_raw_rows + gt_loss_rows + pred_rows
    if not row_images:
        return None
    return cv2.vconcat(row_images)


def _project_points_to_canvas(points_xy, point_cloud_range, image_size):
    width, height = image_size
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    mask = (
        np.isfinite(points_xy[:, 0]) & np.isfinite(points_xy[:, 1]) &
        (points_xy[:, 0] >= x_min) & (points_xy[:, 0] <= x_max) &
        (points_xy[:, 1] >= y_min) & (points_xy[:, 1] <= y_max))
    if not np.any(mask):
        return np.empty((0, ), dtype=np.int64), np.empty((0, ),
                                                         dtype=np.int64), mask

    points_xy = points_xy[mask]
    px = (points_xy[:, 0] - x_min) / max(x_max - x_min, 1e-6)
    py = (points_xy[:, 1] - y_min) / max(y_max - y_min, 1e-6)
    # Use cell/bin scaling instead of endpoint scaling so point BEV aligns better
    # with occupancy grids that are visualized from discretized voxel cells.
    px = np.clip((px * width).astype(np.int32), 0, width - 1)
    py = np.clip((py * height).astype(np.int32), 0, height - 1)
    px = width - 1 - px
    py = height - 1 - py
    return px, py, mask


def render_points_bev(points,
                      point_cloud_range,
                      image_size=DEFAULT_IMAGE_SIZE,
                      raster_size=None):
    points = _to_numpy(points)
    if points is None or points.size == 0:
        return None
    canvas_size = raster_size or image_size
    canvas = np.full((canvas_size[1], canvas_size[0], 3), 255, dtype=np.uint8)
    px, py, mask = _project_points_to_canvas(points[:, :2], point_cloud_range,
                                             canvas_size)
    if px.size == 0:
        if canvas_size != image_size:
            return cv2.resize(
                canvas, image_size, interpolation=cv2.INTER_NEAREST)
        return canvas
    points_xy = points[mask, :2]
    dist = np.linalg.norm(points_xy, axis=1)
    intensity = np.clip(255 - (dist / max(np.max(dist), 1e-6)) * 180, 40,
                        255).astype(np.uint8)
    colors = np.stack([intensity, intensity, intensity], axis=1)
    canvas[py, px] = colors
    if canvas_size != image_size:
        canvas = cv2.resize(
            canvas, image_size, interpolation=cv2.INTER_NEAREST)
    return canvas


def render_history_points_bev(points,
                              frame_splits,
                              point_cloud_range,
                              image_size=DEFAULT_IMAGE_SIZE,
                              raster_size=None):
    points = _to_numpy(points)
    if points is None or points.size == 0 or not frame_splits or len(
            frame_splits) <= 1:
        return None

    canvas_size = raster_size or image_size
    canvas = np.full((canvas_size[1], canvas_size[0], 3), 255, dtype=np.uint8)
    palette = [
        (30, 30, 220),
        (60, 180, 60),
        (220, 120, 30),
        (160, 60, 180),
    ]
    start = 0
    for frame_index, count in enumerate(frame_splits):
        frame_points = points[start:start + int(count)]
        start += int(count)
        if frame_points.size == 0:
            continue
        px, py, _ = _project_points_to_canvas(frame_points[:, :2],
                                              point_cloud_range, canvas_size)
        if px.size == 0:
            continue
        color = palette[min(frame_index, len(palette) - 1)]
        canvas[py, px] = np.array(color, dtype=np.uint8)
    if canvas_size != image_size:
        canvas = cv2.resize(
            canvas, image_size, interpolation=cv2.INTER_NEAREST)
    return canvas


def _project_bev_feature_map(voxel_feat):
    voxel_feat = _to_numpy(voxel_feat)
    if voxel_feat is None:
        return None
    if voxel_feat.ndim == 5:
        voxel_feat = voxel_feat[0]
    if voxel_feat.ndim == 4:
        feat_map = np.abs(voxel_feat).max(axis=0).max(axis=0)
    elif voxel_feat.ndim == 3:
        feat_map = np.abs(voxel_feat).max(axis=0)
    else:
        return None
    feat_map = feat_map.astype(np.float32)
    # Voxel/BEV feature maps are reduced to [y, x] before visualization.
    feat_map = _orient_yx_grid_to_image(feat_map)
    return feat_map


def _normalize_feature_map(feat_map):
    feat_map = np.asarray(feat_map, dtype=np.float32)
    feat_map -= feat_map.min()
    if feat_map.max() > 0:
        feat_map /= feat_map.max()
    return feat_map


def render_bev_feature(voxel_feat, image_size=DEFAULT_IMAGE_SIZE):
    feat_map = _project_bev_feature_map(voxel_feat)
    if feat_map is None:
        return None
    feat_map = _normalize_feature_map(feat_map)
    feat_map = (feat_map * 255).astype(np.uint8)
    feat_map = cv2.resize(feat_map, image_size, interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(feat_map, cv2.COLORMAP_VIRIDIS)


def _scale_sort_key(name):
    if name is None:
        return 0
    match = re.search(r'1_(\d+)', str(name))
    if match:
        return -int(match.group(1))
    return -1 if str(name).lower() == 'final' else 0


def _format_scale_label(name):
    if name is None:
        return 'unknown'
    if str(name).lower() == 'final':
        return 'final'
    match = re.search(r'1_(\d+)', str(name))
    if match:
        return f'1/{match.group(1)}'
    return str(name)


def _select_sample_feature(value, sample_index):
    value = _to_numpy(value)
    if value is None:
        return None
    if value.ndim >= 4 and value.shape[0] <= 32:
        if sample_index >= value.shape[0]:
            return None
        value = value[sample_index]
    return value


def _feature_collection_items(feature_collection):
    if feature_collection is None:
        return []
    if isinstance(feature_collection, dict):
        items = list(feature_collection.items())
    elif isinstance(feature_collection, (list, tuple)):
        items = [(f'1_{2 ** (index + 1)}', value)
                 for index, value in enumerate(feature_collection)]
    else:
        items = [('feature', feature_collection)]
    return sorted(items, key=lambda item: _scale_sort_key(item[0]))


def _stack_section_images(section_images):
    section_images = [image for image in section_images if image is not None]
    if not section_images:
        return None
    if len(section_images) == 1:
        return section_images[0]

    width = max(image.shape[1] for image in section_images)
    stacked = []
    for image in section_images:
        if image.shape[1] != width:
            image = cv2.copyMakeBorder(
                image,
                0,
                0,
                0,
                width - image.shape[1],
                cv2.BORDER_CONSTANT,
                value=(235, 235, 235))
        stacked.append(image)
        stacked.append(np.full((12, width, 3), 235, dtype=np.uint8))
    stacked.pop()
    return cv2.vconcat(stacked)


def render_feature_collection_grid(feature_collection,
                                   title,
                                   sample_index=0,
                                   image_size=(320, 320)):
    panels = []
    for scale_name, feature in _feature_collection_items(feature_collection):
        feature = _select_sample_feature(feature, sample_index)
        panel = render_bev_feature(feature, image_size=image_size)
        if panel is None:
            continue
        panels.append(
            _draw_panel_title(panel, _format_scale_label(scale_name)))
    if not panels:
        return None
    return _compose_panel_grid(panels, title)


def render_feature_collection_summary(feature_collection,
                                      sample_index=0,
                                      image_size=(320, 320)):
    summary_maps = []
    for _, feature in _feature_collection_items(feature_collection):
        feature = _select_sample_feature(feature, sample_index)
        feat_map = _project_bev_feature_map(feature)
        if feat_map is None:
            continue
        feat_map = _normalize_feature_map(feat_map)
        feat_map = cv2.resize(
            feat_map, image_size, interpolation=cv2.INTER_LINEAR)
        summary_maps.append(feat_map)

    if not summary_maps:
        return None

    summary_map = np.mean(np.stack(summary_maps, axis=0), axis=0)
    summary_map = _normalize_feature_map(summary_map)
    summary_map = (summary_map * 255).astype(np.uint8)
    return cv2.applyColorMap(summary_map, cv2.COLORMAP_VIRIDIS)


def _to_stage_occ_labels(value, sample_index):
    value = _to_numpy(value)
    if value is None:
        return None
    if value.ndim == 5:
        if sample_index >= value.shape[0]:
            return None
        value = value[sample_index]
    if value.ndim == 4:
        # Float 4D tensors are stage logits/probabilities [X, Y, Z, C].
        # Integer 4D tensors are batched labels [B, X, Y, Z].
        if np.issubdtype(value.dtype, np.floating):
            value = value.argmax(axis=-1)
        else:
            if sample_index >= value.shape[0]:
                return None
            value = value[sample_index]
    if value.ndim != 3:
        return None
    return np.asarray(value).astype(np.uint8)


def _to_stage_flow_volume(value, sample_index):
    value = _to_numpy(value)
    if value is None:
        return None
    if value.ndim == 5:
        if sample_index >= value.shape[0]:
            return None
        value = value[sample_index]
    if value.ndim != 4 or value.shape[-1] != 2:
        return None
    return np.asarray(value, dtype=np.float32)


def render_occ_collection_grid(occ_collection,
                               empty_idx,
                               color_map,
                               sample_index=0,
                               image_size=(320, 320),
                               title=None):
    panels = []
    for scale_name, occ_value in _feature_collection_items(occ_collection):
        occ_labels = _to_stage_occ_labels(occ_value, sample_index)
        panel = render_occ_bev(
            occ_labels, empty_idx, color_map, image_size=image_size)
        if panel is None:
            continue
        panels.append(
            _draw_panel_title(panel, _format_scale_label(scale_name)))
    if not panels:
        return None
    return _compose_panel_grid(panels, title)


def _to_prior_static_logits(value, sample_index):
    value = _to_numpy(value)
    if value is None:
        return None
    if value.ndim == 5:
        if sample_index >= value.shape[0]:
            return None
        value = value[sample_index]
    if value.ndim != 4:
        return None
    return np.asarray(value, dtype=np.float32)


def render_scalar_volume_bev(volume, image_size=DEFAULT_IMAGE_SIZE):
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        return None
    scalar_bev = np.max(volume, axis=2)
    if not np.all(np.isfinite(scalar_bev)):
        scalar_bev = np.nan_to_num(scalar_bev, nan=0.0, posinf=0.0, neginf=0.0)
    scalar_bev = scalar_bev - scalar_bev.min()
    max_value = float(scalar_bev.max())
    if max_value > 0:
        scalar_bev = scalar_bev / max_value
    scalar_bev = np.round(scalar_bev * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(scalar_bev, cv2.COLORMAP_VIRIDIS)
    heatmap = _orient_xy_grid_to_image(heatmap)
    return cv2.resize(heatmap, image_size, interpolation=cv2.INTER_NEAREST)


def render_scalar_volume_bev_fixed(volume,
                                   image_size=DEFAULT_IMAGE_SIZE,
                                   value_min=0.0,
                                   value_max=1.0):
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        return None
    scalar_bev = np.max(volume, axis=2)
    if not np.all(np.isfinite(scalar_bev)):
        scalar_bev = np.nan_to_num(scalar_bev, nan=0.0, posinf=0.0, neginf=0.0)
    value_min = float(value_min)
    value_max = max(float(value_max), value_min + 1e-6)
    scalar_bev = np.clip((scalar_bev - value_min) / (value_max - value_min),
                         0.0, 1.0)
    scalar_bev = np.round(scalar_bev * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(scalar_bev, cv2.COLORMAP_VIRIDIS)
    heatmap = _orient_xy_grid_to_image(heatmap)
    return cv2.resize(heatmap, image_size, interpolation=cv2.INTER_NEAREST)


def _compute_label_run_length(labels, axis, reverse=False):
    """Compute one VoxNT direction without requiring instance annotations."""

    labels = np.asarray(labels)
    if reverse:
        flipped = np.flip(labels, axis=axis)
        distances = _compute_label_run_length(flipped, axis, reverse=False)
        return np.flip(distances, axis=axis)
    distances = np.ones(labels.shape, dtype=np.float32)
    length = labels.shape[axis]
    for index in range(length - 2, -1, -1):
        current = [slice(None)] * labels.ndim
        following = [slice(None)] * labels.ndim
        current[axis] = index
        following[axis] = index + 1
        distances[tuple(current)] = np.where(
            labels[tuple(current)] == labels[tuple(following)],
            distances[tuple(following)] + 1.0, 1.0)
    return distances


def _compute_voxdet_gt_offsets(gt_occ, target_shape, dynamic_class_indices):
    """Build normalized dynamic VoxNT targets at the prediction resolution."""

    gt_occ = np.asarray(gt_occ)
    if gt_occ.ndim != 3 or not dynamic_class_indices:
        return None, None
    offsets = np.stack([
        _compute_label_run_length(gt_occ, 0, reverse=False),
        _compute_label_run_length(gt_occ, 0, reverse=True),
        _compute_label_run_length(gt_occ, 1, reverse=False),
        _compute_label_run_length(gt_occ, 1, reverse=True),
        _compute_label_run_length(gt_occ, 2, reverse=False),
        _compute_label_run_length(gt_occ, 2, reverse=True),
    ],
                       axis=0).astype(np.float32)
    offsets[0:2] /= max(gt_occ.shape[0], 1)
    offsets[2:4] /= max(gt_occ.shape[1], 1)
    offsets[4:6] /= max(gt_occ.shape[2], 1)
    dynamic_mask = np.isin(gt_occ, np.asarray(dynamic_class_indices))

    if tuple(offsets.shape[1:]) != tuple(target_shape):
        offsets_tensor = torch.from_numpy(offsets).unsqueeze(0)
        offsets = F.interpolate(
            offsets_tensor,
            size=tuple(int(value) for value in target_shape),
            mode='nearest').squeeze(0).numpy()
        mask_tensor = torch.from_numpy(dynamic_mask.astype(
            np.float32)).unsqueeze(0).unsqueeze(0)
        dynamic_mask = F.interpolate(
            mask_tensor,
            size=tuple(int(value) for value in target_shape),
            mode='nearest').squeeze(0).squeeze(0).numpy() > 0.5
    return offsets, dynamic_mask


def _to_voxdet_offsets(value, sample_index):
    value = _to_numpy(value)
    if value is None:
        return None
    if value.ndim == 5:
        if sample_index >= value.shape[0]:
            return None
        value = value[sample_index]
    if value.ndim != 4 or value.shape[0] != 6:
        return None
    return np.asarray(value, dtype=np.float32)


def _render_voxdet_regression_row(volumes,
                                  titles,
                                  image_size=(260, 260),
                                  value_max=1.0,
                                  value_maxes=None,
                                  value_unit='vox',
                                  display_gamma=0.5):
    panels = []
    if value_maxes is None:
        value_maxes = [value_max] * len(volumes)
    for index, (volume, title) in enumerate(zip(volumes, titles)):
        panel_max = float(value_maxes[index])
        # Keep one physical scale for Pred/GT/Err, but use a concave display
        # mapping so small dynamic-object offsets remain visible.
        display_volume = np.clip(volume / max(panel_max, 1e-6), 0.0, 1.0)
        display_volume = np.power(display_volume, float(display_gamma))
        panel = render_scalar_volume_bev_fixed(
            display_volume,
            image_size=image_size,
            value_min=0.0,
            value_max=1.0)
        if panel is None:
            panel = _create_placeholder_panel(image_size,
                                              'regression unavailable')
        panels.append(
            _draw_panel_title(panel,
                              f'{title} | 0-{panel_max:.0f} {value_unit}'))
    while len(panels) < 3:
        panels.append(_create_placeholder_panel(image_size, 'unused'))
    return cv2.hconcat(panels[:3])


def _largest_dynamic_component_bounds(dynamic_mask, padding=6):
    """Return a readable local ROI around the largest dynamic component."""

    mask_bev = np.asarray(dynamic_mask).any(axis=2).astype(np.uint8)
    if not np.any(mask_bev):
        return None
    _, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bev, connectivity=8)
    if stats.shape[0] <= 1:
        return None
    component_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[component_index, cv2.CC_STAT_LEFT])
    y = int(stats[component_index, cv2.CC_STAT_TOP])
    width = int(stats[component_index, cv2.CC_STAT_WIDTH])
    height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
    x0 = max(0, x - int(padding))
    y0 = max(0, y - int(padding))
    x1 = min(mask_bev.shape[0], x + width + int(padding))
    y1 = min(mask_bev.shape[1], y + height + int(padding))
    return x0, x1, y0, y1


def render_voxdet_regression_grid(feature_payload,
                                  gt_occ,
                                  dynamic_class_indices=None,
                                  sample_index=0,
                                  image_size=(260, 260),
                                  value_max=None,
                                  zoom=False):
    """Render readable dynamic-only VoxDet regression maps in BEV.

    Distances are shown in voxel units rather than the training normalization.
    X/Y and Z use separate fixed ranges because their normalized scales differ
    substantially for a ``[320, 320, 16]`` occupancy volume. Predictions remain
    masked to GT dynamic voxels, so this view diagnoses regression values rather
    than false positives or semantic recall.
    """

    if not isinstance(feature_payload, dict):
        return None
    pred_offsets = _to_voxdet_offsets(
        feature_payload.get('voxdet_regression'), sample_index)
    if pred_offsets is None:
        return None
    regression_class_indices = feature_payload.get('voxdet_reg_class_indices',
                                                   dynamic_class_indices)
    dynamic_class_indices = tuple(regression_class_indices or ())
    gt_offsets, dynamic_mask = _compute_voxdet_gt_offsets(
        gt_occ, pred_offsets.shape[1:], dynamic_class_indices)
    if gt_offsets is None or dynamic_mask is None:
        return None

    pred_offsets = np.where(dynamic_mask[None], pred_offsets, 0.0)
    gt_offsets = np.where(dynamic_mask[None], gt_offsets, 0.0)
    abs_error = np.abs(pred_offsets - gt_offsets)

    # The head predicts distances normalized by the output volume shape. Bring
    # them back to voxel units so small X/Y extents are not hidden by Z's much
    # larger normalized values. The same conversion is used for Pred, GT, and
    # AbsErr to preserve a direct visual comparison.
    volume_shape = np.asarray(pred_offsets.shape[1:], dtype=np.float32)
    direction_scales = np.asarray([
        volume_shape[0],
        volume_shape[0],
        volume_shape[1],
        volume_shape[1],
        volume_shape[2],
        volume_shape[2],
    ],
                                  dtype=np.float32).reshape(6, 1, 1, 1)
    pred_offsets = pred_offsets * direction_scales
    gt_offsets = gt_offsets * direction_scales
    abs_error = abs_error * direction_scales

    if value_max is None:
        # Fixed scales keep comparisons meaningful across samples while
        # matching the observed validation-set dynamic-distance quantiles.
        direction_maxes = np.asarray([24.0, 24.0, 24.0, 24.0, 12.0, 12.0],
                                     dtype=np.float32)
        extent_maxes = np.asarray([48.0, 48.0, 24.0], dtype=np.float32)
    else:
        direction_maxes = np.full(6, float(value_max), dtype=np.float32)
        extent_maxes = np.full(3, float(value_max), dtype=np.float32)
    if zoom:
        bounds = _largest_dynamic_component_bounds(dynamic_mask)
        if bounds is not None:
            x0, x1, y0, y1 = bounds
            pred_offsets = pred_offsets[:, x0:x1, y0:y1, :]
            gt_offsets = gt_offsets[:, x0:x1, y0:y1, :]
            abs_error = abs_error[:, x0:x1, y0:y1, :]
    direction_names = ('x+', 'x-', 'y+', 'y-', 'z+', 'z-')
    pred_direction_rows = []
    gt_direction_rows = []
    for start in (0, 3):
        pred_direction_rows.append(
            _render_voxdet_regression_row(
                [pred_offsets[index] for index in range(start, start + 3)], [
                    f'P@GTdyn {name}'
                    for name in direction_names[start:start + 3]
                ],
                image_size=image_size,
                value_maxes=direction_maxes[start:start + 3],
                value_unit='vox'))
        gt_direction_rows.append(
            _render_voxdet_regression_row(
                [gt_offsets[index] for index in range(start, start + 3)],
                [f'GT {name}' for name in direction_names[start:start + 3]],
                image_size=image_size,
                value_maxes=direction_maxes[start:start + 3],
                value_unit='vox'))

    pred_extents = [
        pred_offsets[0] + pred_offsets[1] - 1.0,
        pred_offsets[2] + pred_offsets[3] - 1.0,
        pred_offsets[4] + pred_offsets[5] - 1.0,
    ]
    gt_extents = [
        gt_offsets[0] + gt_offsets[1] - 1.0,
        gt_offsets[2] + gt_offsets[3] - 1.0,
        gt_offsets[4] + gt_offsets[5] - 1.0,
    ]
    error_extents = [
        np.abs(pred_extents[index] - gt_extents[index]) for index in range(3)
    ]
    rows = pred_direction_rows + gt_direction_rows
    error_direction_rows = []
    for start in (0, 3):
        error_direction_rows.append(
            _render_voxdet_regression_row(
                [abs_error[index] for index in range(start, start + 3)],
                [f'Err {name}' for name in direction_names[start:start + 3]],
                image_size=image_size,
                value_maxes=direction_maxes[start:start + 3],
                value_unit='vox'))
    rows.extend(error_direction_rows)
    rows.append(
        _render_voxdet_regression_row(
            pred_extents, ['P X size', 'P Y size', 'P Z size'],
            image_size=image_size,
            value_maxes=extent_maxes,
            value_unit='vox'))
    rows.append(
        _render_voxdet_regression_row(
            gt_extents, ['GT X size', 'GT Y size', 'GT Z size'],
            image_size=image_size,
            value_maxes=extent_maxes,
            value_unit='vox'))
    rows.append(
        _render_voxdet_regression_row(
            error_extents, ['Err X size', 'Err Y size', 'Err Z size'],
            image_size=image_size,
            value_maxes=extent_maxes,
            value_unit='vox'))
    return cv2.vconcat(rows)


def render_dynamic_candidate_volume_bev(volume,
                                        image_size=DEFAULT_IMAGE_SIZE,
                                        value_max=0.1):
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        return None
    scalar_bev = np.max(volume, axis=2)
    if not np.all(np.isfinite(scalar_bev)):
        scalar_bev = np.nan_to_num(scalar_bev, nan=0.0, posinf=0.0, neginf=0.0)
    scalar_bev = np.clip(scalar_bev, 0.0, float(value_max))
    scalar_bev = _orient_xy_grid_to_image(scalar_bev)
    scalar_bev = cv2.resize(
        scalar_bev,
        image_size,
        interpolation=cv2.INTER_LINEAR,
    )
    scalar_bev = cv2.GaussianBlur(scalar_bev, (0, 0), sigmaX=2.0, sigmaY=2.0)
    positive = scalar_bev[scalar_bev > 0]
    if positive.size > 0:
        lower = float(np.percentile(positive, 90))
        upper = float(np.percentile(positive, 99))
        upper = max(upper, lower + 1e-6)
        scalar_bev = np.clip((scalar_bev - lower) / (upper - lower), 0.0, 1.0)
    elif value_max > 0:
        scalar_bev = np.clip(scalar_bev / float(value_max), 0.0, 1.0)
    scalar_bev = np.round(scalar_bev * 255.0).astype(np.uint8)
    return cv2.applyColorMap(scalar_bev, cv2.COLORMAP_VIRIDIS)


def render_prior_static_bias_grid(feature_payload,
                                  color_map,
                                  empty_idx,
                                  sample_index=0,
                                  image_size=(320, 320)):
    if not isinstance(feature_payload, dict):
        feature_payload = {}
    logits_collection = feature_payload.get('prior_static_logits') or {}
    alpha_collection = feature_payload.get('prior_static_alpha') or {}
    prior_meta = feature_payload.get('prior_static_meta') or {}
    static_class_indices = tuple(prior_meta.get('static_class_indices') or ())
    if not static_class_indices:
        return None
    static_class_indices = np.asarray(static_class_indices, dtype=np.int64)
    panels = []
    for scale_name, logits_value in _feature_collection_items(
            logits_collection):
        prior_static_logits = _to_prior_static_logits(logits_value,
                                                      sample_index)
        if prior_static_logits is None or prior_static_logits.shape[-1] < 2:
            continue
        static_logits = prior_static_logits[..., :len(static_class_indices)]
        nonstatic_logits = prior_static_logits[..., len(static_class_indices)]
        static_labels = np.argmax(static_logits, axis=-1)
        static_scores = np.max(static_logits, axis=-1)
        prior_occ_labels = np.full(
            static_labels.shape, fill_value=empty_idx, dtype=np.int64)
        static_mask = static_scores > nonstatic_logits
        prior_occ_labels[static_mask] = static_class_indices[
            static_labels[static_mask]]
        prior_occ_panel = render_occ_bev(
            prior_occ_labels, empty_idx, color_map, image_size=image_size)
        if prior_occ_panel is None:
            continue
        alpha = alpha_collection.get(scale_name, None)
        alpha_text = f'alpha={float(alpha):.3f}' if alpha is not None else 'alpha=?'
        panels.append(
            _draw_panel_title(
                prior_occ_panel,
                f'{_format_scale_label(scale_name)} | {alpha_text}'))
    if not panels:
        return None
    return _compose_panel_grid(panels, title=None)


def render_static_occ_grid(feature_payload,
                           empty_idx,
                           color_map,
                           sample_index=0,
                           image_size=(320, 320)):
    if not isinstance(feature_payload, dict):
        feature_payload = {}
    occ_collection = dict(
        feature_payload.get('stage_static_occ_predictions') or {})
    occ_collection.pop('1_1', None)
    return render_occ_collection_grid(
        occ_collection,
        empty_idx=empty_idx,
        color_map=color_map,
        sample_index=sample_index,
        image_size=image_size,
        title=None,
    )


def _to_stage_scalar_volume(value, sample_index):
    value = _to_numpy(value)
    if value is None:
        return None
    if value.ndim == 5:
        if sample_index >= value.shape[0]:
            return None
        value = value[sample_index]
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 3:
        return None
    return np.asarray(value, dtype=np.float32)


def render_scalar_collection_grid(volume_collection,
                                  sample_index=0,
                                  image_size=(320, 320),
                                  title=None):
    panels = []
    for scale_name, volume_value in _feature_collection_items(
            volume_collection):
        volume = _to_stage_scalar_volume(volume_value, sample_index)
        panel = render_scalar_volume_bev(volume, image_size=image_size)
        if panel is None:
            continue
        panels.append(
            _draw_panel_title(panel, _format_scale_label(scale_name)))
    if not panels:
        return None
    return _compose_panel_grid(panels, title)


def render_dynamic_candidate_grid(feature_payload,
                                  sample_index=0,
                                  image_size=(320, 320)):
    if not isinstance(feature_payload, dict):
        feature_payload = {}
    volume_collection = dict(
        feature_payload.get('stage_dynamic_candidate_maps') or {})
    panels = []
    for scale_name, volume_value in _feature_collection_items(
            volume_collection):
        volume = _to_stage_scalar_volume(volume_value, sample_index)
        panel = render_dynamic_candidate_volume_bev(
            volume, image_size=image_size)
        if panel is None:
            continue
        panels.append(
            _draw_panel_title(panel, _format_scale_label(scale_name)))
    if not panels:
        return None
    return _compose_panel_grid(panels, title=None)


def _render_placeholder_panel(image_size, title, message):
    panel = np.full(
        (int(image_size[1]), int(image_size[0]), 3),
        _PLACEHOLDER_BG,
        dtype=np.uint8,
    )
    cv2.rectangle(
        panel,
        (0, 0),
        (panel.shape[1] - 1, panel.shape[0] - 1),
        (180, 180, 180),
        1,
    )
    cv2.putText(
        panel,
        str(message),
        (16, max(54, panel.shape[0] // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        _PLACEHOLDER_FG,
        2,
        cv2.LINE_AA,
    )
    return _draw_panel_title(panel, title)


def render_current_dynamic_candidate_panel(feature_payload,
                                           sample_index=0,
                                           image_size=(320, 320)):
    if not isinstance(feature_payload, dict):
        return None
    volume_collection = dict(
        feature_payload.get('stage_dynamic_candidate_maps') or {})
    selected = None
    for scale_name, volume_value in _feature_collection_items(
            volume_collection):
        volume = _to_stage_scalar_volume(volume_value, sample_index)
        if volume is None:
            continue
        selected = (scale_name, volume)
    if selected is None:
        return None
    scale_name, volume = selected
    panel = render_dynamic_candidate_volume_bev(volume, image_size=image_size)
    if panel is None:
        return None
    return _draw_panel_title(
        panel, f'Candidate Map | {_format_scale_label(scale_name)}')


def render_temporal_topk_grid(feature_payload,
                              sample_meta=None,
                              gt_occ=None,
                              empty_idx=None,
                              color_map=None,
                              dynamic_class_indices=None,
                              sample_index=0,
                              image_size=(320, 320)):
    if not isinstance(feature_payload, dict):
        feature_payload = {}
    guidance_collection = dict(
        feature_payload.get('stage_temporal_fusion_guidance_maps') or {})
    mask_collection = dict(
        feature_payload.get('stage_temporal_fusion_fg_masks') or {})
    true_occ_collection = dict(
        feature_payload.get('true_stage_voxel_semantic') or {})

    panels = []
    for scale_name, guidance_value in _feature_collection_items(
            guidance_collection):
        guidance_volume = _to_stage_scalar_volume(guidance_value, sample_index)
        if guidance_volume is None:
            continue
        guidance_panel = render_scalar_volume_bev_fixed(
            guidance_volume,
            image_size=image_size,
            value_min=0.0,
            value_max=1.0,
        )
        mask_value = mask_collection.get(scale_name)
        mask_panel = None
        if mask_value is not None:
            mask_volume = _to_stage_scalar_volume(mask_value, sample_index)
            if mask_volume is not None:
                mask_panel = render_scalar_volume_bev_fixed(
                    mask_volume,
                    image_size=image_size,
                    value_min=0.0,
                    value_max=1.0,
                )
        if guidance_panel is None and mask_panel is None:
            continue
        if guidance_panel is None:
            guidance_panel = _render_placeholder_panel(image_size, 'Guidance',
                                                       'guidance unavailable')
        else:
            guidance_panel = _draw_panel_title(guidance_panel, 'Guidance')
        if mask_panel is None:
            mask_panel = _render_placeholder_panel(image_size, 'Top-K Mask',
                                                   'mask unavailable')
        else:
            mask_panel = _draw_panel_title(mask_panel, 'Top-K Mask')

        stage_gt_occ = _to_stage_occ_labels(
            true_occ_collection.get(scale_name), sample_index)
        if stage_gt_occ is None and scale_name == '1_1':
            stage_gt_occ = _to_numpy_occ(gt_occ, sample_index)
        if stage_gt_occ is None and sample_meta is not None:
            stage_gt_occ = load_stage_occ_gt_from_meta(sample_meta, scale_name)

        dynamic_gt_panel = None
        gt_occ_panel = None
        if stage_gt_occ is not None and empty_idx is not None and color_map is not None:
            dynamic_gt_occ = _dynamic_only_occ(stage_gt_occ,
                                               dynamic_class_indices,
                                               empty_idx)
            dynamic_gt_panel = render_occ_bev(
                dynamic_gt_occ, empty_idx, color_map, image_size=image_size)
            gt_occ_panel = render_occ_bev(
                stage_gt_occ, empty_idx, color_map, image_size=image_size)
        if dynamic_gt_panel is None:
            dynamic_gt_panel = _render_placeholder_panel(
                image_size, 'Dynamic GT', 'dynamic gt unavailable')
        else:
            dynamic_gt_panel = _draw_panel_title(dynamic_gt_panel,
                                                 'Dynamic GT')
        if gt_occ_panel is None:
            gt_occ_panel = _render_placeholder_panel(image_size, 'GT Occ',
                                                     'gt occ unavailable')
        else:
            gt_occ_panel = _draw_panel_title(gt_occ_panel, 'GT Occ')

        combined_panel = _compose_panel_row(
            [guidance_panel, mask_panel, dynamic_gt_panel, gt_occ_panel],
            title=_format_scale_label(scale_name))
        if combined_panel is not None:
            panels.append(combined_panel)
    if not panels:
        return None
    return _stack_section_images(panels)


def render_footprint_token_grid(feature_payload,
                                sample_index=0,
                                image_size=(320, 320)):
    """Render centre, local pool, and fixed-budget PTR token selections."""
    if not isinstance(feature_payload, dict):
        return None
    center_collection = dict(
        feature_payload.get('stage_ptr_footprint_center_masks') or {})
    pool_collection = dict(
        feature_payload.get('stage_ptr_footprint_pool_masks') or {})
    anchor_collection = dict(
        feature_payload.get('stage_ptr_footprint_anchor_masks') or {})
    final_collection = dict(
        feature_payload.get('stage_temporal_fusion_fg_masks') or {})

    panels = []
    for scale_name, center_value in _feature_collection_items(
            center_collection):
        center = _to_stage_scalar_volume(center_value, sample_index)
        pool = _to_stage_scalar_volume(
            pool_collection.get(scale_name), sample_index)
        final = _to_stage_scalar_volume(
            final_collection.get(scale_name), sample_index)
        anchor = _to_stage_scalar_volume(
            anchor_collection.get(scale_name), sample_index)
        if center is None or pool is None or final is None:
            continue
        masks = [
            np.asarray(value).max(axis=2) > 0
            for value in (center, pool, final)
        ]
        stage_panels = []
        for mask, title in zip(
                masks, ('DCA centres', 'Footprint pool', 'Final fixed-K')):
            image = np.where(mask[..., None], 255, 0).astype(np.uint8)
            image = np.repeat(image, 3, axis=2)
            image = cv2.resize(
                image, image_size, interpolation=cv2.INTER_NEAREST)
            stage_panels.append(_draw_panel_title(image, title))

        center_mask, _, final_mask = masks
        change = np.full((*center_mask.shape, 3), 235, dtype=np.uint8)
        retained = center_mask & final_mask
        removed = center_mask & ~final_mask
        added = final_mask & ~center_mask
        # BGR: blue=retained centre, red=removed, green=added footprint token.
        change[retained] = (220, 130, 35)
        change[removed] = (45, 45, 225)
        change[added] = (55, 195, 65)
        title = 'B=kept R=dropped G=added'
        if anchor is not None:
            anchor_mask = np.asarray(anchor).max(axis=2) > 0
            # Yellow marks conserved DCA discovery peaks without adding a
            # fifth panel or changing the established diagnostic layout.
            change[anchor_mask] = (0, 215, 255)
            title += ' Y=anchor'
        change = cv2.resize(
            change, image_size, interpolation=cv2.INTER_NEAREST)
        stage_panels.append(_draw_panel_title(change, title))
        panels.append(
            _compose_panel_row(
                stage_panels, title=_format_scale_label(scale_name)))
    if not panels:
        return None
    return _stack_section_images(panels)


def _to_transition_gate_volume(value, sample_index=0):
    """Return one [X, Y, Z, 3] Persist/Transport/Refresh gate tensor."""
    volume = _to_numpy(value)
    if volume is None:
        return None
    if volume.ndim == 5:
        if sample_index >= volume.shape[0]:
            return None
        volume = volume[sample_index]
    if volume.ndim != 4:
        return None
    if volume.shape[-1] == 3:
        return volume.astype(np.float32, copy=False)
    if volume.shape[0] == 3:
        # CanonicalPTRController stores volumes as [C,Z,Y,X].  Merely moving
        # C to the end yields [Z,Y,X,C], which silently rotates the BEV state
        # and makes its support incompatible with GT flow [X,Y,Z,2].  Restore
        # the public visualization convention explicitly.
        return volume.transpose(3, 2, 1, 0).astype(np.float32, copy=False)
    return None


def _resize_bev_mask(mask, target_shape):
    if mask is None or mask.ndim != 2:
        return None
    return cv2.resize(
        mask.astype(np.uint8), (int(target_shape[1]), int(target_shape[0])),
        interpolation=cv2.INTER_NEAREST).astype(bool)


def _render_transition_state_context(pred_occ, gt_occ, gt_flow,
                                     dynamic_class_indices, empty_idx,
                                     target_shape):
    """Show GT still/moving support and prediction-only dynamic ghosts."""
    if gt_occ is None or gt_occ.ndim != 3:
        return None
    gt_dynamic = np.isin(gt_occ, dynamic_class_indices).any(axis=2)
    still_dynamic = gt_dynamic.copy()
    moving_dynamic = np.zeros_like(gt_dynamic)
    if gt_flow is not None and gt_flow.ndim == 4 and gt_flow.shape[-1] >= 2:
        speed = np.linalg.norm(gt_flow[..., :2].astype(np.float32), axis=-1)
        moving_dynamic = (speed >= 2.0).any(axis=2) & gt_dynamic
        still_dynamic &= ~moving_dynamic
    still_dynamic = _resize_bev_mask(still_dynamic, target_shape)
    moving_dynamic = _resize_bev_mask(moving_dynamic, target_shape)
    context = np.full((*target_shape, 3), 245, dtype=np.uint8)
    # BGR: cyan=still dynamic, red=moving dynamic, magenta=dynamic ghost.
    context[still_dynamic] = (220, 190, 40)
    context[moving_dynamic] = (45, 45, 225)
    if pred_occ is not None and pred_occ.ndim == 3 and empty_idx is not None:
        pred_dynamic = np.isin(pred_occ, dynamic_class_indices).any(axis=2)
        ghosts = pred_dynamic & (gt_occ == int(empty_idx)).all(axis=2)
        ghosts = _resize_bev_mask(ghosts, target_shape)
        context[ghosts] = (185, 45, 185)
    return context


def render_transition_gate_grid(feature_payload,
                                pred_occ=None,
                                gt_occ=None,
                                gt_flow=None,
                                dynamic_class_indices=None,
                                empty_idx=None,
                                sample_index=0,
                                image_size=(260, 260)):
    """Render PTR's three operator weights and their motion/ghost context.

    Gate values exist only on selected temporal tokens; unselected pixels stay
    gray in the operator map. This keeps sparse token selection distinguishable
    from a confident Refresh decision.
    """
    if not isinstance(feature_payload, dict):
        return None
    gate_collection = dict(
        feature_payload.get('stage_transition_gate_maps') or {})
    dynamic_class_indices = tuple(dynamic_class_indices or ())
    panels = []
    gate_names = ('Persist', 'Transport', 'Refresh')
    gate_colors = np.asarray([(220, 130, 35), (35, 155, 235), (55, 195, 65)],
                             dtype=np.uint8)
    for scale_name, gate_value in _feature_collection_items(gate_collection):
        gate = _to_transition_gate_volume(gate_value, sample_index)
        if gate is None:
            continue
        gate_bev = np.clip(gate.max(axis=2), 0.0, 1.0)
        stage_panels = []
        for channel, gate_name in enumerate(gate_names):
            heatmap = cv2.applyColorMap(
                np.round(gate_bev[..., channel] * 255.0).astype(np.uint8),
                cv2.COLORMAP_VIRIDIS)
            heatmap = cv2.resize(
                heatmap, image_size, interpolation=cv2.INTER_NEAREST)
            stage_panels.append(_draw_panel_title(heatmap, gate_name))

        selected = gate_bev.sum(axis=-1) > 0
        operator = np.full((*gate_bev.shape[:2], 3), 180, dtype=np.uint8)
        operator[selected] = gate_colors[gate_bev.argmax(axis=-1)[selected]]
        operator = cv2.resize(
            operator, image_size, interpolation=cv2.INTER_NEAREST)
        stage_panels.append(
            _draw_panel_title(operator,
                              'Selected (blue=P, orange=T, green=R)'))

        state_context = _render_transition_state_context(
            pred_occ, gt_occ, gt_flow, dynamic_class_indices, empty_idx,
            gate_bev.shape[:2])
        if state_context is None:
            state_context = _render_placeholder_panel(image_size, 'GT State',
                                                      'GT state unavailable')
        else:
            state_context = cv2.resize(
                state_context, image_size, interpolation=cv2.INTER_NEAREST)
            state_context = _draw_panel_title(
                state_context, 'GT: cyan=still red=moving purple=ghost')
        stage_panels.append(state_context)
        panels.append(
            _compose_panel_row(
                stage_panels, title=_format_scale_label(scale_name)))
    if not panels:
        return None
    return _stack_section_images(panels)


def render_canonical_ptr_grid(feature_payload,
                              sample_index=0,
                              image_size=(220, 220)):
    """Render flow-conditioned P/T/R predictions at all supervised scales."""
    if not isinstance(feature_payload, dict):
        return None
    prediction_collection = dict(
        feature_payload.get('canonical_ptr_predicted_maps') or {})
    target_collection = dict(
        feature_payload.get('canonical_ptr_target_maps') or {})
    mask_collection = dict(
        feature_payload.get('canonical_ptr_supervision_masks') or {})
    flow_collection = dict(
        feature_payload.get('canonical_ptr_gt_flow_maps') or {})
    if not prediction_collection:
        return None

    # Do not present an all-grey PTR panel as a model diagnostic.  Training
    # batches legitimately contain samples with no valid dynamic voxel after
    # the current/history visibility check.  They have no native P/T/R label,
    # so a prediction panel would only visualise unconstrained background
    # priors while the raw flow tensor can still contain unrelated entries.
    # Returning ``None`` lets the caller omit this optional artifact for that
    # sample; the same-token manifest then only advertises genuine PTR audits.
    native_target = _to_transition_gate_volume(
        target_collection.get('1_1'), sample_index)
    if native_target is None or not np.any(native_target.sum(axis=-1) > 0):
        return None

    gate_names = ('Persist', 'Transport', 'Refresh')
    # BGR: blue=P, orange=T, green=R. This legend is shared by pred and GT.
    gate_colors = np.asarray([(220, 130, 35), (35, 155, 235),
                              (55, 195, 65)], dtype=np.uint8)

    def state_panel(gate, support, title):
        # Match the voxel-level argmax diagnostics used by the PTR objective.
        # A BEV column may contain several height bins at native scale, but
        # only a subset is supervised.  Summing probabilities over the full
        # column lets unconstrained height bins dominate the displayed state
        # (most visibly at 1/1, which has many height bins).  Mask in 3D first,
        # the categorical state per voxel, and only then collapse by vote.
        voxel_selected = gate.sum(axis=-1) > 0
        if support is not None:
            support = np.asarray(support, dtype=bool)
            if support.shape == voxel_selected.shape:
                voxel_selected &= support
        state_index = gate.argmax(axis=-1)
        state_votes = np.eye(
            len(gate_names), dtype=np.int16)[state_index]
        state_votes[~voxel_selected] = 0
        gate_bev = state_votes.sum(axis=2)
        selected = gate_bev.sum(axis=-1) > 0
        image = np.full((*gate_bev.shape[:2], 3), 180, dtype=np.uint8)
        if np.any(selected):
            image[selected] = gate_colors[gate_bev.argmax(axis=-1)[selected]]
        # PTR volumes use public [X,Y,Z,C] coordinates.  The flow renderer
        # maps that grid to image rows/columns via `_orient_xy_grid_to_image`;
        # applying the identical transform here is essential for a state
        # pixel and its GT-flow vector to refer to the same BEV location.
        image = _orient_xy_grid_to_image(image)
        image = cv2.resize(image, image_size, interpolation=cv2.INTER_NEAREST)
        return _draw_panel_title(image, title)

    def gt_flow_panel(gt_flow, title, ptr_support):
        """Render the scale-matched PTR flow with the reference renderer."""
        if gt_flow is None or ptr_support is None:
            return _render_placeholder_panel(image_size, title,
                                             'GT flow unavailable')
        ptr_support = np.asarray(ptr_support, dtype=bool)
        if ptr_support.shape != tuple(gt_flow.shape[:3]):
            return _render_placeholder_panel(image_size, title,
                                             'GT flow unavailable')
        ptr_semantics = np.zeros(ptr_support.shape, dtype=np.uint8)
        ptr_semantics[ptr_support] = 1
        panel = render_flow_bev(
            gt_flow,
            ptr_semantics,
            dynamic_class_indices=(1, ),
            image_size=image_size,
            max_speed=8.0)
        if panel is None:
            return _render_placeholder_panel(image_size, title,
                                             'GT flow unavailable')
        return _draw_panel_title(panel, title)

    panels = []
    for scale_name, prediction_value in _feature_collection_items(
            prediction_collection):
        prediction = _to_transition_gate_volume(prediction_value, sample_index)
        if prediction is None:
            continue
        target = _to_transition_gate_volume(
            target_collection.get(scale_name), sample_index)
        support = _to_stage_scalar_volume(
            mask_collection.get(scale_name), sample_index)
        gt_flow = _to_numpy_flow(flow_collection.get(scale_name),
                                 sample_index)
        # ``target`` is the most direct visualization mask: at coarser
        # execution scales average pooling can turn a binary supervision mask
        # into a broader footprint than the actual P/T/R target.  Prefer the
        # non-zero native target view whenever it is available, so predicted
        # heatmaps and GT state panels refer to exactly the same voxels.
        display_support = (target.sum(axis=-1) > 0
                           if target is not None else support)
        stage_panels = []
        masked_prediction = prediction
        if (display_support is not None
                and display_support.shape == prediction.shape[:3]):
            masked_prediction = np.where(
                display_support[..., None], prediction, 0.0)
        prediction_bev = np.clip(masked_prediction.max(axis=2), 0.0, 1.0)
        support_bev = (None if display_support is None else
                       display_support.max(axis=2) > 0)
        for channel, name in enumerate(gate_names):
            channel_value = prediction_bev[..., channel]
            # The 1/1 PTR objective intentionally has no label outside
            # valid dynamic voxels.  Rendering that unconstrained background
            # as a probability heatmap is misleading (the identity Persist
            # prior is expected there), so display it as neutral grey.
            if support_bev is not None:
                channel_value = channel_value.copy()
                channel_value[~support_bev] = 0.0
            heatmap = cv2.applyColorMap(
                np.round(channel_value * 255.0).astype(
                    np.uint8), cv2.COLORMAP_VIRIDIS)
            if support_bev is not None:
                heatmap[~support_bev] = 180
            # Keep every P/T/R panel in the same BEV image convention as the
            # state and flow panels below.
            heatmap = _orient_xy_grid_to_image(heatmap)
            heatmap = cv2.resize(
                heatmap, image_size, interpolation=cv2.INTER_NEAREST)
            stage_panels.append(_draw_panel_title(heatmap, 'Pred ' + name))
        stage_panels.append(
            state_panel(prediction, display_support, 'Pred state (P/T/R)'))
        if target is None:
            stage_panels.append(
                _render_placeholder_panel(image_size, 'GT state',
                                          'native target unavailable'))
        else:
            stage_panels.append(
                state_panel(target, display_support, 'GT state (P/T/R)'))
        stage_panels.append(
            gt_flow_panel(gt_flow, 'GT flow (P/T/R support)',
                          display_support))
        panels.append(
            _compose_panel_row(stage_panels, title=_format_scale_label(scale_name)))
    return _stack_section_images(panels) if panels else None


def render_flow_prior_panel(feature_payload,
                            pred_occ,
                            dynamic_class_indices,
                            sample_index=0,
                            image_size=(320, 320),
                            max_speed=8.0):
    if not isinstance(feature_payload, dict):
        return None
    prior_flow = _to_numpy_flow(
        feature_payload.get('flow_prior'), sample_index)
    pred_occ_labels = _to_numpy_occ(pred_occ, sample_index)
    if prior_flow is None or pred_occ_labels is None:
        return None
    panel = render_flow_bev(
        prior_flow,
        pred_occ_labels,
        dynamic_class_indices=dynamic_class_indices,
        image_size=image_size,
        max_speed=max_speed,
    )
    if panel is None:
        return None
    return _draw_panel_title(panel, 'Prior Flow')


def render_dense_flow_panel(flow,
                            sample_index=0,
                            image_size=(320, 320),
                            max_speed=8.0,
                            title='Dense Flow'):
    dense_flow = _to_numpy_flow(flow, sample_index)
    if dense_flow is None:
        return None
    panel = render_dense_flow_bev(
        dense_flow,
        image_size=image_size,
        max_speed=max_speed,
    )
    if panel is None:
        return None
    return _draw_panel_title(panel, title)


def render_dynamic_occ_grid(feature_payload,
                            gt_occ,
                            empty_idx,
                            color_map,
                            dynamic_class_indices=None,
                            sample_index=0,
                            image_size=(320, 320)):
    if not isinstance(feature_payload, dict):
        feature_payload = {}
    occ_collection = dict(feature_payload.get('stage_occ_predictions') or {})
    dynamic_class_indices = tuple(dynamic_class_indices or ())

    panels = []
    for scale_name, occ_value in _feature_collection_items(occ_collection):
        occ_labels = _to_stage_occ_labels(occ_value, sample_index)
        if occ_labels is None:
            continue
        valid_mask = np.isin(occ_labels, dynamic_class_indices)
        panel = render_masked_occ_bev(
            occ_labels,
            valid_mask,
            empty_idx,
            color_map,
            image_size=image_size)
        if panel is None:
            continue
        panels.append(
            _draw_panel_title(panel, _format_scale_label(scale_name)))

    gt_occ_labels = _to_numpy_occ(gt_occ, sample_index)
    if gt_occ_labels is not None and gt_occ_labels.ndim == 3:
        gt_dynamic_mask = np.isin(gt_occ_labels, dynamic_class_indices)
        gt_panel = render_masked_occ_bev(
            gt_occ_labels,
            gt_dynamic_mask,
            empty_idx,
            color_map,
            image_size=image_size)
        if gt_panel is not None:
            panels.append(_draw_panel_title(gt_panel, 'GT Dynamic'))

    if not panels:
        return None
    return _compose_panel_grid(panels, title=None)


def render_modality_feature_grid(feature_payload,
                                 sample_index=0,
                                 final_bev_feature=None):
    if not isinstance(feature_payload, dict):
        feature_payload = {}
    panels = []
    for key, title in (
        ('camera', 'Camera'),
        ('points', 'Points'),
        ('fused', 'Fused'),
    ):
        panel = render_feature_collection_summary(
            feature_payload.get(key), sample_index=sample_index)
        if panel is not None:
            panels.append(_draw_panel_title(panel, title))
    final_bev_panel = render_bev_feature(
        _select_sample_feature(final_bev_feature, sample_index),
        image_size=(320, 320))
    if final_bev_panel is not None:
        panels.append(_draw_panel_title(final_bev_panel, 'Final BEV'))
    if not panels:
        return None
    return _compose_panel_grid(panels, title=None)


def render_recursive_occ_grid(feature_payload,
                              pred_occ,
                              empty_idx,
                              color_map,
                              sample_index=0):
    if not isinstance(feature_payload, dict):
        feature_payload = {}
    occ_collection = dict(feature_payload.get('stage_occ_predictions') or {})
    final_occ = _to_numpy_occ(pred_occ, sample_index)
    if final_occ is not None and final_occ.ndim == 3 and '1_1' not in occ_collection:
        occ_collection['1_1'] = final_occ
    recursive_occ_grid = render_occ_collection_grid(
        occ_collection,
        empty_idx=empty_idx,
        color_map=color_map,
        sample_index=sample_index)
    return recursive_occ_grid


def _select_sample_matrix(value, sample_index):
    value = _to_numpy(value)
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3:
        if sample_index >= value.shape[0]:
            return None
        value = value[sample_index]
    if value.shape == (3, 3):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = value
        return matrix
    if value.shape == (4, 4):
        return value.astype(np.float32)
    return None


def _occ_numpy_to_tensor(semantics):
    semantics = np.asarray(semantics, dtype=np.float32)
    if semantics.ndim != 3:
        return None
    return torch.from_numpy(semantics).permute(2, 1,
                                               0).unsqueeze(0).unsqueeze(0)


def _flow_numpy_to_tensor(flow):
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 4 or flow.shape[-1] != 2:
        return None
    return torch.from_numpy(flow).permute(3, 2, 1, 0).unsqueeze(0)


def _tensor_to_occ_numpy(tensor):
    if tensor is None or tensor.ndim != 5:
        return None
    return tensor[0, 0].detach().cpu().numpy().transpose(
        2, 1, 0).round().astype(np.uint8)


def _infer_occ_grid_meta(semantics, point_cloud_range):
    semantics = np.asarray(semantics)
    if semantics.ndim != 3 or point_cloud_range is None:
        return None
    w, h, z = semantics.shape
    x_min, y_min, z_min, x_max, y_max, z_max = [
        float(v) for v in point_cloud_range
    ]
    dx = torch.tensor([(x_max - x_min) / max(w, 1),
                       (y_max - y_min) / max(h, 1),
                       (z_max - z_min) / max(z, 1)],
                      dtype=torch.float32)
    bx = torch.tensor(
        [x_min + dx[0] / 2.0, y_min + dx[1] / 2.0, z_min + dx[2] / 2.0],
        dtype=torch.float32)
    return w, h, z, dx, bx


def _build_occ_sampling_grid(w, h, z, device, dtype):
    xs = torch.linspace(
        0, w - 1, w, dtype=dtype, device=device).view(1, w, 1).expand(h, w, z)
    ys = torch.linspace(
        0, h - 1, h, dtype=dtype, device=device).view(h, 1, 1).expand(h, w, z)
    zs = torch.linspace(
        0, z - 1, z, dtype=dtype, device=device).view(1, 1, z).expand(h, w, z)
    grid = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1)
    return grid.view(1, h, w, z, 4, 1)


def _build_feat2occ(dx, bx, device, dtype):
    feat2occ = torch.zeros((4, 4), dtype=dtype, device=device)
    feat2occ[0, 0] = dx[0]
    feat2occ[1, 1] = dx[1]
    feat2occ[2, 2] = dx[2]
    feat2occ[0, 3] = bx[0] - dx[0] / 2.0
    feat2occ[1, 3] = bx[1] - dx[1] / 2.0
    feat2occ[2, 3] = bx[2] - dx[2] / 2.0
    feat2occ[3, 3] = 1.0
    return feat2occ.view(1, 4, 4)


def _load_neighbor_occ_from_meta(sample_meta, neighbor_key='prev'):
    if not isinstance(sample_meta, dict):
        return None
    current_occ_path = sample_meta.get('occ_gt_path') or sample_meta.get(
        'occ_path')
    neighbor_token = sample_meta.get(f'{neighbor_key}_sample_idx',
                                     '') or sample_meta.get(neighbor_key, '')
    if not current_occ_path or not neighbor_token:
        return None
    current_occ_path = os.path.abspath(current_occ_path)
    if current_occ_path.endswith('.npz'):
        scene_dir = os.path.dirname(os.path.dirname(current_occ_path))
    elif os.path.isdir(current_occ_path) and os.path.exists(
            os.path.join(current_occ_path, 'labels.npz')):
        scene_dir = os.path.dirname(current_occ_path)
    else:
        scene_dir = os.path.dirname(current_occ_path)
    label_path = os.path.join(scene_dir, neighbor_token, 'labels.npz')
    if not os.path.exists(label_path):
        return None
    occ_label = np.load(label_path, allow_pickle=True)
    if 'semantics' not in occ_label:
        return None
    return occ_label['semantics'].astype(np.uint8)


def _load_neighbor_flow_from_meta(sample_meta,
                                  neighbor_key='prev',
                                  flow_gt_root=None):
    if not isinstance(sample_meta, dict):
        return None
    current_occ_path = sample_meta.get('occ_gt_path') or sample_meta.get(
        'occ_path')
    neighbor_token = sample_meta.get(f'{neighbor_key}_sample_idx',
                                     '') or sample_meta.get(neighbor_key, '')
    if not current_occ_path or not neighbor_token:
        return None
    current_occ_path = os.path.abspath(current_occ_path)
    if current_occ_path.endswith('.npz'):
        scene_dir = os.path.dirname(os.path.dirname(current_occ_path))
    elif os.path.isdir(current_occ_path) and os.path.exists(
            os.path.join(current_occ_path, 'labels.npz')):
        scene_dir = os.path.dirname(current_occ_path)
    else:
        scene_dir = os.path.dirname(current_occ_path)
    scene_name = os.path.basename(scene_dir)
    if flow_gt_root is not None:
        scene_dir = os.path.join(os.path.abspath(flow_gt_root), scene_name)
    label_path = os.path.join(scene_dir, neighbor_token, 'labels.npz')
    if not os.path.exists(label_path):
        return None
    occ_label = np.load(label_path, allow_pickle=True)
    if 'flow' not in occ_label:
        return None
    return occ_label['flow'].astype(np.float32)


def _ego_to_global_rt(sample_meta):
    rt = np.eye(4, dtype=np.float32)
    rt[:3, :3] = Quaternion(
        sample_meta['ego2global_rotation']).rotation_matrix.astype(np.float32)
    rt[:3, 3] = np.asarray(
        sample_meta['ego2global_translation'], dtype=np.float32)
    return rt


def _infer_curr_to_prev_ego_rt(sample_meta):
    if not isinstance(sample_meta, dict):
        return None
    curr_to_prev = sample_meta.get('curr_to_prev_ego_rt')
    if curr_to_prev is not None:
        return curr_to_prev
    current_token = _sample_meta_cache_key(sample_meta)
    prev_token = sample_meta.get('prev_sample_idx') or sample_meta.get('prev')
    if current_token:
        _SPARSE_WARP_META_CACHE[str(current_token)] = sample_meta
    if not prev_token:
        return None
    prev_meta = _SPARSE_WARP_META_CACHE.get(str(prev_token))
    if prev_meta is None:
        return None
    try:
        curr_ego_to_global = _ego_to_global_rt(sample_meta)
        prev_ego_to_global = _ego_to_global_rt(prev_meta)
        return (np.linalg.inv(prev_ego_to_global) @ curr_ego_to_global).astype(
            np.float32)
    except Exception:
        return None


def _rigid_align_neighbor_occ(neighbor_occ,
                              current_occ,
                              curr_to_prev_ego_rt,
                              point_cloud_range,
                              bda_mat=None):
    if neighbor_occ is None or current_occ is None or curr_to_prev_ego_rt is None:
        return None
    grid_meta = _infer_occ_grid_meta(current_occ, point_cloud_range)
    if grid_meta is None:
        return None
    w, h, z, dx, bx = grid_meta
    history_tensor = _occ_numpy_to_tensor(neighbor_occ)
    if history_tensor is None:
        return None
    device = history_tensor.device
    dtype = history_tensor.dtype
    dx = dx.to(device=device, dtype=dtype)
    bx = bx.to(device=device, dtype=dtype)
    feat2occ = _build_feat2occ(dx, bx, device=device, dtype=dtype)
    curr_to_prev = torch.as_tensor(
        curr_to_prev_ego_rt, dtype=dtype, device=device).view(1, 4, 4)
    current_bda = _select_sample_matrix(bda_mat, 0)
    if current_bda is None:
        current_bda = np.eye(4, dtype=np.float32)
    current_bda = torch.from_numpy(current_bda).to(
        device=device, dtype=dtype).view(1, 4, 4)
    rt_flow = torch.inverse(feat2occ) @ curr_to_prev @ torch.inverse(
        current_bda) @ feat2occ
    grid = _build_occ_sampling_grid(w, h, z, device=device, dtype=dtype)
    grid = rt_flow.view(1, 1, 1, 1, 4, 4) @ grid
    normalize_factor = torch.tensor(
        [max(w - 1.0, 1.0),
         max(h - 1.0, 1.0),
         max(z - 1.0, 1.0)],
        dtype=dtype,
        device=device,
    )
    grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1,
                                                           3) * 2.0 - 1.0
    aligned = F.grid_sample(
        history_tensor,
        grid.permute(0, 3, 1, 2, 4),
        align_corners=True,
        mode='nearest')
    return _tensor_to_occ_numpy(aligned)


def _rigid_align_neighbor_flow(neighbor_flow,
                               current_occ,
                               curr_to_prev_ego_rt,
                               point_cloud_range,
                               bda_mat=None):
    if neighbor_flow is None or current_occ is None or curr_to_prev_ego_rt is None:
        return None
    grid_meta = _infer_occ_grid_meta(current_occ, point_cloud_range)
    if grid_meta is None:
        return None
    w, h, z, dx, bx = grid_meta
    flow_tensor = _flow_numpy_to_tensor(neighbor_flow)
    if flow_tensor is None:
        return None
    device = flow_tensor.device
    dtype = flow_tensor.dtype
    dx = dx.to(device=device, dtype=dtype)
    bx = bx.to(device=device, dtype=dtype)
    feat2occ = _build_feat2occ(dx, bx, device=device, dtype=dtype)
    curr_to_prev = torch.as_tensor(
        curr_to_prev_ego_rt, dtype=dtype, device=device).view(1, 4, 4)
    current_bda = _select_sample_matrix(bda_mat, 0)
    if current_bda is None:
        current_bda = np.eye(4, dtype=np.float32)
    current_bda = torch.from_numpy(current_bda).to(
        device=device, dtype=dtype).view(1, 4, 4)
    rt_flow = torch.inverse(feat2occ) @ curr_to_prev @ torch.inverse(
        current_bda) @ feat2occ
    grid = _build_occ_sampling_grid(w, h, z, device=device, dtype=dtype)
    grid = rt_flow.view(1, 1, 1, 1, 4, 4) @ grid
    normalize_factor = torch.tensor(
        [max(w - 1.0, 1.0),
         max(h - 1.0, 1.0),
         max(z - 1.0, 1.0)],
        dtype=dtype,
        device=device,
    )
    sample_grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(
        1, 1, 1, 1, 3) * 2.0 - 1.0
    aligned = F.grid_sample(
        flow_tensor,
        sample_grid.permute(0, 3, 1, 2, 4),
        align_corners=True,
        mode='bilinear')
    rotation_prev_to_curr = torch.inverse(rt_flow)[:, :2, :2].type_as(aligned)
    flow_flat = aligned.permute(0, 2, 3, 4, 1).reshape(1, -1, 2)
    flow_rot = torch.einsum('bij,bnj->bni', rotation_prev_to_curr, flow_flat)
    flow_rot = flow_rot.reshape(1, z, h, w, 2).permute(0, 4, 1, 2,
                                                       3).contiguous()
    return flow_rot[0].detach().cpu().numpy().transpose(3, 2, 1,
                                                        0).astype(np.float32)


def _flow_warp_occ_history(rigid_history,
                           pred_flow,
                           point_cloud_range,
                           history_dt=0.5):
    if rigid_history is None or pred_flow is None:
        return None
    grid_meta = _infer_occ_grid_meta(rigid_history, point_cloud_range)
    if grid_meta is None:
        return None
    w, h, z, dx, _ = grid_meta
    history_tensor = _occ_numpy_to_tensor(rigid_history)
    flow_tensor = _flow_numpy_to_tensor(pred_flow)
    if history_tensor is None or flow_tensor is None:
        return None
    if tuple(flow_tensor.shape[-3:]) != (z, h, w):
        flow_tensor = F.interpolate(
            flow_tensor, size=(z, h, w), mode='trilinear', align_corners=False)
    device = history_tensor.device
    dtype = history_tensor.dtype
    flow_tensor = flow_tensor.to(device=device, dtype=dtype)
    dx = dx.to(device=device, dtype=dtype)
    x_coords = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, w)
    y_coords = torch.arange(h, device=device, dtype=dtype).view(1, 1, h, 1)
    z_coords = torch.arange(z, device=device, dtype=dtype).view(1, z, 1, 1)
    normalize_factor = torch.tensor(
        [max(w - 1.0, 1.0),
         max(h - 1.0, 1.0),
         max(z - 1.0, 1.0)],
        dtype=dtype,
        device=device,
    )
    shift_x = flow_tensor[:, 0] * (float(history_dt) / dx[0])
    shift_y = flow_tensor[:, 1] * (float(history_dt) / dx[1])
    warp_grid = torch.stack([
        x_coords.expand(1, z, h, w) - shift_x,
        y_coords.expand(1, z, h, w) - shift_y,
        z_coords.expand(1, z, h, w)
    ],
                            dim=-1)
    warp_grid = warp_grid / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
    warped = F.grid_sample(
        history_tensor, warp_grid, align_corners=True, mode='nearest')
    return _tensor_to_occ_numpy(warped)


def _dynamic_only_occ(semantics, dynamic_class_indices, empty_idx):
    if semantics is None:
        return None
    semantics = np.asarray(semantics).copy()
    if semantics.ndim != 3:
        return None
    if not dynamic_class_indices:
        return semantics
    dynamic_mask = np.isin(semantics,
                           np.asarray(dynamic_class_indices, dtype=np.int32))
    semantics[~dynamic_mask] = empty_idx
    return semantics


def _render_occ_diff_panel(current_occ,
                           history_occ,
                           dynamic_class_indices,
                           empty_idx,
                           image_size=DEFAULT_IMAGE_SIZE):
    current_dyn = _dynamic_only_occ(current_occ, dynamic_class_indices,
                                    empty_idx)
    history_dyn = _dynamic_only_occ(history_occ, dynamic_class_indices,
                                    empty_idx)
    current_bev = project_occ_to_bev_labels(current_dyn, empty_idx)
    history_bev = project_occ_to_bev_labels(history_dyn, empty_idx)
    if current_bev is None or history_bev is None:
        return None, None

    cur_occ = current_bev != empty_idx
    hist_occ = history_bev != empty_idx
    same_cls = np.logical_and(cur_occ, history_bev == current_bev)
    both_occ = np.logical_and(cur_occ, hist_occ)
    cls_mismatch = np.logical_and(both_occ, np.logical_not(same_cls))
    cur_only = np.logical_and(cur_occ, np.logical_not(hist_occ))
    hist_only = np.logical_and(hist_occ, np.logical_not(cur_occ))

    canvas = np.full(current_bev.shape + (3, ), 255, dtype=np.uint8)
    canvas[same_cls] = np.array([80, 200, 120], dtype=np.uint8)
    canvas[cls_mismatch] = np.array([0, 215, 255], dtype=np.uint8)
    canvas[cur_only] = np.array([80, 80, 255], dtype=np.uint8)
    canvas[hist_only] = np.array([255, 170, 0], dtype=np.uint8)
    canvas = _orient_xy_grid_to_image(canvas)
    canvas = cv2.resize(canvas, image_size, interpolation=cv2.INTER_NEAREST)

    union = np.logical_or(cur_occ, hist_occ).sum()
    score = float(same_cls.sum()) / float(max(union, 1))
    return canvas, score


def _lighten_occ_panel(image_rgb, white_mix=0.45):
    image_rgb = np.asarray(image_rgb, dtype=np.float32)
    return np.clip(
        image_rgb * (1.0 - float(white_mix)) + 255.0 * float(white_mix),
        0.0,
        255.0,
    ).astype(np.uint8)


def _build_occ_grid_meta_np(semantics, point_cloud_range):
    semantics = np.asarray(semantics)
    if semantics.ndim != 3 or point_cloud_range is None:
        return None
    w, h, z = semantics.shape
    x_min, y_min, z_min, x_max, y_max, z_max = [
        float(v) for v in point_cloud_range
    ]
    voxel_size = np.array(
        [(x_max - x_min) / max(w, 1), (y_max - y_min) / max(h, 1),
         (z_max - z_min) / max(z, 1)],
        dtype=np.float32,
    )
    return dict(
        point_cloud_range=np.array([x_min, y_min, z_min, x_max, y_max, z_max],
                                   dtype=np.float32),
        voxel_size=voxel_size,
        x_centers=(x_min +
                   (np.arange(w, dtype=np.float32) + 0.5) * voxel_size[0]),
        y_centers=(y_min +
                   (np.arange(h, dtype=np.float32) + 0.5) * voxel_size[1]),
        z_centers=(z_min +
                   (np.arange(z, dtype=np.float32) + 0.5) * voxel_size[2]),
    )


def _build_dynamic_history_points(source_semantics, flow, grid_meta,
                                  dynamic_class_indices, empty_idx):
    source_semantics = np.asarray(source_semantics)
    dynamic_mask = np.isin(source_semantics,
                           np.asarray(dynamic_class_indices, dtype=np.int32))
    if flow is not None:
        flow = np.asarray(flow, dtype=np.float32)
        dynamic_mask &= np.isfinite(flow).all(axis=-1)
    coords = np.argwhere(dynamic_mask)
    if coords.size == 0:
        return None
    points_ego = np.stack([
        grid_meta['x_centers'][coords[:, 0]],
        grid_meta['y_centers'][coords[:, 1]],
        grid_meta['z_centers'][coords[:, 2]],
    ],
                          axis=1).astype(np.float32)
    current_classes = source_semantics[coords[:, 0], coords[:, 1],
                                       coords[:, 2]].astype(np.int32)
    flow_xy = None
    if flow is not None:
        flow_xy = flow[coords[:, 0], coords[:, 1],
                       coords[:, 2]].astype(np.float32)
    return coords, points_ego, flow_xy, current_classes


def _sample_occ_at_points(points_ego, semantics, grid_meta):
    points_ego = np.asarray(points_ego, dtype=np.float32).reshape(-1, 3)
    semantics = np.asarray(semantics)
    voxel_indices = np.floor(
        (points_ego - grid_meta['point_cloud_range'][:3]) /
        grid_meta['voxel_size']).astype(np.int32)
    valid_mask = ((voxel_indices[:, 0] >= 0) &
                  (voxel_indices[:, 0] < semantics.shape[0]) &
                  (voxel_indices[:, 1] >= 0) &
                  (voxel_indices[:, 1] < semantics.shape[1]) &
                  (voxel_indices[:, 2] >= 0) &
                  (voxel_indices[:, 2] < semantics.shape[2]))
    return voxel_indices, valid_mask


def _classify_warp_hits(semantics_volume,
                        voxel_indices,
                        current_classes,
                        valid_mask,
                        dynamic_class_indices,
                        empty_idx,
                        neighborhood_radius=1):
    semantics_volume = np.asarray(semantics_volume, dtype=np.int32)
    voxel_indices = np.asarray(voxel_indices, dtype=np.int32).reshape(-1, 3)
    current_classes = np.asarray(current_classes, dtype=np.int32).reshape(-1)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    statuses = np.full((voxel_indices.shape[0], ),
                       _WARP_STATUS_OUTSIDE,
                       dtype=np.uint8)
    radius = max(int(neighborhood_radius), 0)
    x_limit, y_limit, z_limit = semantics_volume.shape
    dynamic_class_indices = tuple(
        int(class_idx) for class_idx in dynamic_class_indices)
    for index in np.flatnonzero(valid_mask):
        x_idx, y_idx, z_idx = voxel_indices[index]
        x0 = max(int(x_idx) - radius, 0)
        x1 = min(int(x_idx) + radius + 1, x_limit)
        y0 = max(int(y_idx) - radius, 0)
        y1 = min(int(y_idx) + radius + 1, y_limit)
        z0 = max(int(z_idx) - radius, 0)
        z1 = min(int(z_idx) + radius + 1, z_limit)
        local_semantics = semantics_volume[x0:x1, y0:y1, z0:z1]
        if local_semantics.size <= 0:
            statuses[index] = _WARP_STATUS_EMPTY
            continue
        if np.any(local_semantics == current_classes[index]):
            statuses[index] = _WARP_STATUS_SAME_CLASS
            continue
        occupied_mask = local_semantics != int(empty_idx)
        if not np.any(occupied_mask):
            statuses[index] = _WARP_STATUS_EMPTY
            continue
        if np.any(np.isin(local_semantics, dynamic_class_indices)):
            statuses[index] = _WARP_STATUS_DYNAMIC_OTHER
        else:
            statuses[index] = _WARP_STATUS_OCCUPIED_OTHER
    return statuses


def _draw_occ_points(image_rgb, voxel_indices, grid_shape_xy, colors,
                     point_size):
    image_rgb = np.asarray(image_rgb)
    voxel_indices = np.asarray(voxel_indices, dtype=np.int32)
    if voxel_indices.size == 0:
        return image_rgb
    canvas = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(canvas)
    cells_x, cells_y = int(grid_shape_xy[0]), int(grid_shape_xy[1])
    scale_x = canvas.width / float(cells_x)
    scale_y = canvas.height / float(cells_y)
    # Allow point_size=0 to render a single-pixel marker. This is useful for
    # dense hit-map diagnostics where 320x320 panels with 5x5 blocks look too coarse.
    half_size = max(int(point_size), 0)
    colors = np.asarray(colors, dtype=np.uint8)
    if colors.ndim == 1:
        colors = np.repeat(colors[None, :], voxel_indices.shape[0], axis=0)
    for index, (x_idx, y_idx) in enumerate(voxel_indices[:, :2]):
        center_col = (cells_x - 1 - int(x_idx) + 0.5) * scale_x
        center_row = (cells_y - 1 - int(y_idx) + 0.5) * scale_y
        draw.rectangle(
            [
                (center_col - half_size, center_row - half_size),
                (center_col + half_size, center_row + half_size),
            ],
            fill=tuple(int(value) for value in colors[index]),
        )
    return np.asarray(canvas)


def _build_warp_point_result(source_semantics, flow, grid_meta,
                             dynamic_class_indices, empty_idx, delta_t_sec,
                             use_flow):
    point_pack = _build_dynamic_history_points(
        source_semantics,
        flow if use_flow else None,
        grid_meta=grid_meta,
        dynamic_class_indices=dynamic_class_indices,
        empty_idx=empty_idx,
    )
    if point_pack is None:
        return None
    _, points_ego, flow_xy, current_classes = point_pack
    warped_points = points_ego.copy()
    if use_flow and flow_xy is not None and warped_points.shape[0] > 0:
        warped_points[:, :2] += flow_xy * float(delta_t_sec)
    voxel_indices, valid_mask = _sample_occ_at_points(warped_points,
                                                      source_semantics,
                                                      grid_meta)
    return dict(
        voxel_indices=voxel_indices,
        valid_mask=valid_mask,
        current_classes=current_classes,
    )


def _point_warp_occ_volume(source_occ,
                           flow,
                           point_cloud_range,
                           dynamic_class_indices,
                           empty_idx,
                           delta_t_sec=0.5):
    if source_occ is None or flow is None:
        return None
    grid_meta = _build_occ_grid_meta_np(source_occ, point_cloud_range)
    if grid_meta is None:
        return None
    point_pack = _build_dynamic_history_points(
        source_occ,
        flow,
        grid_meta=grid_meta,
        dynamic_class_indices=dynamic_class_indices,
        empty_idx=empty_idx,
    )
    if point_pack is None:
        return source_occ
    warped = np.asarray(source_occ).copy()
    dynamic_mask = np.isin(warped,
                           np.asarray(dynamic_class_indices, dtype=np.int32))
    warped[dynamic_mask] = int(empty_idx)
    _, points_ego, flow_xy, current_classes = point_pack
    warped_points = points_ego.copy()
    warped_points[:, :2] += flow_xy * float(delta_t_sec)
    voxel_indices, valid_mask = _sample_occ_at_points(warped_points, warped,
                                                      grid_meta)
    valid_xyz = voxel_indices[valid_mask]
    valid_classes = current_classes[valid_mask]
    if valid_xyz.shape[0] > 0:
        warped[
            valid_xyz[:, 0],
            valid_xyz[:, 1],
            valid_xyz[:, 2],
        ] = valid_classes.astype(
            warped.dtype, copy=False)
    return warped.astype(np.uint8, copy=False)


def _render_warp_hit_panel(current_occ,
                           warp_result,
                           zero_result,
                           dynamic_class_indices,
                           empty_idx,
                           color_map,
                           image_size,
                           title,
                           neighborhood_radius=1,
                           point_size=2):
    if warp_result is None or zero_result is None or current_occ is None:
        return None, None
    flow_status = _classify_warp_hits(
        current_occ,
        warp_result['voxel_indices'],
        current_classes=warp_result['current_classes'],
        valid_mask=warp_result['valid_mask'],
        dynamic_class_indices=dynamic_class_indices,
        empty_idx=empty_idx,
        neighborhood_radius=neighborhood_radius,
    )
    zero_status = _classify_warp_hits(
        current_occ,
        zero_result['voxel_indices'],
        current_classes=zero_result['current_classes'],
        valid_mask=zero_result['valid_mask'],
        dynamic_class_indices=dynamic_class_indices,
        empty_idx=empty_idx,
        neighborhood_radius=neighborhood_radius,
    )
    base = render_occ_bev_rgb(
        _dynamic_only_occ(current_occ, dynamic_class_indices, empty_idx),
        empty_idx=empty_idx,
        color_map=color_map,
        image_size_hw=image_size,
    )
    base = _lighten_occ_panel(base, white_mix=0.55)
    valid_mask = np.asarray(warp_result['valid_mask'], dtype=bool)
    voxel_indices = warp_result['voxel_indices'][valid_mask]
    flow_colors = np.array([
        _WARP_HIT_COLORS.get(
            int(status), _WARP_HIT_COLORS[_WARP_STATUS_EMPTY])
        for status in flow_status[valid_mask]
    ],
                           dtype=np.uint8)
    panel = _draw_occ_points(
        image_rgb=base,
        voxel_indices=voxel_indices,
        grid_shape_xy=current_occ.shape[:2],
        colors=flow_colors,
        point_size=point_size,
    )
    zero_better_mask = np.asarray(
        zero_result['valid_mask'], dtype=bool) & (
            zero_status > flow_status)
    panel = _draw_occ_points(
        image_rgb=panel,
        voxel_indices=zero_result['voxel_indices'][zero_better_mask],
        grid_shape_xy=current_occ.shape[:2],
        colors=_WARP_ZERO_BETTER_COLOR,
        point_size=point_size,
    )
    same_class = int((flow_status == _WARP_STATUS_SAME_CLASS).sum())
    valid_count = int(np.asarray(warp_result['valid_mask'], dtype=bool).sum())
    score = float(same_class) / float(max(valid_count, 1))
    return _draw_panel_title(
        panel, f'{title} | score={score:.2f} gray=zero-better'), score


def render_sparse_warp_grid(sample_meta,
                            gt_occ,
                            gt_flow,
                            pred_flow,
                            dynamic_class_indices,
                            point_cloud_range,
                            empty_idx,
                            color_map,
                            sample_index=0,
                            image_size=(480, 480),
                            bda_mat=None,
                            max_speed=8.0,
                            flow_gt_root=None):
    current_occ = _to_numpy_occ(gt_occ, sample_index)
    if current_occ is None:
        current_occ = load_occ_gt_from_meta(sample_meta)
    current_token = _sample_meta_cache_key(sample_meta) if isinstance(
        sample_meta, dict) else None
    prev_token = None
    if isinstance(sample_meta, dict):
        prev_token = sample_meta.get('prev_sample_idx') or sample_meta.get(
            'prev')
        if current_token:
            _SPARSE_WARP_META_CACHE[current_token] = sample_meta
    current_panel = render_occ_bev(
        current_occ,
        empty_idx=empty_idx,
        color_map=color_map,
        image_size=image_size,
    )
    if current_panel is None:
        current_panel = _render_placeholder_panel(
            image_size,
            'Current GT | 1_1',
            'feature unavailable',
        )
    else:
        current_panel = _draw_panel_title(current_panel, 'Current GT | 1_1')

    sample_bda = _select_sample_matrix(bda_mat, sample_index)
    rigid_history = None
    pred_backwarped_current = None
    gt_backwarped_current = None
    if (isinstance(sample_meta, dict)
            and not bool(sample_meta.get('start_of_sequence', False))):
        prev_occ = _load_neighbor_occ_from_meta(
            sample_meta, neighbor_key='prev')
        curr_to_prev_ego_rt = _infer_curr_to_prev_ego_rt(sample_meta)
        if prev_occ is not None and curr_to_prev_ego_rt is not None and current_occ is not None:
            rigid_history = _rigid_align_neighbor_occ(
                prev_occ,
                current_occ,
                curr_to_prev_ego_rt=curr_to_prev_ego_rt,
                point_cloud_range=point_cloud_range,
                bda_mat=sample_bda)
            pred_backwarped_current = _point_warp_occ_volume(
                current_occ,
                pred_flow,
                point_cloud_range=point_cloud_range,
                dynamic_class_indices=dynamic_class_indices,
                empty_idx=empty_idx,
                delta_t_sec=-0.5)
            gt_backwarped_current = _point_warp_occ_volume(
                current_occ,
                gt_flow,
                point_cloud_range=point_cloud_range,
                dynamic_class_indices=dynamic_class_indices,
                empty_idx=empty_idx,
                delta_t_sec=-0.5)

    aligned_panel = render_occ_bev(
        rigid_history,
        empty_idx=empty_idx,
        color_map=color_map,
        image_size=image_size,
    )
    if aligned_panel is None:
        aligned_panel = _render_placeholder_panel(
            image_size,
            'Rigid History | 1_1',
            'history unavailable',
        )
    else:
        aligned_panel = _draw_panel_title(aligned_panel, 'Rigid History | 1_1')

    pred_warped_panel = render_occ_bev(
        pred_backwarped_current,
        empty_idx=empty_idx,
        color_map=color_map,
        image_size=image_size,
    )
    if pred_warped_panel is None:
        pred_warped_panel = _render_placeholder_panel(
            image_size,
            'Pred-Backwarped Current | 1_1',
            'pred warp unavailable',
        )
    else:
        pred_warped_panel = _draw_panel_title(pred_warped_panel,
                                              'Pred-Backwarped Current | 1_1')

    flow_panel = render_flow_bev(
        pred_flow,
        current_occ,
        dynamic_class_indices=dynamic_class_indices,
        image_size=image_size,
        max_speed=max_speed,
    )
    if flow_panel is None:
        flow_panel = _render_placeholder_panel(
            image_size,
            'Pred Flow | 1_1',
            'flow unavailable',
        )
    else:
        flow_panel = _draw_panel_title(flow_panel, 'Pred Flow | 1_1')

    gt_flow_panel = render_flow_bev(
        gt_flow,
        current_occ,
        dynamic_class_indices=dynamic_class_indices,
        image_size=image_size,
        max_speed=max_speed,
    )
    if gt_flow_panel is None:
        gt_flow_panel = _render_placeholder_panel(
            image_size,
            'GT Flow | 1_1',
            'gt flow unavailable',
        )
    else:
        gt_flow_panel = _draw_panel_title(gt_flow_panel, 'GT Flow | 1_1')

    gt_warped_panel = render_occ_bev(
        gt_backwarped_current,
        empty_idx=empty_idx,
        color_map=color_map,
        image_size=image_size,
    )
    if gt_warped_panel is None:
        gt_warped_panel = _render_placeholder_panel(
            image_size,
            'GT-Backwarped Current | 1_1',
            'gt warp unavailable',
        )
    else:
        gt_warped_panel = _draw_panel_title(gt_warped_panel,
                                            'GT-Backwarped Current | 1_1')

    grid_meta = _build_occ_grid_meta_np(current_occ, point_cloud_range)
    zero_result = None
    pred_hit_panel = None
    gt_hit_panel = None
    zero_hit_panel = None
    if rigid_history is not None and current_occ is not None and grid_meta is not None:
        zero_result = _build_warp_point_result(
            current_occ,
            flow=None,
            grid_meta=grid_meta,
            dynamic_class_indices=dynamic_class_indices,
            empty_idx=empty_idx,
            delta_t_sec=-0.5,
            use_flow=False,
        )
        pred_result = _build_warp_point_result(
            current_occ,
            flow=pred_flow,
            grid_meta=grid_meta,
            dynamic_class_indices=dynamic_class_indices,
            empty_idx=empty_idx,
            delta_t_sec=-0.5,
            use_flow=True,
        )
        gt_result = _build_warp_point_result(
            current_occ,
            flow=gt_flow,
            grid_meta=grid_meta,
            dynamic_class_indices=dynamic_class_indices,
            empty_idx=empty_idx,
            delta_t_sec=-0.5,
            use_flow=True,
        )
        if zero_result is not None:
            zero_hit_panel, _ = _render_warp_hit_panel(
                current_occ=rigid_history,
                warp_result=zero_result,
                zero_result=zero_result,
                dynamic_class_indices=dynamic_class_indices,
                empty_idx=empty_idx,
                color_map=color_map,
                image_size=image_size,
                title='Zero-Motion vs History',
                neighborhood_radius=1,
                point_size=0,
            )
        if pred_result is not None and zero_result is not None:
            pred_hit_panel, _ = _render_warp_hit_panel(
                current_occ=rigid_history,
                warp_result=pred_result,
                zero_result=zero_result,
                dynamic_class_indices=dynamic_class_indices,
                empty_idx=empty_idx,
                color_map=color_map,
                image_size=image_size,
                title='Pred Backwarp Hit Map',
                neighborhood_radius=1,
                point_size=0,
            )
        if gt_result is not None and zero_result is not None:
            gt_hit_panel, _ = _render_warp_hit_panel(
                current_occ=rigid_history,
                warp_result=gt_result,
                zero_result=zero_result,
                dynamic_class_indices=dynamic_class_indices,
                empty_idx=empty_idx,
                color_map=color_map,
                image_size=image_size,
                title='GT Backwarp Hit Map',
                neighborhood_radius=1,
                point_size=0,
            )

    if zero_hit_panel is None:
        zero_hit_panel = _render_placeholder_panel(
            image_size,
            'Zero-Motion vs History',
            'hit map unavailable',
        )
    if pred_hit_panel is None:
        pred_hit_panel = _render_placeholder_panel(
            image_size,
            'Pred Backwarp Hit Map',
            'hit map unavailable',
        )
    if gt_hit_panel is None:
        gt_hit_panel = _render_placeholder_panel(
            image_size,
            'GT Backwarp Hit Map',
            'hit map unavailable',
        )

    return cv2.vconcat([
        cv2.hconcat([current_panel, flow_panel, gt_flow_panel]),
        cv2.hconcat([aligned_panel, pred_warped_panel, gt_warped_panel]),
        cv2.hconcat([zero_hit_panel, pred_hit_panel, gt_hit_panel]),
    ])


def project_occ_to_bev_labels(semantics, empty_idx, num_classes=None):
    semantics = np.asarray(semantics)
    if semantics.ndim != 3:
        return None
    height, width, depth = semantics.shape
    semantics_valid = np.logical_not(semantics == empty_idx)
    if num_classes is not None:
        semantics_valid &= semantics >= 0
        semantics_valid &= semantics < int(num_classes)
    else:
        semantics_valid &= semantics != 255
    d = (np.arange(depth, dtype=np.float32) + 1).reshape(1, 1, depth)
    d = np.repeat(d, height, axis=0)
    d = np.repeat(d, width, axis=1)
    d = d * semantics_valid
    selected = np.argmax(d, axis=2)
    selected_torch = torch.from_numpy(selected)
    semantics_torch = torch.from_numpy(semantics)
    occ_bev_torch = torch.gather(
        semantics_torch, dim=2, index=selected_torch.unsqueeze(-1))
    occ_bev = occ_bev_torch.numpy().squeeze(-1).astype(np.int32)
    occ_bev[~semantics_valid.any(axis=2)] = int(empty_idx)
    return occ_bev


def render_occ_label_bev(occ_bev,
                         empty_idx,
                         color_map,
                         image_size=DEFAULT_IMAGE_SIZE):
    occ_bev = np.asarray(occ_bev)
    if occ_bev.ndim != 2:
        return None
    invalid_mask = np.logical_or(occ_bev < 0, occ_bev >= len(color_map))
    occ_bev = np.clip(occ_bev, 0, len(color_map) - 1)
    occ_bev_vis = color_map[occ_bev]
    occ_bev_vis[invalid_mask] = np.array([255, 255, 255], dtype=np.uint8)
    # Occupancy BEV labels are indexed as [x, y].
    occ_bev_vis = _orient_xy_grid_to_image(occ_bev_vis)[..., :3]
    occ_bev_vis = cv2.resize(
        occ_bev_vis, image_size, interpolation=cv2.INTER_NEAREST)
    return occ_bev_vis


def render_occ_bev(semantics,
                   empty_idx,
                   color_map,
                   image_size=DEFAULT_IMAGE_SIZE):
    occ_bev = project_occ_to_bev_labels(
        semantics, empty_idx, num_classes=len(color_map))
    if occ_bev is None:
        return None
    return render_occ_label_bev(
        occ_bev, empty_idx, color_map, image_size=image_size)


def render_binary_occ_bev(occupancy, image_size=DEFAULT_IMAGE_SIZE):
    occupancy = np.asarray(occupancy)
    if occupancy.ndim != 3:
        return None
    z_count = occupancy.sum(axis=2).astype(np.float32)
    if z_count.max() <= 0:
        canvas = np.full((*z_count.shape, 3), 255, dtype=np.uint8)
        canvas = _orient_xy_grid_to_image(canvas)
        return cv2.resize(canvas, image_size, interpolation=cv2.INTER_NEAREST)

    z_dim = max(int(occupancy.shape[2]), 1)
    z_count_norm = np.clip(z_count / float(z_dim), 0.0, 1.0)
    heatmap = cv2.applyColorMap(
        np.round(z_count_norm * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heatmap[z_count <= 0] = np.array([255, 255, 255], dtype=np.uint8)
    canvas = _orient_xy_grid_to_image(heatmap)
    return cv2.resize(canvas, image_size, interpolation=cv2.INTER_NEAREST)


def render_masked_occ_bev(semantics,
                          valid_mask,
                          empty_idx,
                          color_map,
                          image_size=DEFAULT_IMAGE_SIZE):
    semantics = np.asarray(semantics)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if semantics.ndim != 3 or valid_mask.shape != semantics.shape:
        return None
    masked_semantics = semantics.copy()
    masked_semantics[~valid_mask] = empty_idx
    return render_occ_bev(
        masked_semantics, empty_idx, color_map, image_size=image_size)


def project_flow_to_bev(flow, semantics, dynamic_class_indices):
    flow = np.asarray(flow, dtype=np.float32)
    semantics = np.asarray(semantics)
    if flow.ndim != 4 or flow.shape[
            -1] != 2 or semantics.shape != flow.shape[:3]:
        return None, None
    dynamic_mask = np.zeros_like(semantics, dtype=bool)
    for class_index in tuple(dynamic_class_indices or ()):
        dynamic_mask |= semantics == int(class_index)
    has_dynamic = dynamic_mask.any(axis=2)
    if not has_dynamic.any():
        return np.zeros(
            semantics.shape[:2] + (2, ), dtype=np.float32), has_dynamic

    z_index = (np.arange(semantics.shape[2], dtype=np.int32) + 1).reshape(
        1, 1, -1)
    selected = np.argmax(dynamic_mask.astype(np.int32) * z_index, axis=2)
    flow_bev = np.take_along_axis(
        flow,
        selected[..., None, None],
        axis=2,
    )[..., 0, :]
    flow_bev[~has_dynamic] = 0.0
    return flow_bev.astype(np.float32), has_dynamic


def _window_sum_2d(values, radius):
    values = np.asarray(values, dtype=np.float32)
    radius = max(int(radius), 0)
    if radius <= 0:
        return values.copy()
    padded = np.pad(
        values, ((radius, radius), (radius, radius)), mode='constant')
    height, width = values.shape
    window_sum = np.zeros((height, width), dtype=np.float32)
    for offset_x in range(2 * radius + 1):
        for offset_y in range(2 * radius + 1):
            window_sum += padded[
                offset_x:offset_x + height,
                offset_y:offset_y + width,
            ]
    return window_sum


def fill_flow_bev_holes(flow_bev,
                        valid_mask,
                        fill_iterations=2,
                        neighbor_radius=1,
                        min_neighbors=5):
    flow_bev = np.asarray(flow_bev, dtype=np.float32).copy()
    valid_mask = np.asarray(valid_mask, dtype=bool).copy()
    fill_iterations = max(int(fill_iterations), 0)
    if fill_iterations <= 0:
        return flow_bev, valid_mask

    for _ in range(fill_iterations):
        valid_float = valid_mask.astype(np.float32)
        neighbor_count = _window_sum_2d(valid_float, radius=neighbor_radius)
        fill_mask = np.logical_and(~valid_mask, neighbor_count
                                   >= float(min_neighbors))
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


def project_dynamic_flow_to_bev(flow,
                                semantics,
                                dynamic_class_indices,
                                fill_iterations=2,
                                neighbor_radius=1,
                                min_neighbors=5):
    flow = np.asarray(flow, dtype=np.float32)
    semantics = np.asarray(semantics)
    if flow.ndim != 4 or flow.shape[
            -1] != 2 or semantics.shape != flow.shape[:3]:
        return None, None
    dynamic_mask = np.zeros_like(semantics, dtype=bool)
    for class_idx in tuple(dynamic_class_indices or ()):
        dynamic_mask |= semantics == int(class_idx)
    has_dynamic = dynamic_mask.any(axis=2)
    if not has_dynamic.any():
        return np.zeros(
            semantics.shape[:2] + (2, ), dtype=np.float32), has_dynamic

    reversed_mask = dynamic_mask[:, :, ::-1]
    top_rev = np.argmax(reversed_mask, axis=2)
    top_idx = dynamic_mask.shape[2] - 1 - top_rev
    top_idx = np.where(has_dynamic, top_idx, 0).astype(np.int64)
    flow_bev = np.take_along_axis(
        flow, top_idx[..., None, None], axis=2)[..., 0, :]
    flow_bev[~has_dynamic] = 0.0
    return fill_flow_bev_holes(
        flow_bev,
        has_dynamic,
        fill_iterations=fill_iterations,
        neighbor_radius=neighbor_radius,
        min_neighbors=min_neighbors,
    )


def project_dense_flow_to_bev(flow):
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 4 or flow.shape[-1] != 2:
        return None, None
    magnitude = np.linalg.norm(flow, axis=-1)
    top_idx = np.argmax(magnitude, axis=2).astype(np.int64)
    flow_bev = np.take_along_axis(
        flow,
        top_idx[..., None, None],
        axis=2,
    )[..., 0, :]
    valid_mask = np.ones(flow.shape[:2], dtype=bool)
    return flow_bev.astype(np.float32), valid_mask


def flow_to_rgb(vx, vy, max_magnitude=8.0):
    vx = np.asarray(vx, dtype=np.float32)
    vy = np.asarray(vy, dtype=np.float32)
    magnitude = np.sqrt(vx**2 + vy**2)
    max_magnitude = max(float(max_magnitude), 1e-6)
    angle = np.arctan2(vy, vx)
    hue = (angle + np.pi) / (2.0 * np.pi)
    saturation = np.clip(magnitude / max_magnitude, 0.0, 1.0)
    value = np.ones_like(saturation)
    hsv = np.stack((hue, saturation, value), axis=-1)
    rgb = np.apply_along_axis(lambda item: colorsys.hsv_to_rgb(*item), -1, hsv)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def render_occ_bev_rgb(semantics,
                       empty_idx,
                       color_map,
                       image_size_hw=(480, 480)):
    occ_bev = project_occ_to_bev_labels(
        semantics, empty_idx, num_classes=len(color_map))
    invalid_mask = np.logical_or(occ_bev < 0, occ_bev >= len(color_map))
    occ_bev = np.clip(occ_bev, 0, len(color_map) - 1)
    occ_bev_rgb = color_map[occ_bev]
    occ_bev_rgb[invalid_mask] = np.array([255, 255, 255], dtype=np.uint8)
    occ_bev_rgb = _orient_xy_grid_to_image(occ_bev_rgb)
    return cv2.resize(
        occ_bev_rgb, (int(image_size_hw[1]), int(image_size_hw[0])),
        interpolation=cv2.INTER_NEAREST)


def _draw_flow_arrow(draw, start_pt, end_pt, color):
    draw.line([start_pt, end_pt], fill=color, width=1)
    dx = end_pt[0] - start_pt[0]
    dy = end_pt[1] - start_pt[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux = dx / length
    uy = dy / length
    head_len = min(6.0, max(3.0, length * 0.3))
    left = (
        end_pt[0] - head_len * ux + head_len * 0.45 * uy,
        end_pt[1] - head_len * uy - head_len * 0.45 * ux,
    )
    right = (
        end_pt[0] - head_len * ux - head_len * 0.45 * uy,
        end_pt[1] - head_len * uy + head_len * 0.45 * ux,
    )
    draw.polygon([end_pt, left, right], fill=color)


def draw_flow_arrows(image_rgb,
                     flow_bev,
                     valid_mask,
                     point_cloud_range,
                     arrow_stride=8,
                     arrow_horizon=0.5,
                     min_speed=0.5,
                     max_arrow_pixels=36.0):
    canvas = Image.fromarray(np.asarray(image_rgb))
    draw = ImageDraw.Draw(canvas)
    cells_x, cells_y = flow_bev.shape[:2]
    scale_x = canvas.width / float(cells_x)
    scale_y = canvas.height / float(cells_y)
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    voxel_size_x = (x_max - x_min) / max(cells_x, 1)
    voxel_size_y = (y_max - y_min) / max(cells_y, 1)
    stride = max(int(arrow_stride), 1)

    for x in range(0, cells_x, stride):
        for y in range(0, cells_y, stride):
            if not valid_mask[x, y]:
                continue
            vx, vy = flow_bev[x, y]
            speed = float(np.hypot(vx, vy))
            if speed < float(min_speed):
                continue
            start_col = (cells_x - 1 - x + 0.5) * scale_x
            start_row = (cells_y - 1 - y + 0.5) * scale_y
            delta_col = -(vx / max(voxel_size_x, 1e-6)) * scale_x * float(
                arrow_horizon)
            delta_row = -(vy / max(voxel_size_y, 1e-6)) * scale_y * float(
                arrow_horizon)
            arrow_length = float(np.hypot(delta_col, delta_row))
            if arrow_length > float(max_arrow_pixels):
                scale = float(max_arrow_pixels) / max(arrow_length, 1e-6)
                delta_col *= scale
                delta_row *= scale
            _draw_flow_arrow(
                draw,
                (start_col, start_row),
                (start_col + delta_col, start_row + delta_row),
                color=(32, 32, 32),
            )
    return np.asarray(canvas)


def render_flow_bev(flow,
                    semantics,
                    dynamic_class_indices,
                    image_size=DEFAULT_IMAGE_SIZE,
                    max_speed=8.0):
    flow_bev, valid_mask = project_dynamic_flow_to_bev(
        flow,
        semantics,
        dynamic_class_indices,
        fill_iterations=2,
        neighbor_radius=1,
        min_neighbors=5,
    )
    if flow_bev is None or valid_mask is None:
        return None
    flow_bev = flow_bev.copy()
    flow_bev[..., 0] = -flow_bev[..., 0]
    flow_bev[..., 1] = -flow_bev[..., 1]
    flow_bev = _orient_xy_grid_to_image(flow_bev)
    valid_mask = _orient_xy_grid_to_image(valid_mask.astype(
        np.uint8)).astype(bool)
    flow_rgb = flow_to_rgb(
        flow_bev[..., 0], flow_bev[..., 1], max_magnitude=max_speed)
    flow_rgb[~valid_mask] = np.array([255, 255, 255], dtype=np.uint8)
    return cv2.resize(flow_rgb, image_size, interpolation=cv2.INTER_NEAREST)


def render_dense_flow_bev(flow, image_size=DEFAULT_IMAGE_SIZE, max_speed=8.0):
    flow_bev, valid_mask = project_dense_flow_to_bev(flow)
    if flow_bev is None or valid_mask is None:
        return None
    flow_bev = flow_bev.copy()
    flow_bev[..., 0] = -flow_bev[..., 0]
    flow_bev[..., 1] = -flow_bev[..., 1]
    flow_bev = _orient_xy_grid_to_image(flow_bev)
    valid_mask = _orient_xy_grid_to_image(valid_mask.astype(
        np.uint8)).astype(bool)
    flow_rgb = flow_to_rgb(
        flow_bev[..., 0], flow_bev[..., 1], max_magnitude=max_speed)
    flow_rgb[~valid_mask] = np.array([255, 255, 255], dtype=np.uint8)
    return cv2.resize(flow_rgb, image_size, interpolation=cv2.INTER_NEAREST)


def render_multiscale_flow_comparison_bev(feature_payload,
                                          dynamic_class_indices,
                                          sample_meta=None,
                                          sample_index=0,
                                          image_size=(320, 320),
                                          max_speed=8.0,
                                          flow_gt_root=None):
    if not isinstance(feature_payload, dict):
        return None
    pred_flow_collection = dict(
        feature_payload.get('stage_occflows_predictions') or {})
    learned_prior_collection = dict(
        feature_payload.get('stage_learned_prior_predictions') or {})
    learned_blended_prior_collection = dict(
        feature_payload.get('stage_learned_blended_prior_predictions') or {})
    temporal_prior_collection = dict(
        feature_payload.get('stage_temporal_prior_predictions') or {})
    temporal_blended_prior_collection = dict(
        feature_payload.get('stage_temporal_blended_prior_predictions') or {})
    pred_occ_collection = dict(
        feature_payload.get('stage_occ_predictions') or {})
    true_flow_collection = dict(
        feature_payload.get('true_stage_voxel_occflows') or {})
    true_occ_collection = dict(
        feature_payload.get('true_stage_voxel_semantic') or {})

    scale_name_set = {
        scale_name
        for scale_name, _ in _feature_collection_items(pred_flow_collection)
        if scale_name != '1_1'
    }
    scale_name_set.update({
        scale_name
        for scale_name, _ in _feature_collection_items(true_flow_collection)
        if scale_name != '1_1'
    })
    scale_name_set.update({
        scale_name
        for scale_name, _ in _feature_collection_items(true_occ_collection)
        if scale_name != '1_1'
    })
    if sample_meta is not None:
        scale_name_set.update(('1_2', '1_4', '1_8'))
    scale_names = sorted(scale_name_set, key=_scale_sort_key)
    if not scale_names:
        return None

    true_panels = []
    learned_prior_panels = []
    learned_blended_panels = []
    temporal_prior_panels = []
    temporal_blended_panels = []
    pred_panels = []
    for scale_name in scale_names:
        true_flow = _to_stage_flow_volume(
            true_flow_collection.get(scale_name), sample_index)
        if true_flow is None and sample_meta is not None:
            true_flow = load_stage_occ_flow_from_meta(
                sample_meta, scale_name, flow_gt_root=flow_gt_root)
        true_panel = None
        if true_flow is not None:
            true_panel = render_dense_flow_bev(
                true_flow,
                image_size=image_size,
                max_speed=max_speed,
            )
        if true_panel is None:
            true_panel = _render_placeholder_panel(
                image_size,
                f'GT | {_format_scale_label(scale_name)}',
                'flow unavailable',
            )
        else:
            true_panel = _draw_panel_title(
                true_panel, f'GT | {_format_scale_label(scale_name)}')
        true_panels.append(true_panel)

        learned_prior_flow = _to_stage_flow_volume(
            learned_prior_collection.get(scale_name), sample_index)
        learned_prior_panel = None
        if learned_prior_flow is not None:
            learned_prior_panel = render_dense_flow_bev(
                learned_prior_flow,
                image_size=image_size,
                max_speed=max_speed,
            )
        if learned_prior_panel is None:
            learned_prior_panel = _render_placeholder_panel(
                image_size,
                f'Learned Prior | {_format_scale_label(scale_name)}',
                'prior unavailable',
            )
        else:
            learned_prior_panel = _draw_panel_title(
                learned_prior_panel,
                f'Learned Prior | {_format_scale_label(scale_name)}')
        learned_prior_panels.append(learned_prior_panel)

        learned_blended_flow = _to_stage_flow_volume(
            learned_blended_prior_collection.get(scale_name), sample_index)
        learned_blended_panel = None
        if learned_blended_flow is not None:
            learned_blended_panel = render_dense_flow_bev(
                learned_blended_flow,
                image_size=image_size,
                max_speed=max_speed,
            )
        if learned_blended_panel is None:
            learned_blended_panel = _render_placeholder_panel(
                image_size,
                f'Learned Prior Blended | {_format_scale_label(scale_name)}',
                'prior unavailable',
            )
        else:
            learned_blended_panel = _draw_panel_title(
                learned_blended_panel,
                f'Learned Prior Blended | {_format_scale_label(scale_name)}')
        learned_blended_panels.append(learned_blended_panel)

        temporal_prior_flow = _to_stage_flow_volume(
            temporal_prior_collection.get(scale_name), sample_index)
        temporal_prior_panel = None
        if temporal_prior_flow is not None:
            temporal_prior_panel = render_dense_flow_bev(
                temporal_prior_flow,
                image_size=image_size,
                max_speed=max_speed,
            )
        if temporal_prior_panel is None:
            temporal_prior_panel = _render_placeholder_panel(
                image_size,
                f'Temporal Prior | {_format_scale_label(scale_name)}',
                'prior unavailable',
            )
        else:
            temporal_prior_panel = _draw_panel_title(
                temporal_prior_panel,
                f'Temporal Prior | {_format_scale_label(scale_name)}')
        temporal_prior_panels.append(temporal_prior_panel)

        temporal_blended_flow = _to_stage_flow_volume(
            temporal_blended_prior_collection.get(scale_name), sample_index)
        temporal_blended_panel = None
        if temporal_blended_flow is not None:
            temporal_blended_panel = render_dense_flow_bev(
                temporal_blended_flow,
                image_size=image_size,
                max_speed=max_speed,
            )
        if temporal_blended_panel is None:
            temporal_blended_panel = _render_placeholder_panel(
                image_size,
                f'Temporal Prior Blended | {_format_scale_label(scale_name)}',
                'prior unavailable',
            )
        else:
            temporal_blended_panel = _draw_panel_title(
                temporal_blended_panel,
                f'Temporal Prior Blended | {_format_scale_label(scale_name)}')
        temporal_blended_panels.append(temporal_blended_panel)

        pred_flow = _to_stage_flow_volume(
            pred_flow_collection.get(scale_name), sample_index)
        pred_panel = None
        if pred_flow is not None:
            pred_panel = render_dense_flow_bev(
                pred_flow,
                image_size=image_size,
                max_speed=max_speed,
            )
        if pred_panel is None:
            pred_panel = _render_placeholder_panel(
                image_size,
                f'Pred | {_format_scale_label(scale_name)}',
                'flow unavailable',
            )
        else:
            pred_panel = _draw_panel_title(
                pred_panel, f'Pred | {_format_scale_label(scale_name)}')
        pred_panels.append(pred_panel)

    return cv2.vconcat([
        cv2.hconcat(true_panels),
        cv2.hconcat(learned_prior_panels),
        cv2.hconcat(learned_blended_panels),
        cv2.hconcat(temporal_prior_panels),
        cv2.hconcat(temporal_blended_panels),
        cv2.hconcat(pred_panels),
    ])


def _build_flow_panels(flow,
                       semantics,
                       dynamic_class_indices,
                       empty_idx,
                       color_map,
                       point_cloud_range,
                       image_size=(320, 320),
                       max_speed=8.0,
                       arrow_stride=4,
                       arrow_horizon=0.5,
                       min_speed=0.3,
                       max_arrow_pixels=36.0):
    semantics = np.asarray(semantics)
    if semantics.ndim != 3:
        return None
    semantic_panel = render_occ_bev_rgb(
        semantics,
        empty_idx=empty_idx,
        color_map=color_map,
        image_size_hw=image_size,
    )
    semantic_panel = _draw_panel_title(semantic_panel, 'Semantic BEV')

    flow_bev, valid_mask = project_dynamic_flow_to_bev(
        flow,
        semantics,
        dynamic_class_indices,
        fill_iterations=2,
        neighbor_radius=1,
        min_neighbors=5,
    )
    if flow_bev is None or valid_mask is None:
        return None

    flow_image = flow_bev.copy()
    flow_image[..., 0] = -flow_image[..., 0]
    flow_image[..., 1] = -flow_image[..., 1]
    flow_image = _orient_xy_grid_to_image(flow_image)
    valid_mask_image = _orient_xy_grid_to_image(valid_mask.astype(
        np.uint8)).astype(bool)
    flow_panel = flow_to_rgb(
        flow_image[..., 0],
        flow_image[..., 1],
        max_magnitude=max_speed,
    )
    flow_panel[~valid_mask_image] = np.array([255, 255, 255], dtype=np.uint8)
    flow_panel = cv2.resize(
        flow_panel, (int(image_size[0]), int(image_size[1])),
        interpolation=cv2.INTER_NEAREST)
    flow_panel = _draw_panel_title(flow_panel,
                                   f'Flow BEV | clip={max_speed:.1f}m/s')

    arrow_panel = draw_flow_arrows(
        render_occ_bev_rgb(
            semantics,
            empty_idx=empty_idx,
            color_map=color_map,
            image_size_hw=image_size,
        ),
        flow_bev=flow_bev,
        valid_mask=valid_mask,
        point_cloud_range=point_cloud_range,
        arrow_stride=arrow_stride,
        arrow_horizon=arrow_horizon,
        min_speed=min_speed,
        max_arrow_pixels=max_arrow_pixels,
    )
    arrow_panel = _draw_panel_title(
        arrow_panel,
        f'Semantic + Arrows | horizon={float(arrow_horizon):.2f}s')
    return semantic_panel, flow_panel, arrow_panel


def render_flow_comparison_bev(pred_flow,
                               gt_flow,
                               pred_occ,
                               gt_occ,
                               feature_payload,
                               dynamic_class_indices,
                               empty_idx,
                               color_map,
                               point_cloud_range,
                               sample_index=0,
                               image_size=(320, 320),
                               max_speed=8.0,
                               arrow_stride=4,
                               arrow_horizon=0.5,
                               min_speed=0.3,
                               max_arrow_pixels=36.0):
    pred_panels = _build_flow_panels(
        pred_flow,
        pred_occ,
        dynamic_class_indices=dynamic_class_indices,
        empty_idx=empty_idx,
        color_map=color_map,
        point_cloud_range=point_cloud_range,
        image_size=image_size,
        max_speed=max_speed,
        arrow_stride=arrow_stride,
        arrow_horizon=arrow_horizon,
        min_speed=min_speed,
        max_arrow_pixels=max_arrow_pixels,
    )
    gt_panels = _build_flow_panels(
        gt_flow,
        gt_occ,
        dynamic_class_indices=dynamic_class_indices,
        empty_idx=empty_idx,
        color_map=color_map,
        point_cloud_range=point_cloud_range,
        image_size=image_size,
        max_speed=max_speed,
        arrow_stride=arrow_stride,
        arrow_horizon=arrow_horizon,
        min_speed=min_speed,
        max_arrow_pixels=max_arrow_pixels,
    )
    if pred_panels is None or gt_panels is None:
        return None
    _, pred_flow_panel, pred_arrow_panel = pred_panels
    _, gt_flow_panel, gt_arrow_panel = gt_panels
    pred_arrow_panel = _draw_panel_title(pred_arrow_panel,
                                         'Pred Occ + Pred Flow Arrows')
    gt_arrow_panel = _draw_panel_title(gt_arrow_panel,
                                       'GT Occ + GT Flow Arrows')
    pred_flow_panel = _draw_panel_title(pred_flow_panel, 'Pred Flow')
    gt_flow_panel = _draw_panel_title(gt_flow_panel, 'GT Flow')
    candidate_panel = render_current_dynamic_candidate_panel(
        feature_payload,
        sample_index=sample_index,
        image_size=image_size,
    )
    if candidate_panel is None:
        candidate_panel = _render_placeholder_panel(image_size,
                                                    'Candidate Map',
                                                    'candidate unavailable')
    prior_panel = render_flow_prior_panel(
        feature_payload,
        pred_occ=pred_occ,
        dynamic_class_indices=dynamic_class_indices,
        sample_index=sample_index,
        image_size=image_size,
        max_speed=max_speed,
    )
    if prior_panel is None:
        prior_panel = _render_placeholder_panel(image_size, 'Prior Flow',
                                                'prior unavailable')
    dense_pred_panel = render_dense_flow_panel(
        feature_payload.get('dense_flow_results') if isinstance(
            feature_payload, dict) else None,
        sample_index=sample_index,
        image_size=image_size,
        max_speed=max_speed,
        title='Dense Pred Flow',
    )
    if dense_pred_panel is None:
        dense_pred_panel = _render_placeholder_panel(image_size,
                                                     'Dense Pred Flow',
                                                     'dense pred unavailable')
    dense_prior_panel = render_dense_flow_panel(
        feature_payload.get('flow_prior')
        if isinstance(feature_payload, dict) else None,
        sample_index=sample_index,
        image_size=image_size,
        max_speed=max_speed,
        title='Dense Prior Flow',
    )
    if dense_prior_panel is None:
        dense_prior_panel = _render_placeholder_panel(
            image_size, 'Dense Prior Flow', 'dense prior unavailable')
    top_row = cv2.hconcat(
        [pred_arrow_panel, gt_arrow_panel, candidate_panel, dense_pred_panel])
    bottom_row = cv2.hconcat(
        [pred_flow_panel, gt_flow_panel, prior_panel, dense_prior_panel])
    return cv2.vconcat([top_row, bottom_row])


def render_occ_error_bev(pred_semantics,
                         gt_semantics,
                         empty_idx,
                         image_size=DEFAULT_IMAGE_SIZE):
    pred_semantics = np.asarray(pred_semantics)
    gt_semantics = np.asarray(gt_semantics)
    if pred_semantics.shape != gt_semantics.shape or pred_semantics.ndim != 3:
        return None

    pred_occ = pred_semantics != empty_idx
    gt_occ = gt_semantics != empty_idx
    error_volume = np.zeros_like(pred_semantics, dtype=np.uint8)
    error_volume[np.logical_and(pred_occ, np.logical_not(gt_occ))] = 1
    error_volume[np.logical_and(np.logical_not(pred_occ), gt_occ)] = 2
    error_volume[np.logical_and(pred_occ, gt_occ)
                 & (pred_semantics != gt_semantics)] = 3

    depth = error_volume.shape[2]
    score = np.arange(depth, dtype=np.float32).reshape(1, 1, depth)
    score = np.repeat(score, error_volume.shape[0], axis=0)
    score = np.repeat(score, error_volume.shape[1], axis=1)
    score = score * (error_volume > 0)
    selected = np.argmax(score, axis=2)
    error_bev = np.take_along_axis(
        error_volume, selected[..., None], axis=2)[..., 0]

    color_map = np.array([
        [255, 255, 255],
        [255, 0, 255],
        [255, 255, 0],
        [0, 0, 255],
    ],
                         dtype=np.uint8)
    error_vis = color_map[error_bev]
    error_vis = _orient_xy_grid_to_image(error_vis)
    return cv2.resize(error_vis, image_size, interpolation=cv2.INTER_NEAREST)


def _ensure_parent_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _draw_panel_title(image_bgr, title):
    if image_bgr is None:
        return None
    image_bgr = image_bgr.copy()
    cv2.rectangle(image_bgr, (0, 0), (image_bgr.shape[1] - 1, 28),
                  (255, 255, 255), -1)
    cv2.putText(
        image_bgr,
        str(title),
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _PLACEHOLDER_FG,
        1,
        cv2.LINE_AA,
    )
    return image_bgr


def compose_occ_region_bev(pred_occ,
                           empty_idx,
                           color_map,
                           static_prior=None,
                           image_size=DEFAULT_IMAGE_SIZE):
    pred_occ = np.asarray(pred_occ)
    if pred_occ.ndim != 3:
        return None

    occ_bev = project_occ_to_bev_labels(
        pred_occ, empty_idx, num_classes=len(color_map))
    if occ_bev is None:
        return None

    occupied_bev = render_occ_label_bev(
        occ_bev, empty_idx, color_map, image_size=image_size)
    prior_bev = render_binary_occ_bev(static_prior, image_size=image_size)

    if occupied_bev is None:
        occupied_bev = np.full((image_size[1], image_size[0], 3),
                               255,
                               dtype=np.uint8)
    if prior_bev is None:
        prior_bev = np.full((image_size[1], image_size[0], 3),
                            255,
                            dtype=np.uint8)

    occupied_bev = _draw_panel_title(occupied_bev, 'Occupied Semantics')
    prior_bev = _draw_panel_title(prior_bev, 'Static Prior Binary BEV')
    return cv2.hconcat([occupied_bev, prior_bev])


def _save_image(path, image_bgr, info_lines):
    _ensure_parent_dir(path)
    cv2.imwrite(path, add_info_banner(image_bgr, info_lines))


def _compose_occ_comparison_panels(pred_panel, gt_panel, pred_title, gt_title):
    if pred_panel is None:
        pred_panel = _create_placeholder_panel('pred unavailable')
    if gt_panel is None:
        gt_panel = _create_placeholder_panel(
            'gt unavailable',
            size=(pred_panel.shape[1], pred_panel.shape[0]),
        )
    if pred_panel.shape[:2] != gt_panel.shape[:2]:
        gt_panel = cv2.resize(
            gt_panel,
            (pred_panel.shape[1], pred_panel.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    pred_panel = _draw_panel_title(pred_panel, pred_title)
    gt_panel = _draw_panel_title(gt_panel, gt_title)
    return cv2.hconcat([pred_panel, gt_panel])


def _occ_range_and_voxel_size(semantics, point_cloud_range):
    shape = np.array(semantics.shape, dtype=np.float32)
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    voxel_size = (
        float((x_max - x_min) / max(shape[0], 1)),
        float((y_max - y_min) / max(shape[1], 1)),
        float((z_max - z_min) / max(shape[2], 1)),
    )
    return voxel_size, (x_min, y_min, z_min, x_max, y_max, z_max)


def _render_occ_3d_matplotlib(semantics,
                              point_cloud_range,
                              color_map,
                              empty_idx,
                              image_size=DEFAULT_IMAGE_SIZE,
                              max_voxels=50000):
    semantics = np.asarray(semantics)
    valid_mask = np.logical_and(semantics != empty_idx, semantics != 255)
    coords = np.argwhere(valid_mask)
    if coords.size == 0:
        return None

    classes = semantics[valid_mask].astype(np.int32)
    if coords.shape[0] > max_voxels:
        stride = int(np.ceil(coords.shape[0] / float(max_voxels)))
        coords = coords[::stride]
        classes = classes[::stride]

    voxel_size, occ_range = _occ_range_and_voxel_size(semantics,
                                                      point_cloud_range)
    x_min, y_min, z_min, x_max, y_max, z_max = occ_range
    xs = x_min + (coords[:, 0] + 0.5) * voxel_size[0]
    ys = y_min + (coords[:, 1] + 0.5) * voxel_size[1]
    zs = z_min + (coords[:, 2] + 0.5) * voxel_size[2]
    colors = color_map[np.clip(classes, 0,
                               len(color_map) - 1)][:, ::-1] / 255.0

    import matplotlib
    matplotlib.use('Agg', force=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    dpi = 100
    fig = Figure(
        figsize=(image_size[0] / dpi, image_size[1] / dpi),
        dpi=dpi,
        facecolor='white')
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        xs,
        ys,
        zs,
        c=colors,
        s=8,
        marker='s',
        linewidths=0,
        depthshade=False,
    )
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((max(x_max - x_min,
                           1e-6), max(y_max - y_min,
                                      1e-6), max(z_max - z_min, 1e-6)))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_axis_off()
    ax.set_facecolor('white')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[..., :3]
    return np.ascontiguousarray(image[..., ::-1])


def _image_has_visible_content(image_bgr):
    if image_bgr is None or image_bgr.size == 0:
        return False
    if float(image_bgr.std()) < 1.0:
        return False
    if np.unique(image_bgr.reshape(-1, 3), axis=0).shape[0] < 8:
        return False
    return True


def _has_nonempty_occ(semantics, empty_idx):
    semantics = np.asarray(semantics)
    if semantics.size == 0:
        return False
    valid_mask = np.logical_and(semantics != empty_idx, semantics != 255)
    return bool(np.any(valid_mask))


def _render_occ_3d_image(semantics, point_cloud_range, color_map, empty_idx,
                         info_lines, view_json_path):
    if semantics is None:
        return _create_placeholder_panel('occupancy unavailable')
    image = None
    has_display = bool(
        os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
    if has_display:
        try:
            from open3d import io as o3d_io
            from tools.visualizer.occupancy_visualizer import OccupancyVisualizer

            voxel_size, occ_range = _occ_range_and_voxel_size(
                semantics, point_cloud_range)
            visualizer = OccupancyVisualizer(
                color_map=color_map,
                background_color=(255, 255, 255),
            )
            view_json = None
            if os.path.exists(view_json_path):
                view_json = o3d_io.read_pinhole_camera_parameters(
                    view_json_path)
            with tempfile.NamedTemporaryFile(
                    suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name
            try:
                visualizer.vis_occ(
                    semantics,
                    save_path=temp_path,
                    view_json=view_json,
                    ignore_labels=[empty_idx, 255],
                    voxelSize=voxel_size,
                    range=occ_range,
                    wait_time=0.05,
                )
                if os.path.exists(temp_path):
                    image = cv2.imread(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception:
            image = None

    if not _image_has_visible_content(image):
        try:
            image = _render_occ_3d_matplotlib(semantics, point_cloud_range,
                                              color_map, empty_idx)
        except Exception as exc:
            return _create_placeholder_panel(f'3D render failed: {exc}')

    if not _image_has_visible_content(image):
        message = ('no occupied voxels'
                   if not _has_nonempty_occ(semantics, empty_idx) else
                   '3D render unavailable')
        image = _create_placeholder_panel(message)

    return image


def _save_occ_3d_render(path, semantics, point_cloud_range, color_map,
                        empty_idx, info_lines, view_json_path):
    _ensure_parent_dir(path)
    image = _render_occ_3d_image(
        semantics,
        point_cloud_range,
        color_map,
        empty_idx,
        info_lines,
        view_json_path,
    )
    cv2.imwrite(path, add_info_banner(image, info_lines))


def save_occ_visualizations(output_dirs,
                            split,
                            epoch,
                            vis_index,
                            sample_meta,
                            img_inputs,
                            points,
                            voxel_feat,
                            pred_occ,
                            pred_flow=None,
                            gt_occ=None,
                            gt_flow=None,
                            pred_depth=None,
                            gt_depth=None,
                            point_cloud_range=None,
                            empty_idx=None,
                            color_map=None,
                            sample_index=0,
                            camera_num_frame=1,
                            view_json_path=None,
                            static_prior=None,
                            depth_bin_config=None,
                            gt_depth_mode='raw',
                            gt_depth_downsample=1,
                            gt_depth_sid=False,
                            feature_payload=None,
                            bda_mat=None,
                            save_history_bev=True,
                            dynamic_class_indices=None,
                            flow_gt_root=None,
                            flow_vis_max_speed=8.0,
                            camera_max_panels=4):
    pred_occ = _to_numpy_occ(pred_occ, sample_index)
    pred_flow = _to_numpy_flow(pred_flow, sample_index)
    gt_occ = _to_numpy_occ(gt_occ, sample_index)
    gt_flow = _to_numpy_flow(gt_flow, sample_index)
    if isinstance(voxel_feat, (list, tuple)):
        voxel_feat = voxel_feat[0] if len(voxel_feat) > 0 else None
    voxel_feat = _to_numpy(voxel_feat)
    if voxel_feat is not None and voxel_feat.ndim == 5:
        voxel_feat = voxel_feat[sample_index]
    static_prior = _to_numpy_prior(
        static_prior,
        sample_index,
        target_shape=None if pred_occ is None else pred_occ.shape)
    feature_payload = {} if feature_payload is None else feature_payload
    dynamic_class_indices = tuple(dynamic_class_indices or ())
    if gt_flow is None:
        gt_flow = load_occ_flow_from_meta(
            sample_meta, flow_gt_root=flow_gt_root)

    scene_name = str(sample_meta.get('scene_name', 'unknown_scene'))
    sample_token = _sample_meta_cache_key(sample_meta)
    token = sample_token or 'unknown_token'
    info_lines = [
        f'{split} | epoch {int(epoch):03d} | vis {int(vis_index):06d}',
        f'scene: {scene_name}',
        f'token: {token}',
    ]
    file_prefix = f'{int(epoch):03d}_{int(vis_index):06d}'

    artifact_paths = OrderedDict()

    def build_path(type_name, suffix=''):
        filename = f'{file_prefix}_{OCC_VISUALIZATION_TYPES[type_name]}{suffix}.png'
        path = os.path.join(output_dirs[split][type_name], filename)
        artifact_paths[f'{type_name}{suffix}'] = path
        return path

    camera_image = render_camera_grid(
        img_inputs,
        sample_index,
        camera_num_frame=camera_num_frame,
        cam_names=sample_meta.get('cam_names'),
        max_panels=camera_max_panels)
    if camera_image is None:
        camera_image = create_placeholder_image(info_lines,
                                                'camera unavailable')
        cv2.imwrite(build_path('camera_grid'), camera_image)
    else:
        _save_image(build_path('camera_grid'), camera_image, info_lines)

    depth_image = render_depth_grid(
        pred_depth,
        gt_depth,
        img_inputs=img_inputs,
        sample_index=sample_index,
        cam_names=sample_meta.get('cam_names'),
        depth_bin_config=depth_bin_config,
        camera_num_frame=camera_num_frame,
        panel_size=infer_camera_panel_size(
            img_inputs, sample_index=sample_index),
        gt_depth_mode=gt_depth_mode,
        gt_depth_downsample=gt_depth_downsample,
        gt_depth_sid=gt_depth_sid,
        max_panels=camera_max_panels)
    if depth_image is None:
        cv2.imwrite(
            build_path('depth_grid'),
            create_placeholder_image(info_lines, 'depth unavailable'))
    else:
        _save_image(build_path('depth_grid'), depth_image, info_lines)

    if ('lidar_bev' in output_dirs[split]
            or 'lidar_history_bev' in output_dirs[split]):
        point_batch = points[sample_index] if isinstance(
            points, (list, tuple)) else points
        point_array = _to_numpy(point_batch)
        bev_raster_size = None
        if pred_occ is not None and pred_occ.ndim == 3:
            bev_raster_size = (int(pred_occ.shape[0]), int(pred_occ.shape[1]))
        elif gt_occ is not None and gt_occ.ndim == 3:
            bev_raster_size = (int(gt_occ.shape[0]), int(gt_occ.shape[1]))
        if 'lidar_bev' in output_dirs[split]:
            lidar_bev = render_points_bev(
                point_array, point_cloud_range, raster_size=bev_raster_size)
            if lidar_bev is None:
                cv2.imwrite(
                    build_path('lidar_bev'),
                    create_placeholder_image(info_lines, 'lidar unavailable'))
            else:
                _save_image(build_path('lidar_bev'), lidar_bev, info_lines)

        if save_history_bev and 'lidar_history_bev' in output_dirs[split]:
            history_bev = render_history_points_bev(
                point_array,
                sample_meta.get('points_frame_splits'),
                point_cloud_range,
                raster_size=bev_raster_size)
            if history_bev is None:
                cv2.imwrite(
                    build_path('lidar_history_bev'),
                    create_placeholder_image(info_lines,
                                             'history lidar unavailable'))
            else:
                _save_image(
                    build_path('lidar_history_bev'), history_bev, info_lines)

    bev_feat = render_bev_feature(voxel_feat)
    if 'bev_feat' in output_dirs[split]:
        if bev_feat is None:
            cv2.imwrite(
                build_path('bev_feat'),
                create_placeholder_image(info_lines,
                                         'bev feature unavailable'))
        else:
            _save_image(build_path('bev_feat'), bev_feat, info_lines)

    if 'modality_feat_grid' in output_dirs[split]:
        modality_feat_grid = render_modality_feature_grid(
            feature_payload.get('modality_features'),
            sample_index=sample_index,
            final_bev_feature=voxel_feat)
        if modality_feat_grid is None:
            cv2.imwrite(
                build_path('modality_feat_grid'),
                create_placeholder_image(info_lines,
                                         'modality features unavailable'))
        else:
            _save_image(
                build_path('modality_feat_grid'), modality_feat_grid,
                info_lines)

    if 'recursive_occ_bev' in output_dirs[split]:
        recursive_occ_bev = render_recursive_occ_grid(
            feature_payload,
            pred_occ,
            empty_idx=empty_idx,
            color_map=color_map,
            sample_index=sample_index)
        if recursive_occ_bev is None:
            cv2.imwrite(
                build_path('recursive_occ_bev'),
                create_placeholder_image(info_lines,
                                         'recursive occupancy unavailable'))
        else:
            _save_image(
                build_path('recursive_occ_bev'), recursive_occ_bev, info_lines)

    if 'occ_static_bev' in output_dirs[split]:
        static_occ_bev = render_static_occ_grid(
            feature_payload,
            empty_idx=empty_idx,
            color_map=color_map,
            sample_index=sample_index)
        if static_occ_bev is None:
            cv2.imwrite(
                build_path('occ_static_bev'),
                create_placeholder_image(info_lines,
                                         'static occupancy unavailable'))
        else:
            _save_image(
                build_path('occ_static_bev'), static_occ_bev, info_lines)

    if 'dynamic_candidate_bev' in output_dirs[split]:
        dynamic_candidate_bev = render_dynamic_candidate_grid(
            feature_payload, sample_index=sample_index)
        if dynamic_candidate_bev is None:
            cv2.imwrite(
                build_path('dynamic_candidate_bev'),
                create_placeholder_image(info_lines,
                                         'dynamic candidate unavailable'))
        else:
            _save_image(
                build_path('dynamic_candidate_bev'), dynamic_candidate_bev,
                info_lines)

    if 'dynamic_occ_bev' in output_dirs[split]:
        dynamic_occ_bev = render_dynamic_occ_grid(
            feature_payload,
            gt_occ,
            empty_idx=empty_idx,
            color_map=color_map,
            dynamic_class_indices=dynamic_class_indices,
            sample_index=sample_index)
        if dynamic_occ_bev is None:
            cv2.imwrite(
                build_path('dynamic_occ_bev'),
                create_placeholder_image(info_lines,
                                         'dynamic occupancy unavailable'))
        else:
            _save_image(
                build_path('dynamic_occ_bev'), dynamic_occ_bev, info_lines)

    if 'temporal_topk_bev' in output_dirs[split]:
        temporal_topk_bev = render_temporal_topk_grid(
            feature_payload,
            sample_meta=sample_meta,
            gt_occ=gt_occ,
            empty_idx=empty_idx,
            color_map=color_map,
            dynamic_class_indices=dynamic_class_indices,
            sample_index=sample_index,
        )
        if temporal_topk_bev is None:
            cv2.imwrite(
                build_path('temporal_topk_bev'),
                create_placeholder_image(info_lines,
                                         'temporal top-k unavailable'))
        else:
            _save_image(
                build_path('temporal_topk_bev'), temporal_topk_bev, info_lines)

    if 'footprint_token_bev' in output_dirs[split]:
        footprint_token_bev = render_footprint_token_grid(
            feature_payload, sample_index=sample_index)
        if footprint_token_bev is None:
            cv2.imwrite(
                build_path('footprint_token_bev'),
                create_placeholder_image(info_lines,
                                         'PTR footprint tokens unavailable'))
        else:
            _save_image(
                build_path('footprint_token_bev'), footprint_token_bev,
                info_lines)

    if 'transition_gate_bev' in output_dirs[split]:
        transition_gate_bev = render_transition_gate_grid(
            feature_payload,
            pred_occ=pred_occ,
            gt_occ=gt_occ,
            gt_flow=gt_flow,
            dynamic_class_indices=dynamic_class_indices,
            empty_idx=empty_idx,
            sample_index=sample_index,
        )
        if transition_gate_bev is None:
            cv2.imwrite(
                build_path('transition_gate_bev'),
                create_placeholder_image(info_lines,
                                         'PTR transition gate unavailable'))
        else:
            _save_image(
                build_path('transition_gate_bev'), transition_gate_bev,
                info_lines)

    if 'canonical_ptr_bev' in output_dirs[split]:
        canonical_ptr_bev = render_canonical_ptr_grid(
            feature_payload, sample_index=sample_index)
        # Canonical PTR is an optional supervised-state diagnostic.  Omit it
        # for samples without a valid native P/T/R target rather than saving a
        # placeholder into the same-token evidence set.
        if canonical_ptr_bev is not None:
            _save_image(build_path('canonical_ptr_bev'), canonical_ptr_bev,
                        info_lines)

    if 'occ_flow_bev' in output_dirs[split]:
        occ_flow_bev = render_flow_comparison_bev(
            pred_flow=pred_flow,
            gt_flow=gt_flow,
            pred_occ=pred_occ,
            gt_occ=gt_occ,
            feature_payload=feature_payload,
            dynamic_class_indices=dynamic_class_indices,
            empty_idx=empty_idx,
            color_map=color_map,
            point_cloud_range=point_cloud_range,
            sample_index=sample_index,
            image_size=(480, 480),
            max_speed=flow_vis_max_speed,
        )
        if occ_flow_bev is None:
            cv2.imwrite(
                build_path('occ_flow_bev'),
                create_placeholder_image(info_lines, 'flow unavailable'))
        else:
            _save_image(build_path('occ_flow_bev'), occ_flow_bev, info_lines)

    if 'stage_occ_flow_bev' in output_dirs[split]:
        stage_occ_flow_bev = render_multiscale_flow_comparison_bev(
            feature_payload=feature_payload,
            dynamic_class_indices=dynamic_class_indices,
            sample_meta=sample_meta,
            sample_index=sample_index,
            image_size=(320, 320),
            max_speed=flow_vis_max_speed,
            flow_gt_root=flow_gt_root,
        )
        if stage_occ_flow_bev is None:
            cv2.imwrite(
                build_path('stage_occ_flow_bev'),
                create_placeholder_image(info_lines, 'stage flow unavailable'))
        else:
            _save_image(
                build_path('stage_occ_flow_bev'), stage_occ_flow_bev,
                info_lines)

    if 'sparse_warp_bev' in output_dirs[split]:
        sparse_warp_bev = render_sparse_warp_grid(
            sample_meta=sample_meta,
            gt_occ=gt_occ,
            gt_flow=gt_flow,
            pred_flow=pred_flow,
            dynamic_class_indices=dynamic_class_indices,
            point_cloud_range=point_cloud_range,
            empty_idx=empty_idx,
            color_map=color_map,
            sample_index=sample_index,
            image_size=(480, 480),
            bda_mat=bda_mat,
            max_speed=flow_vis_max_speed,
            flow_gt_root=flow_gt_root,
        )
        if sparse_warp_bev is None:
            cv2.imwrite(
                build_path('sparse_warp_bev'),
                create_placeholder_image(info_lines,
                                         'sparse warp unavailable'))
        else:
            _save_image(
                build_path('sparse_warp_bev'), sparse_warp_bev, info_lines)

    if 'prior_static_bias_bev' in output_dirs[split]:
        prior_static_bias_bev = render_prior_static_bias_grid(
            feature_payload,
            color_map=color_map,
            empty_idx=empty_idx,
            sample_index=sample_index)
        if prior_static_bias_bev is None:
            cv2.imwrite(
                build_path('prior_static_bias_bev'),
                create_placeholder_image(info_lines,
                                         'prior static bias unavailable'))
        else:
            _save_image(
                build_path('prior_static_bias_bev'), prior_static_bias_bev,
                info_lines)

    pred_bev = render_occ_bev(pred_occ, empty_idx, color_map)
    gt_bev = render_occ_bev(gt_occ, empty_idx,
                            color_map) if gt_occ is not None else None
    if 'occ_bev' in output_dirs[split]:
        occ_bev = _compose_occ_comparison_panels(
            pred_bev if pred_bev is not None else
            _create_placeholder_panel('pred occupancy unavailable'),
            gt_bev if gt_bev is not None else
            _create_placeholder_panel('gt occupancy unavailable'),
            'Pred Occupancy BEV',
            'GT Occupancy BEV',
        )
        _save_image(build_path('occ_bev'), occ_bev, info_lines)
    if 'occ_pred_bev' in output_dirs[split]:
        if pred_bev is None:
            cv2.imwrite(
                build_path('occ_pred_bev'),
                create_placeholder_image(info_lines,
                                         'pred occupancy unavailable'))
        else:
            _save_image(build_path('occ_pred_bev'), pred_bev, info_lines)
    if 'occ_gt_bev' in output_dirs[split]:
        if gt_bev is None:
            cv2.imwrite(
                build_path('occ_gt_bev'),
                create_placeholder_image(info_lines,
                                         'gt occupancy unavailable'))
        else:
            _save_image(build_path('occ_gt_bev'), gt_bev, info_lines)

    if 'occ_fg_bg_prior_bev' in output_dirs[split]:
        region_bev = compose_occ_region_bev(
            pred_occ, empty_idx, color_map, static_prior=static_prior)
        if region_bev is None:
            cv2.imwrite(
                build_path('occ_fg_bg_prior_bev'),
                create_placeholder_image(
                    info_lines, 'pred occupancy or static prior unavailable'))
        else:
            _save_image(
                build_path('occ_fg_bg_prior_bev'), region_bev, info_lines)

    if ('occ_3d' in output_dirs[split] or 'occ_pred_3d' in output_dirs[split]
            or 'occ_gt_3d' in output_dirs[split]):
        if view_json_path is None:
            render_root = output_dirs[split].get(
                'occ_3d', output_dirs[split].get(
                    'occ_pred_3d', output_dirs[split].get('occ_gt_3d')))
            view_json_path = os.path.join(
                os.path.dirname(render_root), 'occ_view.json')
        if 'occ_3d' in output_dirs[split]:
            pred_occ_3d = _render_occ_3d_image(
                pred_occ,
                point_cloud_range,
                color_map,
                empty_idx,
                info_lines,
                view_json_path,
            )
            gt_occ_3d = _render_occ_3d_image(
                gt_occ,
                point_cloud_range,
                color_map,
                empty_idx,
                info_lines,
                view_json_path,
            )
            occ_3d = _compose_occ_comparison_panels(
                pred_occ_3d,
                gt_occ_3d,
                'Pred Occupancy 3D',
                'GT Occupancy 3D',
            )
            _save_image(build_path('occ_3d'), occ_3d, info_lines)
        if 'occ_pred_3d' in output_dirs[split]:
            _save_occ_3d_render(
                build_path('occ_pred_3d'),
                pred_occ,
                point_cloud_range,
                color_map,
                empty_idx,
                info_lines,
                view_json_path,
            )
        if 'occ_gt_3d' in output_dirs[split]:
            _save_occ_3d_render(
                build_path('occ_gt_3d'),
                gt_occ,
                point_cloud_range,
                color_map,
                empty_idx,
                info_lines,
                view_json_path,
            )

    _append_same_token_manifest(
        output_dirs=output_dirs,
        split=split,
        epoch=epoch,
        vis_index=vis_index,
        scene_name=scene_name,
        sample_token=sample_token,
        artifact_paths=artifact_paths,
    )
