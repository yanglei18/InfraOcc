import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint, wrap_fp16_model

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mmdet
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

from mmdet3d.core.visualizer.occ_visualization import (
    _render_occ_3d_image,
    make_occ_color_map,
    render_camera_grid,
    render_occ_bev,
    render_points_bev,
)
from tools.infraocc.common import flatten_results, get_split_cfg, load_config, load_occ_gt

if mmdet.__version__ > '2.23.0':
    from mmdet.utils import compat_cfg, setup_multi_processes
else:
    from mmdet3d.utils import compat_cfg, setup_multi_processes

CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free'
]
DYNAMIC_INDICES = np.array([2, 3, 4, 6, 7, 10], dtype=np.uint8)
STATIC_INDICES = np.array([0, 1, 5, 8, 9, 11, 12, 13, 14, 15, 16],
                          dtype=np.uint8)
EMPTY_IDX = 17
COLOR_MAP = make_occ_color_map(CLASS_NAMES)
DYNAMIC_ERROR_LEGEND = [
    ('Correct', (70, 170, 70)),
    ('Missed', (70, 70, 230)),
    ('False', (230, 90, 60)),
]
DYNAMIC_CHANGE_LEGEND = [
    ('Already correct', (120, 120, 120)),
    ('Recovered by ProSD', (50, 170, 70)),
    ('Still missed', (70, 70, 230)),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate InfraOcc paper figures.')
    parser.add_argument(
        '--output-dir', default='projects/InfraOcc/figures_update')
    parser.add_argument('--max-samples', type=int, default=24)
    parser.add_argument('--num-select', type=int, default=4)
    parser.add_argument(
        '--selected-indices',
        nargs='*',
        type=int,
        default=None,
        help='Validation sample indices to render directly. When set, the '
        'script skips candidate ranking and reruns only these samples.')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--config-plain',
        default=
        'projects/InfraOcc/configs/main_table_update/infraocc_c_4x4_36e_plain.py'
    )
    parser.add_argument(
        '--ckpt-plain',
        default='projects/InfraOcc/checkpoints_update/c_plain.pth')
    parser.add_argument(
        '--config-c',
        default=
        'projects/InfraOcc/configs/main_table_update/infraocc_c_4x4_36e_prosd.py'
    )
    parser.add_argument(
        '--ckpt-c', default='projects/InfraOcc/checkpoints_update/c_prosd.pth')
    parser.add_argument(
        '--config-l',
        default=
        'projects/InfraOcc/configs/main_table_update/infraocc_l_4x4_36e_prosd.py'
    )
    parser.add_argument(
        '--ckpt-l', default='projects/InfraOcc/checkpoints_update/l_prosd.pth')
    parser.add_argument(
        '--config-m',
        default=
        'projects/InfraOcc/configs/main_table_update/infraocc_m_4x4_36e_prosd.py'
    )
    parser.add_argument(
        '--ckpt-m', default='projects/InfraOcc/checkpoints_update/m_prosd.pth')
    parser.add_argument(
        '--skip-modality',
        action='store_true',
        help='Only generate camera/plain/progressive figures.')
    parser.add_argument(
        '--skip-multimodal',
        action='store_true',
        help='Generate LiDAR assets but skip the C+L model.')
    parser.add_argument(
        '--skip-3d',
        action='store_true',
        help='Skip Open3D occupancy renderings and write only BEV/input assets.')
    parser.add_argument(
        '--only-modality',
        action='store_true',
        help=
        'Use an existing manifest and generate only modality comparison panels.'
    )
    parser.add_argument(
        '--only-lidar-inputs',
        action='store_true',
        help='Use an existing manifest and regenerate only LiDAR input panels.')
    return parser.parse_args()


def build_data_loader(config_path):
    cfg = load_config(config_path)
    cfg = compat_cfg(cfg)
    split_cfg = get_split_cfg(cfg, 'test')
    if isinstance(split_cfg, dict) and cfg.data.get('test_dataloader', {}).get(
            'samples_per_gpu', 1) > 1:
        split_cfg.pipeline = replace_ImageToTensor(split_cfg.pipeline)
    dataset = build_dataset(split_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        runner_type='EpochBasedRunnerEval')
    return cfg, dataset, data_loader


def build_runner(config_path, checkpoint_path, gpu_id, export_analysis=False):
    cfg, dataset, data_loader = build_data_loader(config_path)
    cfg.model.pretrained = None
    cfg.gpu_ids = [gpu_id]
    if export_analysis:
        meta_info = dict(cfg.model.get('meta_info', {}))
        meta_info['export_analysis'] = True
        cfg.model.meta_info = meta_info

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = MMDataParallel(model.cuda(gpu_id), device_ids=[gpu_id])
    model.eval()
    return cfg, dataset, data_loader, model


def first_batch_by_index(data_loader, indices, max_samples):
    if indices is not None and hasattr(data_loader, 'dataset'):
        for index in indices:
            data = data_loader.dataset[int(index)]
            data = collate([data], samples_per_gpu=1)
            metas = data['img_metas'][0].data[0]
            yield int(metas[0]['index']), data
        return

    wanted = set(indices) if indices is not None else None
    yielded = 0
    for data in data_loader:
        metas = data['img_metas'][0].data[0]
        index = int(metas[0]['index'])
        if wanted is not None and index not in wanted:
            continue
        yield index, data
        yielded += 1
        if max_samples is not None and yielded >= max_samples:
            break


def run_predictions(model,
                    data_loader,
                    max_samples=None,
                    indices=None,
                    export_analysis=False):
    results = {}
    with torch.no_grad():
        for index, data in first_batch_by_index(data_loader, indices,
                                                max_samples):
            kwargs = dict(return_loss=False, rescale=True, **data)
            if export_analysis:
                kwargs['export_analysis'] = True
            outputs = model(**kwargs)
            flat = flatten_results(outputs)
            if not flat:
                continue
            result = flat[0]
            occ = np.asarray(result['occ_results'], dtype=np.uint8)
            entry = {
                'occ': occ,
                'data': data,
            }
            if result.get('analysis_payload') is not None:
                payload = result['analysis_payload']
                entry['analysis'] = payload
            results[index] = entry
    return results


def unwrap_datacontainer(value):
    if hasattr(value, 'data'):
        value = value.data
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return unwrap_datacontainer(value[0])
    return value


def infer_camera_num_frame(img_inputs, cam_names=None):
    tensor = img_inputs[0] if isinstance(img_inputs,
                                         (list, tuple)) and img_inputs else img_inputs
    if tensor is None or not hasattr(tensor, 'dim'):
        return 1
    if tensor.dim() == 4:
        total_slots = int(tensor.shape[0])
    elif tensor.dim() == 5:
        total_slots = int(tensor.shape[1])
    else:
        return 1
    num_cams = len(cam_names) if cam_names else 4
    if num_cams > 0 and total_slots > num_cams and total_slots % num_cams == 0:
        return total_slots // num_cams
    return 1


def render_current_camera_grid(img_inputs, sample_meta):
    cam_names = sample_meta.get('cam_names')
    return render_camera_grid(
        img_inputs,
        0,
        camera_num_frame=infer_camera_num_frame(img_inputs, cam_names),
        cam_names=cam_names)


def project_dynamic_mask(semantics):
    return np.isin(semantics, DYNAMIC_INDICES).any(axis=2)


def project_static_mask(semantics):
    return np.isin(semantics, STATIC_INDICES).any(axis=2)


def dynamic_iou_bev(pred, gt):
    pred_dyn = project_dynamic_mask(pred)
    gt_dyn = project_dynamic_mask(gt)
    inter = np.logical_and(pred_dyn, gt_dyn).sum()
    union = np.logical_or(pred_dyn, gt_dyn).sum()
    return float(inter / max(union, 1)), int(gt_dyn.sum())


def rank_candidates(plain_results, prosd_results, dataset, max_candidates):
    candidates = []
    common = sorted(set(plain_results) & set(prosd_results))
    for index in common[:max_candidates]:
        gt = load_occ_gt(dataset, index)
        plain_iou, dyn_pixels = dynamic_iou_bev(plain_results[index]['occ'],
                                                gt)
        prosd_iou, _ = dynamic_iou_bev(prosd_results[index]['occ'], gt)
        if dyn_pixels <= 0:
            continue
        score = (prosd_iou - plain_iou) * 1000.0 + min(dyn_pixels,
                                                       12000) / 12000.0
        candidates.append(
            dict(
                index=int(index),
                token=str(dataset.data_infos[index]['token']),
                scene=str(dataset.data_infos[index].get('scene_token', '')),
                plain_dyn_iou=plain_iou,
                prosd_dyn_iou=prosd_iou,
                dyn_pixels=dyn_pixels,
                score=score))
    candidates.sort(key=lambda item: item['score'], reverse=True)
    return candidates


def build_selected_from_indices(indices, plain_results, prosd_results, dataset):
    selected = []
    for index in indices:
        index = int(index)
        if index not in plain_results:
            raise KeyError(f'Missing plain prediction for index {index}.')
        if index not in prosd_results:
            raise KeyError(f'Missing ProSD prediction for index {index}.')
        gt = load_occ_gt(dataset, index)
        plain_iou, dyn_pixels = dynamic_iou_bev(plain_results[index]['occ'],
                                                gt)
        prosd_iou, _ = dynamic_iou_bev(prosd_results[index]['occ'], gt)
        selected.append(
            dict(
                index=index,
                token=str(dataset.data_infos[index]['token']),
                scene=str(dataset.data_infos[index].get('scene_token', '')),
                plain_dyn_iou=plain_iou,
                prosd_dyn_iou=prosd_iou,
                dyn_pixels=dyn_pixels,
                score=(prosd_iou - plain_iou) * 1000.0 +
                min(dyn_pixels, 12000) / 12000.0))
    return selected


def add_label(image, label, bg=(0, 0, 0)):
    image = np.asarray(image).copy()
    pad = 38
    canvas = np.full((image.shape[0] + pad, image.shape[1], 3),
                     255,
                     dtype=np.uint8)
    canvas[pad:] = image
    cv2.rectangle(canvas, (0, 0), (image.shape[1], pad), bg, -1)
    cv2.putText(canvas, label, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def resize_to(image, width=420):
    h, w = image.shape[:2]
    scale = width / float(w)
    return cv2.resize(
        image, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def stack_grid(images, labels, cols=3, width=420):
    panels = [
        add_label(resize_to(img, width), lab)
        for img, lab in zip(images, labels)
    ]
    max_h = max(p.shape[0] for p in panels)
    max_w = max(p.shape[1] for p in panels)
    padded = []
    for panel in panels:
        canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
        canvas[:panel.shape[0], :panel.shape[1]] = panel
        padded.append(canvas)
    rows = []
    for start in range(0, len(padded), cols):
        row = padded[start:start + cols]
        while len(row) < cols:
            row.append(np.full_like(padded[0], 255))
        rows.append(cv2.hconcat(row))
    return cv2.vconcat(rows)


def hconcat_resized(images, width=720):
    images = [np.asarray(image) for image in images if image is not None]
    if not images:
        return np.full((600, 600, 3), 235, dtype=np.uint8)
    panel_width = max(int(width // len(images)), 1)
    panels = [resize_to(image, panel_width) for image in images]
    max_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        canvas = np.full((max_h, panel_width, 3), 255, dtype=np.uint8)
        canvas[:panel.shape[0], :panel.shape[1]] = panel
        padded.append(canvas)
    return cv2.hconcat(padded)


def placeholder_panel(message, image_size=(600, 600)):
    width, height = image_size
    image = np.full((height, width, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (width - 1, height - 1), (180, 180, 180), 1)
    cv2.putText(image, message, (30, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (60, 60, 60), 2, cv2.LINE_AA)
    return image


def points_to_numpy(points):
    points = unwrap_datacontainer(points)
    if points is None:
        return None
    if torch.is_tensor(points):
        return points.detach().cpu().numpy()
    if hasattr(points, 'tensor') and torch.is_tensor(points.tensor):
        return points.tensor.detach().cpu().numpy()
    return np.asarray(points)


def render_gt_colored_points_bev(points,
                                 gt_semantics,
                                 point_cloud_range,
                                 color_map,
                                 image_size=(600, 600),
                                 raster_size=(900, 900),
                                 point_radius=2):
    points = points_to_numpy(points)
    if points is None or points.size == 0 or gt_semantics is None:
        return None
    points = np.asarray(points, dtype=np.float32)
    gt_semantics = np.asarray(gt_semantics)
    if points.ndim != 2 or points.shape[1] < 3 or gt_semantics.ndim != 3:
        return None

    range_values = np.asarray(point_cloud_range, dtype=np.float32)
    range_min = range_values[:3]
    range_max = range_values[3:]
    grid_shape = np.asarray(gt_semantics.shape, dtype=np.int32)
    voxel_size = (range_max - range_min) / np.maximum(grid_shape, 1)

    xyz = points[:, :3]
    finite_mask = np.isfinite(xyz).all(axis=1)
    range_mask = ((xyz >= range_min) & (xyz < range_max)).all(axis=1)
    valid_mask = finite_mask & range_mask
    if not np.any(valid_mask):
        return np.full((image_size[1], image_size[0], 3), 255, dtype=np.uint8)

    valid_xyz = xyz[valid_mask]
    voxel_indices = np.floor((valid_xyz - range_min) /
                             np.maximum(voxel_size, 1e-6)).astype(np.int32)
    voxel_indices = np.clip(voxel_indices, 0, grid_shape - 1)
    labels = gt_semantics[voxel_indices[:, 0], voxel_indices[:, 1],
                          voxel_indices[:, 2]]
    labels = np.clip(labels.astype(np.int64), 0, len(color_map) - 1)
    non_free_mask = labels != EMPTY_IDX
    if not np.any(non_free_mask):
        return np.full((image_size[1], image_size[0], 3), 255, dtype=np.uint8)
    valid_xyz = valid_xyz[non_free_mask]
    labels = labels[non_free_mask]
    draw_order = np.argsort(valid_xyz[:, 2], kind='stable')
    valid_xyz = valid_xyz[draw_order]
    labels = labels[draw_order]
    colors = color_map[labels]

    canvas = np.full((raster_size[1], raster_size[0], 3), 255, dtype=np.uint8)
    x_min, y_min = range_min[:2]
    x_max, y_max = range_max[:2]
    px = ((valid_xyz[:, 0] - x_min) / max(x_max - x_min, 1e-6) *
          raster_size[0]).astype(np.int32)
    py = ((valid_xyz[:, 1] - y_min) / max(y_max - y_min, 1e-6) *
          raster_size[1]).astype(np.int32)
    px = raster_size[0] - 1 - np.clip(px, 0, raster_size[0] - 1)
    py = raster_size[1] - 1 - np.clip(py, 0, raster_size[1] - 1)

    fill_radius = max(int(point_radius), 1)
    for x_pos, y_pos, color in zip(px, py, colors):
        center = (int(x_pos), int(y_pos))
        cv2.circle(canvas, center, fill_radius,
                   tuple(int(channel) for channel in color), -1)
    if raster_size != image_size:
        canvas = cv2.resize(canvas, image_size, interpolation=cv2.INTER_AREA)
    return canvas


def render_lidar_input_bev(data,
                           point_cloud_range,
                           gt_semantics=None,
                           color_map=None):
    points = data.get('points') if isinstance(data, dict) else None
    points = unwrap_datacontainer(points)
    if isinstance(points, (list, tuple)):
        points = unwrap_datacontainer(points[0]) if points else None
    if gt_semantics is not None and color_map is not None:
        lidar_bev = render_gt_colored_points_bev(
            points,
            gt_semantics,
            point_cloud_range,
            color_map,
            image_size=(600, 600),
            raster_size=(600, 600),
            point_radius=2)
        if lidar_bev is not None:
            return lidar_bev
    lidar_bev = render_points_bev(
        points,
        point_cloud_range,
        image_size=(600, 600),
        raster_size=(900, 900))
    if lidar_bev is None:
        return placeholder_panel('LiDAR unavailable')
    return lidar_bev


def render_dynamic_error(pred, gt):
    pred_dyn = project_dynamic_mask(pred)
    gt_dyn = project_dynamic_mask(gt)
    canvas = np.full((*pred_dyn.shape, 3), 245, dtype=np.uint8)
    tp = pred_dyn & gt_dyn
    fp = pred_dyn & ~gt_dyn
    fn = ~pred_dyn & gt_dyn
    canvas[tp] = (70, 170, 70)
    canvas[fp] = (230, 90, 60)
    canvas[fn] = (70, 70, 230)
    canvas = canvas.transpose(1, 0, 2)[::-1, ::-1]
    canvas = cv2.resize(canvas, (600, 600), interpolation=cv2.INTER_NEAREST)
    return add_legend(canvas, DYNAMIC_ERROR_LEGEND)


def render_dynamic_recovery(plain, prosd, gt):
    plain_dyn = project_dynamic_mask(plain)
    prosd_dyn = project_dynamic_mask(prosd)
    gt_dyn = project_dynamic_mask(gt)
    recovered = (~plain_dyn) & prosd_dyn & gt_dyn
    already_correct = plain_dyn & prosd_dyn & gt_dyn
    still_missed = (~prosd_dyn) & gt_dyn
    canvas = np.full((*gt_dyn.shape, 3), 245, dtype=np.uint8)
    canvas[already_correct] = (120, 120, 120)
    canvas[recovered] = (50, 170, 70)
    canvas[still_missed] = (70, 70, 230)
    canvas = canvas.transpose(1, 0, 2)[::-1, ::-1]
    canvas = cv2.resize(canvas, (600, 600), interpolation=cv2.INTER_NEAREST)
    return add_legend(canvas, DYNAMIC_CHANGE_LEGEND)


def add_legend(image, entries, strip_height=None):
    image = np.asarray(image).copy()
    if strip_height is None:
        strip_height = 48 if len(entries) <= 3 else 78
    strip = np.full((strip_height, image.shape[1], 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = 18, 28
    col_count = min(max(len(entries), 1), 3)
    col_w = max((image.shape[1] - 2 * x) // col_count, 1)
    for entry_index, (label, color) in enumerate(entries):
        row = entry_index // col_count
        col = entry_index % col_count
        px = x + col * col_w
        py = y + row * 32
        cv2.rectangle(strip, (px, py - 15), (px + 22, py + 7), color, -1)
        cv2.rectangle(strip, (px, py - 15), (px + 22, py + 7), (80, 80, 80), 1)
        cv2.putText(strip, label, (px + 32, py + 3), font, 0.56, (35, 35, 35),
                    1, cv2.LINE_AA)
    return cv2.vconcat([image, strip])


def project_scalar_volume_bev(volume, projection='max'):
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        return None
    if projection == 'mean':
        scalar_bev = np.mean(volume, axis=2)
    else:
        scalar_bev = np.max(volume, axis=2)
    return np.nan_to_num(scalar_bev, nan=0.0, posinf=0.0, neginf=0.0)


def render_scalar_bev_robust(scalar_bev,
                             image_size=(600, 600),
                             value_min=None,
                             value_max=None,
                             percentile=(1.0, 99.0),
                             colormap=cv2.COLORMAP_VIRIDIS,
                             gamma=1.0,
                             positive_only_percentile=False):
    scalar_bev = np.asarray(scalar_bev, dtype=np.float32)
    if scalar_bev.ndim != 2:
        return np.full((*image_size, 3), 235, dtype=np.uint8)
    scalar_bev = np.nan_to_num(scalar_bev, nan=0.0, posinf=0.0, neginf=0.0)
    finite_values = scalar_bev[np.isfinite(scalar_bev)]
    if finite_values.size == 0:
        finite_values = np.array([0.0], dtype=np.float32)
    normalization_values = finite_values
    if positive_only_percentile:
        positive_values = finite_values[finite_values > 1e-6]
        if positive_values.size >= 32:
            normalization_values = positive_values
    if value_min is None or value_max is None:
        lo, hi = np.percentile(normalization_values, percentile)
        if value_min is not None:
            lo = value_min
        if value_max is not None:
            hi = value_max
    else:
        lo, hi = float(value_min), float(value_max)
    if float(hi - lo) < 1e-6:
        lo, hi = float(finite_values.min()), float(finite_values.max())
    if float(hi - lo) < 1e-6:
        lo, hi = 0.0, 1.0
    scalar_bev = np.clip((scalar_bev - lo) / (hi - lo), 0.0, 1.0)
    if abs(float(gamma) - 1.0) > 1e-6:
        scalar_bev = np.power(scalar_bev, float(gamma))
    scalar_bev = np.round(scalar_bev * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(scalar_bev, colormap)
    heatmap = heatmap.transpose(1, 0, 2)[::-1, ::-1]
    return cv2.resize(heatmap, image_size, interpolation=cv2.INTER_NEAREST)


def render_scalar_volume_bev_robust(volume,
                                    image_size=(600, 600),
                                    value_min=None,
                                    value_max=None,
                                    percentile=(1.0, 99.0),
                                    projection='max',
                                    colormap=cv2.COLORMAP_VIRIDIS,
                                    gamma=1.0,
                                    positive_only_percentile=False):
    scalar_bev = project_scalar_volume_bev(volume, projection=projection)
    if scalar_bev is None:
        return np.full((*image_size, 3), 235, dtype=np.uint8)
    return render_scalar_bev_robust(
        scalar_bev,
        image_size=image_size,
        value_min=value_min,
        value_max=value_max,
        percentile=percentile,
        colormap=colormap,
        gamma=gamma,
        positive_only_percentile=positive_only_percentile)


def render_volume_scalar_from_payload(payload, key, image_size=(600, 600)):
    if payload is None or key not in payload:
        return np.full((*image_size, 3), 235, dtype=np.uint8)
    volume = np.asarray(payload[key], dtype=np.float32)
    return render_scalar_volume_bev_robust(
        volume, image_size=image_size, value_min=0.0, value_max=1.0)


def render_dynamic_input_energy(payload, image_size=(900, 900)):
    if payload is None or 'dynamic_input_energy' not in payload:
        return np.full((*image_size, 3), 235, dtype=np.uint8)
    volume = np.asarray(payload['dynamic_input_energy'], dtype=np.float32)
    energy = project_scalar_volume_bev(volume, projection='mean')
    if energy is None:
        return np.full((*image_size, 3), 235, dtype=np.uint8)
    values = energy[np.isfinite(energy)]
    if values.size > 0:
        lo, hi = np.percentile(values, (1.0, 99.6))
    else:
        lo, hi = 0.0, 1.0
    if float(hi - lo) < 1e-6:
        lo, hi = 0.0, 1.0
    energy_norm = np.clip((energy - lo) / (hi - lo), 0.0, 1.0)

    context = 0.34 * np.power(energy_norm, 1.45)
    dynamic_weight = np.ones_like(energy_norm, dtype=np.float32)
    dynamic_conf = payload.get('dynamic_conf')
    if dynamic_conf is not None:
        dynamic_conf = np.asarray(dynamic_conf, dtype=np.float32)
        if dynamic_conf.shape == volume.shape:
            dynamic_conf = project_scalar_volume_bev(
                dynamic_conf, projection='max')
            dynamic_weight *= np.power(
                np.clip(dynamic_conf, 0.0, 1.0), 0.40)
    suppression_gate = payload.get('suppression_gate')
    if suppression_gate is not None:
        suppression_gate = np.asarray(suppression_gate, dtype=np.float32)
        if suppression_gate.shape == volume.shape:
            suppression_strength = 1.0 - project_scalar_volume_bev(
                suppression_gate, projection='mean')
            dynamic_weight *= np.power(
                np.clip(1.0 - suppression_strength, 0.0, 1.0), 0.18)
    scalar_bev = context + 0.66 * energy_norm * dynamic_weight
    return render_scalar_bev_robust(
        scalar_bev,
        image_size=image_size,
        value_min=None,
        value_max=None,
        percentile=(1.0, 99.6),
        colormap=cv2.COLORMAP_MAGMA,
        gamma=1.05)


def render_suppression_strength(payload, image_size=(900, 900)):
    if payload is None or 'suppression_gate' not in payload:
        return np.full((*image_size, 3), 235, dtype=np.uint8)
    gate = np.asarray(payload['suppression_gate'], dtype=np.float32)
    strength = np.clip(1.0 - gate, 0.0, 1.0)
    alpha = float(payload.get('suppression_alpha', 0.0) or 0.0)
    value_max = max(alpha, float(np.percentile(strength, 99.0)), 1e-3)
    return render_scalar_volume_bev_robust(
        strength,
        image_size=image_size,
        value_min=0.0,
        value_max=value_max,
        projection='mean',
        colormap=cv2.COLORMAP_INFERNO)


def render_occ_3d_asset(semantics, point_cloud_range, out_dir):
    view_json_path = str(Path(out_dir) / 'occ_view.json')
    return _render_occ_3d_image(
        semantics,
        point_cloud_range,
        COLOR_MAP,
        EMPTY_IDX,
        info_lines=[],
        view_json_path=view_json_path)


def get_point_cloud_range(config_path):
    cfg = load_config(config_path)
    return list(
        cfg.get('point_cloud_range', [-64.0, -64.0, -4.8, 64.0, 64.0, 1.6]))


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def build_figures(args, dataset, selected, plain, c_res, l_res, m_res):
    out_dir = Path(args.output_dir)
    assets_dir = out_dir / 'assets'
    panels_dir = out_dir / 'panels'
    assets_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)
    point_cloud_range = get_point_cloud_range(args.config_c)

    manifest = {
        'selected': selected,
        'figures': [],
        'purpose': {
            'qualitative_main':
            'GT/plain/ProSD comparison for dynamic occupancy recovery.',
            'mechanism':
            'Static confidence, suppression strength, and error maps.',
            'occupancy_3d':
            '3D GT/plain/ProSD occupancy renderings.',
            'modality':
            'Camera/LiDAR/multimodal complementarity under the same ProSD head.',
        }
    }

    for rank, item in enumerate(selected):
        index = item['index']
        gt = load_occ_gt(dataset, index)
        sample_meta = dataset.get_data_info(index)
        token = item['token']
        stem = f'sample{rank:02d}_idx{index:04d}_{token[:8]}'

        img_inputs = unwrap_datacontainer(c_res[index]['data']['img_inputs'])
        cam_img = render_current_camera_grid(img_inputs, sample_meta)
        if cam_img is None:
            cam_img = np.full((600, 600, 3), 235, dtype=np.uint8)
        l_input = render_lidar_input_bev(
            l_res[index]['data'],
            point_cloud_range,
            gt_semantics=gt,
            color_map=COLOR_MAP) if index in l_res else None
        if index in m_res and 'img_inputs' in m_res[index]['data']:
            m_img_inputs = unwrap_datacontainer(
                m_res[index]['data']['img_inputs'])
            m_cam_img = render_current_camera_grid(m_img_inputs, sample_meta)
        else:
            m_cam_img = None
        if m_cam_img is None:
            m_cam_img = cam_img
        m_lidar_input = render_lidar_input_bev(
            m_res[index]['data'],
            point_cloud_range,
            gt_semantics=gt,
            color_map=COLOR_MAP) if index in m_res else None
        m_input = hconcat_resized(
            [m_cam_img, m_lidar_input],
            width=720) if m_lidar_input is not None else None
        gt_bev = render_occ_bev(gt, EMPTY_IDX, COLOR_MAP)
        plain_bev = render_occ_bev(plain[index]['occ'], EMPTY_IDX, COLOR_MAP)
        c_bev = render_occ_bev(c_res[index]['occ'], EMPTY_IDX, COLOR_MAP)
        l_bev = render_occ_bev(l_res[index]['occ'], EMPTY_IDX,
                               COLOR_MAP) if index in l_res else None
        m_bev = render_occ_bev(m_res[index]['occ'], EMPTY_IDX,
                               COLOR_MAP) if index in m_res else None
        plain_err = render_dynamic_error(plain[index]['occ'], gt)
        c_err = render_dynamic_error(c_res[index]['occ'], gt)
        recovery = render_dynamic_recovery(plain[index]['occ'],
                                           c_res[index]['occ'], gt)
        asset_map = {
            'camera_grid': cam_img,
            'gt_bev': gt_bev,
            'plain_bev': plain_bev,
            'prosd_c_bev': c_bev,
            'plain_dynamic_error': plain_err,
            'prosd_dynamic_error': c_err,
            'dynamic_recovery': recovery,
        }
        if not args.skip_3d:
            gt_3d = render_occ_3d_asset(gt, point_cloud_range, out_dir)
            plain_3d = render_occ_3d_asset(plain[index]['occ'],
                                           point_cloud_range, out_dir)
            c_3d = render_occ_3d_asset(c_res[index]['occ'], point_cloud_range,
                                       out_dir)
            asset_map.update({
                'gt_3d': gt_3d,
                'plain_3d': plain_3d,
                'prosd_c_3d': c_3d,
            })
        if l_bev is not None:
            asset_map['prosd_l_bev'] = l_bev
        if l_input is not None:
            asset_map['lidar_input_bev'] = l_input
        if m_bev is not None:
            asset_map['prosd_m_bev'] = m_bev
        if m_input is not None:
            asset_map['multimodal_input'] = m_input

        payload = c_res[index].get('analysis')
        mechanism_map = {
            'static_conf':
            render_volume_scalar_from_payload(payload, 'static_conf'),
            'dynamic_conf':
            render_volume_scalar_from_payload(payload, 'dynamic_conf'),
            'suppression_gate':
            render_suppression_strength(payload),
            'dynamic_input_energy':
            render_dynamic_input_energy(payload),
        }
        asset_map.update(mechanism_map)

        for name, image in asset_map.items():
            save_image(assets_dir / f'{stem}_{name}.png', image)

        qualitative = stack_grid(
            [cam_img, gt_bev, plain_bev, c_bev, plain_err, recovery], [
                'Roadside Cameras', 'GT Occupancy BEV', 'Plain Camera BEV',
                'ProSD-Occ BEV', 'Plain Dynamic Error', 'Dynamic Change Map'
            ],
            cols=3,
            width=430)
        qualitative_path = panels_dir / f'{stem}_qualitative_main.png'
        save_image(qualitative_path, qualitative)

        occupancy_3d_path = None
        if not args.skip_3d:
            occupancy_3d = stack_grid(
                [gt_3d, plain_3d, c_3d],
                ['GT Occupancy 3D', 'Plain Camera 3D', 'ProSD-Occ 3D'],
                cols=3,
                width=520)
            occupancy_3d_path = panels_dir / f'{stem}_occupancy_3d.png'
            save_image(occupancy_3d_path, occupancy_3d)

        mechanism = stack_grid([
            mechanism_map['static_conf'],
            mechanism_map['suppression_gate'],
            mechanism_map['dynamic_input_energy'],
        ], [
            'Static Confidence', 'Suppression Strength',
            'Dynamic-aware Feature'
        ],
                               cols=3,
                               width=360)
        mechanism_path = panels_dir / f'{stem}_mechanism.png'
        save_image(mechanism_path, mechanism)

        if l_bev is not None and m_bev is not None:
            modality = stack_grid([
                cam_img,
                cam_img,
                l_input,
                m_input,
                plain_bev,
                c_bev,
                l_bev,
                m_bev,
            ], [
                'Plain (C) Input',
                'ProSD-Occ (C) Input',
                'ProSD-Occ (L) Input',
                'ProSD-Occ (C+L) Input',
                'Plain (C) Occ',
                'ProSD-Occ (C) Occ',
                'ProSD-Occ (L) Occ',
                'ProSD-Occ (C+L) Occ',
            ],
                                  cols=4,
                                  width=330)
            modality_path = panels_dir / f'{stem}_modality_comparison.png'
            save_image(modality_path, modality)
        else:
            modality_path = None

        manifest['figures'].append(
            dict(
                index=index,
                token=token,
                qualitative_main=str(qualitative_path),
                occupancy_3d=str(occupancy_3d_path)
                if occupancy_3d_path else None,
                mechanism=str(mechanism_path),
                modality_comparison=str(modality_path)
                if modality_path else None,
                assets={
                    name: str(assets_dir / f'{stem}_{name}.png')
                    for name in asset_map
                },
                metrics={
                    k: item[k]
                    for k in ('plain_dyn_iou', 'prosd_dyn_iou', 'dyn_pixels',
                              'score')
                },
            ))

    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)


def build_modality_only(args, dataset, selected, c_res, l_res, m_res):
    out_dir = Path(args.output_dir)
    assets_dir = out_dir / 'assets'
    panels_dir = out_dir / 'panels'
    manifest_path = out_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    point_cloud_range = get_point_cloud_range(args.config_c)
    for rank, item in enumerate(selected):
        index = item['index']
        gt = load_occ_gt(dataset, index)
        sample_meta = dataset.get_data_info(index)
        token = item['token']
        stem = f'sample{rank:02d}_idx{index:04d}_{token[:8]}'
        img_inputs = unwrap_datacontainer(c_res[index]['data']['img_inputs'])
        cam_img = render_current_camera_grid(img_inputs, sample_meta)
        if cam_img is None:
            cam_img = np.full((600, 600, 3), 235, dtype=np.uint8)
        l_input = render_lidar_input_bev(
            l_res[index]['data'],
            point_cloud_range,
            gt_semantics=gt,
            color_map=COLOR_MAP)
        if 'img_inputs' in m_res[index]['data']:
            m_img_inputs = unwrap_datacontainer(
                m_res[index]['data']['img_inputs'])
            m_cam_img = render_current_camera_grid(m_img_inputs, sample_meta)
        else:
            m_cam_img = None
        if m_cam_img is None:
            m_cam_img = cam_img
        m_lidar_input = render_lidar_input_bev(
            m_res[index]['data'],
            point_cloud_range,
            gt_semantics=gt,
            color_map=COLOR_MAP)
        m_input = hconcat_resized([m_cam_img, m_lidar_input], width=720)
        gt_bev = render_occ_bev(gt, EMPTY_IDX, COLOR_MAP)
        plain_bev_path = assets_dir / f'{stem}_plain_bev.png'
        plain_bev = (
            cv2.imread(str(plain_bev_path)) if plain_bev_path.exists() else
            placeholder_panel('Plain unavailable'))
        c_bev = render_occ_bev(c_res[index]['occ'], EMPTY_IDX, COLOR_MAP)
        l_bev = render_occ_bev(l_res[index]['occ'], EMPTY_IDX, COLOR_MAP)
        m_bev = render_occ_bev(m_res[index]['occ'], EMPTY_IDX, COLOR_MAP)
        asset_map = {
            'lidar_input_bev': l_input,
            'multimodal_input': m_input,
            'prosd_l_bev': l_bev,
            'prosd_m_bev': m_bev,
        }
        for name, image in asset_map.items():
            save_image(assets_dir / f'{stem}_{name}.png', image)
        modality = stack_grid([
            cam_img,
            cam_img,
            l_input,
            m_input,
            plain_bev,
            c_bev,
            l_bev,
            m_bev,
        ], [
            'Plain (C) Input',
            'ProSD-Occ (C) Input',
            'ProSD-Occ (L) Input',
            'ProSD-Occ (C+L) Input',
            'Plain (C) Occ',
            'ProSD-Occ (C) Occ',
            'ProSD-Occ (L) Occ',
            'ProSD-Occ (C+L) Occ',
        ],
                              cols=4,
                              width=330)
        modality_path = panels_dir / f'{stem}_modality_comparison.png'
        save_image(modality_path, modality)
        for fig in manifest.get('figures', []):
            if int(fig.get('index', -1)) != int(index):
                continue
            fig['modality_comparison'] = str(modality_path)
            fig.setdefault('assets', {}).update({
                name:
                str(assets_dir / f'{stem}_{name}.png')
                for name in asset_map
            })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def build_lidar_inputs_only(args, dataset, data_loader, selected):
    out_dir = Path(args.output_dir)
    assets_dir = out_dir / 'assets'
    manifest_path = out_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    point_cloud_range = get_point_cloud_range(args.config_c)
    selected_indices = [int(item['index']) for item in selected]
    data_by_index = {
        index: data
        for index, data in first_batch_by_index(data_loader, selected_indices,
                                                None)
    }

    for rank, item in enumerate(selected):
        index = int(item['index'])
        if index not in data_by_index:
            raise KeyError(f'Missing LiDAR input data for index {index}.')
        gt = load_occ_gt(dataset, index)
        token = item['token']
        stem = f'sample{rank:02d}_idx{index:04d}_{token[:8]}'
        l_input = render_lidar_input_bev(
            data_by_index[index],
            point_cloud_range,
            gt_semantics=gt,
            color_map=COLOR_MAP)
        asset_path = assets_dir / f'{stem}_lidar_input_bev.png'
        save_image(asset_path, l_input)

        for fig in manifest.get('figures', []):
            if int(fig.get('index', -1)) != index:
                continue
            fig.setdefault('assets', {})['lidar_input_bev'] = str(asset_path)
            break
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def main():
    args = parse_args()
    setup_multi_processes(load_config(args.config_c))
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=False)
    torch.backends.cudnn.benchmark = True

    if args.only_lidar_inputs:
        manifest_path = Path(args.output_dir) / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        selected = manifest['selected']
        print('Loading LiDAR dataset for input panels...')
        _, l_dataset, l_loader = build_data_loader(args.config_l)
        build_lidar_inputs_only(args, l_dataset, l_loader, selected)
        print(f'Saved LiDAR input panels to {args.output_dir}')
        return

    if args.only_modality:
        manifest_path = Path(args.output_dir) / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        selected = manifest['selected']
        selected_indices = [int(item['index']) for item in selected]
        print('Loading ProSD camera model for modality panels...')
        _, c_dataset, c_loader, c_model = build_runner(
            args.config_c, args.ckpt_c, args.gpu_id, export_analysis=False)
        c_results = run_predictions(
            c_model, c_loader, indices=selected_indices, export_analysis=False)
        del c_model
        torch.cuda.empty_cache()
        print('Loading LiDAR ProSD model for modality panels...')
        _, _, l_loader, l_model = build_runner(
            args.config_l, args.ckpt_l, args.gpu_id, export_analysis=False)
        l_results = run_predictions(
            l_model, l_loader, indices=selected_indices, export_analysis=False)
        del l_model
        torch.cuda.empty_cache()
        print('Loading multimodal ProSD model for modality panels...')
        _, _, m_loader, m_model = build_runner(
            args.config_m, args.ckpt_m, args.gpu_id, export_analysis=False)
        m_results = run_predictions(
            m_model, m_loader, indices=selected_indices, export_analysis=False)
        del m_model
        torch.cuda.empty_cache()
        build_modality_only(args, c_dataset, selected, c_results, l_results,
                            m_results)
        print(f'Saved modality figures to {args.output_dir}')
        return

    print('Loading plain camera model...')
    _, plain_dataset, plain_loader, plain_model = build_runner(
        args.config_plain, args.ckpt_plain, args.gpu_id, export_analysis=False)
    if args.selected_indices:
        selected_indices = [int(index) for index in args.selected_indices]
        print('Running plain camera selected indices:', selected_indices)
        plain_results = run_predictions(
            plain_model,
            plain_loader,
            max_samples=len(selected_indices),
            indices=selected_indices,
            export_analysis=False)
    else:
        print('Running plain camera candidates...')
        plain_results = run_predictions(
            plain_model,
            plain_loader,
            max_samples=args.max_samples,
            export_analysis=False)
    del plain_model
    torch.cuda.empty_cache()

    print('Loading ProSD camera model...')
    _, c_dataset, c_loader, c_model = build_runner(
        args.config_c, args.ckpt_c, args.gpu_id, export_analysis=True)
    if args.selected_indices:
        print('Running ProSD camera selected indices:', selected_indices)
        c_results = run_predictions(
            c_model,
            c_loader,
            max_samples=len(selected_indices),
            indices=selected_indices,
            export_analysis=True)
        selected = build_selected_from_indices(selected_indices, plain_results,
                                               c_results, c_dataset)
    else:
        print('Running ProSD camera candidates...')
        c_results = run_predictions(
            c_model,
            c_loader,
            max_samples=args.max_samples,
            export_analysis=True)
        candidates = rank_candidates(plain_results, c_results, c_dataset,
                                     args.max_samples)
        selected = candidates[:args.num_select]
    if not selected:
        raise RuntimeError('No valid dynamic-occupancy candidates found.')
    selected_indices = [item['index'] for item in selected]
    print('Selected indices:', selected_indices)

    # Re-run selected camera samples with analysis to ensure all payloads exist.
    missing = [
        idx for idx in selected_indices
        if idx not in c_results or 'analysis' not in c_results[idx]
    ]
    if missing:
        c_results.update(
            run_predictions(
                c_model, c_loader, indices=missing, export_analysis=True))
    del c_model
    torch.cuda.empty_cache()

    l_results, m_results = {}, {}
    if not args.skip_modality:
        print('Loading LiDAR ProSD model...')
        _, _, l_loader, l_model = build_runner(
            args.config_l, args.ckpt_l, args.gpu_id, export_analysis=False)
        l_results = run_predictions(
            l_model, l_loader, indices=selected_indices, export_analysis=False)
        del l_model
        torch.cuda.empty_cache()

        if not args.skip_multimodal:
            print('Loading multimodal ProSD model...')
            _, _, m_loader, m_model = build_runner(
                args.config_m, args.ckpt_m, args.gpu_id, export_analysis=False)
            m_results = run_predictions(
                m_model,
                m_loader,
                indices=selected_indices,
                export_analysis=False)
            del m_model
            torch.cuda.empty_cache()

    build_figures(args, c_dataset, selected, plain_results, c_results,
                  l_results, m_results)
    print(f'Saved figures to {args.output_dir}')


if __name__ == '__main__':
    main()
