#!/usr/bin/env python3
"""Render uniform occupancy, flow, and camera-depth diagnostics for baselines."""

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmdet3d.core.visualizer.occ_visualization import (
    draw_flow_arrows, project_dynamic_flow_to_bev, render_flow_bev)


CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free'
]
CLASS_COLORS = np.array([
    (0, 0, 0), (50, 120, 255), (203, 192, 255), (0, 255, 255),
    (245, 150, 0), (255, 255, 0), (0, 127, 255), (0, 0, 255),
    (150, 240, 255), (0, 60, 135), (240, 32, 160), (255, 0, 255),
    (137, 137, 139), (75, 0, 75), (80, 240, 150), (250, 230, 230),
    (0, 175, 0), (255, 255, 255)
], dtype=np.uint8)
CAMERA_NAMES = ['CAM_FRONT', 'CAM_LEFT', 'CAM_BACK', 'CAM_RIGHT']
EMPTY_INDEX = 17
DYNAMIC_INDICES = np.array([2, 3, 4, 6, 7, 10], dtype=np.int64)
DYNAMIC_CLASS_INDICES = tuple(DYNAMIC_INDICES.tolist())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prediction', help='NPZ emitted by export_occflow_prediction.py.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--ann-file', default='data/v2xreal_nuscenes/v2xreal_infos_val.pkl')
    parser.add_argument('--stride', type=int, default=12)
    parser.add_argument('--max-speed', type=float, default=12.0)
    parser.add_argument('--point-cloud-range', type=float, nargs=6,
                        default=[-64.0, -64.0, -4.8, 64.0, 64.0, 1.6])
    return parser.parse_args()


def load_info(ann_file, sample_index):
    payload = mmcv.load(ann_file)
    infos = payload['infos'] if isinstance(payload, dict) else payload
    return infos[int(sample_index)]


def project_occ_to_bev(occupancy):
    occupied = occupancy != EMPTY_INDEX
    depth = np.arange(occupancy.shape[2], dtype=np.float32)[None, None, :]
    selected = np.argmax(depth * occupied, axis=2)
    return np.take_along_axis(occupancy, selected[..., None], axis=2)[..., 0]


def render_occ_bev(occupancy, size=600):
    labels = project_occ_to_bev(np.asarray(occupancy))
    valid = (labels >= 0) & (labels < len(CLASS_COLORS))
    image = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    image[valid] = CLASS_COLORS[labels[valid]]
    image = image.transpose(1, 0, 2)[::-1, ::-1]
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)


def titled_panel(image, title):
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 30), (255, 255, 255), -1)
    cv2.putText(panel, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (55, 55, 55), 1, cv2.LINE_AA)
    return panel


def render_occ_comparison(prediction, gt):
    return cv2.hconcat([
        titled_panel(render_occ_bev(prediction), 'Pred Occupancy BEV'),
        titled_panel(render_occ_bev(gt), 'GT Occupancy BEV')
    ])


def top_surface(occupancy):
    occupied = (occupancy != EMPTY_INDEX) & (occupancy != 255)
    indices = np.where(
        occupied, np.arange(occupancy.shape[2])[None, None, :], -1).max(axis=2)
    valid = indices >= 0
    indices = np.maximum(indices, 0)
    labels = np.take_along_axis(occupancy, indices[..., None], axis=2)[..., 0]
    return labels, indices, valid


def render_flow(occupancy, flow, output_path, stride, max_speed):
    labels, indices, valid = top_surface(occupancy)
    top_flow = np.take_along_axis(flow, indices[..., None, None], axis=2)[..., 0, :]
    canvas = render_occ_bev(occupancy)
    dynamic = valid & np.isin(labels, DYNAMIC_INDICES)
    height, width = canvas.shape[:2]
    figure, axis = plt.subplots(figsize=(6, 6), dpi=200)
    axis.imshow(canvas)
    sampled = np.zeros_like(dynamic)
    sampled[::stride, ::stride] = True
    rows, cols = np.where(dynamic & sampled)
    if len(rows):
        vectors = top_flow[rows, cols]
        speed = np.linalg.norm(vectors, axis=1)
        keep = (speed > 0.15) & (speed <= max_speed)
        rows, cols, vectors = rows[keep], cols[keep], vectors[keep] * 1.25
        axis.quiver(width - 1 - rows, height - 1 - cols,
                    -vectors[:, 0], -vectors[:, 1],
                    color='black', angles='xy', scale_units='xy', scale=1.0,
                    width=0.0025, headwidth=3.5, headlength=4.5)
    axis.set_axis_off()
    figure.tight_layout(pad=0)
    figure.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(figure)


def render_flow_comparison(prediction,
                           prediction_flow,
                           gt,
                           gt_flow,
                           point_cloud_range,
                           max_speed=12.0):
    """Render RoadOcc-style flow diagnostics in the shared V2X frame."""
    panel_size = 600
    flow_size = (panel_size, panel_size)
    pred_flow_panel = render_flow_bev(
        prediction_flow, prediction, DYNAMIC_CLASS_INDICES,
        image_size=flow_size, max_speed=max_speed)
    gt_flow_panel = render_flow_bev(
        gt_flow, gt, DYNAMIC_CLASS_INDICES,
        image_size=flow_size, max_speed=max_speed)
    if pred_flow_panel is None or gt_flow_panel is None:
        raise ValueError('Flow visualization requires aligned [X,Y,Z,2] flow volumes.')

    def semantic_with_arrows(occupancy, flow):
        flow_bev, valid_mask = project_dynamic_flow_to_bev(
            flow, occupancy, DYNAMIC_CLASS_INDICES,
            fill_iterations=2, neighbor_radius=1, min_neighbors=5)
        if flow_bev is None or valid_mask is None:
            raise ValueError('Unable to project flow into the shared BEV grid.')
        return draw_flow_arrows(
            render_occ_bev(occupancy, size=panel_size), flow_bev, valid_mask,
            point_cloud_range=point_cloud_range, arrow_stride=8,
            arrow_horizon=0.5, min_speed=0.3, max_arrow_pixels=36.0)

    return cv2.vconcat([
        cv2.hconcat([
            titled_panel(pred_flow_panel, 'Pred Flow'),
            titled_panel(gt_flow_panel, 'GT Flow'),
        ]),
        cv2.hconcat([
            titled_panel(semantic_with_arrows(prediction, prediction_flow),
                         'Pred Occupancy + Flow Arrows'),
            titled_panel(semantic_with_arrows(gt, gt_flow),
                         'GT Occupancy + Flow Arrows'),
        ]),
    ])


def load_occflow_gt(info):
    """Locate generated flow labels next to the matching occupancy labels."""
    occ_path = Path(info['occ_path'])
    flow_path = (occ_path.parents[2] / 'occflow_gts' /
                 occ_path.parent.name / occ_path.name / 'labels.npz')
    if not flow_path.exists():
        raise FileNotFoundError(f'Missing generated flow labels: {flow_path}')
    with np.load(flow_path) as payload:
        return payload['flow'].astype(np.float32)


def camera_depth(points, camera, image_shape):
    rotation = np.asarray(camera['sensor2lidar_rotation'], dtype=np.float32)
    translation = np.asarray(camera['sensor2lidar_translation'], dtype=np.float32)
    intrinsic = np.asarray(camera['cam_intrinsic'], dtype=np.float32)
    camera_points = (points - translation) @ rotation
    depth = camera_points[:, 2]
    projected = camera_points @ intrinsic.T
    valid = depth > 0.1
    projected[valid, :2] /= depth[valid, None]
    height, width = image_shape[:2]
    columns = np.rint(projected[:, 0]).astype(np.int32)
    rows = np.rint(projected[:, 1]).astype(np.int32)
    valid &= (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    result = np.full((height, width), np.inf, dtype=np.float32)
    flat = rows[valid] * width + columns[valid]
    np.minimum.at(result.reshape(-1), flat, depth[valid])
    result[~np.isfinite(result)] = np.nan
    return result


def camera_depth_from_augmented_geometry(points, camera_geometry, image_shape):
    """Project augmented-frame points to an augmented training camera image.

    ``PrepareImageInputs`` changes both the image plane and the BEV frame.
    Training diagnostics must use those matrices instead of the unaugmented
    metadata calibration, otherwise the depth panels compare different views.
    """
    rots = np.asarray(camera_geometry['rots'], dtype=np.float32)
    trans = np.asarray(camera_geometry['trans'], dtype=np.float32)
    intrins = np.asarray(camera_geometry['intrins'], dtype=np.float32)
    post_rots = np.asarray(camera_geometry['post_rots'], dtype=np.float32)
    post_trans = np.asarray(camera_geometry['post_trans'], dtype=np.float32)
    bda = np.asarray(camera_geometry['bda'], dtype=np.float32)

    camera_to_lidar = np.eye(4, dtype=np.float32)
    camera_to_lidar[:3, :3] = rots
    camera_to_lidar[:3, 3] = trans
    intrinsic = np.eye(4, dtype=np.float32)
    intrinsic[:3, :3] = intrins
    image_aug = np.eye(4, dtype=np.float32)
    image_aug[:3, :3] = post_rots
    image_aug[:3, 3] = post_trans
    bda_hom = np.eye(4, dtype=np.float32)
    bda_hom[:3, :3] = bda

    # The same column-vector transform used by each adapted view transformer.
    camera_from_augmented_lidar = np.linalg.inv(camera_to_lidar) @ np.linalg.inv(bda_hom)
    camera_points = np.concatenate(
        [np.asarray(points, dtype=np.float32),
         np.ones((len(points), 1), dtype=np.float32)], axis=1)
    camera_points = camera_points @ camera_from_augmented_lidar.T
    depth = camera_points[:, 2]
    projected = camera_points @ (image_aug @ intrinsic).T
    valid = depth > 0.1
    projected[valid, :2] /= depth[valid, None]
    height, width = image_shape[:2]
    columns = np.rint(projected[:, 0]).astype(np.int32)
    rows = np.rint(projected[:, 1]).astype(np.int32)
    valid &= (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    result = np.full((height, width), np.inf, dtype=np.float32)
    flat = rows[valid] * width + columns[valid]
    np.minimum.at(result.reshape(-1), flat, depth[valid])
    result[~np.isfinite(result)] = np.nan
    return result


def occupancy_points(occupancy, point_cloud_range):
    occupied = np.argwhere((occupancy != EMPTY_INDEX) & (occupancy != 255))
    if not len(occupied):
        return np.empty((0, 3), dtype=np.float32)
    lower = np.asarray(point_cloud_range[:3], dtype=np.float32)
    upper = np.asarray(point_cloud_range[3:], dtype=np.float32)
    voxel_size = (upper - lower) / np.asarray(occupancy.shape, dtype=np.float32)
    return (occupied.astype(np.float32) + 0.5) * voxel_size + lower


def render_depth_panel(depth, title, panel_size):
    finite = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if finite.any():
        normalized[finite] = np.clip((depth[finite] - 1.0) / 59.0 * 255,
                                     0, 255).astype(np.uint8)
    panel = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    panel[~finite] = (235, 235, 235)
    panel = cv2.resize(panel, panel_size, interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def render_image_panel(image, title, panel_size):
    panel = cv2.resize(image, panel_size, interpolation=cv2.INTER_LINEAR)
    cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def denormalize_input_image(image):
    """Convert the shared ImageToTensor-normalized input to OpenCV BGR."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f'Expected camera image [C,H,W], got {image.shape}.')
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    image = image * np.array([58.395, 57.12, 57.375], dtype=np.float32)
    image += np.array([123.675, 116.28, 103.53], dtype=np.float32)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def downsample_depth_for_loss(depth, output_shape):
    """Match BEVDepth's nearest-point depth target construction for display."""
    depth = np.asarray(depth, dtype=np.float32)
    output_height, output_width = (int(output_shape[0]), int(output_shape[1]))
    if depth.shape == (output_height, output_width):
        return depth
    height, width = depth.shape
    block_height = max(height // output_height, 1)
    block_width = max(width // output_width, 1)
    cropped = depth[:output_height * block_height, :output_width * block_width]
    cropped = cropped.reshape(
        output_height, block_height, output_width, block_width).transpose(0, 2, 1, 3)
    cropped = cropped.reshape(output_height, output_width, -1)
    valid = np.isfinite(cropped) & (cropped > 0.0)
    nearest = np.where(valid, cropped, np.inf).min(axis=-1)
    nearest[~np.isfinite(nearest)] = 0.0
    return nearest.astype(np.float32)


def depth_grid(info,
               prediction,
               gt,
               point_cloud_range,
               predicted_depth=None,
               predicted_depth_label='Pred(occ projection)',
               input_images=None,
               target_depth=None,
               camera_geometry=None):
    """Render camera depth diagnostics in the common validation layout.

    ``predicted_depth`` is an optional native depth estimate in metres with
    shape ``[N_cam, H, W]``.  Baselines without a depth branch leave it unset,
    in which case the last row is the occupancy-projected diagnostic.
    """
    lidar = np.fromfile(info['lidar_path'], dtype=np.float32).reshape(-1, 5)[:, :3]
    pred_points = occupancy_points(prediction, point_cloud_range)
    gt_points = occupancy_points(gt, point_cloud_range)
    if input_images is not None:
        input_images = np.asarray(input_images)
        if input_images.ndim != 4 or input_images.shape[0] != len(CAMERA_NAMES):
            raise ValueError(
                'Training input images must be [N_cam, C, H, W], got '
                f'{input_images.shape}.')
    if target_depth is not None:
        target_depth = np.asarray(target_depth, dtype=np.float32)
        if target_depth.ndim != 3 or target_depth.shape[0] != len(CAMERA_NAMES):
            raise ValueError(
                'Training depth targets must be [N_cam, H, W], got '
                f'{target_depth.shape}.')
    if camera_geometry is not None:
        required = {'rots', 'trans', 'intrins', 'post_rots', 'post_trans', 'bda'}
        if not required.issubset(camera_geometry):
            raise ValueError('Training camera geometry is incomplete.')
    if predicted_depth is not None:
        predicted_depth = np.asarray(predicted_depth, dtype=np.float32)
        if predicted_depth.ndim != 3 or predicted_depth.shape[0] != len(CAMERA_NAMES):
            raise ValueError(
                'Native predicted depth must have shape [N_cam, H, W], got '
                f'{predicted_depth.shape}.')
    rows = [[], [], [], []]
    for camera_index, camera_name in enumerate(CAMERA_NAMES):
        camera = info['cams'][camera_name]
        if input_images is None:
            image = cv2.imread(camera['data_path'], cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(camera['data_path'])
        else:
            image = denormalize_input_image(input_images[camera_index])
        panel_size = (352, 256)
        rows[0].append(render_image_panel(image, f'{camera_name} | Image', panel_size))
        if camera_geometry is None:
            raw_depth = camera_depth(lidar, camera, image.shape)
            occ_depth = camera_depth(gt_points, camera, image.shape)
            pred_occ_depth = camera_depth(pred_points, camera, image.shape)
        else:
            geometry = {
                key: np.asarray(value)[camera_index]
                if key != 'bda' else np.asarray(value)
                for key, value in camera_geometry.items()
            }
            # ``lidar`` still lives in the raw ego frame, whereas occupancy
            # voxels have already received BDA. Move raw points into the
            # augmented frame before the inverse-BDA camera projection.
            raw_lidar = lidar @ np.asarray(geometry['bda'], dtype=np.float32).T
            raw_depth = camera_depth_from_augmented_geometry(
                raw_lidar, geometry, image.shape)
            occ_depth = camera_depth_from_augmented_geometry(
                gt_points, geometry, image.shape)
            pred_occ_depth = camera_depth_from_augmented_geometry(
                pred_points, geometry, image.shape)
        rows[1].append(render_depth_panel(
            raw_depth if target_depth is None else target_depth[camera_index],
            f'{camera_name} | GT(raw)', panel_size))
        if target_depth is None:
            loss_depth = occ_depth
        else:
            loss_depth = downsample_depth_for_loss(
                target_depth[camera_index],
                (max(image.shape[0] // 16, 1), max(image.shape[1] // 16, 1)))
        rows[2].append(render_depth_panel(
            loss_depth, f'{camera_name} | GT(loss)', panel_size))
        if predicted_depth is not None:
            rows[3].append(render_depth_panel(
                predicted_depth[camera_index],
                f'{camera_name} | {predicted_depth_label}', panel_size))
        else:
            rows[3].append(render_depth_panel(
                pred_occ_depth,
                f'{camera_name} | {predicted_depth_label}', panel_size))
    image_rows = []
    for row in rows:
        image_rows.extend([cv2.hconcat(row[:2]), cv2.hconcat(row[2:])])
    return cv2.vconcat(image_rows)


def main():
    args = parse_args()
    payload = np.load(args.prediction)
    occupancy = payload['occ']
    flow = payload['flow']
    sample_index = int(payload['sample_index']) if 'sample_index' in payload else 0
    if occupancy.shape != flow.shape[:3] or flow.shape[-1] != 2:
        raise ValueError('Expected occ [X, Y, Z] and flow [X, Y, Z, 2].')
    info = load_info(args.ann_file, sample_index)
    gt = np.load(Path(info['occ_path']) / 'labels.npz')['semantics'].astype(np.uint8)
    gt_flow = load_occflow_gt(info)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / 'occ_bev.png'), render_occ_comparison(occupancy, gt))
    cv2.imwrite(
        str(output_dir / 'flow_bev.png'),
        render_flow_comparison(
            occupancy, flow, gt, gt_flow, args.point_cloud_range,
            max_speed=args.max_speed))
    cv2.imwrite(str(output_dir / 'depth_grid.png'),
                depth_grid(info, occupancy, gt, args.point_cloud_range))


if __name__ == '__main__':
    main()
