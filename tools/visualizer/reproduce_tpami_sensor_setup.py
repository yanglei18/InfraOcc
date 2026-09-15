#!/usr/bin/env python3
"""Reproduce the visual grammar of TPAMI Fig. 2 from V2X-Real raw data.

This utility never writes into ``docs/``.  It combines a colored roadside
LiDAR backdrop with four calibrated camera views, their LiDAR projections,
infrastructure-unit labels, and leader lines.  The result is a reproducible
counterpart to the reference layout, not a copied or cropped reference image.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INFO = ROOT / 'data/v2xreal_nuscenes/v2xreal_infos_train.pkl'
DEFAULT_OUTPUT = ROOT / 'tools/visualizer/tpami_reproduction/fig02-sensor-setup-reproduction.png'
DEFAULT_ASSET_DIR = ROOT / 'tools/visualizer/tpami_reproduction/fig02_sensor_assets'
CAMERA_ORDER = ('CAM_FRONT', 'CAM_RIGHT', 'CAM_LEFT', 'CAM_BACK')
INK = '#4b4b4b'


def load_infos(path):
    with path.open('rb') as handle:
        data = pickle.load(handle)
    if isinstance(data, dict):
        data = data.get('infos', data.get('data_list', data))
    if not isinstance(data, list):
        raise TypeError(f'Unsupported info format: {type(data)}')
    return data


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_lidar(info, max_points=None):
    points = np.fromfile(resolve(info['lidar_path']), dtype=np.float32).reshape(-1, 5)
    points = points[np.isfinite(points[:, :3]).all(axis=1)]
    keep = ((np.abs(points[:, 0]) < 70) & (np.abs(points[:, 1]) < 70)
            & (points[:, 2] > -4) & (points[:, 2] < 12))
    points = points[keep]
    if max_points is None or len(points) <= max_points:
        return points
    # Deterministic thinning retains the point-cloud texture without raster overload.
    return points[::max(1, len(points) // max_points)]


def point_colors(points):
    """Emphasize the ground plane while retaining colored vertical structure."""
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    height = np.clip((points[:, 2] + 2.0) / 7.0, 0.0, 1.0)
    palette = np.array([[0.30, 0.84, 0.88], [0.32, 0.83, 0.46], [0.90, 0.38, 0.52]])
    index = np.mod(np.floor((azimuth + np.pi) / (2 * np.pi / 3)).astype(int), 3)
    colors = palette[index]
    colors[points[:, 2] < 0.25] = np.array([0.32, 0.84, 0.43])
    return np.clip(colors * (0.72 + 0.28 * height[:, None]), 0, 1)


def project_lidar(points_xyz, cam):
    rotation = np.asarray(cam['sensor2lidar_rotation'], dtype=np.float32)
    translation = np.asarray(cam['sensor2lidar_translation'], dtype=np.float32)
    points_cam = (points_xyz - translation) @ rotation.T
    depth = points_cam[:, 2]
    intrinsic = np.asarray(cam['cam_intrinsic'], dtype=np.float32)
    image_xy = (points_cam @ intrinsic.T)[:, :2] / np.maximum(depth[:, None], 1e-5)
    image = np.asarray(Image.open(resolve(cam['data_path'])).convert('RGB'))
    height, width = image.shape[:2]
    valid = ((depth > 0.5) & (image_xy[:, 0] >= 0) & (image_xy[:, 0] < width)
             & (image_xy[:, 1] >= 0) & (image_xy[:, 1] < height))
    return image, image_xy[valid], depth[valid], np.flatnonzero(valid)


def sample_camera_colors(points_xyz, cameras):
    """Paint LiDAR points from calibrated camera observations when available."""
    fallback = point_colors(np.column_stack([points_xyz, np.zeros(len(points_xyz))]))
    colors = fallback.copy()
    observed = np.zeros(len(points_xyz), dtype=bool)
    for camera_name in CAMERA_ORDER:
        image, uv, _, indices = project_lidar(points_xyz, cameras[camera_name])
        if not len(indices):
            continue
        pixels = np.rint(uv).astype(np.int32)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, image.shape[1] - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, image.shape[0] - 1)
        colors[indices] = image[pixels[:, 1], pixels[:, 0]].astype(np.float32) / 255.0
        observed[indices] = True
    return colors, observed


def make_view_panel(fig, bounds, image, uv, depth, title):
    ax = fig.add_axes(bounds, zorder=8)
    ax.imshow(image)
    if len(uv):
        sample = np.arange(len(uv))[::max(1, len(uv) // 7000)]
        ax.scatter(uv[sample, 0], uv[sample, 1], c=depth[sample], cmap='turbo',
                   s=0.55, alpha=0.74, linewidths=0)
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.8)
        spine.set_edgecolor(INK)
    ribbon_height = 0.115
    ax.add_patch(Rectangle((0, 1 - ribbon_height), 0.36, ribbon_height,
                           transform=ax.transAxes, facecolor=INK, edgecolor='none', alpha=0.94,
                           zorder=12))
    ax.text(0.02, 1 - ribbon_height / 2, title, transform=ax.transAxes, color='white',
            fontsize=15, weight='medium', va='center', ha='left', zorder=13)
    return ax


def cloud_limits(points):
    lower = np.quantile(points[:, :3], 0.01, axis=0)
    upper = np.quantile(points[:, :3], 0.99, axis=0)
    padding = 0.06 * (upper - lower)
    return lower - padding, upper + padding


def render_lidar_backdrop(points, output, dpi, colors=None):
    """Export only the dense colored LiDAR background for manual composition."""
    figure = plt.figure(figsize=(12.8, 8.4), dpi=dpi, facecolor='white')
    cloud = figure.add_axes([0, 0, 1, 1], projection='3d')
    if colors is None:
        colors = point_colors(points)
    cloud.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors,
                  s=0.34, alpha=0.84, depthshade=False, linewidths=0)
    lower, upper = cloud_limits(points)
    cloud.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
              zlim=(lower[2], upper[2]))
    cloud.view_init(elev=22, azim=-132)
    # A taller visual depth makes building facades read as a point-cloud scene
    # rather than a flattened BEV ribbon when the asset is placed at centre.
    cloud.set_box_aspect((1.35, 1.0, 0.82))
    cloud.set_axis_off()
    cloud.patch.set_alpha(0)
    figure.savefig(output, dpi=dpi, facecolor='white', bbox_inches='tight', pad_inches=0)
    plt.close(figure)


def render_projected_view(image, uv, depth, output, dpi):
    """Export one camera frame with calibrated LiDAR color projection."""
    height, width = image.shape[:2]
    figure = plt.figure(figsize=(width / 120.0, height / 120.0), dpi=dpi, facecolor='white')
    ax = figure.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    if len(uv):
        sample = np.arange(len(uv))[::max(1, len(uv) // 65000)]
        ax.scatter(uv[sample, 0], uv[sample, 1], c=depth[sample], cmap='turbo',
                   s=1.25, alpha=0.88, linewidths=0)
    ax.set_axis_off()
    figure.savefig(output, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(figure)


def export_assets(info, asset_dir, dpi):
    """Write the central LiDAR image and four independent projected views."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    points = load_lidar(info)
    painted_colors, observed = sample_camera_colors(points[:, :3], info['cams'])
    # Points outside every camera frustum retain the vivid geometric fallback.
    painted_colors[~observed] = point_colors(points)[~observed]
    render_lidar_backdrop(points, asset_dir / 'central_lidar_backdrop.png', dpi,
                          colors=painted_colors)
    projection_points = load_lidar(info, max_points=180000)
    for index, camera_name in enumerate(CAMERA_ORDER, start=1):
        image, uv, depth, _ = project_lidar(projection_points[:, :3], info['cams'][camera_name])
        name = f'view{index}_{camera_name.lower()}_lidar_projection.png'
        render_projected_view(image, uv, depth, asset_dir / name, dpi)


def add_leader(fig, panel_anchor, cloud_anchor):
    line = Line2D([panel_anchor[0], cloud_anchor[0]], [panel_anchor[1], cloud_anchor[1]],
                  transform=fig.transFigure, color=INK, linewidth=2.0, zorder=10)
    dot = Circle(cloud_anchor, 0.006, transform=fig.transFigure,
                 facecolor=INK, edgecolor='none', zorder=11)
    fig.add_artist(line)
    fig.add_artist(dot)


def add_unit_label(fig, position, name):
    fig.text(*position, name, color='white', fontsize=16, va='center', ha='left', zorder=12,
             bbox=dict(boxstyle='square,pad=0.22', facecolor=INK, edgecolor=INK, alpha=0.94))


def render(info, output, dpi):
    points = load_lidar(info)
    figure = plt.figure(figsize=(18, 12), dpi=dpi, facecolor='white')
    cloud = figure.add_axes([0.0, 0.0, 1.0, 1.0], projection='3d', zorder=0)
    cloud.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_colors(points),
                  s=0.48, alpha=0.76, depthshade=False, linewidths=0)
    cloud.set(xlim=(-65, 65), ylim=(-65, 65), zlim=(-4, 12))
    cloud.view_init(elev=23, azim=-132)
    cloud.set_box_aspect((1.35, 1.0, 0.42))
    cloud.set_axis_off()
    cloud.patch.set_alpha(0)

    panel_bounds = ((0.015, 0.57, 0.255, 0.39), (0.015, 0.16, 0.255, 0.39),
                    (0.735, 0.42, 0.25, 0.38), (0.735, 0.02, 0.25, 0.38))
    panel_anchors = ((0.27, 0.82), (0.27, 0.67), (0.735, 0.56), (0.735, 0.14))
    for index, (camera_name, bounds) in enumerate(zip(CAMERA_ORDER, panel_bounds), start=1):
        image, uv, depth, _ = project_lidar(points[:, :3], info['cams'][camera_name])
        make_view_panel(figure, bounds, image, uv, depth, f'view {index}')

    add_leader(figure, panel_anchors[0], (0.31, 0.66))
    add_leader(figure, panel_anchors[1], (0.31, 0.60))
    add_leader(figure, panel_anchors[2], (0.86, 0.57))
    add_leader(figure, panel_anchors[3], (0.74, 0.14))
    add_unit_label(figure, (0.14, 0.47), 'Infrastructure unit 1')
    add_unit_label(figure, (0.49, 0.52), 'Infrastructure unit 2')
    figure.savefig(output, dpi=dpi, facecolor='white', bbox_inches='tight', pad_inches=0.01)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--info-pkl', type=Path, default=DEFAULT_INFO)
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--asset-dir', type=Path, default=DEFAULT_ASSET_DIR,
                        help='Directory for the separate LiDAR backdrop and view assets.')
    parser.add_argument('--compose', action='store_true',
                        help='Also produce the provisional composed layout.')
    parser.add_argument('--dpi', type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    infos = load_infos(args.info_pkl)
    if not 0 <= args.sample_index < len(infos):
        raise IndexError(f'--sample-index must be in [0, {len(infos) - 1}].')
    export_assets(infos[args.sample_index], args.asset_dir, args.dpi)
    print(f'Saved independent assets to {args.asset_dir}')
    if args.compose:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        render(infos[args.sample_index], args.output, args.dpi)
        print(f'Saved provisional composition to {args.output}')


if __name__ == '__main__':
    main()
