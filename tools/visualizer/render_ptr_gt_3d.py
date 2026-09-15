#!/usr/bin/env python3
"""Render a P/T/R ground-truth volume with translucent static context.

The target construction is shared with the verified native-scale P/T/R
visualizer.  The camera and voxel rendering match the source packet used for
the RoadOcc qualitative comparison so that the result remains spatially
comparable with the occupancy renderings.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.visualizer.compose_ptr_gt_max_scale import (  # noqa: E402
    DYNAMIC_INDICES, EMPTY_IDX, POINT_CLOUD_RANGE, align_previous,
    build_target, load_semantics)


IGNORE_IDX = 255
STATIC_ALPHA = 0.42
ROUTE_COLORS = {
    'persist': np.asarray([54, 116, 217], dtype=np.float64) / 255.0,
    'transport': np.asarray([232, 142, 38], dtype=np.float64) / 255.0,
    'refresh': np.asarray([44, 160, 92], dtype=np.float64) / 255.0,
}

# RGB palette shared by the 3D occupancy comparison renderer.
CLASS_COLORS = np.asarray([
    [0, 0, 0],
    [255, 120, 50],
    [255, 192, 203],
    [255, 255, 0],
    [0, 150, 245],
    [0, 255, 255],
    [255, 127, 0],
    [255, 0, 0],
    [255, 240, 150],
    [135, 60, 0],
    [160, 32, 240],
    [255, 0, 255],
    [139, 137, 137],
    [75, 0, 75],
    [150, 240, 80],
    [230, 230, 250],
    [0, 175, 0],
    [255, 255, 255],
], dtype=np.float64) / 255.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file',
        type=Path,
        default=ROOT / 'data/v2xreal_nuscenes/v2xreal_infos_val.pkl')
    parser.add_argument('--index', type=int, default=558)
    parser.add_argument(
        '--token',
        default=None,
        help='Select a sample by token. When set, this overrides --index.')
    parser.add_argument(
        '--flow-root',
        type=Path,
        default=ROOT / 'data/v2xreal_nuscenes/occflow_gts')
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'docs/paper/ICLR2027/figures/'
        'roadocc_ptr_gt_3d.png')
    parser.add_argument('--width', type=int, default=1200)
    parser.add_argument('--height', type=int, default=760)
    parser.add_argument('--fov', type=float, default=31.0)
    parser.add_argument('--azimuth', type=float, default=-132.0)
    parser.add_argument('--elevation', type=float, default=36.0)
    parser.add_argument('--static-alpha', type=float, default=STATIC_ALPHA)
    parser.add_argument('--frame-dt', type=float, default=0.5)
    parser.add_argument('--flow-epsilon', type=float, default=1e-3)
    parser.add_argument('--correspondence-radius', type=int, default=1)
    return parser.parse_args()


def visible_surface_mask(semantics):
    occupied = (semantics != EMPTY_IDX) & (semantics != IGNORE_IDX)
    interior = occupied.copy()
    for axis in range(3):
        forward = np.zeros_like(occupied)
        backward = np.zeros_like(occupied)
        forward_slice = [slice(None)] * 3
        backward_slice = [slice(None)] * 3
        source_forward = [slice(None)] * 3
        source_backward = [slice(None)] * 3
        forward_slice[axis] = slice(0, -1)
        source_forward[axis] = slice(1, None)
        backward_slice[axis] = slice(1, None)
        source_backward[axis] = slice(0, -1)
        forward[tuple(forward_slice)] = occupied[tuple(source_forward)]
        backward[tuple(backward_slice)] = occupied[tuple(source_backward)]
        interior &= forward & backward
    return occupied & ~interior


def make_voxel_grid(semantics, selected, colors):
    indices = np.argwhere(selected).astype(np.int64)
    if not len(indices):
        return None
    shape = np.asarray(semantics.shape, dtype=np.float64)
    point_cloud_range = np.asarray(POINT_CLOUD_RANGE, dtype=np.float64)
    voxel_size = ((point_cloud_range[3:] - point_cloud_range[:3]) / shape)
    centers = point_cloud_range[:3] + (indices + 0.5) * voxel_size
    if colors.ndim == 1:
        point_colors = np.repeat(colors[None, :], len(indices), axis=0)
    else:
        point_colors = colors[tuple(indices.T)]
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(centers)
    points.colors = o3d.utility.Vector3dVector(point_colors)
    return o3d.geometry.VoxelGrid.create_from_point_cloud(
        points, voxel_size=float(voxel_size[0] * 0.96))


def transparent_material(alpha):
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = 'defaultLitTransparency'
    material.base_color = [1.0, 1.0, 1.0, float(alpha)]
    material.base_roughness = 0.9
    material.base_reflectance = 0.0
    return material


def opaque_material():
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = 'defaultLit'
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.base_roughness = 0.72
    material.base_reflectance = 0.12
    return material


def render(semantics, target, args):
    if semantics.shape != (320, 320, 16):
        raise ValueError(
            'Expected a (320, 320, 16) volume, got {}.'.format(
                semantics.shape))
    surface = visible_surface_mask(semantics)
    dynamic = np.isin(semantics, DYNAMIC_INDICES)
    static = surface & ~dynamic
    semantic_colors = CLASS_COLORS[np.clip(
        semantics.astype(np.int64), 0, len(CLASS_COLORS) - 1)]

    renderer = o3d.visualization.rendering.OffscreenRenderer(
        args.width, args.height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.view.set_post_processing(False)
    renderer.scene.scene.set_sun_light(
        [-0.35, -0.55, -0.75], [1.0, 1.0, 1.0], 65000.0)
    renderer.scene.scene.enable_sun_light(True)

    static_grid = make_voxel_grid(semantics, static, semantic_colors)
    if static_grid is not None:
        renderer.scene.add_geometry(
            'static_context', static_grid,
            transparent_material(args.static_alpha))

    route_counts = {}
    for route_name in ('persist', 'transport', 'refresh'):
        selected = surface & target[route_name]
        route_counts[route_name] = int(target[route_name].sum())
        route_grid = make_voxel_grid(
            semantics, selected, ROUTE_COLORS[route_name])
        if route_grid is not None:
            renderer.scene.add_geometry(
                route_name, route_grid, opaque_material())

    point_cloud_range = np.asarray(POINT_CLOUD_RANGE, dtype=np.float64)
    center = (point_cloud_range[:3] + point_cloud_range[3:]) * 0.5
    planar_extent = float(np.max(
        point_cloud_range[3:5] - point_cloud_range[:2]))
    distance = planar_extent * 1.15
    azimuth = np.deg2rad(args.azimuth)
    elevation = np.deg2rad(args.elevation)
    eye = center + distance * np.asarray([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    renderer.setup_camera(args.fov, center, eye, [0.0, 0.0, 1.0])

    image = renderer.render_to_image()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_image(str(args.output), image, 9):
        raise RuntimeError('Failed to write {}.'.format(args.output))
    del renderer
    return route_counts, int(static.sum())


def resolve_flow_path(flow_root, occ_path):
    occ_path = Path(occ_path)
    parts = occ_path.parts
    if 'gts' not in parts:
        raise ValueError('Cannot resolve flow label from {}.'.format(occ_path))
    relative = Path(*parts[parts.index('gts') + 1:])
    return flow_root / relative / 'labels.npz'


def main():
    args = parse_args()
    with args.ann_file.open('rb') as annotation_file:
        annotation = pickle.load(annotation_file)
    infos = annotation['infos'] if isinstance(annotation, dict) else annotation
    lookup = {str(info['token']): info for info in infos}
    if args.token is None:
        current_info = infos[args.index]
    else:
        if args.token not in lookup:
            raise KeyError('Token {} is absent from {}.'.format(
                args.token, args.ann_file))
        current_info = lookup[args.token]
        args.index = next(
            index for index, info in enumerate(infos)
            if str(info['token']) == args.token)
    previous_info = lookup.get(str(current_info.get('prev', '')))
    if previous_info is None:
        raise ValueError('Sample {} has no labelled previous frame.'.format(
            args.index))

    current_path = ROOT / current_info['occ_path']
    previous_path = ROOT / previous_info['occ_path']
    current_gt = load_semantics(current_path / 'labels.npz')
    previous_raw = load_semantics(previous_path / 'labels.npz')
    previous_gt = align_previous(
        current_info, previous_info, current_gt, previous_raw)
    flow_path = resolve_flow_path(args.flow_root,
                                  current_info['occ_path'])
    with np.load(flow_path, allow_pickle=False) as payload:
        flow = payload['flow'].astype(np.float32)
    target = build_target(current_gt, previous_gt, flow, args)
    route_counts, static_surface_count = render(current_gt, target, args)

    audit_dir = ROOT / 'work_dirs/stcroadocc_c_2x4_24e/' \
        'figures_3d_comparison/Fig4-comparison-source/volumes/PTR-GT'
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_stem = '{:04d}_{}'.format(args.index, current_info['token'])
    route_volume = np.full(current_gt.shape, 255, dtype=np.uint8)
    for route_index, route_name in enumerate(
            ('persist', 'transport', 'refresh')):
        route_volume[target[route_name]] = route_index
    np.savez_compressed(
        audit_dir / '{}.npz'.format(audit_stem),
        semantics=current_gt.astype(np.uint8),
        ptr_target=route_volume)

    dynamic_count = int(target['dynamic'].sum())
    manifest = {
        'dataset_index': int(args.index),
        'token': str(current_info['token']),
        'current_occupancy': str(current_path / 'labels.npz'),
        'previous_occupancy': str(previous_path / 'labels.npz'),
        'flow_target': str(flow_path),
        'target_definition': {
            'frame_dt_seconds': float(args.frame_dt),
            'flow_epsilon_mps': float(args.flow_epsilon),
            'correspondence_radius_voxels': int(
                args.correspondence_radius),
        },
        'route_counts': route_counts,
        'route_percent': {
            key: 100.0 * value / max(dynamic_count, 1)
            for key, value in route_counts.items()
        },
        'dynamic_voxels': dynamic_count,
        'static_surface_voxels': static_surface_count,
        'static_alpha': float(args.static_alpha),
        'camera': {
            'width': int(args.width),
            'height': int(args.height),
            'fov_degrees': float(args.fov),
            'azimuth_degrees': float(args.azimuth),
            'elevation_degrees': float(args.elevation),
        },
        'output': str(args.output),
    }
    manifest_path = args.output.with_suffix('.json')
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
