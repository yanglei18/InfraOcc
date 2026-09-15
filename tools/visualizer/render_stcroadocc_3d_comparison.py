"""Render VoxDet-style 3D occupancy panels for STCRoadOcc.

The renderer follows the visualization contract in
``docs/assets/code/VoxDet/tools/visualize.py``: occupied semantic voxels are
shown as solid, class-colored cubes under a fixed perspective camera.  Open3D
is used instead of Mayavi so the figure can be rendered offscreen on the
training server.
"""

import argparse
import os
import os.path as osp
import pickle

import open3d as o3d
import numpy as np


CLASS_NAMES = (
    'others', 'barrier', 'bicycle', 'bus', 'car',
    'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free')

# RGB palette matching the project's occupancy visualizations.
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
], dtype=np.float32) / 255.0

POINT_CLOUD_RANGE = np.asarray(
    [-64.0, -64.0, -4.8, 64.0, 64.0, 1.6], dtype=np.float32)
EMPTY_INDEX = CLASS_NAMES.index('free')
IGNORE_INDEX = 255


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render VoxDet-style STCRoadOcc occupancy volumes')
    parser.add_argument(
        '--ann-file',
        default='data/v2xreal_nuscenes/v2xreal_infos_val.pkl')
    parser.add_argument('--indices', type=int, nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument(
        '--prediction-dir',
        default=None,
        help='Directory containing <index>_<token>.npz predictions. Omit for GT.')
    parser.add_argument(
        '--tag', default='gt', help='Filename tag for the rendered volume.')
    parser.add_argument('--width', type=int, default=1200)
    parser.add_argument('--height', type=int, default=760)
    parser.add_argument('--fov', type=float, default=31.0)
    parser.add_argument(
        '--azimuth',
        type=float,
        default=-132.0,
        help='Camera azimuth in degrees around the scene center.')
    parser.add_argument(
        '--elevation', type=float, default=36.0,
        help='Camera elevation in degrees above the ground plane.')
    parser.add_argument(
        '--pure-white-background', action='store_true',
        help=('Disable Open3D post-processing so the empty canvas is rendered '
              'as uniform white.'))
    return parser.parse_args()


def visible_surface_mask(semantics):
    occupied = (semantics != EMPTY_INDEX) & (semantics != IGNORE_INDEX)
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


def voxel_centers(semantics, selected):
    indices = np.argwhere(selected).astype(np.float64)
    shape = np.asarray(semantics.shape, dtype=np.float64)
    extent = POINT_CLOUD_RANGE[3:] - POINT_CLOUD_RANGE[:3]
    voxel_size = extent / shape
    centers = POINT_CLOUD_RANGE[:3] + (indices + 0.5) * voxel_size
    return indices, centers, voxel_size


def render_volume(semantics, output_path, width, height, fov, azimuth,
                  elevation, pure_white_background=False):
    semantics = np.asarray(semantics)
    if semantics.shape != (320, 320, 16):
        raise ValueError(
            f'Expected a (320, 320, 16) volume, got {semantics.shape}.')
    selected = visible_surface_mask(semantics)
    indices, centers, voxel_size = voxel_centers(semantics, selected)
    labels = semantics[tuple(indices.astype(np.int64).T)].astype(np.int64)

    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(centers)
    points.colors = o3d.utility.Vector3dVector(
        CLASS_COLORS[np.clip(labels, 0, len(CLASS_COLORS) - 1)])
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        points, voxel_size=float(voxel_size[0] * 0.96))

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    if pure_white_background:
        renderer.scene.view.set_post_processing(False)
    renderer.scene.scene.set_sun_light(
        [-0.35, -0.55, -0.75], [1.0, 1.0, 1.0], 65000.0)
    renderer.scene.scene.enable_sun_light(True)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = 'defaultLit'
    renderer.scene.add_geometry('semantic_voxels', voxel_grid, material)

    # Keep the camera independent of the prediction.  A per-volume bounding
    # box makes matched GT/checkpoint crops drift by a few pixels whenever a
    # method misses voxels close to the scene boundary.
    center = (POINT_CLOUD_RANGE[:3] + POINT_CLOUD_RANGE[3:]) * 0.5
    planar_extent = float(np.max(
        POINT_CLOUD_RANGE[3:5] - POINT_CLOUD_RANGE[:2]))
    distance = planar_extent * 1.15
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)
    eye = center + distance * np.asarray([
        np.cos(elevation_rad) * np.cos(azimuth_rad),
        np.cos(elevation_rad) * np.sin(azimuth_rad),
        np.sin(elevation_rad),
    ])
    renderer.setup_camera(fov, center, eye, [0.0, 0.0, 1.0])

    image = renderer.render_to_image()
    os.makedirs(osp.dirname(osp.abspath(output_path)), exist_ok=True)
    if not o3d.io.write_image(output_path, image, 9):
        raise RuntimeError(f'Failed to write {output_path}.')
    del renderer
    return int(selected.sum())


def main():
    args = parse_args()
    with open(args.ann_file, 'rb') as annotation_file:
        annotation = pickle.load(annotation_file)
    infos = annotation['infos'] if isinstance(annotation, dict) else annotation
    os.makedirs(args.output_dir, exist_ok=True)
    for dataset_index in args.indices:
        info = infos[dataset_index]
        if args.prediction_dir is None:
            volume_path = osp.join(info['occ_path'], 'labels.npz')
        else:
            volume_path = osp.join(
                args.prediction_dir,
                f'{dataset_index:04d}_{info["token"]}.npz')
        semantics = np.load(volume_path)['semantics']
        output_path = osp.join(
            args.output_dir,
            f'{dataset_index:04d}_{info["token"]}_{args.tag}.png')
        surface_voxels = render_volume(
            semantics,
            output_path,
            width=args.width,
            height=args.height,
            fov=args.fov,
            azimuth=args.azimuth,
            elevation=args.elevation,
            pure_white_background=args.pure_white_background)
        print(
            f'{args.tag} index={dataset_index} token={info["token"]} '
            f'surface_voxels={surface_voxels} -> {output_path}',
            flush=True)


if __name__ == '__main__':
    main()
