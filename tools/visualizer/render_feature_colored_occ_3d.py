#!/usr/bin/env python3
"""Render real VDSF voxel-feature activation on occupancy geometry."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import collate
from mmcv.runner import load_checkpoint
from mmcv.utils import import_modules_from_strings


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmdet3d.core.visualizer.occ_visualization import (  # noqa: E402
    _orient_xy_grid_to_image)
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from tools.visualizer.compose_ptr_gt_max_scale import (  # noqa: E402
    DYNAMIC_INDICES, EMPTY_IDX, POINT_CLOUD_RANGE, load_semantics, resolve)
from tools.visualizer.render_ptr_gt_3d import (  # noqa: E402
    CLASS_COLORS, make_voxel_grid, opaque_material, transparent_material,
    visible_surface_mask)


DEFAULT_TOKEN = 'aafb051d-8810-426c-ae1b-7e66cf5c52a2'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        type=Path,
        default=ROOT / 'projects/STCRoadOcc/configs/'
        'stcroadocc_c_2x4_24e.py')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=ROOT / 'work_dirs/stcroadocc_c_2x4_24e/epoch_24.pth')
    parser.add_argument('--token', default=DEFAULT_TOKEN)
    parser.add_argument('--history-steps', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'docs/paper/ICLR2027/figures/'
        'roadocc_vdsf_feature_3d.png')
    parser.add_argument('--width', type=int, default=1200)
    parser.add_argument('--height', type=int, default=760)
    parser.add_argument('--fov', type=float, default=31.0)
    parser.add_argument('--azimuth', type=float, default=-132.0)
    parser.add_argument('--elevation', type=float, default=36.0)
    parser.add_argument('--static-alpha', type=float, default=0.24)
    parser.add_argument('--lower-percentile', type=float, default=2.0)
    parser.add_argument('--upper-percentile', type=float, default=99.5)
    return parser.parse_args()


def recursive_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(recursive_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [recursive_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {
            key: recursive_to_device(item, device)
            for key, item in value.items()
        }
    return value


def prepare_test_sample(dataset, index, device):
    batch = collate([dataset[index]], samples_per_gpu=1)
    img_metas = batch['img_metas'][0].data[0]
    points = recursive_to_device(batch['points'][0].data[0], device)
    img_inputs = recursive_to_device(batch['img_inputs'][0], device)
    return points, img_inputs, img_metas


def resolve_history_indices(data_infos, token, history_steps):
    lookup = {
        str(info.get('token')): index
        for index, info in enumerate(data_infos)
    }
    if token not in lookup:
        raise KeyError('Token {} is absent from the selected annotation file.'
                       .format(token))
    indices = [lookup[token]]
    info = data_infos[indices[0]]
    for _ in range(max(int(history_steps), 0)):
        previous_token = str(info.get('prev', ''))
        if not previous_token or previous_token not in lookup:
            break
        previous_index = lookup[previous_token]
        indices.append(previous_index)
        info = data_infos[previous_index]
    return list(reversed(indices))


def extract_feature_activation(args):
    cfg = Config.fromfile(str(args.config))
    import_modules_from_strings(**cfg.custom_imports)
    dataset_cfg = cfg.data.test.copy()
    dataset_cfg.ann_file = cfg.train_ann_file
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)
    indices = resolve_history_indices(
        dataset.data_infos, args.token, args.history_steps)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(
        model, str(args.checkpoint), map_location='cpu', strict=False)
    device = torch.device(args.device)
    model.to(device).eval()
    if hasattr(model, '_reset_temporal_runtime_state'):
        model._reset_temporal_runtime_state()

    final_output = None
    with torch.no_grad():
        for index in indices:
            points, img_inputs, img_metas = prepare_test_sample(
                dataset, index, device)
            final_output = model.obtain_voxel_feats(
                points, img_inputs=img_inputs, img_metas=img_metas)

    feature = final_output['voxel_feats']
    feature_shape = [int(value) for value in feature.shape[1:]]
    head_output = model._decode_occupancy_and_flow_features(
        feature, prev_occ_pred=final_output['occ_pred'])
    predicted_semantics = head_output['semantic_logits'].argmax(
        dim=-1)[0].to(torch.uint8).cpu().numpy()
    # Finest VDSF feature: [B, C, Z, Y, X]. Preserve spatial variation in all
    # three dimensions before mapping the scalar response to colour.
    activation = feature.detach().abs().amax(dim=1, keepdim=True)
    current_info = dataset.data_infos[indices[-1]]
    semantics = load_semantics(resolve(current_info['occ_path']) /
                               'labels.npz')
    native_shape_zyx = (
        int(semantics.shape[2]), int(semantics.shape[1]),
        int(semantics.shape[0]))
    activation = F.interpolate(
        activation,
        size=native_shape_zyx,
        mode='trilinear',
        align_corners=False)
    activation_zyx = activation[0, 0].float().cpu().numpy()
    activation_xyz = activation_zyx.transpose(2, 1, 0)
    del model
    torch.cuda.empty_cache()
    return (semantics, predicted_semantics, activation_xyz, feature_shape,
            indices, current_info)


def normalize_activation(activation, support, lower_percentile,
                         upper_percentile):
    values = activation[support]
    if not len(values):
        raise ValueError('Feature activation has no occupied support.')
    lower = float(np.percentile(values, lower_percentile))
    upper = float(np.percentile(values, upper_percentile))
    if upper <= lower:
        upper = lower + 1e-6
    normalized = np.clip((activation - lower) / (upper - lower), 0.0, 1.0)
    return normalized.astype(np.float32), lower, upper


def activation_colors(normalized):
    # Reverse only the visual scale so compact vehicle responses are bright;
    # the underlying activation values and percentile normalization are kept.
    scalar = np.rint((1.0 - normalized) * 255.0).astype(np.uint8)
    # OpenCV applies colour maps to images rather than scalar volumes. Build a
    # compact BGR lookup table, index it voxel-wise, then convert to RGB for
    # Open3D.
    lookup = cv2.applyColorMap(
        np.arange(256, dtype=np.uint8).reshape(256, 1),
        cv2.COLORMAP_VIRIDIS).reshape(256, 3)
    return lookup[scalar][..., ::-1] / 255.0


def configure_renderer(args):
    renderer = o3d.visualization.rendering.OffscreenRenderer(
        args.width, args.height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    # Disable Filament's vignette so the empty canvas stays paper-white.
    renderer.scene.view.set_post_processing(False)
    renderer.scene.scene.set_sun_light(
        [-0.35, -0.55, -0.75], [1.0, 1.0, 1.0], 65000.0)
    renderer.scene.scene.enable_sun_light(True)

    return renderer


def setup_camera(renderer, args):
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


def render_feature_volume(semantics, colors, args, output, dynamic_focus):
    surface = visible_surface_mask(semantics)
    dynamic = np.isin(semantics, DYNAMIC_INDICES)
    renderer = configure_renderer(args)
    if dynamic_focus:
        static_grid = make_voxel_grid(
            semantics, surface & ~dynamic, colors)
        dynamic_grid = make_voxel_grid(
            semantics, surface & dynamic, colors)
        if static_grid is not None:
            renderer.scene.add_geometry(
                'static_feature_context', static_grid,
                transparent_material(args.static_alpha))
        if dynamic_grid is not None:
            renderer.scene.add_geometry(
                'dynamic_feature_response', dynamic_grid,
                opaque_material())
    else:
        feature_grid = make_voxel_grid(semantics, surface, colors)
        renderer.scene.add_geometry(
            'feature_response', feature_grid, opaque_material())

    # Open3D initializes its scene bounds when geometry is submitted. Apply
    # the fixed paper camera afterwards so adding the first voxel grid cannot
    # reset the view to an empty-scene default.
    setup_camera(renderer, args)
    image = renderer.render_to_image()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_image(str(output), image, 9):
        raise RuntimeError('Failed to write {}.'.format(output))
    del renderer


def render_feature_bev(activation,
                       semantics,
                       output,
                       masked_output,
                       lower_percentile,
                       upper_percentile,
                       size=1200):
    """Normalize the height-reduced BEV independently from the 3D surface."""
    feature_bev = activation.max(axis=2)
    finite = np.isfinite(feature_bev)
    lower = float(np.percentile(
        feature_bev[finite], lower_percentile))
    upper = float(np.percentile(
        feature_bev[finite], upper_percentile))
    if upper <= lower:
        upper = lower + 1e-6
    feature_bev = np.clip(
        (feature_bev - lower) / (upper - lower), 0.0, 1.0)
    feature_bev = _orient_xy_grid_to_image(feature_bev)
    feature_bev = cv2.resize(
        feature_bev, (size, size), interpolation=cv2.INTER_LINEAR)
    feature_color = cv2.applyColorMap(
        np.rint((1.0 - feature_bev) * 255.0).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS)

    occupied_bev = ((semantics != EMPTY_IDX) &
                    (semantics != 255)).any(axis=2).astype(np.uint8)
    occupied_bev = _orient_xy_grid_to_image(occupied_bev)
    occupied_bev = cv2.resize(
        occupied_bev, (size, size), interpolation=cv2.INTER_NEAREST).astype(
            bool)
    masked_color = np.full_like(feature_color, 245)
    masked_color[occupied_bev] = feature_color[occupied_bev]

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), feature_color)
    cv2.imwrite(str(masked_output), masked_color)
    return lower, upper


def main():
    args = parse_args()
    (semantics, predicted_semantics, activation, feature_shape, indices,
     current_info) = extract_feature_activation(args)
    surface = visible_surface_mask(semantics)
    normalized, lower, upper = normalize_activation(
        activation,
        surface,
        args.lower_percentile,
        args.upper_percentile)
    colors = activation_colors(normalized)

    focused_output = args.output.with_name(
        args.output.stem + '_dynamic' + args.output.suffix)
    bev_output = args.output.with_name(
        args.output.stem.replace('_3d', '_bev') + args.output.suffix)
    masked_bev_output = bev_output.with_name(
        bev_output.stem + '_occupied' + bev_output.suffix)
    predicted_occ_output = args.output.with_name(
        'roadocc_vdsf_occ_pred_3d' + args.output.suffix)
    gt_occ_output = args.output.with_name(
        'roadocc_occ_gt_3d' + args.output.suffix)
    render_feature_volume(
        semantics, colors, args, args.output, dynamic_focus=False)
    render_feature_volume(
        semantics, colors, args, focused_output, dynamic_focus=True)
    predicted_colors = CLASS_COLORS[np.clip(
        predicted_semantics.astype(np.int64), 0, len(CLASS_COLORS) - 1)]
    gt_colors = CLASS_COLORS[np.clip(
        semantics.astype(np.int64), 0, len(CLASS_COLORS) - 1)]
    render_feature_volume(
        semantics,
        gt_colors,
        args,
        gt_occ_output,
        dynamic_focus=False)
    render_feature_volume(
        predicted_semantics,
        predicted_colors,
        args,
        predicted_occ_output,
        dynamic_focus=False)
    bev_lower, bev_upper = render_feature_bev(
        activation,
        semantics,
        bev_output,
        masked_bev_output,
        args.lower_percentile,
        args.upper_percentile,
        size=args.width)

    cache_dir = ROOT / 'work_dirs/stcroadocc_c_2x4_24e/' \
        'feature_coloring'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / '{}_vdsf_feature_activation.npz'.format(
        args.token)
    np.savez_compressed(
        cache_path,
        activation=activation.astype(np.float16),
        semantics=semantics.astype(np.uint8),
        predicted_semantics=predicted_semantics.astype(np.uint8))
    manifest = {
        'token': args.token,
        'dataset_index': int(indices[-1]),
        'history_indices': [int(index) for index in indices[:-1]],
        'checkpoint': str(args.checkpoint),
        'feature': 'finest 1/2 VDSF output before the occupancy head',
        'feature_shape': feature_shape,
        'activation': 'maximum absolute response over channels',
        'colormap': {
            'name': 'reversed viridis',
            'mapping': 'lower normalized response is brighter',
        },
        'native_activation_shape': list(activation.shape),
        'normalization': {
            'support': 'visible GT occupied surface voxels',
            'lower_percentile': float(args.lower_percentile),
            'upper_percentile': float(args.upper_percentile),
            'lower_value': lower,
            'upper_value': upper,
        },
        'bev_normalization': {
            'support': 'all finite BEV cells after height-wise maximum',
            'lower_percentile': float(args.lower_percentile),
            'upper_percentile': float(args.upper_percentile),
            'lower_value': bev_lower,
            'upper_value': bev_upper,
        },
        'camera': {
            'width': int(args.width),
            'height': int(args.height),
            'fov_degrees': float(args.fov),
            'azimuth_degrees': float(args.azimuth),
            'elevation_degrees': float(args.elevation),
        },
        'outputs': {
            'full_3d': str(args.output),
            'dynamic_focus_3d': str(focused_output),
            'bev': str(bev_output),
            'occupied_bev': str(masked_bev_output),
            'predicted_occ_3d': str(predicted_occ_output),
            'gt_occ_3d': str(gt_occ_output),
            'activation_cache': str(cache_path),
        },
        'source_occupancy': str(resolve(current_info['occ_path']) /
                                'labels.npz'),
    }
    manifest_path = args.output.with_suffix('.json')
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
