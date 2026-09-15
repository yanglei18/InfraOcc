#!/usr/bin/env python3
"""Export paired native-scale GT flow and P/T/R BEV visualizations."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import mmcv
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compose_ptr_gt_max_scale import (  # noqa: E402
    CLASS_NAMES, DYNAMIC_INDICES, EMPTY_IDX, P_COLOR, R_COLOR, T_COLOR,
    combined_panel, gray_context, route_panel)
from compose_ptr_gt_multiscale import load_scale_state  # noqa: E402
from mmdet3d.core.visualizer.occ_visualization import (  # noqa: E402
    _orient_xy_grid_to_image, flow_to_rgb, make_occ_color_map,
    project_dynamic_flow_to_bev, render_occ_bev_rgb)


TOKEN = 'aafb051d-8810-426c-ae1b-7e66cf5c52a2'
DYNAMIC_COLOR = (124, 92, 230)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file',
        type=Path,
        default=ROOT / 'data/v2xreal_nuscenes/v2xreal_infos_train.pkl')
    parser.add_argument('--token', default=TOKEN)
    parser.add_argument(
        '--flow-root',
        type=Path,
        default=ROOT / 'data/v2xreal_nuscenes/occflow_gts')
    parser.add_argument('--frame-dt', type=float, default=0.5)
    parser.add_argument('--flow-epsilon', type=float, default=1e-3)
    parser.add_argument('--correspondence-radius', type=int, default=1)
    parser.add_argument('--boundary-epsilon', type=float, default=1e-6)
    parser.add_argument('--size', type=int, default=1200)
    parser.add_argument('--max-speed', type=float, default=8.0)
    parser.add_argument('--min-arrow-speed', type=float, default=0.4)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=ROOT / 'docs/paper/ICLR2027/figures')
    return parser.parse_args()


def draw_component_arrows(image, display_flow, moving_mask, args):
    """Draw one representative GT displacement arrow per BEV component."""
    component_count, labels = cv2.connectedComponents(
        moving_mask.astype(np.uint8), connectivity=8)
    output = image.copy()
    pixel_scale = float(args.size) / float(display_flow.shape[0])
    voxel_size = 0.4
    thickness = max(int(round(args.size / 500.0)), 2)
    for component_index in range(1, component_count):
        component = labels == component_index
        rows, cols = np.nonzero(component)
        if not len(rows):
            continue
        vectors = display_flow[component]
        vector = np.median(vectors, axis=0)
        speed = float(np.linalg.norm(vector))
        if speed < float(args.min_arrow_speed):
            continue
        start = np.asarray([
            (float(np.median(cols)) + 0.5) * pixel_scale,
            (float(np.median(rows)) + 0.5) * pixel_scale,
        ])
        delta = np.asarray([
            vector[0] / voxel_size * pixel_scale * args.frame_dt,
            vector[1] / voxel_size * pixel_scale * args.frame_dt,
        ])
        length = float(np.linalg.norm(delta))
        if length > 52.0:
            delta *= 52.0 / max(length, 1e-6)
        end = start + delta
        cv2.arrowedLine(
            output,
            tuple(np.rint(start).astype(np.int32)),
            tuple(np.rint(end).astype(np.int32)),
            (28, 28, 28),
            thickness=thickness,
            line_type=cv2.LINE_AA,
            tipLength=0.24)
    return output


def render_flow_with_context(flow, semantics, base, args):
    flow_bev, valid_mask = project_dynamic_flow_to_bev(
        flow,
        semantics,
        DYNAMIC_INDICES,
        fill_iterations=0,
        neighbor_radius=1,
        min_neighbors=5)
    if flow_bev is None or valid_mask is None:
        raise ValueError('Could not project native GT flow to BEV.')

    display_flow = flow_bev.copy()
    display_flow[..., 0] *= -1.0
    display_flow[..., 1] *= -1.0
    display_flow = _orient_xy_grid_to_image(display_flow)
    display_valid = _orient_xy_grid_to_image(
        valid_mask.astype(np.uint8)).astype(bool)
    speed = np.linalg.norm(display_flow, axis=-1)
    moving_native = display_valid & (speed > float(args.flow_epsilon))
    flow_rgb = flow_to_rgb(
        display_flow[..., 0],
        display_flow[..., 1],
        max_magnitude=args.max_speed)
    flow_rgb = cv2.resize(
        flow_rgb, (args.size, args.size), interpolation=cv2.INTER_NEAREST)
    moving = cv2.resize(
        moving_native.astype(np.uint8), (args.size, args.size),
        interpolation=cv2.INTER_NEAREST).astype(bool)

    output = base.astype(np.float32).copy()
    output[moving] = (
        0.08 * output[moving] + 0.92 * flow_rgb[moving].astype(np.float32))
    output = np.clip(output, 0, 255).astype(np.uint8)
    output = draw_component_arrows(
        output, display_flow, moving_native, args)
    return output, flow_bev, valid_mask


def main():
    args = parse_args()
    payload = mmcv.load(str(args.ann_file))
    infos = payload['infos'] if isinstance(payload, dict) else payload
    lookup = {str(info['token']): info for info in infos}
    if args.token not in lookup:
        raise KeyError('Token {} is absent from {}.'.format(
            args.token, args.ann_file))
    current_info = lookup[args.token]
    previous_info = lookup[str(current_info['prev'])]
    semantics, target, flow = load_scale_state(
        current_info, previous_info, scale=1, args=args)

    # ``make_occ_color_map`` is BGR for OpenCV. Matplotlib/PIL display RGB.
    rgb_palette = make_occ_color_map(CLASS_NAMES)[:, ::-1]
    semantic_bev = render_occ_bev_rgb(
        semantics,
        EMPTY_IDX,
        rgb_palette,
        image_size_hw=(args.size, args.size))
    base = gray_context(semantic_bev)
    dynamic_bev = route_panel(base, target['dynamic'], DYNAMIC_COLOR)
    ptr_bev = combined_panel(base, target)
    flow_bev_image, projected_flow, valid_mask = render_flow_with_context(
        flow, semantics, base, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    flow_path = args.output_dir / 'roadocc_flow_gt_bev.png'
    dynamic_path = args.output_dir / 'roadocc_dynamic_gt_bev.png'
    ptr_path = args.output_dir / 'roadocc_ptr_gt_bev.png'
    pair_path = args.output_dir / 'roadocc_flow_ptr_gt_bev_pair.png'
    Image.fromarray(flow_bev_image).save(flow_path, compress_level=9)
    Image.fromarray(dynamic_bev).save(dynamic_path, compress_level=9)
    Image.fromarray(ptr_bev).save(ptr_path, compress_level=9)
    gutter = np.full((args.size, 20, 3), 255, dtype=np.uint8)
    Image.fromarray(
        np.concatenate([flow_bev_image, gutter, ptr_bev], axis=1)).save(
            pair_path, compress_level=9)

    speed = np.linalg.norm(projected_flow, axis=-1)
    moving_speed = speed[valid_mask & (speed > args.flow_epsilon)]
    dynamic_count = int(target['dynamic'].sum())
    route_counts = {
        route: int(target[route].sum())
        for route in ('persist', 'transport', 'refresh')
    }
    manifest = {
        'token': str(args.token),
        'ann_file': str(args.ann_file),
        'occ_path': str(current_info['occ_path']),
        'previous_occ_path': str(previous_info['occ_path']),
        'native_grid': list(semantics.shape),
        'voxel_size_xy_m': 0.4,
        'frame_dt_seconds': float(args.frame_dt),
        'flow_display': {
            'direction': 'HSV hue and arrow direction',
            'speed': 'HSV saturation and arrow length',
            'max_speed_mps': float(args.max_speed),
            'min_arrow_speed_mps': float(args.min_arrow_speed),
            'moving_bev_cells': int(len(moving_speed)),
            'mean_moving_speed_mps': (
                float(moving_speed.mean()) if len(moving_speed) else 0.0),
            'max_moving_speed_mps': (
                float(moving_speed.max()) if len(moving_speed) else 0.0),
        },
        'ptr_target': {
            'route_counts': route_counts,
            'route_percent': {
                key: 100.0 * value / max(dynamic_count, 1)
                for key, value in route_counts.items()
            },
        },
        'outputs': {
            'flow_bev': str(flow_path),
            'dynamic_bev': str(dynamic_path),
            'ptr_bev': str(ptr_path),
            'paired_bev': str(pair_path),
        },
    }
    manifest_path = args.output_dir / 'roadocc_flow_ptr_gt_bev.json'
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
