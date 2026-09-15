#!/usr/bin/env python3
"""Compose the canonical P/T/R GT at 1/1, 1/2, 1/4, and 1/8."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compose_ptr_gt_max_scale import (  # noqa: E402
    CLASS_NAMES, DYNAMIC_INDICES, EMPTY_IDX, POINT_CLOUD_RANGE, P_COLOR,
    R_COLOR, TEXT, T_COLOR, add_panel, build_target, combined_panel,
    gray_context, load_semantics, resolve, route_panel)
from mmdet3d.core.visualizer.occ_visualization import (  # noqa: E402
    make_occ_color_map, render_flow_bev, render_occ_bev_rgb)
from mmdet3d.datasets.utils import nuscenes_get_rt_matrix  # noqa: E402
from projects.STCRoadOcc.mmdet3d_plugin.datasets.pipelines.ptr_state_targets import \
    LoadPTRPreviousOccTarget  # noqa: E402,E501


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file',
        type=Path,
        default=ROOT / 'data/v2xreal_nuscenes/'
        'reference-v2xreal_infos_train_occ_enriched.pkl')
    parser.add_argument(
        '--flow-root',
        type=Path,
        default=ROOT / 'data/v2xreal_nuscenes/occflow_gts')
    parser.add_argument(
        '--token', default='aafb051d-8810-426c-ae1b-7e66cf5c52a2')
    parser.add_argument('--scales', type=int, nargs='+', default=(1, 2, 4, 8))
    parser.add_argument('--frame-dt', type=float, default=0.5)
    parser.add_argument('--flow-epsilon', type=float, default=1e-3)
    parser.add_argument('--correspondence-radius', type=int, default=1)
    parser.add_argument('--boundary-epsilon', type=float, default=1e-6)
    parser.add_argument('--panel-size', type=int, default=640)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'docs/paper/ICLR2027/figures/'
        'roadocc_ptr_gt_multiscale.png')
    return parser.parse_args()


def align_previous_boundary_safe(current_info, previous_info, current_gt,
                                 previous_raw, scale, epsilon):
    """Align history while tolerating round-off at normalized grid limits."""
    loader = LoadPTRPreviousOccTarget(
        point_cloud_range=POINT_CLOUD_RANGE,
        scale=int(scale),
        empty_idx=EMPTY_IDX,
        load_flow=False,
        boundary_epsilon=epsilon)
    results = dict(
        curr_to_prev_lidar_rt=torch.as_tensor(
            nuscenes_get_rt_matrix(current_info, previous_info, 'lidar',
                                   'lidar'),
            dtype=torch.float32),
        bda_mat=torch.eye(4))
    grid, _, _, _ = loader._previous_sampling_grid(results, current_gt.shape)
    in_bounds = ((grid >= -1.0 - float(epsilon)) &
                 (grid <= 1.0 + float(epsilon))).all(dim=-1)
    in_bounds = in_bounds.squeeze(0).permute(2, 1, 0)
    sampled = F.grid_sample(
        torch.as_tensor(previous_raw, dtype=torch.float32).permute(
            2, 1, 0).unsqueeze(0).unsqueeze(0),
        grid.clamp(-1.0, 1.0),
        mode='nearest',
        padding_mode='zeros',
        align_corners=True)
    sampled = sampled.squeeze(0).squeeze(0).permute(2, 1, 0)
    sampled = sampled.round().to(torch.uint8)
    sampled[~in_bounds] = 255
    return sampled.numpy().astype(np.int64, copy=False)


def load_scale_state(current_info, previous_info, scale, args):
    label_name = ('labels.npz' if int(scale) == 1 else
                  'labels_1_{}.npz'.format(scale))
    current_path = resolve(current_info['occ_path'])
    previous_path = resolve(previous_info['occ_path'])
    current_gt = load_semantics(current_path / label_name)
    previous_raw = load_semantics(previous_path / label_name)
    flow_path = (
        args.flow_root / current_path.parent.name / current_path.name /
        label_name)
    with np.load(flow_path, allow_pickle=False) as payload:
        flow = payload['flow'].astype(np.float32)
    previous_gt = align_previous_boundary_safe(
        current_info,
        previous_info,
        current_gt,
        previous_raw,
        scale=scale,
        epsilon=args.boundary_epsilon)
    target_args = SimpleNamespace(
        frame_dt=args.frame_dt,
        flow_epsilon=args.flow_epsilon,
        correspondence_radius=args.correspondence_radius)
    target = build_target(current_gt, previous_gt, flow, target_args)
    return current_gt, target, flow


def print_scale_report(scale, target):
    dynamic_count = int(target['dynamic'].sum())
    still_count = int(target['still'].sum())
    moving_count = int(target['moving'].sum())
    persist_count = int(target['persist'].sum())
    transport_count = int(target['transport'].sum())
    refresh_count = int(target['refresh'].sum())
    print('1/{} grid={} voxel={:.1f}m dynamic={} P={} T={} R={} '
          'P/support={:.2%} T/support={:.2%}'.format(
              scale, target['dynamic'].shape, target['voxel_size'],
              dynamic_count, persist_count, transport_count, refresh_count,
              persist_count / max(still_count, 1),
              transport_count / max(moving_count, 1)))


def main():
    args = parse_args()
    payload = mmcv.load(str(args.ann_file))
    infos = payload['infos'] if isinstance(payload, dict) else payload
    lookup = {str(info['token']): info for info in infos}
    current_info = lookup[args.token]
    previous_info = lookup[str(current_info['prev'])]
    rgb_palette = make_occ_color_map(CLASS_NAMES)[:, ::-1]

    panels = []
    for scale in args.scales:
        current_gt, target, flow = load_scale_state(current_info,
                                                    previous_info, scale,
                                                    args)
        print_scale_report(scale, target)
        semantic = render_occ_bev_rgb(
            current_gt,
            EMPTY_IDX,
            rgb_palette,
            image_size_hw=(args.panel_size, args.panel_size))
        base = gray_context(semantic)
        scale_name = '1/{}'.format(scale)
        flow_image = render_flow_bev(
            flow,
            current_gt,
            DYNAMIC_INDICES,
            image_size=(args.panel_size, args.panel_size),
            max_speed=8.0)
        if flow_image is None:
            flow_image = np.full_like(base, 255)
        panels.extend((
            (route_panel(base, target['persist'], P_COLOR),
             '{}  Persist'.format(scale_name), '#3674d9'),
            (route_panel(base, target['transport'], T_COLOR),
             '{}  Transport'.format(scale_name), '#e88e26'),
            (route_panel(base, target['refresh'], R_COLOR),
             '{}  Refresh'.format(scale_name), '#2ca05c'),
            (combined_panel(base, target),
             '{}  P/T/R target'.format(scale_name), TEXT),
            (flow_image, '{}  GT flow'.format(scale_name), TEXT),
        ))

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'text.color': TEXT,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    row_count = len(args.scales)
    fig = plt.figure(figsize=(16.0, 3.2 * row_count), facecolor='white')
    grid = fig.add_gridspec(
        row_count,
        5,
        left=0.004,
        right=0.996,
        bottom=0.005,
        top=0.995,
        wspace=0.012,
        hspace=0.012)
    for index, (image, title, color) in enumerate(panels):
        add_panel(
            fig.add_subplot(grid[index // 5, index % 5]), image, title, color)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor='white')
    fig.savefig(
        args.output.with_suffix('.pdf'), dpi=args.dpi, facecolor='white')
    plt.close(fig)
    print(args.output.resolve())
    print(args.output.with_suffix('.pdf').resolve())


if __name__ == '__main__':
    main()
