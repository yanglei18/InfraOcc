#!/usr/bin/env python3
"""Compose the native-scale P/T/R audit with consistent RGB semantics."""

import argparse
import sys
from pathlib import Path

# Keep visualization runs from leaving Python bytecode beside the script.
sys.dont_write_bytecode = True

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmdet3d.core.visualizer.occ_visualization import (  # noqa: E402
    _orient_xy_grid_to_image, make_occ_color_map, render_flow_bev,
    render_occ_bev_rgb)
from mmdet3d.datasets.utils import nuscenes_get_rt_matrix  # noqa: E402
from projects.STCRoadOcc.mmdet3d_plugin.datasets.pipelines.ptr_state_targets import \
    LoadPTRPreviousOccTarget  # noqa: E402,E501

CLASS_NAMES = ('others', 'barrier', 'bicycle', 'bus', 'car',
               'construction_vehicle', 'motorcycle', 'pedestrian',
               'traffic_cone', 'trailer', 'truck', 'driveable_surface',
               'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation',
               'free')
DYNAMIC_INDICES = (2, 3, 4, 6, 7, 10)
EMPTY_IDX = 17
POINT_CLOUD_RANGE = (-64.0, -64.0, -4.8, 64.0, 64.0, 1.6)
TOKEN = 'aafb051d-8810-426c-ae1b-7e66cf5c52a2'
P_COLOR = (54, 116, 217)
T_COLOR = (232, 142, 38)
R_COLOR = (44, 160, 92)
TEXT = '#18212b'
BORDER = '#cfd6de'


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
    parser.add_argument('--token', default=TOKEN)
    parser.add_argument('--frame-dt', type=float, default=0.5)
    parser.add_argument('--flow-epsilon', type=float, default=1e-3)
    parser.add_argument('--correspondence-radius', type=int, default=1)
    parser.add_argument('--panel-size', type=int, default=640)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument(
        '--source-composite',
        type=Path,
        default=ROOT / 'docs/paper/ICLR2027/figures/'
        'roadocc_ptr_gt_max_scale.png',
        help='Existing composite retaining the real Pred OCC and feature map.')
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'docs/paper/ICLR2027/figures/'
        'roadocc_ptr_gt_max_scale.png')
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_semantics(path):
    with np.load(path, allow_pickle=False) as payload:
        return payload['semantics'].astype(np.int64)


def align_previous(current_info,
                   previous_info,
                   current_gt,
                   previous_raw,
                   scale=1):
    loader = LoadPTRPreviousOccTarget(
        point_cloud_range=POINT_CLOUD_RANGE,
        scale=int(scale),
        empty_idx=EMPTY_IDX,
        load_flow=False)
    transform = dict(
        curr_to_prev_lidar_rt=torch.as_tensor(
            nuscenes_get_rt_matrix(current_info, previous_info, 'lidar',
                                   'lidar'),
            dtype=torch.float32),
        bda_mat=torch.eye(4))
    aligned, _ = loader._sample_previous(
        previous_raw, np.ones_like(previous_raw, dtype=bool), transform,
        current_gt.shape)
    return np.asarray(aligned, dtype=np.int64)


def dilate_xy(volume, radius):
    if radius <= 0:
        return np.asarray(volume, dtype=bool)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    output = np.zeros_like(volume, dtype=bool)
    for z_index in range(volume.shape[2]):
        output[...,
               z_index] = cv2.dilate(volume[..., z_index].astype(np.uint8),
                                     kernel).astype(bool)
    return output


def build_target(current_gt, previous_gt, flow, args):
    dynamic = np.isin(current_gt, DYNAMIC_INDICES)
    speed = np.linalg.norm(flow[..., :2], axis=-1)
    still = dynamic & (speed <= args.flow_epsilon)
    moving = dynamic & ~still
    rigid_support = np.zeros_like(dynamic)
    transport_support = np.zeros_like(dynamic)
    source_in_bounds = np.zeros_like(dynamic)
    voxel_size = ((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) /
                  current_gt.shape[0])

    for class_index in DYNAMIC_INDICES:
        current_class = current_gt == class_index
        previous_class = dilate_xy(previous_gt == class_index,
                                   args.correspondence_radius)
        rigid_support[current_class] = previous_class[current_class]
        coordinates = np.argwhere(current_class & moving)
        if not len(coordinates):
            continue
        velocity = flow[coordinates[:, 0], coordinates[:, 1],
                        coordinates[:, 2], :2]
        source_xy = np.rint(coordinates[:, :2] -
                            velocity * args.frame_dt / voxel_size).astype(
                                np.int64)
        valid = ((source_xy[:, 0] >= 0) &
                 (source_xy[:, 0] < current_gt.shape[0]) &
                 (source_xy[:, 1] >= 0) &
                 (source_xy[:, 1] < current_gt.shape[1]))
        coordinates = coordinates[valid]
        source_xy = source_xy[valid]
        source_in_bounds[coordinates[:, 0], coordinates[:, 1],
                         coordinates[:, 2]] = True
        supported = previous_class[source_xy[:, 0], source_xy[:, 1],
                                   coordinates[:, 2]]
        coordinates = coordinates[supported]
        transport_support[coordinates[:, 0], coordinates[:, 1],
                          coordinates[:, 2]] = True

    persist = still & rigid_support
    transport = moving & transport_support
    refresh = dynamic & ~(persist | transport)
    assert np.array_equal(dynamic, persist | transport | refresh)
    return dict(
        dynamic=dynamic,
        still=still,
        moving=moving,
        rigid_support=rigid_support,
        transport_support=transport_support,
        source_in_bounds=source_in_bounds,
        persist=persist,
        transport=transport,
        refresh=refresh,
        voxel_size=voxel_size)


def load_model_panels(path, image_size):
    """Preserve the real model panels after their raw cache was removed."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] == (1905, 3840):
        pred = image[68:882, 1052:1866]
        feature = image[68:882, 2944:3758]
    elif image.shape[:2] == (1920, 3840):
        # Tight 2x4 layout. The enlarged common title box fully covers the
        # smaller title already embedded in these preserved model panels.
        pred = image[15:956, 973:1911]
        feature = image[15:956, 2884:3822]
        pred = erase_embedded_title_text(pred)
        feature = erase_embedded_title_text(feature)
    else:
        raise ValueError('Unexpected source-composite shape: {}'.format(
            image.shape))
    pred = cv2.resize(
        pred, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    feature = cv2.resize(
        feature, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    return pred, feature


def erase_embedded_title_text(panel):
    """Remove only bright title glyphs retained in a composed model panel."""
    panel = panel.copy()
    region_height = max(int(round(panel.shape[0] * 0.12)), 1)
    region_width = max(int(round(panel.shape[1] * 0.82)), 1)
    region = panel[:region_height, :region_width]
    channel_range = np.ptp(region.astype(np.int16), axis=-1)
    # Repeated composition attenuates old white glyphs to mid-gray, so retain
    # a low achromatic threshold inside the known title-box footprint.
    text_mask = ((region.min(axis=-1) >= 50) & (channel_range <= 42))
    text_mask = cv2.dilate(
        text_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8))
    mask = np.zeros(panel.shape[:2], dtype=np.uint8)
    mask[:region_height, :region_width] = text_mask
    return cv2.inpaint(panel, mask, 3, cv2.INPAINT_TELEA)


def gray_context(image):
    image = image.astype(np.float32)
    gray = (0.299 * image[..., 0] + 0.587 * image[..., 1] +
            0.114 * image[..., 2])
    gray = 247.0 - 0.22 * (255.0 - gray)
    return np.repeat(
        np.clip(gray, 190, 250)[..., None], 3, axis=-1).astype(np.uint8)


def route_panel(base, mask, color):
    mask = _orient_xy_grid_to_image(mask.any(axis=2).astype(np.uint8))
    mask = cv2.resize(
        mask, (base.shape[1], base.shape[0]),
        interpolation=cv2.INTER_NEAREST).astype(bool)
    output = base.astype(np.float32).copy()
    output[mask] = 0.04 * output[mask] + 0.96 * np.asarray(color)
    return np.clip(output, 0, 255).astype(np.uint8)


def combined_panel(base, target):
    output = base.copy()
    for key, color in (('persist', P_COLOR), ('transport', T_COLOR),
                       ('refresh', R_COLOR)):
        output = route_panel(output, target[key], color)
    return output


def add_panel(ax, image, title, color=TEXT):
    ax.imshow(image, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_edgecolor(BORDER)
    title_box = FancyBboxPatch((0.012, 0.884),
                               0.80,
                               0.104,
                               boxstyle='round,pad=0.0,rounding_size=0.018',
                               transform=ax.transAxes,
                               facecolor=(0.02, 0.025, 0.03, 0.68),
                               edgecolor='none',
                               linewidth=0.0,
                               clip_on=True,
                               zorder=4)
    ax.add_patch(title_box)
    ax.text(
        0.030,
        0.936,
        title,
        transform=ax.transAxes,
        ha='left',
        va='center',
        fontsize=13.5,
        fontweight='bold',
        color='white',
        zorder=5)


def report(target):
    dynamic = int(target['dynamic'].sum())
    still = int(target['still'].sum())
    moving = int(target['moving'].sum())
    persist = int(target['persist'].sum())
    transport = int(target['transport'].sum())
    refresh = int(target['refresh'].sum())
    print('native_grid={} voxel_size_xy={:.3f}m'.format(
        target['dynamic'].shape, target['voxel_size']))
    print('dynamic={} P={} T={} R={}'.format(dynamic, persist, transport,
                                             refresh))
    print('P support: {}/{} ({:.2%})'.format(persist, still,
                                             persist / max(still, 1)))
    print('T support: {}/{} ({:.2%})'.format(transport, moving,
                                             transport / max(moving, 1)))


def main():
    args = parse_args()
    pred_image, feature_image = load_model_panels(args.source_composite,
                                                  args.panel_size)
    payload = mmcv.load(str(args.ann_file))
    infos = payload['infos'] if isinstance(payload, dict) else payload
    lookup = {str(info['token']): info for info in infos}
    current_info = lookup[args.token]
    previous_info = lookup[str(current_info['prev'])]
    current_path = resolve(current_info['occ_path'])
    previous_path = resolve(previous_info['occ_path'])
    current_gt = load_semantics(current_path / 'labels.npz')
    previous_raw = load_semantics(previous_path / 'labels.npz')
    flow_path = (
        args.flow_root / current_path.parent.name / current_path.name /
        'labels.npz')
    with np.load(flow_path, allow_pickle=False) as payload:
        flow = payload['flow'].astype(np.float32)
    previous_gt = align_previous(current_info, previous_info, current_gt,
                                 previous_raw)
    target = build_target(current_gt, previous_gt, flow, args)
    report(target)

    # make_occ_color_map is BGR for the OpenCV training visualizer. Matplotlib
    # requires display RGB; this swap is the palette bug fixed by this script.
    rgb_palette = make_occ_color_map(CLASS_NAMES)[:, ::-1]
    gt_image = render_occ_bev_rgb(
        current_gt,
        EMPTY_IDX,
        rgb_palette,
        image_size_hw=(args.panel_size, args.panel_size))
    flow_image = render_flow_bev(
        flow,
        current_gt,
        DYNAMIC_INDICES,
        image_size=(args.panel_size, args.panel_size),
        max_speed=8.0)
    base = gray_context(gt_image)
    panels = (
        (gt_image, '(a) GT Occupancy', TEXT),
        (pred_image, '(b) Pred Occupancy', TEXT),
        (flow_image, '(c) GT occupancy flow', TEXT),
        (feature_image, '(d) Fused BEV feature', TEXT),
        (route_panel(base, target['persist'],
                     P_COLOR), '(e) Persist', '#3674d9'),
        (route_panel(base, target['transport'],
                     T_COLOR), '(f) Transport', '#e88e26'),
        (route_panel(base, target['refresh'],
                     R_COLOR), '(g) Refresh', '#2ca05c'),
        (combined_panel(base, target), '(h) P/T/R ground truth', TEXT),
    )

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'text.color': TEXT,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    fig = plt.figure(figsize=(12.8, 6.4), facecolor='white')
    grid = fig.add_gridspec(
        2,
        4,
        left=0.004,
        right=0.996,
        bottom=0.008,
        top=0.992,
        wspace=0.012,
        hspace=0.012)
    for index, (image, title, color) in enumerate(panels):
        add_panel(
            fig.add_subplot(grid[index // 4, index % 4]), image, title, color)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor='white')
    fig.savefig(
        args.output.with_suffix('.pdf'), dpi=args.dpi, facecolor='white')
    plt.close(fig)
    print(args.output.resolve())
    print(args.output.with_suffix('.pdf').resolve())


if __name__ == '__main__':
    main()
