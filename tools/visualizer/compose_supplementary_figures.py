#!/usr/bin/env python3
"""Compose the two RoadOcc supplementary qualitative figures."""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
SUPP_DIR = ROOT / 'docs/paper/ICLR2027/supplementary'
CRTFUSION_ROOT = (
    ROOT / 'work_dirs/crtfusion_c_4x4_24e_bs1/figures_path/test/occ_bev')
ROADOCC_TEST_ROOT = (
    ROOT / 'work_dirs/stcroadocc_c_4x4_24e/figures_path/test')
ROADOCC_TRAIN_ROOT = (
    ROOT / 'work_dirs/stcroadocc_c_4x4_24e/figures_path/train')

P_COLOR = (35, 130, 220)
T_COLOR = (235, 155, 35)
R_COLOR = (65, 195, 55)
BORDER = '#cfd6de'
BOX_COLOR = (0.02, 0.025, 0.03, 0.68)
ZOOM_COLOR = (38, 176, 245)

# Stored validation composites that share exactly the same GT panel.  The
# RoadOcc images were written by the code that produced their checkpoints;
# this avoids reloading an older checkpoint into a later controller layout.
COMPARISON_PAIRS = (
    ('022_000023_occ_bev.png', '020_000023_occ_bev.png'),
    ('024_000026_occ_bev.png', '022_000026_occ_bev.png'),
    ('020_000020_occ_bev.png', '018_000020_occ_bev.png'),
    ('018_000018_occ_bev.png', '016_000018_occ_bev.png'),
)
PTR_VIS_IDS = (
    '023_000158',
    '017_000113',
    '014_000092',
)

# OpenCV BGR palette used by the occupancy visualizer.
PALETTE_BGR = np.asarray([
    (0, 0, 0), (50, 120, 255), (203, 192, 255), (0, 255, 255),
    (245, 150, 0), (255, 255, 0), (0, 127, 255), (0, 0, 255),
    (150, 240, 255), (0, 60, 135), (240, 32, 160), (255, 0, 255),
    (137, 137, 139), (75, 0, 75), (80, 240, 150), (250, 230, 230),
    (0, 175, 0), (255, 255, 255),
], dtype=np.uint8)
DYNAMIC_INDICES = np.asarray((2, 3, 4, 6, 7, 10), dtype=np.int16)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=SUPP_DIR)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--panel-size', type=int, default=720)
    return parser.parse_args()


def read_bgr(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def center_square(image):
    height, width = image.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    return image[y0:y0 + side, x0:x0 + side]


def occ_panels(path):
    image = read_bgr(path)
    if image.shape[:2] != (688, 1200):
        raise ValueError(f'Unexpected occupancy composite: {path} {image.shape}')
    pred = center_square(image[116:688, 0:600])
    gt = center_square(image[116:688, 600:1200])
    return pred, gt


def labels_from_bgr(image):
    labels = np.full(image.shape[:2], -1, dtype=np.int16)
    for index, color in enumerate(PALETTE_BGR):
        labels[np.all(image == color, axis=-1)] = index
    return labels


def choose_recovery_roi(gt, crt, road):
    """Find one actor region recovered by RoadOcc but missed by CRTFusion."""
    gt_labels = labels_from_bgr(gt)
    crt_labels = labels_from_bgr(crt)
    road_labels = labels_from_bgr(road)
    dynamic = np.isin(gt_labels, DYNAMIC_INDICES)
    recovered = dynamic & (road_labels == gt_labels) & (crt_labels != gt_labels)

    grouped = cv2.dilate(
        recovered.astype(np.uint8), np.ones((13, 13), dtype=np.uint8))
    count, components, _, _ = cv2.connectedComponentsWithStats(grouped)
    if count > 1:
        best = max(
            range(1, count),
            key=lambda index: int(recovered[components == index].sum()))
        selected = recovered & (components == best)
    else:
        selected = recovered
    ys, xs = np.where(selected)
    if len(xs) == 0:
        ys, xs = np.where(dynamic)
    if len(xs) == 0:
        return 0, 0, gt.shape[1], gt.shape[0]

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    side = min(max(max(x1 - x0, y1 - y0) + 34, 74), 124)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    x0 = max(0, min(gt.shape[1] - side, cx - side // 2))
    y0 = max(0, min(gt.shape[0] - side, cy - side // 2))
    return x0, y0, x0 + side, y0 + side


def dashed_rectangle(image, bounds, width=2, dash=7):
    output = image.copy()
    x0, y0, x1, y1 = bounds
    x1 -= 1
    y1 -= 1
    for start in range(x0, x1 + 1, 2 * dash):
        cv2.line(output, (start, y0), (min(start + dash, x1), y0),
                 ZOOM_COLOR, width, cv2.LINE_AA)
        cv2.line(output, (start, y1), (min(start + dash, x1), y1),
                 ZOOM_COLOR, width, cv2.LINE_AA)
    for start in range(y0, y1 + 1, 2 * dash):
        cv2.line(output, (x0, start), (x0, min(start + dash, y1)),
                 ZOOM_COLOR, width, cv2.LINE_AA)
        cv2.line(output, (x1, start), (x1, min(start + dash, y1)),
                 ZOOM_COLOR, width, cv2.LINE_AA)
    return output


def add_recovery_inset(image, bounds):
    """Embed the selected recovery region at the lower-right of a BEV panel."""
    x0, y0, x1, y1 = bounds
    crop = image[y0:y1, x0:x1]
    output = dashed_rectangle(image, bounds)
    side = int(min(image.shape[:2]) * 0.34)
    crop = cv2.resize(crop, (side, side), interpolation=cv2.INTER_NEAREST)
    margin = 10
    outer = 5
    inset_x = image.shape[1] - side - margin
    inset_y = image.shape[0] - side - margin
    cv2.rectangle(
        output, (inset_x - outer, inset_y - outer),
        (inset_x + side + outer - 1, inset_y + side + outer - 1),
        (255, 255, 255), -1)
    cv2.rectangle(
        output, (inset_x - 2, inset_y - 2),
        (inset_x + side + 1, inset_y + side + 1), ZOOM_COLOR, 2)
    output[inset_y:inset_y + side, inset_x:inset_x + side] = crop
    return output


def add_panel(ax, image_bgr, title, scene_label=None, font_size=13.5):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(image_rgb, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_edgecolor(BORDER)
    title_box = FancyBboxPatch(
        (0.012, 0.872),
        0.944,
        0.116,
        boxstyle='round,pad=0.0,rounding_size=0.020',
        transform=ax.transAxes,
        facecolor=BOX_COLOR,
        edgecolor='none',
        clip_on=True,
        zorder=4)
    ax.add_patch(title_box)
    ax.text(
        0.032,
        0.930,
        title,
        transform=ax.transAxes,
        ha='left',
        va='center',
        fontsize=font_size,
        fontweight='bold',
        color='white',
        zorder=5)
    if scene_label:
        ax.text(
            0.028,
            0.045,
            scene_label,
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=11.5,
            fontweight='bold',
            color='white',
            bbox=dict(
                boxstyle='round,pad=0.28',
                facecolor=(0.02, 0.025, 0.03, 0.68),
                edgecolor='none'),
            zorder=5)


def compose_comparison(args):
    samples = []
    for crt_name, road_name in COMPARISON_PAIRS:
        crt, crt_gt = occ_panels(CRTFUSION_ROOT / crt_name)
        road, road_gt = occ_panels(
            ROADOCC_TEST_ROOT / 'occ_bev' / road_name)
        if not np.array_equal(crt_gt, road_gt):
            raise ValueError(f'GT mismatch for {crt_name} and {road_name}')
        bounds = choose_recovery_roi(crt_gt, crt, road)
        samples.append(tuple(
            add_recovery_inset(panel, bounds)
            for panel in (crt_gt, crt, road)))

    titles = ('GT Occupancy', 'CRTFusion', 'RoadOcc')
    fig = plt.figure(figsize=(15.6, 5.2), facecolor='white')
    grid = fig.add_gridspec(
        2, 6, left=0.003, right=0.997, bottom=0.008, top=0.992,
        wspace=0.010, hspace=0.012)
    for sample_index, panels in enumerate(samples):
        row_index = sample_index // 2
        sample_column = (sample_index % 2) * 3
        for panel_index, panel in enumerate(panels):
            column = sample_column + panel_index
            panel = cv2.resize(
                panel, (args.panel_size, args.panel_size),
                interpolation=cv2.INTER_NEAREST)
            letter_index = row_index * 6 + column
            add_panel(
                fig.add_subplot(grid[row_index, column]), panel,
                f'({chr(97 + letter_index)}) {titles[panel_index]}',
                scene_label=(f'Scene {sample_index + 1}'
                             if panel_index == 0 else None),
                font_size=11.2)
    output = args.output_dir / 'supp_crtfusion_comparison.png'
    fig.savefig(output, dpi=args.dpi, facecolor='white')
    plt.close(fig)
    return output


def categorical_state(panel):
    output = np.full_like(panel, 255)
    # Source image is BGR: blue=P, orange=T, green=R.
    for rgb in (P_COLOR, T_COLOR, R_COLOR):
        bgr = np.asarray(rgb[::-1], dtype=np.uint8)
        mask = np.all(panel == bgr, axis=-1)
        output[mask] = bgr
    return output


def flow_on_black(panel):
    """Replace only the exact white flow canvas while preserving flow colors."""
    output = panel.copy()
    white_background = np.all(output == 255, axis=-1)
    output[white_background] = 0
    return output


def ptr_sample_panels(vis_id):
    occ_path = ROADOCC_TRAIN_ROOT / 'occ_bev' / f'{vis_id}_occ_bev.png'
    flow_path = (ROADOCC_TRAIN_ROOT / 'occ_flow_bev' /
                 f'{vis_id}_occ_flow_bev.png')
    feature_path = (ROADOCC_TRAIN_ROOT / 'modality_feat_grid' /
                    f'{vis_id}_modality_feat_grid.png')
    ptr_path = (ROADOCC_TRAIN_ROOT / 'canonical_ptr_bev' /
                f'{vis_id}_canonical_ptr_bev.png')

    pred_occ, gt_occ = occ_panels(occ_path)
    flow = read_bgr(flow_path)
    feature = read_bgr(feature_path)
    ptr = read_bgr(ptr_path)
    if flow.shape[:2] != (1048, 1920):
        raise ValueError(f'Unexpected occupancy-flow composite: {flow.shape}')
    if feature.shape[:2] != (728, 640):
        raise ValueError(f'Unexpected feature composite: {feature.shape}')
    if ptr.shape[:2] != (1180, 1320):
        raise ValueError(f'Unexpected canonical PTR composite: {ptr.shape}')

    gt_flow = center_square(flow[596:1048, 480:960])
    fused_feature = center_square(feature[436:728, 0:320])
    pred_state = center_square(ptr[988:1180, 660:880])
    gt_state = center_square(ptr[988:1180, 880:1100])
    return (
        gt_occ,
        pred_occ,
        flow_on_black(gt_flow),
        fused_feature,
        categorical_state(pred_state),
        categorical_state(gt_state),
    )


def compose_ptr(args):
    samples = [ptr_sample_panels(vis_id) for vis_id in PTR_VIS_IDS]
    titles = (
        'GT Occupancy',
        'Pred Occupancy',
        'GT occupancy flow',
        'Fused BEV feature',
        'Pred P/T/R',
        'GT P/T/R',
    )
    fig = plt.figure(
        figsize=(15.6, 2.6 * len(samples)), facecolor='white')
    grid = fig.add_gridspec(
        len(samples), 6, left=0.003, right=0.997, bottom=0.005, top=0.995,
        wspace=0.010, hspace=0.012)
    for sample_index, panels in enumerate(samples):
        for panel_index, panel in enumerate(panels):
            interpolation = (cv2.INTER_CUBIC
                             if panel_index == 3 else cv2.INTER_NEAREST)
            panel = cv2.resize(
                panel, (args.panel_size, args.panel_size),
                interpolation=interpolation)
            letter_index = sample_index * 6 + panel_index
            add_panel(
                fig.add_subplot(grid[sample_index, panel_index]), panel,
                f'({chr(97 + letter_index)}) {titles[panel_index]}',
                scene_label=(f'Sample {sample_index + 1}'
                             if panel_index == 0 else None),
                font_size=10.2)
    output = args.output_dir / 'supp_ptr_native.png'
    fig.savefig(output, dpi=args.dpi, facecolor='white')
    plt.close(fig)
    return output


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    for output in (compose_comparison(args), compose_ptr(args)):
        print(output.resolve())


if __name__ == '__main__':
    main()
