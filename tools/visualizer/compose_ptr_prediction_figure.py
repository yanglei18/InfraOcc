#!/usr/bin/env python3
"""Compose the paper P/T/R prediction figure from one training token."""

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
DEFAULT_VIS_ID = '023_000158'
TEXT = '#18212b'
BORDER = '#cfd6de'
P_COLOR = (35, 130, 220)
T_COLOR = (235, 155, 35)
R_COLOR = (65, 195, 55)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vis-id', default=DEFAULT_VIS_ID)
    parser.add_argument(
        '--input-root',
        type=Path,
        default=ROOT / 'work_dirs/stcroadocc_c_4x4_24e/figures_path/train')
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'docs/paper/ICLR2027/Fig5-ptrcontrol.png')
    parser.add_argument('--panel-size', type=int, default=720)
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def square_crop(image, y0, y1, x0, x1):
    panel = image[y0:y1, x0:x1]
    height, width = panel.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return panel[top:top + side, left:left + side]


def resize_panel(panel, size, continuous=False):
    interpolation = cv2.INTER_CUBIC if continuous else cv2.INTER_NEAREST
    return cv2.resize(panel, (size, size), interpolation=interpolation)


def isolate_route(state_panel, route_color):
    """Render one discrete route from an argmax P/T/R state on white."""
    output = 255 * np.ones_like(state_panel)
    mask = np.all(state_panel == np.asarray(route_color), axis=-1)
    output[mask] = route_color
    return output


def white_state_panel(state_panel):
    """Keep the categorical P/T/R colors while replacing gray context."""
    output = 255 * np.ones_like(state_panel)
    for route_color in (P_COLOR, T_COLOR, R_COLOR):
        mask = np.all(state_panel == np.asarray(route_color), axis=-1)
        output[mask] = route_color
    return output


def load_panels(args):
    vis_id = args.vis_id
    occ_path = (args.input_root / 'occ_bev' /
                f'{vis_id}_occ_bev.png')
    flow_path = (args.input_root / 'occ_flow_bev' /
                 f'{vis_id}_occ_flow_bev.png')
    feature_path = (args.input_root / 'modality_feat_grid' /
                    f'{vis_id}_modality_feat_grid.png')
    ptr_path = (args.input_root / 'canonical_ptr_bev' /
                f'{vis_id}_canonical_ptr_bev.png')

    occ = read_rgb(occ_path)
    flow = read_rgb(flow_path)
    feature = read_rgb(feature_path)
    ptr = read_rgb(ptr_path)
    if occ.shape[:2] != (688, 1200):
        raise ValueError(f'Unexpected occupancy image shape: {occ.shape}')
    if flow.shape[:2] != (1048, 1920):
        raise ValueError(f'Unexpected flow image shape: {flow.shape}')
    if feature.shape[:2] != (728, 640):
        raise ValueError(f'Unexpected feature image shape: {feature.shape}')
    if ptr.shape[:2] != (1180, 1320):
        raise ValueError(f'Unexpected P/T/R image shape: {ptr.shape}')

    # Training composites contain an 88-pixel metadata banner. Their panel
    # titles occupy the next 28 pixels, which are removed before recomposition.
    gt_occ = square_crop(occ, 116, 688, 600, 1200)
    pred_occ = square_crop(occ, 116, 688, 0, 600)
    gt_flow = square_crop(flow, 596, 1048, 480, 960)
    fused_feature = square_crop(feature, 436, 728, 0, 320)

    # The native 1/1 row starts at y=916; its scale banner and embedded titles
    # end at y=988. Split the discrete predicted state rather than displaying
    # the three probability heatmaps in columns 0--2.
    pred_state = square_crop(ptr, 988, 1180, 660, 880)
    gt_state = square_crop(ptr, 988, 1180, 880, 1100)
    ptr_panels = [
        isolate_route(pred_state, P_COLOR),
        isolate_route(pred_state, T_COLOR),
        isolate_route(pred_state, R_COLOR),
        white_state_panel(gt_state),
    ]
    panels = [gt_occ, pred_occ, gt_flow, fused_feature] + ptr_panels
    return [
        resize_panel(panel, args.panel_size, continuous=(index in (0, 1, 3)))
        for index, panel in enumerate(panels)
    ]


def add_panel(ax, image, title):
    ax.imshow(image, interpolation='nearest')
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
        facecolor=(0.02, 0.025, 0.03, 0.68),
        edgecolor='none',
        linewidth=0.0,
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
        fontsize=13.8,
        fontweight='bold',
        color='white',
        zorder=5)


def main():
    args = parse_args()
    panels = load_panels(args)
    titles = (
        '(a) GT Occupancy',
        '(b) Pred Occupancy',
        '(c) GT occupancy flow',
        '(d) Fused BEV feature',
        '(e) Pred Persist',
        '(f) Pred Transport',
        '(g) Pred Refresh',
        '(h) GT P/T/R',
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
    for index in range(8):
        add_panel(fig.add_subplot(grid[index // 4, index % 4]), panels[index],
                  titles[index])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor='white')
    plt.close(fig)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
