#!/usr/bin/env python3
"""Reproduce the class-wise IoU gains used in the RoadOcc supplement."""

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / 'docs/paper/ICLR2027/supplementary/Fig6-classwise-gains.pdf')

DYNAMIC_LABELS = ('Bike', 'Bus', 'Car', 'Motorcycle', 'Pedestrian', 'Truck')
DYNAMIC_GAINS = (10.47, 6.68, 4.43, 2.78, -0.19, 4.10)
STATIC_LABELS = (
    'Other', 'Barrier', 'Cone', 'Driveable', 'Sidewalk', 'Terrain',
    'Manmade', 'Vegetation', 'Free')
STATIC_GAINS = (18.82, 1.12, 4.43, 1.97, 3.40, 1.54, 1.06, 1.62, 0.17)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--preview', type=Path, default=None)
    return parser.parse_args()


def configure_style():
    regular = '/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf'
    bold = '/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf'
    font_manager.fontManager.addfont(regular)
    font_manager.fontManager.addfont(bold)
    mpl.rcParams.update({
        'font.family': 'Liberation Serif',
        'font.size': 8.2,
        'axes.titlesize': 8.9,
        'axes.titleweight': 'bold',
        'axes.labelsize': 8.2,
        'xtick.labelsize': 7.6,
        'ytick.labelsize': 7.2,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 0.62,
    })


def draw_panel(ax, labels, gains, color, title):
    positions = list(range(len(labels)))
    ax.barh(positions, gains, height=0.55, color=color, edgecolor='none')
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.48, 21.25)
    ax.set_xticks((0, 5, 10, 15, 20))
    ax.grid(axis='x', color='#cfcfcf', linewidth=0.42)
    ax.set_axisbelow(True)
    ax.axvline(0, color='#333333', linewidth=0.62)
    ax.set_title(title, pad=1.8)
    ax.tick_params(axis='x', pad=0.8, length=2.2, width=0.5)
    ax.tick_params(axis='y', pad=-0.2, length=1.8, width=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for position, gain in zip(positions, gains):
        is_positive = gain >= 0
        ax.text(
            gain + 0.26 if is_positive else 0.18,
            position,
            f'{gain:+.2f}',
            ha='left',
            va='center',
            color=color if is_positive else '#d64545',
            fontsize=7.35,
            fontweight='bold')


def main():
    args = parse_args()
    configure_style()
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(2.96, 3.05),
        gridspec_kw={'height_ratios': (6, 9), 'hspace': 0.20})
    dynamic_axis, static_axis = axes

    draw_panel(
        dynamic_axis, DYNAMIC_LABELS, DYNAMIC_GAINS, '#347fd1',
        '(a) Dynamic classes')
    draw_panel(
        static_axis, STATIC_LABELS, STATIC_GAINS, '#2dad5c',
        '(b) Static classes')
    dynamic_axis.set_xlabel('')
    static_axis.set_xlabel('IoU gain over STCOcc (points)', labelpad=1.4)
    figure.subplots_adjust(left=0.19, right=0.995, top=0.97, bottom=0.105)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output, bbox_inches='tight', pad_inches=0.006,
        facecolor='white')
    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            args.preview, dpi=220, bbox_inches='tight', pad_inches=0.006,
            facecolor='white')
    plt.close(figure)
    print(args.output)


if __name__ == '__main__':
    main()
