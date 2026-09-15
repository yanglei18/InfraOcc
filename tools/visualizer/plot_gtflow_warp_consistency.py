#!/usr/bin/env python3
"""Render the speed-stratified GT-flow transport consistency dumbbell plot.

Example:
    python tools/visualizer/plot_gtflow_warp_consistency.py \
        --summary-json /tmp/roadocc_temporal_alignment/summary.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_STATS = [
    dict(name='Slow (<2 m/s)', count_m=7.87, zero=85.19, gt_flow=86.51),
    dict(name='Medium (2--5 m/s)', count_m=1.16, zero=71.08, gt_flow=96.65),
    dict(name='Fast ($\\geq$5 m/s)', count_m=2.80, zero=39.57, gt_flow=93.73),
]

ZERO_COLOR = '#D85A5A'
WARP_COLOR = '#258B66'
LINE_COLOR = '#A9B1B8'
TEXT_COLOR = '#1D2730'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot GT-flow warp consistency against speed bins.')
    parser.add_argument(
        '--summary-json',
        type=Path,
        default=None,
        help='Optional speed-stratified output from verify_occflow_alignment.py.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).resolve().parent,
        help='Directory for the rendered PNG and PDF.')
    parser.add_argument(
        '--output-name',
        type=str,
        default='gtflow_warp_consistency',
        help='Filename stem for the rendered PNG and PDF.')
    return parser.parse_args()


def load_stats(summary_json):
    if summary_json is None:
        return DEFAULT_STATS

    with summary_json.open('r', encoding='utf-8') as file:
        summary = json.load(file)
    rows = summary.get('speed_strata')
    if not rows:
        raise ValueError('The summary does not contain speed_strata statistics.')
    labels = {
        'Slow (<2)': 'Slow (<2 m/s)',
        'Medium (2--5)': 'Medium (2--5 m/s)',
        'Fast ($\\geq$5)': 'Fast (≥5 m/s)',
    }
    return [
        dict(
            name=labels.get(row['name'], row['name']),
            count_m=row['valid_flow_count'] / 1e6,
            zero=100.0 * row['same_class_ratio_zero'],
            gt_flow=100.0 * row['same_class_ratio_flow'],
        )
        for row in rows
    ]


def draw_plot(stats, output_dir, output_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{row['name']}\n($N={row['count_m']:.2f}$M)" for row in stats]
    zero = np.array([row['zero'] for row in stats])
    gt_flow = np.array([row['gt_flow'] for row in stats])
    gains = gt_flow - zero
    y_positions = np.arange(len(stats))[::-1]

    figure, axis = plt.subplots(figsize=(8.0, 3.86), dpi=240)
    figure.patch.set_facecolor('white')
    axis.set_facecolor('#FBFCFD')

    for y_position, zero_value, warp_value, gain in zip(
            y_positions, zero, gt_flow, gains):
        axis.plot(
            [zero_value, warp_value], [y_position, y_position],
            color=LINE_COLOR, linewidth=3.4, solid_capstyle='round', zorder=1)
        midpoint = (zero_value + warp_value) / 2.0
        axis.text(
            midpoint, y_position + 0.14, f'+{gain:.2f}%',
            ha='center', va='bottom', color=TEXT_COLOR, fontsize=14,
            fontweight='bold')

    axis.scatter(
        zero, y_positions, s=190, color=ZERO_COLOR, edgecolor='white',
        linewidth=1.8, zorder=3, label='Zero flow')
    axis.scatter(
        gt_flow, y_positions, s=190, color=WARP_COLOR, edgecolor='white',
        linewidth=1.8, zorder=3, label='GT-flow warp')

    for y_position, zero_value, warp_value in zip(y_positions, zero, gt_flow):
        close_pair = abs(warp_value - zero_value) < 5.0
        axis.text(
            zero_value - (0.35 if close_pair else 0.0), y_position - 0.14,
            f'{zero_value:.2f}', ha='right' if close_pair else 'center', va='top',
            color=ZERO_COLOR, fontsize=13,
            fontweight='bold')
        axis.text(
            warp_value + (0.35 if close_pair else 0.0), y_position - 0.14,
            f'{warp_value:.2f}', ha='left' if close_pair else 'center', va='top',
            color=WARP_COLOR, fontsize=13,
            fontweight='bold')

    axis.set_xlim(25, 104)
    axis.set_ylim(-0.48, len(stats) - 0.42)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels, fontsize=16, color=TEXT_COLOR)
    axis.set_xlabel('Adjacent-frame same-class consistency (%)', fontsize=16,
                    color=TEXT_COLOR, labelpad=5)
    axis.grid(axis='x', color='#DCE1E5', linewidth=0.8, alpha=0.85)
    axis.set_axisbelow(True)
    axis.tick_params(axis='x', labelsize=13, colors=TEXT_COLOR, length=0)
    axis.tick_params(axis='y', length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)

    legend = axis.legend(
        loc='upper left', bbox_to_anchor=(0.01, 0.99), ncol=2,
        frameon=True, framealpha=0.92, facecolor='white', edgecolor='none',
        fontsize=14, handletextpad=0.4, columnspacing=1.0)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    figure.suptitle(
        'Motion Compensation Restores Temporal Consistency', y=0.965, fontsize=18,
        color=TEXT_COLOR, fontweight='bold')
    figure.subplots_adjust(left=0.23, right=0.985, top=0.78, bottom=0.15)

    stem = output_dir / output_name
    figure.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
    figure.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(figure)
    return stem


def main():
    args = parse_args()
    stem = draw_plot(
        load_stats(args.summary_json), args.output_dir, args.output_name)
    print(f'Saved {stem.with_suffix(".png")} and {stem.with_suffix(".pdf")}')


if __name__ == '__main__':
    main()
