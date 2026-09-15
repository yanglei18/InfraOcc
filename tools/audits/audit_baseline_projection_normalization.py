#!/usr/bin/env python3
"""Visualize the two image-coordinate normalization conventions."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_audit(image_height, image_width):
    if image_height <= 1 or image_width <= 1:
        raise ValueError('image dimensions must both exceed one pixel')

    source_valid_pixels = (image_width - 1) * (image_height - 1)
    released_valid_width = min(image_width - 1, image_height - 1)
    released_valid_height = min(image_height - 1, image_width - 1)
    released_valid_pixels = released_valid_width * released_valid_height
    total_pixels = image_height * image_width
    return {
        'image_height': image_height,
        'image_width': image_width,
        'source_mode': 'x/image_width, y/image_height',
        'released_mode': 'x/image_height, y/image_width',
        'released_to_source_x_scale': image_width / image_height,
        'released_to_source_y_scale': image_height / image_width,
        'source_open_unit_pixel_fraction': source_valid_pixels / total_pixels,
        'released_open_unit_pixel_fraction': (
            released_valid_pixels / total_pixels),
        'released_continuous_image_area_fraction': min(
            1.0, image_height / image_width),
        'source_bottom_right_normalized': [
            (image_width - 1) / image_width,
            (image_height - 1) / image_height,
        ],
        'released_bottom_right_normalized': [
            (image_width - 1) / image_height,
            (image_height - 1) / image_width,
        ],
    }


def render_audit(payload, output_path):
    height = payload['image_height']
    width = payload['image_width']
    u_values = np.linspace(0.0, width - 1.0, 12)
    v_values = np.linspace(0.0, height - 1.0, 8)

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    modes = [
        ('Source V2X: x/W, y/H', width, height),
        ('Released rewrite: x/H, y/W', height, width),
    ]
    for axis, (title, x_denominator, y_denominator) in zip(axes[:2], modes):
        for u_value in u_values:
            axis.plot(
                np.full_like(v_values, u_value / x_denominator),
                v_values / y_denominator,
                color='#277da1',
                alpha=0.65,
                linewidth=1.0)
        for v_value in v_values:
            axis.plot(
                u_values / x_denominator,
                np.full_like(u_values, v_value / y_denominator),
                color='#f9844a',
                alpha=0.65,
                linewidth=1.0)
        axis.add_patch(
            plt.Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                fill=False,
                linestyle='--',
                linewidth=1.8,
                edgecolor='black',
                label='attention valid domain'))
        axis.set_xlim(-0.03, 1.9)
        axis.set_ylim(-0.03, 1.05)
        axis.set_aspect('equal')
        axis.set_xlabel('normalized image x')
        axis.set_ylabel('normalized image y')
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(loc='upper right', fontsize=8)

    x_scale = payload['released_to_source_x_scale']
    y_scale = payload['released_to_source_y_scale']
    area = payload['released_continuous_image_area_fraction']
    bars = axes[2].bar(
        ['x scale', 'y scale', 'valid\nimage area'],
        [x_scale, y_scale, area],
        color=['#577590', '#43aa8b', '#f9c74f'])
    axes[2].axhline(1.0, color='black', linestyle='--', linewidth=1.0)
    axes[2].set_ylim(0.0, max(2.0, x_scale + 0.15))
    axes[2].set_ylabel('released / source')
    axes[2].set_title('Geometric effect at 384x704')
    axes[2].grid(axis='y', alpha=0.2)
    for bar, value in zip(bars, (x_scale, y_scale, area)):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.03,
            f'{value:.3f}',
            ha='center',
            va='bottom')

    figure.suptitle(
        'Single-factor image-projection provenance audit',
        fontsize=14,
        fontweight='bold')
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-height', type=int, default=384)
    parser.add_argument('--image-width', type=int, default=704)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-figure', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_audit(args.image_height, args.image_width)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    render_audit(payload, args.output_figure)
    print(json.dumps(payload, sort_keys=True))


if __name__ == '__main__':
    main()
