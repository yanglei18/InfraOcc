#!/usr/bin/env python3
"""Render the editable sensor-setup illustration used by the TPAMI draft.

The drawing deliberately encodes the calibrated geometry rather than relying on
photographic thumbnails.  It can therefore be regenerated at the final paper
resolution while keeping its claims aligned with the dataset specification:
two infrastructure units, two cameras and one LiDAR per unit, and GPS time
synchronization in a fixed roadside coordinate system.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'tools/visualizer/tpami_reproduction/fig02-sensor-setup-schematic.png'

INK = '#1f2937'
MUTED = '#64748b'
ROAD = '#e8edf2'
LANE = '#ffffff'
BLUE = '#2563eb'
CYAN = '#0891b2'
AMBER = '#f59e0b'
GREEN = '#15803d'


def rounded_box(ax, xy, width, height, facecolor, edgecolor='none', radius=0.12,
                linewidth=1.0, zorder=1):
    box = FancyBboxPatch(
        xy, width, height, boxstyle=f'round,pad=0.02,rounding_size={radius}',
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)
    ax.add_patch(box)
    return box


def draw_camera(ax, x, y, angle_deg, color):
    """Draw one fixed camera and its image-plane field of view."""
    angle = np.deg2rad(angle_deg)
    direction = np.array([np.cos(angle), np.sin(angle)])
    lateral = np.array([-direction[1], direction[0]])
    origin = np.array([x, y])
    tip = origin + 2.25 * direction
    wedge = Polygon(
        [origin + 0.14 * lateral, tip + 0.90 * lateral,
         tip - 0.90 * lateral, origin - 0.14 * lateral],
        closed=True, facecolor=color, edgecolor='none', alpha=0.12, zorder=2)
    ax.add_patch(wedge)
    ax.add_patch(Rectangle((x - 0.22, y - 0.13), 0.44, 0.26, angle=angle_deg,
                           facecolor=INK, edgecolor='white', linewidth=0.7, zorder=5))
    lens = origin + 0.23 * direction
    ax.add_patch(Circle(lens, 0.065, facecolor=color, edgecolor='white', linewidth=0.5,
                        zorder=6))


def draw_lidar(ax, x, y):
    """Draw an OS1-like LiDAR head, scanning rings, and a mast."""
    ax.plot([x, x], [y - 1.0, y - 0.08], color=INK, lw=3.0, solid_capstyle='round', zorder=5)
    ax.plot([x - 0.28, x + 0.28], [y - 1.0, y - 1.0], color=INK, lw=3.0,
            solid_capstyle='round', zorder=5)
    for radius, alpha in ((0.32, 0.40), (0.58, 0.22), (0.85, 0.10)):
        ax.add_patch(Circle((x, y), radius, fill=False, edgecolor=AMBER,
                            linewidth=1.1, alpha=alpha, zorder=3))
    ax.add_patch(Circle((x, y), 0.16, facecolor=AMBER, edgecolor='white', linewidth=0.8,
                        zorder=7))
    ax.text(x, y + 0.40, 'LiDAR', fontsize=7.5, color=INK, ha='center', va='bottom', zorder=7)


def draw_unit(ax, x, y, unit_name, camera_angles):
    draw_lidar(ax, x, y)
    for index, angle in enumerate(camera_angles, start=1):
        draw_camera(ax, x, y - 0.05, angle, BLUE if index == 1 else CYAN)
    label = rounded_box(ax, (x - 0.92, y - 1.67), 1.84, 0.43, '#ffffff', '#cbd5e1',
                        radius=0.10, linewidth=0.8, zorder=8)
    label.set_alpha(0.96)
    ax.text(x, y - 1.45, unit_name, fontsize=9, weight='bold', color=INK,
            ha='center', va='center', zorder=9)
    ax.text(x, y - 1.74, '2 cameras + 1 LiDAR', fontsize=7.2, color=MUTED,
            ha='center', va='center', zorder=9)


def draw_intersection(ax):
    ax.add_patch(Rectangle((1.9, 0.0), 6.2, 8.0, facecolor='#f8fafc', edgecolor='#dbe3eb',
                           linewidth=0.8, zorder=0))
    ax.add_patch(Rectangle((1.9, 2.75), 6.2, 2.5, facecolor=ROAD, edgecolor='none', zorder=0))
    ax.add_patch(Rectangle((4.75, 0.0), 2.5, 8.0, facecolor=ROAD, edgecolor='none', zorder=0))
    for y in (3.33, 4.67):
        ax.plot([2.05, 7.95], [y, y], color=LANE, lw=1.2, ls=(0, (5, 5)), zorder=1)
    for x in (5.33, 6.67):
        ax.plot([x, x], [0.15, 7.85], color=LANE, lw=1.2, ls=(0, (5, 5)), zorder=1)
    for start, horizontal in (((4.35, 2.76), True), ((4.35, 5.02), True),
                              ((1.92, 3.70), False), ((7.70, 3.70), False)):
        for offset in np.linspace(0.0, 1.2, 7):
            if horizontal:
                ax.add_patch(Rectangle((start[0] + offset, start[1]), 0.10, 0.23,
                                       facecolor='white', edgecolor='none', zorder=2))
            else:
                ax.add_patch(Rectangle((start[0], start[1] + offset), 0.23, 0.10,
                                       facecolor='white', edgecolor='none', zorder=2))
    ax.text(5.0, 7.70, 'Fixed roadside coordinate system', fontsize=9, color=INK,
            weight='bold', ha='center', va='center', zorder=4)
    ax.text(5.0, 7.37, 'shared calibration for images, LiDAR, and occupancy labels',
            fontsize=7.1, color=MUTED, ha='center', va='center', zorder=4)
    origin = (5.93, 3.92)
    ax.annotate('', xy=(origin[0] + 0.58, origin[1]), xytext=origin,
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.4), zorder=5)
    ax.annotate('', xy=(origin[0], origin[1] + 0.58), xytext=origin,
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.4), zorder=5)
    ax.text(origin[0] + 0.66, origin[1] - 0.05, '$x$', color=GREEN, fontsize=8, va='top')
    ax.text(origin[0] - 0.05, origin[1] + 0.67, '$y$', color=GREEN, fontsize=8, ha='right')


def draw_topology(ax):
    rounded_box(ax, (0.25, 8.35), 9.5, 1.28, '#f8fafc', '#dbe3eb', radius=0.16, linewidth=0.8)
    ax.text(0.55, 9.27, 'INFRAOCC  ·  SENSOR CONFIGURATION',
            fontsize=10.4, weight='bold', color=INK, va='center')
    ax.text(0.55, 8.57, 'Two GPS-synchronized roadside units observe one shared traffic space.',
            fontsize=8.0, color=MUTED, va='center')
    items = [('4 RGB', BLUE), ('2 LiDAR', AMBER), ('GPS sync', GREEN)]
    x = 5.85
    for label, color in items:
        rounded_box(ax, (x, 8.90), 1.12, 0.42, '#ffffff', '#dbe3eb', radius=0.12, linewidth=0.7)
        ax.add_patch(Circle((x + 0.16, 9.11), 0.06, facecolor=color, edgecolor='none', zorder=3))
        ax.text(x + 0.29, 9.11, label, fontsize=6.7, color=INK, va='center')
        x += 1.18


def draw_outputs(ax):
    y = -0.92
    ax.text(1.95, y + 0.29, 'Unified evaluation protocol', fontsize=8.3, color=MUTED,
            va='center', ha='left')
    labels = [('Camera only', BLUE), ('LiDAR only', AMBER), ('C + L fusion', GREEN)]
    x = 4.48
    for label, color in labels:
        rounded_box(ax, (x, y), 1.57, 0.56, color, radius=0.15, zorder=3)
        ax.text(x + 0.785, y + 0.28, label, fontsize=7.3, color='white', weight='bold',
                ha='center', va='center', zorder=4)
        x += 1.75


def render(output_path, dpi):
    fig, ax = plt.subplots(figsize=(11.7, 8.05), dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.2, 9.9)
    ax.set_aspect('equal')
    ax.axis('off')
    draw_topology(ax)
    draw_intersection(ax)
    draw_unit(ax, 2.35, 6.23, 'Infrastructure unit 1', (-18, 38))
    draw_unit(ax, 7.65, 1.77, 'Infrastructure unit 2', (162, -142))
    ax.add_patch(FancyArrowPatch((3.1, 6.12), (4.92, 5.25), arrowstyle='-|>',
                                 mutation_scale=10, lw=1.0, linestyle='--', color=GREEN,
                                 alpha=0.75, zorder=4))
    ax.add_patch(FancyArrowPatch((6.9, 1.93), (5.96, 2.76), arrowstyle='-|>',
                                 mutation_scale=10, lw=1.0, linestyle='--', color=GREEN,
                                 alpha=0.75, zorder=4))
    draw_outputs(ax)
    # Keep the wide canvas: this figure introduces a multi-sensor topology and
    # should read as a horizontal overview in the two-column paper layout.
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    fig.savefig(output_path, dpi=dpi, pad_inches=0.03)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help='PNG output path.')
    parser.add_argument('--dpi', type=int, default=300, help='Rasterization DPI.')
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(args.output, args.dpi)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
