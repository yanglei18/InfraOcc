#!/usr/bin/env python3
"""Render InfraOcc static--dynamic temporal statistics from frame annotations.

The script computes BEV occupied-frame ratios directly from ``labels.npz``
files listed in a V2X-Real info pickle.  Static and dynamic maps use the same
fixed roadside grid, and the lower panel is the empirical survival curve of
the resulting nonzero BEV cells.  A saved ``--stats-cache`` avoids repeating
the annotation pass when polishing the artwork.
"""

import argparse
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'tools/visualizer/tpami_reproduction/fig06-temporal-statistics-reproduction.png'
DEFAULT_CURVE_OUTPUT = ROOT / 'tools/visualizer/tpami_reproduction/fig06c-occupancy-ratio-distributions.png'
DEFAULT_INFO_PKLS = (
    ROOT / 'data/v2xreal_nuscenes/v2xreal_infos_train.pkl',
    ROOT / 'data/v2xreal_nuscenes/v2xreal_infos_val.pkl',
)

CLASS_NAMES = (
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free')
STATIC_IDS = np.array([0, 1, 11, 12, 13, 14, 15, 16], dtype=np.uint8)
DYNAMIC_IDS = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint8)
STATIC_COLORS = {
    # Match the project's occupancy visualizer: driveable_surface is magenta.
    'driveable / flat': '#ff00ff',
    'structure': '#8992a3',
    'vegetation': '#00af00',
}
DYNAMIC_COLORS = {
    # Coarse groups retain the warm/cool visual separation used for Fig. 6.
    'vehicles': '#ef6c4f',
    'pedestrians': '#c76ca6',
    'cyclists': '#0878b9',
}
INK = '#1f2937'
MUTED = '#64748b'


def load_infos(path):
    with path.open('rb') as handle:
        obj = pickle.load(handle)
    if isinstance(obj, dict):
        obj = obj.get('infos', obj.get('data_list', obj))
    if not isinstance(obj, list):
        raise TypeError(f'Unsupported info format in {path}: {type(obj)}')
    return obj


def category_map(semantics, group):
    """Return one BEV class map; each group is mutually exclusive in a frame."""
    if group == 'static':
        ground = np.isin(semantics, [11, 12, 13, 14]).any(axis=2)
        structure = np.isin(semantics, [0, 1, 15]).any(axis=2)
        vegetation = (semantics == 16).any(axis=2)
        category = np.zeros(ground.shape, dtype=np.uint8)
        category[ground] = 1
        category[structure] = 2
        category[vegetation] = 3
        return category
    vehicle = np.isin(semantics, [3, 4, 5, 9, 10]).any(axis=2)
    pedestrian = np.isin(semantics, [7, 8]).any(axis=2)
    two_wheeler = np.isin(semantics, [2, 6]).any(axis=2)
    category = np.zeros(vehicle.shape, dtype=np.uint8)
    category[vehicle] = 1
    category[pedestrian] = 2
    category[two_wheeler] = 3
    return category


def accumulate_statistics(infos, max_frames=-1, frame_stride=1, frame_start=0,
                          progress_every=300):
    """Accumulate frame occupancy count and dominant group category for each BEV cell."""
    infos = infos[::frame_stride]
    infos = infos[frame_start:]
    infos = infos if max_frames < 0 else infos[:max_frames]
    static_count = dynamic_count = static_votes = dynamic_votes = None
    valid_frames = 0
    for index, info in enumerate(infos, start=1):
        label_path = Path(info['occ_path']) / 'labels.npz'
        if not label_path.exists():
            continue
        with np.load(label_path) as labels:
            semantics = labels['semantics']
        static = category_map(semantics, 'static')
        dynamic = category_map(semantics, 'dynamic')
        if static_count is None:
            shape = static.shape
            static_count = np.zeros(shape, dtype=np.uint32)
            dynamic_count = np.zeros(shape, dtype=np.uint32)
            static_votes = np.zeros((4, *shape), dtype=np.uint32)
            dynamic_votes = np.zeros((4, *shape), dtype=np.uint32)
        static_count += static > 0
        dynamic_count += dynamic > 0
        for category in range(1, 4):
            static_votes[category] += static == category
            dynamic_votes[category] += dynamic == category
        valid_frames += 1
        if progress_every and index % progress_every == 0:
            print(f'Processed {index}/{len(infos)} frames ({valid_frames} valid labels).')
    if valid_frames == 0:
        raise RuntimeError('No labels.npz files were found from the provided info pickle.')
    return {
        'num_frames': np.array(valid_frames, dtype=np.int32),
        'static_ratio': static_count.astype(np.float32) / valid_frames,
        'dynamic_ratio': dynamic_count.astype(np.float32) / valid_frames,
        'static_category': static_votes.argmax(axis=0).astype(np.uint8),
        'dynamic_category': dynamic_votes.argmax(axis=0).astype(np.uint8),
    }


def merge_caches(paths):
    """Merge disjoint temporal shards without requiring a second label pass."""
    shards = []
    for path in paths:
        with np.load(path) as cached:
            shards.append({key: cached[key] for key in cached.files})
    required = {'num_frames', 'static_ratio', 'dynamic_ratio', 'static_category', 'dynamic_category'}
    for path, shard in zip(paths, shards):
        missing = required.difference(shard)
        if missing:
            raise ValueError(f'Cache {path} misses keys: {sorted(missing)}')
    weights = np.array([int(shard['num_frames']) for shard in shards], dtype=np.float32)
    total_frames = int(weights.sum())
    static_ratio = sum(weight * shard['static_ratio'] for weight, shard in zip(weights, shards)) / total_frames
    dynamic_ratio = sum(weight * shard['dynamic_ratio'] for weight, shard in zip(weights, shards)) / total_frames

    def majority_category(name):
        categories = np.stack([shard[name] for shard in shards])
        result = np.zeros(categories.shape[1:], dtype=np.uint8)
        for category in range(1, 4):
            votes = (categories == category).sum(axis=0)
            result[votes > (categories == result).sum(axis=0)] = category
        return result

    return {
        'num_frames': np.array(total_frames, dtype=np.int32),
        'static_ratio': static_ratio.astype(np.float32),
        'dynamic_ratio': dynamic_ratio.astype(np.float32),
        'static_category': majority_category('static_category'),
        'dynamic_category': majority_category('dynamic_category'),
    }


def load_or_compute(args):
    if args.merge_caches:
        result = merge_caches(args.merge_caches)
        print(f'Merged {len(args.merge_caches)} temporal-statistics shards.')
        if args.stats_cache:
            args.stats_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.stats_cache, **result)
            print(f'Saved merged temporal statistics cache to {args.stats_cache}')
        return result
    if args.stats_cache and args.stats_cache.exists():
        with np.load(args.stats_cache) as cached:
            result = {key: cached[key] for key in cached.files}
        print(f'Loaded temporal statistics from {args.stats_cache}')
        return result
    infos = []
    for info_pkl in args.info_pkl:
        infos.extend(load_infos(info_pkl))
    result = accumulate_statistics(
        infos, max_frames=args.max_frames, frame_stride=args.frame_stride,
        frame_start=args.frame_start,
        progress_every=args.progress_every)
    if args.stats_cache:
        args.stats_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.stats_cache, **result)
        print(f'Saved temporal statistics cache to {args.stats_cache}')
    return result


def orient(array):
    return np.asarray(array).T[::-1, ::-1]


def semantic_image(ratio, category, colors, intensity_gamma=0.45, min_ratio=0.0,
                   max_alpha=1.0, blur_sigma=0.0):
    ratio = orient(ratio)
    category = orient(category)
    image = np.full((*ratio.shape, 3), 0.985, dtype=np.float32)
    for category_id, key in enumerate(colors, start=1):
        color = np.array(plt.matplotlib.colors.to_rgb(colors[key]), dtype=np.float32)
        mask = (category == category_id) & (ratio >= min_ratio)
        alpha = np.zeros_like(ratio, dtype=np.float32)
        alpha[mask] = max_alpha * np.power(
            np.clip(ratio[mask], 0.0, 1.0), intensity_gamma)
        if blur_sigma > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
            alpha = np.minimum(alpha, max_alpha)
        # Standard alpha compositing keeps weak dynamic support nearly white,
        # while its spatially smoothed support becomes a continuous motion trail.
        image = image * (1.0 - alpha[..., None]) + color * alpha[..., None]
    return image


def persistence_curve(ratio):
    values = ratio[ratio > 0]
    thresholds = np.linspace(0.0, 1.0, 401)
    if not len(values):
        return thresholds, np.zeros_like(thresholds)
    return thresholds, np.array([(values >= threshold).mean() for threshold in thresholds])


def ridge_histogram(values, bins):
    """Return a smooth, per-group normalized persistence density ridge."""
    counts, _ = np.histogram(values * 100.0, bins=bins)
    positions = np.linspace(-2.8, 2.8, 17)
    kernel = np.exp(-0.5 * positions ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(counts.astype(np.float32), kernel, mode='same')
    return smooth / max(float(smooth.max()), 1.0)


def add_panel_label(ax, label, title):
    ax.text(0.0, 1.035, label, transform=ax.transAxes, fontsize=11, color=INK,
            weight='bold', va='bottom')
    ax.text(0.105, 1.037, title, transform=ax.transAxes, fontsize=9.2, color=INK,
            weight='medium', va='bottom')


def draw_distribution_panel(ax, static_ratio, dynamic_ratio):
    """Draw panel (c), a compact same-axis persistence distribution comparison."""
    static_values = static_ratio[static_ratio > 0]
    dynamic_values = dynamic_ratio[dynamic_ratio > 0]
    static_median = np.median(static_values)
    dynamic_median = np.median(dynamic_values)

    add_panel_label(ax, '(c)', 'Occupancy-ratio distributions')
    ax.set_facecolor('white')
    bins = np.linspace(0.0, 100.0, 51)
    centers = 0.5 * (bins[:-1] + bins[1:])
    static_ridge = ridge_histogram(static_values, bins)
    dynamic_ridge = ridge_histogram(dynamic_values, bins)
    height = 0.92
    static_density = height * static_ridge
    dynamic_density = height * dynamic_ridge
    ax.fill_between(centers, 0, static_density, color='#2a9d8f', alpha=0.70,
                    label='Static')
    ax.fill_between(centers, 0, dynamic_density, color='#e76f51', alpha=0.70,
                    label='Dynamic')
    ax.plot(centers, static_density, color='#1f766c', lw=1.25)
    ax.plot(centers, dynamic_density, color='#c95038', lw=1.25)
    ax.axvline(static_median * 100, color='#1f766c', lw=1.05, ls='--')
    ax.axvline(dynamic_median * 100, color='#c95038', lw=1.05, ls='--')
    ax.text(static_median * 100 - 2.0, 1.03,
            f'Static median {static_median * 100:.0f}%', fontsize=7.4,
            color='#1f766c', ha='right', va='bottom')
    ax.text(dynamic_median * 100 + 2.2, 1.03,
            f'Dynamic median {dynamic_median * 100:.1f}%', fontsize=7.4,
            color='#c95038', ha='left', va='bottom')
    ax.set(xlim=(0, 100), ylim=(0, 1.20), xlabel='Occupied-frame ratio (%)',
           ylabel='Normalized density')
    ax.set_yticks([])
    ax.grid(axis='x', color='#eef2f7', lw=0.7)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#64748b')
    ax.tick_params(labelsize=8, colors=INK)
    ax.legend(loc='upper center', bbox_to_anchor=(0.58, 0.80), fontsize=7.6,
              frameon=False, ncol=2)


def draw_portrait_distribution_panel(ax, static_ratio, dynamic_ratio):
    """Draw a tall, narrow distribution view for compact manuscript layouts."""
    static_values = static_ratio[static_ratio > 0]
    dynamic_values = dynamic_ratio[dynamic_ratio > 0]
    static_median = np.median(static_values) * 100.0
    dynamic_median = np.median(dynamic_values) * 100.0
    bins = np.linspace(0.0, 100.0, 51)
    centers = 0.5 * (bins[:-1] + bins[1:])
    static_density = 0.88 * ridge_histogram(static_values, bins)
    dynamic_density = 0.88 * ridge_histogram(dynamic_values, bins)
    static_support = static_density > 0.02
    dynamic_support = dynamic_density > 0.02

    ax.fill_betweenx(centers, 0, static_density, color='#2a9d8f', alpha=0.70)
    ax.fill_betweenx(centers, 0, -dynamic_density, color='#e76f51', alpha=0.70)
    ax.plot(static_density[static_support], centers[static_support], color='#1f766c', lw=1.2)
    ax.plot(-dynamic_density[dynamic_support], centers[dynamic_support], color='#c95038', lw=1.2)
    ax.axhline(static_median, color='#1f766c', lw=1.0, ls='--')
    ax.axhline(dynamic_median, color='#c95038', lw=1.0, ls='--')
    ax.axvline(0, color='#64748b', lw=0.75)
    ax.text(0.0, 1.035, '(c)', transform=ax.transAxes, fontsize=10.5,
            color=INK, weight='bold', va='bottom')
    ax.text(0.19, 1.037, 'Temporal persistence', transform=ax.transAxes,
            fontsize=8.5, color=INK, weight='medium', va='bottom')
    ax.text(-0.83, 30.0, f'Dynamic\nmedian {dynamic_median:.1f}%', fontsize=6.6,
            color='#c95038', ha='left', va='center')
    ax.text(0.83, 70.0, f'Static\nmedian {static_median:.0f}%', fontsize=6.6,
            color='#1f766c', ha='right', va='center')
    ax.text(-0.83, 47, 'Dynamic', fontsize=7.2, color='#c95038', ha='left',
            weight='medium')
    ax.text(0.83, 53, 'Static', fontsize=7.2, color='#1f766c', ha='right',
            weight='medium')
    ax.set(xlim=(-0.98, 0.98), ylim=(0, 100), ylabel='Occupied-frame ratio (%)')
    ax.set_xticks([])
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis='y', color='#eef2f7', lw=0.7)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.spines['left'].set_color('#64748b')
    ax.tick_params(labelsize=7.6, colors=INK)


def render(stats, output, dpi):
    static_ratio = stats['static_ratio']
    dynamic_ratio = stats['dynamic_ratio']

    fig = plt.figure(figsize=(8.35, 8.25), dpi=dpi, facecolor='white')
    grid = fig.add_gridspec(2, 2, height_ratios=(1.08, 0.74), hspace=0.34, wspace=0.16)
    ax_static = fig.add_subplot(grid[0, 0])
    ax_dynamic = fig.add_subplot(grid[0, 1])
    ax_curve = fig.add_subplot(grid[1, :])

    for ax, ratio, category, colors, label, title in (
            (ax_static, static_ratio, stats['static_category'], STATIC_COLORS,
             '(a)', 'Static semantic persistence'),
            (ax_dynamic, dynamic_ratio, stats['dynamic_category'], DYNAMIC_COLORS,
             '(b)', 'Dynamic transience')):
        if label == '(b)':
            # Dynamic cells are transient.  A linear-ish intensity mapping keeps
            # one-off occupancy nearly white and exposes only recurrent paths.
            image = semantic_image(ratio, category, colors, intensity_gamma=0.48,
                                   min_ratio=0.002, max_alpha=0.82, blur_sigma=0.78)
        else:
            image = semantic_image(ratio, category, colors, max_alpha=0.58)
        ax.imshow(image, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        add_panel_label(ax, label, title)
        ax.add_patch(Rectangle((0.05, 0.05), 0.23, 0.012, transform=ax.transAxes,
                               facecolor=INK, edgecolor='none'))
        ax.text(0.05, 0.085, '10 m', transform=ax.transAxes, fontsize=7.5, color=INK,
                va='bottom')

    static_legend = [Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                            markeredgecolor='none', markersize=6, label=name)
                     for name, color in (('driveable / flat', STATIC_COLORS['driveable / flat']),
                                         ('structure', STATIC_COLORS['structure']),
                                         ('vegetation', STATIC_COLORS['vegetation']))]
    dynamic_legend = [Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                             markeredgecolor='none', markersize=6, label=name)
                      for name, color in (('Vehicles', DYNAMIC_COLORS['vehicles']),
                                          ('Pedestrians', DYNAMIC_COLORS['pedestrians']),
                                          ('Cyclists', DYNAMIC_COLORS['cyclists']))]
    ax_static.legend(handles=static_legend, loc='lower right', fontsize=6.8, frameon=False,
                     borderpad=0.2, handletextpad=0.35)
    ax_dynamic.legend(handles=dynamic_legend, loc='lower right', fontsize=6.8, frameon=False,
                      borderpad=0.2, handletextpad=0.35)

    draw_distribution_panel(ax_curve, static_ratio, dynamic_ratio)
    fig.text(0.5, 0.015, f'Computed from {int(stats["num_frames"]):,} annotated keyframes in fixed roadside coordinates.',
             ha='center', va='bottom', fontsize=7.4, color=MUTED)
    fig.savefig(output, dpi=dpi, bbox_inches='tight', pad_inches=0.045)
    plt.close(fig)


def render_curve_only(stats, output, dpi):
    """Export panel (c) on its own for paper layout experiments."""
    fig, ax = plt.subplots(figsize=(2.15, 3.45), dpi=dpi, facecolor='white')
    draw_portrait_distribution_panel(ax, stats['static_ratio'], stats['dynamic_ratio'])
    fig.subplots_adjust(left=0.27, right=0.97, top=0.88, bottom=0.10)
    fig.savefig(output, dpi=dpi, bbox_inches='tight', pad_inches=0.045)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--info-pkl', type=Path, nargs='+', default=DEFAULT_INFO_PKLS,
                        help='One or more V2X-Real info pickles containing occ_path entries.')
    parser.add_argument('--stats-cache', type=Path,
                        help='Optional compressed NPZ cache for accumulated statistics.')
    parser.add_argument('--max-frames', type=int, default=-1,
                        help='Use only the first N keyframes; -1 uses all available labels.')
    parser.add_argument('--frame-stride', type=int, default=1,
                        help='Subsample the temporal stream with this positive stride.')
    parser.add_argument('--frame-start', type=int, default=0,
                        help='Start index after temporal subsampling; useful for disjoint shards.')
    parser.add_argument('--progress-every', type=int, default=300,
                        help='Progress-report interval during annotation accumulation.')
    parser.add_argument('--merge-caches', type=Path, nargs='+',
                        help='Merge disjoint statistics-cache shards instead of reading labels.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='PNG output path.')
    parser.add_argument('--only-c', action='store_true',
                        help='Export only panel (c), the occupancy-ratio comparison.')
    parser.add_argument('--curve-output', type=Path, default=DEFAULT_CURVE_OUTPUT,
                        help='PNG output path used with --only-c.')
    parser.add_argument('--dpi', type=int, default=300, help='Rasterization DPI.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_stride < 1 or args.frame_start < 0:
        raise ValueError('--frame-stride must be positive and --frame-start nonnegative.')
    if not args.stats_cache and args.max_frames < 0:
        print('Computing all-frame temporal statistics without a cache.')
    stats = load_or_compute(args)
    output = args.curve_output if args.only_c else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.only_c:
        render_curve_only(stats, output, args.dpi)
    else:
        render(stats, output, args.dpi)
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
