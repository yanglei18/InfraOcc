"""Analyze dynamic occupancy errors from saved semantic predictions.

The script uses the same GT loading, re-anchoring, and nearest-neighbour
resizing rules as ``eval_occ_tables.py``.  It reports per-class TP/FP/FN,
distance-wise confusion, and 2D BEV connected-component coverage for dynamic
classes.
"""

import argparse
import json
import os
import sys
from os import path as osp

import numpy as np
from scipy import ndimage

_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mmdet3d.datasets import build_dataset

from tools.infraocc.common import (
    class_names_from_cfg,
    get_split_cfg,
    group_indices_from_cfg,
    load_config,
    load_occ_gt,
    load_results,
    resize_semantics,
    warp_semantics_by_reanchor,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze dynamic occupancy TP/FP/FN and BEV components.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--distance-bins', nargs='+', type=float,
                        default=[0, 20, 40, 60, 80, 100])
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--disable-reanchor-gt-warp', action='store_true')
    return parser.parse_args()


def build_distance_map(shape, cfg):
    point_cloud_range = cfg.get('point_cloud_range', None)
    if point_cloud_range is None:
        point_cloud_range = cfg.model.spatials_config.point_cloud_range
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in point_cloud_range]
    xs = np.linspace(x_min, x_max, int(shape[0]), endpoint=False)
    ys = np.linspace(y_min, y_max, int(shape[1]), endpoint=False)
    xs += (x_max - x_min) / max(int(shape[0]), 1) / 2.0
    ys += (y_max - y_min) / max(int(shape[1]), 1) / 2.0
    return np.sqrt(xs[:, None, None] ** 2 + ys[None, :, None] ** 2)


def new_confusion(class_count):
    return np.zeros((class_count, class_count), dtype=np.int64)


def add_confusion(confusion, prediction, target):
    prediction = np.asarray(prediction, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    valid = ((target >= 0) & (target < confusion.shape[0])
             & (prediction >= 0) & (prediction < confusion.shape[1]))
    if not np.any(valid):
        return
    values = np.bincount(
        target[valid] * confusion.shape[1] + prediction[valid],
        minlength=confusion.size)
    confusion += values.reshape(confusion.shape)


def class_metrics(confusion, class_idx, class_names=None):
    true_positive = int(confusion[class_idx, class_idx])
    false_positive = int(confusion[:, class_idx].sum() - true_positive)
    false_negative = int(confusion[class_idx, :].sum() - true_positive)
    gt_count = true_positive + false_negative
    pred_count = true_positive + false_positive
    union = true_positive + false_positive + false_negative
    metrics = {
        'gt': gt_count,
        'pred': pred_count,
        'tp': true_positive,
        'fp': false_positive,
        'fn': false_negative,
        'precision': round(true_positive / max(pred_count, 1) * 100.0, 3),
        'recall': round(true_positive / max(gt_count, 1) * 100.0, 3),
        'iou': round(true_positive / max(union, 1) * 100.0, 3),
    }
    if class_names is not None:
        fn_breakdown = confusion[class_idx, :].copy()
        fp_breakdown = confusion[:, class_idx].copy()
        fn_breakdown[class_idx] = 0
        fp_breakdown[class_idx] = 0
        metrics['fn_by_pred'] = {
            class_names[index]: int(fn_breakdown[index])
            for index in np.argsort(fn_breakdown)[::-1]
            if fn_breakdown[index] > 0
        }
        metrics['fp_by_gt'] = {
            class_names[index]: int(fp_breakdown[index])
            for index in np.argsort(fp_breakdown)[::-1]
            if fp_breakdown[index] > 0
        }
    return metrics


def dynamic_group_counts(confusion, dynamic_indices, static_indices, empty_idx):
    dynamic_indices = np.asarray(dynamic_indices, dtype=np.int64)
    static_indices = np.asarray(static_indices, dtype=np.int64)
    gt_dynamic = int(confusion[dynamic_indices, :].sum())
    pred_dynamic = int(confusion[:, dynamic_indices].sum())
    tp_dynamic = int(confusion[np.ix_(dynamic_indices, dynamic_indices)].sum())
    gt_static = int(confusion[static_indices, :].sum())
    pred_static = int(confusion[:, static_indices].sum())
    gt_free = int(confusion[empty_idx, :].sum())
    pred_free = int(confusion[:, empty_idx].sum())
    return {
        'gt_dynamic': gt_dynamic,
        'pred_dynamic': pred_dynamic,
        'dynamic_tp': tp_dynamic,
        'dynamic_fn': gt_dynamic - tp_dynamic,
        'dynamic_fp': pred_dynamic - tp_dynamic,
        'dynamic_recall': round(tp_dynamic / max(gt_dynamic, 1) * 100.0, 3),
        'dynamic_precision': round(tp_dynamic / max(pred_dynamic, 1) * 100.0, 3),
        'gt_static': gt_static,
        'pred_static': pred_static,
        'gt_free': gt_free,
        'pred_free': pred_free,
    }


def component_stats(gt, prediction, class_idx):
    """Measure same-class coverage of GT dynamic footprints in BEV."""
    gt_bev = np.any(gt == class_idx, axis=2)
    pred_bev = np.any(prediction == class_idx, axis=2)
    structure = np.ones((3, 3), dtype=np.uint8)
    components, count = ndimage.label(gt_bev, structure=structure)
    size_bins = {
        '1-2': {'components': 0, 'hit50': 0, 'coverage_sum': 0.0},
        '3-5': {'components': 0, 'hit50': 0, 'coverage_sum': 0.0},
        '6+': {'components': 0, 'hit50': 0, 'coverage_sum': 0.0},
    }
    coverages = []
    for component_id in range(1, count + 1):
        component = components == component_id
        area = int(component.sum())
        overlap = int(np.logical_and(component, pred_bev).sum())
        coverage = overlap / max(area, 1)
        coverages.append(coverage)
        if area <= 2:
            bucket = size_bins['1-2']
        elif area <= 5:
            bucket = size_bins['3-5']
        else:
            bucket = size_bins['6+']
        bucket['components'] += 1
        bucket['hit50'] += int(coverage >= 0.5)
        bucket['coverage_sum'] += coverage

    for bucket in size_bins.values():
        bucket['hit_rate'] = round(
            bucket['hit50'] / max(bucket['components'], 1) * 100.0, 3)
        bucket['mean_coverage'] = round(
            bucket['coverage_sum'] / max(bucket['components'], 1) * 100.0, 3)
        del bucket['coverage_sum']
    return {
        'components': int(count),
        'hit50': int(sum(item['hit50'] for item in size_bins.values())),
        'hit_rate': round(
            sum(item['hit50'] for item in size_bins.values())
            / max(count, 1) * 100.0, 3),
        'mean_coverage': round(np.mean(coverages) * 100.0, 3)
        if coverages else 0.0,
        'size_bins': size_bins,
    }


def main():
    args = parse_args()
    if len(args.distance_bins) < 2:
        raise ValueError('--distance-bins requires at least two edges.')

    cfg = load_config(args.config)
    dataset = build_dataset(get_split_cfg(cfg, args.split))
    results = load_results(args.results)
    class_names = class_names_from_cfg(cfg)
    group_info = group_indices_from_cfg(cfg)
    ignored_names = set(getattr(dataset, 'miou_ignore_class_names', []))
    dynamic_indices = [
        index for index in group_info['dynamic_indices']
        if class_names[index] not in ignored_names
    ]
    static_indices = group_info['static_indices']
    empty_idx = group_info['empty_idx']
    class_count = len(class_names)

    reanchor_mat = (
        dataset._get_eval_reanchor_mat()
        if hasattr(dataset, '_get_eval_reanchor_mat') else None)
    if args.disable_reanchor_gt_warp:
        reanchor_mat = None
    point_cloud_range = (
        dataset._get_eval_point_cloud_range()
        if hasattr(dataset, '_get_eval_point_cloud_range') else None)
    if point_cloud_range is None:
        point_cloud_range = cfg.get('point_cloud_range', None)
    if point_cloud_range is None:
        point_cloud_range = cfg.model.spatials_config.point_cloud_range

    overall = new_confusion(class_count)
    distance_confusions = [new_confusion(class_count)
                           for _ in range(len(args.distance_bins) - 1)]
    components = {
        class_names[index]: {
            'components': 0,
            'hit50': 0,
            'size_bins': {
                '1-2': {'components': 0, 'hit50': 0, 'coverage_sum': 0.0},
                '3-5': {'components': 0, 'hit50': 0, 'coverage_sum': 0.0},
                '6+': {'components': 0, 'hit50': 0, 'coverage_sum': 0.0},
            },
            'coverage_sum': 0.0,
        }
        for index in dynamic_indices
    }

    for result in results:
        index = result['index']
        prediction = np.asarray(result['occ_results'])
        target = load_occ_gt(dataset, index)
        if reanchor_mat is not None:
            target = warp_semantics_by_reanchor(
                target,
                point_cloud_range=point_cloud_range,
                reanchor_mat=reanchor_mat,
                empty_idx=empty_idx)
        if prediction.shape != target.shape:
            prediction = resize_semantics(prediction, target.shape)

        add_confusion(overall, prediction, target)
        distance_map = np.broadcast_to(
            build_distance_map(target.shape, cfg), target.shape)
        for bin_index, (low, high) in enumerate(
                zip(args.distance_bins[:-1], args.distance_bins[1:])):
            mask = (distance_map >= low) & (distance_map < high)
            add_confusion(distance_confusions[bin_index],
                          prediction[mask], target[mask])

        for class_idx in dynamic_indices:
            current = component_stats(target, prediction, class_idx)
            aggregate = components[class_names[class_idx]]
            aggregate['components'] += current['components']
            aggregate['hit50'] += current['hit50']
            aggregate['coverage_sum'] += (
                current['mean_coverage'] / 100.0 * current['components'])
            for size_name, item in current['size_bins'].items():
                bucket = aggregate['size_bins'][size_name]
                bucket['components'] += item['components']
                bucket['hit50'] += item['hit50']
                bucket['coverage_sum'] += (
                    item['mean_coverage'] / 100.0 * item['components'])

    per_class = {
        class_names[index]: class_metrics(overall, index, class_names)
        for index in dynamic_indices
    }
    distance_summary = {}
    for bin_index, (low, high) in enumerate(
            zip(args.distance_bins[:-1], args.distance_bins[1:])):
        confusion = distance_confusions[bin_index]
        distance_summary[f'{low:g}-{high:g}m'] = {
            'group': dynamic_group_counts(
                confusion, dynamic_indices, static_indices, empty_idx),
            'per_class': {
                class_names[index]: class_metrics(confusion, index, class_names)
                for index in dynamic_indices
            },
        }

    component_summary = {}
    for class_name, aggregate in components.items():
        component_count = aggregate['components']
        class_summary = {
            'components': component_count,
            'hit50': aggregate['hit50'],
            'hit_rate': round(
                aggregate['hit50'] / max(component_count, 1) * 100.0, 3),
            'mean_coverage': round(
                aggregate['coverage_sum'] / max(component_count, 1) * 100.0, 3),
            'size_bins': {},
        }
        for size_name, bucket in aggregate['size_bins'].items():
            count = bucket['components']
            class_summary['size_bins'][size_name] = {
                'components': count,
                'hit50': bucket['hit50'],
                'hit_rate': round(
                    bucket['hit50'] / max(count, 1) * 100.0, 3),
                'mean_coverage': round(
                    bucket['coverage_sum'] / max(count, 1) * 100.0, 3),
            }
        component_summary[class_name] = class_summary

    output = {
        'num_samples': len(results),
        'per_class': per_class,
        'distance_bins': distance_summary,
        'bev_components': component_summary,
        'dynamic_classes': [class_names[index] for index in dynamic_indices],
        'notes': {
            'tp_fp_fn': 'Semantic voxel counts from the full class confusion matrix.',
            'bev_components': (
                'GT same-class footprints are 8-connected in XY after projecting '
                'over Z; hit50 means same-class prediction covers at least 50% '
                'of the GT footprint.'),
        },
    }
    output_dir = osp.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(output, handle, indent=2, sort_keys=True)

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
