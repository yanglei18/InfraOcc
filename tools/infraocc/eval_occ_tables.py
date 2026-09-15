import argparse
import json
import os
import sys
from os import path as osp

import numpy as np

_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mmdet3d.datasets import build_dataset

from tools.infraocc.common import (
    class_names_from_cfg,
    confusion_to_iou,
    get_split_cfg,
    group_indices_from_cfg,
    load_config,
    load_occ_gt,
    load_results,
    resize_semantics,
    warp_semantics_by_reanchor,
    update_confusion,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate InfraOcc results.pkl into table-ready metrics.')
    parser.add_argument('--config', required=True, help='Config path.')
    parser.add_argument('--results', required=True, help='Pickle results path.')
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--output-json', default=None, help='Optional JSON output path.')
    parser.add_argument('--latex-name', default=None, help='Optional method name for a LaTeX row.')
    parser.add_argument(
        '--distance-bins',
        nargs='*',
        type=float,
        default=None,
        help='Optional distance bin edges in meters, e.g. 0 20 40 60 80.')
    parser.add_argument(
        '--scene-density-tertiles',
        action='store_true',
        help='Report Sparse/Medium/Dense scene bins by GT dynamic voxel count tertiles.')
    parser.add_argument(
        '--occlusion-tertiles',
        action='store_true',
        help=(
            'Report Clear/Partial/Heavy bins by a GT static-contact proxy: '
            'the fraction of dynamic voxels adjacent to static voxels.'))
    parser.add_argument(
        '--disable-reanchor-gt-warp',
        action='store_true',
        help='Evaluate against the original GT frame even if the test pipeline has re-anchoring.')
    return parser.parse_args()


def mean_over_indices(values, indices):
    valid_values = [float(values[index]) for index in indices if index is not None]
    if not valid_values:
        return None
    return float(np.mean(valid_values))


def maybe_round(value):
    if value is None:
        return None
    return round(float(value), 2)


def summarize_confusion(confusion, empty_idx, static_indices, dynamic_indices,
                        valid_indices, geom_counts):
    per_class_iou = confusion_to_iou(confusion)
    geom_tp, geom_fp, geom_fn = geom_counts
    return dict(
        gIoU=maybe_round(
            geom_tp / max(geom_tp + geom_fp + geom_fn, 1) * 100.0),
        mIoU=maybe_round(mean_over_indices(per_class_iou, valid_indices)),
        mean_static=maybe_round(mean_over_indices(per_class_iou, static_indices)),
        mean_dynamic=maybe_round(mean_over_indices(per_class_iou, dynamic_indices)),
        free_iou=maybe_round(float(per_class_iou[empty_idx])),
    )


def update_geometry_counts(prediction, target, empty_idx, valid_mask):
    gt_occ = (target != empty_idx) & valid_mask
    pred_occ = (prediction != empty_idx) & valid_mask
    return (
        int(np.logical_and(gt_occ, pred_occ).sum()),
        int(np.logical_and(~gt_occ, pred_occ).sum()),
        int(np.logical_and(gt_occ, ~pred_occ).sum()))


def build_group_mapping(num_classes, static_indices, dynamic_indices, empty_idx):
    mapping = np.full((num_classes,), fill_value=255, dtype=np.int64)
    mapping[np.asarray(static_indices, dtype=np.int64)] = 0
    mapping[np.asarray(dynamic_indices, dtype=np.int64)] = 1
    mapping[int(empty_idx)] = 2
    return mapping


def update_group_confusion(group_confusion, prediction, target, group_mapping):
    prediction = np.asarray(prediction, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    valid_mask = (
        (target != 255)
        & (target >= 0)
        & (target < group_mapping.shape[0])
        & (prediction >= 0)
        & (prediction < group_mapping.shape[0]))
    if not np.any(valid_mask):
        return
    target_group = group_mapping[target[valid_mask]]
    pred_group = group_mapping[prediction[valid_mask]]
    group_valid = (target_group != 255) & (pred_group != 255)
    if not np.any(group_valid):
        return
    bincount = np.bincount(
        target_group[group_valid] * group_confusion.shape[1] + pred_group[group_valid],
        minlength=group_confusion.size)
    group_confusion += bincount.reshape(group_confusion.shape)


def summarize_group_confusion(group_confusion):
    names = ['static', 'dynamic', 'free']
    row_sum = group_confusion.sum(axis=1, keepdims=True).clip(min=1)
    row_percent = group_confusion.astype(np.float64) / row_sum * 100.0
    return {
        gt_name: {
            pred_name: maybe_round(row_percent[gt_index, pred_index])
            for pred_index, pred_name in enumerate(names)
        }
        for gt_index, gt_name in enumerate(names)
    }


def build_distance_map(shape, cfg):
    point_cloud_range = cfg.get('point_cloud_range', None)
    if point_cloud_range is None:
        point_cloud_range = cfg.model.spatials_config.point_cloud_range
    x_min, y_min, _, x_max, y_max, _ = [float(value) for value in point_cloud_range]
    xs = np.linspace(x_min, x_max, int(shape[0]), endpoint=False)
    ys = np.linspace(y_min, y_max, int(shape[1]), endpoint=False)
    xs += (x_max - x_min) / max(int(shape[0]), 1) / 2.0
    ys += (y_max - y_min) / max(int(shape[1]), 1) / 2.0
    return np.sqrt(xs[:, None, None] ** 2 + ys[None, :, None] ** 2)


def summarize_class_regimes(per_class_iou, class_names):
    regime_names = {
        'large_dynamic': ['bus', 'car', 'truck'],
        'vulnerable_dynamic': ['bicycle', 'motorcycle', 'pedestrian'],
        'traffic_control_static': ['barrier', 'traffic_cone'],
        'road_surface_static': ['driveable_surface', 'sidewalk', 'terrain'],
        'scene_layout_static': ['others', 'manmade', 'vegetation'],
    }
    summary = {}
    for regime_name, names in regime_names.items():
        indices = [
            class_names.index(name)
            for name in names
            if name in class_names
        ]
        summary[regime_name] = maybe_round(mean_over_indices(per_class_iou, indices))
    return summary


def dynamic_voxel_count(target, dynamic_indices):
    if not dynamic_indices:
        return 0
    return int(np.isin(target, np.asarray(dynamic_indices, dtype=np.int64)).sum())


def static_contact_score(target, static_indices, dynamic_indices):
    if not static_indices or not dynamic_indices:
        return 0.0
    static_mask = np.isin(target, np.asarray(static_indices, dtype=np.int64))
    dynamic_mask = np.isin(target, np.asarray(dynamic_indices, dtype=np.int64))
    dynamic_count = int(dynamic_mask.sum())
    if dynamic_count == 0:
        return 0.0

    contact_mask = np.zeros_like(dynamic_mask, dtype=bool)
    # Six-neighbour static contact without scipy dependency.
    for axis in range(3):
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        src[axis] = slice(1, None)
        dst[axis] = slice(0, -1)
        contact_mask[tuple(dst)] |= static_mask[tuple(src)]

        src[axis] = slice(0, -1)
        dst[axis] = slice(1, None)
        contact_mask[tuple(dst)] |= static_mask[tuple(src)]
    return float(np.logical_and(dynamic_mask, contact_mask).sum() / dynamic_count)


def tertile_assignments(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}, (0.0, 0.0)
    low, high = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    assignments = {}
    for index, value in enumerate(values):
        if value <= low:
            assignments[index] = 0
        elif value <= high:
            assignments[index] = 1
        else:
            assignments[index] = 2
    return assignments, (float(low), float(high))


def summarize_sample_bins(records, assignments, names, class_count, empty_idx,
                          static_indices, dynamic_indices, valid_indices):
    bins = [
        dict(
            name=name,
            confusion=np.zeros((class_count, class_count), dtype=np.int64),
            geom=[0, 0, 0],
            count=0)
        for name in names
    ]
    for record_index, record in enumerate(records):
        bin_index = assignments.get(record_index)
        if bin_index is None:
            continue
        bin_item = bins[bin_index]
        prediction = record['prediction']
        target = record['target']
        valid_mask = record['valid_mask']
        update_confusion(bin_item['confusion'], prediction, target)
        bin_tp, bin_fp, bin_fn = update_geometry_counts(
            prediction, target, empty_idx, valid_mask)
        bin_item['geom'][0] += bin_tp
        bin_item['geom'][1] += bin_fp
        bin_item['geom'][2] += bin_fn
        bin_item['count'] += 1
    return {
        bin_item['name']: dict(
            **summarize_confusion(
                bin_item['confusion'],
                empty_idx=empty_idx,
                static_indices=static_indices,
                dynamic_indices=dynamic_indices,
                valid_indices=valid_indices,
                geom_counts=tuple(bin_item['geom'])),
            num_samples=bin_item['count'])
        for bin_item in bins
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dataset = build_dataset(get_split_cfg(cfg, args.split))
    flat_results = load_results(args.results)
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

    group_info = group_indices_from_cfg(cfg)
    class_names = class_names_from_cfg(cfg)
    empty_idx = group_info['empty_idx']
    ignore_names = list(getattr(dataset, 'miou_ignore_class_names', []))
    ignore_indices = {
        class_names.index(name)
        for name in ignore_names
        if name in class_names
    }
    valid_indices = [
        index for index in range(len(class_names))
        if index not in ignore_indices and index != empty_idx
    ]
    static_indices = [
        index for index in group_info['static_indices']
        if index not in ignore_indices
    ]
    dynamic_indices = [
        index for index in group_info['dynamic_indices']
        if index not in ignore_indices
    ]
    group_mapping = build_group_mapping(
        len(class_names),
        group_info['static_indices'],
        group_info['dynamic_indices'],
        empty_idx)

    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    group_confusion = np.zeros((3, 3), dtype=np.int64)
    geom_tp = 0
    geom_fp = 0
    geom_fn = 0
    sample_records = []
    distance_bins = []
    if args.distance_bins is not None:
        if len(args.distance_bins) < 2:
            raise ValueError('--distance-bins requires at least two edges.')
        distance_bins = [
            dict(
                name=f'{args.distance_bins[index]:g}-{args.distance_bins[index + 1]:g}m',
                low=float(args.distance_bins[index]),
                high=float(args.distance_bins[index + 1]),
                confusion=np.zeros((len(class_names), len(class_names)), dtype=np.int64),
                geom=[0, 0, 0])
            for index in range(len(args.distance_bins) - 1)
        ]

    for result in flat_results:
        index = result['index']
        prediction = np.asarray(result['occ_results'])
        gt_semantics = load_occ_gt(dataset, index)
        if reanchor_mat is not None:
            gt_semantics = warp_semantics_by_reanchor(
                gt_semantics,
                point_cloud_range=point_cloud_range,
                reanchor_mat=reanchor_mat,
                empty_idx=empty_idx)
        if prediction.shape != gt_semantics.shape:
            prediction = resize_semantics(prediction, gt_semantics.shape)

        update_confusion(confusion, prediction, gt_semantics)
        update_group_confusion(
            group_confusion,
            prediction,
            gt_semantics,
            group_mapping)

        valid_mask = (
            (gt_semantics >= 0)
            & (gt_semantics < len(class_names))
            & (gt_semantics != 255))
        if args.scene_density_tertiles or args.occlusion_tertiles:
            sample_records.append(
                dict(
                    prediction=prediction,
                    target=gt_semantics,
                    valid_mask=valid_mask,
                    dynamic_count=dynamic_voxel_count(
                        gt_semantics,
                        dynamic_indices),
                    static_contact=static_contact_score(
                        gt_semantics,
                        static_indices,
                        dynamic_indices)))
        batch_tp, batch_fp, batch_fn = update_geometry_counts(
            prediction, gt_semantics, empty_idx, valid_mask)
        geom_tp += batch_tp
        geom_fp += batch_fp
        geom_fn += batch_fn

        if distance_bins:
            distance_map = build_distance_map(gt_semantics.shape, cfg)
            for bin_item in distance_bins:
                bin_mask = (
                    valid_mask
                    & (distance_map >= bin_item['low'])
                    & (distance_map < bin_item['high']))
                if not np.any(bin_mask):
                    continue
                update_confusion(
                    bin_item['confusion'],
                    prediction[bin_mask],
                    gt_semantics[bin_mask])
                bin_tp, bin_fp, bin_fn = update_geometry_counts(
                    prediction, gt_semantics, empty_idx, bin_mask)
                bin_item['geom'][0] += bin_tp
                bin_item['geom'][1] += bin_fp
                bin_item['geom'][2] += bin_fn

    per_class_iou = confusion_to_iou(confusion)
    distance_summary = {}
    for bin_item in distance_bins:
        distance_summary[bin_item['name']] = summarize_confusion(
            bin_item['confusion'],
            empty_idx=empty_idx,
            static_indices=static_indices,
            dynamic_indices=dynamic_indices,
            valid_indices=valid_indices,
            geom_counts=tuple(bin_item['geom']))

    scene_density_summary = {}
    scene_density_thresholds = None
    if args.scene_density_tertiles:
        assignments, thresholds = tertile_assignments([
            record['dynamic_count'] for record in sample_records
        ])
        scene_density_thresholds = dict(low=thresholds[0], high=thresholds[1])
        scene_density_summary = summarize_sample_bins(
            sample_records,
            assignments,
            names=['sparse', 'medium', 'dense'],
            class_count=len(class_names),
            empty_idx=empty_idx,
            static_indices=static_indices,
            dynamic_indices=dynamic_indices,
            valid_indices=valid_indices)

    occlusion_summary = {}
    occlusion_thresholds = None
    if args.occlusion_tertiles:
        assignments, thresholds = tertile_assignments([
            record['static_contact'] for record in sample_records
        ])
        occlusion_thresholds = dict(low=thresholds[0], high=thresholds[1])
        occlusion_summary = summarize_sample_bins(
            sample_records,
            assignments,
            names=['clear', 'partial', 'heavy'],
            class_count=len(class_names),
            empty_idx=empty_idx,
            static_indices=static_indices,
            dynamic_indices=dynamic_indices,
            valid_indices=valid_indices)

    metrics = dict(
        **summarize_confusion(
            confusion,
            empty_idx=empty_idx,
            static_indices=static_indices,
            dynamic_indices=dynamic_indices,
            valid_indices=valid_indices,
            geom_counts=(geom_tp, geom_fp, geom_fn)),
        per_class_iou={
            class_names[index]: maybe_round(float(per_class_iou[index]))
            for index in range(len(class_names))
        },
        group_confusion=summarize_group_confusion(group_confusion),
        class_regimes=summarize_class_regimes(per_class_iou, class_names),
        distance_bins=distance_summary,
        scene_density_bins=scene_density_summary,
        scene_density_thresholds=scene_density_thresholds,
        occlusion_bins=occlusion_summary,
        occlusion_thresholds=occlusion_thresholds,
        split_definitions=dict(
            mIoU='Mean IoU over semantic classes excluding ignored classes and free space.',
            scene_density=(
                'Sparse/medium/dense are validation/test-set tertiles of '
                'GT dynamic voxel count per sample.'),
            occlusion=(
                'Clear/partial/heavy are tertiles of the GT static-contact proxy, '
                'defined as the fraction of dynamic voxels adjacent to static voxels.')),
        ignore_classes=ignore_names,
        num_samples=len(flat_results),
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.latex_name:
        row = ' & '.join([
            args.latex_name,
            f"{metrics['gIoU']:.2f}",
            f"{metrics['mIoU']:.2f}",
            f"{metrics['mean_static']:.2f}",
            f"{metrics['mean_dynamic']:.2f}",
            f"{metrics['free_iou']:.2f}",
        ]) + r' \\'
        print('\nLaTeX row:')
        print(row)

    if args.output_json:
        output_dir = osp.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
