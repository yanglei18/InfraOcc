import argparse
import json
import os
import sys
from collections import defaultdict
from os import path as osp

import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mmdet
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

from tools.infraocc.common import (
    confusion_to_iou,
    flatten_results,
    get_split_cfg,
    load_config,
    load_occ_gt,
    resize_semantics,
    update_confusion,
)

if mmdet.__version__ > '2.23.0':
    from mmdet.utils import compat_cfg, setup_multi_processes
else:
    from mmdet3d.utils import compat_cfg, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run progressive debug analysis for InfraOcc.')
    parser.add_argument('--config', required=True, help='Config path.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path.')
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--work-dir', default=None, help='Optional artifact directory.')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, default=None)
    parser.add_argument('--output-json', default=None, help='Optional JSON output path.')
    return parser.parse_args()


def map_labels(target, mapping, ignore_index=255):
    target = np.asarray(target, dtype=np.int64)
    mapped = np.full_like(target, fill_value=ignore_index, dtype=np.int64)
    valid_mask = (
        (target != ignore_index)
        & (target >= 0)
        & (target < mapping.shape[0]))
    if np.any(valid_mask):
        mapped[valid_mask] = mapping[target[valid_mask]]
    return mapped


def mean_or_none(values):
    if not values:
        return None
    return float(np.mean(values))


def round_or_none(value):
    if value is None:
        return None
    return round(float(value), 4)


def update_group_stats(stats, group_name, mask, payload):
    if not np.any(mask):
        return
    fusion_weights = payload['fusion_weights']
    routed_group = np.asarray(fusion_weights).argmax(axis=-1)
    stats[group_name]['count'] += int(mask.sum())
    stats[group_name]['static_conf'] += float(payload['static_conf'][mask].sum())
    stats[group_name]['dynamic_conf'] += float(payload['dynamic_conf'][mask].sum())
    stats[group_name]['suppression_gate'] += float(payload['suppression_gate'][mask].sum())
    stats[group_name]['w_sta'] += float(fusion_weights[..., 0][mask].sum())
    stats[group_name]['w_dyn'] += float(fusion_weights[..., 1][mask].sum())
    stats[group_name]['w_emp'] += float(fusion_weights[..., 2][mask].sum())
    stats[group_name]['route_sta'] += int((routed_group[mask] == 0).sum())
    stats[group_name]['route_dyn'] += int((routed_group[mask] == 1).sum())
    stats[group_name]['route_emp'] += int((routed_group[mask] == 2).sum())


def summarize_group_stats(stats):
    summary = {}
    for group_name, values in stats.items():
        count = max(values['count'], 1)
        summary[group_name] = dict(
            count=int(values['count']),
            static_conf=round(values['static_conf'] / count, 4),
            dynamic_conf=round(values['dynamic_conf'] / count, 4),
            suppression_gate=round(values['suppression_gate'] / count, 4),
            w_sta=round(values['w_sta'] / count, 4),
            w_dyn=round(values['w_dyn'] / count, 4),
            w_emp=round(values['w_emp'] / count, 4),
            route_sta=round(values['route_sta'] / count * 100.0, 4),
            route_dyn=round(values['route_dyn'] / count * 100.0, 4),
            route_emp=round(values['route_emp'] / count * 100.0, 4),
        )
    return summary


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.gpu_ids = [args.gpu_id]
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    meta_info = dict(cfg.model.get('meta_info', {}))
    meta_info['export_analysis'] = True
    cfg.model.meta_info = meta_info

    split_cfg = get_split_cfg(cfg, args.split)
    samples_per_gpu = 1
    if isinstance(split_cfg, dict) and cfg.data.get('test_dataloader', {}).get('samples_per_gpu', 1) > 1:
        split_cfg.pipeline = replace_ImageToTensor(split_cfg.pipeline)
    dataset = build_dataset(split_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 2),
        dist=False,
        shuffle=False,
        runner_type='EpochBasedRunnerEval')

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model.cuda(args.gpu_id), device_ids=[args.gpu_id])
    model.eval()

    head = model.module.occupancy_head
    prior_branch = model.module.prior_static_bias
    num_classes = int(head.num_classes)
    empty_idx = int(head.empty_idx)
    static_mapping = head.static_label_mapping.detach().cpu().numpy()
    dynamic_mapping = head.dynamic_label_mapping.detach().cpu().numpy()
    prior_mapping = prior_branch.target_map.detach().cpu().numpy()

    static_confusion = np.zeros(
        (len(head.static_class_indices) + 1, len(head.static_class_indices) + 1),
        dtype=np.int64)
    dynamic_confusion = np.zeros(
        (len(head.dynamic_class_indices) + 1, len(head.dynamic_class_indices) + 1),
        dtype=np.int64)
    prior_confusions = {}
    prior_correct = defaultdict(int)
    prior_total = defaultdict(int)
    prior_alpha = defaultdict(list)
    prior_consistency = defaultdict(list)
    routing_stats = {
        'static': defaultdict(float),
        'dynamic': defaultdict(float),
        'free': defaultdict(float),
    }
    module_scalars = defaultdict(list)
    processed_samples = 0
    found_analysis_payload = False

    for data in data_loader:
        if args.max_samples is not None and processed_samples >= args.max_samples:
            break
        with torch.no_grad():
            outputs = model(return_loss=False, rescale=True, export_analysis=True, **data)
        batch_results = flatten_results(outputs)
        for batch_result in batch_results:
            if args.max_samples is not None and processed_samples >= args.max_samples:
                break
            payload = batch_result.get('analysis_payload')
            if payload is None:
                continue
            found_analysis_payload = True
            index = batch_result['index']
            gt_semantics = load_occ_gt(dataset, index)

            static_target = map_labels(gt_semantics, static_mapping)
            dynamic_target = map_labels(gt_semantics, dynamic_mapping)
            update_confusion(static_confusion, payload['static_pred'], static_target)
            update_confusion(dynamic_confusion, payload['dynamic_pred'], dynamic_target)

            valid_mask = (
                (gt_semantics >= 0)
                & (gt_semantics < num_classes)
                & (gt_semantics != 255))
            static_mask = valid_mask & np.isin(
                gt_semantics,
                np.asarray(head.static_class_indices, dtype=np.int64))
            dynamic_mask = valid_mask & np.isin(
                gt_semantics,
                np.asarray(head.dynamic_class_indices, dtype=np.int64))
            free_mask = valid_mask & (gt_semantics == empty_idx)

            update_group_stats(routing_stats, 'static', static_mask, payload)
            update_group_stats(routing_stats, 'dynamic', dynamic_mask, payload)
            update_group_stats(routing_stats, 'free', free_mask, payload)

            for scale_name, scale_logits in payload.get('prior_static_logits', {}).items():
                stage_target = resize_semantics(gt_semantics, scale_logits.shape[:3])
                prior_target = map_labels(stage_target, prior_mapping)
                stage_prediction = np.asarray(scale_logits).argmax(axis=-1)
                if scale_name not in prior_confusions:
                    num_prior_classes = int(scale_logits.shape[-1])
                    prior_confusions[scale_name] = np.zeros(
                        (num_prior_classes, num_prior_classes),
                        dtype=np.int64)
                update_confusion(
                    prior_confusions[scale_name],
                    stage_prediction,
                    prior_target)
                valid_prior = prior_target != 255
                prior_correct[scale_name] += int(
                    (stage_prediction[valid_prior] == prior_target[valid_prior]).sum())
                prior_total[scale_name] += int(valid_prior.sum())

            for scale_name, alpha_value in payload.get('prior_static_alpha', {}).items():
                prior_alpha[scale_name].append(float(alpha_value))
            for scale_name, consistency_value in payload.get('prior_static_consistency', {}).items():
                prior_consistency[scale_name].append(float(consistency_value))

            module_scalars['suppression_alpha'].append(float(payload['suppression_alpha']))
            module_scalars['raw_bypass_scale'].append(float(payload['raw_bypass_scale']))
            processed_samples += 1

    if not found_analysis_payload:
        raise RuntimeError(
            'No analysis payload was returned. Use an InfraOccProgressiveHead config '
            'or enable meta_info.export_analysis.')

    static_iou = confusion_to_iou(static_confusion)
    dynamic_iou = confusion_to_iou(dynamic_confusion)
    prior_summary = {}
    for scale_name in sorted(prior_confusions.keys()):
        scale_iou = confusion_to_iou(prior_confusions[scale_name])
        prior_summary[scale_name] = dict(
            miou=round(float(scale_iou.mean()), 4),
            acc=round(
                float(prior_correct[scale_name] / max(prior_total[scale_name], 1) * 100.0),
                4),
            consistency=round_or_none(mean_or_none(prior_consistency[scale_name])),
            alpha=round_or_none(mean_or_none(prior_alpha[scale_name])),
        )

    summary = dict(
        num_samples=processed_samples,
        branch_specialization=dict(
            static_group_miou=round(float(static_iou.mean()), 4),
            dynamic_group_miou=round(float(dynamic_iou.mean()), 4),
            static_group_iou=[round(float(value), 4) for value in static_iou.tolist()],
            dynamic_group_iou=[round(float(value), 4) for value in dynamic_iou.tolist()],
        ),
        prior_stage_quality=prior_summary,
        routing_statistics=summarize_group_stats(routing_stats),
        module_scalars=dict(
            suppression_alpha=round_or_none(mean_or_none(module_scalars['suppression_alpha'])),
            raw_bypass_scale=round_or_none(mean_or_none(module_scalars['raw_bypass_scale'])),
        ),
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json:
        output_dir = osp.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
