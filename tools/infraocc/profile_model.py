import argparse
import json
import os
import sys
import time
from os import path as osp

import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mmdet
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

from tools.infraocc.common import get_split_cfg, load_config

if mmdet.__version__ > '2.23.0':
    from mmdet.utils import compat_cfg, setup_multi_processes
else:
    from mmdet3d.utils import compat_cfg, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile InfraOcc parameters and optional inference latency.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--warmup-iters', type=int, default=5)
    parser.add_argument('--profile-iters', type=int, default=20)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, default=None)
    parser.add_argument('--output-json', default=None)
    return parser.parse_args()


def count_params(module):
    return sum(parameter.numel() for parameter in module.parameters())


def module_params(model, name):
    module = getattr(model, name, None)
    if module is None:
        return 0
    return count_params(module)


def build_summary(model):
    total_params = count_params(model)
    head_params = module_params(model, 'occupancy_head')
    prior_params = module_params(model, 'prior_static_bias')
    return dict(
        total_params_m=round(total_params / 1e6, 4),
        occupancy_head_params_m=round(head_params / 1e6, 4),
        prior_branch_params_m=round(prior_params / 1e6, 4),
    )


def profile_latency(model, data_loader, warmup_iters, profile_iters):
    timings = []
    processed = 0
    with torch.no_grad():
        for data in data_loader:
            if processed >= warmup_iters + profile_iters:
                break
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(return_loss=False, rescale=True, **data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if processed >= warmup_iters:
                timings.append(elapsed_ms)
            processed += 1

    if not timings:
        return dict(latency_ms=None, fps=None, num_profile_iters=0)
    latency_ms = sum(timings) / len(timings)
    return dict(
        latency_ms=round(latency_ms, 4),
        fps=round(1000.0 / latency_ms, 4),
        num_profile_iters=len(timings),
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    cfg.model.pretrained = None
    cfg.gpu_ids = [args.gpu_id]

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    summary = build_summary(model)

    if args.checkpoint is not None:
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        load_checkpoint(model, args.checkpoint, map_location='cpu')
        model = MMDataParallel(model.cuda(args.gpu_id), device_ids=[args.gpu_id])
        model.eval()

        split_cfg = get_split_cfg(cfg, args.split)
        if isinstance(split_cfg, dict) and cfg.data.get('test_dataloader', {}).get('samples_per_gpu', 1) > 1:
            split_cfg.pipeline = replace_ImageToTensor(split_cfg.pipeline)
        dataset = build_dataset(split_cfg)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.get('workers_per_gpu', 2),
            dist=False,
            shuffle=False,
            runner_type='EpochBasedRunnerEval')
        summary.update(profile_latency(
            model,
            data_loader,
            warmup_iters=args.warmup_iters,
            profile_iters=args.profile_iters))

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json:
        output_dir = osp.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
