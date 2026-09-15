#!/usr/bin/env python3
"""Run one adapted baseline checkpoint through its real V2X-Real test pipeline.

Unlike ``tools/test.py``, this diagnostic intentionally does not collect every
dense occupancy prediction into a pickle.  It is meant to quickly verify the
checkpoint/input contract and emit the shared BEV, flow, and depth diagnostics
for one deterministic validation sample.
"""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmcv.utils import import_modules_from_strings

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--work-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_modules_from_strings(**cfg.custom_imports)

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg.work_dir = str(work_dir)
    cfg.model.meta_info = dict(figures_path=str(work_dir / 'figures_path'))
    cfg.model.visualization_interval = 1

    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    if not 0 <= args.sample_index < len(dataset):
        raise ValueError(
            f'sample index {args.sample_index} is outside [0, {len(dataset)})')

    # A one-item subset retains the exact test pipeline and collation contract.
    subset = torch.utils.data.Subset(dataset, [args.sample_index])
    data_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        collate_fn=partial(collate, samples_per_gpu=1))

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **next(iter(data_loader)))

    if not isinstance(result, list) or len(result) != 1:
        raise RuntimeError(f'Expected one result, got {type(result)} with {len(result)} entries')
    prediction = np.asarray(result[0]['pred_occupancy'])
    flow = np.asarray(result[0]['pred_flow'])
    classes, counts = np.unique(prediction, return_counts=True)
    print('prediction_shape=', tuple(prediction.shape))
    print('flow_shape=', tuple(flow.shape))
    print('class_histogram=', dict(zip(classes.tolist(), counts.tolist())))


if __name__ == '__main__':
    main()
