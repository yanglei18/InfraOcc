import argparse
import math
import os
import pickle
import sys
from os import path as osp

from mmcv import Config


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit InfraOcc configs for batch-size diagnosis.')
    parser.add_argument('--configs', nargs='+', required=True)
    return parser.parse_args()


def load_train_count(cfg):
    ann_file = getattr(cfg, 'train_ann_file', None)
    if ann_file is None:
        ann_file = cfg.data.train.ann_file
    ann_path = osp.join(REPO_ROOT, ann_file) if not osp.isabs(ann_file) else ann_file
    if not osp.exists(ann_path):
        return None, ann_path
    with open(ann_path, 'rb') as handle:
        infos = pickle.load(handle)
    try:
        count = len(infos['infos'])
    except Exception:  # noqa: BLE001
        count = len(infos)
    return count, ann_path


def get_value(cfg, key, default=None):
    return getattr(cfg, key, default)


def main():
    args = parse_args()
    rows = []
    for config_path in args.configs:
        cfg = Config.fromfile(config_path)
        samples_per_gpu = cfg.data.samples_per_gpu
        workers_per_gpu = cfg.data.workers_per_gpu
        gpus = get_value(cfg, 'num_gpus', 4)
        global_batch = gpus * samples_per_gpu
        lr = cfg.optimizer.get('lr', None)
        max_epochs = cfg.runner.get('max_epochs', None)
        train_count, ann_path = load_train_count(cfg)
        iters_per_epoch = None
        total_steps = None
        if train_count is not None:
            iters_per_epoch = math.ceil(train_count / global_batch)
            if max_epochs is not None:
                total_steps = iters_per_epoch * max_epochs
        work_dir = get_value(cfg, 'work_dir', None)
        load_from = get_value(cfg, 'load_from', None)
        norm_eval = None
        try:
            norm_eval = cfg.model.forward_projection.img_backbone.norm_eval
        except Exception:  # noqa: BLE001
            norm_eval = None
        rows.append(dict(
            config=config_path,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=workers_per_gpu,
            num_gpus=gpus,
            global_batch=global_batch,
            lr=lr,
            max_epochs=max_epochs,
            train_samples=train_count,
            iters_per_epoch=iters_per_epoch,
            total_steps=total_steps,
            load_from=load_from,
            work_dir=work_dir,
            norm_eval=norm_eval,
            train_ann_file=ann_path,
        ))

    headers = [
        'config', 'samples_per_gpu', 'global_batch', 'lr', 'max_epochs',
        'iters_per_epoch', 'total_steps', 'norm_eval', 'work_dir'
    ]
    print('\t'.join(headers))
    for row in rows:
        print('\t'.join(str(row.get(header, '')) for header in headers))


if __name__ == '__main__':
    main()
