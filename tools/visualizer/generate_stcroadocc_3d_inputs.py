"""Generate matched STCRoadOcc predictions for 3D comparison figures."""

import argparse
import os
import os.path as osp
import pickle

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from torch.utils.data import Subset

from mmdet3d.core.visualizer.occ_visualization import render_camera_grid
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


DYNAMIC_INDICES = (2, 3, 4, 6, 7, 10)
IGNORED_INDICES = (5, 9, 12)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate matched predictions for STCRoadOcc 3D figures')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--indices', type=int, nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=2)
    return parser.parse_args()


def scene_start(infos, index):
    scene_token = infos[index]['scene_token']
    while index > 0 and infos[index - 1]['scene_token'] == scene_token:
        index -= 1
    return index


def required_sequence_indices(infos, targets):
    selected = set()
    for target in targets:
        selected.update(range(scene_start(infos, target), target + 1))
    return sorted(selected)


def unwrap_test_output(outputs):
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise RuntimeError(f'Expected one test output, got {type(outputs)}.')
    output = outputs[0]
    semantics = np.asarray(output['occ_results'])
    if semantics.ndim == 4 and semantics.shape[0] == 1:
        semantics = semantics[0]
    if semantics.shape != (320, 320, 16):
        raise RuntimeError(
            f'Unexpected occupancy prediction shape {semantics.shape}.')
    return semantics.astype(np.uint8)


def class_mean_iou(prediction, target, class_indices):
    values = []
    valid = target != 255
    for class_index in class_indices:
        predicted = valid & (prediction == class_index)
        expected = valid & (target == class_index)
        union = np.count_nonzero(predicted | expected)
        if union:
            values.append(np.count_nonzero(predicted & expected) / union)
    return float(np.mean(values)) if values else float('nan')


def main():
    args = parse_args()
    targets = set(args.indices)
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.SAVED_INTERVALS = 0
    cfg.data.test.test_mode = True

    with open(cfg.data.test.ann_file, 'rb') as annotation_file:
        annotation = pickle.load(annotation_file)
    infos = annotation['infos'] if isinstance(annotation, dict) else annotation
    traversal = required_sequence_indices(infos, targets)

    dataset = build_dataset(cfg.data.test)
    subset = Subset(dataset, traversal)
    data_loader = build_dataloader(
        subset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
        runner_type='PlainSequentialRunner',
        pin_memory=True)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model.save_flow_results = False
    model.eval()
    model = MMDataParallel(model.cuda(), device_ids=[0])

    prediction_dir = osp.join(args.output_dir, 'predictions')
    camera_dir = osp.join(args.output_dir, 'camera')
    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(camera_dir, exist_ok=True)

    with torch.no_grad():
        for local_index, data in enumerate(data_loader):
            dataset_index = traversal[local_index]
            outputs = model(return_loss=False, rescale=True, **data)
            if dataset_index not in targets:
                continue
            prediction = unwrap_test_output(outputs)
            info = infos[dataset_index]
            token = info['token']
            target = np.load(osp.join(info['occ_path'], 'labels.npz'))[
                'semantics']
            np.savez_compressed(
                osp.join(prediction_dir, f'{dataset_index:04d}_{token}.npz'),
                semantics=prediction,
                dataset_index=np.asarray(dataset_index),
                token=np.asarray(token))

            dataset_item = dataset[dataset_index]
            meta = dataset_item['img_metas'][0].data
            camera_grid = render_camera_grid(
                dataset_item['img_inputs'],
                sample_index=0,
                camera_num_frame=1,
                cam_names=meta.get('cam_names'))
            if camera_grid is not None:
                cv2.imwrite(
                    osp.join(camera_dir,
                             f'{dataset_index:04d}_{token}_camera.png'),
                    camera_grid)

            included_classes = tuple(
                index for index in range(17) if index not in IGNORED_INDICES)
            all_iou = class_mean_iou(prediction, target, included_classes)
            dynamic_iou = class_mean_iou(
                prediction, target, DYNAMIC_INDICES)
            print(
                f'checkpoint={osp.basename(args.checkpoint)} '
                f'index={dataset_index} token={token} '
                f'mIoU={all_iou * 100.0:.2f} '
                f'DynIoU={dynamic_iou * 100.0:.2f}',
                flush=True)


if __name__ == '__main__':
    main()
