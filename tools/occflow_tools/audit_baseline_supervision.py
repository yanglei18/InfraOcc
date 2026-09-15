#!/usr/bin/env python3
"""Audit one real training sample for each adapted occupancy-flow baseline."""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from mmcv import Config
from mmcv.parallel import DataContainer
from mmcv.utils import import_modules_from_strings
from mmdet3d.datasets import build_dataset


DEFAULT_CONFIGS = (
    'projects/ALOcc/configs/alocc_c_2x4_24e.py',
    'projects/CRTFusion/configs/crtfusion_c_2x4_24e.py',
    'projects/LetOccFlow/configs/letoccflow_c_2x4_24e.py',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configs', nargs='*', default=DEFAULT_CONFIGS)
    parser.add_argument('--index', type=int, default=0)
    return parser.parse_args()


def unwrap(value):
    return value.data if isinstance(value, DataContainer) else value


def shape(value):
    return list(value.shape) if hasattr(value, 'shape') else None


def expected_occ_shape(cfg):
    model = cfg.model
    if model.type == 'LetOccFlow':
        return list(model.segmentor.head.occ_size)
    if model.type == 'CRTFusion':
        return list(model.occ_head.occ_size)
    grid = model.grid_config
    return [
        round((grid[axis][1] - grid[axis][0]) / grid[axis][2])
        for axis in ('x', 'y', 'z')
    ]


def expected_depth_shape(cfg, image_shape):
    """Return the per-camera depth-loss plane implied by the model config."""
    _, _, image_height, image_width = image_shape
    if cfg.model.type == 'ALOCC':
        downsample = cfg.img_feat_downsample_rate
    else:
        downsample = cfg.img_feat_downsample_rate
    if image_height % downsample or image_width % downsample:
        raise AssertionError(
            f'{cfg.filename}: image {image_height}x{image_width} is not divisible '
            f'by depth downsample {downsample}')
    return [image_height // downsample, image_width // downsample]


def occupancy_head_config(cfg):
    if cfg.model.type == 'ALOCC':
        return cfg.model.alocc_head
    if cfg.model.type == 'LetOccFlow':
        return cfg.model.segmentor.head
    return cfg.model.occ_head


def supervised_target_report(cfg, target, flow, dynamic_class_indices):
    """Validate the shared semantic and motion supervision contract."""
    target = torch.as_tensor(target)
    flow = torch.as_tensor(flow)
    valid = target != 255
    if not valid.any():
        raise AssertionError(f'{cfg.filename}: the sampled occupancy target is empty')

    head = occupancy_head_config(cfg)
    num_classes = getattr(head, 'num_classes', None)
    if num_classes is None:
        num_classes = head.num_occupancy_classes
    labels = target[valid]
    if labels.min() < 0 or labels.max() >= num_classes:
        raise AssertionError(
            f'{cfg.filename}: labels [{int(labels.min())}, {int(labels.max())}] '
            f'are outside [0, {num_classes - 1}]')

    dynamic = target.new_zeros(target.shape, dtype=bool)
    for class_index in dynamic_class_indices:
        dynamic |= target == int(class_index)
    flow_valid = torch.isfinite(flow).all(dim=-1)
    # RoadOcc's shared GT stores zero displacement for static cells. Every
    # adapted head explicitly intersects this finite mask with ``dynamic``
    # before computing the velocity loss, so expose both quantities here.
    flow_loss_mask = flow_valid & dynamic
    flow_magnitude = torch.linalg.vector_norm(flow.float(), dim=-1)
    moving_dynamic = flow_loss_mask & (flow_magnitude > 1e-3)
    if flow_loss_mask.any():
        dynamic_magnitudes = flow_magnitude[flow_loss_mask]
        flow_magnitude_report = dict(
            dynamic_flow_mean=float(dynamic_magnitudes.mean()),
            dynamic_flow_median=float(dynamic_magnitudes.median()),
            dynamic_flow_max=float(dynamic_magnitudes.max()),
        )
    else:
        flow_magnitude_report = dict(
            dynamic_flow_mean=0.0,
            dynamic_flow_median=0.0,
            dynamic_flow_max=0.0,
        )
    return dict(
        valid_occupancy_voxels=int(valid.sum()),
        semantic_label_range=[int(labels.min()), int(labels.max())],
        dynamic_voxels=int(dynamic.sum()),
        finite_flow_voxels=int(flow_valid.sum()),
        flow_loss_voxels=int(flow_loss_mask.sum()),
        moving_dynamic_voxels=int(moving_dynamic.sum()),
        moving_dynamic_ratio=(
            float(moving_dynamic.sum() / flow_loss_mask.sum())
            if flow_loss_mask.any() else 0.0),
        static_flow_ignored_voxels=int((flow_valid & ~dynamic).sum()),
        **flow_magnitude_report,
    )


def audit_config(path, index):
    cfg = Config.fromfile(path)
    import_modules_from_strings(**cfg.custom_imports)
    sample = build_dataset(cfg.data.train)[index]
    model_type = cfg.model.type
    expected_occ = expected_occ_shape(cfg)
    report = dict(config=path, model=model_type, expected_occ=expected_occ)

    img_inputs = unwrap(sample['img_inputs'])
    report['image_tensor'] = shape(img_inputs[0])
    report['image_calibration'] = shape(img_inputs[1])
    report['depth_loss_plane'] = expected_depth_shape(cfg, report['image_tensor'])
    if model_type == 'ALOCC':
        target = unwrap(sample['gt_occupancy'])
        flow = unwrap(sample['gt_occ_flow'])
        raw_depth = unwrap(sample['gt_depth'])
        report.update(supervised_target_report(
            cfg, target, flow, occupancy_head_config(cfg).dynamic_class_indices))
        report.update(
            occupancy_target=shape(target),
            flow_target=shape(flow),
            depth_target=shape(raw_depth),
            depth_contract=(
                'raw [N_cam, 256, 704]; CM_DepthNet downsamples it to '
                '[N_cam, 16, 44] before the depth loss.'),
            valid_depth_pixels=int((raw_depth > 0).sum()),
            camera_contract='8 images: four camera-major current/adjacent pairs.',
        )
    else:
        target = unwrap(sample['voxel_semantic'])
        flow = unwrap(sample['voxel_occflows'])
        head = occupancy_head_config(cfg)
        report.update(supervised_target_report(
            cfg, target, flow, head.dynamic_class_indices))
        report.update(
            occupancy_target=shape(target),
            flow_target=shape(flow),
        )
        if model_type == 'CRTFusion':
            depth = unwrap(sample['gt_depth'])
            report.update(
                depth_target=shape(depth),
                depth_contract=(
                    'raw [N_cam, 256, 704]; CRTFusion downsamples it to '
                    '[N_cam, 16, 44] inside get_depth_loss.'),
                valid_depth_pixels=int((depth > 0).sum()),
                camera_contract='4 current-frame images; adapter selects indices [0, 2, 4, 6].',
            )
        else:
            depth = unwrap(sample['gt_depth'])
            report.update(
                depth_target=shape(depth),
                depth_contract=(
                    'raw [N_cam, 256, 704] for the common training '
                    'diagnostic; LetOccFlow has no depth loss.'),
                valid_depth_pixels=int((depth > 0).sum()),
                camera_contract='8 images reordered from camera-major to [current cameras, previous cameras].',
            )

    if report['occupancy_target'] != expected_occ:
        raise AssertionError(
            f'{path}: occupancy target {report["occupancy_target"]} != {expected_occ}')
    if report['flow_target'] != expected_occ + [2]:
        raise AssertionError(
            f'{path}: flow target {report["flow_target"]} != {expected_occ + [2]}')
    if model_type == 'ALOCC':
        expected_raw_depth = [cfg.data_config.Ncams] + report['image_tensor'][-2:]
        if report['depth_target'] != expected_raw_depth:
            raise AssertionError(
                f'{path}: raw depth {report["depth_target"]} != {expected_raw_depth}')
    elif model_type in ('CRTFusion', 'LetOccFlow'):
        expected_raw_depth = [cfg.data_config.Ncams] + report['image_tensor'][-2:]
        if report['depth_target'] != expected_raw_depth:
            raise AssertionError(
                f'{path}: raw depth {report["depth_target"]} != {expected_raw_depth}')
    return report


def main():
    args = parse_args()
    reports = [audit_config(path, args.index) for path in args.configs]
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
