# Copyright (c) OpenMMLab. All rights reserved.
import platform
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from mmcv.utils import Registry, build_from_cfg
from mmcv.parallel import collate
from mmcv.runner import get_dist_info

from mmdet.datasets import DATASETS
from mmdet.datasets.builder import _concat_dataset, worker_init_fn
from torch.utils.data import DataLoader

from mmdet.datasets.samplers import (DistributedGroupSampler,
                       DistributedSampler, GroupSampler)

from .samplers import (CustomDistributedSampler,
                       GroupEachSampleInBatchSamplerEpoch,
                       GroupEachSampleInBatchSamplerEpochEval,
                       GroupEachSampleInBatchSamplerEpochWeighted,
                       GroupEachSampleInBatchSamplerIter,
                       GroupEachSampleInBatchSamplerIterEval,
                       GroupEachSampleInBatchSamplerIterWeighted,
                       TTADistributedSampler)


def _get_dataset_class_names(dataset, cfg):
    class_names = cfg.get('class_names')
    if class_names is not None:
        return list(class_names)
    if hasattr(dataset, 'CLASSES') and dataset.CLASSES is not None:
        return list(dataset.CLASSES)
    return None


def _count_targets_from_annotations(info, target_class_names):
    if 'gt_names' not in info:
        return None
    gt_names = np.asarray(info['gt_names']).astype(str)
    if gt_names.size == 0:
        return 0
    return int(np.isin(gt_names, list(target_class_names)).sum())


def _resolve_occ_label_path(dataset, info):
    dataset_name = getattr(dataset, 'dataset_name', 'openocc')
    if hasattr(dataset, '_resolve_occ_label_path'):
        return dataset._resolve_occ_label_path(info, dataset_name)
    return f"{info['occ_path']}/labels.npz"


def _count_targets_from_occ_labels(dataset, info, target_class_indices):
    occ_path = _resolve_occ_label_path(dataset, info)
    semantics = np.load(occ_path, allow_pickle=True)['semantics']
    return int(np.isin(semantics, list(target_class_indices)).sum())


def _count_classes_from_annotations(info, target_class_names):
    if 'gt_names' not in info:
        return None
    gt_names = np.asarray(info['gt_names']).astype(str)
    return np.asarray([
        np.count_nonzero(gt_names == class_name)
        for class_name in target_class_names
    ], dtype=np.int64)


def _count_classes_from_occ_labels(dataset, info, target_class_indices):
    occ_path = _resolve_occ_label_path(dataset, info)
    semantics = np.load(occ_path, allow_pickle=True)['semantics']
    class_counts = np.bincount(
        semantics.reshape(-1).astype(np.int64),
        minlength=max(target_class_indices) + 1)
    return class_counts[np.asarray(target_class_indices, dtype=np.int64)]


def _broadcast_group_weights(group_weights, groups_num):
    """Broadcast sequence weights without repeating occupancy-label scans."""
    rank, world_size = get_dist_info()
    if world_size == 1 or not dist.is_available() or not dist.is_initialized():
        return group_weights

    if dist.get_backend() == 'nccl':
        device = torch.device('cuda', torch.cuda.current_device())
    else:
        device = torch.device('cpu')
    if rank == 0:
        weights = torch.as_tensor(
            group_weights, dtype=torch.float32, device=device)
    else:
        weights = torch.empty(groups_num, dtype=torch.float32, device=device)
    dist.broadcast(weights, src=0)
    return weights.cpu().numpy()


def _build_sequence_group_weights(dataset, group_weight_cfg):
    if not group_weight_cfg:
        return None
    if not hasattr(dataset, 'flag') or not hasattr(dataset, 'data_infos'):
        raise ValueError(
            'Sequence group weighting requires a dataset with flag and '
            'data_infos attributes.')

    cfg = dict(group_weight_cfg)
    target_class_names = cfg.get('target_class_names', [])
    target_class_indices = cfg.get('target_class_indices')
    class_names = _get_dataset_class_names(dataset, cfg)
    if target_class_indices is None:
        if class_names is None:
            raise ValueError(
                'target_class_indices or class_names must be provided for '
                'sequence group weighting.')
        missing = [
            class_name for class_name in target_class_names
            if class_name not in class_names
        ]
        if missing:
            raise ValueError(
                'Unknown target class names for sequence group weighting: '
                f'{missing}. Available classes: {class_names}')
        target_class_indices = [
            class_names.index(class_name) for class_name in target_class_names
        ]
    elif class_names is not None and target_class_names:
        missing = [
            class_name for class_name in target_class_names
            if class_name not in class_names
        ]
        if missing:
            raise ValueError(
                'Unknown target class names for sequence group weighting: '
                f'{missing}. Available classes: {class_names}')
        expected_indices = [
            class_names.index(class_name) for class_name in target_class_names
        ]
        if list(target_class_indices) != expected_indices:
            raise ValueError(
                'target_class_indices do not match target_class_names: '
                f'{list(target_class_indices)} vs {expected_indices}')

    source = str(cfg.get('source', 'annotations')).lower()
    base_weight = float(cfg.get('base_weight', 1.0))
    max_weight = float(cfg.get('max_weight', cfg.get('positive_weight', 1.0)))
    min_count = float(cfg.get('min_count', cfg.get('min_target_count', 1.0)))
    mode = str(cfg.get('mode', 'binary')).lower()

    flag = np.asarray(dataset.flag, dtype=np.int64)
    groups_num = len(np.bincount(flag))
    rank, world_size = get_dist_info()
    distributed = (
        world_size > 1 and dist.is_available() and dist.is_initialized())
    if distributed and rank != 0:
        return _broadcast_group_weights(None, groups_num)

    if mode == 'cbgs':
        if not target_class_indices:
            raise ValueError(
                'Sequence-aware CBGS requires at least one target class.')
        group_class_counts = np.zeros(
            (groups_num, len(target_class_indices)), dtype=np.int64)
        for sample_index, group_idx in enumerate(flag):
            info = dataset.data_infos[sample_index]
            counts = None
            if source in ('annotations', 'auto'):
                counts = _count_classes_from_annotations(
                    info, target_class_names)
            if counts is None:
                if source not in ('occ_labels', 'occupancy', 'auto'):
                    raise ValueError(
                        f'Unsupported group weight source: {source}')
                counts = _count_classes_from_occ_labels(
                    dataset, info, target_class_indices)
            group_class_counts[int(group_idx)] += counts

        # Sequence-level CBGS is equivalent to first sampling a class
        # uniformly, then sampling uniformly from sequences containing it.
        # Summing those class-conditional probabilities gives each sequence's
        # multinomial weight while preserving frame order inside the sequence.
        group_presence = group_class_counts >= min_count
        class_group_counts = group_presence.sum(axis=0)
        missing_classes = np.flatnonzero(class_group_counts == 0)
        if missing_classes.size:
            missing_names = [
                (target_class_names[index]
                 if index < len(target_class_names)
                 else str(target_class_indices[index]))
                for index in missing_classes
            ]
            raise ValueError(
                'Sequence-aware CBGS found no positive groups for classes: '
                f'{missing_names}')
        class_sampling_weights = cfg.get('class_sampling_weights')
        if class_sampling_weights is None:
            class_sampling_weights = np.ones(
                len(target_class_indices), dtype=np.float64)
        elif isinstance(class_sampling_weights, dict):
            class_sampling_weights = np.asarray([
                class_sampling_weights.get(class_name, 1.0)
                for class_name in target_class_names
            ], dtype=np.float64)
        else:
            class_sampling_weights = np.asarray(
                class_sampling_weights, dtype=np.float64)
        if class_sampling_weights.shape != (len(target_class_indices),):
            raise ValueError(
                'class_sampling_weights must match the target class count: '
                f'{class_sampling_weights.shape} vs '
                f'{len(target_class_indices)}')
        if np.any(class_sampling_weights <= 0):
            raise ValueError('class_sampling_weights must all be positive.')
        inverse_class_frequency = (
            class_sampling_weights /
            class_group_counts.astype(np.float64))
        group_weights = (
            group_presence.astype(np.float64) @ inverse_class_frequency)
        positive_groups = group_weights > 0
        group_weights[positive_groups] /= group_weights[
            positive_groups].mean()
        group_weights[~positive_groups] = float(
            cfg.get('empty_group_weight', 0.0))
        max_weight = cfg.get('max_weight')
        if max_weight is not None:
            group_weights = np.minimum(group_weights, float(max_weight))
        return _broadcast_group_weights(
            group_weights.astype(np.float32), groups_num)

    group_counts = np.zeros(groups_num, dtype=np.float64)
    for sample_index, group_idx in enumerate(flag):
        info = dataset.data_infos[sample_index]
        count = None
        if source in ('annotations', 'auto'):
            count = _count_targets_from_annotations(info, target_class_names)
        if count is None:
            if source not in ('occ_labels', 'occupancy', 'auto'):
                raise ValueError(f'Unsupported group weight source: {source}')
            count = _count_targets_from_occ_labels(
                dataset, info, target_class_indices)
        group_counts[int(group_idx)] += count

    group_weights = np.full_like(group_counts, base_weight, dtype=np.float32)
    positive_mask = group_counts >= min_count
    if mode == 'binary':
        group_weights[positive_mask] = max_weight
    elif mode == 'log':
        if np.any(positive_mask):
            positive_counts = group_counts[positive_mask]
            denominator = np.log1p(max(positive_counts.max(), min_count))
            if denominator > 0:
                scaled = np.log1p(positive_counts) / denominator
                group_weights[positive_mask] = (
                    base_weight + (max_weight - base_weight) * scaled)
    else:
        raise ValueError(f'Unsupported sequence group weight mode: {mode}')

    return _broadcast_group_weights(group_weights, groups_num)


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     runner_type='EpochBasedRunner',
                     val=False,
                     **kwargs):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    group_weight_cfg = kwargs.pop('group_weight_cfg', None)

    if dist:
        # When model is :obj:`DistributedDataParallel`,
        # `batch_size` of :obj:`dataloader` is the
        # number of training samples on each GPU.
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # When model is obj:`DataParallel`
        # the batch size is samples on all the GPUS
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu
    if val:
        # runner_type = 'EpochBasedRunner'
        assert not shuffle
    if runner_type == 'IterBasedRunner':
        # TODO: original has more options, but I'm not using them
        # https://github.com/open-mmlab/mmdetection/blob/3b72b12fe9b14de906d1363982b9fba05e7d47c1/mmdet/datasets/builder.py#L145-L157

        batch_sampler = GroupEachSampleInBatchSamplerIter(
            dataset,
            batch_size,
            world_size,
            rank,
            seed=seed)
        batch_size = 1
        sampler = None
    elif runner_type == 'IterBasedRunnerEval':
        # TODO: original has more options, but I'm not using them
        # https://github.com/open-mmlab/mmdetection/blob/3b72b12fe9b14de906d1363982b9fba05e7d47c1/mmdet/datasets/builder.py#L145-L157

        batch_sampler = GroupEachSampleInBatchSamplerIterEval(
            dataset,
            batch_size,
            world_size,
            rank,
            seed=seed)
        batch_size = 1
        sampler = None
    elif runner_type == 'EpochBasedRunner':
        # Keep temporal sequence continuity for epoch-based training:
        # shuffle group order, preserve sample order inside each group.
        batch_sampler = GroupEachSampleInBatchSamplerEpoch(
            dataset,
            batch_size,
            world_size,
            rank,
            seed=seed)
        batch_size = 1
        sampler = None
    elif runner_type == 'EpochBasedRunnerWeighted':
        group_weights = _build_sequence_group_weights(
            dataset, group_weight_cfg)
        batch_sampler = GroupEachSampleInBatchSamplerEpochWeighted(
            dataset,
            batch_size,
            world_size,
            rank,
            seed=seed,
            group_weights=group_weights)
        batch_size = 1
        sampler = None
    elif runner_type == 'EpochBasedRunnerEval':
        batch_sampler = GroupEachSampleInBatchSamplerEpochEval(
            dataset,
            batch_size,
            world_size,
            rank,
            seed=seed,
            shuffle=False)
        batch_size = 1
        sampler = None
    elif runner_type == 'TTARunnerEval':
        # TODO: original has more options, but I'm not using them
        # https://github.com/open-mmlab/mmdetection/blob/3b72b12fe9b14de906d1363982b9fba05e7d47c1/mmdet/datasets/builder.py#L145-L157

        batch_sampler = TTADistributedSampler(
            dataset,
            samples_per_gpu,
            world_size,
            rank,
            seed=seed)
        sampler = None
    else:
        if dist:
            # DistributedGroupSampler will definitely shuffle the data to satisfy
            # that images on each GPU are in the same group
            if shuffle:
                sampler = DistributedGroupSampler(
                    dataset, samples_per_gpu, world_size, rank, seed=seed)
            else:
                if val:
                    sampler = CustomDistributedSampler(
                        dataset, world_size, rank, shuffle=False, seed=seed)
                else:
                    sampler = DistributedSampler(
                        dataset, world_size, rank, shuffle=False, seed=seed)
        else:
            sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None

        batch_sampler = None

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    pin_memory = kwargs.pop('pin_memory', False)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader
