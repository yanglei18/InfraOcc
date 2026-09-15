# Copyright (c) OpenMMLab. All rights reserved.
import gc
import inspect
from os import path as osp
import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmcv.utils import build_from_cfg
from torch import distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet3d.core import build_optimizer
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.custom_builder import build_dataloader as mmdet3d_build_dataloader
from mmdet3d.apis.test import multi_gpu_test as mmdet3d_multi_gpu_test
from mmdet3d.apis.test import single_gpu_test as mmdet3d_single_gpu_test
from mmdet3d.utils import find_latest_checkpoint
from mmdet.core import DistEvalHook as MMDET_DistEvalHook
from mmdet.core import EvalHook as MMDET_EvalHook
from mmdet.datasets import replace_ImageToTensor
from mmdet.utils import get_root_logger as get_mmdet_root_logger
from mmseg.core import DistEvalHook as MMSEG_DistEvalHook
from mmseg.core import EvalHook as MMSEG_EvalHook
from mmseg.datasets import build_dataloader as build_mmseg_dataloader
from mmseg.utils import get_root_logger as get_mmseg_root_logger


def _build_ddp_kwargs(cfg, logger, find_unused_parameters):
    ddp_kwargs = dict(
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters)
    signature = inspect.signature(MMDistributedDataParallel.__init__)
    ddp_static_graph = cfg.get('ddp_static_graph', False)
    if ddp_static_graph:
        if 'static_graph' not in signature.parameters:
            raise RuntimeError(
                'ddp_static_graph=True requires DistributedDataParallel '
                'static_graph support in the installed PyTorch version.')
        ddp_kwargs['static_graph'] = True
        logger.warning(
            'Enabling DDP static_graph for a fixed execution graph with '
            'activation checkpointing and intentionally unused parameters.')
    if 'init_sync' not in signature.parameters:
        return ddp_kwargs

    ddp_init_sync = cfg.get('ddp_init_sync', None)
    if ddp_init_sync is None:
        major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
        ddp_init_sync = major < 12 or cfg.get('diff_seed', False)

    if not ddp_init_sync:
        logger.warning(
            'Disabling DDP init_sync on this GPU to avoid illegal memory '
            'access during the initial parameter broadcast.')
    ddp_kwargs['init_sync'] = ddp_init_sync
    return ddp_kwargs


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, optional): The seed. Default to None.
        device (str, optional): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


class MMDET3D_EvalHook(MMDET_EvalHook):

    def __init__(self, *args, use_ema=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_ema = use_ema

    def _evaluation_model(self, runner):
        if not self.use_ema:
            return runner.model
        if not hasattr(runner, 'ema_model'):
            raise RuntimeError(
                'use_ema=True requires an EMA hook to initialize '
                'runner.ema_model before evaluation.')
        return runner.ema_model.ema_model

    def _do_evaluate(self, runner):
        if not self._should_evaluate(runner):
            return

        results = mmdet3d_single_gpu_test(
            self._evaluation_model(runner), self.dataloader, show=False)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)
        # Occupancy-flow predictions are several gigabytes for a full split.
        # They are consumed synchronously by ``evaluate`` and have no users
        # afterwards, so do not retain them on the hook between evaluations.
        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MMDET3D_DistEvalHook(MMDET_DistEvalHook):

    def __init__(self, *args, use_ema=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_ema = use_ema

    def _evaluation_model(self, runner):
        if not self.use_ema:
            return runner.model
        if not hasattr(runner, 'ema_model'):
            raise RuntimeError(
                'use_ema=True requires an EMA hook to initialize '
                'runner.ema_model before evaluation.')
        return runner.ema_model.ema_model

    def _do_evaluate(self, runner):
        model = self._evaluation_model(runner)
        if self.broadcast_bn_buffer:
            for _, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = mmdet3d_multi_gpu_test(
            model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)

        # ``collect_results_cpu`` returns the complete occupancy/flow split on
        # rank 0.  Retaining it as ``latest_results`` leaked many gigabytes of
        # host memory per evaluation, while CUDA's validation cache also left
        # little headroom for the following training epoch.  Evaluation is
        # synchronous, so release both once every rank has finished using it.
        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_mmseg_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_mmseg_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(model.cuda(), **_build_ddp_kwargs(
            cfg, logger, find_unused_parameters))
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_mmseg_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = MMSEG_DistEvalHook if distributed else MMSEG_EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        if cfg.revise_keys is not None:
            runner.load_checkpoint(cfg.load_from)
        else:
            runner.load_checkpoint(cfg.load_from, revise_keys=cfg.revise_keys)
    runner.run(data_loaders, cfg.workflow)


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_mmdet_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']
    train_dataloader_default_args = dict(
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=cfg.data.get('persistent_workers', False),
        pin_memory=cfg.data.get('pin_memory', False))
    train_dataloader_cfg = dict(cfg.data.get('train_dataloader', {}))
    train_dataloader_default_args.update(train_dataloader_cfg)
    data_loaders = [
        mmdet3d_build_dataloader(
            ds,
            **train_dataloader_default_args)
        for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(model.cuda(), **_build_ddp_kwargs(
            cfg, logger, find_unused_parameters))
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        val_loader_cfg = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
        # Backward-compat: keep supporting legacy placement under data.val.
        if isinstance(cfg.data.val, dict) and 'samples_per_gpu' in cfg.data.val:
            val_loader_cfg['samples_per_gpu'] = cfg.data.val.pop(
                'samples_per_gpu')

        if isinstance(cfg.data.val, dict):
            if val_loader_cfg.get('samples_per_gpu', 1) > 1:
                cfg.data.val.pipeline = replace_ImageToTensor(
                    cfg.data.val.pipeline)
        elif isinstance(cfg.data.val, list):
            if val_loader_cfg.get('samples_per_gpu', 1) > 1:
                for ds_cfg in cfg.data.val:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = mmdet3d_build_dataloader(val_dataset, **val_loader_cfg)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_cfg['runner'] = runner
        eval_hook = MMDET3D_DistEvalHook if distributed else MMDET3D_EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)

    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        if cfg.revise_keys is not None:
            runner.load_checkpoint(cfg.load_from, revise_keys=cfg.revise_keys)
        else:
            runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    if cfg.model.type in ['EncoderDecoder3D']:
        train_segmentor(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
    else:
        train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
