# Copyright (c) OpenMMLab. All rights reserved.
# modified from megvii-bevdepth.
import math
import os
import re
from copy import deepcopy

import torch
from mmcv.runner import load_state_dict
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook

from mmdet3d.core.hook.utils import is_parallel

__all__ = ['ModelEMA']


class ModelEMA:
    """Model Exponential Moving Average from https://github.com/rwightman/
    pytorch-image-models Keep a moving average of everything in the model
    state_dict (parameters and buffers).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/
    ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training
    schemes to perform well.
    This class is sensitive where it is initialized in the sequence
    of model init, GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema_model = deepcopy(model).eval()
        self.ema = self.ema_model.module.module if is_parallel(
            self.ema_model.module) else self.ema_model.module
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, trainer, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(
                model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()


@HOOKS.register_module()
class MEGVIIEMAHook(Hook):
    """EMAHook used in BEVDepth.

    Modified from https://github.com/Megvii-Base
    Detection/BEVDepth/blob/main/callbacks/ema.py.
    """

    def __init__(self,
                 init_updates=0,
                 init_updates_epoch=None,
                 decay=0.9990,
                 resume=None,
                 interval=-1,
                 max_keep_ckpts=None):
        super().__init__()
        self.init_updates = init_updates
        self.init_updates_epoch = init_updates_epoch
        self.resume = resume
        self.decay = decay
        self.interval = interval
        self.max_keep_ckpts = max_keep_ckpts
        self._resolved_max_keep_ckpts = max_keep_ckpts
        self._checkpoint_hook = None

    def _resolve_init_updates(self, runner):
        if self.init_updates_epoch is None:
            return int(self.init_updates)

        iters_per_epoch = None
        max_iters = getattr(runner, 'max_iters', None)
        max_epochs = getattr(runner, 'max_epochs', None)
        if max_iters is not None and max_epochs:
            iters_per_epoch = float(max_iters) / float(max_epochs)
        else:
            data_loader = getattr(runner, 'data_loader', None)
            if data_loader is not None:
                try:
                    iters_per_epoch = float(len(data_loader))
                except TypeError:
                    iters_per_epoch = None

        if iters_per_epoch is None or iters_per_epoch <= 0:
            runner.logger.warning(
                'MEGVIIEMAHook cannot resolve init_updates_epoch; '
                f'falling back to init_updates={self.init_updates}.')
            return int(self.init_updates)

        init_updates = int(float(self.init_updates_epoch) * iters_per_epoch)
        runner.logger.info(
            'MEGVIIEMAHook resolved init_updates_epoch='
            f'{self.init_updates_epoch} to init_updates={init_updates}.')
        return init_updates

    def before_run(self, runner):
        from torch.nn.modules.batchnorm import SyncBatchNorm

        bn_model_list = list()
        bn_model_dist_group_list = list()
        for model_ref in runner.model.modules():
            if isinstance(model_ref, SyncBatchNorm):
                bn_model_list.append(model_ref)
                bn_model_dist_group_list.append(model_ref.process_group)
                model_ref.process_group = None
        runner.ema_model = ModelEMA(runner.model, self.decay)

        for bn_model, dist_group in zip(bn_model_list,
                                        bn_model_dist_group_list):
            bn_model.process_group = dist_group
        runner.ema_model.updates = self._resolve_init_updates(runner)

        if self.resume is not None:
            runner.logger.info(f'resume ema checkpoint from {self.resume}')
            cpt = torch.load(self.resume, map_location='cpu')
            load_state_dict(runner.ema_model.ema, cpt['state_dict'])
            runner.ema_model.updates = cpt['updates']

        self._checkpoint_hook = self._resolve_checkpoint_hook(runner)
        self._resolved_max_keep_ckpts = self._resolve_max_keep_ckpts(runner)

    def after_train_iter(self, runner):
        runner.ema_model.update(runner, runner.model.module)
        curr_step = runner.iter
        if self.interval>0:
            if curr_step % self.interval==0 and curr_step>0:
                self.save_checkpoint_iter(runner)

    def after_run(self, runner):
        # Keep the final iter EMA checkpoint only when iter-based saving is
        # explicitly enabled. For the common interval=-1 setup we already save
        # epoch_k_ema.pth in after_train_epoch, so writing iter_xxx_ema.pth at
        # shutdown is redundant.
        if self.interval > 0:
            self.save_checkpoint_iter(runner)

    def after_train_epoch(self, runner):
        if self.interval > 0:
            return
        if self._should_save_epoch_checkpoint(runner):
            self.save_checkpoint(runner)

    @master_only
    def save_checkpoint(self, runner):
        state_dict = runner.ema_model.ema.state_dict()
        ema_checkpoint = {
            'epoch': runner.epoch,
            'state_dict': state_dict,
            'updates': runner.ema_model.updates
        }
        save_path = f'epoch_{runner.epoch+1}_ema.pth'
        save_path = os.path.join(runner.work_dir, save_path)
        torch.save(ema_checkpoint, save_path)
        self._cleanup_old_ema_checkpoints(runner, by_epoch=True)
        runner.logger.info(f'Saving ema checkpoint at {save_path}')

    @master_only
    def save_checkpoint_iter(self, runner):
        state_dict = runner.ema_model.ema.state_dict()
        ema_checkpoint = {
            'iter': runner.iter,
            'state_dict': state_dict,
            'updates': runner.ema_model.updates
        }
        save_path = f'iter_{runner.iter}_ema.pth'
        save_path = os.path.join(runner.work_dir, save_path)
        torch.save(ema_checkpoint, save_path)
        self._cleanup_old_ema_checkpoints(runner, by_epoch=False)
        runner.logger.info(f'Saving ema checkpoint at {save_path}')

    def _resolve_checkpoint_hook(self, runner):
        for hook in runner.hooks:
            if hook is self:
                continue
            if hook.__class__.__name__ == 'CheckpointHook':
                return hook
        return None

    def _should_save_epoch_checkpoint(self, runner):
        if self._checkpoint_hook is None:
            return True
        if not getattr(self._checkpoint_hook, 'by_epoch', True):
            return False

        interval = getattr(self._checkpoint_hook, 'interval', -1)
        save_last = getattr(self._checkpoint_hook, 'save_last', True)
        return self.every_n_epochs(runner, interval) or \
            (save_last and self.is_last_epoch(runner))

    def _resolve_max_keep_ckpts(self, runner):
        if self.max_keep_ckpts is not None:
            return self.max_keep_ckpts

        checkpoint_hook = self._checkpoint_hook
        if checkpoint_hook is None:
            checkpoint_hook = self._resolve_checkpoint_hook(runner)
        if checkpoint_hook is None:
            return -1

        return getattr(checkpoint_hook, 'max_keep_ckpts', -1)

    def _cleanup_old_ema_checkpoints(self, runner, by_epoch=True):
        if self._resolved_max_keep_ckpts is None or \
                self._resolved_max_keep_ckpts <= 0:
            return

        filename_pattern = (r'^epoch_(\d+)_ema\.pth$'
                            if by_epoch else r'^iter_(\d+)_ema\.pth$')
        filename_regex = re.compile(filename_pattern)
        ema_checkpoints = []
        for filename in os.listdir(runner.work_dir):
            match = filename_regex.match(filename)
            if match is None:
                continue
            ema_checkpoints.append((int(match.group(1)), filename))

        if by_epoch:
            normal_regex = re.compile(r'^epoch_(\d+)\.pth$')
            normal_checkpoint_ids = {
                int(match.group(1))
                for filename in os.listdir(runner.work_dir)
                for match in [normal_regex.match(filename)]
                if match is not None
            }
            if normal_checkpoint_ids:
                matched_ema_checkpoints = []
                for step, filename in ema_checkpoints:
                    if step in normal_checkpoint_ids:
                        matched_ema_checkpoints.append((step, filename))
                        continue
                    checkpoint_path = os.path.join(runner.work_dir, filename)
                    if os.path.isfile(checkpoint_path):
                        os.remove(checkpoint_path)
                        runner.logger.info(
                            f'Removed unmatched ema checkpoint {checkpoint_path}')
                ema_checkpoints = matched_ema_checkpoints

        if len(ema_checkpoints) <= self._resolved_max_keep_ckpts:
            return

        ema_checkpoints.sort(key=lambda item: item[0])
        stale_checkpoints = ema_checkpoints[:-self._resolved_max_keep_ckpts]
        for _, filename in stale_checkpoints:
            checkpoint_path = os.path.join(runner.work_dir, filename)
            if not os.path.isfile(checkpoint_path):
                continue
            os.remove(checkpoint_path)
            runner.logger.info(f'Removed ema checkpoint {checkpoint_path}')
