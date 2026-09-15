# Copyright (c) OpenMMLab. All rights reserved.
"""Checkpoint adaptation hooks for RoadOcc."""

import torch
import torch.distributed as dist

from mmcv.runner.hooks import HOOKS, Hook

from mmdet3d.core.hook.utils import is_parallel

__all__ = ['ResetCanonicalPTRControllerHook']


@HOOKS.register_module()
class ResetCanonicalPTRControllerHook(Hook):
    """Discard a legacy P/T/R router after loading an occupancy checkpoint.

    ``load_from`` restores all occupancy parameters, but the old router may
    have been optimized with a non-class-balanced objective.  Resetting only
    that module preserves the validated occupancy endpoint and creates the
    neutral P/T/R initialization required by the current route objective.

    A real ``resume_from`` restores a positive runner epoch before
    ``before_run``.  In that case the controller is part of the learned
    training state and must not be reset again.  A continuation that uses
    ``load_from`` can likewise opt out explicitly with
    ``reset_loaded_controller=False``.  This distinction is important: a
    legacy occupancy checkpoint needs a neutral route head, whereas a RoadOcc
    checkpoint already contains the learned P/T/R controller.
    """

    def __init__(self, reset_loaded_controller=True):
        self.reset_loaded_controller = bool(reset_loaded_controller)

    @staticmethod
    def _model(runner):
        return runner.model.module if is_parallel(
            runner.model) else runner.model

    @staticmethod
    def _sync(module):
        if not (dist.is_available() and dist.is_initialized()):
            return
        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor, src=0)

    def before_run(self, runner):
        if runner.epoch > 0 or not self.reset_loaded_controller:
            reason = ('while resuming from epoch %d' % runner.epoch
                      if runner.epoch > 0 else
                      'because reset_loaded_controller is disabled')
            runner.logger.info(
                'Preserve canonical PTR controller %s.', reason)
            return
        model = self._model(runner)
        controller = getattr(model, 'canonical_ptr_controller', None)
        if controller is None:
            raise RuntimeError('Canonical PTR controller is not available.')
        controller.reset_parameters()
        self._sync(controller)
        runner.logger.info(
            'Reset canonical PTR controller after checkpoint loading; '
            'occupancy parameters remain intact.')
