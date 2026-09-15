# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook

from mmdet3d.core.hook.utils import is_parallel

__all__ = ['VisualizationContextHook']


@HOOKS.register_module()
class VisualizationContextHook(Hook):

    def _unwrap_model(self, runner):
        model = runner.model
        if is_parallel(model):
            return model.module
        return model

    def _set_epoch(self, runner):
        model = self._unwrap_model(runner)
        if hasattr(model, 'set_visualization_epoch'):
            model.set_visualization_epoch(runner.epoch + 1)
        elif hasattr(model, '_set_visualization_epoch'):
            model._set_visualization_epoch(runner.epoch + 1)
        else:
            setattr(model, 'current_visualization_epoch', runner.epoch + 1)

    def before_run(self, runner):
        self._set_epoch(runner)

    def before_train_epoch(self, runner):
        self._set_epoch(runner)

    def before_val_epoch(self, runner):
        self._set_epoch(runner)
