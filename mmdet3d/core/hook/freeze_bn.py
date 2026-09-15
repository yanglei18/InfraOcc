# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from torch.nn.modules.batchnorm import _BatchNorm


@HOOKS.register_module()
class FreezeBNHook(Hook):
    """Keep BatchNorm layers in eval mode during training.

    Some models call ``train()`` at the start of every training epoch, which
    resets BatchNorm layers to training mode even when a config sets
    ``norm_eval=True`` on a nested module. This hook enforces eval mode after
    that transition and before every train iteration.
    """

    def __init__(self, freeze_affine=False):
        self.freeze_affine = freeze_affine
        self.bn_modules = None

    def _model(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, 'module'):
            model = model.module
        return model

    def before_run(self, runner):
        self.bn_modules = [
            module for module in self._model(runner).modules()
            if isinstance(module, _BatchNorm)
        ]
        self._freeze_bn(runner)

    def _freeze_bn(self, runner):
        if self.bn_modules is None:
            self.before_run(runner)
            return
        for module in self.bn_modules:
            module.eval()
            if self.freeze_affine:
                for param in module.parameters(recurse=False):
                    param.requires_grad = False

    def before_train_epoch(self, runner):
        self._freeze_bn(runner)

    def before_train_iter(self, runner):
        self._freeze_bn(runner)
