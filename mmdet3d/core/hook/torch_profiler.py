"""Opt-in, short-window PyTorch profiling for real training runs."""

import os

import torch
import torch.distributed as dist
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class TorchProfilerHook(Hook):
    """Capture a bounded CPU/CUDA trace without changing training semantics.

    The hook is only injected by ``tools/train.py`` when the explicit
    ``STCROADOCC_PROFILE_TRAIN`` environment flag is set. It deliberately
    neither changes a config value nor stops the runner; callers may interrupt
    training once the report has been written.
    """

    def __init__(self, output_dir, wait_steps=2, warmup_steps=2,
                 active_steps=4):
        self.output_dir = output_dir
        self.wait_steps = int(wait_steps)
        self.warmup_steps = int(warmup_steps)
        self.active_steps = int(active_steps)
        self._profiler = None
        self._steps = 0
        self._written = False
        self._stopped = False

    @staticmethod
    def _rank():
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    def before_run(self, runner):
        os.makedirs(self.output_dir, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.wait_steps,
                warmup=self.warmup_steps,
                active=self.active_steps,
                repeat=1),
            # The first detailed run proved that shape/memory tracing makes a
            # multi-GPU Chrome trace exceed 1 GB. Scoped timings are enough
            # for the engineering decision and keep this diagnostic light.
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        self._install_scopes(runner)
        self._profiler.start()
        runner.logger.info(
            'Torch profiler enabled for %d warm-up/profiled train steps.',
            self.wait_steps + self.warmup_steps + self.active_steps)

    @staticmethod
    def _profile_method(instance, name, label):
        method = getattr(instance, name, None)
        if method is None or getattr(method, '_torch_profiler_wrapped', False):
            return

        def profiled(*args, **kwargs):
            with torch.autograd.profiler.record_function(label):
                return method(*args, **kwargs)

        profiled._torch_profiler_wrapped = True
        setattr(instance, name, profiled)

    def _install_scopes(self, runner):
        model = getattr(runner.model, 'module', runner.model)
        for name, label in (
                ('forward_train', 'STCRoadOcc/forward_train'),
                ('obtain_voxel_feats', 'STCRoadOcc/obtain_voxel_feats'),
                ('run_stage_decoder', 'STCRoadOcc/stage_decoders'),
                ('_forward_occupancy_with_detection',
                 'STCRoadOcc/occupancy_and_detection'),
                ('_build_training_losses', 'STCRoadOcc/training_losses')):
            self._profile_method(model, name, label)
        if getattr(model, 'forward_projection', None) is not None:
            self._profile_method(model.forward_projection, 'extract_img_feat',
                                 'STCRoadOcc/image_to_bev')
        if getattr(model, 'flow_head', None) is not None:
            self._profile_method(model.flow_head, 'forward',
                                 'STCRoadOcc/flow_head')

    def _write_report(self, runner):
        if self._written or self._profiler is None:
            return
        rank = self._rank()
        prefix = os.path.join(self.output_dir, 'rank{:d}'.format(rank))
        if not self._stopped:
            self._profiler.stop()
            self._stopped = True
        averages = self._profiler.key_averages()
        with open(prefix + '.txt', 'w') as report:
            report.write('CUDA self time\n')
            report.write(averages.table(
                sort_by='self_cuda_time_total', row_limit=120))
            report.write('\n\nCPU self time\n')
            report.write(averages.table(
                sort_by='self_cpu_time_total', row_limit=120))
        if rank == 0:
            runner.logger.info('Torch profiler report written to %s.txt',
                               prefix)
        self._written = True

    def after_train_iter(self, runner):
        if self._profiler is None:
            return
        self._profiler.step()
        self._steps += 1
        if self._steps == (self.wait_steps + self.warmup_steps +
                           self.active_steps):
            self._write_report(runner)

    def after_run(self, runner):
        if self._profiler is None:
            return
        self._write_report(runner)
        if not self._stopped:
            self._profiler.stop()
            self._stopped = True
