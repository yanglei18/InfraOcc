# Copyright (c) OpenMMLab. All rights reserved.
import collections
import os
import time

from mmcv.utils import build_from_cfg

from mmdet.datasets.builder import PIPELINES as MMDET_PIPELINES
from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose:
    """Compose multiple transforms sequentially. The pipeline registry of
    mmdet3d separates with mmdet, however, sometimes we may need to use mmdet's
    pipeline. So the class is rewritten to be able to use pipelines from both
    mmdet3d and mmdet.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                _, key = PIPELINES.split_scope_key(transform['type'])
                if key in PIPELINES._module_dict.keys():
                    transform = build_from_cfg(transform, PIPELINES)
                else:
                    transform = build_from_cfg(transform, MMDET_PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')
        # A process-local, opt-in timing probe for DataLoader workers. It is
        # deliberately controlled by an environment flag rather than config,
        # so ordinary data loading has no extra work or altered ordering.
        self._profile_pipeline = os.getenv('STCROADOCC_PIPELINE_PROFILE') == '1'
        self._profile_limit = int(
            os.getenv('STCROADOCC_PIPELINE_PROFILE_SAMPLES', '8'))
        self._profile_count = 0
        self._profile_totals = collections.defaultdict(float)

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            start_time = time.perf_counter() if self._profile_pipeline else None
            data = t(data)
            if self._profile_pipeline:
                self._profile_totals[type(t).__name__] += (
                    time.perf_counter() - start_time)
            if data is None:
                return None
        if self._profile_pipeline:
            self._profile_count += 1
            if self._profile_count == self._profile_limit:
                summary = ', '.join(
                    '{}={:.1f}ms'.format(name, seconds * 1000.0 /
                                          self._profile_count)
                    for name, seconds in sorted(
                        self._profile_totals.items(),
                        key=lambda item: item[1], reverse=True))
                print('[pipeline-profile pid={}] average over {} samples: {}'.
                      format(os.getpid(), self._profile_count, summary),
                      flush=True)
                self._profile_pipeline = False
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
