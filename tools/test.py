# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from os import path as osp
import warnings
import sys

# 抑制测试时常见第三方/库内 UserWarning，保持日志简洁
warnings.filterwarnings('ignore', message='Setting MKL_NUM_THREADS')
warnings.filterwarnings('ignore', message='.*meshgrid.*')
warnings.filterwarnings('ignore', message='.*BaseTransformerLayer.*deprecated.*')
warnings.filterwarnings('ignore', message='.*embed_dims in MultiScaleDeformAttention.*')

# 保证能 import projects.xxx（与 dist_train.sh 里 PYTHONPATH=.. 等价）
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test, multi_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from torch.nn.modules.batchnorm import _BatchNorm

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', type=str, default="projects/RoadOcc/configs/roadocc_m_4x4_24e.py", help='test config file path')
    parser.add_argument('--checkpoint', type=str, default="work_dirs/roadocc_m_4x4_24e/epoch_6.pth", help='checkpoint file')
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='output result file in pickle format. '
        'If not set, defaults to <work_dir>/results.pkl')
    parser.add_argument(
        '--compact-out',
        action='store_true',
        help=('serialize only index and semantic occupancy predictions. '
              'Full in-memory outputs remain available to evaluation.'))
    parser.add_argument(
        '--compact-keep-diagnostics',
        action='store_true',
        help=('with --compact-out, retain lightweight transition diagnostics '
              'while still dropping dense flow and visualization payloads'))
    parser.add_argument(
        '--work-dir',
        type=str,
        default=None,
        help='directory for test artifacts. '
        'Default: work_dirs/<config_name>')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--broadcast-bn-buffer',
        action='store_true',
        help=('broadcast rank-0 BatchNorm running statistics before testing; '
              'this reproduces the distributed training EvalHook protocol'))
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['bbox'],
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--no-aavt',
        action='store_true',
        help='Do not align after view transformer.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # torchrun / torch.distributed.launch 传 --local-rank（连字符），需与 --local_rank 等价
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    args.eval_explicit = '--eval' in sys.argv[1:]
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def compact_occ_outputs_for_serialization(outputs, keep_diagnostics=False):
    """Drop physical/diagnostic payloads from an occupancy audit copy.

    Dataset evaluation still consumes ``outputs`` itself.  The returned list
    is only the persisted semantic artifact used by post-hoc error analysis.
    """
    compact_outputs = []
    for output in outputs:
        if not isinstance(output, dict):
            raise TypeError('Compact occupancy output requires dict results.')
        missing = {'index', 'occ_results'} - set(output)
        if missing:
            raise KeyError(
                f'Compact occupancy output is missing keys: {sorted(missing)}')
        compact_output = {
            'index': output['index'],
            'occ_results': output['occ_results'],
        }
        if keep_diagnostics and output.get('transition_gate_diagnostics'):
            compact_output['transition_gate_diagnostics'] = output[
                'transition_gate_diagnostics']
        compact_outputs.append(compact_output)
    return compact_outputs


def _maybe_adjust_dist_backend(args, cfg):
    """Optionally force torch.distributed GLOO (see use_gloo_dist / env)."""
    if args.launcher == 'none' or not torch.cuda.is_available():
        return

    if 'dist_params' not in cfg:
        cfg.dist_params = dict(backend='nccl')

    backend_override = os.environ.get('OCC_V2X_DIST_BACKEND', '').strip().lower()
    want_gloo = cfg.get('use_gloo_dist', False) or backend_override == 'gloo'
    force_nccl = cfg.get('force_nccl_dist', False) or backend_override == 'nccl'
    if (not want_gloo and not force_nccl
            and cfg.dist_params.get('backend', 'nccl') == 'nccl'):
        major, minor = torch.cuda.get_device_capability(0)
        if major >= 12:
            cfg.dist_params['backend'] = 'gloo'
            warnings.warn(
                'Auto-switching distributed backend to GLOO on compute '
                f'capability {major}.{minor} GPUs because NCCL reproduces '
                'illegal memory access on this stack. Set '
                'OCC_V2X_DIST_BACKEND=nccl or force_nccl_dist=True to force '
                'NCCL.',
                UserWarning)
            return

    if want_gloo:
        cfg.dist_params['backend'] = 'gloo'
        warnings.warn(
            'Distributed backend is GLOO (use_gloo_dist=True or '
            'OCC_V2X_DIST_BACKEND=gloo).',
            UserWarning)


def _infer_eval_metrics(args, cfg):
    if args.eval_explicit or args.eval != ['bbox']:
        return

    eval_metric = cfg.get('eval_metric', None)
    if eval_metric is None:
        data_cfg = cfg.get('data', {}).get('test', {})
        if isinstance(data_cfg, dict):
            eval_metric = data_cfg.get('eval_metric', None)

    if eval_metric:
        args.eval = [eval_metric]


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    _infer_eval_metrics(args, cfg)

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join(
            './work_dirs', osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    if args.out is None:
        args.out = osp.join(cfg.work_dir, 'results.pkl')

    cfg = compat_cfg(cfg)
    _maybe_adjust_dist_backend(args, cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    figures_path = osp.join(cfg.work_dir, 'figures_path')
    os.makedirs(figures_path, exist_ok=True)
    project_name = osp.basename(osp.dirname(osp.dirname(args.config)))
    if isinstance(cfg.model, dict):
        cfg.model.update(
            meta_info=dict(
                figures_path=figures_path,
                project_name=project_name,
                checkpoint_path=args.checkpoint))

    samples_per_gpu = cfg.data.get('test_dataloader', {}).get('samples_per_gpu', 1)
    test_dataloader_default_args = dict(
        samples_per_gpu=samples_per_gpu, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir, batch_size=samples_per_gpu)
    else:
        model = MMDistributedDataParallel(model.cuda(),device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        if args.broadcast_bn_buffer:
            for module in model.modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect, batch_size=samples_per_gpu)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out and not os.path.exists(args.out):
            print(f'\nwriting results to {args.out}')
            persisted_outputs = (
                compact_occ_outputs_for_serialization(
                    outputs, keep_diagnostics=args.compact_keep_diagnostics)
                if args.compact_out else outputs)
            mmcv.dump(persisted_outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
