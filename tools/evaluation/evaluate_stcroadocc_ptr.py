"""Evaluate native-scale canonical P/T/R routing on the validation set.

The metric support and argmax convention match the canonical P/T/R training
objective: every valid GT dynamic voxel contributes exactly once, and the
reported table is computed from one dataset-level 3x3 confusion matrix.
"""

import argparse
import json
import os
import os.path as osp
import sys

import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmdet.apis import set_random_seed

_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mmdet3d.datasets import build_dataloader, build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate native-scale STCRoadOcc P/T/R routing')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    return args


def unwrap_augmentation_tensor(data, key, device):
    value = data.get(key)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                f'{key} must contain exactly one test augmentation, got '
                f'{len(value)}.')
        value = value[0]
    if hasattr(value, 'data') and not torch.is_tensor(value):
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not torch.is_tensor(value):
        raise TypeError(f'{key} must resolve to a tensor, got {type(value)}.')
    return value.to(device=device, non_blocking=True)


def update_confusion(confusion, detector, data):
    logits_collection = detector.latest_canonical_ptr_gate_logits
    if torch.is_tensor(logits_collection):
        logits_collection = {'1_1': logits_collection}
    logits = logits_collection.get('1_1')
    if logits is None:
        raise RuntimeError('The model did not produce native-scale PTR logits.')
    gate = torch.softmax(logits.float(), dim=1).permute(
        0, 4, 3, 2, 1).contiguous()
    device = gate.device
    target, weight = detector._build_canonical_ptr_target(
        gate,
        unwrap_augmentation_tensor(data, 'voxel_semantic', device),
        unwrap_augmentation_tensor(data, 'voxel_occflows', device),
        unwrap_augmentation_tensor(
            data, 'ptr_previous_semantic_1_1', device),
        unwrap_augmentation_tensor(
            data, 'ptr_previous_visible_1_1', device),
        detector.canonical_ptr_controller,
        voxel_size=float(detector.canonical_ptr_controller.voxel_size),
    )
    if target is None or weight is None:
        return
    supervised = (target.sum(dim=-1) > 0) & (weight > 0)
    if not torch.any(supervised):
        return
    predicted_state = gate.argmax(dim=-1)[supervised]
    target_state = target.argmax(dim=-1)[supervised]
    encoded = target_state.to(torch.int64) * 3 + predicted_state.to(
        torch.int64)
    confusion += torch.bincount(encoded, minlength=9).reshape(3, 3)


def summarize(confusion):
    confusion = confusion.to(torch.float64)
    target_count = confusion.sum(dim=1)
    predicted_count = confusion.sum(dim=0)
    correct = confusion.diag()
    total = confusion.sum()
    union = target_count + predicted_count - correct
    state_names = ('Persist', 'Transport', 'Refresh')
    rows = []
    for index, state_name in enumerate(state_names):
        rows.append({
            'state': state_name,
            'target_percent':
            float(target_count[index] / total * 100.0),
            'predicted_percent':
            float(predicted_count[index] / total * 100.0),
            'precision_percent':
            float(correct[index] / predicted_count[index] * 100.0),
            'recall_percent':
            float(correct[index] / target_count[index] * 100.0),
            'iou_percent':
            float(correct[index] / union[index] * 100.0),
        })
    return {
        'support': 'valid GT dynamic voxels',
        'scale': '1_1',
        'aggregation': 'dataset-level argmax confusion matrix',
        'total_voxels': int(total.item()),
        'confusion_target_rows_predicted_columns': [
            [int(value) for value in row] for row in confusion.tolist()
        ],
        'rows': rows,
    }


def print_table(summary):
    print('\nState       Target    Pred.  Precision  Recall     IoU')
    for row in summary['rows']:
        print('{state:<10s} {target_percent:7.2f} {predicted_percent:7.2f} '
              '{precision_percent:10.2f} {recall_percent:7.2f} '
              '{iou_percent:7.2f}'.format(**row))
    print(f"Valid GT dynamic voxels: {summary['total_voxels']}")


def infer_non_padded_batch_count(data_loader, rank):
    """Return this rank's real sample count for the temporal eval sampler."""
    sampler = data_loader.batch_sampler
    required = ('_group_order', 'global_batch_size', 'batch_size',
                'group_idx_to_sample_idxs')
    if not all(hasattr(sampler, name) for name in required):
        return len(data_loader)
    if int(sampler.batch_size) != 1:
        raise ValueError('PTR evaluation requires one sample per GPU.')
    global_slot_index = int(rank) * int(sampler.batch_size)
    group_order = sampler._group_order()
    assigned_groups = group_order[
        global_slot_index::int(sampler.global_batch_size)]
    return sum(
        len(sampler.group_idx_to_sample_idxs[group_index])
        for group_index in assigned_groups)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    set_random_seed(args.seed, deterministic=True)
    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)
    rank, world_size = get_dist_info()

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.val.test_mode = True
    dataset = build_dataset(cfg.data.val)
    loader_options = dict(
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        seed=args.seed,
        runner_type='EpochBasedRunnerEval',
        val=True,
        pin_memory=True,
    )
    data_loader = build_dataloader(dataset, **loader_options)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model.save_flow_results = False
    model.eval()
    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
    else:
        model = MMDataParallel(model.cuda(), device_ids=[0])

    detector = model.module
    non_padded_batches = infer_non_padded_batch_count(data_loader, rank)
    print(
        f'PTR evaluation rank {rank}: {non_padded_batches} unique samples, '
        f'{len(data_loader) - non_padded_batches} sampler padding batches.',
        flush=True)
    confusion = torch.zeros(
        (3, 3), dtype=torch.int64, device=torch.cuda.current_device())
    with torch.no_grad():
        for iteration, data in enumerate(data_loader, 1):
            model(return_loss=False, rescale=True, **data)
            # The temporal eval sampler pads shorter sequence slots with their
            # last sample so all DDP ranks execute the same number of forwards.
            # Keep the forward for synchronization/state parity, but never
            # count those repeated samples in the dataset-level table.
            if iteration <= non_padded_batches:
                update_confusion(confusion, detector, data)
            if rank == 0 and (iteration == 1 or iteration % 100 == 0):
                print(
                    f'PTR evaluation: rank-0 batch {iteration}/'
                    f'{len(data_loader)}', flush=True)

    if distributed:
        dist.all_reduce(confusion, op=dist.ReduceOp.SUM)
    if rank != 0:
        return

    summary = summarize(confusion.cpu())
    summary.update({
        'config': osp.abspath(args.config),
        'checkpoint': osp.abspath(args.checkpoint),
        'dataset_samples': len(dataset),
        'world_size': world_size,
    })
    output_directory = osp.dirname(osp.abspath(args.output))
    os.makedirs(output_directory, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write('\n')
    print_table(summary)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
