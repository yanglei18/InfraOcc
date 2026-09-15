# Copyright (c) OpenMMLab. All rights reserved.
import pickle
import shutil
import tempfile
import time
from itertools import zip_longest
from os import path as osp

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet3d.models import Base3DDetector
from mmdet.core import encode_mask_results


def _normalize_result_list(result):
    """Normalize model outputs to list format for downstream collection."""
    if isinstance(result, list):
        return result
    if isinstance(result, tuple):
        return [result]
    if isinstance(result, dict):
        return [result]
    return [result]


def _to_sample_index_list(index_value):
    """Normalize result indices into a Python list."""
    if isinstance(index_value, list):
        return index_value
    if isinstance(index_value, tuple):
        return list(index_value)
    if torch.is_tensor(index_value):
        index_value = index_value.detach().cpu().tolist()
    elif hasattr(index_value, 'tolist') and not isinstance(index_value, (str, bytes)):
        index_value = index_value.tolist()

    if isinstance(index_value, list):
        return index_value
    return [index_value]


def _slice_sample_aligned_value(value, keep_indices, batch_size):
    """Slice values whose leading dimension matches the sample batch size."""
    if isinstance(value, list):
        return [value[i] for i in keep_indices] if len(value) == batch_size else value
    if isinstance(value, tuple):
        return tuple(value[i] for i in keep_indices) if len(value) == batch_size else value
    if torch.is_tensor(value):
        return value[keep_indices] if value.ndim > 0 and value.size(0) == batch_size else value
    if hasattr(value, 'shape'):
        try:
            if len(value) == batch_size:
                return value[keep_indices]
        except TypeError:
            pass
    return value


def _trim_results_by_sample_index(ordered_results, size):
    """Trim collected results by unique sample ids when available.

    Sequence-preserving eval samplers may pad shorter slots with repeated tail
    samples. Raw list truncation can then drop real samples while keeping these
    padded duplicates. When results expose per-sample ``index`` values, keep the
    first ``size`` unique sample ids instead.
    """
    if not ordered_results:
        return ordered_results

    first_result = ordered_results[0]
    if not isinstance(first_result, dict) or 'index' not in first_result:
        return ordered_results[:size]

    seen_indices = set()
    trimmed_results = []
    for result in ordered_results:
        if not isinstance(result, dict) or 'index' not in result:
            continue

        sample_indices = _to_sample_index_list(result['index'])
        batch_size = len(sample_indices)
        keep_indices = []
        locally_seen_indices = set(seen_indices)
        for pos, sample_index in enumerate(sample_indices):
            if sample_index in locally_seen_indices:
                continue
            keep_indices.append(pos)
            locally_seen_indices.add(sample_index)
        if not keep_indices:
            continue

        trimmed_result = {}
        for key, value in result.items():
            if key == 'index':
                trimmed_result[key] = [sample_indices[pos] for pos in keep_indices]
            else:
                trimmed_result[key] = _slice_sample_aligned_value(
                    value, keep_indices, batch_size)

        trimmed_results.append(trimmed_result)
        for pos in keep_indices:
            seen_indices.add(sample_indices[pos])
        if len(seen_indices) >= size:
            break

    return trimmed_results


def _merge_distributed_result_parts(part_list):
    """Merge per-rank result parts without truncating longer tails.

    Some eval pipelines may yield uneven numbers of result entries per rank.
    Using plain ``zip(*part_list)`` drops the tail of longer ranks entirely.
    ``zip_longest`` preserves these entries while keeping the usual interleaved
    rank order.
    """
    ordered_results = []
    sentinel = object()
    for res_group in zip_longest(*part_list, fillvalue=sentinel):
        for res in res_group:
            if res is sentinel:
                continue
            ordered_results.append(res)
    return ordered_results


def _extract_result_sample_indices(result):
    """Extract per-sample dataset indices from a normalized result list."""
    sample_indices = []
    for item in result:
        if not isinstance(item, dict) or 'index' not in item:
            return None
        sample_indices.extend(_to_sample_index_list(item['index']))
    return sample_indices


def _count_new_sample_indices(sample_indices, seen_indices, size):
    """Count newly observed sample ids while capping at dataset size."""
    if sample_indices is None:
        return None

    step = 0
    for sample_index in sample_indices:
        if sample_index in seen_indices:
            continue
        if len(seen_indices) >= size:
            break
        seen_indices.add(sample_index)
        step += 1
    return step

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    batch_size=1,
                    ):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): Whether to save viualization results.
            Default: True.
        out_dir (str, optional): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar_completed = 0
    prog_bar_seen_indices = set()
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            result = _normalize_result_list(result)

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(
                    data,
                    result,
                    out_dir=out_dir,
                    show=show,
                    score_thr=show_score_thr)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        sample_indices = _extract_result_sample_indices(result)
        step = _count_new_sample_indices(
            sample_indices, prog_bar_seen_indices, len(dataset))
        if step is None:
            step = min(batch_size, len(dataset) - prog_bar_completed)
        for _ in range(step):
            prog_bar.update()
        prog_bar_completed += step
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, batch_size=1):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    prog_bar_completed = 0
    prog_bar_seen_indices = set()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            result = _normalize_result_list(result)
            # encode mask results
            if len(result) > 0 and isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
            # This logic is only used in panoptic segmentation test.
            elif len(result) > 0 and isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (
                        bbox_results, encode_mask_results(mask_results))

        results.extend(result)

        sample_indices = _extract_result_sample_indices(result)
        use_index_progress = sample_indices is not None
        use_index_progress_list = [None for _ in range(world_size)]
        dist.all_gather_object(use_index_progress_list, use_index_progress)
        use_index_progress = all(use_index_progress_list)

        if use_index_progress:
            gathered_sample_indices = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_sample_indices, sample_indices)
            if rank == 0:
                flat_sample_indices = []
                for rank_sample_indices in gathered_sample_indices:
                    flat_sample_indices.extend(rank_sample_indices)
                step = _count_new_sample_indices(
                    flat_sample_indices, prog_bar_seen_indices, len(dataset))
            else:
                step = None
        elif rank == 0:
            step = min(batch_size * world_size, len(dataset) - prog_bar_completed)
        else:
            step = None

        if rank == 0:
            for _ in range(step):
                prog_bar.update()
            prog_bar_completed += step

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = _merge_distributed_result_parts(part_list)
        ordered_results = _trim_results_by_sample_index(ordered_results, size)
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = _merge_distributed_result_parts(part_list)
        ordered_results = _trim_results_by_sample_index(ordered_results, size)
        return ordered_results
