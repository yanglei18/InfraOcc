"""Metrics for occupancy transported from a frame to its next keyframe."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


def _safe_percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float('nan')
    return float(numerator) / float(denominator) * 100.0


def _binary_iou(pred_mask: np.ndarray,
                gt_mask: np.ndarray) -> Tuple[float, int, int, int]:
    true_positive = int(np.logical_and(pred_mask, gt_mask).sum())
    false_positive = int(np.logical_and(pred_mask, ~gt_mask).sum())
    false_negative = int(np.logical_and(~pred_mask, gt_mask).sum())
    denominator = true_positive + false_positive + false_negative
    return _safe_percent(
        true_positive,
        denominator), true_positive, false_positive, false_negative


def _quaternion_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    """Return the rotation matrix for a nuScenes-format ``(w, x, y, z)`` quaternion."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float32).reshape(4)
    norm = float(np.sqrt(w * w + x * x + y * y + z * z))
    if norm <= 0.0:
        raise ValueError('ego2global_rotation must be a non-zero quaternion.')
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [
            1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 *
            (x * z + y * w)
        ],
        [
            2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 *
            (y * z - x * w)
        ],
        [
            2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 *
            (x * x + y * y)
        ],
    ],
                    dtype=np.float32)


def _voxel_centers(indices: np.ndarray, grid_shape: Sequence[int],
                   point_cloud_range: Sequence[float]) -> np.ndarray:
    point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
    voxel_size = (point_cloud_range[3:] - point_cloud_range[:3]) / np.asarray(
        grid_shape, dtype=np.float32)
    return point_cloud_range[:3][None, :] + (indices.astype(np.float32) +
                                             0.5) * voxel_size[None, :]


def _transform_to_next_ego(points_ego: np.ndarray, current_info: Mapping,
                           next_info: Mapping) -> np.ndarray:
    current_rotation = _quaternion_to_rotation(
        current_info['ego2global_rotation'])
    current_translation = np.asarray(
        current_info['ego2global_translation'], dtype=np.float32).reshape(3)
    next_rotation = _quaternion_to_rotation(next_info['ego2global_rotation'])
    next_translation = np.asarray(
        next_info['ego2global_translation'], dtype=np.float32).reshape(3)
    points_global = points_ego @ current_rotation.T + current_translation[
        None, :]
    return (points_global - next_translation[None, :]) @ next_rotation


def transport_dynamic_semantics(pred_semantics: np.ndarray,
                                pred_flow: np.ndarray, delta_t_sec: float,
                                current_info: Mapping, next_info: Mapping,
                                point_cloud_range: Sequence[float],
                                dynamic_class_indices: Sequence[int],
                                empty_idx: int) -> np.ndarray:
    """Transport predicted dynamic voxels into the next ego frame.

    The result only contains dynamic labels; all other cells are ``empty_idx``.
    A deterministic per-class vote resolves multiple source voxels that land in
    the same target voxel.
    """
    pred_semantics = np.asarray(pred_semantics)
    pred_flow = np.asarray(pred_flow, dtype=np.float32)
    if pred_flow.shape[:3] != pred_semantics.shape or pred_flow.shape[-1] < 2:
        raise ValueError(
            'pred_flow must have shape pred_semantics.shape + (2,), got '
            f'{pred_flow.shape} for {pred_semantics.shape}.')

    dynamic_mask = np.isin(pred_semantics, list(dynamic_class_indices))
    source_indices = np.argwhere(dynamic_mask)
    transported = np.full(
        pred_semantics.shape, int(empty_idx), dtype=pred_semantics.dtype)
    if source_indices.size == 0:
        return transported

    source_points = _voxel_centers(source_indices, pred_semantics.shape,
                                   point_cloud_range)
    source_flow = pred_flow[source_indices[:, 0], source_indices[:, 1],
                            source_indices[:, 2], :2]
    source_points[:, :2] += source_flow * float(delta_t_sec)
    target_points = _transform_to_next_ego(source_points, current_info,
                                           next_info)

    point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
    voxel_size = (point_cloud_range[3:] - point_cloud_range[:3]) / np.asarray(
        pred_semantics.shape, dtype=np.float32)
    target_indices = np.floor(
        (target_points - point_cloud_range[:3][None, :]) /
        voxel_size[None, :]).astype(np.int32)
    valid = np.all(
        (target_indices >= 0) & (target_indices < np.asarray(
            pred_semantics.shape, dtype=np.int32)[None, :]),
        axis=1)
    if not np.any(valid):
        return transported

    target_indices = target_indices[valid]
    source_labels = pred_semantics[source_indices[valid,
                                                  0], source_indices[valid, 1],
                                   source_indices[valid, 2]]
    linear_indices = np.ravel_multi_index(target_indices.T,
                                          pred_semantics.shape)
    dynamic_class_indices = np.asarray(dynamic_class_indices, dtype=np.int32)
    class_lookup = np.full(
        int(dynamic_class_indices.max()) + 1, -1, dtype=np.int32)
    class_lookup[dynamic_class_indices] = np.arange(
        dynamic_class_indices.size, dtype=np.int32)
    source_class_positions = class_lookup[source_labels.astype(
        np.int64, copy=False)]

    # Count only target/class pairs that actually occur.  This avoids the
    # dense ``num_targets x num_dynamic_classes`` vote matrix and its indexed
    # ``add.at``.  Codes are sorted by target and then class; choosing the
    # first maximum preserves ``argmax``'s lowest-class tie break exactly.
    pair_codes = (
        linear_indices.astype(np.int64, copy=False) *
        dynamic_class_indices.size + source_class_positions)
    unique_pairs, pair_counts = np.unique(pair_codes, return_counts=True)
    pair_targets = unique_pairs // dynamic_class_indices.size
    pair_classes = (unique_pairs % dynamic_class_indices.size).astype(
        np.int32, copy=False)
    group_starts = np.r_[
        0,
        np.flatnonzero(pair_targets[1:] != pair_targets[:-1]) + 1,
    ]
    group_lengths = np.diff(np.r_[group_starts, pair_counts.size])
    group_max_counts = np.maximum.reduceat(pair_counts, group_starts)
    maximum_pairs = pair_counts == np.repeat(group_max_counts, group_lengths)
    maximum_indices = np.flatnonzero(maximum_pairs)
    _, first_maximum = np.unique(
        pair_targets[maximum_indices], return_index=True)
    winning_pairs = maximum_indices[first_maximum]
    transported.reshape(-1)[pair_targets[group_starts]] = \
        dynamic_class_indices[pair_classes[winning_pairs]]
    return transported


def evaluate_next_frame_dynamic_iou_samples(
        samples: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, Mapping,
                                Mapping, float]],
        point_cloud_range: Sequence[float],
        dynamic_class_indices: Sequence[int],
        empty_idx: int,
        ignore_label: int = 255) -> dict:
    """Compute next-frame IoU from a stream of prediction/GT samples.

    Streaming keeps the next-frame labels out of the resident result set.
    """
    dynamic_class_indices = tuple(
        int(index) for index in dynamic_class_indices)
    if not dynamic_class_indices:
        raise ValueError('dynamic_class_indices must not be empty.')
    true_positive = np.zeros(len(dynamic_class_indices), dtype=np.int64)
    false_positive = np.zeros(len(dynamic_class_indices), dtype=np.int64)
    false_negative = np.zeros(len(dynamic_class_indices), dtype=np.int64)
    binary_true_positive = 0
    binary_false_positive = 0
    binary_false_negative = 0
    transported_count = 0

    for pred_semantics, pred_flow, next_semantics, current_info, next_info, delta_t_sec in samples:
        next_semantics = np.asarray(next_semantics)
        transported = transport_dynamic_semantics(
            pred_semantics=pred_semantics,
            pred_flow=pred_flow,
            delta_t_sec=delta_t_sec,
            current_info=current_info,
            next_info=next_info,
            point_cloud_range=point_cloud_range,
            dynamic_class_indices=dynamic_class_indices,
            empty_idx=empty_idx,
        )
        if transported.shape != next_semantics.shape:
            raise ValueError(
                'Transported prediction and next-frame GT have different shapes: '
                f'{transported.shape} vs {next_semantics.shape}.')

        valid_mask = next_semantics != int(ignore_label)
        # ``transport_dynamic_semantics`` only emits dynamic labels or empty,
        # so this is exactly equivalent to another full-volume ``np.isin``.
        pred_dynamic = (transported != int(empty_idx)) & valid_mask
        gt_dynamic = np.isin(next_semantics,
                             dynamic_class_indices) & valid_mask
        binary_true_positive += int(
            np.logical_and(pred_dynamic, gt_dynamic).sum())
        binary_false_positive += int(
            np.logical_and(pred_dynamic, ~gt_dynamic).sum())
        binary_false_negative += int(
            np.logical_and(~pred_dynamic, gt_dynamic).sum())
        for class_position, class_index in enumerate(dynamic_class_indices):
            pred_mask = (transported == class_index) & valid_mask
            gt_mask = (next_semantics == class_index) & valid_mask
            true_positive[class_position] += np.logical_and(
                pred_mask, gt_mask).sum()
            false_positive[class_position] += np.logical_and(
                pred_mask, ~gt_mask).sum()
            false_negative[class_position] += np.logical_and(
                ~pred_mask, gt_mask).sum()
        transported_count += 1

    denominators = true_positive + false_positive + false_negative
    class_iou = np.full(len(dynamic_class_indices), np.nan, dtype=np.float64)
    valid_classes = denominators > 0
    class_iou[valid_classes] = true_positive[valid_classes] / denominators[
        valid_classes]
    binary_denominator = (
        binary_true_positive + binary_false_positive + binary_false_negative)
    return {
        'next_dynamic_iou':
        float(np.nanmean(class_iou) *
              100.0) if np.any(np.isfinite(class_iou)) else float('nan'),
        'next_dynamic_iou_per_class':
        class_iou * 100.0,
        'next_dynamic_binary_iou':
        _safe_percent(binary_true_positive, binary_denominator),
        'next_dynamic_ghost_rate':
        _safe_percent(binary_false_positive,
                      binary_true_positive + binary_false_positive),
        'next_dynamic_tp':
        int(binary_true_positive),
        'next_dynamic_fp':
        int(binary_false_positive),
        'next_dynamic_fn':
        int(binary_false_negative),
        'samples':
        int(transported_count),
    }


def evaluate_next_frame_dynamic_iou(pred_semantics_list: Sequence[np.ndarray],
                                    pred_flow_list: Sequence[np.ndarray],
                                    next_semantics_list: Sequence[np.ndarray],
                                    current_infos: Sequence[Mapping],
                                    next_infos: Sequence[Mapping],
                                    delta_t_sec_list: Sequence[float],
                                    point_cloud_range: Sequence[float],
                                    dynamic_class_indices: Sequence[int],
                                    empty_idx: int,
                                    ignore_label: int = 255) -> dict:
    """Compute class-mean dynamic IoU after predicted next-frame transport."""
    sequences = (
        pred_semantics_list,
        pred_flow_list,
        next_semantics_list,
        current_infos,
        next_infos,
        delta_t_sec_list,
    )
    sample_count = len(pred_semantics_list)
    if any(len(values) != sample_count for values in sequences[1:]):
        raise ValueError(
            'All next-frame evaluation inputs must have the same length.')
    return evaluate_next_frame_dynamic_iou_samples(
        samples=zip(*sequences),
        point_cloud_range=point_cloud_range,
        dynamic_class_indices=dynamic_class_indices,
        empty_idx=empty_idx,
        ignore_label=ignore_label,
    )


def evaluate_dynamic_motion_state_iou(
        pred_semantics_list: Sequence[np.ndarray],
        pred_flow_list: Sequence[np.ndarray],
        gt_semantics_list: Sequence[np.ndarray],
        gt_flow_list: Sequence[np.ndarray],
        dynamic_class_indices: Sequence[int],
        speed_threshold: float = 2.0,
        fast_speed_threshold: float = 5.0,
        valid_mask_list: Optional[Sequence[np.ndarray]] = None,
        ignore_label: int = 255) -> dict:
    """Compute current-frame dynamic IoU split by motion state.

    The split is intentionally binary and flow-based.  A voxel is dynamic when
    its semantic label belongs to ``dynamic_class_indices``.  It is moving when
    the corresponding xy flow speed is at least ``speed_threshold``; otherwise
    it is still.  This keeps the diagnostic tied to the model's velocity output
    and avoids tuning semantic class subsets to explain temporal behavior.
    """
    sequences = (
        pred_semantics_list,
        pred_flow_list,
        gt_semantics_list,
        gt_flow_list,
    )
    sample_count = len(pred_semantics_list)
    if any(len(values) != sample_count for values in sequences[1:]):
        raise ValueError(
            'All dynamic motion inputs must have the same length.')
    if valid_mask_list is None:
        valid_mask_list = [None] * sample_count
    elif len(valid_mask_list) != sample_count:
        raise ValueError('valid_mask_list must match the number of samples.')

    speed_threshold = float(speed_threshold)
    fast_speed_threshold = float(fast_speed_threshold)
    if fast_speed_threshold <= speed_threshold:
        raise ValueError('fast_speed_threshold must exceed speed_threshold.')
    dynamic_class_indices = tuple(
        int(index) for index in dynamic_class_indices)
    moving_tp = moving_fp = moving_fn = 0
    still_tp = still_fp = still_fn = 0
    speed_counts = {
        'slow': [0, 0, 0],
        'medium': [0, 0, 0],
        'fast': [0, 0, 0],
    }
    dynamic_fp = dynamic_pred = 0
    valid_samples = 0

    for pred_semantics, pred_flow, gt_semantics, gt_flow, valid_mask in zip(
            pred_semantics_list, pred_flow_list, gt_semantics_list,
            gt_flow_list, valid_mask_list):
        pred_semantics = np.asarray(pred_semantics)
        gt_semantics = np.asarray(gt_semantics)
        pred_flow = np.asarray(pred_flow, dtype=np.float32)
        gt_flow = np.asarray(gt_flow, dtype=np.float32)
        if pred_semantics.shape != gt_semantics.shape:
            raise ValueError(
                'Predicted and GT semantics have different shapes: '
                f'{pred_semantics.shape} vs {gt_semantics.shape}.')
        if (pred_flow.shape[:3] != pred_semantics.shape
                or gt_flow.shape[:3] != gt_semantics.shape):
            raise ValueError(
                'Predicted and GT flow must match semantic grid shapes, got '
                f'{pred_flow.shape}, {gt_flow.shape}, {pred_semantics.shape}.')

        if valid_mask is None:
            valid = gt_semantics != int(ignore_label)
        else:
            valid = np.asarray(valid_mask).astype(bool) & (
                gt_semantics != int(ignore_label))
        gt_speed = np.linalg.norm(gt_flow[..., :2], axis=-1)
        pred_speed = np.linalg.norm(pred_flow[..., :2], axis=-1)
        gt_finite = np.isfinite(gt_speed)
        pred_finite = np.isfinite(pred_speed)
        valid &= gt_finite

        gt_dynamic = np.isin(gt_semantics, dynamic_class_indices) & valid
        pred_dynamic = np.isin(pred_semantics, dynamic_class_indices) & valid
        gt_moving = gt_dynamic & (gt_speed >= speed_threshold)
        pred_moving = pred_dynamic & pred_finite & (
            pred_speed >= speed_threshold)
        gt_still = gt_dynamic & (gt_speed < speed_threshold)
        pred_still = pred_dynamic & pred_finite & (
            pred_speed < speed_threshold)

        _, tp, fp, fn = _binary_iou(pred_moving, gt_moving)
        moving_tp += tp
        moving_fp += fp
        moving_fn += fn
        _, tp, fp, fn = _binary_iou(pred_still, gt_still)
        still_tp += tp
        still_fp += fp
        still_fn += fn
        speed_masks = {
            'slow': (gt_speed < speed_threshold, pred_speed < speed_threshold),
            'medium': (np.logical_and(gt_speed >= speed_threshold, gt_speed
                                      < fast_speed_threshold),
                       np.logical_and(pred_speed >= speed_threshold, pred_speed
                                      < fast_speed_threshold)),
            'fast': (gt_speed >= fast_speed_threshold, pred_speed
                     >= fast_speed_threshold),
        }
        for name, (gt_speed_mask, pred_speed_mask) in speed_masks.items():
            gt_state = gt_dynamic & gt_speed_mask
            pred_state = pred_dynamic & pred_finite & pred_speed_mask
            _, tp, fp, fn = _binary_iou(pred_state, gt_state)
            speed_counts[name][0] += tp
            speed_counts[name][1] += fp
            speed_counts[name][2] += fn
        dynamic_fp += int(np.logical_and(pred_dynamic, ~gt_dynamic).sum())
        dynamic_pred += int(pred_dynamic.sum())
        valid_samples += 1

    moving_iou = _safe_percent(moving_tp, moving_tp + moving_fp + moving_fn)
    still_iou = _safe_percent(still_tp, still_tp + still_fp + still_fn)
    speed_ious = {
        name: _safe_percent(tp, tp + fp + fn)
        for name, (tp, fp, fn) in speed_counts.items()
    }
    return {
        'moving_dynamic_iou': moving_iou,
        'still_dynamic_iou': still_iou,
        'slow_dynamic_iou': speed_ious['slow'],
        'medium_dynamic_iou': speed_ious['medium'],
        'fast_dynamic_iou': speed_ious['fast'],
        'dynamic_ghost_rate': _safe_percent(dynamic_fp, dynamic_pred),
        'moving_dynamic_tp': int(moving_tp),
        'moving_dynamic_fp': int(moving_fp),
        'moving_dynamic_fn': int(moving_fn),
        'still_dynamic_tp': int(still_tp),
        'still_dynamic_fp': int(still_fp),
        'still_dynamic_fn': int(still_fn),
        'dynamic_motion_samples': int(valid_samples),
        'dynamic_motion_speed_threshold': speed_threshold,
        'dynamic_motion_fast_speed_threshold': fast_speed_threshold,
    }
