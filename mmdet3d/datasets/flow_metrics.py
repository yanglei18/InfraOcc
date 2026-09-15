import numpy as np


def _safe_nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.any(np.isfinite(values)):
        return float('nan')
    return float(np.nanmean(values))


def calc_flow_epe(flow_pred_list,
                  flow_gt_list,
                  sem_gt_list,
                  class_names,
                  flow_classes):
    class_names = list(class_names)
    flow_classes = set(flow_classes)

    ave_list = np.full(len(class_names) - 1, np.nan, dtype=np.float64)

    for class_index, class_name in enumerate(class_names[:-1]):
        if class_name not in flow_classes:
            continue

        class_errors = []
        for flow_pred, flow_gt, sem_gt in zip(flow_pred_list, flow_gt_list, sem_gt_list):
            sem_gt = np.asarray(sem_gt)
            flow_pred = np.asarray(flow_pred)
            flow_gt = np.asarray(flow_gt)
            valid_mask = sem_gt != 255
            class_mask = valid_mask & (sem_gt == class_index)
            if not np.any(class_mask):
                continue
            pred_flow_i = flow_pred[class_mask]
            gt_flow_i = flow_gt[class_mask]
            class_errors.append(np.linalg.norm(pred_flow_i - gt_flow_i, axis=-1))

        if class_errors:
            ave_list[class_index] = float(np.mean(np.concatenate(class_errors, axis=0)))

    return ave_list


def calc_voxel_flow_metrics(flow_pred_list,
                            flow_gt_list,
                            sem_pred_list,
                            sem_gt_list,
                            class_names,
                            flow_classes):
    class_names = list(class_names)
    flow_classes = set(flow_classes)
    num_classes = len(class_names) - 1

    direct_ave_list = np.full(num_classes, np.nan, dtype=np.float64)
    tp_only_ave_list = np.full(num_classes, np.nan, dtype=np.float64)
    match_recall_list = np.full(num_classes, np.nan, dtype=np.float64)

    total_matched = 0
    total_gt = 0

    for class_index, class_name in enumerate(class_names[:-1]):
        if class_name not in flow_classes:
            continue

        direct_errors = []
        tp_only_errors = []
        matched_count = 0
        gt_count = 0

        for flow_pred, flow_gt, sem_pred, sem_gt in zip(
                flow_pred_list, flow_gt_list, sem_pred_list, sem_gt_list):
            sem_gt = np.asarray(sem_gt)
            sem_pred = np.asarray(sem_pred)
            flow_pred = np.asarray(flow_pred)
            flow_gt = np.asarray(flow_gt)

            valid_mask = sem_gt != 255
            class_gt_mask = valid_mask & (sem_gt == class_index)
            if not np.any(class_gt_mask):
                continue

            gt_count += int(class_gt_mask.sum())
            direct_errors.append(np.linalg.norm(
                flow_pred[class_gt_mask] - flow_gt[class_gt_mask], axis=-1))

            class_match_mask = class_gt_mask & (sem_pred == class_index)
            if not np.any(class_match_mask):
                continue

            matched_count += int(class_match_mask.sum())
            tp_only_errors.append(np.linalg.norm(
                flow_pred[class_match_mask] - flow_gt[class_match_mask], axis=-1))

        if gt_count > 0:
            match_recall_list[class_index] = matched_count / float(gt_count)
            total_matched += matched_count
            total_gt += gt_count
        if direct_errors:
            direct_ave_list[class_index] = float(
                np.mean(np.concatenate(direct_errors, axis=0)))
        if tp_only_errors:
            tp_only_ave_list[class_index] = float(
                np.mean(np.concatenate(tp_only_errors, axis=0)))

    return {
        'direct_ave_list': direct_ave_list,
        'tp_only_ave_list': tp_only_ave_list,
        'match_recall_list': match_recall_list,
        'direct_mave': _safe_nanmean(direct_ave_list),
        'tp_only_mave': _safe_nanmean(tp_only_ave_list),
        'dynamic_match_recall': (
            total_matched / float(total_gt) if total_gt > 0 else float('nan')),
    }
