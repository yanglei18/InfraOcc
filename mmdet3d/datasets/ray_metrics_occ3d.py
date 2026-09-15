# Acknowledgments: https://github.com/tarashakhurana/4d-occ-forecasting
# Modified by Haisong Liu

import math
import copy
import time
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load, _get_build_directory
from tqdm import tqdm
from prettytable import PrettyTable
from mmcv.utils import print_log

from .flow_metrics import calc_voxel_flow_metrics

_dvr = None
_DVR_LOCK_STALE_SECONDS = 600
_DVR_LOCK_FORCE_STALE_SECONDS = 1800


def _clear_stale_dvr_lock():
    try:
        build_dir = Path(_get_build_directory("dvr", verbose=False))
    except Exception:
        return

    lock_path = build_dir / "lock"
    if not lock_path.exists():
        return

    try:
        lock_age = time.time() - lock_path.stat().st_mtime
    except OSError:
        return

    if lock_age < _DVR_LOCK_STALE_SECONDS:
        return

    so_path = build_dir / "dvr.so"
    if not so_path.exists() and lock_age < _DVR_LOCK_FORCE_STALE_SECONDS:
        return

    try:
        lock_path.unlink()
        warnings.warn(
            f'Removed stale DVR torch-extension lock: {lock_path}. '
            'This prevents RayIoU from waiting forever in '
            'torch.utils.cpp_extension.load().')
    except FileNotFoundError:
        pass


def _get_dvr():
    global _dvr
    if _dvr is None:
        _clear_stale_dvr_lock()
        _dvr = load(
            "dvr",
            sources=["libs/dvr/dvr.cpp", "libs/dvr/dvr.cu"],
            verbose=False,
            extra_cuda_cflags=['-allow-unsupported-compiler'])
    return _dvr

_pc_range = [-40, -40, -1.0, 40, 40, 5.4]
_voxel_size = 0.4
_occ_size = [352, 352, 32]

occ_class_names = [
    'others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation','free'
]

flow_class_names = [
    'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'trailer', 'truck'
]


# https://github.com/tarashakhurana/4d-occ-forecasting/blob/ff986082cd6ea10e67ab7839bf0e654736b3f4e2/test_fgbg.py#L29C1-L46C16
def get_rendered_pcds(origin, points, tindex, pred_dist):
    pcds = []

    for t in range(len(origin)):
        mask = (tindex == t)
        # skip the ones with no data
        if not mask.any():
            continue
        _pts = points[mask, :3]
        # use ground truth lidar points for the raycasting direction
        v = _pts - origin[t][None, :]
        d = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
        pred_pts = origin[t][None, :] + d * pred_dist[mask][:, None]
        pcds.append(torch.from_numpy(pred_pts))

    return pcds


def meshgrid3d(occ_size, pc_range):
    W, H, D = occ_size

    xs = torch.linspace(0.5, W - 0.5, W).view(W, 1, 1).expand(W, H, D) / W
    ys = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(W, H, D) / H
    zs = torch.linspace(0.5, D - 0.5, D).view(1, 1, D).expand(W, H, D) / D
    xs = xs * (pc_range[3] - pc_range[0]) + pc_range[0]
    ys = ys * (pc_range[4] - pc_range[1]) + pc_range[1]
    zs = zs * (pc_range[5] - pc_range[2]) + pc_range[2]
    xyz = torch.stack((xs, ys, zs), -1)

    return xyz


def generate_lidar_rays():
    # prepare lidar ray angles
    pitch_angles = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)

    # nuscenes lidar fov: [0.2107773983152201, -0.5439104895672159] (rad)
    while pitch_angles[-1] < 0.21:
        delta = pitch_angles[-1] - pitch_angles[-2]
        pitch_angles.append(pitch_angles[-1] + delta)

    lidar_rays = []
    for pitch_angle in pitch_angles:
        for azimuth_angle in np.arange(0, 360, 1):
            azimuth_angle = np.deg2rad(azimuth_angle)

            x = np.cos(pitch_angle) * np.cos(azimuth_angle)
            y = np.cos(pitch_angle) * np.sin(azimuth_angle)
            z = np.sin(pitch_angle)

            lidar_rays.append((x, y, z))

    return np.array(lidar_rays, dtype=np.float32)


def process_one_sample(sem_pred,
                       lidar_rays,
                       output_origin,
                       flow_pred,
                       pc_range=None):
    # lidar origin in ego coordinate
    # lidar_origin = torch.tensor([[[0.9858, 0.0000, 1.8402]]])
    T = output_origin.shape[1]
    pred_pcds_t = []
    sem_pred = np.asarray(sem_pred)
    occ_shape = sem_pred.shape
    if pc_range is None:
        pc_range = _pc_range
    pc_range = np.asarray(pc_range, dtype=np.float32)
    voxel_size = float((pc_range[3] - pc_range[0]) / max(int(occ_shape[0]), 1))

    free_id = len(occ_class_names) - 1
    occ_pred = copy.deepcopy(sem_pred)
    occ_pred[sem_pred < free_id] = 1
    occ_pred[sem_pred == free_id] = 0
    occ_pred = torch.from_numpy(occ_pred).permute(2, 1, 0)
    occ_pred = occ_pred[None, None, :].contiguous().float()

    offset = torch.Tensor(pc_range[:3])[None, None, :]
    scaler = torch.Tensor([voxel_size] * 3)[None, None, :]

    lidar_tindex = torch.zeros([1, lidar_rays.shape[0]])

    for t in range(T):
        lidar_origin = output_origin[:, t:t + 1, :]  # [1, 1, 3]
        lidar_endpts = lidar_rays[None] + lidar_origin  # [1, 15840, 3]

        output_origin_render = ((lidar_origin - offset) / scaler).float()  # [1, 1, 3]
        output_points_render = ((lidar_endpts - offset) / scaler).float()  # [1, N, 3]
        output_tindex_render = lidar_tindex  # [1, N], all zeros

        with torch.no_grad():
            dvr = _get_dvr()
            pred_dist, _, coord_index = dvr.render_forward(
                occ_pred.cuda(),
                output_origin_render.cuda(),
                output_points_render.cuda(),
                output_tindex_render.cuda(),
                [1, int(occ_shape[2]), int(occ_shape[1]), int(occ_shape[0])],
                "test"
            )
            pred_dist *= voxel_size

        pred_pcds = get_rendered_pcds(
            lidar_origin[0].cpu().numpy(),
            lidar_endpts[0].cpu().numpy(),
            lidar_tindex[0].cpu().numpy(),
            pred_dist[0].cpu().numpy()
        )
        coord_index = coord_index[0, :, :].int().cpu()  # [N, 3]

        pred_flow = torch.from_numpy(flow_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]])  # [N, 2]
        pred_label = torch.from_numpy(sem_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]])[:, None]  # [N, 1]
        pred_dist = pred_dist[0, :, None].cpu()
        pred_pcds = torch.cat([pred_pcds[0], pred_label, pred_dist, pred_flow], dim=-1)  # [N, 7]  5: [x, y, z, label, dist, dx, dy]

        pred_pcds_t.append(pred_pcds)

    pred_pcds_t = torch.cat(pred_pcds_t, dim=0)

    return pred_pcds_t.numpy()


def calc_metrics(pcd_pred_list, pcd_gt_list, class_names=None, flow_classes=None):
    class_names = list(class_names or occ_class_names)
    flow_classes = set(flow_classes or flow_class_names)
    thresholds = [1, 2, 4]

    gt_cnt = np.zeros([len(class_names)])
    pred_cnt = np.zeros([len(class_names)])
    tp_cnt = np.zeros([len(thresholds), len(class_names)])

    for pcd_pred, pcd_gt in zip(pcd_pred_list, pcd_gt_list):
        for j, threshold in enumerate(thresholds):
            # L1
            depth_pred = pcd_pred[:, 4]
            depth_gt = pcd_gt[:, 4]
            l1_error = np.abs(depth_pred - depth_gt)
            tp_dist_mask = (l1_error < threshold)

            for i, cls in enumerate(class_names):
                cls_id = class_names.index(cls)
                cls_mask_pred = (pcd_pred[:, 3] == cls_id)
                cls_mask_gt = (pcd_gt[:, 3] == cls_id)

                gt_cnt_i = cls_mask_gt.sum()
                pred_cnt_i = cls_mask_pred.sum()
                if j == 0:
                    gt_cnt[i] += gt_cnt_i
                    pred_cnt[i] += pred_cnt_i

                tp_cls = cls_mask_gt & cls_mask_pred  # [N]
                tp_mask = np.logical_and(tp_cls, tp_dist_mask)
                tp_cnt[j][i] += tp_mask.sum()

    iou_list = []
    for j, threshold in enumerate(thresholds):
        iou_list.append((tp_cnt[j] / (gt_cnt + pred_cnt - tp_cnt[j]))[:-1])

    return iou_list


def main(sem_pred_list,
         sem_gt_list,
         flow_pred_list,
         flow_gt_list,
         lidar_origin_list,
         logger,
         point_cloud_range=None,
         class_names=None,
         flow_class_names_override=None,
         near_clip=0.0):
    torch.cuda.empty_cache()
    class_names = list(class_names or occ_class_names)
    flow_class_names_local = list(flow_class_names_override or flow_class_names)
    if point_cloud_range is None:
        point_cloud_range = _pc_range

    # generate lidar rays
    lidar_rays = generate_lidar_rays()
    lidar_rays = torch.from_numpy(lidar_rays)

    pcd_pred_list, pcd_gt_list = [], []
    epe_pred_flows, epe_gt_flows = [], []
    epe_pred_sems, epe_gt_sems = [], []
    n_samples = len(sem_pred_list)
    for sem_pred, sem_gt, flow_pred, flow_gt, lidar_origins in tqdm(
            zip(sem_pred_list, sem_gt_list, flow_pred_list, flow_gt_list, lidar_origin_list),
            total=n_samples, ncols=50):
        sem_gt = np.asarray(sem_gt)
        if sem_gt.ndim != 3:
            sem_gt = np.reshape(sem_gt, _occ_size)
        occ_shape = sem_gt.shape

        sem_pred = np.asarray(sem_pred)
        if sem_pred.shape != occ_shape:
            if sem_pred.ndim == 3:
                # Align prediction resolution to GT grid for robust eval.
                sem_pred_t = torch.from_numpy(sem_pred).float()[None, None]
                sem_pred_t = F.interpolate(sem_pred_t, size=occ_shape, mode='nearest')
                sem_pred = sem_pred_t[0, 0].cpu().numpy().astype(np.uint8)
            else:
                sem_pred = np.reshape(sem_pred, occ_shape)

        flow_shape = occ_shape + (2,)
        flow_pred = np.asarray(flow_pred)
        if flow_pred.shape != flow_shape:
            if flow_pred.ndim == 4 and flow_pred.shape[-1] == 2:
                flow_pred_t = torch.from_numpy(flow_pred).permute(3, 0, 1, 2).float()[None]
                flow_pred_t = F.interpolate(flow_pred_t, size=occ_shape, mode='trilinear', align_corners=False)
                flow_pred = flow_pred_t[0].permute(1, 2, 3, 0).cpu().numpy().astype(np.float32)
            else:
                flow_pred = np.reshape(flow_pred, flow_shape)
        flow_gt = np.asarray(flow_gt)
        if flow_gt.shape != flow_shape:
            flow_gt = np.reshape(flow_gt, flow_shape)

        epe_pred_flows.append(flow_pred)
        epe_gt_flows.append(flow_gt)
        epe_pred_sems.append(sem_pred)
        epe_gt_sems.append(sem_gt)

        pcd_pred = process_one_sample(
            sem_pred, lidar_rays, lidar_origins, flow_pred, pc_range=point_cloud_range)
        pcd_gt = process_one_sample(
            sem_gt, lidar_rays, lidar_origins, flow_gt, pc_range=point_cloud_range)

        if near_clip > 0:
            # Ignore first-hit voxels too close to the virtual origin to avoid
            # the sensor stand or nearby infrastructure dominating the metric.
            near_clip_mask = pcd_gt[:, 4] > near_clip
            pcd_pred = pcd_pred[near_clip_mask]
            pcd_gt = pcd_gt[near_clip_mask]

        # evalute on non-free rays
        valid_mask = (pcd_gt[:, 3] != len(class_names) - 1)
        pcd_pred = pcd_pred[valid_mask]
        pcd_gt = pcd_gt[valid_mask]

        assert pcd_pred.shape == pcd_gt.shape
        pcd_pred_list.append(pcd_pred)
        pcd_gt_list.append(pcd_gt)

    iou_list = calc_metrics(
        pcd_pred_list,
        pcd_gt_list,
        class_names=class_names,
        flow_classes=flow_class_names_local,
    )
    flow_metrics = calc_voxel_flow_metrics(
        epe_pred_flows,
        epe_gt_flows,
        epe_pred_sems,
        epe_gt_sems,
        class_names=class_names,
        flow_classes=flow_class_names_local,
    )
    ave_list = flow_metrics['tp_only_ave_list']
    direct_mave = flow_metrics['direct_mave']
    dynamic_match_recall = flow_metrics['dynamic_match_recall']

    table = PrettyTable([
        'Class Names',
        'IoU@1', 'IoU@2', 'IoU@4', 'AVE'
    ])
    table.float_format = '.3'

    for i in range(len(class_names) - 1):
        table.add_row([
            class_names[i],
            iou_list[0][i], iou_list[1][i], iou_list[2][i], ave_list[i]
        ], divider=(i == len(class_names) - 2))

    table.add_row([
        'MEAN',
        np.nanmean(iou_list[0]),
        np.nanmean(iou_list[1]),
        np.nanmean(iou_list[2]),
        np.nanmean(ave_list)
    ])

    print_log(table, logger=logger)

    miou = np.nanmean(iou_list)
    mave = flow_metrics['tp_only_mave']

    occ_score = miou * 0.9 + max(1 - mave, 0.0) * 0.1
    print_log('RayIoU: {:.2f}'.format(miou * 100.0), logger=logger)
    print_log(
        'MAVE: {:.3f} m/s (voxel tp-only flow EPE on GT dynamic voxels with matched semantics)'.format(
            mave),
        logger=logger)
    print_log(
        'DirectMAVE: {:.3f} m/s (voxel direct flow EPE on GT dynamic voxels)'.format(
            direct_mave),
        logger=logger)
    print_log(
        'DynamicMatchRecall: {:.2f}% (GT dynamic voxels with matched predicted semantics)'.format(
            dynamic_match_recall * 100.0),
        logger=logger)
    print_log('Occ score: {:.2f}'.format(occ_score * 100.0), logger=logger)


    torch.cuda.empty_cache()

    return miou, mave, occ_score, {
        'direct_mave': direct_mave,
        'dynamic_match_recall': dynamic_match_recall,
    }
