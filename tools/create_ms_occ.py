import os
import argparse

import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Generate multi-scale occ')
    parser.add_argument(
        '--dataset',
        type=str,
        default='v2xreal',
        choices=['v2xreal', 'occ3d', 'openocc'],
        help='dataset label layout')
    parser.add_argument(
        '--pkl_path',
        type=str,
        default='data/v2xreal_nuscenes/v2xreal_infos_val.pkl',
        help='path to the pkl file')
    parser.add_argument(
        '--empty-cls-idx',
        type=int,
        default=None,
        help='empty/free class id. If omitted, use the dataset default.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='overwrite existing labels_1_*.npz files')
    args = parser.parse_args()
    return args


def resolve_empty_cls_idx(dataset, empty_cls_idx=None):
    if empty_cls_idx is not None:
        return int(empty_cls_idx)
    dataset_empty_cls_idx = {
        'v2xreal': 17,
        'occ3d': 17,
        'openocc': 16,
    }
    return dataset_empty_cls_idx[dataset]


def downsample_label(label, downscale=2, empty_cls_idx=17, ignore_idx=255):
    if downscale == 1:
        return label

    label = np.asarray(label)
    if label.ndim != 3:
        raise ValueError(f'Expected a 3D voxel grid, got shape {label.shape}.')

    ds = int(downscale)
    h, w, d = label.shape
    if h % ds != 0 or w % ds != 0 or d % ds != 0:
        raise ValueError(
            f'Label shape {label.shape} is not divisible by downscale {ds}.')

    hs, ws, ds_z = h // ds, w // ds, d // ds
    blocks = label.reshape(hs, ds, ws, ds, ds_z,
                           ds).transpose(0, 2, 4, 1, 3,
                                         5).reshape(hs, ws, ds_z, ds**3)

    empty_count = (blocks == empty_cls_idx).sum(axis=-1)
    ignore_count = (blocks == ignore_idx).sum(axis=-1)
    counts = np.stack([(blocks == cls_id).sum(axis=-1)
                       for cls_id in range(empty_cls_idx)],
                      axis=-1)
    majority = counts.argmax(axis=-1).astype(np.uint8)
    empty_threshold = int(np.ceil(0.95 * (ds**3)))

    label_downscale = np.where(empty_count >= empty_threshold, empty_cls_idx,
                               majority).astype(np.uint8)
    label_downscale[ignore_count == ds**3] = ignore_idx
    return label_downscale


def downsample_mask(mask, downscale=2):
    if downscale == 1:
        return mask
    mask = np.asarray(mask)
    ds = int(downscale)
    h, w, d = mask.shape
    if h % ds != 0 or w % ds != 0 or d % ds != 0:
        raise ValueError(
            f'Mask shape {mask.shape} is not divisible by downscale {ds}.')
    hs, ws, ds_z = h // ds, w // ds, d // ds
    blocks = mask.reshape(hs, ds, ws, ds, ds_z,
                          ds).transpose(0, 2, 4, 1, 3,
                                        5).reshape(hs, ws, ds_z, ds**3)
    return (blocks.max(axis=-1) > 0).astype(mask.dtype)


def save_ms_occ(label_file, save_path, downscale, empty_cls_idx, overwrite):
    if os.path.exists(save_path) and not overwrite:
        return

    labels = label_file['semantics']
    labels_downscale = downsample_label(
        labels, downscale=downscale, empty_cls_idx=empty_cls_idx)

    save_kwargs = dict(semantics=labels_downscale, flow=labels_downscale)
    if 'mask_camera' in label_file.files:
        save_kwargs['mask_camera'] = downsample_mask(
            label_file['mask_camera'], downscale=downscale)
    np.savez_compressed(save_path, **save_kwargs)


def main(args):
    data_infos = mmcv.load(args.pkl_path)

    for info in mmcv.track_iter_progress(data_infos['infos']):
        occ_path = info['occ_path']
        if args.dataset == 'openocc':
            occ_path = info['occ_path'].replace('gts', 'openocc_v2')
        label_path = os.path.join(occ_path, 'labels.npz')

        label_file = np.load(label_path)
        empty_cls_idx = resolve_empty_cls_idx(args.dataset, args.empty_cls_idx)
        for downscale in (2, 4, 8):
            save_path = os.path.join(occ_path, f'labels_1_{downscale}.npz')
            save_ms_occ(
                label_file,
                save_path,
                downscale=downscale,
                empty_cls_idx=empty_cls_idx,
                overwrite=args.overwrite)


if __name__ == "__main__":
    args = parse_args()
    main(args)
