#!/usr/bin/env python3
"""Audit projection-normalization support on a real validation sample."""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch import nn


@lru_cache(maxsize=1)
def _projection_harness_type():
    """Build the narrow projection adapter only for a real-sample audit.

    Keeping the project plugin import off the module path lets the pure support
    statistics be reused by repository-wide tests without mutating MMDetection's
    global registries.
    """
    from projects.STCOcc.mmdet3d_plugin.models.stcocc.view_transformation.backward_projection.bevformer_utils.bevformer_encoder import \
        BEVFormerEncoder  # noqa: E501

    class ProjectionHarness(nn.Module):
        """Expose the encoder's projection routine without a model build."""

        get_reference_points = BEVFormerEncoder.get_reference_points
        point_sampling = BEVFormerEncoder.point_sampling

        def __init__(self, final_dim, mode):
            super().__init__()
            self.final_dim = tuple(final_dim)
            self.projection_normalization = mode
            self.fp16_enabled = False

    return ProjectionHarness


def summarize_support(source_mask, released_mask):
    """Summarize mutually exclusive support categories for boolean masks."""
    source_mask = np.asarray(source_mask, dtype=bool)
    released_mask = np.asarray(released_mask, dtype=bool)
    if source_mask.shape != released_mask.shape:
        raise ValueError('projection masks must have identical shapes')
    if source_mask.ndim < 2:
        raise ValueError('projection masks must include camera and query axes')

    both = source_mask & released_mask
    source_only = source_mask & ~released_mask
    released_only = released_mask & ~source_mask
    neither = ~source_mask & ~released_mask
    total = int(source_mask.size)

    def count_and_fraction(mask):
        count = int(mask.sum())
        return dict(count=count, fraction=count / total if total else 0.0)

    return dict(
        total_camera_queries=total,
        source_valid=count_and_fraction(source_mask),
        released_valid=count_and_fraction(released_mask),
        both_valid=count_and_fraction(both),
        source_only=count_and_fraction(source_only),
        released_only=count_and_fraction(released_only),
        neither_valid=count_and_fraction(neither),
    )


def _unwrap_sample(sample):
    meta = sample['img_metas'][0].data
    img_inputs = sample['img_inputs'][0]
    images, sensor2egos, ego2globals, intrins, post_augs, bda = img_inputs
    cam_params = tuple(
        tensor.unsqueeze(0)
        for tensor in (sensor2egos, ego2globals, intrins, post_augs, bda))
    return meta, images, cam_params


def _project(mode, final_dim, reference_points, pc_range, meta, cam_params):
    harness = _projection_harness_type()(final_dim, mode)
    coordinates, _, mask = harness.point_sampling(reference_points, pc_range,
                                                  [meta], cam_params)
    return coordinates[:, 0].cpu(), mask[:, 0].cpu()


def build_real_sample_audit(config_path, sample_index=0):
    # Keep the repository model stack off the import path for callers that use
    # only ``summarize_support`` (for example lightweight audit tests).
    from mmcv import Config
    from mmdet3d.datasets import build_dataset

    config = Config.fromfile(str(config_path))
    dataset = build_dataset(config.data.val)
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f'sample index {sample_index} outside dataset of size {len(dataset)}'
        )
    meta, images, cam_params = _unwrap_sample(dataset[sample_index])

    pc_range = list(config.point_cloud_range)
    grid = config.grid_config
    bev_h = int((pc_range[3] - pc_range[0]) / grid['x'][2])
    bev_w = int((pc_range[4] - pc_range[1]) / grid['y'][2])
    bev_z = int((pc_range[5] - pc_range[2]) / grid['z'][2])
    reference_points = _projection_harness_type()(
        config.data_config['input_size'],
        'v2x_source_width_height').get_reference_points(
            bev_h,
            bev_w,
            pc_range[5] - pc_range[2],
            num_points_in_pillar=bev_z,
            dim='3d',
            bs=1,
            device='cpu',
            dtype=torch.float32)

    source_coordinates, source_mask = _project(
        'v2x_source_width_height', config.data_config['input_size'],
        reference_points, pc_range, meta, cam_params)
    released_coordinates, released_mask = _project(
        'released_height_width', config.data_config['input_size'],
        reference_points, pc_range, meta, cam_params)

    # A BEV query is supported by a camera when at least one vertical anchor is
    # in front of and inside that camera under the declared normalization.
    source_query_mask = source_mask.any(dim=-1).numpy()
    released_query_mask = released_mask.any(dim=-1).numpy()
    summary = summarize_support(source_query_mask, released_query_mask)
    summary.update(
        config=str(Path(config_path)),
        sample_index=int(sample_index),
        sample_token=str(meta.get('sample_idx')),
        scene_name=str(meta.get('scene_name')),
        camera_names=list(meta.get('cam_names', [])),
        image_height=int(config.data_config['input_size'][0]),
        image_width=int(config.data_config['input_size'][1]),
        bev_shape=[bev_h, bev_w, bev_z],
    )
    return dict(
        summary=summary,
        images=images.cpu(),
        source_coordinates=source_coordinates,
        released_coordinates=released_coordinates,
        source_query_mask=source_query_mask,
        released_query_mask=released_query_mask,
        bev_shape=(bev_h, bev_w),
    )


def _display_image(tensor):
    mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
    image = tensor.permute(1, 2, 0).numpy() * std + mean
    return np.clip(image / 255.0, 0.0, 1.0)


def render_real_sample_audit(audit, output_path):
    images = audit['images']
    source_coordinates = audit['source_coordinates']
    released_coordinates = audit['released_coordinates']
    source_mask = audit['source_query_mask']
    released_mask = audit['released_query_mask']
    bev_h, bev_w = audit['bev_shape']
    summary = audit['summary']
    camera_names = summary['camera_names'] or [
        f'camera {index}' for index in range(images.shape[0])
    ]
    height = summary['image_height']
    width = summary['image_width']

    figure, axes = plt.subplots(2, images.shape[0], figsize=(18, 8.7))
    support_cmap = ListedColormap(['#eeeeee', '#2a9d8f', '#e76f51', '#7b2cbf'])
    for camera_index, camera_name in enumerate(camera_names):
        image_axis = axes[0, camera_index]
        image_axis.imshow(_display_image(images[camera_index]))
        # Plot a deterministic sparse subset of vertical anchors. Coordinates
        # are converted to the pixel sampled by deformable attention.
        source = source_coordinates[camera_index].reshape(-1, 2)
        released = released_coordinates[camera_index].reshape(-1, 2)
        stride = max(1, source.shape[0] // 1600)
        source = source[::stride]
        released = released[::stride]
        source_valid = ((source[:, 0] > 0) & (source[:, 0] < 1)
                        & (source[:, 1] > 0) & (source[:, 1] < 1))
        released_valid = ((released[:, 0] > 0) & (released[:, 0] < 1)
                          & (released[:, 1] > 0) & (released[:, 1] < 1))
        image_axis.scatter(
            source[source_valid, 0] * width,
            source[source_valid, 1] * height,
            s=3,
            alpha=0.34,
            color='#2a9d8f',
            label='source x/W,y/H')
        image_axis.scatter(
            released[released_valid, 0] * width,
            released[released_valid, 1] * height,
            s=3,
            alpha=0.34,
            color='#e76f51',
            label='released x/H,y/W')
        image_axis.set_title(camera_name)
        image_axis.set_xlim(0, width)
        image_axis.set_ylim(height, 0)
        image_axis.axis('off')
        if camera_index == 0:
            image_axis.legend(loc='lower left', fontsize=7, markerscale=2)

        support_axis = axes[1, camera_index]
        category = np.zeros(source_mask[camera_index].shape, dtype=np.uint8)
        category[source_mask[camera_index] & released_mask[camera_index]] = 3
        category[source_mask[camera_index] & ~released_mask[camera_index]] = 1
        category[~source_mask[camera_index] & released_mask[camera_index]] = 2
        support_axis.imshow(
            category.reshape(bev_h, bev_w),
            origin='lower',
            interpolation='nearest',
            cmap=support_cmap,
            vmin=0,
            vmax=3)
        support_axis.set_title('BEV camera support')
        support_axis.set_xlabel('BEV y index')
        if camera_index == 0:
            support_axis.set_ylabel('BEV x index')
        else:
            support_axis.set_yticks([])

    fractions = summary
    figure.suptitle(
        ('Real-calibration projection-normalization support audit\n'
         f'validation index {summary["sample_index"]} · '
         f'{summary["scene_name"]}'),
        fontsize=17,
        fontweight='bold')
    figure.text(
        0.5,
        0.015,
        ('BEV camera-query fractions — source valid '
         f'{fractions["source_valid"]["fraction"]:.3f}, released valid '
         f'{fractions["released_valid"]["fraction"]:.3f}, source-only '
         f'{fractions["source_only"]["fraction"]:.3f}, released-only '
         f'{fractions["released_only"]["fraction"]:.3f}.  '
         'BEV colors: gray neither, green source-only, orange released-only, '
         'purple both.'),
        ha='center',
        fontsize=10)
    figure.tight_layout(rect=(0, 0.045, 1, 0.95))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(
            'projects/STCOcc/configs/'
            'stcocc_c_4x4_24e_main_singleframe_historical_data_recipe.py'))
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-figure', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    audit = build_real_sample_audit(args.config, args.sample_index)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit['summary'], indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    render_real_sample_audit(audit, args.output_figure)
    print(json.dumps(audit['summary'], sort_keys=True))


if __name__ == '__main__':
    main()
