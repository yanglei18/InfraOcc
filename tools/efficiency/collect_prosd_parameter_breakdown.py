import argparse
import json
import os
import sys
from os import path as osp

from mmcv import Config
from mmdet3d.models import build_model


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import projects.InfraOcc.mmdet3d_plugin  # noqa: F401,E402


ROWS = [
    dict(
        label='Plain',
        config='projects/InfraOcc/configs/ablation/core_chain/infraocc_c_4x4_24e_plain.py',
        latency_ms=129.80),
    dict(
        label='Parallel S2D',
        config='projects/InfraOcc/configs/ablation/core_chain/infraocc_c_4x4_24e_prosd_no_supp_no_raw_bypass.py',
        latency_ms=181.12),
    dict(
        label='Progressive S2D',
        config='projects/InfraOcc/configs/ablation/core_chain/infraocc_c_4x4_24e_prosd_full.py',
        latency_ms=225.36),
]


def count_params(module):
    if module is None:
        return 0.0
    return sum(parameter.numel() for parameter in module.parameters()) / 1e6


def rounded(value):
    return round(float(value), 4)


def build_row(row):
    cfg = Config.fromfile(row['config'])
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    forward_projection = model.forward_projection
    head_params = (
        count_params(getattr(model, 'occupancy_head', None))
        + count_params(getattr(model, 'prior_static_bias', None))
        + count_params(getattr(model, 'prior_fusion_list', None)))
    return dict(
        label=row['label'],
        config=row['config'],
        img_backbone_m=rounded(count_params(getattr(forward_projection, 'img_backbone', None))),
        img_neck_m=rounded(count_params(getattr(forward_projection, 'img_neck', None))),
        view_transform_m=rounded(count_params(getattr(forward_projection, 'img_view_transformer', None))),
        encoder3d_m=rounded(count_params(getattr(forward_projection, 'img_bev_encoder_backbone', None))),
        stage_decoder_m=rounded(count_params(getattr(model, 'camera_stage_decoder_list', None))),
        occ_prior_head_m=rounded(head_params),
        total_params_m=rounded(count_params(model)),
        latency_ms=row['latency_ms'])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect component-wise parameter breakdown for ProSD-Occ variants.')
    parser.add_argument(
        '--output-json',
        default='tools/efficiency/prosd_parameter_breakdown.json')
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [build_row(row) for row in ROWS]
    output_dir = osp.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
