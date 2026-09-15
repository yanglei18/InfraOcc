#!/usr/bin/env python3
"""Verify a retained EMA checkpoint against a strict single-frame config."""

import argparse
import importlib
import json
from pathlib import Path


def compare_state_dicts(model_state, checkpoint_state):
    model_keys = set(model_state)
    checkpoint_keys = set(checkpoint_state)
    shared_keys = model_keys & checkpoint_keys
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    shape_mismatch = sorted(
        key for key in shared_keys
        if tuple(model_state[key].shape) != tuple(checkpoint_state[key].shape))
    return dict(
        model_state_entries=len(model_state),
        checkpoint_state_entries=len(checkpoint_state),
        missing_count=len(missing),
        unexpected_count=len(unexpected),
        shape_mismatch_count=len(shape_mismatch),
        missing_examples=missing[:10],
        unexpected_examples=unexpected[:10],
        shape_mismatch_examples=shape_mismatch[:10])


def build_schedule_contract(configured_max_epochs, checkpoint_epoch,
                            expected_epoch):
    configured_max_epochs = int(configured_max_epochs)
    checkpoint_epoch = (None
                        if checkpoint_epoch is None else int(checkpoint_epoch))
    expected_epoch = int(expected_epoch)
    return dict(
        configured_max_epochs=configured_max_epochs,
        checkpoint_is_expected_epoch=checkpoint_epoch == expected_epoch,
        checkpoint_is_terminal_epoch=(
            checkpoint_epoch == configured_max_epochs),
        complete_expected_epoch_recipe=(configured_max_epochs == expected_epoch
                                        and checkpoint_epoch
                                        == expected_epoch),
    )


def audit(config_path, checkpoint_path, expected_epoch):
    import torch
    from mmcv import Config
    from mmdet3d.models import build_model

    cfg = Config.fromfile(str(config_path))
    for module in cfg.get('custom_imports', {}).get('imports', []):
        importlib.import_module(module)
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    checkpoint_state = checkpoint.get('state_dict', checkpoint)
    state_contract = compare_state_dicts(model.state_dict(), checkpoint_state)
    zero_based_epoch = checkpoint.get('epoch')
    one_based_epoch = (
        int(zero_based_epoch) + 1 if zero_based_epoch is not None else None)
    schedule_contract = build_schedule_contract(cfg.runner.max_epochs,
                                                one_based_epoch,
                                                expected_epoch)
    uses_temporal_context = bool(cfg.get('uses_temporal_context', True))
    temporal_fusion = cfg.model.get('temporal_fusion')
    compatible = (
        schedule_contract['checkpoint_is_expected_epoch']
        and not uses_temporal_context and temporal_fusion is None
        and state_contract['missing_count'] == 0
        and state_contract['unexpected_count'] == 0
        and state_contract['shape_mismatch_count'] == 0)
    eligible_complete_recipe_endpoint = (
        compatible and schedule_contract['complete_expected_epoch_recipe'])
    return dict(
        audit_protocol='strict_singleframe_ema_checkpoint_contract',
        compatible=compatible,
        eligible_complete_recipe_endpoint=eligible_complete_recipe_endpoint,
        config=str(config_path),
        checkpoint=str(checkpoint_path),
        checkpoint_bytes=checkpoint_path.stat().st_size,
        checkpoint_epoch_zero_based=zero_based_epoch,
        checkpoint_epoch_one_based=one_based_epoch,
        expected_epoch=int(expected_epoch),
        ema_updates=checkpoint.get('updates'),
        uses_temporal_context=uses_temporal_context,
        temporal_fusion_is_none=temporal_fusion is None,
        **schedule_contract,
        **state_contract)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--expected-epoch', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    report = audit(args.config, args.checkpoint, args.expected_epoch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report['compatible'] else 2)


if __name__ == '__main__':
    main()
