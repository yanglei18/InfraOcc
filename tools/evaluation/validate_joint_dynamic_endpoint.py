#!/usr/bin/env python3
"""Validate the paper-facing artifacts of a completed joint 24e run."""

import argparse
import json
import math
from pathlib import Path

JOINT_TABLE_METRICS = (
    'miou',
    'dynamic_miou',
    'static_miou',
    'geom_iou',
    'rayiou',
    'direct_mave',
    'mave',
    'dynamic_match_recall',
)


def _require_nonempty_file(path, label):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f'{label} is missing or empty: {path}')
    return path


def validate_diagnostics(work_dir, epoch):
    work_dir = Path(work_dir)
    analysis_dir = work_dir / 'analysis'
    summary_path = _require_nonempty_file(
        analysis_dir / f'training_diagnostics_e{epoch}.json',
        'epoch diagnostics JSON')
    figure_path = _require_nonempty_file(
        analysis_dir / f'training_diagnostics_e{epoch}.png',
        'epoch diagnostics PNG')
    try:
        payload = json.loads(summary_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f'epoch diagnostics JSON is invalid: {summary_path}: {exc}'
        ) from exc

    validation = payload.get('latest_validation')
    if not isinstance(validation, dict):
        raise ValueError('latest_validation is missing from epoch diagnostics')
    if int(validation.get('epoch', -1)) != int(epoch):
        raise ValueError(
            f'latest validation is epoch {validation.get("epoch")}, expected {epoch}'
        )

    missing = []
    non_finite = []
    for key in JOINT_TABLE_METRICS:
        value = validation.get(key)
        if value is None:
            missing.append(key)
            continue
        if isinstance(value, bool):
            non_finite.append(key)
            continue
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            non_finite.append(key)
    if missing or non_finite:
        raise ValueError('invalid Joint-table metrics: '
                         f'missing={missing}, non_finite={non_finite}')

    return {
        'summary': str(summary_path),
        'figure': str(figure_path),
        'metrics': {
            key: float(validation[key])
            for key in JOINT_TABLE_METRICS
        },
    }


def _artifact_is_real(split_root, artifact_path):
    if not isinstance(artifact_path, str) or not artifact_path:
        return False
    split_root = Path(split_root).resolve()
    resolved = (split_root / artifact_path).resolve()
    try:
        resolved.relative_to(split_root)
    except ValueError:
        return False
    return resolved.is_file() and resolved.stat().st_size > 0


def validate_same_token_manifest(work_dir, epoch):
    split_root = Path(work_dir) / 'figures_path' / 'test'
    manifest_path = _require_nonempty_file(
        split_root / 'same_token_manifest.jsonl',
        'same-token visualization manifest')
    valid_records = []
    for line_number, line in enumerate(
            manifest_path.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f'invalid manifest JSON at line {line_number}: {exc}') from exc
        if record.get('split') != 'test' or int(record.get('epoch',
                                                           -1)) != int(epoch):
            continue
        if not str(record.get('sample_token', '')).strip():
            continue
        artifacts = record.get('artifacts')
        if not isinstance(artifacts, dict) or not artifacts:
            continue
        if all(
                _artifact_is_real(split_root, artifact_path)
                for artifact_path in artifacts.values()):
            valid_records.append(record)
    if not valid_records:
        raise ValueError(
            f'no real test visualization record exists for epoch {epoch} in '
            f'{manifest_path}')
    return {
        'manifest': str(manifest_path),
        'valid_epoch_records': len(valid_records),
    }


def validate_endpoint(work_dir, epoch=24):
    work_dir = Path(work_dir)
    diagnostics = validate_diagnostics(work_dir, epoch)
    visualization = validate_same_token_manifest(work_dir, epoch)
    return {
        'protocol': 'joint_dynamic_endpoint_contract',
        'work_dir': str(work_dir),
        'epoch': int(epoch),
        'diagnostics': diagnostics,
        'visualization': visualization,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--epoch', type=int, default=24)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        payload = validate_endpoint(args.work_dir, epoch=args.epoch)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f'Joint dynamic endpoint contract failed: {exc}')
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
