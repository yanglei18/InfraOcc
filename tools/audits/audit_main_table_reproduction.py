#!/usr/bin/env python3
"""Audit a full main-table endpoint against its four published metrics.

The audit deliberately reports every metric instead of allowing a checkpoint
that happens to match just mIoU. It is model-agnostic: callers provide a
completed validation log and the script records whether its selected epoch
matches the declared four-way target after two-decimal reporting.
"""

import argparse
import json
import re
from pathlib import Path

TARGETS = {
    'miou': 49.70,
    'dynamic_miou': 23.45,
    'static_miou': 69.39,
    'geom_iou': 73.11,
}
METRIC_RE = re.compile(r'([a-z_]+):\s*(-?\d+(?:\.\d+)?)')
ROW_RE = re.compile(r'Epoch\(val\)\s*\[(\d+)\]\[\d+\]\s+(.*)')
SUMMARY_RE = re.compile(
    r'===>\s+(mIoU|dynamic mIoU|static mIoU|geometric IoU)\s+'
    r'of\s+(\d+)\s+samples:\s*(-?\d+(?:\.\d+)?)')
SUMMARY_KEYS = {
    'mIoU': 'miou',
    'dynamic mIoU': 'dynamic_miou',
    'static mIoU': 'static_miou',
    'geometric IoU': 'geom_iou',
}


def report_is_complete_reproduction(report,
                                    endpoint_epoch=24,
                                    expected_samples=1970,
                                    tolerance=0.005):
    """Validate a persisted audit before it can release a training queue."""
    if (report.get('audit_protocol') !=
            'full_24_epoch_four_metric_main_table'):
        return False
    if report.get('endpoint_epoch') != int(endpoint_epoch):
        return False
    sample_check = report.get('sample_check', {})
    if (sample_check.get('expected') != int(expected_samples)
            or sample_check.get('matches_expected') is not True):
        return False
    if report.get('reproduced') is not True:
        return False
    metrics = report.get('metrics', {})
    metric_checks = report.get('metric_checks', {})
    for name, target in TARGETS.items():
        if name not in metrics or name not in metric_checks:
            return False
        check = metric_checks[name]
        if (abs(float(metrics[name]) - target) > float(tolerance)
                or abs(float(check.get('target', float('inf'))) - target) >
                float(tolerance)
                or check.get('two_decimal_match') is not True):
            return False
    return True


def parse_log_records(log_path):
    """Return validation rows and unpaired standalone metric summaries."""
    records = []
    standalone_records = []
    summary_samples = {}
    summary_metrics = {}
    for line in Path(log_path).read_text(encoding='utf-8').splitlines():
        summary = SUMMARY_RE.search(line)
        if summary:
            key = SUMMARY_KEYS[summary.group(1)]
            if key in summary_metrics:
                if set(summary_metrics) == set(TARGETS):
                    standalone_records.append({
                        'metrics': dict(summary_metrics),
                        'summary_samples': dict(summary_samples),
                    })
                summary_samples = {}
                summary_metrics = {}
            summary_samples[key] = int(summary.group(2))
            summary_metrics[key] = float(summary.group(3))
            continue

        row = ROW_RE.search(line)
        if not row:
            continue
        metrics = {
            key: float(value)
            for key, value in METRIC_RE.findall(row.group(2))
        }
        if set(metrics) != set(TARGETS):
            continue
        records.append({
            'epoch': int(row.group(1)),
            'metrics': metrics,
            'summary_samples': dict(summary_samples),
        })
        summary_samples = {}
        summary_metrics = {}
    if set(summary_metrics) == set(TARGETS):
        standalone_records.append({
            'metrics': dict(summary_metrics),
            'summary_samples': dict(summary_samples),
        })
    return records, standalone_records


def parse_validation_rows(log_path):
    """Return complete ``Epoch(val)`` rows with local sample evidence."""
    return parse_log_records(log_path)[0]


def audit(log_paths,
          epoch=24,
          expected_samples=1970,
          tolerance=0.005,
          allow_standalone_eval=False):
    """Select the last complete requested epoch and return its audit record."""
    matches = []
    standalone_matches = []
    for log_path in log_paths:
        validation_records, standalone_records = parse_log_records(log_path)
        for record in validation_records:
            if record['epoch'] == epoch:
                record['source'] = str(log_path)
                record['provenance'] = 'epoch_validation_row'
                matches.append(record)
        for record in standalone_records:
            record['source'] = str(log_path)
            record['provenance'] = 'standalone_full_evaluation'
            standalone_matches.append(record)
    if not matches:
        if not allow_standalone_eval:
            raise ValueError(f'no complete validation row for epoch {epoch}')
        if len(standalone_matches) != 1:
            raise ValueError(
                'standalone evaluation must contain exactly one unambiguous '
                f'four-metric summary block; found {len(standalone_matches)}')
        matches = standalone_matches

    selected = matches[-1]
    metric_checks = {
        name: {
            'value':
            selected['metrics'][name],
            'target':
            target,
            'delta':
            round(selected['metrics'][name] - target, 6),
            'two_decimal_match':
            abs(selected['metrics'][name] - target) <= tolerance,
        }
        for name, target in TARGETS.items()
    }
    sample_values = selected['summary_samples']
    sample_check = {
        'expected':
        expected_samples,
        'values':
        sample_values,
        'complete':
        set(sample_values) == set(TARGETS),
        'consistent': (set(sample_values) == set(TARGETS)
                       and len(set(sample_values.values())) == 1),
        'matches_expected': (set(sample_values) == set(TARGETS)
                             and all(value == expected_samples
                                     for value in sample_values.values())),
    }
    reproduced = (
        sample_check['matches_expected']
        and all(check['two_decimal_match']
                for check in metric_checks.values()))
    return {
        'audit_protocol': 'full_24_epoch_four_metric_main_table',
        'endpoint_epoch': epoch,
        'source': selected['source'],
        'provenance': selected['provenance'],
        'metrics': selected['metrics'],
        'metric_checks': metric_checks,
        'sample_check': sample_check,
        'reproduced': reproduced,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Audit a main-table four-metric reproduction endpoint.')
    parser.add_argument('--logs', type=Path, nargs='+', required=True)
    parser.add_argument('--epoch', type=int, default=24)
    parser.add_argument('--expected-samples', type=int, default=1970)
    parser.add_argument('--tolerance', type=float, default=0.005)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-match', action='store_true')
    parser.add_argument(
        '--allow-standalone-eval',
        action='store_true',
        help=('accept one complete, unambiguous four-metric summary from an '
              'independent evaluation when no Epoch(val) row is present'))
    args = parser.parse_args()

    try:
        report = audit(
            args.logs,
            epoch=args.epoch,
            expected_samples=args.expected_samples,
            tolerance=args.tolerance,
            allow_standalone_eval=args.allow_standalone_eval)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, sort_keys=True))
    if args.require_match and not report['reproduced']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
