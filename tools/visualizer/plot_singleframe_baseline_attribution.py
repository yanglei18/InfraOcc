#!/usr/bin/env python3
"""Render structural attribution evidence for single-frame reproduction.

The endpoint panels read completed audited 24-epoch JSON files.  Projection
support remains a separate geometric observation; when its accuracy control is
included, the dashboard reports the completed endpoint without conflating the
support audit itself with accuracy evidence.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRICS = ('miou', 'dynamic_miou', 'static_miou', 'geom_iou')
METRIC_LABELS = ('Overall', 'Dynamic', 'Static', 'Geometric')
DEFAULT_AUDITS = (
    ('Historical Top-K', 'work_dirs/stcroadocc/baseline/'
     'stcocc_c_4x4_24e_main_singleframe_historical_topk_weights/'
     'epoch_24_four_metric_audit.json'),
    ('Width/depth control', 'work_dirs/stcroadocc/baseline/'
     'stcocc_c_4x4_24e_main_singleframe_historical_topk_weights_w64_l111/'
     'epoch_24_four_metric_audit.json'),
    ('Historical data recipe', 'work_dirs/stcroadocc/baseline/'
     'stcocc_c_4x4_24e_main_singleframe_historical_data_recipe/'
     'epoch_24_four_metric_audit.json'),
    ('Spatial-only control', 'work_dirs/stcroadocc/baseline/'
     'stcocc_c_4x4_24e_main_singleframe_historical_data_recipe_spatial_only/'
     'epoch_24_four_metric_audit.json'),
    ('Source projection', 'work_dirs/stcroadocc/baseline/'
     'stcocc_c_4x4_24e_main_singleframe_historical_data_recipe_source_projection/'
     'epoch_24_four_metric_audit.json'),
    ('Independent logits', 'work_dirs/stcroadocc/baseline/'
     'stcocc_c_4x4_24e_main_singleframe_historical_data_recipe_independent_scale_logits/'
     'epoch_24_four_metric_audit.json'),
)
DEFAULT_SUPPORT = ('work_dirs/stcroadocc/baseline/source_projection_audit/'
                   'real_sample_support_audit.json')
DEFAULT_OUTPUT = ('work_dirs/stcroadocc/baseline/'
                  'singleframe_structural_attribution_dashboard.png')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--audit',
        action='append',
        nargs=2,
        metavar=('LABEL', 'JSON'),
        help='Audited epoch-24 endpoint. May be repeated.')
    parser.add_argument('--projection-support', default=DEFAULT_SUPPORT)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--summary', default=None)
    return parser.parse_args()


def load_endpoint(label, path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('endpoint_epoch') != 24:
        raise ValueError(f'{path} is not an epoch-24 endpoint audit')
    sample_check = payload.get('sample_check', {})
    if not sample_check.get('matches_expected'):
        raise ValueError(f'{path} does not cover the canonical validation set')
    metrics = payload.get('metrics', {})
    if any(name not in metrics for name in METRICS):
        raise ValueError(f'{path} is missing one or more required metrics')
    targets = {
        name: payload['metric_checks'][name]['target']
        for name in METRICS
    }
    return {
        'label': label,
        'path': str(path),
        'metrics': {
            name: float(metrics[name])
            for name in METRICS
        },
        'targets': {
            name: float(targets[name])
            for name in METRICS
        },
        'reproduced': bool(payload.get('reproduced')),
    }


def annotate_bars(axis, bars, size=7):
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.45,
            f'{bar.get_height():.2f}',
            ha='center',
            va='bottom',
            fontsize=size,
            rotation=90)


def render(endpoints, support, output):
    target = np.array([endpoints[0]['targets'][name] for name in METRICS])
    values = np.array([[item['metrics'][name] for name in METRICS]
                       for item in endpoints])
    deltas = values - target[None, :]

    figure = plt.figure(figsize=(17, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.08, 0.92))
    endpoint_axis = figure.add_subplot(grid[0, :])
    delta_axis = figure.add_subplot(grid[1, 0])
    support_axis = figure.add_subplot(grid[1, 1])

    x = np.arange(len(METRICS))
    group_width = 0.84
    width = group_width / (len(endpoints) + 1)
    colors = ('#4472c4', '#ed7d31', '#70ad47', '#a64d79', '#5b9bd5', '#ffc000')
    offsets = ((np.arange(len(endpoints) + 1) - len(endpoints) / 2) * width)
    target_bars = endpoint_axis.bar(
        x + offsets[0], target, width, label='Paper target', color='#444444')
    annotate_bars(endpoint_axis, target_bars)
    for index, item in enumerate(endpoints):
        bars = endpoint_axis.bar(
            x + offsets[index + 1],
            values[index],
            width,
            label=item['label'],
            color=colors[index % len(colors)])
        annotate_bars(endpoint_axis, bars)
    endpoint_axis.set_xticks(x, METRIC_LABELS)
    endpoint_axis.set_ylabel('IoU (%)')
    endpoint_axis.set_ylim(0, max(96, float(values.max()) + 7))
    endpoint_axis.set_title(
        'Audited strict single-frame endpoints (24 epochs, 1,970 frames)')
    endpoint_axis.grid(axis='y', alpha=0.25)
    endpoint_axis.legend(
        ncol=min(len(endpoints) + 1, 6), loc='upper left', fontsize=8)

    bound = max(1.0, float(np.abs(deltas).max()))
    image = delta_axis.imshow(
        deltas, cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
    delta_axis.set_xticks(np.arange(len(METRICS)), METRIC_LABELS)
    delta_axis.set_yticks(
        np.arange(len(endpoints)), [item['label'] for item in endpoints])
    delta_axis.set_title('Endpoint error relative to paper row (points)')
    for row in range(deltas.shape[0]):
        for column in range(deltas.shape[1]):
            value = deltas[row, column]
            text_color = 'white' if abs(value) > 0.55 * bound else 'black'
            delta_axis.text(
                column,
                row,
                f'{value:+.2f}',
                ha='center',
                va='center',
                color=text_color,
                fontsize=9)
    figure.colorbar(image, ax=delta_axis, fraction=0.045, pad=0.03)

    source_valid = 100.0 * float(support['source_valid']['fraction'])
    rewrite_valid = 100.0 * float(support['released_valid']['fraction'])
    source_only = 100.0 * float(support['source_only']['fraction'])
    rewrite_only = 100.0 * float(support['released_only']['fraction'])
    support_bars = support_axis.bar((0, 1), (source_valid, rewrite_valid),
                                    color=('#4472c4', '#ed7d31'),
                                    width=0.62)
    for bar in support_bars:
        support_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f'{bar.get_height():.2f}',
            ha='center',
            va='center',
            fontsize=10,
            color='white',
            fontweight='bold')
    support_axis.set_xticks((0, 1), ('Sourced x/W, y/H', 'Rewrite x/H, y/W'))
    support_axis.set_ylabel('Valid calibrated camera queries (%)')
    support_axis.set_ylim(0, max(source_valid, rewrite_valid) + 10)
    support_axis.grid(axis='y', alpha=0.25)
    projection_endpoint = next(
        (item for item in endpoints if 'projection' in item['label'].lower()),
        None)
    projection_status = ('completed 24e control' if projection_endpoint
                         is not None else 'accuracy pending')
    support_axis.set_title(f'Projection support + {projection_status}')
    projection_line = ''
    if projection_endpoint is not None:
        projection_delta = (
            projection_endpoint['metrics']['miou'] -
            projection_endpoint['targets']['miou'])
        projection_line = (
            f'\n• projection control: {projection_endpoint["metrics"]["miou"]:.2f} '
            f'mIoU ({projection_delta:+.2f} vs paper), not reproduced')
    support_axis.text(
        0.02,
        0.96, f'Source-only: {source_only:.2f}%\n'
        f'Rewrite-only: {rewrite_only:.2f}%\n'
        f'Valid-support gap: {source_valid - rewrite_valid:+.2f} pp\n\n'
        'Interpretation boundary:\n'
        '• all completed structural controls miss the four-cell row\n'
        '• spatial-path deletion makes the endpoint stronger\n'
        '• projection provenance materially changes spatial support'
        f'{projection_line}',
        transform=support_axis.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    figure.suptitle(
        'Single-frame Baseline Strength: Structural Attribution Without '
        'Hyperparameter Tuning',
        fontsize=15,
        fontweight='bold')
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return target, values, deltas, {
        'source_valid_percent': source_valid,
        'rewrite_valid_percent': rewrite_valid,
        'valid_support_gap_points': source_valid - rewrite_valid,
        'source_only_percent': source_only,
        'rewrite_only_percent': rewrite_only,
    }


def main():
    args = parse_args()
    audit_specs = args.audit or DEFAULT_AUDITS
    endpoints = [load_endpoint(label, path) for label, path in audit_specs]
    if not endpoints:
        raise ValueError('The dashboard requires at least one endpoint audit')
    first_targets = endpoints[0]['targets']
    if any(item['targets'] != first_targets for item in endpoints[1:]):
        raise ValueError('Endpoint audits do not share the same paper target')

    support_path = Path(args.projection_support)
    support = json.loads(support_path.read_text(encoding='utf-8'))
    output = Path(args.output)
    target, values, deltas, support_summary = render(endpoints, support,
                                                     output)
    summary_path = (
        Path(args.summary) if args.summary else output.with_suffix('.json'))
    summary = {
        'protocol': 'strict_singleframe_structural_attribution',
        'paper_target': dict(zip(METRICS, target.tolist())),
        'endpoints': endpoints,
        'paper_deltas': {
            item['label']: dict(zip(METRICS, row.tolist()))
            for item, row in zip(endpoints, deltas)
        },
        'projection_support': {
            'source':
            str(support_path),
            **support_summary,
            'accuracy_status': ('completed_24_epoch_control' if any(
                'projection' in item['label'].lower()
                for item in endpoints) else 'pending_24_epoch_control'),
        },
        'claims': {
            'reproduction_complete':
            False,
            'completed_controls_reproduce':
            [item['label'] for item in endpoints if item['reproduced']],
            'projection_support_is_accuracy_evidence':
            False,
            'projection_control_evaluated':
            any('projection' in item['label'].lower() for item in endpoints),
        },
        'figure': str(output),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
