#!/usr/bin/env python3
"""Visualize the strict single-frame cross-scale 2x2 structural control."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRICS = ('miou', 'dynamic_miou', 'static_miou', 'geom_iou')
METRIC_LABELS = ('Overall', 'Dynamic', 'Static', 'Geometric')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', required=True)
    parser.add_argument('--no-logits', required=True)
    parser.add_argument('--no-features', required=True)
    parser.add_argument('--neither', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--summary', default=None)
    return parser.parse_args()


def load_audit(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    sample_check = payload.get('sample_check', {})
    if payload.get('endpoint_epoch') != 24:
        raise ValueError(f'{path} is not an epoch-24 endpoint audit')
    if not sample_check.get('matches_expected'):
        raise ValueError(f'{path} is not a complete canonical evaluation')
    metrics = payload.get('metrics', {})
    targets = payload.get('metric_checks', {})
    if any(name not in metrics or name not in targets for name in METRICS):
        raise ValueError(f'{path} is missing a required metric')
    return dict(
        path=str(path),
        metrics=np.asarray([float(metrics[name]) for name in METRICS]),
        target=np.asarray(
            [float(targets[name]['target']) for name in METRICS]))


def compute_factorial(full, no_logits, no_features, neither):
    """Return endpoint errors, conditional main effects, and interaction."""
    target = full['target']
    for item in (no_logits, no_features, neither):
        if not np.allclose(item['target'], target):
            raise ValueError('The four endpoint audits use different targets')
    endpoint_values = np.stack((full['metrics'], no_logits['metrics'],
                                no_features['metrics'], neither['metrics']))
    effects = np.stack((
        no_logits['metrics'] - full['metrics'],
        neither['metrics'] - no_features['metrics'],
        no_features['metrics'] - full['metrics'],
        neither['metrics'] - no_logits['metrics'],
        neither['metrics'] - no_logits['metrics'] - no_features['metrics'] +
        full['metrics'],
    ))
    return target, endpoint_values, effects


def _annotate_heatmap(axis, values, bound):
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = 'white' if abs(value) > 0.55 * bound else 'black'
            axis.text(
                column,
                row,
                f'{value:+.2f}',
                ha='center',
                va='center',
                color=color,
                fontsize=9)


def render(target, endpoint_values, effects, output):
    endpoint_deltas = endpoint_values - target[None, :]
    endpoint_labels = ('Full cascade', 'No logit path', 'No feature path',
                       'Neither path')
    effect_labels = ('Remove logits | features on',
                     'Remove logits | features off',
                     'Remove features | logits on',
                     'Remove features | logits off', '2x2 interaction')

    figure, axes = plt.subplots(
        2, 1, figsize=(12, 8), constrained_layout=True,
        gridspec_kw=dict(height_ratios=(0.9, 1.1)))
    endpoint_bound = max(1.0, float(np.abs(endpoint_deltas).max()))
    endpoint_image = axes[0].imshow(
        endpoint_deltas,
        cmap='RdBu_r',
        vmin=-endpoint_bound,
        vmax=endpoint_bound,
        aspect='auto')
    axes[0].set_xticks(np.arange(len(METRICS)), METRIC_LABELS)
    axes[0].set_yticks(np.arange(4), endpoint_labels)
    axes[0].set_title('Epoch-24 endpoint error relative to paper row (points)')
    _annotate_heatmap(axes[0], endpoint_deltas, endpoint_bound)
    figure.colorbar(endpoint_image, ax=axes[0], fraction=0.025, pad=0.02)

    effect_bound = max(1.0, float(np.abs(effects).max()))
    effect_image = axes[1].imshow(
        effects,
        cmap='RdBu_r',
        vmin=-effect_bound,
        vmax=effect_bound,
        aspect='auto')
    axes[1].set_xticks(np.arange(len(METRICS)), METRIC_LABELS)
    axes[1].set_yticks(np.arange(5), effect_labels)
    axes[1].set_title(
        'Cross-scale path removal effects (negative means a weaker endpoint)')
    _annotate_heatmap(axes[1], effects, effect_bound)
    figure.colorbar(effect_image, ax=axes[1], fraction=0.025, pad=0.02)
    figure.suptitle(
        'Strict Single-Frame Cross-Scale Cascade 2x2 Structural Attribution',
        fontsize=14,
        fontweight='bold')
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return endpoint_deltas


def main():
    args = parse_args()
    inputs = dict(
        full=load_audit(args.full),
        no_logits=load_audit(args.no_logits),
        no_features=load_audit(args.no_features),
        neither=load_audit(args.neither))
    target, endpoint_values, effects = compute_factorial(**inputs)
    output = Path(args.output)
    endpoint_deltas = render(target, endpoint_values, effects, output)
    summary_path = (Path(args.summary) if args.summary else
                    output.with_suffix('.json'))
    summary = dict(
        protocol='strict_singleframe_cross_scale_cascade_2x2',
        endpoint_epoch=24,
        canonical_samples=1970,
        inputs={name: item['path'] for name, item in inputs.items()},
        paper_target=dict(zip(METRICS, target.tolist())),
        endpoint_values={
            name: dict(zip(METRICS, row.tolist()))
            for name, row in zip(inputs, endpoint_values)
        },
        endpoint_deltas={
            name: dict(zip(METRICS, row.tolist()))
            for name, row in zip(inputs, endpoint_deltas)
        },
        effects={
            name: dict(zip(METRICS, row.tolist()))
            for name, row in zip((
                'remove_logits_features_on', 'remove_logits_features_off',
                'remove_features_logits_on', 'remove_features_logits_off',
                'interaction'), effects)
        },
        figure=str(output))
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
