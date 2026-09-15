#!/usr/bin/env python3
"""Compare audited single-frame endpoint dynamic classes with the paper row."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DYNAMIC_CLASSES = (
    'bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'truck')
PAPER_TARGET = {
    'bicycle': 13.33,
    'bus': 49.57,
    'car': 42.64,
    'motorcycle': 10.37,
    'pedestrian': 18.09,
    'truck': 6.67,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--trajectory', action='append', nargs=2, required=True,
        metavar=('LABEL', 'JSON'),
        help='Complete dynamic-class trajectory. May be repeated.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--endpoint-epoch', type=int, default=24)
    parser.add_argument('--expected-samples', type=int, default=1970)
    return parser.parse_args()


def load_endpoint(label, path, endpoint_epoch=24, expected_samples=1970):
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('expected_samples') != expected_samples:
        raise ValueError(f'{path} has a different expected sample count')
    records = [
        record for record in payload.get('records', [])
        if record.get('epoch') == endpoint_epoch
    ]
    if len(records) != 1:
        raise ValueError(
            f'{path} must contain exactly one epoch-{endpoint_epoch} record')
    record = records[0]
    if record.get('samples') != expected_samples:
        raise ValueError(f'{path} does not cover the canonical split')
    classes = record.get('classes', {})
    if set(classes) != set(DYNAMIC_CLASSES):
        raise ValueError(f'{path} has an incomplete dynamic-class ledger')
    values = {name: float(classes[name]) for name in DYNAMIC_CLASSES}
    dynamic_miou = float(record['dynamic_miou'])
    if not np.isclose(
            dynamic_miou, np.mean(list(values.values())), atol=0.01):
        raise ValueError(f'{path} dynamic mean disagrees with its class ledger')
    return {
        'label': label,
        'source': str(path),
        'epoch': endpoint_epoch,
        'samples': expected_samples,
        'dynamic_miou': dynamic_miou,
        'classes': values,
    }


def compute_attribution(endpoints):
    target = np.asarray([PAPER_TARGET[name] for name in DYNAMIC_CLASSES])
    values = np.asarray([[item['classes'][name] for name in DYNAMIC_CLASSES]
                         for item in endpoints])
    deltas = values - target[None, :]
    positive = np.maximum(deltas, 0.0)
    positive_sums = positive.sum(axis=1)
    shares = np.divide(
        positive,
        positive_sums[:, None],
        out=np.zeros_like(positive),
        where=positive_sums[:, None] > 0)
    return target, values, deltas, shares


def render(endpoints, output):
    target, values, deltas, shares = compute_attribution(endpoints)
    figure, (bar_axis, delta_axis) = plt.subplots(
        2, 1, figsize=(13, 8), constrained_layout=True,
        gridspec_kw={'height_ratios': (1.1, 0.9)})

    x = np.arange(len(DYNAMIC_CLASSES))
    width = 0.82 / (len(endpoints) + 1)
    offsets = ((np.arange(len(endpoints) + 1) - len(endpoints) / 2) * width)
    bar_axis.bar(
        x + offsets[0], target, width, color='#444444', label='Paper target')
    colors = ('#4472c4', '#ed7d31', '#70ad47', '#a64d79', '#5b9bd5')
    for index, endpoint in enumerate(endpoints):
        bars = bar_axis.bar(
            x + offsets[index + 1], values[index], width,
            color=colors[index % len(colors)],
            label=f'{endpoint["label"]} (dyn {endpoint["dynamic_miou"]:.2f})')
        for bar in bars:
            bar_axis.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35,
                f'{bar.get_height():.2f}', ha='center', va='bottom',
                rotation=90, fontsize=7)
    bar_axis.set_xticks(x, DYNAMIC_CLASSES)
    bar_axis.set_ylabel('Class IoU (%)')
    bar_axis.set_title('Strict single-frame dynamic classes at epoch 24')
    bar_axis.grid(axis='y', alpha=0.25)
    bar_axis.legend(ncol=2, fontsize=8)

    bound = max(1.0, float(np.abs(deltas).max()))
    image = delta_axis.imshow(
        deltas, cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
    delta_axis.set_xticks(np.arange(len(DYNAMIC_CLASSES)), DYNAMIC_CLASSES)
    delta_axis.set_yticks(
        np.arange(len(endpoints)), [item['label'] for item in endpoints])
    delta_axis.set_title(
        'Endpoint class error vs paper row; parenthesis = share of positive excess')
    for row in range(deltas.shape[0]):
        for column in range(deltas.shape[1]):
            value = deltas[row, column]
            share = 100.0 * shares[row, column]
            color = 'white' if abs(value) > 0.55 * bound else 'black'
            delta_axis.text(
                column, row, f'{value:+.2f}\n({share:.1f}%)',
                ha='center', va='center', fontsize=8, color=color)
    figure.colorbar(image, ax=delta_axis, fraction=0.025, pad=0.02,
                    label='IoU points')
    figure.suptitle(
        'Single-frame Baseline: Dynamic-Class Endpoint Attribution',
        fontsize=15, fontweight='bold')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return target, values, deltas, shares


def main():
    args = parse_args()
    endpoints = [
        load_endpoint(label, path, args.endpoint_epoch, args.expected_samples)
        for label, path in args.trajectory
    ]
    target, _, deltas, shares = render(endpoints, args.output)
    summary = {
        'protocol': 'strict_singleframe_dynamic_class_endpoint_attribution',
        'endpoint_epoch': args.endpoint_epoch,
        'expected_samples': args.expected_samples,
        'paper_target': dict(zip(DYNAMIC_CLASSES, target.tolist())),
        'endpoints': endpoints,
        'paper_deltas': {
            item['label']: dict(zip(DYNAMIC_CLASSES, row.tolist()))
            for item, row in zip(endpoints, deltas)
        },
        'positive_excess_shares': {
            item['label']: dict(zip(DYNAMIC_CLASSES, row.tolist()))
            for item, row in zip(endpoints, shares)
        },
        'figure': str(args.output),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
