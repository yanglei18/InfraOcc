#!/usr/bin/env python3
"""Visualize audited comparator interventions without cherry-picking metrics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METRICS = ('mIoU', 'Dynamic IoU', 'Static IoU', 'gIoU', 'DMR')


def parse_record(value):
    fields = value.split(',')
    if len(fields) != 7:
        raise argparse.ArgumentTypeError(
            'record must be method,label,mIoU,dynamic,static,gIoU,DMR')
    method, label = fields[:2]
    try:
        values = [float(item) for item in fields[2:]]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return {'method': method, 'label': label, 'values': values}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='append', type=parse_record,
                        required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--summary', required=True)
    return parser.parse_args()


def render(records, output):
    methods = list(dict.fromkeys(item['method'] for item in records))
    if any(sum(item['method'] == method for item in records) < 2
           for method in methods):
        raise ValueError('each method needs at least two audited records')
    figure, axes = plt.subplots(
        len(methods), 2, figsize=(12, 4.2 * len(methods)),
        squeeze=False, constrained_layout=True)
    colors = ('#4c78a8', '#f58518', '#54a24b', '#e45756')
    for row, method in enumerate(methods):
        entries = [item for item in records if item['method'] == method]
        values = np.asarray([item['values'] for item in entries])
        labels = [item['label'] for item in entries]
        x = np.arange(4)
        width = 0.82 / len(entries)
        for index, (label, entry) in enumerate(zip(labels, values)):
            offset = (index - (len(entries) - 1) / 2) * width
            axes[row, 0].bar(x + offset, entry[:4], width,
                             label=label, color=colors[index % len(colors)])
        axes[row, 0].set_xticks(x, METRICS[:4])
        axes[row, 0].set_ylabel('Score (%)')
        axes[row, 0].set_title(f'{method}: occupancy endpoints')
        axes[row, 0].grid(axis='y', alpha=0.25)
        axes[row, 0].legend(fontsize=8)

        dynamic = values[:, 1]
        dmr = values[:, 4]
        axes[row, 1].plot(dynamic, dmr, color='#888888', linewidth=1.5)
        for index, label in enumerate(labels):
            axes[row, 1].scatter(dynamic[index], dmr[index], s=70,
                                 color=colors[index % len(colors)])
            axes[row, 1].annotate(
                label, (dynamic[index], dmr[index]), xytext=(5, 5),
                textcoords='offset points', fontsize=8)
        axes[row, 1].set_xlabel('Dynamic IoU (%)')
        axes[row, 1].set_ylabel('Dynamic match recall (%)')
        axes[row, 1].set_title(f'{method}: dynamic support/identity response')
        axes[row, 1].grid(alpha=0.25)
    figure.suptitle(
        'Comparator Repair Audit: Full Validation Evidence',
        fontsize=15, fontweight='bold')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    render(args.record, args.output)
    payload = {
        'protocol': 'full_validation_comparator_repair_audit',
        'metrics': METRICS,
        'records': args.record,
        'figure': str(args.output),
    }
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
