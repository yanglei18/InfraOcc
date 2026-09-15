#!/usr/bin/env python3
"""Plot support, identity, and optimization signals from a training log."""

import argparse
import json
import os
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ITERATION = re.compile(r'Epoch \[(\d+)\]\[(\d+)/(\d+)\]')
VALIDATION = re.compile(r'Epoch\(val\) \[(\d+)\]\[\d+\]')
SCALAR = re.compile(
    r'([A-Za-z_][A-Za-z0-9_]*):\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)',
    re.IGNORECASE)

REQUESTED_TRAINING_METRICS = {
    'loss': ('loss', ),
    'gradient': ('grad_norm', ),
    'dynamic_precision_recall': ('dynamic_precision', 'dynamic_recall'),
    'dynamic_identity':
    ('dynamic_identity_match', 'dynamic_identity_accuracy'),
    'predicted_ground_truth_support':
    ('dynamic_pred_fraction', 'dynamic_target_fraction'),
}
REQUESTED_VALIDATION_METRICS = {
    'overall': ('miou', ),
    'dynamic': ('dynamic_miou', ),
    'static': ('static_miou', ),
    'geometric': ('geom_iou', ),
    'ray': ('rayiou', ),
    'velocity': ('direct_mave', 'mave'),
    'dynamic_match_recall': ('dynamic_match_recall', ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--summary', required=True)
    parser.add_argument(
        '--snapshot-epochs',
        action='store_true',
        help='Also freeze an eN figure/JSON after each completed validation.')
    parser.add_argument(
        '--watch-pid',
        type=int,
        default=0,
        help='Watch the live log until this supervisor PID exits.')
    parser.add_argument('--poll-seconds', type=float, default=30.0)
    return parser.parse_args()


def parse_training(text):
    """Merge the complete trajectory, retaining the latest resumed step."""
    by_step = {}
    for line in text.splitlines():
        match = ITERATION.search(line)
        if match is None:
            continue
        epoch, iteration, total = map(int, match.groups())
        values = {key: float(value) for key, value in SCALAR.findall(line)}
        # ``dynamic_target_fraction`` is the canonical name emitted by the
        # current ALOcc/STCRoadOcc heads.  Keep the legacy alias readable so
        # archived experiments remain comparable, but never hide an actually
        # emitted GT fraction behind the old field name in the live plot.
        if ('dynamic_target_fraction' not in values
                and 'dynamic_gt_fraction' in values):
            values['dynamic_target_fraction'] = values['dynamic_gt_fraction']
        if ('dynamic_pred_fraction' in values
                and values.get('dynamic_target_fraction', 0.0) > 0.0):
            values['dynamic_expansion'] = (
                values['dynamic_pred_fraction'] /
                values['dynamic_target_fraction'])
        values.update(
            epoch=epoch,
            iteration=iteration,
            total=total,
            progress=(epoch - 1) + iteration / total)
        by_step[(epoch, iteration, total)] = values
    records = sorted(
        by_step.values(),
        key=lambda item: (item['progress'], item['iteration']))
    if not records:
        raise ValueError('no training iterations found')
    return records


def parse_validation(text):
    """Extract the full evaluator line emitted after each epoch."""
    by_epoch = {}
    for line in text.splitlines():
        match = VALIDATION.search(line)
        if match is None:
            continue
        epoch = int(match.group(1))
        values = {key: float(value) for key, value in SCALAR.findall(line)}
        values['epoch'] = epoch
        if 'dynamic_match_recall' in values:
            values['dynamic_match_recall_ratio'] = (
                values['dynamic_match_recall'] / 100.0)
        by_epoch[epoch] = values
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def parse_log(path):
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    return parse_training(text), parse_validation(text)


def series(records, key):
    points = [(item['progress'], item[key]) for item in records if key in item]
    return ([item[0] for item in points], [item[1] for item in points])


def plot_if_present(axis, records, key, label=None):
    x, y = series(records, key)
    if x:
        axis.plot(
            x,
            y,
            marker='o',
            markersize=2.5,
            linewidth=1.4,
            label=label or key)


def plot_validation_if_present(axis, records, key, label=None):
    points = [(item['epoch'], item[key]) for item in records if key in item]
    if points:
        axis.plot([item[0] for item in points], [item[1] for item in points],
                  marker='o',
                  markersize=3.0,
                  linewidth=1.4,
                  label=label or key)


def annotate_missing(axis, labels):
    if not labels:
        return
    axis.text(
        0.5,
        0.5,
        'Not emitted by log:\n' + '\n'.join(labels),
        ha='center',
        va='center',
        transform=axis.transAxes,
        fontsize=8,
        color='dimgray')


def metric_availability(records, requested):
    available_keys = set().union(*(record.keys() for record in records)) \
        if records else set()
    groups = {}
    missing = []
    for group, keys in requested.items():
        present = [key for key in keys if key in available_keys]
        absent = [key for key in keys if key not in available_keys]
        groups[group] = {'present': present, 'missing': absent}
        missing.extend(absent)
    return groups, missing


def render(records, validation_records, output):
    figure, axes = plt.subplots(
        2, 3, figsize=(16, 8.4), constrained_layout=True)
    axes = axes.ravel()
    for key in ('loss', 'loss_dense_semantic', 'loss_dynamic_support_focal',
                'loss_dynamic_bce', 'loss_dynamic_dice',
                'loss_dynamic_identity', 'loss_dynamic_classification'):
        plot_if_present(axes[0], records, key)
    axes[0].set_yscale('symlog', linthresh=0.05)
    axes[0].set_title('Joint objectives')
    axes[0].set_ylabel('Loss (symlog)')

    for key, label in (('dynamic_precision', 'support precision'),
                       ('dynamic_recall', 'support recall')):
        plot_if_present(axes[1], records, key, label)
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_title('Dynamic support')
    axes[1].set_ylabel('Ratio')

    for key, label in (('dynamic_identity_match', 'exact identity match'),
                       ('dynamic_identity_accuracy',
                        'conditional identity accuracy')):
        plot_if_present(axes[2], records, key, label)
    axes[2].set_ylim(bottom=0.0)
    axes[2].set_title('Dynamic identity')
    axes[2].set_ylabel('Ratio')
    if not axes[2].get_legend_handles_labels()[0]:
        annotate_missing(axes[2], ['dynamic identity'])

    for key, label, color in (
            ('dynamic_pred_fraction', 'predicted support', 'tab:blue'),
            ('dynamic_target_fraction', 'GT support', 'tab:orange')):
        x, y = series(records, key)
        if x:
            axes[3].plot(
                x,
                y,
                marker='o',
                markersize=2.5,
                linewidth=1.4,
                color=color,
                label=label)
    expansion_axis = axes[3].twinx()
    x, y = series(records, 'dynamic_expansion')
    if x:
        expansion_axis.plot(
            x,
            y,
            marker='o',
            markersize=2.5,
            linewidth=1.4,
            color='tab:green',
            label='predicted / GT support')
    expansion_axis.set_ylabel('Support expansion ratio')
    axes[3].set_ylim(bottom=0.0)
    axes[3].set_title('Predicted / GT support')
    axes[3].set_ylabel('Fraction')
    missing_support = [
        key for key in ('dynamic_pred_fraction', 'dynamic_target_fraction')
        if not series(records, key)[0]
    ]
    annotate_missing(axes[3], missing_support)

    plot_if_present(axes[4], records, 'grad_norm', 'gradient norm')
    learning_axis = axes[4].twinx()
    plot_if_present(learning_axis, records, 'lr', 'learning rate')
    axes[4].set_yscale('symlog', linthresh=1.0)
    axes[4].set_title('Optimization health')
    axes[4].set_ylabel('Gradient norm (symlog)')
    learning_axis.set_ylabel('Learning rate')

    for key, label in (('miou', 'mIoU'), ('dynamic_miou', 'Dynamic IoU'),
                       ('static_miou', 'Static IoU'), ('geom_iou', 'gIoU'),
                       ('dynamic_match_recall', 'DMR')):
        plot_validation_if_present(axes[5], validation_records, key, label)
    axes[5].set_title('Full validation')
    axes[5].set_ylabel('Metric (%)')
    if not axes[5].get_legend_handles_labels()[0]:
        annotate_missing(axes[5], ['completed validation'])

    for axis in axes:
        axis.set_xlabel('Epoch progress')
        axis.grid(alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    if learning_axis.get_legend_handles_labels()[0]:
        learning_axis.legend(loc='upper right', fontsize=7)
    if axes[3].get_legend_handles_labels()[0]:
        axes[3].legend(loc='upper left', fontsize=7)
    if expansion_axis.get_legend_handles_labels()[0]:
        expansion_axis.legend(loc='lower right', fontsize=7)
    figure.suptitle(
        'Joint Dynamic Training Diagnostics', fontsize=14, fontweight='bold')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    figure.savefig(temporary, dpi=180, format=output.suffix.lstrip('.'))
    plt.close(figure)
    temporary.replace(output)


def epoch_path(path, epoch):
    path = Path(path)
    stem = path.stem
    if stem.endswith('_latest'):
        stem = stem[:-len('_latest')]
    return path.with_name(f'{stem}_e{epoch}{path.suffix}')


def write_summary(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def build_payload(args, records, validation_records, figure):
    tail = records[-4:]
    tail_keys = ('dynamic_precision', 'dynamic_recall',
                 'dynamic_identity_match', 'dynamic_identity_accuracy',
                 'dynamic_pred_fraction', 'dynamic_target_fraction',
                 'dynamic_expansion', 'loss_dynamic_support_focal', 'loss',
                 'grad_norm')
    tail_average = {}
    for key in tail_keys:
        values = [item[key] for item in tail if key in item]
        if values:
            tail_average[key] = sum(values) / len(values)
    training_availability, missing_training = metric_availability(
        records, REQUESTED_TRAINING_METRICS)
    validation_availability, missing_validation = metric_availability(
        validation_records, REQUESTED_VALIDATION_METRICS)
    return {
        'protocol':
        'joint_dynamic_training_and_validation_diagnostics',
        'source_log':
        args.log,
        'num_records':
        len(records),
        'last_record':
        records[-1],
        'tail_records':
        len(tail),
        'tail_average':
        tail_average,
        'num_validation_records':
        len(validation_records),
        'validation_epochs': [item['epoch'] for item in validation_records],
        'latest_validation':
        (validation_records[-1] if validation_records else None),
        'validation_records':
        validation_records,
        'metric_availability': {
            'training': training_availability,
            'validation': validation_availability,
        },
        'missing_requested_metrics': {
            'training': missing_training,
            'validation': missing_validation,
        },
        'figure':
        str(figure),
    }


def generate(args):
    records, validation_records = parse_log(args.log)
    render(records, validation_records, args.output)
    payload = build_payload(args, records, validation_records, args.output)
    write_summary(payload, args.summary)

    if args.snapshot_epochs:
        for validation in validation_records:
            epoch = validation['epoch']
            epoch_output = epoch_path(args.output, epoch)
            epoch_summary = epoch_path(args.summary, epoch)
            if epoch_output.is_file() and epoch_summary.is_file():
                try:
                    frozen = json.loads(epoch_summary.read_text())
                except (OSError, json.JSONDecodeError):
                    frozen = {}
                if frozen.get('latest_validation') == validation:
                    continue
            epoch_records = [
                item for item in records if item['epoch'] <= epoch
            ]
            epoch_validations = [
                item for item in validation_records if item['epoch'] <= epoch
            ]
            render(epoch_records, epoch_validations, epoch_output)
            epoch_payload = build_payload(args, epoch_records,
                                          epoch_validations, epoch_output)
            write_summary(epoch_payload, epoch_summary)
    return payload


def process_exists(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main():
    args = parse_args()
    if args.watch_pid <= 0:
        print(json.dumps(generate(args), indent=2))
        return

    last_signature = None
    while process_exists(args.watch_pid):
        path = Path(args.log)
        if path.is_file():
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if signature != last_signature:
                try:
                    payload = generate(args)
                except ValueError:
                    pass
                else:
                    last_signature = signature
                    print(json.dumps(payload, indent=2), flush=True)
        time.sleep(max(args.poll_seconds, 1.0))
    if Path(args.log).is_file():
        try:
            print(json.dumps(generate(args), indent=2))
        except ValueError:
            pass


if __name__ == '__main__':
    main()
