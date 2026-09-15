#!/usr/bin/env python3
"""Extract one complete 19-cell occupancy result from one evaluation log."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

CLASS_NAMES = ('others', 'barrier', 'bicycle', 'bus', 'car', 'motorcycle',
               'pedestrian', 'traffic_cone', 'truck', 'driveable_surface',
               'sidewalk', 'terrain', 'manmade', 'vegetation', 'empty')
AGGREGATE_NAMES = ('mIoU', 'dynamic_mIoU', 'static_mIoU', 'geometric_IoU')
HEADER_RE = re.compile(r'===> per class IoU of (\d+) samples:')
CLASS_RE = re.compile(r'===>\s+([A-Za-z0-9_]+)\s+-\s+IoU\s+=\s+'
                      r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
AGGREGATE_RES = {
    'mIoU':
    re.compile(r'===> mIoU of (\d+) samples:\s+([-+\d.eE]+)'),
    'dynamic_mIoU':
    re.compile(r'===> dynamic mIoU of (\d+) samples:\s+([-+\d.eE]+)'),
    'static_mIoU':
    re.compile(r'===> static mIoU of (\d+) samples:\s+([-+\d.eE]+)'),
    'geometric_IoU':
    re.compile(r'===> geometric IoU of (\d+) samples:\s+([-+\d.eE]+)'),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--expected-samples', type=int, default=1970)
    parser.add_argument('--require-checkpoint-mention', action='store_true')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_complete_blocks(text, expected_samples):
    blocks = []
    current = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        header_match = HEADER_RE.search(line)
        if header_match:
            current = dict(
                start_line=line_number,
                sample_count=int(header_match.group(1)),
                classes={},
                aggregates={},
                aggregate_sample_counts={})
            continue
        if current is None:
            continue
        class_match = CLASS_RE.search(line)
        if class_match:
            name = class_match.group(1)
            if name == 'free':
                name = 'empty'
            if name in CLASS_NAMES:
                if name in current['classes']:
                    current = None
                    continue
                current['classes'][name] = float(class_match.group(2))
            continue
        for metric_name, pattern in AGGREGATE_RES.items():
            aggregate_match = pattern.search(line)
            if aggregate_match:
                current['aggregate_sample_counts'][metric_name] = int(
                    aggregate_match.group(1))
                current['aggregates'][metric_name] = float(
                    aggregate_match.group(2))
                if metric_name == 'geometric_IoU':
                    current['end_line'] = line_number
                    complete_names = (
                        set(current['classes']) == set(CLASS_NAMES)
                        and set(current['aggregates']) == set(AGGREGATE_NAMES))
                    sample_counts = [current['sample_count']]
                    sample_counts.extend(
                        current['aggregate_sample_counts'].values())
                    complete_samples = all(count == expected_samples
                                           for count in sample_counts)
                    all_values = list(current['classes'].values())
                    all_values.extend(current['aggregates'].values())
                    if (complete_names and complete_samples and all(
                            math.isfinite(value) for value in all_values)):
                        blocks.append(current)
                    current = None
                break
    return blocks


def main():
    args = parse_args()
    log_path = Path(args.log)
    checkpoint_path = Path(args.checkpoint)
    text = log_path.read_text(errors='replace')
    if (args.require_checkpoint_mention and checkpoint_path.name not in text
            and str(checkpoint_path) not in text
            and str(checkpoint_path.resolve()) not in text):
        raise AssertionError(
            'Evaluation log does not mention the audited checkpoint.')
    blocks = parse_complete_blocks(text, args.expected_samples)
    if not blocks:
        raise AssertionError('No contiguous complete 19-cell block with '
                             f'{args.expected_samples} samples was found.')
    selected = blocks[-1]
    ordered_cells = [
        dict(name=name, value=selected['classes'][name])
        for name in CLASS_NAMES
    ]
    ordered_cells.extend(
        dict(name=name, value=selected['aggregates'][name])
        for name in AGGREGATE_NAMES)
    if len(ordered_cells) != 19:
        raise AssertionError('The canonical result must contain 19 cells.')

    report = dict(
        audit='main_table_same_checkpoint_19_cells',
        passed=True,
        checkpoint=str(checkpoint_path.resolve()),
        checkpoint_sha256=sha256(checkpoint_path),
        evaluation_log=str(log_path.resolve()),
        evaluation_log_sha256=sha256(log_path),
        expected_samples=args.expected_samples,
        complete_blocks_found=len(blocks),
        selected_block_lines=[selected['start_line'], selected['end_line']],
        cell_count=len(ordered_cells),
        ordered_cells=ordered_cells)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
