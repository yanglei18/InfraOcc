import argparse
import glob
import json
import os
import re
from os import path as osp


PATTERNS = {
    'mIoU': re.compile(r'===> mIoU of \d+ samples: ([0-9.]+)'),
    'mean_dynamic': re.compile(r'===> dynamic mIoU of \d+ samples: ([0-9.]+)'),
    'mean_static': re.compile(r'===> static mIoU of \d+ samples: ([0-9.]+)'),
    'gIoU': re.compile(r'===> geometric IoU of \d+ samples: ([0-9.]+)'),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract final InfraOcc validation metrics from a work_dir log.')
    parser.add_argument(
        '--work-dir', required=True, help='Experiment work directory containing *.log.')
    parser.add_argument(
        '--output-json', default=None, help='Optional path to dump parsed metrics.')
    return parser.parse_args()


def latest_log_path(work_dir):
    candidates = sorted(glob.glob(osp.join(work_dir, '*.log')))
    if not candidates:
        raise FileNotFoundError(f'No log file found under: {work_dir}')
    return candidates[-1]


def parse_log(log_path):
    with open(log_path, 'r', encoding='utf-8') as handle:
        text = handle.read()

    metrics = {}
    for key, pattern in PATTERNS.items():
        matches = pattern.findall(text)
        if not matches:
            raise RuntimeError(f'Failed to locate metric {key} in {log_path}')
        metrics[key] = round(float(matches[-1]), 2)

    return metrics


def main():
    args = parse_args()
    log_path = latest_log_path(args.work_dir)
    metrics = parse_log(log_path)
    payload = {
        'work_dir': osp.abspath(args.work_dir),
        'log_path': osp.abspath(log_path),
        **metrics,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.output_json:
        output_dir = osp.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
