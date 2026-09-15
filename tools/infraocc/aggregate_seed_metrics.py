import argparse
import json
import math
import os
from os import path as osp


def parse_args():
    parser = argparse.ArgumentParser(
        description='Aggregate repeated InfraOcc metric JSONs into mean+-std.')
    parser.add_argument('--inputs', nargs='+', required=True, help='Metric JSON files.')
    parser.add_argument('--keys', nargs='+', default=[
        'gIoU', 'mIoU', 'mean_static', 'mean_dynamic', 'free_iou'
    ])
    parser.add_argument('--output-json', default=None, help='Optional output path.')
    return parser.parse_args()


def load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def main():
    args = parse_args()
    payloads = [load_json(path) for path in args.inputs]
    summary = dict(num_runs=len(payloads), metrics={})
    for key in args.keys:
        values = [float(payload[key]) for payload in payloads if key in payload]
        if not values:
            continue
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        summary['metrics'][key] = dict(
            mean=round(mean_value, 4),
            std=round(math.sqrt(variance), 4),
            values=[round(value, 4) for value in values],
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json:
        output_dir = osp.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
