import argparse
import json
import re
from os import path as osp


def parse_args():
    parser = argparse.ArgumentParser(
        description='Apply collected efficiency metrics into the TPAMI LaTeX table.')
    parser.add_argument('--metrics-json', required=True)
    parser.add_argument('--tex', required=True)
    return parser.parse_args()


def fmt(value):
    if value is None:
        return ''
    return f'{float(value):.2f}'


def normalize_label(label):
    label = re.sub(r'~\\cite\{[^}]+\}', '', label)
    label = re.sub(r'\s+', ' ', label)
    return label.strip()


def main():
    args = parse_args()
    with open(args.metrics_json, 'r', encoding='utf-8') as handle:
        metrics = json.load(handle)
    with open(args.tex, 'r', encoding='utf-8') as handle:
        lines = handle.readlines()

    mapping = {}
    for item in metrics:
        mapping[(item.get('section'), normalize_label(item['label']))] = item
        mapping[(None, normalize_label(item['label']))] = item
    pattern = re.compile(r'^(?P<label>.+?)\s*&\s*(.*?)\\\\\s*$')
    section_pattern = re.compile(r'^\\multicolumn\{5\}\{l\}\{\\textit\{(?P<section>.+?)\}\}\s*\\\\\s*$')
    updated = []
    current_section = None
    for line in lines:
        section_match = section_pattern.match(line.strip())
        if section_match:
            current_section = section_match.group('section')
            updated.append(line)
            continue
        match = pattern.match(line.strip())
        if not match:
            updated.append(line)
            continue
        label = match.group('label')
        normalized_label = normalize_label(label)
        item = mapping.get((current_section, normalized_label))
        if item is None:
            item = mapping.get((None, normalized_label))
        if item is None:
            updated.append(line)
            continue
        updated.append(
            f"{label} & {fmt(item.get('params_m'))} & {fmt(item.get('latency_ms'))} & "
            f"{fmt(item.get('miou_all'))} & {fmt(item.get('miou_dyn'))} \\\\\n")

    with open(args.tex, 'w', encoding='utf-8') as handle:
        handle.writelines(updated)


if __name__ == '__main__':
    main()
