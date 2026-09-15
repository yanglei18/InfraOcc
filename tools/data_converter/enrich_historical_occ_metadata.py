#!/usr/bin/env python3
"""Add loader-only occupancy fields to recovered historical info pickles.

The recovered roadside metadata predates the current occupancy dataset loader
and therefore lacks a small set of fields produced by ``create_info_pkl.py``.
This utility preserves every historical field and fills only missing adapter
fields from a token-aligned, loader-ready info pickle.
"""

import argparse
import copy
import os
import pickle
import tempfile


ADAPTER_FIELDS = (
    'ann_infos',
    'gt_ann_tokens',
    'gt_instance_tokens',
    'occ_path',
    'prev',
    'scene_token',
)


def enrich_metadata(reference, adapter):
    """Return a copy of reference with missing loader adapter fields filled."""
    if not isinstance(reference, dict) or 'infos' not in reference:
        raise TypeError('reference must be a metadata dict containing infos')
    if not isinstance(adapter, dict) or 'infos' not in adapter:
        raise TypeError('adapter must be a metadata dict containing infos')

    reference_infos = reference['infos']
    adapter_infos = adapter['infos']
    adapter_by_token = {info['token']: info for info in adapter_infos}
    if len(adapter_by_token) != len(adapter_infos):
        raise ValueError('adapter contains duplicate frame tokens')

    reference_tokens = [info['token'] for info in reference_infos]
    adapter_tokens = [info['token'] for info in adapter_infos]
    if reference_tokens != adapter_tokens:
        raise ValueError('reference and adapter token order must match exactly')

    enriched = copy.deepcopy(reference)
    filled_counts = {field: 0 for field in ADAPTER_FIELDS}
    for info in enriched['infos']:
        source = adapter_by_token[info['token']]
        for field in ADAPTER_FIELDS:
            if field in info:
                continue
            if field not in source:
                raise KeyError(
                    f'adapter token {info["token"]} lacks required {field}')
            info[field] = copy.deepcopy(source[field])
            filled_counts[field] += 1

    metadata = copy.deepcopy(enriched.get('metadata', {}))
    metadata['historical_loader_adapter'] = dict(
        fields=list(ADAPTER_FIELDS),
        sample_count=len(reference_infos),
        token_order_verified=True,
        filled_counts=filled_counts,
    )
    enriched['metadata'] = metadata
    return enriched


def parse_args():
    parser = argparse.ArgumentParser(
        description='Enrich historical occupancy metadata without overwriting it')
    parser.add_argument('--reference', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.abspath(args.output) == os.path.abspath(args.reference):
        raise ValueError('output must not overwrite the historical reference')
    with open(args.reference, 'rb') as file:
        reference = pickle.load(file)
    with open(args.adapter, 'rb') as file:
        adapter = pickle.load(file)
    enriched = enrich_metadata(reference, adapter)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix='.metadata-', suffix='.pkl', dir=output_dir)
    try:
        with os.fdopen(fd, 'wb') as file:
            pickle.dump(enriched, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, args.output)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise

    counts = enriched['metadata']['historical_loader_adapter']['filled_counts']
    print(f'Wrote {len(enriched["infos"])} samples to {args.output}')
    print('Filled fields: ' + ', '.join(
        f'{field}={counts[field]}' for field in ADAPTER_FIELDS))


if __name__ == '__main__':
    main()
