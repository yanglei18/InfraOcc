#!/usr/bin/env python3
"""Export one occupancy-flow prediction from an MMDetection3D result pickle."""

import argparse

import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', help='Result pickle written by tools/test.py.')
    parser.add_argument('output', help='Output NPZ path.')
    parser.add_argument('--index', type=int, default=0, help='Sample index to export.')
    return parser.parse_args()


def unwrap_prediction(value, expected_ndim):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f'Expected one prediction in a result entry, got {len(value)}.')
        value = value[0]
    value = np.asarray(value)
    if value.ndim == expected_ndim + 1 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != expected_ndim:
        raise ValueError(f'Expected {expected_ndim} dimensions, got {value.shape}.')
    return value


def first_available(result, names, expected_ndim):
    for name in names:
        if name in result and result[name] is not None:
            return unwrap_prediction(result[name], expected_ndim)
    raise KeyError(f'None of {names} is present in the result entry.')


def result_sample_index(result, fallback):
    value = result.get('index', fallback)
    while isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 1:
            raise ValueError(f'Expected one result index, got {value}.')
        value = value[0]
    return int(value)


def main():
    args = parse_args()
    results = mmcv.load(args.results)
    result = results[args.index]
    if not isinstance(result, dict):
        raise TypeError(f'Expected a result dictionary, got {type(result).__name__}.')

    occupancy = first_available(
        result, ('pred_occupancy', 'occ_results', 'occ_pred'), expected_ndim=3)
    flow = first_available(
        result, ('pred_flow', 'flow_results', 'flow_pred'), expected_ndim=4)
    if occupancy.shape != flow.shape[:3] or flow.shape[-1] != 2:
        raise ValueError(
            f'Expected occ [X, Y, Z] and flow [X, Y, Z, 2], got '
            f'{occupancy.shape} and {flow.shape}.')
    np.savez(
        args.output,
        occ=occupancy,
        flow=flow,
        sample_index=np.int64(result_sample_index(result, args.index)))


if __name__ == '__main__':
    main()
