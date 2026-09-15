"""Create matched-anchor validation annotations for temporal-gap evaluation.

The source validation set is split into phase-offset streams.  For a stride of
``k``, phase ``p`` contains frames ``p, p + k, p + 2k, ...`` from each scene.
This preserves every input token while making consecutive model updates exactly
``k`` keyframes apart.  A common validity flag keeps only anchors that have
both past and future context at the largest requested stride.
"""

import argparse
import copy
import pickle
from collections import OrderedDict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create temporal-gap zero-shot evaluation annotations')
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--strides', type=int, nargs='+', default=[1, 2, 3])
    return parser.parse_args()


def load_payload(path):
    with path.open('rb') as source_file:
        payload = pickle.load(source_file)
    if not isinstance(payload, dict) or 'infos' not in payload:
        raise ValueError(
            f'{path} must contain a dictionary with an infos key.')
    return payload


def group_scenes(infos):
    scenes = OrderedDict()
    for info in sorted(infos, key=lambda item: int(item['timestamp'])):
        scene_token = info.get('scene_token')
        if scene_token is None:
            raise KeyError('Every info must contain scene_token.')
        scenes.setdefault(scene_token, []).append(info)
    return scenes


def timestamp_scale(metadata):
    scale = float(metadata.get('sample_timestamp_to_sec', 1e-4))
    if scale <= 0.0:
        raise ValueError('sample_timestamp_to_sec must be positive.')
    return scale


def infer_base_gap_seconds(scenes, scale):
    deltas = []
    for scene_infos in scenes.values():
        deltas.extend(
            int(current['timestamp']) - int(previous['timestamp'])
            for previous, current in zip(scene_infos, scene_infos[1:]))
    if not deltas or min(deltas) <= 0:
        raise ValueError('Source scenes must contain increasing timestamps.')
    unique_deltas = sorted(set(deltas))
    if len(unique_deltas) != 1:
        raise ValueError(
            'Temporal-gap evaluation requires a uniform source keyframe gap, '
            f'but found raw timestamp deltas {unique_deltas[:8]}.')
    return unique_deltas[0] * scale


def build_gap_payload(payload, stride, max_stride, source_path):
    if stride <= 0:
        raise ValueError('stride must be positive.')
    scenes = group_scenes(payload['infos'])
    metadata = copy.deepcopy(payload.get('metadata', {}))
    scale = timestamp_scale(metadata)
    base_gap = infer_base_gap_seconds(scenes, scale)

    maximum_scene_span = max(
        int(scene_infos[-1]['timestamp']) - int(scene_infos[0]['timestamp'])
        for scene_infos in scenes.values())
    timestamp_padding = max(int(round(10.0 / scale)), 1)
    stream_span = maximum_scene_span + timestamp_padding
    timestamp_origin = min(int(info['timestamp']) for info in payload['infos'])

    output_infos = []
    stream_index = 0
    for original_scene_token, scene_infos in scenes.items():
        scene_length = len(scene_infos)
        for phase in range(min(stride, scene_length)):
            stream = scene_infos[phase::stride]
            synthetic_scene_token = (
                f'{original_scene_token}__temporal_gap_s{stride}_p{phase}')
            stream_timestamp_origin = timestamp_origin + stream_index * stream_span
            first_original_timestamp = int(stream[0]['timestamp'])
            copied_stream = []
            for original_local_index, source_info in zip(
                    range(phase, scene_length, stride), stream):
                info = copy.deepcopy(source_info)
                info['temporal_gap_original_timestamp'] = int(
                    source_info['timestamp'])
                info[
                    'temporal_gap_original_scene_token'] = original_scene_token
                info[
                    'temporal_gap_original_local_index'] = original_local_index
                info['temporal_gap_eval_valid'] = bool(
                    original_local_index >= max_stride
                    and original_local_index + max_stride < scene_length)
                info['scene_token'] = synthetic_scene_token
                info['timestamp'] = (
                    stream_timestamp_origin + int(source_info['timestamp']) -
                    first_original_timestamp)
                copied_stream.append(info)

            for local_index, info in enumerate(copied_stream):
                info['prev'] = (
                    copied_stream[local_index -
                                  1]['token'] if local_index > 0 else '')
                info['next'] = (
                    copied_stream[local_index + 1]['token'] if local_index +
                    1 < len(copied_stream) else '')
            output_infos.extend(copied_stream)
            stream_index += 1

    metadata['sample_timestamp_to_sec'] = scale
    metadata['temporal_gap_eval'] = {
        'source_ann_file':
        str(source_path.resolve()),
        'stride':
        stride,
        'base_gap_seconds':
        base_gap,
        'expected_gap_seconds':
        base_gap * stride,
        'common_context_stride':
        max_stride,
        'num_original_scenes':
        len(scenes),
        'num_phase_streams':
        stream_index,
        'num_inference_tokens':
        len(output_infos),
        'num_scored_anchors':
        sum(int(info['temporal_gap_eval_valid']) for info in output_infos),
    }
    return {'infos': output_infos, 'metadata': metadata}


def validate_gap_payload(source_payload, gap_payload):
    source_tokens = {info['token'] for info in source_payload['infos']}
    output_infos = gap_payload['infos']
    output_tokens = {info['token'] for info in output_infos}
    if len(output_infos) != len(source_payload['infos']):
        raise AssertionError(
            'The phase streams must retain every source token.')
    if output_tokens != source_tokens:
        raise AssertionError('The phase streams changed the source token set.')

    metadata = gap_payload['metadata']['temporal_gap_eval']
    expected_gap = float(metadata['expected_gap_seconds'])
    scale = timestamp_scale(gap_payload['metadata'])
    scenes = group_scenes(output_infos)
    for scene_infos in scenes.values():
        for previous, current in zip(scene_infos, scene_infos[1:]):
            measured_gap = (int(current['timestamp']) -
                            int(previous['timestamp'])) * scale
            if abs(measured_gap - expected_gap) > 1e-9:
                raise AssertionError(
                    f'Expected {expected_gap:.6f}s but measured '
                    f'{measured_gap:.6f}s from timestamp.')
            if current.get('prev') != previous.get('token'):
                raise AssertionError(
                    'A phase-stream prev link is inconsistent.')

    valid_tokens = {
        info['token']
        for info in output_infos if info.get('temporal_gap_eval_valid', False)
    }
    if len(valid_tokens) != int(metadata['num_scored_anchors']):
        raise AssertionError('The scored-anchor count is inconsistent.')
    return valid_tokens


def main():
    args = parse_args()
    strides = sorted(set(args.strides))
    if not strides or strides[0] <= 0:
        raise ValueError('--strides must contain positive integers.')

    source_payload = load_payload(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common_valid_tokens = None
    for stride in strides:
        gap_payload = build_gap_payload(source_payload, stride, max(strides),
                                        args.source)
        valid_tokens = validate_gap_payload(source_payload, gap_payload)
        if common_valid_tokens is None:
            common_valid_tokens = valid_tokens
        elif valid_tokens != common_valid_tokens:
            raise AssertionError('Scored anchors must match across all gaps.')

        gap_ms = round(gap_payload['metadata']['temporal_gap_eval']
                       ['expected_gap_seconds'] * 1000)
        output_path = args.output_dir / (
            f'v2xreal_infos_val_temporal_gap_{gap_ms:04d}ms.pkl')
        with output_path.open('wb') as output_file:
            pickle.dump(
                gap_payload, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        summary = gap_payload['metadata']['temporal_gap_eval']
        print(f'Wrote {output_path} with {summary["num_inference_tokens"]} '
              f'inference tokens, {summary["num_scored_anchors"]} matched '
              f'anchors, and timestamp gap '
              f'{summary["expected_gap_seconds"]:.1f}s.')


if __name__ == '__main__':
    main()
