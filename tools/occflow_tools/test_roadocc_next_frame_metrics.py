import importlib.util
from pathlib import Path

import numpy as np


_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / 'projects' / 'RoadOcc' /
    'mmdet3d_plugin' / 'datasets' / 'next_frame_metrics.py')
_SPEC = importlib.util.spec_from_file_location('roadocc_next_frame_metrics', _MODULE_PATH)
_METRICS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_METRICS)
evaluate_next_frame_dynamic_iou = _METRICS.evaluate_next_frame_dynamic_iou
transport_dynamic_semantics = _METRICS.transport_dynamic_semantics


IDENTITY_INFO = dict(
    ego2global_rotation=[1.0, 0.0, 0.0, 0.0],
    ego2global_translation=[0.0, 0.0, 0.0],
)
POINT_CLOUD_RANGE = [0.0, 0.0, 0.0, 4.0, 4.0, 1.0]
EMPTY = 3
DYNAMIC = [1]


def test_transport_dynamic_semantics_applies_flow_and_preserves_class():
    semantics = np.full((4, 4, 1), EMPTY, dtype=np.uint8)
    semantics[1, 1, 0] = 1
    flow = np.zeros((4, 4, 1, 2), dtype=np.float32)
    flow[1, 1, 0, 0] = 1.0

    transported = transport_dynamic_semantics(
        pred_semantics=semantics,
        pred_flow=flow,
        delta_t_sec=1.0,
        current_info=IDENTITY_INFO,
        next_info=IDENTITY_INFO,
        point_cloud_range=POINT_CLOUD_RANGE,
        dynamic_class_indices=DYNAMIC,
        empty_idx=EMPTY,
    )

    assert transported[2, 1, 0] == 1
    assert transported[1, 1, 0] == EMPTY


def test_next_frame_dynamic_iou_is_perfect_for_exact_transport():
    semantics = np.full((4, 4, 1), EMPTY, dtype=np.uint8)
    semantics[1, 1, 0] = 1
    flow = np.zeros((4, 4, 1, 2), dtype=np.float32)
    flow[1, 1, 0, 0] = 1.0
    next_semantics = np.full((4, 4, 1), EMPTY, dtype=np.uint8)
    next_semantics[2, 1, 0] = 1

    metrics = evaluate_next_frame_dynamic_iou(
        pred_semantics_list=[semantics],
        pred_flow_list=[flow],
        next_semantics_list=[next_semantics],
        current_infos=[IDENTITY_INFO],
        next_infos=[IDENTITY_INFO],
        delta_t_sec_list=[1.0],
        point_cloud_range=POINT_CLOUD_RANGE,
        dynamic_class_indices=DYNAMIC,
        empty_idx=EMPTY,
    )

    assert metrics['samples'] == 1
    assert metrics['next_dynamic_iou'] == 100.0
