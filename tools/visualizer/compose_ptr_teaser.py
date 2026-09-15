#!/usr/bin/env python3
"""Compose the RoadOcc P/T/R teaser directly from lossless source data."""

import argparse
import pickle
import sys
import zlib
from pathlib import Path

sys.dont_write_bytecode = True

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmdet3d.core.visualizer.occ_visualization import (  # noqa: E402
    _orient_xy_grid_to_image, project_dynamic_flow_to_bev,
    project_occ_to_bev_labels, render_flow_bev)
from mmdet3d.datasets.utils import nuscenes_get_rt_matrix  # noqa: E402
from projects.STCRoadOcc.mmdet3d_plugin.datasets.pipelines.ptr_state_targets import (  # noqa: E402,E501
    LoadPTRPreviousOccTarget)

TOKEN = '161da02f-bbfe-4edb-b45c-6d87137cc28e'
ANN_FILE = (
    ROOT / 'data/v2xreal_nuscenes/'
    'reference-v2xreal_infos_val_occ_enriched.pkl')
FLOW_ROOT = ROOT / 'data/v2xreal_nuscenes/occflow_gts'
OUTPUT_PNG = (
    ROOT / 'docs/paper/ICLR2027/figures/'
    'roadocc_ptr_teaser.png')
OUTPUT_PDFS = (
    ROOT / 'docs/paper/ICLR2027/Fig1-ptrteaser.pdf',
    ROOT / 'docs/paper/ICLR2027/figures/roadocc_ptr_teaser.pdf',
)
APPROVED_TOKEN = '287c5fd3-2afe-47cf-9c24-43c87bf4b168'
APPROVED_HISTORY_STEPS = 4

CLASS_NAMES = ('others', 'barrier', 'bicycle', 'bus', 'car',
               'construction_vehicle', 'motorcycle', 'pedestrian',
               'traffic_cone', 'trailer', 'truck', 'driveable_surface',
               'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation',
               'free')
DYNAMIC_INDICES = (2, 3, 4, 6, 7, 10)
EMPTY_IDX = 17
POINT_CLOUD_RANGE = (-64.0, -64.0, -4.8, 64.0, 64.0, 1.6)
GRID_SIZE = 320
PANEL_SIZE = 420
TITLE_HEIGHT = 54
GAP = 14
CANVAS_SIZE = (3 * PANEL_SIZE + 2 * GAP,
               2 * PANEL_SIZE + 2 * TITLE_HEIGHT + GAP)
HISTORY_STEPS = 5
FRAME_DT = 0.5

TEXT = (27, 36, 46)
BORDER = (205, 214, 223)
OUTLINE = (27, 36, 44)
P_COLOR = (54, 116, 217)
T_COLOR = (232, 142, 38)
R_COLOR = (44, 160, 92)
HISTORY_COLOR = (105, 113, 122)

# The semantic palette is intentionally interpreted as display RGB here.  It
# is the palette used by the existing RoadOcc semantic visualizer, in which
# cars are orange and buses are cyan.
OCC_COLORS = np.asarray(
    ((0, 0, 0), (50, 120, 255), (203, 192, 255), (0, 255, 255), (245, 150, 0),
     (255, 255, 0), (0, 127, 255), (0, 0, 255), (150, 240, 255), (0, 60, 135),
     (240, 32, 160), (255, 0, 255), (137, 137, 139), (75, 0, 75),
     (80, 240, 150), (250, 230, 230), (0, 175, 0), (255, 255, 255)),
    dtype=np.uint8)

GRAY_COLORS = np.asarray(
    ((159, 159, 159), (205, 205, 205), (220, 220, 220), (205, 205, 205),
     (205, 205, 205), (212, 212, 212), (205, 205, 205), (205, 205, 205),
     (218, 218, 218), (205, 205, 205), (205, 205, 205), (196, 196, 196),
     (207, 207, 207), (170, 170, 170), (223, 223, 223), (242, 242, 242),
     (195, 195, 195), (255, 255, 255)),
    dtype=np.uint8)

FONT_REGULAR = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
FONT_BOLD = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token', default=TOKEN)
    parser.add_argument('--ann-file', type=Path, default=ANN_FILE)
    parser.add_argument('--flow-root', type=Path, default=FLOW_ROOT)
    parser.add_argument('--output', type=Path, default=OUTPUT_PNG)
    parser.add_argument('--history-steps', type=int, default=HISTORY_STEPS)
    parser.add_argument(
        '--approved-source',
        type=Path,
        default=None,
        help='Recompose the previously approved lossless 2x3 sample.')
    parser.add_argument('--dpi', type=float, default=300.0)
    return parser.parse_args()


def load_font(path, size):
    if not path.is_file():
        raise FileNotFoundError(path)
    return ImageFont.truetype(str(path), size=size)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_labels(path):
    with np.load(path, allow_pickle=False) as payload:
        semantics = payload['semantics'].astype(np.int64)
        visibility = (
            payload['mask_camera'].astype(bool)
            if 'mask_camera' in payload.files else semantics != 255)
    return semantics, visibility


def align_history(current_info, history_info, history_semantics,
                  history_visibility, output_shape):
    loader = LoadPTRPreviousOccTarget(
        point_cloud_range=POINT_CLOUD_RANGE,
        scale=1,
        empty_idx=EMPTY_IDX,
        load_flow=False)
    transform = dict(
        curr_to_prev_lidar_rt=torch.as_tensor(
            nuscenes_get_rt_matrix(current_info, history_info, 'lidar',
                                   'lidar'),
            dtype=torch.float32),
        bda_mat=torch.eye(4))
    semantics, visibility = loader._sample_previous(history_semantics,
                                                    history_visibility,
                                                    transform, output_shape)
    return np.asarray(
        semantics, dtype=np.int64), np.asarray(
            visibility, dtype=bool)


def project_labels(semantics):
    return _orient_xy_grid_to_image(
        project_occ_to_bev_labels(semantics, EMPTY_IDX))


def resize_nearest(image, size=PANEL_SIZE):
    return cv2.resize(
        np.asarray(image), (size, size), interpolation=cv2.INTER_NEAREST)


def semantic_panel(semantics, color_dynamic=True, outline_dynamic=True):
    labels = project_labels(semantics)
    safe = np.clip(labels, 0, len(OCC_COLORS) - 1)
    image = GRAY_COLORS[safe]
    invalid = (labels < 0) | (labels >= len(OCC_COLORS))
    image[invalid] = 255
    dynamic = np.isin(labels, DYNAMIC_INDICES)
    if color_dynamic:
        image[dynamic] = OCC_COLORS[safe[dynamic]]
    image = resize_nearest(image)
    dynamic_large = resize_nearest(dynamic.astype(np.uint8)).astype(bool)
    if outline_dynamic:
        contours, _ = cv2.findContours(
            dynamic_large.astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, OUTLINE, 2, cv2.LINE_8)
    return image, labels


def connected_components(labels, class_index=4, min_area=5):
    count, component_map, stats, centroids = cv2.connectedComponentsWithStats(
        (labels == class_index).astype(np.uint8), connectivity=8)
    components = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, width, height = (int(value) for value in stats[index, :4])
        components.append(
            dict(
                index=index,
                area=area,
                box=(x, y, x + width - 1, y + height - 1),
                center=tuple(float(value) for value in centroids[index]),
                mask=component_map == index))
    return components


def nearest_component(components, point):
    point = np.asarray(point, dtype=np.float32)
    return min(
        components,
        key=lambda component: np.linalg.norm(
            np.asarray(component['center'], dtype=np.float32) - point))


def point_box_distance(point, box):
    """Euclidean distance from an image point to an inclusive box."""
    point = np.asarray(point, dtype=np.float32)
    lower = np.asarray(box[:2], dtype=np.float32)
    upper = np.asarray(box[2:], dtype=np.float32)
    return float(np.linalg.norm(point - np.clip(point, lower, upper)))


def union_box(boxes):
    """Tight inclusive box containing every box in ``boxes``."""
    return (min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes))


def scaled_box(box, padding=5):
    scale = PANEL_SIZE / float(GRID_SIZE)
    x0, y0, x1, y1 = box
    return (
        max(int(np.floor(x0 * scale)) - padding, 0),
        max(int(np.floor(y0 * scale)) - padding, 0),
        min(int(np.ceil((x1 + 1) * scale)) + padding - 1, PANEL_SIZE - 1),
        min(int(np.ceil((y1 + 1) * scale)) + padding - 1, PANEL_SIZE - 1),
    )


def component_flow_source(component, flow, semantics, delta_t):
    flow_bev, _ = project_dynamic_flow_to_bev(
        flow, semantics, DYNAMIC_INDICES, fill_iterations=0)
    flow_image = _orient_xy_grid_to_image(flow_bev)
    vectors = flow_image[component['mask']]
    moving = np.linalg.norm(vectors, axis=-1) > 1e-3
    vectors = vectors[moving]
    if not len(vectors):
        return component['center']
    # Image coordinates reverse the corresponding lidar axes.  Starting from
    # the target image address, a backward source read adds v*dt/voxel_size.
    voxel_size = ((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / GRID_SIZE)
    displacement = np.median(vectors, axis=0) * float(delta_t) / voxel_size
    centre = np.asarray(component['center']) + displacement
    return tuple(float(value) for value in centre)


def crop_camera(path):
    image = Image.open(path).convert('RGB')
    return ImageOps.fit(
        image, (PANEL_SIZE // 2, PANEL_SIZE // 2),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.52))


def add_camera_label(panel, xy, text):
    draw = ImageDraw.Draw(panel, 'RGBA')
    font = load_font(FONT_BOLD, 24)
    x, y = xy
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.rounded_rectangle((x, y, x + width + 16, y + height + 13),
                           radius=4,
                           fill=(15, 22, 29, 232))
    draw.text((x + 8, y + 4), text, font=font, fill=(255, 255, 255, 255))


def camera_panel(current_info):
    panel = Image.new('RGB', (PANEL_SIZE, PANEL_SIZE), 'white')
    order = (('CAM_FRONT', 'Front', (0, 0)), ('CAM_LEFT', 'Left',
                                              (PANEL_SIZE // 2, 0)),
             ('CAM_BACK', 'Back', (0, PANEL_SIZE // 2)),
             ('CAM_RIGHT', 'Right', (PANEL_SIZE // 2, PANEL_SIZE // 2)))
    for key, label, position in order:
        camera = crop_camera(resolve(current_info['cams'][key]['data_path']))
        panel.paste(camera, position)
        add_camera_label(panel, (position[0] + 7, position[1] + 7), label)
    draw = ImageDraw.Draw(panel)
    draw.line((PANEL_SIZE // 2, 0, PANEL_SIZE // 2, PANEL_SIZE),
              fill=(255, 255, 255),
              width=4)
    draw.line((0, PANEL_SIZE // 2, PANEL_SIZE, PANEL_SIZE // 2),
              fill=(255, 255, 255),
              width=4)
    return np.asarray(panel)


def metric_to_panel(xy):
    """Register lidar XY to the approved roadside occupancy-map crop."""
    x, y = float(xy[0]), float(xy[1])
    return np.asarray((274.0 - 2.88 * x, 178.5 + 4.45 * y), dtype=np.float32)


def draw_dashed_segment(image, start, end, color, width=1, dash=7, gap=5):
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        return
    direction = vector / length
    position = 0.0
    while position < length:
        segment_end = min(position + dash, length)
        p0 = tuple(np.rint(start + direction * position).astype(int))
        p1 = tuple(np.rint(start + direction * segment_end).astype(int))
        cv2.line(image, p0, p1, color, width, cv2.LINE_AA)
        position += dash + gap


def fov_panel(current_info, current_semantics):
    base, _ = semantic_panel(
        current_semantics, color_dynamic=False, outline_dynamic=False)
    overlay = base.copy()
    mask = np.zeros_like(base)
    legend_height = 29
    fov_specs = (
        ('CAM_FRONT', 'F', 'Front', (226, 75, 66)),
        ('CAM_LEFT', 'L', 'Left', (55, 116, 180)),
        ('CAM_BACK', 'B', 'Back', (65, 155, 100)),
        ('CAM_RIGHT', 'R', 'Right', (148, 86, 166)),
    )
    line_segments = []
    centres = []
    for key, short, name, color in fov_specs:
        camera = current_info['cams'][key]
        origin = np.asarray(
            camera['sensor2lidar_translation'][:2], dtype=np.float32)
        rotation = np.asarray(
            camera['sensor2lidar_rotation'], dtype=np.float32)
        direction = rotation[:2, 2]
        direction /= max(np.linalg.norm(direction), 1e-6)
        intrinsic = np.asarray(camera['cam_intrinsic'], dtype=np.float32)
        image_width = Image.open(resolve(camera['data_path'])).size[0]
        half_fov = np.arctan(image_width / (2.0 * intrinsic[0, 0]))
        yaw = np.arctan2(direction[1], direction[0])
        rays = []
        for angle in (yaw - half_fov, yaw + half_fov):
            endpoint = origin + 95.0 * np.asarray(
                (np.cos(angle), np.sin(angle)), dtype=np.float32)
            rays.append(metric_to_panel(endpoint))
        centre = metric_to_panel(origin)
        polygon = np.rint(np.stack(
            (centre, rays[0], rays[1]))).astype(np.int32)
        cv2.fillPoly(mask, [polygon], color)
        line_segments.append((centre, rays, color))
        centres.append((centre, short, color))
    valid_height = PANEL_SIZE - legend_height
    alpha = 0.105
    overlay[:valid_height] = np.clip(
        (1.0 - alpha) * overlay[:valid_height].astype(np.float32) +
        alpha * mask[:valid_height].astype(np.float32), 0,
        255).astype(np.uint8)
    for centre, rays, color in line_segments:
        for ray in rays:
            draw_dashed_segment(overlay, centre, ray, color, width=1)
    panel = Image.fromarray(overlay)
    draw = ImageDraw.Draw(panel, 'RGBA')
    font = load_font(FONT_BOLD, 20)
    label_offsets = {
        'F': (-35, -42),
        'R': (10, 4),
        'L': (10, 4),
        'B': (-38, -42)
    }
    for centre, short, color in centres:
        x, y = (int(round(value)) for value in centre)
        draw.regular_polygon((x, y, 5),
                             n_sides=4,
                             rotation=45,
                             fill=tuple(color) + (255, ),
                             outline=(255, 255, 255, 255))
        box = draw.textbbox((0, 0), short, font=font)
        width = box[2] - box[0]
        height = box[3] - box[1]
        offset_x, offset_y = label_offsets[short]
        tx = min(max(x + offset_x, 2), PANEL_SIZE - width - 12)
        ty = min(max(y + offset_y, 2), valid_height - height - 9)
        draw.rounded_rectangle((tx, ty, tx + width + 9, ty + height + 7),
                               radius=3,
                               fill=tuple(color) + (225, ))
        draw.text((tx + 4, ty + 1),
                  short,
                  font=font,
                  fill=(255, 255, 255, 255))
    draw.rectangle((0, valid_height, PANEL_SIZE, PANEL_SIZE),
                   fill=(255, 255, 255, 238))
    legend_font = load_font(FONT_REGULAR, 14)
    x = 9
    for _, short, name, color in fov_specs:
        draw.rounded_rectangle(
            (x, valid_height + 8, x + 13, valid_height + 21),
            radius=3,
            fill=tuple(color) + (255, ))
        legend = '{} {}'.format(short, name)
        draw.text((x + 18, valid_height + 5),
                  legend,
                  font=legend_font,
                  fill=TEXT + (255, ))
        x += 18 + draw.textlength(legend, font=legend_font) + 13
    return np.asarray(panel)


def draw_frame(image, box, color, width=3, dashed=False):
    x0, y0, x1, y1 = box
    if not dashed:
        cv2.rectangle(image, (x0, y0), (x1, y1), color, width, cv2.LINE_8)
        return
    dash, gap = 7, 5
    for start in range(x0, x1 + 1, dash + gap):
        cv2.line(image, (start, y0), (min(start + dash, x1), y0), color, width,
                 cv2.LINE_8)
        cv2.line(image, (start, y1), (min(start + dash, x1), y1), color, width,
                 cv2.LINE_8)
    for start in range(y0, y1 + 1, dash + gap):
        cv2.line(image, (x0, start), (x0, min(start + dash, y1)), color, width,
                 cv2.LINE_8)
        cv2.line(image, (x1, start), (x1, min(start + dash, y1)), color, width,
                 cv2.LINE_8)


def add_badge(image, text, anchor, color, font_size=19, target_box=None):
    """Draw a compact evidence label, optionally linked to a target frame.

    The leader is deliberately thin and has no arrowhead.  It lets us keep
    labels on low-information static regions instead of covering the road and
    dynamic occupancy evidence.
    """
    panel = Image.fromarray(image)
    draw = ImageDraw.Draw(panel, 'RGBA')
    font = load_font(FONT_BOLD, font_size)
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = min(max(int(anchor[0]), 3), PANEL_SIZE - width - 18)
    y = min(max(int(anchor[1]), 3), PANEL_SIZE - height - 13)
    badge_box = (x, y, x + width + 14, y + height + 10)
    if target_box is not None:
        tx0, ty0, tx1, ty1 = target_box
        badge_centre = np.asarray(
            ((badge_box[0] + badge_box[2]) / 2.0,
             (badge_box[1] + badge_box[3]) / 2.0))
        target_centre = np.asarray(((tx0 + tx1) / 2.0,
                                    (ty0 + ty1) / 2.0))
        target_point = np.clip(badge_centre, (tx0, ty0), (tx1, ty1))
        badge_point = np.clip(target_centre,
                              (badge_box[0], badge_box[1]),
                              (badge_box[2], badge_box[3]))
        draw.line((tuple(target_point), tuple(badge_point)),
                  fill=tuple(color) + (245, ),
                  width=2)
    draw.rounded_rectangle(badge_box,
                           radius=4,
                           fill=tuple(color) + (246, ))
    draw.text((x + 7, y + 3), text, font=font, fill=(255, 255, 255, 255))
    image[:] = np.asarray(panel)


def add_letter_callout(image, letter, box, color, offset):
    draw_frame(image, box, color, width=4)
    panel = Image.fromarray(image)
    draw = ImageDraw.Draw(panel, 'RGBA')
    font = load_font(FONT_BOLD, 28)
    x = min(max(box[0] + offset[0], 4), PANEL_SIZE - 41)
    y = min(max(box[1] + offset[1], 4), PANEL_SIZE - 45)
    draw.rounded_rectangle((x, y, x + 39, y + 39),
                           radius=5,
                           fill=tuple(color) + (250, ))
    bounds = draw.textbbox((0, 0), letter, font=font)
    tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.text((x + (39 - tw) / 2, y + (39 - th) / 2 - 2),
              letter,
              font=font,
              fill=(255, 255, 255, 255))
    image[:] = np.asarray(panel)


def add_ptr_legend(image):
    panel = Image.fromarray(image)
    draw = ImageDraw.Draw(panel, 'RGBA')
    height = 31
    draw.rectangle((0, PANEL_SIZE - height, PANEL_SIZE, PANEL_SIZE),
                   fill=(255, 255, 255, 238))
    font = load_font(FONT_REGULAR, 14)
    x = 12
    for label, color in (('Persist', P_COLOR), ('Transport', T_COLOR),
                         ('Refresh', R_COLOR)):
        draw.rounded_rectangle((x, PANEL_SIZE - 22, x + 15, PANEL_SIZE - 7),
                               radius=3,
                               fill=tuple(color) + (255, ))
        draw.text((x + 21, PANEL_SIZE - 25),
                  label,
                  font=font,
                  fill=TEXT + (255, ))
        x += 21 + draw.textlength(label, font=font) + 16
    image[:] = np.asarray(panel)


def add_panel_border(image):
    cv2.rectangle(image, (0, 0), (PANEL_SIZE - 1, PANEL_SIZE - 1), BORDER, 2,
                  cv2.LINE_8)


def compose_evidence_panels(current_semantics, history_semantics, flow):
    current, current_labels = semantic_panel(current_semantics)
    history, history_labels = semantic_panel(history_semantics)
    ptr = current.copy()

    current_cars = connected_components(current_labels, class_index=4)
    history_cars = connected_components(history_labels, class_index=4)
    current_buses = connected_components(current_labels, class_index=3)
    history_buses = connected_components(history_labels, class_index=3)
    # Stable semantic anchors for this scene.  Selection is component-based,
    # so minor renderer changes do not move the annotations onto another car.
    transport = nearest_component(current_buses, (180.3, 219.6))
    persist = nearest_component(current_cars, (57.2, 127.7))
    refresh = nearest_component(current_cars, (233.5, 152.6))
    source_point = component_flow_source(transport, flow, current_semantics,
                                         HISTORY_STEPS * FRAME_DT)
    source = nearest_component(history_buses, source_point)
    refresh_source = component_flow_source(refresh, flow, current_semantics,
                                           HISTORY_STEPS * FRAME_DT)

    flow_bev, _ = project_dynamic_flow_to_bev(
        flow, current_semantics, DYNAMIC_INDICES, fill_iterations=0)
    flow_image = _orient_xy_grid_to_image(flow_bev)
    persist_speed = np.linalg.norm(flow_image[persist['mask']], axis=-1)
    transport_target_is_empty = not np.any(
        history_labels[transport['mask']] == 3)
    source_x, source_y = (int(round(value)) for value in source_point)
    refresh_x, refresh_y = (int(round(value)) for value in refresh_source)
    transport_source_is_bus = history_labels[source_y, source_x] == 3
    refresh_source_is_car = history_labels[refresh_y, refresh_x] == 4
    persist_rigid_ratio = np.mean(history_labels[persist['mask']] == 4)
    if not transport_target_is_empty:
        raise AssertionError(
            'Selected Transport target is occupied in history.')
    if not transport_source_is_bus:
        raise AssertionError('Selected Transport source lacks bus support.')
    if refresh_source_is_car:
        raise AssertionError('Selected Refresh source has historical support.')
    if np.max(persist_speed) > 1e-3 or persist_rigid_ratio < 0.9:
        raise AssertionError(
            'Selected Persist component is not rigidly supported.')

    transport_box = scaled_box(transport['box'], padding=5)
    persist_box = scaled_box(persist['box'], padding=5)
    refresh_box = scaled_box(refresh['box'], padding=5)
    source_box = scaled_box(source['box'], padding=5)

    draw_frame(current, transport_box, T_COLOR, width=4)
    add_badge(current, 'Target: bus',
              (transport_box[2] + 12, transport_box[1] + 20), T_COLOR)
    draw_frame(current, refresh_box, R_COLOR, width=4)
    add_badge(current, 'Newly visible',
              (refresh_box[0] - 80, refresh_box[1] - 45), R_COLOR)

    draw_frame(history, source_box, T_COLOR, width=4)
    add_badge(history, 'Source: bus', (source_box[2] + 12, source_box[1] + 5),
              T_COLOR)
    draw_frame(history, transport_box, HISTORY_COLOR, width=2, dashed=True)
    add_badge(history, 'Empty',
              (transport_box[0] - 45, transport_box[3] + 13), HISTORY_COLOR)

    source_scale = PANEL_SIZE / float(GRID_SIZE)
    no_history_box = (
        int(refresh_source[0] * source_scale) - 11,
        int(refresh_source[1] * source_scale) - 11,
        int(refresh_source[0] * source_scale) + 11,
        int(refresh_source[1] * source_scale) + 11,
    )
    no_history_box = tuple(
        max(2, min(value, PANEL_SIZE - 3)) for value in no_history_box)
    draw_frame(history, no_history_box, HISTORY_COLOR, width=2, dashed=True)
    add_badge(history, 'No history',
              (no_history_box[0] - 78, no_history_box[3] + 12), HISTORY_COLOR)

    add_letter_callout(ptr, 'P', persist_box, P_COLOR, (7, -46))
    add_letter_callout(ptr, 'T', transport_box, T_COLOR, (-45, -4))
    add_letter_callout(ptr, 'R', refresh_box, R_COLOR, (7, -46))
    add_ptr_legend(ptr)

    diagnostics = dict(
        transport_current=transport['box'],
        transport_source=source['box'],
        transport_source_prediction=tuple(round(v, 2) for v in source_point),
        persist=persist['box'],
        refresh=refresh['box'],
        refresh_source_prediction=tuple(round(v, 2) for v in refresh_source),
        transport_target_empty=transport_target_is_empty,
        transport_source_bus=bool(transport_source_is_bus),
        persist_max_speed=round(float(np.max(persist_speed)), 4),
        persist_rigid_ratio=round(float(persist_rigid_ratio), 4),
        refresh_source_car=bool(refresh_source_is_car))
    return current, history, ptr, diagnostics


def compose_approved_evidence_panels(current_semantics, history_semantics,
                                     flow):
    """Render c--f from one BEV grid without any post-hoc registration."""
    current, current_labels = semantic_panel(current_semantics)
    history, history_labels = semantic_panel(history_semantics)
    ptr = current.copy()

    current_cars = connected_components(current_labels, class_index=4)
    history_cars = connected_components(history_labels, class_index=4)
    current_buses = connected_components(current_labels, class_index=3)
    # The horizontal traffic stream moves from image right to left.  Use its
    # leftmost car as the clean Transport example and the two entering cars at
    # the right boundary as the long-horizon Refresh examples.
    transport = nearest_component(current_cars, (146.6, 147.7))
    persist = nearest_component(current_buses, (170.5, 80.9))
    refreshes = [
        nearest_component(current_cars, centre)
        for centre in ((253.2, 153.5), (278.2, 155.3))
    ]
    delta_t = APPROVED_HISTORY_STEPS * FRAME_DT
    source_point = component_flow_source(transport, flow, current_semantics,
                                         delta_t)
    source = nearest_component(history_cars, source_point)
    refresh_sources = [
        component_flow_source(component, flow, current_semantics, delta_t)
        for component in refreshes
    ]

    flow_bev, _ = project_dynamic_flow_to_bev(
        flow, current_semantics, DYNAMIC_INDICES, fill_iterations=0)
    flow_image = _orient_xy_grid_to_image(flow_bev)
    persist_speed = np.linalg.norm(flow_image[persist['mask']], axis=-1)
    transport_rigid_ratio = np.mean(
        history_labels[transport['mask']] == 4)
    persist_rigid_ratio = np.mean(history_labels[persist['mask']] == 3)
    # Over a 2 s visual horizon, instantaneous velocity is only an approximate
    # source locator.  Validate the actual historical component with a small
    # spatial tolerance instead of requiring its centre pixel to match.
    transport_source_distance = point_box_distance(source_point, source['box'])
    refresh_source_distances = [
        min(point_box_distance(point, component['box'])
            for component in history_cars)
        for point in refresh_sources
    ]
    transport_source_is_car = transport_source_distance <= 3.0
    refresh_source_is_car = [distance <= 3.0
                             for distance in refresh_source_distances]
    if not transport_source_is_car or transport_rigid_ratio >= 0.1:
        raise AssertionError('Approved Transport example lacks clean support.')
    if any(refresh_source_is_car):
        raise AssertionError('Approved Refresh example has historical support.')
    if np.max(persist_speed) > 1e-3 or persist_rigid_ratio < 0.9:
        raise AssertionError('Approved Persist example is not stationary.')

    transport_box = scaled_box(transport['box'], padding=5)
    persist_box = scaled_box(persist['box'], padding=5)
    refresh_boxes = [scaled_box(component['box'], padding=5)
                     for component in refreshes]
    refresh_group_box = union_box(refresh_boxes)
    source_box = scaled_box(source['box'], padding=5)
    draw_frame(current, transport_box, T_COLOR, width=4)
    add_badge(current, 'Target: car',
              (20, 108), T_COLOR, target_box=transport_box)
    for refresh_box in refresh_boxes:
        draw_frame(current, refresh_box, R_COLOR, width=4)
    add_badge(current, 'Newly visible',
              (245, 240), R_COLOR, target_box=refresh_group_box)

    draw_frame(history, source_box, T_COLOR, width=4)
    add_badge(history, 'Source: car',
              (275, 108), T_COLOR, target_box=source_box)
    draw_frame(history, transport_box, HISTORY_COLOR, width=2, dashed=True)
    add_badge(history, 'Empty',
              (70, 280), HISTORY_COLOR, target_box=transport_box)
    source_scale = PANEL_SIZE / float(GRID_SIZE)
    no_history_boxes = []
    for refresh_source in refresh_sources:
        no_history_box = (
            int(refresh_source[0] * source_scale) - 11,
            int(refresh_source[1] * source_scale) - 11,
            int(refresh_source[0] * source_scale) + 11,
            int(refresh_source[1] * source_scale) + 11,
        )
        no_history_box = tuple(
            max(2, min(value, PANEL_SIZE - 3)) for value in no_history_box)
        no_history_boxes.append(no_history_box)
        draw_frame(history, no_history_box, HISTORY_COLOR, width=2,
                   dashed=True)
    add_badge(history, 'No history',
              (295, 240), HISTORY_COLOR,
              target_box=union_box(no_history_boxes))

    add_letter_callout(ptr, 'P', persist_box, P_COLOR, (7, -46))
    add_letter_callout(ptr, 'T', transport_box, T_COLOR, (-45, -4))
    draw_frame(ptr, refresh_boxes[0], R_COLOR, width=4)
    add_letter_callout(ptr, 'R', refresh_boxes[1], R_COLOR, (7, -46))
    add_ptr_legend(ptr)
    diagnostics = dict(
        current_token=APPROVED_TOKEN,
        history_steps=APPROVED_HISTORY_STEPS,
        transport_current=transport['box'],
        transport_source=source['box'],
        transport_source_prediction=tuple(round(v, 2)
                                          for v in source_point),
        transport_source_distance=round(transport_source_distance, 2),
        transport_rigid_ratio=round(float(transport_rigid_ratio), 4),
        persist=persist['box'],
        persist_max_speed=round(float(np.max(persist_speed)), 4),
        persist_rigid_ratio=round(float(persist_rigid_ratio), 4),
        refresh=[component['box'] for component in refreshes],
        refresh_source_prediction=[tuple(round(v, 2) for v in point)
                                   for point in refresh_sources],
        refresh_source_distance=[round(value, 2)
                                 for value in refresh_source_distances],
        refresh_source_car=refresh_source_is_car)
    return current, history, ptr, diagnostics


def add_title(canvas, title, column, row):
    draw = ImageDraw.Draw(canvas)
    font = load_font(FONT_BOLD, 31)
    x = column * (PANEL_SIZE + GAP) + 8
    y = row * (PANEL_SIZE + TITLE_HEIGHT + GAP) + 4
    draw.text((x, y), title, font=font, fill=TEXT)


def save_lossless_pdf(image, path, dpi=300.0):
    """Write one Flate-compressed RGB image; never introduce JPEG loss."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    page_width = width * 72.0 / float(dpi)
    page_height = height * 72.0 / float(dpi)
    stream = zlib.compress(np.ascontiguousarray(image).tobytes(), level=9)
    content = ('q {:.6f} 0 0 {:.6f} 0 0 cm /Im0 Do Q\n'.format(
        page_width, page_height)).encode('ascii')
    objects = (
        b'<< /Type /Catalog /Pages 2 0 R >>',
        b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        ('<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {:.6f} {:.6f}] '
         '/Resources << /XObject << /Im0 4 0 R >> >> '
         '/Contents 5 0 R >>'.format(page_width, page_height)).encode('ascii'),
        (('<< /Type /XObject /Subtype /Image /Width {} /Height {} '
          '/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode '
          '/Length {} >>\nstream\n').format(
              width, height, len(stream)).encode('ascii') + stream +
         b'\nendstream'),
        (b'<< /Length ' + str(len(content)).encode('ascii') +
         b' >>\nstream\n' + content + b'endstream'),
    )
    payload = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend('{} 0 obj\n'.format(index).encode('ascii'))
        payload.extend(obj)
        payload.extend(b'\nendobj\n')
    xref = len(payload)
    payload.extend('xref\n0 {}\n'.format(len(objects) + 1).encode('ascii'))
    payload.extend(b'0000000000 65535 f \n')
    for offset in offsets[1:]:
        payload.extend('{:010d} 00000 n \n'.format(offset).encode('ascii'))
    payload.extend(
        ('trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n'.format(
            len(objects) + 1, xref)).encode('ascii'))
    path.write_bytes(payload)


def recompose_approved_source(args):
    """Preserve a/b and rebuild c--f directly on one approved BEV grid."""
    source = Image.open(args.approved_source).convert('RGB')
    if source.size != CANVAS_SIZE:
        raise ValueError('Approved source must be {}x{}, got {}x{}.'.format(
            CANVAS_SIZE[0], CANVAS_SIZE[1], source.size[0], source.size[1]))
    with ANN_FILE.open('rb') as handle:
        payload = pickle.load(handle)
    infos = payload['infos'] if isinstance(payload, dict) else payload
    lookup = {str(info['token']): info for info in infos}
    current_info = lookup[APPROVED_TOKEN]
    history_info = current_info
    for _ in range(APPROVED_HISTORY_STEPS):
        history_info = lookup[str(history_info['prev'])]
    current_semantics, _ = load_labels(
        resolve(current_info['occ_path']) / 'labels.npz')
    raw_history, raw_visibility = load_labels(
        resolve(history_info['occ_path']) / 'labels.npz')
    history_semantics, _ = align_history(
        current_info, history_info, raw_history, raw_visibility,
        current_semantics.shape)
    flow_path = (FLOW_ROOT / Path(current_info['occ_path']).parent.name /
                 APPROVED_TOKEN / 'labels.npz')
    with np.load(flow_path, allow_pickle=False) as payload:
        flow = payload['flow'].astype(np.float32)
    current, history, ptr, diagnostics = compose_approved_evidence_panels(
        current_semantics, history_semantics, flow)
    flow_panel = render_flow_bev(
        flow,
        current_semantics,
        DYNAMIC_INDICES,
        image_size=(PANEL_SIZE, PANEL_SIZE),
        max_speed=8.0)
    for panel in (flow_panel, current, history, ptr):
        add_panel_border(panel)

    image = np.asarray(source).copy()
    image[:TITLE_HEIGHT] = 255
    second_title_y = TITLE_HEIGHT + PANEL_SIZE
    image[second_title_y:second_title_y + TITLE_HEIGHT + GAP] = 255
    flow_x = 2 * (PANEL_SIZE + GAP)
    image[TITLE_HEIGHT:TITLE_HEIGHT + PANEL_SIZE,
          flow_x:flow_x + PANEL_SIZE] = flow_panel
    image[second_title_y + TITLE_HEIGHT + GAP:
          second_title_y + TITLE_HEIGHT + GAP + PANEL_SIZE,
          0:PANEL_SIZE] = current
    image[second_title_y + TITLE_HEIGHT + GAP:
          second_title_y + TITLE_HEIGHT + GAP + PANEL_SIZE,
          PANEL_SIZE + GAP:2 * PANEL_SIZE + GAP] = history
    image[second_title_y + TITLE_HEIGHT + GAP:
          second_title_y + TITLE_HEIGHT + GAP + PANEL_SIZE,
          flow_x:flow_x + PANEL_SIZE] = ptr
    canvas = Image.fromarray(image)
    titles = ('(a) Current cameras', '(b) Camera FOV', '(c) OCC flow',
              '(d) Current OCC', '(e) Past OCC', '(f) P/T/R target')
    for index, title in enumerate(titles):
        add_title(canvas, title, index % 3, index // 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, format='PNG', optimize=True)
    image = np.asarray(canvas)
    pdf_outputs = list(OUTPUT_PDFS)
    generated_peer = args.output.with_suffix('.pdf')
    if generated_peer not in pdf_outputs:
        pdf_outputs.append(generated_peer)
    for path in pdf_outputs:
        save_lossless_pdf(image, path, dpi=args.dpi)
    print('approved_source={}'.format(args.approved_source.resolve()))
    print('approved_current_token={}'.format(APPROVED_TOKEN))
    print('approved_history_token={}'.format(history_info['token']))
    print('approved_flow={}'.format(flow_path.resolve()))
    for key, value in diagnostics.items():
        print('{}={}'.format(key, value))
    print(args.output.resolve())
    for path in pdf_outputs:
        print(path.resolve())


def main():
    args = parse_args()
    if args.approved_source is not None:
        recompose_approved_source(args)
        return
    with args.ann_file.open('rb') as handle:
        payload = pickle.load(handle)
    infos = payload['infos'] if isinstance(payload, dict) else payload
    lookup = {str(info['token']): info for info in infos}
    current_info = lookup[args.token]
    history_info = current_info
    for _ in range(args.history_steps):
        previous = str(history_info.get('prev', ''))
        if not previous or previous not in lookup:
            raise ValueError('Insufficient history for {}'.format(args.token))
        history_info = lookup[previous]

    current_semantics, _ = load_labels(
        resolve(current_info['occ_path']) / 'labels.npz')
    raw_history, raw_visibility = load_labels(
        resolve(history_info['occ_path']) / 'labels.npz')
    history_semantics, _ = align_history(current_info, history_info,
                                         raw_history, raw_visibility,
                                         current_semantics.shape)
    flow_path = (
        args.flow_root / Path(current_info['occ_path']).parent.name /
        args.token / 'labels.npz')
    with np.load(flow_path, allow_pickle=False) as payload:
        flow = payload['flow'].astype(np.float32)

    current, history, ptr, diagnostics = compose_evidence_panels(
        current_semantics, history_semantics, flow)
    panels = (
        camera_panel(current_info),
        fov_panel(current_info, current_semantics),
        render_flow_bev(
            flow,
            current_semantics,
            DYNAMIC_INDICES,
            image_size=(PANEL_SIZE, PANEL_SIZE),
            max_speed=8.0),
        current,
        history,
        ptr,
    )
    titles = ('(a) Current cameras', '(b) Camera FOV', '(c) OCC flow',
              '(d) Current OCC', '(e) Past OCC', '(f) P/T/R target')
    canvas = Image.new('RGB', CANVAS_SIZE, 'white')
    for index, (panel, title) in enumerate(zip(panels, titles)):
        panel = np.asarray(panel, dtype=np.uint8).copy()
        add_panel_border(panel)
        column, row = index % 3, index // 3
        x = column * (PANEL_SIZE + GAP)
        y = row * (PANEL_SIZE + TITLE_HEIGHT + GAP) + TITLE_HEIGHT
        canvas.paste(Image.fromarray(panel), (x, y))
        add_title(canvas, title, column, row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, format='PNG', optimize=True)
    image = np.asarray(canvas)
    pdf_outputs = list(OUTPUT_PDFS)
    generated_peer = args.output.with_suffix('.pdf')
    if generated_peer not in pdf_outputs:
        pdf_outputs.append(generated_peer)
    for path in pdf_outputs:
        save_lossless_pdf(image, path, dpi=args.dpi)

    print('current_token={}'.format(args.token))
    print('history_token={}'.format(history_info['token']))
    print('history_steps={} visual_dt={:.1f}s'.format(
        args.history_steps, args.history_steps * FRAME_DT))
    for key, value in diagnostics.items():
        print('{}={}'.format(key, value))
    print(args.output.resolve())
    for path in pdf_outputs:
        print(path.resolve())


if __name__ == '__main__':
    main()
