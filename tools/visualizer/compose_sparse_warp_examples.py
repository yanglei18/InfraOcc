#!/usr/bin/env python3
"""Compose clean sparse-backwarp paper examples from cached diagnostics."""

import argparse
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = (ROOT / 'work_dirs/stcroadocc_c_2x4_24e/figures_path/'
              'train/sparse_warp_bev')
PRED_OCC_DIR = (ROOT / 'work_dirs/stcroadocc_c_2x4_24e/figures_path/'
                'train/occ_bev')
DEFAULT_SAMPLES = (
    '020_000136_sparse_warp_bev.png',
)
HIT_SCORES = (
    (0.50, 0.96),
)
PANEL_LABELS = (
    'Current GT Occ.',
    'Pred Occ.',
    'Pred Flow',
    'Past GT Occ.',
    'Rigid hit',
    'VVE-backwarp hit',
)
PANEL_ORDER = (
    (0, 'top'),  # Current GT occupancy
    (None, 'pred_occ'),
    (1, 'top'),   # Predicted flow
    (0, 'middle'),  # Rigid history / past GT
    (0, 'bottom'),  # Zero-motion hit map
    (1, 'bottom'),  # Predicted-backwarp hit map
)
PANEL_SEPARATOR_COLOR = (205, 205, 205)
MATCH_COLOR = (64, 176, 64)
# BGR canvas value so the saved PNG is visibly red after cv2.imwrite.
MISMATCH_COLOR = (55, 55, 220)
REMAINING_COLOR = (160, 160, 160)
_MATCH_SOURCE_COLOR = np.array((64, 176, 64), dtype=np.uint8)
_MISMATCH_SOURCE_COLORS = (
    np.array((255, 208, 64), dtype=np.uint8),
    np.array((220, 72, 72), dtype=np.uint8),
    np.array((160, 160, 160), dtype=np.uint8),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=SOURCE_DIR)
    parser.add_argument('--pred-occ-dir', type=Path, default=PRED_OCC_DIR)
    parser.add_argument('--samples', nargs=2, default=DEFAULT_SAMPLES)
    parser.add_argument('--cell-size', type=int, default=360)
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'docs/paper/ICLR2027/Fig5-backwarp.png')
    return parser.parse_args()


def extract_sparse_panels(image):
    """Extract aligned map bodies, excluding titles and run metadata.

    The cached diagnostic contains a metadata header followed by three rows of
    square panels.  Cropping all three rows with the same title offset is
    essential: the hit-map row must not receive a shorter crop and then be
    stretched independently.
    """
    height, width = image.shape[:2]
    panel_size = width // 3
    metadata_height = height - 3 * panel_size
    title_height = 29
    if panel_size <= 0 or metadata_height < 0:
        raise ValueError(f'Unexpected sparse-warp canvas: {width}x{height}')
    panels = {}
    for column in range(3):
        x_start = column * panel_size
        x_end = x_start + panel_size
        for row_index, row_name in enumerate(('top', 'middle', 'bottom')):
            y_start = metadata_height + row_index * panel_size + title_height
            y_end = metadata_height + (row_index + 1) * panel_size
            panels[column, row_name] = image[y_start:y_end, x_start:x_end]
    return panels


def extract_pred_occ(path):
    """Extract the prediction half of an occupancy-BEV pair without labels."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    panel_size = width // 2
    metadata_height = height - panel_size
    # Match the normalized title crop used by the 480-pixel sparse-warp panels.
    title_height = int(round(panel_size * 29.0 / 480.0))
    return image[metadata_height + title_height:metadata_height + panel_size,
                 :panel_size]


def fit_panel(panel, size):
    """Resize an aligned panel without changing its aspect ratio."""
    height, width = panel.shape[:2]
    target_height = int(round(size * height / float(width)))
    interpolation = cv2.INTER_AREA if width >= size else cv2.INTER_CUBIC
    panel = cv2.resize(
        panel, (size, target_height), interpolation=interpolation)
    cv2.rectangle(panel, (0, 0), (size - 1, target_height - 1),
                  PANEL_SEPARATOR_COLOR, 1)
    return panel


def add_zoom_inset(panel):
    """Add the same motion-focused crop to every panel for direct alignment."""
    panel = panel.copy()
    height, width = panel.shape[:2]
    # A shared normalized crop encloses the moving vehicles on the north arm.
    x0 = int(round(0.40 * width))
    x1 = int(round(0.58 * width))
    y0 = int(round(0.06 * height))
    y1 = int(round(0.25 * height))
    crop = panel[y0:y1, x0:x1].copy()
    cv2.rectangle(panel, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), 2)

    max_width = int(round(0.43 * width))
    max_height = int(round(0.45 * height))
    scale = min(max_width / float(crop.shape[1]),
                max_height / float(crop.shape[0]))
    inset_width = int(round(crop.shape[1] * scale))
    inset_height = int(round(crop.shape[0] * scale))
    inset = cv2.resize(
        crop, (inset_width, inset_height), interpolation=cv2.INTER_NEAREST)
    inset_x = 7
    inset_y = height - inset_height - 7
    panel[inset_y:inset_y + inset_height,
          inset_x:inset_x + inset_width] = inset
    cv2.rectangle(panel, (inset_x, inset_y),
                  (inset_x + inset_width - 1, inset_y + inset_height - 1),
                  (0, 0, 0), 2)
    return panel


def simplify_hit_panel(panel, mismatch_color):
    """Map original hit statuses to the paper's three-colour comparison."""
    panel = np.asarray(panel)
    # Past GT already supplies the spatial context in the preceding column.
    # Keep hit maps binary-colour so gray cannot be confused with the original
    # diagnostic's ``zero-motion better`` status.
    output = np.full_like(panel, 255)
    match_mask = np.all(panel == _MATCH_SOURCE_COLOR, axis=-1)
    mismatch_mask = np.zeros(panel.shape[:2], dtype=bool)
    for color in _MISMATCH_SOURCE_COLORS:
        mismatch_mask |= np.all(panel == color, axis=-1)
    output[match_mask] = MATCH_COLOR
    output[mismatch_mask] = mismatch_color
    return output


def recolor_hit_panel(panel):
    """Recolour, without deleting, every original hit-map pixel.

    Green marks a semantic match at the Past-GT target; red marks an unmatched
    target; every other non-white original hit-map pixel is retained in gray.
    """
    panel = np.asarray(panel)
    output = panel.copy()
    nonwhite_mask = np.any(panel != 255, axis=-1)
    output[nonwhite_mask] = REMAINING_COLOR

    match_mask = np.all(panel == _MATCH_SOURCE_COLOR, axis=-1)
    mismatch_mask = np.zeros(panel.shape[:2], dtype=bool)
    for color in _MISMATCH_SOURCE_COLORS[:2]:
        mismatch_mask |= np.all(panel == color, axis=-1)
    output[match_mask] = MATCH_COLOR
    output[mismatch_mask] = MISMATCH_COLOR
    return output


def annotate_hit_score(panel, score):
    panel = panel.copy()
    label = f'Score {score:.2f}'
    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.90, 2)[0][0]
    cv2.putText(panel, label, (panel.shape[1] - text_width - 10,
                               panel.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 0), 2, cv2.LINE_AA)
    return panel


def column_headers(size, gap):
    cells = []
    for label in PANEL_LABELS:
        header = np.full((53, size, 3), 255, dtype=np.uint8)
        text_width, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                        1.02, 2)[0]
        cv2.putText(header, label, ((size - text_width) // 2, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.02, (0, 0, 0), 2,
                    cv2.LINE_AA)
        cells.append(header)
    return cv2.hconcat([
        cell if index == 0 else np.hstack(
            [np.full((cells[0].shape[0], gap, 3), PANEL_SEPARATOR_COLOR,
                     dtype=np.uint8), cell])
        for index, cell in enumerate(cells)
    ])


def hit_map_caption(width):
    """Compact colour caption for the recoloured hit maps."""
    caption = np.full((44, width, 3), (246, 248, 250), dtype=np.uint8)
    text_color = (35, 43, 54)
    entries = (
        ('Same-class match', MATCH_COLOR),
        ('Unmatched target', MISMATCH_COLOR),
        ('Other queried voxel', REMAINING_COLOR),
    )
    text_widths = [
        cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 1)[0][0]
        for label, _ in entries
    ]
    total_width = sum(text_widths) + 32 * len(entries) + 50 * (len(entries) - 1)
    x = (width - total_width) // 2
    for (label, color), text_width in zip(entries, text_widths):
        cv2.rectangle(caption, (x, 9), (x + 22, 31), color, -1)
        cv2.rectangle(caption, (x, 9), (x + 22, 31), (90, 90, 90), 1)
        cv2.putText(caption, label, (x + 32, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, text_color, 1, cv2.LINE_AA)
        x += text_width + 82
    return caption


def main():
    args = parse_args()
    if len(args.samples) != 1:
        raise ValueError('The paper figure uses exactly one representative sample.')
    gap = 0
    size = args.cell_size
    row_width = len(PANEL_ORDER) * size + (len(PANEL_ORDER) - 1) * gap
    rows = []
    for row_index, sample_name in enumerate(args.samples):
        image = cv2.imread(str(args.source_dir / sample_name), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(args.source_dir / sample_name)
        panels = extract_sparse_panels(image)
        occ_name = sample_name.replace('_sparse_warp_bev.png', '_occ_bev.png')
        panels[None, 'pred_occ'] = extract_pred_occ(args.pred_occ_dir / occ_name)
        for column in range(3):
            panels[column, 'bottom'] = recolor_hit_panel(
                panels[column, 'bottom'])
        cells = [add_zoom_inset(fit_panel(panels[key], size))
                 for key in PANEL_ORDER]
        for cell_index, score in zip((4, 5), HIT_SCORES[row_index]):
            cells[cell_index] = annotate_hit_score(cells[cell_index], score)
        rows.append(cv2.hconcat([
            cell if index == 0 else np.hstack(
                [np.full((cell.shape[0], gap, 3), PANEL_SEPARATOR_COLOR,
                         dtype=np.uint8), cell])
            for index, cell in enumerate(cells)
        ]))
    output = cv2.vconcat([
        column_headers(size, gap),
        rows[0],
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output):
        raise RuntimeError(f'Failed to write {args.output}')
    print(args.output)


if __name__ == '__main__':
    main()
