#!/usr/bin/env python3
"""Compose two recursive occupancy samples into a 2-by-5 BEV grid."""

import argparse
from pathlib import Path

import cv2
import numpy as np


PANEL_SIZE = 320
BANNER_HEIGHT = 88
TITLE_HEIGHT = 29
CONTENT_HEIGHT = PANEL_SIZE - TITLE_HEIGHT
GAP = 0
ROW_LABELS = ("1/8", "1/4", "1/2", "1/1", "GT")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--samples", nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read {path}")
    return image


def add_caption(panel, title):
    panel = panel[TITLE_HEIGHT:].copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (_, text_height), _ = cv2.getTextSize(
        title, font, font_scale, thickness)
    x0, y0 = 6, 6
    text_origin = (x0, y0 + text_height + 5)
    cv2.putText(
        panel,
        title,
        text_origin,
        font,
        font_scale,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        title,
        text_origin,
        font,
        font_scale,
        (35, 35, 35),
        thickness,
        cv2.LINE_AA,
    )
    return panel


def extract_panels(source_root, sample):
    recursive_path = (
        source_root / "recursive_occ_bev" /
        f"{sample}_recursive_occ_bev.png")
    comparison_path = source_root / "occ_bev" / f"{sample}_occ_bev.png"
    recursive = read_image(recursive_path)
    comparison = read_image(comparison_path)

    expected_recursive_shape = (BANNER_HEIGHT + 2 * PANEL_SIZE,
                                2 * PANEL_SIZE)
    if recursive.shape[:2] != expected_recursive_shape:
        raise ValueError(
            f"Unexpected recursive image shape {recursive.shape[:2]} for "
            f"{recursive_path}; expected {expected_recursive_shape}")
    if comparison.shape[0] <= BANNER_HEIGHT or comparison.shape[1] % 2:
        raise ValueError(
            f"Unexpected occupancy comparison shape {comparison.shape[:2]} "
            f"for {comparison_path}")

    grid = recursive[BANNER_HEIGHT:]
    panels = [
        grid[:PANEL_SIZE, :PANEL_SIZE],
        grid[:PANEL_SIZE, PANEL_SIZE:],
        grid[PANEL_SIZE:, :PANEL_SIZE],
        grid[PANEL_SIZE:, PANEL_SIZE:],
    ]

    comparison_grid = comparison[BANNER_HEIGHT:]
    half_width = comparison_grid.shape[1] // 2
    gt_panel = comparison_grid[:, half_width:]
    gt_panel = cv2.resize(
        gt_panel,
        (PANEL_SIZE, PANEL_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    panels.append(gt_panel)
    return [add_caption(panel, label)
            for panel, label in zip(panels, ROW_LABELS)]


def compose_grid(sample_panels):
    height = 2 * CONTENT_HEIGHT + GAP
    width = (len(ROW_LABELS) * PANEL_SIZE +
             (len(ROW_LABELS) - 1) * GAP)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for row_index in range(2):
        y0 = row_index * (CONTENT_HEIGHT + GAP)
        for column_index in range(len(ROW_LABELS)):
            x0 = column_index * (PANEL_SIZE + GAP)
            canvas[y0:y0 + CONTENT_HEIGHT, x0:x0 + PANEL_SIZE] = (
                sample_panels[row_index][column_index])
    return canvas


def main():
    args = parse_args()
    sample_panels = [
        extract_panels(args.source_root, sample) for sample in args.samples
    ]
    output = compose_grid(sample_panels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output,
                       [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"Could not write {args.output}")


if __name__ == "__main__":
    main()
