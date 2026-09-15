#!/usr/bin/env python3
"""Export clean GT and RoadOcc semantic-occupancy BEV sample images."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmdet3d.core.visualizer.occ_visualization import (  # noqa: E402
    make_occ_color_map, render_occ_bev_rgb)


CLASS_NAMES = (
    'others', 'barrier', 'bicycle', 'bus', 'car',
    'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free')
EMPTY_IDX = CLASS_NAMES.index('free')
DEFAULT_SOURCE = (
    ROOT / 'work_dirs/stcroadocc_c_2x4_24e/figures_3d_comparison/'
    'Fig4-comparison-source/volumes')
DEFAULT_OUTPUT = ROOT / 'docs/paper/ICLR2027/figures/bev_occ_samples'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--size', type=int, default=1200)
    parser.add_argument('--preview-tile-size', type=int, default=500)
    return parser.parse_args()


def load_semantics(path):
    with np.load(path, allow_pickle=False) as payload:
        semantics = payload['semantics']
    if semantics.ndim != 3:
        raise ValueError(
            'Expected a three-dimensional occupancy volume, got {} from {}.'
            .format(semantics.shape, path))
    return semantics.astype(np.int64, copy=False)


def render_volume(path, size, palette):
    semantics = load_semantics(path)
    return render_occ_bev_rgb(
        semantics,
        EMPTY_IDX,
        palette,
        image_size_hw=(size, size))


def label_font(size):
    candidates = (
        '/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def make_contact_sheet(records, output_path, tile_size):
    rows = len(records)
    header = 42
    gutter = 12
    canvas = Image.new(
        'RGB',
        (2 * tile_size + gutter, rows * (tile_size + header) +
         (rows - 1) * gutter),
        'white')
    draw = ImageDraw.Draw(canvas)
    font = label_font(24)
    for row, record in enumerate(records):
        top = row * (tile_size + header + gutter)
        for column, key in enumerate(('gt', 'roadocc')):
            left = column * (tile_size + gutter)
            panel = Image.open(record[key]).convert('RGB').resize(
                (tile_size, tile_size), Image.Resampling.NEAREST)
            canvas.paste(panel, (left, top + header))
            title = '{}  {}'.format(record['sample_id'],
                                    'GT' if key == 'gt' else 'RoadOcc')
            draw.text((left + 4, top + 7), title, fill=(24, 33, 43), font=font)
    canvas.save(output_path, compress_level=9)


def main():
    args = parse_args()
    gt_paths = sorted((args.source_root / 'GT').glob('*.npz'))
    if not gt_paths:
        raise FileNotFoundError(
            'No GT occupancy volumes found under {}.'.format(args.source_root))

    # ``make_occ_color_map`` follows the OpenCV BGR convention. Convert it to
    # RGB before passing the palette to the PIL-facing export path.
    palette = make_occ_color_map(CLASS_NAMES)[:, ::-1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for gt_path in gt_paths:
        sample_id, token = gt_path.stem.split('_', 1)
        roadocc_path = args.source_root / 'RoadOcc' / gt_path.name
        if not roadocc_path.is_file():
            raise FileNotFoundError(roadocc_path)

        gt_output = args.output_dir / '{}_gt_occ_bev.png'.format(sample_id)
        roadocc_output = (
            args.output_dir / '{}_roadocc_occ_bev.png'.format(sample_id))
        Image.fromarray(render_volume(gt_path, args.size, palette)).save(
            gt_output, compress_level=9)
        Image.fromarray(render_volume(roadocc_path, args.size, palette)).save(
            roadocc_output, compress_level=9)
        records.append({
            'sample_id': sample_id,
            'token': token,
            'gt_source': str(gt_path),
            'roadocc_source': str(roadocc_path),
            'gt': str(gt_output),
            'roadocc': str(roadocc_output),
        })

    contact_sheet = args.output_dir / 'bev_occ_contact_sheet.png'
    make_contact_sheet(records, contact_sheet, args.preview_tile_size)
    manifest = {
        'projection': 'topmost occupied semantic voxel along height',
        'orientation': 'shared project BEV image convention',
        'image_size': [args.size, args.size],
        'class_names': list(CLASS_NAMES),
        'samples': records,
        'contact_sheet': str(contact_sheet),
    }
    manifest_path = args.output_dir / 'manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
