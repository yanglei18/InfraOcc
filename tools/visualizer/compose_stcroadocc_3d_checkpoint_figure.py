"""Compose a VoxDet-style checkpoint comparison from matched 3D renders."""

import argparse
import glob
import os.path as osp

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compose a multi-sample STCRoadOcc 3D comparison')
    parser.add_argument('--indices', type=int, nargs='+', required=True)
    parser.add_argument('--gt-dir', required=True)
    parser.add_argument('--baseline-dir', required=True)
    parser.add_argument('--baseline-tag', default='epoch10')
    parser.add_argument('--baseline-label', default='STCOcc')
    parser.add_argument('--final-dir', required=True)
    parser.add_argument('--final-tag', default='final')
    parser.add_argument('--final-label', default='RoadOcc (Ours)')
    parser.add_argument('--sample-font-size', type=int, default=40)
    parser.add_argument('--row-label-font-size', type=int, default=40)
    parser.add_argument(
        '--detail-boxes', nargs='+', required=True,
        help=('One source-image crop per column, formatted as x0,y0,x1,y1. '
              'The same crop is used for GT, baseline, and final.'))
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def font(size, bold=False):
    family = ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
              if bold else
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(family, size=size)


def one_match(directory, pattern):
    paths = sorted(glob.glob(osp.join(directory, pattern)))
    if len(paths) != 1:
        raise RuntimeError(
            f'Expected one file for {directory}/{pattern}, got {paths}.')
    return paths[0]


def fit_inside(image, size, background=(255, 255, 255)):
    image = image.convert('RGB')
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)),
         max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', size, background)
    offset = ((size[0] - resized.width) // 2,
              (size[1] - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas, scale, offset


def parse_detail_box(value):
    coordinates = tuple(int(item) for item in value.split(','))
    if len(coordinates) != 4:
        raise ValueError(
            f'Detail box must be x0,y0,x1,y1, got {value!r}.')
    x0, y0, x1, y1 = coordinates
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f'Invalid detail box {value!r}.')
    return coordinates


def add_detail_inset(cell, source, source_box, scale, offset):
    """Add a matched VoxDet-style source box and magnified inset."""
    draw = ImageDraw.Draw(cell)
    x0, y0, x1, y1 = source_box
    mapped_box = (
        round(offset[0] + x0 * scale),
        round(offset[1] + y0 * scale),
        round(offset[0] + x1 * scale),
        round(offset[1] + y1 * scale),
    )
    draw.rectangle(mapped_box, outline=(0, 0, 0), width=5)

    inset_width = 260
    inset_height = 186
    inset = source.crop(source_box).convert('RGB').resize(
        (inset_width, inset_height), Image.Resampling.LANCZOS)
    # Keep every inset in the upper-left background area so the main road and
    # intersection remain visible in the full-scene rendering.
    inset_y = 10
    inset_x = 10
    draw.rectangle(
        (inset_x - 5, inset_y - 5,
         inset_x + inset_width + 4, inset_y + inset_height + 4),
        fill=(0, 0, 0))
    cell.paste(inset, (inset_x, inset_y))
    return cell


def draw_vertical_label(canvas, label, box, font_size=40, bold=False):
    label_font = font(font_size, bold=bold)
    bounds = label_font.getbbox(label)
    label_image = Image.new(
        'RGBA', (bounds[2] - bounds[0] + 20, bounds[3] - bounds[1] + 20),
        (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_image)
    label_draw.text((10 - bounds[0], 10 - bounds[1]), label,
                    font=label_font, fill=(22, 22, 22, 255))
    label_image = label_image.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    x = box[0] + (box[2] - box[0] - label_image.width) // 2
    y = box[1] + (box[3] - box[1] - label_image.height) // 2
    canvas.paste(label_image, (x, y), label_image)


def compose(args):
    if len(args.detail_boxes) != len(args.indices):
        raise ValueError(
            'Provide exactly one --detail-boxes entry per dataset index.')
    detail_boxes = [parse_detail_box(value) for value in args.detail_boxes]

    column_width = 640
    column_gap = 20
    left_margin = 142
    right_margin = 24
    top_margin = 62
    volume_height = 430
    row_gap = 10
    bottom_margin = 20
    rows = (
        ('Ground-truth', volume_height),
        (args.baseline_label, volume_height),
        (args.final_label, volume_height),
    )
    width = (left_margin + len(args.indices) * column_width
             + (len(args.indices) - 1) * column_gap + right_margin)
    height = (top_margin + sum(row[1] for row in rows)
              + row_gap * (len(rows) - 1) + bottom_margin)
    canvas = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    row_y = []
    current_y = top_margin
    for _, row_height in rows:
        row_y.append(current_y)
        current_y += row_height + row_gap

    for column, dataset_index in enumerate(args.indices):
        x = left_margin + column * (column_width + column_gap)
        gt_path = one_match(args.gt_dir, f'{dataset_index:04d}_*_gt.png')
        baseline_path = one_match(
            args.baseline_dir,
            f'{dataset_index:04d}_*_{args.baseline_tag}.png')
        final_path = one_match(
            args.final_dir, f'{dataset_index:04d}_*_{args.final_tag}.png')
        paths = (gt_path, baseline_path, final_path)
        for row_index, path in enumerate(paths):
            source = Image.open(path).convert('RGB')
            source_box = detail_boxes[column]
            if (source_box[0] < 0 or source_box[1] < 0
                    or source_box[2] > source.width
                    or source_box[3] > source.height):
                raise ValueError(
                    f'Detail box {source_box} lies outside {source.size}.')
            cell, scale, offset = fit_inside(
                source, (column_width, rows[row_index][1]))
            cell = add_detail_inset(
                cell, source, source_box, scale, offset)
            canvas.paste(cell, (x, row_y[row_index]))

        title = f'Sample ID: {dataset_index:04d}'
        title_font = font(args.sample_font_size)
        title_bounds = title_font.getbbox(title)
        title_y = ((top_margin - (title_bounds[3] - title_bounds[1])) // 2
                   - title_bounds[1])
        draw.text((x + 5, title_y), title, font=title_font,
                  fill=(20, 20, 20))

    for row_index, (label, row_height) in enumerate(rows):
        draw_vertical_label(
            canvas,
            label,
            (0, row_y[row_index], left_margin - 12,
             row_y[row_index] + row_height),
            font_size=args.row_label_font_size,
            bold=row_index == 2)

    canvas.save(args.output, quality=96, subsampling=0)
    return canvas.size


def main():
    args = parse_args()
    output_parent = osp.dirname(osp.abspath(args.output))
    if output_parent:
        import os
        os.makedirs(output_parent, exist_ok=True)
    size = compose(args)
    print(f'Wrote {args.output} at {size[0]}x{size[1]}', flush=True)


if __name__ == '__main__':
    main()
