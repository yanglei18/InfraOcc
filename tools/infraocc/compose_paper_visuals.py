import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PROJECT_FIG_DIR = ROOT / 'projects' / 'InfraOcc' / 'figures'
UPDATE_FIG_DIR = ROOT / 'projects' / 'InfraOcc' / 'figures_update' / 'robustness'
PAPER_FIG_DIR = ROOT / 'docs' / 'TPAMI2026' / 'local-git' / 'figures'
RERUN_FIG_DIR = ROOT / 'projects' / 'InfraOcc' / 'figures_update' / 'rerun_occ'
FIG09_CAMERA_METHOD_DIR = (
    ROOT / 'projects' / 'InfraOcc' / 'figures_update' / 'fig09_camera_methods')
DYNAMIC_OCC_COLORS = np.array(
    [
        (203, 192, 255),  # bicycle
        (0, 255, 255),    # bus
        (245, 150, 0),    # car
        (0, 127, 255),    # motorcycle
        (0, 0, 255),      # pedestrian
        (240, 32, 160),   # truck
    ],
    dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compose InfraOcc paper-ready visual figures from cached assets.')
    parser.add_argument(
        '--source-dir',
        type=Path,
        default=RERUN_FIG_DIR,
        help='Freshly regenerated figure directory containing manifest.json.')
    parser.add_argument(
        '--modality-source-dir',
        type=Path,
        default=None,
        help='Optional manifest directory for Fig. 9 camera-method panels.')
    parser.add_argument(
        '--project-output-dir',
        type=Path,
        default=UPDATE_FIG_DIR,
        help='Directory for project-side figure outputs.')
    parser.add_argument(
        '--paper-output-dir',
        type=Path,
        default=PAPER_FIG_DIR,
        help='Directory for LaTeX figure outputs.')
    return parser.parse_args()


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(path))
    return image


def strip_existing_banner(image):
    h, w = image.shape[:2]
    if w == 600 and h == 648:
        return image[:600]
    if w == 600 and h >= 688:
        return image[h - 600:h]
    if w == 1408 and h >= 600:
        return image[h - 512:h]
    return image


def resize_to_fit(image, width, height, interpolation=None):
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    if interpolation is None:
        interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LANCZOS4
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def resize_to_fill(image, width, height):
    h, w = image.shape[:2]
    scale = max(width / max(w, 1), height / max(h, 1))
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_LANCZOS4
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    y0 = max((new_h - height) // 2, 0)
    x0 = max((new_w - width) // 2, 0)
    return resized[y0:y0 + height, x0:x0 + width]


def add_label(image, label, label_height=50, bg=(28, 28, 28)):
    image = np.asarray(image)
    canvas = np.full((image.shape[0] + label_height, image.shape[1], 3),
                     255,
                     dtype=np.uint8)
    canvas[label_height:] = image
    cv2.rectangle(canvas, (0, 0), (image.shape[1], label_height), bg, -1)
    cv2.putText(canvas, label, (14, int(label_height * 0.68)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def draw_centered_text(canvas,
                       text,
                       font_scale=0.58,
                       thickness=1,
                       color=(28, 28, 28)):
    lines = text.split('\n')
    sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        thickness)[0] for line in lines
    ]
    line_height = max((height for _, height in sizes), default=16) + 8
    total_height = line_height * len(lines)
    y = (canvas.shape[0] - total_height) // 2 + line_height - 4
    for line, (width, _) in zip(lines, sizes):
        x = (canvas.shape[1] - width) // 2
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)
        y += line_height


def framed_cell(image, border=(205, 210, 218)):
    cell = image.copy()
    cv2.rectangle(cell, (0, 0), (cell.shape[1] - 1, cell.shape[0] - 1),
                  border, 2)
    return cell


def text_cell(text,
              width,
              height,
              bg=(246, 248, 250),
              color=(26, 32, 44),
              font_scale=0.58,
              thickness=1):
    cell = np.full((height, width, 3), bg, dtype=np.uint8)
    draw_centered_text(cell, text, font_scale=font_scale, thickness=thickness,
                       color=color)
    return framed_cell(cell)


def header_cell(text, width, height):
    return text_cell(text,
                     width,
                     height,
                     bg=(36, 44, 58),
                     color=(255, 255, 255),
                     font_scale=0.95,
                     thickness=2)


def image_cell(image, width, height, preserve_edges=False):
    interpolation = cv2.INTER_NEAREST if preserve_edges else None
    return framed_cell(
        resize_to_fit(
            strip_existing_banner(image),
            width,
            height,
            interpolation=interpolation))


def pad_to_canvas(image, width, height, y_align='center', x_align='center'):
    image = strip_existing_banner(np.asarray(image))
    h, w = image.shape[:2]
    if h > height or w > width:
        raise ValueError(
            f'Native image {w}x{h} exceeds target cell {width}x{height}.')
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    if y_align == 'top':
        y0 = 0
    elif y_align == 'bottom':
        y0 = height - h
    else:
        y0 = (height - h) // 2
    if x_align == 'left':
        x0 = 0
    elif x_align == 'right':
        x0 = width - w
    else:
        x0 = (width - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = image
    return canvas


def image_cell_native(image, width, height):
    return framed_cell(pad_to_canvas(image, width, height))


def image_cell_highres(image, width, height):
    image = strip_existing_banner(np.asarray(image))
    resized = resize_to_fit(image, width, height)
    return framed_cell(resized)


def blank_cell(width, height, bg=(255, 255, 255)):
    return framed_cell(np.full((height, width, 3), bg, dtype=np.uint8))


def scale_brightness(image, factor):
    image = np.asarray(image, dtype=np.float32) * float(factor)
    return np.clip(image, 0, 255).astype(np.uint8)


def image_cell_stretched(image, width, height):
    image = strip_existing_banner(np.asarray(image))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return framed_cell(resized)


def mechanism_error_cell(image, width, height):
    image = np.asarray(image)
    if image.shape[1] == 600 and image.shape[0] == 648:
        image = image[:600]
    cell = pad_to_canvas(image, width, height)
    legend = [
        ('Correct', (70, 170, 70)),
        ('Missed', (70, 70, 230)),
        ('False', (230, 90, 60)),
    ]
    y = height - 38
    x_offsets = [18, width // 2 - 58, width - 190]
    for (label, color), x in zip(legend, x_offsets):
        cv2.rectangle(cell, (x - 3, y - 3), (x + 92, y + 23),
                      (245, 245, 245), -1)
        cv2.rectangle(cell, (x, y), (x + 17, y + 17), color, -1)
        cv2.putText(cell, label, (x + 24, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (60, 60, 60), 1,
                    cv2.LINE_AA)
    return framed_cell(cell)


def dynamic_gt_foreground_cell(image, width, height):
    image = strip_existing_banner(np.asarray(image))
    mask = np.zeros(image.shape[:2], dtype=bool)
    for color in DYNAMIC_OCC_COLORS:
        mask |= np.all(image == color, axis=2)
    foreground = np.full_like(image, 255)
    foreground[mask] = image[mask]
    return image_cell_highres(foreground, width, height)


def stack_grid(cells, cols):
    rows = []
    for start in range(0, len(cells), cols):
        rows.append(cv2.hconcat(cells[start:start + cols]))
    return cv2.vconcat(rows)


def save_png_and_pdf(image, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f'{stem}.png'
    pdf_path = output_dir / f'{stem}.pdf'
    cv2.imwrite(str(png_path), image)
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(str(pdf_path))
    return png_path, pdf_path


def save_png(image, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f'{stem}.png'
    cv2.imwrite(str(png_path), image)
    return png_path


def copy_existing_outputs(source_dir,
                          dest_dir,
                          stems,
                          stem_map=None,
                          suffixes=('.png', '.pdf', '.json')):
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem_map = stem_map or {}
    for stem in stems:
        dest_stem = stem_map.get(stem, stem)
        for suffix in suffixes:
            src = source_dir / f'{stem}{suffix}'
            if src.exists():
                shutil.copy2(src, dest_dir / f'{dest_stem}{suffix}')


def resolve_manifest_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def load_manifest_figures(source_dir):
    manifest_path = source_dir / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(
            f'{manifest_path} does not exist. Regenerate visual assets first.')
    import json

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    figures = manifest.get('figures', [])
    if not figures:
        raise ValueError(f'No figures listed in {manifest_path}.')
    resolved = []
    for figure in figures:
        assets = figure.get('assets', {})
        if not assets:
            raise ValueError(f'No assets listed in {manifest_path}.')
        figure = dict(figure)
        figure['assets'] = {
            name: resolve_manifest_path(path)
            for name, path in assets.items()
        }
        resolved.append(figure)
    return manifest, resolved


def read_asset(assets, name):
    if name not in assets:
        raise KeyError(f'Missing regenerated asset {name!r}.')
    return read_image(assets[name])


def read_optional_asset(assets, name, anchor_name=None):
    if name in assets:
        return read_asset(assets, name)
    if anchor_name is None or anchor_name not in assets:
        return None
    anchor_path = Path(assets[anchor_name])
    anchor_suffix = f'_{anchor_name}.png'
    if not anchor_path.name.endswith(anchor_suffix):
        return None
    candidate = anchor_path.with_name(
        anchor_path.name[:-len(anchor_suffix)] + f'_{name}.png')
    if not candidate.exists():
        return None
    return read_image(candidate)


_COLORMAP_INDEX_CACHE = {}


def decode_colormap_scalar(image, colormap):
    image = strip_existing_banner(np.asarray(image))
    cache_key = int(colormap)
    if cache_key not in _COLORMAP_INDEX_CACHE:
        lut = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8).reshape(256, 1),
            colormap).reshape(256, 3)
        lut_keys = ((lut[:, 0].astype(np.uint32) << 16)
                    | (lut[:, 1].astype(np.uint32) << 8)
                    | lut[:, 2].astype(np.uint32))
        _COLORMAP_INDEX_CACHE[cache_key] = (lut.astype(np.int32), {
            int(key): index
            for index, key in enumerate(lut_keys)
        })
    lut, exact_index = _COLORMAP_INDEX_CACHE[cache_key]
    pixels = image.reshape(-1, 3)
    pixel_keys = ((pixels[:, 0].astype(np.uint32) << 16)
                  | (pixels[:, 1].astype(np.uint32) << 8)
                  | pixels[:, 2].astype(np.uint32))
    unique_keys, inverse = np.unique(pixel_keys, return_inverse=True)
    unique_values = np.empty(len(unique_keys), dtype=np.uint8)
    for key_index, key in enumerate(unique_keys):
        exact = exact_index.get(int(key))
        if exact is not None:
            unique_values[key_index] = exact
            continue
        color = np.array(
            [(key >> 16) & 255, (key >> 8) & 255, key & 255],
            dtype=np.int32)
        distances = ((lut - color) ** 2).sum(axis=1)
        unique_values[key_index] = int(np.argmin(distances))
    scalar = unique_values[inverse].reshape(image.shape[:2])
    return scalar.astype(np.float32) / 255.0


def normalize_scalar(scalar,
                     percentile=(1.0, 99.6),
                     gamma=1.05,
                     eps=1e-6):
    scalar = np.nan_to_num(np.asarray(scalar, dtype=np.float32),
                           nan=0.0,
                           posinf=0.0,
                           neginf=0.0)
    values = scalar[np.isfinite(scalar)]
    if values.size > 0:
        lo, hi = np.percentile(values, percentile)
    else:
        lo, hi = 0.0, 1.0
    if float(hi - lo) < eps:
        hi = lo + 1.0
    scalar = np.clip((scalar - lo) / (hi - lo), 0.0, 1.0)
    if abs(float(gamma) - 1.0) > 1e-6:
        scalar = np.power(scalar, float(gamma))
    return scalar


def dynamic_aware_feature_cell(assets, width, height):
    energy_image = read_asset(assets, 'dynamic_input_energy')
    dynamic_conf_image = read_optional_asset(
        assets, 'dynamic_conf', anchor_name='dynamic_input_energy')
    suppression_image = read_optional_asset(
        assets, 'suppression_gate', anchor_name='dynamic_input_energy')

    energy = decode_colormap_scalar(energy_image, cv2.COLORMAP_MAGMA)
    context = 0.34 * np.power(np.clip(energy, 0.0, 1.0), 1.45)
    dynamic_weight = np.ones_like(energy, dtype=np.float32)
    if dynamic_conf_image is not None:
        dynamic_conf = decode_colormap_scalar(dynamic_conf_image,
                                              cv2.COLORMAP_VIRIDIS)
        if dynamic_conf.shape != energy.shape:
            dynamic_conf = cv2.resize(
                dynamic_conf,
                (energy.shape[1], energy.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        dynamic_weight *= np.power(np.clip(dynamic_conf, 0.0, 1.0), 0.40)
    if suppression_image is not None:
        suppression_strength = decode_colormap_scalar(suppression_image,
                                                      cv2.COLORMAP_INFERNO)
        if suppression_strength.shape != energy.shape:
            suppression_strength = cv2.resize(
                suppression_strength,
                (energy.shape[1], energy.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        dynamic_weight *= np.power(
            np.clip(1.0 - suppression_strength, 0.0, 1.0), 0.18)

    weighted_energy = context + 0.66 * energy * dynamic_weight
    scalar = normalize_scalar(weighted_energy)
    heatmap = cv2.applyColorMap(
        np.round(scalar * 255.0).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return image_cell_highres(heatmap, width, height)


def native_column_widths(figures, columns):
    widths = []
    for _, asset_name in columns:
        widths.append(
            max(
                strip_existing_banner(read_asset(figure['assets'],
                                                 asset_name)).shape[1]
                for figure in figures))
    return widths


def native_row_height(figures, columns):
    return max(
        strip_existing_banner(read_asset(figure['assets'],
                                         asset_name)).shape[0]
        for figure in figures for _, asset_name in columns)


def compose_modality_qualitative(figures, max_samples=2):
    figures = figures[:max_samples]
    header_h = 96
    columns = [
        ('Camera', 'camera_grid'),
        ('GT OCC', 'gt_bev'),
        ('TPVFormer', 'tpvformer_bev'),
        ('SparseOCC', 'sparseocc_bev'),
        ('STCOcc', 'plain_bev'),
        ('ProSD-Occ (C)', 'prosd_c_bev'),
    ]
    row_h = 720
    camera_image = strip_existing_banner(
        read_asset(figures[0]['assets'], 'camera_grid'))
    camera_ratio = camera_image.shape[1] / max(camera_image.shape[0], 1)
    widths = [int(round(row_h * camera_ratio))] + [row_h] * (len(columns) - 1)

    cells = [
        text_cell(
            title,
            width,
            header_h,
            bg=(36, 44, 58),
            color=(255, 255, 255),
            font_scale=1.5,
            thickness=3) for (title, _), width in zip(columns, widths)
    ]
    for figure in figures:
        assets = figure['assets']
        for (_, asset_name), width in zip(columns, widths):
            cells.append(
                image_cell_highres(read_asset(assets, asset_name), width,
                                   row_h))
    return stack_grid(cells, cols=len(columns))


def compose_mechanism_features(figures, max_samples=3):
    figures = figures[:max_samples]
    header_h = 96
    columns = [
        ('Static Conf.', 'static_conf'),
        ('Supp. Strength', 'suppression_gate'),
        ('Dynamic-aware Feature', 'dynamic_input_energy'),
        ('GT Dynamic Foreground', 'gt_bev'),
    ]
    widths = native_column_widths(figures, columns)
    row_h = native_row_height(figures, columns)
    max_width = max(widths)
    widths = [max_width for _ in widths]
    cells = []
    for (title, _), width in zip(columns, widths):
        cells.append(
            text_cell(
                title,
                width,
                header_h,
                bg=(36, 44, 58),
                color=(255, 255, 255),
                font_scale=1.7,
                thickness=3))
    for figure in figures:
        assets = figure['assets']
        for (_, asset_name), width in zip(columns, widths):
            if asset_name == 'gt_bev':
                cells.append(
                    dynamic_gt_foreground_cell(
                        read_asset(assets, asset_name), width, row_h))
                continue
            if asset_name == 'static_conf':
                cells.append(
                    image_cell_highres(
                        scale_brightness(read_asset(assets, asset_name),
                                         0.75), width, row_h))
                continue
            if asset_name == 'dynamic_input_energy':
                cells.append(dynamic_aware_feature_cell(assets, width, row_h))
                continue
            cells.append(
                image_cell_highres(read_asset(assets, asset_name), width,
                                   row_h))
    return stack_grid(cells, cols=len(columns))


def compose_mechanism_errors(figures, max_samples=3):
    figures = figures[:max_samples]
    header_h = 96
    columns = [
        ('GT Occ', 'gt_bev'),
        ('STCOcc', 'plain_bev'),
        ('ProSD-Occ', 'prosd_c_bev'),
        ('Plain Error', 'plain_dynamic_error'),
        ('ProSD Error', 'prosd_dynamic_error'),
    ]
    widths = native_column_widths(figures, columns)
    row_h = native_row_height(figures, columns)
    cells = []
    for (title, _), width in zip(columns, widths):
        cells.append(
            text_cell(
                title,
                width,
                header_h,
                bg=(36, 44, 58),
                color=(255, 255, 255),
                font_scale=1.7,
                thickness=3))
    for figure in figures:
        assets = figure['assets']
        for (_, asset_name), width in zip(columns, widths):
            cells.append(
                image_cell_native(read_asset(assets, asset_name), width,
                                  row_h))
    return stack_grid(cells, cols=len(columns))


def main():
    args = parse_args()
    _, figures = load_manifest_figures(args.source_dir)
    modality_source_dir = args.modality_source_dir
    if modality_source_dir is None and (FIG09_CAMERA_METHOD_DIR /
                                        'manifest.json').exists():
        modality_source_dir = FIG09_CAMERA_METHOD_DIR
    if modality_source_dir is None:
        modality_figures = figures
    else:
        _, modality_figures = load_manifest_figures(modality_source_dir)
    generated = {
        'infraocc_modality_qualitative':
        compose_modality_qualitative(modality_figures),
        'prosd_mechanism_features':
        compose_mechanism_features(figures),
        'prosd_mechanism_errors':
        compose_mechanism_errors(figures),
    }
    paper_stems = {
        'infraocc_modality_qualitative': 'fig09-modality-qualitative',
        'prosd_mechanism_features': 'fig10-prosd-mechanism-features',
        'prosd_mechanism_errors': 'fig11-prosd-mechanism-errors',
    }
    for stem, image in generated.items():
        save_png_and_pdf(image, args.project_output_dir, stem)
        paper_stem = paper_stems.get(stem, stem)
        if stem == 'infraocc_modality_qualitative':
            save_png(image, args.paper_output_dir, paper_stem)
        else:
            save_png_and_pdf(image, args.paper_output_dir, paper_stem)

    copy_existing_outputs(
        args.project_output_dir,
        args.paper_output_dir,
        stems=['robustness_c_prosd_vs_plain'],
        stem_map={'robustness_c_prosd_vs_plain': 'fig12-robustness-curves'},
        suffixes=('.pdf', ))
    print(f'Saved composed figures to {args.project_output_dir}')
    print(f'Saved LaTeX copies to {args.paper_output_dir}')


if __name__ == '__main__':
    main()
