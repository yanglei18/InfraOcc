#!/usr/bin/env python3
"""Correct the Transport callout in the RoadOcc P/T/R teaser.

The teaser is a raster-first scientific figure.  This patch keeps every panel
unchanged except the orange T callout in panel (f): the old frame highlights a
horizontal car, whereas panels (d) and (e) establish the slanted car as the
current Transport target.
"""

import argparse
import sys
import zlib
from pathlib import Path

sys.dont_write_bytecode = True

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PNG = (ROOT / 'docs/paper/ICLR2027/figures/'
               'roadocc_ptr_teaser.png')
DEFAULT_PDFS = (
    ROOT / 'docs/paper/ICLR2027/Fig1-ptrteaser.pdf',
    ROOT / 'docs/paper/ICLR2027/figures/roadocc_ptr_teaser.pdf',
)
SOURCE_SEMANTIC = (
    ROOT / 'projects/RoadOcc/preprocess/occflow_vis/'
    '2a92544c-1e8e-43ae-825e-074a9ff1ef2f/'
    '0a5577df-259d-443a-b5ff-b171220290e4.png')

# Panel (f) begins here in the 1288 x 962 composite.
PANEL_F_XY = (868, 542)
TRANSPORT_RGB = np.asarray((232, 142, 38), dtype=np.uint8)
REFRESH_RGB = np.asarray((44, 160, 92), dtype=np.uint8)
CAR_RGB = np.asarray((245, 150, 0), dtype=np.uint8)
OUTLINE_RGB = np.asarray((35, 42, 48), dtype=np.uint8)
HISTORY_RGB = np.asarray((105, 113, 122), dtype=np.uint8)

# Inclusive local coordinates measured from panel (f).  The correction region
# covers both the published callout and this script's corrected placement, so
# rerunning the script is idempotent.
CORRECTION_ROI = (115, 125, 305, 225)
NEW_BADGE_XY = (124, 141)
NEW_FRAME = (126, 184, 155, 202)

# The evidence frames in (d) and (e) should be as tight as the T frame in (f).
D_FRAME_ROI = (120, 150, 275, 216)
E_FRAME_ROI = (170, 155, 310, 220)
D_FRAME = NEW_FRAME
E_FRAME = (176, 188, 205, 206)
E_EMPTY_ROI = (120, 150, 275, 214)
E_EMPTY_FRAME = (128, 184, 154, 201)
E_NO_HISTORY_ROI = (360, 188, 408, 224)
E_NO_HISTORY_FRAME = (367, 192, 402, 220)

D_TARGET_LABEL = (178, 120, 303, 147)
D_REFRESH_LABEL = (260, 158, 419, 185)
E_SOURCE_LABEL = (215, 124, 345, 153)
E_NO_HISTORY_LABEL = (288, 158, 403, 187)
F_REBUILD_ROI = (190, 110, 330, 220)

FLAT_PALETTE = np.asarray((
    (255, 255, 255), (242, 242, 242), (223, 223, 223),
    (207, 214, 222), (196, 196, 196), (195, 195, 195),
    (170, 170, 170), (159, 159, 159), (105, 113, 122),
    (35, 42, 48), (245, 150, 0), (232, 142, 38),
    (44, 160, 92), (54, 116, 217), (0, 255, 255),
    (50, 120, 255), (0, 0, 255), (150, 240, 255),
    (203, 192, 255),
), dtype=np.int16)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_PNG)
    parser.add_argument('--output', type=Path, default=DEFAULT_PNG)
    parser.add_argument(
        '--pdf-output',
        type=Path,
        nargs='*',
        default=list(DEFAULT_PDFS),
        help='PDF copies rebuilt from the corrected 300-dpi PNG.')
    return parser.parse_args()


def inclusive_slice(box):
    x0, y0, x1, y1 = box
    return np.s_[y0:y1 + 1, x0:x1 + 1]


def color_mask(image, color, tolerance=10):
    """Match flat figure colors robustly after prior PDF/JPEG encoding."""
    difference = np.abs(image.astype(np.int16) - color.astype(np.int16))
    return np.max(difference, axis=-1) <= int(tolerance)


def target_car_support(current_panel):
    """Return the semantic pixels and outline of d's selected slanted car."""
    car = color_mask(current_panel, CAR_RGB, tolerance=12).astype(np.uint8)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        car, connectivity=8)
    candidates = range(1, component_count)
    component = min(
        candidates,
        key=lambda index: np.linalg.norm(centroids[index] - (141, 191)))
    if stats[component, cv2.CC_STAT_AREA] < 20:
        raise ValueError('The slanted Transport target was not found.')
    seed = (labels == component).astype(np.uint8)
    vicinity = cv2.dilate(seed, np.ones((7, 7), dtype=np.uint8)).astype(bool)
    source_pixel = (color_mask(current_panel, CAR_RGB, tolerance=12) |
                    color_mask(current_panel, OUTLINE_RGB, tolerance=12))
    return vicinity, vicinity & source_pixel


def remove_old_transport(panel, current_panel):
    """Erase only the old orange annotation, preserving semantic colors."""
    panel = panel.copy()
    annotation = color_mask(panel, TRANSPORT_RGB, tolerance=12)
    mask = np.zeros(panel.shape[:2], dtype=np.uint8)

    roi_slice = inclusive_slice(CORRECTION_ROI)
    roi_annotation = annotation[roi_slice].astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        roi_annotation, connectivity=8)
    if component_count <= 1:
        raise ValueError('Transport badge was not found in panel (f).')
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = stats[component]
    if area < 900 or not (35 <= width <= 42 and 35 <= height <= 42):
        raise ValueError('Unexpected Transport badge geometry: {}'.format(
            (x, y, width, height, area)))
    roi_x0, roi_y0, _, _ = CORRECTION_ROI
    badge_box = (roi_x0 + int(x), roi_y0 + int(y),
                 roi_x0 + int(x + width - 1),
                 roi_y0 + int(y + height - 1))
    badge_slice = inclusive_slice(badge_box)
    badge = panel[badge_slice]
    badge_patch = badge.copy()
    badge_mask = color_mask(badge, TRANSPORT_RGB, tolerance=12)

    # Include the white T glyph and its antialiasing, but not unrelated white
    # occupancy context outside the known badge footprint.
    neutral = np.ptp(badge.astype(np.int16), axis=-1) <= 8
    glyph = neutral & (badge.min(axis=-1) >= 205)
    badge_mask |= glyph
    mask[badge_slice] = badge_mask.astype(np.uint8) * 255
    mask[roi_slice] = np.maximum(
        mask[roi_slice], roi_annotation * 255)

    # Remove the entire published badge footprint as well as the detected
    # footprint.  This clears antialiasing and makes repeated runs clean.
    for box in (badge_box, ):
        x0, y0, x1, y1 = box
        x0, y0 = max(x0 - 2, 0), max(y0 - 2, 0)
        x1 = min(x1 + 2, panel.shape[1] - 1)
        y1 = min(y1 + 2, panel.shape[0] - 1)
        mask[inclusive_slice((x0, y0, x1, y1))] = 255

    # Temporarily remove the target itself from the inpainting boundary.  Its
    # exact pixels are restored from the unmodified current-OCC panel below,
    # preventing orange color from bleeding into the erased badge area.
    target_vicinity, target_restore = target_car_support(current_panel)
    mask[target_vicinity] = 255

    # A small-radius inpaint reconstructs the locally uniform road context;
    # semantic car pixels use a distinct orange and remain untouched.
    panel = cv2.inpaint(panel, mask, 3, cv2.INPAINT_NS)
    panel[target_restore] = current_panel[target_restore]
    return panel, badge_patch, badge_mask


def add_correct_transport(panel, badge_patch, badge_mask):
    """Place T beside, and frame, the slanted target car from panel (d)."""
    panel = panel.copy()
    badge_x, badge_y = NEW_BADGE_XY
    badge_h, badge_w = badge_mask.shape
    destination = panel[badge_y:badge_y + badge_h,
                        badge_x:badge_x + badge_w]
    destination[badge_mask] = badge_patch[badge_mask]

    x0, y0, x1, y1 = NEW_FRAME
    cv2.rectangle(
        panel, (x0, y0), (x1, y1), tuple(int(v) for v in TRANSPORT_RGB),
        thickness=4,
        lineType=cv2.LINE_8)
    return panel


def box_mask(shape, boxes):
    mask = np.zeros(shape[:2], dtype=bool)
    for box in boxes:
        mask[inclusive_slice(box)] = True
    return mask


def load_static_context():
    """Recover crisp static classes from the original semantic rendering."""
    source = np.asarray(Image.open(SOURCE_SEMANTIC).convert('RGB'))
    source = cv2.resize(
        source[100:880, :800], (420, 420),
        interpolation=cv2.INTER_NEAREST)
    output = np.full_like(source, 255)
    mapping = {
        (255, 255, 255): (255, 255, 255),
        (255, 0, 255): (196, 196, 196),
        (75, 0, 75): (170, 170, 170),
        (250, 230, 230): (242, 242, 242),
        (0, 175, 0): (195, 195, 195),
        (80, 240, 150): (223, 223, 223),
        (0, 0, 0): (159, 159, 159),
    }
    for source_color, output_color in mapping.items():
        output[np.all(source == source_color, axis=-1)] = output_color
    dynamic = np.zeros(source.shape[:2], dtype=bool)
    for color in ((245, 150, 0), (50, 120, 255), (0, 0, 255),
                  (150, 240, 255), (203, 192, 255)):
        dynamic |= np.all(source == color, axis=-1)
    output[dynamic] = (196, 196, 196)
    return output


def rebuild_transport_context(panel, current_panel, past_panel,
                              static_context):
    """Replace the previously inpainted T region with crisp OCC evidence."""
    output = panel.copy()
    x0, y0, x1, y1 = F_REBUILD_ROI
    roi = inclusive_slice(F_REBUILD_ROI)
    patch = current_panel[roi].copy()

    current_annotation = (
        box_mask(current_panel.shape,
                 (D_TARGET_LABEL, D_REFRESH_LABEL)) |
        color_mask(current_panel, TRANSPORT_RGB, tolerance=12) |
        color_mask(current_panel, REFRESH_RGB, tolerance=12))
    past_annotation = (
        box_mask(past_panel.shape,
                 (E_SOURCE_LABEL, E_NO_HISTORY_LABEL)) |
        color_mask(past_panel, TRANSPORT_RGB, tolerance=12) |
        color_mask(past_panel, HISTORY_RGB, tolerance=10))
    current_mask = current_annotation[roi]
    past_mask = past_annotation[roi]
    use_past = current_mask & ~past_mask
    patch[use_past] = past_panel[roi][use_past]

    # Where both evidence panels contain labels, use the original semantic
    # rendering for a sharp static surface rather than interpolating pixels.
    both = current_mask & past_mask
    patch[both] = static_context[roi][both]
    output[roi] = patch

    # The Persist overlay is already clean and should remain pixel-identical.
    output[30:140, 205:252] = panel[30:140, 205:252]
    return output


def quantize_flat_panel(panel):
    """Undo JPEG halos in flat-color BEV panels without spatial filtering."""
    pixels = panel.reshape(-1, 3).astype(np.int32)
    palette = FLAT_PALETTE.astype(np.int32)
    distance = np.sum(
        (pixels[:, None, :] - palette[None, :, :])**2, axis=-1)
    nearest = palette[np.argmin(distance, axis=1)]
    return nearest.reshape(panel.shape).astype(np.uint8)


def replace_evidence_frame(panel, roi, frame):
    """Remove a large evidence frame and redraw it tightly around its car."""
    panel = panel.copy()
    annotation = color_mask(panel, TRANSPORT_RGB, tolerance=12)
    mask = np.zeros(panel.shape[:2], dtype=np.uint8)
    roi_slice = inclusive_slice(roi)
    mask[roi_slice] = annotation[roi_slice].astype(np.uint8) * 255
    panel = cv2.inpaint(panel, mask, 3, cv2.INPAINT_TELEA)
    x0, y0, x1, y1 = frame
    cv2.rectangle(
        panel, (x0, y0), (x1, y1), tuple(int(v) for v in TRANSPORT_RGB),
        thickness=4,
        lineType=cv2.LINE_8)
    return panel


def draw_dashed_line(panel, start, end, color, thickness=3, dash=7, gap=3):
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length == 0:
        return
    direction = vector / length
    position = 0.0
    while position < length:
        segment_end = min(position + dash, length)
        p0 = np.rint(start + direction * position).astype(int)
        p1 = np.rint(start + direction * segment_end).astype(int)
        cv2.line(
            panel, tuple(p0), tuple(p1), tuple(int(v) for v in color),
            thickness=thickness,
            lineType=cv2.LINE_8)
        position += dash + gap


def replace_target_empty_frame(panel):
    """Draw Target empty and No history with one dashed-box style."""
    panel = panel.copy()
    history = color_mask(panel, HISTORY_RGB, tolerance=10)
    mask = np.zeros(panel.shape[:2], dtype=np.uint8)
    for roi in (E_EMPTY_ROI, E_NO_HISTORY_ROI):
        roi_slice = inclusive_slice(roi)
        mask[roi_slice] = history[roi_slice].astype(np.uint8) * 255
    panel = cv2.inpaint(panel, mask, 3, cv2.INPAINT_TELEA)

    for x0, y0, x1, y1 in (E_EMPTY_FRAME, E_NO_HISTORY_FRAME):
        draw_dashed_line(panel, (x0, y0), (x1, y0), HISTORY_RGB)
        draw_dashed_line(panel, (x1, y0), (x1, y1), HISTORY_RGB)
        draw_dashed_line(panel, (x1, y1), (x0, y1), HISTORY_RGB)
        draw_dashed_line(panel, (x0, y1), (x0, y0), HISTORY_RGB)
    return panel


def save_pdf(image, path):
    """Write a one-page lossless PDF without JPEG re-encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    page_width = width * 72.0 / 300.0
    page_height = height * 72.0 / 300.0
    image_stream = zlib.compress(image.astype(np.uint8).tobytes(), level=9)
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
              width, height, len(image_stream)).encode('ascii') +
         image_stream + b'\nendstream'),
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
    xref_offset = len(payload)
    payload.extend('xref\n0 {}\n'.format(len(objects) + 1).encode('ascii'))
    payload.extend(b'0000000000 65535 f \n')
    for offset in offsets[1:]:
        payload.extend('{:010d} 00000 n \n'.format(offset).encode('ascii'))
    payload.extend(
        ('trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n'.format(
            len(objects) + 1, xref_offset)).encode('ascii'))
    path.write_bytes(payload)


def main():
    args = parse_args()
    image = np.asarray(Image.open(args.input).convert('RGB')).copy()
    if image.shape[:2] != (962, 1288):
        raise ValueError('Expected teaser size 1288x962, got {}'.format(
            image.shape[1::-1]))

    panel_x, panel_y = PANEL_F_XY
    current_panel = image[panel_y:panel_y + 420, 0:420].copy()
    past_panel = image[panel_y:panel_y + 420, 434:854].copy()
    panel = image[panel_y:panel_y + 420, panel_x:panel_x + 420]
    _, badge_patch, badge_mask = remove_old_transport(
        panel, current_panel)
    panel = rebuild_transport_context(
        panel, current_panel, past_panel, load_static_context())
    panel = add_correct_transport(panel, badge_patch, badge_mask)
    current_panel = replace_evidence_frame(
        current_panel, D_FRAME_ROI, D_FRAME)
    past_panel = replace_evidence_frame(past_panel, E_FRAME_ROI, E_FRAME)
    past_panel = replace_target_empty_frame(past_panel)
    current_panel = quantize_flat_panel(current_panel)
    past_panel = quantize_flat_panel(past_panel)
    panel = quantize_flat_panel(panel)
    image[panel_y:panel_y + 420, 0:420] = current_panel
    image[panel_y:panel_y + 420, 434:854] = past_panel
    image[panel_y:panel_y + 420, panel_x:panel_x + 420] = panel

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.output, optimize=True)
    for pdf_path in args.pdf_output:
        save_pdf(image, pdf_path)

    print(args.output.resolve())
    for pdf_path in args.pdf_output:
        print(pdf_path.resolve())


if __name__ == '__main__':
    main()
