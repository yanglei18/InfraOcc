#!/usr/bin/env python3
"""Draw the code-backed RoadOcc framework as an editable SVG.

The diagram is intentionally generated from repository facts rather than from
an image prompt.  Existing diagnostic panels are embedded as small execution
traces while labels, modules, arrows, and route semantics remain vector-native.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


CANVAS_W = 1800
CANVAS_H = 525

INK = "#1F2937"
MUTED = "#5F6B7A"
LINE = "#A8B3C2"
LIGHT_LINE = "#D7DEE8"
PANEL_BG = "#FFFFFF"
PAGE_BG = "#FFFFFF"

NEUTRAL = "#64748B"
NEUTRAL_FILL = "#F3F6F9"
ENCODER = "#6E628F"
ENCODER_FILL = "#F3F0F8"
DCA = "#2F6F9F"
DCA_FILL = "#EAF3F9"
VVE = "#C7781C"
VVE_FILL = "#FFF3E3"
PERSIST = "#2D7DD2"
PERSIST_FILL = "#EAF3FC"
TRANSPORT = "#E99214"
TRANSPORT_FILL = "#FFF2DD"
REFRESH = "#2EAD65"
REFRESH_FILL = "#E9F8F0"
VDSF = "#27887A"
VDSF_FILL = "#E9F7F4"


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def png_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def load_existing_traces(paper_dir: Path) -> dict[str, Image.Image]:
    """Load camera and model traces without altering their semantic content."""
    with tempfile.TemporaryDirectory(prefix="roadocc-framework-") as tmp:
        tmp_dir = Path(tmp)
        teaser_prefix = tmp_dir / "teaser"
        run_checked([
            "pdftoppm", "-png", "-r", "300", "-singlefile",
            str(paper_dir / "Fig0-ptrteaser.pdf"), str(teaser_prefix)
        ])
        teaser = Image.open(teaser_prefix.with_suffix(".png")).convert("RGB")
        width, height = teaser.size
        # Panel (a) is the upper-left third of the validated teaser.  Exclude
        # the panel title because the framework supplies its own vector label.
        cameras = teaser.crop((0, int(0.055 * height),
                               int(0.333 * width), int(0.49 * height)))
        camera_fov = teaser.crop((int(0.333 * width), int(0.055 * height),
                                  int(0.666 * width), int(0.49 * height)))

        panel_prefix = tmp_dir / "trace"
        execution_trace = paper_dir / "figures" / "roadocc_core_mechanism.pdf"
        run_checked([
            "pdfimages", "-png", str(execution_trace), str(panel_prefix)
        ])
        panels = []
        for path in sorted(tmp_dir.glob("trace-*.png")):
            panels.append(Image.open(path).convert("RGB"))
        if len(panels) != 8:
            raise RuntimeError(
                f"Expected eight diagnostic panels, found {len(panels)}")
        return {
            "cameras": cameras,
            "camera_fov": camera_fov,
            "occupancy": panels[0],
            "dca": panels[2],
            "vve": panels[3],
            "persist": panels[4],
            "transport": panels[5],
            "refresh": panels[6],
            "routed": panels[7],
        }


class Svg:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, value: str) -> None:
        self.parts.append(value)

    def rect(self, x: float, y: float, w: float, h: float, *,
             fill: str = "none", stroke: str = "none", sw: float = 1.5,
             radius: float = 0, dash: str | None = None,
             opacity: float = 1.0, element_id: str | None = None) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        id_attr = f' id="{esc(element_id)}"' if element_id else ""
        self.add(
            f'<rect{id_attr} x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" opacity="{opacity}"{dash_attr}/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, *,
             stroke: str = LINE, sw: float = 2.0, arrow: bool = False,
             dash: str | None = None, opacity: float = 1.0) -> None:
        arrow_attr = ' marker-end="url(#arrow)"' if arrow else ""
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}" '
            f'stroke-linecap="round"{arrow_attr}{dash_attr}/>'
        )

    def path(self, d: str, *, stroke: str = LINE, sw: float = 2.0,
             arrow: bool = False, dash: str | None = None,
             fill: str = "none", opacity: float = 1.0) -> None:
        arrow_attr = ' marker-end="url(#arrow)"' if arrow else ""
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" opacity="{opacity}" '
            f'stroke-linecap="round" stroke-linejoin="round"'
            f'{arrow_attr}{dash_attr}/>'
        )

    def text(self, x: float, y: float, value: str, *, size: float = 21,
             fill: str = INK, weight: str = "normal",
             anchor: str = "middle", italic: bool = False,
             letter_spacing: float = 0, opacity: float = 1.0) -> None:
        style = "italic" if italic else "normal"
        self.add(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="Times New Roman, Times, serif" font-size="{size}" '
            f'font-weight="{weight}" font-style="{style}" fill="{fill}" '
            f'letter-spacing="{letter_spacing}" opacity="{opacity}">'
            f'{esc(value)}</text>'
        )

    def multiline(self, x: float, y: float, lines: list[str], *,
                  size: float = 20, fill: str = INK,
                  weight: str = "normal", anchor: str = "middle",
                  line_height: float = 24, italic: bool = False) -> None:
        style = "italic" if italic else "normal"
        spans = []
        for index, line in enumerate(lines):
            dy = 0 if index == 0 else line_height
            spans.append(
                f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
        self.add(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="Times New Roman, Times, serif" font-size="{size}" '
            f'font-weight="{weight}" font-style="{style}" fill="{fill}">'
            f'{"".join(spans)}</text>'
        )

    def image(self, x: float, y: float, w: float, h: float,
              image: Image.Image, *, radius: float = 5,
              element_id: str | None = None) -> None:
        clip_id = f"clip-{len(self.parts)}"
        id_attr = f' id="{esc(element_id)}"' if element_id else ""
        uri = png_data_uri(image)
        self.add(
            f'<defs><clipPath id="{clip_id}"><rect x="{x}" y="{y}" '
            f'width="{w}" height="{h}" rx="{radius}"/></clipPath></defs>'
            f'<image{id_attr} x="{x}" y="{y}" width="{w}" height="{h}" '
            f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{clip_id})" '
            f'xlink:href="{uri}"/>'
        )

    def circle(self, cx: float, cy: float, r: float, *, fill: str,
               stroke: str = "none", sw: float = 1.5) -> None:
        self.add(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>')


def card(svg: Svg, x: float, y: float, w: float, h: float, *,
         stroke: str, fill: str, title: str, subtitle: str | None = None,
         title_size: float = 25, element_id: str) -> None:
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, sw=2.0, radius=10,
             element_id=element_id)
    svg.rect(x, y, w, 42, fill=stroke, radius=9)
    svg.rect(x, y + 30, w, 12, fill=stroke)
    svg.text(x + w / 2, y + 29, title, size=title_size, fill="#FFFFFF",
             weight="bold")
    if subtitle:
        svg.text(x + w / 2, y + h - 17, subtitle, size=15.5, fill=MUTED)


def question_chip(svg: Svg, x: float, y: float, w: float, label: str,
                  color: str, fill: str) -> None:
    svg.rect(x, y, w, 34, fill=fill, stroke=color, sw=1.4, radius=17)
    svg.text(x + w / 2, y + 23, label, size=15.5, fill=color, weight="bold")


def draw_voxel_stack(svg: Svg, cx: float, cy: float) -> None:
    for index, (dx, dy, color) in enumerate([
            (-22, 14, "#D9D3E8"), (-11, 7, "#C1B7D9"), (0, 0, ENCODER)]):
        x = cx - 38 + dx
        y = cy - 30 + dy
        svg.path(
            f"M {x} {y + 18} L {x + 38} {y} L {x + 76} {y + 18} "
            f"L {x + 38} {y + 36} Z",
            fill=color, stroke="#FFFFFF", sw=1.2, opacity=0.95)
        for offset in (19, 38, 57):
            svg.line(x + offset, y + 9, x + offset + 19, y + 18,
                     stroke="#FFFFFF", sw=0.7, opacity=0.8)
    svg.text(cx, cy + 54, "multi-scale voxels", size=17,
             fill=ENCODER, weight="bold")


def draw_sparse_fusion(svg: Svg, x: float, y: float, w: float) -> None:
    colors = [PERSIST, TRANSPORT, REFRESH]
    starts = [(x + 15, y + 8), (x + 15, y + 41), (x + 15, y + 74)]
    for (sx, sy), color in zip(starts, colors):
        for offset in (0, 12, 24):
            svg.rect(sx + offset, sy, 8, 8, fill=color, radius=1)
        svg.line(sx + 38, sy + 4, x + w * 0.48, y + 45,
                 stroke=color, sw=2.0, arrow=True)
    svg.circle(x + w * 0.55, y + 45, 18, fill=VDSF, stroke="#FFFFFF", sw=2)
    svg.text(x + w * 0.55, y + 51, "Σ", size=23, fill="#FFFFFF",
             weight="bold")
    for row in range(3):
        for col in range(4):
            fill = VDSF if (row + col) % 3 else "#B7DED8"
            svg.rect(x + w * 0.73 + col * 13, y + 22 + row * 13,
                     9, 9, fill=fill, radius=1)
    svg.line(x + w * 0.65, y + 45, x + w * 0.71, y + 45,
             stroke=VDSF, sw=2.2, arrow=True)


def _build_svg_dashboard(traces: dict[str, Image.Image]) -> str:
    svg = Svg()
    svg.add(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{CANVAS_W}" height="{CANVAS_H}" '
        f'viewBox="0 0 {CANVAS_W} {CANVAS_H}">')
    svg.add(
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" '
        'refX="7" refY="4" orient="auto" markerUnits="strokeWidth">'
        f'<path d="M 0 0 L 8 4 L 0 8 z" fill="{LINE}"/>'
        '</marker></defs>')
    svg.rect(0, 0, CANVAS_W, CANVAS_H, fill=PAGE_BG)

    # Coarse-to-fine execution header.
    svg.text(805, 23, "shared decision chain at each decoder scale",
             size=19, fill=MUTED, weight="bold", letter_spacing=0.5)
    for x, label in [(1160, "1/8"), (1212, "1/4"), (1264, "1/2")]:
        svg.rect(x, 7, 43, 24, fill="#F6F8FA", stroke=LIGHT_LINE,
                 sw=1.0, radius=12)
        svg.text(x + 21.5, 24, label, size=16, fill=MUTED, weight="bold")
    svg.line(1309, 19, 1340, 19, stroke=LINE, sw=1.5, arrow=True)
    svg.text(1422, 24, "native P/T/R endpoint", size=17, fill=MUTED,
             weight="bold")

    question_chip(svg, 440, 42, 190, "WHERE is evidence needed?", DCA,
                  DCA_FILL)
    question_chip(svg, 660, 42, 190, "WHERE should history be read?", VVE,
                  VVE_FILL)
    question_chip(svg, 880, 42, 300, "WHICH source is admissible?", ENCODER,
                  ENCODER_FILL)

    y = 88
    h = 254
    card(svg, 24, y, 176, h, stroke=NEUTRAL, fill=NEUTRAL_FILL,
         title="Roadside images", subtitle="current frame  t",
         title_size=23, element_id="roadside-images")
    svg.image(36, y + 55, 152, 164, traces["cameras"], radius=5,
              element_id="camera-trace")

    card(svg, 230, y, 180, h, stroke=ENCODER, fill=ENCODER_FILL,
         title="Lift & encode", subtitle="voxel feature  Qₜˢ",
         title_size=24, element_id="lift-encode")
    draw_voxel_stack(svg, 320, y + 132)
    for index, label in enumerate(("1/8", "1/4", "1/2")):
        svg.rect(253 + index * 48, y + 188, 40, 25, fill="#FFFFFF",
                 stroke="#C9C1DB", sw=1, radius=12)
        svg.text(273 + index * 48, y + 206, label, size=15,
                 fill=ENCODER, weight="bold")

    card(svg, 440, y, 190, h, stroke=DCA, fill=DCA_FILL,
         title="DCA", subtitle="motion-sensitive image re-query",
         title_size=27, element_id="dca")
    svg.image(456, y + 55, 158, 137, traces["dca"], radius=5,
              element_id="dca-trace")
    svg.rect(468, y + 202, 134, 26, fill="#FFFFFF", stroke="#B9D4E7",
             sw=1.0, radius=13)
    svg.text(535, y + 220, "candidate scores  cₜˢ", size=16, fill=DCA,
             weight="bold")

    card(svg, 660, y, 190, h, stroke=VVE, fill=VVE_FILL,
         title="VVE", subtitle="coarse-to-fine correspondence",
         title_size=27, element_id="vve")
    svg.image(676, y + 55, 158, 137, traces["vve"], radius=5,
              element_id="vve-trace")
    svg.rect(688, y + 202, 134, 26, fill="#FFFFFF", stroke="#E7C69D",
             sw=1.0, radius=13)
    svg.text(755, y + 220, "voxel velocity  v̂ₜˢ", size=16, fill=VVE,
             weight="bold")

    card(svg, 880, y, 300, h, stroke=ENCODER, fill="#FBFAFD",
         title="P/T/R controller", subtitle="soft source-admissibility routing",
         title_size=25, element_id="ptr-controller")
    route_specs = [
        ("P", "Persist", PERSIST, PERSIST_FILL, traces["persist"], "H(x)"),
        ("T", "Transport", TRANSPORT, TRANSPORT_FILL,
         traces["transport"], "H(x − τv̂)"),
        ("R", "Refresh", REFRESH, REFRESH_FILL, traces["refresh"], "Qₜ(x)"),
    ]
    for index, (abbr, name, color, fill, image, source) in enumerate(route_specs):
        rx = 894 + index * 92
        svg.rect(rx, y + 54, 84, 155, fill=fill, stroke=color, sw=1.4,
                 radius=7, element_id=f"route-{name.lower()}")
        svg.rect(rx + 5, y + 59, 22, 22, fill=color, radius=11)
        svg.text(rx + 16, y + 75.5, abbr, size=14, fill="#FFFFFF",
                 weight="bold")
        svg.text(rx + 32, y + 75.5, name, size=13.5, fill=color,
                 weight="bold", anchor="start")
        svg.image(rx + 7, y + 89, 70, 72, image, radius=3,
                  element_id=f"{name.lower()}-trace")
        svg.text(rx + 42, y + 184, source, size=16, fill=INK,
                 italic=True)
        svg.text(rx + 42, y + 203, f"α{abbr}", size=16, fill=color,
                 weight="bold")

    card(svg, 1210, y, 200, h, stroke=VDSF, fill=VDSF_FILL,
         title="VDSF", subtitle="sparse temporal fusion",
         title_size=27, element_id="vdsf")
    draw_sparse_fusion(svg, 1226, y + 70, 168)
    svg.multiline(1310, y + 184,
                  ["routed history + current", "residual sparse fusion"],
                  size=17, fill=VDSF, weight="bold", line_height=21)

    card(svg, 1440, y, 216, h, stroke=INK, fill="#F7F8FA",
         title="Predictions", subtitle="occupancy + velocity",
         title_size=24, element_id="predictions")
    svg.image(1455, y + 55, 186, 144, traces["occupancy"], radius=5,
              element_id="occupancy-output")
    # Small orange arrows make the second output head visible without adding a
    # synthetic quantitative trace.
    for ax, ay, dx, dy in [(1483, y + 213, 24, -9),
                           (1530, y + 220, 28, 0),
                           (1580, y + 215, 23, 10)]:
        svg.line(ax, ay, ax + dx, ay + dy, stroke=VVE, sw=2.4, arrow=True)

    # Main left-to-right execution path.
    cy = y + 127
    for x1, x2 in [(200, 230), (410, 440), (630, 660), (850, 880),
                   (1180, 1210), (1410, 1440)]:
        svg.line(x1 + 2, cy, x2 - 5, cy, stroke=LINE, sw=3.0, arrow=True)

    # Historical-memory lane.  It is deliberately below the execution path so
    # temporal source provenance is visible without crossing main arrows.
    lane_y = 376
    lane_h = 143
    svg.rect(230, lane_y, 1180, lane_h, fill="#FAFBFC", stroke=LIGHT_LINE,
             sw=1.5, radius=11, element_id="temporal-memory-lane")
    svg.text(252, lane_y + 27, "TEMPORAL EVIDENCE PATH", size=17,
             fill=MUTED, weight="bold", anchor="start", letter_spacing=1.0)

    # FIFO stack.
    for index in range(3):
        sx = 269 + index * 17
        sy = lane_y + 58 - index * 7
        svg.rect(sx, sy, 78, 51, fill="#EEF2F6", stroke=NEUTRAL,
                 sw=1.3, radius=4)
        svg.text(sx + 39, sy + 31, f"Hₜ₋{3-index}ˢ", size=17,
                 fill=INK, weight="bold")
    svg.text(310, lane_y + 126, "stage-local FIFO", size=17,
             fill=MUTED, weight="bold")

    svg.line(377, lane_y + 82, 445, lane_y + 82, stroke=LINE, sw=2.6,
             arrow=True)
    svg.rect(451, lane_y + 53, 136, 58, fill="#FFFFFF", stroke=NEUTRAL,
             sw=1.4, radius=8)
    svg.multiline(519, lane_y + 75, ["shared roadside", "voxel grid"],
                  size=17, fill=INK, weight="bold", line_height=20)

    # Same-grid history feeds VVE correlation and the Persist source. VVE's
    # velocity then defines the transported source used by the router.
    svg.path(f"M 587 {lane_y + 82} L 744 {lane_y + 82} L 744 {y + h + 5}",
             stroke=VVE, sw=2.4, arrow=True)
    svg.text(660, lane_y + 72, "same-grid history", size=16, fill=VVE,
             weight="bold")

    svg.rect(785, lane_y + 49, 146, 66, fill=VVE_FILL, stroke=VVE,
             sw=1.4, radius=8)
    svg.multiline(858, lane_y + 74,
                  ["source-consistent", "inverse trajectory"],
                  size=17, fill=VVE, weight="bold", line_height=20)
    svg.path(f"M 755 {y + h} L 755 {lane_y + 40} L 785 {lane_y + 40} "
             f"L 785 {lane_y + 61}", stroke=VVE, sw=2.2, arrow=True)
    svg.text(770, lane_y + 31, "v̂ₜˢ + slot age", size=16, fill=VVE,
             weight="bold", anchor="start")

    svg.line(931, lane_y + 82, 1017, lane_y + 82, stroke=LINE, sw=2.6,
             arrow=True)
    source_rows = [
        (PERSIST, "P", "history at x"),
        (TRANSPORT, "T", "transported history"),
        (REFRESH, "R", "current evidence"),
    ]
    for index, (color, abbr, label) in enumerate(source_rows):
        row_y = lane_y + 57 + index * 25
        svg.circle(1018, row_y - 5, 9, fill=color)
        svg.text(1018, row_y, abbr, size=12.5, fill="#FFFFFF",
                 weight="bold")
        svg.text(1034, row_y, label, size=16, fill=INK, weight="bold",
                 anchor="start")
    svg.path(f"M 1050 {lane_y + 82} L 1050 {y + h + 5}",
             stroke=ENCODER, sw=2.4, arrow=True)

    svg.path(f"M 1310 {y + h} L 1310 {lane_y + 48}",
             stroke=VDSF, sw=2.4, arrow=True)
    svg.rect(1210, lane_y + 49, 171, 67, fill=VDSF_FILL, stroke=VDSF,
             sw=1.4, radius=8)
    svg.multiline(1295.5, lane_y + 74,
                  ["fused feature + VVE prior", "cached for next frame"],
                  size=15.5, fill=VDSF, weight="bold", line_height=19)

    # Training-only supervision.  Dashed arrows keep it visually distinct from
    # inference-time data flow.
    sup_y = 555
    svg.text(250, sup_y + 19, "TRAINING ONLY", size=16, fill=MUTED,
             weight="bold", anchor="start", letter_spacing=1.0)
    supervision = [
        (440, 630, "semantic / candidate targets", DCA),
        (660, 850, "multi-scale velocity targets", VVE),
        (880, 1180, "source-admissibility targets", ENCODER),
    ]
    for x1, x2, label, color in supervision:
        svg.rect(x1, sup_y, x2 - x1, 35, fill="#FFFFFF", stroke=color,
                 sw=1.2, radius=17, dash="5 4")
        svg.text((x1 + x2) / 2, sup_y + 23, label, size=14.5,
                 fill=color, weight="bold")
        svg.line((x1 + x2) / 2, sup_y, (x1 + x2) / 2, lane_y + lane_h + 3,
                 stroke=color, sw=1.3, dash="5 4")

    svg.add("</svg>")
    return "\n".join(svg.parts)


def module_panel(svg: Svg, x: float, y: float, w: float, h: float, *,
                 color: str, fill: str, title: str, question: str,
                 tab_width: float = 82, element_id: str) -> None:
    """Draw a paper-style module panel with a compact coloured section tab."""
    svg.rect(x, y, w, h, fill=fill, stroke=color, sw=1.7, radius=10,
             element_id=element_id)
    svg.rect(x + 10, y - 12, tab_width, 29, fill=color, stroke=color,
             sw=1.0, radius=5)
    svg.text(x + 10 + tab_width / 2, y + 8, title, size=18,
             fill="#FFFFFF", weight="bold")
    svg.text(x + w - 10, y + 8, question, size=17,
             fill=INK, weight="bold", anchor="end", italic=True)


def draw_voxel_cube(svg: Svg, x: float, y: float, size: float,
                    *, front: str, top: str, side: str,
                    stroke: str = "#536171") -> None:
    """Draw a compact isometric voxel feature volume."""
    depth = size * 0.27
    svg.path(
        f"M {x} {y + depth} L {x + size} {y + depth} "
        f"L {x + size} {y + size + depth} L {x} {y + size + depth} Z",
        fill=front, stroke=stroke, sw=1.3)
    svg.path(
        f"M {x} {y + depth} L {x + depth} {y} L {x + size + depth} {y} "
        f"L {x + size} {y + depth} Z",
        fill=top, stroke=stroke, sw=1.3)
    svg.path(
        f"M {x + size} {y + depth} L {x + size + depth} {y} "
        f"L {x + size + depth} {y + size} "
        f"L {x + size} {y + size + depth} Z",
        fill=side, stroke=stroke, sw=1.3)
    for index in range(1, 4):
        frac = index / 4
        svg.line(x + size * frac, y + depth,
                 x + size * frac, y + size + depth,
                 stroke="#FFFFFF", sw=0.7, opacity=0.8)
        svg.line(x, y + depth + size * frac,
                 x + size, y + depth + size * frac,
                 stroke="#FFFFFF", sw=0.7, opacity=0.8)


def build_svg(traces: dict[str, Image.Image]) -> str:
    """Build the compact publication-style RoadOcc framework."""
    width = 1800
    height = 525
    svg = Svg()
    svg.add(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.add(
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" '
        'refX="7" refY="4" orient="auto" markerUnits="strokeWidth">'
        f'<path d="M 0 0 L 8 4 L 0 8 z" fill="{INK}"/>'
        '</marker></defs>')
    svg.rect(0, 0, width, height, fill=PAGE_BG)

    # Input side follows the image-led style of recent occupancy frameworks.
    svg.text(118, 32, "Roadside images", size=24, fill=INK, weight="bold")
    svg.rect(16, 50, 205, 215, fill="#FFFFFF", stroke=LINE, sw=1.5,
             radius=9, element_id="input-images")
    svg.image(28, 64, 181, 176, traces["cameras"], radius=5,
              element_id="camera-trace")
    svg.text(118, 257, "current frame  t", size=17, fill=MUTED)

    # Reuse the paper's calibrated camera-FOV visual to make the projective
    # geometry explicit rather than leaving the lower input region decorative.
    svg.rect(34, 287, 260, 210, fill="#FFFFFF", stroke=LINE, sw=1.4,
             radius=8, element_id="camera-geometry")
    svg.image(46, 299, 236, 168, traces["camera_fov"], radius=4,
              element_id="camera-fov-trace")
    svg.text(164, 488, "camera geometry", size=17, fill=INK,
             weight="bold")
    svg.path("M 164 287 L 164 276 L 278 276 L 278 216",
             stroke=NEUTRAL, sw=1.5, dash="5 4", arrow=True)
    svg.text(222, 269, "calibration + depth", size=14.5, fill=MUTED,
             weight="bold")

    # A compact encoder prism feeds the multi-scale voxel hierarchy.
    svg.line(222, 157, 247, 157, stroke=INK, sw=2.3, arrow=True)
    svg.path("M 250 100 L 306 116 L 306 198 L 250 214 Z",
             fill=ENCODER_FILL, stroke=ENCODER, sw=1.7)
    svg.multiline(278, 144, ["image", "encoder"], size=17, fill=ENCODER,
                  weight="bold", line_height=20)
    svg.line(307, 157, 337, 157, stroke=INK, sw=2.3, arrow=True)

    # RoadOcc decoder boundary mirrors the nested module structure used by the
    # reference papers while keeping the execution order strictly left-to-right.
    outer_x, outer_y, outer_w, outer_h = 330, 34, 1205, 485
    svg.rect(outer_x, outer_y, outer_w, outer_h, fill="#FFFFFF",
             stroke="#64748B", sw=1.6, radius=13,
             element_id="roadocc-decoder")
    svg.text(745, 58, "RoadOcc coarse-to-fine decoder", size=22,
             fill=INK, weight="bold")
    svg.text(1110, 58, "shared across", size=16, fill=MUTED,
             weight="bold")
    for index, label in enumerate(("1/8", "1/4", "1/2")):
        x = 1170 + index * 54
        svg.rect(x, 41, 45, 24, fill=NEUTRAL_FILL, stroke=LIGHT_LINE,
                 sw=1.0, radius=12)
        svg.text(x + 22.5, 58, label, size=15, fill=MUTED, weight="bold")
    svg.line(1331, 53, 1360, 53, stroke=LINE, sw=1.4, arrow=True)
    svg.text(1452, 58, "native route", size=15.5, fill=MUTED,
             weight="bold")

    # Multi-scale current representation.
    draw_voxel_cube(svg, 354, 128, 74, front="#A9C3E3",
                    top="#D7E5F4", side="#7698C2")
    svg.text(405, 236, "current voxel", size=17, fill=INK, weight="bold")
    svg.text(405, 257, "feature  Qₜˢ", size=17, fill=MUTED)
    svg.line(455, 171, 480, 171, stroke=INK, sw=2.3, arrow=True)

    # DCA panel.
    module_panel(svg, 480, 88, 255, 223, color=DCA, fill="#F7FBFE",
                 title="DCA", question="Where needed?",
                 tab_width=70, element_id="dca")
    svg.image(494, 119, 126, 132, traces["dca"], radius=5,
              element_id="dca-trace")
    for index in range(3):
        rx = 643 + index * 8
        ry = 139 - index * 7
        svg.rect(rx, ry, 48, 34, fill="#E4EEF7", stroke=DCA, sw=1.0,
                 radius=3)
        svg.line(rx + 7, ry + 10, rx + 39, ry + 10,
                 stroke="#9CBAD3", sw=1.0)
        svg.line(rx + 7, ry + 21, rx + 31, ry + 21,
                 stroke="#9CBAD3", sw=1.0)
    svg.line(647, 202, 688, 202, stroke=DCA, sw=2.0, arrow=True)
    svg.rect(660, 224, 59, 39, fill=DCA_FILL, stroke=DCA, sw=1.2,
             radius=5)
    for row in range(3):
        for col in range(5):
            if (row + 2 * col) % 4 == 0:
                svg.rect(667 + col * 9, 231 + row * 9, 6, 6,
                         fill=DCA, radius=1)
    svg.text(607, 290, "candidate-guided re-query", size=18,
             fill=DCA, weight="bold")

    svg.line(736, 171, 762, 171, stroke=INK, sw=2.3, arrow=True)

    # VVE panel.
    module_panel(svg, 762, 88, 255, 223, color=VVE, fill="#FFFBF5",
                 title="VVE", question="Where to read?",
                 tab_width=70, element_id="vve")
    svg.image(776, 119, 126, 132, traces["vve"], radius=5,
              element_id="vve-trace")
    grid_x, grid_y, cell = 930, 137, 15
    for row in range(5):
        for col in range(5):
            fill = "#FFFFFF"
            if (row, col) == (3, 1):
                fill = "#C7D4E3"
            if (row, col) == (1, 3):
                fill = TRANSPORT_FILL
            svg.rect(grid_x + col * cell, grid_y + row * cell,
                     cell, cell, fill=fill, stroke="#9AA8B8", sw=0.7)
    svg.path(
        f"M {grid_x + 1.5 * cell} {grid_y + 3.5 * cell} "
        f"L {grid_x + 3.35 * cell} {grid_y + 1.65 * cell}",
        stroke=VVE, sw=2.4, arrow=True)
    svg.text(967, 229, "x − τv̂", size=18, fill=VVE,
             weight="bold", italic=True)
    svg.text(889, 290, "local current–history match", size=18,
             fill=VVE, weight="bold")

    svg.line(1018, 171, 1044, 171, stroke=INK, sw=2.3, arrow=True)

    # VDSF panel. P/T/R occupies the largest area because source admissibility
    # is the central contribution rather than a generic temporal-fusion block.
    module_panel(svg, 1044, 88, 467, 223, color=VDSF, fill="#F7FCFB",
                 title="VDSF", question="Which source?",
                 tab_width=76, element_id="vdsf")
    routes = [
        ("P", "Persist", "H(x)", PERSIST, PERSIST_FILL, traces["persist"]),
        ("T", "Transport", "H(x − τv̂)", TRANSPORT, TRANSPORT_FILL,
         traces["transport"]),
        ("R", "Refresh", "Qₜ(x)", REFRESH, REFRESH_FILL,
         traces["refresh"]),
    ]
    for index, (abbr, name, formula, color, fill, trace) in enumerate(routes):
        rx = 1058 + index * 111
        svg.rect(rx, 117, 101, 159, fill=fill, stroke=color, sw=1.25,
                 radius=7, element_id=f"route-{name.lower()}")
        svg.circle(rx + 15, 134, 10, fill=color)
        svg.text(rx + 15, 139, abbr, size=13, fill="#FFFFFF",
                 weight="bold")
        svg.text(rx + 30, 139, name, size=15.5, fill=INK,
                 weight="bold", anchor="start")
        svg.image(rx + 8, 149, 85, 80, trace, radius=3,
                  element_id=f"{name.lower()}-trace")
        svg.text(rx + 50.5, 249, formula, size=17, fill=INK,
                 italic=True)
        svg.text(rx + 50.5, 269, f"α{abbr}", size=17, fill=color,
                 weight="bold")

    fusion_x = 1409
    for index, color in enumerate((PERSIST, TRANSPORT, REFRESH)):
        sy = 143 + index * 40
        svg.line(1391, sy, fusion_x - 18, 194, stroke=color, sw=2.0,
                 arrow=True)
    svg.circle(fusion_x, 194, 21, fill=VDSF, stroke="#FFFFFF", sw=2)
    svg.text(fusion_x, 201, "Σ", size=25, fill="#FFFFFF", weight="bold")
    svg.line(1432, 194, 1457, 194, stroke=INK, sw=2.0, arrow=True)
    for row in range(4):
        for col in range(4):
            active = (row + col) % 3 != 0
            svg.rect(1461 + col * 10, 175 + row * 10, 7, 7,
                     fill=VDSF if active else "#C8E5E1", radius=1)
    svg.multiline(1421, 248, ["soft route", "sparse fusion"], size=17,
                  fill=VDSF, weight="bold", line_height=19)
    svg.text(1277, 290, "supervised P/T/R source routing", size=18,
             fill=VDSF, weight="bold")

    # Training-only labels echo DeGO's restrained dashed supervision cues.
    training_specs = [
        (607, 328, 142, "candidate targets", DCA),
        (889, 328, 142, "velocity targets", VVE),
        (1277, 328, 178, "P/T/R targets", VDSF),
    ]
    for cx, cy, box_w, label, color in training_specs:
        svg.rect(cx - box_w / 2, cy - 13, box_w, 27, fill="#FFFFFF",
                 stroke=color, sw=1.0, radius=13, dash="5 4")
        svg.text(cx, cy + 5, label, size=15.5, fill=color, weight="bold")
        svg.line(cx, cy - 14, cx, 311, stroke=color, sw=1.2,
                 dash="5 4", arrow=True)
    svg.text(351, 333, "training only", size=14.5, fill=MUTED,
             weight="bold", italic=True, anchor="start")

    # The lower lane shows only the recurrent memory lifecycle. Source
    # admissibility remains in the VDSF panel above, avoiding a second,
    # competing P/T/R explanation in the temporal lane.
    lane_y = 363
    svg.rect(348, lane_y, 1168, 134, fill="#FAFBFC",
             stroke=LIGHT_LINE, sw=1.3, radius=9,
             element_id="temporal-memory-loop")
    svg.text(369, lane_y + 25, "TEMPORAL MEMORY LOOP", size=18,
             fill=MUTED, weight="bold", anchor="start", letter_spacing=0.8)

    # Read the previously fused feature and its cached VVE prior. Fixed
    # roadside inference already shares one physical voxel grid, so coordinate
    # normalization is not presented as a separate model operation.
    for index in range(3):
        sx = 382 + index * 12
        sy = lane_y + 51 - index * 6
        svg.rect(sx, sy, 84, 41, fill="#EEF2F6", stroke=NEUTRAL,
                 sw=1.1, radius=3)
        svg.text(sx + 42, sy + 26, f"slot  t−{3-index}", size=16.5,
                 fill=INK, weight="bold")
    svg.text(430, lane_y + 116, "cached feature + VVE prior", size=17,
             fill=MUTED, weight="bold")
    svg.line(486, lane_y + 72, 518, lane_y + 72, stroke=INK, sw=1.9,
             arrow=True)
    svg.rect(524, lane_y + 45, 332, 55, fill="#FFFFFF", stroke=NEUTRAL,
             sw=1.2, radius=6)
    svg.multiline(690, lane_y + 67,
                  ["same-grid multi-slot memory", "fixed roadside coordinates"],
                  size=17, fill=INK, weight="bold", line_height=19)

    # The nearest slot supports VVE correspondence and every valid slot
    # supplies candidate reads to VDSF.
    svg.line(856, lane_y + 61, 882, lane_y + 61, stroke=VVE, sw=1.8,
             arrow=True)
    svg.rect(888, lane_y + 43, 151, 35, fill=VVE_FILL, stroke=VVE,
             sw=1.0, radius=6)
    svg.text(963.5, lane_y + 66, "nearest slot → VVE", size=16.5,
             fill=VVE, weight="bold")
    svg.line(856, lane_y + 84, 882, lane_y + 84, stroke=VDSF, sw=1.8,
             arrow=True)
    svg.rect(888, lane_y + 82, 174, 35, fill=VDSF_FILL, stroke=VDSF,
             sw=1.0, radius=6)
    svg.text(975, lane_y + 105, "all valid slots → VDSF", size=16.5,
             fill=VDSF, weight="bold")

    # The current VVE prior and VDSF-fused feature are appended together.
    svg.path(f"M 1005 311 L 1005 {lane_y + 30} L 1195 {lane_y + 30} "
             f"L 1195 {lane_y + 47}", stroke=VVE, sw=1.7, arrow=True)
    svg.text(1090, lane_y + 23, "VVE prior  v̂ₜˢ", size=16.5, fill=VVE,
             weight="bold")
    svg.rect(1195, lane_y + 45, 280, 58, fill=VDSF_FILL,
             stroke=VDSF, sw=1.2, radius=6)
    svg.text(1335, lane_y + 66, "append current state", size=17,
             fill=VDSF, weight="bold")
    svg.text(1335, lane_y + 89, "fused Hₜˢ  +  VVE v̂ₜˢ", size=16.5,
             fill=INK, weight="bold")
    svg.path(f"M 1409 312 L 1409 {lane_y + 44}", stroke=VDSF,
             sw=1.8, arrow=True)

    # Closing the loop makes the frame-to-frame state transition explicit.
    svg.path(f"M 1440 {lane_y + 104} L 1440 {lane_y + 122} "
             f"L 430 {lane_y + 122} L 430 {lane_y + 101}",
             stroke=NEUTRAL, sw=1.5, arrow=True)
    svg.rect(775, lane_y + 111, 320, 22, fill="#FAFBFC")
    svg.text(935, lane_y + 127,
             "reuse as history at  t+1", size=16.5,
             fill=MUTED, weight="bold")

    # Output panel remains image-led and visually separate from the method box.
    svg.line(1536, 171, 1562, 171, stroke=INK, sw=2.3, arrow=True)
    svg.text(1680, 32, "Predictions", size=24, fill=INK, weight="bold")
    svg.rect(1565, 50, 219, 447, fill="#FFFFFF", stroke=LINE, sw=1.5,
             radius=9, element_id="predictions")
    svg.image(1578, 70, 193, 183, traces["occupancy"], radius=5,
              element_id="occupancy-output")
    svg.text(1674.5, 276, "semantic occupancy", size=18, fill=INK,
             weight="bold")
    svg.line(1605, 317, 1640, 304, stroke=VVE, sw=2.4, arrow=True)
    svg.line(1650, 320, 1688, 320, stroke=VVE, sw=2.4, arrow=True)
    svg.line(1695, 305, 1730, 319, stroke=VVE, sw=2.4, arrow=True)
    svg.text(1674.5, 351, "voxel velocity", size=18, fill=VVE,
             weight="bold")
    svg.rect(1592, 387, 165, 62, fill=NEUTRAL_FILL, stroke=LIGHT_LINE,
             sw=1.0, radius=7)
    svg.multiline(1674.5, 411, ["full-resolution", "scene state"],
                  size=17, fill=MUTED, weight="bold", line_height=20)

    svg.add("</svg>")
    return "\n".join(svg.parts)


def write_evidence_manifest(path: Path, paper_dir: Path) -> None:
    manifest = {
        "main_claim": (
            "RoadOcc resolves temporal memory through three staged decisions: "
            "where evidence is needed, where history is read, and whether the "
            "retrieved source is admissible before sparse fusion."),
        "drawing_method": "editable SVG with embedded, unmodified diagnostics",
        "style_references": [
            "CVPR2026 DeGO Figure 2",
            "CVPR2026 Gau-Occ Figure 1",
            "CVPR2026 TT-Occ Figure 2",
            "ECCV2024 OccWorld Figure 2",
            "ECCV2026 StreamOcc Figure 2",
            "ICLR2026 PG-Occ Figure 2",
            "arXiv2025 Pragmatic Heterogeneous Collaborative Perception "
            "via Generative Communication Mechanism Figure 2",
        ],
        "style_transfer": (
            "Uses the shared visual grammar of image-led inputs and outputs, "
            "a compact left-to-right overview, nested light module boundaries, "
            "small colored section tabs, thin execution routes, compact dashed "
            "submodule cues, and visually subordinate training paths. No "
            "reference artwork is copied."),
        "asset_sources": [
            str(paper_dir / "Fig0-ptrteaser.pdf"),
            str(paper_dir / "figures" / "roadocc_core_mechanism.pdf"),
        ],
        "code_trace": {
            "coarse_to_fine_order": (
                "projects/STCRoadOcc/mmdet3d_plugin/models/stcroadocc/features.py"
                ":330-542"),
            "dca": (
                "projects/STCRoadOcc/configs/stcroadocc_c_2x4_24e.py:230-300"),
            "vve": (
                "projects/STCRoadOcc/mmdet3d_plugin/models/stcroadocc/features.py"
                ":443-503"),
            "ptr_controller": (
                "projects/STCRoadOcc/mmdet3d_plugin/models/modules/"
                "canonical_ptr_controller.py:11-210"),
            "vdsf": (
                "projects/STCRoadOcc/mmdet3d_plugin/models/modules/"
                "sparse_fusion.py:1855-2040"),
            "temporal_memory_loop": (
                "projects/STCRoadOcc/mmdet3d_plugin/models/modules/"
                "sparse_fusion.py:39-360"),
            "active_config": (
                "projects/STCRoadOcc/configs/stcroadocc_c_2x4_24e.py:149-352"),
        },
        "visual_integrity": (
            "Raster traces are crops of existing model or dataset diagnostics. "
            "No prediction, label, metric, or mechanism was synthesized."),
        "coordinate_frame": (
            "Fixed roadside inference uses a shared voxel grid. Relative BEV "
            "augmentation is normalized in code during training but is not "
            "drawn as a learned framework module."),
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-dir", type=Path,
        default=Path("docs/paper/ICLR2027"),
        help="Directory containing the manuscript and existing figure assets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper_dir = args.paper_dir.resolve()
    figure_dir = paper_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    traces = load_existing_traces(paper_dir)
    svg_path = figure_dir / "roadocc_framework.svg"
    pdf_path = paper_dir / "Fig3-framework.pdf"
    preview_path = figure_dir / "roadocc_framework_preview.png"
    manifest_path = figure_dir / "roadocc_framework_evidence.json"

    svg_path.write_text(build_svg(traces), encoding="utf-8")
    run_checked([
        "rsvg-convert", "--format=pdf", "--output", str(pdf_path),
        str(svg_path)
    ])
    run_checked([
        "rsvg-convert", "--format=png", "--width", str(CANVAS_W * 2),
        "--height", str(CANVAS_H * 2), "--output", str(preview_path),
        str(svg_path)
    ])
    write_evidence_manifest(manifest_path, paper_dir)

    print(f"Editable SVG: {svg_path}")
    print(f"Vector PDF: {pdf_path}")
    print(f"QA preview: {preview_path}")
    print(f"Evidence manifest: {manifest_path}")


if __name__ == "__main__":
    main()
