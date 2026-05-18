"""
Shared symbol definitions for consistent rendering across PIL, matplotlib, and SVG.

Designed for scientific image annotation — thin, precise markers.

Each symbol is defined as normalized coordinates in [-0.5, 0.5] centered at origin.
Rendered as thin strokes (outlines) by default, with optional small fills.
"""
import math
from typing import List, Tuple

PolyList = List[List[Tuple[float, float]]]


def _rot(points: List[Tuple[float, float]], angle_deg: float) -> List[Tuple[float, float]]:
    """Rotate points around origin by angle_deg."""
    if abs(angle_deg) < 0.1:
        return points
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in points]


def get_symbol_polys(shape: str, rotation: float = 0, width: float = 1.0) -> dict:
    """
    Returns dict with:
      - 'fill': list of polygons (each = list of (x,y) in [-0.5, 0.5])
      - 'stroke': list of polylines
      - 'filled': True = solid fill, False = outline only

    `width` is a cross-axis thickness multiplier used ONLY by the
    direction-based symbols (Arrow, NarrowTriangle): it scales the shape
    perpendicular to its pointing direction without changing its length.
    All other shapes ignore it.
    """
    fill: PolyList = []
    stroke: PolyList = []
    filled = False  # default: outline only (scientific style)

    # ── TIP-BASED symbols: the coordinate (0,0) = the TIP point ──
    # After rotation, the tip points in the rotation direction.

    if shape == "Arrow":
        # Single filled polygon — a clean arrow with a sharp tip. Tip at
        # origin, head + shaft extend backward (-x). Defining it as ONE
        # filled polygon (rather than a thin STROKED shaft + a filled head)
        # is what makes it render identically in the SVG edit-panel
        # overlay, the matplotlib final figure, and the PIL path: a stroke
        # width has no shared unit across those three (viewBox units vs
        # matplotlib points vs pixels), so the old stroked shaft looked
        # chunkier in the final preview than in the editor. The head is
        # long and narrow (head_len >> head_hw) to give a pointed tip.
        head_len = 0.38                        # head length (unaffected by width)
        head_hw = 0.11 * width                 # head base half-width
        shaft_len = 0.6                        # total length (unaffected by width)
        shaft_hw = 0.03 * width                # shaft half-thickness
        arrow: List[Tuple[float, float]] = [
            (0.0, 0.0),               # tip
            (-head_len, -head_hw),    # head back, upper
            (-head_len, -shaft_hw),   # head -> shaft junction, upper
            (-shaft_len, -shaft_hw),  # shaft tail, upper
            (-shaft_len, shaft_hw),   # shaft tail, lower
            (-head_len, shaft_hw),    # head -> shaft junction, lower
            (-head_len, head_hw),     # head back, lower
        ]
        fill = [_rot(arrow, rotation)]
        filled = True

    elif shape == "NarrowTriangle":
        # Tip at origin pointing RIGHT (same direction as Arrow at rot=0)
        # Body extends to the left. `width` scales the base half-width.
        hw = 0.12 * width
        pts = [(0, 0), (-1.0, -hw), (-1.0, hw)]
        fill = [_rot(pts, rotation)]
        filled = True

    elif shape == "Arrowhead":
        # Tip at origin
        pts = [(0, 0), (-0.35, -0.2), (-0.2, 0), (-0.35, 0.2)]
        fill = [_rot(pts, rotation)]
        filled = True

    # ── CENTER-BASED symbols: the coordinate (0,0) = the CENTER ──

    elif shape == "Triangle":
        pts: List[Tuple[float, float]] = [(0, -0.45), (-0.39, 0.22), (0.39, 0.22)]
        fill = [_rot(pts, rotation)]
        filled = False

    elif shape == "Star":
        pts = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            r = 0.48 if i % 2 == 0 else 0.2
            pts.append((r * math.cos(angle), r * math.sin(angle)))
        fill = [_rot(pts, rotation)]
        filled = False

    elif shape == "Asterisk":
        arms: PolyList = []
        for i in range(6):
            angle = math.radians(i * 30)
            arms.append([(0, 0), (0.45 * math.cos(angle), 0.45 * math.sin(angle))])
        stroke = [_rot(arm, rotation) for arm in arms]

    elif shape == "Rectangle":
        pts = [(-0.4, -0.4), (0.4, -0.4), (0.4, 0.4), (-0.4, 0.4)]
        fill = [_rot(pts, rotation)]
        filled = False

    elif shape == "Ellipse":
        # Circle — 32 points
        pts = []
        for i in range(32):
            angle = math.radians(i * 360 / 32)
            pts.append((0.4 * math.cos(angle), 0.4 * math.sin(angle)))
        fill = [_rot(pts, rotation)]
        filled = False

    elif shape == "Cross":
        h_line: List[Tuple[float, float]] = [(-0.45, 0), (0.45, 0)]
        v_line: List[Tuple[float, float]] = [(0, -0.45), (0, 0.45)]
        stroke = [_rot(h_line, rotation), _rot(v_line, rotation)]

    else:
        # Fallback: small circle
        pts = [(0.3 * math.cos(math.radians(i * 30)),
                0.3 * math.sin(math.radians(i * 30))) for i in range(12)]
        fill = [pts]

    return {"fill": fill, "stroke": stroke, "filled": filled}


def symbol_to_pixels(shape: str, cx: float, cy: float, size: float,
                     rotation: float = 0, width: float = 1.0) -> dict:
    """
    Convert normalized symbol to pixel coordinates.
    Returns same structure but with absolute pixel coordinates.
    `width` is the Arrow / NarrowTriangle cross-axis thickness multiplier.
    """
    data = get_symbol_polys(shape, rotation, width)
    result = {"filled": data["filled"], "fill": [], "stroke": []}
    for poly in data["fill"]:
        result["fill"].append([(cx + x * size, cy + y * size) for x, y in poly])
    for poly in data["stroke"]:
        result["stroke"].append([(cx + x * size, cy + y * size) for x, y in poly])
    return result
