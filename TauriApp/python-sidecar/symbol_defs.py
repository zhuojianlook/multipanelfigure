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


def get_symbol_polys(shape: str, rotation: float = 0) -> dict:
    """
    Returns dict with:
      - 'fill': list of polygons (each = list of (x,y) in [-0.5, 0.5])
      - 'stroke': list of polylines
      - 'filled': True = solid fill, False = outline only
    """
    fill: PolyList = []
    stroke: PolyList = []
    filled = False  # default: outline only (scientific style)

    # ── TIP-BASED symbols: the coordinate (0,0) = the TIP point ──
    # After rotation, the tip points in the rotation direction.

    if shape == "Arrow":
        # Tip at origin, shaft extends backward
        shaft: List[Tuple[float, float]] = [(0, 0), (-0.6, 0)]
        head: List[Tuple[float, float]] = [(0, 0), (-0.22, -0.1), (-0.22, 0.1)]
        stroke = [_rot(shaft, rotation)]
        fill = [_rot(head, rotation)]
        filled = True

    elif shape == "NarrowTriangle":
        # Tip at origin pointing RIGHT (same direction as Arrow at rot=0)
        # Body extends to the left
        pts = [(0, 0), (-1.0, -0.12), (-1.0, 0.12)]
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
                     rotation: float = 0) -> dict:
    """
    Convert normalized symbol to pixel coordinates.
    Returns same structure but with absolute pixel coordinates.
    """
    data = get_symbol_polys(shape, rotation)
    result = {"filled": data["filled"], "fill": [], "stroke": []}
    for poly in data["fill"]:
        result["fill"].append([(cx + x * size, cy + y * size) for x, y in poly])
    for poly in data["stroke"]:
        result["stroke"].append([(cx + x * size, cy + y * size) for x, y in poly])
    return result
