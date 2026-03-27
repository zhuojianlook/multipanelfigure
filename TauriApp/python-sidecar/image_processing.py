"""
Image processing engine — all pure PIL/numpy, no Qt dependencies.
Handles: cropping, adjustments, labels, scale bars, symbols, lines, areas, zoom insets.
"""
from __future__ import annotations
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageColor, ImageFilter, ImageOps
import numpy as np
import colorsys
import math
import re
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from models import (
    PanelInfo, ScaleBarSettings, LabelSettings, SymbolSettings,
    ZoomInsetSettings, FigureConfig, LineAnnotation, AreaAnnotation,
    compute_line_measurement, compute_area_measurement,
)


# -- Hue shifting -----------------------------------------------------------
def shift_hue(img: Image.Image, hue_shift_degrees: float) -> Image.Image:
    if abs(hue_shift_degrees) < 0.5:
        return img
    rgb = img.convert("RGB")
    arr = np.array(rgb, dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)
    shift = hue_shift_degrees / 360.0
    r, g, b = flat[:, 0], flat[:, 1], flat[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    s = np.where(maxc == 0, 0.0, (maxc - minc) / maxc)
    delta = maxc - minc
    hue = np.zeros_like(r)
    mask = delta > 0
    idx = mask & (maxc == r); hue[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6
    idx = mask & (maxc == g); hue[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2
    idx = mask & (maxc == b); hue[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4
    hue = hue / 6.0
    hue = (hue + shift) % 1.0
    hi = (hue * 6.0).astype(int) % 6
    f = hue * 6.0 - hi
    p = v * (1 - s); q = v * (1 - f * s); t = v * (1 - (1 - f) * s)
    out = np.zeros_like(flat)
    for i_case, (c0, c1, c2) in enumerate([(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]):
        m = hi == i_case
        out[m, 0] = c0[m]; out[m, 1] = c1[m]; out[m, 2] = c2[m]
    out = (out.reshape(h, w, 3) * 255).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(out, "RGB")
    if img.mode == "RGBA":
        result = result.convert("RGBA"); result.putalpha(img.split()[-1])
    return result


# -- Aspect-ratio cropping --------------------------------------------------
def parse_aspect_ratio(ratio_str: str):
    if not ratio_str.strip():
        return None
    try:
        w, h = ratio_str.split(":")
        return float(w) / float(h)
    except Exception:
        return None


def crop_with_aspect(image: Image.Image, aspect: float,
                     offset_x: int, offset_y: int) -> Image.Image:
    if aspect is None or aspect <= 0:
        return image
    iw, ih = image.size
    rw = iw; rh = int(rw / aspect)
    if rh > ih:
        rh = ih; rw = int(rh * aspect)
    max_ox = (iw - rw) // 2; max_oy = (ih - rh) // 2
    ox = max(-max_ox, min(max_ox, offset_x))
    oy = max(-max_oy, min(max_oy, offset_y))
    left = (iw - rw) // 2 + ox; upper = (ih - rh) // 2 + oy
    return image.crop((left, upper, left + rw, upper + rh))


# -- Coloured-text parser ---------------------------------------------------
_COLOR_RE = re.compile(r"::([0-9A-Fa-f]{6})::(.+?)::", re.DOTALL)

def parse_colored_text(text: str, default_color: str = "#000000"):
    segments = []
    last = 0
    for m in _COLOR_RE.finditer(text):
        if m.start() > last:
            segments.append((default_color, text[last:m.start()]))
        segments.append((f"#{m.group(1)}", m.group(2)))
        last = m.end()
    if last < len(text):
        segments.append((default_color, text[last:]))
    return segments if segments else [(default_color, text)]


# -- Drawing helpers ---------------------------------------------------------
def _load_font(font_path: Optional[str], size: int, font_name: Optional[str] = None,
               font_style: Optional[List[str]] = None) -> ImageFont.FreeTypeFont:
    """Load a font with optional bold/italic variant resolution."""
    size = max(8, size)
    styles = font_style or []
    is_bold = "Bold" in styles
    is_italic = "Italic" in styles

    import platform
    if platform.system() == "Windows":
        search_dirs = [
            os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
            os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts"),
        ]
    else:
        search_dirs = [
            "/System/Library/Fonts", "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
            "/System/Library/Fonts/Supplemental",
            "/usr/share/fonts/truetype",
        ]
    # Also search persistent custom fonts dir
    persistent_dir = str(Path.home() / ".multipanelfigure" / "fonts")
    if os.path.isdir(persistent_dir):
        search_dirs.append(persistent_dir)

    # If we have a font_name (e.g. "arial.ttf"), try to find bold/italic variant
    effective_name = font_name or (os.path.basename(font_path) if font_path else None)
    if effective_name and (is_bold or is_italic):
        base = os.path.splitext(effective_name)[0]
        # Common naming patterns for variants
        suffixes = []
        if is_bold and is_italic:
            suffixes = ["Bold Italic", "BoldItalic", "Bold_Italic", "BI", "bi"]
        elif is_bold:
            suffixes = ["Bold", "Bd", "bold", "b"]
        elif is_italic:
            suffixes = ["Italic", "It", "Oblique", "italic", "i"]

        # Windows-specific common font variant filenames
        _win_variants = {
            "arial": {"Bold": "arialbd", "Italic": "ariali", "Bold Italic": "arialbi"},
            "times": {"Bold": "timesbd", "Italic": "timesi", "Bold Italic": "timesbi"},
            "calibri": {"Bold": "calibrib", "Italic": "calibrii", "Bold Italic": "calibriz"},
            "consola": {"Bold": "consolab", "Italic": "consolai", "Bold Italic": "consolaz"},
            "cour": {"Bold": "courbd", "Italic": "couri", "Bold Italic": "courbi"},
            "verdana": {"Bold": "verdanab", "Italic": "verdanai", "Bold Italic": "verdanaz"},
            "tahoma": {"Bold": "tahomabd"},
            "comic": {"Bold": "comicbd", "Italic": "comici", "Bold Italic": "comicz"},
            "georgia": {"Bold": "georgiab", "Italic": "georgiai", "Bold Italic": "georgiaz"},
            "trebuc": {"Bold": "trebucbd", "Italic": "trebucit", "Bold Italic": "trebucbi"},
        }
        style_key = "Bold Italic" if (is_bold and is_italic) else ("Bold" if is_bold else "Italic")
        base_lower = base.lower()
        if base_lower in _win_variants and style_key in _win_variants[base_lower]:
            win_name = _win_variants[base_lower][style_key]
            for ext in [".ttf", ".ttc", ".otf"]:
                for d in search_dirs:
                    candidate = os.path.join(d, f"{win_name}{ext}")
                    if os.path.isfile(candidate):
                        try:
                            return ImageFont.truetype(candidate, size)
                        except Exception:
                            pass

        for sfx in suffixes:
            for ext in [".ttf", ".ttc", ".otf"]:
                variant_name = f"{base} {sfx}{ext}"
                for d in search_dirs:
                    candidate = os.path.join(d, variant_name)
                    if os.path.isfile(candidate):
                        try:
                            return ImageFont.truetype(candidate, size)
                        except Exception:
                            pass
                # Also try without space
                variant_name2 = f"{base}{sfx}{ext}"
                for d in search_dirs:
                    candidate = os.path.join(d, variant_name2)
                    if os.path.isfile(candidate):
                        try:
                            return ImageFont.truetype(candidate, size)
                        except Exception:
                            pass

    # Try exact path
    if font_path and os.path.isfile(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    # Try font by name in system locations (exact match first)
    if effective_name:
        for d in search_dirs:
            candidate = os.path.join(d, effective_name)
            if os.path.isfile(candidate):
                try:
                    return ImageFont.truetype(candidate, size)
                except Exception:
                    # Fallback: read via BytesIO to bypass Windows virtual folder issues
                    try:
                        with open(candidate, "rb") as f:
                            return ImageFont.truetype(io.BytesIO(f.read()), size)
                    except Exception:
                        pass
        # Case-insensitive scan (Windows font folder has varying casing)
        name_lower = effective_name.lower()
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            try:
                for fn in os.listdir(d):
                    if fn.lower() == name_lower:
                        candidate = os.path.join(d, fn)
                        if os.path.isfile(candidate):
                            try:
                                return ImageFont.truetype(candidate, size)
                            except Exception:
                                try:
                                    with open(candidate, "rb") as f:
                                        return ImageFont.truetype(io.BytesIO(f.read()), size)
                                except Exception:
                                    pass
            except (PermissionError, OSError):
                continue
    # Fallback to well-known system fonts
    fallbacks = []
    if platform.system() == "Windows":
        win_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        fallbacks = [
            os.path.join(win_fonts, "arial.ttf"),
            os.path.join(win_fonts, "segoeui.ttf"),
            os.path.join(win_fonts, "calibri.ttf"),
            os.path.join(win_fonts, "tahoma.ttf"),
        ]
    else:
        fallbacks = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSText.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for fallback in fallbacks:
        if os.path.isfile(fallback):
            try:
                return ImageFont.truetype(fallback, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _format_measurement(value: float, unit_label: str) -> str:
    """Format a measurement value with adaptive precision."""
    if value < 0.01:
        return f"{value:.3e} {unit_label}"
    elif value < 1:
        return f"{value:.4g} {unit_label}"
    elif value < 1000:
        return f"{value:.2f} {unit_label}"
    else:
        return f"{value:.6g} {unit_label}"


def draw_scale_bar(image: Image.Image, sb: ScaleBarSettings,
                   font_reference_width: int = 0) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    iw, ih = img.size
    bar_length_px = int(sb.bar_length_microns / max(sb.micron_per_pixel, 1e-9))

    # Calculate position from preset or percentage
    # Constants must match figure_builder._add_panel_scale_bars exactly
    if sb.position_preset and sb.position_preset != "Custom":
        edge = sb.edge_distance / 100.0
        if "Right" in sb.position_preset:
            x = int(iw * (1 - edge)) - bar_length_px
        else:
            x = int(iw * edge)
        if "Bottom" in sb.position_preset:
            y = int(ih * (1 - edge)) - sb.bar_height - 5
        else:
            y = int(ih * edge) + 5
    else:
        x = int(sb.position_x / 100.0 * iw) - bar_length_px // 2
        y = int(sb.position_y / 100.0 * ih)

    x = max(0, min(x, iw - bar_length_px - 1))
    y = max(0, min(y, ih - sb.bar_height - 1))

    draw.rectangle([x, y, x + bar_length_px, y + sb.bar_height], fill=sb.bar_color)
    # font_size pts on a 3-inch (216pt) panel = font_size * iw / 216 pixels
    scaled_font_size = max(8, int(sb.font_size * iw / 216))
    # Try font_path first, then font_name for lookup
    font_path = sb.font_path
    if not font_path or not os.path.isfile(font_path or ""):
        font_path = getattr(sb, 'font_name', None)
    font = _load_font(font_path, scaled_font_size)
    label_color = getattr(sb, 'label_color', sb.bar_color) or sb.bar_color
    # Auto-generate label text from bar length in selected unit
    label_text = sb.label
    if not label_text:
        unit = getattr(sb, 'unit', 'um') or 'um'
        unit_display = {'km': 'km', 'm': 'm', 'nm': 'nm', 'mm': 'mm', 'cm': 'cm', 'pm': 'pm'}.get(unit, '\u00B5m')
        from models import UNIT_TO_MICRONS
        um_per_unit = UNIT_TO_MICRONS.get(unit, 1.0)
        bar_in_unit = sb.bar_length_microns / um_per_unit
        label_text = f"{bar_in_unit:g} {unit_display}"
    # Place label above bar if near bottom, below if near top
    is_bottom = y > ih * 0.5
    if is_bottom:
        text_y = y - 4
        # Get text height for positioning above bar
        try:
            bbox = font.getbbox(label_text)
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_h = scaled_font_size
        text_y = y - text_h - 2
    else:
        text_y = y + sb.bar_height + 4
    # Center text on bar
    try:
        text_w = font.getlength(label_text)
    except Exception:
        text_w = len(label_text) * scaled_font_size * 0.6
    text_x = x + bar_length_px / 2 - text_w / 2 + sb.label_x_offset
    draw.text((int(text_x), int(text_y)), label_text, fill=label_color, font=font)
    return img


def draw_labels(image: Image.Image, labels: List[LabelSettings],
                reference_width: int = 1000, base_font_size: int = 20,
                font_reference_width: int = 0) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # font_size pts on a 3-inch (216pt) panel = font_size * w / 216 pixels
    # This matches matplotlib's absolute pt sizing on the same panel width
    for lbl in labels:
        font_size = max(8, int(lbl.font_size * w / 216))
        # Use font_name (e.g. "arial.ttf") if font_path is not set
        font_ref = lbl.font_path if lbl.font_path and os.path.isfile(lbl.font_path) else lbl.font_name
        font = _load_font(font_ref, font_size)
        px = int(lbl.position_x / 100.0 * w)
        py = int(lbl.position_y / 100.0 * h)
        segments = parse_colored_text(lbl.text, lbl.default_color)
        x_cursor = px
        for color, text_chunk in segments:
            if lbl.rotation != 0:
                txt_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                txt_draw = ImageDraw.Draw(txt_img)
                txt_draw.text((0, 0), text_chunk, fill=color, font=font)
                bbox = txt_img.getbbox()
                if bbox:
                    txt_img = txt_img.crop(bbox)
                    txt_img = txt_img.rotate(lbl.rotation, expand=True, resample=Image.BICUBIC)
                    img.paste(txt_img, (x_cursor, py), txt_img)
                    x_cursor += txt_img.size[0]
            else:
                draw.text((x_cursor, py), text_chunk, fill=color, font=font)
                try:
                    bbox = draw.textbbox((x_cursor, py), text_chunk, font=font)
                    x_cursor = bbox[2]
                except Exception:
                    x_cursor += len(text_chunk) * int(font_size * 0.6)
    return img


def _rotate_point(px, py, cx, cy, angle_deg):
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    dx, dy = px - cx, py - cy
    return cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a


def draw_star(draw, cx, cy, size, color):
    points = []
    for i in range(10):
        angle = math.radians(i * 36 - 90)
        r = size if i % 2 == 0 else size * 0.4
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(points, fill=color)


def draw_symbols(image: Image.Image, symbols: List[SymbolSettings],
                 reference_width: int = 0) -> Image.Image:
    from symbol_defs import symbol_to_pixels
    img = image.copy()
    draw = ImageDraw.Draw(img)
    iw, ih = img.size
    # Use reference_width (original pre-crop width) for consistent scaling
    ref_w = reference_width if reference_width > 0 else iw
    scale = ref_w / 216.0  # Match label scaling: pts * origW/216
    for sym in symbols:
        cx = int(sym.x / 100.0 * iw)
        cy = int(sym.y / 100.0 * ih)
        sz = max(3, int(sym.size * scale))
        color = sym.color
        lw = max(1, sz // 12)

        data = symbol_to_pixels(sym.shape, cx, cy, sz, sym.rotation)
        # Stroke width scales with symbol size for visibility on high-res images
        stroke_w = max(2, min(6, sz // 15))

        # Draw filled polygons
        for poly in data["fill"]:
            int_pts = [(int(px), int(py)) for px, py in poly]
            if len(int_pts) >= 3:
                if data["filled"]:
                    draw.polygon(int_pts, fill=color, outline=color)
                else:
                    # Outline only — draw lines for precise width control
                    for j in range(len(int_pts)):
                        p1 = int_pts[j]
                        p2 = int_pts[(j + 1) % len(int_pts)]
                        draw.line([p1, p2], fill=color, width=stroke_w)

        # Draw stroke lines
        for polyline in data["stroke"]:
            int_pts = [(int(px), int(py)) for px, py in polyline]
            if len(int_pts) >= 2:
                draw.line(int_pts, fill=color, width=stroke_w)

        # Label
        if sym.label_text:
            label_font_px = max(10, int(sym.label_font_size * scale))
            font = _load_font(getattr(sym, 'label_font_path', None), label_font_px)
            label_color = getattr(sym, 'label_color', '#FFFFFF')
            # Use absolute position if set, otherwise auto (near symbol)
            lpos_x = getattr(sym, 'label_position_x', -1)
            lpos_y = getattr(sym, 'label_position_y', -1)
            if lpos_x >= 0 and lpos_y >= 0:
                lx = int(lpos_x / 100.0 * iw)
                ly = int(lpos_y / 100.0 * ih)
            else:
                lx = cx + sz // 2
                ly = cy - sz // 4
            draw.text((lx, ly), sym.label_text, fill=label_color, font=font)
    return img


# -- Line annotation drawing ------------------------------------------------
def _get_dash_pattern(style: str, width: float):
    """Return dash on/off pixel lengths for the given style."""
    if style == "dashed":
        return (int(8 * width), int(4 * width))
    elif style == "dotted":
        return (int(2 * width), int(3 * width))
    elif style == "dash-dot":
        return (int(8 * width), int(3 * width), int(2 * width), int(3 * width))
    return None  # solid


def draw_lines(image: Image.Image, lines: List[LineAnnotation],
               micron_per_pixel: float = 1.0) -> Image.Image:
    """Draw line annotations on the image."""
    if not lines:
        return image
    img = image.copy()
    draw = ImageDraw.Draw(img)
    iw, ih = img.size

    for line in lines:
        if len(line.points) < 2:
            continue

        # Convert percentage coords to pixels
        px_points = [(p[0] / 100.0 * iw, p[1] / 100.0 * ih) for p in line.points]
        color = line.color
        # Font scale uses iw/216 (same as labels/symbols)
        # Line width uses a gentler scale to keep lines thin
        font_scale = iw / 216.0
        width_scale = max(1.0, iw / 1000.0)
        width = max(1, int(line.width * width_scale))

        # Draw line segments
        if line.line_type == "curved" or line.is_curved:
            # Spline approximation using intermediate points
            if len(px_points) >= 3:
                _draw_spline(draw, px_points, color, width, line.dash_style)
            else:
                draw.line(px_points, fill=color, width=width)
        else:
            # Straight or multijointed
            dash = _get_dash_pattern(line.dash_style, line.width)
            if dash:
                for i in range(len(px_points) - 1):
                    _draw_dashed_line(draw, px_points[i], px_points[i+1], color, width, dash)
            else:
                draw.line(px_points, fill=color, width=width)

        # Measurement text
        if line.show_measure:
            text = line.measure_text
            if not text:
                unit = getattr(line, 'measure_unit', 'um')
                # Use adaptive precision formatting
                from models import compute_line_length_pixels, UNIT_TO_MICRONS
                px_len = compute_line_length_pixels(line.points, iw, ih)
                len_um = px_len * micron_per_pixel
                len_in_unit = len_um / UNIT_TO_MICRONS.get(unit, 1.0)
                unit_labels = {"km": "km", "m": "m", "cm": "cm", "mm": "mm", "um": "\u03bcm", "nm": "nm", "pm": "pm"}
                text = _format_measurement(len_in_unit, unit_labels.get(unit, unit))
            scaled_font_size = max(8, int(line.measure_font_size * font_scale))
            line_font_style = getattr(line, 'measure_font_style', []) or []
            line_font_name = getattr(line, 'measure_font_name', 'arial.ttf')
            font = _load_font(getattr(line, 'measure_font_path', None), scaled_font_size,
                            font_name=line_font_name, font_style=line_font_style)
            # Place text at midpoint of line
            mid_idx = len(px_points) // 2
            if mid_idx > 0:
                mx = (px_points[mid_idx-1][0] + px_points[mid_idx][0]) / 2
                my = (px_points[mid_idx-1][1] + px_points[mid_idx][1]) / 2
            else:
                mx, my = px_points[0]
            # Use absolute position if available, otherwise auto (midpoint offset)
            lpos_x = getattr(line, 'measure_position_x', -1)
            lpos_y = getattr(line, 'measure_position_y', -1)
            if lpos_x >= 0 and lpos_y >= 0:
                tx = int(lpos_x / 100.0 * iw)
                ty = int(lpos_y / 100.0 * ih)
            else:
                tx = int(mx) + int(5 * font_scale)
                ty = int(my) - int(15 * font_scale)
            draw.text((tx, ty), text, fill=line.measure_color, font=font)

    return img


def _draw_dashed_line(draw, p1, p2, color, width, pattern):
    """Draw a dashed line between two points."""
    dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
    length = math.sqrt(dx*dx + dy*dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    pos = 0; on = True; pat_idx = 0
    while pos < length:
        seg_len = pattern[pat_idx % len(pattern)]
        end_pos = min(pos + seg_len, length)
        if on:
            x1 = p1[0] + ux * pos; y1 = p1[1] + uy * pos
            x2 = p1[0] + ux * end_pos; y2 = p1[1] + uy * end_pos
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
        pos = end_pos
        on = not on
        pat_idx += 1


def _draw_spline(draw, points, color, width, dash_style="solid"):
    """Draw a smooth curve through points using Catmull-Rom approximation."""
    if len(points) < 2:
        return
    # Generate smooth points
    smooth = []
    n = len(points)
    for i in range(n - 1):
        p0 = points[max(0, i - 1)]
        p1 = points[i]
        p2 = points[min(n - 1, i + 1)]
        p3 = points[min(n - 1, i + 2)]
        for t_step in range(10):
            t = t_step / 10.0
            t2 = t * t; t3 = t2 * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
                        (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                        (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
                        (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                        (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            smooth.append((x, y))
    smooth.append(points[-1])
    draw.line(smooth, fill=color, width=width)


# -- Area annotation drawing ------------------------------------------------
def draw_areas(image: Image.Image, areas: List[AreaAnnotation],
               micron_per_pixel: float = 1.0) -> Image.Image:
    """Draw area annotations with semi-transparent fill."""
    if not areas:
        return image
    img = image.copy().convert("RGBA")
    iw, ih = img.size

    for area in areas:
        if len(area.points) < 2:
            continue

        # Create overlay for transparency
        overlay = Image.new("RGBA", (iw, ih), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)

        # Parse fill color with alpha
        fill_alpha = getattr(area, 'fill_alpha', 0.25)
        try:
            border_rgb = ImageColor.getrgb(area.border_color)
        except Exception:
            border_rgb = (255, 0, 0)
        try:
            fill_hex = area.color[:7] if len(area.color) > 7 else area.color
            fill_rgb = ImageColor.getrgb(fill_hex)
        except Exception:
            fill_rgb = border_rgb
        fill_rgba = fill_rgb + (int(fill_alpha * 255),)

        if area.shape == "Rectangle" and len(area.points) >= 2:
            cx, cy = area.points[0][0] / 100 * iw, area.points[0][1] / 100 * ih
            hw, hh = area.points[1][0] / 100 * iw / 2, area.points[1][1] / 100 * ih / 2
            odraw.rectangle([cx - hw, cy - hh, cx + hw, cy + hh],
                            fill=fill_rgba, outline=border_rgb,
                            width=max(1, int(area.border_width)))

        elif area.shape == "Ellipse" and len(area.points) >= 2:
            cx, cy = area.points[0][0] / 100 * iw, area.points[0][1] / 100 * ih
            hw, hh = area.points[1][0] / 100 * iw / 2, area.points[1][1] / 100 * ih / 2
            odraw.ellipse([cx - hw, cy - hh, cx + hw, cy + hh],
                          fill=fill_rgba, outline=border_rgb,
                          width=max(1, int(area.border_width)))

        elif area.shape == "Triangle" and len(area.points) >= 3:
            pts = [(p[0] / 100 * iw, p[1] / 100 * ih) for p in area.points[:3]]
            odraw.polygon(pts, fill=fill_rgba, outline=border_rgb)

        elif area.shape in ("Custom", "Magic") and len(area.points) >= 3:
            pts = [(p[0] / 100 * iw, p[1] / 100 * ih) for p in area.points]
            odraw.polygon(pts, fill=fill_rgba, outline=border_rgb)

        img = Image.alpha_composite(img, overlay)

        # Measurement text
        if area.show_measure:
            draw = ImageDraw.Draw(img)
            unit = getattr(area, 'measure_unit', 'um')
            text = area.measure_text
            if not text:
                # Use adaptive precision formatting
                from models import compute_area_pixels, UNIT_TO_MICRONS
                px_area = compute_area_pixels(area.points, area.shape, iw, ih)
                area_um2 = px_area * (micron_per_pixel ** 2)
                um_per_unit = UNIT_TO_MICRONS.get(unit, 1.0)
                area_in_unit = area_um2 / (um_per_unit ** 2)
                unit_labels = {"km": "km\u00B2", "m": "m\u00B2", "cm": "cm\u00B2", "mm": "mm\u00B2",
                              "um": "\u03bcm\u00B2", "nm": "nm\u00B2", "pm": "pm\u00B2"}
                text = _format_measurement(area_in_unit, unit_labels.get(unit, unit + "\u00B2"))
            scale = iw / 216 if iw > 0 else 1
            scaled_font_size = max(8, int(area.measure_font_size * scale))
            area_font_style = getattr(area, 'measure_font_style', []) or []
            area_font_name = getattr(area, 'measure_font_name', 'arial.ttf')
            font = _load_font(getattr(area, 'measure_font_path', None), scaled_font_size,
                            font_name=area_font_name, font_style=area_font_style)
            # Use custom position if set, otherwise centroid
            mpos_x = getattr(area, 'measure_position_x', -1)
            mpos_y = getattr(area, 'measure_position_y', -1)
            if mpos_x >= 0 and mpos_y >= 0:
                cx = mpos_x / 100 * iw
                cy = mpos_y / 100 * ih
            elif area.shape in ("Custom", "Magic") and len(area.points) >= 3:
                cx = sum(p[0] for p in area.points) / len(area.points) / 100 * iw
                cy = sum(p[1] for p in area.points) / len(area.points) / 100 * ih
            elif len(area.points) >= 1:
                cx = area.points[0][0] / 100 * iw
                cy = area.points[0][1] / 100 * ih
            else:
                cx, cy = iw / 2, ih / 2
            try:
                bbox = font.getbbox(text)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx -= tw / 2; cy -= th / 2
            except Exception:
                pass
            draw.text((int(cx), int(cy)), text, fill=area.measure_color, font=font)

    return img.convert("RGB")


# -- Zoom-inset drawing -----------------------------------------------------
def _line_cross(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(p1,q1,q2) != ccw(p2,q1,q2) and ccw(p1,p2,q1) != ccw(p1,p2,q2)


def _draw_connecting_lines(draw, src_rect, dst_rect, color, width):
    """Draw funnel-shaped connecting lines between two rectangles.
    Picks the 2 source corners closest to the inset center and
    the 2 inset corners closest to the source center, then connects
    them without crossing."""
    x1, y1, x2, y2 = src_rect
    tx1, ty1, tx2, ty2 = dst_rect
    src_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    dst_corners = [(tx1, ty1), (tx2, ty1), (tx2, ty2), (tx1, ty2)]
    # Centers
    src_cx, src_cy = (x1 + x2) / 2, (y1 + y2) / 2
    dst_cx, dst_cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    # 2 source corners closest to inset center
    src_by_dist = sorted(range(4), key=lambda i: math.dist(src_corners[i], (dst_cx, dst_cy)))
    s1, s2 = src_corners[src_by_dist[0]], src_corners[src_by_dist[1]]
    # 2 inset corners closest to source center
    dst_by_dist = sorted(range(4), key=lambda i: math.dist(dst_corners[i], (src_cx, src_cy)))
    d1, d2 = dst_corners[dst_by_dist[0]], dst_corners[dst_by_dist[1]]
    # Try both pairings, pick non-crossing
    if _line_cross(s1, d1, s2, d2):
        d1, d2 = d2, d1  # swap
    draw.line([s1, d1], fill=color, width=width)
    draw.line([s2, d2], fill=color, width=width)


def _draw_adjacent_panel_source(img: Image.Image, zi) -> Image.Image:
    """Draw source rectangle and connecting lines to panel edge for Adjacent Panel zoom."""
    draw = ImageDraw.Draw(img)
    iw, ih = img.size
    x, y, w, h = zi.x, zi.y, zi.width, zi.height
    rect_color = zi.rectangle_color
    rect_width = max(1, zi.rectangle_width)
    line_color = zi.line_color
    line_width = max(1, zi.line_width)

    # Draw source rectangle
    draw.rectangle([x, y, x + w, y + h], outline=rect_color, width=rect_width)

    # Draw simple parallel lines from source rect to panel edge
    side = zi.side or "Right"
    if side == "Right":
        draw.line([(x + w, y), (iw, y)], fill=line_color, width=line_width)
        draw.line([(x + w, y + h), (iw, y + h)], fill=line_color, width=line_width)
    elif side == "Left":
        draw.line([(x, y), (0, y)], fill=line_color, width=line_width)
        draw.line([(x, y + h), (0, y + h)], fill=line_color, width=line_width)
    elif side == "Top":
        draw.line([(x, y), (x, 0)], fill=line_color, width=line_width)
        draw.line([(x + w, y), (x + w, 0)], fill=line_color, width=line_width)
    else:  # Bottom
        draw.line([(x, y + h), (x, ih)], fill=line_color, width=line_width)
        draw.line([(x + w, y + h), (x + w, ih)], fill=line_color, width=line_width)
    return img


def draw_zoom_inset(image: Image.Image, panel: PanelInfo,
                    images_dict: Dict[str, Image.Image]) -> Image.Image:
    zi = panel.zoom_inset
    if zi is None:
        return image
    img = image.copy()
    if zi.inset_type == "Standard Zoom":
        return _draw_standard_zoom(img, zi, panel, images_dict)
    elif zi.inset_type == "Separate Image":
        return _draw_separate_zoom(img, zi, images_dict, panel)
    elif zi.inset_type == "Adjacent Panel":
        # Source rectangle — scale zi coords from full-res to processed image size
        iw, ih = img.size
        # Get full-res dimensions (before thumbnailing)
        if panel and panel.crop_image and panel.crop and len(panel.crop) == 4:
            full_w = panel.crop[2] - panel.crop[0]
            full_h = panel.crop[3] - panel.crop[1]
        else:
            full_w, full_h = iw, ih  # no crop, assume processed = full
        sx = zi.x * iw / max(full_w, 1)
        sy = zi.y * ih / max(full_h, 1)
        sw = zi.width * iw / max(full_w, 1)
        sh = zi.height * ih / max(full_h, 1)
        draw = ImageDraw.Draw(img)
        # Auto-contrast: if rect color is white on likely white bg, use black
        src_outline = zi.rectangle_color or "#FF0000"
        if src_outline.lower() in ("#ffffff", "white", "#fff"):
            src_outline = "#000000"
        draw.rectangle([int(sx), int(sy), int(sx + sw), int(sy + sh)],
                       outline=src_outline, width=max(1, zi.rectangle_width))
    return img


def _draw_standard_zoom(img: Image.Image, zi, panel=None, images_dict=None) -> Image.Image:
    draw = ImageDraw.Draw(img)
    iw, ih = img.size
    x = max(0, min(zi.x, iw - 1))
    y = max(0, min(zi.y, ih - 1))
    w = min(zi.width, iw - x)
    h = min(zi.height, ih - y)
    if w < 2 or h < 2:
        return img

    # Check for external image
    ext_name = getattr(zi, 'separate_image_name', '') or ''
    if ext_name and ext_name not in ('', 'select') and images_dict and ext_name in images_dict:
        ext_img = images_dict[ext_name].convert("RGB")
        xi = getattr(zi, 'x_inset', 0) or 0
        yi = getattr(zi, 'y_inset', 0) or 0
        wi = getattr(zi, 'width_inset', ext_img.size[0]) or ext_img.size[0]
        hi = getattr(zi, 'height_inset', ext_img.size[1]) or ext_img.size[1]
        xi = max(0, min(xi, ext_img.size[0]-1))
        yi = max(0, min(yi, ext_img.size[1]-1))
        wi = max(1, min(wi, ext_img.size[0]-xi))
        hi = max(1, min(hi, ext_img.size[1]-yi))
        region = ext_img.crop((xi, yi, xi+wi, yi+hi))
        # For Standard Zoom with external image: use external crop at natural size
        # scaled to fit the target area (w * zoom_factor for display size)
        zw = int(w * zi.zoom_factor) if zi.zoom_factor > 1 else wi
        zh = int(zw * hi / max(wi, 1))  # maintain external crop aspect ratio
        zw = max(1, zw); zh = max(1, zh)
        region = region.resize((zw, zh), Image.LANCZOS)
    else:
        region = img.crop((x, y, x + w, y + h))
        zw, zh = int(w * zi.zoom_factor), int(h * zi.zoom_factor)
    region = region.resize((zw, zh), Image.LANCZOS)

    # Scale bar in zoom (adjusted for zoom factor)
    if zi.show_scale_bar_in_zoom and panel and panel.scale_bar:
        zoomed_sb = ScaleBarSettings(
            micron_per_pixel=panel.scale_bar.micron_per_pixel / zi.zoom_factor,
            bar_length_microns=panel.scale_bar.bar_length_microns,
            bar_height=panel.scale_bar.bar_height,
            bar_color=panel.scale_bar.bar_color,
            label=panel.scale_bar.label,
            font_size=panel.scale_bar.font_size,
            font_name=panel.scale_bar.font_name,
            font_path=panel.scale_bar.font_path,
            position_preset="Bottom-Right",
            edge_distance=5.0,
        )
        region = draw_scale_bar(region, zoomed_sb)
    elif zi.scale_bar:
        region = draw_scale_bar(region, zi.scale_bar)

    # Source rectangle
    draw.rectangle([x, y, x + w, y + h], outline=zi.rectangle_color, width=zi.rectangle_width)

    # Paste zoomed region
    tx = max(0, min(zi.target_x, iw - zw))
    ty = max(0, min(zi.target_y, ih - zh))
    img.paste(region, (tx, ty))

    # Destination rectangle
    draw = ImageDraw.Draw(img)
    draw.rectangle([tx, ty, tx + zw, ty + zh], outline=zi.rectangle_color, width=zi.rectangle_width)

    # Connecting lines
    _draw_connecting_lines(draw, (x, y, x + w, y + h), (tx, ty, tx + zw, ty + zh),
                           zi.line_color, zi.line_width)
    return img


def _draw_separate_zoom(img: Image.Image, zi, images_dict, panel=None) -> Image.Image:
    draw = ImageDraw.Draw(img)
    if zi.separate_image_name not in images_dict:
        return img
    inset_img = images_dict[zi.separate_image_name].copy()

    draw.rectangle([zi.x_main, zi.y_main,
                    zi.x_main + zi.width_main, zi.y_main + zi.height_main],
                   outline=zi.rectangle_color, width=zi.rectangle_width)

    region = inset_img.crop((zi.x_inset, zi.y_inset,
                             zi.x_inset + zi.width_inset, zi.y_inset + zi.height_inset))

    # Separate image has its own scale bar (not inherited)
    if zi.scale_bar:
        region = draw_scale_bar(region, zi.scale_bar)

    tx, ty = zi.target_x, zi.target_y
    img.paste(region, (tx, ty))
    draw = ImageDraw.Draw(img)
    draw.rectangle([tx, ty, tx + zi.width_inset, ty + zi.height_inset],
                   outline=zi.rectangle_color, width=zi.rectangle_width)

    _draw_connecting_lines(draw,
        (zi.x_main, zi.y_main, zi.x_main + zi.width_main, zi.y_main + zi.height_main),
        (tx, ty, tx + zi.width_inset, ty + zi.height_inset),
        zi.line_color, zi.line_width)
    return img


# -- Image adjustment helpers -----------------------------------------------
def _apply_gamma(img, gamma):
    if abs(gamma - 1.0) < 0.01: return img
    inv_gamma = 1.0 / max(0.01, gamma)
    lut = [int(((i / 255.0) ** inv_gamma) * 255) for i in range(256)]
    channels = img.split()
    if len(channels) >= 3:
        channels = list(channels)
        for i in range(3): channels[i] = channels[i].point(lut)
        return Image.merge(img.mode, tuple(channels))
    return img.point(lut)

def _apply_color_temperature(img, temp):
    if abs(temp) < 0.5: return img
    arr = np.array(img, dtype=np.float32)
    f = temp / 100.0
    arr[:,:,0] = np.clip(arr[:,:,0] + f * 30, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2] - f * 30, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), img.mode)

def _apply_tint(img, tint_val):
    if abs(tint_val) < 0.5: return img
    arr = np.array(img, dtype=np.float32)
    arr[:,:,1] = np.clip(arr[:,:,1] - tint_val / 100.0 * 30, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), img.mode)

def _apply_exposure(img, ev):
    if abs(ev) < 0.01: return img
    arr = np.array(img, dtype=np.float32)
    arr[:,:,:3] = np.clip(arr[:,:,:3] * (2.0 ** ev), 0, 255)
    return Image.fromarray(arr.astype(np.uint8), img.mode)

def _apply_vibrance(img, amount):
    if abs(amount) < 0.5: return img
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    maxc = arr.max(axis=2); minc = arr.min(axis=2)
    sat = np.where(maxc > 0, (maxc - minc) / (maxc + 1e-6), 0)
    weight = (1.0 - sat) * (amount / 100.0)
    gray = arr.mean(axis=2, keepdims=True)
    arr = arr + (arr - gray) * weight[:,:,np.newaxis]
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    result = Image.fromarray(arr, "RGB")
    if img.mode == "RGBA":
        result = result.convert("RGBA"); result.putalpha(img.split()[-1])
    return result

def _apply_highlights_shadows(img, highlights, shadows, midtones=0.0):
    if abs(highlights) < 0.5 and abs(shadows) < 0.5 and abs(midtones) < 0.5: return img
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    lum = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    if abs(highlights) >= 0.5:
        mask = np.clip((lum - 128) / 127.0, 0, 1)
        for c in range(3): arr[:,:,c] += mask * highlights / 100.0 * 50
    if abs(shadows) >= 0.5:
        mask = np.clip((128 - lum) / 128.0, 0, 1)
        for c in range(3): arr[:,:,c] += mask * shadows / 100.0 * 50
    if abs(midtones) >= 0.5:
        mask = 1.0 - np.abs(lum - 128.0) / 128.0
        for c in range(3): arr[:,:,c] += np.clip(mask, 0, 1) * midtones / 100.0 * 50
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    result = Image.fromarray(arr, "RGB")
    if img.mode == "RGBA":
        result = result.convert("RGBA"); result.putalpha(img.split()[-1])
    return result

def _apply_denoise(img, strength):
    if strength < 0.01: return img
    return img.filter(ImageFilter.GaussianBlur(radius=max(1, int(strength * 3))))


# -- Full panel processing pipeline -----------------------------------------
def process_panel(image: Image.Image, panel: PanelInfo,
                  min_dims: Tuple[int, int],
                  images_dict: Dict[str, Image.Image],
                  skip_labels: bool = False,
                  skip_symbols: bool = False) -> Image.Image:
    img = image.copy()
    # Save original dimensions for font scaling — labels/scale bars/symbols
    # should maintain the same visual size regardless of crop
    orig_w = image.size[0]
    # Store on panel for figure_builder to access
    panel._orig_w = orig_w

    # 0) Rotation
    rot = getattr(panel, 'rotation', 0) or 0
    if isinstance(rot, (int, float)) and abs(rot) > 0.1:
        rot_int = int(rot) % 360
        if rot_int == 90: img = img.transpose(Image.ROTATE_270)
        elif rot_int == 180: img = img.transpose(Image.ROTATE_180)
        elif rot_int == 270: img = img.transpose(Image.ROTATE_90)
        elif rot_int != 0:
            # Rotate without warping aspect ratio
            img = img.rotate(-rot, expand=True, resample=Image.BICUBIC,
                             fillcolor=(0, 0, 0) if img.mode == "RGB" else (0, 0, 0, 0))

    # 0b) Flip
    if getattr(panel, 'flip_horizontal', False):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if getattr(panel, 'flip_vertical', False):
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # 0c) Direct pixel crop
    _direct_crop = False
    if panel.crop is not None:
        try:
            left, top, right, bottom = panel.crop
            iw, ih = img.size
            left = max(0, min(int(left), iw - 1)); top = max(0, min(int(top), ih - 1))
            right = max(left + 1, min(int(right), iw)); bottom = max(top + 1, min(int(bottom), ih))
            if right > left + 1 and bottom > top + 1:
                img = img.crop((left, top, right, bottom)); _direct_crop = True
        except (TypeError, ValueError):
            pass

    # 0d) Per-channel levels
    def _make_lut(ib, iw_):
        ib = max(0, min(254, ib)); iw_ = max(ib + 1, min(255, iw_))
        return [int(max(0, min(255, (max(ib, min(iw_, v)) - ib) / (iw_ - ib) * 255))) for v in range(256)]

    ibr = int(getattr(panel, 'input_black_r', 0) or 0)
    iwr = int(getattr(panel, 'input_white_r', 255) or 255)
    ibg = int(getattr(panel, 'input_black_g', 0) or 0)
    iwg = int(getattr(panel, 'input_white_g', 255) or 255)
    ibb = int(getattr(panel, 'input_black_b', 0) or 0)
    iwb = int(getattr(panel, 'input_white_b', 255) or 255)
    if ibr > 0 or iwr < 255 or ibg > 0 or iwg < 255 or ibb > 0 or iwb < 255:
        channels = img.split()
        if len(channels) >= 3:
            mapped = [channels[0].point(_make_lut(ibr, iwr)),
                      channels[1].point(_make_lut(ibg, iwg)),
                      channels[2].point(_make_lut(ibb, iwb))]
            if len(channels) == 4: mapped.append(channels[3])
            img = Image.merge(img.mode, mapped)

    # 1) Tone: Brightness, Contrast, Exposure, Gamma
    if panel.brightness != 1.0: img = ImageEnhance.Brightness(img).enhance(panel.brightness)
    if panel.contrast != 1.0: img = ImageEnhance.Contrast(img).enhance(panel.contrast)
    if abs(getattr(panel, 'exposure', 0) or 0) > 0.01: img = _apply_exposure(img, panel.exposure)
    if abs((getattr(panel, 'gamma', 1) or 1) - 1.0) > 0.01: img = _apply_gamma(img, panel.gamma)

    # 2) Color: Hue, Saturation, Vibrance, Temperature, Tint
    if abs(panel.hue) > 0.5: img = shift_hue(img, panel.hue)
    if abs((getattr(panel, 'saturation', 1) or 1) - 1.0) > 0.01:
        img = ImageEnhance.Color(img).enhance(panel.saturation)
    if abs(getattr(panel, 'vibrance', 0) or 0) > 0.5: img = _apply_vibrance(img, panel.vibrance)
    if abs(getattr(panel, 'color_temperature', 0) or 0) > 0.5:
        img = _apply_color_temperature(img, panel.color_temperature)
    if abs(getattr(panel, 'tint', 0) or 0) > 0.5: img = _apply_tint(img, panel.tint)

    # 3) Detail
    if (getattr(panel, 'sharpen', 0) or 0) > 0.01:
        img = ImageEnhance.Sharpness(img).enhance(1.0 + panel.sharpen)
    if (getattr(panel, 'blur', 0) or 0) > 0.01:
        img = img.filter(ImageFilter.GaussianBlur(radius=panel.blur))
    if (getattr(panel, 'denoise', 0) or 0) > 0.01: img = _apply_denoise(img, panel.denoise)

    # 4) Tone curve
    h_ = getattr(panel, 'highlights', 0) or 0
    s_ = getattr(panel, 'shadows', 0) or 0
    m_ = getattr(panel, 'midtones', 0) or 0
    if abs(h_) > 0.5 or abs(s_) > 0.5 or abs(m_) > 0.5:
        img = _apply_highlights_shadows(img, h_, s_, m_)

    # 5) Effects
    if getattr(panel, 'grayscale', False): img = ImageOps.grayscale(img).convert("RGB")
    if getattr(panel, 'invert', False):
        if img.mode == "RGBA":
            r, g, b, a = img.split()
            img = ImageOps.invert(Image.merge("RGB", (r, g, b))).convert("RGBA"); img.putalpha(a)
        else:
            img = ImageOps.invert(img.convert("RGB"))

    # 6) Crop
    if not _direct_crop and panel.crop_image:
        aspect = parse_aspect_ratio(panel.aspect_ratio_str)
        if aspect:
            img = crop_with_aspect(img, aspect, panel.crop_offset_x, panel.crop_offset_y)
        elif panel.crop:
            # Explicit crop region set by user (x0, y0, x1, y1)
            cr = panel.crop
            img = img.crop((cr[0], cr[1], cr[2], cr[3]))
        # Note: when crop_image is True but no aspect_ratio_str and no
        # explicit crop region, we do NOT auto-crop to min_dims.
        # Images keep their native dimensions; the figure builder
        # handles mixed sizes via padding/centering.

    # 7) Final resize
    if panel.final_resize:
        img = img.resize((panel.final_width, panel.final_height), Image.LANCZOS)

    # 8) Scale bar — extract micron_per_pixel for measurement calculations.
    # In the full figure pipeline, scale bar rendering is done via matplotlib.
    # When skip_labels is False (rendered preview mode), also draw via PIL.
    micron_per_pixel = 1.0
    if panel.add_scale_bar and panel.scale_bar:
        micron_per_pixel = panel.scale_bar.micron_per_pixel
        if not skip_labels:  # rendered preview mode — draw scale bar via PIL
            img = draw_scale_bar(img, panel.scale_bar, font_reference_width=orig_w)

    # 9) Lines
    if panel.lines:
        img = draw_lines(img, panel.lines, micron_per_pixel)

    # 10) Areas
    if panel.areas:
        img = draw_areas(img, panel.areas, micron_per_pixel)

    # 11) Labels
    if panel.labels and not skip_labels:
        img = draw_labels(img, panel.labels, font_reference_width=orig_w)

    # 12) Symbols
    if panel.symbols and not skip_symbols:
        img = draw_symbols(img, panel.symbols, reference_width=orig_w)

    # 13) Zoom inset
    if panel.add_zoom_inset and panel.zoom_inset:
        img = draw_zoom_inset(img, panel, images_dict)

    return img.convert("RGB")
