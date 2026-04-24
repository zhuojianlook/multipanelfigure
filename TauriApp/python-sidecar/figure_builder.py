"""
Figure assembly engine — takes a FigureConfig + loaded images and produces
a final composite image using matplotlib (matching the Streamlit version output).
"""
from __future__ import annotations
import io
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image, ImageDraw

from models import (
    FigureConfig, PanelInfo, AxisLabel, HeaderLevel, HeaderGroup,
    ScaleBarSettings, StyledSegment, SymbolSettings,
)
from image_processing import parse_colored_text


def _get_segments(obj):
    """Get segment list from a header/label object.
    Prefers styled_segments if populated, falls back to parse_colored_text markup.
    Returns list of dicts with keys: color, text, font_name, font_size, font_style."""
    if hasattr(obj, 'styled_segments') and obj.styled_segments:
        result = []
        for seg in obj.styled_segments:
            result.append({
                'color': seg.color,
                'text': seg.text,
                'font_name': getattr(seg, 'font_name', None),
                'font_size': getattr(seg, 'font_size', None),
                'font_style': getattr(seg, 'font_style', None),
            })
        return result
    # Fallback to colored text markup — return as dicts for uniform interface
    return [{'color': c, 'text': t, 'font_name': None, 'font_size': None, 'font_style': None}
            for c, t in parse_colored_text(obj.text, obj.default_color)]


def _count_header_lines(hdr):
    """Return the number of rendered lines for a header/label, based on explicit
    newlines in its text or in its styled_segments. Used to reserve enough header
    band space so multi-line headers don't clip off the figure."""
    segs = None
    if hasattr(hdr, 'styled_segments') and hdr.styled_segments:
        segs = hdr.styled_segments
    if segs:
        joined = "".join((getattr(s, 'text', None) or '') for s in segs)
        return max(1, joined.count("\n") + 1)
    text = getattr(hdr, 'text', '') or ''
    return max(1, text.count("\n") + 1)


def _draw_colored_text(fig, x, y, segments, fp, ha="center", va="bottom",
                       rotation=0):
    """Draw multi-colored/styled text segments at a position.
    Segments can be dicts with keys: color, text, font_name, font_size, font_style
    or tuples (color, text) for backward compatibility.
    Supports: Bold, Italic, Underline, Strikethrough, Superscript, Subscript."""
    if not segments:
        return

    # Normalize segments to dict format
    norm_segs = []
    for seg in segments:
        if isinstance(seg, dict):
            norm_segs.append(seg)
        elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
            norm_segs.append({'color': seg[0], 'text': seg[1],
                              'font_name': None, 'font_size': None, 'font_style': None})
        else:
            continue
    if not norm_segs:
        return

    # Check if simple case: all same color, no special styles
    colors = set(s['color'] for s in norm_segs)
    has_styles = any(s.get('font_style') for s in norm_segs)
    has_mixed_fonts = any(s.get('font_name') or s.get('font_size') for s in norm_segs)
    full_text = "".join(s['text'] for s in norm_segs)

    if len(colors) <= 1 and not has_styles and not has_mixed_fonts:
        color = norm_segs[0]['color']
        # Check if the base font_style (from fp) has decoration styles
        base_styles = []
        if hasattr(fp, '_slant') and fp.get_style() == 'italic':
            pass  # Already in fp
        # Get base styles from the caller's context - check weight/style
        has_underline = False
        has_strikethrough = False
        is_superscript = False
        is_subscript = False
        # The fp carries Bold/Italic but not Underline/Strikethrough/Super/Sub
        # These need to be checked from the original style list
        # We'll use a marker attribute on fp (set by caller)
        extra_styles = getattr(fp, '_extra_styles', [])
        has_underline = 'Underline' in extra_styles
        has_strikethrough = 'Strikethrough' in extra_styles
        is_superscript = 'Superscript' in extra_styles
        is_subscript = 'Subscript' in extra_styles

        # Adjust for superscript/subscript
        adj_fp = fp
        adj_y = y
        if is_superscript or is_subscript:
            new_size = fp.get_size() * 0.7
            adj_fp = fp.copy()
            adj_fp.set_size(new_size)
            offset = fp.get_size() / 72.0 * 0.3  # 30% of font size in inches
            fig_h = fig.get_size_inches()[1]
            if is_superscript:
                adj_y = y + offset / fig_h
            else:
                adj_y = y - offset / fig_h

        # Use rotation_mode='anchor' so the (x,y) anchor point is the
        # fixed rotation origin. This matches the multi-styled path
        # convention — without this, single-color rotated headers
        # position differently than styled ones at the same cx/cy
        # (user-visible: styled 2-line row headers drifted up relative
        # to plain siblings in the same tier).
        txt = fig.text(x, adj_y, full_text, ha=ha, va=va, fontproperties=adj_fp,
                       color=color, rotation=rotation,
                       rotation_mode='anchor',
                       multialignment=ha if ha in ("left", "center", "right") else "center")

        # Draw underline/strikethrough manually after rendering
        if has_underline or has_strikethrough:
            fig.canvas.draw()
            try:
                bb = txt.get_window_extent(fig.canvas.get_renderer())
                bb_fig = bb.transformed(fig.transFigure.inverted())
                lw = max(0.5, fp.get_size() / 20)
                if has_underline:
                    uy = bb_fig.y0 - 0.002
                    fig.add_artist(plt.Line2D(
                        [bb_fig.x0, bb_fig.x1], [uy, uy],
                        transform=fig.transFigure, color=color,
                        linewidth=lw, clip_on=False))
                if has_strikethrough:
                    sy = (bb_fig.y0 + bb_fig.y1) / 2
                    fig.add_artist(plt.Line2D(
                        [bb_fig.x0, bb_fig.x1], [sy, sy],
                        transform=fig.transFigure, color=color,
                        linewidth=lw, clip_on=False))
            except Exception:
                pass  # Fallback: skip decoration if bbox fails
        return

    # Multi-styled rendering — one fig.text per segment, placed along
    # the rotated baseline. Handles:
    #   * Per-character colour / font / size / style via segments.
    #   * Rotation (row headers) — each segment rotated and the anchor
    #     advances along the rotated baseline.
    #   * Explicit newlines — segments containing "\n" are split into
    #     sub-segments on separate lines. Each line is offset
    #     perpendicular to the baseline so the text reads line-by-line
    #     even when rotated.
    #   * ha / va alignment — overall string alignment is respected by
    #     offsetting the per-line starting anchor along / perpendicular
    #     to the baseline.
    import math as _math
    fig_w_in = float(fig.get_size_inches()[0])
    fig_h_in = float(fig.get_size_inches()[1])
    rot_deg = float(rotation) if rotation else 0.0
    rot_rad = _math.radians(rot_deg)
    cos_r = _math.cos(rot_rad)
    sin_r = _math.sin(rot_rad)

    # Renderer used to measure actual rendered text widths — falls back to
    # a conservative char-width estimate when the backend doesn't expose
    # one (e.g. before the first canvas.draw() on some backends).
    try:
        _renderer = fig.canvas.get_renderer()  # Agg / AggBase supports this
    except Exception:
        _renderer = None
    _dpi = float(fig.dpi) if fig.dpi else 100.0

    def _measure_inches(txt: str, seg_fp) -> float:
        """Measure the LAYOUT (advance) width of `txt` in inches using the
        actual font.

        Prefers ``renderer.get_text_width_height_descent`` — this returns the
        typographic advance width (how far the pen moves after rendering),
        which is what matplotlib uses when laying out the next character.
        Using advance width is what makes adjacent segments line up without
        a visible gap at the colour boundary; a get_window_extent()-based
        measurement reports the tight INK bbox which omits each segment's
        leading/trailing side bearing and causes stair-stepped baselines
        when multiple coloured segments are joined (user-reported
        'changing color shifts characters slightly')."""
        if not txt:
            return 0.0
        if _renderer is not None:
            try:
                w_px, _h, _d = _renderer.get_text_width_height_descent(
                    txt, seg_fp, ismath=False)
                return float(w_px) / _dpi
            except Exception:
                pass
            # Fallback: figure.text() + bbox — less accurate but better
            # than the constant-char-width estimate below.
            t = fig.text(0, 0, txt, fontproperties=seg_fp, color="none")
            try:
                bb = t.get_window_extent(_renderer)
                return bb.width / _dpi
            except Exception:
                pass
            finally:
                try:
                    t.remove()
                except Exception:
                    pass
        size_pts = seg_fp.get_size() if hasattr(seg_fp, 'get_size') else 10
        return (size_pts / 72.0) * 0.55 * max(1, len(txt))

    # Helper: build FontProperties + inches-width for a seg.
    def _seg_fp_and_width(seg_text: str, seg: dict):
        seg_styles = seg.get('font_style') or []
        seg_size = seg.get('font_size') or fp.get_size()
        seg_font = seg.get('font_name')
        seg_fp = _font_props(
            fp._fname if hasattr(fp, '_fname') else None,
            seg_font or (fp.get_name() if hasattr(fp, 'get_name') else "arial.ttf"),
            seg_size,
            seg_styles if seg_styles else (list(fp.get_style()) if hasattr(fp, 'get_style') else []),
        )
        size_pts = seg_fp.get_size() if hasattr(seg_fp, 'get_size') else 10
        w_in = _measure_inches(seg_text, seg_fp)
        return seg_fp, w_in, size_pts

    # Split segments at "\n" into a list of LINES, where each line is a
    # list of (text, seg_dict) pairs. Preserves each segment's original
    # styling for every piece after the split.
    lines: list[list[tuple[str, dict]]] = [[]]
    max_size_per_line: list[float] = [10.0]
    for seg in norm_segs:
        txt = seg.get('text') or ''
        parts = txt.split('\n')
        for i, part in enumerate(parts):
            if i > 0:
                lines.append([])
                max_size_per_line.append(10.0)
            if part != '':
                lines[-1].append((part, seg))
                sz = float(seg.get('font_size') or fp.get_size() or 10)
                if sz > max_size_per_line[-1]:
                    max_size_per_line[-1] = sz

    # Line height in inches (slightly generous for breathing room).
    # Use the MAX font size seen overall so all lines have a consistent
    # height — matches matplotlib's default multi-line behaviour.
    overall_max_size = max(max_size_per_line) if max_size_per_line else 10.0
    line_height_in = overall_max_size * 1.2 / 72.0
    num_lines = len(lines)

    # Map the CALLER's va ('top'/'center'/'bottom') to the baseline offset
    # needed when per-segment draws use va='baseline'. Without this, the
    # whole styled text block shifts relative to where the plain-path
    # (non-styled) render would land — user-reported as "changing colour
    # shifts column header downwards" (va='bottom' caller, styled path
    # was drawing at baseline instead of ink-bottom-at-y, so baseline
    # was at y instead of y+descent — text effectively moved down by
    # `descent` relative to the plain equivalent).
    #
    # Probe matplotlib's own bbox for a test text anchored at (0,0) with
    # va='baseline' — bb.y0 is the descent below baseline and bb.y1 is
    # the ascent above. Using fig.text+get_window_extent matches what
    # matplotlib uses for va='top/center/bottom' alignment in the
    # single-color (plain) path, so the styled path lands at the same
    # visual position.
    ref_fp_for_metrics = fp
    ascent_in = 0.0
    descent_in = 0.0
    if _renderer is not None:
        try:
            probe = fig.text(0, 0, "Mg", fontproperties=ref_fp_for_metrics,
                             ha='left', va='baseline', color='none')
            try:
                bb = probe.get_window_extent(_renderer)
                ascent_in = bb.y1 / _dpi
                descent_in = -bb.y0 / _dpi
            finally:
                probe.remove()
        except Exception:
            pass
    if ascent_in <= 0:
        # Fallback estimates from font size (Arial-like 80/20 split).
        size_pts = ref_fp_for_metrics.get_size() if hasattr(ref_fp_for_metrics, 'get_size') else 10
        ascent_in = size_pts * 0.8 / 72.0
        descent_in = size_pts * 0.2 / 72.0

    if va == 'bottom':
        # Caller wants INK-BBOX BOTTOM at y. Plain-path puts baseline at
        # y + descent. Per-seg baseline draws put baseline at y. Shift
        # the block up by +descent (in PERP-UP direction) to match.
        baseline_offset_perp_in = descent_in
    elif va == 'top':
        # Caller wants INK-BBOX TOP at y. Baseline at y - ascent. Shift
        # block down (in PERP-DOWN direction) by ascent.
        baseline_offset_perp_in = -ascent_in
    elif va in ('center', 'center_baseline'):
        # Caller wants INK-BBOX CENTER at y. Center = baseline + (ascent-descent)/2.
        # To put center at y: baseline = y - (ascent-descent)/2.
        baseline_offset_perp_in = -(ascent_in - descent_in) / 2.0
    else:  # 'baseline'
        baseline_offset_perp_in = 0.0

    for line_idx, line_segs in enumerate(lines):
        # Perpendicular-to-baseline offset (positive = "up" relative to
        # baseline direction — i.e. opposite to where the next line
        # should be placed).
        if va == 'top':
            perp_offset_in = -line_idx * line_height_in
        elif va == 'bottom':
            perp_offset_in = (num_lines - 1 - line_idx) * line_height_in
        else:  # center / baseline
            perp_offset_in = ((num_lines - 1) / 2.0 - line_idx) * line_height_in

        # Perpendicular direction in figure coords: (-sin_r, cos_r)
        # — at rotation=0, that's (0, 1) so lines stack vertically.
        # Apply the baseline_offset in the SAME perpendicular direction so
        # the per-seg va='baseline' draws land at the same ink-bbox
        # position the caller's va asked for.
        total_perp_in = perp_offset_in + baseline_offset_perp_in
        line_base_x = x + (-sin_r * total_perp_in) / max(0.001, fig_w_in)
        line_base_y = y + (cos_r * total_perp_in) / max(0.001, fig_h_in)

        # Layout segments within this line along the baseline.
        if not line_segs:
            continue
        # Resolve each seg's FontProperties + color up front.
        seg_fps = []
        seg_colors = []
        seg_texts = []
        for part_text, seg in line_segs:
            seg_fp, _, _ = _seg_fp_and_width(part_text, seg)
            seg_fps.append(seg_fp)
            seg_colors.append(seg.get('color') or '#000000')
            seg_texts.append(part_text)

        # Position each segment using PREFIX widths — measuring a
        # single rendered string "AB" and subtracting width("A") gives
        # the true cursor-advance for "B" (including leading/trailing
        # bearing and kerning at the A|B boundary). Measuring each
        # segment in isolation misses the kerning adjustment and shows
        # up as a per-segment drift / gap — visible in rotated row
        # headers as a stair-stepped baseline (user-reported
        # "changing color shifts characters slightly").
        #
        # We measure prefixes using the FIRST segment's font. When all
        # segments share a font (the common per-colour-only case) this
        # is exactly right. When fonts differ we fall back to summing
        # per-segment measurements — kerning across a font change is
        # rare enough that a bearing-sized gap is acceptable.
        #
        # Compare by actual font parameters (name + size + file), NOT by
        # object identity — every seg gets a fresh FontProperties object
        # from _font_props(), so `is not` would always say "mixed" and
        # we'd lose kerning even when the fonts are effectively identical.
        def _fp_key(fp_x):
            try:
                return (
                    fp_x.get_name() if hasattr(fp_x, 'get_name') else None,
                    fp_x.get_size() if hasattr(fp_x, 'get_size') else None,
                    getattr(fp_x, '_fname', None),
                    fp_x.get_style() if hasattr(fp_x, 'get_style') else None,
                    fp_x.get_weight() if hasattr(fp_x, 'get_weight') else None,
                )
            except Exception:
                return id(fp_x)
        first_key = _fp_key(seg_fps[0]) if seg_fps else None
        mixed_fonts = any(_fp_key(fp_i) != first_key for fp_i in seg_fps[1:])
        seg_starts_in = [0.0]  # along-baseline start for each seg
        if not mixed_fonts and seg_fps:
            full_line_text = "".join(seg_texts)
            # Prefix lengths: 0, len(seg[0]), len(seg[0])+len(seg[1]), ...
            cum = 0
            prefix_widths = [0.0]
            for st in seg_texts:
                cum += len(st)
                prefix_widths.append(_measure_inches(full_line_text[:cum], seg_fps[0]))
            seg_starts_in = prefix_widths[:-1]
            total_line_w = prefix_widths[-1]
        else:
            # Mixed-font fallback: sum per-segment widths.
            total_line_w = 0.0
            for st, fp_i in zip(seg_texts, seg_fps):
                seg_starts_in.append(total_line_w + _measure_inches(st, fp_i))
                total_line_w = seg_starts_in[-1]
            seg_starts_in = seg_starts_in[:-1]

        if ha == 'right':
            along_offset_in = total_line_w
        elif ha == 'left':
            along_offset_in = 0.0
        else:  # center
            along_offset_in = total_line_w / 2.0

        base_cx = line_base_x - (along_offset_in * cos_r) / max(0.001, fig_w_in)
        base_cy = line_base_y - (along_offset_in * sin_r) / max(0.001, fig_h_in)

        # Always anchor each segment at its TYPOGRAPHIC BASELINE
        # (va='baseline'), not at the ink bounding-box edge. matplotlib
        # computes va='top'/'center'/'bottom' from the per-string ink
        # bbox — and ink bbox varies per glyph (e.g. "A" vs "B" vs "y"
        # differ by 1-2 px). For rotated text that per-glyph difference
        # becomes a HORIZONTAL stair-step in the final image, which is
        # the "color change shifts characters slightly" user report.
        # Baseline alignment is stable across glyphs and collapses the
        # stair-step.
        for seg_fp, seg_start, part_text, color in zip(seg_fps, seg_starts_in, seg_texts, seg_colors):
            cur_x = base_cx + (seg_start * cos_r) / max(0.001, fig_w_in)
            cur_y = base_cy + (seg_start * sin_r) / max(0.001, fig_h_in)
            fig.text(cur_x, cur_y, part_text, ha='left', va='baseline',
                     fontproperties=seg_fp, color=color,
                     rotation=rot_deg, rotation_mode='anchor')


# Cache of font name -> file path lookups
_font_path_cache: Dict[str, Optional[str]] = {}


def _wrap_text_to_width(text: str, width_inches: float, font_size: int) -> str:
    """Insert \\n at word boundaries so `text` fits within `width_inches`.

    Preserves any newlines the user already typed (Shift+Enter). Uses a
    conservative average-character-width estimate — exact width depends on
    the specific font, but 0.55 × font-size works reasonably for typical
    sans-serif fonts used in scientific figures.
    """
    import textwrap as _tw
    if not text or width_inches <= 0 or font_size <= 0:
        return text
    avg_char_w_in = font_size / 72.0 * 0.55
    if avg_char_w_in <= 0:
        return text
    max_chars = max(4, int(width_inches / avg_char_w_in))
    # Respect explicit newlines already in the text — each user-defined
    # line is wrapped independently so shift+Enter behaviour is preserved.
    out_lines = []
    for raw_line in text.split("\n"):
        if len(raw_line) <= max_chars:
            out_lines.append(raw_line)
        else:
            wrapped = _tw.wrap(
                raw_line, width=max_chars,
                break_long_words=True, break_on_hyphens=True,
            )
            out_lines.extend(wrapped or [raw_line])
    return "\n".join(out_lines)

def _resolve_font_path(font_name: str) -> Optional[str]:
    """Resolve a font filename to its full path using the font discovery system."""
    if font_name in _font_path_cache:
        return _font_path_cache[font_name]
    from pathlib import Path
    import platform
    if platform.system() == "Windows":
        sys_dirs = [
            os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
            os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts"),
        ]
    else:
        sys_dirs = ["/System/Library/Fonts", "/System/Library/Fonts/Supplemental",
                    "/Library/Fonts", os.path.expanduser("~/Library/Fonts")]
    app_dir = os.path.dirname(os.path.abspath(__file__))
    persistent_dir = str(Path.home() / ".multipanelfigure" / "fonts")
    search_dirs = sys_dirs + [app_dir, os.path.join(app_dir, "..")] + [persistent_dir]
    # First try exact match
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        candidate = os.path.join(d, font_name)
        if os.path.isfile(candidate):
            _font_path_cache[font_name] = candidate
            return candidate
    # Case-insensitive scan (important for Windows virtual font folder)
    font_lower = font_name.lower()
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        try:
            for fn in os.listdir(d):
                if fn.lower() == font_lower and os.path.isfile(os.path.join(d, fn)):
                    result = os.path.join(d, fn)
                    _font_path_cache[font_name] = result
                    return result
        except (PermissionError, OSError):
            continue

    # Fallback 1 — strip the extension and look for any ttf/otf/ttc variant.
    # Helps on Windows where a user picks "arial.ttf" but the installed file
    # is actually "Arial.TTF" or Arial is only present as part of "ArialMT.ttc".
    stem_lower = os.path.splitext(font_lower)[0]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        try:
            for fn in os.listdir(d):
                fn_lower = fn.lower()
                if not fn_lower.endswith((".ttf", ".otf", ".ttc")):
                    continue
                fn_stem = os.path.splitext(fn_lower)[0]
                if fn_stem == stem_lower:
                    result = os.path.join(d, fn)
                    if os.path.isfile(result):
                        _font_path_cache[font_name] = result
                        return result
        except (PermissionError, OSError):
            continue

    # Fallback 2 — ask matplotlib's font_manager. It caches system fonts
    # across every supported platform and understands family names. Useful
    # when the selected filename is just a family name ("Arial") without
    # the extension, or when Windows reports a different filename than
    # what's on disk.
    try:
        from matplotlib import font_manager as _fm
        family = os.path.splitext(font_name)[0]
        found = _fm.findfont(family, fallback_to_default=False)
        if found and os.path.isfile(found):
            _font_path_cache[font_name] = found
            return found
    except Exception:
        pass

    import sys
    print(f"[fonts] Could not resolve '{font_name}' — falling back to matplotlib default", file=sys.stderr, flush=True)
    _font_path_cache[font_name] = None
    return None


def _font_props(font_path: Optional[str], font_name: str, size: int,
                styles: List[str] = None) -> FontProperties:
    styles = styles or []
    kwargs = {"size": size}
    want_bold = "Bold" in styles
    want_italic = "Italic" in styles

    # Try explicit font_path first, then resolve font_name
    resolved = None
    if font_path and os.path.isfile(font_path):
        resolved = font_path
    elif font_name:
        resolved = _resolve_font_path(font_name)

    if resolved and (want_bold or want_italic):
        # Look for the proper Bold/Italic variant file
        base_name = os.path.splitext(os.path.basename(resolved))[0]
        font_dir = os.path.dirname(resolved)
        # Strip existing style suffixes to get the base family name
        for strip in [" Bold Italic", " Bold", " Italic", " Regular"]:
            if base_name.endswith(strip):
                base_name = base_name[:-len(strip)]
                break
        # Build the variant filename
        if want_bold and want_italic:
            variant_suffix = " Bold Italic"
        elif want_bold:
            variant_suffix = " Bold"
        else:
            variant_suffix = " Italic"
        for ext in (".ttf", ".otf", ".ttc"):
            variant_path = os.path.join(font_dir, base_name + variant_suffix + ext)
            if os.path.isfile(variant_path):
                resolved = variant_path
                break

    if resolved:
        kwargs["fname"] = resolved

    fp = FontProperties(**kwargs)
    # Attach extra styles for manual rendering (underline, strikethrough, super/sub)
    fp._extra_styles = [s for s in styles if s in ("Underline", "Strikethrough", "Superscript", "Subscript")]
    fp._want_bold = want_bold
    fp._want_italic = want_italic
    return fp


# ── Column / Row header drawing ───────────────────────────────────────────

def _inches_to_frac(inches, fig_dim):
    """Convert an absolute distance in inches to a figure-fraction."""
    return inches / fig_dim if fig_dim > 0 else 0


def _add_column_headers(fig, axes, header_levels: List[HeaderLevel],
                        rows: int, cols: int, has_labels: bool = False,
                        column_labels=None):
    fig_w, fig_h = fig.get_size_inches()
    ref = max(fig_w, fig_h)

    # Track the Y position of each tier's text top so the next tier
    # starts from there with the configured distance offset.
    base_y_top = None
    base_y_bottom = None

    # Render in reverse order: index 0 = outermost (furthest from panels),
    # last index = innermost (closest to panels). So iterate reversed.
    for level_idx, level in enumerate(reversed(header_levels)):
        max_offset = 0.0
        max_font_size = 10
        for hdr in level.headers:
            if not hdr.columns_or_rows:
                continue
            max_offset = max(max_offset, hdr.distance)
            max_font_size = max(max_font_size, hdr.font_size)

        for hdr in level.headers:
            if not hdr.columns_or_rows:
                continue
            group_cols = sorted(hdr.columns_or_rows)
            fp = _font_props(hdr.font_path, hdr.font_name, hdr.font_size, hdr.font_style)
            segments = _get_segments(hdr)

            ax_left = axes[0, group_cols[0]]
            ax_right = axes[0, group_cols[-1]]
            bbox_l = ax_left.get_position()
            bbox_r = ax_right.get_position()
            cx = (bbox_l.x0 + bbox_r.x1) / 2

            # Auto-wrap long headers onto multiple lines so they stay within
            # the span-width. Keep any user-inserted newlines (Shift+Enter)
            # intact. We only auto-wrap when the header is rendered with a
            # uniform style — per-character styled_segments would require
            # splitting segments at line breaks which we defer.
            uniform = (not segments) or (
                len(set(s.get("color") for s in segments if s.get("color"))) <= 1
                and not any(s.get("font_size") or s.get("font_name") for s in segments)
            )
            if uniform and hdr.text:
                span_w_inches = max(0.0, (bbox_r.x1 - bbox_l.x0)) * fig_w
                wrapped = _wrap_text_to_width(hdr.text, span_w_inches, hdr.font_size)
                if wrapped != hdr.text:
                    # Replace segments with a single default-coloured one
                    # so multi-line rendering happens cleanly.
                    segments = [{
                        "color": hdr.default_color or "#000000",
                        "text": wrapped,
                        "font_name": None,
                        "font_size": None,
                        "font_style": None,
                    }]

            # For the first tier (closest to panels), start from the primary label position.
            # IMPORTANT: scale label_h by the MAX rendered line-count across all
            # primary column labels at this position — otherwise a multi-line
            # primary label (Shift+Enter newlines, shipped in 0.1.113) overlaps
            # the secondary header above/below it.
            if base_y_top is None:
                base_y_top = bbox_l.y1
                if has_labels and column_labels:
                    top_lbls = [l for l in column_labels
                                if l.text.strip() and getattr(l, 'position', 'Top') == 'Top']
                    if top_lbls:
                        lbl_dist = max(l.distance for l in top_lbls)
                        lbl_fs = max(l.font_size for l in top_lbls)
                        max_lines = max((_count_header_lines(l) for l in top_lbls), default=1)
                    else:
                        lbl_dist = column_labels[0].distance
                        lbl_fs = column_labels[0].font_size
                        max_lines = 1
                    label_offset = lbl_dist * ref
                    # Full text-block height: one line + (N-1) × line_height
                    # at matplotlib's 1.2 line-height factor.
                    label_h = (lbl_fs / 72.0) * (1.0 + 1.2 * (max_lines - 1))
                    # Compensate for rotated row label font centering
                    font_h_frac_correction = _inches_to_frac(lbl_fs / 72.0 * 0.5, fig_h)
                    base_y_top += _inches_to_frac(label_offset + label_h, fig_h) + font_h_frac_correction
            if base_y_bottom is None:
                ax_bottom = axes[-1, group_cols[0]]
                base_y_bottom = ax_bottom.get_position().y0
                if has_labels and column_labels:
                    bot_lbls = [l for l in column_labels
                                if l.text.strip() and getattr(l, 'position', 'Top') == 'Bottom']
                    if bot_lbls:
                        lbl_dist = max(l.distance for l in bot_lbls)
                        lbl_fs = max(l.font_size for l in bot_lbls)
                        max_lines = max((_count_header_lines(l) for l in bot_lbls), default=1)
                    else:
                        lbl_dist = column_labels[0].distance
                        lbl_fs = column_labels[0].font_size
                        max_lines = 1
                    label_offset = lbl_dist * ref
                    label_h = (lbl_fs / 72.0) * (1.0 + 1.2 * (max_lines - 1))
                    font_h_frac_correction = _inches_to_frac(lbl_fs / 72.0 * 0.5, fig_h)
                    base_y_bottom -= _inches_to_frac(label_offset + label_h, fig_h) - font_h_frac_correction

            # The distance controls the gap from previous element as % of ref
            gap_inches = max_offset * ref
            gap_inches = max(gap_inches, 0.04)  # small minimum to prevent text overlap
            dist_y = _inches_to_frac(gap_inches, fig_h)

            if hdr.position == "Top":
                cy = base_y_top + dist_y
            else:
                cy = base_y_bottom - dist_y

            _draw_colored_text(fig, cx, cy, segments, fp,
                               ha="center",
                               va="bottom" if hdr.position == "Top" else "top",
                               rotation=hdr.rotation)

            # Draw line for ALL headers (not just spanning)
            if hdr.line_width > 0:
                line_gap_y = _inches_to_frac(0.05, fig_h)
                y_line = cy - line_gap_y if hdr.position == "Top" else cy + line_gap_y
                line_length = getattr(hdr, 'line_length', 1.0) or 1.0
                span_w = bbox_r.x1 - bbox_l.x0
                half_span = span_w * line_length / 2
                mid_x = (bbox_l.x0 + bbox_r.x1) / 2
                line_style = getattr(hdr, 'line_style', 'solid') or 'solid'
                ls_map = {'solid': '-', 'dashed': '--', 'dotted': ':', 'dash-dot': '-.'}
                ls = ls_map.get(line_style, '-')
                fig.add_artist(plt.Line2D(
                    [mid_x - half_span, mid_x + half_span], [y_line, y_line],
                    transform=fig.transFigure, color=hdr.line_color,
                    linewidth=hdr.line_width, linestyle=ls, clip_on=False))
                # End caps: small perpendicular lines at each end toward previous header
                if getattr(hdr, 'end_caps', False):
                    cap_len = _inches_to_frac(0.06, fig_h)  # ~2pt cap
                    # Caps point toward the panels (downward for Top, upward for Bottom)
                    cap_dir = -1 if hdr.position == "Top" else 1
                    for cap_x in [mid_x - half_span, mid_x + half_span]:
                        fig.add_artist(plt.Line2D(
                            [cap_x, cap_x],
                            [y_line, y_line + cap_dir * cap_len],
                            transform=fig.transFigure, color=hdr.line_color,
                            linewidth=hdr.line_width, linestyle='-', clip_on=False))

        # After this tier, update base_y ONLY for the position that was used.
        # Account for multi-line headers (explicit \n or styled segments containing \n)
        # so the NEXT tier starts above/below the full rendered text block.
        font_h_inches = max_font_size / 72.0
        max_lines_in_tier = max(
            (_count_header_lines(h) for h in level.headers if h.columns_or_rows),
            default=1,
        )
        tier_h_inches = font_h_inches * 1.2 * max_lines_in_tier
        tier_h_frac = _inches_to_frac(tier_h_inches, fig_h)
        has_top = any(h.position == "Top" and h.columns_or_rows for h in level.headers)
        has_bottom = any(h.position == "Bottom" and h.columns_or_rows for h in level.headers)
        if has_top:
            base_y_top = base_y_top + dist_y + tier_h_frac
        if has_bottom:
            base_y_bottom = base_y_bottom - dist_y - tier_h_frac


def _add_row_headers(fig, axes, header_levels: List[HeaderLevel],
                     rows: int, cols: int, has_labels: bool = False,
                     row_labels=None):
    fig_w, fig_h = fig.get_size_inches()
    ref = max(fig_w, fig_h)

    base_x_left = None
    base_x_right = None

    # Render in reverse order: index 0 = outermost, last = innermost
    for level_idx, level in enumerate(reversed(header_levels)):
        max_offset = 0.0
        max_font_size = 10
        for hdr in level.headers:
            if not hdr.columns_or_rows:
                continue
            max_offset = max(max_offset, hdr.distance)
            max_font_size = max(max_font_size, hdr.font_size)

        for hdr in level.headers:
            if not hdr.columns_or_rows:
                continue
            group_rows = sorted(hdr.columns_or_rows)
            fp = _font_props(hdr.font_path, hdr.font_name, hdr.font_size, hdr.font_style)
            segments = _get_segments(hdr)

            ax_top = axes[group_rows[0], 0]
            ax_bot = axes[group_rows[-1], 0]
            bbox_t = ax_top.get_position()
            bbox_b = ax_bot.get_position()
            cy = (bbox_t.y1 + bbox_b.y0) / 2

            # Auto-wrap long row headers. "Width" for a row header is the
            # span height if rotated (common case — vertical row headers)
            # or the fixed horizontal label-band width when rendered
            # horizontally. Wrap on the axis that the text actually
            # extends along.
            rot = float(getattr(hdr, "rotation", 90.0) or 0.0)
            is_vertical = abs((rot % 180.0)) > 45.0
            uniform = (not segments) or (
                len(set(s.get("color") for s in segments if s.get("color"))) <= 1
                and not any(s.get("font_size") or s.get("font_name") for s in segments)
            )
            if uniform and hdr.text:
                if is_vertical:
                    # Text runs vertically along the row span — use the
                    # span-height as the available text length.
                    wrap_extent_inches = max(0.0, (bbox_t.y1 - bbox_b.y0)) * fig_h
                else:
                    # Horizontal row header — only the label-band width
                    # (roughly the distance × ref) is available.
                    wrap_extent_inches = max(0.4, max_offset * ref * 0.9)
                wrapped = _wrap_text_to_width(hdr.text, wrap_extent_inches, hdr.font_size)
                if wrapped != hdr.text:
                    segments = [{
                        "color": hdr.default_color or "#000000",
                        "text": wrapped,
                        "font_name": None,
                        "font_size": None,
                        "font_style": None,
                    }]

            # Same multi-line accounting as column headers: account for the
            # MAX line count across primary row labels at this position so
            # secondary/tertiary row headers don't overlap a multi-line primary.
            if base_x_left is None:
                base_x_left = bbox_t.x0
                if has_labels and row_labels:
                    left_lbls = [l for l in row_labels
                                 if l.text.strip() and getattr(l, 'position', 'Left') == 'Left']
                    if left_lbls:
                        lbl_dist = max(l.distance for l in left_lbls)
                        lbl_fs = max(l.font_size for l in left_lbls)
                        max_lines = max((_count_header_lines(l) for l in left_lbls), default=1)
                    else:
                        lbl_dist = row_labels[0].distance
                        lbl_fs = row_labels[0].font_size
                        max_lines = 1
                    label_offset = lbl_dist * ref
                    label_w = (lbl_fs / 72.0) * (1.0 + 1.2 * (max_lines - 1))
                    font_w_frac_correction = _inches_to_frac(lbl_fs / 72.0 * 0.5, fig_w)
                    base_x_left -= _inches_to_frac(label_offset + label_w, fig_w) + font_w_frac_correction
            if base_x_right is None:
                ax_r = axes[group_rows[0], -1]
                base_x_right = ax_r.get_position().x1
                if has_labels and row_labels:
                    right_lbls = [l for l in row_labels
                                  if l.text.strip() and getattr(l, 'position', 'Left') == 'Right']
                    if right_lbls:
                        lbl_dist = max(l.distance for l in right_lbls)
                        lbl_fs = max(l.font_size for l in right_lbls)
                        max_lines = max((_count_header_lines(l) for l in right_lbls), default=1)
                    else:
                        lbl_dist = row_labels[0].distance
                        lbl_fs = row_labels[0].font_size
                        max_lines = 1
                    label_offset = lbl_dist * ref
                    label_w = (lbl_fs / 72.0) * (1.0 + 1.2 * (max_lines - 1))
                    font_w_frac_correction = _inches_to_frac(lbl_fs / 72.0 * 0.5, fig_w)
                    base_x_right += _inches_to_frac(label_offset + label_w, fig_w) + font_w_frac_correction

            gap_inches = max_offset * ref
            gap_inches = max(gap_inches, 0.04)
            dist_x = _inches_to_frac(gap_inches, fig_w)

            if hdr.position == "Left":
                cx = base_x_left - dist_x
            else:
                cx = base_x_right + dist_x

            _draw_colored_text(fig, cx, cy, segments, fp,
                               ha="right" if hdr.position == "Left" else "left",
                               va="center", rotation=hdr.rotation)

            # Draw line for ALL headers (not just spanning)
            if hdr.line_width > 0:
                line_gap_x = _inches_to_frac(0.05, fig_w)
                x_line = cx + line_gap_x if hdr.position == "Left" else cx - line_gap_x
                line_length = getattr(hdr, 'line_length', 1.0) or 1.0
                span_h = bbox_t.y1 - bbox_b.y0
                half_span = span_h * line_length / 2
                mid_y = (bbox_b.y0 + bbox_t.y1) / 2
                line_style = getattr(hdr, 'line_style', 'solid') or 'solid'
                ls_map = {'solid': '-', 'dashed': '--', 'dotted': ':', 'dash-dot': '-.'}
                ls = ls_map.get(line_style, '-')
                fig.add_artist(plt.Line2D(
                    [x_line, x_line], [mid_y - half_span, mid_y + half_span],
                    transform=fig.transFigure, color=hdr.line_color,
                    linewidth=hdr.line_width, linestyle=ls, clip_on=False))
                # End caps: small perpendicular lines at each end toward previous header
                if getattr(hdr, 'end_caps', False):
                    cap_len = _inches_to_frac(0.06, fig_w)
                    # Caps point toward the panels (rightward for Left, leftward for Right)
                    cap_dir = 1 if hdr.position == "Left" else -1
                    for cap_y in [mid_y - half_span, mid_y + half_span]:
                        fig.add_artist(plt.Line2D(
                            [x_line, x_line + cap_dir * cap_len],
                            [cap_y, cap_y],
                            transform=fig.transFigure, color=hdr.line_color,
                            linewidth=hdr.line_width, linestyle='-', clip_on=False))

        # After this tier, update base_x to account for the text width.
        # Account for multi-line headers so the NEXT tier sits clear of the
        # full stacked-line block of the current tier.
        font_h_inches = max_font_size / 72.0
        max_lines_in_tier = max(
            (_count_header_lines(h) for h in level.headers if h.columns_or_rows),
            default=1,
        )
        tier_w_inches = font_h_inches * 1.2 * max_lines_in_tier
        tier_w_frac = _inches_to_frac(tier_w_inches, fig_w)
        has_left = any(h.position == "Left" and h.columns_or_rows for h in level.headers)
        has_right = any(h.position == "Right" and h.columns_or_rows for h in level.headers)
        if has_left:
            base_x_left = base_x_left - dist_x - tier_w_frac
        if has_right:
            base_x_right = base_x_right + dist_x + tier_w_frac


# ── Simple column/row labels ─────────────────────────────────────────────

def _add_column_labels(fig, axes, labels: List[AxisLabel], rows: int, cols: int):
    fig_w, fig_h = fig.get_size_inches()
    # Use the geometric mean so the same distance value produces
    # the same physical offset regardless of figure aspect ratio.
    ref = (fig_w * fig_h) ** 0.5
    for ci, lbl in enumerate(labels):
        if not lbl.text.strip():
            continue
        fp = _font_props(lbl.font_path, lbl.font_name, lbl.font_size, lbl.font_style)
        segments = _get_segments(lbl)

        ax = axes[0, ci]
        bbox = ax.get_position()
        cx = (bbox.x0 + bbox.x1) / 2
        dist_frac = lbl.distance * ref / fig_h

        # Auto-wrap long column labels onto multiple lines based on the
        # column-width. Shift+Enter newlines in the label text are
        # preserved.
        uniform = (not segments) or (
            len(set(s.get("color") for s in segments if s.get("color"))) <= 1
            and not any(s.get("font_size") or s.get("font_name") for s in segments)
        )
        if uniform and lbl.text:
            col_w_in = max(0.0, (bbox.x1 - bbox.x0)) * fig_w
            wrapped = _wrap_text_to_width(lbl.text, col_w_in, lbl.font_size)
            if wrapped != lbl.text:
                segments = [{
                    "color": lbl.default_color or "#000000",
                    "text": wrapped,
                    "font_name": None, "font_size": None, "font_style": None,
                }]

        if lbl.position == "Top":
            cy = bbox.y1 + dist_frac
            va = "bottom"
        else:
            ax_b = axes[-1, ci]
            cy = ax_b.get_position().y0 - dist_frac
            va = "top"
        _draw_colored_text(fig, cx, cy, segments, fp,
                           ha="center", va=va, rotation=lbl.rotation)


def _add_row_labels(fig, axes, labels: List[AxisLabel], rows: int, cols: int):
    fig_w, fig_h = fig.get_size_inches()
    ref = (fig_w * fig_h) ** 0.5
    for ri, lbl in enumerate(labels):
        if not lbl.text.strip():
            continue
        fp = _font_props(lbl.font_path, lbl.font_name, lbl.font_size, lbl.font_style)
        segments = _get_segments(lbl)

        ax = axes[ri, 0]
        bbox = ax.get_position()
        cy = (bbox.y0 + bbox.y1) / 2
        rot = lbl.rotation

        # Auto-wrap long row labels along their text direction — row
        # height if rotated vertical, else the allotted side-band width.
        is_vertical = abs((rot % 180.0)) > 45.0
        uniform = (not segments) or (
            len(set(s.get("color") for s in segments if s.get("color"))) <= 1
            and not any(s.get("font_size") or s.get("font_name") for s in segments)
        )
        if uniform and lbl.text:
            if is_vertical:
                row_h_in = max(0.0, (bbox.y1 - bbox.y0)) * fig_h
                wrap_extent_inches = row_h_in
            else:
                wrap_extent_inches = max(0.4, lbl.distance * ref * 0.9)
            wrapped = _wrap_text_to_width(lbl.text, wrap_extent_inches, lbl.font_size)
            if wrapped != lbl.text:
                segments = [{
                    "color": lbl.default_color or "#000000",
                    "text": wrapped,
                    "font_name": None, "font_size": None, "font_style": None,
                }]
        dist_frac = lbl.distance * ref / fig_w
        # For rotated text with ha="center", the text center (not edge)
        # sits at the coordinate.  Add half the font height so the nearest
        # text edge is at the same visual distance as column labels.
        if abs(rot) > 1:
            font_h_frac = (lbl.font_size / 72.0) / fig_w  # font height in inches → fig frac
            edge_correction = font_h_frac * 0.5
        else:
            edge_correction = 0
        if lbl.position == "Left":
            cx = bbox.x0 - dist_frac - edge_correction
            ha = "center" if abs(rot) > 1 else "right"
        else:
            ax_r = axes[ri, -1]
            cx = ax_r.get_position().x1 + dist_frac + edge_correction
            ha = "center" if abs(rot) > 1 else "left"
        _draw_colored_text(fig, cx, cy, segments, fp,
                           ha=ha, va="center", rotation=rot)


# ── Panel labels (drawn at figure DPI, like headers) ─────────────────────

def _add_panel_labels(fig, axes, cfg, rows: int, cols: int,
                      processed_images: List[List[Image.Image]] = None,
                      col_widths: list = None, row_heights: list = None,
                      panel_override=None):
    """Draw each panel's labels using matplotlib text at figure DPI.
    panel_override: optional (src_r, src_c) to render a specific panel's
    labels on axes[0,0] (used for single-panel preview)."""
    for r in range(rows):
        for c in range(cols):
            if panel_override:
                panel = cfg.panels[panel_override[0]][panel_override[1]]
            else:
                panel = cfg.panels[r][c]
            if not panel.labels:
                continue
            ax = axes[r, c]
            bbox = ax.get_position()  # in figure fraction coords
            ax_w = bbox.x1 - bbox.x0
            ax_h = bbox.y1 - bbox.y0

            # Compute the image content area within the (possibly padded) cell
            # so labels are positioned relative to the actual image, not the padding
            img_x_offset_frac = 0.0
            img_y_offset_frac = 0.0
            img_w_frac = ax_w
            img_h_frac = ax_h
            if processed_images and col_widths and row_heights:
                img = processed_images[r][c]
                if img is not None:
                    iw, ih = img.size
                    tw, th = int(col_widths[c]), int(row_heights[r])
                    if iw < tw or ih < th:
                        # Image was padded — compute the content region
                        ox = (tw - iw) / 2 / tw  # padding offset as fraction of cell
                        oy = (th - ih) / 2 / th
                        img_x_offset_frac = ox * ax_w
                        img_y_offset_frac = oy * ax_h
                        img_w_frac = (iw / tw) * ax_w
                        img_h_frac = (ih / th) * ax_h

            for lbl in panel.labels:
                # Position within the ACTUAL image area (not the padded cell)
                fx = bbox.x0 + img_x_offset_frac + (lbl.position_x / 100.0) * img_w_frac
                # Y is inverted: 0% = top of image, 100% = bottom
                fy = bbox.y1 - img_y_offset_frac - (lbl.position_y / 100.0) * img_h_frac
                fp = _font_props(lbl.font_path, lbl.font_name,
                                 lbl.font_size, lbl.font_style)
                segments = _get_segments(lbl) if hasattr(lbl, 'styled_segments') and lbl.styled_segments else [
                    {'color': c, 'text': t, 'font_name': None, 'font_size': None, 'font_style': None}
                    for c, t in parse_colored_text(lbl.text, getattr(lbl, 'default_color', lbl.color))
                ]
                _draw_colored_text(fig, fx, fy, segments, fp,
                                   ha="left", va="top", rotation=lbl.rotation)


def _rotate_point_xy(px, py, cx, cy, angle_deg):
    """Rotate point (px,py) around center (cx,cy) by angle_deg degrees."""
    import math
    a = math.radians(angle_deg)
    dx, dy = px - cx, py - cy
    return (cx + dx * math.cos(a) - dy * math.sin(a),
            cy + dx * math.sin(a) + dy * math.cos(a))


# ── Panel symbols (drawn at figure DPI via matplotlib) ───────────────────

def _add_panel_symbols(fig, axes, cfg, rows: int, cols: int,
                       processed_images: List[List[Image.Image]],
                       panel_override=None, original_images=None):
    """Draw each panel's symbols using matplotlib patches at figure DPI.
    Symbol coordinates (x, y, size) are percentage-based (0-100)."""
    for r in range(rows):
        for c in range(cols):
            if panel_override:
                panel = cfg.panels[panel_override[0]][panel_override[1]]
            else:
                panel = cfg.panels[r][c]
            if not panel.symbols:
                continue
            ax = axes[r, c]
            img = processed_images[r][c]
            if img is None:
                continue
            iw, ih = img.size
            # Convert symbol size from abstract units to data coordinates.
            # We want size to behave like font_size (absolute visual size).
            # Use axes transform: size_inches = sym.size / 72
            # data_units_per_inch = iw / (axes_width_inches)
            bbox = ax.get_position()
            ax_w_inches = bbox.width * fig.get_figwidth()
            data_per_inch = iw / max(ax_w_inches, 0.1)
            scale_factor = data_per_inch / 72.0  # converts points to data units
            for sym in panel.symbols:
                from symbol_defs import symbol_to_pixels
                from matplotlib.patches import Polygon as MplPolygon
                # Convert percentage (0-100) to pixel/data coordinates
                cx = sym.x / 100.0 * iw
                cy = sym.y / 100.0 * ih
                sz = max(3, sym.size * scale_factor)
                color = sym.color
                # Stroke scales with symbol size for visibility
                lw = max(0.8, min(2.5, sz / 20))

                data = symbol_to_pixels(sym.shape, cx, cy, sz, sym.rotation)

                # Draw filled polygons
                for poly in data["fill"]:
                    if len(poly) >= 3:
                        if data["filled"]:
                            patch = MplPolygon(poly, closed=True,
                                             facecolor=color, edgecolor=color,
                                             lw=0.3, clip_on=False)
                        else:
                            patch = MplPolygon(poly, closed=True,
                                             fill=False, edgecolor=color,
                                             lw=lw, clip_on=False)
                        ax.add_patch(patch)

                # Draw stroke lines
                for polyline in data["stroke"]:
                    if len(polyline) >= 2:
                        xs = [p[0] for p in polyline]
                        ys = [p[1] for p in polyline]
                        ax.plot(xs, ys, color=color, lw=lw, clip_on=False,
                                solid_capstyle='round')

                # Symbol text label
                if sym.label_text:
                    font_name = getattr(sym, 'label_font_name', 'arial.ttf') or 'arial.ttf'
                    font_path = getattr(sym, 'label_font_path', None)
                    font_style = getattr(sym, 'label_font_style', []) or []
                    fp = _font_props(font_path, font_name, sym.label_font_size, font_style)
                    # Use absolute position if set, otherwise auto
                    lpos_x = getattr(sym, 'label_position_x', -1)
                    lpos_y = getattr(sym, 'label_position_y', -1)
                    if lpos_x >= 0 and lpos_y >= 0:
                        lx = lpos_x / 100.0 * iw
                        ly = lpos_y / 100.0 * ih
                    else:
                        lx = cx + sz * 0.5
                        ly = cy - sz * 0.25
                    ax.text(lx, ly, sym.label_text, color=sym.label_color,
                            fontproperties=fp, clip_on=False)


def _add_panel_scale_bars(fig, axes, cfg, rows: int, cols: int,
                          processed_images: List[List[Image.Image]],
                          panel_override=None, pre_norm_sizes=None,
                          col_widths=None, row_heights=None):
    """Draw each panel's scale bar using matplotlib at figure DPI.
    Uses original (pre-padded) images for positioning within the image content area."""
    for r in range(rows):
        for c in range(cols):
            if panel_override:
                panel = cfg.panels[panel_override[0]][panel_override[1]]
            else:
                panel = cfg.panels[r][c]
            if not panel.add_scale_bar or not panel.scale_bar:
                continue
            ax = axes[r, c]
            img = processed_images[r][c]
            if img is None:
                continue
            iw, ih = img.size  # actual image size (pre-padded)
            sb = panel.scale_bar

            # Compute padding offset (image is centered in the padded cell)
            pad_ox, pad_oy = 0, 0
            cell_w, cell_h = iw, ih
            if col_widths and row_heights and not panel_override:
                cell_w = int(col_widths[c])
                cell_h = int(row_heights[r])
                pad_ox = (cell_w - iw) // 2
                pad_oy = (cell_h - ih) // 2

            # bar_length_px: calibrated for original (pre-normalize) image
            bar_length_px = int(sb.bar_length_microns / max(sb.micron_per_pixel, 1e-9))
            bar_height = sb.bar_height

            # If image was normalized (scaled up), scale bar dimensions too
            if pre_norm_sizes and not panel_override:
                pre = pre_norm_sizes[r][c]
                if pre and pre[0] > 0:
                    nsf = iw / pre[0]
                    bar_length_px = int(bar_length_px * nsf)
                    bar_height = int(bar_height * nsf)

            # Position relative to image content area (not padding)
            edge_frac = sb.edge_distance / 100.0
            if sb.position_preset and sb.position_preset != "Custom":
                if "Right" in sb.position_preset:
                    bx = iw * (1 - edge_frac) - bar_length_px
                else:
                    bx = iw * edge_frac
                if "Bottom" in sb.position_preset:
                    by = ih * (1 - edge_frac) - bar_height - 5
                else:
                    by = ih * edge_frac + 5
            else:
                bx = sb.position_x / 100.0 * iw - bar_length_px / 2
                by = sb.position_y / 100.0 * ih

            # Clamp to image content area
            bx = max(0, min(bx, iw - bar_length_px - 1))
            by = max(0, min(by, ih - bar_height - 1))
            # Offset by padding to position within the padded cell
            bx += pad_ox
            by += pad_oy

            # Draw bar as a filled rectangle
            bar_rect = Rectangle((bx, by), bar_length_px, bar_height,
                                 facecolor=sb.bar_color, edgecolor='none')
            ax.add_patch(bar_rect)

            # Label text — auto-generate if empty, using the selected unit
            label_text = sb.label
            if not label_text:
                unit = getattr(sb, 'unit', 'um') or 'um'
                unit_display = {'km': 'km', 'm': 'm', 'nm': 'nm', 'mm': 'mm', 'cm': 'cm', 'pm': 'pm'}.get(unit, '\u00B5m')
                from models import UNIT_TO_MICRONS
                um_per_unit = UNIT_TO_MICRONS.get(unit, 1.0)
                bar_in_unit = sb.bar_length_microns / um_per_unit
                # Clean display: remove trailing zeros
                label_text = f"{bar_in_unit:g} {unit_display}"

            # Render text using matplotlib fonts (same quality as column/row labels)
            fp = _font_props(
                getattr(sb, 'font_path', None),
                getattr(sb, 'font_name', 'arial.ttf'),
                sb.font_size,
                getattr(sb, 'label_font_style', [])
            )
            label_color = getattr(sb, 'label_color', sb.bar_color) or sb.bar_color

            # Place label above or below bar depending on position
            # If bar is in the bottom half, put label above to avoid clipping
            is_bottom = by > ih * 0.5
            if is_bottom:
                text_y = by - 2
                text_va = 'bottom'
            else:
                text_y = by + bar_height + 2
                text_va = 'top'

            ax.text(bx + bar_length_px / 2 + sb.label_x_offset,
                    text_y,
                    label_text,
                    color=label_color,
                    fontproperties=fp,
                    ha='center', va=text_va,
                    clip_on=False)


# ── Adjacent Panel zoom cross-panel lines ─────────────────────────────────

def _draw_adjacent_zoom_lines(fig, axes, cfg, rows, cols, processed_images, orig_sizes):
    """Draw connecting lines from source rect to adjacent panel zoomed image."""
    from matplotlib.patches import ConnectionPatch

    for r in range(rows):
        for c in range(cols):
            panel = cfg.panels[r][c]
            if not panel.add_zoom_inset or not panel.zoom_inset:
                continue
            zi = panel.zoom_inset
            if zi.inset_type != "Adjacent Panel":
                continue

            ar, ac = r, c
            side = zi.side or "Right"
            if side == "Top": ar -= 1
            elif side == "Bottom": ar += 1
            elif side == "Left": ac -= 1
            elif side == "Right": ac += 1
            if not (0 <= ar < rows and 0 <= ac < cols):
                continue

            src_ax = axes[r, c] if axes.ndim == 2 else axes[r * cols + c]
            tgt_ax = axes[ar, ac] if axes.ndim == 2 else axes[ar * cols + ac]

            # Source rect corners in source image data coordinates
            # (no padding offset needed — PIL draws rect at zi.x, zi.y on original image,
            #  padding centers it, so data coords include padding offset)
            src_orig = orig_sizes.get((r, c), (1, 1))
            src_padded = processed_images[r][c].size if processed_images[r][c] else src_orig
            spad_x = (src_padded[0] - src_orig[0]) / 2
            spad_y = (src_padded[1] - src_orig[1]) / 2
            sx1 = spad_x + zi.x
            sy1 = spad_y + zi.y
            sx2 = spad_x + zi.x + zi.width
            sy2 = spad_y + zi.y + zi.height

            # Target zoomed image position within padded target image
            tgt_orig = orig_sizes.get((ar, ac), (1, 1))
            tgt_padded = processed_images[ar][ac].size if processed_images[ar][ac] else tgt_orig
            tpx = (tgt_padded[0] - tgt_orig[0]) / 2
            tpy = (tgt_padded[1] - tgt_orig[1]) / 2
            # Corners of actual zoomed content in target data coords
            tl_x, tl_y = tpx, tpy                          # top-left
            br_x, br_y = tpx + tgt_orig[0], tpy + tgt_orig[1]  # bottom-right

            color = zi.line_color or "#FF0000"
            bg = (cfg.background or "white").lower()
            if color.lower() in ("#ffffff", "white", "#fff") and bg in ("white", "#ffffff", "#fff"):
                color = "#333333"
            lw = max(0.5, zi.line_width * 0.5)

            # Connect source corners to target zoomed image corners
            if side == "Right":
                pairs = [((sx2, sy1), (tl_x, tl_y)), ((sx2, sy2), (tl_x, br_y))]
            elif side == "Left":
                pairs = [((sx1, sy1), (br_x, tl_y)), ((sx1, sy2), (br_x, br_y))]
            elif side == "Top":
                pairs = [((sx1, sy1), (tl_x, br_y)), ((sx2, sy1), (br_x, br_y))]
            else:
                pairs = [((sx1, sy2), (tl_x, tl_y)), ((sx2, sy2), (br_x, tl_y))]

            # Render the figure to finalize transforms before computing coordinates
            fig.canvas.draw()

            for (src_pt, tgt_pt) in pairs:
                # Convert data coords to figure fraction using finalized transforms
                src_fig = (src_ax.transData + fig.transFigure.inverted()).transform(src_pt)
                tgt_fig = (tgt_ax.transData + fig.transFigure.inverted()).transform(tgt_pt)

                import matplotlib.lines as mlines
                line = mlines.Line2D(
                    [src_fig[0], tgt_fig[0]], [src_fig[1], tgt_fig[1]],
                    transform=fig.transFigure,
                    color=color, linewidth=lw, alpha=0.8,
                    clip_on=False, zorder=100)
                fig.add_artist(line)



# ── Main assembly ─────────────────────────────────────────────────────────

def assemble_figure(cfg: FigureConfig,
                    processed_images: List[List[Image.Image]],
                    dpi: int = 300,
                    full_res_sizes: Optional[Dict] = None,
                    ) -> bytes:
    """
    Assemble the final figure.  *processed_images* is a rows×cols list of
    already-processed PIL images (or None for empty panels).
    *dpi* controls the rendering resolution for all text and vector elements
    (labels, headers, symbols) — ensures crisp output at any resolution.
    Returns PNG or TIFF bytes.
    """
    rows, cols = cfg.rows, cfg.cols

    # Compute margins in INCHES for consistent physical spacing on all sides.
    # Then convert to figure-fractions after fig_w/fig_h are known.
    has_col_labels = any(lbl.text.strip() for lbl in cfg.column_labels)
    has_row_labels = any(lbl.text.strip() for lbl in cfg.row_labels)
    margin_inches = 0.08                # outer padding

    # Calculate header space needed per level:
    # Each level needs: gap (min 0.04") + font height + breathing room
    # Use actual header font sizes when available
    def _header_space(header_levels, ref_inches=10.0):
        """Calculate total inches needed for header levels."""
        total = 0.0
        for level in header_levels:
            max_fs = 10
            max_dist = 0.0
            for hdr in level.headers:
                if hdr.columns_or_rows:
                    max_fs = max(max_fs, hdr.font_size)
                    max_dist = max(max_dist, hdr.distance)
            gap = max(max_dist * ref_inches, 0.04)
            font_h = max_fs / 72.0
            total += gap + font_h + 0.05  # 0.05" breathing room
        return total

    ref_dim = 10.0  # approximate reference dimension for distance calc

    # Split header levels by position for accurate margin calculation
    def _header_space_by_pos(header_levels, position, ref_inches=10.0):
        """Calculate space for headers at a specific position (Top/Bottom/Left/Right)."""
        total = 0.0
        for level in header_levels:
            has_pos = any(h.position == position and h.columns_or_rows for h in level.headers)
            if not has_pos:
                continue
            max_fs = 10
            max_dist = 0.0
            max_lines = 1
            for hdr in level.headers:
                if hdr.columns_or_rows and hdr.position == position:
                    max_fs = max(max_fs, hdr.font_size)
                    max_dist = max(max_dist, hdr.distance)
                    max_lines = max(max_lines, _count_header_lines(hdr))
            gap = max(max_dist * ref_inches, 0.04)
            font_h = max_fs / 72.0 * 1.2 * max_lines
            total += gap + font_h + 0.04
        return total

    # Column/row labels can individually be at different positions.
    # Check ALL labels for each position, not just the first one.
    def _label_space_by_pos(labels, has_labels, position):
        """Calculate space needed for labels at a specific position.

        Accounts for multi-line labels (Shift+Enter newlines) by counting
        the max line-count across all labels at this position — otherwise a
        label like "Row 1\\nDetail" clips off the figure edge because we
        only reserved one line's worth of space.
        """
        if not has_labels or not labels:
            return 0.0
        pos_labels = [l for l in labels if l.text.strip() and getattr(l, 'position', 'Top') == position]
        if not pos_labels:
            return 0.0
        fs = max((l.font_size for l in pos_labels), default=12)
        dist = max((l.distance for l in pos_labels), default=0.01)
        max_lines = max((_count_header_lines(l) for l in pos_labels), default=1)
        return max(0.18, dist * ref_dim + fs / 72.0 * 1.2 * max_lines + 0.06)

    top_label_space = _label_space_by_pos(cfg.column_labels, has_col_labels, "Top")
    bottom_label_space = _label_space_by_pos(cfg.column_labels, has_col_labels, "Bottom")
    left_label_space = _label_space_by_pos(cfg.row_labels, has_row_labels, "Left")
    right_label_space = _label_space_by_pos(cfg.row_labels, has_row_labels, "Right")

    top_margin_inches = margin_inches + top_label_space + _header_space_by_pos(cfg.column_headers, "Top", ref_dim)
    bottom_margin_inches = margin_inches + bottom_label_space + _header_space_by_pos(cfg.column_headers, "Bottom", ref_dim)
    left_margin_inches = margin_inches + left_label_space + _header_space_by_pos(cfg.row_headers, "Left", ref_dim)
    right_margin_inches = margin_inches + right_label_space + _header_space_by_pos(cfg.row_headers, "Right", ref_dim)

    # --- Mixed aspect ratio sizing (4.2) ---
    # When normalize_widths is enabled, scale smaller images to match the
    # widest (or tallest) in each column/row before computing ratios.
    normalize = getattr(cfg, 'normalize_widths', False)
    norm_mode = getattr(cfg, 'normalize_mode', 'width')

    # Save pre-normalize sizes for scale bar pixel calculation
    pre_norm_sizes = [[None]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            img = processed_images[r][c]
            if img is not None:
                pre_norm_sizes[r][c] = img.size

    if normalize:
        if norm_mode == 'width':
            # Match width: scale ALL images to the widest image across the ENTIRE grid.
            # This ensures uniform width regardless of column, so rows don't get
            # disproportionately tall from a single wide image in one column.
            global_max_w = 1
            for r in range(rows):
                for c in range(cols):
                    img = processed_images[r][c]
                    if img is not None:
                        global_max_w = max(global_max_w, img.size[0])
            for r in range(rows):
                for c in range(cols):
                    img = processed_images[r][c]
                    if img is not None and img.size[0] < global_max_w:
                        scale = global_max_w / img.size[0]
                        processed_images[r][c] = img.resize(
                            (global_max_w, int(img.size[1] * scale)), Image.LANCZOS)
        elif norm_mode == 'height':
            # Match height: scale ALL images to the tallest image across the ENTIRE grid.
            global_max_h = 1
            for r in range(rows):
                for c in range(cols):
                    img = processed_images[r][c]
                    if img is not None:
                        global_max_h = max(global_max_h, img.size[1])
            for r in range(rows):
                for c in range(cols):
                    img = processed_images[r][c]
                    if img is not None and img.size[1] < global_max_h:
                        scale = global_max_h / img.size[1]
                        processed_images[r][c] = img.resize(
                            (int(img.size[0] * scale), global_max_h), Image.LANCZOS)

    # Compute col_widths (widest image per column) and row_heights
    # (tallest image per row, after scaling to column width) in pixels.
    col_widths = [1.0] * cols
    for r in range(rows):
        for c in range(cols):
            img = processed_images[r][c]
            if img is not None:
                col_widths[c] = max(col_widths[c], img.size[0])

    row_heights = [1.0] * rows
    for r in range(rows):
        for c in range(cols):
            img = processed_images[r][c]
            if img is not None:
                row_heights[r] = max(row_heights[r], img.size[1])

    # Use a SINGLE pixel-to-inches scale so the figure aspect ratio
    # matches the actual content.  This prevents dead-space padding
    # around images when they are not 1:1.
    reference_inches = 3.0   # inches for the largest column
    px_per_inch = max(col_widths) / reference_inches


    gap_inches = cfg.spacing * reference_inches

    # Content dimensions in inches — derived from the same pixel scale
    col_inches = [w / px_per_inch for w in col_widths]
    row_inches = [h / px_per_inch for h in row_heights]
    content_w = sum(col_inches) + (cols - 1) * gap_inches
    content_h = sum(row_inches) + (rows - 1) * gap_inches

    # Figure dimensions: content + margins (all in inches)
    fig_w = content_w + left_margin_inches + right_margin_inches
    fig_h = content_h + top_margin_inches + bottom_margin_inches

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    # Convert inch-based margins to figure-fractions for axes placement
    left_margin = left_margin_inches / fig_w
    top_margin = top_margin_inches / fig_h
    right_margin = right_margin_inches / fig_w
    bottom_margin = bottom_margin_inches / fig_h

    # Manually position each axes for pixel-exact spacing.
    # All coordinates are in figure-fraction [0, 1].
    gap_w_frac = gap_inches / fig_w
    gap_h_frac = gap_inches / fig_h
    col_frac = [ci / fig_w for ci in col_inches]
    row_frac = [ri / fig_h for ri in row_inches]

    axes = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            x = left_margin + sum(col_frac[:c]) + c * gap_w_frac
            # y is from bottom; rows go top-to-bottom
            y = (1.0 - top_margin) - sum(row_frac[:r+1]) - r * gap_h_frac
            axes[r, c] = fig.add_axes([x, y, col_frac[c], row_frac[r]])

    # Pad ALL images to exactly (col_widths[c], row_heights[r]) then render
    # with aspect='auto'.  Since padded images exactly match cell proportions,
    # aspect='auto' introduces NO distortion — it just tells matplotlib to
    # fill the cell without the dead-space that aspect='equal' would add.
    # Images are NEVER stretched — only padded with the background colour.
    bg_color = (255, 255, 255) if cfg.background == "White" else (0, 0, 0)
    # Save original (pre-padded) image sizes for label positioning
    original_images = [[None]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            img = processed_images[r][c]
            original_images[r][c] = img  # save pre-padded
            target_w = int(col_widths[c])
            target_h = int(row_heights[r])
            if img is not None:
                iw, ih = img.size
                if iw != target_w or ih != target_h:
                    # Pad to exact cell size (centered), never scale
                    padded = Image.new("RGB", (target_w, target_h), bg_color)
                    ox = (target_w - iw) // 2
                    oy = (target_h - ih) // 2
                    padded.paste(img, (max(0, ox), max(0, oy)))
                    img = padded
                ax.imshow(img, aspect='auto')
            ax.set_aspect('auto')
            ax.axis("off")
            ax.set_frame_on(False)

    # Panel labels, symbols, and scale bars rendered via matplotlib AFTER normalize
    # so font sizes are absolute (immutable) and unaffected by image scaling.
    _add_panel_labels(fig, axes, cfg, rows, cols,
                      original_images, col_widths, row_heights)
    _add_panel_symbols(fig, axes, cfg, rows, cols, processed_images,
                       original_images=original_images)
    _add_panel_scale_bars(fig, axes, cfg, rows, cols, original_images,
                          pre_norm_sizes=pre_norm_sizes,
                          col_widths=col_widths, row_heights=row_heights)

    # Column/Row labels & headers (respect visibility toggles)
    show_col = getattr(cfg, 'show_column_labels', True)
    show_row = getattr(cfg, 'show_row_labels', True)
    if show_col:
        _add_column_labels(fig, axes, cfg.column_labels, rows, cols)
    if show_row:
        _add_row_labels(fig, axes, cfg.row_labels, rows, cols)
    has_col_labels = show_col and any(lbl.text.strip() for lbl in cfg.column_labels)
    has_row_labels = show_row and any(lbl.text.strip() for lbl in cfg.row_labels)
    _add_column_headers(fig, axes, cfg.column_headers, rows, cols, has_col_labels,
                        column_labels=cfg.column_labels if has_col_labels else None)
    _add_row_headers(fig, axes, cfg.row_headers, rows, cols, has_row_labels,
                     row_labels=cfg.row_labels if has_row_labels else None)

    # Background
    if cfg.background == "Transparent":
        fig.patch.set_alpha(0.0)
    elif cfg.background == "Black":
        fig.patch.set_facecolor("black")
    elif cfg.background.startswith("#"):
        fig.patch.set_facecolor(cfg.background)
    else:
        fig.patch.set_facecolor("white")

    # Capture axes positions BEFORE closing the figure
    axes_positions = {}
    for r_ax in range(rows):
        for c_ax in range(cols):
            ax_obj = axes[r_ax, c_ax] if axes.ndim == 2 else axes[r_ax * cols + c_ax]
            pos = ax_obj.get_position()
            axes_positions[(r_ax, c_ax)] = (pos.x0, pos.y0, pos.x1, pos.y1)

    buf = io.BytesIO()
    fmt = "tiff" if cfg.output_format == "TIFF" else "png"
    fig.savefig(buf, format=fmt, dpi=dpi,
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    # Post-render: draw Adjacent Panel zoom connecting lines via PIL
    buf.seek(0)
    has_adj_zoom = any(
        cfg.panels[r][c].add_zoom_inset and cfg.panels[r][c].zoom_inset
        and cfg.panels[r][c].zoom_inset.inset_type == "Adjacent Panel"
        for r in range(rows) for c in range(cols)
    )
    if has_adj_zoom:
        out_img = Image.open(buf).convert("RGB")
        ow, oh = out_img.size
        draw_out = ImageDraw.Draw(out_img)

        for r in range(rows):
            for c in range(cols):
                panel = cfg.panels[r][c]
                if not panel.add_zoom_inset or not panel.zoom_inset:
                    continue
                zi = panel.zoom_inset
                if zi.inset_type != "Adjacent Panel":
                    continue
                ar, ac = r, c
                side = zi.side or "Right"
                if side == "Top": ar -= 1
                elif side == "Bottom": ar += 1
                elif side == "Left": ac -= 1
                elif side == "Right": ac += 1
                if not (0 <= ar < rows and 0 <= ac < cols):
                    continue

                sp = axes_positions[(r, c)]  # (x0, y0, x1, y1) in figure fraction
                tp = axes_positions[(ar, ac)]

                # Helper: convert (frac_x, frac_y) within axes bbox to output pixels
                # frac_x: 0=left, 1=right of axes
                # frac_y: 0=top, 1=bottom of axes (imshow convention)
                def _ax_to_px(bbox, fx, fy):
                    fig_x = bbox[0] + fx * (bbox[2] - bbox[0])
                    fig_y = bbox[3] - fy * (bbox[3] - bbox[1])  # bbox[3]=y1=top, fy=0→top
                    return int(fig_x * ow), int((1 - fig_y) * oh)

                # Source rect as fraction of source panel image
                src_img = original_images[r][c]
                siw = src_img.size[0] if src_img else 1
                sih = src_img.size[1] if src_img else 1
                sfx1 = zi.x / siw
                sfy1 = zi.y / sih
                sfx2 = (zi.x + zi.width) / siw
                sfy2 = (zi.y + zi.height) / sih

                # Target: zoomed image fills the entire target axes (no separate padding calc needed)
                # Just use 0,0 to 1,1

                # Source corners → output pixels
                s_tr = _ax_to_px(sp, sfx2, sfy1)  # top-right of source rect
                s_br = _ax_to_px(sp, sfx2, sfy2)  # bottom-right of source rect

                # Target corners → output pixels
                t_tl = _ax_to_px(tp, 0, 0)  # top-left of target
                t_bl = _ax_to_px(tp, 0, 1)  # bottom-left of target

                zcolor = zi.line_color or "#FF0000"
                bg2 = (cfg.background or "white").lower()
                if zcolor.lower() in ("#ffffff", "white", "#fff") and bg2 in ("white", "#ffffff", "#fff"):
                    zcolor = "#333333"
                zlw = max(1, zi.line_width)

                # Compute source rect center position in the output image
                # Use axes data limits (from imshow) — these reflect the ACTUAL
                # displayed image dimensions including any padding
                src_ax_obj = axes[r, c] if axes.ndim == 2 else axes[r * cols + c]
                xlim = src_ax_obj.get_xlim()
                ylim = src_ax_obj.get_ylim()
                # Axes data range (imshow: ylim is inverted, y0=top)
                data_w = abs(xlim[1] - xlim[0])
                data_h = abs(ylim[0] - ylim[1])
                # Full-res dimensions for zi coords
                frs = full_res_sizes or {}
                full_w, full_h = frs.get((r, c), (data_w, data_h))
                # zi center in data coordinates
                zi_data_x = (zi.x + zi.width / 2) * data_w / max(full_w, 1)
                zi_data_y = (zi.y + zi.height / 2) * data_h / max(full_h, 1)
                # Account for padding: if data range > original image, image is centered
                oi = original_images[r][c]
                oiw = oi.size[0] if oi else data_w
                oih = oi.size[1] if oi else data_h
                # Padding in data coords
                data_pad_x = (data_w - oiw * data_w / max(full_w, 1) * full_w / max(oiw, 1)) / 2 if full_w != oiw else (data_w - data_w) / 2
                # Simpler: just compute fraction of axes
                sel_fx = zi_data_x / data_w
                sel_fy = zi_data_y / data_h
                # But if image is padded in axes (data_h > image proportion),
                # the image starts at an offset
                img_data_h = oih  # original image height in data coords
                if data_h > img_data_h:
                    pad_data_y = (data_h - img_data_h) / 2
                    zi_data_y_padded = pad_data_y + (zi.y + zi.height / 2) * img_data_h / max(full_h, 1)
                    sel_fy = zi_data_y_padded / data_h
                img_data_w = oiw
                if data_w > img_data_w:
                    pad_data_x = (data_w - img_data_w) / 2
                    zi_data_x_padded = pad_data_x + (zi.x + zi.width / 2) * img_data_w / max(full_w, 1)
                    sel_fx = zi_data_x_padded / data_w
                # Helper: convert axes fraction to output pixel
                def _frac_to_px(bbox, fx, fy):
                    px_x = int((bbox[0] + fx * (bbox[2] - bbox[0])) * ow)
                    f_y = bbox[3] - fy * (bbox[3] - bbox[1])
                    px_y = int((1 - f_y) * oh)
                    return px_x, px_y

                # Compute source rect CORNERS using same padding-aware logic
                def _zi_to_frac(zx, zy):
                    fx = zx * data_w / max(full_w, 1) / data_w
                    fy = zy * data_h / max(full_h, 1) / data_h
                    if data_h > img_data_h * 1.01:
                        p_y = (data_h - img_data_h) / 2
                        fy = (p_y + zy * img_data_h / max(full_h, 1)) / data_h
                    if data_w > img_data_w * 1.01:
                        p_x = (data_w - img_data_w) / 2
                        fx = (p_x + zx * img_data_w / max(full_w, 1)) / data_w
                    return fx, fy

                s_tl_f = _zi_to_frac(zi.x, zi.y)
                s_tr_f = _zi_to_frac(zi.x + zi.width, zi.y)
                s_bl_f = _zi_to_frac(zi.x, zi.y + zi.height)
                s_br_f = _zi_to_frac(zi.x + zi.width, zi.y + zi.height)

                s_tr_px = _frac_to_px(sp, s_tr_f[0], s_tr_f[1])
                s_br_px = _frac_to_px(sp, s_br_f[0], s_br_f[1])
                s_tl_px = _frac_to_px(sp, s_tl_f[0], s_tl_f[1])
                s_bl_px = _frac_to_px(sp, s_bl_f[0], s_bl_f[1])

                # Target: compute zoomed image position within target axes
                # (same padding-aware logic as source)
                tgt_ax_obj = axes[ar, ac] if axes.ndim == 2 else axes[ar * cols + ac]
                t_xlim = tgt_ax_obj.get_xlim()
                t_ylim = tgt_ax_obj.get_ylim()
                t_data_w = abs(t_xlim[1] - t_xlim[0])
                t_data_h = abs(t_ylim[0] - t_ylim[1])
                tgt_oi = original_images[ar][ac]
                t_img_w = tgt_oi.size[0] if tgt_oi else t_data_w
                t_img_h = tgt_oi.size[1] if tgt_oi else t_data_h
                # Padding fractions within target axes
                t_pad_fx = max(0, ((t_data_w - t_img_w) / 2) / t_data_w) if t_data_w > t_img_w else 0
                t_img_fx = t_img_w / t_data_w if t_data_w > t_img_w else 1
                t_pad_fy = max(0, ((t_data_h - t_img_h) / 2) / t_data_h) if t_data_h > t_img_h else 0
                t_img_fy = t_img_h / t_data_h if t_data_h > t_img_h else 1
                # Target image corners within the padded axes
                t_tl_px = _frac_to_px(tp, t_pad_fx, t_pad_fy)
                t_tr_px = _frac_to_px(tp, t_pad_fx + t_img_fx, t_pad_fy)
                t_bl_px = _frac_to_px(tp, t_pad_fx, t_pad_fy + t_img_fy)
                t_br_px = _frac_to_px(tp, t_pad_fx + t_img_fx, t_pad_fy + t_img_fy)

                # Draw border around zoomed image in target panel
                border_color = zi.rectangle_color or zcolor
                bg_lower = (cfg.background or "white").lower()
                rect_lower = border_color.lower()
                if rect_lower in ("#ffffff", "white", "#fff") and bg_lower in ("white", "#ffffff", "#fff"):
                    border_color = "#000000"
                # Use rectangle for uniform border thickness on all sides
                draw_out.rectangle(
                    [t_tl_px, t_br_px],
                    outline=border_color, width=zlw
                )

                if side == "Right":
                    draw_out.line([s_tr_px, t_tl_px], fill=zcolor, width=zlw)
                    draw_out.line([s_br_px, t_bl_px], fill=zcolor, width=zlw)
                elif side == "Left":
                    draw_out.line([s_tl_px, t_tr_px], fill=zcolor, width=zlw)
                    draw_out.line([s_bl_px, t_br_px], fill=zcolor, width=zlw)
                elif side == "Top":
                    draw_out.line([s_tl_px, t_bl_px], fill=zcolor, width=zlw)
                    draw_out.line([s_tr_px, t_br_px], fill=zcolor, width=zlw)
                else:  # Bottom
                    draw_out.line([s_bl_px, t_tl_px], fill=zcolor, width=zlw)
                    draw_out.line([s_br_px, t_tr_px], fill=zcolor, width=zlw)

        buf = io.BytesIO()
        out_img.save(buf, format="TIFF" if fmt == "tiff" else "PNG")

    buf.seek(0)
    return buf.read()
