"""Local test harness for figure_builder rendering.

Run: python3 test_rendering.py
Outputs test PNGs to /tmp/mpf-tests/ that you can open and visually verify.
Each test documents what the output should look like.
"""
import os
import sys
from pathlib import Path
from PIL import Image

# Matplotlib backend for headless rendering
os.environ.setdefault("MPLBACKEND", "Agg")

from figure_builder import assemble_figure
from models import (
    FigureConfig,
    PanelInfo,
    AxisLabel,
    HeaderLevel,
    HeaderGroup,
    StyledSegment,
)

OUT_DIR = Path("/tmp/mpf-tests")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def placeholder_image(w: int = 400, h: int = 300, label: str = "") -> Image.Image:
    """A coloured placeholder so we can see panel boundaries."""
    img = Image.new("RGB", (w, h), (230, 230, 245))
    if label:
        try:
            from PIL import ImageDraw
            d = ImageDraw.Draw(img)
            d.text((10, 10), label, fill=(80, 80, 120))
        except Exception:
            pass
    return img


def make_config(
    rows=2, cols=2,
    col_header_texts=("Column 1", "Column 2"),
    col_header_segs=(None, None),
    row_header_texts=("Row 1", "Row 2"),
    row_header_segs=(None, None),
    col_label_texts=("", ""),
    row_label_texts=("", ""),
):
    """Build a FigureConfig with one header level per axis."""
    cfg = FigureConfig()
    cfg.rows = rows
    cfg.cols = cols
    cfg.ensure_grid()

    # Column headers — one HeaderGroup per column at level 0
    col_level = HeaderLevel()
    for c in range(cols):
        hg = HeaderGroup()
        hg.text = col_header_texts[c] if c < len(col_header_texts) else f"Col {c+1}"
        hg.columns_or_rows = [c]
        hg.font_size = 12
        hg.default_color = "#000000"
        hg.line_width = 0  # no bracket line for this test
        if c < len(col_header_segs) and col_header_segs[c] is not None:
            hg.styled_segments = col_header_segs[c]
        col_level.headers.append(hg)
    cfg.column_headers = [col_level]

    # Row headers — one HeaderGroup per row at level 0, rotated 90° (vertical)
    row_level = HeaderLevel()
    for r in range(rows):
        hg = HeaderGroup()
        hg.text = row_header_texts[r] if r < len(row_header_texts) else f"Row {r+1}"
        hg.columns_or_rows = [r]
        hg.font_size = 12
        hg.rotation = 90.0
        hg.position = "Left"
        hg.default_color = "#000000"
        hg.line_width = 0
        if r < len(row_header_segs) and row_header_segs[r] is not None:
            hg.styled_segments = row_header_segs[r]
        row_level.headers.append(hg)
    cfg.row_headers = [row_level]

    # Column / row labels
    cfg.column_labels = [AxisLabel(text=col_label_texts[i] if i < len(col_label_texts) else "")
                         for i in range(cols)]
    cfg.row_labels = [AxisLabel(text=row_label_texts[i] if i < len(row_label_texts) else "")
                      for i in range(rows)]

    # Defaults
    cfg.background = "White"
    cfg.spacing = 0.02
    cfg.output_format = "PNG"
    return cfg


def make_processed(rows, cols, with_images=True):
    if not with_images:
        return [[None for _ in range(cols)] for _ in range(rows)]
    return [[placeholder_image(400, 300, f"{r},{c}") for c in range(cols)] for r in range(rows)]


def run(name, cfg, with_images=True):
    """Render cfg to PNG and save to OUT_DIR. Returns path."""
    processed = make_processed(cfg.rows, cfg.cols, with_images=with_images)
    data = assemble_figure(cfg, processed, dpi=150)
    out = OUT_DIR / f"{name}.png"
    out.write_bytes(data)
    print(f"  wrote {out} ({len(data)} bytes)")
    return out


def test_plain_headers():
    """Expected: 'Column 1' / 'Column 2' at the top, 'Row 1' / 'Row 2' on the left (rotated).
       All in black, no per-char styling, no wrapping."""
    print("test_plain_headers")
    cfg = make_config()
    run("01-plain", cfg)


def test_column_with_per_char_color():
    """Expected: 'Col' in black, 'umn 1' in magenta (column 1). Column 2 plain black.
       Both on SINGLE line — should match each other in position / line count."""
    print("test_column_with_per_char_color")
    seg = [
        StyledSegment(text="Col", color="#000000"),
        StyledSegment(text="umn 1", color="#ff00ff"),
    ]
    cfg = make_config(col_header_segs=(seg, None))
    run("02-col-styled", cfg)


def test_both_columns_same_length_one_styled():
    """Regression for the user-reported bug where Column 1 (styled) stayed on
       one line while Column 2 (plain) was auto-wrapped even though both
       have identical text lengths. After fix they should behave the same."""
    print("test_both_columns_same_length_one_styled")
    seg = [
        StyledSegment(text="Col", color="#000000"),
        StyledSegment(text="umn 1", color="#ff00ff"),
    ]
    cfg = make_config(col_header_segs=(seg, None))
    # Render without panel images — narrows columns so auto-wrap is tempted.
    run("03-narrow-cols", cfg, with_images=False)


def test_row_per_char_color():
    """Row 1: 'Row 1' rotated 90°, with 'R' in red.
       Row 2: 'Row 2' plain black, rotated 90°.
       Both should remain vertical (not go horizontal) and same vertical position."""
    print("test_row_per_char_color")
    seg = [
        StyledSegment(text="R", color="#ff0000"),
        StyledSegment(text="ow 1", color="#000000"),
    ]
    cfg = make_config(row_header_segs=(seg, None))
    run("04-row-styled", cfg)


def test_long_header_wrapping():
    """Long column header should auto-wrap to multiple lines."""
    print("test_long_header_wrapping")
    cfg = make_config(col_header_texts=("A very long header that should wrap nicely", "Short"))
    run("05-long-wrap", cfg)


def test_explicit_newline_in_header():
    """Header with explicit \\n should render on multiple lines."""
    print("test_explicit_newline_in_header")
    cfg = make_config(col_header_texts=("Line 1\nLine 2", "Simple"))
    run("06-explicit-newline", cfg)


def test_explicit_newline_with_per_char():
    """Explicit \\n + per-char styling: both should work together."""
    print("test_explicit_newline_with_per_char")
    seg = [
        StyledSegment(text="Red", color="#ff0000"),
        StyledSegment(text="\nBlue", color="#0000ff"),
    ]
    cfg = make_config(col_header_texts=("Red\nBlue", "Normal"), col_header_segs=(seg, None))
    run("07-newline-plus-styling", cfg)


if __name__ == "__main__":
    tests = [
        test_plain_headers,
        test_column_with_per_char_color,
        test_both_columns_same_length_one_styled,
        test_row_per_char_color,
        test_long_header_wrapping,
        test_explicit_newline_in_header,
        test_explicit_newline_with_per_char,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nAll tests written to {OUT_DIR}")
    print("Open the PNGs to visually verify each test.")
