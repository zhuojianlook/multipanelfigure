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


def test_long_row_wrapping():
    """Long ROW header should auto-wrap. Rotated 90° so wrapping means stacked
       lines side-by-side along the rotated baseline (perpendicular to text direction)."""
    print("test_long_row_wrapping")
    cfg = make_config(
        row_header_texts=("A very long row header that should wrap nicely", "Short"),
    )
    run("08-long-row-wrap", cfg)


def test_explicit_newline_in_row():
    """Row header with explicit \\n — still rotated 90°. The two lines should both be
       vertical and visibly separated (stacked perpendicular to text baseline)."""
    print("test_explicit_newline_in_row")
    cfg = make_config(row_header_texts=("Line 1\nLine 2", "Simple"))
    run("09-row-explicit-newline", cfg)


def test_explicit_newline_with_per_char_row():
    """Row header: 'Red\\nBlue' with per-char color, rotated 90°. Should show 'Red' in
       red on one rotated line, 'Blue' in blue on a second rotated line next to it.
       This is the row analogue of test 07."""
    print("test_explicit_newline_with_per_char_row")
    seg = [
        StyledSegment(text="Red", color="#ff0000"),
        StyledSegment(text="\nBlue", color="#0000ff"),
    ]
    cfg = make_config(row_header_texts=("Red\nBlue", "Normal"), row_header_segs=(seg, None))
    run("10-row-newline-plus-styling", cfg)


def test_row_both_same_length_one_styled():
    """Row analogue of test 03: two rows, same text length, one styled one plain.
       Both should render as single rotated lines at matching positions."""
    print("test_row_both_same_length_one_styled")
    seg = [
        StyledSegment(text="R", color="#ff0000"),
        StyledSegment(text="ow 1", color="#000000"),
    ]
    # with_images=False narrows the panel area so auto-wrap could be tempted
    cfg = make_config(row_header_segs=(seg, None))
    run("11-narrow-rows", cfg, with_images=False)


# ---------------------------------------------------------------------------
# Multiple line breaks + edge positions (leading / trailing / consecutive)
# ---------------------------------------------------------------------------

def test_four_lines_col_plain():
    """4 plain lines in a column header. Expected: L1, L2, L3, L4 stacked top→bottom,
       equal spacing, all black."""
    print("test_four_lines_col_plain")
    cfg = make_config(col_header_texts=("L1\nL2\nL3\nL4", "Plain"))
    run("12-four-lines-col", cfg)


def test_four_lines_col_styled():
    """4 styled lines in a column header. Expected: each line its own color,
       stacked top→bottom."""
    print("test_four_lines_col_styled")
    seg = [
        StyledSegment(text="Red", color="#ff0000"),
        StyledSegment(text="\nGreen", color="#00aa00"),
        StyledSegment(text="\nBlue", color="#0000ff"),
        StyledSegment(text="\nPurple", color="#aa00aa"),
    ]
    cfg = make_config(col_header_texts=("Red\nGreen\nBlue\nPurple", "Plain"),
                      col_header_segs=(seg, None))
    run("13-four-lines-col-styled", cfg)


def test_leading_newline_col_styled():
    """Styled column header starting with a newline (blank line first).
       Expected: blank line on top, then 'Hello' in red."""
    print("test_leading_newline_col_styled")
    seg = [
        StyledSegment(text="\nHello", color="#ff0000"),
    ]
    cfg = make_config(col_header_texts=("\nHello", "Plain"), col_header_segs=(seg, None))
    run("14-leading-newline-col", cfg)


def test_trailing_newline_col_styled():
    """Styled column header ending with a newline (blank line at bottom).
       Expected: 'Hello' in red, then a blank line under it (reserved space)."""
    print("test_trailing_newline_col_styled")
    seg = [
        StyledSegment(text="Hello\n", color="#ff0000"),
    ]
    cfg = make_config(col_header_texts=("Hello\n", "Plain"), col_header_segs=(seg, None))
    run("15-trailing-newline-col", cfg)


def test_consecutive_newlines_col_styled():
    """Styled column with 'a\\n\\n\\nb' — two empty lines between a and b.
       Expected: 'a' red, two blank lines, 'b' blue — total 4 line heights."""
    print("test_consecutive_newlines_col_styled")
    seg = [
        StyledSegment(text="a", color="#ff0000"),
        StyledSegment(text="\n\n\nb", color="#0000ff"),
    ]
    cfg = make_config(col_header_texts=("a\n\n\nb", "Plain"), col_header_segs=(seg, None))
    run("16-consecutive-newlines-col", cfg)


def test_four_lines_row_plain():
    """4 plain lines in a rotated ROW header. Expected: L1, L2, L3, L4 stacked
       perpendicular to the rotated baseline (i.e. side-by-side when looking at the figure)."""
    print("test_four_lines_row_plain")
    cfg = make_config(row_header_texts=("L1\nL2\nL3\nL4", "Plain"))
    run("17-four-lines-row", cfg)


def test_four_lines_row_styled():
    """4 styled lines in a rotated row header. Each line its own color."""
    print("test_four_lines_row_styled")
    seg = [
        StyledSegment(text="Red", color="#ff0000"),
        StyledSegment(text="\nGreen", color="#00aa00"),
        StyledSegment(text="\nBlue", color="#0000ff"),
        StyledSegment(text="\nPurple", color="#aa00aa"),
    ]
    cfg = make_config(row_header_texts=("Red\nGreen\nBlue\nPurple", "Plain"),
                      row_header_segs=(seg, None))
    run("18-four-lines-row-styled", cfg)


def test_leading_newline_row_styled():
    """Rotated row header starting with newline. Expected: blank first line, then 'Hello' red."""
    print("test_leading_newline_row_styled")
    seg = [
        StyledSegment(text="\nHello", color="#ff0000"),
    ]
    cfg = make_config(row_header_texts=("\nHello", "Plain"), row_header_segs=(seg, None))
    run("19-leading-newline-row", cfg)


def test_trailing_newline_row_styled():
    """Rotated row header ending with newline. Expected: 'Hello' red, blank line after."""
    print("test_trailing_newline_row_styled")
    seg = [
        StyledSegment(text="Hello\n", color="#ff0000"),
    ]
    cfg = make_config(row_header_texts=("Hello\n", "Plain"), row_header_segs=(seg, None))
    run("20-trailing-newline-row", cfg)


def test_consecutive_newlines_row_styled():
    """Rotated row with 'a\\n\\n\\nb' per-char. 4 lines, 2 blank. Both end chars stay rotated."""
    print("test_consecutive_newlines_row_styled")
    seg = [
        StyledSegment(text="a", color="#ff0000"),
        StyledSegment(text="\n\n\nb", color="#0000ff"),
    ]
    cfg = make_config(row_header_texts=("a\n\n\nb", "Plain"), row_header_segs=(seg, None))
    run("21-consecutive-newlines-row", cfg)


def test_triple_newline_clip_repro():
    """Repro for 'triple linebreaks clip off top' bug.  User said 'Colu\\n\\n\\nmn 1' gets
       its top lines clipped off the figure border. Use a larger font size (like the real
       app default of 12) and per-char style to exercise the multi-styled path."""
    print("test_triple_newline_clip_repro")
    seg = [
        StyledSegment(text="Colu", color="#000000", font_size=12),
        StyledSegment(text="\n\n\nmn 1", color="#0000ff", font_size=12),
    ]
    cfg = make_config(col_header_texts=("Colu\n\n\nmn 1", "Plain"),
                      col_header_segs=(seg, None))
    # Bump font size on both headers to match app default
    for c in range(cfg.cols):
        cfg.column_headers[0].headers[c].font_size = 12
    run("22-triple-newline-clip-repro", cfg)


def test_color_change_position_shift_repro():
    """Repro for 'color changes cause the position to shift' bug. A single-line header
       where the first half is one color and the second half is another should lay out
       as if it were one continuous line (no visible gap at the color boundary)."""
    print("test_color_change_position_shift_repro")
    # Deliberately split at a point where width estimate matters most.
    seg = [
        StyledSegment(text="Colu", color="#000000", font_size=12),
        StyledSegment(text="mn 2", color="#ff00ff", font_size=12),
    ]
    cfg = make_config(col_header_segs=(None, seg))
    for c in range(cfg.cols):
        cfg.column_headers[0].headers[c].font_size = 12
    run("23-color-change-shift-repro", cfg)


def test_styled_multiline_row_vs_plain_row_alignment():
    """Repro for user's reported 'secondary headers with color customization
       appear to be not correct position' issue (the preview image screenshot
       showed a 2-line styled row header NOT centered on the same y as its
       plain-single-line sibling in the same tier).

       Setup: two row headers at the same tier, same panel bbox:
         Row 0 header: "Red\\nBlue" with red+blue per-char styling
         Row 1 header: "Plain"    — single line, single color
       Expected: BOTH rotated rows centered on their panel's mid-y, and the
       *visual center* of the 2-line stack should land at exactly the same
       y position as the plain single line would (their midpoints match)."""
    print("test_styled_multiline_row_vs_plain_row_alignment")
    seg = [
        StyledSegment(text="Red", color="#ff0000", font_size=12),
        StyledSegment(text="\nBlue", color="#0000ff", font_size=12),
    ]
    cfg = make_config(
        row_header_texts=("Red\nBlue", "Plain"),
        row_header_segs=(seg, None),
    )
    # Bump font size to match app default
    for r in range(cfg.rows):
        cfg.row_headers[0].headers[r].font_size = 12
    run("24-styled-multiline-row-vs-plain", cfg)


def test_styled_multiline_header_with_label_alignment():
    """Follow-up to test 24: a multi-line STYLED row header at outer tier, with
       a plain row LABEL in the inner lane. Both should appear centered on the
       same panel row's mid-y. If they drift, _count_header_lines space
       reservation vs. the perpendicular offset formula is out of sync."""
    print("test_styled_multiline_header_with_label_alignment")
    seg = [
        StyledSegment(text="Red", color="#ff0000", font_size=12),
        StyledSegment(text="\nBlue", color="#0000ff", font_size=12),
    ]
    cfg = make_config(
        row_header_texts=("Red\nBlue", ""),  # only first row has a header
        row_header_segs=(seg, None),
        row_label_texts=("Row 1", "Row 2"),
    )
    for r in range(cfg.rows):
        cfg.row_headers[0].headers[r].font_size = 12
    for r in range(cfg.rows):
        cfg.row_labels[r].font_size = 12
        cfg.row_labels[r].rotation = 90.0
    run("25-styled-header-plus-label-align", cfg)


if __name__ == "__main__":
    tests = [
        test_plain_headers,
        test_column_with_per_char_color,
        test_both_columns_same_length_one_styled,
        test_row_per_char_color,
        test_long_header_wrapping,
        test_explicit_newline_in_header,
        test_explicit_newline_with_per_char,
        test_long_row_wrapping,
        test_explicit_newline_in_row,
        test_explicit_newline_with_per_char_row,
        test_row_both_same_length_one_styled,
        test_four_lines_col_plain,
        test_four_lines_col_styled,
        test_leading_newline_col_styled,
        test_trailing_newline_col_styled,
        test_consecutive_newlines_col_styled,
        test_four_lines_row_plain,
        test_four_lines_row_styled,
        test_leading_newline_row_styled,
        test_trailing_newline_row_styled,
        test_consecutive_newlines_row_styled,
        test_triple_newline_clip_repro,
        test_color_change_position_shift_repro,
        test_styled_multiline_row_vs_plain_row_alignment,
        test_styled_multiline_header_with_label_alignment,
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
