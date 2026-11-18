"""Human-readable panel names for the Analysis workspace.

A panel used to be called "R8C1" everywhere in analysis. The figure already
says what each row and column IS — its primary label plus any secondary
headers stacked above/beside it — so a panel inherits those instead:

    row chain  = outermost row header › … › row label
    col chain  = outermost column header › … › column label
    name       = "<row chain>, <col chain>"        e.g. "1hr, Sample 1"
                                                        "Day 2 › 1hr, KO › Sample 1"

Names are computed from the live config on every request, so renaming a
header renames the source. They are DISPLAY names only: source keys, CSV
`Panel` columns and saved workflows keep using grid coordinates.

Pure functions, no sidecar state, so they're unit-testable on their own.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

# Same markup the renderer parses (image_processing._COLOR_RE): ::RRGGBB::text::
_COLOR_RE = re.compile(r"::([0-9A-Fa-f]{6})::(.+?)::", re.DOTALL)
_WS_RE = re.compile(r"\s+")

LEVEL_SEP = " › "
AXIS_SEP = ", "


def plain_text(obj) -> str:
    """The words a label/header shows, without styling: styled segments when
    present (that's what the renderer draws), else `text` with colour markup
    stripped. Line breaks and runs of whitespace collapse to one space."""
    segs = getattr(obj, "styled_segments", None) or []
    raw = "".join(getattr(s, "text", "") or "" for s in segs) if segs else (getattr(obj, "text", "") or "")
    raw = _COLOR_RE.sub(lambda m: m.group(2), raw)
    return _WS_RE.sub(" ", raw).strip()


def axis_chain(labels: Sequence, header_levels: Sequence, idx: int, show_labels: bool = True) -> List[str]:
    """Outermost → innermost names that cover row/column `idx`: every header
    level with a non-empty header spanning it, then the primary label when
    labels are shown. Consecutive duplicates collapse ("A › A" → "A")."""
    out: List[str] = []
    for level in header_levels or []:
        for hdr in getattr(level, "headers", None) or []:
            if idx in (getattr(hdr, "columns_or_rows", None) or []):
                t = plain_text(hdr)
                if t:
                    out.append(t)
                break  # one header per level covers an index
    if show_labels and labels is not None and 0 <= idx < len(labels):
        lbl = labels[idx]
        if getattr(lbl, "visible", True):
            t = plain_text(lbl)
            if t:
                out.append(t)
    dedup: List[str] = []
    for t in out:
        if not dedup or dedup[-1] != t:
            dedup.append(t)
    return dedup


def coord_name(r: int, c: int) -> str:
    return f"R{r + 1}C{c + 1}"


def panel_display_names(cfg) -> Dict[Tuple[int, int], str]:
    """{(row, col): display name} for every grid cell of `cfg`.

    Falls back to "R#C#" when a cell has no row or column names at all, and
    appends " (R#C#)" to every cell that shares its name with another, so two
    panels never read the same in a picker or a CSV."""
    rows = int(getattr(cfg, "rows", 0) or 0)
    cols = int(getattr(cfg, "cols", 0) or 0)
    show_col = bool(getattr(cfg, "show_column_labels", True))
    show_row = bool(getattr(cfg, "show_row_labels", True))
    col_labels = list(getattr(cfg, "column_labels", None) or [])
    row_labels = list(getattr(cfg, "row_labels", None) or [])
    col_headers = list(getattr(cfg, "column_headers", None) or [])
    row_headers = list(getattr(cfg, "row_headers", None) or [])

    row_names = [LEVEL_SEP.join(axis_chain(row_labels, row_headers, r, show_row)) for r in range(rows)]
    col_names = [LEVEL_SEP.join(axis_chain(col_labels, col_headers, c, show_col)) for c in range(cols)]

    names: Dict[Tuple[int, int], str] = {}
    for r in range(rows):
        for c in range(cols):
            parts = [p for p in (row_names[r], col_names[c]) if p]
            names[(r, c)] = AXIS_SEP.join(parts) if parts else coord_name(r, c)

    counts: Dict[str, int] = {}
    for nm in names.values():
        counts[nm] = counts.get(nm, 0) + 1
    for (r, c), nm in list(names.items()):
        if counts[nm] > 1 and nm != coord_name(r, c):
            names[(r, c)] = f"{nm} ({coord_name(r, c)})"
    return names


def panel_name(names: Optional[Dict[Tuple[int, int], str]], r: int, c: int) -> str:
    return (names or {}).get((r, c)) or coord_name(r, c)
