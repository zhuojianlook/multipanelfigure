"""The planner's Pivot (rows <-> columns) as the sidecar receives it.

The transpose itself is frontend-only (src/utils/pivotGrid.ts) and reaches
the sidecar as ONE ``PUT /api/config`` of the pivoted config: there is no
Python transpose by design. This is the contract test for that PUT, over the
fixture the frontend's own tests use (``before`` hand-written, ``after``
generated from it by pivotFigureConfig):

  * ``put_config`` runs ``ensure_grid`` on what it receives — it must be a
    no-op (nothing trimmed or padded), or the PUT would lose a row/column;
  * the renderer reads the swapped labels, headers and inset sides the way
    the planner meant them.
"""

import json
import os
import re

from models import FigureConfig, _from_dict, _to_dict
from panel_naming import AXIS_SEP, panel_display_names

_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "src", "utils", "__fixtures__pivotGrid.json")
with open(_FIXTURE, encoding="utf-8") as f:
    _DATA = json.load(f)
BEFORE = _DATA["before"]
AFTER = _DATA["after"]


def _cfg(d) -> FigureConfig:
    return _from_dict(FigureConfig, json.loads(json.dumps(d)))


def _adjacent_target(r, c, side):
    """Where an Adjacent Panel inset at (r, c) draws — the renderer's rule
    (figure_builder.py, the Adjacent Panel pass; api_server's copy agrees):
    an empty side means Right."""
    side = side or "Right"
    if side == "Top":
        return r - 1, c
    if side == "Bottom":
        return r + 1, c
    if side == "Left":
        return r, c - 1
    if side == "Right":
        return r, c + 1
    return r, c


def _adjacent_insets(cfg):
    """(row, col, side) of every Adjacent Panel inset, as the renderer walks
    them: the zoom_insets list, else the legacy single inset."""
    out = []
    for r, row in enumerate(cfg.panels):
        for c, p in enumerate(row):
            if not p.add_zoom_inset:
                continue
            insets = p.zoom_insets if p.zoom_insets else ([p.zoom_inset] if p.zoom_inset else [])
            for zi in insets:
                if zi.inset_type == "Adjacent Panel":
                    out.append((r, c, zi.side))
    return out


def test_put_is_lossless_ensure_grid_changes_nothing():
    cfg = _cfg(AFTER)
    as_sent = _to_dict(cfg)
    cfg.ensure_grid()
    assert _to_dict(cfg) == as_sent
    assert (cfg.rows, cfg.cols) == (3, 2)
    assert len(cfg.panels) == 3 and all(len(row) == 2 for row in cfg.panels)
    assert len(cfg.column_labels) == 2 and len(cfg.row_labels) == 3


def test_panels_are_transposed():
    b, a = _cfg(BEFORE), _cfg(AFTER)
    assert (a.rows, a.cols) == (b.cols, b.rows)
    for r in range(b.rows):
        for c in range(b.cols):
            assert a.panels[c][r].image_name == b.panels[r][c].image_name


def test_labels_and_headers_swap_axes_keeping_level_order():
    b, a = _cfg(BEFORE), _cfg(AFTER)
    assert [l.text for l in a.column_labels] == [l.text for l in b.row_labels]
    assert [l.text for l in a.row_labels] == [l.text for l in b.column_labels]
    as_spans = lambda levels: [[(g.text, list(g.columns_or_rows)) for g in lv.headers] for lv in levels]
    assert as_spans(a.row_headers) == as_spans(b.column_headers)
    assert as_spans(a.column_headers) == as_spans(b.row_headers)
    assert (a.show_column_labels, a.show_row_labels) == (b.show_row_labels, b.show_column_labels)


def test_every_side_is_one_its_axis_draws():
    # The renderer draws a column label / header that is not "Top" at the
    # Bottom, and a row one that is not "Left" at the Right — a "Left" left
    # on a column header would silently move it. After the pivot each side
    # is a real side of its own axis.
    for cfg in (_cfg(BEFORE), _cfg(AFTER)):
        cols = [l.position for l in cfg.column_labels]
        cols += [g.position for lv in cfg.column_headers for g in lv.headers]
        rows = [l.position for l in cfg.row_labels]
        rows += [g.position for lv in cfg.row_headers for g in lv.headers]
        assert set(cols) <= {"Top", "Bottom"}, cols
        assert set(rows) <= {"Left", "Right"}, rows


_COORD_SUFFIX = re.compile(r" \(R\d+C\d+\)$")


def _swap_axis_parts(name):
    """"<row chain>, <col chain>" read the other way round. Neither chain
    holds ", " in this fixture."""
    parts = name.split(AXIS_SEP)
    assert len(parts) == 2, name
    return AXIS_SEP.join(reversed(parts))


def test_display_names_follow_the_panels():
    b_names = panel_display_names(_cfg(BEFORE))
    a_names = panel_display_names(_cfg(AFTER))
    b = _cfg(BEFORE)
    for r in range(b.rows):
        for c in range(b.cols):
            # Duplicate names carry their cell, "(R#C#)" — of the cell the
            # panel is in now, which the pivot changed.
            before_name = _COORD_SUFFIX.sub("", b_names[(r, c)])
            after_name = a_names[(c, r)]
            assert _COORD_SUFFIX.sub("", after_name) == _swap_axis_parts(before_name)
            if after_name != _COORD_SUFFIX.sub("", after_name):
                assert after_name.endswith(f" (R{c + 1}C{r + 1})")


def test_adjacent_insets_still_draw_into_the_panel_they_drew_into():
    b, a = _cfg(BEFORE), _cfg(AFTER)
    before_targets = {}
    for r, c, side in _adjacent_insets(b):
        tr, tc = _adjacent_target(r, c, side)
        assert 0 <= tr < b.rows and 0 <= tc < b.cols
        before_targets[b.panels[r][c].image_name] = b.panels[tr][tc].image_name
    after_targets = {}
    for r, c, side in _adjacent_insets(a):
        tr, tc = _adjacent_target(r, c, side)
        assert 0 <= tr < a.rows and 0 <= tc < a.cols
        after_targets[a.panels[r][c].image_name] = a.panels[tr][tc].image_name
    assert before_targets and after_targets == before_targets
