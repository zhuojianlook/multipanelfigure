"""Panel display names from row/column labels + nested headers."""

from models import AxisLabel, FigureConfig, HeaderGroup, HeaderLevel, StyledSegment
from panel_naming import panel_display_names, plain_text


def _cfg(rows=2, cols=2, **kw):
    cfg = FigureConfig(rows=rows, cols=cols)
    cfg.ensure_grid()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _labels(*texts, **kw):
    return [AxisLabel(text=t, **kw) for t in texts]


def test_row_then_column_labels():
    cfg = _cfg(row_labels=_labels("1hr", "2hr"), column_labels=_labels("Sample 1", "Sample 2"))
    n = panel_display_names(cfg)
    assert n[(0, 0)] == "1hr, Sample 1"
    assert n[(1, 1)] == "2hr, Sample 2"


def test_nested_headers_outermost_first():
    cfg = _cfg(rows=1, cols=4,
               row_labels=_labels("1hr"),
               column_labels=_labels("S1", "S2", "S3", "S4"),
               column_headers=[
                   HeaderLevel([HeaderGroup(text="Day 2", columns_or_rows=[0, 1, 2, 3])]),   # outermost
                   HeaderLevel([HeaderGroup(text="WT", columns_or_rows=[0, 1]),
                                HeaderGroup(text="KO", columns_or_rows=[2, 3])]),
               ])
    n = panel_display_names(cfg)
    assert n[(0, 0)] == "1hr, Day 2 › WT › S1"
    assert n[(0, 3)] == "1hr, Day 2 › KO › S4"


def test_row_headers_and_partial_spans():
    cfg = _cfg(rows=3, cols=1, row_labels=_labels("a", "b", "c"), column_labels=_labels("X"),
               row_headers=[HeaderLevel([HeaderGroup(text="Treated", columns_or_rows=[1, 2])])])
    n = panel_display_names(cfg)
    assert n[(0, 0)] == "a, X"            # not under the header
    assert n[(2, 0)] == "Treated › c, X"


def test_hidden_or_empty_labels_fall_back_to_coords():
    cfg = _cfg(rows=1, cols=2, row_labels=_labels(""), column_labels=_labels("", ""))
    assert panel_display_names(cfg) == {(0, 0): "R1C1", (0, 1): "R1C2"}
    cfg2 = _cfg(rows=1, cols=2, row_labels=_labels("r"), column_labels=_labels("A", "B"),
                show_row_labels=False)
    assert panel_display_names(cfg2) == {(0, 0): "A", (0, 1): "B"}
    cfg3 = _cfg(rows=1, cols=1, row_labels=[AxisLabel(text="r", visible=False)], column_labels=_labels("A"))
    assert panel_display_names(cfg3)[(0, 0)] == "A"


def test_duplicates_are_disambiguated_everywhere():
    cfg = _cfg(rows=1, cols=2, row_labels=_labels("t0"), column_labels=_labels("Ctrl", "Ctrl"))
    assert panel_display_names(cfg) == {(0, 0): "t0, Ctrl (R1C1)", (0, 1): "t0, Ctrl (R1C2)"}


def test_markup_segments_and_line_breaks_are_flattened():
    assert plain_text(AxisLabel(text="::FF0000::Red::\nchannel")) == "Red channel"
    seg = AxisLabel(text="ignored", styled_segments=[StyledSegment(text="Dr"), StyledSegment(text="ug A")])
    assert plain_text(seg) == "Drug A"


def test_header_equal_to_label_collapses():
    cfg = _cfg(rows=1, cols=1, row_labels=_labels("1hr"), column_labels=_labels("WT"),
               column_headers=[HeaderLevel([HeaderGroup(text="WT", columns_or_rows=[0])])])
    assert panel_display_names(cfg)[(0, 0)] == "1hr, WT"


def test_default_grid_labels():
    cfg = _cfg(rows=1, cols=2)   # ensure_grid seeds "Row 1" / "Column 1"
    assert panel_display_names(cfg)[(0, 1)] == "Row 1, Column 2"
