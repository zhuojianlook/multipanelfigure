"""Curved Surface readings: per-reading label size / colour overrides.

`ThicknessReading.font_size` / `color` (None = inherit the measurement's
measure_font_size / measure_color) are set in batch in the Edit Panel. They
must survive the dict round trip the .mpf save/load uses, tolerate files
written before they existed, and change what draw_thickness_measurements
actually draws. Pure in-process: no server.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from image_processing import draw_thickness_measurements  # noqa: E402
from models import (  # noqa: E402
    PanelInfo, ThicknessMeasurement, ThicknessReading, _from_dict, _to_dict,
)

CYAN = "#00E5FF"
RED = "#FF0000"


def _reading(**kw):
    d = {"top": [50.0, 30.0], "bottom": [50.0, 70.0], "hidden": False, "text": "",
         "edited": False, "measure_position_x": -1.0, "measure_position_y": -1.0,
         "styled_segments": []}
    d.update(kw)
    return d


def _measurement(readings, **kw):
    d = {"name": "m", "top_points": [[20, 30], [50, 30], [80, 30], [90, 35]],
         "bottom_points": [[20, 70], [80, 70]], "num_readings": len(readings),
         "center": 0.5, "spacing": 0.1, "readings": readings,
         "top_samples": [], "bottom_samples": [], "color": CYAN, "width": 1.0,
         "show_curves": False, "show_measure": True, "show_connecting_lines": False,
         "measure_unit": "um", "measure_font_size": 5, "measure_color": CYAN,
         "measure_font_name": "arial.ttf"}
    d.update(kw)
    return d


def test_reading_overrides_round_trip_and_default_to_none():
    rd = _from_dict(ThicknessReading, _reading(font_size=9, color=RED))
    assert rd.font_size == 9 and rd.color == RED
    back = _to_dict(rd)
    assert back["font_size"] == 9 and back["color"] == RED
    assert _from_dict(ThicknessReading, back) == rd

    # Written before the fields existed → inherit.
    old = _from_dict(ThicknessReading, _reading())
    assert old.font_size is None and old.color is None
    # Explicitly cleared back to the measurement default.
    cleared = _from_dict(ThicknessReading, _reading(font_size=None, color=None))
    assert cleared.font_size is None and cleared.color is None


def test_overrides_survive_panel_round_trip_with_n_point_surfaces():
    m = _measurement([_reading(font_size=12, color=RED), _reading(top=[60, 30], bottom=[60, 70])])
    panel = _from_dict(PanelInfo, {"thickness_measurements": [m]})
    tm = panel.thickness_measurements[0]
    assert isinstance(tm, ThicknessMeasurement)
    assert isinstance(tm.readings[0], ThicknessReading)
    assert len(tm.top_points) == 4 and len(tm.bottom_points) == 2   # not capped at 3
    again = _from_dict(PanelInfo, _to_dict(panel)).thickness_measurements[0]
    assert again.readings[0].font_size == 12 and again.readings[0].color == RED
    assert again.readings[1].font_size is None and again.readings[1].color is None
    assert [list(p) for p in again.top_points] == [[20, 30], [50, 30], [80, 30], [90, 35]]


def _render(readings, **kw):
    items = [_from_dict(ThicknessMeasurement, _measurement(readings, **kw))]
    img = Image.new("RGB", (400, 400), (0, 0, 0))
    return np.asarray(draw_thickness_measurements(img, items, 1.0)).astype(int)


def _mask(arr, rgb):
    """Pixels whose hue is that colour's (anti-aliased glyph edges included)."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    if rgb == (255, 0, 0):
        return (r > 80) & (g < r * 0.4) & (b < r * 0.4)
    return (g > 80) & (b > 80) & (r < g * 0.4)   # cyan


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    assert len(xs), "nothing drawn in that colour"
    return xs.min(), ys.min(), xs.max(), ys.max()


def test_colour_override_draws_that_reading_in_its_own_colour():
    arr = _render([
        _reading(top=[25, 30], bottom=[25, 70], color=RED),
        _reading(top=[75, 30], bottom=[75, 70]),
    ], measure_font_size=10)
    red = _mask(arr, (255, 0, 0))
    cyan = _mask(arr, (0, 229, 255))
    assert red.sum() > 20
    # The red label belongs to the left reading; the tick lines stay the
    # measurement colour, and the right reading keeps the default colour.
    x0, _, x1, _ = _bbox(red)
    assert x1 < 200
    assert cyan[:, 200:].sum() > 20
    assert not red[:, 200:].any()


def test_size_override_draws_a_larger_label_and_clears_back():
    def label_box(fs):
        arr = _render([_reading(top=[50, 40], bottom=[50, 80], color=RED, font_size=fs)])
        x0, y0, x1, y1 = _bbox(_mask(arr, (255, 0, 0)))
        return x1 - x0, y1 - y0

    small_w, small_h = label_box(5)
    big_w, big_h = label_box(12)
    assert big_w > small_w * 1.5 and big_h > small_h * 1.5
    # None = the measurement's measure_font_size (5) → same as the explicit 5.
    assert label_box(None) == (small_w, small_h)


def test_overrides_reach_styled_segment_labels():
    seg = {"text": "abc", "color": None, "font_name": None, "font_size": None, "font_style": []}
    arr_small = _render([_reading(top=[50, 40], bottom=[50, 80], text="abc", styled_segments=[seg],
                                  color=RED, font_size=5)])
    arr_big = _render([_reading(top=[50, 40], bottom=[50, 80], text="abc", styled_segments=[seg],
                                color=RED, font_size=12)])
    sw = np.ptp(np.nonzero(_mask(arr_small, (255, 0, 0)))[1])
    bw = np.ptp(np.nonzero(_mask(arr_big, (255, 0, 0)))[1])
    assert bw > sw * 1.5


def test_bad_colour_falls_back_to_measurement_colour():
    arr = _render([_reading(color="not-a-colour")])
    assert _mask(arr, (0, 229, 255)).sum() > 20
