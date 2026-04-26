"""
Data models for Multi-Panel Figure Builder.
All state is held in plain dataclass objects — no Qt dependencies here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import json, copy, zipfile, base64


# ---------------------------------------------------------------------------
# Per-label settings (text overlay on a panel)
# ---------------------------------------------------------------------------
@dataclass
class LabelSettings:
    text: str = "a"
    font_path: Optional[str] = None
    font_name: str = "arial.ttf"
    font_size: int = 20
    font_style: List[str] = field(default_factory=list)  # e.g. ["Bold","Italic","Superscript","Subscript","Strikethrough","Underline"]
    color: str = "#000000"
    position_x: int = 5          # percentage 0-100
    position_y: int = 5          # percentage 0-100
    rotation: float = 0.0

    # Convenience for colored-text markup  (::RRGGBB::text::)
    default_color: str = "#000000"

    # Position presets and linking
    position_preset: str = "Custom"   # "Custom", "Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"
    edge_distance: float = 3.0       # percentage from edge for preset positions
    linked_to_header: bool = True    # whether font/size syncs with column/row headers
    styled_segments: List[StyledSegment] = field(default_factory=list)  # per-character styling
    draggable: bool = False          # True for custom placement labels


# ---------------------------------------------------------------------------
# Scale-bar settings
# ---------------------------------------------------------------------------
@dataclass
class ScaleBarSettings:
    micron_per_pixel: float = 1.0
    bar_length_microns: float = 100.0
    bar_position: Tuple[int, int] = (0, 0)   # (x, y) in pixels
    bar_height: int = 5
    bar_color: str = "#FFFFFF"
    label: str = "100 \u03bcm"
    font_size: int = 20
    font_name: str = "arial.ttf"
    font_path: Optional[str] = None
    label_x_offset: int = 0
    label_font_style: List[str] = field(default_factory=list)

    # Position presets
    position_preset: str = "Bottom-Right"  # position preset like labels
    position_x: float = 90.0              # percentage 0-100
    position_y: float = 90.0              # percentage 0-100
    edge_distance: float = 5.0            # percentage from edge
    unit: str = "um"                       # "cm", "mm", "um", "nm"
    scale_name: str = ""                   # name of predefined scale (empty = custom)
    styled_segments: List[StyledSegment] = field(default_factory=list)  # for label text customization
    draggable: bool = False                # True for custom placement
    label_color: str = "#FFFFFF"


# ---------------------------------------------------------------------------
# Predefined scale definition (for sidebar scale bars)
# ---------------------------------------------------------------------------
@dataclass
class ScaleDefinition:
    name: str = ""                  # e.g. "Microscope 1"
    unit: str = "um"                # "cm", "mm", "um", "nm"
    value_per_pixel: float = 1.0    # e.g. 1 um = 1 pixel


# ---------------------------------------------------------------------------
# Symbol / annotation
# ---------------------------------------------------------------------------
@dataclass
class SymbolSettings:
    name: str = ""                          # user-assigned name (e.g. "ROI marker")
    shape: str = "Arrow"                    # Arrow | Star | Rectangle | Ellipse | Cross | NarrowTriangle | Triangle
    x: float = 50.0                         # percentage 0-100 within panel
    y: float = 50.0                         # percentage 0-100 within panel
    color: str = "#FF0000"
    size: float = 25.0                      # size in points (rendered at figure DPI)
    rotation: float = -45.0
    label_text: str = ""
    label_color: str = "#FFFFFF"
    label_offset_x: int = 0                  # legacy, kept for compat
    label_offset_y: int = 0                  # legacy, kept for compat
    label_position_x: float = -1.0           # absolute position (% 0-100), -1 = auto (near symbol)
    label_position_y: float = -1.0           # absolute position (% 0-100), -1 = auto (near symbol)
    label_font_name: str = "arial.ttf"
    label_font_size: int = 12
    label_font_path: Optional[str] = None
    label_font_style: List[str] = field(default_factory=list)
    label_styled_segments: List[StyledSegment] = field(default_factory=list)
    # Position preset and dragging
    position_preset: str = "Custom"   # "Custom" = freely placed; other presets not draggable
    edge_distance: float = 5.0
    draggable: bool = True            # True for free placement


# ---------------------------------------------------------------------------
# Line annotation (for measurement / boundary marking)
# ---------------------------------------------------------------------------
@dataclass
class LineAnnotation:
    name: str = ""                      # user-assigned name
    points: List[Tuple[float, float]] = field(default_factory=list)  # list of (x%, y%) 0-100
    color: str = "#FFFF00"
    width: float = 2.0
    dash_style: str = "solid"           # "solid", "dashed", "dotted", "dash-dot"
    line_type: str = "straight"         # "straight", "multijointed", "curved"
    is_curved: bool = False             # straight line segments vs spline (legacy, use line_type)
    show_measure: bool = False          # show length text on image
    measure_text: str = ""              # computed measurement text
    measure_unit: str = "um"            # "cm", "mm", "um", "nm"
    measure_font_size: int = 12
    measure_color: str = "#FFFF00"
    measure_font_name: str = "arial.ttf"
    measure_font_path: Optional[str] = None
    measure_font_style: List[str] = field(default_factory=list)
    measure_styled_segments: List[StyledSegment] = field(default_factory=list)
    measure_position_x: float = -1.0   # absolute text position (% 0-100), -1 = auto
    measure_position_y: float = -1.0   # absolute text position (% 0-100), -1 = auto


# ---------------------------------------------------------------------------
# Area annotation (for region marking / area measurement)
# ---------------------------------------------------------------------------
@dataclass
class AreaAnnotation:
    name: str = ""                      # user-assigned name
    shape: str = "Rectangle"            # "Rectangle", "Ellipse", "Custom", "Magic"
    points: List[Tuple[float, float]] = field(default_factory=list)  # boundary points (x%, y%) 0-100
    # For simple shapes: points[0]=(x,y) center, points[1]=(w,h) dimensions
    color: str = "#FF000040"            # RGBA hex with alpha for transparency
    border_color: str = "#FF0000"
    border_width: float = 1.0
    show_measure: bool = False          # show area text on image
    measure_text: str = ""              # computed measurement text
    measure_unit: str = "um"            # "cm", "mm", "um", "nm"
    measure_font_size: int = 12
    measure_color: str = "#FFFF00"
    measure_font_name: str = "arial.ttf"
    measure_font_path: Optional[str] = None
    measure_font_style: List[str] = field(default_factory=list)
    measure_styled_segments: List[StyledSegment] = field(default_factory=list)
    measure_position_x: float = -1      # custom label position (-1 = auto at centroid)
    measure_position_y: float = -1
    fill_alpha: float = 0.25            # 0.0 - 1.0 transparency
    smooth: bool = False                # smooth custom polygon edges (Catmull-Rom spline)
    dash_style: str = "solid"           # "solid", "dashed", "dotted"
    magic_tolerance: int = 30           # magic wand tolerance
    magic_click_x: float = -1          # stored click point for re-selection
    magic_click_y: float = -1


# ---------------------------------------------------------------------------
# Zoom-inset settings  (3 modes: Standard, Separate Image, Adjacent Panel)
# ---------------------------------------------------------------------------
@dataclass
class ZoomInsetSettings:
    inset_type: str = "Standard Zoom"       # Standard Zoom | Separate Image | Adjacent Panel

    # --- Common fields ---
    zoom_factor: float = 2.0
    rectangle_color: str = "#FF0000"
    rectangle_width: int = 1
    line_color: str = "#FF0000"
    line_width: int = 1

    # --- Standard Zoom fields ---
    x: int = 0
    y: int = 0
    width: int = 50
    height: int = 50
    target_x: int = 0
    target_y: int = 0

    # --- Separate Image fields ---
    separate_image_name: str = ""
    x_main: int = 0
    y_main: int = 0
    width_main: int = 50
    height_main: int = 50
    x_inset: int = 0
    y_inset: int = 0
    width_inset: int = 50
    height_inset: int = 50

    # --- Adjacent Panel fields ---
    side: str = "Right"                     # Top | Bottom | Left | Right
    margin_offset: int = 20
    adjacent_panel_row: int = -1            # row of the adjacent panel (-1 = auto)
    adjacent_panel_col: int = -1            # col of the adjacent panel (-1 = auto)

    # --- Optional sub-features ---
    scale_bar: Optional[ScaleBarSettings] = None
    zoom_label: Optional[LabelSettings] = None
    show_scale_bar_in_zoom: bool = False    # show scale bar inside zoomed area


# ---------------------------------------------------------------------------
# Parked panel (for parking drawer)
# ---------------------------------------------------------------------------
@dataclass
class ParkedPanel:
    original_row: int = -1
    original_col: int = -1
    panel: PanelInfo = field(default_factory=lambda: PanelInfo())
    image_name: str = ""


# ---------------------------------------------------------------------------
# Per-panel (cell) configuration
# ---------------------------------------------------------------------------
@dataclass
class PanelInfo:
    image_name: str = ""                    # filename of assigned image, "" = empty
    crop_image: bool = True
    aspect_ratio_str: str = ""              # e.g. "1:1", "16:9"
    crop_offset_x: int = 0
    crop_offset_y: int = 0
    crop: Optional[Tuple[int, int, int, int]] = None
    final_resize: bool = False
    final_width: int = 400
    final_height: int = 300

    brightness: float = 1.0
    contrast: float = 1.0
    hue: float = 0.0

    labels: List[LabelSettings] = field(default_factory=list)
    scale_bar: Optional[ScaleBarSettings] = None
    add_scale_bar: bool = False
    symbols: List[SymbolSettings] = field(default_factory=list)
    lines: List[LineAnnotation] = field(default_factory=list)
    areas: List[AreaAnnotation] = field(default_factory=list)

    zoom_inset: Optional[ZoomInsetSettings] = None  # legacy single inset
    add_zoom_inset: bool = False                    # legacy flag
    zoom_insets: List[ZoomInsetSettings] = field(default_factory=list)  # NEW: array
    is_zoom_target: bool = False            # True if this panel is an adjacent-zoom target
    zoom_source_row: int = -1               # row of the source panel for adjacent zoom
    zoom_source_col: int = -1               # col of the source panel for adjacent zoom

    rotation: float = 0.0                   # 0-360 degrees
    flip_horizontal: bool = False           # mirror left-right
    flip_vertical: bool = False             # mirror top-bottom
    saturation: float = 1.0                 # 0-2.0, default 1.0
    gamma: float = 1.0                      # 0.1-3.0, default 1.0
    color_temperature: float = 0.0          # -100 to 100, default 0
    tint: float = 0.0                       # -100 to 100, default 0
    sharpen: float = 0.0                    # 0-2.0, default 0
    blur: float = 0.0                       # 0-20, default 0
    denoise: float = 0.0                    # 0-1.0, default 0
    exposure: float = 0.0                   # -3 to 3, default 0
    vibrance: float = 0.0                   # -100 to 100, default 0
    highlights: float = 0.0                 # -100 to 100, default 0
    shadows: float = 0.0                    # -100 to 100, default 0
    midtones: float = 0.0                   # -100 to 100, default 0
    input_black_r: int = 0                  # 0-255, red channel black-point
    input_white_r: int = 255               # 0-255, red channel white-point
    input_black_g: int = 0                  # 0-255, green channel black-point
    input_white_g: int = 255               # 0-255, green channel white-point
    input_black_b: int = 0                  # 0-255, blue channel black-point
    input_white_b: int = 255               # 0-255, blue channel white-point
    invert: bool = False                    # default False
    grayscale: bool = False                 # default False
    pseudocolor: str = ""                   # "" = none, or colormap name

    # Video range fields. Default to a static single-frame display (frame
    # index = current static frame from `frame`). When `play_range` is
    # True AND the panel's image is a video, a Save → Video render plays
    # frames [frame_start, frame_end] inclusive at the user's chosen FPS.
    # Values are clamped to [0, frame_count-1] at render time.
    frame: int = 0                          # statically displayed frame
    frame_start: int = 0                    # video play-range start (inclusive)
    frame_end: int = 0                      # video play-range end (inclusive)
    play_range: bool = False                # if True → animate across [start..end] in video export


# ---------------------------------------------------------------------------
# Row / Column header (one level, one group)
# ---------------------------------------------------------------------------
@dataclass
class StyledSegment:
    """One run of text with its own color.  Used for per-character coloring."""
    text: str = ""
    color: str = "#000000"
    font_name: Optional[str] = None
    font_size: Optional[int] = None
    font_style: Optional[List[str]] = None


@dataclass
class HeaderGroup:
    text: str = "Header"
    columns_or_rows: List[int] = field(default_factory=list)
    font_size: int = 10
    font_name: str = "arial.ttf"
    font_path: Optional[str] = None
    font_style: List[str] = field(default_factory=list)
    default_color: str = "#000000"
    distance: float = 0.05
    position: str = "Top"          # Top/Bottom for column headers, Left/Right for row headers
    rotation: float = 0.0
    line_color: str = "#000000"
    line_width: float = 1.0
    line_style: str = "solid"      # "solid", "dashed", "dotted", "dash-dot"
    line_length: float = 1.0       # fraction of span width for bracket line
    end_caps: bool = False         # small perpendicular caps at line ends toward previous header
    # Per-character color segments (optional).
    # If non-empty, these override parse_colored_text / default_color.
    styled_segments: List[StyledSegment] = field(default_factory=list)


@dataclass
class HeaderLevel:
    headers: List[HeaderGroup] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simple column / row label  (the basic per-column / per-row label)
# ---------------------------------------------------------------------------
@dataclass
class AxisLabel:
    text: str = ""
    font_size: int = 12
    font_name: str = "arial.ttf"
    font_path: Optional[str] = None
    font_style: List[str] = field(default_factory=list)
    default_color: str = "#000000"
    distance: float = 0.01
    position: str = "Top"           # Top/Bottom or Left/Right
    rotation: float = 0.0
    styled_segments: List[StyledSegment] = field(default_factory=list)
    visible: bool = True            # can be hidden via X button


# ---------------------------------------------------------------------------
# Clipboard for copy/paste panel settings
# ---------------------------------------------------------------------------
@dataclass
class PanelSettingsClipboard:
    """Stores panel settings for copy/paste (excludes zoom inset)."""
    crop_image: bool = True
    aspect_ratio_str: str = ""
    crop_offset_x: int = 0
    crop_offset_y: int = 0
    brightness: float = 1.0
    contrast: float = 1.0
    hue: float = 0.0
    labels: List[LabelSettings] = field(default_factory=list)
    scale_bar: Optional[ScaleBarSettings] = None
    add_scale_bar: bool = False
    symbols: List[SymbolSettings] = field(default_factory=list)
    lines: List[LineAnnotation] = field(default_factory=list)
    areas: List[AreaAnnotation] = field(default_factory=list)
    rotation: float = 0.0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    saturation: float = 1.0
    gamma: float = 1.0
    color_temperature: float = 0.0
    tint: float = 0.0
    sharpen: float = 0.0
    blur: float = 0.0
    denoise: float = 0.0
    exposure: float = 0.0
    vibrance: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    midtones: float = 0.0
    input_black_r: int = 0
    input_white_r: int = 255
    input_black_g: int = 0
    input_white_g: int = 255
    input_black_b: int = 0
    input_white_b: int = 255
    invert: bool = False
    grayscale: bool = False


# ---------------------------------------------------------------------------
# Top-level figure configuration
# ---------------------------------------------------------------------------
@dataclass
class FigureConfig:
    rows: int = 2
    cols: int = 2
    spacing: float = 0.02
    output_format: str = "TIFF"              # TIFF | PNG | JPEG
    background: str = "White"                # White | Transparent

    panels: List[List[PanelInfo]] = field(default_factory=list)  # [row][col]

    column_labels: List[AxisLabel] = field(default_factory=list)
    row_labels: List[AxisLabel] = field(default_factory=list)
    column_headers: List[HeaderLevel] = field(default_factory=list)
    row_headers: List[HeaderLevel] = field(default_factory=list)

    # Resolution mapping (microscope_name -> um/pixel)
    resolution_entries: Dict[str, float] = field(default_factory=dict)

    dpi: int = 300  # DPI setting for figure output

    # Scale bar definitions (predefined scales for sidebar)
    scale_definitions: List[ScaleDefinition] = field(default_factory=list)

    # Parked panels (for parking drawer)
    parked_panels: List[ParkedPanel] = field(default_factory=list)

    # Show primary labels toggle
    show_column_labels: bool = True
    show_row_labels: bool = True

    # Normalize panel widths: scale smaller images to match the widest in each column
    normalize_widths: bool = False    # "width" or "height" normalization
    normalize_mode: str = "width"     # "width" or "height"

    # ------------------------------------------------------------------
    def ensure_grid(self):
        """Make sure panels list matches rows x cols."""
        while len(self.panels) < self.rows:
            self.panels.append([])
        for r in range(self.rows):
            while len(self.panels[r]) < self.cols:
                self.panels[r].append(PanelInfo())
        # trim extras
        self.panels = self.panels[:self.rows]
        for r in range(self.rows):
            self.panels[r] = self.panels[r][:self.cols]

        # Copy formatting from the first existing label so new labels match
        col_ref = self.column_labels[0] if self.column_labels else None
        while len(self.column_labels) < self.cols:
            idx = len(self.column_labels) + 1
            new_lbl = AxisLabel(text=f"Column {idx}", position="Top")
            if col_ref:
                new_lbl.font_size = col_ref.font_size
                new_lbl.font_name = col_ref.font_name
                new_lbl.font_path = col_ref.font_path
                new_lbl.font_style = list(col_ref.font_style)
                new_lbl.default_color = col_ref.default_color
                new_lbl.distance = col_ref.distance
                new_lbl.rotation = col_ref.rotation
            self.column_labels.append(new_lbl)
        self.column_labels = self.column_labels[:self.cols]

        row_ref = self.row_labels[0] if self.row_labels else None
        while len(self.row_labels) < self.rows:
            idx = len(self.row_labels) + 1
            new_lbl = AxisLabel(text=f"Row {idx}", position="Left", rotation=90.0)
            if row_ref:
                new_lbl.font_size = row_ref.font_size
                new_lbl.font_name = row_ref.font_name
                new_lbl.font_path = row_ref.font_path
                new_lbl.font_style = list(row_ref.font_style)
                new_lbl.default_color = row_ref.default_color
                new_lbl.distance = row_ref.distance
            self.row_labels.append(new_lbl)
        self.row_labels = self.row_labels[:self.rows]


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------
UNIT_TO_MICRONS = {
    "km": 1e9,
    "m": 1e6,
    "cm": 10000.0,
    "mm": 1000.0,
    "um": 1.0,
    "nm": 0.001,
    "pm": 1e-6,
}

def convert_scale_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a scale value between units."""
    microns = value * UNIT_TO_MICRONS.get(from_unit, 1.0)
    return microns / UNIT_TO_MICRONS.get(to_unit, 1.0)


def compute_line_length_pixels(points: List[Tuple[float, float]],
                                image_width: int, image_height: int) -> float:
    """Compute the total pixel length of a polyline from percentage coordinates."""
    import math
    total = 0.0
    for i in range(len(points) - 1):
        x1 = points[i][0] / 100.0 * image_width
        y1 = points[i][1] / 100.0 * image_height
        x2 = points[i + 1][0] / 100.0 * image_width
        y2 = points[i + 1][1] / 100.0 * image_height
        total += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return total


def compute_line_measurement(points: List[Tuple[float, float]],
                              image_width: int, image_height: int,
                              micron_per_pixel: float, unit: str) -> str:
    """Compute a human-readable measurement string for a line."""
    px_length = compute_line_length_pixels(points, image_width, image_height)
    length_um = px_length * micron_per_pixel
    length_in_unit = length_um / UNIT_TO_MICRONS.get(unit, 1.0)
    unit_labels = {"km": "km", "m": "m", "cm": "cm", "mm": "mm", "um": "\u03bcm", "nm": "nm", "pm": "pm"}
    return f"{length_in_unit:.2f} {unit_labels.get(unit, unit)}"


def compute_area_pixels(points: List[Tuple[float, float]], shape: str,
                         image_width: int, image_height: int) -> float:
    """Compute area in pixels^2 from percentage coordinates."""
    import math
    if shape == "Rectangle" and len(points) >= 2:
        cx, cy = points[0]
        w, h = points[1]
        pw = w / 100.0 * image_width
        ph = h / 100.0 * image_height
        return pw * ph
    elif shape == "Ellipse" and len(points) >= 2:
        cx, cy = points[0]
        w, h = points[1]
        pw = w / 100.0 * image_width / 2.0
        ph = h / 100.0 * image_height / 2.0
        return math.pi * pw * ph
    elif shape == "Triangle" and len(points) >= 3:
        # Shoelace formula
        coords = [(p[0] / 100.0 * image_width, p[1] / 100.0 * image_height) for p in points[:3]]
        return abs(0.5 * ((coords[1][0] - coords[0][0]) * (coords[2][1] - coords[0][1]) -
                          (coords[2][0] - coords[0][0]) * (coords[1][1] - coords[0][1])))
    elif shape in ("Custom", "Magic") and len(points) >= 3:
        # Shoelace formula for arbitrary polygon
        coords = [(p[0] / 100.0 * image_width, p[1] / 100.0 * image_height) for p in points]
        n = len(coords)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += coords[i][0] * coords[j][1]
            area -= coords[j][0] * coords[i][1]
        return abs(area) / 2.0
    return 0.0


def compute_area_measurement(points: List[Tuple[float, float]], shape: str,
                              image_width: int, image_height: int,
                              micron_per_pixel: float, unit: str) -> str:
    """Compute a human-readable area measurement string."""
    px_area = compute_area_pixels(points, shape, image_width, image_height)
    area_um2 = px_area * (micron_per_pixel ** 2)
    um_per_unit = UNIT_TO_MICRONS.get(unit, 1.0)
    area_in_unit = area_um2 / (um_per_unit ** 2)
    unit_labels = {"km": "km\u00b2", "m": "m\u00b2", "cm": "cm\u00b2", "mm": "mm\u00b2", "um": "\u03bcm\u00b2", "nm": "nm\u00b2", "pm": "pm\u00b2"}
    sq = '\u00b2'
    return f"{area_in_unit:.2f} {unit_labels.get(unit, unit + sq)}"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _to_dict(obj):
    """Recursively convert dataclass -> dict."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _to_dict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def save_config(cfg: FigureConfig, path: str):
    with open(path, 'w') as f:
        json.dump(_to_dict(cfg), f, indent=2)


def _resolve_type(cls, type_hint):
    """Resolve a type hint (possibly a string) to an actual type."""
    import typing, dataclasses as dc
    if isinstance(type_hint, str):
        # Evaluate string annotation in the context of our module
        ns = {
            'List': List, 'Optional': Optional, 'Tuple': Tuple, 'Dict': Dict,
            'FigureConfig': FigureConfig, 'PanelInfo': PanelInfo,
            'LabelSettings': LabelSettings, 'ScaleBarSettings': ScaleBarSettings,
            'SymbolSettings': SymbolSettings, 'ZoomInsetSettings': ZoomInsetSettings,
            'LineAnnotation': LineAnnotation, 'AreaAnnotation': AreaAnnotation,
            'AxisLabel': AxisLabel, 'HeaderLevel': HeaderLevel,
            'HeaderGroup': HeaderGroup, 'StyledSegment': StyledSegment,
            'ScaleDefinition': ScaleDefinition, 'ParkedPanel': ParkedPanel,
            'PanelSettingsClipboard': PanelSettingsClipboard,
            'int': int, 'float': float, 'str': str, 'bool': bool,
        }
        try:
            return eval(type_hint, ns)
        except Exception:
            return type_hint
    return type_hint


# Map of field name -> dataclass type for known nested lists
_FIELD_TYPE_MAP = {
    'panels': ('list_list', PanelInfo),
    'labels': ('list', LabelSettings),
    'symbols': ('list', SymbolSettings),
    'column_labels': ('list', AxisLabel),
    'row_labels': ('list', AxisLabel),
    'column_headers': ('list', HeaderLevel),
    'row_headers': ('list', HeaderLevel),
    'headers': ('list', HeaderGroup),
    'columns_or_rows': ('plain', None),
    'styled_segments': ('list', StyledSegment),
    'font_style': ('plain', None),
    'scale_bar': ('dataclass', ScaleBarSettings),
    'zoom_inset': ('dataclass', ZoomInsetSettings),
    'zoom_label': ('dataclass', LabelSettings),
    'crop': ('tuple', None),
    'bar_position': ('tuple', None),
    'resolution_entries': ('plain', None),
    'label_styled_segments': ('list', StyledSegment),
    'measure_styled_segments': ('list', StyledSegment),
    'lines': ('list', LineAnnotation),
    'areas': ('list', AreaAnnotation),
    'label_font_style': ('plain', None),
    'scale_definitions': ('list', ScaleDefinition),
    'parked_panels': ('list', ParkedPanel),
    'panel': ('dataclass', PanelInfo),
    'measure_font_style': ('plain', None),
    'label_font_path': ('plain', None),
    'measure_font_path': ('plain', None),
    'points': ('plain', None),
}


def _from_dict(cls, d):
    """Recursively reconstruct a dataclass from a dict."""
    if d is None:
        return None
    import dataclasses as dc
    if not dc.is_dataclass(cls):
        return d
    if not isinstance(d, dict):
        return d

    field_names = {f.name for f in dc.fields(cls)}
    kwargs = {}
    for k, v in d.items():
        if k not in field_names:
            continue
        if v is None:
            kwargs[k] = None
            continue

        # Use the explicit type map if available
        if k in _FIELD_TYPE_MAP:
            kind, inner_cls = _FIELD_TYPE_MAP[k]
            if kind == 'list_list' and inner_cls and isinstance(v, list):
                kwargs[k] = [[_from_dict(inner_cls, item) for item in row]
                             if isinstance(row, list) else row for row in v]
            elif kind == 'list' and inner_cls and isinstance(v, list):
                kwargs[k] = [_from_dict(inner_cls, item) for item in v]
            elif kind == 'dataclass' and inner_cls and isinstance(v, dict):
                kwargs[k] = _from_dict(inner_cls, v)
            elif kind == 'tuple' and isinstance(v, list):
                kwargs[k] = tuple(v)
            else:
                kwargs[k] = v
        elif isinstance(v, dict):
            # Try to figure out if it's a nested dataclass
            kwargs[k] = v
        elif isinstance(v, list) and v and isinstance(v[0], list):
            kwargs[k] = v
        else:
            kwargs[k] = v
    return cls(**kwargs)


def load_config(path: str) -> FigureConfig:
    with open(path, 'r') as f:
        d = json.load(f)
    cfg = _from_dict(FigureConfig, d)
    cfg.ensure_grid()
    return cfg


# ---------------------------------------------------------------------------
# Copy / paste panel settings helpers
# ---------------------------------------------------------------------------
def copy_panel_settings(panel: PanelInfo) -> PanelSettingsClipboard:
    """Extract copyable settings from a panel (excludes zoom inset)."""
    return PanelSettingsClipboard(
        crop_image=panel.crop_image,
        aspect_ratio_str=panel.aspect_ratio_str,
        crop_offset_x=panel.crop_offset_x,
        crop_offset_y=panel.crop_offset_y,
        brightness=panel.brightness,
        contrast=panel.contrast,
        hue=panel.hue,
        labels=copy.deepcopy(panel.labels),
        scale_bar=copy.deepcopy(panel.scale_bar),
        add_scale_bar=panel.add_scale_bar,
        symbols=copy.deepcopy(panel.symbols),
        lines=copy.deepcopy(panel.lines),
        areas=copy.deepcopy(panel.areas),
        rotation=panel.rotation,
        flip_horizontal=panel.flip_horizontal,
        flip_vertical=panel.flip_vertical,
        saturation=panel.saturation,
        gamma=panel.gamma,
        color_temperature=panel.color_temperature,
        tint=panel.tint,
        sharpen=panel.sharpen,
        blur=panel.blur,
        denoise=panel.denoise,
        exposure=panel.exposure,
        vibrance=panel.vibrance,
        highlights=panel.highlights,
        shadows=panel.shadows,
        midtones=panel.midtones,
        input_black_r=panel.input_black_r,
        input_white_r=panel.input_white_r,
        input_black_g=panel.input_black_g,
        input_white_g=panel.input_white_g,
        input_black_b=panel.input_black_b,
        input_white_b=panel.input_white_b,
        invert=panel.invert,
        grayscale=panel.grayscale,
    )


def paste_panel_settings(panel: PanelInfo, clipboard: PanelSettingsClipboard,
                          target_image_size: Optional[Tuple[int, int]] = None) -> List[str]:
    """Apply clipboard settings to a panel. Returns list of status/warning messages."""
    messages = []

    # Crop settings
    panel.crop_image = clipboard.crop_image
    panel.aspect_ratio_str = clipboard.aspect_ratio_str
    panel.crop_offset_x = clipboard.crop_offset_x
    panel.crop_offset_y = clipboard.crop_offset_y
    messages.append("crop area...ok")

    # Image adjustments
    panel.brightness = clipboard.brightness
    panel.contrast = clipboard.contrast
    panel.hue = clipboard.hue
    panel.rotation = clipboard.rotation
    panel.flip_horizontal = clipboard.flip_horizontal
    panel.flip_vertical = clipboard.flip_vertical
    panel.saturation = clipboard.saturation
    panel.gamma = clipboard.gamma
    panel.color_temperature = clipboard.color_temperature
    panel.tint = clipboard.tint
    panel.sharpen = clipboard.sharpen
    panel.blur = clipboard.blur
    panel.denoise = clipboard.denoise
    panel.exposure = clipboard.exposure
    panel.vibrance = clipboard.vibrance
    panel.highlights = clipboard.highlights
    panel.shadows = clipboard.shadows
    panel.midtones = clipboard.midtones
    panel.input_black_r = clipboard.input_black_r
    panel.input_white_r = clipboard.input_white_r
    panel.input_black_g = clipboard.input_black_g
    panel.input_white_g = clipboard.input_white_g
    panel.input_black_b = clipboard.input_black_b
    panel.input_white_b = clipboard.input_white_b
    panel.invert = clipboard.invert
    panel.grayscale = clipboard.grayscale
    messages.append("image adjustments...ok")

    # Labels
    panel.labels = copy.deepcopy(clipboard.labels)
    messages.append("labels...ok")

    # Scale bar
    panel.scale_bar = copy.deepcopy(clipboard.scale_bar)
    panel.add_scale_bar = clipboard.add_scale_bar
    messages.append("scale bar...ok")

    # Annotations
    panel.symbols = copy.deepcopy(clipboard.symbols)
    panel.lines = copy.deepcopy(clipboard.lines)
    panel.areas = copy.deepcopy(clipboard.areas)
    messages.append("annotations...ok")

    return messages


# ---------------------------------------------------------------------------
# Zip-based project save / load  (config + images in one file)
# ---------------------------------------------------------------------------

def save_project(cfg: FigureConfig, images: Dict[str, bytes], path: str,
                  fonts: Dict[str, bytes] = None):
    """Save a full project: config JSON + images + videos + custom fonts into a .mpf zip."""
    import io
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('config.json', json.dumps(_to_dict(cfg), indent=2))
        for name, img_bytes in images.items():
            zf.writestr(f'images/{name}', img_bytes)
        if fonts:
            for name, font_bytes in fonts.items():
                zf.writestr(f'fonts/{name}', font_bytes)


def load_project(path: str) -> tuple:
    """Load a .mpfig zip -> (FigureConfig, Dict[str, bytes], Dict[str, bytes]).
    Returns (cfg, images_dict, fonts_dict)."""
    images: Dict[str, bytes] = {}
    fonts: Dict[str, bytes] = {}
    with zipfile.ZipFile(path, 'r') as zf:
        cfg_data = json.loads(zf.read('config.json'))
        cfg = _from_dict(FigureConfig, cfg_data)
        cfg.ensure_grid()
        for name in zf.namelist():
            if name.startswith('images/') and name != 'images/':
                img_name = name[len('images/'):]
                images[img_name] = zf.read(name)
            elif name.startswith('fonts/') and name != 'fonts/':
                font_name = name[len('fonts/'):]
                fonts[font_name] = zf.read(name)
    return cfg, images, fonts
