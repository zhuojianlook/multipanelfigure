/* ──────────────────────────────────────────────────────────
   TypeScript interfaces mirroring the Python dataclasses
   used by the FastAPI backend (models.py).
   ────────────────────────────────────────────────────────── */

// ── Per-panel decoration settings ────────────────────────

export interface LabelSettings {
  text: string;
  font_path: string | null;
  font_name: string;
  font_size: number;
  font_style: string[];
  color: string;
  position_x: number;
  position_y: number;
  rotation: number;
  default_color: string;
  position_preset: string;
  edge_distance: number;
  linked_to_header: boolean;
  styled_segments: StyledSegment[];
}

export interface ScaleBarSettings {
  micron_per_pixel: number;
  bar_length_microns: number;
  bar_position: [number, number];
  bar_height: number;
  bar_color: string;
  label: string;
  font_size: number;
  font_name: string;
  font_path: string | null;
  label_x_offset: number;
  label_font_style: string[];
  label_color: string;
  position_preset: string;
  position_x: number;
  position_y: number;
  edge_distance: number;
  unit: string;
  scale_name: string;
  styled_segments: StyledSegment[];
  draggable: boolean;
}

export interface SymbolSettings {
  name: string;
  shape: string;
  x: number;
  y: number;
  color: string;
  size: number;
  rotation: number;
  /** Arrow / NarrowTriangle only: cross-axis thickness multiplier (1.0 = default).
   *  Optional for backwards compat with projects saved before this field. */
  width?: number;
  /** Outline shapes: border line style — "solid" | "dashed" | "dotted". */
  border_style?: string;
  /** Outline shapes: centre fill colour ("" = outline only). */
  fill_color?: string;
  /** Outline shapes: centre fill opacity, 0-100 %. */
  fill_opacity?: number;
  label_text: string;
  label_color: string;
  label_offset_x: number;
  label_offset_y: number;
  label_position_x: number;  // absolute position (% 0-100), -1 = auto
  label_position_y: number;  // absolute position (% 0-100), -1 = auto
  label_font_name: string;
  label_font_size: number;
  label_font_path: string | null;
  label_font_style: string[];
  label_styled_segments: StyledSegment[];
}

export interface LineAnnotation {
  name: string;
  points: [number, number][];  // (x%, y%) 0-100
  color: string;
  width: number;
  dash_style: string;          // "solid", "dashed", "dotted", "dash-dot"
  line_type: string;           // "straight", "multijointed", "curved"
  is_curved: boolean;
  show_measure: boolean;
  measure_text: string;
  measure_unit: string;
  measure_font_size: number;
  measure_color: string;
  measure_font_name: string;
  measure_font_style: string[];
  measure_styled_segments: StyledSegment[];
  measure_position_x: number;  // absolute text position (% 0-100), -1 = auto
  measure_position_y: number;  // absolute text position (% 0-100), -1 = auto
}

export interface AreaAnnotation {
  name: string;
  shape: string;               // "Rectangle", "Ellipse", "Triangle", "Custom"
  points: [number, number][];  // boundary points (x%, y%) 0-100
  color: string;               // RGBA hex with alpha
  border_color: string;
  border_width: number;
  show_measure: boolean;
  measure_text: string;
  measure_unit?: string;       // "km" | "m" | "cm" | "mm" | "um" | "nm" (area unit)
  measure_font_size: number;
  measure_color: string;
  measure_font_name: string;
  measure_styled_segments: StyledSegment[];
}

export interface ZoomInsetSettings {
  inset_type: string;
  zoom_factor: number;
  rectangle_color: string;
  rectangle_width: number;
  line_color: string;
  line_width: number;
  /** Line style for the selection-box outline and the connecting
   *  lines, set independently. One of "solid", "dashed", "dotted",
   *  "dash-dot". Optional for backwards compat (defaults to "solid"). */
  rectangle_style?: string;
  line_style?: string;
  x: number;
  y: number;
  width: number;
  height: number;
  target_x: number;
  target_y: number;
  separate_image_name: string;
  x_main: number;
  y_main: number;
  width_main: number;
  height_main: number;
  x_inset: number;
  y_inset: number;
  width_inset: number;
  height_inset: number;
  side: string;
  margin_offset: number;
  scale_bar: ScaleBarSettings | null;
  zoom_label: LabelSettings | null;
  /** Standard / Separate insets: when true, the inset draws the
   *  source panel's scale bar at zoomed scale (mpp ÷ zoom_factor).
   *  When false but `scale_bar` is set, a custom bar is drawn
   *  instead. When both are unset, no bar is drawn on the inset.
   *  Has no effect on Adjacent Panel insets — those use the
   *  target cell's own scale_bar. Optional for backwards compat. */
  show_scale_bar_in_zoom?: boolean;
  /** Stable id assigned at creation. Used by `parent_inset_id` to
   *  refer to the inset that spawned this one (for cascade-inherit
   *  scalebar semantics). Optional for backwards compat with
   *  projects saved before this field. */
  id?: string;
  /** When set: this inset was SPAWNED from another inset on the
   *  same parent panel (the one referenced by id). "Inherit"
   *  scalebar on this inset derives from the parent inset's
   *  effective scalebar × this zoom_factor — so spawned chains
   *  cascade scaling correctly even when an intermediate inset
   *  uses a Custom bar. Unset / empty = no parent inset (the root
   *  panel is the inherit-source). */
  parent_inset_id?: string;
  /** Source-rectangle rotation in degrees (clockwise). The
   *  rendered zoom output is ALWAYS axis-aligned — the backend
   *  rotates the source image inversely around the rectangle's
   *  centre before cropping, so the resulting inset shows
   *  un-tilted content. 0 = no rotation. Optional for backwards
   *  compat with projects saved before this field. */
  rotation?: number;
  /** When true, this inset's cropped pixels are surfaced as an
   *  input dataset on the Analysis tab so the user can run custom
   *  Python pipelines on them (intensity histograms, segmentation
   *  masks, custom region statistics, etc.). Optional / defaults
   *  to false for backwards compat. */
  include_in_analysis?: boolean;
}

// ── Panel information ────────────────────────────────────

export interface PanelInfo {
  image_name: string;
  crop_image: boolean;
  aspect_ratio_str: string;
  crop_offset_x: number;
  crop_offset_y: number;
  crop: [number, number, number, number] | null;
  final_resize: boolean;
  final_width: number;
  final_height: number;
  brightness: number;
  contrast: number;
  hue: number;
  labels: LabelSettings[];
  scale_bar: ScaleBarSettings | null;
  add_scale_bar: boolean;
  symbols: SymbolSettings[];
  lines: LineAnnotation[];
  areas: AreaAnnotation[];
  zoom_inset: ZoomInsetSettings | null;   // legacy single inset (backward compat)
  add_zoom_inset: boolean;                // legacy flag
  zoom_insets: ZoomInsetSettings[];       // NEW: array of zoom insets
  rotation: number;           // 0-360 degrees
  flip_horizontal: boolean;   // mirror left-right
  flip_vertical: boolean;     // mirror top-bottom
  saturation: number;         // 0-2.0, default 1.0
  gamma: number;              // 0.1-3.0, default 1.0
  color_temperature: number;  // -100 to 100, default 0
  tint: number;               // -100 to 100, default 0
  sharpen: number;            // 0-2.0, default 0
  blur: number;               // 0-20, default 0
  denoise: number;            // 0-1.0, default 0
  exposure: number;           // -3 to 3, default 0
  vibrance: number;           // -100 to 100, default 0
  highlights: number;         // -100 to 100, default 0
  shadows: number;            // -100 to 100, default 0
  midtones: number;           // -100 to 100, default 0
  input_black_r: number;      // 0-255, red black-point
  input_white_r: number;      // 0-255, red white-point
  input_black_g: number;      // 0-255, green black-point
  input_white_g: number;      // 0-255, green white-point
  input_black_b: number;      // 0-255, blue black-point
  input_white_b: number;      // 0-255, blue white-point
  invert: boolean;            // default false
  grayscale: boolean;         // default false
  pseudocolor: string;        // "" = none, or colormap name: "hot", "cool", "viridis", "magma", "inferno", "plasma", "green", "red", "blue", "cyan", "magenta", "yellow"

  // Video fields — only meaningful when image_name points to a video
  // file. `frame` is the statically-displayed frame; `frame_start`,
  // `frame_end`, `play_range` describe the range to animate during a
  // Save → Video export. Default to 0/0/0/false on legacy projects.
  frame?: number;
  frame_start?: number;
  frame_end?: number;
  play_range?: boolean;
  /** When this panel's play range is shorter than the longest range
   *  in the export, what should it show after its own range ends?
   *    false (default) → hold on frame_end (last frame of range)
   *    true            → snap back to `frame` (the static-selected
   *                      frame the user chose outside the range) */
  return_to_selected_on_end?: boolean;
}

// ── Header / axis labels ─────────────────────────────────

export interface StyledSegment {
  text: string;
  color: string;
  font_name?: string;
  font_size?: number;
  font_style?: string[]; // ["Bold", "Italic", "Underline", "Strikethrough", "Superscript", "Subscript"]
}

export interface HeaderGroup {
  text: string;
  columns_or_rows: number[];
  font_size: number;
  font_name: string;
  font_path: string | null;
  font_style: string[];
  default_color: string;
  distance: number;
  position: string;
  rotation: number;
  line_color: string;
  line_width: number;
  line_style: string;           // "solid", "dashed", "dotted", "dash-dot"
  line_length: number;          // fraction 0-1 of span width for bracket line
  end_caps?: boolean;           // small perpendicular caps at line ends toward previous header
  styled_segments: StyledSegment[];
}

export interface HeaderLevel {
  headers: HeaderGroup[];
}

export interface AxisLabel {
  text: string;
  font_size: number;
  font_name: string;
  font_path: string | null;
  font_style: string[];
  default_color: string;
  distance: number;
  position: string;
  rotation: number;
  styled_segments: StyledSegment[];
  visible: boolean;
}

// ── Top-level figure configuration ───────────────────────

export interface FigureConfig {
  rows: number;
  cols: number;
  spacing: number;
  output_format: string;
  background: string;
  panels: PanelInfo[][];          // [row][col]
  column_labels: AxisLabel[];
  row_labels: AxisLabel[];
  column_headers: HeaderLevel[];
  row_headers: HeaderLevel[];
  resolution_entries: Record<string, number>;
  dpi: number;
  normalize_widths?: boolean;
  normalize_mode?: string;        // "width" or "height"
  scale_definitions?: Array<{ name: string; unit: string; value_per_pixel: number }>;
  parked_panels?: Array<{ original_row: number; original_col: number; panel: PanelInfo; image_name: string }>;
  show_column_labels?: boolean;
  show_row_labels?: boolean;
}

// ── API response types ───────────────────────────────────

export interface UploadResponse {
  names: string[];
  thumbnails: Record<string, string>;  // name -> base64 png
}

export interface ImagesListResponse {
  names: string[];
  used: string[];
  hidden?: string[];
}

export interface PreviewResponse {
  image: string;   // base64 png
  width: number;
  height: number;
  format: string;
}

export interface ProjectLoadResponse {
  config: FigureConfig;
  image_names: string[];
  thumbnails: Record<string, string>;
  /** Analysis state previously saved into the .mpf zip (tabs, plots,
   *  tables). Null when the project was saved without it. */
  analysis?: AnalysisPayload | null;
}

/** Wire-format for the analysis blob stuffed into / pulled out of a
 *  .mpf zip. The backend stores `plots` as binary PNG files and
 *  `tables` as utf-8 CSV files keyed by id; both are transported
 *  as base64 (plots) and raw text (tables) in this JSON envelope.
 *  `manifest` is arbitrary frontend-defined JSON — currently it
 *  holds the tabs array with per-plot/per-table id references. */
export interface AnalysisPayload {
  manifest: unknown;
  plots: Record<string, string>;   // {id: base64 PNG}
  tables: Record<string, string>;  // {id: CSV text}
}

export interface ImageGroup {
  id: string;
  name: string;
  imageNames: string[];
}
