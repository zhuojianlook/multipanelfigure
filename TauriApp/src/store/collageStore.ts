/* ──────────────────────────────────────────────────────────
   collageStore — state for the Collage Assembly workspace.

   A collage is a free-form canvas of items (each an arbitrary
   image — a rendered multi-panel figure, an imported PNG/JPEG,
   etc.) positioned by x/y/width/height with optional rotation.

   For v1 the entire collage is held in-memory (and persisted to
   localStorage on every change so a window reload preserves
   work). Disk save/load via the backend can come later.
   ────────────────────────────────────────────────────────── */

import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

export type CollageItemKind = "figure" | "image" | "text" | "line";

/** Editable text slots of an R (ggplot) plot, surfaced as clickable hotspots
 *  on the collage canvas. Mirrors the backend `text_overrides` slot names. */
export type RTextSlot =
  | "title" | "subtitle" | "xaxis" | "yaxis" | "caption"
  | "legend_title" | "xticks" | "yticks" | "legend_text";

/** The editable text slots of an R/ggplot figure, in display order. Shared by
 *  the canvas "Edit text" dropdown and the Synchronize panel's nested tree. */
export const R_TEXT_SLOTS: { slot: RTextSlot; label: string }[] = [
  { slot: "title", label: "Title" },
  { slot: "subtitle", label: "Subtitle" },
  { slot: "xaxis", label: "X axis title" },
  { slot: "yaxis", label: "Y axis title" },
  { slot: "xticks", label: "X tick labels" },
  { slot: "yticks", label: "Y tick labels" },
  { slot: "legend_title", label: "Legend title" },
  { slot: "legend_text", label: "Legend labels" },
  { slot: "caption", label: "Caption" },
];

/** Slots whose TEXT can be renamed (ggplot labs()). The rest (tick labels,
 *  legend labels) are data-driven — only size/colour/font apply. */
export const R_TEXT_BEARING: Set<RTextSlot> = new Set<RTextSlot>([
  "title", "subtitle", "xaxis", "yaxis", "caption", "legend_title",
]);

/** A single R text-element override (text content + style). All fields
 *  optional — only set ones are applied. Sizes are in points. */
export type RTextOverride = {
  text?: string;
  size?: number;
  color?: string;
  bold?: boolean;
  italic?: boolean;
  /** ggplot font family (element_text(family=…)). Only families R has
   *  registered render — generic "sans"/"serif"/"mono" always work; common
   *  system names work where the device resolves them. "" / undefined = keep
   *  the plot's default. */
  font?: string;
};

/** One styled run within a header (per-character styling support). */
export type CollageHeaderSegment = {
  text: string;
  color?: string | null;
  font_name?: string | null;
  font_size?: number | null;       // pt; null = inherit header pt
  font_style?: string[];
};

/** A column/row header extracted from a decomposed .mpf, in
 *  resolution-independent figure fractions (0..1, y from BOTTOM —
 *  matplotlib convention). The collage renders these as live HTML
 *  overlays so the "unify headers" button is instant. */
export type CollageHeader = {
  orientation: "column" | "row";
  position: string;                // Top|Bottom (column) or Left|Right (row)
  cx_frac: number;
  cy_frac: number;
  span_x0_frac?: number;
  span_x1_frac?: number;
  span_y0_frac?: number;
  span_y1_frac?: number;
  rotation: number;                // degrees (matplotlib CCW-positive)
  text: string;
  segments: CollageHeaderSegment[];
  font_name: string;
  font_size_pt: number;
  color: string;
  font_style: string[];
  line_color: string;
  line_width: number;
  line_style: string;
  line_length: number;
  end_caps: boolean;
  level_idx: number;
};

export type CollageItem = {
  id: string;
  /** "figure" = produced via Add to Collage from the multi-panel builder.
   *  "image" = imported from disk via Import Image. The distinction lets
   *  the strip badge them and lets future return-to-builder code route
   *  only figure-kinded items. */
  kind: CollageItemKind;
  /** base64-encoded PNG/JPEG payload. data: prefix included for direct <img src> use. */
  src: string;
  /** Display name shown in tooltips / future label rendering. */
  name: string;
  /** Top-left position in canvas pixels. */
  x: number;
  y: number;
  /** Render size in canvas pixels. Aspect ratio of the source image
   *  is preserved by the consumer; w + h are independently adjustable. */
  w: number;
  h: number;
  /** Source image's natural pixel dimensions — used for export at full
   *  quality and for resize handles that preserve aspect ratio. */
  naturalW: number;
  naturalH: number;
  /** Rotation in degrees. Phase 2. */
  rotation: number;
  /** Stacking order. Higher z is drawn later (on top). */
  z: number;
  /** Absolute path of the .mpf this figure-kind item came from.
   *  Set when the user saves their project before clicking
   *  "Add to Collage" (and on subsequent re-renders). The Multi-Panel
   *  Builder button uses it to round-trip back to that figure's
   *  editor. Doubles as a uniqueness key — a single .mpf can appear
   *  at most once in the collage. Null for image-kind items. */
  projectPath: string | null;
  /** True when this item was an analysis plot moved here from the
   *  Analysis dialog. The collage builder doesn't yet save to disk,
   *  so on app-close we warn the user to download any such plots. */
  fromAnalysis?: boolean;
  /** Provenance for re-running an analysis (R) plot at a new font size via
   *  the collage's "Synchronize headers". Captured when the plot is added
   *  from the Analysis dialog. rPlotIndex selects which plot from the
   *  re-run's output array (a script may emit several). */
  rCode?: string;
  rDataCsv?: string;
  rInterpreter?: string | null;
  rPlotIndex?: number;
  /** Per-text-element overrides for an R (ggplot) plot item, keyed by slot.
   *  Edited by clicking the plot's text hotspots on the canvas; re-applied on
   *  every re-render so title/axis/legend customizations persist. */
  rTextOverrides?: Partial<Record<RTextSlot, RTextOverride>>;
  /** Text-item ("text" kind) styling. `text` is the content; the rest are
   *  optional style overrides with sensible defaults. */
  text?: string;
  /** Whole-box font size in POINTS (pt). Rendered as fontSize × PT_TO_PX px.
   *  Legacy items stored px and are migrated on load (see loadInitial). */
  fontSize?: number;
  /** Marker that `fontSize` (and any styledSegments sizes) are in points.
   *  Absent on legacy px-based items so the loader can convert them once. */
  fontSizeUnit?: "pt";
  fontColor?: string;
  fontFamily?: string;
  fontBold?: boolean;
  fontItalic?: boolean;
  fontUnderline?: boolean;
  align?: "left" | "center" | "right";
  /** Per-character rich styling for a text item. When present (non-empty)
   *  it supersedes the whole-box font* props on render/export, enabling
   *  mixed fonts, colours, bold/italic/underline/strikethrough and
   *  super/subscript within a single text box (edited via RichTextEditor).
   *  `text` is kept in sync as the plain concatenation for search/snapshot. */
  styledSegments?: import("../api/types").StyledSegment[];
  /** For image-kind items: the ORIGINAL (un-cropped) data URL, kept so the
   *  user can re-crop from the full image. Set on first crop. */
  cropOrigSrc?: string;
  /** "line" (divider) styling — a straight line drawn across the item box
   *  (rotate via `rotation`). */
  lineColor?: string;
  lineThickness?: number;
  lineStyle?: "solid" | "dashed" | "dotted";
  /** Decomposed-figure overlay data (figure-kind items only). When
   *  present, `src` is the header-LESS body raster and `headers` are
   *  rendered as live HTML overlays — so resizing + the "unify headers"
   *  button never re-render through matplotlib. `bodyNaturalW/H` is the
   *  body's natural pixel size (distinct from naturalW/H which, for
   *  back-compat, may still hold the full figure size). */
  headers?: CollageHeader[];
  bodyNaturalW?: number;
  bodyNaturalH?: number;
  /** Auto panel label (a, b, c…) overlaid on a figure/image item. Present =
   *  this element is labeled. offsetX/offsetY place the label's top-left
   *  relative to the item's top-left in canvas (page) px (default: top-left
   *  corner). `text` overrides the auto letter when the user edits it. The
   *  label lives ON the item (moves + rotates with it) and never appears in the
   *  items timeline. */
  panelLabel?: PanelLabel;
};

export type PanelLabel = {
  offsetX: number;
  offsetY: number;
  /** Explicit text override; undefined = use the auto reading-order letter. */
  text?: string;
  /** Font in POINTS; sensible bold defaults applied at creation. */
  fontSize?: number;
  color?: string;
  bold?: boolean;
  fontFamily?: string;
};

/** Default panel-label appearance (journal style: bold, ~16pt). */
export const PANEL_LABEL_DEFAULT_PT = 16;
export const PANEL_LABEL_DEFAULT_COLOR = "#000000";

/** Items that can carry a panel label: visual content (figures/MPFs + images,
 *  including R/analysis plots). Text boxes and divider lines are excluded. */
export function isLabelable(it: CollageItem): boolean {
  return it.kind === "figure" || it.kind === "image";
}

/** A fresh top-left panel label for an item of the given display width. */
function _defaultPanelLabel(_itemW: number): PanelLabel {
  const px = PANEL_LABEL_DEFAULT_PT * PT_TO_PX;
  const inset = Math.max(8, Math.round(px * 0.35));
  return {
    offsetX: inset,
    offsetY: inset,
    fontSize: PANEL_LABEL_DEFAULT_PT,
    color: PANEL_LABEL_DEFAULT_COLOR,
    bold: true,
  };
}

/** Render labeled items in figure reading order (top-to-bottom, left-to-right),
 *  clustering items into rows by vertical overlap so a slightly-misaligned row
 *  still reads left→right. Only items that HAVE a panelLabel participate. */
export function panelReadingOrder(items: CollageItem[]): CollageItem[] {
  const labeled = items.filter((it) => it.panelLabel && isLabelable(it));
  const byTop = [...labeled].sort((a, b) => a.y - b.y);
  const rows: CollageItem[][] = [];
  for (const it of byTop) {
    const cy = it.y + it.h / 2;
    const row = rows.find((r) => {
      const rcy = r[0].y + r[0].h / 2;
      const tol = Math.min(it.h, r[0].h) * 0.5;
      return Math.abs(cy - rcy) <= tol;
    });
    if (row) row.push(it);
    else rows.push([it]);
  }
  rows.sort((a, b) => Math.min(...a.map((i) => i.y)) - Math.min(...b.map((i) => i.y)));
  for (const r of rows) r.sort((a, b) => a.x - b.x);
  return rows.flat();
}

/** Spreadsheet-style letter for a 0-based index: a,b,…,z,aa,ab,… */
function _indexToLetter(i: number): string {
  let n = i, s = "";
  do { s = String.fromCharCode(97 + (n % 26)) + s; n = Math.floor(n / 26) - 1; } while (n >= 0);
  return s;
}

/** Format an auto letter per the global case/paren options. */
export function formatPanelLetter(index: number, upper: boolean, paren: boolean): string {
  let s = _indexToLetter(index);
  if (upper) s = s.toUpperCase();
  return paren ? `(${s})` : s;
}

/** Map of itemId → displayed label text, honoring per-label text overrides and
 *  the global case/paren format, computed in reading order. `startIndex`
 *  defaults to 0 (the natural start); for "continuous" multi-page numbering
 *  the caller passes the count of labelable items on all PRIOR pages so this
 *  page's letters pick up where the last page left off. */
export function panelLabelTextMap(
  items: CollageItem[], upper: boolean, paren: boolean,
  startIndex: number = 0,
): Record<string, string> {
  const order = panelReadingOrder(items);
  const out: Record<string, string> = {};
  order.forEach((it, i) => {
    const ov = it.panelLabel?.text;
    out[it.id] = (ov !== undefined && ov !== "") ? ov : formatPanelLetter(startIndex + i, upper, paren);
  });
  return out;
}

/** Count of items on a page that would PARTICIPATE in panel labelling (i.e.
 *  appear in `panelReadingOrder`). Used by continuous multi-page numbering
 *  to compute the offset for each page's letters. */
export function countLabelableItems(items: CollageItem[]): number {
  return panelReadingOrder(items).length;
}

/** A text element of an mpf figure, for per-element font sync. `geom`
 *  (when present) is the element's on-figure bbox in figure fractions
 *  (cx/cy centre, s0/s1 span) so the collage can overlay a clickable
 *  hotspot directly on the figure. */
export type CollageFigElement = {
  id: string;
  type: string;
  text: string;
  font_size: number | null;
  font_name?: string | null;
  color?: string | null;
  font_style?: string[];
  styled_segments?: import("../api/types").StyledSegment[];
  geom?: { orientation: "column" | "row"; cx: number; cy: number; s0: number; s1: number };
};

/** A per-element STYLE override applied to a collage figure on re-render
 *  (double-click customization). Mirrors the backend element_overrides. */
export type CollageElemOverride = {
  font_name?: string;
  color?: string;
  font_style?: string[];
  font_size?: number;
  styled_segments?: import("../api/types").StyledSegment[];
};

export type WorkspaceMode = "builder" | "collage" | "analysis";

/** An open .mpf document represented as a tab. `path` is null for a
 *  never-saved ("Untitled") working document. Order in `openDocs` is
 *  stable — switching tabs does NOT reorder them. */
export type DocTab = {
  id: string;
  path: string | null;
  name: string;
};

/** A single collage page — now a VISUAL REGION of a shared "spread"
 *  canvas, not a separate canvas with its own items. All items live in
 *  state.items with global (spread-space) x/y; the page just defines a
 *  rectangle (width × height + background colour) tiled left-to-right
 *  on the spread with `state.pageGutter` between adjacent pages. The
 *  active page is the one the preset / background / guide settings on
 *  the top-level state describe, and the one the page-tabs strip will
 *  scroll the viewport to focus on.
 *
 *  Per-page items (the previous model) are migrated on load by
 *  concatenating each page's items into the global list and shifting
 *  their x by the page's cumulative offset on the spread. */
export type CollagePageSnapshot = {
  /** Page width in canvas pixels (NOT spread width). */
  canvasW: number;
  /** Page height in canvas pixels. */
  canvasH: number;
  /** Per-page background colour (RGB hex or "transparent"). */
  background: string;
  /** Per-page guide overlay — same semantics as the legacy top-level
   *  fields, but scoped per-page so each spread page can carry its own
   *  column-guide preset. */
  guideColumns: number;
  guideGutter: number;
  guidesVisible: boolean;
  /** Per-page panel-labels toggle. */
  panelLabelsOn: boolean;
};

export type CollagePage = {
  id: string;
  name: string;
  snapshot: CollagePageSnapshot;
};

/** Geometry of every page on the spread — computed each render from the
 *  pages array, the page gutter, and the chosen orientation (currently
 *  horizontal only). The arrays are parallel to `pages`. */
export type CollagePageGeom = {
  /** spread-x of the page's top-left corner (px). */
  offsetX: number;
  /** spread-y of the page's top-left corner (px). */
  offsetY: number;
  /** page width (px) (= snapshot.canvasW). */
  width: number;
  /** page height (px) (= snapshot.canvasH). */
  height: number;
};

/** Compute every page's spread-space rectangle. Horizontal layout: pages
 *  are tiled left-to-right with `gutter` px between them; each page's
 *  vertical alignment is top. Pure function (no React / store
 *  dependencies) so it can be called from anywhere (render, save, etc.). */
export function computePageGeometry(pages: CollagePage[], gutter: number): {
  geoms: CollagePageGeom[]; spreadW: number; spreadH: number;
} {
  const g = Math.max(0, gutter);
  const geoms: CollagePageGeom[] = [];
  let x = 0;
  let maxH = 0;
  for (const p of pages) {
    const w = Math.max(1, p.snapshot.canvasW);
    const h = Math.max(1, p.snapshot.canvasH);
    geoms.push({ offsetX: x, offsetY: 0, width: w, height: h });
    x += w + g;
    if (h > maxH) maxH = h;
  }
  // Strip the trailing gutter so the spread bounds the last page exactly.
  const spreadW = pages.length > 0 ? x - g : 0;
  const spreadH = maxH;
  return { geoms, spreadW, spreadH };
}

/** Panel-label numbering modes across multi-page collages:
 *   - "continuous" → page 1 = A..F, page 2 = G..M, … (offset by previous-
 *     page label count, naturally restarts at the start of each row order).
 *   - "per-page" → every page starts at A independently. */
export type PanelLabelNumberingMode = "continuous" | "per-page";

/** Default canvas: Nature full-page proportions (183 mm × 247 mm) at 300 DPI
 *  → 2161 × 2917 px. Common scientific-publication target so users don't
 *  have to look up dimensions before assembling. */
export const DEFAULT_CANVAS_W = 2161;
export const DEFAULT_CANVAS_H = 2917;
/** Grid step in canvas pixels. 50 px ≈ 4.2 mm at 300 DPI — fine enough for
 *  alignment without being claustrophobic on the screen. */
export const DEFAULT_GRID_STEP = 50;
/** Canvas is a 300-DPI virtual page, so one typographic point = 300/72 px.
 *  Text-box font sizes are stored in POINTS (matching figure headers + the
 *  Synchronize panel); renderers multiply by this to get canvas pixels. */
export const PT_TO_PX = 300 / 72;
/** Default text-box size in points (was 28 px ≈ 6.7 pt; bumped to a legible
 *  publication size on a 300-DPI page). */
export const DEFAULT_TEXT_PT = 14;

export type CollageState = {
  /** Top-level workspace toggle. The builder shows the panel grid +
   *  preview; the collage shows the assembly canvas. */
  mode: WorkspaceMode;
  items: CollageItem[];
  /** Logical canvas size in pixels. Treated as a virtual page; the
   *  view scales to fit the available area. */
  canvasW: number;
  canvasH: number;
  /** Background color of the canvas. The sentinel "transparent" renders a
   *  checkerboard in the editor and exports a PNG with alpha (no fill). */
  background: string;
  /** Primary selected item id (the last one clicked), or null. Drives the
   *  single-item affordances (resize handles, the strip highlight). */
  selectedId: string | null;
  /** Full multi-selection set. selectedId is always the last entry (or
   *  null when empty). Move operations act on every id in this set. */
  selectedIds: string[];
  /** Gridline visibility + snap-to-grid step. Both default ON. */
  gridVisible: boolean;
  snapEnabled: boolean;
  gridStep: number;
  /** Journal column guide overlay. guideColumns is the number of columns
   *  the active preset implies (0 = no guides); guideGutter is the gutter
   *  width in canvas px between columns. Set when a column preset is
   *  applied; toggleable via guidesVisible. */
  guideColumns: number;
  guideGutter: number;
  guidesVisible: boolean;
  /** Export resolution (DPI) for Save Collage. The canvas is a 300-DPI
   *  virtual page, so factor = exportDpi/300 scales the output pixels; the
   *  exported PNG is also tagged with this DPI (pHYs). Default 300. */
  exportDpi: number;
  /** Output file format for Save Collage: "png" | "jpeg" | "tiff" | "pdf".
   *  PNG is composed client-side; the rest are converted by the backend
   *  (PIL) with the chosen DPI embedded. Default "png". */
  exportFormat: string;
  /** Panel-label auto-letter format (global). upper: A,B,C vs a,b,c.
   *  paren: (a) vs a. Persisted. */
  panelLabelUpper: boolean;
  panelLabelParen: boolean;
  /** When on, every labelable item (incl. newly added ones) carries a panel
   *  label. Set by "Label panels" / cleared by "Clear labels". Persisted.
   *  Default ON. */
  panelLabelsOn: boolean;
  /** One-time marker: once true, panelLabelsOn reflects the user's explicit
   *  choice. Before it's set (legacy/older data), panelLabelsOn is forced ON so
   *  the new "labels on by default" applies to existing collages too. */
  panelLabelsDefaultedOn: boolean;
  /** Target visual point size for ALL figure-kind item headers /
   *  primary labels in the collage. Null = no normalisation (each
   *  item shows headers at its own pt × its collage scale). When
   *  set, the renderer compensates per-item so headers land at this
   *  pt regardless of how the user has scaled the figure. */
  globalHeaderPt: number | null;

  /** Collage pages (multi-page collage). Always has length >= 1.
   *  Pages are REGIONS of one shared spread — all items live in
   *  state.items with global (spread-space) x/y. The active page is
   *  the one the preset / background / guide settings on the top-level
   *  state describe (and the one the page-tabs strip will scroll the
   *  viewport to focus on). */
  pages: CollagePage[];
  /** Id of the currently-active page (must be in `pages`). */
  activePageId: string;
  /** Whether panel-label letters (A,B,C,…) continue across pages or
   *  restart at A on every page. */
  panelLabelNumberingMode: PanelLabelNumberingMode;
  /** Gutter (px) between adjacent pages on the spread. Default 60 —
   *  big enough to read as separate pages, small enough not to dominate
   *  the viewport. */
  pageGutter: number;

  /** Open .mpf documents shown as tabs. Order is stable. Session-only
   *  (not persisted) — rebuilt on mount from the current builder doc +
   *  collage figure items. */
  openDocs: DocTab[];
  /** id of the doc currently loaded in the builder. */
  activeDocId: string | null;
  /** ids of open docs that have unsaved in-memory snapshots — edits that
   *  aren't live in the backend right now (the user switched away from a
   *  dirty tab). Reactive mirror of projectNav's snapshot map so tabs can
   *  show an unsaved dot. Session-only, never persisted. */
  snapshotDirtyDocIds: string[];

  /** Per-element font-sync state (shared so the canvas can overlay
   *  clickable hotspots in sync with the sidebar checkboxes). Session-only.
   *  - elemSyncItemId: figure item whose element hotspots show on the canvas
   *  - elemListByItem: each figure's text elements (+ geometry)
   *  - elemSelByItem: each figure's per-element selection */
  elemSyncItemId: string | null;
  elemListByItem: Record<string, CollageFigElement[]>;
  elemSelByItem: Record<string, Record<string, boolean>>;
  /** Element currently hovered in the sidebar tree, highlighted on the
   *  canvas hotspot (and vice-versa). Session-only. */
  hoveredElem: { itemId: string; elemId: string } | null;
  /** Per-figure per-element STYLE overrides (double-click customization),
   *  keyed by item id then element id. Persisted so customizations survive
   *  reloads and re-renders. */
  elemOverridesByItem: Record<string, Record<string, CollageElemOverride>>;

  setMode: (m: WorkspaceMode) => void;
  setGridVisible: (v: boolean) => void;
  setSnapEnabled: (v: boolean) => void;
  setGridStep: (n: number) => void;
  setGlobalHeaderPt: (pt: number | null) => void;

  addItem: (item: Omit<CollageItem, "id" | "z" | "rotation" | "kind" | "projectPath"> & Partial<Pick<CollageItem, "rotation" | "kind" | "projectPath">>) => string;
  updateItem: (id: string, patch: Partial<CollageItem>) => void;
  removeItem: (id: string) => void;
  moveItem: (id: string, dx: number, dy: number) => void;
  bringToFront: (id: string) => void;
  /** Change an item's stacking order: to the very front/back, or one step
   *  forward/backward relative to its neighbour. */
  reorderItem: (id: string, dir: "front" | "back" | "forward" | "backward") => void;
  setSelectedId: (id: string | null) => void;
  /** Replace the whole multi-selection. */
  setSelectedIds: (ids: string[]) => void;
  /** Add/remove an id from the multi-selection (shift/cmd-click). */
  toggleSelected: (id: string) => void;
  setCanvasSize: (w: number, h: number) => void;
  setBackground: (bg: string) => void;
  /** Set the column guide overlay (columns, gutter px). columns=0 clears it. */
  setColumnGuides: (columns: number, gutter: number) => void;
  setGuidesVisible: (v: boolean) => void;
  setExportDpi: (dpi: number) => void;
  setExportFormat: (fmt: string) => void;
  clear: () => void;

  // ── Panel labels (a, b, c…) ──
  /** Add a default top-right panel label to every labelable item that lacks
   *  one (figures/MPFs + images). */
  addPanelLabels: () => void;
  /** Remove the panel label from every item. */
  clearPanelLabels: () => void;
  /** Merge a patch into one item's panel label (creates it if missing). */
  setPanelLabel: (id: string, patch: Partial<PanelLabel>) => void;
  /** Remove one item's panel label. */
  removePanelLabel: (id: string) => void;
  /** Nudge one label's offset by (dx, dy) in canvas px. */
  movePanelLabel: (id: string, dx: number, dy: number) => void;
  setPanelLabelUpper: (v: boolean) => void;
  setPanelLabelParen: (v: boolean) => void;
  setPanelLabelNumberingMode: (m: PanelLabelNumberingMode) => void;

  // ── Pages (multi-page collage) ──
  /** Add a new blank page to the right of the active one and switch to it.
   *  Returns the new page's id. */
  pageAdd: (name?: string) => string;
  /** Duplicate the given page (or active if no id) and switch to the copy. */
  pageDuplicate: (id?: string) => string;
  /** Remove a page. Callers should confirm with the user if the page has
   *  items (`page.snapshot.items.length > 0`); the store does NOT prompt. */
  pageRemove: (id: string) => void;
  /** Switch the active page. Snapshots the current flat fields into the
   *  outgoing page, then loads the target's snapshot into flat fields. */
  pageSetActive: (id: string) => void;
  /** Rename a page (display label on the page-tab strip). */
  pageRename: (id: string, name: string) => void;
  /** Reorder pages (drag-to-reorder UI). */
  pageReorder: (orderedIds: string[]) => void;
  /** Snapshot the current top-level flat fields into the active page entry.
   *  Called before persist / before reading multi-page state. Idempotent. */
  pageSyncFlat: () => void;

  // ── Document tabs ──
  /** Append a new doc tab (stable order). Returns its id. */
  docAdd: (doc: { path?: string | null; name?: string }) => string;
  /** Remove a doc tab by id. */
  docRemove: (id: string) => void;
  /** Set the active builder document. */
  docSetActive: (id: string | null) => void;
  /** Update a doc's path + name (e.g. after an Untitled is saved, or a
   *  tab is loaded from disk). */
  docSetPath: (id: string, path: string, name?: string) => void;
  /** Rename a doc tab's display label (user-editable). For an unsaved doc this
   *  also seeds the Save Project modal's filename. */
  docRename: (id: string, name: string) => void;
  /** Ensure a tab exists for `path`; returns its id (existing or new). */
  docEnsure: (path: string, name?: string) => string;
  /** Replace the set of docs that have unsaved in-memory snapshots. */
  setSnapshotDirtyDocIds: (ids: string[]) => void;

  // ── Per-element font sync ──
  /** Set which figure's element hotspots show on the canvas (null = none). */
  setElemSyncItem: (itemId: string | null) => void;
  /** Store a figure's element list, defaulting the selection to its
   *  header/axis-label elements if not already chosen. */
  setElemList: (itemId: string, list: CollageFigElement[]) => void;
  /** Toggle one element's membership in a figure's selection. */
  toggleElemSel: (itemId: string, elemId: string) => void;
  /** Select all / none of a figure's elements. */
  setAllElemSel: (itemId: string, on: boolean) => void;
  /** Apply an element-selection mode to ALL loaded figures at once:
   *  "defaults" = headers + axis labels selected, "none" = nothing. Used
   *  by the figure-level All / None controls so they also clear/restore
   *  the on-figure header selection. */
  applyAllElemSel: (mode: "defaults" | "none") => void;
  /** Set (or clear, when ov is null) a per-element style override. */
  setElemOverride: (itemId: string, elemId: string, ov: CollageElemOverride | null) => void;
  /** Set the element hovered in the sidebar tree (highlighted on canvas). */
  setHoveredElem: (h: { itemId: string; elemId: string } | null) => void;
};

const STORAGE_KEY = "mpfig_collage_v1";

type Persisted = Pick<CollageState, "items" | "canvasW" | "canvasH" | "background" | "gridVisible" | "snapEnabled" | "gridStep" | "globalHeaderPt" | "guideColumns" | "guideGutter" | "guidesVisible" | "exportDpi" | "exportFormat" | "elemOverridesByItem" | "panelLabelUpper" | "panelLabelParen" | "panelLabelsOn" | "panelLabelsDefaultedOn" | "pages" | "activePageId" | "panelLabelNumberingMode" | "pageGutter">;

let _pageSeq = 1;
const _newPageId = () => `page_${Date.now().toString(36)}_${_pageSeq++}`;

/** Snapshot ONLY the per-page top-level fields (NOT items, which are
 *  shared across pages in the spread model). Used by pageAdd / pageRemove /
 *  pageSetActive to keep the page list synced with the active page's
 *  current dims + background + guides. */
function _snapshotFromFlat(s: {
  canvasW: number; canvasH: number; background: string;
  guideColumns: number; guideGutter: number; guidesVisible: boolean; panelLabelsOn: boolean;
}): CollagePageSnapshot {
  return {
    canvasW: s.canvasW,
    canvasH: s.canvasH,
    background: s.background,
    guideColumns: s.guideColumns,
    guideGutter: s.guideGutter,
    guidesVisible: s.guidesVisible,
    panelLabelsOn: s.panelLabelsOn,
  };
}

function loadInitial(): Persisted {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const data = JSON.parse(raw);
      if (data && Array.isArray(data.items)) {
        // Migrate older collage items that pre-date later fields.
        // The 0.1.145 build briefly used `stashPath`; rename to
        // `projectPath` while preserving any existing values.
        const items = data.items.map((it: any) => {
          const migrated: any = {
            kind: it.kind || "image",
            projectPath: it.projectPath ?? it.stashPath ?? null,
            ...it,
          };
          // Migrate legacy px-based text sizes → points (preserving the
          // visual size: pt = px / PT_TO_PX, since renderers now multiply
          // pt × PT_TO_PX). Marked with fontSizeUnit so it runs only once.
          if (migrated.kind === "text" && migrated.fontSizeUnit !== "pt") {
            if (typeof migrated.fontSize === "number") {
              migrated.fontSize = Math.max(1, Math.round(migrated.fontSize / PT_TO_PX));
            }
            if (Array.isArray(migrated.styledSegments)) {
              migrated.styledSegments = migrated.styledSegments.map((s: any) =>
                typeof s?.font_size === "number"
                  ? { ...s, font_size: Math.max(1, Math.round(s.font_size / PT_TO_PX)) }
                  : s,
              );
            }
            migrated.fontSizeUnit = "pt";
          }
          // Lines are 1-D — their box only needs to be tall enough for the end
          // handles. Shrink any over-tall line box to a thin strip, preserving
          // the line's vertical centre. Idempotent (24 is not > 40).
          if (migrated.kind === "line" && typeof migrated.h === "number" && migrated.h > 40) {
            migrated.y = (migrated.y || 0) + (migrated.h - 24) / 2;
            migrated.h = 24;
          }
          return migrated;
        }) as CollageItem[];
        const pageGutter = typeof data.pageGutter === "number" && data.pageGutter >= 0 ? data.pageGutter : 60;
        // Spread model: ALL items live in a single global list (state.items)
        // with global x/y in spread coordinates. Pages are regions of the
        // spread. Migrate two legacy schemas:
        //   (a) Flat single-page (pre-pages): wrap `items` + top-level
        //       canvasW/H/background into one CollagePage.
        //   (b) Per-page items (the 0.1.287 → 0.1.289 schema): for each
        //       page beyond the first, shift its items.x by the cumulative
        //       (sum widths + gutters) offset on the spread, then concat
        //       into the global items list. Page entries keep their dims +
        //       background but lose `items`.
        let pages: CollagePage[] = [];
        let activePageId = "";
        let mergedItems: CollageItem[] = items;
        if (Array.isArray(data.pages) && data.pages.length > 0) {
          // Step 1: normalise page records to the new shape (drop items).
          pages = data.pages.map((p: any, i: number) => ({
            id: typeof p.id === "string" ? p.id : `page_${i}`,
            name: typeof p.name === "string" ? p.name : `Page ${i + 1}`,
            snapshot: {
              canvasW: p?.snapshot?.canvasW || DEFAULT_CANVAS_W,
              canvasH: p?.snapshot?.canvasH || DEFAULT_CANVAS_H,
              background: p?.snapshot?.background || "#FFFFFF",
              guideColumns: typeof p?.snapshot?.guideColumns === "number" ? p.snapshot.guideColumns : 0,
              guideGutter: typeof p?.snapshot?.guideGutter === "number" ? p.snapshot.guideGutter : 0,
              guidesVisible: p?.snapshot?.guidesVisible ?? true,
              panelLabelsOn: p?.snapshot?.panelLabelsOn ?? true,
            },
          }));
          activePageId = typeof data.activePageId === "string"
            && pages.find((p) => p.id === data.activePageId) ? data.activePageId : pages[0].id;
          // Step 2: concatenate any per-page items into the global list with
          // their offsetX shift applied. `data.items` (the top-level field)
          // is from the OLD active page; everything else lives on
          // data.pages[i].snapshot.items. Order pages left-to-right so the
          // shifts line up with the new spread layout.
          let runningX = 0;
          const allItems: CollageItem[] = [];
          for (let i = 0; i < pages.length; i++) {
            const srcPage = data.pages[i];
            const isActive = pages[i].id === activePageId;
            const pageItems = isActive
              ? items
              : (Array.isArray(srcPage?.snapshot?.items) ? srcPage.snapshot.items : []);
            for (const it of pageItems) {
              const x = typeof it.x === "number" ? it.x : 0;
              allItems.push({ ...it, x: x + runningX });
            }
            runningX += pages[i].snapshot.canvasW + pageGutter;
          }
          mergedItems = allItems;
        } else {
          // Legacy single-page → one page, items untouched.
          const id = _newPageId();
          pages = [{ id, name: "Page 1", snapshot: {
            canvasW: data.canvasW || DEFAULT_CANVAS_W,
            canvasH: data.canvasH || DEFAULT_CANVAS_H,
            background: data.background || "#FFFFFF",
            guideColumns: typeof data.guideColumns === "number" ? data.guideColumns : 0,
            guideGutter: typeof data.guideGutter === "number" ? data.guideGutter : 0,
            guidesVisible: data.guidesVisible ?? true,
            panelLabelsOn: data.panelLabelsDefaultedOn ? !!data.panelLabelsOn : true,
          } }];
          activePageId = id;
        }
        // Top-level flat fields are the ACTIVE page's snapshot (so existing
        // consumers reading canvasW/H/background see the active page's dims).
        const activePage = pages.find((p) => p.id === activePageId) ?? pages[0];
        const flat = {
          items: mergedItems,
          canvasW: activePage.snapshot.canvasW,
          canvasH: activePage.snapshot.canvasH,
          background: activePage.snapshot.background,
          guideColumns: activePage.snapshot.guideColumns,
          guideGutter: activePage.snapshot.guideGutter,
          guidesVisible: activePage.snapshot.guidesVisible,
          panelLabelsOn: activePage.snapshot.panelLabelsOn,
        };
        return {
          ...flat,
          gridVisible: data.gridVisible ?? true,
          snapEnabled: data.snapEnabled ?? true,
          gridStep: data.gridStep || DEFAULT_GRID_STEP,
          globalHeaderPt: typeof data.globalHeaderPt === "number" ? data.globalHeaderPt : null,
          exportDpi: typeof data.exportDpi === "number" && data.exportDpi > 0 ? data.exportDpi : 300,
          // TIFF is now the default export format (lossless, single-page TIFF
          // per page, multi-page TIFF supported when multiple pages exist).
          exportFormat: typeof data.exportFormat === "string" ? data.exportFormat : "tiff",
          elemOverridesByItem: (data.elemOverridesByItem && typeof data.elemOverridesByItem === "object") ? data.elemOverridesByItem : {},
          panelLabelUpper: !!data.panelLabelUpper,
          panelLabelParen: !!data.panelLabelParen,
          panelLabelsDefaultedOn: true,
          pages, activePageId,
          panelLabelNumberingMode: (data.panelLabelNumberingMode === "per-page" ? "per-page" : "continuous") as PanelLabelNumberingMode,
          pageGutter,
        };
      }
    }
  } catch {
    /* ignore corrupt storage */
  }
  const initialId = _newPageId();
  const initialFlat = {
    items: [], canvasW: DEFAULT_CANVAS_W, canvasH: DEFAULT_CANVAS_H,
    background: "#FFFFFF", guideColumns: 0, guideGutter: 0, guidesVisible: true,
    panelLabelsOn: true,
  };
  return {
    ...initialFlat,
    gridVisible: true,
    snapEnabled: true,
    gridStep: DEFAULT_GRID_STEP,
    globalHeaderPt: null,
    exportDpi: 300,
    exportFormat: "tiff",
    elemOverridesByItem: {},
    panelLabelUpper: false,
    panelLabelParen: false,
    panelLabelsDefaultedOn: true,
    pages: [{ id: initialId, name: "Page 1", snapshot: _snapshotFromFlat(initialFlat) }],
    activePageId: initialId,
    panelLabelNumberingMode: "continuous",
    pageGutter: 60,
  };
}

function persist(s: Persisted) {
  try {
    // Refresh the active page entry with the current flat fields so the
    // persisted multi-page state reflects what's actually on-screen
    // (rather than a stale snapshot from the last page switch).
    const pages = s.pages.map((p) =>
      p.id === s.activePageId
        ? { ...p, snapshot: _snapshotFromFlat(s) }
        : p);
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      items: s.items,
      canvasW: s.canvasW,
      canvasH: s.canvasH,
      background: s.background,
      gridVisible: s.gridVisible,
      snapEnabled: s.snapEnabled,
      gridStep: s.gridStep,
      globalHeaderPt: s.globalHeaderPt,
      guideColumns: s.guideColumns,
      guideGutter: s.guideGutter,
      guidesVisible: s.guidesVisible,
      exportDpi: s.exportDpi,
      exportFormat: s.exportFormat,
      elemOverridesByItem: s.elemOverridesByItem,
      panelLabelUpper: s.panelLabelUpper,
      panelLabelParen: s.panelLabelParen,
      panelLabelsOn: s.panelLabelsOn,
      panelLabelsDefaultedOn: s.panelLabelsDefaultedOn,
      pages,
      activePageId: s.activePageId,
      panelLabelNumberingMode: s.panelLabelNumberingMode,
      pageGutter: s.pageGutter,
    }));
  } catch {
    /* quota — ignore */
  }
}

let _docSeq = 1;
const _newDocId = () => `doc_${Date.now()}_${_docSeq++}`;

export const useCollageStore = create<CollageState>()(
  immer((set, get) => ({
    ...loadInitial(),
    selectedId: null,
    selectedIds: [],
    mode: "builder" as WorkspaceMode,
    // Seed with a single Untitled working document.
    openDocs: [{ id: "doc_initial", path: null, name: "Untitled" }],
    activeDocId: "doc_initial",
    snapshotDirtyDocIds: [],
    elemSyncItemId: null,
    elemListByItem: {},
    elemSelByItem: {},
    hoveredElem: null,

    setMode: (m) => set((s) => { s.mode = m; }),
    setGridVisible: (v) => { set((s) => { s.gridVisible = v; }); persist(get()); },
    setSnapEnabled: (v) => { set((s) => { s.snapEnabled = v; }); persist(get()); },
    setGridStep: (n) => { set((s) => { s.gridStep = Math.max(2, Math.min(500, Math.round(n))); }); persist(get()); },
    setGlobalHeaderPt: (pt) => { set((s) => { s.globalHeaderPt = pt; }); persist(get()); },

    addItem: (item) => {
      const id = `item_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      set((s) => {
        const maxZ = s.items.reduce((acc, it) => Math.max(acc, it.z), 0);
        const created = {
          ...item,
          id,
          z: maxZ + 1,
          rotation: item.rotation ?? 0,
          kind: item.kind ?? "image",
          projectPath: item.projectPath ?? null,
        } as CollageItem;
        // Auto-label new visual content when panel labelling is on.
        if (s.panelLabelsOn && (created.kind === "figure" || created.kind === "image") && !created.panelLabel) {
          created.panelLabel = _defaultPanelLabel(created.w);
        }
        s.items.push(created);
        s.selectedId = id;
        s.selectedIds = [id];
      });
      const cur = get();
      persist(cur);
      return id;
    },

    updateItem: (id, patch) => {
      set((s) => {
        const it = s.items.find((i) => i.id === id);
        if (it) Object.assign(it, patch);
      });
      persist(get());
    },

    removeItem: (id) => {
      set((s) => {
        s.items = s.items.filter((i) => i.id !== id);
        s.selectedIds = s.selectedIds.filter((x) => x !== id);
        if (s.selectedId === id) s.selectedId = s.selectedIds[s.selectedIds.length - 1] ?? null;
      });
      persist(get());
    },

    moveItem: (id, dx, dy) => {
      set((s) => {
        const it = s.items.find((i) => i.id === id);
        if (it) { it.x += dx; it.y += dy; }
      });
      persist(get());
    },

    bringToFront: (id) => {
      set((s) => {
        const maxZ = s.items.reduce((acc, it) => Math.max(acc, it.z), 0);
        const it = s.items.find((i) => i.id === id);
        if (it) it.z = maxZ + 1;
      });
      persist(get());
    },

    reorderItem: (id, dir) => {
      set((s) => {
        // Normalise z to a dense 0..n-1 ordering first so swaps are reliable
        // even if z values were equal or sparse.
        const sorted = [...s.items].sort((a, b) => a.z - b.z);
        sorted.forEach((it, i) => { const o = s.items.find((x) => x.id === it.id); if (o) o.z = i; });
        const cur = s.items.find((x) => x.id === id);
        if (!cur) return;
        const n = s.items.length;
        if (dir === "front") {
          cur.z = n; // above everyone; renormalised next time
        } else if (dir === "back") {
          for (const it of s.items) it.z += 1;
          cur.z = 0;
        } else if (dir === "forward" && cur.z < n - 1) {
          const above = s.items.find((x) => x.z === cur.z + 1);
          if (above) { above.z = cur.z; cur.z += 1; }
        } else if (dir === "backward" && cur.z > 0) {
          const below = s.items.find((x) => x.z === cur.z - 1);
          if (below) { below.z = cur.z; cur.z -= 1; }
        }
      });
      persist(get());
    },

    setSelectedId: (id) => set((s) => { s.selectedId = id; s.selectedIds = id ? [id] : []; }),

    setSelectedIds: (ids) => set((s) => {
      s.selectedIds = ids;
      s.selectedId = ids.length ? ids[ids.length - 1] : null;
    }),

    toggleSelected: (id) => set((s) => {
      if (s.selectedIds.includes(id)) {
        s.selectedIds = s.selectedIds.filter((x) => x !== id);
      } else {
        s.selectedIds = [...s.selectedIds, id];
      }
      s.selectedId = s.selectedIds[s.selectedIds.length - 1] ?? null;
    }),

    // All per-page setters below MIRROR the new flat-field value onto
    // the active page's snapshot in the same set() call. Without that
    // mirror, the in-memory pages array stays stale until the next
    // persist() (which writes to localStorage but doesn't refresh
    // state), so `computePageGeometry(pages, ...)` keeps returning
    // the OLD spread bounds — meaning preset / background / guide
    // changes don't reflow the viewport until the user does
    // something else that triggers a re-render. Mirroring in-place
    // fixes that.
    setCanvasSize: (w, h) => {
      set((s) => {
        s.canvasW = w; s.canvasH = h;
        const i = s.pages.findIndex((p) => p.id === s.activePageId);
        if (i >= 0) s.pages[i].snapshot = _snapshotFromFlat(s);
      });
      persist(get());
    },

    setBackground: (bg) => {
      set((s) => {
        s.background = bg;
        const i = s.pages.findIndex((p) => p.id === s.activePageId);
        if (i >= 0) s.pages[i].snapshot = _snapshotFromFlat(s);
      });
      persist(get());
    },

    setColumnGuides: (columns, gutter) => {
      set((s) => {
        s.guideColumns = Math.max(0, Math.round(columns));
        s.guideGutter = Math.max(0, Math.round(gutter));
        const i = s.pages.findIndex((p) => p.id === s.activePageId);
        if (i >= 0) s.pages[i].snapshot = _snapshotFromFlat(s);
      });
      persist(get());
    },

    setGuidesVisible: (v) => {
      set((s) => {
        s.guidesVisible = v;
        const i = s.pages.findIndex((p) => p.id === s.activePageId);
        if (i >= 0) s.pages[i].snapshot = _snapshotFromFlat(s);
      });
      persist(get());
    },

    setExportDpi: (dpi) => {
      set((s) => { s.exportDpi = Math.max(72, Math.min(1200, Math.round(dpi))); });
      persist(get());
    },

    setExportFormat: (fmt) => {
      set((s) => { s.exportFormat = fmt; });
      persist(get());
    },

    clear: () => {
      set((s) => { s.items = []; s.selectedId = null; s.selectedIds = []; });
      persist(get());
    },

    // ── Panel labels (a, b, c…) ── (default placement: top-left corner)
    addPanelLabels: () => {
      set((s) => {
        s.panelLabelsOn = true;
        for (const it of s.items) {
          if ((it.kind === "figure" || it.kind === "image") && !it.panelLabel) {
            it.panelLabel = _defaultPanelLabel(it.w);
          }
        }
      });
      persist(get());
    },

    clearPanelLabels: () => {
      set((s) => { s.panelLabelsOn = false; for (const it of s.items) delete it.panelLabel; });
      persist(get());
    },

    setPanelLabel: (id, patch) => {
      set((s) => {
        const it = s.items.find((i) => i.id === id);
        if (!it) return;
        it.panelLabel = { ...(it.panelLabel ?? _defaultPanelLabel(it.w)), ...patch };
      });
      persist(get());
    },

    removePanelLabel: (id) => {
      set((s) => { const it = s.items.find((i) => i.id === id); if (it) delete it.panelLabel; });
      persist(get());
    },

    movePanelLabel: (id, dx, dy) => {
      set((s) => {
        const it = s.items.find((i) => i.id === id);
        if (it?.panelLabel) { it.panelLabel.offsetX += dx; it.panelLabel.offsetY += dy; }
      });
      persist(get());
    },

    setPanelLabelUpper: (v) => { set((s) => { s.panelLabelUpper = v; }); persist(get()); },
    setPanelLabelParen: (v) => { set((s) => { s.panelLabelParen = v; }); persist(get()); },
    setPanelLabelNumberingMode: (m) => { set((s) => { s.panelLabelNumberingMode = m; }); persist(get()); },

    // ── Pages (spread-model: items live globally, pages are regions) ──
    /** Mirror the active page's snapshot from the flat fields. Items are
     *  NOT in the snapshot — they live on state.items, shared across the
     *  whole spread. Called by every mutator that changes a per-page
     *  field (canvas dims, background, guides) so the persisted pages
     *  array always reflects the live state. */
    pageSyncFlat: () => {
      set((s) => {
        const i = s.pages.findIndex((p) => p.id === s.activePageId);
        if (i >= 0) {
          s.pages[i].snapshot = _snapshotFromFlat(s);
        }
      });
    },

    pageAdd: (name) => {
      const id = _newPageId();
      set((s) => {
        // Sync the active page's snapshot first.
        const ai = s.pages.findIndex((p) => p.id === s.activePageId);
        if (ai >= 0) s.pages[ai].snapshot = _snapshotFromFlat(s);
        // Pick a name that doesn't collide ("Page 1" / "Page 2" / ...).
        const usedNames = new Set(s.pages.map((p) => p.name));
        let auto = name?.trim() || "";
        if (!auto) {
          let n = s.pages.length + 1;
          while (usedNames.has(`Page ${n}`)) n++;
          auto = `Page ${n}`;
        }
        // New page inherits the outgoing page's dims + background so the
        // spread looks visually uniform until the user picks a different
        // preset. Items array is global → not duplicated; the new page
        // just opens up an empty region to the right.
        const fresh: CollagePageSnapshot = {
          canvasW: s.canvasW,
          canvasH: s.canvasH,
          background: s.background,
          guideColumns: s.guideColumns,
          guideGutter: s.guideGutter,
          guidesVisible: s.guidesVisible,
          panelLabelsOn: s.panelLabelsOn,
        };
        // Insert to the RIGHT of the currently-active page.
        const insertAt = ai >= 0 ? ai + 1 : s.pages.length;
        s.pages.splice(insertAt, 0, { id, name: auto, snapshot: fresh });
        // Activate the new page (flat fields already match its snapshot
        // because it inherited from the outgoing one — no swap needed).
        s.activePageId = id;
      });
      persist(get());
      return id;
    },

    pageDuplicate: (id) => {
      // In the spread model "duplicate" just makes another page region
      // with the same dims + background (items can't be duplicated
      // independently — they're global). For an actual content copy the
      // user multi-selects items and pastes them into the new region.
      const newId = _newPageId();
      set((s) => {
        const ai = s.pages.findIndex((p) => p.id === s.activePageId);
        if (ai >= 0) s.pages[ai].snapshot = _snapshotFromFlat(s);
        const src = s.pages.find((p) => p.id === (id || s.activePageId));
        if (!src) return;
        const cloned: CollagePageSnapshot = { ...src.snapshot };
        const srcIdx = s.pages.findIndex((p) => p.id === src.id);
        const usedNames = new Set(s.pages.map((p) => p.name));
        let copyName = `${src.name} copy`;
        let i = 2;
        while (usedNames.has(copyName)) { copyName = `${src.name} copy ${i++}`; }
        s.pages.splice(srcIdx + 1, 0, { id: newId, name: copyName, snapshot: cloned });
        s.activePageId = newId;
      });
      persist(get());
      return newId;
    },

    pageRemove: (id) => {
      // Items inside the removed page's region remain in state.items —
      // they may now overlap a neighbouring page or sit in the gutter.
      // CollageView shows them at their original x/y; the user can
      // move them. Confirmation when items overlap the removed region
      // is handled by the page-tabs UI (it knows the page geometry).
      set((s) => {
        if (s.pages.length <= 1) return;  // never remove the last page
        const idx = s.pages.findIndex((p) => p.id === id);
        if (idx < 0) return;
        const wasActive = s.activePageId === id;
        s.pages.splice(idx, 1);
        if (wasActive) {
          const next = s.pages[Math.max(0, idx - 1)];
          s.activePageId = next.id;
          // Re-load the new active page's snapshot into the flat fields
          // so the preset / background controls reflect the new active.
          s.canvasW = next.snapshot.canvasW;
          s.canvasH = next.snapshot.canvasH;
          s.background = next.snapshot.background;
          s.guideColumns = next.snapshot.guideColumns;
          s.guideGutter = next.snapshot.guideGutter;
          s.guidesVisible = next.snapshot.guidesVisible;
          s.panelLabelsOn = next.snapshot.panelLabelsOn;
        }
      });
      persist(get());
    },

    pageSetActive: (id) => {
      // Spread-model semantics: switching the active page does NOT swap
      // the items — they're shared. We just mirror the new active
      // page's snapshot into the flat fields so the preset / background
      // controls describe it. The viewport scroll-to-page is done by
      // the page-tabs UI in CollageView (it has the spread geometry).
      set((s) => {
        if (id === s.activePageId) return;
        const target = s.pages.find((p) => p.id === id);
        if (!target) return;
        const ai = s.pages.findIndex((p) => p.id === s.activePageId);
        if (ai >= 0) s.pages[ai].snapshot = _snapshotFromFlat(s);
        s.activePageId = id;
        s.canvasW = target.snapshot.canvasW;
        s.canvasH = target.snapshot.canvasH;
        s.background = target.snapshot.background;
        s.guideColumns = target.snapshot.guideColumns;
        s.guideGutter = target.snapshot.guideGutter;
        s.guidesVisible = target.snapshot.guidesVisible;
        s.panelLabelsOn = target.snapshot.panelLabelsOn;
      });
      persist(get());
    },

    pageRename: (id, name) => {
      set((s) => {
        const p = s.pages.find((x) => x.id === id);
        const trimmed = name.trim();
        if (p && trimmed) p.name = trimmed;
      });
      persist(get());
    },

    pageReorder: (orderedIds) => {
      set((s) => {
        const byId = new Map(s.pages.map((p) => [p.id, p] as const));
        const next: CollagePage[] = [];
        for (const id of orderedIds) {
          const p = byId.get(id);
          if (p) { next.push(p); byId.delete(id); }
        }
        // Append any pages not mentioned in `orderedIds` (defensive).
        for (const p of byId.values()) next.push(p);
        if (next.length === s.pages.length) s.pages = next;
      });
      persist(get());
    },

    // ── Document tabs (session-only, not persisted) ──
    docAdd: (doc) => {
      const id = _newDocId();
      const name = doc.name
        ?? (doc.path ? (doc.path.split(/[\\/]/).pop()?.replace(/\.mpf$/i, "") || doc.path) : "Untitled");
      set((s) => { s.openDocs.push({ id, path: doc.path ?? null, name }); });
      return id;
    },

    docRemove: (id) => {
      set((s) => {
        s.openDocs = s.openDocs.filter((d) => d.id !== id);
        if (s.activeDocId === id) s.activeDocId = s.openDocs[0]?.id ?? null;
      });
    },

    docSetActive: (id) => set((s) => { s.activeDocId = id; }),

    docSetPath: (id, path, name) => set((s) => {
      const d = s.openDocs.find((x) => x.id === id);
      if (d) {
        d.path = path;
        d.name = name ?? (path.split(/[\\/]/).pop()?.replace(/\.mpf$/i, "") || path);
      }
    }),

    docRename: (id, name) => set((s) => {
      const d = s.openDocs.find((x) => x.id === id);
      const trimmed = name.trim();
      if (d && trimmed) d.name = trimmed;
    }),

    docEnsure: (path, name) => {
      const existing = get().openDocs.find((d) => d.path === path);
      if (existing) return existing.id;
      return get().docAdd({ path, name });
    },

    setSnapshotDirtyDocIds: (ids) => set((s) => { s.snapshotDirtyDocIds = ids; }),

    setElemSyncItem: (itemId) => set((s) => { s.elemSyncItemId = itemId; }),

    setElemList: (itemId, list) => set((s) => {
      s.elemListByItem[itemId] = list;
      if (!s.elemSelByItem[itemId]) {
        const sel: Record<string, boolean> = {};
        for (const e of list) {
          // Default selection = the header / axis-label set.
          sel[e.id] = ["colhdr", "rowhdr", "collbl", "rowlbl"].includes(e.id.split(":")[0]);
        }
        s.elemSelByItem[itemId] = sel;
      }
    }),

    toggleElemSel: (itemId, elemId) => set((s) => {
      const cur = s.elemSelByItem[itemId] || {};
      cur[elemId] = !cur[elemId];
      s.elemSelByItem[itemId] = cur;
    }),

    setAllElemSel: (itemId, on) => set((s) => {
      const list = s.elemListByItem[itemId] || [];
      const sel: Record<string, boolean> = {};
      for (const e of list) sel[e.id] = on;
      s.elemSelByItem[itemId] = sel;
    }),

    applyAllElemSel: (mode) => set((s) => {
      for (const id of Object.keys(s.elemListByItem)) {
        const list = s.elemListByItem[id] || [];
        const sel: Record<string, boolean> = {};
        for (const e of list) {
          sel[e.id] = mode === "defaults"
            ? ["colhdr", "rowhdr", "collbl", "rowlbl"].includes(e.id.split(":")[0])
            : false;
        }
        s.elemSelByItem[id] = sel;
      }
    }),

    setHoveredElem: (h) => set((s) => { s.hoveredElem = h; }),

    setElemOverride: (itemId, elemId, ov) => {
      set((s) => {
        if (!s.elemOverridesByItem[itemId]) s.elemOverridesByItem[itemId] = {};
        if (ov === null) {
          delete s.elemOverridesByItem[itemId][elemId];
        } else {
          s.elemOverridesByItem[itemId][elemId] = ov;
        }
      });
      persist(get());
    },
  })),
);
