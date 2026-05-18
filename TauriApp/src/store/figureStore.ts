/* ──────────────────────────────────────────────────────────
   Zustand store — single source of truth for the app state.
   Uses immer middleware for immutable-friendly mutations.
   All API calls go through the api client.
   ────────────────────────────────────────────────────────── */

import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import type {
  FigureConfig,
  AxisLabel,
  HeaderLevel,
  HeaderGroup,
  PanelInfo,
  ImageGroup,
} from "../api/types";
import { api, checkHealth, lastHealthError } from "../api/client";

// ── Loaded image metadata kept client-side ───────────────

export interface LoadedImage {
  name: string;            // filename used as key in backend
  thumbnailB64: string;    // base64 PNG data from backend
}

// ── Store shape ──────────────────────────────────────────

interface FigureState {
  config: FigureConfig | null;
  loadedImages: Record<string, LoadedImage>;
  panelThumbnails: Record<string, string>;   // "r-c" → processed base64 PNG
  fonts: string[];
  previewImageB64: string | null;
  previewLoading: boolean;
  apiError: string | null;
  configDirty: boolean;     // true when local changes haven't been previewed
  /** Absolute path of the .mpf the user last saved or loaded. Cleared
   *  by "New". Used by the collage's Add-to-Collage / Multi-Panel
   *  Builder round-trip to remember which project a figure-kind item
   *  came from — collage items now store the user's chosen path
   *  rather than an auto-generated stash, so the user is in control
   *  of where their work lives on disk. */
  currentProjectPath: string | null;
  drawerPanels: PanelInfo[];
  drawerThumbnails: Record<number, string>;  // drawerIdx → processed base64 PNG
  imageGroups: ImageGroup[];
  pendingUploads: string[];  // filenames currently being uploaded/processed

  // ── actions ────────────────────────────────────────────
  fetchConfig: () => Promise<void>;
  fetchFonts: () => Promise<void>;
  fetchImages: () => Promise<void>;

  setConfig: (cfg: FigureConfig) => void;
  updateGridSize: (rows: number, cols: number) => Promise<void>;
  setSpacing: (spacing: number) => void;
  setBackground: (bg: string) => void;
  setOutputFormat: (fmt: string) => void;

  setPanelImage: (row: number, col: number, imageName: string) => void;
  updatePanel: (row: number, col: number, patch: Partial<PanelInfo>) => void;
  refreshPanelThumbnail: (row: number, col: number) => Promise<void>;

  // `opts.skipSync` updates the store config only — no debounced
  // backend patch / preview refresh. Used while the user is actively
  // typing in a header/label field; the deferred sync is flushed by
  // `flushHeaderEdits()` once the field loses focus, so the preview
  // doesn't churn on every keystroke.
  updateColumnLabel: (colIdx: number, text: string, opts?: { skipSync?: boolean }) => void;
  updateRowLabel: (rowIdx: number, text: string, opts?: { skipSync?: boolean }) => void;
  // Patch all four header/label arrays to the backend and request a
  // single preview refresh. Called on blur of a header/label editor.
  flushHeaderEdits: () => void;

  addColumnHeaderLevel: () => void;
  removeColumnHeaderLevel: (levelIdx: number) => void;
  updateHeaderGroupText: (
    axis: "col" | "row",
    level: number,
    groupIdx: number,
    text: string,
    opts?: { skipSync?: boolean },
  ) => void;
  addRowHeaderLevel: () => void;
  removeRowHeaderLevel: (levelIdx: number) => void;

  // Spanning header manipulation
  extendHeaderGroup: (axis: "col" | "row", level: number, groupIdx: number, direction: "left" | "right" | "up" | "down") => void;
  removeHeaderGroup: (axis: "col" | "row", level: number, groupIdx: number) => void;
  splitHeaderGroup: (axis: "col" | "row", level: number, groupIdx: number) => void;
  addHeaderGroupAt: (axis: "col" | "row", level: number, index: number) => void;
  createHeaderGroupAt: (axis: "col" | "row", levelIdx: number, cellIdx: number) => void;
  resizeHeaderGroup: (axis: "col" | "row", levelIdx: number, groupIdx: number, newIndices: number[]) => void;

  // Header/label formatting
  updateHeaderGroupFormatting: (
    axis: "col" | "row",
    level: number,
    groupIdx: number,
    patch: Partial<Pick<HeaderGroup, "font_size" | "font_name" | "font_style" | "default_color" | "position" | "styled_segments">>,
    opts?: { skipSync?: boolean },
  ) => void;
  updateLabelFormatting: (
    axis: "col" | "row",
    index: number,
    patch: Partial<Pick<import("../api/types").AxisLabel, "font_size" | "font_name" | "font_style" | "default_color" | "position" | "distance" | "rotation" | "styled_segments">>,
    opts?: { skipSync?: boolean },
  ) => void;

  // Header level swapping
  swapColumnHeaderLevels: (level1: number, level2: number) => void;
  swapRowHeaderLevels: (level1: number, level2: number) => void;

  // Panel drag-and-drop
  swapPanels: (r1: number, c1: number, r2: number, c2: number) => void;
  movePanelToDrawer: (r: number, c: number, drawerIdx: number) => void;
  movePanelFromDrawer: (drawerIdx: number, r: number, c: number) => void;

  /** Upload File objects (e.g. drag-drop in browser dev mode, or
   *  analysis plots being pushed onto the timeline) through the
   *  backend. Returns the canonical names the backend assigned so
   *  callers can map their inputs to loadedImages entries. */
  uploadImages: (files: File[]) => Promise<string[]>;
  uploadImagesFromPaths: (filePaths: string[]) => Promise<string[]>;
  removeImage: (name: string) => Promise<void>;

  // Image group management
  createImageGroup: (name: string) => void;
  renameImageGroup: (groupId: string, name: string) => void;
  deleteImageGroup: (groupId: string) => void;
  moveImageToGroup: (imageName: string, groupId: string) => void;
  moveImageToTimeline: (imageName: string) => void;

  requestPreview: () => void;
  syncToBackend: () => Promise<void>;

  saveProject: (path: string) => Promise<void>;
  loadProject: (path: string) => Promise<void>;
  saveFigure: (path: string, format?: string, background?: string, dpi?: number) => Promise<void>;

  // Check if reducing grid would lose images
  checkGridResizeConflict: (newRows: number, newCols: number) => string[];

  /** Re-letter every panel's auto-numbering label to match its
   *  current (row, col) position. Triggered automatically on grid
   *  resize / swap / drawer move; also exposed as a manual button
   *  so users can fix legacy projects whose letters got out of sync
   *  before auto-renumbering existed. */
  renumberPanels: () => void;
}

// ── Helpers ──────────────────────────────────────────────

/** Compute the auto-letter label for a panel at (row, col) in a
 *  `cols`-wide grid: a, b, …, z, aa, ab, …  Mirrors the formula
 *  used by setPanelImage when an auto-label is first created. */
function _autoLetterFor(row: number, col: number, cols: number): string {
  const idx = row * cols + col;
  return idx < 26
    ? String.fromCharCode(97 + idx)
    : `${String.fromCharCode(97 + Math.floor(idx / 26) - 1)}${String.fromCharCode(97 + (idx % 26))}`;
}

/** Re-letter the auto-numbering label on a single panel, in place.
 *  Returns true if anything changed. Only acts on labels that look
 *  auto-managed:
 *    - linked_to_header is true (the marker setPanelImage stamps), and
 *    - current text matches /^[a-z]{1,2}$/ (so manually-typed strings
 *      survive even if linked_to_header was left on).
 *  Clears styled_segments since they referenced characters that no
 *  longer exist in the new text. */
function _relabelAutoLetter(panel: { labels?: { text: string; linked_to_header?: boolean; styled_segments?: unknown[] }[] }, row: number, col: number, cols: number): boolean {
  const lbl = panel.labels?.[0];
  if (!lbl || !lbl.linked_to_header) return false;
  if (!/^[a-z]{1,2}$/.test(lbl.text || "")) return false;
  const wanted = _autoLetterFor(row, col, cols);
  if (lbl.text === wanted) return false;
  lbl.text = wanted;
  if (lbl.styled_segments && lbl.styled_segments.length) lbl.styled_segments = [];
  return true;
}

/** True when SOME other panel in `cfg` has an Adjacent-Panel zoom
 *  inset pointing at (row, col). Walks both the legacy singular
 *  `zoom_inset` and the new `zoom_insets[]` array so panels with
 *  multiple adjacent insets all have their targets recognised. */
function _isAdjacentZoomTarget(
  cfg: { rows: number; cols: number; panels: { add_zoom_inset?: boolean; zoom_inset?: { inset_type?: string; side?: string } | null; zoom_insets?: { inset_type?: string; side?: string }[] }[][] },
  row: number, col: number,
): boolean {
  for (let r = 0; r < cfg.rows; r++) {
    for (let c = 0; c < cfg.cols; c++) {
      const p = cfg.panels[r]?.[c];
      if (!p?.add_zoom_inset) continue;
      const insets = (p.zoom_insets && p.zoom_insets.length > 0)
        ? p.zoom_insets
        : (p.zoom_inset ? [p.zoom_inset] : []);
      for (const zi of insets) {
        if (zi.inset_type !== "Adjacent Panel") continue;
        const side = zi.side || "Right";
        let tr = r, tc = c;
        if (side === "Top") tr--; else if (side === "Bottom") tr++;
        else if (side === "Left") tc--; else if (side === "Right") tc++;
        if (tr === row && tc === col) return true;
      }
    }
  }
  return false;
}

/** Adjacent-zoom target panels never go through setPanelImage (the
 *  user doesn't drag an image onto them — the renderer paints them
 *  with the zoomed source). They were therefore missing the default
 *  auto-letter label that every other panel gets on assignment.
 *
 *  This helper creates that label on a zoom-target panel the first
 *  time it's seen without one. Returns true when it inserted a label.
 *  Subsequent runs of `_relabelAutoLetter` then keep the letter in
 *  sync with the panel's (row, col) just like for image-bearing
 *  panels. */
function _ensureZoomTargetLabel(
  panel: { labels?: { text: string; linked_to_header?: boolean; styled_segments?: unknown[]; [k: string]: unknown }[] },
  row: number, col: number, cols: number,
): boolean {
  if (panel.labels && panel.labels.length > 0) return false;
  const letter = _autoLetterFor(row, col, cols);
  if (!panel.labels) panel.labels = [];
  panel.labels.push({
    text: letter,
    font_path: null,
    font_name: "arial.ttf",
    font_size: 20,
    font_style: [],
    color: "#FFFFFF",
    position_x: 3,
    position_y: 3,
    rotation: 0,
    default_color: "#FFFFFF",
    position_preset: "Top-Left",
    edge_distance: 3,
    linked_to_header: true,
    styled_segments: [],
  } as unknown as { text: string; linked_to_header?: boolean; styled_segments?: unknown[] });
  return true;
}

/** Re-letter every panel in the grid AND ensure both image-bearing
 *  panels and zoom-target panels have their default auto-letter
 *  label. Used when the grid size changes (cols altered → all
 *  letter indices shift), when many panels rearrange at once, or
 *  when a new adjacent zoom inset is committed (so the newly-promoted
 *  target cell gets its letter immediately). Promoting image-bearing
 *  panels here is a safety net for cases where setPanelImage's
 *  auto-label didn't run (loaded projects, direct config edits, …).
 */
function _renumberAllAutoLetters(cfg: { rows: number; cols: number; panels: { image_name?: string; labels?: { text: string; linked_to_header?: boolean; styled_segments?: unknown[] }[]; add_zoom_inset?: boolean; zoom_inset?: { inset_type?: string; side?: string } | null; zoom_insets?: { inset_type?: string; side?: string }[] }[][] } | null) {
  if (!cfg) return;
  for (let r = 0; r < cfg.rows; r++) {
    for (let c = 0; c < cfg.cols; c++) {
      const p = cfg.panels[r]?.[c];
      if (!p) continue;
      const isImagePanel = !!(p.image_name && p.image_name.trim());
      const isZoomTarget = _isAdjacentZoomTarget(cfg, r, c);
      // Promote either image-bearing or zoom-target cells to have a
      // default auto-label first (so _relabelAutoLetter below has
      // something to relabel). Empty cells stay label-less.
      if (isImagePanel || isZoomTarget) {
        _ensureZoomTargetLabel(p, r, c, cfg.cols);
      }
      _relabelAutoLetter(p, r, c, cfg.cols);
    }
  }
}

function defaultAxisLabel(text: string, position: string, rotation = 0): AxisLabel {
  return {
    text,
    font_size: 12,
    font_name: "arial.ttf",
    font_path: null,
    font_style: [],
    default_color: "#000000",
    distance: 0.025,
    position,
    rotation,
    styled_segments: [],
    visible: true,
  };
}

function defaultPanel(): PanelInfo {
  return {
    image_name: "",
    crop_image: true,
    aspect_ratio_str: "",
    crop_offset_x: 0,
    crop_offset_y: 0,
    crop: null,
    final_resize: false,
    final_width: 400,
    final_height: 300,
    brightness: 1.0,
    contrast: 1.0,
    hue: 0.0,
    labels: [],
    scale_bar: null,
    add_scale_bar: false,
    symbols: [],
    lines: [],
    areas: [],
    zoom_inset: null,
    add_zoom_inset: false,
    zoom_insets: [],
    rotation: 0,
    flip_horizontal: false,
    flip_vertical: false,
    saturation: 1.0,
    gamma: 1.0,
    color_temperature: 0,
    tint: 0,
    sharpen: 0,
    blur: 0,
    denoise: 0,
    exposure: 0,
    vibrance: 0,
    highlights: 0,
    shadows: 0,
    midtones: 0,
    input_black_r: 0, input_white_r: 255,
    input_black_g: 0, input_white_g: 255,
    input_black_b: 0, input_white_b: 255,
    invert: false,
    grayscale: false,
    pseudocolor: "",
  };
}

function defaultHeaderGroup(colsOrRows: number[]): HeaderGroup {
  return {
    text: "Header",
    columns_or_rows: colsOrRows,
    font_size: 12,
    font_name: "arial.ttf",
    font_path: null,
    font_style: [],
    default_color: "#000000",
    distance: 0.01,
    position: "Top",
    rotation: 0,
    line_color: "#000000",
    line_width: 1.0,
    line_style: "solid",
    line_length: 1.0,
    end_caps: false,
    styled_segments: [],
  };
}

function buildDefaultConfig(rows: number, cols: number): FigureConfig {
  const panels: PanelInfo[][] = [];
  for (let r = 0; r < rows; r++) {
    const row: PanelInfo[] = [];
    for (let c = 0; c < cols; c++) {
      row.push(defaultPanel());
    }
    panels.push(row);
  }
  return {
    rows,
    cols,
    spacing: 0.02,
    output_format: "TIFF",
    background: "White",
    panels,
    column_labels: Array.from({ length: cols }, (_, i) =>
      defaultAxisLabel(`Column ${i + 1}`, "Top"),
    ),
    row_labels: Array.from({ length: rows }, (_, i) =>
      defaultAxisLabel(`Row ${i + 1}`, "Left", 90),
    ),
    column_headers: [],
    row_headers: [],
    resolution_entries: {},
    dpi: 300,
  };
}

// ── Debounce helper ──────────────────────────────────────

let previewTimer: ReturnType<typeof setTimeout> | null = null;
let syncTimer: ReturnType<typeof setTimeout> | null = null;
// True when a header/label edit (text OR formatting/colour/style) has been
// applied to the store with `skipSync` and not yet pushed to the backend.
// flushHeaderEdits() consults this so it can no-op when nothing is pending
// (and so the closeToolbar + onBlur double-fire collapses to one flush).
let headerEditsPending = false;
// Monotonic sequence for preview requests so an older in-flight GET
// response can't overwrite a newer one (race condition seen as a
// "ghost" of the previous header state appearing in the preview).
let previewSeq = 0;

// ── Store ────────────────────────────────────────────────

export const useFigureStore = create<FigureState>()(
  immer((set, get) => ({
    config: null,
    loadedImages: {},
    panelThumbnails: {},
    fonts: [],
    previewImageB64: null,
    previewLoading: false,
    apiError: null,
    configDirty: false,
    currentProjectPath: null,
    drawerPanels: [],
    drawerThumbnails: {},
    imageGroups: [],
    pendingUploads: [],

    // ── Fetch initial state from backend ──────────────────

    fetchConfig: async () => {
      // Give sidecar time to start (PyInstaller --onefile extracts on first run)
      await new Promise(r => setTimeout(r, 2000));
      // Wait for sidecar to be ready (retry up to 30 times with 1s delay)
      let connected = false;
      for (let attempt = 0; attempt < 30; attempt++) {
        if (await checkHealth()) { connected = true; break; }
        await new Promise(r => setTimeout(r, 1000));
      }
      if (!connected) {
        console.error("API server not reachable after 30 attempts");
        // Gather diagnostic info
        let diagMsg = "";
        // Last health check error
        if (lastHealthError) diagMsg += ` Last error: ${lastHealthError}`;
        // Try to get sidecar error from Tauri
        try {
          const { invoke } = await import("@tauri-apps/api/core");
          const err = await invoke("get_sidecar_error");
          if (err) diagMsg += ` Sidecar stderr: ${String(err).substring(0, 300)}`;
        } catch { /* not in Tauri context */ }
        set((s) => {
          s.config = buildDefaultConfig(2, 2);
          s.apiError = `Cannot connect to backend server. Image loading and preview will not work.${diagMsg}`;
        });
        return;
      }
      try {
        const cfg = await api.getConfig();
        set((s) => {
          s.config = cfg;
          s.apiError = null;
        });
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        console.warn("Failed to fetch config, using default", err);
        set((s) => {
          s.config = buildDefaultConfig(2, 2);
          s.apiError = `Connected to backend but failed to load configuration. Error: ${errMsg}`;
        });
      }
    },

    fetchFonts: async () => {
      try {
        const { fonts } = await api.listFonts();
        // fonts is Record<string, string> (name -> path), extract names
        const fontNames = typeof fonts === 'object' && !Array.isArray(fonts)
          ? Object.keys(fonts)
          : (fonts as unknown as string[]);
        set((s) => {
          s.fonts = fontNames;
        });
      } catch {
        // fonts will stay empty
      }
    },

    fetchImages: async () => {
      try {
        const { names } = await api.listImages();
        const images: Record<string, LoadedImage> = {};
        for (const name of names) {
          const { thumbnail } = await api.getImageThumbnail(name);
          images[name] = { name, thumbnailB64: thumbnail };
        }
        set((s) => {
          s.loadedImages = images;
        });
      } catch {
        // no images loaded yet
      }
    },

    // ── Config mutations ──────────────────────────────────

    setConfig: (cfg) => {
      set((s) => {
        s.config = cfg;
        s.configDirty = true;
      });
      get().syncToBackend();
      get().requestPreview();
    },

    updateGridSize: async (rows, cols) => {
      try {
        // Flush the entire local cfg first. Without this, any in-flight
        // patchPanel from a recent edit (e.g., the user just adjusted a
        // crop and immediately changed grid size) can race patchGrid;
        // patchGrid's response would carry stale panel data and overwrite
        // the local crop. syncToBackend PUTs the full cfg, so when
        // patchGrid runs the backend already has the latest panels +
        // labels + headers.
        await get().syncToBackend();
        const cfg = await api.patchGrid(rows, cols, get().config?.spacing ?? 0.02);
        // After the grid changes, every panel's auto-letter index may
        // shift (cols changed → all letters re-derive). Renumber locally
        // and push the corrected cfg back so the backend stores the
        // up-to-date letters.
        _renumberAllAutoLetters(cfg as unknown as Parameters<typeof _renumberAllAutoLetters>[0]);
        set((s) => {
          s.config = cfg;
          s.panelThumbnails = {};
          s.configDirty = true;
        });
        // Push renumbered cfg to backend (best-effort).
        get().syncToBackend().catch(console.error);
        get().requestPreview();
      } catch (err) {
        console.error("Failed to update grid", err);
      }
    },

    setSpacing: (spacing) => {
      set((s) => {
        if (s.config) {
          s.config.spacing = spacing;
          s.configDirty = true;
        }
      });
      if (syncTimer) clearTimeout(syncTimer);
      syncTimer = setTimeout(() => {
        const cfg = get().config;
        if (cfg) {
          api.patchGrid(cfg.rows, cfg.cols, cfg.spacing).catch(console.error);
        }
        get().requestPreview();
      }, 300);
    },

    setBackground: (bg) => {
      set((s) => {
        if (s.config) {
          s.config.background = bg;
          s.configDirty = true;
        }
      });
      api.patchBackground(bg).catch(console.error);
      get().requestPreview();
    },

    setOutputFormat: (fmt) => {
      set((s) => {
        if (s.config) {
          s.config.output_format = fmt;
        }
      });
    },

    setPanelImage: (row, col, imageName) => {
      set((s) => {
        if (s.config && s.config.panels[row] && s.config.panels[row][col]) {
          const panel = s.config.panels[row][col];
          panel.image_name = imageName;
          // Auto-create panel letter label if no labels exist and image is being assigned
          if (imageName && panel.labels.length === 0) {
            const cols = s.config.cols;
            const idx = row * cols + col;
            const letter = idx < 26 ? String.fromCharCode(97 + idx) : `${String.fromCharCode(97 + Math.floor(idx / 26) - 1)}${String.fromCharCode(97 + (idx % 26))}`;
            panel.labels.push({
              text: letter,
              font_path: null,
              font_name: "arial.ttf",
              font_size: 20,
              font_style: [],
              color: "#FFFFFF",
              position_x: 3,
              position_y: 3,
              rotation: 0,
              default_color: "#FFFFFF",
              position_preset: "Top-Left",
              edge_distance: 3,
              linked_to_header: true,
              styled_segments: [],
            });
          }
          // Clear processed thumbnail — will show raw until refreshed
          delete s.panelThumbnails[`${row}-${col}`];
          s.configDirty = true;
        }
      });
      const panel = get().config?.panels[row]?.[col];
      if (panel) {
        api.patchPanel(row, col, panel as unknown as Record<string, unknown>).catch(console.error);
      }
      get().requestPreview();
    },

    updatePanel: (row, col, patch) => {
      // Detect whether THIS patch touches adjacent zoom-inset state.
      // If it does we have to walk the grid afterwards to promote any
      // newly-created target panels with default auto-letter labels
      // (those cells never go through setPanelImage so they'd
      // otherwise stay label-less). The patch can carry inset state
      // via `add_zoom_inset`, `zoom_inset`, or the newer `zoom_insets`
      // array — any of those gates the work.
      const touchesZoom =
        Object.prototype.hasOwnProperty.call(patch, "add_zoom_inset") ||
        Object.prototype.hasOwnProperty.call(patch, "zoom_inset") ||
        Object.prototype.hasOwnProperty.call(patch, "zoom_insets");
      // Track which OTHER cells got their labels mutated so we know
      // which to push back to the backend below.
      const touchedAdjacent: Array<{ r: number; c: number }> = [];
      // Track which other cells had their scale bar values cascaded
      // (live inheritance through adjacent-zoom chains). They need
      // their backend state pushed AND their PanelGrid thumbnails
      // refreshed.
      const cascadedScaleBar: Array<{ r: number; c: number }> = [];

      set((s) => {
        if (s.config && s.config.panels[row] && s.config.panels[row][col]) {
          Object.assign(s.config.panels[row][col], patch);
          s.configDirty = true;
        }
        if (touchesZoom && s.config) {
          // Snapshot each panel's labels[0].text before the relabel so
          // we know which adjacent-zoom targets actually changed.
          const before: Record<string, string | undefined> = {};
          for (let r2 = 0; r2 < s.config.rows; r2++) {
            for (let c2 = 0; c2 < s.config.cols; c2++) {
              before[`${r2},${c2}`] = s.config.panels[r2]?.[c2]?.labels?.[0]?.text;
            }
          }
          _renumberAllAutoLetters(s.config as unknown as Parameters<typeof _renumberAllAutoLetters>[0]);
          for (let r2 = 0; r2 < s.config.rows; r2++) {
            for (let c2 = 0; c2 < s.config.cols; c2++) {
              if (r2 === row && c2 === col) continue; // source already pushed below
              const after = s.config.panels[r2]?.[c2]?.labels?.[0]?.text;
              if (after !== before[`${r2},${c2}`]) {
                touchedAdjacent.push({ r: r2, c: c2 });
              }
            }
          }
        }

        // ── Live scale-bar cascade through adjacent-zoom chains ──
        // Whenever any panel's scale_bar OR outgoing adjacent inset
        // geometry changes, re-derive every downstream target's bar
        // values using the selection-size ratio formula. Walks the
        // chain depth-first so secondary/tertiary insets update too,
        // not just the immediate primary target.
        if (s.config) {
          const cfg = s.config;
          // Source-coord width for an inset's source panel.
          const srcCoordW = (sp: { crop_image?: boolean; crop?: number[]; image_name?: string }): number => {
            if (sp.crop_image && sp.crop && sp.crop.length === 4) {
              return Math.max(1, sp.crop[2] - sp.crop[0]);
            }
            if (!sp.image_name) return 1000;  // zoom-target source
            return 1000;  // uncropped image-bearing source (rare)
          };
          type PInfo = { add_zoom_inset?: boolean; zoom_inset?: unknown; zoom_insets?: unknown[]; add_scale_bar?: boolean; scale_bar?: { micron_per_pixel: number; bar_length_microns: number } | null; image_name?: string };
          const cascadeFrom = (sr: number, sc: number, visited: Set<string>) => {
            const key = `${sr},${sc}`;
            if (visited.has(key)) return;  // cycle guard
            visited.add(key);
            const sp = cfg.panels[sr]?.[sc] as PInfo | undefined;
            if (!sp || !sp.add_zoom_inset || !sp.add_scale_bar || !sp.scale_bar) return;
            const arr = (sp.zoom_insets && (sp.zoom_insets as unknown[]).length > 0)
              ? (sp.zoom_insets as Array<Record<string, unknown>>)
              : (sp.zoom_inset ? [sp.zoom_inset as Record<string, unknown>] : []);
            for (const zi of arr) {
              if (!zi || zi.inset_type !== "Adjacent Panel") continue;
              const side = (zi.side as string | undefined) || "Right";
              let tr = sr, tc = sc;
              if (side === "Top") tr--; else if (side === "Bottom") tr++;
              else if (side === "Left") tc--; else if (side === "Right") tc++;
              if (tr < 0 || tr >= cfg.rows || tc < 0 || tc >= cfg.cols) continue;
              const tp = cfg.panels[tr]?.[tc] as PInfo | undefined;
              if (!tp || !tp.add_scale_bar) continue;
              const w = Math.max(1, Number(zi.width) || 1);
              const ratio = w / srcCoordW(sp);
              const newMpp = sp.scale_bar.micron_per_pixel * ratio;
              const newBar = sp.scale_bar.bar_length_microns * ratio;
              const curMpp = tp.scale_bar?.micron_per_pixel ?? 0;
              const curBar = tp.scale_bar?.bar_length_microns ?? 0;
              if (Math.abs(curMpp - newMpp) > 1e-9 || Math.abs(curBar - newBar) > 1e-6) {
                tp.scale_bar = {
                  ...(tp.scale_bar ?? sp.scale_bar),
                  micron_per_pixel: newMpp,
                  bar_length_microns: newBar,
                };
                cascadedScaleBar.push({ r: tr, c: tc });
              }
              // Recurse into the target — its own outgoing insets
              // (if any) need to refresh too.
              cascadeFrom(tr, tc, visited);
            }
          };
          // Start cascade from THIS panel — it's the one whose state
          // just changed. Recursion fans out through the chain.
          cascadeFrom(row, col, new Set());
        }
      });

      const panel = get().config?.panels[row]?.[col];
      if (panel) {
        api.patchPanel(row, col, panel as unknown as Record<string, unknown>)
          .then(() => get().refreshPanelThumbnail(row, col))
          .catch(console.error);
      }
      // Push label fixups for any neighbouring panels the relabel pass
      // touched (newly-promoted zoom targets). Fire in parallel; their
      // thumbnails get refreshed when the patch resolves.
      for (const { r: rr, c: cc } of touchedAdjacent) {
        const p2 = get().config?.panels[rr]?.[cc];
        if (p2) {
          api.patchPanel(rr, cc, p2 as unknown as Record<string, unknown>)
            .then(() => get().refreshPanelThumbnail(rr, cc))
            .catch(console.error);
        }
      }
      // Push the cascaded scale-bar updates and refresh their
      // thumbnails so the panel planner shows live updates.
      for (const { r: rr, c: cc } of cascadedScaleBar) {
        const p2 = get().config?.panels[rr]?.[cc];
        if (p2) {
          api.patchPanel(rr, cc, p2 as unknown as Record<string, unknown>)
            .then(() => get().refreshPanelThumbnail(rr, cc))
            .catch(console.error);
        }
      }
      get().requestPreview();
    },

    refreshPanelThumbnail: async (row, col) => {
      try {
        const resp = await api.getPanelPreview(row, col);
        if (resp.image) {
          set((s) => {
            s.panelThumbnails[`${row}-${col}`] = resp.image;
          });
        }
      } catch {
        // ignore — cell will show raw thumbnail
      }
    },

    updateColumnLabel: (colIdx, text, opts) => {
      console.log("[mpf] updateColumnLabel", { colIdx, text: JSON.stringify(text), skipSync: !!opts?.skipSync });
      set((s) => {
        if (s.config && colIdx < s.config.column_labels.length) {
          const lbl = s.config.column_labels[colIdx];
          lbl.text = text;
          // Mirror the header-text clearing behaviour: if the plain text
          // no longer matches the concatenation of styled_segments, drop
          // the segments so the backend renders `text` (possibly empty)
          // instead of the stale styled fragments. This was the root
          // cause of the "ghost header / label returns after clear"
          // bug: backend's _get_segments prefers styled_segments over
          // the plain text, so empty text + non-empty segments =
          // backend rendered the old styled string.
          if (lbl.styled_segments && lbl.styled_segments.length > 0) {
            const concat = lbl.styled_segments.map((seg: { text: string }) => seg.text).join("");
            if (text === "" || concat !== text) {
              lbl.styled_segments = [];
            }
          }
          s.configDirty = true;
        }
      });
      if (opts?.skipSync) { headerEditsPending = true; return; }  // typing — defer to flushHeaderEdits() on blur
      if (syncTimer) clearTimeout(syncTimer);
      syncTimer = setTimeout(async () => {
        const cfg = get().config;
        if (!cfg) return;
        try {
          await api.patchColumnLabels(cfg.column_labels);
        } catch (e) {
          console.error(e);
        }
        get().requestPreview();
      }, 200);
    },

    updateRowLabel: (rowIdx, text, opts) => {
      console.log("[mpf] updateRowLabel", { rowIdx, text: JSON.stringify(text), skipSync: !!opts?.skipSync });
      set((s) => {
        if (s.config && rowIdx < s.config.row_labels.length) {
          const lbl = s.config.row_labels[rowIdx];
          lbl.text = text;
          if (lbl.styled_segments && lbl.styled_segments.length > 0) {
            const concat = lbl.styled_segments.map((seg: { text: string }) => seg.text).join("");
            if (text === "" || concat !== text) {
              lbl.styled_segments = [];
            }
          }
          s.configDirty = true;
        }
      });
      if (opts?.skipSync) { headerEditsPending = true; return; }  // typing — defer to flushHeaderEdits() on blur
      if (syncTimer) clearTimeout(syncTimer);
      syncTimer = setTimeout(async () => {
        const cfg = get().config;
        if (!cfg) return;
        try {
          await api.patchRowLabels(cfg.row_labels);
        } catch (e) {
          console.error(e);
        }
        get().requestPreview();
      }, 200);
    },

    addColumnHeaderLevel: () => {
      set((s) => {
        if (!s.config) return;
        // Create an empty lane with NO headers — outermost (furthest from panels)
        const newLevel: HeaderLevel = { headers: [] };
        s.config.column_headers.unshift(newLevel);
        s.configDirty = true;
      });
      const cfg = get().config;
      if (cfg) {
        api.patchColumnHeaders(cfg.column_headers).catch(console.error);
        get().requestPreview();
      }
    },

    removeColumnHeaderLevel: (levelIdx) => {
      set((s) => {
        if (!s.config) return;
        s.config.column_headers.splice(levelIdx, 1);
        s.configDirty = true;
      });
      const cfg = get().config;
      if (cfg) {
        api.patchColumnHeaders(cfg.column_headers).catch(console.error);
        get().requestPreview();
      }
    },

    updateHeaderGroupText: (axis, level, groupIdx, text, opts) => {
      console.log("[mpf] updateHeaderGroupText", { axis, level, groupIdx, text: JSON.stringify(text), len: text.length, skipSync: !!opts?.skipSync });
      set((s) => {
        if (!s.config) return;
        const headers = axis === "col" ? s.config.column_headers : s.config.row_headers;
        if (level < headers.length && groupIdx < headers[level].headers.length) {
          const group = headers[level].headers[groupIdx];
          group.text = text;
          // Clear per-character styled segments whenever the plain text no
          // longer matches them. Covers three cases that previously let
          // stale segments restore the OLD header visually after a
          // delete-then-retype sequence:
          //   (a) text went empty
          //   (b) text is a strict prefix/substring of the old concat
          //   (c) text is a different string entirely
          if (group.styled_segments && group.styled_segments.length > 0) {
            const concat = group.styled_segments.map((seg: any) => seg.text).join("");
            if (text === "" || concat !== text) {
              group.styled_segments = [];
            }
          }
          s.configDirty = true;
        }
      });
      // While the user is actively typing (skipSync) we update the
      // store only — no backend patch, no preview refresh. The
      // deferred sync is flushed by flushHeaderEdits() on blur.
      if (opts?.skipSync) { headerEditsPending = true; return; }
      if (syncTimer) clearTimeout(syncTimer);
      syncTimer = setTimeout(async () => {
        const cfg = get().config;
        if (!cfg) return;
        try {
          const patchFn = axis === "col" ? api.patchColumnHeaders : api.patchRowHeaders;
          const headers = axis === "col" ? cfg.column_headers : cfg.row_headers;
          // Summarise what we're about to send so the user can see on DevTools
          // whether the cleared-text / empty-segments state actually reached
          // the backend.
          try {
            const summary = headers.flatMap((lvl: { headers: { text: string; styled_segments?: unknown[] }[] }) =>
              lvl.headers.map((h) => ({ text: h.text, segs: h.styled_segments?.length ?? 0 })),
            );
            console.log(`[mpf] sync patch ${axis}-headers`, summary);
          } catch { /* ignore */ }
          await patchFn.call(api, headers);
          console.log(`[mpf] sync patch ${axis}-headers DONE`);
        } catch (e) {
          console.error("[mpf] sync patch failed", e);
        }
        get().requestPreview();
      }, 200);
    },

    // Flush any deferred header/label edits: patch all four
    // header/label arrays to the backend, then request ONE preview
    // refresh. Wired to the blur event of every header/label editor
    // so the preview updates once the user finishes typing, not on
    // each keystroke.
    flushHeaderEdits: () => {
      // No deferred header/label edit outstanding — nothing to push, and
      // skipping here means a focus-in/focus-out with no edits (or the
      // closeToolbar + editor-onBlur double-fire) doesn't spam the backend
      // with redundant patch+preview round-trips.
      if (!headerEditsPending) return;
      headerEditsPending = false;
      if (syncTimer) { clearTimeout(syncTimer); syncTimer = null; }
      const cfg = get().config;
      if (!cfg) return;
      (async () => {
        try {
          await Promise.all([
            api.patchColumnHeaders(cfg.column_headers),
            api.patchRowHeaders(cfg.row_headers),
            api.patchColumnLabels(cfg.column_labels),
            api.patchRowLabels(cfg.row_labels),
          ]);
          console.log("[mpf] flushHeaderEdits patched all header/label arrays");
        } catch (e) {
          console.error("[mpf] flushHeaderEdits failed", e);
        }
        get().requestPreview();
      })();
    },

    addRowHeaderLevel: () => {
      set((s) => {
        if (!s.config) return;
        // Create an empty lane with NO headers — outermost (furthest from panels)
        const newLevel: HeaderLevel = { headers: [] };
        s.config.row_headers.unshift(newLevel);
        s.configDirty = true;
      });
      const cfg = get().config;
      if (cfg) {
        api.patchRowHeaders(cfg.row_headers).catch(console.error);
        get().requestPreview();
      }
    },

    removeRowHeaderLevel: (levelIdx) => {
      set((s) => {
        if (!s.config) return;
        s.config.row_headers.splice(levelIdx, 1);
        s.configDirty = true;
      });
      const cfg = get().config;
      if (cfg) {
        api.patchRowHeaders(cfg.row_headers).catch(console.error);
        get().requestPreview();
      }
    },

    // ── Spanning header manipulation ──────────────────────

    extendHeaderGroup: (axis, level, groupIdx, direction) => {
      set((s) => {
        if (!s.config) return;
        const headers = axis === "col" ? s.config.column_headers : s.config.row_headers;
        if (level >= headers.length) return;
        const group = headers[level].headers[groupIdx];
        if (!group) return;
        const maxIdx = axis === "col" ? s.config.cols - 1 : s.config.rows - 1;
        const currentIdxs = [...group.columns_or_rows].sort((a, b) => a - b);
        const minIdx = currentIdxs[0];
        const maxCurrent = currentIdxs[currentIdxs.length - 1];

        // Find indices used by other groups at this level
        const usedByOthers = new Set<number>();
        headers[level].headers.forEach((g, gi) => {
          if (gi !== groupIdx) g.columns_or_rows.forEach((i) => usedByOthers.add(i));
        });

        if (direction === "right" || direction === "down") {
          const nextIdx = maxCurrent + 1;
          if (nextIdx <= maxIdx && !usedByOthers.has(nextIdx)) {
            group.columns_or_rows.push(nextIdx);
          }
        } else if (direction === "left" || direction === "up") {
          const prevIdx = minIdx - 1;
          if (prevIdx >= 0 && !usedByOthers.has(prevIdx)) {
            group.columns_or_rows.unshift(prevIdx);
          }
        }
        s.configDirty = true;
      });
      const cfg = get().config;
      if (cfg) {
        const patchFn = axis === "col" ? api.patchColumnHeaders : api.patchRowHeaders;
        const headers = axis === "col" ? cfg.column_headers : cfg.row_headers;
        patchFn.call(api, headers).catch(console.error);
        get().requestPreview();
      }
    },

    removeHeaderGroup: (axis, level, groupIdx) => {
      set((s) => {
        if (!s.config) return;
        const headers = axis === "col" ? s.config.column_headers : s.config.row_headers;
        if (level >= headers.length) return;
        headers[level].headers.splice(groupIdx, 1);
        // If no groups left, remove the level
        if (headers[level].headers.length === 0) {
          headers.splice(level, 1);
        }
        s.configDirty = true;
      });
      const cfg = get().config;
      if (cfg) {
        const patchFn = axis === "col" ? api.patchColumnHeaders : api.patchRowHeaders;
        const headers = axis === "col" ? cfg.column_headers : cfg.row_headers;
        patchFn.call(api, headers).catch(console.error);
        get().requestPreview();
      }
    },

    splitHeaderGroup: (axis, level, groupIdx) => {
      set((s) => {
        if (!s.config) return;
        const headers = axis === "col" ? s.config.column_headers : s.config.row_headers;
        if (level >= headers.length) return;
        const group = headers[level].headers[groupIdx];
        if (!group || group.columns_or_rows.length <= 1) return;
        // Replace with individual groups
        const newGroups = group.columns_or_rows.map((idx) => ({
          ...defaultHeaderGroup([idx]),
          text: group.text,
          position: group.position,
          rotation: group.rotation,
        }));
        headers[level].headers.splice(groupIdx, 1, ...newGroups);
        s.configDirty = true;
      });
      const cfg = get().config;
      if (cfg) {
        const patchFn = axis === "col" ? api.patchColumnHeaders : api.patchRowHeaders;
        const headers = axis === "col" ? cfg.column_headers : cfg.row_headers;
        patchFn.call(api, headers).catch(console.error);
        get().requestPreview();
      }
    },

    addHeaderGroupAt: (axis, level, index) => {
      set((s) => {
        if (!s.config) return;
        const headers = axis === "col" ? s.config.column_headers : s.config.row_headers;
        if (level >= headers.length) return;
        // Check the index isn't already covered
        const used = new Set(headers[level].headers.flatMap((g) => g.columns_or_rows));
        if (used.has(index)) return;
        const newGroup = axis === "col"
          ? defaultHeaderGroup([index])
          : { ...defaultHeaderGroup([index]), position: "Left", rotation: 90 };
        headers[level].headers.push(newGroup);
        s.configDirty = true;
      });
      const cfg = get().config;
      if (cfg) {
        const patchFn = axis === "col" ? api.patchColumnHeaders : api.patchRowHeaders;
        const headers = axis === "col" ? cfg.column_headers : cfg.row_headers;
        patchFn.call(api, headers).catch(console.error);
        get().requestPreview();
      }
    },

    // ── Create header at position (click-to-create) ──────

    createHeaderGroupAt: (axis, levelIdx, cellIdx) => {
      set((s) => {
        if (!s.config) return;
        const headers = axis === "col" ? s.config.column_headers : s.config.row_headers;
        if (levelIdx >= headers.length) return;
        // Check the index isn't already covered
        const used = new Set(headers[levelIdx].headers.flatMap((g) => g.columns_or_rows));
        if (used.has(cellIdx)) return;
        const newGroup = axis === "col"
          ? defaultHeaderGroup([cellIdx])
          : { ...defaultHeaderGroup([cellIdx]), position: "Left", rotation: 90 };
        newGroup.text = ""; // start empty so user types immediately
        headers[levelIdx].headers.push(newGroup);
        // Sort groups by their first index for consistent ordering
        headers[levelIdx].headers.sort((a, b) => {
          const aMin = Math.min(...a.columns_or_rows);
          const bMin = Math.min(...b.columns_or_rows);
          return aMin - bMin;
        });
        s.configDirty = true;
      });
      const cfg = get().config;
      if (cfg) {
        const patchFn = axis === "col" ? api.patchColumnHeaders : api.patchRowHeaders;
        const headers = axis === "col" ? cfg.column_headers : cfg.row_headers;
        patchFn.call(api, headers).catch(console.error);
        get().requestPreview();
      }
    },

    // ── Resize header group (drag-to-resize) ──────────────

    resizeHeaderGroup: (axis, levelIdx, groupIdx, newIndices) => {
      set((s) => {
        if (!s.config) return;
        const headers = axis === "col" ? s.config.column_headers : s.config.row_headers;
        if (levelIdx >= headers.length) return;
        const group = headers[levelIdx].headers[groupIdx];
        if (!group) return;
        // Validate: new indices must be contiguous and not overlap with other groups
        const sorted = [...newIndices].sort((a, b) => a - b);
        if (sorted.length === 0) return;
        // Check contiguity
        for (let i = 1; i < sorted.length; i++) {
          if (sorted[i] !== sorted[i - 1] + 1) return;
        }
        // Check no overlap with other groups
        const usedByOthers = new Set<number>();
        headers[levelIdx].headers.forEach((g, gi) => {
          if (gi !== groupIdx) g.columns_or_rows.forEach((idx) => usedByOthers.add(idx));
        });
        if (sorted.some((idx) => usedByOthers.has(idx))) return;
        // Check bounds
        const maxIdx = axis === "col" ? s.config.cols - 1 : s.config.rows - 1;
        if (sorted[0] < 0 || sorted[sorted.length - 1] > maxIdx) return;

        group.columns_or_rows = sorted;
        s.configDirty = true;
      });
      const cfg = get().config;
      if (cfg) {
        const patchFn = axis === "col" ? api.patchColumnHeaders : api.patchRowHeaders;
        const headers = axis === "col" ? cfg.column_headers : cfg.row_headers;
        patchFn.call(api, headers).catch(console.error);
        get().requestPreview();
      }
    },

    // ── Header level swapping ─────────────────────────────

    swapColumnHeaderLevels: (level1, level2) => {
      set((s) => {
        if (!s.config) return;
        const h = s.config.column_headers;
        if (level1 < 0 || level2 < 0 || level1 >= h.length || level2 >= h.length) return;
        const tmp = h[level1];
        h[level1] = h[level2];
        h[level2] = tmp;
        s.configDirty = true;
      });
      const cfg = get().config;
      if (cfg) {
        api.patchColumnHeaders(cfg.column_headers).catch(console.error);
        get().requestPreview();
      }
    },

    swapRowHeaderLevels: (level1, level2) => {
      set((s) => {
        if (!s.config) return;
        const h = s.config.row_headers;
        if (level1 < 0 || level2 < 0 || level1 >= h.length || level2 >= h.length) return;
        const tmp = h[level1];
        h[level1] = h[level2];
        h[level2] = tmp;
        s.configDirty = true;
      });
      const cfg = get().config;
      if (cfg) {
        api.patchRowHeaders(cfg.row_headers).catch(console.error);
        get().requestPreview();
      }
    },

    // ── Header / label formatting ─────────────────────────

    updateHeaderGroupFormatting: (axis, level, groupIdx, patch, opts) => {
      set((s) => {
        if (!s.config) return;
        const headers = axis === "col" ? s.config.column_headers : s.config.row_headers;
        if (level < headers.length && groupIdx < headers[level].headers.length) {
          Object.assign(headers[level].headers[groupIdx], patch);
          // 1.5: When updating font_size on level 0, sync to ALL level-0 headers across both axes
          if (patch.font_size !== undefined && level === 0) {
            const fontSize = patch.font_size;
            // Sync across all headers in column_headers level 0
            if (s.config.column_headers.length > 0) {
              for (const h of s.config.column_headers[0].headers) {
                h.font_size = fontSize;
              }
            }
            // Sync across all headers in row_headers level 0
            if (s.config.row_headers.length > 0) {
              for (const h of s.config.row_headers[0].headers) {
                h.font_size = fontSize;
              }
            }
          }
          s.configDirty = true;
        }
      });
      // Styling change applied via the floating toolbar (colour, font,
      // size, bold/italic, per-character segments). Like text typing,
      // defer the backend push + preview render to flushHeaderEdits()
      // when the header/label editing context loses focus — otherwise
      // every colour/style tweak triggers its own preview render (lag).
      if (opts?.skipSync) { headerEditsPending = true; return; }
      if (syncTimer) clearTimeout(syncTimer);
      syncTimer = setTimeout(async () => {
        const cfg = get().config;
        if (!cfg) return;
        // AWAIT the patch so the backend definitely has the latest state
        // before we ask for a new preview. Without the await, the preview
        // GET can race with the patch POST and render with stale state —
        // the user's reported "old header ghost reappears in preview".
        try {
          if (patch.font_size !== undefined && level === 0) {
            await Promise.all([
              api.patchColumnHeaders(cfg.column_headers),
              api.patchRowHeaders(cfg.row_headers),
            ]);
          } else {
            const patchFn = axis === "col" ? api.patchColumnHeaders : api.patchRowHeaders;
            const headers = axis === "col" ? cfg.column_headers : cfg.row_headers;
            await patchFn.call(api, headers);
          }
        } catch (e) {
          console.error(e);
        }
        get().requestPreview();
      }, 150);
    },

    updateLabelFormatting: (axis, index, patch, opts) => {
      set((s) => {
        if (!s.config) return;
        const labels = axis === "col" ? s.config.column_labels : s.config.row_labels;
        if (index < labels.length) {
          Object.assign(labels[index], patch);
          // 1.5: Sync font_size across ALL primary labels (both axes) for consistency
          if (patch.font_size !== undefined) {
            for (const lbl of [...s.config.column_labels, ...s.config.row_labels]) {
              lbl.font_size = patch.font_size;
            }
          }
          s.configDirty = true;
        }
      });
      // See updateHeaderGroupFormatting — defer styling pushes to
      // flushHeaderEdits() on blur so the preview renders once, not per tweak.
      if (opts?.skipSync) { headerEditsPending = true; return; }
      if (syncTimer) clearTimeout(syncTimer);
      syncTimer = setTimeout(async () => {
        const cfg = get().config;
        if (!cfg) return;
        // Await label patch so the backend has the latest labels before
        // we ask for a preview (same race that caused header ghosts).
        try {
          await Promise.all([
            api.patchColumnLabels(cfg.column_labels),
            api.patchRowLabels(cfg.row_labels),
          ]);
        } catch (e) {
          console.error(e);
        }
        get().requestPreview();
      }, 200);
    },

    // ── Panel drag-and-drop ─────────────────────────────────

    swapPanels: (r1, c1, r2, c2) => {
      set((s) => {
        if (!s.config) return;
        const p1 = s.config.panels[r1]?.[c1];
        const p2 = s.config.panels[r2]?.[c2];
        if (!p1 || !p2) return;
        const tmp = { ...p1 };
        s.config.panels[r1][c1] = { ...p2 };
        s.config.panels[r2][c2] = tmp;
        // The two cells now hold each other's panels — their
        // auto-letter labels travelled with the panel object and are
        // wrong for the new positions. Re-letter both.
        const cols = s.config.cols;
        _relabelAutoLetter(s.config.panels[r1][c1] as unknown as Parameters<typeof _relabelAutoLetter>[0], r1, c1, cols);
        _relabelAutoLetter(s.config.panels[r2][c2] as unknown as Parameters<typeof _relabelAutoLetter>[0], r2, c2, cols);
        // Swap processed thumbnails too
        const t1 = s.panelThumbnails[`${r1}-${c1}`];
        const t2 = s.panelThumbnails[`${r2}-${c2}`];
        if (t1) s.panelThumbnails[`${r2}-${c2}`] = t1; else delete s.panelThumbnails[`${r2}-${c2}`];
        if (t2) s.panelThumbnails[`${r1}-${c1}`] = t2; else delete s.panelThumbnails[`${r1}-${c1}`];
        s.configDirty = true;
      });
      const cfg = get().config;
      if (cfg) {
        Promise.all([
          api.patchPanel(r1, c1, cfg.panels[r1][c1] as unknown as Record<string, unknown>),
          api.patchPanel(r2, c2, cfg.panels[r2][c2] as unknown as Record<string, unknown>),
        ]).then(() => {
          // Refresh thumbnails after backend sync to ensure correct display
          get().refreshPanelThumbnail(r1, c1);
          get().refreshPanelThumbnail(r2, c2);
        }).catch(console.error);
      }
      get().requestPreview();
    },

    movePanelToDrawer: (r, c, drawerIdx) => {
      set((s) => {
        if (!s.config) return;
        const panel = s.config.panels[r]?.[c];
        if (!panel) return;
        // Ensure drawer is large enough
        while (s.drawerPanels.length <= drawerIdx) {
          s.drawerPanels.push(defaultPanel());
        }
        s.drawerPanels[drawerIdx] = { ...panel };
        // Preserve the processed thumbnail in the drawer
        const thumbKey = `${r}-${c}`;
        if (s.panelThumbnails[thumbKey]) {
          s.drawerThumbnails[drawerIdx] = s.panelThumbnails[thumbKey];
        }
        s.config.panels[r][c] = defaultPanel();
        s.configDirty = true;
        delete s.panelThumbnails[thumbKey];
      });
      const cfg = get().config;
      if (cfg) {
        api.patchPanel(r, c, cfg.panels[r][c] as unknown as Record<string, unknown>)
          .catch(console.error);
      }
      get().requestPreview();
    },

    movePanelFromDrawer: (drawerIdx, r, c) => {
      set((s) => {
        if (!s.config) return;
        const drawerPanel = s.drawerPanels[drawerIdx];
        if (!drawerPanel || !drawerPanel.image_name) return;
        const currentCell = s.config.panels[r]?.[c];
        if (!currentCell) return;
        const thumbKey = `${r}-${c}`;
        // If target cell has an image, move it to drawer (swap)
        if (currentCell.image_name) {
          s.drawerPanels[drawerIdx] = { ...currentCell };
          if (s.panelThumbnails[thumbKey]) {
            s.drawerThumbnails[drawerIdx] = s.panelThumbnails[thumbKey];
          }
        } else {
          s.drawerPanels[drawerIdx] = defaultPanel();
          delete s.drawerThumbnails[drawerIdx];
        }
        // Move drawer panel to cell and restore its thumbnail
        s.config.panels[r][c] = { ...drawerPanel };
        // Re-letter the destination cell — the panel just landed at a
        // new grid position, so its auto-label needs to match (r, c).
        _relabelAutoLetter(s.config.panels[r][c] as unknown as Parameters<typeof _relabelAutoLetter>[0], r, c, s.config.cols);
        if (s.drawerThumbnails[drawerIdx]) {
          s.panelThumbnails[thumbKey] = s.drawerThumbnails[drawerIdx];
        }
        s.configDirty = true;
      });
      const cfg = get().config;
      if (cfg) {
        api.patchPanel(r, c, cfg.panels[r][c] as unknown as Record<string, unknown>)
          .then(() => get().refreshPanelThumbnail(r, c))
          .catch(console.error);
      }
      get().requestPreview();
    },

    // ── Image management ──────────────────────────────────

    uploadImages: async (files) => {
      const pendingNames = files.map((f) => f.name);
      set((s) => {
        s.pendingUploads.push(...pendingNames);
      });
      try {
        const { names, thumbnails } = await api.uploadImages(files);
        set((s) => {
          for (const name of names) {
            s.loadedImages[name] = {
              name,
              thumbnailB64: thumbnails[name] ?? "",
            };
          }
          s.apiError = null;
        });
        get().requestPreview();
        return names;
      } catch (err) {
        console.error("Upload failed", err);
        const msg = err instanceof Error ? err.message : String(err);
        set((s) => { s.apiError = `Image upload failed: ${msg}`; });
        return [];
      } finally {
        set((s) => {
          s.pendingUploads = s.pendingUploads.filter((n) => !pendingNames.includes(n));
        });
      }
    },

    uploadImagesFromPaths: async (filePaths) => {
      const pendingNames = filePaths.map((p) => {
        // Extract filename from path (works for both / and \ separators)
        const parts = p.split(/[\\/]/);
        return parts[parts.length - 1] || p;
      });
      set((s) => {
        s.pendingUploads.push(...pendingNames);
      });
      try {
        const { names, thumbnails } = await api.uploadImagesFromPaths(filePaths);
        set((s) => {
          for (const name of names) {
            s.loadedImages[name] = {
              name,
              thumbnailB64: thumbnails[name] ?? "",
            };
          }
          s.apiError = null;
        });
        get().requestPreview();
        return names;
      } catch (err) {
        console.error("Upload from paths failed", err);
        const msg = err instanceof Error ? err.message : String(err);
        set((s) => { s.apiError = `Image upload failed: ${msg}`; });
        return [];
      } finally {
        set((s) => {
          s.pendingUploads = s.pendingUploads.filter((n) => !pendingNames.includes(n));
        });
      }
    },

    removeImage: async (name) => {
      try {
        await api.deleteImage(name);
        set((s) => {
          delete s.loadedImages[name];
          // Remove from any image group
          s.imageGroups.forEach(g => {
            g.imageNames = g.imageNames.filter(n => n !== name);
          });
          if (s.config) {
            for (let r = 0; r < s.config.rows; r++) {
              for (let c = 0; c < s.config.cols; c++) {
                if (s.config.panels[r][c].image_name === name) {
                  s.config.panels[r][c].image_name = "";
                }
              }
            }
          }
        });
        get().requestPreview();
      } catch (err) {
        console.error("Delete failed", err);
      }
    },

    // ── Image Groups ─────────────────────────────────────

    createImageGroup: (name) => {
      set((s) => {
        s.imageGroups.push({
          id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
          name,
          imageNames: [],
        });
      });
    },

    renameImageGroup: (groupId, name) => {
      set((s) => {
        const group = s.imageGroups.find(g => g.id === groupId);
        if (group) group.name = name;
      });
    },

    deleteImageGroup: (groupId) => {
      set((s) => {
        s.imageGroups = s.imageGroups.filter(g => g.id !== groupId);
      });
    },

    moveImageToGroup: (imageName, groupId) => {
      set((s) => {
        // Remove from any existing group first (single-location invariant)
        s.imageGroups.forEach(g => {
          g.imageNames = g.imageNames.filter(n => n !== imageName);
        });
        // Add to target group
        const target = s.imageGroups.find(g => g.id === groupId);
        if (target) target.imageNames.push(imageName);
      });
    },

    moveImageToTimeline: (imageName) => {
      set((s) => {
        s.imageGroups.forEach(g => {
          g.imageNames = g.imageNames.filter(n => n !== imageName);
        });
      });
    },

    // ── Preview ───────────────────────────────────────────

    requestPreview: () => {
      // Keep the top frame of the stack so we can see what code path
      // triggered this preview refresh. Useful for diagnosing "why did a
      // preview fetch happen when I didn't type anything" symptoms.
      try {
        const stack = (new Error().stack || "").split("\n").slice(2, 5).join(" | ");
        console.log("[mpf] requestPreview called", stack);
      } catch { /* noop */ }
      if (previewTimer) clearTimeout(previewTimer);
      // Short debounce — the upstream syncTimer already batches typing, so
      // we only need enough here to coalesce multiple near-simultaneous
      // `get().requestPreview()` calls (e.g., when several headers are
      // updated at once).
      previewTimer = setTimeout(async () => {
        const mySeq = ++previewSeq;
        console.log(`[mpf] preview fetch seq=${mySeq}`);
        set((s) => {
          s.previewLoading = true;
        });
        try {
          const resp = await api.getPreview();
          // Discard this response if a newer preview request has been
          // started since we fired this one — prevents a slow older
          // render from clobbering a fresher one (the reported "ghost
          // old header in preview" symptom).
          if (mySeq !== previewSeq) {
            console.log(`[mpf] preview fetch seq=${mySeq} DISCARDED (newer=${previewSeq})`);
            return;
          }
          console.log(`[mpf] preview fetch seq=${mySeq} APPLIED (imgBytes=${resp.image?.length ?? 0})`);
          set((s) => {
            s.previewImageB64 = resp.image;
            s.previewLoading = false;
            s.configDirty = false;
          });
        } catch {
          set((s) => {
            s.previewLoading = false;
          });
        }
      }, 100);
    },

    // ── Full config sync ──────────────────────────────────

    syncToBackend: async () => {
      const cfg = get().config;
      if (!cfg) return;
      try {
        await api.updateConfig(cfg);
      } catch (err) {
        console.error("Sync failed", err);
      }
    },

    // ── Project save/load ─────────────────────────────────

    saveProject: async (path) => {
      try {
        await get().syncToBackend();
        // Pull the latest analysis snapshot from the dedicated bridge
        // store — the AnalysisDialog publishes it on every state change.
        // Imported lazily to avoid a circular import on module load.
        const analysisSnapshot = (await import("./analysisStore")).useAnalysisStore.getState().snapshot;
        await api.saveProject(path, analysisSnapshot ?? null);
        // Remember the path so subsequent "Add to Collage" actions can
        // associate the rendered figure with this on-disk project,
        // and so the collage's Multi-Panel Builder button can offer
        // to reload it.
        set((s) => { s.currentProjectPath = path; s.configDirty = false; });
      } catch (err) {
        console.error("Save project failed", err);
      }
    },

    loadProject: async (path) => {
      try {
        const resp = await api.loadProject(path);
        set((s) => {
          s.config = resp.config;
          s.loadedImages = {};
          s.panelThumbnails = {};
          for (const name of resp.image_names) {
            s.loadedImages[name] = {
              name,
              thumbnailB64: resp.thumbnails[name] ?? "",
            };
          }
          s.currentProjectPath = path;
          s.configDirty = false;
        });
        // Push any analysis blob into the bridge store; the dialog
        // observes `hydrate` and rebuilds its tabs/plots from it.
        if (resp.analysis) {
          (await import("./analysisStore")).useAnalysisStore.getState().requestHydrate(resp.analysis);
        } else {
          // Loaded a project that has no analysis state — clear out
          // whatever was hanging around so the dialog is empty too.
          (await import("./analysisStore")).useAnalysisStore.getState().requestHydrate({
            manifest: { version: 1, tabs: [], activeTabId: "", tableMeta: {} },
            plots: {},
            tables: {},
          });
        }
        get().requestPreview();
        // Repopulate per-panel processed thumbnails (crop, levels,
        // pseudocolor, etc.). Without this, every panel cell falls back
        // to the raw uploaded image — so a cropped/processed panel would
        // suddenly appear uncropped right after a project load. We fire
        // these in parallel and don't await: the dialog returns
        // immediately, and each cell pops in as its render completes.
        const cfg = resp.config;
        for (let r = 0; r < cfg.rows; r++) {
          for (let c = 0; c < cfg.cols; c++) {
            if (cfg.panels[r]?.[c]?.image_name) {
              get().refreshPanelThumbnail(r, c).catch(console.error);
            }
          }
        }
      } catch (err) {
        console.error("Load project failed", err);
        // Re-throw so callers (e.g., the Sidebar's Load Project dialog)
        // can surface a useful error in the UI instead of pretending the
        // load succeeded after a silently-swallowed exception.
        throw err;
      }
    },

    saveFigure: async (path, format = "TIFF", background?: string, dpi = 300) => {
      try {
        await get().syncToBackend();
        const bg = background ?? get().config?.background ?? "White";
        await api.saveFigure(path, format, bg, dpi);
      } catch (err) {
        console.error("Save figure failed", err);
      }
    },

    // ── Grid resize conflict check ────────────────────────

    checkGridResizeConflict: (newRows, newCols) => {
      const cfg = get().config;
      if (!cfg) return [];
      const conflicts: string[] = [];
      for (let r = 0; r < cfg.rows; r++) {
        for (let c = 0; c < cfg.cols; c++) {
          if (r >= newRows || c >= newCols) {
            const name = cfg.panels[r]?.[c]?.image_name;
            if (name) {
              conflicts.push(`R${r + 1}C${c + 1}: ${name}`);
            }
          }
        }
      }
      return conflicts;
    },

    renumberPanels: () => {
      set((s) => {
        if (!s.config) return;
        _renumberAllAutoLetters(s.config as unknown as Parameters<typeof _renumberAllAutoLetters>[0]);
        s.configDirty = true;
      });
      // Push the renumbered cfg so the backend renders match.
      get().syncToBackend().catch(console.error);
      get().requestPreview();
    },
  })),
);
