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

  updateColumnLabel: (colIdx: number, text: string) => void;
  updateRowLabel: (rowIdx: number, text: string) => void;

  addColumnHeaderLevel: () => void;
  removeColumnHeaderLevel: (levelIdx: number) => void;
  updateHeaderGroupText: (
    axis: "col" | "row",
    level: number,
    groupIdx: number,
    text: string,
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
  ) => void;
  updateLabelFormatting: (
    axis: "col" | "row",
    index: number,
    patch: Partial<Pick<import("../api/types").AxisLabel, "font_size" | "font_name" | "font_style" | "default_color" | "position" | "distance" | "rotation" | "styled_segments">>,
  ) => void;

  // Header level swapping
  swapColumnHeaderLevels: (level1: number, level2: number) => void;
  swapRowHeaderLevels: (level1: number, level2: number) => void;

  // Panel drag-and-drop
  swapPanels: (r1: number, c1: number, r2: number, c2: number) => void;
  movePanelToDrawer: (r: number, c: number, drawerIdx: number) => void;
  movePanelFromDrawer: (drawerIdx: number, r: number, c: number) => void;

  uploadImages: (files: File[]) => Promise<void>;
  uploadImagesFromPaths: (filePaths: string[]) => Promise<void>;
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
}

// ── Helpers ──────────────────────────────────────────────

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
        const cfg = await api.patchGrid(rows, cols, get().config?.spacing ?? 0.02);
        set((s) => {
          s.config = cfg;
          s.panelThumbnails = {};
          s.configDirty = true;
        });
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
      set((s) => {
        if (s.config && s.config.panels[row] && s.config.panels[row][col]) {
          Object.assign(s.config.panels[row][col], patch);
          s.configDirty = true;
        }
      });
      const panel = get().config?.panels[row]?.[col];
      if (panel) {
        api.patchPanel(row, col, panel as unknown as Record<string, unknown>)
          .then(() => get().refreshPanelThumbnail(row, col))
          .catch(console.error);
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

    updateColumnLabel: (colIdx, text) => {
      set((s) => {
        if (s.config && colIdx < s.config.column_labels.length) {
          s.config.column_labels[colIdx].text = text;
          s.configDirty = true;
        }
      });
      if (syncTimer) clearTimeout(syncTimer);
      syncTimer = setTimeout(() => {
        const cfg = get().config;
        if (cfg) {
          api.patchColumnLabels(cfg.column_labels).catch(console.error);
          get().requestPreview();
        }
      }, 500);
    },

    updateRowLabel: (rowIdx, text) => {
      set((s) => {
        if (s.config && rowIdx < s.config.row_labels.length) {
          s.config.row_labels[rowIdx].text = text;
          s.configDirty = true;
        }
      });
      if (syncTimer) clearTimeout(syncTimer);
      syncTimer = setTimeout(() => {
        const cfg = get().config;
        if (cfg) {
          api.patchRowLabels(cfg.row_labels).catch(console.error);
          get().requestPreview();
        }
      }, 500);
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

    updateHeaderGroupText: (axis, level, groupIdx, text) => {
      console.log("[mpf] updateHeaderGroupText", { axis, level, groupIdx, text: JSON.stringify(text), len: text.length });
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

    updateHeaderGroupFormatting: (axis, level, groupIdx, patch) => {
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

    updateLabelFormatting: (axis, index, patch) => {
      set((s) => {
        if (!s.config) return;
        const labels = axis === "col" ? s.config.column_labels : s.config.row_labels;
        if (index < labels.length) {
          Object.assign(labels[index], patch);
          // 1.5: Sync font_size across ALL primary labels (both axes) for consistency
          if (patch.font_size !== undefined) {
            const otherLabels = axis === "col" ? s.config.row_labels : s.config.column_labels;
            for (const lbl of [...s.config.column_labels, ...s.config.row_labels]) {
              lbl.font_size = patch.font_size;
            }
          }
          s.configDirty = true;
        }
      });
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
      } catch (err) {
        console.error("Upload failed", err);
        const msg = err instanceof Error ? err.message : String(err);
        set((s) => { s.apiError = `Image upload failed: ${msg}`; });
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
      } catch (err) {
        console.error("Upload from paths failed", err);
        const msg = err instanceof Error ? err.message : String(err);
        set((s) => { s.apiError = `Image upload failed: ${msg}`; });
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
        await api.saveProject(path);
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
        });
        get().requestPreview();
      } catch (err) {
        console.error("Load project failed", err);
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
  })),
);
