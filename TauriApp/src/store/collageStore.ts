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
  /** Text-item ("text" kind) styling. `text` is the content; the rest are
   *  optional style overrides with sensible defaults. */
  text?: string;
  fontSize?: number;
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
};

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

/** Default canvas: Nature full-page proportions (183 mm × 247 mm) at 300 DPI
 *  → 2161 × 2917 px. Common scientific-publication target so users don't
 *  have to look up dimensions before assembling. */
export const DEFAULT_CANVAS_W = 2161;
export const DEFAULT_CANVAS_H = 2917;
/** Grid step in canvas pixels. 50 px ≈ 4.2 mm at 300 DPI — fine enough for
 *  alignment without being claustrophobic on the screen. */
export const DEFAULT_GRID_STEP = 50;

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
  /** Target visual point size for ALL figure-kind item headers /
   *  primary labels in the collage. Null = no normalisation (each
   *  item shows headers at its own pt × its collage scale). When
   *  set, the renderer compensates per-item so headers land at this
   *  pt regardless of how the user has scaled the figure. */
  globalHeaderPt: number | null;

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
  clear: () => void;

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

type Persisted = Pick<CollageState, "items" | "canvasW" | "canvasH" | "background" | "gridVisible" | "snapEnabled" | "gridStep" | "globalHeaderPt" | "guideColumns" | "guideGutter" | "guidesVisible" | "elemOverridesByItem">;

function loadInitial(): Persisted {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const data = JSON.parse(raw);
      if (data && Array.isArray(data.items)) {
        // Migrate older collage items that pre-date later fields.
        // The 0.1.145 build briefly used `stashPath`; rename to
        // `projectPath` while preserving any existing values.
        const items = data.items.map((it: any) => ({
          kind: it.kind || "image",
          projectPath: it.projectPath ?? it.stashPath ?? null,
          ...it,
        })) as CollageItem[];
        return {
          items,
          canvasW: data.canvasW || DEFAULT_CANVAS_W,
          canvasH: data.canvasH || DEFAULT_CANVAS_H,
          background: data.background || "#FFFFFF",
          gridVisible: data.gridVisible ?? true,
          snapEnabled: data.snapEnabled ?? true,
          gridStep: data.gridStep || DEFAULT_GRID_STEP,
          globalHeaderPt: typeof data.globalHeaderPt === "number" ? data.globalHeaderPt : null,
          guideColumns: typeof data.guideColumns === "number" ? data.guideColumns : 0,
          guideGutter: typeof data.guideGutter === "number" ? data.guideGutter : 0,
          guidesVisible: data.guidesVisible ?? true,
          elemOverridesByItem: (data.elemOverridesByItem && typeof data.elemOverridesByItem === "object") ? data.elemOverridesByItem : {},
        };
      }
    }
  } catch {
    /* ignore corrupt storage */
  }
  return {
    items: [],
    canvasW: DEFAULT_CANVAS_W,
    canvasH: DEFAULT_CANVAS_H,
    background: "#FFFFFF",
    gridVisible: true,
    snapEnabled: true,
    gridStep: DEFAULT_GRID_STEP,
    globalHeaderPt: null,
    guideColumns: 0,
    guideGutter: 0,
    guidesVisible: true,
    elemOverridesByItem: {},
  };
}

function persist(s: Persisted) {
  try {
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
      elemOverridesByItem: s.elemOverridesByItem,
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
        s.items.push({
          ...item,
          id,
          z: maxZ + 1,
          rotation: item.rotation ?? 0,
          kind: item.kind ?? "image",
          projectPath: item.projectPath ?? null,
        } as CollageItem);
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

    setCanvasSize: (w, h) => {
      set((s) => { s.canvasW = w; s.canvasH = h; });
      persist(get());
    },

    setBackground: (bg) => {
      set((s) => { s.background = bg; });
      persist(get());
    },

    setColumnGuides: (columns, gutter) => {
      set((s) => { s.guideColumns = Math.max(0, Math.round(columns)); s.guideGutter = Math.max(0, Math.round(gutter)); });
      persist(get());
    },

    setGuidesVisible: (v) => {
      set((s) => { s.guidesVisible = v; });
      persist(get());
    },

    clear: () => {
      set((s) => { s.items = []; s.selectedId = null; s.selectedIds = []; });
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
