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

export type CollageItemKind = "figure" | "image";

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
};

export type WorkspaceMode = "builder" | "collage";

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
  /** Background color of the canvas. */
  background: string;
  /** Currently-selected item id, or null. */
  selectedId: string | null;
  /** Gridline visibility + snap-to-grid step. Both default ON. */
  gridVisible: boolean;
  snapEnabled: boolean;
  gridStep: number;
  /** Target visual point size for ALL figure-kind item headers /
   *  primary labels in the collage. Null = no normalisation (each
   *  item shows headers at its own pt × its collage scale). When
   *  set, the renderer compensates per-item so headers land at this
   *  pt regardless of how the user has scaled the figure. */
  globalHeaderPt: number | null;

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
  setCanvasSize: (w: number, h: number) => void;
  setBackground: (bg: string) => void;
  clear: () => void;
};

const STORAGE_KEY = "mpfig_collage_v1";

type Persisted = Pick<CollageState, "items" | "canvasW" | "canvasH" | "background" | "gridVisible" | "snapEnabled" | "gridStep" | "globalHeaderPt">;

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
    }));
  } catch {
    /* quota — ignore */
  }
}

export const useCollageStore = create<CollageState>()(
  immer((set, get) => ({
    ...loadInitial(),
    selectedId: null,
    mode: "builder" as WorkspaceMode,

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
        if (s.selectedId === id) s.selectedId = null;
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

    setSelectedId: (id) => set((s) => { s.selectedId = id; }),

    setCanvasSize: (w, h) => {
      set((s) => { s.canvasW = w; s.canvasH = h; });
      persist(get());
    },

    setBackground: (bg) => {
      set((s) => { s.background = bg; });
      persist(get());
    },

    clear: () => {
      set((s) => { s.items = []; s.selectedId = null; });
      persist(get());
    },
  })),
);
