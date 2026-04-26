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

export type CollageItem = {
  id: string;
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
};

export type WorkspaceMode = "builder" | "collage";

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

  setMode: (m: WorkspaceMode) => void;

  addItem: (item: Omit<CollageItem, "id" | "z" | "rotation"> & Partial<Pick<CollageItem, "rotation">>) => string;
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

function loadInitial(): Pick<CollageState, "items" | "canvasW" | "canvasH" | "background"> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const data = JSON.parse(raw);
      if (data && Array.isArray(data.items)) {
        return {
          items: data.items,
          canvasW: data.canvasW || 1600,
          canvasH: data.canvasH || 1200,
          background: data.background || "#FFFFFF",
        };
      }
    }
  } catch {
    /* ignore corrupt storage */
  }
  return { items: [], canvasW: 1600, canvasH: 1200, background: "#FFFFFF" };
}

function persist(s: Pick<CollageState, "items" | "canvasW" | "canvasH" | "background">) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      items: s.items, canvasW: s.canvasW, canvasH: s.canvasH, background: s.background,
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

    addItem: (item) => {
      const id = `item_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      set((s) => {
        const maxZ = s.items.reduce((acc, it) => Math.max(acc, it.z), 0);
        s.items.push({
          ...item,
          id,
          z: maxZ + 1,
          rotation: item.rotation ?? 0,
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
