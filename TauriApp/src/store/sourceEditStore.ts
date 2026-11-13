/* ──────────────────────────────────────────────────────────
   sourceEditStore — the bridge between Edit Panel and Analysis
   for "this image changed after Analysis used it".

   The Analysis view is unmounted in Builder mode, so the two sides
   meet here (see utils/analysisStale.ts for the model):
     • revs        — per source key (nsKey), bumped by figureStore.
                     updatePanel when a panel edit changes the pixels
                     that source extracts;
     • withResults — published by the Analysis graph: source keys that
                     feed a node with results (survives its unmount);
     • rerun       — keys the user chose "Update" for; the graph re-runs
                     the stale nodes that read them the next time it is
                     mounted and its sources are loaded.
   In memory only — the same lifetime as Analysis results, which are
   not saved in the .mpf either.
   ────────────────────────────────────────────────────────── */

import { create } from "zustand";

interface SourceEditState {
  revs: Record<string, number>;
  withResults: string[];
  rerun: Record<string, true>;
  bump: (keys: string[]) => void;
  publishWithResults: (keys: string[]) => void;
  requestRerun: (keys: string[]) => void;
  clearRerun: () => void;
}

export const useSourceEditStore = create<SourceEditState>()((set) => ({
  revs: {},
  withResults: [],
  rerun: {},
  bump: (keys) => {
    if (!keys.length) return;
    set((s) => {
      const revs = { ...s.revs };
      for (const k of keys) revs[k] = (revs[k] ?? 0) + 1;
      return { revs };
    });
  },
  publishWithResults: (keys) => set((s) => {
    const same = s.withResults.length === keys.length && s.withResults.every((k, i) => k === keys[i]);
    return same ? s : { withResults: keys };
  }),
  requestRerun: (keys) => {
    if (!keys.length) return;
    set((s) => {
      const rerun = { ...s.rerun };
      for (const k of keys) rerun[k] = true;
      return { rerun };
    });
  },
  clearRerun: () => set((s) => (Object.keys(s.rerun).length ? { rerun: {} } : s)),
}));
