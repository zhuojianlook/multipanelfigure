/* ──────────────────────────────────────────────────────────
   analysisStore — minimal Zustand store used as a bridge
   between the AnalysisDialog (where analysis state lives as
   React useState for ergonomic editing) and the Sidebar's
   project save/load handlers (which need a snapshot/restore
   surface accessible from outside the dialog).

   The dialog calls publishSnapshot() whenever its state
   changes; Sidebar's save handler reads the latest snapshot
   and forwards it to the backend. On load, Sidebar pushes a
   pending hydrate payload here; the dialog observes and
   rebuilds its React state from it on next mount/effect.

   Plots are kept as base64 PNGs and tables as CSV text — the
   same wire format used in the .mpf zip exchange.
   ────────────────────────────────────────────────────────── */

import { create } from "zustand";
import type { AnalysisPayload } from "../api/types";

// ── Snapshot shape (matches AnalysisPayload but with typed manifest) ──

export interface AnalysisPlotSnapshot {
  id: string;
  /** "" if the binary lives in payload.plots[id]; otherwise inline.
   *  When publishing a snapshot for project save the dialog hoists
   *  PNG bytes into a side-map (payload.plots) keyed by id, and
   *  leaves this empty so the snapshot JSON stays small. */
  b64?: string;
  mainName?: string;
  collageId?: string;
  /** References to AnalysisTable ids whose CSV lives in
   *  payload.tables. */
  tableIds?: string[];
}

export interface AnalysisTabSnapshot {
  id: string;
  name: string;
  measureType: string;
  plotType: string;
  statTest: string;
  code: string;
  plots: AnalysisPlotSnapshot[];
}

export interface AnalysisTableSnapshot {
  id: string;
  name: string;
}

/** The frontend-defined manifest shape that gets stored in
 *  analysis/manifest.json inside the .mpf zip. */
export interface AnalysisManifest {
  version: 1;
  tabs: AnalysisTabSnapshot[];
  activeTabId: string;
  /** Names → AnalysisTable id metadata (the CSV bodies live in
   *  the AnalysisPayload.tables side-map). */
  tableMeta: Record<string, AnalysisTableSnapshot>;
}

interface AnalysisStoreState {
  /** Last snapshot published by the AnalysisDialog. The Sidebar's
   *  save handler reads this synchronously. */
  snapshot: AnalysisPayload | null;
  /** A pending hydrate payload set by the Sidebar's load handler.
   *  The dialog watches this and replaces its React state when
   *  it changes. The dialog clears it back to null after consuming. */
  hydrate: AnalysisPayload | null;

  publishSnapshot: (s: AnalysisPayload | null) => void;
  requestHydrate: (p: AnalysisPayload | null) => void;
  consumeHydrate: () => AnalysisPayload | null;
}

export const useAnalysisStore = create<AnalysisStoreState>((set, get) => ({
  snapshot: null,
  hydrate: null,

  publishSnapshot: (s) => set({ snapshot: s }),
  requestHydrate: (p) => set({ hydrate: p }),
  consumeHydrate: () => {
    const cur = get().hydrate;
    if (cur) set({ hydrate: null });
    return cur;
  },
}));
