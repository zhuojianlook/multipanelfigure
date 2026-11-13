/* ──────────────────────────────────────────────────────────
   drawerState — which parts of the Groups drawer are collapsed.

   It lives here, not in the rows' own React state: collapsing the
   whole drawer unmounts every group row, so per-row state came back
   "expanded" for every group each time the drawer reopened. Kept for
   the app session (it also survives the strip remounting on a mode
   switch); it is a view preference, so it is not saved in the .mpf
   and never marks the document unsaved.
   ────────────────────────────────────────────────────────── */

import { create } from "zustand";

interface DrawerState {
  /** The Groups drawer as a whole. */
  drawerOpen: boolean;
  /** Group ids the user collapsed. Absent = expanded (the default). */
  collapsedGroups: Record<string, true>;
  setDrawerOpen: (open: boolean) => void;
  setGroupCollapsed: (groupId: string, collapsed: boolean) => void;
}

export const useDrawerState = create<DrawerState>()((set) => ({
  drawerOpen: true,
  collapsedGroups: {},
  setDrawerOpen: (open) => set({ drawerOpen: open }),
  setGroupCollapsed: (groupId, collapsed) => set((s) => {
    if (!!s.collapsedGroups[groupId] === collapsed) return s;
    const next = { ...s.collapsedGroups };
    if (collapsed) next[groupId] = true;
    else delete next[groupId];
    return { collapsedGroups: next };
  }),
}));
