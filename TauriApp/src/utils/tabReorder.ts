/* ──────────────────────────────────────────────────────────
   tabReorder — the arithmetic behind drag-to-reorder for the .mpf
   document tabs (components/layout/DocumentTabs). Pure, so it is
   tested without a DOM.

   Positions are INSERTION SLOTS in the current order: slot 0 is before
   the first tab, slot n after the last.
   ────────────────────────────────────────────────────────── */

/** The slot a pointer at `x` drops into, given each tab's horizontal
 *  midpoint in current order: the number of tabs whose midpoint lies
 *  left of the pointer. */
export function dropSlot(mids: readonly number[], x: number): number {
  let n = 0;
  for (const m of mids) if (m < x) n++;
  return n;
}

/** `ids` with `dragId` moved into `slot` (0..ids.length, measured in the
 *  order BEFORE the move). Returns `ids` itself — the same array — when
 *  nothing would change (unknown id, or the slot right before / after the
 *  tab), so callers can skip a no-op store write. */
export function moveToSlot<T extends string>(ids: readonly T[], dragId: T, slot: number): readonly T[] {
  const from = ids.indexOf(dragId);
  if (from < 0) return ids;
  const s = Math.max(0, Math.min(ids.length, Math.round(slot)));
  if (s === from || s === from + 1) return ids;
  const out = ids.filter((id) => id !== dragId);
  out.splice(s > from ? s - 1 : s, 0, dragId);
  return out;
}
