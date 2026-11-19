/* ──────────────────────────────────────────────────────────
   analysisKeys — the identities Analysis binds sources by.
   Pure; shared by the Analysis graph and the figure store (which
   records Edit Panel changes to analysed sources, see
   utils/analysisStale.ts) and the workflow persistence helpers
   (utils/workflowOwnership, utils/workflowPersist). Moved verbatim
   from AnalysisNodeGraph, which re-exports figKey / nsKey / rawKeyOf.
   ────────────────────────────────────────────────────────── */

import { DEFAULT_FIGURE_ID } from "./figures";
import type { MeasurementRef } from "./analysisTypes";

/** Stable identity for the figure a source or measurement belongs to.
 *
 *  A doc id is only meaningful within one session: saved workflows persist to
 *  localStorage but `openDocs` does not (collageStore's `Persisted` omits it),
 *  so ids are re-minted on every launch. A path survives a restart; an Untitled
 *  figure has no durable identity at all, and a binding to one is deliberately
 *  allowed to go stale rather than silently resolve to a different figure.
 *
 *  Paths are compared as the raw strings the store holds. Two strings can name
 *  one file (bare "x.mpf" vs "~/Documents/x.mpf" — see docEnsure), which splits
 *  one figure into two identities. That over-separates, which is the safe way to
 *  be wrong here: it can strand a binding, never cross-bind it.
 *
 *  A document can hold several figure tabs (utils/figures.ts). Every figure but
 *  the project's first (DEFAULT_FIGURE_ID) appends `#<figureId>`, so bindings
 *  made before figure tabs — which carry no figure id — keep resolving to the
 *  first figure, which is the only figure those projects had. */
export function figKey(d: { id: string; path?: string | null }, figureId?: string): string {
  const p = (d.path || "").trim();
  const doc = p ? `p:${p}` : `d:${d.id}`;
  return figureId && figureId !== DEFAULT_FIGURE_ID ? `${doc}#${figureId}` : doc;
}

/** Globally unique source key: which figure, then which inset inside it.
 *  FRONTEND ONLY — it embeds a path, so it must never reach the sidecar
 *  (see runtimeKeyFor). */
export function nsKey(fig: string, raw: string): string {
  return `${fig}::${raw}`;
}

/** The backend's own key for an inset — bare grid coords, e.g. "r0c0_i0".
 *  Unique within a figure and identical ACROSS figures, which is why every
 *  key that leaves this module is namespaced by figKey. */
export function rawKeyOf(s: { key: string; rawKey?: string; name?: string }): string {
  if (s.rawKey) return s.rawKey;
  // Standalone uploads ("img:<filename>") belong to no figure, are never
  // namespaced, and their key is already raw. Check this FIRST: a filename can
  // contain "__", which the legacy split below would chop.
  if (s.name) return s.key;
  const ns = s.key.lastIndexOf("::");
  if (ns >= 0) return s.key.slice(ns + 2);
  // Legacy: either bare ("r0c0_i0") or doc-id-namespaced ("doc_x__r0c0_i0").
  // Doc ids contain underscores, so split on the LAST "__" — no backend key
  // contains one (r{r}c{c}_i{n} / _panel / _area{n}).
  const i = s.key.lastIndexOf("__");
  return i >= 0 ? s.key.slice(i + 2) : s.key;
}

/** Every measurement table attached to a node, oldest first.
 *
 *  Normalises the two shapes a node can carry: the legacy single
 *  `measurementsRef` (all that saved workflows / stored sessions ever wrote)
 *  and the newer `measurementRefs` list. Slot 0 is always the legacy field, so
 *  the first table keeps the stable `out_table_measurements` handle id and old
 *  edges stay connected. */
export function measRefsOf(
  data: { measurementsRef?: MeasurementRef; measurementRefs?: MeasurementRef[] } | undefined,
): MeasurementRef[] {
  if (!data) return [];
  if (data.measurementRefs && data.measurementRefs.length) return data.measurementRefs;
  return data.measurementsRef ? [data.measurementsRef] : [];
}

/** Handle id for the i-th measurement table. Index 0 keeps the original id so
 *  every previously-saved edge keeps resolving. */
export function measHandleId(i: number): string {
  return i === 0 ? "out_table_measurements" : `out_table_measurements_${i}`;
}

/** Inverse of measHandleId — the bare id (and anything unparseable) is slot 0,
 *  which is what keeps legacy edges pointing at the first table. */
export function measIndexFromHandle(handle: string | null | undefined): number {
  const m = /^out_table_measurements_(\d+)$/.exec(handle || "");
  return m ? Number(m[1]) : 0;
}
