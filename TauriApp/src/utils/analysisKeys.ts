/* ──────────────────────────────────────────────────────────
   analysisKeys — the identities Analysis binds sources by.
   Pure; shared by the Analysis graph and the figure store (which
   records Edit Panel changes to analysed sources, see
   utils/analysisStale.ts). Moved verbatim from AnalysisNodeGraph,
   which re-exports both.
   ────────────────────────────────────────────────────────── */

import { DEFAULT_FIGURE_ID } from "./figures";

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
