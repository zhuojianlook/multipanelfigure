/* ──────────────────────────────────────────────────────────
   pivotGrid — the panel planner's Pivot button: an R×C figure
   becomes C×R. The panel at (r, c) moves to (c, r), the column
   labels and headers become the row labels and headers and the
   other way round. Pure, no store (figureStore._pivotGrid applies
   it, the Analysis workspace moves its keys with pivotAnalysisKey).

   Rules, chosen so that pivoting twice gives back the same figure
   (there is no undo stack — pressing Pivot again IS the undo):
     • sides swap Top↔Left and Bottom↔Right — a column label on Top
       becomes a row label on the Left, a row header on the Right
       becomes a column header at the Bottom. The defaults (row
       headers left, column headers top) stay the defaults;
     • text rotations swap exactly 0↔90 (horizontal column text
       becomes vertical row text); any other angle is kept;
     • header levels keep their order (index 0 is the outermost on
       both axes) and their spans: column i becomes row i;
     • an Adjacent Panel inset's side swaps the same way, so it still
       draws into the panel it drew into (which moved with it).
   Everything inside a panel (crop, labels, scale bar, annotations,
   insets) is in the panel's own image space and moves with it.

   Two normalisations: a label / header with no position or rotation
   comes back with the default one, and an Adjacent inset with no side
   comes back as "Right" (which is what an empty side means).
   ────────────────────────────────────────────────────────── */

import type { AxisLabel, FigureConfig, HeaderLevel, PanelInfo, ZoomInsetSettings } from "../api/types";

/** The side a position / inset side lands on after a pivot. */
export function pivotSide(s: string | null | undefined): string {
  switch (s) {
    case "Top": return "Left";
    case "Left": return "Top";
    case "Bottom": return "Right";
    case "Right": return "Bottom";
    default: return s ?? "";
  }
}

/** A label's / header's text rotation after a pivot: horizontal (0) and
 *  vertical (90) text swap; any other angle is the user's and is kept. */
export function pivotAxisRotation(r: number | null | undefined): number {
  const v = r ?? 0;
  if (v === 0) return 90;
  if (v === 90) return 0;
  return v;
}

/** A per-cell record keyed "r-c" (the planner's thumbnail caches) with every
 *  key moved to "c-r". Other keys are kept as they are. */
export function pivotCellRecord<T>(rec: Readonly<Record<string, T>>): Record<string, T> {
  const out: Record<string, T> = {};
  for (const [k, v] of Object.entries(rec)) {
    const m = /^(\d+)-(\d+)$/.exec(k);
    out[m ? `${m[2]}-${m[1]}` : k] = v;
  }
  return out;
}

function pivotLabel(l: AxisLabel): void {
  l.position = pivotSide(l.position ?? "Top");
  l.rotation = pivotAxisRotation(l.rotation);
}

function pivotHeaderLevels(levels: HeaderLevel[] | undefined): void {
  for (const level of levels || []) {
    for (const g of level?.headers || []) {
      g.position = pivotSide(g.position ?? "Top");
      g.rotation = pivotAxisRotation(g.rotation);
    }
  }
}

/** Swap two numeric fields of `obj` (only when both are numbers). */
function swapNumbers(obj: Record<string, unknown>, a: string, b: string): void {
  if (typeof obj[a] === "number" && typeof obj[b] === "number") {
    const t = obj[a];
    obj[a] = obj[b];
    obj[b] = t;
  }
}

/** An Adjacent Panel inset draws into the neighbour on its `side`; after the
 *  pivot that neighbour is on the swapped side. Other inset types draw
 *  inside their own panel and need nothing. */
function pivotInset(zi: ZoomInsetSettings | null | undefined): void {
  if (!zi || zi.inset_type !== "Adjacent Panel") return;
  zi.side = pivotSide(zi.side || "Right");
  // Stored but never read (not even in the TS type) — kept consistent.
  swapNumbers(zi as unknown as Record<string, unknown>, "adjacent_panel_row", "adjacent_panel_col");
}

function pivotPanel(p: PanelInfo): PanelInfo {
  for (const zi of p.zoom_insets || []) pivotInset(zi);
  pivotInset(p.zoom_inset);
  swapNumbers(p as unknown as Record<string, unknown>, "zoom_source_row", "zoom_source_col");
  return p;
}

/** `cfg` with its rows and columns swapped (a fresh copy; `cfg` is not
 *  touched). Auto panel letters are NOT re-derived here — the store does
 *  that afterwards, by position, as every other grid rearrangement does. */
export function pivotFigureConfig(cfg: FigureConfig): FigureConfig {
  const src = JSON.parse(JSON.stringify(cfg)) as FigureConfig;
  const rows = src.rows;
  const cols = src.cols;
  const panels: PanelInfo[][] = [];
  for (let c = 0; c < cols; c++) {
    const row: PanelInfo[] = [];
    // The store's grid is always full (the sidecar's ensure_grid makes it
    // so); a missing cell would be a default panel there, and `{}` is one.
    for (let r = 0; r < rows; r++) row.push(pivotPanel(src.panels?.[r]?.[c] ?? ({} as PanelInfo)));
    panels.push(row);
  }
  const columnLabels = src.row_labels || [];
  const rowLabels = src.column_labels || [];
  for (const l of columnLabels) pivotLabel(l);
  for (const l of rowLabels) pivotLabel(l);
  const columnHeaders = src.row_headers || [];
  const rowHeaders = src.column_headers || [];
  pivotHeaderLevels(columnHeaders);
  pivotHeaderLevels(rowHeaders);
  const out: FigureConfig = {
    ...src,
    rows: cols,
    cols: rows,
    panels,
    column_labels: columnLabels,
    row_labels: rowLabels,
    column_headers: columnHeaders,
    row_headers: rowHeaders,
  };
  // The label toggles swap as they are (absent means shown, on both axes),
  // so a figure that never set them doesn't gain them.
  const showCol = src.show_column_labels;
  const showRow = src.show_row_labels;
  delete out.show_column_labels;
  delete out.show_row_labels;
  if (showRow !== undefined) out.show_column_labels = showRow;
  if (showCol !== undefined) out.show_row_labels = showCol;
  return out;
}

// ── Analysis keys ───────────────────────────────────────────

/** The sidecar's raw source key: r{r}c{c}_panel / _i{n} / _area{n}. */
const RAW_KEY_RE = /^r(\d+)c(\d+)_(panel|i\d+|area\d+)$/;
/** A measurement row's panel field: R{r+1}C{c+1}. */
const MEAS_CELL_RE = /^R(\d+)C(\d+)$/;

/** A raw source key for the panel's new position: "r0c1_i0" → "r1c0_i0".
 *  Anything that isn't a raw source key is returned as it is. */
export function pivotRawKey(raw: string): string {
  const m = RAW_KEY_RE.exec(raw);
  return m ? `r${m[2]}c${m[1]}_${m[3]}` : raw;
}

/** A key of figure `fk` for the panel's new position: a source key
 *  `${fk}::r0c1_i0` or a measurement row `${fk}|R1C2|ann|reading`. Only
 *  that exact shape moves — another figure of the document (`${fk}#fig-2…`)
 *  or a document whose path merely starts like this one's is left alone.
 *  One key at a time and never by string replace, so r0c1 and r1c0 trade
 *  places instead of both ending up in the same cell. */
export function pivotAnalysisKey(key: string, fk: string): string {
  if (!key || !fk || !key.startsWith(fk)) return key;
  const rest = key.slice(fk.length);
  if (rest.startsWith("::")) {
    const raw = rest.slice(2);
    const moved = pivotRawKey(raw);
    return moved === raw ? key : `${fk}::${moved}`;
  }
  if (rest.startsWith("|")) {
    const parts = rest.slice(1).split("|");
    if (parts.length !== 3) return key;
    const m = MEAS_CELL_RE.exec(parts[0]);
    if (!m) return key;
    return `${fk}|R${m[2]}C${m[1]}|${parts[1]}|${parts[2]}`;
  }
  return key;
}
