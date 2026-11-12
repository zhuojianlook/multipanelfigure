/* ──────────────────────────────────────────────────────────
   Figure tabs — several figures built from ONE project's images.

   Data model (FigureTab in api/types.ts):
     - The project keeps ONE set of uploaded images and image groups,
       shared by every figure. Everything else in FigureConfig (grid,
       panels, labels, headers, dpi, background, …) is per-figure.
     - The ACTIVE figure's config is the store's `config` — and the
       sidecar's `cfg` — so every existing row/col consumer keeps working
       unchanged. Inactive figures are "parked" as full configs in the
       `figures` list; the active entry's `config` is null.
     - On disk (config.json in the .mpf) the top level is the active
       figure exactly as before, plus `figures` (every figure in tab
       order, the active one without `config`) and `active_figure_id`.
       Builds that predate figure tabs ignore both keys and open the
       figure that was active at save time.
     - A project that has never had a second figure has exactly one
       figure, id DEFAULT_FIGURE_ID. Legacy collage items and analysis
       bindings (which carry no figure id) resolve to that id.

   Every function here is pure and immutable: inputs are never written
   (the store feeds it immer-frozen state) and a no-op returns the very
   same reference so callers can skip the unsaved flag + backend sync.
   ────────────────────────────────────────────────────────── */

import type { FigureConfig, FigureTab } from "../api/types";

export const DEFAULT_FIGURE_ID = "fig-1";

/** Keys that belong to the PROJECT, not to one figure: they follow the
 *  user from figure to figure on a switch. (Uploaded images live outside
 *  the config altogether.) */
export const SHARED_FIGURE_KEYS = ["resolution_entries", "scale_definitions", "image_groups"] as const;

/** Keys that describe the project's figure list itself; never stored
 *  inside a parked figure's config. */
const PROJECT_ONLY_KEYS = ["figures", "active_figure_id", "image_groups"] as const;

/** The parts of the store a figure operation reads and replaces. */
export interface FiguresState {
  config: FigureConfig;
  figures: FigureTab[];
  activeFigureId: string;
}

/** Per-image usage across the project's figures. */
export interface ImageUsage {
  /** Panels of the ACTIVE figure that show the image. */
  inCurrent: number;
  /** Other figures that show it, in tab order. */
  others: { id: string; name: string; count: number }[];
}

type Obj = Record<string, unknown>;

const isObj = (v: unknown): v is Obj =>
  typeof v === "object" && v !== null && !Array.isArray(v);

/** A figure id must be safe inside analysis runtime keys and file names. */
const VALID_ID = /^[A-Za-z0-9-]{1,64}$/;

/** True when `v` looks enough like a FigureConfig to render. */
export function isUsableConfig(v: unknown): v is FigureConfig {
  return isObj(v)
    && typeof v.rows === "number"
    && typeof v.cols === "number"
    && Array.isArray(v.panels);
}

/** The single figure of a project saved before figure tabs existed. */
function defaultFigures(): { figures: FigureTab[]; activeFigureId: string } {
  return {
    figures: [{ id: DEFAULT_FIGURE_ID, name: "Figure 1", config: null }],
    activeFigureId: DEFAULT_FIGURE_ID,
  };
}

/** A fresh figure id not in `figures`. */
export function newFigureId(figures: readonly FigureTab[]): string {
  const taken = new Set(figures.map((f) => f.id));
  for (;;) {
    const id = "fig-" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    if (!taken.has(id)) return id;
  }
}

/** `base`, `base-2`, `base-3`… — whichever is first not in `taken`. */
function uniqueId(base: string, taken: Set<string>): string {
  if (!taken.has(base)) return base;
  for (let n = 2; ; n++) {
    const id = `${base}-${n}`;
    if (!taken.has(id)) return id;
  }
}

/** "Figure N" for the next tab, skipping names already used. */
export function uniqueFigureName(figures: readonly FigureTab[], base = "Figure"): string {
  const names = new Set(figures.map((f) => f.name));
  for (let n = figures.length + 1; ; n++) {
    const name = `${base} ${n}`;
    if (!names.has(name)) return name;
  }
}

/** Load-time normalization of `config.figures` + `config.active_figure_id`.
 *  Never throws. Old files (no list) become one DEFAULT_FIGURE_ID figure.
 *  Non-active entries without a usable config are dropped (there is
 *  nothing to show for them); the active entry always ends with
 *  `config: null` because its config is the file's top level. */
export function normalizeFigures(
  raw: unknown,
  rawActiveId: unknown,
): { figures: FigureTab[]; activeFigureId: string } {
  if (!Array.isArray(raw) || raw.length === 0) return defaultFigures();

  const taken = new Set<string>();
  const entries: FigureTab[] = [];
  raw.forEach((e, i) => {
    if (!isObj(e)) return;
    const wanted = typeof e.id === "string" && VALID_ID.test(e.id) ? e.id : `fig-${i + 1}`;
    const id = uniqueId(wanted, taken);
    taken.add(id);
    const name = typeof e.name === "string" && e.name.trim() ? e.name : `Figure ${i + 1}`;
    const config = isUsableConfig(e.config) ? e.config : null;
    entries.push({ ...e, id, name, config });
  });
  if (entries.length === 0) return defaultFigures();

  const activeId = typeof rawActiveId === "string" ? rawActiveId : "";
  let active = entries.find((f) => f.id === activeId) ?? entries.find((f) => f.config === null);
  if (!active) {
    // Every entry carries a config, so none of them is the top level:
    // the top level is a figure of its own.
    const id = uniqueId(DEFAULT_FIGURE_ID, taken);
    active = { id, name: uniqueFigureName(entries), config: null };
    entries.unshift(active);
  }
  const figures = entries
    .filter((f) => f === active || f.config !== null)
    .map((f) => (f === active && f.config !== null ? { ...f, config: null } : f));
  return { figures, activeFigureId: active.id };
}

/** `cfg` without the project-level keys (figure list, image groups). */
export function stripProjectKeys(cfg: FigureConfig): FigureConfig {
  const out: Obj = { ...cfg };
  for (const k of PROJECT_ONLY_KEYS) delete out[k];
  return out as unknown as FigureConfig;
}

/** The wire/on-disk form of the figure list: every figure in tab order,
 *  the active one without `config`, parked configs without project keys. */
export function serializeFigures(figures: readonly FigureTab[], activeFigureId: string): Obj[] {
  return figures.map((f) => {
    const { config, ...rest } = f;
    if (f.id === activeFigureId || !config) return rest;
    return { ...rest, config: stripProjectKeys(config) };
  });
}

/** Copy the project-shared keys of `from` onto a copy of `to`. */
function withSharedKeys(to: FigureConfig, from: FigureConfig): FigureConfig {
  const out: Obj = { ...stripProjectKeys(to) };
  const src = from as unknown as Obj;
  for (const k of SHARED_FIGURE_KEYS) {
    if (k in src && src[k] !== undefined) out[k] = src[k];
  }
  return out as unknown as FigureConfig;
}

/** Make `targetId` the active figure: park the current config on the
 *  outgoing entry and bring the target's config to the top level, carrying
 *  the project-shared keys across. Unknown id / already active → input. */
export function switchActiveFigure(state: FiguresState, targetId: string): FiguresState {
  if (targetId === state.activeFigureId) return state;
  const target = state.figures.find((f) => f.id === targetId);
  if (!target || !target.config) return state;
  const outgoing = stripProjectKeys(state.config);
  const incoming = withSharedKeys(target.config, state.config);
  const figures = state.figures.map((f) => {
    if (f.id === state.activeFigureId) return { ...f, config: outgoing };
    if (f.id === targetId) return { ...f, config: null };
    return f;
  });
  return { config: incoming, figures, activeFigureId: targetId };
}

/** A deep, plain-JSON copy (configs are JSON; immer-frozen input is fine). */
function cloneConfig(cfg: FigureConfig): FigureConfig {
  return JSON.parse(JSON.stringify(cfg)) as FigureConfig;
}

/** Append a new figure with `config` and make it active. */
export function addFigure(
  state: FiguresState,
  opts: { config: FigureConfig; name?: string },
): FiguresState {
  const id = newFigureId(state.figures);
  const name = opts.name?.trim() || uniqueFigureName(state.figures);
  const figures = [...state.figures, { id, name, config: stripProjectKeys(opts.config) }];
  return switchActiveFigure({ ...state, figures }, id);
}

/** Copy figure `id` (active or parked) into a new tab right after it and
 *  make the copy active. Unknown id → input. */
export function duplicateFigure(state: FiguresState, id: string): FiguresState {
  const idx = state.figures.findIndex((f) => f.id === id);
  if (idx < 0) return state;
  const src = state.figures[idx];
  const srcConfig = id === state.activeFigureId ? state.config : src.config;
  if (!srcConfig) return state;
  const copyId = newFigureId(state.figures);
  const names = new Set(state.figures.map((f) => f.name));
  let name = `${src.name} copy`;
  for (let n = 2; names.has(name); n++) name = `${src.name} copy ${n}`;
  const figures = [...state.figures];
  figures.splice(idx + 1, 0, { id: copyId, name, config: cloneConfig(stripProjectKeys(srcConfig)) });
  return switchActiveFigure({ ...state, figures }, copyId);
}

/** Remove figure `id`. The last figure can't be removed. Removing the
 *  active figure first activates its right neighbour (else the left). */
export function deleteFigure(state: FiguresState, id: string): FiguresState {
  if (state.figures.length <= 1) return state;
  const idx = state.figures.findIndex((f) => f.id === id);
  if (idx < 0) return state;
  let next = state;
  if (id === state.activeFigureId) {
    const neighbour = state.figures[idx + 1] ?? state.figures[idx - 1];
    next = switchActiveFigure(state, neighbour.id);
    if (next === state) return state;
  }
  return { ...next, figures: next.figures.filter((f) => f.id !== id) };
}

/** Rename figure `id`. Blank or unchanged name → same array. */
export function renameFigure(figures: FigureTab[], id: string, name: string): FigureTab[] {
  const clean = name.trim();
  const f = figures.find((x) => x.id === id);
  if (!f || !clean || f.name === clean) return figures;
  return figures.map((x) => (x.id === id ? { ...x, name: clean } : x));
}

/** Move figure `id` one tab left (-1) or right (+1). At an edge → same array. */
export function moveFigure(figures: FigureTab[], id: string, dir: -1 | 1): FigureTab[] {
  const i = figures.findIndex((f) => f.id === id);
  const j = i + dir;
  if (i < 0 || j < 0 || j >= figures.length) return figures;
  const out = [...figures];
  [out[i], out[j]] = [out[j], out[i]];
  return out;
}

/** How many panels of `cfg` show each image. */
function countPanels(cfg: FigureConfig | null): Map<string, number> {
  const counts = new Map<string, number>();
  if (!cfg || !Array.isArray(cfg.panels)) return counts;
  for (const row of cfg.panels) {
    if (!Array.isArray(row)) continue;
    for (const p of row) {
      const name = p?.image_name;
      if (name) counts.set(name, (counts.get(name) ?? 0) + 1);
    }
  }
  return counts;
}

/** Where every image is used: panels of the active figure, and which other
 *  figures show it. Images used nowhere are absent from the map. */
export function imageUsage(
  figures: readonly FigureTab[],
  activeFigureId: string,
  activeConfig: FigureConfig | null,
): Map<string, ImageUsage> {
  const usage = new Map<string, ImageUsage>();
  const entry = (name: string) => {
    let u = usage.get(name);
    if (!u) { u = { inCurrent: 0, others: [] }; usage.set(name, u); }
    return u;
  };
  for (const [name, n] of countPanels(activeConfig)) entry(name).inCurrent = n;
  for (const f of figures) {
    if (f.id === activeFigureId) continue;
    for (const [name, count] of countPanels(f.config)) {
      entry(name).others.push({ id: f.id, name: f.name, count });
    }
  }
  return usage;
}

/** Blank `imageName` in every PARKED figure's panels (the active figure is
 *  the store's config and is handled there). No match → same array. */
export function stripImageFromFigures(figures: FigureTab[], imageName: string): FigureTab[] {
  let changed = false;
  const out = figures.map((f) => {
    const cfg = f.config;
    if (!cfg || !cfg.panels.some((row) => row.some((p) => p?.image_name === imageName))) return f;
    changed = true;
    const panels = cfg.panels.map((row) =>
      row.map((p) => (p?.image_name === imageName ? { ...p, image_name: "" } : p)));
    return { ...f, config: { ...cfg, panels } };
  });
  return changed ? out : figures;
}
