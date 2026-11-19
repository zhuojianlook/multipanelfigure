/* ──────────────────────────────────────────────────────────
   intensityPlotOptions — the options behind the "Plot intensities"
   node: data model, normalisation, R code generation, and how the
   node's code relates to those options. Pure (no React, no I/O).

   The options live on the R node as `data.intensityPlot`, stored
   sparsely: only values that differ from the defaults are kept.
   `data.code` is generated from them; `data.intensityPlotHash` records
   what was generated, which is how a later hand edit is recognised.
   ────────────────────────────────────────────────────────── */

import type { FluorIntensityConfig } from "../components/dialogs/IntensityPickerDialog";
import type { NodeData } from "./analysisTypes";
import { INTENSITY_PLOT_RUNTIME_R } from "./intensityPlotRuntime";

// ── Types ───────────────────────────────────────────────────

export type IntensityGraphId =
  | "channel_intensity" | "channel_intensity_norm" | "channel_intensity_per_image"
  | "channel_intensity_norm_per_image" | "channel_intensity_fc" | "channel_intensity_norm_fc"
  | "control_intensity";
export type CompId = "whole_image" | "whole_cell" | "nucleus" | "cytoplasm";

export interface IntensityGraphOptions {
  chart: "bar" | "box" | "violin" | "mean";
  /** Bar and mean charts only. */
  error: "sd" | "sem" | "ci" | "none";
  points: boolean;
  test: "tukey" | "welch" | "wilcox" | "none";
  compare: "all" | "ref";
  /** null = first included group. Also the fold-change baseline. */
  refGroup: string | null;
  sig: "stars" | "p";
  hideNs: boolean;
  layout: "panels" | "dodged";
  /** shared = one y axis per channel, across its compartments. */
  yScale: "shared" | "independent" | "same";
  yZero: boolean;
  yLog: boolean;
  /** okabe = colour-blind-safe (Okabe-Ito). */
  palette: "set2" | "okabe" | "dark" | "grey";
  /** null = automatic text. */
  title: string | null;
  yLabel: string | null;
  legend: "bottom" | "right" | "top" | "none";
  xAngle: 0 | 30 | 45 | 90;
  fontSize: number;
  /** Inches per panel, null = automatic. */
  panelW: number | null;
  panelH: number | null;
  exclude: { groups: string[]; channels: string[]; compartments: CompId[] };
  order: { groups: string[]; channels: string[]; compartments: CompId[] };
}
export type IntensitySettingKey = keyof IntensityGraphOptions;
export type IntensityGraphOverride = { enabled?: boolean } & Partial<IntensityGraphOptions>;
export type ResolvedIntensityGraph = IntensityGraphOptions & { enabled: boolean };

export interface IntensityPlotConfig {
  version: 1;
  /** Sparse: a missing graph or key means the default. */
  graphs: Partial<Record<IntensityGraphId, IntensityGraphOverride>>;
}

export type IntensityPlotCodeState = "none" | "generated" | "legacy" | "edited";

export interface IntensityGraphInfo {
  id: IntensityGraphId;
  label: string;
  description: string;
  section: "descriptive" | "perImage" | "foldChange" | "control";
}

// ── Graph catalogue ─────────────────────────────────────────

export const INTENSITY_GRAPHS: readonly IntensityGraphInfo[] = [
  { id: "channel_intensity", section: "descriptive", label: "Channel intensities",
    description: "Every measured value, by group and channel" },
  { id: "channel_intensity_norm", section: "descriptive", label: "Normalised intensities",
    description: "Divided by each image's control channel mean" },
  { id: "channel_intensity_per_image", section: "perImage", label: "Per-image means",
    description: "One value per image, so n is the number of images" },
  { id: "channel_intensity_norm_per_image", section: "perImage", label: "Normalised per-image means",
    description: "Per-image means on the normalised scale" },
  { id: "channel_intensity_fc", section: "foldChange", label: "Fold change",
    description: "log2 ratio of each group to the baseline group" },
  { id: "channel_intensity_norm_fc", section: "foldChange", label: "Normalised fold change",
    description: "Fold change on the normalised scale" },
  { id: "control_intensity", section: "control", label: "Control channel",
    description: "Checks that the control stain matches across groups" },
];
export const INTENSITY_GRAPH_IDS: readonly IntensityGraphId[] = INTENSITY_GRAPHS.map((g) => g.id);

export const INTENSITY_COMPARTMENTS: readonly CompId[] = ["whole_image", "whole_cell", "nucleus", "cytoplasm"];
export const INTENSITY_COMPARTMENT_LABELS: Record<CompId, string> = {
  whole_image: "Whole image", whole_cell: "Whole cell", nucleus: "Nuclear", cytoplasm: "Cytoplasm",
};

export const isFoldChangeGraph = (id: IntensityGraphId): boolean =>
  id === "channel_intensity_fc" || id === "channel_intensity_norm_fc";

const CHOICES = {
  chart: ["bar", "box", "violin", "mean"],
  error: ["sd", "sem", "ci", "none"],
  test: ["tukey", "welch", "wilcox", "none"],
  compare: ["all", "ref"],
  sig: ["stars", "p"],
  layout: ["panels", "dodged"],
  yScale: ["shared", "independent", "same"],
  palette: ["set2", "okabe", "dark", "grey"],
  legend: ["bottom", "right", "top", "none"],
} as const;
export const INTENSITY_OPTION_CHOICES = CHOICES;
export const INTENSITY_X_ANGLES: readonly IntensityGraphOptions["xAngle"][] = [0, 30, 45, 90];
export const INTENSITY_FONT_RANGE = { min: 6, max: 24 } as const;
export const INTENSITY_PANEL_RANGE = { min: 0.5, max: 12 } as const;

const ALL_SETTINGS: readonly IntensitySettingKey[] = [
  "chart", "error", "points", "test", "compare", "refGroup", "sig", "hideNs", "layout", "yScale",
  "yZero", "yLog", "palette", "title", "yLabel", "legend", "xAngle", "fontSize", "panelW", "panelH",
  "exclude", "order",
];
const FOLD_CHANGE_SETTINGS: readonly IntensitySettingKey[] = [
  "refGroup", "layout", "yScale", "palette", "title", "yLabel", "legend", "xAngle", "fontSize",
  "panelW", "panelH", "exclude", "order",
];

/** The settings a graph type offers; the generator emits nothing else. */
export const graphSettings = (id: IntensityGraphId): readonly IntensitySettingKey[] =>
  isFoldChangeGraph(id) ? FOLD_CHANGE_SETTINGS : ALL_SETTINGS;
export const graphHasSetting = (id: IntensityGraphId, key: IntensitySettingKey): boolean =>
  graphSettings(id).includes(key);

/** Why a setting currently does nothing given the graph's other settings,
 *  or null when it is live. The dialog disables the control and shows this. */
export function settingInactiveReason(o: IntensityGraphOptions, key: IntensitySettingKey): string | null {
  switch (key) {
    case "error":
      return o.chart === "box" || o.chart === "violin" ? "Box and violin plots show their own spread" : null;
    case "compare": case "sig": case "hideNs":
      return o.test === "none" ? "No statistical test is selected" : null;
    case "yLog":
      return o.chart === "bar" ? "A bar starts at 0, which a log axis cannot show" : null;
    case "yZero":
      return o.chart === "bar" ? "Bars always start at 0" : o.yLog ? "A log axis has no 0" : null;
    default:
      return null;
  }
}

/** Whether the y axis starts at 0, given the other settings: bars always do,
 *  a log axis never does. The dialog's checkbox shows this. */
export const yAxisStartsAtZero = (o: Pick<IntensityGraphOptions, "chart" | "yZero" | "yLog">): boolean =>
  o.chart === "bar" || (o.yZero && !o.yLog);

// ── Defaults and normalisation ──────────────────────────────

export function defaultIntensityGraphOptions(): IntensityGraphOptions {
  return {
    chart: "bar", error: "sd", points: true, test: "tukey", compare: "all", refGroup: null,
    sig: "stars", hideNs: false, layout: "panels", yScale: "shared", yZero: true, yLog: false,
    palette: "set2", title: null, yLabel: null, legend: "bottom", xAngle: 30, fontSize: 11,
    panelW: null, panelH: null,
    exclude: { groups: [], channels: [], compartments: [] },
    order: { groups: [], channels: [], compartments: [] },
  };
}
export const defaultIntensityPlotConfig = (): IntensityPlotConfig => ({ version: 1, graphs: {} });

const MAX_TEXT = 300;
const MAX_LIST = 500;
const isObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

function cleanText(v: unknown): string | null | undefined {
  if (v === null) return null;
  if (typeof v !== "string") return undefined;
  return v.trim().slice(0, MAX_TEXT) || null;
}
function cleanNames(v: unknown): string[] {
  if (!Array.isArray(v)) return [];
  const out = new Set<string>();
  for (const x of v) if (typeof x === "string" && x && x.length <= MAX_TEXT && out.size < MAX_LIST) out.add(x);
  return [...out];
}
function cleanComps(v: unknown): CompId[] {
  return cleanNames(v).filter((x): x is CompId => (INTENSITY_COMPARTMENTS as readonly string[]).includes(x));
}
function cleanLists(v: unknown): IntensityGraphOptions["order"] | undefined {
  if (!isObject(v)) return undefined;
  return { groups: cleanNames(v.groups), channels: cleanNames(v.channels), compartments: cleanComps(v.compartments) };
}
function cleanSize(v: unknown): number | null | undefined {
  if (v === null) return null;
  if (typeof v !== "number" || !Number.isFinite(v)) return undefined;
  return Math.round(Math.min(INTENSITY_PANEL_RANGE.max, Math.max(INTENSITY_PANEL_RANGE.min, v)) * 100) / 100;
}
const oneOf = <T extends string>(list: readonly T[], v: unknown): T | undefined =>
  list.find((x) => x === v);
const bool = (v: unknown): boolean | undefined => (typeof v === "boolean" ? v : undefined);

/** Valid value for a setting, or undefined when `v` is not usable. */
function cleanSetting(key: IntensitySettingKey, v: unknown): unknown {
  switch (key) {
    case "chart": case "error": case "test": case "compare": case "sig":
    case "layout": case "yScale": case "palette": case "legend":
      return oneOf<string>(CHOICES[key], v);
    case "points": case "hideNs": case "yZero": case "yLog":
      return bool(v);
    case "refGroup": case "title": case "yLabel":
      return cleanText(v);
    case "xAngle":
      return INTENSITY_X_ANGLES.find((x) => x === v);
    case "fontSize":
      return typeof v === "number" && Number.isFinite(v)
        ? Math.round(Math.min(INTENSITY_FONT_RANGE.max, Math.max(INTENSITY_FONT_RANGE.min, v)) * 2) / 2
        : undefined;
    case "panelW": case "panelH":
      return cleanSize(v);
    case "exclude": case "order":
      return cleanLists(v);
  }
}

const sameValue = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

/** True for a config written by a newer build: keep it untouched, and treat
 *  the node as hand-edited so that this build never rewrites it. */
export const isFutureIntensityPlotConfig = (raw: unknown): boolean =>
  isObject(raw) && typeof raw.version === "number" && raw.version > 1;

/** Canonical sparse form: unknown keys, bad values and values equal to the
 *  default are dropped. Returns null for anything that is not a config this
 *  build can read (callers then use the defaults, see also
 *  isFutureIntensityPlotConfig). */
export function normalizeIntensityPlotConfig(raw: unknown): IntensityPlotConfig | null {
  if (!isObject(raw) || isFutureIntensityPlotConfig(raw)) return null;
  const graphs: IntensityPlotConfig["graphs"] = {};
  const defaults = defaultIntensityGraphOptions() as unknown as Record<string, unknown>;
  const rawGraphs = isObject(raw.graphs) ? raw.graphs : {};
  for (const id of INTENSITY_GRAPH_IDS) {
    const src = rawGraphs[id];
    if (!isObject(src)) continue;
    const out: Record<string, unknown> = {};
    if (src.enabled === false) out.enabled = false;
    for (const key of ALL_SETTINGS) {
      if (!(key in src)) continue;
      const v = cleanSetting(key, src[key]);
      if (v !== undefined && !sameValue(v, defaults[key])) out[key] = v;
    }
    // A bar has no position on a log axis.
    const chart = (out.chart as string | undefined) ?? defaults.chart;
    if (out.yLog === true && chart === "bar") delete out.yLog;
    if (Object.keys(out).length) graphs[id] = out as IntensityGraphOverride;
  }
  return { version: 1, graphs };
}

/** Defaults merged with the graph's overrides. Always a fresh object. */
export function resolveGraph(cfg: IntensityPlotConfig, id: IntensityGraphId): ResolvedIntensityGraph {
  const { enabled, ...over } = cfg.graphs[id] ?? {};
  const o = { ...defaultIntensityGraphOptions(), ...over };
  return {
    ...o,
    yLog: o.yLog && o.chart !== "bar",
    exclude: {
      groups: [...o.exclude.groups], channels: [...o.exclude.channels], compartments: [...o.exclude.compartments],
    },
    order: { groups: [...o.order.groups], channels: [...o.order.channels], compartments: [...o.order.compartments] },
    enabled: enabled !== false,
  };
}

/** Which values a graph shows, in what order: stored order first, then values
 *  not yet seen (so new groups appear and are included), minus exclusions.
 *  The R runtime's .ipo_pick is the same rule. */
export function resolveOrder<T extends string>(
  available: readonly T[], order: readonly T[], exclude: readonly T[] = [],
): T[] {
  const present = new Set(available);
  const known = new Set(order);
  const skip = new Set(exclude);
  const seq = [...new Set(order)].filter((x) => present.has(x));
  for (const x of new Set(available)) if (!known.has(x)) seq.push(x);
  return seq.filter((x) => !skip.has(x));
}

// ── Editing helpers for the dialog ──────────────────────────

/** A new config with `patch` applied to one graph (canonicalised). */
export function updateIntensityGraph(
  cfg: IntensityPlotConfig, id: IntensityGraphId, patch: IntensityGraphOverride,
): IntensityPlotConfig {
  const next = { version: 1, graphs: { ...cfg.graphs, [id]: { ...cfg.graphs[id], ...patch } } };
  return normalizeIntensityPlotConfig(next) ?? defaultIntensityPlotConfig();
}

/** Back to the defaults for one graph, keeping whether it is enabled. */
export function resetIntensityGraph(cfg: IntensityPlotConfig, id: IntensityGraphId): IntensityPlotConfig {
  const graphs = { ...cfg.graphs };
  delete graphs[id];
  if (cfg.graphs[id]?.enabled === false) graphs[id] = { enabled: false };
  return { version: 1, graphs };
}

/** Copy one graph's settings to the others. Only settings both graph types
 *  offer are copied; never `enabled`, `title` or `yLabel`. The control graph
 *  lists control channels and the others analysis channels, so between the
 *  two the channel exclusions and order stay as they are. */
export function applyIntensityGraphToAll(cfg: IntensityPlotConfig, from: IntensityGraphId): IntensityPlotConfig {
  const src = resolveGraph(cfg, from);
  const srcMap = src as unknown as Record<string, unknown>;
  const graphs: Record<string, IntensityGraphOverride> = { ...cfg.graphs };
  const isControl = (id: IntensityGraphId) => id === "control_intensity";
  for (const id of INTENSITY_GRAPH_IDS) {
    if (id === from) continue;
    const next: Record<string, unknown> = { ...cfg.graphs[id] };
    const own = resolveGraph(cfg, id);
    const sameChannels = isControl(id) === isControl(from);
    for (const key of graphSettings(id)) {
      if (key === "title" || key === "yLabel" || !graphHasSetting(from, key)) continue;
      if (!sameChannels && (key === "exclude" || key === "order")) {
        next[key] = { ...src[key], channels: own[key].channels };
      } else {
        next[key] = srcMap[key];
      }
    }
    graphs[id] = next as IntensityGraphOverride;
  }
  return normalizeIntensityPlotConfig({ version: 1, graphs }) ?? defaultIntensityPlotConfig();
}

const CHART_WORDS: Record<IntensityGraphOptions["chart"], string> = {
  bar: "bars", box: "box plots", violin: "violins", mean: "means",
};
const ERROR_WORDS: Record<IntensityGraphOptions["error"], string> = { sd: "SD", sem: "SEM", ci: "95% CI", none: "" };
const TEST_WORDS: Record<IntensityGraphOptions["test"], string> = {
  tukey: "Tukey", welch: "Welch", wilcox: "Wilcoxon", none: "no test",
};

/** Inspector caption, e.g. "5 of 7 graphs · bars + SD · Tukey". */
export function intensityPlotSummary(cfg: IntensityPlotConfig): string {
  const on = INTENSITY_GRAPH_IDS.filter((id) => resolveGraph(cfg, id).enabled);
  const parts = [`${on.length} of ${INTENSITY_GRAPH_IDS.length} graphs`];
  const stat = on.filter((id) => !isFoldChangeGraph(id)).map((id) => resolveGraph(cfg, id));
  if (stat.length) {
    const charts = new Set(stat.map((o) => {
      const err = o.chart === "bar" || o.chart === "mean" ? ERROR_WORDS[o.error] : "";
      return err ? `${CHART_WORDS[o.chart]} + ${err}` : CHART_WORDS[o.chart];
    }));
    const tests = new Set(stat.map((o) => TEST_WORDS[o.test]));
    parts.push(charts.size === 1 ? [...charts][0] : "mixed charts");
    parts.push(tests.size === 1 ? [...tests][0] : "mixed tests");
  }
  return parts.join(" · ");
}

// ── R text ──────────────────────────────────────────────────

/** An R string literal, pure printable ASCII. This is the only way user text
 *  enters the generated code. Quotes are always escaped and `mpfig_` is
 *  broken up, so text cannot look like an output declaration to the regexes
 *  that read the node's code. */
export function rStr(s: string): string {
  let out = "";
  for (const ch of s) {
    const cp = ch.codePointAt(0) as number;
    if (cp >= 0xd800 && cp <= 0xdfff) continue;
    if (ch === "\\") out += "\\\\";
    else if (ch === '"') out += '\\"';
    else if (ch === "'") out += "\\'";
    else if (ch === "\n") out += "\\n";
    else if (ch === "\t") out += "\\t";
    else if (cp < 0x20 || cp === 0x7f) continue;
    else if (cp > 0x7e) out += `\\U{${cp.toString(16)}}`;
    else out += ch;
  }
  return `"${out.replace(/mpfig_/g, "mpfig\\U{5f}")}"`;
}

const rVec = (xs: readonly string[]) => (xs.length ? `c(${xs.map(rStr).join(", ")})` : "character(0)");
const rBool = (b: boolean) => (b ? "TRUE" : "FALSE");
const rNum = (n: number | null) => (n !== null && Number.isFinite(n) ? String(n) : "NULL");
const rNullable = (s: string | null) => (s === null ? "NULL" : rStr(s));
/** Emits the whitelist's own spelling, never the stored string. */
const rChoice = <T extends string>(list: readonly T[], v: T) => `"${list.find((x) => x === v) ?? list[0]}"`;

/** FNV-1a over the code, ignoring line-ending style and trailing whitespace. */
export function hashCode(code: string): string {
  const s = code.replace(/\r\n/g, "\n").trimEnd();
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    h = Math.imul(h ^ (c & 0xff), 0x01000193);
    h = Math.imul(h ^ (c >>> 8), 0x01000193);
  }
  return `${s.length}:${(h >>> 0).toString(16).padStart(8, "0")}`;
}

// ── Code generation ─────────────────────────────────────────

export const INTENSITY_PLOT_GEN = 1;
const NAME_LINE = "# @name: Plot intensities (groups × channels × compartments)";
const MARKER = "# @mpf-intensity-plot";
const MARKER_LINE = `${MARKER} gen=${INTENSITY_PLOT_GEN} — generated from Plot options (double-click the node). Hand edits are kept until options are applied again.`;

function rGraphOptions(id: IntensityGraphId, o: ResolvedIntensityGraph): string {
  const f: string[] = [];
  if (!isFoldChangeGraph(id)) {
    const spreadShown = o.chart === "bar" || o.chart === "mean";
    f.push(
      `chart = ${rChoice(CHOICES.chart, o.chart)}`,
      `error = ${rChoice(CHOICES.error, spreadShown ? o.error : "none")}`,
      `points = ${rBool(o.points)}`,
      `test = ${rChoice(CHOICES.test, o.test)}`,
      `compare = ${rChoice(CHOICES.compare, o.compare)}`,
      `sig = ${rChoice(CHOICES.sig, o.sig)}`,
      `hide_ns = ${rBool(o.hideNs)}`,
      `y_zero = ${rBool(yAxisStartsAtZero(o))}`,
      `y_log = ${rBool(o.yLog)}`,
    );
  }
  f.push(
    `ref_group = ${rNullable(o.refGroup)}`,
    `layout = ${rChoice(CHOICES.layout, o.layout)}`,
    `y_scale = ${rChoice(CHOICES.yScale, o.yScale)}`,
    `palette = ${rChoice(CHOICES.palette, o.palette)}`,
    `title = ${rNullable(o.title)}`,
    `y_label = ${rNullable(o.yLabel)}`,
    `legend = ${rChoice(CHOICES.legend, o.legend)}`,
    `x_angle = ${rNum(o.xAngle)}`,
    `font_size = ${rNum(o.fontSize)}`,
    `panel_w = ${rNum(o.panelW)}`,
    `panel_h = ${rNum(o.panelH)}`,
    `exclude_groups = ${rVec(o.exclude.groups)}`,
    `exclude_channels = ${rVec(o.exclude.channels)}`,
    `exclude_compartments = ${rVec(o.exclude.compartments)}`,
    `order_groups = ${rVec(o.order.groups)}`,
    `order_channels = ${rVec(o.order.channels)}`,
    `order_compartments = ${rVec(o.order.compartments)}`,
  );
  return `  ${id} = list(${f.join(", ")})`;
}

const emitBlock = (prepare: string, fileExpr: string) =>
  `.g <- ${prepare}; if (!is.null(.g)) { mpfig_plot(${fileExpr}, width = .g$w, height = .g$h, res = 300); print(.ipo_plot(.g)) }`;

const byCodeUnit = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);
const WARNING_FILE = "channel_intensity_warning.png";

/** The node's complete R code. Graph blocks run in file-name order because
 *  the sidecar returns plots sorted by file name while it indexes its plot
 *  snapshots by call order, and a plot's re-render relies on the two
 *  agreeing. Only enabled graphs appear, so a disabled graph declares no
 *  output. The warning and empty-state panels use computed file names for
 *  the same reason: they are not declared outputs. */
export function generateIntensityPlotR(cfg: IntensityPlotConfig): string {
  const clean = normalizeIntensityPlotConfig(cfg) ?? defaultIntensityPlotConfig();
  const enabled = INTENSITY_GRAPH_IDS.filter((id) => resolveGraph(clean, id).enabled);
  const options = enabled.map((id) => rGraphOptions(id, resolveGraph(clean, id)));
  const blocks = enabled.map((id) => ({
    file: `${id}.png`,
    code: emitBlock(`.ipo_prepare(${rStr(id)})`, rStr(`${id}.png`)),
  }));
  if (enabled.some(isFoldChangeGraph)) {
    blocks.push({
      file: WARNING_FILE,
      code: emitBlock(".ipo_warning()", 'paste0("channel_intensity_", "warning.png")'),
    });
  }
  blocks.sort((a, b) => byCodeUnit(a.file, b.file));
  return [
    NAME_LINE,
    MARKER_LINE,
    options.length ? `.ipo <- list(\n${options.join(",\n")}\n)` : ".ipo <- list()",
    "",
    INTENSITY_PLOT_RUNTIME_R.trimEnd(),
    "",
    "# -- Graphs, in file-name order ---------------------------------------",
    ...blocks.map((b) => b.code),
    emitBlock(".ipo_empty()", 'paste0("channel_", "intensity.png")'),
    "",
  ].join("\n");
}

export const DEFAULT_INTENSITY_PLOT_CODE = generateIntensityPlotR(defaultIntensityPlotConfig());
export const DEFAULT_INTENSITY_PLOT_HASH = hashCode(DEFAULT_INTENSITY_PLOT_CODE);

// ── How the node's code relates to the options ──────────────

/** hashCode of every version of the old hand-written template in git history
 *  (the INTENSITY_PLOT_R literal in AnalysisNodeGraph.tsx, by the commit that
 *  introduced that version). An unmodified copy of one of them is replaced by
 *  generated code on the node's next run. */
export const LEGACY_HASHES: ReadonlySet<string> = new Set<string>([
  "1737:13a5ef3f", // af98cd8
  "4082:2baff286", // e9b20db
  "7731:bbf7f1df", // 3cd879b
  "10786:7f8f9441", // b482d5c
  "14891:a39a7e6f", // fdcd203
  "19541:69b1cf34", // 3f4fcbc
  "20390:ac78053b", // dd338bb
  "21292:bc4570db", // 77f2ed7, current through 3a720e4
]);

type IntensityNodeFields = Pick<NodeData, "kind" | "code" | "intensityPlot" | "intensityPlotHash">;

/** Generated code, or the old template, possibly edited: it has the marker
 *  line, a graph block, or the old template's renderer. */
const isPlotFamilyCode = (code: string): boolean =>
  code.includes(MARKER) || code.includes(".ipo_prepare(")
  || code.includes(".render_panel <- function(d, title, flag_warning");

export function intensityPlotState(data: IntensityNodeFields): IntensityPlotCodeState {
  if (data.kind !== "r") return "none";
  const code = typeof data.code === "string" ? data.code : "";
  if (isFutureIntensityPlotConfig(data.intensityPlot)) return "edited";
  const hash = hashCode(code);
  if (data.intensityPlotHash) {
    if (hash === data.intensityPlotHash) return "generated";
    // Other code put on the node (a preset, a paste) is not an edit of the plot.
    return isPlotFamilyCode(code) ? "edited" : "none";
  }
  if (LEGACY_HASHES.has(hash)) return "legacy";
  return code.startsWith("# @name: Plot intensities") && isPlotFamilyCode(code) ? "edited" : "none";
}

/** The node offers the options dialog: a "Plot intensities" node whose
 *  options were not written by a newer build (those stay read-only). */
export const intensityPlotEditable = (data: IntensityNodeFields): boolean =>
  intensityPlotState(data) !== "none" && !isFutureIntensityPlotConfig(data.intensityPlot);

// ── Node updates (used by AnalysisNodeGraph) ────────────────

type IntensityNodeEdit = Pick<NodeData,
  "kind" | "code" | "intensityPlot" | "intensityPlotHash" | "intensityPlotBackup" | "status" | "currentPreset">;

/** A new "Plot intensities" node's option fields: the defaults and their code. */
export const defaultIntensityPlotFields = () => ({
  intensityPlot: defaultIntensityPlotConfig(),
  code: DEFAULT_INTENSITY_PLOT_CODE,
  intensityPlotHash: DEFAULT_INTENSITY_PLOT_HASH,
});

/** The node's data after Apply in the options dialog: the config, the code
 *  generated from it and that code's hash. Code edited by hand goes to the
 *  one-slot backup first. A changed code marks an `ok` node `stale`, the same
 *  rule as typing in the code editor. Returns `data` itself when nothing
 *  changes. */
export function applyIntensityPlotOptions<T extends IntensityNodeEdit>(
  data: T, cfg: IntensityPlotConfig, savedAt: string,
): T {
  const intensityPlot = normalizeIntensityPlotConfig(cfg) ?? defaultIntensityPlotConfig();
  const code = generateIntensityPlotR(intensityPlot);
  const intensityPlotHash = hashCode(code);
  const sameCode = code === data.code;
  if (sameCode && intensityPlotHash === data.intensityPlotHash && sameValue(intensityPlot, data.intensityPlot)) return data;
  // Options of a disabled graph change the config but not the code.
  if (sameCode) return { ...data, intensityPlot, intensityPlotHash };
  const backup = intensityPlotState(data) === "edited" && typeof data.code === "string"
    ? { intensityPlotBackup: { code: data.code, savedAt } } : {};
  return {
    ...data, ...backup, intensityPlot, intensityPlotHash, code, currentPreset: "custom",
    status: data.status === "ok" ? "stale" : data.status,
  };
}

/** The node's data without its plot options, for when other code (a preset)
 *  replaces the generated code: the node is then an ordinary R node. */
export function withoutIntensityPlot<T extends Partial<IntensityNodeEdit>>(data: T): T {
  if (!("intensityPlot" in data || "intensityPlotHash" in data || "intensityPlotBackup" in data)) return data;
  const next: T = { ...data };
  delete next.intensityPlot;
  delete next.intensityPlotHash;
  delete next.intensityPlotBackup;
  return next;
}

/** "Restore previous code": the backup becomes the node's code. The config
 *  and hash stay, so the node reads as edited by hand. When the current code
 *  is itself a hand edit it takes the backup slot, so neither is lost. */
export function restoreIntensityPlotBackup<T extends IntensityNodeEdit>(data: T, savedAt: string): T {
  const backup = data.intensityPlotBackup;
  if (!backup || typeof backup.code !== "string") return data;
  if (backup.code === data.code) {
    const same: T = { ...data };
    delete same.intensityPlotBackup;
    return same;
  }
  const next: T = {
    ...data, code: backup.code, currentPreset: "custom",
    status: data.status === "ok" ? "stale" : data.status,
  };
  if (intensityPlotState(data) === "edited" && typeof data.code === "string") {
    next.intensityPlotBackup = { code: data.code, savedAt };
  } else {
    delete next.intensityPlotBackup;
  }
  return next;
}

// ── What the upstream data offers ───────────────────────────

export interface IntensityPlotContext {
  groups: string[];
  channels: string[];
  controlChannels: string[];
  compartments: CompId[];
  mode: "simple" | "cellpose" | null;
  hasNorm: boolean;
  source: "run" | "picker" | "unknown";
}

const MAX_CSV_ROWS = 300_000;
const CSV_COLUMNS = ["group", "channel", "is_control", "compartment", "mean_intensity_norm"] as const;
type CsvColumn = (typeof CSV_COLUMNS)[number];

/** Minimal RFC 4180 reader: calls `onRow` with the wanted columns only, so a
 *  large per-cell table costs no per-field allocation for the other columns.
 *  Returns the header, or null for an empty text. */
function scanCsv(
  text: string, wanted: readonly string[], maxRows: number,
  onRow: (values: Record<string, string>) => void,
): string[] | null {
  const n = text.length;
  let i = text.charCodeAt(0) === 0xfeff ? 1 : 0;
  let header: string[] | null = null;
  let picks = new Map<number, string>();
  let rows = 0;
  const readField = (keep: boolean): string => {
    if (text[i] === '"') {
      let val = "";
      i++;
      for (;;) {
        const q = text.indexOf('"', i);
        if (q < 0) { if (keep) val += text.slice(i); i = n; return val; }
        if (keep) val += text.slice(i, q);
        if (text[q + 1] === '"') { if (keep) val += '"'; i = q + 2; continue; }
        i = q + 1;
        // Anything between the closing quote and the delimiter is dropped.
        while (i < n && text[i] !== "," && text[i] !== "\n" && text[i] !== "\r") i++;
        return val;
      }
    }
    const start = i;
    while (i < n) {
      const c = text.charCodeAt(i);
      if (c === 44 || c === 10 || c === 13) break;
      i++;
    }
    return keep ? text.slice(start, i) : "";
  };
  while (i < n && rows < maxRows) {
    const record: Record<string, string> = {};
    const names: string[] = [];
    let col = 0;
    let first = "";
    for (;;) {
      const key = header ? picks.get(col) : undefined;
      const val = readField(!header || key !== undefined || col === 0);
      if (col === 0) first = val;
      if (!header) names.push(val.trim());
      else if (key !== undefined) record[key] = val;
      col++;
      if (i < n && text[i] === ",") { i++; continue; }
      break;
    }
    if (text[i] === "\r") i++;
    if (text[i] === "\n") i++;
    if (col === 1 && first === "") continue;
    if (!header) {
      header = names;
      picks = new Map(names.flatMap((name, idx): [number, string][] => (wanted.includes(name) ? [[idx, name]] : [])));
      continue;
    }
    rows++;
    onRow(record);
  }
  return header;
}

interface IntensityFacts {
  groups: string[]; channels: string[]; controlChannels: string[]; compartments: CompId[];
  mode: "simple" | "cellpose"; hasNorm: boolean;
}

function factsFromTable(csv: string): IntensityFacts | null {
  const groups = new Set<string>();
  const channels = new Set<string>();
  const controls = new Set<string>();
  const comps = new Set<string>();
  let hasNorm = false;
  let rows = 0;
  const header = scanCsv(csv, CSV_COLUMNS, MAX_CSV_ROWS, (r: Partial<Record<CsvColumn, string>>) => {
    rows++;
    // Same fallbacks as the R runtime's reading of the table.
    groups.add(r.group || (r.group === undefined ? "all" : "(unassigned)"));
    const ch = r.channel || "Channel";
    if (/^(T|TRUE|true|True)$/.test((r.is_control ?? "").trim())) controls.add(ch);
    else channels.add(ch);
    comps.add(r.compartment || "whole_image");
    const norm = r.mean_intensity_norm;
    if (!hasNorm && norm && norm.trim() && Number.isFinite(Number(norm))) hasNorm = true;
  });
  if (!header || !rows || !(header.includes("channel") || header.includes("group"))) return null;
  const compartments = INTENSITY_COMPARTMENTS.filter((c) => comps.has(c));
  return {
    groups: [...groups], channels: [...channels], controlChannels: [...controls], compartments, hasNorm,
    mode: compartments.some((c) => c !== "whole_image") ? "cellpose" : "simple",
  };
}

/** What the picker will produce when run: same naming rules as its Python. */
function factsFromPicker(cfg: FluorIntensityConfig): IntensityFacts {
  const keys = ["r", "g", "b"] as const;
  const nameOf = (k: (typeof keys)[number]) => cfg.channels?.[k] || `Channel ${k.toUpperCase()}`;
  const simple = cfg.mode === "simple";
  const measured = keys.filter((k) => !simple || cfg.thresholds?.[k]?.enabled !== false).map(nameOf);
  const control = cfg.controlChannel ? nameOf(cfg.controlChannel) : null;
  const groups = (cfg.groups ?? []).map((g) => (g.name ?? "").trim()).filter(Boolean);
  const split = !simple && !!cfg.cellpose?.measureCompartments && !!cfg.cellpose?.nucleiChannel;
  const compartments: CompId[] = simple ? ["whole_image"] : split ? ["whole_cell", "nucleus", "cytoplasm"] : ["whole_cell"];
  return {
    groups: [...new Set(groups)],
    channels: [...new Set(measured.filter((c) => c !== control))],
    controlChannels: control && measured.includes(control) ? [control] : [],
    compartments, mode: simple ? "simple" : "cellpose", hasNorm: control !== null,
  };
}

const union = <T,>(first: readonly T[], then: readonly T[]) => [...new Set([...first, ...then])];

/** Groups, channels and compartments the dialog should list.
 *  `tableCsv` is the upstream node's newest table output, `pickerCfg` its
 *  Intensity picker config. `pickerChanged` says the picker was reconfigured
 *  since that table was made; its names then come first. */
export function deriveIntensityPlotContext(input: {
  pickerCfg?: FluorIntensityConfig | null;
  tableCsv?: string | null;
  pickerChanged?: boolean;
}): IntensityPlotContext {
  const table = input.tableCsv ? factsFromTable(input.tableCsv) : null;
  const picker = input.pickerCfg ? factsFromPicker(input.pickerCfg) : null;
  if (table && picker && input.pickerChanged) {
    return {
      groups: union(picker.groups, table.groups),
      channels: union(picker.channels, table.channels),
      controlChannels: union(picker.controlChannels, table.controlChannels),
      compartments: INTENSITY_COMPARTMENTS.filter((c) => picker.compartments.includes(c) || table.compartments.includes(c)),
      mode: picker.mode, hasNorm: picker.hasNorm || table.hasNorm, source: "picker",
    };
  }
  if (table) return { ...table, source: "run" };
  if (picker) return { ...picker, source: "picker" };
  return { groups: [], channels: [], controlChannels: [], compartments: [], mode: null, hasNorm: false, source: "unknown" };
}

/** Why a graph cannot be drawn from this data, or null. With unknown data
 *  nothing is ruled out. */
export function intensityGraphUnavailableReason(id: IntensityGraphId, ctx: IntensityPlotContext): string | null {
  if (ctx.source === "unknown") return null;
  const noControl = "No control channel set in the Intensity picker";
  const isNorm = id.includes("_norm");
  if (id === "control_intensity") return ctx.controlChannels.length ? null : noControl;
  if (isNorm && !ctx.hasNorm) return noControl;
  if (id.endsWith("_per_image") && ctx.mode === "simple") return "Simple mode already measures one value per image";
  if (isFoldChangeGraph(id) && ctx.groups.filter((g) => g !== "(unassigned)").length < 2) return "Needs at least 2 groups";
  return null;
}
