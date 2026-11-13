/* ──────────────────────────────────────────────────────────
   analysisStale — which Analysis results an Edit Panel change
   invalidates. Pure (no React, no store).

   Analysis never snapshots pixels: the sidecar extracts a source's
   image from the live panel each time a node runs. So when a panel's
   crop / rotation / adjustments change after a pipeline ran, its
   results describe pixels the figure no longer has.

   The model:
     • every analysable source has a SIGNATURE — the panel fields the
       sidecar's extraction applies (+ the inset rect / area outline);
     • an edit that changes a signature bumps that source's REVISION
       (store/sourceEditStore);
     • each process node records, when it runs, the revision of every
       source upstream of it (NodeData.ranWith);
     • a node with results is STALE while some upstream source's
       revision is newer than the one it ran with. Re-running it
       records the new revisions, which clears it — per node, so a
       source feeding several graphs clears in each one separately.

   Source keys mirror the sidecar's _collect_analysis_insets:
     r{r}c{c}_panel   panel.include_in_analysis && image_name
     r{r}c{c}_area{i} areas[i].include_in_analysis
     r{r}c{c}_i{i}    _panel_zoom_insets(panel)[i].include_in_analysis
   ────────────────────────────────────────────────────────── */

import type { PanelInfo } from "../api/types";

/** Panel fields the sidecar applies when it extracts a source's pixels
 *  (process_panel + _get_panel_image), each with its model default
 *  (python-sidecar/models.py PanelInfo) so a missing field and its default
 *  compare equal. Labels, scale bars, symbols, lines, inset styling and
 *  video playback ranges are stripped before extraction — not listed. */
const PANEL_PIXEL_DEFAULTS: Readonly<Record<string, unknown>> = {
  image_name: "", frame: 0,
  rotation: 0, flip_horizontal: false, flip_vertical: false,
  crop: null, crop_image: true, aspect_ratio_str: "", crop_offset_x: 0, crop_offset_y: 0,
  final_resize: false, final_width: 400, final_height: 300,
  brightness: 1, contrast: 1, exposure: 0, gamma: 1, hue: 0, saturation: 1, vibrance: 0,
  color_temperature: 0, tint: 0, sharpen: 0, blur: 0, denoise: 0,
  highlights: 0, shadows: 0, midtones: 0,
  input_black_r: 0, input_white_r: 255, input_black_g: 0, input_white_g: 255,
  input_black_b: 0, input_white_b: 255,
  grayscale: false, invert: false, pseudocolor: "",
};
export const PANEL_PIXEL_FIELDS = Object.keys(PANEL_PIXEL_DEFAULTS);

/** Zoom-inset fields that choose its pixels (api_server _extract_inset_image). */
const INSET_PIXEL_FIELDS = ["x", "y", "width", "height", "rotation", "zoom_factor"] as const;
/** Area fields that choose its pixels (api_server _extract_area_image). */
const AREA_PIXEL_FIELDS = ["shape", "points"] as const;

type Obj = Record<string, unknown>;

function panelField(p: Obj, f: string): unknown {
  const v = p[f];
  if (v !== undefined && v !== null) return v;
  // Old projects stored one black/white point for all channels; the dialog
  // migrates it to the per-channel fields on open (same rule here, so
  // opening and saving an old panel untouched is not a change).
  if (f.startsWith("input_black_") && p.input_black != null) return p.input_black;
  if (f.startsWith("input_white_") && p.input_white != null) return p.input_white;
  return PANEL_PIXEL_DEFAULTS[f];
}

function pick(o: Obj | null | undefined, fields: readonly string[]): unknown[] {
  return fields.map((f) => (o && o[f] !== undefined ? o[f] : null));
}

/** The sidecar's _panel_zoom_insets: [] unless add_zoom_inset; the array,
 *  else the legacy single inset. */
function panelZoomInsets(p: Obj): Obj[] {
  if (!p.add_zoom_inset) return [];
  const arr = Array.isArray(p.zoom_insets) ? (p.zoom_insets as Obj[]) : [];
  if (arr.length) return arr;
  return p.zoom_inset ? [p.zoom_inset as Obj] : [];
}

/** Raw source key → pixel signature, for every source this panel exposes
 *  to Analysis (unflagged regions are absent). */
export function analysisSourceSigs(panel: PanelInfo | null | undefined, row: number, col: number): Record<string, string> {
  const out: Record<string, string> = {};
  if (!panel) return out;
  const p = panel as unknown as Obj;
  const base = PANEL_PIXEL_FIELDS.map((f) => panelField(p, f));
  const at = `r${row}c${col}`;
  if (p.include_in_analysis && p.image_name) out[`${at}_panel`] = JSON.stringify(base);
  const areas = Array.isArray(p.areas) ? (p.areas as Obj[]) : [];
  areas.forEach((a, i) => {
    if (a && a.include_in_analysis) out[`${at}_area${i}`] = JSON.stringify([base, pick(a, AREA_PIXEL_FIELDS)]);
  });
  panelZoomInsets(p).forEach((zi, i) => {
    if (zi && zi.include_in_analysis) out[`${at}_i${i}`] = JSON.stringify([base, pick(zi, INSET_PIXEL_FIELDS)]);
  });
  return out;
}

/** Raw keys of sources that exist both before and after the edit and whose
 *  pixels changed. A source flagged or unflagged by the edit isn't "changed":
 *  a new one has no results yet, and a removed one has no pixels to compare. */
export function changedSourceRawKeys(
  before: PanelInfo | null | undefined, after: PanelInfo | null | undefined, row: number, col: number,
): string[] {
  const a = analysisSourceSigs(before, row, col);
  const b = analysisSourceSigs(after, row, col);
  return Object.keys(a).filter((k) => k in b && a[k] !== b[k]);
}

/** "R1C2 inset 1" for "…::r0c1_i0" (namespaced or raw), for messages. */
export function describeSourceKey(key: string): string {
  const ns = key.lastIndexOf("::");
  const raw = ns >= 0 ? key.slice(ns + 2) : key;
  const m = raw.match(/^r(\d+)c(\d+)_(panel|i(\d+)|area(\d+))$/);
  if (!m) return raw;
  const at = `R${Number(m[1]) + 1}C${Number(m[2]) + 1}`;
  if (m[3] === "panel") return `${at} panel`;
  if (m[4] !== undefined) return `${at} inset ${Number(m[4]) + 1}`;
  return `${at} area ${Number(m[5]) + 1}`;
}

// ── Graph side ──────────────────────────────────────────────

export interface StaleNodeLike {
  id: string;
  data: {
    kind: string;
    sources?: Array<{ key: string }>;
    outputs?: unknown[];
    /** Source revision each upstream source had when this node last ran. */
    ranWith?: Record<string, number>;
  };
}
export interface StaleEdgeLike { source: string; target: string; sourceHandle?: string | null }

/** Every source key feeding `nodeId`, directly or through other nodes. One
 *  image edge from a source node carries ALL of that node's sources (the
 *  runner fans out the same way — collectInputs); a measurements edge
 *  carries none. */
export function upstreamSourceKeys(nodeId: string, nodes: readonly StaleNodeLike[], edges: readonly StaleEdgeLike[]): string[] {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const keys = new Set<string>();
  const seen = new Set<string>([nodeId]);
  const queue = [nodeId];
  while (queue.length) {
    const cur = queue.shift()!;
    for (const e of edges) {
      if (e.target !== cur) continue;
      const up = byId.get(e.source);
      if (!up) continue;
      if (up.data.kind === "source") {
        if ((e.sourceHandle || "").startsWith("out_image_")) {
          for (const s of up.data.sources || []) keys.add(s.key);
        }
      } else if (!seen.has(up.id)) {
        seen.add(up.id);
        queue.push(up.id);
      }
    }
  }
  return [...keys];
}

/** The `ranWith` record a node stores when a run starts. */
export function ranWithFor(nodeId: string, nodes: readonly StaleNodeLike[], edges: readonly StaleEdgeLike[],
  revs: Readonly<Record<string, number>>): Record<string, number> {
  return Object.fromEntries(upstreamSourceKeys(nodeId, nodes, edges).map((k) => [k, revs[k] ?? 0]));
}

/** Nodes whose results predate an edit to a source they read, and those
 *  sources (to badge on their source nodes). */
export function computeStale(nodes: readonly StaleNodeLike[], edges: readonly StaleEdgeLike[],
  revs: Readonly<Record<string, number>>): { nodeIds: Set<string>; sourceKeys: Set<string> } {
  const nodeIds = new Set<string>();
  const sourceKeys = new Set<string>();
  for (const n of nodes) {
    if (n.data.kind === "source" || !(n.data.outputs && n.data.outputs.length)) continue;
    for (const k of upstreamSourceKeys(n.id, nodes, edges)) {
      if ((revs[k] ?? 0) > (n.data.ranWith?.[k] ?? 0)) {
        nodeIds.add(n.id);
        sourceKeys.add(k);
      }
    }
  }
  return { nodeIds, sourceKeys };
}

/** Source keys that feed a node with results, in any workflow tab — the
 *  sources whose edit is worth warning about. */
export function sourcesWithResults(tabs: ReadonlyArray<{ nodes: readonly StaleNodeLike[]; edges: readonly StaleEdgeLike[] }>): string[] {
  const keys = new Set<string>();
  for (const t of tabs) {
    for (const n of t.nodes) {
      if (n.data.kind === "source" || !(n.data.outputs && n.data.outputs.length)) continue;
      for (const k of upstreamSourceKeys(n.id, t.nodes, t.edges)) keys.add(k);
    }
  }
  return [...keys].sort();
}
