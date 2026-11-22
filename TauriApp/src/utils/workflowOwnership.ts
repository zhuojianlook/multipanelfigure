/* ──────────────────────────────────────────────────────────
   workflowOwnership — which .mpf documents an Analysis workflow
   belongs to, and the key rewrites that follow a document being
   opened, closed, saved under a new path or located again.
   Pure: no React, no store.

   Identities
     • docUid ("du_<random>") — durable per document, stored in its
       workflows/manifest.json. Ownership is decided in uids only.
     • key base — the prefix a document's sources and measurement
       rows are keyed under: figKey(doc) ("p:<path>" / "d:<docId>")
       while it is open, "u:<docUid>" while it is closed. Opening,
       closing and Save As are pure prefix re-keys between bases
       (rekeyKey / rekeyTabs) — the raw inset key never changes.

   Ownership (deriveOwnership)
     participants  uids of every figure-bound source and measurement
                   table, in attach order (node order, then sources,
                   then tables);
     home          one stored uid, auto-maintained: forced to the
                   only participant; with 2+ participants kept if it
                   is one of them, else the first; with none, left
                   alone (set at creation / by the upload prompt);
     owners        participants, or [home] when there are none;
     shared        2+ owners — such a workflow is saved into every
                   owner's .mpf as graph + config only.
   ────────────────────────────────────────────────────────── */

import type { Edge } from "@xyflow/react";
import type { InsetSource, MeasurementRef, NodeData, WorkflowTab } from "./analysisTypes";
import { figKey, measHandleId, measIndexFromHandle, measRefsOf, nsKey, rawKeyOf } from "./analysisKeys";
import { DEFAULT_FIGURE_ID } from "./figures";
import { pivotAnalysisKey, pivotRawKey } from "./pivotGrid";

/** An OPEN document as ownership sees it: session id + path (its key base)
 *  and its durable uid. */
export interface DocRef {
  id: string;
  uid: string;
  name?: string;
  path?: string | null;
}

/** A CLOSED document some workflow still names, from a manifest's `docs`. */
export interface KnownDoc {
  name?: string;
  /** Absolute path it was last saved at. */
  path?: string;
  /** Its session doc id when that manifest was written (runtime-key remap). */
  sessionId?: string;
  /** Where "Open missing" looks for it, best first (see locateCandidates). */
  candidates?: string[];
}

export interface OwnershipCtx {
  /** Open documents. */
  docs: readonly DocRef[];
  /** Closed documents by uid — lets a source still keyed by a closed
   *  document's path resolve to its uid. */
  known?: Readonly<Record<string, KnownDoc>>;
}

export interface Ownership {
  /** Uids of the documents whose figure data the workflow reads. */
  participants: string[];
  /** The maintained home uid (may differ from the tab's stored `home`). */
  home?: string;
  /** Who saves it: participants, or [home] when there are none. */
  owners: string[];
  shared: boolean;
}

export const CLOSED_KEY_PREFIX = "u:";
/** Names "+" gives new tabs ("Workflow 3") — part of the pristine test. */
export const DEFAULT_WORKFLOW_NAME_RE = /^Workflow \d+$/;

/** A fresh durable document uid. */
export function newDocUid(): string {
  return `du_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
}

/** The key base of an OPEN document: figKey without a figure id. */
export function docKeyBase(doc: { id: string; path?: string | null }): string {
  return figKey(doc);
}

/** The key base of a CLOSED document. */
export function closedKeyBase(uid: string): string {
  return `${CLOSED_KEY_PREFIX}${uid}`;
}

/** figKey of one figure of a CLOSED document — mirrors figKey's rule that the
 *  project's first figure carries no suffix. */
export function closedFigKey(uid: string, figureId?: string): string {
  const base = closedKeyBase(uid);
  return figureId && figureId !== DEFAULT_FIGURE_ID ? `${base}#${figureId}` : base;
}

/** The uid a closed-document figKey ("u:<uid>" / "u:<uid>#<fig>") names.
 *  Uids never contain "#", so cutting there is safe (unlike a path). */
export function closedKeyUid(fk: string | undefined): string | undefined {
  if (!fk || !fk.startsWith(CLOSED_KEY_PREFIX)) return undefined;
  const rest = fk.slice(CLOSED_KEY_PREFIX.length);
  const i = rest.indexOf("#");
  return (i >= 0 ? rest.slice(0, i) : rest) || undefined;
}

/** Open documents joined with their uids (documents without one are left out). */
export function docRefs(
  openDocs: ReadonlyArray<{ id: string; name?: string; path?: string | null }>,
  uids: Readonly<Record<string, string>>,
): DocRef[] {
  const out: DocRef[] = [];
  for (const d of openDocs) {
    const uid = uids[d.id];
    if (uid) out.push({ id: d.id, uid, name: d.name, path: d.path });
  }
  return out;
}

/** Uid of the document a figure-bound source belongs to; undefined for an
 *  upload, or a source whose document can't be identified (e.g. an Untitled
 *  document from an earlier session). */
export function sourceDocUid(
  s: Pick<InsetSource, "figKey" | "figureId" | "mpfId" | "name">,
  ctx: OwnershipCtx,
): string | undefined {
  if (s.name) return undefined;
  const fk = s.figKey;
  if (fk) {
    const closed = closedKeyUid(fk);
    if (closed) return closed;
    // Two open tabs can hold the same path (an Untitled saved over an open
    // file): prefer the source's own tab, so the binding never depends on
    // which of them comes first in the (user-reorderable) tab strip.
    const matches = ctx.docs.filter((d) => figKey(d, s.figureId) === fk);
    if (matches.length) return (matches.find((d) => d.id === s.mpfId) ?? matches[0]).uid;
    if (ctx.known) {
      for (const [uid, k] of Object.entries(ctx.known)) {
        if (k.path && figKey({ id: "", path: k.path }, s.figureId) === fk) return uid;
      }
    }
    return undefined;
  }
  // Legacy source (no figKey): bound by its session doc id, if still open.
  return s.mpfId ? ctx.docs.find((d) => d.id === s.mpfId)?.uid : undefined;
}

/** Uid of the document a measurement table came from. */
export function measRefDocUid(ref: MeasurementRef, ctx: OwnershipCtx): string | undefined {
  if (ref.docUid) return ref.docUid;
  return ref.mpfId ? ctx.docs.find((d) => d.id === ref.mpfId)?.uid : undefined;
}

/** Uids of the documents a workflow reads figure data from, in attach order. */
export function participantsOf(wf: Pick<WorkflowTab, "nodes">, ctx: OwnershipCtx): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  const add = (u: string | undefined) => {
    if (u && !seen.has(u)) { seen.add(u); out.push(u); }
  };
  for (const n of wf.nodes) {
    if (n.data.kind !== "source") continue;
    for (const s of n.data.sources || []) add(sourceDocUid(s, ctx));
    for (const r of measRefsOf(n.data)) add(measRefDocUid(r, ctx));
  }
  return out;
}

export function deriveOwnership(wf: Pick<WorkflowTab, "nodes" | "home">, ctx: OwnershipCtx): Ownership {
  const participants = participantsOf(wf, ctx);
  let home = wf.home;
  if (participants.length === 1) home = participants[0];
  else if (participants.length >= 2 && (!home || !participants.includes(home))) home = participants[0];
  const owners = participants.length ? participants : (home ? [home] : []);
  return { participants, home, owners, shared: owners.length >= 2 };
}

/** Owners of the workflow whose document is not open. */
export function missingOwners(wf: Pick<WorkflowTab, "nodes" | "home">, ctx: OwnershipCtx): string[] {
  const open = new Set(ctx.docs.map((d) => d.uid));
  return deriveOwnership(wf, ctx).owners.filter((u) => !open.has(u));
}

/** Locked (greyed out, read-only): some owner's document is not open. */
export function isLocked(wf: Pick<WorkflowTab, "nodes" | "home">, ctx: OwnershipCtx): boolean {
  return missingOwners(wf, ctx).length > 0;
}

/** A tab nobody has touched: default name, one source node holding nothing,
 *  no edges. Never saved and never makes a document dirty. Node position is
 *  ignored — moving the lone empty node isn't work worth keeping. */
export function isPristineTab(wf: Pick<WorkflowTab, "name" | "nodes" | "edges">): boolean {
  if (!DEFAULT_WORKFLOW_NAME_RE.test((wf.name || "").trim())) return false;
  if (wf.edges.length !== 0 || wf.nodes.length !== 1) return false;
  const d = wf.nodes[0].data;
  return d.kind === "source"
    && !(d.sources && d.sources.length)
    && measRefsOf(d).length === 0
    && (!d.label || d.label === "Source");
}

// ── Re-keying ───────────────────────────────────────────────

/** "#<figureId>" followed by the end, "::" or "|" — figure ids are
 *  "fig-<base36>", never a path fragment like "#b.mpf". */
const FIG_SUFFIX_RE = /^#[A-Za-z0-9_-]+(?=::|\||$)/;
/** A measurement-row tail: exactly "|panel|ann|reading" (measRowKey). */
const MEAS_TAIL_RE = /^\|[^|]*\|[^|]*\|[^|]*$/;

/** Move one key from `oldBase` to `newBase`.
 *
 *  Keys the bases prefix: the base itself (a figKey), `base#fig` (a figKey of
 *  another figure tab), `base[#fig]::raw` (a source key) and
 *  `base[#fig]|panel|ann|reading` (a measurement row). Matching is by prefix
 *  plus the shape of what follows — never by splitting on "#", because a path
 *  may contain one: with base "p:/x/a", the key "p:/x/a#b.mpf::r0c0" belongs to
 *  the document "/x/a#b.mpf" and must not move. */
export function rekeyKey(key: string, oldBase: string, newBase: string): string {
  if (!key || !oldBase || oldBase === newBase || !key.startsWith(oldBase)) return key;
  let rest = key.slice(oldBase.length);
  let fig = "";
  if (rest.startsWith("#")) {
    const m = FIG_SUFFIX_RE.exec(rest);
    if (!m) return key;
    fig = m[0];
    rest = rest.slice(fig.length);
  }
  if (rest === "") return newBase + fig;
  // Exactly one "::" — a second one means the base was only a prefix of a
  // path that itself contains "::".
  if (rest.startsWith("::")) return rest.indexOf("::", 2) < 0 ? newBase + fig + rest : key;
  if (rest.startsWith("|")) return MEAS_TAIL_RE.test(rest) ? newBase + fig + rest : key;
  return key;
}

/** `rec` with every key moved by `fn`; the SAME object when nothing moved. */
export function mapRecordKeys<T>(rec: Readonly<Record<string, T>>, fn: (k: string) => string): Record<string, T> {
  let changed = false;
  const out: Record<string, T> = {};
  for (const [k, v] of Object.entries(rec)) {
    const nk = fn(k);
    if (nk !== k) changed = true;
    out[nk] = v;
  }
  return changed ? out : (rec as Record<string, T>);
}

/** A record keyed by source keys / measurement rows (source names, group
 *  labels, sourceEditStore.revs …) moved from one base to another. */
export function rekeyRecord<T>(rec: Readonly<Record<string, T>>, oldBase: string, newBase: string): Record<string, T> {
  return mapRecordKeys(rec, (k) => rekeyKey(k, oldBase, newBase));
}

function mapArr<T>(arr: readonly T[], fn: (x: T) => T): T[] {
  let changed = false;
  const out = arr.map((x) => {
    const y = fn(x);
    if (y !== x) changed = true;
    return y;
  });
  return changed ? out : (arr as T[]);
}

/** Apply `fn` to every node's data; unchanged nodes, tabs and the array keep
 *  their identity (the graph's setNodes bail-out depends on it). */
export function mapTabNodes(
  tabs: readonly WorkflowTab[],
  fn: (data: NodeData, tab: WorkflowTab) => NodeData,
): WorkflowTab[] {
  return mapArr(tabs as WorkflowTab[], (t) => {
    const nodes = mapArr(t.nodes, (n) => {
      const d = fn(n.data, t);
      return d === n.data ? n : { ...n, data: d };
    });
    return nodes === t.nodes ? t : { ...t, nodes };
  });
}

/** Image ids the interactive pickers key their per-image config by: band
 *  picker `roi.images` keys, intensity / Cellpose `groups[].images`. They are
 *  runtime keys (runtimeKeyFor) or, for the Cellpose fallback, source keys. */
export function mapPickerKeys(data: NodeData, fn: (k: string) => string): NodeData {
  let out = data;
  const roi = data.roi;
  if (roi && roi.images) {
    const images = mapRecordKeys(roi.images, fn);
    if (images !== roi.images) out = { ...out, roi: { ...roi, images } };
  }
  const mapGroups = <G extends { images: string[] }>(gs: G[] | undefined): G[] | undefined => {
    if (!gs) return gs;
    return mapArr(gs, (g) => {
      const images = mapArr(g.images || [], fn);
      return images === g.images ? g : { ...g, images };
    });
  };
  if (data.intensity) {
    const groups = mapGroups(data.intensity.groups);
    if (groups !== data.intensity.groups) out = { ...out, intensity: { ...data.intensity, groups: groups! } };
  }
  if (data.cellposeSeg) {
    const groups = mapGroups(data.cellposeSeg.groups);
    if (groups !== data.cellposeSeg.groups) out = { ...out, cellposeSeg: { ...data.cellposeSeg, groups } };
  }
  return out;
}

function rekeyNodeData(data: NodeData, fn: (k: string) => string): NodeData {
  let out = data;
  if (data.sources && data.sources.length) {
    const sources = mapArr(data.sources, (s) => {
      const key = fn(s.key);
      const fk = s.figKey ? fn(s.figKey) : s.figKey;
      return key === s.key && fk === s.figKey ? s : { ...s, key, figKey: fk };
    });
    if (sources !== data.sources) out = { ...out, sources };
  }
  if (data.ranWith) {
    const ranWith = mapRecordKeys(data.ranWith, fn);
    if (ranWith !== data.ranWith) out = { ...out, ranWith };
  }
  return mapPickerKeys(out, fn);
}

/** Every figKey-prefixed key in the tabs moved from `oldBase` to `newBase`:
 *  source keys and figKeys, `ranWith`, and picker image ids. Returns `tabs`
 *  itself when nothing moved. */
export function rekeyTabs(tabs: readonly WorkflowTab[], oldBase: string, newBase: string): WorkflowTab[] {
  if (oldBase === newBase) return tabs as WorkflowTab[];
  const fn = (k: string) => rekeyKey(k, oldBase, newBase);
  return mapTabNodes(tabs, (d) => rekeyNodeData(d, fn));
}

// ── Pivot (the planner swapped a figure's rows and columns) ─────

/** The figure that was pivoted: its figKey, and the session doc id and
 *  figure tab its runtime keys (runtimeKeyFor) are minted from. */
export interface PivotKeyCtx {
  fk: string;
  docId: string;
  figureId: string;
}

/** A picker image id (a runtime key, see runtimeKeyFor — or a source key, the
 *  Cellpose fallback) for the panel's new position:
 *    1. `${fk}::raw`                         → pivotAnalysisKey;
 *    2. `${docId}-${figureId}__raw`, and `${docId}__raw` for the project's
 *       first figure                         → the prefix + the moved raw key;
 *    3. `${figureId}__raw`                   → moved only when `ids.fig` has raw;
 *    4. a bare raw key                       → moved only when `ids.bare` has it.
 *  Session doc ids are unique, so (2) names this document for sure. (3) was
 *  minted while ITS document was the active one — possibly another document
 *  with the same figure id (every document has "fig-1"; a Save As copy has
 *  them all). A bare key is the figure that was on screen when it was minted —
 *  possibly another one. Hence the explicit lists (pivotTabs builds them). */
export function pivotRuntimeKey(k: string, ctx: PivotKeyCtx, ids: PivotIdSets): string {
  const viaFk = pivotAnalysisKey(k, ctx.fk);
  if (viaFk !== k) return viaFk;
  const prefixes = [`${ctx.docId}-${ctx.figureId}__`];
  if (ctx.figureId === DEFAULT_FIGURE_ID) prefixes.push(`${ctx.docId}__`);
  for (const p of prefixes) {
    if (!k.startsWith(p)) continue;
    const raw = k.slice(p.length);
    const moved = pivotRawKey(raw);
    if (moved !== raw) return p + moved;
  }
  const fp = `${ctx.figureId}__`;
  if (k.startsWith(fp)) {
    const raw = k.slice(fp.length);
    return ids.fig.has(raw) ? fp + pivotRawKey(raw) : k;
  }
  return ids.bare.has(k) ? pivotRawKey(k) : k;
}

/** The raw key of a source of figure `fk` that a pivot moves (its key is
 *  `${fk}::raw`); null for uploads and other figures' sources. */
function pivotedSourceRaw(s: InsetSource, fk: string): string | null {
  if (s.name || s.figKey !== fk) return null;
  const raw = rawKeyOf(s);
  return s.key === nsKey(fk, raw) ? raw : null;
}

/** The raw keys a tab's picker ids may name the pivoted figure by. `bare`
 *  for a bare id; `fig` for a `${figureId}__raw` id. */
export interface PivotIdSets {
  bare: ReadonlySet<string>;
  fig: ReadonlySet<string>;
}

/** The raw keys of the tab's sources of `ctx.fk`, each with its pivoted twin,
 *  unless either of the two is also the raw key of a source that the id could
 *  equally name: for `bare`, a source of any other figure; for `fig`, a source
 *  of another document's figure with the same figure id. Such ids can't be told
 *  apart and are left alone. Closed under the pivot, so the remap is a
 *  permutation (no two ids land on one) and a second pivot undoes it. Null
 *  when the tab reads nothing of `ctx.fk`: its pickers are left alone. */
function pivotIdSets(t: WorkflowTab, ctx: PivotKeyCtx): PivotIdSets | null {
  const mine = new Set<string>();
  const others = new Set<string>();
  const sameFigId = new Set<string>();
  for (const n of t.nodes) {
    if (n.data.kind !== "source") continue;
    for (const s of n.data.sources || []) {
      if (s.name) continue;
      const raw = pivotedSourceRaw(s, ctx.fk);
      if (raw !== null) { mine.add(raw); continue; }
      const r = rawKeyOf(s);
      others.add(r);
      if ((s.figureId || DEFAULT_FIGURE_ID) === ctx.figureId) sameFigId.add(r);
    }
  }
  if (!mine.size) return null;
  const closed = (blocked: ReadonlySet<string>) => {
    const ok = new Set<string>();
    for (const k of mine) {
      const twin = pivotRawKey(k);
      if (blocked.has(k) || blocked.has(twin)) continue;
      ok.add(k);
      ok.add(twin);
    }
    return ok;
  };
  return { bare: closed(others), fig: closed(sameFigId) };
}

/** Figure `ctx.fk` was pivoted (panel (r, c) is now at (c, r)): every key the
 *  tabs hold for one of its panels follows the panel — its sources (key,
 *  rawKey, runKey, row/col), `ranWith`, and the pickers' image ids. Other
 *  figures, other documents and uploads are untouched. Applying it twice
 *  gives the tabs back. Returns `tabs` itself when nothing moved. */
export function pivotTabs(tabs: readonly WorkflowTab[], ctx: PivotKeyCtx): WorkflowTab[] {
  const { fk } = ctx;
  if (!fk) return tabs as WorkflowTab[];
  const idsByTab = new Map<WorkflowTab, PivotIdSets | null>();
  const idsOf = (t: WorkflowTab) => {
    if (!idsByTab.has(t)) idsByTab.set(t, pivotIdSets(t, ctx));
    return idsByTab.get(t)!;
  };
  const moveKey = (k: string) => pivotAnalysisKey(k, fk);
  return mapTabNodes(tabs, (d, t) => {
    let out = d;
    if (d.sources && d.sources.length) {
      const sources = mapArr(d.sources, (s) => {
        const raw = pivotedSourceRaw(s, fk);
        if (raw === null) return s;
        const moved = pivotRawKey(raw);
        if (moved === raw) return s;                       // a diagonal panel stays put
        const rk = s.runKey;
        const runKey = rk === raw ? moved
          : rk && rk.endsWith(`__${raw}`) ? rk.slice(0, rk.length - raw.length) + moved
            : rk;
        return { ...s, key: nsKey(fk, moved), rawKey: moved, runKey, row: s.col, col: s.row };
      });
      if (sources !== d.sources) out = { ...out, sources };
    }
    if (d.ranWith) {
      const ranWith = mapRecordKeys(d.ranWith, moveKey);
      if (ranWith !== d.ranWith) out = { ...out, ranWith };
    }
    const ids = idsOf(t);
    return ids ? mapPickerKeys(out, (k) => pivotRuntimeKey(k, ctx, ids)) : out;
  });
}

function withMeasRefs(data: NodeData, refs: MeasurementRef[]): NodeData {
  return { ...data, measurementsRef: refs[0], measurementRefs: refs.length ? refs : undefined };
}

/** Bind the measurement tables of document `uid` to session doc `docId`.
 *  Tables already carrying `docUid === uid` get `mpfId = docId` (pass null to
 *  leave mpfId alone, e.g. on close). Tables with no docUid whose mpfId is
 *  `legacyDocId` are back-filled with the uid. */
export function bindMeasurementRefs(
  tabs: readonly WorkflowTab[], uid: string, docId: string | null, legacyDocId?: string,
): WorkflowTab[] {
  return mapTabNodes(tabs, (d) => {
    const refs = measRefsOf(d);
    if (!refs.length) return d;
    const next = mapArr(refs, (r) => {
      const mine = r.docUid === uid || (!r.docUid && legacyDocId != null && r.mpfId === legacyDocId);
      if (!mine) return r;
      const mpfId = docId ?? r.mpfId;
      return r.docUid === uid && r.mpfId === mpfId ? r : { ...r, docUid: uid, mpfId };
    });
    return next === refs ? d : withMeasRefs(d, next);
  });
}

/** Re-point a source node's edges after some of its sources (kind "image",
 *  handles out_image_<i>) or tables (kind "table", out_table_measurements*)
 *  were removed. `keep` lists the surviving OLD indices in order; an edge off
 *  a removed slot is dropped, one off a surviving slot follows it to its new
 *  index (same id scheme as the graph's own detach). */
function remapHandles(edges: Edge[], nodeId: string, kind: "image" | "table", keep: number[]): Edge[] {
  const map = new Map(keep.map((oldIdx, newIdx) => [oldIdx, newIdx]));
  const out: Edge[] = [];
  const ids = new Set<string>();
  const push = (e: Edge) => {
    if (ids.has(e.id)) return;
    ids.add(e.id);
    out.push(e);
  };
  for (const e of edges) {
    const h = e.sourceHandle || "";
    if (e.source !== nodeId) { push(e); continue; }
    if (kind === "image") {
      if (!h.startsWith("out_image_")) { push(e); continue; }
      const i = parseInt(h.slice("out_image_".length), 10);
      if (!Number.isFinite(i)) { push(e); continue; }
      const ni = map.get(i);
      if (ni == null) continue;
      if (ni === i) { push(e); continue; }
      const nh = `out_image_${ni}`;
      push({ ...e, sourceHandle: nh, id: `e_${nodeId}_${nh}__${e.target}_${e.targetHandle || "in_image"}` });
    } else {
      if (!h.startsWith("out_table_measurements")) { push(e); continue; }
      const i = measIndexFromHandle(h);
      const ni = map.get(i);
      if (ni == null) continue;
      if (ni === i) { push(e); continue; }
      const nh = measHandleId(ni);
      push({ ...e, sourceHandle: nh, id: `e_${nodeId}_${nh}__${e.target}_${e.targetHandle || "in_table"}` });
    }
  }
  return out;
}

/** The workflow without document `uid`'s sources and measurement tables
 *  ("Remove their sources" for a document that is gone for good). Edges off
 *  removed slots are dropped, the rest re-indexed. A `home` pointing at `uid`
 *  moves to the first remaining participant, else `fallbackHome`. Returns
 *  `wf` itself when it held nothing of `uid`. */
export function removeDocSources(wf: WorkflowTab, uid: string, ctx: OwnershipCtx, fallbackHome?: string): WorkflowTab {
  let edges = wf.edges;
  let changed = false;
  const nodes = wf.nodes.map((n) => {
    if (n.data.kind !== "source") return n;
    const srcs = n.data.sources || [];
    const keepSrc = srcs.flatMap((s, i) => (sourceDocUid(s, ctx) === uid ? [] : [i]));
    const refs = measRefsOf(n.data);
    const keepRef = refs.flatMap((r, i) => (measRefDocUid(r, ctx) === uid ? [] : [i]));
    if (keepSrc.length === srcs.length && keepRef.length === refs.length) return n;
    changed = true;
    let data = n.data;
    if (keepSrc.length !== srcs.length) {
      data = { ...data, sources: keepSrc.map((i) => srcs[i]) };
      edges = remapHandles(edges, n.id, "image", keepSrc);
    }
    if (keepRef.length !== refs.length) {
      data = withMeasRefs(data, keepRef.map((i) => refs[i]));
      edges = remapHandles(edges, n.id, "table", keepRef);
    }
    return { ...n, data };
  });
  if (!changed && wf.home !== uid) return wf;
  const next: WorkflowTab = { ...wf, nodes: changed ? nodes : wf.nodes, edges };
  if (wf.home === uid) {
    next.home = participantsOf(next, ctx)[0] ?? fallbackHome;
    next.homeConfirmed = undefined;
  }
  return next;
}

/** Point everything the workflow holds of CLOSED document `fromUid` at
 *  `toUid` (the user located a file with a different uid and confirmed it is
 *  the same document): `u:` keys, measurement tables and `home`. */
export function rebindDoc(wf: WorkflowTab, fromUid: string, toUid: string): WorkflowTab {
  if (!fromUid || fromUid === toUid) return wf;
  let [t] = rekeyTabs([wf], closedKeyBase(fromUid), closedKeyBase(toUid));
  [t] = mapTabNodes([t], (d) => {
    const refs = measRefsOf(d);
    if (!refs.length) return d;
    const next = mapArr(refs, (r) => (r.docUid === fromUid ? { ...r, docUid: toUid } : r));
    return next === refs ? d : withMeasRefs(d, next);
  });
  return t.home === fromUid ? { ...t, home: toUid } : t;
}
