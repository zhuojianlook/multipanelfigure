/* ──────────────────────────────────────────────────────────
   workflowPersist — Analysis workflows <-> an .mpf's `workflows/`
   section (manifest v1 + blobs + embedded uploads). Pure: no React,
   no store; the analysis workspace store feeds it plain state.

   The stored form is RELOCATABLE: a figure-bound source is stored as
   { doc: <docUid>, figureId, rawKey, … } — never a path — and rebuilt
   on load under whatever key base its document has then (figKey when
   open, "u:<uid>" when closed). Session-scoped and display-only
   fields (runKey, mpfId, thumbnails, ReactFlow transients) are not
   stored.

   What a document's manifest holds (serializeForDoc):
     • every non-pristine workflow it owns (utils/workflowOwnership);
     • single-owner: the latest output per (kind, name) plus pinned
       versions, each payload a blob; results reading an edited (or,
       with figureDiscarded, a discarded) image are flagged `stale`,
       and a node whose code changed since its run keeps status "stale";
     • shared: graph + config only — no outputs, no blobs, idle;
     • the uploads of workflows whose `home` it is (the sidecar embeds
       their original files).
   ────────────────────────────────────────────────────────── */

import type { Edge, Node } from "@xyflow/react";
import type {
  RestoredUpload, WorkflowBlob, WorkflowBlobKind, WorkflowsManifest, WorkflowsPayload,
} from "../api/types";
import type { InsetSource, MeasurementRef, NodeData, NodeOutput, WorkflowTab } from "./analysisTypes";
import { figKey, measRefsOf, nsKey, rawKeyOf } from "./analysisKeys";
import { computeStale, upstreamSourceKeys } from "./analysisStale";
import { renameMetaUploads } from "./analysisMeta";
import {
  closedFigKey, deriveOwnership, isPristineTab, mapPickerKeys, mapRecordKeys, mapTabNodes,
  measRefDocUid, sourceDocUid,
  type DocRef, type KnownDoc, type Ownership, type OwnershipCtx,
} from "./workflowOwnership";

export const WORKFLOWS_FORMAT = "mpf-workflows" as const;
export const WORKFLOWS_VERSION = 1;
/** consoleOut is kept up to this many characters. */
export const CONSOLE_CAP = 20_000;
/** A manifest keeps at most this many (newest) tombstones. */
export const TOMBSTONE_CAP = 200;
/** What the sidecar accepts as a blob id. */
export const BLOB_ID_RE = /^[A-Za-z0-9_-]{1,128}$/;

// ── Stored (manifest) shapes ────────────────────────────────

/** A source as stored. Figure-bound: `doc` (+ figureId, rawKey); upload:
 *  `name`. A figure-bound source whose document has no uid keeps its binding
 *  verbatim in `key` / `figKey` and restores unresolved, as it was. */
export type StoredSource =
  Omit<InsetSource, "key" | "runKey" | "mpfId" | "figKey" | "thumbnail" | "parent_thumbnail" | "figure_thumbnail">
  & { doc?: string; key?: string; figKey?: string };

export interface StoredMeasurementRef {
  docUid?: string;
  mpf: string;
  label: string;
  csv: string;
}

/** An output as stored: its payload (and R script) live in blobs. */
export type StoredOutput = Omit<NodeOutput, "payload" | "rCode" | "blobId"> & { blob: string; rCodeBlob?: string };

export interface StoredNodeData {
  label: string;
  kind: NodeData["kind"];
  status?: NodeData["status"];
  sources?: StoredSource[];
  measurementRefs?: StoredMeasurementRef[];
  outputs?: StoredOutput[];
  /** Results predate an edit to (or a discard of) an image they read. */
  stale?: true;
  /** code, currentPreset, interactive, roi, intensity, cellposeSeg, error,
   *  consoleOut and any field a newer build added. */
  [key: string]: unknown;
}

export interface StoredNode {
  id: string;
  type?: string;
  position: { x: number; y: number };
  data: StoredNodeData;
  [key: string]: unknown;
}

export type StoredEdge = { id: string; source: string; target: string; [key: string]: unknown };

export interface StoredWorkflow {
  id: string;
  name: string;
  updatedAt: number;
  home?: string;
  homeConfirmed?: boolean;
  owners: string[];
  shared: boolean;
  nodes: StoredNode[];
  edges: StoredEdge[];
}

export interface StoredDocEntry {
  name?: string;
  /** Absolute path (other documents only — the holder is wherever it is). */
  path?: string;
  /** Path relative to the holder's folder (other documents only). */
  rel?: string;
  /** Session doc id when written — remaps runtime keys on load. */
  sessionId?: string;
}

export interface WorkflowsManifestV1 extends WorkflowsManifest {
  docUid: string;
  docs: Record<string, StoredDocEntry>;
  workflows: StoredWorkflow[];
  tombstones?: Record<string, number>;
  meta?: Record<string, unknown>;
}

/** What serializeForDoc returns — api.writeWorkflows' payload, typed. */
export interface DocWorkflowsPayload extends WorkflowsPayload {
  manifest: WorkflowsManifestV1;
}

// ── Input state ─────────────────────────────────────────────

export interface PersistState extends OwnershipCtx {
  tabs: readonly WorkflowTab[];
  /** sourceEditStore.revs — which results an Edit Panel change made stale. */
  revs?: Readonly<Record<string, number>>;
  /** Per document uid: workflow id → deletion time (ms). */
  tombstones?: Readonly<Record<string, Readonly<Record<string, number>>>>;
  /** Per document uid: the manifest's `meta` (source names, measurement
   *  group labels, source folders — utils/analysisMeta), written as given and
   *  signed as given (docSig): pass the same form to both sides of a compare. */
  meta?: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
  /** Written as `savedWith`. */
  appVersion?: string;
}

export interface SerializeOptions {
  /** This document's figure edits are being thrown away (Don't save /
   *  Discard & close): results reading its images are stored stale. */
  figureDiscarded?: boolean;
}

// ── Hashing ─────────────────────────────────────────────────

/** cyrb53 — a fast 53-bit string hash (not cryptographic). */
function cyrb53(str: string, seed = 0): number {
  let h1 = 0xdeadbeef ^ seed;
  let h2 = 0x41c6ce57 ^ seed;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
  h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
  h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return 4294967296 * (2097151 & h2) + (h1 >>> 0);
}

export function hashString(s: string): string {
  return cyrb53(s).toString(36);
}

const FULL_HASH_LIMIT = 65_536;
/** Identity of a string, cheap even for multi-MB payloads: a full hash up to
 *  64K chars, above that length + a hash of the head, tail and 64 windows
 *  spread through the middle. */
export function stringSig(s: string): string {
  if (s.length <= FULL_HASH_LIMIT) return hashString(s);
  const parts = [s.slice(0, 4096), s.slice(-4096)];
  const step = Math.floor(s.length / 65);
  for (let i = 1; i <= 64; i++) parts.push(s.substr(i * step, 64));
  return `${s.length}:${hashString(parts.join("\u0001"))}`;
}

/** Canonical form for signatures: sorted keys, undefined dropped, long
 *  strings replaced by their stringSig. */
function canon(v: unknown): unknown {
  if (typeof v === "string") return v.length > 256 ? `\u0000${v.length}:${stringSig(v)}` : v;
  if (Array.isArray(v)) return v.map((x) => (x === undefined ? null : canon(x)));
  if (v && typeof v === "object") {
    const o: Record<string, unknown> = {};
    const src = v as Record<string, unknown>;
    for (const k of Object.keys(src).sort()) {
      const x = src[k];
      if (x === undefined || typeof x === "function") continue;
      o[k] = canon(x);
    }
    return o;
  }
  return v;
}

function sigOf(v: unknown): string {
  return hashString(JSON.stringify(canon(v)));
}

/** A fresh blob id — mint one whenever an output payload is produced or
 *  re-rendered (NodeOutput.blobId). */
export function newBlobId(): string {
  return `b_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
}

/** A fresh workflow id (same shape as the graph's own). */
export function newWorkflowId(): string {
  return `wf_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}

// ── Paths (for "Open missing") ──────────────────────────────

interface ParsedPath { root: string; parts: string[]; sep: string; win: boolean }

function parsePath(p: string): ParsedPath {
  const drive = /^[A-Za-z]:[\\/]/.test(p);
  const unc = p.startsWith("\\\\");
  const win = drive || unc;
  const sep = win ? "\\" : "/";
  let root = "";
  let rest = p;
  if (drive) { root = p.slice(0, 2).toUpperCase() + "\\"; rest = p.slice(3); }
  else if (unc) {
    const segs = p.slice(2).split(/[\\/]/);
    root = "\\\\" + segs.slice(0, 2).join("\\") + "\\";
    rest = segs.slice(2).join("\\");
  } else if (p.startsWith("/")) { root = "/"; rest = p.slice(1); }
  const parts = rest.split(win ? /[\\/]/ : /\//).filter((s) => s && s !== ".");
  return { root, parts, sep, win };
}

function formatPath(pp: ParsedPath): string {
  return pp.root + pp.parts.join(pp.sep);
}

export function dirnameOf(p: string): string {
  const pp = parsePath(p);
  return formatPath({ ...pp, parts: pp.parts.slice(0, -1) });
}

export function basenameOf(p: string): string {
  const parts = p.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] || "";
}

/** `to` relative to the folder `fromDir` ("../b/x.mpf"); undefined when
 *  either is not absolute or they sit on different roots. */
export function relativePath(fromDir: string, to: string): string | undefined {
  const a = parsePath(fromDir);
  const b = parsePath(to);
  if (!a.root || !b.root || a.win !== b.win) return undefined;
  const eq = (x: string, y: string) => (a.win ? x.toLowerCase() === y.toLowerCase() : x === y);
  if (!eq(a.root, b.root)) return undefined;
  let i = 0;
  while (i < a.parts.length && i < b.parts.length && eq(a.parts[i], b.parts[i])) i++;
  const up = a.parts.slice(i).map(() => "..");
  const rel = [...up, ...b.parts.slice(i)];
  return rel.length ? rel.join(a.sep) : ".";
}

/** Resolve `rel` (either separator, "." / ".." allowed) against folder `dir`. */
export function joinPath(dir: string, rel: string): string {
  const pp = parsePath(dir);
  const parts = [...pp.parts];
  for (const seg of rel.split(/[\\/]/)) {
    if (!seg || seg === ".") continue;
    if (seg === "..") parts.pop();
    else parts.push(seg);
  }
  return formatPath({ ...pp, parts });
}

/** Where to look for a missing document, best first: its saved absolute
 *  path, then its path relative to the file that named it (the pair moved
 *  together), then the same file name next to that file. */
export function locateCandidates(holderPath: string | null | undefined, ref: { path?: string; rel?: string }): string[] {
  const out: string[] = [];
  const add = (p: string | undefined) => { if (p && !out.includes(p)) out.push(p); };
  add(ref.path);
  if (holderPath) {
    const dir = dirnameOf(holderPath);
    if (ref.rel) add(joinPath(dir, ref.rel));
    if (ref.path) add(joinPath(dir, basenameOf(ref.path)));
  }
  return out;
}

// ── Serialize ───────────────────────────────────────────────

const NODE_TRANSIENTS = ["selected", "dragging", "measured", "resizing", "positionAbsolute", "internals", "handleBounds"];

function versionOf(o: NodeOutput): number {
  const m = /^v(\d+)$/.exec(o.versionLabel || "");
  return m ? Number(m[1]) : 0;
}

/** The outputs worth saving: the newest version of each (kind, name) — what
 *  downstream nodes consume — plus every pinned one. Unpinned comparison
 *  archives (v-1, v-2) are session-only. Order is preserved. */
export function persistedOutputs(outputs: readonly NodeOutput[]): NodeOutput[] {
  const latest = new Map<string, NodeOutput>();
  for (const o of outputs) {
    const k = `${o.kind}/${o.name}`;
    const cur = latest.get(k);
    if (!cur || versionOf(o) > versionOf(cur)) latest.set(k, o);
  }
  const keep = new Set(latest.values());
  return outputs.filter((o) => keep.has(o) || o.pinned);
}

function storeSource(s: InsetSource, ctx: OwnershipCtx): StoredSource {
  const {
    key, runKey: _rk, mpfId: _m, figKey: fk, thumbnail: _t, parent_thumbnail: _pt, figure_thumbnail: _ft, ...rest
  } = s;
  if (s.name) return rest;                                   // upload — key is img:<name>
  const raw = rawKeyOf(s);
  const doc = sourceDocUid(s, ctx);
  if (doc) return { ...rest, rawKey: raw, doc };
  return { ...rest, rawKey: raw, key, ...(fk ? { figKey: fk } : {}) };
}

function storeMeasRef(r: MeasurementRef, ctx: OwnershipCtx): StoredMeasurementRef {
  const docUid = measRefDocUid(r, ctx);
  return { ...(docUid ? { docUid } : {}), mpf: r.mpf, label: r.label, csv: r.csv };
}

interface StoreCtx {
  ctx: PersistState;
  /** Fill blobs (null = signature mode: record payload identities instead). */
  blobs: Record<string, WorkflowBlob> | null;
  /** Uid whose figure is being discarded. */
  discardedUid?: string;
}

function derivedBlobId(wfId: string, nodeId: string, o: NodeOutput): string {
  return `o_${hashString(`${wfId}\u0000${nodeId}\u0000${o.id}`)}${hashString(stringSig(o.payload || ""))}`;
}

function storeOutput(o: NodeOutput, wfId: string, nodeId: string, sc: StoreCtx): StoredOutput {
  const { payload, rCode, blobId, ...rest } = o;
  const kind: WorkflowBlobKind = o.kind === "table" ? "csv" : "png";
  const data = payload || "";
  let id = blobId && BLOB_ID_RE.test(blobId) ? blobId : derivedBlobId(wfId, nodeId, o);
  const out: StoredOutput = { ...rest, blob: id };
  if (sc.blobs) {
    // Two outputs claiming one id with different bytes: give the later its own.
    const base = id.slice(0, 120);
    for (let n = 2; sc.blobs[id] && (sc.blobs[id].kind !== kind || sc.blobs[id].data !== data); n++) id = `${base}_${n}`;
    sc.blobs[id] = { kind, data };
    out.blob = id;
    if (rCode) {
      const rid = `t_${hashString(rCode)}`;
      sc.blobs[rid] = { kind: "txt", data: rCode };
      out.rCodeBlob = rid;
    }
  } else {
    // Signature mode: an in-place re-render without a new blobId still counts.
    (out as Record<string, unknown>).payloadSig = stringSig(data);
    if (rCode) out.rCodeBlob = `t_${hashString(rCode)}`;
  }
  return out;
}

const TRUNCATED = "\n…(truncated)";
/** At most CONSOLE_CAP chars, marker included — so re-capping a restored
 *  console is a no-op (a growing marker would make every file look dirty). */
function capText(s: string): string {
  return s.length > CONSOLE_CAP ? s.slice(0, CONSOLE_CAP - TRUNCATED.length) + TRUNCATED : s;
}

function storeNode(n: Node<NodeData>, wf: WorkflowTab, own: Ownership, staleIds: Set<string>, sc: StoreCtx): StoredNode {
  const nodeRest: Record<string, unknown> = { ...(n as unknown as Record<string, unknown>) };
  for (const k of NODE_TRANSIENTS) delete nodeRest[k];
  delete nodeRest.data;
  const {
    outputs, sources, measurementsRef: _mr, measurementRefs: _mrs, ranWith: _rw, declaredOutputs: _d,
    status, consoleOut, error, ...dataRest
  } = n.data;
  const sd: StoredNodeData = { ...dataRest };
  if (n.data.kind === "source") {
    sd.status = status === "running" ? "ok" : status;
    sd.sources = (sources || []).map((s) => storeSource(s, sc.ctx));
    const refs = measRefsOf(n.data);
    if (refs.length) sd.measurementRefs = refs.map((r) => storeMeasRef(r, sc.ctx));
  } else {
    const kept = own.shared ? [] : persistedOutputs(outputs || []);
    sd.outputs = kept.map((o) => storeOutput(o, wf.id, n.id, sc));
    if (!own.shared && status === "error") {
      sd.status = "error";
      if (error) sd.error = error;
    } else {
      // "stale": the node's code changed after its results were made.
      sd.status = !kept.length ? "idle" : status === "stale" ? "stale" : "ok";
    }
    if (!own.shared && consoleOut) sd.consoleOut = capText(consoleOut);
    if (kept.length && staleIds.has(n.id)) sd.stale = true;
  }
  return { ...nodeRest, id: n.id, position: n.position, data: sd } as StoredNode;
}

function storeEdge(e: Edge): StoredEdge {
  const { selected: _s, ...rest } = e as Edge & { selected?: boolean };
  return rest as StoredEdge;
}

/** Nodes whose saved results must be flagged stale: by sourceEditStore revs,
 *  and — when `discardedUid` is set — every result node reading an image
 *  source of that document. */
function staleNodeIds(wf: WorkflowTab, sc: StoreCtx): Set<string> {
  const ids = computeStale(wf.nodes, wf.edges, sc.ctx.revs || {}).nodeIds;
  if (sc.discardedUid) {
    const keyDoc = new Map<string, string | undefined>();
    for (const n of wf.nodes) {
      if (n.data.kind !== "source") continue;
      for (const s of n.data.sources || []) if (!s.name) keyDoc.set(s.key, sourceDocUid(s, sc.ctx));
    }
    for (const n of wf.nodes) {
      if (n.data.kind === "source" || !(n.data.outputs && n.data.outputs.length) || ids.has(n.id)) continue;
      if (upstreamSourceKeys(n.id, wf.nodes, wf.edges).some((k) => keyDoc.get(k) === sc.discardedUid)) ids.add(n.id);
    }
  }
  return ids;
}

function storeWorkflow(wf: WorkflowTab, own: Ownership, sc: StoreCtx): StoredWorkflow {
  const staleIds = own.shared ? new Set<string>() : staleNodeIds(wf, sc);
  return {
    id: wf.id,
    name: wf.name,
    updatedAt: wf.updatedAt ?? 0,
    ...(own.home ? { home: own.home } : {}),
    ...(wf.homeConfirmed ? { homeConfirmed: true } : {}),
    owners: own.owners,
    shared: own.shared,
    nodes: wf.nodes.map((n) => storeNode(n, wf, own, staleIds, sc)),
    edges: wf.edges.map(storeEdge),
  };
}

function docEntry(u: string, state: OwnershipCtx): { name?: string; path?: string; sessionId?: string } {
  const open = state.docs.find((d) => d.uid === u);
  if (open) return { name: open.name, path: open.path || undefined, sessionId: open.id };
  const k = state.known?.[u];
  return { name: k?.name, path: k?.path, sessionId: k?.sessionId };
}

function docsTable(uid: string, uids: Iterable<string>, state: OwnershipCtx): Record<string, StoredDocEntry> {
  const holder = state.docs.find((d) => d.uid === uid)?.path || undefined;
  const out: Record<string, StoredDocEntry> = {};
  for (const u of uids) {
    const e = docEntry(u, state);
    const entry: StoredDocEntry = {};
    if (e.name) entry.name = e.name;
    if (u !== uid && e.path) {
      entry.path = e.path;
      const rel = holder ? relativePath(dirnameOf(holder), e.path) : undefined;
      if (rel) entry.rel = rel;
    }
    if (e.sessionId) entry.sessionId = e.sessionId;
    out[u] = entry;
  }
  return out;
}

function capTombstones(ts: Readonly<Record<string, number>> | undefined): Record<string, number> {
  if (!ts) return {};
  const entries = Object.entries(ts).filter(([, v]) => typeof v === "number" && Number.isFinite(v));
  entries.sort((a, b) => b[1] - a[1]);
  return Object.fromEntries(entries.slice(0, TOMBSTONE_CAP));
}

/** The workflows document `uid` owns now, in tab order, with their ownership. */
function ownedBy(uid: string, state: OwnershipCtx & { tabs: readonly WorkflowTab[] }): Array<{ wf: WorkflowTab; own: Ownership }> {
  const out: Array<{ wf: WorkflowTab; own: Ownership }> = [];
  for (const wf of state.tabs) {
    if (isPristineTab(wf)) continue;
    const own = deriveOwnership(wf, state);
    if (own.owners.includes(uid)) out.push({ wf, own });
  }
  return out;
}

/** Upload names a workflow's source nodes hold, in node / source order. */
function uploadNamesOf(wf: WorkflowTab, into: string[]): void {
  for (const n of wf.nodes) {
    for (const s of n.data.sources || []) if (s.name && !into.includes(s.name)) into.push(s.name);
  }
}

/** The uploads document `uid` embeds: those of the (non-pristine) workflows
 *  it owns whose `home` it is — a shared workflow's uploads go to its home
 *  only. Their labels / folders go into that document's `meta` too. */
export function uploadsOwnedBy(uid: string, state: OwnershipCtx & { tabs: readonly WorkflowTab[] }): string[] {
  const out: string[] = [];
  for (const { wf, own } of ownedBy(uid, state)) if (own.home === uid) uploadNamesOf(wf, out);
  return out;
}

/** Everything document `uid`'s `workflows/` section is written from. */
export function serializeForDoc(
  uid: string, state: PersistState, opts: SerializeOptions = {},
): DocWorkflowsPayload {
  const blobs: Record<string, WorkflowBlob> = {};
  const sc: StoreCtx = { ctx: state, blobs, discardedUid: opts.figureDiscarded ? uid : undefined };
  const uploads: string[] = [];
  const referenced = new Set<string>([uid]);
  const workflows: StoredWorkflow[] = [];
  for (const { wf, own } of ownedBy(uid, state)) {
    workflows.push(storeWorkflow(wf, own, sc));
    for (const u of own.owners) referenced.add(u);
    if (own.home === uid) uploadNamesOf(wf, uploads);
  }
  const manifest: WorkflowsManifestV1 = {
    format: WORKFLOWS_FORMAT,
    version: WORKFLOWS_VERSION,
    ...(state.appVersion ? { savedWith: state.appVersion } : {}),
    docUid: uid,
    docs: docsTable(uid, referenced, state),
    workflows,
  };
  const ts = capTombstones(state.tombstones?.[uid]);
  if (Object.keys(ts).length) manifest.tombstones = ts;
  const meta = state.meta?.[uid];
  if (meta && Object.keys(meta).length) manifest.meta = { ...meta };
  return { manifest, blobs, uploads };
}

/** Identity of a workflow's saved content: changes on code, config, wiring,
 *  node position, name, saved outputs, pins, ownership and stale flags; NOT
 *  on thumbnails, selection, runKey/mpfId, unpinned archives or updatedAt.
 *  Pass the open documents (and revs) so the signature survives the
 *  workflow's documents being re-keyed by open / close. */
export function contentSig(wf: WorkflowTab, ctx: Partial<PersistState> = {}): string {
  const state: PersistState = { ...ctx, docs: ctx.docs ?? [], tabs: ctx.tabs ?? [] };
  const own = deriveOwnership(wf, state);
  const { updatedAt: _u, ...rest } = storeWorkflow(wf, own, { ctx: state, blobs: null });
  return sigOf(rest);
}

/** Identity of what serializeForDoc would write for `uid`, minus session
 *  noise (session ids, app version, tombstones). Equal to the signature of
 *  the same document's file as read ⇔ the file is up to date. */
export function docSig(uid: string, state: PersistState): string {
  const referenced = new Set<string>([uid]);
  const workflows = ownedBy(uid, state).map(({ wf, own }) => {
    for (const u of own.owners) referenced.add(u);
    return storeWorkflow(wf, own, { ctx: state, blobs: null });
  });
  const docs = [...referenced].sort().map((u) => {
    const e = docEntry(u, state);
    return [u, e.name ?? null, u === uid ? null : (e.path ?? null)];
  });
  return sigOf({ v: WORKFLOWS_VERSION, workflows, docs, meta: state.meta?.[uid] ?? null });
}

// ── Deserialize ─────────────────────────────────────────────

export interface DeserializeCtx extends OwnershipCtx {
  /** The open document the payload was read from. */
  doc: { id: string; name?: string; path?: string | null };
  /** The uid the workspace gives `doc`: the manifest's docUid, unless it had
   *  to be re-minted (another open document holds it) or there was none. */
  uid: string;
  revs?: Readonly<Record<string, number>>;
}

/** The parts of the read response (api.readWorkflows) deserializeDoc uses. */
export interface DeserializePayload {
  manifest: WorkflowsManifest | null;
  blobs?: Readonly<Record<string, { kind: WorkflowBlobKind; data: string }>>;
  unsupported?: boolean;
  uploads?: readonly RestoredUpload[];
  renamed?: Readonly<Record<string, string>>;
}

export interface DeserializedDoc {
  /** The uid the file was saved under (undefined: no saved workflows). */
  fileUid?: string;
  /** In workspace form: own sources under figKey(doc), other open documents'
   *  under theirs, closed documents' under "u:<uid>". */
  tabs: WorkflowTab[];
  tombstones: Record<string, number>;
  /** Every other document the manifest names, by (remapped) uid. */
  known: Record<string, KnownDoc>;
  /** Tiles for the restored uploads, under their final names. */
  uploads: InsetSource[];
  /** The manifest's labels (utils/analysisMeta), upload keys renamed like
   *  the uploads; unvalidated — see readDocMeta. */
  meta?: Record<string, unknown>;
  /** Saved session id → current session id, applied to picker keys. */
  idMap: Record<string, string>;
}

/** The Sources-panel tile of a restored upload (same shape the graph builds
 *  for a fresh upload). */
export function uploadTile(u: RestoredUpload): InsetSource {
  const [w, h] = u.size || [0, 0];
  return {
    key: `img:${u.name}`, name: u.name, row: -1, col: -1, inset_index: -1,
    label: u.name, natural_width: w, natural_height: h, thumbnail: u.thumbnail || "",
  };
}

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);

/** A runtime key minted while document `oldId` was not the active one
 *  ("<oldId>__raw" / "<oldId>-<fig>__raw") re-pointed at `idMap[oldId]`. */
export function remapRuntimeKey(key: string, idMap: Readonly<Record<string, string>>): string {
  for (const [o, n] of Object.entries(idMap)) {
    if (o && o !== n && (key.startsWith(`${o}__`) || key.startsWith(`${o}-`))) return n + key.slice(o.length);
  }
  return key;
}

/** Best-effort remap of session doc ids inside picker image ids (band picker
 *  `roi.images`, intensity / Cellpose groups). A miss only makes the picker
 *  ask again. */
export function remapRuntimeKeys(tabs: readonly WorkflowTab[], idMap: Readonly<Record<string, string>>): WorkflowTab[] {
  if (!Object.keys(idMap).length) return tabs as WorkflowTab[];
  return mapTabNodes(tabs, (d) => mapPickerKeys(d, (k) => remapRuntimeKey(k, idMap)));
}

const MASK_FIELDS = ["editedMasks", "editedNucleiMasks", "editedChannelMasks"] as const;

/** Rename restored uploads ({old: new}, from the read response) in the tabs:
 *  upload sources (name, key, default label), `ranWith`, picker image ids
 *  (`img:<old>`, or a bare label `<old>`) and the pickers' per-label masks. */
export function applyRenames(tabs: readonly WorkflowTab[], renamed: Readonly<Record<string, string>>): WorkflowTab[] {
  const map = new Map<string, string>();
  for (const [o, n] of Object.entries(renamed || {})) if (o && n && o !== n) map.set(o, n);
  if (!map.size) return tabs as WorkflowTab[];
  const keyMap = new Map<string, string>();
  for (const [o, n] of map) { keyMap.set(`img:${o}`, `img:${n}`); keyMap.set(o, n); }
  const fn = (k: string) => keyMap.get(k) ?? k;
  return mapTabNodes(tabs, (d) => {
    let out = d;
    if (d.sources && d.sources.length) {
      let changed = false;
      const sources = d.sources.map((s) => {
        const nn = s.name ? map.get(s.name) : undefined;
        if (!s.name || !nn) return s;
        changed = true;
        return {
          ...s, name: nn, key: `img:${nn}`, label: s.label === s.name ? nn : s.label,
          ...(s.rawKey ? { rawKey: fn(s.rawKey) } : {}),
        };
      });
      if (changed) out = { ...out, sources };
    }
    if (d.ranWith) {
      const ranWith = mapRecordKeys(d.ranWith, fn);
      if (ranWith !== d.ranWith) out = { ...out, ranWith };
    }
    out = mapPickerKeys(out, fn);
    for (const field of ["intensity", "cellposeSeg"] as const) {
      const cfg = out[field] as Record<string, unknown> | undefined;
      if (!cfg) continue;
      let next = cfg;
      for (const m of MASK_FIELDS) {
        const rec = cfg[m];
        if (!isObj(rec)) continue;
        const moved = mapRecordKeys(rec, fn);
        if (moved !== rec) next = { ...next, [m]: moved };
      }
      if (next !== cfg) out = { ...out, [field]: next };
    }
    return out;
  });
}

/** One document's `workflows/` section back into workspace tabs. */
export function deserializeDoc(payload: DeserializePayload, ctx: DeserializeCtx): DeserializedDoc {
  const tiles = (payload.uploads || []).map(uploadTile);
  const empty: DeserializedDoc = { tabs: [], tombstones: {}, known: {}, uploads: tiles, idMap: {} };
  const m = payload.manifest as Partial<WorkflowsManifestV1> | null;
  if (!m || payload.unsupported || m.format !== WORKFLOWS_FORMAT
      || typeof m.version !== "number" || m.version > WORKFLOWS_VERSION) return empty;

  const fileUid = typeof m.docUid === "string" && m.docUid ? m.docUid : undefined;
  const mapUid = (u: string | undefined) => (u && fileUid && u === fileUid ? ctx.uid : u);
  // Open documents by uid — the one being read under the uid it now has.
  const docs: DocRef[] = [
    { ...ctx.doc, uid: ctx.uid },
    ...ctx.docs.filter((d) => d.id !== ctx.doc.id && d.uid !== ctx.uid),
  ];
  const byUid = new Map(docs.map((d) => [d.uid, d]));
  const entries: Record<string, StoredDocEntry> = isObj(m.docs) ? (m.docs as Record<string, StoredDocEntry>) : {};

  const idMap: Record<string, string> = {};
  const known: Record<string, KnownDoc> = {};
  for (const [u, e] of Object.entries(entries)) {
    if (!isObj(e)) continue;
    const open = byUid.get(mapUid(u)!);
    if (open && typeof e.sessionId === "string" && e.sessionId && e.sessionId !== open.id) idMap[e.sessionId] = open.id;
    if (u === fileUid) continue;
    const k: KnownDoc = {};
    if (typeof e.name === "string") k.name = e.name;
    if (typeof e.path === "string") k.path = e.path;
    if (typeof e.sessionId === "string") k.sessionId = e.sessionId;
    const cands = locateCandidates(ctx.doc.path, {
      path: k.path, rel: typeof e.rel === "string" ? e.rel : undefined,
    });
    if (cands.length) k.candidates = cands;
    known[u] = k;
  }

  const renamed = payload.renamed || {};
  const thumbs = new Map((payload.uploads || []).map((u) => [u.name, u]));
  const blobs = payload.blobs || {};

  const restoreSource = (ss: StoredSource): InsetSource => {
    const { doc, key, figKey: fk, ...rest } = ss;
    if (ss.name) {
      // Under its STORED name: applyRenames (below) renames every reference
      // to it in one pass — renaming here too would chain {a→b, b→c}.
      return {
        ...rest, name: ss.name, key: `img:${ss.name}`,
        thumbnail: thumbs.get(renamed[ss.name] || ss.name)?.thumbnail || "",
      };
    }
    if (doc) {
      const u = mapUid(doc)!;
      const raw = ss.rawKey || "";
      const open = byUid.get(u);
      const f = open ? figKey(open, ss.figureId) : closedFigKey(u, ss.figureId);
      return { ...rest, rawKey: raw, figKey: f, key: nsKey(f, raw), thumbnail: "", ...(open ? { mpfId: open.id } : {}) };
    }
    return { ...rest, key: key || "", ...(fk ? { figKey: fk } : {}), thumbnail: "" };
  };

  const restoreMeasRef = (sr: StoredMeasurementRef): MeasurementRef => {
    const u = mapUid(sr.docUid);
    return {
      mpfId: (u && byUid.get(u)?.id) || "", mpf: sr.mpf || "", label: sr.label || "", csv: sr.csv || "",
      ...(u ? { docUid: u } : {}),
    };
  };

  const restoreOutput = (so: StoredOutput): NodeOutput | null => {
    const b = blobs[so.blob];
    if (!b || typeof b.data !== "string") return null;
    const { blob, rCodeBlob, payloadSig: _ps, ...rest } = so as StoredOutput & { payloadSig?: string };
    const out: NodeOutput = { ...rest, payload: b.data, blobId: blob };
    const rc = rCodeBlob ? blobs[rCodeBlob] : undefined;
    if (rc && typeof rc.data === "string") out.rCode = rc.data;
    return out;
  };

  const tabs: WorkflowTab[] = [];
  for (const raw of Array.isArray(m.workflows) ? m.workflows : []) {
    const sw = raw as StoredWorkflow;
    if (!isObj(sw) || typeof sw.id !== "string" || typeof sw.name !== "string"
        || !Array.isArray(sw.nodes) || !Array.isArray(sw.edges)) continue;
    const staleIds = new Set<string>();
    const nodes: Node<NodeData>[] = [];
    for (const sn of sw.nodes) {
      if (!isObj(sn) || typeof sn.id !== "string" || !isObj(sn.data) || typeof sn.data.kind !== "string") continue;
      const { data, ...nodeRest } = sn;
      const { sources, measurementRefs, outputs, stale, status, ...dataRest } = data;
      let nd: NodeData = { ...dataRest, label: String(data.label ?? ""), kind: data.kind } as NodeData;
      if (data.kind === "source") {
        nd.sources = (Array.isArray(sources) ? sources : []).map(restoreSource);
        const refs = (Array.isArray(measurementRefs) ? measurementRefs : []).map(restoreMeasRef);
        if (refs.length) { nd.measurementsRef = refs[0]; nd.measurementRefs = refs; }
        if (status) nd.status = status === "running" ? "ok" : status;
      } else {
        const outs = (Array.isArray(outputs) ? outputs : [])
          .map(restoreOutput).filter((o): o is NodeOutput => !!o);
        nd.outputs = outs;
        nd.status = status === "error" ? "error" : !outs.length ? "idle" : status === "stale" ? "stale" : "ok";
        if (stale && outs.length) staleIds.add(sn.id);
      }
      nd = mapPickerKeys(nd, (k) => remapRuntimeKey(k, idMap));
      nodes.push({
        ...nodeRest,
        id: sn.id,
        type: sn.type ?? data.kind,
        position: isObj(sn.position) ? (sn.position as { x: number; y: number }) : { x: 0, y: 0 },
        data: nd,
      } as Node<NodeData>);
    }
    // Verbatim: the graph gives every edge type "deletable", and a default
    // added here would make the file's own signature differ from what it reads.
    const edges: Edge[] = sw.edges
      .filter((e) => isObj(e) && typeof e.id === "string" && typeof e.source === "string" && typeof e.target === "string")
      .map((e) => ({ ...e }) as Edge);
    // Results come back at the CURRENT revision of what they read — or one
    // below it when saved stale, which computeStale then flags until re-run.
    const revs = ctx.revs || {};
    for (const n of nodes) {
      if (n.data.kind === "source" || !(n.data.outputs && n.data.outputs.length)) continue;
      const lag = staleIds.has(n.id) ? 1 : 0;
      n.data.ranWith = Object.fromEntries(
        upstreamSourceKeys(n.id, nodes, edges).map((k) => [k, (revs[k] ?? 0) - lag]),
      );
    }
    const tab: WorkflowTab = { id: sw.id, name: sw.name, nodes, edges };
    if (typeof sw.updatedAt === "number" && Number.isFinite(sw.updatedAt)) tab.updatedAt = sw.updatedAt;
    const home = mapUid(typeof sw.home === "string" ? sw.home : undefined);
    if (home) tab.home = home;
    if (sw.homeConfirmed) tab.homeConfirmed = true;
    tabs.push(tab);
  }

  const tombstones: Record<string, number> = {};
  if (isObj(m.tombstones)) {
    for (const [id, t] of Object.entries(m.tombstones)) {
      if (typeof t === "number" && Number.isFinite(t)) tombstones[id] = t;
    }
  }
  return {
    fileUid,
    tabs: applyRenames(tabs, renamed),
    tombstones,
    known,
    uploads: tiles,
    // Upload keys in the labels follow the restored uploads' final names.
    ...(isObj(m.meta) ? { meta: renameMetaUploads(m.meta, renamed) as Record<string, unknown> } : {}),
    idMap,
  };
}

// ── Merge ───────────────────────────────────────────────────

/** Deletion times by document uid, then workflow id. */
export type TombstonesByDoc = Readonly<Record<string, Readonly<Record<string, number>>>>;

export interface MergeResult {
  tabs: WorkflowTab[];
  /** Incoming copies not taken (and existing ones a tombstone removed). */
  dropped: Array<{ id: string; reason: "tombstone" | "superseded" }>;
  /** Incoming ids re-minted because they collided with a different
   *  document's single-owner workflow (Finder copy / Save-As twin). */
  reminted: Record<string, string>;
  /** Existing ids whose copy was replaced by a newer incoming one. */
  replaced: string[];
}

/** Fold one file's workflows into the workspace (plan §1.5):
 *    1. updatedAt not newer than tombstones[id] → dropped;
 *    2. id not present → added;
 *    3. present, and both copies single-owner of DIFFERENT documents →
 *       the incoming one gets a new id and is added;
 *    4. otherwise the newer updatedAt wins; a tie keeps the existing one.
 *  Existing workflows a tombstone outdates are dropped too, so the result
 *  doesn't depend on which file was opened first. `tombstones` is per
 *  document uid (the incoming file's included), and a tombstone only removes
 *  a copy that document owns: deleting a workflow in a Save-As or Finder
 *  copy must never delete the same-id workflow from the original. Returns
 *  `existing` itself when nothing changed. */
export function mergeIncoming(
  existing: readonly WorkflowTab[],
  incoming: readonly WorkflowTab[],
  tombstones: TombstonesByDoc,
  ctx: OwnershipCtx,
  opts: { mintId?: () => string } = {},
): MergeResult {
  const mint = opts.mintId || newWorkflowId;
  const dropped: MergeResult["dropped"] = [];
  const reminted: Record<string, string> = {};
  const replaced: string[] = [];
  const outdated = (t: WorkflowTab) => {
    const at = t.updatedAt ?? 0;
    return deriveOwnership(t, ctx).owners.some((u) => {
      const ts = tombstones[u]?.[t.id];
      return typeof ts === "number" && at <= ts;
    });
  };
  let changed = false;
  const out: WorkflowTab[] = [];
  for (const t of existing) {
    if (outdated(t)) { dropped.push({ id: t.id, reason: "tombstone" }); changed = true; continue; }
    out.push(t);
  }
  const index = new Map(out.map((t, i) => [t.id, i]));
  for (const inc of incoming) {
    if (outdated(inc)) { dropped.push({ id: inc.id, reason: "tombstone" }); continue; }
    const at = index.get(inc.id);
    if (at == null) {
      index.set(inc.id, out.length);
      out.push(inc);
      changed = true;
      continue;
    }
    const cur = out[at];
    const a = deriveOwnership(cur, ctx);
    const b = deriveOwnership(inc, ctx);
    if (a.owners.length === 1 && b.owners.length === 1 && a.owners[0] !== b.owners[0]) {
      let id = mint();
      while (index.has(id)) id = mint();
      reminted[inc.id] = id;
      index.set(id, out.length);
      out.push({ ...inc, id });
      changed = true;
      continue;
    }
    if ((inc.updatedAt ?? 0) > (cur.updatedAt ?? 0)) {
      out[at] = inc;
      replaced.push(inc.id);
      changed = true;
    } else {
      dropped.push({ id: inc.id, reason: "superseded" });
    }
  }
  return { tabs: changed ? out : (existing as WorkflowTab[]), dropped, reminted, replaced };
}

