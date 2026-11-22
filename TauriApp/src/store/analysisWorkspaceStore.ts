/* ──────────────────────────────────────────────────────────
   analysisWorkspaceStore — the Analysis workspace's state, outside
   the graph component.

   The Analysis view unmounts in Builder / Collage mode, so the
   workflow tabs, the active tab and the "Direct upload" tiles live
   here: they (and every node's outputs) survive leaving and
   returning to Analysis, and save / close / quit can read them while
   the view is not on screen. This replaces the graph's former
   localStorage session (mpfig.workflow_session), its module-level
   outputs cache and the mpfig.analysis.uploads tiles. Like those it
   is in memory only — a fresh launch starts empty; workflows reach
   disk through each .mpf's `workflows/` section (utils/workflowPersist).

   PLAIN zustand, deliberately not immer: immer auto-freezes what it
   produces, which would deep-freeze ReactFlow nodes and multi-MB
   payload strings, and the graph mutates nothing through a draft.

   setTabs keeps the graph's useState contract EXACTLY: it takes a
   value or an updater, and an updater returning the SAME array is a
   no-op (no notification, no re-render). The graph's setNodes /
   setEdges bail out on a same-reference return; without that the
   parseDeclaredOutputs effect loops ("infinite update", blank view).

   Document identity (utils/workflowOwnership, utils/docIdentity): every
   open document gets a durable uid in `docUids` as soon as it appears in
   collageStore.openDocs. A document read from disk adopts the uid its
   file was saved under (adoptUid): its doc.json uid, else its manifest
   docUid. A figure save writes the tab's uid into the file, so the file
   is recognised after a rename; a file found at a new path (renamed or
   moved outside the app) is re-pointed with docMoved — a move, never a
   new uid.

   A debounced subscriber (syncWorkspaceNow) keeps each workflow's
   `updatedAt` (bumped when its saved content changes — never while it
   is locked) and `home` (deriveOwnership) up to date.

   Persistence — each .mpf carries the workflows it owns in its
   `workflows/` section (utils/workflowPersist):
     • hydrate: a module-level openDocs subscriber reads every document
       that has a file ONCE per doc id (ensureDocLoaded), strictly one
       file at a time, outside the doc-op queue — Open, session-restore
       cold tabs and "Open missing" all arrive this way. Tab switches
       and disk reloads never touch workflows.
     • persist (persistDocWorkflows): the workflow-only atomic rewrite
       of the document's file, skipped while its signature equals the
       file's (savedSig), before its read finished, or when the file's
       workflows could not be read / are a newer format. Runs on figure
       save (afterDocSaved), document close and app quit (projectNav).
     • close (closeDoc): its single-owner workflows leave the
       workspace; shared ones are re-keyed to "u:<uid>" (locked).
     • uploads: the workflow's `home` document embeds its uploaded
       images. A workflow reading no figure data asks which document
       that is the first time it gets an upload (askUploadHome →
       ChooseDocumentDialog); a shared one changes it from the tab menu
       (setUploadHome). A file's restored uploads come back as tiles in
       "Direct upload", renamed when the session holds a different file
       of the same name.
     • labels (utils/analysisMeta): source renames, measurement group
       labels and source-library folders of a document's keys (and of the
       uploads it embeds) are saved in its manifest's `meta`, document-
       relative, and merged back into localStorage — the mirror the panel
       reads — when the file is read. They count towards its savedSig.

   Shared workflows (plan §1.3-1.5) — a workflow is LOCKED (greyed out,
   read-only) while one of its owner documents is closed:
     • the single hard guard is in setTabNodes / setTabEdges: a write to
       a locked tab is discarded unless flagged { view: true } (selection,
       measured size); deleteWorkflow refuses one;
     • openMissing looks for the closed documents (saved path, path
       relative to an owner's file, same name next to it, the paths the
       last-seen table knows, and a file carrying the uid in those
       folders — one api.resolveMpf call) and opens hits as cold tabs;
       the read that follows re-keys "u:<uid>" back to the document. A
       file without a uid (re-saved by an older build) adopts the
       expected one; a file with another uid is used only after the user
       confirms (rebindDoc).
       Not found → locate (native open dialog);
     • removeMissingSources ("Remove their sources") drops the closed
       documents' sources and tables, leaving it editable;
     • a workflow that comes back unlocked has its results flagged
       "needs re-run".
   ────────────────────────────────────────────────────────── */

import { create } from "zustand";
import type { Edge, Node } from "@xyflow/react";
import { api } from "../api/client";
import type { MpfResolveRef, MpfResolveResult, WorkflowsReadResponse, WorkflowsWriteResult } from "../api/types";
import type { InsetSource, NodeData, WorkflowTab } from "../utils/analysisTypes";
import { upstreamSourceKeys } from "../utils/analysisStale";
import {
  basenameOf, contentSig, deserializeDoc, dirnameOf, docSig, locateCandidates, mergeIncoming, newWorkflowId,
  remapRuntimeKeys, serializeForDoc, uploadsOwnedBy, WORKFLOWS_FORMAT, WORKFLOWS_VERSION,
  type PersistState,
} from "../utils/workflowPersist";
import {
  bindMeasurementRefs, closedKeyBase, deriveOwnership, docKeyBase, docRefs, isLocked, isPristineTab,
  missingOwners, newDocUid, rebindDoc, rekeyKey, rekeyRecord, rekeyTabs, removeDocSources,
  type DocRef, type KnownDoc, type OwnershipCtx,
} from "../utils/workflowOwnership";
import {
  asLabels, DIRECT_UPLOAD_FOLDER, docMetaOf, fileUnderFolder, mergeDocMeta, metaAsRead, metaSigForm, readDocMeta,
  uploadKey,
  type AnalysisLabels, type DocMeta,
} from "../utils/analysisMeta";
import {
  clearFileUid, fileUidOf, forgetDocFile, isMissingFileError, isOtherDocumentError, noteDocFile, refreshDocFileId,
  registryPathsOf, registryUidAt, resetDocIdentityForTests, setFileIdOf, setFileUid,
} from "../utils/docIdentity";
import { confirm as confirmDialog } from "../components/shared/ConfirmDialog";
import { chooseDocument } from "../components/shared/ChooseDocumentDialog";
import { useCollageStore, type DocTab } from "./collageStore";
import { useSourceEditStore } from "./sourceEditStore";

/** Same shape as React's SetStateAction<WorkflowTab[]>. */
export type TabsUpdater = WorkflowTab[] | ((tabs: WorkflowTab[]) => WorkflowTab[]);

/** setTabNodes / setTabEdges options. `view`: the update only carries view
 *  state (ReactFlow selection / measured size) — allowed on a locked tab. */
export interface TabWriteOptions {
  view?: boolean;
}

/** askUploadHome options. */
export interface UploadHomeAskOptions {
  /** How an open document is named in the dialog (by session id). */
  label?: (docId: string) => string | undefined;
  /** Ask even when the home was chosen before / asked this session (tab menu). */
  force?: boolean;
}

/** What a workflow tab's menu offers for its uploaded images (uploadHomeOptions):
 *    shared        it reads several documents' figures: which of them (open)
 *                  embeds the uploads — "Store uploaded images in ▸";
 *    uploads-only  it reads no figure data: which document saves it at all
 *                  (askUploadHome, forced). */
export type UploadHomeOptions =
  | { kind: "shared"; choices: DocRef[]; home?: string }
  | { kind: "uploads-only"; home?: string };

/** A closed document a locked workflow needs, as the banner names it. */
export interface MissingDoc {
  uid: string;
  /** "b.mpf" (its file name), or the name of a never-saved document. */
  label: string;
  /** Its file is known (saved path / locate candidates): worth looking for. */
  locatable: boolean;
}

/** Where a document's saved workflows stand:
 *    pending      its file is being read;
 *    done         read (or it has no file) — it may be persisted;
 *    failed       the read failed — never overwrite its workflows;
 *    unsupported  written by a newer format — never overwrite. */
export type HydrationStatus = "pending" | "done" | "failed" | "unsupported";

export interface PersistOptions {
  /** The document's figure edits are being thrown away (Don't save /
   *  Discard & close): results reading its images are stored stale. */
  figureDiscarded?: boolean;
  /** Write even when the signature says the file is up to date (path change). */
  force?: boolean;
}

/** What persistDocWorkflows did:
 *    written       the file's workflows/ section was rewritten;
 *    clean         the file already holds exactly this (nothing sent);
 *    no-path       Untitled (or closed) — nowhere to write;
 *    not-hydrated  its file's workflows were never read (read failed) —
 *                  never overwritten;
 *    unsupported   its file holds a newer format — never overwritten. */
export type PersistOutcome =
  | "written" | "clean" | "no-path" | "not-hydrated" | "unsupported";

export interface AnalysisWorkspaceState {
  /** Every open workflow, of every document. Never empty. */
  tabs: WorkflowTab[];
  activeId: string;
  /** Session doc id → durable document uid. */
  docUids: Record<string, string>;
  /** Session doc id → how far reading its file's workflows got. */
  hydrated: Record<string, HydrationStatus>;
  /** Session doc id → docSig of its file as last read / written. */
  savedSig: Record<string, string>;
  /** Document uid → workflow id → deletion time (ms); written into that
   *  document's manifest so a stale copy elsewhere can't resurrect it. */
  tombstones: Record<string, Record<string, number>>;
  /** "Direct upload" tiles (standalone Analysis uploads), newest first. */
  uploads: InsetSource[];
  /** Closed documents some workflow names, by uid (from manifests). */
  knownDocs: Record<string, KnownDoc>;

  /** The graph's setWorkflowTabs: a value or an updater; returning the same
   *  array is a no-op. An empty result is replaced by one pristine tab. */
  setTabs: (updater: TabsUpdater) => void;
  /** The graph's setNodes for tab `tabId`. An updater returning the same
   *  nodes array changes nothing — no new tabs array, no notification; a
   *  change replaces that tab only (every other tab keeps its identity).
   *  THE lock guard: on a locked tab the updater still runs (callers read
   *  state through it) but its result is discarded, unless `opts.view`. */
  setTabNodes: (tabId: string, updater: (nodes: Node<NodeData>[]) => Node<NodeData>[], opts?: TabWriteOptions) => void;
  /** The graph's setEdges for tab `tabId`; same bail-out and lock guard as
   *  setTabNodes. */
  setTabEdges: (tabId: string, updater: (edges: Edge[]) => Edge[], opts?: TabWriteOptions) => void;
  setActive: (id: string) => void;
  /** The document's uid, minted (and recorded) on first need. */
  uidFor: (docId: string) => string;
  /** Give open document `docId` the uid its file was saved under, moving
   *  everything that named its previous uid (home, measurement tables,
   *  tombstones) over. */
  adoptUid: (docId: string, uid: string) => void;
  /** Remove a workflow tab: tombstoned in its open owners' manifests (unless
   *  pristine), the neighbour activated, and a fresh pristine tab created
   *  when it was the last one. A locked workflow is refused (its closed
   *  owner couldn't record the deletion): returns false. */
  deleteWorkflow: (wfId: string) => boolean;
  /** "Remove their sources": drop the closed documents' sources and
   *  measurement tables from locked workflow `wfId`, which leaves it owned
   *  by the open documents only (editable; its results flagged stale). */
  removeMissingSources: (wfId: string) => void;
  /** The first upload attached to workflow `wfId` while it reads no
   *  document's figure data (and its home was never chosen): ask which
   *  document saves it — and embeds its uploaded images — in the
   *  ChooseDocumentDialog: the open documents (its current home pre-selected)
   *  or "New document…" (a new Untitled tab). Asked once per workflow per
   *  session (`force` asks again — the tab menu); Cancel keeps the home it
   *  had, so a workflow is never ownerless. Resolves whether the home changed. */
  askUploadHome: (wfId: string, opts?: UploadHomeAskOptions) => Promise<boolean>;
  /** Make document `uid` workflow `wfId`'s confirmed home — the document
   *  whose .mpf embeds its uploaded images ("Store uploaded images in ▸").
   *  A workflow reading figures takes one of those documents; one reading
   *  none, any open document. Refused while locked. Returns whether it changed. */
  setUploadHome: (wfId: string, uid: string) => boolean;
  /** Add (or refresh) "Direct upload" tiles, newest first. */
  addUploads: (entries: InsetSource[]) => void;
  removeUpload: (key: string) => void;
  /** Move every key under `oldBase` to `newBase` (a document's figKey base
   *  changed): sources, ranWith and picker ids in all tabs, and the Edit
   *  Panel revisions / results / re-run requests in sourceEditStore. */
  rekeyDoc: (oldBase: string, newBase: string) => void;

  // ── Persistence ──
  /** Read document `docId`'s saved workflows once per doc id (one file at a
   *  time, queued). A document without a file is done at once. Resolves when
   *  its hydration finished (or was dropped because it closed). */
  ensureDocLoaded: (docId: string, path: string | null) => Promise<void>;
  /** Fold one file's read response into the workspace: adopt the file's uid,
   *  merge its workflows (plan §1.5), restore results and upload tiles, take
   *  savedSig from the file as read. A second call for the doc is a no-op. */
  hydrateDoc: (docId: string, payload: WorkflowsReadResponse) => void;
  /** Write what document `docId` owns now into its .mpf, when dirty. Waits
   *  for its pending read. Rejects when the write fails. */
  persistDocWorkflows: (docId: string, opts?: PersistOptions) => Promise<PersistOutcome>;
  /** A figure save finished: the tab takes the uid the file was saved under
   *  (`opts.docUid`, from the sidecar — without it, the old rule: a Save As
   *  re-mints); on a path change point the tab and every key at the new
   *  file; then persist (forced on a path change). `opts.sourceKept`: the
   *  previous file is another file that is still there (the sidecar says
   *  so) — a copy, so the old file keeps its labels too. `opts.fileId`: the
   *  physical file written (docIdentity). Rejects when the persist fails. */
  afterDocSaved: (docId: string, oldPath: string | null, newPath: string,
    opts?: { workflowsCarried?: boolean; docUid?: string; sourceKept?: boolean; fileId?: string | null }) => Promise<void>;
  /** The file of every open tab at `oldPath` is at `newPath` now (renamed or
   *  moved outside the app, found again by its uid): the tabs, their keys and
   *  their labels follow it. The same document — no new uid, no re-read; a
   *  document whose Analysis was saved stays saved. `opts.reread`: the file
   *  at `newPath` is not known to be the one that was read (Locate… picked
   *  a file whose uid can't be told): it is read there before any write,
   *  instead of counting as read. `opts.ids`: only these tabs on `oldPath`
   *  move (docLinks.repointOpenDocs: in a swap, the tabs one hop delivered
   *  to a path stay there when the next hop leaves it). */
  docMoved: (oldPath: string, newPath: string, opts?: { reread?: boolean; ids?: readonly string[] }) => void;
  /** Document `docId` is closing (call before collageStore.docRemove; the
   *  openDocs subscriber covers a tab removed any other way): its
   *  single-owner workflows leave, shared ones are re-keyed "u:<uid>". */
  closeDoc: (docId: string) => void;
  /** Open the closed documents locked workflow `wfId` needs: each one's
   *  saved path, its path relative to an open owner's file, the same file
   *  name next to it, or a file carrying its uid near any of those (one
   *  api.resolveMpf call); a hit opens as a cold tab (no builder swap) and is
   *  read. A document found nowhere goes to locate. */
  openMissing: (wfId: string) => Promise<void>;
  /** Let the user point at closed document `uid`'s file (native open dialog),
   *  then open it the same way. */
  locate: (wfId: string, uid: string) => Promise<void>;
}

// ── Factories ───────────────────────────────────────────────

/** A Source node (the graph's own factory, shared with the store so it can
 *  make a pristine tab without the component). */
export function newSourceNode(
  sources: InsetSource[],
  opts?: { id?: string; label?: string; position?: { x: number; y: number } },
): Node<NodeData> {
  const id = opts?.id ?? "source";
  return {
    id,
    type: "source",
    position: opts?.position ?? { x: 30, y: 30 },
    data: { label: opts?.label ?? "Source", kind: "source", sources, status: "ok" } as NodeData,
    draggable: true,
    // Only the primary source is undeletable — extra sources can be removed.
    deletable: id !== "source",
  };
}

/** A pristine workflow tab (one empty Source node) saved in `home`. */
export function newWorkflowTab(name: string, home?: string, id: string = newWorkflowId()): WorkflowTab {
  const tab: WorkflowTab = { id, name, nodes: [newSourceNode([])], edges: [] };
  if (home) tab.home = home;
  return tab;
}

// ── Helpers ─────────────────────────────────────────────────

/** Open documents (with their uids) + closed ones the workspace knows. */
export function workspaceOwnershipCtx(
  s: Pick<AnalysisWorkspaceState, "docUids" | "knownDocs"> = useAnalysisWorkspaceStore.getState(),
  openDocs: ReadonlyArray<{ id: string; name?: string; path?: string | null }> = useCollageStore.getState().openDocs,
): OwnershipCtx {
  return { docs: docRefs(openDocs, s.docUids), known: s.knownDocs };
}

/** A workflow tab's owner chip: "a", or "a + b · results not saved" when it
 *  reads figures of several documents. `label` names an open document (by
 *  session id); closed ones use their recorded name. null when no owner. */
export function ownerChip(
  wf: WorkflowTab, ctx: OwnershipCtx, label: (docId: string) => string | undefined,
): { text: string; shared: boolean } | null {
  const own = deriveOwnership(wf, ctx);
  if (!own.owners.length) return null;
  const names = own.owners.map((u) => {
    const d = ctx.docs.find((x) => x.uid === u);
    if (d) return label(d.id) || d.name || "Untitled";
    return ctx.known?.[u]?.name || "closed document";
  });
  return own.shared
    ? { text: `${names.join(" + ")} · results not saved`, shared: true }
    : { text: names[0], shared: false };
}

function activeDocUid(): string | undefined {
  const id = useCollageStore.getState().activeDocId;
  return id ? useAnalysisWorkspaceStore.getState().uidFor(id) : undefined;
}

/** The graph always shows a tab: an empty list becomes one pristine tab. */
function nonEmpty(tabs: WorkflowTab[], home: () => string | undefined): WorkflowTab[] {
  return tabs.length ? tabs : [newWorkflowTab("Workflow 1", home())];
}

/** Open documents (session ids, openDocs order) that own at least one
 *  non-pristine workflow — the analysis that document saves. */
export function docsWithAnalysis(
  s: Pick<AnalysisWorkspaceState, "tabs" | "docUids" | "knownDocs"> = useAnalysisWorkspaceStore.getState(),
  openDocs: ReadonlyArray<{ id: string; name?: string; path?: string | null }> = useCollageStore.getState().openDocs,
): string[] {
  const ctx = workspaceOwnershipCtx(s, openDocs);
  const owners = new Set<string>();
  for (const wf of s.tabs) {
    if (isPristineTab(wf)) continue;
    for (const u of deriveOwnership(wf, ctx).owners) owners.add(u);
  }
  return ctx.docs.filter((d) => owners.has(d.uid)).map((d) => d.id);
}

/** Untitled (never saved) documents owning analysis: they can't auto-save,
 *  so close / quit treat them as unsaved even with an empty figure. */
export function untitledAnalysisDocIds(
  s: Pick<AnalysisWorkspaceState, "tabs" | "docUids" | "knownDocs"> = useAnalysisWorkspaceStore.getState(),
  openDocs: ReadonlyArray<{ id: string; name?: string; path?: string | null }> = useCollageStore.getState().openDocs,
): string[] {
  const untitled = new Set(openDocs.filter((d) => !d.path).map((d) => d.id));
  if (!untitled.size) return [];
  return docsWithAnalysis(s, openDocs).filter((id) => untitled.has(id));
}

/** Whether open document `docId` owns any non-pristine workflow. */
export function docHasAnalysis(docId: string): boolean {
  return docsWithAnalysis().includes(docId);
}

// ── Locked (shared) workflows ───────────────────────────────

/** Whether workflow `wfId` is locked right now (an owner document is closed). */
export function workflowLockedNow(wfId: string): boolean {
  const s = useAnalysisWorkspaceStore.getState();
  const wf = s.tabs.find((t) => t.id === wfId);
  return !!wf && isLocked(wf, workspaceOwnershipCtx(s));
}

/** How a closed document is named to the user: its file name ("b.mpf"), or
 *  the name of a document that was never saved. */
export function closedDocLabel(uid: string, known?: Readonly<Record<string, KnownDoc>>): string {
  const k = known?.[uid];
  if (k?.path) return basenameOf(k.path);
  if (k?.name) return `${k.name} (never saved)`;
  return "a closed document";
}

/** The closed documents locked workflow `wf` needs, in owner order. */
export function missingDocsOf(wf: Pick<WorkflowTab, "nodes" | "home">, ctx: OwnershipCtx): MissingDoc[] {
  return missingOwners(wf, ctx).map((uid) => {
    const k = ctx.known?.[uid];
    return { uid, label: closedDocLabel(uid, ctx.known), locatable: !!(k?.path || k?.candidates?.length) };
  });
}

/** Workflows the "open the other documents?" prompt was shown for. */
const promptedLocked = new Set<string>();
/** True the first time it is asked for workflow `wfId` this session: the
 *  prompt appears once per locked workflow (the banner stays). */
export function claimLockedPrompt(wfId: string): boolean {
  if (promptedLocked.has(wfId)) return false;
  promptedLocked.add(wfId);
  return true;
}

// ── Uploads-only / shared workflows: where uploads are saved ─

/** Whether the workflow's source nodes hold any standalone upload. */
function hasUploads(wf: Pick<WorkflowTab, "nodes">): boolean {
  return wf.nodes.some((n) => n.data.kind === "source" && (n.data.sources || []).some((s) => !!s.name));
}

/** What workflow `wf`'s tab menu offers for its uploaded images (see
 *  UploadHomeOptions); null when it holds none, or reads one document's
 *  figures (that document embeds them — nothing to choose). */
export function uploadHomeOptions(wf: Pick<WorkflowTab, "nodes" | "home">, ctx: OwnershipCtx): UploadHomeOptions | null {
  if (!hasUploads(wf)) return null;
  const own = deriveOwnership(wf, ctx);
  if (!own.participants.length) return { kind: "uploads-only", home: own.home };
  if (!own.shared) return null;
  const choices = own.participants
    .map((u) => ctx.docs.find((d) => d.uid === u))
    .filter((d): d is DocRef => !!d);
  return { kind: "shared", choices, home: own.home };
}

/** Workflows the upload-home prompt was shown for this session. */
const askedUploadHome = new Set<string>();

// ── Labels kept outside the workflows (localStorage) ────────

/** Fired after the store rewrote the panel's labels in localStorage —
 *  rekeyAnalysisStorage moved keys, or a document's file brought its labels
 *  (hydrate) — so a mounted Analysis view reloads its copies instead of
 *  writing stale ones back. */
export const ANALYSIS_KEYS_MOVED_EVENT = "mpfig:analysis-keys-moved";
const SOURCE_NAMES_KEY = "mpfig.source_names";
const SOURCE_GROUPS_KEY = "mpfig.source_groups";
const MEAS_GROUPS_KEY = "mpfig.measurement_groups";

/** The Analysis panel's source renames, source-library folders and
 *  measurement group labels (localStorage, keyed by source key / measurement
 *  row) follow a document to a new key base — an Untitled document saved, or
 *  Save As. `keepOld` copies instead of moving (Save As: the old file is still
 *  there and keeps its labels). Returns whether anything moved. */
export function rekeyAnalysisStorage(oldBase: string, newBase: string, keepOld: boolean): boolean {
  if (!oldBase || oldBase === newBase) return false;
  const move = (k: string) => rekeyKey(k, oldBase, newBase);
  const read = (key: string): unknown => {
    try { return JSON.parse(localStorage.getItem(key) || "null"); } catch { return null; }
  };
  const write = (key: string, v: unknown) => {
    try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* storage full / disabled */ }
  };
  let changed = false;
  for (const key of [SOURCE_NAMES_KEY, MEAS_GROUPS_KEY]) {
    const rec = read(key);
    if (!rec || typeof rec !== "object" || Array.isArray(rec)) continue;
    const keep: Array<[string, unknown]> = [];
    const moved: Array<[string, unknown]> = [];
    for (const [k, v] of Object.entries(rec as Record<string, unknown>)) {
      const nk = move(k);
      if (nk === k) { keep.push([k, v]); continue; }
      moved.push([nk, v]);
      if (keepOld) keep.push([k, v]);
    }
    if (!moved.length) continue;
    // The document's own labels win over stale ones already under the new base.
    write(key, Object.fromEntries([...keep, ...moved]));
    changed = true;
  }
  const groups = read(SOURCE_GROUPS_KEY);
  if (Array.isArray(groups)) {
    let moved = false;
    const next = groups.map((g) => {
      if (!g || typeof g !== "object" || !Array.isArray((g as { sourceKeys?: unknown }).sourceKeys)) return g;
      const keys = (g as { sourceKeys: unknown[] }).sourceKeys;
      const out: unknown[] = [];
      for (const k of keys) {
        const nk = typeof k === "string" ? move(k) : k;
        if (nk !== k) moved = true;
        if (nk !== k && keepOld) out.push(k);
        if (!out.includes(nk)) out.push(nk);
      }
      return { ...(g as object), sourceKeys: out };
    });
    if (moved) { write(SOURCE_GROUPS_KEY, next); changed = true; }
  }
  if (changed) {
    try { window.dispatchEvent(new CustomEvent(ANALYSIS_KEYS_MOVED_EVENT, { detail: { oldBase, newBase } })); }
    catch { /* no window (tests) */ }
  }
  return changed;
}

/** The panel's labels as localStorage holds them now (source names,
 *  measurement group labels, source folders — absolute keys). */
export function readAnalysisLabels(): AnalysisLabels {
  const read = (key: string): unknown => {
    try { return JSON.parse(localStorage.getItem(key) || "null"); } catch { return null; }
  };
  return asLabels(read(SOURCE_NAMES_KEY), read(MEAS_GROUPS_KEY), read(SOURCE_GROUPS_KEY));
}

/** Write the sections of `next` that differ from `prev` (both from
 *  readAnalysisLabels) and tell a mounted Analysis view to reload its
 *  copies (ANALYSIS_KEYS_MOVED_EVENT) — it writes whole maps back. */
function writeAnalysisLabels(next: AnalysisLabels, prev: AnalysisLabels): boolean {
  if (next === prev) return false;
  const write = (key: string, v: unknown) => {
    try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* storage full / disabled */ }
  };
  let changed = false;
  if (next.sourceNames !== prev.sourceNames) { write(SOURCE_NAMES_KEY, next.sourceNames); changed = true; }
  if (next.measurementGroups !== prev.measurementGroups) { write(MEAS_GROUPS_KEY, next.measurementGroups); changed = true; }
  if (next.sourceGroups !== prev.sourceGroups) { write(SOURCE_GROUPS_KEY, next.sourceGroups); changed = true; }
  if (changed) {
    try { window.dispatchEvent(new CustomEvent(ANALYSIS_KEYS_MOVED_EVENT, { detail: { labels: true } })); }
    catch { /* no window (tests) */ }
  }
  return changed;
}

/** PersistState.meta for one document (undefined: no labels). */
function metaFor(uid: string, meta: Record<string, unknown> | undefined): PersistState["meta"] {
  return meta ? { [uid]: meta } : undefined;
}

/** A fresh "Direct upload" folder id (the panel mints the same shape). */
const directFolderId = () => `grp_direct_${Date.now().toString(36)}`;

/** Results of a workflow that comes back unlocked (its closed document
 *  reopened or located, or its sources removed) are flagged "needs re-run":
 *  each result node records one revision below what it reads, which
 *  computeStale reports, and shows the stale status. They were computed
 *  before it was locked and could not be refreshed meanwhile. */
function markResultsStale(wf: WorkflowTab, revs: Readonly<Record<string, number>>): WorkflowTab {
  let changed = false;
  const nodes = wf.nodes.map((n) => {
    if (n.data.kind === "source" || !(n.data.outputs && n.data.outputs.length)) return n;
    const keys = upstreamSourceKeys(n.id, wf.nodes, wf.edges);
    const status = n.data.status === "ok" ? "stale" as const : n.data.status;
    if (!keys.length && status === n.data.status) return n;
    changed = true;
    const data: NodeData = { ...n.data, status };
    if (keys.length) data.ranWith = Object.fromEntries(keys.map((k) => [k, (revs[k] ?? 0) - 1]));
    return { ...n, data };
  });
  return changed ? { ...wf, nodes } : wf;
}

/** A workflow that just became locked: a run still in flight can no longer
 *  write back (the lock guard drops it), so no node is left "running". */
function settleRunning(wf: WorkflowTab): WorkflowTab {
  if (!wf.nodes.some((n) => n.data.status === "running")) return wf;
  return {
    ...wf,
    nodes: wf.nodes.map((n) => (n.data.status !== "running" ? n : {
      ...n, data: { ...n.data, status: n.data.outputs && n.data.outputs.length ? "stale" as const : "idle" as const },
    })),
  };
}

/** Workflows holding closed document `uid` ("u:<uid>" keys) put back under
 *  open document `doc`: keys re-keyed to its figKey, its measurement tables
 *  bound to the session doc, picker runtime ids remapped from the session id
 *  it had (`oldSession`), and every workflow that moved has its results
 *  flagged "needs re-run". Returns `tabs` itself when nothing changed. */
function reopenTabs(
  tabs: WorkflowTab[], uid: string, doc: { id: string; path?: string | null },
  revs: Readonly<Record<string, number>>, oldSession?: string,
): WorkflowTab[] {
  const reopened = rekeyTabs(tabs, closedKeyBase(uid), docKeyBase(doc));
  let out = reopened === tabs ? tabs : reopened.map((t, i) => (t === tabs[i] ? t : markResultsStale(t, revs)));
  out = bindMeasurementRefs(out, uid, doc.id);
  if (oldSession && oldSession !== doc.id) out = remapRuntimeKeys(out, { [oldSession]: doc.id });
  return out;
}

/** Fold a manifest's `docs` into knownDocs: what the workspace already knows
 *  (e.g. the path a document had when it closed here) wins; locate
 *  candidates are unioned. Documents open now are skipped. */
function mergeKnown(
  cur: Record<string, KnownDoc>, inc: Record<string, KnownDoc>, openUids: ReadonlySet<string>,
): Record<string, KnownDoc> {
  let out = cur;
  for (const [u, k] of Object.entries(inc)) {
    if (openUids.has(u)) continue;
    const prev = cur[u];
    const next: KnownDoc = { ...k };
    if (prev) {
      for (const [f, v] of Object.entries(prev)) if (v !== undefined) (next as Record<string, unknown>)[f] = v;
      const cands = [...(prev.candidates || []), ...(k.candidates || [])].filter((c, i, a) => a.indexOf(c) === i);
      if (cands.length) next.candidates = cands;
    }
    if (out === cur) out = { ...cur };
    out[u] = next;
  }
  return out;
}

/** The app version written as the manifest's `savedWith` (best effort). */
let appVersionP: Promise<string | undefined> | null = null;
function appVersion(): Promise<string | undefined> {
  if (!appVersionP) {
    appVersionP = (async () => {
      try {
        if (typeof window === "undefined" || !("__TAURI_INTERNALS__" in window)) return undefined;
        const { getVersion } = await import("@tauri-apps/api/app");
        return await getVersion();
      } catch { return undefined; }
    })();
  }
  return appVersionP;
}

// One-time cleanup of the graph's former session storage. Nothing to migrate:
// it never survived a launch (the first mount of every run wiped it).
try {
  localStorage.removeItem("mpfig.workflow_session");
  localStorage.removeItem("mpfig.analysis.uploads");
  sessionStorage.removeItem("mpfig.analysis.sessionMarker");
} catch { /* storage disabled */ }

// ── Store ───────────────────────────────────────────────────

type WorkspaceData = Pick<AnalysisWorkspaceState,
  "tabs" | "activeId" | "docUids" | "hydrated" | "savedSig" | "tombstones" | "uploads" | "knownDocs">;

/** One pristine tab saved in the active document. */
function initialState(): WorkspaceData {
  const docId = useCollageStore.getState().activeDocId;
  const uid = newDocUid();
  const first = newWorkflowTab("Workflow 1", docId ? uid : undefined);
  return {
    tabs: [first],
    activeId: first.id,
    docUids: docId ? { [docId]: uid } : {},
    hydrated: {},
    savedSig: {},
    tombstones: {},
    uploads: [],
    knownDocs: {},
  };
}

export const useAnalysisWorkspaceStore = create<AnalysisWorkspaceState>()((set, get) => {
  /** Read-only while an owner document is closed (the single lock guard). */
  const tabLocked = (t: WorkflowTab) => isLocked(t, workspaceOwnershipCtx(get()));
  return {
  ...initialState(),

  setTabs: (updater) => {
    const cur = get().tabs;
    const next = typeof updater === "function" ? updater(cur) : updater;
    if (next === cur) return;                       // same reference → no-op
    set({ tabs: nonEmpty(next, activeDocUid) });
  },

  setTabNodes: (tabId, updater, opts) => {
    get().setTabs((tabs) => {
      let touched = false;
      const next = tabs.map((t) => {
        if (t.id !== tabId) return t;
        // The updater always runs — the graph reads state through it (Run
        // graph's snapshot) — but a locked tab keeps its nodes.
        const nn = updater(t.nodes);
        if (nn === t.nodes) return t;
        if (!opts?.view && tabLocked(t)) return t;
        touched = true;
        return { ...t, nodes: nn };
      });
      return touched ? next : tabs;
    });
  },

  setTabEdges: (tabId, updater, opts) => {
    get().setTabs((tabs) => {
      let touched = false;
      const next = tabs.map((t) => {
        if (t.id !== tabId) return t;
        const ne = updater(t.edges);
        if (ne === t.edges) return t;
        if (!opts?.view && tabLocked(t)) return t;
        touched = true;
        return { ...t, edges: ne };
      });
      return touched ? next : tabs;
    });
  },

  setActive: (id) => {
    if (get().activeId !== id) set({ activeId: id });
  },

  uidFor: (docId) => {
    const cur = get().docUids[docId];
    if (cur) return cur;
    const uid = newDocUid();
    set((s) => ({ docUids: { ...s.docUids, [docId]: uid } }));
    return uid;
  },

  adoptUid: (docId, uid) => {
    const s = get();
    const prev = s.docUids[docId];
    if (!uid || prev === uid) return;
    let tabs = s.tabs;
    let tombstones = s.tombstones;
    if (prev) {
      let changed = false;
      const moved = tabs.map((t) => {
        const r = rebindDoc(t, prev, uid);
        if (r !== t) changed = true;
        return r;
      });
      if (changed) tabs = moved;
      if (tombstones[prev]) {
        const { [prev]: mine, ...rest } = tombstones;
        tombstones = { ...rest, [uid]: { ...(rest[uid] || {}), ...mine } };
      }
    }
    set({ docUids: { ...s.docUids, [docId]: uid }, tabs, tombstones });
  },

  deleteWorkflow: (wfId) => {
    const s = get();
    const idx = s.tabs.findIndex((t) => t.id === wfId);
    if (idx < 0) return false;
    const wf = s.tabs[idx];
    // Not while greyed out: the closed owner's file can't record the
    // deletion, and its copy would come back the next time it opens.
    if (tabLocked(wf)) return false;
    let tombstones = s.tombstones;
    if (!isPristineTab(wf)) {
      const ctx = workspaceOwnershipCtx(s);
      const open = new Set(ctx.docs.map((d) => d.uid));
      const owners = deriveOwnership(wf, ctx).owners.filter((u) => open.has(u));
      if (owners.length) {
        // Not older than the copy it deletes — merge drops updatedAt <= stone.
        const at = Math.max(Date.now(), wf.updatedAt ?? 0);
        tombstones = { ...tombstones };
        for (const u of owners) tombstones[u] = { ...(tombstones[u] || {}), [wfId]: at };
      }
    }
    const rest = s.tabs.filter((t) => t.id !== wfId);
    const tabs = nonEmpty(rest, activeDocUid);
    let activeId = s.activeId;
    if (!rest.length) activeId = tabs[0].id;
    else if (activeId === wfId) activeId = rest[Math.max(0, idx - 1)]?.id || rest[0].id;
    set({ tabs, activeId, tombstones });
    return true;
  },

  removeMissingSources: (wfId) => {
    const s = get();
    const wf = s.tabs.find((t) => t.id === wfId);
    if (!wf) return;
    const ctx = workspaceOwnershipCtx(s);
    const gone = missingOwners(wf, ctx);
    if (!gone.length) return;
    const open = new Set(ctx.docs.map((d) => d.uid));
    // A home that was one of them moves to an open owner, else the active document.
    const fallback = deriveOwnership(wf, ctx).owners.find((u) => open.has(u)) ?? activeDocUid();
    let next = wf;
    for (const u of gone) next = removeDocSources(next, u, ctx, fallback);
    next = markResultsStale(next, useSourceEditStore.getState().revs);
    if (next === wf) return;
    set({ tabs: get().tabs.map((t) => (t.id === wfId ? next : t)) });
  },

  askUploadHome: async (wfId, opts = {}) => {
    const s = get();
    const wf = s.tabs.find((t) => t.id === wfId);
    if (!wf || !hasUploads(wf)) return false;
    const ctx = workspaceOwnershipCtx(s);
    if (isLocked(wf, ctx)) return false;
    const own = deriveOwnership(wf, ctx);
    // A workflow reading a document's figures is saved with it (several:
    // its home, changed from the tab menu) — nothing to ask.
    if (own.participants.length) return false;
    if (!opts.force && (wf.homeConfirmed || askedUploadHome.has(wfId))) return false;
    askedUploadHome.add(wfId);
    const openDocs = useCollageStore.getState().openDocs;
    const current = ctx.docs.find((d) => d.uid === own.home)?.id;
    const choice = await chooseDocument({
      title: "Save this analysis in…",
      body: `"${wf.name}" uses uploaded images only — they aren't part of any figure document. `
        + "Choose the document this analysis is saved in (its uploaded images are stored in that .mpf):",
      options: openDocs.map((d) => ({
        id: d.id,
        label: opts.label?.(d.id) || d.name || "Untitled",
        detail: d.path ? basenameOf(d.path) : "not saved yet",
      })),
      selectedId: current,
      newLabel: "New document…",
      confirmLabel: "Save here",
    });
    if (!choice) return false;                                // Cancel keeps its home
    const docId = choice.kind === "new"
      ? useCollageStore.getState().docAdd({ path: null })    // a cold Untitled tab
      : choice.id;
    if (!openDoc(docId)) return false;                        // closed meanwhile
    return get().setUploadHome(wfId, get().uidFor(docId));
  },

  setUploadHome: (wfId, uid) => {
    const s = get();
    const wf = s.tabs.find((t) => t.id === wfId);
    if (!wf || !uid) return false;
    const ctx = workspaceOwnershipCtx(s);
    if (isLocked(wf, ctx)) return false;
    const own = deriveOwnership(wf, ctx);
    const allowed = own.participants.length ? own.participants : ctx.docs.map((d) => d.uid);
    if (!allowed.includes(uid)) return false;
    if (wf.home === uid && wf.homeConfirmed) return false;
    set({ tabs: s.tabs.map((t) => (t.id === wfId ? { ...t, home: uid, homeConfirmed: true } : t)) });
    return true;
  },

  addUploads: (entries) => {
    if (!entries.length) return;
    // A re-upload of the same filename replaced the backend image — refresh
    // its card (thumbnail / size) instead of keeping the stale one.
    const fresh = new Set(entries.map((e) => e.key));
    set((s) => ({ uploads: [...entries, ...s.uploads.filter((u) => !fresh.has(u.key))] }));
  },

  removeUpload: (key) => {
    const s = get();
    if (!s.uploads.some((u) => u.key === key)) return;
    set({ uploads: s.uploads.filter((u) => u.key !== key) });
  },

  rekeyDoc: (oldBase, newBase) => {
    if (!oldBase || oldBase === newBase) return;
    const s = get();
    const tabs = rekeyTabs(s.tabs, oldBase, newBase);
    if (tabs !== s.tabs) set({ tabs });
    const se = useSourceEditStore.getState();
    const revs = rekeyRecord(se.revs, oldBase, newBase);
    const rerun = rekeyRecord(se.rerun, oldBase, newBase);
    if (revs !== se.revs || rerun !== se.rerun) useSourceEditStore.setState({ revs, rerun });
    se.publishWithResults(se.withResults.map((k) => rekeyKey(k, oldBase, newBase)));
  },

  ensureDocLoaded: (docId, path) => {
    const inFlight = pendingLoads.get(docId);
    if (inFlight) return inFlight;
    const status = get().hydrated[docId];
    if (status && status !== "pending") return Promise.resolve();
    if (!path) {
      // Nothing on disk to read: an Untitled document starts with no saved analysis.
      if (openDoc(docId)) { hydratedPath.set(docId, null); setHydrated(docId, "done"); }
      return Promise.resolve();
    }
    setHydrated(docId, "pending");
    const gen = hydrateGen;
    // Strictly one file at a time (the read can re-ingest large uploads).
    const p: Promise<void> = hydrateChain
      .then(() => loadOne(docId, gen))
      .finally(() => { if (pendingLoads.get(docId) === p) pendingLoads.delete(docId); });
    hydrateChain = p.catch(() => undefined);
    pendingLoads.set(docId, p);
    return p;
  },

  hydrateDoc: (docId, payload) => {
    const status = get().hydrated[docId];
    if (status && status !== "pending") return;            // once per doc id
    const cs = useCollageStore.getState();
    const doc = cs.openDocs.find((d) => d.id === docId);
    if (!doc) return;                                       // closed meanwhile
    ensureOpenDocUids();
    hydratedPath.set(docId, doc.path ?? null);
    const m = payload.manifest;
    // The document is the one its file names: its doc.json uid (written by
    // every figure save), else the docUid its saved Analysis carries (older
    // files, and files an older build re-saved — that drops doc.json) — in
    // whatever format that Analysis is, so a newer-format file is still
    // recognised. Unless another open document already holds that uid (the
    // same file under a second path string, a Finder copy): then it keeps
    // its own, merge rule 3 keeps them apart, and its next save writes its
    // own uid into the file.
    const docJsonUid = typeof payload.doc_uid === "string" && payload.doc_uid ? payload.doc_uid : undefined;
    const manifestUid = m && typeof m.docUid === "string" && m.docUid ? m.docUid : undefined;
    const fileUid = docJsonUid ?? manifestUid;
    // The physical file read (docIdentity): what tells it from a copy later.
    const fileId = typeof payload.file_id === "string" && payload.file_id ? payload.file_id : undefined;
    if (fileUid) {
      const held = cs.openDocs.some((d) => d.id !== docId && get().docUids[d.id] === fileUid);
      if (!held) {
        get().adoptUid(docId, fileUid);
        if (doc.path) noteDocFile(fileUid, doc.path, doc.name, fileId);
      }
    }
    setFileUid(docId, fileUid ?? null, doc.path ?? null, fileId);
    if (payload.unsupported || (m && (m.format !== WORKFLOWS_FORMAT || typeof m.version !== "number"
        || m.version > WORKFLOWS_VERSION))) {
      // Written by a newer build (or unreadable): shown nowhere, never overwritten.
      setHydrated(docId, "unsupported");
      return;
    }
    fileHasSection.set(docId, !!m);
    // The file's two copies of its uid disagree — a Save As copy whose
    // Analysis was never rewritten still names the original in its manifest.
    // The doc.json one is the file's (the manifest's was mapped onto it by
    // deserializeDoc below); not clean, so the next persist rewrites the
    // manifest under it.
    const uidMismatch = !!docJsonUid && !!manifestUid && docJsonUid !== manifestUid;
    const s = get();
    const uid = s.uidFor(docId);
    const revs = useSourceEditStore.getState().revs;
    const docs = docRefs(cs.openDocs, s.docUids);
    const d = deserializeDoc(payload, { doc, uid, docs, known: s.knownDocs, revs });
    const knownDocs = mergeKnown(s.knownDocs, d.known, new Set(docs.map((x) => x.uid)));
    // Its labels (source names, measurement group labels, source folders —
    // utils/analysisMeta): the file's merge into the local ones (localStorage,
    // the mirror the panel reads) under the document's keys as they are now;
    // its restored upload tiles are filed in "Direct upload" like fresh ones.
    const base = docKeyBase(doc);
    const ownedUploads = new Set(uploadsOwnedBy(uid, { docs, known: knownDocs, tabs: d.tabs }).map(uploadKey));
    const fileMeta = m ? readDocMeta(d.meta, ownedUploads) : undefined;
    const localLabels = readAnalysisLabels();
    let labels = m ? mergeDocMeta(localLabels, base, fileMeta, ownedUploads) : localLabels;
    const restored = d.uploads.map((u) => u.key);
    labels = fileUnderFolder(labels, restored, DIRECT_UPLOAD_FOLDER, directFolderId);
    // The file AS READ — a losing or stale copy in it makes the doc dirty, so
    // its next persist rewrites it (plan §1.5); so do labels only this machine
    // has. A file with no saved analysis yet is compared with the labels as
    // they are here: just opening an older file never rewrites it (its labels
    // go in with its next analysis change).
    const sigMeta: DocMeta | undefined = m
      ? metaAsRead(fileMeta, restored.filter((k) => ownedUploads.has(k)))
      : docMetaOf(labels, base, ownedUploads);
    const sig = docSig(uid, { docs, known: knownDocs, tabs: d.tabs, revs, meta: metaFor(uid, metaSigForm(sigMeta)) });
    let tombstones = s.tombstones;
    if (Object.keys(d.tombstones).length) {
      const mine = { ...(tombstones[uid] || {}) };
      for (const [id, t] of Object.entries(d.tombstones)) if (!(mine[id] >= t)) mine[id] = t;
      tombstones = { ...tombstones, [uid]: mine };
    }
    const merged = mergeIncoming(s.tabs, d.tabs, tombstones, { docs, known: knownDocs });
    let tabs = merged.tabs;
    const before = new Set(s.tabs.map((t) => t.id));
    const added = tabs.filter((t) => !before.has(t.id));
    // The file brought workflows: the empty placeholder tab(s) make way.
    if (added.length) tabs = tabs.filter((t) => !before.has(t.id) || !isPristineTab(t));
    // Workflows the workspace kept while this document was closed (shared,
    // keyed "u:<uid>") come back under its keys — unlocked; their in-memory
    // results are flagged "needs re-run" (they were not editable meanwhile).
    tabs = reopenTabs(tabs, uid, doc, revs, s.knownDocs[uid]?.sessionId);
    if (!tabs.length) tabs = [newWorkflowTab("Workflow 1", uid)];
    const activeId = tabs.some((t) => t.id === s.activeId) ? s.activeId : (added[0] ?? tabs[0]).id;
    // Copies just read are "first seen": they keep the updatedAt they were saved with.
    for (const t of d.tabs) seen.delete(t.id);
    for (const id of Object.values(merged.reminted)) seen.delete(id);
    set({
      tabs, activeId, knownDocs, tombstones,
      savedSig: { ...get().savedSig, [docId]: uidMismatch ? `${sig}|uid-mismatch` : sig },
      hydrated: { ...get().hydrated, [docId]: "done" },
    });
    if (d.uploads.length) get().addUploads(d.uploads);
    writeAnalysisLabels(labels, localLabels);
  },

  persistDocWorkflows: (docId, opts = {}) => {
    // One write at a time, in call order: a later snapshot never lands
    // before an earlier one (savedSig stays truthful).
    const run = () => persistNow(docId, opts);
    const p = persistChain.then(run, run);
    persistChain = p.then(() => undefined, () => undefined);
    return p;
  },

  afterDocSaved: async (docId, oldPath, newPath, opts = {}) => {
    // A read of the previous file still queued: its workflows come in first,
    // under the old key base, so the re-key below moves them too.
    const pending = pendingLoads.get(docId);
    if (pending) await pending;
    const cs = useCollageStore.getState();
    const doc = cs.openDocs.find((d) => d.id === docId);
    if (!doc) return;
    const prev = oldPath || null;
    const moved = prev !== newPath;
    const uidBefore = get().docUids[docId];
    // The uid its previous file named. Another one than the tab's: that file
    // is another document's (a Finder copy opened while its original was
    // open keeps a uid of its own — the held rule).
    const prevFileUid = fileUidOf(docId);
    // The uid the file was written under (its doc.json), as the sidecar chose
    // it: the tab's own when it was sent, the file's when it wasn't (a tab
    // whose file was never read), a new one for a Save As copy whose
    // original is still there. The tab takes it unless another open tab
    // holds it. An older sidecar names none: then a Save As still makes the
    // copy a new document here, and the old file keeps its uid.
    const fileUid = opts.docUid || undefined;
    if (fileUid) {
      const held = cs.openDocs.some((d) => d.id !== docId && get().docUids[d.id] === fileUid);
      if (!held && uidBefore !== fileUid) get().adoptUid(docId, fileUid);
    } else if (moved && prev) {
      get().adoptUid(docId, newDocUid());
    }
    if (moved) {
      const oldBase = docKeyBase({ id: docId, path: prev });
      const newBase = docKeyBase({ id: docId, path: newPath });
      // Recorded before the tab moves, so the openDocs subscriber doesn't
      // take the new path for a file whose workflows were never read.
      hydratedPath.set(docId, newPath);
      if (doc.path !== newPath) cs.docSetPath(docId, newPath);
      get().rekeyDoc(oldBase, newBase);
      // A copy — its previous file is still there — keeps its labels on the
      // old file too. The sidecar says whether it is (sourceKept); an older
      // one doesn't, and then the uid tells: it changed (a Save As copy), or
      // the previous file named another document (a held tab keeps its own
      // uid on Save As, so its uid alone can't tell). Its old file gone, the
      // document moved: the labels move with it, and the last-seen table no
      // longer has it at the old path.
      const copy = !!prev && (opts.sourceKept ?? (get().docUids[docId] !== uidBefore
        || (!!prevFileUid && prevFileUid !== uidBefore)));
      rekeyAnalysisStorage(oldBase, newBase, copy);
      if (!copy && prev && registryUidAt(prev) === get().docUids[docId]) forgetDocFile(prev);
      // The new file holds what the save carried, not what savedSig describes.
      set((s) => {
        const { [docId]: _gone, ...savedSig } = s.savedSig;
        return { savedSig };
      });
    }
    if (fileUid) {
      setFileUid(docId, fileUid, newPath, opts.fileId);
      if (get().docUids[docId] === fileUid) noteDocFile(fileUid, newPath, openDoc(docId)?.name, opts.fileId);
      // Collage figures placed from the file at newPath follow the uid it now
      // carries — how items added before uids get one, how they stay with a
      // file re-identified by this save, and (a save over a file already in
      // the collage — an Untitled tab or a Save As onto it) how they keep
      // naming what is at that path, never a same-uid copy of what was there.
      // Figures of the previous file (a Save As) sit on its path: untouched.
      followFileUid(newPath, fileUid);
    } else if (moved) {
      // Not known until the persist below writes the uid into its manifest.
      clearFileUid(docId);
    }
    // The save couldn't bring the file's saved Analysis along (its previous
    // file was gone or unreadable): the new file lacks it while savedSig still
    // says "clean" — write it again from the workspace.
    const lost = !moved && opts.workflowsCarried === false && fileHasSection.get(docId) === true;
    if (lost) fileHasSection.set(docId, false);
    await get().persistDocWorkflows(docId, { force: moved || lost });
  },

  docMoved: (oldPath, newPath, opts = {}) => {
    if (!oldPath || !newPath || oldPath === newPath) return;
    const cs = useCollageStore.getState();
    // Every tab on the old path moves together: they share its key base
    // (opts.ids: only those, see above).
    const moving = cs.openDocs.filter((d) => d.path === oldPath && (!opts.ids || opts.ids.includes(d.id)));
    if (!moving.length) return;
    // Signatures carry each document's own name, which a rename changes: a
    // document whose file is up to date is signed again afterwards so it
    // stays up to date (the file's stored name for itself is never read).
    // One with unsaved Analysis keeps its old signature — still unsaved.
    syncWorkspaceNow();                                     // signed as persist signs
    const s0 = get();
    const upToDate = new Set(moving.filter((d) => !opts.reread && s0.hydrated[d.id] === "done"
      && s0.savedSig[d.id] !== undefined && s0.savedSig[d.id] === persistSnapshot(d).sig).map((d) => d.id));
    const oldName = basenameOf(oldPath).replace(/\.mpf$/i, "");
    for (const d of moving) {
      // Recorded before the tab moves, so the openDocs subscriber doesn't
      // forget the read (it is the same file, already read). A read that
      // failed — most often because the file was already gone from the old
      // path — is not carried over: the subscriber forgets it and the file
      // is read again where it is now; and so is a file not known to be the
      // one that was read (opts.reread): what it holds is read and merged
      // before anything is written to it, and until then a figure save
      // leaves its uid alone.
      if (!opts.reread && hydratedPath.get(d.id) === oldPath && s0.hydrated[d.id] !== "failed") {
        hydratedPath.set(d.id, newPath);
      }
      // A tab named after its file is renamed with it; a name the user gave stays.
      cs.docSetPath(d.id, newPath, d.name === oldName ? undefined : d.name);
    }
    const oldBase = docKeyBase({ id: moving[0].id, path: oldPath });
    const newBase = docKeyBase({ id: moving[0].id, path: newPath });
    get().rekeyDoc(oldBase, newBase);
    // The old path holds nothing any more: the labels move, not copy.
    rekeyAnalysisStorage(oldBase, newBase, false);
    if (upToDate.size) {
      syncWorkspaceNow();
      const savedSig = { ...get().savedSig };
      for (const id of upToDate) {
        const d = openDoc(id);
        if (d) savedSig[id] = persistSnapshot(d).sig;
      }
      set({ savedSig });
    }
  },

  closeDoc: (docId) => {
    const cs = useCollageStore.getState();
    const doc = cs.openDocs.find((d) => d.id === docId);
    if (doc) closeDocState(doc, cs.openDocs.filter((d) => d.id !== docId));
  },

  openMissing: async (wfId) => {
    if (lookingFor.has(wfId)) return;                      // a second click while it runs
    lookingFor.add(wfId);
    try {
      const s = get();
      const wf = s.tabs.find((t) => t.id === wfId);
      if (!wf) return;
      const ctx = workspaceOwnershipCtx(s);
      const missing = missingOwners(wf, ctx);
      if (!missing.length) return;
      const holders = ownerPaths(wf, ctx);
      const cands = new Map(missing.map((u) => [u, missingDocCandidates(s.knownDocs[u], holders)]));
      const found = await findMissingDocs(missing, cands, holders, s.knownDocs);
      for (const uid of missing) {
        if (!stillMissing(wfId, uid)) continue;           // opened meanwhile
        const hit = found.get(uid);
        // openLocated still checks the uid the file names once it is read.
        if (hit) await openLocated(wfId, uid, hit);
        // Saved somewhere, just not where it was: ask. A document that was
        // never saved has no file to look for ("Remove their sources").
        else if ((cands.get(uid) || []).length || registryPathsOf(uid).length) await get().locate(wfId, uid);
      }
    } finally {
      lookingFor.delete(wfId);
    }
  },

  locate: async (wfId, uid) => {
    if (!stillMissing(wfId, uid)) return;
    const s = get();
    const wf = s.tabs.find((t) => t.id === wfId)!;
    const holders = ownerPaths(wf, workspaceOwnershipCtx(s));
    const known = s.knownDocs[uid];
    const startIn = holders[0] ? dirnameOf(holders[0]) : (known?.path ? dirnameOf(known.path) : undefined);
    let picked: unknown;
    try {
      const dialog = await import("@tauri-apps/plugin-dialog");
      picked = await dialog.open({
        title: `Locate ${closedDocLabel(uid, s.knownDocs)}`,
        multiple: false,
        directory: false,
        filters: [{ name: "Project", extensions: ["mpf"] }],
        ...(startIn ? { defaultPath: startIn } : {}),
      });
    } catch (e) {
      console.warn("[analysis] locate: the file dialog is unavailable", e);
      return;
    }
    const path = typeof picked === "string" ? picked
      : picked && typeof picked === "object" && typeof (picked as { path?: unknown }).path === "string"
        ? (picked as { path: string }).path : null;
    if (!path || !stillMissing(wfId, uid)) return;         // cancelled, or opened meanwhile
    await openLocated(wfId, uid, path);
  },
  };
});

// ── Hydrate / persist / close machinery ─────────────────────

/** Reads queued or in flight, by doc id (ensureDocLoaded). */
const pendingLoads = new Map<string, Promise<void>>();
/** The path each document's workflows were read from (null: Untitled) —
 *  a path that changes any other way than a save is read again. */
const hydratedPath = new Map<string, string | null>();
let hydrateChain: Promise<void> = Promise.resolve();
/** Documents a persist (close / quit / save) is waiting to see read: their
 *  read stops retrying an unanswered sidecar and fails instead. */
const awaitedLoads = new Set<string>();
/** Bumped by the test reset: queued reads of an earlier generation drop out. */
let hydrateGen = 0;
let persistChain: Promise<void> = Promise.resolve();
// Doc id → the uid its file names (doc.json, else its manifest's docUid):
// utils/docIdentity's fileUidOf / setFileUid — the collage and the
// document tabs need it too.
/** Doc id → its file holds a workflows/ section (read, or written since). */
const fileHasSection = new Map<string, boolean>();
/** Workflows openMissing is looking for documents of (no double run). */
const lookingFor = new Set<string>();

function openDoc(docId: string): DocTab | undefined {
  return useCollageStore.getState().openDocs.find((d) => d.id === docId);
}

function setHydrated(docId: string, status: HydrationStatus): void {
  const cur = useAnalysisWorkspaceStore.getState().hydrated;
  if (cur[docId] !== status) useAnalysisWorkspaceStore.setState({ hydrated: { ...cur, [docId]: status } });
}

/** Forget that document `docId` was read (it closed, or now names another file). */
function forgetHydration(docId: string): void {
  pendingLoads.delete(docId);
  hydratedPath.delete(docId);
  clearFileUid(docId);
  fileHasSection.delete(docId);
  const s = useAnalysisWorkspaceStore.getState();
  if (!(docId in s.hydrated) && !(docId in s.savedSig)) return;
  const { [docId]: _h, ...hydrated } = s.hydrated;
  const { [docId]: _s, ...savedSig } = s.savedSig;
  useAnalysisWorkspaceStore.setState({ hydrated, savedSig });
}

/** A read the sidecar did not answer (still starting — tabs seeded from the
 *  collage appear before it is up — or restarting) is retried; a reply about
 *  the file itself (missing, not an .mpf) is final. */
export const READ_RETRY_MS = 2000;
const READ_RETRIES = 30;
function isSidecarReply(e: unknown): boolean {
  return /^API error:/.test(e instanceof Error ? e.message : String(e ?? ""));
}

/** One queued read: the document's file as it is named NOW (read again if
 *  it moved while being read); nothing happens if it closed or the
 *  workspace was reset meanwhile. */
async function loadOne(docId: string, gen: number): Promise<void> {
  const store = useAnalysisWorkspaceStore;
  const live = () => gen === hydrateGen && !!openDoc(docId) && store.getState().hydrated[docId] === "pending";
  let retries = 0;
  for (;;) {
    if (!live()) return;
    const path = openDoc(docId)!.path;
    if (!path) {
      hydratedPath.set(docId, null);
      setHydrated(docId, "done");
      return;
    }
    let resp: WorkflowsReadResponse | null = null;
    let failure: unknown = null;
    try { resp = await api.readWorkflows(path); } catch (e) { failure = e ?? new Error("read failed"); }
    if (!live()) return;
    if (openDoc(docId)!.path !== path) continue;
    if (!resp && !isSidecarReply(failure) && retries < READ_RETRIES && !awaitedLoads.has(docId)) {
      retries++;
      await new Promise((r) => setTimeout(r, READ_RETRY_MS));
      continue;
    }
    if (!resp) {
      console.error("[analysis] could not read the saved workflows of", path, failure);
      // Never overwritten: a write now could drop analysis the file holds.
      hydratedPath.set(docId, path);
      setHydrated(docId, "failed");
      return;
    }
    // Renamed or moved while it was closed: the last-seen table saw its uid
    // at another path, which is gone now. The labels this machine keeps for
    // it under that path (source names, measurement group labels, source
    // folders) follow it — before hydrateDoc compares them with the file, so
    // they count as its own. Not when the old file is still there (a copy) or
    // open in another tab; a failed check moves nothing.
    const uid = fileUidIn(resp);
    const old = uid ? registryPathsOf(uid).find((p) => p !== path) : undefined;
    if (uid && old && registryUidAt(path) !== uid
        && !useCollageStore.getState().openDocs.some((d) => d.path === old)) {
      let gone = false;
      try {
        gone = (await api.resolveMpf([{ key: "old", path: old }], { scan: false })).old?.exists === false;
      } catch { gone = false; }
      if (!live()) return;
      if (openDoc(docId)!.path !== path) continue;
      if (gone) {
        rekeyAnalysisStorage(docKeyBase({ id: docId, path: old }), docKeyBase({ id: docId, path }), false);
        forgetDocFile(old);
      }
    }
    store.getState().hydrateDoc(docId, resp);
    return;
  }
}

/** The uid a read file names: its doc.json uid, else its manifest's docUid. */
function fileUidIn(resp: WorkflowsReadResponse): string | undefined {
  if (typeof resp.doc_uid === "string" && resp.doc_uid) return resp.doc_uid;
  const m = resp.manifest;
  return m && typeof m.docUid === "string" && m.docUid ? m.docUid : undefined;
}

async function persistNow(docId: string, opts: PersistOptions): Promise<PersistOutcome> {
  const store = useAnalysisWorkspaceStore;
  let doc = openDoc(docId);
  if (!doc?.path) return "no-path";
  // Its file must have been read first — a write would replace what it holds.
  if (!store.getState().hydrated[docId]) void store.getState().ensureDocLoaded(docId, doc.path);
  const pending = pendingLoads.get(docId);
  if (pending) {
    awaitedLoads.add(docId);
    try { await pending; } finally { awaitedLoads.delete(docId); }
  }
  const status = store.getState().hydrated[docId];
  if (status === "unsupported") return "unsupported";
  doc = openDoc(docId);
  if (!doc?.path) return "no-path";
  if (status !== "done" || hydratedPath.get(docId) !== doc.path) return "not-hydrated";
  let path = doc.path;
  const version = await appVersion();
  syncWorkspaceNow();                                       // updatedAt / home current
  let snap = persistSnapshot(doc, version);
  if (!opts.force && store.getState().savedSig[docId] === snap.sig) return "clean";
  const payloadOf = (sn: PersistSnapshot) => serializeForDoc(
    sn.uid, { ...sn.state, meta: metaFor(sn.uid, sn.meta) }, { figureDiscarded: opts.figureDiscarded });
  // A file whose doc.json names another uid than the one this manifest is
  // written under (a Finder copy the held rule gave a uid of its own): its
  // doc.json follows in the same write, so the file's two copies of its uid
  // never disagree — read again, it is this document, not the original.
  // The uid the file named (expectUid): a file at the path naming another
  // (another document has its name now) is never written — the sidecar
  // refuses, and the tab looks for its own file as for a missing one.
  const write = (p: string, sn: PersistSnapshot) => {
    const f = fileUidOf(docId);
    return f && f !== sn.uid
      ? api.writeWorkflows(p, payloadOf(sn), { docUid: sn.uid, expectUid: f })
      : f ? api.writeWorkflows(p, payloadOf(sn), { expectUid: f })
        : api.writeWorkflows(p, payloadOf(sn));
  };
  let res: WorkflowsWriteResult;
  try {
    res = await write(path, snap);
  } catch (e) {
    // Its file is no longer where the tab says — renamed or moved outside the
    // app while open (or another document now has its name). Found by its
    // uid, the tab follows it (docMoved) and the write goes there, once;
    // signed again, since the tab's name may have changed with the file's.
    // Not found: the failure stands (close / quit report it). Imported
    // lazily: docLinks imports this store.
    if (!isMissingFileError(e) && !isOtherDocumentError(e)) throw e;
    const { relocateOpenDoc } = await import("../utils/docLinks");
    const found = await relocateOpenDoc(docId).catch(() => null);
    if (!found) throw e;
    path = found;
    snap = persistSnapshot(openDoc(docId) ?? doc, version);
    res = await write(path, snap);
  }
  if (!res.ok) {
    if ("unsupported" in res) { setHydrated(docId, "unsupported"); return "unsupported"; }
    throw new Error(`the file is missing ${res.missing.length} stored result(s) expected to be kept`);
  }
  if (res.missing_uploads?.length) {
    console.warn("[analysis] uploads not embedded (no original kept):", res.missing_uploads);
  }
  if (openDoc(docId)?.path === path) {
    // Results stored "needs re-run" for a discarded figure: not what the live
    // workspace signs as, so a later persist (figure saved after all) rewrites.
    const saved = opts.figureDiscarded ? `${snap.sig}|figure-discarded` : snap.sig;
    store.setState((st) => ({ savedSig: { ...st.savedSig, [docId]: saved } }));
    fileHasSection.set(docId, true);
    // The uid the file names now: its doc.json's, as the sidecar reports it
    // (rewritten to this one when it named another), else — no doc.json —
    // its manifest's, just written: this one. An older sidecar reports
    // none: a file that named no uid names this one now.
    const now = fileUidOf(docId);
    const onDisk = res.doc_uid !== undefined ? (res.doc_uid || snap.uid) : (now ?? snap.uid);
    if (onDisk !== now) {
      setFileUid(docId, onDisk, path, res.file_id);
      if (onDisk === snap.uid) noteDocFile(snap.uid, path, openDoc(docId)?.name, res.file_id);
    } else {
      // The same uid in a new physical file (every write replaces it).
      setFileIdOf(docId, res.file_id);
      refreshDocFileId(path, onDisk, res.file_id);
    }
  }
  return "written";
}

/** What persistNow writes for open document `doc` now, and its signature. */
interface PersistSnapshot {
  uid: string;
  state: PersistState;
  meta: DocMeta | undefined;
  sig: string;
}

/** The workspace as document `doc`'s file would hold it now (persistNow;
 *  docMoved signs with it too). */
function persistSnapshot(doc: DocTab, version?: string): PersistSnapshot {
  const s = useAnalysisWorkspaceStore.getState();
  const uid = s.uidFor(doc.id);
  const state: PersistState = {
    docs: workspaceOwnershipCtx(s).docs,
    known: s.knownDocs,
    tabs: s.tabs,
    revs: useSourceEditStore.getState().revs,
    tombstones: s.tombstones,
    ...(version ? { appVersion: version } : {}),
  };
  // Its labels as the panel holds them now (localStorage): those keyed under
  // the document, and those of the uploads it embeds — signed order- and
  // folder-id-insensitively, written as they are.
  const ownedUploads = new Set(uploadsOwnedBy(uid, state).map(uploadKey));
  const meta = docMetaOf(readAnalysisLabels(), docKeyBase(doc), ownedUploads);
  const sig = docSig(uid, { ...state, meta: metaFor(uid, metaSigForm(meta)) });
  return { uid, state, meta, sig };
}

/** Collage figures of the file at `path` carry document `uid` (it was just
 *  saved under it). */
function followFileUid(path: string, uid: string): void {
  const cs = useCollageStore.getState();
  for (const it of cs.items) {
    if (it.projectPath === path && it.projectUid !== uid) cs.updateItem(it.id, { projectUid: uid });
  }
}

/** Document `doc` leaves the open set (`others` stay). Its single-owner
 *  workflows — and shared ones no other open document owns — leave the
 *  workspace (each owner wrote its copy at its own close); shared ones still
 *  open elsewhere are re-keyed to "u:<uid>" (locked until it reopens). */
function closeDocState(doc: DocTab, others: readonly DocTab[]): void {
  const store = useAnalysisWorkspaceStore;
  const s = store.getState();
  const uid = s.docUids[doc.id];
  forgetHydration(doc.id);
  if (!uid) return;
  const ctxOpen: OwnershipCtx = { docs: docRefs([...others, doc], s.docUids), known: s.knownDocs };
  const stillOpen = new Set(docRefs(others, s.docUids).map((d) => d.uid));
  stillOpen.delete(uid);
  const locking = new Set<string>();
  const keep = s.tabs.filter((wf) => {
    const owners = deriveOwnership(wf, ctxOpen).owners;
    if (!owners.includes(uid)) return true;
    if (owners.some((u) => u !== uid && stillOpen.has(u))) { locking.add(wf.id); return true; }
    return false;
  });
  let tabs = keep.length === s.tabs.length ? s.tabs : keep;
  let knownDocs = s.knownDocs;
  if (locking.size) {
    // Legacy tables bound only by session id get the uid first, so they still
    // name this document once it is gone.
    tabs = bindMeasurementRefs(tabs, uid, null, doc.id);
    tabs = rekeyTabs(tabs, docKeyBase(doc), closedKeyBase(uid));
    tabs = tabs.map((t) => (locking.has(t.id) ? settleRunning(t) : t));
    const entry: KnownDoc = { ...(s.knownDocs[uid] || {}), name: doc.name, sessionId: doc.id };
    if (doc.path) entry.path = doc.path;
    knownDocs = { ...knownDocs, [uid]: entry };
  }
  // A replacement placeholder gets no home: the sync pass gives it whichever
  // document is active once this one is gone.
  if (!tabs.length) tabs = [newWorkflowTab("Workflow 1")];
  const activeId = tabs.some((t) => t.id === s.activeId) ? s.activeId : tabs[0].id;
  const { [doc.id]: _u, ...docUids } = store.getState().docUids;
  store.setState({ tabs, activeId, knownDocs, docUids });
}

// ── Finding a locked workflow's closed documents ────────────

/** Files of the workflow's open owners — where its other documents' files
 *  are looked for (relative path, same name next to them). */
function ownerPaths(wf: WorkflowTab, ctx: OwnershipCtx): string[] {
  const out: string[] = [];
  for (const u of deriveOwnership(wf, ctx).owners) {
    const p = ctx.docs.find((d) => d.uid === u)?.path;
    if (p && !out.includes(p)) out.push(p);
  }
  return out;
}

/** Where closed document `k` may be, best first: the candidates its
 *  manifest entry gave (saved path, path relative to the file that named
 *  it, same name next to that file), then the same name next to each open
 *  owner's file. Empty for a document that was never saved. */
function missingDocCandidates(k: KnownDoc | undefined, holders: readonly string[]): string[] {
  const out: string[] = [];
  const add = (p: string | undefined) => { if (p && !out.includes(p)) out.push(p); };
  for (const c of k?.candidates || []) add(c);
  add(k?.path);
  for (const h of holders) for (const c of locateCandidates(h, { path: k?.path })) add(c);
  return out;
}

/** Where each closed document in `missing` is now, by uid → path (one
 *  api.resolveMpf call): the first of its candidates (`cands`) when that file
 *  is still there, else the file carrying its uid among the paths the
 *  last-seen table saw it at, the other candidates, and the folders of all
 *  of those, of its owners' files and of its saved path. Found nowhere (or a
 *  sidecar without that lookup): the first candidate that exists, uid or not
 *  (one api.pathsExist call, as before) — a file re-saved by an older build
 *  names none, and openLocated takes it. Several files carrying the uid: none
 *  is picked (the user locates it). A document never saved is not looked for. */
async function findMissingDocs(
  missing: readonly string[], cands: ReadonlyMap<string, string[]>, holders: readonly string[],
  known: Readonly<Record<string, KnownDoc>>,
): Promise<Map<string, string>> {
  const out = new Map<string, string>();
  const refs: MpfResolveRef[] = [];
  for (const uid of missing) {
    const tried = cands.get(uid) || [];
    const seen = registryPathsOf(uid);
    const path = tried[0] ?? seen[0];
    if (!path) continue;
    const candidates = [...seen, ...tried].filter((p, i, a) => p !== path && a.indexOf(p) === i);
    const dirs = [...holders, ...(known[uid]?.path ? [known[uid].path!] : [])]
      .map(dirnameOf).filter((d, i, a) => !!d && a.indexOf(d) === i);
    refs.push({ key: uid, path, uid, ...(candidates.length ? { candidates } : {}), ...(dirs.length ? { dirs } : {}) });
  }
  if (!refs.length) return out;
  let results: Record<string, MpfResolveResult> | null = null;
  try { results = await api.resolveMpf(refs); } catch { results = null; }
  const byName: string[] = [];
  for (const r of refs) {
    const res = results?.[r.key];
    if (res && (res.status === "ok" || res.status === "moved") && res.path) out.set(r.key, res.path);
    else if (!res || res.status === "missing") byName.push(r.key);
  }
  const paths = [...new Set(byName.flatMap((u) => cands.get(u) || []))];
  if (paths.length) {
    let exists: Record<string, boolean> = {};
    try { exists = await api.pathsExist(paths); } catch { exists = {}; }
    for (const u of byName) {
      const hit = (cands.get(u) || []).find((p) => exists[p]);
      if (hit) out.set(u, hit);
    }
  }
  return out;
}

/** Whether workflow `wfId` still waits for closed document `uid`. */
function stillMissing(wfId: string, uid: string): boolean {
  const s = useAnalysisWorkspaceStore.getState();
  const wf = s.tabs.find((t) => t.id === wfId);
  return !!wf && missingOwners(wf, workspaceOwnershipCtx(s)).includes(uid);
}

/** Open `path` for closed document `uid` of workflow `wfId`: a cold tab (no
 *  builder swap, no doc op — like a session-restore tab) whose read puts the
 *  workflows holding "u:<uid>" back under it when its file names `uid`. A
 *  file naming no uid (re-saved by an older build) is taken as that
 *  document; one naming another uid only once the user confirms (then
 *  everything holding `uid` is re-bound to it). Resolves whether `uid` is
 *  open now. */
async function openLocated(wfId: string, uid: string, path: string): Promise<boolean> {
  const store = useAnalysisWorkspaceStore;
  const docId = useCollageStore.getState().docEnsure(path);
  await store.getState().ensureDocLoaded(docId, path);
  if (!stillMissing(wfId, uid)) return true;               // its read resolved it
  let doc = openDoc(docId);
  if (!doc) return false;
  const s = store.getState();
  const heldElsewhere = () => useCollageStore.getState().openDocs
    .some((d) => d.id !== docId && store.getState().docUids[d.id] === uid);
  if (heldElsewhere()) return false;
  if (s.hydrated[docId] !== "done") {
    // Its saved analysis couldn't be read (or is a newer format): there's no
    // telling which document it is, and it could never save this workflow.
    console.warn("[analysis] can't use", path, "for a closed document: its analysis is", s.hydrated[docId]);
    return false;
  }
  if (fileUidOf(docId) === null) {
    s.adoptUid(docId, uid);
    reopenIn(uid, doc, s.knownDocs[uid]?.sessionId);
    return true;
  }
  const expected = closedDocLabel(uid, s.knownDocs);
  const name = basenameOf(path);
  const ok = await confirmDialog({
    title: "Use this document?",
    body: `"${name}" doesn't look like the document this analysis was saved with`
      + `${expected !== name ? ` (${expected})` : ""} — it may be a different figure, or a copy saved separately.`
      + `\n\nUse it anyway?`,
    confirmLabel: "Use it",
  });
  if (!ok) return false;
  doc = openDoc(docId);
  const now = store.getState();
  const to = now.docUids[docId];
  if (!doc || !to || !stillMissing(wfId, uid) || heldElsewhere()) return false;
  // Everything that named the missing document names this one instead.
  let changed = false;
  const rebound = now.tabs.map((t) => {
    const r = rebindDoc(t, uid, to);
    if (r !== t) changed = true;
    return r;
  });
  const tabs = reopenTabs(changed ? rebound : now.tabs, to, doc, useSourceEditStore.getState().revs,
    now.knownDocs[uid]?.sessionId);
  if (tabs !== now.tabs) store.setState({ tabs });
  return true;
}

/** reopenTabs on the workspace: closed document `uid` is open as `doc` now. */
function reopenIn(uid: string, doc: DocTab, oldSession?: string): void {
  const store = useAnalysisWorkspaceStore;
  const s = store.getState();
  const tabs = reopenTabs(s.tabs, uid, doc, useSourceEditStore.getState().revs, oldSession);
  if (tabs !== s.tabs) store.setState({ tabs });
}

// ── Document uids ───────────────────────────────────────────

/** Give every open document a uid (one store write for all that lack one). */
function ensureOpenDocUids(): void {
  const { docUids } = useAnalysisWorkspaceStore.getState();
  const missing = useCollageStore.getState().openDocs.filter((d) => !docUids[d.id]);
  if (!missing.length) return;
  const next = { ...docUids };
  for (const d of missing) next[d.id] = newDocUid();
  useAnalysisWorkspaceStore.setState({ docUids: next });
}

// ── Debounced sync: updatedAt + home ────────────────────────

export const SYNC_DEBOUNCE_MS = 500;
let syncTimer: ReturnType<typeof setTimeout> | null = null;
let applying = false;
/** Per workflow id: the tab object last examined and its contentSig. */
const seen = new Map<string, { tab: WorkflowTab; sig: string }>();
/** The context those signatures were computed in. */
let seenCtx: { docs: string; known: unknown; revs: unknown } | null = null;

function scheduleSync(): void {
  if (applying) return;
  if (syncTimer) clearTimeout(syncTimer);
  syncTimer = setTimeout(syncWorkspaceNow, SYNC_DEBOUNCE_MS);
}

/** Run the debounced pass now (persist calls it before taking a signature):
 *    • `updatedAt` — bumped when a workflow's contentSig changed since the
 *      last pass, and never while it is locked. A workflow first seen with no
 *      updatedAt (made this session) gets one unless pristine; one first seen
 *      WITH one (read from a file) keeps it, so a freshly read file is clean.
 *    • `home` — written back when deriveOwnership moves it; a workflow with
 *      no figure data and no home is given the active document. */
export function syncWorkspaceNow(): void {
  if (syncTimer) { clearTimeout(syncTimer); syncTimer = null; }
  ensureOpenDocUids();
  const st = useAnalysisWorkspaceStore.getState();
  const ctx = workspaceOwnershipCtx(st);
  const revs = useSourceEditStore.getState().revs;
  const docsKey = ctx.docs.map((d) => `${d.id}\u0000${d.uid}\u0000${d.path ?? ""}\u0000${d.name ?? ""}`).join("\u0001");
  const sameCtx = !!seenCtx && seenCtx.docs === docsKey && seenCtx.known === ctx.known && seenCtx.revs === revs;
  seenCtx = { docs: docsKey, known: ctx.known, revs };
  const activeId = useCollageStore.getState().activeDocId;
  const fallbackHome = activeId ? st.docUids[activeId] : undefined;
  const now = Date.now();
  let changed = false;
  const live = new Set<string>();
  const next = st.tabs.map((t) => {
    live.add(t.id);
    const prev = seen.get(t.id);
    let out = t;
    const own = deriveOwnership(t, ctx);
    // An untouched placeholder tab belongs to whichever document is active —
    // whatever the user builds in it is then saved with the figure on screen.
    const home = isPristineTab(t) && !t.homeConfirmed && fallbackHome ? fallbackHome : (own.home ?? fallbackHome);
    if (home && home !== t.home) out = { ...out, home };
    // Signed WITH the home it is about to get, so the next pass (which sees
    // `out`) compares like with like.
    const sig = prev && sameCtx && prev.tab === t
      ? prev.sig
      : contentSig(out, { docs: ctx.docs, known: ctx.known, revs });
    if (!isLocked(out, ctx)) {
      const bump = prev ? prev.sig !== sig : (t.updatedAt === undefined && !isPristineTab(t));
      if (bump) out = { ...out, updatedAt: Math.max(now, (t.updatedAt ?? 0) + 1) };
    }
    seen.set(t.id, { tab: out, sig });
    if (out !== t) changed = true;
    return out;
  });
  for (const id of [...seen.keys()]) if (!live.has(id)) seen.delete(id);
  if (!changed) return;
  applying = true;
  try { useAnalysisWorkspaceStore.setState({ tabs: next }); }
  finally { applying = false; }
}

/** Tests: a fresh workspace (one pristine tab for the active document),
 *  nothing remembered by the sync pass, no read queued or document marked
 *  read (call ensureDocLoaded / hydrateOpenDocs to read again). */
export function resetAnalysisWorkspaceForTests(): void {
  if (syncTimer) { clearTimeout(syncTimer); syncTimer = null; }
  seen.clear();
  seenCtx = null;
  hydrateGen++;
  pendingLoads.clear();
  awaitedLoads.clear();
  hydratedPath.clear();
  resetDocIdentityForTests();
  fileHasSection.clear();
  lookingFor.clear();
  promptedLocked.clear();
  askedUploadHome.clear();
  hydrateChain = Promise.resolve();
  persistChain = Promise.resolve();
  applying = true;
  try { useAnalysisWorkspaceStore.setState(initialState()); }
  finally { applying = false; }
  ensureOpenDocUids();
  if (syncTimer) { clearTimeout(syncTimer); syncTimer = null; }
}

/** Queue a read of every open document not read yet (Untitled ones are done
 *  at once). Called whenever openDocs changes, so Open, session-restore cold
 *  tabs and "Open missing" all hydrate the same way. */
export function hydrateOpenDocs(): void {
  const ws = useAnalysisWorkspaceStore.getState();
  for (const d of useCollageStore.getState().openDocs) {
    if (!ws.hydrated[d.id] || (ws.hydrated[d.id] === "pending" && !pendingLoads.has(d.id))) {
      void ws.ensureDocLoaded(d.id, d.path);
    }
  }
}

useAnalysisWorkspaceStore.subscribe((s, p) => {
  if (s.tabs !== p.tabs || s.docUids !== p.docUids || s.knownDocs !== p.knownDocs) scheduleSync();
});
useCollageStore.subscribe((s, p) => {
  if (s.openDocs === p.openDocs) return;
  ensureOpenDocUids();
  // A tab removed without closeDoc (e.g. session restore dropping the blank
  // tab): the workspace lets go of it the same way.
  for (const d of p.openDocs) {
    if (!s.openDocs.some((x) => x.id === d.id) && useAnalysisWorkspaceStore.getState().docUids[d.id]) {
      closeDocState(d, s.openDocs);
    }
  }
  // A file name that changed outside a save (afterDocSaved records the new
  // one first) names a file whose workflows were never read: read it.
  for (const d of s.openDocs) {
    const read = hydratedPath.get(d.id);
    if (read !== undefined && read !== (d.path ?? null)) forgetHydration(d.id);
  }
  hydrateOpenDocs();
  scheduleSync();
});
useSourceEditStore.subscribe((s, p) => {
  if (s.revs !== p.revs) scheduleSync();
});
ensureOpenDocUids();
hydrateOpenDocs();
