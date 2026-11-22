/* Durable file identity in the Analysis workspace (uid plan §2.3-2.4): the
   doc.json uid a read returns is adopted (over the manifest's, with the held
   rule, for a newer-format file too), a figure save sends the tab's uid and
   takes the one the file was written under, docMoved re-points a renamed
   file without a new uid or a re-read, a write to a vanished file follows it
   (docLinks.relocateOpenDoc), openMissing finds a renamed file by its uid,
   and labels follow a file renamed while it was closed. The sidecar is a fake
   in-memory "disk"; docLinks is mocked. */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { Edge, Node } from "@xyflow/react";
import type { MpfResolveRef, MpfResolveResult, WorkflowsPayload, WorkflowsReadResponse } from "../api/types";
import type { InsetSource, NodeData, NodeOutput, WorkflowTab } from "../utils/analysisTypes";
import { figKey, nsKey } from "../utils/analysisKeys";
import { deriveOwnership, isLocked } from "../utils/workflowOwnership";
import type { WorkflowsManifestV1 } from "../utils/workflowPersist";
import { fileUidOf, noteDocFile, registryPathsOf, registryUidAt } from "../utils/docIdentity";

// ── fakes: sidecar disk, file existence, dialogs, docLinks ──

const disk = vi.hoisted(() => new Map<string, unknown>());
/** .mpf files that exist (with or without a workflows/ section). */
const files = vi.hoisted(() => new Set<string>());
const mockApi = vi.hoisted(() => ({
  readWorkflows: vi.fn(),
  writeWorkflows: vi.fn(),
  saveProject: vi.fn(),
  loadProject: vi.fn(),
  pathsExist: vi.fn(),
  resolveMpf: vi.fn(),
  updateConfig: vi.fn(async () => ({})),
  getPreview: vi.fn(async () => ({ image: "p", width: 1, height: 1, format: "png" })),
  getPanelPreview: vi.fn(async () => ({ image: "t" })),
}));
vi.mock("../api/client", () => ({
  api: mockApi,
  checkHealth: vi.fn(async () => true),
  lastHealthError: "",
  settleConfigWrites: vi.fn(async () => {}),
  default: vi.fn(),
}));
const mockConfirm = vi.hoisted(() => vi.fn(async () => true));
vi.mock("../components/shared/ConfirmDialog", () => ({
  confirm: mockConfirm, confirmThree: vi.fn(), alert: vi.fn(),
}));
const mockOpenDialog = vi.hoisted(() => vi.fn(async (): Promise<string | null> => null));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: mockOpenDialog }));
const mockRelocate = vi.hoisted(() => vi.fn(async (_docId: string): Promise<string | null> => null));
vi.mock("../utils/docLinks", () => ({ relocateOpenDoc: mockRelocate }));

import { useCollageStore, type CollageItem } from "./collageStore";
import { useSourceEditStore } from "./sourceEditStore";
import { useFigureStore } from "./figureStore";
import {
  useAnalysisWorkspaceStore, resetAnalysisWorkspaceForTests, newSourceNode, workspaceOwnershipCtx,
} from "./analysisWorkspaceStore";

// ── fixtures ────────────────────────────────────────────────

type Doc = { id: string; path: string | null; name: string };
const A: Doc = { id: "doc_a", path: "/exp/a.mpf", name: "a" };
const B: Doc = { id: "doc_b", path: "/exp/b.mpf", name: "b" };
const U: Doc = { id: "doc_u", path: null, name: "Untitled_1" };

const EMPTY: WorkflowsReadResponse = {
  manifest: null, blobs: {}, unsupported: false, version: null, uploads: [], renamed: {},
  missing_uploads: [], failed_uploads: [],
};

function src(doc: { id: string; path?: string | null; name?: string }, raw: string): InsetSource {
  const fk = figKey(doc);
  return {
    key: nsKey(fk, raw), rawKey: raw, figKey: fk, mpfId: doc.id, mpf: doc.name,
    row: 0, col: 0, inset_index: 0, label: raw, natural_width: 10, natural_height: 10,
    thumbnail: "THUMB", runKey: raw,
  };
}
function output(name: string, payload: string): NodeOutput {
  return { id: `out_plot_${name}`, kind: "plot", name, payload, blobId: `b_${name}` };
}
function procNode(id: string, extra: Partial<NodeData> = {}): Node<NodeData> {
  return { id, type: "python", position: { x: 300, y: 30 }, data: { label: id, kind: "python", code: `# ${id}`, ...extra } as NodeData };
}
function edge(source: string, target: string): Edge {
  return { id: `e_${source}_out_image_0__${target}_in_image`, source, sourceHandle: "out_image_0", target, targetHandle: "in_image", type: "deletable" };
}
function wf(id: string, sources: InsetSource[], payload = "PNG_RESULT"): WorkflowTab {
  const ranWith = Object.fromEntries(sources.map((s) => [s.key, 0]));
  return {
    id, name: `Analysis ${id}`, updatedAt: 1_000,
    nodes: [newSourceNode(sources), procNode("p1", { outputs: [output("plot1", payload)], status: "ok", ranWith })],
    edges: [edge("source", "p1")],
  };
}
/** A shared workflow: A's figure on "source", B's on "srcB". */
function sharedWf(id: string, a: Doc, b: Doc): WorkflowTab {
  const sa = src(a, "r0c0_i0");
  const sb = src(b, "r0c1_i0");
  return {
    id, name: `Shared ${id}`, updatedAt: 1_000,
    nodes: [
      newSourceNode([sa]),
      newSourceNode([sb], { id: "srcB", label: "Source 2", position: { x: 30, y: 220 } }),
      procNode("pA", { outputs: [output("a", "RA")], status: "ok", ranWith: { [sa.key]: 0 } }),
      procNode("pB", { outputs: [output("b", "RB")], status: "ok", ranWith: { [sb.key]: 0 } }),
    ],
    edges: [edge("source", "pA"), edge("srcB", "pB")],
  };
}

const st = () => useAnalysisWorkspaceStore.getState();
const tab = (id: string) => st().tabs.find((t) => t.id === id);
const docOf = (id: string) => useCollageStore.getState().openDocs.find((d) => d.id === id);
const docAt = (path: string) => useCollageStore.getState().openDocs.find((d) => d.path === path);
const sourceKeys = (id: string) => tab(id)!.nodes.flatMap((n) => (n.data.sources || []).map((s) => s.key));
const locked = (id: string) => isLocked(tab(id)!, workspaceOwnershipCtx());
const names = () => JSON.parse(localStorage.getItem("mpfig.source_names") || "{}");

function readBack(p: WorkflowsPayload, docUid?: string): WorkflowsReadResponse {
  const blobs: WorkflowsReadResponse["blobs"] = {};
  for (const [id, b] of Object.entries(p.blobs)) blobs[id] = { kind: b.kind, data: b.data ?? "" };
  return {
    ...EMPTY,
    manifest: JSON.parse(JSON.stringify(p.manifest)),
    blobs,
    version: 1,
    uploads: [],
    ...(docUid ? { doc_uid: docUid } : {}),
  };
}
const fileOf = (path: string) => disk.get(path) as WorkflowsReadResponse | undefined;
const manifestAt = (path: string) => fileOf(path)?.manifest as WorkflowsManifestV1 | undefined;
/** Give the file at `path` a doc.json uid (what a figure save writes). */
function stampDocJson(path: string, uid: string) {
  disk.set(path, { ...(fileOf(path) ?? EMPTY), doc_uid: uid });
  files.add(path);
}
function move(from: string, to: string) {
  disk.set(to, disk.get(from));
  disk.delete(from);
  files.delete(from);
  files.add(to);
}

async function flush(n = 80) { for (let i = 0; i < n; i++) await Promise.resolve(); }

/** Fresh session with `docs` open (the first active unless given), files read. */
async function open(docs: Doc[], active: string = docs[0]?.id ?? A.id) {
  useCollageStore.setState({ openDocs: [], activeDocId: null });
  resetAnalysisWorkspaceForTests();
  useCollageStore.setState({ openDocs: docs, activeDocId: active });
  await flush();
}
function closeDoc(doc: Doc) {
  st().closeDoc(doc.id);
  useCollageStore.getState().docRemove(doc.id);
}
/** A figure item of the collage placed from `path`. */
function figItem(id: string, path: string, projectUid?: string): CollageItem {
  return {
    id, kind: "figure", src: "data:", x: 0, y: 0, w: 10, h: 10, naturalW: 10, naturalH: 10,
    rotation: 0, z: 1, name: "fig", projectPath: path, ...(projectUid ? { projectUid } : {}),
  } as CollageItem;
}
const item = (id: string) => useCollageStore.getState().items.find((i) => i.id === id)!;

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
  vi.setSystemTime(10_000);
  vi.clearAllMocks();
  disk.clear();
  files.clear();
  localStorage.clear();
  mockApi.readWorkflows.mockImplementation(async (path: string) => {
    if (!disk.has(path) && !files.has(path)) throw new Error(`API error: "Project file not found: ${path}"`);
    return (disk.get(path) as WorkflowsReadResponse) ?? EMPTY;
  });
  mockApi.writeWorkflows.mockImplementation(async (path: string, payload: WorkflowsPayload) => {
    if (!disk.has(path) && !files.has(path)) throw new Error(`API error: "Project file not found: ${path}"`);
    const prev = fileOf(path);
    disk.set(path, readBack(payload, prev?.doc_uid ?? undefined));
    files.add(path);
    return { ok: true, size_bytes: 1, workflows_bytes: 1, missing_uploads: [], fast_path: true };
  });
  mockApi.saveProject.mockImplementation(async () => ({ ok: true }));
  mockApi.pathsExist.mockImplementation(async (paths: string[]) =>
    Object.fromEntries(paths.map((p) => [p, files.has(p) || disk.has(p)])));
  // A fake mpf-resolve: the ref's path when it is there; else the one other
  // file carrying the uid (several: ambiguous).
  mockApi.resolveMpf.mockImplementation(async (refs: MpfResolveRef[]) => {
    const out: Record<string, MpfResolveResult> = {};
    const uidAt = (p: string) => fileOf(p)?.doc_uid ?? (fileOf(p)?.manifest?.docUid as string | undefined) ?? null;
    for (const r of refs) {
      const exists = files.has(r.path) || disk.has(r.path);
      const there = exists ? uidAt(r.path) : null;
      if (exists && (!r.uid || !there || there === r.uid)) { out[r.key] = { status: "ok", path: r.path, exists, uid: there }; continue; }
      const hits = r.uid ? [...new Set([...files, ...disk.keys()])].filter((p) => p !== r.path && uidAt(p) === r.uid) : [];
      if (hits.length === 1) out[r.key] = { status: "moved", path: hits[0], exists, uid: r.uid! };
      else if (hits.length > 1) out[r.key] = { status: "ambiguous", path: null, exists, uid: null, matches: hits };
      else out[r.key] = exists ? { status: "ok", path: r.path, exists, uid: there } : { status: "missing", path: null, exists, uid: null };
    }
    return out;
  });
  mockConfirm.mockImplementation(async () => true);
  mockOpenDialog.mockImplementation(async () => null);
  mockRelocate.mockImplementation(async () => null);
  useSourceEditStore.setState({ revs: {}, withResults: [], rerun: {} });
  useCollageStore.setState({ items: [], missingItemIds: [] });
  files.add(A.path!);
  files.add(B.path!);
});
afterEach(() => {
  useCollageStore.setState({ openDocs: [], activeDocId: null, items: [] });
  resetAnalysisWorkspaceForTests();
  vi.useRealTimers();
});

// ── hydrate: the file's uid ─────────────────────────────────

describe("hydrate adopts the uid the file names", () => {
  it("its doc.json uid, with no saved Analysis at all — recorded, not written", async () => {
    stampDocJson(A.path!, "du_file_a");
    await open([A]);
    expect(st().hydrated[A.id]).toBe("done");
    expect(st().docUids[A.id]).toBe("du_file_a");
    expect(fileUidOf(A.id)).toBe("du_file_a");
    expect(registryUidAt(A.path)).toBe("du_file_a");
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("clean");
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
  });

  it("a file naming no uid: null, and its first workflows write records the tab's", async () => {
    files.add(A.path!);
    await open([A]);
    expect(fileUidOf(A.id)).toBeNull();
    st().setTabs([wf("wfA", [src(A, "r0c0_i0")])]);
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("written");
    expect(fileUidOf(A.id)).toBe(st().docUids[A.id]);
    expect(registryUidAt(A.path)).toBe(st().docUids[A.id]);
  });

  it("doc.json beats the manifest's docUid: own sources still resolve, the doc is dirty, the persist rewrites the manifest", async () => {
    await open([A]);
    st().adoptUid(A.id, "du_orig");
    st().setTabs([wf("wfA", [src(A, "r0c0_i0")])]);
    await st().persistDocWorkflows(A.id, { force: true });
    expect(manifestAt(A.path!)!.docUid).toBe("du_orig");
    stampDocJson(A.path!, "du_copy");                         // a Save As copy whose Analysis was never rewritten
    await open([A]);
    expect(st().docUids[A.id]).toBe("du_copy");
    expect(sourceKeys("wfA")).toEqual([nsKey(figKey(A), "r0c0_i0")]);
    expect(deriveOwnership(tab("wfA")!, workspaceOwnershipCtx()).owners).toEqual(["du_copy"]);
    expect(locked("wfA")).toBe(false);
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("written");
    expect(manifestAt(A.path!)!.docUid).toBe("du_copy");
    expect(manifestAt(A.path!)!.workflows[0].owners).toEqual(["du_copy"]);
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("clean");
  });

  it("the held rule applies to doc.json: a second file with the same uid keeps its own", async () => {
    const A2: Doc = { id: "doc_a2", path: "/exp/a copy.mpf", name: "a copy" };
    stampDocJson(A.path!, "du_same");
    stampDocJson(A2.path!, "du_same");                        // a Finder copy
    await open([A, A2]);
    expect(st().docUids[A.id]).toBe("du_same");
    expect(st().docUids[A2.id]).not.toBe("du_same");
    expect(fileUidOf(A2.id)).toBe("du_same");                  // what its file names, until its next save
    expect(registryPathsOf("du_same")).toEqual([A.path]);
  });

  it("a held copy's Analysis write makes its file that document, doc.json too: reopened alone it is still itself", async () => {
    const A2: Doc = { id: "doc_a2", path: "/exp/a copy.mpf", name: "a copy" };
    stampDocJson(A.path!, "du_same");
    stampDocJson(A2.path!, "du_same");                        // a Finder copy
    // The sidecar: a doc.json naming another uid than the one sent is
    // rewritten to it; the response says what doc.json names now.
    mockApi.writeWorkflows.mockImplementation(async (path: string, payload: WorkflowsPayload, opts?: { docUid?: string }) => {
      if (!disk.has(path) && !files.has(path)) throw new Error(`API error: "Project file not found: ${path}"`);
      const had = fileOf(path)?.doc_uid ?? null;
      const docUid = had && opts?.docUid ? opts.docUid : had;
      disk.set(path, readBack(payload, docUid ?? undefined));
      files.add(path);
      return { ok: true, size_bytes: 1, workflows_bytes: 1, missing_uploads: [], fast_path: !opts?.docUid, doc_uid: docUid };
    });
    await open([A, A2]);
    const own = st().docUids[A2.id];
    expect(own).not.toBe("du_same");
    st().setTabs([wf("wfC", [src(A2, "r0c0_i0")])]);
    await expect(st().persistDocWorkflows(A2.id)).resolves.toBe("written");
    // expectUid: the uid the file named before this write (the original's).
    expect(mockApi.writeWorkflows).toHaveBeenCalledWith(A2.path, expect.anything(), { docUid: own, expectUid: "du_same" });
    expect(fileOf(A2.path!)!.doc_uid).toBe(own);
    expect(manifestAt(A2.path!)!.docUid).toBe(own);
    expect(fileUidOf(A2.id)).toBe(own);
    expect(registryUidAt(A2.path)).toBe(own);
    // The original's own writes send no docUid (its file already names it),
    // only the check that the file at its path is still itself.
    st().setTabs([...st().tabs, wf("wfA", [src(A, "r0c0_i0")])]);
    await st().persistDocWorkflows(A.id);
    expect(mockApi.writeWorkflows).toHaveBeenLastCalledWith(A.path, expect.anything(), { expectUid: "du_same" });
    expect(fileOf(A.path!)!.doc_uid).toBe("du_same");
    // Reopened alone, the original closed: still the copy's own document.
    await open([A2]);
    expect(st().docUids[A2.id]).toBe(own);
  });

  it("a newer-format file is never overwritten, but its uid is still recorded", async () => {
    disk.set(A.path!, { ...EMPTY, unsupported: true, version: 2, doc_uid: "du_newer" });
    disk.set(B.path!, { ...EMPTY, version: 9, manifest: { format: "mpf-workflows", version: 9, docUid: "du_newer_b" } });
    await open([A, B]);
    expect(st().hydrated[A.id]).toBe("unsupported");
    expect(st().hydrated[B.id]).toBe("unsupported");
    expect(st().docUids[A.id]).toBe("du_newer");
    expect(st().docUids[B.id]).toBe("du_newer_b");
    expect(fileUidOf(A.id)).toBe("du_newer");
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("unsupported");
  });
});

// ── a figure save ───────────────────────────────────────────

describe("afterDocSaved takes the uid the file was written under", () => {
  it("Save As: the response's new uid; keys re-keyed, labels copied, the copy's manifest under it", async () => {
    stampDocJson(A.path!, "du_a");
    await open([A]);
    st().setTabs([wf("wfA", [src(A, "r0c0_i0")])]);
    const oldKey = nsKey(figKey(A), "r0c0_i0");
    localStorage.setItem("mpfig.source_names", JSON.stringify({ [oldKey]: "Ctrl" }));
    files.add("/exp/a-copy.mpf");
    await st().afterDocSaved(A.id, A.path, "/exp/a-copy.mpf", { docUid: "du_copy" });
    const newKey = nsKey(figKey({ id: A.id, path: "/exp/a-copy.mpf" }), "r0c0_i0");
    expect(st().docUids[A.id]).toBe("du_copy");
    expect(fileUidOf(A.id)).toBe("du_copy");
    expect(registryUidAt("/exp/a-copy.mpf")).toBe("du_copy");
    expect(registryUidAt(A.path)).toBe("du_a");               // the original keeps its own
    expect(sourceKeys("wfA")).toEqual([newKey]);
    expect(names()).toEqual({ [oldKey]: "Ctrl", [newKey]: "Ctrl" });
    expect(manifestAt("/exp/a-copy.mpf")!.docUid).toBe("du_copy");
  });

  it("no uid in the response (an older sidecar): a Save As mints one, written by the persist", async () => {
    stampDocJson(A.path!, "du_a");
    await open([A]);
    files.add("/exp/a-copy.mpf");
    await st().afterDocSaved(A.id, A.path, "/exp/a-copy.mpf", {});
    const uid = st().docUids[A.id];
    expect(uid).not.toBe("du_a");
    expect(uid).toMatch(/^du_/);
    expect(manifestAt("/exp/a-copy.mpf")!.docUid).toBe(uid);
    expect(fileUidOf(A.id)).toBe(uid);                         // known once the persist wrote it
  });

  it("the same uid at a new path (its old file was gone): the document moved — labels move, no new uid", async () => {
    stampDocJson(A.path!, "du_a");
    await open([A]);
    const oldKey = nsKey(figKey(A), "r0c0_i0");
    localStorage.setItem("mpfig.source_names", JSON.stringify({ [oldKey]: "Ctrl" }));
    files.add("/exp/moved.mpf");
    await st().afterDocSaved(A.id, A.path, "/exp/moved.mpf", { docUid: "du_a" });
    expect(st().docUids[A.id]).toBe("du_a");
    expect(names()).toEqual({ [nsKey(figKey({ id: A.id, path: "/exp/moved.mpf" }), "r0c0_i0")]: "Ctrl" });
  });

  it("Save As from a held tab (a Finder copy keeps its own uid): its file is still there, so the labels are copied", async () => {
    const A2: Doc = { id: "doc_a2", path: "/exp/a copy.mpf", name: "a copy" };
    stampDocJson(A.path!, "du_same");
    stampDocJson(A2.path!, "du_same");
    await open([A, A2]);
    const own = st().docUids[A2.id];
    expect(own).not.toBe("du_same");
    const oldKey = nsKey(figKey(A2), "r0c0_i0");
    localStorage.setItem("mpfig.source_names", JSON.stringify({ [oldKey]: "Ctrl" }));
    files.add("/exp/bar.mpf");
    // The sidecar keeps the tab's uid (the file it came from names another
    // document); an older one says nothing more — the file's uid tells.
    await st().afterDocSaved(A2.id, A2.path, "/exp/bar.mpf", { docUid: own });
    const newKey = nsKey(figKey({ id: A2.id, path: "/exp/bar.mpf" }), "r0c0_i0");
    expect(st().docUids[A2.id]).toBe(own);
    expect(names()).toEqual({ [oldKey]: "Ctrl", [newKey]: "Ctrl" });
  });

  it("the sidecar's sourceKept decides copy or move when given", async () => {
    stampDocJson(A.path!, "du_a");
    await open([A]);
    const oldKey = nsKey(figKey(A), "r0c0_i0");
    localStorage.setItem("mpfig.source_names", JSON.stringify({ [oldKey]: "Ctrl" }));
    files.add("/exp/x.mpf");
    await st().afterDocSaved(A.id, A.path, "/exp/x.mpf", { docUid: "du_a", sourceKept: true });
    const xKey = nsKey(figKey({ id: A.id, path: "/exp/x.mpf" }), "r0c0_i0");
    expect(names()).toEqual({ [oldKey]: "Ctrl", [xKey]: "Ctrl" });
    // Its previous file gone (sourceKept false): moved, and the table no
    // longer has it at the old path.
    files.add("/exp/y.mpf");
    noteDocFile("du_a", "/exp/x.mpf");
    await st().afterDocSaved(A.id, "/exp/x.mpf", "/exp/y.mpf", { docUid: "du_a", sourceKept: false });
    const yKey = nsKey(figKey({ id: A.id, path: "/exp/y.mpf" }), "r0c0_i0");
    expect(names()).toEqual({ [oldKey]: "Ctrl", [yKey]: "Ctrl" });
    expect(registryUidAt("/exp/x.mpf")).toBeUndefined();
    expect(registryUidAt("/exp/y.mpf")).toBe("du_a");
  });

  it("a uid another open tab holds is not taken", async () => {
    stampDocJson(A.path!, "du_a");
    await open([A, B]);
    const ub = st().docUids[B.id];
    await st().afterDocSaved(A.id, A.path, A.path!, { docUid: ub });
    expect(st().docUids[A.id]).toBe("du_a");
    expect(fileUidOf(A.id)).toBe(ub);                          // still what the file names
  });

  it("a re-save stamps collage figures of that file with its uid; a Save As leaves them on the original", async () => {
    stampDocJson(A.path!, "du_a");
    await open([A]);
    useCollageStore.setState({ items: [figItem("i1", A.path!), figItem("i2", A.path!, "du_stale"), figItem("i3", "/exp/other.mpf")] });
    await st().afterDocSaved(A.id, A.path, A.path!, { docUid: "du_a" });
    expect(item("i1").projectUid).toBe("du_a");
    expect(item("i2").projectUid).toBe("du_a");
    expect(item("i3").projectUid).toBeUndefined();
    files.add("/exp/a-copy.mpf");
    await st().afterDocSaved(A.id, A.path, "/exp/a-copy.mpf", { docUid: "du_copy" });
    expect(item("i1")).toMatchObject({ projectPath: A.path, projectUid: "du_a" });
  });

  it("a save over a file already placed in the collage: its figures take the uid that file carries now", async () => {
    // fig1.mpf (du_old) is in the collage; a Save As from A — or an Untitled
    // tab's first save — writes another document over it. Its figures must
    // not keep du_old: a same-uid backup of the old fig1 would be followed
    // at the next launch.
    stampDocJson(A.path!, "du_a");
    files.add("/exp/fig1.mpf");
    await open([A]);
    useCollageStore.setState({ items: [figItem("i1", A.path!, "du_a"), figItem("i4", "/exp/fig1.mpf", "du_old")] });
    await st().afterDocSaved(A.id, A.path, "/exp/fig1.mpf", { docUid: "du_new", sourceKept: true });
    expect(item("i4")).toMatchObject({ projectPath: "/exp/fig1.mpf", projectUid: "du_new" });
    expect(item("i1")).toMatchObject({ projectPath: A.path, projectUid: "du_a" });
  });
});

describe("figureStore.saveProject sends the tab's uid once its file was read", () => {
  const cfg = { rows: 1, cols: 1, panels: [[{}]] } as never;

  it("a read document: its uid goes out, the response's comes back", async () => {
    stampDocJson(A.path!, "du_a");
    await open([A]);
    useFigureStore.setState({ config: cfg, currentProjectPath: A.path });
    mockApi.saveProject.mockImplementationOnce(async () => ({ ok: true, path: "/exp/a2.mpf", doc_uid: "du_a2", uid_minted: true }));
    files.add("/exp/a2.mpf");
    await expect(useFigureStore.getState().saveProject("/exp/a2.mpf")).resolves.toBe(true);
    expect(mockApi.saveProject).toHaveBeenCalledWith("/exp/a2.mpf", { carryFrom: A.path, docUid: "du_a" });
    expect(st().docUids[A.id]).toBe("du_a2");
    expect(docOf(A.id)!.path).toBe("/exp/a2.mpf");
  });

  it("an Untitled document sends the uid it was given at creation, and keeps it", async () => {
    await open([U]);
    const uid = st().docUids[U.id];
    useFigureStore.setState({ config: cfg, currentProjectPath: null });
    mockApi.saveProject.mockImplementationOnce(async () => ({ ok: true, path: "/exp/u.mpf", doc_uid: uid, uid_minted: false }));
    files.add("/exp/u.mpf");
    await useFigureStore.getState().saveProject("/exp/u.mpf");
    expect(mockApi.saveProject).toHaveBeenCalledWith("/exp/u.mpf", { carryFrom: null, docUid: uid });
    expect(st().docUids[U.id]).toBe(uid);
    expect(registryUidAt("/exp/u.mpf")).toBe(uid);
  });

  it("a document whose file was never read sends none (the file keeps its own) and takes the file's", async () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    mockApi.readWorkflows.mockRejectedValueOnce(new Error('API error: "Not a readable .mpf"'));
    await open([A]);
    expect(st().hydrated[A.id]).toBe("failed");
    useFigureStore.setState({ config: cfg, currentProjectPath: A.path });
    mockApi.saveProject.mockImplementationOnce(async () => ({ ok: true, path: A.path, doc_uid: "du_in_file", uid_minted: true }));
    await useFigureStore.getState().saveProject(A.path!);
    expect(mockApi.saveProject).toHaveBeenCalledWith(A.path, { carryFrom: A.path });
    expect(st().docUids[A.id]).toBe("du_in_file");
    err.mockRestore();
  });

  it("setCurrentProjectPath only moves the path", () => {
    useFigureStore.setState({ currentProjectPath: A.path, unsaved: true });
    useFigureStore.getState().setCurrentProjectPath("/exp/renamed.mpf");
    expect(useFigureStore.getState().currentProjectPath).toBe("/exp/renamed.mpf");
    expect(useFigureStore.getState().unsaved).toBe(true);
  });
});

// ── a file renamed while open ───────────────────────────────

describe("docMoved: a renamed file, the same document", () => {
  it("re-keys tabs and labels, no new uid, no second read, and a saved document stays saved", async () => {
    await open([A, B]);
    st().setTabs([wf("wfA", [src(A, "r0c0_i0")])]);
    await st().persistDocWorkflows(A.id);
    const ua = st().docUids[A.id];
    const oldKey = nsKey(figKey(A), "r0c0_i0");
    localStorage.setItem("mpfig.source_names", JSON.stringify({ [oldKey]: "Ctrl" }));
    await st().persistDocWorkflows(A.id);                     // labels saved too: up to date
    vi.clearAllMocks();
    move(A.path!, "/exp/renamed.mpf");
    st().docMoved(A.path!, "/exp/renamed.mpf");
    await flush();
    const newKey = nsKey(figKey({ id: A.id, path: "/exp/renamed.mpf" }), "r0c0_i0");
    expect(docOf(A.id)).toMatchObject({ path: "/exp/renamed.mpf", name: "renamed" });
    expect(st().docUids[A.id]).toBe(ua);
    expect(sourceKeys("wfA")).toEqual([newKey]);
    expect(names()).toEqual({ [newKey]: "Ctrl" });
    expect(mockApi.readWorkflows).not.toHaveBeenCalled();
    expect(st().hydrated[A.id]).toBe("done");
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("clean");
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
  });

  it("a name the user gave the tab stays; unsaved Analysis stays unsaved; every tab on the path moves", async () => {
    const A2: Doc = { id: "doc_a2", path: A.path, name: "My figure" };
    await open([A, A2]);
    st().setTabs([wf("wfA", [src(A, "r0c0_i0")])]);           // not written yet
    st().docMoved(A.path!, "/exp/renamed.mpf");
    expect(docOf(A.id)!.name).toBe("renamed");
    expect(docOf(A2.id)).toMatchObject({ path: "/exp/renamed.mpf", name: "My figure" });
    files.add("/exp/renamed.mpf");
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("written");
    expect(mockApi.writeWorkflows).toHaveBeenLastCalledWith("/exp/renamed.mpf", expect.anything());
  });

  it("does nothing for a path no tab has", async () => {
    await open([A]);
    const before = useCollageStore.getState().openDocs;
    st().docMoved("/exp/none.mpf", "/exp/x.mpf");
    st().docMoved(A.path!, A.path!);
    expect(useCollageStore.getState().openDocs).toBe(before);
  });
});

describe("persist to a file that vanished", () => {
  it("follows the file to where docLinks finds it and writes there, once", async () => {
    await open([A]);
    st().setTabs([wf("wfA", [src(A, "r0c0_i0")])]);
    move(A.path!, "/exp/renamed.mpf");
    mockRelocate.mockImplementationOnce(async (docId: string) => {
      st().docMoved(A.path!, "/exp/renamed.mpf");
      return docOf(docId)!.path;
    });
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("written");
    expect(mockRelocate).toHaveBeenCalledWith(A.id);
    expect(mockApi.writeWorkflows.mock.calls.map((c) => c[0])).toEqual([A.path, "/exp/renamed.mpf"]);
    expect(manifestAt("/exp/renamed.mpf")!.workflows.map((w) => w.id)).toEqual(["wfA"]);
    expect(manifestAt("/exp/renamed.mpf")!.docs[st().docUids[A.id]].name).toBe("renamed");
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("clean");
  });

  it("not found: the write fails as before; another failure never looks", async () => {
    await open([A]);
    st().setTabs([wf("wfA", [src(A, "r0c0_i0")])]);
    files.delete(A.path!);
    disk.delete(A.path!);
    await expect(st().persistDocWorkflows(A.id)).rejects.toThrow(/Project file not found/);
    expect(mockRelocate).toHaveBeenCalledTimes(1);
    files.add(A.path!);
    mockRelocate.mockClear();
    mockApi.writeWorkflows.mockRejectedValueOnce(new Error("disk full"));
    await expect(st().persistDocWorkflows(A.id)).rejects.toThrow("disk full");
    expect(mockRelocate).not.toHaveBeenCalled();
  });
});

// ── closed documents ────────────────────────────────────────

describe("openMissing finds a renamed file by its uid", () => {
  async function writeBoth() {
    await open([A, B]);
    st().setTabs([sharedWf("wfAB", A, B)]);
    await st().persistDocWorkflows(A.id);
    await st().persistDocWorkflows(B.id);
    return { ub: st().docUids[B.id] };
  }

  it("moved: opened from its new path with no dialog, and unlocked", async () => {
    const { ub } = await writeBoth();
    move(B.path!, "/exp/b renamed.mpf");
    await open([A]);
    expect(locked("wfAB")).toBe(true);
    await st().openMissing("wfAB");
    // One lookup by uid (the read of the found file then checks its old path).
    expect(mockApi.resolveMpf.mock.calls[0][0]).toEqual([expect.objectContaining({ key: ub, path: B.path, uid: ub })]);
    expect(mockApi.pathsExist).not.toHaveBeenCalled();
    expect(mockOpenDialog).not.toHaveBeenCalled();
    expect(mockConfirm).not.toHaveBeenCalled();
    const b2 = docAt("/exp/b renamed.mpf")!;
    expect(b2).toBeDefined();
    expect(st().docUids[b2.id]).toBe(ub);
    expect(locked("wfAB")).toBe(false);
    expect(registryPathsOf(ub)[0]).toBe("/exp/b renamed.mpf");
  });

  it("several files carry it: the user locates it", async () => {
    const { ub } = await writeBoth();
    move(B.path!, "/exp/b1.mpf");
    disk.set("/exp/b2.mpf", disk.get("/exp/b1.mpf"));
    files.add("/exp/b2.mpf");
    await open([A]);
    mockOpenDialog.mockImplementationOnce(async () => "/exp/b2.mpf");
    await st().openMissing("wfAB");
    expect(mockOpenDialog).toHaveBeenCalledTimes(1);
    expect(st().docUids[docAt("/exp/b2.mpf")!.id]).toBe(ub);
    expect(locked("wfAB")).toBe(false);
  });

  it("not found by uid: a candidate that exists under another uid is still offered (the user confirms)", async () => {
    await writeBoth();
    disk.delete(B.path!);
    files.delete(B.path!);
    // Gone from its path; a file of that name elsewhere (one of its
    // candidates) is a different document.
    const B3: Doc = { id: "doc_b3", path: "/other/b.mpf", name: "b" };
    files.add(B3.path!);
    stampDocJson(B3.path!, "du_someone_else");
    await open([A]);
    useAnalysisWorkspaceStore.setState({
      knownDocs: Object.fromEntries(Object.entries(st().knownDocs).map(([u, k]) => [u, { ...k, candidates: [B.path!, B3.path!] }])),
    });
    mockConfirm.mockImplementationOnce(async () => false);
    await st().openMissing("wfAB");
    expect(mockApi.pathsExist).toHaveBeenCalledWith([B.path, B3.path]);
    expect(mockConfirm).toHaveBeenCalledTimes(1);                // "Use this document?"
    expect(locked("wfAB")).toBe(true);
  });
});

describe("a file renamed while it was closed", () => {
  it("brings the labels this machine keeps for it to its new path", async () => {
    stampDocJson(A.path!, "du_a");
    await open([A]);
    const oldKey = nsKey(figKey(A), "r0c0_i0");
    localStorage.setItem("mpfig.source_names", JSON.stringify({ [oldKey]: "Ctrl", "p:/x/other.mpf::r0c0_i0": "Keep" }));
    closeDoc(A);
    move(A.path!, "/exp/a renamed.mpf");
    const A2: Doc = { id: "doc_a2", path: "/exp/a renamed.mpf", name: "a renamed" };
    await open([A2]);
    expect(st().docUids[A2.id]).toBe("du_a");
    expect(names()).toEqual({ "p:/x/other.mpf::r0c0_i0": "Keep", [nsKey(figKey(A2), "r0c0_i0")]: "Ctrl" });
    expect(registryPathsOf("du_a")).toEqual([A2.path]);
    await expect(st().persistDocWorkflows(A2.id)).resolves.toBe("clean");
  });

  it("not when the old file is still there (a copy)", async () => {
    stampDocJson(A.path!, "du_a");
    await open([A]);
    const oldKey = nsKey(figKey(A), "r0c0_i0");
    localStorage.setItem("mpfig.source_names", JSON.stringify({ [oldKey]: "Ctrl" }));
    closeDoc(A);
    stampDocJson("/exp/a copy.mpf", "du_a");
    const A2: Doc = { id: "doc_a2", path: "/exp/a copy.mpf", name: "a copy" };
    await open([A2]);
    expect(names()).toEqual({ [oldKey]: "Ctrl" });
    expect(registryPathsOf("du_a")).toEqual([A2.path, A.path]);
  });

  it("a sidecar that can't tell moves nothing", async () => {
    files.delete(A.path!);
    noteDocFile("du_a", A.path!);
    const oldKey = nsKey(figKey(A), "r0c0_i0");
    localStorage.setItem("mpfig.source_names", JSON.stringify({ [oldKey]: "Ctrl" }));
    stampDocJson("/exp/a2.mpf", "du_a");
    mockApi.resolveMpf.mockRejectedValueOnce(new Error("offline"));
    await open([{ id: "doc_a2", path: "/exp/a2.mpf", name: "a2" }]);
    expect(st().hydrated.doc_a2).toBe("done");
    expect(names()).toEqual({ [oldKey]: "Ctrl" });
  });
});

describe("files stored online only (Google Drive, Dropbox… evicted them)", () => {
  /** Files not on this Mac: a background read is answered at once, unread;
   *  any other read downloads the file. */
  const online = new Set<string>();
  /** B's file holds a saved workflow of its own ("wfB"). */
  async function savedInB() {
    await open([B]);
    st().setTabs([wf("wfB", [src(B, "r0c0_i0")])]);
    await expect(st().persistDocWorkflows(B.id)).resolves.toBe("written");
  }
  const readsOf = (path: string) => mockApi.readWorkflows.mock.calls.filter((c) => c[0] === path).map((c) => c[1] ?? null);

  beforeEach(() => {
    online.clear();
    const base = mockApi.readWorkflows.getMockImplementation()!;
    mockApi.readWorkflows.mockImplementation(async (path: string, opts?: { noDownload?: boolean }) => {
      if (online.has(path)) {
        if (opts?.noDownload) throw new Error(`API error: "Cloud file is stored online only: ${path}"`);
        online.delete(path);                                     // downloaded by this read
      }
      return base(path);
    });
  });

  it("a restored tab's file is not downloaded in the background; opening the tab reads its Analysis", async () => {
    await savedInB();
    online.add(B.path!);
    mockApi.readWorkflows.mockClear();
    await open([A, B], A.id);                                    // relaunch: B restored, not opened
    expect(readsOf(B.path!)).toEqual([{ noDownload: true }]);
    expect(st().hydrated[B.id]).toBe("failed");                  // never overwritten meanwhile
    expect(tab("wfB")).toBeUndefined();
    // Other tab changes do not try again (no download, no loop).
    useCollageStore.setState({ openDocs: [A, B, U] });
    await flush();
    expect(readsOf(B.path!)).toHaveLength(1);
    // The user opens B's tab (activateDoc): read again, downloading.
    await st().ensureDocLoaded(B.id, B.path, { download: true });
    expect(readsOf(B.path!)).toEqual([{ noDownload: true }, null]);
    expect(st().hydrated[B.id]).toBe("done");
    expect(tab("wfB")).toBeDefined();
  });

  it("entering Analysis reads the documents put off", async () => {
    await savedInB();
    online.add(B.path!);
    await open([A, B], A.id);
    expect(tab("wfB")).toBeUndefined();
    useCollageStore.getState().setMode("analysis");
    await flush();
    expect(st().hydrated[B.id]).toBe("done");
    expect(tab("wfB")).toBeDefined();
    useCollageStore.getState().setMode("builder");
  });

  it("a quit downloads a put-off file only when there is Analysis of its own to write into it", async () => {
    await savedInB();
    online.add(B.path!);
    await open([A, B], A.id);
    mockApi.writeWorkflows.mockClear();
    // Nothing of B's in the workspace: not read, not written.
    await expect(st().persistDocWorkflows(B.id)).resolves.toBe("not-hydrated");
    expect(online.has(B.path!)).toBe(true);
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
    // New Analysis on B's figure: its file is read first (downloaded), then written —
    // the workflow already saved in it is kept.
    st().setTabs([...st().tabs, wf("wfNew", [src(B, "r0c1_i0")])]);
    await expect(st().persistDocWorkflows(B.id)).resolves.toBe("written");
    expect(online.has(B.path!)).toBe(false);
    const saved = (manifestAt(B.path!)!.workflows as Array<{ id: string }>).map((w) => w.id).sort();
    expect(saved).toEqual(["wfB", "wfNew"]);
  });

  it("a download that fails (cloud app not running, offline) is tried again when the tab is opened", async () => {
    await savedInB();
    online.add(B.path!);
    await open([A, B], A.id);
    const base = mockApi.readWorkflows.getMockImplementation()!;
    mockApi.readWorkflows.mockImplementationOnce(async (path: string) => {
      throw new Error(`API error: "Cloud file could not be downloaded: ${path} (Operation timed out)"`);
    });
    await st().ensureDocLoaded(B.id, B.path, { download: true });
    expect(st().hydrated[B.id]).toBe("failed");
    mockApi.readWorkflows.mockImplementation(base);
    await st().ensureDocLoaded(B.id, B.path, { download: true });
    expect(st().hydrated[B.id]).toBe("done");
    expect(tab("wfB")).toBeDefined();
  });
});
