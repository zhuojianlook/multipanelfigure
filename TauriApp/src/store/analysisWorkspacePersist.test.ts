/* Analysis workflows saved in each .mpf (plan step 5): the openDocs subscriber
   and its strictly sequential read queue, hydrate (results restored, stale
   flagged, the file's uid adopted, merge), persist (skipped when clean / not
   read / Untitled; figureDiscarded), afterDocSaved (Untitled → saved, Save
   As), closeDoc, and figureStore's save / load wiring. The sidecar is a fake
   in-memory "disk" — no request ever leaves the test. */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { Edge, Node } from "@xyflow/react";
import type { WorkflowsPayload, WorkflowsReadResponse } from "../api/types";
import type { InsetSource, NodeData, NodeOutput, WorkflowTab } from "../utils/analysisTypes";
import { figKey, nsKey } from "../utils/analysisKeys";
import { computeStale } from "../utils/analysisStale";
import { closedFigKey, isLocked } from "../utils/workflowOwnership";
import type { DocWorkflowsPayload, WorkflowsManifestV1 } from "../utils/workflowPersist";

// ── fake sidecar: an in-memory disk of `workflows/` sections ─

const disk = vi.hoisted(() => new Map<string, unknown>());
const mockApi = vi.hoisted(() => ({
  readWorkflows: vi.fn(),
  writeWorkflows: vi.fn(),
  saveProject: vi.fn(async () => ({ ok: true })),
  loadProject: vi.fn(),
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

import { useCollageStore } from "./collageStore";
import { useSourceEditStore } from "./sourceEditStore";
import { useFigureStore } from "./figureStore";
import {
  useAnalysisWorkspaceStore, resetAnalysisWorkspaceForTests, hydrateOpenDocs, newSourceNode, newWorkflowTab,
  docsWithAnalysis, untitledAnalysisDocIds, rekeyAnalysisStorage, syncWorkspaceNow, READ_RETRY_MS,
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
function upload(name: string): InsetSource {
  return {
    key: `img:${name}`, name, row: -1, col: -1, inset_index: -1, label: name,
    natural_width: 5, natural_height: 5, thumbnail: "UP",
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
/** A workflow reading `sources`, with one result node holding `payload`. */
function wf(id: string, sources: InsetSource[], payload = "PNG_RESULT", extra: Partial<WorkflowTab> = {}): WorkflowTab {
  const ranWith = Object.fromEntries(sources.map((s) => [s.key, 0]));
  return {
    id, name: `Analysis ${id}`, updatedAt: 1_000,
    nodes: [newSourceNode(sources), procNode("p1", { outputs: [output("plot1", payload)], status: "ok", ranWith })],
    edges: [edge("source", "p1")],
    ...extra,
  };
}

const st = () => useAnalysisWorkspaceStore.getState();
const tab = (id: string) => st().tabs.find((t) => t.id === id);

/** What the sidecar's read returns for a payload it wrote (blobs as stored,
 *  embedded uploads restored under their names). */
function readBack(p: WorkflowsPayload): WorkflowsReadResponse {
  const blobs: WorkflowsReadResponse["blobs"] = {};
  for (const [id, b] of Object.entries(p.blobs)) blobs[id] = { kind: b.kind, data: b.data ?? "" };
  return {
    ...EMPTY,
    manifest: JSON.parse(JSON.stringify(p.manifest)),
    blobs,
    version: 1,
    uploads: p.uploads.map((name) => ({ name, thumbnail: "UPTHUMB", size: [5, 5] as [number, number] })),
  };
}
const fileManifest = (path: string) => (disk.get(path) as WorkflowsReadResponse | undefined)?.manifest as WorkflowsManifestV1 | undefined;

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}
async function flush(n = 60) { for (let i = 0; i < n; i++) await Promise.resolve(); }

/** Fresh session: `docs` open (A active unless given), nothing read yet. */
function session(docs: Doc[], active: string = docs[0]?.id ?? A.id) {
  useCollageStore.setState({ openDocs: [], activeDocId: null });
  resetAnalysisWorkspaceForTests();
  useCollageStore.setState({ openDocs: docs, activeDocId: active });
}
/** Open `docs` and let their reads finish. */
async function open(docs: Doc[], active?: string) {
  session(docs, active);
  await flush();
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
  vi.setSystemTime(10_000);
  vi.clearAllMocks();
  disk.clear();
  localStorage.clear();
  mockApi.readWorkflows.mockImplementation(async (path: string) => (disk.get(path) as WorkflowsReadResponse) ?? EMPTY);
  mockApi.writeWorkflows.mockImplementation(async (path: string, payload: WorkflowsPayload) => {
    disk.set(path, readBack(payload));
    return { ok: true, size_bytes: 1, workflows_bytes: 1, missing_uploads: [], fast_path: true };
  });
  useSourceEditStore.setState({ revs: {}, withResults: [], rerun: {} });
});
afterEach(() => {
  useCollageStore.setState({ openDocs: [], activeDocId: null });
  resetAnalysisWorkspaceForTests();
  vi.useRealTimers();
});

// ── hydrate ─────────────────────────────────────────────────

describe("hydrate: the openDocs subscriber reads each file once, one at a time", () => {
  it("queues one read per document with a file; Untitled is done at once; reads never overlap", async () => {
    const reads: Array<ReturnType<typeof deferred<WorkflowsReadResponse>>> = [];
    mockApi.readWorkflows.mockImplementation(() => { const d = deferred<WorkflowsReadResponse>(); reads.push(d); return d.promise; });
    session([U, A, B], U.id);
    await flush();
    expect(st().hydrated).toEqual({ [U.id]: "done", [A.id]: "pending", [B.id]: "pending" });
    expect(mockApi.readWorkflows).toHaveBeenCalledTimes(1);           // B waits for A
    expect(mockApi.readWorkflows).toHaveBeenLastCalledWith(A.path);
    reads[0].resolve(EMPTY);
    await flush();
    expect(st().hydrated[A.id]).toBe("done");
    expect(mockApi.readWorkflows).toHaveBeenCalledTimes(2);
    expect(mockApi.readWorkflows).toHaveBeenLastCalledWith(B.path);
    reads[1].resolve(EMPTY);
    await flush();
    expect(st().hydrated[B.id]).toBe("done");
    // Tab switches / renames / disk reloads never read again.
    useCollageStore.getState().docSetActive(B.id);
    useCollageStore.getState().docRename(A.id, "renamed");
    await flush();
    expect(mockApi.readWorkflows).toHaveBeenCalledTimes(2);
  });

  it("a document opened later (Open / a cold tab) is read when its tab appears", async () => {
    await open([A]);
    expect(mockApi.readWorkflows).toHaveBeenCalledTimes(1);
    useCollageStore.getState().docEnsure(B.path!, undefined, B.id);
    await flush();
    expect(mockApi.readWorkflows).toHaveBeenLastCalledWith(B.path);
    expect(st().hydrated[B.id]).toBe("done");
  });

  it("a document closed while its read is queued is dropped; nothing of it is merged", async () => {
    disk.set(B.path!, readBack(await payloadFor(B, [wf("wfB", [src(B, "r0c0_i0")])])));
    const d = deferred<WorkflowsReadResponse>();
    mockApi.readWorkflows.mockReturnValueOnce(d.promise);
    session([A, B]);
    await flush();
    useCollageStore.getState().docRemove(B.id);                       // B closes while A is still being read
    d.resolve(EMPTY);
    await flush();
    expect(st().hydrated[B.id]).toBeUndefined();
    expect(tab("wfB")).toBeUndefined();
    expect(mockApi.readWorkflows).toHaveBeenCalledTimes(1);
  });

  it("a read the sidecar refuses leaves the document 'failed' — never overwritten", async () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    mockApi.readWorkflows.mockRejectedValueOnce(new Error('API error: "Not a readable .mpf"'));
    await open([A]);
    err.mockRestore();
    expect(st().hydrated[A.id]).toBe("failed");
    st().setTabs((tabs) => [...tabs, { ...wf("wfA", [src(A, "r0c0_i0")]), updatedAt: undefined }]);
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("not-hydrated");
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
  });

  it("a read the sidecar didn't answer (still starting) is retried, not given up", async () => {
    disk.set(A.path!, readBack(await payloadFor(A, [wf("wfA", [src(A, "r0c0_i0")])])));
    mockApi.readWorkflows.mockRejectedValueOnce(new Error("connection refused"));
    await open([A]);
    expect(st().hydrated[A.id]).toBe("pending");
    expect(mockApi.readWorkflows).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(READ_RETRY_MS);
    await flush();
    expect(mockApi.readWorkflows).toHaveBeenCalledTimes(2);
    expect(st().hydrated[A.id]).toBe("done");
    expect(tab("wfA")).toBeDefined();
  });

  it("a close waiting on a read the sidecar doesn't answer stops the retries instead of hanging", async () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    mockApi.readWorkflows.mockRejectedValue(new Error("connection refused"));
    await open([A]);
    let outcome: string | null = null;
    void st().persistDocWorkflows(A.id).then((o) => { outcome = o; });
    vi.advanceTimersByTime(READ_RETRY_MS);
    await flush();
    expect(outcome).toBe("not-hydrated");
    expect(st().hydrated[A.id]).toBe("failed");
    expect(mockApi.readWorkflows).toHaveBeenCalledTimes(2);
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
    err.mockRestore();
  });

  it("a newer-format file is 'unsupported': nothing shown, never overwritten", async () => {
    disk.set(A.path!, { ...EMPTY, unsupported: true, version: 2 });
    await open([A]);
    expect(st().hydrated[A.id]).toBe("unsupported");
    st().setTabs((tabs) => [...tabs, wf("wfA", [src(A, "r0c0_i0")], "X", { updatedAt: undefined })]);
    await expect(st().persistDocWorkflows(A.id, { force: true })).resolves.toBe("unsupported");
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
  });
});

/** The payload document `doc` would write holding `tabs` (a previous session). */
async function payloadFor(doc: Doc, tabs: WorkflowTab[], opts: { figureDiscarded?: boolean; revs?: Record<string, number> } = {}): Promise<DocWorkflowsPayload> {
  session([doc]);
  await flush();
  useSourceEditStore.setState({ revs: opts.revs ?? {} });
  st().setTabs(tabs);
  let sent: DocWorkflowsPayload | null = null;
  mockApi.writeWorkflows.mockImplementationOnce(async (_p: string, payload: DocWorkflowsPayload) => {
    sent = payload;
    return { ok: true, size_bytes: 1, workflows_bytes: 1, missing_uploads: [], fast_path: true };
  });
  await st().persistDocWorkflows(doc.id, { force: true, figureDiscarded: opts.figureDiscarded });
  useSourceEditStore.setState({ revs: {} });
  vi.clearAllMocks();
  mockApi.readWorkflows.mockImplementation(async (path: string) => (disk.get(path) as WorkflowsReadResponse) ?? EMPTY);
  mockApi.writeWorkflows.mockImplementation(async (path: string, payload: WorkflowsPayload) => {
    disk.set(path, readBack(payload));
    return { ok: true, size_bytes: 1, workflows_bytes: 1, missing_uploads: [], fast_path: true };
  });
  if (!sent) throw new Error("nothing written");
  return sent;
}

describe("hydrate: a reopened file's analysis comes back without a re-run", () => {
  it("restores a single-owner workflow with its results under the new session's document", async () => {
    const p = await payloadFor(A, [wf("wfA", [src(A, "r0c0_i0"), upload("stack.tif")], "PNG_BYTES")]);
    disk.set(A.path!, readBack(p));
    const A2 = { ...A, id: "doc_a_next_launch" };                    // a new session id
    await open([A2]);
    const t = tab("wfA")!;
    expect(t).toBeDefined();
    const p1 = t.nodes.find((n) => n.id === "p1")!;
    expect(p1.data.outputs!.map((o) => o.payload)).toEqual(["PNG_BYTES"]);
    expect(p1.data.status).toBe("ok");
    const srcs = t.nodes.find((n) => n.id === "source")!.data.sources!;
    expect(srcs[0].key).toBe(nsKey(figKey(A2), "r0c0_i0"));
    expect(srcs[0].mpfId).toBe(A2.id);
    expect(srcs[1]).toMatchObject({ name: "stack.tif", key: "img:stack.tif" });
    // Not stale, and its upload tile is back in "Direct upload".
    expect(computeStale(t.nodes, t.edges, useSourceEditStore.getState().revs).nodeIds.size).toBe(0);
    expect(st().uploads.map((u) => u.key)).toEqual(["img:stack.tif"]);
    // The document adopted the uid its file was saved under.
    expect(st().docUids[A2.id]).toBe(p.manifest.docUid);
    // A freshly read file is clean: closing it writes nothing.
    await expect(st().persistDocWorkflows(A2.id)).resolves.toBe("clean");
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
  });

  it("results saved while their figure was discarded come back flagged 'needs re-run' (ranWith one below)", async () => {
    const p = await payloadFor(A, [wf("wfA", [src(A, "r0c0_i0")])], { figureDiscarded: true });
    expect((p.manifest.workflows[0].nodes.find((n) => n.id === "p1")!.data).stale).toBe(true);
    disk.set(A.path!, readBack(p));
    await open([A]);
    const t = tab("wfA")!;
    const p1 = t.nodes.find((n) => n.id === "p1")!;
    const key = nsKey(figKey(A), "r0c0_i0");
    expect(p1.data.ranWith).toEqual({ [key]: -1 });
    expect([...computeStale(t.nodes, t.edges, {}).nodeIds]).toEqual(["p1"]);
    expect(p1.data.outputs![0].payload).toBe("PNG_RESULT");          // still shown
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("clean");
  });

  it("A then B keeps both; hydrating twice is a no-op", async () => {
    disk.set(A.path!, readBack(await payloadFor(A, [wf("wfA", [src(A, "r0c0_i0")])])));
    disk.set(B.path!, readBack(await payloadFor(B, [wf("wfB", [src(B, "r0c0_i0")])])));
    await open([A, B]);
    expect(st().tabs.map((t) => t.id).sort()).toEqual(["wfA", "wfB"]);   // the placeholder made way
    const before = st().tabs;
    st().hydrateDoc(A.id, disk.get(A.path!) as WorkflowsReadResponse);
    expect(st().tabs).toBe(before);
    await st().ensureDocLoaded(A.id, A.path);
    expect(mockApi.readWorkflows).toHaveBeenCalledTimes(2);
  });

  it("the same file under a second tab keeps a uid of its own; its copy gets a new workflow id", async () => {
    const p = await payloadFor(A, [wf("wfA", [src(A, "r0c0_i0")])]);
    const twin: Doc = { id: "doc_twin", path: "/exp/copy-of-a.mpf", name: "copy-of-a" };
    disk.set(A.path!, readBack(p));
    disk.set(twin.path!, readBack(p));                                // Finder copy: same uid inside
    await open([A, twin]);
    expect(st().docUids[A.id]).toBe(p.manifest.docUid);
    expect(st().docUids[twin.id]).not.toBe(p.manifest.docUid);
    expect(st().tabs).toHaveLength(2);
    const other = st().tabs.find((t) => t.id !== "wfA")!;
    expect(other.nodes[0].data.sources![0].key).toBe(nsKey(figKey(twin), "r0c0_i0"));
    // The twin's file holds the old id: it is dirty and gets rewritten.
    await expect(st().persistDocWorkflows(twin.id)).resolves.toBe("written");
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("clean");
  });

  it("an older file (no workflows) adds nothing and is clean", async () => {
    await open([A]);
    expect(st().tabs).toHaveLength(1);
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("clean");
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
  });
});

// ── persist ─────────────────────────────────────────────────

describe("persistDocWorkflows", () => {
  it("an edit makes the document dirty: written once, then clean", async () => {
    await open([A]);
    st().setTabs([wf("wfA", [src(A, "r0c0_i0")], "R1", { updatedAt: undefined })]);
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("written");
    expect(mockApi.writeWorkflows).toHaveBeenCalledWith(A.path, expect.objectContaining({ manifest: expect.objectContaining({ docUid: st().docUids[A.id] }) }));
    const m = fileManifest(A.path!)!;
    expect(m.workflows.map((w) => w.id)).toEqual(["wfA"]);
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("clean");
    expect(mockApi.writeWorkflows).toHaveBeenCalledTimes(1);
    // Thumbnails / selection don't count; code does.
    st().setTabNodes("wfA", (nodes) => nodes.map((n) => ({ ...n, selected: true })));
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("clean");
    st().setTabNodes("wfA", (nodes) => nodes.map((n) => (n.id === "p1" ? { ...n, data: { ...n.data, code: "# changed" } } : n)));
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("written");
    expect(mockApi.writeWorkflows).toHaveBeenCalledTimes(2);
  });

  it("each document writes only what it owns", async () => {
    await open([A, B]);
    st().setTabs([wf("wfA", [src(A, "r0c0_i0")], "RA"), wf("wfB", [src(B, "r0c0_i0")], "RB")]);
    await st().persistDocWorkflows(A.id);
    await st().persistDocWorkflows(B.id);
    expect(fileManifest(A.path!)!.workflows.map((w) => w.id)).toEqual(["wfA"]);
    expect(fileManifest(B.path!)!.workflows.map((w) => w.id)).toEqual(["wfB"]);
    const blobsA = Object.values((disk.get(A.path!) as WorkflowsReadResponse).blobs).map((b) => b.data);
    expect(blobsA).toContain("RA");
    expect(blobsA).not.toContain("RB");
  });

  it("waits for the document's pending read before writing (and then finds it clean)", async () => {
    const p = await payloadFor(A, [wf("wfA", [src(A, "r0c0_i0")])]);
    const d = deferred<WorkflowsReadResponse>();
    mockApi.readWorkflows.mockReturnValueOnce(d.promise);
    session([A]);
    await flush();
    const out = st().persistDocWorkflows(A.id);
    await flush();
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
    d.resolve(readBack(p));
    await expect(out).resolves.toBe("clean");
    expect(tab("wfA")).toBeDefined();
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
  });

  it("Untitled: nowhere to write", async () => {
    await open([U]);
    st().setTabs([wf("wfU", [src(U, "r0c0_i0")])]);
    await expect(st().persistDocWorkflows(U.id)).resolves.toBe("no-path");
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
  });

  it("figureDiscarded: results reading the document's images are stored stale, uploads-only results are not", async () => {
    await open([A]);
    st().setTabs([wf("wfImg", [src(A, "r0c0_i0")]), wf("wfUp", [upload("u.tif")], "UP_RESULT", { home: st().docUids[A.id] })]);
    await st().persistDocWorkflows(A.id, { figureDiscarded: true });
    const m = fileManifest(A.path!)!;
    const p1 = (id: string) => m.workflows.find((w) => w.id === id)!.nodes.find((n) => n.id === "p1")!.data;
    expect(p1("wfImg").stale).toBe(true);
    expect(p1("wfUp").stale).toBeUndefined();
    expect((disk.get(A.path!) as WorkflowsReadResponse).uploads.map((u) => u.name)).toEqual(["u.tif"]);
    // Not what the live workspace signs as: a later save (figure kept after
    // all) writes again rather than leave the "needs re-run" flags behind.
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("written");
    expect(p1("wfImg").stale).toBe(true);                             // the old manifest object
    expect(fileManifest(A.path!)!.workflows.find((w) => w.id === "wfImg")!.nodes.find((n) => n.id === "p1")!.data.stale).toBeUndefined();
  });

  it("a write the sidecar refuses as a newer format marks the document unsupported", async () => {
    await open([A]);
    st().setTabs([wf("wfA", [src(A, "r0c0_i0")])]);
    mockApi.writeWorkflows.mockResolvedValueOnce({ ok: false, unsupported: true, version: 2 });
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("unsupported");
    expect(st().hydrated[A.id]).toBe("unsupported");
    await expect(st().persistDocWorkflows(A.id, { force: true })).resolves.toBe("unsupported");
    expect(mockApi.writeWorkflows).toHaveBeenCalledTimes(1);
  });

  it("a failed write rejects and leaves the document dirty", async () => {
    await open([A]);
    st().setTabs([wf("wfA", [src(A, "r0c0_i0")])]);
    mockApi.writeWorkflows.mockRejectedValueOnce(new Error("disk full"));
    await expect(st().persistDocWorkflows(A.id)).rejects.toThrow("disk full");
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("written");
  });

  it("writes go out one at a time, in call order", async () => {
    await open([A, B]);
    st().setTabs([wf("wfA", [src(A, "r0c0_i0")]), wf("wfB", [src(B, "r0c0_i0")])]);
    const first = deferred<unknown>();
    const order: string[] = [];
    mockApi.writeWorkflows.mockImplementationOnce(async (path: string) => { order.push(`start ${path}`); await first.promise; order.push(`end ${path}`); return { ok: true, size_bytes: 1, workflows_bytes: 1, missing_uploads: [], fast_path: true }; });
    mockApi.writeWorkflows.mockImplementationOnce(async (path: string) => { order.push(`start ${path}`); return { ok: true, size_bytes: 1, workflows_bytes: 1, missing_uploads: [], fast_path: true }; });
    const pa = st().persistDocWorkflows(A.id);
    const pb = st().persistDocWorkflows(B.id);
    await flush();
    expect(order).toEqual([`start ${A.path}`]);
    first.resolve(null);
    await Promise.all([pa, pb]);
    expect(order).toEqual([`start ${A.path}`, `end ${A.path}`, `start ${B.path}`]);
  });
});

// ── afterDocSaved ───────────────────────────────────────────

describe("afterDocSaved", () => {
  it("Untitled saved: the tab, every key and the labels move to the file; its workflows are written there", async () => {
    await open([U]);
    const uid = st().docUids[U.id];
    const s = src(U, "r0c0_i0");
    st().setTabs([wf("wfU", [s])]);
    useSourceEditStore.setState({ revs: { [s.key]: 2 } });
    localStorage.setItem("mpfig.source_names", JSON.stringify({ [s.key]: "Control" }));
    await st().afterDocSaved(U.id, null, "/exp/new.mpf");
    const fk = figKey({ id: U.id, path: "/exp/new.mpf" });
    expect(useCollageStore.getState().openDocs[0]).toMatchObject({ id: U.id, path: "/exp/new.mpf", name: "new" });
    expect(tab("wfU")!.nodes[0].data.sources![0].key).toBe(nsKey(fk, "r0c0_i0"));
    expect(useSourceEditStore.getState().revs).toEqual({ [nsKey(fk, "r0c0_i0")]: 2 });
    expect(JSON.parse(localStorage.getItem("mpfig.source_names")!)).toEqual({ [nsKey(fk, "r0c0_i0")]: "Control" });
    expect(st().docUids[U.id]).toBe(uid);                             // same document
    expect(mockApi.writeWorkflows).toHaveBeenCalledTimes(1);
    expect(fileManifest("/exp/new.mpf")!.docUid).toBe(uid);
    expect(mockApi.readWorkflows).not.toHaveBeenCalled();             // its own new file is not read back
    await expect(st().persistDocWorkflows(U.id)).resolves.toBe("clean");
  });

  it("Save As: a new uid, keys re-keyed to the new file, the old file untouched, labels copied", async () => {
    disk.set(A.path!, readBack(await payloadFor(A, [wf("wfA", [src(A, "r0c0_i0")])])));
    await open([A]);
    const oldUid = st().docUids[A.id];
    const before = disk.get(A.path!);
    const oldKey = nsKey(figKey(A), "r0c0_i0");
    localStorage.setItem("mpfig.source_names", JSON.stringify({ [oldKey]: "Ctrl" }));
    await st().afterDocSaved(A.id, A.path, "/exp/a-copy.mpf");
    const newKey = nsKey(figKey({ id: A.id, path: "/exp/a-copy.mpf" }), "r0c0_i0");
    expect(st().docUids[A.id]).not.toBe(oldUid);
    expect(tab("wfA")!.nodes[0].data.sources![0].key).toBe(newKey);
    expect(fileManifest("/exp/a-copy.mpf")!.docUid).toBe(st().docUids[A.id]);
    expect(fileManifest("/exp/a-copy.mpf")!.workflows[0].owners).toEqual([st().docUids[A.id]]);
    expect(disk.get(A.path!)).toBe(before);
    expect(JSON.parse(localStorage.getItem("mpfig.source_names")!)).toEqual({ [oldKey]: "Ctrl", [newKey]: "Ctrl" });
  });

  it("same path (a plain save): persist only when dirty", async () => {
    await open([A]);
    await st().afterDocSaved(A.id, A.path, A.path!);
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
    st().setTabs([wf("wfA", [src(A, "r0c0_i0")])]);
    await st().afterDocSaved(A.id, A.path, A.path!);
    expect(mockApi.writeWorkflows).toHaveBeenCalledTimes(1);
  });
});

// ── close ───────────────────────────────────────────────────

describe("closeDoc", () => {
  it("removes the document's single-owner workflows; shared ones are re-keyed u:<uid> (locked); closing B too removes them", async () => {
    await open([A, B]);
    const ua = st().docUids[A.id];
    st().setTabs([
      wf("wfA", [src(A, "r0c0_i0")]),
      wf("wfB", [src(B, "r0c0_i0")]),
      wf("wfAB", [src(A, "r0c1_i0"), src(B, "r0c1_i0")]),
    ]);
    st().closeDoc(A.id);
    useCollageStore.getState().docRemove(A.id);
    expect(st().tabs.map((t) => t.id)).toEqual(["wfB", "wfAB"]);
    const shared = tab("wfAB")!;
    expect(shared.nodes[0].data.sources![0].key).toBe(nsKey(closedFigKey(ua), "r0c1_i0"));
    const ctx = { docs: [{ id: B.id, uid: st().docUids[B.id], name: "b", path: B.path }], known: st().knownDocs };
    expect(isLocked(shared, ctx)).toBe(true);
    expect(st().knownDocs[ua]).toMatchObject({ name: "a", path: A.path, sessionId: A.id });
    expect(st().docUids[A.id]).toBeUndefined();
    expect(st().hydrated[A.id]).toBeUndefined();
    st().closeDoc(B.id);
    useCollageStore.getState().docRemove(B.id);
    expect(st().tabs).toHaveLength(1);                                // a fresh placeholder
    expect(st().tabs[0].name).toBe("Workflow 1");
  });

  it("a shared workflow comes back unlocked when its document reopens; its in-memory results are flagged 'needs re-run'", async () => {
    await open([A, B]);
    st().setTabs([wf("wfAB", [src(A, "r0c1_i0"), src(B, "r0c1_i0")])]);
    await st().persistDocWorkflows(A.id);                             // A's file holds its copy
    st().closeDoc(A.id);
    useCollageStore.getState().docRemove(A.id);
    const A2 = { ...A, id: "doc_a_again" };
    useCollageStore.getState().docEnsure(A2.path!, A2.name, A2.id);
    await flush();
    const t = tab("wfAB")!;
    expect(t.nodes[0].data.sources!.map((s) => s.key)).toEqual([nsKey(figKey(A2), "r0c1_i0"), nsKey(figKey(B), "r0c1_i0")]);
    expect([...computeStale(t.nodes, t.edges, useSourceEditStore.getState().revs).nodeIds]).toEqual(["p1"]);
    expect(t.nodes.find((n) => n.id === "p1")!.data.outputs![0].payload).toBe("PNG_RESULT");
  });

  it("a tab removed without closeDoc (session restore dropping the blank tab) is let go the same way", async () => {
    await open([U, A], U.id);
    syncWorkspaceNow();
    expect(st().tabs[0].home).toBe(st().docUids[U.id]);
    useCollageStore.getState().docRemove(U.id);
    useCollageStore.getState().docSetActive(A.id);
    expect(st().docUids[U.id]).toBeUndefined();
    expect(st().tabs).toHaveLength(1);
    vi.advanceTimersByTime(600);
    expect(st().tabs[0].home).toBe(st().docUids[A.id]);               // the placeholder follows the active doc
  });
});

describe("the empty placeholder tab", () => {
  it("follows the active document, so work started in it is saved with the figure on screen", async () => {
    await open([U, A], U.id);
    st().setTabs((tabs) => [...tabs, wf("wfUp", [upload("x.tif")], "X", { home: st().docUids[U.id] })]);
    syncWorkspaceNow();
    expect(st().tabs[0].home).toBe(st().docUids[U.id]);
    useCollageStore.getState().docSetActive(A.id);
    syncWorkspaceNow();
    expect(st().tabs[0].home).toBe(st().docUids[A.id]);               // untouched placeholder moves
    expect(tab("wfUp")!.home).toBe(st().docUids[U.id]);               // real work keeps its home
  });
});

// ── which documents hold analysis ───────────────────────────

describe("docsWithAnalysis / untitledAnalysisDocIds", () => {
  it("counts non-pristine workflows by owner; only Untitled ones are 'unsaved'", async () => {
    await open([A, U]);
    expect(docsWithAnalysis()).toEqual([]);
    st().setTabs([newWorkflowTab("Workflow 1", st().docUids[U.id]), wf("wfA", [src(A, "r0c0_i0")])]);
    expect(docsWithAnalysis()).toEqual([A.id]);
    expect(untitledAnalysisDocIds()).toEqual([]);
    st().setTabs((tabs) => [...tabs, wf("wfU", [upload("x.tif")], "X", { home: st().docUids[U.id] })]);
    expect(docsWithAnalysis()).toEqual([A.id, U.id]);
    expect(untitledAnalysisDocIds()).toEqual([U.id]);
  });
});

describe("rekeyAnalysisStorage", () => {
  it("moves names, measurement-group labels and folder keys; tells a mounted view", () => {
    const onMoved = vi.fn();
    window.addEventListener("mpfig:analysis-keys-moved", onMoved);
    localStorage.setItem("mpfig.source_names", JSON.stringify({ "d:doc_u::r0c0_i0": "Ctl", "p:/x/other.mpf::r0c0_i0": "Keep" }));
    localStorage.setItem("mpfig.measurement_groups", JSON.stringify({ "d:doc_u|R1C1|0|0": "Treated" }));
    localStorage.setItem("mpfig.source_groups", JSON.stringify([{ id: "g1", name: "G", sourceKeys: ["d:doc_u::r0c0_i0", "img:a.tif"] }]));
    expect(rekeyAnalysisStorage("d:doc_u", "p:/x/new.mpf", false)).toBe(true);
    expect(JSON.parse(localStorage.getItem("mpfig.source_names")!)).toEqual({ "p:/x/other.mpf::r0c0_i0": "Keep", "p:/x/new.mpf::r0c0_i0": "Ctl" });
    expect(JSON.parse(localStorage.getItem("mpfig.measurement_groups")!)).toEqual({ "p:/x/new.mpf|R1C1|0|0": "Treated" });
    expect(JSON.parse(localStorage.getItem("mpfig.source_groups")!)[0].sourceKeys).toEqual(["p:/x/new.mpf::r0c0_i0", "img:a.tif"]);
    expect(onMoved).toHaveBeenCalledTimes(1);
    expect(rekeyAnalysisStorage("d:doc_none", "p:/x/y.mpf", false)).toBe(false);
    expect(onMoved).toHaveBeenCalledTimes(1);
    window.removeEventListener("mpfig:analysis-keys-moved", onMoved);
  });
});

// ── figureStore wiring ──────────────────────────────────────

describe("figureStore save / load", () => {
  it("saveProject carries the document's previous file and then writes its workflows to the new path", async () => {
    await open([A]);
    st().setTabs([wf("wfA", [src(A, "r0c0_i0")])]);
    useFigureStore.setState({ config: { rows: 1, cols: 1, panels: [[{}]] } as never, currentProjectPath: A.path });
    const uid = st().docUids[A.id];
    await expect(useFigureStore.getState().saveProject("/exp/a2.mpf")).resolves.toBe(true);
    expect(mockApi.saveProject).toHaveBeenCalledWith("/exp/a2.mpf", { carryFrom: A.path, docUid: uid });
    expect(useCollageStore.getState().openDocs[0].path).toBe("/exp/a2.mpf");
    expect(fileManifest("/exp/a2.mpf")!.workflows.map((w) => w.id)).toEqual(["wfA"]);
  });

  it("saveProject of an Untitled document carries nothing (carryFrom null)", async () => {
    await open([U]);
    useFigureStore.setState({ config: { rows: 1, cols: 1, panels: [[{}]] } as never, currentProjectPath: null });
    await useFigureStore.getState().saveProject("/exp/u.mpf");
    expect(mockApi.saveProject).toHaveBeenCalledWith("/exp/u.mpf", { carryFrom: null, docUid: st().docUids[U.id] });
  });

  it("a failed figure save writes no workflows; a failed workflow write doesn't fail the figure save", async () => {
    await open([A]);
    st().setTabs([wf("wfA", [src(A, "r0c0_i0")])]);
    useFigureStore.setState({ config: { rows: 1, cols: 1, panels: [[{}]] } as never, currentProjectPath: A.path });
    mockApi.saveProject.mockRejectedValueOnce(new Error("no space"));
    await expect(useFigureStore.getState().saveProject(A.path!)).resolves.toBe(false);
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
    mockApi.writeWorkflows.mockRejectedValueOnce(new Error("locked"));
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    await expect(useFigureStore.getState().saveProject(A.path!)).resolves.toBe(true);
    err.mockRestore();
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("written");   // still dirty
  });

  it("loadProject (also every tab-switch reload) never reads or replaces workflows", async () => {
    await open([A]);
    st().setTabs([wf("wfA", [src(A, "r0c0_i0")])]);
    const tabs = st().tabs;
    mockApi.loadProject.mockResolvedValue({
      config: { rows: 1, cols: 1, spacing: 0, panels: [[{ image_name: "" }]], column_labels: [], row_labels: [], column_headers: [], row_headers: [] },
      image_names: [], thumbnails: {},
    });
    await useFigureStore.getState().loadProject(A.path!, A.id);
    await flush();
    expect(mockApi.readWorkflows).toHaveBeenCalledTimes(1);            // only the first open's read
    expect(st().tabs).toBe(tabs);
  });
});

describe("hydrateOpenDocs", () => {
  it("re-queues documents that are open but not read (e.g. after a reset)", async () => {
    session([A]);
    resetAnalysisWorkspaceForTests();                                  // drops the queued read
    await flush();
    expect(st().hydrated[A.id]).toBeUndefined();
    hydrateOpenDocs();
    await flush();
    expect(st().hydrated[A.id]).toBe("done");
  });
});

describe("tombstones stay with the document that recorded them", () => {
  it("deleting a workflow in a Save-As copy keeps it in the original", async () => {
    disk.set(A.path!, readBack(await payloadFor(A, [wf("wfA", [src(A, "r0c0_i0")])])));
    await open([A]);
    await st().afterDocSaved(A.id, A.path, "/exp/b.mpf");
    expect(st().deleteWorkflow("wfA")).toBe(true);
    await st().persistDocWorkflows(A.id);
    expect(fileManifest("/exp/b.mpf")!.workflows).toHaveLength(0);
    const orig: Doc = { id: "doc_orig", path: A.path, name: "a" };
    useCollageStore.setState({ openDocs: [...useCollageStore.getState().openDocs, orig] });
    await flush();
    expect(st().hydrated[orig.id]).toBe("done");
    expect(st().tabs.filter((t) => t.nodes.length > 1)).toHaveLength(1);
    await st().persistDocWorkflows(orig.id);
    expect(fileManifest(A.path!)!.workflows).toHaveLength(1);
  });

  it("deleting a workflow in a Finder copy keeps it in the original when both open later", async () => {
    const p = await payloadFor(A, [wf("wfA", [src(A, "r0c0_i0")])]);
    const twin: Doc = { id: "doc_twin", path: "/exp/copy-of-a.mpf", name: "copy-of-a" };
    disk.set(A.path!, readBack(p));
    disk.set(twin.path!, readBack(p));
    await open([twin]);
    expect(st().deleteWorkflow("wfA")).toBe(true);
    await st().persistDocWorkflows(twin.id);
    for (const order of [[twin, A], [A, twin]]) {
      disk.set(A.path!, readBack(p));
      await open(order.map((d) => ({ ...d, id: `${d.id}_2` })));
      expect(st().tabs.filter((t) => t.nodes.length > 1)).toHaveLength(1);
      await st().persistDocWorkflows(`${A.id}_2`);
      expect(fileManifest(A.path!)!.workflows).toHaveLength(1);
      expect(fileManifest(twin.path!)!.workflows).toHaveLength(0);
    }
  });
});

describe("a figure save that could not carry the saved Analysis", () => {
  it("writes the workflows again; a save that carried them stays clean", async () => {
    disk.set(A.path!, readBack(await payloadFor(A, [wf("wfA", [src(A, "r0c0_i0")])])));
    await open([A]);
    await st().afterDocSaved(A.id, A.path, A.path!, { workflowsCarried: true });
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
    disk.delete(A.path!);                                  // the re-save dropped workflows/
    await st().afterDocSaved(A.id, A.path, A.path!, { workflowsCarried: false });
    expect(mockApi.writeWorkflows).toHaveBeenCalledTimes(1);
    expect(fileManifest(A.path!)!.workflows).toHaveLength(1);
  });

  it("leaves a file that never had saved Analysis alone", async () => {
    await open([A]);
    await st().afterDocSaved(A.id, A.path, A.path!, { workflowsCarried: false });
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
  });
});
