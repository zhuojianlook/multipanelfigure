/* Shared workflows across .mpf documents (plan step 6): a workflow reading
   figures of several documents is LOCKED while one of them is closed — the
   single hard guard in setTabNodes / setTabEdges (view-only updates pass),
   deleteWorkflow refused — and the ways out: openMissing (saved path,
   relative path, same name next to an open owner; one pathsExist call; cold
   tabs; uid adoption for a file without one; confirm + rebind for a file
   with another), locate (native dialog) and removeMissingSources. Results
   that come back unlocked are flagged "needs re-run"; merge keeps the newer
   copy and honours tombstones. The sidecar is a fake in-memory "disk". */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { Edge, Node } from "@xyflow/react";
import type { WorkflowsPayload, WorkflowsReadResponse } from "../api/types";
import type { InsetSource, NodeData, NodeOutput, WorkflowTab } from "../utils/analysisTypes";
import { figKey, nsKey } from "../utils/analysisKeys";
import { computeStale } from "../utils/analysisStale";
import { closedFigKey, isLocked } from "../utils/workflowOwnership";
import type { WorkflowsManifestV1 } from "../utils/workflowPersist";

// ── fakes: sidecar disk, file existence, dialogs ────────────

const disk = vi.hoisted(() => new Map<string, unknown>());
/** .mpf files that exist (with or without a workflows/ section). */
const files = vi.hoisted(() => new Set<string>());
const mockApi = vi.hoisted(() => ({
  readWorkflows: vi.fn(),
  writeWorkflows: vi.fn(),
  pathsExist: vi.fn(),
  loadProject: vi.fn(),
}));
vi.mock("../api/client", () => ({ api: mockApi, default: vi.fn() }));
const mockConfirm = vi.hoisted(() => vi.fn(async () => true));
vi.mock("../components/shared/ConfirmDialog", () => ({
  confirm: mockConfirm, confirmThree: vi.fn(), alert: vi.fn(),
}));
const mockOpenDialog = vi.hoisted(() => vi.fn(async (): Promise<string | null> => null));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: mockOpenDialog }));

import { useCollageStore } from "./collageStore";
import { useSourceEditStore } from "./sourceEditStore";
import {
  useAnalysisWorkspaceStore, resetAnalysisWorkspaceForTests, newSourceNode, workspaceOwnershipCtx,
  workflowLockedNow, missingDocsOf, claimLockedPrompt, closedDocLabel,
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
function edge(source: string, target: string, handle = "out_image_0"): Edge {
  return { id: `e_${source}_${handle}__${target}_in_image`, source, sourceHandle: handle, target, targetHandle: "in_image", type: "deletable" };
}
/** A workflow with one source node holding `sources`, wired to result node p1. */
function wf(id: string, sources: InsetSource[], payload = "PNG_RESULT", extra: Partial<WorkflowTab> = {}): WorkflowTab {
  const ranWith = Object.fromEntries(sources.map((s) => [s.key, 0]));
  return {
    id, name: `Analysis ${id}`, updatedAt: 1_000,
    nodes: [newSourceNode(sources), procNode("p1", { outputs: [output("plot1", payload)], status: "ok", ranWith })],
    edges: [edge("source", "p1")],
    ...extra,
  };
}
/** A shared workflow: A's figure on source node "source", B's on "srcB",
 *  each wired to its own result node. */
function sharedWf(id: string, a: Doc, b: Doc, extra: Partial<WorkflowTab> = {}): WorkflowTab {
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
    ...extra,
  };
}

const st = () => useAnalysisWorkspaceStore.getState();
const tab = (id: string) => st().tabs.find((t) => t.id === id);
const ctx = () => workspaceOwnershipCtx();
const locked = (id: string) => isLocked(tab(id)!, ctx());
const docAt = (path: string) => useCollageStore.getState().openDocs.find((d) => d.path === path);

/** What the sidecar's read returns for a payload it wrote. */
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
const manifestAt = (path: string) => (disk.get(path) as WorkflowsReadResponse | undefined)?.manifest as WorkflowsManifestV1 | undefined;

async function flush(n = 80) { for (let i = 0; i < n; i++) await Promise.resolve(); }

/** Fresh session with `docs` open (A active unless given), their files read. */
async function open(docs: Doc[], active: string = docs[0]?.id ?? A.id) {
  useCollageStore.setState({ openDocs: [], activeDocId: null });
  resetAnalysisWorkspaceForTests();
  useCollageStore.setState({ openDocs: docs, activeDocId: active });
  await flush();
}
/** Close document `doc` the way projectNav does (workspace first, then the tab). */
function closeDoc(doc: Doc) {
  st().closeDoc(doc.id);
  useCollageStore.getState().docRemove(doc.id);
}
/** Session 1: A and B open, `tabs` set, both files written. */
async function writeBoth(tabs: WorkflowTab[]): Promise<{ ua: string; ub: string }> {
  await open([A, B]);
  st().setTabs(tabs);
  await st().persistDocWorkflows(A.id);
  await st().persistDocWorkflows(B.id);
  return { ua: st().docUids[A.id], ub: st().docUids[B.id] };
}
/** Move a file on the fake disk. */
function move(from: string, to: string) {
  disk.set(to, disk.get(from));
  disk.delete(from);
  files.delete(from);
  files.add(to);
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
  vi.setSystemTime(10_000);
  vi.clearAllMocks();
  disk.clear();
  files.clear();
  localStorage.clear();
  mockApi.readWorkflows.mockImplementation(async (path: string) => (disk.get(path) as WorkflowsReadResponse) ?? EMPTY);
  mockApi.writeWorkflows.mockImplementation(async (path: string, payload: WorkflowsPayload) => {
    disk.set(path, readBack(payload));
    files.add(path);
    return { ok: true, size_bytes: 1, workflows_bytes: 1, missing_uploads: [], fast_path: true };
  });
  mockApi.pathsExist.mockImplementation(async (paths: string[]) =>
    Object.fromEntries(paths.map((p) => [p, files.has(p) || disk.has(p)])));
  mockConfirm.mockImplementation(async () => true);
  mockOpenDialog.mockImplementation(async () => null);
  useSourceEditStore.setState({ revs: {}, withResults: [], rerun: {} });
});
afterEach(() => {
  useCollageStore.setState({ openDocs: [], activeDocId: null });
  resetAnalysisWorkspaceForTests();
  vi.useRealTimers();
});

// ── the lock guard ──────────────────────────────────────────

describe("a locked workflow is read-only (the single hard guard)", () => {
  it("setTabNodes / setTabEdges drop edits to a locked tab; the updater still runs; view-only updates pass", async () => {
    await open([A, B]);
    st().setTabs([sharedWf("wfAB", A, B), wf("wfA", [src(A, "r1c0_i0")])]);
    closeDoc(B);
    expect(locked("wfAB")).toBe(true);
    expect(workflowLockedNow("wfAB")).toBe(true);
    expect(workflowLockedNow("wfA")).toBe(false);
    const before = st().tabs;
    let notified = 0;
    const stop = useAnalysisWorkspaceStore.subscribe(() => { notified++; });
    const seen = vi.fn();
    // An edit (code) and a structural change (edge removal): dropped.
    st().setTabNodes("wfAB", (nodes) => { seen(nodes.length); return nodes.map((n) => (n.id === "pA" ? { ...n, data: { ...n.data, code: "# edited" } } : n)); });
    st().setTabEdges("wfAB", () => []);
    expect(seen).toHaveBeenCalledWith(4);           // Run graph reads its snapshot through the updater
    expect(st().tabs).toBe(before);
    expect(notified).toBe(0);
    // Selection / measured size (ReactFlow view state) still apply.
    st().setTabNodes("wfAB", (nodes) => nodes.map((n) => (n.id === "pA" ? { ...n, selected: true } : n)), { view: true });
    st().setTabEdges("wfAB", (edges) => edges.map((e) => ({ ...e, selected: true })), { view: true });
    expect(tab("wfAB")!.nodes.find((n) => n.id === "pA")!.selected).toBe(true);
    expect(tab("wfAB")!.edges.every((e) => e.selected)).toBe(true);
    // The unlocked tab next to it is untouched by the guard.
    st().setTabNodes("wfA", (nodes) => nodes.map((n) => (n.id === "p1" ? { ...n, data: { ...n.data, code: "# ok" } } : n)));
    expect(tab("wfA")!.nodes.find((n) => n.id === "p1")!.data.code).toBe("# ok");
    stop();
  });

  it("deleting a locked workflow is refused; once its document is back it can go", async () => {
    await open([A, B]);
    st().setTabs([sharedWf("wfAB", A, B), wf("wfA", [src(A, "r1c0_i0")])]);
    await st().persistDocWorkflows(B.id);
    closeDoc(B);
    expect(st().deleteWorkflow("wfAB")).toBe(false);
    expect(tab("wfAB")).toBeDefined();
    expect(st().tombstones).toEqual({});
    files.add(B.path!);
    await st().openMissing("wfAB");
    expect(locked("wfAB")).toBe(false);
    expect(st().deleteWorkflow("wfAB")).toBe(true);
    expect(tab("wfAB")).toBeUndefined();
    expect(Object.keys(st().tombstones).sort()).toEqual([st().docUids[A.id], st().docUids[docAt(B.path!)!.id]].sort());
  });

  it("closing a document under a running workflow leaves no node 'running'; the run's late write is dropped", async () => {
    await open([A, B]);
    const t = sharedWf("wfAB", A, B);
    t.nodes[2] = { ...t.nodes[2], data: { ...t.nodes[2].data, status: "running" } };
    t.nodes[3] = { ...t.nodes[3], data: { ...t.nodes[3].data, status: "running", outputs: [] } };
    st().setTabs([t]);
    closeDoc(B);
    const after = tab("wfAB")!;
    expect(after.nodes.find((n) => n.id === "pA")!.data.status).toBe("stale");   // has results
    expect(after.nodes.find((n) => n.id === "pB")!.data.status).toBe("idle");    // has none
    // The run finishing now can't write into it.
    st().setTabNodes("wfAB", (nodes) => nodes.map((n) => (n.id === "pB" ? { ...n, data: { ...n.data, status: "ok", outputs: [output("late", "LATE")] } } : n)));
    expect(tab("wfAB")!.nodes.find((n) => n.id === "pB")!.data.outputs).toEqual([]);
  });
});

// ── naming what is missing ──────────────────────────────────

describe("missingDocsOf / closedDocLabel / claimLockedPrompt", () => {
  it("names a closed document by its file, a never-saved one by its name; only files are 'locatable'", async () => {
    await open([A, B, U]);
    const sb = src(B, "r0c0_i0");
    const su = src(U, "r0c0_i0");
    st().setTabs([wf("wf3", [src(A, "r0c0_i0"), sb, su])]);
    const ub = st().docUids[B.id];
    const uu = st().docUids[U.id];
    closeDoc(B);
    closeDoc(U);
    const miss = missingDocsOf(tab("wf3")!, ctx());
    expect(miss).toEqual([
      { uid: ub, label: "b.mpf", locatable: true },
      { uid: uu, label: "Untitled_1 (never saved)", locatable: false },
    ]);
    expect(closedDocLabel("du_nobody", {})).toBe("a closed document");
  });

  it("the 'open the other documents?' prompt is claimed once per workflow per session", () => {
    expect(claimLockedPrompt("wf1")).toBe(true);
    expect(claimLockedPrompt("wf1")).toBe(false);
    expect(claimLockedPrompt("wf2")).toBe(true);
    resetAnalysisWorkspaceForTests();
    expect(claimLockedPrompt("wf1")).toBe(true);
  });
});

// ── Open missing ────────────────────────────────────────────

describe("openMissing", () => {
  it("opens the closed document from its saved path as a cold tab; its read unlocks the workflow", async () => {
    const { ub } = await writeBoth([sharedWf("wfAB", A, B)]);
    await open([A]);                                                  // next launch: only a.mpf
    expect(locked("wfAB")).toBe(true);
    expect(st().knownDocs[ub]).toMatchObject({ name: "b", path: B.path });
    await st().openMissing("wfAB");
    expect(mockApi.pathsExist).toHaveBeenCalledTimes(1);
    expect(mockApi.pathsExist).toHaveBeenCalledWith([B.path]);
    const b2 = docAt(B.path!)!;
    expect(b2).toBeDefined();
    expect(useCollageStore.getState().activeDocId).toBe(A.id);        // a cold tab: no builder swap
    expect(mockApi.loadProject).not.toHaveBeenCalled();
    expect(st().docUids[b2.id]).toBe(ub);
    expect(locked("wfAB")).toBe(false);
    const keys = tab("wfAB")!.nodes.flatMap((n) => (n.data.sources || []).map((s) => s.key));
    expect(keys).toEqual([nsKey(figKey(A), "r0c0_i0"), nsKey(figKey(b2), "r0c1_i0")]);
    expect(mockConfirm).not.toHaveBeenCalled();
    expect(mockOpenDialog).not.toHaveBeenCalled();
  });

  it("files moved together: the path relative to the open owner's file is tried next (one pathsExist call)", async () => {
    await writeBoth([sharedWf("wfAB", A, B)]);
    move(A.path!, "/moved/a.mpf");
    move(B.path!, "/moved/b.mpf");
    const A2: Doc = { id: "doc_a2", path: "/moved/a.mpf", name: "a" };
    await open([A2]);
    expect(locked("wfAB")).toBe(true);
    await st().openMissing("wfAB");
    expect(mockApi.pathsExist).toHaveBeenCalledTimes(1);
    expect(mockApi.pathsExist).toHaveBeenCalledWith(["/exp/b.mpf", "/moved/b.mpf"]);
    expect(docAt("/moved/b.mpf")).toBeDefined();
    expect(locked("wfAB")).toBe(false);
  });

  it("a file kept elsewhere: the same file name next to the open owner's file is tried last", async () => {
    const B3: Doc = { id: "doc_b3", path: "/other/b.mpf", name: "b" };
    await open([A, B3]);
    st().setTabs([sharedWf("wfAB", A, B3)]);
    await st().persistDocWorkflows(A.id);
    await st().persistDocWorkflows(B3.id);
    move(B3.path!, "/exp/b.mpf");                                     // now next to a.mpf
    await open([A]);
    await st().openMissing("wfAB");
    expect(mockApi.pathsExist).toHaveBeenCalledWith(["/other/b.mpf", "/exp/b.mpf"]);
    expect(docAt("/exp/b.mpf")).toBeDefined();
    expect(locked("wfAB")).toBe(false);
  });

  it("a document closed this session is looked for at its path, then by name next to the open owner's file", async () => {
    const B3: Doc = { id: "doc_b3", path: "/other/b.mpf", name: "b" };
    await open([A, B3]);
    st().setTabs([sharedWf("wfAB", A, B3)]);
    await st().persistDocWorkflows(B3.id);
    closeDoc(B3);                                                     // known from the close: a path, no candidates
    move(B3.path!, "/exp/b.mpf");
    await st().openMissing("wfAB");
    expect(mockApi.pathsExist).toHaveBeenCalledWith(["/other/b.mpf", "/exp/b.mpf"]);
    expect(docAt("/exp/b.mpf")).toBeDefined();
    expect(locked("wfAB")).toBe(false);
  });

  it("a file without a uid (re-saved by an older build) is taken as the expected document", async () => {
    await open([A, B]);
    st().setTabs([sharedWf("wfAB", A, B)]);
    const ub = st().docUids[B.id];
    await st().persistDocWorkflows(A.id);
    files.add(B.path!);                                               // b.mpf exists, no workflows/ in it
    await open([A]);
    await st().openMissing("wfAB");
    const b2 = docAt(B.path!)!;
    expect(st().docUids[b2.id]).toBe(ub);                             // adopted
    expect(mockConfirm).not.toHaveBeenCalled();
    expect(locked("wfAB")).toBe(false);
    // Its next persist writes the workflow (and the uid) into b.mpf.
    await expect(st().persistDocWorkflows(b2.id)).resolves.toBe("written");
    expect(manifestAt(B.path!)!.docUid).toBe(ub);
    expect(manifestAt(B.path!)!.workflows.map((w) => w.id)).toEqual(["wfAB"]);
  });

  it("a file naming another uid is used only after the user confirms; then everything is re-bound to it", async () => {
    const { ub } = await writeBoth([sharedWf("wfAB", A, B)]);
    // b.mpf was replaced by a different document since.
    await open([B]);
    st().setTabs([wf("wfOther", [src(B, "r5c5_i0")])]);
    const uOther = "du_other_b";
    st().adoptUid(B.id, uOther);
    await st().persistDocWorkflows(B.id, { force: true });
    expect(manifestAt(B.path!)!.docUid).toBe(uOther);
    await open([A]);
    mockConfirm.mockImplementationOnce(async () => false);
    await st().openMissing("wfAB");
    expect(mockConfirm).toHaveBeenCalledTimes(1);
    const b2 = docAt(B.path!)!;
    expect(b2).toBeDefined();                                         // opened, but not used
    expect(locked("wfAB")).toBe(true);
    expect(missingDocsOf(tab("wfAB")!, ctx()).map((m) => m.uid)).toEqual([ub]);
    // Asked again, the user agrees.
    await st().openMissing("wfAB");
    expect(mockConfirm).toHaveBeenCalledTimes(2);
    expect(locked("wfAB")).toBe(false);
    const t = tab("wfAB")!;
    expect(t.nodes.find((n) => n.id === "srcB")!.data.sources![0].key).toBe(nsKey(figKey(b2), "r0c1_i0"));
    expect(st().docUids[b2.id]).toBe(uOther);
    expect(tab("wfOther")).toBeDefined();                             // the file's own workflow is there too
  });

  it("a file whose saved analysis is a newer format isn't used: no prompt, still locked", async () => {
    await writeBoth([sharedWf("wfAB", A, B)]);
    disk.set(B.path!, { ...EMPTY, unsupported: true, version: 2 });
    await open([A]);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    await st().openMissing("wfAB");
    const b2 = docAt(B.path!)!;
    expect(st().hydrated[b2.id]).toBe("unsupported");
    expect(mockConfirm).not.toHaveBeenCalled();
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
    expect(locked("wfAB")).toBe(true);
  });

  it("not found anywhere: the native dialog asks for it ('Locate b.mpf'), starting next to the open owner", async () => {
    const { ub } = await writeBoth([sharedWf("wfAB", A, B)]);
    move(B.path!, "/somewhere/else/renamed.mpf");
    await open([A]);
    mockOpenDialog.mockImplementationOnce(async () => "/somewhere/else/renamed.mpf");
    await st().openMissing("wfAB");
    expect(mockOpenDialog).toHaveBeenCalledTimes(1);
    expect(mockOpenDialog).toHaveBeenCalledWith(expect.objectContaining({
      title: "Locate b.mpf", multiple: false, defaultPath: "/exp",
      filters: [{ name: "Project", extensions: ["mpf"] }],
    }));
    const b2 = docAt("/somewhere/else/renamed.mpf")!;
    expect(st().docUids[b2.id]).toBe(ub);
    expect(locked("wfAB")).toBe(false);
  });

  it("locate: cancelling opens nothing; a document already back is not asked for", async () => {
    await writeBoth([sharedWf("wfAB", A, B)]);
    await open([A]);
    const ub = missingDocsOf(tab("wfAB")!, ctx())[0].uid;
    await st().locate("wfAB", ub);                                    // dialog → null
    expect(mockOpenDialog).toHaveBeenCalledTimes(1);
    expect(useCollageStore.getState().openDocs).toHaveLength(1);
    expect(locked("wfAB")).toBe(true);
    await st().openMissing("wfAB");                                   // found at its path
    mockOpenDialog.mockClear();
    await st().locate("wfAB", ub);
    expect(mockOpenDialog).not.toHaveBeenCalled();
  });

  it("a document that was never saved has no file to look for: no lookup, no dialog", async () => {
    await open([A, U]);
    st().setTabs([wf("wfAU", [src(A, "r0c0_i0"), src(U, "r0c0_i0")])]);
    closeDoc(U);
    expect(locked("wfAU")).toBe(true);
    await st().openMissing("wfAU");
    expect(mockApi.pathsExist).not.toHaveBeenCalled();
    expect(mockOpenDialog).not.toHaveBeenCalled();
    expect(locked("wfAU")).toBe(true);
  });

  it("results a workflow held while locked come back flagged 'needs re-run'", async () => {
    await open([A, B]);
    st().setTabs([sharedWf("wfAB", A, B)]);
    await st().persistDocWorkflows(B.id);
    closeDoc(B);
    expect(tab("wfAB")!.nodes.find((n) => n.id === "pA")!.data.outputs![0].payload).toBe("RA");   // kept in memory
    await st().openMissing("wfAB");
    const t = tab("wfAB")!;
    expect(locked("wfAB")).toBe(false);
    expect([...computeStale(t.nodes, t.edges, useSourceEditStore.getState().revs).nodeIds].sort()).toEqual(["pA", "pB"]);
    expect(t.nodes.filter((n) => n.data.kind !== "source").map((n) => n.data.status)).toEqual(["stale", "stale"]);
    expect(t.nodes.find((n) => n.id === "pB")!.data.outputs![0].payload).toBe("RB");
  });
});

// ── Remove their sources ────────────────────────────────────

describe("removeMissingSources", () => {
  it("drops the closed document's sources: single-owner, editable, edges re-indexed, results stale", async () => {
    await open([A, B]);
    const sa = src(A, "r0c0_i0");
    const sb = src(B, "r0c1_i0");
    const sa2 = src(A, "r0c2_i0");
    // One source node mixing both documents: B's image sits between A's two.
    const t: WorkflowTab = {
      id: "wfMix", name: "Mixed", updatedAt: 1_000, home: undefined,
      nodes: [newSourceNode([sa, sb, sa2]), procNode("p1", { outputs: [output("o", "R")], status: "ok", ranWith: { [sa.key]: 0, [sb.key]: 0, [sa2.key]: 0 } })],
      edges: [edge("source", "p1", "out_image_0"), edge("source", "p1", "out_image_1"), edge("source", "p1", "out_image_2")],
    };
    st().setTabs([t]);
    const ub = st().docUids[B.id];
    closeDoc(B);
    expect(locked("wfMix")).toBe(true);
    st().removeMissingSources("wfMix");
    const after = tab("wfMix")!;
    expect(locked("wfMix")).toBe(false);
    expect(after.nodes[0].data.sources!.map((s) => s.key)).toEqual([sa.key, sa2.key]);
    expect(after.edges.map((e) => e.sourceHandle).sort()).toEqual(["out_image_0", "out_image_1"]);
    expect(after.nodes[0].data.sources!.some((s) => s.key.startsWith(closedFigKey(ub)))).toBe(false);
    expect(after.nodes[1].data.status).toBe("stale");
    expect([...computeStale(after.nodes, after.edges, {}).nodeIds]).toEqual(["p1"]);
    // Now A's alone: a single-owner workflow A saves with its results.
    await st().persistDocWorkflows(A.id);
    const m = manifestAt(A.path!)!;
    expect(m.workflows[0]).toMatchObject({ id: "wfMix", shared: false, owners: [st().docUids[A.id]] });
    expect(m.workflows[0].nodes.find((n) => n.id === "p1")!.data.outputs).toHaveLength(1);
  });

  it("an unlocked workflow is left alone", async () => {
    await open([A, B]);
    st().setTabs([sharedWf("wfAB", A, B)]);
    const before = st().tabs;
    st().removeMissingSources("wfAB");
    expect(st().tabs).toBe(before);
  });

  it("a never-saved document's sources: the only way out, and it works", async () => {
    await open([A, U]);
    st().setTabs([wf("wfAU", [src(A, "r0c0_i0"), src(U, "r0c0_i0")])]);
    closeDoc(U);
    st().removeMissingSources("wfAU");
    expect(locked("wfAU")).toBe(false);
    expect(tab("wfAU")!.nodes[0].data.sources!.map((s) => s.key)).toEqual([nsKey(figKey(A), "r0c0_i0")]);
  });
});

// ── merge across files: newer copy, tombstones ──────────────

describe("copies of a shared workflow in several files", () => {
  it("the newer copy wins when the other document opens", async () => {
    await writeBoth([sharedWf("wfAB", A, B)]);                         // both copies at t=1000
    // A later session edits it with both open, but only a.mpf gets written.
    await open([A, B]);
    vi.advanceTimersByTime(600);                                      // the copies as read are seen
    vi.setSystemTime(50_000);
    st().setTabNodes("wfAB", (nodes) => nodes.map((n) => (n.id === "pA" ? { ...n, data: { ...n.data, code: "# newer" } } : n)));
    vi.advanceTimersByTime(600);
    expect(tab("wfAB")!.updatedAt).toBeGreaterThan(1_000);
    await st().persistDocWorkflows(A.id);
    // Next launch: b.mpf first (its older copy, locked), then a.mpf through Open missing.
    await open([B]);
    expect(tab("wfAB")!.nodes.find((n) => n.id === "pA")!.data.code).toBe("# pA");
    expect(locked("wfAB")).toBe(true);
    await st().openMissing("wfAB");
    expect(locked("wfAB")).toBe(false);
    expect(tab("wfAB")!.nodes.find((n) => n.id === "pA")!.data.code).toBe("# newer");
    // b.mpf held the losing copy: it is dirty and gets rewritten.
    await expect(st().persistDocWorkflows(B.id)).resolves.toBe("written");
  });

  it("a workflow deleted later doesn't come back from a closed document's stale copy", async () => {
    await writeBoth([sharedWf("wfAB", A, B)]);                         // both files hold it
    // Next session: only a.mpf; its sources of b are removed, then it is deleted.
    await open([A]);
    st().removeMissingSources("wfAB");
    vi.setSystemTime(60_000);
    expect(st().deleteWorkflow("wfAB")).toBe(true);
    await st().persistDocWorkflows(A.id);
    expect(manifestAt(A.path!)!.tombstones).toEqual({ wfAB: 60_000 });
    // b.mpf still holds the old shared copy: opened alone it shows (locked)…
    await open([B]);
    expect(tab("wfAB")).toBeDefined();
    expect(locked("wfAB")).toBe(true);
    // …until a.mpf, which recorded the deletion, is open too.
    await st().openMissing("wfAB");
    expect(docAt(A.path!)).toBeDefined();
    expect(tab("wfAB")).toBeUndefined();
    await expect(st().persistDocWorkflows(B.id)).resolves.toBe("written");
    expect(manifestAt(B.path!)!.workflows.map((w) => w.id)).toEqual([]);
  });
});
