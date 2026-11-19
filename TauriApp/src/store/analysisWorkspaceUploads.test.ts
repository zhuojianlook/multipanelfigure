/* Uploaded images in saved workflows (plan step 7): which .mpf embeds a
   workflow's uploads — the upload-home prompt (ChooseDocumentDialog) for a
   workflow that reads no figure document, "Store uploaded images in ▸" for a
   shared one — and how a reopened file's uploads come back (renamed on a
   name clash, tiles filed in "Direct upload"). The sidecar is a fake
   in-memory "disk"; the dialog is mocked. */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { Edge, Node } from "@xyflow/react";
import type { WorkflowsPayload, WorkflowsReadResponse } from "../api/types";
import type { InsetSource, NodeData, NodeOutput, WorkflowTab } from "../utils/analysisTypes";
import { figKey, nsKey } from "../utils/analysisKeys";
import type { DocWorkflowsPayload, WorkflowsManifestV1 } from "../utils/workflowPersist";
import type { ChooseDocumentResult } from "../components/shared/ChooseDocumentDialog";

const disk = vi.hoisted(() => new Map<string, unknown>());
const mockApi = vi.hoisted(() => ({ readWorkflows: vi.fn(), writeWorkflows: vi.fn(), pathsExist: vi.fn() }));
vi.mock("../api/client", () => ({ api: mockApi, default: vi.fn() }));
const mockChoose = vi.hoisted(() => vi.fn(async (_o: unknown): Promise<ChooseDocumentResult> => null));
vi.mock("../components/shared/ChooseDocumentDialog", () => ({ chooseDocument: mockChoose }));

import { useCollageStore } from "./collageStore";
import { useSourceEditStore } from "./sourceEditStore";
import {
  useAnalysisWorkspaceStore, resetAnalysisWorkspaceForTests, newSourceNode, workspaceOwnershipCtx,
  uploadHomeOptions, untitledAnalysisDocIds, ANALYSIS_KEYS_MOVED_EVENT,
} from "./analysisWorkspaceStore";

// ── fixtures ────────────────────────────────────────────────

type Doc = { id: string; path: string | null; name: string };
const A: Doc = { id: "doc_a", path: "/exp/a.mpf", name: "a" };
const B: Doc = { id: "doc_b", path: "/exp/b.mpf", name: "b" };

const EMPTY: WorkflowsReadResponse = {
  manifest: null, blobs: {}, unsupported: false, version: null, uploads: [], renamed: {},
  missing_uploads: [], failed_uploads: [],
};

function src(doc: { id: string; path?: string | null; name?: string }, raw: string): InsetSource {
  const fk = figKey(doc);
  return {
    key: nsKey(fk, raw), rawKey: raw, figKey: fk, mpfId: doc.id, mpf: doc.name,
    row: 0, col: 0, inset_index: 0, label: raw, natural_width: 10, natural_height: 10, thumbnail: "THUMB", runKey: raw,
  };
}
function upload(name: string): InsetSource {
  return { key: `img:${name}`, name, row: -1, col: -1, inset_index: -1, label: name, natural_width: 5, natural_height: 5, thumbnail: "UP" };
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
function wf(id: string, sources: InsetSource[], extra: Partial<WorkflowTab> = {}): WorkflowTab {
  return {
    id, name: `Analysis ${id}`, updatedAt: 1_000,
    nodes: [newSourceNode(sources), procNode("p1", { outputs: [output("plot1", "PNG")], status: "ok" })],
    edges: [edge("source", "p1")],
    ...extra,
  };
}
/** A's figure + an upload on one source node, B's figure on another. */
function sharedWithUpload(id: string): WorkflowTab {
  return {
    id, name: `Shared ${id}`, updatedAt: 1_000,
    nodes: [
      newSourceNode([src(A, "r0c0_i0"), upload("stack.tif")]),
      newSourceNode([src(B, "r0c1_i0")], { id: "srcB", label: "Source 2", position: { x: 30, y: 220 } }),
      procNode("pA"),
    ],
    edges: [edge("source", "pA")],
  };
}

const st = () => useAnalysisWorkspaceStore.getState();
const tab = (id: string) => st().tabs.find((t) => t.id === id)!;
const uid = (d: Doc) => st().docUids[d.id];

function readBack(p: WorkflowsPayload): WorkflowsReadResponse {
  const blobs: WorkflowsReadResponse["blobs"] = {};
  for (const [id, b] of Object.entries(p.blobs)) blobs[id] = { kind: b.kind, data: b.data ?? "" };
  return {
    ...EMPTY, manifest: JSON.parse(JSON.stringify(p.manifest)), blobs, version: 1,
    uploads: p.uploads.map((name) => ({ name, thumbnail: "UPTHUMB", size: [5, 5] as [number, number] })),
  };
}
/** The last workflows payload written to `path`. */
function written(path: string): DocWorkflowsPayload | undefined {
  const calls = mockApi.writeWorkflows.mock.calls.filter((c) => c[0] === path);
  return calls.length ? (calls[calls.length - 1][1] as DocWorkflowsPayload) : undefined;
}
const manifestAt = (path: string) => (disk.get(path) as WorkflowsReadResponse | undefined)?.manifest as WorkflowsManifestV1 | undefined;

async function flush(n = 80) { for (let i = 0; i < n; i++) await Promise.resolve(); }
async function open(docs: Doc[], active: string = docs[0]?.id ?? A.id) {
  useCollageStore.setState({ openDocs: [], activeDocId: null });
  resetAnalysisWorkspaceForTests();
  useCollageStore.setState({ openDocs: docs, activeDocId: active });
  await flush();
}
const groups = () => JSON.parse(localStorage.getItem("mpfig.source_groups") || "[]");

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
  vi.setSystemTime(10_000);
  vi.clearAllMocks();
  disk.clear();
  localStorage.clear();
  mockChoose.mockImplementation(async () => null);
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

// ── the upload-home prompt ──────────────────────────────────

describe("askUploadHome: a workflow using uploaded images only asks which document saves it", () => {
  it("asks once, with the open documents and its current home pre-selected; the chosen file embeds the uploads", async () => {
    await open([A, B]);
    st().setTabs([wf("wfU", [upload("stack.tif")], { home: uid(A) })]);
    mockChoose.mockResolvedValueOnce({ kind: "doc", id: B.id });
    await expect(st().askUploadHome("wfU", { label: (id) => `label:${id}` })).resolves.toBe(true);
    expect(mockChoose).toHaveBeenCalledTimes(1);
    expect(mockChoose.mock.calls[0][0]).toMatchObject({
      options: [
        { id: A.id, label: `label:${A.id}`, detail: "a.mpf" },
        { id: B.id, label: `label:${B.id}`, detail: "b.mpf" },
      ],
      selectedId: A.id,
      newLabel: "New document…",
    });
    expect(tab("wfU")).toMatchObject({ home: uid(B), homeConfirmed: true });
    // Asked once: a second upload attached later doesn't ask again.
    await expect(st().askUploadHome("wfU")).resolves.toBe(false);
    expect(mockChoose).toHaveBeenCalledTimes(1);
    // B's file holds the workflow and embeds the upload; A's file is not touched.
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("clean");
    await expect(st().persistDocWorkflows(B.id)).resolves.toBe("written");
    expect(written(B.path!)!.uploads).toEqual(["stack.tif"]);
    expect(manifestAt(B.path!)!.workflows.map((w) => [w.id, w.home, w.homeConfirmed])).toEqual([["wfU", uid(B), true]]);
    expect(written(A.path!)).toBeUndefined();
  });

  it("New document… adds a cold Untitled tab that saves it (and is then 'unsaved' on close / quit)", async () => {
    await open([A]);
    st().setTabs([wf("wfU", [upload("stack.tif")], { home: uid(A) })]);
    mockChoose.mockResolvedValueOnce({ kind: "new" });
    await expect(st().askUploadHome("wfU")).resolves.toBe(true);
    const docs = useCollageStore.getState().openDocs;
    expect(docs).toHaveLength(2);
    const fresh = docs[1];
    expect(fresh.path).toBeNull();
    expect(useCollageStore.getState().activeDocId).toBe(A.id);      // a cold tab — no switch
    expect(tab("wfU").home).toBe(st().docUids[fresh.id]);
    expect(untitledAnalysisDocIds()).toEqual([fresh.id]);
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("clean");
  });

  it("Cancel keeps its home (never ownerless) and isn't asked again this session — unless forced (tab menu)", async () => {
    await open([A, B]);
    st().setTabs([wf("wfU", [upload("stack.tif")], { home: uid(A) })]);
    await expect(st().askUploadHome("wfU")).resolves.toBe(false);
    expect(tab("wfU").home).toBe(uid(A));
    expect(tab("wfU").homeConfirmed).toBeUndefined();
    await st().askUploadHome("wfU");
    expect(mockChoose).toHaveBeenCalledTimes(1);
    mockChoose.mockResolvedValueOnce({ kind: "doc", id: B.id });
    await expect(st().askUploadHome("wfU", { force: true })).resolves.toBe(true);
    expect(tab("wfU").home).toBe(uid(B));
  });

  it("never asks for a workflow that reads a figure, already chose, holds no upload, or is locked", async () => {
    await open([A, B]);
    st().setTabs([
      wf("wfFig", [src(A, "r0c0_i0"), upload("stack.tif")]),
      wf("wfDone", [upload("stack.tif")], { home: uid(B), homeConfirmed: true }),
      wf("wfNone", [], { home: uid(A), name: "Named" }),
      wf("wfGone", [upload("stack.tif")], { home: "du_closed_doc" }),
    ]);
    for (const id of ["wfFig", "wfDone", "wfNone", "wfGone"]) await expect(st().askUploadHome(id)).resolves.toBe(false);
    expect(mockChoose).not.toHaveBeenCalled();
  });
});

// ── "Store uploaded images in ▸" ────────────────────────────

describe("setUploadHome / uploadHomeOptions: which document embeds a shared workflow's uploads", () => {
  it("the tab menu offers a shared workflow's documents; an uploads-only one the dialog; nothing otherwise", async () => {
    await open([A, B]);
    const ctx = () => workspaceOwnershipCtx();
    expect(uploadHomeOptions(sharedWithUpload("s"), ctx())).toMatchObject({
      kind: "shared", home: uid(A), choices: [{ id: A.id, uid: uid(A) }, { id: B.id, uid: uid(B) }],
    });
    expect(uploadHomeOptions(wf("u", [upload("x.tif")], { home: uid(B) }), ctx())).toEqual({ kind: "uploads-only", home: uid(B) });
    expect(uploadHomeOptions(wf("f", [src(A, "r0c0_i0"), upload("x.tif")]), ctx())).toBeNull();
    expect(uploadHomeOptions(wf("n", [src(A, "r0c0_i0"), src(B, "r0c0_i0")]), ctx())).toBeNull();
  });

  it("moves the embedded uploads to the chosen document — one of the workflow's own; not while locked", async () => {
    await open([A, B]);
    st().setTabs([sharedWithUpload("wfS")]);
    await st().persistDocWorkflows(A.id);
    await st().persistDocWorkflows(B.id);
    expect(written(A.path!)!.uploads).toEqual(["stack.tif"]);       // home: the first document it drew from
    expect(written(B.path!)!.uploads).toEqual([]);
    expect(st().setUploadHome("wfS", "du_not_a_participant")).toBe(false);
    expect(st().setUploadHome("wfS", uid(B))).toBe(true);
    expect(tab("wfS")).toMatchObject({ home: uid(B), homeConfirmed: true });
    expect(st().setUploadHome("wfS", uid(B))).toBe(false);           // already there
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("written");
    await expect(st().persistDocWorkflows(B.id)).resolves.toBe("written");
    expect(written(A.path!)!.uploads).toEqual([]);
    expect(written(B.path!)!.uploads).toEqual(["stack.tif"]);
    expect(manifestAt(A.path!)!.workflows[0].home).toBe(uid(B));     // both copies agree
    // B closes: the workflow is greyed out — its home can't change.
    st().closeDoc(B.id);
    useCollageStore.getState().docRemove(B.id);
    expect(st().setUploadHome("wfS", st().docUids[A.id])).toBe(false);
  });
});

// ── hydrate: restored uploads ───────────────────────────────

describe("hydrate: a reopened file's uploads", () => {
  async function writeA(tabs: WorkflowTab[], labels: Record<string, unknown> = {}) {
    await open([A]);
    for (const [k, v] of Object.entries(labels)) localStorage.setItem(k, JSON.stringify(v));
    st().setTabs(tabs);
    await st().persistDocWorkflows(A.id, { force: true });
    localStorage.clear();
    mockApi.writeWorkflows.mockClear();
  }

  it("a name clash is renamed in the workflow, its tile and its label; tiles land in the existing Direct upload; the file reads back clean", async () => {
    await writeA([wf("wfU", [src(A, "r0c0_i0"), upload("stack.tif")])], { "mpfig.source_names": { "img:stack.tif": "My stack" } });
    const r = disk.get(A.path!) as WorkflowsReadResponse;
    disk.set(A.path!, { ...r, renamed: { "stack.tif": "stack (2).tif" }, uploads: [{ name: "stack (2).tif", thumbnail: "T2", size: [5, 5] }] });
    localStorage.setItem("mpfig.source_groups", JSON.stringify([{ id: "grp_direct_here", name: "Direct upload", sourceKeys: ["img:stack.tif"] }]));
    const moved = vi.fn();
    window.addEventListener(ANALYSIS_KEYS_MOVED_EVENT, moved);
    try {
      await open([{ ...A, id: "doc_a_next" }]);
    } finally {
      window.removeEventListener(ANALYSIS_KEYS_MOVED_EVENT, moved);
    }
    const s = tab("wfU").nodes[0].data.sources!;
    expect(s[1]).toMatchObject({ name: "stack (2).tif", key: "img:stack (2).tif", label: "stack (2).tif", thumbnail: "T2" });
    expect(st().uploads.map((u) => u.key)).toEqual(["img:stack (2).tif"]);
    // The session's own stack.tif keeps its place; the restored one joins it.
    expect(groups()).toEqual([{ id: "grp_direct_here", name: "Direct upload", sourceKeys: ["img:stack.tif", "img:stack (2).tif"] }]);
    expect(JSON.parse(localStorage.getItem("mpfig.source_names")!)).toEqual({ "img:stack (2).tif": "My stack" });
    expect(moved).toHaveBeenCalled();                                  // a mounted view reloads
    await expect(st().persistDocWorkflows("doc_a_next")).resolves.toBe("clean");
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
  });

  it("creates Direct upload when there is none", async () => {
    // An uploads-only workflow with no home yet is saved with the active document.
    await writeA([wf("wfU", [upload("cells.tif")], { home: undefined })]);
    expect(manifestAt(A.path!)!.workflows.map((w) => w.id)).toEqual(["wfU"]);
    expect((disk.get(A.path!) as WorkflowsReadResponse).uploads.map((u) => u.name)).toEqual(["cells.tif"]);
    await open([A]);
    expect(groups()).toEqual([{ id: expect.stringMatching(/^grp_direct_/), name: "Direct upload", sourceKeys: ["img:cells.tif"] }]);
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("clean");
  });
});
