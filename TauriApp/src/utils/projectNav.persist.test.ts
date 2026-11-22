/* Document close / app quit / "Save all" and the Analysis workflows saved in
   each .mpf (plan step 5): close auto-saves a document's workflows (no
   prompt), "Don't save" stores its image results as "needs re-run", a failed
   write asks Close anyway / Cancel, an Untitled document holding analysis
   gets the normal save prompt, quit writes every changed document. The
   sidecar is a fake in-memory disk. */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { Edge, Node } from "@xyflow/react";
import type { WorkflowsPayload, WorkflowsReadResponse } from "../api/types";
import type { InsetSource, NodeData, WorkflowTab } from "./analysisTypes";
import type { WorkflowsManifestV1 } from "./workflowPersist";
import { figKey, nsKey } from "./analysisKeys";

const disk = vi.hoisted(() => new Map<string, unknown>());
const EMPTY = vi.hoisted(() => ({
  manifest: null, blobs: {}, unsupported: false, version: null, uploads: [], renamed: {},
  missing_uploads: [], failed_uploads: [],
}));
const mockApi = vi.hoisted(() => ({
  readWorkflows: vi.fn(),
  writeWorkflows: vi.fn(),
  saveProject: vi.fn(async () => ({ ok: true })),
  loadProject: vi.fn(),
  updateConfig: vi.fn(async () => ({})),
  getPreview: vi.fn(async () => ({ image: "p", width: 1, height: 1, format: "png" })),
  getPanelPreview: vi.fn(async () => ({ image: "t" })),
  stashState: vi.fn(async () => ({ ok: true })),
  dropState: vi.fn(async () => ({ ok: true })),
  listImages: vi.fn(async () => ({ names: [] })),
  deleteImage: vi.fn(async () => ({})),
  getResolutions: vi.fn(async () => ({})),
}));
vi.mock("../api/client", () => ({
  api: mockApi,
  checkHealth: vi.fn(async () => true),
  lastHealthError: "",
  settleConfigWrites: vi.fn(async () => {}),
  default: vi.fn(),
}));
const dialogs = vi.hoisted(() => ({
  confirmThree: vi.fn(async () => "cancel"),
  alert: vi.fn(async () => true),
  saveProjectDialog: vi.fn(async () => null as string | null),
}));
vi.mock("../components/shared/ConfirmDialog", () => ({ confirmThree: dialogs.confirmThree, alert: dialogs.alert }));
vi.mock("../components/shared/SaveProjectDialog", () => ({ saveProjectDialog: dialogs.saveProjectDialog }));

import { closeDoc, persistAllWorkflowsForQuit, saveAllOpenDocs, unsavedAnalysisDocIds } from "./projectNav";
import { useCollageStore } from "../store/collageStore";
import { useFigureStore } from "../store/figureStore";
import { useSourceEditStore } from "../store/sourceEditStore";
import { useAnalysisWorkspaceStore, resetAnalysisWorkspaceForTests, newSourceNode } from "../store/analysisWorkspaceStore";

type Doc = { id: string; path: string | null; name: string };
const A: Doc = { id: "doc_a", path: "/exp/a.mpf", name: "a" };
const B: Doc = { id: "doc_b", path: "/exp/b.mpf", name: "b" };
const U: Doc = { id: "doc_u", path: null, name: "Untitled_1" };

function src(doc: Doc, raw: string): InsetSource {
  const fk = figKey(doc);
  return {
    key: nsKey(fk, raw), rawKey: raw, figKey: fk, mpfId: doc.id, mpf: doc.name,
    row: 0, col: 0, inset_index: 0, label: raw, natural_width: 10, natural_height: 10, thumbnail: "T",
  };
}
function upload(name: string): InsetSource {
  return { key: `img:${name}`, name, row: -1, col: -1, inset_index: -1, label: name, natural_width: 5, natural_height: 5, thumbnail: "U" };
}
function wf(id: string, sources: InsetSource[], extra: Partial<WorkflowTab> = {}): WorkflowTab {
  const p1: Node<NodeData> = {
    id: "p1", type: "python", position: { x: 300, y: 30 },
    data: {
      label: "p1", kind: "python", code: "x = 1", status: "ok",
      outputs: [{ id: "out_plot_0", kind: "plot", name: "plot_1", payload: `PNG_${id}`, blobId: `b_${id}` }],
      ranWith: Object.fromEntries(sources.map((s) => [s.key, 0])),
    } as NodeData,
  };
  const e: Edge = { id: `e_${id}`, source: "source", sourceHandle: "out_image_0", target: "p1", targetHandle: "in_image" };
  return { id, name: `Analysis ${id}`, updatedAt: 1_000, nodes: [newSourceNode(sources), p1], edges: [e], ...extra };
}
function readBack(p: WorkflowsPayload): WorkflowsReadResponse {
  const blobs: WorkflowsReadResponse["blobs"] = {};
  for (const [id, b] of Object.entries(p.blobs)) blobs[id] = { kind: b.kind, data: b.data ?? "" };
  return { ...EMPTY, manifest: JSON.parse(JSON.stringify(p.manifest)), blobs, version: 1, uploads: [] } as WorkflowsReadResponse;
}
const manifestAt = (path: string) => (disk.get(path) as WorkflowsReadResponse | undefined)?.manifest as WorkflowsManifestV1 | undefined;
const ws = () => useAnalysisWorkspaceStore.getState();
async function flush(n = 80) { for (let i = 0; i < n; i++) await Promise.resolve(); }

/** `docs` open (`active` live), every file read, `tabs` in the workspace. */
async function setup(docs: Doc[], active: string, tabs: (uids: Record<string, string>) => WorkflowTab[]) {
  useCollageStore.setState({ openDocs: [], activeDocId: null, items: [] });
  resetAnalysisWorkspaceForTests();
  useCollageStore.setState({ openDocs: docs, activeDocId: active });
  await flush();
  ws().setTabs(tabs(ws().docUids));
}

beforeEach(() => {
  vi.clearAllMocks();
  disk.clear();
  dialogs.confirmThree.mockResolvedValue("cancel");
  dialogs.saveProjectDialog.mockResolvedValue(null);
  mockApi.readWorkflows.mockImplementation(async (path: string) => (disk.get(path) as WorkflowsReadResponse) ?? EMPTY);
  mockApi.writeWorkflows.mockImplementation(async (path: string, payload: WorkflowsPayload) => {
    disk.set(path, readBack(payload));
    return { ok: true, size_bytes: 1, workflows_bytes: 1, missing_uploads: [], fast_path: true };
  });
  mockApi.loadProject.mockResolvedValue({
    config: { rows: 1, cols: 1, spacing: 0, panels: [[{ image_name: "" }]], column_labels: [], row_labels: [], column_headers: [], row_headers: [] },
    image_names: [], thumbnails: {},
  });
  useSourceEditStore.setState({ revs: {}, withResults: [], rerun: {} });
  useFigureStore.setState({ unsaved: false, currentProjectPath: null, config: { rows: 1, cols: 1, panels: [[{}]] } as never });
});
afterEach(() => {
  useCollageStore.setState({ openDocs: [], activeDocId: null });
  resetAnalysisWorkspaceForTests();
});

describe("closing a document", () => {
  it("a saved document's changed workflows are written into its file — no prompt — then leave the workspace", async () => {
    await setup([A, B], B.id, () => [wf("wfA", [src(A, "r0c0_i0")]), wf("wfB", [src(B, "r0c0_i0")])]);
    await expect(closeDoc(A.id)).resolves.toBe(true);
    expect(dialogs.confirmThree).not.toHaveBeenCalled();
    expect(mockApi.writeWorkflows).toHaveBeenCalledTimes(1);
    expect(mockApi.writeWorkflows).toHaveBeenCalledWith(A.path, expect.anything());
    expect(manifestAt(A.path!)!.workflows.map((w) => w.id)).toEqual(["wfA"]);
    expect(useCollageStore.getState().openDocs.map((d) => d.id)).toEqual([B.id]);
    expect(ws().tabs.map((t) => t.id)).toEqual(["wfB"]);
  });

  it("nothing changed since it was read: nothing is written", async () => {
    await setup([A, B], B.id, () => []);
    await expect(closeDoc(A.id)).resolves.toBe(true);
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
  });

  it("Don't save on the live document: its image results are stored 'needs re-run'", async () => {
    await setup([A, B], A.id, () => [wf("wfA", [src(A, "r0c0_i0")])]);
    useFigureStore.setState({ unsaved: true, currentProjectPath: A.path });
    dialogs.confirmThree.mockResolvedValueOnce("tertiary");            // Don't save
    await expect(closeDoc(A.id)).resolves.toBe(true);
    const p1 = manifestAt(A.path!)!.workflows[0].nodes.find((n) => n.id === "p1")!;
    expect(p1.data.stale).toBe(true);
    expect(p1.data.outputs).toHaveLength(1);                          // results kept, flagged
    expect(useCollageStore.getState().activeDocId).toBe(B.id);
  });

  it("a failed write asks Close anyway / Cancel: Cancel keeps the document and its workflows", async () => {
    await setup([A, B], B.id, () => [wf("wfA", [src(A, "r0c0_i0")])]);
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    mockApi.writeWorkflows.mockRejectedValueOnce(new Error("disk full"));
    dialogs.confirmThree.mockResolvedValueOnce("cancel");
    await expect(closeDoc(A.id)).resolves.toBe(false);
    expect(dialogs.confirmThree).toHaveBeenCalledWith(expect.objectContaining({
      title: "Analysis not saved", confirmLabel: "Close anyway", cancelLabel: "Cancel",
    }));
    expect(String((dialogs.confirmThree.mock.calls[0] as unknown as [{ body: string }])[0].body)).toContain("disk full");
    expect(useCollageStore.getState().openDocs.map((d) => d.id)).toEqual([A.id, B.id]);
    expect(ws().tabs.map((t) => t.id)).toEqual(["wfA"]);
    // Close anyway: gone.
    mockApi.writeWorkflows.mockRejectedValueOnce(new Error("disk full"));
    dialogs.confirmThree.mockResolvedValueOnce("confirm");
    await expect(closeDoc(A.id)).resolves.toBe(true);
    expect(useCollageStore.getState().openDocs.map((d) => d.id)).toEqual([B.id]);
    expect(ws().tabs.find((t) => t.id === "wfA")).toBeUndefined();
    err.mockRestore();
  });

  it("an Untitled document holding analysis gets Save / Don't save / Cancel, even with an empty figure", async () => {
    await setup([A, U], A.id, (u) => [wf("wfU", [upload("x.tif")], { home: u[U.id] })]);
    expect(unsavedAnalysisDocIds()).toEqual([U.id]);
    dialogs.confirmThree.mockResolvedValueOnce("cancel");
    await expect(closeDoc(U.id)).resolves.toBe(false);
    expect(dialogs.confirmThree).toHaveBeenCalledWith(expect.objectContaining({
      title: "Unsaved analysis", confirmLabel: "Save", tertiaryLabel: "Don't save", cancelLabel: "Cancel",
    }));
    expect(useCollageStore.getState().openDocs.map((d) => d.id)).toEqual([A.id, U.id]);
    dialogs.confirmThree.mockResolvedValueOnce("tertiary");
    await expect(closeDoc(U.id)).resolves.toBe(true);
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
    expect(ws().tabs.find((t) => t.id === "wfU")).toBeUndefined();
  });

  it("an Untitled document with only the empty placeholder closes without a prompt", async () => {
    await setup([A, U], A.id, (u) => [{ id: "p", name: "Workflow 1", home: u[U.id], nodes: [newSourceNode([])], edges: [] }]);
    expect(unsavedAnalysisDocIds()).toEqual([]);
    await expect(closeDoc(U.id)).resolves.toBe(true);
    expect(dialogs.confirmThree).not.toHaveBeenCalled();
  });
});

describe("app quit", () => {
  it("writes every changed saved document, skips Untitled ones, and reports the ones that failed", async () => {
    await setup([A, B, U], A.id, (u) => [
      wf("wfA", [src(A, "r0c0_i0")]), wf("wfB", [src(B, "r0c0_i0")]), wf("wfU", [upload("x.tif")], { home: u[U.id] }),
    ]);
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    mockApi.writeWorkflows.mockImplementation(async (path: string, payload: WorkflowsPayload) => {
      if (path === B.path) throw new Error("read-only volume");
      disk.set(path, readBack(payload));
      return { ok: true, size_bytes: 1, workflows_bytes: 1, missing_uploads: [], fast_path: true };
    });
    await expect(persistAllWorkflowsForQuit()).resolves.toEqual(["b"]);
    expect(mockApi.writeWorkflows.mock.calls.map((c) => c[0])).toEqual([A.path, B.path]);
    expect(manifestAt(A.path!)!.workflows.map((w) => w.id)).toEqual(["wfA"]);
    err.mockRestore();
  });

  it("'Close without saving': only documents whose figure edits are dropped get 'needs re-run' results", async () => {
    await setup([A, B], A.id, () => [wf("wfA", [src(A, "r0c0_i0")]), wf("wfB", [src(B, "r0c0_i0")])]);
    useFigureStore.setState({ unsaved: true });                        // A's figure edits are dropped
    await expect(persistAllWorkflowsForQuit({ figuresDiscarded: true })).resolves.toEqual([]);
    const stale = (path: string) => manifestAt(path)!.workflows[0].nodes.find((n) => n.id === "p1")!.data.stale;
    expect(stale(A.path!)).toBe(true);
    expect(stale(B.path!)).toBeUndefined();
  });

  it("nothing changed: nothing written", async () => {
    await setup([A, B], A.id, () => []);
    await expect(persistAllWorkflowsForQuit()).resolves.toEqual([]);
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
  });
});

describe("Save all", () => {
  it("includes an Untitled document holding analysis: saved to the chosen file, its workflows with it", async () => {
    await setup([A, U], U.id, (u) => [wf("wfU", [upload("x.tif")], { home: u[U.id] })]);
    dialogs.saveProjectDialog.mockResolvedValueOnce("/exp/untitled.mpf");
    await expect(saveAllOpenDocs()).resolves.toBe(true);
    expect(dialogs.saveProjectDialog).toHaveBeenCalledTimes(1);
    expect(mockApi.saveProject).toHaveBeenCalledWith("/exp/untitled.mpf", { carryFrom: null, docUid: ws().docUids[U.id] });
    expect(useCollageStore.getState().openDocs.find((d) => d.id === U.id)!.path).toBe("/exp/untitled.mpf");
    expect(manifestAt("/exp/untitled.mpf")!.workflows.map((w) => w.id)).toEqual(["wfU"]);
    expect(unsavedAnalysisDocIds()).toEqual([]);
  });

  it("cancelling its location aborts (nothing saved)", async () => {
    await setup([A, U], U.id, (u) => [wf("wfU", [upload("x.tif")], { home: u[U.id] })]);
    await expect(saveAllOpenDocs()).resolves.toBe(false);
    expect(mockApi.saveProject).not.toHaveBeenCalled();
  });
});
