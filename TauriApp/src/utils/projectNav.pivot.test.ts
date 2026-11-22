/* The planner's Pivot and the document / app flows around it (projectNav):
   - app quit with "Close without saving" undoes the unsaved Pivots in the
     Analysis keys of EVERY discarded document before any file is written (a
     workflow shared by two open documents goes into both files), and leaves
     them undone — the panel's labels in localStorage then match the files —
     unless the quit is called off (redoQuitPivots);
   - pivotFigure pivots the figure Pivot was pressed on, never one switched
     to while it waited.
   The sidecar is a fake in-memory disk. */

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
  confirm: vi.fn(async () => true),
  confirmThree: vi.fn(async () => "cancel"),
  alert: vi.fn(async () => true),
  saveProjectDialog: vi.fn(async () => null as string | null),
}));
vi.mock("../components/shared/ConfirmDialog", () => ({ confirm: dialogs.confirm, confirmThree: dialogs.confirmThree, alert: dialogs.alert }));
vi.mock("../components/shared/SaveProjectDialog", () => ({ saveProjectDialog: dialogs.saveProjectDialog }));

import { persistAllWorkflowsForQuit, pivotFigure, redoQuitPivots } from "./projectNav";
import { useCollageStore } from "../store/collageStore";
import { useFigureStore } from "../store/figureStore";
import { useSourceEditStore } from "../store/sourceEditStore";
import {
  useAnalysisWorkspaceStore, resetAnalysisWorkspaceForTests, newSourceNode, syncWorkspaceNow,
} from "../store/analysisWorkspaceStore";

type Doc = { id: string; path: string | null; name: string };
const A: Doc = { id: "doc_a", path: "/exp/a.mpf", name: "a" };
const B: Doc = { id: "doc_b", path: "/exp/b.mpf", name: "b" };

/** A source of `doc`'s figure at the cell its raw key names. */
function src(doc: Doc, raw: string): InsetSource {
  const fk = figKey(doc);
  const m = /^r(\d+)c(\d+)_/.exec(raw)!;
  return {
    key: nsKey(fk, raw), rawKey: raw, figKey: fk, mpfId: doc.id, mpf: doc.name,
    row: Number(m[1]), col: Number(m[2]), inset_index: 0, label: raw, natural_width: 10, natural_height: 10, thumbnail: "T",
  };
}
function wf(id: string, sources: InsetSource[]): WorkflowTab {
  const p1: Node<NodeData> = {
    id: "p1", type: "python", position: { x: 300, y: 30 },
    data: { label: "p1", kind: "python", code: "x = 1", ranWith: Object.fromEntries(sources.map((s) => [s.key, 0])) } as NodeData,
  };
  const e: Edge = { id: `e_${id}`, source: "source", sourceHandle: "out_image_0", target: "p1", targetHandle: "in_image" };
  return { id, name: `Analysis ${id}`, updatedAt: 1_000, nodes: [newSourceNode(sources), p1], edges: [e] };
}
function readBack(p: WorkflowsPayload): WorkflowsReadResponse {
  const blobs: WorkflowsReadResponse["blobs"] = {};
  for (const [id, b] of Object.entries(p.blobs)) blobs[id] = { kind: b.kind, data: b.data ?? "" };
  return { ...EMPTY, manifest: JSON.parse(JSON.stringify(p.manifest)), blobs, version: 1, uploads: [] } as WorkflowsReadResponse;
}
const manifestAt = (path: string) => (disk.get(path) as WorkflowsReadResponse | undefined)?.manifest as WorkflowsManifestV1 | undefined;
const ws = () => useAnalysisWorkspaceStore.getState();
async function flush(n = 80) { for (let i = 0; i < n; i++) await Promise.resolve(); }

/** Workflow `id` as stored in the file at `path`: its stamp and the raw keys
 *  of the sources of the document whose uid is `docUid`. */
function storedRaws(path: string, id: string, docUid: string) {
  const w = manifestAt(path)!.workflows.find((x) => x.id === id)!;
  const sources = (w.nodes.find((n) => n.id === "source")!.data.sources || []) as Array<{ doc?: string; rawKey?: string }>;
  return { updatedAt: w.updatedAt, raws: sources.filter((s) => s.doc === docUid).map((s) => s.rawKey) };
}
const liveRaws = (id: string, doc: Doc) =>
  ws().tabs.find((t) => t.id === id)!.nodes[0].data.sources!.filter((s) => s.mpfId === doc.id).map((s) => s.rawKey);

/** `docs` open (`active` live), every file read, `tabs` in the workspace. */
async function setup(docs: Doc[], active: string, tabs: WorkflowTab[]) {
  useCollageStore.setState({ openDocs: [], activeDocId: null, items: [] });
  resetAnalysisWorkspaceForTests();
  useCollageStore.setState({ openDocs: docs, activeDocId: active });
  await flush();
  ws().setTabs(tabs);
}

const LS_KEYS = ["mpfig.source_names", "mpfig.source_groups", "mpfig.measurement_groups"];
const readLS = (key: string) => JSON.parse(localStorage.getItem(key) || "null");

beforeEach(() => {
  vi.clearAllMocks();
  disk.clear();
  for (const k of LS_KEYS) localStorage.removeItem(k);
  mockApi.readWorkflows.mockImplementation(async (path: string) => (disk.get(path) as WorkflowsReadResponse) ?? EMPTY);
  mockApi.writeWorkflows.mockImplementation(async (path: string, payload: WorkflowsPayload) => {
    disk.set(path, readBack(payload));
    return { ok: true, size_bytes: 1, workflows_bytes: 1, missing_uploads: [], fast_path: true };
  });
  useSourceEditStore.setState({ revs: {}, withResults: [], rerun: {} });
  useFigureStore.setState({ unsaved: false, currentProjectPath: null, activeFigureId: "fig-1", config: { rows: 1, cols: 1, panels: [[{}]] } as never });
});
afterEach(() => {
  redoQuitPivots();                       // nothing left over for the next test
  useCollageStore.setState({ openDocs: [], activeDocId: null });
  resetAnalysisWorkspaceForTests();
  for (const k of LS_KEYS) localStorage.removeItem(k);
});

describe("app quit, 'Close without saving', after an unsaved Pivot", () => {
  for (const order of ["AB", "BA"] as const) {
    it(`a workflow shared with another open document is written un-pivoted into BOTH files (tabs ${order})`, async () => {
      // W reads A's panel at (0, 1) and one of B's. Both files hold it.
      await setup(order === "AB" ? [A, B] : [B, A], A.id, [wf("W", [src(A, "r0c1_i0"), src(B, "r0c0_i0")])]);
      syncWorkspaceNow();
      await expect(persistAllWorkflowsForQuit()).resolves.toEqual([]);
      const uidA = ws().docUids[A.id];
      // A's figure is pivoted (not saved), then the app quits without saving it.
      ws().pivotFigureKeys(A, "fig-1");
      syncWorkspaceNow();
      useFigureStore.setState({ unsaved: true });
      await expect(persistAllWorkflowsForQuit({ figuresDiscarded: true })).resolves.toEqual([]);
      const inA = storedRaws(A.path!, "W", uidA);
      const inB = storedRaws(B.path!, "W", uidA);
      expect(inA.raws).toEqual(["r0c1_i0"]);
      expect(inB.raws).toEqual(["r0c1_i0"]);
      expect(inB.updatedAt).toBe(inA.updatedAt);
      // The quit goes ahead: the keys stay as written.
      expect(liveRaws("W", A)).toEqual(["r0c1_i0"]);
      // Called off: the pivoted figure is still on screen, its keys follow it again.
      redoQuitPivots();
      expect(liveRaws("W", A)).toEqual(["r1c0_i0"]);
    });
  }

  it("the panel's labels in localStorage are left as the file was written, and pivoted again only when the quit is called off", async () => {
    await setup([A], A.id, [wf("W", [src(A, "r0c1_i0")])]);
    const fk = figKey(A);
    localStorage.setItem("mpfig.source_names", JSON.stringify({ [nsKey(fk, "r0c1_i0")]: "A" }));
    localStorage.setItem("mpfig.measurement_groups", JSON.stringify({ [`${fk}|R1C2|0|0`]: "Treated" }));
    ws().pivotFigureKeys(A, "fig-1");
    expect(readLS("mpfig.source_names")).toEqual({ [nsKey(fk, "r1c0_i0")]: "A" });
    useFigureStore.setState({ unsaved: true });
    await expect(persistAllWorkflowsForQuit({ figuresDiscarded: true })).resolves.toEqual([]);
    expect(readLS("mpfig.source_names")).toEqual({ [nsKey(fk, "r0c1_i0")]: "A" });
    expect(readLS("mpfig.measurement_groups")).toEqual({ [`${fk}|R1C2|0|0`]: "Treated" });
    redoQuitPivots();
    expect(readLS("mpfig.source_names")).toEqual({ [nsKey(fk, "r1c0_i0")]: "A" });
    expect(readLS("mpfig.measurement_groups")).toEqual({ [`${fk}|R2C1|0|0`]: "Treated" });
    // Only once: a second call has nothing left to redo.
    redoQuitPivots();
    expect(readLS("mpfig.source_names")).toEqual({ [nsKey(fk, "r1c0_i0")]: "A" });
  });
});

describe("pivotFigure", () => {
  /** _pivotGrid without a sidecar: just the Analysis hook. */
  const pivotGridSpy = () => vi.fn(async (onPivoted?: () => void) => { onPivoted?.(); return true; });

  /** Run `body` with the saved-Analysis read held until `release()` and
   *  _pivotGrid replaced by a spy. */
  async function withHeldRead(body: (release: () => void, pivotGrid: ReturnType<typeof pivotGridSpy>) => Promise<void>) {
    const realGrid = useFigureStore.getState()._pivotGrid;
    const realLoad = ws().ensureDocLoaded;
    let release!: () => void;
    const pivotGrid = pivotGridSpy();
    useAnalysisWorkspaceStore.setState({ ensureDocLoaded: vi.fn(() => new Promise<void>((r) => { release = r; })) });
    useFigureStore.setState({ _pivotGrid: pivotGrid });
    try {
      await body(() => release(), pivotGrid);
    } finally {
      useFigureStore.setState({ _pivotGrid: realGrid });
      useAnalysisWorkspaceStore.setState({ ensureDocLoaded: realLoad });
    }
  }

  it("a figure tab clicked while the saved Analysis is read: nothing is pivoted", async () => {
    await setup([A], A.id, [wf("W", [src(A, "r0c1_i0")])]);
    await withHeldRead(async (release, pivotGrid) => {
      const p = pivotFigure();
      useFigureStore.setState({ activeFigureId: "fig-2" });          // Figure 2 clicked meanwhile
      release();
      await expect(p).resolves.toBe(false);
      expect(pivotGrid).not.toHaveBeenCalled();
      expect(ws().pivotPending).toEqual({});
      expect(liveRaws("W", A)).toEqual(["r0c1_i0"]);
    });
  });

  it("the same figure still on screen: it is pivoted, with its own keys", async () => {
    await setup([A], A.id, [wf("W", [src(A, "r0c1_i0")])]);
    await withHeldRead(async (release, pivotGrid) => {
      const p = pivotFigure();
      release();
      await expect(p).resolves.toBe(true);
      expect(pivotGrid).toHaveBeenCalledTimes(1);
      expect(ws().pivotPending).toEqual({ [A.id]: { "fig-1": true } });
      expect(liveRaws("W", A)).toEqual(["r1c0_i0"]);
    });
  });
});
