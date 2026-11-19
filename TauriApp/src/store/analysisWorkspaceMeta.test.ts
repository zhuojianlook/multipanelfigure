/* The Analysis panel's labels saved per .mpf (plan step 8): source renames,
   measurement group labels and source-library folders go into each
   document's `workflows/manifest.json` (`meta`, document-relative keys) and
   come back — merged into localStorage, which stays the mirror the panel
   reads — when the file is opened again, wherever it is. They count towards
   the document's dirty state. A file never touches another document's
   labels. The sidecar is a fake in-memory "disk". */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { Edge, Node } from "@xyflow/react";
import type { WorkflowsPayload, WorkflowsReadResponse } from "../api/types";
import type { InsetSource, NodeData, WorkflowTab } from "../utils/analysisTypes";
import { figKey, nsKey } from "../utils/analysisKeys";
import type { WorkflowsManifestV1 } from "../utils/workflowPersist";

const disk = vi.hoisted(() => new Map<string, unknown>());
const mockApi = vi.hoisted(() => ({ readWorkflows: vi.fn(), writeWorkflows: vi.fn(), pathsExist: vi.fn() }));
vi.mock("../api/client", () => ({ api: mockApi, default: vi.fn() }));

import { useCollageStore } from "./collageStore";
import { useSourceEditStore } from "./sourceEditStore";
import {
  useAnalysisWorkspaceStore, resetAnalysisWorkspaceForTests, newSourceNode, ANALYSIS_KEYS_MOVED_EVENT,
} from "./analysisWorkspaceStore";

// ── fixtures ────────────────────────────────────────────────

type Doc = { id: string; path: string | null; name: string };
const A: Doc = { id: "doc_a", path: "/exp/a.mpf", name: "a" };
const B: Doc = { id: "doc_b", path: "/exp/b.mpf", name: "b" };
const BASE_A = figKey(A);
const BASE_B = figKey(B);

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
function procNode(id: string): Node<NodeData> {
  return { id, type: "python", position: { x: 300, y: 30 }, data: { label: id, kind: "python", code: `# ${id}` } as NodeData };
}
function edge(source: string, target: string): Edge {
  return { id: `e_${source}_out_image_0__${target}_in_image`, source, sourceHandle: "out_image_0", target, targetHandle: "in_image", type: "deletable" };
}
function wf(id: string, nodes: Node<NodeData>[]): WorkflowTab {
  return { id, name: `Analysis ${id}`, updatedAt: 1_000, nodes: [...nodes, procNode("p1")], edges: [edge("source", "p1")] };
}
const one = (id: string, sources: InsetSource[]) => wf(id, [newSourceNode(sources)]);

const st = () => useAnalysisWorkspaceStore.getState();

type Labels = {
  names?: Record<string, string>;
  groups?: Record<string, string>;
  folders?: Array<{ id: string; name: string; sourceKeys: string[] }>;
};
function setLabels(l: Labels) {
  if (l.names) localStorage.setItem("mpfig.source_names", JSON.stringify(l.names));
  if (l.groups) localStorage.setItem("mpfig.measurement_groups", JSON.stringify(l.groups));
  if (l.folders) localStorage.setItem("mpfig.source_groups", JSON.stringify(l.folders));
}
const names = () => JSON.parse(localStorage.getItem("mpfig.source_names") || "{}");
const mgroups = () => JSON.parse(localStorage.getItem("mpfig.measurement_groups") || "{}");
const folders = () => JSON.parse(localStorage.getItem("mpfig.source_groups") || "[]");

function readBack(p: WorkflowsPayload): WorkflowsReadResponse {
  const blobs: WorkflowsReadResponse["blobs"] = {};
  for (const [id, b] of Object.entries(p.blobs)) blobs[id] = { kind: b.kind, data: b.data ?? "" };
  return {
    ...EMPTY, manifest: JSON.parse(JSON.stringify(p.manifest)), blobs, version: 1,
    uploads: p.uploads.map((name) => ({ name, thumbnail: "UPTHUMB", size: [5, 5] as [number, number] })),
  };
}
const metaAt = (path: string) => ((disk.get(path) as WorkflowsReadResponse | undefined)?.manifest as WorkflowsManifestV1 | undefined)?.meta;

async function flush(n = 80) { for (let i = 0; i < n; i++) await Promise.resolve(); }
async function open(docs: Doc[], active: string = docs[0]?.id ?? A.id) {
  useCollageStore.setState({ openDocs: [], activeDocId: null });
  resetAnalysisWorkspaceForTests();
  useCollageStore.setState({ openDocs: docs, activeDocId: active });
  await flush();
}
/** Session 1: `docs` open with `tabs` and `labels`; every file written. Then
 *  this machine's labels are forgotten (a fresh machine, unless reset). */
async function session1(docs: Doc[], tabs: WorkflowTab[], labels: Labels) {
  await open(docs);
  setLabels(labels);
  st().setTabs(tabs);
  for (const d of docs) await st().persistDocWorkflows(d.id, { force: true });
  localStorage.clear();
  mockApi.writeWorkflows.mockClear();
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

const LABELS: Labels = {
  names: {
    [`${BASE_A}::r0c0_i0`]: "Control_1",
    [`${BASE_B}::r0c0_i0`]: "B one",
    "img:stack.tif": "My stack",
  },
  groups: { [`${BASE_A}|R1C1|0|0`]: "Control", [`${BASE_A}#fig-2|R1C1|0|1`]: "Treated", [`${BASE_B}|R1C1|0|0`]: "B group" },
  folders: [
    { id: "grp_direct_1", name: "Direct upload", sourceKeys: ["img:stack.tif"] },
    { id: "g_cond", name: "Conditions", sourceKeys: [`${BASE_A}::r0c0_i0`, `${BASE_B}::r0c0_i0`] },
  ],
};

describe("persist: a document's labels go into its own file, document-relative", () => {
  it("names, measurement group labels and folders of its own keys (and of the uploads it embeds) — never another document's", async () => {
    await session1([A, B], [one("wfA", [src(A, "r0c0_i0"), upload("stack.tif")]), one("wfB", [src(B, "r0c0_i0")])], LABELS);
    expect(metaAt(A.path!)).toEqual({
      sourceNames: { "@::r0c0_i0": "Control_1", "img:stack.tif": "My stack" },
      measurementGroups: { "@|R1C1|0|0": "Control", "@#fig-2|R1C1|0|1": "Treated" },
      sourceGroups: [
        { id: "grp_direct_1", name: "Direct upload", sourceKeys: ["img:stack.tif"] },
        { id: "g_cond", name: "Conditions", sourceKeys: ["@::r0c0_i0"] },
      ],
    });
    expect(metaAt(B.path!)).toEqual({
      sourceNames: { "@::r0c0_i0": "B one" },
      measurementGroups: { "@|R1C1|0|0": "B group" },
      sourceGroups: [{ id: "g_cond", name: "Conditions", sourceKeys: ["@::r0c0_i0"] }],
    });
    expect(JSON.stringify(disk.get(A.path!))).not.toContain(B.path!);
  });

  it("a shared workflow's upload labels go with its home only", async () => {
    const shared = wf("wfS", [
      newSourceNode([src(A, "r0c0_i0"), upload("stack.tif")]),
      newSourceNode([src(B, "r0c1_i0")], { id: "srcB", label: "Source 2" }),
    ]);
    await session1([A, B], [shared], LABELS);
    expect(metaAt(A.path!)!.sourceNames).toHaveProperty(["img:stack.tif"], "My stack");
    expect(metaAt(B.path!)!.sourceNames).not.toHaveProperty(["img:stack.tif"]);
    expect((metaAt(B.path!)!.sourceGroups as Array<{ name: string }>).map((g) => g.name)).toEqual(["Conditions"]);
  });
});

describe("hydrate: the file's labels come back", () => {
  it("under the document's keys wherever the file is now; a mounted view is told; the file reads back clean", async () => {
    await session1([A], [one("wfA", [src(A, "r0c0_i0"), upload("stack.tif")])], LABELS);
    const moved: Doc = { id: "doc_a_moved", path: "/elsewhere/a.mpf", name: "a" };
    disk.set(moved.path!, disk.get(A.path!));
    const heard = vi.fn();
    window.addEventListener(ANALYSIS_KEYS_MOVED_EVENT, heard);
    try { await open([moved]); } finally { window.removeEventListener(ANALYSIS_KEYS_MOVED_EVENT, heard); }
    const base = figKey(moved);
    expect(names()).toEqual({ [`${base}::r0c0_i0`]: "Control_1", "img:stack.tif": "My stack" });
    expect(mgroups()).toEqual({ [`${base}|R1C1|0|0`]: "Control", [`${base}#fig-2|R1C1|0|1`]: "Treated" });
    expect(folders()).toEqual([
      { id: "grp_direct_1", name: "Direct upload", sourceKeys: ["img:stack.tif"] },
      { id: "g_cond", name: "Conditions", sourceKeys: [`${base}::r0c0_i0`] },
    ]);
    expect(heard).toHaveBeenCalled();
    await expect(st().persistDocWorkflows(moved.id)).resolves.toBe("clean");
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
  });

  it("the file's labels override this machine's; labels only this machine has stay and are saved next; others' are untouched", async () => {
    await session1([A], [one("wfA", [src(A, "r0c0_i0")])], LABELS);
    setLabels({
      names: { [`${BASE_A}::r0c0_i0`]: "stale", [`${BASE_A}::r0c1_i0`]: "only here", [`${BASE_B}::r0c0_i0`]: "B one" },
      folders: [{ id: "grp_direct_9", name: "Direct upload", sourceKeys: ["img:session.tif"] }, { id: "g_x", name: "Conditions", sourceKeys: [] }],
    });
    await open([A, B]);
    expect(names()).toEqual({
      [`${BASE_A}::r0c0_i0`]: "Control_1", [`${BASE_A}::r0c1_i0`]: "only here", [`${BASE_B}::r0c0_i0`]: "B one",
    });
    // "Conditions" merged by name (ids differ per machine), not duplicated.
    expect(folders()).toEqual([
      { id: "grp_direct_9", name: "Direct upload", sourceKeys: ["img:session.tif"] },
      { id: "g_x", name: "Conditions", sourceKeys: [`${BASE_A}::r0c0_i0`] },
    ]);
    await expect(st().persistDocWorkflows(B.id)).resolves.toBe("clean");  // B's file had none of this
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("written");
    expect(metaAt(A.path!)!.sourceNames).toEqual({ "@::r0c0_i0": "Control_1", "@::r0c1_i0": "only here" });
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("clean");
  });

  it("a file can't write another open document's labels, whatever its meta says", async () => {
    await session1([A], [one("wfA", [src(A, "r0c0_i0")])], {});
    const r = disk.get(A.path!) as WorkflowsReadResponse;
    const m = r.manifest as WorkflowsManifestV1;
    disk.set(A.path!, { ...r, manifest: { ...m, meta: {
      sourceNames: { [`${BASE_B}::r0c0_i0`]: "hijacked", "img:not-its-upload.tif": "no", "@::r0c0_i0": "mine" },
      measurementGroups: { [`${BASE_B}|R1C1|0|0`]: "hijacked" },
      sourceGroups: [{ id: "g", name: "G", sourceKeys: [`${BASE_B}::r0c0_i0`] }],
    } } });
    setLabels({ names: { [`${BASE_B}::r0c0_i0`]: "B one" }, groups: { [`${BASE_B}|R1C1|0|0`]: "B group" } });
    await open([B, A]);
    expect(names()).toEqual({ [`${BASE_B}::r0c0_i0`]: "B one", [`${BASE_A}::r0c0_i0`]: "mine" });
    expect(mgroups()).toEqual({ [`${BASE_B}|R1C1|0|0`]: "B group" });
    expect(folders()).toEqual([]);
  });
});

describe("dirty state: labels count", () => {
  it("renaming a source (or labelling a group) makes its document dirty; another document's label doesn't", async () => {
    await session1([A, B], [one("wfA", [src(A, "r0c0_i0")]), one("wfB", [src(B, "r0c0_i0")])], LABELS);
    await open([A, B]);
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("clean");
    // What the panel's rename / group setters write.
    setLabels({ names: { ...names(), [`${BASE_B}::r0c0_i0`]: "B renamed" } });
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("clean");
    setLabels({ groups: { ...mgroups(), [`${BASE_A}|R2C1|0|0`]: "Treated" } });
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("written");
    expect(metaAt(A.path!)!.measurementGroups).toEqual({ "@|R1C1|0|0": "Control", "@#fig-2|R1C1|0|1": "Treated", "@|R2C1|0|0": "Treated" });
    await expect(st().persistDocWorkflows(B.id)).resolves.toBe("written");
    expect(metaAt(B.path!)!.sourceNames).toEqual({ "@::r0c0_i0": "B renamed" });
    // Moving a source to another folder counts too.
    setLabels({ folders: [{ id: "g_new", name: "Treated", sourceKeys: [`${BASE_A}::r0c0_i0`] }] });
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("written");
    expect(metaAt(A.path!)!.sourceGroups).toEqual([{ id: "g_new", name: "Treated", sourceKeys: ["@::r0c0_i0"] }]);
  });

  it("an older file (no saved analysis) isn't rewritten just for this machine's labels — they go in with its next change", async () => {
    setLabels({ names: { [`${BASE_A}::r0c0_i0`]: "Control_1" } });
    await open([A]);
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("clean");
    expect(mockApi.writeWorkflows).not.toHaveBeenCalled();
    setLabels({ names: { [`${BASE_A}::r0c0_i0`]: "Control_1", [`${BASE_A}::r0c1_i0`]: "Drug_1" } });
    await expect(st().persistDocWorkflows(A.id)).resolves.toBe("written");
    expect(metaAt(A.path!)).toEqual({ sourceNames: { "@::r0c0_i0": "Control_1", "@::r0c1_i0": "Drug_1" } });
  });

  it("Save As: the copy's file carries the labels under the new document", async () => {
    await session1([A], [one("wfA", [src(A, "r0c0_i0")])], LABELS);
    await open([A]);
    const before = JSON.stringify(disk.get(A.path!));
    await st().afterDocSaved(A.id, A.path, "/exp/copy.mpf");
    expect(metaAt("/exp/copy.mpf")!.sourceNames).toEqual({ "@::r0c0_i0": "Control_1" });
    expect(JSON.stringify(disk.get(A.path!))).toBe(before);           // the original is left alone
    expect(names()[`${BASE_A}::r0c0_i0`]).toBe("Control_1");          // and keeps its labels here
  });
});
