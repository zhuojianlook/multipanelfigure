import { describe, it, expect } from "vitest";
import type { Edge, Node } from "@xyflow/react";
import type { InsetSource, MeasurementRef, NodeData, NodeOutput, WorkflowTab } from "./analysisTypes";
import type { WorkflowsReadResponse } from "../api/types";
import { figKey, nsKey } from "./analysisKeys";
import { computeStale } from "./analysisStale";
import { deriveOwnership, isLocked, missingOwners, rekeyTabs, closedKeyBase, docKeyBase, type DocRef } from "./workflowOwnership";
import {
  serializeForDoc, deserializeDoc, mergeIncoming, contentSig, docSig, applyRenames, remapRuntimeKeys,
  persistedOutputs, relativePath, joinPath, locateCandidates, uploadsOwnedBy,
  BLOB_ID_RE, CONSOLE_CAP, TOMBSTONE_CAP,
  type PersistState, type StoredWorkflow,
} from "./workflowPersist";

// ── fixtures ────────────────────────────────────────────────

const A: DocRef = { id: "doc_1_0", uid: "du_a", name: "a", path: "/exp/a.mpf" };
const B: DocRef = { id: "doc_1_1", uid: "du_b", name: "b", path: "/exp/sub/b.mpf" };
const C: DocRef = { id: "doc_1_2", uid: "du_c", name: "c", path: "/exp/c.mpf" };
const THUMB = "THUMB_SENTINEL";

function src(doc: { id: string; path?: string | null; name?: string }, raw: string, figureId?: string): InsetSource {
  const fk = figKey(doc, figureId);
  return {
    key: nsKey(fk, raw), rawKey: raw, figKey: fk, figureId, mpfId: doc.id, mpf: doc.name,
    row: 0, col: 0, inset_index: 0, label: raw, natural_width: 10, natural_height: 10,
    thumbnail: THUMB, parent_thumbnail: THUMB, figure_thumbnail: THUMB, parent_bbox: [1, 2, 3, 4], runKey: raw,
  };
}
function upload(name: string): InsetSource {
  return {
    key: `img:${name}`, name, row: -1, col: -1, inset_index: -1, label: name,
    natural_width: 5, natural_height: 5, thumbnail: THUMB,
  };
}
function meas(doc: DocRef): MeasurementRef {
  return { mpfId: doc.id, mpf: doc.name || "", label: `${doc.name} — measurements`, csv: "Panel,Name\nR1C1,L1", docUid: doc.uid };
}
function out(node: string, kind: NodeOutput["kind"], name: string, v: number, extra: Partial<NodeOutput> = {}): NodeOutput {
  return {
    id: `${node}_${name}__v${v}`, kind, name, versionLabel: `v${v}`,
    payload: kind === "table" ? `x,y\n${v},${v}\n` : `iVBOR${name}${v}`.repeat(50),
    ...extra,
  };
}
function sourceNode(id: string, sources: InsetSource[], refs: MeasurementRef[] = []): Node<NodeData> {
  return {
    id, type: "source", position: { x: 30, y: 30 }, draggable: true, deletable: id !== "source",
    data: {
      label: "Source", kind: "source", sources, status: "ok",
      ...(refs.length ? { measurementsRef: refs[0], measurementRefs: refs } : {}),
    } as NodeData,
  };
}
function procNode(id: string, data: Partial<NodeData> = {}): Node<NodeData> {
  return {
    id, type: (data.kind as string) || "python", position: { x: 300, y: 30 },
    data: { label: id, kind: "python", code: `# ${id}`, inputs: [], currentPreset: "custom", ...data } as NodeData,
  };
}
function edge(source: string, sourceHandle: string, target: string, targetHandle = "in_image"): Edge {
  return {
    id: `e_${source}_${sourceHandle}__${target}_${targetHandle}`, source, sourceHandle, target, targetHandle,
    type: "deletable", animated: false, style: { stroke: "#4a9d92", strokeWidth: 2 },
  };
}

const aKey = nsKey("p:/exp/a.mpf", "r0c0_i0");

/** Single-owner (A): an inset of A + an upload → Python → R. */
function singleWf(): WorkflowTab {
  return {
    id: "wf_single", name: "Blots", updatedAt: 1000, home: "du_a",
    nodes: [
      sourceNode("source", [src(A, "r0c0_i0"), upload("stack.tif")]),
      {
        ...procNode("p1", {
          outputs: [
            out("p1", "plot", "bar", 3, { blobId: "b_explicit", rCode: "plot(1)" }),
            out("p1", "table", "res", 3),
            out("p1", "plot", "bar", 2),
            out("p1", "plot", "bar", 1, { pinned: true, sentTo: ["My Figure"] }),
            out("p1", "table", "res", 2),
          ],
          status: "ok", consoleOut: "x".repeat(CONSOLE_CAP + 5000),
          ranWith: { [aKey]: 0, "img:stack.tif": 0 },
        }),
        selected: true, dragging: false, measured: { width: 180, height: 90 },
      } as Node<NodeData>,
      procNode("p2", { kind: "r", outputs: [out("p2", "plot", "box", 1, { rCode: "boxplot(1)" })], status: "ok", ranWith: { [aKey]: 0, "img:stack.tif": 0 } }),
    ],
    edges: [
      edge("source", "out_image_0", "p1"), edge("source", "out_image_1", "p1"),
      edge("p1", "out_table", "p2", "in_table"),
    ],
  };
}

/** Shared (A + B's second figure + B's measurements), uploads stored by B. */
function sharedWf(): WorkflowTab {
  return {
    id: "wf_shared", name: "Compare", updatedAt: 2000, home: "du_b",
    nodes: [
      sourceNode("source", [src(A, "r0c1_i0"), src(B, "r0c0_i0", "fig-2"), upload("shared.tif")], [meas(B)]),
      procNode("p1", {
        outputs: [out("p1", "plot", "cmp", 1)], status: "error", error: "boom", consoleOut: "hello",
        roi: { version: 1, images: { "doc_1_1-fig-2__r0c0_i0": { lanes: [], groups: [] }, r0c1_i0: { lanes: [], groups: [] } } } as unknown as NodeData["roi"],
      }),
    ],
    edges: [edge("source", "out_image_0", "p1"), edge("source", "out_table_measurements", "p1", "in_table")],
  };
}

const pristineWf = (): WorkflowTab => ({ id: "wf_p", name: "Workflow 3", home: "du_a", nodes: [sourceNode("source", [])], edges: [] });

/** Uploads only, saved in B. */
function uploadsWf(): WorkflowTab {
  return {
    id: "wf_up", name: "Uploads", updatedAt: 3000, home: "du_b",
    nodes: [sourceNode("source", [upload("u.tif")]), procNode("p1", { outputs: [out("p1", "table", "t", 1)], status: "ok", ranWith: { "img:u.tif": 0 } })],
    edges: [edge("source", "out_image_0", "p1")],
  };
}

function state(extra: Partial<PersistState> = {}): PersistState {
  return { docs: [A, B], tabs: [singleWf(), sharedWf(), pristineWf(), uploadsWf()], revs: {}, appVersion: "0.1.422", ...extra };
}

/** What api.readWorkflows returns for a file written from `p`. */
function readBack(p: ReturnType<typeof serializeForDoc>, extra: Partial<WorkflowsReadResponse> = {}): WorkflowsReadResponse {
  return {
    manifest: JSON.parse(JSON.stringify(p.manifest)),
    blobs: Object.fromEntries(Object.entries(p.blobs).map(([id, b]) => [id, { kind: b.kind, data: b.data as string }])),
    unsupported: false, version: 1, uploads: [], renamed: {}, missing_uploads: [], failed_uploads: [],
    ...extra,
  };
}

const wfById = (ws: StoredWorkflow[], id: string) => ws.find((w) => w.id === id)!;
const tabById = (ts: WorkflowTab[], id: string) => ts.find((t) => t.id === id)!;
const nodeById = (t: WorkflowTab, id: string) => t.nodes.find((n) => n.id === id)!;

// ── serializeForDoc ─────────────────────────────────────────

describe("serializeForDoc", () => {
  it("single-owner: keeps the latest output per name plus pinned versions, payloads as blobs", () => {
    const p = serializeForDoc("du_a", state());
    expect(p.manifest).toMatchObject({ format: "mpf-workflows", version: 1, savedWith: "0.1.422", docUid: "du_a" });
    const w = wfById(p.manifest.workflows, "wf_single");
    expect([w.shared, w.owners, w.home, w.updatedAt]).toEqual([false, ["du_a"], "du_a", 1000]);
    const p1 = w.nodes.find((n) => n.id === "p1")!;
    expect(p1.data.outputs!.map((o) => o.id)).toEqual(["p1_bar__v3", "p1_res__v3", "p1_bar__v1"]);
    expect(p1.data.status).toBe("ok");
    const [bar3, res3, bar1] = p1.data.outputs!;
    expect(bar3.blob).toBe("b_explicit");                         // minted id is kept
    expect(p.blobs.b_explicit).toEqual({ kind: "png", data: out("p1", "plot", "bar", 3).payload });
    expect(p.blobs[res3.blob]).toEqual({ kind: "csv", data: "x,y\n3,3\n" });
    expect(bar1).toMatchObject({ pinned: true, sentTo: ["My Figure"], versionLabel: "v1" });
    expect(p.blobs[bar3.rCodeBlob!]).toEqual({ kind: "txt", data: "plot(1)" });
    for (const id of Object.keys(p.blobs)) expect(id).toMatch(BLOB_ID_RE);
    // Payloads never sit in the manifest.
    expect(JSON.stringify(p.manifest)).not.toContain("iVBOR");
    // Derived blob ids are deterministic (stable file across re-saves).
    expect(serializeForDoc("du_a", state()).blobs).toEqual(p.blobs);
  });

  it("persistedOutputs: unpinned comparison archives are session-only", () => {
    const outs = singleWf().nodes[1].data.outputs!;
    expect(persistedOutputs(outs).map((o) => o.id)).toEqual(["p1_bar__v3", "p1_res__v3", "p1_bar__v1"]);
  });

  it("shared: graph and config only, in every owner's file", () => {
    const pa = serializeForDoc("du_a", state());
    const pb = serializeForDoc("du_b", state());
    expect(pa.manifest.workflows.map((w) => w.id)).toEqual(["wf_single", "wf_shared"]);
    expect(pb.manifest.workflows.map((w) => w.id)).toEqual(["wf_shared", "wf_up"]);
    for (const p of [pa, pb]) {
      const w = wfById(p.manifest.workflows, "wf_shared");
      expect([w.shared, w.owners, w.home]).toEqual([true, ["du_a", "du_b"], "du_b"]);
      const p1 = w.nodes.find((n) => n.id === "p1")!;
      expect(p1.data.outputs).toEqual([]);
      expect(p1.data.status).toBe("idle");
      expect(p1.data.consoleOut).toBeUndefined();
      expect(p1.data.error).toBeUndefined();
      expect(p1.data.roi).toBeDefined();                          // config is kept
      expect(w.nodes[0].data.measurementRefs).toEqual([{ docUid: "du_b", mpf: "b", label: "b — measurements", csv: "Panel,Name\nR1C1,L1" }]);
    }
    // B's only blobs are its uploads-only workflow's table.
    expect(Object.values(pb.blobs)).toEqual([{ kind: "csv", data: "x,y\n1,1\n" }]);
  });

  it("names every owner in `docs` — relocatable: path + relative path for the others only", () => {
    const pa = serializeForDoc("du_a", state());
    expect(pa.manifest.docs).toEqual({
      du_a: { name: "a", sessionId: "doc_1_0" },
      du_b: { name: "b", path: "/exp/sub/b.mpf", rel: "sub/b.mpf", sessionId: "doc_1_1" },
    });
    const pb = serializeForDoc("du_b", state());
    expect(pb.manifest.docs.du_a).toEqual({ name: "a", path: "/exp/a.mpf", rel: "../a.mpf", sessionId: "doc_1_0" });
    // A closed owner is named from what the workspace knows about it.
    const closed = rekeyTabs(state().tabs, docKeyBase(B), closedKeyBase("du_b"));
    const pc = serializeForDoc("du_a", state({ docs: [A], tabs: closed, known: { du_b: { name: "b", path: "/exp/sub/b.mpf", sessionId: "doc_0_9" } } }));
    expect(pc.manifest.docs.du_b).toEqual({ name: "b", path: "/exp/sub/b.mpf", rel: "sub/b.mpf", sessionId: "doc_0_9" });
    expect(wfById(pc.manifest.workflows, "wf_shared").owners).toEqual(["du_a", "du_b"]);
  });

  it("never saves a pristine tab", () => {
    expect(serializeForDoc("du_a", state()).manifest.workflows.map((w) => w.id)).not.toContain("wf_p");
    expect(serializeForDoc("du_a", state({ tabs: [pristineWf()] })).manifest.workflows).toEqual([]);
  });

  it("embeds uploads only for the workflows whose home it is", () => {
    expect(serializeForDoc("du_a", state()).uploads).toEqual(["stack.tif"]);
    expect(serializeForDoc("du_b", state()).uploads).toEqual(["shared.tif", "u.tif"]);
    expect(serializeForDoc("du_c", state({ docs: [A, B, C] })).uploads).toEqual([]);
    // uploadsOwnedBy is the same set (the store files their labels with it).
    expect(uploadsOwnedBy("du_a", state())).toEqual(["stack.tif"]);
    expect(uploadsOwnedBy("du_b", state())).toEqual(["shared.tif", "u.tif"]);
    // Moving a shared workflow's home moves its uploads.
    const moved = state({ tabs: [{ ...sharedWf(), home: "du_a" }] });
    expect(uploadsOwnedBy("du_a", moved)).toEqual(["shared.tif"]);
    expect(serializeForDoc("du_b", moved).uploads).toEqual([]);
  });

  it("writes the document's labels (meta) as given", () => {
    const meta = { sourceNames: { "@::r0c0_i0": "Control_1" } };
    expect(serializeForDoc("du_a", state({ meta: { du_a: meta, du_b: { sourceNames: { "@::x": "no" } } } })).manifest.meta).toEqual(meta);
    expect(serializeForDoc("du_a", state()).manifest.meta).toBeUndefined();
  });

  it("figureDiscarded flags every result that reads the document's images (transitively); uploads aren't affected", () => {
    const stale = (p: ReturnType<typeof serializeForDoc>, wfId: string) =>
      wfById(p.manifest.workflows, wfId).nodes.filter((n) => n.data.stale).map((n) => n.id);
    expect(stale(serializeForDoc("du_a", state()), "wf_single")).toEqual([]);
    expect(stale(serializeForDoc("du_a", state(), { figureDiscarded: true }), "wf_single")).toEqual(["p1", "p2"]);
    expect(stale(serializeForDoc("du_b", state(), { figureDiscarded: true }), "wf_up")).toEqual([]);
    // An Edit Panel change since the run (revs) is stale without any discard.
    expect(stale(serializeForDoc("du_a", state({ revs: { [aKey]: 1 } })), "wf_single")).toEqual(["p1", "p2"]);
  });

  it("strips thumbnails, runKey, session ids and ReactFlow transients; caps the console", () => {
    const p = serializeForDoc("du_a", state());
    const json = JSON.stringify(p.manifest);
    for (const banned of [THUMB, "runKey", "mpfId", "\"selected\"", "\"measured\"", "\"dragging\"", "/exp/a.mpf"]) {
      expect(json).not.toContain(banned);
    }
    const w = wfById(p.manifest.workflows, "wf_single");
    expect(w.nodes[0].data.sources).toEqual([
      { rawKey: "r0c0_i0", doc: "du_a", mpf: "a", row: 0, col: 0, inset_index: 0, label: "r0c0_i0",
        natural_width: 10, natural_height: 10, parent_bbox: [1, 2, 3, 4], figureId: undefined },
      { name: "stack.tif", row: -1, col: -1, inset_index: -1, label: "stack.tif", natural_width: 5, natural_height: 5 },
    ]);
    expect(w.nodes[0]).toMatchObject({ id: "source", type: "source", draggable: true, deletable: false });
    const co = w.nodes.find((n) => n.id === "p1")!.data.consoleOut as string;
    expect(co.length).toBe(CONSOLE_CAP);
    expect(co.endsWith("(truncated)")).toBe(true);
    expect(w.edges[0]).toEqual(edge("source", "out_image_0", "p1"));
  });

  it("keeps the newest tombstones, capped", () => {
    const ts: Record<string, number> = {};
    for (let i = 0; i < TOMBSTONE_CAP + 50; i++) ts[`wf_${i}`] = i;
    const m = serializeForDoc("du_a", state({ tombstones: { du_a: ts, du_b: { wf_other: 1 } } })).manifest;
    expect(Object.keys(m.tombstones!).length).toBe(TOMBSTONE_CAP);
    expect(m.tombstones!.wf_249).toBe(249);
    expect(m.tombstones!.wf_0).toBeUndefined();
    expect(m.tombstones!.wf_other).toBeUndefined();
  });
});

// ── deserializeDoc ──────────────────────────────────────────

describe("deserializeDoc", () => {
  it("rebuilds the file's own sources under wherever the file is now", () => {
    const moved: DocRef = { id: "doc_9_0", uid: "du_a", name: "a copy", path: "/moved/a copy.mpf" };
    const d = deserializeDoc(readBack(serializeForDoc("du_a", state())), { doc: moved, uid: "du_a", docs: [] });
    expect(d.fileUid).toBe("du_a");
    const t = tabById(d.tabs, "wf_single");
    const s = t.nodes[0].data.sources!;
    const k = nsKey("p:/moved/a copy.mpf", "r0c0_i0");
    expect(s[0]).toMatchObject({ key: k, figKey: "p:/moved/a copy.mpf", rawKey: "r0c0_i0", mpfId: "doc_9_0", thumbnail: "" });
    expect(s[1]).toMatchObject({ key: "img:stack.tif", name: "stack.tif" });
    const p1 = nodeById(t, "p1");
    expect(p1.data.outputs!.map((o) => [o.id, o.payload])).toEqual([
      ["p1_bar__v3", out("p1", "plot", "bar", 3).payload],
      ["p1_res__v3", "x,y\n3,3\n"],
      ["p1_bar__v1", out("p1", "plot", "bar", 1).payload],
    ]);
    expect(p1.data.outputs![0]).toMatchObject({ blobId: "b_explicit", rCode: "plot(1)" });
    expect(p1.data.status).toBe("ok");
    expect(p1.data.ranWith).toEqual({ [k]: 0, "img:stack.tif": 0 });
    expect(t.edges).toEqual(singleWf().edges);
    expect([t.updatedAt, t.home]).toEqual([1000, "du_a"]);
    expect(isLocked(t, { docs: [moved] })).toBe(false);
    expect(computeStale(t.nodes, t.edges, {}).nodeIds.size).toBe(0);
  });

  it("documents that aren't open come back as u: keys — locked, and known with places to look", () => {
    const d = deserializeDoc(readBack(serializeForDoc("du_a", state())), { doc: A, uid: "du_a", docs: [A] });
    const t = tabById(d.tabs, "wf_shared");
    expect(t.nodes[0].data.sources!.map((s) => s.key)).toEqual([
      nsKey("p:/exp/a.mpf", "r0c1_i0"), nsKey("u:du_b#fig-2", "r0c0_i0"), "img:shared.tif",
    ]);
    expect(t.nodes[0].data.sources![1].mpfId).toBeUndefined();
    expect(t.nodes[0].data.measurementRefs![0]).toMatchObject({ docUid: "du_b", mpfId: "" });
    expect(isLocked(t, { docs: [A] })).toBe(true);
    expect(missingOwners(t, { docs: [A] })).toEqual(["du_b"]);
    expect(d.known).toEqual({
      du_b: { name: "b", path: "/exp/sub/b.mpf", sessionId: "doc_1_1", candidates: ["/exp/sub/b.mpf", "/exp/b.mpf"] },
    });
    // The pair moved together: the relative path finds it.
    const moved = deserializeDoc(readBack(serializeForDoc("du_a", state())), {
      doc: { id: "doc_5", path: "/new/a.mpf" }, uid: "du_a", docs: [],
    });
    expect(moved.known.du_b.candidates).toEqual(["/exp/sub/b.mpf", "/new/sub/b.mpf", "/new/b.mpf"]);
  });

  it("re-binds other open documents to their new session ids — picker runtime keys too", () => {
    const A2: DocRef = { ...A, id: "doc_7_0" };
    const B2: DocRef = { ...B, id: "doc_7_1" };
    const d = deserializeDoc(readBack(serializeForDoc("du_a", state())), { doc: A2, uid: "du_a", docs: [A2, B2] });
    expect(d.idMap).toEqual({ doc_1_0: "doc_7_0", doc_1_1: "doc_7_1" });
    const t = tabById(d.tabs, "wf_shared");
    const s = t.nodes[0].data.sources!;
    expect(s[1]).toMatchObject({ key: nsKey("p:/exp/sub/b.mpf#fig-2", "r0c0_i0"), mpfId: "doc_7_1", figureId: "fig-2" });
    expect(t.nodes[0].data.measurementRefs![0].mpfId).toBe("doc_7_1");
    expect(Object.keys(nodeById(t, "p1").data.roi!.images!)).toEqual(["doc_7_1-fig-2__r0c0_i0", "r0c1_i0"]);
    expect(isLocked(t, { docs: [A2, B2] })).toBe(false);
  });

  it("stale results restore one revision behind, so computeStale flags them until re-run", () => {
    const p = serializeForDoc("du_a", state(), { figureDiscarded: true });
    const d = deserializeDoc(readBack(p), { doc: A, uid: "du_a", docs: [A, B] });
    const t = tabById(d.tabs, "wf_single");
    expect(nodeById(t, "p1").data.ranWith).toEqual({ [aKey]: -1, "img:stack.tif": -1 });
    expect([...computeStale(t.nodes, t.edges, {}).nodeIds].sort()).toEqual(["p1", "p2"]);
    // Against the session's current revisions.
    const d3 = deserializeDoc(readBack(p), { doc: A, uid: "du_a", docs: [A, B], revs: { [aKey]: 3 } });
    const t3 = tabById(d3.tabs, "wf_single");
    expect(nodeById(t3, "p1").data.ranWith![aKey]).toBe(2);
    expect(computeStale(t3.nodes, t3.edges, { [aKey]: 3 }).nodeIds.has("p1")).toBe(true);
    // ...and it stays stale when written again.
    const again = serializeForDoc("du_a", { docs: [A, B], tabs: d.tabs });
    expect(wfById(again.manifest.workflows, "wf_single").nodes.filter((n) => n.data.stale).map((n) => n.id)).toEqual(["p1", "p2"]);
    // Not stale → current revision.
    const fresh = deserializeDoc(readBack(serializeForDoc("du_a", state())), { doc: A, uid: "du_a", docs: [A, B], revs: { [aKey]: 3 } });
    const tf = tabById(fresh.tabs, "wf_single");
    expect(computeStale(tf.nodes, tf.edges, { [aKey]: 3 }).nodeIds.size).toBe(0);
  });

  it("applies a re-minted uid to the file's own sources, tables and home", () => {
    const A2: DocRef = { ...A, uid: "du_a2" };
    const d = deserializeDoc(readBack(serializeForDoc("du_a", state())), { doc: A2, uid: "du_a2", docs: [A2, B] });
    expect(tabById(d.tabs, "wf_single").home).toBe("du_a2");
    expect(deriveOwnership(tabById(d.tabs, "wf_single"), { docs: [A2, B] }).owners).toEqual(["du_a2"]);
    expect(deriveOwnership(tabById(d.tabs, "wf_shared"), { docs: [A2, B] }).owners).toEqual(["du_a2", "du_b"]);
    expect(d.known.du_a).toBeUndefined();
  });

  it("rewrites renamed uploads everywhere and gives them their restored thumbnails", () => {
    const p = serializeForDoc("du_a", state());
    const d = deserializeDoc(
      readBack(p, { renamed: { "stack.tif": "stack (2).tif" }, uploads: [{ name: "stack (2).tif", thumbnail: "T2", size: [3, 4] }] }),
      { doc: A, uid: "du_a", docs: [A, B] },
    );
    const t = tabById(d.tabs, "wf_single");
    expect(t.nodes[0].data.sources![1]).toMatchObject({ name: "stack (2).tif", key: "img:stack (2).tif", label: "stack (2).tif", thumbnail: "T2" });
    expect(Object.keys(nodeById(t, "p1").data.ranWith!)).toContain("img:stack (2).tif");
    expect(d.uploads).toEqual([{
      key: "img:stack (2).tif", name: "stack (2).tif", row: -1, col: -1, inset_index: -1,
      label: "stack (2).tif", natural_width: 3, natural_height: 4, thumbnail: "T2",
    }]);
  });

  it("renames upload keys in the file's labels (meta) the same way", () => {
    const p = serializeForDoc("du_a", state({ meta: { du_a: {
      sourceNames: { "img:stack.tif": "My stack", "@::r0c0_i0": "Control_1" },
      sourceGroups: [{ id: "g", name: "Direct upload", sourceKeys: ["img:stack.tif"] }],
    } } }));
    const d = deserializeDoc(readBack(p, { renamed: { "stack.tif": "stack (2).tif" } }), { doc: A, uid: "du_a", docs: [A, B] });
    expect(d.meta).toEqual({
      sourceNames: { "img:stack (2).tif": "My stack", "@::r0c0_i0": "Control_1" },
      sourceGroups: [{ id: "g", name: "Direct upload", sourceKeys: ["img:stack (2).tif"] }],
    });
  });

  it("shows nothing for a missing, newer, unsupported or foreign manifest", () => {
    const p = serializeForDoc("du_a", state());
    const ctx = { doc: A, uid: "du_a", docs: [A] };
    expect(deserializeDoc(readBack(p, { manifest: null }), ctx).tabs).toEqual([]);
    expect(deserializeDoc(readBack(p, { unsupported: true }), ctx).tabs).toEqual([]);
    const m2 = { ...JSON.parse(JSON.stringify(p.manifest)), version: 2 };
    expect(deserializeDoc(readBack(p, { manifest: m2 }), ctx).tabs).toEqual([]);
    const foreign = { ...JSON.parse(JSON.stringify(p.manifest)), format: "something-else" };
    expect(deserializeDoc(readBack(p, { manifest: foreign }), ctx).tabs).toEqual([]);
  });

  it("drops an output whose blob is missing, and skips malformed workflows", () => {
    const p = serializeForDoc("du_a", state());
    const rb = readBack(p);
    delete rb.blobs.b_explicit;
    (rb.manifest!.workflows as unknown[]).push({ id: 7 }, null);
    (rb.manifest!.workflows as StoredWorkflow[])[1].nodes.push({ id: "bad", position: { x: 0, y: 0 }, data: { label: "no kind" } } as never);
    const d = deserializeDoc(rb, { doc: A, uid: "du_a", docs: [A, B] });
    expect(d.tabs.map((t) => t.id)).toEqual(["wf_single", "wf_shared"]);
    expect(tabById(d.tabs, "wf_shared").nodes.map((n) => n.id)).toEqual(["source", "p1"]);
    expect(nodeById(tabById(d.tabs, "wf_single"), "p1").data.outputs!.map((o) => o.id)).toEqual(["p1_res__v3", "p1_bar__v1"]);
  });
});

// ── round trip / signatures ─────────────────────────────────

describe("round trip and signatures", () => {
  it("a file read back has the signature it was written with — no false dirty — and re-serializes identically", () => {
    const st = state();
    for (const doc of [A, B]) {
      const p = serializeForDoc(doc.uid, st);
      const d = deserializeDoc(readBack(p), { doc, uid: doc.uid, docs: [A, B] });
      const asRead: PersistState = { docs: [A, B], tabs: d.tabs, revs: {} };
      expect(docSig(doc.uid, asRead)).toBe(docSig(doc.uid, st));
      const again = serializeForDoc(doc.uid, { ...asRead, appVersion: "0.1.422" });
      expect(JSON.parse(JSON.stringify(again.manifest.workflows))).toEqual(JSON.parse(JSON.stringify(p.manifest.workflows)));
      expect(again.blobs).toEqual(p.blobs);
    }
  });

  it("contentSig ignores thumbnails, selection, runKey/mpfId, unpinned archives and updatedAt", () => {
    const base = singleWf();
    const sig = contentSig(base, { docs: [A, B] });
    const variants: WorkflowTab[] = [
      { ...base, updatedAt: 99999 },
      { ...base, nodes: base.nodes.map((n) => ({ ...n, selected: !n.selected, measured: { width: 1, height: 1 } })) },
      { ...base, nodes: [sourceNode("source", [{ ...src(A, "r0c0_i0"), thumbnail: "other", runKey: "doc_x__r0c0_i0", mpfId: "doc_x" }, { ...upload("stack.tif"), thumbnail: "z" }]), ...base.nodes.slice(1)] },
      { ...base, nodes: base.nodes.map((n) => n.id !== "p1" ? n : { ...n, data: { ...n.data, outputs: n.data.outputs!.filter((o) => o.id !== "p1_bar__v2") } }) },
      { ...base, nodes: base.nodes.map((n) => n.id !== "p1" ? n : { ...n, data: { ...n.data, status: "running" as const } }) },
    ];
    for (const v of variants) expect(contentSig(v, { docs: [A, B] })).toBe(sig);
  });

  it("contentSig changes on code, position, output, pin, an in-place re-render and the name", () => {
    const base = singleWf();
    const sig = contentSig(base, { docs: [A, B] });
    const p1 = (fn: (n: Node<NodeData>) => Node<NodeData>): WorkflowTab => ({ ...base, nodes: base.nodes.map((n) => (n.id === "p1" ? fn(n) : n)) });
    const outs = (fn: (o: NodeOutput[]) => NodeOutput[]) => p1((n) => ({ ...n, data: { ...n.data, outputs: fn(n.data.outputs!) } }));
    const variants: WorkflowTab[] = [
      p1((n) => ({ ...n, data: { ...n.data, code: "# p1 changed" } })),
      p1((n) => ({ ...n, position: { x: 301, y: 30 } })),
      outs((o) => [out("p1", "plot", "bar", 4), ...o]),
      outs((o) => o.map((x) => (x.id === "p1_res__v3" ? { ...x, pinned: true } : x))),
      outs((o) => o.map((x) => (x.id === "p1_bar__v3" ? { ...x, payload: x.payload + "AAAA" } : x))),
      { ...base, name: "Blots 2" },
    ];
    const sigs = variants.map((v) => contentSig(v, { docs: [A, B] }));
    for (const s of sigs) expect(s).not.toBe(sig);
    expect(new Set(sigs).size).toBe(sigs.length);
  });

  it("contentSig survives a document closing (p: → u: re-key) when given the open documents", () => {
    const shared = sharedWf();
    const closed = rekeyTabs([shared], docKeyBase(B), closedKeyBase("du_b"))[0];
    expect(closed).not.toBe(shared);
    expect(contentSig(closed, { docs: [A], known: { du_b: { name: "b", path: B.path! } } }))
      .toBe(contentSig(shared, { docs: [A, B] }));
  });

  it("docSig follows ownership: attaching another document's data adds the workflow to it, detaching removes it", () => {
    const st = state();
    const sigB = docSig("du_b", st);
    const sigA = docSig("du_a", st);
    const attached = st.tabs.map((t) => (t.id !== "wf_single" ? t : {
      ...t, nodes: [sourceNode("source", [...t.nodes[0].data.sources!, src(B, "r2c2_i0")]), ...t.nodes.slice(1)],
    }));
    expect(docSig("du_b", { ...st, tabs: attached })).not.toBe(sigB);
    expect(docSig("du_a", { ...st, tabs: attached })).not.toBe(sigA);   // its copy lost its results
    const detached = attached.map((t) => (t.id !== "wf_single" ? t : {
      ...t, nodes: [sourceNode("source", t.nodes[0].data.sources!.slice(0, 2)), ...t.nodes.slice(1)],
    }));
    expect(docSig("du_b", { ...st, tabs: detached })).toBe(sigB);
    expect(docSig("du_a", { ...st, tabs: detached })).toBe(sigA);
  });

  it("docSig ignores other documents' workflows and session ids, but not updatedAt", () => {
    const st = state();
    const sigA = docSig("du_a", st);
    const otherEdited = st.tabs.map((t) => (t.id === "wf_up" ? { ...t, name: "Renamed", updatedAt: 9 } : t));
    expect(docSig("du_a", { ...st, tabs: otherEdited })).toBe(sigA);
    const newSession = { ...st, docs: [{ ...A, id: "doc_8_0" }, { ...B, id: "doc_8_1" }] };
    expect(docSig("du_a", newSession)).toBe(sigA);
    const bumped = st.tabs.map((t) => (t.id === "wf_single" ? { ...t, updatedAt: 1001 } : t));
    expect(docSig("du_a", { ...st, tabs: bumped })).not.toBe(sigA);
  });
});

// ── mergeIncoming ───────────────────────────────────────────

describe("mergeIncoming", () => {
  const ctx = { docs: [A, B, C] };
  const singleOf = (doc: DocRef, id: string, updatedAt: number, name = id): WorkflowTab => ({
    id, name, updatedAt, nodes: [sourceNode("source", [src(doc, "r0c0_i0")])], edges: [],
  });
  const sharedOf = (id: string, updatedAt: number, name = id): WorkflowTab => ({
    id, name, updatedAt, nodes: [sourceNode("source", [src(A, "r0c0_i0"), src(B, "r0c0_i0")])], edges: [],
  });

  it("adds workflows it doesn't have", () => {
    const x = singleOf(A, "x", 1);
    const r = mergeIncoming([x], [singleOf(B, "y", 1)], {}, ctx);
    expect(r.tabs.map((t) => t.id)).toEqual(["x", "y"]);
    expect(r.tabs[0]).toBe(x);
  });

  it("the newer copy wins, either way round; a tie keeps the one already loaded", () => {
    const older = sharedOf("s", 5, "old");
    const newer = sharedOf("s", 10, "new");
    let r = mergeIncoming([older], [newer], {}, ctx);
    expect(r.tabs.map((t) => t.name)).toEqual(["new"]);
    expect(r.replaced).toEqual(["s"]);
    r = mergeIncoming([newer], [older], {}, ctx);
    expect(r.tabs.map((t) => t.name)).toEqual(["new"]);
    expect(r.dropped).toEqual([{ id: "s", reason: "superseded" }]);
    const existing = [sharedOf("s", 7, "loaded")];
    r = mergeIncoming(existing, [sharedOf("s", 7, "incoming")], {}, ctx);
    expect(r.tabs).toBe(existing);
    // Same single owner (the same document read twice): newest wins too.
    r = mergeIncoming([singleOf(A, "x", 1, "old")], [singleOf(A, "x", 2, "new")], {}, ctx);
    expect(r.tabs.map((t) => t.name)).toEqual(["new"]);
  });

  it("re-mints a single-owner twin from a different document (Finder copy / Save-As original)", () => {
    const mine = singleOf(A, "x", 50, "mine");
    const twin = singleOf(C, "x", 10, "twin");
    const r = mergeIncoming([mine], [twin], {}, ctx, { mintId: () => "wf_new" });
    expect(r.tabs.map((t) => [t.id, t.name])).toEqual([["x", "mine"], ["wf_new", "twin"]]);
    expect(r.reminted).toEqual({ x: "wf_new" });
    expect(r.tabs[0]).toBe(mine);
    // A shared copy of the same id is the same workflow → no re-mint.
    const r2 = mergeIncoming([mine], [sharedOf("x", 60, "shared")], {}, ctx, { mintId: () => "wf_new" });
    expect(r2.tabs.map((t) => [t.id, t.name])).toEqual([["x", "shared"]]);
  });

  it("tombstones drop copies that aren't newer than the deletion — incoming or already loaded", () => {
    let r = mergeIncoming([], [singleOf(A, "dead", 5), singleOf(A, "alive", 20)], { du_a: { dead: 10, alive: 10 } }, ctx);
    expect(r.tabs.map((t) => t.id)).toEqual(["alive"]);
    expect(r.dropped).toEqual([{ id: "dead", reason: "tombstone" }]);
    r = mergeIncoming([singleOf(B, "dead", 10)], [], { du_b: { dead: 10 } }, ctx);
    expect(r.tabs).toEqual([]);
    expect(r.dropped).toEqual([{ id: "dead", reason: "tombstone" }]);
    // A shared workflow goes when any of its owners deleted it.
    r = mergeIncoming([], [sharedOf("s", 5)], { du_b: { s: 6 } }, ctx);
    expect(r.tabs).toEqual([]);
  });

  it("a tombstone only removes copies its own document owns (Save-As / Finder copy)", () => {
    // Deleted in C (the copy); A's same-id workflow is a different one.
    let r = mergeIncoming([], [singleOf(A, "x", 5)], { du_c: { x: 10 } }, ctx);
    expect(r.tabs.map((t) => t.id)).toEqual(["x"]);
    r = mergeIncoming([singleOf(A, "x", 5)], [], { du_c: { x: 10 } }, ctx);
    expect(r.tabs.map((t) => t.id)).toEqual(["x"]);
  });
});

// ── renames / runtime keys / paths ──────────────────────────

describe("applyRenames", () => {
  const tab = (): WorkflowTab => ({
    id: "w", name: "w", edges: [],
    nodes: [
      sourceNode("source", [upload("a.tif"), { ...upload("b.tif"), label: "Control_1" }, upload("keep.tif"), src(A, "r0c0_i0")]),
      procNode("cp", {
        kind: "cellpose", ranWith: { "img:a.tif": 0, [aKey]: 0 },
        cellposeSeg: { groups: [{ id: "g", name: "Ctl", images: ["img:a.tif", "img:keep.tif"] }], editedMasks: { "a.tif": "data:x" } } as unknown as NodeData["cellposeSeg"],
      }),
    ],
  });
  it("renames upload sources, ranWith, picker ids and masks — in one pass", () => {
    const [t] = applyRenames([tab()], { "a.tif": "b.tif", "b.tif": "b (2).tif" });
    expect(t.nodes[0].data.sources!.map((s) => [s.name, s.key, s.label])).toEqual([
      ["b.tif", "img:b.tif", "b.tif"],
      ["b (2).tif", "img:b (2).tif", "Control_1"],
      ["keep.tif", "img:keep.tif", "keep.tif"],
      [undefined, aKey, "r0c0_i0"],
    ]);
    const cp = t.nodes[1].data;
    expect(Object.keys(cp.ranWith!)).toEqual(["img:b.tif", aKey]);
    expect(cp.cellposeSeg!.groups![0].images).toEqual(["img:b.tif", "img:keep.tif"]);
    expect(Object.keys(cp.cellposeSeg!.editedMasks!)).toEqual(["b.tif"]);
  });
  it("returns the tabs untouched when nothing applies", () => {
    const tabs = [tab()];
    expect(applyRenames(tabs, {})).toBe(tabs);
    expect(applyRenames(tabs, { "zzz.tif": "zzz (2).tif" })).toBe(tabs);
  });
});

describe("remapRuntimeKeys", () => {
  it("re-points keys minted under an old session doc id, and nothing else", () => {
    const keys = ["doc_1_1__r0c0_i0", "doc_1_1-fig-2__r0c0_i0", "doc_1_10__r0c0_i0", "r0c0_i0", "fig-2__r0c0_i0", "img:doc_1_1__x.tif", aKey];
    const t: WorkflowTab = {
      id: "w", name: "w", edges: [],
      nodes: [procNode("int", { intensity: { groups: [{ id: "g", name: "G", images: keys }] } as unknown as NodeData["intensity"] })],
    };
    const [r] = remapRuntimeKeys([t], { doc_1_1: "doc_9_9" });
    expect(r.nodes[0].data.intensity!.groups[0].images).toEqual([
      "doc_9_9__r0c0_i0", "doc_9_9-fig-2__r0c0_i0", "doc_1_10__r0c0_i0", "r0c0_i0", "fig-2__r0c0_i0", "img:doc_1_1__x.tif", aKey,
    ]);
    expect(remapRuntimeKeys([t], {})[0]).toBe(t);
    expect(remapRuntimeKeys([t], { doc_5: "doc_6" })[0]).toBe(t);
  });
});

describe("paths", () => {
  it("relativePath / joinPath (POSIX and Windows)", () => {
    expect(relativePath("/exp", "/exp/sub/b.mpf")).toBe("sub/b.mpf");
    expect(relativePath("/exp/sub", "/exp/a.mpf")).toBe("../a.mpf");
    expect(relativePath("/exp", "~/b.mpf")).toBeUndefined();
    expect(relativePath("C:\\Data\\exp", "c:\\data\\b.mpf")).toBe("..\\b.mpf");
    expect(relativePath("C:\\data", "D:\\b.mpf")).toBeUndefined();
    expect(joinPath("/new", "../a.mpf")).toBe("/a.mpf");
    expect(joinPath("/new/x", "./sub/b.mpf")).toBe("/new/x/sub/b.mpf");
    expect(joinPath("C:\\data\\exp", "..\\b.mpf")).toBe("C:\\data\\b.mpf");
  });
  it("locateCandidates: saved path, then relative to the holder, then next to it", () => {
    expect(locateCandidates("/new/a.mpf", { path: "/exp/sub/b.mpf", rel: "sub/b.mpf" }))
      .toEqual(["/exp/sub/b.mpf", "/new/sub/b.mpf", "/new/b.mpf"]);
    expect(locateCandidates(null, { path: "/exp/b.mpf" })).toEqual(["/exp/b.mpf"]);
    expect(locateCandidates("/exp/a.mpf", { path: "/exp/b.mpf", rel: "b.mpf" })).toEqual(["/exp/b.mpf"]);
  });
});
