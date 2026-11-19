import { describe, it, expect } from "vitest";
import type { Edge, Node } from "@xyflow/react";
import type { InsetSource, MeasurementRef, NodeData, WorkflowTab } from "./analysisTypes";
import { figKey, nsKey } from "./analysisKeys";
import {
  deriveOwnership, isLocked, missingOwners, isPristineTab, rekeyKey, rekeyRecord, rekeyTabs,
  removeDocSources, rebindDoc, bindMeasurementRefs, docKeyBase, closedKeyBase, closedFigKey,
  closedKeyUid, sourceDocUid, docRefs, newDocUid,
  type DocRef, type OwnershipCtx,
} from "./workflowOwnership";

// ── fixtures ────────────────────────────────────────────────

const A: DocRef = { id: "doc_1_0", uid: "du_a", name: "a", path: "/exp/a.mpf" };
const B: DocRef = { id: "doc_1_1", uid: "du_b", name: "b", path: "/exp/b.mpf" };
const C: DocRef = { id: "doc_1_2", uid: "du_c", name: "c", path: "/exp/c.mpf" };
const ctx = (...docs: DocRef[]): OwnershipCtx => ({ docs });

function src(doc: { id: string; path?: string | null; name?: string }, raw: string, figureId?: string): InsetSource {
  const fk = figKey(doc, figureId);
  return {
    key: nsKey(fk, raw), rawKey: raw, figKey: fk, figureId, mpfId: doc.id, mpf: doc.name,
    row: 0, col: 0, inset_index: 0, label: raw, natural_width: 10, natural_height: 10,
    thumbnail: "THUMB", runKey: raw,
  };
}
/** A source of a CLOSED document, as re-keyed on close. */
function closedSrc(uid: string, raw: string, figureId?: string): InsetSource {
  const fk = closedFigKey(uid, figureId);
  return { ...src(A, raw, figureId), key: nsKey(fk, raw), figKey: fk, mpfId: "doc_gone", mpf: "gone" };
}
function upload(name: string): InsetSource {
  return {
    key: `img:${name}`, name, row: -1, col: -1, inset_index: -1, label: name,
    natural_width: 5, natural_height: 5, thumbnail: "UP",
  };
}
function meas(doc: DocRef, csv = "Panel,Name\nR1C1,L1", withUid = true): MeasurementRef {
  return { mpfId: doc.id, mpf: doc.name || "", label: `${doc.name} — measurements`, csv, ...(withUid ? { docUid: doc.uid } : {}) };
}
function sourceNode(id: string, sources: InsetSource[], refs: MeasurementRef[] = []): Node<NodeData> {
  return {
    id, type: "source", position: { x: 30, y: 30 },
    data: {
      label: "Source", kind: "source", sources, status: "ok",
      ...(refs.length ? { measurementsRef: refs[0], measurementRefs: refs } : {}),
    } as NodeData,
  };
}
function procNode(id: string, extra: Partial<NodeData> = {}): Node<NodeData> {
  return { id, type: "python", position: { x: 300, y: 30 }, data: { label: id, kind: "python", code: "x = 1", ...extra } as NodeData };
}
function edge(source: string, sourceHandle: string, target: string, targetHandle = "in_image"): Edge {
  return { id: `e_${source}_${sourceHandle}__${target}_${targetHandle}`, source, sourceHandle, target, targetHandle, type: "deletable" };
}
function wf(id: string, nodes: Node<NodeData>[], edges: Edge[] = [], extra: Partial<WorkflowTab> = {}): WorkflowTab {
  return { id, name: "Analysis " + id, nodes, edges, ...extra };
}
const withSources = (w: WorkflowTab, sources: InsetSource[], refs: MeasurementRef[] = []): WorkflowTab => ({
  ...w, nodes: [sourceNode("source", sources, refs), ...w.nodes.filter((n) => n.id !== "source")],
});

// ── deriveOwnership ─────────────────────────────────────────

describe("deriveOwnership", () => {
  it("no participants: owners is [home], or nothing without a home", () => {
    const w = wf("w", [sourceNode("source", [])], [], { home: "du_a" });
    expect(deriveOwnership(w, ctx(A, B))).toEqual({ participants: [], home: "du_a", owners: ["du_a"], shared: false });
    expect(deriveOwnership({ ...w, home: undefined }, ctx(A)).owners).toEqual([]);
  });

  it("one participant: single-owner, and home is forced to it", () => {
    const w = wf("w", [sourceNode("source", [src(B, "r0c0_i0"), src(B, "r0c1_panel")])], [], { home: "du_a" });
    const o = deriveOwnership(w, ctx(A, B));
    expect(o).toEqual({ participants: ["du_b"], home: "du_b", owners: ["du_b"], shared: false });
  });

  it("two participants: shared, in attach order; home kept when it participates, else the first", () => {
    const nodes = [sourceNode("source", [src(A, "r0c0_i0")]), sourceNode("src_2", [src(B, "r0c0_i0"), src(A, "r1c0_i0")])];
    const both = ctx(A, B);
    expect(deriveOwnership(wf("w", nodes, [], { home: "du_b" }), both))
      .toEqual({ participants: ["du_a", "du_b"], home: "du_b", owners: ["du_a", "du_b"], shared: true });
    expect(deriveOwnership(wf("w", nodes, [], { home: "du_zzz" }), both).home).toBe("du_a");
    expect(deriveOwnership(wf("w", nodes), both).home).toBe("du_a");
  });

  it("measurement tables count as participants — by docUid, or by a legacy ref's open mpfId", () => {
    const w = wf("w", [sourceNode("source", [src(A, "r0c0_i0")], [meas(B)])]);
    expect(deriveOwnership(w, ctx(A, B)).owners).toEqual(["du_a", "du_b"]);
    const legacy = wf("w", [sourceNode("source", [], [meas(C, undefined, false)])]);
    expect(deriveOwnership(legacy, ctx(A, C)).participants).toEqual(["du_c"]);
    // A legacy ref whose session doc is gone can't be attributed.
    expect(deriveOwnership(legacy, ctx(A)).participants).toEqual([]);
  });

  it("uploads never make a document a participant", () => {
    const w = wf("w", [sourceNode("source", [upload("stack.tif"), upload("b.png")])], [], { home: "du_c" });
    expect(deriveOwnership(w, ctx(A, B, C))).toEqual({ participants: [], home: "du_c", owners: ["du_c"], shared: false });
    // Uploads next to figure data: the figure decides.
    expect(deriveOwnership(withSources(w, [upload("stack.tif"), src(A, "r0c0_i0")]), ctx(A, B, C)).owners).toEqual(["du_a"]);
  });

  it("maintains home across an attach / detach sequence (store writes the derived home back)", () => {
    const docs = ctx(A, B);
    let w = wf("w", [sourceNode("source", [])], [], { home: "du_a" });   // created in A
    const step = (sources: InsetSource[], refs: MeasurementRef[] = []) => {
      w = withSources(w, sources, refs);
      const o = deriveOwnership(w, docs);
      w = { ...w, home: o.home };
      return o;
    };
    expect(step([upload("u.tif")]).owners).toEqual(["du_a"]);                       // uploads only: A
    expect(step([upload("u.tif"), src(B, "r0c0_i0")]).home).toBe("du_b");          // B only → B
    let o = step([upload("u.tif"), src(B, "r0c0_i0"), src(A, "r0c0_i0")]);
    expect([o.home, o.shared]).toEqual(["du_b", true]);                             // shared, home stays B
    o = step([upload("u.tif"), src(A, "r0c0_i0")]);                                  // detach B
    expect([o.home, o.owners, o.shared]).toEqual(["du_a", ["du_a"], false]);
    o = step([upload("u.tif")], [meas(B)]);                                           // only B's table
    expect(o.owners).toEqual(["du_b"]);
    o = step([upload("u.tif")]);                                                      // detach everything
    expect([o.home, o.owners]).toEqual(["du_b", ["du_b"]]);                          // last home kept
  });

  it("resolves closed-document keys and a known closed document's path", () => {
    const w = wf("w", [sourceNode("source", [src(A, "r0c0_i0"), closedSrc("du_b", "r0c0_i0", "fig-2")])]);
    expect(deriveOwnership(w, ctx(A)).owners).toEqual(["du_a", "du_b"]);
    // Still path-keyed (not yet re-keyed) but the document is known closed.
    const stale = wf("w", [sourceNode("source", [src(A, "r0c0_i0"), src(B, "r0c0_i0")])]);
    expect(deriveOwnership(stale, { docs: [A], known: { du_b: { path: "/exp/b.mpf" } } }).owners).toEqual(["du_a", "du_b"]);
    // Unknown: not attributable, so not an owner.
    expect(deriveOwnership(stale, ctx(A)).owners).toEqual(["du_a"]);
  });
});

describe("isLocked", () => {
  const shared = wf("w", [sourceNode("source", [src(A, "r0c0_i0"), closedSrc("du_b", "r0c0_i0")])]);
  it("is locked while some owner is closed", () => {
    expect(isLocked(shared, ctx(A))).toBe(true);
    expect(missingOwners(shared, ctx(A))).toEqual(["du_b"]);
  });
  it("unlocks once every owner is open", () => {
    const open = rekeyTabs([shared], closedKeyBase("du_b"), docKeyBase(B))[0];
    expect(isLocked(open, ctx(A, B))).toBe(false);
    // u:-keyed but its document is open (not re-keyed yet) — still not locked.
    expect(isLocked(shared, ctx(A, B))).toBe(false);
  });
  it("single-owner and uploads-only workflows lock only on their own document", () => {
    expect(isLocked(wf("w", [sourceNode("source", [src(A, "r0c0_i0")])]), ctx(A))).toBe(false);
    const up = wf("w", [sourceNode("source", [upload("x.tif")])], [], { home: "du_b" });
    expect(isLocked(up, ctx(A))).toBe(true);
    expect(isLocked(up, ctx(A, B))).toBe(false);
  });
});

// ── removeDocSources / rebindDoc ────────────────────────────

describe("removeDocSources", () => {
  const nodes = [
    sourceNode("source", [src(A, "r0c0_i0"), closedSrc("du_b", "r0c0_i0"), src(A, "r1c0_i0")], [meas({ ...B, id: "doc_old" }), meas(A)]),
    procNode("p1"), procNode("r1", { kind: "r" }),
  ];
  const edges = [
    edge("source", "out_image_0", "p1"), edge("source", "out_image_1", "p1"), edge("source", "out_image_2", "p1"),
    edge("source", "out_table_measurements", "r1", "in_table"), edge("source", "out_table_measurements_1", "r1", "in_table"),
    edge("p1", "out_table", "r1", "in_table"),
  ];
  const shared = wf("w", nodes, edges, { home: "du_b" });

  it("drops the missing document's sources and tables, making the workflow single-owner", () => {
    expect(isLocked(shared, ctx(A))).toBe(true);
    const out = removeDocSources(shared, "du_b", ctx(A), "du_a");
    const s = out.nodes[0].data;
    expect(s.sources!.map((x) => x.key)).toEqual([nsKey("p:/exp/a.mpf", "r0c0_i0"), nsKey("p:/exp/a.mpf", "r1c0_i0")]);
    expect(s.measurementRefs!.map((r) => r.docUid)).toEqual(["du_a"]);
    expect(s.measurementsRef!.docUid).toBe("du_a");
    expect(deriveOwnership(out, ctx(A))).toMatchObject({ owners: ["du_a"], shared: false });
    expect(isLocked(out, ctx(A))).toBe(false);
    expect(out.home).toBe("du_a");
  });

  it("drops edges off removed slots and re-indexes the survivors", () => {
    const out = removeDocSources(shared, "du_b", ctx(A));
    const fromSource = out.edges.filter((e) => e.source === "source").map((e) => [e.sourceHandle, e.target, e.id]);
    expect(fromSource).toEqual([
      ["out_image_0", "p1", "e_source_out_image_0__p1_in_image"],
      ["out_image_1", "p1", "e_source_out_image_1__p1_in_image"],
      ["out_table_measurements", "r1", "e_source_out_table_measurements__r1_in_table"],
    ]);
    expect(out.edges.some((e) => e.source === "p1")).toBe(true);
  });

  it("returns the workflow itself when it holds nothing of that document", () => {
    const single = wf("w", [sourceNode("source", [src(A, "r0c0_i0")])], [], { home: "du_a" });
    expect(removeDocSources(single, "du_b", ctx(A))).toBe(single);
  });
});

describe("rebindDoc", () => {
  it("moves u: keys, tables and home from one uid to another, leaving other documents alone", () => {
    const w = wf("w", [
      sourceNode("source", [closedSrc("du_old", "r0c0_i0"), closedSrc("du_old", "r0c0_i0", "fig-2"), src(A, "r0c0_i0")],
        [{ ...meas(B), docUid: "du_old" }]),
      procNode("p1", { ranWith: { [nsKey("u:du_old", "r0c0_i0")]: 2, [nsKey("p:/exp/a.mpf", "r0c0_i0")]: 1 } }),
    ], [], { home: "du_old" });
    const out = rebindDoc(w, "du_old", "du_new");
    const s = out.nodes[0].data.sources!;
    expect(s.map((x) => x.key)).toEqual(["u:du_new::r0c0_i0", "u:du_new#fig-2::r0c0_i0", "p:/exp/a.mpf::r0c0_i0"]);
    expect(s.map((x) => x.figKey)).toEqual(["u:du_new", "u:du_new#fig-2", "p:/exp/a.mpf"]);
    expect(out.nodes[0].data.measurementRefs![0].docUid).toBe("du_new");
    expect(Object.keys(out.nodes[1].data.ranWith!).sort()).toEqual(["p:/exp/a.mpf::r0c0_i0", "u:du_new::r0c0_i0"]);
    expect(out.home).toBe("du_new");
    expect(deriveOwnership(out, ctx(A)).owners).toEqual(["du_new", "du_a"]);
  });
  it("is a no-op for the same uid", () => {
    const w = wf("w", [sourceNode("source", [closedSrc("du_x", "r0c0_i0")])]);
    expect(rebindDoc(w, "du_x", "du_x")).toBe(w);
  });
});

// ── re-keying ───────────────────────────────────────────────

describe("rekeyKey", () => {
  it("d: → p: when an Untitled document is first saved", () => {
    expect(rekeyKey("d:doc_1::r0c0_i0", "d:doc_1", "p:/exp/new.mpf")).toBe("p:/exp/new.mpf::r0c0_i0");
    expect(rekeyKey("d:doc_1", "d:doc_1", "p:/exp/new.mpf")).toBe("p:/exp/new.mpf");
    // A different doc id that merely starts the same.
    expect(rekeyKey("d:doc_12::r0c0_i0", "d:doc_1", "p:/x.mpf")).toBe("d:doc_12::r0c0_i0");
  });
  it("p: ↔ u: on close and reopen", () => {
    const k = "p:/exp/a.mpf::r0c0_i0";
    const closed = rekeyKey(k, "p:/exp/a.mpf", closedKeyBase("du_a"));
    expect(closed).toBe("u:du_a::r0c0_i0");
    expect(rekeyKey(closed, "u:du_a", "p:/exp/a.mpf")).toBe(k);
    expect(rekeyKey("u:du_ab::r0c0_i0", "u:du_a", "p:/x.mpf")).toBe("u:du_ab::r0c0_i0");
  });
  it("carries a figure-tab suffix across", () => {
    expect(rekeyKey("p:/exp/a.mpf#fig-2::r0c0_i0", "p:/exp/a.mpf", "u:du_a")).toBe("u:du_a#fig-2::r0c0_i0");
    expect(rekeyKey("p:/exp/a.mpf#fig-lx3k9a", "p:/exp/a.mpf", "u:du_a")).toBe("u:du_a#fig-lx3k9a");
    expect(rekeyKey("u:du_a#fig-2::r1c1_area0", "u:du_a", "p:/moved/a.mpf")).toBe("p:/moved/a.mpf#fig-2::r1c1_area0");
  });
  it("never confuses a path containing '#' (or '::') with a figure suffix", () => {
    expect(rekeyKey("p:/x/a#b.mpf::r0c0_i0", "p:/x/a", "u:du_z")).toBe("p:/x/a#b.mpf::r0c0_i0");
    expect(rekeyKey("p:/x/a#b.mpf", "p:/x/a", "u:du_z")).toBe("p:/x/a#b.mpf");
    expect(rekeyKey("p:/x/a#b.mpf::r0c0_i0", "p:/x/a#b.mpf", "u:du_z")).toBe("u:du_z::r0c0_i0");
    expect(rekeyKey("p:/x/a#b.mpf#fig-3::r0c0_i0", "p:/x/a#b.mpf", "u:du_z")).toBe("u:du_z#fig-3::r0c0_i0");
    expect(rekeyKey("p:/x/a::b.mpf::r0c0_i0", "p:/x/a", "u:du_z")).toBe("p:/x/a::b.mpf::r0c0_i0");
  });
  it("moves measurement row keys (with and without a figure suffix)", () => {
    expect(rekeyKey("p:/exp/a.mpf|R1C1|0|1", "p:/exp/a.mpf", "p:/exp/b.mpf")).toBe("p:/exp/b.mpf|R1C1|0|1");
    expect(rekeyKey("p:/exp/a.mpf#fig-2|R1C1|0|", "p:/exp/a.mpf", "u:du_a")).toBe("u:du_a#fig-2|R1C1|0|");
    expect(rekeyKey("d:doc_1|R2C3|1|", "d:doc_1", "p:/n.mpf")).toBe("p:/n.mpf|R2C3|1|");
    // "/exp/a.mpf|x.mpf" is another file whose path has a "|" in it.
    expect(rekeyKey("p:/exp/a.mpf|x.mpf|R1C1|0|0", "p:/exp/a.mpf", "u:du_a")).toBe("p:/exp/a.mpf|x.mpf|R1C1|0|0");
  });
  it("leaves uploads, raw and runtime keys alone", () => {
    for (const k of ["img:p:/exp/a.mpf.tif", "r0c0_i0", "doc_1_0__r0c0_i0", ""]) {
      expect(rekeyKey(k, "p:/exp/a.mpf", "u:du_a")).toBe(k);
    }
    expect(rekeyKey("p:/exp/a.mpf::r0c0_i0", "p:/exp/a.mpf", "p:/exp/a.mpf")).toBe("p:/exp/a.mpf::r0c0_i0");
  });
});

describe("rekeyRecord / rekeyTabs", () => {
  it("rekeyRecord moves matching keys and returns the same object when none match", () => {
    const rec = { "p:/exp/a.mpf::r0c0_i0": "Control_1", "p:/exp/b.mpf::r0c0_i0": "Drug_1", "r0c0_i0": "legacy" };
    const out = rekeyRecord(rec, "p:/exp/a.mpf", "u:du_a");
    expect(out).toEqual({ "u:du_a::r0c0_i0": "Control_1", "p:/exp/b.mpf::r0c0_i0": "Drug_1", "r0c0_i0": "legacy" });
    expect(rekeyRecord(rec, "p:/nope.mpf", "u:x")).toBe(rec);
  });

  it("rekeyTabs moves sources, ranWith and picker image ids; untouched tabs keep identity", () => {
    const a0 = nsKey("p:/exp/a.mpf", "r0c0_i0");
    const a2 = nsKey("p:/exp/a.mpf#fig-2", "r0c0_i0");
    const b0 = nsKey("p:/exp/b.mpf", "r0c0_i0");
    const t1 = wf("t1", [
      sourceNode("source", [src(A, "r0c0_i0"), src(A, "r0c0_i0", "fig-2"), src(B, "r0c0_i0")]),
      procNode("bands", {
        ranWith: { [a0]: 1, [b0]: 2 },
        roi: { version: 1, images: { [a0]: { lanes: [], groups: [] }, "doc_1_1__r0c0_i0": { lanes: [], groups: [] } } } as unknown as NodeData["roi"],
      }),
      procNode("int", { intensity: { groups: [{ id: "g", name: "Ctl", images: [a2, "r0c0_i0"] }] } as unknown as NodeData["intensity"] }),
      procNode("cp", { kind: "cellpose", cellposeSeg: { groups: [{ id: "g", name: "T", images: [a0] }] } as unknown as NodeData["cellposeSeg"] }),
    ]);
    const t2 = wf("t2", [sourceNode("source", [src(B, "r0c0_i0")])]);
    const tabs = [t1, t2];
    const out = rekeyTabs(tabs, "p:/exp/a.mpf", "u:du_a");
    expect(out[1]).toBe(t2);
    const s = out[0].nodes[0].data.sources!;
    expect(s.map((x) => x.key)).toEqual(["u:du_a::r0c0_i0", "u:du_a#fig-2::r0c0_i0", b0]);
    expect(s.map((x) => x.figKey)).toEqual(["u:du_a", "u:du_a#fig-2", "p:/exp/b.mpf"]);
    expect(out[0].nodes[0].data.sources![2]).toBe(t1.nodes[0].data.sources![2]);
    expect(out[0].nodes[1].data.ranWith).toEqual({ "u:du_a::r0c0_i0": 1, [b0]: 2 });
    expect(Object.keys(out[0].nodes[1].data.roi!.images!)).toEqual(["u:du_a::r0c0_i0", "doc_1_1__r0c0_i0"]);
    expect(out[0].nodes[2].data.intensity!.groups[0].images).toEqual(["u:du_a#fig-2::r0c0_i0", "r0c0_i0"]);
    expect(out[0].nodes[3].data.cellposeSeg!.groups![0].images).toEqual(["u:du_a::r0c0_i0"]);
    // Nothing under the base → the very same array.
    expect(rekeyTabs(tabs, "p:/elsewhere.mpf", "u:x")).toBe(tabs);
  });
});

describe("bindMeasurementRefs", () => {
  it("re-binds a reopened document's tables to its new session id", () => {
    const w = wf("w", [sourceNode("source", [], [{ ...meas(A), mpfId: "doc_old" }, meas(B)])]);
    const [out] = bindMeasurementRefs([w], "du_a", "doc_new");
    expect(out.nodes[0].data.measurementRefs!.map((r) => [r.docUid, r.mpfId])).toEqual([["du_a", "doc_new"], ["du_b", B.id]]);
    expect(out.nodes[0].data.measurementsRef!.mpfId).toBe("doc_new");
    expect(bindMeasurementRefs([out], "du_a", "doc_new")[0]).toBe(out);
  });
  it("back-fills the uid of legacy tables from their session id", () => {
    const w = wf("w", [sourceNode("source", [], [meas(A, undefined, false)])]);
    const [out] = bindMeasurementRefs([w], "du_a", null, A.id);
    expect(out.nodes[0].data.measurementRefs![0]).toMatchObject({ docUid: "du_a", mpfId: A.id });
  });
});

// ── small helpers ───────────────────────────────────────────

describe("isPristineTab", () => {
  const fresh = (): WorkflowTab => ({ id: "w", name: "Workflow 2", nodes: [sourceNode("source", [])], edges: [] });
  it("a brand-new tab is pristine, even with its node moved", () => {
    expect(isPristineTab(fresh())).toBe(true);
    const moved = fresh();
    moved.nodes[0] = { ...moved.nodes[0], position: { x: 400, y: 90 } };
    expect(isPristineTab(moved)).toBe(true);
  });
  it("any real content makes it non-pristine", () => {
    expect(isPristineTab({ ...fresh(), name: "Western blot" })).toBe(false);
    expect(isPristineTab(withSources(fresh(), [src(A, "r0c0_i0")]))).toBe(false);
    expect(isPristineTab(withSources(fresh(), [], [meas(A)]))).toBe(false);
    expect(isPristineTab(withSources(fresh(), [upload("x.tif")]))).toBe(false);
    expect(isPristineTab({ ...fresh(), nodes: [...fresh().nodes, procNode("p1")] })).toBe(false);
    const relabeled = fresh();
    relabeled.nodes[0] = { ...relabeled.nodes[0], data: { ...relabeled.nodes[0].data, label: "Blots" } };
    expect(isPristineTab(relabeled)).toBe(false);
  });
});

describe("identities", () => {
  it("closed keys round-trip their uid; figKey-style suffix rule", () => {
    expect(closedFigKey("du_a")).toBe("u:du_a");
    expect(closedFigKey("du_a", "fig-1")).toBe("u:du_a");
    expect(closedFigKey("du_a", "fig-2")).toBe("u:du_a#fig-2");
    expect(closedKeyUid("u:du_a#fig-2")).toBe("du_a");
    expect(closedKeyUid("p:/x#y.mpf")).toBeUndefined();
    expect(sourceDocUid(closedSrc("du_q", "r0c0_i0", "fig-9"), ctx())).toBe("du_q");
    expect(sourceDocUid(upload("x.tif"), ctx(A))).toBeUndefined();
  });
  it("docRefs joins open tabs with their uids; newDocUid is fresh", () => {
    expect(docRefs([{ id: "d1", name: "a", path: "/a.mpf" }, { id: "d2", name: "b", path: null }], { d1: "du_1" }))
      .toEqual([{ id: "d1", uid: "du_1", name: "a", path: "/a.mpf" }]);
    const u = newDocUid();
    expect(u).toMatch(/^du_[a-z0-9]+$/);
    expect(newDocUid()).not.toBe(u);
  });
});

describe("sourceDocUid — two open tabs on the same path", () => {
  it("binds to the source's own tab, whatever order the tabs are in", () => {
    const P: DocRef = { id: "doc_p", uid: "du_p", name: "fig", path: "/exp/fig.mpf" };
    const Q: DocRef = { id: "doc_q", uid: "du_q2", name: "fig", path: "/exp/fig.mpf" };
    const s = src(Q, "r0c0_i0");
    expect(sourceDocUid(s, ctx(P, Q))).toBe("du_q2");
    expect(sourceDocUid(s, ctx(Q, P))).toBe("du_q2");
    // A source whose session tab is gone falls back to the first match.
    expect(sourceDocUid({ ...s, mpfId: "doc_gone" }, ctx(P, Q))).toBe("du_p");
  });
});
