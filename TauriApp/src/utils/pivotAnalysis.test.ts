/* The Analysis side of the planner's Pivot (workflowOwnership.pivotTabs):
   after panel (r, c) moves to (c, r), every key the workflows hold for it
   follows it — sources, ranWith, the pickers' image ids — as ONE permutation
   (r0c1 and r1c0 trade places), for the pivoted figure only. Applying it
   twice gives the tabs back, which is how a discarded pivot is undone. */

import { describe, it, expect } from "vitest";
import type { Node } from "@xyflow/react";
import type { InsetSource, NodeData, WorkflowTab } from "./analysisTypes";
import { figKey, nsKey } from "./analysisKeys";
import { pivotRuntimeKey, pivotTabs, type PivotKeyCtx } from "./workflowOwnership";

// ── fixtures ────────────────────────────────────────────────

const A = { id: "doc_a", path: "/exp/a.mpf", name: "a" };
const B = { id: "doc_b", path: "/exp/b.mpf", name: "b" };
const CTX: PivotKeyCtx = { fk: figKey(A), docId: A.id, figureId: "fig-1" };

/** A source of figure `figureId` of `doc`, at the cell its raw key names. */
function src(doc: { id: string; path?: string | null; name?: string }, raw: string, figureId?: string, runKey = raw): InsetSource {
  const fk = figKey(doc, figureId);
  const m = /^r(\d+)c(\d+)_/.exec(raw)!;
  return {
    key: nsKey(fk, raw), rawKey: raw, figKey: fk, figureId, mpfId: doc.id, mpf: doc.name,
    row: Number(m[1]), col: Number(m[2]), inset_index: 0, label: raw,
    natural_width: 10, natural_height: 10, thumbnail: "THUMB", runKey,
  };
}
function upload(name: string): InsetSource {
  return {
    key: `img:${name}`, name, row: -1, col: -1, inset_index: -1, label: name,
    natural_width: 5, natural_height: 5, thumbnail: "UP",
  };
}
function sourceNode(sources: InsetSource[]): Node<NodeData> {
  return { id: "source", type: "source", position: { x: 30, y: 30 }, data: { label: "Source", kind: "source", sources } as NodeData };
}
function node(id: string, data: Partial<NodeData>): Node<NodeData> {
  return { id, type: "python", position: { x: 300, y: 30 }, data: { label: id, kind: "python", code: "x = 1", ...data } as NodeData };
}
function tab(id: string, sources: InsetSource[], extra: Node<NodeData>[] = []): WorkflowTab {
  return { id, name: `Analysis ${id}`, nodes: [sourceNode(sources), ...extra], edges: [] };
}
const sourcesOf = (t: WorkflowTab) => t.nodes[0].data.sources!;
const dataOf = (t: WorkflowTab, id: string) => t.nodes.find((n) => n.id === id)!.data;
const fkA = figKey(A);

// ── sources ─────────────────────────────────────────────────

describe("pivotTabs — sources", () => {
  it("two sources trade places without colliding; row / col and runKey follow", () => {
    const t = tab("w", [src(A, "r0c1_i0"), src(A, "r1c0_i0"), src(A, "r1c1_panel")]);
    const [out] = pivotTabs([t], CTX);
    const s = sourcesOf(out);
    expect(s.map((x) => [x.key, x.rawKey, x.runKey, x.row, x.col])).toEqual([
      [nsKey(fkA, "r1c0_i0"), "r1c0_i0", "r1c0_i0", 1, 0],
      [nsKey(fkA, "r0c1_i0"), "r0c1_i0", "r0c1_i0", 0, 1],
      [nsKey(fkA, "r1c1_panel"), "r1c1_panel", "r1c1_panel", 1, 1],
    ]);
    // A diagonal panel didn't move: the very same object.
    expect(s[2]).toBe(sourcesOf(t)[2]);
    // Everything else about a source is its own.
    expect(s[0].label).toBe("r0c1_i0");
    expect(s[0].thumbnail).toBe("THUMB");
  });

  it("a namespaced runKey keeps its prefix", () => {
    const t = tab("w", [src(A, "r0c2_area1", undefined, "fig-1__r0c2_area1")]);
    const [out] = pivotTabs([t], CTX);
    expect(sourcesOf(out)[0].runKey).toBe("fig-1__r2c0_area1");
  });

  it("another document's, another figure's and uploaded sources are untouched", () => {
    const other = [src(B, "r0c1_i0"), src(A, "r0c1_i0", "fig-2"), upload("r0c1.png")];
    const t = tab("w", other);
    expect(pivotTabs([t], CTX)[0]).toBe(t);
  });

  it("the figure tab named by the context is the one that moves", () => {
    const ctx2: PivotKeyCtx = { fk: figKey(A, "fig-2"), docId: A.id, figureId: "fig-2" };
    const t = tab("w", [src(A, "r0c1_i0"), src(A, "r0c1_i0", "fig-2", "fig-2__r0c1_i0")]);
    const [out] = pivotTabs([t], ctx2);
    expect(sourcesOf(out)[0]).toBe(sourcesOf(t)[0]);
    expect([sourcesOf(out)[1].key, sourcesOf(out)[1].runKey]).toEqual([nsKey(figKey(A, "fig-2"), "r1c0_i0"), "fig-2__r1c0_i0"]);
  });
});

// ── ranWith and picker ids ──────────────────────────────────

describe("pivotTabs — ranWith and picker image ids", () => {
  it("ranWith and bare picker ids follow their panel", () => {
    const t = tab("w", [src(A, "r0c1_i0"), src(A, "r1c0_i0")], [
      node("p1", { ranWith: { [nsKey(fkA, "r0c1_i0")]: 3, [nsKey(fkA, "r1c0_i0")]: 5, "p:/exp/b.mpf::r0c1_i0": 7 } }),
      node("roi", { roi: { version: 1, images: { r0c1_i0: { lanes: [] }, r1c0_i0: { lanes: [{ id: "L" }] } } } as unknown as NodeData["roi"] }),
      node("fi", { intensity: { groups: [{ id: "g", name: "G", images: ["r0c1_i0", nsKey(fkA, "r1c0_i0")] }] } as unknown as NodeData["intensity"] }),
      node("cp", { cellposeSeg: { groups: [{ id: "g", name: "G", images: ["r1c0_i0"] }] } as unknown as NodeData["cellposeSeg"] }),
    ]);
    const [out] = pivotTabs([t], CTX);
    expect(dataOf(out, "p1").ranWith).toEqual({
      [nsKey(fkA, "r1c0_i0")]: 3, [nsKey(fkA, "r0c1_i0")]: 5, "p:/exp/b.mpf::r0c1_i0": 7,
    });
    expect(dataOf(out, "roi").roi!.images).toEqual({ r1c0_i0: { lanes: [] }, r0c1_i0: { lanes: [{ id: "L" }] } });
    expect(dataOf(out, "fi").intensity!.groups[0].images).toEqual(["r1c0_i0", nsKey(fkA, "r0c1_i0")]);
    expect(dataOf(out, "cp").cellposeSeg!.groups![0].images).toEqual(["r0c1_i0"]);
  });

  it("a bare id that another figure's source also carries is left alone (it can't be told apart)", () => {
    const t = tab("w", [src(A, "r0c1_i0"), src(B, "r1c0_i0"), src(A, "r0c2_i0")], [
      node("roi", { roi: { version: 1, images: { r0c1_i0: { lanes: [] }, r1c0_i0: { lanes: [] }, r0c2_i0: { lanes: [] } } } as unknown as NodeData["roi"] }),
    ]);
    const [out] = pivotTabs([t], CTX);
    // r0c1 ↔ r1c0 is blocked (B has r1c0); r0c2 → r2c0 is free.
    expect(Object.keys(dataOf(out, "roi").roi!.images!).sort()).toEqual(["r0c1_i0", "r1c0_i0", "r2c0_i0"]);
    // The source itself still moves: its key names its figure.
    expect(sourcesOf(out)[0].key).toBe(nsKey(fkA, "r1c0_i0"));
  });

  it("namespaced runtime ids of the document move; another document's don't", () => {
    const none = { bare: new Set<string>(), fig: new Set<string>() };
    const k = (x: string) => pivotRuntimeKey(x, CTX, none);
    expect(k("fig-1__r0c1_i0")).toBe("fig-1__r0c1_i0");    // "<figureId>__": only when listed
    expect(pivotRuntimeKey("fig-1__r0c1_i0", CTX, { ...none, fig: new Set(["r0c1_i0"]) })).toBe("fig-1__r1c0_i0");
    expect(k("doc_a__r0c1_i0")).toBe("doc_a__r1c0_i0");
    expect(k("doc_a-fig-1__r0c1_i0")).toBe("doc_a-fig-1__r1c0_i0");
    expect(k(nsKey(fkA, "r0c1_i0"))).toBe(nsKey(fkA, "r1c0_i0"));
    expect(k("doc_b__r0c1_i0")).toBe("doc_b__r0c1_i0");
    expect(k("fig-2__r0c1_i0")).toBe("fig-2__r0c1_i0");
    expect(k("r0c1_i0")).toBe("r0c1_i0");                   // bare: only when listed
    expect(pivotRuntimeKey("r0c1_i0", CTX, { ...none, bare: new Set(["r0c1_i0"]) })).toBe("r1c0_i0");
    // "<docId>__" is the project's FIRST figure only; another figure is "<docId>-<fig>__".
    const ctx2: PivotKeyCtx = { fk: figKey(A, "fig-2"), docId: A.id, figureId: "fig-2" };
    expect(pivotRuntimeKey("doc_a__r0c1_i0", ctx2, none)).toBe("doc_a__r0c1_i0");
    expect(pivotRuntimeKey("doc_a-fig-2__r0c1_i0", ctx2, none)).toBe("doc_a-fig-2__r1c0_i0");
  });

  it("a \"<figureId>__\" id is moved for the figure's own sources, but not when another document's figure with that id has the cell", () => {
    // Picked while A was active with its fig-2 on screen: A's fig-1 panels are "fig-1__…".
    const own = tab("w", [src(A, "r0c1_i0"), src(A, "r0c2_i0")], [
      node("roi", { roi: { version: 1, images: { "fig-1__r0c1_i0": { lanes: [] }, "fig-1__r0c2_i0": { lanes: [] } } } as unknown as NodeData["roi"] }),
    ]);
    expect(Object.keys(dataOf(pivotTabs([own], CTX)[0], "roi").roi!.images!).sort()).toEqual(["fig-1__r1c0_i0", "fig-1__r2c0_i0"]);
    // B's fig-1 shares the figure id: B's "fig-1__r0c1" / "fig-1__r1c0" (picked while B was
    // active) must not trade places when A is pivoted; A's r0c2 is not shared and moves.
    const shared = tab("w", [src(A, "r0c1_i0"), src(A, "r0c2_i0"), src(B, "r0c1_i0"), src(B, "r1c0_i0")], [
      node("roi", { roi: { version: 1, images: { "fig-1__r0c1_i0": { lanes: [{ id: "B01" }] }, "fig-1__r1c0_i0": { lanes: [{ id: "B10" }] }, "fig-1__r0c2_i0": { lanes: [] }, "doc_b__r0c1_i0": { lanes: [] } } } as unknown as NodeData["roi"] }),
    ]);
    const once = pivotTabs([shared], CTX);
    expect(dataOf(once[0], "roi").roi!.images).toEqual({
      "fig-1__r0c1_i0": { lanes: [{ id: "B01" }] }, "fig-1__r1c0_i0": { lanes: [{ id: "B10" }] },
      "fig-1__r2c0_i0": { lanes: [] }, "doc_b__r0c1_i0": { lanes: [] },
    });
    // Another figure id of B doesn't block A's fig-1 ids.
    const other = tab("w", [src(A, "r0c1_i0"), src(B, "r0c1_i0", "fig-2")], [
      node("roi", { roi: { version: 1, images: { "fig-1__r0c1_i0": { lanes: [] } } } as unknown as NodeData["roi"] }),
    ]);
    expect(Object.keys(dataOf(pivotTabs([other], CTX)[0], "roi").roi!.images!)).toEqual(["fig-1__r1c0_i0"]);
    expect(pivotTabs(once, CTX)).toEqual([shared]);
  });

  it("a left-over id on the cell a source moves to trades places with it instead of being overwritten", () => {
    // The source at r1c0 was removed but its picker settings stayed.
    const t = tab("w", [src(A, "r0c1_i0")], [
      node("roi", { roi: { version: 1, images: { r0c1_i0: { lanes: [{ id: "live" }] }, r1c0_i0: { lanes: [{ id: "left-over" }] } } } as unknown as NodeData["roi"] }),
    ]);
    const once = pivotTabs([t], CTX);
    expect(dataOf(once[0], "roi").roi!.images).toEqual({ r1c0_i0: { lanes: [{ id: "live" }] }, r0c1_i0: { lanes: [{ id: "left-over" }] } });
    expect(pivotTabs(once, CTX)).toEqual([t]);
  });

  it("a tab that reads nothing of the figure keeps its pickers", () => {
    const t = tab("w", [src(B, "r0c1_i0")], [
      node("roi", { roi: { version: 1, images: { r0c1_i0: { lanes: [] }, "doc_a__r0c1_i0": { lanes: [] } } } as unknown as NodeData["roi"] }),
    ]);
    expect(pivotTabs([t], CTX)[0]).toBe(t);
  });
});

// ── involution / identity ───────────────────────────────────

describe("pivotTabs — twice is nothing", () => {
  it("applying it twice gives deep-equal tabs back", () => {
    const tabs = [
      tab("w1", [src(A, "r0c1_i0"), src(A, "r1c0_i0"), src(A, "r0c2_panel", undefined, "fig-1__r0c2_panel"), src(B, "r0c1_i0"), upload("x.png")], [
        node("p1", { ranWith: { [nsKey(fkA, "r0c1_i0")]: 1, [nsKey(fkA, "r2c0_panel")]: 2 } }),
        node("roi", { roi: { version: 1, images: { r0c1_i0: { lanes: [] }, r2c0_i0: { lanes: [] }, "doc_a__r0c2_panel": { lanes: [] } } } as unknown as NodeData["roi"] }),
      ]),
      tab("w2", [src(A, "r1c2_area0")]),
    ];
    const once = pivotTabs(tabs, CTX);
    expect(once).not.toBe(tabs);
    expect(pivotTabs(once, CTX)).toEqual(tabs);
  });

  it("nothing to move returns the same array", () => {
    const tabs = [tab("w1", [src(A, "r0c0_i0"), src(A, "r1c1_panel"), upload("x.png")]), tab("w2", [src(B, "r0c1_i0")])];
    expect(pivotTabs(tabs, CTX)).toBe(tabs);
  });
});
