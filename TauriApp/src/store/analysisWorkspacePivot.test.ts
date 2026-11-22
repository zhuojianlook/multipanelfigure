/* The Analysis workspace's side of the planner's Pivot: pivotFigureKeys
   moves every key of the pivoted figure's panels (tabs, sourceEditStore,
   the panel's labels in localStorage) and remembers the pivot as pending
   until the figure is saved; revertPendingPivots undoes it when the figure
   edits are thrown away (its workflows are still written to the file then,
   which keeps the figure as it was). */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Never let a test reach a real sidecar. Reads report "no saved workflows".
const mockApi = vi.hoisted(() => ({
  readWorkflows: vi.fn(async () => ({
    manifest: null, blobs: {}, unsupported: false, version: null, uploads: [], renamed: {},
    missing_uploads: [], failed_uploads: [],
  })),
  writeWorkflows: vi.fn(async () => ({ ok: true, size_bytes: 1, workflows_bytes: 1, missing_uploads: [], fast_path: true })),
}));
vi.mock("../api/client", () => ({ api: mockApi, default: vi.fn() }));

import type { Node } from "@xyflow/react";
import type { InsetSource, NodeData, WorkflowTab } from "../utils/analysisTypes";
import { figKey, nsKey } from "../utils/analysisKeys";
import { closedFigKey } from "../utils/workflowOwnership";
import { useCollageStore } from "./collageStore";
import { useSourceEditStore } from "./sourceEditStore";
import {
  ANALYSIS_KEYS_MOVED_EVENT, newSourceNode, resetAnalysisWorkspaceForTests, syncWorkspaceNow, useAnalysisWorkspaceStore,
} from "./analysisWorkspaceStore";

// ── fixtures ────────────────────────────────────────────────

const A = { id: "doc_a", path: "/exp/a.mpf", name: "a" };
const B = { id: "doc_b", path: "/exp/b.mpf", name: "b" };
const fkA = figKey(A);
const fkB = figKey(B);

function src(doc: { id: string; path?: string | null; name?: string }, raw: string): InsetSource {
  const fk = figKey(doc);
  const m = /^r(\d+)c(\d+)_/.exec(raw)!;
  return {
    key: nsKey(fk, raw), rawKey: raw, figKey: fk, mpfId: doc.id, mpf: doc.name,
    row: Number(m[1]), col: Number(m[2]), inset_index: 0, label: raw,
    natural_width: 10, natural_height: 10, thumbnail: "THUMB", runKey: raw,
  };
}
function procNode(id: string, extra: Partial<NodeData> = {}): Node<NodeData> {
  return { id, type: "python", position: { x: 300, y: 30 }, data: { label: id, kind: "python", code: "x = 1", ...extra } as NodeData };
}

const st = () => useAnalysisWorkspaceStore.getState();
const se = () => useSourceEditStore.getState();
const k01 = nsKey(fkA, "r0c1_i0");
const k10 = nsKey(fkA, "r1c0_i0");
const kB = nsKey(fkB, "r0c1_i0");

const readLS = (key: string) => JSON.parse(localStorage.getItem(key) || "null");
const LS_KEYS = ["mpfig.source_names", "mpfig.source_groups", "mpfig.measurement_groups"];

/** Document A's figure with two off-diagonal sources (and B's at the same
 *  cell), their revisions / results / re-run requests and their labels. */
function seed() {
  resetAnalysisWorkspaceForTests();
  const tab: WorkflowTab = {
    id: "w1", name: "Analysis w1",
    nodes: [
      newSourceNode([src(A, "r0c1_i0"), src(A, "r1c0_i0"), src(B, "r0c1_i0")]),
      procNode("p1", { ranWith: { [k01]: 2, [k10]: 5, [kB]: 1 }, outputs: [] }),
    ],
    edges: [],
  };
  useAnalysisWorkspaceStore.setState({ tabs: [tab], activeId: "w1", docUids: { doc_a: "du_a", doc_b: "du_b" } });
  useSourceEditStore.setState({ revs: { [k01]: 2, [k10]: 5, [kB]: 1 }, withResults: [k01, kB], rerun: { [k10]: true } });
  localStorage.setItem("mpfig.source_names", JSON.stringify({ [k01]: "Left inset", [kB]: "B inset" }));
  localStorage.setItem("mpfig.source_groups", JSON.stringify([{ id: "g1", name: "Insets", sourceKeys: [k01, kB] }]));
  localStorage.setItem("mpfig.measurement_groups", JSON.stringify({
    [`${fkA}|R1C2|0|0`]: "ctrl", [`${fkA}|R2C2|0|0`]: "diag", [`${fkB}|R1C2|0|0`]: "other",
  }));
}

/** Everything a pivot moves, as plain data (without the tabs' updatedAt,
 *  which every pivot of their keys bumps). */
function snapshot() {
  return {
    tabs: JSON.parse(JSON.stringify(st().tabs.map(({ updatedAt: _u, ...t }) => t))),
    revs: se().revs, rerun: se().rerun, withResults: se().withResults,
    ls: LS_KEYS.map(readLS),
  };
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(10_000);
  useCollageStore.setState({ openDocs: [A, B], activeDocId: A.id });
  for (const k of LS_KEYS) localStorage.removeItem(k);
  seed();
});
afterEach(() => {
  resetAnalysisWorkspaceForTests();
  for (const k of LS_KEYS) localStorage.removeItem(k);
  vi.useRealTimers();
});

describe("pivotFigureKeys", () => {
  it("moves the figure's keys everywhere, leaves other documents alone, and marks the pivot pending", () => {
    const moved = vi.fn();
    window.addEventListener(ANALYSIS_KEYS_MOVED_EVENT, moved);
    st().pivotFigureKeys(A, "fig-1");
    window.removeEventListener(ANALYSIS_KEYS_MOVED_EVENT, moved);

    const [source, p1] = st().tabs[0].nodes;
    expect(source.data.sources!.map((s) => [s.key, s.row, s.col])).toEqual([[k10, 1, 0], [k01, 0, 1], [kB, 0, 1]]);
    expect(p1.data.ranWith).toEqual({ [k10]: 2, [k01]: 5, [kB]: 1 });
    expect(se().revs).toEqual({ [k10]: 2, [k01]: 5, [kB]: 1 });
    expect(se().rerun).toEqual({ [k01]: true });
    expect(se().withResults).toEqual([k10, kB]);
    expect(readLS("mpfig.source_names")).toEqual({ [k10]: "Left inset", [kB]: "B inset" });
    expect(readLS("mpfig.source_groups")).toEqual([{ id: "g1", name: "Insets", sourceKeys: [k10, kB] }]);
    expect(readLS("mpfig.measurement_groups")).toEqual({
      [`${fkA}|R2C1|0|0`]: "ctrl", [`${fkA}|R2C2|0|0`]: "diag", [`${fkB}|R1C2|0|0`]: "other",
    });
    expect(moved).toHaveBeenCalledTimes(1);
    expect(st().pivotPending).toEqual({ doc_a: { "fig-1": true } });
  });

  it("a second pivot of the same figure moves everything back and clears the pending flag", () => {
    const before = snapshot();
    st().pivotFigureKeys(A, "fig-1");
    st().pivotFigureKeys(A, "fig-1");
    expect(snapshot()).toEqual(before);
    expect(st().pivotPending).toEqual({});
  });

  it("a locked workflow (a co-owner's file is closed) whose keys moved gets a newer updatedAt than its copies on disk", () => {
    // Only A open: B's source is keyed by B's uid, so the tab is greyed out
    // and the sync pass never bumps it. Its copy in B's file is at 5000 too.
    useCollageStore.setState({ openDocs: [A], activeDocId: A.id });
    const fkGone = closedFigKey("du_b");
    const gone: InsetSource = { ...src(B, "r0c1_i0"), key: nsKey(fkGone, "r0c1_i0"), figKey: fkGone, mpfId: "doc_gone", mpf: "gone" };
    const tab: WorkflowTab = {
      id: "w2", name: "Analysis w2", updatedAt: 5000,
      nodes: [newSourceNode([src(A, "r0c1_i0"), gone]), procNode("p1")], edges: [],
    };
    useAnalysisWorkspaceStore.setState({ tabs: [tab], activeId: "w2", docUids: { doc_a: "du_a" } });
    syncWorkspaceNow();
    expect(st().tabs[0].updatedAt).toBe(5000);
    st().pivotFigureKeys(A, "fig-1");
    syncWorkspaceNow();
    expect(st().tabs[0].nodes[0].data.sources!.map((s) => s.key)).toEqual([k10, nsKey(fkGone, "r0c1_i0")]);
    expect(st().tabs[0].updatedAt).toBeGreaterThan(5000);
  });

  it("each figure tab is pending on its own", () => {
    st().pivotFigureKeys(A, "fig-1");
    st().pivotFigureKeys(A, "fig-2");
    expect(st().pivotPending).toEqual({ doc_a: { "fig-1": true, "fig-2": true } });
    // fig-2's keys are `${fkA}#fig-2::…`: none here, so fig-1's stay moved.
    expect(st().tabs[0].nodes[0].data.sources![0].key).toBe(k10);
  });
});

describe("revertPendingPivots", () => {
  it("restores tabs, sourceEditStore and the labels, and clears the flag", () => {
    const before = snapshot();
    st().pivotFigureKeys(A, "fig-1");
    st().pivotFigureKeys(A, "fig-2");
    expect(st().revertPendingPivots(A.id).sort()).toEqual(["fig-1", "fig-2"]);
    expect(snapshot()).toEqual(before);
    expect(st().pivotPending).toEqual({});
  });

  it("nothing pending: nothing changes", () => {
    const tabs = st().tabs;
    expect(st().revertPendingPivots(A.id)).toEqual([]);
    expect(st().tabs).toBe(tabs);
  });
});

describe("the pending flag ends with the figure's fate", () => {
  it("closing the document clears it", () => {
    st().pivotFigureKeys(A, "fig-1");
    st().closeDoc(A.id);
    expect(st().pivotPending[A.id]).toBeUndefined();
  });

  it("saving the figure clears it: the file now has the pivot", async () => {
    st().pivotFigureKeys(A, "fig-1");
    await st().afterDocSaved(A.id, A.path, A.path);
    expect(st().pivotPending[A.id]).toBeUndefined();
    // The moved keys stay.
    expect(st().tabs[0].nodes[0].data.sources![0].key).toBe(k10);
    expect(st().revertPendingPivots(A.id)).toEqual([]);
  });

  it("Save As to a new file, the old one kept: the copy's labels are pivoted, the old file's go back to its cells", async () => {
    st().pivotFigureKeys(A, "fig-1");
    await st().afterDocSaved(A.id, A.path, "/exp/a-copy.mpf", { sourceKept: true });
    expect(st().pivotPending[A.id]).toBeUndefined();
    const fkCopy = figKey({ id: A.id, path: "/exp/a-copy.mpf" });
    const c10 = nsKey(fkCopy, "r1c0_i0");
    expect(readLS("mpfig.source_names")).toEqual({ [k01]: "Left inset", [kB]: "B inset", [c10]: "Left inset" });
    expect(readLS("mpfig.source_groups")).toEqual([{ id: "g1", name: "Insets", sourceKeys: [k01, c10, kB] }]);
    expect(readLS("mpfig.measurement_groups")).toEqual({
      [`${fkA}|R1C2|0|0`]: "ctrl", [`${fkA}|R2C2|0|0`]: "diag", [`${fkB}|R1C2|0|0`]: "other",
      [`${fkCopy}|R2C1|0|0`]: "ctrl", [`${fkCopy}|R2C2|0|0`]: "diag",
    });
  });
});
