/* The Analysis workspace store (plan step 4): the graph's workflow state lifted
   out of the component. Pins the useState contract the graph relies on (an
   updater returning the same reference is a no-op — without it the graph's
   effects loop), that state outlives the view, the document uids, deleting a
   workflow (tombstones), the upload tiles, re-keying, and the debounced pass
   that keeps updatedAt / home. Persistence is covered by
   analysisWorkspacePersist.test.ts. */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// The store reads / writes .mpf workflows through the sidecar: never let a
// test reach a real one. Reads report "no saved workflows".
const mockApi = vi.hoisted(() => ({
  readWorkflows: vi.fn(async () => ({
    manifest: null, blobs: {}, unsupported: false, version: null, uploads: [], renamed: {},
    missing_uploads: [], failed_uploads: [],
  })),
  writeWorkflows: vi.fn(async () => ({ ok: true, size_bytes: 1, workflows_bytes: 1, missing_uploads: [], fast_path: true })),
}));
vi.mock("../api/client", () => ({ api: mockApi, default: vi.fn() }));
import { useCallback, useEffect } from "react";
import { act, renderHook } from "@testing-library/react";
import type { Edge, Node } from "@xyflow/react";
import type { InsetSource, MeasurementRef, NodeData, WorkflowTab } from "../utils/analysisTypes";
import { figKey, nsKey } from "../utils/analysisKeys";
import { closedFigKey } from "../utils/workflowOwnership";
import { useCollageStore } from "./collageStore";
import { useSourceEditStore } from "./sourceEditStore";
import {
  useAnalysisWorkspaceStore, resetAnalysisWorkspaceForTests, syncWorkspaceNow, SYNC_DEBOUNCE_MS,
  newSourceNode, newWorkflowTab, ownerChip, workspaceOwnershipCtx,
  type AnalysisWorkspaceState,
} from "./analysisWorkspaceStore";

// ── fixtures ────────────────────────────────────────────────

const A = { id: "doc_a", path: "/exp/a.mpf", name: "a" };
const B = { id: "doc_b", path: "/exp/b.mpf", name: "b" };

function src(doc: { id: string; path?: string | null; name?: string }, raw: string): InsetSource {
  const fk = figKey(doc);
  return {
    key: nsKey(fk, raw), rawKey: raw, figKey: fk, mpfId: doc.id, mpf: doc.name,
    row: 0, col: 0, inset_index: 0, label: raw, natural_width: 10, natural_height: 10,
    thumbnail: "THUMB", runKey: raw,
  };
}
/** A source of a CLOSED document (keyed u:<uid>). */
function closedSrc(uid: string, raw: string): InsetSource {
  const fk = closedFigKey(uid);
  return { ...src(A, raw), key: nsKey(fk, raw), figKey: fk, mpfId: "doc_gone", mpf: "gone" };
}
function upload(name: string, thumbnail = "UP"): InsetSource {
  return {
    key: `img:${name}`, name, row: -1, col: -1, inset_index: -1, label: name,
    natural_width: 5, natural_height: 5, thumbnail,
  };
}
function sourceNode(sources: InsetSource[], refs: MeasurementRef[] = []): Node<NodeData> {
  const n = newSourceNode(sources);
  if (refs.length) n.data = { ...n.data, measurementsRef: refs[0], measurementRefs: refs };
  return n;
}
function procNode(id: string, extra: Partial<NodeData> = {}): Node<NodeData> {
  return { id, type: "python", position: { x: 300, y: 30 }, data: { label: id, kind: "python", code: "x = 1", ...extra } as NodeData };
}
function edge(source: string, target: string): Edge {
  return { id: `e_${source}_out_image_0__${target}_in_image`, source, sourceHandle: "out_image_0", target, targetHandle: "in_image", type: "deletable" };
}
function wf(id: string, sources: InsetSource[], extra: Partial<WorkflowTab> = {}): WorkflowTab {
  return {
    id, name: `Analysis ${id}`,
    nodes: [sourceNode(sources), procNode("p1", { outputs: [{ id: "out_plot_0__v1", kind: "plot", name: "plot_1", payload: "PNG1", versionLabel: "v1", blobId: "b_1" }] })],
    edges: [edge("source", "p1")],
    ...extra,
  };
}

const st = () => useAnalysisWorkspaceStore.getState();

/** A workspace with deterministic uids for the two open documents. */
function seed(over: Partial<AnalysisWorkspaceState> = {}) {
  resetAnalysisWorkspaceForTests();
  useAnalysisWorkspaceStore.setState({ docUids: { doc_a: "du_a", doc_b: "du_b" }, ...over });
}

/** Counts store notifications until the returned stop function is called. */
function countNotifications(): { count: () => number; stop: () => void } {
  let n = 0;
  const stop = useAnalysisWorkspaceStore.subscribe(() => { n++; });
  return { count: () => n, stop };
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(10_000);
  useCollageStore.setState({ openDocs: [A, B], activeDocId: A.id });
  useSourceEditStore.setState({ revs: {}, withResults: [], rerun: {} });
  resetAnalysisWorkspaceForTests();
});
afterEach(() => {
  resetAnalysisWorkspaceForTests();
  vi.useRealTimers();
});

// ── initial state ───────────────────────────────────────────

describe("initial workspace", () => {
  it("one pristine 'Workflow 1', active, saved in the active document", () => {
    const s = st();
    expect(s.tabs).toHaveLength(1);
    expect(s.tabs[0].name).toBe("Workflow 1");
    expect(s.tabs[0].nodes.map((n) => n.id)).toEqual(["source"]);
    expect(s.activeId).toBe(s.tabs[0].id);
    expect(s.tabs[0].home).toBe(s.docUids[A.id]);
    expect(s.uploads).toEqual([]);
  });

  it("every open document has a uid; opening another one gives it one", () => {
    const s = st();
    expect(s.docUids[A.id]).toMatch(/^du_/);
    expect(s.docUids[B.id]).toMatch(/^du_/);
    expect(s.docUids[A.id]).not.toBe(s.docUids[B.id]);
    useCollageStore.setState({ openDocs: [A, B, { id: "doc_c", path: null, name: "Untitled_1" }] });
    expect(st().docUids.doc_c).toMatch(/^du_/);
    expect(st().docUids[A.id]).toBe(s.docUids[A.id]);   // existing uids untouched
  });

  it("the graph's former session storage is removed on load", async () => {
    localStorage.setItem("mpfig.workflow_session", "{\"tabs\":[]}");
    localStorage.setItem("mpfig.analysis.uploads", "[]");
    sessionStorage.setItem("mpfig.analysis.sessionMarker", "1");
    localStorage.setItem("mpfig.saved_workflows", "[]");          // a real preference — kept
    vi.resetModules();
    await import("./analysisWorkspaceStore");
    expect(localStorage.getItem("mpfig.workflow_session")).toBeNull();
    expect(localStorage.getItem("mpfig.analysis.uploads")).toBeNull();
    expect(sessionStorage.getItem("mpfig.analysis.sessionMarker")).toBeNull();
    expect(localStorage.getItem("mpfig.saved_workflows")).toBe("[]");
    localStorage.removeItem("mpfig.saved_workflows");
  });
});

// ── setTabs: the graph's useState contract ──────────────────

describe("setTabs keeps setWorkflowTabs' contract", () => {
  it("an updater (or value) returning the same array is a no-op: no notification", () => {
    const n = countNotifications();
    const before = st().tabs;
    st().setTabs((tabs) => tabs);
    st().setTabs(before);
    expect(st().tabs).toBe(before);
    expect(n.count()).toBe(0);
    n.stop();
  });

  it("a new array replaces the tabs (value or updater)", () => {
    const t = wf("w1", [src(A, "r0c0_i0")]);
    st().setTabs([t]);
    expect(st().tabs).toEqual([t]);
    st().setTabs((tabs) => [...tabs, wf("w2", [])]);
    expect(st().tabs.map((x) => x.id)).toEqual(["w1", "w2"]);
    expect(st().tabs[0]).toBe(t);
  });

  it("never empty: clearing leaves one pristine tab saved in the active document", () => {
    seed();
    st().setTabs([]);
    expect(st().tabs).toHaveLength(1);
    expect(st().tabs[0].name).toBe("Workflow 1");
    expect(st().tabs[0].home).toBe("du_a");
  });

  it("plain zustand: tabs, nodes and payloads are not frozen", () => {
    const t = wf("w1", [src(A, "r0c0_i0")]);
    st().setTabs([t]);
    const tab = st().tabs[0];
    expect(Object.isFrozen(tab)).toBe(false);
    expect(Object.isFrozen(tab.nodes)).toBe(false);
    expect(Object.isFrozen(tab.nodes[1].data)).toBe(false);
    expect(Object.isFrozen(tab.nodes[1].data.outputs)).toBe(false);
  });
});

describe("setTabNodes / setTabEdges (the graph's setNodes / setEdges)", () => {
  it("an updater returning the same nodes / edges changes nothing", () => {
    st().setTabs([wf("w1", [src(A, "r0c0_i0")]), wf("w2", [src(B, "r0c0_i0")])]);
    const before = st().tabs;
    const n = countNotifications();
    st().setTabNodes("w1", (nodes) => nodes);
    st().setTabEdges("w1", (edges) => edges);
    st().setTabNodes("nope", () => []);                 // unknown tab
    expect(st().tabs).toBe(before);
    expect(n.count()).toBe(0);
    n.stop();
  });

  it("a change replaces only that tab; the others keep their identity", () => {
    st().setTabs([wf("w1", [src(A, "r0c0_i0")]), wf("w2", [src(B, "r0c0_i0")])]);
    const [w1, w2] = st().tabs;
    st().setTabNodes("w1", (nodes) => nodes.map((x) => (x.id === "p1" ? { ...x, position: { x: 1, y: 2 } } : x)));
    expect(st().tabs[1]).toBe(w2);
    expect(st().tabs[0]).not.toBe(w1);
    expect(st().tabs[0].edges).toBe(w1.edges);
    expect(st().tabs[0].nodes[0]).toBe(w1.nodes[0]);
    expect(st().tabs[0].nodes[1].position).toEqual({ x: 1, y: 2 });
    st().setTabEdges("w2", () => []);
    expect(st().tabs[1].edges).toEqual([]);
    expect(st().tabs[1].nodes).toBe(w2.nodes);
  });

  it("an effect that writes nodes back on every change settles (no infinite update)", () => {
    // The graph has effects that run on every tabs / nodes change and write
    // back through setNodes (the old parseDeclaredOutputs write-back was
    // one); they terminate only because a no-op write is a no-op. Without
    // the bail-out each write makes a new tabs array, which re-runs the
    // effect forever — the documented "infinite update" blank view.
    st().setTabs([wf("w1", [src(A, "r0c0_i0")])]);
    let renders = 0;
    const { result } = renderHook(() => {
      if (++renders > 25) throw new Error("render loop");
      const tabs = useAnalysisWorkspaceStore((s) => s.tabs);
      const activeId = useAnalysisWorkspaceStore((s) => s.activeId);
      const active = tabs.find((t) => t.id === activeId) || tabs[0];
      const nodes = active.nodes;
      const setNodes = useCallback((u: (n: Node<NodeData>[]) => Node<NodeData>[]) => {
        useAnalysisWorkspaceStore.getState().setTabNodes(active.id, u);
      }, [active.id]);
      useEffect(() => {
        setNodes((cur) => {
          let changed = false;
          const next = cur.map((x): Node<NodeData> => {
            if (x.data.kind === "source" || x.data.declaredOutputs) return x;
            changed = true;
            return { ...x, data: { ...x.data, declaredOutputs: [{ kind: "plot", name: "plot_1" }] } };
          });
          return changed ? next : cur;
        });
      }, [tabs, setNodes]);
      return { nodes, setNodes };
    });
    expect(result.current.nodes[1].data.declaredOutputs).toEqual([{ kind: "plot", name: "plot_1" }]);
    expect(renders).toBeLessThan(6);
  });

  it("outputs survive the view unmounting and mounting again", () => {
    st().setTabs([wf("w1", [src(A, "r0c0_i0")])]);
    const useNodes = () => {
      const tabs = useAnalysisWorkspaceStore((s) => s.tabs);
      return tabs[0].nodes;
    };
    const first = renderHook(useNodes);
    act(() => {
      st().setTabNodes("w1", (nodes) => nodes.map((x) => (x.id !== "p1" ? x : {
        ...x, data: { ...x.data, outputs: [{ id: "out_table_t__v2", kind: "table", name: "t", payload: "a,b\n1,2", versionLabel: "v2", blobId: "b_2" }] },
      })));
    });
    first.unmount();
    const second = renderHook(useNodes);
    expect(second.result.current[1].data.outputs?.[0].payload).toBe("a,b\n1,2");
    second.unmount();
  });
});

// ── active tab ──────────────────────────────────────────────

describe("setActive", () => {
  it("switches the active tab; the same id is a no-op", () => {
    st().setTabs([wf("w1", []), wf("w2", [])]);
    st().setActive("w2");
    expect(st().activeId).toBe("w2");
    const n = countNotifications();
    st().setActive("w2");
    expect(n.count()).toBe(0);
    n.stop();
  });
});

// ── document uids ───────────────────────────────────────────

describe("uidFor / adoptUid", () => {
  it("uidFor mints once per document and is stable", () => {
    const u = st().uidFor("doc_new");
    expect(u).toMatch(/^du_/);
    expect(st().uidFor("doc_new")).toBe(u);
    expect(st().docUids.doc_new).toBe(u);
  });

  it("adoptUid moves home, measurement tables and tombstones to the file's uid", () => {
    const ref: MeasurementRef = { mpfId: A.id, mpf: "a", label: "a — measurements", csv: "x", docUid: "du_a" };
    const mine: WorkflowTab = { id: "w1", name: "W", nodes: [sourceNode([], [ref])], edges: [], home: "du_a" };
    const other = wf("w2", [src(B, "r0c0_i0")], { home: "du_b" });
    seed({ tabs: [mine, other], tombstones: { du_a: { gone: 5 } } });
    st().adoptUid(A.id, "du_file");
    const s = st();
    expect(s.docUids[A.id]).toBe("du_file");
    expect(s.tabs[0].home).toBe("du_file");
    expect(s.tabs[0].nodes[0].data.measurementRefs?.[0].docUid).toBe("du_file");
    expect(s.tabs[1]).toBe(other);                      // untouched tabs keep identity
    expect(s.tombstones).toEqual({ du_file: { gone: 5 } });
    const n = countNotifications();
    st().adoptUid(A.id, "du_file");                     // already adopted
    expect(n.count()).toBe(0);
    n.stop();
  });
});

// ── deleteWorkflow ──────────────────────────────────────────

describe("deleteWorkflow", () => {
  it("removes the tab, tombstones it in its owner's manifest and activates the left neighbour", () => {
    seed({ tabs: [wf("w1", [src(B, "r0c0_i0")]), wf("w2", [src(A, "r0c0_i0")], { updatedAt: 5_000 }), wf("w3", [])], activeId: "w2" });
    st().deleteWorkflow("w2");
    expect(st().tabs.map((t) => t.id)).toEqual(["w1", "w3"]);
    expect(st().activeId).toBe("w1");
    expect(st().tombstones).toEqual({ du_a: { w2: 10_000 } });
  });

  it("a shared workflow is tombstoned in every open owner; one with a closed owner (locked) can't be deleted", () => {
    seed({ tabs: [wf("w1", [src(A, "r0c0_i0"), src(B, "r0c0_i0")]), wf("w2", [src(A, "r0c1_i0"), closedSrc("du_c", "r0c0_i0")])] });
    expect(st().deleteWorkflow("w1")).toBe(true);
    expect(st().deleteWorkflow("w2")).toBe(false);
    expect(st().tabs.map((t) => t.id)).toEqual(["w2"]);
    expect(st().tombstones).toEqual({ du_a: { w1: 10_000 }, du_b: { w1: 10_000 } });
  });

  it("the stone is never older than the copy it deletes", () => {
    seed({ tabs: [wf("w1", [src(A, "r0c0_i0")], { updatedAt: 99_000 }), wf("w2", [])] });
    st().deleteWorkflow("w1");
    expect(st().tombstones.du_a.w1).toBe(99_000);
  });

  it("a pristine tab leaves no tombstone", () => {
    seed({ tabs: [newWorkflowTab("Workflow 1", "du_a", "w1"), wf("w2", [])] });
    st().deleteWorkflow("w1");
    expect(st().tombstones).toEqual({});
    expect(st().tabs.map((t) => t.id)).toEqual(["w2"]);
  });

  it("deleting the last workflow leaves a fresh pristine one for the active document", () => {
    seed({ tabs: [wf("w1", [src(A, "r0c0_i0")])], activeId: "w1" });
    st().deleteWorkflow("w1");
    const [t] = st().tabs;
    expect(st().tabs).toHaveLength(1);
    expect(t.id).not.toBe("w1");
    expect(t.name).toBe("Workflow 1");
    expect(t.home).toBe("du_a");
    expect(st().activeId).toBe(t.id);
  });

  it("an unknown id changes nothing", () => {
    const before = st().tabs;
    st().deleteWorkflow("nope");
    expect(st().tabs).toBe(before);
  });
});

// ── upload tiles ────────────────────────────────────────────

describe("upload tiles", () => {
  it("newest first; re-uploading a name refreshes its tile", () => {
    st().addUploads([upload("a.tif")]);
    st().addUploads([upload("b.tif"), upload("c.tif")]);
    expect(st().uploads.map((u) => u.name)).toEqual(["b.tif", "c.tif", "a.tif"]);
    st().addUploads([upload("a.tif", "NEW")]);
    expect(st().uploads.map((u) => u.name)).toEqual(["a.tif", "b.tif", "c.tif"]);
    expect(st().uploads[0].thumbnail).toBe("NEW");
  });

  it("removeUpload drops one; an absent key is a no-op", () => {
    st().addUploads([upload("a.tif"), upload("b.tif")]);
    st().removeUpload("img:a.tif");
    expect(st().uploads.map((u) => u.name)).toEqual(["b.tif"]);
    const n = countNotifications();
    st().removeUpload("img:zzz.tif");
    expect(n.count()).toBe(0);
    n.stop();
  });

  it("tiles outlive the view (they are store state, not component state)", () => {
    st().addUploads([upload("a.tif")]);
    const { result, unmount } = renderHook(() => useAnalysisWorkspaceStore((s) => s.uploads));
    unmount();
    const again = renderHook(() => useAnalysisWorkspaceStore((s) => s.uploads));
    expect(again.result.current).toBe(result.current);
    again.unmount();
  });
});

// ── rekeyDoc ────────────────────────────────────────────────

describe("rekeyDoc", () => {
  it("moves a document's keys in every tab and in sourceEditStore", () => {
    const untitled = { id: "doc_u", path: null, name: "Untitled_1" };
    const s0 = src(untitled, "r0c0_i0");
    const keep = src(B, "r0c0_i0");
    const t = wf("w1", [s0, keep]);
    t.nodes[1] = { ...t.nodes[1], data: { ...t.nodes[1].data, ranWith: { [s0.key]: 2, [keep.key]: 1 } } };
    const other = wf("w2", [keep]);
    seed({ tabs: [t, other] });
    useSourceEditStore.setState({ revs: { [s0.key]: 3, [keep.key]: 1 }, withResults: [s0.key, keep.key], rerun: { [s0.key]: true } });

    st().rekeyDoc("d:doc_u", "p:/exp/u.mpf");
    const moved = "p:/exp/u.mpf::r0c0_i0";
    const tab = st().tabs[0];
    expect(tab.nodes[0].data.sources?.map((x) => x.key)).toEqual([moved, keep.key]);
    expect(tab.nodes[0].data.sources?.[0].figKey).toBe("p:/exp/u.mpf");
    expect(tab.nodes[1].data.ranWith).toEqual({ [moved]: 2, [keep.key]: 1 });
    expect(st().tabs[1]).toBe(other);
    const se = useSourceEditStore.getState();
    expect(se.revs).toEqual({ [moved]: 3, [keep.key]: 1 });
    expect(se.withResults).toEqual([moved, keep.key]);
    expect(se.rerun).toEqual({ [moved]: true });
  });

  it("nothing under the old base: no change", () => {
    seed({ tabs: [wf("w1", [src(B, "r0c0_i0")])] });
    const before = st().tabs;
    st().rekeyDoc("d:doc_zz", "p:/exp/zz.mpf");
    expect(st().tabs).toBe(before);
  });
});

// ── debounced sync: updatedAt + home ────────────────────────

describe("debounced sync", () => {
  it("a content change bumps updatedAt after the debounce", () => {
    seed({ tabs: [wf("w1", [src(A, "r0c0_i0")], { updatedAt: 1_000, home: "du_a" })] });
    syncWorkspaceNow();
    expect(st().tabs[0].updatedAt).toBe(1_000);           // read as is: not dirty
    vi.setSystemTime(20_000);
    st().setTabNodes("w1", (nodes) => nodes.map((x) => (x.id === "p1" ? { ...x, data: { ...x.data, code: "x = 2" } } : x)));
    vi.advanceTimersByTime(SYNC_DEBOUNCE_MS - 1);
    expect(st().tabs[0].updatedAt).toBe(1_000);
    vi.advanceTimersByTime(1);
    expect(st().tabs[0].updatedAt).toBe(20_000 + SYNC_DEBOUNCE_MS);   // the clock moved with the timers
    // Settled: further passes find nothing new.
    const after = st().tabs;
    vi.advanceTimersByTime(SYNC_DEBOUNCE_MS * 4);
    syncWorkspaceNow();
    expect(st().tabs).toBe(after);
  });

  it("thumbnail, runKey and selection changes don't count", () => {
    seed({ tabs: [wf("w1", [src(A, "r0c0_i0")], { updatedAt: 1_000, home: "du_a" })] });
    syncWorkspaceNow();
    st().setTabNodes("w1", (nodes) => nodes.map((x) => (x.id !== "source" ? { ...x, selected: true } : {
      ...x, data: { ...x.data, sources: x.data.sources?.map((s) => ({ ...s, thumbnail: "NEWTHUMB", runKey: "doc_a__raw" })) },
    })));
    vi.advanceTimersByTime(SYNC_DEBOUNCE_MS);
    expect(st().tabs[0].updatedAt).toBe(1_000);
  });

  it("first sight: a workflow made this session gets updatedAt; pristine and read-from-file ones don't change", () => {
    seed({ tabs: [
      wf("new", [src(A, "r0c0_i0")], { home: "du_a" }),
      newWorkflowTab("Workflow 2", "du_a", "pristine"),
      wf("fromFile", [src(B, "r0c0_i0")], { updatedAt: 0, home: "du_b" }),
    ] });
    syncWorkspaceNow();
    const [a, b, c] = st().tabs;
    expect(a.updatedAt).toBe(10_000);
    expect(b.updatedAt).toBeUndefined();
    expect(c.updatedAt).toBe(0);
  });

  it("a locked workflow (an owner's document is closed) is never bumped", () => {
    seed({ tabs: [wf("w1", [src(A, "r0c0_i0"), closedSrc("du_c", "r0c0_i0")], { updatedAt: 1_000 })] });
    syncWorkspaceNow();
    // setTabs, not setTabNodes: the graph's own writes to a locked tab are
    // dropped (analysisWorkspaceShared.test.ts) — this is the sync rule.
    st().setTabs((tabs) => tabs.map((t) => ({
      ...t, nodes: t.nodes.map((x) => (x.id === "p1" ? { ...x, data: { ...x.data, code: "y = 1" } } : x)),
    })));
    vi.advanceTimersByTime(SYNC_DEBOUNCE_MS);
    expect(st().tabs[0].nodes[1].data.code).toBe("y = 1");
    expect(st().tabs[0].updatedAt).toBe(1_000);
  });

  it("an Edit Panel revision that makes saved results stale counts as a change", () => {
    const s0 = src(A, "r0c0_i0");
    const t = wf("w1", [s0], { updatedAt: 1_000, home: "du_a" });
    t.nodes[1] = { ...t.nodes[1], data: { ...t.nodes[1].data, ranWith: { [s0.key]: 0 } } };
    seed({ tabs: [t] });
    syncWorkspaceNow();
    useSourceEditStore.getState().bump([s0.key]);
    vi.advanceTimersByTime(SYNC_DEBOUNCE_MS);
    expect(st().tabs[0].updatedAt).toBe(10_000 + SYNC_DEBOUNCE_MS);
  });

  it("home follows the figure data: forced to the only document, kept while it participates", () => {
    seed({ tabs: [wf("w1", [], { home: "du_a", updatedAt: 1 }), wf("w2", [], { home: "du_a", updatedAt: 1 })] });
    syncWorkspaceNow();
    // w1: only b's figure → home b.
    st().setTabNodes("w1", (nodes) => nodes.map((x) => (x.id === "source" ? sourceNode([src(B, "r0c0_i0")]) : x)));
    // w2: a and b → home a is a participant, kept.
    st().setTabNodes("w2", (nodes) => nodes.map((x) => (x.id === "source" ? sourceNode([src(B, "r0c0_i0"), src(A, "r0c0_i0")]) : x)));
    vi.advanceTimersByTime(SYNC_DEBOUNCE_MS);
    expect(st().tabs[0].home).toBe("du_b");
    expect(st().tabs[1].home).toBe("du_a");
  });

  it("a workflow with no figure data and no home is given the active document", () => {
    seed({ tabs: [{ ...wf("w1", [upload("u.tif")]), home: undefined }] });
    syncWorkspaceNow();
    expect(st().tabs[0].home).toBe("du_a");
  });

  it("being given a home is not itself an edit: a later no-op change doesn't bump", () => {
    seed({ tabs: [{ ...wf("w1", [upload("u.tif")]), home: undefined, updatedAt: 1_000 }] });
    syncWorkspaceNow();
    expect(st().tabs[0]).toMatchObject({ home: "du_a", updatedAt: 1_000 });
    st().setTabNodes("w1", (nodes) => nodes.map((x) => ({ ...x, selected: true })));
    vi.advanceTimersByTime(SYNC_DEBOUNCE_MS);
    expect(st().tabs[0].updatedAt).toBe(1_000);
  });

  it("a pass that changes nothing writes nothing", () => {
    seed({ tabs: [wf("w1", [src(A, "r0c0_i0")], { updatedAt: 1_000, home: "du_a" })] });
    syncWorkspaceNow();
    const n = countNotifications();
    syncWorkspaceNow();
    expect(n.count()).toBe(0);
    n.stop();
  });
});

// ── owner chip ──────────────────────────────────────────────

describe("ownerChip", () => {
  const label = (id: string) => ({ doc_a: "a", doc_b: "b" } as Record<string, string>)[id];

  it("single owner: its name; shared: names + 'results not saved'; none: null", () => {
    seed({ knownDocs: { du_c: { name: "c" } } });
    const ctx = workspaceOwnershipCtx();
    expect(ownerChip(wf("w", [src(A, "r0c0_i0")]), ctx, label)).toEqual({ text: "a", shared: false });
    expect(ownerChip(wf("w", [src(A, "r0c0_i0"), src(B, "r0c0_i0")]), ctx, label))
      .toEqual({ text: "a + b · results not saved", shared: true });
    expect(ownerChip(wf("w", [src(B, "r0c0_i0"), closedSrc("du_c", "r0c0_i0")]), ctx, label))
      .toEqual({ text: "b + c · results not saved", shared: true });
    expect(ownerChip(wf("w", [upload("u.tif")], { home: "du_b" }), ctx, label)).toEqual({ text: "b", shared: false });
    expect(ownerChip(wf("w", [upload("u.tif")]), ctx, label)).toBeNull();
  });
});

// ── persistence (wired: see analysisWorkspacePersist.test.ts) ──

describe("persistence actions", () => {
  it("openMissing / locate do nothing for a workflow that misses no document (analysisWorkspaceShared.test.ts)", async () => {
    seed({ tabs: [wf("w1", [src(A, "r0c0_i0")])] });
    const before = st();
    await expect(st().openMissing("w1")).resolves.toBeUndefined();
    await expect(st().locate("w1", "du_c")).resolves.toBeUndefined();
    expect(st().tabs).toBe(before.tabs);
    expect(useCollageStore.getState().openDocs).toHaveLength(2);
  });
});
