import { describe, it, expect } from "vitest";
import type { PanelInfo } from "../api/types";
import {
  analysisSourceSigs, changedSourceRawKeys, describeSourceKey,
  upstreamSourceKeys, computeStale, ranWithFor, sourcesWithResults,
  type StaleNodeLike, type StaleEdgeLike,
} from "./analysisStale";

type P = Record<string, unknown>;
const panel = (extra: P = {}): PanelInfo => ({
  image_name: "a.png",
  include_in_analysis: true,
  crop: [0, 0, 100, 100],
  brightness: 1,
  labels: [{ text: "A" }],
  areas: [],
  add_zoom_inset: false,
  zoom_inset: null,
  zoom_insets: [],
  ...extra,
} as unknown as PanelInfo);
const inset = (extra: P = {}) => ({ x: 10, y: 10, width: 20, height: 20, zoom_factor: 2, include_in_analysis: true, rectangle_color: "red", ...extra });
const area = (extra: P = {}) => ({ shape: "Rectangle", points: [[0, 0], [10, 10]], include_in_analysis: true, color: "red", ...extra });

describe("analysisSourceSigs — one entry per source the sidecar exposes", () => {
  it("uses the sidecar's keys for panel, area and inset sources", () => {
    const p = panel({ areas: [area(), area({ include_in_analysis: false })], add_zoom_inset: true, zoom_insets: [inset(), inset()] });
    expect(Object.keys(analysisSourceSigs(p, 1, 2)).sort())
      .toEqual(["r1c2_area0", "r1c2_i0", "r1c2_i1", "r1c2_panel"]);
  });
  it("omits unflagged and image-less panels", () => {
    expect(analysisSourceSigs(panel({ include_in_analysis: false }), 0, 0)).toEqual({});
    expect(analysisSourceSigs(panel({ image_name: "" }), 0, 0)).toEqual({});
  });
  it("has no inset sources unless add_zoom_inset, and falls back to the legacy single inset", () => {
    expect(Object.keys(analysisSourceSigs(panel({ include_in_analysis: false, zoom_insets: [inset()] }), 0, 0))).toEqual([]);
    expect(Object.keys(analysisSourceSigs(panel({ include_in_analysis: false, add_zoom_inset: true, zoom_inset: inset() }), 0, 0)))
      .toEqual(["r0c0_i0"]);
  });
});

describe("changedSourceRawKeys — which sources an edit changes the pixels of", () => {
  const full = (extra: P = {}) => panel({ areas: [area()], add_zoom_inset: true, zoom_insets: [inset(), inset({ x: 50 })], ...extra });
  it("a crop, rotation, adjustment or frame change marks every source on the panel", () => {
    for (const edit of [{ crop: [5, 5, 90, 90] }, { rotation: 90 }, { brightness: 1.4 }, { frame: 3 }, { input_white_g: 200 }]) {
      expect(changedSourceRawKeys(full(), full(edit), 0, 0).sort())
        .toEqual(["r0c0_area0", "r0c0_i0", "r0c0_i1", "r0c0_panel"]);
    }
  });
  it("labels, scale bar and inset / area styling change nothing", () => {
    const after = full({
      labels: [{ text: "B" }], add_scale_bar: true, scale_bar: { micron_per_pixel: 0.2 },
      zoom_insets: [inset({ rectangle_color: "blue" }), inset({ x: 50, rectangle_color: "blue" })],
      areas: [area({ color: "blue" })],
    });
    expect(changedSourceRawKeys(full(), after, 0, 0)).toEqual([]);
  });
  it("moving one inset or reshaping one area marks only that source", () => {
    expect(changedSourceRawKeys(full(), full({ zoom_insets: [inset({ x: 11 }), inset({ x: 50 })] }), 0, 0)).toEqual(["r0c0_i0"]);
    expect(changedSourceRawKeys(full(), full({ areas: [area({ points: [[0, 0], [20, 20]] })] }), 0, 0)).toEqual(["r0c0_area0"]);
  });
  it("a missing field equals its default, and legacy black/white points equal the migrated ones", () => {
    expect(changedSourceRawKeys(panel(), panel({ gamma: 1, hue: 0, crop_image: true }), 0, 0)).toEqual([]);
    expect(changedSourceRawKeys(panel({ input_white: 200 }),
      panel({ input_white_r: 200, input_white_g: 200, input_white_b: 200 }), 0, 0)).toEqual([]);
  });
  it("a newly flagged or unflagged source is not a change", () => {
    expect(changedSourceRawKeys(panel({ include_in_analysis: false }), panel({ crop: [1, 1, 5, 5] }), 0, 0)).toEqual([]);
    expect(changedSourceRawKeys(panel(), panel({ include_in_analysis: false, crop: [1, 1, 5, 5] }), 0, 0)).toEqual([]);
  });
});

describe("describeSourceKey", () => {
  it("reads like the grid", () => {
    expect(describeSourceKey("p:/x.mpf#fig-2::r0c1_i0")).toBe("R1C2 inset 1");
    expect(describeSourceKey("r2c0_panel")).toBe("R3C1 panel");
    expect(describeSourceKey("d:doc_1::r0c0_area1")).toBe("R1C1 area 2");
  });
});

// A two-step pipeline: source S (two images) → A → B, plus a measurements-only
// edge S → M, and an unrelated node U fed by source T.
const S = (keys: string[]): StaleNodeLike => ({ id: "S", data: { kind: "source", sources: keys.map((key) => ({ key })) } });
function graph(outputs = true, ranWith?: Record<string, Record<string, number>>) {
  const proc = (id: string): StaleNodeLike => ({
    id, data: { kind: "python", outputs: outputs ? [{}] : [], ranWith: ranWith?.[id] },
  });
  const nodes: StaleNodeLike[] = [
    S(["k1", "k2"]), proc("A"), proc("B"), proc("M"),
    { id: "T", data: { kind: "source", sources: [{ key: "k3" }] } }, proc("U"),
  ];
  const edges: StaleEdgeLike[] = [
    { source: "S", target: "A", sourceHandle: "out_image_0" },
    { source: "A", target: "B", sourceHandle: "out_table_x" },
    { source: "S", target: "M", sourceHandle: "out_table_measurements" },
    { source: "T", target: "U", sourceHandle: "out_image_0" },
  ];
  return { nodes, edges };
}

describe("upstreamSourceKeys", () => {
  it("one image edge carries every image on the source node, transitively", () => {
    const { nodes, edges } = graph();
    expect(upstreamSourceKeys("A", nodes, edges).sort()).toEqual(["k1", "k2"]);
    expect(upstreamSourceKeys("B", nodes, edges).sort()).toEqual(["k1", "k2"]);
  });
  it("a measurements edge carries no image", () => {
    const { nodes, edges } = graph();
    expect(upstreamSourceKeys("M", nodes, edges)).toEqual([]);
  });
});

describe("computeStale", () => {
  it("nothing is stale before any edit", () => {
    const { nodes, edges } = graph();
    expect(computeStale(nodes, edges, {}).nodeIds.size).toBe(0);
  });
  it("an edit stales every node downstream of that image — and only those", () => {
    const { nodes, edges } = graph();
    const st = computeStale(nodes, edges, { k2: 1 });
    expect([...st.nodeIds].sort()).toEqual(["A", "B"]);
    expect([...st.sourceKeys]).toEqual(["k2"]);
  });
  it("a node without results is never stale", () => {
    const { nodes, edges } = graph(false);
    expect(computeStale(nodes, edges, { k1: 3 }).nodeIds.size).toBe(0);
  });
  it("re-running a node records the new revisions and clears it; its downstream stays stale", () => {
    const g = graph();
    const revs = { k1: 2 };
    const ranA = ranWithFor("A", g.nodes, g.edges, revs);
    expect(ranA).toEqual({ k1: 2, k2: 0 });
    const after = graph(true, { A: ranA });
    expect([...computeStale(after.nodes, after.edges, revs).nodeIds]).toEqual(["B"]);
    const all = graph(true, { A: ranA, B: ranWithFor("B", g.nodes, g.edges, revs) });
    expect(computeStale(all.nodes, all.edges, revs).nodeIds.size).toBe(0);
  });
  it("the same image in two workflows clears separately", () => {
    const one = graph(true, { A: { k1: 1, k2: 0 } });
    const two = graph(true);
    const revs = { k1: 1 };
    expect(computeStale(one.nodes, one.edges, revs).nodeIds.has("A")).toBe(false);
    expect(computeStale(two.nodes, two.edges, revs).nodeIds.has("A")).toBe(true);
  });
});

describe("sourcesWithResults", () => {
  it("lists images feeding a node with results, across tabs", () => {
    expect(sourcesWithResults([graph(), { nodes: [S(["k9"])], edges: [] }])).toEqual(["k1", "k2", "k3"]);
    expect(sourcesWithResults([graph(false)])).toEqual([]);
  });
});
