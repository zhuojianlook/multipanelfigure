/* ──────────────────────────────────────────────────────────
   figureStore._pivotGrid — what a Pivot sends to the backend: the
   in-flight writes settle first, then ONE full PUT of the transposed
   config (never patchGrid, whose ensure_grid would trim the extra row
   or column first, and no per-panel PATCH). The thumbnails move with
   their panels, auto letters re-derive by position, and the Analysis
   hook runs before that PUT resolves. Pivoting twice restores the
   figure.
   ────────────────────────────────────────────────────────── */

import { describe, it, expect, vi, beforeEach } from "vitest";

const mockApi = vi.hoisted(() => ({
  updateConfig: vi.fn(async () => ({})),
  patchGrid: vi.fn(async () => ({})),
  patchPanel: vi.fn(async () => ({})),
  getPreview: vi.fn(async () => ({ image: "p", width: 1, height: 1, format: "png" })),
  getPanelPreview: vi.fn(async () => ({ image: "thumb" })),
}));
const settle = vi.hoisted(() => vi.fn(async () => {}));

vi.mock("../api/client", () => ({
  api: mockApi,
  checkHealth: vi.fn(async () => true),
  lastHealthError: "",
  settleConfigWrites: settle,
  default: vi.fn(),
}));

import { useFigureStore } from "./figureStore";
import type { FigureConfig } from "../api/types";
import fixture from "../utils/__fixtures__pivotGrid.json";

const before = fixture.before as unknown as FigureConfig;
const clone = <T,>(x: T): T => JSON.parse(JSON.stringify(x));

type Synced = FigureConfig & { figures?: unknown; active_figure_id?: string };
function lastSync(): Synced {
  const calls = mockApi.updateConfig.mock.calls as unknown as Array<[Synced]>;
  return calls[calls.length - 1][0];
}

/** The fixture's 2×3 figure on screen, saved, with a thumbnail per cell. */
function seed() {
  const thumbs: Record<string, string> = {};
  const sizes: Record<string, { w: number; h: number }> = {};
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 3; c++) {
      thumbs[`${r}-${c}`] = `T${r}${c}`;
      sizes[`${r}-${c}`] = { w: 10 + r, h: 20 + c };
    }
  }
  useFigureStore.setState({
    config: clone(before),
    panelThumbnails: thumbs,
    panelProcessedSizes: sizes,
    unsaved: false,
    configDirty: false,
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  mockApi.updateConfig.mockImplementation(async () => ({}));
  seed();
});

describe("_pivotGrid", () => {
  it("settles the in-flight writes, then PUTs the 3×2 transposed config — no patchGrid / patchPanel", async () => {
    const ok = await useFigureStore.getState()._pivotGrid();
    expect(ok).toBe(true);
    expect(settle).toHaveBeenCalled();
    expect(mockApi.updateConfig).toHaveBeenCalledTimes(1);
    expect(settle.mock.invocationCallOrder[0]).toBeLessThan(mockApi.updateConfig.mock.invocationCallOrder[0]);
    const sent = lastSync();
    expect([sent.rows, sent.cols]).toEqual([3, 2]);
    expect(sent.panels.map((row) => row.map((p) => p.image_name))).toEqual([
      ["r0c0.png", "r1c0.png"],
      ["r0c1.png", "r1c1.png"],
      ["r0c2.png", "r1c2.png"],
    ]);
    expect(sent.column_labels.map((l) => l.text)).toEqual(["R1", "R2"]);
    expect(sent.row_labels.map((l) => l.text)).toEqual(["C1", "C2", "C3"]);
    expect(mockApi.patchGrid).not.toHaveBeenCalled();
    expect(mockApi.patchPanel).not.toHaveBeenCalled();
  });

  it("moves the thumbnails with their panels and marks the document unsaved", async () => {
    await useFigureStore.getState()._pivotGrid();
    const s = useFigureStore.getState();
    expect(s.panelThumbnails).toEqual({ "0-0": "T00", "1-0": "T01", "2-0": "T02", "0-1": "T10", "1-1": "T11", "2-1": "T12" });
    expect(s.panelProcessedSizes["2-1"]).toEqual({ w: 11, h: 22 });
    expect(s.unsaved).toBe(true);
    expect(s.configDirty).toBe(true);
    // Every cell kept its thumbnail: none is re-rendered.
    expect(mockApi.getPanelPreview).not.toHaveBeenCalled();
  });

  it("re-letters the auto letters a…f row-major in the new layout", async () => {
    await useFigureStore.getState()._pivotGrid();
    const cfg = useFigureStore.getState().config!;
    expect(cfg.panels.map((row) => row.map((p) => p.labels[0].text))).toEqual([["a", "b"], ["c", "d"], ["e", "f"]]);
    expect(lastSync().panels.map((row) => row.map((p) => p.labels[0].text))).toEqual([["a", "b"], ["c", "d"], ["e", "f"]]);
  });

  it("calls onPivoted once, with the pivoted config in the store, before the PUT resolves", async () => {
    let release!: () => void;
    mockApi.updateConfig.mockImplementation(() => new Promise((res) => { release = () => res({}); }));
    const seen: number[] = [];
    const onPivoted = vi.fn(() => { seen.push(useFigureStore.getState().config!.rows); });
    const p = useFigureStore.getState()._pivotGrid(onPivoted);
    await vi.waitFor(() => expect(mockApi.updateConfig).toHaveBeenCalled());
    expect(onPivoted).toHaveBeenCalledTimes(1);
    expect(seen).toEqual([3]);
    release();
    await expect(p).resolves.toBe(true);
    expect(onPivoted).toHaveBeenCalledTimes(1);
  });

  it("a formatted auto letter keeps its formatting through the re-lettering, so pivoting back restores it", async () => {
    // Bold red "b", styled in the Edit Panel's Labels field (per-character runs).
    const cfg = clone(useFigureStore.getState().config!);
    cfg.panels[0][1].labels[0].styled_segments = [{ text: "b", font_style: ["Bold"], color: "#FF0000" }] as never;
    useFigureStore.setState({ config: cfg });
    const styled = clone(cfg.panels[0][1].labels[0]);
    await useFigureStore.getState()._pivotGrid();
    const moved = useFigureStore.getState().config!.panels[1][0].labels[0];
    expect(moved.text).toBe("c");
    expect(moved.styled_segments).toEqual([{ text: "c", font_style: ["Bold"], color: "#FF0000" }]);
    await useFigureStore.getState()._pivotGrid();
    expect(useFigureStore.getState().config!.panels[0][1].labels[0]).toEqual(styled);
  });

  it("pivoting twice restores the figure and its thumbnails", async () => {
    const orig = clone(useFigureStore.getState().config);
    const thumbs = { ...useFigureStore.getState().panelThumbnails };
    await useFigureStore.getState()._pivotGrid();
    await useFigureStore.getState()._pivotGrid();
    expect(useFigureStore.getState().config).toEqual(orig);
    expect(useFigureStore.getState().panelThumbnails).toEqual(thumbs);
    expect(lastSync().panels[0].map((p) => p.image_name)).toEqual(["r0c0.png", "r0c1.png", "r0c2.png"]);
  });

  it("a cell without a thumbnail gets one rendered at its new position", async () => {
    const { "0-1": _gone, ...rest } = useFigureStore.getState().panelThumbnails;
    useFigureStore.setState({ panelThumbnails: rest });
    await useFigureStore.getState()._pivotGrid();
    expect(mockApi.getPanelPreview).toHaveBeenCalledTimes(1);
    expect(mockApi.getPanelPreview).toHaveBeenCalledWith(1, 0);
  });

  it("a thumbnail being rendered as the pivot starts is dropped and rendered again where its panel went", async () => {
    let late!: (v: { image: string }) => void;
    mockApi.getPanelPreview.mockImplementationOnce(() => new Promise((res) => { late = res; }));
    const inFlight = useFigureStore.getState().refreshPanelThumbnail(0, 1);   // e.g. right after an edit
    await useFigureStore.getState()._pivotGrid();
    expect(mockApi.getPanelPreview).toHaveBeenLastCalledWith(1, 0);
    await vi.waitFor(() => expect(useFigureStore.getState().panelThumbnails["1-0"]).toBe("thumb"));
    // The old render answers last: it was for (0,1) of the old layout — dropped.
    late({ image: "LATE" });
    await inFlight;
    expect(useFigureStore.getState().panelThumbnails["0-1"]).toBe("T10");
    expect(Object.values(useFigureStore.getState().panelThumbnails)).not.toContain("LATE");
  });

  it("nothing on screen: nothing sent", async () => {
    useFigureStore.setState({ config: null });
    expect(await useFigureStore.getState()._pivotGrid()).toBe(false);
    expect(mockApi.updateConfig).not.toHaveBeenCalled();
  });
});
