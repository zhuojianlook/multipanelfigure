/* ──────────────────────────────────────────────────────────
   Store-level tests for image groups + subgroup dividers:
   the load paths (old .mpf shape must come out unchanged), the
   group/divider actions, and what they sync to the backend (which
   is what ends up serialized into the .mpf).
   ────────────────────────────────────────────────────────── */

import { describe, it, expect, vi, beforeEach } from "vitest";

const mockApi = vi.hoisted(() => ({
  updateConfig: vi.fn(async () => ({})),
  deleteImage: vi.fn(async () => ({})),
  getPreview: vi.fn(async () => ({ image: "p", width: 1, height: 1, format: "png" })),
  loadProject: vi.fn(),
  restoreProject: vi.fn(),
  activateState: vi.fn(),
}));

vi.mock("../api/client", () => ({
  api: mockApi,
  checkHealth: vi.fn(async () => true),
  lastHealthError: "",
  default: vi.fn(),
}));

import { useFigureStore } from "./figureStore";
import type { FigureConfig, ImageGroup } from "../api/types";

/** Smallest config the store will sync: one empty panel. The group code
 *  never looks inside it. */
function makeConfig(imageGroups?: unknown): FigureConfig {
  return {
    rows: 1, cols: 1, spacing: 0.02, output_format: "TIFF", background: "White",
    panels: [[{ image_name: "" } as never]],
    column_labels: [], row_labels: [], column_headers: [], row_headers: [],
    resolution_entries: {}, dpi: 300,
    ...(imageGroups === undefined ? {} : { image_groups: imageGroups as ImageGroup[] }),
  };
}

const names = ["a", "b", "c", "d", "e"];
const thumbs = Object.fromEntries(names.map((n) => [n, `t-${n}`]));

function seed(groups: ImageGroup[]) {
  useFigureStore.setState({
    config: makeConfig(),
    loadedImages: Object.fromEntries(names.map((n) => [n, { name: n, thumbnailB64: "" }])),
    imageGroups: groups,
    unsaved: false,
  });
}

/** The image_groups the last backend sync carried — i.e. what a save writes. */
function lastSyncedGroups(): ImageGroup[] {
  const calls = mockApi.updateConfig.mock.calls as unknown as Array<[FigureConfig]>;
  return calls[calls.length - 1][0].image_groups!;
}

const twoGroups = (): ImageGroup[] => [
  { id: "g1", name: "One", imageNames: ["a", "b", "c"], dividers: [{ id: "d1", name: "Late", imageNames: ["c"] }] },
  { id: "g2", name: "Two", imageNames: ["d"] },
];

beforeEach(() => {
  vi.clearAllMocks();
});

describe("load paths normalize image_groups", () => {
  const oldShape = [
    { id: "g1", name: "Controls", imageNames: ["a", "gone.png", "b"] },
    { id: "g2", name: "Empty", imageNames: [] },
  ];
  // Exactly what loadProject / restoreDoc / restoreDocState used to do inline.
  const oldInline = oldShape.map((g) => ({ ...g, imageNames: g.imageNames.filter((n) => names.includes(n)) }));

  it("loadProject: an old-shape project loads exactly as before (no dividers key appears)", async () => {
    mockApi.loadProject.mockResolvedValue({ config: makeConfig(oldShape), image_names: names, thumbnails: thumbs });
    await useFigureStore.getState().loadProject("/tmp/x.mpf");
    const got = useFigureStore.getState().imageGroups;
    expect(got).toEqual(oldInline);
    expect(got.some((g) => "dividers" in g)).toBe(false);
  });

  it("loadProject: a project saved before groups existed → no groups", async () => {
    mockApi.loadProject.mockResolvedValue({ config: makeConfig(), image_names: names, thumbnails: thumbs });
    await useFigureStore.getState().loadProject("/tmp/x.mpf");
    expect(useFigureStore.getState().imageGroups).toEqual([]);
  });

  it("loadProject: dividers survive; stale names and junk are dropped; never throws", async () => {
    const raw = [
      null,
      { id: "g1", name: "One", imageNames: ["a", "b", "zz"], dividers: [{ id: "d1", name: "Late", imageNames: ["b", "zz"], x: 1 }] },
    ];
    mockApi.loadProject.mockResolvedValue({ config: makeConfig(raw), image_names: names, thumbnails: thumbs });
    await useFigureStore.getState().loadProject("/tmp/x.mpf");
    expect(useFigureStore.getState().imageGroups).toEqual([
      { id: "g1", name: "One", imageNames: ["a", "b"], dividers: [{ id: "d1", name: "Late", imageNames: ["b"], x: 1 }] },
    ]);
  });

  it("restoreDoc and restoreDocState normalize the same way", async () => {
    const raw = [{ id: "g1", name: "One", imageNames: ["a", "zz"], dividers: "bad" }];
    const expected = [{ id: "g1", name: "One", imageNames: ["a"] }];
    mockApi.restoreProject.mockResolvedValue({ config: makeConfig(raw), image_names: names, thumbnails: thumbs });
    await useFigureStore.getState().restoreDoc("blob", { path: null });
    expect(useFigureStore.getState().imageGroups).toEqual(expected);

    useFigureStore.setState({ imageGroups: [] });
    mockApi.activateState.mockResolvedValue({ config: makeConfig(raw), image_names: names, thumbnails: thumbs });
    await useFigureStore.getState().restoreDocState("doc-1", { path: null });
    expect(useFigureStore.getState().imageGroups).toEqual(expected);
  });
});

describe("moveImageToGroup", () => {
  it("2-arg call still works: unsectioned append, marks unsaved, syncs groups", () => {
    seed(twoGroups());
    useFigureStore.getState().moveImageToGroup("e", "g2");
    expect(useFigureStore.getState().imageGroups[1].imageNames).toEqual(["d", "e"]);
    expect(useFigureStore.getState().unsaved).toBe(true);
    expect(lastSyncedGroups()[1].imageNames).toEqual(["d", "e"]);
  });

  it("re-dropping into the same section is a no-op: no dirty flag, no sync", () => {
    seed(twoGroups());
    useFigureStore.getState().moveImageToGroup("a", "g1");
    useFigureStore.getState().moveImageToGroup("c", "g1", { dividerId: "d1" });
    expect(useFigureStore.getState().unsaved).toBe(false);
    expect(mockApi.updateConfig).not.toHaveBeenCalled();
    expect(useFigureStore.getState().imageGroups).toEqual(twoGroups());
  });

  it("cross-group move (inserted before a tile) strips the image from the source divider", () => {
    seed(twoGroups());
    useFigureStore.getState().moveImageToGroup("c", "g2", { dividerId: null, beforeImage: "d" });
    const [g1, g2] = useFigureStore.getState().imageGroups;
    expect(g1).toEqual({ id: "g1", name: "One", imageNames: ["a", "b"], dividers: [{ id: "d1", name: "Late", imageNames: [] }] });
    expect(g2.imageNames).toEqual(["c", "d"]);
  });
});

describe("divider actions", () => {
  it("addGroupDivider (split) → rename → delete, each synced", () => {
    seed(twoGroups());
    const id = useFigureStore.getState().addGroupDivider("g1", { splitBeforeImage: "b" });
    expect(id).not.toBe("");
    expect(useFigureStore.getState().unsaved).toBe(true);
    let g1 = useFigureStore.getState().imageGroups[0];
    expect(g1.dividers!.map((d) => [d.name, d.imageNames])).toEqual([["Subgroup 2", ["b"]], ["Late", ["c"]]]);
    expect(lastSyncedGroups()[0]).toEqual(g1);

    useFigureStore.getState().renameGroupDivider("g1", id, "Day 1");
    g1 = useFigureStore.getState().imageGroups[0];
    expect(g1.dividers![0].name).toBe("Day 1");
    expect(lastSyncedGroups()[0].dividers![0].name).toBe("Day 1");

    useFigureStore.getState().deleteGroupDivider("g1", id);
    g1 = useFigureStore.getState().imageGroups[0];
    expect(g1).toEqual(twoGroups()[0]);   // "b" merged back into the unsectioned area
    expect(mockApi.updateConfig).toHaveBeenCalledTimes(3);
  });

  it("addGroupDivider on a missing group returns '' and changes nothing", () => {
    seed(twoGroups());
    expect(useFigureStore.getState().addGroupDivider("nope")).toBe("");
    expect(mockApi.updateConfig).not.toHaveBeenCalled();
    expect(useFigureStore.getState().unsaved).toBe(false);
  });

  it("moveImageToTimeline strips the image from its divider", () => {
    seed(twoGroups());
    useFigureStore.getState().moveImageToTimeline("c");
    expect(useFigureStore.getState().imageGroups[0]).toEqual({
      id: "g1", name: "One", imageNames: ["a", "b"], dividers: [{ id: "d1", name: "Late", imageNames: [] }],
    });
    expect(useFigureStore.getState().unsaved).toBe(true);
  });

  it("removeImage strips the image from its divider too", async () => {
    seed(twoGroups());
    await useFigureStore.getState().removeImage("c");
    const g1 = useFigureStore.getState().imageGroups[0];
    expect(g1.imageNames).toEqual(["a", "b"]);
    expect(g1.dividers![0].imageNames).toEqual([]);
    expect(useFigureStore.getState().loadedImages.c).toBeUndefined();
  });
});

describe("setResolutionEntries (Measure → Save as scale bar entry)", () => {
  beforeEach(() => { mockApi.updateConfig.mockClear(); });
  it("updates the store's entries and marks the doc unsaved WITHOUT a full-config push", () => {
    seed([]);
    useFigureStore.getState().setResolutionEntries({ "Scope|um": 0.5 });
    const s = useFigureStore.getState();
    expect(s.config!.resolution_entries).toEqual({ "Scope|um": 0.5 });
    expect(s.unsaved).toBe(true);
    // A full PUT would overwrite the backend's copy of the panel the Edit
    // dialog is editing with the store's stale one.
    expect(mockApi.updateConfig).not.toHaveBeenCalled();
  });
});

describe("moveImageToGroup with atEnd", () => {
  beforeEach(() => { mockApi.updateConfig.mockClear(); });
  it("reorders within the section, marks unsaved and syncs", () => {
    seed(twoGroups());
    useFigureStore.getState().moveImageToGroup("a", "g1", { dividerId: null, atEnd: true });
    const g1 = useFigureStore.getState().imageGroups.find((g) => g.id === "g1")!;
    expect(g1.imageNames.indexOf("a")).toBeGreaterThan(g1.imageNames.indexOf("b"));
    expect(useFigureStore.getState().unsaved).toBe(true);
    expect(mockApi.updateConfig).toHaveBeenCalled();
  });
});
