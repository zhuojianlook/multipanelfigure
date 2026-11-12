/* ──────────────────────────────────────────────────────────
   Store-level tests for figure tabs: the load paths (an old .mpf
   comes out as one figure), what a switch / add / delete sends to
   the backend (= what a save writes), and that removing an image
   clears it from parked figures too.
   ────────────────────────────────────────────────────────── */

import { describe, it, expect, vi, beforeEach } from "vitest";

const mockApi = vi.hoisted(() => ({
  updateConfig: vi.fn(async () => ({})),
  deleteImage: vi.fn(async () => ({})),
  getPreview: vi.fn(async () => ({ image: "p", width: 1, height: 1, format: "png" })),
  getPanelPreview: vi.fn(async () => ({ image: "thumb" })),
  loadProject: vi.fn(),
  restoreProject: vi.fn(),
  activateState: vi.fn(),
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

function cfg(images: string[], extra: Record<string, unknown> = {}): FigureConfig {
  return {
    rows: 1, cols: images.length, spacing: 0.02, output_format: "TIFF", background: "White",
    panels: [images.map((image_name) => ({ image_name } as never))],
    column_labels: [], row_labels: [], column_headers: [], row_headers: [],
    resolution_entries: {}, dpi: 300,
    ...extra,
  } as FigureConfig;
}

type Synced = FigureConfig & { figures: Array<Record<string, unknown>>; active_figure_id: string };
function lastSync(): Synced {
  const calls = mockApi.updateConfig.mock.calls as unknown as Array<[Synced]>;
  return calls[calls.length - 1][0];
}

const images = ["a.png", "b.png", "c.png"];
const resp = (config: FigureConfig) => ({
  config, image_names: images, thumbnails: Object.fromEntries(images.map((n) => [n, "t"])),
});

beforeEach(() => {
  vi.clearAllMocks();
});

describe("load paths", () => {
  it("an old single-figure project loads as one default figure", async () => {
    mockApi.loadProject.mockResolvedValue(resp(cfg(["a.png"])));
    await useFigureStore.getState().loadProject("/tmp/old.mpf");
    const s = useFigureStore.getState();
    expect(s.figures).toEqual([{ id: "fig-1", name: "Figure 1", config: null }]);
    expect(s.activeFigureId).toBe("fig-1");
    expect(s.config?.panels[0][0].image_name).toBe("a.png");
  });

  it("a multi-figure project restores its tabs; the store config holds no figure keys", async () => {
    const file = cfg(["b.png"], {
      active_figure_id: "fig-x",
      figures: [
        { id: "fig-1", name: "Main", config: cfg(["a.png", "c.png"]) },
        { id: "fig-x", name: "WB" },
      ],
    });
    for (const load of [
      async () => { mockApi.loadProject.mockResolvedValue(resp(file)); await useFigureStore.getState().loadProject("/tmp/m.mpf"); },
      async () => { mockApi.restoreProject.mockResolvedValue(resp(file)); await useFigureStore.getState().restoreDoc("blob", { path: null }); },
      async () => { mockApi.activateState.mockResolvedValue(resp(file)); await useFigureStore.getState().restoreDocState("d1", { path: null }); },
    ]) {
      await load();
      const s = useFigureStore.getState();
      expect(s.figures.map((f) => f.name)).toEqual(["Main", "WB"]);
      expect(s.activeFigureId).toBe("fig-x");
      expect(s.config?.panels[0][0].image_name).toBe("b.png");
      expect(s.config).not.toHaveProperty("figures");
      expect(s.config).not.toHaveProperty("active_figure_id");
    }
  });
});

describe("figure operations", () => {
  async function loadTwo() {
    mockApi.loadProject.mockResolvedValue(resp(cfg(["a.png"], {
      active_figure_id: "fig-1",
      resolution_entries: { scope: 0.5 },
      figures: [
        { id: "fig-1", name: "Main" },
        { id: "fig-x", name: "WB", config: cfg(["b.png", "c.png"]) },
      ],
    })));
    await useFigureStore.getState().loadProject("/tmp/m.mpf");
    vi.clearAllMocks();
  }

  it("switch waits for in-flight writes, then PUTs the target with every figure", async () => {
    await loadTwo();
    const ok = await useFigureStore.getState()._switchFigure("fig-x");
    expect(ok).toBe(true);
    expect(settle).toHaveBeenCalled();
    const sent = lastSync();
    expect(sent.active_figure_id).toBe("fig-x");
    expect(sent.panels[0].map((p) => p.image_name)).toEqual(["b.png", "c.png"]);
    expect(sent.resolution_entries).toEqual({ scope: 0.5 });
    expect(sent.figures.map((f) => f.id)).toEqual(["fig-1", "fig-x"]);
    expect((sent.figures[0].config as FigureConfig).panels[0][0].image_name).toBe("a.png");
    expect(sent.figures[1]).not.toHaveProperty("config");
    // A switch is not an edit.
    expect(useFigureStore.getState().unsaved).toBe(false);
    // The new figure's cells get fresh thumbnails.
    expect(mockApi.getPanelPreview).toHaveBeenCalledTimes(2);
  });

  it("switching to the active or an unknown figure does nothing", async () => {
    await loadTwo();
    expect(await useFigureStore.getState()._switchFigure("fig-1")).toBe(false);
    expect(await useFigureStore.getState()._switchFigure("nope")).toBe(false);
    expect(mockApi.updateConfig).not.toHaveBeenCalled();
  });

  it("add creates a blank active figure and marks the doc unsaved", async () => {
    await loadTwo();
    await useFigureStore.getState()._addFigure();
    const s = useFigureStore.getState();
    expect(s.figures).toHaveLength(3);
    expect(s.activeFigureId).toBe(s.figures[2].id);
    expect(s.figures[2].name).toBe("Figure 3");
    expect(s.config?.panels.flat().every((p) => !p.image_name)).toBe(true);
    expect(s.unsaved).toBe(true);
    expect(lastSync().active_figure_id).toBe(s.figures[2].id);
  });

  it("rename and delete sync the list", async () => {
    await loadTwo();
    await useFigureStore.getState()._renameFigure("fig-x", "Blots");
    expect(lastSync().figures.map((f) => f.name)).toEqual(["Main", "Blots"]);
    await useFigureStore.getState()._deleteFigure("fig-1");
    const s = useFigureStore.getState();
    expect(s.figures.map((f) => f.id)).toEqual(["fig-x"]);
    expect(s.activeFigureId).toBe("fig-x");
    expect(s.config?.panels[0][0].image_name).toBe("b.png");
  });

  it("removeImage clears the image from parked figures too", async () => {
    await loadTwo();
    await useFigureStore.getState().removeImage("b.png");
    expect(mockApi.deleteImage).toHaveBeenCalledWith("b.png");
    const parked = useFigureStore.getState().figures.find((f) => f.id === "fig-x");
    expect(parked?.config?.panels[0].map((p) => p.image_name)).toEqual(["", "c.png"]);
  });

  it("New resets to a single figure", async () => {
    await loadTwo();
    (mockApi as unknown as { listImages: unknown }).listImages = vi.fn(async () => ({ names: [] }));
    (mockApi as unknown as { getResolutions: unknown }).getResolutions = vi.fn(async () => ({}));
    await useFigureStore.getState().newBlankFigure();
    const s = useFigureStore.getState();
    expect(s.figures).toEqual([{ id: "fig-1", name: "Figure 1", config: null }]);
    expect(lastSync().figures).toEqual([{ id: "fig-1", name: "Figure 1" }]);
  });
});
