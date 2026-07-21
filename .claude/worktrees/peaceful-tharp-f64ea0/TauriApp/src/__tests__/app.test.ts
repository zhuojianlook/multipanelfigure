/* ──────────────────────────────────────────────────────────
   Tests for the Multi-Panel Figure TauriApp.
   Covers: API endpoints, store actions, spanning header logic.
   ────────────────────────────────────────────────────────── */

import { describe, it, expect, vi, beforeEach } from "vitest";

// ── Mock the api client before importing store ───────────
// Use vi.hoisted so the mock is available at hoist time

const mockApi = vi.hoisted(() => ({
  getConfig: vi.fn(),
  updateConfig: vi.fn(),
  patchGrid: vi.fn(),
  patchPanel: vi.fn(),
  patchColumnLabels: vi.fn(),
  patchRowLabels: vi.fn(),
  patchColumnHeaders: vi.fn(),
  patchRowHeaders: vi.fn(),
  patchBackground: vi.fn(),
  uploadImages: vi.fn(),
  deleteImage: vi.fn(),
  listImages: vi.fn(),
  getImageThumbnail: vi.fn(),
  getPanelPreview: vi.fn(),
  getPreview: vi.fn(),
  saveProject: vi.fn(),
  loadProject: vi.fn(),
  saveFigure: vi.fn(),
  listFonts: vi.fn(),
  getResolutions: vi.fn(),
  updateResolutions: vi.fn(),
}));

vi.mock("../api/client", () => ({
  api: mockApi,
  default: vi.fn(),
}));

import { useFigureStore } from "../store/figureStore";
import type { FigureConfig, HeaderLevel } from "../api/types";

// ── Helper: build a minimal config ───────────────────────

function makeConfig(rows = 2, cols = 3): FigureConfig {
  const panels = Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => ({
      image_name: "",
      crop_image: true,
      aspect_ratio_str: "",
      crop_offset_x: 0,
      crop_offset_y: 0,
      crop: null,
      final_resize: false,
      final_width: 400,
      final_height: 300,
      brightness: 1.0,
      contrast: 1.0,
      hue: 0.0,
      labels: [],
      scale_bar: null,
      add_scale_bar: false,
      symbols: [],
      lines: [],
      areas: [],
      zoom_inset: null,
      add_zoom_inset: false,
      zoom_insets: [],
      rotation: 0,
      flip_horizontal: false,
      flip_vertical: false,
      saturation: 1.0,
      gamma: 1.0,
      color_temperature: 0,
      tint: 0,
      sharpen: 0,
      blur: 0,
      denoise: 0,
      exposure: 0,
      vibrance: 0,
      highlights: 0,
      shadows: 0,
      midtones: 0,
      input_black_r: 0, input_white_r: 255,
      input_black_g: 0, input_white_g: 255,
      input_black_b: 0, input_white_b: 255,
      invert: false,
      grayscale: false,
      pseudocolor: "",
    })),
  );

  return {
    rows,
    cols,
    spacing: 0.02,
    output_format: "TIFF",
    background: "White",
    panels,
    column_labels: Array.from({ length: cols }, (_, i) => ({
      text: `Column ${i + 1}`,
      font_size: 10,
      font_name: "arial.ttf",
      font_path: null,
      font_style: [],
      default_color: "#000000",
      distance: 0.025,
      position: "Top",
      rotation: 0,
      styled_segments: [],
      visible: true,
    })),
    row_labels: Array.from({ length: rows }, (_, i) => ({
      text: `Row ${i + 1}`,
      font_size: 10,
      font_name: "arial.ttf",
      font_path: null,
      font_style: [],
      default_color: "#000000",
      distance: 0.025,
      position: "Left",
      rotation: 90,
      styled_segments: [],
      visible: true,
    })),
    column_headers: [],
    row_headers: [],
    resolution_entries: {},
    dpi: 300,
  };
}

function makeHeaderGroup(overrides: Partial<import("../api/types").HeaderGroup> = {}): import("../api/types").HeaderGroup {
  return {
    text: "Header",
    columns_or_rows: [],
    font_size: 10,
    font_name: "arial.ttf",
    font_path: null,
    font_style: [],
    default_color: "#000000",
    distance: 0.05,
    position: "Top",
    rotation: 0,
    line_color: "#000000",
    line_width: 1.0,
    line_style: "solid",
    line_length: 1.0,
    styled_segments: [],
    ...overrides,
  };
}

function resetStore() {
  useFigureStore.setState({
    config: null,
    loadedImages: {},
    fonts: [],
    previewImageB64: null,
    previewLoading: false,
    configDirty: false,
    drawerPanels: [],
  });
}

// ──────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────

describe("API endpoint integration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("fetchConfig sets config from API", async () => {
    const cfg = makeConfig(2, 2);
    mockApi.getConfig.mockResolvedValue(cfg);

    await useFigureStore.getState().fetchConfig();

    expect(mockApi.getConfig).toHaveBeenCalledTimes(1);
    expect(useFigureStore.getState().config).toEqual(cfg);
  });

  it("fetchConfig falls back to default on error", async () => {
    mockApi.getConfig.mockRejectedValue(new Error("Network error"));

    await useFigureStore.getState().fetchConfig();

    const cfg = useFigureStore.getState().config;
    expect(cfg).not.toBeNull();
    expect(cfg!.rows).toBe(2);
    expect(cfg!.cols).toBe(2);
  });

  it("fetchImages populates loadedImages", async () => {
    mockApi.listImages.mockResolvedValue({ names: ["a.tif", "b.png"], used: [] });
    mockApi.getImageThumbnail
      .mockResolvedValueOnce({ thumbnail: "base64_a" })
      .mockResolvedValueOnce({ thumbnail: "base64_b" });

    await useFigureStore.getState().fetchImages();

    const imgs = useFigureStore.getState().loadedImages;
    expect(Object.keys(imgs)).toEqual(["a.tif", "b.png"]);
    expect(imgs["a.tif"].thumbnailB64).toBe("base64_a");
  });

  it("uploadImages calls API and stores images", async () => {
    mockApi.uploadImages.mockResolvedValue({
      names: ["new.tif"],
      thumbnails: { "new.tif": "thumb_data" },
    });

    useFigureStore.setState({ config: makeConfig() });
    mockApi.getPreview.mockResolvedValue({ image: "prev", width: 100, height: 100, format: "png" });

    const fakeFile = new File([""], "new.tif", { type: "image/tiff" });
    await useFigureStore.getState().uploadImages([fakeFile]);

    expect(mockApi.uploadImages).toHaveBeenCalledTimes(1);
    const imgs = useFigureStore.getState().loadedImages;
    expect(imgs["new.tif"]).toBeDefined();
    expect(imgs["new.tif"].thumbnailB64).toBe("thumb_data");
  });

  it("saveFigure syncs config then calls API", async () => {
    const cfg = makeConfig();
    useFigureStore.setState({ config: cfg });
    mockApi.updateConfig.mockResolvedValue(cfg);
    mockApi.saveFigure.mockResolvedValue({ ok: true, path: "/tmp/fig.tiff" });

    await useFigureStore.getState().saveFigure("/tmp/fig.tiff", "TIFF", "White", 300);

    expect(mockApi.updateConfig).toHaveBeenCalledTimes(1);
    expect(mockApi.saveFigure).toHaveBeenCalledWith("/tmp/fig.tiff", "TIFF", "White", 300);
  });

  it("saveProject calls sync then API", async () => {
    const cfg = makeConfig();
    useFigureStore.setState({ config: cfg });
    mockApi.updateConfig.mockResolvedValue(cfg);
    mockApi.saveProject.mockResolvedValue({ ok: true });

    await useFigureStore.getState().saveProject("/tmp/test.mpfig");

    expect(mockApi.saveProject).toHaveBeenCalledWith("/tmp/test.mpfig");
  });

  it("loadProject restores config and images", async () => {
    const cfg = makeConfig(3, 3);
    mockApi.loadProject.mockResolvedValue({
      config: cfg,
      image_names: ["x.tif"],
      thumbnails: { "x.tif": "xthumb" },
    });
    mockApi.getPreview.mockResolvedValue({ image: "prev", width: 100, height: 100, format: "png" });

    await useFigureStore.getState().loadProject("/tmp/test.mpfig");

    expect(useFigureStore.getState().config).toEqual(cfg);
    expect(useFigureStore.getState().loadedImages["x.tif"].thumbnailB64).toBe("xthumb");
  });
});

describe("Store actions: updateGridSize", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("updates config via API and triggers preview", async () => {
    const cfg = makeConfig(2, 2);
    useFigureStore.setState({ config: cfg });
    const newCfg = makeConfig(3, 4);
    mockApi.patchGrid.mockResolvedValue(newCfg);
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    await useFigureStore.getState().updateGridSize(3, 4);

    expect(mockApi.patchGrid).toHaveBeenCalledWith(3, 4, 0.02);
    expect(useFigureStore.getState().config!.rows).toBe(3);
    expect(useFigureStore.getState().config!.cols).toBe(4);
  });
});

describe("Store actions: setPanelImage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("sets image_name on the correct panel", () => {
    const cfg = makeConfig(2, 3);
    useFigureStore.setState({ config: cfg });
    mockApi.patchPanel.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().setPanelImage(0, 1, "test.tif");

    expect(useFigureStore.getState().config!.panels[0][1].image_name).toBe("test.tif");
    expect(mockApi.patchPanel).toHaveBeenCalledWith(0, 1, expect.objectContaining({ image_name: "test.tif" }));
  });
});

describe("Store actions: addColumnHeaderLevel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("adds an empty header level with no headers", () => {
    const cfg = makeConfig(2, 3);
    useFigureStore.setState({ config: cfg });
    mockApi.patchColumnHeaders.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().addColumnHeaderLevel();

    const headers = useFigureStore.getState().config!.column_headers;
    expect(headers.length).toBe(1);
    expect(headers[0].headers.length).toBe(0);
  });
});

describe("Store actions: createHeaderGroupAt", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("creates a single-cell header group at a given position", () => {
    const cfg = makeConfig(2, 3);
    useFigureStore.setState({ config: cfg });
    mockApi.patchColumnHeaders.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    // First add an empty level
    useFigureStore.getState().addColumnHeaderLevel();
    expect(useFigureStore.getState().config!.column_headers[0].headers.length).toBe(0);

    // Then create a header at column 1
    useFigureStore.getState().createHeaderGroupAt("col", 0, 1);

    const headers = useFigureStore.getState().config!.column_headers;
    expect(headers[0].headers.length).toBe(1);
    expect(headers[0].headers[0].columns_or_rows).toEqual([1]);
    expect(headers[0].headers[0].text).toBe("");
  });

  it("does not create a header at an already-covered position", () => {
    const cfg = makeConfig(2, 3);
    useFigureStore.setState({ config: cfg });
    mockApi.patchColumnHeaders.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().addColumnHeaderLevel();
    useFigureStore.getState().createHeaderGroupAt("col", 0, 1);
    useFigureStore.getState().createHeaderGroupAt("col", 0, 1); // duplicate

    expect(useFigureStore.getState().config!.column_headers[0].headers.length).toBe(1);
  });
});

describe("Store actions: resizeHeaderGroup", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("resizes a header group to new contiguous indices", () => {
    const cfg = makeConfig(2, 4);
    cfg.column_headers = [{
      headers: [makeHeaderGroup({ text: "A", columns_or_rows: [1] })],
    }];
    useFigureStore.setState({ config: cfg });
    mockApi.patchColumnHeaders.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().resizeHeaderGroup("col", 0, 0, [0, 1, 2]);

    const group = useFigureStore.getState().config!.column_headers[0].headers[0];
    expect(group.columns_or_rows).toEqual([0, 1, 2]);
  });

  it("rejects non-contiguous indices", () => {
    const cfg = makeConfig(2, 4);
    cfg.column_headers = [{
      headers: [makeHeaderGroup({ text: "A", columns_or_rows: [1] })],
    }];
    useFigureStore.setState({ config: cfg });
    mockApi.patchColumnHeaders.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().resizeHeaderGroup("col", 0, 0, [0, 2]); // non-contiguous

    const group = useFigureStore.getState().config!.column_headers[0].headers[0];
    expect(group.columns_or_rows).toEqual([1]); // unchanged
  });
});

describe("Store actions: spanning header manipulation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  function setupWithHeaders() {
    const cfg = makeConfig(2, 4);
    const level: HeaderLevel = {
      headers: [
        makeHeaderGroup({ text: "Group A", columns_or_rows: [0, 1] }),
        makeHeaderGroup({ text: "Group B", columns_or_rows: [2] }),
      ],
    };
    cfg.column_headers = [level];
    useFigureStore.setState({ config: cfg });
    mockApi.patchColumnHeaders.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });
  }

  it("extendHeaderGroup extends right into uncovered column", () => {
    setupWithHeaders();

    useFigureStore.getState().extendHeaderGroup("col", 0, 1, "right");

    const group = useFigureStore.getState().config!.column_headers[0].headers[1];
    expect(group.columns_or_rows).toContain(3);
    expect(group.columns_or_rows.length).toBe(2);
  });

  it("extendHeaderGroup does not extend into occupied column", () => {
    setupWithHeaders();

    useFigureStore.getState().extendHeaderGroup("col", 0, 1, "left");

    const group = useFigureStore.getState().config!.column_headers[0].headers[1];
    expect(group.columns_or_rows).toEqual([2]);
  });

  it("splitHeaderGroup breaks a spanning group into individuals", () => {
    setupWithHeaders();

    useFigureStore.getState().splitHeaderGroup("col", 0, 0);

    const headers = useFigureStore.getState().config!.column_headers[0].headers;
    expect(headers.length).toBe(3);
    expect(headers[0].columns_or_rows).toEqual([0]);
    expect(headers[1].columns_or_rows).toEqual([1]);
    expect(headers[2].text).toBe("Group B");
  });

  it("removeHeaderGroup removes a single group", () => {
    setupWithHeaders();

    useFigureStore.getState().removeHeaderGroup("col", 0, 0);

    const headers = useFigureStore.getState().config!.column_headers[0].headers;
    expect(headers.length).toBe(1);
    expect(headers[0].text).toBe("Group B");
  });

  it("removeHeaderGroup removes the entire level if last group removed", () => {
    setupWithHeaders();

    useFigureStore.getState().removeHeaderGroup("col", 0, 0);
    useFigureStore.getState().removeHeaderGroup("col", 0, 0);

    expect(useFigureStore.getState().config!.column_headers.length).toBe(0);
  });

  it("addHeaderGroupAt adds a group at an uncovered index", () => {
    setupWithHeaders();

    useFigureStore.getState().addHeaderGroupAt("col", 0, 3);

    const headers = useFigureStore.getState().config!.column_headers[0].headers;
    expect(headers.length).toBe(3);
    const newGroup = headers.find((g) => g.columns_or_rows.includes(3));
    expect(newGroup).toBeDefined();
  });

  it("addHeaderGroupAt does not add at an already-covered index", () => {
    setupWithHeaders();

    useFigureStore.getState().addHeaderGroupAt("col", 0, 0);

    const headers = useFigureStore.getState().config!.column_headers[0].headers;
    expect(headers.length).toBe(2);
  });
});

describe("Store actions: checkGridResizeConflict", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("returns empty array when no images would be lost", () => {
    const cfg = makeConfig(3, 3);
    useFigureStore.setState({ config: cfg });

    const conflicts = useFigureStore.getState().checkGridResizeConflict(2, 2);
    expect(conflicts).toEqual([]);
  });

  it("returns conflict list when images are in to-be-removed cells", () => {
    const cfg = makeConfig(3, 3);
    cfg.panels[2][2].image_name = "corner.tif";
    cfg.panels[0][2].image_name = "top_right.tif";
    useFigureStore.setState({ config: cfg });

    const conflicts = useFigureStore.getState().checkGridResizeConflict(2, 2);
    expect(conflicts).toContain("R1C3: top_right.tif");
    expect(conflicts).toContain("R3C3: corner.tif");
    expect(conflicts.length).toBe(2);
  });
});

describe("Store actions: updatePanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore();
  });

  it("patches panel properties", () => {
    const cfg = makeConfig(2, 2);
    useFigureStore.setState({ config: cfg });
    mockApi.patchPanel.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().updatePanel(0, 0, { brightness: 1.5, contrast: 0.8 });

    const panel = useFigureStore.getState().config!.panels[0][0];
    expect(panel.brightness).toBe(1.5);
    expect(panel.contrast).toBe(0.8);
  });
});

// ── Phase 1-2 Feature Tests ─────────────────────────────────

describe("New feature: DPI setting", () => {
  beforeEach(() => { vi.clearAllMocks(); resetStore(); });

  it("default config has dpi=300", () => {
    const cfg = makeConfig();
    expect(cfg.dpi).toBe(300);
  });

  it("saveFigure passes DPI to API", async () => {
    const cfg = makeConfig();
    useFigureStore.setState({ config: cfg });
    mockApi.updateConfig.mockResolvedValue(cfg);
    mockApi.saveFigure.mockResolvedValue({ ok: true, path: "/tmp/fig.tiff" });

    await useFigureStore.getState().saveFigure("/tmp/fig.tiff", "TIFF", "White", 600);

    expect(mockApi.saveFigure).toHaveBeenCalledWith("/tmp/fig.tiff", "TIFF", "White", 600);
  });
});

describe("New feature: auto panel labels", () => {
  beforeEach(() => { vi.clearAllMocks(); resetStore(); });

  it("setPanelImage creates auto-label with linked_to_header=true and non-bold", () => {
    const cfg = makeConfig(2, 2);
    useFigureStore.setState({ config: cfg, loadedImages: {} });
    mockApi.patchPanel.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().setPanelImage(0, 0, "test.tif");

    const panel = useFigureStore.getState().config!.panels[0][0];
    expect(panel.labels.length).toBe(1);
    expect(panel.labels[0].text).toBe("a");
    expect(panel.labels[0].font_style).toEqual([]); // NOT bold
    expect(panel.labels[0].position_preset).toBe("Top-Left");
    expect(panel.labels[0].linked_to_header).toBe(true);
    expect(panel.labels[0].styled_segments).toEqual([]);
  });

  it("auto-labels generate sequential letters", () => {
    const cfg = makeConfig(2, 3);
    useFigureStore.setState({ config: cfg, loadedImages: {} });
    mockApi.patchPanel.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().setPanelImage(0, 0, "a.tif");
    useFigureStore.getState().setPanelImage(0, 1, "b.tif");
    useFigureStore.getState().setPanelImage(0, 2, "c.tif");
    useFigureStore.getState().setPanelImage(1, 0, "d.tif");

    const panels = useFigureStore.getState().config!.panels;
    expect(panels[0][0].labels[0].text).toBe("a");
    expect(panels[0][1].labels[0].text).toBe("b");
    expect(panels[0][2].labels[0].text).toBe("c");
    expect(panels[1][0].labels[0].text).toBe("d");
  });
});

describe("New feature: panel swap and drawer", () => {
  beforeEach(() => { vi.clearAllMocks(); resetStore(); });

  it("swapPanels exchanges two panels", () => {
    const cfg = makeConfig(2, 2);
    cfg.panels[0][0].image_name = "img1.tif";
    cfg.panels[1][1].image_name = "img2.tif";
    useFigureStore.setState({ config: cfg });
    mockApi.patchPanel.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().swapPanels(0, 0, 1, 1);

    const panels = useFigureStore.getState().config!.panels;
    expect(panels[0][0].image_name).toBe("img2.tif");
    expect(panels[1][1].image_name).toBe("img1.tif");
  });

  it("movePanelToDrawer moves panel out and replaces with empty", () => {
    const cfg = makeConfig(2, 2);
    cfg.panels[0][0].image_name = "img1.tif";
    cfg.panels[0][0].brightness = 1.5;
    useFigureStore.setState({ config: cfg });
    mockApi.patchPanel.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().movePanelToDrawer(0, 0, 0);

    expect(useFigureStore.getState().config!.panels[0][0].image_name).toBe("");
    expect(useFigureStore.getState().drawerPanels[0].image_name).toBe("img1.tif");
    expect(useFigureStore.getState().drawerPanels[0].brightness).toBe(1.5);
  });
});

describe("New feature: resolution entries for scale bars", () => {
  beforeEach(() => { vi.clearAllMocks(); resetStore(); });

  it("config stores resolution_entries", () => {
    const cfg = makeConfig();
    cfg.resolution_entries = { "Microscope 1": 0.5, "Microscope 2": 1.0 };
    useFigureStore.setState({ config: cfg });

    const stored = useFigureStore.getState().config!.resolution_entries;
    expect(stored["Microscope 1"]).toBe(0.5);
    expect(stored["Microscope 2"]).toBe(1.0);
  });
});

describe("New feature: label formatting and position", () => {
  beforeEach(() => { vi.clearAllMocks(); resetStore(); });

  it("updateLabelFormatting changes font_style", () => {
    const cfg = makeConfig(2, 2);
    useFigureStore.setState({ config: cfg });
    mockApi.patchColumnLabels.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().updateLabelFormatting("col", 0, { font_style: ["Bold", "Italic"] });

    const lbl = useFigureStore.getState().config!.column_labels[0];
    expect(lbl.font_style).toEqual(["Bold", "Italic"]);
  });

  it("updateLabelFormatting changes position", () => {
    const cfg = makeConfig(2, 2);
    useFigureStore.setState({ config: cfg });
    mockApi.patchColumnLabels.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().updateLabelFormatting("col", 0, { position: "Bottom" });

    const lbl = useFigureStore.getState().config!.column_labels[0];
    expect(lbl.position).toBe("Bottom");
  });

  it("updateLabelFormatting changes row label position Left to Right", () => {
    const cfg = makeConfig(2, 2);
    useFigureStore.setState({ config: cfg });
    mockApi.patchRowLabels.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().updateLabelFormatting("row", 0, { position: "Right" });

    const lbl = useFigureStore.getState().config!.row_labels[0];
    expect(lbl.position).toBe("Right");
  });
});

describe("New feature: header group formatting", () => {
  beforeEach(() => { vi.clearAllMocks(); resetStore(); });

  it("updateHeaderGroupFormatting changes font_style to include Superscript", () => {
    const cfg = makeConfig(2, 3);
    cfg.column_headers = [{
      headers: [makeHeaderGroup({ text: "Test Header", columns_or_rows: [0, 1], font_size: 12, font_style: ["Bold"] })],
    }];
    useFigureStore.setState({ config: cfg });
    mockApi.patchColumnHeaders.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().updateHeaderGroupFormatting("col", 0, 0, {
      font_style: ["Bold", "Superscript"],
    });

    const group = useFigureStore.getState().config!.column_headers[0].headers[0];
    expect(group.font_style).toContain("Superscript");
    expect(group.font_style).toContain("Bold");
  });
});

describe("New feature: panel lines and areas", () => {
  beforeEach(() => { vi.clearAllMocks(); resetStore(); });

  it("panel has lines and areas arrays", () => {
    const cfg = makeConfig(2, 2);
    expect(cfg.panels[0][0].lines).toEqual([]);
    expect(cfg.panels[0][0].areas).toEqual([]);
  });

  it("updatePanel can add lines", () => {
    const cfg = makeConfig(2, 2);
    useFigureStore.setState({ config: cfg });
    mockApi.patchPanel.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().updatePanel(0, 0, {
      lines: [{
        name: "Line 1",
        points: [[10, 10], [90, 90]],
        color: "#FFFF00",
        width: 2,
        dash_style: "solid",
        is_curved: false,
        show_measure: false,
        measure_text: "",
        measure_font_size: 12,
        measure_color: "#FFFF00",
        measure_font_name: "arial.ttf",
        measure_styled_segments: [],
      }],
    } as any);

    const panel = useFigureStore.getState().config!.panels[0][0];
    expect((panel as any).lines.length).toBe(1);
    expect((panel as any).lines[0].name).toBe("Line 1");
  });

  it("updatePanel can add areas", () => {
    const cfg = makeConfig(2, 2);
    useFigureStore.setState({ config: cfg });
    mockApi.patchPanel.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().updatePanel(0, 0, {
      areas: [{
        name: "Area 1",
        shape: "Rectangle",
        points: [[50, 50], [20, 20]],
        color: "#FF000040",
        border_color: "#FF0000",
        border_width: 1,
        show_measure: true,
        measure_text: "150 um²",
        measure_font_size: 12,
        measure_color: "#FFFF00",
        measure_font_name: "arial.ttf",
        measure_styled_segments: [],
      }],
    } as any);

    const panel = useFigureStore.getState().config!.panels[0][0];
    expect((panel as any).areas.length).toBe(1);
    expect((panel as any).areas[0].shape).toBe("Rectangle");
    expect((panel as any).areas[0].show_measure).toBe(true);
  });
});

describe("New feature: movePanelFromDrawer", () => {
  beforeEach(() => { vi.clearAllMocks(); resetStore(); });

  it("moves panel from drawer back to grid", () => {
    const cfg = makeConfig(2, 2);
    useFigureStore.setState({
      config: cfg,
      drawerPanels: [{
        ...cfg.panels[0][0],
        image_name: "drawer_img.tif",
        brightness: 2.0,
      }],
    });
    mockApi.patchPanel.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().movePanelFromDrawer(0, 1, 1);

    const panel = useFigureStore.getState().config!.panels[1][1];
    expect(panel.image_name).toBe("drawer_img.tif");
    expect(panel.brightness).toBe(2.0);
  });
});

// ── New tests for Phase 1-8 features ───────────────────────────

describe("HeaderGroup line_style and line_length", () => {
  beforeEach(() => { vi.clearAllMocks(); resetStore(); });

  it("default header group has line_style and line_length", () => {
    const group = makeHeaderGroup({});
    expect(group.line_style).toBe("solid");
    expect(group.line_length).toBe(1.0);
  });

  it("supports dashed, dotted, dash-dot line styles", () => {
    const group = makeHeaderGroup({ line_style: "dashed", line_length: 0.75 });
    expect(group.line_style).toBe("dashed");
    expect(group.line_length).toBe(0.75);
  });
});

describe("AxisLabel visible field", () => {
  it("default axis labels have visible: true", () => {
    const cfg = makeConfig(2, 2);
    expect(cfg.column_labels[0].visible).toBe(true);
    expect(cfg.row_labels[0].visible).toBe(true);
  });
});

describe("Header group formatting sync", () => {
  beforeEach(() => { vi.clearAllMocks(); resetStore(); });

  it("updateHeaderGroupFormatting syncs font_size across level 0 of both axes", () => {
    const cfg = makeConfig(2, 3);
    cfg.column_headers = [{ headers: [makeHeaderGroup({ text: "A", columns_or_rows: [0], font_size: 10 })] }];
    cfg.row_headers = [{ headers: [makeHeaderGroup({ text: "B", columns_or_rows: [0], font_size: 10, position: "Left" })] }];
    useFigureStore.setState({ config: cfg });
    mockApi.patchColumnHeaders.mockResolvedValue({});
    mockApi.patchRowHeaders.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().updateHeaderGroupFormatting("col", 0, 0, { font_size: 16 });

    const state = useFigureStore.getState().config!;
    // Both axes should have font_size 16 at level 0
    expect(state.column_headers[0].headers[0].font_size).toBe(16);
    expect(state.row_headers[0].headers[0].font_size).toBe(16);
  });
});

describe("Label formatting sync", () => {
  beforeEach(() => { vi.clearAllMocks(); resetStore(); });

  it("updateLabelFormatting syncs font_size across both axes", () => {
    const cfg = makeConfig(2, 3);
    useFigureStore.setState({ config: cfg });
    mockApi.patchColumnLabels.mockResolvedValue({});
    mockApi.patchRowLabels.mockResolvedValue({});
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().updateLabelFormatting("col", 0, { font_size: 18 });

    const state = useFigureStore.getState().config!;
    // All labels should have font_size 18
    for (const lbl of state.column_labels) {
      expect(lbl.font_size).toBe(18);
    }
    for (const lbl of state.row_labels) {
      expect(lbl.font_size).toBe(18);
    }
  });
});

describe("Panel swap preserves settings", () => {
  beforeEach(() => { vi.clearAllMocks(); resetStore(); });

  it("swapPanels exchanges panel data including brightness", () => {
    const cfg = makeConfig(2, 2);
    cfg.panels[0][0].brightness = 1.5;
    cfg.panels[0][0].image_name = "img1.tif";
    cfg.panels[0][1].brightness = 0.8;
    cfg.panels[0][1].image_name = "img2.tif";
    useFigureStore.setState({ config: cfg });
    mockApi.patchPanel.mockResolvedValue({});
    mockApi.getPanelPreview.mockResolvedValue({ image: "", width: 0, height: 0, format: "png" });
    mockApi.getPreview.mockResolvedValue({ image: "img", width: 100, height: 100, format: "png" });

    useFigureStore.getState().swapPanels(0, 0, 0, 1);

    const state = useFigureStore.getState().config!;
    expect(state.panels[0][0].image_name).toBe("img2.tif");
    expect(state.panels[0][0].brightness).toBe(0.8);
    expect(state.panels[0][1].image_name).toBe("img1.tif");
    expect(state.panels[0][1].brightness).toBe(1.5);
  });
});

describe("Config normalize_widths", () => {
  it("makeConfig does not include normalize_widths by default", () => {
    const cfg = makeConfig(2, 2);
    expect(cfg.normalize_widths).toBeUndefined();
  });

  it("can set normalize_widths and normalize_mode", () => {
    const cfg = makeConfig(2, 2);
    const updated = { ...cfg, normalize_widths: true, normalize_mode: "height" };
    expect(updated.normalize_widths).toBe(true);
    expect(updated.normalize_mode).toBe("height");
  });
});
