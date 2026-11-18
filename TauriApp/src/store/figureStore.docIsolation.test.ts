/* Store-level guarantees that keep open documents apart (0.1.421):
   debounced edits are flushed before a tab is stashed and never fire into the
   next one, late thumbnails of a previous tab are dropped, uploads are one
   doc op, and new / load tell the sidecar which tab they belong to. */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

const mockApi = vi.hoisted(() => ({
  updateConfig: vi.fn(async () => ({})),
  patchGrid: vi.fn(async () => ({})),
  deleteImage: vi.fn(async () => ({})),
  listImages: vi.fn(async () => ({ names: ["x.png"] })),
  newBlank: vi.fn(async () => ({ ok: true })),
  getResolutions: vi.fn(async () => ({})),
  getPreview: vi.fn(async () => ({ image: "p", width: 1, height: 1, format: "png" })),
  getPanelPreview: vi.fn(),
  loadProject: vi.fn(),
  activateState: vi.fn(),
  uploadImages: vi.fn(),
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
import { serializeDocOp, setBackendTracksLiveDoc } from "../utils/docOpQueue";
import type { FigureConfig } from "../api/types";

function cfg(): FigureConfig {
  return {
    rows: 1, cols: 1, spacing: 0.02, output_format: "TIFF", background: "White",
    panels: [[{ image_name: "a.png" } as never]],
    column_labels: [], row_labels: [], column_headers: [], row_headers: [],
    resolution_entries: {}, dpi: 300,
  } as FigureConfig;
}
const resp = () => ({ config: cfg(), image_names: ["a.png"], thumbnails: { "a.png": "t" } });

function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
}

beforeEach(() => {
  vi.clearAllMocks();
  setBackendTracksLiveDoc(false);
  useFigureStore.setState({ config: cfg(), unsaved: false });
});
afterEach(() => { vi.useRealTimers(); });

describe("debounced edits around a tab switch", () => {
  it("flushConfigWrites pushes a pending edit now and cancels its timer", async () => {
    vi.useFakeTimers();
    useFigureStore.getState().setSpacing(0.3);         // 300 ms debounce armed
    await useFigureStore.getState().flushConfigWrites();
    expect(mockApi.updateConfig).toHaveBeenCalledTimes(1);
    expect((mockApi.updateConfig.mock.calls[0] as unknown as [FigureConfig])[0].spacing).toBe(0.3);
    vi.runAllTimers();
    expect(mockApi.patchGrid).not.toHaveBeenCalled();  // can't fire into the next tab
  });

  it("a clean document with nothing pending sends nothing", async () => {
    await useFigureStore.getState().flushConfigWrites();
    expect(mockApi.updateConfig).not.toHaveBeenCalled();
  });

  it("replacing the document drops a pending debounce instead of firing it", async () => {
    vi.useFakeTimers();
    useFigureStore.getState().setSpacing(0.4);
    mockApi.activateState.mockResolvedValue(resp());
    await useFigureStore.getState().restoreDocState("B", { path: null, dirty: false });
    vi.runAllTimers();
    expect(mockApi.patchGrid).not.toHaveBeenCalled();
  });
});

describe("late responses of the previous document", () => {
  it("a thumbnail rendered for the old tab is not painted into the new one", async () => {
    const d = deferred<{ image: string }>();
    mockApi.getPanelPreview.mockReturnValueOnce(d.promise).mockResolvedValue({ image: "new" });
    const late = useFigureStore.getState().refreshPanelThumbnail(0, 0);
    mockApi.activateState.mockResolvedValue(resp());
    await useFigureStore.getState().restoreDocState("B", { path: null, dirty: false });
    d.resolve({ image: "OLD-TAB" });
    await late;
    expect(useFigureStore.getState().panelThumbnails["0-0"]).not.toBe("OLD-TAB");
    expect(useFigureStore.getState().panelProcessedSizes).toEqual({});
  });
});

describe("uploads are one doc op", () => {
  it("a tab switch queued during an upload waits for the whole batch", async () => {
    const d = deferred<{ names: string[]; thumbnails: Record<string, string> }>();
    mockApi.uploadImages.mockReturnValueOnce(d.promise);
    const files = [new File(["x"], "u.png")];
    const up = useFigureStore.getState().uploadImages(files);
    let switched = false;
    const sw = serializeDocOp(async () => { switched = true; });
    await new Promise((r) => setTimeout(r, 0));
    expect(switched).toBe(false);
    d.resolve({ names: ["u.png"], thumbnails: { "u.png": "t" } });
    await up;
    await sw;
    expect(switched).toBe(true);
    expect(useFigureStore.getState().loadedImages["u.png"]).toBeDefined();
  });
});

describe("the sidecar is told which tab a document belongs to", () => {
  it("new blank uses the new-blank endpoint (no per-image DELETE) when supported", async () => {
    setBackendTracksLiveDoc(true);
    await useFigureStore.getState().newBlankFigure("d9");
    expect(mockApi.newBlank).toHaveBeenCalledWith("d9");
    expect(mockApi.listImages).not.toHaveBeenCalled();
    expect(mockApi.deleteImage).not.toHaveBeenCalled();
  });

  it("an older sidecar keeps the legacy reset", async () => {
    await useFigureStore.getState().newBlankFigure("d9");
    expect(mockApi.newBlank).not.toHaveBeenCalled();
    expect(mockApi.deleteImage).toHaveBeenCalledWith("x.png");
  });

  it("loadProject passes the tab id", async () => {
    mockApi.loadProject.mockResolvedValue(resp());
    await useFigureStore.getState().loadProject("/tmp/a.mpf", "d3");
    expect(mockApi.loadProject).toHaveBeenCalledWith("/tmp/a.mpf", "d3");
  });
});
