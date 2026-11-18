/* The builder is shielded from the moment a tab switch is QUEUED, not only
   once it starts running — otherwise a click queued behind a long upload
   leaves a window in which a dialog of the outgoing document can be opened
   and later saved into the incoming one. */

import { describe, it, expect, vi } from "vitest";

const mockApi = vi.hoisted(() => ({
  stashState: vi.fn(async () => ({ ok: true })),
  activateState: vi.fn(),
  listImages: vi.fn(async () => ({ names: [] })),
  deleteImage: vi.fn(async () => ({})),
  getResolutions: vi.fn(async () => ({})),
  updateConfig: vi.fn(async () => ({})),
  getPreview: vi.fn(async () => ({ image: "p", width: 1, height: 1, format: "png" })),
  getPanelPreview: vi.fn(async () => ({ image: "t" })),
  dropState: vi.fn(async () => ({ ok: true })),
}));
vi.mock("../api/client", () => ({
  api: mockApi,
  checkHealth: vi.fn(async () => true),
  lastHealthError: "",
  settleConfigWrites: vi.fn(async () => {}),
  default: vi.fn(),
}));
vi.mock("../components/shared/ConfirmDialog", () => ({
  confirmThree: vi.fn(async () => "cancel"),
  alert: vi.fn(async () => {}),
}));
vi.mock("../components/shared/SaveProjectDialog", () => ({ saveProjectDialog: vi.fn(async () => null) }));

import { switchToDocument } from "./projectNav";
import { serializeDocOp } from "./docOpQueue";
import { useCollageStore } from "../store/collageStore";

describe("document-swap shield", () => {
  it("is raised when the switch is queued and dropped when it finishes", async () => {
    const cs = useCollageStore.getState();
    const a = cs.docAdd({ path: null, name: "A" });
    const b = cs.docAdd({ path: null, name: "B" });
    useCollageStore.getState().docSetActive(a);

    let release!: () => void;
    const gate = new Promise<void>((r) => { release = r; });
    const blocker = serializeDocOp(() => gate);   // e.g. a long upload holding the queue
    const sw = switchToDocument(b);
    expect(useCollageStore.getState().docSwapBusy).toBe(true);   // shielded while merely queued
    release();
    await blocker;
    await sw;
    expect(useCollageStore.getState().activeDocId).toBe(b);
    expect(useCollageStore.getState().docSwapBusy).toBe(false);
  });

  it("re-selecting the live document does not raise the shield", async () => {
    const id = useCollageStore.getState().activeDocId!;
    const p = switchToDocument(id);
    expect(useCollageStore.getState().docSwapBusy).toBe(false);
    await p;
  });
});
