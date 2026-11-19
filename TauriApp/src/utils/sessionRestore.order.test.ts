/* "Re-open tabs on relaunch" puts the document tabs back in the order the
   user left them (tabs can be dragged), including a tab that was already
   seeded from a collage figure before the restore ran. */

import { describe, it, expect, vi, beforeEach } from "vitest";

const mockApi = vi.hoisted(() => ({
  pathsExist: vi.fn(async (paths: string[]) => Object.fromEntries(paths.map((p) => [p, true]))),
}));
vi.mock("../api/client", () => ({
  api: new Proxy(mockApi, { get: (t, k) => (k in t ? t[k as keyof typeof t] : vi.fn(async () => ({}))) }),
  checkHealth: vi.fn(async () => true),
  lastHealthError: "",
  settleConfigWrites: vi.fn(async () => {}),
  default: vi.fn(),
}));
vi.mock("../components/shared/ConfirmDialog", () => ({
  confirm: vi.fn(async () => true),
  confirmThree: vi.fn(async () => "cancel"),
  alert: vi.fn(async () => true),
}));
vi.mock("./projectNav", () => ({ serializeDocOp: <T,>(fn: () => Promise<T>) => fn() }));

describe("maybeRestoreSession — tab order", () => {
  beforeEach(() => {
    vi.resetModules();
    localStorage.clear();
  });

  it("restores the saved order even when a tab was seeded from the collage first", async () => {
    localStorage.setItem("mpfig_reopen_tabs_v1", "1");
    localStorage.setItem("mpfig_open_docs_v1", JSON.stringify({
      docs: [
        { path: "/x/a.mpf", name: "a" },
        { path: "/x/b.mpf", name: "b" },
        { path: "/x/c.mpf", name: "c" },
      ],
      activePath: "/x/b.mpf",
    }));
    // Fresh modules: the session is read when sessionRestore loads.
    const { useCollageStore } = await import("../store/collageStore");
    const { useFigureStore } = await import("../store/figureStore");
    const { maybeRestoreSession } = await import("./sessionRestore");
    const loadProject = vi.fn(async () => {});
    useFigureStore.setState({ loadProject } as never);
    // DocumentTabs' mount effect seeds a tab for a collage-placed figure
    // BEFORE the restore runs: [Untitled_1, c].
    useCollageStore.getState().docEnsure("/x/c.mpf");

    expect(await maybeRestoreSession()).toBe(true);
    const tabs = useCollageStore.getState().openDocs;
    expect(tabs.map((d) => d.path)).toEqual(["/x/a.mpf", "/x/b.mpf", "/x/c.mpf"]);
    expect(useCollageStore.getState().activeDocId).toBe(tabs[1].id);
    expect(loadProject).toHaveBeenCalledWith("/x/b.mpf", tabs[1].id);
  });

  it("re-opens same-name tabs in their OPENING order, so their labels don't swap after a drag", async () => {
    // Last session: /x/fig1 opened first, /y/fig1 second ("fig1 (2)"), then
    // the user dragged /y in front. The list is in tab order; rank is the
    // opening order.
    localStorage.setItem("mpfig_reopen_tabs_v1", "1");
    localStorage.setItem("mpfig_open_docs_v1", JSON.stringify({
      docs: [
        { path: "/y/fig1.mpf", name: "fig1", rank: 1 },
        { path: "/x/fig1.mpf", name: "fig1", rank: 0 },
      ],
      activePath: "/x/fig1.mpf",
    }));
    const { useCollageStore } = await import("../store/collageStore");
    const { useFigureStore } = await import("../store/figureStore");
    const { maybeRestoreSession, persistOpenDocs } = await import("./sessionRestore");
    const { figureLabels } = await import("../components/dialogs/AnalysisNodeGraph");
    useFigureStore.setState({ loadProject: vi.fn(async () => {}) } as never);
    // /y was also seeded from the collage before the restore ran — it got an
    // EARLIER seq than /x would; the restore must hand them back out.
    useCollageStore.getState().docEnsure("/y/fig1.mpf", "fig1");

    expect(await maybeRestoreSession()).toBe(true);
    const tabs = useCollageStore.getState().openDocs;
    expect(tabs.map((d) => d.path)).toEqual(["/y/fig1.mpf", "/x/fig1.mpf"]);   // tab order kept
    const labels = figureLabels(tabs);
    const byPath = Object.fromEntries(tabs.map((d) => [d.path, labels[d.id]]));
    expect(byPath).toEqual({ "/x/fig1.mpf": "fig1", "/y/fig1.mpf": "fig1 (2)" });

    // And the next save of the session writes the same ranks back.
    persistOpenDocs();
    const saved = JSON.parse(localStorage.getItem("mpfig_open_docs_v1") as string);
    expect(saved.docs).toEqual([
      { path: "/y/fig1.mpf", name: "fig1", rank: 1 },
      { path: "/x/fig1.mpf", name: "fig1", rank: 0 },
    ]);
  });
});
