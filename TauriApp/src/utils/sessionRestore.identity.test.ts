/* "Re-open tabs on relaunch" and a .mpf renamed while the app was closed
   (uid plan §2.8): the file is looked up by the uid the last-seen table saw
   at its path and reopens where it is now — in its saved tab order, with its
   opening rank and the active tab kept, a tab named after the old file
   taking the new name — and what still named the old path follows it
   (collage figures, labels, the table). A file found nowhere is dropped and
   named in the prompt. The sidecar is a fake in-memory "disk". */

import { describe, it, expect, vi, beforeEach } from "vitest";
import type { MpfResolveRef, MpfResolveResult } from "../api/types";

const files = vi.hoisted(() => new Map<string, string | null>());
const mockApi = vi.hoisted(() => ({
  readWorkflows: vi.fn(async (p: string) => ({
    manifest: null, blobs: {}, unsupported: false, version: null, uploads: [], renamed: {},
    missing_uploads: [], failed_uploads: [], doc_uid: files.get(p) ?? null,
  })),
  pathsExist: vi.fn(async (paths: string[]) => Object.fromEntries(paths.map((p) => [p, files.has(p)]))),
  // The sidecar's rules, every file in one folder.
  resolveMpf: vi.fn(async (refs: MpfResolveRef[]) => {
    const out: Record<string, MpfResolveResult> = {};
    for (const r of refs) {
      const exists = files.has(r.path);
      const there = exists ? files.get(r.path) ?? null : null;
      if (exists && (!r.uid || !there || there === r.uid)) { out[r.key] = { status: "ok", path: r.path, exists, uid: there }; continue; }
      const hits = r.uid ? [...files.keys()].filter((p) => p !== r.path && files.get(p) === r.uid) : [];
      out[r.key] = hits.length === 1
        ? { status: "moved", path: hits[0], exists, uid: r.uid! }
        : { status: "missing", path: null, exists, uid: null };
    }
    return out;
  }),
}));
vi.mock("../api/client", () => ({
  api: new Proxy(mockApi, { get: (t, k) => (k in t ? t[k as keyof typeof t] : vi.fn(async () => ({}))) }),
  checkHealth: vi.fn(async () => true),
  lastHealthError: "",
  settleConfigWrites: vi.fn(async () => {}),
  default: vi.fn(),
}));
const dialogs = vi.hoisted(() => ({ confirm: vi.fn(async () => true) }));
vi.mock("../components/shared/ConfirmDialog", () => ({
  confirm: dialogs.confirm,
  confirmThree: vi.fn(async () => "cancel"),
  alert: vi.fn(async () => true),
}));
vi.mock("./projectNav", () => ({ serializeDocOp: <T,>(fn: () => Promise<T>) => fn() }));

/** Last session: c, a, b (dragged into that order; opened a, b, c) and d,
 *  with a active; the last-seen table knows each file's uid. */
function lastSession() {
  localStorage.setItem("mpfig_reopen_tabs_v1", "1");
  localStorage.setItem("mpfig_open_docs_v1", JSON.stringify({
    docs: [
      { path: "/x/c.mpf", name: "c", rank: 2 },
      { path: "/x/a.mpf", name: "a", rank: 0 },
      { path: "/x/b.mpf", name: "My B", rank: 1 },
      { path: "/x/d.mpf", name: "d", rank: 3 },
    ],
    activePath: "/x/a.mpf",
  }));
  localStorage.setItem("mpfig_doc_registry_v1", JSON.stringify({
    v: 1,
    entries: ["a", "b", "c", "d"].map((n, i) => ({ uid: `du_${n}`, path: `/x/${n}.mpf`, t: 10 - i })),
  }));
}

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  localStorage.clear();
  files.clear();
  dialogs.confirm.mockResolvedValue(true);
});

describe("maybeRestoreSession — files renamed while the app was closed", () => {
  it("reopens them where they are now, in the saved order, ranks and active tab kept; one found nowhere is named", async () => {
    lastSession();
    files.set("/x/a-final.mpf", "du_a");                    // a.mpf renamed
    files.set("/x/b-v2.mpf", "du_b");                       // b.mpf renamed (its tab has a name of its own)
    files.set("/x/c.mpf", "du_c");                          // d.mpf deleted
    localStorage.setItem("mpfig.source_names", JSON.stringify({ "p:/x/a.mpf::r0c0_i0": "Ctrl" }));
    // Fresh modules: the session is read when sessionRestore loads.
    const { useCollageStore } = await import("../store/collageStore");
    const { useFigureStore } = await import("../store/figureStore");
    const { maybeRestoreSession, persistOpenDocs } = await import("./sessionRestore");
    const { registryUidAt } = await import("./docIdentity");
    const loadProject = vi.fn(async () => {});
    useFigureStore.setState({ loadProject } as never);
    useCollageStore.setState({
      items: [{
        id: "ia", kind: "figure", src: "data:", x: 0, y: 0, w: 1, h: 1, naturalW: 1, naturalH: 1,
        rotation: 0, z: 1, name: "a", projectPath: "/x/a.mpf",
      }],
    } as never);

    expect(await maybeRestoreSession()).toBe(true);
    expect(mockApi.resolveMpf).toHaveBeenCalledTimes(1);
    expect(mockApi.resolveMpf.mock.calls[0][0].map((r) => [r.path, r.uid])).toEqual([
      ["/x/c.mpf", "du_c"], ["/x/a.mpf", "du_a"], ["/x/b.mpf", "du_b"], ["/x/d.mpf", "du_d"],
    ]);
    expect(dialogs.confirm).toHaveBeenCalledWith(expect.objectContaining({
      body: expect.stringContaining("Not found (moved or deleted): d.mpf"),
      confirmLabel: "Reopen 3 tabs",
    }));
    const tabs = useCollageStore.getState().openDocs;
    expect(tabs.map((d) => [d.path, d.name])).toEqual([
      ["/x/c.mpf", "c"], ["/x/a-final.mpf", "a-final"], ["/x/b-v2.mpf", "My B"],
    ]);
    // Opening order (seq) follows the saved ranks: a, b, c.
    const bySeq = [...tabs].sort((p, q) => (p.seq ?? 0) - (q.seq ?? 0)).map((d) => d.path);
    expect(bySeq).toEqual(["/x/a-final.mpf", "/x/b-v2.mpf", "/x/c.mpf"]);
    expect(useCollageStore.getState().activeDocId).toBe(tabs[1].id);
    expect(loadProject).toHaveBeenCalledWith("/x/a-final.mpf", tabs[1].id);
    // What named the old path follows the file.
    expect(useCollageStore.getState().items[0]).toMatchObject({ projectPath: "/x/a-final.mpf", projectUid: "du_a", name: "a-final" });
    expect(JSON.parse(localStorage.getItem("mpfig.source_names")!)).toEqual({ "p:/x/a-final.mpf::r0c0_i0": "Ctrl" });
    expect(registryUidAt("/x/a-final.mpf")).toBe("du_a");

    // The next save of the session records the new paths, same ranks.
    persistOpenDocs();
    expect(JSON.parse(localStorage.getItem("mpfig_open_docs_v1")!).docs).toEqual([
      { path: "/x/c.mpf", name: "c", rank: 2 },
      { path: "/x/a-final.mpf", name: "a-final", rank: 0 },
      { path: "/x/b-v2.mpf", name: "My B", rank: 1 },
    ]);
  });

  it("Start fresh: nothing is re-pointed", async () => {
    lastSession();
    files.set("/x/a-final.mpf", "du_a");
    dialogs.confirm.mockResolvedValue(false);
    const { useCollageStore } = await import("../store/collageStore");
    const { maybeRestoreSession } = await import("./sessionRestore");
    useCollageStore.setState({
      items: [{
        id: "ia", kind: "figure", src: "data:", x: 0, y: 0, w: 1, h: 1, naturalW: 1, naturalH: 1,
        rotation: 0, z: 1, name: "a", projectPath: "/x/a.mpf",
      }],
    } as never);
    expect(await maybeRestoreSession()).toBe(false);
    expect(dialogs.confirm).toHaveBeenCalledWith(expect.objectContaining({
      body: expect.stringContaining("Not found (moved or deleted): c.mpf, b.mpf, d.mpf"),
      confirmLabel: "Reopen tab",
    }));
    expect(useCollageStore.getState().openDocs.map((d) => d.path)).toEqual([null]);
    expect(useCollageStore.getState().items[0].projectPath).toBe("/x/a.mpf");
  });

  it("a file still at its path is reopened there when its uid is found only in a copy it must not follow", async () => {
    lastSession();
    // The table saw a.mpf's uid in "a copy.mpf" too (opened alone once); a.mpf
    // was since replaced by another document from outside this machine.
    const reg = JSON.parse(localStorage.getItem("mpfig_doc_registry_v1")!);
    reg.entries.push({ uid: "du_a", path: "/x/a copy.mpf", t: 1 });
    localStorage.setItem("mpfig_doc_registry_v1", JSON.stringify(reg));
    files.set("/x/a.mpf", "du_w");
    files.set("/x/a copy.mpf", "du_a");
    files.set("/x/c.mpf", "du_c");
    const { useCollageStore } = await import("../store/collageStore");
    const { useFigureStore } = await import("../store/figureStore");
    const { maybeRestoreSession } = await import("./sessionRestore");
    useFigureStore.setState({ loadProject: vi.fn(async () => {}) } as never);
    expect(await maybeRestoreSession()).toBe(true);
    expect(dialogs.confirm).toHaveBeenCalledWith(expect.objectContaining({
      body: expect.stringContaining("Not found (moved or deleted): b.mpf, d.mpf"),
    }));
    expect(useCollageStore.getState().openDocs.map((d) => d.path)).toEqual(["/x/c.mpf", "/x/a.mpf"]);
  });

  it("last session's record is kept until \"Reopen last session?\" is answered", async () => {
    lastSession();
    files.set("/x/a.mpf", "du_a");
    const saved = localStorage.getItem("mpfig_open_docs_v1");
    const { useCollageStore } = await import("../store/collageStore");
    const { maybeRestoreSession, persistOpenDocs } = await import("./sessionRestore");
    // A tab seeded from the collage while the prompt is up (AppShell's
    // subscriber persists every change).
    useCollageStore.getState().docEnsure("/x/seeded.mpf");
    persistOpenDocs();
    expect(localStorage.getItem("mpfig_open_docs_v1")).toBe(saved);
    let answer!: (v: boolean) => void;
    dialogs.confirm.mockImplementationOnce(() => new Promise<boolean>((r) => { answer = r; }));
    const done = maybeRestoreSession();
    await vi.waitFor(() => expect(dialogs.confirm).toHaveBeenCalled());
    persistOpenDocs();
    expect(localStorage.getItem("mpfig_open_docs_v1")).toBe(saved);
    answer(false);                                           // Start fresh
    expect(await done).toBe(false);
    expect(localStorage.getItem("mpfig_open_docs_v1")).toBe(saved);
    // Answered: the next change is recorded as before.
    persistOpenDocs();
    expect(JSON.parse(localStorage.getItem("mpfig_open_docs_v1")!).docs.map((d: { path: string }) => d.path))
      .toEqual(["/x/seeded.mpf"]);
  });
});
