/* Document tabs and a .mpf renamed, moved or deleted outside the app (uid
   plan §2.7): opening the renamed file re-points and focuses its open tab
   (edits kept, nothing reloaded) while a copy gets a tab of its own; Save
   writes the renamed file under its uid — never a new file at the old name —
   and a file found nowhere asks Save here again / Save somewhere else… /
   Cancel; a tab whose file was never read sends no uid; switching to a tab
   whose file moved loads it from where it is, and one found nowhere offers
   Locate… / Close tab / Cancel; the close guard ignores collage figures
   whose file is gone. The sidecar is a fake in-memory "disk" of path → the
   uid its doc.json carries. */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { MpfResolveRef, MpfResolveResult, WorkflowsReadResponse } from "../api/types";

const files = vi.hoisted(() => new Map<string, string | null>());
const mockApi = vi.hoisted(() => ({
  readWorkflows: vi.fn(),
  writeWorkflows: vi.fn(),
  saveProject: vi.fn(),
  loadProject: vi.fn(),
  activateState: vi.fn(),
  pathsExist: vi.fn(),
  resolveMpf: vi.fn(),
  stashState: vi.fn(async () => ({ ok: true })),
  dropState: vi.fn(async () => ({ ok: true })),
  updateConfig: vi.fn(async () => ({})),
  getPreview: vi.fn(async () => ({ image: "p", width: 1, height: 1, format: "png" })),
  getPanelPreview: vi.fn(async () => ({ image: "t" })),
}));
vi.mock("../api/client", () => ({
  api: mockApi,
  checkHealth: vi.fn(async () => true),
  lastHealthError: "",
  settleConfigWrites: vi.fn(async () => {}),
  default: vi.fn(),
}));
const dialogs = vi.hoisted(() => ({
  confirm: vi.fn(async () => true),
  confirmThree: vi.fn(async () => "cancel"),
  alert: vi.fn(async () => true),
  saveProjectDialog: vi.fn(async () => null as string | null),
  open: vi.fn(async () => null as string | null),
}));
vi.mock("../components/shared/ConfirmDialog", () => ({
  confirm: dialogs.confirm, confirmThree: dialogs.confirmThree, alert: dialogs.alert,
}));
vi.mock("../components/shared/SaveProjectDialog", () => ({ saveProjectDialog: dialogs.saveProjectDialog }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: dialogs.open }));

import {
  closeDoc, ensureProjectSaved, locateCollageItem, openProjectIntoTab, saveAllOpenDocs, saveDocument,
  saveProjectWithDialog, switchToDocument,
} from "./projectNav";
import { useCollageStore, type CollageItem, type DocTab } from "../store/collageStore";
import { useFigureStore } from "../store/figureStore";
import { useAnalysisWorkspaceStore, resetAnalysisWorkspaceForTests } from "../store/analysisWorkspaceStore";

const EMPTY: WorkflowsReadResponse = {
  manifest: null, blobs: {}, unsupported: false, version: null, uploads: [], renamed: {},
  missing_uploads: [], failed_uploads: [],
};
const CFG = {
  rows: 1, cols: 1, spacing: 0, panels: [[{ image_name: "" }]],
  column_labels: [], row_labels: [], column_headers: [], row_headers: [],
};
const notFound = (p: string) => new Error(`API error: "Project file not found: ${p}"`);
// What api.writeWorkflows throws for the sidecar's 409 {other_document}.
const otherDocument = () => new Error("Its file now holds another document (renamed or replaced outside the app), so nothing was written into it.");

const A: DocTab = { id: "doc_a", path: "/exp/a.mpf", name: "a", seq: 1 };
const B: DocTab = { id: "doc_b", path: "/exp/b.mpf", name: "b", seq: 2 };

/** A fake /api/fs/mpf-resolve with the sidecar's rules: the candidates, then
 *  (unless `scan: false`) the files in the folders of the path, of the
 *  candidates and in `dirs` — no other folder. */
function fakeResolve(refs: MpfResolveRef[], opts: { scan?: boolean } = {}): Record<string, MpfResolveResult> {
  const dirOf = (p: string) => p.slice(0, p.lastIndexOf("/"));
  const out: Record<string, MpfResolveResult> = {};
  for (const r of refs) {
    const exists = files.has(r.path);
    const there = exists ? files.get(r.path) ?? null : null;
    if (exists && (!r.uid || !there || there === r.uid)) { out[r.key] = { status: "ok", path: r.path, exists, uid: there }; continue; }
    if (!r.uid) { out[r.key] = { status: "missing", path: null, exists, uid: null }; continue; }
    const cands = r.candidates ?? [];
    const folders = new Set([r.path, ...cands].map(dirOf).concat(r.dirs ?? []));
    const scanned = opts.scan === false ? [] : [...files.keys()].filter((p) => folders.has(dirOf(p)));
    const fromCand = cands.filter((p) => p !== r.path && files.get(p) === r.uid);
    const hits = [...new Set([...fromCand, ...scanned])].filter((p) => p !== r.path && files.get(p) === r.uid);
    // Several files carry the uid: none is picked, not even a candidate.
    if (hits.length === 1) out[r.key] = { status: "moved", path: hits[0], exists, uid: r.uid };
    else if (hits.length > 1) out[r.key] = { status: "ambiguous", path: null, exists, uid: null, matches: hits };
    else out[r.key] = exists ? { status: "ok", path: r.path, exists, uid: there } : { status: "missing", path: null, exists, uid: null };
  }
  return out;
}

function move(from: string, to: string) {
  files.set(to, files.get(from) ?? null);
  files.delete(from);
}
async function flush(n = 80) { for (let i = 0; i < n; i++) await Promise.resolve(); }
const tabs = () => useCollageStore.getState().openDocs;
const tabOf = (id: string) => tabs().find((d) => d.id === id);
const ws = () => useAnalysisWorkspaceStore.getState();
const fig = () => useFigureStore.getState();
function figItem(id: string, path: string, projectUid?: string): CollageItem {
  return {
    id, kind: "figure", src: "data:", x: 0, y: 0, w: 10, h: 10, naturalW: 10, naturalH: 10,
    rotation: 0, z: 1, name: path.split("/").pop()!.replace(/\.mpf$/, ""), projectPath: path,
    ...(projectUid ? { projectUid } : {}),
  } as CollageItem;
}

/** Fresh session: `docs` open, `active` live in the builder, every file read. */
async function open(docs: DocTab[], active: string, items: CollageItem[] = []) {
  useCollageStore.setState({ openDocs: [], activeDocId: null, items: [], missingItemIds: [] });
  resetAnalysisWorkspaceForTests();
  useCollageStore.setState({ openDocs: docs.map((d) => ({ ...d })), activeDocId: active, items, mode: "builder" });
  useFigureStore.setState({ currentProjectPath: docs.find((d) => d.id === active)?.path ?? null, unsaved: false });
  await flush();
}

let quiet: { mockRestore: () => void };
beforeEach(() => {
  vi.clearAllMocks();
  quiet = vi.spyOn(console, "error").mockImplementation(() => {});
  files.clear();
  localStorage.clear();
  files.set(A.path!, "du_a");
  files.set(B.path!, "du_b");
  // Answers queued by a test (mockResolvedValueOnce) never reach the next one.
  for (const d of Object.values(dialogs)) d.mockReset();
  dialogs.confirm.mockResolvedValue(true);
  dialogs.alert.mockResolvedValue(true);
  dialogs.confirmThree.mockResolvedValue("cancel");
  dialogs.saveProjectDialog.mockResolvedValue(null);
  dialogs.open.mockResolvedValue(null);
  mockApi.readWorkflows.mockImplementation(async (p: string) => {
    if (!files.has(p)) throw notFound(p);
    return { ...EMPTY, doc_uid: files.get(p) ?? null };
  });
  mockApi.writeWorkflows.mockImplementation(async (p: string, _payload: unknown, opts?: { expectUid?: string }) => {
    if (!files.has(p)) throw notFound(p);
    // The sidecar's check: a file naming another uid than expected is left alone.
    const there = files.get(p);
    if (opts?.expectUid && there && there !== opts.expectUid) throw otherDocument();
    return { ok: true, size_bytes: 1, workflows_bytes: 1, missing_uploads: [], fast_path: true };
  });
  mockApi.loadProject.mockImplementation(async (p: string) => {
    if (!files.has(p)) throw notFound(p);
    return { config: CFG, image_names: [], thumbnails: {} };
  });
  mockApi.activateState.mockImplementation(async () => ({ config: CFG, image_names: [], thumbnails: {} }));
  // The sidecar's uid rule: a copy of a file that is still there is a new
  // document; else the uid sent, else the file's own.
  let minted = 0;
  mockApi.saveProject.mockImplementation(async (path: string, opts?: { carryFrom?: string | null; docUid?: string }) => {
    const src = opts?.carryFrom === undefined ? path : (opts.carryFrom ?? "");
    const want = opts?.docUid;
    let uid: string;
    if (src && src !== path && files.has(src) && (files.get(src) == null || files.get(src) === (want ?? files.get(src)))) {
      uid = `du_minted_${++minted}`;
    } else uid = want ?? files.get(path) ?? `du_minted_${++minted}`;
    files.set(path, uid);
    return { ok: true, path, doc_uid: uid, uid_minted: uid !== want };
  });
  mockApi.pathsExist.mockImplementation(async (paths: string[]) => Object.fromEntries(paths.map((p) => [p, files.has(p)])));
  mockApi.resolveMpf.mockImplementation(async (refs: MpfResolveRef[], opts?: { scan?: boolean }) => fakeResolve(refs, opts));
  useFigureStore.setState({ unsaved: false, currentProjectPath: null, config: CFG as never });
});
afterEach(() => {
  useCollageStore.setState({ openDocs: [], activeDocId: null, items: [], missingItemIds: [] });
  resetAnalysisWorkspaceForTests();
  quiet.mockRestore();
});

describe("opening a file that an open tab had before it was renamed", () => {
  it("re-points and focuses that tab — its unsaved edits kept, nothing reloaded", async () => {
    await open([A, B], B.id);
    useFigureStore.setState({ unsaved: true });
    await switchToDocument(A.id);                             // b's edits are parked
    expect(useCollageStore.getState().snapshotDirtyDocIds).toEqual([B.id]);
    move(B.path!, "/exp/renamed.mpf");
    mockApi.loadProject.mockClear();
    await openProjectIntoTab("/exp/renamed.mpf");
    expect(mockApi.loadProject).not.toHaveBeenCalled();
    expect(mockApi.activateState).toHaveBeenCalledWith(B.id);
    expect(tabs()).toEqual([
      { ...A },
      { id: B.id, path: "/exp/renamed.mpf", name: "renamed", seq: 2 },
    ]);
    expect(useCollageStore.getState().activeDocId).toBe(B.id);
    expect(fig().unsaved).toBe(true);
    expect(fig().currentProjectPath).toBe("/exp/renamed.mpf");
    expect(ws().docUids[B.id]).toBe("du_b");
  });

  it("the live tab's file: re-pointed in place", async () => {
    await open([A, B], A.id);
    move(A.path!, "/exp/a-v2.mpf");
    await openProjectIntoTab("/exp/a-v2.mpf");
    expect(mockApi.loadProject).not.toHaveBeenCalled();
    expect(tabs().map((d) => [d.id, d.path])).toEqual([[A.id, "/exp/a-v2.mpf"], [B.id, B.path]]);
    expect(fig().currentProjectPath).toBe("/exp/a-v2.mpf");
  });

  describe("a copy's tab, its own file renamed, is never re-pointed onto the original", () => {
    const C: DocTab = { id: "doc_c", path: "/exp/a copy.mpf", name: "a copy", seq: 3 };

    it("a copy opened alone (adopted the uid), the original seen on this machine", async () => {
      files.set(C.path!, "du_a");
      localStorage.setItem("mpfig_doc_registry_v1", JSON.stringify({ v: 1, entries: [{ uid: "du_a", path: A.path, t: 1 }] }));
      await open([C], C.id);
      expect(ws().docUids[C.id]).toBe("du_a");
      move(C.path!, "/exp/a final.mpf");
      await openProjectIntoTab(A.path!);
      expect(mockApi.loadProject).toHaveBeenCalledWith(A.path, expect.any(String));
      expect(tabOf(C.id)!.path).toBe(C.path);
      expect(tabs().find((d) => d.path === A.path)!.id).not.toBe(C.id);
    });

    it("a held copy (its original's tab closed since)", async () => {
      files.set(C.path!, "du_a");
      await open([A, C], A.id);
      expect(ws().docUids[C.id]).not.toBe("du_a");            // held
      await closeDoc(A.id);
      expect(tabOf(A.id)).toBeUndefined();
      move(C.path!, "/exp/a final.mpf");
      mockApi.loadProject.mockClear();
      await openProjectIntoTab(A.path!);
      expect(mockApi.loadProject).toHaveBeenCalledWith(A.path, expect.any(String));
      expect(tabOf(C.id)!.path).toBe(C.path);
      expect(files.get(A.path!)).toBe("du_a");
    });

    it("but it still follows its own renamed file", async () => {
      files.set(C.path!, "du_a");
      await open([A, C], A.id);
      move(C.path!, "/exp/a copy 2.mpf");
      await openProjectIntoTab("/exp/a copy 2.mpf");
      expect(tabs().map((d) => [d.id, d.path])).toEqual([[A.id, A.path], [C.id, "/exp/a copy 2.mpf"]]);
      expect(useCollageStore.getState().activeDocId).toBe(C.id);
    });
  });

  it("a copy whose original is still there opens in a tab of its own, as a new document", async () => {
    await open([A, B], A.id);
    files.set("/exp/b copy.mpf", "du_b");
    await openProjectIntoTab("/exp/b copy.mpf");
    expect(tabs()).toHaveLength(3);
    const copy = tabs()[2];
    expect(copy).toMatchObject({ path: "/exp/b copy.mpf" });
    expect(mockApi.loadProject).toHaveBeenCalledWith("/exp/b copy.mpf", copy.id);
    expect(tabOf(B.id)!.path).toBe(B.path);
    await flush();
    expect(ws().docUids[copy.id]).not.toBe("du_b");          // held: b's tab keeps the uid
  });
});

describe("saving a tab whose file was renamed, moved or deleted", () => {
  it("renamed: the save goes to the renamed file with its uid — no new file at the old name", async () => {
    await open([A], A.id);
    useFigureStore.setState({ unsaved: true });
    move(A.path!, "/exp/final.mpf");
    await expect(saveDocument(A.id)).resolves.toBe(true);
    expect(mockApi.saveProject).toHaveBeenCalledTimes(1);
    expect(mockApi.saveProject).toHaveBeenCalledWith("/exp/final.mpf", { carryFrom: "/exp/final.mpf", docUid: "du_a" });
    expect(files.has(A.path!)).toBe(false);
    expect(tabOf(A.id)).toMatchObject({ path: "/exp/final.mpf", name: "final" });
    expect(dialogs.confirmThree).not.toHaveBeenCalled();
  });

  it("found nowhere: Save here again / Save somewhere else… / Cancel — the document keeps its uid", async () => {
    await open([A], A.id);
    useFigureStore.setState({ unsaved: true });
    files.delete(A.path!);
    await expect(saveDocument(A.id)).resolves.toBe(false);   // Cancel
    expect(dialogs.confirmThree).toHaveBeenCalledWith(expect.objectContaining({
      title: '"a.mpf" is no longer where it was',
      confirmLabel: "Save here again", tertiaryLabel: "Save somewhere else…", cancelLabel: "Cancel",
    }));
    expect(mockApi.saveProject).not.toHaveBeenCalled();

    dialogs.confirmThree.mockResolvedValueOnce("tertiary");
    dialogs.saveProjectDialog.mockResolvedValueOnce("/exp/new/a.mpf");
    await expect(saveDocument(A.id)).resolves.toBe(true);
    expect(mockApi.saveProject).toHaveBeenLastCalledWith("/exp/new/a.mpf", { carryFrom: A.path, docUid: "du_a" });
    expect(tabOf(A.id)!.path).toBe("/exp/new/a.mpf");
    expect(ws().docUids[A.id]).toBe("du_a");
    expect(files.get("/exp/new/a.mpf")).toBe("du_a");

    files.delete("/exp/new/a.mpf");
    useFigureStore.setState({ unsaved: true });
    dialogs.confirmThree.mockResolvedValueOnce("confirm");
    await expect(saveDocument(A.id)).resolves.toBe(true);
    expect(mockApi.saveProject).toHaveBeenLastCalledWith("/exp/new/a.mpf", { carryFrom: "/exp/new/a.mpf", docUid: "du_a" });
  });

  it("a tab whose file was never read sends no uid (the file keeps its own)", async () => {
    mockApi.readWorkflows.mockImplementationOnce(async () => { throw new Error('API error: "Not a readable .mpf"'); });
    await open([A], A.id);
    expect(ws().hydrated[A.id]).toBe("failed");
    useFigureStore.setState({ unsaved: true });
    await expect(saveDocument(A.id)).resolves.toBe(true);
    expect(mockApi.saveProject).toHaveBeenCalledWith(A.path, { carryFrom: A.path });
  });

  it("Save all & close names the renamed file in its overwrite prompt", async () => {
    await open([A], A.id);
    useFigureStore.setState({ unsaved: true });
    move(A.path!, "/exp/final.mpf");
    dialogs.confirmThree.mockResolvedValueOnce("confirm");
    await expect(saveAllOpenDocs({ promptForLocation: true })).resolves.toBe(true);
    expect(dialogs.confirmThree).toHaveBeenCalledWith(expect.objectContaining({
      body: expect.stringContaining("/exp/final.mpf"), confirmLabel: "Overwrite",
    }));
    expect(mockApi.saveProject).toHaveBeenCalledWith("/exp/final.mpf", { carryFrom: "/exp/final.mpf", docUid: "du_a" });
    expect(tabOf(A.id)!.name).toBe("final");
  });

  it("ensureProjectSaved: a clean document whose file moved is where it is now; one that is gone is saved again", async () => {
    await open([A], A.id);
    move(A.path!, "/exp/a-moved.mpf");
    await expect(ensureProjectSaved()).resolves.toBe("/exp/a-moved.mpf");
    expect(dialogs.saveProjectDialog).not.toHaveBeenCalled();
    expect(tabOf(A.id)!.path).toBe("/exp/a-moved.mpf");
    files.delete("/exp/a-moved.mpf");
    await expect(ensureProjectSaved()).resolves.toBeNull();
    expect(dialogs.saveProjectDialog).toHaveBeenCalledWith({ defaultPath: "/exp/a-moved.mpf" });
  });

  it("the Sidebar's Save Project: a renamed file is where it is now, so another name is a copy of it", async () => {
    await open([A], A.id);
    move(A.path!, "/exp/renamed.mpf");
    dialogs.saveProjectDialog.mockResolvedValueOnce("/exp/c.mpf");
    await expect(saveProjectWithDialog()).resolves.toBe(true);
    expect(dialogs.saveProjectDialog).toHaveBeenCalledWith({ defaultPath: "/exp/renamed.mpf" });
    expect(mockApi.saveProject).toHaveBeenCalledWith("/exp/c.mpf", { carryFrom: "/exp/renamed.mpf", docUid: "du_a" });
    expect(files.get("/exp/renamed.mpf")).toBe("du_a");          // the original keeps its uid
    expect(files.get("/exp/c.mpf")).not.toBe("du_a");            // the copy is a new document
    expect(files.has(A.path!)).toBe(false);                      // nothing re-created at the old name
    // Gone: the user is asked, not handed the old name.
    files.delete("/exp/c.mpf");
    await expect(saveProjectWithDialog()).resolves.toBe(false);  // Cancel
    expect(dialogs.confirmThree).toHaveBeenCalledWith(expect.objectContaining({ confirmLabel: "Save here again" }));
    expect(dialogs.saveProjectDialog).toHaveBeenCalledTimes(1);
  });

  it("another document now has the tab's file name: Save never offers to write over it", async () => {
    // Replaced outside the app (the tab's own file is nowhere).
    await open([A], A.id);
    useFigureStore.setState({ unsaved: true });
    files.set(A.path!, "du_z");
    dialogs.confirm.mockResolvedValueOnce(false);
    await expect(saveDocument(A.id)).resolves.toBe(false);
    expect(dialogs.confirm).toHaveBeenLastCalledWith(expect.objectContaining({
      title: '"a.mpf" now holds a different document', confirmLabel: "Save somewhere else…",
    }));
    expect(dialogs.confirmThree).not.toHaveBeenCalled();
    expect(mockApi.saveProject).not.toHaveBeenCalled();
    // The names of two open documents swapped: the same, never onto the other tab's file.
    files.set(A.path!, "du_a");
    await open([A, B], A.id);
    useFigureStore.setState({ unsaved: true });
    files.set(A.path!, "du_b");
    files.set(B.path!, "du_a");
    dialogs.saveProjectDialog.mockResolvedValueOnce("/exp/a2.mpf");
    await expect(saveDocument(A.id)).resolves.toBe(true);
    expect(dialogs.confirm).toHaveBeenLastCalledWith(expect.objectContaining({ title: '"a.mpf" now holds a different document' }));
    expect(dialogs.confirmThree).not.toHaveBeenCalled();
    expect(mockApi.saveProject).toHaveBeenCalledTimes(1);
    expect(mockApi.saveProject.mock.calls[0][0]).toBe("/exp/a2.mpf");
    expect(files.get(A.path!)).toBe("du_b");
    expect(files.get(B.path!)).toBe("du_a");
  });

  it("an original and its Finder copy both open, the original deleted: Save asks, never writes the copy's file", async () => {
    await open([A, B], A.id);
    files.set("/exp/a copy.mpf", "du_a");
    await openProjectIntoTab("/exp/a copy.mpf");
    await flush();
    const copy = tabs().find((d) => d.path === "/exp/a copy.mpf")!;
    expect(ws().docUids[copy.id]).not.toBe("du_a");            // held
    await switchToDocument(A.id);
    await flush();
    files.delete(A.path!);
    useFigureStore.setState({ unsaved: true });
    mockApi.saveProject.mockClear();
    await expect(saveDocument(A.id)).resolves.toBe(false);    // Cancel
    expect(dialogs.confirmThree).toHaveBeenCalledWith(expect.objectContaining({ confirmLabel: "Save here again" }));
    expect(mockApi.saveProject).not.toHaveBeenCalled();
    expect(tabOf(A.id)!.path).toBe(A.path);
    expect(tabOf(copy.id)!.path).toBe("/exp/a copy.mpf");
  });
});

describe("the Analysis write (close / quit) of a tab whose file name now holds another document", () => {
  const writesTo = () => mockApi.writeWorkflows.mock.calls.map((c) => [c[0], (c[2] as { expectUid?: string } | undefined)?.expectUid]);

  it("its own file renamed beside it: the write follows it, never into the other document's file", async () => {
    await open([A], A.id);
    move(A.path!, "/exp/a_old.mpf");                         // same file, new name
    files.set(A.path!, "du_w");                               // another document takes the name
    mockApi.writeWorkflows.mockClear();
    await expect(ws().persistDocWorkflows(A.id, { force: true })).resolves.toBe("written");
    expect(writesTo()).toEqual([[A.path, "du_a"], ["/exp/a_old.mpf", "du_a"]]);
    expect(tabOf(A.id)!.path).toBe("/exp/a_old.mpf");
    expect(files.get(A.path!)).toBe("du_w");
  });

  it("two open tabs' files swapped names: neither is written into the other's file — the failure stands", async () => {
    await open([A, B], A.id);
    files.set(A.path!, "du_b");
    files.set(B.path!, "du_a");
    mockApi.writeWorkflows.mockClear();
    await expect(ws().persistDocWorkflows(A.id, { force: true })).rejects.toThrow(/now holds another document/);
    await expect(ws().persistDocWorkflows(B.id, { force: true })).rejects.toThrow(/now holds another document/);
    expect(writesTo()).toEqual([[A.path, "du_a"], [B.path, "du_b"]]);
    expect(tabs().map((d) => [d.id, d.path])).toEqual([[A.id, A.path], [B.id, B.path]]);
  });
});

describe("switching to a tab whose file is not where it was", () => {
  it("moved: loaded from where it is now, no prompt", async () => {
    await open([A, B], A.id);
    move(B.path!, "/exp/b-renamed.mpf");
    await expect(switchToDocument(B.id)).resolves.toBe(true);
    expect(mockApi.loadProject.mock.calls.map((c) => c[0])).toEqual([B.path, "/exp/b-renamed.mpf"]);
    expect(tabOf(B.id)).toMatchObject({ path: "/exp/b-renamed.mpf", name: "b-renamed", seq: 2 });
    expect(useCollageStore.getState().activeDocId).toBe(B.id);
    expect(dialogs.confirmThree).not.toHaveBeenCalled();
  });

  it("stored online only and its cloud app could not download it: says so, no Locate…, the tab stays", async () => {
    await open([A, B], A.id);
    let label: string | null = null;
    mockApi.loadProject.mockImplementationOnce(async (path: string) => {
      label = useCollageStore.getState().docSwapLabel;          // what the shield says meanwhile
      throw new Error(`API error: "Cloud file could not be downloaded: ${path} (Resource deadlock avoided)"`);
    });
    await expect(switchToDocument(B.id)).resolves.toBe(false);
    expect(label).toBe('Opening "b"…');
    expect(useCollageStore.getState().docSwapLabel).toBeNull();
    expect(dialogs.confirmThree).not.toHaveBeenCalled();
    expect(dialogs.alert).toHaveBeenCalledWith(expect.objectContaining({
      title: "Could not download the file",
      body: expect.stringContaining('"b" is stored online only'),
    }));
    expect(useCollageStore.getState().activeDocId).toBe(A.id);
    expect(tabOf(B.id)!.path).toBe(B.path);
    expect(useCollageStore.getState().missingItemIds).toEqual([]);
  });

  it("found nowhere: Locate… / Close tab / Cancel", async () => {
    await open([A, B], A.id, [figItem("ib", B.path!, "du_b")]);
    files.delete(B.path!);
    // Cancel: nothing changes; the figure is marked as having no file.
    await expect(switchToDocument(B.id)).resolves.toBe(false);
    expect(dialogs.confirmThree).toHaveBeenCalledWith(expect.objectContaining({
      confirmLabel: "Locate…", tertiaryLabel: "Close tab", cancelLabel: "Cancel",
    }));
    expect(useCollageStore.getState().activeDocId).toBe(A.id);
    expect(useCollageStore.getState().missingItemIds).toEqual(["ib"]);

    // Locate… another document: it opens in a tab of its own.
    files.set("/exp/other.mpf", "du_other");
    dialogs.confirmThree.mockResolvedValueOnce("confirm");
    dialogs.open.mockResolvedValueOnce("/exp/other.mpf");
    await expect(switchToDocument(B.id)).resolves.toBe(false);
    expect(dialogs.alert).toHaveBeenCalledWith(expect.objectContaining({ title: "A different document" }));
    expect(tabs().map((d) => d.path)).toEqual([A.path, B.path, "/exp/other.mpf"]);

    // Locate… the document itself (somewhere no lookup goes): the tab and
    // its figure follow it.
    files.set("/backup/b.mpf", "du_b");
    dialogs.confirmThree.mockResolvedValueOnce("confirm");
    dialogs.open.mockResolvedValueOnce("/backup/b.mpf");
    await expect(switchToDocument(B.id)).resolves.toBe(true);
    expect(dialogs.open).toHaveBeenCalledTimes(2);
    expect(tabOf(B.id)).toMatchObject({ path: "/backup/b.mpf", seq: 2 });
    expect(useCollageStore.getState().activeDocId).toBe(B.id);
    expect(mockApi.loadProject).toHaveBeenLastCalledWith("/backup/b.mpf", B.id);
    expect(useCollageStore.getState().items[0].projectPath).toBe("/backup/b.mpf");
    expect(useCollageStore.getState().missingItemIds).toEqual([]);
  });

  it("Locate… a file whose uid can't be told: its saved Analysis is read there before anything is written", async () => {
    await open([A, B], A.id);
    expect(ws().hydrated[B.id]).toBe("done");
    files.delete(B.path!);
    files.set("/elsewhere/other.mpf", null);                   // no uid (legacy, or a cloud placeholder)
    mockApi.readWorkflows.mockClear();
    dialogs.confirmThree.mockResolvedValueOnce("confirm");
    dialogs.open.mockResolvedValueOnce("/elsewhere/other.mpf");
    await expect(switchToDocument(B.id)).resolves.toBe(true);
    await flush();
    expect(tabOf(B.id)!.path).toBe("/elsewhere/other.mpf");
    expect(mockApi.readWorkflows.mock.calls.map((c) => c[0])).toContain("/elsewhere/other.mpf");
    expect(ws().hydrated[B.id]).toBe("done");
  });

  it("Locate… from a tab that knew no uid: the file's own is read and kept, not replaced by the tab's", async () => {
    files.set(B.path!, null);                                  // b names no uid
    await open([A, B], A.id);
    const provisional = ws().docUids[B.id];
    files.delete(B.path!);
    files.set("/elsewhere/other.mpf", "du_other");
    dialogs.confirmThree.mockResolvedValueOnce("confirm");
    dialogs.open.mockResolvedValueOnce("/elsewhere/other.mpf");
    await expect(switchToDocument(B.id)).resolves.toBe(true);
    await flush();
    expect(mockApi.readWorkflows.mock.calls.map((c) => c[0])).toContain("/elsewhere/other.mpf");
    expect(ws().docUids[B.id]).toBe("du_other");
    useFigureStore.setState({ unsaved: true });
    await expect(saveDocument(B.id)).resolves.toBe(true);
    expect(mockApi.saveProject).toHaveBeenLastCalledWith("/elsewhere/other.mpf",
      { carryFrom: "/elsewhere/other.mpf", docUid: "du_other" });
    expect(files.get("/elsewhere/other.mpf")).toBe("du_other");
    expect(files.get("/elsewhere/other.mpf")).not.toBe(provisional);
  });

  it("Locate… the very file (the same uid): counted as read, not read again", async () => {
    await open([A, B], A.id);
    files.delete(B.path!);
    files.set("/backup/b.mpf", "du_b");
    mockApi.readWorkflows.mockClear();
    dialogs.confirmThree.mockResolvedValueOnce("confirm");
    dialogs.open.mockResolvedValueOnce("/backup/b.mpf");
    await expect(switchToDocument(B.id)).resolves.toBe(true);
    await flush();
    expect(mockApi.readWorkflows).not.toHaveBeenCalled();
    expect(ws().hydrated[B.id]).toBe("done");
  });

  it("Close tab closes it, though its figure is in the collage (the file is gone)", async () => {
    await open([A, B], A.id, [figItem("ib", B.path!, "du_b")]);
    files.delete(B.path!);
    dialogs.confirmThree.mockResolvedValueOnce("tertiary");
    await expect(switchToDocument(B.id)).resolves.toBe(false);
    expect(tabs().map((d) => d.id)).toEqual([A.id]);
    expect(useCollageStore.getState().activeDocId).toBe(A.id);
  });
});

describe("Locate… for a collage figure whose file is missing", () => {
  it("the same document: the figure follows it, is no longer missing, and its file gets a tab", async () => {
    await open([A], A.id, [figItem("ig", "/exp/gone.mpf", "du_g"), figItem("ig2", "/exp/gone.mpf", "du_g")]);
    useCollageStore.getState().setItemMissing("ig", true);
    useCollageStore.getState().setItemMissing("ig2", true);
    files.set("/far/g.mpf", "du_g");
    dialogs.open.mockResolvedValueOnce("/far/g.mpf");
    await expect(locateCollageItem("ig")).resolves.toBe(true);
    const items = useCollageStore.getState().items;
    expect(items.map((i) => [i.id, i.projectPath, i.projectUid, i.name])).toEqual([
      ["ig", "/far/g.mpf", "du_g", "g"], ["ig2", "/far/g.mpf", "du_g", "g"],
    ]);
    expect(useCollageStore.getState().missingItemIds).toEqual([]);
    expect(tabs().map((d) => d.path)).toEqual([A.path, "/far/g.mpf"]);
    expect(dialogs.confirm).not.toHaveBeenCalled();
  });

  it("another document: only after \"Use it anyway?\", and only that figure", async () => {
    await open([A], A.id, [figItem("ig", "/exp/gone.mpf", "du_g"), figItem("ig2", "/exp/gone.mpf", "du_g")]);
    useCollageStore.getState().setItemMissing("ig", true);
    files.set("/far/other.mpf", "du_other");
    dialogs.open.mockResolvedValue("/far/other.mpf");
    dialogs.confirm.mockResolvedValueOnce(false);
    await expect(locateCollageItem("ig")).resolves.toBe(false);
    expect(useCollageStore.getState().items[0]).toMatchObject({ projectPath: "/exp/gone.mpf", projectUid: "du_g" });
    expect(useCollageStore.getState().missingItemIds).toEqual(["ig"]);
    dialogs.confirm.mockResolvedValueOnce(true);
    await expect(locateCollageItem("ig")).resolves.toBe(true);
    expect(dialogs.confirm).toHaveBeenLastCalledWith(expect.objectContaining({ title: "A different document" }));
    const items = useCollageStore.getState().items;
    expect(items.map((i) => [i.id, i.projectPath, i.projectUid])).toEqual([
      ["ig", "/far/other.mpf", "du_other"], ["ig2", "/exp/gone.mpf", "du_g"],
    ]);
    expect(useCollageStore.getState().missingItemIds).toEqual([]);
  });

  it("cancelled: nothing changes", async () => {
    await open([A], A.id, [figItem("ig", "/exp/gone.mpf", "du_g")]);
    useCollageStore.getState().setItemMissing("ig", true);
    await expect(locateCollageItem("ig")).resolves.toBe(false);
    expect(useCollageStore.getState().items[0].projectPath).toBe("/exp/gone.mpf");
    expect(useCollageStore.getState().missingItemIds).toEqual(["ig"]);
  });
});

describe("the close guard", () => {
  it("ignores collage figures whose file is gone", async () => {
    await open([A, B], B.id, [figItem("ia", A.path!, "du_a")]);
    await expect(closeDoc(A.id)).resolves.toBe(false);
    expect(dialogs.alert).toHaveBeenCalledWith(expect.objectContaining({ title: "Figure is in the collage" }));
    useCollageStore.getState().setItemMissing("ia", true);
    await expect(closeDoc(A.id)).resolves.toBe(true);
    expect(tabs().map((d) => d.id)).toEqual([B.id]);
  });
});
