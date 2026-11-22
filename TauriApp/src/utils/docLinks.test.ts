/* docLinks (uid plan §2.5): open tabs and collage figures follow a .mpf
   renamed or moved outside the app. Launch seeding looks each collage
   figure's file up by its uid (moved → follows, gone → marked missing, no
   phantom tab, a figure without a uid takes the file's), a render that
   finds no file looks again and retries once, collage figures take the uid
   a re-save writes (not a Save As copy's), a file identified anywhere
   re-links the figures marked missing that carry its uid, and twin
   detection picks by seq, never tab position. The sidecar is a fake
   in-memory "disk" of path → the uid its doc.json carries. */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { MpfResolveRef, MpfResolveResult, WorkflowsReadResponse } from "../api/types";

const files = vi.hoisted(() => new Map<string, string | null>());
/** The physical file at a path (the sidecar's file_id: dev:inode) — only
 *  for files a test gives one; a rename (move) keeps it, a copy or a write
 *  gets a new one. */
const fids = vi.hoisted(() => new Map<string, string>());
const health = vi.hoisted(() => ({ up: true, calls: 0, upAfter: 0 }));
const mockApi = vi.hoisted(() => ({
  readWorkflows: vi.fn(),
  writeWorkflows: vi.fn(),
  saveProject: vi.fn(),
  loadProject: vi.fn(),
  pathsExist: vi.fn(),
  resolveMpf: vi.fn(),
  updateConfig: vi.fn(async () => ({})),
  getPreview: vi.fn(async () => ({ image: "p", width: 1, height: 1, format: "png" })),
  getPanelPreview: vi.fn(async () => ({ image: "t" })),
}));
vi.mock("../api/client", () => ({
  api: mockApi,
  checkHealth: vi.fn(async () => { health.calls++; return health.up && health.calls > health.upAfter; }),
  lastHealthError: "",
  settleConfigWrites: vi.fn(async () => {}),
  default: vi.fn(),
}));
vi.mock("../components/shared/ConfirmDialog", () => ({
  confirm: vi.fn(async () => true), confirmThree: vi.fn(async () => "cancel"), alert: vi.fn(async () => true),
}));

import { useCollageStore, type CollageItem, type DocTab } from "../store/collageStore";
import { useFigureStore } from "../store/figureStore";
import { useAnalysisWorkspaceStore, resetAnalysisWorkspaceForTests } from "../store/analysisWorkspaceStore";
import { figKey, nsKey } from "./analysisKeys";
import { fileIdOf, fileUidOf, registryFileIdAt, registryUidAt, setFileUid } from "./docIdentity";
import {
  findRenamedOpenTab, locateOpenDoc, relocateOpenDoc, repointOpenDoc, repointOpenDocs, resolveRefs,
  seedTabsFromCollage, withItemPath,
} from "./docLinks";

const EMPTY: WorkflowsReadResponse = {
  manifest: null, blobs: {}, unsupported: false, version: null, uploads: [], renamed: {},
  missing_uploads: [], failed_uploads: [],
};
const notFound = (p: string) => new Error(`API error: "Project file not found: ${p}"`);
let minted = 0;

/** A fake /api/fs/mpf-resolve with the sidecar's rules: the candidates, then
 *  (unless `scan: false`) the files in the folders of the path, of the
 *  candidates and in `dirs` — no other folder. */
function fakeResolve(refs: MpfResolveRef[], opts: { scan?: boolean } = {}): Record<string, MpfResolveResult> {
  const dirOf = (p: string) => p.slice(0, p.lastIndexOf("/"));
  const out: Record<string, MpfResolveResult> = {};
  for (const r of refs) {
    const exists = files.has(r.path);
    const there = exists ? files.get(r.path) ?? null : null;
    const fid = (p: string) => (fids.has(p) ? { file_id: fids.get(p) } : {});
    if (exists && (!r.uid || !there || there === r.uid)) { out[r.key] = { status: "ok", path: r.path, exists, uid: there, ...fid(r.path) }; continue; }
    if (!r.uid) { out[r.key] = { status: "missing", path: null, exists, uid: null }; continue; }
    const cands = r.candidates ?? [];
    const folders = new Set([r.path, ...cands].map(dirOf).concat(r.dirs ?? []));
    const scanned = opts.scan === false ? [] : [...files.keys()].filter((p) => folders.has(dirOf(p)));
    const fromCand = cands.filter((p) => p !== r.path && files.get(p) === r.uid);
    const hits = [...new Set([...fromCand, ...scanned])].filter((p) => p !== r.path && files.get(p) === r.uid);
    // Several files carry the uid: none is picked, not even a candidate. A
    // single one that is not the ref's physical file (a copy): not either.
    if (hits.length === 1 && r.file_id && fids.get(hits[0]) !== r.file_id) {
      out[r.key] = { status: "ambiguous", path: null, exists, uid: null, matches: hits };
    } else if (hits.length === 1) out[r.key] = { status: "moved", path: hits[0], exists, uid: r.uid, ...fid(hits[0]) };
    else if (hits.length > 1) out[r.key] = { status: "ambiguous", path: null, exists, uid: null, matches: hits };
    else out[r.key] = exists ? { status: "ok", path: r.path, exists, uid: there, ...fid(r.path) } : { status: "missing", path: null, exists, uid: null };
  }
  return out;
}

function move(from: string, to: string) {
  files.set(to, files.get(from) ?? null);
  files.delete(from);
  const f = fids.get(from);
  fids.delete(from);
  if (f) fids.set(to, f);
}
let inode = 0;
/** A new physical file at `path` carrying `uid` (a Finder copy, say). */
function put(path: string, uid: string | null) {
  files.set(path, uid);
  fids.set(path, `ino_${++inode}`);
}
async function flush(n = 80) { for (let i = 0; i < n; i++) await Promise.resolve(); }
let quiet: { mockRestore: () => void };

function figItem(id: string, path: string, extra: Partial<CollageItem> = {}): CollageItem {
  return {
    id, kind: "figure", src: "data:", x: 0, y: 0, w: 10, h: 10, naturalW: 10, naturalH: 10,
    rotation: 0, z: 1, name: path.split("/").pop()!.replace(/\.mpf$/, ""), projectPath: path, ...extra,
  } as CollageItem;
}
const item = (id: string) => useCollageStore.getState().items.find((i) => i.id === id)!;
const tabs = () => useCollageStore.getState().openDocs;
const missing = () => useCollageStore.getState().missingItemIds;
const ws = () => useAnalysisWorkspaceStore.getState();
const names = () => JSON.parse(localStorage.getItem("mpfig.source_names") || "{}");

/** Fresh session: `docs` open (the first active), their files read; `items` in the collage. */
async function open(docs: DocTab[], items: CollageItem[] = []) {
  useCollageStore.setState({ openDocs: [], activeDocId: null, items: [], missingItemIds: [] });
  resetAnalysisWorkspaceForTests();
  useCollageStore.setState({ openDocs: docs, activeDocId: docs[0]?.id ?? null, items });
  await flush();
}

beforeEach(() => {
  vi.clearAllMocks();
  // Reads of files that are gone are reported on the console: expected here.
  quiet = vi.spyOn(console, "error").mockImplementation(() => {});
  files.clear();
  fids.clear();
  localStorage.clear();
  health.up = true;
  health.calls = 0;
  health.upAfter = 0;
  minted = 0;
  mockApi.readWorkflows.mockImplementation(async (p: string) => {
    if (!files.has(p)) throw notFound(p);
    return { ...EMPTY, doc_uid: files.get(p) ?? null, ...(fids.has(p) ? { file_id: fids.get(p) } : {}) };
  });
  mockApi.writeWorkflows.mockImplementation(async (p: string) => {
    if (!files.has(p)) throw notFound(p);
    // Every write replaces the file (a new physical file).
    if (fids.has(p)) fids.set(p, `ino_${++inode}`);
    return {
      ok: true, size_bytes: 1, workflows_bytes: 1, missing_uploads: [], fast_path: true,
      ...(fids.has(p) ? { file_id: fids.get(p) } : {}),
    };
  });
  // The sidecar's uid rule: a copy of a file that is still there is a new
  // document; else the uid sent, else the file's own.
  mockApi.saveProject.mockImplementation(async (path: string, opts?: { carryFrom?: string | null; docUid?: string }) => {
    const src = opts?.carryFrom === undefined ? path : (opts.carryFrom ?? "");
    const want = opts?.docUid;
    let uid: string;
    if (src && src !== path && files.has(src) && (files.get(src) == null || files.get(src) === (want ?? files.get(src)))) {
      uid = `du_minted_${++minted}`;
    } else uid = want ?? files.get(path) ?? `du_minted_${++minted}`;
    files.set(path, uid);
    // A test that tracks physical files: the save puts a new one in place.
    if (fids.size) fids.set(path, `ino_${++inode}`);
    return { ok: true, path, doc_uid: uid, uid_minted: uid !== want, ...(fids.has(path) ? { file_id: fids.get(path) } : {}) };
  });
  mockApi.pathsExist.mockImplementation(async (paths: string[]) => Object.fromEntries(paths.map((p) => [p, files.has(p)])));
  mockApi.resolveMpf.mockImplementation(async (refs: MpfResolveRef[], opts?: { scan?: boolean }) => fakeResolve(refs, opts));
  useFigureStore.setState({
    unsaved: false, currentProjectPath: null,
    config: { rows: 1, cols: 1, panels: [[{}]] } as never,
  });
});
afterEach(() => {
  useCollageStore.setState({ openDocs: [], activeDocId: null, items: [], missingItemIds: [] });
  resetAnalysisWorkspaceForTests();
  vi.useRealTimers();
  quiet.mockRestore();
});

describe("resolveRefs", () => {
  it("never rejects: an older sidecar is asked by path, and a path it can't check counts as there", async () => {
    mockApi.resolveMpf.mockRejectedValueOnce(new Error("404"));
    files.set("/exp/a.mpf", "du_a");
    const r = await resolveRefs([{ key: "a", path: "/exp/a.mpf", uid: "du_a" }, { key: "b", path: "/exp/b.mpf", uid: "du_b" }]);
    expect(r.a).toMatchObject({ status: "ok", path: "/exp/a.mpf" });
    expect(r.b).toMatchObject({ status: "missing", path: null });
    mockApi.resolveMpf.mockRejectedValueOnce(new Error("down"));
    mockApi.pathsExist.mockRejectedValueOnce(new Error("down"));
    expect((await resolveRefs([{ key: "b", path: "/exp/b.mpf" }])).b.status).toBe("ok");
  });
});

describe("launch: seeding document tabs from the collage", () => {
  it("moved → the figure follows and its tab opens there; gone → marked missing, no tab; a figure without a uid takes the file's", async () => {
    files.set("/exp/a2.mpf", "du_a");                         // a.mpf renamed while the app was closed
    files.set("/exp/c.mpf", "du_c");
    const oldKey = nsKey(figKey({ id: "", path: "/exp/a.mpf" }), "r0c0_i0");
    localStorage.setItem("mpfig.source_names", JSON.stringify({ [oldKey]: "Ctrl" }));
    await open([], [
      figItem("ia", "/exp/a.mpf", { projectUid: "du_a", name: "a – Figure 2", figureId: "fig_2" }),
      figItem("ib", "/exp/b.mpf", { projectUid: "du_b" }),
      figItem("ic", "/exp/c.mpf"),
    ]);
    useCollageStore.setState({ openDocs: [{ id: "doc_initial", path: null, name: "Untitled_1", seq: 0 }], activeDocId: "doc_initial" });
    await seedTabsFromCollage();
    expect(mockApi.resolveMpf).toHaveBeenCalledTimes(1);
    expect(tabs().map((d) => d.path)).toEqual([null, "/exp/a2.mpf", "/exp/c.mpf"]);
    expect(item("ia")).toMatchObject({ projectPath: "/exp/a2.mpf", projectUid: "du_a", name: "a2 – Figure 2" });
    expect(item("ib").projectPath).toBe("/exp/b.mpf");
    expect(missing()).toEqual(["ib"]);
    expect(item("ic").projectUid).toBe("du_c");
    // Its labels moved with it, and the table knows where it is.
    expect(names()).toEqual({ [nsKey(figKey({ id: "", path: "/exp/a2.mpf" }), "r0c0_i0")]: "Ctrl" });
    expect(registryUidAt("/exp/a2.mpf")).toBe("du_a");
    expect(registryUidAt("/exp/a.mpf")).toBeUndefined();
  });

  it("waits for the sidecar to answer before looking anything up", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    health.upAfter = 2;
    files.set("/exp/a2.mpf", "du_a");
    await open([], [figItem("ia", "/exp/a.mpf", { projectUid: "du_a" })]);
    const p = seedTabsFromCollage();
    await vi.advanceTimersByTimeAsync(2500);
    await p;
    expect(health.calls).toBe(3);
    expect(tabs().map((d) => d.path)).toEqual(["/exp/a2.mpf"]);
  });

  it("a file still at the path, its uid found only in copies: its tab opens there, never marked missing", async () => {
    files.set("/exp/a.mpf", "du_w");                          // another document has the name now
    files.set("/exp/a copy 1.mpf", "du_a");
    files.set("/exp/a copy 2.mpf", "du_a");
    await open([], [figItem("ia", "/exp/a.mpf", { projectUid: "du_a" })]);
    await seedTabsFromCollage();
    expect(tabs().map((d) => d.path)).toEqual(["/exp/a.mpf"]);
    expect(missing()).toEqual([]);
    expect(item("ia").projectPath).toBe("/exp/a.mpf");
  });

  it("a sidecar that never answers: a tab per path, as before", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    health.up = false;
    await open([], [figItem("ia", "/exp/a.mpf", { projectUid: "du_a" }), figItem("ib", "/exp/b.mpf")]);
    const p = seedTabsFromCollage();
    await vi.advanceTimersByTimeAsync(15_000);
    await p;
    expect(health.calls).toBe(10);
    expect(mockApi.resolveMpf).not.toHaveBeenCalled();
    expect(tabs().map((d) => d.path)).toEqual(["/exp/a.mpf", "/exp/b.mpf"]);
  });
});

describe("withItemPath", () => {
  const render = vi.fn(async (p: string) => { if (!files.has(p)) throw notFound(p); return `R:${p}`; });

  it("a file renamed since: found by its uid, the figure and the tab on it follow, the call runs again there", async () => {
    files.set("/exp/a.mpf", "du_a");
    await open([{ id: "doc_a", path: "/exp/a.mpf", name: "a", seq: 1 }], [figItem("ia", "/exp/a.mpf")]);
    move("/exp/a.mpf", "/exp/renamed.mpf");
    render.mockClear();
    await expect(withItemPath("ia", render)).resolves.toBe("R:/exp/renamed.mpf");
    expect(render.mock.calls.map((c) => c[0])).toEqual(["/exp/a.mpf", "/exp/renamed.mpf"]);
    expect(item("ia")).toMatchObject({ projectPath: "/exp/renamed.mpf", projectUid: "du_a", name: "renamed" });
    expect(tabs()[0]).toMatchObject({ id: "doc_a", path: "/exp/renamed.mpf", name: "renamed", seq: 1 });
    expect(ws().docUids.doc_a).toBe("du_a");
  });

  it("found nowhere: marked missing and the error stands; another error never looks", async () => {
    await open([], [figItem("ia", "/exp/gone.mpf", { projectUid: "du_gone" })]);
    await expect(withItemPath("ia", render)).rejects.toThrow(/Project file not found/);
    expect(missing()).toEqual(["ia"]);
    mockApi.resolveMpf.mockClear();
    await expect(withItemPath("ia", async () => { throw new Error("boom"); })).rejects.toThrow("boom");
    expect(mockApi.resolveMpf).not.toHaveBeenCalled();
    // The file is back: a call that works clears the mark.
    files.set("/exp/gone.mpf", "du_gone");
    await withItemPath("ia", render);
    expect(missing()).toEqual([]);
  });
});

describe("collage figures and the uid their file carries", () => {
  it("a re-save stamps the file's uid on its figures; a Save As copy leaves them on the original", async () => {
    files.set("/exp/a.mpf", null);                             // a file saved before uids
    await open([{ id: "doc_a", path: "/exp/a.mpf", name: "a", seq: 1 }], [figItem("ia", "/exp/a.mpf")]);
    useFigureStore.setState({ currentProjectPath: "/exp/a.mpf" });
    const ua = ws().docUids.doc_a;
    await useFigureStore.getState().saveProject("/exp/a.mpf");
    expect(files.get("/exp/a.mpf")).toBe(ua);
    expect(item("ia").projectUid).toBe(ua);

    await useFigureStore.getState().saveProject("/exp/copy.mpf");
    expect(ws().docUids.doc_a).not.toBe(ua);                  // the copy is a new document
    expect(item("ia")).toMatchObject({ projectPath: "/exp/a.mpf", projectUid: ua });
  });

  it("a file identified anywhere re-links the figures marked missing that carry its uid", async () => {
    files.set("/exp/elsewhere/fig.mpf", "du_x");
    await open([], [
      figItem("gone", "/exp/fig.mpf", { projectUid: "du_x" }),
      figItem("other", "/exp/other.mpf", { projectUid: "du_x" }),   // not missing: left alone
    ]);
    useCollageStore.getState().setItemMissing("gone", true);
    useCollageStore.getState().docAdd({ id: "doc_f", path: "/exp/elsewhere/fig.mpf" });
    await flush();
    expect(fileUidOf("doc_f")).toBe("du_x");
    expect(item("gone").projectPath).toBe("/exp/elsewhere/fig.mpf");
    expect(missing()).toEqual([]);
    expect(item("other").projectPath).toBe("/exp/other.mpf");
  });
});

describe("repointOpenDoc", () => {
  it("moves every tab on the path (seq and a given name kept), the live path, the figures and the table", async () => {
    files.set("/exp/a.mpf", "du_a");
    await open([
      { id: "doc_a", path: "/exp/a.mpf", name: "a", seq: 4 },
      { id: "doc_a2", path: "/exp/a.mpf", name: "My figure", seq: 2 },
      { id: "doc_b", path: "/exp/b.mpf", name: "b", seq: 3 },
    ], [figItem("ia", "/exp/a.mpf", { name: "Custom" })]);
    useFigureStore.setState({ currentProjectPath: "/exp/a.mpf" });
    repointOpenDoc("/exp/a.mpf", "/exp/x/a-final.mpf", "du_a");
    expect(tabs()).toEqual([
      { id: "doc_a", path: "/exp/x/a-final.mpf", name: "a-final", seq: 4 },
      { id: "doc_a2", path: "/exp/x/a-final.mpf", name: "My figure", seq: 2 },
      { id: "doc_b", path: "/exp/b.mpf", name: "b", seq: 3 },
    ]);
    expect(useFigureStore.getState().currentProjectPath).toBe("/exp/x/a-final.mpf");
    expect(item("ia")).toMatchObject({ projectPath: "/exp/x/a-final.mpf", projectUid: "du_a", name: "Custom" });
    expect(registryUidAt("/exp/x/a-final.mpf")).toBe("du_a");
  });

  it("two tabs on one file whose known uids differ still move together", async () => {
    files.set("/exp/new.mpf", "du_h");
    await open([
      { id: "t1", path: "/exp/one.mpf", name: "one", seq: 1 },
      { id: "t2", path: "/exp/one.mpf", name: "one", seq: 2 },
    ]);
    setFileUid("t1", "du_u");
    setFileUid("t2", "du_h");
    repointOpenDoc("/exp/one.mpf", "/exp/new.mpf", "du_h");
    expect(tabs().map((d) => d.path)).toEqual(["/exp/new.mpf", "/exp/new.mpf"]);
  });
});

describe("names chained or swapped while the app was closed", () => {
  // fig.mpf (U) → fig_old.mpf, then fig_v2.mpf (W) → fig.mpf: each figure,
  // tab and label follows its own file, whatever order the figures are in.
  it("a chain, in both figure orders: figures, tabs, labels and the table land on their own file", async () => {
    for (const order of [["iw", "iu"], ["iu", "iw"]]) {
      files.clear();
      localStorage.clear();
      files.set("/exp/fig_old.mpf", "du_u");
      files.set("/exp/fig.mpf", "du_w");
      localStorage.setItem("mpfig.source_names", JSON.stringify({
        "p:/exp/fig.mpf::r0c0_i0": "U-label", "p:/exp/fig_v2.mpf::r0c0_i0": "W-label", "p:/exp/fig_v2.mpf::r0c1_i0": "W2",
      }));
      const mk: Record<string, CollageItem> = {
        iw: figItem("iw", "/exp/fig_v2.mpf", { projectUid: "du_w" }),
        iu: figItem("iu", "/exp/fig.mpf", { projectUid: "du_u" }),
      };
      await open([], order.map((k) => mk[k]));
      await seedTabsFromCollage();
      expect(item("iw")).toMatchObject({ projectPath: "/exp/fig.mpf", projectUid: "du_w" });
      expect(item("iu")).toMatchObject({ projectPath: "/exp/fig_old.mpf", projectUid: "du_u" });
      expect(names()).toEqual({
        "p:/exp/fig_old.mpf::r0c0_i0": "U-label", "p:/exp/fig.mpf::r0c0_i0": "W-label", "p:/exp/fig.mpf::r0c1_i0": "W2",
      });
      expect(tabs().map((d) => d.path).sort()).toEqual(["/exp/fig.mpf", "/exp/fig_old.mpf"]);
      expect(registryUidAt("/exp/fig.mpf")).toBe("du_w");
      expect(registryUidAt("/exp/fig_old.mpf")).toBe("du_u");
    }
  });

  it("a swap a ↔ b: each figure and its labels follow its own file", async () => {
    files.set("/exp/a.mpf", "du_b");
    files.set("/exp/b.mpf", "du_a");
    localStorage.setItem("mpfig.source_names", JSON.stringify({ "p:/exp/a.mpf::k": "A", "p:/exp/b.mpf::k": "B" }));
    await open([], [figItem("ia", "/exp/a.mpf", { projectUid: "du_a" }), figItem("ib", "/exp/b.mpf", { projectUid: "du_b" })]);
    await seedTabsFromCollage();
    expect(item("ia")).toMatchObject({ projectPath: "/exp/b.mpf", projectUid: "du_a" });
    expect(item("ib")).toMatchObject({ projectPath: "/exp/a.mpf", projectUid: "du_b" });
    expect(names()).toEqual({ "p:/exp/b.mpf::k": "A", "p:/exp/a.mpf::k": "B" });
    expect(tabs().map((d) => d.path).sort()).toEqual(["/exp/a.mpf", "/exp/b.mpf"]);
  });

  it("a swap a ↔ b with a tab on each, their uids known from before it: each tab goes with its own file", async () => {
    files.set("/exp/a.mpf", "du_a");
    files.set("/exp/b.mpf", "du_b");
    await open([{ id: "ta", path: "/exp/a.mpf", name: "a", seq: 1 }, { id: "tb", path: "/exp/b.mpf", name: "b", seq: 2 }]);
    localStorage.setItem("mpfig.source_names", JSON.stringify({ "p:/exp/a.mpf::k": "A", "p:/exp/b.mpf::k": "B" }));
    files.set("/exp/a.mpf", "du_b");                         // swapped, not read since
    files.set("/exp/b.mpf", "du_a");
    repointOpenDocs([
      { from: "/exp/a.mpf", to: "/exp/b.mpf", uid: "du_a" },
      { from: "/exp/b.mpf", to: "/exp/a.mpf", uid: "du_b" },
    ]);
    expect(tabs().map((d) => [d.id, d.path])).toEqual([["ta", "/exp/b.mpf"], ["tb", "/exp/a.mpf"]]);
    expect(names()).toEqual({ "p:/exp/b.mpf::k": "A", "p:/exp/a.mpf::k": "B" });
    expect(registryUidAt("/exp/a.mpf")).toBe("du_b");
    expect(registryUidAt("/exp/b.mpf")).toBe("du_a");
  });

  it("a stale batch (looked up before seeding re-pointed everything) leaves the path's new document alone", async () => {
    files.set("/exp/fig_old.mpf", "du_u");
    files.set("/exp/fig.mpf", "du_w");
    await open([], [figItem("iu", "/exp/fig.mpf", { projectUid: "du_u" }), figItem("iw", "/exp/fig_v2.mpf", { projectUid: "du_w" })]);
    await seedTabsFromCollage();
    await flush();
    const before = tabs().map((d) => [d.id, d.path]);
    repointOpenDocs([
      { from: "/exp/fig.mpf", to: "/exp/fig_old.mpf", uid: "du_u" },
      { from: "/exp/fig_v2.mpf", to: "/exp/fig.mpf", uid: "du_w" },
    ]);
    expect(tabs().map((d) => [d.id, d.path])).toEqual(before);
    expect(item("iw")).toMatchObject({ projectPath: "/exp/fig.mpf", projectUid: "du_w" });
    expect(item("iu")).toMatchObject({ projectPath: "/exp/fig_old.mpf", projectUid: "du_u" });
  });
});

describe("a Finder copy and its original carry one uid", () => {
  const A: DocTab = { id: "doc_a", path: "/exp/a.mpf", name: "a", seq: 1 };
  const C: DocTab = { id: "doc_c", path: "/exp/a copy.mpf", name: "a copy", seq: 2 };

  it("the original's file deleted: its tab never moves onto the copy's open tab's file", async () => {
    files.set(A.path!, "du_a");
    files.set(C.path!, "du_a");
    await open([A, C]);
    expect(ws().docUids.doc_a).toBe("du_a");                    // not held: the original
    files.delete(A.path!);
    await expect(relocateOpenDoc("doc_a")).resolves.toBeNull();
    expect(tabs().map((d) => [d.id, d.path])).toEqual([[A.id, A.path], [C.id, C.path]]);
  });

  it("the copy deleted: a figure placed from it is marked missing, its tab never joins the original's", async () => {
    files.set(A.path!, "du_a");
    files.set(C.path!, "du_a");
    await open([A, C], [figItem("ic", C.path!, { projectUid: "du_a" })]);
    files.delete(C.path!);
    const seen: string[] = [];
    await expect(withItemPath("ic", async (p) => { seen.push(p); if (!files.has(p)) throw notFound(p); return p; }))
      .rejects.toThrow(/not found/);
    expect(seen).toEqual([C.path]);
    expect(tabs().map((d) => [d.id, d.path])).toEqual([[A.id, A.path], [C.id, C.path]]);
    expect(missing()).toEqual(["ic"]);
  });

  it("a held copy never follows into the original, its tab closed or not", async () => {
    files.set(A.path!, "du_a");
    files.set(C.path!, "du_a");
    await open([A, C]);
    useCollageStore.setState({ openDocs: tabs().filter((d) => d.id !== A.id), activeDocId: C.id });
    await flush();
    files.delete(C.path!);
    await expect(relocateOpenDoc("doc_c")).resolves.toBeNull();
    expect(tabs().map((d) => d.path)).toEqual([C.path]);
  });

  it("a copy opened alone, the original seen on this machine: deleted or renamed, never followed into the original", async () => {
    files.set(A.path!, "du_a");
    files.set(C.path!, "du_a");
    localStorage.setItem("mpfig_doc_registry_v1", JSON.stringify({ v: 1, entries: [{ uid: "du_a", path: A.path, t: 1 }] }));
    await open([C]);
    expect(ws().docUids.doc_c).toBe("du_a");                    // adopted: the original isn't open
    files.delete(C.path!);
    await expect(relocateOpenDoc("doc_c")).resolves.toBeNull();
    files.set("/exp/a v2.mpf", "du_a");                         // renamed instead: two files carry it
    await expect(relocateOpenDoc("doc_c")).resolves.toBeNull();
    expect(tabs().map((d) => d.path)).toEqual([C.path]);
  });
});

describe("a Finder copy of a file since moved away or deleted: another physical file, never followed unasked", () => {
  const F: DocTab = { id: "doc_f", path: "/drafts/foo.mpf", name: "foo", seq: 1 };
  const BACKUP = "/drafts/foo backup.mpf";

  it("Save's lookup offers it without picking it; a write's relocation and File > Open never follow it", async () => {
    put(F.path!, "du_f");
    await open([F]);
    expect(fileIdOf(F.id)).toBe(fids.get(F.path!));
    // Saved in the app since: a new physical file, recorded.
    useFigureStore.setState({ currentProjectPath: F.path });
    await useFigureStore.getState().saveProject(F.path!);
    await flush();
    expect(fileIdOf(F.id)).toBe(fids.get(F.path!));
    expect(registryFileIdAt(F.path!)).toBe(fids.get(F.path!));
    put(BACKUP, "du_f");                                      // a Finder duplicate: same bytes, same uid
    files.delete(F.path!);                                    // the original moved where no lookup looks
    fids.delete(F.path!);
    await expect(locateOpenDoc(F.id)).resolves.toEqual({
      status: "ambiguous", path: F.path, exists: false, matches: [BACKUP],
    });
    await expect(relocateOpenDoc(F.id)).resolves.toBeNull();
    expect(await findRenamedOpenTab(BACKUP)).toBeNull();      // opened, it gets a tab of its own
    expect(tabs().map((d) => [d.id, d.path, d.name])).toEqual([[F.id, F.path, "foo"]]);
    expect(registryUidAt(BACKUP)).toBeUndefined();
  });

  it("a rename (the same physical file) is still followed, after the app wrote it too", async () => {
    put(F.path!, "du_f");
    await open([F]);
    const first = fids.get(F.path!);
    await ws().persistDocWorkflows(F.id, { force: true });   // a write: a new physical file, recorded
    expect(fids.get(F.path!)).not.toBe(first);
    expect(fileIdOf(F.id)).toBe(fids.get(F.path!));
    move(F.path!, "/drafts/foo final.mpf");
    expect((await findRenamedOpenTab("/drafts/foo final.mpf"))?.id).toBe(F.id);
    await expect(relocateOpenDoc(F.id)).resolves.toBe("/drafts/foo final.mpf");
    expect(registryFileIdAt("/drafts/foo final.mpf")).toBe(fids.get("/drafts/foo final.mpf"));
  });
});

describe("relocateOpenDoc / findRenamedOpenTab", () => {
  it("relocates by the uid the tab's file carried; with none known it doesn't look", async () => {
    files.set("/exp/a.mpf", "du_a");
    files.set("/other/n.mpf", null);
    await open([{ id: "doc_a", path: "/exp/a.mpf", name: "a", seq: 1 }, { id: "doc_n", path: "/other/n.mpf", name: "n", seq: 2 }]);
    // Moved next to another open document: its folder is looked in too.
    move("/exp/a.mpf", "/other/a.mpf");
    await expect(relocateOpenDoc("doc_a")).resolves.toBe("/other/a.mpf");
    expect(tabs()[0].path).toBe("/other/a.mpf");
    expect(mockApi.resolveMpf.mock.lastCall![0][0]).toMatchObject({ uid: "du_a", dirs: ["/other"] });
    move("/other/n.mpf", "/other/n2.mpf");
    mockApi.resolveMpf.mockClear();
    await expect(relocateOpenDoc("doc_n")).resolves.toBeNull();
    expect(mockApi.resolveMpf).not.toHaveBeenCalled();
  });

  it("a tab on a Finder copy (held: a uid of its own, not yet saved into its file) never follows into the original", async () => {
    files.set("/exp/a.mpf", "du_a");
    files.set("/exp/a copy.mpf", "du_a");
    await open([{ id: "doc_a", path: "/exp/a.mpf", name: "a", seq: 1 }, { id: "doc_c", path: "/exp/a copy.mpf", name: "a copy", seq: 2 }]);
    expect(fileUidOf("doc_c")).toBe("du_a");
    expect(ws().docUids.doc_c).not.toBe("du_a");
    files.delete("/exp/a copy.mpf");
    await expect(relocateOpenDoc("doc_c")).resolves.toBeNull();
    expect(tabs().map((d) => d.path)).toEqual(["/exp/a.mpf", "/exp/a copy.mpf"]);
  });

  it("a tab whose file was already gone when it was read: found by the table's uid, then read where it is now", async () => {
    localStorage.setItem("mpfig_doc_registry_v1", JSON.stringify({ v: 1, entries: [{ uid: "du_a", path: "/exp/a.mpf", t: 1 }] }));
    files.set("/exp/a2.mpf", "du_a");
    await open([{ id: "doc_a", path: "/exp/a.mpf", name: "a", seq: 1 }]);
    expect(ws().hydrated.doc_a).toBe("failed");
    expect(fileUidOf("doc_a")).toBeUndefined();
    await expect(relocateOpenDoc("doc_a")).resolves.toBe("/exp/a2.mpf");
    await flush();
    expect(mockApi.readWorkflows).toHaveBeenLastCalledWith("/exp/a2.mpf");
    expect(ws().hydrated.doc_a).toBe("done");
    expect(ws().docUids.doc_a).toBe("du_a");
    expect(fileUidOf("doc_a")).toBe("du_a");
  });

  it("the open tab whose file was renamed — the one opened first, not the leftmost; a copy is not one", async () => {
    await open([
      { id: "late", path: "/exp/old2.mpf", name: "old2", seq: 7 },
      { id: "early", path: "/exp/old1.mpf", name: "old1", seq: 3 },
    ]);
    setFileUid("late", "du_a", "/exp/old2.mpf");
    setFileUid("early", "du_a", "/exp/old1.mpf");
    files.set("/exp/new.mpf", "du_a");
    expect((await findRenamedOpenTab("/exp/new.mpf"))?.id).toBe("early");
    expect(mockApi.resolveMpf.mock.lastCall?.[1]).toEqual({ scan: false });
    files.set("/exp/old1.mpf", "du_a");                       // its old file is still there: a copy
    expect((await findRenamedOpenTab("/exp/new.mpf"))?.id).toBe("late");
    files.set("/exp/old2.mpf", "du_a");
    expect(await findRenamedOpenTab("/exp/new.mpf")).toBeNull();
  });
});
