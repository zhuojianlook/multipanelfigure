/* ──────────────────────────────────────────────────────────
   docLinks — keeps open tabs and collage figures pointed at their
   .mpf files when a file is renamed or moved outside the app.

   A file is recognised by the document uid inside it (utils/docIdentity);
   api.resolveMpf finds a uid's file again: at its path, among the paths
   the last-seen table saw it at, or in the folders of those. When a file
   is found somewhere else, everything that named its old path follows it
   (repointOpenDoc):
     • the open tabs on it — analysisWorkspaceStore.docMoved: the same
       document, no new uid, no re-read, its labels move — and the
       builder's current path when one of them is the live document;
     • the collage figures placed from it (path, uid, name);
     • the last-seen table.

   Entry points: a load or write that found no file (relocateOpenDoc,
   withItemPath), opening a renamed file while its old tab is still open
   (findRenamedOpenTab), a save of a tab whose file is gone (locateOpenDoc,
   for projectNav's ensureDocFile), launch (seedTabsFromCollage), and any
   read or save that identifies a file — collage figures marked missing
   that carry its uid are re-linked to it.

   Every step is a plain await on api.* or a synchronous store write, so
   all of it can run inside a document op (it never queues one itself).
   ────────────────────────────────────────────────────────── */

import { api, checkHealth } from "../api/client";
import type { MpfResolveRef, MpfResolveResult, MpfResolveStatus } from "../api/types";
import { useCollageStore, type DocTab } from "../store/collageStore";
import { useFigureStore } from "../store/figureStore";
import { useAnalysisWorkspaceStore, rekeyAnalysisStorage } from "../store/analysisWorkspaceStore";
import { docKeyBase } from "./workflowOwnership";
import { basenameOf, dirnameOf } from "./workflowPersist";
import {
  buildRef, fileIdOf, fileUidOf, forgetDocFile, isMissingFileError, noteDocFile, onIdentityChange,
  refreshDocFileId, registryFileIdAt, registryUidAt,
} from "./docIdentity";

// ── Looking files up ────────────────────────────────────────

/** Where each file in `refs` is now (api.resolveMpf), by ref key. Never
 *  rejects. A ref the sidecar could not answer (it is unreachable, or an
 *  older one without the lookup) is checked by path only — and a path that
 *  can't even be checked counts as there, like pathsExist's own fallback:
 *  a lookup that failed never drops or re-points anything. */
export async function resolveRefs(
  refs: MpfResolveRef[], opts: { scan?: boolean } = {},
): Promise<Record<string, MpfResolveResult>> {
  if (!refs.length) return {};
  const out: Record<string, MpfResolveResult> = {};
  try {
    const got = await api.resolveMpf(refs, opts);
    if (got && typeof got === "object") Object.assign(out, got);
  } catch { /* by path only, below */ }
  const left = refs.filter((r) => !isResult(out[r.key]));
  if (left.length) {
    const paths = [...new Set(left.map((r) => r.path).filter(Boolean))];
    let exists: Record<string, boolean> = {};
    try { exists = (await api.pathsExist(paths)) || {}; } catch { exists = {}; }
    for (const r of left) {
      const there = !!r.path && exists[r.path] !== false;
      out[r.key] = { status: there ? "ok" : "missing", path: there ? r.path : null, exists: there, uid: null };
    }
  }
  return out;
}

function isResult(r: unknown): r is MpfResolveResult {
  return !!r && typeof r === "object" && typeof (r as { status?: unknown }).status === "string";
}

/** The uid open tab `docId`'s file is known to carry: the one it named when
 *  it was last read or written, else — never read (a cold tab, or a read
 *  that failed) — the one the last-seen table saw at `path`. Undefined when
 *  the file named none: it can't be looked for by uid. */
export function knownFileUid(docId: string, path: string | null | undefined): string | undefined {
  const f = fileUidOf(docId);
  if (f) return f;
  if (f === null) return undefined;
  return registryUidAt(path);
}

/** The physical file (docIdentity: the sidecar's file_id) open tab `docId`'s
 *  file was when last read or written — as knownFileUid, the last-seen
 *  table's for `path` when the tab never identified its file. Undefined: not
 *  known (a lookup then follows a single hit as before). */
export function knownFileId(docId: string, path: string | null | undefined): string | undefined {
  if (fileUidOf(docId) !== undefined) return fileIdOf(docId);
  return registryFileIdAt(path);
}

/** The physical file last seen at `path` (collage figures): what a tab on
 *  it knows, else the last-seen table's. */
function fileIdAt(path: string): string | undefined {
  for (const d of useCollageStore.getState().openDocs) {
    if (d.path !== path) continue;
    const f = knownFileId(d.id, path);
    if (f) return f;
  }
  return registryFileIdAt(path);
}

function openDoc(docId: string): DocTab | undefined {
  return useCollageStore.getState().openDocs.find((d) => d.id === docId);
}

/** Folders of the open documents other than `docId` (extra places a lookup
 *  may look in: files of one project tend to sit together). */
function otherDocDirs(docId?: string): string[] {
  const out: string[] = [];
  for (const d of useCollageStore.getState().openDocs) {
    if (d.id === docId || !d.path) continue;
    const dir = dirnameOf(d.path);
    if (dir && !out.includes(dir)) out.push(dir);
  }
  return out;
}

/** A file's name without ".mpf" — what tabs and collage figures are named after. */
function stem(path: string): string {
  return basenameOf(path).replace(/\.mpf$/i, "");
}

/** A collage figure's name after its file was renamed `from` → `to`: the
 *  file-name part changes ("fig" or "fig – Figure 2", as Add to Collage
 *  names it); a name the user gave stays. */
function renamedItemName(name: string, from: string, to: string): string {
  if (!from || from === to) return name;
  if (name === from) return to;
  if (name.startsWith(`${from} – `)) return `${to}${name.slice(from.length)}`;
  return name;
}

/** `path` and `other` were both seen carrying `uid` (the last-seen table):
 *  two files of one document — a Finder or Dropbox copy and its original —
 *  not one file that moved. A move forgets its old path (repointOpenDoc,
 *  loadOne, a save to a new path whose old file is gone), so a real rename
 *  never leaves both. Such a hit is never followed silently: the other file
 *  is another document's now. */
export function knownCopies(path: string, other: string, uid: string | null | undefined): boolean {
  return !!uid && registryUidAt(path) === uid && registryUidAt(other) === uid;
}

/** Re-pointing `oldPath` → `newPath` would move the open tab(s) on `oldPath`
 *  onto a file another open tab already has. That tab is a document of its
 *  own (a Finder copy and its original carry the same uid until one of them
 *  is saved): the two would share its keys, and a save would write over the
 *  other tab's file. So it is not done; the file counts as not found. Only
 *  when a tab sits on `oldPath`: collage figures alone may follow. */
export function ontoAnotherTab(oldPath: string, newPath: string): boolean {
  const docs = useCollageStore.getState().openDocs;
  return newPath !== oldPath && docs.some((d) => d.path === oldPath) && docs.some((d) => d.path === newPath);
}

// ── Following a file to its new path ────────────────────────

/** The file at `oldPath` is at `newPath` now (renamed or moved outside the
 *  app, found again by `uid`). Synchronous, so it can run inside a doc op:
 *   • every open tab on `oldPath` moves with it (they share its keys) —
 *     analysisWorkspaceStore.docMoved; the builder's current path follows
 *     when one of them is live. With no tab on it, the labels this machine
 *     keeps under its old path move on their own;
 *   • collage figures placed from it point at `newPath` (and carry `uid`),
 *     and are no longer marked missing;
 *   • the last-seen table records it at `newPath`.
 *  Another document may have taken `oldPath` meanwhile (names chained or
 *  swapped — fig.mpf → fig_old.mpf, then fig_v2.mpf → fig.mpf — re-pointed
 *  one after the other, or a lookup made before another re-point): what
 *  carries that document's uid stays. Its tabs on `oldPath` (none of them
 *  this file's) keep the path and its labels; figures carrying another uid
 *  stay. Returns whether the labels under `oldPath` followed the file.
 *  `opts.reread`: the file at `newPath` is not known to be the one the tabs
 *  read (Locate… picked a file whose uid can't be told): its saved Analysis
 *  is read there before anything is written to it (docMoved).
 *  `opts.fileId`: the physical file at `newPath` (the lookup's file_id) —
 *  recorded in the last-seen table; omitted, the one recorded for `oldPath`
 *  carries over (a rename keeps it), unless `reread`. `opts.tabs`: only
 *  these tabs on `oldPath` may move, judged by the uid given with each
 *  (repointOpenDocs: what they knew before any hop of a batch ran).
 *  Tab order and `seq` are untouched. */
export function repointOpenDoc(
  oldPath: string, newPath: string, uid?: string | null,
  opts: { reread?: boolean; fileId?: string | null; tabs?: ReadonlyArray<{ id: string; known?: string }> } = {},
): boolean {
  if (!oldPath || !newPath || oldPath === newPath) return false;
  const cs = useCollageStore.getState();
  const given = opts.tabs;
  const onOld = cs.openDocs.filter((d) => d.path === oldPath && (!given || given.some((t) => t.id === d.id)));
  const u = uid || onOld.map((d) => fileUidOf(d.id)).find((x): x is string => !!x) || registryUidAt(oldPath);
  const other = (k: string | null | undefined) => !!u && !!k && k !== u;
  const known = onOld.map((d) => (given ? given.find((t) => t.id === d.id)?.known : knownFileUid(d.id, oldPath)));
  const taken = !!u && known.some(other) && !known.includes(u);
  const moving = taken ? [] : onOld;
  const liveMoves = moving.some((d) => d.id === cs.activeDocId);
  const fid = opts.fileId !== undefined ? opts.fileId
    : !opts.reread && u && registryUidAt(oldPath) === u ? registryFileIdAt(oldPath) : undefined;
  let labels = false;
  if (moving.length) {
    useAnalysisWorkspaceStore.getState().docMoved(oldPath, newPath, {
      reread: opts.reread, ...(given ? { ids: moving.map((d) => d.id) } : {}),
    });
    labels = true;
  } else if (!taken && !cs.openDocs.some((d) => d.path === newPath)) {
    // Not open (a collage figure, a restored session): its labels (source
    // names, measurement group labels, source folders) are kept under its
    // path — they move now, before a tab reads it at the new one. Not when a
    // tab already has the new path: its labels are the live ones.
    rekeyAnalysisStorage(docKeyBase({ id: "", path: oldPath }), docKeyBase({ id: "", path: newPath }), false);
    labels = true;
  }
  if (liveMoves) useFigureStore.getState().setCurrentProjectPath(newPath);
  const from = stem(oldPath);
  const to = stem(newPath);
  for (const it of useCollageStore.getState().items) {
    if (it.kind !== "figure" || it.projectPath !== oldPath) continue;
    if (other(it.projectUid) || (taken && !it.projectUid)) continue;
    useCollageStore.getState().updateItem(it.id, {
      projectPath: newPath,
      name: renamedItemName(it.name, from, to),
      ...(u ? { projectUid: u } : {}),
    });
    useCollageStore.getState().setItemMissing(it.id, false);
  }
  if (u) {
    // The old path no longer holds it.
    if (registryUidAt(oldPath) === u) forgetDocFile(oldPath);
    const named = useCollageStore.getState().openDocs.find((d) => d.path === newPath);
    noteDocFile(u, newPath, named?.name, fid);
  }
  return labels;
}

/** Several files found at new paths at once (launch seeding, session
 *  restore), each re-pointed with repointOpenDoc. A path is left before
 *  another file arrives at it (fig.mpf → fig_old.mpf before fig_v2.mpf →
 *  fig.mpf), so the tabs on it are still its own file's. The labels kept
 *  under a path carry no uid to tell two documents apart, so each file's
 *  wait under a key no file has while all of them move, then land where
 *  repointOpenDoc sent them — names that swapped included. So do the tabs:
 *  each move takes only the tabs that were on its path before any hop ran,
 *  judged by what they knew then — in a swap, the tabs the first hop
 *  delivered to a path stay there when the second hop leaves it. */
export function repointOpenDocs(
  moves: Array<{ from: string; to: string; uid?: string | null; fileId?: string | null }>,
): void {
  const left = moves.filter((m) => m.from && m.to && m.from !== m.to);
  if (!left.length) return;
  const docs0 = useCollageStore.getState().openDocs;
  const carry = left.map((m) => docs0.filter((d) => d.path === m.from)
    .map((d) => ({ id: d.id, known: knownFileUid(d.id, m.from) })));
  const base = (path: string) => docKeyBase({ id: "", path });
  const parked = left.map((m, i) => {
    const tmp = docKeyBase({ id: `\u0000moving-${i}`, path: null });
    rekeyAnalysisStorage(base(m.from), tmp, false);
    return tmp;
  });
  const pending = left.map((_, i) => i);
  const landing: Array<[string, string]> = [];
  while (pending.length) {
    // One whose new path no other file still has to leave; a cycle (swapped
    // names) has none — any of it then.
    const k = pending.findIndex((i) => !pending.some((j) => j !== i && left[j].from === left[i].to));
    const i = pending.splice(Math.max(k, 0), 1)[0];
    const m = left[i];
    const moved = repointOpenDoc(m.from, m.to, m.uid, {
      tabs: carry[i], ...(m.fileId !== undefined ? { fileId: m.fileId } : {}),
    });
    landing.push([parked[i], base(moved ? m.to : m.from)]);
  }
  for (const [tmp, to] of landing) rekeyAnalysisStorage(tmp, to, false);
}

/** Where open tab `docId`'s file is now — the tab has a path. ok: at its
 *  path (or no telling: the lookup failed). moved: found elsewhere by its
 *  uid, and the tab — with everything on its old path — re-pointed there
 *  (repointOpenDoc); `path` is the new one. missing: not found (a file
 *  carrying no uid is only looked for at its path; among several files
 *  carrying it, the very file the tab had — its file id — is followed).
 *  ambiguous: several other files carry its uid (`matches`) and none is
 *  that very file, so none is picked — or the only one is not the very
 *  file the tab had (another physical file: a copy, it may be). `path` of
 *  the last two is where it was, and `exists` says whether a
 *  file (another document's) is there now — also "missing" when the file at
 *  `path` names another uid than the one this tab's file named and its own
 *  is nowhere: never written over unasked. Null: no such tab, or it has no
 *  path. */
export async function locateOpenDoc(
  docId: string,
): Promise<{ status: MpfResolveStatus; path: string; matches?: string[]; exists?: boolean } | null> {
  const doc = openDoc(docId);
  if (!doc?.path) return null;
  const path = doc.path;
  const uid = knownFileUid(docId, path);
  const r = (await resolveRefs([buildRef("doc", path, uid, otherDocDirs(docId), knownFileId(docId, path))])).doc;
  const now = openDoc(docId);
  if (!now?.path) return null;
  // Re-pointed while we looked (another lookup found it first).
  if (now.path !== path) return { status: now.path === r?.path ? "moved" : "ok", path: now.path };
  // The uid the tab's file named when it was last read or written (not the
  // last-seen table's: a stale entry must not raise a prompt).
  const f = fileUidOf(docId);
  // Another document now has the tab's file name (replaced outside the app),
  // and the tab's own file is nowhere: a save must not write over it unasked.
  if (r?.status === "ok" && f && r.uid && r.uid !== f) return { status: "missing", path, exists: true };
  if (!r || r.status === "ok") return { status: "ok", path };
  // Never followed silently into a file that may be another document's —
  // the user is asked (ensureDocFile), and a load or write that needed it
  // fails as it would have:
  //   • from a held tab: the held rule gave it a uid of its own, so its file
  //     is a copy of another document's (a Finder copy, its uid not yet
  //     written into its file), whether or not the original is still open;
  //   • into a file the last-seen table saw carrying the uid next to this
  //     tab's own (knownCopies: the original, or a copy of it);
  //   • onto a file another open tab has (ontoAnotherTab) — the original,
  //     its own file deleted, into its copy's, say.
  // (A single hit that is not the very file the tab had — a Finder copy of a
  // file since deleted — comes back "ambiguous" from the sidecar: asked too.)
  const held = !!f && useAnalysisWorkspaceStore.getState().docUids[docId] !== f;
  if (r.status === "moved" && r.path && !held && !knownCopies(path, r.path, uid)
      && !ontoAnotherTab(path, r.path)) {
    repointOpenDoc(path, r.path, uid, { fileId: r.file_id ?? undefined });
    return { status: "moved", path: r.path };
  }
  return {
    status: r.status === "ambiguous" ? "ambiguous" : "missing", path, exists: !!r.exists,
    ...(r.matches?.length ? { matches: r.matches } : {}),
  };
}

/** Open tab `docId`'s file is not at its path any more (a load or write
 *  said "Project file not found"): find it by its uid; found in one place,
 *  re-point the tab (and everything on its path) there and return the new
 *  path. Null: its uid is unknown, or it is nowhere to be found (or in
 *  several places). analysisWorkspaceStore's persist and projectNav's
 *  document load retry at the path returned. */
export async function relocateOpenDoc(docId: string): Promise<string | null> {
  const doc = openDoc(docId);
  if (!doc?.path || !knownFileUid(docId, doc.path)) return null;
  const at = await locateOpenDoc(docId);
  return at?.status === "moved" ? at.path : null;
}

/** `path` is being opened: is it the file of an open tab whose own path no
 *  longer exists — renamed or moved outside the app while the tab was open?
 *  Recognised by the uid in the file (one lookup, no folder scan). Several
 *  such tabs: the one opened first (lowest `seq`, never tab position).
 *  Null when the file names no uid, no open tab knows it, or the tab's old
 *  file is still there (then `path` is a copy: a tab of its own) — and, as
 *  locateOpenDoc, never for a file that may be another document's: another
 *  physical file than the tab had (a Finder copy of a file since gone), one
 *  the last-seen table saw carrying the uid next to the tab's own file
 *  (knownCopies), or the original's file for a held tab (a copy's). Such a
 *  file opens in a tab of its own. */
export async function findRenamedOpenTab(path: string): Promise<DocTab | null> {
  if (!path) return null;
  const others = useCollageStore.getState().openDocs.filter((d) => d.path && d.path !== path);
  if (!others.length) return null;
  const paths = [...new Set(others.map((d) => d.path as string))];
  const refs: MpfResolveRef[] = [{ key: "pick", path }, ...paths.map((p, i) => ({ key: `t${i}`, path: p }))];
  const res = await resolveRefs(refs, { scan: false });
  const pick = res.pick;
  if (!pick || pick.status !== "ok" || !pick.uid) return null;
  const gone = new Set(paths.filter((_, i) => res[`t${i}`]?.exists === false));
  const docUids = useAnalysisWorkspaceStore.getState().docUids;
  const twins = useCollageStore.getState().openDocs.filter((d) => {
    if (!d.path || d.path === path || !gone.has(d.path) || knownFileUid(d.id, d.path) !== pick.uid) return false;
    const fid = knownFileId(d.id, d.path);
    if (pick.file_id && fid && pick.file_id !== fid) return false;          // another physical file
    const f = fileUidOf(d.id);
    const held = !!f && docUids[d.id] !== f;
    return !knownCopies(d.path, path, pick.uid) && !(held && registryUidAt(path) === pick.uid);
  });
  if (!twins.length) return null;
  const seqOf = (d: DocTab) => d.seq ?? Number.MAX_SAFE_INTEGER;
  return twins.reduce((a, b) => (seqOf(b) < seqOf(a) ? b : a));
}

// ── Collage figures ─────────────────────────────────────────

/** Mark every collage figure placed from `path` as having no file to be
 *  found (collageStore.missingItemIds: the close guard ignores them, and a
 *  file identified later with their uid re-links them) — or, `missing`
 *  false, as found again. */
export function markItemsMissingAt(path: string, missing = true): void {
  const cs = useCollageStore.getState();
  for (const it of cs.items) {
    if (it.kind === "figure" && it.projectPath === path) cs.setItemMissing(it.id, missing);
  }
}

/** Collage figure `itemId` alone now points at `path` (the user located its
 *  file, and it is another document's, or a file another tab has — the
 *  figures and tabs on its old path stay): its name follows the file's
 *  name, it carries the uid the file names (none when that can't be told),
 *  and it is no longer marked missing. */
export function relinkCollageItem(itemId: string, path: string, uid?: string | null): void {
  const cs = useCollageStore.getState();
  const it = cs.items.find((i) => i.id === itemId);
  if (!it || it.kind !== "figure") return;
  if (it.projectPath !== path) {
    cs.updateItem(itemId, {
      projectPath: path,
      name: it.projectPath ? renamedItemName(it.name, stem(it.projectPath), stem(path)) : it.name,
      projectUid: uid || undefined,
    });
  } else if (uid && it.projectUid !== uid) {
    cs.updateItem(itemId, { projectUid: uid });
  }
  cs.setItemMissing(itemId, false);
}

/** The uid collage figure(s) at `path` should find: one they carry, else
 *  what an open tab on it knows, else the last-seen table. */
export function itemUidAt(path: string, itemId?: string): string | undefined {
  const items = useCollageStore.getState().items;
  const own = itemId ? items.find((i) => i.id === itemId)?.projectUid : undefined;
  if (own) return own;
  const carried = items.find((i) => i.kind === "figure" && i.projectPath === path && i.projectUid)?.projectUid;
  if (carried) return carried;
  for (const d of useCollageStore.getState().openDocs) {
    if (d.path !== path) continue;
    const u = knownFileUid(d.id, d.path);
    if (u) return u;
  }
  return registryUidAt(path);
}

/** Run `fn` with collage figure `itemId`'s .mpf path (a render, an element
 *  listing). When the sidecar says the file is not there: look for it by
 *  its uid; found in one place, the figure (and any tab on that file)
 *  follows it and `fn` runs once more at the new path. Not found: the
 *  figure is marked missing and the error stands. */
export async function withItemPath<T>(itemId: string, fn: (path: string) => Promise<T>): Promise<T> {
  const path = useCollageStore.getState().items.find((i) => i.id === itemId)?.projectPath;
  if (!path) throw new Error("This collage figure has no .mpf file.");
  let out: T;
  try {
    out = await fn(path);
  } catch (e) {
    if (!isMissingFileError(e)) throw e;
    const found = await findItemFile(itemId, path);
    if (!found) throw e;
    out = await fn(found);
  }
  useCollageStore.getState().setItemMissing(itemId, false);
  return out;
}

/** Collage figure `itemId`'s file is not at `path`: where it is now (and
 *  everything on `path` re-pointed there), or null (marked missing). */
async function findItemFile(itemId: string, path: string): Promise<string | null> {
  const uid = itemUidAt(path, itemId);
  const r = uid ? (await resolveRefs([buildRef("item", path, uid, otherDocDirs(), fileIdAt(path))])).item : undefined;
  const cur = useCollageStore.getState().items.find((i) => i.id === itemId);
  if (!cur?.projectPath) return null;
  if (cur.projectPath !== path) return cur.projectPath;      // re-pointed while we looked
  // Not into a copy's or the original's file (knownCopies), nor with a tab
  // on `path` onto a file another tab has (ontoAnotherTab): marked missing.
  if (r?.status === "moved" && r.path && !knownCopies(path, r.path, uid) && !ontoAnotherTab(path, r.path)) {
    repointOpenDoc(path, r.path, uid, { fileId: r.file_id ?? undefined });
    return r.path;
  }
  markItemsMissingAt(path);
  return null;
}

/** Launch: the sidecar may still be starting. Looked for this many times,
 *  this far apart, before the figures are seeded by path alone. */
const SEED_HEALTH_TRIES = 10;
const SEED_HEALTH_MS = 1000;
let seeding: Promise<void> | null = null;

/** Launch: a document tab for every .mpf the collage's figures were placed
 *  from — so they open in one click — each looked up by its uid first (one
 *  api.resolveMpf call):
 *   • ok: a tab on it; a figure without a uid takes the file's;
 *   • moved (renamed / moved while the app was closed): the figures follow
 *     it (repointOpenDoc) and the tab opens at the new path;
 *   • not followed (a copy, several, another tab's file) but a file is
 *     still at the path: a tab on it, as before files had uids;
 *   • missing / ambiguous with nothing at the path: no tab — a phantom tab
 *     would only fail to load; the figures are marked missing.
 *  While the sidecar doesn't answer (it is starting), waits up to ~10 s,
 *  then seeds a tab per path as before. Only adds tabs; never reorders or
 *  removes any. Calls made while one runs share it. */
export function seedTabsFromCollage(): Promise<void> {
  if (!seeding) {
    seeding = seedNow()
      .catch((e) => { console.error("[docLinks] seeding tabs from the collage failed", e); })
      .finally(() => { seeding = null; });
  }
  return seeding;
}

function figurePaths(): string[] {
  const out: string[] = [];
  for (const it of useCollageStore.getState().items) {
    if (it.kind === "figure" && it.projectPath && !out.includes(it.projectPath)) out.push(it.projectPath);
  }
  return out;
}

async function seedNow(): Promise<void> {
  if (!figurePaths().length) return;
  let up = false;
  for (let i = 0; i < SEED_HEALTH_TRIES; i++) {
    if (await checkHealth().catch(() => false)) { up = true; break; }
    if (i < SEED_HEALTH_TRIES - 1) await new Promise((r) => setTimeout(r, SEED_HEALTH_MS));
  }
  if (!up) {
    // Not answering: a tab per path, as before files had uids (a file that
    // is gone says so when its tab is opened).
    for (const p of figurePaths()) useCollageStore.getState().docEnsure(p);
    return;
  }
  const paths = figurePaths();
  const uids = paths.map((p) => itemUidAt(p));
  const dirs = otherDocDirs();
  const res = await resolveRefs(paths.map((p, i) => buildRef(`f${i}`, p, uids[i], dirs, fileIdAt(p))));
  // The moves go together (repointOpenDocs: names chained or swapped among
  // them land right), the tabs after them, in figure order.
  const moves: Array<{ from: string; to: string; uid?: string | null; fileId?: string | null }> = [];
  const tabsAt: string[] = [];
  // The paths these moves leave: a tab there moves away in the same batch,
  // so arriving at one is not landing on another tab's file.
  const leaving = new Set(paths.filter((_, i) => res[`f${i}`]?.status === "moved"));
  paths.forEach((p, i) => {
    const r = res[`f${i}`];
    const cs = useCollageStore.getState();
    const at = cs.items.filter((it) => it.kind === "figure" && it.projectPath === p);
    if (!at.length || !r) return;                            // removed meanwhile
    if (r.status === "ok") {
      // A figure placed before files carried uids takes the file's; one whose
      // file now carries another uid (its document is nowhere else) heals.
      for (const it of at) {
        if (r.uid && it.projectUid !== r.uid) cs.updateItem(it.id, { projectUid: r.uid });
        cs.setItemMissing(it.id, false);
      }
      // The file as it is now (written elsewhere since, say): what tells it
      // from a copy later — only for the uid the table has for it.
      if (r.uid && r.file_id) refreshDocFileId(p, r.uid, r.file_id);
      tabsAt.push(p);
    } else if (r.status === "moved" && r.path && !knownCopies(p, r.path, uids[i])
        && !(ontoAnotherTab(p, r.path) && !leaving.has(r.path))) {
      // (Not into a copy's or the original's file, nor onto another tab's.)
      moves.push({ from: p, to: r.path, uid: r.uid || uids[i], ...(r.file_id ? { fileId: r.file_id } : {}) });
      tabsAt.push(r.path);
    } else if (r.exists) {
      // Not followed, but a file is still at the path (another document's
      // now, its id found only in a copy or several): it stays, as before
      // files had uids — never marked missing while it is there.
      for (const it of at) cs.setItemMissing(it.id, false);
      tabsAt.push(p);
    } else {
      for (const it of at) cs.setItemMissing(it.id, true);
    }
  });
  repointOpenDocs(moves);
  for (const p of tabsAt) useCollageStore.getState().docEnsure(p);
}

/* Any read or save that identifies a file (docIdentity.setFileUid — a tab
   hydrated, saved, or its first workflows write) re-links the collage
   figures marked missing that carry its uid: opening a moved file anywhere
   repairs the collage. */
onIdentityChange(({ uid, path }) => {
  if (!path) return;
  const cs = useCollageStore.getState();
  if (!cs.missingItemIds.length) return;
  for (const it of cs.items) {
    if (it.kind !== "figure" || it.projectUid !== uid || !cs.missingItemIds.includes(it.id)) continue;
    if (it.projectPath && it.projectPath !== path) {
      cs.updateItem(it.id, { projectPath: path, name: renamedItemName(it.name, stem(it.projectPath), stem(path)) });
    }
    cs.setItemMissing(it.id, false);
  }
});
