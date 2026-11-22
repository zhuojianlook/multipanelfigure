/* ──────────────────────────────────────────────────────────
   docIdentity — which .mpf file is which document, across renames.

   Every .mpf carries a durable document uid ("du_…", the same value the
   Analysis workspace gives the open tab — analysisWorkspaceStore.docUids):
   its `doc.json` member, written by every figure save, and — in files
   that hold saved Analysis — its `workflows/manifest.json` docUid, the
   copy older files (and older builds' re-saves, which drop doc.json)
   keep. A file renamed or moved outside the app is still that document,
   so it can be found again by its uid (api.resolveMpf).

   A LEAF module (no store imports): the stores, session restore and the
   collage all use it.

     • The last-seen table (localStorage "mpfig_doc_registry_v1"): where
       each uid's file was last seen. One entry per path; a uid can have
       several (a rename, a copy). Newest first, capped. It only suggests
       where to look — the sidecar checks the uid inside a file before
       anything is pointed at it — so a stale entry is harmless.
     • The session map of each open tab's on-disk uid (fileUidOf): the uid
       the tab's file named when it was last read or written — its
       doc.json uid, else its manifest docUid; null: it names none;
       undefined: not known yet. It can differ from the tab's own uid: a
       second tab on a Finder copy keeps a fresh uid until its next save
       writes it. onIdentityChange tells listeners (the collage re-link)
       each time a tab's file is identified.
     • Both also keep the physical file ("dev:inode", the sidecar's
       file_id) when it is known: a rename or a move on one volume keeps it,
       a Finder / Dropbox copy gets a new one, and so does every write of
       this app (an atomic replace) — so each read and write records it
       again. A lookup sends it (buildRef) so that a single other file
       carrying the uid is followed only when it is that very file.
   ────────────────────────────────────────────────────────── */

import type { MpfResolveRef } from "../api/types";
import { dirnameOf } from "./workflowPersist";

// ── The last-seen table ─────────────────────────────────────

export const DOC_REGISTRY_KEY = "mpfig_doc_registry_v1";
/** Entries kept (the oldest go first). */
export const DOC_REGISTRY_CAP = 500;

/** One file seen carrying document `uid`. `t`: when (ms), for ordering.
 *  `fid`: the physical file then (the sidecar's file_id), when known. */
export interface DocFileEntry {
  uid: string;
  path: string;
  name?: string;
  t: number;
  fid?: string;
}

/** The storage the table lives in (tests inject their own). */
export type DocRegistryStorage = Pick<Storage, "getItem" | "setItem">;
let injected: DocRegistryStorage | null | undefined;

/** Tests: read and write the table in `s` (null: no storage at all;
 *  undefined: back to localStorage). */
export function setDocRegistryStorage(s: DocRegistryStorage | null | undefined): void {
  injected = s;
}

function storage(): DocRegistryStorage | null {
  if (injected !== undefined) return injected;
  try { return typeof localStorage === "undefined" ? null : localStorage; } catch { return null; }
}

/** The table, newest first. Anything unreadable reads as empty. */
function readTable(): DocFileEntry[] {
  const kv = storage();
  if (!kv) return [];
  let raw: unknown;
  try { raw = JSON.parse(kv.getItem(DOC_REGISTRY_KEY) || "null"); } catch { return []; }
  const list = raw && typeof raw === "object" ? (raw as { entries?: unknown }).entries : undefined;
  if (!Array.isArray(list)) return [];
  const out: DocFileEntry[] = [];
  const paths = new Set<string>();
  for (const e of list) {
    if (!e || typeof e !== "object") continue;
    const { uid, path, name, t, fid } = e as Record<string, unknown>;
    if (typeof uid !== "string" || !uid || typeof path !== "string" || !path || paths.has(path)) continue;
    if (typeof t !== "number" || !Number.isFinite(t)) continue;
    paths.add(path);
    out.push({
      uid, path, t,
      ...(typeof name === "string" && name ? { name } : {}),
      ...(typeof fid === "string" && fid ? { fid } : {}),
    });
  }
  return out.sort((a, b) => b.t - a.t);
}

function writeTable(entries: DocFileEntry[]): void {
  const kv = storage();
  if (!kv) return;
  try {
    kv.setItem(DOC_REGISTRY_KEY, JSON.stringify({ v: 1, entries: entries.slice(0, DOC_REGISTRY_CAP) }));
  } catch { /* storage full / disabled: the table only ever helps */ }
}

/** The file at `path` carries document `uid` (seen now): it replaces what
 *  the table had for that path and becomes the newest entry. `fileId`: the
 *  physical file now (none: not known). */
export function noteDocFile(uid: string, path: string, name?: string, fileId?: string | null): void {
  if (!uid || !path) return;
  const entries = readTable();
  const prev = entries.find((e) => e.path === path);
  const keepName = name || (prev && prev.uid === uid ? prev.name : undefined);
  // Strictly newer than every entry, so "newest first" holds within one ms.
  const t = Math.max(Date.now(), (entries[0]?.t ?? 0) + 1);
  const rest = entries.filter((e) => e.path !== path);
  writeTable([{ uid, path, t, ...(keepName ? { name: keepName } : {}), ...(fileId ? { fid: fileId } : {}) }, ...rest]);
}

/** The file at `path`, still carrying `uid`, was written again (a new
 *  physical file, `fileId`): the table's entry for it — only one naming that
 *  uid — records it. Its place in the table is unchanged. */
export function refreshDocFileId(path: string, uid: string, fileId: string | null | undefined): void {
  if (!path || !uid) return;
  const entries = readTable();
  const i = entries.findIndex((e) => e.path === path);
  if (i < 0 || entries[i].uid !== uid || (entries[i].fid ?? null) === (fileId || null)) return;
  const { fid: _old, ...rest } = entries[i];
  entries[i] = fileId ? { ...rest, fid: fileId } : rest;
  writeTable(entries);
}

/** Drop the table's entry for `path` (the file is known to be gone). */
export function forgetDocFile(path: string): void {
  const entries = readTable();
  const rest = entries.filter((e) => e.path !== path);
  if (rest.length !== entries.length) writeTable(rest);
}

/** Every path document `uid` was seen at, newest first. */
export function registryPathsOf(uid: string | null | undefined): string[] {
  if (!uid) return [];
  return readTable().filter((e) => e.uid === uid).map((e) => e.path);
}

/** The uid the file at `path` carried when last seen. */
export function registryUidAt(path: string | null | undefined): string | undefined {
  if (!path) return undefined;
  return readTable().find((e) => e.path === path)?.uid;
}

/** The physical file seen at `path` (when known). */
export function registryFileIdAt(path: string | null | undefined): string | undefined {
  if (!path) return undefined;
  return readTable().find((e) => e.path === path)?.fid;
}

/** A lookup for api.resolveMpf: the file last known at `path`, which should
 *  carry `uid`. The other paths the table saw that uid at are its candidates
 *  (the sidecar also looks in their folders and in `path`'s own); `extraDirs`
 *  are further folders worth a look (e.g. those of the other open documents).
 *  `fileId`: the physical file `path` last named, when known — a single
 *  other file carrying the uid is then followed only when it is that file. */
export function buildRef(
  key: string, path: string, uid?: string | null, extraDirs: readonly string[] = [],
  fileId?: string | null,
): MpfResolveRef {
  const ref: MpfResolveRef = { key, path };
  if (!uid) return ref;
  ref.uid = uid;
  if (fileId) ref.file_id = fileId;
  const candidates = registryPathsOf(uid).filter((p) => p !== path);
  if (candidates.length) ref.candidates = candidates;
  // Folders the sidecar looks in anyway are not repeated (they would only
  // take places under its folder cap).
  const implied = new Set([path, ...candidates].map(dirOf).filter(Boolean));
  const dirs: string[] = [];
  for (const d of extraDirs) if (d && !implied.has(d) && !dirs.includes(d)) dirs.push(d);
  if (dirs.length) ref.dirs = dirs;
  return ref;
}

function dirOf(p: string): string {
  return p ? dirnameOf(p) : "";
}

// ── Each open tab's on-disk uid (session only) ──────────────

/** A tab's file was identified: the file at `path` names `uid`. */
export interface IdentityChange {
  docId: string;
  uid: string;
  path: string | null;
}

const fileIds = new Map<string, { uid: string | null; path: string | null; fileId?: string }>();
const listeners = new Set<(e: IdentityChange) => void>();

/** Record the uid tab `docId`'s file names (null: none), read or written at
 *  `path`, and the physical file it was then (`fileId`; none: not known).
 *  Listeners hear of every uid that is new for the tab (or now at another
 *  path). */
export function setFileUid(
  docId: string, uid: string | null, path: string | null = null, fileId?: string | null,
): void {
  const prev = fileIds.get(docId);
  fileIds.set(docId, { uid, path, ...(fileId ? { fileId } : {}) });
  if (!uid || (prev && prev.uid === uid && prev.path === path)) return;
  for (const cb of [...listeners]) {
    try { cb({ docId, uid, path }); } catch (e) { console.error("[docIdentity] listener failed", e); }
  }
}

/** The uid tab `docId`'s file named when last read / written: a uid, null
 *  (none), or undefined (not known — never read). */
export function fileUidOf(docId: string): string | null | undefined {
  return fileIds.get(docId)?.uid;
}

/** The physical file tab `docId`'s file was when last read / written (the
 *  sidecar's file_id), when known. */
export function fileIdOf(docId: string): string | undefined {
  return fileIds.get(docId)?.fileId;
}

/** Tab `docId`'s file was written again, naming the same uid: only the
 *  physical file it now is changes (none: not known). Nothing to tell
 *  listeners; a tab whose file was never identified is left alone. */
export function setFileIdOf(docId: string, fileId: string | null | undefined): void {
  const cur = fileIds.get(docId);
  if (!cur) return;
  const { fileId: _old, ...rest } = cur;
  fileIds.set(docId, fileId ? { ...rest, fileId } : rest);
}

/** Forget what tab `docId`'s file names (it closed, or names another file). */
export function clearFileUid(docId: string): void {
  fileIds.delete(docId);
}

/** Hear whenever an open tab's file is identified. Returns the unsubscribe. */
export function onIdentityChange(cb: (e: IdentityChange) => void): () => void {
  listeners.add(cb);
  return () => { listeners.delete(cb); };
}

// ── Errors ──────────────────────────────────────────────────

/** The sidecar's answer for a path with no file behind it ("Project file not
 *  found: …", a 404 — apiJson keeps only the message). */
export function isMissingFileError(e: unknown): boolean {
  const msg = e instanceof Error ? e.message : String(e ?? "");
  return msg.includes("Project file not found");
}

/** api.writeWorkflows' answer when the file at the path names another
 *  document than the one expected (the sidecar's 409 {other_document}):
 *  nothing was written. */
export function isOtherDocumentError(e: unknown): boolean {
  const msg = e instanceof Error ? e.message : String(e ?? "");
  return msg.includes("now holds another document");
}

/** Tests: forget every tab's on-disk uid and use localStorage again. The
 *  last-seen table itself lives in that storage (tests clear it), and
 *  listeners stay: modules subscribe once, when they load. */
export function resetDocIdentityForTests(): void {
  fileIds.clear();
  injected = undefined;
}
