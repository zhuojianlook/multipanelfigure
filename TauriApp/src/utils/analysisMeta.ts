/* ──────────────────────────────────────────────────────────
   analysisMeta — the Analysis panel's labels that belong to one
   document, as its .mpf manifest stores them (`meta`, plan §1.11 /
   step 8). Pure: the workspace store reads and writes localStorage.

   The labels live in the app's localStorage, keyed by source key /
   measurement row — keys that embed the document's key base
   ("p:<path>" / "d:<docId>"):
     sourceNames        "<base>[#fig]::<raw>" | "img:<upload>" → name
     measurementGroups  "<base>[#fig]|panel|ann|reading"     → label
     sourceGroups       [{ id, name, sourceKeys: [same keys] }]  folders

   A document's `meta` holds the entries whose key is one of ITS keys,
   DOCUMENT-RELATIVE: the base replaced by "@" ("@::r0c0_i0",
   "@#fig-2|R1C1|0|0"), so the file can move. Upload keys ("img:<name>")
   stay as they are and belong to the uploads' home document — the one
   whose file embeds them. Folders keep only that document's keys (a
   folder with none of them is left out).

   On read the file's labels are merged into localStorage (which stays
   the mirror the panel reads): a label the file has overrides the local
   one; folders merge by id (else by name — "Direct upload" is minted
   per session) and a key the file places in a folder leaves any other
   folder, so each source still lives in one place. A file never writes
   a key that isn't one of its document's.
   ────────────────────────────────────────────────────────── */

import { rekeyKey } from "./workflowOwnership";

/** The key base document-relative keys are written under. */
export const META_REL_BASE = "@";
/** The folder standalone Analysis uploads are parked in. */
export const DIRECT_UPLOAD_FOLDER = "Direct upload";
const UPLOAD_PREFIX = "img:";

/** A source-library folder (the panel's SourceGroup). */
export type SourceFolder = {
  id: string;
  name: string;
  sourceKeys: string[];
};

/** Everything the panel keeps in localStorage, absolute keys. */
export interface AnalysisLabels {
  sourceNames: Record<string, string>;
  measurementGroups: Record<string, string>;
  sourceGroups: SourceFolder[];
}

/** One document's labels as its manifest stores them (relative keys). A type
 *  (not an interface) so it is a PersistState.meta entry as it is. */
export type DocMeta = {
  sourceNames?: Record<string, string>;
  measurementGroups?: Record<string, string>;
  sourceGroups?: SourceFolder[];
};

export const uploadKey = (name: string): string => `${UPLOAD_PREFIX}${name}`;

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);

/** A well-formed relative key: "@" plus what rekeyKey accepts after a base. */
function isRelKey(k: string): boolean {
  return k.startsWith(META_REL_BASE) && rekeyKey(k, META_REL_BASE, "\u0001") !== k;
}

/** `key` in the document's relative form — when it is one of the document's
 *  (under `base`, or one of its upload keys); null otherwise. */
export function relKeyOf(key: string, base: string, uploadKeys: ReadonlySet<string>): string | null {
  if (!key) return null;
  if (key.startsWith(UPLOAD_PREFIX)) return uploadKeys.has(key) ? key : null;
  const rel = rekeyKey(key, base, META_REL_BASE);
  return rel !== key ? rel : null;
}

/** A relative key back under `base`; an upload key only when the document
 *  owns it. null for anything else — never another document's key. */
export function absKeyOf(rel: string, base: string, uploadKeys: ReadonlySet<string>): string | null {
  if (!rel) return null;
  if (rel.startsWith(UPLOAD_PREFIX)) return uploadKeys.has(rel) ? rel : null;
  if (!isRelKey(rel)) return null;
  return rekeyKey(rel, META_REL_BASE, base);
}

function folderList(v: unknown): SourceFolder[] {
  if (!Array.isArray(v)) return [];
  const out: SourceFolder[] = [];
  for (const g of v) {
    if (!isObj(g) || typeof g.id !== "string" || typeof g.name !== "string" || !Array.isArray(g.sourceKeys)) continue;
    out.push({ id: g.id, name: g.name, sourceKeys: g.sourceKeys.filter((k): k is string => typeof k === "string") });
  }
  return out;
}

function nonEmpty<T extends object>(o: T): T | undefined {
  return Object.keys(o).length ? o : undefined;
}

function compact(m: DocMeta): DocMeta | undefined {
  const out: DocMeta = {};
  if (m.sourceNames && Object.keys(m.sourceNames).length) out.sourceNames = m.sourceNames;
  if (m.measurementGroups && Object.keys(m.measurementGroups).length) out.measurementGroups = m.measurementGroups;
  if (m.sourceGroups && m.sourceGroups.length) out.sourceGroups = m.sourceGroups;
  return nonEmpty(out);
}

/** localStorage contents in a known shape (anything malformed dropped). */
export function asLabels(sourceNames: unknown, measurementGroups: unknown, sourceGroups: unknown): AnalysisLabels {
  const rec = (v: unknown): Record<string, string> => {
    if (!isObj(v)) return {};
    const out: Record<string, string> = {};
    for (const [k, x] of Object.entries(v)) if (typeof x === "string") out[k] = x;
    return out;
  };
  return { sourceNames: rec(sourceNames), measurementGroups: rec(measurementGroups), sourceGroups: folderList(sourceGroups) };
}

/** Document `base`'s labels in manifest form (undefined: it has none).
 *  `uploadKeys` are the upload keys it owns (it is their home). */
export function docMetaOf(labels: AnalysisLabels, base: string, uploadKeys: ReadonlySet<string>): DocMeta | undefined {
  const pick = (rec: Record<string, string>) => {
    const out: Record<string, string> = {};
    for (const [k, v] of Object.entries(rec)) {
      if (typeof v !== "string" || !v.trim()) continue;
      const r = relKeyOf(k, base, uploadKeys);
      if (r != null) out[r] = v;
    }
    return out;
  };
  const sourceGroups: SourceFolder[] = [];
  for (const g of labels.sourceGroups) {
    const keys: string[] = [];
    for (const k of g.sourceKeys) {
      const r = relKeyOf(k, base, uploadKeys);
      if (r != null && !keys.includes(r)) keys.push(r);
    }
    if (keys.length) sourceGroups.push({ id: g.id, name: g.name, sourceKeys: keys });
  }
  return compact({
    sourceNames: pick(labels.sourceNames),
    measurementGroups: pick(labels.measurementGroups),
    sourceGroups,
  });
}

/** A manifest's `meta` as read, in a known shape: only well-formed relative
 *  keys and the upload keys the document owns (`uploadKeys`). */
export function readDocMeta(raw: unknown, uploadKeys: ReadonlySet<string>): DocMeta | undefined {
  if (!isObj(raw)) return undefined;
  const ok = (k: string) => (k.startsWith(UPLOAD_PREFIX) ? uploadKeys.has(k) : isRelKey(k));
  const pick = (v: unknown) => {
    const out: Record<string, string> = {};
    if (!isObj(v)) return out;
    for (const [k, x] of Object.entries(v)) if (typeof x === "string" && x.trim() && ok(k)) out[k] = x;
    return out;
  };
  const sourceGroups: SourceFolder[] = [];
  for (const g of folderList(raw.sourceGroups)) {
    const keys = g.sourceKeys.filter((k, i, a) => ok(k) && a.indexOf(k) === i);
    if (keys.length) sourceGroups.push({ ...g, sourceKeys: keys });
  }
  return compact({ sourceNames: pick(raw.sourceNames), measurementGroups: pick(raw.measurementGroups), sourceGroups });
}

/** The form a document's labels are signed in (docSig): folder ids dropped
 *  (minted per machine / session — a folder merged by name is the same
 *  folder), folders and their keys sorted. Order-insensitive, so merging a
 *  file into the local folders doesn't make it look changed. */
export function metaSigForm(meta: DocMeta | undefined): Record<string, unknown> | undefined {
  if (!meta) return undefined;
  const out: Record<string, unknown> = {};
  if (meta.sourceNames) out.sourceNames = meta.sourceNames;
  if (meta.measurementGroups) out.measurementGroups = meta.measurementGroups;
  if (meta.sourceGroups && meta.sourceGroups.length) {
    out.sourceGroups = meta.sourceGroups
      .map((g) => ({ name: g.name, keys: [...g.sourceKeys].sort() }))
      .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : a.keys.join("\u0000") < b.keys.join("\u0000") ? -1 : 1));
  }
  return nonEmpty(out);
}

/** Upload keys in a manifest's `meta` renamed ({old: new} names, from the
 *  read response) — one pass, no chaining. Anything else passes through. */
export function renameMetaUploads(meta: unknown, renamed: Readonly<Record<string, string>> | undefined): unknown {
  if (!isObj(meta) || !renamed || !Object.keys(renamed).length) return meta;
  const map = new Map<string, string>();
  for (const [o, n] of Object.entries(renamed)) if (o && n && o !== n) map.set(uploadKey(o), uploadKey(n));
  if (!map.size) return meta;
  const fn = (k: string) => map.get(k) ?? k;
  const out: Record<string, unknown> = { ...meta };
  for (const f of ["sourceNames", "measurementGroups"] as const) {
    const rec = meta[f];
    if (isObj(rec)) out[f] = Object.fromEntries(Object.entries(rec).map(([k, v]) => [fn(k), v]));
  }
  if (Array.isArray(meta.sourceGroups)) {
    out.sourceGroups = meta.sourceGroups.map((g) => (isObj(g) && Array.isArray(g.sourceKeys)
      ? { ...g, sourceKeys: g.sourceKeys.map((k) => (typeof k === "string" ? fn(k) : k)) }
      : g));
  }
  return out;
}

/** Fold document `base`'s labels from its file (`meta`, relative keys) into
 *  the local ones: the file's names / group labels override; its folders
 *  merge by id, else by name, else are added; a key the file files in a
 *  folder leaves every other folder. Keys that aren't the document's are
 *  ignored. Returns `labels` itself when nothing changed. */
export function mergeDocMeta(
  labels: AnalysisLabels, base: string, meta: DocMeta | undefined, uploadKeys: ReadonlySet<string>,
): AnalysisLabels {
  if (!meta) return labels;
  const abs = (k: string) => absKeyOf(k, base, uploadKeys);
  const mergeRec = (cur: Record<string, string>, inc: Record<string, string> | undefined) => {
    let out = cur;
    for (const [k, v] of Object.entries(inc || {})) {
      const a = abs(k);
      if (a == null || out[a] === v) continue;
      if (out === cur) out = { ...cur };
      out[a] = v;
    }
    return out;
  };
  const sourceNames = mergeRec(labels.sourceNames, meta.sourceNames);
  const measurementGroups = mergeRec(labels.measurementGroups, meta.measurementGroups);
  let groups = labels.sourceGroups;
  let groupsChanged = false;
  for (const fg of meta.sourceGroups || []) {
    const keys: string[] = [];
    for (const k of fg.sourceKeys) {
      const a = abs(k);
      if (a != null && !keys.includes(a)) keys.push(a);
    }
    if (!keys.length) continue;
    let at = groups.findIndex((g) => g.id === fg.id);
    if (at < 0) at = groups.findIndex((g) => g.name === fg.name);
    const moving = new Set(keys);
    const next = groups.map((g, i) => {
      if (i === at) {
        const add = keys.filter((k) => !g.sourceKeys.includes(k));
        return add.length ? { ...g, sourceKeys: [...g.sourceKeys, ...add] } : g;
      }
      return g.sourceKeys.some((k) => moving.has(k)) ? { ...g, sourceKeys: g.sourceKeys.filter((k) => !moving.has(k)) } : g;
    });
    if (at < 0) next.push({ id: fg.id, name: fg.name, sourceKeys: keys });
    if (at < 0 || next.some((g, i) => g !== groups[i])) {
      groups = next;
      groupsChanged = true;
    }
  }
  if (sourceNames === labels.sourceNames && measurementGroups === labels.measurementGroups && !groupsChanged) return labels;
  return { sourceNames, measurementGroups, sourceGroups: groups };
}

/** A document's labels as they stand once its file has been read: restored
 *  uploads (`restoredKeys`, those it owns) that the file files in no folder
 *  land in "Direct upload", as hydrate does (fileUnderFolder) — so reading
 *  a file back doesn't by itself count as a change to it. */
export function metaAsRead(meta: DocMeta | undefined, restoredKeys: readonly string[]): DocMeta | undefined {
  const groups = meta?.sourceGroups ?? [];
  const before: AnalysisLabels = { sourceNames: {}, measurementGroups: {}, sourceGroups: groups };
  const after = fileUnderFolder(before, restoredKeys, DIRECT_UPLOAD_FOLDER, () => "");
  if (after === before) return meta;
  return compact({ ...meta, sourceGroups: after.sourceGroups });
}

/** Put `keys` that are in no folder into the folder named `folderName`
 *  (created, first, when there is none) — restored upload tiles land in
 *  "Direct upload" like fresh uploads do. Returns `labels` itself when
 *  nothing changed. */
export function fileUnderFolder(
  labels: AnalysisLabels, keys: readonly string[], folderName: string, newFolderId: () => string,
): AnalysisLabels {
  const filed = new Set(labels.sourceGroups.flatMap((g) => g.sourceKeys));
  const loose = keys.filter((k, i, a) => k && !filed.has(k) && a.indexOf(k) === i);
  if (!loose.length) return labels;
  const at = labels.sourceGroups.findIndex((g) => g.name === folderName);
  const sourceGroups = at < 0
    ? [{ id: newFolderId(), name: folderName, sourceKeys: loose }, ...labels.sourceGroups]
    : labels.sourceGroups.map((g, i) => (i === at ? { ...g, sourceKeys: [...g.sourceKeys, ...loose] } : g));
  return { ...labels, sourceGroups };
}
