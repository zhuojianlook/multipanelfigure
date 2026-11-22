/* ──────────────────────────────────────────────────────────
   projectNav — shared navigation/save guards for the document
   tabs (open .mpf files) and the Collage Assembly.

   The DocumentTabs strip and any other caller route through these
   helpers so behaviour is identical everywhere: every navigation
   away from a dirty builder offers Save / Don't save / Cancel, and
   the open-document registry (collageStore.openDocs) stays in sync
   with the builder's currently-loaded project.
   ────────────────────────────────────────────────────────── */

import { useFigureStore } from "../store/figureStore";
import { useCollageStore, newDocId } from "../store/collageStore";
import { useAnalysisWorkspaceStore, docHasAnalysis, untitledAnalysisDocIds } from "../store/analysisWorkspaceStore";
import { api } from "../api/client";
import { confirm as confirmDialog, confirmThree, alert as alertDialog } from "../components/shared/ConfirmDialog";
import { saveProjectDialog } from "../components/shared/SaveProjectDialog";
import { serializeDocOp, isStateCacheMiss } from "./docOpQueue";
import { cloudProviderOf, fileUidOf, isMissingFileError, loadFailureNotice } from "./docIdentity";
import {
  findRenamedOpenTab, itemUidAt, knownFileUid, locateOpenDoc, markItemsMissingAt, ontoAnotherTab, relinkCollageItem,
  relocateOpenDoc, repointOpenDoc, resolveRefs,
} from "./docLinks";
import { basenameOf, dirnameOf } from "./workflowPersist";

/* ── Per-document in-memory state cache ──────────────────────────────────
   The Python backend holds exactly ONE builder document's state (config +
   images + fonts) at a time — loading a project clears the previous one. To
   make switching between document tabs SEAMLESS without losing a tab's
   unsaved edits AND without the multi-second cost of re-encoding every image,
   we ask the backend to STASH the outgoing tab's live state in memory
   (api.stashState — a reference swap, no encode) before swapping, and
   ACTIVATE it (figureStore.restoreDocState) when the user returns. Clean tabs
   (already on disk, or blank) need no stash — they reload from disk / reset.

   In-memory only: if the sidecar restarts the stash is lost (saved docs fall
   back to disk; unsaved-only docs are lost — mitigated by the in-tab Save
   button / save guards). `backendStashed` tracks which doc ids have a stash. */
const backendStashed = new Set<string>();

/* ── Clean-document perf cache ───────────────────────────────────────────
   Switching to a CLEAN saved tab used to reload+decode its whole .mpf from
   disk every time (slow). To make repeat-switching fast we ALSO cache clean
   docs' live state in the backend on switch-away and restore them via the fast
   in-memory activate path. This is a SEPARATE layer from `backendStashed` on
   purpose: backendStashed (and every save/close guard built on it) tracks only
   DIRTY docs and is left completely untouched, so this optimisation can never
   mis-count unsaved work or drop edits. `cleanCached` is purely a speed cache —
   never counted as unsaved, evicted LRU to bound memory. Caveat: a clean cached
   doc whose .mpf is modified by an EXTERNAL process won't refresh until its tab
   is closed/reopened (the app itself never writes a non-active doc's file). */
const cleanCached = new Set<string>();
/** Max clean docs cached in the backend at once (bounds memory — each holds a
 *  full decoded image set). Dirty stashes are NOT capped (never dropped). */
const CLEAN_CACHE_MAX = 5;

/** Drop a doc's backend stash (dirty or clean) and forget it in both sets. */
function dropStash(id: string): void {
  if (backendStashed.has(id) || cleanCached.has(id)) {
    void api.dropState(id).catch(() => {});
    backendStashed.delete(id);
    cleanCached.delete(id);
  }
}

/** Evict oldest clean caches (Set keeps insertion order) past the cap. */
function evictCleanCache(): void {
  while (cleanCached.size > CLEAN_CACHE_MAX) {
    const oldest = cleanCached.values().next().value as string | undefined;
    if (oldest === undefined) break;
    cleanCached.delete(oldest);
    void api.dropState(oldest).catch(() => {});
  }
}

/** Push the stashed doc-id set into the collage store so the tab strip can
 *  show the unsaved/save indicator on tabs whose edits aren't live. Only DIRTY
 *  stashes drive the indicator — clean perf caches are invisible to the user. */
function syncSnapshotDirty() {
  useCollageStore.getState().setSnapshotDirtyDocIds(Array.from(backendStashed));
}

/* Document-operation serialization lives in ./docOpQueue (shared with
   figureStore's uploads); re-exported here for existing importers. */
export { serializeDocOp, whenDocOpsIdle } from "./docOpQueue";

/** True if any open document has unsaved in-memory edits (a stash). Used by
 *  the app-close guard to warn about tabs whose edits would be lost on quit. */
export function hasUnsavedSnapshots(): boolean {
  return backendStashed.size > 0;
}

/** Untitled documents whose Analysis workflows have nowhere to auto-save —
 *  close / quit / "Save all" treat them as unsaved (even with an empty
 *  figure). Saved documents never count: their analysis auto-saves. */
export function unsavedAnalysisDocIds(): string[] {
  return untitledAnalysisDocIds();
}

/** Whether document `docId`'s figure edits would be thrown away right now
 *  (it is the live doc with unsaved edits, or its edits are parked). */
function figureDirty(docId: string): boolean {
  return (useCollageStore.getState().activeDocId === docId && useFigureStore.getState().unsaved)
    || backendStashed.has(docId);
}

/** Write document `docId`'s Analysis workflows into its .mpf (when they
 *  changed). Resolves null on success / nothing to do, else a reason the
 *  user should hear: the write failed, or the file's analysis could not be
 *  read / is a newer format while this document holds analysis of its own. */
async function persistWorkflowsOf(docId: string, figureDiscarded: boolean): Promise<string | null> {
  try {
    const out = await useAnalysisWorkspaceStore.getState().persistDocWorkflows(docId, { figureDiscarded });
    if ((out === "unsupported" || out === "not-hydrated") && docHasAnalysis(docId)) {
      return out === "unsupported"
        ? "Its file holds Analysis saved by a newer version of the app, which this version can't update."
        : "The Analysis already saved in its file could not be read, so it was not overwritten.";
    }
    return null;
  } catch (e) {
    console.error("[projectNav] saving analysis workflows failed for", docId, e);
    return e instanceof Error && e.message ? e.message : "The file could not be written.";
  }
}

/** Write every open document's changed Analysis workflows into its .mpf —
 *  the app-quit step, after the save guard resolved. `figuresDiscarded`
 *  ("Close without saving"): results of documents whose unsaved figure edits
 *  are being dropped are stored as "needs re-run". Serialized as one doc op.
 *  Returns the names of the documents that could not be written. */
export function persistAllWorkflowsForQuit(opts: { figuresDiscarded?: boolean } = {}): Promise<string[]> {
  return serializeDocOp(async () => {
    const failed: string[] = [];
    for (const d of useCollageStore.getState().openDocs) {
      if (!d.path) continue;
      const discarded = !!opts.figuresDiscarded && figureDirty(d.id);
      if (await persistWorkflowsOf(d.id, discarded)) failed.push(d.name);
    }
    return failed;
  });
}

/** The app's quit guard (AppShell registers it once the window exists): asks
 *  about unsaved figures, then writes every document's Analysis workflows.
 *  The in-app update runs it before stopping the sidecar. Resolves false
 *  when the user stays; true when there is no guard (browser preview). */
type QuitGuard = (action: "close" | "restart") => Promise<boolean>;
let _quitGuard: QuitGuard | null = null;
export function setQuitGuard(fn: QuitGuard | null): void { _quitGuard = fn; }
export async function runQuitGuard(action: "close" | "restart"): Promise<boolean> {
  return _quitGuard ? _quitGuard(action) : true;
}

/** Stash the currently-active builder doc's live state in the backend so
 *  switching away and back is fast. Dirty docs go in `backendStashed` (their
 *  edits must survive — drives the save guards); clean docs go in the
 *  `cleanCached` perf layer (a fast reload, never counted as unsaved).
 *
 *  Returns true when the live doc is safely parked (or there is nothing to
 *  park). False means the stash failed for a doc that can't simply be
 *  reloaded from disk — the caller MUST NOT replace the live document then,
 *  or its unsaved content is gone. */
async function stashCurrentDoc(): Promise<boolean> {
  const cs = useCollageStore.getState();
  const activeId = cs.activeDocId;
  if (!activeId) return true;
  const doc = cs.openDocs.find((d) => d.id === activeId);
  if (!doc) return true; // already removed (mid-close)
  // Push every pending / debounced config write of THIS doc first, so an edit
  // made just before the click is in the stash — and can't land in the next tab.
  await useFigureStore.getState().flushConfigWrites();
  const st = useFigureStore.getState();
  // An Untitled doc with content has nowhere to reload from: treat it as
  // unsaved even if no edit flagged it, so it is never evicted as "clean".
  const keep = st.unsaved || (!doc.path && Object.keys(st.loadedImages).length > 0);
  try {
    await api.stashState(activeId);
  } catch (e) {
    console.error("[projectNav] stash failed for", activeId, e);
    syncSnapshotDirty();
    // A clean saved doc can be reloaded from disk; anything else must stay live.
    return !keep && !!doc.path;
  }
  if (keep) {
    backendStashed.add(activeId);
    cleanCached.delete(activeId);
  } else {
    // Clean: cache for fast return. Refresh its LRU position, then evict
    // the oldest clean caches past the cap.
    backendStashed.delete(activeId);
    cleanCached.delete(activeId);
    cleanCached.add(activeId);
    evictCleanCache();
  }
  syncSnapshotDirty();
  return true;
}

/** Tell the user a document switch was refused because the outgoing doc
 *  could not be parked safely (nothing was changed). */
async function alertCouldNotPark(): Promise<void> {
  const cs = useCollageStore.getState();
  const name = cs.openDocs.find((d) => d.id === cs.activeDocId)?.name || "This document";
  await alertDialog({
    title: "Could not switch documents",
    body: `"${name}" has changes that could not be set aside safely, so it was kept open.\n\n`
      + "Save it, then try again.",
  });
}

/* The builder content is inert (collageStore.docSwapBusy → AppShell shield)
   from the moment a document-replacing op is QUEUED until it finishes. Raising
   it only when the op starts running left a gap: a tab click queued behind a
   long upload let the user open e.g. Edit Panel on the outgoing doc, whose
   Save then wrote that panel into the incoming one. Reference-counted, so a
   queued wrapper plus its inner withDocSwap (or two queued swaps) never drop
   the shield early. */
let _pendingSwaps = 0;
function beginDocSwap(): void {
  _pendingSwaps++;
  (document.activeElement as HTMLElement | null)?.blur?.();   // fire onBlur → flushHeaderEdits first
  useCollageStore.getState().setDocSwapBusy(true);
}
function endDocSwap(): void {
  _pendingSwaps = Math.max(0, _pendingSwaps - 1);
  if (_pendingSwaps === 0) useCollageStore.getState().setDocSwapBusy(false);
}
/** Queue a document-replacing op with the shield raised immediately. */
function queueDocSwap<T>(fn: () => Promise<T>): Promise<T> {
  beginDocSwap();
  return serializeDocOp(fn).finally(endDocSwap);
}
/** Make the builder content inert while the live document is being swapped,
 *  so no click or keystroke edits the outgoing doc after it was stashed (it
 *  would be lost) or lands in the incoming one. */
async function withDocSwap<T>(fn: () => Promise<T>): Promise<T> {
  beginDocSwap();
  try {
    return await fn();
  } finally {
    endDocSwap();
  }
}

/** Load a document's state into the backend builder: prefer its in-memory
 *  cache — a dirty stash (preserves unsaved edits) or a clean perf cache (fast
 *  reload) — else the on-disk .mpf, else blank. */
async function activateDoc(target: { id: string; path: string | null; name?: string }): Promise<void> {
  // The tab as it is now: its file may have been re-pointed (renamed or moved
  // outside the app) since the caller looked.
  const doc = useCollageStore.getState().openDocs.find((d) => d.id === target.id) ?? target;
  const wasDirty = backendStashed.has(doc.id);
  if (wasDirty || cleanCached.has(doc.id)) {
    try {
      // Restore with the cached dirty state: a dirty stash stays unsaved; a
      // clean cache stays clean (so no false "unsaved" indicator on return).
      await useFigureStore.getState().restoreDocState(doc.id, { path: doc.path, dirty: wasDirty });
      backendStashed.delete(doc.id); // now live in the backend
      cleanCached.delete(doc.id);
      syncSnapshotDirty();
      return;
    } catch (e) {
      // Only a genuine cache miss (e.g. the sidecar restarted) may fall back
      // to disk. Any other failure is transient: keep the stash and its
      // unsaved flag, and let the caller report the failed switch.
      if (!isStateCacheMiss(e)) throw e;
      console.error("[projectNav] activate-state cache miss; falling back to disk/blank", e);
      backendStashed.delete(doc.id);
      cleanCached.delete(doc.id);
      syncSnapshotDirty();
      if (wasDirty) {
        await alertDialog({
          title: "Unsaved changes lost",
          body: `The unsaved changes in "${doc.name || "this document"}" are no longer available `
            + "(the backend restarted). "
            + (doc.path ? "The last saved version was loaded." : "The document was reset."),
        });
      }
    }
  }
  if (doc.path) {
    try {
      await loadProjectLabelled(doc.path, doc.id, doc.name);
    } catch (e) {
      // No file at its path any more: renamed or moved outside the app.
      // Found by its uid → the tab follows it there and loads it from there.
      if (!isMissingFileError(e)) throw e;
      const moved = await relocateOpenDoc(doc.id);
      if (!moved) throw e;
      await loadProjectLabelled(moved, doc.id, doc.name);
    }
    // Its saved Analysis, if that read was put off while the file was stored
    // online only: the file was just downloaded for the figure (queued, not
    // awaited — the swap does not wait for it; a no-op once read).
    const now = useCollageStore.getState().openDocs.find((d) => d.id === doc.id);
    if (now?.path) void useAnalysisWorkspaceStore.getState().ensureDocLoaded(now.id, now.path, { download: true });
  } else {
    await useFigureStore.getState().newBlankFigure(doc.id);
  }
}

/** figureStore.loadProject, saying on the swap shield what is being opened
 *  should it take a while — a file stored online only (Google Drive,
 *  Dropbox…) is downloaded first, which can take many seconds. */
async function loadProjectLabelled(path: string, docId: string, name?: string): Promise<void> {
  const where = cloudProviderOf(path);
  const what = `"${name || basenameOf(path).replace(/\.mpf$/i, "") || "the document"}"`;
  const cs = useCollageStore.getState();
  cs.setDocSwapLabel(where ? `Opening ${what} from ${where}…` : `Opening ${what}…`);
  try {
    await useFigureStore.getState().loadProject(path, docId);
  } finally {
    useCollageStore.getState().setDocSwapLabel(null);
  }
}

/** The document `docId` is (still) on the open-document list. */
function openDoc(docId: string) {
  return useCollageStore.getState().openDocs.find((d) => d.id === docId);
}

/** Native "open .mpf" dialog (as Analysis' Locate uses). Null: cancelled,
 *  or no dialog (a browser preview). */
async function pickMpfFile(title: string, startIn?: string): Promise<string | null> {
  let picked: unknown;
  try {
    const dialog = await import("@tauri-apps/plugin-dialog");
    picked = await dialog.open({
      title,
      multiple: false,
      directory: false,
      filters: [{ name: "Project", extensions: ["mpf"] }],
      ...(startIn ? { defaultPath: startIn } : {}),
    });
  } catch (e) {
    console.warn("[projectNav] the file dialog is unavailable", e);
    return null;
  }
  if (typeof picked === "string") return picked;
  if (picked && typeof picked === "object" && typeof (picked as { path?: unknown }).path === "string") {
    return (picked as { path: string }).path;
  }
  return null;
}

/* ── A tab whose file was renamed, moved or deleted outside the app ──────
   Before a save writes a tab's file, the file is looked up by the uid it
   carries (docLinks.locateOpenDoc): found elsewhere, the tab follows it and
   the save goes there — Save must not quietly re-create "a.mpf" next to the
   "b.mpf" the user renamed it to. Found nowhere, the user decides. */

/** Open tab `docId`'s file is gone (`at`: where it was, missing / ambiguous):
 *  ask where to save it — the same path again, somewhere else (the save
 *  dialog), or not at all (null). Either way the old file is gone, so the
 *  sidecar keeps the document's uid: it is the same document. When another
 *  document's file now has that name (`at.exists`), only somewhere else (the
 *  save dialog, which itself asks before overwriting) or not at all. */
async function askWhereToSaveMissing(
  docId: string, at: { status: string; path: string; matches?: string[]; exists?: boolean },
): Promise<string | null> {
  const name = openDoc(docId)?.name || basenameOf(at.path);
  const found = at.status === "ambiguous" ? at.matches ?? [] : [];
  const several = found.length > 1
    ? `\n\nSeveral copies of it were found, so none was picked:\n${found.slice(0, 3).join("\n")}`
    : found.length === 1
      ? `\n\nA file carrying it was found, but it is not the file this tab had (it may be a copy), so it was not picked:\n${found[0]}`
      : "";
  if (at.exists) {
    const ok = await confirmDialog({
      title: `"${basenameOf(at.path) || at.path}" now holds a different document`,
      body: `This document's file was moved, renamed or replaced outside the app, and another document now has its name:\n${at.path}${several}\n\nSave it somewhere else?`,
      confirmLabel: "Save somewhere else…",
      cancelLabel: "Cancel",
    });
    return ok ? saveProjectDialog({ title: `Save "${name}" as…`, defaultPath: at.path }) : null;
  }
  const choice = await confirmThree({
    title: `"${basenameOf(at.path) || at.path}" is no longer where it was`,
    body: `It was moved, renamed or deleted outside the app.\n${at.path}${several}`,
    confirmLabel: "Save here again",
    tertiaryLabel: "Save somewhere else…",
    cancelLabel: "Cancel",
  });
  if (choice === "cancel") return null;
  if (choice === "confirm") return at.path;
  return saveProjectDialog({ title: `Save "${name}" as…`, defaultPath: at.path });
}

/** Where to save open tab `docId`, which has a file: its path; the file's
 *  new path when it was renamed or moved outside the app (found by its uid —
 *  the tab is re-pointed there, no prompt); when it can't be found, the
 *  user's pick (askWhereToSaveMissing). Null: cancelled. A lookup that
 *  could not be made keeps the tab's path. */
export async function ensureDocFile(docId: string): Promise<string | null> {
  const at = await locateOpenDoc(docId);
  if (!at) return openDoc(docId)?.path ?? null;
  if (at.status === "ok" || at.status === "moved") return at.path;
  return askWhereToSaveMissing(docId, at);
}

/** Mirror the builder's current project path onto the active doc tab,
 *  so an Untitled tab that just got saved (or a tab just loaded from
 *  disk) shows the right name/path. */
function syncActiveDocPath() {
  const cs = useCollageStore.getState();
  const path = useFigureStore.getState().currentProjectPath;
  if (cs.activeDocId && path) cs.docSetPath(cs.activeDocId, path);
}

/** Ensure the current builder project is saved to a known path.
 *  Shows the shared "Save Project" modal (the SAME one the Sidebar's
 *  Save Project button uses) so the user picks where + what name.
 *  Returns the path on success, null if cancelled. */
export async function ensureProjectSaved(): Promise<string | null> {
  // The live document's file may have been renamed or moved outside the app:
  // found by its uid, the tab (and currentProjectPath) follow it; not found,
  // it is saved again even when clean (the dialog below).
  let gone = false;
  const live = openDoc(useCollageStore.getState().activeDocId ?? "");
  if (live?.path && live.path === useFigureStore.getState().currentProjectPath) {
    const at = await locateOpenDoc(live.id);
    gone = !!at && at.status !== "ok" && at.status !== "moved";
  }
  const { currentProjectPath, unsaved, saveProject } = useFigureStore.getState();
  if (currentProjectPath && !unsaved && !gone) return currentProjectPath;

  const picked = await saveProjectDialog({
    defaultPath: currentProjectPath || defaultProjectName(),
  });
  if (!picked) return null;
  const ok = await saveProject(picked);
  if (!ok) {
    await alertDialog({ title: "Save failed", body: `Could not save the figure to ${picked}.` });
    return null;
  }
  syncActiveDocPath();
  return useFigureStore.getState().currentProjectPath || picked;
}

/** The builder Sidebar's "Save Project" button: the Save Project modal for
 *  the live document. Its file is looked up by its uid first, as Save does:
 *  renamed or moved outside the app → the tab follows it and the modal
 *  offers the new path, so saving under another name is a copy of that file
 *  (a new uid), not a move of one that is gone; not found → the user picks
 *  (askWhereToSaveMissing) instead of re-creating the old name unasked.
 *  Serialized, like the tab's Save. */
export function saveProjectWithDialog(): Promise<boolean> {
  return serializeDocOp(async () => {
    const live = openDoc(useCollageStore.getState().activeDocId ?? "");
    const at = live?.path && live.path === useFigureStore.getState().currentProjectPath
      ? await locateOpenDoc(live.id) : null;
    const picked = live && at && at.status !== "ok" && at.status !== "moved"
      ? await askWhereToSaveMissing(live.id, at)
      : await saveProjectDialog({
        // Seeded from the (renamed) active tab when unsaved, else the current
        // path — as it is now — else a timestamp, matching ensureProjectSaved.
        defaultPath: useFigureStore.getState().currentProjectPath || defaultProjectName(),
      });
    if (!picked) return false;
    return useFigureStore.getState().saveProject(picked);
  });
}

/** Save a document tab from its in-tab Save button. Contextual, like ⌘S:
 *  - a tab that already has a file → overwrites that .mpf in place (no dialog).
 *  - an Untitled tab (never saved) → prompts for a path (defaulting to the
 *    tab's name), then attaches that path to the tab.
 *  Switches to the tab first when it isn't the active one, since its unsaved
 *  edits live in an in-memory snapshot until then. Returns true on success. */
async function saveDocumentImpl(docId: string): Promise<boolean> {
  const cs = useCollageStore.getState();
  const doc = cs.openDocs.find((d) => d.id === docId);
  if (!doc) return false;
  // Bring this tab's live state into the backend so saveProject captures it.
  // Abort if the switch didn't land — saving now would write whatever doc is
  // still live to THIS tab's file (wrong-content / duplicate save).
  if (cs.activeDocId !== docId) {
    const switched = await switchToDocumentImpl(docId);
    if (!switched) return false;
  }
  const fs = useFigureStore.getState();
  let path: string | null;
  if (openDoc(docId)?.path) {
    // Its file, wherever it is now (renamed / moved outside the app → the
    // tab follows it); gone → the user picks where (or cancels).
    path = await ensureDocFile(docId);
    if (!path) return false;
  } else {
    path = fs.currentProjectPath;
    if (!path) {
      path = await saveProjectDialog({ defaultPath: defaultProjectName() });
      if (!path) return false; // user cancelled
    }
  }
  const ok = await fs.saveProject(path);
  if (!ok) {
    await alertDialog({ title: "Save failed", body: `Could not save "${doc.name}" to ${path}.` });
    return false;
  }
  path = useFigureStore.getState().currentProjectPath || path;   // as written (".mpf" added…)
  // Newly-saved Untitled → adopt the chosen path + name on the tab. For an
  // overwrite (path unchanged) leave the tab name alone so a custom rename
  // sticks. saveProject already cleared `unsaved`, so the dirty dot/active
  // marker disappears; also drop any parked snapshot.
  if (!doc.path) useCollageStore.getState().docSetPath(docId, path);
  dropStash(docId);
  syncSnapshotDirty();
  return true;
}

/** Save a document tab from its in-tab Save button. Contextual, like ⌘S:
 *  - a tab that already has a file → overwrites that .mpf in place (no dialog).
 *  - an Untitled tab (never saved) → prompts for a path (defaulting to the
 *    tab's name), then attaches that path to the tab.
 *  Serialized (see serializeDocOp) so it can't race a concurrent tab switch. */
export function saveDocument(docId: string): Promise<boolean> {
  return serializeDocOp(() => saveDocumentImpl(docId));
}

/** Default project filename for the Save Project modal. If the user has
 *  renamed the active (unsaved) tab, that name seeds the filename; otherwise
 *  fall back to a timestamped name. Sanitizes path-illegal characters and
 *  ensures a .mpf extension. The Sidebar's Save Project button uses the
 *  identical default (saveProjectWithDialog). */
export function defaultProjectName(): string {
  const cs = useCollageStore.getState();
  const doc = cs.openDocs.find((d) => d.id === cs.activeDocId);
  const custom = doc?.name?.trim();
  if (custom && custom.toLowerCase() !== "untitled") {
    const safe = custom.replace(/[\\/:*?"<>|]+/g, "_").replace(/\.mpf$/i, "");
    if (safe) return `${safe}.mpf`;
  }
  const now = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  const ts = `${now.getFullYear()}${p(now.getMonth() + 1)}${p(now.getDate())}`
    + `_${p(now.getHours())}${p(now.getMinutes())}${p(now.getSeconds())}`;
  return `${ts}_project.mpf`;
}

/** Save one document tab for the app-close "Save all & close" flow.
 *
 *  - A MODIFIED file that already lives on disk → first confirms the overwrite,
 *    offering "Overwrite" / "Save to a different location…" / "Cancel" so the
 *    user never clobbers a file without intent (and can redirect it).
 *  - A never-saved Untitled doc → just prompts for a location.
 *
 *  Switches to the tab first when it isn't active. Returns false if the user
 *  cancels (so the close can be aborted). */
async function saveDocumentAs(docId: string): Promise<boolean> {
  const cs = useCollageStore.getState();
  const doc = cs.openDocs.find((d) => d.id === docId);
  if (!doc) return false;
  // Abort if the switch didn't land — otherwise we'd save whatever doc is still
  // live to THIS tab's file (the duplicate-of-last-saved bug).
  if (cs.activeDocId !== docId) {
    const switched = await switchToDocumentImpl(docId);
    if (!switched) return false;
  }
  const fs = useFigureStore.getState();
  // Its file, wherever it is now (renamed / moved outside the app → the tab
  // follows it, and the overwrite below names the new file).
  const at = await locateOpenDoc(docId);
  const existingPath = at?.path || openDoc(docId)?.path || fs.currentProjectPath;

  let path: string | null;
  if (at && at.status !== "ok" && at.status !== "moved") {
    // Gone: there is nothing to overwrite — ask where to save it instead.
    path = await askWhereToSaveMissing(docId, at);
    if (!path) return false; // cancelled → abort
  } else if (existingPath) {
    // Modified existing .mpf — confirm overwrite, or let the user redirect it.
    const fileName = existingPath.split(/[\\/]/).pop() || existingPath;
    const choice = await confirmThree({
      title: `Save "${doc.name}"?`,
      body: `Your changes will overwrite:\n${existingPath}\n\n`
        + `Overwrite "${fileName}", or save to a different location?`,
      confirmLabel: "Overwrite",
      tertiaryLabel: "Save to different location…",
      cancelLabel: "Cancel",
    });
    if (choice === "cancel") return false;
    if (choice === "confirm") {
      path = existingPath;
    } else {
      path = await saveProjectDialog({
        title: `Save "${doc.name}" as…`,
        defaultPath: existingPath,
      });
      if (!path) return false; // user cancelled the location picker → abort
    }
  } else {
    // Never-saved Untitled — just pick a location.
    path = await saveProjectDialog({
      title: `Save "${doc.name}"`,
      defaultPath: defaultProjectName(),
    });
    if (!path) return false; // user cancelled → abort
  }

  const ok = await fs.saveProject(path);
  if (!ok) {
    await alertDialog({ title: "Save failed", body: `Could not save "${doc.name}" to ${path}.` });
    return false;
  }
  path = useFigureStore.getState().currentProjectPath || path;   // as written (".mpf" added…)
  // Reflect a new location on the tab (leave the name alone when overwriting
  // the same path, so a custom rename sticks).
  if (path !== openDoc(docId)?.path) useCollageStore.getState().docSetPath(docId, path);
  dropStash(docId);
  syncSnapshotDirty();
  return true;
}

/** Save EVERY open document that has unsaved changes — the live active doc
 *  (when `unsaved`) and any tabs whose edits are parked in an in-memory
 *  snapshot. Returns false as soon as the user cancels any save (so an
 *  app-close can be aborted); true once all dirty docs are saved. Used by the
 *  app-close guard so quitting never silently drops work.
 *
 *  When `promptForLocation` is set, each dirty doc shows the Save Project
 *  dialog pre-filled with its current path (the app-close behaviour: the user
 *  picks a location for every figure). Otherwise saved docs overwrite in place
 *  and only never-saved Untitled docs prompt (the in-tab Save behaviour). */
export function saveAllOpenDocs(
  opts: { promptForLocation?: boolean } = {},
): Promise<boolean> {
  // Serialized as ONE unit so a stray tab click mid-save can't interleave with
  // the loop; the loop calls the un-serialized …Impl helpers directly. It
  // switches documents to save each one, so the shield is up throughout.
  return queueDocSwap(async () => {
    const cs = useCollageStore.getState();
    const fs = useFigureStore.getState();
    // Untitled documents holding Analysis work count too: a file is the only
    // place their workflows can be saved.
    const analysisOnly = new Set(unsavedAnalysisDocIds());
    const dirtyIds = cs.openDocs
      .filter((d) => (cs.activeDocId === d.id && fs.unsaved) || backendStashed.has(d.id) || analysisOnly.has(d.id))
      .map((d) => d.id);
    for (const id of dirtyIds) {
      const ok = opts.promptForLocation
        ? await saveDocumentAs(id)
        : await saveDocumentImpl(id);
      if (!ok) return false; // user cancelled / a save failed → abort
    }
    return true;
  });
}

/** If the builder has unsaved changes, prompt Save / Don't save / Cancel.
 *  "clean" = nothing to save, "saved", "discarded" (Don't save), "cancel"
 *  (cancelled, or the save did not happen). */
async function promptSaveLiveDoc(): Promise<"clean" | "saved" | "discarded" | "cancel"> {
  const { unsaved, currentProjectPath } = useFigureStore.getState();
  if (!unsaved) return "clean";
  const name = currentProjectPath ? currentProjectPath.split(/[\\/]/).pop() : "this figure";
  const choice = await confirmThree({
    title: "Unsaved changes",
    body: `You have unsaved changes to ${name}.\n\nSave before continuing?`,
    confirmLabel: "Save",
    tertiaryLabel: "Don't save",
    cancelLabel: "Cancel",
  });
  if (choice === "cancel") return "cancel";
  if (choice === "confirm") return (await ensureProjectSaved()) ? "saved" : "cancel";
  return "discarded";
}

/** If the builder has unsaved changes, prompt Save / Don't save /
 *  Cancel. Returns true to proceed, false to abort. */
export async function maybeSaveBeforeLeavingBuilder(): Promise<boolean> {
  return (await promptSaveLiveDoc()) !== "cancel";
}

/** Switch into the Collage Assembly. Tab switches are seamless — no
 *  save prompt (the figure stays open as a tab); the only save guard
 *  is when a tab is actually closed (closeDoc) or the app quits. */
export async function enterCollage(): Promise<void> {
  if (useCollageStore.getState().mode === "collage") return;
  useCollageStore.getState().setMode("collage");
}

/** Switch into the Analysis workspace (a permanent tab). Seamless — the
 *  active builder doc stays loaded; analysis pulls measurements/sources
 *  from it (and other open MPFs). */
export async function enterAnalysis(): Promise<void> {
  if (useCollageStore.getState().mode === "analysis") return;
  useCollageStore.getState().setMode("analysis");
}

/** Switch the builder to a specific open document tab (by id). Seamless —
 *  no save prompt. The outgoing dirty tab is snapshotted in memory so its
 *  unsaved edits survive; the target is restored from its snapshot (if
 *  any), else loaded from disk, else opened blank.
 *
 *  Returns true only when the target became the live active doc. On a load
 *  failure it leaves the PREVIOUS doc active+live (a consistent state — the
 *  active tab always matches the backend) and returns false, so callers like
 *  saveDocument never save the wrong document's content under this tab. */
async function switchToDocumentImpl(docId: string): Promise<boolean> {
  const cs = useCollageStore.getState();
  const doc = cs.openDocs.find((d) => d.id === docId);
  if (!doc) return false;
  // The backend already holds this doc — just ensure builder mode. Covers
  // returning from the Collage tab to the active figure with NO reload, so
  // its in-progress edits stay intact.
  if (cs.activeDocId === docId) {
    if (cs.mode !== "builder") useCollageStore.getState().setMode("builder");
    return true;
  }
  // Switching to a different doc: snapshot the outgoing one (if dirty),
  // then activate the target — with the builder inert throughout.
  return withDocSwap(async () => {
    if (!(await stashCurrentDoc())) {
      await alertCouldNotPark();
      return false;
    }
    try {
      await activateDoc(doc);
    } catch (e) {
      console.error("[projectNav] switch document failed:", e);
      // Leave the previous doc active/live — do NOT flip activeDocId to a tab
      // whose content failed to load (that would desync active vs backend).
      // A file that is gone (and was not found by its uid): the user can
      // point at it, or close the tab.
      return recoverMissingDoc(docId, e);
    }
    useCollageStore.getState().docSetActive(docId);
    useCollageStore.getState().setMode("builder");
    return true;
  });
}

/** Switching to tab `docId` failed with `err`. Its file gone (renamed,
 *  moved or deleted outside the app, and not found by its uid): Locate… /
 *  Close tab / Cancel. Any other failure: the plain "Load failed" notice.
 *  Runs inside the switch's doc op (only plain awaits, never a queued op).
 *  True only when the tab became the live document after all. */
async function recoverMissingDoc(docId: string, err: unknown): Promise<boolean> {
  const doc = openDoc(docId);
  if (!doc) return false;
  if (!doc.path || !isMissingFileError(err)) {
    // Still at its path but unreadable — e.g. stored online only and its
    // cloud app could not download it: say why (no Locate: it is not lost).
    await alertDialog(loadFailureNotice(doc.name, doc.path, err));
    return false;
  }
  // Its collage figures have no file either: they no longer keep the tab
  // open (closeDocImpl), and re-link if the file is opened later.
  markItemsMissingAt(doc.path);
  const file = basenameOf(doc.path) || doc.path;
  const choice = await confirmThree({
    title: `"${file}" could not be found`,
    body: `"${doc.name}" could not be loaded: its file is no longer at\n${doc.path}\n\n`
      + "It was moved, renamed or deleted outside the app. Locate it, or close this tab?",
    confirmLabel: "Locate…",
    tertiaryLabel: "Close tab",
    cancelLabel: "Cancel",
  });
  if (choice === "cancel") return false;
  if (choice === "tertiary") {
    await closeDocImpl(docId);
    return false;
  }
  const picked = await pickMpfFile(`Locate "${file}"`, dirnameOf(doc.path) || undefined);
  if (!picked) return false;
  return adoptLocatedFile(docId, picked);
}

/** The user located tab `docId`'s file at `picked`. The same document (its
 *  uid matches, or one of the two is unknown): the tab — and everything on
 *  its old path — follows it there and is loaded. Another document (or a
 *  file another tab already has): it opens in a tab of its own, and this
 *  tab stays as it was. */
async function adoptLocatedFile(docId: string, picked: string): Promise<boolean> {
  const doc = openDoc(docId);
  if (!doc?.path) return false;
  if (picked !== doc.path) {                                 // (else: put back where it was)
    const expected = knownFileUid(docId, doc.path);
    const found = (await resolveRefs([{ key: "pick", path: picked }], { scan: false })).pick;
    const there = openDoc(docId);
    if (!there?.path) return false;
    const other = useCollageStore.getState().openDocs.some((d) => d.id !== docId && d.path === picked);
    const differs = !!expected && !!found?.uid && found.uid !== expected;
    if (other || differs) {
      if (differs) {
        await alertDialog({
          title: "A different document",
          body: `"${basenameOf(picked)}" is not the document of the tab "${there.name}", `
            + "so it opens in a tab of its own.",
        });
      }
      await openProjectIntoTabImpl(picked);
      return false;
    }
    // Its Analysis counts as read only for the very file that was read (the
    // two uids match). A file whose uid can't be told (it carries none, or
    // it is a cloud placeholder), or a tab that knew none, may hold Analysis
    // this tab never read: it is read where it is now before anything is
    // written to it (and until then a figure save leaves the file's uid alone).
    // So may the file a held tab (a Finder copy's, a uid of its own) locates:
    // the uid it expects is the original's, which the original's file carries too.
    const f = fileUidOf(docId);
    const held = !!f && useAnalysisWorkspaceStore.getState().docUids[docId] !== f;
    const same = !held && !!expected && found?.uid === expected;
    repointOpenDoc(there.path, picked, found?.uid || expected, {
      reread: !same, ...(found?.file_id ? { fileId: found.file_id } : {}),
    });
  }
  try {
    await activateDoc({ id: docId, path: picked, name: doc.name });
  } catch (e) {
    console.error("[projectNav] loading the located file failed:", e);
    await alertDialog(loadFailureNotice(basenameOf(picked), picked, e));
    return false;
  }
  markItemsMissingAt(picked, false);
  useCollageStore.getState().docSetActive(docId);
  useCollageStore.getState().setMode("builder");
  return true;
}

/** Collage figure `itemId`'s .mpf could not be found (missingItemIds: moved
 *  to a folder no lookup reaches, or several copies of it were found): the
 *  user points at it with the native open dialog. The file picked is peeked
 *  at for its uid. The same document (the uids match, or one of the two is
 *  unknown): everything on the figure's old path follows it there
 *  (repointOpenDoc — other figures placed from it, a tab still on it).
 *  Another document's file — used only after "Use it anyway?" — or a file
 *  another tab already has: only this figure is re-linked. Then a tab on the
 *  file, as at launch. True: re-linked; false: cancelled or not possible. */
async function locateCollageItemImpl(itemId: string): Promise<boolean> {
  const it = useCollageStore.getState().items.find((i) => i.id === itemId);
  if (!it || it.kind !== "figure" || !it.projectPath) return false;
  const old = it.projectPath;
  const picked = await pickMpfFile(`Locate "${basenameOf(old) || old}"`, dirnameOf(old) || undefined);
  if (!picked) return false;
  const found = (await resolveRefs([{ key: "pick", path: picked }], { scan: false })).pick;
  if (!found || found.status !== "ok") {
    await alertDialog({ title: "Could not open that file", body: `"${basenameOf(picked)}" could not be found.` });
    return false;
  }
  const now = useCollageStore.getState().items.find((i) => i.id === itemId);
  if (!now?.projectPath) return false;                     // removed meanwhile
  const expected = itemUidAt(now.projectPath, itemId);
  const differs = !!expected && !!found.uid && found.uid !== expected;
  if (differs && !(await confirmDialog({
    title: "A different document",
    body: `"${basenameOf(picked)}" is not the document "${now.name}" was placed from.\n\nUse it anyway?`,
    confirmLabel: "Use it",
  }))) return false;
  const cur = useCollageStore.getState().items.find((i) => i.id === itemId);
  if (!cur?.projectPath) return false;
  if (picked !== cur.projectPath) {
    if (!differs && !ontoAnotherTab(cur.projectPath, picked)) {
      // A tab still on the old path counts the picked file as read only when
      // it is known to be the very file (the uids match).
      const same = !!expected && found.uid === expected;
      repointOpenDoc(cur.projectPath, picked, found.uid || expected, {
        reread: !same, ...(found.file_id ? { fileId: found.file_id } : {}),
      });
    }
    relinkCollageItem(itemId, picked, found.uid || expected);
  }
  useCollageStore.getState().setItemMissing(itemId, false);
  useCollageStore.getState().docEnsure(picked);
  return true;
}

/** Locate… for a collage figure whose .mpf is missing (Sidebar). Serialized:
 *  it may re-point an open tab. */
export function locateCollageItem(itemId: string): Promise<boolean> {
  return serializeDocOp(() => locateCollageItemImpl(itemId));
}

/** Public, serialized entry point for tab switching (see serializeDocOp). */
export function switchToDocument(docId: string): Promise<boolean> {
  // Re-selecting the live doc only changes mode — no shield needed.
  if (useCollageStore.getState().activeDocId === docId) {
    return serializeDocOp(() => switchToDocumentImpl(docId));
  }
  return queueDocSwap(() => switchToDocumentImpl(docId));
}

/** Create a new blank document tab and switch to it. Seamless — the prior
 *  tab is snapshotted (if dirty) so its edits survive; closing it is what
 *  guards. */
export function newBlankDoc(): Promise<void> {
  return queueDocSwap(() => withDocSwap(async () => {
    if (!(await stashCurrentDoc())) { await alertCouldNotPark(); return; }
    const id = newDocId();
    try {
      await useFigureStore.getState().newBlankFigure(id);
    } catch (e) {
      console.error("[projectNav] new blank document failed:", e);
      await alertDialog({ title: "Could not create a document", body: "The previous document is still open." });
      return;
    }
    // No name → the store assigns the next free "Untitled_N".
    useCollageStore.getState().docAdd({ id, path: null });
    useCollageStore.getState().docSetActive(id);
    useCollageStore.getState().setMode("builder");
  }));
}

/** Open a project file into a (new or existing) document tab. Seamless —
 *  the prior tab is snapshotted (if dirty) so its edits survive. */
export function openProjectIntoTab(path: string): Promise<void> {
  return queueDocSwap(() => openProjectIntoTabImpl(path));
}

/** openProjectIntoTab's body, for callers already inside a doc op. */
async function openProjectIntoTabImpl(path: string): Promise<void> {
  // Already open → just focus its tab. (Re-loading it from disk used to
  // silently discard that tab's unsaved edits.)
  const existing = useCollageStore.getState().openDocs.find((d) => d.path === path);
  if (existing) { await switchToDocumentImpl(existing.id); return; }
  // The file of an open tab, renamed or moved outside the app while it was
  // open (its uid, and the tab's old file gone): that tab follows it and is
  // focused — not reloaded, so its unsaved edits stay. A copy whose
  // original is still there opens in a tab of its own, as below.
  const twin = await findRenamedOpenTab(path);
  const twinNow = twin ? openDoc(twin.id) : undefined;
  if (twinNow?.path && !useCollageStore.getState().openDocs.some((d) => d.path === path)) {
    repointOpenDoc(twinNow.path, path, knownFileUid(twinNow.id, twinNow.path));
    await switchToDocumentImpl(twinNow.id);
    return;
  }
  await withDocSwap(async () => {
    if (!(await stashCurrentDoc())) { await alertCouldNotPark(); return; }
    const newId = newDocId();
    try {
      await loadProjectLabelled(path, newId);
    } catch (e) {
      console.error("[projectNav] open project failed:", e);
      await alertDialog(isMissingFileError(e)
        ? { title: "Load failed", body: "Could not load that project file — it may have been moved or deleted." }
        : loadFailureNotice(basenameOf(path).replace(/\.mpf$/i, "") || path, path, e));
      return;
    }
    const id = useCollageStore.getState().docEnsure(path, undefined, newId);
    // Freshly loaded from disk — drop any stale in-memory stash for it.
    dropStash(id);
    syncSnapshotDirty();
    useCollageStore.getState().docSetActive(id);
    useCollageStore.getState().setMode("builder");
  });
}

/** Close a document tab. Prompts to save if it's the active dirty doc.
 *  Returns true if it was closed, false if the user cancelled. Serialized via
 *  the public closeDoc wrapper below so it can't race a tab switch/save. */
async function closeDocImpl(docId: string): Promise<boolean> {
  const cs = useCollageStore.getState();
  const fs = useFigureStore.getState();
  const doc = cs.openDocs.find((d) => d.id === docId);
  if (!doc) return false;

  // Block closing a tab whose .mpf is placed in the collage — the
  // collage item depends on that figure. The user must remove it from
  // the collage first. Not for figures whose file is gone (missingItemIds):
  // the tab can't show them either, and a dead tab must be closable.
  if (doc.path) {
    const usedInCollage = cs.items.some(
      (it) => it.kind === "figure" && it.projectPath === doc.path && !cs.missingItemIds.includes(it.id),
    );
    if (usedInCollage) {
      await alertDialog({
        title: "Figure is in the collage",
        body: "This figure is placed in the Collage Assembly, so its tab can't "
          + "be closed yet.\n\nRemove the figure from the collage first, then "
          + "close the tab.",
      });
      return false;
    }
  }

  // The doc whose state the backend currently holds (regardless of whether
  // the builder or the collage is on screen).
  let isActiveDoc = cs.activeDocId === docId;
  // Its unsaved figure edits are being thrown away: Analysis results that
  // read its images are saved as "needs re-run".
  let figureDiscarded = false;

  if (isActiveDoc && fs.unsaved) {
    // Closing the live doc with unsaved changes — guard it (Save / Don't
    // save / Cancel; Save opens the file dialog so the user picks where +
    // what name).
    const choice = await promptSaveLiveDoc();
    if (choice === "cancel") return false;
    figureDiscarded = choice === "discarded";
  } else if (!isActiveDoc && backendStashed.has(docId)) {
    // A non-active tab with unsaved edits held only as an in-memory
    // stash. We can't save it without making it active first, so offer
    // a plain discard / cancel.
    const choice = await confirmThree({
      title: "Discard unsaved changes?",
      body: `"${doc.name}" has unsaved changes that aren't saved to disk.\n\n`
        + "Closing this tab discards them. Switch to the tab first if you "
        + "want to save it.",
      confirmLabel: "Discard & close",
      cancelLabel: "Cancel",
    });
    if (choice !== "confirm") return false;
    figureDiscarded = true;
  } else if (!doc.path && unsavedAnalysisDocIds().includes(docId)) {
    // An Untitled document holding Analysis work: its workflows can only be
    // saved with a file, so it gets the normal prompt even with an empty
    // figure. Save switches to it (if needed) and asks where.
    const choice = await confirmThree({
      title: "Unsaved analysis",
      body: `"${doc.name}" has Analysis workflows that are saved only with the document, `
        + "and it has never been saved.\n\nSave it before closing?",
      confirmLabel: "Save",
      tertiaryLabel: "Don't save",
      cancelLabel: "Cancel",
    });
    if (choice === "cancel") return false;
    if (choice === "confirm" && !(await saveDocumentImpl(docId))) return false;
  }
  // Saving may have switched to the document.
  isActiveDoc = useCollageStore.getState().activeDocId === docId;

  // Its Analysis workflows auto-save into its .mpf (no prompt). A failure is
  // reported before anything is closed.
  const failure = await persistWorkflowsOf(docId, figureDiscarded);
  if (failure) {
    const choice = await confirmThree({
      title: "Analysis not saved",
      body: `The Analysis workflows of "${doc.name}" could not be saved into its file.\n\n${failure}\n\n`
        + "Close anyway (they are lost), or cancel and keep the document open?",
      confirmLabel: "Close anyway",
      cancelLabel: "Cancel",
      destructive: true,
    });
    if (choice !== "confirm") return false;
  }

  // Drop any in-memory stash for the closed doc.
  dropStash(docId);
  syncSnapshotDirty();

  const remaining = useCollageStore.getState().openDocs.filter((d) => d.id !== docId);
  // Its workflows leave the Analysis workspace (shared ones stay, locked),
  // in the same tick as the tab.
  useAnalysisWorkspaceStore.getState().closeDoc(docId);
  useCollageStore.getState().docRemove(docId);

  if (isActiveDoc) {
    // The backend still holds the closed doc's state — replace it with a
    // neighbour (or a fresh blank) so the builder is consistent. Done even
    // in collage mode; we don't change the current mode.
    const next = remaining[0];
    await withDocSwap(async () => {
      if (next) {
        try {
          // Activate first, mark active after: if the load fails the tab strip
          // must not claim a document the backend doesn't hold (a later save
          // would write the closed doc's content into the neighbour's file).
          await activateDoc(next);
          useCollageStore.getState().docSetActive(next.id);
          return;
        } catch (e) {
          console.error("[projectNav] activate neighbour after close failed:", e);
          await alertDialog(loadFailureNotice(next.name, next.path, e));
        }
      }
      // No docs left (or the neighbour failed) — a fresh blank one, so the
      // builder and the active tab always match.
      const id = newDocId();
      await useFigureStore.getState().newBlankFigure(id).catch((e) => {
        console.error("[projectNav] new blank after close failed:", e);
      });
      useCollageStore.getState().docAdd({ id, path: null });
      useCollageStore.getState().docSetActive(id);
    });
  }
  return true;
}

/** Upload files into document `targetDocId` — switching to it and back when it
 *  isn't the live doc — as ONE doc op, so a tab click can't slip between the
 *  switch and the upload and send the files to a third document. Returns the
 *  uploaded names. */
export function uploadIntoDocument(targetDocId: string, files: File[]): Promise<string[]> {
  const run = useCollageStore.getState().activeDocId === targetDocId ? serializeDocOp : queueDocSwap;
  return run(async () => {
    const cs = useCollageStore.getState();
    const orig = cs.activeDocId;
    const mode0 = cs.mode;
    const switched = !!orig && targetDocId !== orig;
    if (switched && !(await switchToDocumentImpl(targetDocId))) {
      throw new Error("Could not open the target document.");
    }
    try {
      return await useFigureStore.getState()._uploadImagesNow(files);
    } finally {
      if (switched && orig) {
        await switchToDocumentImpl(orig);
        useCollageStore.getState().setMode(mode0);
      }
    }
  });
}

/** Public, serialized entry point for closing a document tab. */
export function closeDoc(docId: string): Promise<boolean> {
  // Closing the live doc replaces it (neighbour or blank) — shield from now.
  const run = useCollageStore.getState().activeDocId === docId ? queueDocSwap : serializeDocOp;
  return run(() => closeDocImpl(docId));
}

/* ── Figure tabs (several figures inside ONE document) ──────────────────
   Each op swaps the backend's live config (switch/add/duplicate/delete PUT
   a different figure), so it joins the same queue as document switch, save,
   open and close: a figure switch can never interleave with a stash or a
   save of the document it belongs to. */

/** Make figure `figureId` of the active document the one being edited. */
export function switchFigure(figureId: string): Promise<boolean> {
  return serializeDocOp(() => useFigureStore.getState()._switchFigure(figureId));
}

/** Add a blank figure tab to the active document and switch to it. */
export function addFigure(name?: string): Promise<void> {
  return serializeDocOp(() => useFigureStore.getState()._addFigure({ name }));
}

/** Copy figure `figureId` into a new tab next to it and switch to the copy. */
export function duplicateFigure(figureId: string): Promise<void> {
  return serializeDocOp(() => useFigureStore.getState()._duplicateFigure(figureId));
}

/** Remove figure `figureId` (never the last one). */
export function deleteFigure(figureId: string): Promise<void> {
  return serializeDocOp(() => useFigureStore.getState()._deleteFigure(figureId));
}

export function renameFigure(figureId: string, name: string): Promise<void> {
  return serializeDocOp(() => useFigureStore.getState()._renameFigure(figureId, name));
}

export function moveFigure(figureId: string, dir: -1 | 1): Promise<void> {
  return serializeDocOp(() => useFigureStore.getState()._moveFigure(figureId, dir));
}
