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
import { useCollageStore } from "../store/collageStore";
import { api } from "../api/client";
import { confirmThree, alert as alertDialog } from "../components/shared/ConfirmDialog";
import { saveProjectDialog } from "../components/shared/SaveProjectDialog";

/* ── Per-document in-memory snapshots ───────────────────────────────────
   The Python backend holds exactly ONE builder document's state (config +
   images + fonts) at a time — loading a project clears the previous one. To
   make switching between document tabs SEAMLESS without losing a tab's
   unsaved edits, we snapshot the outgoing tab to an in-memory .mpf blob
   (api.snapshotProject) before swapping, and restore it (figureStore
   .restoreDoc) when the user returns. Clean tabs (already on disk, or
   blank) need no snapshot — they reload from disk / reset to blank.
   Session-only; never persisted. */
const docSnapshots = new Map<string, string>();

/** Push the current snapshot doc-id set into the collage store so the tab
 *  strip can show an unsaved dot on tabs whose edits aren't live. */
function syncSnapshotDirty() {
  useCollageStore.getState().setSnapshotDirtyDocIds(Array.from(docSnapshots.keys()));
}

/** True if any open document has unsaved in-memory edits (a snapshot).
 *  Used by the app-close guard to warn about tabs whose edits would be
 *  lost on quit. */
export function hasUnsavedSnapshots(): boolean {
  return docSnapshots.size > 0;
}

/** Snapshot the currently-active builder doc to memory IF it has unsaved
 *  edits, so switching away doesn't lose work. No-op for clean docs (they
 *  reload from disk / reset to blank on return) and for an active doc that
 *  was already removed (e.g. mid-close). */
async function stashCurrentDoc(): Promise<void> {
  const cs = useCollageStore.getState();
  const fs = useFigureStore.getState();
  const activeId = cs.activeDocId;
  if (!activeId) return;
  if (!cs.openDocs.some((d) => d.id === activeId)) return; // already removed
  if (!fs.unsaved) { docSnapshots.delete(activeId); syncSnapshotDirty(); return; }
  try {
    const { blob } = await api.snapshotProject();
    docSnapshots.set(activeId, blob);
  } catch (e) {
    console.error("[projectNav] snapshot failed for", activeId, e);
  }
  syncSnapshotDirty();
}

/** Load a document's state into the backend builder: prefer an in-memory
 *  snapshot (preserves unsaved edits), else the on-disk .mpf, else blank. */
async function activateDoc(doc: { id: string; path: string | null }): Promise<void> {
  const snap = docSnapshots.get(doc.id);
  if (snap) {
    try {
      await useFigureStore.getState().restoreDoc(snap, { path: doc.path });
      docSnapshots.delete(doc.id); // now live in the backend
      syncSnapshotDirty();
      return;
    } catch (e) {
      console.error("[projectNav] restore failed; falling back to disk/blank", e);
      docSnapshots.delete(doc.id);
      syncSnapshotDirty();
    }
  }
  if (doc.path) {
    await useFigureStore.getState().loadProject(doc.path);
  } else {
    await useFigureStore.getState().newBlankFigure();
  }
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
  const { currentProjectPath, unsaved, saveProject } = useFigureStore.getState();
  if (currentProjectPath && !unsaved) return currentProjectPath;

  const picked = await saveProjectDialog({
    defaultPath: currentProjectPath || defaultProjectName(),
  });
  if (!picked) return null;
  await saveProject(picked);
  syncActiveDocPath();
  return useFigureStore.getState().currentProjectPath || picked;
}

/** Timestamped default project filename, matching the Sidebar's
 *  Save Project button so an Untitled doc gets a sensible name. */
function defaultProjectName(): string {
  const now = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  const ts = `${now.getFullYear()}${p(now.getMonth() + 1)}${p(now.getDate())}`
    + `_${p(now.getHours())}${p(now.getMinutes())}${p(now.getSeconds())}`;
  return `${ts}_project.mpf`;
}

/** If the builder has unsaved changes, prompt Save / Don't save /
 *  Cancel. Returns true to proceed, false to abort. */
export async function maybeSaveBeforeLeavingBuilder(): Promise<boolean> {
  const { unsaved, currentProjectPath } = useFigureStore.getState();
  if (!unsaved) return true;
  const name = currentProjectPath ? currentProjectPath.split(/[\\/]/).pop() : "this figure";
  const choice = await confirmThree({
    title: "Unsaved changes",
    body: `You have unsaved changes to ${name}.\n\nSave before continuing?`,
    confirmLabel: "Save",
    tertiaryLabel: "Don't save",
    cancelLabel: "Cancel",
  });
  if (choice === "cancel") return false;
  if (choice === "confirm") {
    const saved = await ensureProjectSaved();
    if (!saved) return false;
  }
  return true;
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
 *  any), else loaded from disk, else opened blank. */
export async function switchToDocument(docId: string): Promise<void> {
  const cs = useCollageStore.getState();
  const doc = cs.openDocs.find((d) => d.id === docId);
  if (!doc) return;
  // The backend already holds this doc — just ensure builder mode. Covers
  // returning from the Collage tab to the active figure with NO reload, so
  // its in-progress edits stay intact.
  if (cs.activeDocId === docId) {
    if (cs.mode !== "builder") useCollageStore.getState().setMode("builder");
    return;
  }
  // Switching to a different doc: snapshot the outgoing one (if dirty),
  // then activate the target.
  await stashCurrentDoc();
  try {
    await activateDoc(doc);
  } catch (e) {
    console.error("[projectNav] switch document failed:", e);
    await alertDialog({
      title: "Load failed",
      body: "Could not load that document — its .mpf may have been moved or deleted.",
    });
    return;
  }
  useCollageStore.getState().docSetActive(docId);
  useCollageStore.getState().setMode("builder");
}

/** Create a new blank document tab and switch to it. Seamless — the prior
 *  tab is snapshotted (if dirty) so its edits survive; closing it is what
 *  guards. */
export async function newBlankDoc(): Promise<void> {
  await stashCurrentDoc();
  await useFigureStore.getState().newBlankFigure();
  const id = useCollageStore.getState().docAdd({ path: null, name: "Untitled" });
  useCollageStore.getState().docSetActive(id);
  useCollageStore.getState().setMode("builder");
}

/** Open a project file into a (new or existing) document tab. Seamless —
 *  the prior tab is snapshotted (if dirty) so its edits survive. */
export async function openProjectIntoTab(path: string): Promise<void> {
  await stashCurrentDoc();
  try {
    await useFigureStore.getState().loadProject(path);
  } catch (e) {
    console.error("[projectNav] open project failed:", e);
    await alertDialog({
      title: "Load failed",
      body: "Could not load that project file — it may have been moved or deleted.",
    });
    return;
  }
  const id = useCollageStore.getState().docEnsure(path);
  // Freshly loaded from disk — drop any stale in-memory snapshot for it.
  docSnapshots.delete(id);
  syncSnapshotDirty();
  useCollageStore.getState().docSetActive(id);
  useCollageStore.getState().setMode("builder");
}

/** Close a document tab. Prompts to save if it's the active dirty doc.
 *  Returns true if it was closed, false if the user cancelled. */
export async function closeDoc(docId: string): Promise<boolean> {
  const cs = useCollageStore.getState();
  const fs = useFigureStore.getState();
  const doc = cs.openDocs.find((d) => d.id === docId);
  if (!doc) return false;

  // Block closing a tab whose .mpf is placed in the collage — the
  // collage item depends on that figure. The user must remove it from
  // the collage first.
  if (doc.path) {
    const usedInCollage = cs.items.some(
      (it) => it.kind === "figure" && it.projectPath === doc.path,
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
  const isActiveDoc = cs.activeDocId === docId;

  if (isActiveDoc && fs.unsaved) {
    // Closing the live doc with unsaved changes — guard it (Save / Don't
    // save / Cancel; Save opens the file dialog so the user picks where +
    // what name).
    const ok = await maybeSaveBeforeLeavingBuilder();
    if (!ok) return false;
  } else if (!isActiveDoc && docSnapshots.has(docId)) {
    // A non-active tab with unsaved edits held only as an in-memory
    // snapshot. We can't save it without making it active first, so offer
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
  }

  // Drop any in-memory snapshot for the closed doc.
  docSnapshots.delete(docId);
  syncSnapshotDirty();

  const remaining = useCollageStore.getState().openDocs.filter((d) => d.id !== docId);
  useCollageStore.getState().docRemove(docId);

  if (isActiveDoc) {
    // The backend still holds the closed doc's state — replace it with a
    // neighbour (or a fresh blank) so the builder is consistent. Done even
    // in collage mode; we don't change the current mode.
    const next = remaining[0];
    if (next) {
      useCollageStore.getState().docSetActive(next.id);
      try {
        await activateDoc(next);
      } catch (e) {
        console.error("[projectNav] activate neighbour after close failed:", e);
      }
    } else {
      // No docs left — make a fresh blank one so the builder isn't empty.
      await useFigureStore.getState().newBlankFigure();
      const id = useCollageStore.getState().docAdd({ path: null, name: "Untitled" });
      useCollageStore.getState().docSetActive(id);
    }
  }
  return true;
}
