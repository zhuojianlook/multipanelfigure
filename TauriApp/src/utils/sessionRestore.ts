/* ──────────────────────────────────────────────────────────
   sessionRestore — persist & restore the set of open .mpf
   document tabs across app launches, gated by an opt-in user
   preference ("Re-open tabs on relaunch").

   Open-document tabs (collageStore.openDocs) are session-only:
   they're rebuilt each launch and never written to the collage's
   persisted state. To support reopening last session's tabs we
   mirror just their on-disk paths (+ display names) to
   localStorage on every change, and — when the preference is on —
   rebuild the tab strip from that record on the next launch.

   Untitled (never-saved) docs are skipped entirely: they have no
   file to reopen, and their unsaved edits don't survive a quit.
   ────────────────────────────────────────────────────────── */

import { useFigureStore } from "../store/figureStore";
import { useCollageStore, newDocId } from "../store/collageStore";
import { alert as alertDialog, confirm } from "../components/shared/ConfirmDialog";
import { serializeDocOp } from "./projectNav";
import { buildRef, isCloudDownloadError, loadFailureNotice, registryFileIdAt, registryUidAt } from "./docIdentity";
import { knownCopies, repointOpenDocs, resolveRefs } from "./docLinks";
import { basenameOf } from "./workflowPersist";

/** Preference flag: when "1", last session's .mpf tabs reopen on launch. */
const REOPEN_PREF_KEY = "mpfig_reopen_tabs_v1";
/** The mirrored open-document record (paths + active doc). */
const OPEN_DOCS_KEY = "mpfig_open_docs_v1";

/** `rank`: the tab's place in OPENING order (DocTab.seq) among the saved
 *  tabs. The list itself is in tab order, which the user can drag; the rank
 *  lets a relaunch re-open same-name tabs in their original order, so their
 *  "name (2)" labels (written into Analysis data) don't swap. */
type PersistedDoc = { path: string; name: string; rank?: number };
type PersistedSession = { docs: PersistedDoc[]; activePath: string | null };

/** Whether the "re-open tabs on relaunch" preference is on. Default OFF
 *  (opt-in) — a fresh install starts with a single blank document. */
export function getReopenPref(): boolean {
  try { return localStorage.getItem(REOPEN_PREF_KEY) === "1"; } catch { return false; }
}

/** Toggle the preference. Turning it ON captures the CURRENT session straight
 *  away, so relaunching right after the toggle still restores these tabs. */
export function setReopenPref(on: boolean): void {
  try { localStorage.setItem(REOPEN_PREF_KEY, on ? "1" : "0"); } catch { /* ignore */ }
  if (on) persistOpenDocs();
}

/** Mirror the current open .mpf tabs (those backed by a disk path) + the active
 *  doc's path to localStorage. Cheap and idempotent — safe to call on every
 *  document change. Stored regardless of the preference so toggling it on later
 *  (and crash recovery) has data to work with; only RESTORE honours the pref.
 *  Not while last session's record is still to be offered ("Reopen last
 *  session?" not answered yet): tabs seeded from the collage meanwhile must
 *  not replace it — a quit or crash at the prompt keeps it for next time. */
export function persistOpenDocs(): void {
  if (_restorePending) return;
  try {
    const { openDocs, activeDocId } = useCollageStore.getState();
    const withPath = openDocs.filter((d) => !!d.path);
    const byOpen = [...withPath].sort(
      (a, b) => (a.seq ?? Number.MAX_SAFE_INTEGER) - (b.seq ?? Number.MAX_SAFE_INTEGER));
    const rankOf = new Map(byOpen.map((d, i) => [d.id, i] as const));
    const docs: PersistedDoc[] = withPath
      .map((d) => ({ path: d.path as string, name: d.name, rank: rankOf.get(d.id) }));
    const active = openDocs.find((d) => d.id === activeDocId);
    const session: PersistedSession = { docs, activePath: active?.path ?? null };
    localStorage.setItem(OPEN_DOCS_KEY, JSON.stringify(session));
  } catch { /* quota / unavailable — ignore */ }
}

function readSession(): PersistedSession | null {
  try {
    const raw = localStorage.getItem(OPEN_DOCS_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (!data || !Array.isArray(data.docs)) return null;
    const docs: PersistedDoc[] = data.docs
      .filter((d: unknown): d is { path: string; name?: unknown; rank?: unknown } =>
        !!d && typeof (d as { path?: unknown }).path === "string")
      .map((d: { path: string; name?: unknown; rank?: unknown }) => ({
        path: d.path,
        name: typeof d.name === "string" ? d.name : "",
        ...(typeof d.rank === "number" && Number.isFinite(d.rank) ? { rank: d.rank } : {}),
      }));
    return {
      docs,
      activePath: typeof data.activePath === "string" ? data.activePath : null,
    };
  } catch { return null; }
}

/** The on-disk session captured ONCE at module load — before any React effect
 *  can run and before the first persistOpenDocs() write. maybeRestoreSession
 *  reads this snapshot rather than localStorage so a restore is immune to the
 *  startup write order (e.g. the persistence subscription, or tab seeding from
 *  collage figures, mutating openDocs before restore gets to read). */
const _bootSession: PersistedSession | null = readSession();

/** One-shot guard so the launch restore (and its prompt) runs at most once,
 *  even if an effect double-invokes (React StrictMode in dev). */
let _restoreAttempted = false;

/** Last session's record will be offered (the preference is on and it has
 *  tabs) and maybeRestoreSession has not settled yet: persistOpenDocs keeps
 *  it as it is meanwhile (tab seeding from the collage runs asynchronously,
 *  after AppShell's open-docs subscriber started). */
let _restorePending = getReopenPref() && !!_bootSession?.docs.length;

/** A file's name without ".mpf" (what a tab named after its file shows). */
function stem(path: string): string {
  return basenameOf(path).replace(/\.mpf$/i, "");
}

/** On launch, when the preference is on, offer to rebuild the document tabs
 *  from the last session.
 *
 *  - Each file is looked up FIRST, by the document uid the last-seen table
 *    saw at its path (one cheap api.resolveMpf call): a file renamed or moved
 *    since last launch reopens where it is now (found in its folder or where
 *    it was seen before); one found nowhere is dropped — never a phantom tab
 *    — and named in the prompt; a path that still holds a file is never
 *    dropped, even when its uid turned up only in a copy it must not follow.
 *    If none survive, we silently start clean.
 *  - Until the prompt is answered, the saved record is not overwritten
 *    (persistOpenDocs); reopened, the restored tabs are recorded.
 *  - The user is then ASKED whether to reopen the surviving tabs or start with
 *    a blank document (the preference enables the offer; it doesn't force it).
 *
 *  On "reopen": a "cold" tab (path only — not loaded into the backend) is made
 *  for every surviving doc, and the previously-active one (or the first if it's
 *  gone) is loaded into the builder. Cold tabs load lazily when clicked.
 *
 *  Returns true if a project was loaded into the builder (so the caller can
 *  skip its own preview kick — loadProject already requests one). Returns false
 *  when the pref is off, nothing survives, the user declines, or the load
 *  fails. */
export async function maybeRestoreSession(): Promise<boolean> {
  if (_restoreAttempted) return false; // one shot per launch (guards StrictMode double-invoke)
  _restoreAttempted = true;
  let built = false;
  try {
    return await restoreNow(() => { built = true; });
  } finally {
    if (_restorePending) {
      _restorePending = false;
      // Reopened: the restored tabs are recorded now (the writes skipped
      // while the prompt was up). Start fresh / nothing left: last session's
      // record stays until the tabs next change, as before.
      if (built) persistOpenDocs();
    }
  }
}

async function restoreNow(onBuilt: () => void): Promise<boolean> {
  if (!getReopenPref()) return false;
  const session = _bootSession;
  if (!session || session.docs.length === 0) return false;

  // Where each .mpf is now. Renamed / moved since last launch: it reopens at
  // its new path (a tab named after the old file takes the new name). Gone:
  // dropped, so deleted files don't reopen as empty phantom tabs.
  const found = await resolveRefs(session.docs.map((d, i) =>
    buildRef(String(i), d.path, registryUidAt(d.path), [], registryFileIdAt(d.path))));
  const liveDocs: PersistedDoc[] = [];
  const moves: { from: string; to: string; uid: string | null; fileId?: string | null }[] = [];
  const pathFor = new Map<string, string>();
  const notFound: string[] = [];
  session.docs.forEach((d, i) => {
    const r = found[String(i)];
    // Not into a file the last-seen table saw carrying the uid next to this
    // one (knownCopies: a Finder copy's original, or the other way round).
    // Not followed (that, or several / a possible copy: ambiguous) while a
    // file is still at the path: it reopens there, as before files had uids.
    const follow = r?.status === "moved" && !!r.path && !knownCopies(d.path, r.path, registryUidAt(d.path));
    const path = r?.status === "ok" ? d.path
      : follow ? (r!.path as string)
      : r?.exists ? d.path : null;
    if (!path) { notFound.push(basenameOf(d.path) || d.path); return; }
    pathFor.set(d.path, path);
    if (liveDocs.some((x) => x.path === path)) return;        // two entries, one file
    if (path !== d.path) {
      moves.push({ from: d.path, to: path, uid: r?.uid ?? null, ...(r?.file_id ? { fileId: r.file_id } : {}) });
      liveDocs.push({ ...d, path, name: !d.name || d.name === stem(d.path) ? stem(path) : d.name });
    } else {
      liveDocs.push(d);
    }
  });
  if (liveDocs.length === 0) return false; // nothing left → start clean

  // Even with the preference on, confirm before reopening so the user can
  // choose a clean start.
  const n = liveDocs.length;
  const ok = await confirm({
    title: "Reopen last session?",
    body: `You have ${n} saved figure ${n === 1 ? "tab" : "tabs"} from your last session.\n\n`
      + "Reopen them, or start with a blank document?"
      + (notFound.length ? `\n\nNot found (moved or deleted): ${notFound.join(", ")}` : ""),
    confirmLabel: n === 1 ? "Reopen tab" : `Reopen ${n} tabs`,
    cancelLabel: "Start fresh",
  });
  if (!ok) return false;

  // Build the tabs + load the active doc as ONE serialized operation, so a tab
  // click during the (slow) active load can't race the backend and leave the
  // session half-restored.
  return await serializeDocOp(async () => {
    const cs = useCollageStore.getState();
    // Files found at a new path: what still names the old one follows them
    // (a tab seeded from the collage meanwhile, collage figures, the labels
    // kept under the old path, the last-seen table) — before any tab opens.
    // Computed before the prompt: a move another one (seeding) already made
    // finds that path taken by the file now there, and leaves it.
    repointOpenDocs(moves);
    // Pick which doc lands in the builder: the previously-active one if it still
    // exists (wherever it is now), else the first surviving one.
    const wasActive = session.activePath ? pathFor.get(session.activePath) : undefined;
    const activePath = wasActive && liveDocs.some((d) => d.path === wasActive)
      ? wasActive
      : liveDocs[0].path;

    // Shape the tab strip to its FINAL form in one synchronous batch — add the
    // restored cold tabs, drop the auto-seeded blank "Untitled" tab, and set the
    // active tab — BEFORE the (slow) load. Doing the removal after the await let
    // React paint a transient state with the stray Untitled_1 sitting next to
    // the restored tabs (the "3 tabs for a moment" glitch). No awaits here, so
    // the strip renders its final shape directly.
    // Re-open in OPENING order (rank; sessions saved before ranks existed:
    // list order), so each tab's seq — and with it any "name (2)" label —
    // matches last session. They are shown in the saved TAB order below.
    const byRank = liveDocs
      .map((d, i) => ({ d, i }))
      .sort((a, b) => (a.d.rank ?? a.i) - (b.d.rank ?? b.i) || a.i - b.i);
    const idOf = new Map<string, string>();
    for (const { d } of byRank) idOf.set(d.path, cs.docEnsure(d.path, d.name || undefined));
    const sessionIds = liveDocs.map((d) => idOf.get(d.path) as string);
    const activeId = useCollageStore.getState().docEnsure(activePath);
    for (const d of useCollageStore.getState().openDocs) {
      if (!d.path && d.id !== activeId) useCollageStore.getState().docRemove(d.id);
    }
    // A tab seeded earlier from a collage figure already existed, so it kept
    // its earlier seq: hand the restored tabs' seqs back out in rank order.
    useCollageStore.getState().docSetOpenOrder(byRank.map(({ d }) => idOf.get(d.path) as string));
    // Put the tabs back in the saved (possibly user-dragged) order. The
    // seeded tab sits wherever docEnsure found it; tabs not in the session
    // keep their order after these.
    {
      const curIds = useCollageStore.getState().openDocs.map((d) => d.id);
      const first = [...new Set(sessionIds)].filter((id) => curIds.includes(id));
      const rest = curIds.filter((id) => !first.includes(id));
      useCollageStore.getState().docReorder([...first, ...rest]);
    }
    useCollageStore.getState().docSetActive(activeId);
    useCollageStore.getState().setMode("builder");
    onBuilt();

    // Now fill the builder with the active doc's content.
    try {
      await useFigureStore.getState().loadProject(activePath, activeId);
    } catch (e) {
      // Rare: the file vanished between the existence check and the load, or
      // it is stored online only and its cloud app could not download it yet
      // (not running so soon after login, or offline). Reset to a clean blank
      // working doc so the active tab and backend stay consistent; the
      // restored tab remains and loads on click — say so for a cloud file,
      // which is not lost.
      if (isCloudDownloadError(e)) {
        const tab = useCollageStore.getState().openDocs.find((d) => d.id === activeId);
        const notice = loadFailureNotice(tab?.name || basenameOf(activePath), activePath, e);
        void alertDialog({ ...notice, body: `${notice.body} Its tab stays open: click it to load it.` });
      }
      const blankId = newDocId();
      await useFigureStore.getState().newBlankFigure(blankId).catch(() => {});
      useCollageStore.getState().docAdd({ id: blankId, path: null });
      useCollageStore.getState().docSetActive(blankId);
      return false;
    }
    return true;
  });
}
