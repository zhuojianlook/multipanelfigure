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
import { api } from "../api/client";
import { confirm } from "../components/shared/ConfirmDialog";
import { serializeDocOp } from "./projectNav";

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
 *  (and crash recovery) has data to work with; only RESTORE honours the pref. */
export function persistOpenDocs(): void {
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

/** On launch, when the preference is on, offer to rebuild the document tabs
 *  from the last session.
 *
 *  - Files deleted/moved since last launch are filtered out FIRST (via a cheap
 *    existence check) so they never reopen as phantom tabs. If none survive,
 *    we silently start clean.
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
  if (!getReopenPref()) return false;
  const session = _bootSession;
  if (!session || session.docs.length === 0) return false;

  // Drop docs whose .mpf no longer exists so deleted files don't reopen as
  // empty phantom tabs.
  const existMap = await api.pathsExist(session.docs.map((d) => d.path));
  const liveDocs = session.docs.filter((d) => existMap[d.path]);
  if (liveDocs.length === 0) return false; // nothing left → start clean

  // Even with the preference on, confirm before reopening so the user can
  // choose a clean start.
  const n = liveDocs.length;
  const ok = await confirm({
    title: "Reopen last session?",
    body: `You have ${n} saved figure ${n === 1 ? "tab" : "tabs"} from your last session.\n\n`
      + "Reopen them, or start with a blank document?",
    confirmLabel: n === 1 ? "Reopen tab" : `Reopen ${n} tabs`,
    cancelLabel: "Start fresh",
  });
  if (!ok) return false;

  // Build the tabs + load the active doc as ONE serialized operation, so a tab
  // click during the (slow) active load can't race the backend and leave the
  // session half-restored.
  return serializeDocOp(async () => {
    const cs = useCollageStore.getState();
    // Pick which doc lands in the builder: the previously-active one if it still
    // exists, else the first surviving one.
    const activePath = (session.activePath && liveDocs.some((d) => d.path === session.activePath))
      ? session.activePath
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

    // Now fill the builder with the active doc's content.
    try {
      await useFigureStore.getState().loadProject(activePath, activeId);
    } catch {
      // Rare: the file vanished between the existence check and the load. Reset
      // to a clean blank working doc so the active tab and backend stay
      // consistent (the other restored tabs remain and load on click).
      const blankId = newDocId();
      await useFigureStore.getState().newBlankFigure(blankId).catch(() => {});
      useCollageStore.getState().docAdd({ id: blankId, path: null });
      useCollageStore.getState().docSetActive(blankId);
      return false;
    }
    return true;
  });
}
