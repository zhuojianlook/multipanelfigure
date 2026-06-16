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
import { useCollageStore } from "../store/collageStore";
import { api } from "../api/client";
import { confirm } from "../components/shared/ConfirmDialog";
import { serializeDocOp } from "./projectNav";

/** Preference flag: when "1", last session's .mpf tabs reopen on launch. */
const REOPEN_PREF_KEY = "mpfig_reopen_tabs_v1";
/** The mirrored open-document record (paths + active doc). */
const OPEN_DOCS_KEY = "mpfig_open_docs_v1";

type PersistedDoc = { path: string; name: string };
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
    const docs: PersistedDoc[] = openDocs
      .filter((d) => !!d.path)
      .map((d) => ({ path: d.path as string, name: d.name }));
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
      .filter((d: unknown): d is { path: string; name?: unknown } =>
        !!d && typeof (d as { path?: unknown }).path === "string")
      .map((d: { path: string; name?: unknown }) => ({
        path: d.path,
        name: typeof d.name === "string" ? d.name : "",
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
    // Cold tabs for every surviving doc (docEnsure dedupes by path, so tabs
    // already seeded from collage figures aren't duplicated).
    for (const d of liveDocs) cs.docEnsure(d.path, d.name || undefined);

    // Load the previously-active doc if it still exists, else the first
    // surviving one — so the user lands back where they left off.
    const activePath = (session.activePath && liveDocs.some((d) => d.path === session.activePath))
      ? session.activePath
      : liveDocs[0].path;

    try {
      await useFigureStore.getState().loadProject(activePath);
    } catch {
      // Raced deletion between the existence check and the load — bail to a
      // clean blank rather than a half-restored state.
      return false;
    }

    const activeId = useCollageStore.getState().docEnsure(activePath);
    // Drop the auto-seeded blank "Untitled" tab(s). At startup these are always
    // the untouched initial doc, now superseded by the restored session — left
    // in place they'd read as a stray empty tab.
    for (const d of useCollageStore.getState().openDocs) {
      if (!d.path && d.id !== activeId) useCollageStore.getState().docRemove(d.id);
    }
    useCollageStore.getState().docSetActive(activeId);
    useCollageStore.getState().setMode("builder");
    return true;
  });
}
