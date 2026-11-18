/* ── Document-operation queue (leaf module, no imports) ──────────────────
   The Python sidecar holds ONE builder document in shared globals. Its doc
   gate (api_server._DocGate) keeps a request from running INSIDE Analysis's
   temporary swap to another document, but it does not order two ordinary
   requests against each other. If two document operations overlap — a tab
   click while a switch is still loading, a save mid-switch, an upload that is
   still sending files when the user clicks another tab — their stash /
   activate / load / upload calls interleave and content lands in the wrong
   document. Every top-level doc operation therefore runs through this single
   promise chain, strictly one at a time.

   Rule: never await serializeDocOp from INSIDE a doc op (there is no async
   re-entrancy in JS — it would deadlock). Internal helpers (…Impl,
   stashCurrentDoc, activateDoc, _uploadImagesNow) run inside an op instead.

   Lives in its own module so figureStore (uploads) and projectNav (tab ops)
   can share it without an import cycle. */

let _docOpChain: Promise<unknown> = Promise.resolve();

/** Run `fn` after all previously-queued document operations finish, and make
 *  later ones wait for it. Failures don't break the chain (the next op still
 *  runs). */
export function serializeDocOp<T>(fn: () => Promise<T>): Promise<T> {
  const run = _docOpChain.then(fn, fn);
  _docOpChain = run.then(() => {}, () => {});
  return run;
}

/** Resolves once the doc-op queue is empty, including ops queued while
 *  waiting. Background readers (the Analysis sources listing) wait on this so
 *  they never start while a tab switch is mid-flight. */
export async function whenDocOpsIdle(): Promise<void> {
  let c: Promise<unknown>;
  do { c = _docOpChain; await c; } while (c !== _docOpChain);
}

/** True when an error from activate-state means "this doc has no cached state
 *  in the sidecar" (e.g. the sidecar restarted) — the only case in which the
 *  caller may fall back to the on-disk .mpf. Anything else (a transient
 *  failure) must keep the stash and the unsaved flag. The Rust proxy hides the
 *  HTTP status, so this matches the backend's stable detail string. */
export function isStateCacheMiss(e: unknown): boolean {
  const msg = e instanceof Error ? e.message : String(e ?? "");
  return /No cached state|no_cached_state/.test(msg);
}

/* Whether the sidecar tracks which frontend document is live (it answered
   /api/project/bind-live). Only then do analysis sources of the ACTIVE doc
   carry its id: an older sidecar would resolve such an id from disk and
   ignore unsaved edits. Set once per webview load by figureStore.fetchConfig. */
let _liveDocTracked = false;
export const backendTracksLiveDoc = (): boolean => _liveDocTracked;
export const setBackendTracksLiveDoc = (v: boolean): void => { _liveDocTracked = v; };
