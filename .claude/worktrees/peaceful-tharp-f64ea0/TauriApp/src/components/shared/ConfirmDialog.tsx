/* ──────────────────────────────────────────────────────────
   ConfirmDialog — shared MUI-styled replacement for the
   browser's native window.confirm / window.alert (and ad-hoc
   alert(...) calls).  Matches the style of the toolbar's
   "New Figure" warning so every confirmation across the app
   reads as a single coherent component:
     • MUI Dialog with DialogTitle / DialogContent / DialogActions
     • Body wrapped in <Typography> so it inherits the theme
     • Destructive action button uses variant="contained"
       color="error"; non-destructive uses color="primary"
     • Single OK button for alert-style messages

   Usage (imperative, anywhere in the app):
     const ok = await confirm({
       title: "Replace image",
       body: "An image is already loaded in this cell.  Replace it?",
       confirmLabel: "Replace",
       destructive: true,
     });
     if (!ok) return;

     await alert({ title: "Done", body: "Export succeeded." });

   The host component <ConfirmHost /> must be mounted ONCE near
   the root of the React tree (we mount it inside <App>).  It
   exposes a module-level queue that confirm()/alert() push to.
   ────────────────────────────────────────────────────────── */

import { useEffect, useState } from "react";
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Typography,
  Button,
} from "@mui/material";

interface ConfirmRequest {
  title: string;
  body: string;
  /** Label for the affirmative action button. */
  confirmLabel?: string;
  /** Label for the cancel button. Set to null to hide cancel
   *  (turns this into a single-OK "alert"). */
  cancelLabel?: string | null;
  /** When true, the confirm button uses color="error" (red).
   *  Use for destructive actions — delete, overwrite, etc. */
  destructive?: boolean;
  /** Resolver attached by the caller; receives true on confirm,
   *  false on cancel / dialog close. */
  resolve: (ok: boolean) => void;
}

// Module-level pub/sub for the singleton host.  We avoid a global
// store (Zustand / Context) so the helper is drop-in importable
// from any file without wiring providers.  Subscribers are the
// host component(s) — typically just one.
type Listener = (r: ConfirmRequest) => void;
const listeners = new Set<Listener>();

function dispatch(r: ConfirmRequest) {
  // If no host is mounted yet, fall back to the native dialog so
  // callers still get a result rather than hanging forever.  This
  // is a safety net for early-startup edge cases; in normal use
  // the host is mounted before the first confirm() call.
  if (listeners.size === 0) {
    const ok = r.cancelLabel === null
      ? (window.alert(r.body), true)
      : window.confirm(`${r.title}\n\n${r.body}`);
    r.resolve(!!ok);
    return;
  }
  listeners.forEach((fn) => fn(r));
}

/** Imperative confirm — returns a promise that resolves to true on
 *  the affirmative button, false on cancel / Esc / backdrop click. */
export function confirm(
  opts: Omit<ConfirmRequest, "resolve">,
): Promise<boolean> {
  return new Promise<boolean>((resolve) => {
    dispatch({ ...opts, resolve });
  });
}

/** Imperative alert — single OK button, returns a promise that
 *  resolves to true when dismissed.  Use for info-only messages
 *  that just need acknowledgement (no choice). */
export function alert(
  opts: Omit<ConfirmRequest, "resolve" | "cancelLabel" | "destructive">,
): Promise<true> {
  return new Promise<true>((resolve) => {
    dispatch({
      ...opts,
      cancelLabel: null,
      destructive: false,
      resolve: () => resolve(true),
    });
  });
}

/** Singleton host — mount ONCE in the React tree.  Receives
 *  dispatched requests via the module-level pub/sub and renders
 *  the topmost one as an MUI Dialog. */
export function ConfirmHost() {
  // Queue of pending requests.  Multiple confirms in flight is
  // unusual but valid — we show them serially so each gets a
  // proper modal experience.
  const [queue, setQueue] = useState<ConfirmRequest[]>([]);

  useEffect(() => {
    const onRequest: Listener = (r) => setQueue((q) => [...q, r]);
    listeners.add(onRequest);
    return () => { listeners.delete(onRequest); };
  }, []);

  const current = queue[0];
  if (!current) return null;

  const close = (result: boolean) => {
    current.resolve(result);
    setQueue((q) => q.slice(1));
  };

  const cancelLabel = current.cancelLabel === undefined ? "Cancel" : current.cancelLabel;
  const confirmLabel = current.confirmLabel ?? "OK";

  return (
    <Dialog
      open
      onClose={() => close(false)}
      // Same defaults as the "New Figure" warning — small size,
      // not full-screen, click-outside cancels.
      maxWidth="xs"
      fullWidth
    >
      <DialogTitle>{current.title}</DialogTitle>
      <DialogContent>
        <Typography sx={{ whiteSpace: "pre-wrap" }}>{current.body}</Typography>
      </DialogContent>
      <DialogActions>
        {cancelLabel !== null && (
          <Button onClick={() => close(false)}>{cancelLabel}</Button>
        )}
        <Button
          variant="contained"
          color={current.destructive ? "error" : "primary"}
          autoFocus
          onClick={() => close(true)}
        >
          {confirmLabel}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
