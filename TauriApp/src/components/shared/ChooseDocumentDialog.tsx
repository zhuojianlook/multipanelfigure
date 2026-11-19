/* ──────────────────────────────────────────────────────────
   ChooseDocumentDialog — "which document should this be saved in?"

   Asked when an Analysis workflow that reads no figure document's
   data (it uses uploaded images only) first gets an upload: such a
   workflow is saved — with its uploaded images embedded — in the one
   document chosen here. Lists the open documents (the current one
   pre-selected) plus "New document…".

   Same promise + singleton-host pattern as ConfirmDialog:
     const r = await chooseDocument({ title, body, options, selectedId });
     if (!r) return;                    // Cancel / Esc / backdrop
     if (r.kind === "new") …            // "New document…"
     else useDoc(r.id);

   <ChooseDocumentHost /> is mounted ONCE (in <App>, beside
   <ConfirmHost />). With no host mounted the request resolves null
   (= Cancel: the caller keeps what it had).
   ────────────────────────────────────────────────────────── */

import { useEffect, useState } from "react";
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControlLabel,
  Radio,
  RadioGroup,
  Typography,
} from "@mui/material";

export interface ChooseDocumentOption {
  id: string;
  label: string;
  /** Second line (e.g. the file name, or "not saved yet"). */
  detail?: string;
}

export type ChooseDocumentResult = { kind: "doc"; id: string } | { kind: "new" } | null;

export interface ChooseDocumentOpts {
  title: string;
  body: string;
  options: ChooseDocumentOption[];
  /** The option selected when the dialog opens (the current choice). */
  selectedId?: string;
  /** Label of the extra "new document" choice; null hides it. */
  newLabel?: string | null;
  /** Affirmative button label. */
  confirmLabel?: string;
}

interface ChooseRequest extends ChooseDocumentOpts {
  resolve: (r: ChooseDocumentResult) => void;
}

const NEW_VALUE = "\u0000new";

type Listener = (r: ChooseRequest) => void;
const listeners = new Set<Listener>();

/** Ask which document to use. Resolves the chosen open document, "new", or
 *  null on Cancel / Esc / backdrop click (and when no host is mounted). */
export function chooseDocument(opts: ChooseDocumentOpts): Promise<ChooseDocumentResult> {
  return new Promise<ChooseDocumentResult>((resolve) => {
    const r: ChooseRequest = { ...opts, resolve };
    if (listeners.size === 0) { resolve(null); return; }
    listeners.forEach((fn) => fn(r));
  });
}

function initialValue(r: ChooseRequest): string {
  if (r.selectedId && r.options.some((o) => o.id === r.selectedId)) return r.selectedId;
  if (r.options.length) return r.options[0].id;
  return NEW_VALUE;
}

/** Singleton host — mount ONCE in the React tree. Requests are shown one at
 *  a time, in order. */
export function ChooseDocumentHost() {
  const [queue, setQueue] = useState<ChooseRequest[]>([]);
  const [value, setValue] = useState("");

  useEffect(() => {
    const onRequest: Listener = (r) => setQueue((q) => [...q, r]);
    listeners.add(onRequest);
    return () => { listeners.delete(onRequest); };
  }, []);

  const current = queue[0];
  // Each request opens on its own pre-selection.
  useEffect(() => { if (current) setValue(initialValue(current)); }, [current]);

  if (!current) return null;
  const showNew = current.newLabel !== null;
  const selected = value || initialValue(current);

  const close = (result: ChooseDocumentResult) => {
    current.resolve(result);
    setQueue((q) => q.slice(1));
    setValue("");
  };
  const confirm = () => {
    if (selected === NEW_VALUE) close(showNew ? { kind: "new" } : null);
    else close({ kind: "doc", id: selected });
  };

  return (
    <Dialog open onClose={() => close(null)} maxWidth="xs" fullWidth>
      <DialogTitle>{current.title}</DialogTitle>
      <DialogContent>
        <Typography sx={{ whiteSpace: "pre-wrap", mb: 1 }}>{current.body}</Typography>
        <RadioGroup value={selected} onChange={(e) => setValue(e.target.value)}>
          {current.options.map((o) => (
            <FormControlLabel
              key={o.id}
              value={o.id}
              control={<Radio size="small" />}
              label={(
                <span>
                  <Typography component="span" variant="body2">{o.label}</Typography>
                  {o.detail && (
                    <Typography component="span" variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                      {o.detail}
                    </Typography>
                  )}
                </span>
              )}
            />
          ))}
          {showNew && (
            <FormControlLabel
              value={NEW_VALUE}
              control={<Radio size="small" />}
              label={<Typography variant="body2">{current.newLabel ?? "New document…"}</Typography>}
            />
          )}
        </RadioGroup>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => close(null)}>Cancel</Button>
        <Button variant="contained" autoFocus onClick={confirm}>{current.confirmLabel ?? "OK"}</Button>
      </DialogActions>
    </Dialog>
  );
}
