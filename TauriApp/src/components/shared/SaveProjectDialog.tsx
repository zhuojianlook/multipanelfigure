/* ──────────────────────────────────────────────────────────
   SaveProjectDialog — the single, shared "Save Project" modal.

   This is the SAME rich modal used by the Sidebar's "Save Project"
   button: a "File path" field the user can type into, a Browse
   button that opens the native Tauri save dialog, and Save / Cancel
   actions.  Exposed imperatively so any flow (the Sidebar button,
   the close-tab save guard, "Add to Collage") shows one identical
   dialog rather than each rolling its own.

   Usage (imperative, anywhere):
     const path = await saveProjectDialog({ defaultPath: "project.mpf" });
     if (!path) return;            // user cancelled
     await saveProject(path);      // caller performs the actual save

   The host <SaveProjectHost /> must be mounted ONCE near the root
   (mounted in <App> alongside <ConfirmHost />).
   ────────────────────────────────────────────────────────── */

import { useEffect, useState } from "react";
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Button,
  TextField,
  Typography,
  Box,
} from "@mui/material";

export interface SaveProjectDialogOpts {
  /** Dialog heading. Defaults to "Save Project". */
  title?: string;
  /** Initial value of the path field. */
  defaultPath?: string;
  /** Affirmative button label. Defaults to "Save". */
  confirmLabel?: string;
}

interface SaveRequest extends SaveProjectDialogOpts {
  resolve: (path: string | null) => void;
}

// Module-level pub/sub for the singleton host — same drop-in pattern
// as ConfirmDialog so this is importable from any file without
// wiring a provider.
type Listener = (r: SaveRequest) => void;
const listeners = new Set<Listener>();

function dispatch(r: SaveRequest) {
  // Safety net: if no host is mounted yet, fall back to a native
  // prompt so the caller still gets a result rather than hanging.
  if (listeners.size === 0) {
    try {
      const p = window.prompt(
        "Enter a path for the project (.mpf):",
        r.defaultPath || "project.mpf",
      );
      r.resolve(p && p.trim() ? p.trim() : null);
    } catch {
      r.resolve(r.defaultPath || null);
    }
    return;
  }
  listeners.forEach((fn) => fn(r));
}

/** Imperative Save Project dialog. Resolves with the chosen path on
 *  Save, or null on Cancel / Esc / backdrop click. The caller is
 *  responsible for performing the actual save with the returned path. */
export function saveProjectDialog(
  opts: SaveProjectDialogOpts = {},
): Promise<string | null> {
  return new Promise<string | null>((resolve) => {
    dispatch({ ...opts, resolve });
  });
}

/** Singleton host — mount ONCE in the React tree. */
export function SaveProjectHost() {
  const [queue, setQueue] = useState<SaveRequest[]>([]);
  const [path, setPath] = useState("");

  useEffect(() => {
    const onRequest: Listener = (r) => setQueue((q) => [...q, r]);
    listeners.add(onRequest);
    return () => { listeners.delete(onRequest); };
  }, []);

  const current = queue[0];

  // Seed the path field whenever a new request becomes current.
  useEffect(() => {
    if (current) setPath(current.defaultPath ?? "");
  }, [current]);

  if (!current) return null;

  const finish = (result: string | null) => {
    current.resolve(result);
    setQueue((q) => q.slice(1));
  };

  return (
    <Dialog open onClose={() => finish(null)} maxWidth="sm" fullWidth>
      <DialogTitle>{current.title ?? "Save Project"}</DialogTitle>
      <DialogContent>
        <Box>
          <Box sx={{ display: "flex", gap: 1, mt: 1, alignItems: "center" }}>
            <TextField
              autoFocus fullWidth size="small"
              label="File path"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && path.trim()) {
                  e.preventDefault();
                  finish(path.trim());
                }
              }}
            />
            <Button
              variant="outlined" size="small"
              sx={{ minWidth: 80, flexShrink: 0 }}
              onClick={async () => {
                try {
                  const { save } = await import("@tauri-apps/plugin-dialog");
                  const selected = await save({
                    defaultPath: path || "project.mpf",
                    filters: [{ name: "Project", extensions: ["mpf"] }],
                  });
                  if (selected) { setPath(selected); return; }
                } catch { /* not in Tauri — web fallback */ }
                const fname = (path || "project.mpf").split("/").pop() || "project.mpf";
                setPath(`~/Documents/${fname}`);
              }}
            >Browse</Button>
          </Box>
          <Typography variant="caption" sx={{ color: "text.secondary", ml: 1.5, mt: 0.25, display: "block", fontSize: "0.65rem" }}>
            Enter full path. In web preview, Browse pre-fills ~/Documents/.
          </Typography>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => finish(null)}>Cancel</Button>
        <Button
          variant="contained"
          disabled={!path.trim()}
          onClick={() => finish(path.trim())}
        >
          {current.confirmLabel ?? "Save"}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
