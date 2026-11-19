/* ──────────────────────────────────────────────────────────
   LockedWorkflowBanner — shown over the Analysis canvas while the
   workflow on screen is locked: it reads figures of several .mpf
   documents and one of them is closed, so it is greyed out and
   read-only (its results couldn't be refreshed, and the closed file's
   copy couldn't follow an edit).

   Three ways out (store/analysisWorkspaceStore):
     • Open missing — look for the files where they were saved / next
       to the open ones and open them;
     • Locate…      — point at a file that moved (native dialog);
     • Remove their sources — drop the closed documents' images and
       tables, leaving a workflow of the open documents only (the way
       out when a file is gone for good, or was never saved).
   ────────────────────────────────────────────────────────── */

import { useState } from "react";
import { Box, Button, Menu, MenuItem, Typography } from "@mui/material";
import LockIcon from "@mui/icons-material/Lock";
import type { MissingDoc } from "../../store/analysisWorkspaceStore";

interface Props {
  /** The closed documents the workflow needs (at least one). */
  missing: readonly MissingDoc[];
  /** An open / locate is in flight. */
  busy?: boolean;
  onOpenMissing: () => void;
  onLocate: (uid: string) => void;
  onRemoveSources: () => void;
}

export function LockedWorkflowBanner({ missing, busy, onOpenMissing, onLocate, onRemoveSources }: Props) {
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);
  const findable = missing.filter((m) => m.locatable);
  const many = missing.length > 1;
  return (
    <Box
      role="status"
      data-testid="locked-workflow-banner"
      sx={{
        display: "flex", alignItems: "center", flexWrap: "wrap", gap: 0.75,
        px: 1.25, py: 0.75, borderRadius: 1,
        bgcolor: "background.paper", border: "1px solid", borderColor: "warning.main",
        boxShadow: 3,
      }}
    >
      <LockIcon sx={{ fontSize: 16, color: "warning.main" }} />
      <Typography variant="caption" sx={{ fontSize: "0.7rem", lineHeight: 1.35, flex: "1 1 220px" }}>
        This workflow also uses{" "}
        {missing.map((m, i) => (
          <span key={m.uid}>
            {i > 0 ? ", " : ""}<strong>{m.label}</strong>
          </span>
        ))}
        . Open {many ? "them" : "it"} to edit and re-run.
      </Typography>
      <Button
        size="small" variant="contained" color="warning"
        disabled={busy || findable.length === 0}
        onClick={onOpenMissing}
        sx={{ fontSize: "0.65rem", textTransform: "none", py: 0.25 }}
      >
        Open missing
      </Button>
      <Button
        size="small" variant="outlined"
        disabled={busy}
        onClick={(e) => (many ? setAnchor(e.currentTarget) : onLocate(missing[0].uid))}
        sx={{ fontSize: "0.65rem", textTransform: "none", py: 0.25 }}
      >
        Locate…
      </Button>
      <Menu anchorEl={anchor} open={!!anchor} onClose={() => setAnchor(null)} MenuListProps={{ dense: true }}>
        {missing.map((m) => (
          <MenuItem
            key={m.uid}
            onClick={() => { setAnchor(null); onLocate(m.uid); }}
            sx={{ fontSize: "0.72rem" }}
          >
            Locate {m.label}…
          </MenuItem>
        ))}
      </Menu>
      <Button
        size="small" variant="text" color="inherit"
        disabled={busy}
        onClick={onRemoveSources}
        sx={{ fontSize: "0.65rem", textTransform: "none", py: 0.25 }}
      >
        Remove their sources
      </Button>
    </Box>
  );
}
