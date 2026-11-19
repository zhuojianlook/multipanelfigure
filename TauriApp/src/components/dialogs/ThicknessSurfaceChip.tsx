/* ──────────────────────────────────────────────────────────
   ThicknessSurfaceChip — Edit Panel → Annotations → Curved Surface.
   Says what a preview click does right now (tracing / editing which
   surface, how many points) and offers Save. The surface's readings stay
   hidden on the preview until it is saved (EditPanelDialog's overlay).

   It floats over the bottom of the preview, which is often exactly where
   a bottom surface runs, so the chip itself lets the mouse through
   (pointer-events: none): clicks and drags under it reach the image and
   its point handles. Only the Save button takes the mouse.
   ────────────────────────────────────────────────────────── */

import { Box, Button, Tooltip, Typography } from "@mui/material";

export interface ThicknessSurfaceChipProps {
  which: "top" | "bottom";
  kind: "trace" | "edit";
  /** Points on the surface being traced / edited. */
  nPts: number;
  /** A transient message that replaces the hint (e.g. "needs 2 points"). */
  message?: string;
  onDone: () => void;
}

export function ThicknessSurfaceChip({ which, kind, nPts, message, onDone }: ThicknessSurfaceChipProps) {
  return (
    <Box
      data-testid="thickness-surface-chip"
      role="status" aria-live="polite"
      // Inline (not sx) so it is plain to see — and to test — that the
      // chip never swallows a click meant for the image beneath it.
      style={{ pointerEvents: "none" }}
      sx={{
        position: "absolute", bottom: 8, left: "50%", transform: "translateX(-50%)", zIndex: 10,
        display: "flex", alignItems: "center", gap: 0.75, maxWidth: "calc(100% - 16px)",
        bgcolor: "background.paper", border: "1px solid", borderColor: "primary.main",
        borderRadius: 1, pl: 1, pr: 0.25, py: 0.1, opacity: 0.95,
      }}
    >
      <Typography variant="caption" sx={{ fontSize: "0.65rem", userSelect: "none" }}>
        <b>{kind === "trace" ? `Tracing ${which} surface` : `Editing ${which} surface`} · {nPts} point{nPts === 1 ? "" : "s"}</b>
        {" — "}
        {message || (kind === "trace"
          ? "click along it; Save, Enter or double-click to finish, Esc to stop"
          : "click to add, Alt / right-click a point to remove, Save or Esc to finish")}
      </Typography>
      <Tooltip title={kind === "trace" ? "Save this surface and show the readings (Enter)" : "Save this surface and show the readings (Esc)"}>
        <Button
          size="small"
          style={{ pointerEvents: "auto" }}
          // Keep the press off the preview pane (shift-drag pan) and the
          // image's click / double-click handlers.
          onMouseDown={(e) => e.stopPropagation()}
          onDoubleClick={(e) => e.stopPropagation()}
          onClick={(e) => {
            e.stopPropagation();
            // A double-click on Done is one Done, not "finish top, then
            // complain that the bottom has no points".
            if (e.detail >= 2) return;
            onDone();
          }}
          aria-label={`Save the ${which} surface`}
          sx={{ fontSize: "0.6rem", textTransform: "none", minWidth: 0, py: 0, px: 0.75 }}
        >
          Save
        </Button>
      </Tooltip>
    </Box>
  );
}
