/* ──────────────────────────────────────────────────────────
   ReadingBatchStyleBar — batch label size / colour for the selected
   readings of a Curved Surface measurement (Edit Panel → Annotations).

   Presentational: the parent owns which readings are selected (ticked in
   the Individual readings list, or Shift / Cmd-clicked on the preview)
   and writes the result. Each control applies to EVERY selected reading
   in one call. Overrides are stored per reading as `font_size` (pt) /
   `color`; null means "use the measurement's style", which is what
   "Use measurement default" writes.
   ────────────────────────────────────────────────────────── */

import { useState } from "react";
import { Box, Button, TextField, Tooltip, Typography } from "@mui/material";
import { ColorPickerButton } from "../shared/ColorPickerButton";

export interface ReadingStylePatch {
  font_size?: number | null;
  color?: string | null;
}

export interface ReadingBatchStyleBarProps {
  /** Per-reading overrides, in reading order (null / absent = inherit). */
  readings: { font_size?: number | null; color?: string | null }[];
  /** Selected reading indices. */
  selected: number[];
  /** The measurement's own label size (pt) and colour. */
  defaultFontSize: number;
  defaultColor: string;
  /** Replace the selection (All / None). */
  onSelect: (indices: number[]) => void;
  /** Apply a style patch to these readings, in one write. */
  onApply: (indices: number[], patch: ReadingStylePatch) => void;
}

const SIZE_MIN = 1;
const SIZE_MAX = 200;

export function ReadingBatchStyleBar({
  readings, selected, defaultFontSize, defaultColor, onSelect, onApply,
}: ReadingBatchStyleBarProps) {
  const sel = selected.filter((i) => i >= 0 && i < readings.length);
  const n = sel.length;
  // Draft text while typing, so the field can be emptied and retyped
  // without snapping back to the applied value. A draft belongs to the
  // selection it was typed for: Shift / Cmd-clicking another reading on the
  // preview does NOT blur the field, so without this the old number would
  // stay on show for a selection that no longer shares it.
  const selKey = [...sel].sort((a, b) => a - b).join(",");
  const [draft, setDraftState] = useState<{ key: string; text: string } | null>(null);
  if (draft && draft.key !== selKey) setDraftState(null);
  const draftText = draft && draft.key === selKey ? draft.text : null;
  const setDraft = (text: string | null) => setDraftState(text === null ? null : { key: selKey, text });

  const effSize = (i: number) => {
    const v = readings[i]?.font_size;
    return v && v > 0 ? v : defaultFontSize;
  };
  const effColor = (i: number) => readings[i]?.color || defaultColor;
  const sizes = new Set(sel.map(effSize));
  const colors = new Set(sel.map((i) => effColor(i).toUpperCase()));
  const commonSize = sizes.size === 1 ? String(effSize(sel[0])) : "";
  const commonColor = colors.size === 1 ? effColor(sel[0]) : defaultColor;
  const anyOverride = sel.some((i) => readings[i]?.font_size != null || !!readings[i]?.color);

  const linkSx = { minWidth: 0, px: 0.5, py: 0, fontSize: "0.6rem", textTransform: "none" } as const;

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }} data-testid="reading-batch-style">
      <Box sx={{ display: "flex", alignItems: "center", gap: 0.25 }}>
        <Tooltip placement="top" title="Select every reading for batch size / colour">
          <Button size="small" sx={linkSx} aria-label="Select all readings"
            disabled={readings.length === 0 || n === readings.length}
            onClick={(e) => { e.stopPropagation(); onSelect(readings.map((_, i) => i)); }}>
            All
          </Button>
        </Tooltip>
        <Tooltip placement="top" title="Clear the reading selection">
          <Button size="small" sx={linkSx} aria-label="Select no readings"
            disabled={n === 0}
            onClick={(e) => { e.stopPropagation(); onSelect([]); }}>
            None
          </Button>
        </Tooltip>
        <Typography variant="caption" sx={{ fontSize: "0.6rem", color: "text.secondary", ml: 0.5 }} aria-live="polite">
          {n === 0 ? "none selected" : `${n} selected`}
        </Typography>
      </Box>

      {n === 0 ? (
        <Typography variant="caption" sx={{ color: "text.disabled", fontSize: "0.6rem", fontStyle: "italic" }}>
          Tick readings below (or Shift-click their values on the image) to set their label size and colour together.
        </Typography>
      ) : (
        <Box
          sx={{ display: "flex", alignItems: "center", gap: 0.75, p: 0.5, border: "1px dashed", borderColor: "divider", borderRadius: 1 }}
          onClick={(e) => e.stopPropagation()}
        >
          <Tooltip placement="top" title={`Label size (pt) for the ${n} selected reading${n > 1 ? "s" : ""}. Blank = they differ.`}>
            <TextField
              label="Size (pt)"
              type="number"
              size="small"
              value={draftText ?? commonSize}
              placeholder="mixed"
              onChange={(e) => {
                const raw = e.target.value;
                setDraft(raw);
                const v = Number(raw);
                if (raw.trim() !== "" && Number.isFinite(v) && v >= SIZE_MIN && v <= SIZE_MAX) {
                  onApply(sel, { font_size: v });
                }
              }}
              onBlur={() => setDraft(null)}
              slotProps={{
                inputLabel: { shrink: true },
                htmlInput: { min: SIZE_MIN, max: SIZE_MAX, step: 0.5, "aria-label": "Label size in points for the selected readings" },
              }}
              sx={{ width: 84, "& input": { fontSize: "0.72rem", py: 0.6 } }}
            />
          </Tooltip>
          <ColorPickerButton
            title={`Label colour for the ${n} selected reading${n > 1 ? "s" : ""}`}
            value={commonColor}
            size={24}
            onChange={(c) => onApply(sel, { color: c })}
          />
          <Tooltip placement="top" title="Clear the size and colour overrides on the selected readings, so they follow the measurement's own label style again.">
            <span>
              <Button
                size="small" variant="outlined"
                disabled={!anyOverride}
                onClick={() => { setDraft(null); onApply(sel, { font_size: null, color: null }); }}
                sx={{ fontSize: "0.6rem", py: 0.1, px: 0.75, textTransform: "none", whiteSpace: "nowrap" }}
              >
                Use measurement default
              </Button>
            </span>
          </Tooltip>
        </Box>
      )}
    </Box>
  );
}
