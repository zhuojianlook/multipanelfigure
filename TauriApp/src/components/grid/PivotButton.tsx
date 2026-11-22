/* ──────────────────────────────────────────────────────────
   PivotButton — the panel planner's corner button where the row
   and column headers meet. It swaps the figure's rows and columns
   (4×3 → 3×4; utils/pivotGrid). Pressing it again swaps them back —
   there is no undo stack, so the pivot is its own undo.
   Styled like the planner's other corner buttons ("+" header levels).
   ────────────────────────────────────────────────────────── */

import { Tooltip } from "@mui/material";
import PivotTableChartIcon from "@mui/icons-material/PivotTableChart";

export interface PivotButtonProps {
  rows: number;
  cols: number;
  disabled?: boolean;
  onPivot: () => void;
}

export function PivotButton({ rows, cols, disabled, onPivot }: PivotButtonProps) {
  const label = `Pivot rows and columns (${rows}×${cols} → ${cols}×${rows})`;
  return (
    <Tooltip title={label} placement="bottom" arrow>
      {/* A disabled button fires no events, so the tooltip needs a wrapper. */}
      <span style={{ display: "inline-flex" }}>
        <button
          type="button"
          className="flex items-center justify-center gap-[3px] rounded hover:opacity-80 transition-opacity"
          style={{
            backgroundColor: "var(--c-surface2)",
            color: "var(--c-text-dim)",
            border: "1px dashed var(--c-border)",
            height: 22,
            padding: "0 5px",
            fontSize: 10,
            lineHeight: 1,
            whiteSpace: "nowrap",
            opacity: disabled ? 0.5 : undefined,
            cursor: disabled ? "default" : "pointer",
          }}
          onClick={() => { if (!disabled) onPivot(); }}
          disabled={disabled}
          aria-label={label}
          data-pivot-button=""
        >
          <PivotTableChartIcon sx={{ fontSize: 12 }} />
          Pivot
        </button>
      </span>
    </Tooltip>
  );
}
