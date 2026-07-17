/* ──────────────────────────────────────────────────────────
   scaleBarGeometry — where the scale bar actually sits.

   This is a direct port of the backend's `_add_panel_scale_bars`
   placement maths (python-sidecar/figure_builder.py). The dialog's
   preview overlay and its drag hitbox BOTH derive from this, so the
   position you drag to, the position you see mid-drag, and the position
   the figure renders can't drift apart.

   Conventions (identical to the backend):
     • bx = bar LEFT edge, by = bar TOP edge, in IMAGE-PIXEL space.
     • Preset  → anchored off `edge_distance` from the named corner.
     • Custom  → `position_x` is the bar's CENTRE-x, `position_y` its TOP.
     • bx / by NEVER depend on which side the label is on — only the text
       moves when "auto" flips. (A previous overlay positioned a flex
       container holding bar+label, so the bar shifted by the label's
       height whenever the side flipped, and landed somewhere different
       from where it was dropped.)
   ────────────────────────────────────────────────────────── */

export interface ScaleBarLike {
  bar_length_microns: number;
  micron_per_pixel: number;
  bar_height: number;
  edge_distance?: number;
  position_preset?: string;
  position_x?: number;
  position_y?: number;
  label_position?: string;   // "auto" | "above" | "below"
}

export interface ScaleBarRect {
  iw: number;
  ih: number;
  bx: number;       // bar left edge (image px)
  by: number;       // bar top edge  (image px)
  barLen: number;   // bar length    (image px)
  barH: number;     // bar height    (image px)
  isBottom: boolean; // true → label sits ABOVE the bar
}

export function computeScaleBarRect(sb: ScaleBarLike, iw: number, ih: number): ScaleBarRect {
  const barLen = sb.bar_length_microns / Math.max(sb.micron_per_pixel || 1, 1e-9);
  const barH = sb.bar_height;
  const edge = (sb.edge_distance ?? 5) / 100;
  const preset = sb.position_preset;

  let bx: number;
  let by: number;
  if (preset && preset !== "Custom") {
    // Backend: bx = iw*(1-edge) - bar_length  if "Right" in preset else iw*edge
    //          by = ih*(1-edge) - bar_height - 5 if "Bottom" in preset else ih*edge + 5
    bx = preset.includes("Right") ? iw * (1 - edge) - barLen : iw * edge;
    by = preset.includes("Bottom") ? ih * (1 - edge) - barH - 5 : ih * edge + 5;
  } else {
    bx = ((sb.position_x ?? 90) / 100) * iw - barLen / 2;
    by = ((sb.position_y ?? 90) / 100) * ih;
  }

  // Clamp to the image content area (backend does the same).
  bx = Math.max(0, Math.min(bx, iw - barLen - 1));
  by = Math.max(0, Math.min(by, ih - barH - 1));

  // Label side. Backend: is_bottom = by > ih * 0.5 — keyed off the BAR's top
  // in pixels, so use the same threshold or the preview disagrees with the
  // render for bars near the midline.
  const mode = (sb.label_position ?? "auto") as "auto" | "above" | "below";
  const isBottom = mode === "above" ? true : mode === "below" ? false : by > ih * 0.5;

  return { iw, ih, bx, by, barLen, barH, isBottom };
}
