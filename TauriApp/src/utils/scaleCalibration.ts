/* ──────────────────────────────────────────────────────────
   scaleCalibration — the maths behind Edit Panel → Scale Bar → "Measure".

   The user clicks two points on the panel preview and types the real
   distance between them; this turns that into the panel's
   micron_per_pixel, and can save it as a scale-bar entry
   (config.resolution_entries) for reuse.

   Conventions honoured here are load-bearing — they are shared with the
   Sidebar's scale-bar list, the Source picker, the backend and every saved
   .mpfs, so none of them may drift:

   • micron_per_pixel is ALWAYS µm/px internally. `unit` only chooses how it
     is DISPLAYED (value / UNIT_TO_UM[unit]).
   • A saved entry is keyed `${name}|${unit}` and its VALUE is µm/px no
     matter which unit is in the key — exactly what Sidebar.tsx writes.
     Picking an entry in the Source dropdown copies that value verbatim into
     micron_per_pixel, so a measured calibration survives save → select
     bit-for-bit (JSON round-trips doubles exactly).
   • Pixels are the scale bar's OWN pixel space. The backend draws
     bar_length_px = bar_length_microns / micron_per_pixel in the
     process_panel OUTPUT image (post-crop, post-final-resize — see
     figure_builder._add_panel_scale_bars and the panel-preview endpoints'
     `processed_width/height = processed.size`). Picked points are
     percentages of the preview <img>, which shows that same image, so
     converting them with the processed width/height measures in exactly the
     space the bar is later drawn in — PROVIDED those pixels are square. A
     non-uniform final_resize (legacy files only; there is no UI for it)
     makes them rectangular, and the chord must then be taken in the bar's
     horizontal axis: see barAxisPixelDistance.
   ────────────────────────────────────────────────────────── */

import { unitLabelFor } from "./lineMeasure";

/** A point as a percentage (0-100) of the image's width / height — the
 *  coordinate system every preview overlay in the Edit Panel dialog uses. */
export type PctPt = [number, number];

/** µm per unit. Identical to the (private) tables in lineMeasure.ts and
 *  thicknessGeometry.ts and to the Sidebar's conversionToUm — the tests pin
 *  that parity, so a change to one that isn't mirrored fails loudly. */
export const UNIT_TO_UM: Readonly<Record<string, number>> = Object.freeze({
  km: 1e9, m: 1e6, cm: 10000, mm: 1000, um: 1, nm: 0.001, pm: 1e-6,
});

/** Units offered by the scale-bar UI, largest first (same order as the
 *  Scale Bar tab's unit toggle and the Sidebar's unit select). */
export const SCALE_UNITS = ["km", "m", "cm", "mm", "um", "nm", "pm"] as const;

/** Below this a calibration is dominated by click jitter (half a pixel at
 *  each end is already ±50% of a 2-px span), so Apply / Save are disabled. */
export const MIN_MEASURE_PX = 2;

/** µm per one `unit`. Unknown units fall back to 1 (µm), matching the
 *  `UNIT_TO_UM[unit] || 1` convention used across the app. hasOwnProperty
 *  so a stray key like "constructor" can't return a prototype function. */
export function umPerUnit(unit: string): number {
  return Object.prototype.hasOwnProperty.call(UNIT_TO_UM, unit) ? UNIT_TO_UM[unit] : 1;
}

/** Clamp a percent point into the image (0-100 on both axes). */
export function clampPct(p: PctPt): PctPt {
  return [Math.max(0, Math.min(100, p[0])), Math.max(0, Math.min(100, p[1]))];
}

/** Distance in image pixels between two percent points on a `w`×`h` image —
 *  the same chord lineMeasure.lineLengthPx uses for a straight line
 *  annotation. NOTE this is a true length only when the image's pixels are
 *  square; after a NON-uniform final_resize they aren't, and a scale bar must
 *  use barAxisPixelDistance instead. */
export function pixelDistance(a: PctPt, b: PctPt, w: number, h: number): number {
  const dx = ((b[0] - a[0]) / 100) * w;
  const dy = ((b[1] - a[1]) / 100) * h;
  return Math.hypot(dx, dy);
}

/** The chord a→b expressed in the scale bar's OWN axis: horizontal output
 *  pixels. That is the quantity micron_per_pixel must be calibrated against,
 *  because the backend draws the bar horizontally in the process_panel
 *  output image.
 *
 *  `pre` is the image size after rotate + crop but BEFORE final_resize. A
 *  final_resize to a different aspect stretches x and y by different factors
 *  (sx = pre.w/outW, sy = pre.h/outH), so output pixels are no longer square:
 *  the chord is first mapped back to square source pixels, measured there,
 *  then re-expressed in horizontal output pixels. Without `pre` (no
 *  final_resize) — or with a uniform resize, where sx === sy — this reduces
 *  exactly to pixelDistance. */
export function barAxisPixelDistance(
  a: PctPt, b: PctPt, outW: number, outH: number,
  pre?: { w: number; h: number } | null,
): number {
  const dx = ((b[0] - a[0]) / 100) * outW;
  const dy = ((b[1] - a[1]) / 100) * outH;
  if (!pre || !(pre.w > 0 && pre.h > 0 && outW > 0 && outH > 0)) return Math.hypot(dx, dy);
  const sx = pre.w / outW;
  const sy = pre.h / outH;
  return Math.hypot(dx * sx, dy * sy) / sx;
}

/** Mirror of image_processing.parse_aspect_ratio: "W:H" → W/H, else null.
 *  Exactly two parts, both numeric (whitespace allowed, like python's
 *  float()); a zero height is null, as python's ZeroDivisionError is. */
export function parseAspectRatio(s: string | null | undefined): number | null {
  if (typeof s !== "string" || !s.trim()) return null;
  const parts = s.split(":");
  if (parts.length !== 2) return null;
  const [ws, hs] = parts.map((p) => p.trim());
  if (!ws || !hs) return null;
  const w = Number(ws), h = Number(hs);
  if (!Number.isFinite(w) || !Number.isFinite(h) || h === 0) return null;
  return w / h;
}

/** The image size process_panel has just BEFORE its final_resize step, i.e.
 *  what barAxisPixelDistance needs as `pre`. Mirrors image_processing.py's
 *  order: rotate (0/90/180/270 exact; other angles expand like PIL) → direct
 *  pixel crop in ROTATED space, clamped, applied only when the box exceeds
 *  1 px each way → otherwise, if crop_image, the aspect crop of
 *  crop_with_aspect (int() truncation included). Flips don't change size.
 *  null when the source size isn't known yet. */
export function preResizeDims(o: {
  origW: number; origH: number; rotationDeg: number;
  crop: readonly number[] | null | undefined; cropImage: boolean; aspect: number | null;
}): { w: number; h: number } | null {
  if (!(o.origW > 0 && o.origH > 0)) return null;
  let w = o.origW, h = o.origH;
  const rot = Number(o.rotationDeg) || 0;
  if (Math.abs(rot) > 0.1) {
    const r = ((Math.trunc(rot) % 360) + 360) % 360;
    if (r === 90 || r === 270) { w = o.origH; h = o.origW; }
    else if (r !== 0 && r !== 180) {
      const rad = (rot * Math.PI) / 180;
      const c = Math.abs(Math.cos(rad)), s = Math.abs(Math.sin(rad));
      w = Math.ceil(o.origW * c + o.origH * s);
      h = Math.ceil(o.origW * s + o.origH * c);
    }
  }
  const cr = o.crop;
  if (cr && cr.length === 4 && cr.every((v) => Number.isFinite(v))) {
    const left = Math.max(0, Math.min(Math.trunc(cr[0]), w - 1));
    const top = Math.max(0, Math.min(Math.trunc(cr[1]), h - 1));
    const right = Math.max(left + 1, Math.min(Math.trunc(cr[2]), w));
    const bottom = Math.max(top + 1, Math.min(Math.trunc(cr[3]), h));
    if (right > left + 1 && bottom > top + 1) return { w: right - left, h: bottom - top };
  }
  if (o.cropImage && o.aspect && o.aspect > 0) {
    let rw = w, rh = Math.trunc(rw / o.aspect);
    if (rh > h) { rh = h; rw = Math.trunc(rh * o.aspect); }
    return { w: rw, h: rh };
  }
  return { w, h };
}

/** µm per pixel when `distance` (expressed in `unit`) spans `px` pixels.
 *  NaN when it can't be calibrated (non-positive / non-finite input) so a
 *  caller can't accidentally store 0 or Infinity as a scale. */
export function umPerPixel(distance: number, unit: string, px: number): number {
  if (!(Number.isFinite(distance) && distance > 0)) return NaN;
  if (!(Number.isFinite(px) && px > 0)) return NaN;
  return (distance * umPerUnit(unit)) / px;
}

/** The inverse used for DISPLAY: stored µm/px → `<unit>`/px. This is what
 *  the Scale Bar tab's "<unit>/pixel" field and the Sidebar list show. */
export function perPixelInUnit(umPerPx: number, unit: string): number {
  return umPerPx / umPerUnit(unit);
}

/** e.g. "0.0813 µm/px". `sig` significant figures, trailing zeros dropped. */
export function formatPerPixel(umPerPx: number, unit: string, sig = 4): string {
  if (!Number.isFinite(umPerPx) || umPerPx <= 0) return "—";
  return `${Number(perPixelInUnit(umPerPx, unit).toPrecision(sig))} ${unitLabelFor(unit)}/px`;
}

/** Parse the "Known distance" field. Empty → NaN (not 0), so a cleared field
 *  reads as "no distance yet" rather than as a zero-length calibration.
 *  Accepts a decimal comma for locales that type one. */
export function parseDistance(text: string): number {
  const t = (text ?? "").trim().replace(",", ".");
  if (t === "") return NaN;
  return Number(t);
}

export type CalibrationIssue = "too-short" | "bad-distance" | null;

/** Why a measurement can't be applied yet (null = it can). The measured
 *  span is checked first: re-measuring is the fix for that one. */
export function calibrationIssue(px: number, distance: number): CalibrationIssue {
  if (!(Number.isFinite(px) && px >= MIN_MEASURE_PX)) return "too-short";
  if (!(Number.isFinite(distance) && distance > 0)) return "bad-distance";
  return null;
}

/* ── Saved scale-bar entries (config.resolution_entries) ──────────── */

/** Entry key — MUST equal the Sidebar's `${name}|${unit}` so an entry made
 *  here lists, edits and deletes there exactly like a hand-typed one. */
export function scaleEntryKey(name: string, unit: string): string {
  return `${name}|${unit}`;
}

/** Inverse of scaleEntryKey, mirroring how every picker parses keys
 *  (`key.split("|")`, unit defaulting to µm for legacy unit-less keys). */
export function parseScaleEntryKey(key: string): { name: string; unit: string } {
  const parts = key.split("|");
  return { name: parts[0], unit: parts[1] || "um" };
}

/** Validation message for a new entry name, or null when it's usable. A "|"
 *  would be read back as the key's unit separator and corrupt the unit. */
export function scaleEntryNameIssue(name: string): string | null {
  const n = (name ?? "").trim();
  if (!n) return "Enter a name";
  if (n.includes("|")) return "Name can't contain \"|\"";
  return null;
}

/** A NEW entries object with `key` set to `valueUm` (µm/px) — the same
 *  copy-then-assign the Sidebar does, never mutating the store's object. */
export function withScaleEntry(
  entries: Record<string, number> | undefined, key: string, valueUm: number,
): Record<string, number> {
  return { ...(entries ?? {}), [key]: valueUm };
}

/** What choosing `key` in the Scale Bar tab's Source picker writes into the
 *  scale bar: the stored µm/px verbatim plus the key's unit. null when the
 *  key isn't a selectable entry (missing / zero), which the picker treats as
 *  "Custom". */
export function scaleFromEntry(
  entries: Record<string, number> | undefined, key: string,
): { micron_per_pixel: number; unit: string } | null {
  if (!key || !entries || !Object.prototype.hasOwnProperty.call(entries, key)) return null;
  const v = entries[key];
  if (!v) return null;
  return { micron_per_pixel: v, unit: parseScaleEntryKey(key).unit };
}

/** The scale bar's auto label for a bar of `barLenUm` µm shown in `unit` —
 *  the same string the Scale Bar tab compares against to decide whether a
 *  label is still "auto" (and so may be cleared when the unit changes). */
export function scaleBarAutoLabel(barLenUm: number, unit: string): string {
  return `${Number((barLenUm / umPerUnit(unit)).toPrecision(6))} ${unitLabelFor(unit)}`;
}
