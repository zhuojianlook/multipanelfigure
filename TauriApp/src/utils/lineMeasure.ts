/* ──────────────────────────────────────────────────────────
   lineMeasure — a line annotation's length and its auto label.

   Single source for the value shown in the editor field and the value
   drawn on the live preview overlay, so the two can't drift apart.

   NOTE a known divergence with the backend: python's
   compute_line_length_pixels sums the chords between CONTROL POINTS and
   ignores line_type, whereas a "curved" line is rendered as a
   Catmull-Rom spline and is therefore genuinely longer than its control
   polygon. This module measures what is actually drawn (sampling the
   spline). For straight / multijointed lines — the common case — the two
   agree exactly.
   ────────────────────────────────────────────────────────── */

export type LinePt = [number, number];

const UNIT_TO_UM: Record<string, number> = {
  km: 1e9, m: 1e6, cm: 10000, mm: 1000, um: 1, nm: 0.001, pm: 1e-6,
};
const UNIT_LABELS: Record<string, string> = {
  km: "km", m: "m", cm: "cm", mm: "mm", um: "µm", nm: "nm", pm: "pm",
};

export function unitLabelFor(unit: string): string {
  return UNIT_LABELS[unit] || unit;
}

/** Exact port of the backend's `_format_measurement` (image_processing.py) —
 *  the adaptive precision it DRAWS with. The auto-populated field must match
 *  the figure character-for-character, otherwise the label would visibly
 *  change format the moment the user styles it (which pins the string). */
export function formatMeasurement(value: number, unitLabel: string): string {
  if (value < 0.01) {
    // Python's %.3e → "1.234e-03"; JS toExponential gives "1.234e-3".
    const [m, e] = value.toExponential(3).split("e");
    const sign = e[0] === "-" ? "-" : "+";
    const digits = e.replace(/[+-]/, "").padStart(2, "0");
    return `${m}e${sign}${digits} ${unitLabel}`;
  }
  if (value < 1) return `${sigFigs(value, 4)} ${unitLabel}`;   // %.4g
  if (value < 1000) return `${value.toFixed(2)} ${unitLabel}`; // %.2f
  return `${sigFigs(value, 6)} ${unitLabel}`;                  // %.6g
}

/** Python's %g: significant figures with trailing zeros stripped. */
function sigFigs(v: number, sig: number): string {
  if (v === 0) return "0";
  const s = v.toPrecision(sig);
  return s.includes(".") ? s.replace(/\.?0+$/, "") : s;
}

/** Total drawn length in image pixels. `lineType` is
 *  "straight" | "multijointed" | "curved". */
export function lineLengthPx(
  points: LinePt[], imgW: number, imgH: number, lineType: string,
): number {
  if (!points || points.length < 2) return 0;
  const px = (p: LinePt): LinePt => [(p[0] / 100) * imgW, (p[1] / 100) * imgH];

  if (lineType === "curved" && points.length >= 3) {
    // Sample the same Catmull-Rom the renderer draws.
    const pts = points.map(px);
    const SAMPLES = 20;
    let total = 0;
    let prev: LinePt | null = null;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[Math.max(0, i - 1)];
      const p1 = pts[i];
      const p2 = pts[i + 1];
      const p3 = pts[Math.min(pts.length - 1, i + 2)];
      for (let s = 0; s <= SAMPLES; s++) {
        const t = s / SAMPLES;
        const t2 = t * t, t3 = t2 * t;
        const x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t
          + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
          + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3);
        const y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
          + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
          + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3);
        if (prev) total += Math.hypot(x - prev[0], y - prev[1]);
        prev = [x, y];
      }
    }
    return total;
  }

  // Straight / multijointed: chord sum over the control points.
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    const a = px(points[i - 1]), b = px(points[i]);
    total += Math.hypot(b[0] - a[0], b[1] - a[1]);
  }
  return total;
}

/** Length in the given unit (no label). */
export function lineLengthInUnit(
  points: LinePt[], imgW: number, imgH: number, lineType: string,
  mpp: number, unit: string,
): number {
  const px = lineLengthPx(points, imgW, imgH, lineType);
  return (px * mpp) / (UNIT_TO_UM[unit] || 1);
}

/** The auto label, e.g. "12.3 µm". "" when the line is incomplete. */
export function lineAutoLabel(
  points: LinePt[], imgW: number, imgH: number, lineType: string,
  mpp: number, unit: string,
): string {
  if (!points || points.length < 2) return "";
  const v = lineLengthInUnit(points, imgW, imgH, lineType, mpp, unit);
  return formatMeasurement(v, unitLabelFor(unit));
}

/** Area's auto label, e.g. "12.3 µm²". Mirrors the backend's
 *  compute_area_measurement (shoelace / rect / ellipse, mpp squared). */
export function areaAutoLabel(
  points: LinePt[], shape: string, imgW: number, imgH: number,
  mpp: number, unit: string,
): string {
  if (!points || points.length < 2) return "";
  const toPx = (p: LinePt): LinePt => [(p[0] / 100) * imgW, (p[1] / 100) * imgH];
  let areaPx = 0;
  if (shape === "Rectangle" && points.length >= 2) {
    const [w, h] = points[1];
    areaPx = ((w / 100) * imgW) * ((h / 100) * imgH);
  } else if (shape === "Ellipse" && points.length >= 2) {
    const [w, h] = points[1];
    areaPx = Math.PI * (((w / 100) * imgW) / 2) * (((h / 100) * imgH) / 2);
  } else if (points.length >= 3) {
    const c = points.map(toPx);
    let a = 0;
    for (let i = 0; i < c.length; i++) {
      const j = (i + 1) % c.length;
      a += c[i][0] * c[j][1] - c[j][0] * c[i][1];
    }
    areaPx = Math.abs(a) / 2;
  }
  const umPerUnit = UNIT_TO_UM[unit] || 1;
  const v = (areaPx * mpp * mpp) / (umPerUnit * umPerUnit);
  return formatMeasurement(v, `${unitLabelFor(unit)}\u00B2`);
}
