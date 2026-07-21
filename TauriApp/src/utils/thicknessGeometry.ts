/* ──────────────────────────────────────────────────────────
   thicknessGeometry — geometry for the Curved Surface Measurement
   annotation (thickness readings between two curved surfaces).

   Each surface is a circular arc through 3 user-clicked points. An odd
   number of readings is placed along the TOP arc: one exactly on the
   user's centre node, the rest spread symmetrically either side, spaced
   `spacing` apart as a fraction of the top arc's length.

   Each reading runs from its anchor on the top arc to the NEAREST point
   on the bottom arc — the minimal distance, which meets the bottom
   surface perpendicularly (and meets the top perpendicularly too when
   the surfaces are roughly parallel, the usual case for a thickness
   measurement). Anchoring on the top arc is what guarantees a reading
   sits exactly on the centre node; a true common normal (perpendicular
   to both by construction) could not honour a user-specified anchor.

   IMPORTANT: all geometry runs in PIXEL space (using the panel's actual
   image width/height), because a circle in the anisotropic %-coordinate
   space is an ellipse in pixels — perpendiculars would be wrong. Inputs
   and outputs are in % (0-100); we convert at the boundary.

   This is the single source of truth for the geometry. The Python
   backend does NOT recompute arcs — it renders the resolved reading
   endpoints + sampled arc polylines produced here.
   ────────────────────────────────────────────────────────── */

export type Pt = [number, number];
export interface ThicknessReadingData {
  top: Pt;      // (x%, y%)
  bottom: Pt;   // (x%, y%)
  hidden: boolean;
  text: string;
  edited: boolean;
  /** Absolute label position (x%, y%); -1 = auto (offset off the line). */
  measure_position_x?: number;
  measure_position_y?: number;
  /** Per-reading rich-text runs from the hover toolbar. */
  styled_segments?: unknown[];
}

const EPS = 1e-9;

const sub = (a: Pt, b: Pt): Pt => [a[0] - b[0], a[1] - b[1]];
const dot = (a: Pt, b: Pt) => a[0] * b[0] + a[1] * b[1];
const norm = (a: Pt) => Math.hypot(a[0], a[1]);

/** Readings must be ODD so one always lands on the centre node. */
export function normalizeCount(n: number): number {
  const v = Math.max(1, Math.round(n || 1));
  return v % 2 === 1 ? v : v + 1;
}

/** Largest spacing (fraction of arc length) that still fits `count` readings
 *  inside the arc, given where the centre node sits.
 *
 *  Readings spread SYMMETRICALLY either side of `center`, so the group spans
 *  (count-1)*spacing and the binding constraint is the SHORTER side:
 *      center - (n-1)/2*s >= 0   and   center + (n-1)/2*s <= 1
 *  ⇒  s <= 2*min(center, 1-center) / (n-1)
 *
 *  So moving the centre off-middle genuinely shrinks the maximum — at
 *  center=0.5 this reduces to the familiar 1/(n-1). A small floor keeps the
 *  slider usable when the node is dragged hard against an end. */
export function maxSpacingFor(count: number, center = 0.5): number {
  const n = normalizeCount(count);
  if (n <= 1) return 1;
  const c = Math.min(1, Math.max(0, center));
  const half = Math.min(c, 1 - c);
  return Math.max(0.005, (2 * half) / (n - 1));
}

interface Curve {
  length: number;
  pointAt(t: number): Pt;
  normalAt(t: number): Pt;
  sample(n: number): Pt[];
}

function circleFrom3(p1: Pt, p2: Pt, p3: Pt): { c: Pt; r: number } | null {
  const [ax, ay] = p1, [bx, by] = p2, [cx, cy] = p3;
  const d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  if (Math.abs(d) < 1e-6) return null;
  const a2 = ax * ax + ay * ay, b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
  const ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
  const uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
  const c: Pt = [ux, uy];
  return { c, r: norm(sub(p1, c)) };
}

const TWO_PI = Math.PI * 2;
const wrap = (a: number) => ((a % TWO_PI) + TWO_PI) % TWO_PI;

function buildCurve(p1: Pt, p2: Pt, p3: Pt): Curve {
  const circle = circleFrom3(p1, p2, p3);

  if (!circle) {
    // Straight-segment fallback (collinear points).
    const len = norm(sub(p3, p1));
    const d: Pt = len < EPS ? [1, 0] : [(p3[0] - p1[0]) / len, (p3[1] - p1[1]) / len];
    const nrm: Pt = [-d[1], d[0]];
    const at = (t: number): Pt => [p1[0] + (p3[0] - p1[0]) * t, p1[1] + (p3[1] - p1[1]) * t];
    return {
      length: len,
      pointAt: at,
      normalAt: () => nrm,
      sample: (n) => Array.from({ length: n }, (_, i) => at(n <= 1 ? 0 : i / (n - 1))),
    };
  }

  const { c, r } = circle;
  const a1 = Math.atan2(p1[1] - c[1], p1[0] - c[0]);
  const a2 = Math.atan2(p2[1] - c[1], p2[0] - c[0]);
  const a3 = Math.atan2(p3[1] - c[1], p3[0] - c[0]);
  const ccwSweep = wrap(a3 - a1);
  const a2rel = wrap(a2 - a1);
  let dir = 1, span = ccwSweep;
  if (a2rel > ccwSweep) { dir = -1; span = TWO_PI - ccwSweep; }
  const angleAt = (t: number) => a1 + dir * span * t;
  return {
    length: r * span,
    pointAt: (t) => {
      const ang = angleAt(t);
      return [c[0] + r * Math.cos(ang), c[1] + r * Math.sin(ang)];
    },
    normalAt: (t) => {
      const ang = angleAt(t);
      return [Math.cos(ang), Math.sin(ang)];
    },
    sample: (n) => Array.from({ length: n }, (_, i) => {
      const ang = angleAt(n <= 1 ? 0 : i / (n - 1));
      return [c[0] + r * Math.cos(ang), c[1] + r * Math.sin(ang)] as Pt;
    }),
  };
}

/** Nearest point on a sampled polyline to `q`, plus that distance. */
function nearestOnPolyline(pts: Pt[], q: Pt): { pt: Pt; dist: number } {
  let best: Pt = pts[0] ?? q, bestD = Infinity;
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1];
    const ab = sub(b, a);
    const len2 = dot(ab, ab);
    let t = len2 < EPS ? 0 : dot(sub(q, a), ab) / len2;
    t = Math.max(0, Math.min(1, t));
    const proj: Pt = [a[0] + ab[0] * t, a[1] + ab[1] * t];
    const d = norm(sub(q, proj));
    if (d < bestD) { bestD = d; best = proj; }
  }
  return { pt: best, dist: bestD };
}

export interface ComputeResult {
  readings: ThicknessReadingData[];
  topSamples: Pt[];
  bottomSamples: Pt[];
}

const ARC_SAMPLES = 96;   // denser sampling → tighter minimal-distance feet

/**
 * Compute thickness readings between two 3-point arcs.
 *
 * @param center 0..1 — the user's centre node along the top arc. A reading
 *               ALWAYS lands exactly here (counts are forced odd).
 * @param spacing step between adjacent readings, fraction of top arc length.
 * @param existing prior readings — `edited` ones are kept verbatim; hidden /
 *                 text / label position are carried over by index.
 */
export function computeThicknessReadings(
  topPct: Pt[], botPct: Pt[],
  numReadings: number, center: number, spacing: number,
  iw: number, ih: number,
  existing: ThicknessReadingData[] = [],
): ComputeResult {
  const toPx = (p: Pt): Pt => [(p[0] / 100) * iw, (p[1] / 100) * ih];
  const toPct = (p: Pt): Pt => [(p[0] / iw) * 100, (p[1] / ih) * 100];

  const n = normalizeCount(numReadings);
  if (topPct.length < 3 || botPct.length < 3 || iw <= 0 || ih <= 0) {
    return { readings: existing.slice(0, n), topSamples: [], bottomSamples: [] };
  }

  const top = buildCurve(toPx(topPct[0]), toPx(topPct[1]), toPx(topPct[2]));
  const bottom = buildCurve(toPx(botPct[0]), toPx(botPct[1]), toPx(botPct[2]));
  const topSamplesPx = top.sample(ARC_SAMPLES);
  const bottomSamplesPx = bottom.sample(ARC_SAMPLES);

  const L = top.length;
  // Cap against the centre's actual room, not just the count.
  const step = Math.max(0, Math.min(spacing, maxSpacingFor(n, center))) * L;
  const centerS = Math.min(1, Math.max(0, center)) * L;
  const mid = (n - 1) / 2;   // integer, since n is odd → a reading sits on centre

  const readings: ThicknessReadingData[] = [];
  for (let k = 0; k < n; k++) {
    const prev = existing[k];
    if (prev && prev.edited) {
      readings.push({ ...prev });
      continue;
    }
    let s = centerS + (k - mid) * step;
    s = Math.min(L, Math.max(0, s));
    const t = L < EPS ? 0 : s / L;
    const T = top.pointAt(t);
    // Minimal distance from the anchor to the bottom surface. The foot of a
    // minimal-distance segment is perpendicular to the bottom curve there.
    const B = nearestOnPolyline(bottomSamplesPx, T).pt;
    readings.push({
      top: toPct(T),
      bottom: toPct(B),
      hidden: prev ? prev.hidden : false,
      text: prev ? prev.text : "",
      edited: false,
      measure_position_x: prev?.measure_position_x ?? -1,
      measure_position_y: prev?.measure_position_y ?? -1,
      styled_segments: prev?.styled_segments ?? [],
    });
  }

  return {
    readings,
    topSamples: topSamplesPx.map(toPct),
    bottomSamples: bottomSamplesPx.map(toPct),
  };
}

/**
 * Snap a freely-dragged point onto a 3-point arc when it lands close to it.
 * Lets a manual reading endpoint sit exactly on the surface without forcing
 * it — beyond `tolPct` of the image's smaller side, the raw point is kept.
 *
 * @param pPct dragged point (x%, y%)
 * @param curvePct the arc's 3 defining points
 * @returns {point, snapped}
 */
export function snapToCurve(
  // tolPct was 1.5% of the smaller side — so tight that dragging a reading
  // endpoint essentially never snapped back onto the surface, and the snap
  // read as non-existent. 4% is a reachable grab distance while still leaving
  // deliberate free placement possible further out.
  pPct: Pt, curvePct: Pt[], iw: number, ih: number, tolPct = 4,
): { point: Pt; snapped: boolean } {
  if (curvePct.length < 3 || iw <= 0 || ih <= 0) return { point: pPct, snapped: false };
  const toPx = (p: Pt): Pt => [(p[0] / 100) * iw, (p[1] / 100) * ih];
  const curve = buildCurve(toPx(curvePct[0]), toPx(curvePct[1]), toPx(curvePct[2]));
  const { pt, dist } = nearestOnPolyline(curve.sample(ARC_SAMPLES), toPx(pPct));
  const tolPx = (tolPct / 100) * Math.min(iw, ih);
  if (dist > tolPx) return { point: pPct, snapped: false };
  return { point: [(pt[0] / iw) * 100, (pt[1] / ih) * 100], snapped: true };
}

/** Parameter (0..1) along the top arc nearest to a dragged point — used by
 *  the centre node, which slides along the surface rather than free-floating. */
export function paramOnCurve(pPct: Pt, curvePct: Pt[], iw: number, ih: number): number {
  if (curvePct.length < 3 || iw <= 0 || ih <= 0) return 0.5;
  const toPx = (p: Pt): Pt => [(p[0] / 100) * iw, (p[1] / 100) * ih];
  const curve = buildCurve(toPx(curvePct[0]), toPx(curvePct[1]), toPx(curvePct[2]));
  const q = toPx(pPct);
  const N = ARC_SAMPLES;
  let bestT = 0.5, bestD = Infinity;
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1);
    const d = norm(sub(curve.pointAt(t), q));
    if (d < bestD) { bestD = d; bestT = t; }
  }
  return bestT;
}

/** Point on the top arc at parameter t — for drawing the centre node. */
export function pointOnCurve(curvePct: Pt[], t: number, iw: number, ih: number): Pt | null {
  if (curvePct.length < 3 || iw <= 0 || ih <= 0) return null;
  const toPx = (p: Pt): Pt => [(p[0] / 100) * iw, (p[1] / 100) * ih];
  const curve = buildCurve(toPx(curvePct[0]), toPx(curvePct[1]), toPx(curvePct[2]));
  const p = curve.pointAt(Math.min(1, Math.max(0, t)));
  return [(p[0] / iw) * 100, (p[1] / ih) * 100];
}

/** Physical length of a reading (top→bottom) in the given unit. Mirrors
 *  the backend's compute_line_measurement_value for live display. */
const UNIT_TO_UM: Record<string, number> = { km: 1e9, m: 1e6, cm: 10000, mm: 1000, um: 1, nm: 0.001, pm: 1e-6 };
export function readingValue(r: ThicknessReadingData, iw: number, ih: number, mpp: number, unit: string): number {
  const dx = ((r.bottom[0] - r.top[0]) / 100) * iw;
  const dy = ((r.bottom[1] - r.top[1]) / 100) * ih;
  const px = Math.hypot(dx, dy);
  return (px * mpp) / (UNIT_TO_UM[unit] || 1);
}
