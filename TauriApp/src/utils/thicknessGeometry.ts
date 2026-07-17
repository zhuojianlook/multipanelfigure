/* ──────────────────────────────────────────────────────────
   thicknessGeometry — geometry for the Thickness Measurement
   annotation (perpendicular readings between two curved surfaces).

   Each surface is a circular arc through 3 user-clicked points. N
   readings are placed along the TOP arc (centred at `center`, spaced
   `spacing` apart by arc length) and each runs perpendicular (radial)
   to the top arc down to the bottom arc.

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
}

const EPS = 1e-9;

const sub = (a: Pt, b: Pt): Pt => [a[0] - b[0], a[1] - b[1]];
const dot = (a: Pt, b: Pt) => a[0] * b[0] + a[1] * b[1];
const norm = (a: Pt) => Math.hypot(a[0], a[1]);
const unit = (a: Pt): Pt => {
  const n = norm(a);
  return n < EPS ? [0, 0] : [a[0] / n, a[1] / n];
};

/** A parametric curve through 3 points, t ∈ [0,1] from p1 → p3.
 *  Circular arc when the points aren't collinear, else a straight
 *  segment. All coordinates are pixels. */
interface Curve {
  length: number;
  centroid: Pt;
  pointAt(t: number): Pt;
  /** Outward unit normal at parameter t (for an arc: radial from the
   *  circle centre; for a line: a fixed perpendicular). */
  normalAt(t: number): Pt;
  sample(n: number): Pt[];
  /** Intersections of the INFINITE line through `o` with direction `d`
   *  (need not be unit) and this curve, clamped to the curve's span. */
  intersectLine(o: Pt, d: Pt): Pt[];
}

/** Circumcircle of 3 points, or null if (near-)collinear. */
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
  const centroid: Pt = [(p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3];

  if (!circle) {
    // Straight-segment fallback (collinear points).
    const d = unit(sub(p3, p1));
    const nrm: Pt = [-d[1], d[0]];
    const length = norm(sub(p3, p1));
    return {
      length,
      centroid,
      pointAt: (t) => [p1[0] + (p3[0] - p1[0]) * t, p1[1] + (p3[1] - p1[1]) * t],
      normalAt: () => nrm,
      sample: (n) => Array.from({ length: n }, (_, i) => {
        const t = n <= 1 ? 0 : i / (n - 1);
        return [p1[0] + (p3[0] - p1[0]) * t, p1[1] + (p3[1] - p1[1]) * t] as Pt;
      }),
      intersectLine: (o, dd) => {
        // line o+u*dd  vs  segment-line p1 + v*(p3-p1)
        const e: Pt = [p3[0] - p1[0], p3[1] - p1[1]];
        const denom = dd[0] * (-e[1]) - dd[1] * (-e[0]);
        if (Math.abs(denom) < EPS) return [];
        const diff = sub(p1, o);
        const u = (diff[0] * (-e[1]) - diff[1] * (-e[0])) / denom;
        return [[o[0] + dd[0] * u, o[1] + dd[1] * u]];
      },
    };
  }

  const { c, r } = circle;
  const a1 = Math.atan2(p1[1] - c[1], p1[0] - c[0]);
  const a2 = Math.atan2(p2[1] - c[1], p2[0] - c[0]);
  const a3 = Math.atan2(p3[1] - c[1], p3[0] - c[0]);
  // Sweep from a1 → a3 in the direction that passes through a2.
  const ccwSweep = wrap(a3 - a1);          // [0, 2π)
  const a2rel = wrap(a2 - a1);
  let dir = 1, span = ccwSweep;
  if (a2rel > ccwSweep) { dir = -1; span = TWO_PI - ccwSweep; }
  const length = r * span;
  const angleAt = (t: number) => a1 + dir * span * t;

  const inSpan = (ang: number) => {
    const rel = wrap((ang - a1) * dir);      // [0, 2π); rel along the chosen dir
    return rel <= span + 1e-6;
  };

  return {
    length,
    centroid: c,   // circle centre is a better "toward" reference than the 3-pt mean
    pointAt: (t) => {
      const ang = angleAt(t);
      return [c[0] + r * Math.cos(ang), c[1] + r * Math.sin(ang)];
    },
    normalAt: (t) => {
      const ang = angleAt(t);
      return [Math.cos(ang), Math.sin(ang)];   // outward radial (unit)
    },
    sample: (n) => Array.from({ length: n }, (_, i) => {
      const t = n <= 1 ? 0 : i / (n - 1);
      const ang = angleAt(t);
      return [c[0] + r * Math.cos(ang), c[1] + r * Math.sin(ang)] as Pt;
    }),
    intersectLine: (o, dd) => {
      // (o + u*dd) on circle |·−c| = r
      const f = sub(o, c);
      const A = dot(dd, dd);
      const B = 2 * dot(f, dd);
      const C = dot(f, f) - r * r;
      const disc = B * B - 4 * A * C;
      if (disc < 0 || A < EPS) return [];
      const sq = Math.sqrt(disc);
      const us = [(-B - sq) / (2 * A), (-B + sq) / (2 * A)];
      const out: Pt[] = [];
      for (const u of us) {
        const pt: Pt = [o[0] + dd[0] * u, o[1] + dd[1] * u];
        const ang = Math.atan2(pt[1] - c[1], pt[0] - c[0]);
        if (inSpan(ang)) out.push(pt);
      }
      // If neither intersection lands on the drawn arc span, still return
      // them (better a perpendicular to the full circle than nothing).
      if (out.length === 0) {
        for (const u of us) out.push([o[0] + dd[0] * u, o[1] + dd[1] * u]);
      }
      return out;
    },
  };
}

/** Nearest point on a sampled polyline to `q` (pixel space). */
function nearestOnPolyline(pts: Pt[], q: Pt): Pt {
  let best: Pt = pts[0], bestD = Infinity;
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
  return best;
}

export interface ComputeResult {
  readings: ThicknessReadingData[];
  topSamples: Pt[];      // (x%, y%)
  bottomSamples: Pt[];
}

const ARC_SAMPLES = 48;

/**
 * Compute thickness readings between two 3-point arcs.
 *
 * @param topPct 3 points (x%,y%) for the top surface
 * @param botPct 3 points (x%,y%) for the bottom surface
 * @param numReadings how many perpendicular readings
 * @param center 0..1 — where the reading group centres on the top arc
 * @param spacing step between adjacent readings as a fraction of the top arc length
 * @param iw,ih panel image pixel dimensions (for correct aspect)
 * @param existing prior readings — `edited` ones are kept; hidden/text carried over by index
 */
export function computeThicknessReadings(
  topPct: Pt[], botPct: Pt[],
  numReadings: number, center: number, spacing: number,
  iw: number, ih: number,
  existing: ThicknessReadingData[] = [],
): ComputeResult {
  const toPx = (p: Pt): Pt => [(p[0] / 100) * iw, (p[1] / 100) * ih];
  const toPct = (p: Pt): Pt => [(p[0] / iw) * 100, (p[1] / ih) * 100];

  if (topPct.length < 3 || botPct.length < 3 || iw <= 0 || ih <= 0) {
    return { readings: existing.slice(0, Math.max(0, numReadings)), topSamples: [], bottomSamples: [] };
  }

  const top = buildCurve(toPx(topPct[0]), toPx(topPct[1]), toPx(topPct[2]));
  const bottom = buildCurve(toPx(botPct[0]), toPx(botPct[1]), toPx(botPct[2]));
  const topSamplesPx = top.sample(ARC_SAMPLES);
  const bottomSamplesPx = bottom.sample(ARC_SAMPLES);

  const n = Math.max(1, Math.floor(numReadings));
  const L = top.length;
  const stepS = Math.max(0, spacing) * L;   // arc-length step in pixels
  const centerS = Math.min(1, Math.max(0, center)) * L;

  const readings: ThicknessReadingData[] = [];
  for (let k = 0; k < n; k++) {
    const prev = existing[k];
    if (prev && prev.edited) {
      // Frozen: keep the user's manual placement verbatim.
      readings.push({ ...prev });
      continue;
    }
    // arc-length position of reading k, symmetric about the centre
    let s = centerS + (k - (n - 1) / 2) * stepS;
    s = Math.min(L, Math.max(0, s));
    const t = L < EPS ? 0 : s / L;
    const T = top.pointAt(t);
    const nOut = top.normalAt(t);
    // Choose the radial direction that points toward the bottom surface.
    // Use the NEAREST point on the bottom arc as the "toward" reference —
    // robust for concentric arcs (where the shared circle centre would
    // point the wrong way) as well as tilted / parallel surfaces.
    const nb = nearestOnPolyline(bottomSamplesPx, T);
    const toward = sub(nb, T);
    const nDir: Pt = dot(nOut, toward) >= 0 ? nOut : [-nOut[0], -nOut[1]];
    // Perpendicular line through T, intersect with the bottom curve.
    const cands = bottom.intersectLine(T, nDir);
    let B: Pt | null = null;
    let bestScore = Infinity;
    for (const cand of cands) {
      const along = dot(sub(cand, T), nDir);     // >0 = toward bottom
      // Prefer candidates in front; distance is the tiebreaker.
      const score = (along > 0 ? 0 : 1e6) + Math.abs(along);
      if (score < bestScore) { bestScore = score; B = cand; }
    }
    if (!B) B = nb;   // fallback: closest bottom point (line missed the arc)
    readings.push({
      top: toPct(T),
      bottom: toPct(B),
      hidden: prev ? prev.hidden : false,   // carry hide/relabel across regen
      text: prev ? prev.text : "",
      edited: false,
    });
  }

  return {
    readings,
    topSamples: topSamplesPx.map(toPct),
    bottomSamples: bottomSamplesPx.map(toPct),
  };
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
