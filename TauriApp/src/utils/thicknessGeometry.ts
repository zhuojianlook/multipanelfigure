/* ──────────────────────────────────────────────────────────
   thicknessGeometry — geometry for the Curved Surface Measurement
   annotation (thickness readings between two curved surfaces).

   Each surface is a curve through the user's clicked points — as many as
   they like: 2 points = a straight segment, 3 = the circular arc through
   them (unchanged from the original 3-point design, so older files measure
   identically), 4 or more = a smooth centripetal Catmull-Rom spline that
   passes through every point. An odd number of readings is placed along
   the TOP surface: one exactly on the user's centre node, the rest spread
   symmetrically either side, spaced `spacing` apart as a fraction of the
   top surface's length.

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
  /** Per-reading label size (pt, same 216pt reference as the measurement's
   *  measure_font_size) / colour. null or absent = use the measurement's. */
  font_size?: number | null;
  color?: string | null;
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
  /** Arc-length fraction (0..1) at which each control point sits. */
  knotT: number[];
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

/** Straight segment a→b. */
function lineCurve(a: Pt, b: Pt): Curve {
  const len = norm(sub(b, a));
  const d: Pt = len < EPS ? [1, 0] : [(b[0] - a[0]) / len, (b[1] - a[1]) / len];
  const nrm: Pt = [-d[1], d[0]];
  const at = (t: number): Pt => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  return {
    length: len,
    knotT: [0, 1],
    pointAt: at,
    normalAt: () => nrm,
    sample: (n) => Array.from({ length: n }, (_, i) => at(n <= 1 ? 0 : i / (n - 1))),
  };
}

/** Circular arc through 3 points (the original surface model — kept
 *  byte-for-byte so 3-point surfaces resolve exactly as before). Collinear
 *  points fall back to the straight segment p1→p3. */
function arcCurve(p1: Pt, p2: Pt, p3: Pt): Curve {
  const circle = circleFrom3(p1, p2, p3);

  if (!circle) {
    // Straight-segment fallback (collinear points).
    const line = lineCurve(p1, p3);
    const len = line.length;
    const mid = len < EPS ? 0.5
      : Math.min(1, Math.max(0, dot(sub(p2, p1), sub(p3, p1)) / (len * len)));
    return { ...line, knotT: [0, mid, 1] };
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
  const midT = span < EPS ? 0.5 : (dir === 1 ? a2rel : TWO_PI - a2rel) / span;
  return {
    length: r * span,
    knotT: [0, Math.min(1, Math.max(0, midT)), 1],
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

/** Reflect `x` across the perpendicular bisector of a-b. That bisector
 *  passes through the centre of any circle through a and b, so a point on
 *  such a circle stays on it — which is why the spline's end phantoms use
 *  it (the ends keep curving instead of flattening out). */
function reflectAcrossBisector(x: Pt, a: Pt, b: Pt): Pt {
  const ab = sub(b, a);
  const l = norm(ab);
  if (l < EPS) return [2 * a[0] - x[0], 2 * a[1] - x[1]];
  const u: Pt = [ab[0] / l, ab[1] / l];
  const m: Pt = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const k = 2 * dot(sub(x, m), u);
  return [x[0] - k * u[0], x[1] - k * u[1]];
}

const SPLINE_STEPS = 32;   // arc-length table sub-steps per span

/** Centripetal Catmull-Rom spline (alpha 0.5, Barry-Goldman evaluation)
 *  through every point (4 or more, consecutive points distinct). Centripetal
 *  parametrisation interpolates each point, is C1 and has no cusps or
 *  self-intersections within a span — uniform Catmull-Rom and natural
 *  cubics loop or overshoot on unevenly spaced clicks. `t` is arc-length
 *  fraction, via a per-span lookup table. */
function splineCurve(P: Pt[]): Curve {
  const n = P.length;
  const ext: Pt[] = [
    reflectAcrossBisector(P[2], P[0], P[1]),
    ...P,
    reflectAcrossBisector(P[n - 3], P[n - 1], P[n - 2]),
  ];
  // Knots: t[i+1] = t[i] + |P[i+1]-P[i]|^0.5 (centripetal).
  const kt: number[] = [0];
  for (let i = 1; i < ext.length; i++) {
    kt.push(kt[i - 1] + Math.max(1e-9, Math.sqrt(norm(sub(ext[i], ext[i - 1])))));
  }
  const lerp = (a: Pt, b: Pt, ta: number, tb: number, t: number): Pt => {
    const d = tb - ta;
    const wa = (tb - t) / d, wb = (t - ta) / d;
    return [a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb];
  };
  /** Point on span s (between P[s] and P[s+1]) at local u in [0,1]. */
  const evalSpan = (s: number, u: number): Pt => {
    if (u <= 0) return P[s];
    if (u >= 1) return P[s + 1];
    const p0 = ext[s], p1 = ext[s + 1], p2 = ext[s + 2], p3 = ext[s + 3];
    const t0 = kt[s], t1 = kt[s + 1], t2 = kt[s + 2], t3 = kt[s + 3];
    const t = t1 + (t2 - t1) * u;
    const a1 = lerp(p0, p1, t0, t1, t);
    const a2 = lerp(p1, p2, t1, t2, t);
    const a3 = lerp(p2, p3, t2, t3, t);
    const b1 = lerp(a1, a2, t0, t2, t);
    const b2 = lerp(a2, a3, t1, t3, t);
    return lerp(b1, b2, t1, t2, t);
  };

  // Arc-length table: entry j = (span, u, cumulative length). Span
  // boundaries are the control points themselves, exactly.
  const tSpan: number[] = [0], tU: number[] = [0], tCum: number[] = [0];
  const knotT: number[] = [0];
  let prev = P[0], cum = 0;
  for (let s = 0; s < n - 1; s++) {
    for (let k = 1; k <= SPLINE_STEPS; k++) {
      const u = k / SPLINE_STEPS;
      const pt = k === SPLINE_STEPS ? P[s + 1] : evalSpan(s, u);
      cum += norm(sub(pt, prev));
      prev = pt;
      tSpan.push(s); tU.push(u); tCum.push(cum);
    }
    knotT.push(cum);
  }
  const L = cum;
  for (let k = 0; k < knotT.length; k++) knotT[k] = L < EPS ? k / (n - 1) : knotT[k] / L;
  knotT[knotT.length - 1] = 1;

  const pointAt = (t: number): Pt => {
    if (L < EPS) return P[0];
    const tc = Math.min(1, Math.max(0, t));
    if (tc <= 0) return P[0];
    if (tc >= 1) return P[n - 1];
    const sTarget = tc * L;
    // First entry with cum >= sTarget (entries 1..last).
    let lo = 1, hi = tCum.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (tCum[mid] < sTarget) lo = mid + 1; else hi = mid;
    }
    const j = lo;
    const span = tSpan[j];
    const uEnd = tU[j];
    const uStart = tSpan[j - 1] === span ? tU[j - 1] : 0;
    const cellLen = tCum[j] - tCum[j - 1];
    const frac = cellLen < EPS ? 0 : (sTarget - tCum[j - 1]) / cellLen;
    return evalSpan(span, uStart + (uEnd - uStart) * frac);
  };
  return {
    length: L,
    knotT,
    pointAt,
    normalAt: (t) => {
      const h = 1e-4;
      const a = pointAt(Math.max(0, t - h)), b = pointAt(Math.min(1, t + h));
      const d = sub(b, a);
      const l = norm(d);
      return l < EPS ? [0, 1] : [-d[1] / l, d[0] / l];
    },
    sample: (m) => Array.from({ length: m }, (_, i) => pointAt(m <= 1 ? 0 : i / (m - 1))),
  };
}

/** Indices of the points kept after dropping consecutive near-duplicates. */
function dedupeIdx(pts: Pt[]): number[] {
  const keep: number[] = [];
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    if (!Number.isFinite(p?.[0]) || !Number.isFinite(p?.[1])) continue;
    if (keep.length && norm(sub(p, pts[keep[keep.length - 1]])) < 1e-6) continue;
    keep.push(i);
  }
  return keep;
}

/** The surface through `ptsPx` (pixel space): 2 points → straight segment,
 *  3 → circular arc (bit-identical to the original model), 4+ → smooth
 *  spline. Consecutive duplicates (a double-click) are ignored. null when
 *  fewer than 2 distinct points remain. */
function buildCurveN(ptsPx: Pt[]): Curve | null {
  const P = dedupeIdx(ptsPx).map((i) => ptsPx[i]);
  if (P.length < 2) return null;
  if (P.length === 2) return lineCurve(P[0], P[1]);
  if (P.length === 3) return arcCurve(P[0], P[1], P[2]);
  return splineCurve(P);
}

/** Polyline sample count: 96 for 2/3-point surfaces (as before), ~32 per
 *  span for splines so long many-point surfaces stay smooth. */
function sampleCount(c: Curve): number {
  const nPts = c.knotT.length;
  return nPts <= 3 ? 96 : Math.min(2048, 32 * (nPts - 1) + 1);
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

/**
 * Compute thickness readings between the two surfaces (2+ points each).
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
  if (iw <= 0 || ih <= 0) {
    return { readings: existing.slice(0, n), topSamples: [], bottomSamples: [] };
  }

  // Each surface is resolved on its own, so a half-traced measurement still
  // shows the surface drawn so far; readings need both.
  const top = topPct.length >= 2 ? buildCurveN(topPct.map(toPx)) : null;
  const bottom = botPct.length >= 2 ? buildCurveN(botPct.map(toPx)) : null;
  const topSamplesPx = top ? top.sample(sampleCount(top)) : [];
  const bottomSamplesPx = bottom ? bottom.sample(sampleCount(bottom)) : [];
  if (!top || !bottom) {
    return {
      readings: existing.slice(0, n),
      topSamples: topSamplesPx.map(toPct),
      bottomSamples: bottomSamplesPx.map(toPct),
    };
  }

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
      font_size: prev?.font_size ?? null,
      color: prev?.color ?? null,
    });
  }

  return {
    readings,
    topSamples: topSamplesPx.map(toPct),
    bottomSamples: bottomSamplesPx.map(toPct),
  };
}

/**
 * Snap a freely-dragged point onto a surface when it lands close to it.
 * Lets a manual reading endpoint sit exactly on the surface without forcing
 * it — beyond `tolPct` of the image's smaller side, the raw point is kept.
 *
 * @param pPct dragged point (x%, y%)
 * @param curvePct the surface's defining points (2+)
 * @returns {point, snapped}
 */
export function snapToCurve(
  // tolPct was 1.5% of the smaller side — so tight that dragging a reading
  // endpoint essentially never snapped back onto the surface, and the snap
  // read as non-existent. 4% is a reachable grab distance while still leaving
  // deliberate free placement possible further out.
  pPct: Pt, curvePct: Pt[], iw: number, ih: number, tolPct = 4,
): { point: Pt; snapped: boolean } {
  if (curvePct.length < 2 || iw <= 0 || ih <= 0) return { point: pPct, snapped: false };
  const toPx = (p: Pt): Pt => [(p[0] / 100) * iw, (p[1] / 100) * ih];
  const curve = buildCurveN(curvePct.map(toPx));
  if (!curve) return { point: pPct, snapped: false };
  const { pt, dist } = nearestOnPolyline(curve.sample(sampleCount(curve)), toPx(pPct));
  const tolPx = (tolPct / 100) * Math.min(iw, ih);
  if (dist > tolPx) return { point: pPct, snapped: false };
  return { point: [(pt[0] / iw) * 100, (pt[1] / ih) * 100], snapped: true };
}

/** Parameter (0..1) along the top arc nearest to a dragged point — used by
 *  the centre node, which slides along the surface rather than free-floating. */
export function paramOnCurve(pPct: Pt, curvePct: Pt[], iw: number, ih: number): number {
  if (curvePct.length < 2 || iw <= 0 || ih <= 0) return 0.5;
  const toPx = (p: Pt): Pt => [(p[0] / 100) * iw, (p[1] / 100) * ih];
  const curve = buildCurveN(curvePct.map(toPx));
  if (!curve) return 0.5;
  const q = toPx(pPct);
  const N = sampleCount(curve);
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
  if (curvePct.length < 2 || iw <= 0 || ih <= 0) return null;
  const toPx = (p: Pt): Pt => [(p[0] / 100) * iw, (p[1] / 100) * ih];
  const curve = buildCurveN(curvePct.map(toPx));
  if (!curve) return null;
  const p = curve.pointAt(Math.min(1, Math.max(0, t)));
  return [(p[0] / iw) * 100, (p[1] / ih) * 100];
}

/** Physical length (px) of the surface through `curvePct`; 0 when it has
 *  fewer than 2 distinct points. */
export function curveLength(curvePct: Pt[], iw: number, ih: number): number {
  if (curvePct.length < 2 || iw <= 0 || ih <= 0) return 0;
  const toPx = (p: Pt): Pt => [(p[0] / 100) * iw, (p[1] / 100) * ih];
  return buildCurveN(curvePct.map(toPx))?.length ?? 0;
}

/** Arc-length fraction (0..1) at which each distinct control point sits on
 *  its surface ([] when fewer than 2 distinct points). */
export function surfaceKnots(curvePct: Pt[], iw: number, ih: number): number[] {
  if (curvePct.length < 2 || iw <= 0 || ih <= 0) return [];
  const toPx = (p: Pt): Pt => [(p[0] / 100) * iw, (p[1] / 100) * ih];
  return buildCurveN(curvePct.map(toPx))?.knotT.slice() ?? [];
}

/** Nearest parameter (0..1) on a curve's dense polyline, projecting onto
 *  each segment (finer than snapping to a sample vertex). */
function nearestParam(curve: Curve, q: Pt): number {
  const N = sampleCount(curve);
  const pts = curve.sample(N);
  let bestT = 0, bestD = Infinity;
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], ab = sub(pts[i + 1], a);
    const len2 = dot(ab, ab);
    const f = len2 < EPS ? 0 : Math.max(0, Math.min(1, dot(sub(q, a), ab) / len2));
    const d = norm(sub(q, [a[0] + ab[0] * f, a[1] + ab[1] * f]));
    if (d < bestD) { bestD = d; bestT = (i + f) / (N - 1); }
  }
  return bestT;
}

/**
 * Where a click on (or near) a finished surface inserts a new point, so the
 * surface grows through it without reordering the others. Clicking past
 * either end extends that end; clicking alongside a span splits it.
 *
 * @returns the new point list and the index the new point landed at.
 */
export function insertSurfacePoint(
  ptsPct: Pt[], pPct: Pt, iw: number, ih: number,
): { points: Pt[]; index: number } {
  const n = ptsPct.length;
  const toPx = (p: Pt): Pt => [(p[0] / 100) * iw, (p[1] / 100) * ih];
  const pxPts = ptsPct.map(toPx);
  const keep = dedupeIdx(pxPts);
  const curve = iw > 0 && ih > 0 && n >= 2 ? buildCurveN(pxPts) : null;
  let index = n;
  if (curve) {
    const t = nearestParam(curve, toPx(pPct));
    if (t <= 1e-6) index = 0;
    else if (t >= 1 - 1e-6) index = n;
    else {
      const count = curve.knotT.filter((k) => k <= t).length;   // >= 1 (knotT[0] = 0)
      // Map back to the caller's indices (consecutive duplicates were
      // dropped from the curve): insert after the last copy of that point.
      const raw = count < keep.length ? keep[count] : n;
      index = Math.min(n - 1, Math.max(1, raw));
    }
  }
  const points = [...ptsPct.slice(0, index), pPct, ...ptsPct.slice(index)];
  return { points, index };
}

/** Remove point `i` (pure; the 2-point minimum is enforced by the UI). */
export function removeSurfacePoint(ptsPct: Pt[], i: number): Pt[] {
  if (i < 0 || i >= ptsPct.length) return [...ptsPct];
  return [...ptsPct.slice(0, i), ...ptsPct.slice(i + 1)];
}

/**
 * Keep the reading set where it physically was when a TOP point is added or
 * removed. `center` / `spacing` are fractions of the top surface's length,
 * so extending an end would otherwise slide every reading along: re-find the
 * centre node's spot on the new surface and rescale the spacing so the
 * physical gap between readings is unchanged. (Drags keep the old fractional
 * behaviour.)
 */
export function reanchorTop(
  oldTop: Pt[], newTop: Pt[], center: number, spacing: number, iw: number, ih: number,
): { center: number; spacing: number } {
  if (iw <= 0 || ih <= 0) return { center, spacing };
  const toPx = (p: Pt): Pt => [(p[0] / 100) * iw, (p[1] / 100) * ih];
  const oldC = oldTop.length >= 2 ? buildCurveN(oldTop.map(toPx)) : null;
  const newC = newTop.length >= 2 ? buildCurveN(newTop.map(toPx)) : null;
  if (!oldC || !newC || newC.length < EPS) return { center, spacing };
  const C = oldC.pointAt(Math.min(1, Math.max(0, center)));
  return {
    center: nearestParam(newC, C),
    spacing: spacing * (oldC.length / newC.length),
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
