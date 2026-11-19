import { describe, it, expect } from "vitest";
import {
  computeThicknessReadings, readingValue, normalizeCount, maxSpacingFor,
  snapToCurve, paramOnCurve, pointOnCurve,
  insertSurfacePoint, removeSurfacePoint, reanchorTop, curveLength, surfaceKnots,
  type Pt, type ThicknessReadingData,
} from "./thicknessGeometry";
import golden from "./thicknessGeometry.golden.json";

// 3 points on a circle (centre cx,cy in px), radius r, at given degrees → %.
function arcPts(cx: number, cy: number, r: number, degs: number[], iw: number, ih: number): Pt[] {
  return degs.map((d) => {
    const a = (d * Math.PI) / 180;
    return [((cx + r * Math.cos(a)) / iw) * 100, ((cy + r * Math.sin(a)) / ih) * 100] as Pt;
  });
}

const IW = 1000, IH = 1000;

describe("counts and spacing", () => {
  it("forces odd counts so a reading always lands on the centre node", () => {
    expect(normalizeCount(1)).toBe(1);
    expect(normalizeCount(2)).toBe(3);
    expect(normalizeCount(4)).toBe(5);
    expect(normalizeCount(5)).toBe(5);
    expect(normalizeCount(0)).toBe(1);
  });

  it("caps spacing so the group always fits inside the arc (centred)", () => {
    expect(maxSpacingFor(1)).toBe(1);
    expect(maxSpacingFor(3)).toBeCloseTo(0.5, 6);   // 2 gaps
    expect(maxSpacingFor(5)).toBeCloseTo(0.25, 6);  // 4 gaps
    expect(maxSpacingFor(4)).toBeCloseTo(0.25, 6);  // rounded up to 5
  });

  it("shrinks the cap when the centre node moves off-middle", () => {
    // Readings spread symmetrically, so the SHORTER side binds:
    // s_max = 2*min(c, 1-c)/(n-1).
    expect(maxSpacingFor(5, 0.5)).toBeCloseTo(0.25, 6);
    expect(maxSpacingFor(5, 0.34)).toBeCloseTo(0.17, 6);  // the user's example
    expect(maxSpacingFor(5, 0.25)).toBeCloseTo(0.125, 6);
    // Mirrored either side of the middle.
    expect(maxSpacingFor(5, 0.2)).toBeCloseTo(maxSpacingFor(5, 0.8), 9);
    // Hard against an end → floored, not zero (keeps the slider usable).
    expect(maxSpacingFor(5, 0)).toBeGreaterThan(0);
  });

  it("readings stay inside the arc when the centre is off-middle", () => {
    const top: Pt[] = [[20, 30], [50, 30], [80, 30]];
    const bot: Pt[] = [[20, 50], [50, 50], [80, 50]];
    const centre = 0.25;
    const cap = maxSpacingFor(5, centre);
    const { readings } = computeThicknessReadings(top, bot, 5, centre, cap, IW, IH);
    // The arc spans x 20..80; every anchor must land within it (no pile-up at
    // an end, which is what an uncapped spacing would cause).
    const xs = readings.map((r) => r.top[0]);
    for (const x of xs) {
      expect(x).toBeGreaterThanOrEqual(20 - 0.5);
      expect(x).toBeLessThanOrEqual(80 + 0.5);
    }
    // and they're all distinct (nothing clamped on top of anything else)
    expect(new Set(xs.map((x) => x.toFixed(2))).size).toBe(5);
  });

  it("an even count still yields an odd number of readings", () => {
    const top: Pt[] = [[20, 30], [50, 30], [80, 30]];
    const bot: Pt[] = [[20, 50], [50, 50], [80, 50]];
    const { readings } = computeThicknessReadings(top, bot, 4, 0.5, 0.2, IW, IH);
    expect(readings.length).toBe(5);
  });
});

describe("reading geometry", () => {
  it("concentric arcs → minimal distance = radius difference", () => {
    const top = arcPts(500, 300, 200, [60, 90, 120], IW, IH);
    const bot = arcPts(500, 300, 260, [60, 90, 120], IW, IH);
    const { readings } = computeThicknessReadings(top, bot, 1, 0.5, 0.1, IW, IH);
    expect(readings.length).toBe(1);
    const r0 = readings[0];
    expect(r0.top[0]).toBeCloseTo(50, 1);
    expect(r0.top[1]).toBeCloseTo(50, 1);
    expect(readingValue(r0, IW, IH, 1.0, "um")).toBeCloseTo(60, 0);
  });

  it("straight parallel surfaces → perpendicular, equal thickness", () => {
    const top: Pt[] = [[20, 30], [50, 30], [80, 30]];
    const bot: Pt[] = [[20, 50], [50, 50], [80, 50]];
    const { readings } = computeThicknessReadings(top, bot, 3, 0.5, 0.25, IW, IH);
    expect(readings.length).toBe(3);
    for (const r of readings) {
      expect(r.bottom[0]).toBeCloseTo(r.top[0], 1);           // straight down
      expect(readingValue(r, IW, IH, 1.0, "um")).toBeCloseTo(200, 0);
    }
  });

  it("a reading always sits exactly on the centre node", () => {
    const top = arcPts(500, 300, 200, [50, 90, 130], IW, IH);
    const bot = arcPts(500, 300, 260, [50, 90, 130], IW, IH);
    for (const centre of [0.2, 0.5, 0.77]) {
      const { readings } = computeThicknessReadings(top, bot, 5, centre, 0.15, IW, IH);
      const mid = readings[(readings.length - 1) / 2];
      const node = pointOnCurve(top, centre, IW, IH)!;
      expect(mid.top[0]).toBeCloseTo(node[0], 4);
      expect(mid.top[1]).toBeCloseTo(node[1], 4);
    }
  });

  it("minimal distance beats a radial cast for offset (non-concentric) surfaces", () => {
    // Bottom circle centred elsewhere → the radial from the top would hit the
    // bottom at a longer chord than the true nearest point.
    const top = arcPts(500, 300, 200, [70, 90, 110], IW, IH);
    const bot = arcPts(560, 340, 260, [70, 90, 110], IW, IH);
    const { readings, bottomSamples } = computeThicknessReadings(top, bot, 1, 0.5, 0.1, IW, IH);
    const r = readings[0];
    // The chosen foot must be the closest bottom sample to the anchor.
    const anchorPx = [r.top[0] / 100 * IW, r.top[1] / 100 * IH];
    let best = Infinity;
    for (const s of bottomSamples) {
      const d = Math.hypot(s[0] / 100 * IW - anchorPx[0], s[1] / 100 * IH - anchorPx[1]);
      if (d < best) best = d;
    }
    const got = Math.hypot(
      r.bottom[0] / 100 * IW - anchorPx[0],
      r.bottom[1] / 100 * IH - anchorPx[1],
    );
    // within a sample-spacing of the true minimum
    expect(got).toBeLessThanOrEqual(best + 1.0);
  });

  it("non-square image: geometry stays correct in pixel space", () => {
    const iw = 2000, ih = 1000;
    const top = arcPts(1000, 300, 200, [60, 90, 120], iw, ih);
    const bot = arcPts(1000, 300, 260, [60, 90, 120], iw, ih);
    const { readings } = computeThicknessReadings(top, bot, 1, 0.5, 0.1, iw, ih);
    expect(readingValue(readings[0], iw, ih, 1.0, "um")).toBeCloseTo(60, 0);
  });
});

describe("overrides", () => {
  it("keeps edited readings and carries hidden/text/label-pos across regen", () => {
    const top: Pt[] = [[20, 30], [50, 30], [80, 30]];
    const bot: Pt[] = [[20, 50], [50, 50], [80, 50]];
    const existing: ThicknessReadingData[] = [
      { top: [10, 30], bottom: [10, 90], hidden: false, text: "custom", edited: true },
      { top: [50, 30], bottom: [50, 50], hidden: true, text: "", edited: false },
      { top: [80, 30], bottom: [80, 50], hidden: false, text: "hi", edited: false,
        measure_position_x: 42, measure_position_y: 43 },
    ];
    const { readings } = computeThicknessReadings(top, bot, 3, 0.5, 0.25, IW, IH, existing);
    expect(readings[0]).toEqual(existing[0]);      // frozen verbatim
    expect(readings[1].hidden).toBe(true);         // hide persists
    expect(readings[2].text).toBe("hi");           // text persists
    expect(readings[2].measure_position_x).toBe(42); // manual label pos persists
  });
});

describe("snapping", () => {
  const curve: Pt[] = [[20, 30], [50, 30], [80, 30]];

  it("snaps a near point onto the surface", () => {
    const { point, snapped } = snapToCurve([50, 30.6], curve, IW, IH, 1.5);
    expect(snapped).toBe(true);
    expect(point[1]).toBeCloseTo(30, 3);
  });

  it("leaves a far point free (free specification still allowed)", () => {
    const { point, snapped } = snapToCurve([50, 40], curve, IW, IH, 1.5);
    expect(snapped).toBe(false);
    expect(point).toEqual([50, 40]);
  });

  it("paramOnCurve finds where along the surface a point lies", () => {
    expect(paramOnCurve([20, 30], curve, IW, IH)).toBeCloseTo(0, 1);
    expect(paramOnCurve([50, 30], curve, IW, IH)).toBeCloseTo(0.5, 1);
    expect(paramOnCurve([80, 30], curve, IW, IH)).toBeCloseTo(1, 1);
  });
});

/* ── N-point surfaces ─────────────────────────────────────────────────── */

type GoldenCase = {
  name: string; iw: number; ih: number; top: Pt[]; bot: Pt[];
  n: number; center: number; spacing: number; existing: ThicknessReadingData[];
  result: { readings: ThicknessReadingData[]; topSamples: Pt[]; bottomSamples: Pt[] };
  snap: { point: Pt; snapped: boolean }[]; snapBot: { point: Pt; snapped: boolean }[];
  param: number[]; pointAt: Pt[];
};
const GOLDEN = (golden as unknown as { cases: GoldenCase[] }).cases;
const PROBES: Pt[] = [[50, 52], [30, 40], [70, 61], [5, 5]];

describe("3-point surfaces are unchanged (goldens from the original arc code)", () => {
  it("has the four captured cases", () => {
    expect(GOLDEN.map((c) => c.name)).toEqual(["concentric", "collinear", "nonsquare", "offcentre_edited"]);
  });
  for (const c of GOLDEN) {
    it(`${c.name}: readings, samples, snap, param and point are bit-identical`, () => {
      const res = computeThicknessReadings(c.top, c.bot, c.n, c.center, c.spacing, c.iw, c.ih, c.existing);
      // The only addition: regenerated readings now carry the per-reading
      // style overrides (null = inherit). Edited readings stay verbatim.
      const expected = c.result.readings.map((r) => (r.edited ? r : { ...r, font_size: null, color: null }));
      expect(res.readings).toEqual(expected);
      expect(res.topSamples).toEqual(c.result.topSamples);
      expect(res.bottomSamples).toEqual(c.result.bottomSamples);
      expect(res.topSamples.length).toBe(96);
      expect(PROBES.map((p) => snapToCurve(p, c.top, c.iw, c.ih))).toEqual(c.snap);
      expect(PROBES.map((p) => snapToCurve(p, c.bot, c.iw, c.ih))).toEqual(c.snapBot);
      expect(PROBES.map((p) => paramOnCurve(p, c.top, c.iw, c.ih))).toEqual(c.param);
      expect([0, 0.13, 0.5, 0.87, 1].map((t) => pointOnCurve(c.top, t, c.iw, c.ih))).toEqual(c.pointAt);
    });
  }
});

describe("2-point surfaces", () => {
  it("are straight: perpendicular readings of equal value, 96 samples", () => {
    const top: Pt[] = [[20, 30], [80, 30]];
    const bot: Pt[] = [[20, 50], [80, 50]];
    const { readings, topSamples, bottomSamples } = computeThicknessReadings(top, bot, 5, 0.5, 0.2, IW, IH);
    expect(readings.length).toBe(5);
    expect(topSamples.length).toBe(96);
    expect(bottomSamples.length).toBe(96);
    for (const s of topSamples) expect(s[1]).toBeCloseTo(30, 9);
    for (const r of readings) {
      expect(r.bottom[0]).toBeCloseTo(r.top[0], 6);
      expect(readingValue(r, IW, IH, 1, "um")).toBeCloseTo(200, 6);
    }
  });
});

describe("4+ point surfaces (smooth spline)", () => {
  it("passes through every control point exactly (at its knot)", () => {
    const pts: Pt[] = [[10, 40], [25, 30], [45, 28], [70, 35], [90, 50]];
    const knots = surfaceKnots(pts, IW, IH);
    expect(knots.length).toBe(5);
    expect(knots[0]).toBe(0);
    expect(knots[4]).toBe(1);
    for (let k = 1; k < 5; k++) expect(knots[k]).toBeGreaterThan(knots[k - 1]);
    pts.forEach((p, k) => {
      const q = pointOnCurve(pts, knots[k], IW, IH)!;
      expect(Math.hypot((q[0] - p[0]) / 100 * IW, (q[1] - p[1]) / 100 * IH)).toBeLessThan(1e-6);
    });
    // and the dense samples run end to end
    const { topSamples } = computeThicknessReadings(pts, pts, 1, 0.5, 0.1, IW, IH);
    expect(topSamples.length).toBe(32 * 4 + 1);
    expect(topSamples[0]).toEqual(pts[0]);
    expect(topSamples[topSamples.length - 1]).toEqual(pts[4]);
  });

  it("follows a circle closely: 5 points 30° apart stay within 0.5% of r", () => {
    const cx = 500, cy = 600, r = 300;
    const pts = arcPts(cx, cy, r, [210, 240, 270, 300, 330], IW, IH);
    const { topSamples } = computeThicknessReadings(pts, pts, 1, 0.5, 0.1, IW, IH);
    for (const s of topSamples) {
      const d = Math.hypot(s[0] / 100 * IW - cx, s[1] / 100 * IH - cy);
      expect(Math.abs(d - r) / r).toBeLessThan(0.005);
    }
    expect(curveLength(pts, IW, IH)).toBeCloseTo(r * (120 * Math.PI / 180), -1);
  });

  it("does not loop or blow up on uneven zig-zag clicks", () => {
    const pts: Pt[] = [[10, 50], [12, 40], [40, 60], [42, 45], [80, 55], [81, 52]];
    const { topSamples } = computeThicknessReadings(pts, pts, 1, 0.5, 0.1, IW, IH);
    for (const s of topSamples) { expect(Number.isFinite(s[0])).toBe(true); expect(Number.isFinite(s[1])).toBe(true); }
    let chord = 0;
    for (let i = 1; i < pts.length; i++) {
      chord += Math.hypot((pts[i][0] - pts[i - 1][0]) / 100 * IW, (pts[i][1] - pts[i - 1][1]) / 100 * IH);
    }
    const L = curveLength(pts, IW, IH);
    expect(L).toBeGreaterThanOrEqual(chord - 1e-6);
    expect(L).toBeLessThan(1.5 * chord);
  });

  it("ignores consecutive duplicate points (a double-click) without NaN", () => {
    const pts: Pt[] = [[10, 40], [30, 30], [30, 30], [60, 32], [90, 45], [90, 45]];
    const bot: Pt[] = [[10, 60], [50, 55], [50, 55], [90, 65]];
    const { readings, topSamples, bottomSamples } = computeThicknessReadings(pts, bot, 5, 0.5, 0.1, IW, IH);
    expect(readings.length).toBe(5);
    for (const s of [...topSamples, ...bottomSamples]) expect(Number.isNaN(s[0]) || Number.isNaN(s[1])).toBe(false);
    for (const r of readings) expect(Number.isFinite(readingValue(r, IW, IH, 1, "um"))).toBe(true);
    expect(surfaceKnots(pts, IW, IH).length).toBe(4);
    // A bottom with 3 points left after de-duplication is the plain arc.
    const deduped = computeThicknessReadings(pts, [[10, 60], [50, 55], [90, 65]], 5, 0.5, 0.1, IW, IH);
    expect(bottomSamples).toEqual(deduped.bottomSamples);
  });
});

describe("adding / removing surface points", () => {
  const pts: Pt[] = [[10, 50], [30, 40], [60, 40], [90, 50]];

  it("a click alongside a span splits that span", () => {
    // Between points 2 and 3 (indices 1, 2) → lands at index 2.
    const { points, index } = insertSurfacePoint(pts, [45, 38.5], IW, IH);
    expect(index).toBe(2);
    expect(points.length).toBe(5);
    expect(points[2]).toEqual([45, 38.5]);
    expect(points.filter((_, i) => i !== 2)).toEqual(pts);
    // Between points 1 and 2.
    expect(insertSurfacePoint(pts, [20, 44], IW, IH).index).toBe(1);
    // Between points 3 and 4.
    expect(insertSurfacePoint(pts, [76, 44], IW, IH).index).toBe(3);
  });

  it("a click beyond either end extends that end", () => {
    expect(insertSurfacePoint(pts, [2, 56], IW, IH).index).toBe(0);
    expect(insertSurfacePoint(pts, [98, 57], IW, IH).index).toBe(4);
  });

  it("works on a 2-point line (→ index 1) and appends while tracing", () => {
    const line: Pt[] = [[10, 50], [90, 50]];
    expect(insertSurfacePoint(line, [50, 47], IW, IH)).toEqual({ points: [[10, 50], [50, 47], [90, 50]], index: 1 });
    expect(insertSurfacePoint([[10, 50]], [50, 47], IW, IH)).toEqual({ points: [[10, 50], [50, 47]], index: 1 });
    expect(insertSurfacePoint([], [50, 47], IW, IH)).toEqual({ points: [[50, 47]], index: 0 });
  });

  it("works on a 3-point arc", () => {
    const arc = arcPts(500, 300, 200, [60, 90, 120], IW, IH);
    const mid12 = pointOnCurve(arc, 0.25, IW, IH)!;
    expect(insertSurfacePoint(arc, mid12, IW, IH).index).toBe(1);
    const mid23 = pointOnCurve(arc, 0.75, IW, IH)!;
    expect(insertSurfacePoint(arc, mid23, IW, IH).index).toBe(2);
  });

  it("removeSurfacePoint splices one point out", () => {
    expect(removeSurfacePoint(pts, 1)).toEqual([[10, 50], [60, 40], [90, 50]]);
    expect(removeSurfacePoint(pts, 9)).toEqual(pts);
    expect(pts.length).toBe(4);   // pure
  });

  it("reanchorTop keeps the centre node in place when a point is added", () => {
    const oldTop: Pt[] = [[20, 40], [50, 32], [80, 40]];
    const center = 0.4, spacing = 0.1;
    const before = pointOnCurve(oldTop, center, IW, IH)!;
    // Extend the start: fractions along the surface would shift without
    // re-anchoring, since the surface just got longer at the front.
    const { points: newTop } = insertSurfacePoint(oldTop, [5, 52], IW, IH);
    expect(newTop[0]).toEqual([5, 52]);
    const re = reanchorTop(oldTop, newTop, center, spacing, IW, IH);
    const after = pointOnCurve(newTop, re.center, IW, IH)!;
    expect(Math.hypot((after[0] - before[0]) / 100 * IW, (after[1] - before[1]) / 100 * IH)).toBeLessThan(0.01 * IW);
    const Lold = curveLength(oldTop, IW, IH), Lnew = curveLength(newTop, IW, IH);
    expect(Lnew).toBeGreaterThan(Lold);
    expect(re.spacing * Lnew).toBeCloseTo(spacing * Lold, 6);   // same physical gap
    // Without re-anchoring the node would have slid noticeably.
    const naive = pointOnCurve(newTop, center, IW, IH)!;
    expect(Math.hypot((naive[0] - before[0]) / 100 * IW, (naive[1] - before[1]) / 100 * IH)).toBeGreaterThan(0.02 * IW);
  });
});

describe("readings on N-point surfaces", () => {
  const top: Pt[] = [[10, 40], [30, 32], [50, 30], [70, 32], [90, 40]];
  const bot: Pt[] = [[10, 60], [35, 52], [65, 52], [90, 60]];

  it("count stays odd and every reading sits between the surfaces", () => {
    for (const n of [1, 4, 7]) {
      const { readings } = computeThicknessReadings(top, bot, n, 0.5, 0.05, IW, IH);
      expect(readings.length).toBe(normalizeCount(n));
      for (const r of readings) expect(r.bottom[1]).toBeGreaterThan(r.top[1]);
    }
    // A reading still lands exactly on the centre node.
    const { readings } = computeThicknessReadings(top, bot, 5, 0.37, 0.05, IW, IH);
    const node = pointOnCurve(top, 0.37, IW, IH)!;
    expect(readings[2].top[0]).toBeCloseTo(node[0], 9);
    expect(readings[2].top[1]).toBeCloseTo(node[1], 9);
  });

  it("an edited reading is kept verbatim after a point is inserted", () => {
    const first = computeThicknessReadings(top, bot, 3, 0.5, 0.1, IW, IH).readings;
    const edited: ThicknessReadingData = { ...first[0], top: [11, 11], bottom: [12, 88], edited: true, text: "x" };
    const existing = [edited, first[1], first[2]];
    const { points } = insertSurfacePoint(top, [60, 29.5], IW, IH);
    const { readings } = computeThicknessReadings(points, bot, 3, 0.5, 0.1, IW, IH, existing);
    expect(readings[0]).toEqual(edited);
    expect(readings[1].edited).toBe(false);
  });

  it("per-reading font size / colour survive regeneration and clear back to null", () => {
    const first = computeThicknessReadings(top, bot, 3, 0.5, 0.1, IW, IH).readings;
    expect(first.every((r) => r.font_size === null && r.color === null)).toBe(true);
    const styled = first.map((r, i) => (i === 1 ? { ...r, font_size: 9, color: "#FF0000" } : r));
    // Reshape the surface (drag a point): readings move, overrides stay.
    const moved = top.map((p, i) => (i === 2 ? [p[0], p[1] - 3] as Pt : p));
    const { readings } = computeThicknessReadings(moved, bot, 3, 0.5, 0.1, IW, IH, styled);
    expect(readings[1].font_size).toBe(9);
    expect(readings[1].color).toBe("#FF0000");
    expect(readings[1].top).not.toEqual(first[1].top);
    expect(readings[0].font_size).toBeNull();
    // Clearing to the measurement default (null) sticks too.
    const cleared = readings.map((r) => ({ ...r, font_size: null, color: null }));
    const again = computeThicknessReadings(moved, bot, 3, 0.5, 0.1, IW, IH, cleared).readings;
    expect(again[1].font_size).toBeNull();
    expect(again[1].color).toBeNull();
  });

  it("one complete surface + one empty: that surface's samples, no new readings", () => {
    const res = computeThicknessReadings(top, [], 5, 0.5, 0.1, IW, IH);
    expect(res.readings).toEqual([]);
    expect(res.topSamples.length).toBe(32 * 4 + 1);
    expect(res.bottomSamples).toEqual([]);
    // A single bottom point is still not a surface.
    const one = computeThicknessReadings(top, [[50, 60]], 5, 0.5, 0.1, IW, IH);
    expect(one.readings).toEqual([]);
    expect(one.bottomSamples).toEqual([]);
    // Existing readings are carried (not recomputed) while a surface is retraced.
    const prior = computeThicknessReadings(top, bot, 3, 0.5, 0.1, IW, IH).readings;
    expect(computeThicknessReadings([], bot, 3, 0.5, 0.1, IW, IH, prior).readings).toEqual(prior);
  });
});
