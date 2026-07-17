import { describe, it, expect } from "vitest";
import {
  computeThicknessReadings, readingValue, normalizeCount, maxSpacingFor,
  snapToCurve, paramOnCurve, pointOnCurve,
  type Pt, type ThicknessReadingData,
} from "./thicknessGeometry";

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

  it("caps spacing so the group always fits inside the arc", () => {
    expect(maxSpacingFor(1)).toBe(1);
    expect(maxSpacingFor(3)).toBeCloseTo(0.5, 6);   // 2 gaps
    expect(maxSpacingFor(5)).toBeCloseTo(0.25, 6);  // 4 gaps
    expect(maxSpacingFor(4)).toBeCloseTo(0.25, 6);  // rounded up to 5
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
