import { describe, it, expect } from "vitest";
import { computeThicknessReadings, readingValue, type Pt, type ThicknessReadingData } from "./thicknessGeometry";

// Helper: 3 points on a circle (center cx,cy pixel), radius r, at given degrees,
// returned in % for a square iw×ih image.
function arcPts(cx: number, cy: number, r: number, degs: number[], iw: number, ih: number): Pt[] {
  return degs.map((d) => {
    const a = (d * Math.PI) / 180;
    return [((cx + r * Math.cos(a)) / iw) * 100, ((cy + r * Math.sin(a)) / ih) * 100] as Pt;
  });
}

describe("thicknessGeometry", () => {
  const IW = 1000, IH = 1000;

  it("concentric arcs → radial thickness = radius difference", () => {
    // Shared center (500,300). Top r=200, bottom r=260. Angles 60/90/120.
    const top = arcPts(500, 300, 200, [60, 90, 120], IW, IH);
    const bot = arcPts(500, 300, 260, [60, 90, 120], IW, IH);
    const { readings } = computeThicknessReadings(top, bot, 1, 0.5, 0.1, IW, IH);
    expect(readings.length).toBe(1);
    // Center reading at t=0.5 → 90° → top point (500,500)px = (50,50)%.
    const r0 = readings[0];
    expect(r0.top[0]).toBeCloseTo(50, 1);
    expect(r0.top[1]).toBeCloseTo(50, 1);
    // Thickness should be ~60px = 60µm at mpp=1.
    const val = readingValue(r0, IW, IH, 1.0, "um");
    expect(val).toBeCloseTo(60, 0);
    // Bottom point on the far (outward) side: (500,560)px = (50,56)%.
    expect(r0.bottom[0]).toBeCloseTo(50, 1);
    expect(r0.bottom[1]).toBeCloseTo(56, 1);
  });

  it("straight parallel surfaces → vertical perpendicular thickness", () => {
    // Collinear top at y=30%, bottom at y=50%, spanning x 20..80.
    const top: Pt[] = [[20, 30], [50, 30], [80, 30]];
    const bot: Pt[] = [[20, 50], [50, 50], [80, 50]];
    const { readings } = computeThicknessReadings(top, bot, 3, 0.5, 0.25, IW, IH);
    expect(readings.length).toBe(3);
    for (const r of readings) {
      // Perpendicular to a horizontal line is vertical → same x, Δy=20% = 200px.
      expect(r.bottom[0]).toBeCloseTo(r.top[0], 1);
      expect(readingValue(r, IW, IH, 1.0, "um")).toBeCloseTo(200, 0);
    }
    // 3 readings centred at 0.5 with spacing 0.25 of the (60%-wide) arc → the
    // middle one sits at x=50%.
    const xs = readings.map((r) => r.top[0]).sort((a, b) => a - b);
    expect(xs[1]).toBeCloseTo(50, 0);
    // symmetric about the centre
    expect(50 - xs[0]).toBeCloseTo(xs[2] - 50, 1);
  });

  it("preserves edited readings and carries hidden/text across regen", () => {
    const top: Pt[] = [[20, 30], [50, 30], [80, 30]];
    const bot: Pt[] = [[20, 50], [50, 50], [80, 50]];
    const existing: ThicknessReadingData[] = [
      { top: [10, 30], bottom: [10, 90], hidden: false, text: "custom", edited: true },
      { top: [50, 30], bottom: [50, 50], hidden: true, text: "", edited: false },
      { top: [80, 30], bottom: [80, 50], hidden: false, text: "hi", edited: false },
    ];
    const { readings } = computeThicknessReadings(top, bot, 3, 0.5, 0.25, IW, IH, existing);
    // Reading 0 is frozen verbatim.
    expect(readings[0]).toEqual(existing[0]);
    // Reading 1 recomputed but keeps hidden.
    expect(readings[1].hidden).toBe(true);
    expect(readings[1].edited).toBe(false);
    // Reading 2 recomputed but keeps its text.
    expect(readings[2].text).toBe("hi");
  });

  it("returns empty when a curve lacks 3 points", () => {
    const { readings, topSamples } = computeThicknessReadings([[10, 10], [20, 20]], [], 5, 0.5, 0.1, IW, IH);
    expect(readings).toEqual([]);
    expect(topSamples).toEqual([]);
  });

  it("non-square image: perpendicular still correct in pixel space", () => {
    // Wide image: a % circle would be an ellipse; geometry must use pixels.
    const iw = 2000, ih = 1000;
    const top = arcPts(1000, 300, 200, [60, 90, 120], iw, ih);
    const bot = arcPts(1000, 300, 260, [60, 90, 120], iw, ih);
    const { readings } = computeThicknessReadings(top, bot, 1, 0.5, 0.1, iw, ih);
    expect(readingValue(readings[0], iw, ih, 1.0, "um")).toBeCloseTo(60, 0);
  });
});
