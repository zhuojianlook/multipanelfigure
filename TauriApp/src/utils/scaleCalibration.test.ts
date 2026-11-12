import { describe, it, expect } from "vitest";
import {
  UNIT_TO_UM, SCALE_UNITS, MIN_MEASURE_PX,
  umPerUnit, clampPct, pixelDistance, umPerPixel, perPixelInUnit, formatPerPixel,
  parseDistance, calibrationIssue,
  scaleEntryKey, parseScaleEntryKey, scaleEntryNameIssue, withScaleEntry, scaleFromEntry,
  scaleBarAutoLabel, barAxisPixelDistance, parseAspectRatio, preResizeDims,
} from "./scaleCalibration";
import type { PctPt } from "./scaleCalibration";
import { lineLengthInUnit, lineLengthPx } from "./lineMeasure";
import { readingValue } from "./thicknessGeometry";

/** Relative closeness — the values span 1e-15 (km/px) to 1e9 (pm/px), so an
 *  absolute-decimals check like toBeCloseTo is meaningless across units. */
function expectRel(actual: number, expected: number, tol = 1e-12) {
  expect(Number.isFinite(actual)).toBe(true);
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(Math.abs(expected) * tol);
}

describe("unit table parity", () => {
  /* lineMeasure / thicknessGeometry keep their tables private, so pin parity
     through their public conversions: a 1000-px horizontal span at 1 µm/px is
     exactly 1000 µm, i.e. 1000 / UNIT_TO_UM[u] in unit u. Same arithmetic,
     same table → bit-identical. */
  it("matches lineMeasure's UNIT_TO_UM for every unit", () => {
    for (const u of SCALE_UNITS) {
      expect(lineLengthInUnit([[0, 0], [100, 0]], 1000, 1, "straight", 1, u)).toBe(1000 / UNIT_TO_UM[u]);
    }
  });

  it("matches thicknessGeometry's UNIT_TO_UM for every unit", () => {
    for (const u of SCALE_UNITS) {
      const r = { top: [0, 0], bottom: [100, 0] } as unknown as Parameters<typeof readingValue>[0];
      expect(readingValue(r, 1000, 1, 1, u)).toBe(1000 / UNIT_TO_UM[u]);
    }
  });

  it("matches the Sidebar's conversionToUm literal", () => {
    // Copied from Sidebar.tsx (scale-bar add form). If this ever changes
    // there, entries saved by the two paths would disagree.
    const sidebar: Record<string, number> = { km: 1e9, m: 1e6, cm: 10000, mm: 1000, um: 1, nm: 0.001, pm: 1e-6 };
    expect({ ...UNIT_TO_UM }).toEqual(sidebar);
    expect([...SCALE_UNITS].sort()).toEqual(Object.keys(sidebar).sort());
  });

  it("umPerUnit falls back to µm for unknown / prototype keys", () => {
    expect(umPerUnit("mm")).toBe(1000);
    expect(umPerUnit("furlong")).toBe(1);
    expect(umPerUnit("constructor")).toBe(1);
    expect(umPerUnit("")).toBe(1);
  });
});

describe("pixelDistance", () => {
  it("measures percent points in image pixels", () => {
    expect(pixelDistance([10, 50], [60, 50], 1000, 800)).toBe(500);
    expect(pixelDistance([0, 0], [0, 100], 1000, 800)).toBe(800);
    // 3-4-5 triangle in pixels
    expect(pixelDistance([0, 0], [30, 40], 100, 100)).toBe(50);
  });

  it("scales x and y independently (non-square / final_resize images)", () => {
    // 10% of 2000 = 200 px across, 10% of 1500 = 150 px down → 250 px.
    expect(pixelDistance([20, 20], [30, 30], 2000, 1500)).toBe(250);
  });

  it("is symmetric and zero for coincident points", () => {
    const a: PctPt = [12.5, 77.25], b: PctPt = [63.1, 4.9];
    expect(pixelDistance(a, b, 1234, 987)).toBe(pixelDistance(b, a, 1234, 987));
    expect(pixelDistance(a, a, 1234, 987)).toBe(0);
  });

  it("agrees with a straight line annotation's length (lineMeasure)", () => {
    const a: PctPt = [12.5, 77.25], b: PctPt = [63.1, 4.9];
    expect(pixelDistance(a, b, 1234, 987)).toBeCloseTo(lineLengthPx([a, b], 1234, 987, "straight"), 9);
  });

  it("clampPct keeps a point inside the image", () => {
    expect(clampPct([-5, 105])).toEqual([0, 100]);
    expect(clampPct([42.5, 0.1])).toEqual([42.5, 0.1]);
  });
});

describe("umPerPixel / perPixelInUnit", () => {
  it("converts a known distance into µm/px in every unit", () => {
    expect(umPerPixel(100, "um", 1000)).toBe(0.1);
    expect(umPerPixel(1, "mm", 500)).toBe(2);           // 1000 µm over 500 px
    expect(umPerPixel(250, "nm", 1000)).toBe(0.00025);  // 0.25 µm over 1000 px
    expect(umPerPixel(3, "cm", 600)).toBe(50);
  });

  it("returns NaN rather than 0 / Infinity for unusable input", () => {
    expect(umPerPixel(0, "um", 100)).toBeNaN();
    expect(umPerPixel(-1, "um", 100)).toBeNaN();
    expect(umPerPixel(NaN, "um", 100)).toBeNaN();
    expect(umPerPixel(Infinity, "um", 100)).toBeNaN();
    expect(umPerPixel(10, "um", 0)).toBeNaN();
    expect(umPerPixel(10, "um", NaN)).toBeNaN();
  });

  it("perPixelInUnit is the display inverse of the stored µm/px", () => {
    expect(perPixelInUnit(2, "um")).toBe(2);
    expect(perPixelInUnit(2, "mm")).toBe(0.002);
    expect(perPixelInUnit(0.0813, "nm")).toBeCloseTo(81.3, 10);
  });

  it("formatPerPixel shows the calibration in the chosen unit", () => {
    expect(formatPerPixel(0.0813, "um")).toBe("0.0813 µm/px");
    expect(formatPerPixel(0.0813, "nm")).toBe("81.3 nm/px");
    expect(formatPerPixel(2, "mm")).toBe("0.002 mm/px");
    expect(formatPerPixel(1 / 3, "um")).toBe("0.3333 µm/px");
    expect(formatPerPixel(NaN, "um")).toBe("—");
    expect(formatPerPixel(0, "um")).toBe("—");
  });
});

describe("input validation", () => {
  it("parseDistance: empty is NaN, decimal comma accepted", () => {
    expect(parseDistance("")).toBeNaN();
    expect(parseDistance("   ")).toBeNaN();
    expect(parseDistance("12.5")).toBe(12.5);
    expect(parseDistance(" 0,5 ")).toBe(0.5);
    expect(parseDistance("1e3")).toBe(1000);
    expect(parseDistance("abc")).toBeNaN();
  });

  it("calibrationIssue gates Apply / Save", () => {
    expect(calibrationIssue(0, 10)).toBe("too-short");
    expect(calibrationIssue(MIN_MEASURE_PX - 0.01, 10)).toBe("too-short");
    expect(calibrationIssue(NaN, 10)).toBe("too-short");
    expect(calibrationIssue(MIN_MEASURE_PX, 10)).toBeNull();
    expect(calibrationIssue(100, 0)).toBe("bad-distance");
    expect(calibrationIssue(100, -3)).toBe("bad-distance");
    expect(calibrationIssue(100, NaN)).toBe("bad-distance");
    expect(calibrationIssue(100, Infinity)).toBe("bad-distance");
    // Too-short wins: re-measuring is the first thing to fix.
    expect(calibrationIssue(1, NaN)).toBe("too-short");
  });

  it("scaleEntryNameIssue rejects empty names and the key separator", () => {
    expect(scaleEntryNameIssue("Scope 40x")).toBeNull();
    expect(scaleEntryNameIssue("  ")).not.toBeNull();
    expect(scaleEntryNameIssue("")).not.toBeNull();
    expect(scaleEntryNameIssue("a|b")).not.toBeNull();
  });
});

describe("scale-bar entries (config.resolution_entries)", () => {
  it("key format is exactly the Sidebar's `${name}|${unit}`", () => {
    const name = "Zeiss 63x oil", unit = "nm";
    expect(scaleEntryKey(name, unit)).toBe(`${name}|${unit}`);
    expect(parseScaleEntryKey(scaleEntryKey(name, unit))).toEqual({ name, unit });
  });

  it("parses legacy unit-less keys as µm, like every picker does", () => {
    expect(parseScaleEntryKey("Old preset")).toEqual({ name: "Old preset", unit: "um" });
  });

  it("withScaleEntry adds / overwrites without mutating the original", () => {
    const orig = { "A|um": 0.5 };
    const added = withScaleEntry(orig, "B|nm", 0.002);
    expect(added).toEqual({ "A|um": 0.5, "B|nm": 0.002 });
    expect(orig).toEqual({ "A|um": 0.5 });
    const over = withScaleEntry(added, "A|um", 0.25);
    expect(over["A|um"]).toBe(0.25);
    expect(Object.keys(over)).toHaveLength(2);
    expect(withScaleEntry(undefined, "C|um", 1)).toEqual({ "C|um": 1 });
  });

  it("scaleFromEntry mirrors the Source picker (missing / zero → Custom)", () => {
    const entries = { "A|mm": 2, "Legacy": 0.3, "Zero|um": 0 };
    expect(scaleFromEntry(entries, "A|mm")).toEqual({ micron_per_pixel: 2, unit: "mm" });
    expect(scaleFromEntry(entries, "Legacy")).toEqual({ micron_per_pixel: 0.3, unit: "um" });
    expect(scaleFromEntry(entries, "Zero|um")).toBeNull();
    expect(scaleFromEntry(entries, "Nope|um")).toBeNull();
    expect(scaleFromEntry(entries, "")).toBeNull();
    expect(scaleFromEntry(undefined, "A|mm")).toBeNull();
    expect(scaleFromEntry(entries, "constructor")).toBeNull();
  });
});

describe("round trip: measure → save entry → select entry → display", () => {
  /* The guarantee the feature makes: whatever µm/px the measurement produced
     is what a panel gets back — EXACTLY — when that entry is chosen later,
     including after the entries go through JSON (api.updateResolutions →
     scale_bars.json, and the .mpfs project file). */
  const a: PctPt = [10.37, 20.81];
  const b: PctPt = [73.29, 61.66];
  const W = 2048, H = 1536;
  const DIST = 37.5;

  for (const unit of SCALE_UNITS) {
    it(`unit ${unit}`, () => {
      const px = pixelDistance(a, b, W, H);
      expect(calibrationIssue(px, DIST)).toBeNull();
      const measured = umPerPixel(DIST, unit, px);

      // Save (Sidebar path): key + value in µm/px.
      const key = scaleEntryKey("Scope 40x", unit);
      const entries = withScaleEntry({ "Other|um": 0.5 }, key, measured);
      // Persisted and reloaded.
      const reloaded = JSON.parse(JSON.stringify(entries)) as Record<string, number>;

      // Select it in the Source picker.
      const sel = scaleFromEntry(reloaded, key);
      expect(sel).not.toBeNull();
      expect(sel!.micron_per_pixel).toBe(measured);  // bit-for-bit
      expect(sel!.unit).toBe(unit);

      // The "<unit>/pixel" field then shows distance / px in that unit…
      expectRel(perPixelInUnit(sel!.micron_per_pixel, sel!.unit), DIST / px);
      // …and the same physical scale in every other unit.
      for (const other of SCALE_UNITS) {
        expectRel(perPixelInUnit(sel!.micron_per_pixel, other) * UNIT_TO_UM[other], measured);
      }
      // A bar of the measured length spans the measured pixels again.
      expectRel((DIST * UNIT_TO_UM[unit]) / sel!.micron_per_pixel, px);
    });
  }
});

describe("scaleBarAutoLabel", () => {
  it("matches the Scale Bar tab's auto label format", () => {
    expect(scaleBarAutoLabel(50, "um")).toBe("50 µm");
    expect(scaleBarAutoLabel(1500, "mm")).toBe("1.5 mm");
    expect(scaleBarAutoLabel(0.25, "nm")).toBe("250 nm");
    expect(scaleBarAutoLabel(1 / 3, "um")).toBe("0.333333 µm");
  });
});

describe("barAxisPixelDistance (non-uniform final_resize)", () => {
  // The verified repro: rotate 90 turns 1600x1000 into 1000x1600, then
  // final_resize 500x1600 halves x only. Two marks 500 SOURCE px apart
  // vertically. The bar is drawn horizontally, where 1 output px = 2 source
  // px, so a "500 um" bar must be 250 output px — mpp 2, not 1.
  const pre = { w: 1000, h: 1600 };
  it("expresses a vertical chord in horizontal output px", () => {
    const px = barAxisPixelDistance([50, 10], [50, 41.25], 500, 1600, pre);
    expect(px).toBeCloseTo(250, 9);
    expect(umPerPixel(500, "um", px)).toBeCloseTo(2, 9);   // was 1 (2x too long a bar)
  });
  it("leaves a horizontal chord unchanged", () => {
    expect(barAxisPixelDistance([10, 50], [60, 50], 500, 1600, pre))
      .toBeCloseTo(pixelDistance([10, 50], [60, 50], 500, 1600), 9);
  });
  it("equals pixelDistance for a uniform resize, and without pre", () => {
    const a: PctPt = [12, 30], b: PctPt = [77, 81];
    expect(barAxisPixelDistance(a, b, 1000, 800, { w: 2000, h: 1600 }))
      .toBeCloseTo(pixelDistance(a, b, 1000, 800), 9);
    expect(barAxisPixelDistance(a, b, 1000, 800, null)).toBe(pixelDistance(a, b, 1000, 800));
    expect(barAxisPixelDistance(a, b, 1000, 800)).toBe(pixelDistance(a, b, 1000, 800));
    expect(barAxisPixelDistance(a, b, 1000, 800, { w: 0, h: 5 })).toBe(pixelDistance(a, b, 1000, 800));
  });
});

describe("parseAspectRatio (mirrors image_processing.parse_aspect_ratio)", () => {
  it("parses W:H", () => {
    expect(parseAspectRatio("4:3")).toBeCloseTo(4 / 3, 12);
    expect(parseAspectRatio(" 16 : 9 ")).toBeCloseTo(16 / 9, 12);
    expect(parseAspectRatio("1:1")).toBe(1);
  });
  it("is null where python returns None", () => {
    for (const s of ["", "  ", "4", "4:3:2", "a:3", "4:", ":3", "4:0", null, undefined]) {
      expect(parseAspectRatio(s as string)).toBeNull();
    }
  });
});

describe("preResizeDims (mirrors process_panel's rotate → crop order)", () => {
  const base = { origW: 1600, origH: 1000, rotationDeg: 0, crop: null, cropImage: false, aspect: null };
  it("rotation 90/270 swaps, 0/180 doesn't", () => {
    expect(preResizeDims({ ...base, rotationDeg: 90 })).toEqual({ w: 1000, h: 1600 });
    expect(preResizeDims({ ...base, rotationDeg: 270 })).toEqual({ w: 1000, h: 1600 });
    expect(preResizeDims({ ...base, rotationDeg: 180 })).toEqual({ w: 1600, h: 1000 });
  });
  it("a direct crop wins, in rotated space, clamped", () => {
    expect(preResizeDims({ ...base, crop: [100, 50, 700, 450] })).toEqual({ w: 600, h: 400 });
    expect(preResizeDims({ ...base, rotationDeg: 90, crop: [0, 0, 5000, 5000] })).toEqual({ w: 1000, h: 1600 });
  });
  it("an aspect crop truncates like crop_with_aspect", () => {
    // 1600x1000 at 1:1 → rw 1600, rh 1600 > 1000 → rh 1000, rw 1000.
    expect(preResizeDims({ ...base, cropImage: true, aspect: 1 })).toEqual({ w: 1000, h: 1000 });
    // 4:3 → rh = trunc(1600 / (4/3)) = 1200 > 1000 → rh 1000, rw = trunc(1333.33) = 1333.
    expect(preResizeDims({ ...base, cropImage: true, aspect: 4 / 3 })).toEqual({ w: 1333, h: 1000 });
  });
  it("is null until the source size is known", () => {
    expect(preResizeDims({ ...base, origW: 0 })).toBeNull();
  });
});
